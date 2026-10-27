//! Sparse modular elimination and flow-witness materialization.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bloq_utils::{Direction, Pauli, PauliString, PhasedPauliString};
use glam::IVec3;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use super::graph::{NodeKind, ZXGraph};
use super::stabilizer::axis_pivot_constraints;

type EdgePair = (usize, usize);

/// Raw signed rows and their pivots share one cache lifetime.
#[derive(Debug)]
pub(crate) struct StabilizerPhaseBasis {
    rows: Vec<PhasedPauliString>,
    pivots: Vec<(usize, Pauli)>,
}

impl StabilizerPhaseBasis {
    pub(super) fn new(rows: Vec<PhasedPauliString>) -> Self {
        Self {
            pivots: phase_pivots(&rows),
            rows,
        }
    }

    pub(crate) fn rows(&self) -> &[PhasedPauliString] {
        &self.rows
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProjectionLimits {
    pub max_frontier_width: usize,
    pub max_witness_nodes: usize,
}

#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum ProjectionError {
    #[error("projected frontier width {observed} exceeds limit {limit}")]
    FrontierLimit { observed: usize, limit: usize },
    #[error("witness nodes {observed} exceed limit {limit}")]
    WitnessLimit { observed: usize, limit: usize },
    #[error(transparent)]
    Cancelled(#[from] crate::ComputationCancelled),
}

#[derive(Debug, Clone)]
pub(crate) struct ProjectedRow {
    pub signed: PhasedPauliString,
    witness: usize,
    pub(crate) flow_witness: FlowWitness,
}

#[derive(Debug, Clone)]
pub(crate) struct ProjectedExternalTable {
    pub boundary_rows: Vec<ProjectedRow>,
    pub closed_rows: Vec<WitnessedRow>,
    arena: WitnessArena,
    width: usize,
}

/// Unsigned compositor-owned rows with exact original raw ZX component queries.
pub(crate) struct CompactSurfaceRelation {
    pub(super) rows: Vec<super::CoeffVec>,
    pub(super) queries: Vec<[Option<usize>; 2]>,
    pub(super) width: usize,
}

/// Directed edge coordinates followed by the independent isolated-site ids.
#[derive(Debug)]
struct EdgeFlowLayout<'a> {
    zx: &'a ZXGraph,
    isolated: Vec<usize>,
    boundary_arms: Vec<(usize, bool)>,
}

impl<'a> EdgeFlowLayout<'a> {
    fn new(zx: &'a ZXGraph) -> Self {
        Self {
            zx,
            isolated: zx
                .nodes
                .iter()
                .filter(|node| zx.neighbor_ids(node.id).is_empty())
                .map(|node| node.id)
                .collect(),
            boundary_arms: zx
                .nodes
                .iter()
                .filter(|node| node.kind.is_boundary())
                .filter_map(|node| {
                    let (_, edge_id) = zx.neighbor_edges(node.id).next()?;
                    let edge = zx.edge_by_id(edge_id);
                    Some((edge_id - zx.nodes.len(), edge.hadamard && node.id > edge.n2))
                })
                .collect(),
        }
    }

    fn width(&self) -> usize {
        self.zx.edges.len() + self.isolated.len()
    }

    fn column(&self, graph_column: usize) -> Option<usize> {
        if graph_column >= self.zx.nodes.len() {
            Some(graph_column - self.zx.nodes.len())
        } else {
            self.isolated
                .binary_search(&graph_column)
                .ok()
                .map(|index| self.zx.edges.len() + index)
        }
    }

    fn project_terms(&self, terms: impl Iterator<Item = (usize, Pauli)>) -> PauliString {
        PauliString::from_terms(
            self.width(),
            terms.filter_map(|(column, pauli)| self.column(column).map(|column| (column, pauli))),
        )
    }

    fn lift(&self, row: &PauliString) -> PauliString {
        let mut lifted = PauliString::from_terms(
            self.zx.total_ids,
            row.iter_support().map(|(column, pauli)| {
                (
                    if column < self.zx.edges.len() {
                        column + self.zx.nodes.len()
                    } else {
                        self.isolated[column - self.zx.edges.len()]
                    },
                    pauli,
                )
            }),
        );
        self.zx.reconstruct_raw_centers(&mut lifted);
        lifted
    }

    fn center_product_phase(&self, left: &PauliString, right: &PauliString) -> u8 {
        let mut phase = 0u8;
        // Spider and Y centers carry only one fixed axis, so their products
        // contribute no phase. Boundary centers carry their arm's full Pauli.
        for &(column, flip) in &self.boundary_arms {
            let mut lhs = left.get(column);
            let mut rhs = right.get(column);
            if flip {
                lhs = lhs.flip();
                rhs = rhs.flip();
            }
            phase = (phase + pauli_product_phase(lhs, rhs)) & 3;
        }
        phase
    }
}

/// Seam elimination stores only edge support and accounts for the omitted
/// boundary-center factors when signed rows multiply.
#[derive(Debug, Clone)]
struct EdgeFlowRow<'a> {
    signed: PhasedPauliString,
    layout: &'a EdgeFlowLayout<'a>,
}

impl EdgeFlowRow<'_> {
    fn materialize(self) -> PhasedPauliString {
        PhasedPauliString::new(self.layout.lift(&self.signed.paulis), self.signed.phase())
    }
}

impl EliminationRow for EdgeFlowRow<'_> {
    fn multiply_assign(&mut self, pivot: &Self) {
        let center_phase = self
            .layout
            .center_product_phase(&self.signed.paulis, &pivot.signed.paulis);
        self.signed.multiply_assign(&pivot.signed);
        self.signed.shift_phase(center_phase);
    }
}

impl ProjectedExternalTable {
    pub(crate) fn materialize(&self, row: &ProjectedRow) -> PhasedPauliString {
        self.arena.materialize(row.witness, self.width)
    }
}

#[derive(Debug, Clone)]
struct SparseRow {
    paulis: BTreeMap<usize, Pauli>,
    witness: usize,
}

#[derive(Debug, Clone)]
struct SourceRow {
    anchor: usize,
    terms: Vec<(usize, Pauli)>,
    phase: u8,
}

impl SourceRow {
    fn multiply_into(&self, row: &mut PhasedPauliString) {
        assert!(
            self.terms
                .iter()
                .all(|&(column, _)| column < row.paulis.len())
        );
        // Parallel-edge aliases retain from_terms' last-write-wins semantics.
        let mut terms = self
            .terms
            .iter()
            .rev()
            .copied()
            .collect::<SmallVec<[_; 8]>>();
        terms.sort_by_key(|&(column, _)| column);
        terms.dedup_by_key(|term| term.0);
        row.shift_phase(self.phase);
        for (column, right) in terms {
            let left = row.paulis.get(column);
            row.shift_phase(pauli_product_phase(left, right));
            row.paulis.set(column, left ^ right);
        }
    }
}

fn pauli_product_phase(left: Pauli, right: Pauli) -> u8 {
    match (left, right) {
        (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::X) => 1,
        (Pauli::Y, Pauli::X) | (Pauli::Z, Pauli::Y) | (Pauli::X, Pauli::Z) => 3,
        _ => 0,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FlowSourceKey {
    position: IVec3,
    node: Pauli,
    edges: Vec<(Direction, Pauli)>,
    intrinsic_phase: u8,
}

// Scoped to one graph/arena operation; only source indices are retained.
type FlowSourceKeys = crate::FxHashMap<usize, Arc<FlowSourceKey>>;

#[derive(Debug, Clone, Default)]
pub(crate) struct FlowWitness {
    /// Sorted, shared source keys; elimination copies witnesses often.
    sources: Vec<Arc<FlowSourceKey>>,
}

#[derive(Debug, Clone)]
pub(crate) struct WitnessedRow {
    pub signed: PhasedPauliString,
    pub(crate) flow_witness: FlowWitness,
}

#[derive(Debug, Clone)]
enum WitnessNode {
    Source(SourceRow),
    Xor(usize, usize),
}

#[derive(Debug, Clone, Default)]
struct WitnessArena {
    nodes: Vec<WitnessNode>,
    source_rows: usize,
}

impl SparseRow {
    fn get(&self, column: usize) -> Pauli {
        self.paulis.get(&column).copied().unwrap_or(Pauli::I)
    }

    fn xor_assign(
        &mut self,
        other: &Self,
        arena: &mut WitnessArena,
        limit: usize,
    ) -> Result<(), ProjectionError> {
        self.witness = arena.push_xor(self.witness, other.witness, limit)?;
        for (&column, &pauli) in &other.paulis {
            let product = self.get(column) ^ pauli;
            if product == Pauli::I {
                self.paulis.remove(&column);
            } else {
                self.paulis.insert(column, product);
            }
        }
        Ok(())
    }
}

impl FlowWitness {
    pub(crate) fn xor_assign(&mut self, other: &Self) {
        if self
            .sources
            .last()
            .zip(other.sources.first())
            .is_none_or(|(left, right)| flow_source_key_cmp(left, right) == Ordering::Less)
        {
            self.sources.extend(other.sources.iter().cloned());
            return;
        }
        let mut left = std::mem::take(&mut self.sources).into_iter().peekable();
        let mut right = other.sources.iter().peekable();
        let mut sources = Vec::with_capacity(left.len() + right.len());
        while let (Some(lhs), Some(rhs)) = (left.peek(), right.peek()) {
            match flow_source_key_cmp(lhs, rhs) {
                Ordering::Less => sources.push(left.next().expect("peeked source")),
                Ordering::Equal => {
                    left.next();
                    right.next();
                }
                Ordering::Greater => sources.push(Arc::clone(right.next().expect("peeked source"))),
            }
        }
        sources.extend(left);
        sources.extend(right.map(Arc::clone));
        self.sources = sources;
    }

    pub(crate) fn transformed(
        &self,
        orientation: crate::ModuleOrientation,
        transform: impl Fn(IVec3) -> Result<IVec3, crate::BlockGraphError>,
    ) -> Result<Self, crate::BlockGraphError> {
        let mut sources = self
            .sources
            .iter()
            .map(|source| {
                let mut source = source.as_ref().clone();
                source.position = transform(source.position)?;
                for (direction, _) in &mut source.edges {
                    *direction = orientation.rotate_direction(*direction);
                }
                source
                    .edges
                    .sort_unstable_by_key(|(direction, _)| direction.to_ivec3().to_array());
                Ok(Arc::new(source))
            })
            .collect::<Result<Vec<_>, crate::BlockGraphError>>()?;
        sources.sort_unstable_by(|left, right| flow_source_key_cmp(left, right));
        Ok(Self { sources })
    }
}

fn flow_source_key_cmp(left: &FlowSourceKey, right: &FlowSourceKey) -> Ordering {
    left.position
        .to_array()
        .cmp(&right.position.to_array())
        .then_with(|| u8::from(left.node).cmp(&u8::from(right.node)))
        .then_with(|| {
            left.edges
                .iter()
                .map(|(direction, pauli)| (direction.index(), u8::from(*pauli)))
                .cmp(
                    right
                        .edges
                        .iter()
                        .map(|(direction, pauli)| (direction.index(), u8::from(*pauli))),
                )
        })
        .then_with(|| left.intrinsic_phase.cmp(&right.intrinsic_phase))
}

impl WitnessArena {
    fn push_source(&mut self, source: SourceRow, limit: usize) -> Result<usize, ProjectionError> {
        self.check_limit(limit)?;
        self.nodes.push(WitnessNode::Source(source));
        self.source_rows += 1;
        Ok(self.nodes.len() - 1)
    }

    fn push_xor(
        &mut self,
        left: usize,
        right: usize,
        limit: usize,
    ) -> Result<usize, ProjectionError> {
        self.check_limit(limit)?;
        self.nodes.push(WitnessNode::Xor(left, right));
        Ok(self.nodes.len() - 1)
    }

    fn check_limit(&self, limit: usize) -> Result<(), ProjectionError> {
        if self.nodes.len() < limit {
            Ok(())
        } else {
            Err(ProjectionError::WitnessLimit {
                observed: self.nodes.len() + 1,
                limit,
            })
        }
    }

    fn source_indices(&self, witness: usize) -> BTreeSet<usize> {
        let mut pending = BTreeSet::from([witness]);
        let mut sources = BTreeSet::new();
        while let Some(node) = pending.pop_last() {
            match &self.nodes[node] {
                WitnessNode::Source(_) => toggle(&mut sources, node),
                WitnessNode::Xor(left, right) => {
                    toggle(&mut pending, *left);
                    toggle(&mut pending, *right);
                }
            }
        }
        sources
    }

    fn materialize(&self, witness: usize, width: usize) -> PhasedPauliString {
        let mut row = PhasedPauliString::positive(PauliString::new(width));
        for source in self.source_indices(witness) {
            let WitnessNode::Source(source) = &self.nodes[source] else {
                unreachable!("source set contains only source nodes")
            };
            source.multiply_into(&mut row);
        }
        row
    }
}

pub(super) fn toggle<T: Ord>(set: &mut BTreeSet<T>, value: T) {
    if !set.remove(&value) {
        set.insert(value);
    }
}

impl ZXGraph {
    pub(crate) fn projected_external_table(
        &self,
        boundary_columns: &[usize],
        limits: ProjectionLimits,
    ) -> Result<ProjectedExternalTable, ProjectionError> {
        if boundary_columns.len() > limits.max_frontier_width {
            return Err(ProjectionError::FrontierLimit {
                observed: boundary_columns.len(),
                limit: limits.max_frontier_width,
            });
        }
        let retained = boundary_columns.iter().copied().collect::<FxHashSet<_>>();
        let mut arena = WitnessArena::default();
        let mut source_keys = FlowSourceKeys::default();
        let mut rows = Vec::new();
        let mut closed_candidates = Vec::new();
        for (node_ids, same_layer_pairs, pending_edge_pairs) in self.stabilizer_layers() {
            let mut local = self.projected_source_rows(
                &node_ids,
                &retained,
                &mut arena,
                limits.max_witness_nodes,
            )?;
            eliminate_projected_edge_pairs(
                &mut local,
                &same_layer_pairs,
                &mut arena,
                limits.max_witness_nodes,
            )?;
            remove_columns(
                &mut local,
                node_ids
                    .iter()
                    .copied()
                    .filter(|column| !retained.contains(column))
                    .chain(
                        same_layer_pairs
                            .iter()
                            .flat_map(|&(left, right)| [left, right]),
                    ),
            );
            rows.extend(local);
            check_frontier(&rows, limits.max_frontier_width)?;
            eliminate_projected_edge_pairs(
                &mut rows,
                &pending_edge_pairs,
                &mut arena,
                limits.max_witness_nodes,
            )?;
            remove_columns(
                &mut rows,
                pending_edge_pairs
                    .iter()
                    .flat_map(|&(left, right)| [left, right]),
            );
            retire_closed(
                self,
                &mut rows,
                &mut closed_candidates,
                &arena,
                &mut source_keys,
            );
        }

        self.finish_projected_table(
            rows,
            closed_candidates,
            arena,
            source_keys,
            boundary_columns,
            limits,
        )
    }

    /// Present an already composed complete relation without solving its seams again.
    /// `None` rejects support or phases that do not belong to the local tensors.
    pub(crate) fn projected_external_table_from_rows(
        &self,
        boundary_columns: &[usize],
        complete: Vec<PhasedPauliString>,
        limits: ProjectionLimits,
    ) -> Result<Option<ProjectedExternalTable>, ProjectionError> {
        if boundary_columns.len() > limits.max_frontier_width {
            return Err(ProjectionError::FrontierLimit {
                observed: boundary_columns.len(),
                limit: limits.max_frontier_width,
            });
        }
        let mut arena = WitnessArena::default();
        let pivots = self.external_row_pivots(&mut arena, limits.max_witness_nodes)?;
        let mut rows = Vec::with_capacity(complete.len());
        for mut full in complete {
            self.clear_cross_centers(std::slice::from_mut(&mut full.paulis));
            if full.paulis.is_identity() {
                if full.phase() != 0 {
                    return Ok(None);
                }
                continue;
            }
            let Some(witness) =
                self.external_row_witness(&full, &pivots, &mut arena, limits.max_witness_nodes)?
            else {
                return Ok(None);
            };
            rows.push(SparseRow {
                paulis: boundary_columns
                    .iter()
                    .filter_map(|&column| {
                        let pauli = full.paulis.get(column);
                        (pauli != Pauli::I).then_some((column, pauli))
                    })
                    .collect(),
                witness,
            });
        }
        self.finish_projected_table(
            rows,
            Vec::new(),
            arena,
            FlowSourceKeys::default(),
            boundary_columns,
            limits,
        )
        .map(Some)
    }

    /// Canonicalize only compositor-owned unsigned support; original local
    /// tensors certify the sign and witness of every resulting basis row.
    pub(crate) fn certify_compact_relation(
        &self,
        complete: CompactSurfaceRelation,
        boundary_columns: &[usize],
        limits: ProjectionLimits,
        mut consume: impl FnMut(WitnessedRow),
    ) -> Result<Option<usize>, ProjectionError> {
        if boundary_columns.len() > limits.max_frontier_width {
            return Err(ProjectionError::FrontierLimit {
                observed: boundary_columns.len(),
                limit: limits.max_frontier_width,
            });
        }
        let CompactSurfaceRelation {
            mut rows,
            queries,
            width,
        } = complete;
        if self.adjacency.has_parallel_edges()
            || self.stabilizer_phase_basis.get().is_some()
            || queries.len() != self.total_ids
            || queries.iter().flatten().flatten().any(|&bit| bit >= width)
        {
            return Ok(None);
        }
        let component = |row: &super::CoeffVec, column: usize, axis: Pauli| {
            queries[column][usize::from(axis == Pauli::Z)].is_some_and(|bit| row.bit(bit))
        };
        let mut boundary_order = vec![None; width];
        for (ordinal, &column) in boundary_columns.iter().enumerate() {
            for (axis, query) in [Pauli::X, Pauli::Z].into_iter().zip(queries[column]) {
                if let Some(bit) = query {
                    boundary_order[bit].get_or_insert((ordinal, axis));
                }
            }
        }
        let mut physical_order = vec![None; width];
        for (column, queries) in queries.iter().enumerate() {
            for (axis, query) in [Pauli::X, Pauli::Z].into_iter().zip(queries) {
                if let Some(bit) = query {
                    physical_order[*bit].get_or_insert((column, axis));
                }
            }
        }
        if rows.iter().any(|row| {
            row.iter_ones()
                .any(|bit| bit >= width || physical_order[bit].is_none())
        }) {
            return Ok(None);
        }
        let cancellation = crate::CancellationToken::current();
        let checkpoint = || {
            cancellation
                .as_ref()
                .map_or(Ok(()), crate::CancellationToken::check)
                .map_err(ProjectionError::from)
        };
        let boundary_len = try_leading_pivot_elimination_from(
            &mut rows,
            0,
            |row| {
                row.iter_ones()
                    .filter_map(|bit| boundary_order[bit])
                    .min_by_key(|&(column, axis)| (column, axis == Pauli::Z))
            },
            |row, &(ordinal, axis)| component(row, boundary_columns[ordinal], axis),
            checkpoint,
        )?;
        let closed_len = try_leading_pivot_elimination_from(
            &mut rows[boundary_len..],
            0,
            |row| {
                row.iter_ones()
                    .filter_map(|bit| physical_order[bit])
                    .min_by_key(|&(column, axis)| (column, axis == Pauli::Z))
            },
            |row, &(column, axis)| component(row, column, axis),
            checkpoint,
        )?;
        rows.truncate(boundary_len + closed_len);
        let mut arena = WitnessArena::default();
        let pivots = self.external_row_pivots(&mut arena, limits.max_witness_nodes)?;
        let local_nodes = arena.nodes.len();
        let mut source_keys = FlowSourceKeys::default();
        for row in rows {
            checkpoint()?;
            let paulis = PauliString::from_terms(
                self.total_ids,
                (0..self.total_ids).filter_map(|column| {
                    let pauli = match (
                        component(&row, column, Pauli::X),
                        component(&row, column, Pauli::Z),
                    ) {
                        (false, false) => return None,
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                    };
                    Some((column, pauli))
                }),
            );
            let phase = self.external_stabilizer_row_phases(std::slice::from_ref(&paulis))[0];
            let signed = PhasedPauliString::new(paulis, phase);
            let Some(witness) =
                self.external_row_witness(&signed, &pivots, &mut arena, limits.max_witness_nodes)?
            else {
                return Ok(None);
            };
            let flow_witness = self.flow_witness(&arena, witness, &mut source_keys);
            consume(WitnessedRow {
                signed,
                flow_witness,
            });
            // FlowWitness owns its source keys before row-local XOR nodes expire.
            arena.nodes.truncate(local_nodes);
        }
        Ok(Some(boundary_len))
    }

    fn external_row_pivots(
        &self,
        arena: &mut WitnessArena,
        limit: usize,
    ) -> Result<BTreeMap<(usize, bool), SparseRow>, ProjectionError> {
        let mut pivots = BTreeMap::new();
        let cancellation = crate::CancellationToken::current();
        for node in &self.nodes {
            if let Some(token) = &cancellation {
                token.check()?;
            }
            let mut local = self.projected_source_rows(
                &[node.id],
                &FxHashSet::from_iter([node.id]),
                arena,
                limit,
            )?;
            let columns = local
                .iter()
                .flat_map(|row| row.paulis.keys().copied())
                .collect::<BTreeSet<_>>();
            let rank = projected_gaussian_elimination(
                &mut local,
                axis_pivot_constraints(columns),
                |row, &(column, axis)| row.get(column) & axis,
                arena,
                limit,
            )?;
            for row in local.into_iter().take(rank) {
                let (&column, &pauli) = row.paulis.first_key_value().expect("local pivot");
                pivots.insert((column, pauli & Pauli::X), row);
            }
        }
        Ok(pivots)
    }

    fn external_row_witness(
        &self,
        full: &PhasedPauliString,
        pivots: &BTreeMap<(usize, bool), SparseRow>,
        arena: &mut WitnessArena,
        limit: usize,
    ) -> Result<Option<usize>, ProjectionError> {
        if !self.row_has_external_edge_pair_support(&full.paulis) {
            return Ok(None);
        }
        let mut residual = full.paulis.iter_support().collect::<BTreeMap<_, _>>();
        let mut witness = None;
        let cancellation = crate::CancellationToken::current();
        while let Some((&column, &pauli)) = residual.first_key_value() {
            if let Some(token) = &cancellation {
                token.check()?;
            }
            let Some(pivot) = pivots.get(&(column, pauli & Pauli::X)) else {
                return Ok(None);
            };
            for (&column, &pauli) in &pivot.paulis {
                let value = residual.get(&column).copied().unwrap_or(Pauli::I) ^ pauli;
                if value == Pauli::I {
                    residual.remove(&column);
                } else {
                    residual.insert(column, value);
                }
            }
            witness = Some(match witness {
                None => pivot.witness,
                Some(previous) => arena.push_xor(previous, pivot.witness, limit)?,
            });
        }
        Ok(witness.filter(|&witness| arena.materialize(witness, self.total_ids) == *full))
    }

    fn finish_projected_table(
        &self,
        mut rows: Vec<SparseRow>,
        mut closed_candidates: Vec<WitnessedRow>,
        mut arena: WitnessArena,
        mut source_keys: FlowSourceKeys,
        boundary_columns: &[usize],
        limits: ProjectionLimits,
    ) -> Result<ProjectedExternalTable, ProjectionError> {
        let rank = projected_gaussian_elimination(
            &mut rows,
            axis_pivot_constraints(boundary_columns.iter().copied()),
            |row, &(column, axis)| row.get(column) & axis,
            &mut arena,
            limits.max_witness_nodes,
        )?;
        let mut boundary_rows = Vec::with_capacity(rank);
        let mut residual = rows.split_off(rank);
        for row in rows {
            let materialized = arena.materialize(row.witness, self.total_ids);
            let flow_witness = self.flow_witness(&arena, row.witness, &mut source_keys);
            boundary_rows.push(ProjectedRow {
                signed: PhasedPauliString::new(
                    PauliString::from_terms(
                        boundary_columns.len(),
                        boundary_columns
                            .iter()
                            .enumerate()
                            .map(|(target, &source)| (target, row.get(source))),
                    ),
                    materialized.phase(),
                ),
                witness: row.witness,
                flow_witness,
            });
        }
        retire_closed(
            self,
            &mut residual,
            &mut closed_candidates,
            &arena,
            &mut source_keys,
        );
        assert!(
            residual.is_empty(),
            "complete boundary pivots leave no support"
        );
        let closed_rows = canonical_witnessed_external_basis(closed_candidates, self.total_ids);
        Ok(ProjectedExternalTable {
            boundary_rows,
            closed_rows,
            arena,
            width: self.total_ids,
        })
    }

    fn projected_source_rows(
        &self,
        node_ids: &[usize],
        retained: &FxHashSet<usize>,
        arena: &mut WitnessArena,
        limit: usize,
    ) -> Result<Vec<SparseRow>, ProjectionError> {
        self.local_signed_flow_supports(node_ids, None)
            .into_iter()
            .map(|source| {
                let paulis = source
                    .terms
                    .iter()
                    .copied()
                    .filter(|&(column, _)| {
                        column >= self.nodes.len()
                            || retained.contains(&column)
                            || self.neighbor_ids(column).is_empty()
                    })
                    .collect();
                let witness = arena.push_source(source, limit)?;
                Ok(SparseRow { paulis, witness })
            })
            .collect()
    }

    fn flow_source_key(&self, source: &SourceRow) -> FlowSourceKey {
        let mut node = Pauli::I;
        let mut edges = Vec::new();
        let mut frame_y = false;
        for &(column, mut pauli) in &source.terms {
            if column < self.nodes.len() {
                debug_assert_eq!(column, source.anchor);
                node = pauli;
                continue;
            }
            let edge = &self.edges[column - self.nodes.len()];
            debug_assert_eq!(edge.n1, source.anchor);
            if edge.hadamard && edge.n1 > edge.n2 {
                frame_y ^= pauli == Pauli::Y;
                pauli = pauli.flip();
            }
            let delta = self.nodes[edge.n2].pos - self.nodes[source.anchor].pos;
            let direction = Direction::try_from(delta).unwrap_or_else(|_| {
                // Walking and patch-rotation blocks keep one ZX node at their
                // start, while their far endpoint only admits a temporal pipe.
                Direction::try_from(IVec3::Z * delta.z.signum())
                    .expect("extended-block ZX edges are temporal")
            });
            edges.push((direction, pauli));
        }
        edges.sort_unstable_by_key(|(direction, _)| direction.to_ivec3().to_array());
        FlowSourceKey {
            position: self.nodes[source.anchor].pos,
            node,
            edges,
            intrinsic_phase: (source.phase + 2 * u8::from(frame_y)) % 4,
        }
    }

    fn flow_witness(
        &self,
        arena: &WitnessArena,
        witness: usize,
        source_keys: &mut FlowSourceKeys,
    ) -> FlowWitness {
        let mut sources = arena
            .source_indices(witness)
            .into_iter()
            .map(|index| {
                let WitnessNode::Source(source) = &arena.nodes[index] else {
                    unreachable!("source set contains only source nodes")
                };
                Arc::clone(
                    source_keys
                        .entry(index)
                        .or_insert_with(|| Arc::new(self.flow_source_key(source))),
                )
            })
            .collect::<Vec<_>>();
        sources.sort_unstable_by(|left, right| flow_source_key_cmp(left, right));
        FlowWitness { sources }
    }

    /// Reuses local sources only while replaying against this exact immutable graph.
    pub(crate) fn flow_witness_replayer(
        &self,
    ) -> impl FnMut(&FlowWitness, &PauliString) -> Option<PhasedPauliString> + '_ {
        let mut local_sources =
            crate::FxHashMap::<usize, Vec<(SourceRow, Option<FlowSourceKey>)>>::default();
        move |witness: &FlowWitness, expected: &PauliString| {
            let exact_sources = witness
                .sources
                .iter()
                .map(|key| {
                    let node = self.node_at(key.position)?;
                    let candidates = local_sources.entry(node.id).or_insert_with(|| {
                        self.local_signed_flow_supports(&[node.id], None)
                            .into_iter()
                            .map(|source| (source, None))
                            .collect()
                    });
                    let ordinal = candidates.iter_mut().position(|(source, stored_key)| {
                        &*stored_key.get_or_insert_with(|| self.flow_source_key(source))
                            == key.as_ref()
                    })?;
                    Some((node.pos.z, node.id, ordinal))
                })
                .collect::<Option<Vec<_>>>();
            if let Some(mut sources) = exact_sources {
                sources.sort_unstable();
                let mut row = PhasedPauliString::positive(PauliString::new(self.total_ids));
                for (_, node, ordinal) in sources {
                    local_sources[&node][ordinal].0.multiply_into(&mut row);
                }
                if &row.paulis == expected {
                    return Some(row);
                }
            }

            let mut anchors = BTreeSet::new();
            for key in &witness.sources {
                anchors.insert(self.node_at(key.position)?.id);
            }
            for (column, _) in expected.iter_support() {
                if column < self.nodes.len() {
                    anchors.insert(column);
                } else if let Some(edge) = self.edges.get(column - self.nodes.len()) {
                    anchors.insert(edge.n1);
                }
            }
            // Rotation can change which canonical local flows span one support.
            // Rebuild only at the witness/support anchors, never from the full graph.
            let anchors = anchors.into_iter().collect::<Vec<_>>();
            let basis = canonical_external_basis(
                self.local_stabilizer_flow_rows_for_nodes(&anchors, None),
                self.total_ids,
            );
            let mut residual = expected.clone();
            let mut row = PhasedPauliString::positive(PauliString::new(self.total_ids));
            for source in &basis {
                let (column, axis) = first_component(&source.paulis)
                    .expect("canonical local flow basis excludes identity rows");
                if residual.get(column) & axis {
                    residual ^= &source.paulis;
                    row.multiply_assign(source);
                }
            }
            residual.is_identity().then_some(row)
        }
    }

    pub(crate) fn to_external_generator_table(&self) -> Vec<PauliString> {
        self.to_signed_external_generator_table()
            .into_iter()
            .map(|row| row.paulis)
            .collect()
    }

    /// Recover a complete tensor row's phase before the YY edge contractions.
    /// Half-edges use the smaller endpoint's frame; spider crossing centers
    /// may be display support rather than independent tensor factors.
    fn local_row_phase(&self, row: &PauliString, ignore_cross_centers: bool) -> Option<u8> {
        let nodes = self.phase_node_candidates(row);
        self.local_row_phase_at(row, ignore_cross_centers, nodes.as_deref())
    }

    pub(super) fn materialized_row_phase(&self, row: &PauliString, nodes: Option<&[usize]>) -> u8 {
        if self.phase_basis().is_some() {
            return self.stabilizer_row_phases(std::slice::from_ref(row))[0];
        }
        self.local_row_phase_at(row, true, nodes).unwrap_or(0)
    }

    fn local_row_phase_at(
        &self,
        row: &PauliString,
        ignore_cross_centers: bool,
        nodes: Option<&[usize]>,
    ) -> Option<u8> {
        if (row.len() > self.total_ids
            && row
                .iter_support()
                .any(|(column, _)| column >= self.total_ids))
            || !self.row_has_external_edge_pair_support(row)
        {
            return None;
        }
        let mut sign = false;
        for index in 0..nodes.map_or(self.nodes.len(), <[usize]>::len) {
            let node = &self.nodes[nodes.map_or(index, |nodes| nodes[index])];
            let center = row.get(node.id);
            let mut arms = self.neighbor_edges(node.id).map(|(neighbor, column)| {
                let edge = self.edge_by_id(column);
                let pauli = row.get(column);
                if edge.hadamard && node.id > neighbor {
                    sign ^= pauli == Pauli::Y;
                    pauli.flip()
                } else {
                    pauli
                }
            });
            match node.kind {
                NodeKind::X | NodeKind::Z => {
                    let cross = node.kind.cross_pauli();
                    let broadcast = cross.flip();
                    if !ignore_cross_centers && center & cross {
                        return None;
                    }
                    let mut parity = false;
                    let mut ys = 0;
                    for arm in arms {
                        if (arm & broadcast) != (center & broadcast) {
                            return None;
                        }
                        parity ^= arm & cross;
                        ys += usize::from(arm == Pauli::Y);
                    }
                    if parity {
                        return None;
                    }
                    // Every pair of Y legs contributes -1 to an X/Z spider.
                    sign ^= ys & 2 != 0;
                }
                NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => {
                    if center != arms.next().unwrap_or(Pauli::I) || arms.any(|arm| arm != Pauli::I)
                    {
                        return None;
                    }
                    // The boundary's XX and ZZ generators multiply to -YY.
                    sign ^= center == Pauli::Y;
                }
                NodeKind::Y => {
                    if !matches!(center, Pauli::I | Pauli::Y)
                        || center != arms.next().unwrap_or(Pauli::I)
                        || arms.any(|arm| arm != Pauli::I)
                    {
                        return None;
                    }
                    sign ^= center == Pauli::Y
                        && self
                            .neighbor_ids(node.id)
                            .iter()
                            .all(|&neighbor| self.nodes[neighbor].pos.z < node.pos.z);
                }
            }
        }
        Some(2 * u8::from(sign))
    }

    fn phase_basis(&self) -> Option<&StabilizerPhaseBasis> {
        if self.adjacency.has_parallel_edges() {
            // Parallel neighbor entries alias their last edge column. Retain
            // signed reduction for that legacy incidence convention.
            Some(self.stabilizer_phase_basis.get_or_init(|| {
                Arc::new(StabilizerPhaseBasis::new(
                    self.to_signed_external_generator_table(),
                ))
            }))
        } else {
            self.stabilizer_phase_basis.get().map(Arc::as_ref)
        }
    }

    pub(crate) fn contains_stabilizer_support(&self, row: &PauliString) -> bool {
        let Some(basis) = self.phase_basis() else {
            return self.local_row_phase(row, true).is_some();
        };
        let mut residual = row.clone();
        self.clear_cross_centers(std::slice::from_mut(&mut residual));
        for (row, &(column, axis)) in basis.rows.iter().zip(&basis.pivots) {
            if residual.get(column) & axis {
                residual ^= &row.paulis;
            }
        }
        residual.is_identity()
    }

    pub(crate) fn stabilizer_row_phases(&self, rows: &[PauliString]) -> Vec<u8> {
        // A composed certificate is authoritative even after an X/Z fill.
        // Complete graph-local rows need only their local tensor signs.
        let Some(basis) = self.phase_basis() else {
            return rows
                .iter()
                .map(|row| self.local_row_phase(row, true).unwrap_or(0))
                .collect();
        };
        // Cross centers are display support, not independent Pauli factors.
        let mut raw = rows.to_vec();
        self.clear_cross_centers(&mut raw);
        self.row_phases_with_pivots(&raw, &basis.rows, &basis.pivots)
    }

    pub(crate) fn external_stabilizer_row_phases(&self, rows: &[PauliString]) -> Vec<u8> {
        if self.adjacency.has_parallel_edges() {
            return self.row_phases_against(rows, &self.to_signed_external_generator_table());
        }
        rows.iter()
            .map(|row| self.local_row_phase(row, false).unwrap_or(0))
            .collect()
    }

    pub(crate) fn row_phases_against(
        &self,
        rows: &[PauliString],
        basis: &[PhasedPauliString],
    ) -> Vec<u8> {
        self.row_phases_with_pivots(rows, basis, &phase_pivots(basis))
    }

    fn row_phases_with_pivots(
        &self,
        rows: &[PauliString],
        basis: &[PhasedPauliString],
        pivots: &[(usize, Pauli)],
    ) -> Vec<u8> {
        rows.iter()
            .map(|row| {
                let mut residual = row.clone();
                let mut product = PhasedPauliString::positive(PauliString::new(self.total_ids));
                for (basis_row, &(col, axis)) in basis.iter().zip(pivots) {
                    if residual.get(col) & axis {
                        residual ^= &basis_row.paulis;
                        product.multiply_assign(basis_row);
                    }
                }
                // Synthetic fixing/correction rows outside the physical span
                // are phaseless.
                if residual.is_identity() {
                    product.phase()
                } else {
                    0
                }
            })
            .collect()
    }

    pub(crate) fn to_signed_external_generator_table(&self) -> Vec<PhasedPauliString> {
        let layout = EdgeFlowLayout::new(self);
        let mut rows = Vec::new();
        for (node_ids, same_layer_pairs, pending_edge_pairs) in self.stabilizer_layers() {
            let mut local = self.local_edge_flow_rows_for_nodes(&node_ids, &layout);
            eliminate_edge_flow_pairs(&mut local, &same_layer_pairs, &layout);
            rows.extend(local);
            eliminate_edge_flow_pairs(&mut rows, &pending_edge_pairs, &layout);
        }
        // Public graph ids and diagnostic consumers still expose materialized
        // center support; it is absent from the seam computation above.
        let rows = rows.into_iter().map(EdgeFlowRow::materialize).collect();
        canonical_external_basis(rows, self.total_ids)
    }

    fn stabilizer_layers(&self) -> Vec<(Vec<usize>, Vec<EdgePair>, Vec<EdgePair>)> {
        let mut layers = BTreeMap::<i32, (Vec<usize>, Vec<EdgePair>, Vec<EdgePair>)>::new();
        for node in &self.nodes {
            layers.entry(node.pos.z).or_default().0.push(node.id);
        }
        for edge in self.edges.iter().filter(|edge| edge.n1 < edge.n2) {
            let n1_z = self.nodes[edge.n1].pos.z;
            let n2_z = self.nodes[edge.n2].pos.z;
            let layer = layers.entry(n1_z.max(n2_z)).or_default();
            if n1_z == n2_z {
                layer.1.push(self.edge_column_pair(edge));
            } else {
                layer.2.push(self.edge_column_pair(edge));
            }
        }
        for (_, same_layer, pending) in layers.values_mut() {
            same_layer.sort_unstable();
            pending.sort_unstable();
        }
        layers.into_values().collect()
    }

    fn local_stabilizer_flow_rows_for_nodes(
        &self,
        layer_nodes: &[usize],
        node_kinds: Option<&[NodeKind]>,
    ) -> Vec<PhasedPauliString> {
        self.local_signed_flow_supports(layer_nodes, node_kinds)
            .into_iter()
            .map(|source| {
                PhasedPauliString::new(
                    PauliString::from_terms(self.total_ids, source.terms),
                    source.phase,
                )
            })
            .collect()
    }

    fn local_edge_flow_rows_for_nodes<'a>(
        &self,
        layer_nodes: &[usize],
        layout: &'a EdgeFlowLayout<'a>,
    ) -> Vec<EdgeFlowRow<'a>> {
        self.local_signed_flow_supports(layer_nodes, None)
            .into_iter()
            .map(|source| EdgeFlowRow {
                signed: PhasedPauliString::new(
                    layout.project_terms(source.terms.into_iter()),
                    source.phase,
                ),
                layout,
            })
            .collect()
    }

    fn local_signed_flow_supports(
        &self,
        layer_nodes: &[usize],
        node_kinds: Option<&[NodeKind]>,
    ) -> Vec<SourceRow> {
        layer_nodes
            .iter()
            .copied()
            .flat_map(|node_id| {
                let kind = node_kinds.map_or(self.nodes[node_id].kind, |kinds| kinds[node_id]);
                self.local_stabilizer_flow_supports_for_node_kind(node_id, kind)
                    .into_iter()
                    .map(move |terms| {
                        // Rows use the smaller endpoint's Pauli frame. Both a
                        // Hadamard-frame Y and an incoming Choi Y transpose
                        // contribute H Y H = Y^T = -Y.
                        let hadamard_y = self
                            .neighbor_edges(node_id)
                            .filter(|&(neighbor, edge_id)| {
                                let edge = self.edge_by_id(edge_id);
                                edge.hadamard
                                    && node_id > neighbor
                                    && terms.iter().any(|&(column, pauli)| {
                                        column == edge.id && pauli == Pauli::Y
                                    })
                            })
                            .count()
                            % 2
                            == 1;
                        let y_effect = kind == NodeKind::Y
                            && self.neighbor_ids(node_id).iter().all(|&neighbor| {
                                self.nodes[neighbor].pos.z < self.nodes[node_id].pos.z
                            });
                        SourceRow {
                            anchor: node_id,
                            terms,
                            phase: 2 * u8::from(hadamard_y ^ y_effect),
                        }
                    })
            })
            .collect()
    }
}

fn eliminate_edge_flow_pairs(
    rows: &mut Vec<EdgeFlowRow<'_>>,
    pairs: &[EdgePair],
    layout: &EdgeFlowLayout<'_>,
) {
    let eliminated = signed_gaussian_elimination(
        rows,
        pairs.iter().flat_map(|&(left, right)| {
            let left = left - layout.zx.nodes.len();
            let right = right - layout.zx.nodes.len();
            [(left, right, Pauli::X), (left, right, Pauli::Z)]
        }),
        |row, &(left, right, axis)| {
            (row.signed.paulis.get(left) ^ row.signed.paulis.get(right)) & axis
        },
    );
    rows.drain(..eliminated);
    rows.retain(|row| !row.signed.paulis.is_identity());
}

#[cfg(test)]
fn eliminate_edge_pairs(rows: &mut Vec<PhasedPauliString>, pairs: &[EdgePair]) {
    let eliminated = signed_gaussian_elimination(
        rows,
        pairs
            .iter()
            .flat_map(|&(c1, c2)| [(c1, c2, Pauli::X), (c1, c2, Pauli::Z)]),
        |row, &(c1, c2, diff)| (row.paulis.get(c1) ^ row.paulis.get(c2)) & diff,
    );
    rows.drain(..eliminated);
    rows.retain(|row| !row.paulis.is_identity());
}

fn eliminate_projected_edge_pairs(
    rows: &mut Vec<SparseRow>,
    pairs: &[EdgePair],
    arena: &mut WitnessArena,
    witness_limit: usize,
) -> Result<(), ProjectionError> {
    let eliminated = projected_gaussian_elimination(
        rows,
        pairs
            .iter()
            .flat_map(|&(left, right)| [(left, right, Pauli::X), (left, right, Pauli::Z)]),
        |row, &(left, right, axis)| (row.get(left) ^ row.get(right)) & axis,
        arena,
        witness_limit,
    )?;
    rows.drain(..eliminated);
    Ok(())
}

fn projected_gaussian_elimination<T>(
    rows: &mut [SparseRow],
    constraints: impl IntoIterator<Item = T>,
    predicate: impl Fn(&SparseRow, &T) -> bool,
    arena: &mut WitnessArena,
    witness_limit: usize,
) -> Result<usize, ProjectionError> {
    let mut solved = 0;
    for constraint in constraints {
        let Some(pivot_index) =
            (solved..rows.len()).find(|&row| predicate(&rows[row], &constraint))
        else {
            continue;
        };
        rows.swap(pivot_index, solved);
        let (before, pivot_and_after) = rows.split_at_mut(solved);
        let (pivot, after) = pivot_and_after
            .split_first_mut()
            .expect("solved pivot is in bounds");
        for row in before.iter_mut().chain(after) {
            if predicate(row, &constraint) {
                row.xor_assign(pivot, arena, witness_limit)?;
            }
        }
        solved += 1;
    }
    Ok(solved)
}

fn remove_columns(rows: &mut [SparseRow], columns: impl IntoIterator<Item = usize>) {
    let columns = columns.into_iter().collect::<FxHashSet<_>>();
    for row in rows {
        row.paulis.retain(|column, _| !columns.contains(column));
    }
}

fn retire_closed(
    zx: &ZXGraph,
    rows: &mut Vec<SparseRow>,
    closed: &mut Vec<WitnessedRow>,
    arena: &WitnessArena,
    source_keys: &mut FlowSourceKeys,
) {
    rows.retain(|row| {
        if !row.paulis.is_empty() {
            return true;
        }
        let signed = arena.materialize(row.witness, zx.total_ids);
        if !signed.paulis.is_identity() {
            closed.push(WitnessedRow {
                signed,
                flow_witness: zx.flow_witness(arena, row.witness, source_keys),
            });
        }
        false
    });
}

fn check_frontier(rows: &[SparseRow], limit: usize) -> Result<(), ProjectionError> {
    let width = rows
        .iter()
        .flat_map(|row| row.paulis.keys().copied())
        .collect::<FxHashSet<_>>()
        .len();
    if width <= limit {
        Ok(())
    } else {
        Err(ProjectionError::FrontierLimit {
            observed: width,
            limit,
        })
    }
}

pub(super) fn canonical_external_basis(
    mut rows: Vec<PhasedPauliString>,
    width: usize,
) -> Vec<PhasedPauliString> {
    let rank = leading_pivot_elimination_from(
        &mut rows,
        0,
        |row| first_component(&row.paulis).filter(|(column, _)| *column < width),
        |row, &(col, axis)| row.paulis.get(col) & axis,
    );
    rows.truncate(rank);
    rows
}

fn canonical_witnessed_external_basis(
    mut rows: Vec<WitnessedRow>,
    width: usize,
) -> Vec<WitnessedRow> {
    let rank = leading_pivot_elimination_from(
        &mut rows,
        0,
        |row| first_component(&row.signed.paulis).filter(|(column, _)| *column < width),
        |row, &(col, axis)| row.signed.paulis.get(col) & axis,
    );
    rows.truncate(rank);
    rows
}

/// A row that Gaussian elimination can fold into another. Implementors carry
/// whatever travels with the relation row — a flow witness, positional support —
/// so everything folds in one pass.
pub(crate) trait EliminationRow {
    fn multiply_assign(&mut self, pivot: &Self);
}

impl EliminationRow for super::CoeffVec {
    fn multiply_assign(&mut self, pivot: &Self) {
        self.xor_assign(pivot);
    }
}

impl EliminationRow for PhasedPauliString {
    fn multiply_assign(&mut self, pivot: &Self) {
        PhasedPauliString::multiply_assign(self, pivot);
    }
}

impl EliminationRow for WitnessedRow {
    fn multiply_assign(&mut self, pivot: &Self) {
        self.signed.multiply_assign(&pivot.signed);
        self.flow_witness.xor_assign(&pivot.flow_witness);
    }
}

/// Reduce columns in ascending order, X before Z, from a solved prefix.
/// `first` names each row's earliest nonzero component under `predicate`.
/// Folding a pivot must clear that component without reintroducing earlier ones.
pub(crate) fn leading_pivot_elimination_from<R: EliminationRow>(
    rows: &mut [R],
    solved: usize,
    first: impl Fn(&R) -> Option<(usize, Pauli)>,
    predicate: impl Fn(&R, &(usize, Pauli)) -> bool,
) -> usize {
    try_leading_pivot_elimination_from(rows, solved, first, predicate, || {
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap_or_else(|never| match never {})
}

/// The same sparse elimination with cancellable checkpoints between row operations.
pub(crate) fn try_leading_pivot_elimination_from<R: EliminationRow, E>(
    rows: &mut [R],
    mut solved: usize,
    first: impl Fn(&R) -> Option<(usize, Pauli)>,
    predicate: impl Fn(&R, &(usize, Pauli)) -> bool,
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<usize, E> {
    checkpoint()?;
    debug_assert!(solved <= rows.len());
    let mut leading: Vec<_> = rows.iter().map(&first).collect();
    let mut pending: BTreeSet<_> = leading
        .iter()
        .enumerate()
        .skip(solved)
        .filter_map(|(index, &component)| {
            component.map(|(column, axis)| (column, axis == Pauli::Z, index))
        })
        .collect();
    // Empty columns have no candidate. Ties retain the original row order.
    while let Some((column, z_axis, pivot_index)) = pending.pop_first() {
        checkpoint()?;
        let component = (column, if z_axis { Pauli::Z } else { Pauli::X });
        if pivot_index != solved
            && let Some((column, axis)) = leading[solved]
        {
            pending.remove(&(column, axis == Pauli::Z, solved));
            pending.insert((column, axis == Pauli::Z, pivot_index));
        }
        rows.swap(pivot_index, solved);
        leading.swap(pivot_index, solved);
        let (before, pivot_and_after) = rows.split_at_mut(solved);
        let (pivot, after) = pivot_and_after
            .split_first_mut()
            .expect("pivot is in bounds");
        for row in before {
            if predicate(row, &component) {
                checkpoint()?;
                row.multiply_assign(pivot);
            }
        }
        // These are exactly the unsolved rows carrying this component, in order.
        while let Some(&(next_column, next_z_axis, index)) = pending.first() {
            if (next_column, next_z_axis) != (column, z_axis) {
                break;
            }
            pending.pop_first();
            checkpoint()?;
            let row = &mut after[index - solved - 1];
            row.multiply_assign(pivot);
            leading[index] = first(row);
            if let Some((next_column, axis)) = leading[index] {
                debug_assert!((next_column, axis == Pauli::Z) > (column, z_axis));
                pending.insert((next_column, axis == Pauli::Z, index));
            }
        }
        solved += 1;
    }
    Ok(solved)
}

pub(crate) fn signed_gaussian_elimination<R: EliminationRow, T>(
    rows: &mut [R],
    pivot_constraints: impl IntoIterator<Item = T>,
    predicate: impl Fn(&R, &T) -> bool,
) -> usize {
    signed_gaussian_elimination_from(rows, 0, pivot_constraints, predicate)
}

/// Continue the same ordered elimination after a previously solved prefix.
pub(crate) fn signed_gaussian_elimination_from<R: EliminationRow, T>(
    rows: &mut [R],
    solved: usize,
    pivot_constraints: impl IntoIterator<Item = T>,
    predicate: impl Fn(&R, &T) -> bool,
) -> usize {
    try_signed_gaussian_elimination_from(
        rows,
        solved,
        pivot_constraints,
        |row, constraint| Ok::<_, std::convert::Infallible>(predicate(row, constraint)),
        || Ok(()),
    )
    .unwrap_or_else(|never| match never {})
}

/// Ordered full RREF with fallible queries and a checkpoint before each row fold.
pub(crate) fn try_signed_gaussian_elimination_from<R: EliminationRow, T, E>(
    rows: &mut [R],
    mut solved: usize,
    pivot_constraints: impl IntoIterator<Item = T>,
    mut predicate: impl FnMut(&R, &T) -> Result<bool, E>,
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<usize, E> {
    debug_assert!(solved <= rows.len());
    for constraint in pivot_constraints {
        let mut pivot_index = None;
        for (index, row) in rows.iter().enumerate().skip(solved) {
            if predicate(row, &constraint)? {
                pivot_index = Some(index);
                break;
            }
        }
        let Some(pivot_index) = pivot_index else {
            continue;
        };
        rows.swap(pivot_index, solved);
        let (before, pivot_and_after) = rows.split_at_mut(solved);
        let (pivot, after) = pivot_and_after
            .split_first_mut()
            .expect("solved pivot is in bounds");
        // The pivot search already proved the skipped unsolved rows zero.
        // Swapping the pivot with `solved` leaves that whole prefix zero.
        for row in before.iter_mut().chain(&mut after[pivot_index - solved..]) {
            if predicate(row, &constraint)? {
                checkpoint()?;
                row.multiply_assign(pivot);
            }
        }
        solved += 1;
        if solved == rows.len() {
            break;
        }
    }
    Ok(solved)
}

fn first_component(row: &PauliString) -> Option<(usize, Pauli)> {
    row.iter_support()
        .next()
        .map(|(column, pauli)| (column, if pauli & Pauli::X { Pauli::X } else { Pauli::Z }))
}

fn phase_pivots(basis: &[PhasedPauliString]) -> Vec<(usize, Pauli)> {
    basis
        .iter()
        .map(|row| first_component(&row.paulis).expect("external basis excludes identity rows"))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};
    use std::sync::Arc;

    use bloq_utils::{Pauli, PauliString, PhasedPauliString};
    use glam::IVec3;

    use super::super::ZXGraph;
    use super::super::stabilizer::{axis_pivot_constraints, gaussian_elimination_with_tracking};
    use super::{FlowSourceKey, FlowWitness, ProjectionLimits};
    use crate::{GalleryItem, NodeKind};

    #[test]
    fn sparse_source_products_match_dense_phases_and_overwritten_terms() {
        const AXES: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y];
        fn check(row: &mut PhasedPauliString, terms: Vec<(usize, Pauli)>, phase: u8) {
            let source = super::SourceRow {
                anchor: 0,
                terms,
                phase,
            };
            let mut expected = row.clone();
            expected.multiply_assign(&PhasedPauliString::new(
                PauliString::from_terms(row.paulis.len(), source.terms.iter().copied()),
                source.phase,
            ));
            source.multiply_into(row);
            assert_eq!(*row, expected);
        }
        for width in [1, 65, 513] {
            for left in AXES {
                for right in AXES {
                    for left_phase in 0..4 {
                        for right_phase in 0..4 {
                            let mut row = PhasedPauliString::new(
                                PauliString::from_terms(width, [(width - 1, left)]),
                                left_phase,
                            );
                            check(&mut row, vec![(width - 1, right)], right_phase);
                        }
                    }
                }
            }
        }
        for width in [65, 513] {
            let mut row = PhasedPauliString::new(
                PauliString::from_terms(width, (0..width).map(|column| (column, AXES[column % 4]))),
                3,
            );
            for phase in 0..4 {
                check(
                    &mut row,
                    vec![
                        (width - 1, Pauli::Y),
                        (0, Pauli::Z),
                        (64, Pauli::X),
                        (width - 1, Pauli::I),
                        (1, Pauli::Y),
                        (0, Pauli::Y),
                        (64, Pauli::Z),
                    ],
                    phase,
                );
            }
            for terms in [
                vec![(0, Pauli::X), (64, Pauli::Y)],
                vec![(0, Pauli::Z), (64, Pauli::X)],
                vec![(0, Pauli::Y), (64, Pauli::Z)],
            ] {
                check(&mut row, terms, 2);
            }
        }
        for count in [8, 9, 64] {
            let mut row = PhasedPauliString::new(
                PauliString::from_terms(65, [(0, Pauli::Y), (64, Pauli::X)]),
                2,
            );
            let mut terms = (0..count)
                .rev()
                .map(|column| (column, AXES[column % 4]))
                .collect::<Vec<_>>();
            terms.extend([(0, Pauli::Z), (count - 1, Pauli::I)]);
            check(&mut row, terms, 3);
        }
        let mut empty = PhasedPauliString::positive(PauliString::new(0));
        check(&mut empty, Vec::new(), 2);
        assert_eq!(empty.phase(), 2);
    }

    #[test]
    fn sparse_source_products_validate_width_before_mutation() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        for width in [0, 1, 65, 513] {
            let source = super::SourceRow {
                anchor: 0,
                terms: vec![(0, Pauli::X), (width, Pauli::Y)],
                phase: 2,
            };
            let mut actual = PhasedPauliString::positive(PauliString::new(width));
            let before = actual.clone();
            catch_unwind(AssertUnwindSafe(|| {
                PauliString::from_terms(width, source.terms.iter().copied())
            }))
            .expect_err("dense source construction rejects an invalid width");
            catch_unwind(AssertUnwindSafe(|| source.multiply_into(&mut actual)))
                .expect_err("sparse source multiplication rejects an invalid width");
            assert_eq!(actual, before);
        }
    }

    #[test]
    fn ordered_elimination_checks_before_folding_a_packed_row() {
        let mut rows = vec![super::super::CoeffVec::singleton(0, 2); 2];
        let before = rows.clone();
        let token = crate::CancellationToken::new();
        let mut tested = 0;
        let error = token
            .scope(|| {
                super::try_signed_gaussian_elimination_from(
                    &mut rows,
                    0,
                    [0],
                    |row, &column| {
                        tested += 1;
                        if tested == 2 {
                            token.cancel();
                        }
                        Ok::<_, crate::StabilizerError>(row.bit(column))
                    },
                    crate::CancellationToken::checkpoint,
                )
            })
            .unwrap_err();
        assert!(matches!(error, crate::StabilizerError::Cancelled(_)));
        assert_eq!(rows, before, "cancel before the pending XOR");
        let denied = super::try_signed_gaussian_elimination_from(
            &mut rows,
            0,
            [0],
            |_, _| Err::<bool, _>("query budget"),
            || Ok(()),
        );
        assert_eq!(denied, Err("query budget"));
        assert_eq!(rows, before);
    }

    fn assert_rows_satisfy_external_edge_pair_support(zx: &ZXGraph, rows: &[PauliString]) {
        for row in rows {
            assert!(zx.row_has_external_edge_pair_support(row), "{row}");
        }
    }

    #[test]
    fn cached_leading_pivots_match_ordered_scalar_elimination() {
        for bits in 0..4096usize {
            let rows = [0usize, 1, 2].map(|index| {
                PhasedPauliString::new(
                    PauliString::from_terms(
                        2,
                        (0..2).map(|column| {
                            (
                                column,
                                [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y]
                                    [(bits >> (4 * index + 2 * column)) & 3],
                            )
                        }),
                    ),
                    ((bits + index) % 4) as u8,
                )
            });
            for width in 0..=2 {
                for solved in 0..=1 {
                    let mut expected = rows.clone();
                    let rank = super::signed_gaussian_elimination_from(
                        &mut expected,
                        solved,
                        axis_pivot_constraints(0..width),
                        |row, &(column, axis)| row.paulis.get(column) & axis,
                    );
                    let mut actual = rows.clone();
                    let actual_rank = super::leading_pivot_elimination_from(
                        &mut actual,
                        solved,
                        |row| {
                            super::first_component(&row.paulis)
                                .filter(|(column, _)| *column < width)
                        },
                        |row, &(column, axis)| row.paulis.get(column) & axis,
                    );
                    assert_eq!(
                        (actual_rank, actual),
                        (rank, expected),
                        "bits {bits}, width {width}, prefix {solved}"
                    );
                }
            }
        }
    }

    #[test]
    fn pivot_search_keeps_signed_basis_when_skipping_zero_prefix() {
        let signed =
            |text, phase| PhasedPauliString::new(PauliString::try_from(text).unwrap(), phase);
        let mut rows = [signed("_Z_", 2), signed("ZZ_", 0), signed("XX_", 2)];
        let rank = super::signed_gaussian_elimination(
            &mut rows,
            axis_pivot_constraints(0..3),
            |row, &(column, axis)| row.paulis.get(column) & axis,
        );
        assert_eq!(rank, 3);
        assert_eq!(rows, [signed("XX_", 2), signed("Z__", 2), signed("_Z_", 2)]);
    }

    #[test]
    fn flow_witness_xor_matches_set_symmetric_difference() {
        let source = |x| {
            Arc::new(FlowSourceKey {
                position: IVec3::new(x, 0, 0),
                node: Pauli::X,
                edges: Vec::new(),
                intrinsic_phase: 0,
            })
        };
        let keys = [0, 1, 2, 3, 4, 5].map(source);
        let select = |mask| {
            keys.iter()
                .enumerate()
                .filter(|&(index, _)| mask & (1 << index) != 0)
                .map(|(_, key)| Arc::clone(key))
                .collect::<Vec<_>>()
        };
        for left in 0..64u32 {
            for right in 0..64u32 {
                let mut actual = FlowWitness {
                    sources: select(left),
                };
                actual.xor_assign(&FlowWitness {
                    sources: select(right)
                        .iter()
                        // Equal contents must cancel even across distinct Arcs.
                        .map(|key| Arc::new(key.as_ref().clone()))
                        .collect(),
                });
                assert_eq!(
                    actual.sources,
                    select(left ^ right),
                    "{left:06b} XOR {right:06b}"
                );
            }
        }
    }

    fn free_boundary_columns(zx: &ZXGraph) -> Vec<usize> {
        zx.nodes
            .iter()
            .filter(|node| node.kind.is_boundary())
            .map(|node| node.id)
            .collect()
    }

    /// Rank of the external table restricted to the free boundary columns.
    ///
    /// A `Port`/`T`/`Selective` node's local flow rows put the surface's Pauli
    /// on the node column as well as on its single incident edge, so the node
    /// columns *are* the boundary data of a correlation surface.
    fn boundary_projection_rank(zx: &ZXGraph, rows: &[PauliString]) -> usize {
        let columns = free_boundary_columns(zx);
        let mut projected = rows
            .iter()
            .map(|row| {
                let mut restricted = PauliString::new(zx.total_ids);
                for &column in &columns {
                    restricted.set(column, row.get(column));
                }
                restricted
            })
            .collect::<Vec<_>>();
        gaussian_elimination_with_tracking(
            &mut projected,
            &mut [],
            axis_pivot_constraints(columns),
            |row, &(column, axis)| row.get(column) & axis,
            None,
        )
    }

    fn compact_projection(row: &PauliString, columns: &[usize]) -> PauliString {
        PauliString::from_terms(
            columns.len(),
            columns
                .iter()
                .enumerate()
                .map(|(target, &source)| (target, row.get(source))),
        )
    }

    /// Number of connected components of the underlying undirected graph that
    /// contain no free boundary node.
    fn boundaryless_component_count(zx: &ZXGraph) -> usize {
        let mut seen = HashSet::new();
        let mut boundaryless = 0;
        for node in &zx.nodes {
            if !seen.insert(node.id) {
                continue;
            }
            let mut has_boundary = node.kind.is_boundary();
            let mut queue = VecDeque::from([node.id]);
            while let Some(current) = queue.pop_front() {
                for &next in zx.neighbor_ids(current) {
                    has_boundary |= zx.nodes[next].kind.is_boundary();
                    if seen.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
            if !has_boundary {
                boundaryless += 1;
            }
        }
        boundaryless
    }

    // The three-bit adder covers composition in ordinary checks; the ten-bit
    // scaling case keeps the same oracle checks in ignored stress coverage.
    fn dense_oracle_galleries() -> impl Iterator<Item = GalleryItem> {
        GalleryItem::iter().filter(|entry| *entry != GalleryItem::TenBitAdder)
    }

    // Bound large structural domains. These checks construct their own ZX
    // tables and need no canonical readout analysis.
    fn gallery_projections(
        graph: &crate::BlockGraph,
    ) -> impl Iterator<Item = crate::BlockGraph> + '_ {
        graph
            .branch_assignments_up_to(16)
            .unwrap()
            .into_iter()
            .map(|assignment| graph.project_branches_deferred(assignment).unwrap())
    }

    #[test]
    fn raw_phase_recovery_clears_reconstructed_cross_centers() {
        let zx = ZXGraph::try_from(
            &GalleryItem::CNOT
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
        )
        .unwrap();
        let raw = zx
            .to_external_generator_table()
            .into_iter()
            .find(|row| {
                let mut reconstructed = row.clone();
                zx.reconstruct_cross_center(std::slice::from_mut(&mut reconstructed));
                reconstructed != *row
            })
            .expect("CNOT has a crossing surface");
        let signed = [PhasedPauliString::new(raw.clone(), 2)];
        let mut reconstructed = raw;
        zx.reconstruct_cross_center(std::slice::from_mut(&mut reconstructed));
        assert_eq!(
            zx.row_phases_against(&[reconstructed.clone()], &signed),
            [0]
        );

        zx.clear_cross_centers(std::slice::from_mut(&mut reconstructed));
        assert_eq!(zx.row_phases_against(&[reconstructed], &signed), [2]);
    }

    fn phase_test_star(kind: NodeKind, leaves: &[usize], hadamard: bool) -> ZXGraph {
        const NODES: usize = 129;
        const CENTER: usize = 64;
        let mut nodes = (0..NODES)
            .map(|id| {
                let z = if id == 0 {
                    0
                } else if id == NODES - 1 {
                    2
                } else {
                    1
                };
                crate::ZXNode::new(id, glam::ivec3(id as i32, 0, z), NodeKind::X)
            })
            .collect::<Vec<_>>();
        nodes[CENTER].kind = kind;
        let mut edges = Vec::new();
        for &leaf in leaves {
            nodes[leaf].kind = NodeKind::Port;
            for (n1, n2) in [(CENTER, leaf), (leaf, CENTER)] {
                edges.push(crate::ZXEdge {
                    n1,
                    n2,
                    id: NODES + edges.len(),
                    hadamard,
                });
            }
        }
        let adjacency = super::super::graph::CsrAdjacency::from_edges(
            NODES,
            &edges
                .iter()
                .map(|edge| (edge.n1, edge.n2, edge.id))
                .collect::<Vec<_>>(),
        );
        let pos_to_node = nodes.iter().map(|node| (node.pos, node.id)).collect();
        ZXGraph {
            total_ids: NODES + edges.len(),
            nodes,
            edges,
            adjacency,
            pos_to_node,
            action_graph: crate::ActionDag::default(),
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        }
    }

    #[test]
    fn sparse_local_phases_match_dense_traversal() {
        const CENTER: usize = 64;
        const AXES: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y];
        fn compare(zx: &ZXGraph, row: &PauliString, seen: &mut [bool; 3]) {
            for ignore_cross_centers in [false, true] {
                let expected = zx.local_row_phase_at(row, ignore_cross_centers, None);
                assert_eq!(
                    zx.local_row_phase(row, ignore_cross_centers),
                    expected,
                    "kind={:?}, ignore_cross={ignore_cross_centers}, row={row:?}",
                    zx.nodes[CENTER].kind,
                );
                seen[match expected {
                    Some(0) => 0,
                    Some(2) => 1,
                    None => 2,
                    Some(phase) => panic!("unexpected non-Hermitian tensor phase {phase}"),
                }] = true;
            }
        }

        let mut seen = [false; 3];
        for kind in [
            NodeKind::X,
            NodeKind::Y,
            NodeKind::Z,
            NodeKind::Port,
            NodeKind::T,
            NodeKind::Selective(crate::SelectiveKind::XY),
            NodeKind::Selective(crate::SelectiveKind::XZ),
            NodeKind::Selective(crate::SelectiveKind::YZ),
        ] {
            for leaves in [&[][..], &[0][..], &[128][..], &[0, 128][..]] {
                for hadamard in [false, true] {
                    let zx = phase_test_star(kind, leaves, hadamard);
                    // Enumerate the center and all original arm values. Paired
                    // directed columns agree; leaf centers use their native H frame.
                    // At most seven nonzero columns force the padded sparse path.
                    for bits in 0..(1usize << (2 * (leaves.len() + 1))) {
                        let mut row = PauliString::new(zx.total_ids);
                        row.set(CENTER, AXES[bits & 3]);
                        for (arm, &leaf) in leaves.iter().enumerate() {
                            let pauli = AXES[(bits >> (2 * (arm + 1))) & 3];
                            row.set(zx.edges[2 * arm].id, pauli);
                            row.set(zx.edges[2 * arm + 1].id, pauli);
                            row.set(
                                leaf,
                                if hadamard && leaf > CENTER {
                                    pauli.flip()
                                } else {
                                    pauli
                                },
                            );
                        }
                        compare(&zx, &row, &mut seen);
                        let mut reconstructed = row.clone();
                        zx.reconstruct_cross_center(std::slice::from_mut(&mut reconstructed));
                        let phase = zx
                            .local_row_phase_at(&reconstructed, true, None)
                            .unwrap_or(0);
                        let expected =
                            zx.pauli_string_to_stabilizer_with_row_phase(reconstructed, phase);
                        assert_eq!(zx.materialize_stabilizer(row), expected);
                    }
                    // Unmatched directed arms must fail before tensor evaluation.
                    for edge in &zx.edges {
                        for pauli in [Pauli::X, Pauli::Z, Pauli::Y] {
                            let mut row = PauliString::new(zx.total_ids);
                            row.set(edge.id, pauli);
                            assert_eq!(zx.local_row_phase_at(&row, true, None), None);
                            compare(&zx, &row, &mut seen);
                        }
                    }
                    // Dense fallback, including nonzero isolated padding centers.
                    for pauli in [Pauli::X, Pauli::Z, Pauli::Y] {
                        let row = PauliString::from_terms(
                            zx.total_ids,
                            (0..zx.total_ids).map(|column| (column, pauli)),
                        );
                        compare(&zx, &row, &mut seen);
                    }
                    // Many distinct zero-arity tensor centers exercise the
                    // gather-then-dense fallback without invalid local support.
                    let majority = PauliString::from_terms(
                        zx.total_ids,
                        (0..zx.nodes.len())
                            .filter(|&node| node != CENTER && !leaves.contains(&node))
                            .take(zx.nodes.len() / 2 + 1)
                            .map(|node| (node, Pauli::Z)),
                    );
                    compare(&zx, &majority, &mut seen);
                    let mut long = PauliString::new(zx.total_ids + 65);
                    compare(&zx, &long, &mut seen);
                    long.set(zx.total_ids + 64, Pauli::X);
                    assert_eq!(zx.local_row_phase_at(&long, true, None), None);
                    compare(&zx, &long, &mut seen);
                }
            }
        }
        assert_eq!(
            seen, [true; 3],
            "cover positive, negative, and rejected rows"
        );
    }

    #[test]
    fn local_phase_width_fallback_preserves_dense_results_and_panics() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        for leaves in [&[][..], &[0][..], &[0, 128][..]] {
            let zx = phase_test_star(NodeKind::Z, leaves, true);
            for width in [0, 64, 128, zx.total_ids - 1] {
                let row = PauliString::new(width);
                for ignore_cross_centers in [false, true] {
                    let expected = catch_unwind(AssertUnwindSafe(|| {
                        zx.local_row_phase_at(&row, ignore_cross_centers, None)
                    }));
                    let actual = catch_unwind(AssertUnwindSafe(|| {
                        zx.local_row_phase(&row, ignore_cross_centers)
                    }));
                    match (actual, expected) {
                        (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
                        (Err(_), Err(_)) => {}
                        (Ok(actual), Err(_)) => {
                            panic!("short width {width} unexpectedly returned {actual:?}")
                        }
                        (Err(_), Ok(expected)) => panic!(
                            "short width {width} unexpectedly panicked; expected {expected:?}"
                        ),
                    }
                }
            }
        }
        let empty = ZXGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: super::super::graph::CsrAdjacency::from_edges(0, &[]),
            pos_to_node: Default::default(),
            action_graph: crate::ActionDag::default(),
            total_ids: 0,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        };
        for ignore_cross_centers in [false, true] {
            let row = PauliString::new(0);
            assert_eq!(
                empty.local_row_phase_at(&row, ignore_cross_centers, None),
                Some(0)
            );
            assert_eq!(empty.local_row_phase(&row, ignore_cross_centers), Some(0));
        }
    }

    #[test]
    fn local_tensor_phases_match_signed_gallery_relations_and_products() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0x51_6e);
        for entry in dense_oracle_galleries() {
            for projection in gallery_projections(&entry.build().flatten().unwrap()) {
                let zx = ZXGraph::try_from(&projection).unwrap();
                let signed = zx.to_signed_external_generator_table();
                let reference = zx.clone();
                reference
                    .stabilizer_phase_basis
                    .set(Arc::new(super::StabilizerPhaseBasis::new(signed.clone())))
                    .unwrap();
                let mut rows = signed.clone();
                for mask in 0..(1 << signed.len().min(8)) {
                    let mut row = PhasedPauliString::positive(PauliString::new(zx.total_ids));
                    for (index, generator) in signed.iter().enumerate() {
                        let include = if signed.len() <= 8 {
                            mask & (1 << index) != 0
                        } else {
                            rng.random::<bool>()
                        };
                        if include {
                            row.multiply_assign(generator);
                        }
                    }
                    rows.push(row);
                }
                for row in rows {
                    assert_eq!(
                        zx.local_row_phase(&row.paulis, false),
                        Some(row.phase()),
                        "{entry}: {}",
                        row.paulis
                    );
                    let phase = row.phase();
                    let mut presented = row.paulis;
                    zx.reconstruct_cross_center(std::slice::from_mut(&mut presented));
                    assert_eq!(
                        zx.local_row_phase(&presented, true),
                        Some(phase),
                        "{entry}: display centers"
                    );
                }
                for _ in 0..16 {
                    let row = PauliString::from_terms(
                        zx.total_ids,
                        (0..zx.total_ids).map(|column| {
                            (
                                column,
                                [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y][rng.random_range(0..4)],
                            )
                        }),
                    );
                    assert_eq!(
                        zx.local_row_phase(&row, true).is_some(),
                        reference.contains_stabilizer_support(&row),
                        "{entry}: arbitrary support {row}"
                    );
                    assert_eq!(
                        zx.local_row_phase(&row, false).unwrap_or(0),
                        zx.row_phases_against(&[row], &signed)[0],
                        "{entry}: synthetic fallback"
                    );
                }
            }
        }
    }

    #[test]
    fn supplied_phase_certificate_remains_authoritative() {
        let zx = ZXGraph::try_from(&GalleryItem::CNOT.build().flatten().unwrap()).unwrap();
        let mut signed = zx.to_signed_external_generator_table();
        let row = signed[0].paulis.clone();
        let original = signed[0].phase();
        signed[0] = PhasedPauliString::new(row.clone(), original ^ 2);
        assert_eq!(zx.local_row_phase(&row, false), Some(original));
        zx.stabilizer_phase_basis
            .set(Arc::new(super::StabilizerPhaseBasis::new(signed)))
            .unwrap();
        assert_eq!(
            zx.stabilizer_row_phases(std::slice::from_ref(&row)),
            [original ^ 2]
        );
        let mut reconstructed = row.clone();
        zx.reconstruct_cross_center(std::slice::from_mut(&mut reconstructed));
        let expected = zx.pauli_string_to_stabilizer_with_row_phase(reconstructed, original ^ 2);
        assert_eq!(zx.materialize_stabilizer(row), expected);
    }

    #[test]
    fn yy_contraction_is_negative_on_an_ordinary_edge_and_positive_on_h() {
        for hadamard in [false, true] {
            let arrow = if hadamard { "-H>" } else { "->" };
            let graph = crate::BlockGraph::from_text(&format!(
                "BLOG 1.0\nmodule main {{\n in q: data = 0\n out r: data = 1\n \
                 0: Port [0,0,0] role=input\n 1: Port [0,0,1] role=output\n 0 {arrow} 1\n}}\n"
            ))
            .unwrap();
            let zx = ZXGraph::try_from(&graph).unwrap();
            let mut row = PhasedPauliString::positive(PauliString::new(zx.total_ids));
            for generator in zx.to_signed_external_generator_table() {
                row.multiply_assign(&generator);
            }
            assert_eq!(row.paulis.get(0), Pauli::Y);
            assert_eq!(row.paulis.get(1), Pauli::Y);
            assert_eq!(zx.local_row_phase(&row.paulis, false), Some(row.phase()));
            assert_eq!(zx.materialize_stabilizer(row.paulis).sign, !hadamard);
        }
    }

    #[test]
    fn edge_only_seam_elimination_preserves_complete_signed_relation() {
        for entry in dense_oracle_galleries() {
            for projection in gallery_projections(
                &entry
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ) {
                let zx = ZXGraph::try_from(&projection).unwrap();
                let mut expected = Vec::new();
                for (nodes, same_layer, pending) in zx.stabilizer_layers() {
                    let mut local = zx.local_stabilizer_flow_rows_for_nodes(&nodes, None);
                    super::eliminate_edge_pairs(&mut local, &same_layer);
                    expected.extend(local);
                    super::eliminate_edge_pairs(&mut expected, &pending);
                }
                let expected = super::canonical_external_basis(expected, zx.total_ids);
                assert_eq!(
                    zx.to_signed_external_generator_table(),
                    expected,
                    "{entry}: edge-only seams must retain every support and phase"
                );
            }
        }
    }

    #[test]
    fn edge_flow_multiplication_recovers_boundary_phase_factors() {
        for entry in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::T,
            GalleryItem::XMemory,
            GalleryItem::YMemory,
        ] {
            let zx = ZXGraph::try_from(
                &entry
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            )
            .unwrap();
            let nodes = zx.nodes.iter().map(|node| node.id).collect::<Vec<_>>();
            let mut sources = zx.local_stabilizer_flow_rows_for_nodes(&nodes, None);
            for node in zx.nodes.iter().filter(|node| node.kind.is_boundary()) {
                let local = zx.local_stabilizer_flow_rows_for_nodes(&[node.id], None);
                if let [left, right] = &local[..] {
                    let mut combination = left.clone();
                    combination.multiply_assign(right);
                    assert_eq!(combination.paulis.get(node.id), Pauli::Y);
                    sources.push(combination);
                }
            }
            let layout = super::EdgeFlowLayout::new(&zx);
            for left in &sources {
                for right in &sources {
                    let mut expected = left.clone();
                    expected.multiply_assign(right);
                    let edge_row = |source: &PhasedPauliString| super::EdgeFlowRow {
                        signed: PhasedPauliString::new(
                            layout.project_terms(source.paulis.iter_support()),
                            source.phase(),
                        ),
                        layout: &layout,
                    };
                    let mut actual = edge_row(left);
                    super::EliminationRow::multiply_assign(&mut actual, &edge_row(right));
                    assert_eq!(actual.signed.paulis.len(), layout.width());
                    assert_eq!(actual.materialize(), expected, "{entry}");
                }
            }
        }
    }

    #[test]
    fn projected_table_matches_eager_boundary_span_signs_and_closed_rank() {
        check_projected_tables(dense_oracle_galleries());
    }

    #[test]
    fn composed_rows_retain_complete_certificates_and_exact_local_witnesses() {
        for entry in dense_oracle_galleries() {
            for projection in gallery_projections(&entry.build().materialize_root_graph().unwrap())
            {
                let zx = ZXGraph::try_from(&projection).unwrap();
                let expected = zx.to_signed_external_generator_table();
                let table = zx
                    .projected_external_table_from_rows(
                        &free_boundary_columns(&zx),
                        expected.clone(),
                        ProjectionLimits {
                            max_frontier_width: usize::MAX,
                            max_witness_nodes: usize::MAX,
                        },
                    )
                    .unwrap()
                    .expect("complete rows belong to their local tensors");
                for row in &table.boundary_rows {
                    let full = table.materialize(row);
                    assert_eq!(
                        zx.flow_witness_replayer()(&row.flow_witness, &full.paulis),
                        Some(full),
                        "{entry}: boundary witness"
                    );
                }
                for row in &table.closed_rows {
                    assert_eq!(
                        zx.flow_witness_replayer()(&row.flow_witness, &row.signed.paulis),
                        Some(row.signed.clone()),
                        "{entry}: closed witness"
                    );
                }
                if let Some(first) = expected.first() {
                    let mut invalid = first.clone();
                    invalid.shift_phase(2);
                    assert!(
                        zx.projected_external_table_from_rows(
                            &free_boundary_columns(&zx),
                            vec![invalid],
                            ProjectionLimits {
                                max_frontier_width: usize::MAX,
                                max_witness_nodes: usize::MAX,
                            },
                        )
                        .unwrap()
                        .is_none()
                    );
                }
            }
        }
    }

    #[test]
    fn compact_certificates_match_ordered_signed_rows_and_original_witnesses() {
        for entry in dense_oracle_galleries() {
            for projection in gallery_projections(&entry.build().materialize_root_graph().unwrap())
            {
                let limits = crate::ModuleCertificationLimits::UNLIMITED;
                let space = crate::GuardedSurfaceSpace::new(
                    crate::GuardedTopology::new(&projection, limits).unwrap(),
                    &std::collections::HashMap::new(),
                    limits,
                )
                .unwrap();
                let values = space
                    .topology
                    .branch_selection_values(&rustc_hash::FxHashSet::default())
                    .unwrap()
                    .unwrap();
                let evaluate = |root| space.topology.diagram.evaluate(root, |index| values[index]);
                let zx = ZXGraph::from_block_graph_for_analysis(&projection).unwrap();
                for columns in [free_boundary_columns(&zx), Vec::new()] {
                    let raw = space.materialize_relation(&zx, evaluate);
                    let phases = zx.external_stabilizer_row_phases(&raw);
                    let signed = raw
                        .into_iter()
                        .zip(phases)
                        .map(|(row, phase)| PhasedPauliString::new(row, phase))
                        .collect();
                    let projection_limits = ProjectionLimits {
                        max_frontier_width: usize::MAX,
                        max_witness_nodes: usize::MAX,
                    };
                    let expected = zx
                        .projected_external_table_from_rows(&columns, signed, projection_limits)
                        .unwrap()
                        .unwrap();
                    let mut actual = Vec::new();
                    let boundary_len = zx
                        .certify_compact_relation(
                            space
                                .compact_relation(&zx, columns.len(), evaluate)
                                .unwrap()
                                .unwrap(),
                            &columns,
                            projection_limits,
                            |row| actual.push(row),
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(boundary_len, expected.boundary_rows.len(), "{entry}");
                    let expected = expected
                        .boundary_rows
                        .iter()
                        .map(|row| expected.materialize(row))
                        .chain(expected.closed_rows.iter().map(|row| row.signed.clone()))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual
                            .iter()
                            .map(|row| row.signed.clone())
                            .collect::<Vec<_>>(),
                        expected,
                        "{entry}: exact canonical order"
                    );
                    for row in actual {
                        assert_eq!(
                            zx.flow_witness_replayer()(&row.flow_witness, &row.signed.paulis),
                            Some(row.signed),
                            "{entry}: original witness replay"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn witness_keys_survive_xor_reuse_and_occurrence_sensitive_transforms() {
        let graph = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let mut arena = super::WitnessArena::default();
        zx.external_row_pivots(&mut arena, usize::MAX).unwrap();
        let sources = arena
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node, super::WitnessNode::Source(_)).then_some(index)
            })
            .take(3)
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 3);
        let prefix = arena.nodes.len();
        let mut keys = super::FlowSourceKeys::default();
        let first_id = arena.push_xor(sources[0], sources[1], usize::MAX).unwrap();
        let first = zx.flow_witness(&arena, first_id, &mut keys);
        arena.nodes.truncate(prefix);
        let second_id = arena.push_xor(sources[1], sources[2], usize::MAX).unwrap();
        assert_eq!(
            first_id, second_id,
            "reuse the XOR suffix, never a source slot"
        );
        let second = zx.flow_witness(&arena, second_id, &mut keys);
        for (witness, indices) in [(&first, &sources[..2]), (&second, &sources[1..])] {
            let mut expected = indices
                .iter()
                .map(|&index| {
                    let super::WitnessNode::Source(source) = &arena.nodes[index] else {
                        unreachable!("selected source")
                    };
                    zx.flow_source_key(source)
                })
                .collect::<Vec<_>>();
            expected.sort_unstable_by(super::flow_source_key_cmp);
            assert_eq!(
                witness.sources.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                expected.iter().collect::<Vec<_>>()
            );
        }
        let original = first.sources[0].as_ref().clone();
        let other = arena
            .nodes
            .iter()
            .filter_map(|node| match node {
                super::WitnessNode::Source(source) => Some(zx.flow_source_key(source)),
                _ => None,
            })
            .find(|key| key.position != original.position)
            .expect("the later callback target must have a different position");
        drop(keys);
        drop(arena);
        assert_eq!(
            first
                .transformed(crate::ModuleOrientation::IDENTITY, Ok)
                .unwrap()
                .sources,
            first.sources,
            "witness keys outlive their cache and arena"
        );

        let shared = Arc::clone(&first.sources[0]);
        let one = FlowWitness {
            sources: vec![Arc::clone(&shared)],
        };
        let two = FlowWitness {
            sources: vec![shared, Arc::new(other)],
        };
        let calls = std::cell::RefCell::new(Vec::new());
        let transform = |position| {
            let mut calls = calls.borrow_mut();
            calls.push(position);
            if calls.len() == 2 {
                crate::checked_add_position(IVec3::splat(i32::MAX), IVec3::ONE)
            } else {
                crate::checked_add_position(position, IVec3::X)
            }
        };
        let moved = one
            .transformed(crate::ModuleOrientation::IDENTITY, transform)
            .unwrap();
        assert_eq!(moved.sources[0].position, original.position + IVec3::X);
        assert!(matches!(
            two.transformed(crate::ModuleOrientation::IDENTITY, transform),
            Err(crate::BlockGraphError::CoordinateOverflow { .. })
        ));
        assert_eq!(*calls.borrow(), [original.position, original.position]);
        assert_eq!(
            *one.sources[0], original,
            "shared input keys remain unchanged"
        );
    }

    #[test]
    fn operation_local_replay_keeps_graph_placement_frames_and_fallbacks() {
        for entry in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::TWithPreparedY,
        ] {
            let original = entry.build().materialize_root_graph().unwrap();
            let turns = match entry {
                GalleryItem::TWithPreparedY => 2,
                _ => 1,
            };
            for graph in [
                original.clone(),
                original
                    .rotate_about_origin_lenient(crate::UDirection::Z, turns)
                    .unwrap()
                    .shift_positions(IVec3::new(20, -30, 40))
                    .unwrap(),
            ] {
                let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
                let signed = zx.to_signed_external_generator_table();
                let table = zx
                    .projected_external_table_from_rows(
                        &free_boundary_columns(&zx),
                        signed,
                        ProjectionLimits {
                            max_frontier_width: usize::MAX,
                            max_witness_nodes: usize::MAX,
                        },
                    )
                    .unwrap()
                    .unwrap();
                let rows = table
                    .boundary_rows
                    .iter()
                    .map(|row| (row.flow_witness.clone(), table.materialize(row)))
                    .chain(
                        table
                            .closed_rows
                            .iter()
                            .map(|row| (row.flow_witness.clone(), row.signed.clone())),
                    )
                    .collect::<Vec<_>>();
                let mut replay = zx.flow_witness_replayer();
                let local = zx
                    .nodes()
                    .iter()
                    .map(|node| zx.local_signed_flow_supports(&[node.id], None))
                    .find(|sources| sources.len() > 1)
                    .expect("fixture has multiple local ordinals");
                let last = local.last().unwrap();
                for source in [last, &local[0], last, &local[0]] {
                    let witness = FlowWitness {
                        sources: vec![Arc::new(zx.flow_source_key(source))],
                    };
                    let expected = PhasedPauliString::new(
                        PauliString::from_terms(zx.total_ids(), source.terms.iter().copied()),
                        source.phase,
                    );
                    assert_eq!(replay(&witness, &expected.paulis), Some(expected));
                }
                for (witness, expected) in rows.iter().chain(rows.iter().rev()) {
                    assert_eq!(replay(witness, &expected.paulis), Some(expected.clone()));
                    // Cached ingredients must still check this call's expected support.
                    let identity = PhasedPauliString::positive(PauliString::new(zx.total_ids));
                    assert_eq!(replay(witness, &identity.paulis), Some(identity));
                    let mut renamed = witness.clone();
                    if !renamed.sources.is_empty() {
                        // An unrecognized local label must still use the existing support fallback.
                        let mut changed = renamed.sources[0].as_ref().clone();
                        changed.intrinsic_phase ^= 1;
                        renamed.sources[0] = Arc::new(changed);
                        assert_eq!(replay(&renamed, &expected.paulis), Some(expected.clone()));
                        // A missing physical owner cannot become valid through cached keys.
                        let mut changed = renamed.sources[0].as_ref().clone();
                        changed.position += IVec3::splat(1_000_000);
                        renamed.sources[0] = Arc::new(changed);
                        assert!(replay(&renamed, &expected.paulis).is_none());
                    }
                    assert_eq!(replay(witness, &expected.paulis), Some(expected.clone()));
                }
            }
        }
    }

    #[test]
    fn compact_certificate_rejects_unsupported_original_support() {
        let graph = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let limits = ProjectionLimits {
            max_frontier_width: usize::MAX,
            max_witness_nodes: usize::MAX,
        };
        let width = 2 * zx.total_ids();
        let malformed = super::CompactSurfaceRelation {
            rows: vec![super::super::CoeffVec::singleton(2 * zx.edges[0].id, width)],
            queries: (0..zx.total_ids())
                .map(|column| [Some(2 * column), Some(2 * column + 1)])
                .collect(),
            width,
        };
        let result = zx
            .certify_compact_relation(malformed, &[], limits, |_| {
                panic!("invalid half-edge support must not be certified")
            })
            .unwrap();
        assert_eq!(result, None);

        let space = crate::GuardedSurfaceSpace::new(
            crate::GuardedTopology::new(&graph, crate::ModuleCertificationLimits::UNLIMITED)
                .unwrap(),
            &std::collections::HashMap::new(),
            crate::ModuleCertificationLimits::UNLIMITED,
        )
        .unwrap();
        let compact = space.compact_relation(&zx, 0, |_| true).unwrap().unwrap();
        let cancellation = crate::CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            cancellation.scope(
                || zx.certify_compact_relation(compact, &[], limits, |_| panic!(
                    "cancelled certificate must not be emitted"
                ),)
            ),
            Err(super::ProjectionError::Cancelled(_))
        ));
        let compact = space.compact_relation(&zx, 0, |_| true).unwrap().unwrap();
        let cancellation = crate::CancellationToken::new();
        let mut consumed = 0;
        assert!(matches!(
            cancellation.scope(|| zx.certify_compact_relation(compact, &[], limits, |_| {
                consumed += 1;
                cancellation.cancel();
            })),
            Err(super::ProjectionError::Cancelled(_))
        ));
        assert_eq!(
            consumed, 1,
            "the next row checks cancellation with a warm cache"
        );
        assert!(matches!(
            zx.certify_compact_relation(
                space.compact_relation(&zx, 0, |_| true).unwrap().unwrap(),
                &[],
                ProjectionLimits {
                    max_witness_nodes: 0,
                    ..limits
                },
                |_| panic!("resource failure must not emit a certificate"),
            ),
            Err(super::ProjectionError::WitnessLimit { limit: 0, .. })
        ));
        zx.stabilizer_phase_basis
            .set(std::sync::Arc::new(super::StabilizerPhaseBasis::new(
                zx.to_signed_external_generator_table(),
            )))
            .unwrap();
        assert!(
            space.compact_relation(&zx, 0, |_| true).unwrap().is_none(),
            "an installed signed authority cannot enter the unsigned path"
        );
    }

    /// The external table splits as `rank = b + z`: one independent surface per
    /// free boundary node, plus the closed surfaces that touch no boundary.
    ///
    /// Equivalently, restricting the table to the free boundary columns has rank
    /// exactly `b` — the boundary data of the correlation surfaces is a full
    /// stabilizer group on the `b` open legs. The closed part `z` is *not* the
    /// cycle rank: `ccz_4x3x6` has eight independent cycles and no closed
    /// surface at all.
    ///
    /// Do not weaken this to `rank == b + beta_1`: that identity is false once
    /// actually exercised, and looks true only if open and dynamic graphs are
    /// skipped, which leaves nothing to check at all.
    fn check_projected_tables(entries: impl IntoIterator<Item = GalleryItem>) {
        let mut checked = 0;
        for entry in entries {
            for projection in gallery_projections(
                &entry
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ) {
                let zx = ZXGraph::try_from(&projection).unwrap();
                let columns = free_boundary_columns(&zx);
                let eager_signed = zx.to_signed_external_generator_table();
                let eager = eager_signed
                    .iter()
                    .map(|row| row.paulis.clone())
                    .collect::<Vec<_>>();
                assert_rows_satisfy_external_edge_pair_support(&zx, &eager);
                assert!(eager.iter().all(|row| row.weight() > 0), "{entry:?}");
                assert_eq!(
                    boundary_projection_rank(&zx, &eager),
                    columns.len(),
                    "boundary projection rank for {entry:?}"
                );
                assert!(
                    eager.len() >= columns.len(),
                    "external rank below the boundary count for {entry:?}"
                );

                // A boundary-free component always contributes at least one
                // closed surface. The TELS factory also has one with every
                // component bounded: its deterministic postselection parity.
                let closed = eager.len() - columns.len();
                let expected_closed = match entry {
                    GalleryItem::XMemory
                    | GalleryItem::YMemory
                    | GalleryItem::Stability
                    | GalleryItem::CCZFactoryWithTels => 1,
                    _ => 0,
                };
                assert_eq!(closed, expected_closed, "closed surfaces for {entry:?}");
                assert!(
                    closed >= boundaryless_component_count(&zx),
                    "a boundary-free component must carry a closed surface: {entry:?}"
                );
                let projected = zx
                    .projected_external_table(
                        &columns,
                        ProjectionLimits {
                            max_frontier_width: usize::MAX,
                            max_witness_nodes: usize::MAX,
                        },
                    )
                    .unwrap();

                assert_eq!(projected.boundary_rows.len(), columns.len(), "{entry:?}");
                assert_eq!(
                    projected.closed_rows.len(),
                    eager.len() - columns.len(),
                    "{entry:?}"
                );
                for row in projected.boundary_rows.iter() {
                    let full = projected.materialize(row);
                    assert_eq!(
                        compact_projection(&full.paulis, &columns),
                        row.signed.paulis
                    );
                    assert_eq!(
                        zx.row_phases_against(std::slice::from_ref(&full.paulis), &eager_signed)[0],
                        row.signed.phase(),
                        "{entry:?}"
                    );
                }
                for row in &projected.closed_rows {
                    assert_eq!(compact_projection(&row.signed.paulis, &columns).weight(), 0);
                    assert_eq!(
                        zx.row_phases_against(
                            std::slice::from_ref(&row.signed.paulis),
                            &eager_signed,
                        )[0],
                        row.signed.phase(),
                        "{entry:?}"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "no gallery entry exercised the rank identity");
    }

    #[test]
    #[ignore = "ten-bit dense ZX oracle stress; run via just test-full"]
    fn ten_bit_adder_tables_match_eager_oracles() {
        check_projected_tables([GalleryItem::TenBitAdder]);
    }
}
