//! Sparse modular elimination and flow-witness materialization.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use bloq_utils::{Direction, Pauli, PauliString, PhasedPauliString};
use glam::IVec3;
use rustc_hash::FxHashSet;

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
    flow_basis: OnceLock<FlowWitnessBasis>,
}

#[derive(Debug, Clone)]
struct TemporalDenseRow<'a> {
    projection: PauliString,
    flow: EdgeFlowRow<'a>,
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

    fn project(&self, row: &PauliString) -> PauliString {
        self.project_terms(row.iter_support())
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
            phase = (phase
                + match (lhs, rhs) {
                    (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::X) => 1,
                    (Pauli::Y, Pauli::X) | (Pauli::Z, Pauli::Y) | (Pauli::X, Pauli::Z) => 3,
                    _ => 0,
                })
                & 3;
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

impl EliminationRow for TemporalDenseRow<'_> {
    fn multiply_assign(&mut self, pivot: &Self) {
        self.projection ^= &pivot.projection;
        self.flow.multiply_assign(&pivot.flow);
    }
}

impl ProjectedExternalTable {
    pub(crate) fn materialize(&self, row: &ProjectedRow) -> PhasedPauliString {
        self.arena.materialize(row.witness, self.width)
    }

    /// The canonical witnessed basis of this table.
    ///
    /// Building it is independent of any row being reduced, so callers that
    /// classify many rows against one table build it once and reuse it.
    pub(crate) fn flow_witness_basis(&self) -> &FlowWitnessBasis {
        self.flow_basis.get_or_init(|| {
            let basis = self
                .boundary_rows
                .iter()
                .map(|candidate| WitnessedRow {
                    signed: self.materialize(candidate),
                    flow_witness: candidate.flow_witness.clone(),
                })
                .chain(self.closed_rows.iter().cloned())
                .collect::<Vec<_>>();
            FlowWitnessBasis {
                rows: canonical_witnessed_external_basis(basis, self.width),
            }
        })
    }
}

/// A reduced basis over one [`ProjectedExternalTable`], reusable across rows.
#[derive(Debug, Clone)]
pub(crate) struct FlowWitnessBasis {
    rows: Vec<WitnessedRow>,
}

impl FlowWitnessBasis {
    pub(crate) fn to_external_basis(&self) -> Vec<PhasedPauliString> {
        self.rows.iter().map(|row| row.signed.clone()).collect()
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct FlowSourceKey {
    position: IVec3,
    node: Pauli,
    edges: Vec<(Direction, Pauli)>,
    intrinsic_phase: u8,
}

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
            row.multiply_assign(&PhasedPauliString::new(
                PauliString::from_terms(width, source.terms.iter().copied()),
                source.phase,
            ));
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
    pub(crate) fn to_temporal_signed_external_generator_table(
        &self,
        semantic_columns: &[usize],
        node_gates: &[Option<i64>],
    ) -> Vec<PhasedPauliString> {
        assert_eq!(node_gates.len(), self.nodes.len());
        let edge_layout = EdgeFlowLayout::new(self);
        let semantic = semantic_columns.iter().copied().collect::<FxHashSet<_>>();

        let effective_node_gates = self
            .nodes
            .iter()
            .map(|node| {
                node_gates[node.id]
                    .or_else(|| {
                        self.neighbor_ids(node.id)
                            .iter()
                            .filter_map(|&neighbor| node_gates[neighbor])
                            .max()
                    })
                    .unwrap_or(i64::MAX)
            })
            .collect::<Vec<_>>();
        let mut layers = BTreeMap::<i64, (Vec<usize>, Vec<EdgePair>)>::new();
        for node in &self.nodes {
            layers
                .entry(effective_node_gates[node.id])
                .or_default()
                .0
                .push(node.id);
        }
        let mut unresolved_semantic_edges = FxHashSet::default();
        for edge in self.edges.iter().filter(|edge| edge.n1 < edge.n2) {
            let pair = self.edge_column_pair(edge);
            if semantic.contains(&pair.0) {
                unresolved_semantic_edges.insert(pair.0);
            }
            if semantic.contains(&pair.1) {
                unresolved_semantic_edges.insert(pair.1);
            }
            let gate = effective_node_gates[edge.n1].max(effective_node_gates[edge.n2]);
            layers.entry(gate).or_default().1.push(pair);
        }
        for (nodes, pairs) in layers.values_mut() {
            nodes.sort_unstable();
            pairs.sort_unstable();
        }

        let mut live = Vec::new();
        let mut physical_columns = BTreeSet::new();
        let mut released = Vec::new();
        for (_, (node_ids, pairs)) in layers {
            for &node_id in &node_ids {
                if self.neighbor_ids(node_id).is_empty() && !semantic.contains(&node_id) {
                    physical_columns.insert(node_id);
                }
                for (_, column) in self.neighbor_edges(node_id) {
                    if !semantic.contains(&column) || unresolved_semantic_edges.contains(&column) {
                        physical_columns.insert(column);
                    }
                }
            }
            live.extend(
                self.local_stabilizer_flow_rows_for_nodes(&node_ids, None)
                    .into_iter()
                    .map(|signed| TemporalDenseRow {
                        projection: PauliString::from_terms(
                            self.total_ids,
                            signed.paulis.iter_support().filter(|&(column, _)| {
                                column >= self.nodes.len()
                                    || semantic.contains(&column)
                                    || self.neighbor_ids(column).is_empty()
                            }),
                        ),
                        flow: EdgeFlowRow {
                            signed: PhasedPauliString::new(
                                edge_layout.project(&signed.paulis),
                                signed.phase(),
                            ),
                            layout: &edge_layout,
                        },
                    }),
            );
            let eliminated = signed_gaussian_elimination(
                &mut live,
                pairs
                    .iter()
                    .flat_map(|&(left, right)| [(left, right, Pauli::X), (left, right, Pauli::Z)]),
                |row, &(left, right, axis)| {
                    (row.projection.get(left) ^ row.projection.get(right)) & axis
                },
            );
            live.drain(..eliminated);
            for &(left, right) in &pairs {
                unresolved_semantic_edges.remove(&left);
                unresolved_semantic_edges.remove(&right);
                physical_columns.remove(&left);
                physical_columns.remove(&right);
            }
            let discharged = node_ids
                .iter()
                .copied()
                .chain(pairs.iter().flat_map(|&(left, right)| [left, right]))
                .filter(|column| !semantic.contains(column))
                .collect::<Vec<_>>();
            for &column in &discharged {
                physical_columns.remove(&column);
            }
            for row in &mut live {
                for &column in &discharged {
                    row.projection.set(column, Pauli::I);
                }
            }
            let physical_rank = signed_gaussian_elimination(
                &mut live,
                physical_columns
                    .iter()
                    .copied()
                    .flat_map(|column| [Pauli::X, Pauli::Z].map(|axis| (column, axis))),
                |row, &(column, axis)| row.projection.get(column) & axis,
            );

            for row in live.drain(physical_rank..) {
                released.push(row.flow.materialize());
            }
        }
        assert!(
            live.is_empty(),
            "every temporal row closes at the final gate"
        );

        canonical_external_basis(released, self.total_ids)
    }

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
        let mut rows = Vec::new();
        let mut closed_candidates = Vec::new();
        let mut max_frontier_width = 0;
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
            update_frontier(&rows, &mut max_frontier_width, limits.max_frontier_width)?;
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
                self.total_ids,
            );
        }

        self.finish_projected_table(rows, closed_candidates, arena, boundary_columns, limits)
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
        let mut pivots = BTreeMap::new();
        for node in &self.nodes {
            let mut local = self.projected_source_rows(
                &[node.id],
                &FxHashSet::from_iter([node.id]),
                &mut arena,
                limits.max_witness_nodes,
            )?;
            let columns = local
                .iter()
                .flat_map(|row| row.paulis.keys().copied())
                .collect::<BTreeSet<_>>();
            let rank = projected_gaussian_elimination(
                &mut local,
                axis_pivot_constraints(columns),
                |row, &(column, axis)| row.get(column) & axis,
                &mut arena,
                limits.max_witness_nodes,
            )?;
            for row in local.into_iter().take(rank) {
                let (&column, &pauli) = row.paulis.first_key_value().expect("local pivot");
                pivots.insert((column, pauli & Pauli::X), row);
            }
        }
        let mut rows = Vec::with_capacity(complete.len());
        for mut full in complete {
            self.clear_cross_centers(std::slice::from_mut(&mut full.paulis));
            if !self.row_has_external_edge_pair_support(&full.paulis) {
                return Ok(None);
            }
            let mut residual = full.paulis.iter_support().collect::<BTreeMap<_, _>>();
            let mut witness = None;
            while let Some((&column, &pauli)) = residual.first_key_value() {
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
                    Some(previous) => {
                        arena.push_xor(previous, pivot.witness, limits.max_witness_nodes)?
                    }
                });
            }
            let Some(witness) = witness else {
                if full.phase() != 0 {
                    return Ok(None);
                }
                continue;
            };
            if arena.materialize(witness, self.total_ids) != full {
                return Ok(None);
            }
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
        self.finish_projected_table(rows, Vec::new(), arena, boundary_columns, limits)
            .map(Some)
    }

    fn finish_projected_table(
        &self,
        mut rows: Vec<SparseRow>,
        mut closed_candidates: Vec<WitnessedRow>,
        mut arena: WitnessArena,
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
            let flow_witness = self.flow_witness(&arena, row.witness);
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
            self.total_ids,
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
            flow_basis: OnceLock::new(),
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

    fn flow_witness(&self, arena: &WitnessArena, witness: usize) -> FlowWitness {
        let mut sources = arena
            .source_indices(witness)
            .into_iter()
            .map(|index| {
                let WitnessNode::Source(source) = &arena.nodes[index] else {
                    unreachable!("source set contains only source nodes")
                };
                Arc::new(self.flow_source_key(source))
            })
            .collect::<Vec<_>>();
        sources.sort_unstable_by(|left, right| flow_source_key_cmp(left, right));
        FlowWitness { sources }
    }

    pub(crate) fn materialize_flow_witness(
        &self,
        witness: &FlowWitness,
        expected: &PauliString,
    ) -> Option<PhasedPauliString> {
        let exact_sources = witness
            .sources
            .iter()
            .map(|key| {
                let node = self.node_at(key.position)?;
                let (ordinal, source) = self
                    .local_signed_flow_supports(&[node.id], None)
                    .into_iter()
                    .enumerate()
                    .find(|(_, source)| self.flow_source_key(source) == *key.as_ref())?;
                Some((
                    (node.pos.z, node.id, ordinal),
                    PhasedPauliString::new(
                        PauliString::from_terms(self.total_ids, source.terms),
                        source.phase,
                    ),
                ))
            })
            .collect::<Option<Vec<_>>>();
        if let Some(mut sources) = exact_sources {
            sources.sort_unstable_by_key(|(key, _)| *key);
            let mut row = PhasedPauliString::positive(PauliString::new(self.total_ids));
            for (_, source) in sources {
                row.multiply_assign(&source);
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

    pub(crate) fn to_external_generator_table(&self) -> Vec<PauliString> {
        self.to_signed_external_generator_table()
            .into_iter()
            .map(|row| row.paulis)
            .collect()
    }

    pub(crate) fn to_external_generator_table_with_signed(
        &self,
    ) -> (Vec<PauliString>, Vec<PhasedPauliString>) {
        let signed = self.to_signed_external_generator_table();
        let rows = signed.iter().map(|row| row.paulis.clone()).collect();
        (rows, signed)
    }

    /// Recover a complete tensor row's phase before the YY edge contractions.
    /// Half-edges use the smaller endpoint's frame; spider crossing centers
    /// may be display support rather than independent tensor factors.
    fn local_row_phase(&self, row: &PauliString, ignore_cross_centers: bool) -> Option<u8> {
        if (row.len() > self.total_ids
            && row
                .iter_support()
                .any(|(column, _)| column >= self.total_ids))
            || !self.row_has_external_edge_pair_support(row)
        {
            return None;
        }
        let mut sign = false;
        for node in &self.nodes {
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

    fn to_signed_external_generator_table(&self) -> Vec<PhasedPauliString> {
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
    width: usize,
) {
    let mut live = Vec::with_capacity(rows.len());
    for row in rows.drain(..) {
        if row.paulis.is_empty() {
            let signed = arena.materialize(row.witness, width);
            if !signed.paulis.is_identity() {
                closed.push(WitnessedRow {
                    signed,
                    flow_witness: zx.flow_witness(arena, row.witness),
                });
            }
        } else {
            live.push(row);
        }
    }
    *rows = live;
}

fn update_frontier(
    rows: &[SparseRow],
    max_frontier_width: &mut usize,
    limit: usize,
) -> Result<(), ProjectionError> {
    let width = rows
        .iter()
        .flat_map(|row| row.paulis.keys().copied())
        .collect::<FxHashSet<_>>()
        .len();
    *max_frontier_width = (*max_frontier_width).max(width);
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
/// whatever travels with the Pauli row — a flow witness, positional support —
/// so everything folds in one pass.
pub(crate) trait EliminationRow {
    fn multiply_assign(&mut self, pivot: &Self);
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
    mut solved: usize,
    pivot_constraints: impl IntoIterator<Item = T>,
    predicate: impl Fn(&R, &T) -> bool,
) -> usize {
    debug_assert!(solved <= rows.len());
    for constraint in pivot_constraints {
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
        // The pivot search already proved the skipped unsolved rows zero.
        // Swapping the pivot with `solved` leaves that whole prefix zero.
        for row in before.iter_mut().chain(&mut after[pivot_index - solved..]) {
            if predicate(row, &constraint) {
                row.multiply_assign(pivot);
            }
        }
        solved += 1;
        if solved == rows.len() {
            break;
        }
    }
    solved
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
    use super::super::stabilizer::test_support::build_t_selective_graph;
    use super::super::stabilizer::{axis_pivot_constraints, gaussian_elimination_with_tracking};
    use super::{FlowSourceKey, FlowWitness, ProjectionLimits};
    use crate::GalleryItem;

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
    fn flow_witness_xor_preserves_sorted_shared_keys() {
        let source = |x| {
            Arc::new(FlowSourceKey {
                position: IVec3::new(x, 0, 0),
                node: Pauli::X,
                edges: Vec::new(),
                intrinsic_phase: 0,
            })
        };
        let [first, shared, last] = [0, 1, 2].map(source);
        let mut witness = FlowWitness {
            sources: vec![Arc::clone(&first), Arc::clone(&shared)],
        };

        witness.xor_assign(&FlowWitness {
            sources: vec![shared, Arc::clone(&last)],
        });

        assert_eq!(witness.sources.len(), 2);
        assert!(Arc::ptr_eq(&witness.sources[0], &first));
        assert!(Arc::ptr_eq(&witness.sources[1], &last));

        let mut append = FlowWitness {
            sources: Vec::with_capacity(8),
        };
        append.sources.push(Arc::clone(&first));
        let allocation = append.sources.as_ptr();
        append.xor_assign(&FlowWitness::default());
        append.xor_assign(&FlowWitness {
            sources: vec![Arc::clone(&last)],
        });
        assert_eq!(append.sources.as_ptr(), allocation);
        assert_eq!(append.sources, [Arc::clone(&first), Arc::clone(&last)]);

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

    #[test]
    fn packed_pivots_preserve_column_then_axis_order() {
        for width in [0, 1, 63, 64, 65, 127, 128, 129, 1025] {
            let mut row = PauliString::new(width);
            assert_eq!(super::first_component(&row), None);
            for column in (0..width).rev() {
                for pauli in [Pauli::Z, Pauli::X, Pauli::Y, Pauli::I] {
                    row.set(column, pauli);
                    let scalar = (0..width)
                        .flat_map(|column| [(column, Pauli::X), (column, Pauli::Z)])
                        .find(|&(column, axis)| row.get(column) & axis);
                    assert_eq!(super::first_component(&row), scalar);
                }
                row.set(column, Pauli::Y);
            }
        }
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
        assert_eq!(zx.stabilizer_row_phases(&[row]), [original ^ 2]);
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
            assert_eq!(zx.pauli_string_to_stabilizer(row.paulis).sign, !hadamard);
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
                            layout.project(&source.paulis),
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
                assert_eq!(
                    table.flow_witness_basis().to_external_basis(),
                    expected,
                    "{entry}"
                );
                for row in &table.boundary_rows {
                    let full = table.materialize(row);
                    assert_eq!(
                        zx.materialize_flow_witness(&row.flow_witness, &full.paulis),
                        Some(full),
                        "{entry}: boundary witness"
                    );
                }
                for row in &table.closed_rows {
                    assert_eq!(
                        zx.materialize_flow_witness(&row.flow_witness, &row.signed.paulis),
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

                assert_eq!(
                    projected.flow_witness_basis().to_external_basis(),
                    eager_signed,
                    "{entry:?}"
                );
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
    fn temporal_external_table_matches_exact_eager_table() {
        check_temporal_tables(dense_oracle_galleries());

        let graph = crate::parse_blog_to_graph(
            "BLOG 1.0\n\n0: Port [0, 0, 0]\n1: Port [0, 0, 1]\n0 -> +Z\n",
        )
        .unwrap();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let actual = zx.to_temporal_signed_external_generator_table(&[0, 1], &[None, None]);
        let expected = zx.to_signed_external_generator_table();
        assert_eq!(
            actual
                .iter()
                .map(|row| (&row.paulis, row.phase()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|row| (&row.paulis, row.phase()))
                .collect::<Vec<_>>(),
        );
    }

    fn check_temporal_tables(entries: impl IntoIterator<Item = GalleryItem>) {
        for entry in entries {
            for projection in gallery_projections(
                &entry
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ) {
                let zx = ZXGraph::try_from(&projection).unwrap();
                let mut semantic = free_boundary_columns(&zx);
                semantic.extend(zx.measurement_columns().into_values());
                semantic.extend(zx.nodes.iter().filter_map(|node| {
                    matches!(node.kind, super::NodeKind::Selective(_)).then_some(node.id)
                }));
                semantic.sort_unstable();
                semantic.dedup();
                let node_gates = zx
                    .nodes
                    .iter()
                    .map(|node| {
                        (!node.is_output_port(&zx)).then_some(
                            if matches!(node.kind, super::NodeKind::Selective(_)) {
                                i64::from(node.pos.z)
                            } else {
                                i64::from(node.pos.z) + 1
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let actual = zx.to_temporal_signed_external_generator_table(&semantic, &node_gates);
                let expected = zx.to_signed_external_generator_table();

                assert_eq!(
                    actual
                        .iter()
                        .map(|row| (&row.paulis, row.phase()))
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|row| (&row.paulis, row.phase()))
                        .collect::<Vec<_>>(),
                    "{entry:?}",
                );
            }
        }
    }

    #[test]
    #[ignore = "ten-bit dense ZX oracle stress; run via just test-full"]
    fn ten_bit_adder_tables_match_eager_oracles() {
        check_projected_tables([GalleryItem::TenBitAdder]);
        check_temporal_tables([GalleryItem::TenBitAdder]);
    }

    #[test]
    fn modular_stabilizer_path_computes_representative_measurement_and_selective_graphs() {
        for graph in [
            GalleryItem::THTH
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            GalleryItem::CCZFactoryWithTels
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            build_t_selective_graph(),
        ] {
            let zx = ZXGraph::try_from(&graph).unwrap();
            let table = zx.to_stabilizer_table().unwrap();

            assert_eq!(table.basis.rows.len(), table.basis.kinds.len());
            assert!(table.basis.rows.iter().all(|row| row.weight() > 0));
            assert!(
                table
                    .basis
                    .rows
                    .iter()
                    .all(|row| zx.row_has_external_edge_pair_support(row))
            );
        }
    }
}
