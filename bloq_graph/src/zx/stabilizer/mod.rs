//! Stabilizer rows of ZX graphs: their materialized surfaces, automatic port
//! filling, and the Pauli-string linear algebra behind them.

mod basis;
mod selective;

use super::ZXGraph;
use crate::{
    Action, Block, BlockGraph, BlockGraphError, BlockKind, CubeKind, Direction, FeedbackTarget,
    ModuleCertificationLimits, Pipe, SelectiveKind,
};
use bloq_utils::{Basis, Pauli, PauliString, UDirection};
use glam::IVec3;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap, hash_map::Entry};
use thiserror::Error;

#[cfg(feature = "verify")]
pub(crate) use basis::solve_coeff_combination;
pub(crate) use basis::{
    CoeffVec, axis_pivot_constraints, gaussian_elimination_with_tracking, reduce_to_basis,
};
pub(crate) use selective::{collect_selective_constraints, normalize_selective_support_in_place};

/// An error from filling a graph's open ports with concrete boundary spiders.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum FillPortsError {
    /// No open port exists to anchor a minimal fill.
    #[error("there are no open ports to fill for minimal simulation")]
    NoOpenPorts,
    /// A port does not have exactly one pipe.
    #[error("port at {pos} requires exactly one pipe to infer a fill, got {degree}")]
    PortFillRequiresSinglePipe {
        /// Port position.
        pos: IVec3,
        /// Observed degree.
        degree: usize,
    },
    /// The cube kind for a filled port cannot be inferred.
    #[error("cannot infer a cube kind for filled port at {pos}")]
    InvalidFilledPortCube {
        /// Port position.
        pos: IVec3,
        /// Underlying block error.
        #[source]
        source: crate::BlockError,
    },
    /// An action control is unavailable in the selected fill.
    #[error("action {ordinal} depends on a measurement outside the selected static fill")]
    UnavailableControl {
        /// Source action ordinal.
        ordinal: usize,
    },
    /// An external stabilizer contains an invalid character.
    #[error("invalid external stabilizer character {ch}")]
    InvalidExternalStabilizerChar {
        /// Invalid character.
        ch: char,
    },
    /// An external stabilizer requests an unsupported port basis.
    #[error("unsupported external stabilizer basis {basis} at port {pos}")]
    UnsupportedPortBasis {
        /// Port position.
        pos: IVec3,
        /// Unsupported basis.
        basis: Pauli,
    },
}

/// An error from deriving stabilizer generators or measurement surfaces.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum StabilizerError {
    /// The caller cancelled this analysis at a safe checkpoint.
    #[error("{0}")]
    Cancelled(#[from] crate::ComputationCancelled),
    /// Stabilizer derivation exceeded a resource limit.
    #[error(
        "stabilizer derivation exceeded {phase} limit: {observed} > {limit}; {help}",
        help = ModuleCertificationLimits::RESOURCE_LIMIT_HELP
    )]
    ResourceLimited {
        /// Derivation phase.
        phase: &'static str,
        /// Observed resource use.
        observed: usize,
        /// Configured limit.
        limit: usize,
    },
    /// Selective support cannot be normalized.
    #[error("selective support at {pos} for {kind} cannot be normalized")]
    SelectiveSupportUnsatisfiable {
        /// Selective position.
        pos: IVec3,
        /// Selective kind.
        kind: SelectiveKind,
    },
    /// A control parity is unavailable before its deadline.
    #[error("control parity for mvar '{mvar}' is not available at deadline {deadline}")]
    UnavailableControlParity {
        /// Measurement variable.
        mvar: String,
        /// Required deadline.
        deadline: i64,
    },
    /// No valid measurement surface exists.
    #[error("measurement surface for '{mvar}' is not available at any valid deadline")]
    MeasurementSurfaceUnavailable {
        /// Measurement variable.
        mvar: String,
    },
    /// A feedback target has no compatible wire.
    #[error("feedback target {target} has no wire on which to apply {pauli}")]
    FeedbackTargetWithoutWire {
        /// Target position.
        target: IVec3,
        /// Feedback Pauli.
        pauli: bloq_utils::PauliBasis,
    },
    /// No deterministic presentation satisfies the composed certificate.
    #[error("composed module certificate has no admissible deterministic presentation")]
    ComposedPresentationUnsatisfied,
}

impl StabilizerError {}

pub(crate) struct SearchBudget {
    visited: usize,
    limits: ModuleCertificationLimits,
    cancellation: Option<crate::CancellationToken>,
}

impl SearchBudget {
    #[cfg(test)]
    pub(crate) fn new(limit: usize) -> Self {
        Self::with_limits(ModuleCertificationLimits {
            max_normalization_states: limit,
            ..ModuleCertificationLimits::UNLIMITED
        })
    }

    pub(crate) fn with_limits(limits: ModuleCertificationLimits) -> Self {
        Self {
            visited: 0,
            limits,
            cancellation: crate::CancellationToken::current(),
        }
    }

    pub(crate) fn visit(&mut self, phase: &'static str) -> Result<(), StabilizerError> {
        let next = self.visited.checked_add(1);
        self.visited = next.unwrap_or(usize::MAX);
        self.check_cancellation()?;
        let limit = self.limits.max_normalization_states;
        if next.is_none() || self.visited > limit {
            Err(StabilizerError::ResourceLimited {
                phase,
                observed: self.visited,
                limit,
            })
        } else {
            Ok(())
        }
    }

    /// Pauli strings own two arrays rounded to binar's 512-bit blocks.
    fn pauli_words(width: usize) -> u128 {
        width.div_ceil(512) as u128 * 16
    }

    fn check_words(&self, words: u128) -> Result<(), StabilizerError> {
        self.check_cancellation()?;
        let limit = self.limits.max_matrix_words;
        if words > limit as u128 {
            return Err(StabilizerError::ResourceLimited {
                phase: "dense matrix words",
                observed: usize::try_from(words).unwrap_or(usize::MAX),
                limit,
            });
        }
        Ok(())
    }

    fn check_cancellation(&self) -> Result<(), StabilizerError> {
        if let Some(token) = &self.cancellation {
            token.check()?;
        }
        Ok(())
    }

    pub(crate) fn check_matrix(
        &self,
        pauli_rows: usize,
        pauli_width: usize,
        coefficient_rows: usize,
        coefficient_width: usize,
    ) -> Result<(), StabilizerError> {
        self.check_words(
            pauli_rows as u128 * Self::pauli_words(pauli_width)
                + coefficient_rows as u128 * coefficient_width.div_ceil(64) as u128,
        )
    }
}

/// A stabilizer of a ZX graph, with its Pauli support broken out by the graph
/// elements (ports, interior nodes, interior edges) it acts on for rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct Stabilizer {
    /// The stabilizer as a dense Pauli string over the graph's id space.
    pub paulis: PauliString,
    /// Constant sign (`true` means `-1`).
    pub sign: bool,
    /// Pauli acting at each boundary port position.
    pub port_stabilizer: FxHashMap<IVec3, Pauli>,
    /// Pauli acting at each interior node position.
    pub interior_nodes: FxHashMap<IVec3, Pauli>,
    /// Pauli acting on each interior edge, keyed by its endpoint positions.
    pub interior_edges: FxHashMap<(IVec3, IVec3), Pauli>,
}

impl Stabilizer {
    /// XORs this stabilizer with `other`, ignoring Pauli-product phase.
    ///
    /// Shared interior edges must use the same endpoint order.
    pub fn phase_free_product(&self, other: &Stabilizer) -> Stabilizer {
        let mut product = self.clone();
        product.phase_free_mul_assign(other);
        product
    }

    /// XORs `other` into this stabilizer, ignoring Pauli-product phase.
    ///
    /// Shared interior edges must use the same endpoint order.
    pub fn phase_free_mul_assign(&mut self, other: &Stabilizer) {
        self.paulis ^= &other.paulis;
        xor_pauli_maps(&mut self.interior_nodes, &other.interior_nodes);
        xor_pauli_maps(&mut self.port_stabilizer, &other.port_stabilizer);
        let mut sign = self.sign ^ other.sign ^ yy_edge_sign(&other.interior_edges);
        for (&edge, &pauli) in &other.interior_edges {
            let entry = self.interior_edges.entry(edge).or_insert(Pauli::I);
            let was_y = *entry == Pauli::Y;
            *entry = *entry ^ pauli;
            sign ^= was_y ^ (*entry == Pauli::Y);
            if *entry == Pauli::I {
                self.interior_edges.remove(&edge);
            }
        }
        self.sign = sign;
    }

    /// Whether feedback anticommutes with this row at an odd number of targets.
    ///
    /// # Panics
    ///
    /// Panics if wire feedback has no `zx`, or validated feedback metadata is inconsistent.
    pub fn odd_anticommutes_feedback(
        &self,
        targets: &[FeedbackTarget],
        zx: Option<&ZXGraph>,
    ) -> bool {
        targets
            .iter()
            .filter(|target| {
                if let Some(zx) = zx {
                    let (column, pauli) =
                        zx.feedback_column(target).expect("validated feedback wire");
                    return self.paulis.get(column).anticommutes(pauli);
                }
                assert!(
                    target.direction.is_none(),
                    "wire feedback requires its graph"
                );
                self.interior_nodes
                    .get(&target.target)
                    .or_else(|| self.port_stabilizer.get(&target.target))
                    .is_some_and(|support| support.anticommutes(Pauli::from(target.pauli)))
            })
            .count()
            % 2
            == 1
    }

    /// Test fixture with only interior-node support.
    #[doc(hidden)]
    pub fn from_interior_nodes(interior_nodes: impl IntoIterator<Item = (IVec3, Pauli)>) -> Self {
        Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: FxHashMap::default(),
            interior_nodes: interior_nodes.into_iter().collect(),
            interior_edges: FxHashMap::default(),
        }
    }

    /// Adds interior-edge support to a test fixture.
    #[doc(hidden)]
    pub fn with_interior_edges(
        mut self,
        interior_edges: impl IntoIterator<Item = ((IVec3, IVec3), Pauli)>,
    ) -> Self {
        let raw_negative = self.sign ^ yy_edge_sign(&self.interior_edges);
        self.interior_edges = interior_edges.into_iter().collect();
        self.sign = raw_negative ^ yy_edge_sign(&self.interior_edges);
        self
    }
}

pub(crate) fn xor_pauli_maps<K: Copy + Eq + std::hash::Hash>(
    left: &mut FxHashMap<K, Pauli>,
    right: &FxHashMap<K, Pauli>,
) {
    for (&key, &pauli) in right {
        let entry = left.entry(key).or_insert(Pauli::I);
        *entry = *entry ^ pauli;
        if *entry == Pauli::I {
            left.remove(&key);
        }
    }
}

fn yy_edge_sign(edges: &FxHashMap<(IVec3, IVec3), Pauli>) -> bool {
    edges.values().filter(|&&pauli| pauli == Pauli::Y).count() % 2 == 1
}

pub(crate) fn contracted_sign(row_phase: u8, edges: &FxHashMap<(IVec3, IVec3), Pauli>) -> bool {
    debug_assert_eq!(row_phase % 2, 0, "stabilizer rows are Hermitian");
    (row_phase == 2) ^ yy_edge_sign(edges)
}

/// The role a stabilizer generator row plays in the basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilizerRowKind {
    /// Witnesses a named measurement variable.
    Measurement {
        /// Measurement name.
        name: String,
    },
    /// A plain logical generator.
    Logical,
}

/// A stabilizer generator paired with the [`StabilizerRowKind`] it plays.
///
/// `PartialEq` but not `Eq`, following [`Stabilizer`]: comparing rows is how a
/// caller checks that two derivations produced the same basis.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizerGenerator {
    /// The generator's stabilizer.
    pub stabilizer: Stabilizer,
    /// The role this generator plays in the basis.
    pub kind: StabilizerRowKind,
}

impl StabilizerGenerator {
    /// Pairs a stabilizer with its row kind.
    pub fn new(stabilizer: Stabilizer, kind: StabilizerRowKind) -> Self {
        Self { stabilizer, kind }
    }

    /// Whether this row witnesses a measurement variable.
    pub fn is_measurement(&self) -> bool {
        matches!(self.kind, StabilizerRowKind::Measurement { .. })
    }

    /// The witnessed measurement variable name, if this is a measurement row.
    pub fn measurement_name(&self) -> Option<&str> {
        match &self.kind {
            StabilizerRowKind::Measurement { name } => Some(name),
            _ => None,
        }
    }
}

/// Stabilizer state for a ZX graph: a stable public presentation plus private
/// algebraic rank completion.
#[derive(Debug, Clone)]
pub struct StabilizerGenerators {
    /// The ZX graph the generators were derived from.
    pub zx_graph: ZXGraph,
    /// Stable public readable/fixing rows. This presentation need not span the
    /// full output projection by itself.
    pub generators: Vec<StabilizerGenerator>,
    /// Private rank-completion directions. They are not readable identities,
    /// but remain part of full-basis output and feedback semantics.
    auxiliary_rows: Vec<PauliString>,
}

impl StabilizerGenerators {
    /// Builds a public generator set with no internal rank-completion rows.
    pub fn new(zx_graph: ZXGraph, generators: Vec<StabilizerGenerator>) -> Self {
        Self {
            zx_graph,
            generators,
            auxiliary_rows: Vec::new(),
        }
    }

    /// Completes the public rows with `relation` rows outside their span, so
    /// the presentation spans the whole relation. Relation rows omit crossing
    /// centres; completion rows regain them like public rows.
    pub(crate) fn completed(
        zx_graph: ZXGraph,
        generators: Vec<StabilizerGenerator>,
        relation: impl IntoIterator<Item = PauliString>,
    ) -> Self {
        let mut span = basis::PauliSpan::new(zx_graph.total_ids());
        for generator in &generators {
            let mut row = generator.stabilizer.paulis.clone();
            zx_graph.clear_cross_centers(std::slice::from_mut(&mut row));
            span.insert_if_independent(row);
        }
        let mut auxiliary_rows = relation
            .into_iter()
            .filter(|row| span.insert_if_independent(row.clone()))
            .collect::<Vec<_>>();
        zx_graph.reconstruct_cross_center(&mut auxiliary_rows);
        Self {
            zx_graph,
            generators,
            auxiliary_rows,
        }
    }

    pub(crate) fn auxiliary_rows(&self) -> &[PauliString] {
        &self.auxiliary_rows
    }
}

fn stabilizer_external_string(stabilizer: &Stabilizer, ordered_ports: &[IVec3]) -> String {
    ordered_ports
        .iter()
        .map(|pos| {
            stabilizer
                .interior_nodes
                .get(pos)
                .copied()
                .unwrap_or(Pauli::I)
        })
        .map(|pauli| match pauli {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        })
        .collect()
}

pub(crate) fn ordered_port_positions(graph: &BlockGraph) -> Vec<IVec3> {
    let mut ports: Vec<&Block> = graph
        .blocks()
        .filter(|block| {
            matches!(
                block.kind,
                BlockKind::Port | BlockKind::T | BlockKind::Selective(_)
            )
        })
        .collect();
    ports.sort();
    ports.into_iter().map(|block| block.pos).collect()
}

pub(crate) fn fill_ports_auto(
    graph: &BlockGraph,
    limits: ModuleCertificationLimits,
) -> Result<Vec<(BlockGraph, Vec<StabilizerGenerator>)>, BlockGraphError> {
    let ordered_ports = ordered_port_positions(graph);
    if ordered_ports.is_empty() {
        return Err(BlockGraphError::from(FillPortsError::NoOpenPorts));
    }

    let mut generator_map = HashMap::<String, StabilizerGenerator>::new();
    let mut generator_order = Vec::new();
    let mut measurement_support = HashMap::new();
    if let Some(region) = graph.branch_regions()?.first() {
        return Err(crate::InvalidActionError::MissingBranchValue(region.target).into());
    }
    for generator in graph.planned_surfaces(&[], limits)? {
        let external = stabilizer_external_string(&generator.stabilizer, &ordered_ports);
        if let Some(name) = generator.measurement_name() {
            measurement_support.insert(name.to_owned(), external.clone());
        }
        if let Entry::Vacant(slot) = generator_map.entry(external.clone()) {
            generator_order.push(external);
            slot.insert(generator);
        }
    }

    let mut filled_graphs = Vec::new();
    for stabilizer_keys in compatible_stabilizer_cliques(&generator_order) {
        let filled_graph = fill_ports_for_stabilizer_clique(
            graph,
            &ordered_ports,
            &stabilizer_keys,
            &measurement_support,
            limits,
        )?;
        let stabilizers = stabilizer_keys
            .into_iter()
            .map(|key| generator_map[&key].clone())
            .collect();
        filled_graphs.push((filled_graph, stabilizers));
    }
    Ok(filled_graphs)
}

fn compatible_stabilizer_cliques(generators: &[String]) -> Vec<Vec<String>> {
    let len = generators.len();
    let mut conflicts = vec![vec![false; len]; len];
    for i in 0..len {
        for j in (i + 1)..len {
            let conflict = !are_compatible_stabilizers(&generators[i], &generators[j]);
            conflicts[i][j] = conflict;
            conflicts[j][i] = conflict;
        }
    }

    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&lhs, &rhs| {
        let lhs_degree = conflicts[lhs].iter().filter(|&&conflict| conflict).count();
        let rhs_degree = conflicts[rhs].iter().filter(|&&conflict| conflict).count();
        rhs_degree
            .cmp(&lhs_degree)
            .then_with(|| generators[lhs].cmp(&generators[rhs]))
    });

    let mut colors = vec![usize::MAX; len];
    for node in order {
        let mut color = 0;
        loop {
            let blocked =
                (0..len).any(|neighbor| conflicts[node][neighbor] && colors[neighbor] == color);
            if !blocked {
                colors[node] = color;
                break;
            }
            color += 1;
        }
    }

    let mut cliques: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (node, &color) in colors.iter().enumerate() {
        cliques
            .entry(color)
            .or_default()
            .push(generators[node].clone());
    }
    cliques.into_values().collect()
}

fn are_compatible_stabilizers(lhs: &str, rhs: &str) -> bool {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.bytes()
        .zip(rhs.bytes())
        .all(|(l, r)| l == b'I' || r == b'I' || l == r)
}

fn fill_ports_for_stabilizer_clique(
    graph: &BlockGraph,
    ordered_ports: &[IVec3],
    stabilizers: &[String],
    measurement_support: &HashMap<String, String>,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, BlockGraphError> {
    let mut port_ops = vec![Pauli::I; ordered_ports.len()];
    for stabilizer in stabilizers {
        overlay_port_ops(&mut port_ops, stabilizer)?;
    }

    let mut filled = graph.clone();
    for (port_pos, pauli) in ordered_ports.iter().copied().zip(port_ops) {
        let fill_kind = infer_port_fill_kind(
            &filled,
            port_pos,
            if pauli == Pauli::I { Pauli::Z } else { pauli },
        )?;
        let block = filled
            .get_block_mut(port_pos)
            .expect("ordered port positions always exist in the graph");
        block.kind = fill_kind;
    }
    let actions = filled.actions();
    let mut removed = actions
        .iter()
        .map(|action| match action {
            Action::Measure { name, .. } => !stabilizers.contains(&measurement_support[name]),
            _ => false,
        })
        .collect::<Vec<_>>();
    let dag = crate::ActionDag::from_actions(&actions);
    let mut dependents = vec![Vec::new(); actions.len()];
    for (from, to, dependency) in dag.dependencies() {
        if dependency == crate::ActionDependency::Classical {
            dependents[from].push(to);
        }
    }
    let mut pending = removed
        .iter()
        .enumerate()
        .filter_map(|(index, &remove)| remove.then_some(index))
        .collect::<Vec<_>>();
    while let Some(predecessor) = pending.pop() {
        for &consumer in &dependents[predecessor] {
            if !removed[consumer] {
                removed[consumer] = true;
                pending.push(consumer);
            }
        }
    }
    for (ordinal, action) in actions.iter().enumerate() {
        if removed[ordinal] && matches!(action, Action::DiscardIf(_) | Action::Branch { .. }) {
            return Err(FillPortsError::UnavailableControl { ordinal }.into());
        }
    }
    let actions = actions
        .into_iter()
        .enumerate()
        .filter_map(|(ordinal, action)| {
            (!removed[ordinal]
                && match &action {
                    Action::Feedback { .. } => false,
                    Action::Resolve { target, .. } => !ordered_ports.contains(target),
                    _ => true,
                })
            .then_some(action)
        })
        .collect();
    filled.set_actions_with_limits(actions, limits)?;
    crate::validate::validate_with_limits(&filled, limits)?;
    Ok(filled)
}

fn overlay_port_ops(port_ops: &mut [Pauli], stabilizer: &str) -> Result<(), FillPortsError> {
    debug_assert_eq!(port_ops.len(), stabilizer.len());
    for (slot, ch) in port_ops.iter_mut().zip(stabilizer.chars()) {
        let pauli = pauli_from_char(ch)?;
        if pauli != Pauli::I {
            *slot = pauli;
        }
    }
    Ok(())
}

fn infer_port_fill_kind(
    graph: &BlockGraph,
    port_pos: IVec3,
    supported_basis: Pauli,
) -> Result<BlockKind, FillPortsError> {
    match supported_basis {
        Pauli::Y => {
            let pipes = pipes_at(graph, port_pos);
            if let Some(pipe) = pipes.first()
                && pipe_direction_at(pipe, port_pos)?.as_udirection() != UDirection::Z
            {
                return Err(FillPortsError::UnsupportedPortBasis {
                    pos: port_pos,
                    basis: supported_basis,
                });
            }
            Ok(BlockKind::Y)
        }
        Pauli::X | Pauli::Z => {
            let basis = Basis::try_from(supported_basis).expect("X/Z always convert to Basis");
            let pipes = pipes_at(graph, port_pos);
            if pipes.len() != 1 {
                return Err(FillPortsError::PortFillRequiresSinglePipe {
                    pos: port_pos,
                    degree: pipes.len(),
                });
            }
            let pipe = pipes[0];
            let bases = graph.infer_pipe_basis_from_endpoint(pipe, port_pos);
            let cube_bases = bases.map(|entry| entry.unwrap_or(basis));
            let cube_kind = CubeKind::try_from(cube_bases).map_err(|e| {
                FillPortsError::InvalidFilledPortCube {
                    pos: port_pos,
                    source: e,
                }
            })?;
            Ok(BlockKind::Cube(cube_kind))
        }
        _ => Err(FillPortsError::UnsupportedPortBasis {
            pos: port_pos,
            basis: supported_basis,
        }),
    }
}

fn pauli_from_char(ch: char) -> Result<Pauli, FillPortsError> {
    match ch {
        'I' => Ok(Pauli::I),
        'X' => Ok(Pauli::X),
        'Y' => Ok(Pauli::Y),
        'Z' => Ok(Pauli::Z),
        _ => Err(FillPortsError::InvalidExternalStabilizerChar { ch }),
    }
}

fn pipes_at(graph: &BlockGraph, pos: IVec3) -> Vec<&Pipe> {
    graph
        .block_pairs_with_pipe()
        .filter_map(|(u, v, pipe)| ((u.pos == pos) || (v.pos == pos)).then_some(pipe))
        .collect()
}

fn pipe_direction_at(pipe: &Pipe, pos: IVec3) -> Result<Direction, FillPortsError> {
    if pipe.src == pos {
        Ok(pipe.dir)
    } else if pipe.dst() == pos {
        Ok(pipe.dir.negate())
    } else {
        Err(FillPortsError::PortFillRequiresSinglePipe { pos, degree: 0 })
    }
}

/// Fixtures shared by the stabilizer submodules' test modules and by
/// `zx::modular`, which each exercise a different slice of the same graphs and
/// coefficient vectors.
#[cfg(test)]
mod tests {
    #[test]
    fn cancellation_preserves_spent_normalization_budget_and_restores_the_scope() {
        for _ in 0..2 {
            let token = crate::CancellationToken::new();
            token.scope(|| {
                let mut budget = SearchBudget::new(1);
                budget.visit("normalization").unwrap();
                token.cancel();
                assert!(matches!(
                    budget.visit("normalization"),
                    Err(StabilizerError::Cancelled(_))
                ));
                assert_eq!(budget.visited, 2);
                assert!(matches!(
                    budget.check_matrix(1, 1, 0, 0),
                    Err(StabilizerError::Cancelled(_))
                ));
            });
        }
        SearchBudget::new(1).visit("normalization").unwrap();
    }

    #[test]
    fn active_normalization_worker_observes_cancellation_from_another_thread() {
        let token = crate::CancellationToken::new();
        let (started, ready) = std::sync::mpsc::channel();
        let (resume, resumed) = std::sync::mpsc::channel();
        std::thread::scope(|threads| {
            let worker_token = token.clone();
            let worker = threads.spawn(move || {
                worker_token.run(|| {
                    let mut budget = SearchBudget::new(10);
                    budget.visit("normalization").unwrap();
                    started.send(()).unwrap();
                    resumed.recv().unwrap();
                    budget.visit("normalization")
                })
            });
            ready.recv().unwrap();
            token.cancel();
            resume.send(()).unwrap();
            assert!(matches!(
                worker.join().unwrap(),
                Err(StabilizerError::Cancelled(_))
            ));
        });
        SearchBudget::new(1).visit("normalization").unwrap();
    }

    #[test]
    fn normalization_counter_overflow_is_a_resource_failure() {
        let mut budget = SearchBudget::with_limits(ModuleCertificationLimits::UNLIMITED);
        budget.visited = usize::MAX - 1;
        budget
            .visit("normalization")
            .expect("last representable state is allowed");
        assert!(matches!(
            budget.visit("normalization"),
            Err(StabilizerError::ResourceLimited {
                phase: "normalization",
                observed: usize::MAX,
                limit: usize::MAX,
            })
        ));
        assert_eq!(budget.visited, usize::MAX);
    }

    use glam::ivec3;
    use std::collections::HashSet;

    use crate::{Expr, GalleryItem, PauliBasis};

    use super::*;

    #[test]
    fn test_cnot() {
        let cnot = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&cnot).unwrap();
        assert_eq!(zx.nodes.len(), cnot.block_count());
        assert_eq!(zx.edges.len(), 2 * cnot.pipe_count());
        let stabilizers = cnot.stabilizers().unwrap();
        assert_eq!(stabilizers.len(), 4);
    }

    #[test]
    fn signed_y_boundaries_track_choi_transposition() {
        for (blog, negative) in [
            (
                "BLOG 1.0\n\n  0: Y [0,0,0]\n  1: Port [0,0,1]\n  [0,0,0] -> +Z\n",
                false,
            ),
            (
                "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: Y [0,0,1]\n  [0,0,0] -> +Z\n",
                true,
            ),
        ] {
            let stabilizers = BlockGraph::from_blog_text(blog)
                .unwrap()
                .stabilizers()
                .unwrap();
            assert_eq!(stabilizers.len(), 1);
            assert_eq!(stabilizers[0].stabilizer.sign, negative);
        }
    }

    #[test]
    fn test_closed_cnot() {
        let mut cnot = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        for pos in [
            ivec3(0, 0, 0),
            ivec3(1, 1, 0),
            ivec3(0, 0, 3),
            ivec3(1, 1, 3),
        ] {
            let block = cnot.get_block_mut(pos).unwrap();
            block.kind = BlockKind::Cube(crate::CubeKind::ZXZ);
        }
        let zx = ZXGraph::try_from(&cnot).unwrap();
        assert!(!zx.is_open());
        let stabilizers = cnot.stabilizers().unwrap();
        assert_eq!(stabilizers.len(), 2);
    }

    #[test]
    fn test_fill_ports_auto_covers_open_generators() {
        let graph = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let ordered_ports = ordered_port_positions(&graph);
        let generators: HashSet<String> = graph
            .stabilizers()
            .unwrap()
            .into_iter()
            .map(|stabilizer| stabilizer_external_string(&stabilizer.stabilizer, &ordered_ports))
            .collect();

        let filled = graph.fill_ports_auto().unwrap();
        assert!(!filled.is_empty());

        let mut covered = HashSet::new();
        for (filled_graph, stabilizers) in &filled {
            assert_eq!(filled_graph.port_count(), 0);
            filled_graph.validate().unwrap();
            let stabilizer_strings: Vec<String> = stabilizers
                .iter()
                .map(|generator| stabilizer_external_string(&generator.stabilizer, &ordered_ports))
                .collect();
            for stabilizer in &stabilizer_strings {
                covered.insert(stabilizer.clone());
            }
            for i in 0..stabilizer_strings.len() {
                for j in (i + 1)..stabilizer_strings.len() {
                    assert!(are_compatible_stabilizers(
                        &stabilizer_strings[i],
                        &stabilizer_strings[j]
                    ));
                }
            }
        }

        assert_eq!(covered, generators);
    }

    #[test]
    fn automatic_fill_honors_explicit_certification_limits() {
        let graph = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let limited = ModuleCertificationLimits {
            max_local_columns: 0,
            ..ModuleCertificationLimits::DEFAULT
        };
        assert!(matches!(
            graph.filled_graphs_with_limits(limited),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "local ZX columns",
                    limit: 0,
                    ..
                }
            ))
        ));
        assert!(
            !graph
                .filled_graphs_with_limits(ModuleCertificationLimits {
                    max_local_columns: usize::MAX,
                    ..limited
                })
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn automatic_fill_keeps_only_selected_readouts_and_their_aliases() {
        for item in [GalleryItem::T, GalleryItem::TWithPreparedY] {
            let mut source = item
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            source
                .add_action(Action::Let {
                    name: "saved".into(),
                    expr: Expr::Var("mzz".into()),
                })
                .unwrap();
            let filled = source.fill_ports_auto().unwrap();
            assert!(!filled.is_empty());
            let mut retained = 0;
            let mut omitted = 0;
            for (graph, selected) in filled {
                let has_readout = selected
                    .iter()
                    .any(|generator| generator.measurement_name() == Some("mzz"));
                if has_readout {
                    retained += 1;
                } else {
                    omitted += 1;
                }
                assert_eq!(
                    graph.actions().iter().any(
                        |action| matches!(action, Action::Measure { name, .. } if name == "mzz")
                    ),
                    has_readout
                );
                assert_eq!(
                    graph.actions().iter().any(
                        |action| matches!(action, Action::Let { name, .. } if name == "saved")
                    ),
                    has_readout
                );
                assert_eq!(graph.selective_count(), 0);
                assert!(
                    !graph
                        .actions()
                        .iter()
                        .any(|action| matches!(action, Action::Resolve { .. }))
                );
                graph.validate().unwrap();
                for generator in selected {
                    for pos in ordered_port_positions(&source) {
                        let pauli = generator
                            .stabilizer
                            .interior_nodes
                            .get(&pos)
                            .copied()
                            .unwrap_or(Pauli::I);
                        if pauli == Pauli::I {
                            continue;
                        }
                        let fill = graph.get_block(pos).unwrap().kind();
                        let basis = match fill {
                            BlockKind::Y => Pauli::Y,
                            BlockKind::Cube(kind) => Pauli::from(kind.z()),
                            _ => panic!("unfilled {pos}"),
                        };
                        assert_eq!(basis, pauli);
                    }
                }
            }
            assert!(retained > 0 && omitted > 0);
            source
                .add_action(Action::DiscardIf(Expr::Var("saved".into())))
                .unwrap();
            assert!(matches!(
                source.fill_ports_auto(),
                Err(BlockGraphError::FillPorts(
                    FillPortsError::UnavailableControl { .. }
                ))
            ));
        }
    }

    #[test]
    fn fill_ports_auto_removes_feedback_from_filled_variants() {
        let mut graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n  1: Port [0,0,1]\n  [0,0,0] -> +Z\n",
        )
        .unwrap();
        graph
            .add_action(Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::Z,
                    target: ivec3(0, 0, 1),
                    direction: None,
                }],
                condition: None,
            })
            .unwrap();

        let filled = graph.fill_ports_auto().unwrap();
        assert!(!filled.is_empty());

        for (filled_graph, _) in filled {
            assert!(
                filled_graph
                    .actions()
                    .iter()
                    .all(|action| !matches!(action, Action::Feedback { .. }))
            );
            filled_graph.validate().unwrap();
        }
    }

    #[test]
    fn fill_ports_auto_uses_patch_rotation_pipe_face_basis() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: rotate Z [0, 0, 0] -> [1, 0, 1]\n  1: Port [0, 0, -1]\n  2: Port [1, 0, 2]\n  [0, 0, 0] -> -Z\n  [1, 0, 1] -> +Z\n",
        )
        .expect("patch rotation graph should parse");

        assert!(
            graph
                .fill_ports_auto()
                .unwrap()
                .into_iter()
                .all(|(filled_graph, _)| filled_graph.port_count() == 0
                    && filled_graph.validate().is_ok())
        );
    }

    #[test]
    fn product_assumes_consistent_edge_orientation() {
        let a = ivec3(0, 0, 0);
        let b = ivec3(1, 0, 0);
        let with_edge = |edge: (IVec3, IVec3)| Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: Default::default(),
            interior_nodes: Default::default(),
            interior_edges: [(edge, Pauli::X)].into_iter().collect(),
        };

        let cancelled = with_edge((a, b)).phase_free_product(&with_edge((a, b)));
        assert!(cancelled.interior_edges.is_empty());

        let doubled = with_edge((a, b)).phase_free_product(&with_edge((b, a)));
        assert_eq!(doubled.interior_edges.len(), 2);
    }

    #[test]
    fn in_place_product_updates_contracted_y_sign() {
        let edge = (ivec3(0, 0, 0), ivec3(1, 0, 0));
        let row = |pauli| Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: Default::default(),
            interior_nodes: Default::default(),
            interior_edges: [(edge, pauli)].into_iter().collect(),
        };
        let mut product = row(Pauli::X);

        product.phase_free_mul_assign(&row(Pauli::Z));

        assert_eq!(product.interior_edges[&edge], Pauli::Y);
        assert!(product.sign);
    }

    #[test]
    fn test_ordered_port_positions_uses_block_order() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(2, 0, 0), BlockKind::Port));
        graph.add_block(
            Block::new(ivec3(1, 0, 0), BlockKind::Port)
                .with_tag("b")
                .expect("valid tag"),
        );
        graph.add_block(
            Block::new(ivec3(0, 0, 0), BlockKind::Port)
                .with_tag("a")
                .expect("valid tag"),
        );

        assert_eq!(
            ordered_port_positions(&graph),
            vec![ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(2, 0, 0)]
        );
    }
}
