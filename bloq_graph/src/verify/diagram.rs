//! Logical-verifier construction, branch instantiation, and map comparison.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use glam::IVec3;
use quizx::fscalar::Zero;
use quizx::graph::{EType, GraphLike, V, VType};
use quizx::params::{Parity, Var};
use quizx::phase::Phase;
use quizx::tensor::{CompareTensors, TensorF, ToTensor};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use rustc_hash::FxHashMap;

use crate::{
    Action, ActionNode, BinaryOp, BlockGraph, Expr, FeedbackTarget, NodeKind, Pauli, PauliBasis,
    PortRole, RuntimeStabilizerBasis, SelectiveKind, ZXGraph,
};

use super::correction::{InternalCorrection, ResolvedCorrection, fresh_var, named_readouts};
use super::{MeasurementKey, VerifyLogicalError, stabilizer_supports_measurement};
use crate::zx::{CoeffVec, solve_coeff_combination};

/// The concrete QuiZX graph type used for logical maps.
pub type QuizxGraph = quizx::vec_graph::Graph;

/// Ordered open boundaries of a logical map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryOrder {
    inputs: Vec<IVec3>,
    outputs: Vec<IVec3>,
}

impl BoundaryOrder {
    /// Creates an explicit input/output order.
    pub fn new(inputs: Vec<IVec3>, outputs: Vec<IVec3>) -> Self {
        Self { inputs, outputs }
    }

    /// Ordered input ports.
    pub fn inputs(&self) -> &[IVec3] {
        &self.inputs
    }

    /// Ordered output ports.
    pub fn outputs(&self) -> &[IVec3] {
        &self.outputs
    }

    fn infer(source: &BlockGraph, zx: &ZXGraph) -> Self {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for node in zx.nodes().iter().filter(|node| node.kind == NodeKind::Port) {
            let neighbors = zx.neighbors(node.id).unwrap_or_default();
            let temporal_input = !neighbors.is_empty()
                && neighbors
                    .iter()
                    .all(|&neighbor| zx.nodes()[neighbor].pos.z > node.pos.z);
            let role = source
                .get_block(node.pos)
                .and_then(crate::Block::port_role)
                .unwrap_or(PortRole::Auto);
            if role.has_input_boundary() || (role == PortRole::Auto && temporal_input) {
                inputs.push(node.pos);
            }
            if role.has_output_boundary() || (role == PortRole::Auto && !temporal_input) {
                outputs.push(node.pos);
            }
        }
        inputs.sort_by_key(IVec3::to_array);
        outputs.sort_by_key(IVec3::to_array);
        let input_keys = inputs
            .iter()
            .map(|&position| boundary_tag_key(source, position, true))
            .collect::<Option<Vec<_>>>();
        if let Some(mut keys) = input_keys {
            let mut tagged_inputs = inputs.into_iter().zip(keys.drain(..)).collect::<Vec<_>>();
            tagged_inputs.sort_by(|(lhs_pos, lhs), (rhs_pos, rhs)| {
                lhs.cmp(rhs)
                    .then_with(|| lhs_pos.to_array().cmp(&rhs_pos.to_array()))
            });
            inputs = tagged_inputs
                .iter()
                .map(|(position, _)| *position)
                .collect();
            let ranks = tagged_inputs
                .into_iter()
                .enumerate()
                .map(|(rank, (_, key))| (key, rank))
                .collect::<FxHashMap<_, _>>();
            outputs.sort_by_key(|&position| {
                let rank = boundary_tag_key(source, position, false)
                    .and_then(|key| ranks.get(&key).copied())
                    .unwrap_or(usize::MAX);
                (rank, position.to_array())
            });
        }
        Self { inputs, outputs }
    }

    fn validate(&self, zx: &ZXGraph) -> Result<(), VerifyLogicalError> {
        let ports = zx
            .nodes()
            .iter()
            .filter(|node| node.kind == NodeKind::Port)
            .map(|node| node.pos.to_array())
            .collect::<BTreeSet<_>>();
        let collect = |positions: &[IVec3]| {
            let mut seen = BTreeSet::new();
            for &position in positions {
                if !ports.contains(&position.to_array()) {
                    return Err(VerifyLogicalError::InvalidBoundary(position));
                }
                if !seen.insert(position.to_array()) {
                    return Err(VerifyLogicalError::DuplicateBoundary(position));
                }
            }
            Ok(seen)
        };
        let inputs = collect(&self.inputs)?;
        let outputs = collect(&self.outputs)?;
        for position in inputs.intersection(&outputs) {
            let pos = IVec3::from_array(*position);
            if zx
                .node_at(pos)
                .is_none_or(|node| node.role != PortRole::Multiplex)
            {
                return Err(VerifyLogicalError::DuplicateBoundary(pos));
            }
        }
        for node in zx
            .nodes()
            .iter()
            .filter(|node| node.role == PortRole::Multiplex)
        {
            if !inputs.contains(&node.pos.to_array()) || !outputs.contains(&node.pos.to_array()) {
                return Err(VerifyLogicalError::MissingBoundary(node.pos));
            }
        }
        let seen = inputs.union(&outputs).copied().collect::<BTreeSet<_>>();
        if let Some(position) = ports.difference(&seen).next() {
            return Err(VerifyLogicalError::MissingBoundary(IVec3::from_array(
                *position,
            )));
        }
        Ok(())
    }
}

fn boundary_tag_key(source: &BlockGraph, position: IVec3, input: bool) -> Option<String> {
    let tag = source.get_block(position)?.tag()?.to_ascii_lowercase();
    let prefixes: &[&str] = if input {
        &["input", "in"]
    } else {
        &["output", "out"]
    };
    prefixes.iter().find_map(|prefix| {
        tag.strip_prefix(prefix)
            .map(|suffix| suffix.trim_start_matches('_').to_string())
    })
}

/// Caller-chosen measurement outcomes and external Boolean inputs.
pub type BranchAssignment = BTreeMap<String, bool>;

/// Result category for one sampled assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchStatus {
    /// The action assignment produced a diagram, but no expected map was checked.
    Instantiated,
    /// A DiscardIf fired; no graph was contracted.
    Rejected,
    /// The concrete diagram contracted to zero, so the assignment is impossible.
    Impossible,
    /// A nonzero map matched the expected map projectively.
    Verified,
}

/// Classical values and status for one branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalBranch {
    status: BranchStatus,
    classical_state: BTreeMap<String, bool>,
}

impl LogicalBranch {
    /// Branch result.
    pub const fn status(&self) -> BranchStatus {
        self.status
    }

    /// External inputs, presampled measurement outcomes, and let bindings
    /// evaluated before the branch was rejected or fully instantiated.
    pub fn classical_state(&self) -> &BTreeMap<String, bool> {
        &self.classical_state
    }
}

/// Aggregate result of randomized branch verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalVerificationReport {
    /// Number of random assignments attempted.
    pub samples: usize,
    /// Accepted, nonzero branches proven equal to the expected map.
    pub verified: usize,
    /// Assignments stopped by DiscardIf before contraction.
    pub rejected: usize,
    /// Assignments whose concrete ZX diagram was the zero map.
    pub impossible: usize,
}

#[derive(Debug, Clone)]
struct SelectiveSite {
    vertex: V,
    kind: SelectiveKind,
    is_state: bool,
}

#[derive(Debug, Clone, Copy)]
struct WireAnchor {
    target: V,
    next: V,
    edge_type: EType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InternalMeasurements {
    Ignore,
    Correct,
}

const MAX_CACHED_STRUCTURAL_PROJECTIONS: usize = 64;

#[derive(Debug, Clone)]
struct StructuralVerifier {
    actions: Vec<Action>,
    source: BlockGraph,
    targets: Vec<IVec3>,
    boundaries: BoundaryOrder,
    internal_measurements: InternalMeasurements,
    arms: Arc<Mutex<BTreeMap<Vec<bool>, Arc<LogicalVerifier>>>>,
}

impl StructuralVerifier {
    fn arm(&self, values: &[bool]) -> Result<Arc<LogicalVerifier>, VerifyLogicalError> {
        if let Some(arm) = self
            .arms
            .lock()
            .expect("projection cache is not poisoned")
            .get(values)
        {
            return Ok(Arc::clone(arm));
        }
        let projected = self
            .source
            .project_branches(self.targets.iter().copied().zip(values.iter().copied()))?;
        let arm = Arc::new(LogicalVerifier::from_projection(
            &projected,
            Some(self.boundaries.clone()),
            self.internal_measurements,
        )?);
        let mut cache = self.arms.lock().expect("projection cache is not poisoned");
        // ponytail: bounded cache, clear on mask churn; use LRU if profiling
        // finds repeated rebuilds of hot projections significant.
        if cache.len() >= MAX_CACHED_STRUCTURAL_PROJECTIONS {
            cache.clear();
        }
        cache.insert(values.to_vec(), Arc::clone(&arm));
        Ok(arm)
    }

    fn values_for(
        &self,
        assignment: &BranchAssignment,
        inputs: &BTreeSet<String>,
    ) -> Result<Vec<bool>, VerifyLogicalError> {
        let mut state = assigned_inputs(inputs, assignment)?;
        for action in &self.actions {
            let Action::Measure { name, .. } = action else {
                continue;
            };
            let value = assignment
                .get(name)
                .copied()
                .ok_or_else(|| VerifyLogicalError::MissingMeasurementValue(name.clone()))?;
            state.insert(name.clone(), value);
        }

        let mut values = Vec::new();
        for action in &self.actions {
            match action {
                Action::Let { name, expr } => {
                    state.insert(name.clone(), eval_expr(expr, &state));
                }
                Action::Branch { condition, .. } => {
                    values.push(eval_expr(condition, &state));
                }
                _ => {}
            }
        }
        Ok(values)
    }
}

/// Reusable symbolic ZX verifier for one static graph or a bounded family of
/// reachable terminal-branch projections.
#[derive(Debug, Clone)]
pub struct LogicalVerifier {
    symbolic: QuizxGraph,
    boundaries: BoundaryOrder,
    actions: Vec<ActionNode>,
    inputs: BTreeSet<String>,
    measurement_vars: BTreeMap<String, Var>,
    feedback_vars: FxHashMap<usize, Var>,
    selectives: FxHashMap<IVec3, SelectiveSite>,
    correction: Option<InternalCorrection>,
    structural: Option<StructuralVerifier>,
}

struct ResolvedAssignment {
    branch: LogicalBranch,
    variable_values: FxHashMap<Var, bool>,
    selective_values: FxHashMap<IVec3, PauliBasis>,
    feedback_values: FxHashMap<usize, bool>,
    correction: Option<ResolvedCorrection>,
}

impl LogicalVerifier {
    /// The tensor boundary order used by this verifier.
    pub fn boundaries(&self) -> &BoundaryOrder {
        &self.boundaries
    }
    /// Builds a symbolic verifier from explicit roles or temporal geometry.
    /// [`crate::CancellationToken::run`] enables cooperative cancellation during
    /// named-measurement normalization.
    ///
    /// # Errors
    ///
    /// Returns an error if graph conversion, boundaries, actions, or symbolic setup are invalid.
    pub fn new(source: &BlockGraph) -> Result<Self, VerifyLogicalError> {
        Self::from_source(source, None, InternalMeasurements::Ignore)
    }

    /// Builds a verifier with caller-specified tensor-axis and boundary order.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph or supplied boundary order is invalid.
    pub fn with_boundaries(
        source: &BlockGraph,
        boundaries: BoundaryOrder,
    ) -> Result<Self, VerifyLogicalError> {
        Self::from_source(source, Some(boundaries), InternalMeasurements::Ignore)
    }

    /// Builds a verifier that samples unnamed boundary/joint measurements and
    /// materializes their terminal output-frame correction.
    /// Sampling and selective fills preserve the supplied named readout parities.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph or its internal correction system is invalid.
    pub fn with_internal_measurements(source: &BlockGraph) -> Result<Self, VerifyLogicalError> {
        Self::from_source(source, None, InternalMeasurements::Correct)
    }

    /// Internal-measurement mode with caller-specified boundary order.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph, boundaries, or internal correction system is invalid.
    pub fn with_boundaries_and_internal_measurements(
        source: &BlockGraph,
        boundaries: BoundaryOrder,
    ) -> Result<Self, VerifyLogicalError> {
        Self::from_source(source, Some(boundaries), InternalMeasurements::Correct)
    }

    fn from_source(
        source: &BlockGraph,
        boundaries: Option<BoundaryOrder>,
        internal_measurements: InternalMeasurements,
    ) -> Result<Self, VerifyLogicalError> {
        source.require_flat_hierarchy("logical verification")?;
        // Records must be readable before live outputs, as in compilation.
        crate::validate::validate(&source.copy_local_geometry().fix_shadowed_faces())?;
        let regions = source.branch_regions()?;
        if regions.is_empty() {
            return Self::from_projection(source, boundaries, internal_measurements);
        }
        let assignments = source
            .branch_assignments_up_to(1)?
            .pop()
            .expect("Boolean branch conditions have a reachable assignment");
        let projected = source.project_branches(assignments.iter().copied())?;
        let mut verifier = Self::from_projection(&projected, boundaries, internal_measurements)?;
        let boundaries = verifier.boundaries.clone();
        let values = assignments.iter().map(|(_, value)| *value).collect();
        let arms = Arc::new(Mutex::new(BTreeMap::from([(
            values,
            Arc::new(verifier.clone()),
        )])));
        verifier.structural = Some(StructuralVerifier {
            actions: source.actions(),
            source: source.clone(),
            targets: assignments.iter().map(|(target, _)| *target).collect(),
            boundaries,
            internal_measurements,
            arms,
        });
        Ok(verifier)
    }

    /// Builds the static verifier of one graph without structural branches.
    fn from_projection(
        graph: &BlockGraph,
        boundaries: Option<BoundaryOrder>,
        internal_measurements: InternalMeasurements,
    ) -> Result<Self, VerifyLogicalError> {
        let zx = graph.to_zx_graph()?;
        let boundaries = boundaries.unwrap_or_else(|| BoundaryOrder::infer(graph, &zx));
        zx.validate_for_program()?;
        boundaries.validate(&zx)?;
        let actions = zx
            .action_graph()
            .ordered_nodes()
            .cloned()
            .collect::<Vec<_>>();
        let inputs = zx.action_graph().inputs().map(str::to_owned).collect();

        let mut next_variable = 0;
        let mut measurement_vars = BTreeMap::new();
        let mut feedback_vars = FxHashMap::default();
        for action in &actions {
            match &action.action {
                Action::Measure { name, .. } => {
                    measurement_vars.insert(name.clone(), fresh_var(&mut next_variable)?);
                }
                Action::Feedback { targets, .. } if !targets.is_empty() => {
                    feedback_vars.insert(action.ordinal, fresh_var(&mut next_variable)?);
                }
                _ => {}
            }
        }

        let records = named_records(&zx, &actions);
        let named_phases = named_measurement_phases(&zx, &actions, &measurement_vars, &records)?;
        let correction = match internal_measurements {
            InternalMeasurements::Ignore => None,
            InternalMeasurements::Correct => Some(InternalCorrection::new(
                &zx,
                &named_phases,
                &measurement_vars,
                records,
                &mut next_variable,
            )?),
        };
        let phases = correction
            .as_ref()
            .map(InternalCorrection::measurement_vars)
            .unwrap_or(&named_phases);
        let (symbolic, selectives) =
            build_symbolic_graph(&zx, &boundaries, &actions, phases, &feedback_vars);
        Ok(Self {
            symbolic,
            boundaries,
            actions,
            inputs,
            measurement_vars,
            feedback_vars,
            selectives,
            correction,
            structural: None,
        })
    }

    /// Instantiates a deterministic assignment. Rejected branches have no diagram.
    ///
    /// # Errors
    ///
    /// Returns an error for missing values or an invalid reachable branch.
    pub fn instantiate(
        &self,
        assignment: &BranchAssignment,
    ) -> Result<(LogicalBranch, Option<QuizxGraph>), VerifyLogicalError> {
        self.instantiate_with(assignment, |_| false)
    }

    /// Samples external inputs and measurements, then instantiates one branch.
    ///
    /// # Errors
    ///
    /// Returns an error if the sampled branch cannot be instantiated.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<(LogicalBranch, Option<QuizxGraph>), VerifyLogicalError> {
        let assignment = self
            .inputs
            .iter()
            .chain(self.measurement_vars.keys())
            .map(|name| (name.clone(), rng.random()))
            .collect::<BranchAssignment>();
        self.instantiate_with(&assignment, |_| rng.random())
    }

    /// Samples one reproducible branch.
    ///
    /// # Errors
    ///
    /// Returns an error if the sampled branch cannot be instantiated.
    pub fn sample_with_seed(
        &self,
        seed: u64,
    ) -> Result<(LogicalBranch, Option<QuizxGraph>), VerifyLogicalError> {
        self.sample(&mut StdRng::seed_from_u64(seed))
    }

    /// Verifies one caller-supplied assignment against expected.
    ///
    /// # Errors
    ///
    /// Returns an arity, assignment, zero-map, or logical-mismatch error.
    pub fn verify_branch(
        &self,
        expected: &QuizxGraph,
        assignment: &BranchAssignment,
    ) -> Result<LogicalBranch, VerifyLogicalError> {
        self.check_boundary_arity(expected)?;
        let expected_tensor = contract(expected);
        if !tensor_is_nonzero(&expected_tensor) {
            return Err(VerifyLogicalError::ZeroExpectedMap);
        }
        let (branch, diagram) = self.instantiate(assignment)?;
        compare_branch(branch, diagram, &expected_tensor)
    }

    /// Verifies random measurement assignments from a stable seed.
    ///
    /// # Errors
    ///
    /// Returns an arity, sampling, zero-map, or logical-mismatch error.
    pub fn verify(
        &self,
        expected: &QuizxGraph,
        samples: usize,
        seed: u64,
    ) -> Result<LogicalVerificationReport, VerifyLogicalError> {
        self.check_boundary_arity(expected)?;
        let expected_tensor = contract(expected);
        if !tensor_is_nonzero(&expected_tensor) {
            return Err(VerifyLogicalError::ZeroExpectedMap);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut report = LogicalVerificationReport {
            samples,
            verified: 0,
            rejected: 0,
            impossible: 0,
        };
        for _ in 0..samples {
            let (branch, diagram) = self.sample(&mut rng)?;
            let branch = compare_branch(branch, diagram, &expected_tensor)?;
            match branch.status {
                BranchStatus::Instantiated => {
                    unreachable!("map comparison always resolves an instantiated branch")
                }
                BranchStatus::Rejected => report.rejected += 1,
                BranchStatus::Impossible => report.impossible += 1,
                BranchStatus::Verified => report.verified += 1,
            }
        }
        if report.verified == 0 {
            return Err(VerifyLogicalError::NoVerifiedBranch { samples });
        }
        Ok(report)
    }

    fn check_boundary_arity(&self, expected: &QuizxGraph) -> Result<(), VerifyLogicalError> {
        let actual_inputs = self.boundaries.inputs.len();
        let actual_outputs = self.boundaries.outputs.len();
        let expected_inputs = expected.inputs().len();
        let expected_outputs = expected.outputs().len();
        if (actual_inputs, actual_outputs) != (expected_inputs, expected_outputs) {
            return Err(VerifyLogicalError::BoundaryArityMismatch {
                actual_inputs,
                actual_outputs,
                expected_inputs,
                expected_outputs,
            });
        }
        Ok(())
    }

    fn instantiate_with(
        &self,
        assignment: &BranchAssignment,
        internal_value: impl FnMut(Var) -> bool,
    ) -> Result<(LogicalBranch, Option<QuizxGraph>), VerifyLogicalError> {
        if let Some(structural) = &self.structural {
            let values = structural.values_for(assignment, &self.inputs)?;
            let arm = structural.arm(&values)?;
            return arm.instantiate_with(assignment, internal_value);
        }

        let resolved = self.resolve_assignment(assignment)?;
        self.instantiate_resolved(&resolved, internal_value)
    }

    fn resolve_assignment(
        &self,
        assignment: &BranchAssignment,
    ) -> Result<ResolvedAssignment, VerifyLogicalError> {
        let mut classical_state = assigned_inputs(&self.inputs, assignment)?;
        let mut variable_values = FxHashMap::<Var, bool>::default();
        let mut selective_values = FxHashMap::<IVec3, PauliBasis>::default();

        // Measurements are branch inputs and may be referenced before their action.
        for action in &self.actions {
            let Action::Measure { name, .. } = &action.action else {
                continue;
            };
            let value = assignment
                .get(name)
                .copied()
                .ok_or_else(|| VerifyLogicalError::MissingMeasurementValue(name.clone()))?;
            classical_state.insert(name.clone(), value);
            variable_values.insert(self.measurement_vars[name], value);
        }

        for action in &self.actions {
            match &action.action {
                Action::Let { name, expr } => {
                    classical_state.insert(name.clone(), eval_expr(expr, &classical_state));
                }
                Action::DiscardIf(expr) if eval_expr(expr, &classical_state) => {
                    return Ok(ResolvedAssignment {
                        branch: LogicalBranch {
                            status: BranchStatus::Rejected,
                            classical_state,
                        },
                        variable_values,
                        selective_values,
                        feedback_values: FxHashMap::default(),
                        correction: None,
                    });
                }
                Action::Resolve { target, condition } => {
                    let site = self
                        .selectives
                        .get(target)
                        .expect("validated resolve target has a selective site");
                    let chosen = if eval_expr(condition, &classical_state) {
                        site.kind.pauli_if_true()
                    } else {
                        site.kind.pauli_if_false()
                    };
                    selective_values.insert(*target, chosen);
                }
                Action::Feedback { condition, .. } => {
                    if let Some(variable) = self.feedback_vars.get(&action.ordinal) {
                        let value = condition
                            .as_ref()
                            .is_none_or(|expr| eval_expr(expr, &classical_state));
                        variable_values.insert(*variable, value);
                    }
                }
                // Conditional sources are projected to a static arm before this
                // evaluator runs; the all-true construction pass has no action
                // work left for the structural marker itself.
                Action::Branch { .. } | Action::Measure { .. } | Action::DiscardIf(_) => {}
            }
        }

        let feedback_values = self
            .feedback_vars
            .iter()
            .map(|(&ordinal, &variable)| (ordinal, variable_values[&variable]))
            .collect();
        let correction = self
            .correction
            .as_ref()
            .map(|correction| correction.resolve(&self.actions, &selective_values))
            .transpose()?;
        Ok(ResolvedAssignment {
            branch: LogicalBranch {
                status: BranchStatus::Instantiated,
                classical_state,
            },
            variable_values,
            selective_values,
            feedback_values,
            correction,
        })
    }

    fn instantiate_resolved(
        &self,
        resolved: &ResolvedAssignment,
        mut internal_value: impl FnMut(Var) -> bool,
    ) -> Result<(LogicalBranch, Option<QuizxGraph>), VerifyLogicalError> {
        let mut variable_values = resolved.variable_values.clone();
        if let Some(correction) = &self.correction {
            for variable in correction.sampled_vars() {
                variable_values.insert(variable, internal_value(variable));
            }
        }
        if resolved.branch.status == BranchStatus::Rejected {
            return Ok((resolved.branch.clone(), None));
        }
        let selective_values = &resolved.selective_values;

        let (mut diagram, sites) = if let Some(correction) = &self.correction {
            build_symbolic_graph(
                correction.zx_graph(),
                &self.boundaries,
                &self.actions,
                &resolved
                    .correction
                    .as_ref()
                    .expect("resolved internal branch")
                    .phases,
                &self.feedback_vars,
            )
        } else {
            (self.symbolic.clone(), self.selectives.clone())
        };
        for (&position, site) in &sites {
            let chosen = selective_values[&position];
            let (vertex_type, base_phase) = measurement_spider(chosen, site.is_state);
            diagram.set_vertex_type(site.vertex, vertex_type);
            diagram.set_phase(site.vertex, base_phase);
        }
        for vertex in diagram.vertex_vec() {
            let odd = super::parity::phase_value(&diagram.vars(vertex), |variable| {
                variable_values.get(&variable).copied().unwrap_or(false)
            });
            if odd {
                diagram.add_to_phase(vertex, Phase::from(1_i64));
            }
            diagram.set_vars(vertex, Parity::default());
        }

        if let Some(correction) = &self.correction {
            correction.apply(
                &mut diagram,
                &self.boundaries.outputs,
                &variable_values,
                resolved
                    .correction
                    .as_ref()
                    .expect("resolved internal branch"),
                &self.actions,
                &resolved.feedback_values,
            )?;
        }

        Ok((resolved.branch.clone(), Some(diagram)))
    }
}

fn assigned_inputs(
    inputs: &BTreeSet<String>,
    assignment: &BranchAssignment,
) -> Result<BTreeMap<String, bool>, VerifyLogicalError> {
    inputs
        .iter()
        .map(|name| {
            assignment
                .get(name)
                .copied()
                .map(|value| (name.clone(), value))
                .ok_or_else(|| VerifyLogicalError::MissingInputValue(name.clone()))
        })
        .collect()
}

/// Builds and fuzzes a verifier in one call.
///
/// # Errors
///
/// Returns verifier-construction or sampled logical-verification errors.
pub fn verify_logical(
    graph: &BlockGraph,
    expected: &QuizxGraph,
    samples: usize,
    seed: u64,
) -> Result<LogicalVerificationReport, VerifyLogicalError> {
    LogicalVerifier::new(graph)?.verify(expected, samples, seed)
}

fn compare_branch(
    mut branch: LogicalBranch,
    diagram: Option<QuizxGraph>,
    expected: &TensorF,
) -> Result<LogicalBranch, VerifyLogicalError> {
    let Some(diagram) = diagram else {
        branch.status = BranchStatus::Rejected;
        return Ok(branch);
    };
    let actual = contract(&diagram);
    if !tensor_is_nonzero(&actual) {
        branch.status = BranchStatus::Impossible;
        return Ok(branch);
    }
    if !<TensorF as CompareTensors>::scalar_eq(&actual, expected) {
        return Err(VerifyLogicalError::MapMismatch {
            classical_state: branch.classical_state,
        });
    }
    branch.status = BranchStatus::Verified;
    Ok(branch)
}

fn tensor_is_nonzero(tensor: &TensorF) -> bool {
    tensor.iter().any(|value| !value.is_zero())
}

fn contract(diagram: &QuizxGraph) -> TensorF {
    let mut reduced = diagram.clone();
    quizx::simplify::full_simp(&mut reduced);
    reduced.to_tensorf()
}

fn eval_expr(expr: &Expr, values: &BTreeMap<String, bool>) -> bool {
    match expr {
        Expr::Var(name) => values[name],
        Expr::Not(inner) => !eval_expr(inner, values),
        Expr::Binary(op, lhs, rhs) => {
            let lhs = eval_expr(lhs, values);
            let rhs = eval_expr(rhs, values);
            match op {
                BinaryOp::Xor => lhs ^ rhs,
                BinaryOp::And => lhs & rhs,
                BinaryOp::Or => lhs | rhs,
            }
        }
    }
}

/// Every named record's native key.
fn named_records(zx: &ZXGraph, actions: &[ActionNode]) -> BTreeMap<String, MeasurementKey> {
    actions
        .iter()
        .filter_map(|action| {
            let Action::Measure { name, .. } = &action.action else {
                return None;
            };
            let key =
                MeasurementKey::for_action(zx, action).expect("validated measurement has a key");
            Some((name.clone(), key))
        })
        .collect()
}

fn named_measurement_phases(
    zx: &ZXGraph,
    actions: &[ActionNode],
    variables: &BTreeMap<String, Var>,
    records: &BTreeMap<String, MeasurementKey>,
) -> Result<BTreeMap<MeasurementKey, Parity>, VerifyLogicalError> {
    let mut phases = BTreeMap::new();
    let mut crossings = Vec::new();
    for action in actions {
        let Action::Measure { name, .. } = &action.action else {
            continue;
        };
        let key = MeasurementKey::for_action(zx, action).expect("validated measurement has a key");
        if let MeasurementKey::Node { pos, pauli } = key {
            let node = zx.node_at(pos).expect("validated measurement node");
            let native = match node.kind {
                NodeKind::X => Pauli::Z,
                NodeKind::Z => Pauli::X,
                _ => pauli,
            };
            if native != pauli {
                crossings.push((name.as_str(), node.id, pauli));
            }
            phases.insert(
                MeasurementKey::Node { pos, pauli: native },
                Parity::single(variables[name]),
            );
        } else {
            phases.insert(key, Parity::single(variables[name]));
        }
    }
    if crossings.is_empty() {
        return Ok(phases);
    }

    let basis = RuntimeStabilizerBasis::from_zx_graph(zx.clone())
        .map_err(|source| VerifyLogicalError::RuntimeBasisInitializationFailed { source })?;
    let surfaces = named_readouts(&basis, records)?;
    let rows = actions
        .iter()
        .filter_map(|action| {
            let Action::Measure { name, .. } = &action.action else {
                return None;
            };
            Some((name.as_str(), &surfaces[name]))
        })
        .collect::<Vec<_>>();
    for (name, node_id, pauli) in crossings {
        let row = rows
            .iter()
            .find(|(measured, _)| *measured == name)
            .expect("named row exists")
            .1;
        for &neighbor in zx.neighbors(node_id).expect("measurement node exists") {
            let edge = zx
                .edge_between(node_id, neighbor)
                .expect("incident edge exists");
            let (first, second) = sorted_pair(node_id, neighbor);
            let key = MeasurementKey::Edge {
                src: zx.nodes()[first].pos,
                dst: zx.nodes()[second].pos,
                pauli: if node_id == second && edge.hadamard {
                    pauli.flip()
                } else {
                    pauli
                },
            };
            if stabilizer_supports_measurement(row, &key) {
                phases.entry(key).or_default();
            }
        }
    }
    // Native phases must toggle the selected named rows jointly: a crossing
    // record can share a broadcast with another record.
    let columns = phases
        .keys()
        .map(|key| {
            let mut column = CoeffVec::zeros(rows.len());
            for (index, (_, row)) in rows.iter().enumerate() {
                column.set_bit(index, stabilizer_supports_measurement(row, key));
            }
            column
        })
        .collect::<Vec<_>>();
    for parity in phases.values_mut() {
        *parity = Parity::default();
    }
    for (index, &(name, _)) in rows.iter().enumerate() {
        let solution =
            solve_coeff_combination(&columns, &CoeffVec::singleton(index, rows.len()))
                .ok_or_else(|| VerifyLogicalError::MeasurementPhaseUnavailable(name.to_owned()))?;
        for (candidate, parity) in phases.values_mut().enumerate() {
            if solution.bit(candidate) {
                *parity = &*parity + &Parity::single(variables[name]);
            }
        }
    }
    phases.retain(|_, parity| !parity.is_empty());
    Ok(phases)
}

fn build_symbolic_graph(
    zx: &ZXGraph,
    boundaries: &BoundaryOrder,
    actions: &[ActionNode],
    measurement_key_vars: &BTreeMap<MeasurementKey, Parity>,
    feedback_vars: &FxHashMap<usize, Var>,
) -> (QuizxGraph, FxHashMap<IVec3, SelectiveSite>) {
    let mut graph = QuizxGraph::new();
    let mut node_vertices = Vec::with_capacity(zx.nodes().len());
    let mut input_boundaries = FxHashMap::default();
    let mut output_boundaries = FxHashMap::default();
    let mut selectives = FxHashMap::default();
    for node in zx.nodes() {
        if node.kind == NodeKind::Port && node.role == PortRole::Multiplex {
            let spider = graph.add_vertex_with_phase(VType::Z, Phase::from(0_i64));
            let input = graph.add_vertex_with_phase(VType::B, Phase::from(0_i64));
            let output = graph.add_vertex_with_phase(VType::B, Phase::from(0_i64));
            graph.set_coord(spider, (f64::from(node.pos.z), f64::from(node.pos.x)));
            graph.set_coord(input, (f64::from(node.pos.z) - 0.25, f64::from(node.pos.x)));
            graph.set_coord(
                output,
                (f64::from(node.pos.z) + 0.25, f64::from(node.pos.x)),
            );
            graph.add_edge_with_type(input, spider, EType::N);
            graph.add_edge_with_type(spider, output, EType::N);
            input_boundaries.insert(node.pos, input);
            output_boundaries.insert(node.pos, output);
            node_vertices.push(spider);
            continue;
        }
        let is_state = is_state_boundary(zx, node.id);
        let (vertex_type, phase) = match node.kind {
            NodeKind::X => (VType::X, Phase::from(0_i64)),
            NodeKind::Z => (VType::Z, Phase::from(0_i64)),
            NodeKind::Y => measurement_spider(PauliBasis::Y, is_state),
            NodeKind::Port => (VType::B, Phase::from(0_i64)),
            NodeKind::T => (VType::Z, Phase::from((1_i64, 4_i64))),
            NodeKind::Selective(kind) => {
                let vertex = graph.add_vertex_with_phase(VType::Z, Phase::from(0_i64));
                graph.set_coord(vertex, (f64::from(node.pos.z), f64::from(node.pos.x)));
                node_vertices.push(vertex);
                selectives.insert(
                    node.pos,
                    SelectiveSite {
                        vertex,
                        kind,
                        is_state,
                    },
                );
                continue;
            }
        };
        let vertex = graph.add_vertex_with_phase(vertex_type, phase);
        graph.set_coord(vertex, (f64::from(node.pos.z), f64::from(node.pos.x)));
        if node.kind == NodeKind::Port {
            input_boundaries.insert(node.pos, vertex);
            output_boundaries.insert(node.pos, vertex);
        }
        node_vertices.push(vertex);
    }

    let mut edge_measurements = FxHashMap::<(usize, usize), Vec<(PauliBasis, &Parity)>>::default();
    for (key, parity) in measurement_key_vars {
        match key {
            MeasurementKey::Node { pos, .. } => {
                let node = zx.node_at(*pos).expect("scheduled measurement node exists");
                graph.add_to_vars(node_vertices[node.id], parity);
            }
            MeasurementKey::Edge { src, dst, pauli } => {
                let lhs = zx.node_at(*src).expect("scheduled measurement source").id;
                let rhs = zx
                    .node_at(*dst)
                    .expect("scheduled measurement destination")
                    .id;
                let basis = match pauli {
                    Pauli::X => PauliBasis::X,
                    Pauli::Z => PauliBasis::Z,
                    Pauli::I | Pauli::Y => unreachable!("wire phases use X or Z components"),
                };
                edge_measurements
                    .entry(sorted_pair(lhs, rhs))
                    .or_default()
                    .push((basis, parity));
            }
        }
    }
    let mut anchors = FxHashMap::<(usize, usize), WireAnchor>::default();
    for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
        let lhs = node_vertices[edge.n1];
        let rhs = node_vertices[edge.n2];
        let edge_type = if edge.hadamard { EType::H } else { EType::N };
        let mut first = None;
        let mut last = lhs;
        for (observable, parity) in edge_measurements
            .get(&(edge.n1, edge.n2))
            .into_iter()
            .flatten()
        {
            let (vertex_type, _) = measurement_spider(*observable, false);
            let measurement = graph.add_vertex_with_phase(vertex_type, Phase::from(0_i64));
            graph.set_vars(measurement, (*parity).clone());
            graph.add_edge_with_type(last, measurement, EType::N);
            first.get_or_insert(measurement);
            last = measurement;
        }
        graph.add_edge_with_type(last, rhs, edge_type);
        anchors.insert(
            (edge.n1, edge.n2),
            WireAnchor {
                target: lhs,
                next: first.unwrap_or(rhs),
                edge_type: if first.is_some() { EType::N } else { edge_type },
            },
        );
        anchors.insert(
            (edge.n2, edge.n1),
            WireAnchor {
                target: rhs,
                next: last,
                edge_type,
            },
        );
    }

    for action in actions {
        match &action.action {
            Action::Feedback { targets, .. } if !targets.is_empty() => {
                let variable = feedback_vars[&action.ordinal];
                for target in targets {
                    insert_feedback(
                        &mut graph,
                        zx,
                        &node_vertices,
                        &mut anchors,
                        target,
                        variable,
                    );
                }
            }
            _ => {}
        }
    }

    graph.set_inputs(
        boundaries
            .inputs
            .iter()
            .map(|position| input_boundaries[position])
            .collect(),
    );
    graph.set_outputs(
        boundaries
            .outputs
            .iter()
            .map(|position| output_boundaries[position])
            .collect(),
    );
    (graph, selectives)
}

fn insert_feedback(
    graph: &mut QuizxGraph,
    zx: &ZXGraph,
    node_vertices: &[V],
    anchors: &mut FxHashMap<(usize, usize), WireAnchor>,
    target: &FeedbackTarget,
    variable: Var,
) {
    let node = zx
        .node_at(target.target)
        .expect("validated feedback target exists");
    let neighbor = zx
        .feedback_edge(target)
        .expect("validated feedback target has a wire")
        .n2;
    let vertex_types = match target.pauli {
        PauliBasis::X => [Some(VType::X), None],
        PauliBasis::Y => [Some(VType::Z), Some(VType::X)],
        PauliBasis::Z => [Some(VType::Z), None],
    };
    for vertex_type in vertex_types.into_iter().flatten() {
        let anchor = *anchors
            .get(&(node.id, neighbor))
            .expect("every original edge has anchors in both directions");
        graph.remove_edge(anchor.target, anchor.next);
        let correction = graph.add_vertex_with_phase(vertex_type, Phase::from(0_i64));
        graph.set_vars(correction, Parity::single(variable));
        graph.add_edge_with_type(anchor.target, correction, EType::N);
        graph.add_edge_with_type(correction, anchor.next, anchor.edge_type);
        anchors.insert(
            (node.id, neighbor),
            WireAnchor {
                target: anchor.target,
                next: correction,
                edge_type: EType::N,
            },
        );

        // If this was still a direct original edge, its reverse anchor also
        // starts at the vertex just replaced. Keep both endpoint frontiers on
        // the same subdivided wire. Once either side already has an inserted
        // spider, its first edge is unaffected by insertions from the other.
        let reverse = anchors
            .get_mut(&(neighbor, node.id))
            .expect("every original edge has a reverse anchor");
        if reverse.next == anchor.target {
            reverse.next = correction;
            reverse.edge_type = anchor.edge_type;
        }
    }
    debug_assert_eq!(node_vertices[node.id], anchors[&(node.id, neighbor)].target);
}

fn is_state_boundary(zx: &ZXGraph, node: usize) -> bool {
    let position = zx.nodes()[node].pos;
    let neighbors = zx.neighbors(node).unwrap_or_default();
    !neighbors.is_empty()
        && neighbors
            .iter()
            .all(|&neighbor| zx.nodes()[neighbor].pos.z > position.z)
}

fn measurement_spider(pauli: PauliBasis, is_state: bool) -> (VType, Phase) {
    match pauli {
        PauliBasis::X => (VType::Z, Phase::from(0_i64)),
        PauliBasis::Z => (VType::X, Phase::from(0_i64)),
        PauliBasis::Y if is_state => (VType::Z, Phase::from((1_i64, 2_i64))),
        PauliBasis::Y => (VType::Z, Phase::from((-1_i64, 2_i64))),
    }
}

fn sorted_pair(lhs: usize, rhs: usize) -> (usize, usize) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

#[cfg(test)]
mod tests {
    use quizx::circuit::Circuit;

    use super::*;
    use crate::{Basis, Block, BlockKind, BranchArm, CubeKind, Direction, MeasureTarget, Pipe};

    fn circuit_graph(qubits: usize, gates: &[(&str, Vec<usize>)]) -> QuizxGraph {
        let mut circuit = Circuit::new(qubits);
        for (gate, operands) in gates {
            circuit.add_gate(gate, operands.clone());
        }
        circuit.to_graph()
    }

    fn identity_source() -> BlockGraph {
        let mut source = BlockGraph::new();
        source.add_block(Block::new(IVec3::new(0, 0, 0), BlockKind::Port));
        source.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        source.add_block(Block::new(IVec3::new(0, 0, 2), BlockKind::Port));
        source.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS));
        source.add_pipe(Pipe::new(IVec3::new(0, 0, 1), Direction::ZPLUS));
        source
    }

    fn add_test_branch(source: &mut BlockGraph, name: impl Into<String>, past: IVec3) -> IVec3 {
        let target = past + IVec3::Z;
        source.add_block(Block::new(past, BlockKind::Cube(CubeKind::ZXZ)));
        let arm = |kind| {
            BranchArm::new(
                vec![Block::new(target, kind)],
                vec![Pipe::new(past, Direction::ZPLUS)],
            )
        };
        source
            .try_add_branch_region(
                name,
                arm(BlockKind::Measurement(Basis::X)),
                arm(BlockKind::Cube(CubeKind::ZXZ)),
            )
            .expect("matching terminal arms")
    }

    fn conditional_scalar_source() -> BlockGraph {
        let past = IVec3::new(0, 0, 0);
        let target = IVec3::new(0, 0, 1);
        let controller = IVec3::new(2, 0, 0);
        let mut source = BlockGraph::new();
        source.add_block(Block::new(controller, BlockKind::Cube(CubeKind::ZXZ)));
        assert_eq!(add_test_branch(&mut source, "b0", past), target);
        source
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(controller),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: Expr::Var("m".into()),
                },
            ])
            .expect("conditional scalar family validates");
        source
    }

    #[test]
    fn cancelled_verification_keeps_its_typed_cause() {
        let source = crate::GalleryItem::ThreeBitAdder.build().flatten().unwrap();
        let token = crate::CancellationToken::new();
        let error = token.scope(|| {
            token.cancel();
            LogicalVerifier::new(&source).unwrap_err()
        });
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !cause.is::<crate::ComputationCancelled>() {
            cause = cause
                .source()
                .expect("verifier preserves cancellation cause");
        }
        LogicalVerifier::new(&identity_source()).unwrap();
    }

    #[test]
    fn identity_wire_matches_quizx_identity() {
        let source = identity_source();
        let verifier = LogicalVerifier::new(&source).unwrap();
        let (branch, diagram) = verifier.instantiate(&BranchAssignment::new()).unwrap();
        assert_eq!(branch.status(), BranchStatus::Instantiated);
        assert!(diagram.is_some());
        let expected = circuit_graph(1, &[]);
        let report = verifier.verify(&expected, 1, 0).unwrap();

        assert_eq!(report.verified, 1);
    }

    #[test]
    fn multiplex_port_is_a_three_leg_z_spider() {
        let mut source = BlockGraph::new();
        source.add_block(
            Block::new(IVec3::ZERO, BlockKind::Port)
                .with_port_role(PortRole::Multiplex)
                .unwrap(),
        );
        source.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        source.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let verifier = LogicalVerifier::new(&source).unwrap();
        assert_eq!(verifier.boundaries.inputs, [IVec3::ZERO]);
        assert_eq!(verifier.boundaries.outputs, [IVec3::ZERO]);
        let input = verifier.symbolic.inputs()[0];
        let output = verifier.symbolic.outputs()[0];
        assert_ne!(input, output);
        let spider = verifier.symbolic.neighbors(input).next().unwrap();
        assert_eq!(verifier.symbolic.vertex_type(spider), VType::Z);
        assert_eq!(verifier.symbolic.degree(spider), 3);
        assert!(verifier.symbolic.neighbors(spider).any(|v| v == output));
    }

    #[test]
    fn conditional_source_selects_a_projected_static_verifier() {
        let verifier = LogicalVerifier::new(&conditional_scalar_source()).unwrap();
        let structural = verifier
            .structural
            .as_ref()
            .expect("structural source caches projected verifiers");
        assert_eq!(structural.arms.lock().unwrap().len(), 1);

        let mut vertex_type_counts = Vec::new();
        for value in [false, true] {
            let assignment = BTreeMap::from([("m".to_string(), value)]);
            let (branch, diagram) = verifier.instantiate(&assignment).unwrap();
            assert_eq!(branch.status(), BranchStatus::Instantiated);
            assert_eq!(branch.classical_state().get("m"), Some(&value));
            let diagram = diagram.expect("selected static arm has a diagram");
            vertex_type_counts.push((
                diagram
                    .vertex_vec()
                    .iter()
                    .filter(|&&vertex| diagram.vertex_type(vertex) == VType::X)
                    .count(),
                diagram
                    .vertex_vec()
                    .iter()
                    .filter(|&&vertex| diagram.vertex_type(vertex) == VType::Z)
                    .count(),
            ));
        }
        assert_ne!(
            vertex_type_counts[0], vertex_type_counts[1],
            "authored false and true arms select different static diagrams"
        );
        assert_eq!(structural.arms.lock().unwrap().len(), 2);
    }

    #[test]
    fn verifier_limit_counts_reachable_tuples_not_branch_sites() {
        let controller = IVec3::new(20, 0, 0);
        let mut source = BlockGraph::new();
        source.add_block(Block::new(controller, BlockKind::Cube(CubeKind::ZXZ)));
        let mut actions = vec![Action::Measure {
            target: MeasureTarget::Node(controller),
            name: "m".into(),
        }];
        for x in 0..9 {
            let past = IVec3::new(2 * x, 0, 0);
            let target = add_test_branch(&mut source, format!("b{x}"), past);
            actions.push(Action::Branch {
                target,
                condition: Expr::Var("m".into()),
            });
        }
        source.set_actions(actions).unwrap();

        let verifier = LogicalVerifier::new(&source).unwrap();
        for value in [false, true] {
            verifier
                .instantiate(&BTreeMap::from([("m".to_owned(), value)]))
                .unwrap();
        }
        assert_eq!(verifier.structural.unwrap().arms.lock().unwrap().len(), 2);
    }

    #[test]
    fn verifier_projection_cache_is_bounded_without_limiting_branch_count() {
        let mut source = BlockGraph::new();
        let mut actions = Vec::new();
        for x in 0..9 {
            let x = 3 * x;
            let past = IVec3::new(x, 0, 0);
            let controller = IVec3::new(x, 2, 0);
            source.add_block(Block::new(controller, BlockKind::Cube(CubeKind::ZXZ)));
            let target = add_test_branch(&mut source, format!("b{x}"), past);
            let name = format!("m{x}");
            actions.push(Action::Measure {
                target: MeasureTarget::Node(controller),
                name: name.clone(),
            });
            actions.push(Action::Branch {
                target,
                condition: Expr::Var(name),
            });
        }
        source.set_actions_lenient(actions).unwrap();

        let verifier = LogicalVerifier::new(&source).unwrap();
        assert_eq!(
            verifier
                .structural
                .as_ref()
                .unwrap()
                .arms
                .lock()
                .unwrap()
                .len(),
            1
        );
        for mask in 0..70 {
            let assignment = (0..9)
                .map(|bit| (format!("m{}", 3 * bit), mask & (1 << bit) != 0))
                .collect();
            verifier.instantiate(&assignment).unwrap();
        }
        assert!(
            verifier
                .structural
                .as_ref()
                .unwrap()
                .arms
                .lock()
                .unwrap()
                .len()
                <= MAX_CACHED_STRUCTURAL_PROJECTIONS
        );
    }

    #[test]
    fn named_selective_phases_survive_both_fills() {
        let source = crate::GalleryItem::TComparison
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let verifier = LogicalVerifier::with_internal_measurements(&source).unwrap();
        let correction = verifier.correction.as_ref().unwrap();
        let site = IVec3::new(1, 0, 2);
        let sampled = correction.sampled_vars().collect::<BTreeSet<_>>();
        assert!(
            correction.measurement_vars().iter().any(|(key, parity)| {
                matches!(key, MeasurementKey::Node { pos, .. } if *pos == site)
                    && parity.iter().any(|variable| sampled.contains(&variable))
            }),
            "named selective must carry a nontrivial unnamed compensation"
        );
        for cmp in [false, true] {
            let assignment =
                BTreeMap::from([("cmp".to_owned(), cmp), ("parity".to_owned(), false)]);
            let resolved = verifier
                .resolve_assignment(&assignment)
                .unwrap()
                .correction
                .unwrap();
            let zx = resolved.basis.zx_graph();
            let readouts =
                named_readouts(&resolved.basis, &named_records(zx, &verifier.actions)).unwrap();
            for (name, row) in &readouts {
                let actual = resolved
                    .phases
                    .iter()
                    .filter(|(key, _)| stabilizer_supports_measurement(row, key))
                    .fold(Parity::default(), |parity, (_, term)| &parity + term);
                assert_eq!(
                    actual,
                    Parity::single(verifier.measurement_vars[name]),
                    "cmp={cmp} {name}"
                );
            }
        }
    }

    #[test]
    fn unnamed_phases_preserve_the_supplied_named_measurement_parities() {
        let source = crate::GalleryItem::CCZInjectedAnd
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = source.to_zx_graph().unwrap();
        let verifier = LogicalVerifier::with_internal_measurements(&source).unwrap();
        let basis = crate::RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
        let readouts = named_readouts(&basis, &named_records(&zx, &verifier.actions)).unwrap();
        let correction = verifier.correction.as_ref().unwrap();
        assert!(correction.sampled_vars().next().is_some());
        for (name, readout) in &readouts {
            let actual = correction
                .measurement_vars()
                .iter()
                .filter(|(key, _)| stabilizer_supports_measurement(readout, key))
                .fold(Parity::default(), |parity, (_, contribution)| {
                    &parity + contribution
                });
            assert_eq!(
                actual,
                Parity::single(verifier.measurement_vars[name]),
                "every unnamed phase assignment must preserve action input {name}"
            );
        }
    }

    #[test]
    fn basis_flipped_selective_matches_each_ordered_cap_map() {
        for spelling in ["XY", "YX", "XZ", "ZX", "YZ", "ZY"] {
            let source = crate::parse_inline_graph(&format!(
                "BLOG 1.0\nmodule main {{\nin control\nin q: data = 0\n0: Port [0,0,0]\n1: {spelling} [0,0,1]\n0 -> +Z\nresolve 1 if control\n}}\n"
            )).unwrap().materialize_flat_graph().unwrap();
            let flipped = source.flip_xz_basis().unwrap();
            assert_eq!(
                flipped.flip_xz_basis().unwrap().to_blog_text(),
                source.to_blog_text()
            );
            let verifier = LogicalVerifier::new(&flipped).unwrap();
            let choices = spelling.chars().collect::<Vec<_>>();
            for control in [false, true] {
                let before = choices[usize::from(!control)];
                let expected_source = BlockGraph::from_blog_text(&format!(
                    "BLOG 1.0\n0: Port [0,0,0]\n1: {before} [0,0,1]\n0 -> +Z\n"
                ))
                .unwrap();
                let (_, expected) = LogicalVerifier::new(&expected_source)
                    .unwrap()
                    .instantiate(&BTreeMap::new())
                    .unwrap();
                let mut dual = circuit_graph(1, &[("h", vec![0])]);
                dual.plug(&expected.unwrap());
                verifier
                    .verify_branch(&dual, &BTreeMap::from([("control".into(), control)]))
                    .unwrap();
            }
            let BlockKind::Selective(kind) = source.get_block(IVec3::Z).unwrap().kind() else {
                unreachable!()
            };
            let axis = match kind {
                crate::SelectiveKind::XY => Some(PauliBasis::X),
                crate::SelectiveKind::YZ => Some(PauliBasis::Z),
                crate::SelectiveKind::XZ => None,
            };
            if let Some(pauli) = axis {
                let mut corrected = source.clone();
                corrected
                    .add_action(Action::Feedback {
                        targets: vec![FeedbackTarget {
                            target: IVec3::Z,
                            pauli,
                            direction: None,
                        }],
                        condition: None,
                    })
                    .unwrap();
                let once = corrected.flip_xz_basis().unwrap();
                assert!(
                    !once
                        .actions()
                        .iter()
                        .any(|action| matches!(action, Action::Feedback { .. })),
                    "adjacent equal corrections cancel as a whole, without empty feedback"
                );
                assert_eq!(
                    once.flip_xz_basis().unwrap().to_blog_text(),
                    corrected.to_blog_text()
                );
            }
        }
    }

    #[test]
    fn basis_flip_dualizes_fixed_resources_through_their_pipes() {
        for (kind, gate) in [(BlockKind::T, "t"), (BlockKind::Y, "s")] {
            for hadamard in [false, true] {
                for reversed in [false, true] {
                    let mut source = identity_source();
                    source.set_block_kind(IVec3::ZERO, kind).unwrap();
                    source.remove_pipe(IVec3::ZERO, IVec3::Z);
                    let mut pipe = Pipe::new(
                        if reversed { IVec3::Z } else { IVec3::ZERO },
                        if reversed {
                            Direction::ZMINUS
                        } else {
                            Direction::ZPLUS
                        },
                    )
                    .with_tag("resource")
                    .unwrap();
                    if hadamard {
                        pipe = pipe.with_hadamard();
                    }
                    source.add_pipe(pipe.clone());
                    let flipped = source.flip_xz_basis().unwrap();
                    let actual = flipped.get_pipe(IVec3::ZERO, IVec3::Z).unwrap();
                    assert_eq!(
                        (actual.src(), actual.dir(), actual.tag()),
                        (pipe.src(), pipe.dir(), pipe.tag())
                    );
                    assert_eq!(actual.is_hadamard(), !hadamard);
                    let mut gates = vec![("h", vec![0]), (gate, vec![0])];
                    if !hadamard {
                        gates.push(("h", vec![0]));
                    }
                    let mut expected = circuit_graph(1, &gates);
                    expected.plug_inputs(&[quizx::graph::BasisElem::Z0]);
                    LogicalVerifier::new(&flipped)
                        .unwrap()
                        .verify_branch(&expected, &BTreeMap::new())
                        .unwrap();
                    assert_eq!(
                        flipped.flip_xz_basis().unwrap().to_blog_text(),
                        source.to_blog_text()
                    );
                }
            }
        }
    }

    #[test]
    fn basis_flip_preserves_native_t_comparison_postselection() {
        let source = crate::GalleryItem::TComparison
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let flipped = source.flip_xz_basis().unwrap();
        assert!(
            flipped
                .actions()
                .iter()
                .any(|action| action == source.actions().last().unwrap())
        );
        for (index, mut graph) in [source, flipped].into_iter().enumerate() {
            let actions = graph
                .actions()
                .into_iter()
                .filter(|action| !matches!(action, Action::DiscardIf(_)))
                .collect();
            graph.set_actions(actions).unwrap();
            let verifier = LogicalVerifier::new(&graph).unwrap();
            for cmp in [false, true] {
                for parity in [false, true] {
                    let (_, branch) = verifier
                        .instantiate(&BTreeMap::from([
                            ("cmp".into(), cmp),
                            ("parity".into(), parity),
                        ]))
                        .unwrap();
                    assert_eq!(
                        tensor_is_nonzero(&contract(&branch.unwrap())),
                        !parity,
                        "variant={index} cmp={cmp} parity={parity}"
                    );
                }
            }
        }
    }

    #[test]
    fn isolated_feedback_returns_missing_wire_error() {
        let mut source = BlockGraph::new();
        source.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        source.set_actions_unchecked(vec![Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::X,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: None,
        }]);
        assert!(matches!(
            source.validate(),
            Err(crate::BlockGraphError::Stabilizer(
                crate::StabilizerError::FeedbackTargetWithoutWire { .. }
            ))
        ));
        assert!(matches!(
            LogicalVerifier::new(&source),
            Err(VerifyLogicalError::Graph(crate::BlockGraphError::Stabilizer(
                crate::StabilizerError::FeedbackTargetWithoutWire { target, pauli: PauliBasis::X }
            ))) if target == IVec3::ZERO
        ));
    }

    #[test]
    fn unconditional_feedback_is_a_literal_pauli_on_the_wire() {
        let target = IVec3::new(0, 0, 1);
        let mut source = identity_source();
        source
            .set_actions(vec![Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::X,
                    target,
                    direction: None,
                }],
                condition: None,
            }])
            .unwrap();

        let verifier = LogicalVerifier::new(&source).unwrap();
        verifier
            .verify(&circuit_graph(1, &[("x", vec![0])]), 1, 0)
            .unwrap();
    }

    #[test]
    fn external_inputs_control_feedback_and_are_sampled() {
        let mut source = identity_source();
        source
            .set_actions_with_inputs(
                vec![Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::X,
                        target: IVec3::Z,
                        direction: None,
                    }],
                    condition: Some(Expr::Var("enabled".into())),
                }],
                ["enabled".into()],
            )
            .unwrap();
        let verifier = LogicalVerifier::new(&source).unwrap();
        for enabled in [false, true] {
            let expected = if enabled {
                circuit_graph(1, &[("x", vec![0])])
            } else {
                circuit_graph(1, &[])
            };
            let branch = verifier
                .verify_branch(&expected, &BTreeMap::from([("enabled".into(), enabled)]))
                .unwrap();
            assert_eq!(branch.status(), BranchStatus::Verified);
            assert_eq!(branch.classical_state()["enabled"], enabled);
        }
        assert!(matches!(
            verifier.instantiate(&BTreeMap::new()),
            Err(VerifyLogicalError::MissingInputValue(name)) if name == "enabled"
        ));
        let sampled = (0..16)
            .map(|seed| verifier.sample_with_seed(seed).unwrap().0.classical_state()["enabled"])
            .collect::<BTreeSet<_>>();
        assert_eq!(sampled, BTreeSet::from([false, true]));
    }

    #[test]
    fn external_inputs_select_structural_arms() {
        let mut source = BlockGraph::new();
        let target = add_test_branch(&mut source, "b0", IVec3::ZERO);
        source
            .set_actions_with_inputs(
                vec![Action::Branch {
                    target,
                    condition: Expr::Var("enabled".into()),
                }],
                ["enabled".into()],
            )
            .unwrap();
        let verifier = LogicalVerifier::new(&source).unwrap();
        for enabled in [false, true] {
            let (branch, diagram) = verifier
                .instantiate(&BTreeMap::from([("enabled".into(), enabled)]))
                .unwrap();
            assert_eq!(branch.classical_state()["enabled"], enabled);
            assert!(diagram.is_some());
        }
    }

    #[test]
    fn repeated_t_parities_keep_the_named_branch_frame() {
        let expected = circuit_graph(1, &[("t", vec![0])]);
        let expected_tensor = contract(&expected);
        for layers in 1..=3 {
            let position = |x, z| IVec3::new(x, 0, z);
            let mut source = BlockGraph::new();
            source.add_block(Block::new(position(0, -1), BlockKind::T));
            source.add_block(Block::new(position(1, -1), BlockKind::Port));
            source.add_block(Block::new(
                position(0, layers),
                BlockKind::Selective(SelectiveKind::XY),
            ));
            source.add_block(Block::new(position(1, layers), BlockKind::Port));
            for layer in 0..layers {
                for x in 0..=1 {
                    source.add_block(Block::new(
                        position(x, layer),
                        BlockKind::Cube(CubeKind::ZXZ),
                    ));
                }
                source.add_pipe(Pipe::new(position(0, layer), crate::Direction::XPLUS));
            }
            for x in 0..=1 {
                for layer in -1..layers {
                    let pipe = Pipe::new(position(x, layer), Direction::ZPLUS);
                    source.add_pipe(if layer == -1 || layer == layers - 1 {
                        pipe.with_hadamard()
                    } else {
                        pipe
                    });
                }
            }
            source
                .set_actions(vec![
                    Action::Measure {
                        target: MeasureTarget::Edge {
                            src: IVec3::ZERO,
                            dir: Direction::XPLUS,
                        },
                        name: "m".into(),
                    },
                    Action::Resolve {
                        target: position(0, layers),
                        condition: Expr::Not(Box::new(Expr::Var("m".into()))),
                    },
                ])
                .unwrap();

            // Pure ZX gives the independent reference: every repeated raw
            // parity agrees, and both all-zero and all-one branches implement T.
            let mut raw = source.clone();
            let mut actions = raw.actions();
            for layer in 1..layers {
                actions.push(Action::Measure {
                    target: MeasureTarget::Edge {
                        src: position(0, layer),
                        dir: Direction::XPLUS,
                    },
                    name: format!("raw{layer}"),
                });
            }
            raw.set_actions_deferred(actions).unwrap();
            let raw_verifier = LogicalVerifier::new(&raw).unwrap();
            for value in [false, true] {
                let mut assignment = BranchAssignment::from([("m".into(), value)]);
                assignment.extend((1..layers).map(|layer| (format!("raw{layer}"), value)));
                assert_eq!(
                    raw_verifier
                        .verify_branch(&expected, &assignment)
                        .unwrap()
                        .status(),
                    BranchStatus::Verified
                );
            }

            let corrected = LogicalVerifier::with_internal_measurements(&source).unwrap();
            let mut verified = BTreeSet::new();
            for seed in 0..128 {
                let (branch, diagram) = corrected.sample_with_seed(seed).unwrap();
                let branch = compare_branch(branch, diagram, &expected_tensor).unwrap();
                if branch.status() == BranchStatus::Verified {
                    verified.insert(branch.classical_state()["m"]);
                }
            }
            assert_eq!(verified, BTreeSet::from([false, true]));
        }
    }

    #[test]
    fn rejected_assignment_stops_without_contracting() {
        let source = crate::GalleryItem::TComparison
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let verifier = LogicalVerifier::new(&source).unwrap();
        let assignment = BTreeMap::from([("cmp".to_string(), false), ("parity".to_string(), true)]);

        let (branch, diagram) = verifier.instantiate(&assignment).unwrap();

        assert_eq!(branch.status(), BranchStatus::Rejected);
        assert_eq!(branch.classical_state().get("cmp"), Some(&false));
        assert_eq!(branch.classical_state().get("parity"), Some(&true));
        assert!(diagram.is_none());
    }

    #[test]
    fn resolve_can_read_a_presampled_future_measurement() {
        let selective = IVec3::new(0, 0, 0);
        let measured = IVec3::new(0, 0, 2);
        let mut source = BlockGraph::new();
        source.add_block(Block::new(
            selective,
            BlockKind::Selective(SelectiveKind::XZ),
        ));
        // The port makes both cap arms reachable.
        source.add_block(Block::new(IVec3::new(0, 0, -1), BlockKind::Port));
        source.add_block(Block::new(measured, BlockKind::Cube(CubeKind::ZXZ)));
        source.add_pipe(Pipe::new(IVec3::new(0, 0, -1), Direction::ZPLUS));
        source
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(measured),
                    name: "future".to_string(),
                },
                Action::Resolve {
                    target: selective,
                    condition: Expr::Var("future".to_string()),
                },
            ])
            .unwrap();
        let verifier = LogicalVerifier::new(&source).unwrap();

        let (branch, diagram) = verifier
            .instantiate(&BTreeMap::from([("future".to_string(), true)]))
            .unwrap();

        assert_eq!(branch.status(), BranchStatus::Instantiated);
        assert_eq!(branch.classical_state().get("future"), Some(&true));
        assert!(diagram.is_some());
    }
}
