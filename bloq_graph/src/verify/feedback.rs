//! Bounded synthesis of missing Pauli feedback from QASM and a realized graph.

use std::collections::{BTreeMap, BTreeSet};

use glam::IVec3;
use quizx::fscalar::{FScalar, Zero};
use quizx::graph::{GraphLike, VType};
use quizx::tensor::ToTensor;

use crate::zx::{CoeffVec, solve_coeff_combination};
use crate::{Action, BlockGraph, Expr, FeedbackTarget, Pauli, PauliBasis};

use super::diagram::eval_expr;
use super::qasm::QasmCircuit;
use super::{
    BoundaryOrder, BranchAssignment, BranchStatus, LogicalVerifier, QasmError, QuizxGraph,
};

/// Correspondence and resource bounds for feedback inference.
///
/// QASM is interpreted as a component: unreset wires are arbitrary inputs,
/// not implicitly initialized to zero. By default its surviving wires map in
/// declaration order to the verifier's boundary order. Single-bit classical
/// registers match same-named BLOG measurements (`m[0]` matches `m`).
#[derive(Debug, Clone)]
pub struct FeedbackOptions {
    boundaries: Option<BoundaryOrder>,
    measurements: BTreeMap<String, Expr>,
    fixed_measurements: BTreeMap<String, bool>,
    discarded_measurements: BTreeSet<String>,
    preparation: Option<String>,
    discard_if: Option<Expr>,
    max_branches: usize,
    max_classes: usize,
    max_tensor_qubits: usize,
    max_trials: usize,
    internal_sites: Option<Vec<IVec3>>,
}

impl Default for FeedbackOptions {
    fn default() -> Self {
        Self {
            boundaries: None,
            measurements: BTreeMap::new(),
            fixed_measurements: BTreeMap::new(),
            discarded_measurements: BTreeSet::new(),
            preparation: None,
            discard_if: None,
            max_branches: 256,
            max_classes: 256,
            max_tensor_qubits: 20,
            max_trials: 4096,
            internal_sites: None,
        }
    }
}

impl FeedbackOptions {
    /// Bounds the fallback search over internal Pauli actions. Zero requests
    /// output feedback only. Each trial checks all named/anonymous branches.
    #[must_use]
    pub fn with_search_limit(mut self, trials: usize) -> Self {
        self.max_trials = trials;
        self
    }

    /// Restricts the internal search to these authored block positions.
    /// By default all unconditional, unambiguous wire anchors are considered.
    #[must_use]
    pub fn with_internal_sites(mut self, sites: Vec<IVec3>) -> Self {
        self.internal_sites = Some(sites);
        self
    }

    /// Sets graph boundaries in the corresponding QASM input/output order.
    #[must_use]
    pub fn with_boundaries(mut self, boundaries: BoundaryOrder) -> Self {
        self.boundaries = Some(boundaries);
        self
    }

    /// Binds an indexed QASM result to an expression over named graph records.
    /// This explicitly supplies any parity or polarity convention.
    #[must_use]
    pub fn with_measurement(mut self, bit: impl Into<String>, value: Expr) -> Self {
        self.measurements.insert(bit.into(), value);
        self
    }

    /// Explicitly postselects a source result instead of mapping it to a graph
    /// record. Useful for gallery kernels with projected boundary effects.
    #[must_use]
    pub fn with_fixed_measurement(mut self, bit: impl Into<String>, value: bool) -> Self {
        self.fixed_measurements.insert(bit.into(), value);
        self
    }

    /// Forgets a source result after executing its consumers. Its Kraus
    /// branches are summed, preserving the measurement's quantum state update.
    #[must_use]
    pub fn with_discarded_measurement(mut self, bit: impl Into<String>) -> Self {
        self.discarded_measurements.insert(bit.into());
        self
    }

    /// Supplies an unconditional preparation component whose ordered outputs
    /// feed all graph inputs. Its unreset inputs must match the source inputs.
    /// Resource states are never inferred from port tags.
    #[must_use]
    pub fn with_input_preparation(mut self, qasm: impl Into<String>) -> Self {
        self.preparation = Some(qasm.into());
        self
    }

    /// Declares the desired rejection predicate over graph records. Graph
    /// `DiscardIf` behavior must agree; inference never silently drops a shot.
    #[must_use]
    pub fn with_discard_if(mut self, condition: Expr) -> Self {
        self.discard_if = Some(condition);
        self
    }

    /// Bounds named assignments, anonymous boundary classes per assignment,
    /// and the largest intermediate tensor's number of binary axes.
    #[must_use]
    pub fn with_limits(mut self, branches: usize, classes: usize, tensor_qubits: usize) -> Self {
        self.max_branches = branches;
        self.max_classes = classes;
        self.max_tensor_qubits = tensor_qubits;
        self
    }
}

/// A source/realization mismatch, unsupported contract, or inference limit.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FeedbackInferenceError {
    /// Neutral QASM parsing or validation failed.
    #[error("{0}")]
    Qasm(#[from] QasmError),
    /// Block-graph construction or validation failed.
    #[error("{0}")]
    Graph(#[from] crate::BlockGraphError),
    /// Logical map verification failed.
    #[error("{0}")]
    Verify(#[from] super::VerifyLogicalError),
    /// Runtime-basis construction or update failed.
    #[error("{0}")]
    RuntimeBasis(#[from] crate::RuntimeBasisError),
    /// Output correction surfaces could not be solved.
    #[error("{0}")]
    OutputCorrection(#[from] crate::OutputCorrectionError),
    /// Caller-supplied source/graph correspondence is invalid.
    #[error("invalid feedback correspondence: {0}")]
    Binding(String),
    /// Inference exceeded an explicit resource limit.
    #[error("feedback inference exceeded the {phase} bound (size {observed}, limit {limit})")]
    Limit {
        /// Bounded inference phase or resource name.
        phase: &'static str,
        /// Resource amount required or observed.
        observed: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Anonymous outcomes do not span the expected boundary space.
    #[error(
        "anonymous outcome certificate is incomplete: boundary rank {rank}, expected {expected}"
    )]
    IncompleteCertificate {
        /// Observed independent boundary rank.
        rank: usize,
        /// Required boundary rank.
        expected: usize,
    },
    /// Source and realization disagree on whether a branch is rejected.
    #[error("source and graph reject different branches: {assignment:?}")]
    RejectionMismatch {
        /// Named outcome assignment selecting the branch.
        assignment: BranchAssignment,
    },
    /// Source and realization disagree on whether a branch map is nonzero.
    #[error("source and graph have different nonzero branch support: {assignment:?}")]
    SupportMismatch {
        /// Named outcome assignment selecting the branch.
        assignment: BranchAssignment,
    },
    /// No output Pauli relates one realization branch to its source branch.
    #[error("no output Pauli makes this branch equal to the source: {assignment:?}")]
    NoOutputPauli {
        /// Named outcome assignment selecting the branch.
        assignment: BranchAssignment,
    },
    /// Internal Pauli trials found no correction satisfying all branches.
    #[error("no valid Pauli feedback found in {trials} internal trials")]
    NoPauliFeedback {
        /// Number of internal candidate trials performed.
        trials: usize,
    },
    /// Rechecking the emitted feedback disagreed with the inferred correction.
    #[error("emitted feedback does not implement the inferred output correction")]
    EmissionMismatch,
    /// The comparison has no admitted nonzero branch.
    #[error("the comparison contains no admitted nonzero branch")]
    EmptyDomain,
    /// QuiZX tensor arithmetic produced a NaN or infinity.
    #[error("QuiZX tensor arithmetic produced a non-finite scalar")]
    NonFiniteTensor,
    /// Checked dyadic arithmetic overflowed or lost exactness.
    #[error("dyadic comparison exceeded its checked arithmetic range or lost precision")]
    ArithmeticLimit,
}

/// Derives additional Pauli feedback for an existing realization.
///
/// Existing actions remain in force. The returned actions are checked on a
/// clone before return; the input graph is unchanged. Use
/// [`infer_feedback_with`] for explicit wire/record/resource correspondence.
///
/// This checks every bounded named branch and every certified anonymous
/// boundary class in the logical ZX model. Discarded outcomes are summed as
/// CP maps, compared up to a branch scalar using QuiZX's dyadic arithmetic.
/// Choi arithmetic and projective cross-products are checked using `i128`
/// dyadic coefficients. Earlier QuiZX simplification/contraction still uses
/// floating-point coefficients: this is numerical logical verification, not
/// an unrestricted exact-arithmetic certificate.
/// Relative probabilities between named branches and noisy physical execution
/// require separate verification. It does not infer missing `Resolve`
/// choices or promise a polynomial algorithm for arbitrary non-Clifford maps.
///
/// ```
/// use bloq_graph::{Action, GalleryItem, infer_feedback};
/// let mut graph = GalleryItem::S.build().materialize_root_graph().expect("gallery flat projection");
/// graph.set_actions(graph.actions().into_iter()
///     .filter(|action| !matches!(action, Action::Feedback { .. })).collect())?;
/// let mut actions = graph.actions();
/// actions.extend(infer_feedback("OPENQASM 2.0; qreg q[1]; s q;", &graph)?);
/// graph.set_actions(actions)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns parsing, binding, support, resource-limit, or verification errors.
pub fn infer_feedback(
    qasm: &str,
    graph: &BlockGraph,
) -> Result<Vec<Action>, FeedbackInferenceError> {
    infer_feedback_with(qasm, graph, &FeedbackOptions::default())
}

/// Derives missing feedback under an explicit component contract.
///
/// Unmatched source measurement results are errors, never silently fixed to
/// zero. Graph-only measurement names are implementation records and are all
/// checked. Only leading resets and measurements last on their own wires are
/// supported; later conditional Paulis on other live wires are allowed.
///
/// # Errors
///
/// Rejects missing bindings, mismatched support/rejection, incomplete anonymous
/// certificates, resource limits, non-Pauli differences, or actions that fail
/// graph/readout validation or re-verification after insertion.
pub fn infer_feedback_with(
    qasm: &str,
    graph: &BlockGraph,
    options: &FeedbackOptions,
) -> Result<Vec<Action>, FeedbackInferenceError> {
    let source = QasmCircuit::parse(qasm)?;
    let names = graph
        .action_graph()
        .inputs()
        .map(str::to_owned)
        .chain(
            graph
                .actions()
                .into_iter()
                .filter_map(|action| match action {
                    Action::Measure { name, .. } => Some(name),
                    _ => None,
                }),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let count = checked_count(names.len(), options.max_branches, "named branches")?;
    check_branch_limit(graph, options.max_branches)?;
    let verifier = make_verifier(graph, options)?;
    let preparation = options
        .preparation
        .as_deref()
        .map(QasmCircuit::parse)
        .transpose()?;
    if let Some(preparation) = &preparation {
        if preparation.measurements.iter().any(Option::is_some) {
            return Err(FeedbackInferenceError::Binding(
                "input preparation must have no measurements".into(),
            ));
        }
        if preparation.output_count() != verifier.boundaries().inputs().len()
            || preparation.input_count() != source.input_count()
        {
            return Err(FeedbackInferenceError::Binding(
                "input preparation does not match source/graph input widths".into(),
            ));
        }
    } else if source.input_count() != verifier.boundaries().inputs().len() {
        return Err(FeedbackInferenceError::Binding("source and graph input widths differ; supply ordered boundaries and any resource preparation".into()));
    }
    if source.output_count() != verifier.boundaries().outputs().len() {
        return Err(FeedbackInferenceError::Binding(
            "source and graph output widths differ".into(),
        ));
    }
    // Reject oversized Choi matrices before contracting and retaining their
    // Kraus tensors. The boundary correspondence gives both sides this width.
    let channel_axes = 2 * (source.input_count() + source.output_count());
    if channel_axes > options.max_tensor_qubits || channel_axes >= usize::BITS as usize {
        return Err(FeedbackInferenceError::Limit {
            phase: "channel tensor qubits",
            observed: channel_axes,
            limit: options.max_tensor_qubits,
        });
    }
    let bindings = source_bindings(&source, &names, options)?;
    if let Some(condition) = &options.discard_if {
        validate_expr(condition, &names)?;
    }
    let preparation = preparation.map(|source| source.instantiate(&[]));
    let discarded = bindings
        .iter()
        .enumerate()
        .filter_map(|(bit, binding)| matches!(binding, BitBinding::Discarded).then_some(bit))
        .collect::<Vec<_>>();
    let source_count = checked_count(
        discarded.len(),
        options.max_classes,
        "discarded source outcomes",
    )?;
    let expected = (0..count)
        .map(|mask| {
            let assignment = assignment(&names, mask);
            let mut values = bindings
                .iter()
                .map(|binding| match binding {
                    BitBinding::Fixed(value) => *value,
                    BitBinding::Record(expr) => eval_expr(expr, &assignment),
                    BitBinding::Discarded => false,
                })
                .collect::<Vec<_>>();
            let maps = (0..source_count)
                .map(|mask| {
                    for (index, &bit) in discarded.iter().enumerate() {
                        values[bit] = mask & (1 << index) != 0;
                    }
                    contract(source.instantiate(&values), options.max_tensor_qubits)
                })
                .collect::<Result<Vec<_>, _>>()?;
            channel(&maps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let problem = FeedbackProblem {
        names: &names,
        expected: &expected,
        preparation: preparation.as_ref(),
        options,
    };
    match problem.complete(graph, &verifier, Vec::new()) {
        Err(
            FeedbackInferenceError::NoOutputPauli { .. }
            | FeedbackInferenceError::SupportMismatch { .. }
            | FeedbackInferenceError::EmissionMismatch,
        ) if options.max_trials != 0 => problem.search(graph),
        result => result,
    }
}

struct FeedbackProblem<'a> {
    names: &'a [String],
    expected: &'a [Vec<FScalar>],
    preparation: Option<&'a QuizxGraph>,
    options: &'a FeedbackOptions,
}

impl FeedbackProblem<'_> {
    fn frames(
        &self,
        verifier: &LogicalVerifier,
    ) -> Result<Vec<Vec<Pauli>>, FeedbackInferenceError> {
        compare_branches(
            verifier,
            self.names,
            self.expected,
            self.preparation,
            self.options,
        )
    }

    fn complete(
        &self,
        graph: &BlockGraph,
        verifier: &LogicalVerifier,
        mut actions: Vec<Action>,
    ) -> Result<Vec<Action>, FeedbackInferenceError> {
        let frames = self.frames(verifier)?;
        let output = frame_actions(self.names, verifier.boundaries().outputs(), &frames);
        if output.is_empty() {
            return Ok(actions);
        }
        let mut candidate = graph.clone();
        let mut combined = candidate.actions();
        combined.extend(output.iter().cloned());
        candidate.set_actions(combined)?;
        check_branch_limit(&candidate, self.options.max_branches)?;
        let remaining = self.frames(&make_verifier(&candidate, self.options)?)?;
        if remaining.iter().flatten().any(|pauli| *pauli != Pauli::I) {
            return Err(FeedbackInferenceError::EmissionMismatch);
        }
        actions.extend(output);
        Ok(actions)
    }

    fn search(&self, graph: &BlockGraph) -> Result<Vec<Action>, FeedbackInferenceError> {
        let graphs = if graph.branch_regions()?.is_empty() {
            vec![graph.to_zx_graph()?]
        } else {
            graph
                .branch_projections_up_to(self.options.max_branches)?
                .iter()
                .map(|projection| projection.graph().to_zx_graph())
                .collect::<Result<Vec<_>, _>>()?
        };
        let zx = &graphs[0];
        let is_wire = |position| {
            graphs.iter().all(|zx| {
                zx.feedback_edge(&FeedbackTarget {
                    target: position,
                    pauli: PauliBasis::X,
                    direction: None,
                })
                .is_some()
            })
        };
        let mut sites = match &self.options.internal_sites {
            Some(sites) => {
                for &site in sites {
                    if !is_wire(site) {
                        return Err(FeedbackInferenceError::Binding(format!(
                            "{site} is not an unambiguous feedback wire in every structural arm"
                        )));
                    }
                }
                sites.clone()
            }
            None => zx
                .nodes()
                .iter()
                .filter(|node| graph.get_block(node.pos).is_some() && is_wire(node.pos))
                .map(|node| node.pos)
                .collect(),
        };
        // Resource sign mistakes have a particularly small fix at their cap.
        // Otherwise prefer late sites; the action DAG decides record causality.
        sites.sort_by_key(|position| {
            (
                zx.node_at(*position)
                    .expect("feedback sites are checked against this ZX graph")
                    .kind
                    != crate::NodeKind::Y,
                std::cmp::Reverse(position.z),
                position.to_array(),
            )
        });
        sites.dedup();
        let mut monomials = (0..self.expected.len()).collect::<Vec<_>>();
        monomials.sort_by_key(|mask| (mask.count_ones(), *mask));
        let atoms = monomials
            .into_iter()
            .flat_map(|mask| {
                let condition = join(
                    crate::BinaryOp::And,
                    self.names
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| mask & (1 << bit) != 0)
                        .map(|(_, name)| Expr::Var(name.clone()))
                        .collect(),
                );
                sites.iter().flat_map(move |&target| {
                    [PauliBasis::Z, PauliBasis::X, PauliBasis::Y].map(|pauli| Action::Feedback {
                        targets: vec![FeedbackTarget {
                            pauli,
                            target,
                            direction: None,
                        }],
                        condition: condition.clone(),
                    })
                })
            })
            .take(self.options.max_trials.saturating_add(1))
            .collect::<Vec<_>>();
        let mut trials = 0;
        // ponytail: bounded exhaustive action search; add Clifford-sign
        // deduplication only if authoring examples exhaust this budget.
        for weight in 1..=atoms.len() {
            let mut indices = (0..weight).collect::<Vec<_>>();
            loop {
                trials += 1;
                if trials > self.options.max_trials {
                    return Err(FeedbackInferenceError::Limit {
                        phase: "internal feedback trials",
                        observed: trials,
                        limit: self.options.max_trials,
                    });
                }
                let additions = indices
                    .iter()
                    .map(|&index| atoms[index].clone())
                    .collect::<Vec<_>>();
                let mut candidate = graph.clone();
                let mut actions = candidate.actions();
                actions.extend(additions.iter().cloned());
                let result = candidate
                    .set_actions(actions)
                    .map_err(FeedbackInferenceError::from)
                    .and_then(|()| check_branch_limit(&candidate, self.options.max_branches))
                    .and_then(|()| make_verifier(&candidate, self.options))
                    .and_then(|verifier| self.complete(&candidate, &verifier, additions));
                match result {
                    Ok(actions) => return Ok(actions),
                    Err(
                        error @ (FeedbackInferenceError::Limit { .. }
                        | FeedbackInferenceError::NonFiniteTensor
                        | FeedbackInferenceError::ArithmeticLimit),
                    ) => return Err(error),
                    Err(_) => {}
                }
                let Some(pivot) = (0..weight)
                    .rev()
                    .find(|&index| indices[index] < atoms.len() - weight + index)
                else {
                    break;
                };
                indices[pivot] += 1;
                for index in pivot + 1..weight {
                    indices[index] = indices[index - 1] + 1;
                }
            }
        }
        Err(FeedbackInferenceError::NoPauliFeedback { trials })
    }
}

fn make_verifier(
    graph: &BlockGraph,
    options: &FeedbackOptions,
) -> Result<LogicalVerifier, FeedbackInferenceError> {
    Ok(match &options.boundaries {
        Some(boundaries) => {
            LogicalVerifier::with_boundaries_and_internal_measurements(graph, boundaries.clone())?
        }
        None => LogicalVerifier::with_internal_measurements(graph)?,
    })
}

/// Bounds structural branches before the verifier, which validates the
/// source, builds one projection per reachable arm.
fn check_branch_limit(graph: &BlockGraph, limit: usize) -> Result<(), FeedbackInferenceError> {
    if graph.branch_regions()?.is_empty() {
        return Ok(());
    }
    let projections = graph
        .branch_assignments_up_to(limit.saturating_add(1))?
        .len();
    if projections > limit {
        return Err(FeedbackInferenceError::Limit {
            phase: "structural branches",
            observed: projections,
            limit,
        });
    }
    Ok(())
}

enum BitBinding {
    Fixed(bool),
    Record(Expr),
    Discarded,
}

fn source_bindings(
    source: &QasmCircuit,
    names: &[String],
    options: &FeedbackOptions,
) -> Result<Vec<BitBinding>, FeedbackInferenceError> {
    let bits = source.program.bits();
    for name in options
        .measurements
        .keys()
        .chain(options.fixed_measurements.keys())
        .chain(options.discarded_measurements.iter())
    {
        if !bits.contains(name) {
            return Err(FeedbackInferenceError::Binding(format!(
                "unknown QASM bit {name:?}"
            )));
        }
        if usize::from(options.measurements.contains_key(name))
            + usize::from(options.fixed_measurements.contains_key(name))
            + usize::from(options.discarded_measurements.contains(name))
            > 1
        {
            return Err(FeedbackInferenceError::Binding(format!(
                "QASM bit {name:?} has conflicting bindings"
            )));
        }
    }
    bits.iter()
        .enumerate()
        .map(|(bit, name)| {
            if options.discarded_measurements.contains(name) {
                return Ok(BitBinding::Discarded);
            }
            if let Some(value) = options.fixed_measurements.get(name) {
                return Ok(BitBinding::Fixed(*value));
            }
            if let Some(expr) = options.measurements.get(name) {
                validate_expr(expr, names)?;
                return Ok(BitBinding::Record(expr.clone()));
            }
            if !source.measurements.contains(&Some(bit)) {
                return Ok(BitBinding::Fixed(false));
            }
            let matched = names
                .iter()
                .find(|candidate| *candidate == name)
                .or_else(|| {
                    let bare = name.strip_suffix("[0]")?;
                    if bits.contains(&format!("{bare}[1]")) {
                        return None;
                    }
                    names.iter().find(|candidate| candidate.as_str() == bare)
                });
            matched
                .map(|name| BitBinding::Record(Expr::Var(name.clone())))
                .ok_or_else(|| FeedbackInferenceError::Binding(format!(
                    "QASM result {name:?} needs a graph binding, fixed outcome, or explicit discard"
                )))
        })
        .collect()
}

fn validate_expr(expr: &Expr, names: &[String]) -> Result<(), FeedbackInferenceError> {
    match expr {
        Expr::Var(name) if !names.contains(name) => Err(FeedbackInferenceError::Binding(format!(
            "unknown graph record {name:?}"
        ))),
        Expr::Var(_) => Ok(()),
        Expr::Not(inner) => validate_expr(inner, names),
        Expr::Binary(_, left, right) => {
            validate_expr(left, names)?;
            validate_expr(right, names)
        }
    }
}

fn assignment(names: &[String], mask: usize) -> BranchAssignment {
    names
        .iter()
        .enumerate()
        .map(|(bit, name)| (name.clone(), mask & (1 << bit) != 0))
        .collect()
}

fn compare_branches(
    verifier: &LogicalVerifier,
    names: &[String],
    expected: &[Vec<FScalar>],
    preparation: Option<&QuizxGraph>,
    options: &FeedbackOptions,
) -> Result<Vec<Vec<Pauli>>, FeedbackInferenceError> {
    let mut frames = Vec::with_capacity(expected.len());
    let mut nonzero = false;
    for (mask, expected) in expected.iter().enumerate() {
        let assignment = assignment(names, mask);
        let (branch, diagrams) = verifier.branch_family(&assignment, options.max_classes)?;
        let rejected = options
            .discard_if
            .as_ref()
            .is_some_and(|condition| eval_expr(condition, &assignment));
        if rejected != (branch.status() == BranchStatus::Rejected) {
            return Err(FeedbackInferenceError::RejectionMismatch { assignment });
        }
        let mut frame = vec![Pauli::I; verifier.boundaries().outputs().len()];
        if rejected {
            frames.push(frame);
            continue;
        }
        let expected_nonzero = expected.iter().any(|value| !value.is_zero());
        let mut maps = Vec::new();
        for diagram in diagrams {
            let prepared = match preparation {
                Some(preparation) => {
                    let mut prepared = preparation.clone();
                    prepared.plug(&diagram);
                    prepared
                }
                None => diagram,
            };
            let actual = contract(prepared, options.max_tensor_qubits)?;
            if actual.iter().all(Zero::is_zero) {
                continue;
            }
            if !expected_nonzero {
                return Err(FeedbackInferenceError::SupportMismatch { assignment });
            }
            maps.push(actual);
        }
        if expected_nonzero == maps.is_empty() {
            return Err(FeedbackInferenceError::SupportMismatch { assignment });
        }
        if !maps.is_empty() {
            let actual = channel(&maps)?;
            frame = output_pauli(&actual, expected, frame.len())?.ok_or_else(|| {
                FeedbackInferenceError::NoOutputPauli {
                    assignment: assignment.clone(),
                }
            })?;
            nonzero = true;
        }
        frames.push(frame);
    }
    if !nonzero {
        return Err(FeedbackInferenceError::EmptyDomain);
    }
    Ok(frames)
}

pub(super) fn checked_count(
    bits: usize,
    limit: usize,
    phase: &'static str,
) -> Result<usize, FeedbackInferenceError> {
    let count = u32::try_from(bits)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits));
    let observed = count.unwrap_or(usize::MAX);
    if count.is_none() || observed > limit {
        Err(FeedbackInferenceError::Limit {
            phase,
            observed,
            limit,
        })
    } else {
        Ok(observed)
    }
}

fn contract(mut graph: QuizxGraph, limit: usize) -> Result<Vec<FScalar>, FeedbackInferenceError> {
    let boundary = graph.inputs().len() + graph.outputs().len();
    if boundary > limit || boundary >= usize::BITS as usize {
        return Err(FeedbackInferenceError::Limit {
            phase: "tensor qubits",
            observed: boundary,
            limit,
        });
    }
    quizx::simplify::full_simp(&mut graph);
    if graph.scalar().is_zero() {
        return Ok(vec![FScalar::zero(); 1 << boundary]);
    }
    graph.x_to_z();
    // Match QuiZX's contraction order and bound the actual intermediate axes,
    // not just the small external tensor. No alternative contraction engine.
    let order = graph
        .inputs()
        .iter()
        .copied()
        .chain(
            graph
                .vertices()
                .filter(|&vertex| graph.vertex_type(vertex) != VType::B),
        )
        .chain(graph.outputs().iter().copied())
        .collect::<Vec<_>>();
    let mut seen = BTreeMap::new();
    let mut live = BTreeSet::new();
    for vertex in order.into_iter().rev() {
        let width = live.len() + 1;
        if width > limit {
            return Err(FeedbackInferenceError::Limit {
                phase: "intermediate tensor qubits",
                observed: width,
                limit,
            });
        }
        live.insert(vertex);
        let mut degree = 0;
        for neighbor in graph.neighbors(vertex) {
            if let Some(previous) = seen.get_mut(&neighbor) {
                degree += 1;
                *previous += 1;
                if graph.vertex_type(neighbor) != VType::B && *previous == graph.degree(neighbor) {
                    live.remove(&neighbor);
                }
            }
        }
        if graph.vertex_type(vertex) != VType::B && degree == graph.degree(vertex) {
            live.remove(&vertex);
        }
        seen.insert(vertex, degree);
    }
    let values = graph.to_tensorf().iter().copied().collect::<Vec<_>>();
    if values.iter().any(|value| {
        let value = value.complex_value();
        !value.re.is_finite() || !value.im.is_finite()
    }) {
        return Err(FeedbackInferenceError::NonFiniteTensor);
    }
    Ok(values)
}

fn channel(maps: &[Vec<FScalar>]) -> Result<Vec<FScalar>, FeedbackInferenceError> {
    let width = maps[0].len();
    let mut choi = vec![FScalar::zero(); width * width];
    // Every feasible anonymous coordinate class has the same fiber size.
    // Its common multiplicity cancels in this projective named-branch check.
    for map in maps {
        for (i, a) in map.iter().enumerate().filter(|(_, value)| !value.is_zero()) {
            for (j, b) in map.iter().enumerate().filter(|(_, value)| !value.is_zero()) {
                let product = *a * b.conj();
                if exact_scalar(product)? != exact_product(*a, b.conj())? {
                    return Err(FeedbackInferenceError::ArithmeticLimit);
                }
                let entry = &mut choi[i * width + j];
                let sum = *entry + product;
                if exact_scalar(sum)?
                    != exact_terms(scalar_terms(*entry).chain(scalar_terms(product)))?
                {
                    return Err(FeedbackInferenceError::ArithmeticLimit);
                }
                *entry = sum;
            }
        }
    }
    Ok(choi)
}

/// Enumerate only X masks. Relative amplitude signs form a GF(2) solve for Z.
fn output_pauli(
    actual: &[FScalar],
    expected: &[FScalar],
    outputs: usize,
) -> Result<Option<Vec<Pauli>>, FeedbackInferenceError> {
    if actual.len() != expected.len() {
        return Ok(None);
    }
    let Some(pivot) = expected.iter().position(|value| !value.is_zero()) else {
        return Ok(None);
    };
    let dimension = 1usize << outputs;
    let half = actual.len().ilog2() as usize / 2;
    let feature = |index: usize| ((index >> half) ^ index) & (dimension - 1);
    for x in 0..dimension {
        let permutation = (x << half) | x;
        let reference = actual[pivot ^ permutation];
        if reference.is_zero() {
            continue;
        }
        let mut signs = vec![None; dimension];
        let mut valid = true;
        for (index, target) in expected.iter().enumerate() {
            let value = actual[index ^ permutation];
            if value.is_zero() || target.is_zero() {
                if value.is_zero() != target.is_zero() {
                    valid = false;
                    break;
                }
                continue;
            }
            let lhs = exact_product(*target, reference)?;
            let rhs = exact_product(value, expected[pivot])?;
            let flip = if lhs == rhs {
                false
            } else if lhs.0 == rhs.0
                && lhs
                    .1
                    .iter()
                    .zip(rhs.1)
                    .all(|(a, b)| a.checked_add(b) == Some(0))
            {
                true
            } else {
                valid = false;
                break;
            };
            let delta = feature(index) ^ feature(pivot);
            if signs[delta]
                .replace(flip)
                .is_some_and(|previous| previous != flip)
            {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }
        let equations = signs
            .into_iter()
            .enumerate()
            .filter_map(|(delta, sign)| sign.map(|sign| (delta, sign)))
            .collect::<Vec<_>>();
        let mut rhs = CoeffVec::zeros(equations.len());
        let mut columns = vec![CoeffVec::zeros(equations.len()); outputs];
        for (row, &(delta, flip)) in equations.iter().enumerate() {
            rhs.set_bit(row, flip);
            for (output, column) in columns.iter_mut().enumerate() {
                column.set_bit(row, delta & (1 << (outputs - 1 - output)) != 0);
            }
        }
        if let Some(z) = solve_coeff_combination(&columns, &rhs) {
            return Ok(Some(
                (0..outputs)
                    .map(
                        |output| match (x & (1 << (outputs - 1 - output)) != 0, z.bit(output)) {
                            (false, false) => Pauli::I,
                            (true, false) => Pauli::X,
                            (false, true) => Pauli::Z,
                            (true, true) => Pauli::Y,
                        },
                    )
                    .collect(),
            ));
        }
    }
    Ok(None)
}

fn scalar_terms(value: FScalar) -> impl Iterator<Item = (usize, i128, i32)> {
    value
        .exact_dyadic_form()
        .into_iter()
        .enumerate()
        .filter(|(_, (coefficient, _))| *coefficient != 0)
        .map(|(index, (coefficient, power))| (index, i128::from(coefficient), i32::from(power)))
}

fn exact_scalar(value: FScalar) -> Result<(i32, [i128; 4]), FeedbackInferenceError> {
    let complex = value.complex_value();
    if !complex.re.is_finite() || !complex.im.is_finite() {
        return Err(FeedbackInferenceError::ArithmeticLimit);
    }
    exact_terms(scalar_terms(value))
}

fn exact_product(
    left: FScalar,
    right: FScalar,
) -> Result<(i32, [i128; 4]), FeedbackInferenceError> {
    exact_scalar(left)?;
    exact_scalar(right)?;
    exact_terms(scalar_terms(left).flat_map(|(i, a, power_a)| {
        scalar_terms(right).map(move |(j, b, power_b)| {
            // Each source mantissa fits in 53 bits, so this product fits i128.
            (
                (i + j) % 4,
                a * b * if i + j >= 4 { -1 } else { 1 },
                power_a + power_b,
            )
        })
    }))
}

/// Exact arithmetic on the returned f64 dyadic coefficients, with a separate
/// exponent. Reject wide coefficients instead of underflowing cross-products.
/// This does not retroactively certify QuiZX's preceding simplification.
fn exact_terms(
    terms: impl Iterator<Item = (usize, i128, i32)>,
) -> Result<(i32, [i128; 4]), FeedbackInferenceError> {
    let terms = terms.collect::<Vec<_>>();
    let Some(mut power) = terms.iter().map(|&(_, _, power)| power).min() else {
        return Ok((0, [0; 4]));
    };
    let mut coefficients = [0_i128; 4];
    for (index, coefficient, exponent) in terms {
        let shift =
            u32::try_from(exponent - power).map_err(|_| FeedbackInferenceError::ArithmeticLimit)?;
        let factor = 1_i128
            .checked_shl(shift)
            .filter(|factor| *factor > 0)
            .ok_or(FeedbackInferenceError::ArithmeticLimit)?;
        coefficients[index] = coefficient
            .checked_mul(factor)
            .and_then(|value| coefficients[index].checked_add(value))
            .ok_or(FeedbackInferenceError::ArithmeticLimit)?;
    }
    let Some(shift) = coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .map(|coefficient| coefficient.trailing_zeros())
        .min()
    else {
        return Ok((0, [0; 4]));
    };
    power += shift as i32;
    for coefficient in &mut coefficients {
        *coefficient >>= shift;
    }
    Ok((power, coefficients))
}

fn frame_actions(names: &[String], outputs: &[IVec3], frames: &[Vec<Pauli>]) -> Vec<Action> {
    let mut actions = Vec::new();
    for (output, &target) in outputs.iter().enumerate() {
        for (axis, pauli) in [(Pauli::X, PauliBasis::X), (Pauli::Z, PauliBasis::Z)] {
            let mut coefficients = frames
                .iter()
                .map(|frame| frame[output] & axis)
                .collect::<Vec<_>>();
            for bit in 0..names.len() {
                for mask in 0..coefficients.len() {
                    if mask & (1 << bit) != 0 {
                        coefficients[mask] ^= coefficients[mask ^ (1 << bit)];
                    }
                }
            }
            let constant = coefficients[0];
            let terms = coefficients
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, bit)| **bit)
                .map(|(mask, _)| {
                    join(
                        crate::BinaryOp::And,
                        names
                            .iter()
                            .enumerate()
                            .filter(|(bit, _)| mask & (1 << bit) != 0)
                            .map(|(_, name)| Expr::Var(name.clone()))
                            .collect(),
                    )
                    .expect("nonconstant monomial has variables")
                })
                .collect();
            let expression = join(crate::BinaryOp::Xor, terms);
            let condition = match (constant, expression) {
                (false, None) => continue,
                (true, None) => None,
                (false, Some(expr)) => Some(expr),
                (true, Some(expr)) => Some(Expr::Not(Box::new(expr))),
            };
            actions.push(Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli,
                    target,
                    direction: None,
                }],
                condition,
            });
        }
    }
    actions
}

fn join(op: crate::BinaryOp, mut expressions: Vec<Expr>) -> Option<Expr> {
    // Balanced trees avoid a stack proportional to the truth-table length.
    while expressions.len() > 1 {
        let mut iter = expressions.into_iter();
        expressions = std::iter::from_fn(|| {
            let left = iter.next()?;
            Some(match iter.next() {
                Some(right) => Expr::Binary(op, Box::new(left), Box::new(right)),
                None => left,
            })
        })
        .collect();
    }
    expressions.pop()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::circuit::Circuit;
    use quizx::fscalar::One;

    #[test]
    fn dyadic_scaling_never_turns_distinct_channels_into_a_match() {
        let zero = FScalar::zero();
        for exponent in [-600, 0, 600] {
            let epsilon = FScalar::dyadic(exponent, [1, 0, 0, 0]);
            let actual = [epsilon, zero, zero, epsilon * FScalar::real(3.0)];
            let different = [epsilon, zero, zero, epsilon * FScalar::real(2.0)];
            assert_eq!(output_pauli(&actual, &different, 1).unwrap(), None);
            assert_eq!(
                output_pauli(&actual, &actual, 1).unwrap(),
                Some(vec![Pauli::I])
            );
        }
        let tiny = FScalar::dyadic(-600, [1, 0, 0, 0]);
        assert!(matches!(
            channel(&[vec![tiny, tiny]]),
            Err(FeedbackInferenceError::ArithmeticLimit)
        ));
        let wide = FScalar::dyadic(500, [1, 0, 0, 0]) + FScalar::dyadic(-500, [0, 1, 0, 0]);
        assert!(matches!(
            exact_product(wide, wide),
            Err(FeedbackInferenceError::ArithmeticLimit)
        ));
    }

    #[test]
    fn output_matching_checks_inputs_and_solves_entangled_paulis() {
        for outputs in 1..=3 {
            let mut circuit = Circuit::new(outputs);
            circuit.add_gate("h", vec![0]);
            if outputs > 1 {
                circuit.add_gate("cx", vec![0, 1]);
            }
            circuit.add_gate("t", vec![outputs - 1]);
            let actual = circuit.to_tensorf().iter().copied().collect::<Vec<_>>();
            for mask in 0..1usize << (2 * outputs) {
                let mut corrected = circuit.clone();
                let mut expected_paulis = Vec::new();
                for output in 0..outputs {
                    let x = mask & (1 << (2 * output)) != 0;
                    let z = mask & (1 << (2 * output + 1)) != 0;
                    if x {
                        corrected.add_gate("x", vec![output]);
                    }
                    if z {
                        corrected.add_gate("z", vec![output]);
                    }
                    expected_paulis.push(match (x, z) {
                        (false, false) => Pauli::I,
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                    });
                }
                let expected = corrected.to_tensorf().iter().copied().collect::<Vec<_>>();
                assert_eq!(
                    output_pauli(
                        &channel(std::slice::from_ref(&actual)).unwrap(),
                        &channel(&[expected]).unwrap(),
                        outputs,
                    )
                    .unwrap(),
                    Some(expected_paulis)
                );
            }
        }
        let one = FScalar::one();
        let zero = FScalar::zero();
        for (actual, expected, outputs, paulis) in [
            (
                vec![one, zero, zero, zero],
                vec![zero, zero, zero, one],
                1,
                None,
            ),
            (vec![one, zero], vec![zero, one], 1, Some(vec![Pauli::X])),
            (vec![one, one], vec![one, zero - one], 0, None),
        ] {
            assert_eq!(
                output_pauli(
                    &channel(&[actual]).unwrap(),
                    &channel(&[expected]).unwrap(),
                    outputs,
                )
                .unwrap(),
                paulis
            );
        }
    }
}
