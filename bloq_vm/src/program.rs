//! Execute a whole compiled [`Bloq`] program without adding noise on the shared
//! [`Simulator`] engine. This is the *one* walk that owns SEM-ORD ordering,
//! region control flow, and measurement-record binding; [`run_bloq`],
//! [`run_bloq_with_hook`], and [`run_bloq_with_io`] are its entry points. (The
//! single-template runner lives with the verifier in [`crate::verify`].)
//!
//! # Whole-program model (`run_bloq`)
//!
//! Measurement ids are allocated contiguously across every instance measurement
//! at every nesting level (`InstanceMeasurement{instance, measurement}` → global
//! id). Nodes are executed in `deterministic_emit_order` onto ONE shared
//! [`Simulator`] over the program's global coordinate layout; a quantum
//! node's circuit comes from `emission_plan` (instances offset-translated), with
//! its local measurement ids remapped to the global ids.
//!
//! ## Region control flow (dynamic, per-shot — the point of a procedural engine)
//!
//! Region bodies are nested [`SubGraph`](bloq_ir::SubGraph)s over the same
//! template pool; the walk recurses into them on the same shared simulator:
//! * `RepeatUntilSuccess` — execute sandboxed body attempts until both the
//!   post-selection restart parities and the IR restart predicate accept, up to
//!   [`DEFAULT_RUS_ATTEMPT_CAP`] attempts. Rejected attempts restore the
//!   pre-body simulator and records (SEM-ATTEMPT); the accepted attempt alone
//!   composes upward.
//!
//! Quantum membership guards select active instances and their record aliases;
//! activation slots suppress inactive classical recipes and region bodies.
//! Classical nodes fold as in a flat program (observable recipes/linear `Compute`;
//! `Discard` rejects the shot and stops its execution). An `Observable` is a
//! complete corrected-parity producer, folding local records and bit-carrying
//! inputs. This no-op executor uses a zero decoder flip; composition
//! therefore has the same bit here. Boundaries contribute only to reports.
//! The per-shot `ObservableReport` is measured on the final state
//! ([`Program::observable_value`]) — the record/feedback fold plus any
//! composed boundary-Pauli face. Detectors (template-local + node
//! instance-space, at every level) are verified per shot with the usual
//! `LoopState`/`RepeatBody` policy.
//!
//! # Observable contract
//!
//! An `Observable` is its raw record parity plus live `Output`-face boundary
//! Paulis. Feedforward byproducts therefore remain visible; callers combine the
//! raw value with the terminal frame sign when they need a deterministic logical
//! value. `Input` faces are source endpoints already closed by composition and
//! are not folded. Because output faces are read from the terminal simulator
//! state, preparation rejects a later quantum node that consumes one.

use std::cell::RefCell;

use bloq_ir::circuit::{DetectorParity, DetectorTerm, PauliMap};
#[cfg(test)]
use bloq_ir::lowering::TemplateDetectorScope;
use bloq_ir::lowering::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstanceId};
use bloq_ir::{
    Bloq, BloqNodeId, BloqNodeKind, BodySelector, ClassicalNode, LevelPath, NodeDetectorAddress,
    ObservableOutput, RegionNode, SubGraph,
};
use glam::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::circuit::ShotRecord;
use super::prepared::PreparedProgramCore;
use super::{ExecError, MAX_PHYSICAL_BOUNDARY_BINDINGS};
use crate::backend::{Pauli, PauliString, Simulator};
use crate::verify::{FramePairReport, ObservableReport, VerifyReport, detector_report};

// ============================================================
// Whole program
// ============================================================

/// Retry bound for a `RepeatUntilSuccess` region: a shot still restarting after
/// this many attempts is discarded rather than looping forever.
pub const DEFAULT_RUS_ATTEMPT_CAP: u32 = 100;

/// How far `|⟨P⟩|` may sit below 1 and still count as a definite face value.
/// Stabilizer expectations land exactly on ±1; the slack is for the amplitude
/// map a `T` introduces.
const DETERMINISTIC_EXPECTATION_TOLERANCE: f64 = 1e-9;

/// Immutable, shot-independent program data.
struct Program {
    core: PreparedProgramCore,
    inputs: Vec<InputPatch>,
    /// Resolved restart parities keyed by enclosing RUS body scope; each entry
    /// retains its nested owner scope for executed-path applicability.
    rus_restarts: FxHashMap<LevelPath, Vec<(LevelPath, ResolvedDetector)>>,
    /// `Output` faces a later node consumes, read at their cut instead of on the
    /// terminal state. Empty for every program whose outputs stay live to the
    /// end, which is most of them.
    early_reads: Vec<super::closure::EarlyOutputRead>,
    /// Possible first consumers, aligned with logical-output metadata.
    output_cuts: Vec<Vec<(LevelPath, BloqNodeId)>>,
    /// Consumed logical outputs to swap onto private ancillas at their cuts.
    /// Empty unless the caller requested captured outputs.
    early_output_captures: Vec<EarlyOutputCapture>,
    /// Reused dense workspace for observables that fold multiple dynamic
    /// boundary bindings. Whole-program shots execute sequentially.
    boundary_scratch: RefCell<PauliString>,
}

struct EarlyOutputCapture {
    before: Vec<(LevelPath, BloqNodeId)>,
    instance: TemplateInstanceId,
    port: IVec3,
    operators: Vec<EarlyOutputPaulis>,
}

struct EarlyOutputPaulis {
    logical_x: PauliString,
    logical_z: PauliString,
    ancilla_x: PauliString,
    ancilla_z: PauliString,
}

#[derive(Clone, Copy)]
enum RunMode {
    Plain,
    PrepareInputs,
    CaptureOutputs,
}

/// Shot-independent data needed to validate terminal folds and build reports.
struct ProgramReportLayout {
    detectors: Vec<(LevelPath, ResolvedDetector)>,
    observable_nodes: Vec<BloqNodeId>,
    outputs: Vec<OutputLogical>,
    frames: Vec<bloq_ir::FramePair>,
}

type StagedInputSeeds = FxHashMap<TemplateInstanceId, Vec<(usize, usize)>>;

#[derive(Clone)]
struct DeferredCcz {
    seeds: [usize; 3],
    prepare_before: [Vec<TemplateInstanceId>; 3],
    prepared: bool,
}

#[derive(Debug, Clone)]
struct InputPatch {
    input_instance: TemplateInstanceId,
    prepare_before: Vec<TemplateInstanceId>,
    port: IVec3,
    data_qubits: Vec<usize>,
    stabilizers: Vec<PauliString>,
    logical_x: PauliString,
    logical_z: PauliString,
}

impl InputPatch {
    fn prepare_injection_scaffold(&self, sim: &mut Simulator) -> Result<usize, ExecError> {
        let mut intersection = None;
        for &qubit in &self.data_qubits {
            let on_x = self.logical_x.xbit(qubit);
            if on_x {
                sim.reset_x(qubit)?;
            } else {
                sim.reset(qubit)?;
            }
            if on_x && self.logical_z.zbit(qubit) && intersection.replace(qubit).is_some() {
                return Err(ExecError::InvalidInjectionPatch { port: self.port });
            }
        }
        intersection.ok_or(ExecError::InvalidInjectionPatch { port: self.port })
    }

    fn postselect_injection_patch(&self, sim: &mut Simulator) -> Result<(), ExecError> {
        for stabilizer in &self.stabilizers {
            sim.postselect_observable(stabilizer, false)?;
        }
        Ok(())
    }
}

/// One private seed qubit prepared before execution and injected at its input
/// Port's safe schedule cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSeed {
    /// The source input port.
    pub port: IVec3,
    /// Private engine qubit initialized to `|+>`.
    pub qubit: usize,
}

/// Per-shot private-seed context for [`run_bloq_with_io`].
#[derive(Debug)]
pub struct PreparationContext<'a> {
    /// Zero-based shot number.
    pub shot: usize,
    /// One private seed per logical input, sorted by source port.
    pub inputs: &'a [InputSeed],
    deferred_ccz: &'a RefCell<Vec<[IVec3; 3]>>,
}

impl PreparationContext<'_> {
    /// Queue an ideal logical CCZ resource on three input seeds. The executor
    /// applies the gate immediately before the first quantum component that
    /// consumes any leg, so earlier block execution cannot overwrite it.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::InvalidCczPreparation`] when any two ports are equal.
    pub fn prepare_ccz(&self, ports: [IVec3; 3]) -> Result<(), ExecError> {
        if ports[0] == ports[1] || ports[0] == ports[2] || ports[1] == ports[2] {
            return Err(ExecError::InvalidCczPreparation);
        }
        self.deferred_ccz.borrow_mut().push(ports);
        Ok(())
    }
}

/// Per-shot mutable state.
struct Shot {
    sim: Simulator,
    staged_inputs: StagedInputSeeds,
    injected_inputs: Vec<bool>,
    deferred_ccz: Vec<DeferredCcz>,
    record: ShotRecord,
    discarded: bool,
    /// Peak stabilizer rank reached during the shot (a `T` grows it before a
    /// post-selection measurement can collapse it back).
    ///
    /// Sampled once per *instruction*, which is finer than the once-per-`Op`
    /// sampling this used before the batch rewiring: an `Op` carrying several
    /// `T` targets is one op but several rank-moving steps, and the old reading
    /// could only see where that op ended. The rank timeline itself is
    /// unchanged — Clifford instructions never touch the amplitude map — so
    /// this only ever reports a peak the shot genuinely passed through. It runs
    /// about 2x higher on the non-Clifford gallery fixtures as a result.
    peak_rank: usize,
    /// Instances excluded by membership guards or inactive regions this shot.
    /// Their measurement terms are dropped; a missing record for any other
    /// instance is a hard error.
    excluded_instances: rustc_hash::FxHashSet<TemplateInstanceId>,
    /// Region-body scopes entered on the terminal execution path. Detector
    /// ownership also gates detectors inside inactive regions.
    executed_scopes: rustc_hash::FxHashSet<LevelPath>,
    active_membership_inputs: rustc_hash::FxHashSet<(LevelPath, BloqNodeId, u32)>,
    /// Branch selector outcomes this shot, in evaluation order.
    branch_choices: Vec<bool>,
    /// Values of `Program::early_reads`, taken at their cuts, in that order.
    /// `None` where the consuming node has not run (or never runs, its arm not
    /// being taken) and the face is still live for the terminal read.
    early_faces: Vec<Option<bool>>,
    /// Evaluated values remain available to gates in ancestor scopes.
    classical_values: FxHashMap<(LevelPath, BloqNodeId), Option<bool>>,
    /// Engine indices holding logical outputs swapped away before reuse.
    saved_outputs: Vec<Option<usize>>,
    /// Seed used to construct this shot, for independent retry RNG streams.
    seed: u64,
}

struct CompletedShot<'a> {
    state: Shot,
    top_values: FxHashMap<u32, Option<bool>>,
    top_bindings: FxHashMap<u32, BoundOperators<'a>>,
}

enum BoundOperators<'a> {
    /// Physical reads retain Output faces; Input faces stay in the IR.
    Owned(Vec<&'a InstanceBoundaryOperator>),
    Recipe(Vec<&'a InstanceBoundaryOperator>, Box<[BloqNodeId]>),
}

impl BoundOperators<'_> {
    fn has_output(&self) -> bool {
        match self {
            Self::Owned(operators) => !operators.is_empty(),
            Self::Recipe(operators, children) => !operators.is_empty() || !children.is_empty(),
        }
    }
}

fn collect_bound_operators<'a>(
    bindings: &FxHashMap<u32, BoundOperators<'a>>,
    producer: BloqNodeId,
    out: &mut Vec<&'a InstanceBoundaryOperator>,
    expanded: &mut usize,
) -> Result<(), ExecError> {
    let mut stack = vec![(producer, false, false)];
    let mut active = FxHashSet::default();
    while let Some((producer, exit, nested_recipe)) = stack.pop() {
        if exit {
            active.remove(&producer);
            continue;
        }
        if nested_recipe {
            *expanded = expanded.saturating_add(1);
            if *expanded > MAX_PHYSICAL_BOUNDARY_BINDINGS {
                return Err(ExecError::BoundaryBindingExpansionLimit {
                    limit: MAX_PHYSICAL_BOUNDARY_BINDINGS,
                });
            }
        }
        if !active.insert(producer) {
            return Err(ExecError::MalformedGraph("cyclic binding recipe"));
        }
        match bindings.get(&producer.0) {
            Some(BoundOperators::Owned(operators)) => {
                if nested_recipe {
                    *expanded = expanded.saturating_add(operators.len());
                    if *expanded > MAX_PHYSICAL_BOUNDARY_BINDINGS {
                        return Err(ExecError::BoundaryBindingExpansionLimit {
                            limit: MAX_PHYSICAL_BOUNDARY_BINDINGS,
                        });
                    }
                }
                out.extend(operators);
                active.remove(&producer);
            }
            Some(BoundOperators::Recipe(operators, children)) => {
                *expanded = expanded.saturating_add(operators.len());
                if *expanded > MAX_PHYSICAL_BOUNDARY_BINDINGS {
                    return Err(ExecError::BoundaryBindingExpansionLimit {
                        limit: MAX_PHYSICAL_BOUNDARY_BINDINGS,
                    });
                }
                out.extend(operators);
                stack.push((producer, true, nested_recipe));
                stack.extend(children.iter().rev().map(|child| (*child, false, true)));
            }
            None => {
                active.remove(&producer);
            }
        }
    }
    Ok(())
}

/// Pre-body state reused by every retry. `clone_from` preserves the live
/// simulator/hash-map allocations instead of reallocating them each attempt.
struct ShotSnapshot {
    sim: Simulator,
    injected_inputs: Vec<bool>,
    deferred_ccz: Vec<DeferredCcz>,
    record: ShotRecord,
    discarded: bool,
    excluded_instances: rustc_hash::FxHashSet<TemplateInstanceId>,
    executed_scopes: rustc_hash::FxHashSet<LevelPath>,
    active_membership_inputs: rustc_hash::FxHashSet<(LevelPath, BloqNodeId, u32)>,
    branch_choices: Vec<bool>,
    early_faces: Vec<Option<bool>>,
    /// Evaluated values remain available to gates in ancestor scopes.
    classical_values: FxHashMap<(LevelPath, BloqNodeId), Option<bool>>,
    saved_outputs: Vec<Option<usize>>,
}

impl ShotSnapshot {
    fn capture(state: &Shot) -> Self {
        Self {
            sim: state.sim.clone(),
            injected_inputs: state.injected_inputs.clone(),
            deferred_ccz: state.deferred_ccz.clone(),
            record: state.record.clone(),
            discarded: state.discarded,
            excluded_instances: state.excluded_instances.clone(),
            executed_scopes: state.executed_scopes.clone(),
            active_membership_inputs: state.active_membership_inputs.clone(),
            branch_choices: state.branch_choices.clone(),
            early_faces: state.early_faces.clone(),
            classical_values: state.classical_values.clone(),
            saved_outputs: state.saved_outputs.clone(),
        }
    }

    fn restore(&self, state: &mut Shot) {
        state.sim.clone_from(&self.sim);
        state.injected_inputs.clone_from(&self.injected_inputs);
        state.deferred_ccz.clone_from(&self.deferred_ccz);
        state.record.clone_from(&self.record);
        state.discarded = self.discarded;
        state
            .excluded_instances
            .clone_from(&self.excluded_instances);
        state.executed_scopes.clone_from(&self.executed_scopes);
        state
            .active_membership_inputs
            .clone_from(&self.active_membership_inputs);
        state.branch_choices.clone_from(&self.branch_choices);
        state.early_faces.clone_from(&self.early_faces);
        state.classical_values.clone_from(&self.classical_values);
        state.saved_outputs.clone_from(&self.saved_outputs);
    }
}

/// Logical `X` and `Z` operators for one program output.
///
/// These engine-index [`PauliString`]s are ready to measure at the end of a
/// shot. [`run_bloq_with_hook`] builds them once and lends them to every shot's
/// hook via [`ShotContext`].
///
/// Both maps come directly from [`Bloq::logical_outputs`], so execution needs
/// no compiler geometry.
#[derive(Debug, Clone)]
pub struct OutputLogical {
    /// The source output port.
    pub port: IVec3,
    /// Logical `X` over engine indices.
    pub logical_x: PauliString,
    /// Logical `Z` over engine indices.
    pub logical_z: PauliString,
    /// Whether a later node can consume this output's patch. Its operators
    /// then refer to a later worldline on the terminal state; hooks must use
    /// [`ShotContext::saved_outputs`] to read outputs captured at an earlier cut.
    pub consumed: bool,
}

/// One output's terminal Pauli-frame value for the current shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShotFrame {
    /// The source output port this frame corrects.
    pub port: IVec3,
    /// Whether an `X` correction is required, or `None` if unevaluable.
    pub x: Option<bool>,
    /// Whether a `Z` correction is required, or `None` if unevaluable.
    pub z: Option<bool>,
}

/// A logical output swapped onto a private engine qubit before its physical
/// patch was reused later in the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedOutput {
    /// The source output port.
    pub port: IVec3,
    /// Engine index holding the logical state before terminal frame correction.
    pub qubit: usize,
}

/// Per-shot context handed to a [`run_bloq_with_hook`] hook, evaluated after the
/// program finishes but before the shot's simulator is dropped.
///
/// The hook receives the pristine post-program simulator (the output patches are
/// still live — an ideal noiseless boundary). A hook that mutates the simulator
/// (e.g. a destructive measurement) therefore invalidates *this shot's* observable and
/// frame reports, but never the record-based detectors.
#[derive(Debug)]
pub struct ShotContext<'a> {
    /// Zero-based shot number (discarded shots never reach this hook).
    pub shot: usize,
    /// This shot's measurement outcomes.
    pub record: &'a ShotRecord,
    /// This shot's stamped selector outcomes, in evaluation order — the
    /// feedforward action bits driving the terminal Pauli frame.
    pub branch_selectors: &'a [bool],
    /// Structural and selective choices, keyed by their source selector names.
    pub named_branch_selectors: &'a [(&'a str, bool)],
    /// Every program output's logical operators (shot-independent).
    pub outputs: &'a [OutputLogical],
    /// Evaluated terminal Pauli frames for this shot, keyed by output port.
    pub frames: &'a [ShotFrame],
    /// Outputs saved at an earlier cut because their physical patch was reused.
    pub saved_outputs: &'a [SavedOutput],
    /// Live engine width. A fresh ancilla lives at this index (the engine grows
    /// on demand for the gates/measurements a hook performs).
    pub qubit_count: usize,
}

/// Verify a whole [`Bloq`] program over a seeded shot batch.
///
/// The program is flattened, then `shots` shots run on a shared simulator with
/// full region control flow. Every detector must be constant across applicable,
/// non-discarded shots; each `Observable` node's per-shot value is reported.
///
/// # Errors
/// See [`ExecError`].
pub fn run_bloq(bloq: &Bloq, shots: usize, seed: u64) -> Result<VerifyReport, ExecError> {
    run_bloq_internal(
        bloq,
        shots,
        seed,
        RunMode::Plain,
        &mut |_, _| Ok(()),
        &mut |_, _| Ok(()),
    )
}

/// Run [`run_bloq`] with a per-shot hook.
///
/// The hook receives the pristine post-program simulator after the program
/// finishes but before the shot's state is dropped or output boundaries are
/// measured. It is called only for non-discarded shots, with a [`ShotContext`]
/// carrying the shot's records, stamped selectors, and [`OutputLogical`]s.
///
/// This is the entry point for out-of-band per-shot analysis a raw record fold
/// cannot express — chiefly logical-state fidelity of a non-Clifford output
/// (a `|T⟩` factory). A hook that mutates the simulator invalidates that shot's
/// observable/frame reports (but not its detectors); the no-op hook of
/// [`run_bloq`] leaves the state untouched, so its full report stays valid.
///
/// # Errors
/// See [`ExecError`]; also propagates any error the `hook` returns.
pub fn run_bloq_with_hook<F>(
    bloq: &Bloq,
    shots: usize,
    seed: u64,
    mut hook: F,
) -> Result<VerifyReport, ExecError>
where
    F: FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
{
    run_bloq_internal(
        bloq,
        shots,
        seed,
        RunMode::Plain,
        &mut |_, _| Ok(()),
        &mut hook,
    )
}

/// [`run_bloq`] with private input seeds and a post-shot output hook.
///
/// Every seed starts in `|+>`. The preparation hook may entangle seeds. Each is
/// injected before its temporal input Port, or before the first wait/cube after
/// an ordinary spatial input's virtual Port. Time-separated inputs may reuse
/// one patch. Multiplex inputs inject before the virtual Port so its split sees them.
///
/// # Errors
/// See [`ExecError`]; also propagates errors returned by either hook.
pub fn run_bloq_with_io<P, F>(
    bloq: &Bloq,
    shots: usize,
    seed: u64,
    mut prepare: P,
    mut peek: F,
) -> Result<VerifyReport, ExecError>
where
    P: FnMut(&mut Simulator, &PreparationContext<'_>) -> Result<(), ExecError>,
    F: FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
{
    run_bloq_internal(
        bloq,
        shots,
        seed,
        RunMode::PrepareInputs,
        &mut prepare,
        &mut peek,
    )
}

/// Run [`run_bloq_with_hook`] while preserving reused output patches.
///
/// Each reused patch is swapped onto a private ancilla at its output cut. The
/// hook reads those ancillas through [`ShotContext::saved_outputs`], allowing a
/// joint-state oracle to include outputs no longer live at program termination.
///
/// As with any mutating hook, only the record-based detector portion of the
/// returned report remains meaningful.
///
/// # Errors
/// See [`ExecError`]; also propagates any error the `hook` returns.
pub fn run_bloq_with_captured_outputs<F>(
    bloq: &Bloq,
    shots: usize,
    seed: u64,
    mut hook: F,
) -> Result<VerifyReport, ExecError>
where
    F: FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
{
    run_bloq_internal(
        bloq,
        shots,
        seed,
        RunMode::CaptureOutputs,
        &mut |_, _| Ok(()),
        &mut hook,
    )
}

/// The shared whole-program walk behind [`run_bloq`] and [`run_bloq_with_hook`].
/// It calls one post-program `hook` per non-discarded shot (out-of-band
/// analysis); the plain [`run_bloq`] path passes a no-op.
fn run_bloq_internal<P, F>(
    bloq: &Bloq,
    shots: usize,
    seed: u64,
    mode: RunMode,
    prepare: &mut P,
    hook: &mut F,
) -> Result<VerifyReport, ExecError>
where
    P: FnMut(&mut Simulator, &PreparationContext<'_>) -> Result<(), ExecError>,
    F: FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
{
    let program = Program::prepare(bloq, mode)?;
    let layout = program.report_layout()?;

    let mut detector_shots: Vec<Vec<bool>> = vec![Vec::new(); layout.detectors.len()];
    let mut observable_shots: Vec<Vec<bool>> = vec![Vec::new(); layout.observable_nodes.len()];
    let mut frame_x: Vec<Vec<Option<bool>>> = vec![Vec::new(); layout.frames.len()];
    let mut frame_z: Vec<Vec<Option<bool>>> = vec![Vec::new(); layout.frames.len()];
    let mut discarded = 0usize;
    let mut max_rank = 0usize;
    let mut branch_selectors: Vec<bool> = Vec::new();
    for shot in 0..shots {
        let mut completed =
            program.execute_shot(shot, seed, matches!(mode, RunMode::PrepareInputs), prepare)?;
        max_rank = max_rank.max(completed.state.peak_rank);
        branch_selectors.extend(completed.state.branch_choices.iter().copied());
        if completed.state.discarded {
            discarded += 1;
            continue;
        }
        if program.core.bloq.logical_outputs().iter().any(|output| {
            completed
                .state
                .excluded_instances
                .contains(&output.instance)
        }) {
            return Err(ExecError::MalformedGraph(
                "logical output owner is not selected",
            ));
        }

        // Per-shot hook on the pristine post-program state (output patches still
        // live). Runs before the observable/frame folds below so a mutating hook
        // sees an undisturbed boundary; a mutating hook thus voids this shot's
        // observable/frame reports but not its record-based detectors.
        {
            let shot_frames = layout
                .frames
                .iter()
                .map(|frame| ShotFrame {
                    port: frame.port,
                    x: completed.top_values.get(&frame.x.0).copied().flatten(),
                    z: completed.top_values.get(&frame.z.0).copied().flatten(),
                })
                .collect::<Vec<_>>();
            let saved_outputs = program
                .early_output_captures
                .iter()
                .zip(&completed.state.saved_outputs)
                .filter_map(|(capture, &qubit)| {
                    qubit.map(|qubit| SavedOutput {
                        port: capture.port,
                        qubit,
                    })
                })
                .collect::<Vec<_>>();
            let named_branch_selectors = program
                .core
                .bloq
                .nodes()
                .filter_map(|(id, node)| {
                    let bloq_ir::NodeProvenance::BranchSelector { name } = &node.provenance else {
                        return None;
                    };
                    let value = completed
                        .state
                        .classical_values
                        .get(&(LevelPath::default(), id))
                        .copied()
                        .flatten()?;
                    Some((name.as_str(), value))
                })
                .collect::<Vec<_>>();
            let ctx = ShotContext {
                shot,
                record: &completed.state.record,
                branch_selectors: &completed.state.branch_choices,
                named_branch_selectors: &named_branch_selectors,
                outputs: &layout.outputs,
                frames: &shot_frames,
                saved_outputs: &saved_outputs,
                qubit_count: completed.state.sim.num_qubits(),
            };
            hook(&mut completed.state.sim, &ctx)?;
        }

        // Detectors owned by inactive memberships or regions are skipped.
        // Any other missing record is a hard error.
        for ((scope, parity), out) in layout.detectors.iter().zip(&mut detector_shots) {
            if !parity.applies(scope, &completed.state) {
                continue;
            }
            let value = xor_records(parity, scope, &completed.state)?;
            out.push(value);
        }
        for (node_id, out) in layout.observable_nodes.iter().zip(&mut observable_shots) {
            if let Some(value) = program.observable_value(
                *node_id,
                &completed.top_values,
                &completed.top_bindings,
                &mut completed.state,
            )? {
                out.push(value);
            }
        }
        for (i, frame) in layout.frames.iter().enumerate() {
            frame_x[i].push(completed.top_values.get(&frame.x.0).copied().flatten());
            frame_z[i].push(completed.top_values.get(&frame.z.0).copied().flatten());
        }
    }

    Ok(VerifyReport {
        shots,
        discarded,
        detectors: detector_shots.into_iter().map(detector_report).collect(),
        observables: observable_shots
            .into_iter()
            .map(|per_shot| ObservableReport { per_shot })
            .collect(),
        frame_pairs: layout
            .frames
            .iter()
            .zip(frame_x)
            .zip(frame_z)
            .map(|((frame, x_bits), z_bits)| FramePairReport {
                port: frame.port,
                x_bits,
                z_bits,
            })
            .collect(),
        max_rank,
        branch_selectors,
    })
}

impl Program {
    fn prepare(bloq: &Bloq, mode: RunMode) -> Result<Self, ExecError> {
        let core = PreparedProgramCore::build(bloq)?;
        let rus_restarts = collect_rus_restarts(&core.bloq, &core.global_meas)?;
        let early_reads = super::closure::plan_early_output_reads(&core.bloq)?;
        let output_cuts = super::closure::logical_output_cuts(&core.bloq)?;
        let mut program = Self {
            boundary_scratch: RefCell::new(PauliString::new(core.coord_index.len())),
            core,
            inputs: Vec::new(),
            rus_restarts,
            early_reads,
            output_cuts,
            early_output_captures: Vec::new(),
        };
        if matches!(mode, RunMode::PrepareInputs) {
            program.inputs = program.input_patches()?;
        }
        if matches!(mode, RunMode::CaptureOutputs) {
            program.early_output_captures = program.plan_early_output_captures()?;
        }
        if program.early_output_captures.is_empty() {
            program.validate_early_read_compatibility()?;
        }
        Ok(program)
    }

    fn validate_early_read_compatibility(&self) -> Result<(), ExecError> {
        let operators = self
            .early_reads
            .iter()
            .map(|read| {
                let mut product = PauliString::new(self.core.coord_index.len());
                for binding in &read.bindings {
                    product = &product
                        * self
                            .core
                            .boundary_operator(&binding.operator(&self.core.bloq).operator)?;
                }
                Ok(product)
            })
            .collect::<Result<Vec<_>, ExecError>>()?;
        for (index, first) in self.early_reads.iter().enumerate() {
            for (other, second) in self.early_reads[..index].iter().enumerate() {
                if first.before == second.before
                    && (first.incomplete || second.incomplete)
                    && !operators[index].commutes_with(&operators[other])
                {
                    return Err(ExecError::IncompatibleEarlyOutputReads {
                        first: first.observable.0,
                        second: second.observable.0,
                        before: first.before.1.0,
                    });
                }
            }
        }
        Ok(())
    }

    fn report_layout(&self) -> Result<ProgramReportLayout, ExecError> {
        let mut detectors = collect_detectors(
            &self.core.bloq,
            self.core.bloq.top(),
            &self.core.global_meas,
        )?;
        let mut owners = FxHashMap::default();
        for (_, level) in self.core.bloq.levels() {
            for (_, node) in level.nodes() {
                if let Some(quantum) = node.try_quantum() {
                    owners.extend(
                        quantum
                            .instances
                            .iter()
                            .map(|instance| (instance.id, quantum)),
                    );
                }
            }
        }
        // Injection precedes the whole owning component, including its walls.
        let input_cuts = self
            .inputs
            .iter()
            .filter(|input| !input.prepare_before.contains(&input.input_instance))
            .map(|input| {
                let consumers = input
                    .prepare_before
                    .iter()
                    .filter_map(|instance| owners.get(instance))
                    .flat_map(|quantum| quantum.instances.iter().map(|instance| instance.id))
                    .collect::<FxHashSet<_>>();
                (input.input_instance, consumers)
            })
            .collect::<Vec<_>>();
        detectors.retain(|(_, detector)| {
            !input_cuts.iter().any(|(input, consumers)| {
                detector.terms.iter().any(|(instance, _)| instance == input)
                    && detector
                        .terms
                        .iter()
                        .any(|(instance, _)| consumers.contains(instance))
            })
        });
        let observable_nodes = collect_observables(&self.core.bloq);
        super::closure::guard_decode_observable_closure(&self.core.bloq)?;
        let outputs = self.output_logicals()?;
        let frames = self.core.bloq.output_frames();
        for output in self.core.bloq.logical_outputs() {
            if !frames.iter().any(|frame| frame.port == output.port) {
                return Err(ExecError::MalformedGraph(
                    "logical output has no complete frame pair",
                ));
            }
        }
        Ok(ProgramReportLayout {
            detectors,
            observable_nodes,
            outputs,
            frames,
        })
    }

    fn execute_shot<P>(
        &self,
        shot: usize,
        seed: u64,
        stage_inputs: bool,
        prepare: &mut P,
    ) -> Result<CompletedShot<'_>, ExecError>
    where
        P: FnMut(&mut Simulator, &PreparationContext<'_>) -> Result<(), ExecError>,
    {
        let shot_seed = seed ^ shot as u64;
        let base_qubits = self.core.coord_index.len();
        let mut state = Shot {
            sim: Simulator::with_seed(base_qubits, shot_seed),
            staged_inputs: StagedInputSeeds::default(),
            injected_inputs: vec![false; self.inputs.len()],
            deferred_ccz: Vec::new(),
            record: ShotRecord::with_len(self.core.record_count),
            discarded: false,
            peak_rank: 1,
            excluded_instances: rustc_hash::FxHashSet::default(),
            executed_scopes: rustc_hash::FxHashSet::default(),
            active_membership_inputs: rustc_hash::FxHashSet::default(),
            branch_choices: Vec::new(),
            early_faces: vec![None; self.early_reads.len()],
            classical_values: FxHashMap::default(),
            saved_outputs: vec![None; self.early_output_captures.len()],
            seed: shot_seed,
        };
        if stage_inputs {
            let mut seeds = Vec::with_capacity(self.inputs.len());
            for input in &self.inputs {
                let qubit = state.sim.num_qubits();
                state.sim.reset_x(qubit)?;
                seeds.push(InputSeed {
                    port: input.port,
                    qubit,
                });
            }
            let deferred_ccz = RefCell::new(Vec::new());
            prepare(
                &mut state.sim,
                &PreparationContext {
                    shot,
                    inputs: &seeds,
                    deferred_ccz: &deferred_ccz,
                },
            )?;
            for ports in deferred_ccz.into_inner() {
                let mut prepared = DeferredCcz {
                    seeds: [0; 3],
                    prepare_before: std::array::from_fn(|_| Vec::new()),
                    prepared: false,
                };
                for (index, port) in ports.into_iter().enumerate() {
                    let input_index = self
                        .inputs
                        .iter()
                        .position(|input| input.port == port)
                        .ok_or(ExecError::UnknownPreparationInput { port })?;
                    prepared.seeds[index] = seeds[input_index].qubit;
                    prepared.prepare_before[index] =
                        self.inputs[input_index].prepare_before.clone();
                }
                state.deferred_ccz.push(prepared);
            }
            for (index, (input, seed)) in self.inputs.iter().zip(seeds).enumerate() {
                for &target in &input.prepare_before {
                    state
                        .staged_inputs
                        .entry(target)
                        .or_default()
                        .push((index, seed.qubit));
                }
            }
        }
        state.peak_rank = state.peak_rank.max(state.sim.rank());
        let mut top_values = FxHashMap::default();
        let mut top_bindings = FxHashMap::default();
        self.execute_level(
            self.core.bloq.top(),
            &LevelPath::default(),
            &mut state,
            &mut top_values,
            &mut top_bindings,
        )?;
        Ok(CompletedShot {
            state,
            top_values,
            top_bindings,
        })
    }

    fn restart_signaled(&self, scope: &LevelPath, state: &Shot) -> Result<bool, ExecError> {
        for (owner, parity) in self.rus_restarts.get(scope).into_iter().flatten() {
            if parity.applies(owner, state) && xor_records(parity, owner, state)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Execute one [`SubGraph`] level in emit order on the shared simulator.
    /// `scope` is the enclosing region chain (empty at the top level); `values`
    /// holds this level's classical node values.
    fn execute_level<'a>(
        &'a self,
        level: &'a SubGraph,
        scope: &LevelPath,
        state: &mut Shot,
        values: &mut FxHashMap<u32, Option<bool>>,
        bindings: &mut FxHashMap<u32, BoundOperators<'a>>,
    ) -> Result<(), ExecError> {
        state.executed_scopes.insert(scope.clone());
        for &id in self.core.emit_order(scope)? {
            let node = level.node(id).expect("emit order yields real ids");
            if let Some(slot) = node.activation {
                let inputs = self.slot_inputs(level, id, values);
                let active = inputs.get(&slot).copied().ok_or(ExecError::MalformedGraph(
                    "classical activation input is unavailable",
                ))?;
                if !active {
                    if let Some(region) = node.try_region() {
                        for (_, body) in region.bodies() {
                            collect_instances(body, &mut state.excluded_instances);
                        }
                    } else if let Some(quantum) = node.try_quantum() {
                        state
                            .excluded_instances
                            .extend(quantum.instances.iter().map(|instance| instance.id));
                    }
                    values.insert(id.0, Some(false));
                    state
                        .classical_values
                        .insert((scope.clone(), id), Some(false));
                    continue;
                }
            }
            if !self.early_output_captures.is_empty() {
                self.capture_early_outputs(scope, id, state)?;
            } else if !self.early_reads.is_empty() {
                self.read_early_faces(scope, id, state)?;
            }
            match (node.try_classical(), &node.kind) {
                (_, BloqNodeKind::Quantum(quantum)) => {
                    if quantum.guards.is_empty() {
                        self.inject_input_seeds(quantum, state)?;
                        self.run_quantum_node(id, scope, state)?;
                    } else {
                        let inputs = self.slot_inputs(level, id, values);
                        let selected = self.core.selected_ops(scope, id, node, &inputs)?;
                        for guard in &quantum.guards {
                            if inputs[&guard.input] {
                                state.active_membership_inputs.insert((
                                    scope.clone(),
                                    id,
                                    guard.input,
                                ));
                            } else {
                                state.excluded_instances.extend(&guard.instances);
                            }
                        }
                        state.record.select_aliases(&selected.aliases);
                        self.inject_input_seeds(&selected.quantum, state)?;
                        let rank = selected.ops.run(&mut state.sim, &mut state.record)?;
                        state.peak_rank = state.peak_rank.max(rank);
                    }
                }
                (Some(ClassicalNode::Discard { condition }), _) => {
                    let inputs = self.slot_inputs(level, id, values);
                    state.discarded = condition
                        .eval(&mut |slot| inputs.get(&slot).copied())
                        .ok_or(ExecError::MalformedGraph(
                            "discard condition is unavailable",
                        ))?;
                }
                (_, BloqNodeKind::Classical(classical)) => {
                    let classical = classical.as_ref();
                    // Frame-content recipes fold over the shot record.
                    let value = self.classical_value(
                        level,
                        id,
                        classical,
                        values,
                        &state.record,
                        &state.excluded_instances,
                    )?;
                    values.insert(id.0, value);
                    if matches!(
                        node.provenance,
                        bloq_ir::NodeProvenance::BranchSelector { .. }
                    ) {
                        state
                            .branch_choices
                            .push(value.ok_or(ExecError::MalformedGraph(
                                "branch selector has no value",
                            ))?);
                        state.classical_values.insert((scope.clone(), id), value);
                    }
                    if let ClassicalNode::Observable { operators, .. } = classical {
                        let children = level
                            .data_inputs(id)
                            .filter(|input| input.output.is_none())
                            .map(|input| input.producer)
                            .filter(|child| {
                                bindings
                                    .get(&child.0)
                                    .is_some_and(BoundOperators::has_output)
                            })
                            .collect();
                        bindings.insert(
                            id.0,
                            BoundOperators::Recipe(
                                operators
                                    .iter()
                                    .filter(|operator| {
                                        operator.face == bloq_ir::BoundaryFace::Output
                                    })
                                    .collect(),
                                children,
                            ),
                        );
                    }
                }
                (_, BloqNodeKind::Region(region)) => {
                    self.execute_region(level, id, region, scope, state, values, bindings)?
                }
            }
            if !self.early_reads.is_empty()
                && let Some(&value) = values.get(&id.0)
            {
                state.classical_values.insert((scope.clone(), id), value);
            }
            if state.discarded {
                // SEM-DISCARD: no later scheduled node executes in this shot.
                break;
            }
        }
        Ok(())
    }

    /// Region control flow (see the [module docs](self)).
    // Threaded walk state (level/scope/state/values) alongside the node params —
    // a tree walk, not a public API surface.
    #[expect(
        clippy::too_many_arguments,
        reason = "recursive execution threads one region-walk state"
    )]
    fn execute_region<'a>(
        &'a self,
        level: &'a SubGraph,
        id: BloqNodeId,
        region: &'a RegionNode,
        scope: &LevelPath,
        state: &mut Shot,
        values: &mut FxHashMap<u32, Option<bool>>,
        bindings: &mut FxHashMap<u32, BoundOperators<'a>>,
    ) -> Result<(), ExecError> {
        let RegionNode::RepeatUntilSuccess {
            body,
            restart_condition,
            restart_source,
        } = region;
        let child = scope.child(id, BodySelector::Body);
        let snapshot = ShotSnapshot::capture(state);
        let mut attempt = 0;
        loop {
            if attempt != 0 {
                snapshot.restore(state);
            }
            state
                .sim
                .reseed_rng(rus_attempt_seed(state.seed, &child, attempt));

            let mut body_values = FxHashMap::default();
            let mut body_bindings = FxHashMap::default();
            self.execute_level(body, &child, state, &mut body_values, &mut body_bindings)?;
            // Both mandatory restart signals are evaluated unconditionally
            // (no short-circuit): an unevaluable restart condition is an
            // error even on an attempt a restart parity already rejected.
            let restart_detector = self.restart_signaled(&child, state)?;
            let open = restart_source.and_then(|source| {
                zero_decoder_output(
                    body_values.get(&source.node.0).copied().flatten(),
                    Some(source.output),
                )
            });
            let restart_predicate = restart_condition
                .eval(&mut |slot| {
                    let input = level.value_inputs(id).find(|input| input.slot == slot);
                    match input {
                        Some(input) => zero_decoder_output(
                            values.get(&input.producer.0).copied().flatten(),
                            input.output,
                        ),
                        None => open,
                    }
                })
                .ok_or(ExecError::MalformedGraph(
                    "RUS restart condition is not evaluable",
                ))?;

            if !(restart_detector || restart_predicate) {
                // Attempt-local draws must not perturb the parent RNG stream.
                state.sim.restore_rng_from(&snapshot.sim);
                values.insert(id.0, bound_body_value(body, &body_values));
                bindings.insert(
                    id.0,
                    BoundOperators::Owned(bound_body_bindings(body, &body_bindings)?),
                );
                break;
            }
            attempt += 1;
            if attempt >= DEFAULT_RUS_ATTEMPT_CAP {
                state.discarded = true;
                break;
            }
        }
        Ok(())
    }

    fn inject_input_seeds(
        &self,
        quantum: &bloq_ir::QuantumNode,
        state: &mut Shot,
    ) -> Result<(), ExecError> {
        for preparation in &mut state.deferred_ccz {
            if !preparation.prepared
                && preparation
                    .prepare_before
                    .iter()
                    .flatten()
                    .any(|cut| quantum.instances.iter().any(|instance| instance.id == *cut))
            {
                let [a, b, c] = preparation.seeds;
                state.sim.ccz(a, b, c)?;
                preparation.prepared = true;
            }
        }
        for instance in &quantum.instances {
            for &(input_index, seed) in state.staged_inputs.get(&instance.id).into_iter().flatten()
            {
                if state.injected_inputs[input_index] {
                    continue;
                }
                let input = &self.inputs[input_index];
                let corner = input.prepare_injection_scaffold(&mut state.sim)?;
                state.sim.swap(seed, corner);
                input.postselect_injection_patch(&mut state.sim)?;
                state.injected_inputs[input_index] = true;
            }
        }
        Ok(())
    }

    /// Replay a quantum node's prepared instruction stream.
    fn run_quantum_node(
        &self,
        id: BloqNodeId,
        scope: &LevelPath,
        state: &mut Shot,
    ) -> Result<(), ExecError> {
        let max_rank = self
            .core
            .prepared_ops(scope, id)?
            .run(&mut state.sim, &mut state.record)?;
        state.peak_rank = state.peak_rank.max(max_rank);
        Ok(())
    }

    /// The `Value` output of a record-based classical producer in one shot.
    fn classical_value(
        &self,
        level: &SubGraph,
        id: BloqNodeId,
        classical: &ClassicalNode,
        values: &FxHashMap<u32, Option<bool>>,
        record: &ShotRecord,
        excluded: &rustc_hash::FxHashSet<TemplateInstanceId>,
    ) -> Result<Option<bool>, ExecError> {
        let mut record_value = false;
        for im in classical.measurements() {
            if excluded.contains(&im.instance) {
                continue;
            }
            let gid = *self
                .core
                .global_meas
                .get(im)
                .ok_or(ExecError::MalformedGraph(
                    "observable names an unknown measurement",
                ))?;
            record_value ^= record.require(gid)?;
        }
        match classical {
            ClassicalNode::Compute { expr } => {
                let inputs = self.slot_inputs(level, id, values);
                Ok(expr.eval(&mut |slot| inputs.get(&slot).copied()))
            }
            // Recipes XOR records and data inputs. Output boundary operators
            // fold at report time ([`Self::observable_value`]); an unavailable
            // data input keeps the recipe unavailable.
            ClassicalNode::Observable { .. } => {
                let mut value = record_value;
                for input in level.data_inputs(id) {
                    match zero_decoder_output(
                        values.get(&input.producer.0).copied().flatten(),
                        input.output,
                    ) {
                        Some(bit) => value ^= bit,
                        None => return Ok(None),
                    }
                }
                Ok(Some(value))
            }
            ClassicalNode::Discard { .. } => Ok(None),
        }
    }

    /// This level's `Value` inputs of `id`, resolved to `slot → bit` from
    /// already-computed producer values.
    fn slot_inputs(
        &self,
        level: &SubGraph,
        id: BloqNodeId,
        values: &FxHashMap<u32, Option<bool>>,
    ) -> FxHashMap<u32, bool> {
        let mut inputs = FxHashMap::default();
        for input in level.value_inputs(id) {
            if let Some(bit) = zero_decoder_output(
                values.get(&input.producer.0).copied().flatten(),
                input.output,
            ) {
                inputs.insert(input.slot, bit);
            }
        }
        inputs
    }

    /// An `Observable` node's value: the XOR of its record producers plus any
    /// live `Output`-face boundary Pauli measured on the final state.
    ///
    /// `Input` faces contribute **nothing** to the physical fold. An
    /// input boundary operator is a *source endpoint* (e.g. a T block's
    /// magic-state anchor), closed internally by the cultivation seam when the
    /// templates compose. Its MPP-boundary
    /// realization is the *static* verifier's perfect-port stand-in; `run_bloq`
    /// executes the composed circuit, where that closure is already carried by the
    /// folded cultivation records, and the input face no longer
    /// exists as a live boundary. Measuring it on the terminal state would read
    /// the wrong operator at the wrong time (the patch has since been
    /// merged/consumed), so we skip it. Only `Output` faces are live terminal
    /// boundaries and are measured here. [`super::closure::plan_early_output_reads`]
    /// moves a consumed face's read ahead of the node that reuses its support.
    fn observable_value(
        &self,
        obs: BloqNodeId,
        top_values: &FxHashMap<u32, Option<bool>>,
        top_bindings: &FxHashMap<u32, BoundOperators<'_>>,
        state: &mut Shot,
    ) -> Result<Option<bool>, ExecError> {
        let guard = self.core.bloq.top()[obs].activation;
        if let Some(slot) = guard {
            let inputs = self.slot_inputs(self.core.bloq.top(), obs, top_values);
            if inputs.get(&slot) != Some(&true) {
                return Ok(None);
            }
        }
        let Some(mut value) = top_values.get(&obs.0).copied().flatten() else {
            return Ok(None);
        };
        let mut terminal_operator: Option<&PauliString> = None;
        let mut combined = false;
        let mut boundary_scratch = self.boundary_scratch.borrow_mut();
        let mut output_operators = self.core.bloq.top()[obs]
            .try_classical()
            .expect("observable is classical")
            .operators()
            .iter()
            .filter(|operator| operator.face == bloq_ir::BoundaryFace::Output)
            .collect::<Vec<_>>();
        let mut expanded = 0;
        for input in self
            .core
            .bloq
            .top()
            .data_inputs(obs)
            .filter(|input| input.output.is_none())
        {
            collect_bound_operators(
                top_bindings,
                input.producer,
                &mut output_operators,
                &mut expanded,
            )?;
        }
        let mut cached_bindings = Vec::new();
        for (read, bit) in self.early_reads.iter().zip(&state.early_faces) {
            if let Some(bit) = bit
                && read.observable == obs
                && read.bindings.iter().all(|binding| {
                    output_operators
                        .iter()
                        .any(|operator| std::ptr::eq(binding.operator(&self.core.bloq), *operator))
                })
            {
                value ^= bit;
                cached_bindings.extend(&read.bindings);
            }
        }
        for operator in output_operators {
            if !cached_bindings
                .iter()
                .any(|binding| std::ptr::eq(binding.operator(&self.core.bloq), operator))
            {
                let prepared = self.core.boundary_operator(&operator.operator)?;
                if let Some(first) = terminal_operator {
                    if !combined {
                        boundary_scratch.clone_from(first);
                        combined = true;
                    }
                    // Keep the exact phase: XX * ZZ = -YY.
                    *boundary_scratch = &*boundary_scratch * prepared;
                } else {
                    terminal_operator = Some(prepared);
                }
            }
        }
        if let Some(first) = terminal_operator {
            let pauli = if combined { &boundary_scratch } else { first };
            value ^= Self::measure_boundary_pauli(pauli, &mut state.sim)?;
        }
        Ok(Some(value))
    }

    fn plan_early_output_captures(&self) -> Result<Vec<EarlyOutputCapture>, ExecError> {
        let captures = self
            .core
            .bloq
            .logical_outputs()
            .iter()
            .zip(&self.output_cuts)
            .filter(|(_, cuts)| !cuts.is_empty())
            .map(|(output, cuts)| {
                Ok((
                    cuts.clone(),
                    output.instance,
                    output.port,
                    self.joint_pauli(&output.x)?,
                    self.joint_pauli(&output.z)?,
                ))
            })
            .collect::<Result<Vec<_>, ExecError>>()?;
        let base_width = self.core.coord_index.len();
        let variant_count = captures.len();
        // ponytail: C² width variants keep the shot path allocation-free;
        // replace them if captured-output counts grow beyond small verifier fixtures.
        Ok(captures
            .into_iter()
            .map(
                |(before, instance, port, logical_x, logical_z)| EarlyOutputCapture {
                    before,
                    instance,
                    port,
                    operators: (0..variant_count)
                        .map(|extra| {
                            let qubit = base_width + extra;
                            let width = qubit + 1;
                            EarlyOutputPaulis {
                                logical_x: super::circuit::widen_pauli(&logical_x, width),
                                logical_z: super::circuit::widen_pauli(&logical_z, width),
                                ancilla_x: PauliString::single(width, qubit, Pauli::X),
                                ancilla_z: PauliString::single(width, qubit, Pauli::Z),
                            }
                        })
                        .collect(),
                },
            )
            .collect())
    }

    /// Swap a logical output away immediately before another node reuses its
    /// patch. The program's Pauli frame remains for the terminal hook to apply.
    fn capture_early_outputs(
        &self,
        scope: &LevelPath,
        id: BloqNodeId,
        state: &mut Shot,
    ) -> Result<(), ExecError> {
        for (slot, capture) in self.early_output_captures.iter().enumerate() {
            if !capture
                .before
                .iter()
                .any(|(cut_scope, cut_id)| *cut_id == id && cut_scope == scope)
                || state.saved_outputs[slot].is_some()
                || state.excluded_instances.contains(&capture.instance)
            {
                continue;
            }
            let qubit = state.sim.num_qubits();
            let operators = capture
                .operators
                .get(qubit.saturating_sub(self.core.coord_index.len()))
                .ok_or(ExecError::MalformedGraph(
                    "early-output capture has no prepared width",
                ))?;
            state.sim.reset(qubit)?;
            state
                .sim
                .controlled_pauli(&operators.logical_z, &operators.ancilla_x)?;
            state
                .sim
                .controlled_pauli(&operators.ancilla_z, &operators.logical_x)?;
            state
                .sim
                .controlled_pauli(&operators.logical_z, &operators.ancilla_x)?;
            state.saved_outputs[slot] = Some(qubit);
        }
        Ok(())
    }

    /// Resolve pure classical dataflow at a cut, including computations whose
    /// inputs are ready but whose nodes have not yet been scheduled.
    fn available_value(
        &self,
        scope: &LevelPath,
        id: BloqNodeId,
        state: &Shot,
        memo: &mut FxHashMap<(LevelPath, BloqNodeId), Option<bool>>,
    ) -> Result<Option<bool>, ExecError> {
        let key = (scope.clone(), id);
        if let Some(&value) = state.classical_values.get(&key).or_else(|| memo.get(&key)) {
            return Ok(value);
        }
        let level = self
            .core
            .bloq
            .level_at(scope)
            .expect("binding scope exists");
        let Some(classical) = level[id].try_classical() else {
            return Ok(None);
        };
        if let Some(slot) = level[id].activation {
            let input = level
                .value_inputs(id)
                .find(|input| input.slot == slot)
                .ok_or(ExecError::MalformedGraph(
                    "classical guard has no value input",
                ))?;
            match zero_decoder_output(
                self.available_value(scope, input.producer, state, memo)?,
                input.output,
            ) {
                Some(true) => {}
                Some(false) => {
                    memo.insert(key, Some(false));
                    return Ok(Some(false));
                }
                None => return Ok(None),
            }
        }
        let mut values = FxHashMap::default();
        for input in level.data_inputs(id) {
            values.insert(
                input.producer.0,
                self.available_value(scope, input.producer, state, memo)?,
            );
        }
        let value = match self.classical_value(
            level,
            id,
            classical,
            &values,
            &state.record,
            &state.excluded_instances,
        ) {
            Err(ExecError::MissingRecord(_)) => Ok(None),
            result => result,
        }?;
        memo.insert(key, value);
        Ok(value)
    }

    fn binding_active(
        &self,
        binding: &super::closure::OutputBinding,
        state: &Shot,
        memo: &mut FxHashMap<(LevelPath, BloqNodeId), Option<bool>>,
    ) -> Result<Option<bool>, ExecError> {
        let mut active = Some(true);
        for (scope, guard) in &binding.guards {
            match zero_decoder_output(
                self.available_value(scope, guard.node, state, memo)?,
                Some(guard.output),
            ) {
                Some(false) => return Ok(Some(false)),
                None => active = None,
                Some(true) => {}
            }
        }
        Ok(active)
    }

    /// Take the planned reads of any `Output` face node `id` is about to destroy
    /// (see [`super::closure::plan_early_output_reads`]).
    fn read_early_faces(
        &self,
        scope: &LevelPath,
        id: BloqNodeId,
        state: &mut Shot,
    ) -> Result<(), ExecError> {
        // LIM-022: decide every gate at this cut before projecting any face.
        // Unknown values are cached only here: a later cut may have their records.
        let mut memo = FxHashMap::default();
        let reads = self
            .early_reads
            .iter()
            .enumerate()
            .filter(|(slot, read)| {
                read.before == (scope.clone(), id) && state.early_faces[*slot].is_none()
            })
            .map(|(slot, read)| {
                let active = self
                    .binding_active(&read.bindings[0], state, &mut memo)?
                    .ok_or(ExecError::EarlyOutputGateUnavailable {
                        observable: read.observable.0,
                        before: id.0,
                    })?;
                Ok((slot, read, active))
            })
            .collect::<Result<Vec<_>, ExecError>>()?;
        for (slot, read, active) in reads {
            if !active {
                continue;
            }
            let operators = read
                .bindings
                .iter()
                .filter(|binding| {
                    !self
                        .early_reads
                        .iter()
                        .zip(&state.early_faces)
                        .any(|(earlier, bit)| {
                            earlier.observable == read.observable
                                && bit.is_some()
                                && earlier.bindings.contains(binding)
                        })
                })
                .map(|binding| binding.operator(&self.core.bloq))
                .collect::<Vec<_>>();
            if operators.is_empty() {
                continue;
            }
            let bit = if let [operator] = operators.as_slice() {
                self.measure_boundary(operator, &mut state.sim)?
            } else {
                let mut pauli = PauliString::new(self.core.coord_index.len());
                for operator in operators {
                    pauli = &pauli * self.core.boundary_operator(&operator.operator)?;
                }
                Self::measure_boundary_pauli(&pauli, &mut state.sim)?
            };
            state.early_faces[slot] = Some(bit);
        }
        Ok(())
    }

    /// Read a boundary-operator Pauli (a logical face). The operator's
    /// coordinates are already instance-global (translated at lowering — see
    /// [`bloq_ir::InstanceBoundaryOperator`]), so they resolve directly; the
    /// instance lookup only validates that the graph names a known instance.
    ///
    /// A live logical face is deterministic, so the non-collapsing expectation
    /// reads it without disturbing the state — which is what makes an early read
    /// at a cut safe, with the rest of the program still to run. A face with no
    /// definite value (`⟨P⟩ = 0`, or a fractional Clifford+T expectation) has
    /// nothing to read non-destructively, so it is projected, as the terminal
    /// read has always done.
    fn measure_boundary(
        &self,
        operator: &bloq_ir::lowering::InstanceBoundaryOperator,
        sim: &mut Simulator,
    ) -> Result<bool, ExecError> {
        if !self.core.instance_offset.contains_key(&operator.instance) {
            return Err(ExecError::MalformedGraph(
                "boundary operator names an unknown instance",
            ));
        }
        if operator.operator.is_empty() {
            return Err(ExecError::EmptyMppProduct);
        }
        let pauli = self.core.boundary_operator(&operator.operator)?;
        Self::measure_boundary_pauli(pauli, sim)
    }

    fn measure_boundary_pauli(pauli: &PauliString, sim: &mut Simulator) -> Result<bool, ExecError> {
        let expectation = sim.peek_observable_expectation(pauli)?;
        if (1.0 - expectation.abs()) < DETERMINISTIC_EXPECTATION_TOLERANCE {
            return Ok(expectation < 0.0);
        }
        Ok(sim.measure_observable(pauli)?.outcome)
    }

    /// Build the joint Pauli of a `PauliMap` in global engine indices. The map's
    /// coordinates must already be instance-global — boundary operators are
    /// translated by their instance offset at lowering (see
    /// [`bloq_ir::InstanceBoundaryOperator`]) — so the layout carries no offset;
    /// a caller holding patch-local coordinates translates them first.
    fn joint_pauli(&self, map: &PauliMap) -> Result<PauliString, ExecError> {
        super::circuit::product_pauli(
            map,
            &super::circuit::LayoutCtx {
                coord_to_index: &self.core.coord_index,
                qubit_count: self.core.coord_index.len(),
            },
        )
    }

    /// Physical patches and logical operators for every compiler-authored
    /// program input.
    fn input_patches(&self) -> Result<Vec<InputPatch>, ExecError> {
        let mut placements = FxHashMap::default();
        let mut spatial_cubes: FxHashMap<_, Vec<_>> = FxHashMap::default();
        let mut spatial_waits = FxHashMap::default();
        self.core.bloq.walk(|cx| {
            if let Some(quantum) = cx.node.try_quantum() {
                placements.extend(
                    quantum
                        .instances
                        .iter()
                        .map(|instance| (instance.id, (instance.template_id, instance.offset))),
                );
                for instance in &quantum.instances {
                    if matches!(
                        instance.provenance,
                        bloq_ir::InstanceProvenance::SpatialPortSubstitution {
                            role: bloq_ir::PortRole::Input,
                            part: bloq_ir::SpatialPortPart::TemporalPort,
                            ..
                        }
                    ) {
                        // Inject before held rounds; a later reset erases their syndrome history.
                        let wait = cx.level.outgoing(cx.id).find_map(|edge| {
                            if !matches!(edge.edge, bloq_ir::BloqEdge::Quantum(_)) {
                                return None;
                            }
                            let node = &cx.level[edge.target];
                            node.memory_rounds()?;
                            node.try_quantum()?
                                .instances
                                .iter()
                                .find(|wait| wait.offset == instance.offset)
                        });
                        if let Some(wait) = wait {
                            spatial_waits.insert(instance.id, wait.id);
                        }
                    }
                    if let bloq_ir::InstanceProvenance::SpatialPortSubstitution {
                        source,
                        role,
                        part: bloq_ir::SpatialPortPart::Cube,
                    } = instance.provenance
                        && role == bloq_ir::PortRole::Input
                    {
                        spatial_cubes.entry(source).or_default().push(instance.id);
                    }
                }
            }
            bloq_ir::WalkControl::Continue
        });
        self.core
            .bloq
            .logical_inputs()
            .iter()
            .map(|input| {
                let &(template_id, offset) =
                    placements
                        .get(&input.instance)
                        .ok_or(ExecError::MalformedGraph(
                            "logical input names an unknown instance",
                        ))?;
                let template = &self.core.bloq.templates()[template_id];
                let maps = template
                    .boundary_flows
                    .iter()
                    .filter(|flow| flow.start.is_empty() && !flow.end.is_empty())
                    .map(|flow| {
                        if flow.sign {
                            return Err(ExecError::MalformedGraph(
                                "input Port has a signed create stabilizer",
                            ));
                        }
                        Ok(flow.end.try_translated(offset)?)
                    })
                    .collect::<Result<Vec<_>, ExecError>>()?;
                let mut data = rustc_hash::FxHashSet::default();
                for map in &maps {
                    data.extend(map.iter().map(|(coordinate, _)| *coordinate));
                }
                data.extend(
                    input
                        .x
                        .iter()
                        .chain(input.z.iter())
                        .map(|(coordinate, _)| *coordinate),
                );
                let mut data = data.into_iter().collect::<Vec<_>>();
                data.sort_by_key(glam::IVec2::to_array);
                let data_qubits = data
                    .into_iter()
                    .map(|coordinate| {
                        self.core
                            .coord_index
                            .get(&coordinate)
                            .map(|&index| index as usize)
                            .ok_or(ExecError::UnknownCoord(coordinate))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let stabilizers = maps
                    .iter()
                    .map(|map| self.joint_pauli(map))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(InputPatch {
                    input_instance: input.instance,
                    prepare_before: spatial_waits
                        .get(&input.instance)
                        .copied()
                        .map(|instance| vec![instance])
                        .or_else(|| spatial_cubes.get(&input.port).cloned())
                        .unwrap_or_else(|| vec![input.instance]),
                    port: input.port,
                    data_qubits,
                    stabilizers,
                    logical_x: self.joint_pauli(&input.x)?,
                    logical_z: self.joint_pauli(&input.z)?,
                })
            })
            .collect()
    }

    /// The logical operators of every program output, for the per-shot hook.
    ///
    /// The IR stores both operators in layout-global coordinates; this method
    /// only resolves those coordinates to engine indices.
    fn output_logicals(&self) -> Result<Vec<OutputLogical>, ExecError> {
        let outputs = self.core.bloq.logical_outputs();
        outputs
            .iter()
            .zip(&self.output_cuts)
            .map(|(output, cuts)| {
                Ok(OutputLogical {
                    port: output.port,
                    logical_x: self.joint_pauli(&output.x)?,
                    logical_z: self.joint_pauli(&output.z)?,
                    consumed: !cuts.is_empty(),
                })
            })
            .collect()
    }
}

// ============================================================
// Static collection (detectors, observables)
// ============================================================

pub(super) fn collect_rus_restarts(
    bloq: &Bloq,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<FxHashMap<LevelPath, Vec<(LevelPath, ResolvedDetector)>>, ExecError> {
    fn visit_regions(
        bloq: &Bloq,
        level: &SubGraph,
        scope: &LevelPath,
        global_meas: &FxHashMap<InstanceMeasurement, u32>,
        out: &mut FxHashMap<LevelPath, Vec<(LevelPath, ResolvedDetector)>>,
    ) -> Result<(), ExecError> {
        for (id, node) in level.nodes() {
            let Some(region) = node.try_region() else {
                continue;
            };
            for (selector, body) in region.bodies() {
                let child = scope.child(id, selector);
                if matches!(region, RegionNode::RepeatUntilSuccess { .. }) {
                    let mut parities = Vec::new();
                    collect_restart_parities(bloq, body, &child, global_meas, &mut parities)?;
                    out.insert(child.clone(), parities);
                }
                visit_regions(bloq, body, &child, global_meas, out)?;
            }
        }
        Ok(())
    }

    fn collect_restart_parities(
        bloq: &Bloq,
        level: &SubGraph,
        scope: &LevelPath,
        global_meas: &FxHashMap<InstanceMeasurement, u32>,
        out: &mut Vec<(LevelPath, ResolvedDetector)>,
    ) -> Result<(), ExecError> {
        for (id, node) in level.nodes() {
            if let Some(quantum) = node.try_quantum() {
                for instance in &quantum.instances {
                    let template = bloq.templates().get(instance.template_id).ok_or(
                        ExecError::MalformedGraph("instance references a missing template"),
                    )?;
                    for restart in &template.restarts {
                        out.push((
                            scope.clone(),
                            resolve_template_parity(&restart.parity, instance.id, global_meas)?
                                .guarded(
                                    id,
                                    quantum
                                        .guards
                                        .iter()
                                        .find(|guard| guard.instances.contains(&instance.id))
                                        .map(|guard| guard.input),
                                ),
                        ));
                    }
                }
                for (index, restart) in quantum.restarts.iter().enumerate() {
                    let restart_index =
                        u32::try_from(index).map_err(|_| ExecError::IdOverflow("restart"))?;
                    out.push((
                        scope.clone(),
                        resolve_node_parity(&restart.parity, global_meas)?
                            .with_contributions(
                                id,
                                quantum.guards.iter().flat_map(|guard| {
                                    guard
                                        .restart_parities
                                        .iter()
                                        .filter(move |(row, _)| *row as usize == index)
                                        .map(move |(_, parity)| {
                                            (guard.input, resolve_node_parity(parity, global_meas))
                                        })
                                }),
                            )?
                            .guarded(
                                id,
                                quantum
                                    .guards
                                    .iter()
                                    .find(|guard| guard.restarts.contains(&restart_index))
                                    .map(|guard| guard.input),
                            ),
                    ));
                }
            }
            if let Some(region) = node.try_region()
                && !matches!(region, RegionNode::RepeatUntilSuccess { .. })
            {
                for (selector, body) in region.bodies() {
                    let child = scope.child(id, selector);
                    collect_restart_parities(bloq, body, &child, global_meas, out)?;
                }
            }
        }
        Ok(())
    }

    let mut out = FxHashMap::default();
    visit_regions(
        bloq,
        bloq.top(),
        &LevelPath::default(),
        global_meas,
        &mut out,
    )?;
    Ok(out)
}

/// Collect all template-local + node detectors with their owning execution
/// scope, resolved to global record-id parities.
fn collect_detectors(
    bloq: &Bloq,
    level: &SubGraph,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<Vec<(LevelPath, ResolvedDetector)>, ExecError> {
    let mut detectors = Vec::new();
    collect_detectors_into(
        bloq,
        level,
        &LevelPath::default(),
        global_meas,
        &mut detectors,
    )?;
    Ok(detectors)
}

fn collect_detectors_into(
    bloq: &Bloq,
    level: &SubGraph,
    scope: &LevelPath,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
    out: &mut Vec<(LevelPath, ResolvedDetector)>,
) -> Result<(), ExecError> {
    for (id, node) in level.nodes() {
        if let Some(quantum) = node.try_quantum() {
            for instance in &quantum.instances {
                let template =
                    bloq.templates()
                        .get(instance.template_id)
                        .ok_or(ExecError::MalformedGraph(
                            "instance references a missing template",
                        ))?;
                for detector in &template.detectors {
                    out.push((
                        scope.clone(),
                        resolve_template_parity(&detector.parity, instance.id, global_meas)?
                            .guarded(
                                id,
                                quantum
                                    .guards
                                    .iter()
                                    .find(|guard| guard.instances.contains(&instance.id))
                                    .map(|guard| guard.input),
                            ),
                    ));
                }
            }
            for detector in bloq.node_detectors(quantum)? {
                let parity = resolve_node_parity(&detector.parity(), global_meas)?;
                let (membership, parity) = match detector.address() {
                    NodeDetectorAddress::Inline(index) => {
                        let membership = quantum
                            .guards
                            .iter()
                            .find(|guard| guard.detectors.contains(&index))
                            .map(|guard| guard.input);
                        let parity = parity.with_contributions(
                            id,
                            quantum.guards.iter().flat_map(|guard| {
                                guard
                                    .detector_parities
                                    .iter()
                                    .filter(move |(row, _)| *row == index)
                                    .map(move |(_, parity)| {
                                        (guard.input, resolve_node_parity(parity, global_meas))
                                    })
                            }),
                        )?;
                        (membership, parity)
                    }
                    NodeDetectorAddress::Bundle { use_index, .. } => {
                        let membership = quantum
                            .guards
                            .iter()
                            .find(|guard| guard.detector_bundles.contains(&use_index))
                            .map(|guard| guard.input);
                        (membership, parity)
                    }
                };
                out.push((scope.clone(), parity.guarded(id, membership)));
            }
        }
        if let Some(region) = node.try_region() {
            for (selector, body) in region.bodies() {
                let child = scope.child(id, selector);
                collect_detectors_into(bloq, body, &child, global_meas, out)?;
            }
        }
    }
    Ok(())
}

/// Top-level `Observable` nodes in node order.
fn collect_observables(bloq: &Bloq) -> Vec<BloqNodeId> {
    bloq.top()
        .nodes()
        .filter_map(|(id, node)| match node.try_classical() {
            Some(ClassicalNode::Observable { index: Some(_), .. }) => Some(id),
            _ => None,
        })
        .collect()
}

/// A detector's sign and `(owning instance, global record id)` terms, so
/// per-shot evaluation can normalize its expected value and distinguish
/// deselected instances' records from genuine ones.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ResolvedDetector {
    pub sign: bool,
    pub terms: Vec<(TemplateInstanceId, u32)>,
    membership: Option<(BloqNodeId, u32)>,
    conditioned: Vec<(BloqNodeId, u32, ResolvedDetector)>,
}

impl ResolvedDetector {
    fn with_contributions(
        mut self,
        node: BloqNodeId,
        contributions: impl Iterator<Item = (u32, Result<ResolvedDetector, ExecError>)>,
    ) -> Result<Self, ExecError> {
        for (input, parity) in contributions {
            let parity = parity?;
            self.conditioned.push((node, input, parity));
        }
        Ok(self)
    }
    fn guarded(mut self, node: BloqNodeId, input: Option<u32>) -> Self {
        self.membership = input.map(|slot| (node, slot));
        self
    }

    fn applies(&self, scope: &LevelPath, state: &Shot) -> bool {
        state.executed_scopes.contains(scope)
            && self.membership.is_none_or(|(node, slot)| {
                state
                    .active_membership_inputs
                    .contains(&(scope.clone(), node, slot))
            })
    }
}

/// Resolve a parity's measurement terms to `(owning instance, global record)`.
/// `owner` maps a term to the `InstanceMeasurement` that keys `global_meas` —
/// template parities synthesise it from their fixed instance, node parities
/// carry it — which is the only way the two callers differ.
fn resolve_parity<T: Copy + Ord>(
    parity: &DetectorParity<T>,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
    owner: impl Fn(T) -> InstanceMeasurement,
    unknown: &'static str,
) -> Result<ResolvedDetector, ExecError> {
    let mut ids = Vec::with_capacity(parity.terms().len());
    for term in parity.terms() {
        match term {
            DetectorTerm::Measurement(m) => {
                let key = owner(*m);
                let global = *global_meas
                    .get(&key)
                    .ok_or(ExecError::MalformedGraph(unknown))?;
                ids.push((key.instance, global));
            }
            DetectorTerm::LoopState(state) => return Err(ExecError::UnresolvedLoopState(state.0)),
        }
    }
    Ok(ResolvedDetector {
        sign: parity.sign(),
        terms: ids,
        membership: None,
        conditioned: Vec::new(),
    })
}

fn resolve_template_parity(
    parity: &DetectorParity<u32>,
    instance: TemplateInstanceId,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<ResolvedDetector, ExecError> {
    resolve_parity(
        parity,
        global_meas,
        |measurement| InstanceMeasurement {
            instance,
            measurement,
        },
        "template detector names an unknown measurement",
    )
}

fn resolve_node_parity(
    parity: &DetectorParity<InstanceMeasurement>,
    global_meas: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<ResolvedDetector, ExecError> {
    resolve_parity(
        parity,
        global_meas,
        |im| im,
        "node detector names an unknown measurement",
    )
}

/// Every instance id living anywhere in `level` (recursing region bodies).
pub(crate) fn collect_instances(
    level: &SubGraph,
    out: &mut rustc_hash::FxHashSet<TemplateInstanceId>,
) {
    for (_, node) in level.nodes() {
        if let Some(quantum) = node.try_quantum() {
            out.extend(quantum.instances.iter().map(|i| i.id));
        }
        if let Some(region) = node.try_region() {
            for (_, body) in region.bodies() {
                collect_instances(body, out);
            }
        }
    }
}

/// The region's declared bit result (SEM-RVAL), or constant false. A selected
/// producer with unavailable inputs remains unavailable.
fn bound_body_value(body: &SubGraph, body_values: &FxHashMap<u32, Option<bool>>) -> Option<bool> {
    match body.value_output() {
        Some(producer) => zero_decoder_output(
            body_values.get(&producer.node.0).copied().flatten(),
            Some(producer.output),
        ),
        None => Some(false),
    }
}

/// Without decoder errors, Flip is false but unavailable producers stay unavailable.
fn zero_decoder_output(value: Option<bool>, output: Option<ObservableOutput>) -> Option<bool> {
    value.map(|bit| bit && output != Some(ObservableOutput::Flip))
}

/// Declared boundary outputs, independently of the body's bit result.
fn bound_body_bindings<'a>(
    body: &'a SubGraph,
    body_bindings: &FxHashMap<u32, BoundOperators<'a>>,
) -> Result<Vec<&'a InstanceBoundaryOperator>, ExecError> {
    let mut operators = Vec::new();
    let mut expanded = 0;
    for &producer in body.boundary_outputs() {
        collect_bound_operators(body_bindings, producer, &mut operators, &mut expanded)?;
    }
    Ok(operators)
}

/// Evaluate a detector whose owner scope ran this shot. Terms naming instances
/// excluded from this shot are dropped. Owner scope and membership guards,
/// not record terms, decide detector applicability.
/// Any other missing record is a hard [`ExecError::MissingRecord`].
fn xor_records(
    parity: &ResolvedDetector,
    scope: &LevelPath,
    state: &Shot,
) -> Result<bool, ExecError> {
    let mut value = parity.sign;
    for &(instance, id) in &parity.terms {
        if state.excluded_instances.contains(&instance) {
            continue;
        }
        value ^= state.record.require(id)?;
    }
    for (node, input, contribution) in &parity.conditioned {
        if !state
            .active_membership_inputs
            .contains(&(scope.clone(), *node, *input))
        {
            continue;
        }
        value ^= contribution.sign;
        for &(instance, id) in &contribution.terms {
            if !state.excluded_instances.contains(&instance) {
                value ^= state.record.require(id)?;
            }
        }
    }
    Ok(value)
}

fn rus_attempt_seed(seed: u64, scope: &LevelPath, attempt: u32) -> u64 {
    let mut state = splitmix64(seed ^ 0x5255_535F_5254_5259);
    for segment in scope.segments() {
        state = splitmix64(state ^ u64::from(segment.region.0));
        state = splitmix64(state);
    }
    splitmix64(state ^ u64::from(attempt))
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{guarded_node, one_qubit_template, quantum_node};
    use bloq_ir::BoundaryFace;
    use bloq_ir::circuit::{CoordCircuit, GateType, Pauli as CircuitPauli, PauliBasis};
    use bloq_ir::lowering::{BloqTemplate, NodeRestart};
    use bloq_ir::{BloqEdge, BloqNode, ClassicalExpr, NodeDetector, TemplateDetector};
    use glam::IVec2;

    #[test]
    fn early_cut_activation_uses_the_selected_observable_port() {
        for output in [ObservableOutput::Corrected, ObservableOutput::Flip] {
            let mut bloq = Bloq::new();
            let bit = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(true),
            }));
            let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(bit, observable, BloqEdge::value(0));
            let mut guarded = BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(true),
            });
            guarded.activation = Some(0);
            let guarded = bloq.add_node(guarded);
            bloq.add_edge(
                observable,
                guarded,
                BloqEdge::Value {
                    output,
                    slot: 0,
                    role: bloq_ir::ValueRole::Data,
                },
            );
            bloq.validate().unwrap();
            let program = Program::prepare(&bloq, RunMode::Plain).unwrap();
            let mut completed = program
                .execute_shot(0, 0, false, &mut |_, _| Ok(()))
                .unwrap();
            let expected = Some(output == ObservableOutput::Corrected);
            assert_eq!(completed.top_values[&guarded.0], expected);

            completed.state.classical_values.clear();
            let scope = LevelPath::default();
            let mut memo = FxHashMap::default();
            assert_eq!(
                program
                    .available_value(&scope, guarded, &completed.state, &mut memo)
                    .unwrap(),
                expected
            );
            completed
                .state
                .classical_values
                .insert((scope.clone(), observable), None);
            memo.clear();
            assert_eq!(
                program
                    .available_value(&scope, guarded, &completed.state, &mut memo)
                    .unwrap(),
                None
            );
        }
    }

    #[test]
    fn unavailable_discard_condition_errors_when_executed() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::In(0),
        }));

        assert!(matches!(
            bloq.validate(),
            Err(bloq_ir::BloqValidationError::MissingClassicalInput { slot: 0, .. })
        ));
        assert!(matches!(
            run_bloq(&bloq, 1, 0),
            Err(ExecError::MalformedGraph(
                "discard condition is unavailable"
            ))
        ));
    }

    #[test]
    fn conditional_members_execute_once_with_selected_record_aliases() {
        let mut bloq = Bloq::new();
        let mut coin = CoordCircuit::new();
        coin.do_gate(GateType::H, [IVec2::X]).unwrap();
        let bit = coin.measure(PauliBasis::Z, [IVec2::X])[0];
        let coin_template = bloq.add_template(BloqTemplate::new(coin));
        let coin = bloq.add_node(quantum_instance(TemplateInstanceId(0), coin_template));
        let selector = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: bit,
            }],
            Vec::new(),
        )));
        bloq.add_edge(coin, selector, BloqEdge::Order);
        let opposite = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(selector, opposite, BloqEdge::value(0));

        // The optional duplicate precedes the common record in instance order,
        // so its selection changes the common record's physical alias.
        let mut measure = CoordCircuit::new();
        let measurement = measure.measure(PauliBasis::Z, [IVec2::ZERO])[0];
        let template = bloq.add_template(BloqTemplate::new(measure));
        let mut registry = quantum_instance(TemplateInstanceId(1), template);
        registry.expect_quantum_mut().instances.push(
            quantum_instance(TemplateInstanceId(2), template)
                .expect_quantum()
                .instances[0],
        );
        registry
            .expect_quantum_mut()
            .guards
            .push(bloq_ir::QuantumGuard {
                input: 0,
                instances: vec![TemplateInstanceId(1)],
                detectors: vec![0],
                restarts: vec![],
                ..Default::default()
            });
        registry.expect_quantum_mut().detectors.push(NodeDetector {
            parity: DetectorParity::default().with_sign(true),
            coords: None,
        });
        let registry = bloq.add_node(registry);
        bloq.add_edge(selector, registry, BloqEdge::value(0));
        let parity = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(2),
                measurement,
            }],
            Vec::new(),
        )));
        bloq.add_edge(registry, parity, BloqEdge::Order);
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(parity, observable, BloqEdge::value(0));
        let choice = bloq.add_node(BloqNode::classical(ClassicalNode::observable(1)));
        bloq.add_edge(selector, choice, BloqEdge::value(0));
        let report = run_bloq(&bloq, 64, 0).unwrap();
        assert_eq!(report.observables[0].per_shot, [false; 64]);
        let enabled = report.observables[1]
            .per_shot
            .iter()
            .filter(|&&bit| bit)
            .count();
        assert!(enabled > 0 && enabled < 64);
        assert_eq!(report.detectors[0].per_shot, vec![true; enabled]);
    }

    #[test]
    fn retry_result_is_independent_of_restart_and_internal_values() {
        let program = Bloq::from_text(
            "BLOQIR 1
graph {
 n0 rus in0 source n0 {
   body {
     n0 compute 0
     n1 compute 1
     n2 compute !in0
     n3 compute 0
     n1 -> n2 value 0
     result n1
   }
 }
 n1 observable 0
 n0 -> n1 value 0
}",
        )
        .unwrap();
        for mut program in [
            program.clone(),
            Bloq::from_binary(&program.to_binary()).unwrap(),
        ] {
            program.optimize().expect("acyclic test program");
            let report = run_bloq(&program, 8, 0).unwrap();
            assert_eq!(report.observables[0].per_shot, [true; 8]);
        }
    }

    #[test]
    fn region_exports_its_declared_observable_measurement() {
        for bit in [false, true] {
            let mut circuit = CoordCircuit::new();
            if bit {
                circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
            }
            let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
            let mut bloq = Bloq::new();
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let mut body = SubGraph::new();
            let quantum = body.add_node(quantum_instance(TemplateInstanceId(0), template));
            let parity = body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                vec![InstanceMeasurement {
                    instance: TemplateInstanceId(0),
                    measurement,
                }],
                Vec::new(),
            )));
            let observable = body.add_node(BloqNode::classical(ClassicalNode::observable(1)));
            body.add_edge(quantum, parity, BloqEdge::Order);
            body.add_edge(parity, observable, BloqEdge::value(0));
            body.set_value_output(Some(observable.into()));
            let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
                restart_condition: ClassicalExpr::Const(false),
                restart_source: None,
                body,
            }));
            let output = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(region, output, BloqEdge::value(0));
            let report = run_bloq(&bloq, 1, 0).unwrap();
            assert_eq!(report.observables[0].per_shot, [bit]);
        }
    }

    #[test]
    fn discarded_shot_keeps_evaluated_branch_selector() {
        let mut bloq = Bloq::new();
        let branch = bloq.add_node(selector_node(ClassicalExpr::Const(true)));
        let discard = bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::Const(true),
        }));
        bloq.add_edge(branch, discard, BloqEdge::Order);

        let report = run_bloq(&bloq, 1, 0).expect("valid program executes");
        assert_eq!(report.discarded, 1);
        assert_eq!(report.branch_selectors, [true]);
    }

    fn selector_node(expr: ClassicalExpr) -> BloqNode {
        BloqNode::classical(ClassicalNode::Compute { expr }).with_provenance(
            bloq_ir::NodeProvenance::BranchSelector {
                name: "choice".into(),
            },
        )
    }

    fn quantum_instance(instance: TemplateInstanceId, template: bloq_ir::TemplateId) -> BloqNode {
        quantum_node(template, instance.0, IVec2::ZERO)
    }

    fn guarded_detector_program(take_arm: bool) -> Bloq {
        let mut prefix_circuit = CoordCircuit::new();
        let prefix_measurement = prefix_circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
        let mut bloq = Bloq::new();
        let prefix_template = bloq.add_template(BloqTemplate::new(prefix_circuit));
        let prefix_instance = TemplateInstanceId(0);
        let prefix = bloq.add_node(quantum_instance(prefix_instance, prefix_template));

        let mut arm_circuit = CoordCircuit::new();
        arm_circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
        let mut arm_template = BloqTemplate::new(arm_circuit);
        arm_template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::default().with_sign(true),
            coords: None,
        });
        let arm_template = bloq.add_template(arm_template);
        let mut arm_node = quantum_instance(TemplateInstanceId(1), arm_template);
        arm_node.expect_quantum_mut().detectors.push(NodeDetector {
            parity: DetectorParity::from_measurements([InstanceMeasurement {
                instance: prefix_instance,
                measurement: prefix_measurement,
            }])
            .with_sign(true),
            coords: None,
        });
        let selector = bloq.add_node(selector_node(ClassicalExpr::Const(take_arm)));
        let selected = bloq.add_node(guarded_node(arm_node, 0));
        bloq.add_edge(prefix, selected, BloqEdge::Order);
        bloq.add_edge(selector, selected, BloqEdge::value(0));
        bloq
    }

    fn selected_detector_contribution_program(take_arm: bool) -> Bloq {
        let mut bloq = Bloq::new();
        let selector = bloq.add_node(selector_node(ClassicalExpr::Const(take_arm)));
        let opposite = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(selector, opposite, BloqEdge::value(0));
        let mut downstream = BloqNode::from_members(Vec::new());
        downstream
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: DetectorParity::default(),
                coords: None,
            });
        let mut owners = Vec::new();
        for (instance, active) in [(0, opposite), (1, selector)] {
            let mut circuit = CoordCircuit::new();
            if instance == 1 {
                circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
            }
            let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let selected = bloq.add_node(guarded_node(
                quantum_node(template, instance, IVec2::ZERO),
                0,
            ));
            bloq.add_edge(active, selected, BloqEdge::value(0));
            owners.push(selected);
            downstream
                .expect_quantum_mut()
                .guards
                .push(bloq_ir::QuantumGuard {
                    input: instance,
                    detector_parities: vec![(
                        0,
                        DetectorParity::from_measurements([InstanceMeasurement {
                            instance: TemplateInstanceId(instance),
                            measurement,
                        }]),
                    )],
                    ..Default::default()
                });
        }
        let downstream = bloq.add_node(downstream);
        bloq.add_edge(opposite, downstream, BloqEdge::value(0));
        bloq.add_edge(selector, downstream, BloqEdge::value(1));
        for owner in owners {
            bloq.add_edge(owner, downstream, BloqEdge::Order);
        }
        bloq
    }

    fn signed_detector_node() -> BloqNode {
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().detectors.push(NodeDetector {
            parity: DetectorParity::default().with_sign(true),
            coords: None,
        });
        node
    }

    fn signed_restart_node() -> BloqNode {
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().restarts.push(NodeRestart {
            parity: DetectorParity::default().with_sign(true),
        });
        node
    }

    #[test]
    fn guarded_detectors_only_report_when_selected() {
        let skipped = run_bloq(&guarded_detector_program(false), 3, 0).unwrap();
        assert_eq!(skipped.detectors.len(), 2);
        assert!(skipped.detectors.iter().all(|detector| {
            detector.constant && detector.value.is_none() && detector.per_shot.is_empty()
        }));

        let taken = run_bloq(&guarded_detector_program(true), 3, 0).unwrap();
        assert_eq!(taken.detectors.len(), 2);
        assert!(
            taken
                .detectors
                .iter()
                .all(|detector| detector.per_shot == [true; 3])
        );
    }

    #[test]
    fn guarded_shared_detector_reads_two_owner_instances() {
        use bloq_ir::{BundleDetector, BundleMeasurement, DetectorBundle, DetectorBundleUse};

        for enabled in [false, true] {
            let mut bloq = Bloq::new();
            let mut circuit = CoordCircuit::new();
            let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let bundle = bloq.add_detector_bundle(DetectorBundle::new(
                vec![template, template],
                vec![BundleDetector {
                    parity: DetectorParity::from_measurements([
                        BundleMeasurement {
                            owner: 0,
                            measurement,
                        },
                        BundleMeasurement {
                            owner: 1,
                            measurement,
                        },
                    ])
                    .with_sign(true),
                    coords: None,
                }],
            ));
            let mut node = BloqNode::from_members(Vec::new());
            let quantum = node.expect_quantum_mut();
            quantum.instances = vec![
                bloq_ir::lowering::TemplateInstance::new(
                    TemplateInstanceId(0),
                    template,
                    IVec2::ZERO,
                ),
                bloq_ir::lowering::TemplateInstance::new(TemplateInstanceId(1), template, IVec2::X),
            ];
            quantum.detector_bundles.push(DetectorBundleUse {
                bundle,
                instances: vec![TemplateInstanceId(0), TemplateInstanceId(1)],
                offset: IVec2::ZERO,
            });
            quantum.guards.push(bloq_ir::QuantumGuard {
                input: 0,
                detector_bundles: vec![0],
                ..Default::default()
            });
            let selector = bloq.add_node(selector_node(ClassicalExpr::Const(enabled)));
            let node = bloq.add_node(node);
            bloq.add_edge(selector, node, BloqEdge::value(0));

            bloq.validate().expect("shared detector binding validates");
            let report = run_bloq(&bloq, 2, 0).expect("program executes");
            assert_eq!(report.detectors.len(), 1);
            assert_eq!(
                report.detectors[0].per_shot,
                if enabled { vec![true; 2] } else { vec![] }
            );

            let other = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
            let wrong_bundle =
                bloq.add_detector_bundle(DetectorBundle::new(vec![other, template], vec![]));
            bloq.node_mut(node)
                .unwrap()
                .expect_quantum_mut()
                .detector_bundles[0]
                .bundle = wrong_bundle;
            assert!(matches!(
                run_bloq(&bloq, 1, 0),
                Err(ExecError::DetectorBundle(
                    bloq_ir::DetectorBundleError::OwnerTemplateMismatch { .. }
                ))
            ));
            assert!(matches!(
                crate::lower::lower(&bloq, &crate::lower::LoweringConfig::default()),
                Err(crate::lower::LowerError::DetectorBundle(
                    bloq_ir::DetectorBundleError::OwnerTemplateMismatch { .. }
                ))
            ));
        }
    }

    #[test]
    fn downstream_detector_folds_only_selected_contribution() {
        for chosen in [false, true] {
            let bloq = selected_detector_contribution_program(chosen);
            bloq.validate()
                .expect("conditional detector program is valid");

            let report = run_bloq(&bloq, 1, 0).expect("guarded program executes");
            assert_eq!(report.detectors[0].per_shot, [chosen]);
        }
    }

    /// Rejected attempts must roll back membership state so their detectors
    /// never leak into the accepted shot.
    #[test]
    fn rejected_rus_attempt_does_not_leave_active_detectors() {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
        let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut body = SubGraph::new();
        let measure = body.add_node(quantum_instance(TemplateInstanceId(0), template));
        let selector = body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
            Vec::new(),
        )));
        let opposite = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        body.add_edge(measure, selector, BloqEdge::Order);
        body.add_edge(selector, opposite, BloqEdge::value(0));
        for (active, restarting) in [(opposite, true), (selector, false)] {
            let mut quantum = signed_detector_node();
            if restarting {
                quantum.expect_quantum_mut().restarts.push(NodeRestart {
                    parity: DetectorParity::default().with_sign(true),
                });
            }
            let node = guarded_node(quantum, 0);
            let selected = body.add_node(node);
            body.add_edge(active, selected, BloqEdge::value(0));
        }
        let restart_source = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(restart_source.into()),
        }));
        let report = run_bloq(&bloq, 8, 0).unwrap();
        assert_eq!(report.discarded, 0);
        assert!(report.detectors[0].per_shot.is_empty());
        assert_eq!(report.detectors[1].per_shot, [true; 8]);
    }

    #[test]
    fn guarded_restart_only_applies_when_selected() {
        let program = |take_restart_arm| {
            let mut body = SubGraph::new();
            let selector = body.add_node(selector_node(ClassicalExpr::Const(take_restart_arm)));
            let restart = body.add_node(guarded_node(signed_restart_node(), 0));
            body.add_edge(selector, restart, BloqEdge::value(0));
            let restart_source = body.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            }));
            let mut bloq = Bloq::new();
            bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
                body,
                restart_condition: ClassicalExpr::Const(false),
                restart_source: Some(restart_source.into()),
            }));
            bloq
        };
        let kept = run_bloq(&program(false), 1, 0).unwrap();
        assert_eq!(kept.discarded, 0);
        let rejected = run_bloq(&program(true), 1, 0).unwrap();
        assert_eq!(rejected.discarded, 1);
    }

    /// A RUS region's attempts draw from their own reseeded RNG stream and hand
    /// the parent stream back untouched (`restore_rng_from`), so a retrying RUS
    /// prefix must not shift the randomness of anything scheduled after it. The
    /// prefix here retries a shot-dependent number of times; its downstream
    /// selectors must still match the program without the prefix.
    #[test]
    fn transparent_rus_retry_preserves_downstream_randomness() {
        // H + Z-measure: one random bit per instance.
        let coin = |coord: IVec2| {
            let mut circuit = CoordCircuit::new();
            circuit.do_gate(GateType::H, [coord]).expect("H is valid");
            let measurement = circuit.measure(PauliBasis::Z, [coord])[0];
            (BloqTemplate::new(circuit), measurement)
        };

        let program = |with_prefix: bool| {
            let mut bloq = Bloq::new();

            // A RUS body that keeps retrying until its own coin lands `true`.
            let prefix = with_prefix.then(|| {
                let (template, measurement) = coin(IVec2::new(2, 0));
                let template = bloq.add_template(template);
                let instance = TemplateInstanceId(1);
                let mut body = SubGraph::new();
                let flip = body.add_node(quantum_instance(instance, template));
                let restart_source =
                    body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                        vec![InstanceMeasurement {
                            instance,
                            measurement,
                        }],
                        Vec::new(),
                    )));
                body.add_edge(flip, restart_source, BloqEdge::Order);
                bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
                    body,
                    restart_condition: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
                    restart_source: Some(restart_source.into()),
                }))
            });

            let (template, measurement) = coin(IVec2::ZERO);
            let template = bloq.add_template(template);
            let instance = TemplateInstanceId(0);
            let quantum = bloq.add_node(quantum_instance(instance, template));
            let selector = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                vec![InstanceMeasurement {
                    instance,
                    measurement,
                }],
                Vec::new(),
            )));
            let branch = bloq.add_node(selector_node(ClassicalExpr::In(0)));
            if let Some(prefix) = prefix {
                bloq.add_edge(prefix, quantum, BloqEdge::Order);
            }
            bloq.add_edge(quantum, selector, BloqEdge::Order);
            bloq.add_edge(selector, branch, BloqEdge::value(0));
            bloq
        };

        let plain = run_bloq(&program(false), 64, 17).expect("prefix-free program executes");
        let retried = run_bloq(&program(true), 64, 17).expect("retrying prefix executes");

        assert_eq!(plain.branch_selectors.len(), 64);
        assert_eq!(retried.discarded, 0);
        assert_eq!(retried.branch_selectors, plain.branch_selectors);
    }

    /// A boundary operator's coordinates are already instance-global, so
    /// [`Program::measure_boundary`] must resolve them without re-adding the
    /// instance offset. This pins the frame at a non-zero offset, where a stray
    /// second offset (the historical double-offset) would resolve coordinate
    /// `(48, 0)` — absent from `coord_index`, surfacing as `UnknownCoord` — while
    /// the correct global coordinate `(24, 0)` reads the prepared qubit.
    #[test]
    fn measure_boundary_does_not_double_apply_offset() {
        let coord = IVec2::new(24, 0);
        let instance = TemplateInstanceId(0);

        let mut bloq = Bloq::new();
        let template_id = one_qubit_template(&mut bloq, GateType::X);
        bloq.add_node(quantum_node(template_id, instance.0, coord));

        // A single-qubit Z face on the global data qubit, in instance-global
        // coordinates exactly as lowering stores it.
        let operator = InstanceBoundaryOperator {
            instance,
            face: BoundaryFace::Output,
            operator: [(coord, CircuitPauli::Z)].into_iter().collect(),
        };
        bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![operator.clone()],
        )));
        let program = Program::prepare(&bloq, RunMode::Plain).unwrap();

        // Prepare |1> so Z reads the −1 eigenvalue (outcome `true`) deterministically.
        let mut sim = Simulator::with_seed(1, 0);
        sim.x(0);
        let outcome = program
            .measure_boundary(&operator, &mut sim)
            .expect("boundary resolves at the true global coordinate (24, 0)");
        assert!(outcome, "Z on |1> is the −1 eigenvalue");
    }

    fn nested_boundary_program(depth: usize) -> Bloq {
        let coord = IVec2::ZERO;
        let instance = TemplateInstanceId(0);
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::X);
        let quantum = bloq.add_node(quantum_node(template, instance.0, IVec2::ZERO));
        let operator = InstanceBoundaryOperator {
            instance,
            face: BoundaryFace::Output,
            operator: [(coord, CircuitPauli::Z)].into_iter().collect(),
        };
        let mut body = SubGraph::new();
        body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![operator],
        )));
        let internal = body.add_node(body[BloqNodeId(0)].clone());
        let local = body.add_node(BloqNode::classical(ClassicalNode::observable(1)));
        body.add_edge(BloqNodeId(0), local, BloqEdge::value(0));
        body.add_edge(internal, local, BloqEdge::value(1));
        body.set_boundary_outputs(vec![BloqNodeId(0)]);
        let mut region = RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        };
        for _ in 1..depth {
            let mut body = SubGraph::new();
            let nested = body.add_node(BloqNode::region(region));
            body.set_boundary_outputs(vec![nested]);
            let consume = body.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }));
            body.add_edge(nested, consume, BloqEdge::value(0));
            region = RegionNode::RepeatUntilSuccess {
                restart_condition: ClassicalExpr::Const(false),
                restart_source: None,
                body,
            };
        }
        let region = bloq.add_node(BloqNode::region(region));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(quantum, region, BloqEdge::Order);
        bloq.add_edge(region, observable, BloqEdge::compose(0));
        bloq
    }

    #[test]
    fn nested_rus_include_bindings_reach_observable() {
        for depth in [1, 2, 3] {
            let mut bloq = nested_boundary_program(depth);
            bloq.optimize().expect("acyclic test program");
            let report = run_bloq(&bloq, 1, 0).expect("nested boundary program executes");
            assert_eq!(
                report.observables[0].per_shot,
                [true],
                "the selected body's Z binding reads the prepared |1>"
            );
        }
    }

    #[test]
    fn early_output_reads_preserve_joint_bell_observables() {
        for reuse in [false, true] {
            let mut bloq = Bloq::new();
            let mut circuit = CoordCircuit::new();
            circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
            circuit
                .do_gate(GateType::CX, [IVec2::ZERO, IVec2::X])
                .unwrap();
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let owner = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
            for (index, basis) in [CircuitPauli::X, CircuitPauli::Z].into_iter().enumerate() {
                let operators = [IVec2::ZERO, IVec2::X]
                    .into_iter()
                    .map(|coord| InstanceBoundaryOperator {
                        instance: TemplateInstanceId(0),
                        face: BoundaryFace::Output,
                        operator: [(coord, basis)].into_iter().collect(),
                    })
                    .collect();
                let include = bloq.add_node(BloqNode::classical(
                    ClassicalNode::observable_fragment(Vec::new(), operators),
                ));
                let observable =
                    bloq.add_node(BloqNode::classical(ClassicalNode::observable(index as u32)));
                bloq.add_edge(owner, include, BloqEdge::Order);
                bloq.add_edge(include, observable, BloqEdge::compose(0));
            }
            if reuse {
                let template = one_qubit_template(&mut bloq, GateType::RZ);
                let reset = bloq.add_node(quantum_node(template, 1, IVec2::ZERO));
                bloq.add_edge(owner, reset, BloqEdge::Order);
            }
            let report = run_bloq(&bloq, 64, 0).unwrap();
            for observable in report.observables {
                assert_eq!(
                    observable.per_shot, [false; 64],
                    "both Bell stabilizers remain +1 (reuse={reuse})"
                );
            }
        }
    }

    #[test]
    fn repeated_output_captures_use_owner_cuts_without_observables() {
        let mut bloq = Bloq::new();
        let mut previous = None;
        let mut outputs = Vec::new();
        for z in 0..3 {
            let mut circuit = CoordCircuit::new();
            circuit.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
            circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let instance = TemplateInstanceId(z as u32);
            let owner = bloq.add_node(quantum_node(template, instance.0, IVec2::ZERO));
            if let Some(previous) = previous {
                bloq.add_edge(previous, owner, BloqEdge::Order);
            }
            previous = Some(owner);
            let port = IVec3::new(0, 0, z);
            outputs.push(bloq_ir::LogicalOutput {
                port,
                instance,
                x: [(IVec2::ZERO, CircuitPauli::X)].into_iter().collect(),
                z: [(IVec2::ZERO, CircuitPauli::Z)].into_iter().collect(),
            });
            for basis in [bloq_ir::Basis::X, bloq_ir::Basis::Z] {
                bloq.add_node(
                    BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(false),
                    })
                    .with_provenance(bloq_ir::NodeProvenance::OutputFrame { port, basis }),
                );
            }
        }
        bloq.set_logical_outputs(outputs);
        // Saved IR must retain the identity needed for each output's cut.
        let bloq = Bloq::from_binary(&bloq.to_binary()).unwrap();
        let bloq = Bloq::from_text(&bloq.to_text()).unwrap();
        run_bloq_with_captured_outputs(&bloq, 1, 0, |sim, context| {
            assert_eq!(context.saved_outputs.len(), 2);
            for saved in context.saved_outputs {
                let z = PauliString::single(sim.num_qubits(), saved.qubit, Pauli::Z);
                assert_eq!(
                    sim.peek_observable_expectation(&z)?,
                    -1.0,
                    "output {} was prepared in |1>",
                    saved.port.z
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn spatial_input_replacement_keeps_wall_checks_on_their_own_side_of_the_cut() {
        use bloq_compile::{CompileConfig, CompileContext};
        use bloq_graph::BlockGraph;

        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0
            0: Port [0,1,0] role=input
            1: ZXX [0,0,0]
            2: Port [0,0,1] role=output
            1 -H> +Y
            1 -> +Z",
        )
        .unwrap();
        let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .compile(&graph)
            .unwrap()
            .bloq;
        let prepared = Program::prepare(&bloq, RunMode::PrepareInputs).unwrap();
        let input = &prepared.inputs[0];
        let records = prepared
            .core
            .global_meas
            .iter()
            .filter_map(|(measurement, &record)| {
                (measurement.instance == input.input_instance).then_some(record)
            })
            .collect::<Vec<_>>();
        let mut histories = vec![Vec::new(); records.len()];
        let report = run_bloq_with_io(
            &bloq,
            16,
            17,
            |sim, context| {
                // The virtual Port first encounters a random old gauge. Ideal
                // input replacement establishes a fresh gauge before the wall.
                for &qubit in &input.data_qubits {
                    sim.h(qubit);
                }
                sim.h(context.inputs[0].qubit);
                Ok(())
            },
            |sim, context| {
                for (&record, history) in records.iter().zip(&mut histories) {
                    history.push(context.record.get(record).unwrap());
                }
                let output = &context.outputs[0];
                let frame = &context.frames[0];
                let bloch = crate::verify::logical_bloch(
                    sim,
                    output,
                    (frame.x.unwrap(), frame.z.unwrap()),
                )?;
                assert!((bloch.0 - 1.0).abs() < 1e-8);
                Ok(())
            },
        )
        .unwrap();
        assert!(
            histories
                .iter()
                .any(|history| history.contains(&false) && history.contains(&true))
        );
        assert!(!report.detectors.is_empty());
        assert!(report.all_detectors_constant());
    }

    #[test]
    fn report_requires_a_complete_frame_pair_for_each_logical_output() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::RZ);
        bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
        let port = IVec3::ZERO;
        bloq.set_logical_outputs(vec![bloq_ir::LogicalOutput {
            port,
            instance: TemplateInstanceId(0),
            x: [(IVec2::ZERO, CircuitPauli::X)].into_iter().collect(),
            z: [(IVec2::ZERO, CircuitPauli::Z)].into_iter().collect(),
        }]);
        bloq.add_node(
            BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })
            .with_provenance(bloq_ir::NodeProvenance::OutputFrame {
                port,
                basis: bloq_ir::Basis::X,
            }),
        );
        assert!(matches!(
            run_bloq(&bloq, 1, 0),
            Err(ExecError::MalformedGraph(
                "logical output has no complete frame pair"
            ))
        ));
    }

    #[test]
    fn output_capture_requires_a_selected_owner() {
        for take_other_arm in [false, true] {
            let mut bloq = Bloq::new();
            let prepared = one_qubit_template(&mut bloq, GateType::X);
            let reset = one_qubit_template(&mut bloq, GateType::RZ);
            let consumer = bloq.add_node(quantum_node(reset, 2, IVec2::ZERO));
            for (instance, template, selected) in
                [(0, prepared, !take_other_arm), (1, reset, take_other_arm)]
            {
                let selector = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(selected),
                }));
                let owner = bloq.add_node(guarded_node(
                    quantum_node(template, instance, IVec2::ZERO),
                    0,
                ));
                bloq.add_edge(selector, owner, BloqEdge::value(0));
                bloq.add_edge(owner, consumer, BloqEdge::Order);
            }
            let port = IVec3::ZERO;
            bloq.set_logical_outputs(vec![bloq_ir::LogicalOutput {
                port,
                instance: TemplateInstanceId(u32::from(take_other_arm)),
                x: [(IVec2::ZERO, CircuitPauli::X)].into_iter().collect(),
                z: [(IVec2::ZERO, CircuitPauli::Z)].into_iter().collect(),
            }]);
            for basis in [bloq_ir::Basis::X, bloq_ir::Basis::Z] {
                bloq.add_node(
                    BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(false),
                    })
                    .with_provenance(bloq_ir::NodeProvenance::OutputFrame { port, basis }),
                );
            }
            run_bloq_with_captured_outputs(&bloq, 1, 0, |sim, context| {
                assert_eq!(context.saved_outputs.len(), 1);
                let saved = &context.saved_outputs[0];
                let z = PauliString::single(sim.num_qubits(), saved.qubit, Pauli::Z);
                assert_eq!(
                    sim.peek_observable_expectation(&z)?,
                    if take_other_arm { 1.0 } else { -1.0 },
                );
                Ok(())
            })
            .unwrap();

            // WF-7: external output metadata cannot advertise an absent owner,
            // even when another branch uses the same physical coordinates.
            let mut outputs = bloq.logical_outputs().to_vec();
            outputs[0].instance = TemplateInstanceId(u32::from(!take_other_arm));
            bloq.set_logical_outputs(outputs);
            assert!(matches!(
                bloq.validate(),
                Err(
                    bloq_ir::BloqValidationError::InvalidInstanceMergeStructure {
                        source:
                            bloq_ir::lowering::NodeTemplateInstanceMergeError::InvalidMembership(_),
                        ..
                    }
                )
            ));
            assert!(matches!(
                run_bloq_with_captured_outputs(&bloq, 1, 0, |_, _| unreachable!()),
                Err(ExecError::MalformedGraph(
                    "logical output owner is not selected"
                ))
            ));
        }
    }

    #[test]
    fn early_output_reads_keep_identical_activated_bindings_distinct() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::X);
        let owner = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        let template = one_qubit_template(&mut bloq, GateType::RZ);
        let reset = bloq.add_node(quantum_node(template, 1, IVec2::ZERO));
        for (slot, selected) in [false, true].into_iter().enumerate() {
            let selector = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(selected),
            }));
            let mut fragment = BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                vec![InstanceBoundaryOperator {
                    instance: TemplateInstanceId(0),
                    face: BoundaryFace::Output,
                    operator: [(IVec2::ZERO, CircuitPauli::Z)].into_iter().collect(),
                }],
            ));
            fragment.activation = Some(0);
            let region = bloq.add_node(fragment);
            bloq.add_edge(owner, region, BloqEdge::Order);
            bloq.add_edge(selector, region, BloqEdge::value(0));
            bloq.add_edge(region, reset, BloqEdge::Order);
            bloq.add_edge(region, observable, BloqEdge::compose(slot as u32));
        }
        bloq.add_edge(reset, observable, BloqEdge::Order);
        let report = run_bloq(&bloq, 8, 0).unwrap();
        assert_eq!(report.observables[0].per_shot, [true; 8]);
    }

    #[test]
    fn anticommuting_partial_output_reads_fail_before_projection() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
        let template = one_qubit_template(&mut bloq, GateType::RZ);
        let reset = bloq.add_node(quantum_node(template, 1, IVec2::ZERO));
        let future = bloq.add_node(quantum_node(template, 2, IVec2::X));
        bloq.add_edge(owner, reset, BloqEdge::Order);
        bloq.add_edge(reset, future, BloqEdge::Order);
        for (index, basis) in [CircuitPauli::X, CircuitPauli::Z].into_iter().enumerate() {
            let include = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                [(0, IVec2::ZERO), (2, IVec2::X)]
                    .into_iter()
                    .map(|(instance, coord)| InstanceBoundaryOperator {
                        instance: TemplateInstanceId(instance),
                        face: BoundaryFace::Output,
                        operator: [(coord, basis)].into_iter().collect(),
                    })
                    .collect(),
            )));
            let observable =
                bloq.add_node(BloqNode::classical(ClassicalNode::observable(index as u32)));
            bloq.add_edge(future, include, BloqEdge::Order);
            bloq.add_edge(include, observable, BloqEdge::compose(0));
        }
        for result in [
            run_bloq(&bloq, 1, 0),
            run_bloq_with_captured_outputs(&bloq, 1, 0, |_, _| Ok(())),
        ] {
            assert!(matches!(
                result,
                Err(ExecError::IncompatibleEarlyOutputReads { .. })
            ));
        }
    }

    #[test]
    fn joint_boundary_products_keep_their_pauli_sign() {
        for reuse in [false, true] {
            for split_includes in [false, true] {
                let mut bloq = Bloq::new();
                let mut circuit = CoordCircuit::new();
                circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
                circuit
                    .do_gate(GateType::CX, [IVec2::ZERO, IVec2::X])
                    .unwrap();
                let template = bloq.add_template(BloqTemplate::new(circuit));
                let owner = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
                let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
                let operators =
                    [CircuitPauli::X, CircuitPauli::Z].map(|basis| InstanceBoundaryOperator {
                        instance: TemplateInstanceId(0),
                        face: BoundaryFace::Output,
                        operator: [(IVec2::ZERO, basis), (IVec2::X, basis)]
                            .into_iter()
                            .collect(),
                    });
                let includes = if split_includes {
                    operators
                        .into_iter()
                        .map(|operator| vec![operator])
                        .collect::<Vec<_>>()
                } else {
                    vec![operators.into_iter().collect()]
                };
                for (slot, operators) in includes.into_iter().enumerate() {
                    let include = bloq.add_node(BloqNode::classical(
                        ClassicalNode::observable_fragment(Vec::new(), operators),
                    ));
                    bloq.add_edge(owner, include, BloqEdge::Order);
                    bloq.add_edge(include, observable, BloqEdge::compose(slot as u32));
                }
                if reuse {
                    let template = one_qubit_template(&mut bloq, GateType::RZ);
                    let reset = bloq.add_node(quantum_node(template, 1, IVec2::ZERO));
                    bloq.add_edge(owner, reset, BloqEdge::Order);
                    bloq.add_edge(reset, observable, BloqEdge::Order);
                }
                let report = run_bloq(&bloq, 8, 0).unwrap();
                assert_eq!(
                    report.observables[0].per_shot, [false; 8],
                    "Bell XX * ZZ = -YY has eigenvalue +1 (reuse={reuse}, split={split_includes})"
                );
            }
        }
    }
}
