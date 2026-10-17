//! Dynamic execution of lowered VM instruction streams.

mod causal;

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::backend::{Gate1Q, Pauli as BackendPauli, PauliBasis, PauliString, SimError, Simulator};
use crate::decoder::{DecoderDecision, DecoderError, MockDecoderConfig};
use crate::instruction::{
    BitId, BoolOp, Clifford1, Instruction, LogicalOutput, MemoryCycle, Pauli, PauliProduct,
    Program, QuantumOp, QuantumStream, RecordId, RecordParity, ResourceId, SourceRole, Stream,
    TaskId, TaskOrigin,
};

pub(crate) const TIME_EPSILON: f64 = 1e-9;

/// Hard bounds for one dynamic run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLimits {
    /// Maximum task, moment, and repeated-memory executions.
    pub max_steps: u64,
    /// Maximum attempts of one repeat-until-success task.
    pub max_attempts: u32,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            max_attempts: 100,
        }
    }
}

/// Runtime and mock-hardware settings.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Seed shared by the simulator and mock decoder through separate streams.
    pub seed: u64,
    /// State injected at every external logical input arrival.
    pub input_state: LogicalInputState,
    /// Mock streaming-decoder assumptions.
    pub decoder: MockDecoderConfig,
    /// Idle depolarization rate per runtime time unit; an interval uses
    /// `min(duration * rate, 0.75)`.
    pub idle_error_rate: f64,
    /// Execution caps.
    pub limits: RuntimeLimits,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            input_state: LogicalInputState::Plus,
            decoder: MockDecoderConfig::default(),
            idle_error_rate: 0.0,
            limits: RuntimeLimits::default(),
        }
    }
}

/// Encoded state injected when a logical input source is released.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalInputState {
    /// Logical `|+>`.
    #[default]
    Plus,
    /// Logical `|0>`.
    Zero,
}

/// Final engine state plus the complete execution trace.
#[derive(Debug)]
pub struct RunResult {
    /// Final quantum state, available to physical-verification hooks.
    pub simulator: Simulator,
    /// JSON-ready run trace.
    pub artifact: ExecutionArtifact,
}

impl RunResult {
    /// Read a signed VM Pauli expectation from the untouched terminal state.
    ///
    /// # Errors
    ///
    /// Returns a fixed-register or backend error.
    pub fn expectation(&self, operator: &PauliProduct) -> Result<f64, RuntimeError> {
        let operator = engine_pauli_product(operator, self.simulator.num_qubits())?;
        Ok(self.simulator.peek_observable_expectation(&operator)?)
    }

    /// Read the terminal `(X, Y, Z)` Bloch vector with VM frame corrections.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::MissingBit`] for an unavailable frame or a
    /// fixed-register/backend error for malformed logical operators.
    pub fn logical_bloch(&self, output: &LogicalOutput) -> Result<(f64, f64, f64), RuntimeError> {
        let frame_x = self
            .artifact
            .final_bits
            .get(output.frame_x as usize)
            .copied()
            .flatten()
            .ok_or(RuntimeError::MissingBit(output.frame_x))?;
        let frame_z = self
            .artifact
            .final_bits
            .get(output.frame_z as usize)
            .copied()
            .flatten()
            .ok_or(RuntimeError::MissingBit(output.frame_z))?;
        let width = self.simulator.num_qubits();
        let mut logical_x = engine_pauli_product(&output.x, width)?;
        let mut logical_z = engine_pauli_product(&output.z, width)?;
        logical_x.phase_shift(2 * i32::from(frame_z));
        logical_z.phase_shift(2 * i32::from(frame_x));
        let mut logical_y = &logical_x * &logical_z;
        logical_y.phase_shift(1);
        Ok((
            self.simulator.peek_observable_expectation(&logical_x)?,
            self.simulator.peek_observable_expectation(&logical_y)?,
            self.simulator.peek_observable_expectation(&logical_z)?,
        ))
    }
}

/// Inspectable output of one dynamic run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionArtifact {
    /// Reproduction settings and dense-task source map.
    pub metadata: RunMetadata,
    /// Every raw measurement, including rejected attempts.
    pub measurements: Vec<MeasurementRecord>,
    /// Signed detector outcomes.
    pub detectors: Vec<DetectorResult>,
    /// Named raw and physical observable outcomes.
    pub observables: Vec<ObservableResult>,
    /// Actual task intervals.
    pub timing: Vec<TaskTiming>,
    /// RUS attempt history.
    pub retries: Vec<RetryRecord>,
    /// Mock decoder outcomes.
    pub decoder_decisions: Vec<DecoderDecisionRecord>,
    /// Dynamic waits, corrections, selections, and state transitions.
    pub events: Vec<ExecutionEvent>,
    /// Whether this shot terminated without a committed result.
    pub discarded: bool,
    /// Terminal reason for a discarded shot.
    pub stop_reason: Option<StopReason>,
    /// Physical stop time.
    pub finished_at: f64,
    /// Highest stabilizer-frame rank reached by the selected engine.
    pub peak_rank: usize,
    /// Final dense classical registers; all unavailable on a discarded shot.
    pub final_bits: Vec<Option<bool>>,
}

/// Settings and source labels needed to reproduce and read an artifact.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Selected backend name.
    pub backend: String,
    /// Root random seed.
    pub seed: u64,
    /// External logical input state.
    pub input_state: LogicalInputState,
    /// Mock decoder assumptions.
    pub decoder: MockDecoderConfig,
    /// Mock-decoder latency used by this program, in physical memory rounds.
    pub decoder_latency_rounds: u32,
    /// Dynamic-idle depolarization rate per runtime time unit.
    pub idle_error_rate: f64,
    /// Runtime work and retry caps.
    pub limits: RuntimeLimits,
    /// Dense-task source map.
    pub tasks: Vec<TaskMetadata>,
}

/// Human-readable metadata for one dense VM task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Dense task id.
    pub task: TaskId,
    /// Lowering-provided IR scope, node, and role.
    pub label: String,
    /// VM instruction kind.
    pub instruction: String,
    /// Backend-independent compiled-IR function and source geometry.
    pub origin: Option<TaskOrigin>,
    /// Dense causal predecessors used by the runtime scheduler.
    pub dependencies: Vec<TaskId>,
    /// Tasks whose completion releases this held patch.
    pub wait_until: Vec<TaskId>,
    /// Dense classical result register, when produced.
    pub output: Option<BitId>,
}

impl ExecutionArtifact {
    /// Render this artifact as compact JSON.
    ///
    /// # Errors
    ///
    /// Propagates JSON serialization errors.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Render this artifact for inspection or storage.
    ///
    /// # Errors
    ///
    /// Propagates JSON serialization errors.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// One raw record write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementRecord {
    /// Dense record id.
    pub record: RecordId,
    /// `true` is the `-1` outcome.
    pub value: bool,
    /// Task that measured it.
    pub task: TaskId,
    /// RUS attempt epoch, if any.
    pub attempt: Option<u64>,
    /// Measurement completion time.
    pub time: f64,
    /// Whether this record belongs to the accepted execution path.
    pub committed: bool,
}

/// One signed detector evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorResult {
    /// Stable order within this artifact.
    pub sequence: u32,
    /// Owning task.
    pub task: TaskId,
    /// Fault-syndrome value, including the authored sign.
    pub value: bool,
    /// RUS attempt epoch, if any.
    pub attempt: Option<u64>,
    /// Evaluation time.
    pub time: f64,
    /// Whether this detector belongs to the accepted execution path.
    pub committed: bool,
}

/// One named observable evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservableResult {
    /// Authored observable index.
    pub index: u32,
    /// Owning task.
    pub task: TaskId,
    /// SEM-OBS dataflow parity, excluding boundary bindings.
    pub raw: Option<bool>,
    /// Physical report value after one signed joint boundary measurement.
    /// Full physical report value, including deferred boundary bindings.
    pub value: Option<bool>,
    /// RUS attempt epoch, if any.
    pub attempt: Option<u64>,
    /// Observable task time.
    pub time: f64,
    /// Whether this observable belongs to the accepted execution path.
    pub committed: bool,
}

/// One cached streaming-decoder solve in its task context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderDecisionRecord {
    /// First task that requested the solve.
    pub task: TaskId,
    /// Causal decoder decision.
    pub decision: DecoderDecision,
    /// Whether this solve belongs to the accepted execution path.
    pub committed: bool,
}

/// Actual interval occupied by one task execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskTiming {
    /// Dense task id.
    pub task: TaskId,
    /// Lowering-provided source label.
    pub label: String,
    /// VM instruction kind.
    pub instruction: String,
    /// RUS attempt epoch, if any.
    pub attempt: Option<u64>,
    /// Start time.
    pub start: f64,
    /// Finish time.
    pub end: f64,
}

/// One complete RUS attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryRecord {
    /// Owning RUS task.
    pub task: TaskId,
    /// Zero-based local attempt number.
    pub attempt: u32,
    /// Globally unique attempt epoch.
    pub epoch: u64,
    /// Attempt start time.
    pub start: f64,
    /// Attempt finish time, including decoder latency.
    pub end: f64,
    /// Whether this attempt committed its state and aliases.
    pub accepted: bool,
}

/// Why a shot ended without a committed result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// An authored discard predicate fired.
    Discard,
    /// A RUS task exhausted its configured attempt cap.
    AttemptLimit,
}

/// Origin of a recorded syndrome-memory round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Authored memory-round padding.
    StaticPadding,
    /// Runtime padding while a join sibling remained active.
    DynamicWait,
    /// Mandatory factory GAP rounds after physical cultivation completes.
    FactoryGap,
}

/// Runtime cause for a dynamically inserted hold interval.
///
/// When blockers overlap, classification uses this conservative precedence:
/// a direct physical sibling or resource is [`Self::Synchronization`], then an
/// unfinished causal query is [`Self::CausalCut`], and only an issued decoder
/// solve is [`Self::DecoderLatency`]. The reason is sampled when the whole
/// interval is issued and does not change partway through a QEC round. Factory
/// GAP rounds remain identified by [`MemoryKind::FactoryGap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitReason {
    /// A directly joined physical sibling or factory resource is unfinished.
    Synchronization,
    /// Measurements or other causal inputs needed to issue a query are unfinished.
    CausalCut,
    /// Query inputs are complete and an issued decoder solve is still running.
    DecoderLatency,
}

/// Time-stamped dynamic events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionEvent {
    /// An independently released source became eligible to execute.
    SourceReleased {
        /// Source task.
        task: TaskId,
        /// Lowered source role.
        role: String,
        /// Absolute release time.
        time: f64,
    },
    /// An external encoded input arrived.
    InputArrival {
        /// Owning source task.
        task: TaskId,
        /// Logical input port.
        port: [i32; 3],
        /// Injected logical state.
        state: LogicalInputState,
        /// Arrival time.
        time: f64,
    },
    /// A physical moment was issued onto free qubits.
    MomentIssued {
        /// Owning task.
        task: TaskId,
        /// RUS attempt epoch, if any.
        attempt: Option<u64>,
        /// Zero-based moment index within its physical stream.
        moment: u32,
        /// Issue time.
        start: f64,
        /// Scheduled completion time.
        end: f64,
    },
    /// A physical moment completed and mutated the simulator.
    MomentCompleted {
        /// Owning task.
        task: TaskId,
        /// RUS attempt epoch, if any.
        attempt: Option<u64>,
        /// Zero-based moment index.
        moment: u32,
        /// Completion time.
        time: f64,
    },
    /// A newly available selected restart parity aborted cultivation early.
    EarlyRestart {
        /// Owning RUS task.
        task: TaskId,
        /// Zero-based local attempt.
        attempt: u32,
        /// Inspection reason naming the selected check.
        reason: String,
        /// First time the rejecting parity was fully available.
        time: f64,
    },
    /// A causal decoder deadline was established.
    DecoderDeadline {
        /// Decoder output task.
        task: TaskId,
        /// Causal observable task.
        observable: TaskId,
        /// Whether factory completion, rather than a measurement cut, anchors latency.
        factory: bool,
        /// Last contributing measurement time when present.
        measurements_ready_at: Option<f64>,
        /// Time at which the runtime issued this decoder query.
        requested_at: f64,
        /// Decision availability time.
        ready_at: f64,
    },
    /// Every incoming seam reached a join.
    JoinReleased {
        /// Wait task representing the held seam.
        task: TaskId,
        /// Release time after any in-flight round completed.
        time: f64,
    },
    /// One guarded quantum alternative was selected.
    QuantumAlternative {
        /// Quantum task.
        task: TaskId,
        /// Zero-based selected alternative.
        alternative: u32,
        /// Human-readable selected gate names, including dynamic S corrections.
        operations: Vec<String>,
        /// Selection time.
        time: f64,
    },
    /// A record-controlled Pauli was considered.
    ConditionalCorrection {
        /// Quantum task.
        task: TaskId,
        /// Gating record.
        record: RecordId,
        /// Pauli axis.
        pauli: String,
        /// Whether the correction fired.
        applied: bool,
        /// Correction time.
        time: f64,
    },
    /// A source-isolated RUS attempt began.
    RusAttemptStarted {
        /// Owning RUS task.
        task: TaskId,
        /// Zero-based local attempt number.
        attempt: u32,
        /// Globally unique attempt epoch.
        epoch: u64,
        /// Start time.
        time: f64,
    },
    /// Explicit timed preparation before a retried attempt.
    RetryPrepare {
        /// Owning RUS task.
        task: TaskId,
        /// Zero-based local attempt number.
        attempt: u32,
        /// Preparation start.
        start: f64,
        /// Preparation finish.
        end: f64,
    },
    /// A RUS attempt rejected without committing aliases.
    RusAttemptRejected {
        /// Owning RUS task.
        task: TaskId,
        /// Zero-based local attempt number.
        attempt: u32,
        /// Globally unique attempt epoch.
        epoch: u64,
        /// Rejection time.
        time: f64,
    },
    /// A RUS attempt accepted and committed.
    RusAttemptAccepted {
        /// Owning RUS task.
        task: TaskId,
        /// Zero-based local attempt number.
        attempt: u32,
        /// Globally unique attempt epoch.
        epoch: u64,
        /// Acceptance time.
        time: f64,
    },
    /// An accepted factory output became available.
    FactoryReady {
        /// Runtime resource id.
        resource: ResourceId,
        /// Ready time.
        time: f64,
    },
    /// One complete syndrome-memory round ran.
    MemoryRound {
        /// Waiting or padding task.
        task: TaskId,
        /// Zero-based round within the task.
        round: u32,
        /// Static or dynamic origin.
        memory_kind: MemoryKind,
        /// Dynamic hold cause sampled when the round was issued.
        /// Authored static padding and factory GAP use `memory_kind` instead.
        wait_reason: Option<WaitReason>,
        /// Round start.
        start: f64,
        /// Round finish.
        end: f64,
    },
    /// A patch performed an authored idle or a residual dynamic wait.
    Idle {
        /// Idling task.
        task: TaskId,
        /// Dynamic hold cause sampled when the idle was issued.
        /// Authored idle instructions have no wait reason.
        wait_reason: Option<WaitReason>,
        /// Idle start.
        start: f64,
        /// Idle finish.
        end: f64,
    },
    /// The shot stopped and no result committed.
    Discarded {
        /// Task that stopped the shot.
        task: TaskId,
        /// Stop reason.
        reason: StopReason,
        /// Stop time.
        time: f64,
    },
}

/// Failures on the *dynamic causal runtime* path.
///
/// That path is [`run()`] and the timing, retry, noise, and mock-decoding
/// machinery it drives. The noiseless physical verifier is the other execution
/// engine in this crate and reports [`ExecError`](crate::ExecError) instead.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// A dense runtime or artifact identifier exhausted `u32`.
    #[error("{0} identifier space exhausted")]
    IdOverflow(&'static str),
    /// A runtime table could not fit or allocate its initial capacity.
    #[error("cannot allocate {register} runtime table: {source}")]
    RegisterAllocation {
        /// Runtime table being initialized.
        register: &'static str,
        /// Capacity or allocation failure from the table.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// A public VM program or runtime setting violated its fixed-register contract.
    #[error("invalid VM program: {0}")]
    InvalidProgram(&'static str),
    /// A task id or dependency was absent.
    #[error("task {0} is unavailable at its scheduled use")]
    MissingTask(TaskId),
    /// A required classical bit was absent.
    #[error("classical bit {0} is unavailable")]
    MissingBit(BitId),
    /// A required record was absent.
    #[error("measurement record {0} is unavailable")]
    MissingRecord(RecordId),
    /// The run exhausted its operation-work cap.
    #[error("runtime step cap {limit} exceeded")]
    StepLimit {
        /// Configured cap.
        limit: u64,
    },
    /// No guarded quantum alternative matched available input bits.
    #[error("quantum task has no selected alternative")]
    NoQuantumAlternative,
    /// The selected simulator rejected an operation.
    #[error("quantum backend error: {0}")]
    Backend(#[from] SimError),
    /// Mock decoder configuration or timing was invalid.
    #[error("mock decoder error: {0}")]
    Decoder(#[from] DecoderError),
    /// Finite instruction durations overflowed the runtime clock.
    #[error("runtime clock overflowed")]
    InvalidTiming,
}

/// Execute one lowered program from time zero.
///
/// # Errors
///
/// Returns a typed program, resource-limit, decoder, or backend failure.
pub fn run(program: &Program, config: RuntimeConfig) -> Result<RunResult, RuntimeError> {
    causal::run(program, config)
}

fn validate_program(program: &Program) -> Result<(), RuntimeError> {
    let tasks = program.tasks.len();
    if let Some(last_task) = tasks.checked_sub(1) {
        u32::try_from(last_task).map_err(|_| RuntimeError::IdOverflow("task"))?;
    }
    let bits = program.bit_count as usize;
    let records = program.record_count as usize;
    let qubits = program.qubit_count as usize;
    if !program.task_origins.is_empty() && program.task_origins.len() != tasks {
        return Err(RuntimeError::InvalidProgram("invalid task origin table"));
    }
    validate_stream(&program.entry, tasks, bits)?;
    let mut input_tasks = HashSet::new();
    for input in &program.inputs {
        let mut data_qubits = HashSet::new();
        if input.task as usize >= tasks
            || program.tasks[input.task as usize].source != Some(SourceRole::LogicalInput)
            || !input_tasks.insert(input.task)
            || input
                .data_qubits
                .iter()
                .any(|&qubit| qubit as usize >= qubits || !data_qubits.insert(qubit))
        {
            return Err(RuntimeError::InvalidProgram("invalid logical input"));
        }
        for stabilizer in &input.stabilizers {
            validate_product(stabilizer, qubits)?;
        }
        validate_product(&input.x, qubits)?;
        validate_product(&input.z, qubits)?;
    }
    for output in &program.outputs {
        if output.frame_x as usize >= bits || output.frame_z as usize >= bits {
            return Err(RuntimeError::InvalidProgram("invalid output frame"));
        }
        validate_product(&output.x, qubits)?;
        validate_product(&output.z, qubits)?;
    }
    for task in &program.tasks {
        if !valid_duration(task.release)
            || !valid_duration(task.duration)
            || task.dependencies.iter().any(|&id| id as usize >= tasks)
            || task.activation.is_some_and(|id| id as usize >= bits)
            || task.output.is_some_and(|id| id as usize >= bits)
            || task.qubits.iter().any(|&id| id as usize >= qubits)
        {
            return Err(RuntimeError::InvalidProgram("invalid task operand"));
        }
        validate_instruction(&task.instruction, tasks, bits, records, qubits)?;
    }
    for task in &program.tasks {
        if let Instruction::Rus { .. } = &task.instruction {
            validate_rus_isolation(program, task)?;
        }
    }
    Ok(())
}

fn validate_rus_isolation(
    program: &Program,
    task: &crate::instruction::Task,
) -> Result<(), RuntimeError> {
    let Instruction::Rus {
        body,
        owned_qubits,
        retry_prepare,
        decoder_hold,
        cultivation_exits,
        ..
    } = &task.instruction
    else {
        return Ok(());
    };
    let owned = owned_qubits.iter().copied().collect::<HashSet<_>>();
    let outside_operation =
        |operation: &QuantumOp| operation.qubits().any(|qubit| !owned.contains(&qubit));
    let outside_stream = |stream: &QuantumStream| {
        stream
            .moments
            .iter()
            .flat_map(|moment| &moment.operations)
            .any(outside_operation)
    };
    let outside_cycle = |cycle: &MemoryCycle| {
        outside_stream(&cycle.stream)
            || cycle
                .boundary_flows
                .iter()
                .flat_map(|flow| flow.start.terms.iter().chain(flow.end.terms.iter()))
                .any(|(qubit, _)| !owned.contains(qubit))
    };
    if owned.len() != owned_qubits.len()
        || decoder_hold.is_empty() != cultivation_exits.is_empty()
        || (task.source == Some(SourceRole::Factory) && decoder_hold.is_empty())
    {
        return Err(RuntimeError::InvalidProgram("invalid RUS isolation"));
    }

    if outside_stream(retry_prepare) {
        return Err(RuntimeError::InvalidProgram(
            "RUS retry preparation touches unowned support",
        ));
    }

    let mut members = HashSet::new();
    for &member in &body.tasks {
        if !members.insert(member) {
            continue;
        }
        let member_task = &program.tasks[member as usize];
        let outside_instruction = match &member_task.instruction {
            Instruction::Quantum(quantum) => quantum
                .alternatives
                .iter()
                .any(|alternative| outside_stream(&alternative.stream)),
            Instruction::MemoryRounds { stream, .. } => outside_stream(stream),
            Instruction::Idle { operations, .. } => operations.iter().any(outside_operation),
            Instruction::WaitFor { memory, .. } => memory.iter().any(outside_cycle),
            Instruction::Bind(bindings) => bindings
                .iter()
                .flat_map(|binding| &binding.operator.terms)
                .any(|(qubit, _)| !owned.contains(qubit)),
            Instruction::Rus { .. } => {
                return Err(RuntimeError::InvalidProgram(
                    "nested dynamic RUS is unsupported",
                ));
            }
            Instruction::Eval(_)
            | Instruction::Accumulate(_)
            | Instruction::ReadoutRecipe { .. }
            | Instruction::Observable { .. }
            | Instruction::Decode(_)
            | Instruction::Discard(_)
            | Instruction::SignalReady { .. } => false,
        };
        if member_task
            .qubits
            .iter()
            .any(|qubit| !owned.contains(qubit))
            || outside_instruction
        {
            return Err(RuntimeError::InvalidProgram(
                "RUS body touches qubits outside its isolated source support",
            ));
        }
    }
    if cultivation_exits.iter().any(|exit| !members.contains(exit)) {
        return Err(RuntimeError::InvalidProgram(
            "RUS cultivation exit is outside its body",
        ));
    }
    if decoder_hold.iter().any(outside_cycle) {
        return Err(RuntimeError::InvalidProgram(
            "RUS decoder hold has invalid data support",
        ));
    }
    Ok(())
}

fn validate_stream(stream: &Stream, tasks: usize, bits: usize) -> Result<(), RuntimeError> {
    if stream.tasks.iter().any(|&id| id as usize >= tasks)
        || stream.bindings.iter().any(|&id| id as usize >= tasks)
        || stream.value.is_some_and(|id| id as usize >= bits)
    {
        return Err(RuntimeError::InvalidProgram("invalid stream operand"));
    }
    Ok(())
}

fn validate_instruction(
    instruction: &Instruction,
    tasks: usize,
    bits: usize,
    records: usize,
    qubits: usize,
) -> Result<(), RuntimeError> {
    match instruction {
        Instruction::Quantum(quantum) => {
            for alternative in &quantum.alternatives {
                if alternative
                    .when
                    .iter()
                    .any(|&(bit, _)| bit as usize >= bits)
                {
                    return Err(RuntimeError::InvalidProgram("invalid quantum selector"));
                }
                if alternative
                    .selected_records
                    .iter()
                    .chain(alternative.excluded_records.iter())
                    .any(|&record| record as usize >= records)
                {
                    return Err(RuntimeError::InvalidProgram("invalid quantum record"));
                }
                validate_quantum_stream(&alternative.stream, records, qubits)?;
                for parity in alternative
                    .detectors
                    .iter()
                    .chain(alternative.restarts.iter())
                {
                    validate_parity(parity, records)?;
                }
            }
        }
        Instruction::Eval(expression) | Instruction::Discard(expression) => {
            validate_bool(expression, bits)?;
        }
        Instruction::Accumulate(parity) => validate_parity(parity, records)?,
        Instruction::Bind(bindings) => {
            for binding in bindings {
                validate_product(&binding.operator, qubits)?;
            }
        }
        Instruction::ReadoutRecipe {
            bits: inputs,
            bindings,
        }
        | Instruction::Observable {
            bits: inputs,
            bindings,
            ..
        } => {
            if inputs.iter().any(|&id| id as usize >= bits)
                || bindings.iter().any(|&id| id as usize >= tasks)
            {
                return Err(RuntimeError::InvalidProgram("invalid observable operand"));
            }
        }
        Instruction::Decode(request) => {
            if request.observable as usize >= tasks
                || request.raw as usize >= bits
                || !valid_duration(request.round_duration)
            {
                return Err(RuntimeError::InvalidProgram("invalid decoder operand"));
            }
        }
        Instruction::Rus {
            body,
            restart,
            owned_qubits,
            attempt_bits,
            attempt_records,
            retry_prepare,
            decoder_hold,
            cultivation_exits,
            ..
        } => {
            validate_stream(body, tasks, bits)?;
            validate_bool(restart, bits)?;
            if owned_qubits.iter().any(|&id| id as usize >= qubits)
                || attempt_bits.iter().any(|&id| id as usize >= bits)
                || attempt_records.iter().any(|&id| id as usize >= records)
                || cultivation_exits.iter().any(|&id| id as usize >= tasks)
            {
                return Err(RuntimeError::InvalidProgram("invalid RUS operand"));
            }
            for cycle in decoder_hold {
                validate_memory_cycle(cycle, records, qubits)?;
            }
            validate_quantum_stream(retry_prepare, records, qubits)?;
        }
        Instruction::WaitFor { until, memory } => {
            if until.iter().any(|&id| id as usize >= tasks) {
                return Err(RuntimeError::InvalidProgram("invalid wait operand"));
            }
            for cycle in memory {
                validate_memory_cycle(cycle, records, qubits)?;
            }
        }
        Instruction::SignalReady { .. } => {}
        Instruction::MemoryRounds {
            stream, detectors, ..
        } => {
            validate_quantum_stream(stream, records, qubits)?;
            for parity in detectors {
                validate_parity(parity, records)?;
            }
        }
        Instruction::Idle {
            duration,
            noise_ratio,
            operations,
        } => {
            if !valid_duration(*duration) || !valid_duration(*noise_ratio) {
                return Err(RuntimeError::InvalidProgram("invalid idle timing"));
            }
            for operation in operations {
                validate_quantum_op(operation, records, qubits)?;
            }
        }
    }
    Ok(())
}

fn validate_memory_cycle(
    cycle: &MemoryCycle,
    records: usize,
    qubits: usize,
) -> Result<(), RuntimeError> {
    let stream_duration = stream_duration(&cycle.stream)?;
    if stream_duration <= 0.0
        || cycle.round_duration <= 0.0
        || !cycle.round_duration.is_finite()
        || (stream_duration - cycle.round_duration).abs() > TIME_EPSILON
    {
        return Err(RuntimeError::InvalidProgram("invalid memory duration"));
    }
    validate_quantum_stream(&cycle.stream, records, qubits)?;
    for parity in &cycle.detectors {
        validate_parity(parity, records)?;
    }
    for initializer in &cycle.initializers {
        if initializer.record as usize >= records {
            return Err(RuntimeError::InvalidProgram("invalid frontier record"));
        }
        validate_parity(&initializer.parity, records)?;
    }
    for flow in &cycle.boundary_flows {
        validate_product(&flow.start, qubits)?;
        validate_product(&flow.end, qubits)?;
        validate_parity(&flow.parity, records)?;
        if flow
            .frontier_record
            .is_some_and(|record| record as usize >= records)
        {
            return Err(RuntimeError::InvalidProgram("invalid frontier record"));
        }
    }
    Ok(())
}

fn validate_quantum_stream(
    stream: &QuantumStream,
    records: usize,
    qubits: usize,
) -> Result<(), RuntimeError> {
    for moment in &stream.moments {
        if !valid_duration(moment.duration) {
            return Err(RuntimeError::InvalidProgram("invalid moment duration"));
        }
        for operation in &moment.operations {
            validate_quantum_op(operation, records, qubits)?;
        }
    }
    Ok(())
}

fn validate_quantum_op(
    operation: &QuantumOp,
    records: usize,
    qubits: usize,
) -> Result<(), RuntimeError> {
    let valid_qubit = |qubit: u32| (qubit as usize) < qubits;
    let valid_record = |record: u32| (record as usize) < records;
    let valid = match operation {
        QuantumOp::Gate1 { qubit, .. }
        | QuantumOp::Pauli { qubit, .. }
        | QuantumOp::T { qubit, .. }
        | QuantumOp::Reset { qubit, .. } => valid_qubit(*qubit),
        QuantumOp::Gate2 {
            control, target, ..
        } => valid_qubit(*control) && valid_qubit(*target) && control != target,
        QuantumOp::Measure {
            observable,
            records: outputs,
            flip_probability,
        } => {
            validate_product(observable, qubits)?;
            valid_probability(*flip_probability)
                && !outputs.is_empty()
                && outputs.iter().all(|&record| valid_record(record))
        }
        QuantumOp::ConditionalPauli { qubit, control, .. } => {
            valid_qubit(*qubit) && valid_record(*control)
        }
        QuantumOp::Depolarize1 {
            probability,
            qubits: operands,
        }
        | QuantumOp::PauliError {
            probability,
            qubits: operands,
            ..
        } => valid_probability(*probability) && operands.iter().all(|&qubit| valid_qubit(qubit)),
        QuantumOp::Depolarize2 { probability, pairs } => {
            valid_probability(*probability)
                && pairs
                    .iter()
                    .all(|&(left, right)| left != right && valid_qubit(left) && valid_qubit(right))
        }
    };
    if valid {
        Ok(())
    } else {
        Err(RuntimeError::InvalidProgram("invalid quantum operand"))
    }
}

fn validate_product(product: &PauliProduct, qubits: usize) -> Result<(), RuntimeError> {
    let mut seen = HashSet::new();
    if product
        .terms
        .iter()
        .any(|&(qubit, _)| qubit as usize >= qubits || !seen.insert(qubit))
    {
        Err(RuntimeError::InvalidProgram("invalid Pauli operand"))
    } else {
        Ok(())
    }
}

fn validate_parity(parity: &RecordParity, records: usize) -> Result<(), RuntimeError> {
    if parity
        .records
        .iter()
        .any(|&record| record as usize >= records)
    {
        Err(RuntimeError::InvalidProgram("invalid record operand"))
    } else {
        Ok(())
    }
}

fn validate_bool(expression: &BoolOp, bits: usize) -> Result<(), RuntimeError> {
    match expression {
        BoolOp::Call { body, inputs } => {
            if inputs.iter().any(|&bit| bit as usize >= bits) {
                return Err(RuntimeError::InvalidProgram("invalid function argument"));
            }
            validate_bool(body, inputs.len())
        }
        BoolOp::Const(_) => Ok(()),
        BoolOp::Copy(bit) => ((*bit as usize) < bits)
            .then_some(())
            .ok_or(RuntimeError::InvalidProgram("invalid bit operand")),
        BoolOp::Not(inner) => validate_bool(inner, bits),
        BoolOp::Parity { inputs, .. } => inputs
            .iter()
            .all(|&bit| (bit as usize) < bits)
            .then_some(())
            .ok_or(RuntimeError::InvalidProgram("invalid bit operand")),
        BoolOp::Xor(operands) | BoolOp::And(operands) | BoolOp::Or(operands) => {
            for operand in operands {
                validate_bool(operand, bits)?;
            }
            Ok(())
        }
        BoolOp::Select {
            condition,
            when_false,
            when_true,
        } => {
            validate_bool(condition, bits)?;
            validate_bool(when_false, bits)?;
            validate_bool(when_true, bits)
        }
    }
}

fn stream_duration(stream: &QuantumStream) -> Result<f64, RuntimeError> {
    stream.moments.iter().try_fold(0.0, |duration, moment| {
        if valid_duration(moment.duration) {
            Ok(duration + moment.duration)
        } else {
            Err(RuntimeError::InvalidProgram("invalid moment duration"))
        }
    })
}

fn quantum_operation_names(stream: &QuantumStream) -> Vec<String> {
    stream
        .moments
        .iter()
        .flat_map(|moment| &moment.operations)
        .filter_map(|operation| match operation {
            QuantumOp::Gate1 { gate, .. } => Some(format!("{gate:?}")),
            QuantumOp::Gate2 {
                control_basis,
                target_basis,
                ..
            } => Some(format!("C{control_basis:?}{target_basis:?}")),
            QuantumOp::Pauli { basis, .. } => Some(format!("{basis:?}")),
            QuantumOp::T { adjoint, .. } => Some(if *adjoint { "T_DAG" } else { "T" }.into()),
            QuantumOp::ConditionalPauli { basis, .. } => Some(format!("conditional_{basis:?}")),
            QuantumOp::Measure { .. }
            | QuantumOp::Reset { .. }
            | QuantumOp::Depolarize1 { .. }
            | QuantumOp::Depolarize2 { .. }
            | QuantumOp::PauliError { .. } => None,
        })
        .collect()
}

fn instruction_kind(instruction: &Instruction) -> &'static str {
    match instruction {
        Instruction::Quantum(_) => "quantum",
        Instruction::Eval(_) => "eval",
        Instruction::Accumulate(_) => "accumulate",
        Instruction::Bind(_) => "bind",
        Instruction::ReadoutRecipe { .. } => "readout_recipe",
        Instruction::Observable { .. } => "observable",
        Instruction::Decode(_) => "decode",
        Instruction::Discard(_) => "discard",
        Instruction::Rus { .. } => "rus",
        Instruction::WaitFor { .. } => "wait_for",
        Instruction::SignalReady { .. } => "signal_ready",
        Instruction::MemoryRounds { .. } => "memory_rounds",
        Instruction::Idle { .. } => "idle",
    }
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_duration(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn max_time(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn basis(pauli: Pauli) -> PauliBasis {
    match pauli {
        Pauli::X => PauliBasis::X,
        Pauli::Y => PauliBasis::Y,
        Pauli::Z => PauliBasis::Z,
    }
}

fn backend_pauli(pauli: Pauli) -> BackendPauli {
    match pauli {
        Pauli::X => BackendPauli::X,
        Pauli::Y => BackendPauli::Y,
        Pauli::Z => BackendPauli::Z,
    }
}

fn set_pauli(product: &mut PauliString, qubit: usize, pauli: Pauli) {
    product.set(qubit, backend_pauli(pauli));
}

fn engine_pauli_product(
    product: &PauliProduct,
    qubit_count: usize,
) -> Result<PauliString, RuntimeError> {
    let mut out = PauliString::new(qubit_count);
    let mut seen = HashSet::new();
    for &(qubit, axis) in &product.terms {
        if (qubit as usize) >= qubit_count || !seen.insert(qubit) {
            return Err(RuntimeError::InvalidProgram("invalid Pauli operand"));
        }
        set_pauli(&mut out, qubit as usize, axis);
    }
    if product.negative {
        out.phase_shift(2);
    }
    Ok(out)
}

fn gate1(gate: Clifford1) -> Gate1Q {
    match gate {
        Clifford1::H => Gate1Q::H,
        Clifford1::H_XY => Gate1Q::Hxy,
        Clifford1::H_YZ => Gate1Q::Hyz,
        Clifford1::H_NXY => Gate1Q::Hnxy,
        Clifford1::H_NXZ => Gate1Q::Hnxz,
        Clifford1::H_NYZ => Gate1Q::Hnyz,
        Clifford1::SQRT_X => Gate1Q::SqrtX,
        Clifford1::SQRT_X_DAG => Gate1Q::SqrtXDag,
        Clifford1::SQRT_Y => Gate1Q::SqrtY,
        Clifford1::SQRT_Y_DAG => Gate1Q::SqrtYDag,
        Clifford1::S => Gate1Q::S,
        Clifford1::S_DAG => Gate1Q::SDag,
        Clifford1::C_XYZ => Gate1Q::Cxyz,
        Clifford1::C_ZYX => Gate1Q::Czyx,
        Clifford1::C_NXYZ => Gate1Q::Cnxyz,
        Clifford1::C_XNYZ => Gate1Q::Cxnyz,
        Clifford1::C_XYNZ => Gate1Q::Cxynz,
        Clifford1::C_NZYX => Gate1Q::Cnzyx,
        Clifford1::C_ZNYX => Gate1Q::Cznyx,
        Clifford1::C_ZYNX => Gate1Q::Czynx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::{Moment, QuantumAlternative, QuantumTask, Task};

    #[test]
    fn select_evaluates_unselected_arm_strictly() {
        let program = Program {
            qubit_count: 0,
            bit_count: 2,
            tasks: vec![Task {
                label: "strict select".into(),
                release: 0.0,
                source: None,
                dependencies: Box::default(),
                activation: None,
                output: Some(0),
                qubits: Box::default(),
                duration: 0.0,
                instruction: Instruction::Eval(BoolOp::Select {
                    condition: Box::new(BoolOp::Const(false)),
                    when_false: Box::new(BoolOp::Const(true)),
                    when_true: Box::new(BoolOp::Copy(1)),
                }),
            }],
            entry: Stream {
                tasks: vec![0].into_boxed_slice(),
                ..Stream::default()
            },
            ..Program::default()
        };
        assert_eq!(
            run(&program, RuntimeConfig::default())
                .unwrap()
                .artifact
                .final_bits[0],
            None
        );
    }

    #[test]
    fn idle_error_rate_is_not_probability_bounded() {
        let mut config = RuntimeConfig {
            idle_error_rate: 4.0,
            ..RuntimeConfig::default()
        };
        run(&Program::default(), config.clone()).unwrap();
        config.idle_error_rate = -1.0;
        assert!(matches!(
            run(&Program::default(), config),
            Err(RuntimeError::InvalidProgram(
                "invalid runtime configuration"
            ))
        ));
    }

    #[test]
    fn rus_rejects_unowned_operation_targets_even_when_support_omits_them() {
        let stream = QuantumStream {
            moments: vec![Moment {
                duration: 1.0,
                operations: vec![QuantumOp::Reset {
                    basis: Pauli::Z,
                    qubit: 1,
                }]
                .into_boxed_slice(),
                ..Moment::default()
            }]
            .into_boxed_slice(),
        };
        let body = Task {
            label: "body".into(),
            release: 0.0,
            source: None,
            dependencies: Box::default(),
            activation: None,
            output: None,
            qubits: Box::new([0]),
            duration: 1.0,
            instruction: Instruction::Quantum(QuantumTask {
                alternatives: vec![QuantumAlternative {
                    stream: stream.clone(),
                    ..QuantumAlternative::default()
                }]
                .into_boxed_slice(),
            }),
        };
        let owner = Task {
            instruction: Instruction::Rus {
                body: Stream {
                    tasks: Box::new([0]),
                    ..Stream::default()
                },
                restart: BoolOp::Const(false),
                owned_qubits: Box::new([0]),
                attempt_bits: Box::default(),
                attempt_records: Box::default(),
                resource: 0,
                retry_prepare: QuantumStream::default(),
                decoder_hold: Box::default(),
                cultivation_exits: Box::default(),
            },
            ..body.clone()
        };
        let mut program = Program {
            qubit_count: 2,
            tasks: vec![body, owner],
            entry: Stream {
                tasks: Box::new([1]),
                ..Stream::default()
            },
            ..Program::default()
        };
        assert!(matches!(
            run(&program, RuntimeConfig::default()),
            Err(RuntimeError::InvalidProgram(
                "RUS body touches qubits outside its isolated source support"
            ))
        ));

        program.tasks[0].instruction = Instruction::Eval(BoolOp::Const(false));
        let Instruction::Rus {
            decoder_hold,
            cultivation_exits,
            ..
        } = &mut program.tasks[1].instruction
        else {
            unreachable!()
        };
        *decoder_hold = Box::new([MemoryCycle {
            stream,
            round_duration: 1.0,
            ..MemoryCycle::default()
        }]);
        *cultivation_exits = Box::new([0]);
        assert!(matches!(
            run(&program, RuntimeConfig::default()),
            Err(RuntimeError::InvalidProgram(
                "RUS decoder hold has invalid data support"
            ))
        ));
    }

    #[test]
    fn fixed_register_rejects_out_of_range_quantum_operand() {
        let program = Program {
            qubit_count: 1,
            tasks: vec![Task {
                release: 0.0,
                source: None,
                dependencies: Box::default(),
                activation: None,
                output: None,
                qubits: Box::default(),
                duration: 0.0,
                label: "bad reset".into(),
                instruction: Instruction::Quantum(QuantumTask {
                    alternatives: vec![QuantumAlternative {
                        when: Box::default(),
                        stream: QuantumStream {
                            moments: vec![Moment {
                                kind: None,
                                duration: 0.0,
                                operations: vec![QuantumOp::Reset {
                                    basis: Pauli::Z,
                                    qubit: 1,
                                }]
                                .into_boxed_slice(),
                            }]
                            .into_boxed_slice(),
                        },
                        selected_records: Box::default(),
                        excluded_records: Box::default(),
                        detectors: Box::default(),
                        restarts: Box::default(),
                    }]
                    .into_boxed_slice(),
                }),
            }],
            entry: Stream {
                tasks: vec![0].into_boxed_slice(),
                ..Stream::default()
            },
            decoder_latency_rounds: 10,
            ..Program::default()
        };
        assert!(matches!(
            run(&program, RuntimeConfig::default()),
            Err(RuntimeError::InvalidProgram("invalid quantum operand"))
        ));
        assert!(matches!(
            validate_product(
                &PauliProduct {
                    negative: false,
                    terms: vec![(0, Pauli::X), (0, Pauli::Z)].into_boxed_slice(),
                },
                1,
            ),
            Err(RuntimeError::InvalidProgram("invalid Pauli operand"))
        ));
    }

    #[test]
    fn vm_y_product_keeps_the_backend_hermitian_phase() {
        let product = engine_pauli_product(
            &PauliProduct {
                negative: true,
                terms: vec![(0, Pauli::Y)].into_boxed_slice(),
            },
            1,
        )
        .unwrap();
        Simulator::with_seed(1, 0)
            .measure_observable(&product)
            .unwrap();
    }

    #[test]
    fn logical_bloch_applies_crossed_frame_signs() {
        let mut simulator = Simulator::with_seed(1, 0);
        simulator.h(0);
        let result = RunResult {
            simulator,
            artifact: ExecutionArtifact {
                final_bits: vec![Some(false), Some(true)],
                ..ExecutionArtifact::default()
            },
        };
        let output = LogicalOutput {
            port: [0; 3],
            x: PauliProduct {
                negative: false,
                terms: vec![(0, Pauli::X)].into_boxed_slice(),
            },
            z: PauliProduct {
                negative: false,
                terms: vec![(0, Pauli::Z)].into_boxed_slice(),
            },
            frame_x: 0,
            frame_z: 1,
        };
        assert_eq!(result.logical_bloch(&output).unwrap(), (-1.0, 0.0, 0.0));
    }
}
