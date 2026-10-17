use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use crate::backend::{Instruction as BackendInstruction, PauliString, Simulator};
use crate::decoder::{DecodeKey, MockStreamingDecoder};
use crate::instruction::{
    BitId, BoolOp, BoundaryBinding, DecodeRequest, DecodeTiming, Instruction, LogicalInput,
    MemoryCycle, Moment, Pauli, Program, QuantumAlternative, QuantumOp, QuantumStream, RecordId,
    RecordParity, SourceRole, Stream, TaskId,
};

use super::{
    DecoderDecisionRecord, DetectorResult, ExecutionArtifact, ExecutionEvent, LogicalInputState,
    MeasurementRecord, MemoryKind, ObservableResult, RetryRecord, RunMetadata, RunResult,
    RuntimeConfig, RuntimeError, StopReason, TIME_EPSILON, TaskMetadata, TaskTiming, WaitReason,
    basis, engine_pauli_product, gate1, instruction_kind, max_time, validate_program,
};

/// The idle model saturates at the maximally mixing single-qubit channel:
/// X, Y, and Z each land with `p/3`, so `p = 3/4` is completely depolarizing.
/// `idle_error_rate` is an unbounded per-time-unit rate; this policy prevents
/// longer waits from passing that point and becoming less mixing.
const MAX_DEPOLARIZE1_PROBABILITY: f64 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq)]
enum TaskState {
    Dormant,
    Pending { epoch: u64 },
    Running { epoch: u64, start: f64 },
    Done { epoch: u64, end: f64 },
    Canceled { epoch: u64 },
}

impl TaskState {
    fn epoch(self) -> Option<u64> {
        match self {
            Self::Dormant => None,
            Self::Pending { epoch }
            | Self::Running { epoch, .. }
            | Self::Done { epoch, .. }
            | Self::Canceled { epoch } => Some(epoch),
        }
    }

    fn end(self) -> Option<f64> {
        match self {
            Self::Done { end, .. } => Some(end),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum WorkKey {
    Task(TaskId),
    Retry(TaskId),
    Cycle(u64, usize),
    WaitIdle(TaskId),
}

#[derive(Debug)]
enum Purpose {
    Quantum {
        detectors: Box<[RecordParity]>,
    },
    StaticMemory {
        rounds: u32,
        detectors: Box<[RecordParity]>,
    },
    TaskIdle,
    Retry {
        attempt: u32,
    },
    Cycle {
        group: u64,
        cycle: usize,
    },
    WaitIdle {
        qubits: Box<[u32]>,
        wait_reason: WaitReason,
    },
}

#[derive(Debug)]
struct PhysicalRun {
    owner: TaskId,
    epoch: u64,
    stream: QuantumStream,
    next_moment: usize,
    event_end: Option<f64>,
    start: f64,
    last_end: f64,
    purpose: Purpose,
}

#[derive(Debug)]
struct ScheduledEvent {
    at: f64,
    sequence: u64,
    kind: EventKind,
}

#[derive(Debug)]
enum EventKind {
    Moment {
        key: WorkKey,
        epoch: u64,
        moment: usize,
        operations: Box<[QuantumOp]>,
    },
    Decoder {
        task: TaskId,
        epoch: u64,
        key: DecodeKey,
        raw: bool,
        measured_at: Option<f64>,
        output: bloq_ir::ObservableOutput,
    },
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at && self.sequence == other.sequence
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at
            .total_cmp(&self.at)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

#[derive(Debug, Clone)]
struct EarlyCheck {
    task: TaskId,
    alternative: usize,
    index: usize,
    parity: RecordParity,
    evaluated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RusPhase {
    Attempt,
    Gap,
    AwaitBody,
    Aborting,
    Preparing,
}

#[derive(Debug, Clone)]
struct RusRun {
    outer_epoch: u64,
    start: f64,
    body: Stream,
    restart: BoolOp,
    attempt_bits: Box<[BitId]>,
    attempt_records: Box<[RecordId]>,
    retry_prepare: QuantumStream,
    decoder_hold: Box<[MemoryCycle]>,
    cultivation_exits: Box<[TaskId]>,
    attempt: u32,
    epoch: u64,
    attempt_start: f64,
    phase: RusPhase,
    checks: Vec<EarlyCheck>,
    gap_round: u32,
    gap_group: Option<u64>,
    gap_ready: Option<f64>,
}

#[derive(Debug, Clone)]
struct WaitRun {
    epoch: u64,
    start: f64,
    until: Box<[TaskId]>,
    memory: Box<[MemoryCycle]>,
    round: u32,
    group: Option<u64>,
    idle: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundKind {
    Gap { rus: TaskId },
    Wait { task: TaskId },
}

#[derive(Debug, Clone)]
struct CycleSnapshot {
    previous: Vec<Option<(RecordId, bool)>>,
}

#[derive(Debug)]
struct RoundGroup {
    owner: TaskId,
    epoch: u64,
    kind: RoundKind,
    round: u32,
    start: f64,
    end: f64,
    remaining: usize,
    cycles: Box<[MemoryCycle]>,
    snapshots: Vec<CycleSnapshot>,
    wait_reason: Option<WaitReason>,
}

#[derive(Default)]
struct CausalBlockers {
    causal_cut: bool,
    decoder_latency: bool,
}

enum BoundValues {
    Owned(Vec<BoundaryBinding>),
    Recipe(Box<[Arc<BoundValues>]>),
}

fn multiply_boundary(product: &mut Option<PauliString>, factor: &PauliString) {
    // Some(identity) still records a real Output product: its phase can be -1.
    *product = Some(match product.take() {
        Some(current) => &current * factor,
        None => factor.clone(),
    });
}

fn binding_product(
    root: &Arc<BoundValues>,
    width: usize,
    products: &mut HashMap<usize, Option<PauliString>>,
) -> Result<(), RuntimeError> {
    let mut stack = vec![(root.clone(), false)];
    let mut active = HashSet::new();
    while let Some((binding, finish)) = stack.pop() {
        let key = Arc::as_ptr(&binding) as usize;
        if products.contains_key(&key) {
            continue;
        }
        if !finish {
            if !active.insert(key) {
                return Err(RuntimeError::InvalidProgram("cyclic binding recipe"));
            }
            stack.push((binding.clone(), true));
            if let BoundValues::Recipe(children) = binding.as_ref() {
                stack.extend(children.iter().rev().cloned().map(|child| (child, false)));
            }
            continue;
        }
        let mut product = None;
        match binding.as_ref() {
            BoundValues::Owned(values) => {
                for value in values.iter().filter(|value| !value.input) {
                    multiply_boundary(&mut product, &engine_pauli_product(&value.operator, width)?);
                }
            }
            BoundValues::Recipe(children) => {
                for child in children {
                    if let Some(factor) = &products[&(Arc::as_ptr(child) as usize)] {
                        multiply_boundary(&mut product, factor);
                    }
                }
            }
        }
        products.insert(key, product);
        active.remove(&key);
    }
    Ok(())
}

pub(super) fn run(program: &Program, config: RuntimeConfig) -> Result<RunResult, RuntimeError> {
    Machine::new(program, config)?.run()
}

fn filled_register<T: Clone>(
    len: usize,
    value: T,
    register: &'static str,
) -> Result<Vec<T>, RuntimeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|source| RuntimeError::RegisterAllocation { register, source })?;
    values.resize(len, value);
    Ok(values)
}

struct Machine<'a> {
    program: &'a Program,
    config: RuntimeConfig,
    simulator: Simulator,
    peak_rank: usize,
    decoder: MockStreamingDecoder,
    now: f64,
    sequence: u64,
    steps: u64,
    events: BinaryHeap<ScheduledEvent>,
    states: Vec<TaskState>,
    bits: Vec<Option<bool>>,
    bit_times: Vec<Option<f64>>,
    records: Vec<Option<bool>>,
    record_times: Vec<Option<f64>>,
    bindings: HashMap<TaskId, Arc<BoundValues>>,
    qubit_free: Vec<f64>,
    actor_rngs: HashMap<(TaskId, u64), Simulator>,
    physical: HashMap<WorkKey, PhysicalRun>,
    rus: HashMap<TaskId, RusRun>,
    waits: HashMap<TaskId, WaitRun>,
    groups: HashMap<u64, RoundGroup>,
    next_group: u64,
    task_rus_owner: Vec<Option<TaskId>>,
    source_released: HashSet<TaskId>,
    logged_decodes: HashSet<DecodeKey>,
    artifact: ExecutionArtifact,
    stop_at: Option<f64>,
}

impl<'a> Machine<'a> {
    fn new(program: &'a Program, config: RuntimeConfig) -> Result<Self, RuntimeError> {
        validate_program(program)?;
        if config.limits.max_steps == 0
            || config.limits.max_attempts == 0
            || !config.idle_error_rate.is_finite()
            || config.idle_error_rate < 0.0
        {
            return Err(RuntimeError::InvalidProgram(
                "invalid runtime configuration",
            ));
        }
        let decoder = MockStreamingDecoder::new(
            config.decoder.clone(),
            config.seed ^ u64::MAX,
            program.decoder_latency_rounds,
        )?;
        let states = filled_register(program.tasks.len(), TaskState::Dormant, "task states")?;
        let bits = filled_register(program.bit_count as usize, None, "bits")?;
        let bit_times = filled_register(program.bit_count as usize, None, "bit timestamps")?;
        let records = filled_register(program.record_count as usize, None, "records")?;
        let record_times =
            filled_register(program.record_count as usize, None, "record timestamps")?;
        let qubit_free = filled_register(program.qubit_count as usize, 0.0, "qubit timestamps")?;
        let task_rus_owner = filled_register(program.tasks.len(), None, "task owners")?;
        let artifact = ExecutionArtifact {
            metadata: RunMetadata {
                backend: "Ticit".into(),
                seed: config.seed,
                input_state: config.input_state,
                decoder: config.decoder.clone(),
                decoder_latency_rounds: program.decoder_latency_rounds,
                idle_error_rate: config.idle_error_rate,
                limits: config.limits,
                tasks: program
                    .tasks
                    .iter()
                    .enumerate()
                    .map(|(task, value)| TaskMetadata {
                        task: task as TaskId,
                        label: value.label.clone(),
                        instruction: instruction_kind(&value.instruction).into(),
                        origin: program.task_origins.get(task).cloned(),
                        dependencies: value.dependencies.to_vec(),
                        wait_until: match &value.instruction {
                            Instruction::WaitFor { until, .. } => until.to_vec(),
                            _ => Vec::new(),
                        },
                        output: value.output,
                    })
                    .collect(),
            },
            ..ExecutionArtifact::default()
        };
        let simulator = Simulator::with_seed(program.qubit_count as usize, config.seed);
        let peak_rank = simulator.rank();
        let mut machine = Self {
            program,
            simulator,
            peak_rank,
            config,
            decoder,
            now: 0.0,
            sequence: 0,
            steps: 0,
            events: BinaryHeap::new(),
            states,
            bits,
            bit_times,
            records,
            record_times,
            bindings: HashMap::new(),
            qubit_free,
            actor_rngs: HashMap::new(),
            physical: HashMap::new(),
            rus: HashMap::new(),
            waits: HashMap::new(),
            groups: HashMap::new(),
            next_group: 0,
            task_rus_owner,
            source_released: HashSet::new(),
            logged_decodes: HashSet::new(),
            artifact,
            stop_at: None,
        };
        machine.map_rus_owners(&program.entry, None)?;
        machine.activate_stream(&program.entry, 0)?;
        Ok(machine)
    }

    fn run(mut self) -> Result<RunResult, RuntimeError> {
        loop {
            self.settle()?;
            if self.artifact.discarded || self.all_entry_done() {
                break;
            }
            let next_event = self.events.peek().map(|event| event.at);
            let next_release = self.next_ready_release();
            let next = match (next_event, next_release) {
                (Some(event), Some(release)) => event.min(release),
                (Some(event), None) => event,
                (None, Some(release)) => release,
                (None, None) => {
                    return Err(RuntimeError::InvalidProgram(
                        "causal scheduler reached an unavailable dependency",
                    ));
                }
            };
            if next + TIME_EPSILON < self.now {
                return Err(RuntimeError::InvalidProgram("VM clock moved backwards"));
            }
            self.now = next;
            self.complete_event_batch()?;
        }
        self.finish_reports();
        Ok(RunResult {
            simulator: self.simulator,
            artifact: self.artifact,
        })
    }

    fn step(&mut self) -> Result<(), RuntimeError> {
        if self.steps >= self.config.limits.max_steps {
            return Err(RuntimeError::StepLimit {
                limit: self.config.limits.max_steps,
            });
        }
        self.steps += 1;
        Ok(())
    }

    fn map_rus_owners(
        &mut self,
        stream: &Stream,
        owner: Option<TaskId>,
    ) -> Result<(), RuntimeError> {
        for &task in &stream.tasks {
            let slot = self
                .task_rus_owner
                .get_mut(task as usize)
                .ok_or(RuntimeError::MissingTask(task))?;
            if slot.is_some() && *slot != owner {
                return Err(RuntimeError::InvalidProgram(
                    "task belongs to two dynamic scopes",
                ));
            }
            *slot = owner;
            if let Instruction::Rus { body, .. } = &self.program.tasks[task as usize].instruction {
                if owner.is_some() {
                    return Err(RuntimeError::InvalidProgram(
                        "nested dynamic RUS is unsupported",
                    ));
                }
                self.map_rus_owners(body, Some(task))?;
            }
        }
        Ok(())
    }

    fn activate_stream(&mut self, stream: &Stream, epoch: u64) -> Result<(), RuntimeError> {
        for &task in &stream.tasks {
            let state = self
                .states
                .get_mut(task as usize)
                .ok_or(RuntimeError::MissingTask(task))?;
            *state = TaskState::Pending { epoch };
            self.bindings.remove(&task);
        }
        Ok(())
    }

    fn settle(&mut self) -> Result<(), RuntimeError> {
        loop {
            if self.artifact.discarded {
                return Ok(());
            }
            let mut progress = self.poll_regions()?;
            progress |= self.start_ready_classical()?;
            if self.event_due_now() {
                return Ok(());
            }
            if progress {
                continue;
            }
            let mut physical_progress = self.start_ready_physical_tasks()?;
            physical_progress |= self.progress_physical()?;
            if self.event_due_now() {
                return Ok(());
            }
            if physical_progress {
                continue;
            }
            let mut hold_progress = self.progress_waits()?;
            hold_progress |= self.progress_gaps()?;
            if self.event_due_now() || !hold_progress {
                return Ok(());
            }
        }
    }

    fn start_ready_classical(&mut self) -> Result<bool, RuntimeError> {
        let ready = (0..self.program.tasks.len())
            .filter_map(|index| {
                let task = index as TaskId;
                self.classical_ready(task).then_some(task)
            })
            .collect::<Vec<_>>();
        if ready.is_empty() {
            return Ok(false);
        }
        for task in ready {
            self.start_classical(task)?;
        }
        Ok(true)
    }

    fn classical_ready(&self, task: TaskId) -> bool {
        let Some(TaskState::Pending { .. }) = self.states.get(task as usize) else {
            return false;
        };
        let value = &self.program.tasks[task as usize];
        if self.now + TIME_EPSILON < value.release || !self.dependencies_done(value) {
            return false;
        }
        if value
            .activation
            .is_some_and(|activation| self.bits[activation as usize].is_none())
        {
            return false;
        }
        match &value.instruction {
            Instruction::Quantum(_)
            | Instruction::WaitFor { .. }
            | Instruction::MemoryRounds { .. }
            | Instruction::Idle { .. } => false,
            Instruction::Decode(request) => {
                if request.timing == DecodeTiming::FactoryCompletion {
                    self.task_rus_owner[task as usize]
                        .and_then(|rus| self.rus.get(&rus))
                        .and_then(|run| run.gap_ready)
                        .is_some_and(|ready| ready <= self.now + TIME_EPSILON)
                } else {
                    true
                }
            }
            Instruction::ReadoutRecipe { bindings, .. }
            | Instruction::Observable { bindings, .. } => bindings
                .iter()
                .all(|binding_task| self.bindings.contains_key(binding_task)),
            _ => true,
        }
    }

    fn dependencies_done(&self, task: &crate::instruction::Task) -> bool {
        task.dependencies
            .iter()
            .all(|dependency| self.states[*dependency as usize].end().is_some())
    }

    fn start_classical(&mut self, task: TaskId) -> Result<(), RuntimeError> {
        self.step()?;
        let value = self.program.tasks[task as usize].clone();
        let epoch = self.states[task as usize]
            .epoch()
            .ok_or(RuntimeError::MissingTask(task))?;
        let start = self.now;
        self.emit_source(task, start)?;
        if let Some(activation) = value.activation {
            match self.bit(activation)? {
                Some(false) => {
                    if let Some(output) = value.output {
                        self.write_bit(output, Some(false), self.bit_times[activation as usize])?;
                    }
                    self.bindings
                        .insert(task, Arc::new(BoundValues::Owned(Vec::new())));
                    self.complete_task(task, epoch, start, start);
                    return Ok(());
                }
                Some(true) => {}
                None => return Ok(()),
            }
        }
        match value.instruction {
            Instruction::Eval(expression) => {
                let (result, measured) = self.eval_bool(&expression)?;
                if let Some(output) = value.output {
                    self.write_bit(output, result, measured)?;
                }
                self.complete_task(task, epoch, start, start);
            }
            Instruction::Accumulate(parity) => {
                let result = self
                    .available_parity(&parity)
                    .ok_or_else(|| first_missing_record(&parity, &self.records))?;
                let measured = self.parity_time(&parity)?;
                if let Some(output) = value.output {
                    self.write_bit(output, Some(result), measured)?;
                }
                self.complete_task(task, epoch, start, start);
            }
            Instruction::Bind(boundaries) => {
                self.bindings
                    .insert(task, Arc::new(BoundValues::Owned(boundaries.into_vec())));
                self.complete_task(task, epoch, start, start);
            }
            Instruction::ReadoutRecipe { bits, bindings } => {
                let mut parity = Some(false);
                let mut measured = None;
                for bit in bits {
                    parity = match (parity, self.bits[bit as usize]) {
                        (Some(left), Some(right)) => Some(left ^ right),
                        _ => None,
                    };
                    measured = max_time(measured, self.bit_times[bit as usize]);
                }
                if let Some(output) = value.output {
                    self.write_bit(output, parity, measured)?;
                }
                let children = self.binding_sources(&bindings)?;
                self.bindings
                    .insert(task, Arc::new(BoundValues::Recipe(children)));
                self.complete_task(task, epoch, start, start);
            }
            Instruction::Observable {
                index,
                bits,
                bindings,
            } => {
                let mut raw = Some(false);
                let mut measured = None;
                for bit in bits {
                    raw = match (raw, self.bits[bit as usize]) {
                        (Some(left), Some(right)) => Some(left ^ right),
                        _ => None,
                    };
                    measured = max_time(measured, self.bit_times[bit as usize]);
                }
                let mut boundary = None;
                let mut products = HashMap::new();
                for &binding_task in &bindings {
                    let root = self
                        .bindings
                        .get(&binding_task)
                        .ok_or(RuntimeError::MissingTask(binding_task))?;
                    binding_product(root, self.program.qubit_count as usize, &mut products)?;
                    if let Some(product) = &products[&(Arc::as_ptr(root) as usize)] {
                        multiply_boundary(&mut boundary, product);
                    }
                }
                if let Some(output) = value.output {
                    self.write_bit(output, raw, measured)?;
                }
                let children = self.binding_sources(&bindings)?;
                self.bindings
                    .insert(task, Arc::new(BoundValues::Recipe(children)));
                let report = if let Some(boundary) = boundary {
                    let measured = self
                        .simulator
                        .clone()
                        .measure_observable(&boundary)?
                        .outcome;
                    raw.map(|raw| raw ^ measured)
                } else {
                    raw
                };
                self.artifact.observables.push(ObservableResult {
                    index,
                    task,
                    raw,
                    value: report,
                    attempt: self.attempt_of(task, epoch),
                    time: start,
                    committed: self.attempt_of(task, epoch).is_none(),
                });
                self.complete_task(task, epoch, start, start);
            }
            Instruction::Decode(request) => {
                self.start_decode(task, epoch, start, &request)?;
            }
            Instruction::Discard(condition) => {
                if self.eval_bool(&condition)?.0 == Some(true) {
                    self.discard(task, StopReason::Discard, start);
                } else {
                    self.complete_task(task, epoch, start, start);
                }
            }
            Instruction::Rus {
                body,
                restart,
                owned_qubits: _,
                attempt_bits,
                attempt_records,
                resource: _,
                retry_prepare,
                decoder_hold,
                cultivation_exits,
            } => {
                self.start_rus(
                    task,
                    epoch,
                    start,
                    body,
                    restart,
                    attempt_bits,
                    attempt_records,
                    retry_prepare,
                    decoder_hold,
                    cultivation_exits,
                )?;
            }
            Instruction::SignalReady { resource } => {
                self.artifact.events.push(ExecutionEvent::FactoryReady {
                    resource,
                    time: start,
                });
                self.complete_task(task, epoch, start, start);
            }
            Instruction::Quantum(_)
            | Instruction::WaitFor { .. }
            | Instruction::MemoryRounds { .. }
            | Instruction::Idle { .. } => unreachable!("physical instruction filtered"),
        }
        Ok(())
    }

    fn start_decode(
        &mut self,
        task: TaskId,
        epoch: u64,
        start: f64,
        request: &DecodeRequest,
    ) -> Result<(), RuntimeError> {
        let &DecodeRequest {
            observable,
            raw,
            round_duration,
            timing,
            ..
        } = request;
        let Some(raw_value) = self.bit(raw)? else {
            if let Some(output) = self.program.tasks[task as usize].output {
                self.write_bit(output, None, self.bit_times[raw as usize])?;
            }
            self.complete_task(task, epoch, start, start);
            return Ok(());
        };
        let measured_at = self.bit_times[raw as usize];
        let attempt = self.attempt_of(task, epoch).unwrap_or(0);
        let key = DecodeKey {
            observable_task: observable,
            attempt,
        };
        let deadline = match timing {
            DecodeTiming::Measurement => measured_at
                .map(|time| time + round_duration * f64::from(self.program.decoder_latency_rounds))
                .unwrap_or(start)
                .max(start),
            DecodeTiming::FactoryCompletion => self.task_rus_owner[task as usize]
                .and_then(|rus| self.rus.get(&rus))
                .and_then(|run| run.gap_ready)
                .ok_or(RuntimeError::InvalidProgram(
                    "factory decode issued before physical GAP rounds",
                ))?
                .max(start),
        };
        if !deadline.is_finite() {
            return Err(RuntimeError::InvalidTiming);
        }
        self.states[task as usize] = TaskState::Running { epoch, start };
        self.artifact.events.push(ExecutionEvent::DecoderDeadline {
            task,
            observable,
            factory: timing == DecodeTiming::FactoryCompletion,
            measurements_ready_at: measured_at,
            requested_at: start,
            ready_at: deadline,
        });
        self.push_event(
            deadline,
            EventKind::Decoder {
                task,
                epoch,
                key,
                raw: raw_value,
                measured_at,
                output: request.output,
            },
        );
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one lowered RUS instruction contract"
    )]
    fn start_rus(
        &mut self,
        task: TaskId,
        outer_epoch: u64,
        start: f64,
        body: Stream,
        restart: BoolOp,
        attempt_bits: Box<[BitId]>,
        attempt_records: Box<[RecordId]>,
        retry_prepare: QuantumStream,
        decoder_hold: Box<[MemoryCycle]>,
        cultivation_exits: Box<[TaskId]>,
    ) -> Result<(), RuntimeError> {
        self.states[task as usize] = TaskState::Running {
            epoch: outer_epoch,
            start,
        };
        let epoch = attempt_epoch(task, 0);
        self.activate_stream(&body, epoch)?;
        self.artifact
            .events
            .push(ExecutionEvent::RusAttemptStarted {
                task,
                attempt: 0,
                epoch,
                time: start,
            });
        self.rus.insert(
            task,
            RusRun {
                outer_epoch,
                start,
                body,
                restart,
                attempt_bits,
                attempt_records,
                retry_prepare,
                decoder_hold,
                cultivation_exits,
                attempt: 0,
                epoch,
                attempt_start: start,
                phase: RusPhase::Attempt,
                checks: Vec::new(),
                gap_round: 0,
                gap_group: None,
                gap_ready: None,
            },
        );
        Ok(())
    }

    fn start_ready_physical_tasks(&mut self) -> Result<bool, RuntimeError> {
        let ready = (0..self.program.tasks.len())
            .filter_map(|index| {
                let task = index as TaskId;
                self.physical_task_ready(task).then_some(task)
            })
            .collect::<Vec<_>>();
        let mut progress = false;
        for task in ready {
            progress |= self.start_physical_task(task)?;
        }
        Ok(progress)
    }

    fn physical_task_ready(&self, task: TaskId) -> bool {
        let Some(TaskState::Pending { .. }) = self.states.get(task as usize) else {
            return false;
        };
        let value = &self.program.tasks[task as usize];
        self.now + TIME_EPSILON >= value.release
            && self.dependencies_done(value)
            && matches!(
                value.instruction,
                Instruction::Quantum(_)
                    | Instruction::WaitFor { .. }
                    | Instruction::MemoryRounds { .. }
                    | Instruction::Idle { .. }
            )
    }

    fn start_physical_task(&mut self, task: TaskId) -> Result<bool, RuntimeError> {
        self.step()?;
        let value = self.program.tasks[task as usize].clone();
        let epoch = self.states[task as usize]
            .epoch()
            .ok_or(RuntimeError::MissingTask(task))?;
        let start = self.now;
        self.emit_source(task, start)?;
        if let Some(activation) = value.activation {
            match self.bit(activation)? {
                Some(false) => {
                    if let Some(output) = value.output {
                        self.write_bit(output, Some(false), self.bit_times[activation as usize])?;
                    }
                    self.complete_task(task, epoch, start, start);
                    return Ok(true);
                }
                Some(true) => {}
                None => return Ok(false),
            }
        }
        match value.instruction {
            Instruction::Quantum(quantum) => {
                let (alternative, selected) = self.select_alternative(&quantum.alternatives)?;
                for &record in selected
                    .selected_records
                    .iter()
                    .chain(selected.excluded_records.iter())
                {
                    self.records[record as usize] = None;
                    self.record_times[record as usize] = None;
                }
                if let Some(rus) = self.task_rus_owner[task as usize]
                    && let Some(run) = self.rus.get_mut(&rus)
                {
                    run.checks.extend(selected.restarts.iter().enumerate().map(
                        |(index, parity)| EarlyCheck {
                            task,
                            alternative,
                            index,
                            parity: parity.clone(),
                            evaluated: false,
                        },
                    ));
                }
                self.artifact
                    .events
                    .push(ExecutionEvent::QuantumAlternative {
                        task,
                        alternative: u32::try_from(alternative)
                            .map_err(|_| RuntimeError::IdOverflow("alternative"))?,
                        operations: super::quantum_operation_names(&selected.stream),
                        time: start,
                    });
                self.states[task as usize] = TaskState::Running { epoch, start };
                self.physical.insert(
                    WorkKey::Task(task),
                    PhysicalRun {
                        owner: task,
                        epoch,
                        stream: selected.stream.clone(),
                        next_moment: 0,
                        event_end: None,
                        start,
                        last_end: start,
                        purpose: Purpose::Quantum {
                            detectors: selected.detectors.clone(),
                        },
                    },
                );
            }
            Instruction::MemoryRounds {
                rounds,
                stream,
                detectors,
            } => {
                self.states[task as usize] = TaskState::Running { epoch, start };
                self.physical.insert(
                    WorkKey::Task(task),
                    PhysicalRun {
                        owner: task,
                        epoch,
                        stream,
                        next_moment: 0,
                        event_end: None,
                        start,
                        last_end: start,
                        purpose: Purpose::StaticMemory { rounds, detectors },
                    },
                );
            }
            Instruction::Idle {
                duration,
                operations,
                ..
            } => {
                self.states[task as usize] = TaskState::Running { epoch, start };
                self.physical.insert(
                    WorkKey::Task(task),
                    PhysicalRun {
                        owner: task,
                        epoch,
                        stream: QuantumStream {
                            moments: vec![Moment {
                                kind: None,
                                duration,
                                operations,
                            }]
                            .into_boxed_slice(),
                        },
                        next_moment: 0,
                        event_end: None,
                        start,
                        last_end: start,
                        purpose: Purpose::TaskIdle,
                    },
                );
            }
            Instruction::WaitFor { until, memory } => {
                for cycle in &memory {
                    self.initialize_cycle(cycle)?;
                }
                self.states[task as usize] = TaskState::Running { epoch, start };
                self.waits.insert(
                    task,
                    WaitRun {
                        epoch,
                        start,
                        until,
                        memory,
                        round: 0,
                        group: None,
                        idle: false,
                    },
                );
            }
            _ => unreachable!("physical task match"),
        }
        Ok(true)
    }

    fn select_alternative<'b>(
        &self,
        alternatives: &'b [QuantumAlternative],
    ) -> Result<(usize, &'b QuantumAlternative), RuntimeError> {
        alternatives
            .iter()
            .enumerate()
            .find(|(_, alternative)| {
                alternative
                    .when
                    .iter()
                    .all(|&(bit, expected)| self.bits[bit as usize] == Some(expected))
            })
            .ok_or(RuntimeError::NoQuantumAlternative)
    }

    fn progress_physical(&mut self) -> Result<bool, RuntimeError> {
        let mut keys = self.physical.keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        let mut progress = false;
        let mut complete = Vec::new();
        for key in keys {
            let (owner, epoch, next_moment, moment) = {
                let Some(run) = self.physical.get(&key) else {
                    continue;
                };
                if run.event_end.is_some() {
                    continue;
                }
                if self.epoch_inactive(run.owner, run.epoch) {
                    complete.push(key);
                    continue;
                }
                if run.next_moment >= run.stream.moments.len() {
                    complete.push(key);
                    continue;
                }
                (
                    run.owner,
                    run.epoch,
                    run.next_moment,
                    run.stream.moments[run.next_moment].clone(),
                )
            };
            let mut qubits = moment
                .operations
                .iter()
                .flat_map(QuantumOp::qubits)
                .collect::<Vec<_>>();
            if qubits.is_empty() {
                qubits.extend(&self.program.tasks[owner as usize].qubits);
            }
            qubits.sort_unstable();
            qubits.dedup();
            if qubits
                .iter()
                .any(|qubit| self.qubit_free[*qubit as usize] > self.now + TIME_EPSILON)
            {
                continue;
            }
            let end = self.now + moment.duration;
            if !end.is_finite() {
                return Err(RuntimeError::InvalidTiming);
            }
            for &qubit in &qubits {
                self.qubit_free[qubit as usize] = end;
            }
            let run = self.physical.get_mut(&key).expect("key retained");
            run.event_end = Some(end);
            let attempt = self.attempt_of(owner, epoch);
            self.artifact.events.push(ExecutionEvent::MomentIssued {
                task: owner,
                attempt,
                moment: u32::try_from(next_moment)
                    .map_err(|_| RuntimeError::IdOverflow("moment"))?,
                start: self.now,
                end,
            });
            self.push_event(
                end,
                EventKind::Moment {
                    key,
                    epoch,
                    moment: next_moment,
                    operations: moment.operations.clone(),
                },
            );
            progress = true;
        }
        for key in complete {
            self.finish_physical(key)?;
            progress = true;
        }
        Ok(progress)
    }

    fn finish_physical(&mut self, key: WorkKey) -> Result<(), RuntimeError> {
        let Some(run) = self.physical.remove(&key) else {
            return Ok(());
        };
        let end = run.last_end.max(run.start);
        let task = run.owner;
        if self.epoch_inactive(run.owner, run.epoch) {
            if self.task_rus_owner[run.owner as usize].is_some() {
                self.states[run.owner as usize] = TaskState::Canceled { epoch: run.epoch };
            }
            return Ok(());
        }
        match run.purpose {
            Purpose::Quantum { detectors } => {
                self.evaluate_detectors(task, &detectors, end)?;
                self.complete_task(task, run.epoch, run.start, end);
            }
            Purpose::StaticMemory { rounds, detectors } => {
                self.evaluate_detectors(task, &detectors, end)?;
                if rounds > 0 {
                    let duration = (end - run.start) / f64::from(rounds);
                    for round in 0..rounds {
                        self.artifact.events.push(ExecutionEvent::MemoryRound {
                            task,
                            round,
                            memory_kind: MemoryKind::StaticPadding,
                            wait_reason: None,
                            start: run.start + f64::from(round) * duration,
                            end: run.start + f64::from(round + 1) * duration,
                        });
                    }
                }
                self.complete_task(task, run.epoch, run.start, end);
            }
            Purpose::TaskIdle => {
                self.artifact.events.push(ExecutionEvent::Idle {
                    task,
                    wait_reason: None,
                    start: run.start,
                    end,
                });
                self.complete_task(task, run.epoch, run.start, end);
            }
            Purpose::Retry { attempt } => {
                self.artifact.events.push(ExecutionEvent::RetryPrepare {
                    task,
                    attempt,
                    start: run.start,
                    end,
                });
                self.begin_rus_attempt(task, end)?;
            }
            Purpose::Cycle { group, cycle } => self.finish_cycle(group, cycle, end)?,
            Purpose::WaitIdle {
                qubits,
                wait_reason,
            } => {
                self.apply_idle_noise(&qubits, end - run.start)?;
                self.artifact.events.push(ExecutionEvent::Idle {
                    task,
                    wait_reason: Some(wait_reason),
                    start: run.start,
                    end,
                });
                if let Some(wait) = self.waits.get_mut(&task) {
                    wait.idle = false;
                }
            }
        }
        Ok(())
    }

    fn complete_event_batch(&mut self) -> Result<(), RuntimeError> {
        let mut batch = Vec::new();
        while self
            .events
            .peek()
            .is_some_and(|event| event.at <= self.now + TIME_EPSILON)
        {
            batch.push(self.events.pop().expect("peeked event"));
        }
        self.now = batch.iter().map(|event| event.at).fold(self.now, f64::max);
        batch.sort_by_key(|event| event.sequence);
        for event in batch {
            self.step()?;
            match event.kind {
                EventKind::Moment {
                    key,
                    epoch,
                    moment,
                    operations,
                } => self.complete_moment(key, epoch, moment, &operations)?,
                EventKind::Decoder {
                    task,
                    epoch,
                    key,
                    raw,
                    measured_at,
                    output,
                } => self.complete_decode(task, epoch, key, raw, measured_at, output)?,
            }
        }
        self.scan_early_restarts()?;
        Ok(())
    }

    fn event_due_now(&self) -> bool {
        self.events
            .peek()
            .is_some_and(|event| event.at <= self.now + TIME_EPSILON)
    }

    fn complete_moment(
        &mut self,
        key: WorkKey,
        epoch: u64,
        moment: usize,
        operations: &[QuantumOp],
    ) -> Result<(), RuntimeError> {
        let Some(run) = self.physical.get(&key) else {
            return Ok(());
        };
        if run.epoch != epoch || run.next_moment != moment || run.event_end.is_none() {
            return Ok(());
        }
        let owner = run.owner;
        let rng_key = self.rng_key(owner, epoch);
        let mut actor_rng = self.actor_rngs.remove(&rng_key).unwrap_or_else(|| {
            Simulator::with_seed(0, actor_seed(self.config.seed, rng_key.0, rng_key.1))
        });
        self.simulator.restore_rng_from(&actor_rng);
        for operation in operations {
            self.execute_quantum_op(owner, epoch, operation, self.now)?;
        }
        actor_rng.restore_rng_from(&self.simulator);
        self.actor_rngs.insert(rng_key, actor_rng);
        self.artifact.events.push(ExecutionEvent::MomentCompleted {
            task: owner,
            attempt: self.attempt_of(owner, epoch),
            moment: u32::try_from(moment).map_err(|_| RuntimeError::IdOverflow("moment"))?,
            time: self.now,
        });
        let run = self.physical.get_mut(&key).expect("run retained");
        run.event_end = None;
        run.next_moment += 1;
        run.last_end = self.now;
        Ok(())
    }

    fn complete_decode(
        &mut self,
        task: TaskId,
        epoch: u64,
        key: DecodeKey,
        raw: bool,
        measured_at: Option<f64>,
        port: bloq_ir::ObservableOutput,
    ) -> Result<(), RuntimeError> {
        if !matches!(self.states[task as usize], TaskState::Running { epoch: current, .. } if current == epoch)
        {
            return Ok(());
        }
        // The runtime has already waited for latency. A later query of the
        // paired result reuses the solve's original deadline and sampled policy.
        let decision = self.decoder.request(key, raw, measured_at, self.now, 0.0)?;
        if self.logged_decodes.insert(key) {
            self.artifact.decoder_decisions.push(DecoderDecisionRecord {
                task,
                decision,
                committed: self.attempt_of(task, epoch).is_none(),
            });
        }
        if let Some(output) = self.program.tasks[task as usize].output {
            self.write_bit(
                output,
                Some(match port {
                    bloq_ir::ObservableOutput::Flip => decision.flip,
                    bloq_ir::ObservableOutput::Corrected => raw ^ decision.flip,
                }),
                measured_at,
            )?;
        }
        let start = match self.states[task as usize] {
            TaskState::Running { start, .. } => start,
            _ => self.now,
        };
        self.complete_task(task, epoch, start, self.now);
        Ok(())
    }

    fn push_event(&mut self, at: f64, kind: EventKind) {
        let sequence = self.sequence;
        self.sequence += 1;
        self.events.push(ScheduledEvent { at, sequence, kind });
    }

    fn poll_regions(&mut self) -> Result<bool, RuntimeError> {
        let mut progress = false;
        let mut rus_tasks = self.rus.keys().copied().collect::<Vec<_>>();
        rus_tasks.sort_unstable();
        for task in rus_tasks {
            progress |= self.poll_rus(task)?;
        }
        Ok(progress)
    }

    fn poll_rus(&mut self, task: TaskId) -> Result<bool, RuntimeError> {
        let Some(run) = self.rus.get(&task).cloned() else {
            return Ok(false);
        };
        match run.phase {
            RusPhase::Aborting => {
                if !self.epoch_in_flight(task, run.epoch) {
                    self.drain_attempt(task, run.epoch);
                    self.reject_rus(task, self.now)?;
                    return Ok(true);
                }
            }
            RusPhase::Attempt => {
                let exits_done = run.cultivation_exits.iter().all(|exit| {
                    matches!(self.states[*exit as usize], TaskState::Done { epoch, .. } if epoch == run.epoch)
                });
                let checks_done = run.checks.iter().all(|check| check.evaluated);
                if exits_done && checks_done {
                    if run.cultivation_exits.is_empty() && run.decoder_hold.is_empty() {
                        let run = self.rus.get_mut(&task).expect("RUS retained");
                        run.phase = RusPhase::AwaitBody;
                        run.gap_ready = Some(self.now);
                    } else {
                        self.start_gap(task)?;
                    }
                    return Ok(true);
                }
            }
            RusPhase::AwaitBody => {
                if self.stream_done(&run.body, run.epoch) {
                    let restart =
                        self.eval_bool(&run.restart)?
                            .0
                            .ok_or(RuntimeError::InvalidProgram(
                                "RUS restart condition is unavailable",
                            ))?;
                    if restart {
                        self.reject_rus(task, self.now)?;
                    } else {
                        self.accept_rus(task, self.now)?;
                    }
                    return Ok(true);
                }
            }
            RusPhase::Gap | RusPhase::Preparing => {}
        }
        Ok(false)
    }

    fn scan_early_restarts(&mut self) -> Result<(), RuntimeError> {
        let mut tasks = self.rus.keys().copied().collect::<Vec<_>>();
        tasks.sort_unstable();
        for task in tasks {
            let Some(run) = self.rus.get(&task) else {
                continue;
            };
            if run.phase != RusPhase::Attempt {
                continue;
            }
            let checks = run.checks.clone();
            for (position, check) in checks.iter().enumerate() {
                if check.evaluated {
                    continue;
                }
                let Some(value) = self.available_parity(&check.parity) else {
                    continue;
                };
                if let Some(run) = self.rus.get_mut(&task) {
                    run.checks[position].evaluated = true;
                }
                if value {
                    let reason = format!(
                        "task {} alternative {} restart {}",
                        check.task, check.alternative, check.index
                    );
                    self.artifact.events.push(ExecutionEvent::EarlyRestart {
                        task,
                        attempt: self.rus[&task].attempt,
                        reason,
                        time: self.now,
                    });
                    self.abort_rus(task)?;
                    break;
                }
            }
        }
        Ok(())
    }

    fn abort_rus(&mut self, task: TaskId) -> Result<(), RuntimeError> {
        let (body, epoch) = {
            let run = self
                .rus
                .get_mut(&task)
                .ok_or(RuntimeError::MissingTask(task))?;
            run.phase = RusPhase::Aborting;
            (run.body.clone(), run.epoch)
        };
        self.cancel_stream(&body, epoch);
        Ok(())
    }

    fn cancel_stream(&mut self, stream: &Stream, epoch: u64) {
        for &task in &stream.tasks {
            match self.states[task as usize] {
                TaskState::Pending { epoch: current } if current == epoch => {
                    self.states[task as usize] = TaskState::Canceled { epoch };
                }
                TaskState::Running { epoch: current, .. }
                    if current == epoch && !self.physical.contains_key(&WorkKey::Task(task)) =>
                {
                    self.states[task as usize] = TaskState::Canceled { epoch };
                }
                _ => {}
            }
        }
    }

    fn epoch_in_flight(&self, rus: TaskId, epoch: u64) -> bool {
        self.physical.values().any(|run| {
            run.epoch == epoch
                && self.task_rus_owner[run.owner as usize] == Some(rus)
                && run.event_end.is_some()
        })
    }

    fn drain_attempt(&mut self, rus: TaskId, epoch: u64) {
        self.physical.retain(|_, run| {
            run.epoch != epoch || self.task_rus_owner[run.owner as usize] != Some(rus)
        });
        self.waits.retain(|task, run| {
            run.epoch != epoch || self.task_rus_owner[*task as usize] != Some(rus)
        });
        self.groups.retain(|_, group| {
            group.epoch != epoch || self.task_rus_owner[group.owner as usize] != Some(rus)
        });
    }

    fn start_gap(&mut self, task: TaskId) -> Result<(), RuntimeError> {
        let (epoch, cycles) = {
            let run = self
                .rus
                .get_mut(&task)
                .ok_or(RuntimeError::MissingTask(task))?;
            if run.decoder_hold.is_empty() {
                return Err(RuntimeError::InvalidProgram(
                    "factory GAP requires a physical memory cycle",
                ));
            }
            if self.program.decoder_latency_rounds == 0 {
                run.gap_ready = Some(self.now);
                run.phase = RusPhase::AwaitBody;
                return Ok(());
            }
            run.phase = RusPhase::Gap;
            (run.epoch, run.decoder_hold.clone())
        };
        let group = self.start_round_group(task, epoch, RoundKind::Gap { rus: task }, 0, cycles)?;
        self.rus.get_mut(&task).expect("RUS retained").gap_group = Some(group);
        Ok(())
    }

    fn progress_gaps(&mut self) -> Result<bool, RuntimeError> {
        let mut tasks = self.rus.keys().copied().collect::<Vec<_>>();
        tasks.sort_unstable();
        let mut progress = false;
        for task in tasks {
            let run = self.rus[&task].clone();
            if run.phase != RusPhase::Gap
                || run.gap_group.is_some()
                || run.gap_round >= self.program.decoder_latency_rounds
            {
                continue;
            }
            let group = self.start_round_group(
                task,
                run.epoch,
                RoundKind::Gap { rus: task },
                run.gap_round,
                run.decoder_hold,
            )?;
            self.rus.get_mut(&task).expect("RUS retained").gap_group = Some(group);
            progress = true;
        }
        Ok(progress)
    }

    fn reject_rus(&mut self, task: TaskId, at: f64) -> Result<(), RuntimeError> {
        let run = self
            .rus
            .get(&task)
            .cloned()
            .ok_or(RuntimeError::MissingTask(task))?;
        self.artifact.retries.push(RetryRecord {
            task,
            attempt: run.attempt,
            epoch: run.epoch,
            start: run.attempt_start,
            end: at,
            accepted: false,
        });
        self.artifact
            .events
            .push(ExecutionEvent::RusAttemptRejected {
                task,
                attempt: run.attempt,
                epoch: run.epoch,
                time: at,
            });
        if run.attempt + 1 >= self.config.limits.max_attempts {
            self.discard(task, StopReason::AttemptLimit, at);
            return Ok(());
        }
        self.clear_attempt(&run)?;
        let next_attempt = run.attempt + 1;
        let epoch = attempt_epoch(task, next_attempt);
        let has_retry_prepare = !run.retry_prepare.moments.is_empty();
        if let Some(current) = self.rus.get_mut(&task) {
            current.attempt = next_attempt;
            current.epoch = epoch;
            current.attempt_start = at;
            current.phase = RusPhase::Preparing;
            current.checks.clear();
            current.gap_round = 0;
            current.gap_group = None;
            current.gap_ready = None;
        }
        if has_retry_prepare {
            self.physical.insert(
                WorkKey::Retry(task),
                PhysicalRun {
                    owner: task,
                    epoch,
                    stream: run.retry_prepare,
                    next_moment: 0,
                    event_end: None,
                    start: at,
                    last_end: at,
                    purpose: Purpose::Retry {
                        attempt: next_attempt,
                    },
                },
            );
        } else {
            self.begin_rus_attempt(task, at)?;
        }
        Ok(())
    }

    fn begin_rus_attempt(&mut self, task: TaskId, at: f64) -> Result<(), RuntimeError> {
        let (body, epoch, attempt) = {
            let run = self
                .rus
                .get_mut(&task)
                .ok_or(RuntimeError::MissingTask(task))?;
            run.phase = RusPhase::Attempt;
            (run.body.clone(), run.epoch, run.attempt)
        };
        self.activate_stream(&body, epoch)?;
        self.artifact
            .events
            .push(ExecutionEvent::RusAttemptStarted {
                task,
                attempt,
                epoch,
                time: at,
            });
        Ok(())
    }

    fn accept_rus(&mut self, task: TaskId, at: f64) -> Result<(), RuntimeError> {
        let run = self
            .rus
            .get(&task)
            .cloned()
            .ok_or(RuntimeError::MissingTask(task))?;
        self.commit_attempt(run.epoch);
        self.artifact.retries.push(RetryRecord {
            task,
            attempt: run.attempt,
            epoch: run.epoch,
            start: run.attempt_start,
            end: at,
            accepted: true,
        });
        self.artifact
            .events
            .push(ExecutionEvent::RusAttemptAccepted {
                task,
                attempt: run.attempt,
                epoch: run.epoch,
                time: at,
            });
        if let Some(output) = self.program.tasks[task as usize].output {
            let value = self.stream_value(&run.body)?;
            let measured = run.body.value.and_then(|bit| self.bit_times[bit as usize]);
            self.write_bit(output, value, measured)?;
        }
        let children = self.binding_sources(&run.body.bindings)?;
        self.bindings
            .insert(task, Arc::new(BoundValues::Recipe(children)));
        self.complete_task(task, run.outer_epoch, run.start, at);
        self.rus.remove(&task);
        Ok(())
    }

    fn clear_attempt(&mut self, run: &RusRun) -> Result<(), RuntimeError> {
        for &bit in &run.attempt_bits {
            self.write_bit(bit, None, None)?;
        }
        for &record in &run.attempt_records {
            self.records[record as usize] = None;
            self.record_times[record as usize] = None;
        }
        for cycle in &run.decoder_hold {
            for initializer in &cycle.initializers {
                self.records[initializer.record as usize] = None;
                self.record_times[initializer.record as usize] = None;
            }
        }
        self.cancel_stream(&run.body, run.epoch);
        Ok(())
    }

    fn progress_waits(&mut self) -> Result<bool, RuntimeError> {
        let mut tasks = self.waits.keys().copied().collect::<Vec<_>>();
        tasks.sort_unstable();
        let mut progress = false;
        for task in tasks {
            let run = self.waits[&task].clone();
            if run.group.is_some() || run.idle {
                continue;
            }
            if run
                .until
                .iter()
                .all(|until| self.states[*until as usize].end().is_some())
            {
                self.artifact.events.push(ExecutionEvent::JoinReleased {
                    task,
                    time: self.now,
                });
                self.complete_task(task, run.epoch, run.start, self.now);
                self.waits.remove(&task);
                progress = true;
                continue;
            }
            if run.memory.is_empty() {
                return Err(RuntimeError::InvalidProgram(
                    "dynamic wait has no memory cycle",
                ));
            }
            let round_duration = run
                .memory
                .iter()
                .map(|cycle| cycle.round_duration)
                .fold(0.0, f64::max);
            let known = run.until.iter().try_fold(self.now, |target, until| {
                Some(target.max(self.known_completion(*until)?))
            });
            if let Some(target) = known
                && target > self.now + TIME_EPSILON
                && target - self.now + TIME_EPSILON < round_duration
            {
                let wait_reason = self.classify_wait(task);
                self.start_wait_idle(task, run.epoch, target - self.now, wait_reason);
                self.waits.get_mut(&task).expect("wait retained").idle = true;
                progress = true;
                continue;
            }
            let group = self.start_round_group(
                task,
                run.epoch,
                RoundKind::Wait { task },
                run.round,
                run.memory.clone(),
            )?;
            self.waits.get_mut(&task).expect("wait retained").group = Some(group);
            progress = true;
        }
        Ok(progress)
    }

    fn known_completion(&self, task: TaskId) -> Option<f64> {
        self.known_completion_inner(task, &mut HashSet::new())
    }

    fn classify_wait(&self, task: TaskId) -> WaitReason {
        let Some(run) = self.waits.get(&task) else {
            return WaitReason::Synchronization;
        };
        let unresolved = run
            .until
            .iter()
            .copied()
            .filter(|until| self.states[*until as usize].end().is_none())
            .collect::<Vec<_>>();
        if unresolved
            .iter()
            .any(|until| self.is_direct_synchronization(*until))
        {
            return WaitReason::Synchronization;
        }
        let mut blockers = CausalBlockers::default();
        let mut visiting = HashSet::new();
        for blocker in unresolved {
            self.collect_causal_blockers(blocker, &mut visiting, &mut blockers);
        }
        if blockers.causal_cut {
            WaitReason::CausalCut
        } else if blockers.decoder_latency {
            WaitReason::DecoderLatency
        } else {
            WaitReason::Synchronization
        }
    }

    fn is_direct_synchronization(&self, task: TaskId) -> bool {
        matches!(
            self.program.tasks[task as usize].instruction,
            Instruction::Quantum(_)
                | Instruction::Rus { .. }
                | Instruction::WaitFor { .. }
                | Instruction::SignalReady { .. }
                | Instruction::MemoryRounds { .. }
                | Instruction::Idle { .. }
        )
    }

    fn collect_causal_blockers(
        &self,
        task: TaskId,
        visiting: &mut HashSet<TaskId>,
        blockers: &mut CausalBlockers,
    ) {
        if self.states[task as usize].end().is_some() || !visiting.insert(task) {
            return;
        }
        if self.scheduled_decoder_deadline(task).is_some() {
            blockers.decoder_latency = true;
            visiting.remove(&task);
            return;
        }
        let value = &self.program.tasks[task as usize];
        let mut descended = false;
        if matches!(
            value.instruction,
            Instruction::Eval(_)
                | Instruction::Accumulate(_)
                | Instruction::Bind(_)
                | Instruction::ReadoutRecipe { .. }
                | Instruction::Observable { .. }
                | Instruction::SignalReady { .. }
        ) {
            for dependency in value
                .dependencies
                .iter()
                .copied()
                .filter(|dependency| self.states[*dependency as usize].end().is_none())
            {
                descended = true;
                self.collect_causal_blockers(dependency, visiting, blockers);
            }
        }
        if !descended {
            blockers.causal_cut = true;
        }
        visiting.remove(&task);
    }

    fn scheduled_decoder_deadline(&self, task: TaskId) -> Option<f64> {
        let TaskState::Running { epoch, .. } = self.states[task as usize] else {
            return None;
        };
        self.events
            .iter()
            .filter_map(|event| match event.kind {
                EventKind::Decoder {
                    task: event_task,
                    epoch: event_epoch,
                    ..
                } if event_task == task && event_epoch == epoch => Some(event.at),
                _ => None,
            })
            .min_by(f64::total_cmp)
    }

    fn known_completion_inner(&self, task: TaskId, visiting: &mut HashSet<TaskId>) -> Option<f64> {
        if !visiting.insert(task) {
            return None;
        }
        let result = (|| {
            let state = *self.states.get(task as usize)?;
            if let Some(end) = state.end() {
                return Some(end);
            }
            if let Some(run) = self.physical.get(&WorkKey::Task(task))
                && let Some(current) = run.event_end
            {
                return Some(
                    run.stream.moments[run.next_moment + 1..]
                        .iter()
                        .map(|moment| moment.duration)
                        .sum::<f64>()
                        + current,
                );
            }
            if let Some(deadline) = self.scheduled_decoder_deadline(task) {
                return Some(deadline);
            }
            let TaskState::Pending { .. } = state else {
                return None;
            };
            let value = &self.program.tasks[task as usize];
            let mut start = value.release;
            for dependency in &value.dependencies {
                start = start.max(self.known_completion_inner(*dependency, visiting)?);
            }
            if matches!(
                value.instruction,
                Instruction::Eval(_)
                    | Instruction::Accumulate(_)
                    | Instruction::Bind(_)
                    | Instruction::ReadoutRecipe { .. }
                    | Instruction::Observable { .. }
                    | Instruction::SignalReady { .. }
            ) {
                return Some(start);
            }
            value.source?;
            if let Some(activation) = value.activation
                && !self.bits[activation as usize]?
            {
                return Some(start);
            }
            let duration = match &value.instruction {
                Instruction::Quantum(quantum) => self
                    .select_alternative(&quantum.alternatives)
                    .ok()?
                    .1
                    .stream
                    .moments
                    .iter()
                    .map(|moment| moment.duration)
                    .sum(),
                Instruction::MemoryRounds { stream, .. } => {
                    stream.moments.iter().map(|moment| moment.duration).sum()
                }
                Instruction::Idle { duration, .. } => *duration,
                _ => return None,
            };
            let end = start + duration;
            end.is_finite().then_some(end)
        })();
        visiting.remove(&task);
        result
    }

    fn start_wait_idle(
        &mut self,
        task: TaskId,
        epoch: u64,
        duration: f64,
        wait_reason: WaitReason,
    ) {
        let qubits = self.program.tasks[task as usize].qubits.clone();
        self.physical.insert(
            WorkKey::WaitIdle(task),
            PhysicalRun {
                owner: task,
                epoch,
                stream: QuantumStream {
                    moments: vec![Moment {
                        kind: None,
                        duration,
                        operations: Box::default(),
                    }]
                    .into_boxed_slice(),
                },
                next_moment: 0,
                event_end: None,
                start: self.now,
                last_end: self.now,
                purpose: Purpose::WaitIdle {
                    qubits,
                    wait_reason,
                },
            },
        );
    }

    fn start_round_group(
        &mut self,
        owner: TaskId,
        epoch: u64,
        kind: RoundKind,
        round: u32,
        cycles: Box<[MemoryCycle]>,
    ) -> Result<u64, RuntimeError> {
        if cycles.is_empty() {
            return Err(RuntimeError::InvalidProgram("memory round has no cycle"));
        }
        let id = self.next_group;
        self.next_group += 1;
        let mut snapshots = Vec::with_capacity(cycles.len());
        for cycle in &cycles {
            self.initialize_cycle(cycle)?;
            snapshots.push(CycleSnapshot {
                previous: cycle
                    .boundary_flows
                    .iter()
                    .map(|flow| {
                        flow.frontier_record.and_then(|record| {
                            self.records[record as usize].map(|value| (record, value))
                        })
                    })
                    .collect(),
            });
        }
        let group = RoundGroup {
            owner,
            epoch,
            kind,
            round,
            start: self.now,
            end: self.now,
            remaining: cycles.len(),
            cycles: cycles.clone(),
            snapshots,
            wait_reason: match kind {
                RoundKind::Gap { .. } => None,
                RoundKind::Wait { task } => Some(self.classify_wait(task)),
            },
        };
        self.groups.insert(id, group);
        for (cycle, memory) in cycles.into_vec().into_iter().enumerate() {
            self.physical.insert(
                WorkKey::Cycle(id, cycle),
                PhysicalRun {
                    owner,
                    epoch,
                    stream: memory.stream,
                    next_moment: 0,
                    event_end: None,
                    start: self.now,
                    last_end: self.now,
                    purpose: Purpose::Cycle { group: id, cycle },
                },
            );
        }
        Ok(id)
    }

    fn finish_cycle(&mut self, group: u64, cycle: usize, end: f64) -> Result<(), RuntimeError> {
        let (owner, epoch, memory, snapshot) = {
            let value = self
                .groups
                .get(&group)
                .ok_or(RuntimeError::InvalidProgram("memory group completed twice"))?;
            (
                value.owner,
                value.epoch,
                value.cycles[cycle].clone(),
                value.snapshots[cycle].clone(),
            )
        };
        self.evaluate_detectors(owner, &memory.detectors, end)?;
        for (flow, previous) in memory.boundary_flows.iter().zip(snapshot.previous) {
            let Some((record, previous)) = previous else {
                continue;
            };
            let current = self
                .available_parity(&flow.parity)
                .ok_or(RuntimeError::MissingRecord(record))?;
            self.artifact.detectors.push(DetectorResult {
                sequence: u32::try_from(self.artifact.detectors.len())
                    .map_err(|_| RuntimeError::IdOverflow("detector"))?,
                task: owner,
                value: previous ^ current,
                attempt: self.attempt_of(owner, epoch),
                time: end,
                committed: self.attempt_of(owner, epoch).is_none(),
            });
            self.records[record as usize] = Some(current);
            self.record_times[record as usize] = Some(end);
        }
        let complete = {
            let value = self.groups.get_mut(&group).expect("group retained");
            value.remaining -= 1;
            value.end = value.end.max(end);
            value.remaining == 0
        };
        if complete {
            let value = self.groups.remove(&group).expect("group retained");
            self.artifact.events.push(ExecutionEvent::MemoryRound {
                task: value.owner,
                round: value.round,
                memory_kind: match value.kind {
                    RoundKind::Gap { .. } => MemoryKind::FactoryGap,
                    RoundKind::Wait { .. } => MemoryKind::DynamicWait,
                },
                wait_reason: value.wait_reason,
                start: value.start,
                end: value.end,
            });
            match value.kind {
                RoundKind::Gap { rus } => {
                    let run = self
                        .rus
                        .get_mut(&rus)
                        .ok_or(RuntimeError::MissingTask(rus))?;
                    run.gap_round += 1;
                    run.gap_group = None;
                    if run.gap_round == self.program.decoder_latency_rounds {
                        run.gap_ready = Some(value.end);
                        run.phase = RusPhase::AwaitBody;
                    }
                }
                RoundKind::Wait { task } => {
                    let run = self
                        .waits
                        .get_mut(&task)
                        .ok_or(RuntimeError::MissingTask(task))?;
                    run.round = run
                        .round
                        .checked_add(1)
                        .ok_or(RuntimeError::IdOverflow("memory round"))?;
                    run.group = None;
                }
            }
        }
        Ok(())
    }

    fn initialize_cycle(&mut self, cycle: &MemoryCycle) -> Result<(), RuntimeError> {
        for initializer in &cycle.initializers {
            if self.records[initializer.record as usize].is_some() {
                continue;
            }
            let value = self
                .available_parity(&initializer.parity)
                .ok_or(RuntimeError::MissingRecord(initializer.record))?;
            self.records[initializer.record as usize] = Some(value);
            self.record_times[initializer.record as usize] =
                self.parity_time(&initializer.parity)?;
        }
        Ok(())
    }

    fn complete_task(&mut self, task: TaskId, epoch: u64, start: f64, end: f64) {
        self.states[task as usize] = TaskState::Done { epoch, end };
        self.artifact.timing.push(TaskTiming {
            task,
            label: self.program.tasks[task as usize].label.clone(),
            instruction: instruction_kind(&self.program.tasks[task as usize].instruction).into(),
            attempt: self.attempt_of(task, epoch),
            start,
            end,
        });
    }

    fn emit_source(&mut self, task: TaskId, time: f64) -> Result<(), RuntimeError> {
        let Some(role) = self.program.tasks[task as usize].source else {
            return Ok(());
        };
        if !self.source_released.insert(task) {
            return Ok(());
        }
        self.artifact.events.push(ExecutionEvent::SourceReleased {
            task,
            role: format!("{role:?}"),
            time,
        });
        if role == SourceRole::LogicalInput {
            self.prepare_task_input(task, time)?;
        }
        Ok(())
    }

    fn prepare_task_input(&mut self, task: TaskId, time: f64) -> Result<(), RuntimeError> {
        let input = self
            .program
            .inputs
            .iter()
            .find(|input| input.task == task)
            .cloned();
        if let Some(input) = input {
            self.prepare_input(&input, time)?;
        }
        Ok(())
    }

    fn prepare_input(&mut self, input: &LogicalInput, time: f64) -> Result<(), RuntimeError> {
        let rng_key = (input.task, 0);
        let mut actor_rng = self.actor_rngs.remove(&rng_key).unwrap_or_else(|| {
            Simulator::with_seed(0, actor_seed(self.config.seed, rng_key.0, rng_key.1))
        });
        self.simulator.restore_rng_from(&actor_rng);
        for &qubit in &input.data_qubits {
            self.simulator.reset(qubit as usize)?;
        }
        for stabilizer in &input.stabilizers {
            self.simulator.postselect_observable(
                &engine_pauli_product(stabilizer, self.program.qubit_count as usize)?,
                false,
            )?;
        }
        let logical = match self.config.input_state {
            LogicalInputState::Plus => &input.x,
            LogicalInputState::Zero => &input.z,
        };
        self.simulator.postselect_observable(
            &engine_pauli_product(logical, self.program.qubit_count as usize)?,
            false,
        )?;
        actor_rng.restore_rng_from(&self.simulator);
        self.actor_rngs.insert(rng_key, actor_rng);
        self.peak_rank = self.peak_rank.max(self.simulator.rank());
        self.artifact.events.push(ExecutionEvent::InputArrival {
            task: input.task,
            port: input.port,
            state: self.config.input_state,
            time,
        });
        Ok(())
    }

    fn execute_quantum_op(
        &mut self,
        task: TaskId,
        epoch: u64,
        operation: &QuantumOp,
        time: f64,
    ) -> Result<(), RuntimeError> {
        match operation {
            QuantumOp::Gate1 { gate, qubit } => self.apply(&BackendInstruction::Gate1 {
                gate: gate1(*gate),
                qubit: *qubit as usize,
            })?,
            QuantumOp::Gate2 {
                control_basis,
                target_basis,
                control,
                target,
            } => self.apply(&BackendInstruction::Gate2 {
                control: basis(*control_basis),
                target: basis(*target_basis),
                control_qubit: *control as usize,
                target_qubit: *target as usize,
            })?,
            QuantumOp::Pauli { basis: axis, qubit } => self.apply(&BackendInstruction::Pauli {
                basis: basis(*axis),
                qubit: *qubit as usize,
            })?,
            QuantumOp::T {
                basis: axis,
                qubit,
                adjoint,
            } => self.apply(&BackendInstruction::T {
                basis: basis(*axis),
                qubit: *qubit as usize,
                adjoint: *adjoint,
            })?,
            QuantumOp::Measure {
                observable,
                records,
                flip_probability,
            } => {
                let outcome =
                    self.simulator
                        .apply_batch(&[BackendInstruction::MeasureWithReadoutError {
                            observable: engine_pauli_product(
                                observable,
                                self.program.qubit_count as usize,
                            )?,
                            probability: *flip_probability,
                        }])?;
                self.peak_rank = self
                    .peak_rank
                    .max(outcome.max_rank)
                    .max(self.simulator.rank());
                let value = outcome.records[0].outcome;
                for &record in records {
                    self.records[record as usize] = Some(value);
                    self.record_times[record as usize] = Some(time);
                    self.artifact.measurements.push(MeasurementRecord {
                        record,
                        value,
                        task,
                        attempt: self.attempt_of(task, epoch),
                        time,
                        committed: self.attempt_of(task, epoch).is_none(),
                    });
                }
            }
            QuantumOp::Reset { basis: axis, qubit } => self.apply(&BackendInstruction::Reset {
                basis: basis(*axis),
                qubit: *qubit as usize,
            })?,
            QuantumOp::ConditionalPauli {
                basis: axis,
                qubit,
                control,
            } => {
                let applied = self.record(*control)?;
                if applied {
                    self.apply(&BackendInstruction::Pauli {
                        basis: basis(*axis),
                        qubit: *qubit as usize,
                    })?;
                }
                self.artifact
                    .events
                    .push(ExecutionEvent::ConditionalCorrection {
                        task,
                        record: *control,
                        pauli: format!("{axis:?}"),
                        applied,
                        time,
                    });
            }
            QuantumOp::Depolarize1 {
                probability,
                qubits,
            } => {
                for &qubit in qubits {
                    self.random_single(qubit, *probability, &[Pauli::X, Pauli::Y, Pauli::Z])?;
                }
            }
            QuantumOp::Depolarize2 { probability, pairs } => {
                for &(first, second) in pairs {
                    self.random_two(first, second, *probability)?;
                }
            }
            QuantumOp::PauliError {
                probability,
                basis: axis,
                qubits,
            } => {
                for &qubit in qubits {
                    self.random_single(qubit, *probability, &[*axis])?;
                }
            }
        }
        Ok(())
    }

    fn apply(&mut self, operation: &BackendInstruction) -> Result<(), RuntimeError> {
        let outcome = self
            .simulator
            .apply_batch(std::slice::from_ref(operation))?;
        self.peak_rank = self
            .peak_rank
            .max(outcome.max_rank)
            .max(self.simulator.rank());
        Ok(())
    }

    fn random_single(
        &mut self,
        qubit: u32,
        probability: f64,
        axes: &[Pauli],
    ) -> Result<(), RuntimeError> {
        let alternatives = axes
            .iter()
            .map(|axis| {
                crate::backend::PauliString::single(
                    self.program.qubit_count as usize,
                    qubit as usize,
                    super::backend_pauli(*axis),
                )
            })
            .collect::<Vec<_>>();
        self.apply(&BackendInstruction::RandomPauli {
            probabilities: vec![probability / axes.len() as f64; axes.len()],
            alternatives,
            heralded: false,
        })
    }

    fn random_two(
        &mut self,
        first: u32,
        second: u32,
        probability: f64,
    ) -> Result<(), RuntimeError> {
        let axes = [None, Some(Pauli::X), Some(Pauli::Y), Some(Pauli::Z)];
        let mut alternatives = Vec::with_capacity(15);
        for left in axes {
            for right in axes {
                if left.is_none() && right.is_none() {
                    continue;
                }
                let mut product = PauliString::new(self.program.qubit_count as usize);
                if let Some(axis) = left {
                    product.set(first as usize, super::backend_pauli(axis));
                }
                if let Some(axis) = right {
                    product.set(second as usize, super::backend_pauli(axis));
                }
                alternatives.push(product);
            }
        }
        self.apply(&BackendInstruction::RandomPauli {
            probabilities: vec![probability / 15.0; 15],
            alternatives,
            heralded: false,
        })
    }

    fn apply_idle_noise(&mut self, qubits: &[u32], ratio: f64) -> Result<(), RuntimeError> {
        let probability = (ratio * self.config.idle_error_rate).min(MAX_DEPOLARIZE1_PROBABILITY);
        for &qubit in qubits {
            self.random_single(qubit, probability, &[Pauli::X, Pauli::Y, Pauli::Z])?;
        }
        Ok(())
    }

    fn evaluate_detectors(
        &mut self,
        task: TaskId,
        detectors: &[RecordParity],
        time: f64,
    ) -> Result<(), RuntimeError> {
        let epoch = self.states[task as usize].epoch().unwrap_or(0);
        for detector in detectors {
            let value = self
                .available_parity(detector)
                .ok_or_else(|| first_missing_record(detector, &self.records))?;
            self.artifact.detectors.push(DetectorResult {
                sequence: u32::try_from(self.artifact.detectors.len())
                    .map_err(|_| RuntimeError::IdOverflow("detector"))?,
                task,
                value,
                attempt: self.attempt_of(task, epoch),
                time,
                committed: self.attempt_of(task, epoch).is_none(),
            });
        }
        Ok(())
    }

    fn available_parity(&self, parity: &RecordParity) -> Option<bool> {
        let mut value = parity.constant;
        for &record in parity.records.iter() {
            let bit = self.records.get(record as usize).copied().flatten()?;
            value ^= bit;
        }
        Some(value)
    }

    fn parity_time(&self, parity: &RecordParity) -> Result<Option<f64>, RuntimeError> {
        let mut time = None;
        for &record in parity.records.iter() {
            if self.records.get(record as usize).is_none() {
                return Err(RuntimeError::MissingRecord(record));
            }
            time = max_time(time, self.record_times[record as usize]);
        }
        Ok(time)
    }

    fn eval_bool(&self, expression: &BoolOp) -> Result<(Option<bool>, Option<f64>), RuntimeError> {
        self.eval_bool_using(expression, &|bit| {
            Ok((self.bit(bit)?, self.bit_times[bit as usize]))
        })
    }

    fn eval_bool_using(
        &self,
        expression: &BoolOp,
        input: &dyn Fn(BitId) -> Result<(Option<bool>, Option<f64>), RuntimeError>,
    ) -> Result<(Option<bool>, Option<f64>), RuntimeError> {
        match expression {
            BoolOp::Call { body, inputs } => {
                let mut available = true;
                let mut time = None;
                for &bit in inputs {
                    let (value, measured) = input(bit)?;
                    available &= value.is_some();
                    time = max_time(time, measured);
                }
                let (value, _) = self.eval_bool_using(body, &|slot| {
                    let bit = inputs
                        .get(slot as usize)
                        .ok_or(RuntimeError::InvalidProgram("missing function argument"))?;
                    input(*bit)
                })?;
                Ok((value.filter(|_| available), time))
            }
            BoolOp::Const(value) => Ok((Some(*value), None)),
            BoolOp::Copy(bit) => input(*bit),
            BoolOp::Not(inner) => {
                let (value, time) = self.eval_bool_using(inner, input)?;
                Ok((value.map(|value| !value), time))
            }
            BoolOp::Parity { inputs, constant } => {
                let mut value = Some(*constant);
                let mut time = None;
                for &bit in inputs {
                    let (operand, measured) = input(bit)?;
                    value = match (value, operand) {
                        (Some(left), Some(right)) => Some(left ^ right),
                        _ => None,
                    };
                    time = max_time(time, measured);
                }
                Ok((value, time))
            }
            BoolOp::Xor(operands) => self.eval_operands(operands, input, false, |a, b| a ^ b),
            BoolOp::And(operands) => self.eval_operands(operands, input, true, |a, b| a & b),
            BoolOp::Or(operands) => self.eval_operands(operands, input, false, |a, b| a | b),
            BoolOp::Select {
                condition,
                when_false,
                when_true,
            } => {
                let condition = self.eval_bool_using(condition, input)?;
                let when_false = self.eval_bool_using(when_false, input)?;
                let when_true = self.eval_bool_using(when_true, input)?;
                Ok((
                    match (condition.0, when_false.0, when_true.0) {
                        (Some(condition), Some(left), Some(right)) => {
                            Some(if condition { right } else { left })
                        }
                        _ => None,
                    },
                    max_time(condition.1, max_time(when_false.1, when_true.1)),
                ))
            }
        }
    }

    fn eval_operands(
        &self,
        operands: &[BoolOp],
        input: &dyn Fn(BitId) -> Result<(Option<bool>, Option<f64>), RuntimeError>,
        identity: bool,
        combine: impl Fn(bool, bool) -> bool,
    ) -> Result<(Option<bool>, Option<f64>), RuntimeError> {
        let mut value = Some(identity);
        let mut time = None;
        for operand in operands {
            let operand = self.eval_bool_using(operand, input)?;
            value = match (value, operand.0) {
                (Some(left), Some(right)) => Some(combine(left, right)),
                _ => None,
            };
            time = max_time(time, operand.1);
        }
        Ok((value, time))
    }

    fn bit(&self, bit: BitId) -> Result<Option<bool>, RuntimeError> {
        self.bits
            .get(bit as usize)
            .copied()
            .ok_or(RuntimeError::MissingBit(bit))
    }

    fn write_bit(
        &mut self,
        bit: BitId,
        value: Option<bool>,
        time: Option<f64>,
    ) -> Result<(), RuntimeError> {
        *self
            .bits
            .get_mut(bit as usize)
            .ok_or(RuntimeError::MissingBit(bit))? = value;
        self.bit_times[bit as usize] = time;
        Ok(())
    }

    fn record(&self, record: RecordId) -> Result<bool, RuntimeError> {
        self.records
            .get(record as usize)
            .copied()
            .flatten()
            .ok_or(RuntimeError::MissingRecord(record))
    }

    fn attempt_of(&self, task: TaskId, epoch: u64) -> Option<u64> {
        self.attempt_owner(task, epoch).map(|_| epoch)
    }

    fn rng_key(&self, task: TaskId, epoch: u64) -> (TaskId, u64) {
        (self.attempt_owner(task, epoch).unwrap_or(task), epoch)
    }

    fn attempt_owner(&self, task: TaskId, epoch: u64) -> Option<TaskId> {
        self.task_rus_owner[task as usize].or_else(|| {
            self.rus
                .get(&task)
                .is_some_and(|run| run.epoch == epoch)
                .then_some(task)
        })
    }

    fn epoch_inactive(&self, task: TaskId, epoch: u64) -> bool {
        if let Some(rus) = self.task_rus_owner[task as usize] {
            return self
                .rus
                .get(&rus)
                .is_none_or(|run| run.epoch != epoch || run.phase == RusPhase::Aborting);
        }
        self.rus
            .get(&task)
            .is_some_and(|run| run.epoch == epoch && run.phase == RusPhase::Aborting)
    }

    fn stream_done(&self, stream: &Stream, epoch: u64) -> bool {
        stream.tasks.iter().all(|task| {
            matches!(self.states[*task as usize], TaskState::Done { epoch: current, .. } if current == epoch)
        })
    }

    fn stream_value(&self, stream: &Stream) -> Result<Option<bool>, RuntimeError> {
        stream.value.map_or(Ok(Some(false)), |bit| self.bit(bit))
    }

    fn binding_sources(&self, tasks: &[TaskId]) -> Result<Box<[Arc<BoundValues>]>, RuntimeError> {
        tasks
            .iter()
            .map(|task| {
                self.bindings
                    .get(task)
                    .cloned()
                    .ok_or(RuntimeError::MissingTask(*task))
            })
            .collect()
    }

    fn next_ready_release(&self) -> Option<f64> {
        self.program
            .tasks
            .iter()
            .enumerate()
            .filter_map(|(index, task)| {
                matches!(self.states[index], TaskState::Pending { .. })
                    .then_some(task.release)
                    .filter(|release| *release > self.now + TIME_EPSILON)
            })
            .min_by(f64::total_cmp)
    }

    fn all_entry_done(&self) -> bool {
        self.program
            .entry
            .tasks
            .iter()
            .all(|task| self.states[*task as usize].end().is_some())
    }

    fn commit_attempt(&mut self, epoch: u64) {
        for value in &mut self.artifact.measurements {
            if value.attempt == Some(epoch) {
                value.committed = true;
            }
        }
        for value in &mut self.artifact.detectors {
            if value.attempt == Some(epoch) {
                value.committed = true;
            }
        }
        for value in &mut self.artifact.observables {
            if value.attempt == Some(epoch) {
                value.committed = true;
            }
        }
        for value in &mut self.artifact.decoder_decisions {
            if value.decision.key.attempt == epoch {
                value.committed = true;
            }
        }
    }

    fn discard(&mut self, task: TaskId, reason: StopReason, time: f64) {
        self.artifact.discarded = true;
        self.artifact.stop_reason = Some(reason);
        self.stop_at = Some(time);
        self.artifact
            .events
            .push(ExecutionEvent::Discarded { task, reason, time });
    }

    fn finish_reports(&mut self) {
        self.artifact.finished_at = self.stop_at.unwrap_or(self.now);
        self.artifact.peak_rank = self.peak_rank;
        if self.artifact.discarded {
            self.artifact.final_bits = vec![None; self.bits.len()];
            for value in &mut self.artifact.measurements {
                value.committed = false;
            }
            for value in &mut self.artifact.detectors {
                value.committed = false;
            }
            for value in &mut self.artifact.observables {
                value.committed = false;
            }
            for value in &mut self.artifact.decoder_decisions {
                value.committed = false;
            }
            return;
        }
        self.artifact.final_bits.clone_from(&self.bits);
    }
}

fn attempt_epoch(task: TaskId, attempt: u32) -> u64 {
    (u64::from(task) << 32) | (u64::from(attempt) + 1)
}

fn actor_seed(seed: u64, actor: TaskId, epoch: u64) -> u64 {
    seed ^ u64::from(actor).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ epoch.wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

fn first_missing_record(parity: &RecordParity, records: &[Option<bool>]) -> RuntimeError {
    let record = parity
        .records
        .iter()
        .copied()
        .find(|record| records.get(*record as usize).is_none_or(Option::is_none))
        .expect("an unavailable parity has an unavailable record");
    RuntimeError::MissingRecord(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_boolean_calls_keep_strict_inputs_timestamps_and_fresh_values() {
        let program = Program {
            bit_count: 3,
            ..Program::default()
        };
        let mut machine = Machine::new(&program, RuntimeConfig::default()).unwrap();
        machine.bit_times = vec![Some(2.0), Some(11.0), Some(5.0)];
        let body = Arc::new(BoolOp::Select {
            condition: Box::new(BoolOp::Copy(0)),
            when_false: Box::new(BoolOp::Copy(1)),
            when_true: Box::new(BoolOp::Copy(2)),
        });
        let call = BoolOp::Call {
            body,
            inputs: Box::new([2, 0, 1]),
        };
        let values = [None, Some(false), Some(true)];
        for condition in values {
            for when_false in values {
                for when_true in values {
                    machine.bits = vec![when_false, when_true, condition];
                    let expected = match (condition, when_false, when_true) {
                        (Some(c), Some(a), Some(b)) => Some(if c { b } else { a }),
                        _ => None,
                    };
                    assert_eq!(machine.eval_bool(&call).unwrap(), (expected, Some(11.0)));
                }
            }
        }
        let parity = BoolOp::Call {
            body: Arc::new(BoolOp::Parity {
                inputs: Box::new([0, 0]),
                constant: true,
            }),
            inputs: Box::new([1]),
        };
        for value in values {
            machine.bits[1] = value;
            assert_eq!(
                machine.eval_bool(&parity).unwrap(),
                (value.map(|_| true), Some(11.0))
            );
        }
        let nested = BoolOp::Call {
            body: Arc::new(parity),
            inputs: Box::new([2, 0]),
        };
        machine.bits[0] = None;
        assert_eq!(machine.eval_bool(&nested).unwrap(), (None, Some(5.0)));
        let unused = BoolOp::Call {
            body: Arc::new(BoolOp::Const(false)),
            inputs: Box::new([0]),
        };
        assert_eq!(machine.eval_bool(&unused).unwrap(), (None, Some(2.0)));
        machine.bits[0] = Some(true);
        assert_eq!(
            machine.eval_bool(&unused).unwrap(),
            (Some(false), Some(2.0))
        );
    }

    #[test]
    fn runtime_register_capacity_failure_is_typed_before_initialization() {
        assert!(matches!(
            filled_register(usize::MAX, None::<f64>, "bit timestamps"),
            Err(RuntimeError::RegisterAllocation {
                register: "bit timestamps",
                ..
            })
        ));
        assert_eq!(
            filled_register(3, None::<f64>, "bit timestamps").unwrap(),
            [None; 3]
        );
    }

    #[test]
    fn dynamic_wait_round_rejects_exhaustion_without_wrapping() {
        let program = Program {
            tasks: vec![crate::instruction::Task {
                label: "wait".into(),
                release: 0.0,
                source: None,
                dependencies: Box::default(),
                activation: None,
                output: None,
                qubits: Box::default(),
                duration: 0.0,
                instruction: Instruction::WaitFor {
                    until: Box::default(),
                    memory: Box::default(),
                },
            }],
            ..Default::default()
        };
        let mut machine = Machine::new(&program, RuntimeConfig::default()).unwrap();
        machine.waits.insert(
            0,
            WaitRun {
                epoch: 0,
                start: 0.0,
                until: Box::default(),
                memory: Box::default(),
                round: u32::MAX,
                group: Some(0),
                idle: false,
            },
        );
        machine.groups.insert(
            0,
            RoundGroup {
                owner: 0,
                epoch: 0,
                kind: RoundKind::Wait { task: 0 },
                round: u32::MAX,
                start: 0.0,
                end: 0.0,
                remaining: 1,
                cycles: vec![MemoryCycle::default()].into_boxed_slice(),
                snapshots: vec![CycleSnapshot {
                    previous: Vec::new(),
                }],
                wait_reason: Some(WaitReason::Synchronization),
            },
        );
        assert!(matches!(
            machine.finish_cycle(0, 0, 1.0),
            Err(RuntimeError::IdOverflow("memory round"))
        ));
        assert_eq!(machine.waits[&0].round, u32::MAX);
    }

    #[test]
    fn step_limit_rejects_at_u64_max_without_wrapping() {
        let program = Program::default();
        let mut machine = Machine::new(&program, RuntimeConfig::default()).unwrap();
        machine.config.limits.max_steps = u64::MAX;
        machine.steps = u64::MAX - 1;
        machine.step().unwrap();
        assert!(matches!(
            machine.step(),
            Err(RuntimeError::StepLimit { limit: u64::MAX })
        ));
        assert_eq!(machine.steps, u64::MAX);
    }

    #[test]
    fn input_only_diamond_has_no_wide_identity_products() {
        let input = Arc::new(BoundValues::Owned(vec![BoundaryBinding {
            input: true,
            operator: Default::default(),
        }]));
        let empty = Arc::new(BoundValues::Owned(Vec::new()));
        let mut root = Arc::new(BoundValues::Recipe(vec![input, empty].into_boxed_slice()));
        for _ in 0..24 {
            root = Arc::new(BoundValues::Recipe(
                vec![root.clone(), root].into_boxed_slice(),
            ));
        }
        let mut products = HashMap::new();
        binding_product(&root, 4096, &mut products).unwrap();
        assert_eq!(products.len(), 27);
        assert!(products.values().all(Option::is_none));
    }
}
