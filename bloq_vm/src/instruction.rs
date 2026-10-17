//! Backend-independent instructions executed by the Bloq VM.
//!
//! The types in this module deliberately contain no Bloq IR node, graph, or
//! expression types. [`crate::lower()`] resolves those into dense identifiers,
//! executable quantum streams, and structured control flow once up front.
#![allow(
    missing_docs,
    reason = "the ISA containers and gate vocabulary are documented at their owning types"
)]

/// Dense task identifier in [`Program::tasks`].
pub type TaskId = u32;
/// Dense classical bit register.
pub type BitId = u32;
/// Dense simulator qubit.
pub type QubitId = u32;
/// Dense measurement record.
pub type RecordId = u32;
/// Runtime readiness signal, normally produced by an accepted RUS region.
pub type ResourceId = u32;

/// Default mock streaming-decoder latency, in physical memory rounds.
pub const DEFAULT_DECODER_LATENCY_ROUNDS: u32 = 10;
/// A lowered executable program.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Program {
    /// Number of simulator qubits.
    pub qubit_count: u32,
    /// Number of classical bit registers.
    pub bit_count: u32,
    /// Number of statically named measurement records.
    pub record_count: u32,
    /// All tasks, addressed by [`TaskId`]. Nested streams index this same arena.
    pub tasks: Vec<Task>,
    /// Backend-independent source geometry, indexed like `tasks`. Hand-built
    /// programs may leave this empty.
    pub task_origins: Vec<TaskOrigin>,
    /// Top-level executable stream.
    pub entry: Stream,
    /// External logical patches injected when their source task is released.
    pub inputs: Vec<LogicalInput>,
    /// Terminal logical operators and their Pauli-frame registers.
    pub outputs: Vec<LogicalOutput>,
    /// Configured mock-decoder latency, in physical memory rounds.
    pub decoder_latency_rounds: u32,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            qubit_count: 0,
            bit_count: 0,
            record_count: 0,
            tasks: Vec::new(),
            task_origins: Vec::new(),
            entry: Stream::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            decoder_latency_rounds: DEFAULT_DECODER_LATENCY_ROUNDS,
        }
    }
}

/// Source geometry and function of one lowered task.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TaskOrigin {
    pub function: TaskFunction,
    /// Reserved logical patch sites in source-lattice `(x, y)` coordinates.
    pub sites: Box<[[i32; 2]]>,
    /// Full source-lattice members retained for audit.
    pub members: Box<[[i32; 3]]>,
}

/// Plot-facing function classification derived from compiled IR provenance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskFunction {
    Classical,
    Cube,
    PreparedY,
    Factory,
    FactoryBody,
    TemporalHadamard,
    Memory,
    AdaptiveMeasurement,
    IdealBoundary,
    Control,
    #[default]
    Synthetic,
}

/// One external logical input in dense VM coordinates.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LogicalInput {
    pub port: [i32; 3],
    /// Source task released when the encoded state arrives.
    pub task: TaskId,
    pub data_qubits: Box<[QubitId]>,
    pub stabilizers: Box<[PauliProduct]>,
    pub x: PauliProduct,
    pub z: PauliProduct,
}

/// One terminal logical output in dense VM coordinates.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LogicalOutput {
    /// Source block-graph port coordinate.
    pub port: [i32; 3],
    /// Logical X observable at the output cut.
    pub x: PauliProduct,
    /// Logical Z observable at the output cut.
    pub z: PauliProduct,
    /// Whether an X frame correction is required.
    pub frame_x: BitId,
    /// Whether a Z frame correction is required.
    pub frame_z: BitId,
}

impl Program {
    /// Every instruction, including instructions owned by nested streams.
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.tasks.iter().map(|task| &task.instruction)
    }

    /// Independently released sources in dense task order.
    #[must_use]
    pub fn sources(&self) -> Vec<SourceSchedule> {
        self.tasks
            .iter()
            .enumerate()
            .filter_map(|(task, source)| {
                source.source.map(|role| SourceSchedule {
                    task: task as TaskId,
                    role,
                    release: source.release,
                    duration: source.duration,
                })
            })
            .collect()
    }

    /// Render the complete lowered instruction stream as compact JSON.
    ///
    /// This is an inspection format, not a stable persistence protocol.
    ///
    /// # Errors
    ///
    /// Propagates JSON serialization errors.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Render the complete lowered instruction stream as readable JSON.
    ///
    /// This is an inspection format, not a stable persistence protocol.
    ///
    /// # Errors
    ///
    /// Propagates JSON serialization errors.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// A set of tasks scheduled together. Dependencies, rather than vector order,
/// define readiness; order only makes inspection deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Stream {
    pub tasks: Box<[TaskId]>,
    /// Declared bit result of a region body. `None` is constant false.
    pub value: Option<BitId>,
    /// Tasks whose boundary bindings the region exports.
    pub bindings: Box<[TaskId]>,
}

/// One schedulable unit.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Task {
    /// Stable inspection label derived from the IR scope, node id, and role.
    pub label: String,
    /// Absolute earliest start time.
    pub release: f64,
    /// Source role, present only on independently released roots.
    pub source: Option<SourceRole>,
    /// All predecessors, including Bloq value, order, and quantum edges and
    /// sequencing added for physical qubit reuse.
    pub dependencies: Box<[TaskId]>,
    /// Optional classical gate. A false gate suppresses physical work and
    /// decoder requests, clears bindings, and writes `false` to `output`.
    pub activation: Option<BitId>,
    /// Classical result register, when this task produces one.
    pub output: Option<BitId>,
    /// Physical support retained for scheduling and wait events.
    pub qubits: Box<[QubitId]>,
    /// Nominal active duration. Dynamic instructions may extend it.
    pub duration: f64,
    pub instruction: Instruction,
}

/// Why an independently released task exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceRole {
    Factory,
    LogicalInput,
    PreparedY,
    Clifford,
}

/// Stable schedule summary for one independently released source.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceSchedule {
    pub task: TaskId,
    pub role: SourceRole,
    pub release: f64,
    pub duration: f64,
}

/// Epoch from which decoder latency is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum DecodeTiming {
    Measurement,
    FactoryCompletion,
}

/// One decoder query against an already-lowered `Observable`.
///
/// Corrected and flip outputs share one sampled decision for a request.
/// The shared payload keeps their query fields aligned. The runtime caches
/// decisions by observable task and attempt epoch, and rejects inconsistent
/// inputs for that causal key.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct DecodeRequest {
    /// Boolean result of this causal solve.
    pub output: bloq_ir::ObservableOutput,
    /// Task id of the `Observable` this estimates.
    pub observable: TaskId,
    /// The observable's program-level index.
    pub index: u32,
    /// Register holding the raw measured parity.
    pub raw: BitId,
    /// Duration of one physical memory round, the unit of decoder latency.
    pub round_duration: f64,
    /// Epoch decoder latency is measured from.
    pub timing: DecodeTiming,
}

/// VM instruction set. Structured bodies refer to the shared task arena.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum Instruction {
    Quantum(QuantumTask),
    Eval(BoolOp),
    Accumulate(RecordParity),
    Bind(Box<[BoundaryBinding]>),
    /// Shared bit parity and boundary bindings; only a complete Observable reads them.
    ReadoutRecipe {
        bits: Box<[BitId]>,
        bindings: Box<[TaskId]>,
    },
    Observable {
        index: u32,
        bits: Box<[BitId]>,
        bindings: Box<[TaskId]>,
    },
    /// Publish one observable's corrected parity or decoder flip.
    Decode(DecodeRequest),
    Discard(BoolOp),
    /// Repeat a source-isolated body without rolling back global state or time.
    Rus {
        body: Stream,
        restart: BoolOp,
        owned_qubits: Box<[QubitId]>,
        attempt_bits: Box<[BitId]>,
        attempt_records: Box<[RecordId]>,
        resource: ResourceId,
        /// Optional explicit preparation before attempts after the first. Compiled
        /// factories replay authored body resets and leave this empty.
        retry_prepare: QuantumStream,
        /// Exact terminal seam cycles used while attempt decoders are pending.
        decoder_hold: Box<[MemoryCycle]>,
        /// Physical cultivation exits after which GAP rounds begin.
        cultivation_exits: Box<[TaskId]>,
    },
    /// Hold a live patch until every task in `until` is ready. Execute as many
    /// whole syndrome rounds as fit, then idle only for a sub-round remainder.
    WaitFor {
        /// Sibling producer exits whose active work must finish.
        until: Box<[TaskId]>,
        memory: Box<[MemoryCycle]>,
    },
    SignalReady {
        resource: ResourceId,
    },
    /// Statically requested memory rounds. The stream already contains the
    /// exact full `rounds` circuit and executes once; `rounds` is inspection
    /// metadata.
    MemoryRounds {
        rounds: u32,
        stream: QuantumStream,
        detectors: Box<[RecordParity]>,
    },
    /// A physical hold shorter than one available syndrome round.
    Idle {
        duration: f64,
        noise_ratio: f64,
        operations: Box<[QuantumOp]>,
    },
}

/// Boolean bytecode with absolute register operands, or argument slots within a
/// shared [`BoolOp::Call`] body.
///
/// Evaluators must load every
/// listed operand, including unselected `Select` arms and duplicate parity
/// inputs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum BoolOp {
    Const(bool),
    Copy(BitId),
    Not(Box<BoolOp>),
    Parity {
        inputs: Box<[BitId]>,
        constant: bool,
    },
    Xor(Box<[BoolOp]>),
    And(Box<[BoolOp]>),
    Or(Box<[BoolOp]>),
    Select {
        condition: Box<BoolOp>,
        when_false: Box<BoolOp>,
        when_true: Box<BoolOp>,
    },
    /// Evaluate a shared pure body. Copies in `body` name positions in `inputs`;
    /// the invocation binds those positions to registers in its enclosing scope.
    /// Every argument is required, including arguments unused by the body.
    /// Values and timestamps are evaluated independently per call.
    Call {
        body: std::sync::Arc<BoolOp>,
        inputs: Box<[BitId]>,
    },
}

/// Affine measurement parity. `constant` is the expected-sign term.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct RecordParity {
    pub records: std::sync::Arc<[RecordId]>,
    pub constant: bool,
}

/// A conditional quantum stage is lowered to its finite executable choices.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct QuantumTask {
    pub alternatives: Box<[QuantumAlternative]>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct QuantumAlternative {
    /// Exact input assignment selecting this alternative.
    pub when: Box<[(BitId, bool)]>,
    pub stream: QuantumStream,
    /// Records produced by this selected membership.
    pub selected_records: Box<[RecordId]>,
    /// Records deliberately unavailable under this selected membership.
    pub excluded_records: Box<[RecordId]>,
    pub detectors: Box<[RecordParity]>,
    pub restarts: Box<[RecordParity]>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct QuantumStream {
    pub moments: Box<[Moment]>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Moment {
    pub kind: Option<MomentKind>,
    pub duration: f64,
    pub operations: Box<[QuantumOp]>,
}

/// Physical moment class used by deterministic lane alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MomentKind {
    Reset,
    Rotation,
    Interaction,
    Measurement,
}

/// One dynamically repeatable memory cycle on a seam. Boundary flows carry
/// the last-round evidence needed to compose detectors across extra cycles.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct MemoryCycle {
    pub stream: QuantumStream,
    pub round_duration: f64,
    pub detectors: Box<[RecordParity]>,
    pub boundary_flows: Box<[BoundaryFlow]>,
    /// Zero-cycle values for the padding frontier, derived from the exact
    /// source-to-padding detector rows materialized by IR seam composition.
    pub initializers: Box<[FrontierInitializer]>,
}

/// Set `record` to `parity` before a dynamic wait executes its first cycle.
/// A cycle overwrites the same frontier record with its measured value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FrontierInitializer {
    pub record: RecordId,
    pub parity: RecordParity,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BoundaryFlow {
    pub start: PauliProduct,
    pub end: PauliProduct,
    pub parity: RecordParity,
    /// Record overwritten by the next cycle for this open frontier, when the
    /// authored flow carries a single measurement.
    pub frontier_record: Option<RecordId>,
}

/// Runtime boundary operator. `input` faces are declarative source endpoints;
/// `output` faces are measured when their observable closes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BoundaryBinding {
    pub input: bool,
    pub operator: PauliProduct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Pauli {
    X,
    Y,
    Z,
}

/// Hermitian Pauli product. `negative` represents the real sign; ±i products
/// are rejected during lowering.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct PauliProduct {
    pub negative: bool,
    pub terms: Box<[(QubitId, Pauli)]>,
}

/// Named single-qubit Clifford gates. Paulis and resets have dedicated ops.
#[expect(
    non_camel_case_types,
    reason = "names match the source gate vocabulary"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Clifford1 {
    H,
    H_XY,
    H_YZ,
    H_NXY,
    H_NXZ,
    H_NYZ,
    SQRT_X,
    SQRT_X_DAG,
    SQRT_Y,
    SQRT_Y_DAG,
    S,
    S_DAG,
    C_XYZ,
    C_ZYX,
    C_NXYZ,
    C_XNYZ,
    C_XYNZ,
    C_NZYX,
    C_ZNYX,
    C_ZYNX,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum QuantumOp {
    Gate1 {
        gate: Clifford1,
        qubit: QubitId,
    },
    Gate2 {
        control_basis: Pauli,
        target_basis: Pauli,
        control: QubitId,
        target: QubitId,
    },
    Pauli {
        basis: Pauli,
        qubit: QubitId,
    },
    T {
        basis: Pauli,
        qubit: QubitId,
        adjoint: bool,
    },
    Measure {
        observable: PauliProduct,
        /// Aliased instance records all receive the same outcome.
        records: Box<[RecordId]>,
        flip_probability: f64,
    },
    Reset {
        basis: Pauli,
        qubit: QubitId,
    },
    ConditionalPauli {
        basis: Pauli,
        qubit: QubitId,
        control: RecordId,
    },
    Depolarize1 {
        probability: f64,
        qubits: Box<[QubitId]>,
    },
    Depolarize2 {
        probability: f64,
        pairs: Box<[(QubitId, QubitId)]>,
    },
    PauliError {
        probability: f64,
        basis: Pauli,
        qubits: Box<[QubitId]>,
    },
}

impl QuantumOp {
    pub(crate) fn qubits(&self) -> Box<dyn Iterator<Item = QubitId> + '_> {
        match self {
            QuantumOp::Gate1 { qubit, .. }
            | QuantumOp::Pauli { qubit, .. }
            | QuantumOp::T { qubit, .. }
            | QuantumOp::Reset { qubit, .. }
            | QuantumOp::ConditionalPauli { qubit, .. } => Box::new(std::iter::once(*qubit)),
            QuantumOp::Gate2 {
                control, target, ..
            } => Box::new([*control, *target].into_iter()),
            QuantumOp::Measure { observable, .. } => {
                Box::new(observable.terms.iter().map(|(qubit, _)| *qubit))
            }
            QuantumOp::Depolarize1 { qubits, .. } | QuantumOp::PauliError { qubits, .. } => {
                Box::new(qubits.iter().copied())
            }
            QuantumOp::Depolarize2 { pairs, .. } => {
                Box::new(pairs.iter().flat_map(|&(left, right)| [left, right]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn program_json_exposes_physical_stream_and_schedule_metadata() {
        let quantum = Task {
            label: "source".into(),
            release: 2.0,
            source: Some(SourceRole::Clifford),
            dependencies: Box::new([]),
            activation: None,
            output: None,
            qubits: Box::new([0]),
            duration: 1.0,
            instruction: Instruction::Quantum(QuantumTask {
                alternatives: Box::new([QuantumAlternative {
                    stream: QuantumStream {
                        moments: Box::new([Moment {
                            kind: Some(MomentKind::Rotation),
                            duration: 1.0,
                            operations: Box::new([QuantumOp::Gate1 {
                                gate: Clifford1::H,
                                qubit: 0,
                            }]),
                        }]),
                    },
                    ..QuantumAlternative::default()
                }]),
            }),
        };
        let dependent = Task {
            label: "ready".into(),
            release: 0.0,
            source: None,
            dependencies: Box::new([0]),
            activation: None,
            output: None,
            qubits: Box::new([]),
            duration: 0.0,
            instruction: Instruction::SignalReady { resource: 0 },
        };
        let decode = Task {
            label: "decode".into(),
            release: 0.0,
            source: None,
            dependencies: Box::new([0]),
            activation: None,
            output: Some(0),
            qubits: Box::new([]),
            duration: 0.0,
            instruction: Instruction::Decode(DecodeRequest {
                output: bloq_ir::ObservableOutput::Corrected,
                observable: 1,
                index: 0,
                raw: 1,
                round_duration: 1.5,
                timing: DecodeTiming::Measurement,
            }),
        };
        let product = PauliProduct {
            negative: false,
            terms: Box::new([(0, Pauli::Z)]),
        };
        let program = Program {
            qubit_count: 1,
            bit_count: 2,
            tasks: vec![quantum, dependent, decode],
            task_origins: vec![
                TaskOrigin {
                    function: TaskFunction::Cube,
                    sites: Box::new([[3, 4]]),
                    members: Box::new([[3, 4, 5]]),
                },
                TaskOrigin::default(),
                TaskOrigin::default(),
            ],
            entry: Stream {
                tasks: Box::new([0, 1, 2]),
                ..Stream::default()
            },
            inputs: vec![LogicalInput {
                port: [3, 4, 5],
                task: 0,
                data_qubits: Box::new([0]),
                stabilizers: Box::new([]),
                x: product.clone(),
                z: product.clone(),
            }],
            outputs: vec![LogicalOutput {
                port: [3, 4, 6],
                x: product.clone(),
                z: product,
                frame_x: 0,
                frame_z: 1,
            }],
            ..Program::default()
        };

        let value: serde_json::Value = serde_json::from_str(&program.to_json().unwrap()).unwrap();
        assert_eq!(value["entry"]["tasks"], json!([0, 1, 2]));
        assert_eq!(value["tasks"][0]["source"], "clifford");
        assert_eq!(value["tasks"][1]["dependencies"], json!([0]));
        assert_eq!(
            value["tasks"][0]["instruction"]["Quantum"]["alternatives"][0]["stream"]["moments"][0]
                ["operations"][0]["Gate1"],
            json!({ "gate": "H", "qubit": 0 })
        );
        // `DecodeRequest` is a newtype payload, so its fields stay flat under
        // the variant tag exactly as the old struct variant serialized them.
        assert_eq!(
            value["tasks"][2]["instruction"]["Decode"],
            json!({
                "output": "Corrected",
                "observable": 1,
                "index": 0,
                "raw": 1,
                "round_duration": 1.5,
                "timing": "Measurement",
            })
        );
        assert_eq!(value["task_origins"][0]["function"], "cube");
        assert_eq!(value["inputs"][0]["task"], 0);
        assert_eq!(value["outputs"][0]["port"], json!([3, 4, 6]));
    }
}
