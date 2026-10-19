//! Execute compiled [`Bloq`](bloq_ir::Bloq) programs.
//!
//! [`run_bloq`] and its variants retain the noiseless physical-verification
//! path. [`lower()`] produces the backend-independent [`Instruction`] stream
//! consumed by [`runtime`], which adds dynamic timing, noise, retries, waits,
//! idling, and mock streaming-decoder decisions.
//!
//! ```
//! use bloq_compile::compile;
//! use bloq_graph::GalleryItem;
//! use bloq_vm::{LoweringConfig, lower};
//!
//! let ir = compile(&GalleryItem::CNOT.build(), 3)?;
//! let options = LoweringConfig::default();
//! let program = lower(&ir, &options)?;
//! let result = program.run(options.runtime_config(42))?;
//! assert!(!result.artifact.discarded);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Physical verification executes compiled circuits, independently of
//! `bloq_graph`'s logical QuiZX verifier. It supports non-Clifford programs;
//! Clifford circuits can also be checked with Stim.
//!
//! The physical verifier adds no noise and assumes zero decoder flips. It still
//! executes explicit noise instructions already present in a circuit. Its
//! IR-mandated retries are bounded by [`DEFAULT_RUS_ATTEMPT_CAP`]. The dynamic
//! runtime uses [`decoder`] instead.
//!
//! [`CircuitExecutor`] is the flat counterpart for a single
//! [`CoordCircuit`](bloq_ir::circuit::CoordCircuit); [`ShotRecord`] is the
//! per-shot measurement outcome map both paths produce.
//!
//! ## Where items live
//!
//! The crate root carries what it takes to *drive* the VM: [`lower()`] and its
//! configuration, [`run()`] and its configuration, the [`run_bloq`] family, the
//! [`Program`] they exchange, and the error types. The public modules carry the
//! full *vocabularies* a consumer reads results back with, which are too large
//! to flatten into one root namespace:
//!
//! - [`instruction`] — the whole lowered ISA ([`Task`](instruction::Task),
//!   [`QuantumOp`](instruction::QuantumOp), [`BoolOp`](instruction::BoolOp),
//!   [`MemoryCycle`](instruction::MemoryCycle), …).
//! - [`runtime`] — the execution trace vocabulary
//!   ([`ExecutionEvent`](runtime::ExecutionEvent),
//!   [`MeasurementRecord`](runtime::MeasurementRecord),
//!   [`WaitReason`](runtime::WaitReason), …) behind an [`ExecutionArtifact`].
//! - [`decoder`] — the mock streaming-decoder policy model.
//! - [`verify`] — the physical verifier's report and expectation readouts.
//!
//! The names that appear in root signatures are re-exported at the root as
//! well, so those have two paths on purpose: a signature stays nameable
//! without reaching into a module. From [`instruction`] that is
//! [`Program`], [`Instruction`], [`SourceRole`], [`SourceSchedule`],
//! [`TaskOrigin`], [`TaskFunction`], and [`DEFAULT_DECODER_LATENCY_ROUNDS`];
//! from [`runtime`] it is [`run`], [`RunResult`], [`RunMetadata`],
//! [`RuntimeConfig`], [`RuntimeLimits`], [`RuntimeError`],
//! [`ExecutionArtifact`], [`LogicalInputState`], and [`TaskMetadata`].
//! Prefer the root path for those; reach into the module for the rest of the
//! vocabulary. Everything in the private modules has exactly one path.
//!
//! The engine (`ticit`) vocabulary is re-exported at the root only, under
//! `Engine`-prefixed aliases where its names would otherwise collide with this
//! crate's own ([`EnginePauli`] against [`instruction::Pauli`],
//! [`EngineInstruction`] against [`Instruction`]).
mod backend;
mod circuit;
mod closure;
pub mod decoder;
mod gate_table;
pub mod instruction;
mod lower;
mod prepared;
mod program;
pub mod runtime;
#[cfg(test)]
mod test_support;
pub mod verify;

// The engine's own vocabulary, re-exported so consumers of `bloq_vm` need
// not name `ticit` directly.
pub use backend::{
    BatchOutcome, Gate1Q, Instruction as EngineInstruction, MeasureResult, Pauli as EnginePauli,
    PauliBasis, PauliString as EnginePauliString, SimError, Simulator,
};
pub use circuit::{CircuitExecutor, ShotRecord};
pub use instruction::{
    DEFAULT_DECODER_LATENCY_ROUNDS, Instruction, Program, SourceRole, SourceSchedule, TaskFunction,
    TaskOrigin,
};
pub use lower::{DEFAULT_MAX_QUANTUM_VARIANTS, LowerError, LoweringConfig, SourceTiming, lower};
pub use program::{
    DEFAULT_RUS_ATTEMPT_CAP, InputSeed, OutputLogical, PreparationContext, SavedOutput,
    ShotContext, ShotFrame, run_bloq, run_bloq_with_captured_outputs, run_bloq_with_hook,
    run_bloq_with_io, run_bloq_with_io_and_captured_outputs,
};
pub use runtime::{
    ExecutionArtifact, LogicalInputState, RunMetadata, RunResult, RuntimeConfig, RuntimeError,
    RuntimeLimits, TaskMetadata, run,
};

// These two inherent impls sit at the crate root rather than beside their
// types because each spans the lowering/runtime seam: neither `lower` nor
// `runtime` may depend on the other, and both of these need names from both.

impl LoweringConfig<'_> {
    /// Runtime defaults consistent with this lowering configuration.
    ///
    /// Circuit noise is inserted while lowering. Dynamic idle noise instead
    /// uses a per-time-unit rate, so this converts the circuit model's
    /// per-tick `p_idle` by the configured gate duration.
    #[must_use]
    pub fn runtime_config(&self, seed: u64) -> RuntimeConfig {
        RuntimeConfig {
            seed,
            idle_error_rate: self
                .noise
                .map_or(0.0, |noise| noise.p_idle / self.gate_duration),
            ..RuntimeConfig::default()
        }
    }
}

impl Program {
    /// Execute this lowered program with one runtime configuration.
    ///
    /// Reuse the same program with different seeds to run independent shots
    /// without recompiling or lowering again.
    ///
    /// # Errors
    ///
    /// Returns a typed runtime, decoder, resource-limit, or backend failure.
    pub fn run(&self, config: RuntimeConfig) -> Result<RunResult, RuntimeError> {
        run(self, config)
    }
}

#[cfg(test)]
mod public_api_tests {
    use bloq_ir::circuit::NoiseModel;

    use super::*;

    #[test]
    fn runtime_config_preserves_idle_noise_per_gate_tick() {
        let noise = NoiseModel::uniform_depolarizing(1e-3);
        let unit = LoweringConfig {
            noise: Some(&noise),
            ..LoweringConfig::default()
        };
        let quarter = LoweringConfig {
            gate_duration: 0.25,
            noise: Some(&noise),
            ..LoweringConfig::default()
        };

        assert_eq!(unit.runtime_config(7).seed, 7);
        assert_eq!(unit.runtime_config(7).idle_error_rate, noise.p_idle);
        assert_eq!(
            quarter.runtime_config(7).idle_error_rate,
            4.0 * noise.p_idle
        );
        assert_eq!(
            quarter.gate_duration * quarter.runtime_config(7).idle_error_rate,
            noise.p_idle
        );
        assert_eq!(
            LoweringConfig::default().runtime_config(7).idle_error_rate,
            0.0
        );
    }
}

use glam::IVec2;

pub(crate) const MAX_PHYSICAL_BOUNDARY_BINDINGS: usize = 1_000_000;

/// Failures on the *noiseless physical-verification* path.
///
/// That path is the [`run_bloq`] family, [`CircuitExecutor`], and the shared
/// program-preparation and output-closure analysis in front of them — which
/// [`lower()`] also runs, and reports as [`LowerError::Analysis`].
///
/// The dynamic causal runtime is the other execution engine in this crate and
/// reports [`RuntimeError`] instead. Neither type is a superset of the other:
/// this one covers *exact* execution and the static analysis in front of it,
/// [`RuntimeError`] covers timing, retries, and mock decoding.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecError {
    /// Dense physical-verification identifiers exhausted `u32`.
    #[error("{0} identifier space exhausted")]
    IdOverflow(&'static str),
    /// Expanding a complete physical readout's binding DAG exceeded the verifier's budget.
    #[error("physical readout binding expansion exceeds {limit} work units")]
    BoundaryBindingExpansionLimit {
        /// Maximum recipe visits and operator occurrences for one query.
        limit: usize,
    },
    /// The circuit still contains an [`Op::Repeat`](bloq_ir::circuit::Op); call
    /// [`CoordCircuit::flatten`](bloq_ir::circuit::CoordCircuit::flatten) first.
    #[error("circuit contains an un-flattened REPEAT; flatten before executing")]
    UnflattenedRepeat,

    /// A coordinate referenced by an op is absent from the coordinate layout.
    #[error("operation references unknown qubit coordinate ({},{})", .0.x, .0.y)]
    UnknownCoord(IVec2),

    /// A raw [`CoordCircuit`](bloq_ir::circuit::CoordCircuit) failed the same
    /// structural preflight used for template emission.
    #[error("invalid circuit: {0}")]
    InvalidCircuit(#[source] bloq_ir::lowering::NodeTemplateInstanceMergeError),

    /// A required Bloq IR emission plan could not be materialized.
    #[error("Bloq IR materialization failed: {0}")]
    InvalidProgram(#[from] bloq_ir::BloqValidationError),

    /// A shared detector use has an invalid bundle or owner binding.
    #[error("{0}")]
    DetectorBundle(#[from] bloq_ir::DetectorBundleError),

    /// The full logical Pauli signature does not fit the platform's index width.
    #[error("the 4^{outputs} logical Pauli signature size overflows usize")]
    LogicalSignatureTooLarge {
        /// Number of logical outputs requested.
        outputs: usize,
    },

    /// Compact surface code injection requires one physical seed at the unique
    /// intersection of the input patch's logical X and Z axes.
    #[error("input port {port:?} is not a canonical single-seed injection patch")]
    InvalidInjectionPatch {
        /// Input port with invalid injection geometry.
        port: glam::IVec3,
    },

    /// A deferred resource preparation named a port not present in the
    /// compiled program's logical-input table.
    #[error("resource preparation names unknown logical input port {port:?}")]
    UnknownPreparationInput {
        /// Unknown input port.
        port: glam::IVec3,
    },

    /// A CCZ resource must bind three different logical input ports.
    #[error("CCZ preparation requires three distinct logical input ports")]
    InvalidCczPreparation,

    /// A conditional Pauli / detector referenced a measurement id that has not
    /// been produced yet in this shot.
    #[error("reference to measurement record {0} that has not been produced")]
    MissingRecord(u32),

    /// A merged circuit record has no local-to-global identity in the prepared
    /// program.
    #[error("measurement record {0} has no local-to-global remap")]
    MissingRecordRemap(u32),

    /// An MPP product was empty (no Paulis).
    #[error("MPP product is empty")]
    EmptyMppProduct,

    /// A detector referenced a loop-carried state term that the flattened
    /// circuit did not resolve to concrete measurement records.
    #[error("detector references an unresolved loop-carried state {0}")]
    UnresolvedLoopState(u32),

    /// Flattening Bloq IR failed.
    #[error("Bloq IR flatten error: {0}")]
    BloqFlatten(#[from] bloq_ir::FlattenError),

    /// An instance offset moves a template qubit outside the global `i32`
    /// coordinate lattice.
    #[error("{0}")]
    CoordinateOverflow(#[from] bloq_ir::CoordinateOverflowError),

    /// The bloq node graph has a cycle, so no deterministic emit order exists.
    #[error("bloq emit-order error: {0}")]
    Cycle(#[from] bloq_ir::CycleDetected),

    /// The bloq graph is structurally invalid for execution.
    #[error("malformed bloq graph: {0}")]
    MalformedGraph(&'static str),

    /// A `Decode`'s named observable folds an `Output`-face boundary operator
    /// whose support a quantum node scheduled AFTER the decode's cut touches: the
    /// operator has no unresolved-but-modeled frontier, so the decoder estimate
    /// computed at the cut would refer to a face the later node reopens/consumes
    /// — a wrong-time boundary. Unlike the *terminal* read, which
    /// `closure::plan_early_output_reads` moves in front of the consuming node,
    /// a decode cut has no fixture and stays rejected up front by
    /// `guard_decode_observable_closure`. `observable` is the `Observable` node
    /// id, `decode`/`later` are node ids, `instance` the boundary operator's
    /// instance.
    #[error(
        "decode {decode} estimates observable {observable}, which folds an Output-face \
         boundary operator on instance {instance} whose support is touched by node {later} \
         scheduled after the decode cut; the face is not closed at the cut"
    )]
    DecodeObservableNotClosed {
        /// Decode node reading the observable.
        decode: u32,
        /// Observable node being read.
        observable: u32,
        /// Boundary-owning template instance.
        instance: u32,
        /// Later node touching the boundary.
        later: u32,
    },

    /// Incompatible partial observable reads cannot be split across output cuts
    /// without destroying their joint correlations.
    #[error(
        "observables {first} and {second} require anticommuting partial output reads before node {before}; their remaining factors are unavailable or gated separately"
    )]
    IncompatibleEarlyOutputReads {
        /// First incompatible observable.
        first: u32,
        /// Second incompatible observable.
        second: u32,
        /// Node before which both reads are required.
        before: u32,
    },

    /// An exported boundary's condition is unavailable before its output is consumed.
    #[error("observable {observable} has an unavailable output-binding gate before node {before}")]
    EarlyOutputGateUnavailable {
        /// Observable whose gate is unavailable.
        observable: u32,
        /// Node requiring the early read.
        before: u32,
    },

    /// The engine raised an error mid-shot.
    #[error("engine error: {0}")]
    Engine(#[from] backend::SimError),
}
