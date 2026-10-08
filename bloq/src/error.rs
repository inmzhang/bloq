//! Shared errors for workflows that cross crate boundaries.

use std::error::Error as StdError;

type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// A result using the facade's [`Error`] by default.
///
/// `Result` means `Result<(), Error>`. An explicit error parameter remains
/// available for functions that need a narrower error type.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// A small container retaining the typed error from a Bloq workflow.
///
/// Subsystem APIs keep their concrete errors. In a function returning [`Result`],
/// `?` wraps those errors automatically. Use [`Self::downcast_ref`] to inspect
/// the original error and [`StdError::source`] to traverse its causes.
/// Application-specific errors can be wrapped with [`Self::new`].
///
/// ```
/// use bloq::prelude::*;
///
/// fn circuit_text(source: &str) -> Result<String> {
///     let graph = parse_blog_to_graph(source)?;
///     let config = CompileConfig::try_new(3)?;
///     let artifacts = compile_with(&graph, config)?;
///     Ok(emit_bloq_stim(&artifacts.bloq)?)
/// }
///
/// let error = circuit_text("invalid BLOG").unwrap_err();
/// assert!(error.is::<bloq::graph::BlockGraphError>());
/// ```
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(#[source] BoxError);

impl Error {
    /// Wrap an owned error, retaining its type, message, and causes.
    ///
    /// ```
    /// let error = bloq::Error::new("invalid".parse::<u32>().unwrap_err());
    /// assert!(error.is::<std::num::ParseIntError>());
    /// ```
    pub fn new(error: impl StdError + Send + Sync + 'static) -> Self {
        Self(Box::new(error))
    }

    /// Whether the original error has type `E`.
    pub fn is<E: StdError + 'static>(&self) -> bool {
        self.0.is::<E>()
    }

    /// Borrow the original error if it has type `E`.
    pub fn downcast_ref<E: StdError + 'static>(&self) -> Option<&E> {
        self.0.downcast_ref::<E>()
    }
}

impl From<BoxError> for Error {
    fn from(error: BoxError) -> Self {
        Self(error)
    }
}

// Keep automatic conversions without duplicating subsystem error enums.
// A blanket From<E> would overlap From<Error> because Error implements StdError.
macro_rules! impl_from {
    ($($(#[$meta:meta])* $error:ty),* $(,)?) => {
        $(
            $(#[$meta])*
            impl From<$error> for Error {
                fn from(error: $error) -> Self {
                    Self::new(error)
                }
            }
        )*
    };
}

impl_from! {
    bloq_graph::BlockGraphError,
    bloq_graph::BlockError,
    bloq_graph::InvalidBlockGraphError,
    bloq_graph::InvalidActionError,
    bloq_graph::ParseError,
    bloq_graph::ModuleError,
    bloq_graph::ModuleCertificationError,
    bloq_graph::UnknownCertificationLimit,
    bloq_graph::ZXError,
    bloq_graph::FillPortsError,
    bloq_graph::StabilizerError,
    bloq_graph::RuntimeBasisError,
    bloq_graph::SymbolicBasisError,
    bloq_graph::OutputCorrectionError,
    bloq_graph::ResolveDomainError,
    bloq_compile::CompileError,
    bloq_compile::InvalidDistance,
    bloq_circuit::CircuitError,
    bloq_circuit::FlowError,
    bloq_circuit::MeasurementFrameError,
    bloq_circuit::DetsliceError,
    bloq_circuit::CoordinateOverflowError,
    bloq_ir::BloqValidationError,
    bloq_ir::BloqStatsError,
    bloq_ir::TextParseError,
    bloq_ir::BinaryDecodeError,
    bloq_ir::CycleDetected,
    bloq_ir::DetectorBundleError,
    bloq_ir::FlattenError,
    bloq_ir::MembershipPinError,
    bloq_ir::EditError,
    bloq_ir::StructureError,
    bloq_ir::ResolveError,
    bloq_ir::MomentAlignmentError,
    bloq_ir::ProgramSliceError,
    bloq_stim::StimEmissionError,
    bloq_utils::DirectionParseError,
    bloq_utils::PauliError,
    bloq_utils::qasm::QasmError,
    bloq_utils::boolean::BooleanResourceError,
    strum::ParseError,
    std::io::Error,
    #[cfg(feature = "graph-verify")]
    bloq_graph::verify::VerifyLogicalError,
    #[cfg(feature = "graph-verify")]
    bloq_graph::FeedbackInferenceError,
    #[cfg(feature = "verify")]
    bloq_stim::StimVerifyError,
    #[cfg(feature = "verify")]
    bloq_stim::StimNoiseError,
    #[cfg(feature = "vm")]
    bloq_vm::LowerError,
    #[cfg(feature = "vm")]
    bloq_vm::RuntimeError,
    #[cfg(feature = "vm")]
    bloq_vm::ExecError,
    #[cfg(feature = "vm")]
    bloq_vm::SimError,
    #[cfg(feature = "vm")]
    bloq_vm::decoder::DecoderError,
    #[cfg(feature = "vm")]
    serde_json::Error,
}
