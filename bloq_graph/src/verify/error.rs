//! Typed failures from logical verification.

use std::collections::BTreeMap;

use glam::IVec3;

/// An error raised while building, sampling, or contracting a logical ZX map.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyLogicalError {
    /// The caller cancelled construction or analysis.
    #[error("{0}")]
    Cancelled(#[from] crate::ComputationCancelled),
    /// Block-graph construction or validation failed.
    #[error("{0}")]
    Graph(#[from] crate::BlockGraphError),
    /// ZX translation or manipulation failed.
    #[error("{0}")]
    ZX(#[from] crate::ZXError),
    /// Output correction surfaces could not be solved.
    #[error("{0}")]
    OutputCorrection(#[from] crate::OutputCorrectionError),
    /// A logical boundary position appears more than once.
    #[error("logical boundary {0} is listed more than once")]
    DuplicateBoundary(IVec3),
    /// A requested logical boundary is not an open port.
    #[error("logical boundary {0} is not an open port")]
    InvalidBoundary(IVec3),
    /// An open port is absent from the requested logical boundary order.
    #[error("open port {0} is absent from the logical boundary order")]
    MissingBoundary(IVec3),
    /// A measurement-dependent branch lacks an assigned value.
    #[error("measurement {0:?} has no supplied branch value")]
    MissingMeasurementValue(String),
    /// An external Boolean input lacks an assigned value.
    #[error("external Boolean input {0:?} has no supplied branch value")]
    MissingInputValue(String),
    /// A measurement outcome cannot be represented as a native ZX phase.
    #[error("measurement {0:?} has no native ZX phase representation")]
    MeasurementPhaseUnavailable(String),
    /// The symbolic phase count exceeds QuiZX's variable-ID range.
    #[error("too many symbolic phases for QuiZX's u32 variable ids")]
    TooManySymbols,
    /// Actual and expected logical maps have different boundary arity.
    #[error(
        "expected map has {expected_inputs} input(s) and {expected_outputs} output(s), but the Bloq map has {actual_inputs} input(s) and {actual_outputs} output(s)"
    )]
    BoundaryArityMismatch {
        /// Number of inputs in the Bloq map.
        actual_inputs: usize,
        /// Number of outputs in the Bloq map.
        actual_outputs: usize,
        /// Number of inputs in the expected map.
        expected_inputs: usize,
        /// Number of outputs in the expected map.
        expected_outputs: usize,
    },
    /// The caller-supplied expected logical map is identically zero.
    #[error("the caller-supplied expected ZX map is the zero map")]
    ZeroExpectedMap,
    /// One sampled branch differs from the expected map up to scalar.
    #[error(
        "sampled nonzero branch does not equal the expected ZX map up to scalar: {classical_state:?}"
    )]
    MapMismatch {
        /// Classical assignment selecting the mismatched branch.
        classical_state: BTreeMap<String, bool>,
    },
    /// Sampling found no accepted branch with a nonzero logical map.
    #[error("no accepted nonzero branch was sampled in {samples} fuzz case(s)")]
    NoVerifiedBranch {
        /// Number of sampled classical assignments.
        samples: usize,
    },
    /// Initial runtime-basis construction failed.
    #[error("could not initialize the logical stabilizer basis")]
    RuntimeBasisInitializationFailed {
        /// Underlying runtime-basis failure.
        #[source]
        source: crate::RuntimeBasisError,
    },
    /// Applying one runtime selection to the basis failed.
    #[error("runtime basis update failed at {pos}")]
    RuntimeBasisUpdateFailed {
        /// Selective-block position being updated.
        pos: IVec3,
        /// Underlying runtime-basis failure.
        #[source]
        source: crate::RuntimeBasisError,
    },
    /// Required logical-output correction surfaces are unavailable.
    #[error("output correction surfaces are unavailable for {0:?}")]
    OutputSurfaceUnavailable(Vec<IVec3>),
}
