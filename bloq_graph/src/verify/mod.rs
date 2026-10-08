//! Graph-level logical verification by sampled ZX-diagram contraction.
//!
//! A [`LogicalVerifier`] converts an ordinary [`BlockGraph`](crate::BlockGraph)
//! into one symbolic QuiZX diagram. For a terminal structural source it caches
//! one such static verifier per reachable projected topology and dispatches by
//! the evaluated Branch conditions.
//! Named outcomes determine native spider phase parities, feedback contributes
//! conditional Pauli spiders, and selective measurements are
//! concretized after the classical action DAG is evaluated. Accepted nonzero
//! branches are compared with a caller-supplied expected ZX map up to arbitrary
//! nonzero scalar.

mod correction;
mod diagram;
mod error;
pub(crate) mod feedback;
mod parity;
mod qasm;
mod replay;

pub use diagram::{
    BoundaryOrder, BranchAssignment, BranchStatus, LogicalBranch, LogicalVerificationReport,
    LogicalVerifier, QuizxGraph, verify_logical,
};
pub use error::VerifyLogicalError;
pub(crate) use parity::{MeasurementKey, stabilizer_supports_measurement};
pub use qasm::{QasmError, parse_qasm};
pub(crate) use replay::{replay_fill, solve_evolved_correction};
