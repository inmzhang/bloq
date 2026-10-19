//! ZX-calculus graph layer for post-conversion analysis and verification.
//!
//! A [`ZXGraph`] is derived from a [`BlockGraph`](crate::BlockGraph) via
//! conversion and carries correlation-surface planning, output-correction
//! solving, and runtime-basis evolution for dynamic (selective and T) blocks.

mod convert;
mod graph;
mod guarded_readouts;
pub use guarded_readouts::{
    GuardedLocalSurface, GuardedReadoutPlan, LocalPauliSurface, SurfaceSupport,
};
mod guarded_surfaces;
pub use guarded_surfaces::{GuardedSurface, GuardedSurfaceKind, GuardedSurfaceSpace};
mod layer;
mod modular;
mod output_correction;
mod restriction;
mod runtime_basis;
mod stabilizer;

pub use graph::{NodeKind, ZXEdge, ZXError, ZXGraph, ZXNode};
pub use layer::ZXLayerView;
pub(crate) use modular::{FlowWitness, ProjectedExternalTable, ProjectionError, ProjectionLimits};
pub use output_correction::{
    OutputCorrectionError, OutputCorrectionRow, SymbolicFramePair, SymbolicOutputCorrection,
    solve_output_correction_symbolic, solve_output_correction_symbolic_prefix,
};
pub use runtime_basis::{RuntimeBasisError, RuntimeStabilizerBasis};
pub(crate) use stabilizer::fill_ports_auto;
#[cfg(test)]
pub(crate) use stabilizer::reduce_to_basis;
#[cfg(feature = "verify")]
pub(crate) use stabilizer::solve_coeff_combination;
pub(crate) use stabilizer::{CoeffVec, gaussian_elimination_with_tracking};
pub use stabilizer::{
    FillPortsError, Stabilizer, StabilizerError, StabilizerGenerator, StabilizerGenerators,
    StabilizerRowKind,
};
