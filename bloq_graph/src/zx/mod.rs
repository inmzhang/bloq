//! ZX-calculus graph layer for post-conversion analysis and verification.
//!
//! A [`ZXGraph`] is derived from a [`BlockGraph`](crate::BlockGraph) via
//! conversion and drives stabilizer computation, output-correction solving, and
//! runtime-basis derivation for dynamic (selective and T) blocks.

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
mod readout;
mod restriction;
mod runtime_basis;
mod stabilizer;
mod symbolic_basis;

pub use graph::{NodeKind, ZXEdge, ZXError, ZXGraph, ZXNode};
pub use layer::ZXLayerView;
pub(crate) use modular::{FlowWitness, ProjectedExternalTable, ProjectionError, ProjectionLimits};
pub use output_correction::{
    OutputCorrectionError, OutputCorrectionRow, SymbolicFramePair, SymbolicOutputCorrection,
    solve_output_correction_symbolic, solve_output_correction_symbolic_prefix,
};
pub use readout::{NamedReadout, ReadoutCoordinates, ReadoutPlan};
pub use runtime_basis::{DerivedSurface, RuntimeBasisError, RuntimeStabilizerBasis};
pub(crate) use stabilizer::fill_ports_auto;
#[cfg(test)]
pub(crate) use stabilizer::reduce_to_basis;
#[cfg(feature = "verify")]
pub(crate) use stabilizer::{CoeffVec, solve_coeff_combination};
pub use stabilizer::{
    FillPortsError, SelectiveFixing, SelectiveFixingTarget, SelectiveFixings, Stabilizer,
    StabilizerError, StabilizerGenerator, StabilizerGenerators, StabilizerRowKind,
    selective_fixings,
};
pub use symbolic_basis::{SymbolicBasisError, SymbolicSite, SymbolicStabilizerBasis};
