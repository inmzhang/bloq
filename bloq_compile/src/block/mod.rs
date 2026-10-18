//! Per-block circuit compilation: turning a single block's signature into a
//! reusable circuit template, plus the shared patch, gateway, and measurement
//! primitives the convention compilers build on.

mod compile;
pub(crate) mod fixed_bulk;
pub(super) mod gateway;
pub(super) mod measurements;
pub(super) mod patch;
mod walk;

pub(crate) use compile::{
    CompiledTemplate, LoweringTemplate, LoweringTemplateId, LoweringTemplatePool,
};
pub(crate) use fixed_bulk::{
    SelectiveTemplates, SpatialHadamardKey, WallSide, compile_bell_port, compile_fixed_bulk,
    compile_multiplex_port, compile_realignment, compile_seam_padding_rounds, compile_selective,
    compile_spatial_hadamard, validate_fixed_bulk,
};
pub(crate) use gateway::{LocalStabilizer, ObservableGateway};
