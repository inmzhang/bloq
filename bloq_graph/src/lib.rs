//! Block graph representation for fault-tolerant surface code quantum computing.
//!
//! This crate provides the core data structures for expressing quantum
//! computations as graphs of surface code blocks connected by pipes
//! (lattice surgery operations). It includes the `.blog` DSL parser,
//! a block-graph gallery of standard circuits, ZX-calculus stabilizer
//! analysis, and GLTF export for 3D visualization.
//!
//! # Public dependency: `glam`
//!
//! `glam` is a public dependency: coordinates in the public API use
//! [`glam::IVec3`], so its types appear in this crate's signatures. The version
//! is pinned to match the `glam` that `bevy` re-exports, keeping the editor and
//! this crate on one vector type. A `glam` major-version bump is therefore a
//! breaking change for this crate's API even when nothing else here changes.

use std::path::PathBuf;
use std::sync::Arc;

use glam::IVec3;
use thiserror::Error;

mod action;
mod action_dag;
mod block;
mod branch;
mod cancellation;
#[doc(hidden)]
pub mod composition;
mod gallery;
#[cfg(feature = "gltf")]
mod gltf;
mod graph;
mod guarded;
mod height;
pub use composition::ModuleJoinOptions;
mod module_view;
mod parser;
mod program;
mod summary;
pub(crate) mod validate;
#[cfg(feature = "verify")]
pub mod verify;
mod zx;

pub use action::{Action, BinaryOp, Expr, FeedbackTarget, MeasureTarget};
pub use action_dag::{
    ActionDag, ActionDependency, ActionNode, ActionOwner, MeasurementObservable,
    ResolveDomainError, ResolveValueDomain,
};
pub use block::{
    Block, BlockError, BlockKind, CubeKind, PatchRotationKind, Pipe, SelectiveKind,
    WalkingBoundaryKind, WalkingKind, is_valid_tag,
};
// Deliberate facade: re-export the shared `bloq_utils` primitives so downstream
// code can import them from `bloq_graph` rather than depending on `bloq_utils`
// directly.
#[doc(no_inline)]
pub use bloq_utils::{
    Basis, Direction, DirectionParseError, Pauli, PauliBasis, PauliError, PauliString,
    PhasedPauliString, PortRole, RGBA, UDirection,
};
pub use branch::{Branch, BranchArm, BranchCut, BranchProjection, BranchRegion};
pub use cancellation::{CancellationToken, ComputationCancelled};
pub use gallery::{GalleryCategory, GalleryEntry, GalleryItem};
#[cfg(feature = "gltf")]
pub use gltf::{
    EDITOR_AMBIENT_BRIGHTNESS, EDITOR_DIRECTIONAL_ILLUMINANCE, GltfData, GltfFaceSelector,
    block_as_gltf_data, block_as_gltf_data_with_pipe_length, block_graph_as_gltf_data,
    block_graph_as_gltf_data_with_popped_faces, pipe_as_gltf_data,
    pipe_between_positions_as_gltf_data, stabilizer_as_gltf_data,
};
pub use graph::{BlockGraph, BlockLayerView};
pub use guarded::{GuardedProjection, GuardedTopology, GuardedVariable, selective_selector_name};
pub use height::{CubeHeight, MAX_CUBE_HEIGHT_CELLS};
pub use module_view::{ModuleView, ModuleViewModule, module_color};
#[cfg(test)]
pub(crate) use parser::load_graph_with_resolver_and_limits;
#[cfg(test)]
pub(crate) use parser::parse_inline_graph;
pub(crate) use parser::parse_inline_graph_with_limits;
pub use parser::{
    ParseError, ast, lower_blog_ast_deferred, lower_blog_ast_lenient,
    lower_blog_graph_ast_deferred, parse_actions, parse_blog_program_to_ast, parse_blog_to_ast,
    parse_blog_to_graph, parse_blog_to_graph_with_limits,
};
pub use program::{
    BitBinding, BitOutput, BitRef, InstancePort, LeafModuleCertificate, LinkedModuleDefinition,
    MaterializedModuleSite, ModuleCertificationError, ModuleCertificationLimits, ModuleError,
    ModuleInstance, ModuleInterface, ModuleOrientation, ModuleRotation, PortDirection,
    QuantumConnection, QuantumPort, UnknownCertificationLimit, flatten_module_definition,
    qualified_name,
};
pub use summary::{ModuleSummary, PreparedModuleRelations, default_module_jobs, map_jobs};
// `Stabilizer`'s support maps are public `FxHashMap`s,
// so re-export the type rather than force downstream code onto a matching
// `rustc-hash` version to name them.
#[doc(no_inline)]
pub use rustc_hash::FxHashMap;
pub use validate::{InvalidActionError, InvalidBlockGraphError};
pub use zx::{
    FillPortsError, GuardedLocalSurface, GuardedReadoutPlan, GuardedSurface, GuardedSurfaceKind,
    GuardedSurfaceSpace, LocalPauliSurface, NodeKind, OutputCorrectionError, OutputCorrectionRow,
    RuntimeBasisError, RuntimeStabilizerBasis, Stabilizer, StabilizerError, StabilizerGenerator,
    StabilizerGenerators, StabilizerRowKind, SurfaceSupport, SymbolicFramePair,
    SymbolicOutputCorrection, ZXEdge, ZXError, ZXGraph, ZXLayerView, ZXNode,
    solve_output_correction_symbolic, solve_output_correction_symbolic_prefix,
};

/// The most commonly used items, for `use bloq_graph::prelude::*`.
///
/// Aggregated into `bloq::prelude` by the `bloq` facade crate.
pub mod prelude {
    #[doc(no_inline)]
    pub use crate::{
        Block, BlockGraph, BlockGraphError, BlockKind, Branch, BranchArm, CubeHeight, CubeKind,
        GalleryItem, Pipe, parse_blog_to_graph,
    };
    #[doc(no_inline)]
    pub use bloq_utils::{
        Basis, Direction, Pauli, PauliBasis, PauliString, PortRole, RGBA, UDirection,
    };
}

/// Errors from constructing, editing, parsing, validating, transforming, or
/// exporting a [`BlockGraph`].
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum BlockGraphError {
    /// A block already occupies the requested anchor position.
    #[error("block already exists at position {0}")]
    BlockExists(IVec3),
    /// A block footprint overlaps an occupied lattice cell.
    #[error("block footprint would overlap occupied position {0}")]
    BlockPositionOccupied(IVec3),
    /// Two materialized module blocks overlap at one lattice cell.
    #[error("module block footprints conflict at cell {position}: {first}; {second}")]
    ModuleBlockOverlap {
        /// Conflicting lattice cell.
        position: IVec3,
        /// First module site occupying the cell.
        first: Arc<MaterializedModuleSite>,
        /// Second module site occupying the cell.
        second: Arc<MaterializedModuleSite>,
    },
    /// No block occupies the requested position.
    #[error("no block found at position {0}")]
    BlockNotFound(IVec3),
    /// A tag is empty or contains a forbidden character.
    #[error(
        "invalid tag `{tag}`: tags must be non-empty and contain no whitespace, controls, `<`, or `>`"
    )]
    InvalidTag {
        /// Rejected tag text.
        tag: String,
    },
    /// No pipe connects the requested endpoints.
    #[error("no pipe found between positions {0} and {1}")]
    PipeNotFound(IVec3, IVec3),
    /// A pipe already connects the requested endpoints.
    #[error("pipe already exists between positions {0} and {1}")]
    PipeExists(IVec3, IVec3),
    /// Adding an offset to a lattice position overflowed.
    #[error("coordinate overflow while adding offset {offset} to position {position}")]
    CoordinateOverflow {
        /// Original lattice position.
        position: IVec3,
        /// Offset that could not be added.
        offset: IVec3,
    },
    /// Normalizing the graph's time span would overflow coordinates.
    #[error(
        "cannot normalize z span {min_z}..={max_z} to start at zero within the i32 coordinate range"
    )]
    CoordinateNormalizationOverflow {
        /// Minimum time coordinate before normalization.
        min_z: i32,
        /// Maximum time coordinate before normalization.
        max_z: i32,
    },
    /// Rotating a position would negate an unrepresentable coordinate.
    #[error("rotation of {position} negates unrepresentable {axis} component")]
    CoordinateRotationOverflow {
        /// Position that could not be rotated.
        position: IVec3,
        /// Rotation axis whose component overflowed.
        axis: UDirection,
    },
    /// A measurement pipe has no inferable temporal basis.
    #[error("measurement edge from {0} to {1} does not infer a time basis")]
    MeasurementEdgeMissingTimeBasis(IVec3, IVec3),
    /// The resulting block graph violates a graph invariant.
    #[error("{0}")]
    Invalid(#[from] validate::InvalidBlockGraphError),
    /// A classical action violates an action invariant.
    #[error("{0}")]
    InvalidAction(#[from] validate::InvalidActionError),
    /// Reading or writing a graph file failed.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// File path being accessed.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: Arc<std::io::Error>,
    },
    /// A block kind or property is invalid.
    #[error("{0}")]
    Block(#[from] block::BlockError),
    /// Bloq graph text could not be parsed.
    #[error("{0}")]
    Parse(#[from] parser::ParseError),
    /// Executable BLOG module declarations or imports are invalid.
    #[error("{0}")]
    ModuleSource(#[source] Arc<ModuleError>),
    /// A flat geometry operation was requested on authored module structure.
    #[error("{operation} requires a flat graph; call BlockGraph::flatten() explicitly first")]
    HierarchyRequiresFlatten {
        /// Operation that cannot preserve the authored hierarchy or interface.
        operation: &'static str,
    },
    /// A BLOG module hierarchy could not be materialized as a graph.
    #[error("{0}")]
    ModuleMaterialization(#[source] Arc<ModuleCertificationError>),
    /// ZX graph construction or analysis failed.
    #[error("{0}")]
    Zx(#[from] zx::ZXError),
    /// Open ports could not be filled consistently.
    #[error("{0}")]
    FillPorts(#[from] zx::FillPortsError),
    /// Stabilizer construction or analysis failed.
    #[error("{0}")]
    Stabilizer(#[from] zx::StabilizerError),
    #[cfg(feature = "gltf")]
    /// glTF JSON serialization failed.
    #[error("glTF JSON serialization failed")]
    GltfSerialize(#[source] Arc<gltf_json::Error>),
    /// Dynamic blocks support rotation only around the time axis.
    #[error("T and Selective blocks can only rotate around the time axis")]
    RotationUnsupportedDynamicBlocks,
    /// Y blocks require rotations by an even number of quarter turns.
    #[error("block graphs containing Y blocks can only rotate in 180-degree increments")]
    RotationRequiresHalfTurns,
    /// A multiplex Port's Z split has no X/Z-flipped form.
    #[error("cannot flip the X/Z basis: the multiplex Port at {0} has a fixed Z split")]
    BasisFlipUnsupportedMultiplexPort(IVec3),
    /// Walking and patch-rotation blocks support only time-axis rotation.
    #[error("Walking and PatchRotation blocks can only rotate around the time axis")]
    RotationWalkingRequiresTimeAxis,
    /// Non-default cube heights support only time-axis rotation.
    #[error("cube blocks with a non-default height can only rotate around the time axis")]
    RotationCubeHeightRequiresTimeAxis,
}

impl From<bloq_utils::boolean::BooleanResourceError> for StabilizerError {
    fn from(error: bloq_utils::boolean::BooleanResourceError) -> Self {
        Self::ResourceLimited {
            phase: error.resource,
            observed: error.observed,
            limit: error.limit,
        }
    }
}

impl From<bloq_utils::boolean::BooleanResourceError> for BlockGraphError {
    fn from(error: bloq_utils::boolean::BooleanResourceError) -> Self {
        StabilizerError::from(error).into()
    }
}

impl From<bloq_utils::boolean::BooleanResourceError> for RuntimeBasisError {
    fn from(error: bloq_utils::boolean::BooleanResourceError) -> Self {
        StabilizerError::from(error).into()
    }
}

impl BlockGraphError {
    /// Whether this action error can be repaired by adding another action.
    pub fn is_incomplete_action_program(&self) -> bool {
        matches!(
            self,
            Self::InvalidAction(
                InvalidActionError::UndefinedVariable(_)
                    | InvalidActionError::MissingResolveForSelective { .. }
                    | InvalidActionError::MissingResolveForBranch(_)
            )
        )
    }
}

/// Adds `offset` to `position`, rejecting an `i32` coordinate overflow.
///
/// # Errors
///
/// Returns [`BlockGraphError::CoordinateOverflow`] if any component overflows.
pub fn checked_add_position(position: IVec3, offset: IVec3) -> Result<IVec3, BlockGraphError> {
    let Some(x) = position.x.checked_add(offset.x) else {
        return Err(BlockGraphError::CoordinateOverflow { position, offset });
    };
    let Some(y) = position.y.checked_add(offset.y) else {
        return Err(BlockGraphError::CoordinateOverflow { position, offset });
    };
    let Some(z) = position.z.checked_add(offset.z) else {
        return Err(BlockGraphError::CoordinateOverflow { position, offset });
    };
    Ok(IVec3::new(x, y, z))
}
