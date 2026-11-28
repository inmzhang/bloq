//! Program IR for bloq compiler output.
//!
//! [`Bloq`] is the public, circuit-bearing graph IR produced by `bloq_compile`
//! and consumed by backend emitters.
//!
//! ## Traversal
//!
//! Reads are id-native: every graph level (the [`Bloq`] top level or a region
//! [`SubGraph`] body) exposes the same accessors — [`SubGraph::node`],
//! [`SubGraph::nodes`], [`SubGraph::incoming`]/[`SubGraph::outgoing`] (yielding
//! [`BloqEdgeRef`]s), the dataflow queries [`SubGraph::value_inputs`] /
//! [`SubGraph::value_output`] / [`SubGraph::boundary_outputs`] — and [`Bloq::walk`]/[`Bloq::levels`]
//! recurse through region bodies with a [`LevelPath`] for cross-level node
//! identity. The underlying petgraph storage is not public API.
//!
//! ## Derived reads
//!
//! Three layers sit on top of raw traversal, for questions a consumer would
//! otherwise answer by matching on node variants itself:
//!
//! - **Structural queries** ([`Bloq::regions_of`], [`Bloq::selection_seams`],
//!   [`Bloq::node_by_block`], [`SubGraph::quantum_tail`], …) name a program's
//!   recurring landmarks. Callers use them to construct [`MemoryRoundTarget`]s
//!   for [`Bloq::insert_memory_rounds`] or [`Bloq::insert_memory_rounds_batch`].
//! - **[`Bloq::resolve_classical`]** folds the classical dataflow graph under
//!   a fixed path into the measurement recipe behind a node's value, and
//!   **[`Bloq::classical_value`]** answers the other half of the same
//!   question: the bit that path already fixes, including an activation predicate.
//! - **[`NodeKey`]** is a provenance-derived node identity that survives
//!   recompilation, for lining one program up against another compile of the
//!   same block graph.
//!
//! ## Exhaustive data enums (deliberate)
//!
//! The data enums ([`BloqNodeKind`], [`ClassicalNode`], [`RegionNode`],
//! [`NodeProvenance`], [`ValueRole`], …) are intentionally **not**
//! `#[non_exhaustive]`: a backend must fail to compile — not silently
//! wildcard — when the IR grows a construct, because an unhandled node kind
//! would emit wrong output rather than error (e.g. `bloq_stim` rejects each region
//! variant by name). Do not "fix" this by adding `#[non_exhaustive]` to them;
//! error enums, by contrast, are non-exhaustive.

mod alignment;
mod binary;
mod detslice;
mod edit;
mod flatten;
mod instantiation;
mod ir;
mod key;
mod membership;
mod optimize;
mod pin;
mod query;
mod readout_frame;
mod resolve;
mod stats;
#[cfg(test)]
mod test_fixture;
mod text;
mod traverse;
mod validation;

pub use alignment::{
    AlignedLayer, AlignedMomentRef, AlignedSlot, MomentAlignmentError, MomentLane,
    align_moment_lanes, aligned_moment_segments,
};
pub use binary::{
    BLOQ_BINARY_EXTENSION, BLOQ_BINARY_MAGIC, BLOQ_BINARY_VERSION, BinaryDecodeError,
};
// Types that appear in this crate's own signatures, re-exported so a consumer
// can name them without depending on `bloq_circuit`/`bloq_utils` directly
// (C-REEXPORT). The whole circuit vocabulary lives in `circuit` (below).
pub use bloq_circuit::{Basis, CoordinateOverflowError, FlattenLimits};
pub use bloq_utils::PortRole;
pub use detslice::{
    MomentKind, MomentSegment, MomentSegmenter, NodeRef, NodeSlices, ProgramRegion,
    ProgramRegionBreak, ProgramRegionId, ProgramSliceError, ProgramSliceOptions, ProgramSlices,
    RegionView, moment_segments, program_detector_slices, program_detector_slices_with_options,
};
pub use edit::{EditError, MemoryRoundTarget};
pub use flatten::FlattenError;
pub use ir::{
    Bloq, BloqEdge, BloqEdgeRef, BloqNode, BloqNodeId, BloqNodeKind, BodySelector, BoundaryFace,
    BundleDetector, BundleMeasurement, ClassicalExpr, ClassicalNode, CycleDetected, DetectorBundle,
    DetectorBundleError, DetectorBundleId, DetectorBundlePool, DetectorBundleUse, FramePair,
    InstanceProvenance, LogicalInput, LogicalOutput, MetadataValue, NodeDetector,
    NodeDetectorAddress, NodeDetectorParity, NodeDetectorView, NodeDetectors, NodeProvenance,
    ObservableOutput, PathScratch, PipePadding, PipeSeam, QuantumEdge, QuantumGuard, QuantumNode,
    QuantumTimeline, RegionNode, SourceBlockRef, SpatialPortPart, SubGraph, TemplateDetector,
    TemplateId, TemporalPipeRef, ValueInput, ValueRef, ValueRole,
};
pub use key::NodeKey;
pub use pin::MembershipPinError;
pub use query::{RegionKind, RegionRef, StructureError};
pub use resolve::{ClassicalAssignment, ClassicalResolution, ResolveError};
pub use stats::{BloqStats, BloqStatsError};
pub use text::{BLOQ_TEXT_EXTENSION, BLOQ_TEXT_VERSION, TextParseError};
pub use traverse::{LevelPath, LevelSegment, NodeCx, WalkControl};
pub use validation::{BloqValidationError, ValidatedPlans};

// Template instantiation and per-instance machinery is compiler/backend
// lowering surface, public only through `lowering` (below). These crate-visible
// aliases keep it reachable as `crate::Foo` so the lowering-pipeline modules
// need not spell out `lowering::`.
pub(crate) use instantiation::NodeTemplateInstanceMergeError;
pub(crate) use ir::{
    BloqTemplate, BloqTemplatePool, InstanceBoundaryOperator, InstanceMeasurement, NodeRestart,
    ReadoutFrame, TemplateDetectorParity, TemplateDetectorScope, TemplateInstance,
    TemplateInstanceId, TemplateRepeatState, TemplateRestart,
};

/// Graph visualization shared with the editor.
pub mod visualization;

/// Compiler and backend lowering surface.
///
/// This contains template instantiation and per-instance machinery a backend
/// consumes when emitting a [`Bloq`]. Kept
/// separate from the stable program-graph API at the crate root; arms-length
/// graph consumers should not need these.
pub mod lowering {
    pub use crate::edit::{PaddingInstance, drain_composed_chains};
    pub use crate::instantiation::{
        InstantiationOptions, NodeEmissionPlan, NodeTemplateInstanceMergeError,
        validate_template_circuit,
    };
    pub use crate::ir::{
        BloqTemplate, BloqTemplatePool, InstanceBoundaryOperator, InstanceMeasurement, NodeRestart,
        ReadoutFrame, TemplateDetectorParity, TemplateDetectorScope, TemplateInstance,
        TemplateInstanceId, TemplateRepeatState, TemplateRestart,
    };
    pub use crate::membership::PredicateAnalysis;
}

/// Circuit vocabulary embedded by Bloq IR templates and annotations.
///
/// Execution backends depend on `bloq_ir` alone and use this module instead of
/// reaching through the IR to its `bloq_circuit` implementation dependency. The
/// glob is deliberate: the whole vocabulary travels together, and enumerating it
/// here would only add a second list to keep in step with `bloq_circuit`.
///
/// The few circuit types that appear in `bloq_ir`'s own signatures
/// ([`Basis`], [`CoordinateOverflowError`], [`FlattenLimits`]) are *also* at the
/// crate root, so those have two paths on purpose — the root one so a signature
/// is nameable without this module, this one so a backend can take the lot.
pub mod circuit {
    pub use bloq_circuit::*;
}

pub(crate) type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;
pub(crate) type FxSet<T> = rustc_hash::FxHashSet<T>;

/// The most commonly used items, for `use bloq_ir::prelude::*`.
///
/// Also available as `bloq::ir::prelude` through the facade crate.
pub mod prelude {
    #[doc(no_inline)]
    pub use crate::{Bloq, BloqNode, BloqNodeId, BloqNodeKind, SubGraph};
}
