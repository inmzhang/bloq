use glam::{IVec2, IVec3};
use thiserror::Error;

/// An error raised while validating or compiling a [`BlockGraph`](bloq_graph::BlockGraph).
///
/// Variants cover invalid configuration, unsupported or malformed blocks,
/// observable/detector composition failures, and errors forwarded from the
/// graph, circuit, and IR layers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompileError {
    /// The caller cancelled compilation at a safe checkpoint.
    #[error("{0}")]
    Cancelled(#[from] bloq_graph::ComputationCancelled),
    /// Compiler-configured Boolean analysis exceeded a resource limit.
    #[error(
        "{0}; {help}",
        help = bloq_graph::ModuleCertificationLimits::RESOURCE_LIMIT_HELP
    )]
    BooleanResource(#[from] bloq_utils::boolean::BooleanResourceError),
    /// The requested code distance is outside the supported odd range.
    #[error("invalid distance {0}: must be odd and in 3..=255")]
    InvalidDistance(u32),
    /// A compiled object was linked under a different compile configuration.
    #[error("compiled object target {object:?} does not match linker target {linker:?}")]
    CompiledObjectTargetMismatch {
        /// Configuration used to build the object.
        object: Box<crate::CompileConfig>,
        /// Configuration used by the linker.
        linker: Box<crate::CompileConfig>,
    },
    /// A linked module site has no compiled template matching its physical signature.
    #[error(
        "module object '{module}' has no physical ABI variant for local block {local_position:?} linked at {linked_position:?}: {signature}"
    )]
    MissingModulePhysicalVariant {
        /// Module definition containing the site.
        module: String,
        /// Site position within the definition.
        local_position: IVec3,
        /// Site position after module linking.
        linked_position: IVec3,
        /// Missing physical signature.
        signature: String,
    },
    /// A Clifford proxy received the wrong number of selective-block pins.
    #[error("Clifford proxy requires {expected} selective pins, got {actual}")]
    CliffordProxyPinCount {
        /// Required number of pins.
        expected: usize,
        /// Supplied number of pins.
        actual: usize,
    },
    /// The two per-basis arms must present the same output signature.
    #[error("Selective({kind}) arm signature mismatch: {reason}")]
    SelectiveArmSignatureMismatch {
        /// Selective block whose arms disagree.
        kind: bloq_graph::SelectiveKind,
        /// Signature difference found between the arms.
        reason: String,
    },
    /// A block's connected faces are unsupported by fixed-bulk lowering.
    #[error("{kind} block has invalid fixed-bulk connectivity {connectivity:?}")]
    InvalidConnectivity {
        /// Block kind being validated.
        kind: bloq_graph::BlockKind,
        /// Unsupported face connectivity.
        connectivity: crate::signature::Connectivity,
    },
    /// A temporal leaf block has no neighboring boundary basis.
    #[error("{kind} block requires a temporal cube neighbor to determine its boundary basis")]
    MissingBoundaryBasis {
        /// Block kind requiring the basis.
        kind: bloq_graph::BlockKind,
    },
    /// A spatial port is connected to a block that cannot supply cube geometry.
    #[error("spatial Port at {port:?} is connected to non-cube block at {neighbor:?}")]
    SpatialPortNeighborNotCube {
        /// Spatial port position.
        port: IVec3,
        /// Connected non-cube position.
        neighbor: IVec3,
    },
    /// A spatial port's virtual cube kind could not be inferred.
    #[error("cannot infer the virtual cube kind for spatial Port at {port:?}")]
    SpatialPortCubeInferenceFailed {
        /// Spatial port position.
        port: IVec3,
    },
    /// A spatial port borders a cube taller than the substitution supports.
    #[error(
        "spatial Port at {port:?} borders a tall cube (height={height}); tall spatial-Port substitution is not supported yet"
    )]
    TallSpatialPortUnsupported {
        /// Spatial port position.
        port: IVec3,
        /// Unsupported neighboring cube height.
        height: bloq_graph::CubeHeight,
    },
    /// No unused coordinate is available for a multiplexed logical output.
    #[error("cannot allocate an external physical qubit for Multiplex Port at {port:?}")]
    MultiplexOutputCoordinateUnavailable {
        /// Multiplex port position.
        port: IVec3,
    },
    /// No unused coordinate is available for an ideal Bell boundary adapter.
    #[error("cannot allocate a private physical qubit for ideal Port boundary at {port:?}")]
    PrivatePortCoordinateUnavailable {
        /// Port position.
        port: IVec3,
    },
    /// A spatial Hadamard wall endpoint is not a cube.
    #[error(
        "spatial Hadamard pipe endpoint {pos:?} is not a cube; the wall's extended stabilizers \
         reach into both neighbours' seam-adjacent data columns, which only a cube patch has"
    )]
    SpatialHadamardUnsupportedEndpoint {
        /// Unsupported endpoint position.
        pos: IVec3,
    },
    /// One spatial component contains Hadamard walls on both spatial axes.
    #[error(
        "spatial component at {pos:?} mixes X- and Y-axis Hadamard walls; fixed-bulk compilation \
         supports one spatial Hadamard axis per connected component"
    )]
    MixedSpatialHadamardUnsupported {
        /// Position identifying the mixed-axis component.
        pos: IVec3,
    },
    /// A cube height resolves to fewer than two syndrome rounds.
    #[error(
        "cube at {pos:?} with height={height} compiles to {rounds} syndrome rounds at distance \
         {distance}; a cube needs at least 2 (one initialization stage, one measurement stage)"
    )]
    CubeHeightTooShort {
        /// Cube position.
        pos: IVec3,
        /// Authored symbolic height.
        height: bloq_graph::CubeHeight,
        /// Code distance used to resolve the height.
        distance: u32,
        /// Resolved syndrome-round count.
        rounds: i64,
    },
    /// A block position cannot be mapped into the signed physical coordinate range.
    #[error(
        "block coordinate {block_xy:?} at distance {distance} exceeds the signed 32-bit qubit layout"
    )]
    BlockLayoutCoordinateOverflow {
        /// Block's spatial coordinate.
        block_xy: glam::IVec2,
        /// Code distance scaling the coordinate.
        distance: u32,
    },
    /// Two same-layer nodes own the same physical qubit.
    #[error(
        "WF-LAYOUT: same-layer Bloq nodes {first:?} and {second:?} own overlapping physical qubit {qubit:?}"
    )]
    OverlappingNodeQubit {
        /// First conflicting node.
        first: bloq_ir::BloqNodeId,
        /// Second conflicting node.
        second: bloq_ir::BloqNodeId,
        /// Shared physical coordinate.
        qubit: IVec2,
    },
    /// A dependency forces another node to reuse a qubit while a temporal
    /// seam still carries its state to an unexecuted consumer.
    #[error(
        "WF-LAYOUT: node {reuser:?} reuses live qubit {qubit:?} after producer {producer:?} but before its required consumer {consumer:?}; the required execution order cannot preserve the pending quantum state"
    )]
    LiveQubitReuse {
        /// Node that has already produced the live state.
        producer: bloq_ir::BloqNodeId,
        /// Node still requiring that state.
        consumer: bloq_ir::BloqNodeId,
        /// Intervening node using the same physical coordinate.
        reuser: bloq_ir::BloqNodeId,
        /// Shared physical coordinate.
        qubit: IVec2,
    },
    /// A reused region footprint needs a boundary summary whose body membership
    /// cannot be expressed in the containing graph's Boolean domain.
    #[error(
        "WF-LAYOUT: cannot prove live-qubit reuse around region {node:?} with conditional body membership"
    )]
    ConditionalRegionLivenessUnsupported {
        /// Region requiring the unavailable boundary summary.
        node: bloq_ir::BloqNodeId,
    },
    /// Patch-rotation circuit construction failed.
    #[error("PatchRotation block construction failed: {reason}")]
    PatchRotationConstructionFailed {
        /// Construction failure detail.
        reason: String,
    },
    /// Stabilizer-flow composition failed for one physical node.
    #[error("program node flow composition failed for layer {layer} members {members:?}: {source}")]
    NodeFlowCompositionFailed {
        /// Physical layer containing the node.
        layer: i64,
        /// Source blocks merged into the node.
        members: Vec<IVec3>,
        /// Underlying flow-composition failure.
        #[source]
        source: bloq_circuit::FlowError,
    },
    /// Program detector-flow composition failed.
    #[error("program detector composition failed: {source}")]
    DetectorFlowCompositionFailed {
        /// Underlying flow-composition failure.
        #[source]
        source: bloq_circuit::FlowError,
    },
    /// A repeat boundary contains an unsupported moving residual operator.
    #[error(
        "loop-boundary residual with a moving operator (non-empty flow start) inside a repeat body is unsupported"
    )]
    LoopBoundaryMovingOperatorUnsupported,
    /// A generated conditional Pauli cannot use the supported feedback paths.
    #[error(
        "compiler-generated ConditionalPauli requires the commuting-feedback path or parity folding"
    )]
    ConditionalPauliFeedbackUnsupported,
    /// A T block has no free neighboring cell for its cultivation spill.
    #[error(
        "T block at {pos:?} needs a free spatial neighbor cell (-y/+x/+y/-x) for the Steane cultivation spill; all four are occupied or reserved by another T block"
    )]
    TBlockNeedsFreeNeighbor {
        /// T-block position.
        pos: IVec3,
    },
    /// Walking-block circuit construction failed.
    #[error("Walking block construction failed: {0}")]
    WalkingConstruction(#[source] WalkError),
    /// Observable lowering found no gateway entry for a local stabilizer.
    #[error(
        "observable composition failed: missing gateway entry {local_stabilizer} at {block_pos:?} for observable {observable_index}"
    )]
    ObservableMissingGatewayEntry {
        /// Source block position.
        block_pos: IVec3,
        /// Observable being lowered.
        observable_index: u32,
        /// Diagnostic description of the missing local gateway key.
        local_stabilizer: String,
    },
    /// A logical boundary operator disagrees across a temporal seam.
    #[error(
        "observable {observable_index} boundary operator is discontinuous across the temporal seam {lower:?} -> {upper:?}: the lower block's +Z `operator_out` must equal the upper block's -Z `operator_in`"
    )]
    ObservableSeamMismatch {
        /// Observable being lowered.
        observable_index: u32,
        /// Lower seam endpoint.
        lower: IVec3,
        /// Upper seam endpoint.
        upper: IVec3,
    },
    /// Continuing-branch assembly cannot be scheduled.
    #[error("continuing branch assembly cannot be scheduled: {reason}")]
    BranchAssembly {
        /// Scheduling failure detail.
        reason: String,
    },
    /// A Clifford proxy was requested for a graph with structural branches.
    #[error(
        "Clifford proxies do not support structural branches because their pins only resolve selective blocks"
    )]
    StructuralBranchCliffordProxyUnsupported,
    /// Memory padding was requested with zero rounds.
    #[error("memory-round padding requires at least one round (U11)")]
    PaddingRoundsZero,
    /// An IR seam edit failed.
    #[error("seam edit failed (U12): {0}")]
    Edit(#[from] bloq_ir::EditError),
    /// The generated Bloq program failed validation.
    #[error("{0}")]
    BloqValidation(#[from] bloq_ir::BloqValidationError),
    /// Physical coordinate arithmetic overflowed.
    #[error("{0}")]
    CoordinateOverflow(#[from] bloq_ir::CoordinateOverflowError),
    /// Rebasing a logical boundary exceeded the signed coordinate range.
    #[error(
        "rebasing logical-boundary coordinate {coordinate:?} around patch offset {offset:?} exceeds the signed 32-bit lattice"
    )]
    LogicalBoundaryRebaseOverflow {
        /// Boundary coordinate before rebasing.
        coordinate: glam::IVec2,
        /// Physical patch offset.
        offset: glam::IVec2,
    },
    /// Circuit construction failed.
    #[error("{0}")]
    Circuit(#[from] bloq_circuit::CircuitError),
    /// Runtime stabilizer-basis construction failed.
    #[error("{0}")]
    RuntimeBasis(#[from] bloq_graph::RuntimeBasisError),
    /// Logical output-correction construction failed.
    #[error("{0}")]
    OutputCorrection(#[from] bloq_graph::OutputCorrectionError),
    /// Source block-graph validation or analysis failed.
    #[error("{0}")]
    BlockGraph(#[from] bloq_graph::BlockGraphError),
    /// Module certification failed.
    #[error("{0}")]
    ModuleCertification(#[from] bloq_graph::ModuleCertificationError),
    /// Hierarchical block-graph validation or linking failed.
    #[error("{0}")]
    Module(#[from] Box<bloq_graph::ModuleError>),
}

impl From<bloq_graph::ModuleError> for CompileError {
    fn from(error: bloq_graph::ModuleError) -> Self {
        Self::Module(Box::new(error))
    }
}

/// Reject `observed` units of `resource` against `limit`.
///
/// Every compiler budget funnels here so a breach stays a typed
/// [`BooleanResourceError`](bloq_utils::boolean::BooleanResourceError) and is
/// never reported as an invalid source or an unreachable choice. Callers check
/// the would-be total *before* allocating for it, so a site growing a
/// collection by one passes `len + 1`.
///
/// This only rejects; it never charges. Whether a breach also spends budget is
/// the caller's ordering decision — charge before checking to retain the spend
/// across a retry, check first to leave it unspent.
pub(crate) fn check_resource(
    resource: &'static str,
    observed: usize,
    limit: usize,
) -> Result<(), CompileError> {
    if observed > limit {
        return Err(bloq_utils::boolean::BooleanResourceError {
            resource,
            observed,
            limit,
        }
        .into());
    }
    Ok(())
}

/// Check a growing resource count without letting an unlimited cap hide overflow.
pub(crate) fn add_resource(
    resource: &'static str,
    current: usize,
    additional: usize,
    limit: usize,
) -> Result<usize, CompileError> {
    let total = current.checked_add(additional).ok_or_else(|| {
        CompileError::BooleanResource(bloq_utils::boolean::BooleanResourceError {
            resource,
            observed: usize::MAX,
            limit,
        })
    })?;
    check_resource(resource, total, limit)?;
    Ok(total)
}

impl CompileError {
    /// The verification audit inherits compiler Boolean limits, while its
    /// depth and flattening limits retain their separate contracts.
    pub(crate) fn from_verification_audit(error: bloq_ir::BloqValidationError) -> Self {
        use bloq_ir::BloqValidationError;
        use bloq_ir::lowering::NodeTemplateInstanceMergeError;

        match error {
            BloqValidationError::BooleanResource(resource)
            | BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::BooleanResource(resource),
                ..
            }
            | BloqValidationError::InvalidTemplateCircuit {
                source: NodeTemplateInstanceMergeError::BooleanResource(resource),
                ..
            } => Self::BooleanResource(resource),
            error => Self::BloqValidation(error),
        }
    }

    /// Whether computation stopped at a work or allocation budget, rather
    /// than proving the source or resulting IR invalid.
    pub fn is_resource_limited(&self) -> bool {
        self.resource_limited_error().is_some()
    }

    /// Whether the caller cancelled this computation, including a forwarded
    /// graph-analysis failure.
    pub fn is_cancelled(&self) -> bool {
        let mut error: &(dyn std::error::Error + 'static) = self;
        loop {
            if error.is::<bloq_graph::ComputationCancelled>() {
                return true;
            }
            let Some(source) = error.source() else {
                return false;
            };
            error = source;
        }
    }

    /// Configuration help when a compilation-certification limit was exceeded.
    ///
    /// Standalone IR validation and circuit-flattening limits have
    /// separate configuration contracts and return no certification hint here.
    pub fn resource_limit_help(&self) -> Option<&'static str> {
        let error = self.resource_limited_error()?;
        (error.is::<bloq_utils::boolean::BooleanResourceError>()
            || error.is::<bloq_graph::StabilizerError>()
            || error.is::<bloq_graph::ModuleCertificationError>())
        .then_some(bloq_graph::ModuleCertificationLimits::RESOURCE_LIMIT_HELP)
    }

    fn resource_limited_error(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let mut error: &(dyn std::error::Error + 'static) = self;
        loop {
            if error.is::<bloq_utils::boolean::BooleanResourceError>()
                || matches!(
                    error.downcast_ref::<bloq_graph::StabilizerError>(),
                    Some(bloq_graph::StabilizerError::ResourceLimited { .. })
                )
                || matches!(
                    error.downcast_ref::<bloq_graph::ModuleCertificationError>(),
                    Some(bloq_graph::ModuleCertificationError::ResourceLimited { .. })
                )
                || error
                    .downcast_ref::<bloq_ir::BloqValidationError>()
                    .is_some_and(bloq_ir::BloqValidationError::is_resource_limited)
                || error
                    .downcast_ref::<bloq_ir::lowering::NodeTemplateInstanceMergeError>()
                    .is_some_and(
                        bloq_ir::lowering::NodeTemplateInstanceMergeError::is_resource_limited,
                    )
                || matches!(
                    error.downcast_ref::<bloq_circuit::CircuitError>(),
                    Some(bloq_circuit::CircuitError::FlattenResourceLimit { .. })
                )
            {
                return Some(error);
            }
            error = error.source()?;
        }
    }
}

/// A surface code distance was rejected: distances must be odd and in
/// `3..=255`, limiting the quadratic grids materialized during compilation.
///
/// Returned by [`CompileConfig::try_new`](crate::CompileConfig::try_new) so an
/// illegal distance fails at the API boundary rather than deep in block
/// compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid code distance {0}: must be odd and in 3..=255")]
pub struct InvalidDistance(pub u32);

impl From<InvalidDistance> for CompileError {
    fn from(err: InvalidDistance) -> Self {
        CompileError::InvalidDistance(err.0)
    }
}

/// Why a Walking block's step construction failed, one variant per failure
/// class of the step walk (see `block/walk.rs`).
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum WalkError {
    /// A step reintroduces a qubit outside its active frame.
    #[error("reinclude qubit {qubit} is outside the step frame")]
    ReincludeOutsideFrame {
        /// Out-of-frame qubit.
        qubit: glam::IVec2,
    },
    /// More than one region reintroduces the same qubit.
    #[error("multiple regions reinclude qubit {qubit}")]
    DuplicateReinclude {
        /// Duplicated qubit.
        qubit: glam::IVec2,
    },
    /// A contracting region leaves an unmeasured qubit active.
    #[error("contracting {basis} region kept unmeasured qubit {qubit}")]
    ContractingKeptUnmeasuredQubit {
        /// Contracting Pauli basis.
        basis: bloq_graph::Basis,
        /// Qubit left unmeasured.
        qubit: glam::IVec2,
    },
    /// An expanding region reaches more than one output measurement.
    #[error("expanding {basis} region hit multiple output measure qubits: {first} and {second}")]
    ExpandingMultipleOutputMeasures {
        /// Expanding Pauli basis.
        basis: bloq_graph::Basis,
        /// First output measurement.
        first: glam::IVec2,
        /// Second output measurement.
        second: glam::IVec2,
    },
    /// A final measurement leaves a continuing output operator.
    #[error("final {basis} measurement left continuing output {output}")]
    FinalMeasurementLeftOutput {
        /// Final measurement basis.
        basis: bloq_graph::Basis,
        /// Remaining output operator.
        output: bloq_circuit::PauliMap,
    },
    /// A stabilizer flow references a qubit that was not measured.
    #[error("flow references unmeasured qubit {qubit}")]
    FlowReferencesUnmeasuredQubit {
        /// Referenced qubit.
        qubit: glam::IVec2,
    },
    /// A final flow contains a Pauli outside its required basis.
    #[error("final {basis} flow contains non-matching Pauli {pauli} at {coord}")]
    FinalFlowMismatchedPauli {
        /// Required final-flow basis.
        basis: bloq_graph::Basis,
        /// Mismatched Pauli.
        pauli: bloq_graph::Pauli,
        /// Coordinate carrying the Pauli.
        coord: glam::IVec2,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_composition_errors_retain_their_cause() {
        let source =
            bloq_circuit::FlowError::CoordinateOverflow(bloq_circuit::CoordinateOverflowError {
                coordinate: IVec2::MAX,
                offset: IVec2::ONE,
            });
        for error in [
            CompileError::NodeFlowCompositionFailed {
                layer: 1,
                members: vec![],
                source: source.clone(),
            },
            CompileError::DetectorFlowCompositionFailed {
                source: source.clone(),
            },
        ] {
            assert_eq!(
                std::error::Error::source(&error)
                    .and_then(|error| error.downcast_ref::<bloq_circuit::FlowError>()),
                Some(&source),
            );
        }
    }
}
