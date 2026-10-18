//! Structural and classical-semantic validation for block graphs.

use crate::MeasureTarget;
use crate::{
    ActionDag, Basis, Block, BlockGraph, BlockGraphError, BlockKind, Direction, PatchRotationKind,
    Pipe, UDirection,
};
use glam::IVec3;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A structural well-formedness violation found while validating a
/// [`BlockGraph`](crate::BlockGraph)'s blocks, pipes, and lattice-surgery
/// geometry.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum InvalidBlockGraphError {
    /// A block kind requiring one pipe has a different degree.
    #[error("expected exactly one pipe connected to the block at {0}")]
    NotExactlySinglePipe(IVec3),
    /// A block kind that accepts only temporal pipes has a spatial pipe.
    #[error("spatial pipe connected to the block at {0}")]
    SpatialPipeNotAllowed(IVec3),
    /// A spatial port lacks an explicit input, output, or multiplex role.
    #[error("spatial Port at {0} must declare role=input, role=output, or role=multiplex")]
    SpatialPortRoleRequired(IVec3),
    /// A temporal port declares a role requiring a spatial connection.
    #[error("temporal Port at {pos} cannot use role {declared}; its inferred role is {inferred}")]
    TemporalPortRoleMismatch {
        /// Port position.
        pos: IVec3,
        /// Explicitly declared role.
        declared: crate::PortRole,
        /// Role inferred from the temporal pipe.
        inferred: crate::PortRole,
    },
    /// A block accepts only a pipe extending into its past.
    #[error("only time-like pipe connected to past is allowed for block at {0}")]
    FuturePipeNotAllowed(IVec3),
    /// A block accepts only a pipe extending into its future.
    #[error("only time-like pipe connected to future is allowed for block at {0}")]
    PastPipeNotAllowed(IVec3),
    /// Adjacent block faces imply incompatible lattice-surgery orientations.
    #[error(
        "cannot resolve the lattice surgery orientation for pipe between blocks at {0} and {1}"
    )]
    OrientationContradiction(IVec3, IVec3),
    /// A multi-pipe junction does not lie in one plane.
    #[error("block at {0} is not a planar junction")]
    JunctionNotPlanar(IVec3),
    /// A pipe face has the same basis on both perpendicular axes.
    #[error(
        "degenerate pipe face: block at {0} has the same basis on both axes perpendicular to pipe toward {1}"
    )]
    DegeneratePipeFace(IVec3, IVec3),
    /// Spatially connected cubes have different symbolic heights.
    #[error(
        "spatially connected cubes at {pos} and {neighbor} have different heights (height={height} vs height={neighbor_height})"
    )]
    CubeHeightSpatialMismatch {
        /// First cube position.
        pos: IVec3,
        /// Neighboring cube position.
        neighbor: IVec3,
        /// First cube height.
        height: crate::CubeHeight,
        /// Neighboring cube height.
        neighbor_height: crate::CubeHeight,
    },
    /// A pipe attaches to an invalid cell of a multi-cell cube.
    #[error(
        "multi-cell cube at {pos} has an invalid pipe endpoint {endpoint}; spatial pipes attach at the anchor, temporal pipes attach at bottom or top"
    )]
    CubeInvalidPipeEndpoint {
        /// Cube anchor position.
        pos: IVec3,
        /// Invalid pipe endpoint.
        endpoint: IVec3,
    },
    /// Two block footprints reserve the same lattice cell.
    #[error("block footprints overlap at occupied position {pos}")]
    BlockFootprintOverlap {
        /// Multiply occupied lattice cell.
        pos: IVec3,
    },
    /// A walking block endpoint has a spatial pipe.
    #[error("Walking block at {pos} has a non-temporal pipe attached at endpoint {endpoint}")]
    WalkingPipeNotTemporal {
        /// Walking-block anchor position.
        pos: IVec3,
        /// Endpoint with the invalid pipe.
        endpoint: IVec3,
    },
    /// A walking block's start endpoint has a non-past pipe.
    #[error("Walking block at {pos} start endpoint only allows past temporal pipes")]
    WalkingStartPipeNotPast {
        /// Walking-block anchor position.
        pos: IVec3,
    },
    /// A walking block's end endpoint has a non-future pipe.
    #[error("Walking block at {pos} end endpoint only allows future temporal pipes")]
    WalkingEndPipeNotFuture {
        /// Walking-block anchor position.
        pos: IVec3,
    },
    /// A patch-rotation endpoint does not have exactly one temporal pipe.
    #[error(
        "Patch rotation block at {pos} endpoint {endpoint} requires exactly one temporal pipe, got {count}"
    )]
    PatchRotationEndpointPipeCount {
        /// Patch-rotation anchor position.
        pos: IVec3,
        /// Endpoint being checked.
        endpoint: IVec3,
        /// Number of attached temporal pipes.
        count: usize,
    },
    /// A patch-rotation endpoint has a spatial pipe.
    #[error(
        "Patch rotation block at {pos} has a non-temporal pipe attached at endpoint {endpoint}"
    )]
    PatchRotationPipeNotTemporal {
        /// Patch-rotation anchor position.
        pos: IVec3,
        /// Endpoint with the invalid pipe.
        endpoint: IVec3,
    },
    /// A patch rotation has a Hadamard pipe at its start endpoint.
    #[error("Patch rotation block at {pos} does not allow a Hadamard pipe at its start endpoint")]
    PatchRotationStartPipeHadamard {
        /// Patch-rotation anchor position.
        pos: IVec3,
    },
    /// A patch-rotation endpoint has the wrong boundary orientation.
    #[error(
        "Patch rotation block at {pos} endpoint {endpoint} has boundary orientation {actual:?}, expected {expected:?}"
    )]
    PatchRotationBoundaryMismatch {
        /// Patch-rotation anchor position.
        pos: IVec3,
        /// Endpoint being checked.
        endpoint: IVec3,
        /// Boundary basis required by the rotation.
        expected: Basis,
        /// Basis inferred from the attached pipe, if any.
        actual: Option<Basis>,
    },
}

/// A semantic or graph-constraint violation found while validating a graph's
/// [`Action`](crate::Action)s.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum InvalidActionError {
    /// An edge measurement targets a temporal pipe.
    #[error("measurement target edge at {0} along {1} is time-like")]
    TimeLikeMeasurementEdge(IVec3, Direction),
    /// A node measurement targets an unsupported block kind.
    #[error("measurement target node at {0} is invalid")]
    InvalidMeasurementNode(IVec3),
    /// More than one measurement action targets the same node or edge.
    #[error("measurement target {0:?} is duplicated")]
    DuplicateMeasurementTarget(MeasureTarget),
    /// An action name is not a valid BLOG identifier.
    #[error("action name '{0}' is not a valid BLOG identifier")]
    InvalidActionName(String),
    /// A classical variable is defined more than once.
    #[error("variable '{0}' defined multiple times")]
    VariableRedefinition(String),
    /// An expression refers to an unavailable variable.
    #[error("undefined variable '{0}' used")]
    UndefinedVariable(String),
    /// A resolve action targets a non-selective block.
    #[error("resolve target {0} is not a selective block")]
    InvalidResolveTarget(IVec3),
    /// More than one resolve action targets the same selective block.
    #[error("resolve target {0} is duplicated")]
    DuplicateResolveTarget(IVec3),
    /// A branch action targets no explicit branch region.
    #[error("branch target {0} is not a branch region")]
    InvalidBranchTarget(IVec3),
    /// More than one branch action targets the same region.
    #[error("branch target {0} is duplicated")]
    DuplicateBranchTarget(IVec3),
    /// More than one explicit branch region uses the same name.
    #[error("branch region '{0}' is duplicated")]
    DuplicateBranchName(String),
    /// An explicit branch region has no selector action.
    #[error("branch region '{0}' has no resolve action")]
    MissingResolveForBranch(String),
    /// A branch action names no explicit branch region.
    #[error("resolve names unknown branch region '{0}'")]
    UnknownBranchName(String),
    /// The two arms of a branch expose incompatible boundary cuts.
    #[error("branch region '{name}' has invalid arm interface: {reason}")]
    InvalidBranchInterface {
        /// Branch region name.
        name: String,
        /// Interface mismatch diagnostic.
        reason: String,
    },
    /// A branch selector depends on a measurement inside that branch.
    #[error("branch target {target} is controlled by measurement {controller:?} inside its region")]
    BranchControllerInside {
        /// Stable branch anchor.
        target: IVec3,
        /// Measurement target used by the selector.
        controller: MeasureTarget,
    },
    /// A physical action target lies inside conditional branch geometry.
    #[error("branch target {target} contains physical target of action ordinal {ordinal}")]
    BranchContainsActionTarget {
        /// Stable branch anchor.
        target: IVec3,
        /// Source ordinal of the conflicting action.
        ordinal: usize,
    },
    /// Two explicit branch regions own the same block.
    #[error("branch regions at {first} and {second} overlap at block {block}")]
    OverlappingBranchRegions {
        /// First branch anchor.
        first: IVec3,
        /// Second branch anchor.
        second: IVec3,
        /// Overlapping block position.
        block: IVec3,
    },
    /// A requested branch projection omits a branch value.
    #[error("branch target {0} has no projection value")]
    MissingBranchValue(IVec3),
    /// A requested branch-arm selection has no reachable source outcomes.
    #[error("requested branch arm selection is unreachable")]
    UnreachableBranchSelection,
    /// A requested branch projection names an unknown branch.
    #[error("projection value names unknown branch target {0}")]
    UnknownBranchValue(IVec3),
    /// A requested branch projection assigns one branch more than once.
    #[error("branch target {0} has more than one projection value")]
    DuplicateBranchValue(IVec3),
    /// A selective block has no resolve action.
    #[error("Selective block at {target} is missing its resolve action")]
    MissingResolveForSelective {
        /// Unresolved selective-block position.
        target: IVec3,
    },
    /// Classical and implicit action dependencies form a cycle.
    #[error("action dependency cycle detected at ordinal {ordinal}")]
    DependencyCycle {
        /// Source action ordinal at which the cycle was detected.
        ordinal: usize,
    },
    /// A feedback action contains no correction targets.
    #[error("feedback requires at least one target")]
    EmptyFeedbackTargets,
    /// A feedback correction targets no block.
    #[error("feedback target {0} is not a block")]
    InvalidFeedbackTarget(IVec3),
}

/// Validates graph structure and action semantics, then — once the cached DAG
/// is stale — the stabilizer-derived measurement surfaces and dependency edges.
pub(crate) fn validate(graph: &BlockGraph) -> Result<(), BlockGraphError> {
    validate_with_limits(graph, crate::ModuleCertificationLimits::DEFAULT)
}

pub(crate) fn validate_with_limits(
    graph: &BlockGraph,
    limits: crate::ModuleCertificationLimits,
) -> Result<(), BlockGraphError> {
    let mut dag = validate_source(graph)?;
    let regions = graph.branch_regions()?;
    if (!regions.is_empty() || graph.has_actions()) && !graph.action_graph().is_analyzed() {
        let mut plan = graph.plan_local_readouts(limits)?;
        dag.attach_guarded_dependencies(plan.action_dependencies()?)?;
    }
    Ok(())
}

/// Structure and action semantics only, returning the DAG built along the way.
///
/// This is the precondition for building a [`ZXGraph`](crate::ZXGraph), so it
/// deliberately stops short of the physical analysis in [`validate`]: that
/// analysis needs a ZX graph to derive the rows it would check.
pub(crate) fn validate_source(graph: &BlockGraph) -> Result<ActionDag, BlockGraphError> {
    validate_structure(graph)?;
    let mut dag = graph.build_action_graph(&graph.actions())?;
    if !graph.has_actions() {
        // `build_action_graph` short-circuits on an empty list, so its
        // graph-constraint pass — every selective block needs a `resolve` —
        // has not run yet.
        dag.validate(Some(graph))?;
    }
    Ok(dag)
}

pub(crate) fn validate_structure(graph: &BlockGraph) -> Result<(), BlockGraphError> {
    structural_errors(graph)
        .into_iter()
        .next()
        .map_or(Ok(()), Err)
}

/// Collects every structural violation instead of stopping at the first.
///
/// Editing tools diff this set across a candidate edit so they can reject only
/// what the edit introduces and leave defects the graph already carried alone;
/// a fail-fast check cannot tell those apart.
///
/// The phases stay ordered and gated exactly as the fail-fast path had them: a
/// later phase runs only once the earlier ones are clean, because its geometry
/// inference assumes resolvable pipe endpoints and non-overlapping footprints.
/// Errors are therefore complete *within* a phase, not across phases.
pub(crate) fn structural_errors(graph: &BlockGraph) -> Vec<BlockGraphError> {
    let mut errors = Vec::new();

    collect_block_footprint_errors(graph, &mut errors);
    if !errors.is_empty() {
        return errors;
    }
    errors.extend(graph.pipes().filter_map(|pipe| pipe.try_endpoints().err()));
    if !errors.is_empty() {
        return errors;
    }
    for pipe in graph.pipes() {
        for endpoint in [pipe.src(), pipe.dst()] {
            if graph.get_endpoint_block(endpoint).is_none() {
                errors.push(BlockGraphError::BlockNotFound(endpoint));
            }
        }
    }
    if !errors.is_empty() {
        return errors;
    }
    collect_lattice_surgery_errors(graph, &mut errors);
    if !errors.is_empty() {
        return errors;
    }
    collect_block_constraint_errors(graph, &mut errors);
    errors
}

fn collect_block_footprint_errors(graph: &BlockGraph, errors: &mut Vec<BlockGraphError>) {
    let mut occupied = HashMap::<IVec3, Vec<(IVec3, BlockKind)>>::new();
    for block in graph.blocks() {
        let reserved = match block.checked_reserved_positions() {
            Ok(reserved) => reserved,
            Err(err) => {
                errors.push(err);
                continue;
            }
        };
        for pos in reserved {
            for (other_pos, other_kind) in occupied.get(&pos).into_iter().flatten() {
                if !block
                    .kind
                    .allows_reserved_overlap(block.pos(), *other_kind, *other_pos, pos)
                {
                    errors.push(InvalidBlockGraphError::BlockFootprintOverlap { pos }.into());
                }
            }
            occupied
                .entry(pos)
                .or_default()
                .push((block.pos(), block.kind));
        }
    }
}

fn collect_lattice_surgery_errors(graph: &BlockGraph, errors: &mut Vec<BlockGraphError>) {
    for (u_pos, v_pos, _, _, pipe) in graph.pipe_endpoints_with_blocks() {
        debug_assert_eq!(pipe.src(), u_pos);
        let u_bases = graph.infer_pipe_endpoint_face_bases(pipe, u_pos);
        let v_bases = graph.infer_pipe_endpoint_face_bases(pipe, v_pos);
        let delta = u_pos - v_pos;
        for ((d, u_basis), v_basis) in delta.to_array().into_iter().zip(u_bases).zip(v_bases) {
            if d == 0
                && let (Some(u_basis), Some(v_basis)) = (u_basis, v_basis)
                && (u_basis != v_basis) ^ pipe.hadamard
            {
                errors.push(InvalidBlockGraphError::OrientationContradiction(u_pos, v_pos).into());
                break;
            }
        }
    }
}

/// One pipe's physical endpoints and owner blocks, as yielded by
/// [`BlockGraph::pipe_endpoints_with_blocks`].
type PipeIncidence<'a> = (IVec3, IVec3, &'a Block, &'a Block, &'a Pipe);

fn collect_block_constraint_errors(graph: &BlockGraph, errors: &mut Vec<BlockGraphError>) {
    // Index pipes by owner block once, keeping validation O(V+E) and preserving
    // pipe order within each block.
    let mut incidence: HashMap<IVec3, Vec<PipeIncidence>> = HashMap::new();
    for tuple in graph.pipe_endpoints_with_blocks() {
        let (_, _, u, v, _) = tuple;
        incidence.entry(u.pos()).or_default().push(tuple);
        if v.pos() != u.pos() {
            incidence.entry(v.pos()).or_default().push(tuple);
        }
    }

    // A failing block stops its own checks and the scan moves on to the next
    // block, rather than abandoning the rest of the graph.
    for block in graph.blocks() {
        let neighbors = graph.neighbor_positions(block.pos());
        let incident: &[PipeIncidence] = match incidence.get(&block.pos()) {
            Some(pipes) => pipes,
            None => &[],
        };
        match block.kind {
            BlockKind::Port
            | BlockKind::Y
            | BlockKind::Measurement(_)
            | BlockKind::T
            | BlockKind::Selective(_) => {
                if neighbors.len() != 1 {
                    errors.push(InvalidBlockGraphError::NotExactlySinglePipe(block.pos()).into());
                    continue;
                }
                let has_spatial_pipe = neighbors
                    .iter()
                    .any(|&pos| (pos - block.pos()).abs() != IVec3::Z);
                if block.kind == BlockKind::Port {
                    let role = block.port_role().expect("Port blocks expose a role");
                    if has_spatial_pipe && role == crate::PortRole::Auto {
                        errors.push(
                            InvalidBlockGraphError::SpatialPortRoleRequired(block.pos()).into(),
                        );
                    }
                    if !has_spatial_pipe && role == crate::PortRole::Multiplex {
                        let inferred = if neighbors[0].z > block.pos().z {
                            crate::PortRole::Input
                        } else {
                            crate::PortRole::Output
                        };
                        errors.push(
                            InvalidBlockGraphError::TemporalPortRoleMismatch {
                                pos: block.pos(),
                                declared: role,
                                inferred,
                            }
                            .into(),
                        );
                        continue;
                    }
                }
                let spatial_allowed = block.kind == BlockKind::Port;
                if has_spatial_pipe && !spatial_allowed {
                    errors.push(InvalidBlockGraphError::SpatialPipeNotAllowed(block.pos()).into());
                    continue;
                }
                if matches!(block.kind, BlockKind::T)
                    && neighbors.iter().any(|&pos| pos - block.pos() != IVec3::Z)
                {
                    // A T block injects into the future, so its single pipe must
                    // point at +Z; a past-facing pipe is the violation here.
                    errors.push(InvalidBlockGraphError::PastPipeNotAllowed(block.pos()).into());
                    continue;
                }
                if matches!(
                    block.kind,
                    BlockKind::Measurement(_) | BlockKind::Selective(_)
                ) && neighbors
                    .iter()
                    .any(|&pos| pos - block.pos() != IVec3::NEG_Z)
                {
                    errors.push(InvalidBlockGraphError::FuturePipeNotAllowed(block.pos()).into());
                    continue;
                }
            }
            BlockKind::Cube(ck) => {
                let height_errors = errors.len();
                collect_cube_height_errors(block, incident, errors);
                if errors.len() > height_errors {
                    continue;
                }
                let cube_neighbors = cube_endpoint_neighbors(block, incident);
                // A pipe face is defined by the two axes perpendicular to the pipe
                // direction. If both perpendicular axes have the same basis, the
                // lattice surgery orientation is degenerate (e.g. XZZ cube with a
                // pipe along +X has Z on both Y and Z faces).
                for (endpoint, delta) in &cube_neighbors {
                    let diff = (*delta).abs();
                    let pipe_axis = diff
                        .to_array()
                        .iter()
                        .position(|&d| d != 0)
                        .expect("neighbor differs on at least one axis");
                    let perp: [usize; 2] = match pipe_axis {
                        0 => [1, 2],
                        1 => [0, 2],
                        _ => [0, 1],
                    };
                    // Each face is independent, so report them all: a cube that
                    // already has one broken face must still report a second one
                    // an edit adds, or a before/after diff would not see it.
                    if ck.bases()[perp[0]] == ck.bases()[perp[1]] {
                        errors.push(
                            InvalidBlockGraphError::DegeneratePipeFace(
                                block.pos(),
                                *endpoint + *delta,
                            )
                            .into(),
                        );
                    }
                }

                if cube_neighbors
                    .into_iter()
                    .map(|(_, delta)| delta.abs())
                    .collect::<HashSet<_>>()
                    .len()
                    == 3
                {
                    errors.push(InvalidBlockGraphError::JunctionNotPlanar(block.pos()).into());
                }
            }
            BlockKind::Walking(kind) => {
                let start = block.pos();
                let end = kind.end_position(start);
                for &(u_pos, v_pos, u, v, _) in incident {
                    let (endpoint, other_pos) = if u.pos() == block.pos() {
                        (u_pos, v_pos)
                    } else if v.pos() == block.pos() {
                        (v_pos, u_pos)
                    } else {
                        continue;
                    };
                    // One violation per incident pipe, then on to the next pipe:
                    // an existing bad pipe must not mask a newly added one.
                    let delta = other_pos - endpoint;
                    if delta.abs() != IVec3::Z {
                        errors.push(
                            InvalidBlockGraphError::WalkingPipeNotTemporal {
                                pos: start,
                                endpoint,
                            }
                            .into(),
                        );
                        continue;
                    }
                    if endpoint == start && delta != IVec3::NEG_Z {
                        errors.push(
                            InvalidBlockGraphError::WalkingStartPipeNotPast { pos: start }.into(),
                        );
                        continue;
                    }
                    if endpoint == end && delta != IVec3::Z {
                        errors.push(
                            InvalidBlockGraphError::WalkingEndPipeNotFuture { pos: start }.into(),
                        );
                        continue;
                    }
                    if endpoint != start && endpoint != end {
                        errors.push(
                            InvalidBlockGraphError::WalkingPipeNotTemporal {
                                pos: start,
                                endpoint,
                            }
                            .into(),
                        );
                        continue;
                    }
                }
            }
            BlockKind::PatchRotation(kind) => {
                collect_patch_rotation_errors(graph, block.pos(), kind, incident, errors);
            }
        }
    }
}

fn cube_endpoint_neighbors(block: &Block, incident: &[PipeIncidence]) -> Vec<(IVec3, IVec3)> {
    incident
        .iter()
        .filter_map(|&(u_pos, v_pos, u, v, _)| {
            if u.pos() == block.pos() {
                Some((u_pos, v_pos - u_pos))
            } else if v.pos() == block.pos() {
                Some((v_pos, u_pos - v_pos))
            } else {
                None
            }
        })
        .collect()
}

/// Reports each endpoint of the rotation independently, so a half-wired block
/// names both of its missing pipes rather than only the first: an editor adding
/// them one at a time needs the untouched endpoint to keep reporting the same
/// violation it already had.
fn collect_patch_rotation_errors(
    graph: &BlockGraph,
    start: IVec3,
    kind: PatchRotationKind,
    incident: &[PipeIncidence],
    errors: &mut Vec<BlockGraphError>,
) {
    let end = kind.end_position(start);
    for endpoint in [start, end] {
        let pipes = pipes_at_endpoint(incident, start, endpoint);
        if pipes.len() != 1 {
            errors.push(
                InvalidBlockGraphError::PatchRotationEndpointPipeCount {
                    pos: start,
                    endpoint,
                    count: pipes.len(),
                }
                .into(),
            );
            continue;
        }
        let pipe = pipes[0];
        let neighbor = pipe_neighbor_at(pipe, endpoint)
            .expect("pipe selected for endpoint must include that endpoint");
        if (neighbor - endpoint).abs() != IVec3::Z {
            errors.push(
                InvalidBlockGraphError::PatchRotationPipeNotTemporal {
                    pos: start,
                    endpoint,
                }
                .into(),
            );
            continue;
        }
        if endpoint == start && pipe.is_hadamard() {
            errors
                .push(InvalidBlockGraphError::PatchRotationStartPipeHadamard { pos: start }.into());
            continue;
        }
        let expected_basis = kind
            .pipe_face_bases_at_endpoint(start, endpoint)
            .expect("patch rotation endpoint was checked above")[UDirection::X.index()];
        let actual_basis = pipe_neighbor_basis_at_endpoint(graph, pipe, endpoint);
        if actual_basis.is_some_and(|actual| (actual != expected_basis) ^ pipe.is_hadamard()) {
            errors.push(
                InvalidBlockGraphError::PatchRotationBoundaryMismatch {
                    pos: start,
                    endpoint,
                    expected: expected_basis,
                    actual: actual_basis,
                }
                .into(),
            );
        }
    }
}

fn pipe_neighbor_basis_at_endpoint(
    graph: &BlockGraph,
    pipe: &Pipe,
    endpoint: IVec3,
) -> Option<Basis> {
    let neighbor_pos = pipe_neighbor_at(pipe, endpoint)?;
    graph.infer_pipe_endpoint_face_bases(pipe, neighbor_pos)[UDirection::X.index()]
}

/// Collects the two rules a cube's height imposes on its pipes.
///
/// Height agreement is checked pairwise across spacelike pipes, which is
/// equivalent to the component-wide statement `parser::lower` propagates:
/// spatial components are exactly the transitive closure of spacelike pipes, so
/// pairwise equality along every edge is component-wide equality. Per-edge is
/// also what lets the message name the offending *pair*.
fn collect_cube_height_errors(
    block: &Block,
    incident: &[PipeIncidence],
    errors: &mut Vec<BlockGraphError>,
) {
    let height = block.height();
    let cells = block.height_cells();
    let top = block.pos() + IVec3::new(0, 0, cells as i32 - 1);

    for &(u_pos, v_pos, u, v, _) in incident {
        let (endpoint, other_endpoint, other_block) = if u.pos() == block.pos() {
            (u_pos, v_pos, v)
        } else if v.pos() == block.pos() {
            (v_pos, u_pos, u)
        } else {
            continue;
        };
        let delta = other_endpoint - endpoint;

        if delta.z == 0 {
            if cells > 1 && endpoint != block.pos() {
                errors.push(
                    InvalidBlockGraphError::CubeInvalidPipeEndpoint {
                        pos: block.pos(),
                        endpoint,
                    }
                    .into(),
                );
                continue;
            }
            if other_block.kind.is_cube() && other_block.height() != height {
                errors.push(
                    InvalidBlockGraphError::CubeHeightSpatialMismatch {
                        pos: block.pos(),
                        neighbor: other_block.pos(),
                        height,
                        neighbor_height: other_block.height(),
                    }
                    .into(),
                );
                continue;
            }
        } else if cells > 1 {
            let valid_temporal_endpoint = (endpoint == block.pos() && delta == IVec3::NEG_Z)
                || (endpoint == top && delta == IVec3::Z);
            if !valid_temporal_endpoint {
                errors.push(
                    InvalidBlockGraphError::CubeInvalidPipeEndpoint {
                        pos: block.pos(),
                        endpoint,
                    }
                    .into(),
                );
            }
        }
    }
}

fn pipes_at_endpoint<'a>(
    incident: &[PipeIncidence<'a>],
    owner_pos: IVec3,
    endpoint: IVec3,
) -> Vec<&'a Pipe> {
    incident
        .iter()
        .filter_map(|&(u_pos, v_pos, u, v, pipe)| {
            let touches_endpoint = (u.pos() == owner_pos && u_pos == endpoint)
                || (v.pos() == owner_pos && v_pos == endpoint);
            touches_endpoint.then_some(pipe)
        })
        .collect()
}

fn pipe_neighbor_at(pipe: &Pipe, endpoint: IVec3) -> Option<IVec3> {
    if pipe.src() == endpoint {
        Some(pipe.dst())
    } else if pipe.dst() == endpoint {
        Some(pipe.src())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, FeedbackTarget, PauliBasis};
    use crate::{CubeKind, Expr, SelectiveKind, WalkingBoundaryKind, WalkingKind};
    use glam::IVec2;

    #[test]
    fn structural_errors_lists_every_offending_block() {
        let mut graph = BlockGraph::new();
        // Two independent lone ports, each missing its single pipe.
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(4, 0, 0), BlockKind::Port));

        let errors = structural_errors(&graph);

        assert_eq!(errors.len(), 2, "{errors:?}");
        assert_eq!(
            validate_structure(&graph)
                .expect_err("graph is invalid")
                .to_string(),
            errors[0].to_string(),
            "fail-fast validation reports the first collected error"
        );
    }

    /// Helper to create a `Measure` action with a dummy block target.
    fn measurement(name: &str) -> Action {
        Action::Measure {
            target: MeasureTarget::Node(IVec3::ZERO),
            name: name.to_string(),
        }
    }

    #[test]
    fn resolve_must_target_selective_block() {
        // Resolve targeting a cube block → rejected during graph validation.
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.set_actions_unchecked(vec![
            Action::Measure {
                target: MeasureTarget::Node(IVec3::new(1, 0, 0)),
                name: "m".to_string(),
            },
            Action::Resolve {
                target: IVec3::new(0, 0, 0),
                condition: Expr::Var("m".to_string()),
            },
        ]);
        let err = validate(&g).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::InvalidResolveTarget(..))
            ),
            "expected InvalidResolveTarget, got: {err:?}",
        );
    }

    #[test]
    fn validate_rechecks_dag_actions_after_graph_mutation() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            glam::ivec3(2, 0, 0),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_block(Block::new(
            glam::ivec3(2, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(glam::ivec3(0, 0, 0), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(glam::ivec3(2, 0, -1), Direction::ZPLUS));
        graph.set_actions_unchecked(vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 1)),
                name: "m".to_string(),
            },
            Action::Resolve {
                target: glam::ivec3(2, 0, 0),
                condition: Expr::Var("m".to_string()),
            },
        ]);
        let err = validate(&graph);
        assert!(err.is_ok(), "{err:?}");

        graph.add_block(Block::new(
            glam::ivec3(4, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            glam::ivec3(4, 0, 0),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_pipe(Pipe::new(glam::ivec3(4, 0, -1), Direction::ZPLUS));

        let err = graph.validate().unwrap_err();
        assert!(matches!(
            err,
            BlockGraphError::InvalidAction(InvalidActionError::MissingResolveForSelective {
                target
            }) if target == glam::ivec3(4, 0, 0)
        ));
    }

    #[test]
    fn validate_rejects_dependency_cycles_from_action_dag() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_pipe(Pipe::new(glam::ivec3(0, 0, -1), Direction::ZPLUS));
        graph.set_actions_unchecked(vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m".into(),
            },
            Action::Resolve {
                target: glam::ivec3(0, 0, 0),
                condition: Expr::Var("m".into()),
            },
        ]);
        let err = validate(&graph).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::DependencyCycle { .. })
            ),
            "expected DependencyCycle, got: {err:?}",
        );
    }

    #[test]
    fn boundary_nodes_require_exactly_one_pipe() {
        for kind in [
            BlockKind::Port,
            BlockKind::Y,
            BlockKind::Measurement(Basis::X),
            BlockKind::T,
            BlockKind::Selective(SelectiveKind::XY),
        ] {
            let mut g = BlockGraph::new();
            g.add_block(Block::new(IVec3::new(0, 0, 0), kind));

            let err = validate(&g).unwrap_err();
            assert!(
                matches!(
                    err,
                    BlockGraphError::Invalid(InvalidBlockGraphError::NotExactlySinglePipe(_))
                ),
                "expected NotExactlySinglePipe for {kind:?}, got: {err:?}",
            );
        }
    }

    #[test]
    fn port_roles_are_independent_of_temporal_direction() {
        let mut spatial = BlockGraph::new();
        spatial.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        spatial.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        spatial.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        assert!(matches!(
            validate(&spatial),
            Err(BlockGraphError::Invalid(
                InvalidBlockGraphError::SpatialPortRoleRequired(IVec3::ZERO)
            ))
        ));
        for role in [
            crate::PortRole::Input,
            crate::PortRole::Output,
            crate::PortRole::Multiplex,
        ] {
            spatial.set_port_role(IVec3::ZERO, role).unwrap();
            validate(&spatial).expect("directed spatial Port validates");
        }

        let mut temporal = BlockGraph::new();
        temporal.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        temporal.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        temporal.add_block(Block::new(2 * IVec3::Z, BlockKind::Port));
        temporal.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        temporal.add_pipe(Pipe::new(IVec3::Z, Direction::ZPLUS));
        for port in [IVec3::ZERO, 2 * IVec3::Z] {
            for role in [
                crate::PortRole::Auto,
                crate::PortRole::Input,
                crate::PortRole::Output,
            ] {
                temporal.set_port_role(port, role).unwrap();
                validate(&temporal).expect("either temporal direction accepts the declared role");
            }
            temporal
                .set_port_role(port, crate::PortRole::Multiplex)
                .unwrap();
            assert!(matches!(
                validate(&temporal),
                Err(BlockGraphError::Invalid(
                    InvalidBlockGraphError::TemporalPortRoleMismatch { .. }
                ))
            ));
            temporal.set_port_role(port, crate::PortRole::Auto).unwrap();
        }
    }

    #[test]
    fn walking_blocks_may_be_unconnected() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::X).unwrap()),
        ));

        validate(&graph).unwrap();
    }

    #[test]
    fn walking_soft_corridor_allows_same_start_layer_ports() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
        ));

        graph
            .try_add_block(
                Block::new(IVec3::new(1, 0, 0), BlockKind::Port)
                    .with_port_role(crate::PortRole::Input)
                    .unwrap(),
            )
            .expect("same-start-layer port may share walking soft corridor");
        graph.add_block(Block::new(
            IVec3::new(2, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 0), Direction::XPLUS));

        validate(&graph).unwrap();
    }

    #[test]
    fn walking_soft_corridor_rejects_upper_layer_ports() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
        ));

        let err = graph
            .try_add_block(Block::new(IVec3::new(1, 0, 1), BlockKind::Port))
            .expect_err("upper-layer port cannot share walking soft corridor");

        assert!(matches!(
            err,
            BlockGraphError::BlockPositionOccupied(pos) if pos == IVec3::new(1, 0, 1)
        ));
    }

    #[test]
    fn walking_soft_corridor_allows_parallel_same_layer_walking() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
        ));

        graph
            .try_add_block(Block::new(
                IVec3::Y,
                BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::XZX, IVec2::ONE).unwrap()),
            ))
            .expect("parallel same-layer walking blocks may share soft corridors");

        validate(&graph).unwrap();
    }

    #[test]
    fn walking_soft_corridor_rejects_nonparallel_walking() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
        ));

        let err = graph
            .try_add_block(Block::new(
                IVec3::X,
                BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::Y).unwrap()),
            ))
            .expect_err("nonparallel walking blocks cannot share soft corridors");

        assert!(matches!(
            err,
            BlockGraphError::BlockPositionOccupied(pos) if pos == IVec3::new(1, 0, 0)
        ));
    }

    #[test]
    fn walking_soft_corridor_rejects_different_start_layer_walking() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
        ));

        let err = graph
            .try_add_block(Block::new(
                IVec3::new(1, 0, 1),
                BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap()),
            ))
            .expect_err("different-start-layer walking blocks cannot share soft corridors");

        assert!(matches!(
            err,
            BlockGraphError::BlockPositionOccupied(pos) if pos == IVec3::new(1, 0, 1)
        ));
    }

    #[test]
    fn walking_start_accepts_past_temporal_pipe() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::X).unwrap()),
        ));
        graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS));

        validate(&graph).unwrap();
    }

    #[test]
    fn walking_end_accepts_future_temporal_pipe() {
        let mut graph = BlockGraph::new();
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::X).expect("valid movement");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Walking(walking)));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(
            walking.end_position(IVec3::ZERO),
            Direction::ZPLUS,
        ));

        validate(&graph).unwrap();
    }

    #[test]
    fn walking_rejects_spatial_pipe_at_endpoint() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::X).unwrap()),
        ));
        graph.add_block(Block::new(IVec3::NEG_X, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XMINUS));

        let err = validate(&graph).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::WalkingPipeNotTemporal {
                pos,
                endpoint
            }) if pos == IVec3::ZERO && endpoint == IVec3::ZERO
        ));
    }

    #[test]
    fn walking_participates_in_lattice_surgery_orientation_checks() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(WalkingKind::new(WalkingBoundaryKind::XZZ, IVec2::X).unwrap()),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 2),
            BlockKind::Cube(CubeKind::ZXX),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 1), Direction::ZPLUS));

        let err = validate(&graph).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::OrientationContradiction(a, b))
                if a == IVec3::new(1, 0, 1) && b == IVec3::new(1, 0, 2)
        ));
    }

    #[test]
    fn patch_rotation_requires_temporal_pipes_at_start_and_end() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, IVec2::X).unwrap();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));

        let err = validate(&graph).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::PatchRotationEndpointPipeCount {
                pos,
                endpoint,
                count: 0
            }) if pos == IVec3::ZERO && endpoint == IVec3::ZERO
        ));
    }

    #[test]
    fn patch_rotation_allows_hadamard_only_at_end() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, IVec2::X).unwrap();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(IVec3::ZERO), Direction::ZPLUS).with_hadamard());

        validate(&graph).unwrap();

        graph
            .set_pipe_hadamard(IVec3::ZERO, IVec3::NEG_Z, true)
            .unwrap();
        assert!(matches!(
            validate(&graph),
            Err(BlockGraphError::Invalid(
                InvalidBlockGraphError::PatchRotationStartPipeHadamard { pos }
            )) if pos == IVec3::ZERO
        ));
    }

    #[test]
    fn patch_rotation_rejects_spatial_pipe_at_endpoint() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, IVec2::X).unwrap();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));
        graph.add_block(Block::new(IVec3::NEG_X, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(IVec3::ZERO), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XMINUS));

        let err = validate(&graph).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::PatchRotationEndpointPipeCount {
                pos,
                endpoint,
                count: 2
            }) if pos == IVec3::ZERO && endpoint == IVec3::ZERO
        ));
    }

    #[test]
    fn patch_rotation_rejects_mismatched_start_boundary_orientation() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, IVec2::X).unwrap();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::XXZ)));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(IVec3::ZERO), Direction::ZPLUS));

        let err = validate(&graph).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::OrientationContradiction(a, b))
                if a == IVec3::ZERO && b == IVec3::NEG_Z
        ));
    }

    #[test]
    fn spatially_connected_cubes_must_have_matching_height() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(
            Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        assert!(matches!(
            graph.validate_structure(),
            Err(BlockGraphError::Invalid(
                InvalidBlockGraphError::CubeHeightSpatialMismatch { .. }
            ))
        ));
    }

    #[test]
    fn multi_cell_cube_spatial_pipe_must_attach_at_anchor() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(
            Block::new(IVec3::new(1, 0, 1), BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 1), Direction::XPLUS));

        assert!(matches!(
            graph.validate_structure(),
            Err(BlockGraphError::Invalid(
                InvalidBlockGraphError::CubeInvalidPipeEndpoint { .. }
            ))
        ));
    }

    #[test]
    fn multi_cell_cube_allows_bottom_past_and_top_future_temporal_pipes() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(Block::new(
            IVec3::new(0, 0, 3),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 2), Direction::ZPLUS));

        graph
            .validate_structure()
            .expect("scaled cube temporal endpoints should validate");
    }

    #[test]
    fn resolve_alias_defined_later_than_raw_measurement_is_accepted() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 2),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        g.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_pipe(Pipe::new(IVec3::new(0, 0, 2), Direction::ZMINUS));
        g.set_actions_unchecked(vec![
            measurement("m"),
            Action::Let {
                name: "alias".to_string(),
                expr: Expr::Var("m".to_string()),
            },
            Action::Resolve {
                target: IVec3::new(0, 0, 2),
                condition: Expr::Var("alias".to_string()),
            },
        ]);

        validate(&g).unwrap();
    }

    #[test]
    fn empty_feedback_is_rejected_atomically_by_every_source_install() {
        for install in [
            BlockGraph::set_actions,
            BlockGraph::set_actions_deferred,
            BlockGraph::set_actions_lenient,
        ] {
            let mut graph =
                BlockGraph::from_blog_text("BLOG 1.0\n0: ZXZ [0,0,0]\nm = measure 0\n").unwrap();
            let previous = graph.actions();
            let error = install(
                &mut graph,
                vec![Action::Feedback {
                    targets: vec![],
                    condition: None,
                }],
            )
            .unwrap_err();
            assert!(matches!(
                error,
                BlockGraphError::InvalidAction(InvalidActionError::EmptyFeedbackTargets)
            ));
            assert_eq!(graph.actions(), previous);
        }
    }

    #[test]
    fn feedback_accepts_a_future_measurement_binding() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        g.add_pipe(crate::Pipe::new(IVec3::ZERO, crate::Direction::ZPLUS));
        g.set_actions_unchecked(vec![
            Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::Z,
                    target: IVec3::ZERO,
                    direction: None,
                }],
                condition: Some(Expr::Var("m".to_string())),
            },
            measurement("m"),
        ]);

        validate(&g).unwrap();
        assert_eq!(
            BlockGraph::from_blog_text(&g.to_blog_text())
                .unwrap()
                .actions(),
            g.actions()
        );
        g.set_actions(vec![Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::Z,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: None,
        }])
        .unwrap();
        assert_eq!(
            BlockGraph::from_blog_text(&g.to_blog_text())
                .unwrap()
                .actions(),
            g.actions()
        );
    }

    #[test]
    fn discard_if_accepts_a_measurement_binding() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_action(measurement("m")).unwrap();
        g.add_action(Action::DiscardIf(Expr::Var("m".to_string())))
            .unwrap();

        validate(&g).unwrap();
    }

    #[test]
    fn feedback_rejects_undefined_variables() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.set_actions_unchecked(vec![Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::Z,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: Some(Expr::Var("missing".to_string())),
        }]);
        let err = validate(&g).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::UndefinedVariable(_))
            ),
            "expected UndefinedVariable, got: {err:?}",
        );
    }

    #[test]
    fn duplicate_measure_pipe_from_both_directions_is_rejected() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::XPLUS));
        g.set_actions_unchecked(vec![
            Action::Measure {
                target: MeasureTarget::Edge {
                    src: IVec3::new(0, 0, 0),
                    dir: Direction::XPLUS,
                },
                name: "m0".into(),
            },
            Action::Measure {
                target: MeasureTarget::Edge {
                    src: IVec3::new(1, 0, 0),
                    dir: Direction::XMINUS,
                },
                name: "m1".into(),
            },
        ]);

        let err = validate(&g).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::DuplicateMeasurementTarget(..))
            ),
            "expected DuplicateMeasurementTarget, got: {err:?}",
        );
    }

    #[test]
    fn terminal_measurements_with_future_pipes_are_rejected() {
        for kind in [
            BlockKind::Measurement(Basis::X),
            BlockKind::Selective(SelectiveKind::XY),
        ] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, kind));
            graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

            let err = validate(&graph).unwrap_err();
            assert!(matches!(
                err,
                BlockGraphError::Invalid(InvalidBlockGraphError::FuturePipeNotAllowed(pos))
                    if pos == IVec3::ZERO
            ));
        }
    }

    #[test]
    fn fixed_measurement_accepts_one_past_pipe() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Measurement(Basis::X)));
        graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS));

        validate(&graph).unwrap();
    }

    #[test]
    fn feedback_rejects_missing_targets() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.set_actions_unchecked(vec![Action::Feedback {
            targets: vec![crate::FeedbackTarget {
                pauli: crate::PauliBasis::Z,
                target: IVec3::new(1, 0, 0),
                direction: None,
            }],
            condition: None,
        }]);

        let err = validate(&g).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::InvalidFeedbackTarget(..))
            ),
            "expected InvalidFeedbackTarget, got: {err:?}",
        );
    }

    #[test]
    fn feedback_accepts_output_port_targets() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(IVec3::new(0, 0, 1), BlockKind::Port));
        g.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS));
        g.add_action(Action::Feedback {
            targets: vec![crate::FeedbackTarget {
                pauli: crate::PauliBasis::Z,
                target: IVec3::new(0, 0, 1),
                direction: None,
            }],
            condition: None,
        })
        .unwrap();

        validate(&g).unwrap();
    }

    #[test]
    fn feedback_accepts_existing_non_output_targets() {
        let mut g = BlockGraph::new();
        g.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        g.add_block(Block::new(IVec3::new(0, 0, 1), BlockKind::Port));
        g.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS));
        g.set_actions_unchecked(vec![Action::Feedback {
            targets: vec![crate::FeedbackTarget {
                pauli: crate::PauliBasis::Z,
                target: IVec3::new(0, 0, 0),
                direction: None,
            }],
            condition: None,
        }]);

        validate(&g).unwrap();
    }

    #[test]
    fn set_actions_rejects_measure_name_outside_identifier_grammar() {
        // A space cannot appear in a BLOG identifier, so the writer would emit a
        // name that fails to reparse — set_actions must reject it up front.
        let mut g = BlockGraph::new();
        g.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let err = g
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(IVec3::ZERO),
                name: "bad name".to_string(),
            }])
            .expect_err("a name with a space cannot round-trip through to_blog_text");
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::InvalidActionName(_))
            ),
            "expected InvalidActionName, got: {err:?}",
        );
    }

    #[test]
    fn set_actions_rejects_reserved_keyword_name() {
        // A reserved keyword re-emitted verbatim would be reparsed as a
        // keyword, not a name.
        let mut g = BlockGraph::new();
        g.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let err = g
            .set_actions(vec![Action::Let {
                name: "measure".to_string(),
                expr: Expr::Var("m".to_string()),
            }])
            .expect_err("a reserved keyword name would not reparse");
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::InvalidActionName(_))
            ),
            "expected InvalidActionName, got: {err:?}",
        );
    }

    #[test]
    fn set_actions_accepts_valid_name_and_round_trips() {
        let graph =
            crate::parse_blog_to_graph("BLOG 1.0\n\n  0: ZXZ [0, 0, 0]\n\n  m0 = measure 0\n")
                .expect("valid name parses");
        let text = graph.to_blog_text();
        let reparsed = crate::parse_blog_to_graph(&text).expect("round-trip reparses");
        assert_eq!(reparsed.to_blog_text(), text);
    }
}
