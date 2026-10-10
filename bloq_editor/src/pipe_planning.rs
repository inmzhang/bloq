//! Plans a single pipe placement before mutating the graph: validates the
//! endpoints, materializes or promotes endpoint blocks as needed, and probes
//! the resulting graph so the edit is only applied when it stays valid.
//!
//! Planning is separated from mutation so the same check drives both the live
//! placement and the preview hints, and so a rejected placement leaves the
//! graph untouched.

use std::cmp::Reverse;
use std::collections::HashSet;

use bloq_graph::{
    Block, BlockGraph, BlockGraphError, BlockKind, Direction, InvalidBlockGraphError, Pipe,
};
use glam::IVec3;

/// Why a pipe placement was rejected.
///
/// The messages surface directly in editor toasts, so they are written for
/// users, not debuggers.
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum PipePlanError {
    #[error("pipe endpoints {src} and {dst} must be distinct and adjacent")]
    InvalidEndpoints { src: IVec3, dst: IVec3 },
    #[error("no block at {0} to attach the pipe to")]
    EndpointMissing(IVec3),
    #[error("a pipe already exists between {src} and {dst}")]
    PipeAlreadyExists { src: IVec3, dst: IVec3 },
    #[error("the {kind} block at {pos} cannot take another pipe")]
    EndpointFull { pos: IVec3, kind: BlockKind },
    #[error("port promotion at {0} conflicts with another endpoint's promotion")]
    PortPromotionConflict(IVec3),
    #[error(
        "cube at {pos} spanning {cells} cells needs a cube of the same span at {other} for a \
         spatial pipe"
    )]
    TallCubeSpatialPipeRequiresMatchingCube {
        pos: IVec3,
        cells: u32,
        other: IVec3,
    },
    #[error(
        "spatial pipe endpoints {src} and {dst} have different cube heights (height={src_height} vs \
         height={dst_height})"
    )]
    SpatialCubeHeightMismatch {
        src: IVec3,
        dst: IVec3,
        src_height: bloq_graph::CubeHeight,
        dst_height: bloq_graph::CubeHeight,
    },
    #[error("{0}")]
    Graph(#[from] BlockGraphError),
}

/// A block change a placement plan must apply at an endpoint before the pipe
/// can be added: create a fresh port, or set an endpoint's kind — promoting a
/// port to a cube, or relabelling a cube whose shadowed faces let it. A pipe
/// into an empty cell does both when a bare port cannot hold it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndpointChange {
    CreatePort { pos: IVec3 },
    SetKind { pos: IVec3, kind: BlockKind },
}

/// A request to place a pipe between two adjacent cells, optionally Hadamard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipePlacementRequest {
    pub(crate) src: IVec3,
    pub(crate) dst: IVec3,
    pub(crate) hadamard: bool,
}

/// A validated pipe placement: the pipe to add plus the endpoint block changes
/// it depends on. Produced by [`can_place_pipe`] and applied atomically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipePlacementPlan {
    pub(crate) pipe: Pipe,
    pub(crate) endpoint_changes: Vec<EndpointChange>,
}

fn candidate_cube_kind_for_endpoint(
    graph: &BlockGraph,
    pos: IVec3,
    extra_pipe: &Pipe,
) -> Option<BlockKind> {
    if !extra_pipe.dir().is_spatial()
        && let Some(kind) = graph.infer_spatial_port_cube_kind(pos)
    {
        return Some(BlockKind::Cube(kind));
    }
    BlockKind::all_kinds()
        .into_iter()
        .enumerate()
        .filter(|(_, kind)| matches!(kind, BlockKind::Cube(_)))
        .filter(|(_, kind)| graph_valid_with_endpoint_kinds(graph, &[pos], &[*kind], extra_pipe))
        .max_by_key(|(index, kind)| {
            (
                promotion_basis_preservation_score(graph, pos, *kind, extra_pipe),
                Reverse(*index),
            )
        })
        .map(|(_, kind)| kind)
}

fn can_promote_port(
    graph: &BlockGraph,
    pos: IVec3,
    extra_pipe: &Pipe,
) -> Result<EndpointChange, PipePlanError> {
    let Some(kind) = candidate_cube_kind_for_endpoint(graph, pos, extra_pipe) else {
        return Err(PipePlanError::PortPromotionConflict(pos));
    };
    Ok(EndpointChange::SetKind { pos, kind })
}

/// Plans a pipe placement without mutating `graph`.
///
/// Checks adjacency, duplicate pipes, and endpoint capacity; materializes or
/// promotes endpoint blocks; and probes the resulting graph so the returned
/// plan is guaranteed to apply cleanly.
///
/// # Errors
///
/// Returns the [`PipePlanError`] describing the first condition that makes the
/// placement invalid.
pub(crate) fn can_place_pipe(
    graph: &BlockGraph,
    request: PipePlacementRequest,
) -> Result<PipePlacementPlan, PipePlanError> {
    if request
        .src
        .as_i64vec3()
        .manhattan_distance(request.dst.as_i64vec3())
        != 1
    {
        return Err(PipePlanError::InvalidEndpoints {
            src: request.src,
            dst: request.dst,
        });
    }
    if graph.has_pipe_between(request.src, request.dst) {
        return Err(PipePlanError::PipeAlreadyExists {
            src: request.src,
            dst: request.dst,
        });
    }
    if !graph.has_endpoint_at(request.src) && !graph.has_endpoint_at(request.dst) {
        return Err(PipePlanError::EndpointMissing(request.src));
    }

    let direction = Direction::try_from(request.dst - request.src)
        .expect("adjacent pipe endpoints have a cardinal direction");
    let pipe = if request.hadamard {
        Pipe::new(request.src, direction).with_hadamard()
    } else {
        Pipe::new(request.src, direction)
    };
    reject_spatial_cube_scale_mismatch(graph, request.src, request.dst, direction)?;

    let mut created_ports = Vec::new();
    let mut scratch = graph.clone();
    for pos in [request.src, request.dst] {
        if !scratch.has_endpoint_at(pos) {
            scratch.try_add_block(Block::new(pos, BlockKind::Port))?;
            created_ports.push(pos);
        }
    }

    let mut promotion_positions = Vec::new();
    for pos in [request.src, request.dst] {
        let block = scratch
            .get_endpoint_block(pos)
            .ok_or(PipePlanError::EndpointMissing(pos))?;
        let degree_before = graph.degree(pos);
        match block.kind() {
            BlockKind::Cube(_) | BlockKind::Walking(_) | BlockKind::PatchRotation(_) => {}
            BlockKind::Y | BlockKind::Measurement(_) | BlockKind::T | BlockKind::Port
                if degree_before == 0 => {}
            BlockKind::Port if degree_before == 1 && block.pos() == pos => {
                promotion_positions.push(pos);
            }
            kind => {
                return Err(PipePlanError::EndpointFull { pos, kind });
            }
        }
    }
    let plan_with = |promotions: Vec<EndpointChange>| PipePlacementPlan {
        pipe: pipe.clone(),
        endpoint_changes: created_ports
            .iter()
            .map(|&pos| EndpointChange::CreatePort { pos })
            .chain(promotions)
            .collect(),
    };

    let err = match endpoint_promotions(&scratch, &promotion_positions, &pipe)
        .map(plan_with)
        .and_then(|plan| probe_plan(graph, &plan).map(|()| plan))
    {
        Ok(plan) => return Ok(plan),
        Err(err) => err,
    };

    // The faces an axis's pipes cover are shadowed: their basis never reaches the
    // compiled program, which is why `fix_shadowed_faces` rewrites them on the way
    // in. Adopting that rewrite here lets a pass-through cube take a pipe on the
    // axis its stale label happened to reserve — a `ZZX` run along x accepting a
    // temporal pipe as `XZX` — instead of making the user relabel it by hand.
    if let Some(plan) = shadow_relabels(graph, &scratch, &pipe, &promotion_positions)
        .map(plan_with)
        .filter(|plan| probe_plan(graph, plan).is_ok())
    {
        return Ok(plan);
    }

    // If the new Port plan fails for another structural reason, retry with it
    // promoted through the same search as the other endpoint. Keep the original
    // error when no cube assignment fits.
    if created_ports.is_empty() {
        return Err(err);
    }
    let promotions = [promotion_positions, created_ports.clone()].concat();
    let Ok(plan) = endpoint_promotions(&scratch, &promotions, &pipe).map(plan_with) else {
        return Err(err);
    };
    probe_plan(graph, &plan).map(|()| plan).map_err(|_| err)
}

/// Relabels the pipe's existing cube endpoints to the kinds `fix_shadowed_faces`
/// gives them once the pipe is in place, then re-runs the port promotions against
/// those kinds so both ends of the new pipe still agree.
///
/// Returns `None` when no endpoint changes label, so the caller keeps the
/// original rejection rather than reporting a relabel that changed nothing.
fn shadow_relabels(
    graph: &BlockGraph,
    scratch: &BlockGraph,
    pipe: &Pipe,
    promotion_positions: &[IVec3],
) -> Option<Vec<EndpointChange>> {
    let mut piped = scratch.clone();
    piped.try_add_pipe(pipe.clone()).ok()?;
    let canonical = piped.fix_shadowed_faces();

    let relabels: Vec<(IVec3, BlockKind)> = [pipe.src(), pipe.dst()]
        .into_iter()
        .filter_map(|pos| {
            let block = graph.get_endpoint_block(pos)?;
            let relabelled = canonical.get_endpoint_block(pos)?.kind();
            (block.kind().is_cube() && relabelled != block.kind())
                .then_some((block.pos(), relabelled))
        })
        .collect();
    if relabels.is_empty() {
        return None;
    }

    let mut relabelled = scratch.clone();
    for &(pos, kind) in &relabels {
        relabelled.set_block_kind(pos, kind).ok()?;
    }
    let promotions = endpoint_promotions(&relabelled, promotion_positions, pipe).ok()?;
    Some(
        relabels
            .into_iter()
            .map(|(pos, kind)| EndpointChange::SetKind { pos, kind })
            .chain(promotions)
            .collect(),
    )
}

fn probe_plan(graph: &BlockGraph, plan: &PipePlacementPlan) -> Result<(), PipePlanError> {
    validate_pipe_edit_structure(graph, &plan.applied_graph(graph)?)
}

fn reject_spatial_cube_scale_mismatch(
    graph: &BlockGraph,
    src: IVec3,
    dst: IVec3,
    direction: Direction,
) -> Result<(), PipePlanError> {
    if !direction.is_spatial() {
        return Ok(());
    }
    let src_block = graph.get_endpoint_block(src);
    let dst_block = graph.get_endpoint_block(dst);
    match (src_block, dst_block) {
        // Spatially merged cubes share their syndrome rounds, so the validator
        // requires one height across a spatial component. Reject the pipe here
        // rather than letting the placement land and fail validation.
        (Some(src_block), Some(dst_block))
            if src_block.kind().is_cube()
                && dst_block.kind().is_cube()
                && src_block.height() != dst_block.height() =>
        {
            Err(PipePlanError::SpatialCubeHeightMismatch {
                src: src_block.pos(),
                dst: dst_block.pos(),
                src_height: src_block.height(),
                dst_height: dst_block.height(),
            })
        }
        // A multi-cell cube's spatial faces only line up against a cube of the
        // same cell span, so an empty or shorter endpoint has nothing to meet.
        // This is a footprint question, not a round-count one: `height=d/2` and
        // `height=d` both occupy one cell and are caught by the arm above instead.
        (Some(block), other) if is_multi_cell_cube(block) && !same_span_cube(other, block) => {
            Err(PipePlanError::TallCubeSpatialPipeRequiresMatchingCube {
                pos: block.pos(),
                cells: block.height_cells(),
                other: dst,
            })
        }
        (other, Some(block)) if is_multi_cell_cube(block) && !same_span_cube(other, block) => {
            Err(PipePlanError::TallCubeSpatialPipeRequiresMatchingCube {
                pos: block.pos(),
                cells: block.height_cells(),
                other: src,
            })
        }
        _ => Ok(()),
    }
}

fn is_multi_cell_cube(block: &Block) -> bool {
    block.kind().is_cube() && block.height_cells() != 1
}

fn same_span_cube(candidate: Option<&Block>, block: &Block) -> bool {
    candidate.is_some_and(|candidate| {
        candidate.kind().is_cube() && candidate.height_cells() == block.height_cells()
    })
}

impl PipePlacementPlan {
    pub(crate) fn apply_prevalidated_to(&self, graph: &mut BlockGraph) {
        *graph = self
            .applied_graph(graph)
            .expect("pipe placement plan was probed against the unchanged graph");
    }

    fn applied_graph(&self, graph: &BlockGraph) -> Result<BlockGraph, PipePlanError> {
        let mut next = graph.clone();
        for &change in &self.endpoint_changes {
            match change {
                EndpointChange::CreatePort { pos } => {
                    if !next.has_endpoint_at(pos) {
                        next.try_add_block(Block::new(pos, BlockKind::Port))?;
                    }
                }
                EndpointChange::SetKind { pos, kind } => {
                    next.set_block_kind(pos, kind)?;
                }
            }
        }
        next.try_add_pipe(self.pipe.clone())?;
        Ok(next)
    }
}

/// Rejects an edit only for the structural violations it introduces.
///
/// Editing is incremental, so intermediate graphs are routinely broken: a block
/// placed before its pipes, a port left dangling by a deletion, a patch rotation
/// with only one of its two temporal pipes wired. Validating `after` outright
/// would blame the edit for all of it and make a graph unrecoverable once any
/// part of it went invalid, so anything already present in `before` is carried
/// over untouched and only genuinely new violations block the edit.
pub(crate) fn validate_pipe_edit_structure(
    before: &BlockGraph,
    after: &BlockGraph,
) -> Result<(), PipePlanError> {
    // Role selection happens in the element editor after pipe placement. `Auto`
    // is an allowed incomplete editor state, though compilation still rejects it.
    let blocks_edit = |err: &BlockGraphError| {
        !matches!(
            err,
            BlockGraphError::Invalid(InvalidBlockGraphError::SpatialPortRoleRequired(_))
        )
    };
    let existing: HashSet<String> = before
        .structural_errors()
        .iter()
        .filter(|err| blocks_edit(err))
        .map(ToString::to_string)
        .collect();
    match after
        .structural_errors()
        .into_iter()
        .filter(blocks_edit)
        .find(|err| !existing.contains(&err.to_string()))
    {
        Some(err) => Err(err.into()),
        None => Ok(()),
    }
}

fn endpoint_promotions(
    graph: &BlockGraph,
    positions: &[IVec3],
    pipe: &Pipe,
) -> Result<Vec<EndpointChange>, PipePlanError> {
    match positions {
        [] => Ok(Vec::new()),
        [pos] => Ok(vec![can_promote_port(graph, *pos, pipe)?]),
        [first, second] => {
            let Some(selected) = best_promotion_pair(graph, *first, *second, pipe) else {
                return Err(PipePlanError::PortPromotionConflict(*first));
            };

            Ok(vec![
                EndpointChange::SetKind {
                    pos: *first,
                    kind: selected[0],
                },
                EndpointChange::SetKind {
                    pos: *second,
                    kind: selected[1],
                },
            ])
        }
        _ => unreachable!("pipe placement has exactly two endpoints"),
    }
}

fn best_promotion_pair(
    graph: &BlockGraph,
    first: IVec3,
    second: IVec3,
    pipe: &Pipe,
) -> Option<[BlockKind; 2]> {
    let cube_kinds: Vec<_> = BlockKind::all_kinds()
        .into_iter()
        .filter(|kind| matches!(kind, BlockKind::Cube(_)))
        .collect();
    let mut combinations = Vec::new();
    for (first_index, &first_kind) in cube_kinds.iter().enumerate() {
        for (second_index, &second_kind) in cube_kinds.iter().enumerate() {
            let pair = [first_kind, second_kind];
            if graph_valid_with_endpoint_kinds(graph, &[first, second], &pair, pipe) {
                let score = promotion_basis_preservation_score(graph, first, first_kind, pipe)
                    + promotion_basis_preservation_score(graph, second, second_kind, pipe);
                combinations.push((score, first_index, second_index, pair));
            }
        }
    }
    combinations
        .into_iter()
        .max_by_key(|(score, first, second, _)| (*score, Reverse(*first), Reverse(*second)))
        .map(|(_, _, _, pair)| pair)
}

fn promotion_basis_preservation_score(
    graph: &BlockGraph,
    pos: IVec3,
    kind: BlockKind,
    extra_pipe: &Pipe,
) -> usize {
    let existing_pipes = graph
        .pipes()
        .filter(|pipe| pipe.src() == pos || pipe.dst() == pos)
        .cloned()
        .collect::<Vec<_>>();
    if existing_pipes.is_empty() {
        return 0;
    }

    let before = existing_pipes
        .iter()
        .map(|pipe| graph.infer_pipe_basis(pipe))
        .collect::<Vec<_>>();
    let mut probe = graph.clone();
    if probe.set_block_kind(pos, kind).is_err() {
        return 0;
    }
    if !probe.has_pipe_between(extra_pipe.src(), extra_pipe.dst())
        && probe.try_add_pipe(extra_pipe.clone()).is_err()
    {
        return 0;
    }

    existing_pipes
        .iter()
        .zip(before)
        .map(|(pipe, before)| {
            let after = probe.infer_pipe_basis(pipe);
            before
                .into_iter()
                .zip(after)
                .filter(|(before, after)| before.is_some() && before == after)
                .count()
        })
        .sum()
}

fn graph_valid_with_endpoint_kinds(
    graph: &BlockGraph,
    positions: &[IVec3],
    kinds: &[BlockKind],
    extra_pipe: &Pipe,
) -> bool {
    let mut probe = graph.clone();
    for (&pos, &kind) in positions.iter().zip(kinds) {
        if probe.has_block_at(pos) {
            if probe.set_block_kind(pos, kind).is_err() {
                return false;
            }
        } else if probe.try_add_block(Block::new(pos, kind)).is_err() {
            return false;
        }
    }
    if !probe.has_pipe_between(extra_pipe.src(), extra_pipe.dst())
        && probe.try_add_pipe(extra_pipe.clone()).is_err()
    {
        return false;
    }
    validate_pipe_edit_structure(graph, &probe).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::{Action, Basis, CubeKind, MeasureTarget, PatchRotationKind};

    fn cube(pos: IVec3) -> Block {
        Block::new(pos, BlockKind::Cube(CubeKind::ZXZ))
    }

    /// Wires a `T` block so the graph counts as dynamic, without adding a defect
    /// of its own.
    fn add_dynamic_anchor(graph: &mut BlockGraph, pos: IVec3) {
        graph.add_block(Block::new(pos, BlockKind::T));
        graph.add_block(Block::new(pos + IVec3::Z, BlockKind::Port));
        graph.add_pipe(Pipe::new(pos, Direction::ZPLUS));
    }

    #[test]
    fn can_place_pipe_between_existing_blocks() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));
        graph.add_block(cube(IVec3::X));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect("compatible adjacent cubes should accept a pipe");

        assert!(plan.endpoint_changes.is_empty());
        plan.apply_prevalidated_to(&mut graph);
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn can_place_first_patch_rotation_temporal_pipe_incrementally() {
        let mut graph = BlockGraph::new();
        let kind =
            PatchRotationKind::new(Basis::X, glam::ivec2(1, 0)).expect("valid patch rotation axis");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::NEG_Z,
                dst: IVec3::ZERO,
                hadamard: false,
            },
        )
        .expect("first patch rotation pipe should be placeable incrementally");

        plan.apply_prevalidated_to(&mut graph);
        assert!(graph.has_pipe_between(IVec3::NEG_Z, IVec3::ZERO));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: kind.end_position(IVec3::ZERO),
                dst: IVec3::new(1, 0, 2),
                hadamard: false,
            },
        )
        .expect("second patch rotation pipe should complete the block");
        plan.apply_prevalidated_to(&mut graph);
        graph
            .validate_structure()
            .expect("completed patch rotation graph is valid");
    }

    #[test]
    fn can_place_patch_rotation_pipe_with_unrelated_incomplete_patch_rotation() {
        let mut graph = BlockGraph::new();
        let kind =
            PatchRotationKind::new(Basis::X, glam::ivec2(0, 1)).expect("valid patch rotation axis");
        let target = IVec3::new(1, 0, 3);
        let target_cube = IVec3::new(1, 0, 2);
        let unrelated = IVec3::new(0, 0, 3);

        graph.add_block(Block::new(target_cube, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(target, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 2),
            BlockKind::Cube(CubeKind::XZZ),
        ));
        graph.add_block(Block::new(unrelated, BlockKind::PatchRotation(kind)));
        graph.add_pipe(Pipe::new(unrelated, Direction::ZMINUS));

        can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: target,
                dst: target_cube,
                hadamard: false,
            },
        )
        .expect("unrelated incomplete patch rotation should not suppress pipe hints");
    }

    #[test]
    fn patch_rotation_incremental_pipe_still_rejects_wrong_endpoint_orientation() {
        let mut graph = BlockGraph::new();
        let kind =
            PatchRotationKind::new(Basis::X, glam::ivec2(1, 0)).expect("valid patch rotation axis");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::XXZ)));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));

        let err = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::NEG_Z,
                dst: IVec3::ZERO,
                hadamard: false,
            },
        )
        .expect_err("wrong boundary orientation should still be rejected");

        assert!(matches!(err, PipePlanError::Graph(_)));
    }

    #[test]
    fn can_place_pipe_into_empty_endpoint_by_creating_port() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect("empty destination should be materialized as a port");

        assert_eq!(
            plan.endpoint_changes,
            vec![EndpointChange::CreatePort { pos: IVec3::X }]
        );
        plan.apply_prevalidated_to(&mut graph);
        assert_eq!(
            graph.get_block(IVec3::X).map(Block::kind),
            Some(BlockKind::Port)
        );
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn structural_defect_elsewhere_neither_blocks_nor_hides_a_pipe_edit() {
        let mut graph = BlockGraph::new();
        add_dynamic_anchor(&mut graph, IVec3::new(5, 0, 0));

        // Pre-existing defect: the spatial port still needs an explicit role.
        // Editing has to stay possible with this in place.
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXX)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        assert!(graph.validate_structure().is_err());

        let unrelated = IVec3::new(3, 0, 0);
        graph.add_block(Block::new(unrelated, BlockKind::Cube(CubeKind::ZXX)));
        can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: unrelated,
                dst: unrelated + IVec3::Y,
                hadamard: false,
            },
        )
        .expect("a defect elsewhere must not block an unrelated pipe");

        // The tolerance is scoped to what was already broken: a pipe that adds a
        // degenerate face of its own is still rejected.
        let degenerate = IVec3::new(7, 0, 0);
        graph.add_block(Block::new(degenerate, BlockKind::Cube(CubeKind::ZXZ)));
        can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: degenerate,
                dst: degenerate + IVec3::Y,
                hadamard: false,
            },
        )
        .expect_err("a newly introduced violation is still rejected");
    }

    #[test]
    fn spatial_pipe_into_empty_cell_keeps_port_in_a_dynamic_graph() {
        let mut graph = BlockGraph::new();
        add_dynamic_anchor(&mut graph, IVec3::ZERO);

        let src = IVec3::new(3, 0, 0);
        let dst = IVec3::new(3, 1, 0);
        graph.add_block(Block::new(src, BlockKind::Cube(CubeKind::ZXX)));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src,
                dst,
                hadamard: false,
            },
        )
        .expect("spatial pipe into empty space should keep its new port");

        assert_eq!(
            plan.endpoint_changes,
            vec![EndpointChange::CreatePort { pos: dst }]
        );
        plan.apply_prevalidated_to(&mut graph);
        assert_eq!(graph.get_block(dst).map(Block::kind), Some(BlockKind::Port));
        graph
            .set_port_role(dst, bloq_graph::PortRole::Input)
            .unwrap();
        graph
            .validate_structure()
            .expect("directed spatial port keeps the dynamic graph valid");
    }

    #[test]
    fn spatial_pipe_from_a_port_into_an_empty_cell_promotes_only_existing_endpoint() {
        let mut graph = BlockGraph::new();
        add_dynamic_anchor(&mut graph, IVec3::ZERO);

        // A port grown one temporal pipe, as pipe mode leaves it: extending it
        // sideways promotes that occupied endpoint. The new cell remains a Port.
        let src = IVec3::new(3, 0, 1);
        let dst = IVec3::new(3, 1, 1);
        graph.add_block(Block::new(
            IVec3::new(3, 0, 0),
            BlockKind::Cube(CubeKind::ZXX),
        ));
        graph.add_block(Block::new(src, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(3, 0, 0), Direction::ZPLUS));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src,
                dst,
                hadamard: false,
            },
        )
        .expect("extending a port sideways should promote its occupied endpoint");

        plan.apply_prevalidated_to(&mut graph);
        assert!(matches!(
            graph.get_block(src).map(Block::kind),
            Some(BlockKind::Cube(_))
        ));
        assert_eq!(graph.get_block(dst).map(Block::kind), Some(BlockKind::Port));
        graph
            .set_port_role(dst, bloq_graph::PortRole::Input)
            .unwrap();
        graph
            .validate_structure()
            .expect("promotion and directed spatial port keep the graph valid");
    }

    #[test]
    fn can_place_temporal_pipe_from_t_and_y_blocks() {
        for kind in [BlockKind::T, BlockKind::Y] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, kind));

            let plan = can_place_pipe(
                &graph,
                PipePlacementRequest {
                    src: IVec3::ZERO,
                    dst: IVec3::Z,
                    hadamard: false,
                },
            )
            .expect("single temporal pipe should be accepted");

            assert_eq!(
                plan.endpoint_changes,
                vec![EndpointChange::CreatePort { pos: IVec3::Z }]
            );
            plan.apply_prevalidated_to(&mut graph);
            assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::Z));
        }
    }

    #[test]
    fn can_place_temporal_pipe_into_measurement_blocks() {
        for basis in [Basis::X, Basis::Z] {
            for hadamard in [false, true] {
                let mut graph = BlockGraph::new();
                graph.add_block(Block::new(IVec3::ZERO, BlockKind::Measurement(basis)));

                let plan = can_place_pipe(
                    &graph,
                    PipePlacementRequest {
                        src: IVec3::ZERO,
                        dst: IVec3::NEG_Z,
                        hadamard,
                    },
                )
                .expect("a measurement accepts its one incoming temporal pipe");

                assert_eq!(
                    plan.endpoint_changes,
                    vec![EndpointChange::CreatePort { pos: IVec3::NEG_Z }]
                );
                assert_eq!(plan.pipe.is_hadamard(), hadamard);
                plan.apply_prevalidated_to(&mut graph);
                assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::NEG_Z));
            }
        }
    }

    #[test]
    fn can_place_pipe_ignores_preexisting_action_semantic_errors() {
        let measured = IVec3::new(3, 0, 0);
        let mut graph = BlockGraph::new();
        graph.add_block(cube(measured));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(measured),
                name: "m0".into(),
            }])
            .expect("measure action targets an existing block");
        graph.remove_block(measured);
        graph.add_block(cube(IVec3::ZERO));

        assert!(graph.validate().is_err());

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect("local structural pipe placement should ignore unrelated action errors");

        plan.apply_prevalidated_to(&mut graph);
        assert_eq!(
            graph.get_block(IVec3::X).map(Block::kind),
            Some(BlockKind::Port)
        );
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn can_place_pipe_ignores_unrelated_incomplete_port() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XZZ)));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::X,
                dst: IVec3::new(1, 1, 0),
                hadamard: false,
            },
        )
        .expect("unrelated incomplete port should not suppress cube pipe hints");

        assert_eq!(
            plan.endpoint_changes,
            vec![EndpointChange::CreatePort {
                pos: IVec3::new(1, 1, 0)
            }]
        );
    }

    #[test]
    fn can_promote_port_ignores_preexisting_action_semantic_errors() {
        let measured = IVec3::new(3, 0, 0);
        let mut graph = BlockGraph::new();
        graph.add_block(cube(measured));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(measured),
                name: "m0".into(),
            }])
            .expect("measure action targets an existing block");
        graph.remove_block(measured);
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(cube(IVec3::X));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::XZX)));

        assert!(graph.validate().is_err());

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::Z,
                hadamard: true,
            },
        )
        .expect("local port promotion should ignore unrelated action errors");

        assert!(matches!(
            plan.endpoint_changes.as_slice(),
            [EndpointChange::SetKind { pos, kind: BlockKind::Cube(_) }]
                if *pos == IVec3::ZERO
        ));
        assert!(plan.pipe.is_hadamard());
        plan.apply_prevalidated_to(&mut graph);
        assert!(matches!(
            graph.get_block(IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Cube(_))
        ));
        assert!(
            graph
                .get_pipe(IVec3::ZERO, IVec3::Z)
                .expect("pipe exists")
                .is_hadamard()
        );
    }

    #[test]
    fn can_place_hadamard_pipe() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XZX)));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: true,
            },
        )
        .expect("compatible hadamard pipe should be accepted");

        assert!(plan.pipe.is_hadamard());
        plan.apply_prevalidated_to(&mut graph);
        assert!(
            graph
                .get_pipe(IVec3::ZERO, IVec3::X)
                .expect("pipe exists")
                .is_hadamard()
        );
    }

    #[test]
    fn spatial_pipe_between_different_scale_cubes_is_rejected_before_mutation() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("cube accepts a height"),
        );
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));

        let err = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect_err("spatial pipe should reject mismatched cube heights");

        assert!(matches!(
            err,
            PipePlanError::SpatialCubeHeightMismatch {
                src,
                dst,
                src_height,
                dst_height
            } if src == IVec3::ZERO
                && dst == IVec3::X
                && src_height == "2d".parse().expect("valid height")
                && dst_height == bloq_graph::CubeHeight::DEFAULT
        ));
        assert!(!graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn tall_cube_spatial_pipe_to_empty_endpoint_is_rejected_before_port_creation() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("cube accepts a height"),
        );

        let err = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect_err("tall cube should not auto-create a spatial port");

        assert!(matches!(
            err,
            PipePlanError::TallCubeSpatialPipeRequiresMatchingCube {
                pos,
                cells: 2,
                other
            } if pos == IVec3::ZERO && other == IVec3::X
        ));
        assert!(!graph.has_block_at(IVec3::X));
        assert!(!graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn can_promote_port_when_second_pipe_is_added() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(cube(IVec3::X));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_block(cube(IVec3::Z));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::Z,
                hadamard: false,
            },
        )
        .expect("second pipe should promote the port");

        assert!(matches!(
            plan.endpoint_changes.as_slice(),
            [EndpointChange::SetKind { pos, kind: BlockKind::Cube(_) }]
                if *pos == IVec3::ZERO
        ));
        plan.apply_prevalidated_to(&mut graph);
        assert!(matches!(
            graph.get_block(IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Cube(_))
        ));
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::Z));
    }

    #[test]
    fn port_promotion_preserves_existing_pipe_basis_when_adding_spatial_pipe() {
        let mut graph = BlockGraph::new();
        let origin = IVec3::ZERO;
        let temporal_end = IVec3::Z;
        let spatial_end = IVec3::new(1, 0, 1);

        graph.add_block(Block::new(origin, BlockKind::Port));
        graph.add_block(Block::new(temporal_end, BlockKind::Port));
        graph.add_pipe(Pipe::new(origin, Direction::ZPLUS));

        let temporal_pipe = graph
            .get_pipe(origin, temporal_end)
            .expect("temporal pipe exists")
            .clone();
        let temporal_basis_before = graph.infer_pipe_basis(&temporal_pipe);
        assert_eq!(
            temporal_basis_before,
            [Some(Basis::X), Some(Basis::Z), None]
        );

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: temporal_end,
                dst: spatial_end,
                hadamard: false,
            },
        )
        .expect("second pipe should promote the temporal end port");

        plan.apply_prevalidated_to(&mut graph);
        assert_eq!(
            graph.get_block(temporal_end).map(Block::kind),
            Some(BlockKind::Cube(CubeKind::XZX))
        );
        assert_eq!(
            graph.infer_pipe_basis(
                graph
                    .get_pipe(temporal_end, spatial_end)
                    .expect("spatial pipe exists")
            ),
            [None, Some(Basis::Z), Some(Basis::X)]
        );
        assert_eq!(
            graph.infer_pipe_basis(&temporal_pipe),
            temporal_basis_before
        );
    }

    #[test]
    fn can_promote_both_degree_one_ports_when_connecting_them() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::NEG_X));
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::NEG_X, Direction::XPLUS));
        graph.add_block(Block::new(IVec3::X, BlockKind::Port));
        graph.add_block(cube(IVec3::new(2, 0, 0)));
        graph.add_pipe(Pipe::new(IVec3::X, Direction::XPLUS));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect("connecting degree-one ports should promote both endpoints");

        assert_eq!(plan.endpoint_changes.len(), 2);
        for pos in [IVec3::ZERO, IVec3::X] {
            assert!(matches!(
                plan.endpoint_changes.iter().find(|change| matches!(
                    change,
                    EndpointChange::SetKind { pos: change_pos, .. } if *change_pos == pos
                )),
                Some(EndpointChange::SetKind {
                    kind: BlockKind::Cube(_),
                    ..
                })
            ));
        }

        plan.apply_prevalidated_to(&mut graph);
        assert!(matches!(
            graph.get_block(IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Cube(_))
        ));
        assert!(matches!(
            graph.get_block(IVec3::X).map(Block::kind),
            Some(BlockKind::Cube(_))
        ));
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn can_place_pipe_from_empty_src_to_existing_dst_by_creating_port() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::X));

        let plan = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect("empty source should be materialized as a port");

        assert_eq!(
            plan.endpoint_changes,
            vec![EndpointChange::CreatePort { pos: IVec3::ZERO }]
        );
        plan.apply_prevalidated_to(&mut graph);
        assert_eq!(
            graph.get_block(IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Port)
        );
        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn duplicate_pipe_is_rejected_before_mutation() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));
        graph.add_block(cube(IVec3::X));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let err = can_place_pipe(
            &graph,
            PipePlacementRequest {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: false,
            },
        )
        .expect_err("duplicate pipe should be rejected");

        assert!(matches!(err, PipePlanError::PipeAlreadyExists { .. }));
    }

    #[test]
    fn invalid_pipe_endpoints_are_rejected_before_mutation() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));

        for dst in [IVec3::ZERO, IVec3::new(2, 0, 0)] {
            let err = can_place_pipe(
                &graph,
                PipePlacementRequest {
                    src: IVec3::ZERO,
                    dst,
                    hadamard: false,
                },
            )
            .expect_err("identical or non-adjacent endpoints should be rejected");

            assert!(matches!(
                err,
                PipePlanError::InvalidEndpoints { src, dst: error_dst }
                    if src == IVec3::ZERO && error_dst == dst
            ));
        }
        assert_eq!(graph.pipes().count(), 0);
    }

    /// A pass-through cube's basis along its piped axis is shadowed by the pipes
    /// covering both faces, so the pipe tool may relabel it instead of refusing a
    /// pipe on the axis the stale label reserves.
    #[test]
    fn pipe_relabels_a_shadowed_pass_through_cube() {
        let mut graph = BlockGraph::new();
        for (x, kind) in [
            (0, CubeKind::XZX),
            (1, CubeKind::ZZX),
            (2, CubeKind::ZZX),
            (3, CubeKind::ZZX),
        ] {
            graph.add_block(Block::new(IVec3::new(x, 0, 1), BlockKind::Cube(kind)));
        }
        for x in 0..3 {
            graph.add_pipe(Pipe::new(IVec3::new(x, 0, 1), Direction::XPLUS));
        }

        for x in [1, 2] {
            let src = IVec3::new(x, 0, 1);
            let plan = can_place_pipe(
                &graph,
                PipePlacementRequest {
                    src,
                    dst: src + IVec3::Z,
                    hadamard: false,
                },
            )
            .expect("a shadowed pass-through cube should accept a temporal pipe");

            plan.apply_prevalidated_to(&mut graph);
            assert!(graph.has_pipe_between(src, src + IVec3::Z));
            assert_eq!(
                graph
                    .get_block(src)
                    .expect("relabelled cube stays in the graph")
                    .kind(),
                BlockKind::Cube(CubeKind::XZX)
            );
        }
        assert!(graph.validate().is_ok(), "{:?}", graph.validate());
    }
}
