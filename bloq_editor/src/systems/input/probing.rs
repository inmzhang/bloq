//! Pipe-candidate probing and hint-target queries.
use super::*;

pub(crate) const WALKING_MOVEMENTS: [IVec2; 8] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
    IVec2::new(1, 1),
    IVec2::new(1, -1),
    IVec2::new(-1, 1),
    IVec2::new(-1, -1),
];

pub(crate) const PATCH_ROTATION_MOVEMENTS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];

pub(crate) const PIPE_HINT_OFFSETS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

pub(crate) fn pipe_candidate_at(
    graph: &BlockGraph,
    start: IVec3,
    offset: IVec3,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    let end = start.checked_add(offset)?;
    let hadamard = pipe_placement_hadamard(graph, start, end, force_hadamard);
    can_place_pipe(
        graph,
        PipePlacementRequest {
            src: start,
            dst: end,
            hadamard,
        },
    )
    .is_ok()
    .then_some((start, end))
}

pub(crate) fn pipe_candidate_at_with_snapshot(
    graph: &BlockGraph,
    snapshot: &GraphAdjacencySnapshot,
    start: IVec3,
    offset: IVec3,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    // No graph-wide degenerate-face bail: placement rejects only the violations
    // the edit introduces, so one broken cube must not blank out every hint.
    let end = start.checked_add(offset)?;
    if start == end || start.as_i64vec3().manhattan_distance(end.as_i64vec3()) != 1 {
        return None;
    }
    if snapshot.has_pipe_between(start, end) {
        return None;
    }
    let start_block = snapshot.endpoint_block(start);
    let end_block = snapshot.endpoint_block(end);
    if start_block.is_none() && end_block.is_none() {
        return None;
    }
    if endpoint_is_definitely_full_for_preview(snapshot, start, start_block)
        || endpoint_is_definitely_full_for_preview(snapshot, end, end_block)
    {
        return None;
    }
    if has_degenerate_cube_pipe_face(snapshot, start, end, start_block)
        || has_degenerate_cube_pipe_face(snapshot, end, start, end_block)
    {
        return None;
    }
    pipe_candidate_at(graph, start, offset, force_hadamard)
}

fn endpoint_is_definitely_full_for_preview(
    snapshot: &GraphAdjacencySnapshot,
    endpoint: IVec3,
    block: Option<crate::utils::SnapshotBlock>,
) -> bool {
    let Some(block) = block else {
        return false;
    };
    match block.kind {
        BlockKind::Cube(_) | BlockKind::Walking(_) | BlockKind::PatchRotation(_) => false,
        BlockKind::Y | BlockKind::Measurement(_) | BlockKind::T | BlockKind::Port
            if snapshot.degree(endpoint) == 0 =>
        {
            false
        }
        BlockKind::Port if snapshot.degree(endpoint) == 1 && block.pos == endpoint => false,
        _ => true,
    }
}

/// A degenerate cube face only settles the answer while no pipe shadows the
/// cube: once one does, placement may relabel the shadowed axis and accept the
/// pipe anyway, so the full check has to run. A tall cube carries pipes on a
/// second cell this degree cannot see, so it skips the shortcut outright.
fn has_degenerate_cube_pipe_face(
    snapshot: &GraphAdjacencySnapshot,
    endpoint: IVec3,
    other_endpoint: IVec3,
    block: Option<crate::utils::SnapshotBlock>,
) -> bool {
    block.is_some_and(|block| {
        block.height_cells == 1
            && snapshot.degree(endpoint) == 0
            && cube_pipe_face_is_degenerate(block.kind, endpoint, other_endpoint)
    })
}

#[cfg(test)]
pub(crate) fn pipe_hint_target(
    graph: &BlockGraph,
    start: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    pipe_hint_target_with_snapshot(
        graph,
        &GraphAdjacencySnapshot::from_graph(graph),
        start,
        index,
        force_hadamard,
    )
}

pub(crate) fn pipe_hint_target_with_snapshot(
    graph: &BlockGraph,
    snapshot: &GraphAdjacencySnapshot,
    start: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    PIPE_HINT_OFFSETS
        .get(index)
        .filter(|offset| !is_hidden_tall_cube_spatial_hint_with_snapshot(snapshot, start, **offset))
        .and_then(|offset| {
            pipe_candidate_at_with_snapshot(graph, snapshot, start, *offset, force_hadamard)
        })
}

fn is_hidden_tall_cube_spatial_hint_with_snapshot(
    snapshot: &GraphAdjacencySnapshot,
    start: IVec3,
    offset: IVec3,
) -> bool {
    if offset.z != 0 {
        return false;
    }
    let Some(start_block) = snapshot.endpoint_block(start) else {
        return false;
    };
    if !start_block.kind.is_cube() || start_block.height_cells == 1 {
        return false;
    }
    let Some(target) = start.checked_add(offset) else {
        return true;
    };
    let Some(target_block) = snapshot.endpoint_block(target) else {
        return true;
    };
    !target_block.kind.is_cube() || target_block.height_cells != start_block.height_cells
}

/// Hint targets for a block spanning two time layers: index 0 exits the bottom
/// endpoint downward, index 1 exits the top endpoint upward. Shared by walking
/// and patch-rotation blocks, which differ only in how their span is derived.
fn axial_pipe_hint_target(
    graph: &BlockGraph,
    span_start: IVec3,
    span_end: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    match index {
        0 => pipe_candidate_at(graph, span_start, IVec3::NEG_Z, force_hadamard),
        1 => pipe_candidate_at(graph, span_end, IVec3::Z, force_hadamard),
        _ => None,
    }
}

pub(crate) fn walking_pipe_hint_target(
    graph: &BlockGraph,
    start: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    let block = graph.get_endpoint_block(start)?;
    let BlockKind::Walking(kind) = block.kind() else {
        return None;
    };
    let span_start = block.pos();
    axial_pipe_hint_target(
        graph,
        span_start,
        kind.end_position(span_start),
        index,
        force_hadamard,
    )
}

pub(crate) fn patch_rotation_pipe_hint_target(
    graph: &BlockGraph,
    start: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    let block = graph.get_endpoint_block(start)?;
    let BlockKind::PatchRotation(kind) = block.kind() else {
        return None;
    };
    let span_start = block.pos();
    axial_pipe_hint_target(
        graph,
        span_start,
        kind.end_position(span_start),
        index,
        force_hadamard,
    )
}

pub(crate) fn tall_cube_pipe_hint_target(
    graph: &BlockGraph,
    start: IVec3,
    index: usize,
    force_hadamard: bool,
) -> Option<(IVec3, IVec3)> {
    let block = graph.get_endpoint_block(start)?;
    if !block.kind().is_cube() || block.height_cells() == 1 {
        return None;
    }
    let cube_start = block.pos();
    let cube_end = cube_start + IVec3::new(0, 0, block.height_cells() as i32 - 1);
    match index {
        4 => pipe_candidate_at(graph, cube_end, IVec3::Z, force_hadamard),
        5 => pipe_candidate_at(graph, cube_start, IVec3::NEG_Z, force_hadamard),
        _ => None,
    }
}

pub(crate) fn patch_rotation_endpoint_for_pipe_click(
    graph: &BlockGraph,
    block_pos: IVec3,
    hovered_grid_pos: Option<IVec3>,
    hadamard: bool,
) -> Option<IVec3> {
    let block = graph.get_block(block_pos)?;
    let BlockKind::PatchRotation(kind) = block.kind() else {
        return None;
    };
    let start = block.pos();
    let end = kind.end_position(start);
    Some(preferred_pipe_endpoint(
        graph,
        start,
        end,
        hovered_grid_pos,
        hadamard,
    ))
}

pub(crate) fn tall_cube_endpoint_for_pipe_click(
    graph: &BlockGraph,
    block_pos: IVec3,
    hovered_grid_pos: Option<IVec3>,
    hadamard: bool,
) -> Option<IVec3> {
    let block = graph.get_block(block_pos)?;
    if !block.kind().is_cube() || block.height_cells() == 1 {
        return None;
    }
    let start = block.pos();
    let end = start + IVec3::new(0, 0, block.height_cells() as i32 - 1);
    Some(preferred_pipe_endpoint(
        graph,
        start,
        end,
        hovered_grid_pos,
        hadamard,
    ))
}

pub(super) fn preferred_pipe_endpoint(
    graph: &BlockGraph,
    start: IVec3,
    end: IVec3,
    hovered_grid_pos: Option<IVec3>,
    hadamard: bool,
) -> IVec3 {
    let start_can_pipe = pipe_candidate_at(graph, start, IVec3::NEG_Z, hadamard).is_some();
    let end_can_pipe = pipe_candidate_at(graph, end, IVec3::Z, hadamard).is_some();
    match (start_can_pipe, end_can_pipe) {
        (false, true) => end,
        (_, false) => start,
        (true, true) => hovered_grid_pos.map_or(start, |pointer_pos| {
            if pointer_pos
                .as_i64vec3()
                .manhattan_distance(end.as_i64vec3())
                < pointer_pos
                    .as_i64vec3()
                    .manhattan_distance(start.as_i64vec3())
            {
                end
            } else {
                start
            }
        }),
    }
}

pub(crate) fn moving_block_movements(kind: BlockKind) -> &'static [IVec2] {
    match kind {
        BlockKind::Walking(_) => &WALKING_MOVEMENTS,
        BlockKind::PatchRotation(_) => &PATCH_ROTATION_MOVEMENTS,
        _ => &[],
    }
}

pub(super) fn moving_block_kind(kind: BlockKind, movement: IVec2) -> eyre::Result<BlockKind> {
    Ok(match kind {
        BlockKind::Walking(kind) => {
            BlockKind::Walking(WalkingKind::new(kind.boundary(), movement)?)
        }
        BlockKind::PatchRotation(kind) => {
            BlockKind::PatchRotation(PatchRotationKind::new(kind.basis(), movement)?)
        }
        _ => bail!("Only walking and patch-rotation blocks have a movement"),
    })
}

pub(crate) fn moving_block_start_has_candidate(
    editor_state: &EditorState,
    graph: &BlockGraph,
    start: IVec3,
) -> bool {
    moving_block_movements(editor_state.block_kind)
        .iter()
        .any(|&movement| {
            moving_block_kind_if_placeable(graph, start, editor_state.block_kind, movement)
                .is_some()
        })
}

pub(crate) fn moving_block_candidate_kind(
    editor_state: &EditorState,
    graph: &BlockGraph,
    end: IVec3,
) -> Option<BlockKind> {
    let start = editor_state.walking_start?;
    let movement = movement_between(start, end, moving_block_movements(editor_state.block_kind))?;
    moving_block_kind_if_placeable(graph, start, editor_state.block_kind, movement)
}

pub(crate) fn walking_port_promotion_candidate(
    graph: &BlockGraph,
    start: IVec3,
    end: IVec3,
) -> Option<WalkingKind> {
    let start_block = graph.get_block(start)?;
    if !start_block.kind().is_port() {
        return None;
    }
    match graph.neighbor_positions(start).as_slice() {
        [] => {}
        [neighbor] if start.checked_add(IVec3::NEG_Z) == Some(*neighbor) => {}
        _ => return None,
    }

    let movement = movement_between(start, end, &WALKING_MOVEMENTS)?;
    if end
        .checked_add(IVec3::Z)
        .is_some_and(|next| graph.has_pipe_between(end, next))
    {
        return None;
    }
    WalkingBoundaryKind::ALL
        .into_iter()
        .filter_map(|boundary| WalkingKind::new(boundary, movement).ok())
        .find(|walking_kind| walking_promotion_fits_graph(graph, start, *walking_kind))
}

fn walking_promotion_fits_graph(
    graph: &BlockGraph,
    start: IVec3,
    walking_kind: WalkingKind,
) -> bool {
    let mut probe = graph.clone();
    probe
        .set_block_kind(start, BlockKind::Walking(walking_kind))
        .is_ok()
        && validate_pipe_edit_structure(graph, &probe).is_ok()
}

fn moving_block_kind_if_placeable(
    graph: &BlockGraph,
    start: IVec3,
    kind: BlockKind,
    movement: IVec2,
) -> Option<BlockKind> {
    let kind = moving_block_kind(kind, movement).ok()?;
    graph
        .can_place_block(&Block::new(start, kind))
        .is_ok()
        .then_some(kind)
}
