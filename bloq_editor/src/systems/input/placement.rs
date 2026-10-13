//! Block and pipe placement, and element deletion.
use super::*;

pub(super) fn place_single_block(
    pos: IVec3,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    let block = Block::new(pos, editor_state.block_kind);
    match graph_state.graph.try_add_block(block) {
        Ok(_) => {
            graph_state.commit();
            editor_state.sync_after_graph_edit(graph_state);
            notifications.push_info(format!("Added block at {pos}"));
        }
        Err(err) => {
            notifications.push_error(format!("Failed to add block at {pos}: {err}"));
        }
    }
}

pub(super) fn place_moving_block_click(
    pos: IVec3,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    let (name, title, step) = match editor_state.block_kind {
        BlockKind::Walking(_) => ("walking", "Walking", "one spatial step"),
        BlockKind::PatchRotation(_) => (
            "patch rotation",
            "Patch rotation",
            "one cardinal spatial step",
        ),
        _ => return,
    };

    if let Some(start) = editor_state.walking_start {
        if start == pos {
            editor_state.walking_start = None;
            notifications.push_info(format!("Cancelled {name} placement"));
            return;
        }
        let movement =
            movement_between(start, pos, moving_block_movements(editor_state.block_kind));
        let Some(movement) = movement else {
            notifications.push_warn(format!(
                "{title} end must be on the next time layer within {step}"
            ));
            return;
        };
        // The movement table and the constructor's rules live in separate
        // modules; degrade to a toast instead of crashing if they drift.
        let kind = match moving_block_kind(editor_state.block_kind, movement) {
            Ok(kind) => kind,
            Err(err) => {
                notifications.push_error(format!("Cannot place {name} block: {err}"));
                return;
            }
        };
        let block = Block::new(start, kind);
        match graph_state.graph.try_add_block(block) {
            Ok(_) => {
                graph_state.commit();
                editor_state.walking_start = None;
                editor_state.sync_after_graph_edit(graph_state);
                notifications.push_info(format!("Added {name} block from {start} to {pos}"));
            }
            Err(err) => {
                notifications.push_error(format!("Failed to add {name} block: {err}"));
            }
        }
        return;
    }

    if !moving_block_start_has_candidate(editor_state, &graph_state.graph, pos) {
        notifications.push_error(format!(
            "Cannot start {name} block at occupied position {pos}"
        ));
        return;
    }
    editor_state.walking_start = Some(pos);
    editor_state.plane_height = pos.z + 1;
    graph_state.needs_rerender = true;
    notifications.push_info(format!(
        "{title} start set at {pos}; choose a highlighted end on z={}",
        pos.z + 1
    ));
}

/// Returns the in-plane step from `start` to `end` when `end` sits exactly one
/// time layer above and the step is one of `allowed`. Walking and patch-rotation
/// placement share this rule, differing only in their allowed movement table.
pub(super) fn movement_between(start: IVec3, end: IVec3, allowed: &[IVec2]) -> Option<IVec2> {
    let delta = end.checked_sub(start)?;
    if delta.z != 1 {
        return None;
    }
    let movement = IVec2::new(delta.x, delta.y);
    allowed.contains(&movement).then_some(movement)
}

fn promote_port_to_walking_block(
    start: IVec3,
    walking_kind: WalkingKind,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    match graph_state
        .graph
        .set_block_kind(start, BlockKind::Walking(walking_kind))
    {
        Ok(()) => {
            graph_state.commit();
            editor_state.pipe_start = None;
            editor_state.sync_after_graph_edit(graph_state);
            notifications.push_info(format!(
                "Promoted port at {start} to walking block ending at {}",
                walking_kind.end_position(start)
            ));
        }
        Err(err) => {
            editor_state.pipe_start = None;
            notifications.push_error(format!("Failed to promote port to walking block: {err}"));
        }
    }
}

pub(super) fn pipe_endpoint_for_block_click(
    graph: &BlockGraph,
    block_pos: IVec3,
    hovered_grid_pos: Option<IVec3>,
    hadamard: bool,
) -> IVec3 {
    let Some(block) = graph.get_block(block_pos) else {
        return block_pos;
    };
    if let Some(endpoint) =
        patch_rotation_endpoint_for_pipe_click(graph, block_pos, hovered_grid_pos, hadamard)
    {
        return endpoint;
    }
    if let Some(endpoint) =
        tall_cube_endpoint_for_pipe_click(graph, block_pos, hovered_grid_pos, hadamard)
    {
        return endpoint;
    }
    let BlockKind::Walking(kind) = block.kind() else {
        return block_pos;
    };
    let start = block.pos();
    let end = kind.end_position(start);
    preferred_pipe_endpoint(graph, start, end, hovered_grid_pos, hadamard)
}

pub(super) fn place_pipe_or_walking_target(
    start: IVec3,
    target: IVec3,
    force_hadamard: bool,
    allow_walking: bool,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    if start.manhattan_distance(target) == 1 {
        place_pipe_between(
            start,
            target,
            force_hadamard,
            editor_state,
            graph_state,
            notifications,
        );
    } else if allow_walking
        && let Some(walking_kind) =
            walking_port_promotion_candidate(&graph_state.graph, start, target)
    {
        promote_port_to_walking_block(
            start,
            walking_kind,
            editor_state,
            graph_state,
            notifications,
        );
    } else {
        editor_state.pipe_start = Some(target);
        notifications.push_warn("Pipe endpoints must be adjacent");
    }
}

fn place_pipe_between(
    start: IVec3,
    end: IVec3,
    force_hadamard: bool,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    if place_pipe(start, end, force_hadamard, graph_state, notifications) {
        editor_state.last_pipe_placement = Some((start, end));
        editor_state.sync_after_graph_edit(graph_state);
    }
    editor_state.pipe_start = None;
}

pub(super) fn place_pipe(
    start: IVec3,
    end: IVec3,
    force_hadamard: bool,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) -> bool {
    let hadamard = pipe_placement_hadamard(&graph_state.graph, start, end, force_hadamard);
    match can_place_pipe(
        &graph_state.graph,
        PipePlacementRequest {
            src: start,
            dst: end,
            hadamard,
        },
    ) {
        Ok(plan) => {
            let edit_delta = if plan.endpoint_changes.is_empty() {
                GraphEditDelta::PipeAdded {
                    src: start,
                    dst: end,
                    hadamard,
                }
            } else {
                GraphEditDelta::Unknown
            };
            plan.apply_prevalidated_to(&mut graph_state.graph);
            graph_state.commit_with_delta(true, edit_delta);
            notifications.push_info(format!("Added pipe between {start} and {end}"));
            true
        }
        Err(err) => {
            notifications.push_error(format!("Cannot place pipe: {err}"));
            false
        }
    }
}

pub(super) fn handle_pipe_keyboard_placement(
    keys: &ButtonInput<KeyCode>,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) -> bool {
    let repeat = keys.just_pressed(KeyCode::KeyR);
    let offset = [
        (KeyCode::KeyW, IVec3::Y),
        (KeyCode::KeyA, IVec3::NEG_X),
        (KeyCode::KeyS, IVec3::NEG_Y),
        (KeyCode::KeyD, IVec3::X),
        (KeyCode::ArrowUp, IVec3::Z),
        (KeyCode::ArrowDown, IVec3::NEG_Z),
    ]
    .into_iter()
    .find_map(|(key, offset)| keys.just_pressed(key).then_some(offset));
    if !repeat && offset.is_none() {
        return false;
    }

    if !editor_state.is_pipe_tool_active() {
        return false;
    }
    let (start, offset) = if repeat {
        let Some((previous_start, start)) = editor_state.last_pipe_placement else {
            notifications.push_warn("Place a pipe before repeating placement");
            return true;
        };
        (start, start - previous_start)
    } else {
        let Some(start) = editor_state.pipe_start else {
            notifications.push_warn("Choose a pipe start before keyboard pipe placement");
            return true;
        };
        (start, offset.expect("checked above"))
    };
    let Ok(end) = bloq_graph::checked_add_position(start, offset) else {
        notifications.push_error("Cannot place pipe: coordinate overflow");
        return true;
    };
    if place_pipe(
        start,
        end,
        is_hadamard_pipe_modifier_pressed(keys),
        graph_state,
        notifications,
    ) {
        editor_state.last_pipe_placement = Some((start, end));
        editor_state.pipe_start = Some(end);
        editor_state.sync_after_graph_edit(graph_state);
    }
    true
}

pub(super) fn delete_selected_elements(
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    if editor_state.mode != EditorMode::View {
        notifications.push_warn("Switch to View mode before deleting selected elements");
        return;
    }

    let selected = editor_state
        .selected_elements()
        .map(GraphElement::canonical)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        notifications.push_warn("Select an element before deleting");
        return;
    }

    if delete_elements(&selected, editor_state, graph_state, notifications) > 0 {
        editor_state.clear_selection();
    }
}

pub(super) fn delete_elements(
    elements: &[GraphElement],
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) -> usize {
    if graph_state.is_composed() {
        notifications.push_warn("Edit a definition in Modules, or open a flat copy");
        return 0;
    }
    let mut removed = 0usize;
    let mut removed_delta = None;
    // Endpoints that lost a pipe, so the sweep below can retire the ports that
    // pipe placement had auto-created for them.
    let mut orphan_candidates = Vec::new();
    for element in elements
        .iter()
        .copied()
        .filter(|element| matches!(element, GraphElement::Pipe(_, _)))
    {
        if let GraphElement::Pipe(u, v) = element
            && let Some(pipe) = graph_state.graph.remove_pipe(u, v)
        {
            let (src, dst) = pipe.endpoints();
            orphan_candidates.extend([src, dst]);
            if removed == 0 {
                removed_delta = Some(GraphEditDelta::PipeRemoved {
                    src,
                    dst,
                    hadamard: pipe.is_hadamard(),
                });
            } else {
                removed_delta = None;
            }
            removed += 1;
        }
    }
    for element in elements
        .iter()
        .copied()
        .filter(|element| matches!(element, GraphElement::Block(_)))
    {
        if let GraphElement::Block(pos) = element {
            // Removing the block drops its pipes too, so its neighbours join the
            // orphan sweep; collect them while the pipes still exist.
            let neighbors = graph_state.graph.neighbor_positions(pos);
            if graph_state.graph.remove_block(pos).is_some() {
                orphan_candidates.extend(neighbors);
                removed_delta = None;
                removed += 1;
            }
        }
    }

    if removed == 0 {
        editor_state.sync_after_graph_edit(graph_state);
        notifications.push_warn("Element was already gone");
        return 0;
    }

    if remove_isolated_ports(&mut graph_state.graph, &orphan_candidates) {
        // The delta only describes a pipe edit; dropping blocks invalidates it.
        removed_delta = None;
    }

    if let Some(delta) = removed_delta {
        graph_state.commit_with_delta(true, delta);
    } else {
        graph_state.commit();
    }
    editor_state.hovered_element = None;
    editor_state.sync_after_graph_edit(graph_state);
    notifications.push_info(format!("Removed {removed} element(s)"));
    removed
}

/// Retires the ports at `candidates` that the deletion just left with no pipes,
/// returning whether anything was removed.
///
/// Pipe placement silently creates a `Port` at an empty endpoint, so deleting
/// the pipe again has to undo that: a leftover port is a near-transparent gray
/// cube (`RGBA::PORT_GRAY` is 35% opaque) that reads as empty space but still
/// occupies the cell, so the next block placed there fails as "occupied".
///
/// This cannot distinguish an auto-created port from one the user placed by
/// hand and later piped; both are dropped once their last pipe goes.
fn remove_isolated_ports(graph: &mut BlockGraph, candidates: &[IVec3]) -> bool {
    let orphans = candidates
        .iter()
        .filter_map(|pos| graph.get_endpoint_block(*pos))
        .filter(|block| block.kind().is_port())
        .map(Block::pos)
        .filter(|pos| graph.degree(*pos) == 0)
        .collect::<HashSet<_>>();
    let mut removed_any = false;
    for pos in orphans {
        removed_any |= graph.remove_block(pos).is_some();
    }
    removed_any
}
