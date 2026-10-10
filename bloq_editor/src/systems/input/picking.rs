//! Pointer observers for element and preview-endpoint picking.
use super::*;

/// Prevents a middle-button camera drag from also deleting its start element.
#[derive(Resource, Default)]
pub(crate) struct MiddleButtonDrag(bool);

/// Pointer observer: records the hovered graph element, ignoring elements the
/// current-layer filter hides.
pub(crate) fn on_element_over(
    trigger: On<PointerOver>,
    mut editor_state: ResMut<EditorState>,
    graph_state: Res<GraphState>,
    query_element: Query<&GraphElement>,
    child_of_query: Query<&ChildOf>,
) {
    let entity = trigger.entity;
    if let Some(element) = picked_graph_element(entity, &query_element, &child_of_query) {
        if element_hidden_by_layer_filter(element, &graph_state.graph, &editor_state) {
            return;
        }

        editor_state.hovered_element = Some(element);
    }
}

/// Pointer observer: clears the hovered element when the pointer leaves it.
pub(crate) fn on_element_out(
    trigger: On<PointerOut>,
    mut editor_state: ResMut<EditorState>,
    query_element: Query<&GraphElement>,
    child_of_query: Query<&ChildOf>,
) {
    let entity = trigger.entity;
    if let Some(element) = picked_graph_element(entity, &query_element, &child_of_query)
        && editor_state.hovered_element == Some(element)
    {
        editor_state.hovered_element = None;
    }
}

pub(crate) fn on_element_drag_start(
    trigger: On<PointerDragStart>,
    mut middle_drag: ResMut<MiddleButtonDrag>,
) {
    if trigger.button == PointerButton::Middle {
        middle_drag.0 = true;
    }
}

pub(crate) fn on_element_drag_end(
    trigger: On<PointerDragEnd>,
    mut middle_drag: ResMut<MiddleButtonDrag>,
) {
    if trigger.button == PointerButton::Middle {
        middle_drag.0 = false;
    }
}

/// Pointer observer: handles clicks on a graph element (selection in View mode,
/// deletion with the middle button, or attribute editing in Edit mode).
pub(crate) fn on_element_click(
    trigger: On<PointerClick>,
    mut intents: ResMut<UiIntentBuffer>,
    time: Res<Time>,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    mut target_state: ResMut<TargetState>,
    mut action_edit: ResMut<ActionEditState>,
    mut notifications: ResMut<Notifications>,
    mut middle_drag: ResMut<MiddleButtonDrag>,
    translation_drag: Res<TranslationDrag>,
    keys: Res<ButtonInput<KeyCode>>,
    query_element: Query<&GraphElement>,
    child_of_query: Query<&ChildOf>,
) {
    let entity = trigger.entity;
    let Some(element) = picked_graph_element(entity, &query_element, &child_of_query) else {
        return;
    };

    if element_hidden_by_layer_filter(element, &graph_state.graph, &editor_state) {
        return;
    }

    if trigger.button == PointerButton::Middle {
        delete_middle_clicked_element(
            element,
            std::mem::take(&mut middle_drag.0),
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        );
        return;
    }
    if trigger.button != PointerButton::Primary {
        return;
    }
    if translation_drag.suppress_click {
        return;
    }
    if editor_state.mode == EditorMode::Module {
        if !is_ctrl_pressed(&keys) && !is_alt_pressed(&keys) {
            intents.push(UiIntent::InspectModuleElement(element));
        }
        return;
    }

    handle_element_click(
        element,
        time.elapsed_secs_f64(),
        &mut editor_state,
        &graph_state,
        &mut target_state,
        &mut action_edit,
        &mut notifications,
        &keys,
    );
}

pub(super) fn delete_middle_clicked_element(
    element: GraphElement,
    was_dragged: bool,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
) {
    if !was_dragged && matches!(editor_state.mode, EditorMode::Edit | EditorMode::View) {
        delete_elements(&[element], editor_state, graph_state, notifications);
    }
}

pub(super) fn handle_element_click(
    element: GraphElement,
    now: f64,
    editor_state: &mut EditorState,
    graph_state: &GraphState,
    target_state: &mut TargetState,
    action_edit: &mut ActionEditState,
    notifications: &mut Notifications,
    keys: &ButtonInput<KeyCode>,
) {
    if editor_state.mode == EditorMode::Module {
        return;
    }

    // An armed action pick owns the click: it was requested by an explicit
    // keypress, so it outranks selection and hold-to-edit alike.
    if action_edit.picking().is_some() {
        if let Err(message) = action_edit.accept_pick(element, &graph_state.graph) {
            notifications.push_warn(message);
        }
        return;
    }

    // Hold-to-edit wins over every other click meaning, in both modes: it is the
    // only way to reach a pipe's attributes while the Pipe tool owns the click.
    if is_attribute_edit_modifier_pressed(keys) {
        editor_state.pipe_start = None;
        editor_state.last_element_click = None;
        open_target_editor(element, graph_state, target_state, notifications);
        return;
    }

    if editor_state.mode == EditorMode::Edit && !is_ctrl_pressed(keys) && !is_alt_pressed(keys) {
        let double = matches!(
            editor_state.last_element_click,
            Some((last_element, last_time))
                if last_element == element && now - last_time < DOUBLE_CLICK_SECS
        );
        editor_state.last_element_click = (!double).then_some((element, now));
        if double {
            editor_state.pipe_start = None;
            open_target_editor(element, graph_state, target_state, notifications);
            return;
        }
    }

    if !editor_state.is_pipe_tool_active() {
        if matches!(editor_state.mode, EditorMode::View | EditorMode::Edit) {
            editor_state.click_select_element(element, is_ctrl_pressed(keys));
        }
        return;
    }

    if let GraphElement::Block(pos) = element {
        let hadamard = is_hadamard_pipe_modifier_pressed(keys);
        if editor_state.pipe_start.is_none() {
            let pos = pipe_endpoint_for_block_click(
                &graph_state.graph,
                pos,
                editor_state.hovered_grid_pos,
                hadamard,
            );
            editor_state.pipe_start = Some(pos);
        }
    }
}

pub(super) fn picked_graph_element(
    entity: Entity,
    query_element: &Query<&GraphElement>,
    child_of_query: &Query<&ChildOf>,
) -> Option<GraphElement> {
    query_element.get(entity).copied().ok().or_else(|| {
        child_of_query
            .iter_ancestors(entity)
            .find_map(|ancestor| query_element.get(ancestor).copied().ok())
    })
}

/// Pointer observer: places a pipe or walking block to the clicked preview
/// endpoint's target while the pipe tool is active.
pub(crate) fn on_preview_endpoint_click(
    trigger: On<PointerClick>,
    translation_drag: Res<TranslationDrag>,
    time: Res<Time>,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    mut target_state: ResMut<TargetState>,
    mut notifications: ResMut<Notifications>,
    keys: Res<ButtonInput<KeyCode>>,
    query_endpoint: Query<&PreviewEndpointMesh>,
) {
    if trigger.button != PointerButton::Primary
        || !editor_state.is_pipe_tool_active()
        || translation_drag.suppress_click
    {
        return;
    }
    let Some(default_start) = editor_state.pipe_start else {
        return;
    };
    let Ok(endpoint) = query_endpoint.get(trigger.entity) else {
        return;
    };
    let start = endpoint.source_pos.unwrap_or(default_start);
    let Some(target) = endpoint.target_pos else {
        return;
    };
    // Endpoint previews sit on top of the blocks they grow from, so hold-to-edit
    // has to be honoured here too or the modifier would appear to do nothing.
    if is_attribute_edit_modifier_pressed(&keys) {
        if let Some(block) = graph_state
            .graph
            .get_endpoint_block(target)
            .or_else(|| graph_state.graph.get_endpoint_block(start))
        {
            editor_state.pipe_start = None;
            editor_state.last_element_click = None;
            open_target_editor(
                GraphElement::Block(block.pos()),
                &graph_state,
                &mut target_state,
                &mut notifications,
            );
        }
        return;
    }
    if target == start {
        if open_pipe_start_editor_if_double_clicked(
            start,
            time.elapsed_secs_f64(),
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut notifications,
        ) {
            return;
        }
        editor_state.pipe_start = None;
        return;
    }
    editor_state.last_element_click = None;

    place_pipe_or_walking_target(
        start,
        target,
        is_hadamard_pipe_modifier_pressed(&keys),
        true,
        &mut editor_state,
        &mut graph_state,
        &mut notifications,
    );
}

pub(super) fn open_pipe_start_editor_if_double_clicked(
    start: IVec3,
    now: f64,
    editor_state: &mut EditorState,
    graph_state: &GraphState,
    target_state: &mut TargetState,
    notifications: &mut Notifications,
) -> bool {
    let Some((last_element, last_time)) = editor_state.last_element_click.take() else {
        return false;
    };
    let Some(block) = graph_state.graph.get_endpoint_block(start) else {
        return false;
    };
    let element = GraphElement::Block(block.pos());
    if last_element != element || now - last_time >= DOUBLE_CLICK_SECS {
        return false;
    }

    editor_state.pipe_start = None;
    open_target_editor(element, graph_state, target_state, notifications);
    true
}

/// Pointer observer: marks a preview endpoint as hovered so it can be
/// highlighted.
pub(crate) fn on_preview_endpoint_over(
    trigger: On<PointerOver>,
    mut editor_state: ResMut<EditorState>,
    query_endpoint: Query<&PreviewEndpointMesh>,
) {
    let Ok(endpoint) = query_endpoint.get(trigger.entity) else {
        return;
    };
    editor_state.hovered_preview_endpoint = editor_state.pipe_start.and(endpoint.target_pos);
    editor_state.hovered_preview_source = editor_state
        .hovered_preview_endpoint
        .and(endpoint.source_pos);
}

/// Pointer observer: clears the hovered preview endpoint on pointer-out.
pub(crate) fn on_preview_endpoint_out(
    trigger: On<PointerOut>,
    mut editor_state: ResMut<EditorState>,
    query_endpoint: Query<&PreviewEndpointMesh>,
) {
    let Ok(endpoint) = query_endpoint.get(trigger.entity) else {
        return;
    };
    if endpoint.target_pos == editor_state.hovered_preview_endpoint {
        editor_state.hovered_preview_source = None;
        editor_state.hovered_preview_endpoint = None;
    }
}
