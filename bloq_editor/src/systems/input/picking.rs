//! Pointer observers for element and preview-endpoint picking.
use super::*;
use crate::resources::ImportExportState;

/// Pointer observer: records the hovered graph element, ignoring elements the
/// current-layer filter hides.
pub(crate) fn on_element_over(
    trigger: On<PointerOver>,
    mut editor_state: ResMut<EditorState>,
    graph_state: Res<GraphState>,
    query_element: Query<&GraphElement>,
    child_of_query: Query<&ChildOf>,
) {
    let entity = trigger.original_event_target();
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
    let entity = trigger.original_event_target();
    if let Some(element) = picked_graph_element(entity, &query_element, &child_of_query)
        && editor_state.hovered_element == Some(element)
    {
        editor_state.hovered_element = None;
    }
}

/// Pointer observer: selects graph elements or opens their attributes.
pub(crate) fn on_element_click(
    trigger: On<PointerClick>,
    import_export: Res<ImportExportState>,
    mut intents: ResMut<UiIntentBuffer>,
    time: Res<Time>,
    mut editor_state: ResMut<EditorState>,
    graph_state: Res<GraphState>,
    mut target_state: ResMut<TargetState>,
    mut action_edit: ResMut<ActionEditState>,
    mut notifications: ResMut<Notifications>,
    translation_drag: Res<TranslationDrag>,
    keys: Res<ButtonInput<KeyCode>>,
    query_element: Query<&GraphElement>,
    child_of_query: Query<&ChildOf>,
) {
    if import_export.blocks_graph_editing()
        || target_state.open_window
        || target_state.capture_input
    {
        return;
    }
    let entity = trigger.original_event_target();
    let Some(element) = picked_graph_element(entity, &query_element, &child_of_query) else {
        return;
    };

    if element_hidden_by_layer_filter(element, &graph_state.graph, &editor_state) {
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

    // Camera gestures must never select or edit the object beneath the pointer.
    if is_alt_pressed(keys) {
        return;
    }

    // An armed action pick owns the click: it was requested by an explicit
    // keypress, so it outranks selection and attribute editing.
    if action_edit.picking().is_some() {
        if let Err(message) = action_edit.accept_pick(element, &graph_state.graph) {
            notifications.push_warn(message);
        }
        return;
    }

    // Selection modifiers have the same meaning with every placement tool.
    if matches!(editor_state.mode, EditorMode::View | EditorMode::Edit)
        && (is_ctrl_pressed(keys) || is_shift_pressed(keys))
    {
        editor_state.last_element_click = None;
        editor_state.click_select_element(element, true);
        return;
    }

    if matches!(editor_state.mode, EditorMode::View | EditorMode::Edit) {
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
            editor_state.click_select_element(element, false);
        }
        return;
    }

    if let GraphElement::Block(pos) = element {
        let hadamard = editor_state.pipe_hadamard || is_hadamard_pipe_modifier_pressed(keys);
        if editor_state.pipe_start.is_none() {
            let pos = pipe_endpoint_for_block_click(
                &graph_state.graph,
                pos,
                editor_state.hovered_grid_pos,
                hadamard,
            );
            editor_state.pipe_start = Some(pos);
            editor_state.click_select_element(element, false);
        }
    } else {
        // Selecting an existing pipe exposes its attributes without moving the
        // independent placement cursor or changing the graph.
        editor_state.click_select_element(element, false);
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
    import_export: Res<ImportExportState>,
    translation_drag: Res<TranslationDrag>,
    time: Res<Time>,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    mut target_state: ResMut<TargetState>,
    mut notifications: ResMut<Notifications>,
    keys: Res<ButtonInput<KeyCode>>,
    query_endpoint: Query<&PreviewEndpointMesh>,
) {
    if import_export.blocks_graph_editing()
        || trigger.button != PointerButton::Primary
        || target_state.open_window
        || target_state.capture_input
        || !editor_state.is_pipe_tool_active()
        || translation_drag.suppress_click
        || is_alt_pressed(&keys)
        || is_ctrl_pressed(&keys)
        || is_shift_pressed(&keys)
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
        editor_state.pipe_hadamard || is_hadamard_pipe_modifier_pressed(&keys),
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

#[cfg(test)]
mod observer_tests {
    use super::*;
    use crate::components::PreviewEndpoint;
    use bevy::camera::NormalizedRenderTarget;
    use bevy::picking::backend::HitData;
    use bevy::picking::pointer::{Location, PointerId};
    use bevy::window::WindowRef;
    use bloq_graph::CubeKind;

    fn observer_app() -> (App, Entity) {
        let mut app = App::new();
        app.init_resource::<ImportExportState>()
            .init_resource::<UiIntentBuffer>()
            .init_resource::<Time>()
            .init_resource::<EditorState>()
            .init_resource::<GraphState>()
            .init_resource::<TargetState>()
            .init_resource::<ActionEditState>()
            .init_resource::<Notifications>()
            .init_resource::<TranslationDrag>()
            .init_resource::<ButtonInput<KeyCode>>();
        app.world_mut()
            .resource_mut::<GraphState>()
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let window = app.world_mut().spawn_empty().id();
        (app, window)
    }

    fn click(app: &mut App, entity: Entity, window: Entity, button: PointerButton) {
        app.world_mut().trigger(PointerClick {
            entity,
            pointer: Pointer::new(
                PointerId::Mouse,
                Location {
                    target: NormalizedRenderTarget::Window(
                        WindowRef::Entity(window).normalize(Some(window)).unwrap(),
                    ),
                    position: Vec2::ZERO,
                },
            ),
            button,
            hit: HitData::new(window, 0.0, None, None),
            duration: std::time::Duration::from_millis(10),
            count: 1,
        });
        app.update();
    }

    #[test]
    fn shared_root_observers_handle_replaced_children_once_and_ignore_outside_elements() {
        let (mut app, window) = observer_app();
        app.world_mut()
            .entity_mut(window)
            .insert(bevy::window::Window::default());
        let root = app
            .world_mut()
            .spawn(crate::components::GraphRoot)
            .observe(on_element_over)
            .observe(on_element_out)
            .observe(on_element_click)
            .id();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::ShiftLeft);
        let pointer = Pointer::new(
            PointerId::Mouse,
            Location {
                target: NormalizedRenderTarget::Window(
                    WindowRef::Entity(window).normalize(Some(window)).unwrap(),
                ),
                position: Vec2::ZERO,
            },
        );
        for _ in 0..2 {
            let element = app
                .world_mut()
                .spawn((GraphElement::Block(IVec3::ZERO), ChildOf(root)))
                .id();
            let mesh = app.world_mut().spawn(ChildOf(element)).id();
            app.update();
            app.world_mut().trigger(PointerOver {
                entity: mesh,
                pointer: pointer.clone(),
                hit: HitData::new(window, 0.0, None, None),
            });
            assert_eq!(
                app.world().resource::<EditorState>().hovered_element,
                Some(GraphElement::Block(IVec3::ZERO))
            );
            // Shift selection toggles, so duplicate delivery would leave it empty.
            click(&mut app, mesh, window, PointerButton::Primary);
            assert_eq!(app.world().resource::<EditorState>().selection_count(), 1);
            click(&mut app, mesh, window, PointerButton::Primary);
            assert_eq!(app.world().resource::<EditorState>().selection_count(), 0);
            app.world_mut().trigger(PointerOut {
                entity: mesh,
                pointer: pointer.clone(),
                hit: HitData::new(window, 0.0, None, None),
            });
            assert_eq!(app.world().resource::<EditorState>().hovered_element, None);
            app.world_mut().despawn(element);
        }
        let outside = app.world_mut().spawn(GraphElement::Block(IVec3::ZERO)).id();
        click(&mut app, outside, window, PointerButton::Primary);
        assert_eq!(app.world().resource::<EditorState>().selection_count(), 0);
    }

    #[test]
    fn middle_click_never_deletes_or_selects_an_element() {
        for mode in [EditorMode::View, EditorMode::Edit] {
            let (mut app, window) = observer_app();
            app.add_observer(on_element_click);
            app.world_mut().resource_mut::<EditorState>().mode = mode;
            let element = app.world_mut().spawn(GraphElement::Block(IVec3::ZERO)).id();
            let before = app.world().resource::<GraphState>().graph.to_blog_text();
            click(&mut app, element, window, PointerButton::Middle);
            assert_eq!(
                app.world().resource::<GraphState>().graph.to_blog_text(),
                before
            );
            assert_eq!(app.world().resource::<EditorState>().selection_count(), 0);
        }
    }

    #[test]
    fn pending_blog_text_blocks_pointer_selection_and_deletion() {
        let (mut app, window) = observer_app();
        app.add_observer(on_element_click);
        let element = app.world_mut().spawn(GraphElement::Block(IVec3::ZERO)).id();
        {
            let mut text = app.world_mut().resource_mut::<ImportExportState>();
            text.mark_synchronized(0);
            text.set_bloq_buffer("BLOG 1.0\n0:".into());
        }
        for button in [PointerButton::Primary, PointerButton::Middle] {
            click(&mut app, element, window, button);
            assert_eq!(app.world().resource::<EditorState>().selection_count(), 0);
            assert_eq!(
                app.world().resource::<GraphState>().graph.blocks().count(),
                1
            );
        }
    }

    #[test]
    fn element_observer_ignores_selection_and_deletion_while_attributes_capture_input() {
        for (open_window, capture_input) in [(true, false), (false, true)] {
            let (mut app, window) = observer_app();
            app.add_observer(on_element_click);
            let element = app.world_mut().spawn(GraphElement::Block(IVec3::ZERO)).id();
            {
                let mut target = app.world_mut().resource_mut::<TargetState>();
                target.open_window = open_window;
                target.capture_input = capture_input;
            }
            let before = app.world().resource::<GraphState>().graph.to_blog_text();
            for button in [PointerButton::Primary, PointerButton::Middle] {
                click(&mut app, element, window, button);
                assert_eq!(app.world().resource::<EditorState>().selection_count(), 0);
                assert_eq!(app.world().resource::<GraphState>().revision, 0);
                assert_eq!(
                    app.world().resource::<GraphState>().graph.to_blog_text(),
                    before
                );
            }
            // Prove the observer is wired and resumes once the modal releases input.
            *app.world_mut().resource_mut::<TargetState>() = TargetState::default();
            click(&mut app, element, window, PointerButton::Primary);
            assert_eq!(app.world().resource::<EditorState>().selection_count(), 1);
        }
    }

    #[test]
    fn shift_click_on_preview_does_not_place_a_pipe() {
        for shift in [KeyCode::ShiftLeft, KeyCode::ShiftRight] {
            let (mut app, window) = observer_app();
            app.add_observer(on_preview_endpoint_click);
            *app.world_mut().resource_mut::<EditorState>() = EditorState {
                mode: EditorMode::Edit,
                placement_tool: PlacementTool::Pipe,
                pipe_start: Some(IVec3::ZERO),
                ..Default::default()
            };
            let preview = app
                .world_mut()
                .spawn(PreviewEndpointMesh {
                    endpoint: PreviewEndpoint::End,
                    source_pos: Some(IVec3::ZERO),
                    target_pos: Some(IVec3::Z),
                })
                .id();
            app.world_mut()
                .resource_mut::<ButtonInput<KeyCode>>()
                .press(shift);
            let before = app.world().resource::<GraphState>().graph.to_blog_text();
            click(&mut app, preview, window, PointerButton::Primary);
            assert_eq!(
                app.world().resource::<GraphState>().graph.to_blog_text(),
                before
            );
            assert_eq!(app.world().resource::<GraphState>().revision, 0);
            assert_eq!(
                app.world().resource::<EditorState>().pipe_start,
                Some(IVec3::ZERO)
            );
            app.world_mut()
                .resource_mut::<ButtonInput<KeyCode>>()
                .release(shift);
            click(&mut app, preview, window, PointerButton::Primary);
            assert!(
                app.world()
                    .resource::<GraphState>()
                    .graph
                    .has_pipe_between(IVec3::ZERO, IVec3::Z)
            );
        }
    }

    #[test]
    fn preview_observer_ignores_placement_while_attributes_capture_input() {
        for (open_window, capture_input) in [(true, false), (false, true)] {
            let (mut app, window) = observer_app();
            app.add_observer(on_preview_endpoint_click);
            *app.world_mut().resource_mut::<EditorState>() = EditorState {
                mode: EditorMode::Edit,
                placement_tool: PlacementTool::Pipe,
                pipe_start: Some(IVec3::ZERO),
                ..EditorState::default()
            };
            {
                let mut target = app.world_mut().resource_mut::<TargetState>();
                target.open_window = open_window;
                target.capture_input = capture_input;
            }
            let preview = app
                .world_mut()
                .spawn(PreviewEndpointMesh {
                    endpoint: PreviewEndpoint::End,
                    source_pos: Some(IVec3::ZERO),
                    target_pos: Some(IVec3::Z),
                })
                .id();
            let before = app.world().resource::<GraphState>().graph.to_blog_text();
            click(&mut app, preview, window, PointerButton::Primary);
            assert_eq!(
                app.world().resource::<EditorState>().pipe_start,
                Some(IVec3::ZERO)
            );
            assert_eq!(
                app.world().resource::<GraphState>().graph.to_blog_text(),
                before
            );
            assert_eq!(app.world().resource::<GraphState>().revision, 0);
            *app.world_mut().resource_mut::<TargetState>() = TargetState::default();
            click(&mut app, preview, window, PointerButton::Primary);
            assert!(
                app.world()
                    .resource::<GraphState>()
                    .graph
                    .has_pipe_between(IVec3::ZERO, IVec3::Z)
            );
            assert_eq!(app.world().resource::<GraphState>().revision, 1);
        }
    }
}
