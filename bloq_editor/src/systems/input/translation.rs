//! Drag preview moves scene entities only. The document changes once, on drop.

use super::*;
use crate::module_authoring::{
    PortSnap, module_elements, module_instance_at, module_snap_points, translate_module,
};
use crate::resources::{CentralViewport, EditorTabId, ImportExportState};
use crate::utils::{graph_to_world, world_to_graph_space};
use bevy_egui::{EguiContexts, egui};
use std::collections::HashMap;

#[derive(Clone)]
pub(crate) enum TranslationTarget {
    Selection(HashSet<GraphElement>),
    Module(String),
}

#[derive(Resource, Default)]
pub(crate) struct TranslationDrag {
    current: Option<Drag>,
    pub(crate) suppress_click: bool,
}

struct Drag {
    tab: EditorTabId,
    revision: u64,
    mode: EditorMode,
    target: TranslationTarget,
    elements: HashSet<GraphElement>,
    poses: HashMap<Entity, Vec3>,
    cursor_start: Vec2,
    anchor: Vec3,
    normal: Vec3,
    hit_start: Vec3,
    axis: Option<UDirection>,
    active: bool,
    requested: IVec3,
    displayed: IVec3,
    status: Result<String, String>,
    compact: bool,
    snap_points: Vec<PortSnap>,
}

impl TranslationDrag {
    pub(crate) fn is_tracking(&self) -> bool {
        self.current.is_some()
    }
}

impl Drag {
    fn update_preview(&mut self, graph: &GraphState, delta: IVec3) {
        if self.requested == delta {
            return;
        }
        self.requested = delta;
        self.displayed = delta;
        let preview = match &self.target {
            _ if delta == IVec3::ZERO => Ok("Release without moving".into()),
            TranslationTarget::Selection(selected) => {
                translate_selected_elements_by(&graph.graph, selected, delta)
                    .map(|_| "Release to move selection".into())
            }
            TranslationTarget::Module(name) => graph
                .resolved_graph()
                .and_then(|program| translate_module(&program, name, delta, self.compact))
                .map(|result| {
                    self.displayed = result.offset;
                    if result.connections == 0 {
                        "Release to move module group".into()
                    } else {
                        format!(
                            "Release to join {} ports · {}",
                            result.connections,
                            if result.compacted {
                                "compact"
                            } else {
                                "connection cubes"
                            }
                        )
                    }
                }),
        };
        self.status = preview.map_err(|error| format!("{error:#}"));
    }
}

/// Restore the old scene before visual synchronization can reuse its entities
/// for a different tab, graph revision, or display scale.
pub(crate) fn restore_translation_preview_system(
    drag: Res<TranslationDrag>,
    mut meshes: Query<&mut Transform, With<GraphElement>>,
) {
    if let Some(current) = &drag.current {
        for (entity, position) in &current.poses {
            if let Ok(mut transform) = meshes.get_mut(*entity) {
                transform.translation = *position;
            }
        }
    }
}

fn plane_hit(ray: Ray3d, anchor: Vec3, normal: Vec3) -> Option<Vec3> {
    let denominator = ray.direction.dot(normal);
    if denominator.abs() < 1e-5 {
        return None;
    }
    let distance = (anchor - ray.origin).dot(normal) / denominator;
    (distance.is_finite() && distance >= 0.0).then(|| ray.get_point(distance))
}

fn drag_delta(start: Vec3, hit: Vec3, axis: Option<UDirection>, pipe_length: f32) -> Vec3 {
    let delta = world_to_graph_space(hit - start, pipe_length);
    axis.map_or(delta, |axis| {
        axis.to_ivec3().as_vec3() * delta[axis.index()]
    })
}

fn snap_offset(
    points: &[PortSnap],
    offset: Vec3,
    axis: Option<UDirection>,
    project: impl Fn(Vec3) -> Option<Vec2>,
) -> IVec3 {
    points
        .iter()
        .filter_map(|point| {
            if axis.is_some_and(|axis| point.offset != axis.to_ivec3() * point.offset[axis.index()])
            {
                return None;
            }
            let distance = (point.offset.as_vec3() - offset).length_squared();
            let screen_distance = project(point.moving.as_vec3() + offset)
                .zip(project(point.fixed.as_vec3()))
                .map(|(a, b)| a.distance_squared(b));
            // Free dragging can change depth when the user brings two visible ports
            // together. Axis handles remain constrained even during a magnetic catch.
            (distance <= 0.8_f32.powi(2)
                || (axis.is_none() && screen_distance.is_some_and(|d| d <= 14.0_f32.powi(2))))
            .then_some((point, screen_distance.unwrap_or(distance * 256.0), distance))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
        .map_or_else(|| offset.round().as_ivec3(), |(point, _, _)| point.offset)
}

fn selection_center(
    graph: &BlockGraph,
    elements: &HashSet<GraphElement>,
    pipe_length: f32,
) -> Option<Vec3> {
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    for block in graph
        .blocks()
        .filter(|block| elements.contains(&GraphElement::Block(block.pos())))
    {
        min = min.min(block.pos());
        max = max.max(block.pos());
    }
    (min.x <= max.x).then(|| graph_to_world((min.as_vec3() + max.as_vec3()) * 0.5, pipe_length))
}

fn axis_handles(
    camera: &Camera,
    transform: &GlobalTransform,
    anchor: Vec3,
) -> Vec<(UDirection, Vec2, Vec2)> {
    let Ok(origin) = camera.world_to_viewport(transform, anchor) else {
        return Vec::new();
    };
    [UDirection::X, UDirection::Y, UDirection::Z]
        .into_iter()
        .filter_map(|axis| {
            let direction = graph_to_world(axis.to_ivec3().as_vec3(), 0.0);
            let tip = camera
                .world_to_viewport(transform, anchor + direction)
                .ok()?;
            let projected = tip - origin;
            (projected.length_squared() > 0.25)
                .then(|| (axis, origin, origin + projected.normalize() * 64.0))
        })
        .collect()
}

fn hit_handle(cursor: Vec2, handles: &[(UDirection, Vec2, Vec2)]) -> Option<UDirection> {
    handles
        .iter()
        .filter_map(|(axis, start, end)| {
            let vector = end - start;
            let t = ((cursor - start).dot(vector) / vector.length_squared()).clamp(0.18, 1.0);
            let distance = cursor.distance_squared(*start + vector * t);
            (distance <= 8.0_f32.powi(2)).then_some((*axis, distance))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(axis, _)| axis)
}

/// Captures the chosen blocks plus their internal pipes, exactly as the
/// keyboard transformation does. A pipe by itself needs its endpoint blocks.
fn drag_target(
    editor: &EditorState,
    graph: &GraphState,
    files: &ImportExportState,
    element: Option<GraphElement>,
) -> eyre::Result<Option<(TranslationTarget, HashSet<GraphElement>)>> {
    if editor.mode == EditorMode::Module {
        let program = graph.resolved_graph()?;
        let name = match element {
            Some(element) => module_instance_at(&program, element)?,
            None => files.module_ui.selected_instance().map(str::to_owned),
        };
        return name
            .map(|name| {
                Ok((
                    TranslationTarget::Module(name.clone()),
                    module_elements(&program, &name)?,
                ))
            })
            .transpose();
    }
    if graph.is_composed() {
        return Ok(None);
    }
    let mut selected = editor.selected_element_set().clone();
    if let Some(element) = element
        && !selected.contains(&element)
    {
        selected.clear();
        match element {
            GraphElement::Block(_) => {
                selected.insert(element);
            }
            GraphElement::Pipe(a, b) => {
                for position in [a, b] {
                    if let Some(block) = graph.graph.get_endpoint_block(position) {
                        selected.insert(GraphElement::Block(block.pos()));
                    }
                }
            }
        }
    }
    if selected.is_empty() {
        return Ok(None);
    }
    if graph.selection_touches_pending_branch_cut(&selected) {
        bail!(
            "Captured arm's input boundary cannot move separately; finish or cancel the branch first"
        );
    }
    let subgraph = copy_selected_subgraph(&graph.graph, &selected)?;
    Ok(Some((
        TranslationTarget::Selection(selected),
        GraphElement::all_in(&subgraph).collect(),
    )))
}

pub(crate) fn translation_drag_system(
    mut drag: ResMut<TranslationDrag>,
    mut editor: ResMut<EditorState>,
    graph: Res<GraphState>,
    tabs: Res<EditorTabs>,
    files: Res<ImportExportState>,
    action_edit: Res<ActionEditState>,
    ui: Res<UiInputState>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    camera: Single<(&Camera, &GlobalTransform), With<EditorCamera>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut meshes: Query<(Entity, &GraphElement, &mut Transform), Without<EditorCamera>>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    // Suppression belongs to the released drag, not the next click (including
    // Ctrl+click, which bypasses dragging below).
    if buttons.just_pressed(MouseButton::Left) {
        drag.suppress_click = false;
    }
    let cancelled = drag.current.as_ref().is_some_and(|current| {
        current.tab != tabs.active
            || current.revision != graph.revision
            || current.mode != editor.mode
    }) || keys.just_pressed(KeyCode::Escape)
        || !window.focused
        || is_alt_pressed(&keys)
        || is_ctrl_pressed(&keys)
        || buttons.pressed(MouseButton::Right)
        || buttons.pressed(MouseButton::Middle);
    if cancelled {
        if drag.current.take().is_some_and(|current| current.active) {
            drag.suppress_click = true;
        }
        return;
    }
    let (camera, camera_transform) = camera.into_inner();
    let cursor = window.cursor_position();
    let ray = cursor.and_then(|cursor| camera.viewport_to_world(camera_transform, cursor).ok());
    if drag.current.is_none() {
        if !buttons.just_pressed(MouseButton::Left) {
            return;
        }
        if ui.blocks_viewport_pointer_input()
            || action_edit.picking().is_some()
            || is_attribute_edit_modifier_pressed(&keys)
            || !matches!(
                editor.mode,
                EditorMode::View | EditorMode::Edit | EditorMode::Module
            )
        {
            return;
        }
        let (Some(cursor), Some(ray)) = (cursor, ray) else {
            return;
        };
        let center = selection_center(
            &graph.graph,
            editor.selected_element_set(),
            editor.pipe_length,
        );
        let axis = center
            .and_then(|center| hit_handle(cursor, &axis_handles(camera, camera_transform, center)));
        let element = if axis.is_some() {
            None
        } else {
            editor.hovered_element
        };
        if axis.is_none() && (element.is_none() || editor.is_pipe_tool_active()) {
            return;
        }
        let (target, elements) = match drag_target(&editor, &graph, &files, element) {
            Ok(Some(target)) => target,
            Ok(None) => return,
            Err(error) => {
                intents.error(format!("Cannot move: {error:#}"));
                return;
            }
        };
        let Some(center) = selection_center(&graph.graph, &elements, editor.pipe_length) else {
            return;
        };
        // Grab at the clicked object's depth; using the group's center makes
        // a distant port slip away from the pointer in a perspective camera.
        let anchor = match (axis, element) {
            (None, Some(GraphElement::Block(position))) => {
                graph_to_world(position.as_vec3(), editor.pipe_length)
            }
            (None, Some(GraphElement::Pipe(a, b))) => {
                graph_to_world((a.as_vec3() + b.as_vec3()) * 0.5, editor.pipe_length)
            }
            _ => center,
        };
        let forward = *camera_transform.forward();
        let normal = axis.map_or(forward, |axis| {
            let direction = graph_to_world(axis.to_ivec3().as_vec3(), 0.0);
            (forward - direction * forward.dot(direction)).normalize_or_zero()
        });
        let Some(hit_start) = plane_hit(ray, anchor, normal) else {
            return;
        };
        let snap_points = match &target {
            TranslationTarget::Module(name) => graph
                .resolved_graph()
                .and_then(|program| module_snap_points(&program, name))
                .unwrap_or_default(),
            TranslationTarget::Selection(_) => Vec::new(),
        };
        drag.current = Some(Drag {
            tab: tabs.active,
            revision: graph.revision,
            mode: editor.mode,
            target,
            elements,
            poses: HashMap::new(),
            cursor_start: cursor,
            anchor,
            normal,
            hit_start,
            axis,
            active: false,
            requested: IVec3::ZERO,
            displayed: IVec3::ZERO,
            status: Ok("Release without moving".into()),
            compact: files.module_ui.compact_connections,
            snap_points,
        });
    }
    let current = drag
        .current
        .as_mut()
        .expect("every path that leaves `current` empty has returned above");
    if let (Some(cursor), Some(ray)) = (cursor, ray)
        && let Some(hit) = plane_hit(ray, current.anchor, current.normal)
    {
        if cursor.distance_squared(current.cursor_start) >= BOX_SELECTION_DRAG_THRESHOLD_PX.powi(2)
        {
            current.active = true;
        }
        if current.active {
            editor.replace_selection(current.elements.iter().copied());
            editor.pipe_start = None;
            editor.last_element_click = None;
            let raw = drag_delta(current.hit_start, hit, current.axis, editor.pipe_length);
            let delta = match &current.target {
                TranslationTarget::Module(_) => {
                    snap_offset(&current.snap_points, raw, current.axis, |position| {
                        camera
                            .world_to_viewport(
                                camera_transform,
                                graph_to_world(position, editor.pipe_length),
                            )
                            .ok()
                    })
                }
                TranslationTarget::Selection(_) => raw.round().as_ivec3(),
            };
            current.update_preview(&graph, delta);
        }
    }
    let active = current.active;
    if !buttons.pressed(MouseButton::Left) {
        let current = drag
            .current
            .take()
            .expect("the drag was borrowed from this field above");
        if current.active
            && buttons.just_released(MouseButton::Left)
            && !ui.blocks_viewport_pointer_input()
            && cursor.is_some()
        {
            match current.status {
                Ok(_) if current.requested != IVec3::ZERO => {
                    intents.push(UiIntent::FinishTranslation {
                        tab: current.tab,
                        revision: current.revision,
                        target: current.target,
                        offset: current.requested,
                        compact: current.compact,
                    })
                }
                Err(message) => intents.error(format!("Move rejected: {message}")),
                _ => {}
            }
        }
    } else if active {
        let offset = graph_to_world(current.displayed.as_vec3(), editor.pipe_length);
        for (entity, element, mut transform) in &mut meshes {
            if current.elements.contains(element) {
                let position = *current.poses.entry(entity).or_insert(transform.translation);
                transform.translation = position + offset;
            }
        }
    }
    if active {
        drag.suppress_click = true;
    }
}

pub(crate) fn draw_translation_gizmo_system(
    mut contexts: EguiContexts,
    editor: Res<EditorState>,
    graph: Res<GraphState>,
    drag: Res<TranslationDrag>,
    central: Res<CentralViewport>,
    camera: Single<(&Camera, &GlobalTransform), With<EditorCamera>>,
) {
    if !matches!(
        editor.mode,
        EditorMode::View | EditorMode::Edit | EditorMode::Module
    ) || (graph.is_composed() && editor.mode != EditorMode::Module)
    {
        return;
    }
    let Some(mut anchor) = selection_center(
        &graph.graph,
        editor.selected_element_set(),
        editor.pipe_length,
    ) else {
        return;
    };
    if let Some(current) = &drag.current {
        anchor += graph_to_world(current.displayed.as_vec3(), editor.pipe_length);
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("translation_gizmo"),
        ))
        .with_clip_rect(central.0);
    let (camera, transform) = camera.into_inner();
    for (axis, start, end) in axis_handles(camera, transform, anchor) {
        let color = match axis {
            UDirection::X => egui::Color32::from_rgb(222, 70, 62),
            UDirection::Y => egui::Color32::from_rgb(60, 170, 90),
            UDirection::Z => egui::Color32::from_rgb(64, 140, 235),
        };
        let start = egui::pos2(start.x, start.y);
        let end = egui::pos2(end.x, end.y);
        painter.arrow(
            start,
            end - start,
            egui::Stroke::new(5.0, egui::Color32::from_black_alpha(180)),
        );
        painter.arrow(start, end - start, egui::Stroke::new(2.5, color));
        painter.circle_filled(end, 6.0, color);
        painter.text(
            end + (end - start).normalized() * 12.0,
            egui::Align2::CENTER_CENTER,
            axis.to_string(),
            egui::FontId::monospace(13.0),
            color,
        );
    }
    if let Some(current) = &drag.current
        && current.active
    {
        let (message, color) = match &current.status {
            Ok(message) => (message, egui::Color32::from_rgb(80, 220, 130)),
            Err(message) => (message, egui::Color32::from_rgb(255, 110, 100)),
        };
        let text = format!("Δ {}  ·  {}\nEsc cancels", current.displayed, message);
        let text = painter.layout(
            text,
            egui::FontId::proportional(14.0),
            color,
            (central.0.width() - 32.0).max(120.0),
        );
        let origin =
            central.0.center_bottom() - egui::vec2(text.size().x * 0.5, text.size().y + 16.0);
        painter.rect_filled(
            egui::Rect::from_min_size(origin, text.size()).expand(7.0),
            5.0,
            egui::Color32::from_black_alpha(220),
        );
        painter.galley(origin, text, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_click_after_drag_can_extend_the_selection() {
        use bevy::camera::RenderTarget;
        use bevy::picking::{
            backend::HitData,
            pointer::{Location, PointerId},
        };

        let mut app = App::new();
        app.init_resource::<EditorState>()
            .init_resource::<GraphState>()
            .init_resource::<EditorTabs>()
            .init_resource::<ImportExportState>()
            .init_resource::<ActionEditState>()
            .init_resource::<TargetState>()
            .init_resource::<Notifications>()
            .init_resource::<super::super::picking::MiddleButtonDrag>()
            .init_resource::<UiInputState>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<UiIntentBuffer>()
            .init_resource::<Time>()
            .insert_resource(TranslationDrag {
                current: None,
                suppress_click: true,
            })
            .add_systems(Update, translation_drag_system)
            .add_observer(super::super::picking::on_element_click);
        app.world_mut().resource_mut::<GraphState>().graph =
            crate::module_authoring::tests::stage().flatten().unwrap();
        app.world_mut()
            .resource_mut::<EditorState>()
            .select_element(GraphElement::Block(IVec3::ZERO), false);
        let camera = app
            .world_mut()
            .spawn((EditorCamera, Camera::default(), GlobalTransform::default()))
            .id();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let cube = app.world_mut().spawn(GraphElement::Block(IVec3::Z)).id();
        let click = PointerClick {
            entity: cube,
            pointer: Pointer::new(
                PointerId::Mouse,
                Location {
                    target: RenderTarget::default().normalize(Some(window)).unwrap(),
                    position: Vec2::ZERO,
                },
            ),
            button: PointerButton::Primary,
            hit: HitData::new(camera, 1.0, None, None),
            duration: std::time::Duration::ZERO,
            count: 1,
        };
        // Releasing the previous drag must not also select its hit element.
        app.world_mut().trigger(click.clone());
        assert_eq!(app.world().resource::<EditorState>().selection_count(), 1);

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::ControlLeft);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .release(MouseButton::Left);
        app.world_mut().trigger(click);
        assert_eq!(
            app.world().resource::<EditorState>().selection_count(),
            2,
            "Ctrl+click after a drag must add the clicked block"
        );
    }

    #[test]
    fn edit_mode_keeps_and_changes_selection_for_mouse_moves() {
        let mut graph = GraphState {
            graph: crate::module_authoring::tests::stage().flatten().unwrap(),
            ..Default::default()
        };
        let mut editor = EditorState::default();
        editor.select_all_elements(&graph.graph);
        editor.set_mode(EditorMode::Edit, &mut graph);
        assert_eq!(editor.selection_count(), 5);
        editor.clear_selection();
        let mut keys = ButtonInput::default();
        let mut target = TargetState::default();
        let mut action = ActionEditState::default();
        let mut notifications = Notifications::default();
        super::super::picking::handle_element_click(
            GraphElement::Block(IVec3::Z),
            1.0,
            &mut editor,
            &graph,
            &mut target,
            &mut action,
            &mut notifications,
            &keys,
        );
        assert_eq!(editor.selection_count(), 1);
        keys.press(KeyCode::ControlLeft);
        super::super::picking::handle_element_click(
            GraphElement::Block(IVec3::ZERO),
            2.0,
            &mut editor,
            &graph,
            &mut target,
            &mut action,
            &mut notifications,
            &keys,
        );
        assert_eq!(editor.selection_count(), 2);
        let delta = IVec3::new(2, 3, 4);
        let _error =
            translate_selected_elements_by(&graph.graph, editor.selected_element_set(), delta)
                .unwrap_err();
        editor.select_all_elements(&graph.graph);
        let moved =
            translate_selected_elements_by(&graph.graph, editor.selected_element_set(), delta)
                .unwrap();
        assert!(moved.graph.has_block_at(delta + IVec3::Z));
        assert_eq!(moved.selected_elements.len(), 5);
    }

    #[test]
    fn drag_planes_and_handles_cover_all_three_graph_axes() {
        let anchor = Vec3::ZERO;
        let ray = Ray3d::new(Vec3::new(2.0, 3.0, 10.0), Dir3::NEG_Z);
        let hit = plane_hit(ray, anchor, Vec3::Z).unwrap();
        assert_eq!(drag_delta(anchor, hit, None, 1.0), Vec3::new(1.0, 0.0, 1.5));
        for axis in [UDirection::X, UDirection::Y, UDirection::Z] {
            let hit = graph_to_world(axis.to_ivec3().as_vec3() * 3.0, 1.0);
            assert_eq!(
                drag_delta(anchor, hit, Some(axis), 1.0),
                axis.to_ivec3().as_vec3() * 3.0
            );
        }
        assert!(plane_hit(ray, anchor, Vec3::X).is_none());
        let handles = [(UDirection::X, Vec2::ZERO, Vec2::new(64.0, 0.0))];
        assert_eq!(
            hit_handle(Vec2::new(48.0, 3.0), &handles),
            Some(UDirection::X)
        );
        assert_eq!(hit_handle(Vec2::ZERO, &handles), None);
    }

    #[test]
    fn port_snapping_can_correct_depth_but_never_breaks_an_axis_constraint() {
        let points = [PortSnap {
            moving: IVec3::ZERO,
            fixed: IVec3::new(3, 0, 2),
            offset: IVec3::new(3, 0, 2),
        }];
        let raw = Vec3::new(3.0, -3.0, 2.0);
        let project = |position: Vec3| Some(Vec2::new(position.x, position.z) * 50.0);
        assert_eq!(
            snap_offset(&points, raw, None, project),
            IVec3::new(3, 0, 2)
        );
        assert_eq!(
            snap_offset(
                &points,
                Vec3::new(3.0, 0.0, 0.0),
                Some(UDirection::X),
                project
            ),
            IVec3::new(3, 0, 0)
        );
        assert_eq!(
            snap_offset(&points, Vec3::new(3.2, 0.2, 2.1), None, |_| None),
            IVec3::new(3, 0, 2)
        );
        assert_eq!(
            snap_offset(&points, Vec3::new(4.0, 0.0, 2.0), None, project),
            IVec3::new(4, 0, 2)
        );
    }
}
