//! Viewport pointer and keyboard interaction: hover/pick handling, block and
//! pipe placement, box selection, undo/redo, whole-graph and selection
//! transforms, and driving the hover/selection highlight overlays.

use crate::components::{EditorCamera, GraphElement, OriginalMaterial, PreviewEndpointMesh};
use crate::pipe_planning::{PipePlacementRequest, can_place_pipe, validate_pipe_edit_structure};
use crate::resources::{
    ActionEditState, ActionViewerState, BloqViewerState, CompileUiState, EditorMode, EditorState,
    EditorTabs, GraphEditDelta, GraphRenderState, GraphRotationAvailability, GraphState,
    GraphUiSummary, Notifications, PlacementTool, RenderedElement, RenderedMeshParts, TargetState,
    UiInputState, ZxViewerState,
};
use crate::systems::EditorUpdateSet;
use crate::systems::jobs::{EditorJobs, schedule_stabilizers_job, start_viewer_compile};
use crate::systems::ui::edit_element::{DEFAULT_PORT_RGB, format_rgb_hex};
use crate::systems::ui::{UiIntent, UiIntentBuffer, apply_fill_ports};
use crate::utils::{
    GraphAdjacencySnapshot, change_plane_height, cube_pipe_face_is_degenerate, intersect_plane,
    is_pipe_visible, pipe_placement_hadamard,
};
use bevy::picking::events::{
    PointerClick, PointerDragEnd, PointerDragStart, PointerOut, PointerOver,
};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
#[cfg(not(target_arch = "wasm32"))]
use bevy_egui::EguiClipboard;
#[cfg(any(target_arch = "wasm32", test))]
use bevy_egui::{PrimaryEguiContext, input::EguiInputEvent};
use bloq_graph::{
    Block, BlockGraph, BlockKind, PatchRotationKind, UDirection, WalkingBoundaryKind, WalkingKind,
};
use color_eyre::eyre::{self, ContextCompat, WrapErr, bail};
use glam::{IVec2, IVec3, Vec2};
use std::collections::HashSet;

/// Drag-rectangle selection in the viewport.
///
/// A drag starts `pending` on press and only becomes `active` once the pointer
/// moves far enough, so a plain click is not treated as a box select.
#[derive(Resource, Default, Clone)]
pub(crate) struct BoxSelectionState {
    pub(crate) active: bool,
    pub(crate) pending: bool,
    pub(crate) additive: bool,
    pub(crate) start: Vec2,
    pub(crate) current: Vec2,
}

impl BoxSelectionState {
    /// Arms a potential box select at `start`; `additive` keeps the existing
    /// selection.
    fn begin_pending(&mut self, start: Vec2, additive: bool) {
        self.active = false;
        self.pending = true;
        self.additive = additive;
        self.start = start;
        self.current = start;
    }

    pub(crate) fn update(&mut self, current: Vec2) {
        if self.active || self.pending {
            self.current = current;
        }
    }

    /// Promotes a pending drag to an active box select.
    pub(crate) fn activate(&mut self) {
        if self.pending {
            self.active = true;
            self.pending = false;
        }
    }

    /// Whether a drag is pending or active.
    pub(crate) fn is_tracking(&self) -> bool {
        self.active || self.pending
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

mod actions;
mod highlight;
mod picking;
mod placement;
mod probing;
mod transforms;
mod translation;
mod undo;
use highlight::*;
#[cfg(test)]
use picking::*;
use placement::*;
use probing::*;

pub(crate) use actions::{action_pick_system, pulse_action_candidate_material_system};
pub(crate) use highlight::{pipe_mode_selected_elements, sync_interaction_highlight_system};
pub(crate) use picking::{
    on_element_click, on_element_drag_end, on_element_drag_start, on_element_out, on_element_over,
    on_preview_endpoint_click, on_preview_endpoint_out, on_preview_endpoint_over,
};
pub(crate) use probing::{
    PATCH_ROTATION_MOVEMENTS, PIPE_HINT_OFFSETS, WALKING_MOVEMENTS, patch_rotation_candidate_kind,
    patch_rotation_pipe_hint_target, patch_rotation_start_has_candidate,
    pipe_candidate_at_with_snapshot, pipe_hint_target_with_snapshot, tall_cube_pipe_hint_target,
    walking_candidate_kind, walking_pipe_hint_target, walking_port_promotion_candidate,
    walking_start_has_candidate,
};
pub(crate) use transforms::{
    copy_selected_subgraph, insert_graph_without_overlap, rotate_selected_elements,
    rotation_degrees_label, translate_graph, translate_selected_elements,
    translate_selected_elements_by,
};
pub(crate) use translation::{TranslationDrag, TranslationTarget, draw_translation_gizmo_system};
pub(crate) use undo::undo_redo_system;

/// Viewport pointer/keyboard interaction. Its interaction-chain systems are
/// interleaved with camera and visuals work; the exact original order
/// (camera → undo/redo → pickability → edit → highlight → preview) is preserved
/// with explicit `.after` edges spread across those plugins.
pub(crate) struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiInputState>()
            .init_resource::<BoxSelectionState>()
            .init_resource::<TranslationDrag>()
            .init_resource::<picking::MiddleButtonDrag>()
            .add_systems(
                Update,
                translation::restore_translation_preview_system
                    .before(crate::systems::visuals::block_graph_visual_system)
                    .in_set(EditorUpdateSet::Rendering),
            )
            .add_systems(
                Update,
                (
                    undo_redo_system.after(crate::systems::camera::camera_control_system),
                    view_shortcut_system
                        .after(undo_redo_system)
                        .before(edit_system),
                    #[cfg(any(target_arch = "wasm32", test))]
                    web_view_paste_system
                        .after(undo_redo_system)
                        .before(edit_system),
                    translation::translation_drag_system
                        .after(crate::systems::visuals::sync_graph_pickability_system)
                        .before(edit_system),
                    edit_system.after(crate::systems::visuals::sync_graph_pickability_system),
                    action_pick_system.after(edit_system),
                    pulse_action_candidate_material_system.after(action_pick_system),
                    sync_interaction_highlight_system.after(action_pick_system),
                )
                    .in_set(EditorUpdateSet::Interaction),
            );
    }
}

const BOX_SELECTION_DRAG_THRESHOLD_PX: f32 = 4.0;

/// Max gap between element clicks for opening the attribute editor.
const DOUBLE_CLICK_SECS: f64 = 0.4;

fn view_shortcut_system(
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    graph_ui_summary: Res<GraphUiSummary>,
    #[cfg(not(target_arch = "wasm32"))] jobs: Res<EditorJobs>,
    ui_input: Res<UiInputState>,
    keys: Res<ButtonInput<KeyCode>>,
    #[cfg(not(target_arch = "wasm32"))] mut clipboard: ResMut<EguiClipboard>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    if ui_input.blocks_viewport_keyboard_input() || editor_state.mode != EditorMode::View {
        return;
    }

    if is_ctrl_pressed(&keys) {
        if keys.just_pressed(KeyCode::KeyC) && !graph_state.graph.is_empty() {
            intents.push(UiIntent::CopyGraphAsBlog);
        } else if keys.just_pressed(KeyCode::KeyD) {
            intents.push(duplicate_selection_intent(
                &graph_state.graph,
                editor_state.selected_element_set(),
            ));
        } else if keys.just_pressed(KeyCode::KeyV) {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let Some(buffer) = clipboard
                    .get_text()
                    .filter(|buffer| !buffer.trim().is_empty())
                else {
                    intents.error("Clipboard contains no BLOG text");
                    return;
                };
                if jobs.parse_blog_running() {
                    intents.error("BLOG parse is already running");
                    return;
                }
                intents.push(UiIntent::NewTab);
                intents.push(UiIntent::LoadBlogFile {
                    title: None,
                    buffer,
                });
            }
        }
        return;
    }

    if graph_state.graph.is_empty() {
        return;
    }
    if let Some((axis, step)) = view_translation_shortcut(&keys) {
        intents.push(UiIntent::TranslateGraph { axis, step });
    } else if keys.just_pressed(KeyCode::KeyR) {
        let step = match graph_ui_summary.rotation_availability {
            GraphRotationAvailability::Disabled => return,
            GraphRotationAvailability::HalfTurnsOnly => 2,
            GraphRotationAvailability::QuarterTurns => 1,
        };
        intents.push(UiIntent::RotateGraph {
            axis: editor_state.transform_axis,
            quarter_turns: if is_shift_pressed(&keys) { step } else { -step },
        });
    }
}

// Web clipboard text arrives with the browser's paste event, potentially after
// the Ctrl+V key frame. EguiClipboard::get_text() only returns cached contents.
#[cfg(any(target_arch = "wasm32", test))]
fn web_view_paste_system(
    mut events: MessageReader<EguiInputEvent>,
    primary_context: Query<Entity, With<PrimaryEguiContext>>,
    editor_state: Res<EditorState>,
    ui_input: Res<UiInputState>,
    jobs: Res<EditorJobs>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    for event in events.read() {
        let bevy_egui::egui::Event::Paste(buffer) = &event.event else {
            continue;
        };
        if !primary_context.contains(event.context)
            || ui_input.blocks_viewport_keyboard_input()
            || editor_state.mode != EditorMode::View
        {
            continue;
        }
        if buffer.trim().is_empty() {
            intents.error("Clipboard contains no BLOG text");
        } else if jobs.parse_blog_running() {
            intents.error("BLOG parse is already running");
        } else {
            intents.push(UiIntent::NewTab);
            intents.push(UiIntent::LoadBlogFile {
                title: None,
                buffer: buffer.clone(),
            });
        }
    }
}

pub(crate) fn duplicate_selection_intent(
    graph: &BlockGraph,
    selected_elements: &HashSet<GraphElement>,
) -> UiIntent {
    let copied = if !graph.is_empty()
        && graph
            .blocks()
            .all(|block| selected_elements.contains(&GraphElement::Block(block.pos())))
    {
        Ok(graph.clone())
    } else {
        copy_selected_subgraph(graph, selected_elements)
    };
    copied.map_or_else(
        |error| UiIntent::Error(format!("Could not duplicate selection: {error:#}")),
        |graph| UiIntent::InsertGraph(Box::new(graph)),
    )
}

fn view_translation_shortcut(keys: &ButtonInput<KeyCode>) -> Option<(UDirection, i32)> {
    let shift = is_shift_pressed(keys);
    if !shift {
        return keys
            .just_pressed(KeyCode::KeyV)
            .then_some((UDirection::Y, -1));
    }
    let xz = if keys.pressed(KeyCode::KeyZ) {
        UDirection::Z
    } else {
        UDirection::X
    };
    [
        (KeyCode::Comma, (xz, -1)),
        (KeyCode::Period, (xz, 1)),
        (KeyCode::Digit6, (UDirection::Y, 1)),
    ]
    .into_iter()
    .find_map(|(key, transform)| keys.just_pressed(key).then_some(transform))
}

/// The main viewport interaction system: dispatches keyboard shortcuts, block
/// and pipe placement, box selection, and mode-specific pointer handling for
/// the current frame.
fn edit_system(
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    mut jobs: ResMut<EditorJobs>,
    mut compile_ui: ResMut<CompileUiState>,
    mut circuit_viewer: ResMut<BloqViewerState>,
    mut zx_viewer: ResMut<ZxViewerState>,
    mut action_viewer: ResMut<ActionViewerState>,
    tabs: Res<EditorTabs>,
    mut notifications: ResMut<Notifications>,
    mut target_state: ResMut<TargetState>,
    gestures: (ResMut<BoxSelectionState>, Res<TranslationDrag>),
    ui_input: Res<UiInputState>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    camera_query: Single<(&Camera, &GlobalTransform), With<EditorCamera>>,
    window_query: Single<&Window, With<PrimaryWindow>>,
) {
    let (mut box_selection, translation_drag) = gestures;
    if translation_drag.is_tracking() {
        return;
    }
    let ctrl_pressed = is_ctrl_pressed(&keys);
    let keyboard_available = !ui_input.blocks_viewport_keyboard_input();
    let plain_key = plain_key_shortcuts_available(*ui_input, &keys);
    let pointer_available = !ui_input.blocks_viewport_pointer_input();

    if keyboard_available && keys.just_pressed(KeyCode::Escape) {
        editor_state.clear_hover();
        editor_state.clear_selection();
        box_selection.clear();
    }
    if keyboard_available
        && matches!(editor_state.mode, EditorMode::View | EditorMode::Edit)
        && ctrl_pressed
        && keys.just_pressed(KeyCode::KeyA)
    {
        editor_state.select_all_elements(&graph_state.graph);
        notifications.push_info(format!(
            "Selected {} element(s)",
            editor_state.selection_count()
        ));
    }

    if plain_key && keys.just_pressed(KeyCode::KeyJ) {
        change_plane_height(-1, &mut editor_state, &mut graph_state);
    }
    if plain_key && keys.just_pressed(KeyCode::KeyK) {
        change_plane_height(1, &mut editor_state, &mut graph_state);
    }
    if plain_key && (keys.just_pressed(KeyCode::Space) || keys.just_pressed(KeyCode::KeyV)) {
        editor_state.set_mode(EditorMode::View, &mut graph_state);
    }
    if plain_key && keys.just_pressed(KeyCode::KeyB) {
        editor_state.show_blog_buffer = !editor_state.show_blog_buffer;
    }
    if plain_key && keys.just_pressed(KeyCode::KeyM) {
        if editor_state.mode == EditorMode::Module {
            editor_state.set_mode(EditorMode::View, &mut graph_state);
        } else {
            editor_state.set_mode(EditorMode::Module, &mut graph_state);
        }
    }
    if plain_key && keys.just_pressed(KeyCode::KeyP) {
        editor_state.set_placement_tool(PlacementTool::Pipe);
        editor_state.set_mode(EditorMode::Edit, &mut graph_state);
    }
    if plain_key
        && handle_pipe_keyboard_placement(
            &keys,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        )
    {
        return;
    }

    if plain_key && editor_state.mode != EditorMode::Module && keys.just_pressed(KeyCode::KeyF) {
        apply_fill_ports(
            &mut graph_state,
            &mut editor_state,
            &mut target_state,
            &mut compile_ui,
            &mut circuit_viewer,
            &mut notifications,
        );
    }

    if plain_key
        && editor_state.mode == EditorMode::View
        && (keys.just_pressed(KeyCode::KeyD) || keys.just_pressed(KeyCode::Backspace))
    {
        delete_selected_elements(&mut editor_state, &mut graph_state, &mut notifications);
    }

    if plain_key
        && keys.just_pressed(KeyCode::KeyQ)
        && editor_state.show_stabilizers
        && !editor_state.stabilizers.is_empty()
    {
        editor_state.prev_stabilizer();
        graph_state.needs_rerender = true;
    }
    if plain_key
        && keys.just_pressed(KeyCode::KeyE)
        && editor_state.show_stabilizers
        && !editor_state.stabilizers.is_empty()
    {
        editor_state.next_stabilizer();
        graph_state.needs_rerender = true;
    }

    if plain_key && keys.just_pressed(KeyCode::KeyS) && editor_state.mode == EditorMode::View {
        if editor_state.toggle_stabilizers() {
            graph_state.needs_rerender = true;
        } else {
            schedule_stabilizers_job(
                &mut jobs,
                tabs.active,
                &graph_state,
                editor_state.view_current_layer_only,
                editor_state.plane_height,
                &mut notifications,
            );
        }
    }

    if plain_key && keys.just_pressed(KeyCode::KeyC) {
        if editor_state.mode == EditorMode::Bloq {
            editor_state.set_mode(EditorMode::View, &mut graph_state);
            circuit_viewer.set_hovered_node(None);
        } else {
            editor_state.set_mode(EditorMode::Bloq, &mut graph_state);
            start_viewer_compile(
                &mut jobs,
                tabs.active,
                &graph_state,
                &compile_ui,
                &mut circuit_viewer,
                &mut notifications,
            );
        }
    }

    if plain_key && keys.just_pressed(KeyCode::KeyX) {
        zx_viewer.toggle();
    }

    // After `handle_pipe_keyboard_placement` above, which claims `A` as the -X
    // pipe step while the Pipe tool is active.
    if plain_key && keys.just_pressed(KeyCode::KeyA) {
        action_viewer.toggle();
    }

    if !pointer_available {
        return;
    }

    let stride = editor_state.pipe_length + 1.0;
    let grid_height = editor_state.plane_height as f32 * stride;

    let (camera, camera_transform) = camera_query.into_inner();
    let window = window_query.into_inner();
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else {
        return;
    };

    let pick_grid_pos = intersect_plane(ray, grid_height, editor_state.pipe_length);
    editor_state.hovered_grid_pos = pick_grid_pos;

    if box_selection.is_tracking() {
        advance_box_selection_drag(&mut box_selection, cursor_pos);
        if buttons.just_released(MouseButton::Left) {
            if box_selection.active {
                let elements = graph_elements_in_screen_rect(
                    &graph_state.graph,
                    &editor_state,
                    camera,
                    camera_transform,
                    box_selection.start,
                    box_selection.current,
                );
                let count = elements.len();
                editor_state.select_elements(elements, box_selection.additive);
                notifications.push_info(format!("Box-selected {count} element(s)"));
            }
            box_selection.clear();
        } else if !buttons.pressed(MouseButton::Left) {
            box_selection.clear();
        }
        return;
    }

    if editor_state.mode == EditorMode::View
        && is_unmodified_box_select_input(&keys)
        && buttons.just_pressed(MouseButton::Left)
        && editor_state.hovered_element.is_none()
    {
        box_selection.begin_pending(cursor_pos, false);
        return;
    }

    if buttons.just_pressed(MouseButton::Left)
        && !ctrl_pressed
        && !is_alt_pressed(&keys)
        && editor_state.is_block_tool_active()
        && editor_state.hovered_element.is_none()
        && let Some(pos) = pick_grid_pos
    {
        if editor_state.block_kind.is_walking() {
            place_walking_click(pos, &mut editor_state, &mut graph_state, &mut notifications);
        } else if editor_state.block_kind.is_patch_rotation() {
            place_patch_rotation_click(
                pos,
                &mut editor_state,
                &mut graph_state,
                &mut notifications,
            );
        } else {
            place_single_block(pos, &mut editor_state, &mut graph_state, &mut notifications);
        }
    }
}

fn open_target_editor(
    element: GraphElement,
    graph_state: &GraphState,
    target_state: &mut TargetState,
    notifications: &mut Notifications,
) {
    let (tag, kind, hadamard, color, role) = match element {
        GraphElement::Block(pos) => {
            if let Some(block) = graph_state.graph.get_block(pos) {
                let color = block
                    .port_color()
                    .map_or(DEFAULT_PORT_RGB, |color| [color.r, color.g, color.b]);
                (
                    block.tag().unwrap_or("").to_owned(),
                    Some(block.kind()),
                    false,
                    color,
                    block.port_role().unwrap_or(bloq_graph::PortRole::Auto),
                )
            } else {
                notifications.push_error("Block not found");
                return;
            }
        }
        GraphElement::Pipe(u, v) => {
            if let Some(pipe) = graph_state.graph.get_pipe(u, v) {
                (
                    pipe.tag().unwrap_or("").to_owned(),
                    None,
                    pipe.is_hadamard(),
                    DEFAULT_PORT_RGB,
                    bloq_graph::PortRole::Auto,
                )
            } else {
                notifications.push_error("Pipe not found");
                return;
            }
        }
    };

    target_state.target = Some(element);
    target_state.tag_buffer = tag;
    target_state.port_color_hex_buffer = format_rgb_hex(color);
    target_state.port_role_buffer = role;
    target_state.pipe_hadamard_buffer = hadamard;
    if let Some(kind) = kind {
        target_state.block_kind_buffer = kind;
    }
    target_state.suppress_enter_confirm_until_release = true;
    target_state.open_window = true;
}

fn graph_elements_in_screen_rect(
    graph: &BlockGraph,
    editor_state: &EditorState,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    start: Vec2,
    end: Vec2,
) -> Vec<GraphElement> {
    let mut elements = Vec::new();
    for block in graph.blocks() {
        let pos = block.pos();
        if editor_state.view_current_layer_only && !block.occupies_layer(editor_state.plane_height)
        {
            continue;
        }
        let world = crate::utils::graph_to_world(pos.as_vec3(), editor_state.pipe_length);
        if screen_rect_contains_world_point(camera, camera_transform, start, end, world) {
            elements.push(GraphElement::Block(pos));
        }
    }

    for (u_pos, v_pos, _, _, _) in graph.pipe_endpoints_with_blocks() {
        if editor_state.view_current_layer_only
            && !is_pipe_visible(u_pos, v_pos, editor_state.plane_height)
        {
            continue;
        }
        let u_world = crate::utils::graph_to_world(u_pos.as_vec3(), editor_state.pipe_length);
        let v_world = crate::utils::graph_to_world(v_pos.as_vec3(), editor_state.pipe_length);
        if screen_rect_contains_world_point(
            camera,
            camera_transform,
            start,
            end,
            u_world.lerp(v_world, 0.5),
        ) {
            elements.push(GraphElement::Pipe(u_pos, v_pos).canonical());
        }
    }
    elements
}

fn screen_rect_contains_world_point(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    start: Vec2,
    end: Vec2,
    world: Vec3,
) -> bool {
    let Ok(screen) = camera.world_to_viewport(camera_transform, world) else {
        return false;
    };
    screen_rect_contains_point(start, end, screen)
}

fn screen_rect_contains_point(start: Vec2, end: Vec2, point: Vec2) -> bool {
    let min = start.min(end);
    let max = start.max(end);
    point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
}

fn advance_box_selection_drag(box_selection: &mut BoxSelectionState, current: Vec2) {
    box_selection.update(current);
    if box_selection.pending
        && box_selection_drag_exceeds_threshold(box_selection.start, box_selection.current)
    {
        box_selection.activate();
    }
}

fn box_selection_drag_exceeds_threshold(start: Vec2, current: Vec2) -> bool {
    start.distance_squared(current) >= BOX_SELECTION_DRAG_THRESHOLD_PX.powi(2)
}

/// Whether an unmodified single-key viewport shortcut should fire. Ctrl chords
/// belong to the application (`Ctrl+Shift+P` opens the command palette), so a
/// bare-letter shortcut must not also answer them.
fn plain_key_shortcuts_available(ui_input: UiInputState, keys: &ButtonInput<KeyCode>) -> bool {
    !ui_input.blocks_viewport_keyboard_input() && !is_ctrl_pressed(keys)
}

fn is_ctrl_pressed(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight])
}

fn is_shift_pressed(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
}

// Alt+Left-drag is reserved for camera orbit (see camera_control_system), so
// Alt must not also trigger box-select or block placement.
fn is_alt_pressed(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight])
}

fn is_unmodified_box_select_input(keys: &ButtonInput<KeyCode>) -> bool {
    !is_ctrl_pressed(keys) && !is_shift_pressed(keys) && !is_alt_pressed(keys)
}

pub(crate) fn is_hadamard_pipe_modifier_pressed(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::KeyH])
}

/// Hold-to-edit modifier: a single click opens the element's attribute editor
/// instead of selecting it or continuing a pipe. Double-click alone cannot,
/// since the Pipe tool spends the first click on an endpoint. `I` rather than
/// the mnemonic `E` because `E` is next-stabilizer.
pub(crate) fn is_attribute_edit_modifier_pressed(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::KeyI])
}

#[cfg(test)]
mod tests {
    use super::{
        GraphRotationAvailability, advance_box_selection_drag,
        collect_changed_interaction_elements, copy_selected_subgraph,
        delete_middle_clicked_element, handle_element_click, handle_pipe_keyboard_placement,
        insert_graph_without_overlap, is_hadamard_pipe_modifier_pressed,
        is_unmodified_box_select_input, open_pipe_start_editor_if_double_clicked,
        open_target_editor, patch_rotation_endpoint_for_pipe_click,
        patch_rotation_pipe_hint_target, picked_graph_element, pipe_endpoint_for_block_click,
        pipe_hint_target, pipe_mode_selected_elements, place_pipe, place_pipe_or_walking_target,
        plain_key_shortcuts_available, rotate_selected_elements, screen_rect_contains_point,
        tall_cube_pipe_hint_target, translate_graph, translate_selected_elements,
        view_translation_shortcut, walking_candidate_kind, walking_pipe_hint_target,
        walking_port_promotion_candidate, walking_start_has_candidate, web_view_paste_system,
    };
    use crate::components::GraphElement;
    use crate::resources::{
        ActionEditState, BoxSelectionState, EditorMode, EditorState, GraphEditDelta, GraphState,
        GraphUiSummary, Notifications, PlacementTool, TargetState, UiInputState,
    };
    use crate::systems::jobs::EditorJobs;
    use crate::systems::ui::{UiIntent, UiIntentBuffer};
    use crate::utils::GraphAdjacencySnapshot;
    use bevy::prelude::{App, ButtonInput, ChildOf, KeyCode, Update, World};
    use bevy_egui::{PrimaryEguiContext, egui, input::EguiInputEvent};
    use bloq_graph::{
        Action, Basis, Block, BlockGraph, BlockKind, CubeKind, Direction, FeedbackTarget,
        GalleryItem, MeasureTarget, PatchRotationKind, PauliBasis, Pipe, UDirection,
        WalkingBoundaryKind, WalkingKind,
    };
    use glam::{IVec2, IVec3, Vec2};
    use std::collections::HashSet;

    #[test]
    fn web_view_paste_imports_event_text_once_without_a_key_press() {
        let source = "BLOG 1.0\n\nmodule main {\n  0: XXZ [0, 0, 0]\n  1: ZXZ [-1, 0, 0]\n  2: XZZ [0, 1, 0]\n  3: ZXZ [1, 0, 0]\n  4: X [1, 0, 1]\n  0 -> 1\n  0 -> 2\n  0 -> 3\n  3 -> 4\n}\n";
        let mut app = App::new();
        app.add_message::<EguiInputEvent>()
            .init_resource::<EditorState>()
            .init_resource::<UiInputState>()
            .init_resource::<EditorJobs>()
            .init_resource::<UiIntentBuffer>()
            .add_systems(Update, web_view_paste_system);
        let context = app.world_mut().spawn(PrimaryEguiContext).id();

        // The key frame can finish before the browser delivers clipboard text.
        app.update();
        assert_eq!(
            app.world_mut()
                .resource_mut::<UiIntentBuffer>()
                .drain()
                .count(),
            0
        );
        app.world_mut().write_message(EguiInputEvent {
            context,
            event: egui::Event::Paste(source.into()),
        });
        app.update();
        let intents: Vec<_> = app
            .world_mut()
            .resource_mut::<UiIntentBuffer>()
            .drain()
            .collect();
        assert_eq!(intents.len(), 2);
        assert!(matches!(intents[0], UiIntent::NewTab));
        let UiIntent::LoadBlogFile { title, buffer } = &intents[1] else {
            panic!("paste must import the event's BLOG text");
        };
        assert!(title.is_none());
        assert_eq!(buffer, source);
        let graph = BlockGraph::from_text(buffer).unwrap().flatten().unwrap();
        assert_eq!((graph.block_count(), graph.pipe_count()), (5, 4));
        app.update();
        assert_eq!(
            app.world_mut()
                .resource_mut::<UiIntentBuffer>()
                .drain()
                .count(),
            0
        );

        // Text fields and non-view modes own their paste; never replay it later.
        for mode in [
            EditorMode::View,
            EditorMode::Edit,
            EditorMode::Module,
            EditorMode::Bloq,
        ] {
            app.world_mut().resource_mut::<EditorState>().mode = mode;
            app.world_mut()
                .resource_mut::<UiInputState>()
                .wants_keyboard_input = mode == EditorMode::View;
            app.world_mut().write_message(EguiInputEvent {
                context,
                event: egui::Event::Paste(source.into()),
            });
            app.update();
            assert_eq!(
                app.world_mut()
                    .resource_mut::<UiIntentBuffer>()
                    .drain()
                    .count(),
                0
            );
        }
        app.world_mut().resource_mut::<EditorState>().mode = EditorMode::View;
        app.world_mut()
            .resource_mut::<UiInputState>()
            .wants_keyboard_input = false;
        app.update();
        assert_eq!(
            app.world_mut()
                .resource_mut::<UiIntentBuffer>()
                .drain()
                .count(),
            0
        );
    }

    #[test]
    fn changed_interaction_elements_returns_only_hover_and_selection_delta() {
        let old_hover = HashSet::from([GraphElement::Block(IVec3::ZERO)]);
        let old_selected = HashSet::from([GraphElement::Pipe(IVec3::ZERO, IVec3::X).canonical()]);
        let new_hover = HashSet::from([GraphElement::Block(IVec3::Y)]);
        let new_selected = old_selected.clone();

        let mut changed = HashSet::new();
        collect_changed_interaction_elements(
            &old_hover,
            &old_selected,
            &new_hover,
            &new_selected,
            &mut changed,
        );

        assert_eq!(
            changed,
            HashSet::from([
                GraphElement::Block(IVec3::ZERO),
                GraphElement::Block(IVec3::Y)
            ])
        );
    }

    #[test]
    fn picked_graph_element_resolves_child_mesh_to_owner_element() {
        let mut world = World::new();
        let owner = world.spawn(GraphElement::Block(IVec3::ZERO)).id();
        let child = world.spawn(ChildOf(owner)).id();
        let mut element_query = world.query::<&GraphElement>();
        let mut child_of_query = world.query::<&ChildOf>();

        assert_eq!(
            picked_graph_element(
                child,
                &element_query.query(&world),
                &child_of_query.query(&world),
            ),
            Some(GraphElement::Block(IVec3::ZERO))
        );
    }

    #[test]
    fn pipe_tool_keeps_valid_hints_with_unrelated_incomplete_port() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XZZ)));

        assert_eq!(
            pipe_hint_target(&graph, IVec3::X, 2, false),
            Some((IVec3::X, IVec3::new(1, 1, 0)))
        );
    }

    #[test]
    fn pipe_tool_shows_temporal_hints_for_t_and_y_blocks() {
        for kind in [BlockKind::T, BlockKind::Y] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, kind));

            assert_eq!(
                pipe_hint_target(&graph, IVec3::ZERO, 4, false),
                Some((IVec3::ZERO, IVec3::Z))
            );
        }
    }

    #[test]
    fn pipe_tool_shows_incoming_hint_for_measurement_blocks() {
        for basis in [Basis::X, Basis::Z] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, BlockKind::Measurement(basis)));

            assert_eq!(
                pipe_hint_target(&graph, IVec3::ZERO, 5, false),
                Some((IVec3::ZERO, IVec3::NEG_Z))
            );
            assert_eq!(
                pipe_hint_target(&graph, IVec3::ZERO, 4, false),
                None,
                "a measurement is a terminal sink"
            );
        }
    }

    #[test]
    fn pipe_candidate_snapshot_rejects_definitely_full_ports() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::T));
        graph.add_block(Block::new(IVec3::X, BlockKind::Port));
        graph.add_block(Block::new(IVec3::NEG_X, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let snapshot = GraphAdjacencySnapshot::from_graph(&graph);

        assert_eq!(
            super::pipe_candidate_at_with_snapshot(
                &graph,
                &snapshot,
                IVec3::ZERO,
                IVec3::NEG_X,
                false,
            ),
            None
        );
    }

    #[test]
    fn pipe_candidate_snapshot_matches_direct_candidate_checks() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Port));
        graph.add_block(Block::new(
            IVec3::new(2, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 2, 0),
            BlockKind::Cube(CubeKind::ZZX),
        ));
        graph.add_block(Block::new(IVec3::new(0, 2, 1), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::X, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 2, 0), Direction::ZPLUS));
        let snapshot = GraphAdjacencySnapshot::from_graph(&graph);

        let mut starts = HashSet::new();
        for block in graph.blocks() {
            for endpoint in block
                .connectable_offsets()
                .into_iter()
                .map(|offset| block.pos() + offset)
            {
                starts.insert(endpoint);
                for offset in super::PIPE_HINT_OFFSETS {
                    starts.insert(endpoint + offset);
                }
            }
        }

        for start in starts {
            for offset in super::PIPE_HINT_OFFSETS {
                assert_eq!(
                    super::pipe_candidate_at_with_snapshot(&graph, &snapshot, start, offset, false),
                    super::pipe_candidate_at(&graph, start, offset, false),
                    "candidate mismatch for start={start:?} offset={offset:?}"
                );
            }
        }
    }

    #[test]
    fn translate_graph_moves_pipes_and_preserves_actions() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::XPLUS));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(IVec3::new(0, 0, 0)),
                name: "m0".to_string(),
            })
            .expect("test action should apply");

        let result = translate_graph(&graph, UDirection::Y, -2).expect("translation succeeds");

        assert!(result.has_block_at(IVec3::new(0, -2, 0)));
        assert!(result.has_block_at(IVec3::new(1, -2, 0)));
        assert!(result.has_pipe_between(IVec3::new(0, -2, 0), IVec3::new(1, -2, 0)));
        assert_eq!(
            result.actions(),
            &[Action::Measure {
                target: MeasureTarget::Node(IVec3::new(0, -2, 0)),
                name: "m0".to_string(),
            }]
        );
    }

    #[test]
    fn translating_selected_elements_moves_actions_with_their_targets() {
        let mut graph = GalleryItem::T.build().flatten().unwrap();
        let selected = GraphElement::all_in(&graph).collect();
        let outside = IVec3::new(10, 0, 0);
        graph.add_block(Block::new(outside, BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(outside),
                name: "outside".into(),
            })
            .expect("stationary action is valid");
        let mut expected_actions = graph.actions();
        for action in &mut expected_actions[..2] {
            *action = action.with_shift(IVec3::Y);
        }

        let result =
            translate_selected_elements(&graph, &selected, UDirection::Y, 1).expect("translate");

        assert_eq!(result.graph.actions(), expected_actions);
        assert_eq!(result.dropped_action_count, 0);
        result.graph.validate().expect("translated graph is valid");
    }

    #[test]
    fn translations_allow_a_stale_feedback_target() {
        let source = GalleryItem::T.build().flatten().unwrap();
        let (mut graph, selected) = insert_graph_without_overlap(&source, &source).unwrap();
        let missing = IVec3::new(999, 0, 0);
        graph
            .set_actions_lenient(vec![Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::X,
                    target: missing,
                    direction: None,
                }],
                condition: None,
            }])
            .unwrap();
        assert!(graph.action_graph_error().is_some());
        translate_graph(&graph, UDirection::Z, -1).unwrap();

        let result =
            translate_selected_elements(&graph, &selected.into_iter().collect(), UDirection::Z, -1)
                .expect("stale actions do not block geometry edits");

        assert!(result.graph.action_graph_error().is_some());
    }

    #[test]
    fn partial_translation_rejects_action_coordinate_overflow() {
        for (bound, step) in [(i32::MAX, 1), (i32::MIN, -1)] {
            let pos = IVec3::new(bound, 0, 0);
            for action in [
                Action::Measure {
                    target: MeasureTarget::Node(pos),
                    name: "m".into(),
                },
                Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::X,
                        target: pos,
                        direction: None,
                    }],
                    condition: None,
                },
            ] {
                let mut graph = BlockGraph::new();
                graph.add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
                graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
                graph.set_actions_lenient(vec![action]).unwrap();
                let before = graph.to_blog_text();
                let selected = HashSet::from([GraphElement::Block(pos)]);
                let _error = translate_selected_elements(&graph, &selected, UDirection::X, step)
                    .unwrap_err();
                assert_eq!(graph.to_blog_text(), before);
            }
        }
    }

    #[test]
    fn translate_selected_elements_moves_closed_subgraph_and_updates_selection() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::new(5, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let selected = HashSet::from([
            GraphElement::Block(IVec3::ZERO),
            GraphElement::Block(IVec3::X),
            GraphElement::Pipe(IVec3::ZERO, IVec3::X),
        ]);

        let result =
            translate_selected_elements(&graph, &selected, UDirection::Y, 1).expect("translate");

        assert!(result.graph.has_block_at(IVec3::Y));
        assert!(result.graph.has_block_at(IVec3::new(1, 1, 0)));
        assert!(result.graph.has_block_at(IVec3::new(5, 0, 0)));
        assert!(result.graph.has_pipe_between(IVec3::Y, IVec3::new(1, 1, 0)));
        assert!(
            result
                .selected_elements
                .contains(&GraphElement::Block(IVec3::Y))
        );
        assert!(
            result
                .selected_elements
                .contains(&GraphElement::Pipe(IVec3::Y, IVec3::new(1, 1, 0)).canonical())
        );
    }

    #[test]
    fn translate_selected_elements_rejects_boundary_pipes() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let selected = HashSet::from([GraphElement::Block(IVec3::ZERO)]);

        let err = translate_selected_elements(&graph, &selected, UDirection::Y, 1)
            .expect_err("boundary pipe should block partial transform");

        assert!(err.to_string().contains("crossing"));
    }

    #[test]
    fn branch_components_move_only_with_the_whole_graph() {
        let source = GalleryItem::CCZGateTeleport.build().flatten().unwrap();
        let selected = GraphElement::all_in(&source).collect();
        let mut graph = source.clone();
        graph.add_block(Block::new(
            IVec3::new(100, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));

        let error = translate_selected_elements(&graph, &selected, UDirection::Y, 1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Branch arms cannot be transformed separately")
        );
        let moved = translate_selected_elements(&source, &selected, UDirection::Y, 1).unwrap();
        assert_eq!(
            moved.graph.branch_definitions().len(),
            source.branch_definitions().len()
        );
        moved.graph.validate_source().unwrap();
    }

    #[test]
    fn copied_selection_keeps_internal_pipes_and_omits_boundary_pipes() {
        let mut graph = BlockGraph::new();
        for position in [IVec3::ZERO, IVec3::X, IVec3::new(2, 0, 0)] {
            graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        }
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::X, Direction::XPLUS));
        let selected = HashSet::from([
            GraphElement::Block(IVec3::ZERO),
            GraphElement::Block(IVec3::X),
        ]);

        let copied = copy_selected_subgraph(&graph, &selected).expect("copy selection");

        assert_eq!(copied.blocks().count(), 2);
        assert_eq!(copied.pipes().count(), 1);
        assert!(copied.has_pipe_between(IVec3::ZERO, IVec3::X));
        assert!(!copied.has_block_at(IVec3::new(2, 0, 0)));
    }

    #[test]
    fn every_gallery_graph_can_be_duplicated_without_overlap() {
        for (entry, shown_true) in
            GalleryItem::iter().flat_map(|entry| [(entry, true), (entry, false)])
        {
            let mut source = entry.build().flatten().unwrap();
            // Without branches, changing the displayed arm repeats the same graph.
            if !shown_true && source.branch_definitions().is_empty() {
                continue;
            }
            let names = source
                .branch_definitions()
                .iter()
                .map(|region| region.name.clone())
                .collect::<Vec<_>>();
            for name in names {
                source.set_shown_branch_arm(&name, shown_true).unwrap();
            }
            let (graph, selection) = insert_graph_without_overlap(&source, &source)
                .unwrap_or_else(|err| panic!("insert {entry}: {err:#}"));

            assert_eq!(
                (
                    graph.blocks().count(),
                    graph.branch_definitions().len(),
                    graph.actions().len(),
                ),
                (
                    source.blocks().count() * 2,
                    source.branch_definitions().len() * 2,
                    source.actions().len() * 2,
                ),
                "{entry}"
            );
            assert_eq!(
                selection.len(),
                source.blocks().count() + source.pipes().count(),
                "{entry}"
            );
            graph
                .validate_source()
                .unwrap_or_else(|err| panic!("validate duplicated {entry}: {err:#}"));
        }
    }

    #[test]
    fn translate_selected_elements_resolves_walking_pipe_endpoints_to_owner_block() {
        let mut graph = BlockGraph::new();
        let walking =
            WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::new(1, 1)).expect("valid walking");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Walking(walking)));
        graph.add_block(Block::new(IVec3::new(1, 1, 2), BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(0, 0, -1), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(1, 1, 1), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        let selected = HashSet::from([
            GraphElement::Block(IVec3::ZERO),
            GraphElement::Block(IVec3::new(1, 1, 2)),
            GraphElement::Block(IVec3::new(0, 0, -1)),
            GraphElement::Pipe(IVec3::new(1, 1, 1), IVec3::new(1, 1, 2)).canonical(),
            GraphElement::Pipe(IVec3::ZERO, IVec3::new(0, 0, -1)).canonical(),
        ]);

        let result =
            translate_selected_elements(&graph, &selected, UDirection::X, 1).expect("translate");

        assert!(result.graph.has_block_at(IVec3::X));
        assert!(result.graph.has_block_at(IVec3::new(2, 1, 2)));
        assert!(
            result
                .graph
                .has_pipe_between(IVec3::new(2, 1, 1), IVec3::new(2, 1, 2))
        );
        assert!(
            result.selected_elements.contains(
                &GraphElement::Pipe(IVec3::new(2, 1, 1), IVec3::new(2, 1, 2)).canonical()
            )
        );
    }

    #[test]
    fn pipe_tool_can_hint_walking_promotion_from_past_connected_port() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::new(0, 0, -1), BlockKind::Port));
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));

        let candidate = walking_port_promotion_candidate(&graph, IVec3::ZERO, IVec3::new(1, 1, 1))
            .expect("past-connected start port can become walking");

        assert_eq!(candidate.boundary(), WalkingBoundaryKind::ZXZ);
        assert_eq!(candidate.movement(), IVec2::new(1, 1));
        assert!(
            walking_port_promotion_candidate(&graph, IVec3::ZERO, IVec3::new(0, 0, 1)).is_none()
        );
    }

    #[test]
    fn pipe_tool_preserves_boundary_when_promoting_port_to_walking() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS));

        let candidate = walking_port_promotion_candidate(&graph, IVec3::ZERO, IVec3::new(0, 1, 1))
            .expect("past-connected start port can become walking");

        assert_eq!(candidate.boundary(), WalkingBoundaryKind::XZZ);
    }

    #[test]
    fn pipe_tool_rejects_walking_promotion_when_end_pipe_exists() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 1), BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 1), Direction::ZPLUS));

        assert!(
            walking_port_promotion_candidate(&graph, IVec3::ZERO, IVec3::new(1, 0, 1)).is_none()
        );
    }

    #[test]
    fn pipe_tool_uses_walking_end_when_start_has_no_pipe_targets() {
        let mut graph = BlockGraph::new();
        let walking =
            WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::new(0, 1)).expect("valid walking");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Walking(walking)));
        graph.add_block(Block::new(IVec3::new(0, 0, -1), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));

        assert_eq!(
            pipe_endpoint_for_block_click(&graph, IVec3::ZERO, None, false),
            IVec3::new(0, 1, 1)
        );
        assert!(
            pipe_hint_target(&graph, IVec3::ZERO, 5, false).is_none(),
            "walking start already has its only valid temporal pipe"
        );
        assert_eq!(
            walking_pipe_hint_target(&graph, IVec3::ZERO, 1, false),
            Some((IVec3::new(0, 1, 1), IVec3::new(0, 1, 2)))
        );
    }

    #[test]
    fn pipe_tool_exposes_patch_rotation_temporal_endpoint_hints() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, IVec2::X).expect("valid patch rotation");
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(1, 0, 2), BlockKind::Port));

        assert_eq!(
            patch_rotation_endpoint_for_pipe_click(&graph, IVec3::ZERO, None, false),
            Some(IVec3::ZERO)
        );
        assert_eq!(
            patch_rotation_pipe_hint_target(&graph, IVec3::ZERO, 0, false),
            Some((IVec3::ZERO, IVec3::NEG_Z))
        );
        assert_eq!(
            patch_rotation_pipe_hint_target(&graph, IVec3::ZERO, 1, false),
            Some((IVec3::new(1, 0, 1), IVec3::new(1, 0, 2)))
        );

        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        assert_eq!(
            patch_rotation_endpoint_for_pipe_click(&graph, IVec3::ZERO, None, false),
            Some(IVec3::new(1, 0, 1))
        );
    }

    #[test]
    fn pipe_tool_exposes_scaled_cube_top_temporal_hint() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );

        assert_eq!(
            tall_cube_pipe_hint_target(&graph, IVec3::ZERO, 4, false),
            Some((IVec3::new(0, 0, 1), IVec3::new(0, 0, 2)))
        );
        assert_eq!(
            pipe_endpoint_for_block_click(&graph, IVec3::ZERO, Some(IVec3::new(0, 0, 2)), false),
            IVec3::new(0, 0, 1)
        );
    }

    #[test]
    fn pipe_tool_hides_scaled_cube_spatial_hints() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );

        for index in 0..4 {
            assert_eq!(pipe_hint_target(&graph, IVec3::ZERO, index, false), None);
        }
        assert_eq!(
            pipe_hint_target(&graph, IVec3::ZERO, 5, false),
            Some((IVec3::ZERO, IVec3::NEG_Z))
        );
    }

    #[test]
    fn pipe_tool_shows_scaled_cube_spatial_hint_to_same_scale_cube() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        graph.add_block(
            Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );

        assert_eq!(
            pipe_hint_target(&graph, IVec3::ZERO, 0, false),
            Some((IVec3::ZERO, IVec3::X))
        );
    }

    #[test]
    fn pipe_mode_treats_start_owner_block_as_selected() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        let editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(IVec3::new(0, 0, 1)),
            ..Default::default()
        };

        assert!(
            pipe_mode_selected_elements(&editor_state, &graph)
                .contains(&GraphElement::Block(IVec3::ZERO))
        );
    }

    #[test]
    fn pipe_tool_double_click_opens_block_and_pipe_editors() {
        let start = IVec3::ZERO;
        let end = IVec3::X;
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(start, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(end, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            ..Default::default()
        };
        let mut target_state = TargetState::default();
        let mut action_edit = ActionEditState::default();
        let mut notifications = Notifications::default();

        handle_element_click(
            GraphElement::Block(start),
            1.0,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut action_edit,
            &mut notifications,
            &ButtonInput::default(),
        );
        assert_eq!(editor_state.pipe_start, Some(start));

        assert!(open_pipe_start_editor_if_double_clicked(
            start,
            1.2,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut notifications,
        ));

        assert_eq!(target_state.target, Some(GraphElement::Block(start)));
        assert!(target_state.open_window);
        assert_eq!(editor_state.pipe_start, None);

        graph_state
            .graph
            .add_pipe(Pipe::new(start, Direction::XPLUS));
        target_state = TargetState::default();
        handle_element_click(
            GraphElement::Pipe(start, end),
            2.0,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut action_edit,
            &mut notifications,
            &ButtonInput::default(),
        );
        handle_element_click(
            GraphElement::Pipe(start, end),
            2.2,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut action_edit,
            &mut notifications,
            &ButtonInput::default(),
        );

        assert_eq!(target_state.target, Some(GraphElement::Pipe(start, end)));
        assert!(target_state.open_window);
    }

    /// The Pipe tool consumes the first click as a pipe endpoint, so double-click
    /// could not reach the attribute editor; the hold-to-edit modifier must.
    #[test]
    fn hold_to_edit_modifier_opens_the_editor_on_a_single_pipe_tool_click() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            ..Default::default()
        };
        let mut target_state = TargetState::default();
        let mut action_edit = ActionEditState::default();
        let mut notifications = Notifications::default();
        let mut keys = ButtonInput::default();
        keys.press(KeyCode::KeyI);

        handle_element_click(
            GraphElement::Block(IVec3::ZERO),
            1.0,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut action_edit,
            &mut notifications,
            &keys,
        );

        assert!(target_state.open_window);
        assert_eq!(target_state.target, Some(GraphElement::Block(IVec3::ZERO)));
        assert_eq!(editor_state.pipe_start, None);
    }

    /// Selection must not steal a click an armed action pick is waiting for.
    #[test]
    fn an_armed_action_pick_consumes_the_click_instead_of_selecting() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState::default();
        let mut target_state = TargetState::default();
        let mut action_edit = ActionEditState::default();
        action_edit.arm(crate::resources::ActionKind::Measure);
        let mut notifications = Notifications::default();

        handle_element_click(
            GraphElement::Block(IVec3::ZERO),
            1.0,
            &mut editor_state,
            &graph_state,
            &mut target_state,
            &mut action_edit,
            &mut notifications,
            &ButtonInput::default(),
        );

        assert_eq!(editor_state.selection_count(), 0);
        assert!(action_edit.picking().is_none());
        assert!(action_edit.draft.is_some());
    }

    #[test]
    fn walking_start_preview_allows_parallel_soft_corridor_start_only_when_placeable() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).expect("valid walking kind"),
            ),
        ));
        let mut editor_state = crate::resources::EditorState {
            block_kind: BlockKind::Walking(WalkingKind::DEFAULT),
            ..Default::default()
        };

        assert!(walking_start_has_candidate(
            &editor_state,
            &graph,
            IVec3::new(0, 1, 0)
        ));
        assert!(!walking_start_has_candidate(
            &editor_state,
            &graph,
            IVec3::new(0, 1, 1)
        ));

        editor_state.walking_start = Some(IVec3::new(1, 0, 0));
        assert!(!walking_start_has_candidate(
            &editor_state,
            &graph,
            IVec3::new(0, 1, 1)
        ));
        assert!(walking_candidate_kind(&editor_state, &graph, IVec3::new(0, 1, 1)).is_none());
    }

    #[test]
    fn rotate_selected_elements_rotates_around_selection_pivot() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        let selected = HashSet::from([
            GraphElement::Block(IVec3::ZERO),
            GraphElement::Block(IVec3::Z),
        ]);

        let result = rotate_selected_elements(&graph, &selected, UDirection::X, 1).expect("rotate");

        assert_eq!(result.graph.blocks().count(), 2);
        assert_eq!(result.graph.pipes().count(), 1);
        assert!(result.graph.has_block_at(IVec3::new(0, 1, 1)));
        assert!(result.graph.has_block_at(IVec3::Z));
        assert!(result.graph.has_pipe_between(IVec3::new(0, 1, 1), IVec3::Z));
        assert!(
            result
                .selected_elements
                .contains(&GraphElement::Block(IVec3::new(0, 1, 1)))
        );
        assert!(
            result
                .selected_elements
                .contains(&GraphElement::Pipe(IVec3::new(0, 1, 1), IVec3::Z).canonical())
        );
    }

    #[test]
    fn rotate_selected_elements_allows_incomplete_graphs() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        let selected = HashSet::from([GraphElement::Block(IVec3::ZERO)]);

        rotate_selected_elements(&graph, &selected, UDirection::Z, 1).unwrap();
    }

    #[test]
    fn keyboard_pipe_placement_rejects_coordinate_overflow() {
        let pos = IVec3::new(i32::MAX, 0, 0);
        let mut editor = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(pos),
            last_pipe_placement: Some((pos - IVec3::X, pos)),
            ..Default::default()
        };
        let mut graph = GraphState::default();
        graph
            .graph
            .add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
        let before = graph.graph.to_blog_text();
        for key in [KeyCode::KeyD, KeyCode::KeyR] {
            let mut keys = ButtonInput::default();
            keys.press(key);
            assert!(handle_pipe_keyboard_placement(
                &keys,
                &mut editor,
                &mut graph,
                &mut Notifications::default()
            ));
            assert_eq!(graph.graph.to_blog_text(), before);
            assert_eq!(graph.revision, 0);
        }
    }

    #[test]
    fn keyboard_pipe_placement_advances_on_success_and_stays_on_failure() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(IVec3::ZERO),
            ..EditorState::default()
        };
        let mut notifications = Notifications::default();
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::ArrowUp);

        assert!(handle_pipe_keyboard_placement(
            &keys,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        ));

        assert!(graph_state.graph.has_pipe_between(IVec3::ZERO, IVec3::Z));
        assert_eq!(editor_state.pipe_start, Some(IVec3::Z));
        assert_eq!(
            graph_state.graph.get_block(IVec3::Z).map(Block::kind),
            Some(BlockKind::Port)
        );

        editor_state.pipe_start = Some(IVec3::ZERO);
        assert!(handle_pipe_keyboard_placement(
            &keys,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        ));

        assert_eq!(editor_state.pipe_start, Some(IVec3::ZERO));
    }

    #[test]
    fn repeat_pipe_placement_reuses_previous_delta() {
        let p1 = IVec3::ZERO;
        let p2 = IVec3::X;
        let p3 = p2 + IVec3::X;
        let p4 = p3 + IVec3::X;
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(p1, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(p1),
            ..EditorState::default()
        };
        let mut notifications = Notifications::default();

        place_pipe_or_walking_target(
            p1,
            p2,
            false,
            true,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        );

        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::KeyR);
        assert!(handle_pipe_keyboard_placement(
            &keys,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        ));
        keys.release_all();
        keys.clear();
        keys.press(KeyCode::KeyR);
        assert!(handle_pipe_keyboard_placement(
            &keys,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        ));

        assert!(graph_state.graph.has_pipe_between(p2, p3));
        assert!(graph_state.graph.has_pipe_between(p3, p4));
        assert_eq!(editor_state.pipe_start, Some(p4));
        assert_eq!(editor_state.last_pipe_placement, Some((p3, p4)));
    }

    #[test]
    fn screen_rect_contains_point_accepts_any_drag_direction() {
        assert!(screen_rect_contains_point(
            Vec2::new(20.0, 40.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(10.0, 20.0),
        ));
        assert!(!screen_rect_contains_point(
            Vec2::new(20.0, 40.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(30.0, 20.0),
        ));
    }

    /// `Ctrl+Shift+P` opens the palette; it must not also reach the viewport's
    /// bare `P`, which would switch to the Pipe tool on the way.
    #[test]
    fn ctrl_chords_do_not_reach_bare_key_viewport_shortcuts() {
        let free = UiInputState::default();
        let mut keys = ButtonInput::<KeyCode>::default();
        assert!(plain_key_shortcuts_available(free, &keys));

        keys.press(KeyCode::ControlLeft);
        assert!(!plain_key_shortcuts_available(free, &keys));

        keys.release(KeyCode::ControlLeft);
        let typing = UiInputState {
            wants_keyboard_input: true,
            ..UiInputState::default()
        };
        assert!(!plain_key_shortcuts_available(typing, &keys));
    }

    #[test]
    fn box_select_input_uses_unmodified_drag() {
        let mut keys = ButtonInput::<KeyCode>::default();
        assert!(is_unmodified_box_select_input(&keys));

        keys.press(KeyCode::ControlLeft);
        keys.press(KeyCode::ShiftLeft);
        assert!(!is_unmodified_box_select_input(&keys));
    }

    #[test]
    fn pending_box_select_activates_only_after_drag_threshold() {
        let mut box_selection = BoxSelectionState::default();
        box_selection.begin_pending(Vec2::new(10.0, 10.0), false);

        advance_box_selection_drag(&mut box_selection, Vec2::new(12.0, 12.0));
        assert!(box_selection.pending);
        assert!(!box_selection.active);

        advance_box_selection_drag(&mut box_selection, Vec2::new(15.0, 10.0));
        assert!(box_selection.active);
        assert!(!box_selection.pending);
        assert!(!box_selection.additive);
    }

    fn rotation_availability_for(kind: BlockKind) -> GraphRotationAvailability {
        let mut graph_state = GraphState::default();
        graph_state.graph.add_block(Block::new(IVec3::ZERO, kind));
        graph_state.commit();
        GraphUiSummary::from_graph_state(&graph_state).rotation_availability
    }

    #[test]
    fn y_graphs_are_limited_to_half_turns() {
        assert_eq!(
            rotation_availability_for(BlockKind::Y),
            GraphRotationAvailability::HalfTurnsOnly
        );
    }

    #[test]
    fn orientation_pinned_graphs_disable_rotation() {
        for kind in [BlockKind::T, BlockKind::Measurement(Basis::X)] {
            assert_eq!(
                rotation_availability_for(kind),
                GraphRotationAvailability::Disabled
            );
        }
    }

    #[test]
    fn view_translation_keys_map_to_graph_axes() {
        let cases: &[(&[KeyCode], (UDirection, i32))] = &[
            (&[KeyCode::ShiftLeft, KeyCode::Comma], (UDirection::X, -1)),
            (&[KeyCode::ShiftLeft, KeyCode::Digit6], (UDirection::Y, 1)),
            (&[KeyCode::KeyV], (UDirection::Y, -1)),
            (
                &[KeyCode::KeyZ, KeyCode::ShiftLeft, KeyCode::Period],
                (UDirection::Z, 1),
            ),
        ];
        for &(pressed, expected) in cases {
            let mut keys = ButtonInput::default();
            for key in pressed {
                keys.press(*key);
            }
            assert_eq!(view_translation_shortcut(&keys), Some(expected));
        }
    }

    #[test]
    fn hadamard_pipe_modifier_uses_h_key() {
        let mut keys = ButtonInput::<KeyCode>::default();
        assert!(!is_hadamard_pipe_modifier_pressed(&keys));

        keys.press(KeyCode::KeyH);
        assert!(is_hadamard_pipe_modifier_pressed(&keys));
    }

    #[test]
    fn place_pipe_into_empty_endpoint_commits_unknown_delta() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        let mut notifications = Notifications::default();

        assert!(place_pipe(
            IVec3::ZERO,
            IVec3::Z,
            false,
            &mut graph_state,
            &mut notifications
        ));

        assert!(graph_state.graph.has_pipe_between(IVec3::ZERO, IVec3::Z));
        assert_eq!(
            graph_state.graph.get_block(IVec3::Z).map(Block::kind),
            Some(BlockKind::Port)
        );
        assert!(matches!(graph_state.edit_delta, GraphEditDelta::Unknown));
    }

    #[test]
    fn middle_click_deletes_in_edit_and_view_but_middle_drag_does_not() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState::default();
        let mut notifications = Notifications::default();

        delete_middle_clicked_element(
            GraphElement::Block(IVec3::ZERO),
            false,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        );
        editor_state.set_mode(EditorMode::Edit, &mut graph_state);
        delete_middle_clicked_element(
            GraphElement::Block(IVec3::X),
            false,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        );
        delete_middle_clicked_element(
            GraphElement::Block(IVec3::Y),
            true,
            &mut editor_state,
            &mut graph_state,
            &mut notifications,
        );

        assert!(!graph_state.graph.has_block_at(IVec3::ZERO));
        assert!(!graph_state.graph.has_block_at(IVec3::X));
        assert!(graph_state.graph.has_block_at(IVec3::Y));
    }

    #[test]
    fn deleting_a_pipe_retires_the_port_it_created() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        // A standalone port the user placed by hand, untouched by the deletion.
        graph_state
            .graph
            .add_block(Block::new(IVec3::new(5, 0, 0), BlockKind::Port));
        graph_state.commit();
        let mut notifications = Notifications::default();

        // Piping into empty space auto-creates the far-end port.
        assert!(place_pipe(
            IVec3::ZERO,
            IVec3::Z,
            false,
            &mut graph_state,
            &mut notifications
        ));
        assert!(graph_state.graph.has_block_at(IVec3::Z));

        let mut editor_state = EditorState::default();
        editor_state.set_mode(EditorMode::View, &mut graph_state);
        editor_state.select_elements(
            vec![GraphElement::Pipe(IVec3::ZERO, IVec3::Z).canonical()],
            false,
        );
        super::delete_selected_elements(&mut editor_state, &mut graph_state, &mut notifications);

        assert!(!graph_state.graph.has_block_at(IVec3::Z));
        graph_state
            .graph
            .can_place_block(&Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)))
            .unwrap();
        assert!(graph_state.graph.has_block_at(IVec3::ZERO));
        assert!(graph_state.graph.has_block_at(IVec3::new(5, 0, 0)));
        assert!(matches!(graph_state.edit_delta, GraphEditDelta::Unknown));
    }

    #[test]
    fn deleting_a_block_retires_the_ports_its_pipes_fed() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state.commit();
        let mut notifications = Notifications::default();
        assert!(place_pipe(
            IVec3::ZERO,
            IVec3::Z,
            false,
            &mut graph_state,
            &mut notifications
        ));

        let mut editor_state = EditorState::default();
        editor_state.set_mode(EditorMode::View, &mut graph_state);
        editor_state.select_elements(vec![GraphElement::Block(IVec3::ZERO)], false);
        super::delete_selected_elements(&mut editor_state, &mut graph_state, &mut notifications);

        assert!(graph_state.graph.is_empty());
    }

    #[test]
    fn open_target_editor_seeds_pipe_hadamard_and_resets_for_block() {
        let pipe_start = IVec3::ZERO;
        let pipe_end = IVec3::X;
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pipe_start, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(pipe_end, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_pipe(Pipe::new(pipe_start, Direction::XPLUS).with_hadamard());

        let mut target_state = TargetState::default();
        let mut notifications = Notifications::default();

        open_target_editor(
            GraphElement::Pipe(pipe_start, pipe_end),
            &graph_state,
            &mut target_state,
            &mut notifications,
        );

        assert!(target_state.open_window);
        assert!(target_state.suppress_enter_confirm_until_release);
        assert_eq!(
            target_state.target,
            Some(GraphElement::Pipe(pipe_start, pipe_end))
        );
        assert!(target_state.pipe_hadamard_buffer);

        open_target_editor(
            GraphElement::Block(pipe_start),
            &graph_state,
            &mut target_state,
            &mut notifications,
        );

        assert_eq!(target_state.target, Some(GraphElement::Block(pipe_start)));
        assert!(!target_state.pipe_hadamard_buffer);
        assert!(notifications.toasts.is_empty());
    }
}
