//! Orbit-camera control: turning mouse input into [`CameraSettings`] and framing
//! the camera to fit a graph.

use crate::components::{CameraSettings, EditorCamera};
use crate::resources::UiInputState;
use crate::systems::EditorUpdateSet;
use crate::utils::{graph_bounds, graph_to_world};
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bloq_graph::BlockGraph;

/// Orbit-camera control. Runs after the UI-input gate so it can stand down while
/// egui owns the pointer; it heads the interaction chain (see `InputPlugin`).
pub(crate) struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            camera_control_system
                .after(crate::systems::ui::sync_ui_input_state_system)
                .in_set(EditorUpdateSet::Interaction),
        );
    }
}

const MIN_CAMERA_RADIUS: f32 = 1.0;
/// Farthest the orbit camera may dolly out.
pub(crate) const MAX_CAMERA_RADIUS: f32 = 8_000.0;
/// The camera's vertical field of view, in radians.
pub(crate) const CAMERA_DEFAULT_FOV_Y: f32 = std::f32::consts::PI / 4.0;
const CAMERA_FIT_MARGIN: f32 = 1.35;
const CAMERA_FIT_MIN_RADIUS: f32 = 8.0;
const CAMERA_WHEEL_ZOOM_RATIO: f32 = 0.08;
const CAMERA_MIN_WHEEL_STEP: f32 = 0.25;
const CAMERA_MAX_WHEEL_STEP: f32 = MAX_CAMERA_RADIUS * CAMERA_WHEEL_ZOOM_RATIO;
const CAMERA_PIXEL_SCROLL_TO_LINES: f32 = 0.01;
const CAMERA_MIN_VIEWPORT_HEIGHT: f32 = 240.0;
const CAMERA_PAN_VIEWPORT_FRACTION: f32 = 1.0;
/// Orbit sensitivity: radians of rotation per pixel of pointer drag.
const CAMERA_ORBIT_RADIANS_PER_PIXEL: f32 = 0.004;
const CAMERA_MIN_BETA: f32 = 0.1;
const CAMERA_MAX_BETA: f32 = std::f32::consts::PI - CAMERA_MIN_BETA;

/// Treat the unobstructed canvas as the main view, extending its projection
/// beneath the surrounding UI. All coordinates share egui's logical units;
/// SubCameraView uses their ratios, so this also works at non-unit DPI scales.
pub(crate) fn viewport_camera_view(
    screen: bevy_egui::egui::Rect,
    viewport: bevy_egui::egui::Rect,
) -> (Option<bevy::camera::SubCameraView>, f32) {
    let shortest = viewport.width().min(viewport.height());
    if shortest < 1.0 {
        return (None, CAMERA_DEFAULT_FOV_Y);
    }
    let fov = 2.0 * ((CAMERA_DEFAULT_FOV_Y * 0.5).tan() * viewport.height() / shortest).atan();
    (
        Some(bevy::camera::SubCameraView {
            full_size: UVec2::new(
                viewport.width().round() as u32,
                viewport.height().round() as u32,
            ),
            offset: Vec2::new(
                screen.left() - viewport.left(),
                screen.top() - viewport.top(),
            ),
            size: UVec2::new(
                screen.width().round() as u32,
                screen.height().round() as u32,
            ),
        }),
        fov,
    )
}

fn clamp_camera_radius(radius: f32) -> f32 {
    radius.clamp(MIN_CAMERA_RADIUS, MAX_CAMERA_RADIUS)
}

fn graph_world_center(graph: &BlockGraph, pipe_length: f32) -> Option<Vec3> {
    let (min, max) = graph_bounds(graph)?;
    let center_graph = (min.as_vec3() + max.as_vec3()) * 0.5;
    Some(graph_to_world(center_graph, pipe_length))
}

fn graph_world_size(graph: &BlockGraph, pipe_length: f32) -> Option<Vec3> {
    let (min, max) = graph_bounds(graph)?;
    let stride = pipe_length + 1.0;
    let span = max.as_vec3() - min.as_vec3();
    Some(span * stride + Vec3::ONE)
}

fn fit_radius_for_graph(graph: &BlockGraph, pipe_length: f32) -> f32 {
    let Some(world_size) = graph_world_size(graph, pipe_length) else {
        return CameraSettings::default().radius;
    };

    let sphere_radius = world_size.length() * 0.5;
    let distance = sphere_radius / (CAMERA_DEFAULT_FOV_Y * 0.5).tan();
    clamp_camera_radius((distance * CAMERA_FIT_MARGIN).max(CAMERA_FIT_MIN_RADIUS))
}

fn normalized_wheel_delta(event: MouseWheel) -> f32 {
    match event.unit {
        MouseScrollUnit::Line => event.y,
        MouseScrollUnit::Pixel => event.y * CAMERA_PIXEL_SCROLL_TO_LINES,
    }
}

fn zoom_radius_for_scroll(radius: f32, scroll_lines: f32) -> f32 {
    let step =
        (radius * CAMERA_WHEEL_ZOOM_RATIO).clamp(CAMERA_MIN_WHEEL_STEP, CAMERA_MAX_WHEEL_STEP);
    clamp_camera_radius(radius - scroll_lines * step)
}

fn pan_step_for_radius(radius: f32, viewport_height: f32) -> f32 {
    let viewport_height = viewport_height.max(CAMERA_MIN_VIEWPORT_HEIGHT);
    let visible_height = 2.0 * radius * (CAMERA_DEFAULT_FOV_Y * 0.5).tan();
    visible_height * CAMERA_PAN_VIEWPORT_FRACTION / viewport_height
}

/// Orbits the camera by a pointer `delta`, clamping the polar angle so it never
/// flips over the poles.
pub(crate) fn orbit_camera_settings(cam: &mut CameraSettings, delta: Vec2) {
    cam.alpha -= delta.x * CAMERA_ORBIT_RADIANS_PER_PIXEL;
    cam.beta -= delta.y * CAMERA_ORBIT_RADIANS_PER_PIXEL;
    cam.beta = cam.beta.clamp(CAMERA_MIN_BETA, CAMERA_MAX_BETA);
}

/// Dollies the camera by scroll input, with a step proportional to the current
/// radius so zoom feels consistent at any distance.
pub(crate) fn zoom_camera_settings(cam: &mut CameraSettings, scroll_lines: f32) {
    cam.radius = zoom_radius_for_scroll(cam.radius, scroll_lines);
}

/// Pans the focus point in the camera's screen plane, with a step scaled to the
/// radius and viewport height.
pub(crate) fn pan_camera_settings(
    cam: &mut CameraSettings,
    delta: Vec2,
    viewport_height: f32,
    right: Vec3,
    up: Vec3,
) {
    let pan_step = pan_step_for_radius(cam.radius, viewport_height);
    cam.focus += -right * delta.x * pan_step + up * delta.y * pan_step;
}

/// Computes camera settings that frame the whole graph, falling back to the
/// default framing for an empty graph.
pub(crate) fn camera_settings_for_graph(graph: &BlockGraph, pipe_length: f32) -> CameraSettings {
    let mut settings = CameraSettings {
        radius: fit_radius_for_graph(graph, pipe_length),
        ..CameraSettings::default()
    };
    if let Some(focus) = graph_world_center(graph, pipe_length) {
        settings.focus = focus;
    }
    settings
}

/// Reframes the camera to fit `graph` and writes the resulting transform.
pub(crate) fn reset_camera_to_graph(
    transform: &mut Transform,
    camera_settings: &mut CameraSettings,
    graph: &BlockGraph,
    pipe_length: f32,
) {
    *camera_settings = camera_settings_for_graph(graph, pipe_length);
    apply_camera_setting(transform, camera_settings);
}

/// Turns viewport mouse input into camera orbit, pan, and zoom, standing down
/// while egui owns the pointer.
pub(crate) fn camera_control_system(
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
    ui_input: Res<UiInputState>,
    translation_drag: Res<crate::systems::input::TranslationDrag>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    camera_query: Single<(&mut Transform, &mut CameraSettings, &Camera), With<EditorCamera>>,
    window_query: Single<&Window, With<PrimaryWindow>>,
) {
    let (mut transform, mut cam, camera) = camera_query.into_inner();
    let window = window_query.into_inner();

    if ui_input.blocks_viewport_pointer_input() || translation_drag.is_tracking() {
        mouse_motion_events.clear();
        mouse_wheel_events.clear();
        return;
    }

    // Trackpad users on the web demo have no middle button, so `Alt` + left-drag
    // is offered as an orbit alias (the Maya/industry convention). Right-drag pan
    // keeps priority so a held `Alt` never hijacks a pan gesture.
    let shift_held = keyboard_input.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let alt_held = keyboard_input.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    let pan = mouse_button_input.pressed(MouseButton::Right)
        || (shift_held && alt_held && mouse_button_input.pressed(MouseButton::Left));
    let orbit = mouse_button_input.pressed(MouseButton::Middle)
        || (alt_held && mouse_button_input.pressed(MouseButton::Left));

    if pan {
        for event in mouse_motion_events.read() {
            pan_camera_settings(
                &mut cam,
                event.delta,
                camera
                    .sub_camera_view
                    .map_or(window.height(), |view| view.full_size.min_element() as f32),
                *transform.right(),
                *transform.up(),
            );
        }
    } else if orbit {
        for event in mouse_motion_events.read() {
            orbit_camera_settings(&mut cam, event.delta);
        }
    } else {
        mouse_motion_events.clear();
    }

    for event in mouse_wheel_events.read() {
        zoom_camera_settings(&mut cam, normalized_wheel_delta(*event));
    }

    if !camera_settings_changed(&cam, &transform) {
        return;
    }
    apply_camera_setting(&mut transform, &cam);
}

fn camera_settings_changed(cam: &CameraSettings, transform: &Transform) -> bool {
    let expected = camera_transform_for_settings(cam);
    transform.translation != expected.translation || transform.rotation != expected.rotation
}

/// Writes the transform derived from `cam` into `transform`.
pub(crate) fn apply_camera_setting(transform: &mut Transform, cam: &CameraSettings) {
    *transform = camera_transform_for_settings(cam);
}

/// Converts orbit settings into a world transform looking at the focus point.
pub(crate) fn camera_transform_for_settings(cam: &CameraSettings) -> Transform {
    let x = cam.radius * cam.beta.sin() * cam.alpha.sin();
    let y = cam.radius * cam.beta.cos();
    let z = cam.radius * cam.beta.sin() * cam.alpha.cos();

    Transform::from_translation(cam.focus + Vec3::new(x, y, z)).looking_at(cam.focus, Vec3::Y)
}

#[cfg(test)]
mod tests {
    use super::{
        camera_settings_changed, camera_settings_for_graph, pan_step_for_radius,
        zoom_radius_for_scroll,
    };
    use crate::components::CameraSettings;
    use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind};
    use glam::ivec3;

    #[test]
    fn viewport_projection_centres_and_fits_geometry_in_the_unobstructed_canvas() {
        use bevy::camera::CameraProjection;
        use bevy::prelude::*;
        use bevy_egui::egui;
        for (size, canvas) in [
            (
                egui::vec2(1280.0, 720.0),
                egui::Rect::from_min_max(egui::pos2(42.0, 90.0), egui::pos2(1238.0, 680.0)),
            ),
            (
                egui::vec2(480.0, 720.0),
                egui::Rect::from_min_max(egui::pos2(42.0, 180.0), egui::pos2(438.0, 680.0)),
            ),
            (
                egui::vec2(570.0, 630.0),
                egui::Rect::from_min_max(egui::pos2(330.0, 170.0), egui::pos2(570.0, 590.0)),
            ),
            (
                egui::vec2(1280.0, 720.0),
                egui::Rect::from_min_max(egui::pos2(400.0, 90.0), egui::pos2(1280.0, 680.0)),
            ),
        ] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let (view, fov) = super::viewport_camera_view(screen, canvas);
            let projection = PerspectiveProjection {
                fov,
                ..Default::default()
            };
            let matrix = projection.get_clip_from_view_for_sub(&view.unwrap());
            let project = |point: Vec3| {
                let ndc = matrix.project_point3(point);
                egui::pos2((ndc.x + 1.0) * size.x / 2.0, (1.0 - ndc.y) * size.y / 2.0)
            };
            assert!(project(Vec3::new(0.0, 0.0, -10.0)).distance(canvas.center()) < 1.0);
            for point in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y] {
                assert!(canvas.contains(project(point * 2.0 + Vec3::new(0.0, 0.0, -10.0))));
            }
        }
    }

    fn graph_with_x_extent(max_x: i32) -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            ivec3(max_x, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
    }

    #[test]
    fn larger_graphs_fit_farther() {
        let small = graph_with_x_extent(1);
        let large = graph_with_x_extent(20);

        assert!(
            camera_settings_for_graph(&large, 2.0).radius
                > camera_settings_for_graph(&small, 2.0).radius
        );
        assert_eq!(
            camera_settings_for_graph(&BlockGraph::default(), 2.0).radius,
            CameraSettings::default().radius
        );
    }

    #[test]
    fn extreme_graph_coordinates_do_not_overflow_camera_framing() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            ivec3(i32::MIN, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            ivec3(i32::MAX, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));

        let settings = camera_settings_for_graph(&graph, 2.0);
        assert!(settings.focus.is_finite());
        assert_eq!(settings.radius, super::MAX_CAMERA_RADIUS);
    }

    #[test]
    fn zoom_step_tracks_current_camera_radius_not_whole_graph_size() {
        let huge = graph_with_x_extent(400);
        let fit_radius = camera_settings_for_graph(&huge, 2.0).radius;
        assert!(fit_radius > CameraSettings::default().radius * 10.0);

        let local_radius = CameraSettings::default().radius;
        let zoomed_local = zoom_radius_for_scroll(local_radius, 1.0);
        let local_step = local_radius - zoomed_local;
        let local_ratio = local_step / local_radius;

        assert!(
            (0.04..0.12).contains(&local_ratio),
            "local zoom should stay proportional, got {local_ratio}"
        );

        let zoomed_fit = zoom_radius_for_scroll(fit_radius, 1.0);
        let fit_ratio = (fit_radius - zoomed_fit) / fit_radius;
        assert!(
            (0.04..0.12).contains(&fit_ratio),
            "fit-distance zoom should be proportional, got {fit_ratio}"
        );
    }

    #[test]
    fn pan_step_tracks_current_camera_radius_not_whole_graph_size() {
        let local_radius = CameraSettings::default().radius;
        let viewport_height = 720.0;
        let local_pan_step = pan_step_for_radius(local_radius, viewport_height);
        assert!(
            (0.02..0.06).contains(&local_pan_step),
            "local pan should move a small fraction of a block per pixel, got {local_pan_step}"
        );

        let fit_pan_step = pan_step_for_radius(2_000.0, viewport_height);
        assert!(
            fit_pan_step > 1.0,
            "fit-distance pan should still cross large graphs efficiently, got {fit_pan_step}"
        );
    }

    #[test]
    fn unchanged_camera_settings_do_not_need_transform_write() {
        let settings = CameraSettings::default();
        let mut transform = bevy::prelude::Transform::default();
        super::apply_camera_setting(&mut transform, &settings);

        assert!(!camera_settings_changed(&settings, &transform));
    }

    #[test]
    fn viewport_pointer_blocked_when_ui_owns_pointer() {
        let mut ui_input = crate::resources::UiInputState::default();
        assert!(!ui_input.blocks_viewport_pointer_input());

        ui_input.pointer_over_ui = true;
        assert!(ui_input.blocks_viewport_pointer_input());

        ui_input.pointer_over_ui = false;
        ui_input.wants_pointer_input = true;
        assert!(ui_input.blocks_viewport_pointer_input());
    }
}
