//! Startup scene construction (camera, lights, axis bars, grid, preview meshes)
//! and the per-frame sync of the current-plane grid overlay.

use crate::components::{
    AxisHelper, CameraSettings, CurrentPlaneGrid, CurrentPlaneGridPart, EditorCamera, GraphRoot,
    PreviewEndpoint, PreviewEndpointMesh, PreviewMesh, WalkingPreviewMesh,
};
use crate::resources::{EditorState, EditorTabs};
use crate::systems::EditorUpdateSet;
use crate::systems::camera::{MAX_CAMERA_RADIUS, camera_transform_for_settings};
use crate::systems::input::{
    on_preview_endpoint_click, on_preview_endpoint_out, on_preview_endpoint_over,
};
use crate::theme::ThemePreset;
use bevy::asset::RenderAssetUsages;
use bevy::camera::CameraUpdateSystems;

/// Startup scene construction and the per-frame current-plane grid sync. Owns
/// the grid overlay cache; `setup` seeds the shared assets at startup.
pub(crate) struct SetupPlugin;

impl Plugin for SetupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(
                PostUpdate,
                activate_editor_camera_after_scale_factor
                    .after(CameraUpdateSystems)
                    .after(EguiPostUpdateSet::ProcessOutput),
            )
            .add_systems(
                Update,
                sync_current_plane_grid_system.in_set(EditorUpdateSet::Rendering),
            );
    }
}
use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::render_resource::{Face, PrimitiveTopology};
use bevy::render::view::Tonemapping;
use bevy::window::{PrimaryWindow, WindowBackendScaleFactorChanged};
use bevy_egui::{EguiContext, EguiOutput, EguiPostUpdateSet, PrimaryEguiContext, egui};
use bloq_graph::{EDITOR_AMBIENT_BRIGHTNESS, EDITOR_DIRECTIONAL_ILLUMINANCE};

/// Length of each world-axis bar drawn from the origin.
const AXIS_LENGTH: f32 = 1000.0;
/// Thickness of the world-axis bars.
const AXIS_WIDTH: f32 = 0.035;
const CAMERA_FAR_PLANE: f32 = MAX_CAMERA_RADIUS * 4.0;
const GRID_HALF_EXTENT: i32 = 22;
const GRID_PLANE_LIFT: f32 = 0.012;
const GRID_LINE_LIFT: f32 = 0.006;
const GRID_CONNECTOR_WIDTH: f32 = 0.010;
const GRID_MARKER_HALF_SIZE: f32 = 0.048;
const GRID_ANCHOR_HALF_SIZE: f32 = 0.58;
const GRID_ANCHOR_WIDTH: f32 = 0.035;
const GRID_ANCHOR_SEGMENT: f32 = 0.24;

// Ambient dominates the single directional light, so a face in shadow still
// reads as its basis color and the lit side only gains enough shading to give
// the solid depth. Both values are calibrated against Bevy's default
// `Exposure::BLENDER` (ev100 9.7, exposure = 2^-9.7/1.2 = 1.0019e-3) with
// `Tonemapping::Linear`: ambient contributes `0.4524 * brightness * exposure`
// (0.4524 is `EnvBRDFApprox` at `perceptual_roughness` 1.0, which is why the
// ambient number has to be more than twice what plain `brightness * exposure`
// would suggest) and the directional contributes
// `0.3255 * NdotL * illuminance * exposure` (`Fd_Burley`, which already carries
// the 1/PI). The key light sits on the (1,1,1) diagonal, so the three faces it
// reaches get NdotL = 1/sqrt(3): a lit face lands at 0.68 of its palette color
// and a shadowed one at 0.48. Retune both together or that 1.41 contrast ratio
// drifts.

/// Direction the key light points from; equidistant on all three axes so no
/// single face pair is favored.
pub(crate) const EDITOR_LIGHT_POSITION: Vec3 = Vec3::splat(10.0);

pub(crate) fn main_editor_directional_light() -> DirectionalLight {
    DirectionalLight {
        illuminance: EDITOR_DIRECTIONAL_ILLUMINANCE,
        shadow_maps_enabled: false,
        ..default()
    }
}

#[cfg(target_os = "linux")]
fn wait_for_initial_wayland_scale_factor() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
        && !matches!(std::env::var("WINIT_UNIX_BACKEND").as_deref(), Ok("x11"))
}

#[cfg(not(target_os = "linux"))]
const fn wait_for_initial_wayland_scale_factor() -> bool {
    false
}

fn activate_editor_camera_after_scale_factor(
    mut scale_factor_events: MessageReader<WindowBackendScaleFactorChanged>,
    mut scale_factor_seen: Local<bool>,
    camera: Single<(&mut Camera, &mut EguiContext, &EguiOutput), With<EditorCamera>>,
    window: Single<&Window, With<PrimaryWindow>>,
) {
    *scale_factor_seen |= scale_factor_events.read().next().is_some();

    let (mut camera, mut egui_context, egui_output) = camera.into_inner();
    if camera.is_active || !*scale_factor_seen {
        return;
    }

    let Some(viewport_size) = camera.physical_viewport_size() else {
        return;
    };
    if viewport_size == window.physical_size()
        && egui_frame_covers_viewport(
            viewport_size,
            egui_context.get_mut().viewport_rect().size(),
            egui_output.pixels_per_point,
        )
    {
        camera.is_active = true;
    }
}

fn egui_frame_covers_viewport(
    viewport_size: UVec2,
    screen_size: egui::Vec2,
    pixels_per_point: f32,
) -> bool {
    let frame_size = Vec2::new(screen_size.x, screen_size.y) * pixels_per_point;
    (frame_size - viewport_size.as_vec2()).abs().max_element() <= 1.0
}

/// Startup system: builds the lights, camera, axis bars, current-plane grid,
/// graph root, and the reusable preview meshes, and seeds the overlay materials
/// on [`EditorState`].
pub(crate) fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut editor_state: ResMut<EditorState>,
    mut clear_color: ResMut<ClearColor>,
    tabs: Res<EditorTabs>,
) {
    clear_color.0 = editor_state.bg_color;
    let palette = crate::theme::palette(editor_state.theme_preset);
    let pick_material = materials.add(StandardMaterial {
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    editor_state.highlight_material = pick_material.clone();
    let selection_material = materials.add(StandardMaterial {
        alpha_mode: AlphaMode::Opaque,
        unlit: true,
        ..default()
    });
    let preview_endpoint_material = materials.add(StandardMaterial {
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    let preview_endpoint_hover_material = materials.add(StandardMaterial {
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    // Pulsed by `pulse_action_candidate_material_system` while an action pick is
    // armed; the alpha is animated, so the material must blend.
    let action_candidate_material = materials.add(StandardMaterial {
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    editor_state.action_candidate_material = action_candidate_material;
    editor_state.selection_material = selection_material;
    editor_state.preview_endpoint_material = preview_endpoint_material.clone();
    editor_state.preview_endpoint_hover_material = preview_endpoint_hover_material;
    apply_editor_material_theme(&editor_state, &mut materials);

    // Lights
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: EDITOR_AMBIENT_BRIGHTNESS,
        ..default()
    });
    commands.spawn((
        Name::new("Editor directional light"),
        main_editor_directional_light(),
        Transform::from_translation(EDITOR_LIGHT_POSITION).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Install named fonts before egui begins its first pass.
    let mut egui_context = EguiContext::default();
    crate::theme::setup_theme(egui_context.get_mut(), editor_state.theme_preset);

    // Camera
    let camera = tabs.active_tab().snapshot.camera_settings;
    commands.spawn((
        Name::new("Editor camera"),
        Camera3d::default(),
        Camera {
            is_active: !wait_for_initial_wayland_scale_factor(),
            ..default()
        },
        // The light constants above are fitted for a linear pipeline. Any curve
        // here changes them: `AcesFitted` (the Stephen Hill fit) needs roughly
        // 1.5x the light to land in the same place, and the LUT-based curves
        // would drag the `tonemapping_luts` feature and its KTX2 blobs into the
        // WASM binary. Retune the constants if this ever stops being `Linear`.
        Tonemapping::Linear,
        Projection::Perspective(PerspectiveProjection {
            far: CAMERA_FAR_PLANE,
            ..default()
        }),
        camera_transform_for_settings(&camera),
        camera,
        egui_context,
        PrimaryEguiContext,
        EditorCamera,
    ));

    // Axis Helper
    for (name, size, translation, axis) in [
        (
            "X axis",
            Vec3::new(AXIS_LENGTH, AXIS_WIDTH, AXIS_WIDTH),
            Vec3::new(AXIS_LENGTH * 0.5, 0.0, 0.0),
            AxisHelper::X,
        ),
        (
            "Y axis",
            Vec3::new(AXIS_WIDTH, AXIS_WIDTH, AXIS_LENGTH),
            Vec3::new(0.0, 0.0, -AXIS_LENGTH * 0.5),
            AxisHelper::Y,
        ),
        (
            "Z axis",
            Vec3::new(AXIS_WIDTH, AXIS_LENGTH, AXIS_WIDTH),
            Vec3::new(0.0, AXIS_LENGTH * 0.5, 0.0),
            AxisHelper::Z,
        ),
    ] {
        commands.spawn((
            Name::new(name),
            Mesh3d(meshes.add(Mesh::from(Cuboid::new(size.x, size.y, size.z)))),
            MeshMaterial3d(axis_material(&mut materials, axis_color(palette, axis))),
            Transform::from_translation(translation),
            axis,
            if editor_state.show_axis {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
            NotShadowCaster,
        ));
    }

    setup_current_plane_grid(&mut commands, &mut meshes, &mut materials, palette);

    commands.spawn((
        Name::new("Graph root"),
        Transform::IDENTITY,
        Visibility::default(),
        GraphRoot,
    ));

    commands.spawn((
        Name::new("Block placement preview"),
        Mesh3d(meshes.add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)))),
        MeshMaterial3d(pick_material),
        Transform::default(),
        Visibility::Hidden,
        PreviewMesh,
        NotShadowCaster,
    ));
    commands.spawn((
        Name::new("Walking placement preview"),
        Transform::default(),
        Visibility::Hidden,
        WalkingPreviewMesh,
    ));

    let endpoint_mesh = meshes.add(Mesh::from(Cuboid::new(0.34, 0.34, 0.34)));
    let preview_endpoints = [PreviewEndpoint::Start, PreviewEndpoint::End]
        .into_iter()
        .chain((0..6).map(PreviewEndpoint::PipeHint))
        .chain((0..8).map(PreviewEndpoint::WalkingHint));
    for endpoint in preview_endpoints {
        spawn_preview_endpoint_mesh(
            &mut commands,
            endpoint_mesh.clone(),
            preview_endpoint_material.clone(),
            endpoint,
        );
    }
}

pub(crate) fn apply_editor_material_theme(
    editor_state: &EditorState,
    materials: &mut Assets<StandardMaterial>,
) {
    let palette = crate::theme::palette(editor_state.theme_preset);
    for (handle, color, alpha) in [
        (
            &editor_state.highlight_material,
            hover_highlight_color32(palette),
            220,
        ),
        (&editor_state.selection_material, palette.accent_warn, 255),
        (
            &editor_state.preview_endpoint_material,
            palette.accent_primary,
            238,
        ),
        (
            &editor_state.preview_endpoint_hover_material,
            palette.accent_warn,
            255,
        ),
        (
            &editor_state.action_candidate_material,
            palette.accent_primary,
            220,
        ),
    ] {
        if let Some(mut material) = materials.get_mut(handle) {
            material.base_color = color32_to_bevy_color(color, alpha);
        }
    }
    if let Some(mut material) = materials.get_mut(&editor_state.preview_endpoint_hover_material) {
        material.emissive = color32_to_bevy_color(palette.accent_warn, 180).into();
    }
}

fn spawn_preview_endpoint_mesh(
    commands: &mut Commands,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    endpoint: PreviewEndpoint,
) {
    commands
        .spawn((
            Name::new(format!("Preview endpoint {endpoint:?}")),
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Hidden,
            Pickable::IGNORE,
            NotShadowCaster,
            PreviewEndpointMesh {
                endpoint,
                source_pos: None,
                target_pos: None,
            },
        ))
        .observe(on_preview_endpoint_over)
        .observe(on_preview_endpoint_out)
        .observe(on_preview_endpoint_click);
}

/// Repositions and recolours the current-plane grid overlay to track the
/// editing plane and camera focus, skipping the work when nothing it depends on
/// changed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct CurrentPlaneGridSyncState {
    visible: bool,
    plane_height: i32,
    pipe_length_bits: u32,
    center_x_bits: u32,
    center_z_bits: u32,
    theme_preset: ThemePreset,
}

pub(crate) fn sync_current_plane_grid_system(
    editor_state: Res<EditorState>,
    camera_query: Query<&CameraSettings, With<EditorCamera>>,
    mut applied: Local<Option<CurrentPlaneGridSyncState>>,
    mut grid_query: Query<(
        &CurrentPlaneGrid,
        &mut Transform,
        &mut Visibility,
        &MeshMaterial3d<StandardMaterial>,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let visibility = current_plane_grid_visibility(&editor_state);
    let stride = editor_state.pipe_length + 1.0;
    let y = editor_state.plane_height as f32 * stride + GRID_PLANE_LIFT;
    let focus = camera_query
        .iter()
        .next()
        .map_or(Vec3::ZERO, |camera| camera.focus);
    let center_x = (focus.x / stride).round() * stride;
    let center_z = (focus.z / stride).round() * stride;
    let palette = crate::theme::palette(editor_state.theme_preset);
    let sync_state = CurrentPlaneGridSyncState {
        visible: matches!(visibility, Visibility::Inherited),
        plane_height: editor_state.plane_height,
        pipe_length_bits: editor_state.pipe_length.to_bits(),
        center_x_bits: center_x.to_bits(),
        center_z_bits: center_z.to_bits(),
        theme_preset: editor_state.theme_preset,
    };
    if *applied == Some(sync_state) {
        return;
    }

    for (grid, mut transform, mut entity_visibility, material) in &mut grid_query {
        *entity_visibility = visibility;
        transform.translation =
            current_plane_grid_translation(grid.part, Vec3::new(center_x, 0.0, center_z), y);
        transform.scale = Vec3::new(stride, 1.0, stride);

        if let Some(mut material) = materials.get_mut(&material.0) {
            material.base_color = current_plane_grid_color(palette, grid.part);
            material.alpha_mode = AlphaMode::Blend;
            material.unlit = true;
            material.cull_mode = None;
        }
    }
    *applied = Some(sync_state);
}

fn current_plane_grid_translation(
    part: CurrentPlaneGridPart,
    camera_focus_center: Vec3,
    y: f32,
) -> Vec3 {
    match part {
        CurrentPlaneGridPart::HeightAnchor => Vec3::new(0.0, y, 0.0),
        CurrentPlaneGridPart::Connectors | CurrentPlaneGridPart::BlockMarkers => {
            Vec3::new(camera_focus_center.x, y, camera_focus_center.z)
        }
    }
}

fn setup_current_plane_grid(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    palette: &crate::theme::ThemePalette,
) {
    for part in [
        CurrentPlaneGridPart::HeightAnchor,
        CurrentPlaneGridPart::Connectors,
        CurrentPlaneGridPart::BlockMarkers,
    ] {
        let mesh = match part {
            CurrentPlaneGridPart::HeightAnchor => current_plane_anchor_mesh(),
            CurrentPlaneGridPart::Connectors => current_plane_connector_mesh(GRID_HALF_EXTENT),
            CurrentPlaneGridPart::BlockMarkers => current_plane_marker_mesh(GRID_HALF_EXTENT),
        };
        commands.spawn((
            Name::new(format!("Current plane grid {part:?}")),
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(current_plane_grid_material(materials, palette, part)),
            Transform::default(),
            Visibility::Inherited,
            CurrentPlaneGrid { part },
            NotShadowCaster,
        ));
    }
}

fn current_plane_anchor_mesh() -> Mesh {
    let mut positions = Vec::new();
    push_grid_corner_brackets(
        &mut positions,
        GRID_ANCHOR_HALF_SIZE,
        GRID_ANCHOR_WIDTH,
        GRID_ANCHOR_SEGMENT,
    );

    current_plane_triangle_mesh(positions)
}

fn push_grid_corner_brackets(positions: &mut Vec<[f32; 3]>, outer: f32, width: f32, segment: f32) {
    let inner = outer - width;

    for x_sign in [-1.0_f32, 1.0] {
        for z_sign in [-1.0_f32, 1.0] {
            let x_outer = x_sign * outer;
            let x_inner = x_sign * inner;
            let x_end = x_sign * (outer - segment);
            let z_outer = z_sign * outer;
            let z_inner = z_sign * inner;
            let z_end = z_sign * (outer - segment);

            push_grid_strip(
                positions,
                x_outer.min(x_end),
                z_outer.min(z_inner),
                x_outer.max(x_end),
                z_outer.max(z_inner),
            );
            push_grid_strip(
                positions,
                x_outer.min(x_inner),
                z_outer.min(z_end),
                x_outer.max(x_inner),
                z_outer.max(z_end),
            );
        }
    }
}

fn current_plane_connector_mesh(half_extent: i32) -> Mesh {
    let mut positions = Vec::new();
    for x in -half_extent..=half_extent {
        for z in -half_extent..=half_extent {
            if x < half_extent {
                push_grid_strip(
                    &mut positions,
                    x as f32 + GRID_MARKER_HALF_SIZE,
                    z as f32 - GRID_CONNECTOR_WIDTH * 0.5,
                    x as f32 + 1.0 - GRID_MARKER_HALF_SIZE,
                    z as f32 + GRID_CONNECTOR_WIDTH * 0.5,
                );
            }
            if z < half_extent {
                push_grid_strip(
                    &mut positions,
                    x as f32 - GRID_CONNECTOR_WIDTH * 0.5,
                    z as f32 + GRID_MARKER_HALF_SIZE,
                    x as f32 + GRID_CONNECTOR_WIDTH * 0.5,
                    z as f32 + 1.0 - GRID_MARKER_HALF_SIZE,
                );
            }
        }
    }

    current_plane_triangle_mesh(positions)
}

fn current_plane_marker_mesh(half_extent: i32) -> Mesh {
    let mut positions = Vec::new();
    for x in -half_extent..=half_extent {
        for z in -half_extent..=half_extent {
            let x = x as f32;
            let z = z as f32;
            push_grid_strip(
                &mut positions,
                x - GRID_MARKER_HALF_SIZE,
                z - GRID_MARKER_HALF_SIZE,
                x + GRID_MARKER_HALF_SIZE,
                z + GRID_MARKER_HALF_SIZE,
            );
        }
    }

    current_plane_triangle_mesh(positions)
}

fn current_plane_triangle_mesh(positions: Vec<[f32; 3]>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.compute_flat_normals();
    mesh
}

fn push_grid_strip(positions: &mut Vec<[f32; 3]>, x0: f32, z0: f32, x1: f32, z1: f32) {
    positions.extend([
        [x0, GRID_LINE_LIFT, z0],
        [x1, GRID_LINE_LIFT, z0],
        [x1, GRID_LINE_LIFT, z1],
        [x0, GRID_LINE_LIFT, z0],
        [x1, GRID_LINE_LIFT, z1],
        [x0, GRID_LINE_LIFT, z1],
    ]);
}

fn current_plane_grid_material(
    materials: &mut Assets<StandardMaterial>,
    palette: &crate::theme::ThemePalette,
    part: CurrentPlaneGridPart,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: current_plane_grid_color(palette, part),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        cull_mode: if matches!(part, CurrentPlaneGridPart::HeightAnchor) {
            None
        } else {
            Some(Face::Back)
        },
        ..default()
    })
}

fn current_plane_grid_color(
    palette: &crate::theme::ThemePalette,
    part: CurrentPlaneGridPart,
) -> Color {
    let (color, alpha) = match (palette.dark_mode, part) {
        (false, CurrentPlaneGridPart::HeightAnchor) => (palette.accent_primary, 172),
        (false, CurrentPlaneGridPart::Connectors) => (palette.border_bright, 28),
        (false, CurrentPlaneGridPart::BlockMarkers) => (palette.accent_secondary, 62),
        (true, CurrentPlaneGridPart::HeightAnchor) => (palette.accent_primary, 190),
        (true, CurrentPlaneGridPart::Connectors) => (palette.grey2, 40),
        (true, CurrentPlaneGridPart::BlockMarkers) => (palette.accent_secondary, 82),
    };
    color32_to_bevy_color(color, alpha)
}

fn current_plane_grid_visibility(editor_state: &EditorState) -> Visibility {
    if editor_state.show_grid && !editor_state.request_screenshot && !editor_state.taking_screenshot
    {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    }
}

fn color32_to_bevy_color(color: bevy_egui::egui::Color32, alpha: u8) -> Color {
    Color::srgba_u8(color.r(), color.g(), color.b(), alpha)
}

fn hover_highlight_color32(palette: &crate::theme::ThemePalette) -> bevy_egui::egui::Color32 {
    if palette.dark_mode {
        bevy_egui::egui::Color32::from_rgb(240, 171, 252)
    } else {
        bevy_egui::egui::Color32::from_rgb(162, 28, 175)
    }
}

pub(crate) fn axis_color(palette: &crate::theme::ThemePalette, axis: AxisHelper) -> Color {
    let (r, g, b, alpha) = match (palette.dark_mode, axis) {
        (false, AxisHelper::X) => (178, 76, 70, 178),
        (false, AxisHelper::Y) => (44, 130, 82, 168),
        (false, AxisHelper::Z) => (66, 95, 190, 178),
        (true, AxisHelper::X) => (234, 105, 98, 196),
        (true, AxisHelper::Y) => (169, 182, 101, 184),
        (true, AxisHelper::Z) => (125, 174, 163, 196),
    };
    Color::srgba_u8(r, g, b, alpha)
}

fn axis_material(
    materials: &mut Assets<StandardMaterial>,
    base_color: Color,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color,
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    })
}

#[cfg(test)]
mod tests {
    use super::{
        EDITOR_AMBIENT_BRIGHTNESS, EDITOR_DIRECTIONAL_ILLUMINANCE, current_plane_grid_translation,
        current_plane_grid_visibility, egui_frame_covers_viewport, main_editor_directional_light,
    };
    use crate::components::CurrentPlaneGridPart;
    use crate::resources::EditorState;
    use bevy::prelude::{UVec2, Vec3, Visibility};
    use bevy_egui::egui;

    #[test]
    fn editor_camera_requires_synchronized_egui_frame() {
        let viewport = UVec2::new(1900, 2064);

        assert!(!egui_frame_covers_viewport(
            viewport,
            egui::vec2(956.0, 1036.0),
            1.0,
        ));
        assert!(!egui_frame_covers_viewport(
            viewport,
            egui::vec2(1900.0, 2064.0),
            2.0,
        ));
        assert!(egui_frame_covers_viewport(
            viewport,
            egui::vec2(950.0, 1032.0),
            2.0,
        ));
    }

    #[test]
    fn current_plane_grid_hides_for_screenshots() {
        let mut editor_state = EditorState::default();

        assert_eq!(
            current_plane_grid_visibility(&editor_state),
            Visibility::Inherited
        );

        editor_state.request_screenshot = true;
        assert_eq!(
            current_plane_grid_visibility(&editor_state),
            Visibility::Hidden
        );
    }

    #[test]
    fn startup_uses_restored_camera_and_axis_visibility() {
        use super::*;

        let mut app = App::new();
        let mut tabs = EditorTabs::default();
        tabs.active_tab_mut().snapshot.camera_settings.radius = 12.0;
        app.insert_resource(tabs)
            .insert_resource(EditorState {
                show_axis: false,
                ..default()
            })
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<ClearColor>()
            .add_systems(Startup, setup);
        app.update();
        let world = app.world_mut();
        let (camera, transform) = world
            .query_filtered::<(&CameraSettings, &Transform), With<EditorCamera>>()
            .single(world)
            .unwrap();
        assert_eq!(camera.radius, 12.0);
        assert_eq!(*transform, camera_transform_for_settings(camera));
        assert!(
            world
                .query_filtered::<&Visibility, With<AxisHelper>>()
                .iter(world)
                .all(|v| *v == Visibility::Hidden)
        );
    }

    #[test]
    fn theme_change_recolors_viewport_highlight() {
        let mut materials = bevy::prelude::Assets::default();
        let mut editor_state = EditorState {
            highlight_material: materials.add(bevy::pbr::StandardMaterial::default()),
            ..Default::default()
        };
        super::apply_editor_material_theme(&editor_state, &mut materials);
        let light = materials
            .get(&editor_state.highlight_material)
            .expect("highlight material")
            .base_color;

        editor_state.theme_preset = crate::theme::ThemePreset::GruvboxMaterial;
        super::apply_editor_material_theme(&editor_state, &mut materials);

        assert_ne!(
            materials
                .get(&editor_state.highlight_material)
                .expect("highlight material")
                .base_color,
            light
        );
    }

    #[test]
    fn current_plane_anchor_is_fixed_while_markers_follow_camera_focus() {
        let y = 12.0;
        let origin_focus = Vec3::new(0.0, 0.0, 0.0);
        let panned_focus = Vec3::new(41.0, 0.0, -17.0);

        assert_ne!(
            current_plane_grid_translation(CurrentPlaneGridPart::BlockMarkers, origin_focus, y),
            current_plane_grid_translation(CurrentPlaneGridPart::BlockMarkers, panned_focus, y)
        );
        assert_eq!(
            current_plane_grid_translation(CurrentPlaneGridPart::HeightAnchor, origin_focus, y),
            current_plane_grid_translation(CurrentPlaneGridPart::HeightAnchor, panned_focus, y)
        );
    }

    #[test]
    fn main_editor_light_does_not_allocate_shadow_maps() {
        assert!(!main_editor_directional_light().shadow_maps_enabled);
    }

    /// Guards the calibration documented next to the two light constants: the
    /// shading factors a face multiplies its palette color by, and the contrast
    /// between them. Changing one constant alone breaks the ratio.
    #[test]
    fn key_light_keeps_the_calibrated_shading_factors() {
        const EXPOSURE: f32 = 1.0019079e-3;
        const AMBIENT_BRDF: f32 = 0.4524;
        const FD_BURLEY: f32 = 0.32550;
        const N_DOT_L: f32 = 0.577_350_3;

        let shadow = AMBIENT_BRDF * EDITOR_AMBIENT_BRIGHTNESS * EXPOSURE;
        let lit = shadow + FD_BURLEY * N_DOT_L * EDITOR_DIRECTIONAL_ILLUMINANCE * EXPOSURE;

        assert!((shadow - 0.483).abs() < 0.005, "shadow factor {shadow}");
        assert!((lit - 0.681).abs() < 0.005, "lit factor {lit}");
        assert!(
            (lit / shadow - 1.412).abs() < 0.01,
            "contrast {}",
            lit / shadow
        );
    }
}
