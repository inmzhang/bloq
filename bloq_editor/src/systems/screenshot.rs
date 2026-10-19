//! Captures the primary window to a numbered PNG on disk when a screenshot is
//! requested.

#[cfg(not(target_arch = "wasm32"))]
use crate::components::{AxisHelper, CameraSettings, EditorCamera, GraphElement};
#[cfg(not(target_arch = "wasm32"))]
use crate::resources::GraphState;
use crate::resources::{EditorState, Notifications};
use crate::systems::EditorUpdateSet;
use bevy::prelude::*;
#[cfg(target_arch = "wasm32")]
use bevy::render::view::screenshot::save_to_disk;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{CursorIcon, PrimaryWindow, SystemCursorIcon};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub(crate) struct AutomatedScreenshot {
    output: PathBuf,
    frames_before_request: u8,
    azimuth: Option<f32>,
    elevation: Option<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AutomatedScreenshot {
    pub(crate) fn new(output: PathBuf, azimuth: Option<f32>, elevation: Option<f32>) -> Self {
        Self {
            output,
            frames_before_request: 8,
            azimuth,
            elevation,
        }
    }
}

/// Fixed graph-space cutaway for one automated capture. Surface overlays are
/// rendered separately and deliberately never enter this geometry transform.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub(crate) struct AutomatedCutaway(pub(crate) Vec<bloq_graph::Direction>);

#[cfg(not(target_arch = "wasm32"))]
fn cutaway_geometry(
    graph: &bloq_graph::BlockGraph,
    length: f32,
    faces: &[bloq_graph::Direction],
) -> bloq_graph::GltfData {
    use bloq_graph::{
        GltfData, block_as_gltf_data_with_pipe_length, pipe_between_positions_as_gltf_data,
    };
    let mut data = GltfData::default();
    let stride = length + 1.0;
    for block in graph.blocks().filter(|block| !block.kind().is_port()) {
        let center = block.pos().as_vec3() * stride;
        data.extend(
            block_as_gltf_data_with_pipe_length(block, graph, length)
                .pop_faces_at_directions(faces)
                .map_points(|point| point + center),
        );
    }
    for (u, v, _, _, pipe) in graph.pipe_endpoints_with_blocks() {
        let center = (u.as_vec3() + v.as_vec3()) * (stride * 0.5);
        data.extend(
            pipe_between_positions_as_gltf_data(u, v, pipe, graph, length)
                .pop_faces_at_directions(faces)
                .map_points(|point| point + center),
        );
    }
    data
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_automated_cutaway(
    mut commands: Commands,
    cutaway: Res<AutomatedCutaway>,
    graph_state: Res<GraphState>,
    editor_state: Res<EditorState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cache: ResMut<crate::systems::visuals::RenderAssetCache>,
    mut elements: Query<(&GraphElement, &mut Visibility)>,
    mut spawned: Local<bool>,
) {
    for (element, mut visibility) in &mut elements {
        if let GraphElement::Block(pos) = element
            && graph_state
                .graph
                .get_block(*pos)
                .is_some_and(|block| block.kind().is_port())
        {
            continue;
        }
        *visibility = Visibility::Hidden;
    }
    if *spawned {
        return;
    }
    let data = cutaway_geometry(&graph_state.graph, editor_state.pipe_length, &cutaway.0)
        .map_points(|point| crate::utils::graph_to_world(point, 0.0));
    let parent = commands
        .spawn((Transform::default(), Visibility::Inherited))
        .id();
    crate::systems::visuals::spawn_gltf_data_parts(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut cache,
        parent,
        data,
        None,
        1.0,
        None,
        None,
        None,
        Pickable::IGNORE,
    );
    *spawned = true;
}

/// Captures the primary window on request. Runs after the grid sync so the
/// screenshot reflects the current-plane overlay state.
pub(crate) struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            screenshot_system
                .after(crate::systems::setup::sync_current_plane_grid_system)
                .run_if(|state: Res<EditorState>| {
                    state.request_screenshot && !state.taking_screenshot
                })
                .in_set(EditorUpdateSet::Rendering),
        );
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(
            Startup,
            prepare_automated_screenshot
                .after(crate::systems::setup::setup)
                .run_if(resource_exists::<AutomatedScreenshot>),
        )
        .add_systems(
            Update,
            apply_automated_cutaway
                .after(crate::systems::visuals::block_graph_visual_system)
                .before(screenshot_system)
                .run_if(resource_exists::<AutomatedCutaway>)
                .in_set(EditorUpdateSet::Rendering),
        )
        .add_systems(
            Update,
            request_automated_screenshot
                .after(screenshot_system)
                .run_if(resource_exists::<AutomatedScreenshot>)
                .in_set(EditorUpdateSet::Rendering),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn prepare_automated_screenshot(
    capture: Res<AutomatedScreenshot>,
    graph_state: Res<GraphState>,
    mut editor_state: ResMut<EditorState>,
    mut clear_color: ResMut<ClearColor>,
    camera: Single<(&mut Transform, &mut CameraSettings), With<EditorCamera>>,
    mut axes: Query<&mut Visibility, With<AxisHelper>>,
) {
    editor_state.show_axis = false;
    editor_state.show_grid = false;
    editor_state.bg_color = Color::WHITE;
    clear_color.0 = Color::WHITE;
    for mut visibility in &mut axes {
        *visibility = Visibility::Hidden;
    }
    let (mut camera_transform, mut camera_settings) = camera.into_inner();
    crate::systems::camera::reset_camera_to_graph(
        &mut camera_transform,
        &mut camera_settings,
        &graph_state.graph,
        editor_state.pipe_length,
    );
    if let Some(azimuth) = capture.azimuth {
        camera_settings.alpha = azimuth;
    }
    if let Some(elevation) = capture.elevation {
        camera_settings.beta = std::f32::consts::FRAC_PI_2 - elevation;
    }
    crate::systems::camera::apply_camera_setting(&mut camera_transform, &camera_settings);
}

#[cfg(not(target_arch = "wasm32"))]
fn request_automated_screenshot(
    mut capture: ResMut<AutomatedScreenshot>,
    mut editor_state: ResMut<EditorState>,
) {
    if capture.frames_before_request > 0 {
        capture.frames_before_request -= 1;
    } else if !editor_state.request_screenshot && !editor_state.taking_screenshot {
        editor_state.request_screenshot = true;
    }
}

/// Requests a primary-window screenshot, saving it to `screenshot-N.png` and
/// clearing the request flags once the capture completes.
fn screenshot_system(
    mut commands: Commands,
    mut next_index: Local<Option<u32>>,
    mut editor_state: ResMut<'_, EditorState>,
    window: Single<Entity, With<PrimaryWindow>>,
    #[cfg(not(target_arch = "wasm32"))] automated: Option<Res<AutomatedScreenshot>>,
) {
    editor_state.taking_screenshot = true;
    #[cfg(not(target_arch = "wasm32"))]
    let (path, exit_after_capture) = if let Some(automated) = automated {
        (automated.output.display().to_string(), true)
    } else {
        // Resume numbering past any screenshots a prior session left on disk so
        // a manual capture never overwrites one from an earlier session.
        let index = next_index.get_or_insert_with(next_screenshot_index);
        let path = format!("screenshot-{}.png", *index);
        *index += 1;
        (path, false)
    };
    #[cfg(target_arch = "wasm32")]
    let path = {
        let index = next_index.get_or_insert_with(next_screenshot_index);
        let path = format!("screenshot-{}.png", *index);
        *index += 1;
        path
    };
    commands
        .entity(*window)
        .insert(CursorIcon::from(SystemCursorIcon::Progress));
    let mut screenshot = commands.spawn(Screenshot::primary_window());

    #[cfg(not(target_arch = "wasm32"))]
    screenshot.observe(
        move |captured: On<ScreenshotCaptured>,
              mut commands: Commands,
              mut editor_state: ResMut<'_, EditorState>,
              mut notifications: ResMut<'_, Notifications>,
              mut app_exit: MessageWriter<AppExit>,
              window: Single<Entity, With<PrimaryWindow>>| {
            use color_eyre::eyre::WrapErr as _;

            let result = captured
                .image
                .clone()
                .try_into_dynamic()
                .wrap_err("convert captured screenshot to an image")
                .and_then(|image| {
                    if let Some(parent) = std::path::Path::new(&path).parent()
                        && !parent.as_os_str().is_empty()
                    {
                        std::fs::create_dir_all(parent)
                            .wrap_err_with(|| format!("create {}", parent.display()))?;
                    }
                    image
                        .to_rgb8()
                        .save(&path)
                        .wrap_err_with(|| format!("write screenshot {path}"))
                });
            let exit = if result.is_ok() {
                AppExit::Success
            } else {
                AppExit::error()
            };
            match result {
                Ok(()) => notifications.push_info(format!("Saved screenshot to {path}")),
                Err(error) => notifications.push_error_report("Failed to save screenshot", &error),
            }
            editor_state.request_screenshot = false;
            editor_state.taking_screenshot = false;
            commands.entity(*window).remove::<CursorIcon>();
            if exit_after_capture {
                app_exit.write(exit);
            }
        },
    );

    #[cfg(target_arch = "wasm32")]
    screenshot.observe(save_to_disk(path.clone())).observe(
        move |_: On<ScreenshotCaptured>,
              mut commands: Commands,
              mut editor_state: ResMut<'_, EditorState>,
              mut notifications: ResMut<'_, Notifications>,
              window: Single<Entity, With<PrimaryWindow>>| {
            notifications.push_info(format!("Captured screenshot for download as {path}"));
            editor_state.request_screenshot = false;
            editor_state.taking_screenshot = false;
            commands.entity(*window).remove::<CursorIcon>();
        },
    );
}

/// The first `screenshot-N.png` index not already taken in the working
/// directory, so a fresh session continues the sequence instead of clobbering
/// earlier captures. Always `0` on the web target, which has no directory to
/// scan and hands saving off to the browser.
fn next_screenshot_index() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read_dir(".")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name();
                name.to_str()
                    .and_then(|name| name.strip_prefix("screenshot-"))
                    .and_then(|rest| rest.strip_suffix(".png"))
                    .and_then(|digits| digits.parse::<u32>().ok())
            })
            .max()
            .map_or(0, |highest| highest + 1)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn cutaway_pops_graph_faces_without_changing_surface_geometry() {
        let graph = bloq_graph::GalleryItem::CNOT.build();
        let closed = cutaway_geometry(&graph, 2.0, &[]);
        let open = cutaway_geometry(&graph, 2.0, &[bloq_graph::Direction::XPLUS]);
        assert!(
            open.triangles.values().map(Vec::len).sum::<usize>()
                < closed.triangles.values().map(Vec::len).sum::<usize>()
        );
        assert_eq!(open.lines, closed.lines);
        assert!(
            open.triangles.values().flatten().all(|[a, b, c]| {
                (*b - *a).cross(*c - *a).normalize_or_zero().dot(Vec3::X) < 0.99
            })
        );
        let surface = graph.stabilizers().unwrap().remove(0);
        assert!(
            !bloq_graph::stabilizer_as_gltf_data(&surface, &graph, 2.0)
                .unwrap()
                .triangles
                .is_empty()
        );
    }

    #[test]
    fn automated_capture_exit_reports_save_result() {
        let directory = std::env::temp_dir().join(format!(
            "bloq-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        for (path, success) in [
            (directory.join("capture.png"), true),
            (directory.clone(), false),
        ] {
            let mut app = App::new();
            app.init_resource::<EditorState>()
                .init_resource::<Notifications>()
                .add_message::<AppExit>()
                .insert_resource(AutomatedScreenshot::new(path.clone(), None, None))
                .add_systems(Update, screenshot_system);
            app.world_mut().spawn(PrimaryWindow);
            app.update();
            let entity = app
                .world_mut()
                .query_filtered::<Entity, With<Screenshot>>()
                .single(app.world())
                .unwrap();
            app.world_mut().trigger(ScreenshotCaptured {
                entity,
                image: Image::default(),
            });
            app.world_mut().flush();
            let exits = app.world().resource::<Messages<AppExit>>();
            let exit = exits.iter_current_update_messages().next().unwrap();
            assert_eq!(matches!(exit, AppExit::Success), success);
            assert_eq!(path.is_file(), success);
            assert!(!app.world().resource::<EditorState>().taking_screenshot);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
