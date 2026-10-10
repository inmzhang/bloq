//! Preload block, pipe, and gallery thumbnails after the editor's first frames.
//! Render one at a time, giving visible cards priority over background work.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::components::CameraSettings;
use crate::resources::RenderAssetCache;
use crate::systems::EditorUpdateSet;
use crate::systems::camera::{CAMERA_DEFAULT_FOV_Y, apply_camera_setting};
use crate::systems::visuals::spawn_gltf_data_parts;
use crate::utils::{graph_bounds, graph_to_world};
use bevy::camera::{RenderTarget, visibility::RenderLayers};
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::{
    RenderApp, RenderSystems,
    render_resource::{CachedPipelineState, PipelineCache, TextureFormat},
    view::Tonemapping,
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};
use bloq_graph::{
    Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryItem, GltfData, Pipe,
    block_as_gltf_data, block_graph_as_gltf_data, pipe_as_gltf_data,
};

/// Off-screen thumbnail preloading, with visible UI cards taking priority.
pub(crate) struct ThumbnailsPlugin;

impl Plugin for ThumbnailsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ThumbnailTextures>()
            .init_resource::<ThumbnailPipelinesReady>();
        #[cfg(not(target_arch = "wasm32"))]
        if app.world().contains_resource::<GalleryThumbnailExport>() {
            app.world_mut().resource_mut::<ThumbnailTextures>().preload =
                GalleryItem::iter().map(ThumbnailKey::Gallery).collect();
        }
        let pipelines_ready = app.world().resource::<ThumbnailPipelinesReady>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(pipelines_ready).add_systems(
                bevy::render::Render,
                update_thumbnail_pipeline_status.in_set(RenderSystems::Cleanup),
            );
        }
        app.add_systems(Startup, setup_thumbnail_light).add_systems(
            Update,
            (
                deactivate_thumbnail_preview_cameras_system,
                render_requested_thumbnail,
            )
                .chain()
                .in_set(EditorUpdateSet::Rendering),
        );
    }
}

// ============================================================================
// Thumbnail texture resources
// ============================================================================

/// Keys a placement thumbnail: one per block kind, plus a single pipe preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlacementPreviewKind {
    Block(BlockKind),
    Pipe,
}

/// The pre-rendered thumbnails for the placement dock and the gallery browser.
#[derive(Resource)]
pub(crate) struct ThumbnailTextures {
    textures: HashMap<ThumbnailKey, egui::TextureId>,
    pending: VecDeque<ThumbnailKey>,
    preload: VecDeque<ThumbnailKey>,
}

impl Default for ThumbnailTextures {
    fn default() -> Self {
        Self {
            textures: default(),
            pending: default(),
            preload: thumbnail_keys().collect(),
        }
    }
}

impl ThumbnailTextures {
    /// Queue a visible card once; its label remains usable while the image loads.
    pub(crate) fn get_or_request(&mut self, key: ThumbnailKey) -> Option<egui::TextureId> {
        if let Some(&texture) = self.textures.get(&key) {
            return Some(texture);
        }
        if !self.pending.contains(&key) {
            self.preload.retain(|queued| *queued != key);
            self.pending.push_back(key);
        }
        None
    }
}

const THUMBNAIL_IMAGE_SIZE: u32 = 384;
const THUMBNAIL_STARTUP_FRAMES: u8 = 2;
const THUMBNAIL_RENDER_FRAMES: u8 = 4;
const THUMBNAIL_CAMERA_BETA: f32 = 0.9;
const THUMBNAIL_DISTANCE_MARGIN: f32 = 1.12;
const THUMBNAIL_DISTANCE_PADDING: f32 = 1.5;

/// Marks an off-screen thumbnail camera and counts down the frames it must
/// render before it deactivates.
#[derive(Component, Default)]
struct ThumbnailPreviewCamera {
    frames_remaining: u8,
    #[cfg(not(target_arch = "wasm32"))]
    export: Option<(std::path::PathBuf, Handle<Image>)>,
}

/// Export the exact gallery textures rendered for the editor's cards.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub(crate) struct GalleryThumbnailExport {
    directory: std::path::PathBuf,
    remaining: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl GalleryThumbnailExport {
    pub(crate) fn new(directory: std::path::PathBuf) -> Self {
        Self {
            directory,
            remaining: GalleryItem::iter().count(),
        }
    }
}

/// Shared with the render world so thumbnail cameras outlive asynchronous GPU
/// pipeline compilation instead of preserving an empty render target.
#[derive(Resource, Clone)]
struct ThumbnailPipelinesReady(Arc<AtomicBool>);

impl Default for ThumbnailPipelinesReady {
    fn default() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }
}

fn update_thumbnail_pipeline_status(
    pipelines: Res<PipelineCache>,
    ready: Res<ThumbnailPipelinesReady>,
) {
    ready.0.store(
        pipelines.pipelines().all(|pipeline| {
            matches!(
                pipeline.state,
                CachedPipelineState::Ok(_) | CachedPipelineState::Err(_)
            )
        }),
        Ordering::Relaxed,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn thumbnail_target_image() -> Image {
    Image::new_target_texture(
        THUMBNAIL_IMAGE_SIZE,
        THUMBNAIL_IMAGE_SIZE,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    )
}

#[cfg(target_arch = "wasm32")]
fn thumbnail_target_image() -> Image {
    Image::new_target_texture(
        THUMBNAIL_IMAGE_SIZE,
        THUMBNAIL_IMAGE_SIZE,
        TextureFormat::Rgba8UnormSrgb,
        None,
    )
}

fn setup_thumbnail_light(mut commands: Commands) {
    let count = thumbnail_keys().count();
    commands.spawn((
        Name::new("Thumbnail directional light"),
        crate::systems::setup::main_editor_directional_light(),
        Transform::from_translation(crate::systems::setup::EDITOR_LIGHT_POSITION)
            .looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::from_layers(&(1..=count).collect::<Vec<_>>()),
    ));
}

/// Build one image after the previous camera finishes. Yielding between frames
/// keeps WASM preloading cooperative without a task's microtask loop blocking paint.
fn render_requested_thumbnail(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_cache: ResMut<RenderAssetCache>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
    mut thumbnails: ResMut<ThumbnailTextures>,
    cameras: Query<(), With<ThumbnailPreviewCamera>>,
    mut startup_frames: Local<u8>,
    #[cfg(not(target_arch = "wasm32"))] export: Option<Res<GalleryThumbnailExport>>,
) {
    if *startup_frames < THUMBNAIL_STARTUP_FRAMES {
        *startup_frames += 1;
        return;
    }
    if !cameras.is_empty() {
        return;
    }
    let Some(key) = thumbnails
        .pending
        .pop_front()
        .or_else(|| thumbnails.preload.pop_front())
    else {
        return;
    };
    let data = match key {
        ThumbnailKey::Placement(PlacementPreviewKind::Block(kind)) => {
            build_block_preview_data(kind)
        }
        ThumbnailKey::Placement(PlacementPreviewKind::Pipe) => build_pipe_preview_data(),
        ThumbnailKey::Gallery(entry) => build_gallery_preview_data(entry),
    }
    .map_points(|point| graph_to_world(point, 0.0));
    let points = collect_points(&data);
    let index = thumbnails.textures.len();
    let layer = index + 1;
    let render_layers = RenderLayers::layer(layer);

    let image = thumbnail_target_image();
    let image_handle = images.add(image);
    let texture_id = egui_user_textures.add_image(EguiTextureHandle::Strong(image_handle.clone()));

    let root = commands
        .spawn((
            Name::new(format!("Thumbnail {key:?} root")),
            Transform::default(),
            Visibility::Inherited,
            render_layers.clone(),
        ))
        .id();

    spawn_gltf_data_parts(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut render_cache,
        root,
        data,
        None,
        1.0,
        None,
        None,
        Some(render_layers.clone()),
        Pickable::IGNORE,
    );

    commands.spawn((
        Name::new(format!("Thumbnail {key:?} camera")),
        Camera3d::default(),
        // Match the main camera.
        Tonemapping::Linear,
        Camera {
            order: -100 - index as isize,
            clear_color: ClearColorConfig::Custom(Color::srgba(0.0, 0.0, 0.0, 0.0)),
            ..default()
        },
        Msaa::Off,
        RenderTarget::Image(image_handle.clone().into()),
        thumbnail_camera_transform(&points),
        render_layers,
        ThumbnailPreviewCamera {
            frames_remaining: THUMBNAIL_RENDER_FRAMES,
            #[cfg(not(target_arch = "wasm32"))]
            export: export.and_then(|export| match key {
                ThumbnailKey::Gallery(entry) => Some((
                    export.directory.join(format!("{}.png", entry.id())),
                    image_handle,
                )),
                ThumbnailKey::Placement(_) => None,
            }),
        },
    ));

    thumbnails.textures.insert(key, texture_id);
}

/// Counts down each thumbnail camera and switches it off once it has rendered
/// its frames, so thumbnails are drawn once rather than every frame.
fn deactivate_thumbnail_preview_cameras_system(
    mut commands: Commands,
    mut query: Populated<(Entity, &mut Camera, &mut ThumbnailPreviewCamera)>,
    pipelines_ready: Res<ThumbnailPipelinesReady>,
) {
    for (entity, mut camera, mut preview_camera) in &mut query {
        if !pipelines_ready.0.load(Ordering::Relaxed) {
            preview_camera.frames_remaining = THUMBNAIL_RENDER_FRAMES;
            continue;
        }
        if !camera.is_active || preview_camera.frames_remaining == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some((path, image)) = preview_camera.export.take() {
                commands.spawn(Screenshot::image(image)).observe(
                    move |captured: On<ScreenshotCaptured>,
                          mut export: ResMut<GalleryThumbnailExport>,
                          mut exit: MessageWriter<AppExit>| {
                        use color_eyre::eyre::WrapErr as _;
                        let result = captured
                            .image
                            .clone()
                            .try_into_dynamic()
                            .wrap_err("convert gallery thumbnail")
                            .and_then(|image| {
                                color_eyre::eyre::ensure!(
                                    image.to_rgba8().pixels().any(|pixel| pixel[3] != 0),
                                    "gallery thumbnail contains no rendered pixels"
                                );
                                image.save(&path).wrap_err("save gallery thumbnail")
                            });
                        if let Err(error) = result {
                            error!("{}: {error:?}", path.display());
                            exit.write(AppExit::error());
                            return;
                        }
                        export.remaining -= 1;
                        if export.remaining == 0 {
                            exit.write(AppExit::Success);
                        }
                    },
                );
                // Screenshot substitutes the target for this frame. Keep its
                // camera active so the captured texture receives the scene.
                continue;
            }
            camera.is_active = false;
            commands.entity(entity).remove::<ThumbnailPreviewCamera>();
        } else {
            preview_camera.frames_remaining -= 1;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ThumbnailKey {
    Placement(PlacementPreviewKind),
    Gallery(GalleryItem),
}

fn thumbnail_keys() -> impl Iterator<Item = ThumbnailKey> {
    BlockKind::all_kinds()
        .into_iter()
        .map(|kind| ThumbnailKey::Placement(PlacementPreviewKind::Block(kind)))
        .chain(std::iter::once(ThumbnailKey::Placement(
            PlacementPreviewKind::Pipe,
        )))
        .chain(GalleryItem::iter().map(ThumbnailKey::Gallery))
}

/// Builds the glTF preview geometry for a single block kind at the origin.
fn build_block_preview_data(kind: BlockKind) -> GltfData {
    let block = Block::new(IVec3::ZERO, kind);
    let mut graph = BlockGraph::default();
    graph.add_block(block.clone());
    block_as_gltf_data(&block, &graph)
}

/// Builds the glTF preview geometry for a pipe between two cubes.
fn build_pipe_preview_data() -> GltfData {
    let mut graph = BlockGraph::default();
    let start = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ));
    let end = Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ));
    let pipe = Pipe::new(IVec3::ZERO, Direction::XPLUS);
    graph.add_block(start.clone());
    graph.add_block(end.clone());
    graph.add_pipe(pipe.clone());
    pipe_as_gltf_data(&start, &end, &pipe, &graph, 2.0)
}

/// Builds the glTF preview geometry for a gallery example, scaling the pipe
/// length to the graph's extent so it frames tidily.
fn build_gallery_preview_data(entry: GalleryItem) -> GltfData {
    let graph = entry.build().flatten().expect("gallery geometry assembles");
    let pipe_length = gallery_thumbnail_pipe_length(&graph);
    block_graph_as_gltf_data(&graph, pipe_length)
}

fn gallery_thumbnail_pipe_length(graph: &BlockGraph) -> f32 {
    let Some((min, max)) = graph_bounds(graph) else {
        return 2.0;
    };
    let span = max - min;
    let max_extent = span.x.max(span.y).max(span.z);
    match max_extent {
        0..=2 => 2.3,
        3..=4 => 1.8,
        5..=7 => 1.4,
        _ => 1.1,
    }
}

fn collect_points(data: &GltfData) -> Vec<Vec3> {
    let mut points = Vec::new();
    for triangles in data.triangles.values() {
        for triangle in triangles {
            points.extend(triangle);
        }
    }
    for lines in data.lines.values() {
        for line in lines {
            points.extend(line);
        }
    }
    for label in &data.texts {
        points.push(label.position);
    }
    points
}

fn thumbnail_camera_transform(points: &[Vec3]) -> Transform {
    let camera_dir = thumbnail_camera_direction();
    if points.is_empty() {
        let mut transform = Transform::default();
        let camera = CameraSettings::default();
        apply_camera_setting(&mut transform, &camera);
        return transform;
    }

    let center = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let forward = (-camera_dir).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward).normalize();
    let tan_half_fov = (CAMERA_DEFAULT_FOV_Y * 0.5).tan();

    let mut required_distance = 0.0f32;
    for point in points {
        let relative = *point - center;
        let toward_camera = relative.dot(camera_dir);
        let horizontal = relative.dot(right).abs();
        let vertical = relative.dot(up).abs();
        required_distance = required_distance.max(horizontal / tan_half_fov + toward_camera);
        required_distance = required_distance.max(vertical / tan_half_fov + toward_camera);
    }

    let (min, max) = points.iter().copied().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), point| (min.min(point), max.max(point)),
    );
    let fallback_distance = (max - min).length() * 0.75 + THUMBNAIL_DISTANCE_PADDING;
    let distance = (required_distance * THUMBNAIL_DISTANCE_MARGIN).max(fallback_distance);

    Transform::from_translation(center + camera_dir * distance).looking_at(center, Vec3::Y)
}

fn thumbnail_camera_direction() -> Vec3 {
    let mut transform = Transform::default();
    let camera = CameraSettings {
        beta: THUMBNAIL_CAMERA_BETA,
        ..Default::default()
    };
    apply_camera_setting(&mut transform, &camera);
    (transform.translation - camera.focus).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_previews_include_child_geometry() {
        let source = GalleryItem::PhaseGradientK4.build();
        let root = block_graph_as_gltf_data(&source, gallery_thumbnail_pipe_length(&source));
        let complete = build_gallery_preview_data(GalleryItem::PhaseGradientK4);
        assert!(collect_points(&complete).len() > 10 * collect_points(&root).len());
    }

    #[test]
    fn thumbnails_preload_after_startup_with_visible_requests_first() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<RenderAssetCache>()
            .init_resource::<EguiUserTextures>()
            .add_plugins(ThumbnailsPlugin);
        for _ in 0..THUMBNAIL_STARTUP_FRAMES {
            app.update();
            assert!(app.world().resource::<Assets<Image>>().is_empty());
        }

        let pipe = ThumbnailKey::Placement(PlacementPreviewKind::Pipe);
        let cnot = ThumbnailKey::Gallery(GalleryItem::CNOT);
        {
            let mut textures = app.world_mut().resource_mut::<ThumbnailTextures>();
            assert!(textures.get_or_request(pipe).is_none());
            assert!(textures.get_or_request(pipe).is_none());
            assert!(textures.get_or_request(cnot).is_none());
            assert_eq!(textures.pending.len(), 2);
        }
        app.update();
        let texture_id = app
            .world_mut()
            .resource_mut::<ThumbnailTextures>()
            .get_or_request(pipe)
            .unwrap();
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 1);
        for _ in 0..THUMBNAIL_RENDER_FRAMES {
            app.update();
            assert_eq!(app.world().resource::<Assets<Image>>().len(), 1);
        }
        app.update();
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 2);
        assert_eq!(
            app.world_mut()
                .query_filtered::<(), With<ThumbnailPreviewCamera>>()
                .iter(app.world())
                .count(),
            1
        );
        let mut textures = app.world_mut().resource_mut::<ThumbnailTextures>();
        assert_eq!(textures.get_or_request(pipe), Some(texture_id));
        assert!(textures.get_or_request(cnot).is_some());
        assert!(textures.pending.is_empty());

        let count = thumbnail_keys().count();
        for _ in 0..count * (usize::from(THUMBNAIL_RENDER_FRAMES) + 1) {
            app.update();
            assert!(
                app.world_mut()
                    .query_filtered::<(), With<ThumbnailPreviewCamera>>()
                    .iter(app.world())
                    .count()
                    <= 1
            );
        }
        assert_eq!(app.world().resource::<Assets<Image>>().len(), count);
        let mut textures = app.world_mut().resource_mut::<ThumbnailTextures>();
        assert!(thumbnail_keys().all(|key| textures.get_or_request(key).is_some()));
        assert!(textures.pending.is_empty());
        assert!(textures.preload.is_empty());
    }

    #[test]
    fn thumbnail_camera_waits_for_pipelines_then_stops_polling() {
        let mut app = App::new();
        app.init_resource::<ThumbnailPipelinesReady>()
            .add_systems(Update, deactivate_thumbnail_preview_cameras_system);
        let marker = ThumbnailPreviewCamera {
            frames_remaining: 0,
            ..default()
        };
        let entity = app.world_mut().spawn((Camera::default(), marker)).id();

        app.world()
            .resource::<ThumbnailPipelinesReady>()
            .0
            .store(false, Ordering::Relaxed);
        app.update();
        let camera = app.world().entity(entity);
        assert!(camera.get::<Camera>().unwrap().is_active);
        assert_eq!(
            camera
                .get::<ThumbnailPreviewCamera>()
                .unwrap()
                .frames_remaining,
            THUMBNAIL_RENDER_FRAMES
        );

        app.world()
            .resource::<ThumbnailPipelinesReady>()
            .0
            .store(true, Ordering::Relaxed);
        for _ in 0..=THUMBNAIL_RENDER_FRAMES {
            app.update();
        }
        let camera = app.world().entity(entity);
        assert!(!camera.get::<Camera>().unwrap().is_active);
        assert!(!camera.contains::<ThumbnailPreviewCamera>());
    }
}
