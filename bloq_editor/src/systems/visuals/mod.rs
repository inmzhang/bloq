//! Synchronizing the 3D scene with the working graph: rebuilding block, pipe,
//! and stabilizer meshes (through the geometry/asset caches), placement and
//! endpoint previews, and pointer pickability.
//!
//! The heavy work is gated by the render sync-state and signature caches in
//! `resources`, so an unchanged graph reuses its meshes and only touched
//! elements are rebuilt.

use std::collections::{HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::num::NonZero;

use crate::components::{
    EditorCamera, GraphElement, GraphRoot, OriginalMaterial, PipeKey, PreviewEndpoint,
    PreviewEndpointMesh, PreviewMesh, WalkingPreviewMesh,
};
use crate::resources::{
    BloqViewerState, EditorMode, EditorState, EditorTabId, EditorTabs, GraphState, Notifications,
    PlacementTool,
};
use crate::systems::input::{
    PIPE_HINT_OFFSETS, WALKING_MOVEMENTS, is_hadamard_pipe_modifier_pressed,
    moving_block_candidate_kind, moving_block_movements, moving_block_start_has_candidate,
    on_element_click, on_element_drag_end, on_element_drag_start, on_element_out, on_element_over,
    patch_rotation_pipe_hint_target, pipe_candidate_at_with_snapshot,
    pipe_hint_target_with_snapshot, pipe_mode_selected_elements, tall_cube_pipe_hint_target,
    walking_pipe_hint_target, walking_port_promotion_candidate,
};
use crate::theme::{self, ThemePreset};
use crate::utils::{
    ADJACENT_DIRECTIONS, GraphAdjacencySnapshot, compact_connectable_offsets,
    displayed_branch_projection, graph_to_world, is_pipe_visible,
};
use bevy::asset::RenderAssetUsages;
use bevy::camera::{primitives::Aabb, visibility::RenderLayers};
use bevy::light::NotShadowCaster;
use bevy::picking::{Pickable, mesh_picking::ray_cast::SimplifiedMesh};
use bevy::prelude::*;
#[cfg(target_arch = "wasm32")]
use bevy::render::batching::NoAutomaticBatching;
use bevy::render::render_resource::{Face, PrimitiveTopology};
use bevy_rich_text3d::{Text3d, Text3dStyling, TextAtlas};
use bloq_graph::{
    Basis, Block, BlockGraph, BlockGraphError, BlockKind, GltfData, Pipe, RGBA,
    block_as_gltf_data_with_pipe_length, pipe_between_positions_as_gltf_data,
    stabilizer_as_gltf_data,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::systems::EditorUpdateSet;

mod geometry;
mod gltf_assets;
mod pickability;
mod signature;

use geometry::*;
use gltf_assets::*;
use pickability::*;
use signature::*;

pub(crate) use geometry::RenderAssetCache;
pub(crate) use gltf_assets::spawn_gltf_data_parts;
pub(crate) use pickability::sync_graph_pickability_system;
pub(crate) use signature::{
    RenderSignatureBlockIdentity, RenderSignatureConnectableOffsets, RenderSignaturePipeIdentity,
};

/// Scene synchronization: registers mesh rebuild, pickability, and previews.
pub(crate) struct VisualsPlugin;

#[derive(Component)]
struct PortTagText;

const PORT_TAG_GAP: f32 = 0.18;
const PORT_TAG_SCALE: f32 = 0.375;
const PORT_TAG_HALF_CHAR_WIDTH: f32 = 0.3 * PORT_TAG_SCALE;
const PORT_TAG_HALF_HEIGHT: f32 = 0.5 * PORT_TAG_SCALE;

impl Plugin for VisualsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GraphRenderState>()
            .init_resource::<RenderAssetCache>()
            .add_systems(
                Update,
                block_graph_visual_system
                    .run_if(|state: Res<GraphState>| state.needs_rerender)
                    .in_set(EditorUpdateSet::Rendering),
            )
            .add_systems(
                Update,
                (
                    sync_graph_pickability_system.after(crate::systems::input::undo_redo_system),
                    face_port_tags_to_camera,
                    add_block_preview_system
                        .after(crate::systems::input::sync_interaction_highlight_system),
                )
                    .in_set(EditorUpdateSet::Interaction),
            );
    }
}

/// Rebuilds the block/pipe/stabilizer scene meshes to match the current graph,
/// reusing cached geometry and touching only elements that changed. Runs only
/// when a rerender was requested.
pub(crate) fn block_graph_visual_system(
    mut commands: Commands,
    mut graph_state: ResMut<GraphState>,
    tabs: Res<EditorTabs>,
    editor_state: Res<EditorState>,
    circuit_viewer: Res<BloqViewerState>,
    mut notifications: ResMut<Notifications>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_state: ResMut<GraphRenderState>,
    mut render_cache: ResMut<RenderAssetCache>,
    mut geometry_cache: Local<RenderGeometryCache>,
    mut signature_context_cache: Local<RenderSignatureContextCache>,
    graph_root: Single<Entity, With<GraphRoot>>,
) {
    let _span = info_span!("editor.block_graph_visual_system").entered();
    graph_state.needs_rerender = false;
    render_state.revision = render_state.revision.saturating_add(1);

    let pipe_length = editor_state.pipe_length;
    evict_stale_render_caches(
        &mut render_cache,
        &mut geometry_cache,
        pipe_length.to_bits(),
    );
    let view_current_layer_only =
        editor_state.view_current_layer_only && editor_state.mode == EditorMode::View;
    let plane_height = editor_state.plane_height;
    let show_stabilizers = editor_state.showing_stabilizers();
    let cleanup_state = RenderSignatureContextSyncState {
        tab_id: tabs.active,
        graph_revision: graph_state.revision,
        pipe_length_bits: pipe_length.to_bits(),
    };
    let mesh_sync_state = RenderMeshSyncState {
        tab_id: tabs.active,
        graph_revision: graph_state.revision,
        pipe_length_bits: pipe_length.to_bits(),
        view_current_layer_only,
        plane_height,
        show_stabilizers,
        show_port_tags: editor_state.show_port_tags,
        module_view_active: editor_state.mode == EditorMode::Module,
        theme_preset: editor_state.theme_preset,
    };

    let stabilizer_sync = RenderStabilizerSyncState {
        tab_id: tabs.active,
        graph_revision: graph_state.revision,
        pipe_length_bits: pipe_length.to_bits(),
        view_current_layer_only,
        plane_height,
        stabilizer_index: editor_state.current_stabilizer_index,
        action_ordinal: editor_state
            .action_stabilizer
            .as_ref()
            .map(|(ordinal, _)| *ordinal),
    };
    sync_stabilizer_overlay(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut render_state,
        &mut render_cache,
        &graph_state,
        &editor_state,
        &mut notifications,
        *graph_root,
        pipe_length,
        view_current_layer_only,
        show_stabilizers,
        stabilizer_sync,
    );

    if render_state.mesh_sync == Some(mesh_sync_state) {
        return;
    }

    // Built only past the early return: on unchanged frames (the common case)
    // the mesh-sync key matches above and we skip this, avoiding the two
    // per-frame hover/selection HashSet clones this context allocates.
    let interaction =
        InteractionMaterialContext::new(&editor_state, &circuit_viewer, &graph_state.graph);
    let signature_context = RenderSignatureContext::cached(
        &mut signature_context_cache,
        &tabs,
        &graph_state,
        pipe_length,
    );

    sync_blocks(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut render_state,
        &mut render_cache,
        &mut geometry_cache,
        &graph_state,
        &editor_state,
        *graph_root,
        pipe_length,
        view_current_layer_only,
        plane_height,
        show_stabilizers,
        &interaction,
        &signature_context,
        cleanup_state,
    );

    sync_pipes(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut render_state,
        &mut render_cache,
        &mut geometry_cache,
        &graph_state,
        &editor_state,
        *graph_root,
        pipe_length,
        view_current_layer_only,
        plane_height,
        show_stabilizers,
        &interaction,
        &signature_context,
        cleanup_state,
    );
    render_state.mesh_sync = Some(mesh_sync_state);
    render_state.stale_cleanup = Some(cleanup_state);
}

/// Drops the render caches that were keyed to a previous pipe length. Both the
/// geometry templates (whose keys embed `pipe_length_bits`) and the shared mesh
/// handles (keyed by geometry-derived hashes that vary with pipe length) would
/// otherwise accumulate one dead generation per slider change, pinning their GPU
/// buffers alive forever. Materials are intentionally left untouched: their keys
/// depend on color/theme/alpha, not pipe length.
fn evict_stale_render_caches(
    render_cache: &mut RenderAssetCache,
    geometry_cache: &mut RenderGeometryCache,
    pipe_length_bits: u32,
) {
    if geometry_cache.pipe_length_generation == Some(pipe_length_bits) {
        return;
    }
    geometry_cache.pipe_length_generation = Some(pipe_length_bits);

    geometry_cache
        .block_templates
        .retain(|key, _| key.pipe_length_bits == pipe_length_bits);
    geometry_cache
        .pipe_templates
        .retain(|key, _| key.pipe_length_bits == pipe_length_bits);
    // The per-element entries only reference the templates above; clear them so
    // stale template keys cannot linger. The whole graph re-syncs on this frame.
    geometry_cache.blocks.clear();
    geometry_cache.pipes.clear();

    // Mesh/pick-proxy keys carry no pipe length, so they cannot be filtered
    // precisely; the re-sync below repopulates whatever the new length needs.
    render_cache.triangle_meshes.clear();
    render_cache.line_meshes.clear();
    render_cache.pick_proxy_meshes.clear();
}

fn sync_stabilizer_overlay(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_state: &mut GraphRenderState,
    render_cache: &mut RenderAssetCache,
    graph_state: &GraphState,
    editor_state: &EditorState,
    notifications: &mut Notifications,
    graph_root: Entity,
    pipe_length: f32,
    view_current_layer_only: bool,
    show_stabilizers: bool,
    stabilizer_sync: RenderStabilizerSyncState,
) {
    if !show_stabilizers {
        if let Some(old) = render_state.stabilizer.take() {
            commands.entity(old.entity).despawn();
        }
        render_state.stabilizer_sync = None;
        return;
    }

    // Skip the graph clone + glTF build + hash below when none of the overlay's
    // inputs changed since the last build. This runs every frame ahead of the
    // mesh-sync early return, so the gate matters even when the graph is idle.
    if render_state.stabilizer_sync == Some(stabilizer_sync) && render_state.stabilizer.is_some() {
        return;
    }

    let stabilizer = editor_state
        .shown_stabilizer()
        .expect("show_stabilizers means one regular or action surface exists");
    let graph = match stabilizer_view_graph(
        &graph_state.graph,
        view_current_layer_only,
        editor_state.plane_height,
    ) {
        Ok(graph) => graph,
        Err(err) => {
            notifications.push_error(format!("Error to resolve stabilizer view: {err}"));
            return;
        }
    };

    let data = match stabilizer_as_gltf_data(stabilizer, &graph, pipe_length) {
        Ok(data) => data.map_points(|p| graph_to_world(p, 0.0)),
        Err(err) => {
            notifications.push_error(format!("Error to convert stabilizer to mesh: {}", err));
            return;
        }
    };

    // Record the key now (fresh data in hand) so both the signature-unchanged
    // return and the respawn path below stop re-deriving it next frame.
    render_state.stabilizer_sync = Some(stabilizer_sync);

    let signature = combine_signature(hash_gltf_data(&data), 1.0f32.to_bits(), "stabilizer");
    if let Some(existing) = &render_state.stabilizer
        && existing.signature == signature
    {
        return;
    }

    if let Some(old) = render_state.stabilizer.take() {
        commands.entity(old.entity).despawn();
    }

    let stabilizer_entity = commands
        .spawn((
            Transform::from_translation(Vec3::ZERO),
            Visibility::Inherited,
        ))
        .id();
    commands.entity(graph_root).add_child(stabilizer_entity);

    spawn_gltf_data_parts(
        commands,
        meshes,
        materials,
        render_cache,
        stabilizer_entity,
        data,
        None,
        1.0,
        None,
        None,
        None,
        Pickable::IGNORE,
    );

    render_state.stabilizer = Some(rendered_element(stabilizer_entity, signature, 0));
}

fn stabilizer_view_graph(
    graph: &BlockGraph,
    current_layer_only: bool,
    plane_height: i32,
) -> Result<BlockGraph, BlockGraphError> {
    if current_layer_only {
        Ok(graph.layer(plane_height).into_graph())
    } else {
        displayed_branch_projection(graph)
    }
}

fn sync_blocks(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_state: &mut GraphRenderState,
    render_cache: &mut RenderAssetCache,
    geometry_cache: &mut RenderGeometryCache,
    graph_state: &GraphState,
    editor_state: &EditorState,
    graph_root: Entity,
    pipe_length: f32,
    view_current_layer_only: bool,
    plane_height: i32,
    show_stabilizers: bool,
    interaction: &InteractionMaterialContext,
    signature_context: &RenderSignatureContext,
    cleanup_state: RenderSignatureContextSyncState,
) {
    let occupied = editor_state.show_port_tags.then(|| {
        graph_state
            .graph
            .occupied_positions()
            .collect::<FxHashSet<_>>()
    });
    for block in graph_state.graph.blocks() {
        let pos = block.pos();
        let element = GraphElement::Block(pos);
        let module_index = displayed_module(editor_state, &graph_state.graph, element);

        let mut alpha: f32 = if view_current_layer_only && !block.occupies_layer(plane_height) {
            0.1
        } else {
            1.0
        };
        alpha *= stabilizer_context_alpha(show_stabilizers, editor_state.theme_preset);

        let base_geometry_signature =
            signature_context.block_geometry_signature(block, pipe_length);
        let port_tag = if block.kind().is_port() {
            block.tag().and_then(|tag| {
                port_tag_direction(&graph_state.graph, pos, occupied.as_ref()?)
                    .map(|direction| (tag, direction, port_tag_position(direction, tag)))
            })
        } else {
            None
        };
        let geometry_signature = port_tag.map_or(base_geometry_signature, |(tag, direction, _)| {
            let mut hasher = DefaultHasher::new();
            "port_tag".hash(&mut hasher);
            base_geometry_signature.hash(&mut hasher);
            tag.hash(&mut hasher);
            direction.hash(&mut hasher);
            hasher.finish()
        });
        let render_signature = module_render_signature(
            combine_signature(geometry_signature, alpha.to_bits(), "block_mesh"),
            module_index,
        );
        if let Some(existing) = render_state.blocks.get(&pos)
            && existing.signature == render_signature
        {
            continue;
        }

        let center_world = graph_to_world(pos.as_vec3(), pipe_length);
        let cached = cached_block_geometry(
            geometry_cache,
            block,
            signature_context,
            &graph_state.graph,
            pipe_length,
            base_geometry_signature,
        );
        let face_material =
            module_index.map(|index| module_face_material(index, render_cache, materials));

        upsert_rendered(
            commands,
            meshes,
            materials,
            render_cache,
            &mut render_state.blocks,
            graph_root,
            pos,
            element,
            center_world,
            render_signature,
            geometry_signature,
            &cached.data,
            &cached.mesh_keys,
            PickProxyBounds {
                min: cached.pick_proxy_min,
                max: cached.pick_proxy_max,
            },
            alpha,
            face_material,
            interaction.material_for(editor_state, element),
            port_tag.map(|(tag, _, position)| (tag, position)),
        );
    }

    if render_state.stale_cleanup != Some(cleanup_state) {
        let stale: Vec<_> = render_state
            .blocks
            .keys()
            .copied()
            .filter(|pos| !graph_state.graph.has_block_at(*pos))
            .collect();
        for key in stale {
            if let Some(old) = render_state.blocks.remove(&key) {
                geometry_cache.blocks.remove(&key);
                commands.entity(old.entity).despawn();
            }
        }
    }
}

fn sync_pipes(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_state: &mut GraphRenderState,
    render_cache: &mut RenderAssetCache,
    geometry_cache: &mut RenderGeometryCache,
    graph_state: &GraphState,
    editor_state: &EditorState,
    graph_root: Entity,
    pipe_length: f32,
    view_current_layer_only: bool,
    plane_height: i32,
    show_stabilizers: bool,
    interaction: &InteractionMaterialContext,
    signature_context: &RenderSignatureContext,
    cleanup_state: RenderSignatureContextSyncState,
) {
    for (u_pos, v_pos, _, _, pipe) in graph_state.graph.pipe_endpoints_with_blocks() {
        let key = PipeKey::new(u_pos, v_pos);
        let element = GraphElement::Pipe(u_pos, v_pos).canonical();
        let module_index = displayed_module(editor_state, &graph_state.graph, element);

        let mut alpha: f32 =
            if view_current_layer_only && !is_pipe_visible(u_pos, v_pos, plane_height) {
                0.1
            } else {
                1.0
            };
        alpha *= stabilizer_context_alpha(show_stabilizers, editor_state.theme_preset);

        let geometry_signature =
            signature_context.pipe_geometry_signature(u_pos, v_pos, pipe, pipe_length);
        let render_signature = module_render_signature(
            combine_signature(geometry_signature, alpha.to_bits(), "pipe_mesh"),
            module_index,
        );

        if let Some(existing) = render_state.pipes.get(&key)
            && existing.signature == render_signature
        {
            continue;
        }

        let diff = (v_pos - u_pos).as_vec3();
        let center_graph = u_pos.as_vec3() + diff * 0.5;
        let center_world = graph_to_world(center_graph, pipe_length);
        let cached = cached_pipe_geometry(
            geometry_cache,
            key,
            pipe,
            signature_context,
            &graph_state.graph,
            pipe_length,
            geometry_signature,
        );
        let face_material =
            module_index.map(|index| module_face_material(index, render_cache, materials));

        upsert_rendered(
            commands,
            meshes,
            materials,
            render_cache,
            &mut render_state.pipes,
            graph_root,
            key,
            element,
            center_world,
            render_signature,
            geometry_signature,
            &cached.data,
            &cached.mesh_keys,
            PickProxyBounds {
                min: cached.pick_proxy_min,
                max: cached.pick_proxy_max,
            },
            alpha,
            face_material,
            interaction.material_for(editor_state, element),
            None,
        );
    }

    if render_state.stale_cleanup != Some(cleanup_state) {
        let stale: Vec<_> = render_state
            .pipes
            .keys()
            .copied()
            .filter(|key| !graph_state.graph.has_pipe_between(key.a, key.b))
            .collect();
        for key in stale {
            if let Some(old) = render_state.pipes.remove(&key) {
                geometry_cache.pipes.remove(&key);
                commands.entity(old.entity).despawn();
            }
        }
    }
}

fn displayed_module(
    editor_state: &EditorState,
    graph: &BlockGraph,
    element: GraphElement,
) -> Option<usize> {
    if editor_state.mode != EditorMode::Module {
        return None;
    }
    editor_state
        .module_view
        .as_ref()?
        .module_for_element(graph, element)
}

fn module_face_material(
    index: usize,
    render_cache: &mut RenderAssetCache,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    get_or_insert_material(
        render_cache,
        materials,
        theme::module_color(index).to_normalized_gamma_f32(),
        MaterialStyle {
            unlit: true,
            double_sided: true,
            cull_none: true,
        },
    )
}

fn module_render_signature(base: u64, module: Option<usize>) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    module.hash(&mut hasher);
    hasher.finish()
}

struct InteractionMaterialContext {
    selected: HashSet<GraphElement>,
    hovered: HashSet<GraphElement>,
    candidates: HashSet<GraphElement>,
}

impl InteractionMaterialContext {
    fn new(
        editor_state: &EditorState,
        circuit_viewer: &BloqViewerState,
        graph: &bloq_graph::BlockGraph,
    ) -> Self {
        let mut hovered = editor_state.blog_hovered_element_set().clone();
        hovered.extend(circuit_viewer.hovered_source_elements_iter());
        hovered.extend(editor_state.zx_hovered_element_set().iter().copied());
        hovered.extend(editor_state.action_hovered_elements.iter().copied());
        hovered.extend(editor_state.hovered_element_for_highlight());
        Self {
            selected: pipe_mode_selected_elements(editor_state, graph),
            hovered,
            candidates: editor_state.action_candidate_elements.clone(),
        }
    }

    fn material_for(
        &self,
        editor_state: &EditorState,
        element: GraphElement,
    ) -> Option<Handle<StandardMaterial>> {
        let element = element.canonical();
        if self.selected.contains(&element) {
            Some(editor_state.selection_material.clone())
        } else if self.hovered.contains(&element) {
            Some(editor_state.highlight_material.clone())
        } else if self.candidates.contains(&element) {
            Some(editor_state.action_candidate_material.clone())
        } else {
            None
        }
    }
}

fn stabilizer_context_alpha(show_stabilizers: bool, theme_preset: ThemePreset) -> f32 {
    if !show_stabilizers {
        return 1.0;
    }

    match theme_preset {
        ThemePreset::Light => 0.42,
        ThemePreset::GruvboxMaterial => 0.24,
    }
}

fn port_tag_direction(
    graph: &BlockGraph,
    port: IVec3,
    occupied: &FxHashSet<IVec3>,
) -> Option<IVec3> {
    let preferred = graph
        .neighbor_positions(port)
        .into_iter()
        .next()
        .map(|neighbor| port - neighbor);
    preferred
        .into_iter()
        .chain([
            IVec3::Z,
            IVec3::X,
            IVec3::NEG_X,
            IVec3::NEG_Z,
            IVec3::Y,
            IVec3::NEG_Y,
        ])
        .find(|direction| {
            port.checked_add(*direction)
                .is_some_and(|pos| !occupied.contains(&pos))
        })
}

fn port_tag_position(direction: IVec3, tag: &str) -> Vec3 {
    let half_extent = if direction.z == 0 {
        tag.chars().count() as f32 * PORT_TAG_HALF_CHAR_WIDTH
    } else {
        PORT_TAG_HALF_HEIGHT
    };
    graph_to_world(direction.as_vec3(), 0.0) * (0.5 + PORT_TAG_GAP + half_extent)
}

fn face_port_tags_to_camera(
    camera: Single<&Transform, With<EditorCamera>>,
    mut tags: Query<&mut Transform, (With<PortTagText>, Without<EditorCamera>)>,
) {
    for mut tag in &mut tags {
        tag.rotation = camera.rotation;
    }
}

/// Inserts or refreshes the rendered entity for a block or pipe. The two kinds
/// share this body verbatim, differing only in which map (`blocks` vs `pipes`)
/// and key type stores the result, so it is generic over the key.
fn upsert_rendered<K: Eq + Hash + Copy>(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_cache: &mut RenderAssetCache,
    map: &mut FxHashMap<K, RenderedElement>,
    graph_root: Entity,
    key: K,
    element: GraphElement,
    center_world: Vec3,
    signature: u64,
    geometry_signature: u64,
    data: &GltfData,
    mesh_keys: &CachedMeshKeys,
    pick_proxy: PickProxyBounds,
    alpha: f32,
    face_material_override: Option<Handle<StandardMaterial>>,
    display_material_override: Option<Handle<StandardMaterial>>,
    port_tag: Option<(&str, Vec3)>,
) {
    if let Some(existing) = map.get(&key) {
        if existing.signature == signature {
            return;
        }
        if existing.geometry_signature == geometry_signature {
            let entity = existing.entity;
            let mesh_parts = existing.mesh_parts.clone();
            update_cached_mesh_part_materials(
                commands,
                materials,
                render_cache,
                data,
                &mesh_parts,
                alpha,
                face_material_override,
                display_material_override,
                editor_triangle_material_style(),
            );
            map.insert(
                key,
                RenderedElement {
                    entity,
                    signature,
                    geometry_signature,
                    mesh_parts,
                },
            );
            return;
        }
    }

    if let Some(old) = map.remove(&key) {
        commands.entity(old.entity).despawn();
    }

    let entity = commands
        .spawn((
            SimplifiedMesh(get_or_insert_pick_proxy_mesh(
                render_cache,
                meshes,
                pick_proxy.min,
                pick_proxy.max,
            )),
            Aabb::from_min_max(pick_proxy.min, pick_proxy.max),
            Transform::from_translation(center_world),
            Visibility::Inherited,
            element,
            Pickable::default(),
        ))
        .observe(on_element_over)
        .observe(on_element_out)
        .observe(on_element_drag_start)
        .observe(on_element_drag_end)
        .observe(on_element_click)
        .id();
    commands.entity(graph_root).add_child(entity);

    let mesh_parts = spawn_gltf_data_parts_ref(
        commands,
        meshes,
        materials,
        render_cache,
        entity,
        data,
        Some(mesh_keys),
        alpha,
        face_material_override,
        display_material_override,
        None,
        Pickable::IGNORE,
    );
    if let Some((text, position)) = port_tag {
        let material = get_or_insert_text_material(render_cache, materials);
        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Text3d::new(text),
                text_label_style(Srgba::RED, PORT_TAG_SCALE),
                Transform::from_translation(position),
                Mesh3d::default(),
                MeshMaterial3d(material),
                Pickable::IGNORE,
                PortTagText,
            ));
        });
    }

    map.insert(
        key,
        RenderedElement {
            entity,
            signature,
            geometry_signature,
            mesh_parts,
        },
    );
}

fn rendered_element(entity: Entity, signature: u64, geometry_signature: u64) -> RenderedElement {
    RenderedElement {
        entity,
        signature,
        geometry_signature,
        mesh_parts: RenderedMeshParts::default(),
    }
}

/// A block or pipe's rendered scene entity plus the signatures used to decide
/// whether its mesh can be reused across frames.
#[derive(Debug, Clone)]
pub(crate) struct RenderedElement {
    pub(crate) entity: Entity,
    pub(crate) signature: u64,
    pub(crate) geometry_signature: u64,
    pub(crate) mesh_parts: RenderedMeshParts,
}

/// The child triangle- and line-mesh entities of a rendered element, tracked so
/// highlight overlays can retint them directly.
#[derive(Debug, Clone, Default)]
pub(crate) struct RenderedMeshParts {
    pub(crate) triangles: Vec<Entity>,
    pub(crate) lines: Vec<Entity>,
}

/// Inputs the block/pipe mesh set depends on; when unchanged, the visual system
/// skips its per-frame rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RenderMeshSyncState {
    pub(crate) tab_id: EditorTabId,
    pub(crate) graph_revision: u64,
    pub(crate) pipe_length_bits: u32,
    pub(crate) view_current_layer_only: bool,
    pub(crate) plane_height: i32,
    pub(crate) show_stabilizers: bool,
    pub(crate) show_port_tags: bool,
    pub(crate) module_view_active: bool,
    pub(crate) theme_preset: ThemePreset,
}

/// Inputs the stabilizer overlay mesh is derived from. Lets the overlay skip
/// its per-frame graph clone + glTF build + hash when nothing it depends on has
/// changed, mirroring how `RenderMeshSyncState` gates the block/pipe meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RenderStabilizerSyncState {
    pub(crate) tab_id: EditorTabId,
    pub(crate) graph_revision: u64,
    pub(crate) pipe_length_bits: u32,
    pub(crate) view_current_layer_only: bool,
    pub(crate) plane_height: i32,
    pub(crate) stabilizer_index: usize,
    pub(crate) action_ordinal: Option<usize>,
}

/// Tracks the scene entities currently rendered for each block, pipe, and the
/// stabilizer overlay, plus the sync states gating their rebuilds.
#[derive(Resource, Default)]
pub(crate) struct GraphRenderState {
    pub(crate) blocks: FxHashMap<IVec3, RenderedElement>,
    pub(crate) pipes: FxHashMap<PipeKey, RenderedElement>,
    pub(crate) stabilizer: Option<RenderedElement>,
    pub(crate) stabilizer_sync: Option<RenderStabilizerSyncState>,
    pub(crate) revision: u64,
    pub(crate) mesh_sync: Option<RenderMeshSyncState>,
    pub(crate) stale_cleanup: Option<RenderSignatureContextSyncState>,
}

impl GraphRenderState {
    /// Resolves the rendered entity for an element by dispatching to the
    /// per-kind map, replacing the previously duplicated `elements` index.
    pub(crate) fn rendered_for(&self, element: GraphElement) -> Option<&RenderedElement> {
        match element.canonical() {
            GraphElement::Block(pos) => self.blocks.get(&pos),
            GraphElement::Pipe(u, v) => self.pipes.get(&PipeKey::new(u, v)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PlacementPreviewSyncState, RenderGeometryCache, RenderSignatureContext,
        RenderSignatureContextCache, RenderSignatureContextSyncState, analyze_gltf_geometry,
        block_render_geometry_signature, editor_triangle_material_style, graph_pickability,
        hash_line_positions, hash_positions, hash_triangle_positions, pick_box_positions,
        pipe_preview_hint_targets, pipe_preview_transform, pipe_render_geometry_signature,
        port_tag_direction, port_tag_position, rebuild_render_signature_context_cache,
        set_pickable_if_changed, set_visibility_if_changed, stabilizer_context_alpha,
        stabilizer_view_graph, sync_render_signature_context_cache,
    };
    use crate::components::PipeKey;
    use crate::resources::{
        EditorMode, EditorState, EditorTabId, EditorTabs, GraphState, PlacementTool,
    };
    use crate::theme::ThemePreset;
    use crate::utils::{GraphAdjacencySnapshot, adjacent_direction_bit};
    use bevy::picking::Pickable;
    use bevy::prelude::Visibility;
    use bevy::render::render_resource::PrimitiveTopology;
    use bloq_graph::{
        Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryItem, Pipe, SelectiveKind,
        block_as_gltf_data_with_pipe_length, stabilizer_as_gltf_data,
    };
    use glam::{IVec3, Vec3};

    fn with_render_signature_context<T>(
        graph: &BlockGraph,
        f: impl FnOnce(&RenderSignatureContext<'_>) -> T,
    ) -> T {
        let mut cache = RenderSignatureContextCache::default();
        rebuild_render_signature_context_cache(&mut cache, graph, 2.0);
        let context = RenderSignatureContext::from_cache(&cache);
        f(&context)
    }

    fn populate_signature_cache(graph: &BlockGraph, revision: u64) -> RenderSignatureContextCache {
        let mut cache = RenderSignatureContextCache::default();
        let sync_state = RenderSignatureContextSyncState {
            tab_id: EditorTabId::new(1),
            graph_revision: revision,
            pipe_length_bits: 2.0f32.to_bits(),
        };
        sync_render_signature_context_cache(&mut cache, sync_state, graph, 2.0, None);
        cache.applied = Some(sync_state);
        cache
    }

    fn assert_incremental_signatures_match_full_rebuild(
        cache: &mut RenderSignatureContextCache,
        graph: &BlockGraph,
        revision: u64,
    ) -> super::RenderSignatureContextCacheSyncMode {
        let sync_state = RenderSignatureContextSyncState {
            tab_id: EditorTabId::new(1),
            graph_revision: revision,
            pipe_length_bits: 2.0f32.to_bits(),
        };
        let mode = sync_render_signature_context_cache(cache, sync_state, graph, 2.0, None);
        cache.applied = Some(sync_state);

        let mut expected = RenderSignatureContextCache::default();
        rebuild_render_signature_context_cache(&mut expected, graph, 2.0);
        for block in graph.blocks() {
            for offset in block.connectable_offsets() {
                let endpoint = block.pos() + offset;
                assert_eq!(
                    cache.adjacency.degree(endpoint),
                    expected.adjacency.degree(endpoint),
                    "degree mismatch at {endpoint}"
                );
                assert_eq!(
                    cache.adjacency.endpoint_block(endpoint),
                    expected.adjacency.endpoint_block(endpoint),
                    "endpoint owner mismatch at {endpoint}"
                );
            }
        }
        for pipe in graph.pipes() {
            let (u, v) = pipe.endpoints();
            assert_eq!(
                cache.adjacency.has_pipe_between(u, v),
                expected.adjacency.has_pipe_between(u, v),
                "pipe mismatch between {u} and {v}"
            );
        }
        assert_eq!(cache.block_signatures, expected.block_signatures);
        assert_eq!(cache.pipe_signatures, expected.pipe_signatures);
        assert_eq!(cache.block_identities, expected.block_identities);
        assert_eq!(cache.pipe_identities, expected.pipe_identities);
        mode
    }

    #[test]
    fn stabilizer_view_keeps_light_theme_graph_boundaries_visible() {
        let alpha = stabilizer_context_alpha(true, ThemePreset::Light);

        assert!(
            alpha >= 0.35,
            "light stabilizer view should keep block graph boundary colors readable, got {alpha}"
        );
    }

    #[test]
    fn stabilizer_view_keeps_dark_theme_graph_context_visible() {
        let alpha = stabilizer_context_alpha(true, ThemePreset::GruvboxMaterial);

        assert!(
            alpha >= 0.18,
            "dark stabilizer view should keep enough graph context, got {alpha}"
        );
    }

    #[test]
    fn ccz_stabilizer_mesh_uses_displayed_branch_arms() {
        let mut graph = GalleryItem::CCZGateTeleport.build().flatten().unwrap();
        graph.set_shown_branch_arm("b0", false).unwrap();
        let graph = stabilizer_view_graph(&graph, false, 0).unwrap();
        let stabilizer = graph
            .stabilizers()
            .unwrap()
            .into_iter()
            .find(bloq_graph::StabilizerGenerator::is_measurement)
            .unwrap();

        stabilizer_as_gltf_data(&stabilizer, &graph, 2.0).unwrap();
    }

    #[test]
    fn editor_triangle_meshes_are_lit_and_double_sided() {
        let style = editor_triangle_material_style();

        assert!(!style.unlit);
        assert!(style.double_sided);
        assert!(style.cull_none);
    }

    #[test]
    fn graph_pick_proxy_bounds_generated_geometry() {
        let graph = BlockGraph::new();
        let data = block_as_gltf_data_with_pipe_length(
            &Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)),
            &graph,
            2.0,
        );

        let summary = analyze_gltf_geometry(&data);
        let proxy_positions = pick_box_positions(summary.pick_proxy.min, summary.pick_proxy.max);

        assert_eq!(proxy_positions.len(), 36);
        assert!(summary.pick_proxy.min.x < -0.5);
        assert!(summary.pick_proxy.max.x > 0.5);
    }

    #[test]
    fn source_geometry_mesh_hashes_match_position_buffer_hashes() {
        let triangles = vec![[
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        ]];
        let triangle_positions = triangles
            .iter()
            .flatten()
            .map(Vec3::to_array)
            .collect::<Vec<_>>();
        assert_eq!(
            hash_triangle_positions(&triangles),
            hash_positions(&triangle_positions, PrimitiveTopology::TriangleList)
        );

        let lines = vec![[Vec3::new(-1.0, 2.5, 3.0), Vec3::new(4.0, -5.0, 6.25)]];
        let line_positions = lines
            .iter()
            .flatten()
            .map(Vec3::to_array)
            .collect::<Vec<_>>();
        assert_eq!(
            hash_line_positions(&lines),
            hash_positions(&line_positions, PrimitiveTopology::LineList)
        );
    }

    #[test]
    fn cached_mesh_keys_match_source_geometry_hashes() {
        let graph = BlockGraph::new();
        let data = block_as_gltf_data_with_pipe_length(
            &Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)),
            &graph,
            2.0,
        );

        let keys = analyze_gltf_geometry(&data).mesh_keys;
        assert_eq!(
            keys.triangles,
            data.triangles
                .values()
                .map(|triangles| hash_triangle_positions(triangles))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            keys.lines,
            data.lines
                .values()
                .map(|lines| hash_line_positions(lines))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn render_signature_context_cache_refreshes_on_active_tab_change() {
        let mut first_graph = BlockGraph::new();
        first_graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Selective(SelectiveKind::XY),
        ));
        first_graph.add_block(Block::new(IVec3::Z, BlockKind::Port));
        first_graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        let mut first_state = GraphState {
            graph: first_graph,
            revision: 7,
            ..Default::default()
        };

        let mut second_graph = BlockGraph::new();
        second_graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Selective(SelectiveKind::XY),
        ));
        second_graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        second_graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS));
        let second_state = GraphState {
            graph: second_graph,
            revision: 7,
            ..Default::default()
        };

        let mut tabs = EditorTabs::default();
        tabs.active = EditorTabId::new(1);
        let mut cache = RenderSignatureContextCache::default();
        let _ = RenderSignatureContext::cached(&mut cache, &tabs, &first_state, 2.0);
        let first_applied = cache.applied;

        first_state.needs_rerender = true;
        let _ = RenderSignatureContext::cached(&mut cache, &tabs, &first_state, 2.0);
        assert_eq!(cache.applied, first_applied);

        tabs.active = EditorTabId::new(2);
        let _ = RenderSignatureContext::cached(&mut cache, &tabs, &second_state, 2.0);
        assert_eq!(
            cache.applied,
            Some(RenderSignatureContextSyncState {
                tab_id: EditorTabId::new(2),
                graph_revision: 7,
                pipe_length_bits: 2.0f32.to_bits(),
            })
        );
    }

    #[test]
    fn render_signature_context_cache_incrementally_tracks_local_pipe_addition() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::ZZX)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let mut cache = populate_signature_cache(&graph, 1);
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::YPLUS));

        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
    }

    #[test]
    fn render_signature_context_cache_incrementally_tracks_local_pipe_removal() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::ZZX)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::YPLUS));

        let mut cache = populate_signature_cache(&graph, 1);
        graph
            .remove_pipe(IVec3::ZERO, IVec3::Y)
            .expect("pipe should exist before removal");

        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
    }

    #[test]
    fn render_signature_context_cache_tracks_tall_cube_endpoint_pipe_identity() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ))
                .with_height("3d".parse().expect("valid height"))
                .expect("time scale should be valid"),
        );
        graph.add_block(Block::new(
            IVec3::new(1, 0, 2),
            BlockKind::Cube(CubeKind::ZXZ),
        ));

        let mut cache = populate_signature_cache(&graph, 1);
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 2), Direction::XPLUS));

        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
        assert!(
            cache
                .pipe_identities
                .contains_key(&PipeKey::new(IVec3::new(0, 0, 2), IVec3::new(1, 0, 2)))
        );
    }

    #[test]
    fn render_signature_context_cache_incrementally_tracks_block_removal() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::ZZX)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::YPLUS));

        let mut cache = populate_signature_cache(&graph, 1);
        graph
            .remove_block(IVec3::Y)
            .expect("block should exist before removal");

        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
    }

    #[test]
    fn render_signature_context_cache_tracks_port_color_changes() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        let mut cache = populate_signature_cache(&graph, 1);
        let old_signature = cache.block_signatures[&IVec3::ZERO];

        graph
            .set_port_color(IVec3::ZERO, [0xeb, 0x40, 0x34])
            .expect("Port accepts a color");

        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
        assert_ne!(cache.block_signatures[&IVec3::ZERO], old_signature);
    }

    #[test]
    fn render_signature_context_cache_syncs_selective_graphs_incrementally() {
        // Selective geometry does not depend on neighbour topology.
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Port));

        let mut cache = populate_signature_cache(&graph, 1);
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        assert_eq!(
            assert_incremental_signatures_match_full_rebuild(&mut cache, &graph, 2),
            super::RenderSignatureContextCacheSyncMode::Incremental
        );
    }

    #[test]
    fn port_tag_uses_clear_outward_side_and_width() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        let occupied = graph.occupied_positions().collect();
        assert_eq!(
            port_tag_direction(&graph, IVec3::Z, &occupied),
            Some(IVec3::Z)
        );

        graph.add_block(Block::new(IVec3::Z * 2, BlockKind::Cube(CubeKind::XZZ)));
        let occupied = graph.occupied_positions().collect();
        assert_eq!(
            port_tag_direction(&graph, IVec3::Z, &occupied),
            Some(IVec3::X)
        );
        assert!(port_tag_position(IVec3::X, "long_tag").x > 1.0);
    }

    #[test]
    fn port_tags_skip_outward_directions_beyond_coordinate_bounds() {
        for bound in [i32::MIN, i32::MAX] {
            let port = IVec3::splat(bound);
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(port, BlockKind::Port));
            let occupied = graph.occupied_positions().collect();

            let direction = port_tag_direction(&graph, port, &occupied).unwrap();
            let position = port.checked_add(direction).unwrap();
            assert!(!occupied.contains(&position));
        }
    }

    #[test]
    fn block_render_signature_tracks_local_pipe_context_only() {
        let block = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ));
        let mut graph = BlockGraph::new();
        graph.add_block(block.clone());
        graph.add_block(Block::new(IVec3::X, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(20, 0, 0), BlockKind::Port));

        let baseline = with_render_signature_context(&graph, |context| {
            block_render_geometry_signature(&block, context, 2.0)
        });
        graph.add_block(Block::new(IVec3::new(21, 0, 0), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(20, 0, 0), Direction::XPLUS));
        assert_eq!(
            baseline,
            with_render_signature_context(&graph, |context| {
                block_render_geometry_signature(&block, context, 2.0)
            })
        );

        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        assert_ne!(
            baseline,
            with_render_signature_context(&graph, |context| {
                block_render_geometry_signature(&block, context, 2.0)
            })
        );
    }

    #[test]
    fn pipe_render_signature_tracks_endpoint_context() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let pipe = graph
            .get_pipe(IVec3::ZERO, IVec3::X)
            .expect("pipe exists")
            .clone();

        let baseline = with_render_signature_context(&graph, |context| {
            pipe_render_geometry_signature(IVec3::ZERO, IVec3::X, &pipe, context, 2.0)
        });
        graph.add_block(Block::new(IVec3::new(20, 0, 0), BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(21, 0, 0), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(20, 0, 0), Direction::XPLUS));
        assert_eq!(
            baseline,
            with_render_signature_context(&graph, |context| {
                pipe_render_geometry_signature(IVec3::ZERO, IVec3::X, &pipe, context, 2.0)
            })
        );

        graph.add_block(Block::new(IVec3::Y, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::YPLUS));
        assert_ne!(
            baseline,
            with_render_signature_context(&graph, |context| {
                pipe_render_geometry_signature(IVec3::ZERO, IVec3::X, &pipe, context, 2.0)
            })
        );
    }

    #[test]
    fn pipe_render_signature_is_independent_of_endpoint_order() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let pipe = graph
            .get_pipe(IVec3::ZERO, IVec3::X)
            .expect("pipe exists")
            .clone();
        with_render_signature_context(&graph, |context| {
            assert_eq!(
                pipe_render_geometry_signature(IVec3::ZERO, IVec3::X, &pipe, context, 2.0),
                pipe_render_geometry_signature(IVec3::X, IVec3::ZERO, &pipe, context, 2.0)
            );
        });
    }

    #[test]
    fn pipe_geometry_template_key_reuses_position_independent_shapes() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::new(10, 0, 0),
            BlockKind::Cube(CubeKind::XZZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(11, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(10, 0, 0), Direction::XPLUS));

        let first_pipe = graph
            .get_pipe(IVec3::ZERO, IVec3::X)
            .expect("first pipe exists");
        let second_pipe = graph
            .get_pipe(IVec3::new(10, 0, 0), IVec3::new(11, 0, 0))
            .expect("second pipe exists");

        with_render_signature_context(&graph, |context| {
            assert_eq!(
                super::pipe_geometry_template_key(IVec3::ZERO, IVec3::X, first_pipe, context, 2.0),
                super::pipe_geometry_template_key(
                    IVec3::new(10, 0, 0),
                    IVec3::new(11, 0, 0),
                    second_pipe,
                    context,
                    2.0
                )
            );
        });
    }

    #[test]
    fn pipe_geometry_template_key_tracks_hadamard_and_scaled_height() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("time scale should be valid"),
        );
        graph.add_block(
            Block::new(IVec3::new(1, 0, 1), BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("time scale should be valid"),
        );
        graph.add_block(Block::new(
            IVec3::new(10, 0, 0),
            BlockKind::Cube(CubeKind::XZZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(11, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 1), Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(10, 0, 0), Direction::XPLUS).with_hadamard());

        let scaled_pipe = graph
            .get_pipe(IVec3::new(0, 0, 1), IVec3::new(1, 0, 1))
            .expect("scaled pipe exists");
        let hadamard_pipe = graph
            .get_pipe(IVec3::new(10, 0, 0), IVec3::new(11, 0, 0))
            .expect("hadamard pipe exists");
        let ordinary = Pipe::new(IVec3::new(10, 0, 0), Direction::XPLUS);

        with_render_signature_context(&graph, |context| {
            assert_ne!(
                super::pipe_geometry_template_key(
                    IVec3::new(0, 0, 1),
                    IVec3::new(1, 0, 1),
                    scaled_pipe,
                    context,
                    2.0
                ),
                super::pipe_geometry_template_key(
                    IVec3::new(10, 0, 0),
                    IVec3::new(11, 0, 0),
                    &ordinary,
                    context,
                    2.0
                )
            );
            assert_ne!(
                super::pipe_geometry_template_key(
                    IVec3::new(10, 0, 0),
                    IVec3::new(11, 0, 0),
                    hadamard_pipe,
                    context,
                    2.0
                ),
                super::pipe_geometry_template_key(
                    IVec3::new(10, 0, 0),
                    IVec3::new(11, 0, 0),
                    &ordinary,
                    context,
                    2.0
                )
            );
        });
    }

    #[test]
    fn render_context_pipe_basis_matches_graph_inference() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::ZZX)));
        graph.add_block(Block::new(IVec3::NEG_Y, BlockKind::Cube(CubeKind::ZXX)));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS).with_hadamard());
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::YPLUS));
        graph.add_pipe(Pipe::new(IVec3::NEG_Y, Direction::YPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        with_render_signature_context(&graph, |context| {
            for pipe in graph.pipes() {
                assert_eq!(
                    super::infer_pipe_basis_from_render_context(pipe, context),
                    graph.infer_pipe_basis(pipe),
                    "basis mismatch for {pipe}"
                );
            }
        });
    }

    #[test]
    fn cached_block_geometry_reuses_position_independent_templates() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(
            IVec3::new(10, 0, 0),
            BlockKind::Cube(CubeKind::XZZ),
        ));
        let first_block = graph.get_block(IVec3::ZERO).expect("first block exists");
        let second_block = graph
            .get_block(IVec3::new(10, 0, 0))
            .expect("second block exists");
        let mut cache = RenderGeometryCache::default();
        let first_signature = 101;
        let second_signature = 202;

        with_render_signature_context(&graph, |context| {
            let first_data = {
                let cached = super::cached_block_geometry(
                    &mut cache,
                    first_block,
                    context,
                    &graph,
                    2.0,
                    first_signature,
                );
                &cached.data as *const _
            };
            let second_data = {
                let cached = super::cached_block_geometry(
                    &mut cache,
                    second_block,
                    context,
                    &graph,
                    2.0,
                    second_signature,
                );
                &cached.data as *const _
            };

            assert_eq!(first_data, second_data);
            assert_eq!(cache.blocks.len(), 2);
            assert_eq!(cache.block_templates.len(), 1);

            let first_again = {
                let cached = super::cached_block_geometry(
                    &mut cache,
                    first_block,
                    context,
                    &graph,
                    2.0,
                    first_signature,
                );
                &cached.data as *const _
            };

            assert_eq!(first_again, first_data);
        });
    }

    #[test]
    fn cached_pipe_geometry_reuses_shared_template_data() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::new(10, 0, 0),
            BlockKind::Cube(CubeKind::XZZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(11, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(10, 0, 0), Direction::XPLUS));

        let first_pipe = graph
            .get_pipe(IVec3::ZERO, IVec3::X)
            .expect("first pipe exists");
        let second_pipe = graph
            .get_pipe(IVec3::new(10, 0, 0), IVec3::new(11, 0, 0))
            .expect("second pipe exists");
        let mut cache = RenderGeometryCache::default();
        let first_signature = 11;
        let second_signature = 22;
        with_render_signature_context(&graph, |context| {
            let first_data = {
                let cached = super::cached_pipe_geometry(
                    &mut cache,
                    PipeKey::new(IVec3::ZERO, IVec3::X),
                    first_pipe,
                    context,
                    &graph,
                    2.0,
                    first_signature,
                );
                &cached.data as *const _
            };
            let second_data = {
                let cached = super::cached_pipe_geometry(
                    &mut cache,
                    PipeKey::new(IVec3::new(10, 0, 0), IVec3::new(11, 0, 0)),
                    second_pipe,
                    context,
                    &graph,
                    2.0,
                    second_signature,
                );
                &cached.data as *const _
            };

            assert_eq!(first_data, second_data);
            assert_eq!(cache.pipes.len(), 2);
            assert_eq!(cache.pipe_templates.len(), 1);

            let first_again = {
                let cached = super::cached_pipe_geometry(
                    &mut cache,
                    PipeKey::new(IVec3::ZERO, IVec3::X),
                    first_pipe,
                    context,
                    &graph,
                    2.0,
                    first_signature,
                );
                &cached.data as *const _
            };

            assert_eq!(first_again, first_data);
        });
    }

    #[test]
    fn render_signature_context_tracks_adjacent_pipe_neighbors() {
        let mut context = GraphAdjacencySnapshot::default();
        context.insert_pipe_neighbor(IVec3::ZERO, IVec3::X);

        assert_eq!(adjacent_direction_bit(IVec3::X), Some(1 << 1));
        assert!(context.has_pipe_between(IVec3::ZERO, IVec3::X));
        assert!(context.has_pipe_between(IVec3::X, IVec3::ZERO));
        assert!(!context.has_pipe_between(IVec3::ZERO, IVec3::Y));
        assert!(!context.has_pipe_between(IVec3::ZERO, IVec3::new(2, 0, 0)));
    }

    #[test]
    fn pipe_preview_transform_spans_graph_x_axis() {
        let transform = pipe_preview_transform(IVec3::ZERO, IVec3::X, 2.0);

        assert_eq!(transform.translation, Vec3::new(1.5, 0.0, 0.0));
        assert_eq!(transform.scale, Vec3::new(2.0, 0.24, 0.24));
    }

    #[test]
    fn pipe_preview_transform_spans_graph_z_as_world_y_axis() {
        let transform = pipe_preview_transform(IVec3::ZERO, IVec3::Z, 2.0);

        assert_eq!(transform.translation, Vec3::new(0.0, 1.5, 0.0));
        assert_eq!(transform.scale, Vec3::new(0.24, 2.0, 0.24));
    }

    #[test]
    fn graph_pickability_ignores_graph_after_pipe_start_is_selected() {
        let editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(IVec3::ZERO),
            ..Default::default()
        };

        assert_eq!(graph_pickability(&editor_state), Pickable::IGNORE);
    }

    #[test]
    fn preview_sync_state_tracks_only_preview_inputs() {
        let graph_state = GraphState::default();
        let editor_state = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(IVec3::ZERO),
            hovered_grid_pos: Some(IVec3::X),
            ..Default::default()
        };

        let state = PlacementPreviewSyncState::from_inputs(&editor_state, &graph_state, false);
        let mut unrelated = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Pipe,
            pipe_start: Some(IVec3::ZERO),
            hovered_grid_pos: Some(IVec3::X),
            ..Default::default()
        };
        unrelated.show_axis = !unrelated.show_axis;
        unrelated.show_side_panel = !unrelated.show_side_panel;

        assert_eq!(
            state,
            PlacementPreviewSyncState::from_inputs(&unrelated, &graph_state, false)
        );

        let changed_modifier =
            PlacementPreviewSyncState::from_inputs(&editor_state, &graph_state, true);
        assert_ne!(state, changed_modifier);
    }

    #[test]
    fn preview_sync_state_invalidates_when_hovering_an_element() {
        let graph_state = GraphState::default();
        let base = EditorState {
            mode: EditorMode::Edit,
            placement_tool: PlacementTool::Block,
            hovered_grid_pos: Some(IVec3::X),
            ..Default::default()
        };
        let mut hovering = base.clone();
        hovering.hovered_element = Some(crate::components::GraphElement::Block(IVec3::X));

        // Hovering an existing element hides the placement preview, so the cache
        // key must change even though the hovered grid cell is identical.
        assert_ne!(
            PlacementPreviewSyncState::from_inputs(&base, &graph_state, false),
            PlacementPreviewSyncState::from_inputs(&hovering, &graph_state, false),
        );
    }

    #[test]
    fn nonvisual_edit_removes_stabilizer_mesh_and_restores_graph_opacity() {
        use super::*;
        let graph_state = GraphState {
            graph: GalleryItem::BellState.build().flatten().unwrap(),
            needs_rerender: true,
            ..Default::default()
        };
        let editor = EditorState {
            stabilizers: graph_state.graph.stabilizers().unwrap(),
            show_stabilizers: false,
            ..Default::default()
        };
        let mut app = App::new();
        app.insert_resource(graph_state)
            .insert_resource(editor)
            .init_resource::<EditorTabs>()
            .init_resource::<BloqViewerState>()
            .init_resource::<Notifications>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GraphRenderState>()
            .init_resource::<RenderAssetCache>()
            .add_systems(
                Update,
                block_graph_visual_system.run_if(|graph: Res<GraphState>| graph.needs_rerender),
            );
        app.world_mut().spawn((GraphRoot, Transform::default()));
        app.update();
        let alphas = |world: &World| {
            let render = world.resource::<GraphRenderState>();
            let materials = world.resource::<Assets<StandardMaterial>>();
            let mut values: Vec<_> = render
                .blocks
                .values()
                .flat_map(|rendered| &rendered.mesh_parts.triangles)
                .map(|&entity| {
                    materials
                        .get(
                            &world
                                .get::<MeshMaterial3d<StandardMaterial>>(entity)
                                .unwrap()
                                .0,
                        )
                        .unwrap()
                        .base_color
                        .alpha()
                })
                .collect();
            values.sort_by(f32::total_cmp);
            values
        };
        let ordinary_alphas = alphas(app.world());
        assert!(!ordinary_alphas.is_empty());
        app.world_mut()
            .resource_mut::<EditorState>()
            .show_stabilizers = true;
        app.world_mut().resource_mut::<GraphState>().needs_rerender = true;
        app.update();
        assert_ne!(alphas(app.world()), ordinary_alphas);
        let overlay = app
            .world()
            .resource::<GraphRenderState>()
            .stabilizer
            .as_ref()
            .unwrap()
            .entity;
        app.world_mut()
            .resource_scope(|world, mut editor: Mut<EditorState>| {
                let mut graph = world.resource_mut::<GraphState>();
                let pos = graph.graph.blocks().next().unwrap().pos();
                graph.graph.set_block_tag(pos, "metadata").unwrap();
                graph.commit_with_rerender(false);
                editor.sync_after_graph_edit(&mut graph);
            });
        app.update();
        assert!(app.world().get_entity(overlay).is_err());
        let render = app.world().resource::<GraphRenderState>();
        assert!(render.stabilizer.is_none());
        assert!(!render.mesh_sync.unwrap().show_stabilizers);
        assert_eq!(alphas(app.world()), ordinary_alphas);
    }

    #[test]
    fn extremal_import_coordinates_render_and_offer_only_in_range_hints() {
        for bound in [i32::MIN, i32::MAX] {
            let pos = IVec3::new(bound, 0, 0);
            let mut source = BlockGraph::new();
            source.add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
            let direction = if bound == i32::MAX {
                Direction::XMINUS
            } else {
                Direction::XPLUS
            };
            source.add_block(Block::new(
                pos + direction.to_ivec3(),
                BlockKind::Cube(CubeKind::ZXZ),
            ));
            source.add_pipe(Pipe::new(pos, direction));
            let graph = bloq_graph::parse_blog_to_graph(&source.to_blog_text()).unwrap();
            let cache = populate_signature_cache(&graph, 1);
            let context = RenderSignatureContext::from_cache(&cache);
            let mut geometry = RenderGeometryCache::default();
            for block in graph.blocks() {
                super::cached_block_geometry(&mut geometry, block, &context, &graph, 2.0, 0);
            }
            for pipe in graph.pipes() {
                let (a, b) = pipe.endpoints();
                super::cached_pipe_geometry(
                    &mut geometry,
                    PipeKey::new(a, b),
                    pipe,
                    &context,
                    &graph,
                    2.0,
                    0,
                );
            }
            let targets = pipe_preview_hint_targets(&graph, &cache.adjacency, pos, false);
            let blocked = if bound == i32::MAX { 0 } else { 1 };
            assert_eq!(targets.pipe[blocked], None);
            let data =
                block_as_gltf_data_with_pipe_length(graph.get_block(pos).unwrap(), &graph, 2.0);
            assert!(!data.triangles.is_empty());
        }
    }

    #[test]
    fn pipe_preview_refreshes_when_switching_equal_revision_tabs() {
        use super::*;
        let mut app = App::new();
        app.init_resource::<GraphState>()
            .init_resource::<EditorTabs>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<RenderAssetCache>()
            .insert_resource(EditorState {
                mode: EditorMode::Edit,
                placement_tool: PlacementTool::Pipe,
                pipe_start: Some(IVec3::ZERO),
                ..Default::default()
            })
            .add_systems(Update, add_block_preview_system);
        app.world_mut()
            .spawn((PreviewMesh, Transform::default(), Visibility::Hidden));
        app.world_mut()
            .spawn((WalkingPreviewMesh, Transform::default(), Visibility::Hidden));
        let endpoint = app
            .world_mut()
            .spawn((
                PreviewEndpointMesh {
                    endpoint: PreviewEndpoint::PipeHint(0),
                    source_pos: None,
                    target_pos: None,
                },
                Transform::default(),
                Visibility::Hidden,
                Pickable::IGNORE,
                MeshMaterial3d::<StandardMaterial>::default(),
            ))
            .id();
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        app.world_mut().resource_mut::<GraphState>().graph = graph.clone();
        app.update();
        assert_eq!(
            app.world()
                .get::<PreviewEndpointMesh>(endpoint)
                .unwrap()
                .target_pos,
            Some(IVec3::X)
        );
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        app.world_mut().resource_mut::<GraphState>().graph = graph;
        let mut tabs = app.world_mut().resource_mut::<EditorTabs>();
        let second = tabs.add_empty_tab(&EditorState::default());
        tabs.active = second;
        app.update();
        assert_eq!(
            app.world()
                .get::<PreviewEndpointMesh>(endpoint)
                .unwrap()
                .target_pos,
            None
        );
    }

    #[test]
    fn pipe_preview_hint_targets_collect_expected_pipe_hint_slots() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));

        let snapshot = GraphAdjacencySnapshot::from_graph(&graph);
        let targets = pipe_preview_hint_targets(&graph, &snapshot, IVec3::ZERO, false);

        assert_eq!(
            targets.pipe.len(),
            crate::systems::input::PIPE_HINT_OFFSETS.len()
        );
        assert_eq!(
            targets.walking.len(),
            crate::systems::input::WALKING_MOVEMENTS.len()
        );
        assert_eq!(targets.pipe[0], Some((IVec3::ZERO, IVec3::X)));
    }

    #[test]
    fn preview_setters_skip_unchanged_components() {
        let mut visibility = Visibility::Hidden;
        assert!(!set_visibility_if_changed(
            &mut visibility,
            Visibility::Hidden
        ));
        assert!(set_visibility_if_changed(
            &mut visibility,
            Visibility::Inherited
        ));

        let mut pickable = Pickable::IGNORE;
        assert!(!set_pickable_if_changed(&mut pickable, Pickable::IGNORE));
        assert!(set_pickable_if_changed(&mut pickable, Pickable::default()));
    }
}
