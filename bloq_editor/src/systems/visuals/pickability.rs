//! Pointer pickability of rendered elements and the placement/endpoint
//! preview meshes and hint markers.

use super::*;

/// Updates whether rendered blocks and pipes accept picking to match the
/// current mode and tool, skipping the sweep when nothing relevant changed.
pub(crate) fn sync_graph_pickability_system(
    mut commands: Commands,
    editor_state: Res<EditorState>,
    render_state: Res<GraphRenderState>,
    mut sync_state: Local<PickabilitySyncState>,
    mut pickables: Query<&mut Pickable>,
) {
    let state = PickabilityState {
        mode: editor_state.mode,
        placement_tool: editor_state.placement_tool,
        pipe_start_active: editor_state.pipe_start.is_some(),
        render_revision: render_state.revision,
    };
    if sync_state.applied == Some(state) {
        return;
    }

    let pickable = graph_pickability(&editor_state);
    for rendered in render_state
        .blocks
        .values()
        .chain(render_state.pipes.values())
    {
        if let Ok(mut entity_pickable) = pickables.get_mut(rendered.entity) {
            set_pickable_if_changed(&mut entity_pickable, pickable);
        } else {
            commands.entity(rendered.entity).insert(pickable);
        }
    }
    sync_state.applied = Some(state);
}

pub(crate) fn graph_pickability(editor_state: &EditorState) -> Pickable {
    if editor_state.is_pipe_tool_active() && editor_state.pipe_start.is_some() {
        Pickable::IGNORE
    } else {
        Pickable::default()
    }
}

const PIPE_PREVIEW_CROSS_SECTION: f32 = 0.24;

pub(super) fn pipe_preview_transform(start: IVec3, end: IVec3, pipe_length: f32) -> Transform {
    let start_world = graph_to_world(start.as_vec3(), pipe_length);
    let end_world = graph_to_world(end.as_vec3(), pipe_length);
    let center = start_world + (end_world - start_world) * 0.5;
    let delta = end_world - start_world;
    let mut scale = Vec3::splat(PIPE_PREVIEW_CROSS_SECTION);
    if delta.x.abs() >= delta.y.abs() && delta.x.abs() >= delta.z.abs() {
        scale.x = pipe_length;
    } else if delta.y.abs() >= delta.z.abs() {
        scale.y = pipe_length;
    } else {
        scale.z = pipe_length;
    }
    Transform {
        translation: center,
        scale,
        ..default()
    }
}

fn preview_endpoint_material(
    editor_state: &EditorState,
    endpoint_pos: IVec3,
) -> Handle<StandardMaterial> {
    if editor_state.hovered_preview_endpoint == Some(endpoint_pos) {
        editor_state.preview_endpoint_hover_material.clone()
    } else {
        editor_state.preview_endpoint_material.clone()
    }
}

pub(super) fn set_visibility_if_changed(visibility: &mut Visibility, next: Visibility) -> bool {
    if *visibility == next {
        return false;
    }
    *visibility = next;
    true
}

pub(super) fn set_pickable_if_changed(pickable: &mut Pickable, next: Pickable) -> bool {
    if pickable.should_block_lower == next.should_block_lower
        && pickable.is_hoverable == next.is_hoverable
    {
        return false;
    }
    *pickable = next;
    true
}

fn set_material_if_changed(
    material: &mut MeshMaterial3d<StandardMaterial>,
    next: Handle<StandardMaterial>,
) {
    if material.0 != next {
        material.0 = next;
    }
}

fn set_transform_if_changed(transform: &mut Transform, next: Transform) {
    if *transform != next {
        *transform = next;
    }
}

type WalkingPreviewQuery<'w, 's> = Single<
    'w,
    's,
    (Entity, &'static mut Transform, &'static mut Visibility),
    (With<WalkingPreviewMesh>, Without<PreviewMesh>),
>;

type PreviewEndpointQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Transform,
        &'static mut Visibility,
        &'static mut Pickable,
        &'static mut PreviewEndpointMesh,
        &'static mut MeshMaterial3d<StandardMaterial>,
    ),
    (Without<PreviewMesh>, Without<WalkingPreviewMesh>),
>;

pub(super) fn pipe_preview_hint_targets(
    graph: &BlockGraph,
    snapshot: &GraphAdjacencySnapshot,
    start: IVec3,
    force_hadamard: bool,
) -> PipePreviewHintTargets {
    debug_assert_eq!(PIPE_PREVIEW_HINT_COUNT, PIPE_HINT_OFFSETS.len());
    debug_assert_eq!(WALKING_PREVIEW_HINT_COUNT, WALKING_MOVEMENTS.len());
    PipePreviewHintTargets {
        pipe: std::array::from_fn(|index| {
            pipe_hint_target_with_snapshot(graph, snapshot, start, index, force_hadamard)
        }),
        walking_pipe: std::array::from_fn(|index| {
            walking_pipe_hint_target(graph, start, index, force_hadamard)
                .filter(|(source, _)| *source != start)
        }),
        patch_rotation_pipe: std::array::from_fn(|index| {
            patch_rotation_pipe_hint_target(graph, start, index, force_hadamard)
                .filter(|(source, _)| *source != start)
        }),
        tall_cube_pipe: std::array::from_fn(|index| {
            tall_cube_pipe_hint_target(graph, start, index, force_hadamard)
                .filter(|(source, _)| *source != start)
        }),
        walking: std::array::from_fn(|index| {
            WALKING_MOVEMENTS.get(index).and_then(|movement| {
                let end = start.checked_add(IVec3::new(movement.x, movement.y, 1))?;
                walking_port_promotion_candidate(graph, start, end)
                    .is_some()
                    .then_some((start, end))
            })
        }),
    }
}

/// Positions and shows the placement preview mesh and the pipe/walking endpoint
/// hint markers for the current tool and hovered cell, or hides them when no
/// placement is possible.
pub(super) fn add_block_preview_system(
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    tabs: Res<crate::resources::EditorTabs>,
    keys: Res<ButtonInput<KeyCode>>,
    mut preview_cache: Local<PlacementPreviewCache>,
    mut pipe_hint_cache: Local<PipePreviewHintCache>,
    preview_query: Single<(&mut Transform, &mut Visibility), With<PreviewMesh>>,
    walking_preview_query: WalkingPreviewQuery,
    mut endpoint_query: PreviewEndpointQuery,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_cache: ResMut<RenderAssetCache>,
) {
    let _span = info_span!("editor.add_block_preview_system").entered();
    if preview_cache.tab_id != Some(tabs.active) {
        *preview_cache = PlacementPreviewCache {
            tab_id: Some(tabs.active),
            ..Default::default()
        };
        *pipe_hint_cache = PipePreviewHintCache::default();
    }
    let preview_state = PlacementPreviewSyncState::from_inputs(
        &editor_state,
        &graph_state,
        is_hadamard_pipe_modifier_pressed(&keys),
    );
    if preview_cache.applied == Some(preview_state) {
        return;
    }
    preview_cache.applied = Some(preview_state);

    let (mut transform, mut vis) = preview_query.into_inner();
    let (walking_preview_entity, mut walking_preview_transform, mut walking_preview_visibility) =
        walking_preview_query.into_inner();
    if set_visibility_if_changed(&mut walking_preview_visibility, Visibility::Hidden) {
        commands
            .entity(walking_preview_entity)
            .despawn_related::<Children>();
    }

    for (_, mut endpoint_visibility, mut pickable, mut endpoint, mut endpoint_material) in
        endpoint_query.iter_mut()
    {
        set_visibility_if_changed(&mut endpoint_visibility, Visibility::Hidden);
        set_pickable_if_changed(&mut pickable, Pickable::IGNORE);
        endpoint.source_pos = None;
        endpoint.target_pos = None;
        set_material_if_changed(
            &mut endpoint_material,
            editor_state.preview_endpoint_material.clone(),
        );
    }

    if editor_state.is_block_tool_active()
        && editor_state.hovered_element.is_none()
        && (editor_state.block_kind.is_walking() || editor_state.block_kind.is_patch_rotation())
        && let Some(start) = editor_state.walking_start
    {
        for (
            mut endpoint_transform,
            mut endpoint_visibility,
            mut pickable,
            mut endpoint,
            mut endpoint_material,
        ) in endpoint_query.iter_mut()
        {
            let endpoint_pos = match endpoint.endpoint {
                PreviewEndpoint::Start => Some(start),
                PreviewEndpoint::End => editor_state.hovered_grid_pos.filter(|pos| {
                    moving_block_candidate_kind(&editor_state, &graph_state.graph, *pos).is_some()
                }),
                PreviewEndpoint::PipeHint(_) => None,
                PreviewEndpoint::WalkingHint(index) => {
                    moving_block_movements(editor_state.block_kind)
                        .get(index)
                        .and_then(|movement| {
                            start.checked_add(IVec3::new(movement.x, movement.y, 1))
                        })
                        .filter(|pos| {
                            moving_block_candidate_kind(&editor_state, &graph_state.graph, *pos)
                                .is_some()
                        })
                }
            };
            let Some(endpoint_pos) = endpoint_pos else {
                continue;
            };
            set_transform_if_changed(
                &mut endpoint_transform,
                Transform::from_translation(graph_to_world(
                    endpoint_pos.as_vec3(),
                    editor_state.pipe_length,
                )),
            );
            set_visibility_if_changed(&mut endpoint_visibility, Visibility::Inherited);
            set_pickable_if_changed(&mut pickable, Pickable::default());
            endpoint.target_pos = Some(endpoint_pos);
            set_material_if_changed(
                &mut endpoint_material,
                preview_endpoint_material(&editor_state, endpoint_pos),
            );
        }
        set_visibility_if_changed(&mut vis, Visibility::Hidden);
        return;
    }

    if editor_state.is_block_tool_active()
        && editor_state.hovered_element.is_none()
        && let Some(pos) = editor_state.hovered_grid_pos
    {
        let can_place = if (editor_state.block_kind.is_walking()
            || editor_state.block_kind.is_patch_rotation())
            && editor_state.walking_start.is_none()
        {
            moving_block_start_has_candidate(&editor_state, &graph_state.graph, pos)
        } else {
            graph_state
                .graph
                .can_place_block(&Block::new(pos, editor_state.block_kind))
                .is_ok()
        };
        if !can_place {
            set_visibility_if_changed(&mut vis, Visibility::Hidden);
            return;
        }
        set_visibility_if_changed(&mut vis, Visibility::Inherited);
        set_transform_if_changed(
            &mut transform,
            Transform::from_translation(graph_to_world(pos.as_vec3(), editor_state.pipe_length)),
        );
        return;
    }

    if editor_state.is_pipe_tool_active()
        && let Some(start) = editor_state.pipe_start
    {
        let force_hadamard = is_hadamard_pipe_modifier_pressed(&keys);
        if pipe_hint_cache.snapshot_revision != Some(graph_state.revision) {
            pipe_hint_cache.snapshot = GraphAdjacencySnapshot::from_graph(&graph_state.graph);
            pipe_hint_cache.snapshot_revision = Some(graph_state.revision);
        }
        let hint_state = PipePreviewHintSyncState::new(graph_state.revision, start, force_hadamard);
        if pipe_hint_cache.applied != Some(hint_state) {
            pipe_hint_cache.targets = pipe_preview_hint_targets(
                &graph_state.graph,
                &pipe_hint_cache.snapshot,
                start,
                force_hadamard,
            );
            pipe_hint_cache.applied = Some(hint_state);
        }
        let pipe_hint_targets = &pipe_hint_cache.targets.pipe;
        let walking_pipe_hint_targets = &pipe_hint_cache.targets.walking_pipe;
        let patch_rotation_pipe_hint_targets = &pipe_hint_cache.targets.patch_rotation_pipe;
        let tall_cube_pipe_hint_targets = &pipe_hint_cache.targets.tall_cube_pipe;
        let walking_hint_targets = &pipe_hint_cache.targets.walking;
        let hovered_target = editor_state
            .hovered_preview_endpoint
            .or(editor_state.hovered_grid_pos);
        let preview_start = editor_state.hovered_preview_source.unwrap_or(start);
        let pipe_preview_end = hovered_target.filter(|end| *end != start).filter(|end| {
            let Some(offset) = end.checked_sub(preview_start) else {
                return false;
            };
            pipe_candidate_at_with_snapshot(
                &graph_state.graph,
                &pipe_hint_cache.snapshot,
                preview_start,
                offset,
                force_hadamard,
            )
            .is_some()
        });
        let hovered_walking = hovered_target.and_then(|end| {
            walking_port_promotion_candidate(&graph_state.graph, start, end).map(|kind| (end, kind))
        });
        let hovered_walking_end = hovered_walking.map(|(end, _)| end);
        let hovered_walking_kind = hovered_walking.map(|(_, kind)| kind);
        let has_walking_hint = walking_hint_targets.iter().any(Option::is_some);
        let has_pipe_hint = (0..PIPE_HINT_OFFSETS.len()).any(|index| {
            pipe_hint_targets[index].is_some()
                || walking_pipe_hint_targets[index].is_some()
                || patch_rotation_pipe_hint_targets[index].is_some()
                || tall_cube_pipe_hint_targets[index].is_some()
        });

        if pipe_preview_end.is_some() || has_pipe_hint || has_walking_hint {
            if let Some(end) = pipe_preview_end {
                set_visibility_if_changed(&mut vis, Visibility::Inherited);
                set_transform_if_changed(
                    &mut transform,
                    pipe_preview_transform(preview_start, end, editor_state.pipe_length),
                );
            } else {
                set_visibility_if_changed(&mut vis, Visibility::Hidden);
            }
            if let Some(walking_kind) = hovered_walking_kind {
                set_visibility_if_changed(&mut walking_preview_visibility, Visibility::Inherited);
                set_transform_if_changed(
                    &mut walking_preview_transform,
                    Transform::from_translation(graph_to_world(
                        start.as_vec3(),
                        editor_state.pipe_length,
                    )),
                );
                let data = block_as_gltf_data_with_pipe_length(
                    &Block::new(start, BlockKind::Walking(walking_kind)),
                    &graph_state.graph,
                    editor_state.pipe_length,
                )
                .map_points(|p| graph_to_world(p, 0.0));
                spawn_gltf_data_parts(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut render_cache,
                    walking_preview_entity,
                    data,
                    None,
                    0.6,
                    None,
                    Some(editor_state.highlight_material.clone()),
                    None,
                    Pickable::IGNORE,
                );
            }
            for (
                mut endpoint_transform,
                mut endpoint_visibility,
                mut pickable,
                mut endpoint,
                mut endpoint_material,
            ) in endpoint_query.iter_mut()
            {
                let candidate = match endpoint.endpoint {
                    PreviewEndpoint::Start => Some((start, start)),
                    PreviewEndpoint::End => pipe_preview_end
                        .map(|pos| (preview_start, pos))
                        .or_else(|| hovered_walking_end.map(|pos| (start, pos))),
                    PreviewEndpoint::PipeHint(index) => pipe_hint_targets
                        .get(index)
                        .copied()
                        .flatten()
                        .or_else(|| walking_pipe_hint_targets.get(index).copied().flatten())
                        .or_else(|| {
                            patch_rotation_pipe_hint_targets
                                .get(index)
                                .copied()
                                .flatten()
                        })
                        .or_else(|| tall_cube_pipe_hint_targets.get(index).copied().flatten()),
                    PreviewEndpoint::WalkingHint(index) => {
                        walking_hint_targets.get(index).copied().flatten()
                    }
                };
                let Some((source_pos, endpoint_pos)) = candidate else {
                    continue;
                };
                set_transform_if_changed(
                    &mut endpoint_transform,
                    Transform::from_translation(graph_to_world(
                        endpoint_pos.as_vec3(),
                        editor_state.pipe_length,
                    )),
                );
                set_visibility_if_changed(&mut endpoint_visibility, Visibility::Inherited);
                set_pickable_if_changed(&mut pickable, Pickable::default());
                endpoint.source_pos = Some(source_pos);
                endpoint.target_pos = Some(endpoint_pos);
                set_material_if_changed(
                    &mut endpoint_material,
                    preview_endpoint_material(&editor_state, endpoint_pos),
                );
            }
            return;
        }
    }

    set_visibility_if_changed(&mut vis, Visibility::Hidden);
}

/// Inputs that determine which rendered elements accept pointer picking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PickabilityState {
    mode: EditorMode,
    placement_tool: PlacementTool,
    pipe_start_active: bool,
    render_revision: u64,
}

/// The last [`PickabilityState`] applied, so pickability is only re-synced on
/// change.
#[derive(Default)]
pub(crate) struct PickabilitySyncState {
    applied: Option<PickabilityState>,
}

/// Inputs the placement preview mesh depends on; a match skips its rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacementPreviewSyncState {
    pub(crate) mode: EditorMode,
    pub(crate) placement_tool: PlacementTool,
    pub(crate) block_kind: BlockKind,
    pub(crate) pipe_length_bits: u32,
    pub(crate) hovered_grid_pos: Option<IVec3>,
    pub(crate) hovering_element: bool,
    pub(crate) hovered_preview_source: Option<IVec3>,
    pub(crate) hovered_preview_endpoint: Option<IVec3>,
    pub(crate) pipe_start: Option<IVec3>,
    pub(crate) walking_start: Option<IVec3>,
    pub(crate) graph_revision: u64,
    pub(crate) hadamard_modifier: bool,
}

impl PlacementPreviewSyncState {
    /// Snapshots the preview-relevant fields of the editor and graph state.
    pub(crate) fn from_inputs(
        editor_state: &EditorState,
        graph_state: &GraphState,
        hadamard_modifier: bool,
    ) -> Self {
        Self {
            mode: editor_state.mode,
            placement_tool: editor_state.placement_tool,
            block_kind: editor_state.block_kind,
            pipe_length_bits: editor_state.pipe_length.to_bits(),
            hovered_grid_pos: editor_state.hovered_grid_pos,
            hovering_element: editor_state.hovered_element.is_some(),
            hovered_preview_source: editor_state.hovered_preview_source,
            hovered_preview_endpoint: editor_state.hovered_preview_endpoint,
            pipe_start: editor_state.pipe_start,
            walking_start: editor_state.walking_start,
            graph_revision: graph_state.revision,
            hadamard_modifier,
        }
    }
}

/// The last [`PlacementPreviewSyncState`] applied to the preview mesh.
#[derive(Default)]
pub(super) struct PlacementPreviewCache {
    tab_id: Option<crate::resources::EditorTabId>,
    applied: Option<PlacementPreviewSyncState>,
}

/// Inputs the pipe/walking endpoint hint markers depend on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PipePreviewHintSyncState {
    pub(crate) graph_revision: u64,
    pub(crate) start: IVec3,
    pub(crate) hadamard_modifier: bool,
}

impl PipePreviewHintSyncState {
    pub(crate) const fn new(graph_revision: u64, start: IVec3, hadamard_modifier: bool) -> Self {
        Self {
            graph_revision,
            start,
            hadamard_modifier,
        }
    }
}

/// Number of directional pipe-hint markers (the six axis neighbours).
const PIPE_PREVIEW_HINT_COUNT: usize = 6;

/// Number of walking-block hint markers (the eight oblique moves).
const WALKING_PREVIEW_HINT_COUNT: usize = 8;

/// Fixed slots for pipe-hint endpoints, each `Some((source, target))` when active.
type PipePreviewHintSlots = [Option<(IVec3, IVec3)>; PIPE_PREVIEW_HINT_COUNT];

/// Fixed slots for walking-hint endpoints, each `Some((source, target))` when active.
type WalkingPreviewHintSlots = [Option<(IVec3, IVec3)>; WALKING_PREVIEW_HINT_COUNT];

/// Precomputed endpoint targets for each family of placement hint marker.
#[derive(Debug, Clone, Default)]
pub(crate) struct PipePreviewHintTargets {
    pub(crate) pipe: PipePreviewHintSlots,
    pub(crate) walking_pipe: PipePreviewHintSlots,
    pub(crate) patch_rotation_pipe: PipePreviewHintSlots,
    pub(crate) tall_cube_pipe: PipePreviewHintSlots,
    pub(crate) walking: WalkingPreviewHintSlots,
}

/// Caches the placement hint targets and the adjacency snapshot they were
/// computed from, so hints rebuild only when inputs change.
#[derive(Default)]
pub(super) struct PipePreviewHintCache {
    applied: Option<PipePreviewHintSyncState>,
    snapshot_revision: Option<u64>,
    snapshot: GraphAdjacencySnapshot,
    targets: PipePreviewHintTargets,
}
