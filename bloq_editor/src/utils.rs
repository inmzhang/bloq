//! Coordinate mapping between graph space and Bevy world space, plane picking,
//! and the graph-adjacency snapshot used to derive render signatures and pipe
//! hints.
//!
//! Graph space is the integer lattice of block/pipe positions; world space is
//! the rendered scene. The two axes are permuted and scaled by a `stride` of
//! `pipe_length + 1.0`: graph `(x, y, z)` maps to world `(x, z, -y) * stride`.

use crate::resources::{
    EditorMode, EditorState, GraphState, RenderSignatureBlockIdentity,
    RenderSignatureConnectableOffsets, RenderSignaturePipeIdentity,
};
use bevy::shape::Ray3d;
use bloq_graph::{Block, BlockGraph, BlockGraphError, BlockKind};
use color_eyre::eyre;
use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;

const PLANE_INTERSECTION_EPSILON: f32 = 1e-6;

#[cfg(test)]
pub(crate) fn one_bit_adder_fixture() -> BlockGraph {
    BlockGraph::from_text(include_str!("../../bloq_test/assets/one_bit_adder.blog"))
        .expect("test-only bulk-adder source loads")
}

/// Maps a graph-space position to world space (axes permuted, scaled by the
/// pipe stride).
pub(crate) fn graph_to_world(pos: Vec3, pipe_length: f32) -> Vec3 {
    let stride = pipe_length + 1.0;
    let p = pos * stride;
    Vec3::new(p.x, p.z, -p.y)
}

/// Maps a world-space position back to graph space; the inverse of
/// [`graph_to_world`].
pub(crate) fn world_to_graph_space(world_pos: Vec3, pipe_length: f32) -> Vec3 {
    let stride = pipe_length + 1.0;
    Vec3::new(world_pos.x, -world_pos.z, world_pos.y) / stride
}

/// The inclusive `(min, max)` graph-space bounding box of all occupied cells,
/// or `None` when the graph is empty.
pub(crate) fn graph_bounds(graph: &BlockGraph) -> Option<(IVec3, IVec3)> {
    let mut positions = graph.occupied_positions();
    let first = positions.next()?;
    Some(positions.fold((first, first), |(min, max), pos| {
        (min.min(pos), max.max(pos))
    }))
}

/// Resolves the displayed geometry; callers own stabilizer analysis.
pub(crate) fn displayed_branch_projection(
    graph: &BlockGraph,
) -> Result<BlockGraph, BlockGraphError> {
    if graph.branch_definitions().is_empty() {
        return Ok(graph.clone());
    }
    graph.project_branches_deferred(
        graph
            .branch_definitions()
            .iter()
            .map(|branch| (branch.target, branch.shown_true())),
    )
}

/// Intersects a world-space ray with the horizontal editing plane at graph
/// height `height` and returns the graph cell it lands in, or `None` if the ray
/// is parallel to the plane or points away from it.
pub(crate) fn intersect_plane(ray: Ray3d, height: f32, pipe_length: f32) -> Option<IVec3> {
    if ray.direction.y.abs() <= PLANE_INTERSECTION_EPSILON {
        return None;
    }

    let t = (height - ray.origin.y) / ray.direction.y;
    if !t.is_finite() || t < 0.0 {
        None
    } else {
        let world_pos = ray.origin + *ray.direction * t;
        if !world_pos.is_finite() {
            return None;
        }

        let graph_pos_float = world_to_graph_space(world_pos, pipe_length);
        if !graph_pos_float.is_finite() {
            return None;
        }

        Some(graph_pos_float.round().as_ivec3())
    }
}

fn infer_hadamard(u: &Block, v: &Block) -> bool {
    if let BlockKind::Cube(uk) = u.kind()
        && let BlockKind::Cube(vk) = v.kind()
    {
        return !uk
            .bases()
            .into_iter()
            .zip(vk.bases())
            .enumerate()
            .any(|(i, (ub, vb))| u.pos()[i] == v.pos()[i] && ub == vb);
    }
    false
}

/// Decides whether a pipe placed between `start` and `end` should be a Hadamard
/// pipe: `true` when forced, otherwise inferred from a basis mismatch across the
/// two endpoint cubes.
pub(crate) fn pipe_placement_hadamard(
    graph: &BlockGraph,
    start: IVec3,
    end: IVec3,
    force_hadamard: bool,
) -> bool {
    if force_hadamard {
        return true;
    }

    match (
        graph.get_endpoint_block(start),
        graph.get_endpoint_block(end),
    ) {
        (Some(u), Some(v)) => infer_hadamard(u, v),
        _ => false,
    }
}

/// Whether a pipe is shown in the current-layer view: it lies in the plane, or
/// it connects the plane to an adjacent layer.
pub(crate) fn is_pipe_visible(u: IVec3, v: IVec3, plane_height: i32) -> bool {
    let u_z = u.z;
    let v_z = v.z;
    (u_z == plane_height && v_z.abs_diff(plane_height) <= 1)
        || (v_z == plane_height && u_z.abs_diff(plane_height) <= 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SnapshotBlock {
    pub(crate) pos: IVec3,
    pub(crate) kind: BlockKind,
    pub(crate) height_cells: u32,
}

/// A compact view of block endpoints and pipe adjacency, built from the graph
/// (or from cached render identities) so pipe-hint and render-signature passes
/// can query neighbours without re-walking the full graph each frame.
#[derive(Default)]
pub(crate) struct GraphAdjacencySnapshot {
    endpoint_blocks: FxHashMap<IVec3, SnapshotBlock>,
    pipe_neighbors: FxHashMap<IVec3, u8>,
}

impl GraphAdjacencySnapshot {
    pub(crate) fn from_graph(graph: &BlockGraph) -> Self {
        let mut snapshot = Self {
            endpoint_blocks: FxHashMap::with_capacity_and_hasher(
                graph.block_count().saturating_mul(2),
                Default::default(),
            ),
            pipe_neighbors: FxHashMap::with_capacity_and_hasher(
                graph.pipe_count().saturating_mul(2),
                Default::default(),
            ),
        };
        for block in graph.blocks() {
            let block_snapshot = SnapshotBlock {
                pos: block.pos(),
                kind: block.kind(),
                height_cells: block.height_cells(),
            };
            for offset in compact_connectable_offsets(block).iter() {
                snapshot
                    .endpoint_blocks
                    .insert(block.pos() + offset, block_snapshot);
            }
        }
        for (u_pos, v_pos, _, _, _) in graph.pipe_endpoints_with_blocks() {
            snapshot.insert_pipe_neighbor(u_pos, v_pos);
        }
        snapshot
    }

    pub(crate) fn from_render_identities(
        block_identities: &FxHashMap<IVec3, RenderSignatureBlockIdentity>,
        pipe_identities: &FxHashMap<crate::components::PipeKey, RenderSignaturePipeIdentity>,
    ) -> Self {
        let mut snapshot = Self {
            endpoint_blocks: FxHashMap::with_capacity_and_hasher(
                block_identities.len().saturating_mul(2),
                Default::default(),
            ),
            pipe_neighbors: FxHashMap::with_capacity_and_hasher(
                pipe_identities.len().saturating_mul(2),
                Default::default(),
            ),
        };
        for (pos, identity) in block_identities {
            let block_snapshot = SnapshotBlock {
                pos: *pos,
                kind: identity.kind,
                height_cells: identity.height_cells,
            };
            for offset in identity.connectable_offsets.iter() {
                snapshot
                    .endpoint_blocks
                    .insert(*pos + offset, block_snapshot);
            }
        }
        for identity in pipe_identities.values() {
            snapshot.insert_identity_pipe_neighbor(identity);
        }
        snapshot
    }

    pub(crate) fn endpoint_block(&self, endpoint: IVec3) -> Option<SnapshotBlock> {
        self.endpoint_blocks.get(&endpoint).copied()
    }

    pub(crate) fn has_pipe_between(&self, u: IVec3, v: IVec3) -> bool {
        let Some(direction_bit) = v.checked_sub(u).and_then(adjacent_direction_bit) else {
            return false;
        };
        self.pipe_neighbors
            .get(&u)
            .is_some_and(|mask| mask & direction_bit != 0)
    }

    pub(crate) fn degree(&self, endpoint: IVec3) -> usize {
        self.pipe_neighbors
            .get(&endpoint)
            .copied()
            .map_or(0, |mask| mask.count_ones() as usize)
    }

    pub(crate) fn insert_identity_pipe_neighbor(&mut self, identity: &RenderSignaturePipeIdentity) {
        self.insert_pipe_neighbor(identity.src, identity.src + identity.dir);
    }

    pub(crate) fn remove_identity_pipe_neighbor(&mut self, identity: &RenderSignaturePipeIdentity) {
        self.remove_pipe_neighbor(identity.src, identity.src + identity.dir);
    }

    pub(crate) fn insert_pipe_neighbor(&mut self, u: IVec3, v: IVec3) {
        if let Some(bit) = adjacent_direction_bit(v - u) {
            *self.pipe_neighbors.entry(u).or_default() |= bit;
        }
        if let Some(bit) = adjacent_direction_bit(u - v) {
            *self.pipe_neighbors.entry(v).or_default() |= bit;
        }
    }

    fn remove_pipe_neighbor(&mut self, u: IVec3, v: IVec3) {
        if let Some(bit) = adjacent_direction_bit(v - u)
            && let Some(mask) = self.pipe_neighbors.get_mut(&u)
        {
            *mask &= !bit;
            if *mask == 0 {
                self.pipe_neighbors.remove(&u);
            }
        }
        if let Some(bit) = adjacent_direction_bit(u - v)
            && let Some(mask) = self.pipe_neighbors.get_mut(&v)
        {
            *mask &= !bit;
            if *mask == 0 {
                self.pipe_neighbors.remove(&v);
            }
        }
    }
}

pub(crate) fn compact_connectable_offsets(block: &Block) -> RenderSignatureConnectableOffsets {
    let offsets = block.connectable_offsets();
    let mut compact = RenderSignatureConnectableOffsets {
        offsets: [IVec3::ZERO; 2],
        len: offsets.len() as u8,
    };
    compact.offsets[..offsets.len()].copy_from_slice(&offsets);
    compact
}

pub(crate) fn cube_pipe_face_is_degenerate(
    kind: BlockKind,
    endpoint: IVec3,
    other_endpoint: IVec3,
) -> bool {
    let BlockKind::Cube(cube) = kind else {
        return false;
    };
    let Some(pipe_axis) = (0..3).find(|&axis| other_endpoint[axis] != endpoint[axis]) else {
        return false;
    };
    let perp = match pipe_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    cube.bases()[perp[0]] == cube.bases()[perp[1]]
}

pub(crate) const ADJACENT_DIRECTIONS: [IVec3; 6] = [
    IVec3::NEG_X,
    IVec3::X,
    IVec3::NEG_Y,
    IVec3::Y,
    IVec3::NEG_Z,
    IVec3::Z,
];

pub(crate) fn adjacent_direction_bit(delta: IVec3) -> Option<u8> {
    ADJACENT_DIRECTIONS
        .into_iter()
        .position(|direction| direction == delta)
        .map(|index| 1 << index)
}

/// Moves the editing plane by `delta` layers.
pub(crate) fn change_plane_height(
    delta: i32,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
) {
    set_plane_height(
        editor_state.plane_height.saturating_add(delta),
        editor_state,
        graph_state,
    );
}

/// Sets the editing plane height, requesting a rerender (and clearing
/// stabilizers) when the current-layer view filter is active.
pub(crate) fn set_plane_height(
    plane_height: i32,
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
) {
    if editor_state.plane_height == plane_height {
        return;
    }

    editor_state.plane_height = plane_height;
    if editor_state.view_current_layer_only && editor_state.mode == EditorMode::View {
        graph_state.needs_rerender = true;
        editor_state.clear_stabilizers();
    }
}

// ============================================================================
// File download
// ============================================================================
//
// A single save path shared by every "export a file" flow (compiled Stim/IR
// downloads, SVG view exports). The two targets diverge completely: native pops
// an `rfd` save dialog and writes to disk, web synthesizes a Blob and clicks a
// hidden anchor to trigger a browser download.

/// Saves `contents` under a user-chosen name. Returns the saved location on
/// success, or `None` when the user canceled the save dialog (not an error).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn save_download(file_name: &str, contents: &[u8]) -> eyre::Result<Option<String>> {
    use color_eyre::eyre::WrapErr as _;

    let Some(path) = rfd::FileDialog::new().set_file_name(file_name).save_file() else {
        return Ok(None);
    };

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .wrap_err_with(|| format!("create export directory {}", parent.display()))?;
    }
    std::fs::write(&path, contents)
        .wrap_err_with(|| format!("write download {}", path.display()))?;
    Ok(Some(path.display().to_string()))
}

/// Saves `contents` under `file_name` by triggering a browser download; the
/// return is always `Some(file_name)` since the web has no cancelable dialog.
#[cfg(target_arch = "wasm32")]
pub(crate) fn save_download(file_name: &str, contents: &[u8]) -> eyre::Result<Option<String>> {
    use color_eyre::eyre::{ContextCompat as _, eyre};
    use wasm_bindgen::{JsCast, JsValue};

    // `JsValue` is not `std::error::Error`, so each DOM edge stringifies its
    // error into a fresh report at the failure site.
    fn js_error(action: &'static str) -> impl Fn(JsValue) -> eyre::Report {
        move |value| {
            let rendered = value.as_string().unwrap_or_else(|| format!("{value:?}"));
            eyre!("{action}: {rendered}")
        }
    }

    let window = web_sys::window().wrap_err("window is unavailable")?;
    let document = window.document().wrap_err("document is unavailable")?;
    let body = document.body().wrap_err("document body is unavailable")?;

    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(contents));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)
        .map_err(js_error("create download blob"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(js_error("create download blob URL"))?;
    let anchor = document
        .create_element("a")
        .map_err(js_error("create download anchor"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| eyre!("failed to create download anchor"))?;
    anchor.set_href(&url);
    anchor.set_download(file_name);
    body.append_child(&anchor)
        .map_err(js_error("attach download anchor"))?;
    let anchor_element = anchor
        .clone()
        .dyn_into::<web_sys::HtmlElement>()
        .map_err(|_| eyre!("failed to cast download anchor to HtmlElement"))?;
    anchor_element.click();
    let _ = body.remove_child(&anchor);
    web_sys::Url::revoke_object_url(&url).map_err(js_error("revoke download blob URL"))?;
    Ok(Some(file_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{
        GraphAdjacencySnapshot, adjacent_direction_bit, graph_to_world, intersect_plane,
        pipe_placement_hadamard, set_plane_height, world_to_graph_space,
    };
    use crate::resources::{EditorMode, EditorState, GraphState};
    use bevy::math::Dir3;
    use bevy::shape::Ray3d;
    use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe};
    use glam::{IVec3, Vec3};

    #[test]
    fn graph_world_mapping_round_trip() {
        let graph_pos = Vec3::new(3.0, -2.0, 5.0);
        let pipe_length = 2.5;
        let world_pos = graph_to_world(graph_pos, pipe_length);
        let reconstructed = world_to_graph_space(world_pos, pipe_length);
        assert!((reconstructed - graph_pos).length() < 1e-6);
    }

    #[test]
    fn intersect_plane_hits_expected_grid_cell() {
        let ray = Ray3d::new(Vec3::new(0.2, 10.0, -0.4), Dir3::NEG_Y);
        let hit = intersect_plane(ray, 0.0, 1.0);
        assert_eq!(hit, Some(IVec3::new(0, 0, 0)));
    }

    #[test]
    fn intersect_plane_returns_none_for_parallel_ray() {
        let ray = Ray3d::new(Vec3::new(2.0, 3.0, -4.0), Dir3::X);
        let hit = intersect_plane(ray, 0.0, 1.0);
        assert_eq!(hit, None);
    }

    #[test]
    fn intersect_plane_returns_none_when_plane_is_behind_ray() {
        let ray = Ray3d::new(Vec3::new(1.0, 2.0, 3.0), Dir3::Y);
        let hit = intersect_plane(ray, 0.0, 1.0);
        assert_eq!(hit, None);
    }

    #[test]
    fn intersect_plane_accepts_origin_on_plane() {
        let ray = Ray3d::new(Vec3::new(1.1, 0.0, -1.1), Dir3::NEG_Y);
        let hit = intersect_plane(ray, 0.0, 1.0);
        assert_eq!(hit, Some(IVec3::new(1, 1, 0)));
    }

    #[test]
    fn pipe_placement_hadamard_can_be_forced_for_empty_endpoint() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));

        assert!(pipe_placement_hadamard(&graph, IVec3::ZERO, IVec3::X, true));
        assert!(!pipe_placement_hadamard(
            &graph,
            IVec3::ZERO,
            IVec3::X,
            false
        ));
    }

    #[test]
    fn pipe_placement_hadamard_keeps_inference_without_modifier() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XZX)));

        assert!(pipe_placement_hadamard(
            &graph,
            IVec3::ZERO,
            IVec3::X,
            false
        ));
    }

    #[test]
    fn set_plane_height_updates_directly_and_rerenders_layer_filtered_view() {
        let mut editor_state = EditorState {
            mode: EditorMode::View,
            view_current_layer_only: true,
            ..EditorState::default()
        };
        let mut graph_state = GraphState::default();

        set_plane_height(27, &mut editor_state, &mut graph_state);

        assert_eq!(editor_state.plane_height, 27);
        assert!(graph_state.needs_rerender);
    }

    #[test]
    fn graph_adjacency_snapshot_tracks_endpoints_and_pipe_neighbors() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let snapshot = GraphAdjacencySnapshot::from_graph(&graph);

        assert_eq!(adjacent_direction_bit(IVec3::X), Some(1 << 1));
        assert_eq!(
            snapshot.endpoint_block(IVec3::ZERO).map(|block| block.pos),
            Some(IVec3::ZERO)
        );
        assert!(snapshot.has_pipe_between(IVec3::ZERO, IVec3::X));
        assert!(snapshot.has_pipe_between(IVec3::X, IVec3::ZERO));
        assert_eq!(snapshot.degree(IVec3::ZERO), 1);
        assert_eq!(snapshot.degree(IVec3::Y), 0);
    }
}
