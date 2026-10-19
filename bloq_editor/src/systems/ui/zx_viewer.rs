//! Orbitable source and 2D fully simplified ZX diagrams. Source hover
//! cross-highlights the block graph.

use super::{graph_layout, tiled_window_rect};
use crate::components::{CameraSettings, GraphElement};
use crate::resources::{EditorState, EditorTabId, GraphState};
use crate::systems::camera::{
    CAMERA_DEFAULT_FOV_Y, camera_transform_for_settings, orbit_camera_settings,
    pan_camera_settings, zoom_camera_settings,
};
use crate::systems::jobs::{EditorJobs, schedule_zx_simplification};
use crate::theme::{ThemePalette, palette, pauli_basis_color, with_alpha};
use crate::utils::graph_to_world;
use bevy::prelude::Resource;
use bevy_egui::egui::{self, Color32, Pos2, Rect, Sense, Shape, Stroke, Vec2 as EguiVec2};
use bloq_graph::verify::{LogicalVerifier, QuizxGraph};
use bloq_graph::{
    BlockGraph, CancellationToken, NodeKind, Pauli, PauliBasis, SelectiveKind, Stabilizer, ZXGraph,
    ZXNode,
};
use color_eyre::eyre::{self, ContextCompat, WrapErr};
use glam::{IVec3, Vec2, Vec3};
use quizx::graph::{EType, GraphLike, V, VType};
use std::collections::{HashMap, HashSet};

// ============================================================================
// ZX-viewer state
// ============================================================================

pub(crate) type ZxViewKey = (u64, bool, u64);

/// ZX-viewer mode, camera, sampling seed, and rendered-graph cache.
#[derive(Resource, Clone, Default)]
pub(crate) struct ZxViewerState {
    pub(crate) open: bool,
    pub(crate) camera: CameraSettings,
    simplified: bool,
    sample_seed: u64,
    simplified_pan: EguiVec2,
    /// Logarithmic zoom relative to fitting the cached 2D layout.
    simplified_zoom: f32,
    /// Graph revision, view mode, and sample seed used by `cached_view`.
    cached_key: Option<ZxViewKey>,
    cached_view: Option<Result<ZxGraphView, String>>,
}

impl ZxViewerState {
    pub(crate) fn toggle(&mut self) {
        self.open = !self.open;
        // The cache can hold a large graph; rebuild it lazily after reopening.
        if !self.open {
            self.cached_key = None;
            self.cached_view = None;
        }
    }

    /// Clone tab state without its large derived graph.
    pub(crate) fn without_cache(&self) -> Self {
        Self {
            open: self.open,
            camera: self.camera,
            simplified: self.simplified,
            sample_seed: self.sample_seed,
            simplified_pan: self.simplified_pan,
            simplified_zoom: self.simplified_zoom,
            cached_key: None,
            cached_view: None,
        }
    }

    pub(crate) fn reset_camera(&mut self) {
        self.camera = CameraSettings::default();
        self.simplified_pan = EguiVec2::ZERO;
        self.simplified_zoom = 0.0;
    }

    pub(crate) fn set_simplified(&mut self, simplified: bool) {
        self.simplified = simplified;
        self.reset_camera();
    }

    fn resample(&mut self) {
        self.sample_seed = self.sample_seed.wrapping_add(1);
    }

    pub(crate) fn view_key(&self, revision: u64) -> ZxViewKey {
        (revision, self.simplified, self.sample_seed)
    }

    pub(crate) fn accepts_simplification(&self, revision: u64, key: ZxViewKey) -> bool {
        self.open && key.1 && self.view_key(revision) == key
    }

    pub(crate) fn cancel_simplification(&mut self, revision: u64, key: ZxViewKey) -> bool {
        self.finish_view(revision, key, Ok(ZxGraphView::Cancelled))
    }

    pub(crate) fn finish_simplification(
        &mut self,
        revision: u64,
        key: ZxViewKey,
        result: Result<SimplifiedZxView, String>,
    ) -> bool {
        self.finish_view(revision, key, result.map(ZxGraphView::Simplified))
    }

    fn finish_view(
        &mut self,
        revision: u64,
        key: ZxViewKey,
        view: Result<ZxGraphView, String>,
    ) -> bool {
        if !self.accepts_simplification(revision, key) {
            return false;
        }
        self.cached_key = Some(key);
        self.cached_view = Some(view);
        true
    }

    /// Source conversion is cheap; logical verification, simplification, and layout run
    /// on the task pool/Worker while the viewer paints its pending state.
    fn refresh_view(
        &mut self,
        graph_state: &GraphState,
        tab_id: EditorTabId,
        jobs: &mut EditorJobs,
    ) {
        let key = self.view_key(graph_state.revision);
        if self.cached_key != Some(key) {
            self.cached_key = Some(key);
            self.cached_view = None;
        }
        if self.cached_view.is_some() {
            return;
        }
        if self.simplified {
            if let Err(error) = schedule_zx_simplification(jobs, tab_id, graph_state, key) {
                self.cached_view = Some(Err(error));
            }
        } else {
            self.cached_view = Some(
                graph_state
                    .graph
                    .to_zx_graph()
                    .map(ZxGraphView::Source)
                    .map_err(|error| format!("convert block graph to ZX graph: {error}")),
            );
        }
    }
}

/// A source or fully simplified ZX graph, tagged with the source revision.
#[derive(Clone)]
enum ZxGraphView {
    Source(ZXGraph),
    Simplified(SimplifiedZxView),
    Cancelled,
}

/// Display geometry is derived once from topology, without changing the signed
/// diagram, scalar, coordinates, or ordered tensor boundaries.
#[derive(Clone)]
pub(crate) struct SimplifiedZxView {
    graph: QuizxGraph,
    positions: HashMap<V, Pos2>,
    edges: Vec<SimplifiedEdge>,
    bounds: Rect,
}

#[derive(Clone, Debug, PartialEq)]
struct SimplifiedEdge {
    vertices: [V; 2],
    points: [Pos2; 3],
    hadamard: bool,
    marker: Pos2,
}

impl SimplifiedZxView {
    pub(crate) fn new(graph: QuizxGraph) -> Self {
        let mut vertices: Vec<_> = graph.vertices().collect();
        vertices.sort_unstable_by(|&lhs, &rhs| {
            let a = graph.coord(lhs);
            let b = graph.coord(rhs);
            a.x.total_cmp(&b.x)
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| lhs.cmp(&rhs))
        });
        let index: HashMap<_, _> = vertices
            .iter()
            .enumerate()
            .map(|(i, &vertex)| (vertex, i as u32))
            .collect();
        let nodes: Vec<_> = (0..vertices.len() as u32)
            .map(|i| (i, egui::vec2(32.0, 32.0)))
            .collect();
        // ZX edges are undirected. Orient only the layout copy by stable source
        // coordinates; the original edge types and tensor boundary order stay put.
        let edges: Vec<_> = graph
            .edges()
            .map(|(a, b, _)| (index[&a].min(index[&b]), index[&a].max(index[&b])))
            .collect();
        let layout = graph_layout::compute_relative(&nodes, &edges, 28.0);
        let positions: HashMap<_, _> = vertices
            .iter()
            .map(|&vertex| {
                let (x, y) = layout.positions[&index[&vertex]];
                (vertex, Pos2::new(x + 16.0, y + 16.0))
            })
            .collect();
        let mut graph_edges: Vec<_> = graph.edges().collect();
        graph_edges.sort_unstable_by_key(|&(a, b, _)| (a.min(b), a.max(b)));
        let mut edges = Vec::with_capacity(graph_edges.len());
        let mut routes = Vec::with_capacity(graph_edges.len());
        for (a, b, kind) in graph_edges {
            let points = route_simplified_edge(a, b, &positions, &routes);
            let hadamard = kind == EType::H;
            edges.push(SimplifiedEdge {
                vertices: [a, b],
                points,
                hadamard,
                marker: quadratic_point(points, 0.5),
            });
            routes.push(sample_simplified_edge(points));
        }
        for index in 0..edges.len() {
            let (previous, remaining) = edges.split_at_mut(index);
            let edge = &mut remaining[0];
            if edge.hadamard {
                edge.marker =
                    simplified_hadamard_position(edge.points, &positions, previous, &routes);
            }
        }
        Self::from_parts(graph, positions, edges)
    }

    fn from_parts(
        graph: QuizxGraph,
        positions: HashMap<V, Pos2>,
        edges: Vec<SimplifiedEdge>,
    ) -> Self {
        let mut bounds = Rect::NOTHING;
        for &position in positions.values() {
            bounds.extend_with(position);
        }
        for edge in &edges {
            for &point in &edge.points {
                bounds.extend_with(point);
            }
        }
        if positions.is_empty() {
            bounds = Rect::from_center_size(Pos2::ZERO, egui::vec2(1.0, 1.0));
        }
        Self {
            graph,
            positions,
            edges,
            bounds: bounds.expand(32.0),
        }
    }

    fn scale(&self, rect: Rect, viewer: &ZxViewerState) -> f32 {
        (rect.width() / self.bounds.width())
            .min(rect.height() / self.bounds.height())
            .min(2.0)
            * viewer.simplified_zoom.exp()
    }
}

fn route_simplified_edge(
    a: V,
    b: V,
    positions: &HashMap<V, Pos2>,
    routes: &[[Pos2; 25]],
) -> [Pos2; 3] {
    let start = positions[&a];
    let end = positions[&b];
    let middle = start.lerp(end, 0.5);
    let delta = end - start;
    let normal = egui::vec2(-delta.y, delta.x).normalized();
    let mut best = [start, middle, end];
    let mut best_clearance = f32::NEG_INFINITY;
    let mut clear_route = None;
    let span = positions
        .values()
        .map(|position| position.distance(middle))
        .fold(delta.length(), f32::max);
    // Coarse fixed bends miss clear lanes, even in the one-bit adder. Search
    // both sides in small steps, with the reach set by this diagram's extent.
    // ponytail: one quadratic per wire; use obstacle routing only if adversarial
    // placements exhaust these lanes. Nonplanar graphs still need crossings.
    let bends = (span * 2.0 / 16.0).ceil() as usize;
    for index in 0..=bends * 2 {
        let bend = index.div_ceil(2) as f32 * 16.0 * if index % 2 == 0 { -1.0 } else { 1.0 };
        let points = [start, middle + normal * bend, end];
        let samples = sample_simplified_edge(points);
        let clearance = positions
            .iter()
            .filter(|(vertex, _)| **vertex != a && **vertex != b)
            .map(|(_, &position)| {
                samples
                    .windows(2)
                    .map(|segment| distance_to_segment(position, segment[0], segment[1]))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(f32::INFINITY, f32::min);
        // The curve lies within this distance of its sampled chords. Account
        // for it so larger bends cannot slip through a node between samples.
        if clearance - bend.abs() / (2.0 * 24.0_f32.powi(2)) >= 20.0 {
            clear_route.get_or_insert(points);
            if !simplified_routes_overlap(&samples, routes) {
                return points;
            }
        }
        if clearance > best_clearance {
            best = points;
            best_clearance = clearance;
        }
    }
    clear_route.unwrap_or(best)
}

fn sample_simplified_edge(points: [Pos2; 3]) -> [Pos2; 25] {
    std::array::from_fn(|index| quadratic_point(points, index as f32 / 24.0))
}

/// Crossing wires are fine; a long nearly parallel overlap hides one wire.
fn simplified_routes_overlap(samples: &[Pos2; 25], routes: &[[Pos2; 25]]) -> bool {
    samples.windows(2).any(|segment| {
        let middle = segment[0].lerp(segment[1], 0.5);
        if middle.distance(samples[0]) < 20.0 || middle.distance(samples[24]) < 20.0 {
            return false;
        }
        let direction = (segment[1] - segment[0]).normalized();
        routes.iter().any(|route| {
            if middle.distance(route[0]) < 20.0 || middle.distance(route[24]) < 20.0 {
                return false;
            }
            route.windows(2).any(|previous| {
                direction
                    .dot((previous[1] - previous[0]).normalized())
                    .abs()
                    > 0.995
                    && distance_to_segment(middle, previous[0], previous[1]) < 3.0
            })
        })
    })
}

fn simplified_hadamard_position(
    points: [Pos2; 3],
    positions: &HashMap<V, Pos2>,
    edges: &[SimplifiedEdge],
    routes: &[[Pos2; 25]],
) -> Pos2 {
    // PyZX also shifts Hadamard boxes along diagonal wires. Prefer positions
    // clear of spiders and other boxes, without changing the wire's type.
    let mut best = quadratic_point(points, 0.5);
    let mut best_clearance = f32::NEG_INFINITY;
    let candidates = [0.5, 0.4, 0.6]
        .into_iter()
        .chain((1..20).flat_map(|index| [0.5 - index as f32 * 0.02, 0.5 + index as f32 * 0.02]));
    for t in candidates {
        let point = quadratic_point(points, t);
        let clearance = positions
            .values()
            .map(|&position| point.distance(position) - 25.0)
            .chain(edges.iter().filter(|edge| edge.hadamard).map(|edge| {
                (point.x - edge.marker.x)
                    .abs()
                    .max((point.y - edge.marker.y).abs())
                    - 12.0
            }))
            .chain(
                routes
                    .iter()
                    .filter(|route| route[0] != points[0] || route[24] != points[2])
                    .flat_map(|route| {
                        route
                            .windows(2)
                            .map(|segment| distance_to_segment(point, segment[0], segment[1]) - 8.0)
                    }),
            )
            .fold(f32::INFINITY, f32::min);
        if clearance >= 0.0 {
            return point;
        }
        if clearance > best_clearance {
            best = point;
            best_clearance = clearance;
        }
    }
    best
}

fn quadratic_point([a, control, b]: [Pos2; 3], t: f32) -> Pos2 {
    a.lerp(control, t).lerp(control.lerp(b, t), t)
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(serde::Serialize, serde::Deserialize)]
struct ZxWorkerView {
    graph: String,
    vertices: Vec<([f64; 2], [f32; 2])>,
    edges: Vec<(usize, usize, [f32; 2], [f32; 2])>,
}

/// Transfer prepared geometry rather than laying out the result on the UI thread.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn encode_zx_worker_view(view: &SimplifiedZxView) -> eyre::Result<String> {
    let mut graph = view.graph.clone();
    let vertices: Vec<_> = graph.vertices().collect();
    let indexes: HashMap<_, _> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let vertices = vertices
        .into_iter()
        .map(|vertex| {
            let coord = graph.coord(vertex);
            // QuiZX's JSON reader renumbers vertices. The transport copy carries
            // identity in coordinates; decoding restores the original coordinates.
            graph.set_coord(vertex, (indexes[&vertex] as f64, 0.0));
            ([coord.x, coord.y], view.positions[&vertex].into())
        })
        .collect();
    let edges = view
        .edges
        .iter()
        .map(|edge| {
            (
                indexes[&edge.vertices[0]],
                indexes[&edge.vertices[1]],
                edge.points[1].into(),
                edge.marker.into(),
            )
        })
        .collect();
    Ok(serde_json::to_string(&ZxWorkerView {
        graph: quizx::json::encode_graph(&graph)?,
        vertices,
        edges,
    })?)
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn decode_zx_worker_view(source: &str) -> eyre::Result<SimplifiedZxView> {
    let source: ZxWorkerView = serde_json::from_str(source)?;
    let mut graph: QuizxGraph = quizx::json::decode_graph(&source.graph)?;
    eyre::ensure!(
        graph.num_vertices() == source.vertices.len(),
        "ZX geometry vertex count differs"
    );
    eyre::ensure!(
        graph.num_edges() == source.edges.len(),
        "ZX geometry edge count differs"
    );
    let mut positions = HashMap::with_capacity(source.vertices.len());
    let mut vertices = HashMap::with_capacity(source.vertices.len());
    for vertex in graph.vertices().collect::<Vec<_>>() {
        let coord = graph.coord(vertex);
        let index = coord.x as usize;
        eyre::ensure!(
            coord.x == index as f64 && coord.y == 0.0,
            "ZX geometry has an invalid vertex index"
        );
        let &(original, position) = source
            .vertices
            .get(index)
            .wrap_err("ZX geometry omits a vertex")?;
        eyre::ensure!(
            original.into_iter().all(f64::is_finite) && position.into_iter().all(f32::is_finite),
            "ZX geometry has non-finite coordinates"
        );
        eyre::ensure!(
            vertices.insert(index, vertex).is_none(),
            "ZX geometry repeats a vertex"
        );
        graph.set_coord(vertex, (original[0], original[1]));
        positions.insert(vertex, Pos2::from(position));
    }
    let mut edges = Vec::with_capacity(source.edges.len());
    let mut seen = HashSet::with_capacity(source.edges.len());
    for (a, b, control, marker) in source.edges {
        eyre::ensure!(
            control.into_iter().chain(marker).all(f32::is_finite),
            "ZX geometry has non-finite wire coordinates"
        );
        let a = *vertices
            .get(&a)
            .wrap_err("ZX geometry has an unknown wire endpoint")?;
        let b = *vertices
            .get(&b)
            .wrap_err("ZX geometry has an unknown wire endpoint")?;
        eyre::ensure!(
            graph.connected(a, b) && seen.insert((a.min(b), a.max(b))),
            "ZX geometry has a missing or repeated wire"
        );
        edges.push(SimplifiedEdge {
            vertices: [a, b],
            points: [positions[&a], Pos2::from(control), positions[&b]],
            hadamard: graph.edge_type(a, b) == EType::H,
            marker: Pos2::from(marker),
        });
    }
    let view = SimplifiedZxView::from_parts(graph, positions, edges);
    eyre::ensure!(
        view.bounds.min.is_finite()
            && view.bounds.max.is_finite()
            && view.bounds.size().is_finite(),
        "ZX geometry bounds are non-finite"
    );
    Ok(view)
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn simplified_zx_graph(graph: &BlockGraph, seed: u64) -> eyre::Result<QuizxGraph> {
    simplified_zx_graph_with_cancellation(graph, seed, &CancellationToken::new())
}

pub(crate) fn simplified_zx_graph_with_cancellation(
    graph: &BlockGraph,
    seed: u64,
    cancellation: &CancellationToken,
) -> eyre::Result<QuizxGraph> {
    cancellation.run(|| {
        let verifier = LogicalVerifier::new(graph).wrap_err("build logical ZX graph")?;
        cancellation.check()?;
        let (_, diagram) = verifier
            .sample_with_seed(seed)
            .wrap_err("presample logical measurements")?;
        cancellation.check()?;
        let mut graph = diagram.wrap_err("sampled branch was rejected; resample")?;
        // QuiZX's complete rewrite includes a private gadget pass; retain that
        // algorithm and check around it rather than changing the simplification.
        quizx::simplify::full_simp(&mut graph);
        cancellation.check()?;
        Ok(graph)
    })
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(serde::Serialize, serde::Deserialize)]
struct ZxWorkerSource {
    blog: String,
    inputs: Vec<String>,
    false_arms: Vec<String>,
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn encode_zx_worker_source(graph: &BlockGraph) -> eyre::Result<String> {
    Ok(serde_json::to_string(&ZxWorkerSource {
        blog: graph.to_blog_body_text(),
        inputs: graph.action_graph().inputs().map(str::to_owned).collect(),
        false_arms: graph
            .branch_definitions()
            .iter()
            .filter(|branch| !branch.shown_true())
            .map(|branch| branch.name.clone())
            .collect(),
    })?)
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn decode_zx_worker_source(source: &str) -> eyre::Result<BlockGraph> {
    let source: ZxWorkerSource = serde_json::from_str(source)?;
    // BLOG text omits external Boolean declarations. Restore them exactly as
    // browser sessions do, before the verifier validates the authored graph.
    let mut graph = bloq_graph::lower_blog_ast_lenient(
        &bloq_graph::parse_blog_to_ast(&source.blog)?,
        source.inputs,
    )?;
    for name in source.false_arms {
        graph.set_shown_branch_arm(&name, false)?;
    }
    Ok(graph)
}

const NODE_RADIUS: f32 = 0.16;
const EDGE_WIDTH: f32 = 2.0;
const STABILIZER_WIDTH: f32 = 3.0;
const HADAMARD_MARKER_HALF: f32 = 5.0;

/// Draws the ZX viewer window, including pending work and conversion errors.
pub(crate) fn draw_zx_viewer(
    ctx: &egui::Context,
    viewport: egui::Rect,
    graph_state: &GraphState,
    editor_state: &mut EditorState,
    zx_viewer: &mut ZxViewerState,
    tab_id: EditorTabId,
    jobs: &mut EditorJobs,
) {
    editor_state.set_zx_hovered_elements(HashSet::new());
    if !zx_viewer.open {
        return;
    }

    let palette = palette(editor_state.theme_preset);
    let default_rect = tiled_window_rect(viewport, egui::Align2::RIGHT_TOP);
    let mut open = zx_viewer.open;
    egui::Window::new("ZX View")
        .open(&mut open)
        .default_rect(default_rect)
        .min_width(300.0_f32.min(default_rect.width()))
        .min_height(240.0_f32.min(default_rect.height()))
        .resizable(true)
        .show(ctx, |ui| {
            draw_window_contents(
                ui,
                graph_state,
                editor_state,
                zx_viewer,
                tab_id,
                jobs,
                palette,
            )
        });
    zx_viewer.open = open;
}

fn draw_window_contents(
    ui: &mut egui::Ui,
    graph_state: &GraphState,
    editor_state: &mut EditorState,
    zx_viewer: &mut ZxViewerState,
    tab_id: EditorTabId,
    jobs: &mut EditorJobs,
    palette: &ThemePalette,
) {
    let key = zx_viewer.view_key(graph_state.revision);
    ui.horizontal_wrapped(|ui| {
        ui.small(format!(
            "{} block(s), {} pipe(s)",
            graph_state.graph.block_count(),
            graph_state.graph.pipe_count()
        ));
        ui.separator();
        if ui
            .small_button(if zx_viewer.simplified {
                "Fit Graph"
            } else {
                "Reset Camera"
            })
            .clicked()
        {
            zx_viewer.reset_camera();
        }
        ui.separator();
        if ui
            .small_button(if zx_viewer.simplified {
                "Source Graph"
            } else {
                "Full Simplify"
            })
            .clicked()
        {
            jobs.cancel_zx_simplification(tab_id, key);
            zx_viewer.set_simplified(!zx_viewer.simplified);
        }
        if zx_viewer.simplified
            && graph_state.graph.has_actions()
            && ui.small_button("Resample").clicked()
        {
            jobs.cancel_zx_simplification(tab_id, key);
            zx_viewer.resample();
        }
        if jobs.zx_simplification_for(tab_id, key) {
            if ui.small_button("Cancel").clicked() {
                jobs.cancel_zx_simplification(tab_id, key);
                zx_viewer.cancel_simplification(graph_state.revision, key);
            }
        } else if zx_viewer.simplified
            && matches!(
                zx_viewer.cached_view,
                Some(Err(_)) | Some(Ok(ZxGraphView::Cancelled))
            )
            && ui.small_button("Retry").clicked()
        {
            zx_viewer.cached_view = None;
        }
        if editor_state.showing_stabilizers() {
            ui.separator();
            if let Some((_, generator)) = &editor_state.action_stabilizer {
                ui.small(format!(
                    "Measurement {}",
                    generator.measurement_name().unwrap_or("surface")
                ));
            } else {
                ui.small(format!(
                    "Stabilizer {}/{}",
                    editor_state.current_stabilizer_index + 1,
                    editor_state.stabilizers.len()
                ));
            }
        }
    });

    let height = (ui.available_height() - 4.0).max(180.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), height), Sense::drag());
    handle_camera_input(ui, rect, &response, zx_viewer);

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, palette.bg_dark);
    painter.rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, palette.border),
        egui::StrokeKind::Inside,
    );

    zx_viewer.refresh_view(graph_state, tab_id, jobs);
    let Some(cached) = zx_viewer.cached_view.as_ref() else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Simplifying ZX graph…",
            egui::FontId::proportional(14.0),
            palette.text_dim,
        );
        ui.ctx().request_repaint();
        return;
    };
    match cached {
        Ok(ZxGraphView::Source(zx_graph)) => {
            let stabilizer = editor_state
                .shown_stabilizer()
                .map(|generator| &generator.stabilizer);
            draw_zx_graph(
                &painter,
                rect,
                zx_viewer,
                zx_graph,
                editor_state.pipe_length,
                stabilizer,
                palette,
            );
            let hovered = response
                .hover_pos()
                .map(|pointer| {
                    hovered_elements(
                        pointer,
                        rect,
                        zx_viewer,
                        zx_graph,
                        graph_state,
                        editor_state.pipe_length,
                    )
                })
                .unwrap_or_default();
            editor_state.set_zx_hovered_elements(hovered);
            draw_revision_label(&painter, rect, graph_state.revision, palette);
        }
        Ok(ZxGraphView::Simplified(view)) => {
            draw_simplified_zx_graph(&painter, rect, zx_viewer, view, palette);
            draw_revision_label(&painter, rect, graph_state.revision, palette);
        }
        Ok(ZxGraphView::Cancelled) => {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "ZX simplification cancelled. Select Retry to run again.",
                egui::FontId::proportional(14.0),
                palette.text_dim,
            );
        }
        Err(err) => {
            ui.put(
                rect.shrink(12.0),
                egui::Label::new(
                    egui::RichText::new(format!("ZX view failed\n{err}"))
                        .monospace()
                        .color(palette.accent_error),
                )
                .wrap()
                .selectable(true),
            );
        }
    }
}

fn handle_camera_input(
    ui: &egui::Ui,
    rect: Rect,
    response: &egui::Response,
    zx_viewer: &mut ZxViewerState,
) {
    if zx_viewer.simplified {
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            zx_viewer.simplified_zoom =
                (zx_viewer.simplified_zoom + scroll * 0.002).clamp(-4.0, 4.0);
        }
        if response.dragged() {
            zx_viewer.simplified_pan += ui.input(|input| input.pointer.delta());
        }
        return;
    }
    if response.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        if scroll.abs() > f32::EPSILON {
            zoom_camera_settings(&mut zx_viewer.camera, scroll * 0.01);
        }
    }

    let delta = ui.input(|input| input.pointer.delta());
    if response.dragged_by(egui::PointerButton::Primary)
        || response.dragged_by(egui::PointerButton::Middle)
    {
        orbit_camera_settings(&mut zx_viewer.camera, Vec2::new(delta.x, delta.y));
    } else if response.dragged_by(egui::PointerButton::Secondary) {
        let transform = camera_transform_for_settings(&zx_viewer.camera);
        pan_camera_settings(
            &mut zx_viewer.camera,
            Vec2::new(delta.x, delta.y),
            rect.height(),
            *transform.right(),
            *transform.up(),
        );
    }
}

fn hovered_elements(
    pointer: Pos2,
    rect: Rect,
    zx_viewer: &ZxViewerState,
    zx: &ZXGraph,
    graph_state: &GraphState,
    pipe_length: f32,
) -> HashSet<GraphElement> {
    let scene = Projector::new(rect, zx_viewer, zx, pipe_length);
    let mut best_node = None;
    let mut best_node_distance = f32::INFINITY;
    for node in zx.nodes() {
        let node_pos = scene.project(world_pos(node.pos, pipe_length));
        let distance = node_pos.distance(pointer);
        if distance <= scene.node_radius(node.pos) + 3.0 && distance < best_node_distance {
            best_node = Some(node.pos);
            best_node_distance = distance;
        }
    }
    if let Some(pos) = best_node {
        return HashSet::from([GraphElement::Block(pos)]);
    }

    let mut best_edge = None;
    let mut best_edge_distance = f32::INFINITY;
    for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
        let a = zx.nodes()[edge.n1].pos;
        let b = zx.nodes()[edge.n2].pos;
        let a2 = scene.project(world_pos(a, pipe_length));
        let b2 = scene.project(world_pos(b, pipe_length));
        let distance = distance_to_segment(pointer, a2, b2);
        if distance <= 6.0 && distance < best_edge_distance {
            best_edge = source_pipe_for_zx_edge(a, b, graph_state);
            best_edge_distance = distance;
        }
    }
    best_edge
        .map(|element| HashSet::from([element]))
        .unwrap_or_default()
}

fn source_pipe_for_zx_edge(a: IVec3, b: IVec3, graph_state: &GraphState) -> Option<GraphElement> {
    let a_block = graph_state.graph.get_endpoint_block(a)?;
    let b_block = graph_state.graph.get_endpoint_block(b)?;
    if graph_state
        .graph
        .has_pipe_between(a_block.pos(), b_block.pos())
    {
        Some(GraphElement::Pipe(a_block.pos(), b_block.pos()).canonical())
    } else {
        None
    }
}

fn distance_to_segment(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 <= f32::EPSILON {
        return point.distance(a);
    }
    let ap = point - a;
    let t = (ap.dot(ab) / len2).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

fn draw_zx_graph(
    painter: &egui::Painter,
    rect: Rect,
    zx_viewer: &ZxViewerState,
    zx: &ZXGraph,
    pipe_length: f32,
    stabilizer: Option<&Stabilizer>,
    palette: &ThemePalette,
) {
    let scene = Projector::new(rect, zx_viewer, zx, pipe_length);
    let mut drawables = Vec::new();

    for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
        let a = zx.nodes()[edge.n1];
        let b = zx.nodes()[edge.n2];
        let pauli = stabilizer.and_then(|stabilizer| {
            stabilizer
                .interior_edges
                .get(&(a.pos, b.pos))
                .or_else(|| stabilizer.interior_edges.get(&(b.pos, a.pos)))
                .copied()
        });
        drawables.push(Drawable {
            depth: scene
                .depth(world_pos(a.pos, pipe_length))
                .min(scene.depth(world_pos(b.pos, pipe_length))),
            kind: DrawableKind::Edge {
                a: a.pos,
                b: b.pos,
                hadamard: edge.hadamard,
                stabilizer: pauli,
            },
        });
    }

    for node in zx.nodes() {
        drawables.push(Drawable {
            depth: scene.depth(world_pos(node.pos, pipe_length)),
            kind: DrawableKind::Node(*node),
        });
    }

    drawables.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    for drawable in drawables {
        match drawable.kind {
            DrawableKind::Edge {
                a,
                b,
                hadamard,
                stabilizer,
            } => draw_edge(painter, &scene, a, b, hadamard, stabilizer, palette),
            DrawableKind::Node(node) => draw_node(painter, &scene, node, palette),
        }
    }
}

fn draw_simplified_zx_graph(
    painter: &egui::Painter,
    rect: Rect,
    zx_viewer: &ZxViewerState,
    view: &SimplifiedZxView,
    palette: &ThemePalette,
) {
    let scale = view.scale(rect, zx_viewer);
    let project = |position: Pos2| {
        rect.center() + (position - view.bounds.center()) * scale + zx_viewer.simplified_pan
    };
    for edge in &view.edges {
        let projected = edge.points.map(project);
        // Dense ZX graphs need crossings. A background halo makes a crossing
        // visibly different from a spider/connection without changing topology.
        for stroke in [
            Stroke::new(EDGE_WIDTH + 3.0, palette.bg_dark),
            Stroke::new(EDGE_WIDTH, with_alpha(palette.text_bright, 190)),
        ] {
            painter.add(egui::epaint::QuadraticBezierShape::from_points_stroke(
                projected,
                false,
                Color32::TRANSPARENT,
                stroke,
            ));
        }
    }
    for edge in view.edges.iter().filter(|edge| edge.hadamard) {
        let marker = Rect::from_center_size(
            project(edge.marker),
            egui::vec2(2.0, 2.0) * HADAMARD_MARKER_HALF * scale.min(1.0),
        );
        painter.rect_filled(marker, 1.5, palette.yellow);
        painter.rect_stroke(
            marker,
            1.5,
            Stroke::new(1.0, palette.text_bright),
            egui::StrokeKind::Inside,
        );
    }
    for vertex in view.graph.vertices() {
        draw_simplified_node(
            painter,
            project(view.positions[&vertex]),
            12.0 * scale,
            view.graph.vertex_type(vertex),
            view.graph.phase(vertex),
            palette,
        );
    }
    // Labels retain the verifier's tensor-axis order even when crossing
    // minimization places boundaries in a different vertical order.
    for (name, vertices) in [("in", view.graph.inputs()), ("out", view.graph.outputs())] {
        for (index, vertex) in vertices.iter().enumerate() {
            let label = painter.layout_no_wrap(
                format!("{name} {index}"),
                egui::FontId::monospace((11.0 * scale).min(14.0)),
                palette.text_dim,
            );
            let label_rect = egui::Align2::CENTER_TOP.anchor_size(
                project(view.positions[vertex]) + egui::vec2(0.0, 17.0 * scale),
                label.size(),
            );
            // Labels can cross long wires even when the boundary spider is clear.
            painter.rect_filled(label_rect.expand(2.0), 0.0, palette.bg_dark);
            painter.galley(label_rect.min, label, palette.text_dim);
        }
    }
    if view.positions.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("Scalar {}", view.graph.scalar()),
            egui::FontId::monospace(14.0),
            palette.text_bright,
        );
    }
}

fn simplified_phase_text_color(fill: Color32) -> Color32 {
    let linear = egui::Rgba::from(fill);
    let luminance = 0.2126 * linear.r() + 0.7152 * linear.g() + 0.0722 * linear.b();
    // Black and white reach equal WCAG contrast at luminance 0.179.
    if luminance > 0.179 {
        Color32::BLACK
    } else {
        Color32::WHITE
    }
}

fn draw_plain_edge(
    painter: &egui::Painter,
    scene: &Projector,
    a: Vec3,
    b: Vec3,
    hadamard: bool,
    palette: &ThemePalette,
) -> [Pos2; 2] {
    let a2 = scene.project(a);
    let b2 = scene.project(b);
    let depth_alpha = scene.edge_alpha_world(a, b);
    painter.line_segment(
        [a2, b2],
        Stroke::new(
            EDGE_WIDTH,
            with_alpha(palette.text_bright, (220.0 * depth_alpha) as u8),
        ),
    );
    if hadamard {
        let marker = Rect::from_center_size(
            a2.lerp(b2, 0.5),
            egui::vec2(2.0, 2.0) * HADAMARD_MARKER_HALF,
        );
        painter.rect_filled(marker, 1.5, palette.yellow);
        painter.rect_stroke(
            marker,
            1.5,
            Stroke::new(1.0, palette.text_bright),
            egui::StrokeKind::Inside,
        );
    }
    [a2, b2]
}

fn draw_simplified_node(
    painter: &egui::Painter,
    pos: Pos2,
    radius: f32,
    kind: VType,
    phase: quizx::phase::Phase,
    palette: &ThemePalette,
) {
    let color = match kind {
        VType::X => pauli_basis_color(PauliBasis::X, palette),
        VType::Z => pauli_basis_color(PauliBasis::Z, palette),
        VType::H => palette.yellow,
        _ => palette.grey2,
    };
    draw_node_body(painter, pos, radius, &[color], palette);
    if phase.to_f64() != 0.0 {
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            format!("{phase}π"),
            egui::FontId::monospace((radius * 0.8).min(14.0)),
            simplified_phase_text_color(color),
        );
    }
}

fn draw_edge(
    painter: &egui::Painter,
    scene: &Projector,
    a: IVec3,
    b: IVec3,
    hadamard: bool,
    stabilizer: Option<Pauli>,
    palette: &ThemePalette,
) {
    let [a2, b2] = draw_plain_edge(
        painter,
        scene,
        scene.world_pos(a),
        scene.world_pos(b),
        hadamard,
        palette,
    );

    if let Some(pauli) = stabilizer {
        for support in pauli.iter_xz() {
            let offset = stabilizer_offset(a, b, support);
            let color = pauli_basis_color(pauli_basis_from_support(support), palette);
            painter.line_segment(
                [a2 + offset, b2 + offset],
                Stroke::new(STABILIZER_WIDTH, with_alpha(color, 230)),
            );
        }
    }
}

fn draw_node(painter: &egui::Painter, scene: &Projector, node: ZXNode, palette: &ThemePalette) {
    let pos = scene.project(scene.world_pos(node.pos));
    let radius = scene.node_radius(node.pos);
    let colors = node_colors(node.kind, palette);
    draw_node_body(painter, pos, radius, &colors, palette);
}

fn draw_node_body(
    painter: &egui::Painter,
    pos: Pos2,
    radius: f32,
    colors: &[Color32],
    palette: &ThemePalette,
) {
    if colors.len() == 1 {
        painter.circle_filled(pos, radius, colors[0]);
    } else {
        draw_split_sphere(painter, pos, radius, colors);
    }
    painter.circle_stroke(
        pos,
        radius,
        Stroke::new(1.0, with_alpha(palette.text_bright, 190)),
    );
    let highlight = pos + egui::vec2(-radius * 0.32, -radius * 0.32);
    painter.circle_filled(highlight, radius * 0.28, Color32::from_white_alpha(90));
}

fn draw_split_sphere(painter: &egui::Painter, center: Pos2, radius: f32, colors: &[Color32]) {
    let segments = colors.len().max(1);
    let step = std::f32::consts::TAU / segments as f32;
    for (index, color) in colors.iter().enumerate() {
        let start = index as f32 * step - std::f32::consts::FRAC_PI_2;
        let end = start + step;
        let mut points = Vec::with_capacity(34);
        points.push(center);
        for i in 0..=32 {
            let t = start + (end - start) * i as f32 / 32.0;
            points.push(center + egui::vec2(t.cos() * radius, t.sin() * radius));
        }
        painter.add(Shape::convex_polygon(points, *color, Stroke::NONE));
    }
}

fn draw_revision_label(painter: &egui::Painter, rect: Rect, revision: u64, palette: &ThemePalette) {
    painter.text(
        rect.left_bottom() + egui::vec2(8.0, -8.0),
        egui::Align2::LEFT_BOTTOM,
        format!("rev {revision}"),
        egui::FontId::monospace(11.0),
        palette.text_dim,
    );
}

fn stabilizer_offset(a: IVec3, b: IVec3, pauli: Pauli) -> EguiVec2 {
    let xz_sign = match pauli {
        Pauli::X => -1.0,
        Pauli::Z => 1.0,
        _ => return EguiVec2::ZERO,
    };
    if is_temporal_edge(a, b) {
        egui::vec2(xz_sign * 5.0, 0.0)
    } else {
        egui::vec2(0.0, xz_sign * 5.0)
    }
}

fn is_temporal_edge(a: IVec3, b: IVec3) -> bool {
    a.z != b.z
}

fn node_colors(kind: NodeKind, palette: &ThemePalette) -> Vec<Color32> {
    match kind {
        NodeKind::X => vec![pauli_basis_color(PauliBasis::X, palette)],
        NodeKind::Y => vec![pauli_basis_color(PauliBasis::Y, palette)],
        NodeKind::Z => vec![pauli_basis_color(PauliBasis::Z, palette)],
        NodeKind::Port => vec![palette.grey2],
        NodeKind::T => vec![Color32::from_rgb(126, 87, 194)],
        NodeKind::Selective(kind) => selective_colors(kind, palette),
    }
}

fn selective_colors(kind: SelectiveKind, palette: &ThemePalette) -> Vec<Color32> {
    let mut colors = vec![
        spider_color(kind.pauli_if_true(), palette),
        spider_color(kind.pauli_if_false(), palette),
    ];
    colors.dedup();
    colors
}

/// Colour of the spider a basis-`P` measurement fills a selective node with.
///
/// A selective node names measurement bases, but the diagram draws the spider
/// each becomes, which swaps X and Z (`filled_node_kind` in `bloq_graph`).
fn spider_color(measured: PauliBasis, palette: &ThemePalette) -> Color32 {
    let spider = match measured {
        PauliBasis::X => PauliBasis::Z,
        PauliBasis::Y => PauliBasis::Y,
        PauliBasis::Z => PauliBasis::X,
    };
    pauli_basis_color(spider, palette)
}

fn pauli_basis_from_support(pauli: Pauli) -> PauliBasis {
    match pauli {
        Pauli::X => PauliBasis::X,
        Pauli::Z | Pauli::I => PauliBasis::Z,
        Pauli::Y => PauliBasis::Y,
    }
}

fn world_pos(pos: IVec3, pipe_length: f32) -> Vec3 {
    graph_to_world(pos.as_vec3(), pipe_length)
}

struct Drawable {
    depth: f32,
    kind: DrawableKind,
}

enum DrawableKind {
    Edge {
        a: IVec3,
        b: IVec3,
        hadamard: bool,
        stabilizer: Option<Pauli>,
    },
    Node(ZXNode),
}

struct Projector {
    rect: Rect,
    center: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    scale: f32,
    focus_offset: Vec3,
    pipe_length: f32,
}

impl Projector {
    fn new(rect: Rect, zx_viewer: &ZxViewerState, zx: &ZXGraph, pipe_length: f32) -> Self {
        let (center, _) = zx_bounds(zx, pipe_length);
        Self::with_center(rect, zx_viewer, center, pipe_length)
    }

    fn with_center(rect: Rect, zx_viewer: &ZxViewerState, center: Vec3, pipe_length: f32) -> Self {
        let transform = camera_transform_for_settings(&zx_viewer.camera);
        let visible_height = 2.0 * zx_viewer.camera.radius * (CAMERA_DEFAULT_FOV_Y * 0.5).tan();
        let scale = rect.height().max(1.0) / visible_height.max(1.0);
        Self {
            rect,
            center,
            right: *transform.right(),
            up: *transform.up(),
            forward: *transform.forward(),
            scale,
            focus_offset: zx_viewer.camera.focus,
            pipe_length,
        }
    }

    fn world_pos(&self, pos: IVec3) -> Vec3 {
        world_pos(pos, self.pipe_length)
    }

    fn project(&self, pos: Vec3) -> Pos2 {
        let relative = pos - self.center - self.focus_offset;
        let x = relative.dot(self.right);
        let y = relative.dot(self.up);
        self.rect.center() + egui::vec2(x, -y) * self.scale
    }

    fn depth(&self, pos: Vec3) -> f32 {
        (pos - self.center - self.focus_offset).dot(self.forward)
    }

    fn node_radius(&self, pos: IVec3) -> f32 {
        self.node_radius_world(self.world_pos(pos))
    }

    fn node_radius_world(&self, pos: Vec3) -> f32 {
        let depth = self.depth(pos);
        (NODE_RADIUS * self.scale * (1.0 + depth * 0.015)).clamp(5.0, 16.0)
    }

    fn edge_alpha_world(&self, a: Vec3, b: Vec3) -> f32 {
        let depth = self.depth(a).min(self.depth(b)).clamp(-10.0, 10.0);
        (0.75 + depth * 0.015).clamp(0.45, 1.0)
    }
}

fn zx_bounds(zx: &ZXGraph, pipe_length: f32) -> (Vec3, f32) {
    if zx.nodes().is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut min = world_pos(zx.nodes()[0].pos, pipe_length);
    let mut max = min;
    for node in zx.nodes() {
        let pos = world_pos(node.pos, pipe_length);
        min = min.min(pos);
        max = max.max(pos);
    }
    let center = (min + max) * 0.5;
    let extent = (max - min).length().max(1.0);
    (center, extent)
}

#[cfg(test)]
mod tests {
    use super::{
        SimplifiedZxView, distance_to_segment, quadratic_point, route_simplified_edge,
        selective_colors, simplified_zx_graph, source_pipe_for_zx_edge, stabilizer_offset,
        zx_bounds,
    };
    use crate::components::GraphElement;
    use crate::resources::GraphState;
    use crate::theme::{ThemePreset, palette, pauli_basis_color};
    use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe};
    use bloq_graph::{PauliBasis, SelectiveKind};
    use glam::IVec3;
    use quizx::graph::{GraphLike, VType};

    fn dense_graph(count: usize) -> bloq_graph::verify::QuizxGraph {
        let mut graph = bloq_graph::verify::QuizxGraph::new();
        let vertices: Vec<_> = (0..count).map(|_| graph.add_vertex(VType::Z)).collect();
        for (i, &a) in vertices.iter().enumerate() {
            for &b in &vertices[i + 1..] {
                graph.add_edge_with_type(a, b, quizx::graph::EType::H);
            }
        }
        graph
    }

    fn minimum_edge_clearance(view: &SimplifiedZxView) -> f32 {
        let mut clearance = f32::INFINITY;
        for edge in &view.edges {
            for (&vertex, &position) in &view.positions {
                if edge.vertices.contains(&vertex) {
                    continue;
                }
                for step in 0..100 {
                    clearance = clearance.min(distance_to_segment(
                        position,
                        quadratic_point(edge.points, step as f32 / 100.0),
                        quadratic_point(edge.points, (step + 1) as f32 / 100.0),
                    ));
                }
            }
        }
        clearance
    }

    #[test]
    fn simplified_layout_is_deterministic_and_preserves_the_signed_diagram() {
        use quizx::tensor::ToTensor;

        let mut graph = bloq_graph::verify::QuizxGraph::new();
        let input = graph.add_vertex(VType::B);
        let first = graph.add_vertex_with_phase(VType::Z, (1, 4));
        let second = graph.add_vertex(VType::X);
        let third = graph.add_vertex(VType::Z);
        let output = graph.add_vertex(VType::B);
        let isolated = graph.add_vertex_with_phase(VType::Z, (1, 2));
        graph.add_edge(input, first);
        graph.add_edge(first, second);
        graph.add_edge(second, third);
        graph.add_edge(third, output);
        graph.add_edge(first, third);
        graph.set_inputs(vec![input]);
        graph.set_outputs(vec![output]);
        graph.scalar_mut().mul_phase(1);
        let mut cases = vec![("coincident".to_owned(), graph)];
        cases.push(("dense_k8".to_owned(), dense_graph(8)));
        for (name, source) in [
            ("cnot", bloq_graph::GalleryItem::CNOT.build()),
            ("three_cnots", bloq_graph::GalleryItem::ThreeCNOTs.build()),
            ("t_gate", bloq_graph::GalleryItem::T.build()),
            ("and_4t", bloq_graph::GalleryItem::And4T.build()),
            ("uma", bloq_graph::GalleryItem::UMA.build()),
        ] {
            cases.push((
                name.to_owned(),
                simplified_zx_graph(&source.flatten().unwrap(), 0).unwrap(),
            ));
        }
        for (name, graph) in cases {
            let tensor = graph.to_tensorf();
            let encoded: serde_json::Value =
                serde_json::from_str(&quizx::json::encode_graph(&graph).unwrap()).unwrap();
            let view = SimplifiedZxView::new(graph.clone());
            let repeat = SimplifiedZxView::new(graph);

            assert_eq!(view.positions, repeat.positions, "{name}");
            assert_eq!(view.edges, repeat.edges, "{name}");
            assert_eq!(view.positions.len(), view.graph.num_vertices(), "{name}");
            if name == "coincident" {
                assert_eq!(view.positions.len(), 6);
                assert!(view.bounds.contains(view.positions[&isolated]));
                assert_eq!(view.graph.inputs(), &[input]);
                assert_eq!(view.graph.outputs(), &[output]);
            }
            for (&a, &lhs) in &view.positions {
                for (&b, &rhs) in &view.positions {
                    if a != b {
                        assert!(
                            lhs.distance(rhs) >= 32.0,
                            "{name}: spiders {a} and {b} overlap"
                        );
                    }
                }
            }
            assert!(
                minimum_edge_clearance(&view) >= 20.0,
                "{name}: edge occludes an unrelated spider"
            );
            let actual: serde_json::Value =
                serde_json::from_str(&quizx::json::encode_graph(&view.graph).unwrap()).unwrap();
            assert_eq!(actual, encoded, "{name}");
            assert_eq!(view.graph.to_tensorf(), tensor, "{name}");
        }
    }

    #[test]
    fn simplified_long_edge_bends_around_unrelated_spider() {
        use bevy_egui::egui::Pos2;

        let positions = [
            (0, Pos2::new(0.0, 0.0)),
            (1, Pos2::new(60.0, 0.0)),
            (2, Pos2::new(120.0, 0.0)),
        ]
        .into();
        let curve = route_simplified_edge(0, 2, &positions, &[]);
        assert_eq!(curve[0], positions[&0]);
        assert_eq!(curve[2], positions[&2]);
        for step in 0..100 {
            let clearance = distance_to_segment(
                positions[&1],
                quadratic_point(curve, step as f32 / 100.0),
                quadratic_point(curve, (step + 1) as f32 / 100.0),
            );
            assert!(clearance >= 20.0, "edge passes through unrelated spider");
        }
    }

    #[test]
    fn simplified_routes_separate_collinear_wires_and_hadamard_boxes() {
        use bevy_egui::egui::Pos2;

        let positions: std::collections::HashMap<usize, Pos2> = [
            (0, Pos2::new(0.0, 0.0)),
            (1, Pos2::new(300.0, 0.0)),
            (2, Pos2::new(100.0, 0.0)),
            (3, Pos2::new(200.0, 0.0)),
        ]
        .into();
        let existing =
            super::sample_simplified_edge([positions[&0], Pos2::new(150.0, 0.0), positions[&1]]);
        let curve = route_simplified_edge(2, 3, &positions, &[existing]);
        assert_ne!(
            curve[1].y, 0.0,
            "nested collinear wires must not hide each other"
        );

        let edge = super::SimplifiedEdge {
            vertices: [2, 3],
            points: curve,
            hadamard: true,
            marker: quadratic_point(curve, 0.5),
        };
        let marker = super::simplified_hadamard_position(
            curve,
            &positions,
            std::slice::from_ref(&edge),
            &[],
        );
        assert!(
            marker.distance(edge.marker) >= 12.0,
            "Hadamard boxes must remain distinct"
        );
        assert!(
            positions
                .values()
                .all(|&position| marker.distance(position) >= 25.0)
        );
    }

    #[test]
    fn simplified_phase_labels_have_readable_contrast_in_both_themes() {
        use bevy_egui::egui::{self, Color32, Pos2, Shape};

        fn luminance(color: Color32) -> f32 {
            let channel = |value: u8| {
                let value = f32::from(value) / 255.0;
                if value <= 0.04045 {
                    value / 12.92
                } else {
                    ((value + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
        }
        for theme in [ThemePreset::Light, ThemePreset::GruvboxMaterial] {
            for kind in [VType::X, VType::Z, VType::H, VType::B] {
                let ctx = egui::Context::default();
                let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                    let painter = ui.ctx().layer_painter(egui::LayerId::background());
                    super::draw_simplified_node(
                        &painter,
                        Pos2::new(40.0, 40.0),
                        18.0,
                        kind,
                        (1, 4).into(),
                        palette(theme),
                    );
                });
                output.textures_delta.clear();
                let fill = output
                    .shapes
                    .iter()
                    .find_map(|shape| match &shape.shape {
                        Shape::Circle(circle)
                            if circle.radius == 18.0 && circle.fill != Color32::TRANSPARENT =>
                        {
                            Some(circle.fill)
                        }
                        _ => None,
                    })
                    .unwrap();
                let text = output
                    .shapes
                    .iter()
                    .find_map(|shape| match &shape.shape {
                        Shape::Text(text) => Some(
                            text.override_text_color
                                .unwrap_or(text.galley.job.sections[0].format.color),
                        ),
                        _ => None,
                    })
                    .unwrap();
                let lighter = luminance(fill).max(luminance(text));
                let darker = luminance(fill).min(luminance(text));
                assert!(
                    (lighter + 0.05) / (darker + 0.05) >= 4.5,
                    "{theme:?} {kind:?}: phase text has low contrast"
                );
            }
        }
    }

    #[test]
    fn worker_view_codec_preserves_prepared_geometry_and_rejects_incomplete_results() {
        use quizx::tensor::ToTensor;

        let mut sparse = bloq_graph::verify::QuizxGraph::new();
        let removed = sparse.add_vertex(VType::Z);
        let input = sparse.add_vertex(VType::B);
        let z = sparse.add_vertex_with_phase(VType::Z, (1, 4));
        let x = sparse.add_vertex(VType::X);
        let output = sparse.add_vertex(VType::B);
        sparse.add_edge_with_type(input, z, quizx::graph::EType::H);
        sparse.add_edge(z, x);
        sparse.add_edge_with_type(x, output, quizx::graph::EType::H);
        sparse.set_inputs(vec![input]);
        sparse.set_outputs(vec![output]);
        sparse.remove_vertex(removed);
        let mut cases = vec![("renumbered".to_owned(), sparse)];
        for (name, source) in [
            ("t_gate", bloq_graph::GalleryItem::T.build()),
            ("uma", bloq_graph::GalleryItem::UMA.build()),
        ] {
            cases.push((
                name.to_owned(),
                simplified_zx_graph(&source.flatten().unwrap(), 0).unwrap(),
            ));
        }
        for (name, mut graph) in cases {
            graph.scalar_mut().mul_phase(1);
            let view = SimplifiedZxView::new(graph);
            let encoded = super::encode_zx_worker_view(&view).unwrap();
            let decoded = super::decode_zx_worker_view(&encoded).unwrap();
            let routes = |view: &SimplifiedZxView| {
                view.edges
                    .iter()
                    .map(|edge| (edge.points, edge.hadamard, edge.marker))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                routes(&decoded),
                routes(&view),
                "{name}: prepared routes changed"
            );
            assert_eq!(decoded.bounds, view.bounds);
            assert_eq!(decoded.graph.scalar(), view.graph.scalar());
            assert_eq!(decoded.graph.to_tensorf(), view.graph.to_tensorf());
            let boundary_positions = |view: &SimplifiedZxView| {
                view.graph
                    .inputs()
                    .iter()
                    .chain(view.graph.outputs())
                    .map(|vertex| view.positions[vertex])
                    .collect::<Vec<_>>()
            };
            assert_eq!(boundary_positions(&decoded), boundary_positions(&view));
            let repeated =
                super::decode_zx_worker_view(&super::encode_zx_worker_view(&decoded).unwrap())
                    .unwrap();
            assert_eq!(
                routes(&repeated),
                routes(&view),
                "{name}: re-encoded routes changed"
            );
            assert_eq!(boundary_positions(&repeated), boundary_positions(&view));
            assert_eq!(repeated.graph.to_tensorf(), view.graph.to_tensorf());
            let mut original_coords: Vec<_> = view
                .graph
                .vertices()
                .map(|v| {
                    let c = view.graph.coord(v);
                    (c.x, c.y)
                })
                .collect();
            let mut decoded_coords: Vec<_> = decoded
                .graph
                .vertices()
                .map(|v| {
                    let c = decoded.graph.coord(v);
                    (c.x, c.y)
                })
                .collect();
            let sort = |a: &(f64, f64), b: &(f64, f64)| {
                a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
            };
            original_coords.sort_by(sort);
            decoded_coords.sort_by(sort);
            assert_eq!(original_coords, decoded_coords);
            let mut damaged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            damaged["vertices"].as_array_mut().unwrap().pop();
            assert!(super::decode_zx_worker_view(&damaged.to_string()).is_err());
            let mut damaged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            damaged["vertices"][0][1][0] = serde_json::json!(1.0e39);
            assert!(super::decode_zx_worker_view(&damaged.to_string()).is_err());
            let mut damaged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            damaged["edges"][0][0] = serde_json::json!(usize::MAX);
            assert!(super::decode_zx_worker_view(&damaged.to_string()).is_err());
        }
    }

    /// Reusable CPU-frame probe; the SVG uses the same painter as the viewer.
    #[test]
    #[ignore = "manual layout/frame timings and optional SVG previews"]
    fn measure_simplified_layout_and_cached_frames() {
        use bevy_egui::egui::{self, Pos2, Rect};
        use std::hint::black_box;
        use std::time::Instant;

        let mut cases = vec![("dense_k12".to_owned(), dense_graph(12))];
        for (name, source) in [
            ("cnot", bloq_graph::GalleryItem::CNOT.build()),
            ("and_4t", bloq_graph::GalleryItem::And4T.build()),
            ("uma", bloq_graph::GalleryItem::UMA.build()),
        ] {
            let started = Instant::now();
            let graph = simplified_zx_graph(&source.flatten().unwrap(), 0).unwrap();
            println!(
                "{}: simplify {:.3} ms",
                name,
                started.elapsed().as_secs_f64() * 1000.0
            );
            cases.push((name.to_owned(), graph));
        }
        let size = egui::vec2(1200.0, 800.0);
        let rect = Rect::from_min_size(Pos2::ZERO, size);
        let palette = palette(ThemePreset::default());
        for (name, graph) in cases {
            let started = Instant::now();
            let view = SimplifiedZxView::new(graph);
            let layout_ms = started.elapsed().as_secs_f64() * 1000.0;
            let clearance = minimum_edge_clearance(&view);
            assert!(
                clearance >= 20.0,
                "{name}: a wire occludes an unrelated spider"
            );
            let started = Instant::now();
            let transport = super::encode_zx_worker_view(&view).unwrap();
            let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut decode_ms = Vec::new();
            for _ in 0..20 {
                let started = Instant::now();
                black_box(super::decode_zx_worker_view(&transport).unwrap());
                decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            decode_ms.sort_by(f64::total_cmp);
            let viewer = super::ZxViewerState::default();
            let ctx = egui::Context::default();
            crate::theme::setup_export_context(&ctx);
            let raw_input = egui::RawInput {
                screen_rect: Some(rect),
                ..Default::default()
            };
            let mut frame_ms = Vec::new();
            for index in 0..101 {
                let started = Instant::now();
                let mut output = ctx.run_ui(raw_input.clone(), |ui| {
                    let painter = ui.ctx().layer_painter(egui::LayerId::background());
                    painter.rect_filled(rect, 0.0, palette.bg_dark);
                    super::draw_simplified_zx_graph(&painter, rect, &viewer, &view, palette);
                });
                output.textures_delta.clear();
                black_box(ctx.tessellate(output.shapes, output.pixels_per_point));
                if index > 0 {
                    frame_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            frame_ms.sort_by(f64::total_cmp);
            println!(
                "{name}: {} nodes, {} edges, layout {layout_ms:.3} ms, clearance {clearance:.2}, cached CPU frame p50 {:.3} ms, p95 {:.3} ms",
                view.graph.num_vertices(),
                view.graph.num_edges(),
                frame_ms[50],
                frame_ms[95]
            );
            println!(
                "{name}: Worker geometry encode {encode_ms:.3} ms, UI decode p50 {:.3} ms, p95 {:.3} ms",
                decode_ms[10], decode_ms[19]
            );
            if let Ok(directory) = std::env::var("BLOQ_ZX_PREVIEW_DIR") {
                let svg = crate::svg_export::render_svg(size, |painter| {
                    painter.rect_filled(rect, 0.0, palette.bg_dark);
                    super::draw_simplified_zx_graph(painter, rect, &viewer, &view, palette);
                });
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    std::path::Path::new(&directory).join(format!("{name}.svg")),
                    svg,
                )
                .unwrap();
            }
        }
    }

    /// A selective node draws the spider each measurement becomes, so an `XY`
    /// site is Z-blue / Y-green rather than the X-red / Y-green of its raw bases.
    #[test]
    fn selective_node_colors_use_the_filled_spider_not_the_measured_basis() {
        let palette = palette(ThemePreset::default());

        let colors = selective_colors(SelectiveKind::XY, palette);

        assert_eq!(
            colors,
            vec![
                pauli_basis_color(PauliBasis::Z, palette),
                pauli_basis_color(PauliBasis::Y, palette),
            ]
        );
        assert_eq!(
            selective_colors(SelectiveKind::XZ, palette),
            vec![
                pauli_basis_color(PauliBasis::Z, palette),
                pauli_basis_color(PauliBasis::X, palette),
            ]
        );
    }

    #[test]
    fn zx_bounds_cover_all_node_positions() {
        let mut graph = BlockGraph::default();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::new(2, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        let zx = graph.to_zx_graph().expect("graph converts to ZX");

        let (center, extent) = zx_bounds(&zx, 0.0);

        assert_eq!(center.x, 1.0);
        assert!(extent >= 2.0);
    }

    #[test]
    fn stabilizer_offsets_use_spatial_top_bottom_and_temporal_left_right() {
        let spatial_x = stabilizer_offset(IVec3::ZERO, IVec3::X, bloq_graph::Pauli::X);
        let spatial_z = stabilizer_offset(IVec3::ZERO, IVec3::X, bloq_graph::Pauli::Z);
        assert_eq!(spatial_x.x, 0.0);
        assert!(spatial_x.y < 0.0);
        assert_eq!(spatial_z.x, 0.0);
        assert!(spatial_z.y > 0.0);

        let temporal_x = stabilizer_offset(IVec3::ZERO, IVec3::Z, bloq_graph::Pauli::X);
        let temporal_z = stabilizer_offset(IVec3::ZERO, IVec3::Z, bloq_graph::Pauli::Z);
        assert!(temporal_x.x < 0.0);
        assert_eq!(temporal_x.y, 0.0);
        assert!(temporal_z.x > 0.0);
        assert_eq!(temporal_z.y, 0.0);
    }

    #[test]
    fn zx_edge_hover_maps_to_source_pipe() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let zx = graph_state
            .graph
            .to_zx_graph()
            .expect("graph converts to ZX");
        let edge = zx.edges().iter().find(|edge| edge.n1 < edge.n2).unwrap();
        let a = zx.nodes()[edge.n1].pos;
        let b = zx.nodes()[edge.n2].pos;

        assert_eq!(
            source_pipe_for_zx_edge(a, b, &graph_state),
            Some(GraphElement::Pipe(IVec3::ZERO, IVec3::X).canonical())
        );
    }

    #[test]
    fn full_simplify_presamples_dynamic_measurements() {
        let source = BlockGraph::from_blog_text(
            r#"BLOG 1.0

  0: ZXZ [0,0,0]

  m = measure 0
"#,
        )
        .unwrap();

        let mut graph = simplified_zx_graph(&source, 0).unwrap();

        assert!(
            graph.vertices().all(|vertex| graph.vars(vertex).is_empty()),
            "the sampled QuiZX graph must contain no symbolic measurement variables"
        );
        assert!(
            !quizx::simplify::full_simp(&mut graph),
            "the displayed graph must already be fully simplified"
        );
    }

    #[test]
    fn simplified_results_only_update_the_requested_open_view() {
        use super::ZxViewerState;

        let mut viewer = ZxViewerState {
            open: true,
            simplified: true,
            sample_seed: 4,
            ..Default::default()
        };
        let key = viewer.view_key(7);
        assert!(!viewer.finish_simplification(8, key, Err("old revision".into())));
        viewer.resample();
        assert!(!viewer.finish_simplification(7, key, Err("old sample".into())));
        viewer.sample_seed = 4;
        viewer.set_simplified(false);
        assert!(!viewer.finish_simplification(7, key, Err("old mode".into())));
        viewer.set_simplified(true);
        viewer.toggle();
        assert!(!viewer.finish_simplification(7, key, Err("closed".into())));
        assert!(viewer.cached_view.is_none());
        viewer.toggle();
        assert!(viewer.finish_simplification(7, key, Err("current error".into())));
        assert!(matches!(viewer.cached_view, Some(Err(ref error)) if error == "current error"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn full_simplify_enters_pending_without_building_on_the_ui_thread() {
        use super::ZxViewerState;
        use crate::resources::EditorTabId;
        use crate::systems::jobs::EditorJobs;
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};

        AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let graph = GraphState {
            graph: bloq_graph::GalleryItem::CNOT.build(),
            ..Default::default()
        };
        let mut viewer = ZxViewerState {
            open: true,
            simplified: true,
            ..Default::default()
        };
        let mut jobs = EditorJobs::default();
        viewer.refresh_view(&graph, EditorTabId::new(1), &mut jobs);
        assert!(jobs.is_busy());
        assert!(!jobs.compilation_running());
        assert!(
            viewer.cached_view.is_none(),
            "results arrive through job polling"
        );
        viewer.set_simplified(false);
        viewer.refresh_view(&graph, EditorTabId::new(1), &mut jobs);
        assert!(matches!(
            viewer.cached_view,
            Some(Ok(super::ZxGraphView::Source(_)))
        ));
    }

    #[test]
    fn worker_codec_preserves_external_feedback_resources_and_signed_diagrams() {
        use super::{decode_zx_worker_source, encode_zx_worker_source};
        use bloq_graph::{Action, Expr, FeedbackTarget, GalleryItem};
        use quizx::tensor::ToTensor;

        let mut external = BlockGraph::from_blog_text(
            "BLOG 1.0\n0: Port [0,0,0]\n1: ZXZ [0,0,1]\n2: Port [0,0,2]\n0 -> +Z\n1 -> +Z\n",
        )
        .unwrap();
        external
            .set_actions_with_inputs(
                vec![Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::X,
                        target: IVec3::Z,
                        direction: None,
                    }],
                    condition: Some(Expr::Var("enabled".into())),
                }],
                ["enabled".into()],
            )
            .unwrap();
        for source in [external, GalleryItem::T.build()] {
            let restored =
                decode_zx_worker_source(&encode_zx_worker_source(&source).unwrap()).unwrap();
            assert_eq!(source.to_blog_body_text(), restored.to_blog_body_text());
            assert_eq!(
                source.action_graph().inputs().collect::<Vec<_>>(),
                restored.action_graph().inputs().collect::<Vec<_>>()
            );
            for seed in [0, 1, u64::MAX] {
                let expected = simplified_zx_graph(&source, seed).unwrap();
                let actual = simplified_zx_graph(&restored, seed).unwrap();
                assert_eq!(expected.to_tensorf(), actual.to_tensorf());
            }
        }
    }

    #[test]
    fn collapsed_viewer_clears_source_hover_without_building_its_graph() {
        use super::{ZxViewerState, draw_zx_viewer};
        use crate::resources::EditorState;
        use bevy_egui::egui;

        let ctx = egui::Context::default();
        let collapsed = egui::collapsing_header::CollapsingState::load_with_default_open(
            &ctx,
            egui::Id::new(Some("ZX View")).with("collapsing"),
            false,
        );
        collapsed.store(&ctx);
        let graph = GraphState::default();
        let mut editor = EditorState::default();
        editor.set_zx_hovered_elements([GraphElement::Block(IVec3::ZERO)].into());
        let mut viewer = ZxViewerState {
            open: true,
            ..Default::default()
        };
        ctx.run_ui(Default::default(), |ui| {
            draw_zx_viewer(
                ui.ctx(),
                ui.ctx().content_rect(),
                &graph,
                &mut editor,
                &mut viewer,
                crate::resources::EditorTabId::new(1),
                &mut crate::systems::jobs::EditorJobs::default(),
            );
        })
        .drop_without_applying_deltas();

        assert!(editor.zx_hovered_element_set().is_empty());
        assert!(viewer.cached_view.is_none());
    }
}
