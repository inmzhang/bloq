//! The Bloq viewer window: a layered graph of the compiled
//! program and a per-node circuit timeline, with hover/selection that
//! cross-highlights back to the source graph.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::{UiIntent, UiIntentBuffer, activated, tiled_window_rect};
use crate::components::GraphElement;
use crate::resources::{CompileUiState, EditorMode, EditorState, GraphState};
use crate::systems::jobs::EditorJobs;
use crate::theme::{self, ThemePalette, ThemePreset};
use bevy::prelude::Resource;
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke};
use bloq_circuit::{DetectorCoords, GateType, PauliBasis, RegionTerm};
use bloq_compile::CompileConfig;
use bloq_graph::BlockGraph;
use bloq_ir::{Bloq, MomentKind};
use glam::{IVec2, IVec3, Vec2};

mod detslice_draw;
mod geometry;

use detslice_draw::{DetsliceColors, draw_detector_slices};
use geometry::{CanvasGeometry, circuit_origin, coord_bounds, fitted_pitch, qubit_pos};

// ============================================================================
// Circuit/graph-viewer presentation model
// ============================================================================
//
// The compiled-program view the viewer window renders: flattened moments,
// node/edge views, the layered layout cache, and the detector-slice state
// machine on `BloqViewerState`. Re-exported from `resources` for other readers.

/// One time step of a node's circuit, flattened for the circuit viewer.
///
/// `repeat_label`/`repeat_span` mark the first moment of a `REPEAT` block and
/// how many moments it spans, so the viewer can bracket the loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlatMoment {
    pub(crate) kind: MomentKind,
    pub(crate) ops: Vec<FlatMomentOp>,
    pub(crate) repeat_label: Option<String>,
    pub(crate) repeat_span: Option<usize>,
}

/// A single operation within a [`FlatMoment`], with qubits given as viewer
/// qubit indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FlatMomentOp {
    Gate {
        gate: GateType,
        qubits: Vec<u32>,
    },
    Measure {
        basis: PauliBasis,
        qubits: Vec<u32>,
    },
    /// A multi-Pauli-product measurement (Stim `MPP`). Each product is the
    /// `(viewer qubit index, basis)` support of one jointly-measured Pauli
    /// product; a `Port` block's boundary stabilizers land here.
    Mpp {
        products: Vec<Vec<(u32, PauliBasis)>>,
    },
}

/// Stable identity of one detector-slice overlay region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SliceRegionId {
    Detector { owner_node: u32, detector: u32 },
    Observable { index: u32 },
}

/// Which parts of the shared detector/observable slice cache are visible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SliceVisibility {
    pub(crate) detectors: bool,
    pub(crate) observables: bool,
}

impl SliceVisibility {
    pub(crate) fn any(self) -> bool {
        self.detectors || self.observables
    }

    pub(crate) fn shows(self, id: SliceRegionId) -> bool {
        match id {
            SliceRegionId::Detector { .. } => self.detectors,
            SliceRegionId::Observable { .. } => self.observables,
        }
    }
}

/// One detector or logical-observable region as the overlay draws it, captured
/// at the end of a single flattened moment. Terms carry grid coordinates (not
/// qubit indices) so the renderer can project them directly and show a region's
/// full extent even where it straddles a node seam.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SliceRegionView {
    pub(crate) id: SliceRegionId,
    /// Stim `DETECTOR` coordinates, when the program carried them.
    pub(crate) coords: Option<Arc<DetectorCoords>>,
    /// Per-qubit Pauli support, sorted by grid coordinate for determinism.
    pub(crate) terms: Vec<RegionTerm>,
}

impl SliceRegionView {
    pub(crate) fn id(&self) -> SliceRegionId {
        self.id
    }
}

/// A node's detector-slice timeline: `[moment][region]`, aligned 1:1 with the
/// node's flattened moment list (index `k` is the state at the end of flat
/// moment `k`).
pub(crate) type NodeSliceTimeline = Vec<Vec<SliceRegionView>>;

/// Anticommutation breaks the tracker hit, keyed by `(node view id, moment)`;
/// the value is the grid coordinates where a region broke at that moment. The
/// overlay draws a marker on each. Compiled programs are break-free, so a
/// non-empty map signals a non-Clifford axis or a region seam.
pub(crate) type DetectorBreaks = HashMap<(u32, usize), Vec<IVec2>>;

/// Broad node class used to colour and label a tile in the Bloq graph view.
/// Quantum nodes carry a circuit; the rest are classical readout/dataflow nodes
/// injected by observable lowering and hidden unless the user toggles them on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BloqNodeCategory {
    QuantumBlock,
    QuantumPipe,
    Observable,
    /// A control-flow region ([`bloq_ir::RegionNode`]); drawn as a
    /// container box holding its body's node tiles.
    Region,
    Other,
}

impl BloqNodeCategory {
    pub(crate) fn is_quantum(self) -> bool {
        matches!(self, Self::QuantumBlock | Self::QuantumPipe)
    }
}

/// The Bloq edge kind, mirrored from `bloq_ir::BloqEdge` so the view can
/// colour seams, dataflow, and ordering edges differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BloqEdgeKind {
    Quantum,
    Value,
    Flip,
    Compose,
    Order,
}

/// One node of the compiled Bloq program, prepared for display in the graph
/// and circuit viewers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BloqNodeView {
    pub(crate) id: u32,
    pub(crate) layer: i64,
    pub(crate) label: String,
    pub(crate) category: BloqNodeCategory,
    /// Human-readable rows for nodes without a fixed circuit, including
    /// conditional quantum components.
    pub(crate) attributes: Vec<(String, String)>,
    /// An `Observable` node's bound boundary operators as `(instance, face,
    /// operator)` rows, rendered as a table below `attributes`. A merged
    /// `Observable` may bind several; empty for every other node kind.
    pub(crate) operator_table: Vec<(String, String, String)>,
    pub(crate) source_elements: HashSet<GraphElement>,
    pub(crate) moments: Vec<FlatMoment>,
    pub(crate) qubit_coords: HashMap<usize, IVec2>,
    pub(crate) num_qubits: usize,
    /// The enclosing region's view id for nodes lifted out of a region body;
    /// `None` for top-level nodes.
    pub(crate) parent: Option<u32>,
    /// For a region: whether its body, recursively, holds any quantum node.
    /// Always `false` for non-region nodes.
    pub(crate) has_quantum_body: bool,
}

impl BloqNodeView {
    /// Whether the node surfaces in the default (classical toggle off) view:
    /// quantum nodes, and regions whose body recursively holds quantum content.
    pub(crate) fn quantum_visible(&self) -> bool {
        self.category.is_quantum() || self.has_quantum_body
    }
}

/// Concurrent circuit for one source z layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerCircuitView {
    pub(crate) moments: Vec<FlatMoment>,
    /// `(node view id, flattened moment)` contributors for each moment.
    pub(crate) source_moments: Vec<Vec<(u32, usize)>>,
    pub(crate) qubit_coords: HashMap<usize, IVec2>,
    pub(crate) num_qubits: usize,
}

/// One edge between two [`BloqNodeView`]s, by node id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BloqEdgeView {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) kind: BloqEdgeKind,
    /// Whether the underlying Bloq edge carries any temporal pipes.
    pub(crate) has_pipes: bool,
}

/// Memoized graph-view layout. The layered pass is expensive and its result
/// depends only on the node/edge set and the classical toggle — not on zoom or
/// pan (those scale/translate linearly) — so it is computed once per
/// `(compile_revision, show_classical)` and reused every frame.
#[derive(Debug, Clone, Default)]
pub(crate) struct GraphLayoutCache {
    /// The `(compile_revision, show_classical)` the positions were computed for.
    pub(crate) key: Option<(Option<u64>, bool)>,
    /// Node top-left positions at zoom 1.0, normalized so the bounding box
    /// starts at `(0, 0)`. Scaled by zoom and offset by pan at draw time.
    pub(crate) positions: HashMap<u32, (f32, f32)>,
    /// Per-node tile sizes at zoom 1.0. Region containers grow to fit their
    /// children; nodes absent here use the default tile size.
    pub(crate) sizes: HashMap<u32, (f32, f32)>,
    /// Routed splines at zoom 1.0, keyed by the original viewer edge index.
    pub(crate) routes: HashMap<usize, super::graph_layout::CubicSpline>,
    /// Bounding-box size at zoom 1.0.
    pub(crate) content: (f32, f32),
}

/// The output of a viewer compile job: the normalized graph that was compiled,
/// its node/edge views, and compile metadata.
///
/// Detector slices and concurrent ops build lazily from `program`.
#[derive(Debug, Clone)]
pub(crate) struct CompiledBloqView {
    pub(crate) viewer_graph: BlockGraph,
    pub(crate) nodes: Vec<BloqNodeView>,
    pub(crate) edges: Vec<BloqEdgeView>,
    pub(crate) code_distance: u32,
    /// Exact lowering settings reused by the lazy Clifford-proxy compile.
    pub(crate) compile_config: CompileConfig,
    pub(crate) compile_duration: Duration,
    /// Retained input for lazy viewer jobs; `Arc` keeps task cloning cheap.
    pub(crate) program: Arc<Bloq>,
    /// Original compilation retained while the viewer pins source choices.
    pub(crate) source_program: Arc<Bloq>,
    pub(crate) branch_pins: Option<BTreeMap<String, bool>>,
    pub(crate) branch_pin_draft: BTreeMap<String, bool>,
    /// Restores normalized compiler coordinates to editor source layers.
    pub(crate) source_offset: IVec3,
}

/// Inputs for the lazy concurrent-ops job.
pub(crate) struct ConcurrentOpsInputs {
    pub(crate) program: Arc<Bloq>,
    pub(crate) viewer_graph: BlockGraph,
    pub(crate) source_offset: IVec3,
    pub(crate) generation: u64,
}

/// The lazily-computed detector-slice overlay data, produced off a
/// [`CompiledBloqView`] the first time the overlay is enabled and cached on
/// [`BloqViewerState`] until the next recompile.
#[derive(Debug, Clone, Default)]
pub(crate) struct DetsliceData {
    /// Per-node flattened (repeats unrolled) moment timelines, keyed by node
    /// view id. The detector-slice mode substitutes these for the default
    /// `BloqNodeView::moments`. Empty when flattening failed.
    pub(crate) flat_moments: HashMap<u32, Vec<FlatMoment>>,
    /// Per-node detector-slice regions aligned with `flat_moments`, keyed by
    /// node view id.
    pub(crate) detector_slices: HashMap<u32, NodeSliceTimeline>,
    /// Anticommutation break markers keyed by `(node view id, moment)`.
    pub(crate) detector_breaks: DetectorBreaks,
    /// Why detector-slice mode is unavailable (e.g. a `FlattenError`), or `None`
    /// when the flattened view built successfully.
    pub(crate) unavailable_reason: Option<String>,
    /// A short footer note when regions break or cross a seam (non-Clifford /
    /// factory programs), or `None` when every region resolved cleanly.
    pub(crate) note: Option<String>,
}

/// The owned inputs the lazy detector-slice job needs, gathered from
/// [`BloqViewerState`] so the work can move onto a background task.
pub(crate) struct DetsliceInputs {
    pub(crate) program: Arc<Bloq>,
    pub(crate) viewer_graph: BlockGraph,
    pub(crate) source_offset: IVec3,
    pub(crate) compile_config: CompileConfig,
    /// Identifies the installed compilation, including recompiles of one source revision.
    pub(crate) generation: u64,
}

/// State shared by lazily-computed viewer toggles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LazyToggle {
    /// No compiled program yet.
    Absent,
    /// Off and ready to compute or reuse its cache.
    Off,
    /// Computing; the mode activates when the result lands.
    Pending,
    /// On.
    On,
    /// Computed but unavailable; the tooltip carries the reason.
    Unavailable(String),
}

/// State for the circuit viewer panel, holding compiled circuit moments, qubit layout, and
/// navigation position within the flattened moment timeline.
#[derive(Resource, Debug, Clone)]
pub(crate) struct BloqViewerState {
    pub(crate) viewer_graph: BlockGraph,
    pub(crate) nodes: Vec<BloqNodeView>,
    pub(crate) edges: Vec<BloqEdgeView>,
    pub(crate) selected_node: Option<u32>,
    pub(crate) hovered_node: Option<u32>,
    pub(crate) layer_circuits: HashMap<i32, LayerCircuitView>,
    pub(crate) concurrent_ops_error: Option<String>,
    pub(crate) concurrent_ops_pending: bool,
    pub(crate) concurrent_ops: bool,
    pub(crate) selected_layer: i32,
    pub(crate) moments: Vec<FlatMoment>,
    pub(crate) qubit_coords: HashMap<usize, IVec2>,
    pub(crate) num_qubits: usize,
    pub(crate) current_moment: usize,
    pub(crate) compile_revision: Option<u64>,
    pub(crate) compile_generation: u64,
    pub(crate) pending_revision: Option<u64>,
    pub(crate) code_distance: Option<u32>,
    /// Exact lowering settings of `program`, retained for a lazy proxy compile.
    pub(crate) compile_config: Option<CompileConfig>,
    pub(crate) compile_duration: Option<Duration>,
    pub(crate) pan_offset: Vec2,
    pub(crate) zoom: f32,
    pub(crate) graph_pan_offset: Vec2,
    pub(crate) graph_zoom: f32,
    /// Whether classical readout/dataflow nodes are shown in the graph view.
    /// Persists across recompiles (a view preference, not compiled output).
    pub(crate) show_classical: bool,
    /// Memoized layered layout; recomputed only when the node set or classical
    /// toggle changes (see [`GraphLayoutCache`]).
    pub(crate) graph_layout_cache: GraphLayoutCache,
    pub(crate) last_error: Option<String>,
    /// Compiled input retained for lazy viewer jobs.
    pub(crate) program: Option<Arc<Bloq>>,
    pub(crate) source_program: Option<Arc<Bloq>>,
    pub(crate) branch_pins: Option<BTreeMap<String, bool>>,
    /// Choices being edited; Apply installs the complete reachable tuple.
    pub(crate) branch_pin_draft: BTreeMap<String, bool>,
    /// Coordinate offset used by lazy viewer jobs.
    pub(crate) source_offset: IVec3,
    /// Per-node flattened moment timelines (see [`DetsliceData::flat_moments`]).
    /// Empty until the lazy job populates it.
    pub(crate) flat_moments: HashMap<u32, Vec<FlatMoment>>,
    /// Per-node detector-slice regions (see [`DetsliceData::detector_slices`]).
    pub(crate) detector_slices: HashMap<u32, NodeSliceTimeline>,
    /// Anticommutation break markers (see [`DetsliceData::detector_breaks`]).
    pub(crate) detector_breaks: DetectorBreaks,
    /// Why detector-slice mode is unavailable, or `None` when it can be enabled.
    pub(crate) detslice_unavailable_reason: Option<String>,
    /// A short footer note when regions break or cross a seam, or `None`.
    pub(crate) detslice_note: Option<String>,
    /// Whether the detector-slice cache above has been computed for the current
    /// compile. Reset on each recompile; distinguishes "not computed yet" from
    /// "computed and empty".
    pub(crate) detslice_computed: bool,
    /// Visibility requested while the lazy slice job is in flight. `None` when
    /// no job is pending; installed when the result lands.
    pub(crate) detslice_pending: Option<SliceVisibility>,
    /// Independently visible detector and logical-observable overlays.
    pub(crate) slice_visibility: SliceVisibility,
    /// The region the pointer is over in the current moment, isolated (others
    /// dimmed) while hovered. Identified by [`SliceRegionView::id`].
    pub(crate) hovered_region: Option<SliceRegionId>,
    /// Region isolated by clicking its row in the slice index column.
    pub(crate) selected_region: Option<SliceRegionId>,
}

impl BloqViewerState {
    fn node_by_id(&self, id: u32) -> Option<&BloqNodeView> {
        self.nodes.iter().find(|node| node.id == id)
    }

    fn node_source_layer(&self, node: &BloqNodeView) -> Option<i32> {
        i32::try_from(node.layer.div_euclid(2)).ok()
    }

    /// Marks a viewer compile as in flight for `revision`, keeping the previous
    /// result visible until it completes.
    pub(crate) fn begin_compile(&mut self, revision: u64) {
        self.pending_revision = Some(revision);
        self.last_error = None;
    }

    /// Reuses the original compilation and its display coordinates for pinning.
    pub(crate) fn pinning_input(&self) -> Option<CompiledBloqView> {
        Some(CompiledBloqView {
            viewer_graph: self.viewer_graph.clone(),
            nodes: Vec::new(),
            edges: Vec::new(),
            code_distance: self.code_distance?,
            compile_config: self.compile_config?,
            compile_duration: self.compile_duration?,
            program: self.program.clone()?,
            source_program: self.source_program.clone()?,
            branch_pins: self.branch_pins.clone(),
            branch_pin_draft: self.branch_pin_draft.clone(),
            source_offset: self.source_offset,
        })
    }

    pub(crate) fn reset_for_new_graph(&mut self) {
        *self = Self::default();
    }

    /// Moves the timeline cursor, clamping to the last moment.
    pub(crate) fn set_current_moment(&mut self, moment: usize) {
        let moment = moment.min(self.moments.len().saturating_sub(1));
        if self.current_moment != moment {
            self.hovered_region = None;
        }
        self.current_moment = moment;
    }

    /// Selects a node (ignoring ids not present) and loads its circuit into the
    /// moment timeline.
    pub(crate) fn set_selected_node(&mut self, node_id: Option<u32>) {
        self.selected_node = node_id.filter(|id| self.nodes.iter().any(|node| node.id == *id));
        if let Some(layer) = self
            .selected_node
            .and_then(|id| self.node_by_id(id))
            .and_then(|node| self.node_source_layer(node))
        {
            self.selected_layer = layer;
        }
        self.load_selected_node_circuit();
    }

    pub(crate) fn set_concurrent_ops(&mut self, enabled: bool) {
        self.concurrent_ops = enabled && !self.layer_circuits.is_empty();
        self.load_selected_node_circuit();
    }

    pub(crate) fn concurrent_ops_toggle(&self) -> LazyToggle {
        if self.concurrent_ops_pending {
            LazyToggle::Pending
        } else if self.concurrent_ops {
            LazyToggle::On
        } else if self.program.is_none() {
            LazyToggle::Absent
        } else if let Some(error) = &self.concurrent_ops_error {
            LazyToggle::Unavailable(error.clone())
        } else {
            LazyToggle::Off
        }
    }

    pub(crate) fn concurrent_ops_inputs(&self) -> Option<ConcurrentOpsInputs> {
        Some(ConcurrentOpsInputs {
            program: self.program.clone()?,
            viewer_graph: self.viewer_graph.clone(),
            source_offset: self.source_offset,
            generation: self.compile_generation,
        })
    }

    pub(crate) fn begin_concurrent_ops(&mut self) {
        self.concurrent_ops_pending = true;
    }

    pub(crate) fn finish_concurrent_ops(
        &mut self,
        generation: u64,
        layer_circuits: HashMap<i32, LayerCircuitView>,
        error: Option<String>,
    ) -> bool {
        if self.compile_generation != generation {
            return false;
        }
        self.layer_circuits = layer_circuits;
        self.concurrent_ops_error = error;
        self.concurrent_ops_pending = false;
        self.set_concurrent_ops(true);
        true
    }

    pub(crate) fn set_concurrent_layer(&mut self, layer: i32) {
        if self.concurrent_ops
            && self.selected_layer != layer
            && self.layer_circuits.contains_key(&layer)
        {
            self.selected_layer = layer;
            self.load_selected_node_circuit();
        }
    }

    pub(crate) fn toggle_selected_node(&mut self, node_id: u32) {
        if self.selected_node == Some(node_id) {
            self.set_selected_node(None);
        } else {
            self.set_selected_node(Some(node_id));
        }
    }

    pub(crate) fn set_hovered_node(&mut self, node_id: Option<u32>) {
        self.hovered_node = node_id.filter(|id| self.nodes.iter().any(|node| node.id == *id));
    }

    /// Drop selection/hover on nodes the classical toggle currently hides, so no
    /// window lingers for a tile that is no longer on the canvas.
    pub(crate) fn deselect_hidden_nodes(&mut self) {
        let is_hidden = |id: Option<u32>| {
            !self.show_classical
                && id
                    .and_then(|id| self.node_by_id(id))
                    .is_some_and(|node| !node.quantum_visible())
        };
        let drop_selected = is_hidden(self.selected_node);
        let drop_hovered = is_hidden(self.hovered_node);
        if drop_selected {
            self.set_selected_node(None);
        }
        if drop_hovered {
            self.hovered_node = None;
        }
    }

    /// The source-graph elements highlighted when hovering the current node.
    pub(crate) fn hovered_source_elements_iter(&self) -> impl Iterator<Item = GraphElement> + '_ {
        self.hovered_node
            .and_then(|id| self.node_by_id(id))
            .into_iter()
            .flat_map(|node| node.source_elements.iter().copied())
    }

    fn load_selected_node_circuit(&mut self) {
        if self.concurrent_ops
            && !self.layer_circuits.contains_key(&self.selected_layer)
            && let Some(layer) = self
                .layer_circuits
                .keys()
                .copied()
                .min_by_key(|layer| (layer.abs_diff(self.selected_layer), *layer))
        {
            self.selected_layer = layer;
        }
        let layout = if self.concurrent_ops {
            self.selected_node
                .and_then(|_| self.layer_circuits.get(&self.selected_layer))
                .map(|layer| (layer.qubit_coords.clone(), layer.num_qubits))
        } else {
            self.selected_node
                .and_then(|id| self.node_by_id(id))
                .map(|node| (node.qubit_coords.clone(), node.num_qubits))
        };
        if let Some((qubit_coords, num_qubits)) = layout {
            self.moments = self.selected_moments();
            self.qubit_coords = qubit_coords;
            self.num_qubits = num_qubits;
        } else {
            self.moments.clear();
            self.qubit_coords.clear();
            self.num_qubits = 0;
        }
        self.current_moment = 0;
        self.hovered_region = None;
        self.selected_region = None;
        self.pan_offset = Vec2::ZERO;
        self.zoom = 1.0;
    }

    fn selected_moments(&self) -> Vec<FlatMoment> {
        if self.concurrent_ops {
            return self
                .layer_circuits
                .get(&self.selected_layer)
                .map(|layer| layer.moments.clone())
                .unwrap_or_default();
        }
        let Some(id) = self.selected_node else {
            return Vec::new();
        };
        if self.slice_visibility.any()
            && let Some(flat) = self.flat_moments.get(&id)
        {
            return flat.clone();
        }
        self.node_by_id(id)
            .map(|node| node.moments.clone())
            .unwrap_or_default()
    }

    /// Whether detector-slice mode can be turned on: the flattened view built
    /// and there is at least one node's flattened timeline to show.
    pub(crate) fn detslice_available(&self) -> bool {
        self.detslice_unavailable_reason.is_none() && !self.flat_moments.is_empty()
    }

    /// Current regions, merged across concurrent lanes when enabled.
    fn current_slice_regions(&self) -> Cow<'_, [SliceRegionView]> {
        if !self.concurrent_ops {
            return Cow::Borrowed(
                self.selected_node
                    .and_then(|id| self.detector_slices.get(&id))
                    .and_then(|timeline| timeline.get(self.current_moment))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );
        }

        let mut regions: Vec<SliceRegionView> = Vec::new();
        let mut indices: HashMap<SliceRegionId, usize> = HashMap::new();
        for &(node, moment) in self
            .layer_circuits
            .get(&self.selected_layer)
            .and_then(|layer| layer.source_moments.get(self.current_moment))
            .into_iter()
            .flatten()
        {
            for region in self
                .detector_slices
                .get(&node)
                .and_then(|timeline| timeline.get(moment))
                .into_iter()
                .flatten()
            {
                if let Some(&index) = indices.get(&region.id) {
                    regions[index].terms.extend(region.terms.iter().copied());
                } else {
                    indices.insert(region.id, regions.len());
                    regions.push(region.clone());
                }
            }
        }
        for region in &mut regions {
            region
                .terms
                .sort_unstable_by_key(|term| (term.qubit.x, term.qubit.y, term.pauli));
            region.terms.dedup();
        }
        Cow::Owned(regions)
    }

    /// Current break coordinates, merged across concurrent lanes when enabled.
    fn current_slice_breaks(&self) -> Cow<'_, [IVec2]> {
        if !self.concurrent_ops {
            return Cow::Borrowed(
                self.selected_node
                    .and_then(|id| self.detector_breaks.get(&(id, self.current_moment)))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );
        }

        let mut breaks = self
            .layer_circuits
            .get(&self.selected_layer)
            .and_then(|layer| layer.source_moments.get(self.current_moment))
            .into_iter()
            .flatten()
            .flat_map(|&(node, moment)| {
                self.detector_breaks
                    .get(&(node, moment))
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect::<Vec<_>>();
        breaks.sort_unstable_by_key(|coord| (coord.x, coord.y));
        breaks.dedup();
        Cow::Owned(breaks)
    }

    /// Applies slice visibility, swapping timelines and clamping the cursor.
    /// Ignored when a requested overlay is unavailable.
    pub(crate) fn set_slice_visibility(&mut self, visibility: SliceVisibility) {
        if visibility.any() && !self.detslice_available() {
            return;
        }
        if self.slice_visibility == visibility {
            return;
        }
        self.slice_visibility = visibility;
        if self.selected_region.is_some_and(|id| !visibility.shows(id)) {
            self.selected_region = None;
        }
        self.moments = self.selected_moments();
        self.current_moment = self
            .current_moment
            .min(self.moments.len().saturating_sub(1));
        self.hovered_region = None;
    }

    /// The owned inputs for the lazy detector-slice job, or `None` before the
    /// first compile has retained a program.
    pub(crate) fn detslice_inputs(&self) -> Option<DetsliceInputs> {
        Some(DetsliceInputs {
            program: self.program.clone()?,
            viewer_graph: self.viewer_graph.clone(),
            source_offset: self.source_offset,
            compile_config: self.compile_config?,
            generation: self.compile_generation,
        })
    }

    /// Marks the lazy detector-slice job as in flight; the mode activates when
    /// its result lands (see [`Self::finish_detslice`]).
    pub(crate) fn begin_detslice(&mut self, visibility: SliceVisibility) {
        self.detslice_pending = Some(visibility);
    }

    /// Clears the intent to activate once the in-flight job lands (e.g. the user
    /// toggled the mode back off while it was computing). The job still completes
    /// and fills the cache, it just no longer flips the mode on.
    pub(crate) fn cancel_detslice_pending(&mut self) {
        self.detslice_pending = None;
    }

    /// Installs a completed detector-slice computation, discarding it if a newer
    /// compile has since replaced the program it was built from. Activates the
    /// mode if the user was waiting on it (unless the overlay came back
    /// unavailable, in which case it stays off with the reason surfaced).
    pub(crate) fn finish_detslice(&mut self, generation: u64, data: DetsliceData) -> bool {
        if self.compile_generation != generation {
            return false;
        }
        self.flat_moments = data.flat_moments;
        self.detector_slices = data.detector_slices;
        self.detector_breaks = data.detector_breaks;
        self.detslice_unavailable_reason = data.unavailable_reason;
        self.detslice_note = data.note;
        self.detslice_computed = true;
        let wanted = self.detslice_pending.take();
        if let Some(visibility) = wanted {
            self.set_slice_visibility(visibility);
        }
        true
    }

    fn slice_toggle(&self, enabled: bool) -> LazyToggle {
        if self.detslice_pending.is_some() {
            return LazyToggle::Pending;
        }
        if enabled {
            return LazyToggle::On;
        }
        if self.nodes.is_empty() {
            return LazyToggle::Absent;
        }
        if self.detslice_computed && !self.detslice_available() {
            let reason = self
                .detslice_unavailable_reason
                .clone()
                .unwrap_or_else(|| "No detector slices for this program".to_string());
            return LazyToggle::Unavailable(reason);
        }
        LazyToggle::Off
    }

    /// State of the detector-slice checkbox and `D` hotkey.
    pub(crate) fn detslice_toggle(&self) -> LazyToggle {
        self.slice_toggle(self.slice_visibility.detectors)
    }

    /// State of the logical-observable-slice checkbox.
    fn obsslice_toggle(&self) -> LazyToggle {
        self.slice_toggle(self.slice_visibility.observables)
    }

    pub(crate) fn previous_moment_index(&self) -> Option<usize> {
        self.current_moment.checked_sub(1)
    }

    pub(crate) fn next_moment_index(&self) -> Option<usize> {
        (self.current_moment + 1 < self.moments.len()).then_some(self.current_moment + 1)
    }

    fn previous_timeline_position(&self) -> Option<(i32, usize)> {
        self.previous_moment_index()
            .map(|moment| (self.selected_layer, moment))
            .or_else(|| {
                self.concurrent_ops.then_some(())?;
                self.layer_circuits
                    .iter()
                    .filter(|(layer, circuit)| {
                        **layer < self.selected_layer && !circuit.moments.is_empty()
                    })
                    .max_by_key(|(layer, _)| *layer)
                    .map(|(layer, circuit)| (*layer, circuit.moments.len() - 1))
            })
    }

    fn next_timeline_position(&self) -> Option<(i32, usize)> {
        self.next_moment_index()
            .map(|moment| (self.selected_layer, moment))
            .or_else(|| {
                self.concurrent_ops.then_some(())?;
                self.layer_circuits
                    .iter()
                    .filter(|(layer, circuit)| {
                        **layer > self.selected_layer && !circuit.moments.is_empty()
                    })
                    .min_by_key(|(layer, _)| *layer)
                    .map(|(layer, _)| (*layer, 0))
            })
    }

    /// Index of the nearest earlier `Reset` moment, for round-boundary stepping.
    fn previous_reset_moment_index(&self) -> Option<usize> {
        self.moments[..self.current_moment]
            .iter()
            .rposition(|moment| moment.kind == MomentKind::Reset)
    }

    /// Index of the nearest later `Reset` moment, for round-boundary stepping.
    fn next_reset_moment_index(&self) -> Option<usize> {
        self.moments
            .iter()
            .enumerate()
            .skip(self.current_moment.saturating_add(1))
            .find(|(_, moment)| moment.kind == MomentKind::Reset)
            .map(|(index, _)| index)
    }

    /// Whether the displayed view was compiled from a graph revision older than
    /// `current_revision`.
    pub(crate) fn is_stale(&self, current_revision: u64) -> bool {
        self.compile_revision
            .is_some_and(|revision| revision != current_revision)
    }

    /// Installs a completed compile result, clearing selection/hover and
    /// resetting the graph-view pan/zoom.
    pub(crate) fn finish_compile(&mut self, revision: u64, output: CompiledBloqView) {
        static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
        self.compile_generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let rebuild_concurrent_ops = self.concurrent_ops || self.concurrent_ops_pending;
        self.viewer_graph = output.viewer_graph;
        self.nodes = output.nodes;
        self.edges = output.edges;
        self.layer_circuits.clear();
        self.concurrent_ops_error = None;
        self.concurrent_ops_pending = rebuild_concurrent_ops;
        self.concurrent_ops = false;
        self.program = Some(output.program);
        self.branch_pin_draft = output.branch_pin_draft;
        self.source_program = Some(output.source_program);
        self.branch_pins = output.branch_pins;
        self.source_offset = output.source_offset;
        // Detector slices are computed lazily on first enable; invalidate the
        // previous compile's cache and degrade the mode to off (it recomputes
        // when the user re-enables it).
        self.flat_moments.clear();
        self.detector_slices.clear();
        self.detector_breaks.clear();
        self.detslice_unavailable_reason = None;
        self.detslice_note = None;
        self.detslice_computed = false;
        self.detslice_pending = None;
        self.slice_visibility = SliceVisibility::default();
        self.hovered_node = None;
        self.hovered_region = None;
        self.selected_node = None;
        self.load_selected_node_circuit();
        self.compile_revision = Some(revision);
        self.pending_revision = None;
        self.code_distance = Some(output.code_distance);
        self.compile_config = Some(output.compile_config);
        self.compile_duration = Some(output.compile_duration);
        self.graph_pan_offset = Vec2::ZERO;
        self.graph_zoom = 1.0;
        self.graph_layout_cache = GraphLayoutCache::default();
        self.last_error = None;
    }
}

impl Default for BloqViewerState {
    fn default() -> Self {
        Self {
            viewer_graph: BlockGraph::default(),
            nodes: Vec::new(),
            edges: Vec::new(),
            selected_node: None,
            hovered_node: None,
            layer_circuits: HashMap::new(),
            concurrent_ops_error: None,
            concurrent_ops_pending: false,
            concurrent_ops: false,
            selected_layer: 0,
            moments: Vec::new(),
            qubit_coords: HashMap::new(),
            num_qubits: 0,
            current_moment: 0,
            compile_revision: None,
            compile_generation: 0,
            pending_revision: None,
            code_distance: None,
            compile_config: None,
            compile_duration: None,
            pan_offset: Vec2::ZERO,
            zoom: 1.0,
            graph_pan_offset: Vec2::ZERO,
            graph_zoom: 1.0,
            show_classical: false,
            graph_layout_cache: GraphLayoutCache::default(),
            last_error: None,
            program: None,
            source_program: None,
            branch_pins: None,
            branch_pin_draft: BTreeMap::new(),
            source_offset: IVec3::ZERO,
            flat_moments: HashMap::new(),
            detector_slices: HashMap::new(),
            detector_breaks: HashMap::new(),
            detslice_unavailable_reason: None,
            detslice_note: None,
            detslice_computed: false,
            detslice_pending: None,
            slice_visibility: SliceVisibility::default(),
            hovered_region: None,
            selected_region: None,
        }
    }
}

const PROGRAM_WINDOW_ID: &str = "bloq-viewer";
const NODE_WINDOW_ID: &str = "program-node-circuit-viewer";
const VIEWER_MIN_WIDTH: f32 = 320.0;
const VIEWER_MIN_HEIGHT: f32 = 220.0;
const VIEWER_CANVAS_MIN_HEIGHT: f32 = 96.0;
const VIEWER_SCRUBBER_HEIGHT: f32 = 50.0;
const VIEWER_MIN_PITCH: f32 = 4.0;
const VIEWER_MAX_PITCH: f32 = 80.0;
const GRAPH_NODE_SIZE: egui::Vec2 = egui::vec2(130.0, 46.0);
const GRAPH_COLUMN_GAP: f32 = 74.0;
/// Inset between a region container's border and its children's layout box.
const REGION_PAD: f32 = 12.0;
/// Extra top inset inside a region container reserving room for its label.
const REGION_HEADER: f32 = 24.0;

#[derive(Clone, Copy)]
struct ViewerBodyLayout {
    scrubber_rect: Rect,
    canvas_rect: Rect,
    footer_rect: Rect,
}

#[derive(Clone, Copy)]
struct SliceColumnLayout {
    circuit_rect: Rect,
    column_rect: Rect,
    row_height: f32,
    row_gap: f32,
}

#[derive(Clone, Copy)]
struct CircuitViewerColors {
    canvas_fill: Color32,
    scrubber_fill: Color32,
    site_fill: Color32,
    site_stroke: Color32,
    wire: Color32,
    gate_stroke: Color32,
    active_marker: Color32,
    default_gate_fill: Color32,
    default_gate_text: Color32,
    clifford_gate_fill: Color32,
    clifford_gate_text: Color32,
    non_clifford_gate_fill: Color32,
    non_clifford_gate_text: Color32,
    reset_gate_fill: Color32,
    reset_gate_text: Color32,
    interaction_label_fill: Color32,
    interaction_label_text: Color32,
    basis_x_fill: Color32,
    basis_y_fill: Color32,
    basis_z_fill: Color32,
    detslice_x: Color32,
    detslice_y: Color32,
    detslice_z: Color32,
    detslice_mixed: Color32,
    detslice_outline: Color32,
    detslice_break: Color32,
}

fn circuit_colors(palette: &ThemePalette) -> CircuitViewerColors {
    if palette.dark_mode {
        CircuitViewerColors {
            canvas_fill: Color32::from_rgb(35, 34, 33),
            scrubber_fill: Color32::from_rgb(50, 48, 47),
            site_fill: Color32::from_rgb(60, 56, 54),
            site_stroke: palette.grey2,
            wire: palette.text_bright,
            gate_stroke: Color32::from_rgb(168, 153, 132),
            active_marker: palette.text_bright,
            default_gate_fill: Color32::from_rgb(80, 73, 69),
            default_gate_text: palette.text_bright,
            clifford_gate_fill: palette.yellow,
            clifford_gate_text: Color32::from_rgb(40, 40, 40),
            non_clifford_gate_fill: palette.orange,
            non_clifford_gate_text: Color32::from_rgb(40, 40, 40),
            reset_gate_fill: palette.grey2,
            reset_gate_text: Color32::from_rgb(40, 40, 40),
            interaction_label_fill: Color32::from_rgb(60, 56, 54),
            interaction_label_text: palette.text_bright,
            basis_x_fill: Color32::from_rgb(50, 48, 47),
            basis_y_fill: palette.accent_secondary,
            basis_z_fill: palette.text_bright,
            // Stim's detslice hues (#FF4040/#59FF7A/#4DA6FF/#AAAAAA), brightened
            // to read over the dark canvas fill.
            detslice_x: Color32::from_rgb(255, 90, 85),
            detslice_y: Color32::from_rgb(120, 220, 130),
            detslice_z: Color32::from_rgb(90, 170, 250),
            detslice_mixed: Color32::from_rgb(175, 175, 175),
            detslice_outline: Color32::from_rgb(18, 18, 18),
            detslice_break: Color32::from_rgb(232, 90, 200),
        }
    } else {
        CircuitViewerColors {
            canvas_fill: Color32::from_rgb(238, 243, 247),
            scrubber_fill: Color32::from_rgb(255, 255, 255),
            site_fill: Color32::from_rgb(255, 255, 255),
            site_stroke: palette.border_bright,
            wire: palette.text_bright,
            gate_stroke: palette.border_bright,
            active_marker: palette.text_bright,
            default_gate_fill: Color32::from_rgb(219, 234, 254),
            default_gate_text: palette.text_bright,
            clifford_gate_fill: Color32::from_rgb(246, 211, 101),
            clifford_gate_text: palette.text_bright,
            non_clifford_gate_fill: Color32::from_rgb(244, 162, 97),
            non_clifford_gate_text: palette.text_bright,
            reset_gate_fill: Color32::from_rgb(219, 226, 235),
            reset_gate_text: palette.text_bright,
            interaction_label_fill: Color32::from_rgb(255, 255, 255),
            interaction_label_text: palette.text_bright,
            basis_x_fill: Color32::from_rgb(255, 255, 255),
            basis_y_fill: palette.accent_secondary,
            basis_z_fill: palette.text_bright,
            // Darker, saturated variants of the Stim detslice hues so the fills
            // stay legible on the light canvas.
            detslice_x: Color32::from_rgb(210, 55, 55),
            detslice_y: Color32::from_rgb(34, 160, 88),
            detslice_z: Color32::from_rgb(45, 116, 208),
            detslice_mixed: Color32::from_rgb(135, 135, 135),
            detslice_outline: Color32::from_rgb(28, 28, 28),
            detslice_break: Color32::from_rgb(196, 26, 156),
        }
    }
}

/// Draws the Bloq viewer (node graph plus per-node circuit timeline) while the
/// editor is in Bloq mode.
pub(crate) fn draw_circuit_viewer(
    ctx: &egui::Context,
    viewport: Rect,
    editor_state: &EditorState,
    graph_state: &GraphState,
    tab_id: crate::resources::EditorTabId,
    compile_ui: &CompileUiState,
    jobs: &EditorJobs,
    circuit_viewer: &mut BloqViewerState,
    intents: &mut UiIntentBuffer,
) {
    if editor_state.mode != EditorMode::Bloq {
        return;
    }
    let palette = theme::palette(editor_state.theme_preset);
    apply_keyboard_shortcuts(ctx, circuit_viewer, intents);
    draw_bloq_window(
        ctx,
        viewport,
        graph_state,
        tab_id,
        compile_ui,
        jobs,
        circuit_viewer,
        intents,
        palette,
    );
    if let Some(id) = circuit_viewer.selected_node {
        let show_attributes = circuit_viewer
            .nodes
            .iter()
            .find(|node| node.id == id)
            .is_some_and(|node| !node.category.is_quantum() || !node.attributes.is_empty());
        if show_attributes && !circuit_viewer.concurrent_ops {
            draw_node_attributes_window(ctx, viewport, circuit_viewer, intents, palette);
        } else {
            draw_node_circuit_window(
                ctx,
                viewport,
                jobs,
                circuit_viewer,
                intents,
                palette,
                editor_state.theme_preset,
            );
        }
    }
}

fn draw_bloq_window(
    ctx: &egui::Context,
    viewport: Rect,
    graph_state: &GraphState,
    tab_id: crate::resources::EditorTabId,
    compile_ui: &CompileUiState,
    jobs: &EditorJobs,
    circuit_viewer: &mut BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let default_rect = tiled_window_rect(viewport, egui::Align2::LEFT_TOP);
    let max_size = viewport.size().max(egui::Vec2::splat(1.0));
    let min_size = egui::vec2(
        VIEWER_MIN_WIDTH.min(default_rect.width()),
        VIEWER_MIN_HEIGHT.min(default_rect.height()),
    );

    egui::Window::new("Program")
        .id(egui::Id::new(PROGRAM_WINDOW_ID))
        .default_rect(default_rect)
        .min_size(min_size)
        .max_size(max_size)
        .collapsible(false)
        .resizable(true)
        .show(ctx, |ui| {
            let fit_requested =
                draw_program_header(ui, graph_state, circuit_viewer, intents, palette);
            draw_compile_status(
                ui,
                graph_state,
                tab_id,
                jobs,
                circuit_viewer,
                intents,
                palette,
            );
            draw_source_branch_pins(ui, graph_state, jobs, circuit_viewer, intents);

            ui.scope_builder(
                egui::UiBuilder::new().id(ui.make_persistent_id("program-body")),
                |ui| {
                    if circuit_viewer.nodes.is_empty() {
                        draw_empty_state(ui, compile_ui, jobs, circuit_viewer, intents, palette);
                    } else {
                        let canvas_height = ui.available_height().max(1.0);
                        draw_bloq_canvas(
                            ui,
                            circuit_viewer,
                            canvas_height,
                            fit_requested,
                            intents,
                            palette,
                        );
                    }
                },
            );
        });
}

fn draw_source_branch_pins(
    ui: &mut egui::Ui,
    graph_state: &GraphState,
    jobs: &EditorJobs,
    viewer: &mut BloqViewerState,
    intents: &mut UiIntentBuffer,
) {
    if viewer.branch_pin_draft.is_empty() {
        return;
    }
    let title = if viewer.branch_pins.is_some() {
        "Branch selections · pinned"
    } else {
        "Branch selections · unpinned"
    };
    egui::CollapsingHeader::new(title)
        .id_salt("source-branch-pins")
        .show(ui, |ui| {
            ui.label("Choose branch outcomes to inspect their circuit timelines.");
            ui.add_enabled_ui(
                !jobs.compilation_running() && !viewer.is_stale(graph_state.revision),
                |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .show(ui, |ui| {
                            for (name, value) in &mut viewer.branch_pin_draft {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(name);
                                    ui.selectable_value(value, false, "false");
                                    ui.selectable_value(value, true, "true");
                                });
                            }
                        });
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Apply pins").clicked() {
                            intents.push(UiIntent::PinViewerBranches(Some(
                                viewer.branch_pin_draft.clone(),
                            )));
                        }
                        if ui
                            .add_enabled(
                                viewer.branch_pins.is_some(),
                                egui::Button::new("Clear pins"),
                            )
                            .clicked()
                        {
                            intents.push(UiIntent::PinViewerBranches(None));
                        }
                        if viewer
                            .branch_pins
                            .as_ref()
                            .is_some_and(|pins| pins != &viewer.branch_pin_draft)
                        {
                            ui.small("Unapplied changes");
                        }
                    });
                },
            );
        });
}

fn draw_program_header(
    ui: &mut egui::Ui,
    graph_state: &GraphState,
    circuit_viewer: &mut BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) -> bool {
    let mut fit = false;
    ui.horizontal_wrapped(|ui| {
        theme::status_chip(
            ui,
            "\u{f0e8}",
            format_args!("nodes {}", circuit_viewer.nodes.len()),
            palette.accent_primary,
        );
        theme::status_chip(
            ui,
            "\u{f074}",
            format_args!("edges {}", circuit_viewer.edges.len()),
            palette.text_primary,
        );
        if let Some(distance) = circuit_viewer.code_distance {
            theme::status_chip(
                ui,
                "\u{f0ac}",
                format_args!("d={distance}"),
                palette.text_primary,
            );
        }
        if let Some(revision) = circuit_viewer.compile_revision {
            let color = if revision == graph_state.revision {
                palette.success
            } else {
                palette.accent_warn
            };
            theme::status_chip(ui, "\u{f15b}", format_args!("r{revision}"), color);
        }
        if let Some(duration) = circuit_viewer.compile_duration {
            theme::status_chip(
                ui,
                "\u{f017}",
                format_args!("{}ms", duration.as_millis()),
                palette.text_dim,
            );
        }
    });

    ui.horizontal_wrapped(|ui| {
        let response = close_icon_button(ui, palette).on_hover_text("Close Program");
        if activated(&response) {
            intents.push(UiIntent::SetMode(EditorMode::View));
        }
        let mut show_classical = circuit_viewer.show_classical;
        if ui
            .checkbox(&mut show_classical, "classical")
            .on_hover_text("Show classical readout / dataflow nodes")
            .changed()
        {
            intents.push(UiIntent::SetShowClassical(show_classical));
        }
        fit = ui
            .add_enabled(
                !circuit_viewer.nodes.is_empty(),
                egui::Button::new("Fit diagram"),
            )
            .on_hover_text("Center and fit all visible program nodes")
            .clicked();
        let svg_response = ui
            .add_enabled(
                !circuit_viewer.nodes.is_empty(),
                egui::Button::new(
                    egui::RichText::new("SVG")
                        .small()
                        .color(palette.accent_primary),
                ),
            )
            .on_hover_text("Export view as SVG");
        if activated(&svg_response)
            && let Some((file_name, contents)) = export_program_svg(circuit_viewer, palette)
        {
            intents.push(UiIntent::SaveSvg {
                file_name,
                contents,
            });
        }
    });
    fit
}

fn draw_compile_status(
    ui: &mut egui::Ui,
    graph_state: &GraphState,
    tab_id: crate::resources::EditorTabId,
    jobs: &EditorJobs,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if let Some(pending_revision) = circuit_viewer.pending_revision {
        ui.horizontal(|ui| {
            ui.spinner();
            let detail = jobs
                .compilation_progress(tab_id, pending_revision)
                .map(|(stage, seconds)| format!("{stage} · {seconds}s"))
                .unwrap_or_else(|| format!("revision r{pending_revision}"));
            ui.label(
                egui::RichText::new(format!("Compiling Bloq: {detail}"))
                    .small()
                    .color(palette.accent_warn),
            );
            if ui.small_button("Cancel").clicked() {
                intents.push(UiIntent::CancelCompilation(tab_id));
            }
        });
        ui.add_space(4.0);
    }

    if circuit_viewer.is_stale(graph_state.revision) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Bloq is stale. Compiled r{} while graph is now r{}.",
                    circuit_viewer.compile_revision.unwrap_or_default(),
                    graph_state.revision
                ))
                .small()
                .color(palette.accent_warn),
            );
            let response = ui.add_enabled(
                !jobs.compilation_running(),
                egui::Button::new(
                    egui::RichText::new("Recompile")
                        .small()
                        .color(palette.accent_primary),
                ),
            );
            if activated(&response) {
                intents.push(UiIntent::CompileForViewer);
            }
        });
        ui.add_space(4.0);
    }
}

/// Interactive wrapper around [`paint_program_graph`]: handles pan/zoom input and
/// the per-node hit testing, then delegates all drawing to the pure paint pass so
/// the on-screen view and the SVG export share one code path.
fn draw_bloq_canvas(
    ui: &mut egui::Ui,
    circuit_viewer: &mut BloqViewerState,
    canvas_height: f32,
    fit_requested: bool,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), canvas_height),
        Sense::click_and_drag(),
    );

    if fit_requested
        || circuit_viewer.graph_layout_cache.key
            != Some((
                circuit_viewer.compile_revision,
                circuit_viewer.show_classical,
            ))
    {
        fit_program_graph(circuit_viewer, rect.size());
    }

    if response.dragged() {
        let delta = ui.ctx().input(|input| input.pointer.delta());
        circuit_viewer.graph_pan_offset += glam::Vec2::new(delta.x, delta.y);
    }

    if response.hovered() {
        let scroll_delta = ui.ctx().input(|input| input.smooth_scroll_delta.y);
        if scroll_delta.abs() > f32::EPSILON {
            let zoom_factor = (scroll_delta / 600.0).exp();
            circuit_viewer.graph_zoom = (circuit_viewer.graph_zoom * zoom_factor).clamp(0.05, 2.5);
        }
    }

    let layout = program_node_layout(rect, circuit_viewer);

    // Interaction pass: register a hit rect per visible tile and collect the
    // hovered node. Painting happens afterwards through the shared pure pass, so
    // splitting the two does not change visual order (everything goes through the
    // painter). A tile panned/zoomed fully out of the canvas is clipped away, so
    // its invisible rect must not catch clicks or hovers either.
    let mut hovered = None;
    for node in &circuit_viewer.nodes {
        let Some(node_rect) = layout.get(&node.id).copied() else {
            continue;
        };
        if !rect.intersects(node_rect) {
            continue;
        }
        let node_response = ui.interact(
            node_rect,
            egui::Id::new(("bloq-node", node.id)),
            Sense::click(),
        );
        node_response
            .widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &node.label));
        if node_response.hovered() {
            hovered = Some(node.id);
            node_response.clone().on_hover_text(node.label.clone());
        }
        if activated(&node_response) {
            intents.push(UiIntent::ToggleBloqNode(node.id));
        }
    }

    // Pan/zoom can push node tiles and edges outside the canvas; clip so they
    // never spill over the header or window chrome.
    let painter = ui.painter().with_clip_rect(rect);
    paint_program_graph(
        &painter,
        rect,
        circuit_viewer,
        &layout,
        program_graph_origin(rect, circuit_viewer),
        hovered,
        palette,
    );

    intents.push(UiIntent::HoverBloqNode(hovered));
}

/// Paints the whole program graph — background, edges, then node tiles — into
/// `painter`. Pure: no input handling, so both the live canvas and the SVG
/// export call it. `layout` supplies each visible node's rect; nodes absent from
/// it (hidden by the classical toggle) are skipped.
fn paint_program_graph(
    painter: &egui::Painter,
    rect: Rect,
    circuit_viewer: &BloqViewerState,
    layout: &HashMap<u32, Rect>,
    origin: Pos2,
    hovered: Option<u32>,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    painter.rect_filled(rect, 6.0, colors.canvas_fill);

    let zoom = circuit_viewer.graph_zoom;
    for (index, edge) in circuit_viewer.edges.iter().enumerate() {
        let Some(route) = circuit_viewer.graph_layout_cache.routes.get(&index) else {
            continue;
        };
        draw_program_edge(painter, route, origin, edge, zoom, palette);
    }

    // Regions draw before other tiles so their container boxes sit underneath:
    // children painted later stay legible. Region ids ascend with nesting depth
    // (bodies allocate view ids after their parent), so sorting by id draws outer
    // boxes before inner ones.
    let mut draw_order: Vec<&BloqNodeView> = circuit_viewer.nodes.iter().collect();
    draw_order.sort_by_key(|node| (node.category != BloqNodeCategory::Region, node.id));
    for node in draw_order {
        // Hidden nodes are absent from `layout`, so this also skips their tiles.
        let Some(node_rect) = layout.get(&node.id).copied() else {
            continue;
        };
        draw_program_node(
            painter,
            node_rect,
            node,
            circuit_viewer,
            hovered == Some(node.id),
            palette,
        );
    }
}

// ============================================================================
// SVG export
// ============================================================================

/// Tight margin around the exported program diagram.
const EXPORT_MARGIN: f32 = 16.0;
/// Fixed nominal grid pitch for the exported circuit view (zoom-independent).
const CIRCUIT_EXPORT_PITCH: f32 = 24.0;

/// Renders the whole program diagram to SVG at zoom 1, ignoring the live
/// pan/zoom. Returns `(file_name, svg)`, or `None` when there is nothing to
/// export. Requires `&mut` only to (re)build the cached layout.
fn export_program_svg(
    circuit_viewer: &mut BloqViewerState,
    palette: &ThemePalette,
) -> Option<(String, String)> {
    if circuit_viewer.nodes.is_empty() {
        return None;
    }
    ensure_graph_layout(circuit_viewer);

    // `draw_program_node` scales its label font and region header by
    // `graph_zoom`, so neutralize the live zoom to keep the zoom-1 export
    // self-consistent, then restore it once the SVG string is built.
    let saved_zoom = std::mem::replace(&mut circuit_viewer.graph_zoom, 1.0);

    // Build the export layout straight from the cache (zoom 1), shifted by the
    // margin, rather than through `program_node_layout` which applies live
    // pan/zoom/centering.
    let cache = &circuit_viewer.graph_layout_cache;
    let size = egui::vec2(cache.content.0, cache.content.1)
        + egui::vec2(2.0 * EXPORT_MARGIN, 2.0 * EXPORT_MARGIN);
    let full_rect = Rect::from_min_size(Pos2::ZERO, size);
    let layout: HashMap<u32, Rect> = cache
        .positions
        .iter()
        .map(|(id, (x, y))| {
            let min = Pos2::new(x + EXPORT_MARGIN, y + EXPORT_MARGIN);
            let node_size = cache
                .sizes
                .get(id)
                .map(|(w, h)| egui::vec2(*w, *h))
                .unwrap_or(GRAPH_NODE_SIZE);
            (*id, Rect::from_min_size(min, node_size))
        })
        .collect();

    let colors = circuit_colors(palette);
    let file_name = format!(
        "bloq-program-r{}.svg",
        circuit_viewer.compile_revision.unwrap_or(0)
    );
    let svg = crate::svg_export::render_svg(size, |painter| {
        // A square background under the rounded canvas fills the corners the
        // rounded rect would otherwise leave transparent.
        painter.rect_filled(full_rect, 0.0, colors.canvas_fill);
        paint_program_graph(
            painter,
            full_rect,
            circuit_viewer,
            &layout,
            Pos2::new(EXPORT_MARGIN, EXPORT_MARGIN),
            None,
            palette,
        );
    });

    circuit_viewer.graph_zoom = saved_zoom;
    Some((file_name, svg))
}

/// Renders the selected node's current-moment patch view to SVG at the fixed
/// export pitch, ignoring the live pan/zoom. Returns `(file_name, svg)`, or
/// `None` when the node has no qubit layout or no current moment.
fn export_circuit_svg(
    circuit_viewer: &BloqViewerState,
    palette: &ThemePalette,
    theme_preset: ThemePreset,
) -> Option<(String, String)> {
    let bounds = coord_bounds(&circuit_viewer.qubit_coords)?;
    let moment = circuit_viewer.moments.get(circuit_viewer.current_moment)?;

    let pitch = CIRCUIT_EXPORT_PITCH;
    // Detector-slice lenses and qubit-site outlines overshoot the qubit centers
    // by about half a pitch, so pad beyond the plain grid margin.
    let margin = EXPORT_MARGIN + pitch;
    let cols = (bounds.max.x - bounds.min.x) as f32;
    let rows = (bounds.max.y - bounds.min.y) as f32;
    let size = egui::vec2(cols * pitch, rows * pitch) + egui::vec2(2.0 * margin, 2.0 * margin);
    let full_rect = Rect::from_min_size(Pos2::ZERO, size);
    // The size was built from the span, so centering the span lands it exactly at
    // the margins.
    let origin = circuit_origin(full_rect, bounds, pitch, glam::Vec2::ZERO);
    let geometry = CanvasGeometry {
        bounds,
        origin,
        pitch,
    };

    let colors = circuit_colors(palette);
    let file_name = if circuit_viewer.concurrent_ops {
        format!(
            "bloq-layer-z{}-m{}.svg",
            circuit_viewer.selected_layer,
            circuit_viewer.current_moment + 1,
        )
    } else {
        let suffix = if circuit_viewer.slice_visibility.detectors {
            "-detslice"
        } else if circuit_viewer.slice_visibility.observables {
            "-obsslice"
        } else {
            ""
        };
        format!(
            "bloq-node-N{}-m{}{}.svg",
            circuit_viewer.selected_node.unwrap_or(0),
            circuit_viewer.current_moment + 1,
            suffix
        )
    };
    let svg = crate::svg_export::render_svg(size, |painter| {
        painter.rect_filled(full_rect, 0.0, colors.canvas_fill);
        paint_circuit_canvas(
            painter,
            full_rect,
            circuit_viewer,
            moment,
            geometry,
            None,
            theme_preset,
        );
    });
    Some((file_name, svg))
}

fn node_visible(node: &BloqNodeView, circuit_viewer: &BloqViewerState) -> bool {
    circuit_viewer.show_classical || node.quantum_visible()
}

/// Recompute the memoized layered layout iff the node set or classical toggle
/// changed since last time. Zoom and pan are applied at draw time, so they
/// never invalidate the cache.
///
/// Region containers lay out hierarchically: each region's visible children are
/// laid out first (recursively), the region tile is sized to that content, and
/// the parent group then places the container like any other tile.
fn ensure_graph_layout(circuit_viewer: &mut BloqViewerState) {
    let key = (
        circuit_viewer.compile_revision,
        circuit_viewer.show_classical,
    );
    if circuit_viewer.graph_layout_cache.key == Some(key) {
        return;
    }

    let visible: Vec<&BloqNodeView> = circuit_viewer
        .nodes
        .iter()
        .filter(|node| node_visible(node, circuit_viewer))
        .collect();
    let visible_ids: HashSet<u32> = visible.iter().map(|node| node.id).collect();
    let edges: Vec<(usize, u32, u32)> = circuit_viewer
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| visible_ids.contains(&edge.from) && visible_ids.contains(&edge.to))
        .map(|(index, edge)| (index, edge.from, edge.to))
        .collect();

    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut roots: Vec<u32> = Vec::new();
    for node in &visible {
        match node.parent {
            Some(parent) if visible_ids.contains(&parent) => {
                children.entry(parent).or_default().push(node.id)
            }
            _ => roots.push(node.id),
        }
    }

    let mut positions = HashMap::new();
    let mut sizes = HashMap::new();
    let mut routes = HashMap::new();
    let content = layout_group(
        &roots,
        &children,
        &edges,
        &mut positions,
        &mut sizes,
        &mut routes,
    );

    circuit_viewer.graph_layout_cache = GraphLayoutCache {
        key: Some(key),
        positions,
        sizes,
        routes,
        content,
    };
}

/// Lay out one nesting level. Regions with visible children are sized from
/// their (recursively laid-out) child content; the children's positions are
/// then shifted inside the parent's rect. `positions` end up absolute in the
/// top-level zoom-1 space because every recursion level applies its offset on
/// the way out.
fn layout_group(
    members: &[u32],
    children: &HashMap<u32, Vec<u32>>,
    edges: &[(usize, u32, u32)],
    positions: &mut HashMap<u32, (f32, f32)>,
    sizes: &mut HashMap<u32, (f32, f32)>,
    routes: &mut HashMap<usize, super::graph_layout::CubicSpline>,
) -> (f32, f32) {
    let mut member_sizes: Vec<(u32, egui::Vec2)> = Vec::new();
    let mut nested: Vec<(u32, Vec<u32>)> = Vec::new();
    for &id in members {
        let size = match children.get(&id) {
            Some(kids) if !kids.is_empty() => {
                let kid_content = layout_group(kids, children, edges, positions, sizes, routes);
                nested.push((id, collect_subtree(kids, children)));
                egui::vec2(
                    kid_content.0 + 2.0 * REGION_PAD,
                    kid_content.1 + REGION_HEADER + REGION_PAD,
                )
            }
            _ => GRAPH_NODE_SIZE,
        };
        sizes.insert(id, (size.x, size.y));
        member_sizes.push((id, size));
    }

    let member_ids: HashSet<u32> = members.iter().copied().collect();
    let group_edges: Vec<_> = edges
        .iter()
        .filter(|(_, from, to)| member_ids.contains(from) && member_ids.contains(to))
        .collect();
    let endpoints: Vec<_> = group_edges
        .iter()
        .map(|(_, from, to)| (*from, *to))
        .collect();
    let relative =
        super::graph_layout::compute_relative(&member_sizes, &endpoints, GRAPH_COLUMN_GAP);
    positions.extend(relative.positions);
    routes.extend(
        group_edges
            .iter()
            .zip(relative.routes)
            .map(|(edge, route)| (edge.0, route)),
    );

    // Shift each region's whole subtree from group-local into this group's
    // coordinates (the subtree was laid out with the region's box at origin).
    for (region_id, subtree) in nested {
        let Some(&(region_x, region_y)) = positions.get(&region_id) else {
            continue;
        };
        let offset = (region_x + REGION_PAD, region_y + REGION_HEADER);
        let subtree_ids: HashSet<_> = subtree.iter().copied().collect();
        for id in subtree {
            if let Some(pos) = positions.get_mut(&id) {
                pos.0 += offset.0;
                pos.1 += offset.1;
            }
        }
        for (index, from, to) in edges {
            if subtree_ids.contains(from)
                && subtree_ids.contains(to)
                && let Some(route) = routes.get_mut(index)
            {
                for (x, y) in route.iter_mut().flatten() {
                    *x += offset.0;
                    *y += offset.1;
                }
            }
        }
    }

    relative.content
}

fn collect_subtree(members: &[u32], children: &HashMap<u32, Vec<u32>>) -> Vec<u32> {
    let mut out = members.to_vec();
    let mut cursor = 0;
    while cursor < out.len() {
        if let Some(kids) = children.get(&out[cursor]) {
            out.extend_from_slice(kids);
        }
        cursor += 1;
    }
    out
}

fn fit_program_graph(viewer: &mut BloqViewerState, size: egui::Vec2) {
    ensure_graph_layout(viewer);
    let (width, height) = viewer.graph_layout_cache.content;
    viewer.graph_zoom = ((size.x - 24.0).max(1.0) / width.max(1.0))
        .min((size.y - 24.0).max(1.0) / height.max(1.0))
        .min(1.0);
    viewer.graph_pan_offset = Vec2::ZERO;
}

/// Screen rects for the current frame: the cached zoom-1 positions scaled by
/// zoom and centered in the canvas (plus the user's pan). Cheap — no layout.
fn program_node_layout(rect: Rect, circuit_viewer: &mut BloqViewerState) -> HashMap<u32, Rect> {
    ensure_graph_layout(circuit_viewer);

    let zoom = circuit_viewer.graph_zoom;
    let cache = &circuit_viewer.graph_layout_cache;
    let origin = program_graph_origin(rect, circuit_viewer);

    cache
        .positions
        .iter()
        .map(|(id, (x, y))| {
            let min = origin + egui::vec2(x * zoom, y * zoom);
            let size = cache
                .sizes
                .get(id)
                .map(|(w, h)| egui::vec2(*w, *h))
                .unwrap_or(GRAPH_NODE_SIZE);
            (*id, Rect::from_min_size(min, size * zoom))
        })
        .collect()
}

fn program_graph_origin(rect: Rect, viewer: &BloqViewerState) -> Pos2 {
    let content = viewer.graph_layout_cache.content;
    rect.center() - egui::vec2(content.0, content.1) * (0.5 * viewer.graph_zoom)
        + egui::vec2(viewer.graph_pan_offset.x, viewer.graph_pan_offset.y)
}

fn draw_program_edge(
    painter: &egui::Painter,
    route: &super::graph_layout::CubicSpline,
    origin: Pos2,
    edge: &BloqEdgeView,
    zoom: f32,
    palette: &ThemePalette,
) {
    let (width, color) = match edge.kind {
        // A seam carrying temporal pipes is the "real" quantum connection; an
        // empty quantum edge is drawn thinner but in the same hue.
        BloqEdgeKind::Quantum => (
            if edge.has_pipes { 2.0 } else { 1.25 },
            palette.accent_primary,
        ),
        BloqEdgeKind::Value => (1.5, palette.success),
        BloqEdgeKind::Flip => (1.5, palette.accent_warn),
        BloqEdgeKind::Compose => (1.5, palette.accent_secondary),
        BloqEdgeKind::Order => (1.25, palette.text_dim),
    };
    let stroke = Stroke::new(width * zoom.max(0.5), color);

    for segment in route {
        painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
            segment.map(|(x, y)| origin + egui::vec2(x, y) * zoom),
            false,
            Color32::TRANSPARENT,
            stroke,
        ));
    }

    let Some(segment) = route.last() else {
        return;
    };
    let [_, _, c2, end] = segment.map(|(x, y)| origin + egui::vec2(x, y) * zoom);
    let dir = (end - c2).normalized();
    let head = 9.0 * zoom.max(0.5);
    let wing = 4.5 * zoom.max(0.5);
    let base = end - dir * head;
    let left = base + egui::vec2(-dir.y, dir.x) * wing;
    let right = base - egui::vec2(-dir.y, dir.x) * wing;
    painter.add(Shape::convex_polygon(
        vec![end, left, right],
        color,
        Stroke::NONE,
    ));
}

fn draw_program_node(
    painter: &egui::Painter,
    rect: Rect,
    node: &BloqNodeView,
    circuit_viewer: &BloqViewerState,
    hovered: bool,
    palette: &ThemePalette,
) {
    let selected = circuit_viewer.selected_node == Some(node.id);
    let is_region = node.category == BloqNodeCategory::Region;
    let accent = node_category_color(node.category, palette);
    let fill = if selected {
        palette.bg_active
    } else if hovered {
        palette.bg_hover
    } else {
        circuit_colors(palette).scrubber_fill
    };
    let stroke = if selected {
        Stroke::new(1.75, palette.accent_primary)
    } else {
        Stroke::new(if hovered { 1.5 } else { 1.25 }, accent)
    };
    // A region is a container: its body holds the child tiles and the seam edge
    // between them (painted in the earlier edge pass). An opaque fill would
    // cover both, so it keeps a translucent body — only a solid header band so
    // its label stays legible over whatever sits behind it.
    if is_region {
        let body = fill.gamma_multiply(0.16);
        painter.rect_filled(rect, 6.0, body);
        let header = Rect::from_min_max(
            rect.min,
            egui::pos2(
                rect.right(),
                rect.top() + REGION_HEADER * circuit_viewer.graph_zoom,
            ),
        );
        painter.rect_filled(header, 6.0, fill);
    } else {
        painter.rect_filled(rect, 6.0, fill);
    }
    painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Outside);
    // Only the node name; quantum tiles read in the bright text colour, classical
    // ones in their category accent so the kind is legible at a glance.
    let label_color = if node.category.is_quantum() {
        palette.text_bright
    } else {
        accent
    };
    // Scale the font with the graph zoom so the label tracks the tile size
    // instead of overflowing when zoomed out.
    let zoom = circuit_viewer.graph_zoom;
    let font = FontId::monospace((11.0 * zoom).clamp(5.0, 22.0));
    // Wrap the label to the tile interior and cap the rows: a region's name
    // rides a single header line, other tiles get two lines so a long instance
    // list (`i16,i17,…`) folds instead of spilling past the fixed-width box.
    // Overflow past the last row truncates with egui's default `…`; the full
    // label stays legible on hover (see the tooltip in the draw loop).
    let interior = (rect.width() - 20.0 * zoom).max(1.0);
    let mut job = egui::text::LayoutJob::simple(node.label.clone(), font, label_color, interior);
    job.wrap.max_rows = if is_region { 1 } else { 2 };
    let galley = painter.layout_job(job);
    // A region is a container: its label sits in the header band so the body
    // tiles drawn inside don't cover it; other tiles centre the block vertically.
    let center_y = if is_region {
        rect.top() + REGION_HEADER * 0.5 * zoom
    } else {
        rect.center().y
    };
    let pos = egui::pos2(rect.left() + 10.0 * zoom, center_y - galley.size().y * 0.5);
    painter.galley(pos, galley, label_color);
}

fn node_category_color(category: BloqNodeCategory, palette: &ThemePalette) -> Color32 {
    match category {
        BloqNodeCategory::QuantumBlock => palette.grey2,
        BloqNodeCategory::QuantumPipe => palette.accent_primary,
        BloqNodeCategory::Observable => palette.success,
        BloqNodeCategory::Region => palette.accent_warn,
        BloqNodeCategory::Other => palette.yellow,
    }
}

fn draw_node_circuit_window(
    ctx: &egui::Context,
    viewport: Rect,
    jobs: &EditorJobs,
    circuit_viewer: &mut BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
    theme_preset: ThemePreset,
) {
    let default_rect = tiled_window_rect(viewport, egui::Align2::RIGHT_BOTTOM);
    let max_size = viewport.size().max(egui::Vec2::splat(1.0));
    let min_size = egui::vec2(
        VIEWER_MIN_WIDTH.min(max_size.x),
        VIEWER_MIN_HEIGHT.min(max_size.y),
    );

    egui::Window::new(selected_node_title(circuit_viewer))
        .id(egui::Id::new(NODE_WINDOW_ID))
        .default_rect(default_rect)
        .constrain_to(viewport)
        .min_size(min_size)
        .max_size(max_size)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            draw_node_header(ui, circuit_viewer, intents, palette, theme_preset);
            if circuit_viewer.moments.is_empty() {
                ui.centered_and_justified(|ui| {
                    let message = if circuit_viewer.concurrent_ops {
                        "Selected layer has no concurrent circuit operations."
                    } else {
                        "Selected Bloq node has no circuit moments."
                    };
                    ui.label(egui::RichText::new(message).small().color(palette.text_dim));
                });
                return;
            }

            let body_size = egui::vec2(
                ui.available_width().max(1.0),
                ui.available_height().max(1.0),
            );
            let (body_rect, _) = ui.allocate_exact_size(body_size, Sense::hover());
            let layout = viewer_body_layout(body_rect, footer_height_budget(ui));

            let mut scrubber_ui =
                clipped_rect_child_ui(ui, "circuit-scrubber", layout.scrubber_rect);
            draw_scrubber(&mut scrubber_ui, circuit_viewer, intents, palette);

            let has_visible_regions = circuit_viewer
                .current_slice_regions()
                .iter()
                .any(|region| circuit_viewer.slice_visibility.shows(region.id));
            let slice_column = has_visible_regions.then(|| slice_column_layout(layout.canvas_rect));
            if let Some(column) = slice_column {
                let mut column_ui = clipped_rect_child_ui(ui, "slice-index", column.column_rect);
                draw_slice_index_column(
                    &mut column_ui,
                    circuit_viewer,
                    column.row_height,
                    column.row_gap,
                    palette,
                );
            }

            let circuit_rect =
                slice_column.map_or(layout.canvas_rect, |column| column.circuit_rect);
            let mut canvas_ui = clipped_rect_child_ui(ui, "circuit-canvas", circuit_rect);
            draw_canvas(
                &mut canvas_ui,
                circuit_viewer,
                circuit_rect.height().max(1.0),
                palette,
                theme_preset,
            );

            let mut footer_ui = clipped_rect_child_ui(ui, "circuit-footer", layout.footer_rect);
            draw_footer(&mut footer_ui, circuit_viewer, jobs, intents, palette);
        });
}

/// Nodes without a fixed circuit show their attributes instead of a timeline.
fn draw_node_attributes_window(
    ctx: &egui::Context,
    viewport: Rect,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let Some(id) = circuit_viewer.selected_node else {
        return;
    };
    let Some(node) = circuit_viewer.nodes.iter().find(|node| node.id == id) else {
        return;
    };

    let default_rect = tiled_window_rect(viewport, egui::Align2::RIGHT_BOTTOM);
    let max_size = viewport.size().max(egui::Vec2::splat(1.0));
    let min_size = egui::vec2(
        VIEWER_MIN_WIDTH.min(max_size.x),
        VIEWER_MIN_HEIGHT.min(max_size.y),
    );
    let accent = node_category_color(node.category, palette);

    egui::Window::new(format!("Program Node N{id}"))
        .id(egui::Id::new(NODE_WINDOW_ID))
        .default_rect(default_rect)
        .min_size(min_size)
        .max_size(max_size)
        .constrain_to(viewport)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                theme::status_chip(ui, "\u{f0e8}", format_args!("{}", node.label), accent);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let response = close_icon_button(ui, palette).on_hover_text("Close Node");
                    if activated(&response) {
                        intents.push(UiIntent::SelectBloqNode(None));
                    }
                });
            });
            ui.separator();

            if node.attributes.is_empty() {
                ui.label(
                    egui::RichText::new("No attributes.")
                        .small()
                        .color(palette.text_dim),
                );
                return;
            }
            // Both long values and large operator tables stay inside the window.
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .id_salt("node-attributes-scroll")
                .show(ui, |ui| {
                    egui::Grid::new("classical-node-attributes")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            for (field, value) in &node.attributes {
                                ui.label(
                                    egui::RichText::new(field).small().color(palette.text_dim),
                                );
                                ui.label(
                                    egui::RichText::new(value)
                                        .monospace()
                                        .color(palette.text_bright),
                                );
                                ui.end_row();
                            }
                        });

                    // An Observable's bound operators as an instance/face/operator
                    // table — an Observable may bind several, so flat rows would be
                    // unreadable.
                    if !node.operator_table.is_empty() {
                        ui.add_space(6.0);
                        egui::Grid::new("include-operator-table")
                            .num_columns(3)
                            .striped(true)
                            .show(ui, |ui| {
                                for header in ["instance", "face", "operator"] {
                                    ui.label(
                                        egui::RichText::new(header)
                                            .small()
                                            .strong()
                                            .color(palette.text_dim),
                                    );
                                }
                                ui.end_row();
                                for (instance, face, operator) in &node.operator_table {
                                    ui.label(
                                        egui::RichText::new(instance)
                                            .monospace()
                                            .color(palette.text_bright),
                                    );
                                    ui.label(
                                        egui::RichText::new(face)
                                            .monospace()
                                            .color(palette.text_bright),
                                    );
                                    ui.label(
                                        egui::RichText::new(operator)
                                            .monospace()
                                            .color(palette.text_bright),
                                    );
                                    ui.end_row();
                                }
                            });
                    }
                });
        });
}

fn selected_node_title(circuit_viewer: &BloqViewerState) -> String {
    if circuit_viewer.concurrent_ops {
        return format!("Program Layer z={} Circuit", circuit_viewer.selected_layer);
    }
    if let Some(id) = circuit_viewer.selected_node {
        format!("Program Node N{id} Circuit")
    } else {
        "Program Node Circuit".to_string()
    }
}

fn draw_node_header(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
    theme_preset: ThemePreset,
) {
    ui.horizontal_wrapped(|ui| {
        theme::status_chip(
            ui,
            "\u{f0e8}",
            format_args!("moments {}", circuit_viewer.moments.len()),
            palette.accent_primary,
        );
        theme::status_chip(
            ui,
            "\u{f111}",
            format_args!("qubits {}", circuit_viewer.num_qubits),
            palette.text_primary,
        );
        if !circuit_viewer.concurrent_ops
            && let Some(id) = circuit_viewer.selected_node
            && let Some(node) = circuit_viewer.nodes.iter().find(|node| node.id == id)
            && let Some(layer) = circuit_viewer.node_source_layer(node)
        {
            theme::status_chip(ui, "\u{f126}", format_args!("z={layer}"), palette.text_dim);
        }
    });

    ui.horizontal_wrapped(|ui| {
        let response = close_icon_button(ui, palette).on_hover_text("Close Node Circuit");
        if activated(&response) {
            intents.push(UiIntent::SelectBloqNode(None));
        }
        draw_slice_toggles(ui, circuit_viewer, intents);
        draw_concurrent_controls(ui, circuit_viewer, intents);

        // Enabled only when there is a patch to draw: a current moment and a
        // qubit layout to project it onto.
        let exportable = !circuit_viewer.moments.is_empty()
            && coord_bounds(&circuit_viewer.qubit_coords).is_some();
        let svg_response = ui
            .add_enabled(
                exportable,
                egui::Button::new(
                    egui::RichText::new("SVG")
                        .small()
                        .color(palette.accent_primary),
                ),
            )
            .on_hover_text("Export view as SVG");
        if activated(&svg_response)
            && let Some((file_name, contents)) =
                export_circuit_svg(circuit_viewer, palette, theme_preset)
        {
            intents.push(UiIntent::SaveSvg {
                file_name,
                contents,
            });
        }
    });
}

fn draw_concurrent_controls(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
) {
    if circuit_viewer.concurrent_ops {
        let mut layer = circuit_viewer.selected_layer;
        let mut layers = circuit_viewer
            .layer_circuits
            .keys()
            .copied()
            .collect::<Vec<_>>();
        layers.sort_unstable();
        egui::ComboBox::from_id_salt("concurrent-source-layer")
            .selected_text(format!("z={layer}"))
            .show_ui(ui, |ui| {
                for candidate in layers {
                    ui.selectable_value(&mut layer, candidate, candidate.to_string());
                }
            })
            .response
            .on_hover_text("Source layer to project and merge");
        if layer != circuit_viewer.selected_layer {
            intents.push(UiIntent::SetConcurrentLayer(layer));
        }
    }

    let state = circuit_viewer.concurrent_ops_toggle();
    if matches!(state, LazyToggle::Pending) {
        ui.add(egui::Spinner::new().size(14.0));
    }
    let mut enabled = matches!(state, LazyToggle::On | LazyToggle::Pending);
    if draw_lazy_checkbox(
        ui,
        &state,
        &mut enabled,
        "concurrent ops",
        "Show compatible node instructions concurrently on one source z layer",
    ) {
        intents.push(UiIntent::SetConcurrentOps(enabled));
    }
}

/// Slice toggles in the node header. Their shared overlay is computed lazily, so
/// both checkboxes are disabled while the job runs or if flattening fails.
fn draw_slice_toggles(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
) {
    let detslice = circuit_viewer.detslice_toggle();
    let obsslice = circuit_viewer.obsslice_toggle();

    if matches!(detslice, LazyToggle::Pending) {
        ui.add(egui::Spinner::new().size(14.0));
    }

    let mut visibility = circuit_viewer.slice_visibility;
    let det_changed = draw_lazy_checkbox(
        ui,
        &detslice,
        &mut visibility.detectors,
        "detslice",
        "Toggle detector-slice overlay (D)",
    );
    let obs_changed = draw_lazy_checkbox(
        ui,
        &obsslice,
        &mut visibility.observables,
        "obsslice",
        "Toggle logical-observable-slice overlay",
    );
    if det_changed || obs_changed {
        intents.push(UiIntent::SetSliceVisibility(visibility));
    }
}

fn draw_lazy_checkbox(
    ui: &mut egui::Ui,
    state: &LazyToggle,
    checked: &mut bool,
    label: &str,
    hover: &str,
) -> bool {
    let clickable = matches!(state, LazyToggle::Off | LazyToggle::On);
    let response = ui
        .push_id(label, |ui| {
            ui.add_enabled(clickable, egui::Checkbox::new(checked, label))
        })
        .inner;
    let response = match state {
        LazyToggle::Unavailable(reason) => response.on_disabled_hover_text(reason.clone()),
        LazyToggle::Pending => response.on_disabled_hover_text("Computing…"),
        LazyToggle::Absent => response.on_disabled_hover_text("Compile a program first"),
        LazyToggle::Off | LazyToggle::On => response.on_hover_text(hover),
    };
    response.changed()
}

fn close_icon_button(ui: &mut egui::Ui, palette: &ThemePalette) -> egui::Response {
    let size = egui::vec2(30.0, 24.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), "Close viewer")
    });
    let text_color = if response.hovered() {
        palette.text_bright
    } else {
        palette.text_primary
    };
    if response.hovered() {
        ui.painter().rect_filled(rect, 4.0, palette.bg_hover);
        ui.painter().rect_stroke(
            rect,
            4.0,
            Stroke::new(1.0, palette.border_bright),
            egui::StrokeKind::Inside,
        );
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "\u{f00d}",
        FontId::monospace(13.0),
        text_color,
    );
    response
}

fn draw_empty_state(
    ui: &mut egui::Ui,
    compile_ui: &CompileUiState,
    jobs: &EditorJobs,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    theme::section_frame(ui, palette, |ui| {
        ui.set_min_height(180.0);
        ui.vertical_centered(|ui| {
            ui.add_space(24.0);
            let message = if let Some(error) = &circuit_viewer.last_error {
                egui::RichText::new(error)
                    .small()
                    .color(palette.accent_error)
            } else if circuit_viewer.pending_revision.is_some() {
                egui::RichText::new("Bloq compilation is running...")
                    .small()
                    .color(palette.accent_warn)
            } else {
                egui::RichText::new("No compiled Bloq is loaded.")
                    .small()
                    .color(palette.text_dim)
            };
            ui.label(message);
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!(
                    "Current compile settings: d={}",
                    compile_ui.code_distance_input.trim()
                ))
                .small()
                .color(palette.text_dim),
            );
            ui.add_space(12.0);
            let response = ui.add_enabled(
                !jobs.compilation_running(),
                egui::Button::new(
                    egui::RichText::new("Compile Bloq")
                        .small()
                        .color(palette.accent_primary),
                ),
            );
            if activated(&response) {
                intents.push(UiIntent::CompileForViewer);
            }
        });
    });
}

fn draw_scrubber(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), VIEWER_SCRUBBER_HEIGHT),
        Sense::click(),
    );
    let painter = ui.painter();
    let colors = circuit_colors(palette);
    painter.rect_filled(rect, 4.0, colors.scrubber_fill);
    let track_rect = Rect::from_min_max(rect.min, egui::pos2(rect.right(), rect.top() + 20.0));

    let moment_count = circuit_viewer.moments.len();
    for (index, moment) in circuit_viewer.moments.iter().enumerate() {
        let slot_rect = scrubber_slot_rect(track_rect, index, moment_count);
        let bar_rect = scrubber_bar_rect(slot_rect, track_rect);
        painter.rect_filled(bar_rect, 1.0, scrubber_color(moment, palette));
        if let (Some(label), Some(span)) = (&moment.repeat_label, moment.repeat_span) {
            draw_repeat_range_label(
                painter,
                label,
                track_rect,
                index,
                span,
                moment_count,
                palette,
            );
        }
        if index == circuit_viewer.current_moment {
            let mid_x = slot_rect.center().x;
            painter.add(Shape::convex_polygon(
                vec![
                    egui::pos2(mid_x, track_rect.top() + 1.0),
                    egui::pos2(mid_x - 4.0, track_rect.top() + 7.0),
                    egui::pos2(mid_x + 4.0, track_rect.top() + 7.0),
                ],
                colors.active_marker,
                Stroke::NONE,
            ));
            painter.rect_stroke(
                bar_rect,
                1.0,
                Stroke::new(1.25, colors.active_marker),
                egui::StrokeKind::Outside,
            );
        }
    }

    if let Some(pos) = response.hover_pos() {
        let hovered_index = hovered_scrubber_index(track_rect, pos, moment_count);
        let repeat_text = circuit_viewer
            .moments
            .get(hovered_index)
            .and_then(|moment| moment.repeat_label.as_deref())
            .map(|label| format!(" · {label}"))
            .unwrap_or_default();
        response.clone().on_hover_text(format!(
            "Moment {}/{}{}",
            hovered_index + 1,
            moment_count,
            repeat_text
        ));
    }

    if activated(&response)
        && let Some(pos) = response.interact_pointer_pos()
    {
        intents.push(UiIntent::SetCircuitMoment(hovered_scrubber_index(
            track_rect,
            pos,
            moment_count,
        )));
    }
}

fn draw_repeat_range_label(
    painter: &egui::Painter,
    label: &str,
    track_rect: Rect,
    start: usize,
    span: usize,
    moment_count: usize,
    palette: &ThemePalette,
) {
    let end = (start + span.saturating_sub(1)).min(moment_count.saturating_sub(1));
    let start_rect = scrubber_slot_rect(track_rect, start, moment_count);
    let end_rect = scrubber_slot_rect(track_rect, end, moment_count);
    let left = start_rect.left();
    let right = end_rect.right();
    let bracket_y = track_rect.bottom() + 14.0;
    let tick_top = track_rect.bottom() + 5.0;
    let stroke = Stroke::new(1.0, palette.accent_secondary);

    painter.line_segment(
        [egui::pos2(left, tick_top), egui::pos2(left, bracket_y)],
        stroke,
    );
    painter.line_segment(
        [egui::pos2(left, bracket_y), egui::pos2(right, bracket_y)],
        stroke,
    );
    painter.line_segment(
        [egui::pos2(right, tick_top), egui::pos2(right, bracket_y)],
        stroke,
    );
    painter.text(
        egui::pos2(f32::midpoint(left, right), bracket_y + 15.0),
        Align2::CENTER_BOTTOM,
        label,
        FontId::monospace(10.0),
        palette.accent_secondary,
    );
}

/// Pitch-derived glyph sizes for the circuit canvas, computed in one place so the
/// interactive wrapper (which also needs the tooltip radius) and the pure paint
/// pass cannot drift apart.
struct CanvasMetrics {
    site_size: f32,
    gate_size: f32,
    control_radius: f32,
}

fn canvas_metrics(pitch: f32) -> CanvasMetrics {
    CanvasMetrics {
        site_size: (pitch * 0.38).clamp(3.0, 16.0),
        gate_size: (pitch * 0.85).clamp(8.0, 34.0),
        control_radius: (pitch * 0.3).clamp(4.0, 14.0),
    }
}

/// Keeps reset/measurement badges inside a single-term slice marker at fit zoom.
fn compact_gate_size(gate_size: f32) -> f32 {
    (gate_size * 0.72).max(8.0)
}

fn draw_slice_index_column(
    ui: &mut egui::Ui,
    circuit_viewer: &mut BloqViewerState,
    row_height: f32,
    row_gap: f32,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    let slice_colors = DetsliceColors::from_circuit_colors(&colors);
    let selected_region = circuit_viewer.selected_region;
    let visibility = circuit_viewer.slice_visibility;
    let column_rect = ui.max_rect();
    ui.painter()
        .rect_filled(column_rect, 6.0, colors.canvas_fill);

    let current_regions = circuit_viewer.current_slice_regions();
    let mut regions: Vec<&SliceRegionView> = current_regions
        .iter()
        .filter(|region| visibility.shows(region.id))
        .collect();
    regions.sort_by_key(|region| match region.id {
        SliceRegionId::Detector {
            owner_node,
            detector,
        } => (0, detector, owner_node),
        SliceRegionId::Observable { index } => (1, index, 0),
    });

    let clicked = egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let mut clicked = None;
            ui.spacing_mut().item_spacing.y = row_gap;
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let row_width = column_rect.width() - 8.0;
                for region in regions {
                    let (label, observable) = match region.id {
                        SliceRegionId::Detector { detector, .. } => (format!("D{detector}"), false),
                        SliceRegionId::Observable { index } => (format!("L{index}"), true),
                    };
                    let selected = selected_region == Some(region.id);
                    let alpha: f32 = if selected_region.is_none() || selected {
                        1.0
                    } else {
                        0.25
                    };
                    let border = colors.detslice_outline.gamma_multiply(alpha.max(0.5));
                    let border_width = if selected { 2.0 } else { 1.0 };
                    let stroke = if observable {
                        Stroke::NONE
                    } else {
                        Stroke::new(border_width, border)
                    };
                    let response = ui
                        .push_id(region.id, |ui| {
                            ui.add_sized(
                                egui::vec2(row_width, row_height),
                                egui::Button::new(
                                    egui::RichText::new(&label)
                                        .monospace()
                                        .size(8.0)
                                        .color(border),
                                )
                                .fill(slice_colors.region_color(region).gamma_multiply(alpha))
                                .stroke(stroke)
                                .corner_radius((row_height * 0.25).min(4.0))
                                .truncate(),
                            )
                        })
                        .inner
                        .on_hover_text(format!("{label} · {}", pauli_composition(region)));

                    if observable {
                        let rect = response.rect.shrink(1.0);
                        let path = [
                            rect.left_top(),
                            rect.right_top(),
                            rect.right_bottom(),
                            rect.left_bottom(),
                            rect.left_top(),
                        ];
                        ui.painter().extend(Shape::dashed_line(
                            &path,
                            Stroke::new(border_width, border),
                            (row_height * 0.3).min(5.0),
                            (row_height * 0.2).min(3.0),
                        ));
                    }
                    if selected {
                        response.scroll_to_me(Some(egui::Align::Center));
                    }
                    if activated(&response) {
                        clicked = Some(region.id);
                    }
                }
            });
            clicked
        })
        .inner;

    if let Some(id) = clicked {
        circuit_viewer.selected_region = (selected_region != Some(id)).then_some(id);
        circuit_viewer.hovered_region = None;
    }
}

/// Interactive wrapper around [`paint_circuit_canvas`]: handles pan/zoom input,
/// derives the geometry from the live rect, tracks the hovered detector region,
/// and shows pointer tooltips. All drawing goes through the shared pure pass.
fn draw_canvas(
    ui: &mut egui::Ui,
    circuit_viewer: &mut BloqViewerState,
    canvas_height: f32,
    palette: &ThemePalette,
    theme_preset: ThemePreset,
) {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), canvas_height),
        Sense::drag(),
    );
    let colors = circuit_colors(palette);

    if response.dragged() {
        let delta = ui.ctx().input(|input| input.pointer.delta());
        circuit_viewer.pan_offset += glam::Vec2::new(delta.x, delta.y);
    }

    if response.hovered() {
        let scroll_delta = ui.ctx().input(|input| input.smooth_scroll_delta.y);
        if scroll_delta.abs() > f32::EPSILON {
            let zoom_factor = (scroll_delta / 600.0).exp();
            circuit_viewer.zoom = (circuit_viewer.zoom * zoom_factor).clamp(0.35, 4.0);
        }
    }

    let painter = ui.painter();

    // Empty-layout guards paint their own background since the pure paint pass
    // (which normally fills it) is not reached in these cases.
    let Some(bounds) = coord_bounds(&circuit_viewer.qubit_coords) else {
        painter.rect_filled(rect, 6.0, colors.canvas_fill);
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "No qubit layout",
            FontId::monospace(14.0),
            palette.text_dim,
        );
        return;
    };

    let Some(moment) = circuit_viewer.moments.get(circuit_viewer.current_moment) else {
        painter.rect_filled(rect, 6.0, colors.canvas_fill);
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "Moment out of range",
            FontId::monospace(14.0),
            palette.text_dim,
        );
        return;
    };

    let content_rect = rect.shrink(16.0);
    let fit_pitch = fitted_pitch(content_rect, bounds);
    let pitch = (fit_pitch * circuit_viewer.zoom).clamp(VIEWER_MIN_PITCH, VIEWER_MAX_PITCH);
    let origin = circuit_origin(content_rect, bounds, pitch, circuit_viewer.pan_offset);
    let geometry = CanvasGeometry {
        bounds,
        origin,
        pitch,
    };
    let tooltip_radius = (canvas_metrics(pitch).site_size * 0.75).max(8.0);

    // Detector-slice hover isolation: find the region under the pointer and store
    // it so the pure paint pass dims the rest and the tooltip below can render.
    let hovered_region = if circuit_viewer.slice_visibility.any() {
        let hovered = response.hover_pos().and_then(|pointer| {
            let regions = circuit_viewer.current_slice_regions();
            detslice_draw::region_at_pointer(
                &regions,
                geometry,
                pointer,
                circuit_viewer.slice_visibility,
            )
        });
        circuit_viewer.hovered_region = hovered;
        hovered
    } else {
        None
    };
    let isolated_region = circuit_viewer.selected_region.or(hovered_region);

    paint_circuit_canvas(
        painter,
        rect,
        circuit_viewer,
        moment,
        geometry,
        isolated_region,
        theme_preset,
    );

    // A hovered region gets a tooltip: which detector owns it, its coords, Pauli
    // composition, and a cross-node badge when the owner is a different node than
    // the one whose timeline this is. Pointer-driven, so it stays out of the pure
    // paint pass (and the export).
    if circuit_viewer.slice_visibility.any()
        && let (Some(id), Some(pointer)) = (hovered_region, response.hover_pos())
    {
        let regions = circuit_viewer.current_slice_regions();
        if let Some(region) = regions.iter().find(|region| region.id() == id) {
            draw_region_hint_window(
                ui.ctx(),
                pointer,
                region,
                circuit_viewer.selected_node,
                palette,
            );
        }
    }

    if let Some(pointer) = response.hover_pos()
        && let Some((qubit, coord)) =
            hovered_qubit_site(circuit_viewer, geometry, pointer, tooltip_radius)
    {
        draw_qubit_hint_window(ui.ctx(), pointer, qubit, coord, palette);
    }
}

/// Paints the circuit background, slices, qubits, and operations.
/// Shared by the live canvas and SVG export.
fn paint_circuit_canvas(
    painter: &egui::Painter,
    rect: Rect,
    circuit_viewer: &BloqViewerState,
    moment: &FlatMoment,
    geometry: CanvasGeometry,
    isolated_region: Option<SliceRegionId>,
    theme_preset: ThemePreset,
) {
    let palette = theme::palette(theme_preset);
    let colors = circuit_colors(palette);
    let metrics = canvas_metrics(geometry.pitch);

    painter.rect_filled(rect, 6.0, colors.canvas_fill);

    // Detector-slice overlay: detection regions for the current flattened moment,
    // drawn under the qubit sites and gate glyphs, plus any anticommutation break
    // markers the tape recorded there.
    if circuit_viewer.slice_visibility.any() {
        let regions = circuit_viewer.current_slice_regions();
        let breaks = circuit_viewer.current_slice_breaks();
        draw_detector_slices(
            painter,
            &regions,
            &breaks,
            geometry,
            DetsliceColors::from_circuit_colors(&colors),
            isolated_region,
            circuit_viewer.slice_visibility,
        );
    }

    for &coord in circuit_viewer.qubit_coords.values() {
        let pos = qubit_pos(coord, geometry);
        let site_rect =
            Rect::from_center_size(pos, egui::vec2(metrics.site_size, metrics.site_size));
        painter.rect_filled(site_rect, 1.0, colors.site_fill);
        painter.rect_stroke(
            site_rect,
            1.0,
            Stroke::new(1.0, colors.site_stroke),
            egui::StrokeKind::Outside,
        );
    }

    draw_moment_ops(
        painter,
        &moment.ops,
        circuit_viewer,
        geometry,
        metrics.gate_size,
        metrics.control_radius,
        palette,
    );
}

fn draw_footer(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    jobs: &EditorJobs,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let footer_height = footer_height_budget(ui);
    egui::ScrollArea::horizontal()
        .auto_shrink([false, true])
        .max_height(footer_height)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let current = circuit_viewer.current_moment;
                let total = circuit_viewer.moments.len();
                let can_step = total > 0;

                let response =
                    ui.add_enabled(can_step && current > 0, egui::Button::new("\u{f053} Prev"));
                if activated(&response) {
                    intents.push(UiIntent::SetCircuitMoment(current.saturating_sub(1)));
                }
                let response = ui.add_enabled(
                    can_step && current + 1 < total,
                    egui::Button::new("Next \u{f054}"),
                );
                if activated(&response) {
                    intents.push(UiIntent::SetCircuitMoment(current + 1));
                }

                ui.label(
                    egui::RichText::new(format!("Moment {}/{}", current + 1, total))
                        .small()
                        .color(palette.text_bright),
                );

                ui.separator();
                let response = ui.add_enabled(
                    !jobs.compilation_running(),
                    egui::Button::new(
                        egui::RichText::new("Recompile")
                            .small()
                            .color(palette.accent_primary),
                    ),
                );
                if activated(&response) {
                    intents.push(UiIntent::CompileForViewer);
                }

                // Kept last in the row so toggling the mode on/off never shifts
                // the interactive buttons' auto-generated ids (egui derives them
                // from widget order, which would otherwise warn on a collision).
                if circuit_viewer.slice_visibility.any() {
                    draw_detslice_legend(ui, circuit_viewer, palette);
                }
            });
        });
}

/// The detector-slice legend shown in the footer while the mode is on: an
/// X/Y/Z colour key and the live count of regions at the current moment.
fn draw_detslice_legend(
    ui: &mut egui::Ui,
    circuit_viewer: &BloqViewerState,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    ui.separator();
    for (label, color) in [
        ("X", colors.detslice_x),
        ("Y", colors.detslice_y),
        ("Z", colors.detslice_z),
    ] {
        ui.label(egui::RichText::new(label).small().strong().color(color));
    }
    let regions = circuit_viewer.current_slice_regions();
    let observables = regions
        .iter()
        .filter(|region| matches!(region.id, SliceRegionId::Observable { .. }))
        .count();
    let detectors = regions.len() - observables;
    ui.label(
        egui::RichText::new(format!("· D {detectors} · L {observables}"))
            .small()
            .color(palette.text_dim),
    );
    // Non-Clifford axes and region seams drop regions from the tape; flag it so
    // T/factory programs do not look like they are silently missing regions.
    if let Some(note) = &circuit_viewer.detslice_note {
        ui.label(
            egui::RichText::new(format!("· {note}"))
                .small()
                .color(palette.accent_warn),
        );
    }
}

fn draw_moment_ops(
    painter: &egui::Painter,
    ops: &[FlatMomentOp],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    gate_size: f32,
    control_radius: f32,
    palette: &ThemePalette,
) {
    for op in ops {
        match op {
            FlatMomentOp::Gate { gate, qubits } if gate.is_two_qubit_gate() => draw_interaction_op(
                painter,
                *gate,
                qubits,
                circuit_viewer,
                geometry,
                control_radius,
                palette,
            ),
            FlatMomentOp::Gate { gate, qubits } => draw_single_qubit_op(
                painter,
                *gate,
                qubits,
                circuit_viewer,
                geometry,
                gate_size,
                palette,
            ),
            FlatMomentOp::Measure { basis, qubits } => draw_measurement_op(
                painter,
                *basis,
                qubits,
                circuit_viewer,
                geometry,
                gate_size,
                palette,
            ),
            FlatMomentOp::Mpp { products } => draw_mpp_op(
                painter,
                products,
                circuit_viewer,
                geometry,
                gate_size,
                palette,
            ),
        }
    }
}

/// A multi-Pauli-product measurement: each product's qubits are joined by a
/// wire and stamped with a per-basis measurement glyph (`MX`/`MY`/`M`), so a
/// `Port` block's boundary stabilizers read like the joint measurement they are.
fn draw_mpp_op(
    painter: &egui::Painter,
    products: &[Vec<(u32, PauliBasis)>],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    gate_size: f32,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    let gate_size = compact_gate_size(gate_size);
    for product in products {
        // Join the product's support so the jointly-measured qubits read as one
        // operator rather than independent single-qubit measurements.
        let positions: Vec<Pos2> = product
            .iter()
            .filter_map(|&(qubit, _)| circuit_viewer.qubit_coords.get(&(qubit as usize)))
            .map(|&coord| qubit_pos(coord, geometry))
            .collect();
        for pair in positions.windows(2) {
            painter.line_segment(
                [pair[0], pair[1]],
                Stroke::new((2.0 * circuit_viewer.zoom).clamp(1.0, 4.0), colors.wire),
            );
        }
        for &(qubit, basis) in product {
            let Some(&coord) = circuit_viewer.qubit_coords.get(&(qubit as usize)) else {
                continue;
            };
            let center = qubit_pos(coord, geometry);
            let style = measurement_style(basis);
            let rect = Rect::from_center_size(center, egui::vec2(gate_size, gate_size));
            painter.rect_filled(rect, 2.0, style.fill);
            painter.rect_stroke(
                rect,
                2.0,
                Stroke::new(1.0, colors.gate_stroke),
                egui::StrokeKind::Outside,
            );
            painter.text(
                center,
                Align2::CENTER_CENTER,
                style.label.as_str(),
                FontId::monospace((gate_size * 0.42).clamp(6.0, 18.0)),
                style.text,
            );
        }
    }
}

fn draw_single_qubit_op(
    painter: &egui::Painter,
    gate: GateType,
    qubits: &[u32],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    gate_size: f32,
    palette: &ThemePalette,
) {
    let gate_size = if gate.is_reset() {
        compact_gate_size(gate_size)
    } else {
        gate_size
    };
    draw_qubit_marker_op(
        painter,
        qubits,
        circuit_viewer,
        geometry,
        gate_size,
        gate_style(gate, palette),
        palette,
    );
}

fn draw_measurement_op(
    painter: &egui::Painter,
    basis: PauliBasis,
    qubits: &[u32],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    gate_size: f32,
    palette: &ThemePalette,
) {
    draw_qubit_marker_op(
        painter,
        qubits,
        circuit_viewer,
        geometry,
        compact_gate_size(gate_size),
        measurement_style(basis),
        palette,
    );
}

fn draw_qubit_marker_op(
    painter: &egui::Painter,
    qubits: &[u32],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    gate_size: f32,
    style: GateStyle,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    for qubit in qubits {
        let qubit = *qubit as usize;
        let Some(&coord) = circuit_viewer.qubit_coords.get(&qubit) else {
            continue;
        };
        let center = qubit_pos(coord, geometry);
        let rect = Rect::from_center_size(center, egui::vec2(gate_size, gate_size));
        painter.rect_filled(rect, 2.0, style.fill);
        painter.rect_stroke(
            rect,
            2.0,
            Stroke::new(1.0, colors.gate_stroke),
            egui::StrokeKind::Outside,
        );
        painter.text(
            center,
            Align2::CENTER_CENTER,
            style.label.as_str(),
            FontId::monospace((gate_size * 0.42).clamp(6.0, 18.0)),
            style.text,
        );
    }
}

fn draw_interaction_op(
    painter: &egui::Painter,
    gate: GateType,
    qubits: &[u32],
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    control_radius: f32,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    for pair in qubits.as_chunks::<2>().0 {
        let lhs = pair[0] as usize;
        let rhs = pair[1] as usize;
        let (Some(&lhs_coord), Some(&rhs_coord)) = (
            circuit_viewer.qubit_coords.get(&lhs),
            circuit_viewer.qubit_coords.get(&rhs),
        ) else {
            continue;
        };
        let lhs_pos = qubit_pos(lhs_coord, geometry);
        let rhs_pos = qubit_pos(rhs_coord, geometry);
        painter.line_segment(
            [lhs_pos, rhs_pos],
            Stroke::new((2.0 * circuit_viewer.zoom).clamp(1.0, 4.0), colors.wire),
        );
        if let Some((lhs_basis, rhs_basis)) = gate.two_qubit_bases() {
            draw_basis_symbol(painter, lhs_pos, lhs_basis, control_radius, palette);
            draw_basis_symbol(painter, rhs_pos, rhs_basis, control_radius, palette);
        } else {
            let mid = lhs_pos.lerp(rhs_pos, 0.5);
            let rect = Rect::from_center_size(
                mid,
                egui::vec2(
                    (geometry.pitch * 1.3).clamp(16.0, 32.0),
                    (geometry.pitch * 0.75).clamp(12.0, 18.0),
                ),
            );
            painter.rect_filled(rect, 2.0, colors.interaction_label_fill);
            painter.rect_stroke(
                rect,
                2.0,
                Stroke::new(1.0, colors.gate_stroke),
                egui::StrokeKind::Outside,
            );
            painter.text(
                mid,
                Align2::CENTER_CENTER,
                gate_label(gate),
                FontId::monospace((geometry.pitch * 0.35).clamp(6.0, 16.0)),
                colors.interaction_label_text,
            );
        }
    }
}

fn draw_basis_symbol(
    painter: &egui::Painter,
    center: Pos2,
    basis: PauliBasis,
    radius: f32,
    palette: &ThemePalette,
) {
    let colors = circuit_colors(palette);
    match basis {
        PauliBasis::Z => {
            painter.circle_filled(center, radius, colors.basis_z_fill);
        }
        PauliBasis::X => {
            painter.circle_filled(center, radius, colors.basis_x_fill);
            painter.circle_stroke(center, radius, Stroke::new(1.0, colors.wire));
            painter.line_segment(
                [
                    egui::pos2(center.x - radius * 0.55, center.y),
                    egui::pos2(center.x + radius * 0.55, center.y),
                ],
                Stroke::new(1.0, colors.wire),
            );
            painter.line_segment(
                [
                    egui::pos2(center.x, center.y - radius * 0.55),
                    egui::pos2(center.x, center.y + radius * 0.55),
                ],
                Stroke::new(1.0, colors.wire),
            );
        }
        PauliBasis::Y => {
            painter.add(Shape::convex_polygon(
                vec![
                    egui::pos2(center.x, center.y - radius),
                    egui::pos2(center.x + radius, center.y),
                    egui::pos2(center.x, center.y + radius),
                    egui::pos2(center.x - radius, center.y),
                ],
                colors.basis_y_fill,
                Stroke::new(1.0, colors.wire),
            ));
        }
    }
}

fn hovered_scrubber_index(rect: Rect, pos: Pos2, moment_count: usize) -> usize {
    if moment_count == 0 || rect.width() <= f32::EPSILON {
        return 0;
    }
    let ratio = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 0.999_999);
    (ratio * moment_count as f32).floor() as usize
}

fn footer_height_budget(ui: &egui::Ui) -> f32 {
    ui.spacing().interact_size.y.max(24.0) + ui.spacing().item_spacing.y + 6.0
}

fn viewer_body_layout(body_rect: Rect, footer_height: f32) -> ViewerBodyLayout {
    let scrubber_rect = Rect::from_min_size(
        body_rect.min,
        egui::vec2(
            body_rect.width(),
            VIEWER_SCRUBBER_HEIGHT.min(body_rect.height()),
        ),
    );
    let desired_footer_top =
        body_rect.min.y + VIEWER_SCRUBBER_HEIGHT + 16.0 + VIEWER_CANVAS_MIN_HEIGHT;
    let footer_top = (body_rect.max.y - footer_height)
        .max(desired_footer_top)
        .min(body_rect.max.y - 1.0);
    let footer_rect = Rect::from_min_max(egui::pos2(body_rect.left(), footer_top), body_rect.max);
    let canvas_top = (scrubber_rect.max.y + 8.0).min(body_rect.max.y);
    let canvas_bottom = (footer_rect.min.y - 8.0).max(canvas_top + 1.0);
    let canvas_rect = Rect::from_min_max(
        egui::pos2(body_rect.left(), canvas_top),
        egui::pos2(body_rect.right(), canvas_bottom),
    );

    ViewerBodyLayout {
        scrubber_rect,
        canvas_rect,
        footer_rect,
    }
}

fn slice_column_layout(canvas_rect: Rect) -> SliceColumnLayout {
    let column_width = (canvas_rect.width() * 0.09).clamp(30.0, 52.0);
    let column_gap = (canvas_rect.width() * 0.012).clamp(4.0, 8.0);
    let column_rect = Rect::from_min_max(
        egui::pos2(canvas_rect.right() - column_width, canvas_rect.top()),
        canvas_rect.right_bottom(),
    );
    let circuit_rect = Rect::from_min_max(
        canvas_rect.left_top(),
        egui::pos2(column_rect.left() - column_gap, canvas_rect.bottom()),
    );

    SliceColumnLayout {
        circuit_rect,
        column_rect,
        row_height: (canvas_rect.height() * 0.055).clamp(12.0, 20.0),
        row_gap: (canvas_rect.height() * 0.008).clamp(2.0, 4.0),
    }
}

fn clipped_rect_child_ui(ui: &mut egui::Ui, id_salt: &'static str, rect: Rect) -> egui::Ui {
    let id = ui.id().with(id_salt);
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id(id)
            .max_rect(rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    child.set_clip_rect(rect);
    child
}

fn scrubber_slot_rect(rect: Rect, index: usize, moment_count: usize) -> Rect {
    if moment_count == 0 {
        return Rect::from_min_max(rect.left_top(), rect.left_top());
    }

    let left = rect.left() + rect.width() * (index as f32 / moment_count as f32);
    let right = rect.left() + rect.width() * ((index + 1) as f32 / moment_count as f32);
    Rect::from_min_max(
        egui::pos2(left, rect.top()),
        egui::pos2(right, rect.bottom()),
    )
}

fn scrubber_bar_rect(slot_rect: Rect, track_rect: Rect) -> Rect {
    let horizontal_inset = if slot_rect.width() > 2.0 { 1.0 } else { 0.0 };
    Rect::from_min_max(
        egui::pos2(slot_rect.left() + horizontal_inset, track_rect.top() + 5.0),
        egui::pos2(
            slot_rect.right() - horizontal_inset,
            track_rect.bottom() - 3.0,
        ),
    )
}

fn hovered_qubit_site(
    circuit_viewer: &BloqViewerState,
    geometry: CanvasGeometry,
    pointer: Pos2,
    radius: f32,
) -> Option<(usize, IVec2)> {
    let radius_sq = radius * radius;
    circuit_viewer
        .qubit_coords
        .iter()
        .filter_map(|(index, coord)| {
            let pos = qubit_pos(*coord, geometry);
            let distance_sq = pos.distance_sq(pointer);
            (distance_sq <= radius_sq).then_some((*index, *coord, distance_sq))
        })
        .min_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2))
        .map(|(index, coord, _)| (index, coord))
}

fn apply_keyboard_shortcuts(
    ctx: &egui::Context,
    circuit_viewer: &BloqViewerState,
    intents: &mut UiIntentBuffer,
) {
    if ctx.egui_wants_keyboard_input() {
        return;
    }

    let (prev_pressed, next_pressed, detslice_pressed, shift) = ctx.input(|input| {
        (
            input.key_pressed(egui::Key::Q),
            input.key_pressed(egui::Key::E),
            input.key_pressed(egui::Key::D),
            input.modifiers.shift,
        )
    });

    if detslice_pressed {
        // Follows the same state machine as the toggle button: enable from off
        // (computing first if needed), disable from on, ignore while pending or
        // unavailable.
        match circuit_viewer.detslice_toggle() {
            LazyToggle::Off | LazyToggle::On => {
                let mut visibility = circuit_viewer.slice_visibility;
                visibility.detectors = !visibility.detectors;
                intents.push(UiIntent::SetSliceVisibility(visibility));
            }
            LazyToggle::Absent | LazyToggle::Pending | LazyToggle::Unavailable(_) => {}
        }
    }

    let mut move_to = |target| {
        if let Some((layer, moment)) = target {
            if layer != circuit_viewer.selected_layer {
                intents.push(UiIntent::SetConcurrentLayer(layer));
            }
            intents.push(UiIntent::SetCircuitMoment(moment));
        }
    };

    if prev_pressed {
        move_to(if shift {
            circuit_viewer
                .previous_reset_moment_index()
                .map(|moment| (circuit_viewer.selected_layer, moment))
        } else {
            circuit_viewer.previous_timeline_position()
        });
    }

    if next_pressed {
        move_to(if shift {
            circuit_viewer
                .next_reset_moment_index()
                .map(|moment| (circuit_viewer.selected_layer, moment))
        } else {
            circuit_viewer.next_timeline_position()
        });
    }
}

fn draw_qubit_hint_window(
    ctx: &egui::Context,
    pointer: Pos2,
    qubit: usize,
    coord: IVec2,
    palette: &ThemePalette,
) {
    egui::Area::new(egui::Id::new("circuit_qubit_hint"))
        .order(egui::Order::Foreground)
        .interactable(false)
        .fixed_pos(pointer + egui::vec2(12.0, 12.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                ui.label(
                    egui::RichText::new(format!("q{qubit}: ({}, {})", coord.x, coord.y))
                        .small()
                        .color(palette.text_bright),
                );
            });
        });
}

/// The floating tooltip for a hovered detector or logical-observable region.
fn draw_region_hint_window(
    ctx: &egui::Context,
    pointer: Pos2,
    region: &SliceRegionView,
    selected_node: Option<u32>,
    palette: &ThemePalette,
) {
    egui::Area::new(egui::Id::new("circuit_region_hint"))
        .order(egui::Order::Foreground)
        .interactable(false)
        .fixed_pos(pointer + egui::vec2(12.0, 12.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                let (title, cross_node) = match region.id {
                    SliceRegionId::Detector {
                        owner_node,
                        detector,
                    } => (
                        format!("N{owner_node} · D{detector}"),
                        selected_node.is_some_and(|node| node != owner_node),
                    ),
                    SliceRegionId::Observable { index } => {
                        (format!("L{index} · observable"), false)
                    }
                };
                ui.label(
                    egui::RichText::new(title)
                        .small()
                        .strong()
                        .color(palette.text_bright),
                );
                if let Some(coords) = &region.coords {
                    let rendered = coords
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(
                        egui::RichText::new(format!("({rendered})"))
                            .small()
                            .color(palette.text_dim),
                    );
                }
                ui.label(
                    egui::RichText::new(format!(
                        "{} · weight {}",
                        pauli_composition(region),
                        region.terms.len()
                    ))
                    .small()
                    .color(palette.text_primary),
                );
                if cross_node {
                    ui.label(
                        egui::RichText::new("cross-node")
                            .small()
                            .strong()
                            .color(palette.accent_warn),
                    );
                }
            });
        });
}

/// A region's Pauli make-up as a compact string (`Z4`, `X2·Z2`), skipping any
/// Pauli it has no support in.
fn pauli_composition(region: &SliceRegionView) -> String {
    let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);
    for term in &region.terms {
        match term.pauli {
            PauliBasis::X => x += 1,
            PauliBasis::Y => y += 1,
            PauliBasis::Z => z += 1,
        }
    }
    [("X", x), ("Y", y), ("Z", z)]
        .into_iter()
        .filter(|&(_, count)| count > 0)
        .map(|(label, count)| format!("{label}{count}"))
        .collect::<Vec<_>>()
        .join("·")
}

fn scrubber_color(moment: &crate::resources::FlatMoment, palette: &ThemePalette) -> Color32 {
    let colors = circuit_colors(palette);
    match moment.kind {
        MomentKind::Interaction => palette.accent_primary,
        MomentKind::Reset | MomentKind::Measurement => colors.reset_gate_fill,
        MomentKind::Rotation => colors.clifford_gate_fill,
    }
}

#[derive(Clone)]
struct GateStyle {
    fill: Color32,
    text: Color32,
    label: String,
}

fn gate_style(gate: GateType, palette: &ThemePalette) -> GateStyle {
    let colors = circuit_colors(palette);
    if gate.is_reset() {
        return GateStyle {
            fill: colors.reset_gate_fill,
            text: colors.reset_gate_text,
            label: gate_label(gate),
        };
    }
    if matches!(
        gate,
        GateType::H
            | GateType::H_XY
            | GateType::H_YZ
            | GateType::H_NXY
            | GateType::H_NXZ
            | GateType::H_NYZ
            | GateType::SQRT_X
            | GateType::SQRT_X_DAG
            | GateType::SQRT_Y
            | GateType::SQRT_Y_DAG
            | GateType::S
            | GateType::S_DAG
    ) {
        return GateStyle {
            fill: colors.clifford_gate_fill,
            text: colors.clifford_gate_text,
            label: gate_label(gate),
        };
    }
    if gate.is_non_clifford() {
        return GateStyle {
            fill: colors.non_clifford_gate_fill,
            text: colors.non_clifford_gate_text,
            label: gate_label(gate),
        };
    }
    GateStyle {
        fill: colors.default_gate_fill,
        text: colors.default_gate_text,
        label: gate_label(gate),
    }
}

fn measurement_style(basis: PauliBasis) -> GateStyle {
    let label = match basis {
        PauliBasis::X => "MX",
        PauliBasis::Y => "MY",
        PauliBasis::Z => "M",
    };
    GateStyle {
        fill: Color32::WHITE,
        text: Color32::from_rgb(24, 32, 44),
        label: label.to_string(),
    }
}

/// Math-style gate label for the canvas: a `_DAG` suffix renders as a dagger
/// (`T_DAG` → `T†`) and a `SQRT_` prefix as a radical (`SQRT_X` → `√X`), so the
/// non-Clifford `T†` in a cultivation circuit reads like its textbook symbol.
fn gate_label(gate: GateType) -> String {
    let raw = gate.to_string();
    let (base, dagger) = match raw.strip_suffix("_DAG") {
        Some(base) => (base, "†"),
        None => (raw.as_str(), ""),
    };
    let base = match base.strip_prefix("SQRT_") {
        Some(rest) => format!("√{rest}"),
        None => base.to_string(),
    };
    format!("{base}{dagger}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{CompileOutputFormat, CompileRequest};
    use crate::systems::jobs::compile_graph_for_viewer;
    use bloq_graph::GalleryItem;

    /// Compiles a lightweight gallery graph into a loaded viewer state for the
    /// export tests. CNOT's open form compiles straight through (no port fill).
    fn compiled_viewer() -> BloqViewerState {
        compiled_viewer_at(0)
    }

    fn compiled_viewer_at(z: i32) -> BloqViewerState {
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .unwrap()
            .shift_positions(IVec3::new(0, 0, z))
            .unwrap();
        let view = compile_graph_for_viewer(
            &graph,
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("CNOT compiles for the viewer");
        let mut viewer = BloqViewerState::default();
        viewer.finish_compile(1, view);
        viewer
    }

    #[test]
    fn and_classical_toggle_places_all_visible_nodes() {
        for item in [GalleryItem::CCZInjectedAnd, GalleryItem::And4T] {
            // Gallery loading retains the module's classical interface; compiling
            // only its materialized graph misses the parallel dataflow edges.
            let artifacts = bloq_compile::CompileContext::new(CompileConfig::default())
                .compile(&item.build())
                .expect("AND compiles for the viewer");
            let view = crate::program_view::build_bloq_circuit_view(
                &artifacts.bloq,
                &item.build().flatten().unwrap(),
                IVec3::ZERO,
            )
            .unwrap();
            let mut viewer = BloqViewerState {
                nodes: view.nodes,
                edges: view.edges,
                ..Default::default()
            };
            for show_classical in [false, true, false] {
                viewer.show_classical = show_classical;
                ensure_graph_layout(&mut viewer);
                let positions = &viewer.graph_layout_cache.positions;
                assert_eq!(
                    positions.len(),
                    viewer
                        .nodes
                        .iter()
                        .filter(|node| node_visible(node, &viewer))
                        .count(),
                    "{item}, classical={show_classical}"
                );
                assert!(
                    positions
                        .values()
                        .all(|(x, y)| { x.is_finite() && y.is_finite() && *x >= 0.0 && *y >= 0.0 })
                );
                let routes = &viewer.graph_layout_cache.routes;
                for (index, edge) in viewer.edges.iter().enumerate() {
                    let visible =
                        positions.contains_key(&edge.from) && positions.contains_key(&edge.to);
                    assert_eq!(routes.contains_key(&index), visible);
                    if let Some(route) = routes.get(&index) {
                        assert!(!route.is_empty());
                        assert!(
                            route
                                .iter()
                                .flatten()
                                .all(|(x, y)| x.is_finite() && y.is_finite())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fit_program_diagram_recovers_all_nodes_after_pan_and_zoom() {
        let mut viewer = compiled_viewer();
        for size in [egui::vec2(320.0, 220.0), egui::vec2(900.0, 600.0)] {
            viewer.graph_pan_offset = Vec2::new(900.0, -500.0);
            viewer.graph_zoom = 2.5;
            fit_program_graph(&mut viewer, size);
            let canvas = Rect::from_min_size(egui::pos2(10.0, 80.0), size);
            let layout = program_node_layout(canvas, &mut viewer);
            assert!(!layout.is_empty());
            assert!(layout.values().all(|rect| canvas.contains_rect(*rect)));
        }
    }

    #[test]
    fn export_program_svg_renders_node_labels() {
        let mut viewer = compiled_viewer();
        let node_id = viewer.nodes.first().expect("compiled program has nodes").id;
        let palette = theme::palette(ThemePreset::default());

        let (file_name, svg) =
            export_program_svg(&mut viewer, palette).expect("program view exports");

        assert_eq!(file_name, "bloq-program-r1.svg");
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert!(svg.contains("<rect"));
        // Every tile label starts with `N{id}`, always on the first galley row.
        assert!(
            svg.contains(&format!("N{node_id}")),
            "exported program should render a node label: {svg}"
        );
        // The export uses cached routes and nodes at zoom 1, independent of the
        // current viewport, and restores live state afterwards.
        viewer.graph_zoom = 2.5;
        viewer.graph_pan_offset = Vec2::new(900.0, -500.0);
        let (_, moved_svg) = export_program_svg(&mut viewer, palette).unwrap();
        assert_eq!(svg, moved_svg);
        assert_eq!(viewer.graph_zoom, 2.5);
        assert_eq!(viewer.graph_pan_offset, Vec2::new(900.0, -500.0));
    }

    #[test]
    fn export_circuit_svg_renders_measurement_glyphs() {
        let mut viewer = compiled_viewer();
        let (node_id, moment) = viewer
            .nodes
            .iter()
            .find_map(|node| {
                node.moments.iter().position(|moment| {
                    moment.ops.iter().any(|op| matches!(op,
                        FlatMomentOp::Mpp { products } if products.iter().flatten().any(|(_, basis)| *basis == PauliBasis::X)
                    ))
                }).map(|moment| (node.id, moment))
            })
            .expect("CNOT has an X stabilizer MPP boundary");
        viewer.set_selected_node(Some(node_id));
        viewer.set_current_moment(moment);
        let palette = theme::palette(ThemePreset::default());

        let (file_name, svg) = export_circuit_svg(&viewer, palette, ThemePreset::default())
            .expect("circuit view exports");

        assert!(file_name.starts_with(&format!("bloq-node-N{node_id}-m")));
        assert!(file_name.ends_with(".svg"));
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert!(
            svg.contains(">MX</text>"),
            "the X measurement glyph is exported"
        );
    }

    #[test]
    fn shifted_compile_selects_original_source_layer() {
        let mut viewer = compiled_viewer_at(5);
        assert_eq!(viewer.source_offset.z, 5);
        let node_id = viewer
            .nodes
            .iter()
            .filter(|node| node.category.is_quantum())
            .min_by_key(|node| node.layer)
            .unwrap()
            .id;
        viewer.set_selected_node(Some(node_id));
        assert_eq!(viewer.selected_layer, 5);
    }

    #[test]
    fn concurrent_view_selects_and_steps_source_layers() {
        let mut viewer = compiled_viewer();
        let node = viewer
            .nodes
            .iter()
            .find(|node| node.category.is_quantum() && !node.moments.is_empty())
            .expect("CNOT has a quantum circuit")
            .clone();
        let node_id = node.id;
        let layer = viewer.node_source_layer(&node).expect("node layer fits");
        let source_moments = (0..node.moments.len())
            .map(|moment| vec![(node_id, moment)])
            .collect();
        let circuit = LayerCircuitView {
            moments: node.moments,
            source_moments,
            qubit_coords: node.qubit_coords,
            num_qubits: node.num_qubits,
        };

        let mut empty = circuit.clone();
        empty.moments.clear();
        empty.source_moments.clear();
        for node in &mut viewer.nodes {
            node.layer += 14;
        }
        viewer.source_offset.z += 7;
        viewer.layer_circuits = HashMap::from([
            (layer, circuit.clone()),
            (layer + 3, empty),
            (layer + 7, circuit.clone()),
        ]);
        viewer.set_selected_node(Some(node_id));
        viewer.set_concurrent_ops(true);

        assert_eq!(viewer.selected_layer, layer + 7);
        assert!(!viewer.moments.is_empty());
        assert_eq!(
            viewer.previous_timeline_position(),
            Some((layer, circuit.moments.len() - 1))
        );
        viewer.set_concurrent_layer(layer);
        viewer.set_current_moment(usize::MAX);
        assert_eq!(viewer.next_timeline_position(), Some((layer + 7, 0)));

        viewer.layer_circuits = HashMap::from([(layer + 10, circuit)]);
        viewer.set_selected_node(Some(node_id));
        assert_eq!(viewer.selected_layer, layer + 10);
    }

    #[test]
    fn concurrent_view_keeps_and_composes_slices() {
        let moment = FlatMoment {
            kind: MomentKind::Rotation,
            ops: Vec::new(),
            repeat_label: None,
            repeat_span: None,
        };
        let slice = |qubit| {
            vec![vec![SliceRegionView {
                id: SliceRegionId::Observable { index: 4 },
                coords: None,
                terms: vec![RegionTerm {
                    qubit,
                    pauli: PauliBasis::X,
                }],
            }]]
        };
        let mut viewer = BloqViewerState {
            selected_node: Some(0),
            layer_circuits: HashMap::from([(
                0,
                LayerCircuitView {
                    moments: vec![moment.clone()],
                    source_moments: vec![vec![(0, 0), (1, 0)]],
                    qubit_coords: HashMap::from([(0, IVec2::ZERO)]),
                    num_qubits: 1,
                },
            )]),
            flat_moments: HashMap::from([(0, vec![moment])]),
            detector_slices: HashMap::from([(0, slice(IVec2::ZERO)), (1, slice(IVec2::X))]),
            detector_breaks: HashMap::from([((0, 0), vec![IVec2::ZERO]), ((1, 0), vec![IVec2::X])]),
            ..Default::default()
        };

        viewer.set_slice_visibility(SliceVisibility {
            detectors: true,
            observables: true,
        });
        viewer.set_concurrent_ops(true);

        assert_eq!(viewer.detslice_toggle(), LazyToggle::On);
        assert_eq!(viewer.obsslice_toggle(), LazyToggle::On);
        let regions = viewer.current_slice_regions();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].terms.len(), 2);
        assert_eq!(&*viewer.current_slice_breaks(), &[IVec2::ZERO, IVec2::X]);
    }

    #[test]
    fn export_program_svg_none_when_empty() {
        let mut viewer = BloqViewerState::default();
        assert!(export_program_svg(&mut viewer, theme::palette(ThemePreset::default())).is_none());
    }

    #[test]
    fn hovered_scrubber_index_maps_slot_centers() {
        let track = Rect::from_min_size(egui::pos2(10.0, 0.0), egui::vec2(300.0, 20.0));
        let moment_count = 3;

        for index in 0..moment_count {
            let slot = scrubber_slot_rect(track, index, moment_count);
            assert_eq!(
                hovered_scrubber_index(track, slot.center(), moment_count),
                index
            );
        }
    }

    // A region (node 0) containing a two-node chain must grow past the default
    // tile size and hold both children strictly inside its padded body area.
    #[test]
    fn region_children_lay_out_inside_their_container() {
        let children_map = HashMap::from([(0u32, vec![1u32, 2u32])]);
        let edges = vec![(7, 1, 2)];
        let mut positions = HashMap::new();
        let mut sizes = HashMap::new();

        let mut routes = HashMap::new();
        let content = layout_group(
            &[0],
            &children_map,
            &edges,
            &mut positions,
            &mut sizes,
            &mut routes,
        );

        let (px, py) = positions[&0];
        let (pw, ph) = sizes[&0];
        assert!(
            pw > GRAPH_NODE_SIZE.x && ph > GRAPH_NODE_SIZE.y,
            "container grows to fit children: {pw}x{ph}"
        );
        for id in [1u32, 2] {
            let (x, y) = positions[&id];
            let (w, h) = sizes[&id];
            assert!(
                x >= px + REGION_PAD - 0.01 && y >= py + REGION_HEADER - 0.01,
                "child {id} starts inside the container body"
            );
            assert!(
                x + w <= px + pw + 0.01 && y + h <= py + ph + 0.01,
                "child {id} ends inside the container"
            );
        }
        assert!(content.0 >= pw && content.1 >= ph);
        let route = &routes[&7];
        assert!(!route.is_empty());
        let start = route.first().unwrap()[0];
        let end = route.last().unwrap()[3];
        assert!((start.0 - (positions[&1].0 + sizes[&1].0)).abs() < 0.01);
        assert!((end.0 - positions[&2].0).abs() < 0.01);
        assert!(route.iter().flatten().all(|(x, y)| {
            *x >= px && *y >= py + REGION_HEADER && *x <= px + pw && *y <= py + ph
        }));
    }

    #[test]
    fn slice_index_column_scales_with_canvas() {
        let small_rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(320.0, 160.0));
        let large_rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(640.0, 320.0));
        let small = slice_column_layout(small_rect);
        let large = slice_column_layout(large_rect);

        assert!(large.column_rect.width() > small.column_rect.width());
        assert!(large.row_height > small.row_height);
        assert_eq!(small.column_rect.right(), small_rect.right());
    }

    #[test]
    fn circuit_child_ids_survive_slice_column_insertion() {
        let ctx = egui::Context::default();
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(120.0, 80.0));
        let mut footer_widget_ids = Vec::new();
        let mut second_pass_scroll_ids = None;

        ctx.run_ui(Default::default(), |ui| {
            let column_scroll_id = (!footer_widget_ids.is_empty()).then(|| {
                let mut column = clipped_rect_child_ui(ui, "slice-index", rect);
                egui::ScrollArea::vertical()
                    .show(&mut column, |ui| ui.label("D0"))
                    .id
            });
            let mut footer = clipped_rect_child_ui(ui, "circuit-footer", rect);
            let footer_output =
                egui::ScrollArea::horizontal().show(&mut footer, |ui| ui.button("Next").id);

            footer_widget_ids.push(footer_output.inner);
            if let Some(column_scroll_id) = column_scroll_id {
                second_pass_scroll_ids = Some((column_scroll_id, footer_output.id));
            } else {
                ui.request_discard("insert slice column");
            }
        })
        .drop_without_applying_deltas();

        assert_eq!(footer_widget_ids.len(), 2);
        assert_eq!(footer_widget_ids[0], footer_widget_ids[1]);
        let (column_scroll_id, footer_scroll_id) = second_pass_scroll_ids.unwrap();
        assert_ne!(column_scroll_id, footer_scroll_id);
    }

    #[test]
    fn node_windows_remain_inside_a_small_panel_free_viewport() {
        for attributes in [true, false] {
            let ctx = egui::Context::default();
            let screen = Rect::from_min_size(Pos2::ZERO, egui::vec2(520.0, 480.0));
            let viewport = Rect::from_min_max(egui::pos2(100.0, 80.0), egui::pos2(520.0, 440.0));
            let mut viewer = compiled_viewer();
            let node = viewer
                .nodes
                .iter_mut()
                .find(|node| node.category.is_quantum() && !node.moments.is_empty())
                .expect("CNOT has a quantum circuit");
            node.attributes = (0..24)
                .map(|index| (format!("field {index}"), "long value ".repeat(40)))
                .collect();
            node.operator_table = (0..24)
                .map(|index| (format!("instance {index}"), "output".into(), "Z".repeat(40)))
                .collect();
            let node_id = node.id;
            viewer.set_selected_node(Some(node_id));
            let mut intents = UiIntentBuffer::default();

            for _ in 0..3 {
                ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(screen),
                        ..Default::default()
                    },
                    |ui| {
                        if attributes {
                            draw_node_attributes_window(
                                ui.ctx(),
                                viewport,
                                &viewer,
                                &mut intents,
                                theme::palette(ThemePreset::Light),
                            );
                        } else {
                            draw_node_circuit_window(
                                ui.ctx(),
                                viewport,
                                &EditorJobs::default(),
                                &mut viewer,
                                &mut intents,
                                theme::palette(ThemePreset::Light),
                                ThemePreset::Light,
                            );
                        }
                    },
                )
                .drop_without_applying_deltas();
            }
            let rect = ctx
                .memory(|memory| memory.area_rect(egui::Id::new(NODE_WINDOW_ID)))
                .unwrap();
            assert!(
                viewport.contains_rect(rect),
                "node window (attributes={attributes}) {rect:?} escaped {viewport:?}"
            );
        }
    }
}
