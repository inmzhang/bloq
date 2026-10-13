//! Bevy resources holding all editor state: the working graph and its undo
//! history, per-tab snapshots, the compile/validation and viewer UI state, and
//! the render caches that gate expensive per-frame mesh rebuilds.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use web_time::{Duration, Instant};

use crate::components::{CameraSettings, GraphElement};
use crate::theme::ThemePreset;
use bevy::prelude::*;
use bevy_egui::egui;
use bloq_graph::{
    BlockGraph, BlockKind, BranchArm, CubeKind, GalleryCategory, StabilizerGenerator, UDirection,
};
use color_eyre::eyre::Report;
use strum::Display;

const MAX_HISTORY_DEQUE_SIZE: usize = 20;

/// Viewport region left after the editor's egui panels claim their edges.
///
/// egui 0.35 stopped tracking a post-panel "available rect" on the context
/// (panels now compose only within a single parent `Ui`). `begin_ui_frame_system`
/// seeds this with the full content rect and each chained panel system shrinks
/// it in turn; overlays and the 3D viewport read the final rect to stay clear
/// of the panels.
#[derive(Resource, Clone, Copy)]
pub(crate) struct CentralViewport(pub(crate) egui::Rect);

/// Holds the current block graph with undo/redo history and revision tracking.
#[derive(Resource, Clone)]
pub(crate) struct GraphState {
    /// Editable local geometry, or a read-only flattened hierarchy preview.
    pub(crate) graph: BlockGraph,
    /// Authored hierarchy. The visible graph is its materialized preview;
    /// a leaf's in-progress geometry is reconciled by `resolved_graph`.
    pub(crate) source_graph: Option<Arc<BlockGraph>>,
    pub(crate) pending_branch_arm: Option<PendingBranchArm>,
    pub(crate) history: VecDeque<GraphHistoryEntry>,
    pub(crate) current_index: usize,
    pub(crate) needs_rerender: bool,
    pub(crate) revision: u64,
    pub(crate) edit_delta: GraphEditDelta,
}

/// Visible geometry and any hidden arm are one undoable authoring transaction.
#[derive(Clone)]
pub(crate) struct GraphHistoryEntry {
    pub(crate) graph: BlockGraph,
    source_graph: Option<Arc<BlockGraph>>,
    pending_branch_arm: Option<PendingBranchArm>,
}

/// Editor adapter for the shared definition-ownership view.
#[derive(Debug, Clone)]
pub(crate) struct ModuleViewState(bloq_graph::ModuleView);

impl ModuleViewState {
    pub(crate) fn from_graph(program: &BlockGraph, graph: &BlockGraph) -> Option<Self> {
        bloq_graph::ModuleView::from_graph(program, graph).map(Self)
    }

    pub(crate) fn modules(&self) -> &[bloq_graph::ModuleViewModule] {
        self.0.modules()
    }

    pub(crate) fn module_for_element(
        &self,
        graph: &BlockGraph,
        element: GraphElement,
    ) -> Option<usize> {
        match element.canonical() {
            GraphElement::Block(position) => self.0.module_for_block(position),
            GraphElement::Pipe(a, b) => self.0.module_for_pipe(graph, a, b),
        }
    }
}

/// What the most recent commit changed, when it is cheap enough to describe
/// precisely. Lets consumers skip a full rerender for localized edits;
/// `Unknown` forces the conservative full path.
#[derive(Debug, Clone, Default)]
pub(crate) enum GraphEditDelta {
    #[default]
    Unknown,
    PipeAdded {
        src: IVec3,
        dst: IVec3,
        hadamard: bool,
    },
    PipeRemoved {
        src: IVec3,
        dst: IVec3,
        hadamard: bool,
    },
}

/// Stable identifier for an editor tab, unique for the lifetime of the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) struct EditorTabId(u64);

impl EditorTabId {
    pub(crate) const fn new(id: u64) -> Self {
        Self(id)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// A tab's full editor state, stashed while another tab is active and restored
/// on switch back. The active tab's live copy is the editor resources
/// themselves, not this snapshot.
#[derive(Clone, Default)]
pub(crate) struct EditorTabSnapshot {
    pub(crate) graph_state: GraphState,
    pub(crate) editor_state: EditorState,
    pub(crate) import_export: ImportExportState,
    pub(crate) compile_ui: CompileUiState,
    pub(crate) circuit_viewer: BloqViewerState,
    pub(crate) zx_viewer: ZxViewerState,
    pub(crate) target_state: TargetState,
    pub(crate) box_selection: BoxSelectionState,
    pub(crate) camera_settings: CameraSettings,
}

impl EditorTabSnapshot {
    pub(crate) fn capture(live: &LiveTabState<'_>) -> Self {
        Self {
            graph_state: live.graph_state.clone(),
            editor_state: live.editor_state.clone(),
            import_export: live.import_export.clone(),
            compile_ui: live.compile_ui.clone(),
            circuit_viewer: live.circuit_viewer.clone(),
            zx_viewer: live.zx_viewer.without_cache(),
            target_state: live.target_state.clone(),
            box_selection: live.box_selection.clone(),
            camera_settings: *live.camera_settings,
        }
    }
}

/// Mutable state owned by the active tab, mirroring [`EditorTabSnapshot`].
///
/// Save, switch, close, and import transfer this bundle together to preserve
/// module hierarchy and drafts.
pub(crate) struct LiveTabState<'a> {
    pub(crate) graph_state: &'a mut GraphState,
    pub(crate) editor_state: &'a mut EditorState,
    pub(crate) import_export: &'a mut ImportExportState,
    pub(crate) compile_ui: &'a mut CompileUiState,
    pub(crate) circuit_viewer: &'a mut BloqViewerState,
    pub(crate) zx_viewer: &'a mut ZxViewerState,
    pub(crate) target_state: &'a mut TargetState,
    pub(crate) box_selection: &'a mut BoxSelectionState,
    pub(crate) camera_settings: &'a mut CameraSettings,
}

/// One open editor tab: its id, title, stashed state, and in-progress rename.
pub(crate) struct EditorTab {
    pub(crate) id: EditorTabId,
    pub(crate) title: String,
    pub(crate) snapshot: EditorTabSnapshot,
    pub(crate) renaming: bool,
    pub(crate) rename_buffer: String,
}

/// The open tabs and which one is active. Always holds at least one tab;
/// removing the last tab spawns a fresh empty one.
#[derive(Resource)]
pub(crate) struct EditorTabs {
    pub(crate) tabs: Vec<EditorTab>,
    pub(crate) active: EditorTabId,
    /// Launch-only welcome state, shared across tabs and never autosaved.
    pub(crate) show_welcome: bool,
    next_id: u64,
    next_untitled: u32,
}

impl EditorTabs {
    /// Install a browser session only after checking tab identities and counters.
    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn from_session(tabs: Vec<EditorTab>, active: EditorTabId) -> Option<Self> {
        let ids: HashSet<_> = tabs.iter().map(|tab| tab.id).collect();
        if ids.len() != tabs.len() || !ids.contains(&active) {
            return None;
        }
        let next_id = ids.iter().map(|id| id.get()).max()?.checked_add(1)?;
        Some(Self {
            next_untitled: u32::try_from(next_id).ok()?,
            next_id,
            tabs,
            active,
            show_welcome: false,
        })
    }

    /// Returns the active tab.
    ///
    /// # Panics
    ///
    /// Panics if `active` names no existing tab, which the invariant that at
    /// least one tab always exists and `active` always points at it prevents.
    pub(crate) fn active_tab(&self) -> &EditorTab {
        self.tabs
            .iter()
            .find(|tab| tab.id == self.active)
            .expect("active editor tab exists")
    }

    pub(crate) fn active_tab_mut(&mut self) -> &mut EditorTab {
        self.tabs
            .iter_mut()
            .find(|tab| tab.id == self.active)
            .expect("active editor tab exists")
    }

    pub(crate) fn title(&self, id: EditorTabId) -> Option<&str> {
        self.tabs
            .iter()
            .find(|tab| tab.id == id)
            .map(|tab| tab.title.as_str())
    }

    pub(crate) fn save_active(&mut self, snapshot: EditorTabSnapshot) {
        self.active_tab_mut().snapshot = snapshot;
    }

    pub(crate) fn restore_active(&self, live: &mut LiveTabState<'_>) {
        restore_tab_snapshot(&self.active_tab().snapshot, live);
    }

    pub(crate) fn add_empty_tab(&mut self, template_editor: &EditorState) -> EditorTabId {
        self.show_welcome = false;
        let id = self.allocate_id();
        let title = self.next_untitled_title();
        self.tabs
            .push(EditorTab::new_empty(id, title, template_editor));
        id
    }

    /// Closes tab `id` and returns the id that ends up active. Closing the last
    /// tab spawns a fresh empty one; closing the active tab moves focus to the
    /// neighbour on its left.
    pub(crate) fn remove(&mut self, id: EditorTabId, template_editor: &EditorState) -> EditorTabId {
        if let Some(index) = self.tabs.iter().position(|tab| tab.id == id) {
            self.tabs.remove(index);
            if self.tabs.is_empty() {
                let id = self.add_empty_tab(template_editor);
                self.active = id;
                return id;
            }
            if self.active == id {
                let next_index = index.saturating_sub(1).min(self.tabs.len() - 1);
                self.active = self.tabs[next_index].id;
            }
        }
        self.active
    }

    /// Makes tab `id` active, returning `false` (and doing nothing) if no such
    /// tab exists.
    pub(crate) fn set_active(&mut self, id: EditorTabId) -> bool {
        if self.tabs.iter().any(|tab| tab.id == id) {
            self.active = id;
            true
        } else {
            false
        }
    }

    pub(crate) fn begin_rename(&mut self, id: EditorTabId) {
        if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.id == id) {
            tab.renaming = true;
            tab.rename_buffer.clone_from(&tab.title);
        }
    }

    pub(crate) fn finish_rename(&mut self, id: EditorTabId) {
        if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.id == id) {
            let title = normalized_tab_title(std::mem::take(&mut tab.rename_buffer));
            tab.set_title(title);
            tab.renaming = false;
        }
    }

    pub(crate) fn cancel_rename(&mut self, id: EditorTabId) {
        if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.id == id) {
            tab.rename_buffer.clone_from(&tab.title);
            tab.renaming = false;
        }
    }

    fn allocate_id(&mut self) -> EditorTabId {
        let id = EditorTabId(self.next_id);
        self.next_id += 1;
        id
    }

    fn next_untitled_title(&mut self) -> String {
        let title = format!("Untitled {}", self.next_untitled);
        self.next_untitled += 1;
        title
    }
}

impl EditorTab {
    pub(crate) fn set_title(&mut self, title: String) {
        self.title = title;
        self.rename_buffer.clone_from(&self.title);
    }

    fn new_empty(id: EditorTabId, title: String, template_editor: &EditorState) -> Self {
        let editor_state = EditorState {
            highlight_material: template_editor.highlight_material.clone(),
            selection_material: template_editor.selection_material.clone(),
            preview_endpoint_material: template_editor.preview_endpoint_material.clone(),
            preview_endpoint_hover_material: template_editor
                .preview_endpoint_hover_material
                .clone(),
            action_candidate_material: template_editor.action_candidate_material.clone(),
            theme_preset: template_editor.theme_preset,
            bg_color: template_editor.bg_color,
            ..Default::default()
        };

        Self {
            id,
            title: title.clone(),
            snapshot: EditorTabSnapshot {
                editor_state,
                ..Default::default()
            },
            renaming: false,
            rename_buffer: title,
        }
    }
}

impl Default for EditorTabs {
    fn default() -> Self {
        let active = EditorTabId(1);
        let template_editor = EditorState::default();
        Self {
            tabs: vec![EditorTab::new_empty(
                active,
                "Untitled 1".to_string(),
                &template_editor,
            )],
            active,
            show_welcome: true,
            next_id: 2,
            next_untitled: 2,
        }
    }
}

/// Restores a tab's stashed state into the live editor resources, preserving
/// the shared material handles (they are session assets, not per-tab state) and
/// forcing a rerender.
fn restore_tab_snapshot(snapshot: &EditorTabSnapshot, live: &mut LiveTabState<'_>) {
    let highlight_material = live.editor_state.highlight_material.clone();
    let selection_material = live.editor_state.selection_material.clone();
    let preview_endpoint_material = live.editor_state.preview_endpoint_material.clone();
    let preview_endpoint_hover_material = live.editor_state.preview_endpoint_hover_material.clone();
    let action_candidate_material = live.editor_state.action_candidate_material.clone();
    *live.graph_state = snapshot.graph_state.clone();
    *live.editor_state = snapshot.editor_state.clone();
    live.editor_state.highlight_material = highlight_material;
    live.editor_state.selection_material = selection_material;
    live.editor_state.preview_endpoint_material = preview_endpoint_material;
    live.editor_state.preview_endpoint_hover_material = preview_endpoint_hover_material;
    live.editor_state.action_candidate_material = action_candidate_material;
    *live.import_export = snapshot.import_export.clone();
    *live.compile_ui = snapshot.compile_ui.clone();
    *live.circuit_viewer = snapshot.circuit_viewer.clone();
    *live.zx_viewer = snapshot.zx_viewer.clone();
    *live.target_state = snapshot.target_state.clone();
    *live.box_selection = snapshot.box_selection.clone();
    *live.camera_settings = snapshot.camera_settings;
    live.graph_state.needs_rerender = true;
}
pub(crate) use crate::systems::ui::zx_viewer::ZxViewerState;

fn normalized_tab_title(title: impl Into<String>) -> String {
    let title = title.into();
    let trimmed = title.trim();
    if trimmed.is_empty() {
        "Untitled".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Which graph rotations the UI may offer, derived from the block kinds present.
///
/// Branch regions, T, measurement, and selective blocks pin orientation
/// (`Disabled`); Y blocks allow only half-turns; otherwise all quarter-turns
/// are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphRotationAvailability {
    Disabled,
    HalfTurnsOnly,
    QuarterTurns,
}

impl GraphRotationAvailability {
    const fn from_block_flags(has_orientation_pinned: bool, has_y: bool) -> Self {
        if has_orientation_pinned {
            Self::Disabled
        } else if has_y {
            Self::HalfTurnsOnly
        } else {
            Self::QuarterTurns
        }
    }
}

/// Cheap graph facts the UI reads every frame, cached by active tab and graph
/// revision.
#[derive(Resource, Debug, Clone, Copy)]
pub(crate) struct GraphUiSummary {
    tab_id: Option<EditorTabId>,
    pub(crate) revision: u64,
    pub(crate) block_count: usize,
    pub(crate) pipe_count: usize,
    /// Whether the graph has at least one `Port` block (an open boundary).
    pub(crate) is_open: bool,
    pub(crate) is_empty: bool,
    pub(crate) rotation_availability: GraphRotationAvailability,
}

impl GraphUiSummary {
    #[cfg(test)]
    pub(crate) fn from_graph_state(graph_state: &GraphState) -> Self {
        let mut summary = Self::default();
        summary.sync_from_graph_state(EditorTabId::new(1), graph_state);
        summary
    }

    /// Recomputes the summary unless the tab and graph revision are unchanged.
    pub(crate) fn sync_from_graph_state(&mut self, tab_id: EditorTabId, graph_state: &GraphState) {
        if self.tab_id == Some(tab_id) && self.revision == graph_state.revision {
            return;
        }

        let mut block_count = 0;
        let mut has_port = false;
        let mut has_orientation_pinned = !graph_state.graph.branch_definitions().is_empty();
        let mut has_y = false;

        for block in graph_state.graph.blocks() {
            block_count += 1;
            match block.kind() {
                BlockKind::Port => has_port = true,
                BlockKind::T | BlockKind::Measurement(_) | BlockKind::Selective(_) => {
                    has_orientation_pinned = true;
                }
                BlockKind::Y => has_y = true,
                _ => {}
            }
        }

        self.tab_id = Some(tab_id);
        self.revision = graph_state.revision;
        self.block_count = block_count;
        self.pipe_count = graph_state.graph.pipe_count();
        self.is_open = has_port;
        self.is_empty = block_count == 0;
        self.rotation_availability =
            GraphRotationAvailability::from_block_flags(has_orientation_pinned, has_y);
    }
}

impl Default for GraphUiSummary {
    fn default() -> Self {
        Self {
            tab_id: None,
            revision: u64::MAX,
            block_count: 0,
            pipe_count: 0,
            is_open: false,
            is_empty: true,
            rotation_availability: GraphRotationAvailability::QuarterTurns,
        }
    }
}

impl Default for GraphState {
    fn default() -> Self {
        let graph = BlockGraph::default();
        let mut history = VecDeque::new();
        history.push_back(GraphHistoryEntry {
            graph: graph.clone(),
            source_graph: None,
            pending_branch_arm: None,
        });
        Self {
            graph,
            source_graph: None,
            pending_branch_arm: None,
            history,
            current_index: 0,
            needs_rerender: false,
            revision: 0,
            edit_delta: GraphEditDelta::Unknown,
        }
    }
}

impl GraphState {
    pub(crate) fn is_composed(&self) -> bool {
        self.source_graph
            .as_ref()
            .is_some_and(|program| !program.root().instances.is_empty())
    }

    pub(crate) fn resolved_graph(&self) -> color_eyre::eyre::Result<BlockGraph> {
        color_eyre::eyre::ensure!(
            self.pending_branch_arm.is_none(),
            "Finish or cancel the captured branch arm first"
        );
        match &self.source_graph {
            Some(program) if self.is_composed() => Ok((**program).clone()),
            Some(program) => crate::module_authoring::replace_leaf_body(program, &self.graph),
            None => {
                // Copy/save resolves the document without running stabilizer analysis.
                let ast = bloq_graph::parse_blog_program_to_ast(&self.graph.to_blog_text())?;
                Ok(bloq_graph::lower_blog_graph_ast_deferred(&ast)?)
            }
        }
    }

    pub(crate) fn selection_touches_pending_branch_cut(
        &self,
        selected: &HashSet<GraphElement>,
    ) -> bool {
        self.pending_branch_arm.as_ref().is_some_and(|pending| {
            pending
                .arm
                .pipes()
                .flat_map(|pipe| {
                    let (a, b) = pipe.endpoints();
                    [a, b]
                })
                .any(|endpoint| {
                    !pending.arm.contains_block(endpoint)
                        && self
                            .graph
                            .get_endpoint_block(endpoint)
                            .is_some_and(|block| {
                                selected.contains(&GraphElement::Block(block.pos()))
                            })
                })
        })
    }

    /// Pushes the current graph onto the undo history and marks it for rerender.
    pub(crate) fn commit(&mut self) {
        self.commit_with_rerender(true);
    }

    /// Pushes the current graph onto the undo history, truncating any redo tail
    /// and bumping the revision.
    ///
    /// Pass `needs_rerender = false` for edits with no visual effect (e.g. a
    /// pipe tag change) to avoid a redundant mesh rebuild.
    pub(crate) fn commit_with_rerender(&mut self, needs_rerender: bool) {
        if self.current_index + 1 < self.history.len() {
            self.history.truncate(self.current_index + 1);
        }
        self.history.push_back(GraphHistoryEntry {
            graph: self.graph.clone(),
            source_graph: self.source_graph.clone(),
            pending_branch_arm: self.pending_branch_arm.clone(),
        });
        if self.history.len() > MAX_HISTORY_DEQUE_SIZE {
            self.history.pop_front();
        }
        self.current_index = self.history.len() - 1;
        self.mark_changed(needs_rerender);
    }

    /// Commits like [`Self::commit_with_rerender`] but records a
    /// precise [`GraphEditDelta`] so consumers can take a localized update path.
    pub(crate) fn commit_with_delta(&mut self, needs_rerender: bool, edit_delta: GraphEditDelta) {
        self.commit_with_rerender(needs_rerender);
        self.edit_delta = edit_delta;
    }

    pub(crate) fn undo(&mut self) {
        if self.current_index > 0 {
            self.current_index -= 1;
            self.restore_current();
        }
    }

    pub(crate) fn redo(&mut self) {
        if self.current_index + 1 < self.history.len() {
            self.current_index += 1;
            self.restore_current();
        }
    }

    /// Replaces the working document with the entry at `current_index`. Undo and
    /// redo share it so both always restore the whole transaction, hierarchy and
    /// captured arm included.
    fn restore_current(&mut self) {
        let snapshot = &self.history[self.current_index];
        self.graph = snapshot.graph.clone();
        self.source_graph = snapshot.source_graph.clone();
        self.pending_branch_arm = snapshot.pending_branch_arm.clone();
        self.mark_changed(true);
    }

    fn mark_changed(&mut self, needs_rerender: bool) {
        self.needs_rerender |= needs_rerender;
        self.revision = self.revision.saturating_add(1);
        self.edit_delta = GraphEditDelta::Unknown;
    }
}

/// Whether egui currently owns pointer/keyboard input, so viewport gestures
/// know to stand down.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub(crate) struct UiInputState {
    pub(crate) pointer_over_ui: bool,
    pub(crate) wants_pointer_input: bool,
    pub(crate) wants_keyboard_input: bool,
}

impl UiInputState {
    /// Whether viewport pointer gestures (orbit, pan, pick) should be suppressed.
    pub(crate) fn blocks_viewport_pointer_input(self) -> bool {
        self.pointer_over_ui || self.wants_pointer_input
    }

    /// Whether viewport keyboard shortcuts should be suppressed.
    pub(crate) fn blocks_viewport_keyboard_input(self) -> bool {
        self.wants_keyboard_input
    }
}

/// The editor's top-level interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, serde::Serialize, serde::Deserialize)]
pub(crate) enum EditorMode {
    /// Inspect, select, and orbit; no placement.
    View,
    /// Inspect the materialized graph colored by reusable module definition.
    Module,
    /// Place blocks/pipes and edit existing ones (double-click to open the
    /// attribute editor).
    Edit,
    /// Inspect the compiled Bloq program in the graph/circuit viewer.
    Bloq,
}

impl EditorMode {
    /// Whether this mode replaces the 3D viewport with a full-width panel, and
    /// therefore hides the panels and overlays that annotate that viewport.
    pub(crate) const fn covers_viewport(self) -> bool {
        matches!(self, Self::Bloq)
    }
}

/// Which primitive the Edit-mode pointer places on click.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Display, Default, serde::Serialize, serde::Deserialize,
)]
pub(crate) enum PlacementTool {
    #[default]
    Block,
    Pipe,
}

pub(crate) use crate::systems::thumbnails::{PlacementPreviewKind, ThumbnailTextures};

pub(crate) use crate::systems::jobs::{
    CompileExecutionState, CompileOutputFormat, CompileRequest, CompileUiState, ValidationState,
};

pub(crate) use crate::systems::ui::circuit_viewer::{
    BloqEdgeKind, BloqEdgeView, BloqNodeCategory, BloqNodeView, BloqViewerState, CompiledBloqView,
    ConcurrentOpsInputs, DetectorBreaks, DetsliceData, FlatMoment, FlatMomentOp, LayerCircuitView,
    NodeSliceTimeline, SliceRegionId, SliceRegionView, SliceVisibility,
};

/// The set of port-filling variants the "fill ports" action cycles through, and
/// the graph revision they were generated for.
#[derive(Debug, Clone)]
pub(crate) struct FillPortsCycleState {
    pub(crate) variants: Vec<BlockGraph>,
    pub(crate) current_index: usize,
    pub(crate) active_revision: u64,
}

pub(crate) use crate::systems::visuals::{
    GraphRenderState, RenderAssetCache, RenderSignatureBlockIdentity,
    RenderSignatureConnectableOffsets, RenderSignaturePipeIdentity, RenderedElement,
    RenderedMeshParts,
};

pub(crate) use crate::systems::ui::edit_element::TargetState;

pub(crate) use crate::systems::ui::action_editor::ActionEditState;
#[cfg(test)]
pub(crate) use crate::systems::ui::action_editor::ActionKind;

pub(crate) use crate::systems::ui::action_viewer::ActionViewerState;

pub(crate) use crate::systems::input::BoxSelectionState;

#[derive(Debug, Clone)]
pub(crate) struct PendingBranchArm {
    pub(crate) name: String,
    pub(crate) arm: BranchArm,
    pub(crate) captured_true: bool,
}

/// The editor's interaction and view state for the active tab: current tool and
/// mode, hover/selection sets, plane and visibility toggles, overlay materials,
/// and cached stabilizer results.
#[derive(Resource, Clone)]
pub(crate) struct EditorState {
    pub(crate) block_kind: BlockKind,
    pub(crate) placement_tool: PlacementTool,
    pub(crate) pipe_length: f32,
    pub(crate) mode: EditorMode,
    pub(crate) transform_axis: UDirection,

    pub(crate) plane_height: i32,
    pub(crate) show_axis: bool,
    pub(crate) show_grid: bool,
    pub(crate) show_port_tags: bool,
    pub(crate) bg_color: Color,

    pub(crate) hovered_grid_pos: Option<IVec3>,
    pub(crate) hovered_element: Option<GraphElement>,
    /// Last element click, retained because a Pipe-tool preview endpoint can
    /// replace a block as the picking target for the second click.
    pub(crate) last_element_click: Option<(GraphElement, f64)>,
    pub(crate) hovered_preview_source: Option<IVec3>,
    pub(crate) hovered_preview_endpoint: Option<IVec3>,
    pub(crate) blog_hovered_elements: HashSet<GraphElement>,
    pub(crate) zx_hovered_elements: HashSet<GraphElement>,
    /// Elements named by the action-list row under the pointer.
    pub(crate) action_hovered_elements: HashSet<GraphElement>,
    /// Elements an armed action pick would accept; pulsed until it resolves.
    /// Mirrored here (rather than read from `ActionEditState`) so the highlight
    /// sync keeps its single-resource read.
    pub(crate) action_candidate_elements: HashSet<GraphElement>,
    pub(crate) selected_elements: HashSet<GraphElement>,
    pub(crate) pipe_start: Option<IVec3>,
    /// Endpoints of the last successfully placed pipe, used by repeat placement.
    pub(crate) last_pipe_placement: Option<(IVec3, IVec3)>,
    pub(crate) walking_start: Option<IVec3>,
    pub(crate) branch_name: String,

    pub(crate) view_current_layer_only: bool,
    pub(crate) show_blog_buffer: bool,
    pub(crate) show_help_window: bool,
    pub(crate) theme_preset: ThemePreset,
    pub(crate) gallery_panel_expanded: bool,
    pub(crate) show_side_panel: bool,
    pub(crate) selected_gallery_category: Option<GalleryCategory>,
    pub(crate) gallery_search: String,
    pub(crate) fill_ports_cycle: Option<FillPortsCycleState>,
    pub(crate) module_view: Option<ModuleViewState>,

    pub(crate) highlight_material: Handle<StandardMaterial>,
    pub(crate) selection_material: Handle<StandardMaterial>,
    pub(crate) preview_endpoint_material: Handle<StandardMaterial>,
    pub(crate) preview_endpoint_hover_material: Handle<StandardMaterial>,
    /// Pulsing overlay for the elements an armed action pick would accept.
    pub(crate) action_candidate_material: Handle<StandardMaterial>,
    pub(crate) request_screenshot: bool,
    pub(crate) taking_screenshot: bool,

    pub(crate) stabilizers: Vec<StabilizerGenerator>,
    pub(crate) current_stabilizer_index: usize,
    pub(crate) show_stabilizers: bool,
    /// Measurement surface selected in the Actions window, with action ordinal.
    pub(crate) action_stabilizer: Option<(usize, StabilizerGenerator)>,
}

impl EditorState {
    /// Switches the placement tool, discarding any in-progress pipe/walking
    /// start when the tool actually changes.
    pub(crate) fn set_placement_tool(&mut self, tool: PlacementTool) {
        if self.placement_tool != tool {
            self.pipe_start = None;
            self.walking_start = None;
            self.last_element_click = None;
        }
        self.placement_tool = tool;
    }

    pub(crate) fn is_pipe_tool_active(&self) -> bool {
        self.mode == EditorMode::Edit && self.placement_tool == PlacementTool::Pipe
    }

    pub(crate) fn is_block_tool_active(&self) -> bool {
        self.mode == EditorMode::Edit && self.placement_tool == PlacementTool::Block
    }

    pub(crate) fn select_block_kind(&mut self, kind: BlockKind) {
        self.block_kind = kind;
        self.set_placement_tool(PlacementTool::Block);
    }

    /// Selects `element`. With `toggle`, adds it to (or removes it from) the
    /// current selection; otherwise it replaces the selection.
    pub(crate) fn select_element(&mut self, element: GraphElement, toggle: bool) {
        let element = element.canonical();
        if toggle {
            if !self.selected_elements.insert(element) {
                self.selected_elements.remove(&element);
            }
        } else {
            self.selected_elements.clear();
            self.selected_elements.insert(element);
        }
    }

    /// Applies click semantics: a plain click on an unselected element replaces
    /// the selection, while `additive` clicks (or clicking an already-selected
    /// element) toggle it.
    pub(crate) fn click_select_element(&mut self, element: GraphElement, additive: bool) {
        let element = element.canonical();
        if additive || self.is_selected(element) {
            self.select_element(element, true);
        } else {
            self.select_element(element, false);
        }
    }

    /// Selects every block and pipe in `graph`.
    pub(crate) fn select_all_elements(&mut self, graph: &BlockGraph) {
        self.selected_elements = GraphElement::all_in(graph).collect();
    }

    /// Selects each of `elements`; with `additive = false` the current
    /// selection is cleared first.
    pub(crate) fn select_elements(
        &mut self,
        elements: impl IntoIterator<Item = GraphElement>,
        additive: bool,
    ) {
        if !additive {
            self.selected_elements.clear();
        }
        self.selected_elements
            .extend(elements.into_iter().map(GraphElement::canonical));
    }

    /// Replaces the whole selection with `elements`.
    pub(crate) fn replace_selection(&mut self, elements: impl IntoIterator<Item = GraphElement>) {
        self.select_elements(elements, false);
    }

    pub(crate) fn clear_selection(&mut self) {
        self.selected_elements.clear();
    }

    pub(crate) fn selected_elements(&self) -> impl Iterator<Item = GraphElement> + '_ {
        self.selected_elements.iter().copied()
    }

    pub(crate) fn selected_element_set(&self) -> &HashSet<GraphElement> {
        &self.selected_elements
    }

    pub(crate) fn blog_hovered_element_set(&self) -> &HashSet<GraphElement> {
        &self.blog_hovered_elements
    }

    pub(crate) fn zx_hovered_element_set(&self) -> &HashSet<GraphElement> {
        &self.zx_hovered_elements
    }

    pub(crate) fn set_zx_hovered_elements(&mut self, elements: HashSet<GraphElement>) {
        self.zx_hovered_elements = elements;
    }

    pub(crate) fn selection_count(&self) -> usize {
        self.selected_elements.len()
    }

    pub(crate) fn is_selected(&self, element: GraphElement) -> bool {
        self.selected_elements.contains(&element.canonical())
    }

    pub(crate) fn hovered_element_for_highlight(&self) -> Option<GraphElement> {
        if self.mode == EditorMode::Module {
            None
        } else {
            self.hovered_element.map(GraphElement::canonical)
        }
    }

    /// Clears all hover, cross-view highlight, and in-progress placement state.
    pub(crate) fn clear_hover(&mut self) {
        self.hovered_grid_pos = None;
        self.hovered_element = None;
        self.last_element_click = None;
        self.hovered_preview_source = None;
        self.hovered_preview_endpoint = None;
        self.blog_hovered_elements.clear();
        self.zx_hovered_elements.clear();
        self.pipe_start = None;
        self.walking_start = None;
    }

    /// The fill-ports cycle if one is active and was built for the current graph
    /// revision (a stale cycle is ignored).
    pub(crate) fn active_fill_ports_cycle(
        &self,
        graph_revision: u64,
    ) -> Option<&FillPortsCycleState> {
        self.fill_ports_cycle
            .as_ref()
            .filter(|cycle| cycle.active_revision == graph_revision)
    }

    pub(crate) fn clear_fill_ports_cycle(&mut self) {
        self.fill_ports_cycle = None;
    }

    /// Switches interaction mode, clearing hover (and selection outside
    /// View/Edit) and requesting a rerender when the mode change affects what is
    /// drawn (layer filtering or stabilizer overlay).
    pub(crate) fn set_mode(&mut self, mode: EditorMode, graph_state: &mut GraphState) {
        if mode == EditorMode::Edit && graph_state.is_composed() {
            return;
        }
        if self.mode == mode {
            return;
        }
        let previous_mode = self.mode;
        let previous_view_current_layer_only = self.view_current_layer_only;
        let previous_showing_stabilizers = self.showing_stabilizers();
        self.mode = mode;
        self.clear_hover();
        if mode == EditorMode::Module {
            self.clear_stabilizers();
        }
        if !matches!(mode, EditorMode::View | EditorMode::Edit) {
            self.clear_selection();
        }
        if self.view_current_layer_only
            && (mode == EditorMode::View || previous_mode == EditorMode::View)
        {
            self.clear_stabilizers();
        }
        if previous_view_current_layer_only
            || previous_showing_stabilizers
            || mode == EditorMode::Module
            || previous_mode == EditorMode::Module
        {
            graph_state.needs_rerender = true;
        }
    }

    /// Reconciles interaction state after a graph edit: drops cached
    /// stabilizers and fill-ports variants, and prunes selection, hover, and
    /// in-progress pipe/walking starts that the new graph no longer supports.
    pub(crate) fn sync_after_graph_edit(&mut self, graph_state: &mut GraphState) {
        if graph_state.is_composed() && self.mode == EditorMode::Edit {
            self.mode = EditorMode::Module;
        }
        graph_state.needs_rerender |= self.showing_stabilizers() || self.mode == EditorMode::Module;
        self.module_view = graph_state
            .source_graph
            .as_ref()
            .and_then(|program| ModuleViewState::from_graph(program, &graph_state.graph));
        self.clear_stabilizers();
        self.clear_fill_ports_cycle();
        self.selected_elements
            .retain(|element| graph_contains_element(&graph_state.graph, *element));
        if self
            .hovered_element
            .is_some_and(|element| !graph_contains_element(&graph_state.graph, element))
        {
            self.hovered_element = None;
        }
        self.blog_hovered_elements
            .retain(|element| graph_contains_element(&graph_state.graph, *element));
        if self
            .pipe_start
            .is_some_and(|pos| !graph_state.graph.has_endpoint_at(pos))
        {
            self.pipe_start = None;
        }
        if self
            .walking_start
            .is_some_and(|pos| graph_state.graph.has_endpoint_at(pos))
        {
            self.walking_start = None;
        }
    }

    /// Whether the stabilizer overlay is both enabled and has data to show.
    pub(crate) fn showing_stabilizers(&self) -> bool {
        self.action_stabilizer.is_some() || (self.show_stabilizers && !self.stabilizers.is_empty())
    }

    pub(crate) fn shown_stabilizer(&self) -> Option<&StabilizerGenerator> {
        self.action_stabilizer
            .as_ref()
            .map(|(_, row)| row)
            .or_else(|| {
                self.show_stabilizers
                    .then(|| self.stabilizers.get(self.current_stabilizer_index))
                    .flatten()
            })
    }

    pub(crate) fn set_action_stabilizer(
        &mut self,
        stabilizer: Option<(usize, StabilizerGenerator)>,
    ) -> bool {
        if self.action_stabilizer == stabilizer {
            return false;
        }
        self.action_stabilizer = stabilizer;
        true
    }

    /// Toggles the stabilizer overlay, returning `false` (a no-op) when no
    /// stabilizers have been computed.
    pub(crate) fn toggle_stabilizers(&mut self) -> bool {
        if self.stabilizers.is_empty() {
            return false;
        }
        self.show_stabilizers = !self.show_stabilizers;
        true
    }

    /// Steps to the previous stabilizer, wrapping to the last.
    pub(crate) fn prev_stabilizer(&mut self) {
        if !self.stabilizers.is_empty() {
            if self.current_stabilizer_index == 0 {
                self.current_stabilizer_index = self.stabilizers.len() - 1;
            } else {
                self.current_stabilizer_index -= 1;
            }
        }
    }

    /// Steps to the next stabilizer, wrapping to the first.
    pub(crate) fn next_stabilizer(&mut self) {
        if !self.stabilizers.is_empty() {
            self.current_stabilizer_index =
                (self.current_stabilizer_index + 1) % self.stabilizers.len();
        }
    }

    /// Drops all cached stabilizers and hides the overlay.
    pub(crate) fn clear_stabilizers(&mut self) {
        self.reset_stabilizer_browse();
        self.action_stabilizer = None;
    }

    /// Drops the Q/E-browsable generator list, leaving any Actions-window
    /// selection alone — that one is re-derived from the viewer every frame,
    /// so clearing it only costs a redundant re-render.
    fn reset_stabilizer_browse(&mut self) {
        self.stabilizers.clear();
        self.current_stabilizer_index = 0;
        self.show_stabilizers = false;
    }
}

impl Default for EditorState {
    fn default() -> Self {
        let default_theme = crate::theme::palette(ThemePreset::default());
        Self {
            block_kind: BlockKind::Cube(CubeKind::XZZ),
            placement_tool: PlacementTool::Block,
            pipe_length: 2.0,
            mode: EditorMode::View,
            transform_axis: UDirection::X,
            plane_height: 0,
            show_axis: true,
            show_grid: true,
            show_port_tags: false,
            bg_color: Color::srgba_u8(
                default_theme.bg_dark.r(),
                default_theme.bg_dark.g(),
                default_theme.bg_dark.b(),
                default_theme.bg_dark.a(),
            ),
            hovered_grid_pos: None,
            hovered_element: None,
            last_element_click: None,
            hovered_preview_source: None,
            hovered_preview_endpoint: None,
            blog_hovered_elements: HashSet::new(),
            zx_hovered_elements: HashSet::new(),
            action_hovered_elements: HashSet::new(),
            action_candidate_elements: HashSet::new(),
            selected_elements: HashSet::new(),
            pipe_start: None,
            last_pipe_placement: None,
            walking_start: None,
            branch_name: "b0".into(),
            view_current_layer_only: false,
            show_blog_buffer: false,
            show_help_window: false,
            theme_preset: ThemePreset::default(),
            gallery_panel_expanded: false,
            show_side_panel: false,
            selected_gallery_category: None,
            gallery_search: String::new(),
            fill_ports_cycle: None,
            module_view: None,
            highlight_material: Handle::default(),
            selection_material: Handle::default(),
            preview_endpoint_material: Handle::default(),
            preview_endpoint_hover_material: Handle::default(),
            action_candidate_material: Handle::default(),
            request_screenshot: false,
            taking_screenshot: false,
            stabilizers: Vec::new(),
            current_stabilizer_index: 0,
            show_stabilizers: false,
            action_stabilizer: None,
        }
    }
}

fn graph_contains_element(graph: &BlockGraph, element: GraphElement) -> bool {
    match element.canonical() {
        GraphElement::Block(pos) => graph.has_block_at(pos),
        GraphElement::Pipe(u, v) => graph.has_pipe_between(u, v),
    }
}

/// Severity controls colour, popup priority, and expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ToastLevel {
    Info,
    Warn,
    Error,
}

/// A notification retained in session history after its popup closes.
pub(crate) struct Toast {
    pub(crate) message: String,
    pub(crate) level: ToastLevel,
    pub(crate) created_at: Instant,
    pub(crate) id: u64,
    pub(crate) occurrences: u32,
    pub(crate) read: bool,
    pub(crate) popup: bool,
    pub(crate) expires_at: Option<Instant>,
}

impl ToastLevel {
    pub(crate) fn popup_duration(self) -> Option<Duration> {
        match self {
            Self::Info => Some(Duration::from_secs(6)),
            Self::Warn => Some(Duration::from_secs(12)),
            Self::Error => None,
        }
    }
}

/// Bound session history even when many distinct failures arrive.
const MAX_TOASTS: usize = 50;

/// Session notification history and its presentation state. Ingress also logs
/// at the matching level; hiding a popup never discards the full message.
#[derive(Resource, Default)]
pub(crate) struct Notifications {
    pub(crate) toasts: VecDeque<Toast>,
    pub(crate) center_open: bool,
    pub(crate) quiet: bool,
    /// Includes the closing frame: egui and Bevy receive separate key events.
    pub(crate) capture_keyboard: bool,
    next_id: u64,
}

impl Notifications {
    /// Logs at info level and shows an info toast.
    pub(crate) fn push_info(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        info!("{}", s);
        self.push(s, ToastLevel::Info, true);
    }

    /// Records routine job activity without interrupting the canvas.
    pub(crate) fn record_info(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        info!("{}", s);
        self.push(s, ToastLevel::Info, false);
    }

    /// Logs at warn level and shows a warning toast.
    pub(crate) fn push_warn(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        warn!("{}", s);
        self.push(s, ToastLevel::Warn, true);
    }

    /// Logs at error level and shows an error toast.
    pub(crate) fn push_error(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        error!("{}", s);
        self.push(s, ToastLevel::Error, true);
    }

    /// The single point where an error report becomes toast text: renders
    /// `action` plus the full cause chain on one line
    /// ("save compile output: create export directory /x: permission denied").
    pub(crate) fn push_error_report(&mut self, action: &str, error: &Report) {
        error!("{action}: {error:?}");
        self.push(format!("{action}: {error:#}"), ToastLevel::Error, true);
    }

    /// Whether a toast with exactly this message is already queued (used to
    /// deduplicate a system that fails every frame).
    pub(crate) fn contains_message(&self, message: &str) -> bool {
        self.toasts.iter().any(|toast| toast.message == message)
    }

    pub(crate) fn unread_count(&self) -> usize {
        self.toasts.iter().filter(|toast| !toast.read).count()
    }

    /// Errors take priority; the newest entry wins within a severity.
    pub(crate) fn visible_toast(&self) -> Option<&Toast> {
        self.toasts
            .iter()
            .filter(|toast| toast.popup)
            .max_by_key(|toast| toast.level)
    }

    pub(crate) fn open_center(&mut self) {
        self.center_open = true;
        self.capture_keyboard = true;
        for toast in &mut self.toasts {
            toast.read = true;
            toast.popup = false;
        }
    }

    pub(crate) fn hide_popup(&mut self, id: u64) {
        if let Some(toast) = self.toasts.iter_mut().find(|toast| toast.id == id) {
            toast.popup = false;
            toast.read = true;
        }
    }

    fn push(&mut self, message: String, level: ToastLevel, popup: bool) {
        let now = Instant::now();
        let read = self.center_open || !popup;
        let popup = popup && !self.center_open && (!self.quiet || level == ToastLevel::Error);
        // Group exact repeats without conflating different severities or causes.
        let (id, occurrences) = if let Some(index) = self
            .toasts
            .iter()
            .position(|toast| toast.level == level && toast.message == message)
        {
            let toast = self
                .toasts
                .remove(index)
                .expect("existing notification index");
            (toast.id, toast.occurrences.saturating_add(1))
        } else {
            let id = self.next_id;
            self.next_id += 1;
            (id, 1)
        };
        self.toasts.push_back(Toast {
            message,
            level,
            created_at: now,
            id,
            occurrences,
            read,
            popup,
            expires_at: level.popup_duration().map(|duration| now + duration),
        });
        while self.toasts.len() > MAX_TOASTS {
            self.toasts.pop_front();
        }
    }
}

pub(crate) use crate::systems::ui::blog_buffer::ImportExportState;

#[cfg(test)]
mod tests {
    use super::{
        BloqNodeCategory, BloqNodeView, BloqViewerState, CompileUiState, CompiledBloqView,
        DetsliceData, EditorMode, EditorState, EditorTabId, EditorTabs, FlatMoment, FlatMomentOp,
        GraphEditDelta, GraphRotationAvailability, GraphState, GraphUiSummary, LayerCircuitView,
        ModuleViewState, Notifications, SliceRegionId, SliceVisibility, ToastLevel,
        ValidationState,
    };
    use crate::components::GraphElement;
    use crate::systems::ui::circuit_viewer::LazyToggle;
    use bloq_circuit::GateType;
    use bloq_graph::{
        Block, BlockGraph, BlockKind, CubeKind, Direction, PauliString, Pipe, Stabilizer,
        StabilizerGenerator, StabilizerRowKind,
    };
    use bloq_ir::{Bloq, MomentKind};
    use glam::{IVec2, IVec3};
    use std::sync::Arc;
    use std::{
        collections::{HashMap, HashSet},
        time::Duration,
    };

    #[test]
    fn errors_keep_history_while_latest_error_takes_popup_priority() {
        let mut notifications = Notifications::default();
        notifications.push_error("old");
        notifications.push_warn("warning");
        notifications.push_error("new");

        assert_eq!(
            notifications
                .toasts
                .iter()
                .map(|toast| (toast.level, toast.message.as_str()))
                .collect::<Vec<_>>(),
            [
                (ToastLevel::Error, "old"),
                (ToastLevel::Warn, "warning"),
                (ToastLevel::Error, "new")
            ]
        );
        notifications.push_info("finished");
        let error_id = notifications.visible_toast().unwrap().id;
        assert_eq!(notifications.visible_toast().unwrap().message, "new");
        notifications.hide_popup(error_id);
        assert!(notifications.contains_message("new"));
        assert_eq!(notifications.unread_count(), 3);
        notifications.open_center();
        assert_eq!(notifications.unread_count(), 0);
        assert!(notifications.visible_toast().is_none());
    }

    #[test]
    fn notifications_group_repeats_and_bound_history() {
        let mut notifications = Notifications::default();
        notifications.push_warn("repeat");
        let id = notifications.toasts[0].id;
        notifications.push_info("another event");
        notifications.push_warn("repeat");
        assert_eq!(notifications.toasts.len(), 2);
        assert_eq!(notifications.toasts.back().unwrap().occurrences, 2);
        assert_eq!(notifications.toasts.back().unwrap().id, id);
        notifications.push_error("repeat");
        assert_eq!(notifications.toasts.len(), 3);
        for index in 0..60 {
            notifications.record_info(format!("event {index}"));
        }
        assert_eq!(notifications.toasts.len(), super::MAX_TOASTS);
        assert_eq!(notifications.toasts.front().unwrap().message, "event 10");
    }

    #[test]
    fn quiet_mode_keeps_history_and_still_surfaces_errors() {
        let mut notifications = Notifications {
            quiet: true,
            ..Default::default()
        };
        notifications.push_info("saved");
        notifications.push_warn("warning");
        assert!(notifications.visible_toast().is_none());
        notifications.push_error("failure");
        assert_eq!(notifications.visible_toast().unwrap().message, "failure");
        notifications.open_center();
        notifications.push_error("another failure");
        assert!(notifications.visible_toast().is_none());
        assert_eq!(notifications.unread_count(), 0);
    }

    /// A minimal compiled view carrying one quantum node, for driving the
    /// viewer-state machine without a real compile.
    fn stub_compiled_view() -> CompiledBloqView {
        CompiledBloqView {
            viewer_graph: BlockGraph::default(),
            nodes: vec![BloqNodeView {
                id: 0,
                layer: 0,
                label: "N0".to_string(),
                category: BloqNodeCategory::QuantumBlock,
                attributes: Vec::new(),
                operator_table: Vec::new(),
                source_elements: HashSet::new(),
                moments: stub_moments(2),
                qubit_coords: HashMap::from([(0usize, IVec2::new(0, 0))]),
                num_qubits: 1,
                parent: None,
                has_quantum_body: false,
            }],
            edges: Vec::new(),
            code_distance: 3,
            compile_config: bloq_compile::CompileConfig::default(),
            compile_duration: Duration::from_millis(1),
            program: Arc::new(Bloq::new()),
            source_program: Arc::new(Bloq::new()),
            branch_pins: None,
            branch_pin_draft: Default::default(),
            source_offset: IVec3::ZERO,
        }
    }

    fn dummy_stabilizer() -> StabilizerGenerator {
        StabilizerGenerator::new(
            Stabilizer {
                paulis: PauliString::new(0),
                sign: false,
                port_stabilizer: Default::default(),
                interior_nodes: Default::default(),
                interior_edges: Default::default(),
            },
            StabilizerRowKind::Measurement {
                name: "s0".to_string(),
            },
        )
    }

    #[test]
    fn graph_ui_summary_tracks_open_counts_and_rotation() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph_state
            .graph
            .add_block(Block::new(IVec3::X, BlockKind::Y));
        graph_state
            .graph
            .add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph_state.commit();

        let summary = GraphUiSummary::from_graph_state(&graph_state);

        assert_eq!(summary.revision, graph_state.revision);
        assert_eq!(summary.block_count, 2);
        assert_eq!(summary.pipe_count, 1);
        assert!(summary.is_open);
        assert!(!summary.is_empty);
        assert_eq!(
            summary.rotation_availability,
            GraphRotationAvailability::HalfTurnsOnly
        );
    }

    #[test]
    fn graph_state_tracks_known_edit_delta_until_next_commit() {
        let mut graph_state = GraphState::default();
        graph_state.commit_with_delta(
            true,
            GraphEditDelta::PipeAdded {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: true,
            },
        );

        assert!(matches!(
            graph_state.edit_delta,
            GraphEditDelta::PipeAdded {
                src: IVec3::ZERO,
                dst: IVec3::X,
                hadamard: true,
            }
        ));

        graph_state.commit();
        assert!(matches!(graph_state.edit_delta, GraphEditDelta::Unknown));
    }

    #[test]
    fn graph_ui_summary_skips_unchanged_revision() {
        let mut graph_state = GraphState::default();
        let mut summary = GraphUiSummary::from_graph_state(&graph_state);

        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        summary.sync_from_graph_state(EditorTabId::new(1), &graph_state);

        assert_eq!(summary.block_count, 0);
        assert!(!summary.is_open);
    }

    fn gate_moment(kind: MomentKind, gate: GateType) -> FlatMoment {
        FlatMoment {
            kind,
            ops: vec![FlatMomentOp::Gate {
                gate,
                qubits: vec![0],
            }],
            repeat_label: None,
            repeat_span: None,
        }
    }

    #[test]
    fn pending_validation_message_turns_stale_when_graph_advances() {
        let mut state = CompileUiState::default();
        state.mark_validation_pending(4);
        state.mark_stale_if_graph_changed(5);

        assert_eq!(state.validation_state, ValidationState::Pending);
        assert_eq!(state.validated_revision, None);
        assert_eq!(state.pending_revision, Some(4));
        assert!(state.validation_message.contains("older revision"));
    }

    #[test]
    fn stale_validation_clears_validated_revision() {
        let mut state = CompileUiState {
            validation_state: ValidationState::Passed,
            validation_message: String::new(),
            validated_revision: Some(7),
            pending_revision: None,
            ..CompileUiState::default()
        };

        state.mark_stale_if_graph_changed(8);

        assert_eq!(state.validation_state, ValidationState::Unknown);
        assert_eq!(state.validated_revision, None);
        assert_eq!(state.pending_revision, None);
    }

    #[test]
    fn editor_selection_canonicalizes_and_toggles_elements() {
        let mut state = EditorState::default();
        let reversed_pipe = GraphElement::Pipe(IVec3::new(2, 0, 0), IVec3::new(1, 0, 0));
        let canonical_pipe = reversed_pipe.canonical();

        state.select_element(reversed_pipe, false);
        assert_eq!(state.selection_count(), 1);
        assert!(state.is_selected(canonical_pipe));

        state.select_element(canonical_pipe, true);
        assert_eq!(state.selection_count(), 0);
        assert!(!state.is_selected(canonical_pipe));
    }

    #[test]
    fn editor_click_selection_toggles_already_selected_element() {
        let mut state = EditorState::default();
        let element = GraphElement::Block(IVec3::new(0, 0, 0));

        state.click_select_element(element, false);
        assert_eq!(state.selection_count(), 1);
        assert!(state.is_selected(element));

        state.click_select_element(element, false);
        assert_eq!(state.selection_count(), 0);
        assert!(!state.is_selected(element));
    }

    #[test]
    fn module_view_keeps_viewport_hover_out_of_highlights() {
        let element = GraphElement::Block(IVec3::ZERO);
        let mut state = EditorState {
            hovered_element: Some(element),
            ..EditorState::default()
        };

        assert_eq!(state.hovered_element_for_highlight(), Some(element));
        state.mode = EditorMode::Module;
        assert_eq!(state.hovered_element_for_highlight(), None);
    }

    #[test]
    fn module_view_groups_reused_instances_by_definition() {
        let adder = crate::utils::one_bit_adder_fixture();
        let graph = adder.flatten().unwrap();
        let view = ModuleViewState::from_graph(&adder, &graph).unwrap();

        assert_eq!(
            view.modules()
                .iter()
                .map(|module| (module.name.as_str(), module.instance_count))
                .collect::<Vec<_>>(),
            [("And", 1), ("Maj", 1), ("Uma", 1)]
        );

        for block in adder.local_body().blocks() {
            assert_eq!(
                view.module_for_element(&graph, GraphElement::Block(block.pos())),
                None
            );
        }
        let mut graph_state = GraphState {
            graph,
            ..GraphState::default()
        };
        let mut editor = EditorState {
            module_view: Some(view),
            ..EditorState::default()
        };
        editor.set_mode(EditorMode::Module, &mut graph_state);
        assert_eq!(editor.mode, EditorMode::Module);
        editor.sync_after_graph_edit(&mut graph_state);
        assert_eq!(editor.mode, EditorMode::Module);
        assert!(editor.module_view.is_none());
    }

    #[test]
    fn select_all_elements_includes_blocks_and_pipes() {
        use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let mut state = EditorState::default();

        state.select_all_elements(&graph);

        assert_eq!(state.selection_count(), 3);
        assert!(state.is_selected(GraphElement::Block(IVec3::ZERO)));
        assert!(state.is_selected(GraphElement::Block(IVec3::X)));
        assert!(state.is_selected(GraphElement::Pipe(IVec3::ZERO, IVec3::X)));
    }

    #[test]
    fn graph_edit_sync_prunes_stale_selection_and_hover() {
        use bloq_graph::{Block, BlockKind, CubeKind};

        let pos = IVec3::new(1, 0, 0);
        let mut graph_state = super::GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pos, BlockKind::Cube(CubeKind::XZZ)));

        let mut state = EditorState::default();
        state.select_element(GraphElement::Block(pos), false);
        state.hovered_element = Some(GraphElement::Block(pos));
        state.pipe_start = Some(pos);

        graph_state.graph.remove_block(pos);
        state.sync_after_graph_edit(&mut graph_state);

        assert_eq!(state.selection_count(), 0);
        assert_eq!(state.hovered_element, None);
        assert_eq!(state.pipe_start, None);
    }

    #[test]
    fn toggling_stabilizers_uses_cached_results() {
        let mut state = EditorState::default();

        assert!(!state.toggle_stabilizers());
        assert!(!state.showing_stabilizers());

        state.stabilizers.push(dummy_stabilizer());

        assert!(state.toggle_stabilizers());
        assert!(state.showing_stabilizers());

        assert!(state.toggle_stabilizers());
        assert!(!state.showing_stabilizers());
    }

    #[test]
    fn clearing_stabilizers_hides_stabilizer_view() {
        let mut state = EditorState::default();
        state.stabilizers.push(dummy_stabilizer());
        state.current_stabilizer_index = 1;
        state.show_stabilizers = true;

        state.clear_stabilizers();

        assert!(state.stabilizers.is_empty());
        assert_eq!(state.current_stabilizer_index, 0);
        assert!(!state.show_stabilizers);
    }

    #[test]
    fn editor_tabs_create_untitled_tabs_and_preserve_snapshots() {
        let mut tabs = EditorTabs::default();
        assert!(tabs.show_welcome);
        let template = EditorState::default();
        let first = tabs.active;
        let second = tabs.add_empty_tab(&template);
        assert!(!tabs.show_welcome);
        assert!(tabs.set_active(second));
        assert!(tabs.set_active(first));
        assert!(!tabs.show_welcome);

        assert_eq!(tabs.title(first), Some("Untitled 1"));
        assert_eq!(tabs.title(second), Some("Untitled 2"));

        let mut snapshot = tabs.active_tab().snapshot.clone();
        snapshot
            .graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZZ)));
        tabs.save_active(snapshot);

        assert!(!tabs.active_tab().snapshot.graph_state.graph.is_empty());

        tabs.remove(second, &template);
        tabs.remove(first, &template);
        assert!(
            !tabs.show_welcome,
            "closing all tabs keeps the welcome dismissed"
        );

        let tabs = EditorTabs::default();
        let restored = EditorTabs::from_session(tabs.tabs, tabs.active).unwrap();
        assert!(!restored.show_welcome);
    }

    #[test]
    fn circuit_viewer_finish_compile_updates_metadata() {
        let mut state = BloqViewerState::default();
        state.graph_layout_cache.key = Some((Some(7), false));

        state.finish_compile(
            7,
            CompiledBloqView {
                code_distance: 5,
                compile_config: bloq_compile::CompileConfig::new(5),
                compile_duration: Duration::from_millis(12),
                ..stub_compiled_view()
            },
        );

        assert_eq!(state.compile_revision, Some(7));
        assert_eq!(state.code_distance, Some(5));
        assert_eq!(state.compile_duration, Some(Duration::from_millis(12)));
        assert_eq!(state.selected_node, None);
        assert_eq!(state.num_qubits, 0);
        assert!(state.moments.is_empty());
        assert!(state.graph_layout_cache.key.is_none());
    }

    #[test]
    fn concurrent_ops_runs_lazily_and_discards_stale_results() {
        let mut state = BloqViewerState::default();
        state.finish_compile(1, stub_compiled_view());
        assert_eq!(state.concurrent_ops_toggle(), LazyToggle::Off);
        assert!(state.layer_circuits.is_empty());

        state.begin_concurrent_ops();
        let old_generation = state.concurrent_ops_inputs().unwrap().generation;
        assert_eq!(state.concurrent_ops_toggle(), LazyToggle::Pending);
        let mut recompiled = stub_compiled_view();
        recompiled.code_distance = 5;
        state.finish_compile(1, recompiled);
        let layers = HashMap::from([(
            0,
            LayerCircuitView {
                moments: stub_moments(1),
                source_moments: vec![Vec::new()],
                qubit_coords: HashMap::new(),
                num_qubits: 0,
            },
        )]);

        assert!(!state.finish_concurrent_ops(old_generation, layers.clone(), None));
        assert_eq!(state.concurrent_ops_toggle(), LazyToggle::Pending);
        assert!(state.finish_concurrent_ops(state.compile_generation, layers, None));
        assert_eq!(state.concurrent_ops_toggle(), LazyToggle::On);
    }

    fn stub_moments(count: usize) -> Vec<FlatMoment> {
        (0..count)
            .map(|_| FlatMoment {
                kind: MomentKind::Rotation,
                ops: Vec::new(),
                repeat_label: None,
                repeat_span: None,
            })
            .collect()
    }

    fn viewer_with_flat_timeline(default_moments: usize, flat_moments: usize) -> BloqViewerState {
        let mut state = BloqViewerState {
            nodes: vec![BloqNodeView {
                id: 0,
                layer: 0,
                label: "N0".to_string(),
                category: BloqNodeCategory::QuantumBlock,
                attributes: Vec::new(),
                operator_table: Vec::new(),
                source_elements: HashSet::new(),
                moments: stub_moments(default_moments),
                qubit_coords: HashMap::from([(0usize, IVec2::new(0, 0))]),
                num_qubits: 1,
                parent: None,
                has_quantum_body: false,
            }],
            flat_moments: HashMap::from([(0u32, stub_moments(flat_moments))]),
            ..Default::default()
        };
        state.set_selected_node(Some(0));
        state
    }

    fn detectors_only() -> SliceVisibility {
        SliceVisibility {
            detectors: true,
            observables: false,
        }
    }

    #[test]
    fn detector_and_observable_slices_toggle_independently() {
        let mut state = viewer_with_flat_timeline(2, 5);
        assert_eq!(state.moments.len(), 2, "default view before enabling");

        state.set_slice_visibility(detectors_only());
        assert_eq!(state.moments.len(), 5, "flattened timeline while on");

        state.set_slice_visibility(SliceVisibility {
            detectors: false,
            observables: true,
        });
        assert_eq!(state.moments.len(), 5, "observable-only view stays flat");

        state.selected_region = Some(SliceRegionId::Observable { index: 4 });
        state.set_slice_visibility(detectors_only());
        assert_eq!(state.selected_region, None, "hidden selection is cleared");

        state.set_slice_visibility(SliceVisibility::default());
        assert_eq!(state.moments.len(), 2, "default timeline restored");
    }

    #[test]
    fn switching_detslice_mode_clamps_the_moment_cursor() {
        let mut state = viewer_with_flat_timeline(2, 5);
        state.set_slice_visibility(detectors_only());
        state.set_current_moment(4);
        assert_eq!(state.current_moment, 4);

        // The shorter default timeline cannot hold moment 4; the cursor clamps
        // to the last valid index instead of dangling out of range.
        state.set_slice_visibility(SliceVisibility::default());
        assert_eq!(state.current_moment, 1);
    }

    #[test]
    fn detslice_mode_stays_off_when_unavailable() {
        let mut state = BloqViewerState::default();
        assert!(!state.detslice_available(), "no flattened view built");

        state.set_slice_visibility(detectors_only());
        assert!(
            !state.slice_visibility.detectors,
            "toggle ignored without slice data"
        );
    }

    /// The lazy state machine: enabling before the cache is computed parks the
    /// viewer in a pending state; applying the job result activates the mode; a
    /// recompile invalidates the cache and degrades the mode back to off.
    #[test]
    fn detslice_lazy_state_machine_pending_activate_invalidate() {
        let mut state = BloqViewerState::default();
        state.finish_compile(1, stub_compiled_view());
        state.set_selected_node(Some(0));

        // Fresh compile: nothing computed yet, so the toggle is simply Off.
        assert_eq!(state.detslice_toggle(), LazyToggle::Off);
        assert_eq!(state.moments.len(), 2, "default timeline before enabling");

        // Enabling before the cache exists parks the viewer in Pending; the mode
        // is not yet on and the default timeline is still shown.
        state.begin_detslice(detectors_only());
        assert_eq!(state.detslice_toggle(), LazyToggle::Pending);
        assert!(!state.slice_visibility.detectors);

        // The job lands with a populated store for the current revision; since a
        // toggle was pending, the mode activates and swaps to the flat timeline.
        let data = DetsliceData {
            flat_moments: HashMap::from([(0u32, stub_moments(5))]),
            ..Default::default()
        };
        state.finish_detslice(state.compile_generation, data);
        assert!(state.detslice_pending.is_none());
        assert!(state.detslice_computed);
        assert_eq!(state.detslice_toggle(), LazyToggle::On);
        assert_eq!(state.moments.len(), 5, "flattened timeline while on");

        // A recompile invalidates the cache and degrades the mode back to off.
        state.finish_compile(2, stub_compiled_view());
        assert!(!state.slice_visibility.detectors);
        assert!(!state.detslice_computed);
        assert!(state.flat_moments.is_empty());
        state.set_selected_node(Some(0));
        assert_eq!(state.detslice_toggle(), LazyToggle::Off);
    }

    /// A detector-slice result from a superseded compile is discarded rather than
    /// polluting the current view's cache.
    #[test]
    fn detslice_result_from_an_earlier_compile_of_the_same_revision_is_discarded() {
        let mut state = BloqViewerState::default();
        state.finish_compile(2, stub_compiled_view());
        let old_generation = state.detslice_inputs().unwrap().generation;
        // Resetting/replacing the viewer must not reuse a previous generation either.
        state.reset_for_new_graph();
        let mut recompiled = stub_compiled_view();
        recompiled.code_distance = 5;
        state.finish_compile(2, recompiled);
        state.begin_detslice(detectors_only());

        // The previous compile has the same source revision but different qubits.
        let data = DetsliceData {
            flat_moments: HashMap::from([(0u32, stub_moments(5))]),
            ..Default::default()
        };
        assert!(!state.finish_detslice(old_generation, data));

        assert!(!state.detslice_computed, "stale store not installed");
        assert!(
            state.detslice_pending.is_some(),
            "newer pending intent preserved"
        );
        assert!(state.flat_moments.is_empty());
    }

    /// A computed-but-unavailable overlay leaves the mode off and reports the
    /// reason through the toggle state.
    #[test]
    fn detslice_unavailable_result_surfaces_reason() {
        let mut state = BloqViewerState::default();
        state.finish_compile(1, stub_compiled_view());
        state.set_selected_node(Some(0));
        state.begin_detslice(detectors_only());

        let data = DetsliceData {
            unavailable_reason: Some("Flattening failed".to_string()),
            ..Default::default()
        };
        state.finish_detslice(state.compile_generation, data);

        assert!(
            !state.slice_visibility.any(),
            "cannot enable an empty overlay"
        );
        assert_eq!(
            state.detslice_toggle(),
            LazyToggle::Unavailable("Flattening failed".to_string())
        );
    }

    #[test]
    fn circuit_viewer_begin_compile_preserves_previous_result_until_success() {
        let mut state = BloqViewerState {
            moments: vec![gate_moment(MomentKind::Reset, GateType::RZ)],
            qubit_coords: HashMap::from([(0usize, IVec2::new(0, 0))]),
            num_qubits: 1,
            compile_revision: Some(3),
            code_distance: Some(5),
            compile_duration: Some(Duration::from_millis(9)),
            ..BloqViewerState::default()
        };

        state.begin_compile(4);

        assert_eq!(state.pending_revision, Some(4));
        assert_eq!(state.compile_revision, Some(3));
        assert_eq!(state.moments.len(), 1);
        assert_eq!(state.num_qubits, 1);
    }

    #[test]
    fn circuit_viewer_navigation_moves_between_moments() {
        let selected = SliceRegionId::Observable { index: 7 };
        let mut state = BloqViewerState {
            moments: vec![
                gate_moment(MomentKind::Rotation, GateType::H),
                gate_moment(MomentKind::Reset, GateType::RZ),
                gate_moment(MomentKind::Rotation, GateType::S),
            ],
            current_moment: 2,
            selected_region: Some(selected),
            ..BloqViewerState::default()
        };

        assert_eq!(state.previous_moment_index(), Some(1));
        assert_eq!(state.next_moment_index(), None);
        state.set_current_moment(1);
        assert_eq!(state.selected_region, Some(selected));
    }
}
