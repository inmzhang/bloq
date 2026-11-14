//! Durable editor documents and undo history. Runtime assets, jobs and caches are rebuilt.
//! The active tab is captured from live resources; its stashed snapshot is stale.

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bloq_graph::{Block, BlockGraph, BlockKind, BranchArm, UDirection};
use color_eyre::eyre::{ContextCompat, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::components::{CameraSettings, GraphElement};
use crate::resources::{
    CompileOutputFormat, CompileUiState, EditorMode, EditorState, EditorTab, EditorTabId,
    EditorTabSnapshot, EditorTabs, GraphState, ImportExportState, PendingBranchArm, PlacementTool,
};
use crate::systems::ui::DefinitionEdit;
use crate::theme::ThemePreset;

#[cfg(target_arch = "wasm32")]
pub(crate) mod browser;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;

#[cfg(any(target_arch = "wasm32", test))]
mod retry;

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod recovery;

// Like Bloq IR, session layouts may change while the version stays at 1.
const SESSION_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Session {
    version: u32,
    active: EditorTabId,
    tabs: Vec<SavedTab>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedTab {
    id: EditorTabId,
    title: String,
    document: Arc<SavedDocument>,
    #[serde(default)]
    history: Option<Arc<SavedHistory>>,
    blog: Arc<str>,
    #[serde(default)]
    text_history: Arc<[String]>,
    #[serde(default)]
    text_conflict: bool,
    export_path: String,
    definition_edit: Option<DefinitionEdit>,
    view: SavedView,
    camera: CameraSettings,
    compile: SavedCompile,
    // Revisions only gate expensive capture; they have no meaning after reload.
    #[serde(skip)]
    graph_revision: u64,
    #[serde(skip)]
    blog_revision: u64,
    #[serde(skip)]
    text_history_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedHistory {
    entries: Vec<Arc<SavedDocument>>,
    current: usize,
    #[serde(skip)]
    cache_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedDocument {
    graph: String,
    inputs: Vec<String>,
    false_arms: Vec<String>,
    // Keep the version-1 wire field while the source becomes a BlockGraph.
    #[serde(rename = "module_program")]
    source_graph: Option<String>,
    pending_arm: Option<SavedArm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedArm {
    name: String,
    captured_true: bool,
    graph: String,
    owned: HashSet<IVec3>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedView {
    #[serde(with = "block_kind")]
    block_kind: BlockKind,
    placement_tool: PlacementTool,
    mode: EditorMode,
    pipe_length: f32,
    transform_axis: usize,
    plane_height: i32,
    show_axis: bool,
    show_grid: bool,
    show_port_tags: bool,
    bg_color: [f32; 4],
    selected_elements: HashSet<GraphElement>,
    branch_name: String,
    view_current_layer_only: bool,
    show_blog_buffer: bool,
    show_help_window: bool,
    theme_preset: ThemePreset,
    gallery_panel_expanded: bool,
    show_side_panel: bool,
    gallery_search: String,
    gallery_category: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SavedCompile {
    code_distance: String,
    format: CompileOutputFormat,
    prepare_t_with_mpps: bool,
}

/// Borrow only the persistent inputs, without cloning compiled viewer caches.
struct TabSource<'a> {
    graph: &'a GraphState,
    editor: &'a EditorState,
    import: &'a ImportExportState,
    compile: &'a CompileUiState,
    camera: &'a CameraSettings,
}

impl<'a> From<&'a EditorTabSnapshot> for TabSource<'a> {
    fn from(snapshot: &'a EditorTabSnapshot) -> Self {
        Self {
            graph: &snapshot.graph_state,
            editor: &snapshot.editor_state,
            import: &snapshot.import_export,
            compile: &snapshot.compile_ui,
            camera: &snapshot.camera_settings,
        }
    }
}

/// Protect work created while an asynchronous startup recovery is pending.
struct RecoveryGuard {
    tab: EditorTabId,
    title: String,
    view: SavedView,
    camera: CameraSettings,
}

impl RecoveryGuard {
    fn capture(world: &World) -> Self {
        let tabs = world.resource::<EditorTabs>();
        Self {
            tab: tabs.active,
            title: tabs.active_tab().title.clone(),
            view: SavedView::capture(world.resource()),
            camera: tabs.active_tab().snapshot.camera_settings,
        }
    }

    fn pristine(&self, world: &mut World) -> bool {
        let camera_unchanged = world
            .query_filtered::<&CameraSettings, With<crate::components::EditorCamera>>()
            .iter(world)
            .all(|camera| *camera == self.camera);
        let tabs = world.resource::<EditorTabs>();
        let graph = world.resource::<GraphState>();
        let text = world.resource::<ImportExportState>();
        camera_unchanged
            && tabs.tabs.len() == 1
            && tabs.active == self.tab
            && tabs.active_tab().title == self.title
            && graph.graph.is_empty()
            && graph.history.len() == 1
            && graph.source_graph.is_none()
            && graph.pending_branch_arm.is_none()
            && text.text_history().is_empty()
            && (text.bloq_buffer.is_empty() || text.bloq_buffer == graph.graph.to_blog_text())
            && SavedView::capture(world.resource()) == self.view
    }
}

fn install_recovered(
    world: &mut World,
    tabs: EditorTabs,
    cache: Session,
    guard: &RecoveryGuard,
    label: &str,
) -> Result<Option<Session>, String> {
    if !guard.pristine(world) {
        let count = world.resource_mut::<EditorTabs>().append_recovered(tabs)?;
        world
            .resource_mut::<crate::resources::Notifications>()
            .push_info(format!(
                "Recovered {count} saved tab(s) beside your new work"
            ));
        return Ok(None);
    }
    let snapshot = tabs.active_tab().snapshot.clone();
    let camera = snapshot.camera_settings;
    world.resource_mut::<ClearColor>().0 = snapshot.editor_state.bg_color;
    let preset = snapshot.editor_state.theme_preset;
    world.insert_resource(snapshot.graph_state);
    world.insert_resource(snapshot.editor_state);
    world.insert_resource(snapshot.import_export);
    world.insert_resource(snapshot.compile_ui);
    world.insert_resource(tabs);
    for (mut transform, mut settings) in world.query_filtered::<(&mut Transform, &mut CameraSettings), With<crate::components::EditorCamera>>().iter_mut(world) {
        *settings = camera;
        crate::systems::camera::apply_camera_setting(&mut transform, &settings);
    }
    for mut context in world.query::<&mut bevy_egui::EguiContext>().iter_mut(world) {
        crate::theme::setup_theme(context.get_mut(), preset);
    }
    world
        .resource_mut::<crate::resources::Notifications>()
        .push_info(format!("Restored {label} session"));
    Ok(Some(cache))
}

impl Session {
    /// Check the envelope before decoding fields whose representation may have
    /// changed, so a future save reports its version instead of a parser error.
    fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Header {
            version: u32,
        }
        let header: Header = serde_json::from_str(json)?;
        Self::check_version(header.version)?;
        Ok(serde_json::from_str(json)?)
    }

    fn check_version(version: u32) -> Result<()> {
        ensure!(
            version == SESSION_VERSION,
            "Unsupported editor session version {version} (expected {SESSION_VERSION})"
        );
        Ok(())
    }

    fn capture(tabs: &EditorTabs, live: TabSource<'_>, previous: Option<&Self>) -> Result<Self> {
        let saved =
            tabs.tabs
                .iter()
                .map(|tab| {
                    let stashed = TabSource::from(&tab.snapshot);
                    let source = if tab.id == tabs.active {
                        &live
                    } else {
                        &stashed
                    };
                    // ponytail: linear lookup; index by ID if sessions grow to hundreds of tabs.
                    let previous = previous
                        .and_then(|session| session.tabs.iter().find(|saved| saved.id == tab.id));
                    let document =
                        match previous {
                            Some(saved)
                                if saved.graph_revision == source.graph.revision
                                    && saved.history.as_ref().and_then(|history| {
                                        history.cache_ids.get(history.current)
                                    }) == source
                                        .graph
                                        .history
                                        .get(source.graph.current_index)
                                        .map(|entry| &entry.cache_id) =>
                            {
                                Arc::clone(&saved.document)
                            }
                            _ => Arc::new(SavedDocument::capture(source.graph)?),
                        };
                    let blog = match previous {
                        Some(saved) if saved.blog_revision == source.import.bloq_revision => {
                            Arc::clone(&saved.blog)
                        }
                        _ => Arc::from(source.import.bloq_buffer.as_str()),
                    };
                    Ok(SavedTab {
                        id: tab.id,
                        title: tab.title.clone(),
                        document,
                        history: Some(Arc::new(SavedHistory::capture(
                            source.graph,
                            previous.and_then(|saved| saved.history.as_deref()),
                        )?)),
                        blog,
                        text_history: match previous {
                            Some(saved)
                                if saved.text_history_revision
                                    == source.import.text_history_revision() =>
                            {
                                Arc::clone(&saved.text_history)
                            }
                            _ => Arc::from(source.import.text_history()),
                        },
                        text_conflict: source.import.sync_conflict,
                        export_path: source.import.export_path.clone(),
                        definition_edit: source.import.definition_edit.clone(),
                        view: SavedView::capture(source.editor),
                        camera: *source.camera,
                        compile: SavedCompile {
                            code_distance: source.compile.code_distance_input.clone(),
                            format: source.compile.format,
                            prepare_t_with_mpps: source.compile.prepare_t_with_mpps,
                        },
                        graph_revision: source.graph.revision,
                        blog_revision: source.import.bloq_revision,
                        text_history_revision: source.import.text_history_revision(),
                    })
                })
                .collect::<Result<_>>()?;
        Ok(Self {
            version: SESSION_VERSION,
            active: tabs.active,
            tabs: saved,
        })
    }

    /// Decode everything before installing anything. A broken save stays intact.
    #[cfg(any(not(target_arch = "wasm32"), test))]
    fn restore(self) -> Result<EditorTabs> {
        self.restore_with(SavedDocument::restore, None)
    }

    fn restore_with(
        self,
        mut document: impl FnMut(&SavedDocument) -> Result<GraphState>,
        sync: Option<Vec<(Option<String>, bool)>>,
    ) -> Result<EditorTabs> {
        if let Some(sync) = &sync {
            ensure!(
                sync.len() == self.tabs.len(),
                "Invalid recovery sync metadata"
            );
        }
        Self::check_version(self.version)?;
        let mut restored = Vec::with_capacity(self.tabs.len());
        for (tab_index, tab) in self.tabs.into_iter().enumerate() {
            ensure!(
                tab.camera.focus.is_finite()
                    && tab.camera.radius > 0.0
                    && tab.camera.radius <= crate::systems::camera::MAX_CAMERA_RADIUS
                    && tab.camera.alpha.is_finite()
                    && (0.0..std::f32::consts::PI).contains(&tab.camera.beta),
                "Invalid saved camera"
            );
            let mut snapshot = EditorTabSnapshot {
                graph_state: document(&tab.document)?,
                camera_settings: tab.camera,
                ..default()
            };
            if let Some(history) = tab.history {
                ensure!(
                    !history.entries.is_empty() && history.current < history.entries.len(),
                    "Invalid saved undo history cursor"
                );
                snapshot.graph_state.history = history
                    .entries
                    .iter()
                    .map(|saved| {
                        let mut restored = document(saved)?;
                        restored
                            .history
                            .pop_back()
                            .wrap_err("Missing restored history entry")
                    })
                    .collect::<Result<_>>()?;
                snapshot.graph_state.current_index = history.current;
            }
            tab.view.restore(&mut snapshot.editor_state)?;
            snapshot
                .editor_state
                .sync_after_graph_edit(&mut snapshot.graph_state);
            snapshot.import_export.set_bloq_buffer(tab.blog.to_string());
            if let Some(sync) = &sync {
                let (text, conflict) = &sync[tab_index];
                snapshot.import_export.restore_recovery_sync(
                    snapshot.graph_state.revision,
                    text.clone(),
                    *conflict,
                );
            } else {
                snapshot
                    .import_export
                    .restore_graph_sync(&snapshot.graph_state);
            }
            snapshot
                .import_export
                .restore_text_history(tab.text_history.to_vec());
            snapshot.import_export.sync_conflict |= tab.text_conflict;
            snapshot.import_export.export_path = tab.export_path;
            snapshot.import_export.definition_edit = tab.definition_edit;
            snapshot.compile_ui.code_distance_input = tab.compile.code_distance;
            snapshot.compile_ui.format = tab.compile.format;
            snapshot.compile_ui.prepare_t_with_mpps = tab.compile.prepare_t_with_mpps;
            restored.push(EditorTab {
                id: tab.id,
                rename_buffer: tab.title.clone(),
                title: tab.title,
                renaming: false,
                snapshot,
            });
        }
        EditorTabs::from_session(restored, self.active).wrap_err("Invalid saved tab identities")
    }
}

impl SavedHistory {
    fn capture(state: &GraphState, previous: Option<&Self>) -> Result<Self> {
        ensure!(
            !state.history.is_empty() && state.current_index < state.history.len(),
            "Cannot save invalid undo history cursor"
        );
        let mut entries = Vec::with_capacity(state.history.len());
        let mut cache_ids = Vec::with_capacity(state.history.len());
        for entry in &state.history {
            let cached = previous.and_then(|saved| {
                saved
                    .cache_ids
                    .iter()
                    .position(|id| *id == entry.cache_id)
                    .and_then(|index| saved.entries.get(index))
            });
            entries.push(match cached {
                Some(document) => Arc::clone(document),
                None => Arc::new(SavedDocument::capture_parts(
                    &entry.graph,
                    entry.source_graph.as_deref(),
                    entry.pending_branch_arm.as_ref(),
                )?),
            });
            cache_ids.push(entry.cache_id);
        }
        Ok(Self {
            entries,
            current: state.current_index,
            cache_ids,
        })
    }
}

impl SavedDocument {
    fn capture(state: &GraphState) -> Result<Self> {
        Self::capture_parts(
            &state.graph,
            state.source_graph.as_deref(),
            state.pending_branch_arm.as_ref(),
        )
    }

    fn capture_parts(
        visible: &BlockGraph,
        source: Option<&BlockGraph>,
        pending: Option<&PendingBranchArm>,
    ) -> Result<Self> {
        Ok(Self {
            graph: visible.to_blog_body_text(),
            inputs: visible.action_graph().inputs().map(str::to_owned).collect(),
            false_arms: visible
                .branch_definitions()
                .iter()
                .filter(|branch| !branch.shown_true())
                .map(|branch| branch.name.clone())
                .collect(),
            source_graph: source.map(BlockGraph::to_blog_text),
            pending_arm: pending
                .map(|pending| -> Result<SavedArm> {
                    // An arm's cut pipes refer to prefix blocks. Include those in its
                    // BLOG container, then retain only the owned blocks on restore.
                    let mut graph = BlockGraph::new();
                    for block in pending.arm.blocks() {
                        graph.try_add_block(block.clone())?;
                    }
                    for pipe in pending.arm.pipes() {
                        for endpoint in [pipe.endpoints().0, pipe.endpoints().1] {
                            if graph.get_endpoint_block(endpoint).is_none() {
                                let block = visible
                                    .get_endpoint_block(endpoint)
                                    .wrap_err("Captured arm is missing a cut endpoint")?;
                                graph.try_add_block(block.clone())?;
                            }
                        }
                        graph.try_add_pipe(pipe.clone())?;
                    }
                    Ok(SavedArm {
                        name: pending.name.clone(),
                        captured_true: pending.captured_true,
                        graph: graph.to_blog_body_text(),
                        owned: pending.arm.blocks().map(Block::pos).collect(),
                    })
                })
                .transpose()?,
        })
    }

    fn restore(&self) -> Result<GraphState> {
        let source_graph = self
            .source_graph
            .as_ref()
            .map(|text| -> Result<_> {
                let ast = bloq_graph::parse_blog_program_to_ast(text)?;
                Ok(Arc::new(bloq_graph::lower_blog_graph_ast_deferred(&ast)?))
            })
            .transpose()?;
        let mut graph = bloq_graph::lower_blog_ast_lenient(
            &bloq_graph::parse_blog_to_ast(&self.graph)?,
            self.inputs.clone(),
        )?;
        for name in &self.false_arms {
            graph.set_shown_branch_arm(name, false)?;
        }
        let pending_branch_arm = self
            .pending_arm
            .as_ref()
            .map(|saved| {
                let graph = BlockGraph::from_blog_text(&saved.graph)?;
                restore_pending_arm(saved.name.clone(), saved.captured_true, graph, &saved.owned)
            })
            .transpose()?;
        let mut state = GraphState {
            graph,
            source_graph,
            pending_branch_arm,
            ..default()
        };
        // Establish one undo baseline; opening a session must not undo to empty.
        state.history.clear();
        state.commit();
        Ok(state)
    }
}

/// Both native parsing and worker hydration validate captured-arm ownership.
fn restore_pending_arm(
    name: String,
    captured_true: bool,
    graph: BlockGraph,
    owned: &HashSet<IVec3>,
) -> Result<PendingBranchArm> {
    ensure!(
        owned.iter().all(|pos| graph.has_block_at(*pos)),
        "Invalid captured arm"
    );
    Ok(PendingBranchArm {
        name,
        captured_true,
        arm: BranchArm::new(
            graph
                .blocks()
                .filter(|block| owned.contains(&block.pos()))
                .cloned()
                .collect(),
            graph.pipes().cloned().collect(),
        ),
    })
}

impl SavedView {
    fn capture(editor: &EditorState) -> Self {
        Self {
            block_kind: editor.block_kind,
            placement_tool: editor.placement_tool,
            mode: editor.mode,
            pipe_length: editor.pipe_length,
            transform_axis: match editor.transform_axis {
                UDirection::X => 0,
                UDirection::Y => 1,
                UDirection::Z => 2,
            },
            plane_height: editor.plane_height,
            show_axis: editor.show_axis,
            show_grid: editor.show_grid,
            show_port_tags: editor.show_port_tags,
            bg_color: editor.bg_color.to_srgba().to_f32_array(),
            selected_elements: editor.selected_elements.clone(),
            branch_name: editor.branch_name.clone(),
            view_current_layer_only: editor.view_current_layer_only,
            show_blog_buffer: editor.show_blog_buffer,
            show_help_window: editor.show_help_window,
            theme_preset: editor.theme_preset,
            gallery_panel_expanded: editor.gallery_panel_expanded,
            show_side_panel: editor.show_side_panel,
            gallery_search: editor.gallery_search.clone(),
            gallery_category: editor
                .selected_gallery_category
                .map(|category| category.to_string()),
        }
    }

    fn restore(self, editor: &mut EditorState) -> Result<()> {
        ensure!(
            self.pipe_length.is_finite() && self.pipe_length >= 0.0,
            "Invalid saved pipe length"
        );
        ensure!(
            self.bg_color
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
            "Invalid saved background color"
        );
        editor.transform_axis = *[UDirection::X, UDirection::Y, UDirection::Z]
            .get(self.transform_axis)
            .wrap_err("Invalid saved transform axis")?;
        editor.block_kind = self.block_kind;
        editor.placement_tool = self.placement_tool;
        editor.mode = self.mode;
        editor.pipe_length = self.pipe_length;
        editor.plane_height = self.plane_height;
        editor.show_axis = self.show_axis;
        editor.show_grid = self.show_grid;
        editor.show_port_tags = self.show_port_tags;
        editor.bg_color = Color::srgba(
            self.bg_color[0],
            self.bg_color[1],
            self.bg_color[2],
            self.bg_color[3],
        );
        editor.selected_elements = self.selected_elements;
        editor.branch_name = self.branch_name;
        editor.view_current_layer_only = self.view_current_layer_only;
        editor.show_blog_buffer = self.show_blog_buffer;
        editor.show_help_window = self.show_help_window;
        editor.theme_preset = self.theme_preset;
        editor.gallery_panel_expanded = self.gallery_panel_expanded;
        editor.show_side_panel = self.show_side_panel;
        editor.gallery_search = self.gallery_search;
        editor.selected_gallery_category = self
            .gallery_category
            .map(|category| category.parse())
            .transpose()?;
        Ok(())
    }
}

/// Reuse BLOG's full block-kind codec (including walking/rotation parameters).
mod block_kind {
    use super::*;

    pub(super) fn serialize<S: serde::Serializer>(
        kind: &BlockKind,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&Block::new(IVec3::ZERO, *kind))
    }

    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BlockKind, D::Error> {
        let text = String::deserialize(deserializer)?;
        let source = format!("BLOG 1.0\n0: {text}\n");
        let graph = BlockGraph::from_blog_text(&source).map_err(|error| {
            serde::de::Error::custom(format!("Invalid saved placement block: {error}"))
        })?;
        if graph.block_count() != 1 {
            return Err(serde::de::Error::custom(
                "Saved placement kind must contain exactly one block",
            ));
        }
        graph
            .get_block(IVec3::ZERO)
            .map(Block::kind)
            .ok_or_else(|| serde::de::Error::custom("Saved placement block is missing"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn recovery_does_not_replace_work_created_while_pending() {
        for draft in ["new draft", " \n\t"] {
            let mut world = World::new();
            world.init_resource::<EditorTabs>();
            world.init_resource::<GraphState>();
            world.init_resource::<EditorState>();
            world.init_resource::<ImportExportState>();
            world.init_resource::<crate::resources::Notifications>();
            let guard = RecoveryGuard::capture(&world);
            let active = world.resource::<EditorTabs>().active;
            world
                .resource_mut::<ImportExportState>()
                .set_bloq_buffer(draft.into());
            let restored = EditorTabs::default();
            let cache = Session::capture(&restored, (&restored.active_tab().snapshot).into(), None)
                .unwrap();
            assert!(
                install_recovered(&mut world, restored, cache, &guard, "test")
                    .unwrap()
                    .is_none()
            );
            assert_eq!(world.resource::<EditorTabs>().active, active);
            assert_eq!(world.resource::<EditorTabs>().tabs.len(), 2);
            assert_eq!(world.resource::<ImportExportState>().bloq_buffer, draft);
        }
    }

    #[test]
    fn typing_reuses_recovery_history_but_history_replacement_invalidates_it() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.import_export
            .replace_text_preserving_history("original".into());
        live.import_export
            .replace_text_preserving_history("draft".into());
        let first = capture(&tabs, &live, None);
        live.import_export.set_bloq_buffer("typing".into());
        let typed = capture(&tabs, &live, Some(&first));
        assert!(Arc::ptr_eq(
            &first.tabs[0].text_history,
            &typed.tabs[0].text_history
        ));
        live.import_export
            .replace_text_preserving_history("recovered".into());
        let recovered = capture(&tabs, &live, Some(&typed));
        assert!(!Arc::ptr_eq(
            &typed.tabs[0].text_history,
            &recovered.tabs[0].text_history
        ));
        assert_eq!(
            recovered.tabs[0].text_history.as_ref(),
            ["original", "typing"]
        );
        live.import_export
            .restore_text_history(vec!["restored separately".into()]);
        let restored = capture(&tabs, &live, Some(&recovered));
        assert_eq!(
            restored.tabs[0].text_history.as_ref(),
            ["restored separately"]
        );
    }

    #[test]
    fn legacy_session_defaults_to_empty_text_recovery_history() {
        let saved = Session::from_json(SAVED_SESSION).unwrap();
        assert!(
            saved
                .tabs
                .iter()
                .all(|tab| tab.text_history.is_empty() && !tab.text_conflict)
        );
        let tabs = saved.restore().unwrap();
        assert!(
            tabs.tabs
                .iter()
                .all(|tab| tab.snapshot.import_export.text_history().is_empty())
        );
    }

    #[test]
    fn recovery_history_is_per_tab_and_survives_repeated_restarts() {
        let mut tabs = EditorTabs::default();
        tabs.add_empty_tab(&EditorState::default());
        tabs.tabs[1]
            .snapshot
            .import_export
            .replace_text_preserving_history("other original".into());
        tabs.tabs[1]
            .snapshot
            .import_export
            .replace_text_preserving_history("other draft".into());
        let mut live = tabs.active_tab().snapshot.clone();
        live.import_export
            .replace_text_preserving_history("original commented text".into());
        live.import_export
            .replace_text_preserving_history("incomplete input {".into());
        live.import_export.sync_conflict = true;
        let saved = capture(&tabs, &live, None);
        tabs = reopen(&saved).unwrap();
        let restored = &tabs.active_tab().snapshot;
        assert_eq!(
            restored.import_export.text_history(),
            ["original commented text"]
        );
        assert_eq!(restored.import_export.bloq_buffer, "incomplete input {");
        assert!(restored.import_export.sync_conflict);
        let second = capture(&tabs, restored, None);
        let reopened = reopen(&second).unwrap();
        assert_eq!(
            reopened.active_tab().snapshot.import_export.text_history(),
            ["original commented text"]
        );
        assert_eq!(second.tabs[0].text_history, saved.tabs[0].text_history);
        assert_eq!(
            reopened.tabs[1].snapshot.import_export.text_history(),
            ["other original"]
        );
        assert_eq!(
            reopened.tabs[1].snapshot.import_export.bloq_buffer,
            "other draft"
        );
    }

    #[test]
    fn recovered_invalid_text_and_later_graph_edits_refresh_cached_history() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.graph_state = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n");
        let original = live.graph_state.graph.to_blog_text();
        live.import_export.set_bloq_buffer(original.clone());
        live.import_export
            .mark_synchronized(live.graph_state.revision);
        let initial = capture(&tabs, &live, None);
        live.import_export.set_bloq_buffer("invalid draft {".into());
        live.import_export
            .replace_text_preserving_history(original.clone());
        live.import_export
            .mark_synchronized(live.graph_state.revision);
        live.graph_state.graph.add_block(Block::new(
            IVec3::X,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        live.graph_state.commit();
        live.import_export.refresh_from_graph(&live.graph_state);
        let saved = capture(&tabs, &live, Some(&initial));
        let reopened = reopen(&saved).unwrap();
        assert_eq!(
            reopened.active_tab().snapshot.import_export.text_history(),
            ["invalid draft {", original.as_str()]
        );
        let unchanged = capture(&tabs, &live, Some(&saved));
        assert!(Arc::ptr_eq(
            &saved.tabs[0].text_history,
            &unchanged.tabs[0].text_history
        ));
    }

    #[test]
    fn restored_text_history_budget_also_updates_autosave_cache() {
        let tabs = EditorTabs::default();
        let mut saved = capture(&tabs, &tabs.active_tab().snapshot, None);
        saved.tabs[0].text_history = (0..100).map(|i| i.to_string()).collect();
        let restored = reopen(&saved).unwrap();
        assert_eq!(
            restored
                .active_tab()
                .snapshot
                .import_export
                .text_history()
                .len(),
            20
        );
        let prepared = recovery::prepare(&serde_json::to_string(&saved).unwrap()).unwrap();
        let (restored, cache) = recovery::restore_prepared_with_cache(&prepared).unwrap();
        let captured = capture(&restored, &restored.active_tab().snapshot, Some(&cache));
        assert_eq!(captured.tabs[0].text_history.len(), 20);
        assert_eq!(captured.tabs[0].text_history[0], "80");
    }

    // Frozen wire format, independent of the current serializer.
    const SAVED_SESSION: &str = include_str!("../tests/fixtures/session_v1_current.json");

    fn capture(tabs: &EditorTabs, live: &EditorTabSnapshot, previous: Option<&Session>) -> Session {
        Session::capture(tabs, live.into(), previous).unwrap()
    }

    fn reopen(session: &Session) -> Result<EditorTabs> {
        let json = serde_json::to_string(session)?;
        let native = Session::from_json(&json)?.restore()?;
        let prepared = recovery::prepare(&json)?;
        let browser =
            recovery::restore_prepared_json(&prepared).map_err(color_eyre::eyre::Error::msg)?;
        let capture_tabs =
            |tabs: &EditorTabs| Session::capture(tabs, (&tabs.active_tab().snapshot).into(), None);
        assert_eq!(
            serde_json::to_value(capture_tabs(&native)?)?,
            serde_json::to_value(capture_tabs(&browser)?)?
        );
        Ok(browser)
    }

    fn graph(source: &str) -> GraphState {
        let mut state = GraphState {
            graph: bloq_graph::lower_blog_ast_lenient(
                &bloq_graph::parse_blog_to_ast(source).unwrap(),
                [],
            )
            .unwrap(),
            ..default()
        };
        state.commit();
        state
    }

    #[test]
    fn reopen_keeps_live_drafts_and_linked_modules() {
        let mut tabs = EditorTabs::default();
        let parent = tabs.active;
        let program = crate::module_authoring::tests::composition();
        let parent_graph = &mut tabs.active_tab_mut().snapshot.graph_state;
        parent_graph.graph = program.flatten().unwrap();
        parent_graph.source_graph = Some(Arc::new(program.clone()));
        parent_graph.commit();
        let child = tabs.add_empty_tab(&EditorState::default());
        tabs.set_active(child);
        // The active tab's stashed snapshot deliberately stays empty.
        let mut live = EditorTabSnapshot::default();
        let leaf = program.extract_definition("Memory").unwrap();
        live.graph_state.graph = leaf.flatten().unwrap();
        live.graph_state.source_graph = Some(Arc::new(leaf.clone()));
        live.graph_state
            .graph
            .set_block_tag(IVec3::Z, "edited")
            .unwrap();
        live.graph_state.commit();
        live.import_export.definition_edit = Some(DefinitionEdit {
            parent,
            name: "Memory".into(),
            original: leaf.to_blog_text(),
        });
        live.import_export
            .set_bloq_buffer("# 未完成 draft\nmodule main {".into());
        live.camera_settings.focus = Vec3::new(9.0, 2.0, 3.0);
        live.camera_settings.radius = 25.0;
        live.editor_state.mode = EditorMode::Edit;
        live.editor_state.theme_preset = ThemePreset::GruvboxMaterial;
        live.editor_state.show_axis = false;
        live.editor_state
            .selected_elements
            .insert(GraphElement::Block(IVec3::Z));
        live.compile_ui.code_distance_input = "7".into();
        let saved = capture(&tabs, &live, None);
        let json = serde_json::to_value(&saved).unwrap();
        assert!(json["tabs"][0]["document"]["module_program"].is_string());
        assert!(json["tabs"][0]["document"].get("source_graph").is_none());
        let mut reopened = reopen(&saved).unwrap();
        let restored = &reopened.active_tab().snapshot;
        assert_eq!(reopened.active, child);
        let mut expected = serde_json::to_value(saved).unwrap();
        for (saved_tab, tab) in expected["tabs"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(&reopened.tabs)
        {
            saved_tab["text_conflict"] = json!(tab.snapshot.import_export.sync_conflict);
        }
        assert!(restored.import_export.sync_conflict);
        assert_eq!(
            serde_json::to_value(capture(&reopened, restored, None)).unwrap(),
            expected
        );
        assert_eq!(
            restored
                .graph_state
                .resolved_graph()
                .unwrap()
                .to_blog_text(),
            live.graph_state.resolved_graph().unwrap().to_blog_text()
        );
        assert_eq!(
            restored.graph_state.history.len(),
            live.graph_state.history.len()
        );
        assert_ne!(reopened.add_empty_tab(&EditorState::default()), child);
    }

    #[test]
    fn capture_tracks_edits_and_closed_tabs_without_rebuilding_unchanged_graphs() {
        let mut tabs = EditorTabs::default();
        let other = tabs.add_empty_tab(&EditorState::default());
        let mut live = EditorTabSnapshot::default();
        let saved = capture(&tabs, &live, None);
        assert_eq!(capture(&tabs, &live, Some(&saved)), saved);
        live.camera_settings.radius = 20.0;
        let next = capture(&tabs, &live, Some(&saved));
        assert!(Arc::ptr_eq(&saved.tabs[0].document, &next.tabs[0].document));
        live.graph_state = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n");
        live.import_export
            .set_bloq_buffer("unfinished draft".into());
        tabs.remove(other, &EditorState::default());
        let next = capture(&tabs, &live, Some(&next));
        let reopened = reopen(&next).unwrap();
        assert_eq!(reopened.tabs.len(), 1);
        assert_eq!(
            reopened
                .active_tab()
                .snapshot
                .graph_state
                .graph
                .block_count(),
            1
        );
        assert_eq!(
            reopened.active_tab().snapshot.import_export.bloq_buffer,
            "unfinished draft"
        );
    }

    #[test]
    fn saved_session_restores_and_resaves_without_losing_drafts() {
        let saved = Session::from_json(SAVED_SESSION).unwrap();
        let mut expected: Value = serde_json::from_str(SAVED_SESSION).unwrap();
        let tabs = saved.restore().unwrap();
        // Version 1 gains optional recovery fields; old document/text values
        // are unchanged, while mismatched drafts acquire an explicit conflict.
        for (saved_tab, tab) in expected["tabs"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(&tabs.tabs)
        {
            saved_tab["text_history"] = json!([]);
            saved_tab["text_conflict"] = json!(tab.snapshot.import_export.sync_conflict);
        }
        let snapshot = &tabs.active_tab().snapshot;
        let saved = Session::capture(&tabs, snapshot.into(), None).unwrap();
        for (tab, saved_tab) in expected["tabs"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(&saved.tabs)
        {
            tab["history"] = serde_json::to_value(&saved_tab.history).unwrap();
        }
        assert_eq!(
            serde_json::from_str::<Value>(&serde_json::to_string(&saved).unwrap()).unwrap(),
            expected
        );
        let reopened = reopen(&saved).unwrap();
        assert_eq!(
            serde_json::to_value(
                Session::capture(&reopened, (&reopened.active_tab().snapshot).into(), None)
                    .unwrap()
            )
            .unwrap(),
            serde_json::to_value(saved).unwrap()
        );
    }

    #[test]
    fn placement_kinds_preserve_parameters() {
        use bloq_graph::{Basis, PatchRotationKind, WalkingBoundaryKind, WalkingKind};

        let mut kinds = BlockKind::all_kinds().to_vec();
        for x in -1..=1 {
            for y in -1..=1 {
                let movement = IVec2::new(x, y);
                kinds.extend(WalkingBoundaryKind::ALL.into_iter().filter_map(|boundary| {
                    WalkingKind::new(boundary, movement)
                        .ok()
                        .map(BlockKind::Walking)
                }));
                kinds.extend([Basis::X, Basis::Z].into_iter().filter_map(|basis| {
                    PatchRotationKind::new(basis, movement)
                        .ok()
                        .map(BlockKind::PatchRotation)
                }));
            }
        }
        let mut json: Value = serde_json::from_str(SAVED_SESSION).unwrap();
        for kind in kinds {
            let block = Block::new(IVec3::ZERO, kind);
            json["tabs"][0]["view"]["block_kind"] = json!(block.to_string());
            let tabs = Session::from_json(&serde_json::to_string(&json).unwrap())
                .unwrap()
                .restore()
                .unwrap();
            let snapshot = &tabs.active_tab().snapshot;
            assert_eq!(snapshot.editor_state.block_kind, kind);
            let reopened = reopen(&capture(&tabs, snapshot, None)).unwrap();
            assert_eq!(reopened.active_tab().snapshot.editor_state.block_kind, kind);
        }
    }

    #[test]
    fn unsupported_versions_are_reported_before_decoding_the_payload() {
        let error =
            Session::from_json(r#"{"version":999,"tabs":"future representation"}"#).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("Unsupported editor session version 999")
        );
    }

    #[test]
    fn history_capture_reuses_entries_and_invalidates_in_place_promotion() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.graph_state.graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        live.graph_state.commit();
        let first = capture(&tabs, &live, None);
        live.graph_state.graph.add_block(Block::new(
            IVec3::X,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        live.graph_state.commit();
        let second = capture(&tabs, &live, Some(&first));
        assert!(Arc::ptr_eq(
            &first.tabs[0].history.as_ref().unwrap().entries[1],
            &second.tabs[0].history.as_ref().unwrap().entries[1]
        ));
        live.graph_state.undo();
        let undone = capture(&tabs, &live, Some(&second));
        for (a, b) in second.tabs[0]
            .history
            .as_ref()
            .unwrap()
            .entries
            .iter()
            .zip(&undone.tabs[0].history.as_ref().unwrap().entries)
        {
            assert!(Arc::ptr_eq(a, b));
        }
        let index = live.graph_state.current_index;
        live.graph_state
            .graph
            .set_block_tag(IVec3::ZERO, "validated")
            .unwrap();
        let graph = live.graph_state.graph.clone();
        let entry = &mut live.graph_state.history[index];
        entry.graph = graph;
        entry.invalidate_snapshot_cache();
        let promoted = capture(&tabs, &live, Some(&undone));
        assert!(!Arc::ptr_eq(
            &undone.tabs[0].document,
            &promoted.tabs[0].document
        ));
        assert!(!Arc::ptr_eq(
            &undone.tabs[0].history.as_ref().unwrap().entries[index],
            &promoted.tabs[0].history.as_ref().unwrap().entries[index]
        ));
        assert!(
            promoted.tabs[0].history.as_ref().unwrap().entries[index]
                .graph
                .contains("validated")
        );
    }

    #[test]
    fn reopen_preserves_undo_redo_cursor_and_new_edits_replace_redo_branch() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        for pos in [IVec3::ZERO, IVec3::X] {
            live.graph_state
                .graph
                .add_block(Block::new(pos, BlockKind::Cube(bloq_graph::CubeKind::ZXZ)));
            live.graph_state.commit();
        }
        let end = capture(&tabs, &live, None);
        live.graph_state.undo();
        let mid = capture(&tabs, &live, Some(&end));
        let mut reopened = reopen(&mid).unwrap();
        let state = &mut reopened.active_tab_mut().snapshot.graph_state;
        assert_eq!(state.graph.block_count(), 1);
        assert_eq!(state.history.len(), 3);
        state.redo();
        assert_eq!(state.graph.block_count(), 2);
        state.undo();
        state.graph.add_block(Block::new(
            IVec3::Y,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        state.commit();
        let branch = capture(&reopened, &reopened.active_tab().snapshot, Some(&mid));
        let mut reopened = reopen(&branch).unwrap();
        let state = &mut reopened.active_tab_mut().snapshot.graph_state;
        state.redo();
        assert!(state.graph.has_block_at(IVec3::Y));
        assert!(!state.graph.has_block_at(IVec3::X));
        state.undo();
        assert_eq!(state.graph.block_count(), 1);
    }

    #[test]
    fn redo_after_restart_restores_authored_module_hierarchy() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        let program = crate::module_authoring::tests::composition();
        live.graph_state.graph = program.flatten().unwrap();
        live.graph_state.source_graph = Some(Arc::new(program));
        live.graph_state.commit();
        let transaction = SavedDocument::capture(&live.graph_state).unwrap();
        live.graph_state.undo();
        let saved = capture(&tabs, &live, None);
        let mut reopened = reopen(&saved).unwrap();
        let state = &mut reopened.active_tab_mut().snapshot.graph_state;
        assert!(state.source_graph.is_none());
        state.redo();
        assert!(state.source_graph.as_ref().unwrap().has_module_structure());
        assert_eq!(SavedDocument::capture(state).unwrap(), transaction);
    }

    #[test]
    fn redo_after_restart_restores_hierarchy_and_captured_arm_transaction() {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.graph_state = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n");
        live.graph_state.source_graph = Some(Arc::new(live.graph_state.graph.clone()));
        let arm = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n1: ZXZ [0,0,1]\n0 -> +Z\n");
        live.graph_state.pending_branch_arm = Some(PendingBranchArm {
            name: "b0".into(),
            captured_true: false,
            arm: BranchArm::new(
                vec![arm.graph.get_block(IVec3::Z).unwrap().clone()],
                arm.graph.pipes().cloned().collect(),
            ),
        });
        live.graph_state.commit();
        let transaction = SavedDocument::capture(&live.graph_state).unwrap();
        live.graph_state.undo();
        let saved = capture(&tabs, &live, None);
        let mut reopened = reopen(&saved).unwrap();
        let state = &mut reopened.active_tab_mut().snapshot.graph_state;
        assert!(state.source_graph.is_none());
        assert!(state.pending_branch_arm.is_none());
        state.redo();
        assert_eq!(SavedDocument::capture(state).unwrap(), transaction);
    }

    #[test]
    fn captured_arms_and_incomplete_actions_survive_reopen() {
        let mut state = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n");
        let arm = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n1: ZXZ [0,0,1]\n0 -> +Z\n");
        state.pending_branch_arm = Some(PendingBranchArm {
            name: "b0".into(),
            captured_true: false,
            arm: BranchArm::new(
                vec![arm.graph.get_block(IVec3::Z).unwrap().clone()],
                arm.graph.pipes().cloned().collect(),
            ),
        });
        let saved = SavedDocument::capture(&state).unwrap();
        assert_eq!(
            SavedDocument::capture(&saved.restore().unwrap()).unwrap(),
            saved
        );
        let state = graph("BLOG 1.0\n0: ZXZ [0,0,0]\n1: XY [1,0,0]\n0 -> +X\nm0 = measure 0\n");
        let restored = SavedDocument::capture(&state).unwrap().restore().unwrap();
        assert_eq!(restored.graph.actions(), state.graph.actions());
        assert!(restored.graph.action_graph_error().is_some());
    }

    #[test]
    fn corrupt_sessions_are_rejected() {
        let saved =
            serde_json::to_value(capture(&EditorTabs::default(), &default(), None)).unwrap();
        let mutations: &[fn(&mut Value)] = &[
            |s| s["version"] = json!(SESSION_VERSION + 1),
            |s| s["tabs"][0]["history"]["current"] = json!(999),
            |s| s["tabs"][0]["history"]["entries"] = json!([]),
            |s| s["tabs"][0]["history"]["entries"][0]["graph"] = json!("broken BLOG"),
            |s| s["active"] = json!(999),
            |s| s["tabs"][0]["camera"]["radius"] = json!(-1),
            |s| s["tabs"][0]["document"]["graph"] = json!("broken BLOG"),
            |s| {
                s["tabs"][0]["document"]
                    .as_object_mut()
                    .unwrap()
                    .remove("false_arms");
            },
            |s| s["tabs"][0]["view"]["block_kind"] = json!("broken block kind"),
            |s| {
                s["tabs"][0]["view"]["block_kind"] =
                    json!("BLOG 1.0\n0: XZZ [0,0,0]\n1: ZXZ [1,0,0]\n")
            },
            |s| {
                let tab = s["tabs"][0].clone();
                s["tabs"].as_array_mut().unwrap().push(tab);
            },
        ];
        for mutate in mutations {
            let mut json = saved.clone();
            mutate(&mut json);
            assert!(
                Session::from_json(&serde_json::to_string(&json).unwrap())
                    .and_then(Session::restore)
                    .is_err()
            );
        }
    }

    #[test]
    fn gallery_documents_preserve_hierarchy_and_branch_display() {
        for item in bloq_graph::GalleryItem::iter() {
            let program = item.build();
            let mut state = GraphState {
                graph: program.flatten().unwrap(),
                source_graph: Some(Arc::new(program)),
                ..default()
            };
            let names: Vec<_> = state
                .graph
                .branch_definitions()
                .iter()
                .map(|b| b.name.clone())
                .collect();
            for (index, name) in names.into_iter().enumerate() {
                state
                    .graph
                    .set_shown_branch_arm(&name, index % 2 == 0)
                    .unwrap();
            }
            let saved = SavedDocument::capture(&state).unwrap();
            let restored = saved
                .restore()
                .unwrap_or_else(|error| panic!("{item}: {error:#}"));
            assert_eq!(SavedDocument::capture(&restored).unwrap(), saved, "{item}");
        }
    }
}
