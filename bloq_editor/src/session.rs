//! Durable editor data. Runtime assets, jobs, caches and undo stacks are rebuilt.
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

// Like Bloq IR, session layouts may change while the version stays at 1.
const SESSION_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Session {
    version: u32,
    active: EditorTabId,
    tabs: Vec<SavedTab>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SavedTab {
    id: EditorTabId,
    title: String,
    document: Arc<SavedDocument>,
    blog: Arc<str>,
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
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SavedDocument {
    graph: String,
    inputs: Vec<String>,
    false_arms: Vec<String>,
    // Keep the version-1 wire field while the source becomes a BlockGraph.
    #[serde(rename = "module_program")]
    source_graph: Option<String>,
    pending_arm: Option<SavedArm>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SavedArm {
    name: String,
    captured_true: bool,
    graph: String,
    owned: HashSet<IVec3>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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
            "Unsupported browser session version {version} (expected {SESSION_VERSION})"
        );
        Ok(())
    }

    fn capture(tabs: &EditorTabs, live: TabSource<'_>, previous: Option<&Self>) -> Result<Self> {
        let saved = tabs
            .tabs
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
                let document = match previous {
                    Some(saved) if saved.graph_revision == source.graph.revision => {
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
                    blog,
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
    fn restore(self) -> Result<EditorTabs> {
        Self::check_version(self.version)?;
        let mut restored = Vec::with_capacity(self.tabs.len());
        for tab in self.tabs {
            ensure!(
                tab.camera.focus.is_finite()
                    && tab.camera.radius > 0.0
                    && tab.camera.radius <= crate::systems::camera::MAX_CAMERA_RADIUS
                    && tab.camera.alpha.is_finite()
                    && (0.0..std::f32::consts::PI).contains(&tab.camera.beta),
                "Invalid saved camera"
            );
            let mut snapshot = EditorTabSnapshot {
                graph_state: tab.document.restore()?,
                camera_settings: tab.camera,
                ..default()
            };
            tab.view.restore(&mut snapshot.editor_state)?;
            snapshot
                .editor_state
                .sync_after_graph_edit(&mut snapshot.graph_state);
            snapshot.import_export.set_bloq_buffer(tab.blog.to_string());
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

impl SavedDocument {
    fn capture(state: &GraphState) -> Result<Self> {
        Ok(Self {
            graph: state.graph.to_blog_body_text(),
            inputs: state
                .graph
                .action_graph()
                .inputs()
                .map(str::to_owned)
                .collect(),
            false_arms: state
                .graph
                .branch_definitions()
                .iter()
                .filter(|branch| !branch.shown_true())
                .map(|branch| branch.name.clone())
                .collect(),
            source_graph: state.source_graph.as_ref().map(|p| p.to_blog_text()),
            pending_arm: state
                .pending_branch_arm
                .as_ref()
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
                                let block = state
                                    .graph
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
                ensure!(
                    saved.owned.iter().all(|pos| graph.has_block_at(*pos)),
                    "Invalid captured arm"
                );
                Ok(PendingBranchArm {
                    name: saved.name.clone(),
                    captured_true: saved.captured_true,
                    arm: BranchArm::new(
                        graph
                            .blocks()
                            .filter(|block| saved.owned.contains(&block.pos()))
                            .cloned()
                            .collect(),
                        graph.pipes().cloned().collect(),
                    ),
                })
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

    // Frozen wire format, independent of the current serializer.
    const SAVED_SESSION: &str = include_str!("../tests/fixtures/session_v1_current.json");

    fn capture(tabs: &EditorTabs, live: &EditorTabSnapshot, previous: Option<&Session>) -> Session {
        Session::capture(tabs, live.into(), previous).unwrap()
    }

    fn reopen(session: &Session) -> Result<EditorTabs> {
        Session::from_json(&serde_json::to_string(session)?)?.restore()
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
        assert_eq!(
            serde_json::to_value(capture(&reopened, restored, None)).unwrap(),
            serde_json::to_value(saved).unwrap()
        );
        assert_eq!(
            restored
                .graph_state
                .resolved_graph()
                .unwrap()
                .to_blog_text(),
            live.graph_state.resolved_graph().unwrap().to_blog_text()
        );
        assert_eq!(restored.graph_state.history.len(), 1);
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
        let expected: Value = serde_json::from_str(SAVED_SESSION).unwrap();
        let tabs = saved.restore().unwrap();
        let snapshot = &tabs.active_tab().snapshot;
        let saved = Session::capture(&tabs, snapshot.into(), None).unwrap();
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
                .starts_with("Unsupported browser session version 999")
        );
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
