//! Applies the queued [`UiIntent`]s. This is the one system with mutable access
//! to the full editor state, so it can carry out the actions the panel draw
//! systems could only enqueue.

use crate::components::{AxisHelper, CameraSettings, EditorCamera, GraphElement};
use crate::module_authoring::{
    ModuleEdit, edit_graph, module_elements, module_instance_at, translate_module,
};
use crate::resources::{
    BloqViewerState, BoxSelectionState, CompileUiState, EditorMode, EditorState, EditorTabId,
    EditorTabSnapshot, EditorTabs, GraphState, ImportExportState, LiveTabState, Notifications,
    PendingBranchArm, TargetState, ZxViewerState,
};
use crate::systems::camera::reset_camera_to_graph;
use crate::systems::input::{
    TranslationTarget, insert_graph_without_overlap, rotate_selected_elements,
    rotation_degrees_label, translate_graph, translate_selected_elements,
    translate_selected_elements_by,
};
use crate::systems::jobs::{
    EditorJobs, cancel_editor_compilation, replace_graph_with_cleanup,
    replace_graph_with_cleanup_and_selection, request_concurrent_ops, request_detslice,
    schedule_import_blog_file_job, schedule_parse_blog_job, schedule_stabilizers_job,
    schedule_validate_graph_job, start_viewer_compile,
};
use crate::theme::palette;
use crate::utils::set_plane_height;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use bloq_graph::{Action, BlockGraph, BranchArm, ModuleRotation};
use color_eyre::eyre::{self, ContextCompat, ensure};

use super::intents::{ElementEditIntent, UiIntent, UiIntentBuffer};

/// Tab-only resources bundled to stay within Bevy's 16-system-parameter limit.
#[derive(SystemParam)]
pub(crate) struct TabSwitchResources<'w> {
    box_selection: ResMut<'w, BoxSelectionState>,
    pending_tab_close: ResMut<'w, super::PendingTabClose>,
}

/// Host-facing state and channels: egui's clipboard and native quit handling.
/// Bundled for the same 16-parameter reason as above.
#[derive(SystemParam)]
pub(crate) struct HostChannels<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    #[cfg(not(target_arch = "wasm32"))]
    app_exit: MessageWriter<'w, AppExit>,
}

use crate::systems::jobs::schedule_compile_job;

/// Drains the [`UiIntentBuffer`] and applies each intent to the editor state.
pub(crate) fn apply_ui_intents_system(
    mut ui_intents: ResMut<UiIntentBuffer>,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    mut target_state: ResMut<TargetState>,
    mut notifications: ResMut<Notifications>,
    mut import_export: ResMut<ImportExportState>,
    mut tabs: ResMut<EditorTabs>,
    mut axis_query: Query<&mut Visibility, With<AxisHelper>>,
    mut clear_color: ResMut<ClearColor>,
    camera_query: Single<(&mut Transform, &mut CameraSettings), With<EditorCamera>>,
    mut jobs: ResMut<EditorJobs>,
    mut compile_ui: ResMut<CompileUiState>,
    mut circuit_viewer: ResMut<BloqViewerState>,
    mut zx_viewer: ResMut<ZxViewerState>,
    tab_switch: TabSwitchResources,
    host: HostChannels,
) {
    let (mut camera_transform, mut camera_settings) = camera_query.into_inner();
    let TabSwitchResources {
        mut box_selection,
        mut pending_tab_close,
    } = tab_switch;
    let HostChannels {
        mut contexts,
        #[cfg(not(target_arch = "wasm32"))]
        mut app_exit,
    } = host;

    let mut pending = ui_intents
        .drain()
        .collect::<std::collections::VecDeque<_>>();
    while let Some(intent) = pending.pop_front() {
        if graph_state.is_composed() && intent.changes_flat_geometry() {
            notifications.push_warn("Edit a definition in Modules, or open a flat copy");
            continue;
        }
        match intent {
            UiIntent::FinishTranslation {
                tab,
                revision,
                target,
                offset,
                compact,
            } => {
                if tab != tabs.active || revision != graph_state.revision {
                    notifications.push_warn("Document changed; drag again");
                    continue;
                }
                match target {
                    TranslationTarget::Selection(selected) => {
                        if graph_state.is_composed() {
                            continue;
                        }
                        if graph_state.selection_touches_pending_branch_cut(&selected) {
                            notifications
                                .push_warn("Captured arm's input boundary cannot move separately");
                            continue;
                        }
                        match translate_selected_elements_by(&graph_state.graph, &selected, offset)
                        {
                            Ok(moved) => {
                                replace_graph_with_cleanup_and_selection(
                                    &mut graph_state,
                                    &mut editor_state,
                                    &mut target_state,
                                    &mut compile_ui,
                                    &mut circuit_viewer,
                                    moved.graph,
                                    moved.selected_elements,
                                );
                            }
                            Err(error) => {
                                notifications.push_error(format!("Move rejected: {error:#}"))
                            }
                        }
                    }
                    TranslationTarget::Module(name) => {
                        let result = graph_state
                            .resolved_graph()
                            .and_then(|program| {
                                Ok(translate_module(&program, &name, offset, compact)?)
                            })
                            .and_then(|moved| {
                                let selected = module_elements(&moved.program, &name)?;
                                install_source_graph(
                                    &mut graph_state,
                                    &mut editor_state,
                                    &mut target_state,
                                    &mut compile_ui,
                                    &mut circuit_viewer,
                                    moved.program,
                                )?;
                                editor_state.replace_selection(selected);
                                import_export.module_ui.select_instance(name);
                                if moved.connections > 0 {
                                    notifications.push_info(format!(
                                        "Connected {} port pairs{}",
                                        moved.connections,
                                        if moved.compacted {
                                            " with compact seams"
                                        } else {
                                            " through cubes"
                                        }
                                    ));
                                }
                                Ok(())
                            });
                        report_module_edit(result, &mut import_export, &mut notifications);
                    }
                }
            }
            UiIntent::ShowBlogBuffer => editor_state.show_blog_buffer = true,
            intent @ (UiIntent::EditModule { .. }
            | UiIntent::ImportModuleTab { .. }
            | UiIntent::WrapAsModule(_)
            | UiIntent::NewModule(_)) => {
                let open = match &intent {
                    UiIntent::NewModule(name) => Some(name.clone()),
                    _ => None,
                };
                let fit = matches!(
                    &intent,
                    UiIntent::EditModule {
                        edit: ModuleEdit::AddInstance { .. }
                            | ModuleEdit::Transform { .. }
                            | ModuleEdit::Connect { .. }
                            | ModuleEdit::Disconnect { .. }
                            | ModuleEdit::RemoveInstance(_),
                        ..
                    }
                );
                let result = (|| {
                    let program = match intent {
                        UiIntent::EditModule { revision, edit } => {
                            ensure!(
                                revision == graph_state.revision,
                                "Composition changed; retry this edit"
                            );
                            edit_graph(&graph_state.resolved_graph()?, edit)?
                        }
                        UiIntent::ImportModuleTab { tab, name } => {
                            let source = tabs
                                .tabs
                                .iter()
                                .find(|source| source.id.get() == tab)
                                .wrap_err("Source tab is no longer open")?;
                            ensure!(source.id != tabs.active, "Choose another tab to import");
                            edit_graph(
                                &graph_state.resolved_graph()?,
                                ModuleEdit::Import {
                                    name,
                                    program: Box::new(
                                        source.snapshot.graph_state.resolved_graph()?,
                                    ),
                                },
                            )?
                        }
                        UiIntent::WrapAsModule(name) => {
                            let source = graph_state.resolved_graph()?;
                            let empty = BlockGraph::new().with_inferred_interface()?;
                            let program = edit_graph(
                                &empty,
                                ModuleEdit::Import {
                                    name: name.clone(),
                                    program: Box::new(source),
                                },
                            )?;
                            edit_graph(
                                &program,
                                ModuleEdit::AddInstance {
                                    definition: name,
                                    name: "stage".into(),
                                },
                            )?
                        }
                        UiIntent::NewModule(name) => edit_graph(
                            &graph_state.resolved_graph()?,
                            ModuleEdit::Import {
                                name,
                                program: Box::new(BlockGraph::new().with_inferred_interface()?),
                            },
                        )?,
                        _ => unreachable!("module mutation matched above"),
                    };
                    install_source_graph(
                        &mut graph_state,
                        &mut editor_state,
                        &mut target_state,
                        &mut compile_ui,
                        &mut circuit_viewer,
                        program,
                    )
                })();
                if result.is_ok() {
                    if fit {
                        reset_camera_to_graph(
                            &mut camera_transform,
                            &mut camera_settings,
                            &graph_state.graph,
                            editor_state.pipe_length,
                        );
                    }
                    if let Some(name) = open {
                        pending.push_front(UiIntent::OpenModule(name));
                    }
                }
                report_module_edit(result, &mut import_export, &mut notifications);
            }
            UiIntent::OpenModule(name) => {
                let result = graph_state
                    .resolved_graph()
                    .and_then(|program| Ok(program.extract_definition(&name)?));
                let program = match result {
                    Ok(program) => program,
                    Err(error) => {
                        report_module_edit(Err(error), &mut import_export, &mut notifications);
                        continue;
                    }
                };
                let origin = super::module_panel::DefinitionEdit {
                    parent: tabs.active,
                    name: name.clone(),
                    original: program.to_blog_text(),
                };
                switch_to_new_empty_tab(
                    &mut tabs,
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                tabs.active_tab_mut().set_title(format!("Edit {name}"));
                import_export.definition_edit = Some(origin);
                import_export.export_path = format!("{name}.blog");
                import_export.set_bloq_buffer(program.to_blog_text());
                let result = install_source_graph(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    program,
                );
                if !graph_state.is_composed() {
                    editor_state.set_mode(EditorMode::Edit, &mut graph_state);
                }
                reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
                report_module_edit(result, &mut import_export, &mut notifications);
            }
            UiIntent::ApplyModuleTab => {
                let Some(origin) = import_export.definition_edit.clone() else {
                    continue;
                };
                let result = (|| {
                    let source = graph_state.resolved_graph()?;
                    let parent = tabs
                        .tabs
                        .iter()
                        .find(|tab| tab.id == origin.parent)
                        .wrap_err(
                            "Parent tab is closed; add this tab to another composition instead",
                        )?;
                    let current = parent.snapshot.graph_state.resolved_graph()?;
                    ensure!(
                        current.extract_definition(&origin.name)?.to_blog_text() == origin.original,
                        "This definition changed in its parent. Reopen the latest definition before applying changes"
                    );
                    let program = edit_graph(
                        &current,
                        ModuleEdit::Replace {
                            name: origin.name.clone(),
                            program: Box::new(source),
                        },
                    )?;
                    let graph = program.flatten()?;
                    Ok((program, graph))
                })();
                match result {
                    Ok((program, graph)) => {
                        import_export
                            .definition_edit
                            .as_mut()
                            .expect("`origin` was cloned out of this same field")
                            .original = program
                            .extract_definition(&origin.name)
                            .expect("the replaced definition re-extracts from a validated program")
                            .to_blog_text();
                        let parent = tabs
                            .tabs
                            .iter_mut()
                            .find(|tab| tab.id == origin.parent)
                            .expect("the parent tab was located above");
                        crate::systems::jobs::install_graph_into_tab(
                            parent,
                            graph,
                            Some(program),
                            None,
                        );
                        parent.snapshot.import_export.module_ui.error = None;
                        parent
                            .snapshot
                            .editor_state
                            .set_mode(EditorMode::Module, &mut parent.snapshot.graph_state);
                        switch_to_existing_tab(
                            &mut tabs,
                            origin.parent,
                            &mut LiveTabState {
                                graph_state: &mut graph_state,
                                editor_state: &mut editor_state,
                                import_export: &mut import_export,
                                compile_ui: &mut compile_ui,
                                circuit_viewer: &mut circuit_viewer,
                                zx_viewer: &mut zx_viewer,
                                target_state: &mut target_state,
                                box_selection: &mut box_selection,
                                camera_settings: &mut camera_settings,
                            },
                        );
                        sync_camera_and_background(
                            &mut clear_color,
                            &mut camera_transform,
                            &editor_state,
                            &camera_settings,
                        );
                        notifications
                            .push_info(format!("Updated all instances of {}", origin.name));
                    }
                    Err(error) => {
                        report_module_edit(Err(error), &mut import_export, &mut notifications)
                    }
                }
            }
            intent @ (UiIntent::InspectModuleInstance(_) | UiIntent::InspectModuleElement(_)) => {
                if let Ok(program) = graph_state.resolved_graph() {
                    let name = match intent {
                        UiIntent::InspectModuleInstance(name) => Some(name),
                        UiIntent::InspectModuleElement(element) => {
                            module_instance_at(&program, element).ok().flatten()
                        }
                        _ => unreachable!("instance inspection matched above"),
                    };
                    if let Some(name) = name
                        && let Ok(elements) = module_elements(&program, &name)
                    {
                        import_export.module_ui.select_instance(name);
                        editor_state.replace_selection(elements);
                    }
                }
            }
            UiIntent::OpenFlatCopy => {
                let graph = graph_state.graph.clone();
                let title = format!("{} flat", tabs.active_tab().title);
                switch_to_new_empty_tab(
                    &mut tabs,
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                tabs.active_tab_mut().set_title(title);
                replace_graph_with_cleanup(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    graph,
                );
                editor_state.set_mode(EditorMode::Edit, &mut graph_state);
                reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
            }
            UiIntent::Error(message) => notifications.push_error(message),
            UiIntent::RerenderGraph => {
                graph_state.needs_rerender = true;
            }
            UiIntent::SetMode(mode) => {
                if mode == EditorMode::Edit && graph_state.is_composed() {
                    notifications.push_warn("Edit a definition in Modules, or open a flat copy");
                    continue;
                }
                let entering_circuit =
                    editor_state.mode != EditorMode::Bloq && mode == EditorMode::Bloq;
                editor_state.set_mode(mode, &mut graph_state);
                if entering_circuit {
                    start_viewer_compile(
                        &mut jobs,
                        tabs.active,
                        &graph_state,
                        &compile_ui,
                        &mut circuit_viewer,
                        &mut notifications,
                    );
                }
            }
            UiIntent::SelectPlacementTool(tool) => {
                editor_state.set_placement_tool(tool);
                editor_state.set_mode(EditorMode::Edit, &mut graph_state);
            }
            UiIntent::SelectPlacementBlock(kind) => {
                editor_state.select_block_kind(kind);
                editor_state.set_mode(EditorMode::Edit, &mut graph_state);
            }
            UiIntent::CompileForViewer => {
                start_viewer_compile(
                    &mut jobs,
                    tabs.active,
                    &graph_state,
                    &compile_ui,
                    &mut circuit_viewer,
                    &mut notifications,
                );
            }
            UiIntent::CancelCompilation(tab_id) => {
                cancel_editor_compilation(
                    &mut jobs,
                    tab_id,
                    &mut tabs,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    &mut notifications,
                );
            }
            UiIntent::SetCircuitMoment(moment) => {
                circuit_viewer.set_current_moment(moment);
            }
            UiIntent::SelectBloqNode(node_id) => {
                circuit_viewer.set_selected_node(node_id);
            }
            UiIntent::ToggleBloqNode(node_id) => {
                circuit_viewer.toggle_selected_node(node_id);
            }
            UiIntent::HoverBloqNode(node_id) => {
                circuit_viewer.set_hovered_node(node_id);
            }
            UiIntent::SetConcurrentOps(enabled) => {
                request_concurrent_ops(
                    &mut jobs,
                    tabs.active,
                    &mut circuit_viewer,
                    &mut notifications,
                    enabled,
                );
            }
            UiIntent::SetConcurrentLayer(layer) => {
                circuit_viewer.set_concurrent_layer(layer);
            }
            UiIntent::SetShowClassical(show) => {
                circuit_viewer.show_classical = show;
                // Hiding classical nodes must not leave a now-invisible node
                // selected with its attributes window floating.
                if !show {
                    circuit_viewer.deselect_hidden_nodes();
                }
            }
            UiIntent::SetSliceVisibility(visibility) => {
                request_detslice(
                    &mut jobs,
                    tabs.active,
                    &mut circuit_viewer,
                    &mut notifications,
                    visibility,
                );
            }
            UiIntent::PinViewerBranches(pins) => {
                crate::systems::jobs::request_viewer_branch_pins(
                    &mut jobs,
                    tabs.active,
                    graph_state.revision,
                    &mut circuit_viewer,
                    &mut notifications,
                    pins,
                );
            }
            UiIntent::CaptureBranchArm { captured_true } => {
                let selected_blocks = editor_state
                    .selected_elements()
                    .filter_map(|element| match element {
                        GraphElement::Block(position) => Some(position),
                        GraphElement::Pipe(..) => None,
                    })
                    .collect::<Vec<_>>();
                if selected_blocks.is_empty() {
                    notifications.push_warn("Select at least one arm block first");
                    continue;
                }
                let selected_pipes = editor_state
                    .selected_elements()
                    .filter_map(|element| match element {
                        GraphElement::Pipe(src, dst) => Some((src, dst)),
                        GraphElement::Block(_) => None,
                    })
                    .collect::<Vec<_>>();
                let mut candidate = graph_state.graph.clone();
                let arm = match candidate.take_branch_arm(selected_blocks, selected_pipes) {
                    Ok(arm) => arm,
                    Err(err) => {
                        notifications.push_error(format!("Could not capture branch arm: {err}"));
                        continue;
                    }
                };
                if let Some(pending) = graph_state.pending_branch_arm.clone() {
                    let (on_false, on_true) = if pending.captured_true {
                        (arm, pending.arm)
                    } else {
                        (pending.arm, arm)
                    };
                    if let Err(err) =
                        candidate.try_add_branch_region(pending.name.clone(), on_false, on_true)
                    {
                        notifications.push_error(format!("Could not create branch region: {err}"));
                        continue;
                    }
                    graph_state.pending_branch_arm = None;
                    replace_graph_with_cleanup(
                        &mut graph_state,
                        &mut editor_state,
                        &mut target_state,
                        &mut compile_ui,
                        &mut circuit_viewer,
                        candidate,
                    );
                    editor_state.branch_name = next_branch_name(&graph_state.graph);
                    notifications.push_info(format!(
                        "Created branch region '{}'; add its Resolve action",
                        pending.name
                    ));
                } else {
                    let name = editor_state.branch_name.trim().to_owned();
                    if name.is_empty() {
                        notifications.push_warn("Branch name cannot be empty");
                        continue;
                    }
                    if graph_state.graph.branch_by_name(&name).is_some() {
                        notifications.push_warn(format!("Branch '{name}' already exists"));
                        continue;
                    }
                    graph_state.pending_branch_arm = Some(PendingBranchArm {
                        name,
                        arm,
                        captured_true,
                    });
                    replace_graph_with_cleanup(
                        &mut graph_state,
                        &mut editor_state,
                        &mut target_state,
                        &mut compile_ui,
                        &mut circuit_viewer,
                        candidate,
                    );
                    notifications.push_info(format!(
                        "{} arm hidden; build and select the {} arm",
                        if captured_true { "True" } else { "False" },
                        if captured_true { "false" } else { "true" }
                    ));
                }
            }
            UiIntent::CancelBranchArm => {
                let Some(pending) = graph_state.pending_branch_arm.clone() else {
                    continue;
                };
                let mut candidate = graph_state.graph.clone();
                match candidate.restore_branch_arm(&pending.arm) {
                    Ok(()) => {
                        graph_state.pending_branch_arm = None;
                        replace_graph_with_cleanup(
                            &mut graph_state,
                            &mut editor_state,
                            &mut target_state,
                            &mut compile_ui,
                            &mut circuit_viewer,
                            candidate,
                        );
                        notifications.push_info("Cancelled branch arm capture");
                    }
                    Err(err) => notifications
                        .push_error(format!("Could not restore captured branch arm: {err}")),
                }
            }
            UiIntent::ShowBranchArm { name, show_true } => {
                let mut candidate = graph_state.graph.clone();
                match candidate.set_shown_branch_arm(&name, show_true) {
                    Ok(()) => {
                        replace_graph_with_cleanup(
                            &mut graph_state,
                            &mut editor_state,
                            &mut target_state,
                            &mut compile_ui,
                            &mut circuit_viewer,
                            candidate,
                        );
                        notifications.push_info(format!(
                            "Showing {} arm of {name}",
                            if show_true { "true" } else { "false" }
                        ));
                    }
                    Err(err) => notifications.push_error(format!("Could not show arm: {err}")),
                }
            }
            UiIntent::SetBackgroundColor(color) => {
                editor_state.bg_color = color;
                clear_color.0 = color;
            }
            UiIntent::SetAxisVisibility(show_axis) => {
                editor_state.show_axis = show_axis;
                let visibility = if show_axis {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                for mut axis_visibility in axis_query.iter_mut() {
                    *axis_visibility = visibility;
                }
            }
            UiIntent::SetViewCurrentLayerOnly(enabled) => {
                editor_state.view_current_layer_only = enabled;
                graph_state.needs_rerender = true;
                if editor_state.mode == EditorMode::View {
                    editor_state.clear_stabilizers();
                }
            }
            UiIntent::SetPortTagVisibility(enabled) => {
                editor_state.show_port_tags = enabled;
                graph_state.needs_rerender = true;
            }
            UiIntent::SetPlaneHeight(plane_height) => {
                set_plane_height(plane_height, &mut editor_state, &mut graph_state);
            }
            UiIntent::SetTheme(theme_preset) => {
                let selected_theme = palette(theme_preset);
                editor_state.theme_preset = theme_preset;
                let background = Color::srgba_u8(
                    selected_theme.bg_dark.r(),
                    selected_theme.bg_dark.g(),
                    selected_theme.bg_dark.b(),
                    selected_theme.bg_dark.a(),
                );
                editor_state.bg_color = background;
                clear_color.0 = background;
                graph_state.needs_rerender = true;
            }
            UiIntent::ResetCamera => {
                crate::systems::camera::reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
            }
            UiIntent::Undo | UiIntent::Redo => {
                let before = graph_state.current_index;
                if matches!(intent, UiIntent::Undo) {
                    graph_state.undo();
                } else {
                    graph_state.redo();
                }
                if graph_state.current_index != before {
                    // The draft belongs to the state that history just replaced.
                    *target_state = TargetState::default();
                    editor_state.clear_hover();
                    editor_state.sync_after_graph_edit(&mut graph_state);
                    if editor_state.mode == EditorMode::Module
                        && let Some(name) = import_export.module_ui.selected_instance()
                    {
                        pending.push_front(UiIntent::InspectModuleInstance(name.to_owned()));
                    }
                }
            }
            UiIntent::ClearGraph => {
                graph_state.pending_branch_arm = None;
                graph_state.source_graph = None;
                replace_graph_with_cleanup(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    BlockGraph::default(),
                );
                crate::systems::camera::reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
                notifications.push_info("Graph cleared");
            }
            UiIntent::NewTab => {
                switch_to_new_empty_tab(
                    &mut tabs,
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                clear_color.0 = editor_state.bg_color;
                crate::systems::camera::reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
                notifications.push_info(format!(
                    "Opened {}",
                    tabs.title(tabs.active).unwrap_or("new tab")
                ));
            }
            UiIntent::SelectTab(id) => {
                let id = EditorTabId::new(id);
                switch_to_existing_tab(
                    &mut tabs,
                    id,
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                sync_camera_and_background(
                    &mut clear_color,
                    &mut camera_transform,
                    &editor_state,
                    &camera_settings,
                );
            }
            close_intent @ (UiIntent::RequestCloseTab(id) | UiIntent::ConfirmCloseTab(id)) => {
                let needs_confirmation = matches!(close_intent, UiIntent::RequestCloseTab(_));
                let id = EditorTabId::new(id);
                if needs_confirmation && tab_has_work(id, &tabs, &graph_state, &import_export) {
                    if let Some(tab) = tabs.tabs.iter().find(|tab| tab.id == id) {
                        pending_tab_close.request = Some(super::tab_bar::PendingClose {
                            id,
                            title: tab.title.clone(),
                        });
                    }
                    continue;
                }
                close_tab(
                    &mut tabs,
                    id,
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                sync_camera_and_background(
                    &mut clear_color,
                    &mut camera_transform,
                    &editor_state,
                    &camera_settings,
                );
            }
            UiIntent::BeginRenameTab(id) => {
                let tab = EditorTabId::new(id);
                tabs.begin_rename(tab);
                if let Ok(ctx) = contexts.ctx_mut() {
                    ctx.memory_mut(|memory| {
                        memory.request_focus(super::tab_bar::rename_field_id(tab))
                    });
                }
            }
            UiIntent::FinishRenameTab(id) => tabs.finish_rename(EditorTabId::new(id)),
            UiIntent::CancelRenameTab(id) => tabs.cancel_rename(EditorTabId::new(id)),
            #[cfg(not(target_arch = "wasm32"))]
            UiIntent::Quit => {
                if tabs
                    .tabs
                    .iter()
                    .any(|tab| tab_has_work(tab.id, &tabs, &graph_state, &import_export))
                {
                    pending_tab_close.quit_requested = true;
                } else {
                    app_exit.write(AppExit::Success);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            UiIntent::ConfirmQuit => {
                app_exit.write(AppExit::Success);
            }
            UiIntent::RequestScreenshot => {
                editor_state.request_screenshot = true;
            }
            UiIntent::ShowShortcuts => {
                editor_state.show_help_window = true;
            }
            UiIntent::CopyGraphAsBlog => match graph_state.resolved_graph() {
                Ok(program) => {
                    pending.push_front(UiIntent::CopyToClipboard(program.to_blog_text()))
                }
                Err(error) => notifications.push_error(format!("Could not export BLOG: {error}")),
            },
            UiIntent::CopyToClipboard(text) => {
                if let Ok(ctx) = contexts.ctx_mut() {
                    ctx.copy_text(text);
                }
            }
            UiIntent::InsertGraph(inserted) => {
                // Inserting into the geometry canvas creates an independent flat copy.
                let inserted = if inserted.has_module_structure() {
                    match inserted.flatten() {
                        Ok(graph) => graph,
                        Err(err) => {
                            notifications.push_error(format!("Could not insert graph: {err}"));
                            continue;
                        }
                    }
                } else {
                    *inserted
                };
                match insert_graph_without_overlap(&graph_state.graph, &inserted) {
                    Ok((graph, selected_elements)) => {
                        replace_graph_with_cleanup_and_selection(
                            &mut graph_state,
                            &mut editor_state,
                            &mut target_state,
                            &mut compile_ui,
                            &mut circuit_viewer,
                            graph,
                            selected_elements,
                        );
                        crate::systems::camera::reset_camera_to_graph(
                            &mut camera_transform,
                            &mut camera_settings,
                            &graph_state.graph,
                            editor_state.pipe_length,
                        );
                        notifications.push_info("Inserted graph");
                    }
                    Err(err) => {
                        notifications.push_error(format!("Could not insert graph: {err:#}"))
                    }
                }
            }
            UiIntent::LoadGallery(entry) => {
                tabs.active_tab_mut().set_title(entry.to_string());
                let program = entry.build();
                let graph = program.flatten().expect("gallery graph has valid geometry");
                import_export.set_bloq_buffer(program.to_blog_text());
                graph_state.pending_branch_arm = None;
                graph_state.source_graph = Some(std::sync::Arc::new(program));
                replace_graph_with_cleanup(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    graph,
                );
                crate::systems::camera::reset_camera_to_graph(
                    &mut camera_transform,
                    &mut camera_settings,
                    &graph_state.graph,
                    editor_state.pipe_length,
                );
                notifications.push_info(format!("Loaded {entry} example"));
            }
            UiIntent::FixShadowedFaces => {
                let next_graph = graph_state.graph.fix_shadowed_faces();
                replace_graph_with_cleanup(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    next_graph,
                );
                notifications.push_info("Fix shadowed face bases");
            }
            UiIntent::FillPorts => {
                apply_fill_ports(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    &mut notifications,
                );
            }
            UiIntent::FlipXZBasis => {
                let next_graph = match graph_state.graph.flip_xz_basis_lenient() {
                    Ok(graph) => graph,
                    Err(err) => {
                        notifications.push_error(format!("Could not flip X/Z basis: {err}"));
                        continue;
                    }
                };
                {
                    let graph_state = &mut *graph_state;
                    if let Some(pending) = &mut graph_state.pending_branch_arm {
                        pending.arm = pending.arm.flip_xz_basis(&graph_state.graph);
                    }
                }
                replace_graph_with_cleanup(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    next_graph,
                );
                notifications.push_info("Flipped X/Z basis");
            }
            UiIntent::RandomlyResolveSelectives => {
                apply_randomly_resolve_selectives(
                    &mut graph_state,
                    &mut editor_state,
                    &mut target_state,
                    &mut compile_ui,
                    &mut circuit_viewer,
                    &mut notifications,
                );
            }
            UiIntent::ValidateGraph => {
                schedule_validate_graph_job(
                    &mut jobs,
                    tabs.active,
                    &graph_state,
                    &mut compile_ui,
                    &mut notifications,
                );
            }
            UiIntent::ToggleStabilizers {
                layer_only,
                plane_height,
            } => {
                if editor_state.toggle_stabilizers() {
                    graph_state.needs_rerender = true;
                } else {
                    schedule_stabilizers_job(
                        &mut jobs,
                        tabs.active,
                        &graph_state,
                        layer_only,
                        plane_height,
                        &mut notifications,
                    );
                }
            }
            UiIntent::CompileGraph(request) => {
                schedule_compile_job(
                    &mut jobs,
                    tabs.active,
                    &graph_state,
                    &mut compile_ui,
                    &mut notifications,
                    request,
                );
            }
            UiIntent::ExportBlog { path } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let export_path = std::path::Path::new(&path);
                    if let Some(parent) = export_path.parent()
                        && !parent.as_os_str().is_empty()
                        && let Err(err) = std::fs::create_dir_all(parent)
                    {
                        notifications.push_error(format!(
                            "Export failed: Could not create directories ({})",
                            err
                        ));
                        continue;
                    }

                    let contents = match graph_state
                        .resolved_graph()
                        .map(|program| program.to_blog_text())
                    {
                        Ok(blog) => blog,
                        Err(error) => {
                            notifications.push_error(format!("Export failed: {error}"));
                            continue;
                        }
                    };
                    match std::fs::write(&path, contents) {
                        Ok(_) => notifications.push_info(format!("Saved graph to {}", path)),
                        Err(err) => notifications
                            .push_error(format!("Export failed: Could not write file ({})", err)),
                    }
                }

                #[cfg(target_arch = "wasm32")]
                {
                    let file_name = std::path::Path::new(&path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .filter(|name| !name.is_empty())
                        .unwrap_or("bloq_graph.blog");
                    let contents = match graph_state
                        .resolved_graph()
                        .map(|program| program.to_blog_text())
                    {
                        Ok(blog) => blog,
                        Err(error) => {
                            notifications.push_error(format!("Export failed: {error}"));
                            continue;
                        }
                    };
                    match crate::utils::save_download(file_name, contents.as_bytes()) {
                        Ok(Some(location)) => {
                            notifications.push_info(format!("Downloaded graph as {location}"));
                        }
                        Ok(None) => {}
                        Err(err) => {
                            notifications.push_error_report("Failed to export BLOG", &err);
                        }
                    }
                }
            }
            UiIntent::ImportBlogFromFile => {
                schedule_import_blog_file_job(&mut jobs, &mut notifications);
            }
            UiIntent::LoadBlogFile { title, buffer } => {
                if jobs.parse_blog_running() {
                    notifications.push_warn("BLOG parse is already running");
                    continue;
                }
                let tab_id = prepare_tab_for_import(
                    &mut tabs,
                    title.clone(),
                    &mut LiveTabState {
                        graph_state: &mut graph_state,
                        editor_state: &mut editor_state,
                        import_export: &mut import_export,
                        compile_ui: &mut compile_ui,
                        circuit_viewer: &mut circuit_viewer,
                        zx_viewer: &mut zx_viewer,
                        target_state: &mut target_state,
                        box_selection: &mut box_selection,
                        camera_settings: &mut camera_settings,
                    },
                );
                sync_camera_and_background(
                    &mut clear_color,
                    &mut camera_transform,
                    &editor_state,
                    &camera_settings,
                );
                schedule_parse_blog_job(
                    &mut jobs,
                    tab_id,
                    graph_state.revision,
                    title,
                    &buffer,
                    &mut notifications,
                );
                import_export.set_bloq_buffer(buffer);
            }
            UiIntent::LoadBlogFromBuffer => {
                if jobs.parse_blog_running() {
                    notifications.push_warn("BLOG parse is already running");
                    continue;
                }
                let buffer = import_export.bloq_buffer.clone();
                let tab_id = if import_export.definition_edit.is_some() {
                    tabs.active
                } else {
                    prepare_tab_for_import(
                        &mut tabs,
                        None,
                        &mut LiveTabState {
                            graph_state: &mut graph_state,
                            editor_state: &mut editor_state,
                            import_export: &mut import_export,
                            compile_ui: &mut compile_ui,
                            circuit_viewer: &mut circuit_viewer,
                            zx_viewer: &mut zx_viewer,
                            target_state: &mut target_state,
                            box_selection: &mut box_selection,
                            camera_settings: &mut camera_settings,
                        },
                    )
                };
                sync_camera_and_background(
                    &mut clear_color,
                    &mut camera_transform,
                    &editor_state,
                    &camera_settings,
                );
                schedule_parse_blog_job(
                    &mut jobs,
                    tab_id,
                    graph_state.revision,
                    None,
                    &buffer,
                    &mut notifications,
                );
                import_export.set_bloq_buffer(buffer);
            }
            UiIntent::StoreGraphToBuffer => match graph_state
                .resolved_graph()
                .map(|program| program.to_blog_text())
            {
                Ok(blog) => {
                    import_export.set_bloq_buffer(blog);
                    notifications.push_info("Stored graph to buffer");
                }
                Err(error) => notifications.push_error(format!("Could not store BLOG: {error}")),
            },
            UiIntent::SaveSvg {
                file_name,
                contents,
            } => match crate::utils::save_download(&file_name, contents.as_bytes()) {
                Ok(Some(location)) => {
                    notifications.push_info(format!("Exported SVG to {location}"))
                }
                // The user canceled the native save dialog — nothing to report.
                Ok(None) => {}
                Err(err) => notifications.push_error_report("Failed to export SVG", &err),
            },
            UiIntent::ApplyElementEdit(edit) => {
                apply_element_edit(
                    &mut editor_state,
                    &mut graph_state,
                    &mut notifications,
                    edit,
                );
            }
            UiIntent::CloseElementEdit => {
                close_element_editor(&mut editor_state, &mut target_state);
            }
            UiIntent::AddAction(action) => {
                let summary = action.to_string();
                let mut actions = graph_state.graph.actions();
                actions.push(action);
                apply_action_edit(
                    &mut graph_state,
                    &mut editor_state,
                    actions,
                    format!("Added `{summary}`"),
                    &mut notifications,
                );
            }
            UiIntent::ReplaceAction {
                index,
                expected,
                action,
            } => {
                let mut actions = graph_state.graph.actions();
                if actions != expected || index >= actions.len() {
                    notifications
                        .push_warn("Action changed while its draft was open; edit it again");
                    continue;
                }
                let summary = action.to_string();
                actions[index] = action;
                apply_action_edit(
                    &mut graph_state,
                    &mut editor_state,
                    actions,
                    format!("Updated `{summary}`"),
                    &mut notifications,
                );
            }
            UiIntent::RemoveAction(index) => {
                let mut actions = graph_state.graph.actions();
                if index >= actions.len() {
                    continue;
                }
                let summary = actions.remove(index).to_string();
                apply_action_edit(
                    &mut graph_state,
                    &mut editor_state,
                    actions,
                    format!("Removed `{summary}`"),
                    &mut notifications,
                );
            }
            UiIntent::TranslateGraph { axis, step } => {
                if selected_pending_branch_cut(&graph_state, &editor_state) {
                    notifications.push_warn("Captured arm's input boundary cannot move separately; deselect to transform the whole graph");
                    continue;
                }
                if editor_state.selection_count() > 0 {
                    match translate_selected_elements(
                        &graph_state.graph,
                        editor_state.selected_element_set(),
                        axis,
                        step,
                    ) {
                        Ok(result) => {
                            let delta = axis.to_ivec3() * step;
                            replace_graph_with_cleanup_and_selection(
                                &mut graph_state,
                                &mut editor_state,
                                &mut target_state,
                                &mut compile_ui,
                                &mut circuit_viewer,
                                result.graph,
                                result.selected_elements,
                            );
                            if result.dropped_action_count == 0 {
                                notifications
                                    .push_info(format!("Translated selection by {delta}."));
                            } else {
                                notifications.push_warn(format!(
                                    "Translated selection by {delta} and discarded {} action(s).",
                                    result.dropped_action_count
                                ));
                            }
                        }
                        Err(err) => notifications.push_warn(format!("{err:#}")),
                    }
                    continue;
                }

                let delta = axis.to_ivec3() * step;
                let transformed =
                    translate_graph(&graph_state.graph, axis, step).and_then(|graph| {
                        Ok((
                            graph,
                            transform_pending_arm(&graph_state, |arm| arm.try_with_shift(delta))?,
                        ))
                    });
                match transformed {
                    Ok((next_graph, pending)) => {
                        graph_state.pending_branch_arm = pending;
                        replace_graph_with_cleanup(
                            &mut graph_state,
                            &mut editor_state,
                            &mut target_state,
                            &mut compile_ui,
                            &mut circuit_viewer,
                            next_graph,
                        );
                        notifications.push_info(format!(
                            "Translated the whole graph by {delta}. Actions were preserved."
                        ));
                    }
                    Err(err) => notifications.push_warn(format!("{err:#}")),
                }
            }
            UiIntent::RotateGraph {
                axis,
                quarter_turns,
            } => {
                if selected_pending_branch_cut(&graph_state, &editor_state) {
                    notifications.push_warn("Captured arm's input boundary cannot move separately; deselect to transform the whole graph");
                    continue;
                }
                if editor_state.selection_count() > 0 {
                    match rotate_selected_elements(
                        &graph_state.graph,
                        editor_state.selected_element_set(),
                        axis,
                        quarter_turns,
                    ) {
                        Ok(result) => {
                            let axis_label = format!("{axis} axis");
                            replace_graph_with_cleanup_and_selection(
                                &mut graph_state,
                                &mut editor_state,
                                &mut target_state,
                                &mut compile_ui,
                                &mut circuit_viewer,
                                result.graph,
                                result.selected_elements,
                            );
                            if result.dropped_action_count == 0 {
                                notifications.push_info(format!(
                                    "Rotated selection {} around {}.",
                                    rotation_degrees_label(quarter_turns),
                                    axis_label
                                ));
                            } else {
                                notifications.push_warn(format!(
                                    "Rotated selection {} around {} and discarded {} action(s).",
                                    rotation_degrees_label(quarter_turns),
                                    axis_label,
                                    result.dropped_action_count
                                ));
                            }
                        }
                        Err(err) => notifications.push_warn(format!("{err:#}")),
                    }
                    continue;
                }

                match graph_state
                    .graph
                    .rotate_about_origin_lenient(axis, quarter_turns)
                    .and_then(|graph| {
                        Ok((
                            graph,
                            transform_pending_arm(&graph_state, |arm| {
                                arm.try_with_orientation(
                                    ModuleRotation::new(axis, quarter_turns).orientation(),
                                )
                            })?,
                        ))
                    }) {
                    Ok((next_graph, pending)) => {
                        graph_state.pending_branch_arm = pending;
                        let dropped_action_count = graph_state.graph.actions().len();
                        let axis_label = format!("{axis} axis");
                        replace_graph_with_cleanup(
                            &mut graph_state,
                            &mut editor_state,
                            &mut target_state,
                            &mut compile_ui,
                            &mut circuit_viewer,
                            next_graph,
                        );
                        if dropped_action_count == 0 {
                            notifications.push_info(format!(
                                "Rotated the whole graph {} around {}.",
                                rotation_degrees_label(quarter_turns),
                                axis_label
                            ));
                        } else {
                            notifications.push_warn(format!(
                                "Rotated the whole graph {} around {} and discarded {} action(s).",
                                rotation_degrees_label(quarter_turns),
                                axis_label,
                                dropped_action_count
                            ));
                        }
                    }
                    Err(err) => notifications.push_warn(err.to_string()),
                }
            }
        }
    }
}

fn install_source_graph(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    program: BlockGraph,
) -> eyre::Result<()> {
    let graph = program.flatten()?;
    graph_state.source_graph = Some(std::sync::Arc::new(program));
    replace_graph_with_cleanup(
        graph_state,
        editor_state,
        target_state,
        compile_ui,
        circuit_viewer,
        graph,
    );
    editor_state.set_mode(EditorMode::Module, graph_state);
    Ok(())
}

fn report_module_edit(
    result: eyre::Result<()>,
    files: &mut ImportExportState,
    notifications: &mut Notifications,
) {
    match result {
        Ok(()) => files.module_ui.error = None,
        Err(error) => {
            let message = format!("{error:#}");
            files.module_ui.error = Some(message.clone());
            notifications.push_error(message);
        }
    }
}

fn next_branch_name(graph: &BlockGraph) -> String {
    (0..)
        .map(|index| format!("b{index}"))
        .find(|name| graph.branch_by_name(name).is_none())
        .expect("an unbounded counter always escapes a finite branch set")
}

fn apply_randomly_resolve_selectives(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if graph_state
        .graph
        .actions()
        .iter()
        .any(|action| matches!(action, bloq_graph::Action::Branch { .. }))
    {
        notifications.push_warn(
            "Project structural branches before randomly resolving selectives; graph unchanged",
        );
        return;
    }

    let (next_graph, replacements) = match graph_state
        .graph
        .randomly_resolve_selectives(rand::random())
    {
        Ok(result) => result,
        Err(err) => {
            notifications.push_error(format!("Could not resolve selectives: {err}"));
            return;
        }
    };
    let resolved_count = replacements.len();
    replace_graph_with_cleanup(
        graph_state,
        editor_state,
        target_state,
        compile_ui,
        circuit_viewer,
        next_graph,
    );
    if resolved_count == 0 {
        notifications.push_info("No selectives to resolve");
    } else {
        notifications.push_info(format!("Randomly resolved {resolved_count} selectives"));
    }
}

/// Applies the "fill ports" action: advances the cached variant cycle if one is
/// active for the current revision, otherwise computes fresh port-filling
/// variants and installs the first.
pub(crate) fn apply_fill_ports(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if graph_state.is_composed() {
        notifications.push_warn("Edit a definition in Modules, or open a flat copy");
        return;
    }
    if let Some(mut cycle) = editor_state
        .fill_ports_cycle
        .take_if(|cycle| cycle.active_revision == graph_state.revision)
    {
        if cycle.variants.is_empty() {
            editor_state.clear_fill_ports_cycle();
            notifications.push_error("Fill Ports cycle is empty");
            return;
        }

        let next_index = (cycle.current_index + 1) % cycle.variants.len();
        let total_variants = cycle.variants.len();
        let next_graph = cycle.variants[next_index].clone();
        replace_graph_with_cleanup(
            graph_state,
            editor_state,
            target_state,
            compile_ui,
            circuit_viewer,
            next_graph,
        );
        cycle.current_index = next_index;
        cycle.active_revision = graph_state.revision;
        editor_state.fill_ports_cycle = Some(cycle);
        notifications.push_info(format!(
            "Showing fill ports variant {}/{}",
            next_index + 1,
            total_variants
        ));
        return;
    }

    if !graph_state.graph.is_open() {
        notifications.push_warn("Graph has no open ports to fill");
        return;
    }

    match graph_state.graph.fill_ports_auto() {
        Ok(filled_variants) => {
            if filled_variants.is_empty() {
                notifications.push_error("Fill Ports produced no closed variants");
                return;
            }

            let variants = filled_variants
                .into_iter()
                .map(|(graph, _)| graph)
                .collect::<Vec<_>>();
            let total_variants = variants.len();
            let first_graph = variants[0].clone();
            replace_graph_with_cleanup(
                graph_state,
                editor_state,
                target_state,
                compile_ui,
                circuit_viewer,
                first_graph,
            );

            if total_variants > 1 {
                editor_state.fill_ports_cycle = Some(crate::resources::FillPortsCycleState {
                    variants,
                    current_index: 0,
                    active_revision: graph_state.revision,
                });
                notifications.push_info(format!(
                    "Filled open ports with variant 1/{}. Click Fill Ports again to cycle.",
                    total_variants
                ));
            } else {
                notifications.push_info("Filled open ports");
            }
        }
        Err(err) => {
            notifications.push_error(format!("Failed to fill open ports: {err}"));
        }
    }
}

fn transform_pending_arm(
    state: &GraphState,
    transform: impl FnOnce(&BranchArm) -> Result<BranchArm, bloq_graph::BlockGraphError>,
) -> Result<Option<PendingBranchArm>, bloq_graph::BlockGraphError> {
    state
        .pending_branch_arm
        .as_ref()
        .map(|pending| {
            Ok(PendingBranchArm {
                arm: transform(&pending.arm)?,
                ..pending.clone()
            })
        })
        .transpose()
}

fn selected_pending_branch_cut(state: &GraphState, editor: &EditorState) -> bool {
    state.selection_touches_pending_branch_cut(editor.selected_element_set())
}

/// Commits actions without physical analysis; validation adds derived metadata.
fn apply_action_edit(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    actions: Vec<Action>,
    success: String,
    notifications: &mut Notifications,
) {
    let mut graph = graph_state.graph.clone();
    if let Err(err) = graph.set_actions_lenient(actions) {
        notifications.push_error(format!("Action rejected: {err}"));
        return;
    }
    let warning = match graph.action_graph_error() {
        Some(err) if err.is_incomplete_action_program() => Some(err.to_string()),
        Some(err) => {
            notifications.push_error(format!("Action rejected: {err}"));
            return;
        }
        None => None,
    };

    graph_state.graph = graph;
    graph_state.commit_with_rerender(false);
    editor_state.sync_after_graph_edit(graph_state);
    if let Some(warning) = warning {
        notifications.push_warn(format!("{success}, but: {warning}"));
    } else {
        notifications.push_info(success);
    }
}

fn apply_element_edit(
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    notifications: &mut Notifications,
    edit: ElementEditIntent,
) {
    if !edit.tag.is_empty() && !bloq_graph::is_valid_tag(&edit.tag) {
        notifications.push_error(format!(
            "Failed to edit element: invalid tag `{}`",
            edit.tag
        ));
        return;
    }
    let mut changed = false;
    let mut needs_rerender = false;
    match edit.target {
        GraphElement::Block(pos) => {
            if let Some(block) = graph_state.graph.get_block(pos) {
                let kind_changed = block.kind() != edit.block_kind;
                let tag_changed = block.tag().unwrap_or("") != edit.tag;
                let current_color = block.port_color().map(|color| [color.r, color.g, color.b]);
                let current_role = block.port_role();
                let desired_color = edit.block_kind.is_port().then(|| {
                    edit.block_color
                        .expect("Port edit intents carry a valid RGB color")
                });
                let desired_role = edit.block_kind.is_port().then_some(edit.port_role);
                let color_changed = current_color != desired_color;
                let role_changed = current_role != desired_role;
                if let Err(error) = graph_state.graph.set_block_kind(pos, edit.block_kind) {
                    notifications.push_error(format!("Failed to edit block kind: {error}"));
                    return;
                }
                if let Err(error) = graph_state.graph.set_block_tag(pos, edit.tag.clone()) {
                    notifications.push_error(format!("Failed to edit block tag: {error}"));
                    return;
                }
                if let Some(color) = desired_color
                    && let Err(error) = graph_state.graph.set_port_color(pos, color)
                {
                    notifications.push_error(format!("Failed to edit Port color: {error}"));
                    return;
                }
                if let Some(role) = desired_role
                    && let Err(error) = graph_state.graph.set_port_role(pos, role)
                {
                    notifications.push_error(format!("Failed to edit Port role: {error}"));
                    return;
                }
                notifications.push_info(format!(
                    "Edited block at {} with kind {}, tag '{}'",
                    pos, edit.block_kind, edit.tag
                ));
                changed = kind_changed || color_changed || role_changed || tag_changed;
                needs_rerender = kind_changed
                    || color_changed
                    || (tag_changed && editor_state.show_port_tags && edit.block_kind.is_port());
            } else {
                notifications.push_error(format!("Block not found at {}", pos));
            }
        }
        GraphElement::Pipe(u, v) => {
            if let Some(pipe) = graph_state.graph.get_pipe(u, v) {
                let hadamard_changed = pipe.is_hadamard() != edit.pipe_hadamard;
                let tag_changed = pipe.tag().unwrap_or("") != edit.tag;
                if let Err(error) = graph_state.graph.set_pipe_tag(u, v, edit.tag.clone()) {
                    notifications.push_error(format!("Failed to edit pipe tag: {error}"));
                    return;
                }
                if let Err(error) = graph_state
                    .graph
                    .set_pipe_hadamard(u, v, edit.pipe_hadamard)
                {
                    notifications.push_error(format!("Failed to edit pipe Hadamard: {error}"));
                    return;
                }
                notifications.push_info(format!(
                    "Edited pipe between {} and {} with tag '{}'",
                    u, v, edit.tag
                ));
                changed = hadamard_changed || tag_changed;
                needs_rerender = hadamard_changed;
            } else {
                notifications.push_error(format!("Pipe not found between {} and {}", u, v));
            }
        }
    }
    if changed {
        graph_state.commit_with_rerender(needs_rerender);
        editor_state.sync_after_graph_edit(graph_state);
    }
}

fn close_element_editor(editor_state: &mut EditorState, target_state: &mut TargetState) {
    target_state.open_window = false;
    target_state.target = None;
    target_state.tag_buffer.clear();
    editor_state.clear_selection();
}

/// Restores the clear color and camera pose after a tab switch.
fn sync_camera_and_background(
    clear_color: &mut ClearColor,
    camera_transform: &mut Transform,
    editor_state: &EditorState,
    camera_settings: &CameraSettings,
) {
    clear_color.0 = editor_state.bg_color;
    crate::systems::camera::apply_camera_setting(camera_transform, camera_settings);
}

fn save_active_tab(tabs: &mut EditorTabs, live: &LiveTabState<'_>) {
    tabs.save_active(EditorTabSnapshot::capture(live));
}

fn switch_to_new_empty_tab(tabs: &mut EditorTabs, live: &mut LiveTabState<'_>) {
    save_active_tab(tabs, live);
    let id = tabs.add_empty_tab(live.editor_state);
    tabs.set_active(id);
    tabs.restore_active(live);
}

fn switch_to_existing_tab(tabs: &mut EditorTabs, id: EditorTabId, live: &mut LiveTabState<'_>) {
    if tabs.active == id {
        return;
    }
    save_active_tab(tabs, live);
    if tabs.set_active(id) {
        tabs.restore_active(live);
    }
}

fn close_tab(tabs: &mut EditorTabs, id: EditorTabId, live: &mut LiveTabState<'_>) {
    save_active_tab(tabs, live);
    tabs.remove(id, live.editor_state);
    tabs.restore_active(live);
}

fn prepare_tab_for_import(
    tabs: &mut EditorTabs,
    title: Option<String>,
    live: &mut LiveTabState<'_>,
) -> EditorTabId {
    if tab_has_work(tabs.active, tabs, live.graph_state, live.import_export) {
        save_active_tab(tabs, live);
        let id = tabs.add_empty_tab(live.editor_state);
        tabs.set_active(id);
        tabs.restore_active(live);
    }
    if let Some(title) = title {
        let tab = tabs.active_tab_mut();
        let title = title.trim();
        tab.set_title(if title.is_empty() {
            "Untitled".to_string()
        } else {
            title.to_string()
        });
    }
    tabs.active
}

fn tab_has_work(
    id: EditorTabId,
    tabs: &EditorTabs,
    graph_state: &GraphState,
    import_export: &ImportExportState,
) -> bool {
    let Some(tab) = tabs.tabs.iter().find(|tab| tab.id == id) else {
        return false;
    };
    let (graph_state, import_export) = if id == tabs.active {
        (graph_state, import_export)
    } else {
        (&tab.snapshot.graph_state, &tab.snapshot.import_export)
    };
    !graph_state.graph.is_empty()
        || graph_state.source_graph.is_some()
        || graph_state.history.len() > 1
        || !import_export.bloq_buffer.is_empty()
}

#[cfg(test)]
mod tests {
    use super::{
        apply_action_edit, apply_element_edit, apply_randomly_resolve_selectives,
        close_element_editor, prepare_tab_for_import, tab_has_work,
    };
    use crate::components::{CameraSettings, GraphElement};
    use crate::resources::{
        BloqViewerState, BoxSelectionState, CompileUiState, EditorState, EditorTabs, GraphState,
        ImportExportState, LiveTabState, Notifications, TargetState, ZxViewerState,
    };
    use crate::systems::ui::edit_element::DEFAULT_PORT_RGB;
    use crate::systems::ui::intents::ElementEditIntent;
    use bloq_graph::{
        Action, Basis, Block, BlockKind, BranchArm, CubeKind, Direction, Expr, GalleryItem,
        MeasureTarget, Pipe, SelectiveKind,
    };
    use glam::IVec3;

    fn intent_app() -> bevy::prelude::App {
        use bevy::prelude::*;

        let mut app = App::new();
        app.init_resource::<super::UiIntentBuffer>()
            .init_resource::<EditorState>()
            .init_resource::<GraphState>()
            .init_resource::<TargetState>()
            .init_resource::<Notifications>()
            .init_resource::<ImportExportState>()
            .init_resource::<EditorTabs>()
            .init_resource::<ClearColor>()
            .init_resource::<super::EditorJobs>()
            .init_resource::<CompileUiState>()
            .init_resource::<BloqViewerState>()
            .init_resource::<ZxViewerState>()
            .init_resource::<BoxSelectionState>()
            .init_resource::<crate::systems::ui::PendingTabClose>()
            .init_resource::<bevy_egui::EguiUserTextures>()
            .add_message::<bevy::app::AppExit>()
            .add_systems(Update, super::apply_ui_intents_system);
        app.world_mut().spawn((
            super::EditorCamera,
            Transform::default(),
            CameraSettings::default(),
        ));
        app
    }

    fn apply_intent(app: &mut bevy::prelude::App, intent: super::UiIntent) {
        app.world_mut()
            .resource_mut::<super::UiIntentBuffer>()
            .push(intent);
        app.update();
    }

    #[test]
    fn inserting_composed_gallery_copies_complete_geometry() {
        let source = GalleryItem::ThreeBitAdder.build();
        let flat = source.flatten().unwrap();
        let mut app = intent_app();
        apply_intent(&mut app, super::UiIntent::InsertGraph(Box::new(source)));
        let state = app.world().resource::<GraphState>();
        assert_eq!(state.graph.block_count(), flat.block_count());
        assert_eq!(state.graph.pipe_count(), flat.pipe_count());
        assert!(!state.graph.has_module_structure());
    }

    #[test]
    fn history_changes_close_element_drafts_without_clearing_selection() {
        use super::UiIntent;

        let mut app = intent_app();
        let position = IVec3::ZERO;
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state
                .graph
                .add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
            state.commit();
            state
                .graph
                .set_block_kind(position, BlockKind::Cube(CubeKind::XZZ))
                .unwrap();
            state.commit();
        }
        app.world_mut()
            .resource_mut::<EditorState>()
            .select_element(GraphElement::Block(position), false);

        for intent in [UiIntent::Undo, UiIntent::Redo] {
            *app.world_mut().resource_mut::<TargetState>() = TargetState {
                open_window: true,
                target: Some(GraphElement::Block(position)),
                tag_buffer: "uncommitted".into(),
                ..Default::default()
            };
            apply_intent(&mut app, intent);
            let target = app.world().resource::<TargetState>();
            assert!(!target.open_window);
            assert!(target.target.is_none());
            assert!(target.tag_buffer.is_empty());
            assert!(
                app.world()
                    .resource::<EditorState>()
                    .selected_element_set()
                    .contains(&GraphElement::Block(position))
            );
        }

        // A shortcut with no available history must leave the draft alone.
        app.world_mut().resource_mut::<TargetState>().open_window = true;
        apply_intent(&mut app, UiIntent::Redo);
        assert!(app.world().resource::<TargetState>().open_window);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn quitting_checks_open_work_and_waits_for_confirmation() {
        use super::UiIntent;
        use bevy::prelude::*;
        let mut empty = intent_app();
        apply_intent(&mut empty, UiIntent::Quit);
        assert!(
            !empty
                .world()
                .resource::<crate::systems::ui::PendingTabClose>()
                .quit_requested
        );
        assert!(
            !empty
                .world()
                .resource::<Messages<bevy::app::AppExit>>()
                .is_empty()
        );

        let mut app = intent_app();
        apply_intent(&mut app, UiIntent::LoadGallery(GalleryItem::CNOT));
        apply_intent(&mut app, UiIntent::NewTab);
        assert!(app.world().resource::<GraphState>().graph.is_empty());
        apply_intent(&mut app, UiIntent::Quit);
        assert!(
            app.world()
                .resource::<crate::systems::ui::PendingTabClose>()
                .quit_requested
        );
        assert!(
            app.world()
                .resource::<Messages<bevy::app::AppExit>>()
                .is_empty()
        );
        apply_intent(&mut app, UiIntent::ConfirmQuit);
        assert!(
            !app.world()
                .resource::<Messages<bevy::app::AppExit>>()
                .is_empty()
        );
    }

    #[test]
    fn copying_blog_keeps_definitions_and_instances() {
        use bevy_egui::{EguiContext, PrimaryEguiContext, egui};
        let mut app = intent_app();
        let program = crate::module_authoring::tests::composition();
        let mut graph = app.world_mut().resource_mut::<GraphState>();
        graph.graph = program.flatten().unwrap();
        graph.source_graph = Some(std::sync::Arc::new(program));
        let mut context = EguiContext::default();
        let ctx = context.get_mut().clone();
        app.world_mut().spawn((context, PrimaryEguiContext));
        let mut output = ctx.run_ui(Default::default(), |_| {
            apply_intent(&mut app, super::UiIntent::CopyGraphAsBlog);
        });
        output.textures_delta.clear();
        let text = output
            .platform_output
            .commands
            .iter()
            .find_map(|command| match command {
                egui::OutputCommand::CopyText(text) => Some(text),
                _ => None,
            })
            .expect("copy produces clipboard text");
        let copied = bloq_graph::BlockGraph::from_text(text).unwrap();
        assert!(copied.module("Memory").is_some());
        assert_eq!(copied.root().instances.len(), 3);
    }

    #[test]
    fn composed_adder_branch_preview_preserves_modules_and_undo() {
        use super::UiIntent;
        let program = std::sync::Arc::new(GalleryItem::ThreeBitAdder.build());
        let mut app = intent_app();
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state.graph = program.flatten().unwrap();
            state.source_graph = Some(program.clone());
            state.commit();
        }
        let names: Vec<_> = app
            .world()
            .resource::<GraphState>()
            .graph
            .branch_definitions()
            .iter()
            .map(|branch| branch.name.clone())
            .collect();
        assert!(!names.is_empty());
        for name in names {
            for show_true in [false, true] {
                apply_intent(
                    &mut app,
                    UiIntent::ShowBranchArm {
                        name: name.clone(),
                        show_true,
                    },
                );
                for (intent, expected) in [
                    (None, show_true),
                    (Some(UiIntent::Undo), !show_true),
                    (Some(UiIntent::Redo), show_true),
                ] {
                    if let Some(intent) = intent {
                        apply_intent(&mut app, intent);
                    }
                    let state = app.world().resource::<GraphState>();
                    assert_eq!(
                        state.graph.branch_by_name(&name).unwrap().shown_true(),
                        expected
                    );
                    assert!(state.is_composed());
                    assert!(std::sync::Arc::ptr_eq(
                        state.source_graph.as_ref().unwrap(),
                        &program
                    ));
                    assert!(state.needs_rerender);
                }
            }
        }
    }

    #[test]
    fn flat_adder_copy_and_store_keep_the_graph_without_analyzing_stabilizers() {
        use super::UiIntent;
        use bevy_egui::{EguiContext, PrimaryEguiContext, egui};
        let program = bloq_graph::GalleryItem::ThreeBitAdder.build();
        let mut app = intent_app();
        {
            let mut graph = app.world_mut().resource_mut::<GraphState>();
            graph.graph = program.flatten().unwrap();
            graph.source_graph = Some(std::sync::Arc::new(program));
        }
        apply_intent(&mut app, UiIntent::OpenFlatCopy);
        let graph = app.world().resource::<GraphState>();
        assert!(graph.source_graph.is_none());
        let expected = graph.graph.to_blog_text();
        let mut context = EguiContext::default();
        let ctx = context.get_mut().clone();
        app.world_mut().spawn((context, PrimaryEguiContext));

        // Exercise both the new flat tab and the same document reopened as a leaf.
        for reopened in [false, true] {
            let mut output = ctx.run_ui(Default::default(), |_| {
                apply_intent(&mut app, UiIntent::CopyGraphAsBlog);
            });
            output.textures_delta.clear();
            let text = output
                .platform_output
                .commands
                .iter()
                .find_map(|command| match command {
                    egui::OutputCommand::CopyText(text) => Some(text),
                    _ => None,
                })
                .expect("copy produces clipboard text");
            assert_eq!(text, &expected, "reopened={reopened}");
            apply_intent(&mut app, UiIntent::StoreGraphToBuffer);
            assert_eq!(
                app.world().resource::<ImportExportState>().bloq_buffer,
                expected
            );
            let copied = bloq_graph::lower_blog_graph_ast_deferred(
                &bloq_graph::parse_blog_program_to_ast(text).unwrap(),
            )
            .unwrap();
            let mut graph = app.world_mut().resource_mut::<GraphState>();
            graph.graph = copied.flatten().unwrap();
            graph.source_graph = Some(std::sync::Arc::new(copied));
        }
    }

    #[test]
    fn mouse_drop_commits_once_and_rejects_stale_or_invalid_moves() {
        use super::{TranslationTarget, UiIntent};
        for module in [false, true] {
            let mut app = intent_app();
            let program = if module {
                crate::module_authoring::tests::composition()
            } else {
                crate::module_authoring::tests::stage()
            };
            {
                let mut graph = app.world_mut().resource_mut::<GraphState>();
                graph.graph = program.flatten().unwrap();
                graph.source_graph = Some(std::sync::Arc::new(program));
                graph.commit();
            }
            let graph = app.world().resource::<GraphState>();
            let before = graph.resolved_graph().unwrap().to_blog_text();
            let revision = graph.revision;
            let history = graph.current_index;
            let target = if module {
                TranslationTarget::Module("second".into())
            } else {
                TranslationTarget::Selection(GraphElement::all_in(&graph.graph).collect())
            };
            let tab = app.world().resource::<EditorTabs>().active;
            let offset = if module {
                IVec3::new(-3, 0, 2)
            } else {
                IVec3::new(2, 3, 4)
            };
            let drop = || UiIntent::FinishTranslation {
                tab,
                revision,
                target: target.clone(),
                offset,
                compact: true,
            };
            apply_intent(&mut app, drop());
            assert_eq!(
                app.world().resource::<GraphState>().current_index,
                history + 1
            );
            apply_intent(&mut app, drop());
            assert_eq!(
                app.world().resource::<GraphState>().current_index,
                history + 1
            );
            apply_intent(&mut app, UiIntent::Undo);
            assert_eq!(
                app.world()
                    .resource::<GraphState>()
                    .resolved_graph()
                    .unwrap()
                    .to_blog_text(),
                before
            );
            if module {
                let revision = app.world().resource::<GraphState>().revision;
                apply_intent(
                    &mut app,
                    UiIntent::FinishTranslation {
                        tab,
                        revision,
                        target: target.clone(),
                        offset: IVec3::new(-3, 0, 0),
                        compact: true,
                    },
                );
                assert_eq!(
                    app.world()
                        .resource::<GraphState>()
                        .resolved_graph()
                        .unwrap()
                        .to_blog_text(),
                    before
                );
            }
            apply_intent(&mut app, UiIntent::NewTab);
            apply_intent(&mut app, drop());
            assert!(app.world().resource::<GraphState>().graph.is_empty());
        }
    }

    #[test]
    fn module_editing_tabs_update_all_instances_with_undo_and_stale_edit_protection() {
        use super::{ModuleEdit, UiIntent};
        let mut app = intent_app();
        let stage = crate::module_authoring::tests::stage();
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state.graph = stage.flatten().unwrap();
            state.source_graph = Some(std::sync::Arc::new(stage));
            state.commit();
        }
        let parent = app.world().resource::<EditorTabs>().active;
        apply_intent(&mut app, UiIntent::WrapAsModule("Memory".into()));
        let revision = app.world().resource::<GraphState>().revision;
        apply_intent(
            &mut app,
            UiIntent::EditModule {
                revision,
                edit: ModuleEdit::AddInstance {
                    definition: "Memory".into(),
                    name: "other".into(),
                },
            },
        );
        let before = app
            .world()
            .resource::<GraphState>()
            .resolved_graph()
            .unwrap()
            .to_blog_text();
        apply_intent(&mut app, UiIntent::OpenModule("Memory".into()));
        let child = app.world().resource::<EditorTabs>().active;
        assert_ne!(child, parent);
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state.graph.set_block_tag(IVec3::Z, "updated").unwrap();
            state.commit();
        }
        apply_intent(&mut app, UiIntent::ApplyModuleTab);
        assert_eq!(app.world().resource::<EditorTabs>().active, parent);
        assert_eq!(
            app.world()
                .resource::<GraphState>()
                .graph
                .blocks()
                .filter(|b| b.tag() == Some("updated"))
                .count(),
            2
        );
        apply_intent(&mut app, UiIntent::Undo);
        assert_eq!(
            app.world()
                .resource::<GraphState>()
                .resolved_graph()
                .unwrap()
                .to_blog_text(),
            before
        );
        apply_intent(&mut app, UiIntent::Redo);
        apply_intent(&mut app, UiIntent::StoreGraphToBuffer);
        assert!(
            app.world()
                .resource::<ImportExportState>()
                .bloq_buffer
                .contains("module Memory")
        );
        let revision = app.world().resource::<GraphState>().revision;
        apply_intent(
            &mut app,
            UiIntent::TranslateGraph {
                axis: bloq_graph::UDirection::X,
                step: 1,
            },
        );
        assert_eq!(app.world().resource::<GraphState>().revision, revision);
        apply_intent(
            &mut app,
            UiIntent::SetMode(crate::resources::EditorMode::Edit),
        );
        assert_eq!(
            app.world().resource::<EditorState>().mode,
            crate::resources::EditorMode::Module
        );

        let mut updated = crate::module_authoring::tests::stage();
        let mut body = updated.flatten().unwrap();
        body.set_block_tag(IVec3::Z, "parent_changed").unwrap();
        updated = crate::module_authoring::replace_leaf_body(&updated, &body).unwrap();
        apply_intent(
            &mut app,
            UiIntent::EditModule {
                revision,
                edit: ModuleEdit::Replace {
                    name: "Memory".into(),
                    program: Box::new(updated),
                },
            },
        );
        apply_intent(&mut app, UiIntent::SelectTab(child.get()));
        apply_intent(&mut app, UiIntent::ApplyModuleTab);
        assert_eq!(app.world().resource::<EditorTabs>().active, child);
        assert!(
            app.world()
                .resource::<ImportExportState>()
                .module_ui
                .error
                .as_ref()
                .unwrap()
                .contains("changed in its parent")
        );
        apply_intent(&mut app, UiIntent::SelectTab(parent.get()));
        apply_intent(&mut app, UiIntent::OpenFlatCopy);
        assert!(!app.world().resource::<GraphState>().is_composed());
        assert!(app.world().resource::<GraphState>().source_graph.is_none());
        apply_intent(&mut app, UiIntent::SelectTab(parent.get()));
        assert!(app.world().resource::<GraphState>().is_composed());
    }

    #[test]
    fn branch_capture_undo_redo_and_insertion_keep_hidden_geometry_owned() {
        use super::UiIntent;

        let mut app = intent_app();
        let select_arm = |app: &mut bevy::prelude::App| {
            app.world_mut()
                .resource_mut::<EditorState>()
                .replace_selection([
                    GraphElement::Block(IVec3::Z),
                    GraphElement::Pipe(IVec3::ZERO, IVec3::Z),
                ]);
        };
        let branch_state = |app: &bevy::prelude::App| {
            let state = app.world().resource::<GraphState>();
            (
                state.pending_branch_arm.is_some(),
                state.graph.branch_definitions().len(),
                state.graph.has_block_at(IVec3::Z),
            )
        };
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            for pos in [IVec3::ZERO, IVec3::Z] {
                state
                    .graph
                    .add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
            }
            state
                .graph
                .add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
            state.commit();
        }
        select_arm(&mut app);
        apply_intent(
            &mut app,
            UiIntent::CaptureBranchArm {
                captured_true: false,
            },
        );
        assert_eq!(branch_state(&app), (true, 0, false));
        apply_intent(&mut app, UiIntent::Undo);
        assert_eq!(branch_state(&app), (false, 0, true));
        apply_intent(&mut app, UiIntent::Redo);
        assert_eq!(branch_state(&app), (true, 0, false));

        let mut inserted = bloq_graph::BlockGraph::new();
        inserted.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        apply_intent(&mut app, UiIntent::InsertGraph(Box::new(inserted)));
        assert_eq!(branch_state(&app), (true, 0, false));
        apply_intent(&mut app, UiIntent::Undo);
        apply_intent(&mut app, UiIntent::CancelBranchArm);
        assert_eq!(branch_state(&app), (false, 0, true));
        apply_intent(&mut app, UiIntent::Undo);
        assert_eq!(branch_state(&app), (true, 0, false));
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state
                .graph
                .add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            state
                .graph
                .add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
            state.commit();
        }
        select_arm(&mut app);
        apply_intent(
            &mut app,
            UiIntent::CaptureBranchArm {
                captured_true: true,
            },
        );
        assert_eq!(branch_state(&app), (false, 1, true));
        apply_intent(&mut app, UiIntent::Undo);
        assert_eq!(branch_state(&app), (true, 0, true));
    }

    #[test]
    fn whole_graph_transforms_include_the_captured_arm_and_fail_atomically() {
        use super::UiIntent;
        let mut app = intent_app();
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state
                .graph
                .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
            state.pending_branch_arm = Some(super::PendingBranchArm {
                name: "b0".into(),
                arm: bloq_graph::BranchArm::new(
                    vec![Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ))],
                    vec![Pipe::new(IVec3::ZERO, Direction::ZPLUS)],
                ),
                captured_true: false,
            });
            state.commit();
        }
        // A partial move of the shared input must leave the capture restorable.
        app.world_mut()
            .resource_mut::<EditorState>()
            .replace_selection([GraphElement::Block(IVec3::ZERO)]);
        let revision = app.world().resource::<GraphState>().revision;
        apply_intent(
            &mut app,
            UiIntent::TranslateGraph {
                axis: bloq_graph::UDirection::X,
                step: 2,
            },
        );
        assert_eq!(app.world().resource::<GraphState>().revision, revision);
        app.world_mut()
            .resource_mut::<EditorState>()
            .clear_selection();

        apply_intent(
            &mut app,
            UiIntent::TranslateGraph {
                axis: bloq_graph::UDirection::X,
                step: 2,
            },
        );
        apply_intent(
            &mut app,
            UiIntent::RotateGraph {
                axis: bloq_graph::UDirection::Z,
                quarter_turns: 1,
            },
        );
        apply_intent(&mut app, UiIntent::FlipXZBasis);
        apply_intent(&mut app, UiIntent::CancelBranchArm);
        let state = app.world().resource::<GraphState>();
        assert!(state.pending_branch_arm.is_none());
        for pos in [IVec3::new(0, 2, 0), IVec3::new(0, 2, 1)] {
            assert_eq!(
                state.graph.get_block(pos).unwrap().kind(),
                BlockKind::Cube(CubeKind::ZXX)
            );
        }

        apply_intent(&mut app, UiIntent::Undo);
        let (before, hidden, revision) = {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            let pending = state.pending_branch_arm.as_mut().unwrap();
            pending.arm = bloq_graph::BranchArm::new(
                pending
                    .arm
                    .blocks()
                    .cloned()
                    .chain([Block::new(
                        IVec3::new(i32::MAX, 0, 0),
                        BlockKind::Cube(CubeKind::ZXZ),
                    )])
                    .collect(),
                pending.arm.pipes().cloned().collect(),
            );
            state.commit();
            (
                state.graph.to_blog_text(),
                state.pending_branch_arm.as_ref().unwrap().arm.clone(),
                state.revision,
            )
        };
        apply_intent(
            &mut app,
            UiIntent::TranslateGraph {
                axis: bloq_graph::UDirection::X,
                step: 1,
            },
        );
        let state = app.world().resource::<GraphState>();
        assert_eq!(state.graph.to_blog_text(), before);
        assert_eq!(state.pending_branch_arm.as_ref().unwrap().arm, hidden);
        assert_eq!(state.revision, revision);
    }

    #[test]
    fn queued_action_replacement_cannot_overwrite_a_shifted_neighbour() {
        use super::UiIntent;
        use bloq_graph::{FeedbackTarget, PauliBasis};

        let mut app = intent_app();
        let actions = [PauliBasis::X, PauliBasis::X, PauliBasis::Y].map(|pauli| Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: None,
        });
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state
                .graph
                .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
            state.graph.set_actions_lenient(actions.to_vec()).unwrap();
            state.commit();
        }
        app.world_mut()
            .resource_mut::<super::UiIntentBuffer>()
            .push(UiIntent::RemoveAction(0));
        apply_intent(
            &mut app,
            UiIntent::ReplaceAction {
                index: 0,
                expected: actions.to_vec(),
                action: actions[2].clone(),
            },
        );
        assert_eq!(
            app.world().resource::<GraphState>().graph.actions(),
            actions[1..]
        );
    }

    #[test]
    fn switching_either_direction_between_layer_and_full_view_clears_stabilizers() {
        let mut app = intent_app();
        let stabilizers = GalleryItem::BellState
            .build()
            .flatten()
            .unwrap()
            .stabilizers()
            .unwrap();
        assert!(!stabilizers.is_empty());
        for enabled in [true, false] {
            app.world_mut().resource_mut::<EditorState>().stabilizers = stabilizers.clone();
            apply_intent(&mut app, super::UiIntent::SetViewCurrentLayerOnly(enabled));
            assert!(app.world().resource::<EditorState>().stabilizers.is_empty());
        }
    }

    #[test]
    fn action_edit_commits_without_physical_analysis() {
        let mut graph_state = GraphState {
            graph: GalleryItem::T.build().flatten().unwrap(),
            ..Default::default()
        };
        let actions = graph_state.graph.actions();

        apply_action_edit(
            &mut graph_state,
            &mut EditorState::default(),
            actions,
            "Updated actions".to_string(),
            &mut Notifications::default(),
        );

        assert_eq!(graph_state.revision, 1);
        assert!(!graph_state.graph.action_graph().is_analyzed());
    }

    #[test]
    fn apply_element_edit_sets_pipe_hadamard_and_commits() {
        let pipe_start = IVec3::ZERO;
        let pipe_end = IVec3::X;
        let mut editor_state = EditorState::default();
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pipe_start, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(pipe_end, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_pipe(Pipe::new(pipe_start, Direction::XPLUS));
        let initial_history_len = graph_state.history.len();

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Pipe(pipe_start, pipe_end),
                block_kind: BlockKind::Cube(CubeKind::ZXZ),
                block_color: None,
                port_role: bloq_graph::PortRole::Auto,
                tag: "bridge".to_string(),
                pipe_hadamard: true,
            },
        );

        let pipe = graph_state
            .graph
            .get_pipe(pipe_start, pipe_end)
            .expect("pipe should still exist after edit");
        assert!(pipe.is_hadamard());
        assert_eq!(pipe.tag(), Some("bridge"));
        assert_eq!(graph_state.history.len(), initial_history_len + 1);
        assert_eq!(graph_state.current_index, initial_history_len);
        assert_eq!(graph_state.revision, 1);
    }

    #[test]
    fn random_selective_resolution_refuses_structural_branches_without_mutation() {
        let prefix = IVec3::ZERO;
        let target = IVec3::Z;
        let reader = IVec3::new(2, 0, 0);
        let mut graph_state = GraphState::default();
        for pos in [prefix, reader] {
            graph_state
                .graph
                .add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let pipe = || Pipe::new(prefix, Direction::ZPLUS);
        let target = graph_state
            .graph
            .try_add_branch_region(
                "b0",
                BranchArm::new(
                    vec![Block::new(target, BlockKind::Measurement(Basis::X))],
                    vec![pipe()],
                ),
                BranchArm::new(
                    vec![Block::new(target, BlockKind::Cube(CubeKind::ZXZ))],
                    vec![pipe()],
                ),
            )
            .expect("matching terminal arms");
        graph_state
            .graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(reader),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: Expr::Var("m".into()),
                },
            ])
            .expect("test branch validates");
        graph_state.graph.add_block(Block::new(
            IVec3::new(4, 0, 0),
            BlockKind::Selective(SelectiveKind::XZ),
        ));
        let before = graph_state.graph.to_blog_text();
        let mut notifications = Notifications::default();

        apply_randomly_resolve_selectives(
            &mut graph_state,
            &mut EditorState::default(),
            &mut TargetState::default(),
            &mut CompileUiState::default(),
            &mut BloqViewerState::default(),
            &mut notifications,
        );

        assert_eq!(graph_state.graph.to_blog_text(), before);
        assert_eq!(graph_state.revision, 0);
        let toast = notifications.toasts.back().expect("warning toast");
        assert_eq!(toast.level, crate::resources::ToastLevel::Warn);
        assert!(toast.message.contains("graph unchanged"));
    }

    #[test]
    fn apply_element_edit_keeps_tag_only_pipe_edit_off_render_path() {
        let pipe_start = IVec3::ZERO;
        let pipe_end = IVec3::X;
        let mut editor_state = EditorState::default();
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pipe_start, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_block(Block::new(pipe_end, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state
            .graph
            .add_pipe(Pipe::new(pipe_start, Direction::XPLUS));

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Pipe(pipe_start, pipe_end),
                block_kind: BlockKind::Cube(CubeKind::ZXZ),
                block_color: None,
                port_role: bloq_graph::PortRole::Auto,
                tag: "metadata".to_string(),
                pipe_hadamard: false,
            },
        );

        let pipe = graph_state
            .graph
            .get_pipe(pipe_start, pipe_end)
            .expect("pipe should still exist after edit");
        assert_eq!(pipe.tag(), Some("metadata"));
        assert_eq!(graph_state.revision, 1);
        assert!(!graph_state.needs_rerender);
    }

    #[test]
    fn apply_element_edit_rerenders_block_kind_changes() {
        let pos = IVec3::ZERO;
        let mut editor_state = EditorState::default();
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Block(pos),
                block_kind: BlockKind::Cube(CubeKind::XZZ),
                block_color: None,
                port_role: bloq_graph::PortRole::Auto,
                tag: String::new(),
                pipe_hadamard: false,
            },
        );

        assert_eq!(graph_state.revision, 1);
        assert!(graph_state.needs_rerender);
    }

    #[test]
    fn apply_element_edit_rerenders_visible_port_tag_changes() {
        let pos = IVec3::ZERO;
        let mut editor_state = EditorState {
            show_port_tags: true,
            ..Default::default()
        };
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pos, BlockKind::Port));

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Block(pos),
                block_kind: BlockKind::Port,
                block_color: Some(DEFAULT_PORT_RGB),
                port_role: bloq_graph::PortRole::Auto,
                tag: "renamed".to_string(),
                pipe_hadamard: false,
            },
        );

        assert!(graph_state.needs_rerender);
    }

    #[test]
    fn apply_element_edit_sets_port_color_and_rerenders() {
        let pos = IVec3::ZERO;
        let mut editor_state = EditorState::default();
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(pos, BlockKind::Port));

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Block(pos),
                block_kind: BlockKind::Port,
                block_color: Some([0xeb, 0x40, 0x34]),
                port_role: bloq_graph::PortRole::Output,
                tag: String::new(),
                pipe_hadamard: false,
            },
        );

        let color = graph_state
            .graph
            .get_block(pos)
            .and_then(Block::port_color)
            .expect("Port has a color");
        assert_eq!([color.r, color.g, color.b], [0xeb, 0x40, 0x34]);
        assert_eq!(
            graph_state.graph.get_block(pos).and_then(Block::port_role),
            Some(bloq_graph::PortRole::Output)
        );
        assert_eq!(graph_state.revision, 1);
        assert!(graph_state.needs_rerender);
    }

    #[test]
    fn invalid_block_tag_does_not_apply_kind_change() {
        let pos = IVec3::ZERO;
        let original_kind = BlockKind::Cube(CubeKind::ZXZ);
        let mut editor_state = EditorState::default();
        let mut graph_state = GraphState::default();
        graph_state.graph.add_block(Block::new(pos, original_kind));

        apply_element_edit(
            &mut editor_state,
            &mut graph_state,
            &mut Notifications::default(),
            ElementEditIntent {
                target: GraphElement::Block(pos),
                block_kind: BlockKind::Cube(CubeKind::XZZ),
                block_color: None,
                port_role: bloq_graph::PortRole::Auto,
                tag: "not a tag".to_string(),
                pipe_hadamard: false,
            },
        );

        assert_eq!(
            graph_state
                .graph
                .get_block(pos)
                .expect("block remains")
                .kind(),
            original_kind
        );
        assert_eq!(graph_state.revision, 0);
        assert_eq!(graph_state.history.len(), 1);
    }

    #[test]
    fn close_element_editor_clears_selection_and_target() {
        let mut editor_state = EditorState::default();
        editor_state.select_element(GraphElement::Block(IVec3::ZERO), false);
        let mut target_state = TargetState {
            open_window: true,
            target: Some(GraphElement::Block(IVec3::ZERO)),
            tag_buffer: "selected".to_string(),
            ..TargetState::default()
        };

        close_element_editor(&mut editor_state, &mut target_state);

        assert_eq!(editor_state.selection_count(), 0);
        assert!(!target_state.open_window);
        assert!(target_state.target.is_none());
        assert!(target_state.tag_buffer.is_empty());
    }

    #[test]
    fn import_preserves_blog_draft_even_when_graph_is_empty() {
        let mut tabs = EditorTabs::default();
        let original = tabs.active;
        let mut graph_state = GraphState::default();
        let mut editor_state = EditorState::default();
        let mut import_export = ImportExportState::default();
        import_export.set_bloq_buffer("unsaved BLOG draft".to_string());
        let mut compile_ui = CompileUiState::default();
        let mut circuit_viewer = BloqViewerState::default();
        let mut zx_viewer = ZxViewerState::default();
        let mut target_state = TargetState::default();
        let mut box_selection = BoxSelectionState::default();
        let mut camera_settings = CameraSettings::default();

        let imported = prepare_tab_for_import(
            &mut tabs,
            Some("Imported".to_string()),
            &mut LiveTabState {
                graph_state: &mut graph_state,
                editor_state: &mut editor_state,
                import_export: &mut import_export,
                compile_ui: &mut compile_ui,
                circuit_viewer: &mut circuit_viewer,
                zx_viewer: &mut zx_viewer,
                target_state: &mut target_state,
                box_selection: &mut box_selection,
                camera_settings: &mut camera_settings,
            },
        );

        assert_ne!(imported, original);
        assert_eq!(tabs.tabs.len(), 2);
        assert_eq!(tabs.title(imported), Some("Imported"));
        assert_eq!(
            tabs.tabs[0].snapshot.import_export.bloq_buffer,
            "unsaved BLOG draft"
        );
        assert!(import_export.bloq_buffer.is_empty());
    }

    #[test]
    fn close_confirmation_protects_drafts_and_redo_history() {
        let tabs = EditorTabs::default();
        let mut graph_state = GraphState::default();
        let mut import_export = ImportExportState::default();
        import_export.set_bloq_buffer("BLOG 1.0".to_string());

        assert!(tab_has_work(
            tabs.active,
            &tabs,
            &graph_state,
            &import_export,
        ));

        import_export.bloq_buffer.clear();
        graph_state
            .graph
            .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph_state.commit();
        graph_state.undo();
        assert!(tab_has_work(
            tabs.active,
            &tabs,
            &graph_state,
            &import_export,
        ));
    }
}
