//! The main toolbar: app actions, the mode selector, and the compile controls.

use super::activated;
use super::command_palette::CommandPaletteState;
use super::intents::{UiIntent, UiIntentBuffer};
use crate::resources::{
    ActionViewerState, CompileUiState, EditorMode, EditorState, GraphState, ImportExportState,
};
use crate::systems::jobs::EditorJobs;
use crate::theme::{self, ThemePalette};
use bevy_egui::egui;

const FORMAT_LABEL: &str = "Format";
const MODES: [EditorMode; 4] = [
    EditorMode::View,
    EditorMode::Module,
    EditorMode::Edit,
    EditorMode::Bloq,
];

/// The mutable editor state the toolbar reads and writes, bundled to keep its
/// draw function's signature manageable.
pub(crate) struct ToolbarState<'a> {
    pub(crate) editor_state: &'a mut EditorState,
    pub(crate) action_viewer: &'a mut ActionViewerState,
    pub(crate) import_export: &'a mut ImportExportState,
    pub(crate) compile_ui: &'a mut CompileUiState,
    pub(crate) graph_state: &'a GraphState,
    pub(crate) jobs: &'a EditorJobs,
    pub(crate) intents: &'a mut UiIntentBuffer,
    pub(crate) command_palette: &'a mut CommandPaletteState,
}

/// The mode's UI label — `Bloq` is shown as `Program`. Shared with the
/// command palette so both name a mode the same way.
pub(super) const fn compact_mode_label(mode: EditorMode) -> &'static str {
    match mode {
        EditorMode::View => "View",
        EditorMode::Module => "Modules",
        EditorMode::Edit => "Edit",
        EditorMode::Bloq => "Program",
    }
}

const fn mode_hover_text(mode: EditorMode) -> &'static str {
    match mode {
        EditorMode::View => {
            "Select, move, inspect, and validate your graph (V). Space switches between View and Edit."
        }
        EditorMode::Module => "Build reusable modules, place instances, and connect their ports",
        EditorMode::Edit => {
            "Place blocks and connect them with pipes. Space switches between Edit and View; P selects the Pipe tool."
        }
        EditorMode::Bloq => "Compile and inspect the program and physical circuits",
    }
}

/// Draws the toolbar row, queuing intents for app actions, mode switches, and
/// compile requests.
pub(crate) fn draw_toolbar_contents(
    ui: &mut egui::Ui,
    state: ToolbarState<'_>,
    palette: &ThemePalette,
) {
    ui.horizontal_wrapped(|ui| {
        ui.set_row_height(36.0);
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
        ui.spacing_mut().interact_size.y = 28.0;
        ui.style_mut().visuals.widgets.hovered.weak_bg_fill = palette.bg_hover;
        ui.style_mut().visuals.widgets.active.weak_bg_fill = palette.bg_active;
        draw_toolbar_identity(ui, palette);
        toolbar_menu("File").ui(ui, |ui| {
            draw_file_actions(ui, state.import_export, state.intents, palette)
        });
        if let Some(origin) = &state.import_export.definition_edit
            && ui
                .button(format!("Apply to {}", origin.name))
                .on_hover_text("Update every instance in the parent composition and return to it")
                .clicked()
        {
            state.intents.push(UiIntent::ApplyModuleTab);
        }
        for (label, enabled, intent, hint) in [
            (
                "Undo",
                state.graph_state.current_index > 0,
                UiIntent::Undo,
                "Undo graph change (Ctrl+Z)",
            ),
            (
                "Redo",
                state.graph_state.current_index + 1 < state.graph_state.history.len(),
                UiIntent::Redo,
                "Redo graph change (Ctrl+Shift+Z)",
            ),
            (
                "Fit",
                !state.graph_state.graph.is_empty(),
                UiIntent::ResetCamera,
                "Fit graph in view",
            ),
        ] {
            if activated(
                &ui.add_enabled(enabled, toolbar_button(label))
                    .on_hover_text(hint),
            ) {
                state.intents.push(intent);
            }
        }
        if activated(
            &ui.add_enabled(
                state.editor_state.selection_count() == 1
                    && matches!(state.editor_state.mode, EditorMode::View | EditorMode::Edit),
                toolbar_button("Attributes"),
            )
            .on_hover_text("Edit the selected block or pipe (Enter). Double-click also works."),
        ) {
            state.intents.push(UiIntent::EditSelection);
        }
        toolbar_divider(ui, palette);
        // Reserve the complete segment before layout so wrapping moves the group as one item.
        let modes_width = MODES
            .into_iter()
            .map(|mode| {
                ui.painter()
                    .layout_no_wrap(
                        compact_mode_label(mode).into(),
                        egui::FontId::proportional(13.0),
                        palette.text_primary,
                    )
                    .size()
                    .x
            })
            .sum::<f32>()
            + 2.0 * MODES.len() as f32 * ui.spacing().button_padding.x
            + 16.0;
        ui.allocate_ui(egui::vec2(modes_width, 36.0), |ui| {
            egui::Frame::new()
                .fill(palette.bg_panel)
                .stroke(egui::Stroke::new(
                    1.0,
                    theme::with_alpha(palette.border, 70),
                ))
                .corner_radius(8)
                .inner_margin(3)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        draw_mode_selector(
                            ui,
                            state.editor_state,
                            state.graph_state.is_composed(),
                            state.intents,
                            palette,
                        );
                    });
                });
        });
        toolbar_divider(ui, palette);
        if activated(
            &ui.add(toolbar_button("Actions").selected(state.action_viewer.open))
                .on_hover_text("Measurements and feedback (A)"),
        ) {
            state.action_viewer.toggle();
        }
        let pending_text = state.import_export.blocks_graph_editing();
        let text_error =
            state.import_export.sync_conflict || state.import_export.sync_error.is_some();
        if activated(
            &ui.add(
                toolbar_button(if text_error {
                    "BLOG !"
                } else if pending_text {
                    "BLOG …"
                } else {
                    "BLOG"
                })
                .selected(state.editor_state.show_blog_buffer),
            )
            .on_hover_text(if text_error {
                "Graph editing paused: open BLOG to fix or resolve the pending text."
            } else if pending_text {
                "Checking text; graph editing resumes when valid."
            } else {
                "Live BLOG view (B)"
            }),
        ) {
            state.editor_state.show_blog_buffer = !state.editor_state.show_blog_buffer;
        }
        if state.editor_state.is_pipe_tool_active() {
            ui.checkbox(&mut state.editor_state.pipe_hadamard, "Hadamard")
                .on_hover_text("Place Hadamard pipes. Alternatively, hold H.");
        }
        toolbar_menu("Compile settings").ui(ui, |ui| {
            draw_compile_controls(ui, state.compile_ui, palette)
        });
        draw_compile_button(
            ui,
            state.compile_ui,
            state.graph_state,
            state.jobs,
            state.intents,
            palette,
        );
        toolbar_divider(ui, palette);
        if activated(
            &ui.add(toolbar_button("Commands"))
                .on_hover_text("Find any command (Ctrl+Shift+P / F1)"),
        ) {
            state.command_palette.open();
        }
        draw_utility_controls(ui, state.editor_state, state.intents);
        draw_status_chips(ui, palette, state.jobs.is_busy());
    });
}

fn draw_toolbar_identity(ui: &mut egui::Ui, palette: &ThemePalette) {
    ui.label(
        egui::RichText::new("Bloq")
            .family(theme::bold_family())
            .size(16.0)
            .color(palette.accent_primary),
    )
    .on_hover_text("Bloq Editor");
}

fn draw_file_actions(
    ui: &mut egui::Ui,
    import_export: &mut ImportExportState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if ui.button("New tab").clicked() {
        intents.push(UiIntent::NewTab);
        ui.close();
    }
    if ui.button("Open BLOG...").clicked() {
        intents.push(UiIntent::ImportBlogFromFile);
        ui.close();
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut import_export.export_path)
                .desired_width(200.0)
                .hint_text("graph.blog"),
        )
        .on_hover_text(if cfg!(target_arch = "wasm32") {
            "BLOG download filename"
        } else {
            "BLOG save path"
        });
        if ui
            .button(if cfg!(target_arch = "wasm32") {
                "Download"
            } else {
                "Save"
            })
            .on_hover_text("Export the graph as BLOG. To preserve comments or unfinished text exactly, use Save/Download draft in the BLOG panel.")
            .clicked()
        {
            intents.push(UiIntent::ExportBlog {
                path: import_export.export_path.clone(),
            });
            ui.close();
        }
    });
    ui.separator();
    if ui
        .button(egui::RichText::new("Clear graph").color(palette.accent_error))
        .on_hover_text("Clear this graph. Undo restores it.")
        .clicked()
    {
        intents.push(UiIntent::ClearGraph);
        ui.close();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        ui.separator();
        if ui.button("Quit...").clicked() {
            intents.push(UiIntent::Quit);
            ui.close();
        }
    }
}

fn draw_mode_selector(
    ui: &mut egui::Ui,
    editor_state: &EditorState,
    composed: bool,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let current_mode = editor_state.mode;
    let mut selected_mode = current_mode;

    for mode in MODES {
        let is_active = current_mode == mode;
        let text = egui::RichText::new(compact_mode_label(mode))
            .size(13.0)
            .color(if is_active {
                palette.accent_primary
            } else {
                palette.text_primary
            });
        let mut button = egui::Button::new(text)
            .frame_when_inactive(is_active)
            .corner_radius(6)
            .min_size(egui::vec2(0.0, 28.0));
        if is_active {
            button = button.fill(palette.bg_surface).stroke(egui::Stroke::new(
                1.0,
                theme::with_alpha(palette.border, 100),
            ));
        }
        let available = mode != EditorMode::Edit || !composed;
        let response = ui
            .add_enabled(available, button.selected(is_active))
            .on_disabled_hover_text("Edit a definition in Modules, or open a flat copy")
            .on_hover_text(if available {
                mode_hover_text(mode)
            } else {
                "Edit a definition in Modules, or open a flat copy"
            });
        if activated(&response) {
            selected_mode = mode;
        }
    }

    if selected_mode != current_mode {
        intents.push(UiIntent::SetMode(selected_mode));
    }
}

fn draw_compile_controls(
    ui: &mut egui::Ui,
    compile_ui: &mut CompileUiState,
    palette: &ThemePalette,
) {
    ui.set_width(280.0);
    ui.horizontal(|ui| {
        ui.label("Code distance");
        ui.add(
            egui::TextEdit::singleline(&mut compile_ui.code_distance_input)
                .id_salt("compile-distance")
                .desired_width(48.0),
        );
    });
    ui.horizontal(|ui| {
        ui.label(FORMAT_LABEL);
        // A submenu shares the settings menu's open state; a ComboBox replaces it.
        ui.menu_button(compile_ui.format.display_label(), |ui| {
            for format in crate::resources::CompileOutputFormat::ALL {
                ui.selectable_value(&mut compile_ui.format, format, format.display_label());
            }
        });
    });
    ui.separator();
    ui.checkbox(&mut compile_ui.prepare_t_with_mpps, "Compact T preparation")
        .on_hover_text("Prepare T blocks with resets, one T gate, and stabilizer MPPs");
    if let Err(error) = compile_ui.build_request() {
        ui.add(egui::Label::new(egui::RichText::new(error).color(palette.accent_error)).wrap());
    }
}

fn draw_utility_controls(
    ui: &mut egui::Ui,
    editor_state: &mut EditorState,
    intents: &mut UiIntentBuffer,
) {
    if activated(&toolbar_icon(ui, "\u{f030}", "Screenshot")) {
        intents.push(UiIntent::RequestScreenshot);
    }

    if activated(&toolbar_icon(
        ui,
        "\u{f059}",
        "Help: keyboard shortcuts and camera controls",
    )) {
        editor_state.show_help_window = !editor_state.show_help_window;
    }

    let (next_theme, hint) = match editor_state.theme_preset {
        theme::ThemePreset::Light => (theme::ThemePreset::GruvboxMaterial, "Switch to dark theme"),
        theme::ThemePreset::GruvboxMaterial => (theme::ThemePreset::Light, "Switch to light theme"),
    };
    if activated(&toolbar_icon(ui, "\u{f042}", hint)) {
        intents.push(UiIntent::SetTheme(next_theme));
    }
}

fn draw_compile_button(
    ui: &mut egui::Ui,
    compile_ui: &CompileUiState,
    graph_state: &GraphState,
    jobs: &EditorJobs,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if let Some((tab_id, _, _)) = jobs.compilation_owner() {
        if ui
            .button("Cancel compilation")
            .on_hover_text("Cancel the running compilation and discard its output")
            .clicked()
        {
            intents.push(UiIntent::CancelCompilation(tab_id));
        }
        return;
    }
    let compile_running = jobs.compilation_running();
    let request = compile_ui.build_request();
    let disabled_reason = if graph_state.pending_branch_arm.is_some() {
        Some("Finish or cancel the captured branch arm first.")
    } else if graph_state.graph.is_empty() {
        Some("Add blocks or open an example before compiling.")
    } else if compile_running {
        Some("Compilation is already running.")
    } else {
        request.as_ref().err().map(String::as_str)
    };
    let enabled = disabled_reason.is_none();
    let compile_label = if compile_running {
        "\u{f110} Compiling..."
    } else {
        "\u{f054} Compile & export"
    };
    let text_color = if !enabled {
        palette.text_dim
    } else if palette.dark_mode {
        palette.bg_dark
    } else {
        egui::Color32::WHITE
    };
    let compile_text = egui::RichText::new(compile_label).small().color(text_color);
    let response = ui
        .add_enabled(
            enabled,
            egui::Button::new(compile_text)
                .fill(if enabled {
                    palette.accent_primary
                } else {
                    palette.bg_panel
                })
                .corner_radius(6)
                .stroke(egui::Stroke::NONE)
                .min_size(egui::vec2(0.0, 28.0)),
        )
        .on_hover_text(format!(
            "Compile and export {} at distance {}. Use Program to inspect without exporting.",
            compile_ui.format.display_label(),
            compile_ui.code_distance_input.trim()
        ))
        .on_disabled_hover_text(disabled_reason.unwrap_or_default());
    if activated(&response) {
        match request {
            Ok(request) => intents.push(UiIntent::CompileGraph(request)),
            Err(err) => intents.error(err),
        }
    }
}

fn draw_status_chips(ui: &mut egui::Ui, palette: &ThemePalette, jobs_busy: bool) {
    if jobs_busy {
        ui.label(
            egui::RichText::new("\u{f110} Busy")
                .small()
                .color(palette.accent_warn),
        )
        .on_hover_text("Editor background work is still running.");
    }
}

fn toolbar_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(egui::RichText::new(label).size(13.0))
        .frame_when_inactive(false)
        .corner_radius(6)
        .min_size(egui::vec2(28.0, 28.0))
}

fn toolbar_divider(ui: &mut egui::Ui, palette: &ThemePalette) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 18.0), egui::Sense::hover());
    ui.painter().vline(
        rect.center().x,
        rect.y_range(),
        egui::Stroke::new(1.0, theme::with_alpha(palette.border, 100)),
    );
}

fn toolbar_icon(ui: &mut egui::Ui, label: &str, hover_text: &str) -> egui::Response {
    let response = ui.add(toolbar_button(label)).on_hover_text(hover_text);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, hover_text));
    response
}

fn toolbar_menu(label: &str) -> egui::containers::menu::MenuButton<'_> {
    egui::containers::menu::MenuButton::from_button(toolbar_button(label)).config(
        egui::containers::menu::MenuConfig::new()
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemePreset;

    #[test]
    fn command_palette_and_toolbar_offer_the_same_four_modes() {
        use super::super::commands::{Command, all_commands};

        let modes: Vec<_> = all_commands()
            .into_iter()
            .filter_map(|command| match command {
                Command::SetMode(mode) => Some(mode),
                _ => None,
            })
            .collect();
        assert_eq!(modes, MODES);
        assert_eq!(
            MODES.map(compact_mode_label),
            ["View", "Modules", "Edit", "Program"]
        );
    }

    #[test]
    fn compile_format_selection_keeps_the_settings_menu_open() {
        use crate::resources::CompileOutputFormat;

        for width in [480.0, 1280.0] {
            let ctx = egui::Context::default();
            theme::setup_theme(&ctx, ThemePreset::Light);
            let mut compile = CompileUiState::default();
            let frame = |compile: &mut CompileUiState, events| {
                rendered_text(ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 720.0),
                        )),
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        toolbar_menu("Compile settings").ui(ui, |ui| {
                            draw_compile_controls(ui, compile, theme::palette(ThemePreset::Light));
                        });
                    },
                ))
            };
            let click = |compile: &mut CompileUiState, pos| {
                for pressed in [true, false] {
                    frame(
                        compile,
                        vec![
                            egui::Event::PointerMoved(pos),
                            egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed,
                                modifiers: Default::default(),
                            },
                        ],
                    );
                }
                frame(compile, Vec::new())
            };
            let position = |labels: &[(String, egui::Rect)], text: &str| {
                labels
                    .iter()
                    .find(|(label, _)| label == text)
                    .unwrap_or_else(|| panic!("missing {text}: {labels:?}"))
                    .1
                    .center()
            };
            frame(&mut compile, Vec::new());
            let labels = frame(&mut compile, Vec::new());
            let mut labels = click(&mut compile, position(&labels, "Compile settings"));
            for format in [
                CompileOutputFormat::IrText,
                CompileOutputFormat::IrBinary,
                CompileOutputFormat::Stim,
            ] {
                if !labels
                    .iter()
                    .any(|(label, _)| label == format.display_label())
                {
                    let current = position(&labels, compile.format.display_label());
                    labels = click(&mut compile, current);
                }
                assert!(
                    labels.iter().any(|(label, _)| label == "Code distance"),
                    "Opening Format must keep its parent settings menu visible"
                );
                labels = click(&mut compile, position(&labels, format.display_label()));
                assert_eq!(compile.format, format);
                assert!(labels.iter().any(|(label, _)| label == "Code distance"));
            }
            let labels = click(&mut compile, egui::pos2(width - 10.0, 710.0));
            assert!(
                !labels.iter().any(|(label, _)| label == "Code distance"),
                "Clicking outside must still dismiss the menu"
            );
        }
    }

    #[test]
    fn toolbar_actions_fit_small_and_large_windows() {
        for width in [320.0, 360.0, 480.0, 640.0, 900.0, 1280.0, 2048.0] {
            let ctx = egui::Context::default();
            theme::setup_theme(&ctx, ThemePreset::Light);
            let mut editor = EditorState::default();
            let mut files = ImportExportState::default();
            let mut compile = CompileUiState::default();
            let graph = GraphState::default();
            let jobs = EditorJobs::default();
            let mut intents = UiIntentBuffer::default();
            // A second frame resolves wrapped layout with the registered fonts.
            for _ in 0..2 {
                let output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 240.0),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        draw_toolbar_contents(
                            ui,
                            ToolbarState {
                                editor_state: &mut editor,
                                action_viewer: &mut ActionViewerState::default(),
                                import_export: &mut files,
                                compile_ui: &mut compile,
                                graph_state: &graph,
                                jobs: &jobs,
                                intents: &mut intents,
                                command_palette: &mut CommandPaletteState::default(),
                            },
                            theme::palette(ThemePreset::Light),
                        );
                    },
                );
                let labels = rendered_text(output);
                let file_y = labels
                    .iter()
                    .find(|(text, _)| text == "File")
                    .unwrap()
                    .1
                    .center()
                    .y;
                for label in [
                    "File",
                    "Undo",
                    "Redo",
                    "Fit",
                    "Attributes",
                    "Actions",
                    "BLOG",
                    "Compile settings",
                    "Commands",
                    "\u{f059}",
                    "View",
                    "Modules",
                    "Edit",
                    "Program",
                ] {
                    let (_, bounds) = labels
                        .iter()
                        .find(|(text, _)| text == label)
                        .unwrap_or_else(|| panic!("missing {label}"));
                    if width >= 1280.0 {
                        assert!(
                            (bounds.center().y - file_y).abs() < 4.0,
                            "{label} should share one compact row at {width}: y={}, file_y={file_y}",
                            bounds.center().y
                        );
                    }
                    assert!(
                        bounds.left() >= 0.0 && bounds.right() <= width,
                        "{label} clipped at width {width}: {bounds:?}"
                    );
                }
                let (_, bounds) = labels
                    .iter()
                    .find(|(text, _)| text.contains("Compile") && !text.contains("settings"))
                    .expect("compile button");
                assert!(
                    bounds.right() <= width,
                    "Compile clipped at {width}: {bounds:?}"
                );
            }
        }
    }

    fn rendered_text(mut output: egui::FullOutput) -> Vec<(String, egui::Rect)> {
        output.textures_delta.clear();
        let mut labels = Vec::new();
        for shape in output.shapes {
            collect_text_bounds(&shape.shape, &mut labels);
        }
        labels
    }

    fn collect_text_bounds(shape: &egui::epaint::Shape, labels: &mut Vec<(String, egui::Rect)>) {
        match shape {
            egui::epaint::Shape::Text(text) => labels.push((
                text.galley.job.text.clone(),
                egui::Rect::from_min_size(text.pos, text.galley.size()),
            )),
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_text_bounds(shape, labels);
                }
            }
            _ => {}
        }
    }
}
