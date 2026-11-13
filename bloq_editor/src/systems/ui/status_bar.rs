//! The bottom status bar: graph counts, tool, validation/compile status, and
//! contextual shortcut hints.

use crate::components::GraphElement;
use crate::resources::{
    ActionEditState, CompileExecutionState, CompileUiState, EditorMode, EditorState, EditorTabId,
    GraphState, GraphUiSummary, Notifications, PlacementTool, ValidationState,
};
use crate::systems::jobs::EditorJobs;
use crate::theme::{self, ThemePalette, with_alpha};
use bevy_egui::egui;
use bloq_graph::BlockKind;

type ShortcutHint = (&'static str, &'static str);
type ShortcutHints = Vec<ShortcutHint>;

/// Draws the bottom status bar into `root`.
pub(crate) fn draw_status_bar(
    root: &mut egui::Ui,
    editor_state: &EditorState,
    graph_state: &GraphState,
    graph_ui_summary: &GraphUiSummary,
    compile_ui: &CompileUiState,
    tab_id: EditorTabId,
    action_edit: &ActionEditState,
    jobs: &EditorJobs,
    notifications: &mut Notifications,
    palette: &ThemePalette,
) {
    egui::Panel::bottom("editor_status_bar")
        .resizable(false)
        .frame(
            egui::Frame::new()
                .fill(palette.bg_panel)
                .stroke(egui::Stroke::new(1.0, palette.border))
                .inner_margin(egui::Margin::symmetric(12, 6)),
        )
        .show(root, |ui| {
            let total_width = ui.available_width().max(0.0);
            let row_height = ui.available_height();

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                super::notifications::draw_notification_button(ui, notifications, palette);
                draw_separator(ui, palette);
                let available_width = ui.available_width().max(0.0);
                let hint_width = if available_width < 560.0 {
                    (available_width * 0.72).max(180.0).min(available_width)
                } else {
                    (total_width * 0.52)
                        .clamp(260.0, 680.0)
                        .min(available_width)
                };
                let status_width =
                    (available_width - hint_width - ui.spacing().item_spacing.x).max(0.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(hint_width, row_height),
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        // egui otherwise reserves only the hints' actual content width.
                        ui.set_min_width(hint_width);
                        draw_shortcut_hints_inline(
                            ui,
                            editor_state,
                            action_edit,
                            palette,
                            hint_width,
                        );
                    },
                );
                ui.allocate_ui_with_layout(
                    egui::vec2(status_width, row_height),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        egui::ScrollArea::horizontal()
                            .id_salt("editor_status_bar_scroll")
                            .auto_shrink([false, true])
                            .max_width(status_width)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(16.0, 0.0);
                                    draw_status_chips(
                                        ui,
                                        editor_state,
                                        graph_state,
                                        graph_ui_summary,
                                        compile_ui,
                                        tab_id,
                                        jobs,
                                        palette,
                                    );
                                });
                            });
                    },
                );
            });
        });
}

fn draw_shortcut_hints_inline(
    ui: &mut egui::Ui,
    editor_state: &EditorState,
    action_edit: &ActionEditState,
    palette: &ThemePalette,
    available_width: f32,
) {
    // An armed action pick replaces the hint row: it is a prompt the user is
    // mid-way through answering, not one shortcut among many.
    if let Some(kind) = action_edit.picking() {
        let hints = vec![("Click", kind.pick_prompt()), ("Esc", "Cancel")];
        draw_hint_row(
            ui,
            &visible_shortcut_hints(&hints, available_width),
            palette,
        );
        return;
    }
    let hints = visible_shortcut_hints(&mode_shortcut_hints(editor_state), available_width);
    draw_hint_row(ui, &hints, palette);
}

fn draw_hint_row(ui: &mut egui::Ui, hints: &[ShortcutHint], palette: &ThemePalette) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
        for (index, &(key, action)) in hints.iter().enumerate() {
            if index > 0 {
                draw_separator(ui, palette);
            }
            draw_shortcut_hint_rtl(ui, key, action, palette);
        }
    });
}

fn draw_separator(ui: &mut egui::Ui, palette: &ThemePalette) {
    ui.label(
        egui::RichText::new("|")
            .small()
            .color(with_alpha(palette.text_dim, 120)),
    );
}

fn draw_shortcut_hint_rtl(ui: &mut egui::Ui, key: &str, action: &str, palette: &ThemePalette) {
    ui.label(
        egui::RichText::new(action)
            .small()
            .color(shortcut_action_text(palette)),
    );
    egui::Frame::new()
        .fill(shortcut_key_fill(palette))
        .stroke(egui::Stroke::new(1.0, shortcut_key_border(palette)))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(key)
                    .monospace()
                    .small()
                    .color(shortcut_key_text(palette)),
            );
        });
}

fn visible_shortcut_hints(hints: &ShortcutHints, available_width: f32) -> ShortcutHints {
    let mut width = 0.0;
    let mut visible = ShortcutHints::new();
    for &hint in hints {
        let next_width = shortcut_hint_estimated_width(hint.0, hint.1)
            + if visible.is_empty() {
                0.0
            } else {
                shortcut_separator_estimated_width()
            };
        if width + next_width > available_width {
            continue;
        }
        width += next_width;
        visible.push(hint);
    }
    visible
}

fn shortcut_hint_estimated_width(key: &str, action: &str) -> f32 {
    // Conservative estimate for egui small monospace keycaps plus action text.
    key.chars().count() as f32 * 7.0 + action.chars().count() as f32 * 6.5 + 34.0
}

fn shortcut_separator_estimated_width() -> f32 {
    18.0
}

fn mode_shortcut_hints(editor_state: &EditorState) -> ShortcutHints {
    match editor_state.mode {
        EditorMode::View => {
            let mut hints = ShortcutHints::new();
            if editor_state.selection_count() > 0 {
                if editor_state.selection_count() == 1 {
                    hints.push(("Enter", "Attributes"));
                }
                hints.push(("Del/Backspace", "Delete"));
            } else {
                hints.push(("B", "BLOG"));
                hints.push(("P", "Pipe"));
                hints.push(("C", "Program"));
                hints.push(("X", "ZX"));
            }
            hints.extend([
                ("Click", "Select"),
                ("Shift+Click", "Add/remove"),
                ("Drag", "Box select"),
                ("Ctrl+A", "Select all"),
                ("J/K", "Plane"),
            ]);
            if editor_state.show_stabilizers && !editor_state.stabilizers.is_empty() {
                hints.push(("Q/E", "Stabilizer"));
            } else {
                hints.push(("S", "Stabilizers"));
            }
            hints.extend([("A", "Actions"), ("Double-click", "Attributes")]);
            hints
        }
        EditorMode::Module => vec![
            ("Click", "Inspect instance"),
            ("M / Space", "View"),
            ("B", "BLOG"),
        ],
        EditorMode::Edit => {
            let mut hints = match editor_state.placement_tool {
                PlacementTool::Block => {
                    if editor_state.block_kind.is_walking() {
                        vec![
                            ("Click", "Place walking start"),
                            ("P", "Pipe"),
                            ("Space/V", "Select"),
                            ("J/K", "Plane"),
                            ("Esc", "Cancel"),
                        ]
                    } else {
                        vec![
                            ("Click", "Place block"),
                            ("Double-click", "Attributes"),
                            ("P", "Pipe"),
                            ("Space/V", "Select"),
                            ("J/K", "Plane"),
                        ]
                    }
                }
                PlacementTool::Pipe => vec![
                    ("Click", "Place pipe endpoint"),
                    ("Double-click", "Attributes"),
                    ("W/A/S/D", "Extend/follow"),
                    ("Up/Down", "Pipe top/bottom"),
                    ("R", "Repeat pipe"),
                    ("B", "BLOG"),
                    ("H/checkbox", "Hadamard"),
                    ("Space/V", "Select"),
                    ("J/K", "Plane"),
                    ("Esc", "Cancel"),
                ],
            };
            if editor_state.walking_start.is_some() {
                hints.insert(1, ("Click hint", "Finish walking"));
            } else if editor_state.pipe_start.is_some() {
                hints.insert(1, ("Click adjacent", "Finish pipe"));
            }
            hints
        }
        EditorMode::Bloq => vec![
            ("Q/E", "Node moment"),
            ("Mouse wheel", "Zoom"),
            ("Space/V", "Select"),
            ("B", "BLOG"),
            ("P", "Pipe"),
        ],
    }
}

fn draw_status_chips(
    ui: &mut egui::Ui,
    editor_state: &EditorState,
    graph_state: &GraphState,
    graph_ui_summary: &GraphUiSummary,
    compile_ui: &CompileUiState,
    tab_id: EditorTabId,
    jobs: &EditorJobs,
    palette: &ThemePalette,
) {
    // Put actionable state before the scrollable graph details.
    if let Some((stage, seconds)) = jobs.compilation_progress(tab_id, graph_state.revision) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                egui::RichText::new(format!("Compiling: {stage} · {seconds}s"))
                    .small()
                    .color(palette.accent_warn),
            );
        });
    } else if compile_ui.compile_state != CompileExecutionState::Idle
        && !(compile_ui.compile_state == CompileExecutionState::Pending
            && jobs.compilation_running())
    {
        draw_compile_chip(
            ui,
            compile_ui.compile_state,
            &compile_ui.compile_message,
            palette,
        );
    }
    if compile_ui.validation_state != ValidationState::Unknown {
        draw_validation_chip(
            ui,
            compile_ui.validation_state,
            &compile_ui.validation_message,
            palette,
        );
    }
    if editor_state.mode == EditorMode::Edit {
        match editor_state.placement_tool {
            PlacementTool::Block => theme::status_chip(
                ui,
                "\u{f1b2}",
                placement_block_label(editor_state.block_kind),
                palette.accent_secondary,
            ),
            PlacementTool::Pipe => {
                theme::status_chip(ui, "\u{f0c1}", "Pipe", palette.accent_secondary);
            }
        }
    }
    theme::status_chip(
        ui,
        "\u{f07d}",
        format_args!("Z={}", editor_state.plane_height),
        palette.text_primary,
    );
    if editor_state.mode == EditorMode::View {
        theme::status_chip(
            ui,
            "\u{f0ec}",
            format_args!("{}", editor_state.transform_axis),
            palette.accent_secondary,
        );
    }
    if editor_state.showing_stabilizers() {
        let label = if let Some((_, generator)) = &editor_state.action_stabilizer {
            format!(
                "Measure {}",
                generator.measurement_name().unwrap_or("surface")
            )
        } else {
            format!(
                "Stab {}/{}  Q/E",
                editor_state.current_stabilizer_index + 1,
                editor_state.stabilizers.len()
            )
        };
        theme::status_chip(ui, "\u{f074}", label, palette.accent_warn);
    }
    theme::status_chip(
        ui,
        "\u{f1b3}",
        format_args!("{} blocks", graph_ui_summary.block_count),
        palette.text_primary,
    );
    theme::status_chip(
        ui,
        "\u{f0c1}",
        format_args!("{} pipes", graph_ui_summary.pipe_count),
        palette.text_primary,
    );

    if editor_state.selection_count() > 0 {
        theme::status_chip(
            ui,
            "\u{f245}",
            selected_element_label(editor_state),
            palette.accent_primary,
        );
    }

    if editor_state.selection_count() == 0
        && let Some(element) = editor_state.hovered_element
    {
        theme::status_chip(
            ui,
            "\u{f140}",
            hovered_element_label(graph_state, element),
            palette.text_primary,
        );
    }

    if jobs.is_busy() {
        theme::status_chip(ui, "\u{f110}", "jobs", palette.accent_warn);
    }
}

fn draw_validation_chip(
    ui: &mut egui::Ui,
    state: ValidationState,
    message: &str,
    palette: &ThemePalette,
) {
    let (icon, color) = match state {
        ValidationState::Unknown => ("\u{f059}", palette.text_dim),
        ValidationState::Pending => ("\u{f252}", palette.accent_warn),
        ValidationState::Passed => ("\u{f058}", palette.success),
        ValidationState::Failed => ("\u{f057}", palette.accent_error),
    };
    let response = ui
        .scope(|ui| theme::status_chip(ui, icon, format_args!("Validation: {state}"), color))
        .response;
    if state == ValidationState::Failed && !message.is_empty() {
        response.on_hover_text(message);
    }
}

fn draw_compile_chip(
    ui: &mut egui::Ui,
    state: CompileExecutionState,
    message: &str,
    palette: &ThemePalette,
) {
    let (icon, color) = match state {
        CompileExecutionState::Idle => ("\u{f0c7}", palette.text_dim),
        CompileExecutionState::Pending => ("\u{f252}", palette.accent_warn),
        CompileExecutionState::Succeeded => ("\u{f058}", palette.success),
        CompileExecutionState::Failed => ("\u{f057}", palette.accent_error),
        CompileExecutionState::Cancelled => ("\u{f05e}", palette.text_dim),
    };
    let response = ui
        .scope(|ui| theme::status_chip(ui, icon, format_args!("Compile: {state}"), color))
        .response;
    if matches!(
        state,
        CompileExecutionState::Failed | CompileExecutionState::Cancelled
    ) && !message.is_empty()
    {
        response.on_hover_text(message);
    }
}

fn placement_block_label(kind: BlockKind) -> String {
    if kind.is_walking() {
        "Walking".to_string()
    } else if kind.is_patch_rotation() {
        "Patch Rotation".to_string()
    } else {
        kind.to_string()
    }
}

fn hovered_element_label(graph_state: &GraphState, element: GraphElement) -> String {
    match element {
        GraphElement::Block(pos) => match graph_state.graph.get_block_id(pos) {
            Some(id) => format!("Block {pos} · id {id}"),
            None => format!("Block {pos}"),
        },
        GraphElement::Pipe(u, v) => format!("Pipe {u}\u{f07e}{v}"),
    }
}

fn selected_element_label(editor_state: &EditorState) -> String {
    let count = editor_state.selection_count();
    format!("{count} selected")
}

fn shortcut_key_border(palette: &ThemePalette) -> egui::Color32 {
    if palette.dark_mode {
        with_alpha(palette.border, 220)
    } else {
        palette.text_bright
    }
}

fn shortcut_key_fill(palette: &ThemePalette) -> egui::Color32 {
    if palette.dark_mode {
        with_alpha(palette.bg_surface, 220)
    } else {
        palette.text_primary
    }
}

fn shortcut_key_text(palette: &ThemePalette) -> egui::Color32 {
    if palette.dark_mode {
        palette.text_bright
    } else {
        egui::Color32::WHITE
    }
}

fn shortcut_action_text(palette: &ThemePalette) -> egui::Color32 {
    palette.text_primary
}

#[cfg(test)]
mod tests {
    use super::{
        draw_status_bar, mode_shortcut_hints, shortcut_action_text, shortcut_hint_estimated_width,
        shortcut_key_fill, shortcut_key_text, shortcut_separator_estimated_width,
        visible_shortcut_hints,
    };
    use crate::components::GraphElement;
    use crate::resources::{
        ActionEditState, CompileUiState, EditorState, GraphState, GraphUiSummary,
    };
    use crate::systems::jobs::EditorJobs;
    use crate::theme::{self, ThemePreset};
    use bevy_egui::egui::{self, Color32};
    use glam::ivec3;

    #[test]
    fn shortcut_hints_render_on_status_bar_line() {
        let editor_state = EditorState::default();
        let graph_state = GraphState::default();
        let paints = render_status_bar_text_paints(&editor_state, &graph_state, 1024.0);
        let status_chip = find_text(&paints, "Z=0");
        let hint = find_text(&paints, "Select");

        assert!(
            (status_chip.pos.y - hint.pos.y).abs() < 4.0,
            "shortcut hints should share the statusline; status at {:?}, hint at {:?}",
            status_chip.pos,
            hint.pos
        );
        assert!(
            hint.pos.x > status_chip.pos.x,
            "shortcut hints should align to the right of status chips; status at {:?}, hint at {:?}",
            status_chip.pos,
            hint.pos
        );
    }

    #[test]
    fn status_bar_does_not_repeat_toolbar_mode_label() {
        let paints =
            render_status_bar_text_paints(&EditorState::default(), &GraphState::default(), 1920.0);
        assert!(!paints.iter().any(|paint| paint.text == "View"));
        find_text(&paints, "Z=0");
    }

    #[test]
    fn status_items_start_at_the_left_edge_and_notifications_have_a_separator() {
        for width in [1024.0, 1280.0, 1920.0] {
            let paints = render_status_bar_text_paints(
                &EditorState::default(),
                &GraphState::default(),
                width,
            );
            let left = paints
                .iter()
                .map(|paint| paint.pos.x)
                .fold(f32::INFINITY, f32::min);
            assert!(
                left <= 20.0,
                "{width}: status items start too far from the left edge ({left})"
            );

            let blog = find_text(&paints, "BLOG");
            let notifications = find_text(&paints, "Notifications");
            assert!(
                paints.iter().any(|paint| {
                    paint.text == "|"
                        && paint.pos.x > blog.pos.x
                        && paint.pos.x < notifications.pos.x
                }),
                "{width}: missing separator before Notifications"
            );
        }
    }

    #[test]
    fn status_bar_keeps_notifications_in_narrow_windows() {
        for width in [120.0, 200.0, 320.0] {
            let paints = render_status_bar_text_paints(
                &EditorState::default(),
                &GraphState::default(),
                width,
            );
            assert!((0.0..width).contains(&find_text(&paints, "Notifications").pos.x));
        }
    }

    #[test]
    fn narrow_shortcut_hints_fit_as_complete_pairs() {
        let hints = visible_shortcut_hints(&mode_shortcut_hints(&EditorState::default()), 320.0);
        let total_width = hints
            .iter()
            .enumerate()
            .map(|(index, &(key, action))| {
                shortcut_hint_estimated_width(key, action)
                    + if index == 0 {
                        0.0
                    } else {
                        shortcut_separator_estimated_width()
                    }
            })
            .sum::<f32>();

        assert!(!hints.is_empty());
        assert!(total_width <= 320.0);
    }

    #[test]
    fn narrow_view_mode_hints_keep_primary_context_action_visible() {
        let mut selected = EditorState::default();
        selected.select_element(GraphElement::Block(ivec3(0, 0, 0)), false);

        let unselected_hints =
            visible_shortcut_hints(&mode_shortcut_hints(&EditorState::default()), 320.0);
        let selected_hints = visible_shortcut_hints(&mode_shortcut_hints(&selected), 320.0);

        assert!(unselected_hints.contains(&("B", "BLOG")));
        assert!(selected_hints.contains(&("Del/Backspace", "Delete")));
    }

    #[test]
    fn attributes_hint_requires_one_selected_element() {
        let mut editor = EditorState::default();
        editor.select_element(GraphElement::Block(ivec3(0, 0, 0)), false);
        assert!(mode_shortcut_hints(&editor).contains(&("Enter", "Attributes")));
        editor.select_element(GraphElement::Block(ivec3(1, 0, 0)), true);
        let hints = mode_shortcut_hints(&editor);
        assert!(!hints.contains(&("Enter", "Attributes")));
        assert!(hints.contains(&("Del/Backspace", "Delete")));
    }

    #[test]
    fn interface_text_and_shortcut_colors_are_readable_in_both_themes() {
        for preset in [ThemePreset::Light, ThemePreset::GruvboxMaterial] {
            let palette = theme::palette(preset);
            assert!(contrast_ratio(shortcut_key_fill(palette), shortcut_key_text(palette)) >= 4.5);
            assert!(contrast_ratio(palette.bg_panel, shortcut_action_text(palette)) >= 4.5);
            for background in [palette.bg_panel, palette.bg_surface] {
                for text in [palette.text_primary, palette.text_dim] {
                    assert!(
                        contrast_ratio(background, text) >= 4.5,
                        "{preset:?}: {text:?} is hard to read on {background:?}"
                    );
                }
            }
        }
    }

    #[derive(Debug)]
    struct TextPaint {
        text: String,
        pos: egui::Pos2,
    }

    fn render_status_bar_text_paints(
        editor_state: &EditorState,
        graph_state: &GraphState,
        width: f32,
    ) -> Vec<TextPaint> {
        let ctx = egui::Context::default();
        let palette = theme::palette(ThemePreset::default());
        let graph_ui_summary = GraphUiSummary::from_graph_state(graph_state);
        let mut full_output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 64.0),
                )),
                ..Default::default()
            },
            |ui| {
                draw_status_bar(
                    ui,
                    editor_state,
                    graph_state,
                    &graph_ui_summary,
                    &CompileUiState::default(),
                    crate::resources::EditorTabId::new(1),
                    &ActionEditState::default(),
                    &EditorJobs::default(),
                    &mut crate::resources::Notifications::default(),
                    palette,
                );
            },
        );
        full_output.textures_delta.clear();
        let mut texts = Vec::new();
        for clipped_shape in full_output.shapes {
            collect_shape_text_paints(&clipped_shape.shape, &mut texts);
        }
        texts
    }

    fn collect_shape_text_paints(shape: &egui::epaint::Shape, texts: &mut Vec<TextPaint>) {
        match shape {
            egui::epaint::Shape::Text(text) => texts.push(TextPaint {
                text: text.galley.job.text.clone(),
                pos: text.pos,
            }),
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_shape_text_paints(shape, texts);
                }
            }
            _ => {}
        }
    }

    fn find_text<'a>(paints: &'a [TextPaint], needle: &str) -> &'a TextPaint {
        paints
            .iter()
            .find(|paint| paint.text.contains(needle))
            .unwrap_or_else(|| {
                panic!("expected rendered text containing {needle:?}, got {paints:?}")
            })
    }

    fn contrast_ratio(a: Color32, b: Color32) -> f32 {
        let lighter = relative_luminance(a).max(relative_luminance(b));
        let darker = relative_luminance(a).min(relative_luminance(b));
        (lighter + 0.05) / (darker + 0.05)
    }

    fn relative_luminance(color: Color32) -> f32 {
        fn channel(value: u8) -> f32 {
            let value = f32::from(value) / 255.0;
            if value <= 0.039_28 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        }

        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
    }
}
