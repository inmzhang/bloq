//! The keyboard-shortcut reference window.

use crate::resources::EditorState;
use crate::theme;
use bevy_egui::egui;

// Keep in sync with the key handlers in `systems/input/mod.rs` and `systems/input/placement.rs` (and the Bloq-mode
// shortcuts in `systems/ui/circuit_viewer.rs`); those are the source of truth.
const HELP_BINDINGS: &[(&str, &str)] = &[
    ("Command Palette", "Ctrl + Shift + P / F1"),
    ("Rotate Camera", "Middle Mouse Drag / Alt + Left Drag"),
    ("Pan Camera", "Right Drag / Shift + Alt + Left Drag"),
    ("Zoom Camera", "Mouse Wheel"),
    ("Undo", "Ctrl + Z"),
    ("Redo", "Ctrl + Shift + Z"),
    ("Copy Graph as BLOG", "Ctrl + C (View)"),
    ("Paste BLOG in New Tab", "Ctrl + V (View)"),
    ("Duplicate Selection", "Ctrl + D (View)"),
    ("Plane Height Down", "J"),
    ("Plane Height Up", "K"),
    ("Toggle View / Edit", "Space"),
    ("Module workspace", "M"),
    ("BLOG Buffer", "B"),
    ("BLOG Completion", "Tab / Shift + Tab (BLOG)"),
    ("Accept / Dismiss Completion", "Enter / Esc (BLOG)"),
    ("BLOG Format", "Format button (BLOG)"),
    ("Pipe Placement Tool", "P"),
    (
        "Extend Pipe To Empty Endpoint",
        "P, click block, click empty adjacent cell",
    ),
    (
        "Extend / Follow Existing Pipe",
        "W/S: +Y/-Y; A/D: -X/+X; Up/Down: +Z/-Z (Pipe tool)",
    ),
    ("Repeat Last Pipe Placement", "R"),
    ("Hadamard Pipe Placement", "Toolbar checkbox or hold H"),
    ("Program Mode", "C"),
    ("ZX View", "X"),
    ("Prev/Next Node Moment", "Q / E (Program)"),
    ("Prev/Next Reset Moment", "Shift + Q / E (Program)"),
    ("Select / Toggle Select", "Click / Shift+Click / Ctrl+Click"),
    ("Box Select", "Drag empty space; Shift+Drag adds"),
    (
        "Move Selection / Module",
        "Left-drag object or X/Y/Z handle",
    ),
    ("Cancel Move", "Esc while dragging"),
    ("Select All Elements", "Ctrl + A"),
    ("Clear Selection", "Esc (viewport)"),
    ("Delete Selected", "Delete / Backspace"),
    (
        "Translate Graph / Selection",
        "< / > / ^ / v / z< / z> (View)",
    ),
    ("Rotate Graph / Selection CCW / CW", "r / R (View)"),
    ("Fill Ports / Cycle Filled Variants", "F"),
    (
        "Edit Target Attributes",
        "Attributes button / Enter / Double-Click (View or Edit)",
    ),
    ("Actions Window (action DAG)", "A"),
    (
        "Add An Action",
        "Actions window: + Measure / + Resolve / + Let / \u{2026}",
    ),
    (
        "Edit Or Remove An Action",
        "Actions window: click a node, then Edit / Remove",
    ),
    (
        "Commit / Cancel Action Draft",
        "Enter / Esc in a draft field; Add/Edit or Cancel buttons",
    ),
    ("Cancel Action Pick", "Esc (viewport)"),
    (
        "Apply / Cancel Attributes",
        "Enter / Esc in Tag or Color; OK or Cancel buttons",
    ),
    ("Toggle Stabilizers", "S (View)"),
    ("Prev/Next Stabilizer", "Q / E (Stabilizer View)"),
];

/// Draws the shortcut help window when `editor_state.show_help_window` is set.
pub(crate) fn draw_help_window(
    ctx: &egui::Context,
    editor_state: &mut EditorState,
    query: &mut String,
) {
    if editor_state.show_help_window {
        let palette = theme::palette(editor_state.theme_preset);
        let viewport = ctx.content_rect();
        let available = (viewport.size() - egui::vec2(48.0, 64.0)).max(egui::vec2(100.0, 100.0));
        egui::Window::new("Shortcuts")
            .open(&mut editor_state.show_help_window)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(600.0, 460.0).min(available))
            .min_size(egui::vec2(240.0, 160.0).min(available))
            .max_size(available)
            .constrain_to(viewport)
            .frame(
                egui::Frame::new()
                    .fill(palette.bg_panel)
                    .stroke(egui::Stroke::new(1.0, palette.border))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| {
                ui.add(
                    egui::TextEdit::singleline(query)
                        .hint_text("Find an action or shortcut...")
                        .desired_width(f32::INFINITY),
                );
                ui.small("Mac: use Command instead of Ctrl.");
                ui.collapsing("Editing tips", |ui| {
                    ui.label("Edit → click an empty cell to place → P to connect. Space switches to selection.");
                    ui.label("With the block tool, click an existing block to select it; double-click opens attributes. Placement does not replace it.");
                    ui.label(
                        "Select, then Delete/Backspace to remove. Connected pipes are removed with a block; Undo restores them. Middle mouse only orbits the camera.",
                    );
                    ui.label("File Save/Download exports the graph. BLOG Save/Download draft preserves exact text, including comments and invalid or unfinished input.");
                    ui.label("Graph and text conflicts require an explicit choice; Text history keeps up to 20 recent versions within 8 MiB per tab. Save draft for permanent recovery.");
                });
                let query = query.trim().to_lowercase();
                let mut count = 0;
                egui::ScrollArea::vertical()
                    .id_salt("help_shortcuts_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("help_grid")
                            .striped(true)
                            .num_columns(2)
                            .spacing([16.0, 8.0])
                            .min_col_width(0.0)
                            .max_col_width(((ui.available_width() - 24.0) / 2.0).max(40.0))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new("Action")
                                        .strong()
                                        .color(palette.text_bright),
                                );
                                ui.label(
                                    egui::RichText::new("Keybinding")
                                        .strong()
                                        .color(palette.text_bright),
                                );
                                ui.end_row();

                                for &(action, key) in HELP_BINDINGS {
                                    if !query.is_empty()
                                        && !format!("{action} {key}")
                                            .to_lowercase()
                                            .contains(&query)
                                    {
                                        continue;
                                    }
                                    count += 1;
                                    ui.add(egui::Label::new(action).wrap());
                                    ui.add(
                                        egui::Label::new(egui::RichText::new(key).code()).wrap(),
                                    );
                                    ui.end_row();
                                }
                            });
                        if count == 0 {
                            ui.label("No matching shortcuts.");
                        }
                    });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::HELP_BINDINGS;
    use crate::systems::ui::commands::all_commands;

    /// The palette and this table both show keybinding labels; a label in only
    /// one of them is drift the user can see.
    #[test]
    fn registry_keybinds_appear_in_the_help_table() {
        for command in all_commands() {
            let Some(keys) = command.keybind() else {
                continue;
            };
            assert!(
                HELP_BINDINGS.iter().any(|(_, binding)| *binding == keys),
                "{} shows keybind {keys}, missing from HELP_BINDINGS",
                command.search_label()
            );
        }
    }
}
