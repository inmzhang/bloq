//! The attribute-editor window for a selected block or pipe (kind, Port role
//! and color, tag, Hadamard flag).

use super::intents::{ElementEditIntent, UiIntent, UiIntentBuffer};
use super::{activated, field_owns_shortcuts};
use crate::components::GraphElement;
use crate::theme::{self, ThemePreset};
use bevy::prelude::Resource;
use bevy_egui::egui;
use bloq_graph::{Basis, BlockKind, CubeKind, PortRole, RGBA, WalkingBoundaryKind};

pub(crate) const DEFAULT_PORT_RGB: [u8; 3] =
    [RGBA::PORT_GRAY.r, RGBA::PORT_GRAY.g, RGBA::PORT_GRAY.b];

/// State of the attribute-editor window: which element is being edited and the
/// in-progress field buffers.
#[derive(Resource, Clone)]
pub(crate) struct TargetState {
    pub(crate) open_window: bool,
    pub(crate) focus_field: bool,
    /// Retains input ownership through the frame that closes the dialog.
    pub(crate) capture_input: bool,
    pub(crate) target: Option<GraphElement>,
    pub(crate) tag_buffer: String,
    pub(crate) block_kind_buffer: BlockKind,
    pub(crate) port_color_hex_buffer: String,
    pub(crate) port_role_buffer: PortRole,
    pub(crate) pipe_hadamard_buffer: bool,
    /// Ignores an Enter that is still held from opening the window, so it does
    /// not immediately re-confirm and close.
    pub(crate) suppress_enter_confirm_until_release: bool,
}

impl Default for TargetState {
    fn default() -> Self {
        Self {
            open_window: false,
            focus_field: false,
            capture_input: false,
            target: None,
            tag_buffer: String::new(),
            block_kind_buffer: BlockKind::Cube(CubeKind::XZZ),
            port_color_hex_buffer: format_rgb_hex(DEFAULT_PORT_RGB),
            port_role_buffer: PortRole::Auto,
            pipe_hadamard_buffer: false,
            suppress_enter_confirm_until_release: false,
        }
    }
}

/// Draws the attribute editor for the targeted element, queuing an
/// [`ElementEditIntent`] when the user commits changes.
pub(crate) fn draw_edit_element_window(
    ctx: &egui::Context,
    tab_id: crate::resources::EditorTabId,
    target_state: &mut TargetState,
    theme_preset: ThemePreset,
    intents: &mut UiIntentBuffer,
) {
    target_state.capture_input = target_state.open_window;
    if target_state.open_window {
        let palette = theme::palette(theme_preset);
        egui::Window::new("Edit Element")
            .id(egui::Id::new(("edit_element", tab_id)))
            .collapsible(false)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(palette.bg_panel)
                    .stroke(egui::Stroke::new(1.0, palette.border))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| {
                let mut block_color = None;
                let mut owns_keyboard = false;
                if let Some(GraphElement::Block(_)) = target_state.target {
                    theme::section_header(ui, palette, "Kind");
                    theme::section_frame(ui, palette, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for kind in BlockKind::all_kinds() {
                                ui.radio_value(
                                    &mut target_state.block_kind_buffer,
                                    kind,
                                    kind.to_string(),
                                );
                            }
                        });
                    });
                    if let BlockKind::Walking(kind) = target_state.block_kind_buffer {
                        theme::section_header(ui, palette, "Walking Boundary");
                        theme::section_frame(ui, palette, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for boundary in WalkingBoundaryKind::ALL {
                                    let candidate =
                                        BlockKind::Walking(kind.with_boundary(boundary));
                                    ui.radio_value(
                                        &mut target_state.block_kind_buffer,
                                        candidate,
                                        boundary.to_string(),
                                    );
                                }
                            });
                            let movement = kind.movement();
                            ui.label(format!("Movement: ({}, {})", movement.x, movement.y));
                        });
                    }
                    if let BlockKind::PatchRotation(kind) = target_state.block_kind_buffer {
                        theme::section_header(ui, palette, "Patch Rotation Basis");
                        theme::section_frame(ui, palette, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for basis in [Basis::X, Basis::Z] {
                                    let candidate =
                                        BlockKind::PatchRotation(kind.with_basis(basis));
                                    ui.radio_value(
                                        &mut target_state.block_kind_buffer,
                                        candidate,
                                        basis.to_string(),
                                    );
                                }
                            });
                            let movement = kind.movement();
                            ui.label(format!("Movement: ({}, {})", movement.x, movement.y));
                        });
                    }
                    ui.add_space(4.0);
                }

                if let Some(GraphElement::Pipe(_, _)) = target_state.target {
                    theme::section_header(ui, palette, "Pipe");
                    theme::section_frame(ui, palette, |ui| {
                        ui.checkbox(&mut target_state.pipe_hadamard_buffer, "Hadamard");
                    });
                    ui.add_space(4.0);
                }

                let editing_port = matches!(target_state.target, Some(GraphElement::Block(_)))
                    && target_state.block_kind_buffer.is_port();
                theme::section_header(ui, palette, "Metadata");
                egui::Grid::new("edit_metadata_grid")
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        if editing_port {
                            ui.label("Role:");
                            ui.horizontal(|ui| {
                                for role in [
                                    PortRole::Auto,
                                    PortRole::Input,
                                    PortRole::Output,
                                    PortRole::Multiplex,
                                ] {
                                    ui.radio_value(
                                        &mut target_state.port_role_buffer,
                                        role,
                                        role.to_string(),
                                    );
                                }
                            });
                            ui.end_row();
                            ui.label("Color:");
                            block_color = draw_port_color_editor(
                                ui,
                                &mut target_state.port_color_hex_buffer,
                                &mut owns_keyboard,
                            );
                            ui.end_row();
                        }
                        ui.label("Tag:");
                        let tag = ui.text_edit_singleline(&mut target_state.tag_buffer);
                        if std::mem::take(&mut target_state.focus_field) {
                            tag.request_focus();
                        }
                        owns_keyboard |= field_owns_shortcuts(ui, tag.id);
                        ui.end_row();
                    });
                ui.add_space(12.0);
                ui.scope_builder(
                    egui::UiBuilder::new().id(ui.make_persistent_id("edit-element-actions")),
                    |ui| {
                        ui.horizontal(|ui| {
                            let can_confirm = !editing_port || block_color.is_some();
                            let ok_clicked =
                                activated(&ui.add_enabled(can_confirm, egui::Button::new("OK")));
                            let enter_confirmed = ui.input(|i| {
                                enter_confirm_allowed(
                                    &mut target_state.suppress_enter_confirm_until_release,
                                    i.key_down(egui::Key::Enter),
                                    i.key_pressed(egui::Key::Enter),
                                )
                            });
                            if (ok_clicked || (can_confirm && owns_keyboard && enter_confirmed))
                                && let Some(target) = target_state.target
                            {
                                intents.push(UiIntent::ApplyElementEdit(ElementEditIntent {
                                    target,
                                    block_kind: target_state.block_kind_buffer,
                                    block_color,
                                    port_role: target_state.port_role_buffer,
                                    tag: target_state.tag_buffer.clone(),
                                    pipe_hadamard: target_state.pipe_hadamard_buffer,
                                }));
                            }
                            if activated(&ui.button("Cancel"))
                                || (owns_keyboard
                                    && ui.input_mut(|i| {
                                        i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)
                                    }))
                            {
                                intents.push(UiIntent::CloseElementEdit);
                                target_state.open_window = false;
                            }
                        });
                    },
                );
            });
    }
}

fn draw_port_color_editor(
    ui: &mut egui::Ui,
    hex: &mut String,
    owns_keyboard: &mut bool,
) -> Option<[u8; 3]> {
    let mut rgb = parse_rgb_hex(hex).unwrap_or(DEFAULT_PORT_RGB);
    ui.horizontal(|ui| {
        if ui.color_edit_button_srgb(&mut rgb).changed() {
            *hex = format_rgb_hex(rgb);
        }
        let field = ui
            .add(
                egui::TextEdit::singleline(hex)
                    .char_limit(6)
                    .desired_width(64.0)
                    .font(egui::TextStyle::Monospace),
            )
            .on_hover_text("Six hex digits; copy or paste this value after color=");
        *owns_keyboard |= field_owns_shortcuts(ui, field.id);
    });
    let color = parse_rgb_hex(hex);
    if color.is_none() {
        ui.colored_label(egui::Color32::RED, "Expected six hex digits");
    }
    color
}

fn parse_rgb_hex(text: &str) -> Option<[u8; 3]> {
    if text.len() != 6 {
        return None;
    }
    let [_, r, g, b] = u32::from_str_radix(text, 16).ok()?.to_be_bytes();
    Some([r, g, b])
}

pub(crate) fn format_rgb_hex([r, g, b]: [u8; 3]) -> String {
    format!("{r:02x}{g:02x}{b:02x}")
}

fn enter_confirm_allowed(
    suppress_until_release: &mut bool,
    enter_down: bool,
    enter_pressed: bool,
) -> bool {
    if *suppress_until_release {
        if !enter_down {
            *suppress_until_release = false;
        }
        return false;
    }
    enter_pressed
}

#[cfg(test)]
mod tests {
    use super::{enter_confirm_allowed, format_rgb_hex, parse_rgb_hex};

    #[test]
    fn attribute_shortcuts_belong_to_its_fields_not_other_editors() {
        use super::*;
        for own_field in [false, true] {
            for key in [egui::Key::Enter, egui::Key::Escape] {
                let ctx = egui::Context::default();
                theme::setup_theme(&ctx, ThemePreset::default());
                let mut target = TargetState {
                    open_window: true,
                    focus_field: own_field,
                    target: Some(GraphElement::Block(glam::IVec3::ZERO)),
                    ..Default::default()
                };
                let mut intents = UiIntentBuffer::default();
                let mut outside = String::from("unrelated draft");
                for first in [true, false] {
                    let events = if first {
                        vec![]
                    } else {
                        vec![egui::Event::Key {
                            key,
                            physical_key: None,
                            pressed: true,
                            repeat: false,
                            modifiers: egui::Modifiers::NONE,
                        }]
                    };
                    let mut output = ctx.run_ui(
                        egui::RawInput {
                            events,
                            ..Default::default()
                        },
                        |ui| {
                            let field = ui.text_edit_singleline(&mut outside);
                            if first && !own_field {
                                field.request_focus();
                            }
                            draw_edit_element_window(
                                &ctx,
                                crate::resources::EditorTabId::new(1),
                                &mut target,
                                ThemePreset::default(),
                                &mut intents,
                            );
                        },
                    );
                    output.textures_delta.clear();
                }
                let queued = intents.drain().collect::<Vec<_>>();
                if own_field {
                    assert_eq!(queued.len(), 1);
                    assert!(match key {
                        egui::Key::Enter => matches!(queued[0], UiIntent::ApplyElementEdit(_)),
                        _ => matches!(queued[0], UiIntent::CloseElementEdit),
                    });
                } else {
                    assert!(
                        queued.is_empty(),
                        "unrelated {key:?} must not affect attributes"
                    );
                }
                assert_eq!(target.open_window, !own_field || key != egui::Key::Escape);
                assert!(target.capture_input);
            }
        }
    }

    #[test]
    fn opening_enter_is_ignored_until_enter_is_released() {
        let mut suppress = true;

        assert!(!enter_confirm_allowed(&mut suppress, true, true));
        assert!(suppress);
        assert!(!enter_confirm_allowed(&mut suppress, false, false));
        assert!(!suppress);
        assert!(enter_confirm_allowed(&mut suppress, true, true));
    }

    #[test]
    fn port_color_hex_roundtrips() {
        let color = parse_rgb_hex("Eb4034").expect("valid RGB");
        assert_eq!(color, [0xeb, 0x40, 0x34]);
        assert_eq!(format_rgb_hex(color), "eb4034");
        assert_eq!(parse_rgb_hex("12345"), None);
        assert_eq!(parse_rgb_hex("gg0000"), None);
    }
}
