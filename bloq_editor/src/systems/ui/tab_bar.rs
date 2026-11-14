//! The tab strip and the confirmation modal guarding destructive tab closes.

use super::{UiIntent, UiIntentBuffer, activated};
use crate::resources::{EditorState, EditorTabId, EditorTabs};
use crate::theme::{ThemePalette, palette};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

const TAB_HEIGHT: f32 = 28.0;
const TAB_MIN_WIDTH: f32 = 118.0;
const ADD_TAB_WIDTH: f32 = 32.0;

/// A non-empty tab whose close is awaiting confirmation, with its title for the
/// prompt.
pub(crate) struct PendingClose {
    pub(crate) id: EditorTabId,
    pub(crate) title: String,
}

/// Holds the tab currently blocked on a close-confirmation modal.
#[derive(Resource, Default)]
pub(crate) struct PendingTabClose {
    pub(crate) request: Option<PendingClose>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) quit_requested: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn request_window_close(
    mut requests: MessageReader<bevy::window::WindowCloseRequested>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    if requests.read().count() > 0 {
        intents.push(UiIntent::Quit);
    }
}

/// Draws the confirmation modal for a staged non-empty tab close. Cancelling
/// (button, backdrop, or Escape) clears the request; confirming emits the
/// `ConfirmCloseTab` intent.
pub(crate) fn draw_close_tab_confirm_system(
    mut contexts: EguiContexts,
    mut pending: ResMut<PendingTabClose>,
    mut intents: ResMut<UiIntentBuffer>,
    editor_state: Res<EditorState>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if pending.quit_requested {
        let Ok(ctx) = contexts.ctx_mut() else { return };
        let modal = egui::Modal::new(egui::Id::new("quit_confirm")).show(ctx, |ui| {
            ui.set_width(280.0);
            ui.heading("Quit Bloq Editor?");
            ui.label(
                "Autosave keeps tabs, text history and recent Undo/Redo unless an error was reported. Export important work before quitting.",
            );
            ui.horizontal(|ui| {
                if ui.button("Keep editing").clicked() {
                    pending.quit_requested = false;
                }
                if ui.button("Quit").clicked() {
                    intents.push(UiIntent::ConfirmQuit);
                    pending.quit_requested = false;
                }
            });
        });
        if modal.should_close() {
            pending.quit_requested = false;
        }
        return;
    }
    let Some(request) = pending.request.as_ref() else {
        return;
    };
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let palette = palette(editor_state.theme_preset);
    let title = request.title.clone();
    let id = request.id;

    let mut confirm = false;
    let mut cancel = false;
    let modal = egui::Modal::new(egui::Id::new("close_tab_confirm")).show(ctx, |ui| {
        ui.set_width(280.0);
        ui.label(
            egui::RichText::new(format!("Close \u{201c}{title}\u{201d}?"))
                .strong()
                .color(palette.text_bright),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Its graph, BLOG draft and editing history will be lost.")
                .small()
                .color(palette.text_primary),
        );
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.button("Cancel").clicked() {
                cancel = true;
            }
            if ui
                .add(
                    egui::Button::new(egui::RichText::new("Close").color(egui::Color32::WHITE))
                        .fill(palette.accent_error),
                )
                .clicked()
            {
                confirm = true;
            }
        });
    });
    if modal.should_close() {
        cancel = true;
    }

    if confirm {
        intents.push(UiIntent::ConfirmCloseTab(id.get()));
        pending.request = None;
    } else if cancel {
        pending.request = None;
    }
}

/// Draws the tab strip and queues select, rename, close, and new-tab intents.
pub(crate) fn draw_tab_bar(
    ui: &mut egui::Ui,
    tabs: &mut EditorTabs,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let active_changed = ui.data_mut(|data| {
        let id = egui::Id::new("visible_editor_tab");
        let previous = data.get_temp::<u64>(id);
        data.insert_temp(id, tabs.active.get());
        previous != Some(tabs.active.get())
    });
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
            let width =
                (ui.available_width() - ADD_TAB_WIDTH - ui.spacing().item_spacing.x).max(0.0);
            egui::ScrollArea::horizontal()
                .id_salt("editor_tab_bar_scroll")
                .max_width(width)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(2.0, 0.0);
                        for tab in &mut tabs.tabs {
                            ui.push_id(tab.id.get(), |ui| {
                                let active = tab.id == tabs.active;
                                let cancel = tab.renaming
                                    && ui.input_mut(|input| {
                                        input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)
                                    });
                                let response = if tab.renaming {
                                    ui.add_sized(
                                        [TAB_MIN_WIDTH, TAB_HEIGHT],
                                        egui::TextEdit::singleline(&mut tab.rename_buffer)
                                            .id(rename_field_id(tab.id))
                                            .font(egui::TextStyle::Small),
                                    )
                                } else {
                                    ui.add_sized(
                                        [TAB_MIN_WIDTH, TAB_HEIGHT],
                                        egui::Button::new(egui::RichText::new(&tab.title).small())
                                            .selected(active)
                                            .truncate()
                                            .fill(if active {
                                                palette.bg_surface
                                            } else {
                                                egui::Color32::TRANSPARENT
                                            })
                                            .corner_radius(4)
                                            .stroke(egui::Stroke::NONE),
                                    )
                                    .on_hover_text(format!("{}\nDouble-click to rename", tab.title))
                                };
                                if active && active_changed {
                                    response.scroll_to_me(None);
                                }
                                if tab.renaming {
                                    if cancel {
                                        intents.push(UiIntent::CancelRenameTab(tab.id.get()));
                                        response.surrender_focus();
                                    } else if response.lost_focus()
                                        || ui.input(|input| input.key_pressed(egui::Key::Enter))
                                    {
                                        intents.push(UiIntent::FinishRenameTab(tab.id.get()));
                                    }
                                } else if response.double_clicked() {
                                    intents.push(UiIntent::BeginRenameTab(tab.id.get()));
                                } else if activated(&response) {
                                    intents.push(UiIntent::SelectTab(tab.id.get()));
                                }
                                let close = ui
                                    .add_sized(
                                        [28.0, TAB_HEIGHT],
                                        egui::Button::new("\u{f00d}")
                                            .fill(if active {
                                                palette.bg_surface
                                            } else {
                                                egui::Color32::TRANSPARENT
                                            })
                                            .stroke(egui::Stroke::NONE),
                                    )
                                    .on_hover_text(format!("Close {}", tab.title));
                                close.widget_info(|| {
                                    egui::WidgetInfo::labeled(
                                        egui::WidgetType::Button,
                                        true,
                                        format!("Close {}", tab.title),
                                    )
                                });
                                if active {
                                    ui.painter().hline(
                                        response.rect.left()..=close.rect.right(),
                                        response.rect.bottom() - 1.0,
                                        egui::Stroke::new(2.0, palette.accent_primary),
                                    );
                                }
                                if close.clicked() {
                                    intents.push(UiIntent::RequestCloseTab(tab.id.get()));
                                }
                            });
                        }
                    });
                });
            if ui
                .add_sized([ADD_TAB_WIDTH, TAB_HEIGHT], egui::Button::new("+"))
                .on_hover_text("New tab")
                .clicked()
            {
                intents.push(UiIntent::NewTab);
            }
        });
    });
}

pub(super) fn rename_field_id(tab: EditorTabId) -> egui::Id {
    egui::Id::new(("tab_rename", tab.get()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_cancels_focused_rename_without_committing() {
        let ctx = egui::Context::default();
        let mut tabs = EditorTabs::default();
        let active = tabs.active;
        let title = tabs.tabs[0].title.clone();
        tabs.begin_rename(active);
        tabs.tabs[0].rename_buffer = "Discard this edit".into();
        ctx.memory_mut(|memory| memory.request_focus(rename_field_id(active)));
        let mut intents = UiIntentBuffer::default();
        for events in [
            vec![],
            vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        ] {
            ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    draw_tab_bar(
                        ui,
                        &mut tabs,
                        &mut intents,
                        palette(crate::theme::ThemePreset::Light),
                    );
                },
            )
            .drop_without_applying_deltas();
        }
        let pending: Vec<_> = intents.drain().collect();
        assert!(
            pending.iter().any(
                |intent| matches!(intent, UiIntent::CancelRenameTab(id) if *id == active.get())
            )
        );
        assert!(
            !pending
                .iter()
                .any(|intent| matches!(intent, UiIntent::FinishRenameTab(_)))
        );
        tabs.cancel_rename(active);
        assert_eq!(tabs.tabs[0].title, title);
    }
}
