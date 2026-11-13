//! The floating placement dock: block-kind and pipe thumbnails the user picks
//! from while in Edit mode.

use std::borrow::Cow;

use super::activated;
use super::intents::{UiIntent, UiIntentBuffer};
use super::thumbnail::paint_texture_thumbnail;
use crate::resources::{
    EditorMode, EditorState, PlacementPreviewKind, PlacementTool, ThumbnailTextures,
};
use crate::systems::thumbnails::ThumbnailKey;
use crate::theme::{ThemePalette, with_alpha};
use bevy_egui::egui::{self, Rect, Stroke};
use bloq_graph::BlockKind;

const DOCK_HORIZONTAL_PADDING: f32 = 6.0;
const DOCK_VERTICAL_PADDING: f32 = 6.0;
const DOCK_ITEM_GAP: f32 = 2.0;
const DOCK_SEPARATOR_WIDTH: f32 = 12.0;
const DOCK_MARGIN: f32 = 4.0;
const DOCK_MAX_WIDTH: f32 = 920.0;
const DOCK_HEIGHT: f32 = 62.0;
const DOCK_ITEM_SIZE: egui::Vec2 = egui::vec2(66.0, 46.0);
const BASE_ICON_SIZE: f32 = 24.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DockItem {
    Block(BlockKind),
    Pipe,
}

/// Draws the placement dock over the viewport in Edit mode, queuing a tool or
/// block-kind selection when a thumbnail is clicked.
pub(crate) fn draw_placement_dock(
    ctx: &egui::Context,
    viewport: Rect,
    editor_state: &EditorState,
    thumbnails: &mut ThumbnailTextures,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if editor_state.mode != EditorMode::Edit {
        return;
    }
    // The BLOG window fills a small viewport; keep its title and close button usable.
    if editor_state.show_blog_buffer && (viewport.width() < 900.0 || viewport.height() < 520.0) {
        return;
    }

    // `viewport` is the panel-free central rect; on a window small enough that the
    // top/bottom bars consume its whole height it collapses to zero or negative,
    // and positioning the dock in it would be meaningless. Skip that frame.
    if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
        return;
    }

    let min_width = DOCK_SEPARATOR_WIDTH + DOCK_ITEM_SIZE.x * 3.0 + DOCK_HORIZONTAL_PADDING * 2.0;
    let dock_width = (viewport.width() - DOCK_MARGIN * 2.0)
        .min(DOCK_MAX_WIDTH)
        .max(min_width)
        .min(viewport.width());
    let dock_height = DOCK_HEIGHT.min(viewport.height());
    let dock_pos = egui::pos2(
        viewport.center().x - dock_width / 2.0,
        viewport.top() + DOCK_MARGIN,
    );

    egui::Area::new(egui::Id::new("placement_mode_dock"))
        .order(egui::Order::Foreground)
        .fixed_pos(dock_pos)
        .show(ctx, |ui| {
            ui.set_width(dock_width);
            ui.set_min_width(dock_width);
            ui.set_max_width(dock_width);
            ui.set_height(dock_height);
            ui.set_min_height(dock_height);
            ui.set_max_height(dock_height);
            let dock_fill = if palette.dark_mode {
                with_alpha(palette.bg_panel, 236)
            } else {
                with_alpha(palette.bg_surface, 250)
            };
            let dock_border = if palette.dark_mode {
                with_alpha(palette.border_bright, 190)
            } else {
                palette.border_bright
            };
            egui::Frame::new()
                .fill(dock_fill)
                .stroke(egui::Stroke::new(1.25, dock_border))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(
                    DOCK_HORIZONTAL_PADDING as i8,
                    DOCK_VERTICAL_PADDING as i8,
                ))
                .show(ui, |ui| {
                    let inner_width = (dock_width - DOCK_HORIZONTAL_PADDING * 2.0).max(0.0);
                    let inner_height = (dock_height - DOCK_VERTICAL_PADDING * 2.0).max(0.0);
                    ui.set_width(inner_width);
                    ui.set_height(inner_height);

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(DOCK_ITEM_GAP, 0.0);
                        ui.allocate_ui_with_layout(
                            egui::vec2(DOCK_ITEM_SIZE.x, inner_height),
                            egui::Layout::centered_and_justified(egui::Direction::TopDown),
                            |ui| {
                                draw_dock_item(
                                    ui,
                                    DockItem::Pipe,
                                    editor_state,
                                    thumbnails,
                                    intents,
                                    palette,
                                );
                            },
                        );
                        draw_dock_separator(ui, inner_height, palette);

                        let used_width = DOCK_ITEM_SIZE.x + DOCK_SEPARATOR_WIDTH;
                        let block_group_width =
                            (inner_width - used_width - DOCK_ITEM_GAP * 4.0).max(DOCK_ITEM_SIZE.x);
                        ui.allocate_ui_with_layout(
                            egui::vec2(block_group_width, inner_height),
                            egui::Layout::centered_and_justified(egui::Direction::TopDown),
                            |ui| {
                                egui::ScrollArea::horizontal()
                                    .id_salt("placement_mode_dock_scroll")
                                    .auto_shrink([false, true])
                                    .max_height(DOCK_ITEM_SIZE.y + 4.0)
                                    .max_width(block_group_width)
                                    .show(ui, |ui| {
                                        ui.spacing_mut().item_spacing =
                                            egui::vec2(DOCK_ITEM_GAP, 0.0);
                                        ui.horizontal(|ui| {
                                            for kind in BlockKind::all_kinds() {
                                                draw_dock_item(
                                                    ui,
                                                    DockItem::Block(kind),
                                                    editor_state,
                                                    thumbnails,
                                                    intents,
                                                    palette,
                                                );
                                            }
                                        });
                                    });
                            },
                        );
                    });
                });
        });
}

fn draw_dock_item(
    ui: &mut egui::Ui,
    item: DockItem,
    editor_state: &EditorState,
    thumbnails: &mut ThumbnailTextures,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let selected = match item {
        DockItem::Block(kind) => {
            editor_state.placement_tool == PlacementTool::Block && editor_state.block_kind == kind
        }
        DockItem::Pipe => editor_state.placement_tool == PlacementTool::Pipe,
    };

    let (rect, response) = ui.allocate_exact_size(DOCK_ITEM_SIZE, egui::Sense::click());
    let response = response.on_hover_text(format!(
        "{}\n{}",
        dock_item_label(item),
        dock_item_hint(item)
    ));
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::SelectableLabel,
            true,
            selected,
            dock_item_label(item),
        )
    });
    if activated(&response) {
        match item {
            DockItem::Block(kind) => intents.push(UiIntent::SelectPlacementBlock(kind)),
            DockItem::Pipe => intents.push(UiIntent::SelectPlacementTool(PlacementTool::Pipe)),
        }
    }

    let scale = if response.hovered() {
        1.16
    } else if selected {
        1.08
    } else {
        1.0
    };
    let icon_size = BASE_ICON_SIZE * scale;
    let icon_center = rect.center_top() + egui::vec2(0.0, 15.0);
    let icon_rect = Rect::from_center_size(icon_center, egui::vec2(icon_size, icon_size));

    if selected || response.hovered() || response.has_focus() {
        let highlight_rect = icon_rect.expand2(egui::vec2(7.0, 7.0));
        ui.painter().rect_filled(
            highlight_rect,
            16.0,
            if selected {
                with_alpha(palette.bg_hover, 220)
            } else {
                with_alpha(palette.bg_surface, 180)
            },
        );
        ui.painter().rect_stroke(
            highlight_rect,
            16.0,
            Stroke::new(
                1.0,
                if selected {
                    palette.accent_primary
                } else {
                    with_alpha(palette.border, 180)
                },
            ),
            egui::StrokeKind::Outside,
        );
    }

    if ui.is_rect_visible(icon_rect)
        && let Some(texture_id) =
            thumbnails.get_or_request(ThumbnailKey::Placement(dock_item_key(item)))
    {
        paint_texture_thumbnail(ui.painter(), texture_id, icon_rect, 1.0);
    }

    ui.painter().text(
        rect.center_bottom() - egui::vec2(0.0, 3.0),
        egui::Align2::CENTER_BOTTOM,
        if matches!(item, DockItem::Block(kind) if kind.is_patch_rotation()) {
            Cow::Borrowed("Rotation")
        } else {
            dock_item_label(item)
        },
        egui::FontId::proportional(11.0),
        if selected {
            palette.accent_primary
        } else {
            palette.text_dim
        },
    );
}

fn draw_dock_separator(ui: &mut egui::Ui, height: f32, palette: &ThemePalette) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(DOCK_SEPARATOR_WIDTH, height),
        egui::Sense::hover(),
    );
    let top = rect.top() + 8.0;
    let bottom = rect.bottom() - 8.0;
    let center_x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(center_x, top), egui::pos2(center_x, bottom)],
        Stroke::new(1.0, with_alpha(palette.border_bright, 220)),
    );
}

fn dock_item_label(item: DockItem) -> Cow<'static, str> {
    match item {
        DockItem::Block(kind) if kind.is_walking() => Cow::Borrowed("Walking"),
        DockItem::Block(kind) if kind.is_patch_rotation() => Cow::Borrowed("Patch Rotation"),
        DockItem::Block(kind) => Cow::Owned(kind.to_string()),
        DockItem::Pipe => Cow::Borrowed("Pipe"),
    }
}

fn dock_item_key(item: DockItem) -> PlacementPreviewKind {
    match item {
        DockItem::Block(kind) => PlacementPreviewKind::Block(kind),
        DockItem::Pipe => PlacementPreviewKind::Pipe,
    }
}

fn dock_item_hint(item: DockItem) -> &'static str {
    match item {
        DockItem::Pipe => {
            "Click a starting block, then an adjacent endpoint. WASD / Up / Down extend or follow existing pipes. Enable Hadamard in the toolbar instead of holding H."
        }
        DockItem::Block(kind) if kind.is_walking() || kind.is_patch_rotation() => {
            "Place on an empty grid cell, then choose a highlighted endpoint. Double-click an existing block to edit its attributes."
        }
        DockItem::Block(_) => {
            "Click an empty grid cell to place. Double-click an existing block to edit its attributes. J/K change the placement layer."
        }
    }
}
