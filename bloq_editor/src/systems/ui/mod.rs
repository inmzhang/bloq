//! The egui interface: edge panels, overlays, and floating windows.
//!
//! Each panel is drawn by its own Bevy system in the `EguiPrimaryContextPass`
//! chain. egui 0.35 no longer composes panels across systems on the context, so
//! every panel system builds its own background-layer root `egui::Ui` over the
//! shared [`CentralViewport`] and shrinks that rect by the space it claims; the
//! leftover rect is what the 3D viewport and pointer-capture read.

pub(crate) mod action_editor;
pub(crate) mod action_viewer;
mod apply_intents;
mod axis_indicator;
pub(crate) mod blog_buffer;
pub(crate) mod circuit_viewer;
mod command_palette;
pub(crate) mod commands;
pub(crate) mod edit_element;
mod empty_state;
pub(crate) mod graph_layout;
mod help_window;
mod intents;
mod module_legend;
mod module_panel;
mod notifications;
mod placement_dock;
mod side_panel;
mod status_bar;
mod tab_bar;
mod thumbnail;
mod toolbar;
#[cfg(any(target_arch = "wasm32", test))]
mod web_keyboard;
pub(crate) mod zx_viewer;

use crate::components::AxisHelper;
use crate::resources::{
    ActionEditState, ActionViewerState, BloqViewerState, BoxSelectionState, CentralViewport,
    CompileUiState, EditorMode, EditorState, EditorTabs, GraphState, GraphUiSummary,
    ImportExportState, Notifications, TargetState, ThumbnailTextures, UiInputState, ZxViewerState,
};
use crate::systems::EditorUpdateSet;
use crate::systems::jobs::EditorJobs;
use crate::theme::{ThemePreset, palette, setup_theme};
use action_editor::draw_action_editor_window;
use action_viewer::draw_action_viewer;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use blog_buffer::draw_blog_buffer;
use circuit_viewer::draw_circuit_viewer;
use command_palette::{CommandPaletteState, draw_command_palette};
use commands::CommandContext;
use edit_element::draw_edit_element_window;
use help_window::draw_help_window;
use module_legend::draw_module_legend;
use notifications::{draw_notifications, prune_expired_toasts};
use placement_dock::draw_placement_dock;
use status_bar::draw_status_bar;
use zx_viewer::draw_zx_viewer;

pub(crate) use axis_indicator::draw_axis_indicator_system;

pub(crate) use apply_intents::{apply_fill_ports, apply_ui_intents_system};
pub(crate) use intents::{UiIntent, UiIntentBuffer};
pub(crate) use module_panel::DefinitionEdit;
pub(crate) use tab_bar::{PendingTabClose, draw_close_tab_confirm_system};

/// The egui interface: the ordered `EguiPrimaryContextPass` panel chain, the
/// UI-input gate that heads the `Update` interaction chain, and all the
/// UI-owned resources (tabs, viewers, intents, notifications, viewport rect).
pub(crate) struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_arch = "wasm32")]
        {
            app.world_mut()
                .resource_mut::<bevy_egui::EguiGlobalSettings>()
                .input_system_settings
                .run_write_keyboard_input_messages_system = false;
            app.add_systems(
                PreUpdate,
                web_keyboard::write_ordered_keyboard_messages
                    .in_set(bevy_egui::EguiInputSet::ReadBevyMessages),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(Update, tab_bar::request_window_close);
        app.init_resource::<EditorTabs>()
            .init_resource::<GraphUiSummary>()
            .init_resource::<Notifications>()
            .init_resource::<ImportExportState>()
            .init_resource::<BloqViewerState>()
            .init_resource::<ZxViewerState>()
            .init_resource::<TargetState>()
            .init_resource::<ActionEditState>()
            .init_resource::<ActionViewerState>()
            .init_resource::<UiIntentBuffer>()
            .init_resource::<CommandPaletteState>()
            .init_resource::<PendingTabClose>()
            .insert_resource(CentralViewport(egui::Rect::ZERO))
            // UI pass. egui 0.35 no longer composes panels across systems on the
            // context, so each panel system draws into its own root `Ui` and
            // shrinks the shared `CentralViewport`; the leftover rect is what the
            // 3D-viewport overlays and pointer-capture read.
            //
            // INVARIANT: the shrinkers (begin_ui_frame → top → status → side)
            // must run before every consumer (placement_dock, axis_indicator, and
            // sync_ui_input_state in the Update schedule). The `.chain()` below
            // pins that order — a new edge panel must be inserted above the
            // consumers, not after them, or they read a rect that still counts its
            // area as free viewport.
            .add_systems(
                EguiPrimaryContextPass,
                (
                    begin_ui_frame_system,
                    crate::plugins::drain_internal_errors_system,
                    sync_graph_ui_summary_system,
                    draw_top_bars_system,
                    draw_status_bar_system,
                    draw_side_panel_system,
                    draw_module_panel_system,
                    (draw_placement_dock_system, draw_empty_state_system).chain(),
                    (
                        draw_axis_indicator_system,
                        draw_module_legend_system,
                        draw_selection_box_system,
                        crate::systems::input::draw_translation_gizmo_system,
                    )
                        .chain(),
                    draw_circuit_viewer_system,
                    (draw_zx_viewer_system, draw_blog_buffer_system).chain(),
                    draw_edit_element_window_system,
                    draw_action_viewer_system,
                    draw_action_editor_window_system,
                    draw_help_window_system,
                    draw_command_palette_system,
                    draw_close_tab_confirm_system,
                    draw_notifications_system,
                    (apply_ui_intents_system, sync_camera_view_system).chain(),
                )
                    .chain()
                    .run_if(|state: Res<EditorState>| !state.request_screenshot),
            )
            // The UI-input gate heads the interaction chain; downstream
            // interaction systems (camera/input/visuals) order after it.
            .add_systems(
                Update,
                sync_ui_input_state_system.in_set(EditorUpdateSet::Interaction),
            );
    }
}

fn draw_empty_state_system(
    mut contexts: EguiContexts,
    central: Res<CentralViewport>,
    mut editor: ResMut<EditorState>,
    graph: Res<GraphState>,
    mut tabs: ResMut<EditorTabs>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let palette = palette(editor.theme_preset);
    empty_state::draw_empty_state(
        ctx,
        central.0,
        &mut editor,
        &graph,
        &mut tabs.show_welcome,
        &mut intents,
        palette,
    );
}

fn draw_module_panel_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    mut import_export: ResMut<ImportExportState>,
    tabs: Res<EditorTabs>,
    mut intents: ResMut<UiIntentBuffer>,
    mut central: ResMut<CentralViewport>,
) {
    if editor_state.mode != EditorMode::Module {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    with_panel_root(ctx, "module_panel_root", &mut central, |root| {
        module_panel::draw_module_panel(
            root,
            &mut import_export.module_ui,
            &graph_state,
            &tabs,
            editor_state.module_view.as_ref(),
            &mut intents,
            palette(editor_state.theme_preset),
        );
    });
}

/// Shift the 3D projection into the space left by panels without cropping the
/// full-window egui camera. Bevy's sub-view supports off-centre projections.
fn sync_camera_view_system(
    mut contexts: EguiContexts,
    editor: Res<EditorState>,
    central: Res<CentralViewport>,
    camera: Single<(&mut Camera, &mut Projection), With<crate::components::EditorCamera>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let (mut camera, mut projection) = camera.into_inner();
    let Projection::Perspective(perspective) = &mut *projection else {
        return;
    };
    let (view, fov) = if !editor.mode.covers_viewport() && !editor.request_screenshot {
        crate::systems::camera::viewport_camera_view(ctx.content_rect(), central.0)
    } else {
        (None, crate::systems::camera::CAMERA_DEFAULT_FOV_Y)
    };
    if camera.sub_camera_view != view {
        camera.sub_camera_view = view;
    }
    if perspective.fov != fov {
        perspective.fov = fov;
    }
}

/// Scores `haystack` against a subsequence `query`, or `None` if the query's
/// characters do not all appear in order. Higher is better: tight matches that
/// start early beat characters scattered across the whole string.
///
/// Shared by the gallery filter and the command palette. Both corpora are small
/// enough that re-scoring every frame beats caching, and that a real
/// fuzzy-matching dependency would be disproportionate.
pub(crate) fn fuzzy_score(haystack: &str, query: &str) -> Option<i32> {
    let mut haystack = haystack.chars().enumerate();
    let (mut first, mut last) = (0, 0);
    for (nth, wanted) in query.chars().enumerate() {
        let (at, _) = haystack.find(|(_, candidate)| candidate.eq_ignore_ascii_case(&wanted))?;
        if nth == 0 {
            first = at;
        }
        last = at;
    }
    Some(-((last - first) as i32) - first as i32)
}

// Single-line Enter and Escape can surrender focus before the commit row is
// drawn. Preserve ownership for that frame, but never steal another field's key.
fn field_owns_shortcuts(ui: &egui::Ui, id: egui::Id) -> bool {
    ui.memory(|memory| {
        memory.has_focus(id) || (memory.focused().is_none() && memory.had_focus_last_frame(id))
    })
}

pub(crate) fn activated(response: &egui::Response) -> bool {
    response.enabled()
        && (response.clicked()
            || (response.contains_pointer()
                && response.ctx.input(|input| {
                    input.pointer.primary_released()
                        && !input.pointer.is_decidedly_dragging()
                        && input
                            .pointer
                            .press_origin()
                            .is_none_or(|origin| response.rect.contains(origin))
                })))
}

const FLOATING_WINDOW_GAP: f32 = 8.0;

/// One quadrant of the panel-free viewport. The four primary floating windows
/// use distinct quadrants, so their initial layouts tile instead of stacking.
pub(super) fn tiled_window_rect(viewport: egui::Rect, corner: egui::Align2) -> egui::Rect {
    let frame = viewport.shrink(FLOATING_WINDOW_GAP);
    let size = if frame.width() < 900.0 || frame.height() < 520.0 {
        // A quadrant on a small screen cannot fit a code editor or graph toolbar.
        frame.size().max(egui::Vec2::splat(1.0))
    } else {
        ((frame.size() - egui::Vec2::splat(FLOATING_WINDOW_GAP)) / 2.0).max(egui::Vec2::splat(1.0))
    };
    corner.align_size_within_rect(size, frame)
}

/// Draws one edge panel into a fresh background-layer root [`egui::Ui`] spanning
/// the current [`CentralViewport`], then shrinks that rect by whatever the panel
/// consumed. egui 0.35 shows panels inside a parent `Ui` rather than on the
/// context, so each panel system builds its own root here; folding the build and
/// the shrink into one call keeps the shrink un-forgettable — a caller can't draw
/// a panel without also reserving its space.
fn with_panel_root(
    ctx: &egui::Context,
    id: impl egui::AsId,
    central: &mut CentralViewport,
    draw: impl FnOnce(&mut egui::Ui),
) {
    let mut root = egui::Ui::new(
        ctx.clone(),
        egui::Id::new(id),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(central.0),
    );
    draw(&mut root);
    central.0 = root.available_rect_before_wrap();
}

// =============================================================================
// System wrappers (called from plugins.rs EguiPrimaryContextPass chain)
// =============================================================================

/// Opens the egui frame: applies a pending theme change and resets the
/// [`CentralViewport`] to the full content rect for the chained panel systems to
/// shrink, then prunes expired toasts.
pub(crate) fn begin_ui_frame_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    mut applied_theme: Local<Option<ThemePreset>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    axes: Query<(&AxisHelper, &MeshMaterial3d<StandardMaterial>)>,
    mut notifications: ResMut<Notifications>,
    mut central: ResMut<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    if *applied_theme != Some(editor_state.theme_preset) {
        setup_theme(ctx, editor_state.theme_preset);
        crate::systems::setup::apply_editor_material_theme(&editor_state, &mut materials);
        let palette = palette(editor_state.theme_preset);
        for (axis, material) in &axes {
            if let Some(mut material) = materials.get_mut(&material.0) {
                material.base_color = crate::systems::setup::axis_color(palette, *axis);
            }
        }
        *applied_theme = Some(editor_state.theme_preset);
    }

    // Reset the panel-free region to the full content rect; the chained panel
    // systems shrink it as they claim their edges.
    central.0 = ctx.content_rect();

    prune_expired_toasts(&mut notifications);
    notifications.capture_keyboard = notifications.center_open;
}

/// Refreshes the cached [`GraphUiSummary`] from the current graph.
fn sync_graph_ui_summary_system(
    graph_state: Res<GraphState>,
    tabs: Res<EditorTabs>,
    mut graph_ui_summary: ResMut<GraphUiSummary>,
) {
    graph_ui_summary.sync_from_graph_state(tabs.active, &graph_state);
}

/// Draws the top tab bar and toolbar.
fn draw_top_bars_system(
    mut action_viewer: ResMut<ActionViewerState>,
    mut contexts: EguiContexts,
    mut editor_state: ResMut<EditorState>,
    mut import_export: ResMut<ImportExportState>,
    mut compile_ui: ResMut<CompileUiState>,
    mut command_palette: ResMut<CommandPaletteState>,
    mut tabs: ResMut<EditorTabs>,
    jobs: Res<EditorJobs>,
    graph_state: Res<GraphState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    mut central: ResMut<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let current_theme = palette(editor_state.theme_preset);
    with_panel_root(ctx, "top_bars_root", &mut central, |root| {
        egui::Panel::top("top_bars")
            .show_separator_line(true)
            .frame(
                egui::Frame::new()
                    .fill(current_theme.bg_surface)
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(egui::Margin::ZERO),
            )
            .show(root, |ui| {
                tab_bar::draw_tab_bar(ui, &mut tabs, &mut ui_intents, current_theme);
                egui::Frame::new()
                    .fill(current_theme.bg_surface)
                    .stroke(egui::Stroke::new(1.0, current_theme.border))
                    .corner_radius(6)
                    .outer_margin(egui::Margin::symmetric(8, 4))
                    .inner_margin(egui::Margin::symmetric(8, 4))
                    .show(ui, |ui| {
                        ui.push_id(tabs.active, |ui| {
                            ui.set_min_width(ui.available_width());
                            toolbar::draw_toolbar_contents(
                                ui,
                                toolbar::ToolbarState {
                                    action_viewer: &mut action_viewer,
                                    editor_state: &mut editor_state,
                                    import_export: &mut import_export,
                                    compile_ui: &mut compile_ui,
                                    graph_state: &graph_state,
                                    jobs: &jobs,
                                    intents: &mut ui_intents,
                                    command_palette: &mut command_palette,
                                },
                                current_theme,
                            );
                        });
                    });
            });
    });
}

/// Draws the bottom status bar.
fn draw_status_bar_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    graph_ui_summary: Res<GraphUiSummary>,
    compile_ui: Res<CompileUiState>,
    tabs: Res<EditorTabs>,
    action_edit: Res<ActionEditState>,
    jobs: Res<EditorJobs>,
    mut notifications: ResMut<Notifications>,
    mut central: ResMut<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    with_panel_root(ctx, "status_bar_root", &mut central, |root| {
        draw_status_bar(
            root,
            &editor_state,
            &graph_state,
            &graph_ui_summary,
            &compile_ui,
            tabs.active,
            &action_edit,
            &jobs,
            &mut notifications,
            palette(editor_state.theme_preset),
        );
    });
}

/// Paints the drag-rectangle overlay while a box selection is active in View mode.
fn draw_selection_box_system(
    mut contexts: EguiContexts,
    box_selection: Res<BoxSelectionState>,
    editor_state: Res<EditorState>,
) {
    if !box_selection.active || editor_state.mode != EditorMode::View {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let palette = palette(editor_state.theme_preset);
    let min = box_selection.start.min(box_selection.current);
    let max = box_selection.start.max(box_selection.current);
    let rect = egui::Rect::from_min_max(egui::pos2(min.x, min.y), egui::pos2(max.x, max.y));
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("selection_box_overlay"),
    ));
    let fill_alpha = if palette.dark_mode { 54 } else { 46 };
    let fill = egui::Color32::from_rgba_unmultiplied(
        palette.accent_primary.r(),
        palette.accent_primary.g(),
        palette.accent_primary.b(),
        fill_alpha,
    );
    painter.rect_filled(rect, 0.0, fill);
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(2.0, palette.accent_primary),
        egui::StrokeKind::Outside,
    );
}

/// Draws the color-to-definition key inside the unobstructed 3D viewport.
fn draw_module_legend_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    central: Res<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_module_legend(
        ctx,
        central.0,
        &editor_state,
        &graph_state,
        palette(editor_state.theme_preset),
    );
}

/// Draws the left side panel (gallery, import/export, tools).
fn draw_side_panel_system(
    mut contexts: EguiContexts,
    mut editor_state: ResMut<EditorState>,
    mut action_viewer: ResMut<ActionViewerState>,
    graph_state: Res<GraphState>,
    graph_ui_summary: Res<GraphUiSummary>,
    jobs: Res<EditorJobs>,
    tabs: Res<EditorTabs>,
    mut thumbnails: ResMut<ThumbnailTextures>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    mut central: ResMut<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let current_theme = palette(editor_state.theme_preset);
    with_panel_root(
        ctx,
        ("side_panel_root", tabs.active),
        &mut central,
        |root| {
            side_panel::draw_side_panel(
                root,
                &mut editor_state,
                &mut action_viewer,
                &graph_state,
                &graph_ui_summary,
                &jobs,
                &mut thumbnails,
                &mut ui_intents,
                current_theme,
            );
        },
    );
}

/// Draws the block/pipe placement dock over the viewport in Edit mode.
fn draw_placement_dock_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    mut thumbnails: ResMut<ThumbnailTextures>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    central: Res<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let current_theme = palette(editor_state.theme_preset);
    draw_placement_dock(
        ctx,
        central.0,
        &editor_state,
        &mut thumbnails,
        &mut ui_intents,
        current_theme,
    );
}

/// Draws the Bloq graph/circuit viewer window.
fn draw_circuit_viewer_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    tabs: Res<EditorTabs>,
    compile_ui: Res<CompileUiState>,
    jobs: Res<EditorJobs>,
    mut circuit_viewer: ResMut<BloqViewerState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    central: Res<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_circuit_viewer(
        ctx,
        central.0,
        &editor_state,
        &graph_state,
        tabs.active,
        &compile_ui,
        &jobs,
        &mut circuit_viewer,
        &mut ui_intents,
    );
}

/// Draws the ZX-diagram viewer window.
fn draw_zx_viewer_system(
    mut contexts: EguiContexts,
    graph_state: Res<GraphState>,
    mut editor_state: ResMut<EditorState>,
    mut zx_viewer: ResMut<ZxViewerState>,
    tabs: Res<EditorTabs>,
    mut jobs: ResMut<EditorJobs>,
    central: Res<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    draw_zx_viewer(
        ctx,
        central.0,
        &graph_state,
        &mut editor_state,
        &mut zx_viewer,
        tabs.active,
        &mut jobs,
    )
}

/// Draws the editable BLOG buffer window.
fn draw_blog_buffer_system(
    graph_state: Res<GraphState>,
    mut contexts: EguiContexts,
    mut editor_state: ResMut<EditorState>,
    mut import_export: ResMut<ImportExportState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    central: Res<CentralViewport>,
    tabs: Res<EditorTabs>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let current_theme = palette(editor_state.theme_preset);
    import_export.refresh_from_graph(&graph_state);
    draw_blog_buffer(
        ctx,
        central.0,
        tabs.active,
        &mut editor_state,
        &mut import_export,
        &mut ui_intents,
        current_theme,
    );
}

/// Draws the keyboard-shortcut help window when open.
fn draw_help_window_system(
    mut contexts: EguiContexts,
    mut editor_state: ResMut<EditorState>,
    mut query: Local<String>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_help_window(ctx, &mut editor_state, &mut query);
}

/// Draws the command palette overlay.
fn draw_command_palette_system(
    mut contexts: EguiContexts,
    mut palette_state: ResMut<CommandPaletteState>,
    editor_state: Res<EditorState>,
    graph_state: Res<GraphState>,
    tabs: Res<EditorTabs>,
    jobs: Res<EditorJobs>,
    import_export: Res<ImportExportState>,
    mut intents: ResMut<UiIntentBuffer>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let theme = palette(editor_state.theme_preset);
    let command_ctx = CommandContext {
        editor_state: &editor_state,
        graph_state: &graph_state,
        tabs: &tabs,
        jobs: &jobs,
        import_export: &import_export,
    };
    draw_command_palette(ctx, &mut palette_state, &command_ctx, &mut intents, theme);
}

/// Draws the attribute-editor window for the targeted block or pipe.
fn draw_edit_element_window_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    tabs: Res<EditorTabs>,
    mut target_state: ResMut<TargetState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_edit_element_window(
        ctx,
        tabs.active,
        &mut target_state,
        editor_state.theme_preset,
        &mut ui_intents,
    );
}

/// Draws the Actions window: the action DAG plus its authoring controls.
fn draw_action_viewer_system(
    mut contexts: EguiContexts,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    tabs: Res<EditorTabs>,
    mut action_viewer: ResMut<ActionViewerState>,
    mut action_edit: ResMut<ActionEditState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    central: Res<CentralViewport>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let output = draw_action_viewer(
        ctx,
        central.0,
        &graph_state,
        tabs.active,
        &mut action_viewer,
        &mut action_edit,
        palette(editor_state.theme_preset),
        &mut ui_intents,
    );
    if editor_state.action_hovered_elements != output.hovered {
        editor_state.action_hovered_elements = output.hovered;
    }
    let selected_surface = output.selected_surface.map(|(ordinal, name, surface)| {
        (
            ordinal,
            bloq_graph::StabilizerGenerator::new(
                surface,
                bloq_graph::StabilizerRowKind::Measurement { name },
            ),
        )
    });
    if editor_state.set_action_stabilizer(selected_surface) {
        graph_state.needs_rerender = true;
    }
}

/// Draws the action draft popup while a draft is open.
fn draw_action_editor_window_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    tabs: Res<EditorTabs>,
    graph_state: Res<GraphState>,
    mut action_edit: ResMut<ActionEditState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_action_editor_window(
        ctx,
        tabs.active,
        &mut action_edit,
        &graph_state.graph,
        editor_state.theme_preset,
        &mut ui_intents,
    );
}

/// Draws the toast stack.
fn draw_notifications_system(
    mut contexts: EguiContexts,
    editor_state: Res<EditorState>,
    central: Res<CentralViewport>,
    mut notifications: ResMut<Notifications>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    draw_notifications(
        ctx,
        central.0,
        &mut notifications,
        palette(editor_state.theme_preset),
    );
}

/// Records whether egui currently wants pointer/keyboard input, treating any
/// pointer outside the panel-free central rect as "over UI" so viewport
/// gestures do not fire while the cursor is on a background-layer panel.
pub(crate) fn sync_ui_input_state_system(
    mut contexts: EguiContexts,
    mut ui_input: ResMut<UiInputState>,
    central: Res<CentralViewport>,
    notifications: Res<Notifications>,
    command_palette: Res<CommandPaletteState>,
    target_state: Res<TargetState>,
    import_export: Res<ImportExportState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    // `is_pointer_over_egui` only recognises non-background layers (windows,
    // menus); our edge panels live on the background layer and bevy_egui runs
    // its own root pass empty, so egui reports the whole screen as free. Treat
    // any pointer outside the panel-free central rect as "over UI" so viewport
    // gestures (e.g. scroll-to-zoom) don't fire while the cursor is on a panel.
    // egui's pass runs in PostUpdate: its pointer is one frame behind the
    // Window cursor used by placement in Update. Hit-test the current cursor
    // against the last rendered UI so a move+click cannot pass through a panel.
    let pointer = windows.single().map_or_else(
        |_| ctx.input(|input| input.pointer.latest_pos()),
        |window| {
            window
                .physical_cursor_position()
                .map(|pos| egui::pos2(pos.x, pos.y) / ctx.pixels_per_point())
        },
    );
    let over_panel = pointer.is_some_and(|pos| pointer_over_editor_ui(ctx, central.0, pos));
    let attribute_dialog = target_state.open_window
        || target_state.capture_input
        || import_export.blocks_graph_editing();
    ui_input.pointer_over_ui = over_panel || attribute_dialog;
    ui_input.wants_pointer_input = ctx.egui_wants_pointer_input() || over_panel || attribute_dialog;
    ui_input.wants_keyboard_input = ctx.egui_wants_keyboard_input()
        || notifications.capture_keyboard
        || command_palette.capture_keyboard
        || attribute_dialog;
}

/// Explicit position avoids egui's previous-pass pointer when Bevy has newer input.
pub(crate) fn pointer_over_editor_ui(
    ctx: &egui::Context,
    viewport: egui::Rect,
    pos: egui::Pos2,
) -> bool {
    !viewport.contains(pos)
        || ctx
            .layer_id_at(pos)
            .is_some_and(|layer| layer.order != egui::Order::Background)
}

// The active tab's live state is the editor Resources themselves; its snapshot
// is only a stash read on tab switch/close/import via `restore_active`, and
// every such read is preceded by an explicit `save_active_tab` that captures the
// current Resources. There is therefore no per-frame snapshot system: capturing
// the full graph, undo history, and viewer state every frame was pure overhead.

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_egui::{EguiContext, EguiUserTextures, PrimaryEguiContext};

    #[test]
    fn attribute_dialog_owns_viewport_input_even_when_pointer_is_outside() {
        let mut app = App::new();
        app.init_resource::<UiInputState>()
            .init_resource::<ImportExportState>()
            .insert_resource(CentralViewport(egui::Rect::EVERYTHING))
            .init_resource::<Notifications>()
            .init_resource::<CommandPaletteState>()
            .init_resource::<TargetState>()
            .init_resource::<EguiUserTextures>()
            .add_systems(Update, sync_ui_input_state_system);
        app.world_mut()
            .spawn((EguiContext::default(), PrimaryEguiContext));
        for (open, captured) in [(true, false), (false, true)] {
            let mut target = app.world_mut().resource_mut::<TargetState>();
            target.open_window = open;
            target.capture_input = captured;
            app.update();
            let input = app.world().resource::<UiInputState>();
            assert!(input.blocks_viewport_keyboard_input());
            assert!(input.blocks_viewport_pointer_input());
        }
    }

    #[test]
    fn current_window_cursor_blocks_panels_despite_stale_egui_pointer() {
        let mut app = App::new();
        app.init_resource::<UiInputState>()
            .init_resource::<ImportExportState>()
            .insert_resource(CentralViewport(egui::Rect::from_min_max(
                egui::pos2(0.0, 0.0),
                egui::pos2(800.0, 600.0),
            )))
            .init_resource::<Notifications>()
            .init_resource::<CommandPaletteState>()
            .init_resource::<TargetState>()
            .init_resource::<EguiUserTextures>()
            .add_systems(Update, sync_ui_input_state_system);
        let mut context = EguiContext::default();
        let ctx = context.get_mut().clone();
        ctx.set_pixels_per_point(2.0);
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::PointerMoved(egui::pos2(100.0, 100.0))],
                ..Default::default()
            },
            |_| {},
        );
        output.textures_delta.clear();
        app.world_mut().spawn((context, PrimaryEguiContext));
        let mut window = Window {
            resolution: bevy::window::WindowResolution::new(2000, 1200),
            ..Default::default()
        };
        window.set_physical_cursor_position(Some(bevy::math::DVec2::new(1700.0, 200.0)));
        let window = app
            .world_mut()
            .spawn((window, bevy::window::PrimaryWindow))
            .id();
        app.update();
        assert!(
            app.world()
                .resource::<UiInputState>()
                .blocks_viewport_pointer_input()
        );
        // Moving back to the canvas must not retain the panel's capture.
        app.world_mut()
            .get_mut::<Window>(window)
            .unwrap()
            .set_physical_cursor_position(Some(bevy::math::DVec2::new(400.0, 200.0)));
        app.update();
        assert!(
            !app.world()
                .resource::<UiInputState>()
                .blocks_viewport_pointer_input()
        );
    }

    #[test]
    fn long_notifications_and_help_stay_inside_small_screens() {
        for preset in [ThemePreset::Light, ThemePreset::GruvboxMaterial] {
            for size in [egui::vec2(360.0, 480.0), egui::vec2(1280.0, 720.0)] {
                for (help, narrow_canvas, center) in [
                    (false, false, false),
                    (false, true, false),
                    (false, false, true),
                    (false, true, true),
                    (true, false, false),
                ] {
                    let ctx = egui::Context::default();
                    setup_theme(&ctx, preset);
                    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
                    let viewport = if narrow_canvas {
                        egui::Rect::from_min_max(
                            egui::pos2(size.x - 90.0, 80.0),
                            egui::pos2(size.x, size.y - 80.0),
                        )
                    } else {
                        screen.shrink2(egui::vec2(8.0, 80.0))
                    };
                    let mut notifications = Notifications::default();
                    notifications.push_warn("A long warning with actionable details. ".repeat(12));
                    notifications
                        .push_error("A long compiler error with details to copy. ".repeat(15));
                    if center {
                        notifications.open_center();
                    }
                    let mut editor = EditorState {
                        theme_preset: preset,
                        show_help_window: help,
                        ..Default::default()
                    };
                    for _ in 0..3 {
                        ctx.run_ui(
                            egui::RawInput {
                                screen_rect: Some(screen),
                                ..Default::default()
                            },
                            |ui| {
                                if help {
                                    draw_help_window(ui.ctx(), &mut editor, &mut String::new());
                                } else {
                                    draw_notifications(
                                        ui.ctx(),
                                        viewport,
                                        &mut notifications,
                                        palette(preset),
                                    );
                                }
                            },
                        )
                        .drop_without_applying_deltas();
                    }
                    let id = if help {
                        egui::Id::new(Some("Shortcuts"))
                    } else {
                        egui::Id::new("notifications")
                    };
                    let rect = ctx
                        .memory(|memory| memory.area_rect(id))
                        .unwrap_or_else(|| panic!("missing overlay: help={help}, size={size:?}"));
                    let bounds = if help || narrow_canvas {
                        screen
                    } else {
                        viewport
                    };
                    assert!(
                        bounds.contains_rect(rect),
                        "help={help}, {size:?}: {rect:?} outside {bounds:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn tab_side_panels_reserve_width_without_consuming_viewport_height() {
        let viewport = egui::Rect::from_min_max(egui::pos2(20.0, 80.0), egui::pos2(1380.0, 900.0));
        let mut context = EguiContext::default();
        let ctx = context.get_mut().clone();
        setup_theme(&ctx, ThemePreset::default());
        let mut app = App::new();
        app.init_resource::<EditorState>()
            .init_resource::<ActionViewerState>()
            .init_resource::<GraphState>()
            .init_resource::<GraphUiSummary>()
            .init_resource::<EditorJobs>()
            .init_resource::<EditorTabs>()
            .init_resource::<ThumbnailTextures>()
            .init_resource::<UiIntentBuffer>()
            .init_resource::<EguiUserTextures>()
            .add_systems(Update, draw_side_panel_system);
        app.world_mut().spawn((context, PrimaryEguiContext));
        for expanded in [false, true] {
            let mut editor = app.world_mut().resource_mut::<EditorState>();
            editor.show_side_panel = expanded;
            editor.gallery_panel_expanded = expanded;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1400.0, 1000.0),
                    )),
                    ..Default::default()
                },
                |_| {
                    app.insert_resource(CentralViewport(viewport));
                    app.update();
                },
            );
            output.textures_delta.clear();
            let remaining = app.world().resource::<CentralViewport>().0;
            assert_eq!(remaining.y_range(), viewport.y_range());
            assert!(remaining.left() > viewport.left());
            assert!(remaining.right() < viewport.right());
            assert!(
                remaining.width() > 0.0,
                "expanded={expanded}: {remaining:?}"
            );
            let mut tabs = app.world_mut().resource_mut::<EditorTabs>();
            let next = tabs.add_empty_tab(&EditorState::default());
            tabs.active = next;
        }
    }

    #[test]
    fn small_viewports_give_viewers_readable_full_size() {
        let viewport = egui::Rect::from_min_size(egui::pos2(0.0, 100.0), egui::vec2(640.0, 480.0));
        for corner in [
            egui::Align2::LEFT_TOP,
            egui::Align2::RIGHT_TOP,
            egui::Align2::LEFT_BOTTOM,
            egui::Align2::RIGHT_BOTTOM,
        ] {
            let rect = tiled_window_rect(viewport, corner);
            assert!(viewport.contains_rect(rect));
            assert_eq!(rect.size(), egui::vec2(624.0, 464.0));
        }
    }

    #[test]
    fn floating_window_tiles_fit_without_overlap() {
        let viewport = egui::Rect::from_min_size(egui::pos2(20.0, 80.0), egui::vec2(1200.0, 800.0));
        let tiles = [
            tiled_window_rect(viewport, egui::Align2::LEFT_TOP),
            tiled_window_rect(viewport, egui::Align2::RIGHT_TOP),
            tiled_window_rect(viewport, egui::Align2::LEFT_BOTTOM),
            tiled_window_rect(viewport, egui::Align2::RIGHT_BOTTOM),
        ];

        assert!(tiles.iter().all(|tile| viewport.contains_rect(*tile)));
        for (index, tile) in tiles.iter().enumerate() {
            assert!(
                tiles[index + 1..]
                    .iter()
                    .all(|other| !tile.intersects(*other))
            );
        }
    }
}
