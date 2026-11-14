//! The Actions window: a node-link view of the graph's action DAG, and the one
//! place actions are added, edited, and removed.
//!
//! Actions form a dependency DAG, not a list — a `resolve` waits on the
//! `measure` its condition names — so this draws the [`bloq_graph::ActionDag`]
//! the compiler builds, laid out with the same Sugiyama pass the Bloq program
//! view uses. Authoring lives here too, so the buttons that start a draft sit
//! next to what the draft will join.

use std::collections::{HashMap, HashSet};

use super::action_editor::{ACTION_KINDS, ActionEditState, action_elements};
use super::graph_layout::compute_relative;
use super::intents::{UiIntent, UiIntentBuffer};
use super::{activated, tiled_window_rect};
use crate::components::GraphElement;
use crate::resources::{EditorTabId, GraphState};
use crate::theme::{self, ThemePalette};
use bevy::prelude::Resource;
use bevy_egui::egui::{self, Color32, FontId, Rect, Sense, Stroke};
use bloq_graph::{Action, ActionDependency, Stabilizer};

const NODE_SIZE: egui::Vec2 = egui::vec2(184.0, 48.0);
const COLUMN_GAP: f32 = 46.0;
const CANVAS_MIN_HEIGHT: f32 = 140.0;

/// Open/camera/selection state for the Actions window, plus the memoized DAG
/// layout.
///
/// Window furniture rather than graph state, so it stays global instead of
/// joining [`EditorTabSnapshot`](crate::resources::EditorTabSnapshot) — the
/// layout cache is keyed by tab so two tabs cannot read each other's placement.
#[derive(Resource, Clone, Default)]
pub(crate) struct ActionViewerState {
    pub(crate) open: bool,
    pan: egui::Vec2,
    zoom: f32,
    /// Ordinal of the action whose details the footer shows.
    selected: Option<usize>,
    // An ordinal only identifies an action within this exact authored list.
    selection_scope: Option<(EditorTabId, Vec<Action>)>,
    layout: LayoutCache,
}

/// Sugiyama output for one graph revision, rebuilt only when the actions change.
#[derive(Clone, Default)]
struct LayoutCache {
    key: Option<(EditorTabId, u64, bool)>,
    positions: HashMap<u32, (f32, f32)>,
    content: (f32, f32),
}

#[derive(Default)]
pub(crate) struct ActionViewerOutput {
    pub(crate) hovered: HashSet<GraphElement>,
    pub(crate) selected_surface: Option<(usize, String, Stabilizer)>,
}

impl ActionViewerState {
    pub(crate) fn toggle(&mut self) {
        self.open = !self.open;
        if !self.open {
            self.layout = LayoutCache::default();
        }
    }

    fn sync_selection_scope(&mut self, tab: EditorTabId, graph: &bloq_graph::BlockGraph) {
        let unchanged = self
            .selection_scope
            .as_ref()
            .is_some_and(|(previous_tab, actions)| {
                *previous_tab == tab
                    && graph
                        .action_graph()
                        .ordered_nodes()
                        .map(|node| &node.action)
                        .eq(actions.iter())
            });
        if !unchanged {
            self.selected = None;
            self.selection_scope = Some((tab, graph.actions()));
        }
    }

    fn reset_camera(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 1.0;
    }

    /// Zoom, defaulting a never-initialized (or snapshot-restored zero) value to
    /// 1.0 so `Default` does not collapse the view.
    fn zoom(&self) -> f32 {
        if self.zoom <= f32::EPSILON {
            1.0
        } else {
            self.zoom
        }
    }
}

/// Draws the Actions window when open, returning the elements the hovered node
/// names so the caller can cross-highlight them in the scene.
pub(crate) fn draw_action_viewer(
    ctx: &egui::Context,
    viewport: egui::Rect,
    graph_state: &GraphState,
    active_tab: EditorTabId,
    viewer: &mut ActionViewerState,
    action_edit: &mut ActionEditState,
    palette: &ThemePalette,
    intents: &mut UiIntentBuffer,
) -> ActionViewerOutput {
    if !viewer.open {
        return ActionViewerOutput::default();
    }
    // Before the footer, canvas, or selected-surface lookup can use an ordinal,
    // invalidate it when its document/actions changed, including while hidden.
    viewer.sync_selection_scope(active_tab, &graph_state.graph);
    let default_rect = tiled_window_rect(viewport, egui::Align2::RIGHT_BOTTOM);
    let mut open = viewer.open;
    let mut hovered = HashSet::new();
    egui::Window::new("Actions")
        .open(&mut open)
        .default_rect(default_rect)
        .min_width(340.0_f32.min(default_rect.width()))
        .min_height(240.0_f32.min(default_rect.height()))
        .resizable(true)
        .show(ctx, |ui| {
            hovered = draw_contents(
                ui,
                graph_state,
                active_tab,
                viewer,
                action_edit,
                palette,
                intents,
            );
        });
    viewer.open = open;
    if !viewer.open {
        // Closing mid-pick would otherwise leave the scene armed with no visible
        // way to see or cancel it.
        action_edit.cancel();
    }
    let selected_surface = viewer.selected.and_then(|ordinal| {
        graph_state
            .graph
            .action_graph()
            .node_by_ordinal(ordinal)
            .and_then(|node| match (&node.action, &node.measurement_stabilizer) {
                (Action::Measure { name, .. }, Some(surface)) => {
                    Some((ordinal, name.clone(), surface.clone()))
                }
                _ => None,
            })
    });
    ActionViewerOutput {
        hovered,
        selected_surface,
    }
}

fn draw_contents(
    ui: &mut egui::Ui,
    graph_state: &GraphState,
    active_tab: EditorTabId,
    viewer: &mut ActionViewerState,
    action_edit: &mut ActionEditState,
    palette: &ThemePalette,
    intents: &mut UiIntentBuffer,
) -> HashSet<GraphElement> {
    let dag = graph_state.graph.action_graph();
    let nodes: Vec<ActionNodeView> = dag
        .ordered_nodes()
        .map(|node| ActionNodeView {
            ordinal: node.ordinal,
            action: node.action.clone(),
            label: action_label(&node.action, &graph_state.graph),
            kind: ActionNodeKind::of(&node.action),
            elements: action_elements(&node.action, &graph_state.graph),
            measurement_support: node.measurement_stabilizer.as_ref().map(|surface| {
                [
                    surface.interior_nodes.len(),
                    surface.interior_edges.len(),
                    surface.port_stabilizer.len(),
                ]
            }),
        })
        .collect();
    let edges = dag
        .dependencies()
        .map(|(from, to, dependency)| ActionEdgeView {
            from: from as u32,
            to: to as u32,
            dependency,
        })
        .collect::<Vec<_>>();

    draw_toolbar(ui, nodes.len(), viewer, action_edit, palette);
    draw_pick_banner(ui, action_edit, palette);
    if graph_state.graph.has_actions() && !dag.is_analyzed() {
        ui.label(
            egui::RichText::new("\u{f071} Action analysis is stale; run Validate Graph to refresh")
                .small()
                .color(palette.accent_warn),
        );
    }
    if let Some(err) = graph_state.graph.action_graph_error() {
        ui.label(
            egui::RichText::new(format!("\u{f057} {err}"))
                .small()
                .color(palette.accent_error),
        );
    }
    draw_edge_legend(ui, palette);
    ui.separator();

    // Footer first, bottom-up, so the canvas can take exactly what is left.
    // Guessing the footer's height grew the window a few pixels every frame: the
    // footer overshot the guess, the window resized to fit the overflow, and the
    // next frame had that much more to hand out.
    let mut hovered = HashSet::new();
    ui.allocate_ui_with_layout(
        ui.available_size(),
        egui::Layout::bottom_up(egui::Align::Min),
        |ui| {
            draw_footer(
                ui,
                &nodes,
                &graph_state.graph,
                viewer,
                action_edit,
                palette,
                intents,
            );
            let canvas_height = ui.available_height().max(CANVAS_MIN_HEIGHT);
            hovered = draw_canvas(
                ui,
                canvas_height,
                &nodes,
                &edges,
                (active_tab, graph_state.revision, dag.is_analyzed()),
                viewer,
                palette,
            );
        },
    );
    hovered
}

/// One add button per action form, plus the count and the camera reset. A button
/// opens a draft rather than inserting anything, and stays lit until the draft
/// commits or a second click drops it.
fn draw_toolbar(
    ui: &mut egui::Ui,
    action_count: usize,
    viewer: &mut ActionViewerState,
    action_edit: &mut ActionEditState,
    palette: &ThemePalette,
) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
        for kind in ACTION_KINDS {
            let drafting = action_edit.drafting() == Some(kind);
            let hint = match kind.pick_prompt() {
                "" => format!("Write a `{}` statement", kind.title().to_lowercase()),
                prompt => prompt.to_string(),
            };
            let button = theme::toggle_button(
                ui,
                palette,
                &format!("\u{f067} {}", kind.title()),
                palette.accent_primary,
                drafting,
            );
            if activated(&button.on_hover_text(hint)) {
                if drafting {
                    action_edit.cancel();
                } else {
                    action_edit.arm(kind);
                }
            }
        }
        ui.separator();
        ui.small(format!("{action_count} action(s)"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if activated(&ui.small_button("\u{f021}").on_hover_text("Reset view")) {
                viewer.reset_camera();
            }
        });
    });
}

/// The "you are mid-gesture" banner, directly under the button that started the
/// pick.
fn draw_pick_banner(ui: &mut egui::Ui, action_edit: &mut ActionEditState, palette: &ThemePalette) {
    let Some(kind) = action_edit.picking() else {
        return;
    };
    egui::Frame::new()
        .fill(palette.bg_active)
        .stroke(Stroke::new(1.0, palette.accent_primary))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("\u{f245} {}", kind.pick_prompt()))
                        .small()
                        .color(palette.text_bright),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if activated(&ui.small_button("Cancel")) {
                        action_edit.cancel();
                    }
                });
            });
        });
}

fn draw_edge_legend(ui: &mut egui::Ui, palette: &ThemePalette) {
    ui.horizontal_wrapped(|ui| {
        ui.small(egui::RichText::new("Edges:").color(palette.text_dim));
        for (dependency, label) in [
            (ActionDependency::Classical, "variable"),
            (ActionDependency::SelectiveSupport, "selective support"),
            (ActionDependency::ReadoutParity, "decoded readout"),
            (ActionDependency::BranchSupport, "branch support"),
            (
                ActionDependency::FeedbackAnticommutation,
                "feedback anticommutation",
            ),
        ] {
            ui.label(
                egui::RichText::new(format!("\u{2014} {label}"))
                    .small()
                    .color(edge_color(dependency, palette)),
            );
        }
    });
}

/// The pan/zoom DAG canvas. Returns the scene elements the hovered node names.
fn draw_canvas(
    ui: &mut egui::Ui,
    height: f32,
    nodes: &[ActionNodeView],
    edges: &[ActionEdgeView],
    layout_key: (EditorTabId, u64, bool),
    viewer: &mut ActionViewerState,
    palette: &ThemePalette,
) -> HashSet<GraphElement> {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        Sense::click_and_drag(),
    );
    if response.dragged() {
        viewer.pan += ui.ctx().input(|input| input.pointer.delta());
    }
    if response.hovered() {
        let scroll = ui.ctx().input(|input| input.smooth_scroll_delta.y);
        if scroll.abs() > f32::EPSILON {
            viewer.zoom = (viewer.zoom() * (scroll / 600.0).exp()).clamp(0.5, 2.5);
        }
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 6.0, palette.bg_dark);
    painter.rect_stroke(
        rect,
        6.0,
        Stroke::new(1.0, palette.border),
        egui::StrokeKind::Inside,
    );

    if nodes.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No actions yet.\nStart one with the buttons above.",
            FontId::proportional(12.0),
            palette.text_dim,
        );
        return HashSet::new();
    }

    refresh_layout(viewer, nodes, edges, layout_key);
    let placement = node_rects(rect, viewer);

    let mut hovered_ordinal = None;
    let mut hovered_elements = HashSet::new();
    for node in nodes {
        let Some(node_rect) = placement.get(&(node.ordinal as u32)).copied() else {
            continue;
        };
        if !rect.intersects(node_rect) {
            continue;
        }
        let node_response = ui.interact(
            node_rect,
            egui::Id::new(("action-node", node.ordinal)),
            Sense::click(),
        );
        if node_response.hovered() {
            hovered_ordinal = Some(node.ordinal);
            hovered_elements = node.elements.clone();
            node_response.clone().on_hover_text(&node.label);
        }
        if activated(&node_response) {
            viewer.selected = (viewer.selected != Some(node.ordinal)).then_some(node.ordinal);
        }
    }

    let zoom = viewer.zoom();
    for edge in edges {
        let (Some(from_rect), Some(to_rect)) = (placement.get(&edge.from), placement.get(&edge.to))
        else {
            continue;
        };
        draw_edge(
            &painter,
            *from_rect,
            *to_rect,
            edge.dependency,
            zoom,
            palette,
        );
    }
    for node in nodes {
        let Some(node_rect) = placement.get(&(node.ordinal as u32)).copied() else {
            continue;
        };
        draw_node(
            &painter,
            node_rect,
            node,
            hovered_ordinal == Some(node.ordinal),
            viewer.selected == Some(node.ordinal),
            zoom,
            palette,
        );
    }
    hovered_elements
}

fn refresh_layout(
    viewer: &mut ActionViewerState,
    nodes: &[ActionNodeView],
    edges: &[ActionEdgeView],
    layout_key: (EditorTabId, u64, bool),
) {
    if viewer.layout.key == Some(layout_key) {
        return;
    }
    let sizes: Vec<(u32, egui::Vec2)> = nodes
        .iter()
        .map(|node| (node.ordinal as u32, NODE_SIZE))
        .collect();
    let layout_edges = edges
        .iter()
        .map(|edge| (edge.from, edge.to))
        .collect::<Vec<_>>();
    let relative = compute_relative(&sizes, &layout_edges, COLUMN_GAP);
    viewer.layout = LayoutCache {
        key: Some(layout_key),
        positions: relative.positions,
        content: relative.content,
    };
}

fn node_rects(rect: Rect, viewer: &ActionViewerState) -> HashMap<u32, Rect> {
    let zoom = viewer.zoom();
    let content = egui::vec2(viewer.layout.content.0, viewer.layout.content.1) * zoom;
    let origin = rect.center() - content * 0.5 + viewer.pan;
    viewer
        .layout
        .positions
        .iter()
        .map(|(id, (x, y))| {
            (
                *id,
                Rect::from_min_size(origin + egui::vec2(x * zoom, y * zoom), NODE_SIZE * zoom),
            )
        })
        .collect()
}

fn draw_edge(
    painter: &egui::Painter,
    from: Rect,
    to: Rect,
    dependency: ActionDependency,
    zoom: f32,
    palette: &ThemePalette,
) {
    let start = egui::pos2(from.right(), from.center().y);
    let end = egui::pos2(to.left(), to.center().y);
    let color = edge_color(dependency, palette);
    let stroke = Stroke::new(1.4 * zoom, color);
    if dependency.is_implicit() {
        painter.extend(egui::Shape::dashed_line(
            &[start, end],
            stroke,
            7.0 * zoom,
            4.0 * zoom,
        ));
    } else {
        painter.line_segment([start, end], stroke);
    }
    // Arrowhead at the consumer end so the dependency direction reads without a
    // legend.
    let dir = (end - start).normalized();
    let head = 6.0 * zoom;
    let back = end - dir * head;
    let side = egui::vec2(-dir.y, dir.x) * head * 0.5;
    painter.add(egui::Shape::convex_polygon(
        vec![end, back + side, back - side],
        color,
        Stroke::NONE,
    ));
}

fn edge_color(dependency: ActionDependency, palette: &ThemePalette) -> Color32 {
    match dependency {
        ActionDependency::Classical => palette.border_bright,
        ActionDependency::SelectiveSupport | ActionDependency::ReadoutParity => {
            palette.accent_secondary
        }
        ActionDependency::BranchSupport => palette.accent_warn,
        ActionDependency::FeedbackAnticommutation => palette.orange,
    }
}

fn draw_node(
    painter: &egui::Painter,
    rect: Rect,
    node: &ActionNodeView,
    hovered: bool,
    selected: bool,
    zoom: f32,
    palette: &ThemePalette,
) {
    let accent = node.kind.color(palette);
    let fill = if selected {
        palette.bg_active
    } else if hovered {
        palette.bg_hover
    } else {
        palette.bg_surface
    };
    painter.rect_filled(rect, 5.0, fill);
    painter.rect_stroke(
        rect,
        5.0,
        Stroke::new(if selected { 2.0 } else { 1.25 }, accent),
        egui::StrokeKind::Outside,
    );
    // A colour bar rather than a coloured border alone: the border already
    // carries hover/selection weight, so the kind needs its own channel.
    painter.rect_filled(
        Rect::from_min_size(rect.min, egui::vec2(4.0 * zoom, rect.height())),
        2.0,
        accent,
    );

    let pad = 10.0 * zoom;
    let interior = (rect.width() - pad * 2.0).max(1.0);
    let mut job = egui::text::LayoutJob::simple(
        node.label.clone(),
        FontId::monospace((10.5 * zoom).clamp(5.0, 20.0)),
        palette.text_bright,
        interior,
    );
    job.wrap.max_rows = 2;
    let galley = painter.layout_job(job);
    painter.galley(
        egui::pos2(rect.left() + pad, rect.top() + 5.0 * zoom),
        galley,
        palette.text_bright,
    );
}

/// Details, edit, and delete for the selected node. The controls live here
/// rather than on every tile so the canvas stays readable and a destructive
/// click needs a deliberate selection first.
fn draw_footer(
    ui: &mut egui::Ui,
    nodes: &[ActionNodeView],
    graph: &bloq_graph::BlockGraph,
    viewer: &mut ActionViewerState,
    action_edit: &mut ActionEditState,
    palette: &ThemePalette,
    intents: &mut UiIntentBuffer,
) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let selected = viewer
            .selected
            .and_then(|ordinal| nodes.iter().find(|node| node.ordinal == ordinal));
        let Some(node) = selected else {
            ui.label(
                egui::RichText::new("Click a node to inspect, edit, or remove it")
                    .small()
                    .color(palette.text_dim),
            );
            return;
        };
        if activated(
            &theme::toggle_button(ui, palette, "\u{f044} Edit", palette.accent_primary, false)
                .on_hover_text("Reopen this action in the draft popup"),
        ) {
            action_edit.edit(node.ordinal, &node.action, graph);
        }
        if activated(
            &theme::toggle_button(ui, palette, "\u{f00d} Remove", palette.accent_error, false)
                .on_hover_text("Remove this action"),
        ) {
            intents.push(UiIntent::RemoveAction(node.ordinal));
            viewer.selected = None;
        }
        ui.add(
            egui::Label::new(
                egui::RichText::new(&node.label)
                    .small()
                    .color(palette.text_primary),
            )
            .truncate(),
        );
        if let Some([nodes, edges, ports]) = node.measurement_support {
            ui.small(
                egui::RichText::new(format!(
                    "surface: {nodes} nodes, {edges} edges, {ports} ports"
                ))
                .color(palette.accent_primary),
            );
        }
    });
}

fn action_label(action: &Action, graph: &bloq_graph::BlockGraph) -> String {
    match action {
        Action::Branch { target, condition } => graph
            .branch_by_target(*target)
            .map(|branch| format!("resolve {} if {condition}", branch.name))
            .unwrap_or_else(|| action.to_string()),
        _ => action.to_string(),
    }
}

#[derive(Clone, Copy)]
struct ActionEdgeView {
    from: u32,
    to: u32,
    dependency: ActionDependency,
}

/// One action rendered for the canvas.
struct ActionNodeView {
    ordinal: usize,
    /// Kept whole so the footer's edit control can reopen it as a draft.
    action: Action,
    label: String,
    kind: ActionNodeKind,
    elements: HashSet<GraphElement>,
    measurement_support: Option<[usize; 3]>,
}

/// Colour bucket for an action, so kinds are separable at a glance.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActionNodeKind {
    Measure,
    Resolve,
    Branch,
    Binding,
    Feedback,
    Discard,
}

impl ActionNodeKind {
    fn of(action: &Action) -> Self {
        match action {
            Action::Measure { .. } => Self::Measure,
            Action::Resolve { .. } => Self::Resolve,
            Action::Branch { .. } => Self::Branch,
            Action::Let { .. } => Self::Binding,
            Action::Feedback { .. } => Self::Feedback,
            Action::DiscardIf(..) => Self::Discard,
        }
    }

    fn color(self, palette: &ThemePalette) -> Color32 {
        match self {
            Self::Measure => palette.accent_primary,
            Self::Resolve => palette.accent_secondary,
            Self::Branch => palette.accent_warn,
            Self::Binding => palette.yellow,
            Self::Feedback => palette.orange,
            Self::Discard => palette.success,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn selection_is_scoped_to_tab_and_action_order_not_geometry() {
        let a = Action::Let {
            name: "a".into(),
            expr: bloq_graph::Expr::Var("input".into()),
        };
        let b = Action::Let {
            name: "b".into(),
            expr: bloq_graph::Expr::Var("input".into()),
        };
        let mut graph = bloq_graph::BlockGraph::default();
        graph
            .set_actions_lenient(vec![a.clone(), b.clone()])
            .unwrap();
        let tab = EditorTabId::new(1);
        let mut viewer = ActionViewerState::default();
        viewer.sync_selection_scope(tab, &graph);
        viewer.selected = Some(0);
        graph.add_block(bloq_graph::Block::new(
            glam::IVec3::ZERO,
            bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        viewer.sync_selection_scope(tab, &graph);
        assert_eq!(viewer.selected, Some(0));
        viewer.sync_selection_scope(EditorTabId::new(2), &graph);
        assert_eq!(
            viewer.selected, None,
            "same ordinal in another tab is not the selection"
        );
        viewer.selected = Some(0);
        graph.set_actions_lenient(vec![b, a.clone()]).unwrap();
        viewer.sync_selection_scope(EditorTabId::new(2), &graph);
        assert_eq!(
            viewer.selected, None,
            "reordering must not retarget a selection"
        );
        graph
            .set_actions_lenient(vec![a.clone(), a.clone()])
            .unwrap();
        viewer.sync_selection_scope(tab, &graph);
        viewer.selected = Some(0);
        graph.set_actions_lenient(vec![a]).unwrap();
        viewer.sync_selection_scope(tab, &graph);
        assert_eq!(
            viewer.selected, None,
            "identical neighbours still have distinct ordinals"
        );
    }

    use super::*;

    #[test]
    fn closing_the_window_drops_the_layout_cache() {
        let mut viewer = ActionViewerState {
            open: true,
            layout: LayoutCache {
                key: Some((EditorTabId::new(1), 7, true)),
                ..LayoutCache::default()
            },
            ..ActionViewerState::default()
        };

        viewer.toggle();

        assert!(!viewer.open);
        assert_eq!(viewer.layout.key, None);
    }

    /// `Default` leaves zoom at 0.0, and a snapshot restore can too; the view
    /// must not collapse to a point because of it.
    #[test]
    fn zoom_falls_back_to_one_when_uninitialized() {
        assert_eq!(ActionViewerState::default().zoom(), 1.0);
    }
}
