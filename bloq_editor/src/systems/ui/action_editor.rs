//! The action draft popup: one editor for all five action forms, used for both
//! adding and editing.
//!
//! Drafts commit by rendering back to BLOG source and handing it to
//! [`parse_actions`] rather than by building [`Action`] values directly, so the
//! free-text fields inherit the real grammar's parser and error messages and
//! there is no second expression syntax to keep in step.

use std::collections::HashSet;

use super::intents::{UiIntent, UiIntentBuffer};
use super::{activated, field_owns_shortcuts};
use crate::components::GraphElement;
use crate::theme::{self, ThemePalette, ThemePreset};
use bevy::prelude::Resource;
use bevy_egui::egui;
use bloq_graph::{
    Action, BlockGraph, BlockKind, BranchRegion, MeasureTarget, PauliBasis, parse_actions,
};
use glam::IVec3;

/// The action forms, in the order the Actions window offers them: the three
/// whose single operand the pointer can name come first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActionKind {
    Measure,
    Resolve,
    Let,
    DiscardIf,
    Feedback,
}

pub(crate) const ACTION_KINDS: [ActionKind; 5] = [
    ActionKind::Measure,
    ActionKind::Resolve,
    ActionKind::Let,
    ActionKind::DiscardIf,
    ActionKind::Feedback,
];

/// What a scene click contributes to a draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetRole {
    /// Nothing to pick — the statement is pure text.
    None,
    /// One node or edge: anything the graph can measure.
    Measurable,
    /// One selective block or named branch region.
    Resolvable,
    /// One or more blocks, each carrying its own correction basis.
    PauliBlocks,
}

impl ActionKind {
    pub(crate) const fn title(self) -> &'static str {
        match self {
            ActionKind::Measure => "Measure",
            ActionKind::Resolve => "Resolve",
            ActionKind::Let => "Let",
            ActionKind::DiscardIf => "Discard If",
            ActionKind::Feedback => "Feedback",
        }
    }

    const fn targets(self) -> TargetRole {
        match self {
            ActionKind::Measure => TargetRole::Measurable,
            ActionKind::Resolve => TargetRole::Resolvable,
            ActionKind::Feedback => TargetRole::PauliBlocks,
            ActionKind::Let | ActionKind::DiscardIf => TargetRole::None,
        }
    }

    /// Placeholder for the expression field, or `None` for the kinds without one.
    const fn expr_hint(self) -> Option<&'static str> {
        match self {
            ActionKind::Let | ActionKind::DiscardIf | ActionKind::Resolve => {
                Some("boolean expression over measurement outcomes")
            }
            ActionKind::Feedback => Some("optional \u{2014} empty means unconditional"),
            ActionKind::Measure => None,
        }
    }

    /// Whether the popup collects a variable name to bind.
    const fn binds_a_variable(self) -> bool {
        matches!(self, ActionKind::Measure | ActionKind::Let)
    }

    /// The prompt shown while this kind's pick is armed, in the popup and the
    /// status bar.
    pub(crate) const fn pick_prompt(self) -> &'static str {
        match self.targets() {
            TargetRole::None => "",
            TargetRole::Measurable => "Click a block or pipe to measure",
            TargetRole::Resolvable => "Click a selective block or highlighted branch region",
            TargetRole::PauliBlocks => "Click each block the correction acts on",
        }
    }

    /// Label for the expression field, which is a condition for most kinds but
    /// a defining expression for `let`.
    const fn expr_label(self) -> &'static str {
        match self {
            ActionKind::Let => "Expression",
            _ => "Condition",
        }
    }
}

/// A statement under construction.
///
/// One struct rather than a variant per kind: the fields overlap heavily, and
/// [`draft_blog_source`] is the single place that decides which of them a given
/// kind actually reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActionDraft {
    pub(crate) kind: ActionKind,
    /// Picked scene targets, in click order. Only `Measure` ever stores an edge.
    targets: Vec<MeasureTarget>,
    /// Correction basis per target; only `Feedback` reads it, and it is kept the
    /// same length as `targets` so the two can be zipped.
    paulis: Vec<PauliBasis>,
    name: String,
    expr: String,
    /// Named structural branch when `Resolve` targets a whole region.
    branch_target: Option<String>,
    /// Source ordinal and complete action list: identical neighbours still have
    /// distinct positions, so checking only the selected value cannot detect deletion.
    editing: Option<(usize, Vec<Action>)>,
    /// True while scene clicks feed `targets`.
    picking: bool,
}

impl ActionDraft {
    fn new(kind: ActionKind) -> Self {
        Self {
            kind,
            targets: Vec::new(),
            paulis: Vec::new(),
            name: String::new(),
            expr: String::new(),
            branch_target: None,
            editing: None,
            // A kind with operands starts armed, so `+ Feedback` then a click is
            // the whole gesture.
            picking: kind.targets() != TargetRole::None,
        }
    }

    /// How many targets the statement takes. `PauliBlocks` is unbounded, so that
    /// pick stays armed until the user disarms it.
    fn wanted_targets(&self) -> usize {
        match self.kind.targets() {
            TargetRole::None => 0,
            TargetRole::Measurable | TargetRole::Resolvable => 1,
            TargetRole::PauliBlocks => usize::MAX,
        }
    }

    /// Block positions among the targets, dropping the edge form that only
    /// `Measure` can produce.
    fn block_positions(&self) -> Vec<IVec3> {
        self.targets
            .iter()
            .filter_map(|target| match target {
                MeasureTarget::Node(pos) => Some(*pos),
                MeasureTarget::Edge { .. } => None,
            })
            .collect()
    }

    /// Folds a clicked element into the draft, rejecting one the kind cannot use.
    fn accept(&mut self, element: GraphElement, graph: &BlockGraph) -> Result<(), String> {
        match self.kind.targets() {
            TargetRole::None => return Ok(()),
            TargetRole::Measurable => {
                self.targets = vec![measure_target_for(element, graph)?];
            }
            TargetRole::Resolvable => {
                if let Some(branch) = branch_for_element(graph, element) {
                    self.targets = vec![MeasureTarget::Node(branch.target)];
                    self.branch_target = Some(branch.name.clone());
                } else {
                    let target = self.checked_block(element, graph)?;
                    let MeasureTarget::Node(position) = target else {
                        unreachable!("checked_block always returns a node")
                    };
                    if !matches!(
                        graph.get_block(position).map(bloq_graph::Block::kind),
                        Some(BlockKind::Selective(_))
                    ) {
                        return Err(format!("Block at {position} is not a selective block"));
                    }
                    self.targets = vec![target];
                    self.branch_target = None;
                }
            }
            TargetRole::PauliBlocks => {
                self.push_unique(self.checked_block(element, graph)?)?;
                self.paulis.push(PauliBasis::X);
            }
        }
        if self.targets.len() >= self.wanted_targets() {
            self.picking = false;
        }
        if self.kind == ActionKind::Measure && self.name.trim().is_empty() {
            self.name = next_measure_name(graph);
        }
        Ok(())
    }

    /// The block a click names, rejected unless the kind accepts it.
    fn checked_block(
        &self,
        element: GraphElement,
        graph: &BlockGraph,
    ) -> Result<MeasureTarget, String> {
        let GraphElement::Block(pos) = element else {
            return Err(format!("{} targets a block, not a pipe", self.kind.title()));
        };
        match graph.get_block(pos) {
            Some(_) => Ok(MeasureTarget::Node(pos)),
            None => Err(format!("No block at {pos}")),
        }
    }

    /// Appends a target, refusing a repeat: re-clicking one is a misclick, not
    /// a request for a duplicate.
    fn push_unique(&mut self, target: MeasureTarget) -> Result<(), String> {
        if self.targets.contains(&target) {
            return Err("That target is already in the list".into());
        }
        self.targets.push(target);
        Ok(())
    }

    /// Loads an existing action for editing, so the popup is the edit surface
    /// as well as the add one.
    fn from_action(ordinal: usize, action: &Action, graph: &BlockGraph) -> Self {
        let mut draft = match action {
            Action::Measure { target, name } => {
                let mut draft = Self::new(ActionKind::Measure);
                draft.targets = vec![*target];
                draft.name = name.clone();
                draft
            }
            Action::Resolve { target, condition } => {
                let mut draft = Self::new(ActionKind::Resolve);
                draft.targets = vec![MeasureTarget::Node(*target)];
                draft.expr = condition.to_string();
                draft
            }
            Action::Branch { target, condition } => {
                let mut draft = Self::new(ActionKind::Resolve);
                draft.targets = vec![MeasureTarget::Node(*target)];
                draft.expr = condition.to_string();
                draft.branch_target = graph
                    .branch_by_target(*target)
                    .map(|branch| branch.name.clone());
                draft
            }
            Action::Let { name, expr } => {
                let mut draft = Self::new(ActionKind::Let);
                draft.name = name.clone();
                draft.expr = expr.to_string();
                draft
            }
            Action::DiscardIf(expr) => {
                let mut draft = Self::new(ActionKind::DiscardIf);
                draft.expr = expr.to_string();
                draft
            }
            Action::Feedback { targets, condition } => {
                let mut draft = Self::new(ActionKind::Feedback);
                draft.targets = targets
                    .iter()
                    .map(|target| {
                        target
                            .direction
                            .map_or(MeasureTarget::Node(target.target), |dir| {
                                MeasureTarget::Edge {
                                    src: target.target,
                                    dir,
                                }
                            })
                    })
                    .collect();
                draft.paulis = targets.iter().map(|target| target.pauli).collect();
                draft.expr = condition
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                draft
            }
        };
        // An edit starts with every operand already supplied; re-picking is an
        // explicit choice from the popup.
        draft.picking = false;
        draft.editing = Some((ordinal, graph.actions()));
        draft
    }
}

/// The draft in progress, if any, plus the two bits of transient input state the
/// popup needs.
#[derive(Resource, Default, Clone)]
pub(crate) struct ActionEditState {
    pub(crate) draft: Option<ActionDraft>,
    /// Focuses the popup's first text field on the frame after it opens, so the
    /// gesture flows straight into typing.
    focus_field: bool,
}

impl ActionEditState {
    /// Starts a fresh draft of `kind`, replacing any draft in progress.
    pub(crate) fn arm(&mut self, kind: ActionKind) {
        self.draft = Some(ActionDraft::new(kind));
        self.focus_field = true;
    }

    /// Starts a draft that will replace the action at `ordinal`.
    pub(crate) fn edit(&mut self, ordinal: usize, action: &Action, graph: &BlockGraph) {
        self.draft = Some(ActionDraft::from_action(ordinal, action, graph));
        self.focus_field = true;
    }

    pub(crate) fn cancel(&mut self) {
        self.draft = None;
    }

    /// A removed or replaced source action must not redirect this draft to its neighbour.
    fn cancel_stale_edit(&mut self, graph: &BlockGraph) {
        if self
            .draft
            .as_ref()
            .and_then(|draft| draft.editing.as_ref())
            .is_some_and(|(_, expected)| {
                !graph
                    .action_graph()
                    .ordered_nodes()
                    .map(|node| &node.action)
                    .eq(expected.iter())
            })
        {
            self.cancel();
        }
    }

    /// The kind of draft in progress, when it is a *new* action. An edit is
    /// excluded so the window's add buttons do not light up for it.
    pub(crate) fn drafting(&self) -> Option<ActionKind> {
        self.draft
            .as_ref()
            .filter(|draft| draft.editing.is_none())
            .map(|draft| draft.kind)
    }

    /// The draft's kind while it is waiting for a scene click.
    pub(crate) fn picking(&self) -> Option<ActionKind> {
        self.draft
            .as_ref()
            .filter(|draft| draft.picking)
            .map(|draft| draft.kind)
    }

    /// Elements the armed pick would accept, pulsed in the scene.
    ///
    /// Broad picks highlight nothing; resolve highlights its selective blocks
    /// and every element of each displayed branch arm.
    pub(crate) fn pick_candidates(&self, graph: &BlockGraph) -> HashSet<GraphElement> {
        let Some(_) = self
            .picking()
            .filter(|kind| kind.targets() == TargetRole::Resolvable)
        else {
            return HashSet::new();
        };
        let mut candidates = graph
            .blocks()
            .filter(|block| matches!(block.kind(), BlockKind::Selective(_)))
            .map(|block| GraphElement::Block(block.pos()))
            .collect::<HashSet<_>>();
        for branch in graph.branch_definitions() {
            candidates.extend(shown_arm_elements(branch));
        }
        candidates
    }

    /// Supplies a clicked element to the armed draft. Returns an error message
    /// when the element cannot serve as one of its targets.
    pub(crate) fn accept_pick(
        &mut self,
        element: GraphElement,
        graph: &BlockGraph,
    ) -> Result<(), String> {
        let Some(draft) = self.draft.as_mut().filter(|draft| draft.picking) else {
            return Ok(());
        };
        draft.accept(element, graph)?;
        self.focus_field = true;
        Ok(())
    }
}

fn branch_for_element(graph: &BlockGraph, element: GraphElement) -> Option<&BranchRegion> {
    let element = element.canonical();
    match element {
        GraphElement::Block(position) => graph.shown_branch_at(position),
        GraphElement::Pipe(..) => graph.branch_definitions().iter().find(|branch| {
            branch.shown_arm().pipes().any(|pipe| {
                let (src, dst) = pipe.endpoints();
                GraphElement::Pipe(src, dst).canonical() == element
            })
        }),
    }
}

fn shown_arm_elements(branch: &BranchRegion) -> HashSet<GraphElement> {
    branch
        .shown_arm()
        .blocks()
        .map(|block| GraphElement::Block(block.pos()))
        .chain(branch.shown_arm().pipes().map(|pipe| {
            let (src, dst) = pipe.endpoints();
            GraphElement::Pipe(src, dst).canonical()
        }))
        .collect()
}

/// The measurement target a clicked element stands for.
fn measure_target_for(element: GraphElement, graph: &BlockGraph) -> Result<MeasureTarget, String> {
    match element {
        GraphElement::Block(pos) => Ok(MeasureTarget::Node(pos)),
        GraphElement::Pipe(u, v) => graph
            .get_pipe(u, v)
            .map(|pipe| MeasureTarget::Edge {
                src: pipe.src(),
                dir: pipe.dir(),
            })
            .ok_or_else(|| format!("No pipe between {u} and {v}")),
    }
}

/// First unused `m0`, `m1`, … so a measure can be confirmed without typing.
fn next_measure_name(graph: &BlockGraph) -> String {
    let taken: HashSet<&str> = graph
        .action_graph()
        .inputs()
        .chain(
            graph
                .action_graph()
                .ordered_nodes()
                .filter_map(|node| action_binding(&node.action)),
        )
        .collect();
    (0..)
        .map(|i| format!("m{i}"))
        .find(|name| !taken.contains(name.as_str()))
        .expect("an unbounded counter always yields a name outside a finite set")
}

/// Names bound by the graph's actions, offered as insertable chips under an
/// expression field.
fn action_binding(action: &Action) -> Option<&str> {
    match action {
        Action::Measure { name, .. } | Action::Let { name, .. } => Some(name),
        _ => None,
    }
}

/// Inputs are in scope throughout this definition; action bindings become
/// available only after their statement, never from a sibling definition.
fn draft_variables(graph: &BlockGraph, draft: &ActionDraft) -> Vec<String> {
    let end = draft
        .editing
        .as_ref()
        .map_or(usize::MAX, |(index, _)| *index);
    let mut seen = HashSet::new();
    graph
        .action_graph()
        .inputs()
        .chain(
            graph
                .action_graph()
                .ordered_nodes()
                .take(end)
                .filter_map(|node| action_binding(&node.action)),
        )
        .filter(|name| seen.insert(*name))
        .map(str::to_owned)
        .collect()
}

/// Graph elements an action names, for cross-highlighting from the action DAG.
pub(crate) fn action_elements(action: &Action, graph: &BlockGraph) -> HashSet<GraphElement> {
    match action {
        Action::Measure {
            target: MeasureTarget::Node(pos),
            ..
        } => HashSet::from([GraphElement::Block(*pos)]),
        Action::Measure {
            target: MeasureTarget::Edge { src, dir },
            ..
        } => HashSet::from([GraphElement::Pipe(*src, *src + dir.to_ivec3()).canonical()]),
        Action::Resolve { target, .. } => HashSet::from([GraphElement::Block(*target)]),
        Action::Branch { target, .. } => graph
            .branch_by_target(*target)
            .map(shown_arm_elements)
            .unwrap_or_else(|| HashSet::from([GraphElement::Block(*target)])),
        Action::Feedback { targets, .. } => targets
            .iter()
            .map(|target| match target.direction {
                Some(dir) => {
                    GraphElement::Pipe(target.target, target.target + dir.to_ivec3()).canonical()
                }
                None => GraphElement::Block(target.target),
            })
            .collect(),
        Action::Let { .. } | Action::DiscardIf(..) => HashSet::new(),
    }
}

// =============================================================================
// Draft popup
// =============================================================================

/// Fixed rather than derived from the title, so a popup that changes kind keeps
/// both the position the user dragged it to and its widget ids.
const DRAFT_WINDOW_ID: &str = "action-draft-window";

/// Draws the draft popup, queuing the add or replace intent on commit.
///
/// The draft is re-parsed every frame, so the popup shows the exact statement it
/// will write (or what is missing) while it is still being edited.
pub(crate) fn draw_action_editor_window(
    ctx: &egui::Context,
    tab_id: crate::resources::EditorTabId,
    state: &mut ActionEditState,
    graph: &BlockGraph,
    theme_preset: ThemePreset,
    intents: &mut UiIntentBuffer,
) {
    state.cancel_stale_edit(graph);
    let Some(draft) = state.draft.as_ref() else {
        return;
    };
    let palette = theme::palette(theme_preset);
    let verb = if draft.editing.is_some() {
        "Edit"
    } else {
        "Add"
    };
    let variables = draft_variables(graph, draft);
    let kind = draft.kind;

    egui::Window::new(format!("{verb} {} Action", kind.title()))
        .id(egui::Id::new((DRAFT_WINDOW_ID, tab_id)))
        .collapsible(false)
        .resizable(false)
        .default_pos([620.0, 120.0])
        .default_width(340.0)
        .frame(
            egui::Frame::new()
                .fill(palette.bg_panel)
                .stroke(egui::Stroke::new(1.0, palette.border))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::same(12)),
        )
        .show(ctx, |ui| {
            let focus = std::mem::take(&mut state.focus_field);
            draw_targets(ui, state, graph, palette);
            let owns_keyboard = draw_fields(ui, state, &variables, focus, palette);
            let preview = draft_preview(state.draft.as_ref().expect("draft is open"));
            ui.add_space(8.0);
            draw_preview(ui, &preview, palette);
            ui.add_space(10.0);
            draw_commit_row(ui, state, &preview, verb, kind, owns_keyboard, intents);
        });
}

/// The picked targets, their per-target options, and the arm/disarm control.
fn draw_targets(
    ui: &mut egui::Ui,
    state: &mut ActionEditState,
    graph: &BlockGraph,
    palette: &ThemePalette,
) {
    let Some(draft) = state.draft.as_mut() else {
        return;
    };
    if draft.kind.targets() == TargetRole::None {
        return;
    }
    theme::section_header(ui, palette, "Targets");

    let mut remove = None;
    for index in 0..draft.targets.len() {
        ui.push_id(draft.targets[index], |ui| {
            ui.horizontal(|ui| {
                if draft.kind == ActionKind::Feedback {
                    for basis in [PauliBasis::X, PauliBasis::Y, PauliBasis::Z] {
                        let selected = draft.paulis[index] == basis;
                        if ui
                            .selectable_label(selected, format!("{basis}"))
                            .on_hover_text("Correction basis")
                            .clicked()
                        {
                            draft.paulis[index] = basis;
                        }
                    }
                }
                ui.label(
                    egui::RichText::new(target_description(draft.targets[index], graph))
                        .monospace()
                        .small()
                        .color(palette.text_bright),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if activated(
                        &ui.small_button("\u{f00d}")
                            .on_hover_text("Drop this target"),
                    ) {
                        remove = Some(index);
                    }
                });
            });
        });
    }
    if let Some(index) = remove {
        draft.targets.remove(index);
        if index < draft.paulis.len() {
            draft.paulis.remove(index);
        }
    }

    let armed = draft.picking;
    let label = if armed {
        "\u{f245} Picking\u{2026}"
    } else {
        "\u{f245} Pick in scene"
    };
    if activated(&theme::toggle_button(
        ui,
        palette,
        label,
        palette.accent_primary,
        armed,
    )) {
        draft.picking = !draft.picking;
    }
    // Kept last in the section: a hint that comes and goes cannot renumber
    // anything below it when nothing is below it.
    let hint = if armed {
        draft.kind.pick_prompt()
    } else if draft.targets.is_empty() {
        "No target picked yet"
    } else {
        ""
    };
    ui.label(egui::RichText::new(hint).small().color(palette.text_dim));
}

/// The name, expression, and observable fields, each drawn only for the kinds
/// that read them.
fn draw_fields(
    ui: &mut egui::Ui,
    state: &mut ActionEditState,
    variables: &[String],
    focus: bool,
    palette: &ThemePalette,
) -> bool {
    let mut owns_keyboard = false;
    let Some(draft) = state.draft.as_mut() else {
        return false;
    };
    let kind = draft.kind;

    if kind.binds_a_variable() {
        theme::section_header(ui, palette, "Variable");
        let field = ui.add(
            egui::TextEdit::singleline(&mut draft.name)
                .id_salt("action-name")
                .hint_text("outcome name")
                .desired_width(f32::INFINITY),
        );
        owns_keyboard |= field_owns_shortcuts(ui, field.id);
        if focus {
            field.request_focus();
        }
    }

    if let Some(hint) = kind.expr_hint() {
        theme::section_header(ui, palette, kind.expr_label());
        let field = ui.add(
            egui::TextEdit::singleline(&mut draft.expr)
                .id_salt("action-expr")
                .hint_text(hint)
                .desired_width(f32::INFINITY),
        );
        owns_keyboard |= field_owns_shortcuts(ui, field.id);
        // The name field, when there is one, owns the initial focus.
        if focus && !kind.binds_a_variable() {
            field.request_focus();
        }
        draw_condition_pad(ui, variables, &mut draft.expr, palette);
    }
    owns_keyboard
}

/// The commit and cancel controls.
fn draw_commit_row(
    ui: &mut egui::Ui,
    state: &mut ActionEditState,
    preview: &Result<String, String>,
    verb: &str,
    kind: ActionKind,
    owns_keyboard: bool,
    intents: &mut UiIntentBuffer,
) {
    ui.horizontal(|ui| {
        let commit = ui.add_enabled(
            preview.is_ok(),
            egui::Button::new(format!("\u{f067} {verb} {}", kind.title())),
        );
        let confirmed = owns_keyboard
            && preview.is_ok()
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        if activated(&commit) || confirmed {
            commit_draft(state, intents);
        }
        if activated(&ui.button("Cancel"))
            || (owns_keyboard
                && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)))
        {
            state.cancel();
        }
    });
}

/// Clickable variables and operators for an expression field.
///
/// Chips append at the end of the field rather than at the caret: egui does not
/// hand back a stable caret across a rebuilt widget, and an expression is short
/// enough that building it left-to-right is the natural order anyway.
fn draw_condition_pad(
    ui: &mut egui::Ui,
    variables: &[String],
    condition: &mut String,
    palette: &ThemePalette,
) {
    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 3.0);
        ui.label(
            egui::RichText::new("Variables")
                .small()
                .color(palette.text_dim),
        );
        if variables.is_empty() {
            ui.label(
                egui::RichText::new("no inputs or earlier bindings")
                    .small()
                    .color(palette.text_dim),
            );
        }
        for variable in variables {
            let chip = egui::Button::new(
                egui::RichText::new(variable)
                    .monospace()
                    .small()
                    .color(palette.accent_primary),
            )
            .small()
            .fill(palette.bg_surface)
            .stroke(egui::Stroke::new(1.0, palette.accent_primary));
            if activated(&ui.add(chip)) {
                append_token(condition, variable);
            }
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 3.0);
        ui.label(
            egui::RichText::new("Operators")
                .small()
                .color(palette.text_dim),
        );
        for (token, tooltip) in CONDITION_OPERATORS {
            if activated(&ui.small_button(*token).on_hover_text(*tooltip)) {
                append_token(condition, token);
            }
        }
        if activated(
            &ui.small_button("Clear")
                .on_hover_text("Empty the expression"),
        ) {
            condition.clear();
        }
    });
}

/// Operator palette for the expression field, mirroring the BLOG expression
/// grammar's operators.
const CONDITION_OPERATORS: &[(&str, &str)] = &[
    ("!", "not"),
    ("^", "xor"),
    ("&", "and"),
    ("|", "or"),
    ("(", "open group"),
    (")", "close group"),
];

/// Appends `token`, inserting a separating space only where the expression
/// reads better for one: a prefix `!` and an open paren bind tight to what
/// follows, and a close paren binds tight to what precedes it.
fn append_token(condition: &mut String, token: &str) {
    let needs_space =
        !condition.is_empty() && !condition.ends_with([' ', '(', '!']) && token != ")";
    if needs_space {
        condition.push(' ');
    }
    condition.push_str(token);
}

/// The statement the draft currently stands for, or why it does not parse.
fn draft_preview(draft: &ActionDraft) -> Result<String, String> {
    let Some(source) = draft_blog_source(draft) else {
        return Err(missing_field_hint(draft));
    };
    match parse_draft_action(draft) {
        Ok(_) => Ok(source),
        Err(err) => Err(err),
    }
}

/// What the draft is still waiting for, in the order the popup asks for it.
fn missing_field_hint(draft: &ActionDraft) -> String {
    if draft.kind.targets() != TargetRole::None && draft.targets.is_empty() {
        return "Pick a target in the scene".to_string();
    }
    if draft.kind.binds_a_variable() && draft.name.trim().is_empty() {
        return "Enter a variable name".to_string();
    }
    format!("Enter {}", draft.kind.expr_label().to_lowercase())
}

/// The statement the draft will write, or the reason it cannot yet. A
/// graph-level rejection is *not* shown here: an incomplete action list is a
/// normal stage of construction, so it is carried on the graph and reported in
/// the Actions window rather than blocking the commit.
fn draw_preview(ui: &mut egui::Ui, preview: &Result<String, String>, palette: &ThemePalette) {
    match preview {
        Ok(source) => ui.label(
            egui::RichText::new(format!("\u{f058} {source}"))
                .monospace()
                .small()
                .color(palette.success),
        ),
        Err(reason) => ui.label(
            egui::RichText::new(format!("\u{f059} {reason}"))
                .small()
                .color(palette.text_dim),
        ),
    };
}

/// Commits the draft. Only ever reached with a valid preview — the commit
/// control is disabled otherwise — so the parse here cannot fail.
fn commit_draft(state: &mut ActionEditState, intents: &mut UiIntentBuffer) {
    let Some(draft) = state.draft.as_ref() else {
        return;
    };
    let Ok(action) = parse_draft_action(draft) else {
        return;
    };
    match &draft.editing {
        Some((index, expected)) => intents.push(UiIntent::ReplaceAction {
            index: *index,
            expected: expected.clone(),
            action,
        }),
        None => intents.push(UiIntent::AddAction(action)),
    }
    // The applying system closes this draft only after the graph accepts it.
}

fn parse_draft_action(draft: &ActionDraft) -> Result<Action, String> {
    let source =
        draft_blog_source_with_branch(draft, None).ok_or_else(|| missing_field_hint(draft))?;
    let [mut action] = parse_actions(&source, |_, _| None)
        .map_err(|err| err.to_string())?
        .try_into()
        .map_err(|_| "draft must contain one action".to_string())?;
    if draft.branch_target.is_some()
        && let Action::Resolve { target, condition } = &action
    {
        action = Action::Branch {
            target: *target,
            condition: condition.clone(),
        }
    }
    Ok(action)
}

/// The BLOG statement a draft stands for, or `None` while a required operand or
/// field is still missing.
///
/// This writer must agree with [`Action`]'s display output, which the BLOG
/// buffer and DAG view show.
fn draft_blog_source(draft: &ActionDraft) -> Option<String> {
    draft_blog_source_with_branch(draft, draft.branch_target.as_deref())
}

// Parsing uses the numeric target, then restores the Branch variant; the preview
// uses its authored name. Both paths borrow the draft, including its edit guard.
fn draft_blog_source_with_branch(draft: &ActionDraft, branch: Option<&str>) -> Option<String> {
    let name = draft.name.trim();
    let expr = draft.expr.trim();
    let blocks = draft.block_positions();
    Some(match draft.kind {
        ActionKind::Measure => {
            let target = draft.targets.first()?;
            if name.is_empty() {
                return None;
            }
            format!("{name} = measure {}", measure_target_label(*target))
        }
        ActionKind::Resolve => {
            let target = blocks.first()?;
            if expr.is_empty() {
                return None;
            }
            match branch {
                Some(name) => format!("resolve {name} if {expr}"),
                None => format!("resolve {target} if {expr}"),
            }
        }
        ActionKind::Let => {
            if name.is_empty() || expr.is_empty() {
                return None;
            }
            format!("{name} = {expr}")
        }
        ActionKind::DiscardIf => {
            if expr.is_empty() {
                return None;
            }
            format!("discard if {expr}")
        }
        ActionKind::Feedback => {
            if draft.targets.is_empty() {
                return None;
            }
            let corrections = draft
                .targets
                .iter()
                .zip(&draft.paulis)
                .map(|(target, pauli)| format!("{pauli} {}", measure_target_label(*target)))
                .collect::<Vec<_>>()
                .join(", ");
            if expr.is_empty() {
                format!("feedback {corrections}")
            } else {
                format!("feedback {corrections} if {expr}")
            }
        }
    })
}

fn measure_target_label(target: MeasureTarget) -> String {
    match target {
        MeasureTarget::Node(pos) => format!("{pos}"),
        MeasureTarget::Edge { src, dir } => format!("{src} -> {dir}"),
    }
}

/// Target line for the popup, naming the kind of thing that was clicked so a
/// misclick is obvious before the action is committed.
fn target_description(target: MeasureTarget, graph: &BlockGraph) -> String {
    match target {
        MeasureTarget::Node(pos) => graph
            .branch_by_target(pos)
            .map(|branch| format!("{}  (branch region)", branch.name))
            .unwrap_or_else(|| match graph.get_block(pos) {
                Some(block) => format!("{pos}  ({})", block.kind()),
                None => format!("{pos}"),
            }),
        MeasureTarget::Edge { src, dir } => format!("{src} -> {dir}  (pipe)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::{Block, BranchArm, CubeKind, Direction, Pipe, SelectiveKind};
    use glam::ivec3;

    #[test]
    fn text_and_enter_in_one_frame_commit_the_current_draft() {
        let ctx = egui::Context::default();
        theme::setup_theme(&ctx, ThemePreset::default());
        let mut state = ActionEditState::default();
        state.arm(ActionKind::Let);
        state.draft.as_mut().unwrap().expr = "m0".into();
        let mut intents = UiIntentBuffer::default();
        for events in [
            vec![],
            vec![
                egui::Event::Text("a".into()),
                egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        ] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    draw_action_editor_window(
                        ui.ctx(),
                        crate::resources::EditorTabId::new(1),
                        &mut state,
                        &two_block_graph(),
                        ThemePreset::default(),
                        &mut intents,
                    );
                },
            );
            output.textures_delta.clear();
        }
        assert!(
            matches!(intents.drain().next(), Some(UiIntent::AddAction(Action::Let { name, .. })) if name == "a")
        );
    }

    #[test]
    fn draft_shortcuts_belong_to_its_fields_not_other_editors() {
        for own_focus in [false, true] {
            for key in [egui::Key::Enter, egui::Key::Escape] {
                let ctx = egui::Context::default();
                theme::setup_theme(&ctx, ThemePreset::default());
                let mut state = ActionEditState::default();
                state.arm(ActionKind::Let);
                let draft = state.draft.as_mut().unwrap();
                draft.name = "a".into();
                draft.expr = "m0".into();
                let mut intents = UiIntentBuffer::default();
                let mut outside = String::new();
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
                            if first && !own_focus {
                                field.request_focus();
                            }
                            let preview = draft_preview(state.draft.as_ref().unwrap());
                            let owns_keyboard = draw_fields(
                                ui,
                                &mut state,
                                &[],
                                first && own_focus,
                                theme::palette(ThemePreset::default()),
                            );
                            draw_commit_row(
                                ui,
                                &mut state,
                                &preview,
                                "Add",
                                ActionKind::Let,
                                owns_keyboard,
                                &mut intents,
                            );
                        },
                    );
                    output.textures_delta.clear();
                }
                let queued = intents.drain().collect::<Vec<_>>();
                assert_eq!(
                    state.draft.is_none(),
                    own_focus && key == egui::Key::Escape,
                    "{key:?}, own_focus={own_focus}"
                );
                assert_eq!(
                    queued.len(),
                    usize::from(own_focus && key == egui::Key::Enter)
                );
                if !own_focus {
                    assert!(ctx.input(|input| input.key_pressed(key)));
                }
            }
        }
    }

    #[test]
    fn draft_variables_include_inputs_but_not_current_or_later_bindings() {
        let mut graph = two_block_graph();
        let actions = vec![
            Action::Let {
                name: "earlier".into(),
                expr: bloq_graph::Expr::Var("m0".into()),
            },
            Action::Let {
                name: "current".into(),
                expr: bloq_graph::Expr::Var("earlier".into()),
            },
            Action::Let {
                name: "later".into(),
                expr: bloq_graph::Expr::Var("current".into()),
            },
        ];
        graph
            .set_actions_with_inputs(actions.clone(), ["m0".to_string()])
            .unwrap();
        let editing = ActionDraft::from_action(1, &actions[1], &graph);
        assert_eq!(draft_variables(&graph, &editing), ["m0", "earlier"]);
        assert_eq!(
            draft_variables(&graph, &ActionDraft::new(ActionKind::Let)),
            ["m0", "earlier", "current", "later"]
        );
        assert_eq!(next_measure_name(&graph), "m1");
        let mut input_only = BlockGraph::default();
        input_only
            .set_actions_with_inputs(vec![], ["enabled".to_string()])
            .unwrap();
        assert_eq!(
            draft_variables(&input_only, &ActionDraft::new(ActionKind::Let)),
            ["enabled"]
        );
    }

    fn two_block_graph() -> BlockGraph {
        let mut graph = BlockGraph::default();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph
    }

    fn every_action_form() -> Vec<Action> {
        vec![
            Action::Measure {
                target: MeasureTarget::Edge {
                    src: IVec3::ZERO,
                    dir: Direction::XPLUS,
                },
                name: "mx".into(),
            },
            Action::Resolve {
                target: IVec3::X,
                condition: bloq_graph::Expr::Var("mx".into()),
            },
            Action::Let {
                name: "parity".into(),
                expr: bloq_graph::Expr::Var("mx".into()),
            },
            Action::DiscardIf(bloq_graph::Expr::Not(Box::new(bloq_graph::Expr::Var(
                "mx".into(),
            )))),
            Action::Feedback {
                targets: vec![bloq_graph::FeedbackTarget {
                    pauli: PauliBasis::Z,
                    target: IVec3::X,
                    direction: None,
                }],
                condition: Some(bloq_graph::Expr::Var("mx".into())),
            },
        ]
    }

    /// The whole popup rests on writer, `Action` display, and parser agreeing:
    /// an action loaded for editing must render back to the statement it came
    /// from, and that statement must parse into the same action.
    #[test]
    fn every_action_form_round_trips_through_the_blog_grammar() {
        for (ordinal, action) in every_action_form().iter().enumerate() {
            let graph = two_block_graph();
            let draft = ActionDraft::from_action(ordinal, action, &graph);
            assert_eq!(draft.editing, Some((ordinal, graph.actions())));
            assert!(!draft.picking, "an edit starts with its operands supplied");

            let source = draft_blog_source(&draft).expect("a loaded action renders");
            assert_eq!(source, action.to_string());
            assert_eq!(
                parse_actions(&source, |_, _| None).expect("the draft is valid BLOG"),
                vec![action.clone()]
            );
        }
    }

    #[test]
    fn removing_an_earlier_action_cancels_the_open_edit() {
        let mut graph = two_block_graph();
        let feedback = |pauli| Action::Feedback {
            targets: vec![bloq_graph::FeedbackTarget {
                pauli,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: None,
        };
        let mut actions = vec![
            feedback(PauliBasis::X),
            feedback(PauliBasis::Z),
            feedback(PauliBasis::Y),
        ];
        graph.set_actions_lenient(actions.clone()).unwrap();
        let mut state = ActionEditState::default();
        state.edit(1, &actions[1], &graph);
        state.cancel_stale_edit(&graph);
        assert!(state.draft.is_some());

        actions.remove(0);
        graph.set_actions_lenient(actions.clone()).unwrap();
        state.cancel_stale_edit(&graph);

        assert!(state.draft.is_none());
        assert_eq!(graph.actions(), actions);
    }

    #[test]
    fn deleting_an_identical_action_cancels_its_open_edit() {
        let mut graph = two_block_graph();
        let action = Action::Feedback {
            targets: vec![bloq_graph::FeedbackTarget {
                pauli: PauliBasis::X,
                target: IVec3::ZERO,
                direction: None,
            }],
            condition: None,
        };
        graph
            .set_actions_lenient(vec![action.clone(), action.clone()])
            .unwrap();
        let mut state = ActionEditState::default();
        state.edit(0, &action, &graph);
        graph.set_actions_lenient(vec![action]).unwrap();
        state.cancel_stale_edit(&graph);
        assert!(state.draft.is_none());
    }

    /// Chips build the expression left to right, so the spacing rule is the only
    /// thing standing between `!m0` and the invalid `! m0 ^m1`.
    #[test]
    fn operator_chips_space_themselves_into_a_parseable_condition() {
        let mut condition = String::new();
        for token in ["!", "m0", "^", "(", "m1", "&", "m2", ")"] {
            append_token(&mut condition, token);
        }

        assert_eq!(condition, "!m0 ^ (m1 & m2)");
        parse_actions(&format!("resolve [0, 0, 0] if {condition}"), |_, _| None)
            .expect("a chip-built condition is valid BLOG");
    }

    /// The preview is the only thing gating the commit button, so it has to name
    /// every way a draft is not ready yet.
    #[test]
    fn the_preview_reports_the_statement_or_what_is_missing() {
        let mut measure = ActionDraft::new(ActionKind::Measure);
        measure.targets = vec![MeasureTarget::Node(IVec3::ZERO)];
        measure.name = "m0".into();
        assert_eq!(draft_preview(&measure), Ok("m0 = measure [0, 0, 0]".into()));
        measure.name.clear();
        assert_eq!(draft_preview(&measure), Err("Enter a variable name".into()));

        let mut resolve = ActionDraft::new(ActionKind::Resolve);
        assert_eq!(
            draft_preview(&resolve),
            Err("Pick a target in the scene".into())
        );
        resolve.targets = vec![MeasureTarget::Node(IVec3::ZERO)];
        resolve.expr = "m0 ^^".into();
        assert!(draft_preview(&resolve).is_err(), "a malformed expression");

        let mut binding = ActionDraft::new(ActionKind::Let);
        binding.name = "a".into();
        binding.expr = "m0\nb = m0".into();
        assert_eq!(
            parse_draft_action(&binding),
            Err("draft must contain one action".into())
        );
    }

    /// A pipe click names an edge, not the block it starts from.
    #[test]
    fn picking_a_pipe_measures_its_edge() {
        let graph = two_block_graph();
        let mut state = ActionEditState::default();
        state.arm(ActionKind::Measure);

        state
            .accept_pick(GraphElement::Pipe(IVec3::ZERO, IVec3::X), &graph)
            .expect("the clicked pipe exists");

        let draft = state.draft.expect("the draft survives the pick");
        assert_eq!(
            draft.targets,
            vec![MeasureTarget::Edge {
                src: IVec3::ZERO,
                dir: Direction::XPLUS,
            }]
        );
        assert_eq!(draft.name, "m0", "a measure pre-fills its outcome name");
        assert!(!draft.picking, "a one-target kind disarms once it is fed");
    }

    /// Feedback is the only kind that keeps taking clicks, and it must not take
    /// the same block twice.
    #[test]
    fn feedback_accumulates_targets_and_refuses_a_repeat() {
        let graph = two_block_graph();
        let mut state = ActionEditState::default();
        state.arm(ActionKind::Feedback);

        state
            .accept_pick(GraphElement::Block(IVec3::ZERO), &graph)
            .expect("the block exists");
        state
            .accept_pick(GraphElement::Block(IVec3::X), &graph)
            .expect("the block exists");
        let repeat = state.accept_pick(GraphElement::Block(IVec3::X), &graph);

        assert!(repeat.is_err());
        let draft = state.draft.expect("still drafting");
        assert_eq!(draft.targets.len(), 2);
        assert_eq!(draft.paulis, vec![PauliBasis::X, PauliBasis::X]);
        assert!(draft.picking, "feedback keeps taking clicks");
    }

    #[test]
    fn resolve_rejects_a_non_selective_target_and_keeps_the_pick_armed() {
        let graph = two_block_graph();
        let mut state = ActionEditState::default();
        state.arm(ActionKind::Resolve);

        let rejected = state.accept_pick(GraphElement::Block(IVec3::ZERO), &graph);

        assert!(rejected.is_err());
        assert_eq!(state.picking(), Some(ActionKind::Resolve));
    }

    #[test]
    fn resolve_targets_and_highlights_a_whole_branch_region() {
        let mut graph = BlockGraph::default();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let inside = ivec3(0, 0, 1);
        let arm = || {
            BranchArm::new(
                vec![Block::new(inside, BlockKind::Cube(CubeKind::ZXZ))],
                vec![Pipe::new(IVec3::ZERO, Direction::ZPLUS)],
            )
        };
        let target = graph
            .try_add_branch_region("b0", arm(), arm())
            .expect("matching terminal arms form a region");
        let mut state = ActionEditState::default();
        state.arm(ActionKind::Resolve);

        state
            .accept_pick(GraphElement::Pipe(IVec3::ZERO, inside), &graph)
            .expect("any displayed arm element names the region");
        let mut draft = state.draft.expect("the draft survives the pick");
        draft.expr = "m0".into();
        assert_eq!(draft.targets, vec![MeasureTarget::Node(target)]);
        assert_eq!(
            draft_blog_source(&draft).as_deref(),
            Some("resolve b0 if m0")
        );
        assert!(matches!(
            parse_draft_action(&draft),
            Ok(Action::Branch { target: parsed, .. }) if parsed == target
        ));

        let mut state = ActionEditState::default();
        state.arm(ActionKind::Resolve);
        assert_eq!(
            state.pick_candidates(&graph),
            HashSet::from([
                GraphElement::Block(inside),
                GraphElement::Pipe(IVec3::ZERO, inside).canonical(),
            ])
        );
    }

    #[test]
    fn candidates_are_the_blocks_the_armed_kind_can_actually_take() {
        let mut graph = two_block_graph();
        graph.add_block(Block::new(
            ivec3(4, 0, 0),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        let mut state = ActionEditState::default();

        state.arm(ActionKind::Resolve);
        assert_eq!(
            state.pick_candidates(&graph),
            HashSet::from([GraphElement::Block(ivec3(4, 0, 0))])
        );

        // Measure and feedback take most of the graph, so they highlight nothing.
        state.arm(ActionKind::Measure);
        assert!(state.pick_candidates(&graph).is_empty());
        state.arm(ActionKind::Feedback);
        assert!(state.pick_candidates(&graph).is_empty());
    }

    /// The text-only kinds must not arm a pick, or the next scene click would be
    /// swallowed with nothing to do.
    #[test]
    fn text_only_kinds_do_not_arm_a_pick() {
        let mut state = ActionEditState::default();
        for kind in [ActionKind::Let, ActionKind::DiscardIf] {
            state.arm(kind);
            assert_eq!(state.picking(), None, "{kind:?} has no scene operand");
        }
    }

    #[test]
    fn measure_names_skip_the_ones_already_bound() {
        let mut graph = two_block_graph();
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(IVec3::X),
                name: "m0".into(),
            }])
            .expect("a single node measurement is valid");

        assert_eq!(next_measure_name(&graph), "m1");
    }
}
