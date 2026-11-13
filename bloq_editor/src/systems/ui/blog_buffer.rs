//! Cross-highlighting between the `.blog` text buffer and the 3D scene: parses
//! the buffer once and maps each source line to the graph elements it names, so
//! the cursor can highlight matching blocks, pipes, and module instances.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::intents::{UiIntent, UiIntentBuffer};
use super::{activated, tiled_window_rect};
use crate::components::GraphElement;
use crate::resources::EditorState;
use crate::theme::ThemePalette;
use bevy::prelude::Resource;
use bevy_egui::egui;
use bloq_graph::ast::{
    ActionStmt, BranchArmStmt, ConnectEndpointAst, ConnectStmt, DataStmt, Expr as AstExpr,
    InterfaceStmt, MeasureTargetAst, ModularSourceFile, ModuleDef, PipeDef, PipeDst, Ref,
    ResolveTargetDef, SourceFile, Span, Spanned,
};
use bloq_graph::{
    Block, BlockGraph, BlockKind, BranchArm, BranchRegion, Direction, Pipe, PortRole,
    WalkingBoundaryKind, checked_add_position, flatten_module_definition, lower_blog_ast_deferred,
    lower_blog_graph_ast_deferred, parse_blog_program_to_ast, parse_blog_to_ast, qualified_name,
};
use egui_extras::syntax_highlighting::{CodeTheme, SyntectSettings, highlight_with};
use glam::IVec3;
use syntect::parsing::SyntaxDefinition;

const BLOG_SYNTAX: &str = r#"%YAML 1.2
---
name: BLOG
file_extensions: [blog]
scope: source.blog
contexts:
  main:
    - match: '#.*$'
      scope: comment.line.number-sign.blog
    - match: '(?i)(?<![A-Za-z0-9_/.+-])(BLOG|branch|false|true|measure|resolve|feedback|if)(?![A-Za-z0-9_/.+-])'
      scope: keyword.control.blog
    - match: '(?i)^([ 	]*)(module)(?=[ 	]+[A-Za-z_][A-Za-z0-9_+-]*[ 	]*\{)'
      captures:
        2: keyword.control.blog
    - match: '(?i)^([ 	]*)(import)(?=[ 	]+")'
      captures:
        2: keyword.control.blog
    - match: '(?i)^([ 	]*)(in|out)(?=[ 	]+[A-Za-z_])'
      captures:
        2: keyword.control.blog
    - match: '(?i)^([ 	]*)(discard)(?=[ 	]+if(?![A-Za-z0-9_/.+-]))'
      captures:
        2: keyword.control.blog
    - match: '(?i)^([ 	]*\d+[ 	]*:[ 	]*)(walk|rotate)(?=[ 	]+)'
      captures:
        2: keyword.control.blog
    - match: '(?i)^([ 	]*\d+[ 	]*:[ 	]*)(Port|T|Y|X|Z|XY|YX|XZ|ZX|YZ|ZY|XZZ|ZXZ|ZZX|ZXX|XZX|XXZ)(?![A-Za-z0-9_/.+-])'
      captures:
        2: storage.type.blog
      push: block-tail
    - match: '"[^"\r\n]*"'
      scope: string.quoted.double.blog
    - match: '<[^>]+>'
      scope: string.unquoted.blog
    - match: '(?<![A-Za-z0-9_/.+\-])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_/.+\-])'
      scope: constant.numeric.blog
    - match: '(?i)-H>|->|=>|@|(?<![A-Za-z0-9_/.+-])[+\-][XYZ](?![A-Za-z0-9_/.+-])|[!&|^=]'
      scope: keyword.operator.blog
  block-tail:
    - match: '$'
      pop: true
    - match: '(?i)(role)([ 	]*=[ 	]*)(auto|input|output|multiplex)(?![A-Za-z0-9_/.+-])'
      captures:
        1: variable.parameter.blog
        3: constant.language.blog
    - match: '(?i)(height)([ 	]*=[ 	]*)(\d+)?(d)(?:[ 	]*/[ 	]*(\d+))?(?:[ 	]*([+\-][ 	]*\d+))?'
      captures:
        1: variable.parameter.blog
        3: constant.numeric.blog
        4: constant.language.blog
        5: constant.numeric.blog
        6: constant.numeric.blog
    - include: main
"#;

/// Per-tab BLOG text and export file name.
#[derive(Resource, Clone)]
pub(crate) struct ImportExportState {
    pub(crate) bloq_buffer: String,
    pub(crate) bloq_revision: u64,
    pub(crate) export_path: String,
    pub(crate) module_ui: super::module_panel::ModuleUiState,
    pub(crate) definition_edit: Option<super::module_panel::DefinitionEdit>,
    language: BlogLanguageState,
    synced_text: Option<String>,
    text_history: Vec<String>,
    text_history_revision: u64,
    synced_graph_revision: Option<u64>,
    attempted_text_revision: Option<u64>,
    last_text_edit: web_time::Instant,
    requested_apply_revision: Option<u64>,
    pub(crate) sync_error: Option<String>,
    pub(crate) sync_conflict: bool,
    confirm_replace_text: bool,
}

impl Default for ImportExportState {
    fn default() -> Self {
        Self {
            bloq_buffer: String::new(),
            bloq_revision: next_blog_revision(),
            export_path: "bloq_graph.blog".to_string(),
            module_ui: super::module_panel::ModuleUiState::default(),
            definition_edit: None,
            language: BlogLanguageState::default(),
            synced_text: None,
            text_history: Vec::new(),
            text_history_revision: next_blog_revision(),
            synced_graph_revision: None,
            attempted_text_revision: None,
            last_text_edit: web_time::Instant::now(),
            requested_apply_revision: None,
            sync_error: None,
            sync_conflict: false,
            confirm_replace_text: false,
        }
    }
}

impl ImportExportState {
    pub(crate) fn text_history_revision(&self) -> u64 {
        self.text_history_revision
    }

    pub(crate) fn text_history(&self) -> &[String] {
        &self.text_history
    }

    pub(crate) fn restore_text_history(&mut self, history: Vec<String>) {
        self.text_history = history;
        self.text_history_revision = next_blog_revision();
    }

    pub(crate) fn set_bloq_buffer(&mut self, buffer: String) {
        self.bloq_buffer = buffer;
        self.touch_bloq_buffer();
    }

    pub(crate) fn synced_text_is_uninitialized(&self) -> bool {
        self.synced_text.is_none()
    }

    /// Only a baseline attached to this exact graph revision is safe for
    /// background semantic no-op comparison. Conflicts intentionally opt out.
    pub(crate) fn comparison_source(&self, revision: u64) -> Option<&str> {
        (self.synced_graph_revision == Some(revision) && !self.sync_conflict)
            .then_some(self.synced_text.as_deref())
            .flatten()
    }

    pub(crate) fn blocks_graph_editing(&self) -> bool {
        self.sync_conflict
            || self
                .synced_text
                .as_ref()
                .is_some_and(|text| text != &self.bloq_buffer)
    }

    /// Refresh clean text after graph edits; never overwrite a pending text edit.
    pub(crate) fn refresh_from_graph(&mut self, graph: &crate::resources::GraphState) {
        if self.synced_graph_revision == Some(graph.revision) {
            return;
        }
        if self.blocks_graph_editing() {
            self.sync_conflict = true;
            return;
        }
        // Unassociated nonempty text may be a restored unresolved conflict.
        if self.synced_text.is_none() && !self.bloq_buffer.is_empty() {
            self.restore_graph_sync(graph);
            return;
        }
        if let Ok(program) = graph.resolved_graph() {
            self.replace_text_preserving_history(program.to_blog_text());
            self.mark_synchronized(graph.revision);
        }
    }

    /// Restored text and graph are independent saved values. Never decide which
    /// one wins unless they represent the same graph; retain original comments.
    pub(crate) fn restore_graph_sync(&mut self, graph: &crate::resources::GraphState) {
        let graph_text = graph
            .resolved_graph()
            .ok()
            .map(|program| program.to_blog_text());
        let same_graph = BlockGraph::from_text(&self.bloq_buffer)
            .ok()
            .is_some_and(|program| Some(program.to_blog_text()) == graph_text);
        if same_graph {
            self.mark_synchronized(graph.revision);
        } else {
            self.synced_text = graph_text;
            self.synced_graph_revision = Some(graph.revision);
            self.sync_conflict = true;
        }
    }

    /// Transfer the worker's recovery decision without reparsing the saved draft.
    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn recovery_sync(&self) -> (Option<String>, bool) {
        (self.synced_text.clone(), self.sync_conflict)
    }

    pub(crate) fn restore_recovery_sync(
        &mut self,
        graph_revision: u64,
        synced_text: Option<String>,
        conflict: bool,
    ) {
        self.synced_text = synced_text;
        self.synced_graph_revision = Some(graph_revision);
        self.sync_conflict = conflict;
    }

    /// Graph refresh and history recovery both retain the text they replace.
    pub(crate) fn replace_text_preserving_history(&mut self, next: String) {
        if self.bloq_buffer == next {
            return;
        }
        if !self.bloq_buffer.is_empty() && self.text_history.last() != Some(&self.bloq_buffer) {
            self.text_history.push(self.bloq_buffer.clone());
            self.text_history_revision = next_blog_revision();
        }
        self.set_bloq_buffer(next);
    }

    pub(crate) fn mark_synchronized(&mut self, revision: u64) {
        self.synced_text = Some(self.bloq_buffer.clone());
        self.synced_graph_revision = Some(revision);
        self.sync_error = None;
        self.sync_conflict = false;
    }

    pub(crate) fn request_text_apply(&mut self) {
        self.requested_apply_revision = Some(self.bloq_revision);
    }

    /// Do not parse while typing. The graph stays unchanged until a current
    /// background result arrives; language analysis follows the same debounce.
    pub(crate) fn take_analysis_request(&mut self, revision: u64, force: bool) -> Option<bool> {
        let force = force || self.requested_apply_revision == Some(self.bloq_revision);
        if !force
            && (self.last_text_edit.elapsed() < std::time::Duration::from_millis(250)
                || self.attempted_text_revision == Some(self.bloq_revision))
        {
            return None;
        }
        if self.synced_graph_revision != Some(revision) && self.blocks_graph_editing() {
            self.sync_conflict = true;
        }
        self.attempted_text_revision = Some(self.bloq_revision);
        self.requested_apply_revision = None;
        Some(self.blocks_graph_editing() && (force || !self.sync_conflict))
    }

    pub(crate) fn install_language(&mut self, language: BlogLanguageState) {
        self.language = language;
    }

    fn touch_bloq_buffer(&mut self) {
        self.confirm_replace_text = false;
        self.sync_error = None;
        self.last_text_edit = web_time::Instant::now();
        self.requested_apply_revision = None;
        self.language.index = BlogElementIndex::default();
        self.language.diagnostic = None;
        self.language.completion = None;
        self.bloq_revision = next_blog_revision();
    }
}

mod optional_span {
    use super::Span;
    use serde::{Deserialize, Serialize};
    pub(super) fn serialize<S: serde::Serializer>(
        span: &Option<Span>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        span.map(|span| (span.start, span.end))
            .serialize(serializer)
    }
    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Span>, D::Error> {
        Ok(
            Option::<(u32, u32)>::deserialize(deserializer)?
                .map(|(start, end)| Span { start, end }),
        )
    }
}

fn next_blog_revision() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Parsed BLOG buffer resolved into the graph elements each statement refers
/// to, grouped by the byte offset of the line the statement starts on.
///
/// Parsing is the expensive part, so callers build this once per buffer change
/// and then resolve a hovered cursor with a cheap [`Self::elements_at_line`]
/// lookup instead of reparsing every frame.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
struct BlogElementIndex {
    by_line_start: HashMap<usize, HashSet<GraphElement>>,
    line_starts: Vec<(usize, usize)>,
    char_len: usize,
}

impl BlogElementIndex {
    fn from_source(buffer: &str, source: &ModularSourceFile, program: Option<&BlockGraph>) -> Self {
        let mut by_line_start = HashMap::<usize, HashSet<GraphElement>>::new();
        let direct_pipes = program
            .map(BlockGraph::direct_pipe_endpoints_by_module)
            .unwrap_or_default();
        for module in &source.modules {
            let fallback = if program.is_none() {
                lower_body(&SourceFile {
                    version: source.version.clone(),
                    data_stmts: module.node.data_stmts.clone(),
                    action_stmts: module.node.action_stmts.clone(),
                })
            } else {
                None
            };
            let Some(graph) = program
                .and_then(|program| program.module(&module.node.name.node))
                .map(BlockGraph::local_body)
                .or(fallback.as_ref())
            else {
                continue;
            };
            index_module(
                buffer,
                &module.node,
                graph,
                direct_pipes
                    .get(&module.node.name.node)
                    .map(Vec::as_slice)
                    .unwrap_or_default(),
                &mut by_line_start,
            );
        }
        if let Some(program) = program {
            index_module_instances(buffer, source, program, &mut by_line_start);
        }
        let (line_starts, char_len) = cursor_line_starts(buffer);
        Self {
            by_line_start,
            line_starts,
            char_len,
        }
    }

    fn elements_at_line(&self, line_start: usize) -> Option<&HashSet<GraphElement>> {
        self.by_line_start.get(&line_start)
    }

    fn line_start_at_char(&self, cursor: usize) -> Option<usize> {
        if cursor > self.char_len {
            return None;
        }
        let index = self
            .line_starts
            .partition_point(|&(character, _)| character <= cursor)
            .checked_sub(1)?;
        Some(self.line_starts[index].1)
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct BlogLanguageState {
    revision: Option<u64>,
    index: BlogElementIndex,
    diagnostic: Option<BlogDiagnostic>,
    words: CompletionWords,
    cursor_char: Option<usize>,
    completion: Option<CompletionSession>,
}

impl BlogLanguageState {
    pub(crate) fn refresh(&mut self, buffer: &str, revision: u64) -> bool {
        if self.revision == Some(revision) {
            return false;
        }
        self.revision = Some(revision);
        self.completion = None;
        match parse_blog_program_to_ast(buffer) {
            Ok(source) if source.modules.is_empty() && source.imports.is_empty() => {
                self.refresh_body(
                    buffer,
                    SourceFile {
                        version: source.version,
                        data_stmts: Vec::new(),
                        action_stmts: Vec::new(),
                    },
                );
            }
            Ok(source) => {
                self.words = completion_words(&source);
                match lower_blog_graph_ast_deferred(&source) {
                    Ok(program) => {
                        self.index = BlogElementIndex::from_source(buffer, &source, Some(&program));
                        self.diagnostic = None;
                    }
                    Err(error) => {
                        self.index = BlogElementIndex::from_source(buffer, &source, None);
                        self.diagnostic = Some(BlogDiagnostic {
                            message: error.to_string(),
                            span: error.span(),
                        });
                    }
                }
            }
            Err(program_error) => match parse_blog_to_ast(buffer) {
                Ok(body) => self.refresh_body(buffer, body),
                Err(body_error) => {
                    // Match the graph parser: report the grammar that got furthest.
                    let error = if program_error.span().map(|span| span.start)
                        >= body_error.span().map(|span| span.start)
                    {
                        program_error
                    } else {
                        body_error
                    };
                    self.index = BlogElementIndex::default();
                    self.diagnostic = Some(BlogDiagnostic {
                        message: error.to_string(),
                        span: error.span(),
                    });
                }
            },
        }
        true
    }

    fn refresh_body(&mut self, buffer: &str, body: SourceFile) {
        let graph = lower_blog_ast_deferred(&body, std::iter::empty());
        // A virtual main module reuses indexing/completion without inserting a
        // textual wrapper, which would shift every diagnostic and hover span.
        let span = Span::from_range(0..buffer.len());
        let source = ModularSourceFile {
            version: body.version,
            imports: Vec::new(),
            modules: vec![Spanned::new(
                ModuleDef {
                    name: Spanned::new("main".into(), span),
                    interface_stmts: Vec::new(),
                    instance_stmts: Vec::new(),
                    data_stmts: body.data_stmts,
                    action_stmts: body.action_stmts,
                    connect_stmts: Vec::new(),
                },
                span,
            )],
        };
        self.words = completion_words(&source);
        self.index = BlogElementIndex::from_source(buffer, &source, graph.as_ref().ok());
        self.diagnostic = graph.err().map(|error| BlogDiagnostic {
            message: error.to_string(),
            span: error.span(),
        });
    }

    fn update_cursor(&mut self, cursor_char: Option<usize>) -> bool {
        if self.cursor_char == cursor_char {
            return false;
        }
        self.cursor_char = cursor_char;
        self.completion = None;
        true
    }

    fn trigger_completion(&mut self, buffer: &str, step: isize) {
        if let Some(completion) = self.completion.as_mut() {
            completion.cycle(step);
            return;
        }
        self.completion = self
            .cursor_char
            .and_then(|cursor| completion_at(buffer, cursor, &self.words));
        if step < 0
            && let Some(completion) = self.completion.as_mut()
        {
            completion.cycle(step);
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct BlogDiagnostic {
    message: String,
    #[serde(with = "optional_span")]
    span: Option<Span>,
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
struct CompletionWords {
    modules: Vec<String>,
    branches: Vec<String>,
    expressions: Vec<String>,
    references: Vec<String>,
}

fn completion_words(source: &ModularSourceFile) -> CompletionWords {
    let interfaces = source
        .modules
        .iter()
        .map(|module| {
            (
                module.node.name.node.as_str(),
                module
                    .node
                    .interface_stmts
                    .iter()
                    .map(|statement| interface_name(&statement.node))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut words = CompletionWords::default();
    for module in &source.modules {
        words.modules.push(module.node.name.node.clone());
        for statement in &module.node.data_stmts {
            match &statement.node {
                DataStmt::Branch(branch) => words.branches.push(branch.name.node.clone()),
                DataStmt::Block(_) | DataStmt::Pipe(_) => {}
            }
        }
        for statement in &module.node.action_stmts {
            match &statement.node {
                ActionStmt::Measure(definition) => {
                    words.expressions.push(definition.name.node.clone());
                }
                ActionStmt::Let(definition) => {
                    words.expressions.push(definition.name.node.clone());
                }
                ActionStmt::DiscardIf(_) | ActionStmt::Resolve(_) | ActionStmt::Feedback(_) => {}
            }
        }
        for statement in &module.node.interface_stmts {
            let name = interface_name(&statement.node).to_owned();
            words.references.push(name.clone());
            if matches!(&statement.node, InterfaceStmt::BitInput(_)) {
                words.expressions.push(name);
            }
        }
        for instance in &module.node.instance_stmts {
            if let Some(members) = interfaces.get(instance.node.definition.node.as_str()) {
                for member in members {
                    words
                        .references
                        .push(format!("{}.{member}", instance.node.name.node));
                }
            }
        }
    }
    for group in [
        &mut words.modules,
        &mut words.branches,
        &mut words.expressions,
        &mut words.references,
    ] {
        group.sort_unstable();
        group.dedup();
    }
    words
}

fn interface_name(statement: &InterfaceStmt) -> &str {
    match statement {
        InterfaceStmt::Quantum(port) => &port.name.node,
        InterfaceStmt::BitInput(name) => &name.node,
        InterfaceStmt::BitOutput(output) => &output.name.node,
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CompletionSession {
    replace: Range<usize>,
    items: Vec<String>,
    selected: usize,
}

impl CompletionSession {
    fn cycle(&mut self, step: isize) {
        self.selected =
            (self.selected as isize + step).rem_euclid(self.items.len() as isize) as usize;
    }
}

#[derive(Clone, Copy)]
enum CompletionKind {
    Any,
    Expression,
    Assignment,
    Module,
    Branch,
    Reference,
    Block,
    Walk,
    Basis,
    Axis,
    Modifier,
    Role,
    Height,
    Pauli,
    Direction,
}

fn completion_at(
    buffer: &str,
    cursor_char: usize,
    words: &CompletionWords,
) -> Option<CompletionSession> {
    let cursor = byte_index_at_char(buffer, cursor_char)?;
    let line_start = line_start_of_byte(buffer, cursor);
    let line_to_cursor = &buffer[line_start..cursor];
    if scan_line(line_to_cursor).mode != LineMode::Code {
        return None;
    }
    let replace = completion_range(buffer, cursor);
    let prefix = &buffer[replace.start..cursor];
    let before = &buffer[line_start..replace.start];
    // A dotted prefix can only match a dotted reference, so instance members
    // fall out of the reference list without a category of their own.
    let kind = if prefix.contains('.') {
        CompletionKind::Reference
    } else {
        completion_kind(before)
    };
    if prefix.is_empty() && matches!(kind, CompletionKind::Any) {
        return None;
    }

    let current = &buffer[replace.clone()];
    let mut items = Vec::new();
    collect_completions(kind, prefix, current, words, &mut items);
    (!items.is_empty()).then_some(CompletionSession {
        replace,
        items,
        selected: 0,
    })
}

fn completion_kind(before: &str) -> CompletionKind {
    let trimmed = before.trim_end();
    if ends_with_ascii_case(trimmed, "role=") {
        return CompletionKind::Role;
    }
    if ends_with_ascii_case(trimmed, "height=") {
        return CompletionKind::Height;
    }
    if last_word_is(trimmed, "walk") {
        return CompletionKind::Walk;
    }
    if last_word_is(trimmed, "rotate") {
        return if trimmed.contains('@') {
            CompletionKind::Axis
        } else {
            CompletionKind::Basis
        };
    }
    if let Some(left) = trimmed.strip_suffix(':') {
        return if is_block_statement(trimmed) {
            CompletionKind::Block
        } else if left.split_ascii_whitespace().count() == 1 {
            CompletionKind::Module
        } else {
            CompletionKind::Any
        };
    }
    if ["-H>", "=>"].iter().any(|arrow| trimmed.ends_with(arrow)) {
        return CompletionKind::Reference;
    }
    if trimmed.ends_with("->") {
        return CompletionKind::Direction;
    }
    if last_word_is(trimmed, "feedback")
        || (trimmed.ends_with(',') && first_word_is(trimmed, "feedback"))
    {
        return CompletionKind::Pauli;
    }
    if last_word_is(trimmed, "resolve") {
        return CompletionKind::Branch;
    }
    if last_word_is(trimmed, "if") {
        return CompletionKind::Expression;
    }
    if trimmed.ends_with('=') {
        return if first_word_is(trimmed, "out") {
            CompletionKind::Expression
        } else {
            CompletionKind::Assignment
        };
    }
    if trimmed.ends_with(']') && is_block_statement(trimmed) {
        return CompletionKind::Modifier;
    }
    CompletionKind::Any
}

fn collect_completions(
    kind: CompletionKind,
    prefix: &str,
    current: &str,
    words: &CompletionWords,
    items: &mut Vec<String>,
) {
    // Kinds backed by a Rust enum render their variants first so the popup
    // still leads with them; everything else reduces to a fixed keyword list
    // plus a list harvested from the buffer.
    match kind {
        CompletionKind::Block => {
            for kind in BlockKind::all_kinds() {
                let text = match kind {
                    BlockKind::Walking(_) => "walk ".to_owned(),
                    BlockKind::PatchRotation(_) => "rotate ".to_owned(),
                    _ => kind.to_string(),
                };
                add_completion(items, prefix, current, text);
            }
        }
        CompletionKind::Walk => {
            for kind in WalkingBoundaryKind::ALL {
                add_completion(items, prefix, current, kind.to_string());
            }
        }
        CompletionKind::Direction => {
            for direction in Direction::iter() {
                add_completion(items, prefix, current, direction.to_string());
            }
        }
        CompletionKind::Role => {
            for role in [
                PortRole::Auto,
                PortRole::Input,
                PortRole::Output,
                PortRole::Multiplex,
            ] {
                add_completion(items, prefix, current, role.as_str());
            }
        }
        _ => {}
    }

    let (keywords, harvested): (&[&str], &[String]) = match kind {
        CompletionKind::Any => (
            &[
                "BLOG 1.0",
                "import ",
                "module ",
                "in ",
                "out ",
                "branch ",
                "measure ",
                "resolve ",
                "discard if ",
                "feedback ",
                "if ",
                "false ",
                "true ",
            ],
            &[],
        ),
        CompletionKind::Expression => (&[], &words.expressions),
        CompletionKind::Assignment => (&["measure "], &words.expressions),
        CompletionKind::Module => (&[], &words.modules),
        CompletionKind::Branch => (&[], &words.branches),
        CompletionKind::Reference | CompletionKind::Direction => (&[], &words.references),
        CompletionKind::Basis => (&["X", "Z"], &[]),
        CompletionKind::Axis | CompletionKind::Pauli => (&["X", "Y", "Z"], &[]),
        CompletionKind::Modifier => (&["height=", "color=", "role="], &[]),
        CompletionKind::Height => (&["d", "2d", "d/2", "3d/2"], &[]),
        CompletionKind::Block | CompletionKind::Walk | CompletionKind::Role => (&[], &[]),
    };
    for keyword in keywords {
        add_completion(items, prefix, current, *keyword);
    }
    // Harvested lists are already unique. Only compare against fixed choices,
    // not all earlier matches: opening completion must stay linear for large modules.
    let fixed_choices = items.len();
    for word in harvested {
        if starts_with_ascii_case(word.trim_end(), prefix)
            && word != current
            && !items[..fixed_choices].contains(word)
        {
            items.push(word.clone());
        }
    }
}

fn add_completion(items: &mut Vec<String>, prefix: &str, current: &str, text: impl AsRef<str>) {
    let text = text.as_ref();
    if starts_with_ascii_case(text.trim_end(), prefix)
        && text != current
        && !items.iter().any(|item| item == text)
    {
        items.push(text.to_owned());
    }
}

fn byte_index_at_char(text: &str, character: usize) -> Option<usize> {
    text.char_indices()
        .nth(character)
        .map(|(byte, _)| byte)
        .or_else(|| (character == text.chars().count()).then_some(text.len()))
}

fn completion_range(buffer: &str, cursor: usize) -> Range<usize> {
    let start = buffer[..cursor].trim_end_matches(is_completion_char).len();
    let end = buffer[cursor..]
        .find(|character: char| !is_completion_char(character))
        .map_or(buffer.len(), |offset| cursor + offset);
    start..end
}

fn is_completion_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '/' | '.' | '+' | '-')
}

fn starts_with_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

fn ends_with_ascii_case(value: &str, suffix: &str) -> bool {
    value
        .get(value.len().saturating_sub(suffix.len())..)
        .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
}

fn last_word(value: &str) -> &str {
    value
        .trim_end()
        .rsplit(|character: char| !is_completion_char(character))
        .next()
        .unwrap_or_default()
}

fn last_word_is(value: &str, expected: &str) -> bool {
    last_word(value).eq_ignore_ascii_case(expected)
}

fn first_word_is(value: &str, expected: &str) -> bool {
    value
        .split_ascii_whitespace()
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case(expected))
}

fn is_block_statement(value: &str) -> bool {
    value
        .trim_start()
        .split_once(':')
        .is_some_and(|(id, _)| id.trim().parse::<u64>().is_ok())
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum LineMode {
    #[default]
    Code,
    Quoted,
    Tag,
    Comment,
}

#[derive(Default)]
struct LineScan {
    opens: usize,
    closes: usize,
    mode: LineMode,
}

fn scan_line(line: &str) -> LineScan {
    let mut scan = LineScan::default();
    for character in line.chars() {
        match scan.mode {
            LineMode::Quoted => {
                if character == '"' {
                    scan.mode = LineMode::Code;
                }
            }
            LineMode::Tag => {
                if character == '>' {
                    scan.mode = LineMode::Code;
                }
            }
            LineMode::Comment => break,
            LineMode::Code => match character {
                '"' => scan.mode = LineMode::Quoted,
                '<' => scan.mode = LineMode::Tag,
                '#' => scan.mode = LineMode::Comment,
                '{' => scan.opens += 1,
                '}' => scan.closes += 1,
                _ => {}
            },
        }
    }
    scan
}

/// Reindents a parseable buffer; `None` when it does not parse.
fn format_blog(source: &str) -> Option<String> {
    if parse_blog_program_to_ast(source).is_err() && parse_blog_to_ast(source).is_err() {
        return None;
    }
    let mut output = String::with_capacity(source.len() + 1);
    let mut depth = 0usize;
    for line in source.lines() {
        let scan = scan_line(line);
        depth = depth.saturating_sub(scan.closes);
        let line = line.trim();
        if !line.is_empty() {
            for _ in 0..depth {
                output.push_str("  ");
            }
            output.push_str(line);
        }
        output.push('\n');
        depth += scan.opens;
    }
    while output.ends_with("\n\n") {
        output.pop();
    }
    Some(output)
}

fn apply_completion(buffer: &mut String, completion: &CompletionSession) -> usize {
    let before = buffer[..completion.replace.start].chars().count();
    let insert = &completion.items[completion.selected];
    buffer.replace_range(completion.replace.clone(), insert);
    before + insert.chars().count()
}

fn insert_indented_newline(buffer: &mut String, selected: Range<usize>) -> Option<usize> {
    let start = byte_index_at_char(buffer, selected.start)?;
    let mut end = byte_index_at_char(buffer, selected.end)?;
    let line_start = line_start_of_byte(buffer, start);
    let before = &buffer[line_start..start];
    let indent = &before[..before.len() - before.trim_start_matches([' ', '\t']).len()];
    let nested = {
        let scan = scan_line(before);
        scan.opens > scan.closes
    };
    // Measure the run of blanks after the caret once, then reuse it to skip
    // them when the closing brace is pulled onto its own line.
    let after = buffer[end..].trim_start_matches([' ', '\t']);
    let closing = nested && after.starts_with('}');
    if closing {
        end = buffer.len() - after.len();
    }

    let mut insertion = format!("\n{indent}");
    if nested {
        insertion.push_str("  ");
    }
    let cursor = selected.start + insertion.chars().count();
    if closing {
        insertion.push('\n');
        insertion.push_str(indent);
    }
    buffer.replace_range(start..end, &insertion);
    Some(cursor)
}

fn set_text_cursor(ctx: &egui::Context, id: egui::Id, character: usize) {
    let Some(mut state) = egui::TextEdit::load_state(ctx, id) else {
        return;
    };
    let cursor = egui::text::CCursor::new(character);
    state
        .cursor
        .set_char_range(Some(egui::text::CCursorRange::one(cursor)));
    egui::TextEdit::store_state(ctx, id, state);
}

fn lower_body(source: &SourceFile) -> Option<BlockGraph> {
    lower_blog_ast_deferred(source, implicit_inputs(&source.action_stmts)).ok()
}

fn implicit_inputs(actions: &[Spanned<ActionStmt>]) -> HashSet<String> {
    let produced = actions
        .iter()
        .filter_map(|action| match &action.node {
            ActionStmt::Measure(definition) => Some(definition.name.node.clone()),
            ActionStmt::Let(definition) => Some(definition.name.node.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut inputs = HashSet::new();
    for action in actions {
        let expression = match &action.node {
            ActionStmt::Let(definition) => Some(&definition.expr.node),
            ActionStmt::DiscardIf(expression) => Some(&expression.node),
            ActionStmt::Resolve(definition) => Some(&definition.condition.node),
            ActionStmt::Feedback(definition) => definition
                .condition
                .as_ref()
                .map(|expression| &expression.node),
            ActionStmt::Measure(_) => None,
        };
        if let Some(expression) = expression {
            collect_expr_names(expression, &mut inputs);
        }
    }
    inputs.retain(|name| !produced.contains(name));
    inputs
}

fn collect_expr_names(expression: &AstExpr, names: &mut HashSet<String>) {
    match expression {
        AstExpr::Var(name) => {
            names.insert(name.clone());
        }
        AstExpr::Not(inner) => collect_expr_names(&inner.node, names),
        AstExpr::Binary(_, left, right) => {
            collect_expr_names(&left.node, names);
            collect_expr_names(&right.node, names);
        }
    }
}

fn cursor_line_starts(buffer: &str) -> (Vec<(usize, usize)>, usize) {
    let mut starts = vec![(0, 0)];
    let mut character = 0;
    for (byte, value) in buffer.char_indices() {
        character += 1;
        if value == '\n' {
            starts.push((character, byte + value.len_utf8()));
        }
    }
    (starts, character)
}

fn line_start_of_byte(buffer: &str, byte: usize) -> usize {
    let byte = byte.min(buffer.len());
    buffer[..byte].rfind('\n').map_or(0, |index| index + 1)
}

fn index_module_instances(
    buffer: &str,
    source: &ModularSourceFile,
    program: &BlockGraph,
    by_line_start: &mut HashMap<usize, HashSet<GraphElement>>,
) {
    let Ok(flattened) = flatten_module_definition(program, program.root(), "") else {
        return;
    };
    let source_modules = source
        .modules
        .iter()
        .map(|module| (module.node.name.node.as_str(), &module.node))
        .collect::<HashMap<_, _>>();
    let mut stack = vec![(program.root(), String::new())];
    while let Some((module, parent_path)) = stack.pop() {
        let source_module = source_modules[&module.name.as_str()];
        for (instance, statement) in module.instances.iter().zip(&source_module.instance_stmts) {
            let path = qualified_name(&parent_path, &instance.name);
            let owns = |candidate: &str| {
                candidate
                    .strip_prefix(&path)
                    .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with("__"))
            };
            let owns_block = |position| {
                flattened
                    .sites
                    .get(&position)
                    .is_some_and(|site| owns(&site.instance_path))
            };
            let elements = by_line_start
                .entry(line_start_of_byte(buffer, statement.span.start as usize))
                .or_default();
            elements.extend(GraphElement::all_in(&flattened.graph).filter(
                |element| match *element {
                    GraphElement::Block(position) => owns_block(position),
                    GraphElement::Pipe(source, target) => {
                        [source, target].into_iter().all(|endpoint| {
                            flattened
                                .graph
                                .get_endpoint_block(endpoint)
                                .is_some_and(|block| owns_block(block.pos()))
                        })
                    }
                },
            ));
            elements.extend(
                flattened
                    .graph
                    .branch_definitions()
                    .iter()
                    .filter(|region| owns_block(region.target))
                    .flat_map(branch_elements),
            );
            stack.push((
                program
                    .module(&instance.definition)
                    .expect("validated instance names a module"),
                path,
            ));
        }
    }
}

fn index_module(
    buffer: &str,
    module: &ModuleDef,
    graph: &BlockGraph,
    direct_pipes: &[(IVec3, IVec3)],
    by_line_start: &mut HashMap<usize, HashSet<GraphElement>>,
) {
    index_body(
        buffer,
        &module.data_stmts,
        &module.action_stmts,
        graph,
        by_line_start,
    );
    let ids = common_block_id_map(&module.data_stmts, graph);
    for stmt in &module.interface_stmts {
        if let InterfaceStmt::Quantum(port) = &stmt.node
            && let Some(block) = ids.get(&port.block_id.node)
        {
            by_line_start
                .entry(line_start_of_byte(buffer, stmt.span.start as usize))
                .or_default()
                .insert(GraphElement::Block(block.pos()));
        }
    }
    let mut direct_pipes = direct_pipes.iter().copied();
    for stmt in &module.connect_stmts {
        let elements = match &stmt.node {
            ConnectStmt::Bind { source, target, .. } => [source, target]
                .into_iter()
                .filter_map(|endpoint| match &endpoint.node {
                    ConnectEndpointAst::Block(id) => ids.get(id).map(Block::pos),
                    ConnectEndpointAst::Name(_) => None,
                })
                .map(GraphElement::Block)
                .collect(),
            ConnectStmt::Pipe { .. } => direct_pipes
                .next()
                .map(|(src, dst)| HashSet::from([GraphElement::Pipe(src, dst).canonical()]))
                .unwrap_or_default(),
        };
        by_line_start
            .entry(line_start_of_byte(buffer, stmt.span.start as usize))
            .or_default()
            .extend(elements);
    }
}

fn index_body(
    buffer: &str,
    data_stmts: &[Spanned<DataStmt>],
    action_stmts: &[Spanned<ActionStmt>],
    graph: &BlockGraph,
    by_line_start: &mut HashMap<usize, HashSet<GraphElement>>,
) {
    let common = common_block_id_map(data_stmts, graph);
    let regions = graph
        .branch_regions()
        .unwrap_or_default()
        .into_iter()
        .map(|region| (region.name.clone(), region))
        .collect::<HashMap<_, _>>();
    let branch_map = regions
        .iter()
        .map(|(name, region)| (name.clone(), branch_elements(region)))
        .collect::<HashMap<_, _>>();
    let common_scope = BlockScope::common(&common);
    for stmt in data_stmts {
        let elements = match &stmt.node {
            DataStmt::Block(block) => HashSet::from([GraphElement::Block(block.pos.node)]),
            DataStmt::Pipe(pipe) => pipe_element(pipe, &common_scope, graph.pipes())
                .into_iter()
                .collect(),
            DataStmt::Branch(branch) => branch_map
                .get(&branch.name.node)
                .cloned()
                .unwrap_or_default(),
        };
        by_line_start
            .entry(line_start_of_byte(buffer, stmt.span.start as usize))
            .or_default()
            .extend(elements);
        if let DataStmt::Branch(branch) = &stmt.node {
            let Some(region) = regions.get(&branch.name.node) else {
                continue;
            };
            for (arm, resolved) in [
                (&branch.on_false, region.on_false()),
                (&branch.on_true, region.on_true()),
            ] {
                let local = arm_block_id_map(arm, resolved);
                let scope = BlockScope {
                    common: &common,
                    local: Some(&local),
                };
                for arm_stmt in arm {
                    let element = match &arm_stmt.node {
                        BranchArmStmt::Block(block) => Some(GraphElement::Block(block.pos.node)),
                        BranchArmStmt::Pipe(pipe) => pipe_element(pipe, &scope, resolved.pipes()),
                    };
                    by_line_start
                        .entry(line_start_of_byte(buffer, arm_stmt.span.start as usize))
                        .or_default()
                        .extend(element);
                }
            }
        }
    }
    for stmt in action_stmts {
        by_line_start
            .entry(line_start_of_byte(buffer, stmt.span.start as usize))
            .or_default()
            .extend(action_stmt_elements(
                stmt,
                &common_scope,
                &branch_map,
                graph,
            ));
    }
}

fn common_block_id_map(
    data_stmts: &[Spanned<DataStmt>],
    graph: &BlockGraph,
) -> HashMap<u32, Block> {
    data_stmts
        .iter()
        .filter_map(|stmt| match &stmt.node {
            DataStmt::Block(block) => graph
                .get_block(block.pos.node)
                .cloned()
                .map(|resolved| (block.id.node, resolved)),
            DataStmt::Pipe(_) | DataStmt::Branch(_) => None,
        })
        .collect()
}

fn arm_block_id_map(arm: &[Spanned<BranchArmStmt>], resolved: &BranchArm) -> HashMap<u32, Block> {
    arm.iter()
        .filter_map(|stmt| {
            let BranchArmStmt::Block(block) = &stmt.node else {
                return None;
            };
            resolved
                .blocks()
                .find(|candidate| candidate.pos() == block.pos.node)
                .cloned()
                .map(|candidate| (block.id.node, candidate))
        })
        .collect()
}

fn branch_elements(region: &BranchRegion) -> HashSet<GraphElement> {
    [false, true]
        .into_iter()
        .flat_map(|value| {
            let arm = region.arm(value);
            arm.blocks()
                .map(|block| GraphElement::Block(block.pos()))
                .chain(
                    arm.pipes()
                        .map(|pipe| GraphElement::Pipe(pipe.src(), pipe.dst()).canonical()),
                )
                .chain(
                    region
                        .arm_incoming(value)
                        .iter()
                        .map(|cut| GraphElement::Pipe(cut.pipe.src(), cut.pipe.dst()).canonical()),
                )
        })
        .collect()
}

struct BlockScope<'a> {
    common: &'a HashMap<u32, Block>,
    local: Option<&'a HashMap<u32, Block>>,
}

impl<'a> BlockScope<'a> {
    fn common(common: &'a HashMap<u32, Block>) -> Self {
        Self {
            common,
            local: None,
        }
    }

    fn get(&self, id: u32) -> Option<&Block> {
        self.local
            .and_then(|local| local.get(&id))
            .or_else(|| self.common.get(&id))
    }
}

fn pipe_element<'a>(
    def: &PipeDef,
    blocks: &BlockScope<'_>,
    pipes: impl Iterator<Item = &'a Pipe>,
) -> Option<GraphElement> {
    let directional = match &def.dst.node {
        PipeDst::Dir(direction) => {
            let src = resolve_pipe_src(&def.src.node, *direction, blocks)?;
            Some((src, checked_add_position(src, direction.to_ivec3()).ok()?))
        }
        PipeDst::Ref(_) => None,
    };
    let mut matches = pipes.filter(|pipe| match &def.dst.node {
        PipeDst::Dir(_) => directional == Some((pipe.src(), pipe.dst())),
        PipeDst::Ref(dst) => {
            ref_matches(&def.src.node, pipe.src(), blocks) && ref_matches(dst, pipe.dst(), blocks)
        }
    });
    let pipe = matches.next()?;
    matches
        .next()
        .is_none()
        .then(|| GraphElement::Pipe(pipe.src(), pipe.dst()).canonical())
}

fn action_stmt_elements(
    stmt: &Spanned<ActionStmt>,
    blocks: &BlockScope<'_>,
    branch_map: &HashMap<String, HashSet<GraphElement>>,
    graph: &BlockGraph,
) -> HashSet<GraphElement> {
    match &stmt.node {
        ActionStmt::Measure(measure) => match &measure.target.node {
            MeasureTargetAst::Node(target) => resolve_ref(target, blocks)
                .map(GraphElement::Block)
                .into_iter()
                .collect(),
            MeasureTargetAst::Edge(target, dir) => pipe_from_ref(target, *dir, blocks, graph)
                .into_iter()
                .collect(),
        },
        ActionStmt::Resolve(resolve) => match &resolve.target.node {
            ResolveTargetDef::Ref(target) => resolve_ref(target, blocks)
                .map(GraphElement::Block)
                .into_iter()
                .collect(),
            ResolveTargetDef::Branch(name) => branch_map.get(name).cloned().unwrap_or_default(),
        },
        ActionStmt::Feedback(feedback) => feedback
            .targets
            .iter()
            .filter_map(|target| resolve_ref(&target.target.node, blocks))
            .map(GraphElement::Block)
            .collect(),
        ActionStmt::Let(_) | ActionStmt::DiscardIf(_) => HashSet::new(),
    }
}

fn pipe_from_ref(
    target: &Ref,
    dir: Direction,
    blocks: &BlockScope<'_>,
    graph: &BlockGraph,
) -> Option<GraphElement> {
    let src = resolve_pipe_src(target, dir, blocks)?;
    let dst = checked_add_position(src, dir.to_ivec3()).ok()?;
    graph
        .get_pipe(src, dst)
        .map(|_| GraphElement::Pipe(src, dst).canonical())
}

fn resolve_pipe_src(target: &Ref, dir: Direction, blocks: &BlockScope<'_>) -> Option<IVec3> {
    match target {
        Ref::Pos(position) => Some(*position),
        Ref::Id(id) => blocks
            .get(*id)
            .map(|block| block.endpoint_for_direction(dir)),
    }
}

fn ref_matches(target: &Ref, endpoint: IVec3, blocks: &BlockScope<'_>) -> bool {
    match target {
        Ref::Pos(position) => *position == endpoint,
        Ref::Id(id) => blocks.get(*id).is_some_and(|block| {
            block
                .connectable_offsets()
                .into_iter()
                .any(|offset| block.pos() + offset == endpoint)
        }),
    }
}

fn resolve_ref(target: &Ref, blocks: &BlockScope<'_>) -> Option<IVec3> {
    match target {
        Ref::Pos(pos) => Some(*pos),
        Ref::Id(id) => blocks.get(*id).map(Block::pos),
    }
}

/// Draws the editable BLOG buffer in the lower-left floating-window tile.
pub(crate) fn draw_blog_buffer(
    ctx: &egui::Context,
    viewport: egui::Rect,
    tab_id: crate::resources::EditorTabId,
    editor_state: &mut EditorState,
    state: &mut ImportExportState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if !editor_state.show_blog_buffer {
        editor_state.blog_hovered_elements.clear();
        return;
    }

    let default_rect = tiled_window_rect(viewport, egui::Align2::LEFT_BOTTOM);
    let mut open = true;
    egui::Window::new("BLOG")
        .open(&mut open)
        .default_rect(default_rect)
        .min_width(300.0_f32.min(default_rect.width()))
        .min_height(200.0_f32.min(default_rect.height()))
        .resizable(true)
        .show(ctx, |ui| {
            draw_contents(ui, tab_id, editor_state, state, intents, palette)
        });
    editor_state.show_blog_buffer = open;
    if !open {
        editor_state.blog_hovered_elements.clear();
    }
}

fn draw_contents(
    ui: &mut egui::Ui,
    tab_id: crate::resources::EditorTabId,
    editor_state: &mut EditorState,
    state: &mut ImportExportState,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    if state.sync_conflict {
        ui.label("Text and graph differ. Keep graph or use text.");
    } else if state.sync_error.is_some() {
        ui.small("Fix text to resume graph editing.");
    }
    if state.language.diagnostic.is_none()
        && let Some(error) = &state.sync_error
    {
        ui.small(error);
    }
    if state.confirm_replace_text {
        ui.horizontal_wrapped(|ui| {
            ui.label("Discard text changes?");
            if ui.button("Replace text").clicked() {
                intents.push(UiIntent::StoreGraphToBuffer);
                state.confirm_replace_text = false;
            }
            if ui.button("Keep draft").clicked() {
                state.confirm_replace_text = false;
            }
        });
    }
    let analysis_changed = state.language.cursor_char.is_none();
    let editor_id = ui.make_persistent_id(("blog_buffer_editor", tab_id));
    #[derive(Clone, Copy)]
    enum EditorKey {
        /// Open or cycle the completion list; the step is the cycle direction.
        Complete(isize),
        Accept,
        Dismiss,
        Newline,
    }
    let completion_open = state.language.completion.is_some();
    let focused = ui
        .ctx()
        .memory(|memory| memory.has_focus(editor_id) || memory.had_focus_last_frame(editor_id));
    // Only the focused editor claims keys, so the state load stays off the
    // unfocused per-frame path.
    let selection = focused
        .then(|| egui::TextEdit::load_state(ui.ctx(), editor_id))
        .flatten()
        .and_then(|state| state.cursor.char_range());
    let editor_key = focused
        .then(|| {
            ui.ctx().input_mut(|input| {
                if input.consume_key(egui::Modifiers::SHIFT, egui::Key::Tab) {
                    Some(EditorKey::Complete(-1))
                } else if input.consume_key(egui::Modifiers::NONE, egui::Key::Tab) {
                    Some(EditorKey::Complete(1))
                } else if completion_open
                    && input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)
                {
                    Some(EditorKey::Dismiss)
                } else if (completion_open || selection.is_some())
                    && input.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                {
                    // An open list takes Enter; otherwise it splits the line.
                    Some(if completion_open {
                        EditorKey::Accept
                    } else {
                        EditorKey::Newline
                    })
                } else {
                    None
                }
            })
        })
        .flatten();
    // Each arm may rewrite the buffer; the revision-guarded refresh below
    // covers all of them.
    match editor_key {
        Some(EditorKey::Accept) => {
            if let Some(completion) = state.language.completion.take() {
                let cursor = apply_completion(&mut state.bloq_buffer, &completion);
                state.touch_bloq_buffer();
                set_text_cursor(ui.ctx(), editor_id, cursor);
            }
        }
        Some(EditorKey::Dismiss) => state.language.completion = None,
        Some(EditorKey::Newline) => {
            let selected = selection
                .expect("Newline is only claimed with a cursor")
                .as_sorted_char_range();
            let selected = selected.start.into()..selected.end.into();
            if let Some(cursor) = insert_indented_newline(&mut state.bloq_buffer, selected) {
                state.touch_bloq_buffer();
                set_text_cursor(ui.ctx(), editor_id, cursor);
            }
        }
        Some(EditorKey::Complete(_)) | None => {}
    }

    let diagnostic_range = state
        .language
        .diagnostic
        .as_ref()
        .and_then(|diagnostic| diagnostic.span)
        .and_then(|span| visible_span(&state.bloq_buffer, span));
    let theme = CodeTheme::from_style(ui.style());
    let mut layouter = |ui: &egui::Ui, buffer: &dyn egui::TextBuffer, _wrap_width: f32| {
        // Syntect is synchronous. Keep large sources plain rather than doing
        // a full syntax parse on every keypress; diagnostics remain available.
        let mut job = if buffer.as_str().len() > 64 * 1024 {
            egui::text::LayoutJob::simple(
                buffer.as_str().to_owned(),
                egui::TextStyle::Monospace.resolve(ui.style()),
                palette.text_primary,
                f32::INFINITY,
            )
        } else {
            highlight_with(
                ui.ctx(),
                ui.style(),
                &theme,
                buffer.as_str(),
                "blog",
                blog_syntax_settings(),
            )
        };
        if let Some(range) = &diagnostic_range {
            underline_range(&mut job, range, palette.accent_error);
        }
        // Scroll long lines instead of relaying out the whole buffer on every resize.
        job.wrap.max_width = f32::INFINITY;
        ui.fonts_mut(|fonts| fonts.layout_job(job))
    };
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    let editor_height =
        (ui.available_height() - 2.0 * ui.spacing().interact_size.y - 12.0).max(80.0);
    let rows = (editor_height / row_height).floor().max(4.0) as usize;
    // A shift-held click must place the caret, never extend the selection.
    // `Ctrl+Shift+Z` is the usual way shift is still down on the next click,
    // but scoping this to a post-redo latch was tried three times and each
    // narrower version left the artifact reachable, so the buffer ignores
    // shift for clicks outright.
    let suppress_shift_click =
        ui.input(|input| input.pointer.primary_pressed() && input.modifiers.shift);
    if suppress_shift_click {
        ui.input_mut(|input| input.modifiers.shift = false);
    }
    let output = egui::ScrollArea::both()
        .max_height(editor_height)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::TextEdit::multiline(&mut state.bloq_buffer)
                .hint_text("BLOG text...")
                .desired_width(f32::INFINITY)
                .desired_rows(rows)
                .code_editor()
                .id(editor_id)
                .layouter(&mut layouter)
                .show(ui)
        })
        .inner;
    if suppress_shift_click {
        ui.input_mut(|input| input.modifiers.shift = true);
    }
    // Out-of-band cursor writes can leave egui holding a selection drag that
    // no pointer owns any more.
    if editor_key.is_some() || output.response.changed() {
        ui.ctx().stop_dragging();
    }
    if editor_key.is_some() {
        output.response.request_focus();
    }
    if output.response.changed() {
        state.touch_bloq_buffer();
    }

    let cursor_char = (output.response.has_focus() || editor_key.is_some())
        .then(|| primary_cursor(&output).map(|cursor| cursor.index.into()))
        .flatten();
    let cursor_changed = state.language.update_cursor(cursor_char);
    if analysis_changed || cursor_changed {
        editor_state.blog_hovered_elements = cursor_char
            .and_then(|cursor| state.language.index.line_start_at_char(cursor))
            .and_then(|line_start| state.language.index.elements_at_line(line_start).cloned())
            .unwrap_or_default();
    }
    // Must follow `update_cursor`, which clears any session the caret left.
    if let Some(EditorKey::Complete(step)) = editor_key {
        state.language.trigger_completion(&state.bloq_buffer, step);
    }
    if let Some(completion) = &state.language.completion {
        show_completion(&output, completion, palette);
    }

    ui.horizontal_wrapped(|ui| {
        if state.sync_conflict && ui.button("Use text").clicked() {
            intents.push(UiIntent::LoadBlogFromBuffer);
        }
        if state.blocks_graph_editing() && ui.button("Keep graph").clicked() {
            state.confirm_replace_text = true;
        }
        if activated(&ui.button("Format"))
            && let Some(formatted) = format_blog(&state.bloq_buffer)
            && formatted != state.bloq_buffer
        {
            state.set_bloq_buffer(formatted);
        }
        if !state.text_history.is_empty() {
            let mut restore = None;
            ui.menu_button("Text history", |ui| {
                restore = draw_text_history(ui, &state.text_history);
            });
            if let Some(index) = restore {
                state.replace_text_preserving_history(state.text_history[index].clone());
            }
        }
        draw_save_draft(ui, state, intents);
        if activated(&ui.button("Copy")) {
            intents.push(UiIntent::CopyToClipboard(state.bloq_buffer.clone()));
        }
    });
    if let Some(diagnostic) = &state.language.diagnostic {
        let summary = diagnostic.message.lines().next().unwrap_or("Invalid BLOG");
        ui.label(
            egui::RichText::new(summary)
                .small()
                .color(palette.accent_error),
        )
        .on_hover_text(&diagnostic.message);
    } else if state.sync_error.is_none()
        && !state.sync_conflict
        && state.language.revision == Some(state.bloq_revision)
    {
        ui.label(
            egui::RichText::new("BLOG valid")
                .small()
                .color(palette.success),
        );
    }
}

fn draw_text_history(ui: &mut egui::Ui, history: &[String]) -> Option<usize> {
    let mut selected = None;
    let row_height = ui.spacing().interact_size.y;
    egui::ScrollArea::vertical()
        .id_salt("blog_text_history")
        .max_height(240.0)
        .show_rows(ui, row_height, history.len(), |ui, rows| {
            for row in rows {
                let index = history.len() - 1 - row;
                let response = ui.add_sized(
                    [ui.available_width(), row_height],
                    egui::Button::new(format!("Revision {}", index + 1)),
                );
                if response
                    .on_hover_text(history_preview(&history[index]))
                    .clicked()
                {
                    selected = Some(index);
                    ui.close();
                }
            }
        });
    selected
}

fn draw_save_draft(
    ui: &mut egui::Ui,
    state: &ImportExportState,
    intents: &mut UiIntentBuffer,
) -> egui::Response {
    let response = ui.button(if cfg!(target_arch = "wasm32") { "Download draft" } else { "Save draft" })
        .on_hover_text("Save the exact text, including comments and unfinished or invalid BLOG. Does not change the graph.");
    if activated(&response) {
        intents.push(UiIntent::SaveBlogDraft(state.bloq_buffer.clone()));
    }
    response
}

fn history_preview(source: &str) -> String {
    // Bound the work before splitting lines: one line can be megabytes long.
    let preview: String = source.chars().take(400).collect();
    let preview = preview.lines().take(6).collect::<Vec<_>>().join("\n");
    format!("{preview}\n({} bytes)", source.len())
}

fn visible_span(text: &str, span: Span) -> Option<Range<usize>> {
    let mut start = (span.start as usize).min(text.len());
    let mut end = (span.end as usize).min(text.len());
    if start == end {
        if end < text.len() {
            end += text[end..].chars().next()?.len_utf8();
        } else {
            start -= text[..start].chars().next_back()?.len_utf8();
        }
    }
    (start < end).then_some(start..end)
}

fn underline_range(job: &mut egui::text::LayoutJob, range: &Range<usize>, color: egui::Color32) {
    // TextEdit may relayout edited text before its changed response clears the
    // old diagnostic. Never split a newly inserted Unicode character.
    let range = job
        .text
        .floor_char_boundary(range.start.min(job.text.len()))
        ..job.text.ceil_char_boundary(range.end.min(job.text.len()));
    if range.is_empty() {
        return;
    }
    // `LayoutJob` sections are sorted, gap-free and non-overlapping, so the
    // ones the span touches form a single run. The layouter reruns every
    // frame, so only that run is rebuilt rather than the whole vector.
    let first = job
        .sections
        .partition_point(|section| section.byte_range.end.0 <= range.start);
    let last = job
        .sections
        .partition_point(|section| section.byte_range.start.0 < range.end);
    let mut patched = Vec::with_capacity(last - first + 2);
    for section in &job.sections[first..last] {
        let start = section.byte_range.start.0;
        let end = section.byte_range.end.0;
        if start < range.start {
            let mut before = section.clone();
            before.byte_range.end = egui::text::ByteIndex(range.start);
            patched.push(before);
        }
        let mut error = section.clone();
        error.byte_range = egui::text::ByteIndex(start.max(range.start))
            ..egui::text::ByteIndex(end.min(range.end));
        error.format.underline = egui::Stroke::new(1.5, color);
        patched.push(error);
        if range.end < end {
            let mut after = section.clone();
            after.byte_range.start = egui::text::ByteIndex(range.end);
            patched.push(after);
        }
    }
    job.sections.splice(first..last, patched);
}

/// egui reports the caret in `cursor_range` on the frame it moves and in the
/// stored state afterwards, so both need consulting.
fn primary_cursor(output: &egui::text_edit::TextEditOutput) -> Option<egui::text::CCursor> {
    output
        .cursor_range
        .or_else(|| output.state.cursor.char_range())
        .map(|range| range.primary)
}

fn show_completion(
    output: &egui::text_edit::TextEditOutput,
    completion: &CompletionSession,
    palette: &ThemePalette,
) {
    let Some(cursor) = primary_cursor(output) else {
        return;
    };
    let caret = output
        .galley
        .pos_from_cursor(cursor)
        .translate(output.galley_pos.to_vec2());
    let first = completion.selected.saturating_sub(4);
    egui::Popup::new(
        output.response.id.with("completion"),
        output.response.ctx.clone(),
        caret,
        output.response.layer_id,
    )
    .kind(egui::PopupKind::Tooltip)
    .interactable(false)
    .sense(egui::Sense::empty())
    .show(|ui| {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        for (index, item) in completion.items.iter().enumerate().skip(first).take(8) {
            egui::Frame::NONE
                .fill(if index == completion.selected {
                    palette.bg_active
                } else {
                    egui::Color32::TRANSPARENT
                })
                .inner_margin(egui::Margin::symmetric(4, 1))
                .show(ui, |ui| {
                    ui.monospace(item.trim_end());
                });
        }
    });
}

fn blog_syntax_settings() -> &'static SyntectSettings {
    static SETTINGS: OnceLock<SyntectSettings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let SyntectSettings { ps, ts } = SyntectSettings::default();
        let mut syntaxes = ps.into_builder();
        syntaxes.add(
            SyntaxDefinition::load_from_str(BLOG_SYNTAX, true, None)
                .expect("embedded BLOG syntax is valid"),
        );
        SyntectSettings {
            ps: syntaxes.build(),
            ts,
        }
    })
}

#[cfg(test)]
fn elements_at_blog_cursor(buffer: &str, cursor_char_index: usize) -> HashSet<GraphElement> {
    let mut language = BlogLanguageState::default();
    language.refresh(buffer, 1);
    let Some(line_start) = language.index.line_start_at_char(cursor_char_index) else {
        return HashSet::new();
    };
    language
        .index
        .elements_at_line(line_start)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #[test]
    fn live_blog_initializes_from_graph_and_preserves_incomplete_text() {
        let graph = crate::resources::GraphState::default();
        let mut state = super::ImportExportState::default();
        state.refresh_from_graph(&graph);
        assert_eq!(
            state.bloq_buffer,
            graph.resolved_graph().unwrap().to_blog_text()
        );
        state.set_bloq_buffer("# unfinished text".into());
        state.refresh_from_graph(&graph);
        assert_eq!(state.bloq_buffer, "# unfinished text");
        state.set_bloq_buffer(String::new());
        state.refresh_from_graph(&graph);
        assert!(
            state.bloq_buffer.is_empty(),
            "an intentionally cleared draft stays empty"
        );
    }

    #[test]
    fn graph_refresh_preserves_previous_source_in_text_history() {
        let mut graph = crate::resources::GraphState::default();
        let mut text = ImportExportState::default();
        text.refresh_from_graph(&graph);
        text.set_bloq_buffer(format!("{}\n# important comment\n", text.bloq_buffer));
        text.mark_synchronized(graph.revision);
        let original = text.bloq_buffer.clone();
        graph.graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.commit();
        text.refresh_from_graph(&graph);
        assert_eq!(text.text_history.last(), Some(&original));
        assert!(!text.blocks_graph_editing());
        text.set_bloq_buffer(String::new());
        assert!(
            crate::systems::jobs::analyze_live_blog(&text.bloq_buffer, text.bloq_revision)
                .result
                .is_err()
        );
        assert!(text.blocks_graph_editing());
    }

    #[test]
    fn restored_text_requires_a_choice_unless_graphs_match() {
        let mut graph = crate::resources::GraphState::default();
        graph.graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.commit();
        for source in ["BLOG 1.0", "BLOG 1.0\n0:", ""] {
            let mut text = ImportExportState::default();
            text.set_bloq_buffer(source.into());
            text.restore_graph_sync(&graph);
            text.refresh_from_graph(&graph);
            assert_eq!(text.bloq_buffer, source);
            assert!(text.sync_conflict);
            text.last_text_edit -= std::time::Duration::from_secs(1);
            assert_eq!(
                text.take_analysis_request(graph.revision, false),
                Some(false)
            );
        }
        let source = format!(
            "{}\n# preserve me\n",
            graph.resolved_graph().unwrap().to_blog_text()
        );
        let mut text = ImportExportState::default();
        text.set_bloq_buffer(source.clone());
        text.restore_graph_sync(&graph);
        assert!(!text.blocks_graph_editing());
        assert_eq!(text.bloq_buffer, source);
    }

    #[test]
    fn recovering_history_preserves_pending_text() {
        let mut text = ImportExportState::default();
        text.set_bloq_buffer("unfinished text".into());
        text.replace_text_preserving_history("BLOG 1.0".into());
        assert_eq!(text.text_history, vec!["unfinished text"]);
        text.replace_text_preserving_history("BLOG 1.0".into());
        assert_eq!(text.text_history.len(), 1);
    }

    #[test]
    fn same_caret_edit_dismisses_completion_before_background_analysis() {
        let mut text = ImportExportState::default();
        text.set_bloq_buffer("mo".into());
        text.language.update_cursor(Some(1));
        text.language.trigger_completion(&text.bloq_buffer, 1);
        assert_eq!(text.language.completion.as_ref().unwrap().replace, 0..2);

        // Forward Delete leaves the caret unchanged, so cursor tracking alone
        // would retain an out-of-bounds replacement until analysis completes.
        text.bloq_buffer.remove(1);
        text.touch_bloq_buffer();
        assert!(!text.language.update_cursor(Some(1)));
        assert!(text.language.completion.is_none());
    }

    #[test]
    fn reverting_invalid_text_clears_stale_error() {
        let graph = crate::resources::GraphState::default();
        let mut text = ImportExportState::default();
        text.refresh_from_graph(&graph);
        let original = text.bloq_buffer.clone();
        text.set_bloq_buffer("invalid".into());
        text.sync_error =
            crate::systems::jobs::analyze_live_blog(&text.bloq_buffer, text.bloq_revision)
                .result
                .err();
        assert!(text.sync_error.is_some());
        text.set_bloq_buffer(original);
        assert!(!text.blocks_graph_editing());
        assert!(text.sync_error.is_none());
    }

    #[test]
    fn large_source_keystrokes_do_not_parse_before_debounce() {
        let mut text = ImportExportState::default();
        let graph = crate::resources::GraphState::default();
        text.refresh_from_graph(&graph);
        let large = "# comment\n".repeat(100_000);
        let started = std::time::Instant::now();
        for _ in 0..20 {
            text.set_bloq_buffer(large.clone());
            assert_eq!(text.take_analysis_request(graph.revision, false), None);
            assert!(text.attempted_text_revision.is_none());
        }
        eprintln!(
            "20 edits of {} bytes without parsing: {:?}",
            large.len(),
            started.elapsed()
        );
        text.last_text_edit -= std::time::Duration::from_secs(1);
        assert_eq!(
            text.take_analysis_request(graph.revision, false),
            Some(true)
        );
        assert_eq!(text.take_analysis_request(graph.revision, false), None);
        assert!(history_preview(&large).chars().count() < 450);
        let single_line = "界".repeat(1_000_000);
        assert_eq!(
            history_preview(&single_line),
            format!("{}\n({} bytes)", "界".repeat(400), single_line.len())
        );
    }

    #[test]
    fn changed_draft_cancels_pending_replacement_confirmation() {
        let mut state = super::ImportExportState {
            confirm_replace_text: true,
            ..Default::default()
        };
        state.set_bloq_buffer("# newer draft".into());
        assert!(!state.confirm_replace_text);
    }

    use super::*;
    use glam::ivec3;

    #[test]
    fn text_history_virtualizes_rows_and_keeps_original_revision_indices() {
        let ctx = egui::Context::default();
        let history = (0..10_000)
            .map(|i| format!("# draft {i}"))
            .collect::<Vec<_>>();
        let frame = |offset: Option<f32>, events| {
            let mut selected = None;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(300.0, 300.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    if let Some(offset) = offset {
                        let id = ui.make_persistent_id(egui::IdSalt::new("blog_text_history"));
                        let mut state =
                            egui::scroll_area::State::load(&ctx, id).unwrap_or_default();
                        state.offset.y = offset;
                        state.store(&ctx, id);
                    }
                    selected = draw_text_history(ui, &history);
                },
            );
            let labels = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::epaint::Shape::Text(text)
                        if text.galley.text().starts_with("Revision ") =>
                    {
                        Some((
                            text.galley.text().to_owned(),
                            text.pos + egui::vec2(3.0, 3.0),
                        ))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            output.textures_delta.clear();
            (selected, labels)
        };
        let click = |pos| {
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ]
        };
        let (_, labels) = frame(None, vec![]);
        assert!(labels.len() < 20, "only viewport rows should be built");
        let newest = labels
            .iter()
            .find(|(label, _)| label == "Revision 10000")
            .unwrap()
            .1;
        assert_eq!(frame(None, click(newest)).0, Some(9999));
        frame(Some(1_000_000.0), vec![]);
        let (_, labels) = frame(None, vec![]);
        assert!(labels.len() < 20);
        let oldest = labels
            .iter()
            .find(|(label, _)| label == "Revision 1")
            .unwrap()
            .1;
        assert_eq!(frame(None, click(oldest)).0, Some(0));
        assert_eq!(history[0], "# draft 0");
        assert_eq!(history.len(), 10_000);
    }

    #[test]
    fn save_draft_exports_invalid_text_exactly_without_changing_state() {
        let ctx = egui::Context::default();
        let source = "# unfinished α draft\r\nBLOG 1.0\r\n0: ZXZ [";
        let mut state = ImportExportState::default();
        state.set_bloq_buffer(source.into());
        state.sync_error = Some("unfinished".into());
        let revision = state.bloq_revision;
        let history = state.text_history.clone();
        let mut intents = UiIntentBuffer::default();
        let mut rect = egui::Rect::NOTHING;
        let mut output = ctx.run_ui(Default::default(), |ui| {
            rect = draw_save_draft(ui, &state, &mut intents).rect;
        });
        output.textures_delta.clear();
        let pos = rect.center();
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |ui| {
                draw_save_draft(ui, &state, &mut intents);
            },
        );
        output.textures_delta.clear();
        let saved = intents.drain().collect::<Vec<_>>();
        assert!(matches!(saved.as_slice(), [UiIntent::SaveBlogDraft(text)] if text == source));
        assert_eq!(state.bloq_buffer, source);
        assert_eq!(state.bloq_revision, revision);
        assert_eq!(state.text_history, history);
        assert_eq!(state.sync_error.as_deref(), Some("unfinished"));
    }

    fn run_buffer_frame(
        context: &egui::Context,
        editor: &mut EditorState,
        state: &mut ImportExportState,
        intents: &mut UiIntentBuffer,
        events: Vec<egui::Event>,
    ) -> egui::Id {
        run_tab_buffer_frame(
            context,
            crate::resources::EditorTabId::new(1),
            editor,
            state,
            intents,
            events,
        )
    }

    fn run_tab_buffer_frame(
        context: &egui::Context,
        tab_id: crate::resources::EditorTabId,
        editor: &mut EditorState,
        state: &mut ImportExportState,
        intents: &mut UiIntentBuffer,
        events: Vec<egui::Event>,
    ) -> egui::Id {
        // UI-only fixture supplies the analysis normally delivered by jobs.
        state
            .language
            .refresh(&state.bloq_buffer, state.bloq_revision);
        let mut editor_id = egui::Id::NULL;
        let mut output = context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                editor_id = ui.make_persistent_id(("blog_buffer_editor", tab_id));
                ui.memory_mut(|memory| {
                    memory.request_focus(editor_id);
                });
                draw_contents(
                    ui,
                    tab_id,
                    editor,
                    state,
                    intents,
                    crate::theme::palette(crate::theme::ThemePreset::default()),
                );
            },
        );
        output.textures_delta.clear();
        state
            .language
            .refresh(&state.bloq_buffer, state.bloq_revision);
        editor_id
    }

    #[test]
    fn blog_buffer_undo_and_redo_shortcuts() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut state = ImportExportState::default();
        let mut intents = UiIntentBuffer::default();
        let initial = "BLOG 1.0\nmodule main {\n}\n";
        state.set_bloq_buffer(initial.to_owned());

        run_buffer_frame(&context, &mut editor, &mut state, &mut intents, Vec::new());
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![egui::Event::Text("#".to_owned())],
        );
        assert_eq!(state.bloq_buffer, format!("{initial}#"));

        let key = |modifiers| egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![key(egui::Modifiers::COMMAND | egui::Modifiers::CTRL)],
        );
        assert_eq!(state.bloq_buffer, initial);
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![key(egui::Modifiers::COMMAND
                | egui::Modifiers::CTRL
                | egui::Modifiers::SHIFT)],
        );
        assert_eq!(state.bloq_buffer, format!("{initial}#"));
    }

    #[test]
    fn blog_undo_stays_with_its_tab() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut intents = UiIntentBuffer::default();
        let mut a = ImportExportState::default();
        let mut b = ImportExportState::default();
        a.set_bloq_buffer("# tab A".into());
        b.set_bloq_buffer("# tab B".into());
        let mut frame = |tab, state: &mut ImportExportState, events| {
            run_tab_buffer_frame(
                &context,
                crate::resources::EditorTabId::new(tab),
                &mut editor,
                state,
                &mut intents,
                events,
            );
        };
        frame(1, &mut a, vec![]);
        frame(1, &mut a, vec![egui::Event::Text(" edited".into())]);
        frame(2, &mut b, vec![]);
        let undo = egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND | egui::Modifiers::CTRL,
        };
        frame(2, &mut b, vec![undo.clone()]);
        assert_eq!(b.bloq_buffer, "# tab B");
        frame(1, &mut a, vec![undo]);
        assert_eq!(a.bloq_buffer, "# tab A");
    }

    /// Sets a caret, then clicks at `pos` with `modifiers`, and reports the
    /// resulting cursor range.
    fn click_after_caret(
        context: &egui::Context,
        editor: &mut EditorState,
        state: &mut ImportExportState,
        intents: &mut UiIntentBuffer,
        editor_id: egui::Id,
        caret: usize,
        modifiers: egui::Modifiers,
    ) -> egui::text::CCursorRange {
        set_text_cursor(context, editor_id, caret);
        let rect = context
            .read_response(editor_id)
            .expect("editor response")
            .rect;
        let pos = rect.left_top() + egui::vec2(120.0, 65.0);
        assert!(rect.contains(pos));
        run_buffer_frame(
            context,
            editor,
            state,
            intents,
            vec![
                egui::Event::ModifiersChanged(modifiers),
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers,
                },
            ],
        );
        egui::TextEdit::load_state(context, editor_id)
            .and_then(|state| state.cursor.char_range())
            .expect("cursor after click")
    }

    const SELECTION_SOURCE: &str =
        "BLOG 1.0\nmodule main {\n  in q0: data = 48\n  in q1: data = 49\n}\n";

    /// `Ctrl+Shift+Z` leaves shift physically held, so without the guard the
    /// click that follows a redo extends the selection from the restored
    /// caret instead of placing it.
    #[test]
    fn redo_does_not_extend_the_next_click() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut state = ImportExportState::default();
        let mut intents = UiIntentBuffer::default();
        state.set_bloq_buffer(SELECTION_SOURCE.to_owned());

        let editor_id =
            run_buffer_frame(&context, &mut editor, &mut state, &mut intents, Vec::new());
        let redo = egui::Modifiers::COMMAND | egui::Modifiers::CTRL | egui::Modifiers::SHIFT;
        set_text_cursor(
            &context,
            editor_id,
            SELECTION_SOURCE[..SELECTION_SOURCE.find("48").unwrap() + 2]
                .chars()
                .count(),
        );
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![
                egui::Event::ModifiersChanged(redo),
                egui::Event::Key {
                    key: egui::Key::Z,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: redo,
                },
            ],
        );
        let cursor = click_after_caret(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            editor_id,
            SELECTION_SOURCE[..SELECTION_SOURCE.find("48").unwrap() + 2]
                .chars()
                .count(),
            redo,
        );

        assert!(cursor.is_empty(), "redo artifact extended the selection");
    }

    #[test]
    fn shift_does_not_extend_the_next_click() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut state = ImportExportState::default();
        let mut intents = UiIntentBuffer::default();
        state.set_bloq_buffer(SELECTION_SOURCE.to_owned());

        let editor_id =
            run_buffer_frame(&context, &mut editor, &mut state, &mut intents, Vec::new());
        let cursor = click_after_caret(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            editor_id,
            SELECTION_SOURCE[..SELECTION_SOURCE.find("48").unwrap() + 2]
                .chars()
                .count(),
            egui::Modifiers::SHIFT,
        );

        assert!(cursor.is_empty(), "shift extended the selection on click");
    }

    /// Writing the buffer behind the widget's back can leave egui holding a
    /// selection drag that no pointer owns any more.
    #[test]
    fn blog_buffer_edit_ends_stale_pointer_drag() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut state = ImportExportState::default();
        let mut intents = UiIntentBuffer::default();
        state.set_bloq_buffer("BLOG 1.0".to_owned());

        let editor_id =
            run_buffer_frame(&context, &mut editor, &mut state, &mut intents, Vec::new());
        context.set_dragged_id(editor_id);
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![egui::Event::Text("#".to_owned())],
        );

        assert_eq!(context.dragged_id(), None);
    }

    #[test]
    fn blog_buffer_completion_is_manual_and_undoable() {
        let context = egui::Context::default();
        let mut editor = EditorState::default();
        let mut state = ImportExportState::default();
        let mut intents = UiIntentBuffer::default();
        let initial = "BLOG 1.0\nmodule main {\n  0: ZX\n}\n";
        state.set_bloq_buffer(initial.to_owned());

        let editor_id =
            run_buffer_frame(&context, &mut editor, &mut state, &mut intents, Vec::new());
        let cursor = initial[..initial.find("ZX").unwrap() + 2].chars().count();
        assert!(completion_at(initial, cursor, &CompletionWords::default()).is_some());
        set_text_cursor(&context, editor_id, cursor);
        let key = |key, modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };

        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![key(egui::Key::Tab, egui::Modifiers::NONE)],
        );
        assert_eq!(state.bloq_buffer, initial);
        assert!(state.language.completion.is_some());
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert_eq!(state.bloq_buffer, initial.replace("ZX", "ZXZ"));
        assert!(state.language.completion.is_none());
        run_buffer_frame(
            &context,
            &mut editor,
            &mut state,
            &mut intents,
            vec![key(egui::Key::Z, egui::Modifiers::COMMAND)],
        );
        assert_eq!(state.bloq_buffer, initial);
    }

    /// Draws a bare TextEdit plus completion popup across frames, recording
    /// what the assertions need. A struct rather than a closure keeps the
    /// per-frame outputs from having to travel as `&mut` out-parameters.
    struct PopupProbe {
        text: String,
        offset: f32,
        editor_id: egui::Id,
        editor_rect: egui::Rect,
        editor_layer: egui::LayerId,
    }

    impl PopupProbe {
        fn frame(
            &mut self,
            context: &egui::Context,
            completion: &CompletionSession,
            events: Vec<egui::Event>,
        ) {
            let mut output = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(500.0, 500.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    ui.set_width(400.0);
                    ui.set_height(300.0);
                    self.editor_id = ui.make_persistent_id("editor");
                    ui.memory_mut(|memory| memory.request_focus(self.editor_id));
                    let output = egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            egui::TextEdit::multiline(&mut self.text)
                                .desired_width(f32::INFINITY)
                                .desired_rows(10)
                                .code_editor()
                                .cursor_at_end(false)
                                .id(self.editor_id)
                                .show(ui)
                        });
                    self.offset = output.state.offset.y;
                    self.editor_rect = output.inner_rect;
                    self.editor_layer = output.inner.response.layer_id;
                    show_completion(
                        &output.inner,
                        completion,
                        crate::theme::palette(crate::theme::ThemePreset::default()),
                    );
                },
            );
            output.textures_delta.clear();
        }
    }

    #[test]
    fn completion_popup_allows_editor_input() {
        let context = egui::Context::default();
        let completion = CompletionSession {
            replace: 0..0,
            items: vec!["module ".to_owned()],
            selected: 0,
        };
        let mut probe = PopupProbe {
            text: (0..100).map(|line| format!("line {line}\n")).collect(),
            offset: 0.0,
            editor_id: egui::Id::NULL,
            editor_rect: egui::Rect::NOTHING,
            editor_layer: egui::LayerId::background(),
        };

        probe.frame(
            &context,
            &completion,
            vec![egui::Event::PointerMoved(egui::pos2(20.0, 20.0))],
        );
        probe.frame(&context, &completion, Vec::new());

        // The popup must not steal the pointer: it sits over the editor, but
        // the editor's layer still owns that point.
        let pointer = context
            .read_response(probe.editor_id.with("completion"))
            .expect("completion popup response")
            .rect
            .center();
        assert!(probe.editor_rect.contains(pointer));
        assert_eq!(context.layer_id_at(pointer), Some(probe.editor_layer));

        probe.frame(
            &context,
            &completion,
            vec![
                egui::Event::PointerMoved(pointer),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, -100.0),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(probe.offset > 0.0);
    }

    fn cursor_on_line(buffer: &str, needle: &str) -> usize {
        buffer[..buffer.find(needle).expect("needle exists")]
            .chars()
            .count()
    }

    #[test]
    fn blog_syntax_highlights_language_tokens() {
        assert!(
            ["module", "-H>"]
                .iter()
                .all(|token| BLOG_SYNTAX.contains(token))
        );
        assert!(
            ["discard_if", "WALKING_"]
                .iter()
                .all(|token| !BLOG_SYNTAX.contains(token))
        );

        let ctx = egui::Context::default();
        let style = egui::Style::default();
        let theme = CodeTheme::from_style(&style);
        let source = "BLOG 1.0\nimport \"child.blog\" as Child\nmodule main {\n  in enabled\n  out done = target.done\n  module = input-bit\n  Port = input-bit\n  height = input-bit\n  role = input-bit\n  ccz/1 = input-bit\n  module-name = input-bit\n  Port-value = input-bit\n  axis+X-name = input-bit\n  source: Child @ [0, 0, 0]\n  target: Child @ [0, 0, 1]\n  0: Port [0, 0, 0] role=output <q>\n  1: ZXZ [1, 0, 0] height=2d/3-1\n  0 -> +X\n  source.q -H> target.q\n  enabled => target.enabled\n}\n";
        let job = highlight_with(&ctx, &style, &theme, source, "blog", blog_syntax_settings());
        let colors = job
            .sections
            .iter()
            .map(|section| section.format.color)
            .collect::<HashSet<_>>();

        assert!(colors.len() >= 3, "BLOG tokens should use distinct colors");
        let color_at = |byte: usize| {
            job.sections
                .iter()
                .find(|section| section.byte_range.contains(&egui::text::ByteIndex(byte)))
                .expect("byte is styled")
                .format
                .color
        };
        assert_ne!(
            color_at(source.find("module main").unwrap()),
            color_at(source.find("module-name").unwrap())
        );
        assert_ne!(
            color_at(source.find("Port [").unwrap()),
            color_at(source.find("Port-value").unwrap())
        );
        let plain = color_at(source.find("input-bit").unwrap());
        assert_eq!(color_at(source.find("module =").unwrap()), plain);
        assert_eq!(color_at(source.find("Port =").unwrap()), plain);
        assert_eq!(color_at(source.find("height =").unwrap()), plain);
        assert_eq!(color_at(source.find("role =").unwrap()), plain);
        let resource = source.find("ccz/1").unwrap();
        assert_eq!(color_at(resource), color_at(resource + 4));
        let direction = source.find("+X\n").unwrap();
        let embedded_direction = source.find("+X-name").unwrap();
        assert!(job.sections.iter().any(|section| {
            section.byte_range.start == egui::text::ByteIndex(direction)
                && section.byte_range.end == egui::text::ByteIndex(direction + 2)
        }));
        assert!(!job.sections.iter().any(|section| {
            section.byte_range.start == egui::text::ByteIndex(embedded_direction)
                && section.byte_range.end == egui::text::ByteIndex(embedded_direction + 2)
        }));
        let at = source.find('@').unwrap();
        assert!(job.sections.iter().any(|section| {
            section.byte_range.start == egui::text::ByteIndex(at)
                && section.byte_range.end == egui::text::ByteIndex(at + 1)
        }));
        let modifier = source.find("role=output").unwrap();
        let height = source.find("height=2d/3-1").unwrap();
        for (start, len) in [
            (modifier + 5, 6),
            (height + 7, 1),
            (height + 10, 1),
            (height + 11, 2),
        ] {
            assert!(job.sections.iter().any(|section| {
                section.byte_range.start == egui::text::ByteIndex(start)
                    && section.byte_range.end == egui::text::ByteIndex(start + len)
                    && section.format.color != plain
            }));
        }
    }

    #[test]
    fn cursor_line_index_handles_unicode_and_end() {
        let blog = "BLOG 1.0\nmodule main {\n# α\n#\n}\n";
        let mut language = BlogLanguageState::default();
        language.refresh(blog, 1);
        let index = language.index;
        let alpha_line = blog.find("# α").unwrap();
        let alpha_cursor = blog[..alpha_line].chars().count();
        assert_eq!(index.line_start_at_char(alpha_cursor + 2), Some(alpha_line));
        assert_eq!(index.line_start_at_char(blog.chars().count() + 1), None);
    }

    #[test]
    fn graph_body_language_matches_application_and_preserves_source_lines() {
        let source = "BLOG 1.0\n0: ZXZ [0,0,0]\n1: ZXZ [0,0,1]\n0 -> +Z\n2: XZX [3,0,0]\n3: XZX [3,0,1]\n2 -> +Z\n";
        assert_eq!(BlockGraph::from_text(source).unwrap().block_count(), 4);
        let mut language = BlogLanguageState::default();
        language.refresh(source, 1);
        assert!(language.diagnostic.is_none());
        let line = source.find("2: XZX").unwrap();
        assert!(
            language
                .index
                .elements_at_line(line)
                .unwrap()
                .contains(&GraphElement::Block(IVec3::new(3, 0, 0)))
        );
        let pipe = source.find("2 ->").unwrap();
        assert!(
            language
                .index
                .elements_at_line(pipe)
                .unwrap()
                .contains(&GraphElement::Pipe(
                    IVec3::new(3, 0, 0),
                    IVec3::new(3, 0, 1)
                ))
        );
        assert_eq!(format_blog(source).as_deref(), Some(source));
        language.refresh("BLOG 1.0\n0: ZXZ [0,0,0]\n1: XZX [0,0,0]\n", 2);
        assert!(language.diagnostic.is_some());
        language.refresh("BLOG 1.0\n0: ZXZ [0,0,\n", 3);
        assert!(language.diagnostic.as_ref().unwrap().span.unwrap().start > 9);
        language.refresh(source, 4);
        assert!(language.diagnostic.is_none());
        language.refresh("BLOG 1.0\n", 5);
        assert!(language.diagnostic.is_none());
    }

    #[test]
    fn language_analysis_is_cached_and_reports_parse_and_lower_errors() {
        let mut language = BlogLanguageState::default();
        let valid = "BLOG 1.0\n\nmodule main {\n}\n";
        assert!(language.refresh(valid, 1));
        assert!(!language.refresh(valid, 1));
        assert!(language.diagnostic.is_none());

        let invalid = "BLOG 1.0\n\nmodule main {\n  0: ZXZ [0, 0,\n}\n";
        assert!(language.refresh(invalid, 2));
        assert!(language.diagnostic.as_ref().unwrap().span.is_some());
        assert!(language.words.modules.iter().any(|word| word == "main"));

        let overlap = "BLOG 1.0\n\nmodule main {\n  0: ZXZ [0, 0, 0]\n  1: ZXZ [0, 0, 0]\n}\n";
        assert!(language.refresh(overlap, 3));
        assert!(language.diagnostic.is_some());
    }

    #[test]
    fn formatter_indents_without_rewriting_comments_or_tags() {
        let source = "BLOG 1.0  \n\nmodule main {   \n0: ZXZ [0, 0, 0] # } ignored  \nbranch b {\nfalse {\n1: X [0, 0, 1] <tag{keep}>\n}\ntrue {\n2: Z [0, 0, 1]\n}\n}\n}\n\n";
        let expected = "BLOG 1.0\n\nmodule main {\n  0: ZXZ [0, 0, 0] # } ignored\n  branch b {\n    false {\n      1: X [0, 0, 1] <tag{keep}>\n    }\n    true {\n      2: Z [0, 0, 1]\n    }\n  }\n}\n";

        let formatted = format_blog(source).unwrap();
        assert_eq!(formatted, expected);
        assert_eq!(format_blog(&formatted).unwrap(), formatted);
    }

    #[test]
    fn stale_diagnostic_ranges_keep_unicode_layout_valid() {
        let mut job = egui::text::LayoutJob::simple(
            "é中".into(),
            egui::FontId::monospace(12.0),
            egui::Color32::WHITE,
            f32::INFINITY,
        );
        underline_range(&mut job, &(1..3), egui::Color32::RED);
        underline_range(&mut job, &(20..30), egui::Color32::RED);
        assert!(job.sections.iter().all(|section| {
            job.text.is_char_boundary(section.byte_range.start.0)
                && job.text.is_char_boundary(section.byte_range.end.0)
        }));
        assert_eq!(job.sections.len(), 1);
        assert_eq!(job.sections[0].format.underline.color, egui::Color32::RED);
    }

    #[test]
    fn newline_keeps_indent_and_splits_braces() {
        let mut block = "module main {}".to_owned();
        let cursor = "module main {".chars().count();
        let cursor = insert_indented_newline(&mut block, cursor..cursor).unwrap();
        assert_eq!(block, "module main {\n  \n}");
        assert_eq!(cursor, "module main {\n  ".chars().count());

        let mut statement = "  0: ZXZ [0, 0, 0]".to_owned();
        let cursor = statement.chars().count();
        insert_indented_newline(&mut statement, cursor..cursor).unwrap();
        assert_eq!(statement, "  0: ZXZ [0, 0, 0]\n  ");
    }

    #[test]
    fn completion_uses_context_and_module_members() {
        let partial = "BLOG 1.0\n\nmodule main {\n  0: ZX";
        let mut language = BlogLanguageState::default();
        language.refresh(partial, 1);
        language.update_cursor(Some(partial.chars().count()));
        assert!(language.completion.is_none());
        language.trigger_completion(partial, 1);
        let block_kinds = language.completion.as_ref().unwrap();
        assert!(block_kinds.items.iter().any(|item| item == "ZXZ"));

        let valid = "BLOG 1.0\n\nmodule Child {\n  in q: data = 0\n  out ready = flag\n  0: Port [0, 0, 0] role=input <q>\n}\n\nmodule main {\n  child: Child @ [0, 0, 0]\n}\n";
        let source = parse_blog_program_to_ast(valid).unwrap();
        let words = completion_words(&source);
        assert!(
            [
                &words.modules,
                &words.branches,
                &words.expressions,
                &words.references,
            ]
            .into_iter()
            .flatten()
            .all(|word| word.parse::<u64>().is_err())
        );
        let editing = format!("{}  child.\n}}\n", valid.strip_suffix("}\n").unwrap());
        let cursor = editing[..editing.find("child.\n").unwrap() + "child.".len()]
            .chars()
            .count();
        let mut members = completion_at(&editing, cursor, &words).unwrap();
        assert_eq!(
            members
                .items
                .iter()
                .map(String::as_str)
                .collect::<HashSet<_>>(),
            HashSet::from(["child.q", "child.ready"])
        );
        members.cycle(-1);
        assert_eq!(members.selected, members.items.len() - 1);
        members.cycle(1);
        assert_eq!(members.selected, 0);

        let complete = |line: &str| completion_at(line, line.chars().count(), &words);
        assert!(complete("  in q: ").is_none());
        assert!(complete("  0: ZXZ [0, ").is_none());
        assert!(complete("  child: Child @ [0, 0, 0]").is_none());
        assert!(
            complete("  0: ZXZ [0, 0, 0]")
                .unwrap()
                .items
                .iter()
                .any(|item| item == "height=")
        );
        let feedback = complete("  feedback X 0, ").unwrap();
        assert_eq!(feedback.items, ["X", "Y", "Z"]);
        let rotation = complete("  child: Child @ [0, 0, 0] rotate ").unwrap();
        assert_eq!(rotation.items, ["X", "Y", "Z"]);
        let bit_target = complete("  enabled => ").unwrap();
        assert!(!bit_target.items.iter().any(|item| item == "+X"));
    }

    #[test]
    fn data_block_statement_maps_to_block_element() {
        let blog = "BLOG 1.0\n\nmodule main {\n  0: ZXZ [1, 2, 3]\n}\n";

        let elements = elements_at_blog_cursor(blog, cursor_on_line(blog, "0:"));

        assert_eq!(
            elements,
            HashSet::from([GraphElement::Block(ivec3(1, 2, 3))])
        );
    }

    #[test]
    fn data_pipe_statement_maps_to_pipe_element() {
        let blog = "\
BLOG 1.0

module main {
  0: ZXZ [0, 0, 0]
  1: ZXZ [1, 0, 0]
  0 -> +X
}
";

        let elements = elements_at_blog_cursor(blog, cursor_on_line(blog, "0 ->"));

        assert_eq!(
            elements,
            HashSet::from([GraphElement::Pipe(ivec3(0, 0, 0), ivec3(1, 0, 0)).canonical()])
        );
    }

    #[test]
    fn hover_uses_resolved_extended_block_endpoints() {
        let blog = "\
BLOG 1.0

module main {
  0: walk XZZ [0, 0, 0] -> [1, 1, 1]
  1: ZXZ [2, 1, 1]
  0 -> +X
  m = measure 0 -> +X
}
";
        let expected =
            HashSet::from([GraphElement::Pipe(ivec3(1, 1, 1), ivec3(2, 1, 1)).canonical()]);

        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "0 -> +X")),
            expected
        );
        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "measure")),
            expected
        );
    }

    #[test]
    fn branch_hover_does_not_resolve_sibling_ids() {
        let blog = "BLOG 1.0\n\nmodule main {\n0: ZXZ [0,0,0]\nbranch b {\n  false {\n    1: ZXZ [0,0,1]\n    0 -> 1\n  }\n  true {\n    2: ZXZ [0,0,1]\n    0 -> 1 # sibling\n  }\n}\n}\n";

        assert!(elements_at_blog_cursor(blog, cursor_on_line(blog, "0 -> 1 # sibling")).is_empty());
    }

    #[test]
    fn modular_hover_uses_each_definition_id_scope() {
        let blog =
            "BLOG 1.0\n\nmodule A {\n  0: ZXZ [1,0,0]\n}\n\nmodule main {\n  0: ZXZ [2,0,0]\n}\n";

        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "0: ZXZ [1")),
            HashSet::from([GraphElement::Block(ivec3(1, 0, 0))])
        );
        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "0: ZXZ [2")),
            HashSet::from([GraphElement::Block(ivec3(2, 0, 0))])
        );
    }

    #[test]
    fn module_instance_statement_maps_to_whole_instance() {
        let blog = "BLOG 1.0\n\nmodule Stage {\n  0: ZXZ [0,0,0]\n  1: XZZ [1,0,0]\n  0 -> +X\n}\n\nmodule main {\n  stage: Stage @ [2,0,0] rotate Z 90\n}\n";

        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "stage:")),
            HashSet::from([
                GraphElement::Block(ivec3(2, 0, 0)),
                GraphElement::Block(ivec3(2, 1, 0)),
                GraphElement::Pipe(ivec3(2, 0, 0), ivec3(2, 1, 0)).canonical(),
            ])
        );
    }

    #[test]
    fn modular_direct_pipe_maps_to_composed_endpoints() {
        let blog = "BLOG 1.0\n\nmodule Stage {\n  in q_in: data = 0\n  out q_out: data = 1\n  0: Port [0,0,0] role=input <q_in>\n  1: Port [0,0,2] role=output <q_out>\n  2: ZXZ [0,0,1]\n  0 -> +Z\n  2 -> +Z\n}\n\nmodule main {\n  in q_in: data = 100\n  out q_out: data = 101\n  lower: Stage @ [0,0,0]\n  upper: Stage @ [0,0,1]\n  100: Port [0,0,0] role=input <q_in>\n  101: Port [0,0,3] role=output <q_out>\n  100 -> lower.q_in\n  lower.q_out -> upper.q_in\n  upper.q_out -> 101\n}\n";

        assert_eq!(
            elements_at_blog_cursor(blog, cursor_on_line(blog, "lower.q_out ->")),
            HashSet::from([GraphElement::Pipe(ivec3(0, 0, 1), ivec3(0, 0, 2)).canonical()])
        );
    }

    #[test]
    fn overflowing_pipe_has_no_hover_element() {
        let blog = "BLOG 1.0\n\nmodule main {\n0: ZXZ [2147483647, 0, 0]\n0 -> +X\n}\n";

        assert!(elements_at_blog_cursor(blog, cursor_on_line(blog, "0 ->")).is_empty());
    }

    #[test]
    fn measure_pipe_action_maps_to_pipe_element() {
        let blog = "\
BLOG 1.0

module main {
  0: ZXZ [0, 0, 0]
  1: ZXZ [1, 0, 0]
  0 -> +X

  mx = measure 0 -> +X
}
";

        let elements = elements_at_blog_cursor(blog, cursor_on_line(blog, "measure"));

        assert_eq!(
            elements,
            HashSet::from([GraphElement::Pipe(ivec3(0, 0, 0), ivec3(1, 0, 0)).canonical()])
        );
    }

    #[test]
    fn feedback_action_maps_to_all_target_blocks() {
        let blog = "\
BLOG 1.0

module main {
  0: ZXZ [0, 0, 0]
  1: ZXZ [1, 0, 0]

  feedback X 0, Z 1 if m
}
";

        let elements = elements_at_blog_cursor(blog, cursor_on_line(blog, "feedback"));

        assert_eq!(
            elements,
            HashSet::from([
                GraphElement::Block(ivec3(0, 0, 0)),
                GraphElement::Block(ivec3(1, 0, 0)),
            ])
        );
    }

    #[test]
    fn named_resolve_maps_to_its_whole_branch_region() {
        let blog = "\
BLOG 1.0

module main {
  0: ZXZ [0, 0, 0]
  branch b0 {
    false {
      1: X [0, 0, 1]
    }
    true {
      2: ZXZ [0, 0, 1]
      3: ZXZ [0, 0, 2]
      [0, 0, 1] -> +Z
    }
  }
  [0, 0, 0] -> +Z

  resolve b0 if m
}
";

        let elements = elements_at_blog_cursor(blog, cursor_on_line(blog, "resolve"));

        assert_eq!(
            elements,
            HashSet::from([
                GraphElement::Block(ivec3(0, 0, 1)),
                GraphElement::Block(ivec3(0, 0, 2)),
                GraphElement::Pipe(ivec3(0, 0, 0), ivec3(0, 0, 1)).canonical(),
                GraphElement::Pipe(ivec3(0, 0, 1), ivec3(0, 0, 2)).canonical(),
            ])
        );
    }
}
