//! Parsing and lowering for BLOG graphs and module programs.

pub mod ast;
mod lower;
mod parse;
mod program;

use crate::action::Action;
use crate::{BlockGraph, BlockGraphError, Direction};
use glam::IVec3;
use thiserror::Error;

pub(crate) use parse::{
    cube_height_from_str, is_valid_identifier, is_valid_resource_type, is_valid_simple_identifier,
};

use crate::parser::ast::Span;
use std::path::Path;
use std::sync::Arc;

/// An error from parsing or lowering `.blog` source text.
///
/// Most variants carry a [`Span`] into the source for diagnostic rendering; see
/// [`ParseError::render_diagnostic`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ParseError {
    /// Source text does not satisfy BLOG grammar.
    #[error("syntax error: {message}")]
    Syntax {
        /// Parser diagnostic.
        message: String,
        /// Source range containing the syntax error.
        span: Span,
    },

    /// A statement references an undeclared block ID.
    #[error("undefined block ID {id}")]
    UndefinedId {
        /// Missing block ID.
        id: u32,
        /// Source range containing the reference.
        span: Span,
    },

    /// A block ID is declared more than once.
    #[error("duplicate block ID {id}")]
    DuplicateId {
        /// Repeated block ID.
        id: u32,
        /// Source range containing the duplicate declaration.
        span: Span,
    },

    /// A branch region name is declared more than once.
    #[error("duplicate branch region '{name}'")]
    DuplicateBranchName {
        /// Repeated branch name.
        name: String,
        /// Source range containing the duplicate declaration.
        span: Span,
    },

    /// An action references an undeclared branch region.
    #[error("undefined branch region '{name}'")]
    UndefinedBranchName {
        /// Missing branch name.
        name: String,
        /// Source range containing the reference.
        span: Span,
    },

    /// Two pipe endpoints do not define a supported direction.
    #[error("invalid direction between {from} and {to}: {message}")]
    InvalidDirection {
        /// First endpoint.
        from: IVec3,
        /// Second endpoint.
        to: IVec3,
        /// Source range containing the pipe.
        span: Span,
        /// Direction validation diagnostic.
        message: String,
    },

    /// A walking block has invalid endpoints or boundary data.
    #[error("invalid walking block from {start} to {end}: {message}")]
    InvalidWalkingBlock {
        /// Starting lattice position.
        start: IVec3,
        /// Ending lattice position.
        end: IVec3,
        /// Source range containing the block.
        span: Span,
        /// Walking-block validation diagnostic.
        message: String,
    },

    /// A patch-rotation block has invalid endpoints or basis data.
    #[error("invalid patch rotation block from {start} to {end}: {message}")]
    InvalidPatchRotationBlock {
        /// Starting lattice position.
        start: IVec3,
        /// Ending lattice position.
        end: IVec3,
        /// Source range containing the block.
        span: Span,
        /// Patch-rotation validation diagnostic.
        message: String,
    },

    /// Coordinate arithmetic overflowed during lowering.
    #[error("coordinate arithmetic overflow: {message}")]
    CoordinateOverflow {
        /// Source range whose lowering overflowed.
        span: Span,
        /// Overflow diagnostic.
        message: String,
    },

    /// Spatially connected cubes specify different symbolic heights.
    #[error(
        "conflicting cube heights in one spatial component: `height={height}` at {pos} and \
         `height={other_height}` at {other_pos}; spatially merged cubes share their syndrome rounds, \
         so one component has one height"
    )]
    ConflictingCubeHeights {
        /// Position of the first conflicting cube.
        pos: IVec3,
        /// Height of the first conflicting cube.
        height: crate::CubeHeight,
        /// Position of the second conflicting cube.
        other_pos: IVec3,
        /// Height of the second conflicting cube.
        other_height: crate::CubeHeight,
        /// Source range containing the conflict.
        span: Span,
    },

    /// The source declares a BLOG version this parser does not support.
    #[error("unsupported BLOG version {major}.{minor}")]
    UnsupportedVersion {
        /// Declared major version.
        major: u32,
        /// Declared minor version.
        minor: u32,
        /// Source range containing the version declaration.
        span: Span,
    },

    /// A graph-construction or action-list failure raised while lowering the
    /// parsed AST into a [`BlockGraph`].
    ///
    /// When present, `span` points at the source statement that introduced the
    /// invalid graph element — the block or pipe statement whose lowering the
    /// graph rejected. It is `None` when no single statement is responsible
    /// (e.g. action-list validation).
    #[error("{source}")]
    Graph {
        /// Underlying graph-construction failure.
        source: Box<BlockGraphError>,
        /// Source statement responsible for the failure, when known.
        span: Option<Span>,
    },
}

impl ParseError {
    /// Source span for diagnostic rendering (if available).
    pub fn span(&self) -> Option<Span> {
        match self {
            ParseError::Syntax { span, .. }
            | ParseError::UndefinedId { span, .. }
            | ParseError::DuplicateId { span, .. }
            | ParseError::DuplicateBranchName { span, .. }
            | ParseError::UndefinedBranchName { span, .. }
            | ParseError::InvalidDirection { span, .. }
            | ParseError::InvalidWalkingBlock { span, .. }
            | ParseError::InvalidPatchRotationBlock { span, .. }
            | ParseError::CoordinateOverflow { span, .. }
            | ParseError::ConflictingCubeHeights { span, .. }
            | ParseError::UnsupportedVersion { span, .. } => Some(*span),
            ParseError::Graph { span, .. } => *span,
        }
    }

    /// Render a rustc-style diagnostic using ariadne, with ANSI colors for
    /// terminal display.
    pub fn render_diagnostic(&self, filename: &str, source: &str) -> String {
        self.render(filename, source, true)
    }

    /// Render the same diagnostic as [`render_diagnostic`] but without ANSI
    /// escape sequences, for non-terminal consumers (exception messages,
    /// logs, files).
    ///
    /// [`render_diagnostic`]: Self::render_diagnostic
    pub fn render_diagnostic_plain(&self, filename: &str, source: &str) -> String {
        self.render(filename, source, false)
    }

    fn render(&self, filename: &str, source: &str, color: bool) -> String {
        use ariadne::{Color, Config, IndexType, Label, Report, ReportKind, Source};

        let message = self.to_string();
        let span = self.span();
        let config = Config::default()
            .with_color(color)
            .with_index_type(IndexType::Byte);

        let mut buf = Vec::new();

        let report = match span {
            Some(s) => {
                let range = s.start as usize..s.end as usize;
                Report::build(ReportKind::Error, (filename, range.clone()))
                    .with_config(config)
                    .with_message(&message)
                    .with_label(
                        // The report already carries the full message; the label
                        // only marks the offending location in the source.
                        Label::new((filename, range))
                            .with_message("here")
                            .with_color(Color::Red),
                    )
                    .finish()
            }
            None => Report::build(ReportKind::Error, (filename, 0..0))
                .with_config(config)
                .with_message(&message)
                .finish(),
        };

        report
            .write_for_stdout((filename, Source::from(source)), &mut buf)
            .expect("write to Vec<u8> cannot fail");

        String::from_utf8(buf).expect("ariadne output is valid UTF-8")
    }
}

/// Parse BLOG source text and lower it into a [`BlockGraph`].
///
/// BLOG is an authoring format, so the returned graph may be structurally
/// incomplete. Call [`BlockGraph::validate`] when a complete graph is required.
///
/// # Errors
///
/// Returns a [`ParseError`] if the input is not valid BLOG syntax, references
/// undefined or duplicate block IDs, uses an unsupported version, or cannot be
/// represented as a graph.
///
/// # Examples
///
/// ```
/// let graph = bloq_graph::parse_blog_to_graph(
///     "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n",
/// )?;
/// assert_eq!(graph.block_count(), 1);
/// # Ok::<(), bloq_graph::BlockGraphError>(())
/// ```
pub fn parse_blog_to_graph(input: &str) -> Result<BlockGraph, BlockGraphError> {
    parse_blog_to_graph_with_limits(input, crate::ModuleCertificationLimits::DEFAULT)
}

/// Parses BLOG into a hierarchy-preserving graph with explicit source budgets.
///
/// # Errors
///
/// Returns typed syntax, graph, module, or resource-limit errors.
pub fn parse_blog_to_graph_with_limits(
    input: &str,
    limits: crate::ModuleCertificationLimits,
) -> Result<BlockGraph, BlockGraphError> {
    // The explicit graph constructor accepts executable BLOG and body serialization.
    // Only syntax selects a grammar; a module or resource failure never falls back
    // to a different lowering path.
    match parse::parse_program(input) {
        Ok(source) if source.modules.is_empty() && source.imports.is_empty() => {
            parse_blog_body_with_limits(input, limits).map_err(Into::into)
        }
        Ok(source) => program::lower(&source, limits).map_err(|error| match error {
            crate::ModuleError::Parse(error) => BlockGraphError::Parse(error),
            error => BlockGraphError::ModuleSource(Arc::new(error)),
        }),
        Err(program_error) => parse_blog_body_with_limits(input, limits).map_err(|error| {
            if matches!(error, ParseError::Syntax { .. })
                && program_error.span().map(|span| span.start)
                    >= error.span().map(|span| span.start)
            {
                program_error.into()
            } else {
                error.into()
            }
        }),
    }
}

/// Parse BLOG graph-body text with explicit source-analysis limits.
///
/// # Errors
///
/// Returns a syntax, graph-construction, action-analysis, or resource-limit error.
#[cfg(test)]
pub(super) fn parse_blog_body(input: &str) -> Result<BlockGraph, ParseError> {
    parse_blog_body_with_limits(input, crate::ModuleCertificationLimits::DEFAULT)
}

fn parse_blog_body_with_limits(
    input: &str,
    limits: crate::ModuleCertificationLimits,
) -> Result<BlockGraph, ParseError> {
    let ast = parse::parse(input)?;
    lower::lower_with_inputs_and_limits(&ast, std::iter::empty(), limits)
}

pub(crate) fn load_graph_input(
    path: &Path,
    limits: crate::ModuleCertificationLimits,
) -> Result<BlockGraph, BlockGraphError> {
    let text = std::fs::read_to_string(path).map_err(|source| BlockGraphError::Io {
        path: path.to_path_buf(),
        source: Arc::new(source),
    })?;
    if !parse::parse_program(&text).is_ok_and(|source| !source.imports.is_empty()) {
        return parse_blog_to_graph_with_limits(&text, limits);
    }
    // Preserve the actual I/O cause and path instead of the generic resolver's
    // display-only Load error. Reuse the root text on the first resolver call.
    let mut root_text = Some(text);
    let mut read_error = None;
    let result = program::load(
        path,
        &mut |source: &Path| {
            if let Some(text) = root_text.take() {
                return Ok(text);
            }
            std::fs::read_to_string(source).map_err(|error| {
                let error = Arc::new(error);
                read_error = Some(BlockGraphError::Io {
                    path: source.to_path_buf(),
                    source: error.clone(),
                });
                error
            })
        },
        limits,
    );
    if let Some(error) = read_error {
        return Err(error);
    }
    result.map_err(|error| BlockGraphError::ModuleSource(Arc::new(error)))
}

/// Lowers parsed BLOG while deferring stabilizer-derived action dependencies.
///
/// # Errors
///
/// Returns an error if the AST cannot be represented as a block graph.
pub fn lower_blog_ast_deferred(
    source: &ast::SourceFile,
    inputs: impl IntoIterator<Item = String>,
) -> Result<BlockGraph, ParseError> {
    lower::lower_deferred_with_inputs(source, inputs)
}

/// Lower editor drafts with checked geometry/syntax and lenient actions.
/// Constraint failures remain in [`BlockGraph::action_graph_error`].
///
/// # Errors
///
/// Returns an error if geometry, names, or syntax cannot be lowered safely.
pub fn lower_blog_ast_lenient(
    source: &ast::SourceFile,
    inputs: impl IntoIterator<Item = String>,
) -> Result<BlockGraph, ParseError> {
    lower::lower_lenient_with_inputs(source, inputs)
}

/// Lowers parsed modular BLOG while deferring stabilizer-derived dependencies.
///
/// # Errors
///
/// Returns an error if module structure, interfaces, or graph bodies are invalid.
pub fn lower_blog_graph_ast_deferred(
    source: &ast::ModularSourceFile,
) -> Result<crate::BlockGraph, crate::ModuleError> {
    program::lower_deferred(source)
}

/// Parse a resolved inline BLOG module hierarchy rooted at `module main`.
///
/// Use [`crate::load_graph`] when the source contains imports.
///
/// # Errors
///
/// Returns a parse or module-validation error for invalid input.
#[cfg(test)]
pub(crate) fn parse_inline_graph(input: &str) -> Result<crate::BlockGraph, crate::ModuleError> {
    parse_inline_graph_with_limits(input, crate::ModuleCertificationLimits::DEFAULT)
}

/// Parse an inline BLOG hierarchy with explicit source-analysis and expansion limits.
///
/// # Errors
///
/// Returns a parse, module-validation, or resource-limit error for invalid input.
pub(crate) fn parse_inline_graph_with_limits(
    input: &str,
    limits: crate::ModuleCertificationLimits,
) -> Result<crate::BlockGraph, crate::ModuleError> {
    program::lower(&parse::parse_program(input)?, limits)
}

/// Load BLOG with a caller-provided resolver and explicit resource limits.
///
/// # Errors
///
/// Returns a resolver, parse, import-resolution, module-validation, or resource-limit error.
#[cfg(test)]
pub(crate) fn load_graph_with_resolver_and_limits<E: std::fmt::Display>(
    root: impl AsRef<Path>,
    mut resolver: impl FnMut(&Path) -> Result<String, E>,
    limits: crate::ModuleCertificationLimits,
) -> Result<crate::BlockGraph, crate::ModuleError> {
    program::load(root.as_ref(), &mut resolver, limits)
}

/// Parse BLOG source text into its [`ast::SourceFile`] without lowering.
///
/// # Errors
///
/// Returns [`ParseError::Syntax`] if the input is not valid BLOG syntax.
///
/// # Examples
///
/// ```
/// let ast = bloq_graph::parse_blog_to_ast(
///     "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n",
/// )?;
/// assert_eq!(ast.data_stmts.len(), 1);
/// # Ok::<(), bloq_graph::ParseError>(())
/// ```
pub fn parse_blog_to_ast(input: &str) -> Result<ast::SourceFile, ParseError> {
    parse::parse(input)
}

/// Parse a BLOG module program without lowering it.
///
/// # Errors
///
/// Returns [`ParseError::Syntax`] if the input is not valid modular BLOG.
pub fn parse_blog_program_to_ast(input: &str) -> Result<ast::ModularSourceFile, ParseError> {
    parse::parse_program(input)
}

/// Parse an action snippet and lower it to `Action` values.
///
/// Used by the editor to parse action buffers that don't have the full BLOG
/// file structure. `i2p` resolves a block ID to a position; the
/// `Option<Direction>` argument requests the block anchor (`None`) or the
/// endpoint facing a direction (`Some(dir)`), so measure-edge sources resolve
/// the same way as the file-parse path.
///
/// # Errors
///
/// Returns an error for invalid syntax or an unresolved block reference.
pub fn parse_actions(
    input: &str,
    i2p: impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<Vec<Action>, ParseError> {
    let stmts = parse::parse_actions_only(input)?;
    lower::lower_actions(&stmts, i2p)
}

#[cfg(test)]
mod tests {
    use crate::ModuleCertificationLimits;

    #[test]
    fn explicit_limits_apply_to_imported_source_before_expansion() {
        let limits = ModuleCertificationLimits {
            max_expanded_blocks: 1,
            ..ModuleCertificationLimits::DEFAULT
        };
        let root = "BLOG 1.0\nimport \"child.blog\" as child\nmodule main {\n0: ZXZ [0,0,0]\n}\n";
        let child = "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n1: ZXZ [0,0,1]\n0 -> +Z\n}\n";
        let load = |limits| {
            super::load_graph_with_resolver_and_limits(
                "root.blog",
                |path| match path.to_str() {
                    Some("root.blog") => Ok(root.to_owned()),
                    Some("child.blog") => Ok(child.to_owned()),
                    _ => Err("unexpected import"),
                },
                limits,
            )
        };
        let error = load(limits).expect_err("the imported source exceeds its block budget");
        let message = error.to_string();
        assert!(message.contains("2 > 1"), "{message}");
        load(ModuleCertificationLimits::DEFAULT).expect("default limits load the same hierarchy");
    }

    #[test]
    fn diagnostic_uses_byte_spans_after_unicode() {
        let source = "BLOG 1.0\n# 中文中文中文中文中文\n0: INVALID [0,0,0]\n";
        let error = super::parse_blog_body(source).unwrap_err();
        let plain = error.render_diagnostic_plain("unicode.blog", source);
        assert!(plain.contains("unicode.blog:3:"), "{plain}");
        assert!(plain.contains("INVALID [0,0,0]"), "{plain}");
        assert!(plain.contains("here"), "{plain}");
        assert!(!plain.contains('\x1b'));
    }
}
