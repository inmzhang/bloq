//! Shared command-line entry point for native and Python installations.

use std::path::PathBuf;

use bloq_graph::{Direction, GalleryItem};
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use color_eyre::eyre::{self, WrapErr};

mod commands;
mod terminal;

const ROOT_EXAMPLES: &str = "  bloq input.blog -d 5
  bloq input.blog -d 3,5 -o out.stim
  bloq compile --gallery cnot -d 5 --auto-fill
  bloq compile --gallery cnot -d 5 --fill 0
  bloq view input.blog --gltf
  bloq gallery";

fn root_after_help() -> String {
    format!("{}\n{ROOT_EXAMPLES}", terminal::heading("Examples:"))
}

/// What `compile` emits: a lowered backend circuit (Stim) or the compiled
/// Bloq IR itself in one of its exchange formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum BuiltInBackend {
    Stim,
    /// The human-readable Bloq IR text exchange format (`.bloqir`).
    IrText,
    /// The compact binary Bloq IR exchange format (`.bloq`).
    IrBinary,
}

impl BuiltInBackend {
    pub(crate) const fn file_extension(self) -> &'static str {
        match self {
            Self::Stim => bloq_stim::STIM_FILE_EXTENSION,
            Self::IrText => bloq_ir::BLOQ_TEXT_EXTENSION,
            Self::IrBinary => bloq_ir::BLOQ_BINARY_EXTENSION,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "bloq",
    version,
    about,
    long_about = None,
    arg_required_else_help = true,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true,
    after_help = root_after_help()
)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress status output
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    #[command(flatten)]
    root: CompileCmd,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Args)]
#[group(id = "source", multiple = false, required = true)]
#[command(next_help_heading = "Source")]
struct SourceArgs {
    /// Input BLOG document (`.blog`); view also accepts saved `.bloqir`/`.bloq` IR
    #[arg(value_hint = ValueHint::FilePath)]
    input: Option<PathBuf>,

    /// Use a built-in gallery entry by id
    #[arg(long, value_name = "ID", value_parser = gallery_parser())]
    gallery: Option<GalleryItem>,
}

impl SourceArgs {
    fn resolve(self) -> eyre::Result<commands::compile::CompileSource> {
        match (self.input, self.gallery) {
            (Some(input), None) => Ok(commands::compile::CompileSource::InputFile(input)),
            (None, Some(entry)) => Ok(commands::compile::CompileSource::GalleryEntry(entry)),
            (None, None) => Err(eyre::eyre!(
                "either an input .blog path or --gallery <ID> is required"
            )),
            (Some(_), Some(_)) => unreachable!("clap enforces at most one source"),
        }
    }
}

#[derive(Debug, Args)]
struct CompileCmd {
    #[command(flatten)]
    source: SourceArgs,

    /// Code distance. Repeat or comma-separate to compile multiple distances.
    #[arg(
        short,
        long,
        value_delimiter = ',',
        required = true,
        help_heading = "Compile"
    )]
    distance: Vec<u32>,

    /// Output backend
    #[arg(long, value_enum, default_value = "stim", help_heading = "Compile")]
    backend: BuiltInBackend,

    /// Prepare T blocks with product-state resets and one stabilizer MPP
    /// instead of MSC-LS cultivation.
    #[arg(long, help_heading = "Compile")]
    prepare_t_with_mpps: bool,

    /// Override a compiler resource budget; repeat for multiple fields.
    /// Use 'unlimited' to disable a budget. Fields: https://bloqec.com/docs/dev/api/rust/bloq_graph/struct.ModuleCertificationLimits.html
    #[arg(
        long = "limit",
        value_name = "FIELD=COUNT|unlimited",
        help_heading = "Compile"
    )]
    limits: Vec<commands::compile::LimitOverride>,

    /// Align and merge Clifford circuit moments within each z layer
    #[arg(long, help_heading = "Compile")]
    align_moments: bool,

    /// Compile one random static Clifford path through T and selective blocks
    #[arg(long, help_heading = "Compile")]
    clifford_proxy: bool,

    /// Reproducible random path seed; requires --clifford-proxy
    #[arg(
        long,
        value_name = "SEED",
        requires = "clifford_proxy",
        help_heading = "Compile"
    )]
    proxy_seed: Option<u64>,

    /// Instead of compiling the graph as-authored, close its open ports into
    /// every memory-experiment variant and compile each. Clifford graphs only.
    #[arg(long, help_heading = "Compile")]
    auto_fill: bool,

    /// Like --auto-fill, but compile only the given closed variant
    #[arg(long, value_name = "N", help_heading = "Compile")]
    fill: Option<usize>,

    /// Compile targets to run concurrently. A target is one (distance, fill
    /// variant) pair, so this flag only affects runs producing several targets.
    /// Defaults to the core count capped at
    /// 4, since each worker holds a whole compiled program in memory.
    #[arg(short = 'j', long, value_name = "N", help_heading = "Compile")]
    jobs: Option<std::num::NonZeroUsize>,

    /// Output file path, or base path when multiple outputs are emitted
    #[arg(short, long, value_hint = ValueHint::FilePath, help_heading = "Output")]
    output: Option<PathBuf>,

    /// Print the compiled circuit to stdout instead of writing files
    #[arg(short = 'p', long, conflicts_with = "output", help_heading = "Output")]
    print: bool,
}

#[derive(Debug, Args)]
struct ViewCmd {
    #[command(flatten)]
    source: SourceArgs,

    /// Render saved Bloq IR (.bloqir or .bloq) as an SVG dependency graph
    #[arg(long, conflicts_with_all = ["gltf", "html"], help_heading = "View")]
    svg: bool,

    /// Include classical nodes and their dependencies in the IR SVG
    #[arg(long, help_heading = "View")]
    include_classical: bool,

    /// IR SVG output path (defaults to the input path with a .svg extension)
    #[arg(short, long, value_hint = ValueHint::FilePath, help_heading = "View")]
    output: Option<PathBuf>,

    /// Generate a glTF model next to the input source
    #[arg(long, help_heading = "View")]
    gltf: bool,

    /// Generate a self-contained HTML viewer next to the input source
    #[arg(long, help_heading = "View")]
    html: bool,

    /// Color source geometry by module definition, retaining its hierarchy
    #[arg(long, conflicts_with = "svg", help_heading = "View")]
    module_view: bool,

    /// Visualized pipe length
    #[arg(long, default_value_t = 2.0, help_heading = "View")]
    pipe_length: f32,

    /// Remove outward faces in this lattice direction; may be repeated
    #[arg(
        long = "pop-face",
        value_name = "DIRECTION",
        allow_hyphen_values = true,
        help_heading = "View"
    )]
    pop_faces_at_directions: Vec<Direction>,
}

#[derive(Debug, Args)]
struct ValidateCmd {
    /// Saved Bloq IR program (`.bloqir` text or `.bloq` binary)
    #[arg(value_hint = ValueHint::FilePath)]
    input: PathBuf,
}

#[derive(Debug, Args)]
struct EmitCmd {
    /// Saved Bloq IR program (`.bloqir` text or `.bloq` binary)
    #[arg(value_hint = ValueHint::FilePath)]
    input: PathBuf,

    /// Output backend
    #[arg(long, value_enum, default_value = "stim", help_heading = "Emit")]
    backend: BuiltInBackend,

    /// Check the saved program's well-formedness before emitting
    #[arg(long, help_heading = "Emit")]
    validate: bool,

    /// Align and merge Clifford circuit moments within each z layer
    #[arg(long, help_heading = "Emit")]
    align_moments: bool,

    /// Output file path (defaults to the input path with the backend extension)
    #[arg(short, long, value_hint = ValueHint::FilePath, help_heading = "Output")]
    output: Option<PathBuf>,

    /// Print to stdout instead of writing a file
    #[arg(short = 'p', long, conflicts_with = "output", help_heading = "Output")]
    print: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Compile a block graph to a backend circuit
    Compile(CompileCmd),
    /// Generate block graph glTF/HTML views or a saved IR SVG
    View(ViewCmd),
    /// Emit a backend target from a saved Bloq IR program
    Emit(EmitCmd),
    /// Check a saved Bloq IR program's well-formedness
    Validate(ValidateCmd),
    /// Print structural statistics for a saved Bloq IR program
    Stats(ValidateCmd),
    /// List the supported gallery block graphs
    Gallery,
    /// Generate shell completion scripts
    Completion {
        /// Target shell to generate completions for
        shell: Shell,
    },
}

/// Gallery ids as a typed parser so help and completions list every entry.
fn gallery_parser() -> impl clap::builder::TypedValueParser<Value = GalleryItem> {
    use clap::builder::TypedValueParser;
    clap::builder::PossibleValuesParser::new(GalleryItem::iter().map(GalleryItem::id)).map(|id| {
        id.parse::<GalleryItem>()
            .expect("value validated against Gallery ids")
    })
}

fn build_cli() -> clap::Command {
    Cli::command().styles(terminal::cli_styles())
}

fn run_compile(cmd: CompileCmd, verbose: bool, quiet: bool) -> eyre::Result<()> {
    let source = cmd.source.resolve()?;
    commands::compile::run(
        source,
        cmd.output,
        commands::compile::CompileCommandOptions {
            distances: cmd.distance,
            backend: cmd.backend,
            print: cmd.print,
            prepare_t_with_mpps: cmd.prepare_t_with_mpps,
            limits: commands::compile::resolve_limits(&cmd.limits),
            align_moments: cmd.align_moments,
            clifford_proxy: cmd.clifford_proxy,
            proxy_seed: cmd.proxy_seed,
            auto_fill: cmd.auto_fill,
            fill: cmd.fill,
            jobs: cmd.jobs,
            verbose,
            quiet,
        },
    )
}

fn run_view(cmd: ViewCmd, quiet: bool) -> eyre::Result<()> {
    let ir_input = cmd.source.input.as_ref().is_some_and(|input| {
        matches!(
            input.extension().and_then(|ext| ext.to_str()),
            Some("bloq" | "bloqir")
        )
    });
    if cmd.svg || ir_input {
        eyre::ensure!(
            !cmd.gltf && !cmd.html && !cmd.module_view,
            "saved Bloq IR supports SVG views only"
        );
        let input =
            cmd.source.input.as_ref().ok_or_else(|| {
                eyre::eyre!("SVG visualization requires a saved Bloq IR input file")
            })?;
        return commands::view::run_ir(input, cmd.output.as_deref(), cmd.include_classical, quiet);
    }
    eyre::ensure!(
        !cmd.include_classical && cmd.output.is_none(),
        "--include-classical and --output apply to saved Bloq IR SVG views"
    );
    let source = cmd.source.resolve()?;
    let loaded = source.load_graph()?;
    let graph = if cmd.module_view {
        loaded
    } else {
        loaded.flatten().wrap_err("flatten source for rendering")?
    };
    // With no explicit selection, `bloq view` generates every view kind.
    let (gltf, html) = if cmd.gltf || cmd.html {
        (cmd.gltf, cmd.html)
    } else {
        (true, true)
    };
    commands::view::run(
        &source,
        &graph,
        commands::view::ViewCommandOptions {
            gltf,
            html,
            module_view: cmd.module_view,
            pipe_length: cmd.pipe_length,
            pop_faces_at_directions: cmd.pop_faces_at_directions,
            quiet,
        },
    )
}

fn parse_cli(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Cli, u8> {
    let mut command = build_cli();
    match command.try_get_matches_from_mut(args) {
        Ok(matches) => Ok(Cli::from_arg_matches(&matches).expect("matches validated by clap")),
        Err(err) => {
            if err.kind() == clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand {
                if err.use_stderr() {
                    anstream::eprint!("{}", terminal::help_banner());
                } else {
                    anstream::print!("{}", terminal::help_banner());
                }
            }
            let code = err.exit_code() as u8;
            let _ = err.print();
            Err(code)
        }
    }
}

fn execute(cli: Cli) -> eyre::Result<()> {
    let Cli {
        verbose,
        quiet,
        root,
        command,
    } = cli;

    match command {
        None => run_compile(root, verbose, quiet)?,
        Some(Commands::Compile(cmd)) => run_compile(cmd, verbose, quiet)?,
        Some(Commands::View(cmd)) => run_view(cmd, quiet)?,
        Some(Commands::Emit(cmd)) => commands::emit::run(
            &cmd.input,
            commands::emit::EmitCommandOptions {
                backend: cmd.backend,
                output: cmd.output,
                print: cmd.print,
                validate: cmd.validate,
                align_moments: cmd.align_moments,
                quiet,
            },
        )?,
        Some(Commands::Validate(cmd)) => commands::validate::run(&cmd.input, quiet)?,
        Some(Commands::Stats(cmd)) => commands::stats::run(&cmd.input)?,
        Some(Commands::Gallery) => commands::gallery::run()?,
        Some(Commands::Completion { shell }) => {
            commands::completion::run(shell, build_cli())?;
        }
    }

    Ok(())
}

/// Run the CLI, with `args` a full argv whose first element is the executable
/// name rather than an argument.
///
/// Public API: this is the one entry point behind both the `bloq` binary and
/// the Python package's `bloq` console script. It reports through the process
/// stdout/stderr and returns a shell exit status instead of a `Result` — zero
/// on success, including `--help` and a reader that closed the output pipe,
/// one on command failure, and two on invalid arguments.
pub fn run(args: impl IntoIterator<Item = std::ffi::OsString>) -> u8 {
    // The `eyre` report hook is process-global, so an embedding host may
    // already own it. That decides only how reports are formatted, never
    // whether the command can run, so a refused install is accepted silently.
    static ERROR_HOOK: std::sync::Once = std::sync::Once::new();
    ERROR_HOOK.call_once(|| {
        let _ = color_eyre::install();
    });
    let cli = match parse_cli(args) {
        Ok(cli) => cli,
        Err(code) => return code,
    };
    match execute(cli) {
        Ok(()) => 0,
        Err(err) => {
            if terminal::is_broken_pipe(&err) {
                return 0;
            }
            anstream::eprintln!("{err:?}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let matches = build_cli().try_get_matches_from(args)?;
        Ok(Cli::from_arg_matches(&matches).expect("build cli from matches"))
    }

    #[test]
    fn clap_definition_is_consistent() {
        build_cli().debug_assert();
    }

    /// A host that already owns the process `eyre` hook — or a second call
    /// after the first installed it — must still get the command run.
    #[test]
    fn run_tolerates_an_error_hook_it_does_not_own() {
        let _ = color_eyre::install();
        let gallery = || run(["bloq", "gallery"].map(std::ffi::OsString::from));
        assert_eq!(gallery(), 0);
        assert_eq!(gallery(), 0);
    }

    #[test]
    fn root_shortcut_parses_as_compile() {
        let cli = parse(&["bloq", "foo.blog", "-d", "3"]).expect("parse root compile shortcut");
        assert!(cli.command.is_none());
        assert_eq!(
            cli.root.source.input.as_deref(),
            Some(std::path::Path::new("foo.blog"))
        );
        assert_eq!(cli.root.distance, vec![3]);
    }

    #[test]
    fn explicit_compile_subcommand_parses() {
        let cli = parse(&["bloq", "compile", "foo.blog", "-d", "3,5", "--fill", "1"])
            .expect("parse compile subcommand");
        let Some(Commands::Compile(cmd)) = cli.command else {
            panic!("expected compile subcommand");
        };
        assert_eq!(cmd.distance, vec![3, 5]);
        assert_eq!(cmd.fill, Some(1));
    }

    #[test]
    fn compiler_limit_overrides_are_repeatable_and_checked() {
        let cli = parse(&[
            "bloq",
            "foo.blog",
            "-d",
            "3",
            "--limit",
            "max_boolean_steps=0",
            "--limit",
            "max_boolean_steps=unlimited",
            "--limit",
            "max_frontier_width=8",
        ])
        .expect("compiler budget overrides parse");
        let limits = commands::compile::resolve_limits(&cli.root.limits);
        assert_eq!(limits.max_boolean_steps, usize::MAX);
        assert_eq!(limits.max_frontier_width, 8);
        assert_eq!(
            limits.max_matrix_words,
            bloq_graph::ModuleCertificationLimits::DEFAULT.max_matrix_words
        );

        for value in [
            "unknown=1",
            "max_boolean_steps=-1",
            "max_boolean_steps=9999999999999999999999999999999999",
            "max_boolean_steps",
        ] {
            let error = parse(&["bloq", "compile", "foo.blog", "-d", "3", "--limit", value])
                .expect_err("invalid compiler budget rejected before loading");
            assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
        }
    }

    #[test]
    fn emit_validation_is_explicit() {
        let cli =
            parse(&["bloq", "emit", "program.bloq"]).expect("emit parses without validation flag");
        let Some(Commands::Emit(cmd)) = cli.command else {
            panic!("expected emit subcommand");
        };
        assert!(!cmd.validate);

        let cli = parse(&["bloq", "emit", "program.bloq", "--validate"])
            .expect("emit accepts explicit validation");
        let Some(Commands::Emit(cmd)) = cli.command else {
            panic!("expected emit subcommand");
        };
        assert!(cmd.validate);
    }

    #[test]
    fn clifford_proxy_seed_requires_and_parses_with_proxy_mode() {
        let error = parse(&["bloq", "foo.blog", "-d", "3", "--proxy-seed", "17"])
            .expect_err("a seed without proxy mode is meaningless");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let cli = parse(&[
            "bloq",
            "foo.blog",
            "-d",
            "3",
            "--clifford-proxy",
            "--proxy-seed",
            "17",
        ])
        .expect("seeded proxy parses");
        assert!(cli.root.clifford_proxy);
        assert_eq!(cli.root.proxy_seed, Some(17));
    }

    #[test]
    fn view_subcommand_parses_gallery_source() {
        let cli = parse(&[
            "bloq",
            "view",
            "--gallery",
            "cnot",
            "--gltf",
            "--pop-face=-Y",
        ])
        .expect("parse view command");
        let Some(Commands::View(cmd)) = cli.command else {
            panic!("expected view subcommand");
        };
        assert_eq!(cmd.source.gallery, Some(GalleryItem::CNOT));
        assert!(cmd.gltf);
        assert!(!cmd.html);
        assert_eq!(cmd.pop_faces_at_directions, vec![Direction::YMINUS]);
    }

    #[test]
    fn root_and_subcommand_require_distance_at_parse_time() {
        for args in [
            &["bloq", "foo.blog"][..],
            &["bloq", "compile", "foo.blog"][..],
        ] {
            let error = parse(args).expect_err("compile should require a code distance");
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
            assert_eq!(error.exit_code(), 2);
        }
    }

    #[test]
    fn root_and_subcommand_require_source_at_parse_time() {
        for args in [
            &["bloq", "-d", "3"][..],
            &["bloq", "compile", "-d", "3"][..],
        ] {
            let error = parse(args).expect_err("compile should require a source");
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
            assert_eq!(error.exit_code(), 2);
        }
    }

    #[test]
    fn distance_does_not_consume_the_positional_source() {
        for args in [
            &["bloq", "-d", "3", "foo.blog"][..],
            &["bloq", "compile", "-d", "3", "foo.blog"][..],
        ] {
            let cli = parse(args).expect("distance and source parse independently");
            let cmd = match cli.command {
                Some(Commands::Compile(cmd)) => cmd,
                None => cli.root,
                _ => panic!("expected compile command"),
            };
            assert_eq!(cmd.distance, [3]);
            assert_eq!(
                cmd.source.input.as_deref(),
                Some(std::path::Path::new("foo.blog"))
            );
        }
    }

    #[test]
    fn root_compile_rejects_multiple_sources() {
        let err = parse(&["bloq", "foo.blog", "--gallery", "cnot", "-d", "3"])
            .expect_err("root compile should reject multiple sources");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn print_conflicts_with_output_at_parse_time() {
        let err = parse(&["bloq", "foo.blog", "-d", "3", "-p", "-o", "out.stim"])
            .expect_err("print and output should conflict");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn unknown_gallery_id_lists_possible_values() {
        let err = parse(&["bloq", "--gallery", "bogus", "-d", "3"])
            .expect_err("unknown gallery id should be a parse error");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
        let rendered = err.to_string();
        assert!(rendered.contains("cnot"), "error should list valid ids");
    }

    #[test]
    fn gallery_help_lists_entry_ids() {
        let mut command = build_cli();
        let rendered = command.render_long_help().to_string();
        assert!(rendered.contains("t_with_prepared_y"));
        assert!(rendered.contains("stim"));
    }

    #[test]
    fn gallery_command_rejects_extra_args() {
        parse(&["bloq", "gallery"]).expect("gallery command should parse without extra args");

        let err = parse(&["bloq", "gallery", "cnot"])
            .expect_err("gallery command should reject positional args");
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn bare_bloq_prints_help() {
        let err = parse(&["bloq"]).expect_err("bare bloq should print help instead of parsing");
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        let rendered = err.to_string();
        assert!(rendered.contains("Usage: bloq"));
        assert!(rendered.contains("Commands:"));
        assert!(rendered.contains("--gallery <ID>"));
    }

    #[test]
    fn quiet_conflicts_with_verbose() {
        let err = parse(&["bloq", "foo.blog", "-d", "3", "-q", "-v"])
            .expect_err("quiet and verbose should conflict");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
