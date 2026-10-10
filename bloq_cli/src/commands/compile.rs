use std::io::IsTerminal as _;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use bloq_compile::{CompileConfig, CompileContext, SharedCompileCache};
use bloq_graph::{BlockGraph, GalleryItem, ModuleCertificationLimits};
use clap::ValueEnum;
use color_eyre::eyre::{self, WrapErr};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{BuiltInBackend, terminal};

use super::{EmittedArtifact, atomic_write};

#[derive(Debug)]
pub(crate) enum CompileSource {
    InputFile(PathBuf),
    GalleryEntry(GalleryItem),
}

impl CompileSource {
    fn load_graph_with_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> eyre::Result<BlockGraph> {
        match self {
            Self::InputFile(path) => BlockGraph::load_with_limits(path, limits)
                .map_err(|error| compile_error_report(error.into()))
                .wrap_err_with(|| format!("load {}", path.display())),
            Self::GalleryEntry(entry) => entry
                .build_with_limits(limits)
                .map_err(|error| compile_error_report(error.into())),
        }
    }

    pub(crate) fn load_graph(&self) -> eyre::Result<BlockGraph> {
        self.load_graph_with_limits(ModuleCertificationLimits::DEFAULT)
    }

    fn display_name(&self) -> String {
        match self {
            Self::InputFile(path) => path.display().to_string(),
            Self::GalleryEntry(entry) => format!("gallery:{}", entry.id()),
        }
    }

    fn output_stem(&self) -> String {
        match self {
            Self::InputFile(path) => path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("output")
                .to_string(),
            Self::GalleryEntry(entry) => entry.id().to_string(),
        }
    }

    fn output_parent(&self) -> &Path {
        match self {
            Self::InputFile(path) => path.parent().unwrap_or_else(|| Path::new(".")),
            Self::GalleryEntry(_) => Path::new("."),
        }
    }

    pub(crate) fn default_output_path_with_extension(&self, extension: &str) -> PathBuf {
        self.output_parent()
            .join(format!("{}.{}", self.output_stem(), extension))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LimitOverride {
    field: String,
    value: usize,
}

impl std::str::FromStr for LimitOverride {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (field, count) = input
            .split_once('=')
            .ok_or_else(|| "expected FIELD=COUNT or FIELD=unlimited".to_string())?;
        let value = if count == "unlimited" {
            usize::MAX
        } else {
            count.parse::<usize>().map_err(|_| {
                format!(
                    "limit '{field}' requires a non-negative count fitting usize or 'unlimited'"
                )
            })?
        };
        let mut limits = ModuleCertificationLimits::DEFAULT;
        limits
            .set(field, value)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            field: field.to_string(),
            value,
        })
    }
}

pub(crate) fn resolve_limits(overrides: &[LimitOverride]) -> ModuleCertificationLimits {
    let mut limits = ModuleCertificationLimits::DEFAULT;
    for limit in overrides {
        limits
            .set(&limit.field, limit.value)
            .expect("limit names were checked by clap");
    }
    limits
}

fn compile_error_report(error: bloq_compile::CompileError) -> eyre::Report {
    let resource_limited = error.resource_limit_help().is_some();
    let report = eyre::Report::new(error);
    if resource_limited {
        report.wrap_err(
            "adjust the matching compiler budget with --limit FIELD=COUNT or \
             --limit FIELD=unlimited (fields: https://bloqec.com/docs/dev/api/rust/bloq_graph/struct.ModuleCertificationLimits.html)",
        )
    } else {
        report
    }
}

#[derive(Debug)]
pub(crate) struct CompileVariant {
    pub(crate) fill_variant: Option<usize>,
    source: BlockGraph,
}

#[derive(Debug, Clone, Copy)]
struct OutputPathShape {
    multiple_variants: bool,
    multiple_distances: bool,
}

#[derive(Debug, Clone, Copy)]
struct OutputDescriptor {
    distance: u32,
    fill_variant: Option<usize>,
}

impl OutputDescriptor {
    /// Human label like `d=3` or `d=3, fill 1` for status and error lines.
    fn display_label(self) -> String {
        match self.fill_variant {
            Some(variant) => format!("d={}, fill {variant}", self.distance),
            None => format!("d={}", self.distance),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CompileCommandOptions {
    pub(crate) distances: Vec<u32>,
    pub(crate) backend: BuiltInBackend,
    pub(crate) print: bool,
    pub(crate) prepare_t_with_mpps: bool,
    pub(crate) limits: ModuleCertificationLimits,
    pub(crate) align_moments: bool,
    pub(crate) clifford_proxy: bool,
    pub(crate) proxy_seed: Option<u64>,
    pub(crate) auto_fill: bool,
    pub(crate) fill: Option<usize>,
    pub(crate) jobs: Option<NonZeroUsize>,
    pub(crate) verbose: bool,
    pub(crate) quiet: bool,
}

pub(crate) fn run(
    source: CompileSource,
    output: Option<PathBuf>,
    options: CompileCommandOptions,
) -> eyre::Result<()> {
    if options.align_moments && options.backend != BuiltInBackend::Stim {
        eyre::bail!("--align-moments requires --backend stim");
    }
    let started = std::time::Instant::now();
    let quiet = options.quiet;
    let progress = CompileProgress::new(!options.quiet);
    let loading = progress.begin("input", "Loading source");
    let graph = source.load_graph_with_limits(options.limits)?;
    drop(loading);
    run_loaded(&source, &graph, output.as_deref(), options, &progress)?;
    if !quiet {
        terminal::status(
            "Finished",
            format!("compilation in {:.2}s", started.elapsed().as_secs_f64()),
        );
    }
    Ok(())
}

fn run_loaded(
    source: &CompileSource,
    graph: &BlockGraph,
    output: Option<&Path>,
    options: CompileCommandOptions,
    progress: &CompileProgress,
) -> eyre::Result<()> {
    let CompileCommandOptions {
        distances,
        backend,
        print,
        prepare_t_with_mpps,
        limits,
        align_moments,
        clifford_proxy,
        proxy_seed,
        auto_fill,
        fill,
        jobs,
        verbose,
        quiet,
    } = options;

    let clifford_proxy_seed = clifford_proxy.then(|| proxy_seed.unwrap_or_else(random_proxy_seed));
    if !quiet && let Some(seed) = clifford_proxy_seed {
        terminal::note(format!("Clifford proxy seed: {seed}"));
    }

    let variants = if auto_fill || fill.is_some() {
        let preparing = progress.begin("input", "Preparing fill variants");
        let flat = graph
            .flatten_with_limits(limits)
            .wrap_err("flatten source for port filling")?;
        let variants = prepare_compile_variants(&flat, fill, limits)?;
        drop(preparing);
        variants
    } else {
        vec![CompileVariant {
            fill_variant: None,
            source: graph.clone(),
        }]
    };
    if !quiet && variants.len() > 1 {
        terminal::note(format!(
            "input has open ports; auto-filled into {} closed variants \
             (use --fill <N> to select one)",
            variants.len()
        ));
    }
    let artifact_count = variants.len() * distances.len();
    if print
        && matches!(backend, BuiltInBackend::IrText | BuiltInBackend::IrBinary)
        && artifact_count > 1
    {
        eyre::bail!(
            "--print with the {} backend emits a single artifact, but this run \
             produces {artifact_count} (pick one --distance and, for open inputs, one --fill)",
            backend
                .to_possible_value()
                .expect("built-in backend name")
                .get_name()
        );
    }

    let ctx = CompileRunContext {
        source,
        output,
        backend,
        output_shape: OutputPathShape {
            multiple_variants: variants.len() > 1,
            multiple_distances: distances.len() > 1,
        },
        print,
        prepare_t_with_mpps,
        limits,
        align_moments,
        clifford_proxy_seed,
        jobs,
        verbose,
        quiet,
        progress,
    };
    log_run_configuration(&ctx, &distances);
    reject_input_output_aliases(&ctx, &variants, &distances)?;
    compile_requested_variants(&ctx, &variants, &distances)
}

#[derive(Debug)]
struct CompileRunContext<'a> {
    source: &'a CompileSource,
    output: Option<&'a Path>,
    backend: BuiltInBackend,
    output_shape: OutputPathShape,
    print: bool,
    prepare_t_with_mpps: bool,
    limits: ModuleCertificationLimits,
    align_moments: bool,
    clifford_proxy_seed: Option<u64>,
    jobs: Option<NonZeroUsize>,
    verbose: bool,
    quiet: bool,
    progress: &'a CompileProgress,
}

/// Terminal animation is opt-in by terminal capability. Piped runs still get
/// one readable line per stage, and quiet runs emit no progress at all.
#[derive(Debug)]
struct CompileProgress {
    bars: Option<MultiProgress>,
    plain: bool,
}

impl CompileProgress {
    fn new(enabled: bool) -> Self {
        let dynamic = enabled
            && std::io::stderr().is_terminal()
            && !std::env::var("TERM").is_ok_and(|term| term == "dumb")
            && std::env::var_os("NO_COLOR").is_none();
        Self {
            bars: dynamic.then(MultiProgress::new),
            plain: enabled && !dynamic,
        }
    }

    fn begin(&self, label: &str, message: &str) -> ProgressTask {
        let sink = if let Some(bars) = &self.bars {
            let bar = bars.add(ProgressBar::new_spinner());
            bar.set_prefix(label.to_owned());
            bar.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {elapsed_precise} {prefix}: {msg}")
                    .expect("valid static progress template")
                    .tick_strings(&["◐", "◓", "◑", "◒"]),
            );
            bar.enable_steady_tick(std::time::Duration::from_millis(100));
            ProgressSink::Bar(bar)
        } else if self.plain {
            ProgressSink::Plain(label.to_owned())
        } else {
            ProgressSink::Quiet
        };
        let task = ProgressTask { sink };
        task.stage(message);
        task
    }
}

#[derive(Clone)]
enum ProgressSink {
    Bar(ProgressBar),
    Plain(String),
    Quiet,
}

impl ProgressSink {
    fn stage(&self, message: &str) {
        match self {
            Self::Bar(bar) => bar.set_message(message.to_owned()),
            Self::Plain(label) => terminal::note(format!("{label}: {message}")),
            Self::Quiet => {}
        }
    }
}

struct ProgressTask {
    sink: ProgressSink,
}

impl ProgressTask {
    fn stage(&self, message: &str) {
        self.sink.stage(message);
    }
}

impl Drop for ProgressTask {
    fn drop(&mut self) {
        if let ProgressSink::Bar(bar) = &self.sink {
            bar.finish_and_clear();
        }
    }
}

fn random_proxy_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    (nanos as u64) ^ ((nanos >> 64) as u64) ^ u64::from(std::process::id())
}

fn log_run_configuration(ctx: &CompileRunContext<'_>, distances: &[u32]) {
    if !ctx.verbose {
        return;
    }

    terminal::note(format!(
        "loaded block graph from {}",
        ctx.source.display_name()
    ));
    terminal::note(format!(
        "code distances: {}",
        distances
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    terminal::note(format!(
        "backend: {}",
        ctx.backend
            .to_possible_value()
            .expect("built-in backends have clap names")
            .get_name()
    ));
    terminal::note(format!(
        "output mode: {}",
        if ctx.print { "print" } else { "write-files" }
    ));
    terminal::note(format!(
        "concurrent compile targets: {}",
        compile_batch_size(ctx.jobs)
    ));
}

/// One compiled target: the bytes to emit, plus the verbose lines that
/// reporting prints ahead of them.
///
/// Compilation happens off the reporting thread, so its notes are collected
/// rather than printed where they are produced.
#[derive(Debug)]
struct CompiledTarget {
    emitted: EmittedArtifact,
    notes: Vec<String>,
    /// Compiler advisories for this target. Every target of a run tends to
    /// raise the same one (they share a graph shape), so reporting dedupes.
    warnings: Vec<&'static str>,
}

fn compile_requested_variants(
    ctx: &CompileRunContext<'_>,
    variants: &[CompileVariant],
    distances: &[u32],
) -> eyre::Result<()> {
    let single_output = variants.len() == 1 && distances.len() == 1;
    let targets: Vec<(&CompileVariant, u32)> = variants
        .iter()
        .flat_map(|variant| distances.iter().map(move |&distance| (variant, distance)))
        .collect();

    // Targets compile independently. One shared cache across
    // the run pays for each distinct circuit template once: the
    // (variant × distance) grid repeats block signatures heavily.
    let cache = SharedCompileCache::new();
    let batch_size = compile_batch_size(ctx.jobs);
    let mut failed = 0usize;
    let mut reported_warnings = std::collections::HashSet::new();
    // Batched rather than compile-everything-then-report: peak memory stays
    // proportional to the batch size instead of the target count, and a
    // broken pipe still stops the run before compiling what is left.
    // Reporting stays sequential and in target order, so output ordering,
    // interleaved error lines, and broken-pipe handling match a serial run.
    for batch in targets.chunks(batch_size) {
        for (&(variant, distance), compiled) in batch.iter().zip(compile_batch(ctx, &cache, batch))
        {
            let descriptor = OutputDescriptor {
                distance,
                fill_variant: variant.fill_variant,
            };
            let reported = compiled.and_then(|target| {
                if !ctx.quiet {
                    for warning in &target.warnings {
                        if reported_warnings.insert(*warning) {
                            terminal::warning(*warning);
                        }
                    }
                }
                report_compiled_target(ctx, descriptor, target, single_output)
            });
            if let Err(error) = reported {
                if terminal::is_broken_pipe(&error) {
                    return Err(error);
                }
                failed += 1;
                terminal::error(format!(
                    "{} ({}): {error:#}",
                    ctx.source.display_name(),
                    descriptor.display_label()
                ));
            }
        }
    }

    if failed == 0 {
        Ok(())
    } else {
        eyre::bail!("{failed} of {} compile targets failed", targets.len());
    }
}

/// How many targets compile at once.
///
/// Each worker retains a compiled program and its emitted artifact until
/// reporting, so the default caps memory use at four concurrent targets.
/// `--jobs` overrides this target budget; module compilation inside each target
/// retains its own worker policy.
fn compile_batch_size(jobs: Option<NonZeroUsize>) -> usize {
    const MAX_CONCURRENT_COMPILES: usize = 4;

    match jobs {
        Some(jobs) => jobs.get(),
        None => std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .min(MAX_CONCURRENT_COMPILES),
    }
}

/// Compile one batch of targets concurrently, returning their results in batch
/// order. The batch is never longer than the thread budget, so each target gets
/// its own worker.
fn compile_batch(
    ctx: &CompileRunContext<'_>,
    cache: &SharedCompileCache,
    batch: &[(&CompileVariant, u32)],
) -> Vec<eyre::Result<CompiledTarget>> {
    if let [(variant, distance)] = batch {
        return vec![compile_target(ctx, cache, variant, *distance)];
    }
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(batch.len());
        for &(variant, distance) in batch {
            workers.push(scope.spawn(move || compile_target(ctx, cache, variant, distance)));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().expect("a compile worker panicked"))
            .collect()
    })
}

fn compile_target(
    ctx: &CompileRunContext<'_>,
    cache: &SharedCompileCache,
    variant: &CompileVariant,
    distance: u32,
) -> eyre::Result<CompiledTarget> {
    let descriptor = OutputDescriptor {
        distance,
        fill_variant: variant.fill_variant,
    };
    let progress = ctx.progress.begin(&descriptor.display_label(), "Compiling");
    if ctx.clifford_proxy_seed.is_some() {
        progress.stage("Compiling Clifford proxy");
    }
    let config = CompileConfig::try_new(distance)
        .wrap_err("build the compile config")?
        .with_prepare_t_with_mpps(ctx.prepare_t_with_mpps)
        .with_certification_limits(ctx.limits);

    let artifacts = match ctx.clifford_proxy_seed {
        Some(seed) => bloq_compile::compile_random_clifford_proxy(config, &variant.source, seed),
        None => {
            let sink = progress.sink.clone();
            CompileContext::with_shared_cache(config, cache)
                .with_progress_observer(move |stage| sink.stage(&stage.to_string()))
                .compile(&variant.source)
        }
    }
    .map_err(compile_error_report)
    .wrap_err("compile block graph")?;

    progress.stage("Emitting output");
    let mut notes = Vec::new();
    if ctx.verbose {
        let prefix = descriptor.display_label();
        notes.push(format!(
            "{prefix}: {} Bloq nodes",
            artifacts.bloq.node_count()
        ));
        notes.push(format!(
            "{prefix}: {} qubits",
            artifacts
                .bloq
                .qubit_count()
                .wrap_err("build Bloq qubit layout")?
        ));
        notes.push(format!(
            "{prefix}: {} static measurement sites",
            artifacts.bloq.measurement_count()
        ));
    }

    let emitted = ctx.backend.emit(
        &artifacts.bloq,
        &bloq_stim::BloqStimOptions::new().with_align_moments(ctx.align_moments),
    )?;
    Ok(CompiledTarget {
        emitted,
        notes,
        warnings: artifacts.warnings,
    })
}

fn report_compiled_target(
    ctx: &CompileRunContext<'_>,
    descriptor: OutputDescriptor,
    target: CompiledTarget,
    single_output: bool,
) -> eyre::Result<()> {
    let CompiledTarget { emitted, notes, .. } = target;
    for note in notes {
        terminal::note(note);
    }

    if ctx.print {
        print_rendered_output(ctx, descriptor, &emitted, single_output)?;
    } else {
        let path = resolve_output_path(ctx, descriptor);
        atomic_write(&path, emitted.bytes())
            .wrap_err_with(|| format!("write output file {}", path.display()))?;
        if !ctx.quiet {
            terminal::status(
                "Compiled",
                format!(
                    "{} ({}) -> {}",
                    ctx.source.display_name(),
                    descriptor.display_label(),
                    path.display()
                ),
            );
        }
    }

    Ok(())
}

fn prepare_compile_variants(
    graph: &BlockGraph,
    fill: Option<usize>,
    limits: ModuleCertificationLimits,
) -> eyre::Result<Vec<CompileVariant>> {
    if !graph.is_open() {
        eyre::bail!("--auto-fill and --fill only apply to inputs with open ports");
    }

    if !graph.is_rigid() {
        eyre::bail!(
            "--auto-fill cannot close a non-Clifford graph: it contains selective \
             or T blocks; compile it without --auto-fill, or close the ports in the source"
        );
    }

    let filled = graph
        .filled_graphs_with_limits(limits)
        .map_err(|error| compile_error_report(error.into()))
        .wrap_err("fill open ports for compilation")?;
    if filled.is_empty() {
        eyre::bail!("fill open ports for compilation produced no closed variants");
    }

    let mut variants = filled
        .into_iter()
        .enumerate()
        .map(|(variant, filled_graph)| CompileVariant {
            fill_variant: Some(variant),
            source: filled_graph,
        })
        .collect::<Vec<_>>();

    if let Some(selected) = fill {
        if selected >= variants.len() {
            eyre::bail!(
                "--fill {selected} is out of range: input has {} variants (0..={})",
                variants.len(),
                variants.len() - 1
            );
        }
        variants = vec![variants.swap_remove(selected)];
    }
    Ok(variants)
}

fn resolve_output_path(ctx: &CompileRunContext, descriptor: OutputDescriptor) -> PathBuf {
    match ctx.output {
        Some(path) => output_path_for_requested_output(path, descriptor, ctx.output_shape),
        None => default_output_path(ctx.source, ctx.backend, descriptor),
    }
}

fn reject_input_output_aliases(
    ctx: &CompileRunContext,
    variants: &[CompileVariant],
    distances: &[u32],
) -> eyre::Result<()> {
    let CompileSource::InputFile(input) = ctx.source else {
        return Ok(());
    };
    if ctx.print {
        return Ok(());
    }

    for variant in variants {
        for &distance in distances {
            let output = resolve_output_path(
                ctx,
                OutputDescriptor {
                    distance,
                    fill_variant: variant.fill_variant,
                },
            );
            super::reject_input_output_alias(input, &output)?;
        }
    }
    Ok(())
}

fn output_path_for_requested_output(
    path: &Path,
    descriptor: OutputDescriptor,
    output_shape: OutputPathShape,
) -> PathBuf {
    let mut suffix = String::new();
    if output_shape.multiple_distances {
        suffix.push_str(&format!("-d{}", descriptor.distance));
    }
    if output_shape.multiple_variants
        && let Some(variant) = descriptor.fill_variant
    {
        suffix.push_str(&format!("-fill{variant}"));
    }

    if suffix.is_empty() {
        return path.to_path_buf();
    }
    append_stem_suffix(path, &suffix)
}

fn append_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(file_name) = path.file_name() else {
        return path.join(suffix.trim_start_matches('-'));
    };
    let file_name = file_name.to_string_lossy();

    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or(&file_name);
            path.with_file_name(format!("{stem}{suffix}.{ext}"))
        }
        None => path.with_file_name(format!("{file_name}{suffix}")),
    }
}

fn default_output_path(
    source: &CompileSource,
    backend: BuiltInBackend,
    descriptor: OutputDescriptor,
) -> PathBuf {
    let mut stem = format!("{}-d{}", source.output_stem(), descriptor.distance);
    if let Some(variant) = descriptor.fill_variant {
        stem.push_str(&format!("-fill{variant}"));
    }
    source
        .output_parent()
        .join(format!("{stem}.{}", backend.file_extension()))
}

fn print_rendered_output(
    ctx: &CompileRunContext,
    descriptor: OutputDescriptor,
    emitted: &EmittedArtifact,
    single_output: bool,
) -> eyre::Result<()> {
    if !single_output && !ctx.quiet {
        terminal::note(format!(
            "printing {} ({})",
            ctx.source.display_name(),
            descriptor.display_label()
        ));
    }
    emitted.write_stdout()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn options(distances: Vec<u32>) -> CompileCommandOptions {
        CompileCommandOptions {
            distances,
            backend: BuiltInBackend::Stim,
            print: false,
            auto_fill: false,
            prepare_t_with_mpps: false,
            limits: ModuleCertificationLimits::DEFAULT,
            align_moments: false,
            clifford_proxy: false,
            proxy_seed: None,
            fill: None,
            jobs: None,
            verbose: false,
            quiet: true,
        }
    }

    fn write_single_cube_blog(root: &Path) -> PathBuf {
        let input_path = root.join("input.blog");
        std::fs::write(
            &input_path,
            "BLOG 1.0\n\nmodule main {\n  0: ZXZ [0, 0, 0]\n}\n",
        )
        .expect("write fixture .blog");
        input_path
    }

    fn write_gallery_blog(root: &Path, name: &str, gallery: GalleryItem) -> PathBuf {
        let input_path = root.join(format!("{name}.blog"));
        std::fs::write(&input_path, gallery.entry().blog()).expect("write gallery fixture .blog");
        input_path
    }

    #[test]
    fn ir_exchange_backends_write_parseable_artifacts() {
        let dir = tempdir().expect("create tempdir");
        let input = write_single_cube_blog(dir.path());

        for (backend, extension) in [
            (BuiltInBackend::IrText, "bloqir"),
            (BuiltInBackend::IrBinary, "bloq"),
        ] {
            let mut opts = options(vec![3]);
            opts.backend = backend;
            run(CompileSource::InputFile(input.clone()), None, opts)
                .expect("compile with IR backend");
            let path = dir.path().join(format!("input-d3.{extension}"));
            let bytes = std::fs::read(&path).expect("read emitted artifact");
            let restored = match backend {
                BuiltInBackend::IrText => bloq_ir::Bloq::from_text(
                    std::str::from_utf8(&bytes).expect("text artifact is UTF-8"),
                )
                .expect("text artifact parses"),
                BuiltInBackend::IrBinary => {
                    bloq_ir::Bloq::from_binary(&bytes).expect("binary artifact decodes")
                }
                BuiltInBackend::Stim => unreachable!("not under test"),
            };
            restored.validate().expect("restored program validates");
        }
    }

    #[test]
    fn print_mode_rejects_multiple_ir_artifacts() {
        let dir = tempdir().expect("create tempdir");
        let input = write_single_cube_blog(dir.path());
        for backend in [BuiltInBackend::IrBinary, BuiltInBackend::IrText] {
            let mut opts = options(vec![3, 5]);
            opts.backend = backend;
            opts.print = true;
            let error = run(CompileSource::InputFile(input.clone()), None, opts)
                .expect_err("two IR artifacts cannot share stdout");
            assert!(error.to_string().contains("single artifact"));
        }
    }

    #[test]
    fn align_moments_rejects_ir_backend_before_loading() {
        let mut opts = options(vec![3]);
        opts.backend = BuiltInBackend::IrText;
        opts.align_moments = true;

        let error = run(CompileSource::InputFile("missing.blog".into()), None, opts)
            .expect_err("moment alignment only changes Stim emission");

        assert!(error.to_string().contains("requires --backend stim"));
    }

    #[test]
    fn compiler_limits_apply_to_source_loading_and_allow_retry() {
        let dir = tempdir().expect("create tempdir");
        let input = write_single_cube_blog(dir.path());
        let output = dir.path().join("program.bloqir");
        let mut opts = options(vec![3]);
        opts.backend = BuiltInBackend::IrText;
        opts.limits.max_expanded_blocks = 0;
        let error = run(
            CompileSource::InputFile(input.clone()),
            Some(output.clone()),
            opts,
        )
        .expect_err("an explicit zero budget is enforced while loading");
        let message = format!("{error:#}");
        assert!(message.contains("> 0"), "{message}");
        assert!(message.contains("--limit FIELD=unlimited"), "{message}");
        assert!(!output.exists());

        let mut opts = options(vec![3]);
        opts.backend = BuiltInBackend::IrText;
        opts.limits.max_expanded_blocks = usize::MAX;
        run(CompileSource::InputFile(input), Some(output.clone()), opts)
            .expect("raising only the exhausted budget permits compilation");
        assert!(output.exists());
    }

    #[test]
    fn graph_source_retains_hierarchy_through_loading_and_compilation() {
        let dir = tempdir().expect("create tempdir");
        let input = dir.path().join("hierarchy.blog");
        std::fs::write(
            &input,
            "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\n\
             module main {\nfirst: Leaf @ [0,0,0]\nsecond: Leaf @ [2,0,0]\n}\n",
        )
        .unwrap();
        let source = CompileSource::InputFile(input);
        let graph = source.load_graph().expect("load hierarchical graph");
        assert!(graph.has_module_structure());
        assert_eq!(graph.modules().count(), 2);
        assert_eq!(graph.instances.len(), 2);
        assert_eq!(graph.block_count(), 0);
        assert_eq!(graph.flatten().unwrap().block_count(), 2);

        let output = dir.path().join("hierarchy.bloqir");
        let mut opts = options(vec![3]);
        opts.backend = BuiltInBackend::IrText;
        run(source, Some(output.clone()), opts).expect("compile the same graph input");
        bloq_ir::Bloq::from_text(&std::fs::read_to_string(output).unwrap())
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn seeded_clifford_proxy_preserves_dynamic_module_sources() {
        for (gallery, align) in [
            (GalleryItem::T, true),
            (GalleryItem::PhaseGradientK4, false),
        ] {
            let dir = tempdir().expect("create tempdir");
            let input = write_gallery_blog(dir.path(), "source", gallery);
            let output = dir.path().join("proxy.stim");
            let mut opts = options(vec![3]);
            opts.align_moments = align;
            opts.clifford_proxy = true;
            opts.proxy_seed = Some(17);

            run(CompileSource::InputFile(input), Some(output.clone()), opts)
                .expect("dynamic module emits through its static proxy");

            let text = std::fs::read_to_string(output).expect("read emitted Stim");
            assert!(text.contains("TICK"));
        }
    }

    #[test]
    fn atomic_output_replaces_existing_file() {
        let temp = tempdir().expect("create temp dir");
        let output = temp.path().join("artifact.stim");
        std::fs::write(&output, b"old").expect("write old output");

        atomic_write(&output, b"complete new output").expect("replace output atomically");

        assert_eq!(std::fs::read(&output).unwrap(), b"complete new output");
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
    }

    #[test]
    fn failed_atomic_outputs_preserve_targets_and_remove_partial_files() {
        let temp = tempdir().expect("create temp dir");
        let output = temp.path().join("artifact.stim");
        std::fs::create_dir(&output).expect("create conflicting output directory");
        let marker = output.join("old");
        std::fs::write(&marker, b"old").expect("write old marker");

        atomic_write(&output, b"new").expect_err("a file cannot replace a non-empty directory");

        assert!(output.is_dir());
        assert_eq!(std::fs::read(marker).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);

        let regular = temp.path().join("regular.stim");
        std::fs::write(&regular, b"old").unwrap();
        let error = super::super::atomic_write_with(&regular, |partial| {
            std::fs::write(partial, b"partial new output")?;
            Err(std::io::Error::other("write interrupted"))
        })
        .expect_err("a partial write must not replace the old output");
        assert_eq!(error.to_string(), "write interrupted");
        assert_eq!(std::fs::read(regular).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn new_atomic_output_uses_normal_creation_permissions() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp = tempdir().expect("create temp dir");
        let ordinary = temp.path().join("ordinary.stim");
        let atomic = temp.path().join("atomic.stim");
        std::fs::write(&ordinary, b"ordinary").expect("create ordinary output");

        atomic_write(&atomic, b"atomic").expect("create atomic output");

        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&atomic), mode(&ordinary));
    }

    #[cfg(unix)]
    #[test]
    fn atomic_output_replacement_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp = tempdir().expect("create temp dir");
        let output = temp.path().join("artifact.stim");
        std::fs::write(&output, b"old").expect("write old output");
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o640))
            .expect("set old permissions");

        atomic_write(&output, b"complete new output").expect("replace output atomically");

        assert_eq!(std::fs::read(&output).unwrap(), b"complete new output");
        assert_eq!(
            std::fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_output_does_not_bypass_target_write_protection() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp = tempdir().expect("create temp dir");
        let output = temp.path().join("artifact.stim");
        std::fs::write(&output, b"old").expect("write old output");
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o444))
            .expect("make output read-only");

        // Elevated users may legitimately bypass mode-bit write protection.
        if std::fs::OpenOptions::new()
            .write(true)
            .open(&output)
            .is_ok()
        {
            return;
        }

        atomic_write(&output, b"new").expect_err("read-only output must reject replacement");

        assert_eq!(std::fs::read(&output).unwrap(), b"old");
    }

    #[cfg(unix)]
    #[test]
    fn atomic_output_rejects_symbolic_links() {
        use std::os::unix::fs::symlink;

        let temp = tempdir().expect("create temp dir");
        let target = temp.path().join("target.stim");
        let output = temp.path().join("artifact.stim");
        std::fs::write(&target, b"old").expect("write link target");
        symlink(&target, &output).expect("create output symlink");

        let error = atomic_write(&output, b"new").expect_err("symlink output must be rejected");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            std::fs::symlink_metadata(&output)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"old");
    }

    #[test]
    fn default_output_path_for_open_fill_variant_includes_fill_suffix() {
        let path = default_output_path(
            &CompileSource::GalleryEntry(GalleryItem::CNOT),
            BuiltInBackend::Stim,
            OutputDescriptor {
                distance: 3,
                fill_variant: Some(1),
            },
        );
        assert_eq!(path, PathBuf::from("./cnot-d3-fill1.stim"));
    }

    #[test]
    fn default_output_path_uses_input_stem_and_distance() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());

        run(CompileSource::InputFile(input), None, options(vec![3]))
            .expect("default output path should compile");

        let output_path = temp.path().join("input-d3.stim");
        assert!(output_path.exists(), "default output file should exist");
    }

    #[test]
    fn equivalent_output_path_cannot_overwrite_input() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());
        let child = temp.path().join("child");
        std::fs::create_dir(&child).expect("create child dir");
        let output = child.join("..").join("input.blog");
        let original = std::fs::read(&input).expect("read input");

        let error = run(
            CompileSource::InputFile(input.clone()),
            Some(output),
            options(vec![3]),
        )
        .expect_err("output must not overwrite input");

        assert!(error.to_string().contains("aliases the input"));
        assert_eq!(
            std::fs::read(input).expect("read preserved input"),
            original
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_output_cannot_overwrite_input() {
        use std::os::unix::fs::symlink;

        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());
        let output = temp.path().join("output.stim");
        symlink(&input, &output).expect("create output symlink");
        let original = std::fs::read(&input).expect("read input");

        let error = run(
            CompileSource::InputFile(input.clone()),
            Some(output),
            options(vec![3]),
        )
        .expect_err("symlink output must not overwrite input");

        assert!(error.to_string().contains("aliases the input"));
        assert_eq!(
            std::fs::read(input).expect("read preserved input"),
            original
        );
    }

    #[cfg(unix)]
    #[test]
    fn hard_link_output_cannot_overwrite_input() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());
        let output = temp.path().join("output.stim");
        std::fs::hard_link(&input, &output).expect("create output hard link");
        let original = std::fs::read(&input).expect("read input");

        let error = run(
            CompileSource::InputFile(input.clone()),
            Some(output),
            options(vec![3]),
        )
        .expect_err("hard-link output must not overwrite input");

        assert!(error.to_string().contains("aliases the input"));
        assert_eq!(
            std::fs::read(input).expect("read preserved input"),
            original
        );
    }

    #[test]
    fn output_alias_preflight_prevents_earlier_writes() {
        let temp = tempdir().expect("create temp dir");
        let fixture = write_single_cube_blog(temp.path());
        let input = temp.path().join("out-d5.stim");
        std::fs::rename(fixture, &input).expect("rename fixture");
        let original = std::fs::read(&input).expect("read input");

        let error = run(
            CompileSource::InputFile(input.clone()),
            Some(temp.path().join("out.stim")),
            options(vec![3, 5]),
        )
        .expect_err("all output aliases must be checked before writing");

        assert!(error.to_string().contains("aliases the input"));
        assert!(!temp.path().join("out-d3.stim").exists());
        assert_eq!(
            std::fs::read(input).expect("read preserved input"),
            original
        );
    }

    #[test]
    fn multiple_distances_write_suffixed_output_files() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());
        let output = temp.path().join("out.stim");

        run(
            CompileSource::InputFile(input),
            Some(output),
            options(vec![3, 5]),
        )
        .expect("multiple distance compile should succeed");

        let d3 = temp.path().join("out-d3.stim");
        let d5 = temp.path().join("out-d5.stim");
        assert!(d3.exists(), "d=3 output should exist");
        assert!(d5.exists(), "d=5 output should exist");

        let d3_text = std::fs::read_to_string(d3).expect("read d=3 output");
        let d5_text = std::fs::read_to_string(d5).expect("read d=5 output");
        assert!(d3_text.contains("QUBIT_COORDS"));
        assert!(d5_text.contains("QUBIT_COORDS"));
    }

    /// `--jobs` overrides the memory-driven default cap in both directions;
    /// the default itself stays within it.
    #[test]
    fn jobs_overrides_the_default_concurrency_cap() {
        assert!((1..=4).contains(&compile_batch_size(None)));
        assert_eq!(compile_batch_size(NonZeroUsize::new(1)), 1);
        assert_eq!(compile_batch_size(NonZeroUsize::new(9)), 9);
    }

    /// Concurrency is a scheduling choice, so every `--jobs` value must emit
    /// the same artifacts — including `--jobs 1`, which never spawns a worker.
    #[test]
    fn jobs_does_not_change_the_emitted_artifacts() {
        let mut outputs = Vec::new();
        for jobs in [Some(1), Some(4), None] {
            let temp = tempdir().expect("create temp dir");
            let input = write_single_cube_blog(temp.path());

            let mut opts = options(vec![3, 5]);
            opts.jobs = jobs.and_then(NonZeroUsize::new);
            run(CompileSource::InputFile(input), None, opts).expect("compile every target");

            outputs.push([
                std::fs::read(temp.path().join("input-d3.stim")).expect("read d=3 output"),
                std::fs::read(temp.path().join("input-d5.stim")).expect("read d=5 output"),
            ]);
        }
        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn open_port_inputs_compile_all_filled_variants_with_distinct_suffixes() {
        let temp = tempdir().expect("create temp dir");
        let input = write_gallery_blog(temp.path(), "cnot", GalleryItem::CNOT);
        let variant_count = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("project CNOT for filling")
            .filled_graphs()
            .expect("fill CNOT")
            .len();
        assert!(variant_count > 1, "expected multiple filled variants");

        let mut opts = options(vec![3]);
        opts.auto_fill = true;
        run(CompileSource::InputFile(input), None, opts)
            .expect("open-port graph should auto-fill and compile");

        for index in 0..variant_count {
            let output_path = temp.path().join(format!("cnot-d3-fill{index}.stim"));
            assert!(output_path.exists(), "filled variant output should exist");
        }
    }

    #[test]
    fn open_temporal_graph_compiles_without_fill() {
        let temp = tempdir().expect("create temp dir");
        let input = write_gallery_blog(temp.path(), "cnot", GalleryItem::CNOT);

        run(CompileSource::InputFile(input), None, options(vec![3]))
            .expect("open temporal graph should compile as-authored");
        assert!(temp.path().join("cnot-d3.stim").exists());
    }

    #[test]
    fn auto_fill_rejects_non_clifford_graph() {
        let temp = tempdir().expect("create temp dir");
        let input = write_gallery_blog(temp.path(), "t_gate", GalleryItem::T);

        let mut opts = options(vec![3]);
        opts.auto_fill = true;
        let error = run(CompileSource::InputFile(input), None, opts)
            .expect_err("--auto-fill on a selective graph should fail");
        assert!(format!("{error:#}").contains("non-Clifford"));
    }

    #[test]
    fn fill_selects_a_single_variant() {
        let temp = tempdir().expect("create temp dir");
        let input = write_gallery_blog(temp.path(), "cnot", GalleryItem::CNOT);

        let mut opts = options(vec![3]);
        opts.fill = Some(1);
        run(CompileSource::InputFile(input), None, opts)
            .expect("--fill should compile a single variant");

        assert!(temp.path().join("cnot-d3-fill1.stim").exists());
    }

    #[test]
    fn fill_rejects_out_of_range_variant() {
        let temp = tempdir().expect("create temp dir");
        let input = write_gallery_blog(temp.path(), "cnot", GalleryItem::CNOT);

        let mut opts = options(vec![3]);
        opts.fill = Some(99);
        let error = run(CompileSource::InputFile(input), None, opts)
            .expect_err("--fill out of range should fail");
        assert!(format!("{error:#}").contains("out of range"));
    }

    #[test]
    fn fill_rejects_closed_inputs() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());

        let mut opts = options(vec![3]);
        opts.fill = Some(0);
        let error = run(CompileSource::InputFile(input), None, opts)
            .expect_err("--fill on a closed graph should fail");
        assert!(format!("{error:#}").contains("open ports"));
    }

    #[test]
    fn print_mode_disables_file_output() {
        let temp = tempdir().expect("create temp dir");
        let input = write_single_cube_blog(temp.path());

        let mut opts = options(vec![3]);
        opts.print = true;
        run(CompileSource::InputFile(input), None, opts).expect("print mode should compile");

        assert!(
            !temp.path().join("input-d3.stim").exists(),
            "print mode should not write the default output file"
        );
    }

    #[test]
    fn default_output_path_for_gallery_uses_gallery_id() {
        let output_path = default_output_path(
            &CompileSource::GalleryEntry(GalleryItem::CNOT),
            BuiltInBackend::Stim,
            OutputDescriptor {
                distance: 3,
                fill_variant: None,
            },
        );
        assert_eq!(output_path, PathBuf::from("./cnot-d3.stim"));
    }
}
