//! Background jobs (validation, stabilizers, `.blog` parsing, compilation) and
//! the system that polls their results.
//!
//! Native builds run each job on the async task pool and route its result to
//! the tab that requested it, discarding results whose graph revision or BLOG
//! source buffer has since moved on; WASM keeps browser file selection async.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};

use crate::components::{CameraSettings, EditorCamera, GraphElement};
use crate::program_view::build_concurrent_layer_views;
use crate::resources::{
    BloqViewerState, CompiledBloqView, ConcurrentOpsInputs, DetsliceData, EditorState, EditorTabId,
    EditorTabs, GraphState, ImportExportState, LayerCircuitView, Notifications, SliceVisibility,
    TargetState, ZxViewerState,
};
use crate::systems::EditorUpdateSet;
use crate::systems::camera::reset_camera_to_graph;
#[cfg(not(target_arch = "wasm32"))]
use crate::systems::ui::zx_viewer::simplified_zx_graph_with_cancellation;
use crate::systems::ui::zx_viewer::{SimplifiedZxView, ZxViewKey};
use crate::systems::ui::{UiIntent, UiIntentBuffer};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
#[cfg(not(target_arch = "wasm32"))]
use bloq_compile::CompileStage;
use bloq_compile::{CompileConfig, MAX_CODE_DISTANCE, spatial_hadamard_distance_warning};
#[cfg(not(target_arch = "wasm32"))]
use bloq_graph::CancellationToken;
use bloq_graph::{BlockGraph, BlockGraphError, ComputationCancelled, StabilizerGenerator};
use color_eyre::eyre::{self, WrapErr};
use strum::Display;
use web_time::Instant;

mod compilation;
#[cfg(target_arch = "wasm32")]
mod web_compilation;

#[cfg(test)]
pub(crate) use compilation::compile_graph_for_viewer;
use compilation::{CompiledDownload, build_detslice, save_compiled_downloads};

/// Background jobs and their result polling. Owns the compile/validation UI
/// state the jobs write back into.
pub(crate) struct JobsPlugin;

impl Plugin for JobsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorJobs>()
            .init_resource::<CompileUiState>()
            .add_systems(
                Update,
                (
                    sync_compile_readiness_system.run_if(resource_changed::<GraphState>),
                    poll_editor_jobs_system.run_if(|jobs: Res<EditorJobs>| jobs.is_busy()),
                )
                    .chain()
                    .in_set(EditorUpdateSet::Jobs),
            );
    }
}

/// The at-most-one in-flight task per job kind. Used both to prevent
/// double-scheduling and to drive the UI's "busy" indicators.
#[derive(Resource, Default)]
pub(crate) struct EditorJobs {
    validation: Option<Task<ValidationJobResult>>,
    stabilizers: Option<Task<StabilizersJobResult>>,
    import_blog_file: Option<Task<ImportBlogFileJobResult>>,
    parse_blog: Option<Task<ParseBlogJobResult>>,
    compile: Option<Task<CompileJobResult>>,
    viewer_compile: Option<Task<ViewerCompileJobResult>>,
    #[cfg(not(target_arch = "wasm32"))]
    progress: Option<CompileProgress>,
    #[cfg(target_arch = "wasm32")]
    web_compile: Option<WebCompileJob>,
    concurrent_ops: Option<Task<ConcurrentOpsJobResult>>,
    detslice: Option<Task<DetsliceJobResult>>,
    #[cfg(not(target_arch = "wasm32"))]
    zx_simplification: Option<Task<ZxSimplificationResult>>,
    #[cfg(target_arch = "wasm32")]
    web_zx_simplification: Option<(EditorTabId, ZxViewKey)>,
    #[cfg(not(target_arch = "wasm32"))]
    zx_request: Option<(EditorTabId, ZxViewKey, CancellationToken)>,
}

impl EditorJobs {
    /// Whether any background job is running.
    pub(crate) fn is_busy(&self) -> bool {
        self.validation.is_some()
            || self.stabilizers.is_some()
            || self.import_blog_file.is_some()
            || self.parse_blog.is_some()
            || self.compile.is_some()
            || self.viewer_compile.is_some()
            || self.web_compilation_running()
            || self.concurrent_ops.is_some()
            || self.detslice.is_some()
            || self.zx_simplification_running()
    }

    /// Whether a validation job is running.
    pub(crate) fn validation_running(&self) -> bool {
        self.validation.is_some()
    }

    /// Whether a stabilizer job is running.
    pub(crate) fn stabilizers_running(&self) -> bool {
        self.stabilizers.is_some()
    }

    /// Whether a download or viewer compile is running.
    pub(crate) fn compilation_running(&self) -> bool {
        self.compile.is_some() || self.viewer_compile.is_some() || self.web_compilation_running()
    }

    pub(crate) fn compilation_owner(&self) -> Option<(EditorTabId, u64, bool)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.progress
                .as_ref()
                .filter(|_| self.compilation_running())
                .map(|progress| {
                    (
                        progress.tab_id,
                        progress.revision,
                        self.viewer_compile.is_some(),
                    )
                })
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.web_compile
                .as_ref()
                .map(|job| (job.tab_id, job.revision, job.viewer))
        }
    }

    fn cancel_compilation(&mut self, tab_id: EditorTabId) -> bool {
        if self
            .compilation_owner()
            .is_none_or(|(owner, _, _)| owner != tab_id)
        {
            return false;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(progress) = self.progress.take() {
            progress.cancellation.cancel();
            self.compile = None;
            self.viewer_compile = None;
        }
        #[cfg(target_arch = "wasm32")]
        {
            web_compilation::cancel();
            self.web_compile = None;
        }
        true
    }

    pub(crate) fn zx_simplification_for(&self, tab_id: EditorTabId, key: ZxViewKey) -> bool {
        self.zx_owner() == Some((tab_id, key)) && self.zx_simplification_running()
    }

    fn zx_owner(&self) -> Option<(EditorTabId, ZxViewKey)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.zx_request.as_ref().map(|(tab, key, _)| (*tab, *key))
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.web_zx_simplification
        }
    }

    pub(crate) fn cancel_zx_simplification(&mut self, tab_id: EditorTabId, key: ZxViewKey) -> bool {
        if self.zx_owner() != Some((tab_id, key)) {
            return false;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((_, _, token)) = self.zx_request.take() {
            token.cancel();
            self.zx_simplification = None;
        }
        #[cfg(target_arch = "wasm32")]
        {
            web_compilation::cancel_zx_simplification();
            self.web_zx_simplification = None;
        }
        true
    }

    fn web_compilation_running(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.web_compile.is_some()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    fn zx_simplification_running(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.zx_simplification.is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.web_zx_simplification.is_some()
        }
    }

    pub(crate) fn compilation_progress(
        &self,
        tab_id: EditorTabId,
        revision: u64,
    ) -> Option<(String, u64)> {
        #[cfg(target_arch = "wasm32")]
        if let Some(job) = self
            .web_compile
            .as_ref()
            .filter(|job| job.tab_id == tab_id && job.revision == revision)
        {
            let stage = match &job.result {
                Some(Ok(_)) if job.viewer => "Building view".to_string(),
                Some(Ok(_)) => "Exporting output".to_string(),
                Some(Err(_)) => "Compilation failed".to_string(),
                None => web_compilation::stage().unwrap_or_else(|| "Starting compiler".to_string()),
            };
            return Some((stage, job.started.elapsed().as_secs()));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.progress.as_ref().and_then(|progress| {
                (self.compilation_running()
                    && progress.tab_id == tab_id
                    && progress.revision == revision)
                    .then(|| (progress.label(), progress.started.elapsed().as_secs()))
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }

    pub(crate) fn parse_blog_running(&self) -> bool {
        self.import_blog_file.is_some() || self.parse_blog.is_some()
    }

    #[cfg(test)]
    fn poll_stabilizers_for_test(&mut self) -> Option<StabilizersJobResult> {
        poll_task(&mut self.stabilizers)
    }
}

#[cfg(target_arch = "wasm32")]
struct WebCompileJob {
    tab_id: EditorTabId,
    revision: u64,
    request: CompileRequest,
    source: Box<BlockGraph>,
    viewer: bool,
    started: Instant,
    result: Option<Result<web_compilation::WorkerResult, String>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum CompileProgressPhase {
    Compiler(CompileStage),
    Exporting,
    BuildingView,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct CompileProgress {
    tab_id: EditorTabId,
    revision: u64,
    started: Instant,
    phase: Arc<Mutex<CompileProgressPhase>>,
    cancellation: CancellationToken,
}

#[cfg(not(target_arch = "wasm32"))]
impl CompileProgress {
    fn new(tab_id: EditorTabId, revision: u64) -> Self {
        Self {
            tab_id,
            revision,
            started: Instant::now(),
            cancellation: CancellationToken::new(),
            phase: Arc::new(Mutex::new(CompileProgressPhase::Compiler(
                CompileStage::Validation,
            ))),
        }
    }

    fn set_phase(&self, phase: CompileProgressPhase) {
        *self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = phase;
    }

    fn observer(&self) -> impl Fn(CompileStage) + Send + Sync + 'static {
        let progress = self.clone();
        move |stage| progress.set_phase(CompileProgressPhase::Compiler(stage))
    }

    fn label(&self) -> String {
        match *self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            CompileProgressPhase::Compiler(stage) => stage.to_string(),
            CompileProgressPhase::Exporting => "Exporting output".to_string(),
            CompileProgressPhase::BuildingView => "Building view".to_string(),
        }
    }
}

struct ValidationJobResult {
    tab_id: EditorTabId,
    revision: u64,
    result: eyre::Result<BlockGraph>,
}

struct StabilizersJobResult {
    tab_id: EditorTabId,
    revision: u64,
    layer: Option<i32>,
    result: eyre::Result<Vec<StabilizerGenerator>>,
}

struct ImportedBlogFile {
    title: Option<String>,
    buffer: String,
}

type ImportBlogFileJobResult = eyre::Result<Option<ImportedBlogFile>>;

struct ParseBlogJobResult {
    tab_id: EditorTabId,
    revision: u64,
    source: String,
    title: Option<String>,
    result: eyre::Result<ParsedBlog>,
}

struct ParsedBlog {
    graph: BlockGraph,
    source_graph: BlockGraph,
}

fn parse_blog_for_editor(source: &str) -> eyre::Result<ParsedBlog> {
    let source_graph = BlockGraph::from_text(source).wrap_err("parse BLOG text")?;
    let graph = source_graph
        .flatten()
        .wrap_err("materialize BLOG modules")?
        .fix_shadowed_faces();
    Ok(ParsedBlog {
        graph,
        source_graph,
    })
}

/// Opens the platform file picker and reads one UTF-8 BLOG file without
/// blocking the editor frame. No-op while another BLOG import is active.
pub(crate) fn schedule_import_blog_file_job(
    jobs: &mut EditorJobs,
    notifications: &mut Notifications,
) {
    if jobs.parse_blog_running() {
        notifications.push_warn("BLOG import is already running");
        return;
    }
    jobs.import_blog_file = Some(AsyncComputeTaskPool::get().spawn(async move {
        let Some(file) = rfd::AsyncFileDialog::new()
            .add_filter("BLOG", &["blog"])
            .pick_file()
            .await
        else {
            return Ok(None);
        };
        let file_name = file.file_name();
        Ok(Some(decode_imported_blog_file(
            &file_name,
            file.read().await,
        )?))
    }));
    notifications.record_info("Opened BLOG file picker");
}

fn decode_imported_blog_file(file_name: &str, bytes: Vec<u8>) -> eyre::Result<ImportedBlogFile> {
    Ok(ImportedBlogFile {
        title: std::path::Path::new(file_name)
            .file_stem()
            .and_then(|name| name.to_str())
            .map(str::to_owned),
        buffer: String::from_utf8(bytes).wrap_err("decode BLOG file as UTF-8")?,
    })
}

struct CompileJobResult {
    tab_id: EditorTabId,
    revision: u64,
    result: eyre::Result<CompiledDownload>,
}

struct ViewerCompileJobResult {
    tab_id: EditorTabId,
    revision: u64,
    result: eyre::Result<CompiledBloqView>,
}

struct ConcurrentOpsJobResult {
    tab_id: EditorTabId,
    generation: u64,
    layer_circuits: HashMap<i32, LayerCircuitView>,
    error: Option<String>,
}

/// The lazy detector-slice computation, tagged with the compile generation it ran
/// against so a stale result (a newer compile landed meanwhile) is discarded.
/// `build_detslice` folds its own unavailability into [`DetsliceData`], so there
/// is no `Result` here.
struct DetsliceJobResult {
    tab_id: EditorTabId,
    generation: u64,
    result: DetsliceData,
}

struct ZxSimplificationResult {
    tab_id: EditorTabId,
    key: ZxViewKey,
    result: eyre::Result<SimplifiedZxView>,
}

fn is_cancellation(error: &eyre::Report) -> bool {
    error
        .chain()
        .any(<dyn std::error::Error + 'static>::is::<ComputationCancelled>)
}

/// Cancels only the requesting tab's current compile and clears its pending UI.
pub(crate) fn cancel_editor_compilation(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    tabs: &mut EditorTabs,
    compile_ui: &mut CompileUiState,
    viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if !jobs.cancel_compilation(tab_id) {
        return;
    }
    let clear = |ui: &mut CompileUiState, viewer: &mut BloqViewerState| {
        ui.mark_compile_cancelled("Compilation cancelled");
        viewer.pending_revision = None;
        viewer.last_error = None;
    };
    if tab_id == tabs.active {
        clear(compile_ui, viewer);
    } else if let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == tab_id) {
        clear(
            &mut tab.snapshot.compile_ui,
            &mut tab.snapshot.circuit_viewer,
        );
    }
    notifications.record_info("Compilation cancelled");
}

fn cancel_invalidated_compilation(
    jobs: &mut EditorJobs,
    tabs: &mut EditorTabs,
    graph: &GraphState,
    editor: &EditorState,
    compile_ui: &mut CompileUiState,
    viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    let Some((tab_id, revision, viewer_job)) = jobs.compilation_owner() else {
        return;
    };
    let current = if tab_id == tabs.active {
        Some((graph.revision, editor.mode))
    } else {
        tabs.tabs.iter().find(|tab| tab.id == tab_id).map(|tab| {
            (
                tab.snapshot.graph_state.revision,
                tab.snapshot.editor_state.mode,
            )
        })
    };
    if current.is_none_or(|(current_revision, mode)| {
        current_revision != revision || (viewer_job && mode != crate::resources::EditorMode::Bloq)
    }) {
        cancel_editor_compilation(jobs, tab_id, tabs, compile_ui, viewer, notifications);
    }
}

fn cancel_invalidated_zx(
    jobs: &mut EditorJobs,
    tabs: &EditorTabs,
    graph: &GraphState,
    viewer: &ZxViewerState,
) {
    let Some((tab_id, key)) = jobs.zx_owner() else {
        return;
    };
    let current = if tab_id == tabs.active {
        viewer.accepts_simplification(graph.revision, key)
    } else {
        tabs.tabs
            .iter()
            .find(|tab| tab.id == tab_id)
            .is_some_and(|tab| {
                tab.snapshot
                    .zx_viewer
                    .accepts_simplification(tab.snapshot.graph_state.revision, key)
            })
    };
    if !current {
        jobs.cancel_zx_simplification(tab_id, key);
    }
}

/// ponytail: one native verifier slot; use per-tab slots if queued previews need
/// more throughput. Cancellation releases stale normalization work at checkpoints.
#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        clippy::unnecessary_wraps,
        reason = "browser Worker startup is fallible"
    )
)]
pub(crate) fn schedule_zx_simplification(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    key: ZxViewKey,
) -> Result<(), String> {
    if jobs.zx_simplification_running() {
        return Ok(());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let graph = graph_state.graph.clone();
        let cancellation = CancellationToken::new();
        jobs.zx_request = Some((tab_id, key, cancellation.clone()));
        jobs.zx_simplification = Some(AsyncComputeTaskPool::get().spawn(async move {
            ZxSimplificationResult {
                tab_id,
                key,
                result: cancellation.run(|| {
                    let graph =
                        simplified_zx_graph_with_cancellation(&graph, key.2, &cancellation)?;
                    cancellation.check()?;
                    Ok(SimplifiedZxView::new(graph))
                }),
            }
        }));
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_compilation::start_zx_simplification(&graph_state.graph, key.2)?;
        jobs.web_zx_simplification = Some((tab_id, key));
    }
    Ok(())
}

fn apply_zx_simplification_result(
    done: ZxSimplificationResult,
    tabs: &mut EditorTabs,
    graph_state: &GraphState,
    zx_viewer: &mut ZxViewerState,
) -> bool {
    let (revision, viewer) = if done.tab_id == tabs.active {
        (graph_state.revision, zx_viewer)
    } else {
        let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
            return false;
        };
        (
            tab.snapshot.graph_state.revision,
            &mut tab.snapshot.zx_viewer,
        )
    };
    if done.result.as_ref().is_err_and(is_cancellation) {
        viewer.cancel_simplification(revision, done.key)
    } else {
        viewer.finish_simplification(
            revision,
            done.key,
            done.result.map_err(|error| format!("{error:#}")),
        )
    }
}

/// Spawns a background job validating the current graph, marking validation
/// pending. No-op (with a warning) if one is already running.
pub(crate) fn schedule_validate_graph_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    compile_ui: &mut CompileUiState,
    notifications: &mut Notifications,
) {
    if jobs.validation.is_some() {
        notifications.push_warn("Validation is already running");
        return;
    }
    let graph = graph_state.graph.clone();
    let revision = graph_state.revision;
    compile_ui.mark_validation_pending(revision);
    let pool = AsyncComputeTaskPool::get();
    jobs.validation = Some(pool.spawn(async move {
        ValidationJobResult {
            tab_id,
            revision,
            result: validate_graph_for_editor(graph).wrap_err("validate block graph"),
        }
    }));
    notifications.record_info("Started graph validation");
}

/// Spawns a background job computing the graph's stabilizer generators,
/// optionally restricted to the current layer. No-op if one is already running.
pub(crate) fn schedule_stabilizers_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    layer_only: bool,
    plane_height: i32,
    notifications: &mut Notifications,
) {
    if jobs.stabilizers.is_some() {
        notifications.push_warn("Stabilizer computation is already running");
        return;
    }
    let graph = if layer_only {
        graph_state.graph.layer(plane_height).into_graph()
    } else {
        graph_state.graph.clone()
    };
    let module = graph_state
        .source_graph
        .as_ref()
        .filter(|_| !layer_only && graph_state.is_composed())
        .cloned();
    let revision = graph_state.revision;
    let pool = AsyncComputeTaskPool::get();
    jobs.stabilizers = Some(pool.spawn(async move {
        let source = module.as_deref().unwrap_or(&graph);
        let result = displayed_surfaces(source, &graph).wrap_err("compute stabilizers");
        StabilizersJobResult {
            tab_id,
            revision,
            layer: layer_only.then_some(plane_height),
            result,
        }
    }));
    notifications.record_info("Started stabilizer computation");
}

/// Plans the shown branch arms of `graph` from its authored `source`, which is
/// the graph itself unless it was materialized from modules.
fn displayed_surfaces(
    source: &BlockGraph,
    graph: &BlockGraph,
) -> Result<Vec<StabilizerGenerator>, BlockGraphError> {
    let assignments = graph
        .branch_definitions()
        .iter()
        .map(|branch| (branch.target, branch.shown_true()))
        .collect::<Vec<_>>();
    source.planned_surfaces(&assignments, bloq_graph::ModuleCertificationLimits::DEFAULT)
}

/// Validates without computing action metadata twice. A hierarchy analyzes
/// its linked actions during validation instead.
fn validate_graph_for_editor(graph: BlockGraph) -> Result<BlockGraph, BlockGraphError> {
    if !graph.has_actions() || !graph.instances.is_empty() {
        graph.validate()?;
        return Ok(graph);
    }
    graph.analyze_actions()
}

/// Spawns a background job parsing `blog_buffer` into a graph, optionally
/// renaming the target tab on success. No-op if one is already running.
pub(crate) fn schedule_parse_blog_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    revision: u64,
    title: Option<String>,
    blog_buffer: &str,
    notifications: &mut Notifications,
) {
    if jobs.parse_blog.is_some() {
        notifications.push_warn("BLOG parse is already running");
        return;
    }
    let buffer = blog_buffer.to_owned();
    let pool = AsyncComputeTaskPool::get();
    jobs.parse_blog = Some(pool.spawn(async move {
        let result = parse_blog_for_editor(&buffer);
        ParseBlogJobResult {
            tab_id,
            revision,
            source: buffer,
            title,
            result,
        }
    }));
    notifications.record_info("Started BLOG parse");
}

/// Spawns a background job compiling the graph to the requested download
/// format. No-op if a compile is already running.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schedule_compile_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    compile_ui: &mut CompileUiState,
    notifications: &mut Notifications,
    request: CompileRequest,
) {
    if !accept_compile_request(jobs, graph_state, notifications) {
        return;
    }
    compile_ui.mark_compile_pending();
    let source = graph_state.resolved_graph();
    let revision = graph_state.revision;
    let progress = CompileProgress::new(tab_id, revision);
    jobs.progress = Some(progress.clone());
    let pool = AsyncComputeTaskPool::get();
    jobs.compile = Some(pool.spawn(async move {
        CompileJobResult {
            tab_id,
            revision,
            result: source.and_then(|graph| {
                compilation::compile_downloads(&graph, &request, Some(&progress))
            }),
        }
    }));
    notifications.record_info("Started compilation");
}

/// Spawns a background job compiling the graph into node/edge views for the
/// Bloq viewer. No-op if a compile is already running.
#[cfg(not(target_arch = "wasm32"))]
fn schedule_viewer_compile_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    request: CompileRequest,
) {
    if !accept_compile_request(jobs, graph_state, notifications) {
        return;
    }
    let source = graph_state.resolved_graph();
    let revision = graph_state.revision;
    let progress = CompileProgress::new(tab_id, revision);
    jobs.progress = Some(progress.clone());
    circuit_viewer.begin_compile(revision);
    let pool = AsyncComputeTaskPool::get();
    jobs.viewer_compile = Some(pool.spawn(async move {
        ViewerCompileJobResult {
            tab_id,
            revision,
            result: source.and_then(|graph| {
                compilation::compile_for_viewer(&graph, &request, Some(&progress))
            }),
        }
    }));
    notifications.record_info("Started Bloq viewer compilation");
}

/// Rebuild the viewer from its retained compilation with source choices pinned.
pub(crate) fn request_viewer_branch_pins(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    current_revision: u64,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    pins: Option<std::collections::BTreeMap<String, bool>>,
) {
    if jobs.compilation_running() || circuit_viewer.is_stale(current_revision) {
        return;
    }
    let Some(input) = circuit_viewer.pinning_input() else {
        return;
    };
    circuit_viewer.begin_compile(current_revision);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let progress = CompileProgress::new(tab_id, current_revision);
        jobs.progress = Some(progress.clone());
        jobs.viewer_compile = Some(AsyncComputeTaskPool::get().spawn(async move {
            ViewerCompileJobResult {
                tab_id,
                revision: current_revision,
                result: progress
                    .cancellation
                    .run(|| compilation::pin_viewer_branches(input, pins)),
            }
        }));
        notifications.record_info("Updating viewer branch selections");
    }
    #[cfg(target_arch = "wasm32")]
    {
        finalize_viewer_compile_result(
            ViewerCompileJobResult {
                tab_id,
                revision: current_revision,
                result: compilation::pin_viewer_branches(input, pins),
            },
            current_revision,
            circuit_viewer,
            notifications,
        );
        if circuit_viewer.concurrent_ops_pending {
            compute_concurrent_ops_now(tab_id, circuit_viewer, notifications);
        }
    }
}

/// Builds a compile request from the UI and starts a viewer compile,
/// reporting a bad request as an error.
pub(crate) fn start_viewer_compile(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    compile_ui: &CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    match compile_ui.build_request() {
        Ok(request) => schedule_viewer_compile_job(
            jobs,
            tab_id,
            graph_state,
            circuit_viewer,
            notifications,
            request,
        ),
        Err(err) => {
            circuit_viewer.pending_revision = None;
            circuit_viewer.last_error = Some(err.clone());
            notifications.push_error(err);
        }
    }
}

/// WASM path: starts the browser compiler worker and returns before painting.
#[cfg(target_arch = "wasm32")]
pub(crate) fn schedule_compile_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    compile_ui: &mut CompileUiState,
    notifications: &mut Notifications,
    request: CompileRequest,
) {
    if !accept_compile_request(jobs, graph_state, notifications) {
        return;
    }
    compile_ui.mark_compile_pending();
    match start_web_compile(jobs, tab_id, graph_state, request, false) {
        Ok(()) => notifications.record_info("Started compilation"),
        Err(err) => {
            compile_ui.mark_compile_failed(&err);
            notifications.push_error(err);
        }
    }
}

/// WASM path: starts the browser compiler worker and returns before painting.
#[cfg(target_arch = "wasm32")]
fn schedule_viewer_compile_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    request: CompileRequest,
) {
    if !accept_compile_request(jobs, graph_state, notifications) {
        return;
    }
    circuit_viewer.begin_compile(graph_state.revision);
    match start_web_compile(jobs, tab_id, graph_state, request, true) {
        Ok(()) => notifications.record_info("Started Bloq viewer compilation"),
        Err(err) => {
            circuit_viewer.pending_revision = None;
            circuit_viewer.last_error = Some(err.clone());
            notifications.push_error(err);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn start_web_compile(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    graph_state: &GraphState,
    request: CompileRequest,
    viewer: bool,
) -> Result<(), String> {
    let source = Box::new(
        graph_state
            .resolved_graph()
            .map_err(|err| err.to_string())?,
    );
    web_compilation::start(&source, &request)?;
    jobs.web_compile = Some(WebCompileJob {
        tab_id,
        revision: graph_state.revision,
        request,
        source,
        viewer,
        started: Instant::now(),
        result: None,
    });
    Ok(())
}

/// Common gate for every compile entry point, native and web: refuses a second
/// concurrent compile or an unfinished branch capture, then warns when a spatial
/// Hadamard needs a larger code distance. `false` means no job may start.
fn accept_compile_request(
    jobs: &EditorJobs,
    graph_state: &GraphState,
    notifications: &mut Notifications,
) -> bool {
    if jobs.compilation_running() {
        notifications.push_warn("Compilation is already running");
        return false;
    }
    if graph_state.pending_branch_arm.is_some() {
        notifications.push_warn("Finish or cancel the captured branch arm first");
        return false;
    }
    warn_spatial_hadamard_distance(&graph_state.graph, notifications);
    true
}

fn warn_spatial_hadamard_distance(graph: &BlockGraph, notifications: &mut Notifications) {
    if let Some(message) = spatial_hadamard_distance_warning(graph) {
        notifications.push_warn(message);
    }
}

pub(crate) fn request_concurrent_ops(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    enabled: bool,
) {
    if !enabled || !circuit_viewer.layer_circuits.is_empty() {
        circuit_viewer.set_concurrent_ops(enabled);
        return;
    }
    if circuit_viewer.concurrent_ops_pending || circuit_viewer.concurrent_ops_error.is_some() {
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = jobs;
        compute_concurrent_ops_now(tab_id, circuit_viewer, notifications);
    }
    #[cfg(not(target_arch = "wasm32"))]
    schedule_concurrent_ops_job(jobs, tab_id, circuit_viewer, notifications);
}

fn run_concurrent_ops(tab_id: EditorTabId, inputs: ConcurrentOpsInputs) -> ConcurrentOpsJobResult {
    let (layer_circuits, error) =
        build_concurrent_layer_views(&inputs.program, &inputs.viewer_graph, inputs.source_offset);
    ConcurrentOpsJobResult {
        tab_id,
        generation: inputs.generation,
        layer_circuits,
        error,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn schedule_concurrent_ops_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    let Some(inputs) = circuit_viewer.concurrent_ops_inputs() else {
        return;
    };
    circuit_viewer.begin_concurrent_ops();
    if jobs.concurrent_ops.is_some() {
        return;
    }
    jobs.concurrent_ops =
        Some(AsyncComputeTaskPool::get().spawn(async move { run_concurrent_ops(tab_id, inputs) }));
    notifications.record_info("Building concurrent ops");
}

#[cfg(not(target_arch = "wasm32"))]
fn schedule_pending_concurrent_ops_job(
    jobs: &mut EditorJobs,
    tabs: &mut EditorTabs,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if jobs.concurrent_ops.is_some() {
        return;
    }
    if circuit_viewer.concurrent_ops_pending {
        schedule_concurrent_ops_job(jobs, tabs.active, circuit_viewer, notifications);
        return;
    }
    let active = tabs.active;
    if let Some(tab) = tabs
        .tabs
        .iter_mut()
        .find(|tab| tab.id != active && tab.snapshot.circuit_viewer.concurrent_ops_pending)
    {
        schedule_concurrent_ops_job(
            jobs,
            tab.id,
            &mut tab.snapshot.circuit_viewer,
            notifications,
        );
    }
}

/// WASM tasks run on the UI thread, matching the existing detector-slice policy.
#[cfg(target_arch = "wasm32")]
fn compute_concurrent_ops_now(
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    let Some(inputs) = circuit_viewer.concurrent_ops_inputs() else {
        return;
    };
    circuit_viewer.begin_concurrent_ops();
    finalize_concurrent_ops_result(
        run_concurrent_ops(tab_id, inputs),
        circuit_viewer,
        notifications,
    );
}

/// Applies detector/observable slice visibility, computing their shared cache
/// lazily the first time either overlay is enabled for the current compile.
///
/// - Enabling with the cache already computed activates the mode instantly.
/// - Enabling with no cache kicks the [`build_detslice`] job (native) or runs it
///   inline (wasm) and marks the viewer pending; the mode activates when the
///   result lands.
/// - Disabling turns the mode off and drops any intent to activate on completion;
///   the cache is kept for the next enable.
pub(crate) fn request_detslice(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    visibility: SliceVisibility,
) {
    if !visibility.any() {
        circuit_viewer.set_slice_visibility(visibility);
        circuit_viewer.cancel_detslice_pending();
        return;
    }
    if circuit_viewer.detslice_pending.is_some() {
        circuit_viewer.begin_detslice(visibility);
        return;
    }
    if circuit_viewer.detslice_computed {
        circuit_viewer.set_slice_visibility(visibility);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = jobs;
        compute_detslice_now(tab_id, circuit_viewer, notifications, visibility);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        schedule_detslice_job(jobs, tab_id, circuit_viewer, notifications, visibility);
    }
}

/// Spawns the native detector-slice job from retained compile inputs.
#[cfg(not(target_arch = "wasm32"))]
fn schedule_detslice_job(
    jobs: &mut EditorJobs,
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    visibility: SliceVisibility,
) {
    let Some(inputs) = circuit_viewer.detslice_inputs() else {
        return;
    };
    circuit_viewer.begin_detslice(visibility);
    if jobs.detslice.is_some() {
        return;
    }
    let pool = AsyncComputeTaskPool::get();
    jobs.detslice = Some(pool.spawn(async move {
        DetsliceJobResult {
            tab_id,
            generation: inputs.generation,
            result: build_detslice(
                &inputs.program,
                &inputs.viewer_graph,
                inputs.source_offset,
                inputs.compile_config,
            ),
        }
    }));
    notifications.record_info("Computing detector slices");
}

#[cfg(not(target_arch = "wasm32"))]
fn schedule_pending_detslice_job(
    jobs: &mut EditorJobs,
    tabs: &mut EditorTabs,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if jobs.detslice.is_some() {
        return;
    }
    if let Some(visibility) = circuit_viewer.detslice_pending
        && !circuit_viewer.detslice_computed
    {
        schedule_detslice_job(jobs, tabs.active, circuit_viewer, notifications, visibility);
        return;
    }
    let active = tabs.active;
    for tab in &mut tabs.tabs {
        if tab.id != active
            && let Some(visibility) = tab.snapshot.circuit_viewer.detslice_pending
            && !tab.snapshot.circuit_viewer.detslice_computed
        {
            schedule_detslice_job(
                jobs,
                tab.id,
                &mut tab.snapshot.circuit_viewer,
                notifications,
                visibility,
            );
            return;
        }
    }
}

/// WASM path: computes the detector-slice overlay synchronously and finalizes it.
/// The first toggle therefore hitches while the tape is built. `AsyncComputeTaskPool`
/// does exist on the web target, but its single-threaded backend only advances a
/// spawned task when polled, on the main thread — so spawning buys no concurrency
/// here; we compute inline and finalize in one step.
#[cfg(target_arch = "wasm32")]
fn compute_detslice_now(
    tab_id: EditorTabId,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
    visibility: SliceVisibility,
) {
    let Some(inputs) = circuit_viewer.detslice_inputs() else {
        return;
    };
    circuit_viewer.begin_detslice(visibility);
    let done = DetsliceJobResult {
        tab_id,
        generation: inputs.generation,
        result: build_detslice(
            &inputs.program,
            &inputs.viewer_graph,
            inputs.source_offset,
            inputs.compile_config,
        ),
    };
    finalize_detslice_result(done, circuit_viewer, notifications);
}

/// Marks validation stale when the graph changes, so the compile panel never
/// shows a pass for an edited graph.
fn sync_compile_readiness_system(
    graph_state: Res<GraphState>,
    mut compile_ui: ResMut<CompileUiState>,
) {
    compile_ui.mark_stale_if_graph_changed(graph_state.revision);
}

/// Installs `graph` as the working graph and resets dependent editor state.
pub(crate) fn replace_graph_with_cleanup(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    graph: BlockGraph,
) {
    replace_graph_with_cleanup_impl(
        graph_state,
        editor_state,
        target_state,
        compile_ui,
        circuit_viewer,
        graph,
        None,
    );
}

/// Like [`replace_graph_with_cleanup`], but also installs a fresh selection
/// (e.g. the transformed elements after a translate/rotate).
pub(crate) fn replace_graph_with_cleanup_and_selection(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    graph: BlockGraph,
    selected_elements: impl IntoIterator<Item = GraphElement>,
) {
    replace_graph_with_cleanup_impl(
        graph_state,
        editor_state,
        target_state,
        compile_ui,
        circuit_viewer,
        graph,
        Some(selected_elements.into_iter().collect()),
    );
}

fn replace_graph_with_cleanup_impl(
    graph_state: &mut GraphState,
    editor_state: &mut EditorState,
    target_state: &mut TargetState,
    compile_ui: &mut CompileUiState,
    circuit_viewer: &mut BloqViewerState,
    graph: BlockGraph,
    selected_elements: Option<Vec<GraphElement>>,
) {
    graph_state.graph = graph;
    graph_state.commit();

    editor_state.clear_hover();
    if let Some(selected_elements) = selected_elements {
        editor_state.replace_selection(selected_elements);
    }
    editor_state.sync_after_graph_edit(graph_state);

    target_state.open_window = false;
    target_state.target = None;
    target_state.tag_buffer.clear();

    compile_ui.reset_for_new_graph();
    circuit_viewer.reset_for_new_graph();
}

fn poll_task<T>(slot: &mut Option<Task<T>>) -> Option<T> {
    let mut task = slot.take()?;
    match future::block_on(future::poll_once(&mut task)) {
        Some(done) => Some(done),
        None => {
            *slot = Some(task);
            None
        }
    }
}

fn finalize_compile_result(
    done: CompileJobResult,
    current_revision: u64,
    compile_ui: &mut CompileUiState,
    notifications: &mut Notifications,
) {
    if done.revision != current_revision {
        compile_ui.mark_compile_cancelled(format!(
            "Compile finished for an older graph revision (r{}); current graph is r{}. Compile again to download output.",
            done.revision, current_revision
        ));
        notifications.push_warn("Discarded stale compile result due to newer graph edits");
        return;
    }

    match done.result {
        Ok(output) => match save_compiled_downloads(&output) {
            Ok(Some(location)) => {
                notifications.push_info(format!("Downloaded compiled output to {}", location));
                compile_ui.mark_compile_succeeded("Downloaded 1 output file".to_string());
            }
            Ok(None) => {
                compile_ui.compile_state = crate::resources::CompileExecutionState::Idle;
                compile_ui.compile_message = "Compile download canceled.".to_string();
                notifications.push_warn("Compile download canceled");
            }
            Err(err) => {
                compile_ui.mark_compile_failed(format!("Download failed: {err:#}"));
                notifications.push_error_report("Failed to download compiled output", &err);
            }
        },
        Err(err) if is_cancellation(&err) => {
            compile_ui.mark_compile_cancelled("Compilation cancelled");
            notifications.record_info("Compilation cancelled");
        }
        Err(err) => {
            compile_ui.mark_compile_failed(format!("{err:#}"));
            notifications.push_error_report("Compilation failed", &err);
        }
    }
}

fn finalize_viewer_compile_result(
    done: ViewerCompileJobResult,
    current_revision: u64,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    if done.revision != current_revision {
        circuit_viewer.pending_revision = None;
        circuit_viewer.last_error = None;
        notifications.push_warn("Discarded stale Bloq viewer result due to newer graph edits");
        return;
    }
    match done.result {
        Ok(output) => {
            let loaded_nodes = output.nodes.len();
            circuit_viewer.finish_compile(done.revision, output);
            notifications.push_info(format!("Loaded {} Bloq nodes into viewer", loaded_nodes));
        }
        Err(err) if is_cancellation(&err) => {
            circuit_viewer.pending_revision = None;
            circuit_viewer.last_error = None;
            notifications.record_info("Bloq viewer compilation cancelled");
        }
        Err(err) => {
            circuit_viewer.pending_revision = None;
            circuit_viewer.last_error = Some(format!("{err:#}"));
            notifications.push_error_report("Bloq viewer compilation failed", &err);
        }
    }
}

fn finalize_concurrent_ops_result(
    done: ConcurrentOpsJobResult,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    let error = done.error.clone();
    if circuit_viewer.finish_concurrent_ops(done.generation, done.layer_circuits, done.error)
        && let Some(error) = error
    {
        notifications.push_warn(format!("Concurrent ops unavailable: {error}"));
    }
}

fn apply_concurrent_ops_result_to_tab(
    tabs: &mut EditorTabs,
    done: ConcurrentOpsJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded concurrent ops for a closed tab");
        return;
    };
    finalize_concurrent_ops_result(done, &mut tab.snapshot.circuit_viewer, notifications);
}

/// Installs a finished detector-slice computation onto a viewer, discarding a
/// result that a newer compile has made stale (checked against the viewer's
/// current compile revision inside `finish_detslice`).
fn finalize_detslice_result(
    done: DetsliceJobResult,
    circuit_viewer: &mut BloqViewerState,
    notifications: &mut Notifications,
) {
    let reason = done.result.unavailable_reason.clone();
    if circuit_viewer.finish_detslice(done.generation, done.result)
        && let Some(reason) = reason
    {
        notifications.push_warn(format!("Detector slices unavailable: {reason}"));
    }
}

fn apply_detslice_result_to_tab(
    tabs: &mut EditorTabs,
    done: DetsliceJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded detector-slice result for a closed tab");
        return;
    };
    finalize_detslice_result(done, &mut tab.snapshot.circuit_viewer, notifications);
}

enum ValidationOutcome {
    Stale,
    Passed,
    Failed(String),
}

/// Drive the shared validation state machine on `compile_ui`. Callers own the
/// notification wording so the active-tab live resources and background-tab
/// snapshots stay in lockstep here.
fn apply_validation_outcome(
    compile_ui: &mut CompileUiState,
    graph_state: &mut GraphState,
    done: ValidationJobResult,
) -> ValidationOutcome {
    compile_ui.pending_revision = None;
    if done.revision != graph_state.revision {
        compile_ui.validation_state = ValidationState::Unknown;
        compile_ui.validation_message =
            "Validation finished on an older revision. Re-run validation.".to_string();
        compile_ui.validated_revision = None;
        return ValidationOutcome::Stale;
    }
    match done.result {
        Ok(graph) => {
            graph_state.graph = graph.clone();
            if let Some(snapshot) = graph_state.history.get_mut(graph_state.current_index) {
                snapshot.graph = graph;
            }
            compile_ui.validation_state = ValidationState::Passed;
            compile_ui.validation_message =
                "Validation passed. Graph is compile-ready.".to_string();
            compile_ui.validated_revision = Some(done.revision);
            ValidationOutcome::Passed
        }
        Err(err) => {
            // Flatten the report chain once; the chip tooltip and the caller's
            // toast share the same rendered String.
            let rendered = format!("{err:#}");
            compile_ui.validation_state = ValidationState::Failed;
            compile_ui.validation_message = rendered.clone();
            compile_ui.validated_revision = Some(done.revision);
            ValidationOutcome::Failed(rendered)
        }
    }
}

enum StabilizersOutcome {
    Stale,
    Computed(usize),
    Failed(eyre::Report),
}

fn apply_stabilizers_outcome(
    editor_state: &mut EditorState,
    graph_state: &mut GraphState,
    done: StabilizersJobResult,
) -> StabilizersOutcome {
    if done.revision != graph_state.revision
        || editor_state.mode != crate::resources::EditorMode::View
        || done.layer
            != editor_state
                .view_current_layer_only
                .then_some(editor_state.plane_height)
    {
        return StabilizersOutcome::Stale;
    }
    match done.result {
        Ok(stabilizers) => {
            editor_state.stabilizers = stabilizers;
            editor_state.current_stabilizer_index = 0;
            editor_state.show_stabilizers = !editor_state.stabilizers.is_empty();
            graph_state.needs_rerender = true;
            StabilizersOutcome::Computed(editor_state.stabilizers.len())
        }
        Err(err) => StabilizersOutcome::Failed(err),
    }
}

fn apply_validation_result_to_tab(
    tabs: &mut EditorTabs,
    done: ValidationJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded validation result for a closed tab");
        return;
    };
    match apply_validation_outcome(
        &mut tab.snapshot.compile_ui,
        &mut tab.snapshot.graph_state,
        done,
    ) {
        ValidationOutcome::Stale => {
            notifications.push_warn(format!("Validation result is stale for {}", tab.title));
        }
        ValidationOutcome::Passed => {
            notifications.push_info(format!("Validation passed for {}", tab.title));
        }
        ValidationOutcome::Failed(err) => {
            notifications.push_error(format!("Validation failed for {}: {}", tab.title, err));
        }
    }
}

fn apply_stabilizers_result_to_tab(
    tabs: &mut EditorTabs,
    done: StabilizersJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded stabilizer result for a closed tab");
        return;
    };
    match apply_stabilizers_outcome(
        &mut tab.snapshot.editor_state,
        &mut tab.snapshot.graph_state,
        done,
    ) {
        StabilizersOutcome::Stale => {
            notifications.push_warn(format!(
                "Discarded stale stabilizer result for {}",
                tab.title
            ));
        }
        StabilizersOutcome::Computed(count) => {
            notifications.push_info(format!("Computed {} stabilizers for {}", count, tab.title));
        }
        StabilizersOutcome::Failed(err) => {
            notifications.push_error_report(
                &format!("Failed to compute stabilizers for {}", tab.title),
                &err,
            );
        }
    }
}

fn apply_parse_blog_result_to_tab(
    tabs: &mut EditorTabs,
    done: ParseBlogJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded BLOG parse result for a closed tab");
        return;
    };
    if done.revision != tab.snapshot.graph_state.revision
        || done.source != tab.snapshot.import_export.bloq_buffer
    {
        notifications.push_warn(format!(
            "Discarded stale BLOG parse result for {}",
            tab.title
        ));
        return;
    }
    match done.result {
        Ok(parsed) => {
            install_graph_into_tab(tab, parsed.graph, Some(parsed.source_graph), done.title);
            notifications.push_info(format!("Loaded graph into {}", tab.title));
        }
        Err(err) => {
            notifications.push_error_report("Failed to parse BLOG buffer", &err);
        }
    }
}

/// Installs a graph produced for a background tab into that tab's snapshot,
/// resetting the dependent state exactly as [`replace_graph_with_cleanup`] does
/// for the active tab, and framing the camera it will be restored with.
pub(super) fn install_graph_into_tab(
    tab: &mut crate::resources::EditorTab,
    graph: BlockGraph,
    program: Option<bloq_graph::BlockGraph>,
    title: Option<String>,
) {
    tab.snapshot.graph_state.pending_branch_arm = None;
    tab.snapshot.graph_state.source_graph = program.map(std::sync::Arc::new);
    tab.snapshot.graph_state.graph = graph;
    tab.snapshot.graph_state.commit();
    tab.snapshot
        .editor_state
        .sync_after_graph_edit(&mut tab.snapshot.graph_state);
    tab.snapshot.target_state = TargetState::default();
    tab.snapshot.compile_ui.reset_for_new_graph();
    tab.snapshot.circuit_viewer.reset_for_new_graph();
    tab.snapshot.camera_settings = crate::systems::camera::camera_settings_for_graph(
        &tab.snapshot.graph_state.graph,
        tab.snapshot.editor_state.pipe_length,
    );
    if let Some(title) = title {
        tab.set_title(title);
    }
}

fn apply_compile_result_to_tab(
    tabs: &mut EditorTabs,
    done: CompileJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded compile result for a closed tab");
        return;
    };
    finalize_compile_result(
        done,
        tab.snapshot.graph_state.revision,
        &mut tab.snapshot.compile_ui,
        notifications,
    );
}

fn apply_viewer_compile_result_to_tab(
    tabs: &mut EditorTabs,
    done: ViewerCompileJobResult,
    notifications: &mut Notifications,
) {
    let Some(tab) = tabs.tabs.iter_mut().find(|tab| tab.id == done.tab_id) else {
        notifications.push_warn("Discarded Bloq viewer result for a closed tab");
        return;
    };
    finalize_viewer_compile_result(
        done,
        tab.snapshot.graph_state.revision,
        &mut tab.snapshot.circuit_viewer,
        notifications,
    );
}

/// Polls every background job once per frame, applying finished results to the
/// active tab's live resources or to a background tab's snapshot, and
/// discarding results that a newer graph revision has made stale.
fn poll_editor_jobs_system(
    mut jobs: ResMut<EditorJobs>,
    mut tabs: ResMut<EditorTabs>,
    mut editor_state: ResMut<EditorState>,
    mut graph_state: ResMut<GraphState>,
    import_export: Res<ImportExportState>,
    mut notifications: ResMut<Notifications>,
    mut compile_ui: ResMut<CompileUiState>,
    mut circuit_viewer: ResMut<BloqViewerState>,
    mut zx_viewer: ResMut<ZxViewerState>,
    mut target_state: ResMut<TargetState>,
    mut ui_intents: ResMut<UiIntentBuffer>,
    camera_query: Single<(&mut Transform, &mut CameraSettings), With<EditorCamera>>,
) {
    let (mut camera_transform, mut camera_settings) = camera_query.into_inner();

    cancel_invalidated_compilation(
        &mut jobs,
        &mut tabs,
        &graph_state,
        &editor_state,
        &mut compile_ui,
        &mut circuit_viewer,
        &mut notifications,
    );
    cancel_invalidated_zx(&mut jobs, &tabs, &graph_state, &zx_viewer);

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(done) = poll_task(&mut jobs.zx_simplification) {
        jobs.zx_request = None;
        apply_zx_simplification_result(done, &mut tabs, &graph_state, &mut zx_viewer);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some((tab_id, key)) = jobs.web_zx_simplification
        && let Some(result) = web_compilation::poll_zx_simplification()
    {
        jobs.web_zx_simplification = None;
        apply_zx_simplification_result(
            ZxSimplificationResult {
                tab_id,
                key,
                result: result.map_err(eyre::Report::msg),
            },
            &mut tabs,
            &graph_state,
            &mut zx_viewer,
        );
    }

    if let Some(done) = poll_task(&mut jobs.validation) {
        if done.tab_id != tabs.active {
            apply_validation_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            match apply_validation_outcome(&mut compile_ui, &mut graph_state, done) {
                ValidationOutcome::Stale => {
                    notifications.push_warn("Validation result is stale due to newer graph edits");
                }
                ValidationOutcome::Passed => notifications.push_info("Validation passed"),
                ValidationOutcome::Failed(err) => {
                    notifications.push_error(format!("Validation failed: {}", err));
                }
            }
        }
    }

    if let Some(done) = poll_task(&mut jobs.stabilizers) {
        if done.tab_id != tabs.active {
            apply_stabilizers_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            match apply_stabilizers_outcome(&mut editor_state, &mut graph_state, done) {
                StabilizersOutcome::Stale => {
                    notifications.push_warn("Discarded stale stabilizer result");
                }
                StabilizersOutcome::Computed(count) => {
                    notifications.push_info(format!("Computed {} stabilizers", count));
                }
                StabilizersOutcome::Failed(err) => {
                    notifications.push_error_report("Failed to compute stabilizers", &err);
                }
            }
        }
    }

    if let Some(done) = poll_task(&mut jobs.import_blog_file) {
        match done {
            Ok(Some(file)) => ui_intents.push(UiIntent::LoadBlogFile {
                title: file.title,
                buffer: file.buffer,
            }),
            Ok(None) => {}
            Err(err) => notifications.push_error_report("Failed to read BLOG file", &err),
        }
    }

    if let Some(done) = poll_task(&mut jobs.parse_blog) {
        if done.tab_id != tabs.active {
            apply_parse_blog_result_to_tab(&mut tabs, done, &mut notifications);
        } else if done.revision != graph_state.revision || done.source != import_export.bloq_buffer
        {
            notifications.push_warn("Discarded stale BLOG parse result");
        } else {
            match done.result {
                Ok(parsed) => {
                    graph_state.pending_branch_arm = None;
                    graph_state.source_graph = Some(std::sync::Arc::new(parsed.source_graph));
                    replace_graph_with_cleanup(
                        &mut graph_state,
                        &mut editor_state,
                        &mut target_state,
                        &mut compile_ui,
                        &mut circuit_viewer,
                        parsed.graph,
                    );
                    reset_camera_to_graph(
                        &mut camera_transform,
                        &mut camera_settings,
                        &graph_state.graph,
                        editor_state.pipe_length,
                    );
                    notifications.push_info("Loaded graph from buffer successfully");
                    if let Some(title) = done.title {
                        tabs.active_tab_mut().set_title(title);
                    }
                }
                Err(err) => {
                    notifications.push_error_report("Failed to parse BLOG buffer", &err);
                }
            }
        }
    }

    // Parsing may have replaced the source since the first cancellation check.
    cancel_invalidated_compilation(
        &mut jobs,
        &mut tabs,
        &graph_state,
        &editor_state,
        &mut compile_ui,
        &mut circuit_viewer,
        &mut notifications,
    );
    if let Some(done) = poll_task(&mut jobs.compile) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            jobs.progress = None;
        }
        if done.tab_id != tabs.active {
            apply_compile_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            finalize_compile_result(
                done,
                graph_state.revision,
                &mut compile_ui,
                &mut notifications,
            );
        }
    }

    if let Some(done) = poll_task(&mut jobs.viewer_compile) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            jobs.progress = None;
        }
        if done.tab_id != tabs.active {
            apply_viewer_compile_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            finalize_viewer_compile_result(
                done,
                graph_state.revision,
                &mut circuit_viewer,
                &mut notifications,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    if jobs
        .web_compile
        .as_ref()
        .is_some_and(|job| job.result.is_some())
    {
        if let Some(mut job) = jobs.web_compile.take() {
            let result = job.result.take().expect("checked worker result");
            if job.viewer {
                let view = result.map_err(eyre::Report::msg).and_then(|result| {
                    let config = compilation::config_from_request(&job.request)?;
                    compilation::build_view_from_artifacts(
                        &job.source,
                        &job.request,
                        result.bloq,
                        config,
                        result.compile_duration,
                    )
                });
                let done = ViewerCompileJobResult {
                    tab_id: job.tab_id,
                    revision: job.revision,
                    result: view,
                };
                if done.tab_id != tabs.active {
                    apply_viewer_compile_result_to_tab(&mut tabs, done, &mut notifications);
                } else {
                    finalize_viewer_compile_result(
                        done,
                        graph_state.revision,
                        &mut circuit_viewer,
                        &mut notifications,
                    );
                }
            } else {
                let output = result
                    .map_err(eyre::Report::msg)
                    .and_then(|result| compilation::export_download(result.bloq, &job.request));
                let done = CompileJobResult {
                    tab_id: job.tab_id,
                    revision: job.revision,
                    result: output,
                };
                if done.tab_id != tabs.active {
                    apply_compile_result_to_tab(&mut tabs, done, &mut notifications);
                } else {
                    finalize_compile_result(
                        done,
                        graph_state.revision,
                        &mut compile_ui,
                        &mut notifications,
                    );
                }
            }
        }
    } else if jobs.web_compile.is_some()
        && let Some(result) = web_compilation::poll()
    {
        jobs.web_compile.as_mut().expect("checked web job").result = Some(result);
    }

    if let Some(done) = poll_task(&mut jobs.concurrent_ops) {
        if done.tab_id != tabs.active {
            apply_concurrent_ops_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            finalize_concurrent_ops_result(done, &mut circuit_viewer, &mut notifications);
        }
    }

    if let Some(done) = poll_task(&mut jobs.detslice) {
        if done.tab_id != tabs.active {
            apply_detslice_result_to_tab(&mut tabs, done, &mut notifications);
        } else {
            finalize_detslice_result(done, &mut circuit_viewer, &mut notifications);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    schedule_pending_concurrent_ops_job(
        &mut jobs,
        &mut tabs,
        &mut circuit_viewer,
        &mut notifications,
    );
    #[cfg(not(target_arch = "wasm32"))]
    schedule_pending_detslice_job(
        &mut jobs,
        &mut tabs,
        &mut circuit_viewer,
        &mut notifications,
    );
}

// ============================================================================
// Compile and validation UI state
// ============================================================================

const VALIDATION_READY_MESSAGE: &str = "Run validation to check graph validity.";

/// Result of the background structural-validation job for the current graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub(crate) enum ValidationState {
    Unknown,
    Pending,
    Passed,
    Failed,
}

/// The download format the compile panel emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, serde::Serialize, serde::Deserialize)]
pub(crate) enum CompileOutputFormat {
    Stim,
    /// The human-readable Bloq IR text exchange format (`.bloqir`).
    IrText,
    /// The compact binary Bloq IR exchange format (`.bloq`).
    IrBinary,
}

impl CompileOutputFormat {
    /// Every format, in menu order.
    pub(crate) const ALL: [Self; 3] = [Self::Stim, Self::IrText, Self::IrBinary];

    /// The file extension (without the dot) for the downloaded output.
    pub(crate) const fn extension(self) -> &'static str {
        match self {
            Self::Stim => "stim",
            Self::IrText => bloq_ir::BLOQ_TEXT_EXTENSION,
            Self::IrBinary => bloq_ir::BLOQ_BINARY_EXTENSION,
        }
    }

    /// The human-readable name shown in the format selector.
    pub(crate) const fn display_label(self) -> &'static str {
        match self {
            Self::Stim => "Stim",
            Self::IrText => "IR text",
            Self::IrBinary => "IR binary",
        }
    }
}

/// Progress of the background compile job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub(crate) enum CompileExecutionState {
    Idle,
    Pending,
    Succeeded,
    Failed,
    Cancelled,
}

/// The parameters of one compile request.
#[derive(Debug, Clone)]
pub(crate) struct CompileRequest {
    pub(crate) code_distance: u32,
    pub(crate) format: CompileOutputFormat,
    pub(crate) prepare_t_with_mpps: bool,
}

/// UI state for the compile panel: validation status, code distance input, output format, and
/// compilation progress.
#[derive(Resource, Debug, Clone)]
pub(crate) struct CompileUiState {
    pub(crate) validation_state: ValidationState,
    pub(crate) validation_message: String,
    pub(crate) validated_revision: Option<u64>,
    pub(crate) pending_revision: Option<u64>,
    pub(crate) code_distance_input: String,
    pub(crate) format: CompileOutputFormat,
    pub(crate) prepare_t_with_mpps: bool,
    pub(crate) compile_state: CompileExecutionState,
    pub(crate) compile_message: String,
}

impl CompileUiState {
    /// Parses the distance input and assembles a [`CompileRequest`].
    ///
    /// # Errors
    ///
    /// Returns a user-facing message if the distance field is empty, not a
    /// number, or not a distance the compiler accepts. The last check defers to
    /// [`CompileConfig::is_valid_distance`] rather than restating the rule, so
    /// the panel rejects `d = 4` here instead of letting a background job fail
    /// with it minutes later.
    pub(crate) fn build_request(&self) -> Result<CompileRequest, String> {
        let trimmed = self.code_distance_input.trim();
        if trimmed.is_empty() {
            return Err("Compile distance d is required".to_string());
        }
        let code_distance = trimmed
            .parse::<u32>()
            .map_err(|_| format!("Invalid compile distance '{trimmed}'"))?;
        if !CompileConfig::is_valid_distance(code_distance) {
            return Err(format!(
                "Compile distance d must be odd and in 3..={MAX_CODE_DISTANCE}, got {code_distance}"
            ));
        }
        Ok(CompileRequest {
            code_distance,
            format: self.format,
            prepare_t_with_mpps: self.prepare_t_with_mpps,
        })
    }

    pub(crate) fn mark_validation_pending(&mut self, revision: u64) {
        self.validation_state = ValidationState::Pending;
        self.validation_message = "Validation in progress...".to_string();
        self.validated_revision = None;
        self.pending_revision = Some(revision);
    }

    /// Downgrades validation status to stale when the graph has advanced past
    /// the validated (or in-flight) revision, so the UI never shows a pass that
    /// no longer applies.
    pub(crate) fn mark_stale_if_graph_changed(&mut self, current_revision: u64) {
        if self
            .pending_revision
            .is_some_and(|revision| revision != current_revision)
        {
            let stale_pending_message = "Validation is running on an older revision and will be discarded if edits continue.";
            if self.validation_message != stale_pending_message {
                self.validation_message = stale_pending_message.to_string();
            }
            return;
        }
        if self.validated_revision != Some(current_revision) {
            let stale_message = "Validation is stale because graph changed. Re-run validation.";
            if self.validation_state != ValidationState::Unknown
                || self.validation_message != stale_message
                || self.validated_revision.is_some()
            {
                self.validation_state = ValidationState::Unknown;
                self.validation_message = stale_message.to_string();
                self.validated_revision = None;
            }
        }
    }

    fn mark_compile_pending(&mut self) {
        self.compile_state = CompileExecutionState::Pending;
        self.compile_message = "Compilation in progress...".to_string();
    }

    fn mark_compile_succeeded(&mut self, message: impl Into<String>) {
        self.compile_state = CompileExecutionState::Succeeded;
        self.compile_message = message.into();
    }

    fn mark_compile_failed(&mut self, message: impl Into<String>) {
        self.compile_state = CompileExecutionState::Failed;
        self.compile_message = message.into();
    }

    fn mark_compile_cancelled(&mut self, message: impl Into<String>) {
        self.compile_state = CompileExecutionState::Cancelled;
        self.compile_message = message.into();
    }

    /// Clears validation and compile status after the graph is replaced
    /// wholesale (load, import, clear).
    pub(crate) fn reset_for_new_graph(&mut self) {
        self.validation_state = ValidationState::Unknown;
        self.validation_message = VALIDATION_READY_MESSAGE.to_string();
        self.validated_revision = None;
        self.pending_revision = None;
        self.compile_state = CompileExecutionState::Idle;
        self.compile_message = "Graph changed. Compile again to download output.".to_string();
    }
}

impl Default for CompileUiState {
    fn default() -> Self {
        Self {
            validation_state: ValidationState::Unknown,
            validation_message: VALIDATION_READY_MESSAGE.to_string(),
            validated_revision: None,
            pending_revision: None,
            code_distance_input: "3".to_string(),
            format: CompileOutputFormat::Stim,
            prepare_t_with_mpps: false,
            compile_state: CompileExecutionState::Idle,
            compile_message: "Click Compile to download the generated output.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompileJobResult, CompileProgress, CompiledDownload, DetsliceJobResult, EditorJobs,
        ParseBlogJobResult, ParsedBlog, ValidationJobResult, ValidationOutcome,
        apply_parse_blog_result_to_tab, apply_validation_outcome, decode_imported_blog_file,
        finalize_compile_result, parse_blog_for_editor, replace_graph_with_cleanup_and_selection,
        request_concurrent_ops, request_detslice, schedule_stabilizers_job,
        validate_graph_for_editor, warn_spatial_hadamard_distance,
    };
    use crate::components::GraphElement;
    use crate::resources::{
        BloqViewerState, CompileExecutionState, CompileUiState, DetsliceData, EditorState,
        EditorTabId, EditorTabs, GraphState, Notifications, SliceVisibility, TargetState,
        ToastLevel, ZxViewerState,
    };
    use crate::systems::ui::circuit_viewer::LazyToggle;
    use bevy::tasks::{AsyncComputeTaskPool, TaskPool, futures_lite::future};
    use bloq_compile::CompileStage;
    use bloq_graph::{
        Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryItem, Pipe, StabilizerGenerator,
    };
    use bloq_ir::Bloq;
    use glam::ivec3;
    use std::sync::Arc;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cancelling_an_active_or_background_compile_stops_work_and_allows_a_fresh_request() {
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        for background in [false, true] {
            let mut tabs = EditorTabs::default();
            let target = tabs.active;
            let mut ui = CompileUiState::default();
            let mut viewer = BloqViewerState::default();
            if background {
                tabs.active_tab_mut()
                    .snapshot
                    .compile_ui
                    .mark_compile_pending();
                tabs.active_tab_mut()
                    .snapshot
                    .circuit_viewer
                    .begin_compile(0);
                let other = tabs.add_empty_tab(&EditorState::default());
                tabs.set_active(other);
            } else {
                ui.mark_compile_pending();
                viewer.begin_compile(0);
            }
            let progress = CompileProgress::new(target, 0);
            let cancellation = progress.cancellation.clone();
            let (started, ready) = std::sync::mpsc::channel();
            let (resume, resumed) = std::sync::mpsc::channel();
            let (finished, stopped) = std::sync::mpsc::channel();
            let worker_token = cancellation.clone();
            let mut jobs = EditorJobs {
                progress: Some(progress),
                compile: Some(pool.spawn(async move {
                    let result = worker_token.run(|| {
                        started.send(()).unwrap();
                        resumed.recv().unwrap();
                        worker_token.check()?;
                        Ok(CompiledDownload::empty())
                    });
                    finished
                        .send(result.as_ref().is_err_and(super::is_cancellation))
                        .unwrap();
                    CompileJobResult {
                        tab_id: target,
                        revision: 0,
                        result,
                    }
                })),
                validation: Some(pool.spawn(std::future::pending())),
                ..Default::default()
            };
            ready.recv().unwrap();
            let mut notifications = Notifications::default();
            super::cancel_editor_compilation(
                &mut jobs,
                target,
                &mut tabs,
                &mut ui,
                &mut viewer,
                &mut notifications,
            );
            resume.send(()).unwrap();
            assert!(stopped.recv().unwrap());
            assert!(cancellation.is_cancelled());
            assert!(!jobs.compilation_running());
            assert!(jobs.validation_running(), "unrelated jobs are retained");
            if background {
                let tab = tabs.tabs.iter().find(|tab| tab.id == target).unwrap();
                assert_eq!(
                    tab.snapshot.compile_ui.compile_state,
                    CompileExecutionState::Cancelled
                );
                assert!(tab.snapshot.circuit_viewer.pending_revision.is_none());
                assert_eq!(ui.compile_state, CompileExecutionState::Idle);
            } else {
                assert_eq!(ui.compile_state, CompileExecutionState::Cancelled);
                assert!(viewer.pending_revision.is_none());
            }
            assert!(
                notifications
                    .toasts
                    .iter()
                    .all(|toast| toast.level != ToastLevel::Error)
            );
            let graph = GraphState {
                graph: GalleryItem::CNOT.build(),
                ..Default::default()
            };
            super::schedule_viewer_compile_job(
                &mut jobs,
                tabs.active,
                &graph,
                &mut viewer,
                &mut notifications,
                ui.build_request().unwrap(),
            );
            assert!(!jobs.progress.as_ref().unwrap().cancellation.is_cancelled());
            let done = future::block_on(jobs.viewer_compile.take().unwrap());
            assert!(done.result.is_ok(), "the next request uses a fresh token");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn stale_compile_sources_modes_and_closed_tabs_cancel_their_own_request() {
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        for reason in ["revision", "mode", "closed"] {
            let mut tabs = EditorTabs::default();
            let target = tabs.active;
            let progress = CompileProgress::new(target, 0);
            let token = progress.cancellation.clone();
            let mut jobs = EditorJobs {
                progress: Some(progress),
                viewer_compile: Some(pool.spawn(std::future::pending())),
                ..Default::default()
            };
            let mut graph = GraphState::default();
            let mut editor = EditorState {
                mode: crate::resources::EditorMode::Bloq,
                ..Default::default()
            };
            if reason == "revision" {
                graph.revision = 1;
            }
            if reason == "mode" {
                editor.mode = crate::resources::EditorMode::View;
            }
            if reason == "closed" {
                tabs.remove(target, &editor);
            }
            let mut ui = CompileUiState::default();
            let mut viewer = BloqViewerState::default();
            viewer.begin_compile(0);
            super::cancel_invalidated_compilation(
                &mut jobs,
                &mut tabs,
                &graph,
                &editor,
                &mut ui,
                &mut viewer,
                &mut Notifications::default(),
            );
            assert!(token.is_cancelled(), "{reason}");
            assert!(!jobs.compilation_running());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn zx_cancellation_releases_changed_revision_mode_seed_and_closed_views() {
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        for reason in ["revision", "mode", "seed", "closed"] {
            let tabs = EditorTabs::default();
            let target = tabs.active;
            let mut viewer = ZxViewerState::default();
            viewer.open = true;
            viewer.set_simplified(true);
            let mut key = viewer.view_key(0);
            let mut graph = GraphState::default();
            if reason == "revision" {
                graph.revision = 1;
            }
            if reason == "mode" {
                viewer.set_simplified(false);
            }
            if reason == "seed" {
                key.2 = 1;
            }
            if reason == "closed" {
                viewer.open = false;
            }
            let token = bloq_graph::CancellationToken::new();
            let mut jobs = EditorJobs {
                zx_request: Some((target, key, token.clone())),
                zx_simplification: Some(pool.spawn(std::future::pending())),
                progress: Some(CompileProgress::new(target, 0)),
                compile: Some(pool.spawn(std::future::pending())),
                ..Default::default()
            };
            super::cancel_invalidated_zx(&mut jobs, &tabs, &graph, &viewer);
            assert!(token.is_cancelled(), "{reason}");
            assert!(!jobs.zx_simplification_running());
            assert!(
                jobs.compilation_running(),
                "ZX cancellation preserves compilation"
            );
            super::schedule_zx_simplification(
                &mut jobs,
                target,
                &GraphState {
                    graph: GalleryItem::CNOT.build(),
                    ..Default::default()
                },
                (0, true, 0),
            )
            .unwrap();
            let done = future::block_on(jobs.zx_simplification.take().unwrap());
            assert!(done.result.is_ok(), "a fresh simplification succeeds");
        }
    }

    #[test]
    fn typed_cancellation_is_displayed_without_a_semantic_error() {
        let mut ui = CompileUiState::default();
        ui.mark_compile_pending();
        let mut viewer = BloqViewerState::default();
        viewer.begin_compile(0);
        let mut notifications = Notifications::default();
        finalize_compile_result(
            CompileJobResult {
                tab_id: EditorTabId::new(1),
                revision: 0,
                result: Err(bloq_graph::ComputationCancelled.into()),
            },
            0,
            &mut ui,
            &mut notifications,
        );
        super::finalize_viewer_compile_result(
            super::ViewerCompileJobResult {
                tab_id: EditorTabId::new(1),
                revision: 0,
                result: Err(bloq_graph::ComputationCancelled.into()),
            },
            0,
            &mut viewer,
            &mut notifications,
        );
        assert_eq!(ui.compile_state, CompileExecutionState::Cancelled);
        assert!(viewer.pending_revision.is_none());
        assert!(viewer.last_error.is_none());
        assert!(
            notifications
                .toasts
                .iter()
                .all(|toast| toast.level != ToastLevel::Error)
        );
    }

    #[test]
    fn zx_results_return_to_the_requesting_tab_and_ignore_closed_tabs() {
        for background in [false, true] {
            let mut tabs = EditorTabs::default();
            let target = tabs.active;
            let mut viewer = ZxViewerState::default();
            viewer.open = true;
            viewer.set_simplified(true);
            let key = viewer.view_key(0);
            if background {
                tabs.active_tab_mut().snapshot.zx_viewer = viewer.without_cache();
                let other = tabs.add_empty_tab(&EditorState::default());
                tabs.set_active(other);
                viewer = ZxViewerState::default();
            }
            let result = || super::ZxSimplificationResult {
                tab_id: target,
                key,
                result: Err(color_eyre::eyre::eyre!("current failure")),
            };
            assert!(super::apply_zx_simplification_result(
                result(),
                &mut tabs,
                &GraphState::default(),
                &mut viewer
            ));
            if background {
                tabs.remove(target, &EditorState::default());
            } else {
                viewer.open = false;
            }
            assert!(!super::apply_zx_simplification_result(
                result(),
                &mut tabs,
                &GraphState::default(),
                &mut viewer
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn zx_simplification_keeps_only_one_native_job_in_flight() {
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut jobs = EditorJobs {
            zx_simplification: Some(pool.spawn(async {
                super::ZxSimplificationResult {
                    tab_id: EditorTabId::new(1),
                    key: (0, true, 0),
                    result: Err(color_eyre::eyre::eyre!("existing job")),
                }
            })),
            ..Default::default()
        };
        super::schedule_zx_simplification(
            &mut jobs,
            EditorTabId::new(2),
            &GraphState::default(),
            (0, true, 1),
        )
        .unwrap();
        let done = future::block_on(jobs.zx_simplification.take().unwrap());
        assert_eq!(done.tab_id, EditorTabId::new(1));
        assert!(matches!(done.result, Err(ref error) if error.to_string() == "existing job"));
    }

    #[test]
    fn unfinished_branch_capture_blocks_export_and_viewer_compilation() {
        use crate::resources::PendingBranchArm;

        let mut graph_state = GraphState {
            graph: GalleryItem::CNOT.build(),
            pending_branch_arm: Some(PendingBranchArm {
                name: "draft".into(),
                captured_true: false,
                arm: bloq_graph::BranchArm::new(vec![], vec![]),
            }),
            ..Default::default()
        };
        let mut jobs = EditorJobs::default();
        let mut compile_ui = CompileUiState::default();
        let mut viewer = BloqViewerState::default();
        let mut notifications = Notifications::default();
        let tab = EditorTabId::new(1);
        let request = compile_ui.build_request().unwrap();
        // No task pool/Worker is initialized: incomplete documents must stop
        // before either platform launches work or marks a result pending.
        super::schedule_compile_job(
            &mut jobs,
            tab,
            &graph_state,
            &mut compile_ui,
            &mut notifications,
            request,
        );
        super::start_viewer_compile(
            &mut jobs,
            tab,
            &graph_state,
            &compile_ui,
            &mut viewer,
            &mut notifications,
        );
        assert!(!jobs.compilation_running());
        assert!(viewer.pending_revision.is_none());
        assert!(notifications.contains_message("Finish or cancel the captured branch arm first"));
        graph_state.pending_branch_arm = None;
        assert!(super::accept_compile_request(
            &jobs,
            &graph_state,
            &mut notifications
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_jobs_execute_on_a_worker_thread() {
        let caller = std::thread::current().id();
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let job = pool.spawn(async { std::thread::current().id() });
        assert_ne!(
            future::block_on(job),
            caller,
            "CPU jobs must not run on the editor thread"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compile_progress_tracks_stage_and_only_its_source_revision() {
        let tab = EditorTabId::new(1);
        let other_tab = EditorTabId::new(2);
        let progress = CompileProgress::new(tab, 7);
        let mut jobs = EditorJobs {
            compile: Some(
                AsyncComputeTaskPool::get_or_init(TaskPool::new)
                    .spawn(async { std::future::pending::<CompileJobResult>().await }),
            ),
            progress: Some(progress.clone()),
            ..Default::default()
        };
        (progress.observer())(CompileStage::Readouts);
        assert_eq!(
            jobs.compilation_progress(tab, 7).unwrap().0,
            "Lowering readouts"
        );
        assert!(jobs.compilation_progress(other_tab, 7).is_none());
        assert!(jobs.compilation_progress(tab, 8).is_none());
        jobs.compile = None;
        jobs.progress = None;
        assert!(jobs.compilation_progress(tab, 7).is_none());
    }

    #[test]
    fn stabilizer_job_uses_the_composed_adder_in_its_display_coordinates() {
        let parsed = parse_blog_for_editor(GalleryItem::ThreeBitAdder.entry().blog()).unwrap();
        let graph = crate::utils::displayed_branch_projection(&parsed.graph).unwrap();
        let state = GraphState {
            graph: parsed.graph,
            source_graph: Some(Arc::new(parsed.source_graph)),
            revision: 1,
            ..Default::default()
        };
        assert!(state.is_composed());
        let generators = run_stabilizers_job_for_state(state);
        let measured = graph
            .actions()
            .into_iter()
            .filter_map(|action| match action {
                bloq_graph::Action::Measure { name, .. } => Some(name),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let shown = generators
            .iter()
            .filter_map(|row| row.measurement_name().map(str::to_owned))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(shown.contains("bit0_uma__m_ikprime"));
        assert_eq!(shown, measured, "every named readout has a planned surface");
        let outputs = bloq_graph::ZXGraph::try_from(&graph)
            .unwrap()
            .output_ports();
        for generator in generators.iter().filter(|row| row.is_measurement()) {
            assert!(
                outputs
                    .iter()
                    .all(|position| !generator.stabilizer.port_stabilizer.contains_key(position)),
                "every planned readout closes before live outputs"
            );
        }
        for generator in &generators {
            bloq_graph::stabilizer_as_gltf_data(generator, &graph, 2.0)
                .expect("the composed surface fits the displayed graph");
        }
    }

    #[test]
    fn stabilizer_result_requires_the_requested_layer_and_view_scope() {
        use crate::resources::EditorMode;

        let mut graph = GraphState::default();
        for (mode, layer_only, layer) in [
            (EditorMode::View, true, 5),
            (EditorMode::View, false, 4),
            (EditorMode::Edit, true, 4),
            (EditorMode::View, true, 4),
        ] {
            let mut editor = EditorState {
                mode,
                view_current_layer_only: layer_only,
                plane_height: layer,
                ..Default::default()
            };
            let outcome = super::apply_stabilizers_outcome(
                &mut editor,
                &mut graph,
                super::StabilizersJobResult {
                    tab_id: EditorTabId::new(1),
                    revision: 0,
                    layer: Some(4),
                    result: Ok(Vec::new()),
                },
            );
            assert_eq!(
                matches!(outcome, super::StabilizersOutcome::Computed(0)),
                mode == EditorMode::View && layer_only && layer == 4,
            );
        }
    }

    /// The panel rejects exactly what the compiler rejects. Before this
    /// deferred to `CompileConfig::is_valid_distance`, `d = 4` and `d = 1000`
    /// were accepted here and failed asynchronously from a background job.
    #[test]
    fn build_request_accepts_only_compilable_distances() {
        let request_for = |input: &str| {
            CompileUiState {
                code_distance_input: input.to_string(),
                ..Default::default()
            }
            .build_request()
            .map(|request| request.code_distance)
        };

        assert_eq!(request_for(" 3 "), Ok(3));
        assert_eq!(request_for("255"), Ok(255));
        for rejected in ["", "d", "0", "1", "2", "4", "256", "1000"] {
            assert!(
                request_for(rejected).is_err(),
                "distance {rejected:?} must not reach a compile job"
            );
        }
    }

    fn run_stabilizers_job(graph: BlockGraph) -> Vec<StabilizerGenerator> {
        run_stabilizers_job_for_state(GraphState {
            graph,
            revision: 1,
            ..Default::default()
        })
    }

    fn run_stabilizers_job_for_state(graph_state: GraphState) -> Vec<StabilizerGenerator> {
        let _ = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut jobs = EditorJobs::default();
        let mut notifications = Notifications::default();

        schedule_stabilizers_job(
            &mut jobs,
            EditorTabId::new(1),
            &graph_state,
            false,
            0,
            &mut notifications,
        );

        let done = future::block_on(async {
            loop {
                if let Some(done) = jobs.poll_stabilizers_for_test() {
                    break done;
                }
                future::yield_now().await;
            }
        });

        done.result.expect("stabilizers should compute")
    }

    #[test]
    fn imported_blog_file_uses_stem_and_requires_utf8() {
        let file =
            decode_imported_blog_file("bell.blog", b"BLOG 1.0".to_vec()).expect("valid BLOG file");

        assert_eq!(file.title.as_deref(), Some("bell"));
        assert_eq!(file.buffer, "BLOG 1.0");
        assert!(decode_imported_blog_file("bad.blog", vec![0xff]).is_err());
    }

    #[test]
    fn editor_parser_keeps_module_ownership_for_resolved_graphs() {
        let source = crate::utils::one_bit_adder_fixture().to_blog_text();
        let parsed = parse_blog_for_editor(&source).expect("resolved modular BLOG loads");

        assert_eq!(
            crate::resources::ModuleViewState::from_graph(&parsed.source_graph, &parsed.graph)
                .unwrap()
                .modules()
                .iter()
                .map(|module| module.name.as_str())
                .collect::<Vec<_>>(),
            ["And", "Maj", "Uma"]
        );
        assert!(!parsed.graph.is_empty());
    }

    #[test]
    fn editor_parser_keeps_root_geometry_without_module_highlights() {
        let source = GalleryItem::BellState.build().to_blog_text();
        let parsed = parse_blog_for_editor(&source).expect("root-only BLOG loads");
        assert!(
            crate::resources::ModuleViewState::from_graph(&parsed.source_graph, &parsed.graph,)
                .is_none()
        );
        assert_eq!(parsed.source_graph.root().name, "main");
        assert!(!parsed.graph.is_empty());
    }

    #[test]
    fn spatial_hadamard_compile_warning_reaches_notifications() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::IVec3::ZERO,
            BlockKind::Cube(CubeKind::XZX),
        ));
        graph.add_block(Block::new(glam::IVec3::X, BlockKind::Cube(CubeKind::XXZ)));
        graph.add_pipe(Pipe::new(glam::IVec3::ZERO, Direction::XPLUS).with_hadamard());

        let mut notifications = Notifications::default();
        warn_spatial_hadamard_distance(&graph, &mut notifications);
        assert_eq!(notifications.toasts.len(), 1);
        assert_eq!(notifications.toasts[0].level, ToastLevel::Warn);
        assert!(
            notifications.toasts[0]
                .message
                .contains("fixed-bulk spatial Hadamard")
        );
    }

    fn stabilizer_signatures(stabilizers: &[StabilizerGenerator]) -> Vec<String> {
        let mut signatures: Vec<String> = stabilizers
            .iter()
            .map(|generator| {
                format!(
                    "{}|measurement={}",
                    generator.stabilizer.paulis,
                    generator.is_measurement()
                )
            })
            .collect();
        signatures.sort();
        signatures
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn request_detslice_while_job_busy_keeps_pending_intent() {
        let _ = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let pool = AsyncComputeTaskPool::get();
        let mut jobs = EditorJobs {
            detslice: Some(pool.spawn(async {
                DetsliceJobResult {
                    tab_id: EditorTabId::new(99),
                    generation: 1,
                    result: DetsliceData::default(),
                }
            })),
            ..Default::default()
        };
        let mut viewer = BloqViewerState {
            program: Some(Arc::new(Bloq::new())),
            compile_revision: Some(2),
            code_distance: Some(3),
            compile_config: Some(bloq_compile::CompileConfig::default()),
            ..Default::default()
        };
        let mut notifications = Notifications::default();

        request_detslice(
            &mut jobs,
            EditorTabId::new(1),
            &mut viewer,
            &mut notifications,
            SliceVisibility {
                detectors: true,
                observables: false,
            },
        );

        assert_eq!(
            viewer.detslice_pending,
            Some(SliceVisibility {
                detectors: true,
                observables: false,
            })
        );
        assert!(!viewer.slice_visibility.any());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn rejected_viewer_pins_keep_the_current_view_and_stale_views_cannot_pin() {
        let _ = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let original = Arc::new(
            Bloq::from_text("BLOQIR 1\ngraph {\n n0 compute 0 from selector choice\n}\n").unwrap(),
        );
        let mut viewer = BloqViewerState {
            program: Some(original.clone()),
            source_program: Some(original.clone()),
            compile_revision: Some(7),
            compile_generation: 1,
            code_distance: Some(3),
            compile_config: Some(bloq_compile::CompileConfig::default()),
            compile_duration: Some(Default::default()),
            ..Default::default()
        };
        let mut jobs = EditorJobs::default();
        let mut notifications = Notifications::default();
        let pins = std::collections::BTreeMap::from([("choice".to_owned(), true)]);
        super::request_viewer_branch_pins(
            &mut jobs,
            EditorTabId::new(1),
            7,
            &mut viewer,
            &mut notifications,
            Some(pins),
        );
        assert_eq!(viewer.pending_revision, Some(7));
        let done = future::block_on(jobs.viewer_compile.take().unwrap());
        assert!(matches!(
            done.result
                .as_ref()
                .unwrap_err()
                .downcast_ref::<bloq_ir::MembershipPinError>(),
            Some(bloq_ir::MembershipPinError::UnreachableAssignment)
        ));
        super::finalize_viewer_compile_result(done, 7, &mut viewer, &mut notifications);
        assert!(Arc::ptr_eq(viewer.program.as_ref().unwrap(), &original));
        assert_eq!(viewer.compile_generation, 1);
        assert!(viewer.branch_pins.is_none());
        assert!(viewer.pending_revision.is_none());
        assert!(viewer.last_error.is_some());
        super::request_viewer_branch_pins(
            &mut jobs,
            EditorTabId::new(1),
            8,
            &mut viewer,
            &mut notifications,
            None,
        );
        assert!(jobs.viewer_compile.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn first_concurrent_ops_toggle_starts_background_job() {
        let _ = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut jobs = EditorJobs::default();
        let mut viewer = BloqViewerState {
            program: Some(Arc::new(Bloq::new())),
            compile_revision: Some(1),
            code_distance: Some(3),
            ..Default::default()
        };

        request_concurrent_ops(
            &mut jobs,
            EditorTabId::new(1),
            &mut viewer,
            &mut Notifications::default(),
            true,
        );

        assert_eq!(viewer.concurrent_ops_toggle(), LazyToggle::Pending);
        assert!(jobs.concurrent_ops.is_some());
    }

    #[test]
    fn replacement_can_install_transformed_selection_before_sync() {
        let mut graph_state = GraphState::default();
        graph_state
            .graph
            .add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        let mut editor_state = EditorState::default();
        editor_state.select_element(GraphElement::Block(ivec3(0, 0, 0)), false);

        let mut next_graph = BlockGraph::new();
        next_graph.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        replace_graph_with_cleanup_and_selection(
            &mut graph_state,
            &mut editor_state,
            &mut TargetState::default(),
            &mut CompileUiState::default(),
            &mut BloqViewerState::default(),
            next_graph,
            [GraphElement::Block(ivec3(1, 0, 0))],
        );

        assert!(editor_state.is_selected(GraphElement::Block(ivec3(1, 0, 0))));
        assert_eq!(editor_state.selection_count(), 1);
    }

    #[test]
    fn stale_viewer_compile_failure_is_discarded_before_finalization() {
        let mut viewer = BloqViewerState::default();
        viewer.begin_compile(3);
        let mut notifications = Notifications::default();

        super::finalize_viewer_compile_result(
            super::ViewerCompileJobResult {
                tab_id: EditorTabId::new(1),
                revision: 3,
                result: Err(color_eyre::eyre::eyre!("older source failure")),
            },
            4,
            &mut viewer,
            &mut notifications,
        );

        assert!(viewer.pending_revision.is_none());
        assert!(viewer.last_error.is_none());
        assert_eq!(notifications.toasts.len(), 1);
        assert_eq!(notifications.toasts[0].level, ToastLevel::Warn);
        assert!(notifications.toasts[0].message.contains("stale"));
    }

    #[test]
    fn stale_compile_download_result_is_discarded_before_finalization() {
        let mut compile_ui = CompileUiState::default();
        compile_ui.mark_compile_pending();
        let mut notifications = Notifications::default();

        finalize_compile_result(
            CompileJobResult {
                tab_id: EditorTabId::new(1),
                revision: 3,
                result: Ok(CompiledDownload::empty()),
            },
            4,
            &mut compile_ui,
            &mut notifications,
        );

        assert_eq!(compile_ui.compile_state, CompileExecutionState::Cancelled);
        assert!(compile_ui.compile_message.contains("older graph revision"));
        assert_eq!(notifications.toasts.len(), 1);
        assert_eq!(notifications.toasts[0].level, ToastLevel::Warn);
        assert!(notifications.toasts[0].message.contains("stale"));
    }

    #[test]
    fn active_blog_load_clears_capture_and_undo_restores_it() {
        use crate::resources::{ImportExportState, PendingBranchArm};
        use bevy::prelude::*;
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut app = App::new();
        app.init_resource::<EditorJobs>()
            .init_resource::<EditorTabs>()
            .init_resource::<EditorState>()
            .init_resource::<GraphState>()
            .init_resource::<ImportExportState>()
            .init_resource::<Notifications>()
            .init_resource::<CompileUiState>()
            .init_resource::<BloqViewerState>()
            .init_resource::<ZxViewerState>()
            .init_resource::<TargetState>()
            .init_resource::<super::UiIntentBuffer>()
            .add_systems(Update, super::poll_editor_jobs_system);
        app.world_mut().spawn((
            super::EditorCamera,
            Transform::default(),
            super::CameraSettings::default(),
        ));
        {
            let mut state = app.world_mut().resource_mut::<GraphState>();
            state
                .graph
                .add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
            state.pending_branch_arm = Some(PendingBranchArm {
                name: "b0".into(),
                captured_true: false,
                arm: bloq_graph::BranchArm::new(
                    vec![Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ))],
                    vec![Pipe::new(IVec3::ZERO, Direction::ZPLUS)],
                ),
            });
            state.commit();
        }
        let revision = app.world().resource::<GraphState>().revision;
        let tab_id = app.world().resource::<EditorTabs>().active;
        let mut loaded = BlockGraph::new();
        loaded.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XZZ)));
        app.world_mut().resource_mut::<EditorJobs>().parse_blog = Some(pool.spawn(async move {
            ParseBlogJobResult {
                tab_id,
                revision,
                source: String::new(),
                title: None,
                result: Ok(ParsedBlog {
                    source_graph: loaded.clone().with_inferred_interface().unwrap(),
                    graph: loaded,
                }),
            }
        }));
        future::block_on(async {
            loop {
                app.update();
                if app.world().resource::<EditorJobs>().parse_blog.is_none() {
                    break;
                }
                future::yield_now().await;
            }
        });
        let mut state = app.world_mut().resource_mut::<GraphState>();
        assert!(state.pending_branch_arm.is_none());
        assert!(state.graph.has_block_at(IVec3::X));
        state.undo();
        assert!(state.pending_branch_arm.is_some());
        assert!(state.graph.has_block_at(IVec3::ZERO));
    }

    #[test]
    fn stale_blog_parse_result_does_not_replace_edited_graph() {
        let mut tabs = EditorTabs::default();
        let tab_id = tabs.active;
        let tab = tabs.active_tab_mut();
        tab.snapshot
            .graph_state
            .graph
            .add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        tab.snapshot.graph_state.commit();
        let mut parsed = BlockGraph::new();
        parsed.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::XZZ)));
        let mut notifications = Notifications::default();

        apply_parse_blog_result_to_tab(
            &mut tabs,
            ParseBlogJobResult {
                tab_id,
                revision: 0,
                source: String::new(),
                title: None,
                result: Ok(ParsedBlog {
                    source_graph: parsed.clone().with_inferred_interface().unwrap(),
                    graph: parsed,
                }),
            },
            &mut notifications,
        );

        assert!(
            tabs.active_tab()
                .snapshot
                .graph_state
                .graph
                .get_block(ivec3(0, 0, 0))
                .is_some()
        );
        assert!(notifications.toasts[0].message.contains("stale"));
    }

    #[test]
    fn newer_blog_draft_discards_parse_result_at_same_graph_revision() {
        let mut tabs = EditorTabs::default();
        let tab_id = tabs.active;
        tabs.active_tab_mut().snapshot.import_export.bloq_buffer = "newer draft".to_string();
        let mut parsed = BlockGraph::new();
        parsed.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::XZZ)));
        let mut notifications = Notifications::default();

        apply_parse_blog_result_to_tab(
            &mut tabs,
            ParseBlogJobResult {
                tab_id,
                revision: 0,
                source: "older draft".to_string(),
                title: None,
                result: Ok(ParsedBlog {
                    source_graph: parsed.clone().with_inferred_interface().unwrap(),
                    graph: parsed,
                }),
            },
            &mut notifications,
        );

        assert!(tabs.active_tab().snapshot.graph_state.graph.is_empty());
        assert!(notifications.toasts[0].message.contains("stale"));
    }

    #[test]
    fn schedule_stabilizers_job_projects_displayed_ccz_arms() {
        let mut graph = GalleryItem::CCZGateTeleport.build().flatten().unwrap();
        graph.set_shown_branch_arm("b0", false).unwrap();
        let assignments = graph
            .branch_definitions()
            .iter()
            .map(|branch| (branch.target, branch.shown_true()))
            .collect::<Vec<_>>();
        let expected = graph
            .planned_surfaces(&assignments, bloq_graph::ModuleCertificationLimits::DEFAULT)
            .unwrap();
        let shown = graph.project_branches(assignments).unwrap();
        let actual = run_stabilizers_job(graph);

        assert_eq!(
            stabilizer_signatures(&actual),
            stabilizer_signatures(&expected)
        );
        for generator in &actual {
            bloq_graph::stabilizer_as_gltf_data(generator, &shown, 2.0)
                .expect("the planned surface fits the displayed arm");
        }
    }

    #[test]
    fn validation_installs_derived_action_surfaces_for_its_revision() {
        let mut graph_state = GraphState {
            graph: GalleryItem::T.build().flatten().unwrap(),
            ..Default::default()
        };
        let actions = graph_state.graph.actions();
        graph_state.graph.set_actions_lenient(actions).unwrap();
        assert!(!graph_state.graph.action_graph().is_analyzed());
        let graph = validate_graph_for_editor(graph_state.graph.clone()).unwrap();
        let mut compile_ui = CompileUiState::default();

        assert!(matches!(
            apply_validation_outcome(
                &mut compile_ui,
                &mut graph_state,
                ValidationJobResult {
                    tab_id: EditorTabId::new(1),
                    revision: 0,
                    result: Ok(graph),
                },
            ),
            ValidationOutcome::Passed
        ));
        assert!(
            graph_state
                .graph
                .action_graph()
                .ordered_nodes()
                .filter(|node| matches!(node.action, bloq_graph::Action::Measure { .. }))
                .all(|node| node.measurement_stabilizer.is_some())
        );
    }
}
