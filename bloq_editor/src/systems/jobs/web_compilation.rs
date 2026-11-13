//! Run browser compilation in a module Worker so progress can paint between stages.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use bloq_compile::{CompileConfig, CompileContext, CompileStage};
use bloq_graph::BlockGraph;
use bloq_ir::Bloq;
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, closure::Closure, prelude::wasm_bindgen};
use web_sys::{
    DedicatedWorkerGlobalScope, ErrorEvent, Event, MessageEvent, Worker, WorkerOptions, WorkerType,
};

use super::CompileRequest;
use crate::systems::ui::zx_viewer::{
    SimplifiedZxView, decode_zx_worker_source, decode_zx_worker_view, encode_zx_worker_source,
    encode_zx_worker_view, simplified_zx_graph,
};

pub(super) struct WorkerResult {
    pub(super) bloq: Bloq,
    pub(super) compile_duration: Duration,
}

thread_local! {
    // Browser UI work is single-threaded; Bevy's EditorJobs must remain Send + Sync.
    static CURRENT: RefCell<Option<WorkerTask>> = const { RefCell::new(None) };
    static CURRENT_ZX: RefCell<Option<WorkerTask>> = const { RefCell::new(None) };
    static CURRENT_BLOG: RefCell<Option<WorkerTask>> = const { RefCell::new(None) };
    static IDLE_BLOG: RefCell<Option<WorkerTask>> = const { RefCell::new(None) };
    static CURRENT_BLOG_IMPORT: RefCell<Option<WorkerTask>> = const { RefCell::new(None) };
}

pub(super) fn start(source: &BlockGraph, request: &CompileRequest) -> Result<(), String> {
    CURRENT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("Compilation is already running".into());
        }
        let message = js_sys::Object::new();
        set(
            &message,
            "source",
            &JsValue::from_str(&source.to_blog_text()),
        );
        set(
            &message,
            "codeDistance",
            &JsValue::from_f64(request.code_distance as f64),
        );
        set(
            &message,
            "prepareTWithMpps",
            &JsValue::from_bool(request.prepare_t_with_mpps),
        );
        *slot = Some(WorkerTask::start(message)?);
        Ok(())
    })
}

pub(super) fn poll() -> Option<Result<WorkerResult, String>> {
    CURRENT
        .with(|slot| WorkerTask::poll(&mut slot.borrow_mut()))
        .map(|result| result.and_then(decode_compilation))
}

pub(super) fn stage() -> Option<String> {
    CURRENT.with(|slot| slot.borrow().as_ref().and_then(WorkerTask::stage))
}

pub(super) fn cancel() {
    CURRENT.with(|slot| {
        slot.borrow_mut().take();
    });
}

pub(super) fn start_zx_simplification(graph: &BlockGraph, seed: u64) -> Result<(), String> {
    CURRENT_ZX.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("ZX simplification is already running".into());
        }
        let message = js_sys::Object::new();
        set(&message, "operation", &JsValue::from_str("simplify-zx"));
        let source = encode_zx_worker_source(graph)
            .map_err(|error| format!("Cannot send ZX source: {error:#}"))?;
        set(&message, "source", &JsValue::from_str(&source));
        // wasm-bindgen's u64 parameter is a BigInt, not a lossy JS Number.
        set(&message, "seed", &JsValue::from_str(&seed.to_string()));
        *slot = Some(WorkerTask::start(message)?);
        Ok(())
    })
}

pub(super) fn poll_zx_simplification() -> Option<Result<SimplifiedZxView, String>> {
    CURRENT_ZX
        .with(|slot| WorkerTask::poll(&mut slot.borrow_mut()))
        .map(|result| {
            result.and_then(|message| {
                let json = field(&message, "graphJson")
                    .as_string()
                    .ok_or("ZX worker returned an invalid graph")?;
                decode_zx_worker_view(&json)
                    .map_err(|error| format!("Cannot read simplified ZX view: {error:#}"))
            })
        })
}

pub(super) fn cancel_zx_simplification() {
    CURRENT_ZX.with(|slot| {
        slot.borrow_mut().take();
    });
}

/// Independent of compilation and ZX jobs. Replacing a stale request must
/// cancel its worker first so old text never consumes the next result slot.
pub(super) fn start_live_blog(
    source: &str,
    revision: u64,
    baseline: Option<&str>,
) -> Result<(), String> {
    CURRENT_BLOG.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("BLOG analysis is already running".into());
        }
        let request = blog_request(source, revision, baseline);
        let idle = IDLE_BLOG.with(|idle| idle.borrow_mut().take());
        *slot = Some(match idle {
            Some(worker) if worker.state.borrow().result.is_none() => {
                worker.send(request)?;
                worker
            }
            _ => WorkerTask::start(request)?,
        });
        Ok(())
    })
}

fn start_blog(
    slot: &mut Option<WorkerTask>,
    source: &str,
    revision: u64,
    baseline: Option<&str>,
) -> Result<(), String> {
    if slot.is_some() {
        return Err("BLOG analysis is already running".into());
    }
    *slot = Some(WorkerTask::start(blog_request(source, revision, baseline))?);
    Ok(())
}

fn blog_request(source: &str, revision: u64, baseline: Option<&str>) -> js_sys::Object {
    let message = js_sys::Object::new();
    set(&message, "operation", &JsValue::from_str("analyze-blog"));
    set(&message, "source", &JsValue::from_str(source));
    set(
        &message,
        "revision",
        &JsValue::from_str(&revision.to_string()),
    );
    if let Some(baseline) = baseline {
        set(&message, "baseline", &JsValue::from_str(baseline));
    }
    message
}

pub(super) fn poll_live_blog() -> Option<Result<super::LiveBlogAnalysis, String>> {
    CURRENT_BLOG.with(|slot| {
        let mut slot = slot.borrow_mut();
        let result = slot.as_ref()?.state.borrow_mut().result.take()?;
        let worker = slot.take();
        let analysis = result.and_then(decode_blog);
        // A parse diagnostic is a successful analysis. Transport/runtime errors
        // discard the worker; only a fully consumed response can become idle.
        if analysis.is_ok() {
            IDLE_BLOG.with(|idle| *idle.borrow_mut() = worker);
        }
        Some(analysis)
    })
}

fn poll_blog(slot: &mut Option<WorkerTask>) -> Option<Result<super::LiveBlogAnalysis, String>> {
    WorkerTask::poll(slot).map(|result| result.and_then(decode_blog))
}

fn decode_blog(message: JsValue) -> Result<super::LiveBlogAnalysis, String> {
    let json = field(&message, "analysisJson")
        .as_string()
        .ok_or("BLOG worker returned invalid analysis")?;
    super::live_blog_transport::decode(&json)
}

/// Explicit imports must not contend with the live-draft worker slot.
pub(super) fn start_blog_import(source: &str) -> Result<(), String> {
    CURRENT_BLOG_IMPORT.with(|slot| start_blog(&mut slot.borrow_mut(), source, 0, None))
}

pub(super) fn poll_blog_import() -> Option<Result<super::ParsedBlog, String>> {
    CURRENT_BLOG_IMPORT
        .with(|slot| poll_blog(&mut slot.borrow_mut()))
        .map(|result| {
            result
                .and_then(|analysis| analysis.result)
                .and_then(|result| match result {
                    super::LiveBlogResult::Changed(parsed) => Ok(*parsed),
                    super::LiveBlogResult::Unchanged => {
                        Err("Import worker unexpectedly returned no graph".into())
                    }
                })
        })
}

pub(super) fn cancel_live_blog() {
    CURRENT_BLOG.with(|slot| {
        slot.borrow_mut().take();
    });
}

fn decode_compilation(message: JsValue) -> Result<WorkerResult, String> {
    let bytes: Uint8Array = field(&message, "bytes")
        .dyn_into()
        .map_err(|_| "Compiler returned invalid bytes".to_owned())?;
    let bloq = Bloq::from_binary(&bytes.to_vec())
        .map_err(|error| format!("Cannot read compiled program: {error}"))?;
    let seconds = field(&message, "durationSeconds")
        .as_f64()
        .ok_or("Compiler returned invalid duration")?;
    let compile_duration =
        Duration::try_from_secs_f64(seconds).map_err(|_| "Compiler returned invalid duration")?;
    Ok(WorkerResult {
        bloq,
        compile_duration,
    })
}

struct State {
    stage: Option<String>,
    result: Option<Result<JsValue, String>>,
}

/// A browser worker with at most one request in flight. Drop terminates it.
struct WorkerTask {
    worker: Worker,
    state: Rc<RefCell<State>>,
    // Keep callbacks alive until Drop detaches the listeners.
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
    _on_message_error: Closure<dyn FnMut(Event)>,
}

impl WorkerTask {
    fn start(request: js_sys::Object) -> Result<Self, String> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options("./compile_worker.js", &options)
            .map_err(|error| format!("Cannot start background worker: {error:?}"))?;
        let state = Rc::new(RefCell::new(State {
            stage: None,
            result: None,
        }));
        let on_message = {
            let state = Rc::clone(&state);
            let worker = worker.clone();
            let mut initial_request = Some(request);
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                let data = event.data();
                match field(&data, "kind").as_string().as_deref() {
                    Some("ready") => {
                        if let Some(request) = initial_request.take()
                            && let Err(error) = worker.post_message(&request)
                        {
                            state.borrow_mut().result =
                                Some(Err(format!("Cannot send worker request: {error:?}")));
                        }
                    }
                    Some("stage") => state.borrow_mut().stage = field(&data, "name").as_string(),
                    Some("complete") => {
                        state.borrow_mut().result = Some(Ok(data));
                    }
                    Some("error") => {
                        state.borrow_mut().result = Some(Err(field(&data, "message")
                            .as_string()
                            .unwrap_or_else(|| "Background job failed".into())));
                    }
                    _ => state.borrow_mut().result = Some(Err("Invalid worker message".into())),
                }
            })
        };
        let on_error = {
            let state = Rc::clone(&state);
            Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
                state.borrow_mut().result = Some(Err(format!(
                    "Background worker failed: {}",
                    event.message()
                )));
            })
        };
        let on_message_error = {
            let state = Rc::clone(&state);
            Closure::<dyn FnMut(Event)>::new(move |_: Event| {
                state.borrow_mut().result =
                    Some(Err("Background worker sent an unreadable message".into()));
            })
        };
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        worker.set_onmessageerror(Some(on_message_error.as_ref().unchecked_ref()));
        Ok(Self {
            worker,
            state,
            _on_message: on_message,
            _on_error: on_error,
            _on_message_error: on_message_error,
        })
    }

    /// Only used after a completed response has been consumed. The initial
    /// ready callback already ran, so there is no startup handshake to repeat.
    fn send(&self, request: js_sys::Object) -> Result<(), String> {
        self.state.borrow_mut().stage = None;
        self.worker
            .post_message(&request)
            .map_err(|error| format!("Cannot send worker request: {error:?}"))
    }

    fn poll(slot: &mut Option<Self>) -> Option<Result<JsValue, String>> {
        let result = slot.as_ref()?.state.borrow_mut().result.take()?;
        slot.take();
        Some(result)
    }

    fn stage(&self) -> Option<String> {
        self.state.borrow().stage.clone()
    }
}

impl Drop for WorkerTask {
    fn drop(&mut self) {
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        self.worker.set_onmessageerror(None);
        self.worker.terminate();
    }
}

fn field(object: &JsValue, name: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
}

fn set(object: &js_sys::Object, name: &str, value: &JsValue) {
    Reflect::set(object, &JsValue::from_str(name), value).expect("set plain request property");
}

fn post_stage(name: &str) {
    let worker: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let message = js_sys::Object::new();
    set(&message, "kind", &JsValue::from_str("stage"));
    set(&message, "name", &JsValue::from_str(name));
    let _ = worker.post_message(&message);
}

/// Called by `compile_worker.js` after the shared WASM module initializes.
#[wasm_bindgen]
#[expect(
    unreachable_pub,
    reason = "wasm-bindgen exports require pub even from a private module"
)]
pub fn compile_in_worker(
    source: &str,
    code_distance: u32,
    prepare_t_with_mpps: bool,
) -> Result<Array, JsValue> {
    let config = CompileConfig::try_new(code_distance)
        .map_err(|error| JsValue::from_str(&error.to_string()))?
        .with_prepare_t_with_mpps(prepare_t_with_mpps);
    let context = CompileContext::new(config)
        .with_progress_observer(|stage: CompileStage| post_stage(&stage.to_string()));
    let graph =
        BlockGraph::from_text(source).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let artifacts = context
        .compile(&graph)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    post_stage("Transferring compiled program");
    let result = Array::new();
    result.push(&Uint8Array::from(artifacts.bloq.to_binary().as_slice()));
    result.push(&JsValue::from_f64(artifacts.compile_duration.as_secs_f64()));
    Ok(result)
}

/// Called by the shared worker for the expensive logical ZX viewer path.
#[wasm_bindgen]
#[expect(
    unreachable_pub,
    reason = "wasm-bindgen exports require pub from a private module"
)]
pub fn simplify_zx_in_worker(source: &str, seed: u64) -> Result<String, JsValue> {
    let graph = decode_zx_worker_source(source)
        .map_err(|error| JsValue::from_str(&format!("{error:#}")))?;
    post_stage("Simplifying ZX graph");
    let graph = simplified_zx_graph(&graph, seed)
        .map_err(|error| JsValue::from_str(&format!("{error:#}")))?;
    post_stage("Preparing ZX layout");
    let view = SimplifiedZxView::new(graph);
    encode_zx_worker_view(&view).map_err(|error| JsValue::from_str(&format!("{error:#}")))
}

/// Performs language analysis, parsing, module materialization and face repair
/// in a real browser Worker, not the browser's single-threaded async executor.
#[wasm_bindgen]
#[expect(
    unreachable_pub,
    reason = "wasm-bindgen exports require pub from a private module"
)]
pub fn analyze_blog_in_worker(
    source: &str,
    revision: u64,
    baseline: Option<String>,
) -> Result<String, JsValue> {
    post_stage("Analyzing BLOG");
    let analysis = super::analyze_live_blog_against(source, revision, baseline.as_deref());
    post_stage("Preparing BLOG result");
    super::live_blog_transport::encode(analysis).map_err(|error| JsValue::from_str(&error))
}
