//! Worker-backed IndexedDB recovery. Never replace work created while loading.
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

use super::*;
use crate::components::EditorCamera;
use crate::resources::Notifications;

const LEGACY_KEY: &str = "bloq.editor.session.v1";

pub(crate) struct SessionPlugin;
impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, initialize)
            .add_systems(Last, capture_session);
    }
}

struct PendingSave {
    worker: Worker,
    session: Option<Session>,
    in_flight: Option<Session>,
    revision: Option<String>,
    ready: bool,
    paused: bool,
    dirty: bool,
    hiding: bool,
    error: Option<String>,
    retry: super::retry::SaveRetry,
    clock: web_time::Instant,
}

impl PendingSave {
    fn flush(&mut self, hiding: bool) {
        if !self.ready || self.paused || !self.dirty || self.in_flight.is_some() {
            return;
        }
        let now = self.clock.elapsed().as_secs_f64();
        if !self.retry.should_attempt(now, hiding) {
            return;
        }
        let result = (|| -> Result<(), String> {
            let session = self.session.as_ref().ok_or("Missing recovery snapshot")?;
            // Conversion/structured clone still copies strings; JSON escaping and
            // IndexedDB I/O happen in the worker, never in the rendering frame.
            let snapshot = session
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .map_err(|error| error.to_string())?;
            let message = js_sys::Object::new();
            set(&message, "operation", &JsValue::from_str("save"));
            set(&message, "snapshot", &snapshot);
            set(
                &message,
                "expectedRevision",
                &self
                    .revision
                    .as_deref()
                    .map(JsValue::from_str)
                    .unwrap_or(JsValue::NULL),
            );
            self.worker.post_message(&message).map_err(js_error)?;
            self.in_flight = Some(session.clone());
            Ok(())
        })();
        if let Err(error) = result {
            self.retry.failed(now);
            self.error = Some(error);
        }
    }
}

struct BrowserSession {
    pending: Rc<RefCell<PendingSave>>,
    messages: Rc<RefCell<VecDeque<JsValue>>>,
    legacy: Result<Option<String>, String>,
    guard: super::RecoveryGuard,
    last_flush: f64,
    window: web_sys::Window,
    document: web_sys::Document,
    listener: Closure<dyn FnMut(web_sys::Event)>,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
    _on_message_error: Closure<dyn FnMut(web_sys::Event)>,
}
impl Drop for BrowserSession {
    fn drop(&mut self) {
        let callback = self.listener.as_ref().unchecked_ref();
        let _ = self
            .document
            .remove_event_listener_with_callback("visibilitychange", callback);
        let _ = self
            .window
            .remove_event_listener_with_callback("pagehide", callback);
        let pending = self.pending.borrow();
        pending.worker.set_onmessage(None);
        pending.worker.set_onerror(None);
        pending.worker.set_onmessageerror(None);
        pending.worker.terminate();
    }
}
fn js_error(error: JsValue) -> String {
    format!("{error:?}")
}
fn field(value: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(value, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}
fn set(object: &js_sys::Object, key: &str, value: &JsValue) {
    js_sys::Reflect::set(object, &JsValue::from_str(key), value).expect("plain recovery object");
}

fn initialize(world: &mut World) {
    match start(world) {
        Ok(browser) => {
            world.insert_non_send(browser);
            world
                .resource_mut::<Notifications>()
                .push_info("Checking saved browser session in the background");
        }
        Err(error) => world.resource_mut::<Notifications>().push_error(format!(
            "Browser recovery unavailable: {error}. Existing saves kept unchanged; export new work."
        )),
    }
}
fn start(world: &World) -> Result<BrowserSession, String> {
    let window = web_sys::window().ok_or("Missing browser window")?;
    let document = window.document().ok_or("Missing document")?;
    let legacy = window
        .local_storage()
        .map_err(js_error)
        .and_then(|storage| {
            storage
                .map(|storage| storage.get_item(LEGACY_KEY).map_err(js_error))
                .transpose()
                .map(Option::flatten)
        });
    let options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let worker = Worker::new_with_options("./recovery_worker.js", &options).map_err(js_error)?;
    let messages = Rc::new(RefCell::new(VecDeque::new()));
    let pending = Rc::new(RefCell::new(PendingSave {
        worker: worker.clone(),
        session: None,
        in_flight: None,
        revision: None,
        ready: false,
        paused: false,
        dirty: false,
        hiding: false,
        error: None,
        retry: super::retry::SaveRetry::default(),
        clock: web_time::Instant::now(),
    }));
    let state = Rc::clone(&pending);
    let queue = Rc::clone(&messages);
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let message = event.data();
        if field(&message, "kind").as_string().as_deref() == Some("saved") {
            let mut pending = state.borrow_mut();
            pending.revision = field(&message, "revision").as_string();
            let saved = pending.in_flight.take();
            pending.dirty = saved != pending.session;
            pending.retry.changed_or_saved();
            if pending.hiding {
                // A succeeded: submit newer state even if rendering stopped.
                // This is not another retry of a failed hide-time snapshot.
                pending.flush(false);
            }
        } else {
            queue.borrow_mut().push_back(message);
        }
    });
    let state = Rc::clone(&pending);
    let on_error = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
        let mut state = state.borrow_mut();
        state.paused = true;
        state.error = Some(format!("Recovery worker failed: {}", event.message()));
    });
    let state = Rc::clone(&pending);
    let on_message_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
        let mut state = state.borrow_mut();
        state.paused = true;
        state.error = Some("Recovery worker sent unreadable data".into());
    });
    worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    worker.set_onmessageerror(Some(on_message_error.as_ref().unchecked_ref()));
    let state = Rc::clone(&pending);
    let visible = document.clone();
    let listener = Closure::<dyn FnMut(web_sys::Event)>::new(move |event: web_sys::Event| {
        if event.type_() == "pagehide" || visible.hidden() {
            let mut state = state.borrow_mut();
            state.hiding = true;
            state.flush(true);
        } else {
            let mut state = state.borrow_mut();
            state.hiding = false;
            state.retry.visible();
        }
    });
    let browser = BrowserSession {
        pending,
        messages,
        legacy,
        guard: super::RecoveryGuard::capture(world),
        last_flush: 0.0,
        window,
        document,
        listener,
        _on_message: on_message,
        _on_error: on_error,
        _on_message_error: on_message_error,
    };
    // Own callbacks before registration: Drop also unwinds partial setup.
    let callback = browser.listener.as_ref().unchecked_ref();
    browser
        .document
        .add_event_listener_with_callback("visibilitychange", callback)
        .map_err(js_error)?;
    browser
        .window
        .add_event_listener_with_callback("pagehide", callback)
        .map_err(js_error)?;
    Ok(browser)
}

fn install_recovery(
    world: &mut World,
    browser: &BrowserSession,
    payload: &str,
) -> Result<Option<Session>, String> {
    let (tabs, cache) = super::recovery::restore_prepared_with_cache(payload)?;
    super::install_recovered(world, tabs, cache, &browser.guard, "browser")
}

fn capture_session(world: &mut World) {
    let Some(mut browser) = world.remove_non_send::<BrowserSession>() else {
        return;
    };
    let messages = std::mem::take(&mut *browser.messages.borrow_mut());
    for message in messages {
        let mut pending = browser.pending.borrow_mut();
        match field(&message, "kind").as_string().as_deref() {
            Some("ready") => {
                let request = js_sys::Object::new();
                set(&request, "operation", &JsValue::from_str("load"));
                match &browser.legacy {
                    Ok(legacy) => set(
                        &request,
                        "legacy",
                        &legacy
                            .as_deref()
                            .map(JsValue::from_str)
                            .unwrap_or(JsValue::NULL),
                    ),
                    Err(error) => set(&request, "legacyError", &JsValue::from_str(error)),
                }
                if let Err(error) = pending.worker.post_message(&request) {
                    pending.paused = true;
                    pending.error = Some(js_error(error));
                }
            }
            Some("loaded") => {
                if let Some(warning) = field(&message, "warning").as_string() {
                    world.resource_mut::<Notifications>().push_warn(warning);
                }
                let prepared = field(&message, "prepared").as_string();
                let result = prepared
                    .as_deref()
                    .map(|payload| install_recovery(world, &browser, payload))
                    .transpose();
                match result {
                    Ok(cache) => {
                        pending.session = cache.flatten();
                        pending.revision = field(&message, "revision").as_string();
                        pending.ready = true;
                    }
                    Err(error) => {
                        pending.paused = true;
                        pending.error = Some(error);
                    }
                }
            }
            Some("error") => {
                pending.in_flight = None;
                pending.paused |= field(&message, "conflict").as_bool().unwrap_or(false)
                    || field(&message, "operation").as_string().as_deref() == Some("load");
                let now = pending.clock.elapsed().as_secs_f64();
                pending.retry.failed(now);
                pending.error = Some(
                    field(&message, "message")
                        .as_string()
                        .unwrap_or_else(|| "Recovery failed".into()),
                );
            }
            _ => {
                pending.paused = true;
                pending.error = Some("Invalid recovery worker response".into());
            }
        }
    }
    let now = browser.pending.borrow().clock.elapsed().as_secs_f64();
    {
        let mut pending = browser.pending.borrow_mut();
        if pending.ready && !pending.paused {
            let camera = world
                .query_filtered::<&CameraSettings, With<EditorCamera>>()
                .iter(world)
                .next()
                .copied();
            if let Some(camera) = camera {
                let source = TabSource {
                    graph: world.resource(),
                    editor: world.resource(),
                    import: world.resource(),
                    compile: world.resource(),
                    camera: &camera,
                };
                match Session::capture(
                    world.resource::<EditorTabs>(),
                    source,
                    pending.session.as_ref(),
                ) {
                    Ok(session) if pending.session.as_ref() != Some(&session) => {
                        pending.session = Some(session);
                        pending.dirty = true;
                        pending.retry.changed_or_saved();
                    }
                    Ok(_) => {}
                    Err(error) => pending.error = Some(error.to_string()),
                }
                if now - browser.last_flush >= 0.5 {
                    pending.flush(false);
                    browser.last_flush = now;
                }
            }
        }
    }
    if let Some(error) = browser.pending.borrow_mut().error.take() {
        let message = format!(
            "Browser autosave failed: {error}. Previous saves kept; export BLOG to keep new work."
        );
        if !world.resource::<Notifications>().contains_message(&message) {
            world.resource_mut::<Notifications>().push_error(message);
        }
    }
    world.insert_non_send(browser);
}
