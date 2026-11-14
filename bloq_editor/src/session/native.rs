//! Native session recovery. Keep one writer, atomically replace saves, and retain
//! the preceding successful file. Invalid or externally changed saves stay intact.
use super::*;
use crate::components::EditorCamera;
use crate::resources::Notifications;
use color_eyre::eyre::WrapErr;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) struct SessionPlugin;
impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, restore_session)
            .add_systems(Last, (poll_recovery, capture_session).chain());
    }
}

fn session_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("BLOQ_EDITOR_SESSION_PATH") {
        let path = PathBuf::from(path);
        ensure!(
            path.is_absolute(),
            "BLOQ_EDITOR_SESSION_PATH must be absolute"
        );
        return Ok(path);
    }
    let base = if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME").wrap_err("HOME is unavailable")?)
            .join("Library/Application Support")
    } else if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("LOCALAPPDATA").wrap_err("LOCALAPPDATA is unavailable")?)
    } else if let Some(path) = std::env::var_os("XDG_STATE_HOME") {
        let path = PathBuf::from(path);
        ensure!(path.is_absolute(), "XDG_STATE_HOME must be absolute");
        path
    } else {
        PathBuf::from(std::env::var_os("HOME").wrap_err("HOME is unavailable")?)
            .join(".local/state")
    };
    Ok(base.join("bloq-editor/session-v1.json"))
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

struct NativeSession {
    path: PathBuf,
    // The OS releases this lock even if the process crashes; no stale lock cleanup.
    _lock: File,
    expected: Option<String>,
    session: Option<Session>,
    dirty: bool,
    paused: bool,
}

impl NativeSession {
    fn open(path: PathBuf) -> Result<Self> {
        fs::create_dir_all(path.parent().wrap_err("Missing session directory")?)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("lock"))?;
        lock.try_lock()
            .wrap_err("Another editor is using this session")?;
        let expected = read_optional(&path)?;
        Ok(Self {
            path,
            _lock: lock,
            expected,
            session: None,
            dirty: false,
            paused: false,
        })
    }

    fn flush(&mut self) -> Result<()> {
        if !self.dirty || self.paused {
            return Ok(());
        }
        if read_optional(&self.path)? != self.expected {
            self.paused = true;
            color_eyre::eyre::bail!("Session changed outside this editor; autosave paused");
        }
        let json =
            serde_json::to_string(self.session.as_ref().wrap_err("Missing session snapshot")?)?;
        if self.expected.as_ref() != Some(&json) {
            // A backup failure prevents replacement, rather than sacrificing recovery.
            if let Some(previous) = &self.expected {
                atomic_write(&self.path.with_extension("previous.json"), previous)?;
            }
            atomic_write(&self.path, &json)?;
            self.expected = Some(json);
        }
        self.dirty = false;
        Ok(())
    }
}

/// Only the worker serializes and writes during editing. At most one snapshot
/// is in flight; newer frames coalesce into `latest` without a growing queue.
#[derive(Resource)]
struct NativeAutosave {
    backend: Option<NativeSession>,
    worker: Option<std::thread::JoinHandle<(NativeSession, Result<(), String>)>>,
    latest: Option<Session>,
}

impl NativeAutosave {
    fn new(backend: NativeSession) -> Self {
        Self {
            latest: backend.session.clone(),
            backend: Some(backend),
            worker: None,
        }
    }

    fn finish(&mut self, wait: bool) -> Result<(), String> {
        if self
            .worker
            .as_ref()
            .is_some_and(|worker| wait || worker.is_finished())
        {
            let (backend, result) = self
                .worker
                .take()
                .expect("worker checked above")
                .join()
                .map_err(|_| "Autosave worker failed; restart before saving again".to_owned())?;
            self.backend = Some(backend);
            result?;
        }
        Ok(())
    }

    fn prepare_capture(&mut self, exiting: bool) -> Result<(), String> {
        match self.finish(exiting) {
            // A prior background write may have failed transiently. On exit,
            // still capture and synchronously retry the newest state. Conflicts
            // and lost workers remain paused rather than overwriting anything.
            Err(_) if exiting && self.backend.as_ref().is_some_and(|backend| !backend.paused) => {
                Ok(())
            }
            result => result,
        }
    }

    fn save_latest(&mut self, exiting: bool) -> Result<(), String> {
        let Some(mut backend) = self.backend.take() else {
            return Ok(());
        };
        if backend.paused || (!backend.dirty && backend.session == self.latest) {
            self.backend = Some(backend);
            return Ok(());
        }
        backend.session = self.latest.clone();
        backend.dirty = true;
        if exiting {
            let result = backend
                .flush()
                .map_err(|error| format!("{}: {error:#}", backend.path.display()));
            self.backend = Some(backend);
            result
        } else {
            self.worker = Some(std::thread::spawn(move || {
                let result = backend
                    .flush()
                    .map_err(|error| format!("{}: {error:#}", backend.path.display()));
                (backend, result)
            }));
            Ok(())
        }
    }
}

fn atomic_write(path: &Path, text: &str) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| -> Result<()> {
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

type RecoveryWorker = std::thread::JoinHandle<Result<(NativeSession, Option<EditorTabs>), String>>;

#[derive(Resource)]
struct PendingNativeRecovery {
    worker: RecoveryWorker,
    guard: RecoveryGuard,
}

fn start_recovery(path: PathBuf) -> Result<RecoveryWorker> {
    Ok(std::thread::Builder::new()
        .name("bloq-session-recovery".into())
        .spawn(move || {
            (|| -> Result<_> {
                let mut native = NativeSession::open(path)?;
                let tabs = native
                    .expected
                    .as_ref()
                    .map(|json| -> Result<_> {
                        let tabs = Session::from_json(json)?.restore()?;
                        native.session = Some(Session::capture(
                            &tabs,
                            (&tabs.active_tab().snapshot).into(),
                            None,
                        )?);
                        Ok(tabs)
                    })
                    .transpose()?;
                Ok((native, tabs))
            })()
            .map_err(|error| format!("{error:#}"))
        })?)
}

fn restore_session(world: &mut World) {
    if world.contains_resource::<crate::systems::screenshot::AutomatedScreenshot>()
        || world.contains_resource::<crate::systems::thumbnails::GalleryThumbnailExport>()
    {
        return;
    }
    let result = session_path().and_then(start_recovery);
    match result {
        Ok(worker) => {
            let guard = RecoveryGuard::capture(world);
            world.insert_resource(PendingNativeRecovery { worker, guard });
            world
                .resource_mut::<Notifications>()
                .push_info("Restoring desktop session; autosave starts when recovery finishes");
        }
        Err(error) => recovery_error(world, &format!("{error:#}")),
    }
}

fn recovery_error(world: &mut World, error: &str) {
    world.resource_mut::<Notifications>().push_error(format!(
        "Desktop autosave paused: {error}. Saved files kept unchanged; export new work before closing."
    ));
}

fn poll_recovery(world: &mut World) {
    // Shutdown may wait: finish recovery before the chained exit save captures
    // edits made during startup. Do not consume the message used by that save.
    let exiting = world
        .get_resource::<Messages<bevy::app::AppExit>>()
        .is_some_and(|messages| !messages.is_empty());
    if !world
        .get_resource::<PendingNativeRecovery>()
        .is_some_and(|pending| exiting || pending.worker.is_finished())
    {
        return;
    }
    let pending = world
        .remove_resource::<PendingNativeRecovery>()
        .expect("finished recovery exists");
    let result = pending
        .worker
        .join()
        .map_err(|_| "Recovery worker failed".to_string())
        .and_then(|result| result)
        .and_then(|(mut backend, tabs)| {
            if let Some(tabs) = tabs {
                let cache = backend
                    .session
                    .take()
                    .ok_or("Missing recovered session cache")?;
                backend.session = install_recovered(world, tabs, cache, &pending.guard, "desktop")?;
            }
            Ok(backend)
        });
    match result {
        Ok(backend) => {
            world.insert_resource(NativeAutosave::new(backend));
        }
        Err(error) => recovery_error(world, &error),
    }
}

fn capture_session(
    native: Option<ResMut<NativeAutosave>>,
    tabs: Res<EditorTabs>,
    graph: Res<GraphState>,
    editor: Res<EditorState>,
    import: Res<ImportExportState>,
    compile: Res<CompileUiState>,
    camera: Single<&CameraSettings, With<EditorCamera>>,
    time: Res<Time<Real>>,
    mut last_save: Local<f64>,
    mut exit: MessageReader<bevy::app::AppExit>,
    mut notifications: ResMut<Notifications>,
) {
    let Some(mut native) = native else { return };
    let exiting = exit.read().next().is_some();
    let result = (|| -> Result<(), String> {
        native.prepare_capture(exiting)?;
        if native
            .backend
            .as_ref()
            .is_some_and(|backend| backend.paused)
        {
            return Ok(());
        }
        if !exiting && time.elapsed_secs_f64() - *last_save < 0.5 {
            return Ok(());
        }
        *last_save = time.elapsed_secs_f64();
        let live = TabSource {
            graph: &graph,
            editor: &editor,
            import: &import,
            compile: &compile,
            camera: &camera,
        };
        native.latest = Some(
            Session::capture(&tabs, live, native.latest.as_ref())
                .map_err(|error| error.to_string())?,
        );
        native.save_latest(exiting)
    })();
    if let Err(error) = result {
        let message = format!("Desktop autosave failed: {error}. Export BLOG before closing.");
        if !notifications.contains_message(&message) {
            notifications.push_error(message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            Self(std::env::temp_dir().join(format!(
                    "bloq-session-{}-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos(),
                    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )))
        }
        fn path(&self) -> PathBuf {
            self.0.join("session-v1.json")
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn save(native: &mut NativeSession, text: &str) {
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.import_export
            .replace_text_preserving_history("earlier source".into());
        live.import_export
            .replace_text_preserving_history(text.into());
        live.import_export.sync_conflict = true;
        native.session = Some(Session::capture(&tabs, (&live).into(), None).unwrap());
        native.dirty = true;
        native.flush().unwrap();
    }

    #[test]
    fn quitting_during_recovery_waits_then_saves_new_edits() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let (release, wait) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            wait.recv().unwrap();
            Ok((NativeSession::open(path).unwrap(), None))
        });
        let mut app = App::new();
        app.init_resource::<EditorTabs>()
            .init_resource::<GraphState>()
            .init_resource::<EditorState>()
            .init_resource::<ImportExportState>()
            .init_resource::<CompileUiState>()
            .init_resource::<Notifications>()
            .init_resource::<Time<Real>>()
            .add_message::<bevy::app::AppExit>()
            .add_systems(Update, (poll_recovery, capture_session).chain());
        let guard = RecoveryGuard::capture(app.world());
        app.world_mut()
            .spawn((EditorCamera, CameraSettings::default()));
        app.world_mut()
            .resource_mut::<ImportExportState>()
            .set_bloq_buffer("draft before immediate quit".into());
        app.insert_resource(PendingNativeRecovery { worker, guard });
        app.world_mut()
            .resource_mut::<Messages<bevy::app::AppExit>>()
            .write(bevy::app::AppExit::Success);
        let release_worker = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            release.send(()).unwrap();
        });
        app.update();
        release_worker.join().unwrap();
        assert!(!app.world().contains_resource::<PendingNativeRecovery>());
        let saved = Session::from_json(&fs::read_to_string(directory.path()).unwrap()).unwrap();
        assert_eq!(saved.tabs[0].blog.as_ref(), "draft before immediate quit");
    }

    #[test]
    fn native_recovery_worker_preserves_save_and_reports_invalid_input() {
        let directory = TestDirectory::new();
        {
            let mut backend = NativeSession::open(directory.path()).unwrap();
            save(&mut backend, "unfinished {");
        }
        let original = fs::read_to_string(directory.path()).unwrap();
        let (backend, tabs) = start_recovery(directory.path())
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(
            tabs.unwrap()
                .active_tab()
                .snapshot
                .import_export
                .bloq_buffer,
            "unfinished {"
        );
        assert_eq!(fs::read_to_string(directory.path()).unwrap(), original);
        assert!(backend.session.is_some());
        drop(backend);
        fs::write(directory.path(), "broken session").unwrap();
        start_recovery(directory.path())
            .unwrap()
            .join()
            .unwrap()
            .err()
            .expect("bad session rejected");
        assert_eq!(
            fs::read_to_string(directory.path()).unwrap(),
            "broken session"
        );
    }

    #[test]
    fn native_restart_restores_invalid_source_history_and_conflict() {
        let directory = TestDirectory::new();
        {
            let mut native = NativeSession::open(directory.path()).unwrap();
            save(&mut native, "unfinished BLOG {");
        }
        let native = NativeSession::open(directory.path()).unwrap();
        let tabs = Session::from_json(native.expected.as_ref().unwrap())
            .unwrap()
            .restore()
            .unwrap();
        let text = &tabs.active_tab().snapshot.import_export;
        assert_eq!(text.bloq_buffer, "unfinished BLOG {");
        assert_eq!(text.text_history(), ["earlier source"]);
        assert!(text.sync_conflict);
    }

    #[test]
    fn background_save_coalesces_and_exit_flushes_latest_source() {
        let directory = TestDirectory::new();
        let mut native = NativeAutosave::new(NativeSession::open(directory.path()).unwrap());
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.import_export.set_bloq_buffer("first draft".into());
        native.latest = Some(Session::capture(&tabs, (&live).into(), None).unwrap());
        native.save_latest(false).unwrap();
        assert!(native.worker.is_some());
        live.import_export
            .replace_text_preserving_history("latest invalid source".into());
        native.latest =
            Some(Session::capture(&tabs, (&live).into(), native.latest.as_ref()).unwrap());
        native.save_latest(false).unwrap(); // No second writer while the first is running.
        native.finish(true).unwrap();
        native.save_latest(true).unwrap();
        drop(native);
        let saved = NativeSession::open(directory.path()).unwrap();
        let tabs = Session::from_json(saved.expected.as_ref().unwrap())
            .unwrap()
            .restore()
            .unwrap();
        assert_eq!(
            tabs.active_tab().snapshot.import_export.bloq_buffer,
            "latest invalid source"
        );
        assert_eq!(
            tabs.active_tab().snapshot.import_export.text_history(),
            ["first draft"]
        );
    }

    #[test]
    fn native_restart_retains_entire_available_undo_stack_and_redo_tail() {
        let directory = TestDirectory::new();
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        for x in 0..25 {
            live.graph_state.graph.add_block(Block::new(
                IVec3::new(x, 0, 0),
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ));
            live.graph_state.commit();
        }
        let retained = live.graph_state.history.len();
        assert_eq!(retained, 20); // Existing runtime policy; persistence adds no pruning.
        live.graph_state.undo();
        {
            let mut native = NativeAutosave::new(NativeSession::open(directory.path()).unwrap());
            native.latest = Some(Session::capture(&tabs, (&live).into(), None).unwrap());
            native.save_latest(true).unwrap();
        }
        let native = NativeSession::open(directory.path()).unwrap();
        let mut tabs = Session::from_json(native.expected.as_ref().unwrap())
            .unwrap()
            .restore()
            .unwrap();
        let state = &mut tabs.active_tab_mut().snapshot.graph_state;
        assert_eq!(state.history.len(), retained);
        assert_eq!(state.graph.block_count(), 24);
        state.redo();
        assert_eq!(state.graph.block_count(), 25);
        for _ in 1..retained {
            state.undo();
        }
        assert_eq!(state.graph.block_count(), 6);
    }

    #[test]
    fn exit_retries_latest_snapshot_after_transient_background_failure() {
        let directory = TestDirectory::new();
        let mut native = NativeAutosave::new(NativeSession::open(directory.path()).unwrap());
        let tabs = EditorTabs::default();
        let mut live = tabs.active_tab().snapshot.clone();
        live.import_export.set_bloq_buffer("older draft".into());
        native.latest = Some(Session::capture(&tabs, (&live).into(), None).unwrap());
        let temp = directory
            .path()
            .with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temp, "test obstruction").unwrap();
        native.save_latest(false).unwrap();
        while !native.worker.as_ref().unwrap().is_finished() {
            std::thread::yield_now();
        }
        fs::remove_file(&temp).unwrap();
        native.prepare_capture(true).unwrap();
        live.import_export
            .set_bloq_buffer("latest draft before quit".into());
        native.latest =
            Some(Session::capture(&tabs, (&live).into(), native.latest.as_ref()).unwrap());
        native.save_latest(true).unwrap();
        drop(native);
        let saved = NativeSession::open(directory.path()).unwrap();
        let tabs = Session::from_json(saved.expected.as_ref().unwrap())
            .unwrap()
            .restore()
            .unwrap();
        assert_eq!(
            tabs.active_tab().snapshot.import_export.bloq_buffer,
            "latest draft before quit"
        );
    }

    #[test]
    fn native_lock_prevents_concurrent_writer_and_releases_on_close() {
        let directory = TestDirectory::new();
        let native = NativeSession::open(directory.path()).unwrap();
        assert!(NativeSession::open(directory.path()).is_err());
        drop(native);
        NativeSession::open(directory.path()).unwrap();
    }

    #[test]
    fn native_save_keeps_previous_and_refuses_external_changes() {
        let directory = TestDirectory::new();
        let mut native = NativeSession::open(directory.path()).unwrap();
        save(&mut native, "first");
        let first = fs::read_to_string(directory.path()).unwrap();
        save(&mut native, "second");
        assert_eq!(
            fs::read_to_string(directory.path().with_extension("previous.json")).unwrap(),
            first
        );
        fs::write(directory.path(), "external save").unwrap();
        native.dirty = true;
        assert!(native.flush().is_err());
        assert!(native.paused);
        assert_eq!(
            fs::read_to_string(directory.path()).unwrap(),
            "external save"
        );
    }

    #[test]
    fn failed_atomic_write_keeps_original_and_does_not_remove_other_temp() {
        let directory = TestDirectory::new();
        fs::create_dir_all(&directory.0).unwrap();
        let path = directory.path();
        fs::write(&path, "original").unwrap();
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temp, "unrecovered temporary data").unwrap();
        assert!(atomic_write(&path, "new").is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "original");
        assert_eq!(
            fs::read_to_string(temp).unwrap(),
            "unrecovered temporary data"
        );
    }
}
