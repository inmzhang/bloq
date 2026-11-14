//! The editor's root Bevy plugin: a thin aggregator that pins the coarse
//! `Update` phase order, seeds the two cross-feature resources, and adds the
//! per-feature plugins. Each feature plugin registers its own systems and
//! resources (see `systems::*`). Also hosts the internal-error-to-toast bridge
//! installed for release and WASM builds.

use std::sync::Mutex;

use bevy::ecs::error::ErrorContext;
use bevy::prelude::*;

use crate::resources::{EditorState, GraphState, Notifications};
use crate::systems::EditorUpdateSet;
use crate::systems::camera::CameraPlugin;
use crate::systems::input::InputPlugin;
use crate::systems::jobs::JobsPlugin;
use crate::systems::screenshot::ScreenshotPlugin;
use crate::systems::setup::SetupPlugin;
use crate::systems::thumbnails::ThumbnailsPlugin;
use crate::systems::ui::UiPlugin;
use crate::systems::visuals::VisualsPlugin;

// =============================================================================
// Internal-error reporting (fallible systems → toast)
// =============================================================================

/// Side-channel from the Bevy error handler to the toast system. Bevy's
/// `ErrorHandler` is a captureless `fn` pointer with no `World` access, so it
/// cannot reach the `Notifications` resource directly; this static queue is
/// the bridge, drained once per frame. Only touched when a system actually
/// fails.
static INTERNAL_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Error handler installed on WASM and release builds (dev-native keeps the
/// default `panic` handler, which is loud at the desk). Logs the full error
/// and queues a user-visible toast, keeping the frame loop alive.
pub(crate) fn report_internal_error(error: BevyError, ctx: ErrorContext) {
    error!(
        "internal error in {} `{}`: {error:?}",
        ctx.kind(),
        ctx.name()
    );
    if let Ok(mut queue) = INTERNAL_ERRORS.lock() {
        queue.push(format!("Internal error in {}: {error}", ctx.name()));
    }
}

/// Deduplicated by exact message so a system failing every frame shows one
/// persistent toast instead of churning the toast cap. Registered by `UiPlugin`
/// as the second step of the egui pass.
pub(crate) fn drain_internal_errors_system(mut notifications: ResMut<Notifications>) {
    let Ok(mut queue) = INTERNAL_ERRORS.lock() else {
        return;
    };
    for message in queue.drain(..) {
        if !notifications.contains_message(&message) {
            notifications.push_error(message);
        }
    }
}

/// The editor's root Bevy plugin: aggregates the feature plugins.
pub(crate) struct BloqEditorPlugin;

impl Plugin for BloqEditorPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_arch = "wasm32")]
        app.add_plugins(crate::session::browser::SessionPlugin);
        #[cfg(not(target_arch = "wasm32"))]
        app.add_plugins(crate::session::native::SessionPlugin);
        // The two genuinely cross-feature resources: the working graph shared by
        // every system, and the god-resource interaction/view state.
        app.init_resource::<GraphState>()
            .init_resource::<EditorState>()
            .configure_sets(
                Update,
                (
                    EditorUpdateSet::Jobs,
                    EditorUpdateSet::Rendering.after(EditorUpdateSet::Jobs),
                    EditorUpdateSet::Interaction.after(EditorUpdateSet::Rendering),
                ),
            )
            .add_plugins((
                SetupPlugin,
                ThumbnailsPlugin,
                CameraPlugin,
                InputPlugin,
                JobsPlugin,
                VisualsPlugin,
                ScreenshotPlugin,
                UiPlugin,
            ))
            .add_systems(Startup, log_startup.after(crate::systems::setup::setup));
    }
}

fn log_startup() {
    info!("BlockGraph Editor started");
}
