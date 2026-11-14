//! Stop a failed browser renderer without also stopping draft persistence.

use bevy::prelude::*;
use bevy::render::error_handler::{RenderError, RenderErrorHandler, RenderErrorPolicy};

#[derive(Resource)]
struct Reported;

pub(crate) fn install(app: &mut App) {
    app.insert_resource(RenderErrorHandler(handle));
}

fn handle(error: &RenderError, world: &mut World, _: &mut World) -> RenderErrorPolicy {
    // Bevy polls StopRendering every frame. Notify only once, never retry a
    // failed device: egui and other plugins retain device-bound resources.
    if !world.contains_resource::<Reported>() {
        world.insert_resource(Reported);
        if let Some(mut keys) = world.get_resource_mut::<ButtonInput<KeyCode>>() {
            keys.reset_all();
        }
        if let Some(mut buttons) = world.get_resource_mut::<ButtonInput<MouseButton>>() {
            buttons.reset_all();
        }
        report(&format!("{:?}: {}", error.ty, error.description));
    }
    RenderErrorPolicy::StopRendering
}

#[cfg(target_arch = "wasm32")]
fn report(detail: &str) {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = window, js_name = showBloqRenderFailure, catch)]
        fn show_failure(detail: &str) -> Result<(), JsValue>;
    }
    if let Err(error) = show_failure(detail) {
        warn!("Unable to display renderer failure: {error:?}");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn report(_: &str) {}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::error_handler::ErrorType;

    #[test]
    fn renderer_failure_stops_rendering_without_exiting_or_repeating_reset() {
        let mut app = App::new();
        app.add_message::<AppExit>();
        install(&mut app);
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::Delete);
        app.insert_resource(keys);
        let error = RenderError {
            ty: ErrorType::DeviceLost,
            description: "test device loss".into(),
            source: None,
        };
        #[derive(Resource, Default)]
        struct MainTicks(u32);
        app.init_resource::<MainTicks>();
        app.add_systems(Update, |mut ticks: ResMut<MainTicks>| ticks.0 += 1);
        let mut render_world = World::new();
        let handler = app.world().resource::<RenderErrorHandler>().0;
        for _ in 0..2 {
            assert!(matches!(
                handler(&error, app.world_mut(), &mut render_world),
                RenderErrorPolicy::StopRendering
            ));
        }
        assert!(
            !app.world()
                .resource::<ButtonInput<KeyCode>>()
                .pressed(KeyCode::Delete)
        );
        assert!(app.world().resource::<Messages<AppExit>>().is_empty());
        assert!(app.world().contains_resource::<Reported>());
        app.update();
        app.update();
        assert_eq!(app.world().resource::<MainTicks>().0, 2);
    }
}
