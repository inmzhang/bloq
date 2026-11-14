//! Undo/redo keyboard handling.
use super::*;

/// Handles the undo/redo keyboard shortcuts.
pub(crate) fn undo_redo_system(
    mut intents: ResMut<UiIntentBuffer>,
    ui_input: Res<UiInputState>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if ui_input.blocks_viewport_keyboard_input() {
        return;
    }

    if shortcut_command(&keyboard_input) && keyboard_input.just_pressed(KeyCode::KeyZ) {
        intents.push(if shortcut_shift(&keyboard_input) {
            UiIntent::Redo
        } else {
            UiIntent::Undo
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_command_shift_chord_still_redoes() {
        let mut app = App::new();
        app.init_resource::<UiIntentBuffer>()
            .init_resource::<UiInputState>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, undo_redo_system);
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        for key in [KeyCode::SuperLeft, KeyCode::ShiftLeft, KeyCode::KeyZ] {
            keys.press(key);
        }
        for key in [KeyCode::KeyZ, KeyCode::ShiftLeft, KeyCode::SuperLeft] {
            keys.release(key);
        }
        app.update();
        assert!(matches!(
            app.world_mut()
                .resource_mut::<UiIntentBuffer>()
                .drain()
                .next(),
            Some(UiIntent::Redo)
        ));
    }
}
