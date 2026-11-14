//! Preserve browser event-time shortcut modifiers. Bevy's processed keyboard
//! messages discard modifier snapshots and can synthesize releases. Winit's
//! raw stream retains the snapshots, even when a shortcut spans render frames.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::winit::{RawWinitWindowEvent, WINIT_WINDOWS};
use bevy_egui::egui;
use bevy_egui::input::ModifierKeysState;
use bevy_egui::input::{EguiContextMessageReader, EguiInputEvent};
use bevy_egui::{EguiContext, EguiContextSettings};
use winit::event::WindowEvent;

pub(super) fn write_ordered_keyboard_messages(
    mut events: EguiContextMessageReader<RawWinitWindowEvent>,
    mut modifiers: Local<ModifierKeysState>,
    contexts: Query<&EguiContextSettings, With<EguiContext>>,
    mut output: MessageWriter<EguiInputEvent>,
) {
    let mut events = events
        .read_with_non_window_focused(|event| {
            WINIT_WINDOWS.with_borrow(|windows| {
                windows
                    .get_window_entity(event.window_id)
                    .unwrap_or(Entity::PLACEHOLDER)
            })
        })
        .peekable();
    while let Some((raw, context)) = events.next() {
        match &raw.event {
            WindowEvent::Focused(false) => *modifiers = ModifierKeysState::default(),
            WindowEvent::ModifiersChanged(snapshot) => {
                set_modifiers(&mut modifiers, snapshot.state())
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } => {
                // Winit 0.30 web emits a key followed immediately by its changed
                // DOM modifier snapshot in the same send_events batch. Apply
                // that snapshot to this key, not merely to the following one.
                let following = events.next_if(|(next, next_context)| {
                    next.window_id == raw.window_id
                        && *next_context == context
                        && matches!(next.event, WindowEvent::ModifiersChanged(_))
                });
                // Blur emits its own empty snapshot before Focused(false).
                // It belongs to losing focus, not the preceding Cmd+L key.
                let loses_focus = events.peek().is_some_and(|(next, next_context)| {
                    next.window_id == raw.window_id
                        && *next_context == context
                        && matches!(next.event, WindowEvent::Focused(false))
                });
                apply_following_modifiers(
                    &mut modifiers,
                    following.map(|(next, _)| &next.event),
                    loses_focus,
                );
                if !contexts.get(context).is_ok_and(|settings| {
                    settings
                        .input_system_settings
                        .run_write_keyboard_input_messages_system
                }) {
                    continue;
                }
                let window = WINIT_WINDOWS.with_borrow(|windows| {
                    windows
                        .get_window_entity(raw.window_id)
                        .unwrap_or(Entity::PLACEHOLDER)
                });
                let key = bevy::winit::converters::convert_keyboard_input(event, window);
                convert_key(&modifiers, &key, |event| {
                    output.write(EguiInputEvent { context, event });
                });
            }
            _ => {}
        }
    }
}

fn apply_following_modifiers(
    modifiers: &mut ModifierKeysState,
    following: Option<&WindowEvent>,
    loses_focus: bool,
) {
    if let Some(WindowEvent::ModifiersChanged(snapshot)) = following
        && !(loses_focus && snapshot.state().is_empty())
    {
        set_modifiers(modifiers, snapshot.state());
    }
}

fn set_modifiers(modifiers: &mut ModifierKeysState, snapshot: winit::keyboard::ModifiersState) {
    modifiers.shift = snapshot.shift_key();
    modifiers.ctrl = snapshot.control_key();
    modifiers.alt = snapshot.alt_key();
    modifiers.win = snapshot.super_key();
}

fn convert_key(
    modifiers: &ModifierKeysState,
    message: &KeyboardInput,
    mut emit: impl FnMut(egui::Event),
) {
    if modifiers.text_input_is_allowed() && message.state.is_pressed() {
        match &message.logical_key {
            Key::Character(text) if !text.chars().any(char::is_control) => {
                emit(egui::Event::Text(text.to_string()));
            }
            Key::Space => emit(egui::Event::Text(" ".into())),
            _ => {}
        }
    }
    let physical_key = bevy_egui::helpers::bevy_to_egui_physical_key(&message.key_code);
    if let Some(key) = bevy_egui::helpers::bevy_to_egui_key(&message.logical_key).or(physical_key) {
        emit(egui::Event::Key {
            key,
            physical_key,
            pressed: message.state.is_pressed(),
            repeat: message.repeat,
            modifiers: modifiers.to_egui_modifiers(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::input::ButtonState;
    use winit::keyboard::ModifiersState;

    #[test]
    fn adapter_validates_without_a_non_send_window_resource() {
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        world.init_resource::<bevy_egui::input::WindowToEguiContextMap>();
        world.init_resource::<Messages<RawWinitWindowEvent>>();
        world.init_resource::<Messages<EguiInputEvent>>();
        // Bevy 0.20 stores windows in WINIT_WINDOWS, not as a World resource.
        world
            .run_system_once(write_ordered_keyboard_messages)
            .expect("web keyboard adapter validates without a non-send window resource");
    }

    #[test]
    fn blur_reset_does_not_turn_address_bar_shortcut_into_text() {
        let mut modifiers = ModifierKeysState::default();
        set_modifiers(&mut modifiers, ModifiersState::SUPER);
        let reset = WindowEvent::ModifiersChanged(ModifiersState::empty().into());
        apply_following_modifiers(&mut modifiers, Some(&reset), true);
        let key = KeyboardInput {
            logical_key: Key::Character("l".into()),
            key_code: KeyCode::KeyL,
            state: ButtonState::Pressed,
            repeat: false,
            window: Entity::PLACEHOLDER,
            text: Some("l".into()),
        };
        let mut events = Vec::new();
        convert_key(&modifiers, &key, |event| events.push(event));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, egui::Event::Text(_)))
        );
        // The subsequent Focused(false) handler clears state for future keys.
        modifiers = ModifierKeysState::default();
        assert!(modifiers.text_input_is_allowed());
        // A real key snapshot without blur must still clear modifiers.
        set_modifiers(&mut modifiers, ModifiersState::SUPER);
        apply_following_modifiers(&mut modifiers, Some(&reset), false);
        assert!(modifiers.text_input_is_allowed());
    }

    #[test]
    fn browser_snapshot_overrides_missing_or_released_modifier_keys() {
        for snapshot in [ModifiersState::SUPER, ModifiersState::CONTROL] {
            let mut modifiers = ModifierKeysState::default();
            // Meta-down can report meta:false; Bevy may synthesize a release.
            // The V event's authoritative meta:true must still suppress text.
            set_modifiers(&mut modifiers, ModifiersState::empty());
            let changed = WindowEvent::ModifiersChanged(snapshot.into());
            apply_following_modifiers(&mut modifiers, Some(&changed), false);
            let mut events = Vec::new();
            let key = KeyboardInput {
                logical_key: Key::Character("v".into()),
                key_code: KeyCode::KeyV,
                state: ButtonState::Pressed,
                repeat: false,
                window: Entity::PLACEHOLDER,
                text: Some("v".into()),
            };
            convert_key(&modifiers, &key, |event| events.push(event));
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, egui::Event::Text(_)))
            );
            // An unchanged snapshot persists across frame boundaries.
            apply_following_modifiers(&mut modifiers, None, false);
            assert!(!modifiers.text_input_is_allowed());
            let released = WindowEvent::ModifiersChanged(ModifiersState::empty().into());
            apply_following_modifiers(&mut modifiers, Some(&released), false);
            convert_key(&modifiers, &key, |event| events.push(event));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, egui::Event::Text(text) if text == "v"))
                    .count(),
                1
            );
        }
    }
}
