//! Editor colour themes and reusable egui styling helpers.

use bevy_egui::egui;
use bloq_circuit::PauliBasis;
use egui::Color32;
use std::fmt;

/// The selectable editor colour themes. `Light` is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub(crate) enum ThemePreset {
    #[default]
    Light,
    GruvboxMaterial,
}

/// The resolved colour set for a theme, shared by the egui UI and the 3D
/// viewport. Obtain one with [`palette`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThemePalette {
    pub(crate) dark_mode: bool,
    pub(crate) bg_dark: Color32,
    pub(crate) bg_panel: Color32,
    pub(crate) bg_surface: Color32,
    pub(crate) bg_hover: Color32,
    pub(crate) bg_active: Color32,
    pub(crate) text_primary: Color32,
    pub(crate) text_dim: Color32,
    pub(crate) text_bright: Color32,
    pub(crate) border: Color32,
    pub(crate) border_bright: Color32,
    pub(crate) accent_primary: Color32,
    pub(crate) accent_secondary: Color32,
    pub(crate) accent_warn: Color32,
    pub(crate) accent_error: Color32,
    pub(crate) success: Color32,
    pub(crate) orange: Color32,
    pub(crate) yellow: Color32,
    pub(crate) grey2: Color32,
}

const LIGHT: ThemePalette = ThemePalette {
    dark_mode: false,
    bg_dark: Color32::from_rgb(216, 224, 230),    // #d8e0e6
    bg_panel: Color32::from_rgb(248, 250, 252),   // #f8fafc
    bg_surface: Color32::from_rgb(255, 255, 255), // #ffffff
    bg_hover: Color32::from_rgb(216, 234, 254),   // #d8eafe
    bg_active: Color32::from_rgb(195, 218, 247),  // #c3daf7
    text_primary: Color32::from_rgb(31, 41, 55),  // #1f2937
    text_dim: Color32::from_rgb(82, 96, 112),     // #526070
    text_bright: Color32::from_rgb(15, 23, 42),   // #0f172a
    border: Color32::from_rgb(148, 163, 178),     // #94a3b2
    border_bright: Color32::from_rgb(88, 103, 124), // #58677c
    accent_primary: Color32::from_rgb(29, 78, 216), // #1d4ed8
    accent_secondary: Color32::from_rgb(4, 120, 87), // #047857
    accent_warn: Color32::from_rgb(180, 83, 9),   // #b45309
    accent_error: Color32::from_rgb(185, 28, 28), // #b91c1c
    success: Color32::from_rgb(21, 128, 61),      // #15803d
    orange: Color32::from_rgb(194, 65, 12),       // #c2410c
    yellow: Color32::from_rgb(161, 98, 7),        // #a16207
    grey2: Color32::from_rgb(71, 85, 105),        // #475569
};

const GRUVBOX_MATERIAL: ThemePalette = ThemePalette {
    dark_mode: true,
    bg_dark: Color32::from_rgb(40, 40, 40),         // #282828
    bg_panel: Color32::from_rgb(50, 48, 47),        // #32302f
    bg_surface: Color32::from_rgb(69, 64, 61),      // #45403d
    bg_hover: Color32::from_rgb(90, 82, 76),        // #5a524c
    bg_active: Color32::from_rgb(102, 94, 86),      // #665e56
    text_primary: Color32::from_rgb(212, 190, 152), // #d4be98
    text_dim: Color32::from_rgb(189, 174, 147),     // #bdae93
    text_bright: Color32::from_rgb(221, 199, 161),  // #ddc7a1
    border: Color32::from_rgb(69, 64, 61),          // #45403d
    border_bright: Color32::from_rgb(90, 82, 76),   // #5a524c
    accent_primary: Color32::from_rgb(137, 180, 130), // #89b482
    accent_secondary: Color32::from_rgb(125, 174, 163), // #7daea3
    accent_warn: Color32::from_rgb(216, 166, 87),   // #d8a657
    accent_error: Color32::from_rgb(234, 105, 98),  // #ea6962
    success: Color32::from_rgb(169, 182, 101),      // #a9b665
    orange: Color32::from_rgb(231, 138, 78),        // #e78a4e
    yellow: Color32::from_rgb(216, 166, 87),        // #d8a657
    grey2: Color32::from_rgb(189, 174, 147),        // #bdae93
};

/// Returns the static palette for a theme preset.
pub(crate) fn palette(preset: ThemePreset) -> &'static ThemePalette {
    match preset {
        ThemePreset::Light => &LIGHT,
        ThemePreset::GruvboxMaterial => &GRUVBOX_MATERIAL,
    }
}

/// Stable, high-contrast colors for module-definition ownership. The first
/// entries use a color-blind-friendly qualitative palette; larger programs
/// continue around the hue wheel instead of silently reusing a swatch.
pub(crate) fn module_color(index: usize) -> Color32 {
    let color = bloq_graph::module_color(index);
    Color32::from_rgb(color.r, color.g, color.b)
}

// Zed's classic pairing: Zed Sans for interface text, Zed Mono for data and
// code, Fantasque kept only for the `\u{f0xx}` Nerd-Font icon glyphs.
const MONO_FONT_FAMILY: &str = "ZedMono";
const ICON_FONT_FAMILY: &str = "FantasqueSansMNerdFontPropo";
const SANS_FONT_FAMILY: &str = "ZedSans";
const SANS_BOLD_FONT_FAMILY: &str = "ZedSans-Bold";

// Minimal subsets (regenerate with assets/fonts/regen-subsets.sh): new icons
// or exotic characters render as fallback/tofu until the script is rerun.
static ICON_FONT_BYTES: &[u8] =
    include_bytes!("../assets/fonts/FantasqueSansMNerdFontPropo-Regular.ttf");
static SANS_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/ZedSans-Regular.ttf");
static SANS_BOLD_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/ZedSans-Bold.ttf");
static MONO_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/ZedMono-Regular.ttf");

/// The font family for headings and section labels: Zed Sans Bold. egui has
/// no weight axis, so the heavier cut is registered as its own family.
pub(crate) fn bold_family() -> egui::FontFamily {
    egui::FontFamily::Name("bold".into())
}

/// Applies the fonts, spacing, and visuals for `preset` to the egui context.
/// Called once at startup and whenever the theme changes.
pub(crate) fn setup_theme(ctx: &egui::Context, preset: ThemePreset) {
    setup_ui_fonts(ctx);
    setup_ui_style(ctx);
    setup_visuals(ctx, palette(preset));
}

/// Applies the same fonts and text styles the live UI uses to a throwaway
/// context. The SVG exporter uses this so galley layout (row splits, glyph
/// positions) matches on-screen exactly; visuals are irrelevant because the
/// export paints shapes directly rather than through widgets.
pub(crate) fn setup_export_context(ctx: &egui::Context) {
    setup_ui_fonts(ctx);
    setup_ui_style(ctx);
}

fn setup_ui_fonts(ctx: &egui::Context) {
    let mut font_definitions = egui::FontDefinitions::default();

    for (name, bytes) in [
        (MONO_FONT_FAMILY, MONO_FONT_BYTES),
        (ICON_FONT_FAMILY, ICON_FONT_BYTES),
        (SANS_FONT_FAMILY, SANS_FONT_BYTES),
        (SANS_BOLD_FONT_FAMILY, SANS_BOLD_FONT_BYTES),
    ] {
        font_definitions
            .font_data
            .insert(name.to_string(), egui::FontData::from_static(bytes).into());
    }

    // Icon face second so inline Nerd-Font icons resolve; egui defaults last.
    let proportional = font_definitions
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default();
    proportional.insert(0, SANS_FONT_FAMILY.to_string());
    proportional.insert(1, ICON_FONT_FAMILY.to_string());

    let monospace = font_definitions
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default();
    monospace.insert(0, MONO_FONT_FAMILY.to_string());
    monospace.insert(1, ICON_FONT_FAMILY.to_string());

    font_definitions.families.insert(
        bold_family(),
        vec![
            SANS_BOLD_FONT_FAMILY.to_string(),
            SANS_FONT_FAMILY.to_string(),
            ICON_FONT_FAMILY.to_string(),
        ],
    );

    ctx.set_fonts(font_definitions);
}

fn setup_ui_style(ctx: &egui::Context) {
    // egui 0.35 keeps a style per theme; mutate them all in one pass instead of
    // the old clone-then-set_style dance.
    ctx.all_styles_mut(|style| {
        // 19/14/12 scale; Bold carries hierarchy, Monospace matches Body so
        // mixed label/value rows share a baseline.
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(19.0, bold_family()),
        );
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(12.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::new(14.0, egui::FontFamily::Monospace),
        );
        style.text_styles.insert(
            egui::TextStyle::Name("SectionHeader".into()),
            egui::FontId::new(12.0, bold_family()),
        );

        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 5.0);
        style.spacing.interact_size.y = 28.0;
        style.spacing.slider_width = 180.0;
        style.spacing.indent = 18.0;
        style.spacing.scroll.bar_width = 8.0;
    });
}

fn setup_visuals(ctx: &egui::Context, palette: &ThemePalette) {
    let mut visuals = if palette.dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.window_fill = palette.bg_panel;
    visuals.panel_fill = palette.bg_panel;
    visuals.extreme_bg_color = palette.bg_dark;
    visuals.faint_bg_color = palette.bg_surface;

    visuals.window_stroke = egui::Stroke::new(1.0, palette.border_bright);
    visuals.window_corner_radius = egui::CornerRadius::same(8);

    visuals.widgets.noninteractive.bg_fill = palette.bg_surface;
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, palette.text_dim);
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(0.5, palette.border);

    visuals.widgets.inactive.bg_fill = palette.bg_surface;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, palette.text_primary);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, palette.border);

    visuals.widgets.hovered.bg_fill = palette.bg_hover;
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, palette.text_bright);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, palette.accent_primary);

    visuals.widgets.active.bg_fill = palette.bg_active;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, palette.accent_primary);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, palette.accent_primary);

    visuals.widgets.open.bg_fill = palette.bg_surface;
    visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, palette.text_bright);
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, palette.accent_secondary);

    visuals.selection.bg_fill = palette.accent_primary.gamma_multiply(0.20);
    visuals.selection.stroke = egui::Stroke::new(1.0, palette.accent_primary);

    visuals.popup_shadow = egui::Shadow {
        offset: [2, 4],
        blur: 8,
        spread: 0,
        color: egui::Color32::from_black_alpha(96),
    };
    visuals.window_shadow = egui::Shadow {
        offset: [2, 4],
        blur: 12,
        spread: 0,
        color: egui::Color32::from_black_alpha(80),
    };

    visuals.override_text_color = Some(palette.text_primary);
    visuals.hyperlink_color = palette.accent_secondary;
    visuals.menu_corner_radius = egui::CornerRadius::same(6);

    ctx.set_visuals(visuals);
}

/// Draws an uppercased section heading with an accent underline.
pub(crate) fn section_header(ui: &mut egui::Ui, palette: &ThemePalette, label: &str) {
    ui.add_space(8.0);
    // Tracked caps; the bold cut makes `.strong()` redundant.
    ui.label(
        egui::RichText::new(label.to_uppercase())
            .text_style(egui::TextStyle::Name("SectionHeader".into()))
            .extra_letter_spacing(0.8)
            .color(palette.grey2),
    );
    accent_separator(ui, palette.border_bright);
    ui.add_space(4.0);
}

/// Draws a full-width one-pixel horizontal rule in `color`.
pub(crate) fn accent_separator(ui: &mut egui::Ui, color: Color32) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, color);
}

/// Wraps `add_contents` in a bordered, rounded surface panel.
pub(crate) fn section_frame(
    ui: &mut egui::Ui,
    palette: &ThemePalette,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let fill = if palette.dark_mode {
        palette.bg_dark
    } else {
        palette.bg_surface
    };
    egui::Frame::new()
        .fill(fill)
        .corner_radius(4.0)
        .stroke(egui::Stroke::new(1.0, palette.border))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            add_contents(ui);
        });
    ui.add_space(2.0);
}

/// Returns `color` with its alpha channel replaced.
pub(crate) fn with_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

/// The display colour for a Pauli basis, tuned per light/dark theme. Shared by
/// the circuit block-region overlay and the ZX node view so the two never drift.
pub(crate) fn pauli_basis_color(basis: PauliBasis, palette: &ThemePalette) -> Color32 {
    match (palette.dark_mode, basis) {
        (false, PauliBasis::X) => Color32::from_rgb(185, 28, 28),
        (false, PauliBasis::Y) => Color32::from_rgb(21, 128, 61),
        (false, PauliBasis::Z) => Color32::from_rgb(29, 78, 216),
        (true, PauliBasis::X) => Color32::from_rgb(234, 105, 98),
        (true, PauliBasis::Y) => Color32::from_rgb(137, 180, 130),
        (true, PauliBasis::Z) => Color32::from_rgb(125, 174, 248),
    }
}

/// Draws a small `icon text` status label in `color`.
pub(crate) fn status_chip(ui: &mut egui::Ui, icon: &str, text: impl fmt::Display, color: Color32) {
    ui.label(
        egui::RichText::new(format!("{icon} {text}"))
            .color(color)
            .small(),
    );
}

/// Adds a surface-filled button whose label is tinted with `accent`.
/// An outlined button that fills with `accent` while `lit`: for controls that
/// stay on until clicked again, and for plain outlined actions at `lit = false`.
pub(crate) fn toggle_button(
    ui: &mut egui::Ui,
    palette: &ThemePalette,
    label: &str,
    accent: Color32,
    lit: bool,
) -> egui::Response {
    let (foreground, background) = if lit {
        (palette.bg_panel, accent)
    } else {
        (accent, palette.bg_surface)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(label).small().color(foreground))
            .small()
            .fill(background)
            .stroke(egui::Stroke::new(1.0, accent)),
    )
}

pub(crate) fn neon_button(
    ui: &mut egui::Ui,
    palette: &ThemePalette,
    label: &str,
    accent: Color32,
) -> egui::Response {
    let button = egui::Button::new(egui::RichText::new(label).color(accent))
        .fill(palette.bg_surface)
        .stroke(egui::Stroke::new(1.0, palette.border));
    ui.add(button)
}

#[cfg(test)]
mod tests {
    use super::module_color;

    #[test]
    fn shared_palette_preserves_editor_hsv_colors() {
        for index in 10..1024 {
            let hue = (index as f32 * 0.618_034).fract();
            assert_eq!(
                module_color(index),
                bevy_egui::egui::Color32::from(bevy_egui::egui::ecolor::Hsva::new(
                    hue, 0.72, 0.82, 1.0
                ))
            );
        }
    }
}
