//! Headless egui → SVG rendering.
//!
//! Editor-agnostic: [`render_svg`] runs one throwaway egui frame at
//! `pixels_per_point = 1.0`, letting a caller paint into a background-layer
//! [`egui::Painter`], then [`shapes_to_svg`] converts the resulting
//! [`ClippedShape`]s into a standalone SVG document. Clip rects are ignored on
//! purpose: the export deliberately paints everything inside one rect.

use bevy_egui::egui::{self, Color32, Pos2, Vec2};
use egui::epaint::{
    CircleShape, ClippedShape, ColorMode, CubicBezierShape, EllipseShape, Mesh, PathShape,
    PathStroke, QuadraticBezierShape, RectShape, Shape, Stroke, TextShape,
};

/// Monospace font stack: the embedded faces first, then generic fallbacks so
/// the file still renders where those faces are unavailable.
const MONOSPACE_STACK: &str =
    "'Zed Mono', 'FantasqueSansM Nerd Font Propo', ui-monospace, monospace";
/// Proportional font stack, mirroring [`MONOSPACE_STACK`].
const PROPORTIONAL_STACK: &str =
    "'Zed Sans', 'FantasqueSansM Nerd Font Propo', system-ui, sans-serif";

/// Renders one egui frame headlessly and returns it as an SVG string.
///
/// `paint` draws into the background layer's painter. It is `Fn` rather than
/// `FnOnce` because egui may run the closure over several layout passes and only
/// the last pass's shapes are kept; capturing solely by shared reference keeps it
/// re-runnable. The context reuses the live UI's fonts and text styles so galley
/// layout (row splits, glyph positions) matches on-screen exactly.
pub(crate) fn render_svg(size: Vec2, paint: impl Fn(&egui::Painter)) -> String {
    let ctx = egui::Context::default();
    crate::theme::setup_export_context(&ctx);
    // Export coordinates are logical points == SVG user units, so pin the DPI
    // scale to 1.0; otherwise glyphs would snap to a different pixel grid.
    ctx.set_pixels_per_point(1.0);

    let screen = egui::Rect::from_min_size(Pos2::ZERO, size);
    let raw_input = egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };

    let mut output = ctx.run_ui(raw_input, |ui| {
        let painter = ui.ctx().layer_painter(egui::LayerId::background());
        paint(&painter);
    });

    output.textures_delta.clear();
    shapes_to_svg(&output.shapes, size)
}

/// Converts a frame's shapes into a self-contained SVG document.
fn shapes_to_svg(shapes: &[ClippedShape], size: Vec2) -> String {
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\">",
        w = n(size.x),
        h = n(size.y),
    );
    for clipped in shapes {
        // Clip rects are intentionally ignored (see module docs).
        append_shape(&mut out, &clipped.shape);
    }
    out.push_str("</svg>");
    out
}

/// Appends one shape (recursing through [`Shape::Vec`]) to the SVG buffer.
fn append_shape(out: &mut String, shape: &Shape) {
    match shape {
        // Backend-specific paint callbacks have no SVG equivalent.
        Shape::Noop | Shape::Callback(_) => {}
        Shape::Vec(shapes) => shapes.iter().for_each(|shape| append_shape(out, shape)),
        Shape::Rect(rect) => append_rect(out, rect),
        Shape::LineSegment { points, stroke } => append_line(out, points, *stroke),
        Shape::Path(path) => append_path(out, path),
        Shape::Circle(circle) => append_circle(out, circle),
        Shape::Ellipse(ellipse) => append_ellipse(out, ellipse),
        Shape::QuadraticBezier(curve) => append_quadratic(out, curve),
        Shape::CubicBezier(curve) => append_cubic(out, curve),
        Shape::Mesh(mesh) => append_mesh(out, mesh),
        Shape::Text(text) => append_text(out, text),
    }
}

fn append_rect(out: &mut String, rect: &RectShape) {
    let fill = paint_attr("fill", rect.fill);
    let stroke = stroke_attr(rect.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    let r = rect.rect;
    // SVG only supports a single corner radius; take the largest of the four so
    // rounded tiles stay rounded. StrokeKind is ignored — a centred stroke is
    // close enough at the 1-2 px widths the canvas uses.
    let cr = rect.corner_radius;
    let radius = cr.nw.max(cr.ne).max(cr.sw).max(cr.se);
    out.push_str(&format!(
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
        n(r.min.x),
        n(r.min.y),
        n(r.width()),
        n(r.height()),
    ));
    if radius > 0 {
        out.push_str(&format!(" rx=\"{}\"", radius));
    }
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

fn append_line(out: &mut String, points: &[Pos2; 2], stroke: Stroke) {
    let Some(stroke_attr) = stroke_attr(stroke) else {
        return;
    };
    out.push_str(&format!(
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" {stroke_attr}/>",
        n(points[0].x),
        n(points[0].y),
        n(points[1].x),
        n(points[1].y),
    ));
}

fn append_path(out: &mut String, path: &PathShape) {
    if path.points.len() < 2 {
        return;
    }
    // Fill is only meaningful (and only supplied by egui) for closed paths;
    // nonzero fill-rule is fine for the convex polygons the canvas produces.
    let fill = path.closed.then(|| paint_attr("fill", path.fill)).flatten();
    let stroke = path_stroke_attr(&path.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    out.push_str(&format!(
        "<path d=\"{}\"",
        path_data(&path.points, path.closed)
    ));
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

fn append_circle(out: &mut String, circle: &CircleShape) {
    let fill = paint_attr("fill", circle.fill);
    let stroke = stroke_attr(circle.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    out.push_str(&format!(
        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"",
        n(circle.center.x),
        n(circle.center.y),
        n(circle.radius),
    ));
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

fn append_ellipse(out: &mut String, ellipse: &EllipseShape) {
    let fill = paint_attr("fill", ellipse.fill);
    let stroke = stroke_attr(ellipse.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    out.push_str(&format!(
        "<ellipse cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\"",
        n(ellipse.center.x),
        n(ellipse.center.y),
        n(ellipse.radius.x),
        n(ellipse.radius.y),
    ));
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

fn append_quadratic(out: &mut String, curve: &QuadraticBezierShape) {
    let [start, control, end] = curve.points;
    let fill = curve
        .closed
        .then(|| paint_attr("fill", curve.fill))
        .flatten();
    let stroke = path_stroke_attr(&curve.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    let mut d = format!(
        "M {} {} Q {} {} {} {}",
        n(start.x),
        n(start.y),
        n(control.x),
        n(control.y),
        n(end.x),
        n(end.y),
    );
    if curve.closed {
        d.push_str(" Z");
    }
    out.push_str(&format!("<path d=\"{d}\""));
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

fn append_cubic(out: &mut String, curve: &CubicBezierShape) {
    let [start, c1, c2, end] = curve.points;
    let fill = curve
        .closed
        .then(|| paint_attr("fill", curve.fill))
        .flatten();
    let stroke = path_stroke_attr(&curve.stroke);
    if fill.is_none() && stroke.is_none() {
        return;
    }
    let mut d = format!(
        "M {} {} C {} {} {} {} {} {}",
        n(start.x),
        n(start.y),
        n(c1.x),
        n(c1.y),
        n(c2.x),
        n(c2.y),
        n(end.x),
        n(end.y),
    );
    if curve.closed {
        d.push_str(" Z");
    }
    out.push_str(&format!("<path d=\"{d}\""));
    push_paint(
        out,
        fill.as_deref().unwrap_or("fill=\"none\""),
        stroke.as_deref(),
    );
    out.push_str("/>");
}

/// Emits each mesh triangle as a flat-shaded polygon. egui only produces
/// textureless per-vertex-coloured meshes here (the detector-slice gradient
/// fans), which SVG cannot express as a smooth gradient, so each triangle takes
/// the average of its three vertex colours — the gradient becomes faceted.
fn append_mesh(out: &mut String, mesh: &Mesh) {
    for triangle in mesh.indices.as_chunks::<3>().0 {
        let [a, b, c] = [
            &mesh.vertices[triangle[0] as usize],
            &mesh.vertices[triangle[1] as usize],
            &mesh.vertices[triangle[2] as usize],
        ];
        let color = average_color(a.color, b.color, c.color);
        let Some(fill) = paint_attr("fill", color) else {
            continue;
        };
        out.push_str(&format!(
            "<polygon points=\"{},{} {},{} {},{}\" {fill}/>",
            n(a.pos.x),
            n(a.pos.y),
            n(b.pos.x),
            n(b.pos.y),
            n(c.pos.x),
            n(c.pos.y),
        ));
    }
}

/// Emits one `<text>` per galley row. Font metrics drift between SVG viewers, so
/// each row is anchored at the centre of its own rect (`text-anchor="middle"`)
/// and placed on its baseline; this keeps the label centred over its glyph run
/// regardless of the viewer's exact advance widths.
fn append_text(out: &mut String, text: &TextShape) {
    // Canvas texts are never rotated; the row baseline logic below assumes it.
    debug_assert_eq!(text.angle, 0.0, "SVG export does not rotate text");

    let job = &text.galley.job;
    let format = job.sections.first().map(|section| &section.format);
    let font_id = format
        .map(|format| format.font_id.clone())
        .unwrap_or_default();
    let base_color = format.map_or(Color32::PLACEHOLDER, |format| format.color);
    // `override_text_color` wins; otherwise the section colour, with any
    // placeholder resolved to the galley's fallback (matches epaint).
    let color = text.override_text_color.unwrap_or({
        if base_color == Color32::PLACEHOLDER {
            text.fallback_color
        } else {
            base_color
        }
    });
    let Some(fill) = paint_attr("fill", color) else {
        return;
    };
    let font_size = font_id.size;
    let family = match font_id.family {
        egui::FontFamily::Monospace => MONOSPACE_STACK,
        _ => PROPORTIONAL_STACK,
    };

    for placed in &text.galley.rows {
        let row = &placed.row;
        let content: String = row.glyphs.iter().map(|glyph| glyph.chr).collect();
        if content.trim().is_empty() {
            continue;
        }
        // Row rect and baseline are relative to the galley; `text.pos` is the
        // galley's top-left in screen space.
        let center_x = text.pos.x + placed.pos.x + row.size.x * 0.5;
        // Glyph `pos.y` is the baseline relative to the row; fall back to a
        // font-proportional guess for an all-whitespace row with no glyphs.
        let baseline_rel = row
            .glyphs
            .first()
            .map_or_else(|| row.size.y - 0.18 * font_size, |glyph| glyph.pos.y);
        let baseline_y = text.pos.y + placed.pos.y + baseline_rel;

        out.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" \
             font-family=\"{}\" font-size=\"{}\" {fill}>{}</text>",
            n(center_x),
            n(baseline_y),
            family,
            n(font_size),
            xml_escape(&content),
        ));
    }
}

// ============================================================================
// Attribute helpers
// ============================================================================

/// Appends a fill attribute and an optional stroke attribute after an element's
/// geometry attributes, ready to close the tag.
fn push_paint(out: &mut String, fill: &str, stroke: Option<&str>) {
    out.push(' ');
    out.push_str(fill);
    if let Some(stroke) = stroke {
        out.push(' ');
        out.push_str(stroke);
    }
}

/// An SVG paint attribute (`fill`/`stroke`) for a premultiplied egui colour, or
/// `None` when fully transparent. egui stores colours premultiplied, but SVG
/// wants a straight colour plus a separate opacity, so translucent colours are
/// un-premultiplied here or they would render too dark.
fn paint_attr(kind: &str, color: Color32) -> Option<String> {
    let alpha = color.a();
    if alpha == 0 {
        return None;
    }
    let (r, g, b) = if alpha == 255 {
        (color.r(), color.g(), color.b())
    } else {
        let straight = |channel: u8| {
            (((channel as u32) * 255 + alpha as u32 / 2) / alpha as u32).min(255) as u8
        };
        (
            straight(color.r()),
            straight(color.g()),
            straight(color.b()),
        )
    };
    let mut attr = format!("{kind}=\"rgb({r},{g},{b})\"");
    if alpha < 255 {
        attr.push_str(&format!(" {kind}-opacity=\"{}\"", n(alpha as f32 / 255.0)));
    }
    Some(attr)
}

fn stroke_attr(stroke: Stroke) -> Option<String> {
    if stroke.width <= 0.0 {
        return None;
    }
    let paint = paint_attr("stroke", stroke.color)?;
    Some(format!("{paint} stroke-width=\"{}\"", n(stroke.width)))
}

fn path_stroke_attr(stroke: &PathStroke) -> Option<String> {
    if stroke.width <= 0.0 {
        return None;
    }
    // Gradient (`UV`) path strokes never occur on the canvas; treat them as
    // absent rather than guessing a colour.
    let ColorMode::Solid(color) = stroke.color else {
        return None;
    };
    let paint = paint_attr("stroke", color)?;
    Some(format!("{paint} stroke-width=\"{}\"", n(stroke.width)))
}

fn path_data(points: &[Pos2], closed: bool) -> String {
    let mut d = format!("M {} {}", n(points[0].x), n(points[0].y));
    for point in &points[1..] {
        d.push_str(&format!(" L {} {}", n(point.x), n(point.y)));
    }
    if closed {
        d.push_str(" Z");
    }
    d
}

fn average_color(a: Color32, b: Color32, c: Color32) -> Color32 {
    let mean = |x: u8, y: u8, z: u8| ((x as u16 + y as u16 + z as u16) / 3) as u8;
    Color32::from_rgba_premultiplied(
        mean(a.r(), b.r(), c.r()),
        mean(a.g(), b.g(), c.g()),
        mean(a.b(), b.b(), c.b()),
        mean(a.a(), b.a(), c.a()),
    )
}

/// Rounds to 2 decimals and formats compactly (no trailing zeros), keeping the
/// SVG small. Non-finite inputs collapse to `0`.
fn n(value: f32) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    let rounded = (value * 100.0).round() / 100.0;
    // `-0` reads oddly; normalize it away.
    let rounded = if rounded == 0.0 { 0.0 } else { rounded };
    format!("{rounded}")
}

fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::{Galley, Vertex, WHITE_UV};
    use std::sync::Arc;

    /// Lays out `text` inside a throwaway headless frame (egui refuses to build
    /// galleys before the first `run`), returning the galley for shape tests.
    fn layout_galley(text: &str) -> Arc<Galley> {
        let ctx = egui::Context::default();
        crate::theme::setup_export_context(&ctx);
        ctx.set_pixels_per_point(1.0);
        let mut galley = None;
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, Vec2::splat(200.0))),
                ..Default::default()
            },
            |ui| {
                galley = Some(ui.painter().layout_no_wrap(
                    text.to_string(),
                    egui::FontId::monospace(14.0),
                    Color32::BLACK,
                ));
            },
        )
        .drop_without_applying_deltas();
        galley.expect("galley laid out during frame")
    }

    #[test]
    fn shapes_to_svg_emits_expected_elements() {
        let shapes = vec![
            // Rect with both fill and a stroke, rounded corners.
            clipped(Shape::Rect(RectShape::new(
                egui::Rect::from_min_size(Pos2::new(1.0, 2.0), Vec2::new(10.0, 20.0)),
                egui::CornerRadius::same(3),
                Color32::from_rgb(200, 100, 50),
                Stroke::new(1.5, Color32::BLACK),
                egui::StrokeKind::Outside,
            ))),
            clipped(Shape::LineSegment {
                points: [Pos2::new(0.0, 0.0), Pos2::new(5.0, 6.0)],
                stroke: Stroke::new(1.0, Color32::RED),
            }),
            clipped(Shape::circle_filled(
                Pos2::new(7.0, 8.0),
                4.0,
                Color32::BLUE,
            )),
            // Closed convex polygon → filled path.
            clipped(Shape::convex_polygon(
                vec![
                    Pos2::new(0.0, 0.0),
                    Pos2::new(4.0, 0.0),
                    Pos2::new(2.0, 4.0),
                ],
                Color32::GREEN,
                Stroke::NONE,
            )),
        ];

        let svg = shapes_to_svg(&shapes, Vec2::new(100.0, 100.0));

        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("viewBox=\"0 0 100 100\""));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("rx=\"3\""));
        assert!(svg.contains("stroke-width=\"1.5\""));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<path"));
        assert!(svg.contains(" Z\""), "closed path should end with Z");
    }

    #[test]
    fn text_content_is_xml_escaped() {
        let galley = layout_galley("A & B");
        let text = Shape::Text(TextShape::new(Pos2::new(3.0, 4.0), galley, Color32::BLACK));

        let svg = shapes_to_svg(&[clipped(text)], Vec2::new(50.0, 50.0));

        assert!(svg.contains("<text"));
        assert!(svg.contains("A &amp; B"));
        assert!(!svg.contains("A & B"));
        assert!(svg.contains("text-anchor=\"middle\""));
    }

    #[test]
    fn mesh_triangle_becomes_averaged_polygon() {
        let mut mesh = Mesh::default();
        mesh.vertices
            .push(vertex(Pos2::new(0.0, 0.0), Color32::from_rgb(255, 0, 0)));
        mesh.vertices
            .push(vertex(Pos2::new(4.0, 0.0), Color32::from_rgb(0, 255, 0)));
        mesh.vertices
            .push(vertex(Pos2::new(2.0, 4.0), Color32::from_rgb(0, 0, 255)));
        mesh.indices.extend([0, 1, 2]);

        let svg = shapes_to_svg(&[clipped(Shape::Mesh(mesh.into()))], Vec2::new(10.0, 10.0));

        assert!(svg.contains("<polygon"));
        // Average of (255,0,0),(0,255,0),(0,0,255) is (85,85,85).
        assert!(svg.contains("rgb(85,85,85)"), "unexpected fill in {svg}");
    }

    #[test]
    fn fully_transparent_shapes_are_skipped() {
        let rect = Shape::Rect(RectShape::filled(
            egui::Rect::from_min_size(Pos2::ZERO, Vec2::splat(5.0)),
            0,
            Color32::TRANSPARENT,
        ));
        let svg = shapes_to_svg(&[clipped(rect)], Vec2::new(10.0, 10.0));
        assert!(!svg.contains("<rect"), "transparent rect should be dropped");
    }

    #[test]
    fn translucent_fill_is_unpremultiplied() {
        // Half-alpha red, premultiplied: rgb halved. SVG should recover full red
        // with 0.5 opacity.
        let color = Color32::from_rgba_unmultiplied(255, 0, 0, 128);
        let attr = paint_attr("fill", color).expect("visible");
        assert!(attr.contains("rgb(255,0,0)"), "got {attr}");
        assert!(attr.contains("fill-opacity"));
    }

    fn clipped(shape: Shape) -> ClippedShape {
        ClippedShape {
            clip_rect: egui::Rect::EVERYTHING,
            shape,
        }
    }

    fn vertex(pos: Pos2, color: Color32) -> Vertex {
        Vertex {
            pos,
            uv: WHITE_UV,
            color,
        }
    }
}
