// Apple ld's compact-unwind table tops out at 16 MiB for large debug binaries.
#![cfg_attr(
    target_os = "macos",
    allow(
        linker_messages,
        reason = "Apple ld overflows compact-unwind tables for this debug binary"
    )
)]
#![expect(
    clippy::too_many_arguments,
    reason = "editor functions combine independent Bevy and UI state"
)]

//! Standalone Bevy-based visual editor for surface code lattice-surgery block
//! graphs.
//!
//! `bloq_editor` lets users author, edit, and inspect `bloq_graph::BlockGraph`s
//! in 3D, then validate and compile them (via `bloq_compile`/`bloq_stim`) to
//! download output or drive the in-app circuit and ZX viewers. It targets both
//! native desktop and web/WASM; the same `BlockGraph` format is shared with the
//! `bloq` CLI.
//!
//! State lives in Bevy resources (the `resources` module); behavior lives in
//! systems (the `systems` module) wired together by `plugins::BloqEditorPlugin`.

use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy_egui::{EguiGlobalSettings, EguiPlugin};
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsString;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
mod app_icon;
mod components;
mod module_authoring;
mod pipe_planning;
mod plugins;
mod program_view;
mod resources;
#[cfg(any(target_arch = "wasm32", test))]
mod session;
mod svg_export;
mod systems;
mod theme;
mod utils;

use bevy_rich_text3d::{LoadFonts, Text3dPlugin};
#[cfg(not(target_arch = "wasm32"))]
use color_eyre::eyre;
use plugins::BloqEditorPlugin;

// 3D labels reuse the Zed Mono subset; non-ASCII labels have no glyph.
const EMBEDDED_FONT: &[u8] = include_bytes!("../assets/fonts/ZedMono-Regular.ttf");

fn main() {
    // The same WASM module also runs inside the compilation Worker.
    #[cfg(target_arch = "wasm32")]
    if web_sys::window().is_none() {
        return;
    }
    color_eyre::install().expect("install color-eyre hooks");
    #[cfg(not(target_arch = "wasm32"))]
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    #[cfg(not(target_arch = "wasm32"))]
    let gallery_export = match gallery_export_from_args(&args) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("{error:?}");
            std::process::exit(2);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let automated_render = match if gallery_export.is_some() {
        Ok(None)
    } else {
        automated_render_from_args(args)
    } {
        Ok(request) => request,
        Err(error) => {
            eprintln!("{error:?}");
            std::process::exit(2);
        }
    };
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::render::RenderPlugin {
                // Automated capture must wait for shaders instead of racing compilation.
                #[cfg(not(target_arch = "wasm32"))]
                synchronous_pipeline_compilation: automated_render.is_some() || gallery_export.is_some(),
                ..default()
            })
            .set(WindowPlugin {
                #[cfg(not(target_arch = "wasm32"))]
                close_when_requested: false,
                primary_window: Some(Window {
                    title: "Bloq Editor".into(),
                    name: Some("bloq_editor".into()),
                    fit_canvas_to_parent: true,
                    // Browser copy/paste events must reach bevy_egui's clipboard backend.
                    prevent_default_event_handling: false,
                    #[cfg(not(target_arch = "wasm32"))]
                    visible: gallery_export.is_none(),
                    #[cfg(not(target_arch = "wasm32"))]
                    resolution: if automated_render.is_some() {
                        bevy::window::WindowResolution::new(1600, 1200)
                    } else {
                        default()
                    },
                    // FIFO can spuriously time out on NVIDIA's Wayland Vulkan path.
                    #[cfg(target_os = "linux")]
                    present_mode: bevy::window::PresentMode::Mailbox,
                    #[cfg(target_os = "linux")]
                    desired_maximum_frame_latency: std::num::NonZeroU32::new(1),
                    ..default()
                }),
                ..default()
            })
            .set(LogPlugin {
                level: bevy::log::Level::INFO,
                filter: "wgpu=error,naga=warn,bevy_picking=warn,bevy_app=info,bevy_ecs::system::system=error,calloop=error,rust_sugiyama=warn"
                    .to_string(),
                ..default()
            }),
    )
    .add_plugins(MeshPickingPlugin)
    .add_plugins(EguiPlugin::default())
    .insert_resource(EguiGlobalSettings {
        auto_create_primary_context: false,
        ..default()
    })
    .add_plugins(Text3dPlugin {
        default_atlas_dimension: (1024, 1024),
        ..Default::default()
    })
    .insert_resource(LoadFonts {
        font_embedded: vec![EMBEDDED_FONT],
        ..Default::default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(request) = automated_render {
        let source = std::fs::read_to_string(&request.input).unwrap_or_else(|error| {
            eprintln!("failed to read {}: {error}", request.input.display());
            std::process::exit(2);
        });
        let graph = bloq_graph::BlockGraph::from_blog_text(&source).unwrap_or_else(|error| {
            eprintln!("failed to parse {}: {error}", request.input.display());
            std::process::exit(2);
        });
        let graph = graph.fix_shadowed_faces();
        if let Some(indices) = request.stabilizer {
            let generators = graph
                .stabilizers()
                .unwrap_or_else(|error| {
                    eprintln!("failed to derive correlation surfaces: {error}");
                    std::process::exit(2);
                })
                .generators;
            if indices.iter().any(|&index| index >= generators.len()) {
                eprintln!(
                    "stabilizer indices {indices:?} out of range ({} generators)",
                    generators.len()
                );
                std::process::exit(2);
            }
            let mut selected = generators[indices[0]].clone();
            for &index in &indices[1..] {
                selected
                    .stabilizer
                    .phase_free_mul_assign(&generators[index].stabilizer);
            }
            println!(
                "Rendered surface {indices:?}: {:?}",
                selected.stabilizer.port_stabilizer
            );
            app.insert_resource(resources::EditorState {
                stabilizers: vec![selected],
                current_stabilizer_index: 0,
                show_stabilizers: true,
                ..default()
            });
        }
        if !request.pop_faces.is_empty() {
            app.insert_resource(systems::screenshot::AutomatedCutaway(request.pop_faces));
        }
        let mut graph_state = resources::GraphState { graph, ..default() };
        graph_state.commit();
        app.insert_resource(graph_state).insert_resource(
            systems::screenshot::AutomatedScreenshot::new(
                request.output,
                request.azimuth,
                request.elevation,
            ),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(output) = gallery_export {
        std::fs::create_dir_all(&output).unwrap_or_else(|error| {
            eprintln!("failed to create {}: {error}", output.display());
            std::process::exit(2);
        });
        app.insert_resource(systems::thumbnails::GalleryThumbnailExport::new(output));
        app.insert_resource(bevy::winit::WinitSettings::continuous());
    }

    app.add_plugins(BloqEditorPlugin);
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, app_icon::set_window_icons);
    #[cfg(target_os = "macos")]
    app.add_systems(Startup, app_icon::set_dock_icon);
    // Dev-native keeps Bevy's default `panic` error handler: a broken
    // invariant should be loud at the desk. Release and WASM must keep the
    // frame loop alive (a WASM panic is a frozen canvas), so failed systems
    // route to log + toast instead.
    if cfg!(target_arch = "wasm32") || !cfg!(debug_assertions) {
        app.set_error_handler(plugins::report_internal_error);
    }
    app.run();
}

#[cfg(not(target_arch = "wasm32"))]
fn gallery_export_from_args(args: &[OsString]) -> eyre::Result<Option<PathBuf>> {
    if args
        .first()
        .is_none_or(|flag| flag != "--export-gallery-thumbnails")
    {
        return Ok(None);
    }
    let [_, output] = args else {
        eyre::bail!("usage: bloq_editor --export-gallery-thumbnails OUTPUT_DIR");
    };
    Ok(Some(output.into()))
}

#[cfg(not(target_arch = "wasm32"))]
struct AutomatedRenderRequest {
    input: PathBuf,
    output: PathBuf,
    azimuth: Option<f32>,
    elevation: Option<f32>,
    pop_faces: Vec<bloq_graph::Direction>,
    stabilizer: Option<Vec<usize>>,
}

#[cfg(not(target_arch = "wasm32"))]
fn automated_render_from_args(
    args: impl IntoIterator<Item = OsString>,
) -> eyre::Result<Option<AutomatedRenderRequest>> {
    use color_eyre::eyre::{ContextCompat as _, WrapErr as _, bail};

    let mut args = args.into_iter();
    let Some(flag) = args.next() else {
        return Ok(None);
    };
    if flag != "--render-blog" {
        bail!(
            "usage: bloq_editor --render-blog INPUT.blog --output OUTPUT.png [--azimuth DEGREES] [--elevation DEGREES] [--stabilizer INDICES] [--pop-face +x|-x|+y|-y|+z|-z]"
        );
    }
    let input = args
        .next()
        .wrap_err("missing BLOG path after --render-blog")?;
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--output")) {
        bail!("expected --output after the BLOG path");
    }
    let output = args.next().wrap_err("missing PNG path after --output")?;
    let mut azimuth = None;
    let mut elevation = None;
    let mut stabilizer = None;
    let mut pop_faces = Vec::new();
    while let Some(flag) = args.next() {
        let value = args.next().wrap_err("missing render option value")?;
        let value = value.to_str().wrap_err("render option must be UTF-8")?;
        if flag == "--azimuth" {
            let degrees: f32 = value.parse().wrap_err("azimuth must be a number")?;
            if !degrees.is_finite() {
                bail!("azimuth must be finite");
            }
            azimuth = Some((degrees % 360.0).to_radians());
        } else if flag == "--elevation" {
            let degrees: f32 = value.parse().wrap_err("elevation must be a number")?;
            if !degrees.is_finite() || degrees <= -90.0 || degrees >= 90.0 {
                bail!("elevation must be finite and strictly between -90 and 90 degrees");
            }
            elevation = Some(degrees.to_radians());
        } else if flag == "--pop-face" {
            use bloq_graph::Direction;
            pop_faces.push(match value {
                "+x" => Direction::XPLUS,
                "-x" => Direction::XMINUS,
                "+y" => Direction::YPLUS,
                "-y" => Direction::YMINUS,
                "+z" => Direction::ZPLUS,
                "-z" => Direction::ZMINUS,
                _ => bail!("pop-face must be +x, -x, +y, -y, +z, or -z"),
            });
        } else if flag == "--stabilizer" {
            stabilizer = Some(
                value
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<usize>, _>>()
                    .wrap_err("stabilizer must be comma-separated indices")?,
            );
        } else {
            bail!("unknown render option {flag:?}");
        }
    }
    Ok(Some(AutomatedRenderRequest {
        input: input.into(),
        output: output.into(),
        azimuth,
        elevation,
        pop_faces,
        stabilizer,
    }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn gallery_export_requires_one_directory_and_leaves_other_modes_alone() {
        assert_eq!(
            gallery_export_from_args(&["--export-gallery-thumbnails".into(), "images".into()])
                .unwrap(),
            Some(PathBuf::from("images"))
        );
        for args in [
            vec!["--export-gallery-thumbnails".into()],
            vec![
                "--export-gallery-thumbnails".into(),
                "images".into(),
                "extra".into(),
            ],
        ] {
            let error = gallery_export_from_args(&args).unwrap_err();
            assert_eq!(
                error.to_string(),
                "usage: bloq_editor --export-gallery-thumbnails OUTPUT_DIR"
            );
        }
        assert!(
            gallery_export_from_args(&["--render-blog".into()])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parses_cutaway_directions_and_rejects_unknown_faces() {
        for (face, expected) in [
            ("+x", Some(bloq_graph::Direction::XPLUS)),
            ("-z", Some(bloq_graph::Direction::ZMINUS)),
            ("x", None),
        ] {
            let request = automated_render_from_args([
                "--render-blog".into(),
                "cube.blog".into(),
                "--output".into(),
                "cube.png".into(),
                "--pop-face".into(),
                face.into(),
            ]);
            if let Some(direction) = expected {
                assert_eq!(request.unwrap().unwrap().pop_faces, vec![direction]);
            } else {
                assert!(request.is_err());
            }
        }
    }

    #[test]
    fn parses_automated_render_arguments() {
        let request = automated_render_from_args([
            "--render-blog".into(),
            "cube.blog".into(),
            "--output".into(),
            "cube.png".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(request.input, PathBuf::from("cube.blog"));
        assert_eq!(request.output, PathBuf::from("cube.png"));
        assert_eq!(request.azimuth, None);
        assert_eq!(request.stabilizer, None);
        let surface = automated_render_from_args([
            "--render-blog".into(),
            "cube.blog".into(),
            "--output".into(),
            "cube.png".into(),
            "--stabilizer".into(),
            "2,3".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(surface.stabilizer, Some(vec![2, 3]));
        for (angle, expected) in [
            ("45", Some(std::f32::consts::FRAC_PI_4)),
            ("NaN", None),
            ("inf", None),
        ] {
            let result = automated_render_from_args([
                "--render-blog".into(),
                "cube.blog".into(),
                "--output".into(),
                "cube.png".into(),
                "--azimuth".into(),
                angle.into(),
            ]);
            match expected {
                Some(expected) => assert_eq!(result.unwrap().unwrap().azimuth, Some(expected)),
                None => assert!(result.is_err()),
            }
        }
    }

    #[test]
    fn parses_render_elevation() {
        for angle in ["-35", "0", "35", "-90", "90", "NaN", "inf"] {
            let result = automated_render_from_args([
                "--render-blog".into(),
                "cube.blog".into(),
                "--output".into(),
                "cube.png".into(),
                "--elevation".into(),
                angle.into(),
            ]);
            let degrees: f32 = angle.parse().unwrap();
            if degrees.is_finite() && degrees.abs() < 90.0 {
                assert_eq!(
                    result.unwrap().unwrap().elevation,
                    Some(degrees.to_radians())
                );
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn rejects_incomplete_automated_render_arguments() {
        assert!(automated_render_from_args(["--render-blog".into()]).is_err());
    }
}
