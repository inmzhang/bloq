use color_eyre::eyre::{self, WrapErr};

use bloq_graph::GltfFaceSelector;

use super::atomic_write_with;
use super::compile::CompileSource;
use crate::{ViewCmd, terminal};

fn run_ir(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    include_classical: bool,
    quiet: bool,
) -> eyre::Result<()> {
    let default_output = input.with_extension("svg");
    let output = output.unwrap_or(&default_output);
    super::reject_input_output_alias(input, output)?;
    let program = super::load_bloq_ir(input)?;
    super::atomic_write(output, program.to_svg(include_classical).as_bytes())
        .wrap_err("write IR SVG")?;
    if !quiet {
        terminal::status("Generated", format!("IR graph -> {}", output.display()));
    }
    Ok(())
}

pub(crate) fn run(cmd: ViewCmd, quiet: bool) -> eyre::Result<()> {
    let ViewCmd { source, options } = cmd;
    let ir_input = source.input.as_ref().is_some_and(|input| {
        matches!(
            input.extension().and_then(|ext| ext.to_str()),
            Some(bloq_ir::BLOQ_BINARY_EXTENSION | bloq_ir::BLOQ_TEXT_EXTENSION)
        )
    });
    if options.svg || ir_input {
        eyre::ensure!(
            !options.gltf && !options.html && !options.module_view,
            "saved Bloq IR supports SVG views only"
        );
        let input = source
            .input
            .as_deref()
            .ok_or_else(|| eyre::eyre!("SVG visualization requires a saved Bloq IR input file"))?;
        return run_ir(
            input,
            options.output.as_deref(),
            options.include_classical,
            quiet,
        );
    }
    eyre::ensure!(
        !options.include_classical && options.output.is_none(),
        "--include-classical and --output apply to saved Bloq IR SVG views"
    );
    let source = source.resolve()?;
    let loaded = source.load_graph()?;
    let graph = if options.module_view {
        loaded
    } else {
        loaded.flatten().wrap_err("flatten source for rendering")?
    };
    // With no explicit selection, `bloq view` generates every view kind.
    let (gltf, html) = if options.gltf || options.html {
        (options.gltf, options.html)
    } else {
        (true, true)
    };
    let outputs = [(gltf, "gltf"), (html, "html")]
        .into_iter()
        .filter(|(enabled, _)| *enabled)
        .map(|(_, extension)| source.default_output_path_with_extension(extension));
    if let CompileSource::InputFile(input) = &source {
        for output in outputs {
            super::reject_input_output_alias(input, &output)?;
        }
    }
    let mut failures = 0usize;
    let popped_faces = options
        .pop_faces_at_directions
        .into_iter()
        .map(GltfFaceSelector::All)
        .collect::<Vec<_>>();
    if gltf {
        let output = source.default_output_path_with_extension("gltf");
        if let Err(error) = atomic_write_with(&output, |temporary| {
            let result = if options.module_view {
                graph.write_module_gltf_file(options.pipe_length, temporary, &popped_faces)
            } else {
                graph.write_to_gltf_file(options.pipe_length, temporary, None, &popped_faces)
            };
            result.map_err(std::io::Error::other)
        })
        .wrap_err("write glTF file")
        {
            failures += 1;
            terminal::error(format!(
                "generate glTF view at {}: {error:#}",
                output.display()
            ));
        } else if !quiet {
            terminal::status("Generated", format!("glTF model -> {}", output.display()));
        }
    }

    if html {
        let output = source.default_output_path_with_extension("html");
        if let Err(error) = atomic_write_with(&output, |temporary| {
            let result = if options.module_view {
                graph.write_module_html_viewer(options.pipe_length, temporary, &popped_faces)
            } else {
                graph.write_to_gltf_html_viewer(options.pipe_length, temporary, None, &popped_faces)
            };
            result.map_err(std::io::Error::other)
        })
        .wrap_err("write HTML viewer")
        {
            failures += 1;
            terminal::error(format!(
                "generate HTML viewer at {}: {error:#}",
                output.display()
            ));
        } else if !quiet {
            terminal::status("Generated", format!("HTML viewer -> {}", output.display()));
        }
    }

    if failures == 0 {
        Ok(())
    } else {
        eyre::bail!("{failures} view output(s) failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SourceArgs, ViewArgs};

    #[test]
    fn all_view_outputs_are_checked_before_writing() {
        for extension in ["gltf", "html"] {
            for alias in ["direct", "hardlink", "symlink"] {
                let dir = tempfile::tempdir().unwrap();
                let input = dir.path().join(if alias == "direct" {
                    format!("source.{extension}")
                } else {
                    "source.blog".into()
                });
                let original = bloq_graph::GalleryItem::CNOT.entry().blog();
                std::fs::write(&input, original).unwrap();
                let output = input.with_extension(extension);
                match alias {
                    "hardlink" => std::fs::hard_link(&input, &output).unwrap(),
                    #[cfg(unix)]
                    "symlink" => std::os::unix::fs::symlink(&input, &output).unwrap(),
                    "direct" => {}
                    _ => continue,
                }
                let error = run(
                    ViewCmd {
                        source: SourceArgs {
                            input: Some(input.clone()),
                            gallery: None,
                        },
                        options: ViewArgs {
                            svg: false,
                            include_classical: false,
                            output: None,
                            gltf: true,
                            html: true,
                            module_view: false,
                            pipe_length: 1.0,
                            pop_faces_at_directions: Vec::new(),
                        },
                    },
                    true,
                )
                .unwrap_err();
                assert!(error.to_string().contains("aliases the input file"));
                assert_eq!(std::fs::read_to_string(&input).unwrap(), original);
                let other = input.with_extension(if extension == "html" { "gltf" } else { "html" });
                assert!(!other.exists(), "preflight must precede either export");
            }
        }
    }
}
