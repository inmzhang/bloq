//! Compile one frozen QASM workload; measure process RSS with the calling runner.

use std::{collections::BTreeSet, error::Error, fs, path::Path, time::Instant};

use bloq_compile::CompileConfig;
use bloq_graph::{lower_blog_graph_ast_deferred, parse_blog_program_to_ast};
use bloq_interop::{GridOptions, LayoutOptions, SpiderReduction, Strategy, layout_qasm};
use bloq_ir::Bloq;
use serde_json::json;

const USAGE: &str = "usage: compile_qasm SOURCE.qasm topols|spiderls NEW_OUTPUT_DIR [DISTANCE=25] [EXTRA_PATCHES=0] [--extra-patches N] [--columns N] [--spacing N] [--routing-margin N] [--route-expansions N] [--batch N] [--reduction auto|conservative|full] [--fusion on|off] [--block-rows N] [--max-time N] [--max-blocks N]";

fn options(args: &[String]) -> Result<(LayoutOptions, u32), Box<dyn Error>> {
    if args.len() < 3 {
        return Err(USAGE.into());
    }
    let strategy = match args[1].as_str() {
        "topols" => Strategy::TopoLs,
        "spiderls" => Strategy::SpiderLs,
        _ => return Err("strategy must be topols or spiderls".into()),
    };
    let mut rest = &args[3..];
    let mut positional = || {
        if rest.first().is_some_and(|s| !s.starts_with("--")) {
            let value = &rest[0];
            rest = &rest[1..];
            Some(value)
        } else {
            None
        }
    };
    let distance = positional().map_or(Ok(25), |s| s.parse::<u32>())?;
    let positional_extra = positional().map(|s| s.parse::<usize>()).transpose()?;
    let mut options = LayoutOptions {
        strategy,
        extra_patches: positional_extra.unwrap_or(0),
        ..Default::default()
    };
    let mut seen = BTreeSet::new();
    let (pairs, remainder) = rest.as_chunks::<2>();
    for pair in pairs {
        let (flag, value) = (pair[0].as_str(), pair[1].as_str());
        if !seen.insert(flag) {
            return Err(format!("duplicate option {flag}").into());
        }
        match flag {
            "--extra-patches" => {
                if positional_extra.is_some() {
                    return Err("extra patches specified both positionally and by flag".into());
                }
                options.extra_patches = value.parse()?;
            }
            "--columns" => {
                options
                    .grid
                    .get_or_insert_with(GridOptions::default)
                    .columns = value.parse()?
            }
            "--spacing" => {
                options
                    .grid
                    .get_or_insert_with(GridOptions::default)
                    .spacing = value.parse()?
            }
            "--routing-margin" => {
                options
                    .grid
                    .get_or_insert_with(GridOptions::default)
                    .routing_margin = value.parse()?
            }
            "--route-expansions" => {
                options
                    .grid
                    .get_or_insert_with(GridOptions::default)
                    .max_route_expansions = value.parse()?
            }
            "--batch" => options.measurement_batch_size = Some(value.parse()?),
            "--max-time" => options.max_temporal_extent = Some(value.parse()?),
            "--max-blocks" => options.max_blocks = value.parse()?,
            "--block-rows" => options.topology.block_rows = value.parse()?,
            "--fusion" => {
                options.topology.fusion = match value {
                    "on" => true,
                    "off" => false,
                    _ => return Err("fusion must be on or off".into()),
                }
            }
            "--reduction" => {
                options.spider_reduction = match value {
                    "auto" => SpiderReduction::Auto,
                    "conservative" => SpiderReduction::Conservative,
                    "full" => SpiderReduction::Full,
                    _ => return Err("reduction must be auto, conservative or full".into()),
                }
            }
            _ => return Err(format!("unknown option {flag}; {USAGE}").into()),
        }
    }
    if !remainder.is_empty() {
        return Err("every flag needs a value".into());
    }
    if strategy == Strategy::TopoLs && seen.contains("--reduction") {
        return Err("--reduction applies only to spiderls".into());
    }
    if strategy == Strategy::SpiderLs
        && (seen.contains("--fusion") || seen.contains("--block-rows"))
    {
        return Err("--fusion and --block-rows apply only to topols".into());
    }
    options.validate()?;
    Ok((options, distance))
}

fn grid_json(grid: Option<GridOptions>) -> serde_json::Value {
    json!(grid.map(|g| json!({
        "columns": g.columns,
        "spacing": g.spacing,
        "routing_margin": g.routing_margin,
        "max_route_expansions": g.max_route_expansions,
    })))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (options, distance) = options(&args)?;
    let config = CompileConfig::try_new(distance)?.with_prepare_t_with_mpps(true);
    let matched_started = Instant::now();
    let source = fs::read_to_string(&args[0])?;
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    let started = Instant::now();
    let mut result = json!({
        "compiler": args[1], "status": "running", "distance": distance,
        "t_preparation": "ideal_t_proxy", "execution_verified": false,
        "max_blocks": options.max_blocks,
        "requested_grid": grid_json(options.grid),
        "measurement_batch_size": options.measurement_batch_size,
        "max_temporal_extent": options.max_temporal_extent,
        "spider_reduction": match options.spider_reduction { SpiderReduction::Auto => "auto", SpiderReduction::Conservative => "conservative", SpiderReduction::Full => "full" },
        "topology": {"fusion": options.topology.fusion, "block_rows": options.topology.block_rows},
        "extra_patch_budget": options.extra_patches,
    });
    let outcome = (|| -> Result<(), Box<dyn Error>> {
        result["failed_stage"] = "layout".into();
        let mut layout = layout_qasm(&source, &options)?;
        result["layout_seconds"] = started.elapsed().as_secs_f64().into();
        result["grid"] = grid_json(layout.grid);
        result["patch_locations"] = json!(
            layout
                .patch_locations
                .iter()
                .map(glam::IVec2::to_array)
                .collect::<Vec<_>>()
        );
        result["patch_rows"] = layout
            .patch_locations
            .iter()
            .map(|p| p.y)
            .collect::<BTreeSet<_>>()
            .len()
            .into();
        result["routing_bounds"] = json!(
            layout
                .routing_bounds
                .map(|bounds| bounds.map(|p| p.to_array()))
        );
        result["reduction"] = layout.reduction.into();
        result["reduction_fallback"] = json!(layout.reduction_fallback);
        let blog = layout.graph.to_blog_text();
        fs::write(output.join("layout.blog"), &blog)?;
        layout.graph = {
            let parsed = lower_blog_graph_ast_deferred(&parse_blog_program_to_ast(
                &fs::read_to_string(output.join("layout.blog"))?,
            )?)?;
            if parsed.to_blog_text() != blog {
                return Err("BLOG round trip changed the generated graph".into());
            }
            // Restore flat dispatch; reject any geometry or interface normalization.
            let flat = parsed.flatten()?;
            if flat.to_blog_text() != blog {
                return Err("BLOG flattening changed the generated graph".into());
            }
            flat
        };
        let columns: BTreeSet<_> = layout
            .graph
            .occupied_positions()
            .map(|p| (p.x, p.y))
            .collect();
        let (xs, ys, zs) = layout.graph.spans().ok_or("empty generated graph")?;
        let extents =
            [xs, ys, zs].map(|span| i64::from(*span.end()) - i64::from(*span.start()) + 1);
        result["blocks"] = layout.graph.block_count().into();
        result["occupied_cells"] = layout.graph.occupied_positions().count().into();
        result["occupied_columns"] = columns.len().into();
        result["anchor_columns"] = layout
            .graph
            .positions()
            .map(|p| (p.x, p.y))
            .collect::<BTreeSet<_>>()
            .len()
            .into();
        result["footprint"] = (extents[0] * extents[1]).into();
        result["temporal_extent"] = extents[2].into();
        result["extents"] = json!(extents);
        result["input_ports"] = json!(
            layout
                .input_ports
                .iter()
                .map(glam::IVec3::to_array)
                .collect::<Vec<_>>()
        );
        result["output_ports"] = json!(
            layout
                .output_ports
                .iter()
                .map(glam::IVec3::to_array)
                .collect::<Vec<_>>()
        );
        result["failed_stage"] = "ir".into();
        fs::write(
            output.join("result.json"),
            serde_json::to_string_pretty(&result)?,
        )?;
        let native_started = Instant::now();
        let artifacts = layout.compile(config)?;
        result["native_seconds"] = native_started.elapsed().as_secs_f64().into();
        let program = artifacts.bloq;
        let actual: BTreeSet<_> = program
            .logical_outputs()
            .iter()
            .map(|output| output.port.to_array())
            .collect();
        if actual
            != layout
                .output_ports
                .iter()
                .map(glam::IVec3::to_array)
                .collect()
            || program.logical_outputs().len() != layout.output_ports.len()
        {
            return Err("IR output interface differs from the source".into());
        }
        let binary = program.to_binary();
        let ir_path = output.join(format!("program_d{distance}.bloq"));
        fs::write(&ir_path, &binary)?;
        // Keep only one full IR in memory during persistence verification.
        drop(program);
        let persisted = Bloq::from_binary(&fs::read(ir_path)?)?;
        if persisted.to_binary() != binary {
            return Err("IR binary round trip changed the validated program".into());
        }
        result["matched_pipeline_seconds"] = matched_started.elapsed().as_secs_f64().into();
        result["physical_qubits"] = persisted.qubit_count()?.into();
        result["measurements"] = persisted.measurement_count().into();
        result["warnings"] = json!(artifacts.warnings);
        result["validated"] = true.into();
        result["blog_round_trip"] = true.into();
        result["ir_round_trip"] = true.into();
        result["status"] = "compiled".into();
        result
            .as_object_mut()
            .ok_or("result is not an object")?
            .remove("failed_stage");
        Ok(())
    })();
    result["pipeline_seconds"] = started.elapsed().as_secs_f64().into();
    if let Err(error) = &outcome {
        result["status"] = "failed".into();
        result["error"] = error.to_string().into();
    }
    fs::write(
        output.join("result.json"),
        serde_json::to_string_pretty(&result)?,
    )?;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &str) -> Result<(LayoutOptions, u32), Box<dyn Error>> {
        options(
            &arguments
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn controls_preserve_old_arguments_and_reject_ignored_or_invalid_options() {
        let (old, distance) = parse("in spiderls out 25 4").unwrap();
        assert_eq!(distance, 25);
        assert_eq!(old.grid.unwrap(), GridOptions::default());
        assert_eq!(old.extra_patches, 4);
        let (default, _) = parse("in spiderls out").unwrap();
        assert_eq!(default.extra_patches, 0);
        assert_eq!(default.grid.unwrap().columns, 0);
        assert_eq!(
            parse("in spiderls out --extra-patches 3")
                .unwrap()
                .0
                .extra_patches,
            3
        );
        let (grid, _) =
            parse("in topols out --columns 2 --spacing 6 --fusion off --batch 1").unwrap();
        assert_eq!(grid.grid.unwrap().columns, 2);
        assert!(!grid.topology.fusion);
        assert_eq!(grid.measurement_batch_size, Some(1));
        for arguments in [
            "in topols out --columns",
            "in topols out --spacing 3",
            "in topols out --columns 2 --columns 4",
            "in topols out --columns 18446744073709551616000",
            "in topols out --reduction auto",
            "in spiderls out --fusion on",
            "in spiderls out 25 4 --objective compact",
            "in spiderls out --objective volume",
            "in spiderls out 25 0 --extra-patches 2",
            "in spiderls out --extra-patches 2 --extra-patches 3",
            "in spiderls out --batch 0",
            "in spiderls out --unknown 4",
        ] {
            assert!(parse(arguments).is_err(), "{arguments}");
        }
    }
}
