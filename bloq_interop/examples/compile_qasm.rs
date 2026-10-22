//! Compile one frozen QASM workload; measure process RSS with the calling runner.

use std::{collections::BTreeSet, error::Error, fs, path::Path, time::Instant};

use bloq_compile::CompileConfig;
use bloq_graph::BlockGraph;
use bloq_interop::{LayoutOptions, Strategy, layout_qasm};
use bloq_ir::Bloq;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: compile_qasm SOURCE.qasm topols|spiderls NEW_OUTPUT_DIR [DISTANCE=25]".into(),
        );
    }
    let strategy = match args[1].as_str() {
        "topols" => Strategy::TopoLs,
        "spiderls" => Strategy::SpiderLs,
        _ => return Err("strategy must be topols or spiderls".into()),
    };
    let distance = args.get(3).map_or(Ok(25), |s| s.parse::<u32>())?;
    let config = CompileConfig::try_new(distance)?.with_prepare_t_with_mpps(true);
    let source = fs::read_to_string(&args[0])?;
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    let started = Instant::now();
    let mut result = json!({
        "compiler": args[1], "status": "running", "distance": distance,
        "t_preparation": "ideal_t_proxy", "execution_verified": false,
        "max_blocks": LayoutOptions::default().max_blocks,
    });
    let outcome = (|| -> Result<(), Box<dyn Error>> {
        result["failed_stage"] = "layout".into();
        let layout = layout_qasm(
            &source,
            &LayoutOptions {
                strategy,
                ..Default::default()
            },
        )?;
        result["layout_seconds"] = started.elapsed().as_secs_f64().into();
        result["reduction"] = layout.reduction.into();
        result["reduction_fallback"] = json!(layout.reduction_fallback);
        let blog = layout.graph.to_blog_text();
        fs::write(output.join("layout.blog"), &blog)?;
        if BlockGraph::from_text(&fs::read_to_string(output.join("layout.blog"))?)?.to_blog_text()
            != blog
        {
            return Err("BLOG round trip changed the generated graph".into());
        }
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
        result["physical_qubits"] = program.qubit_count()?.into();
        result["measurements"] = program.measurement_count().into();
        let binary = program.to_binary();
        let ir_path = output.join(format!("program_d{distance}.bloq"));
        fs::write(&ir_path, &binary)?;
        // Keep only one full IR in memory during persistence verification.
        drop(program);
        if Bloq::from_binary(&fs::read(ir_path)?)?.to_binary() != binary {
            return Err("IR binary round trip changed the validated program".into());
        }
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
