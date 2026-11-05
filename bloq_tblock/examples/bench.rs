//! Compile QASM files under several configurations; print one JSON line each.
//!
//! usage: bench FILE.qasm... [--t-lead N] [--verify-compile D] [--verify-logical MAX_T] [--only CONFIG]

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
use bloq_interop::parse_qasm;
use bloq_tblock::{
    Deferral, MachineOptions, Options, Placement, Reduction, TMode, compile, lower, metrics, reduce,
};
use serde_json::json;

fn main() {
    let mut files = Vec::new();
    let mut t_lead = 0;
    let mut physical: Option<u32> = None;
    let mut logical: Option<usize> = None;
    let mut only: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--t-lead" => {
                t_lead = args
                    .next()
                    .expect("bench input is valid")
                    .parse()
                    .expect("bench input is valid")
            }
            "--verify-compile" => {
                physical = Some(
                    args.next()
                        .expect("bench input is valid")
                        .parse()
                        .expect("bench input is valid"),
                )
            }
            "--verify-logical" => {
                logical = Some(
                    args.next()
                        .expect("bench input is valid")
                        .parse()
                        .expect("bench input is valid"),
                )
            }
            "--only" => only = Some(args.next().expect("bench input is valid")),
            _ => files.push(arg),
        }
    }
    for path in files {
        let name = std::path::Path::new(&path)
            .file_stem()
            .expect("bench input is valid")
            .to_string_lossy()
            .to_string();
        let source = std::fs::read_to_string(&path).expect("bench input is valid");
        let circuit = match parse_qasm(&source) {
            Ok(c) => c,
            Err(e) => {
                println!("{}", json!({"circuit": name, "error": e}));
                continue;
            }
        };
        let k = circuit.num_qubits();
        let mut placements = vec![("strip".to_string(), Placement::Strip)];
        let square = (k as f64).sqrt().ceil() as usize;
        let mut widths: Vec<usize> = vec![square, square.div_ceil(2), square * 2, k.div_ceil(2)];
        widths.retain(|&w| w >= 1 && w <= k);
        widths.sort_unstable();
        widths.dedup();
        for w in widths {
            placements.push((format!("grid{w}"), Placement::Grid { columns: w }));
        }
        for (reduction, rname) in [(Reduction::Raw, "raw"), (Reduction::Zx, "zx")] {
            let reduced = reduce(&circuit, reduction);
            let program = lower(&reduced, Deferral::Full).expect("bench input is valid");
            let batches = program.batches();
            let t: usize = batches.iter().map(|b| b.1).sum();
            let rank: usize = batches.iter().map(|b| b.2).sum();
            for (deferral, dname) in [(Deferral::Full, "defer"), (Deferral::None, "inplace")] {
                for (t_mode, tname) in [(TMode::Gadget, "gadget"), (TMode::Block, "block")] {
                    if deferral == Deferral::None && t_mode == TMode::Block {
                        continue;
                    }
                    for (pname, placement) in &placements {
                        let label = format!("{rname}/{dname}/{tname}/{pname}");
                        if only
                            .as_ref()
                            .is_some_and(|o| !label.starts_with(o.as_str()))
                        {
                            continue;
                        }
                        let options = Options {
                            reduction,
                            deferral,
                            machine: MachineOptions {
                                t_lead,
                                t_mode,
                                placement: *placement,
                                ..Default::default()
                            },
                        };
                        let start = std::time::Instant::now();
                        let built = compile(&circuit, &options);
                        let seconds = start.elapsed().as_secs_f64();
                        let mut row = json!({
                            "circuit": name, "qubits": k, "reduction": rname,
                            "deferral": dname, "t_mode": tname, "placement": pname,
                            "t_lead": t_lead, "t_terms": t, "t_rank": rank,
                            "layout_seconds": seconds,
                        });
                        match built {
                            Ok(built) => {
                                let m = metrics(&built.graph);
                                row["extents"] = json!(m.extents);
                                row["volume"] = json!(m.volume);
                                row["occupied"] = json!(m.occupied);
                                row["t_blocks"] = json!(m.t_blocks);
                                row["selective"] = json!(m.selective);
                                row["blocks"] = json!(m.blocks);
                                row["checks"] = json!(built.checks);
                                if logical.is_some_and(|max| m.t_blocks <= max) {
                                    let start = std::time::Instant::now();
                                    let ok =
                                        LogicalVerifier::with_boundaries_and_internal_measurements(
                                            &built.graph,
                                            BoundaryOrder::new(
                                                built.inputs.clone(),
                                                built.outputs.clone(),
                                            ),
                                        )
                                        .map_err(|e| e.to_string())
                                        .and_then(|v| {
                                            v.verify(&circuit.to_graph::<QuizxGraph>(), 8, 7)
                                                .map_err(|e| e.to_string())
                                        });
                                    row["logical_ok"] = json!(ok.is_ok());
                                    row["logical_seconds"] = json!(start.elapsed().as_secs_f64());
                                    if let Err(e) = ok {
                                        row["logical_error"] = json!(e);
                                    }
                                }
                                if let Some(d) = physical {
                                    let ok = CompileContext::new(
                                        CompileConfig::new(d).with_prepare_t_with_mpps(true),
                                    )
                                    .compile_and_validate(&built.graph)
                                    .map(|_| true)
                                    .unwrap_or_else(|e| {
                                        row["compile_error"] = json!(e.to_string());
                                        false
                                    });
                                    row["physical_ok"] = json!(ok);
                                }
                            }
                            Err(e) => row["error"] = json!(e),
                        }
                        println!("{row}");
                    }
                }
            }
        }
    }
}
