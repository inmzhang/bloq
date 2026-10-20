//! Check an imported upstream layout against its Clifford+T source with QuiZX.
//!
//! Reads `layout.blog`, `clifford_t.qasm`, and the ports in `result.json` from
//! an interop run directory and prints a JSON report of sampled branches.

use bloq_graph::BlockGraph;
use bloq_graph::verify::{BoundaryOrder, LogicalVerifier};
use glam::IVec3;
use serde::Deserialize;

#[path = "../tests/common/mod.rs"]
mod common;

#[derive(Deserialize)]
struct Ports {
    input_ports: Vec<[i32; 3]>,
    output_ports: Vec<[i32; 3]>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(
        args.next()
            .expect("usage: verify_interop_layout RUN_DIR [SAMPLES=32] [SEED=7]"),
    );
    let samples = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(32);
    let seed = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(7);
    let graph = BlockGraph::load(dir.join("layout.blog"))?;
    let qasm = std::fs::read_to_string(dir.join("clifford_t.qasm"))?;
    let ports: Ports = serde_json::from_str(&std::fs::read_to_string(dir.join("result.json"))?)?;
    let points = |ports: Vec<[i32; 3]>| ports.into_iter().map(IVec3::from).collect();
    let boundaries = BoundaryOrder::new(points(ports.input_ports), points(ports.output_ports));
    let expected = common::parse_qasm(&qasm)?.to_graph();
    let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(&graph, boundaries)?;
    let report = verifier.verify(&expected, samples, seed)?;
    println!(
        "{}",
        serde_json::json!({
            "samples": report.samples,
            "verified": report.verified,
            "rejected": report.rejected,
            "impossible": report.impossible,
            "seed": seed,
        })
    );
    Ok(())
}
