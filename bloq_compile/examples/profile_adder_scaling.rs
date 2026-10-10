//! Cold compile probe: `profile_adder_scaling BITS|yoke:WIDTH DISTANCE [--warm]`.
//! Run each cold sample in a fresh process. Stage events go to stderr as they
//! happen, so a failed or interrupted large run still shows its last stage.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_ir::Bloq;
use bloq_test::benchmark::controlled_adder;

#[derive(Default)]
struct ProgramStats {
    ir_nodes: usize,
    ir_edges: usize,
    classical_nodes: usize,
    readout_terms: usize,
    template_instances: usize,
    max_quantum_node_instances: usize,
    inline_detector_rows: usize,
    bundle_definitions: usize,
    bundle_rows: usize,
    bundle_uses: usize,
    expanded_detector_rows: usize,
}

fn program_stats(bloq: &Bloq) -> ProgramStats {
    let mut stats = ProgramStats {
        bundle_definitions: bloq.detector_bundles().len(),
        bundle_rows: bloq
            .detector_bundles()
            .iter()
            .map(|(_, bundle)| bundle.detectors().len())
            .sum(),
        ..ProgramStats::default()
    };
    for (_, level) in bloq.levels() {
        stats.ir_nodes += level.node_count();
        stats.ir_edges += level.edge_count();
        for (_, node) in level.nodes() {
            if let Some(classical) = node.try_classical() {
                stats.classical_nodes += 1;
                stats.readout_terms += classical.measurements().len() + classical.operators().len();
            }
        }
        for (_, node) in level.quantum_nodes() {
            stats.template_instances += node.instances.len();
            stats.max_quantum_node_instances =
                stats.max_quantum_node_instances.max(node.instances.len());
            stats.inline_detector_rows += node.detectors.len();
            stats.bundle_uses += node.detector_bundles.len();
            stats.expanded_detector_rows += bloq
                .node_detector_count(node)
                .expect("valid compiled detector bundles");
        }
    }
    stats
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        matches!(args.len(), 3 | 4) && args.get(3).is_none_or(|arg| arg == "--warm"),
        "usage: profile_adder_scaling BITS|yoke:WIDTH DISTANCE [--warm]"
    );
    let yoke = args[1].strip_prefix("yoke:");
    let bits = yoke.unwrap_or(&args[1]).parse::<usize>().expect("width");
    assert!(
        bits >= if yoke.is_some() { 2 } else { 3 },
        "benchmark width is too small"
    );
    let distance = args[2].parse::<u32>().expect("code distance");
    let source_start = Instant::now();
    let (source_kind, program) = if yoke.is_some() {
        ("yoked_memory", bloq_test::benchmark::yoked_memory(bits))
    } else {
        ("controlled_adder", controlled_adder(bits))
    };
    let source_ms = source_start.elapsed().as_secs_f64() * 1_000.0;
    let compile_start = Instant::now();
    let warm = Arc::new(AtomicBool::new(false));
    let observer_warm = Arc::clone(&warm);
    let context =
        CompileContext::new(CompileConfig::new(distance)).with_progress_observer(move |stage| {
            eprintln!(
                "{{\"event\":\"stage\",\"run\":\"{}\",\"stage\":\"{stage:?}\",\"elapsed_ms\":{}}}",
                if observer_warm.load(Ordering::Relaxed) {
                    "warm"
                } else {
                    "cold"
                },
                compile_start.elapsed().as_secs_f64() * 1_000.0
            );
        });
    for run in ["cold", "warm"].into_iter().take(args.len() - 2) {
        warm.store(run == "warm", Ordering::Relaxed);
        let start = if run == "cold" {
            compile_start
        } else {
            Instant::now()
        };
        let compiled = context
            .compile(&program)
            .unwrap_or_else(|error| panic!("compile {run} adder: {error}"));
        let compile_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let stats = program_stats(&compiled.bloq);
        let source = if run == "cold" {
            format!(",\"source_ms\":{source_ms}")
        } else {
            String::new()
        };
        println!(
            "{{\"event\":\"result\",\"run\":\"{run}\",\"bits\":{bits},\"distance\":{distance},\"source_kind\":\"{source_kind}\"{source},\"compile_ms\":{compile_ms},\"quantum_nodes\":{},\"templates\":{},\"measurements\":{},\"template_instances\":{},\"max_quantum_node_instances\":{},\"inline_detector_rows\":{},\"bundle_definitions\":{},\"bundle_rows\":{},\"bundle_uses\":{},\"expanded_detector_rows\":{},\"ir_nodes\":{},\"ir_edges\":{},\"classical_nodes\":{},\"readout_terms\":{}}}",
            compiled.bloq.quantum_node_count(),
            compiled.bloq.templates().len(),
            compiled.bloq.measurement_count(),
            stats.template_instances,
            stats.max_quantum_node_instances,
            stats.inline_detector_rows,
            stats.bundle_definitions,
            stats.bundle_rows,
            stats.bundle_uses,
            stats.expanded_detector_rows,
            stats.ir_nodes,
            stats.ir_edges,
            stats.classical_nodes,
            stats.readout_terms
        );
    }
}
