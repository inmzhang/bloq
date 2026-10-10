//! Profile a saved artifact: `profile_ir INPUT [describe|validate|plans|decode] [ITERATIONS]`.
//! File I/O and initial decoding are outside the timed operation. Use `perf` or
//! `samply` for CPU profiles, or `/usr/bin/time -v` for process RSS.

use std::{hint::black_box, time::Instant};

use bloq_ir::{
    Bloq, BloqEdge, BloqNode, ClassicalExpr, ClassicalNode, lowering::InstantiationOptions,
};

fn expression_storage(expr: &ClassicalExpr, cells: &mut usize, slots: &mut usize) {
    *cells += 1;
    if let ClassicalExpr::Parity { inputs, .. } = expr {
        *slots += inputs.len();
    }
    for operand in expr.operands() {
        expression_storage(operand, cells, slots);
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 2,
        "INPUT [describe|validate|plans|decode] [ITERATIONS]"
    );
    let bytes = std::fs::read(&args[1]).expect("read artifact");
    let program = Bloq::from_binary(&bytes).expect("decode artifact");
    let operation = args.get(2).map_or("describe", String::as_str);
    let iterations = args
        .get(3)
        .map_or(1, |value| value.parse::<usize>().expect("iterations"));
    assert!(iterations > 0, "iterations must be positive");
    if operation == "describe" {
        program.validate().expect("valid artifact");
        let mut nodes = [0usize; 3];
        let mut edges = [0usize; 4];
        let mut records = 0;
        let (mut expression_cells, mut parity_slots) = (0, 0);
        for (_, level) in program.levels() {
            for (_, node) in level.nodes() {
                nodes[if node.try_quantum().is_some() {
                    0
                } else if node.try_region().is_some() {
                    2
                } else {
                    1
                }] += 1;
                if let Some(ClassicalNode::Observable { measurements, .. }) = node.try_classical() {
                    records += measurements.len();
                }
                if let Some(
                    ClassicalNode::Compute { expr } | ClassicalNode::Discard { condition: expr },
                ) = node.try_classical()
                {
                    expression_storage(expr, &mut expression_cells, &mut parity_slots);
                }
            }
            for edge in level.edges() {
                edges[match edge.edge {
                    BloqEdge::Quantum(_) => 0,
                    BloqEdge::Value { .. } => 1,
                    BloqEdge::Order => 2,
                    BloqEdge::Compose { .. } => 3,
                }] += 1;
            }
        }
        println!(
            "{{\"nodes_quantum_classical_region\":{nodes:?},\"edges_quantum_value_order_compose\":{edges:?},\"record_terms\":{records},\"expression_cells\":{expression_cells},\"parity_slots\":{parity_slots},\"templates\":{},\"binary_bytes\":{},\"text_bytes\":{},\"node_bytes\":{},\"edge_bytes\":{}}}",
            program.templates().len(),
            bytes.len(),
            program.to_text().len(),
            size_of::<BloqNode>(),
            size_of::<BloqEdge>()
        );
        return;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        match operation {
            "validate" => program.validate().expect("valid artifact"),
            "plans" => {
                black_box(
                    program
                        .validate_with_plans(&InstantiationOptions::default())
                        .expect("valid artifact"),
                );
            }
            "decode" => {
                black_box(Bloq::from_binary(&bytes).expect("decode artifact"));
            }
            _ => panic!("unknown operation {operation:?}"),
        }
    }
    eprintln!(
        "{operation}: {iterations} iterations, {:.3} ms/iteration",
        started.elapsed().as_secs_f64() * 1000.0 / iterations as f64
    );
}
