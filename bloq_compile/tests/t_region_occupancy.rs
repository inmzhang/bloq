//! T regions keep their source layers and causal order across shared qubits.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::Bloq;
use bloq_test::{CompileReadyCase, compile_ready_test_cases};

#[test]
fn t_regions_keep_their_source_layers_through_modules_and_serialization() {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0

module TStage {
  out q: data = 2
  0: T [0, 0, 0]
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q>
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
}

module main {
  out q: data = 3
  out anchor: data = 5
  s: TStage @ [0, 0, 2]
  3: Port [0, 0, 4] role=output <q>
  4: XZX [5, 0, 0]
  5: Port [5, 0, 1] role=output <anchor>
  s.q -> 3
  4 -> 5
}
",
    )
    .expect("modules parse");
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&graph)
        .expect("modules compile")
        .bloq;
    for bloq in [
        Bloq::from_binary(&bloq.to_binary()).unwrap(),
        Bloq::from_text(&bloq.to_text()).unwrap(),
        bloq,
    ] {
        let mut regions = bloq
            .top()
            .nodes()
            .filter(|(_, node)| node.try_region().is_some());
        // Compilation normalizes z from the anchor, and source layers count double.
        assert_eq!(regions.next().map(|(_, node)| node.layer()), Some(4));
    }
}

/// Layout validation no longer checks this, so the occupancy pass must hold it.
#[test]
fn corpus_orders_every_region_overlap() {
    let context = CompileContext::new(CompileConfig::new(3));
    for case in compile_ready_test_cases(None).expect("build compilation corpus") {
        let id = case.id().to_string();
        let ready = CompileReadyCase::from_test_case(case).expect("corpus case builds");
        let bloq = context.compile(&ready.graph).expect("corpus compiles").bloq;
        let level = bloq.top();
        let regions = level
            .nodes()
            .filter(|(_, node)| node.try_region().is_some())
            .map(|(node_id, node)| (node_id, bloq.node_qubits(node).unwrap()))
            .collect::<Vec<_>>();
        for (index, (left, left_qubits)) in regions.iter().enumerate() {
            for (right, right_qubits) in &regions[index + 1..] {
                assert!(
                    !left_qubits.iter().any(|qubit| right_qubits.contains(qubit))
                        || level.has_path(*left, *right)
                        || level.has_path(*right, *left),
                    "{id}: {left:?} and {right:?} share a qubit unordered"
                );
            }
        }
    }
}
