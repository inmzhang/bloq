//! T regions keep their source layers and causal order across shared qubits.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::Bloq;
use bloq_test::{CompileReadyCase, compile_ready_test_cases};
use rustc_hash::{FxHashMap, FxHashSet};

const T_MODULES: &str = "BLOG 1.0

module TStage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, 0] role=input <q_in>
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q_out>
  3: T [1, 0, 0]
  4: XZX [1, 0, 1]
  5: YX [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 0] -> +Z
  [1, 0, 2] -> -Z
  mzz = measure 1 -> +X
  resolve 5 if mzz
}

module main {
  in q_in: data = 100
  out q_out: data = 101
  s0: TStage @ [0, 0, 0]
  s1: TStage @ [0, 0, 2] rotate Z 180
  100: Port [0, 0, 0] role=input <q_in>
  200: XZX [0, 0, 2]
  101: Port [0, 0, 4] role=output <q_out>
  100 -> s0.q_in
  s0.q_out -> 200
  200 -> s1.q_in
  s1.q_out -> 101
}
";

#[test]
fn t_regions_keep_their_source_layers_through_modules_and_serialization() {
    let region_layers = |bloq: &Bloq| {
        let mut layers = bloq
            .top()
            .nodes()
            .filter(|(_, node)| node.try_region().is_some())
            .map(|(_, node)| node.layer())
            .collect::<Vec<_>>();
        layers.sort_unstable();
        layers
    };
    let graph = BlockGraph::from_blog_text(T_MODULES).expect("modules parse");
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&graph)
        .expect("modules compile")
        .bloq;
    // The second stage sits two source layers up, and source layers count double.
    assert_eq!(region_layers(&bloq), [0, 4]);
    assert_eq!(
        region_layers(&Bloq::from_binary(&bloq.to_binary()).unwrap()),
        [0, 4]
    );
    assert_eq!(
        region_layers(&Bloq::from_text(&bloq.to_text()).unwrap()),
        [0, 4]
    );
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
        let mut users = FxHashMap::<_, Vec<_>>::default();
        for (node_id, node) in level.nodes() {
            if node.try_quantum().is_some() || node.try_region().is_some() {
                for qubit in bloq.node_qubits(node).unwrap() {
                    users.entry(qubit).or_default().push(node_id);
                }
            }
        }
        let mut pairs = FxHashSet::default();
        for list in users.values() {
            for (index, &left) in list.iter().enumerate() {
                for &right in &list[index + 1..] {
                    if level[left].try_region().is_some() || level[right].try_region().is_some() {
                        pairs.insert((left.min(right), left.max(right)));
                    }
                }
            }
        }
        for (left, right) in pairs {
            assert!(
                level.has_path(left, right) || level.has_path(right, left),
                "{id}: {left:?} and {right:?} share a qubit unordered"
            );
        }
    }
}
