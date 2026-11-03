//! T regions keep their source layers and causal order across shared qubits.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::{Bloq, SubGraph};
use bloq_test::{CompileReadyCase, compile_ready_test_cases};
use rustc_hash::{FxHashMap, FxHashSet};

/// `stages` T gates in series, alternating sides so each side reuses a footprint.
fn t_chain(stages: usize) -> BlockGraph {
    let stage = "module TStage {
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
";
    let instances = (0..stages)
        .map(|i| {
            let turn = if i % 2 == 1 { " rotate Z 180" } else { "" };
            format!("  s{i}: TStage @ [0, 0, {}]{turn}\n", 2 * i)
        })
        .collect::<String>();
    let wires = (0..stages - 1)
        .map(|i| format!("  {}: XZX [0, 0, {}]\n", 200 + i, 2 * i + 2))
        .collect::<String>();
    let links = (0..stages - 1)
        .map(|i| {
            format!(
                "  s{i}.q_out -> {wire}\n  {wire} -> s{next}.q_in\n",
                wire = 200 + i,
                next = i + 1
            )
        })
        .collect::<String>();
    let source = format!(
        "BLOG 1.0\n\n{stage}\nmodule main {{\n  in q_in: data = 100\n  out q_out: data = 101\n\
         {instances}  100: Port [0, 0, 0] role=input <q_in>\n{wires}\
         \x20 101: Port [0, 0, {top}] role=output <q_out>\n  100 -> s0.q_in\n{links}\
         \x20 s{last}.q_out -> 101\n}}\n",
        top = 2 * stages,
        last = stages - 1,
    );
    BlockGraph::from_blog_text(&source).expect("T chain parses")
}

fn region_layers(bloq: &Bloq) -> Vec<i64> {
    let mut layers = bloq
        .top()
        .nodes()
        .filter(|(_, node)| node.try_region().is_some())
        .map(|(_, node)| node.layer())
        .collect::<Vec<_>>();
    layers.sort_unstable();
    layers
}

/// Pairs of regions and quantum nodes sharing a qubit with no path either way.
fn unordered_region_overlaps(bloq: &Bloq, level: &SubGraph) -> Vec<(u32, u32)> {
    let mut users = FxHashMap::<_, Vec<_>>::default();
    for (id, node) in level.nodes() {
        if node.try_quantum().is_some() || node.try_region().is_some() {
            for qubit in bloq.node_qubits(node).expect("corpus coordinates fit") {
                users.entry(qubit).or_default().push(id);
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
    let mut unordered = pairs
        .into_iter()
        .filter(|&(left, right)| !level.has_path(left, right) && !level.has_path(right, left))
        .map(|(left, right)| (left.0, right.0))
        .collect::<Vec<_>>();
    unordered.sort_unstable();
    unordered
}

#[test]
fn t_regions_keep_their_source_layers_through_modules_and_serialization() {
    const STAGES: usize = 8;
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&t_chain(STAGES))
        .expect("T chain compiles")
        .bloq;
    // Stages sit two source layers apart, and source layers count double.
    let expected = (0..STAGES as i64).map(|i| 4 * i).collect::<Vec<_>>();
    assert_eq!(region_layers(&bloq), expected);
    assert_eq!(
        region_layers(&Bloq::from_binary(&bloq.to_binary()).expect("binary IR round-trips")),
        expected
    );
    assert_eq!(
        region_layers(&Bloq::from_text(&bloq.to_text()).expect("text IR round-trips")),
        expected
    );
}

#[test]
fn t_chain_orders_every_region_overlap() {
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&t_chain(16))
        .expect("T chain compiles")
        .bloq;
    assert_eq!(unordered_region_overlaps(&bloq, bloq.top()), []);
}

#[test]
fn t_chain_order_edges_grow_linearly() {
    let order_edges = |stages| {
        CompileContext::new(CompileConfig::new(3))
            .compile(&t_chain(stages))
            .expect("T chain compiles")
            .bloq
            .edges()
            .filter(|edge| matches!(edge.edge, bloq_ir::BloqEdge::Order))
            .count()
    };
    let (small, medium, large) = (order_edges(16), order_edges(32), order_edges(64));
    assert_eq!(large - medium, 2 * (medium - small));
}

/// The occupancy pass, not layout validation, orders regions on different layers.
#[test]
fn corpus_orders_every_region_overlap() {
    let context = CompileContext::new(CompileConfig::new(3));
    for case in compile_ready_test_cases(None).expect("build compilation corpus") {
        let id = case.id().to_string();
        let ready = CompileReadyCase::from_test_case(case).expect("corpus case builds");
        let bloq = context.compile(&ready.graph).expect("corpus compiles").bloq;
        assert_eq!(unordered_region_overlaps(&bloq, bloq.top()), [], "{id}");
    }
}
