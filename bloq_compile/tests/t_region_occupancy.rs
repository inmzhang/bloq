//! T regions keep their source layers.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::Bloq;

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
