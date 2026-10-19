// [build-start]
use bloq::prelude::*;

fn build_graph() -> Result<BlockGraph> {
    let mut cnot = GalleryItem::CNOT.build();
    cnot.name = "CNOT".into();
    let mut graph = BlockGraph::from_definitions(vec![cnot, BlockGraph::new()])?;
    for index in 0..3 {
        graph.place_module(&format!("child{index}"), "CNOT")?;
    }
    graph.connect_modules("child0.control_out", "child1.control_in")?;
    graph.connect_modules("child1.control_out", "child2.control_in")?;
    Ok(graph)
}
// [build-end]

fn main() -> Result {
    let graph = build_graph()?;
    graph.to_file("module-three-cnots.blog")?;
    compile(&graph, 3)?.validate()?;
    #[cfg(feature = "graph-verify")]
    {
        use bloq::graph::verify::{BoundaryOrder, LogicalVerifier};
        use quizx::circuit::Circuit;

        let mut expected = Circuit::new(2);
        for _ in 0..3 {
            expected.add_gate("cx", vec![0, 1]);
        }
        let boundaries = BoundaryOrder::new(
            vec![IVec3::new(0, 0, 0), IVec3::new(1, 1, 0)],
            vec![IVec3::new(0, 0, 7), IVec3::new(1, 1, 7)],
        );
        LogicalVerifier::with_boundaries(&graph.flatten()?, boundaries)?.verify(
            &expected.to_graph(),
            8,
            0x5eed,
        )?;
    }
    Ok(())
}
