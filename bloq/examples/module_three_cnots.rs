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
    Ok(())
}
