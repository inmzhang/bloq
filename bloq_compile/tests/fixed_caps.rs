//! Fixed caps on geometry with reused data patches.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;

// Unchanged geometry cut from Topologiq's adder: CX(15,14) followed by T(17),
// upstream d010bd3e63686c93c12c922a84a944a3386ac785.
#[test]
fn fixed_x_cap_on_reused_patch_geometry_compiles() {
    let source = include_str!("data/late_control_t.blog")
        .replace("4: T [", "4: XZX [")
        .replace("5: XY [", "5: X [")
        .replace("  m = measure 0\n", "")
        .replace("  resolve 5 if !m\n", "")
        .replace("  feedback X 2 -> -X if m\n", "");
    let graph = BlockGraph::from_blog_text(&source).unwrap();
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&graph)
        .unwrap()
        .bloq;
    bloq.validate().unwrap();
}
