//! Authored and hierarchical sources use the native Hadamard rewrite.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::InstanceProvenance;

const BODY: &str = "\
0: Port [0,0,0] role=input\n\
1: XZX [0,0,1]\n\
2: ZXZ [1,0,1]\n\
3: XZX [1,1,1]\n\
4: ZXZ [2,1,1]\n\
5: Port [2,1,2] role=output\n\
0 -> +Z\n1 -H> +X\n2 -H> +Y\n3 -H> +X\n4 -> +Z\n";

#[test]
fn flat_and_child_sources_compile_without_mixed_wall_axes() {
    for hierarchical in [false, true] {
        let text = if hierarchical {
            format!(
                "BLOG 1.0\nmodule Leaf {{\nin i: data = 0\nout o: data = 5\n{BODY}}}\n\
                 module main {{\nin i: data = 0\nout o: data = 1\n\
                 child: Leaf @ [0,0,0]\n\
                 0: Port [0,0,0] role=input\n1: Port [2,1,2] role=output\n\
                 0 -> child.i\nchild.o -> 1\n}}\n"
            )
        } else {
            format!("BLOG 1.0\n{BODY}")
        };
        let source = BlockGraph::from_blog_text(&text).unwrap();
        let before = source.to_blog_text();
        let context = CompileContext::new(CompileConfig::new(3));
        let artifacts = context.compile_and_validate(&source).unwrap();
        // Original geometry has walls on X and Y in one spatial component.
        // The native compiler only needs an X wall after color changes.
        let mut walls = Vec::new();
        for (_, node) in artifacts.bloq.quantum_nodes() {
            for instance in &node.instances {
                if let InstanceProvenance::Pipe { src, dst } = instance.provenance
                    && src.z == dst.z
                {
                    walls.push((src, dst));
                }
            }
        }
        assert!(!walls.is_empty());
        assert!(walls.iter().all(|(src, dst)| src.y == dst.y));
        assert_eq!(source.to_blog_text(), before);
    }
}
