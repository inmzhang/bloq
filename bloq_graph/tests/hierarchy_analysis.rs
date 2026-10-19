use bloq_graph::{
    BlockGraph, BlockGraphError, GalleryItem, ModuleInstance, ModuleRotation, UDirection, ZXGraph,
};
use glam::IVec3;

fn hierarchy() -> BlockGraph {
    BlockGraph::from_text(
        "BLOG 1.0\nmodule Leaf {\n0: T [0,0,0]\n1: XZ [2,0,0]\n}\n\
         module main {\nfirst: Leaf @ [0,0,0]\nsecond: Leaf @ [4,0,0]\n}\n",
    )
    .expect("hierarchical fixture is valid")
}

fn requires_flat<T>(result: Result<T, BlockGraphError>) {
    assert!(matches!(
        result,
        Err(BlockGraphError::HierarchyRequiresFlatten { .. })
    ));
}

#[test]
fn scientific_queries_cannot_ignore_child_geometry() {
    let graph = hierarchy();
    requires_flat(graph.to_zx_graph());
    requires_flat(graph.clone().analyze_actions());
    requires_flat(graph.fill_ports_auto());
    requires_flat(graph.branch_projections());
    requires_flat(graph.project_branches([]));
    requires_flat(graph.randomly_resolve_selectives(1));
    assert!(
        ZXGraph::try_from(&graph)
            .unwrap_err()
            .to_string()
            .contains("flatten")
    );
    #[cfg(feature = "verify")]
    assert!(
        bloq_graph::verify::LogicalVerifier::new(&graph)
            .unwrap_err()
            .to_string()
            .contains("flatten")
    );
}

#[test]
fn flat_transforms_preserve_authored_sources_by_refusing_them() {
    for graph in [hierarchy(), GalleryItem::CNOT.build()] {
        let source = graph.to_blog_text();
        requires_flat(graph.shift_positions(IVec3::Z));
        requires_flat(graph.rotate_about_origin_lenient(UDirection::Z, 1));
        requires_flat(graph.with_zero_min_z());
        requires_flat(graph.flip_xz_basis_lenient());
        requires_flat(graph.fill_ports_auto());
        requires_flat(graph.randomly_resolve_selectives(1));
        assert_eq!(graph.to_blog_text(), source);
        let flat = graph.flatten().unwrap();
        assert!(!flat.has_module_structure());
        assert!(
            !flat
                .shift_positions(IVec3::Z)
                .unwrap()
                .has_module_structure()
        );
    }
    // An interface by itself does not prevent leaf scientific analysis.
    GalleryItem::CNOT.build().stabilizers().unwrap();
}

#[test]
fn classifications_include_reachable_definitions_and_handle_invalid_references() {
    let mut graph = hierarchy();
    assert_eq!(graph.block_count(), 0);
    assert_eq!(graph.t_count(), 0);
    assert!(!graph.is_empty());
    assert!(!graph.is_clifford());
    assert!(!graph.is_rigid());
    assert!(!graph.is_open());

    graph.instances[0].definition = "Missing".into();
    assert!(!graph.is_empty());
    assert!(!graph.is_clifford());
    assert!(!graph.is_rigid());

    let mut cycle = BlockGraph::new();
    cycle.instances.push(ModuleInstance {
        name: "again".into(),
        definition: "main".into(),
        rotation: ModuleRotation::IDENTITY,
        translation: IVec3::ZERO,
    });
    assert!(cycle.is_empty());
    assert!(cycle.is_clifford());
    assert!(cycle.is_rigid());
}

#[test]
fn authored_validation_keeps_local_action_constraints() {
    let mut leaf = GalleryItem::T.build();
    leaf.clear_actions();
    assert!(matches!(
        leaf.validate_source(),
        Err(BlockGraphError::InvalidAction(_))
    ));
    assert!(matches!(
        leaf.validate(),
        Err(BlockGraphError::InvalidAction(_))
    ));

    let child_source = GalleryItem::T
        .entry()
        .blog()
        .replace("module main", "module Child");
    let root = "\nmodule main {\n}\n";
    let mut graph = BlockGraph::from_text(&(child_source + root)).unwrap();
    graph.module_mut("Child").unwrap().clear_actions();
    assert!(matches!(
        graph.validate_source(),
        Err(BlockGraphError::InvalidAction(_))
    ));

    let parsed = GalleryItem::CNOT.build();
    parsed.validate_source().unwrap();
    parsed.validate().unwrap();
}
