use std::error::Error;

use bloq_graph::{
    BlockGraph, BlockGraphError, GalleryItem, ModuleCertificationError, ModuleCertificationLimits,
    ModuleError,
};

fn has_resource_limit(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if matches!(
            error.downcast_ref::<ModuleCertificationError>(),
            Some(ModuleCertificationError::ResourceLimited { .. })
        ) || matches!(
            error.downcast_ref::<bloq_graph::StabilizerError>(),
            Some(bloq_graph::StabilizerError::ResourceLimited { .. })
        ) {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

#[test]
fn graph_constructors_and_from_str_preserve_the_same_source_hierarchy() {
    for item in [GalleryItem::T, GalleryItem::CNOT, GalleryItem::YMemory] {
        let expected = item.build();
        let executable = item.entry().blog();
        let parsed = BlockGraph::from_text(executable).unwrap();
        assert_eq!(parsed.to_blog_text(), expected.to_blog_text());
        let body = expected.to_blog_body_text();
        assert_eq!(
            BlockGraph::from_text(&body).unwrap().to_blog_body_text(),
            body
        );
        assert_eq!(
            body.parse::<BlockGraph>().unwrap().to_blog_body_text(),
            body
        );
        assert_eq!(
            BlockGraph::from_blog_text(executable)
                .unwrap()
                .to_blog_text(),
            expected.to_blog_text()
        );
        assert_eq!(
            executable.parse::<BlockGraph>().unwrap().to_blog_text(),
            expected.to_blog_text()
        );
        assert!(parsed.has_module_structure());
    }
    assert!(BlockGraph::from_text("BLOG 1.0\n").unwrap().is_empty());
}

#[test]
fn module_errors_and_expansion_budgets_never_fall_back_to_body_lowering() {
    let duplicate = "BLOG 1.0\nmodule main {\n}\nmodule main {\n}\n";
    assert!(matches!(
        BlockGraph::from_text(duplicate),
        Err(BlockGraphError::ModuleSource(error))
            if matches!(error.as_ref(), ModuleError::DuplicateModule(name) if name == "main")
    ));
    let limits = ModuleCertificationLimits {
        max_expanded_blocks: 0,
        ..ModuleCertificationLimits::DEFAULT
    };
    for source in [
        "BLOG 1.0\n0: ZXZ [0,0,0]\n",
        "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n}\n",
    ] {
        let error = BlockGraph::from_text_with_limits(source, limits).unwrap_err();
        assert!(has_resource_limit(&error), "{error:?}");
        assert_eq!(BlockGraph::from_text(source).unwrap().block_count(), 1);
    }
}

#[test]
fn file_loading_resolves_nested_imports_and_preserves_the_original_io_failure() {
    let directory =
        std::env::temp_dir().join(format!("bloq-graph-input-{}", rand::random::<u64>()));
    std::fs::create_dir_all(directory.join("children")).unwrap();
    let directory = directory.canonicalize().unwrap();
    let child_directory = directory.join("children");
    let root = directory.join("main.blog");
    let child = child_directory.join("leaf.blog");
    let grandchild = child_directory.join("body.blog");
    std::fs::write(
        &root,
        "BLOG 1.0\nimport \"children/leaf.blog\" as Child\nmodule main {\nfirst: Child @ [2,0,0]\n}\n",
    )
    .unwrap();
    std::fs::write(
        &child,
        "BLOG 1.0\nimport \"body.blog\" as Body\nmodule main {\nfirst: Body @ [0,0,0]\n}\n",
    )
    .unwrap();
    std::fs::write(&grandchild, "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n}\n").unwrap();
    let graph = BlockGraph::load(&root).unwrap();
    assert_eq!(graph.block_count(), 0);
    assert_eq!(graph.modules().len(), 3);
    assert!(graph.module("Child").is_some());
    assert!(graph.module("Child__Body").is_some());
    let projected = graph.flatten().unwrap();
    assert!(!projected.has_module_structure());
    assert_eq!(projected.block_count(), 1);
    assert!(projected.get_block([2, 0, 0]).is_some());
    let restored = BlockGraph::from_text(&graph.to_blog_text()).unwrap();
    assert_eq!(restored.to_blog_text(), graph.to_blog_text());
    assert_eq!(
        restored.flatten().unwrap().to_blog_body_text(),
        projected.to_blog_body_text()
    );
    let refused = BlockGraph::load_with_limits(
        &root,
        ModuleCertificationLimits {
            max_expanded_blocks: 0,
            ..ModuleCertificationLimits::DEFAULT
        },
    )
    .unwrap_err();
    assert!(has_resource_limit(&refused), "{refused:?}");
    assert!(matches!(
        refused,
        BlockGraphError::ModuleSource(error)
            if matches!(error.as_ref(), ModuleError::Parse(bloq_graph::ParseError::Graph { .. }))
    ));

    std::fs::write(
        &grandchild,
        "BLOG 1.0\nimport \"../main.blog\" as Root\nmodule main {\n}\n",
    )
    .unwrap();
    assert!(matches!(
        BlockGraph::load(&root),
        Err(BlockGraphError::ModuleSource(error))
            if matches!(error.as_ref(), ModuleError::ImportCycle { path } if path == &root.display().to_string())
    ));

    let invalid = "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n0: ZXZ [2,0,0]\n}\n";
    std::fs::write(&grandchild, invalid).unwrap();
    assert!(matches!(
        BlockGraph::load(&root),
        Err(BlockGraphError::ModuleSource(error))
            if matches!(error.as_ref(), ModuleError::Parse(bloq_graph::ParseError::DuplicateId { .. }))
    ));
    assert!(matches!(
        BlockGraph::load(&grandchild),
        Err(BlockGraphError::Parse(
            bloq_graph::ParseError::DuplicateId { .. }
        ))
    ));

    std::fs::remove_file(&grandchild).unwrap();
    match BlockGraph::load(&root).unwrap_err() {
        BlockGraphError::Io { path, source } => {
            assert_eq!(path, grandchild);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        error => panic!("expected original import I/O error, got {error:?}"),
    }
    std::fs::write(&root, graph.to_blog_text()).unwrap();
    assert_eq!(
        BlockGraph::load(&root).unwrap().to_blog_text(),
        graph.to_blog_text()
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn the_executable_graph_owns_its_root_topology_and_serializes_local_edits() {
    let mut graph = BlockGraph::from_text(
        "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\nmodule main {\nfirst: Leaf @ [2,0,0]\n0: ZXZ [0,0,0]\n}\n",
    )
    .unwrap();
    assert!(std::ptr::eq(&graph, graph.root()));
    assert!(std::ptr::eq(&graph, graph.module("main").unwrap()));
    graph
        .try_add_block(bloq_graph::Block::new(
            [4, 0, 0],
            bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ))
        .unwrap();
    let restored = BlockGraph::from_text(&graph.to_blog_text()).unwrap();
    assert_eq!(restored.block_count(), 2);
    assert_eq!(restored.instances.len(), 1);
    assert_eq!(restored.module("Leaf").unwrap().block_count(), 1);
    assert_eq!(restored.flatten().unwrap().block_count(), 3);
    assert!(!restored.flatten().unwrap().has_module_structure());
}

#[test]
fn extraction_preserves_dependencies_and_resolves_a_reachable_main_name_collision() {
    let graph = BlockGraph::from_text(
        "BLOG 1.0\nmodule source_main {\n0: ZXZ [0,0,0]\n}\nmodule Selected {\nfirst: main @ [0,0,0]\nsecond: source_main @ [2,0,0]\n}\nmodule main {\n0: ZXZ [0,0,0]\n}\n",
    )
    .unwrap();
    let extracted = graph.extract_definition("Selected").unwrap();
    assert_eq!(extracted.name, "main");
    assert_eq!(extracted.instances[0].definition, "source_main_1");
    assert!(extracted.module("source_main").is_some());
    assert!(extracted.module("source_main_1").is_some());
    assert!(extracted.module("Selected").is_none());
    assert_eq!(extracted.flatten().unwrap().block_count(), 2);
    assert_eq!(
        BlockGraph::from_text(&extracted.to_blog_text())
            .unwrap()
            .to_blog_text(),
        extracted.to_blog_text()
    );
    assert_eq!(
        graph.extract_definition("Selected").unwrap().to_blog_text(),
        extracted.to_blog_text()
    );
    assert_eq!(graph.name, "main");
    assert!(graph.module("Selected").is_some());
}

#[test]
fn public_certificate_validation_rejects_mutated_ports_and_nested_libraries() {
    let mut graph = GalleryItem::CNOT.build();
    graph.interface.quantum_ports[0].position = [123, 0, 0].into();
    let error = graph
        .certify_leaf("main", ModuleCertificationLimits::DEFAULT)
        .unwrap_err();
    assert!(error.to_string().contains("port"));

    let nested = BlockGraph::from_text(
        "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\nmodule main {\nfirst: Leaf @ [0,0,0]\n}\n",
    )
    .unwrap();
    let error = BlockGraph::from_definitions(vec![nested]).unwrap_err();
    assert!(error.to_string().contains("nested helper libraries"));
}
