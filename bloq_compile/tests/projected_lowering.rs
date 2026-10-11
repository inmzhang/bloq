use bloq_graph::{
    Action, Basis, Block, BlockKind, BranchArm, CubeKind, Direction, Expr, GalleryItem,
    MeasureTarget, Pipe, SelectiveKind, WalkingBoundaryKind, WalkingKind,
};
use bloq_ir::{BloqEdge, ClassicalNode};

mod common;
use glam::ivec3;

#[test]
fn structural_projection_lowers_spatial_ports() {
    // This experiment replaces authored ports, so work on an explicit topology projection.
    let mut graph = GalleryItem::CZSpatialH.build().flatten().unwrap();
    let past = ivec3(0, 0, 2);
    let target = ivec3(0, 0, 3);
    let control = ivec3(20, 2, 0);
    graph
        .set_block_kind(past, BlockKind::Cube(CubeKind::XZX))
        .unwrap();
    graph.add_block(Block::new(control, BlockKind::Cube(CubeKind::ZXZ)));
    let arm = |kind| {
        BranchArm::new(
            vec![Block::new(target, kind)],
            vec![Pipe::new(past, Direction::ZPLUS)],
        )
    };
    let target = graph
        .try_add_branch_region(
            "b",
            arm(BlockKind::Measurement(Basis::Z)),
            arm(BlockKind::Cube(CubeKind::XZX)),
        )
        .unwrap();
    graph
        .set_actions(vec![
            Action::Measure {
                target: MeasureTarget::Node(control),
                name: "m".into(),
            },
            Action::Branch {
                target,
                condition: Expr::Var("m".into()),
            },
        ])
        .unwrap();

    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    assert!(bloq.has_conditional_membership());
    for pinned in common::pinned_memberships(&bloq) {
        assert!(pinned.node_by_block(target).is_some());
        assert!(
            pinned
                .quantum_nodes()
                .flat_map(|(_, node)| &node.instances)
                .any(|instance| instance.provenance.is_spatial_port_substitution())
        );
        bloq_stim::emit_bloq_stim(&pinned).unwrap();
    }
}

#[test]
fn structural_parallel_cut_preserves_both_hadamard_realignments() {
    let mut graph = bloq_graph::BlockGraph::new();
    let past = [ivec3(0, 0, 0), ivec3(1, 0, 0)];
    let future = [ivec3(0, 0, 1), ivec3(1, 0, 1)];
    let control = ivec3(10, 0, 0);
    for position in [past[0], past[1], control] {
        graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
    }
    graph.add_pipe(Pipe::new(past[0], Direction::XPLUS));
    let arm = || {
        BranchArm::new(
            future
                .into_iter()
                .map(|position| Block::new(position, BlockKind::Cube(CubeKind::XZX)))
                .collect(),
            vec![
                Pipe::new(past[0], Direction::ZPLUS).with_hadamard(),
                Pipe::new(past[1], Direction::ZPLUS).with_hadamard(),
                Pipe::new(future[0], Direction::XPLUS),
            ],
        )
    };
    let branch = graph.try_add_branch_region("b", arm(), arm()).unwrap();
    graph
        .set_actions(vec![
            Action::Measure {
                target: MeasureTarget::Node(control),
                name: "m".into(),
            },
            Action::Branch {
                target: branch,
                condition: Expr::Var("m".into()),
            },
        ])
        .unwrap();

    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    for pinned in common::pinned_memberships(&bloq) {
        let owner = pinned.node_by_block(future[0]).unwrap();
        assert_eq!(pinned.node_by_block(future[1]), Some(owner));
        let incoming = pinned
            .incoming(owner)
            .filter(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
            .collect::<Vec<_>>();
        assert_eq!(incoming.len(), 2);
        let past_owner = pinned.node_by_block(past[0]).unwrap();
        assert_eq!(pinned.node_by_block(past[1]), Some(past_owner));
        let mut pipes = Vec::new();
        for edge in incoming {
            assert!(pinned.has_path(past_owner, edge.source));
            assert!(pinned[edge.source].try_quantum().is_some());
            pipes.extend(edge.edge.pipes().iter().map(|seam| seam.pipe));
        }
        pipes.sort_by_key(|pipe| pipe.src.x);
        assert_eq!(pipes.len(), 2);
        for (index, pipe) in pipes.iter().enumerate() {
            assert_eq!(
                (pipe.src, pipe.dst, pipe.hadamard),
                (past[index], future[index], true)
            );
        }
    }
}

#[test]
fn structural_walking_arms_share_transit_rounds() {
    for movement in [glam::IVec2::X, glam::ivec2(-1, 1)] {
        let mut graph = bloq_graph::BlockGraph::new();
        let first = ivec3(0, 0, 1);
        let second = ivec3(1, 0, 1);
        for position in [ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(10, 0, 0)] {
            graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, movement).unwrap();
        let output = |position| walking.end_position(position) + glam::IVec3::Z;
        let arm = || {
            BranchArm::new(
                vec![
                    Block::new(first, BlockKind::Walking(walking)),
                    Block::new(second, BlockKind::Walking(walking)),
                    Block::new(output(first), BlockKind::Measurement(Basis::Z)),
                    Block::new(output(second), BlockKind::Measurement(Basis::Z)),
                ],
                vec![
                    Pipe::new(ivec3(0, 0, 0), Direction::ZPLUS),
                    Pipe::new(ivec3(1, 0, 0), Direction::ZPLUS),
                    Pipe::new(walking.end_position(first), Direction::ZPLUS),
                    Pipe::new(walking.end_position(second), Direction::ZPLUS),
                ],
            )
        };
        let branch = graph.try_add_branch_region("walk", arm(), arm()).unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(ivec3(10, 0, 0)),
                    name: "m".into(),
                },
                Action::Branch {
                    target: branch,
                    condition: Expr::Var("m".into()),
                },
            ])
            .unwrap();

        let bloq = bloq_compile::compile(&graph, 3).unwrap();
        for pinned in common::pinned_memberships(&bloq) {
            assert_eq!(pinned.node_by_block(first), pinned.node_by_block(second));
            bloq_stim::emit_bloq_stim(&pinned).unwrap();
        }
    }
}

#[test]
fn selective_y_arm_keeps_spatial_hadamard_records() {
    // This experiment replaces authored ports, so work on an explicit topology projection.
    let mut graph = GalleryItem::CZSpatialH.build().flatten().unwrap();
    let cube = ivec3(0, 0, 1);
    let selective = ivec3(0, 0, 2);
    let side = ivec3(1, 0, 1);
    graph
        .set_block_kind(cube, BlockKind::Cube(CubeKind::ZXZ))
        .unwrap();
    graph
        .set_block_kind(selective, BlockKind::Selective(SelectiveKind::XY))
        .unwrap();
    let control = ivec3(20, 2, 0);
    graph.add_block(Block::new(control, BlockKind::Cube(CubeKind::ZXZ)));
    graph
        .set_actions(vec![
            Action::Measure {
                target: MeasureTarget::Node(control),
                name: "m".into(),
            },
            Action::Resolve {
                target: selective,
                condition: Expr::Var("m".into()),
            },
        ])
        .unwrap();

    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    let wall_instance = bloq
        .quantum_nodes()
        .flat_map(|(_, quantum)| &quantum.instances)
        .find(|instance| {
            matches!(instance.provenance, bloq_ir::InstanceProvenance::Pipe { src, dst }
            if (src == cube && dst == side) || (src == side && dst == cube))
        })
        .expect("spatial Hadamard wall instance")
        .id;
    fn value(bloq: &bloq_ir::Bloq, id: bloq_ir::BloqNodeId, selector: bool) -> bool {
        match bloq[id].try_classical().unwrap() {
            ClassicalNode::Observable { .. } => selector,
            ClassicalNode::Compute { expr } => expr
                .eval(&mut |slot| {
                    let input = bloq.value_inputs(id).find(|input| input.slot == slot)?;
                    Some(value(bloq, input.producer, selector))
                })
                .unwrap(),
            other => panic!("unexpected condition producer {other:?}"),
        }
    }
    // Check active Y-arm records independently of generator ordinals and
    // affine producer layout. XY selects Y when the control is false.
    let selected = false;
    assert!(
        bloq.nodes().any(|(id, node)| {
            let reads_wall = |node: &bloq_ir::BloqNode| {
                matches!(node.try_classical(), Some(ClassicalNode::Observable { measurements, .. })
                if measurements.iter().any(|record| record.instance == wall_instance))
            };
            reads_wall(node)
                && node.activation.is_none_or(|slot| {
                    let input = bloq
                        .value_inputs(id)
                        .find(|input| input.slot == slot)
                        .unwrap();
                    value(&bloq, input.producer, selected)
                })
        }),
        "selected {selected}: spatial wall records remain in the readout"
    );
}
