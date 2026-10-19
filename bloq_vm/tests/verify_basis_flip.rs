#![cfg(test)]

//! Small physical oracles for resource and signed selective basis duality.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{Action, BlockGraph, GalleryItem};
use bloq_vm::run_bloq;

mod common;

use common::choi::exact::ExactChoi;

const SHOTS: usize = 8;
const SEED: u64 = 0xC0FFEE;

#[test]
fn fixed_resource_basis_flip_has_the_hadamard_choi_state_d3() {
    for kind in ["T", "Y"] {
        for boundary in ["ZXZ", "XZX"] {
            let source = BlockGraph::from_blog_text(&format!(
                "BLOG 1.0\n0: {kind} [0,0,0]\n1: {boundary} [0,0,1]\n2: Port [0,0,2]\n0 -> +Z\n1 -> +Z\n"
            )).unwrap();
            for flipped in [false, true] {
                let graph = if flipped {
                    source.flip_xz_basis().unwrap()
                } else {
                    source.clone()
                };
                let oracle = ExactChoi::new(&[], &[glam::ivec3(0, 0, 2)], |sim| {
                    sim.h(0);
                    if kind == "T" {
                        sim.t(0)?;
                    } else {
                        sim.s(0);
                    }
                    if flipped {
                        sim.h(0);
                    }
                    Ok(())
                });
                let compiler =
                    CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true));
                let module = graph.clone().with_inferred_interface().unwrap();
                for artifacts in [compiler.compile(&graph), compiler.compile(&module)] {
                    let bloq = artifacts.unwrap().bloq;
                    let report = oracle.run(&bloq, SHOTS, SEED, |_, _| Ok(()));
                    assert_eq!(report.discarded, 0);
                    assert!(report.all_detectors_constant());
                    if kind == "T" {
                        assert!(report.max_rank > 1);
                    }
                }
            }
        }
    }
}

#[test]
fn y_readout_matches_hadamard_conjugation_for_both_boundary_colors() {
    for boundary in ["ZXZ", "XZX"] {
        for hadamard in [false, true] {
            let mut graph = BlockGraph::from_blog_text(&format!(
                "BLOG 1.0\n0: Port [0,0,0]\n1: {boundary} [0,0,1]\n2: Y [0,0,2]\n0 -> +Z\n1 -> +Z\nm = measure 2\n"
            )).unwrap();
            graph
                .set_pipe_hadamard(glam::IVec3::Z, glam::IVec3::new(0, 0, 2), hadamard)
                .unwrap();
            let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
                .compile(&graph)
                .unwrap()
                .bloq;
            let report = bloq_vm::run_bloq_with_io(
                &bloq,
                SHOTS,
                SEED,
                |sim, context| {
                    sim.s(context.inputs[0].qubit);
                    Ok(())
                },
                |_, _| Ok(()),
            )
            .unwrap();
            assert!(
                report.observables[0]
                    .per_shot
                    .iter()
                    .all(|&value| value == hadamard),
                "{boundary} H={hadamard}"
            );
        }
    }
}

#[test]
fn native_t_basis_flip_preserves_authored_postselection_d3() {
    let source = GalleryItem::TComparison.build().flatten().unwrap();
    // The native case also covers the longer noiseless-postselection batch.
    for (graph, shots) in [
        (source.clone(), 32),
        (source.flip_xz_basis().unwrap(), SHOTS),
    ] {
        assert_eq!(graph.t_count(), 2);
        assert!(
            graph
                .actions()
                .iter()
                .any(|action| matches!(action, Action::DiscardIf(_)))
        );
        let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .compile(&graph)
            .unwrap()
            .bloq;
        let report = run_bloq(&bloq, shots, SEED).unwrap();
        assert_eq!(
            report.discarded, 0,
            "authored noiseless postselection accepts every shot"
        );
        assert!(report.max_rank > 1);
        assert!(report.all_detectors_constant());
        assert!(report.branch_selectors.contains(&true));
        assert!(report.branch_selectors.contains(&false));
    }
}

#[test]
fn basis_flip_preserves_a_branch_with_an_arm_owned_y_hadamard_d3() {
    use bloq_graph::{Block, BlockKind, BranchArm, CubeKind, Direction, Expr, MeasureTarget, Pipe};
    use glam::ivec3;
    let mut source = BlockGraph::new();
    for x in [0, 3] {
        source.add_block(Block::new(ivec3(x, 0, 0), BlockKind::Port));
        source.add_block(Block::new(ivec3(x, 0, 1), BlockKind::Cube(CubeKind::ZXZ)));
        source.add_pipe(Pipe::new(ivec3(x, 0, 0), Direction::ZPLUS));
    }
    let pipe = Pipe::new(ivec3(0, 0, 1), Direction::ZPLUS);
    let target = source
        .try_add_branch_region(
            "choice",
            BranchArm::new(
                vec![Block::new(ivec3(0, 0, 2), BlockKind::Y)],
                vec![pipe.clone()],
            ),
            BranchArm::new(
                vec![Block::new(ivec3(0, 0, 2), BlockKind::Cube(CubeKind::ZXZ))],
                vec![pipe],
            ),
        )
        .unwrap();
    source
        .set_actions(vec![
            Action::Measure {
                name: "m".into(),
                target: MeasureTarget::Node(ivec3(3, 0, 1)),
            },
            Action::Branch {
                target,
                condition: Expr::Var("m".into()),
            },
        ])
        .unwrap();
    for graph in [source.clone(), source.flip_xz_basis().unwrap()] {
        let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .compile(&graph)
            .unwrap()
            .bloq;
        let report = bloq_vm::run_bloq_with_io(
            &bloq,
            SHOTS,
            SEED,
            |sim, context| {
                let controller = context
                    .inputs
                    .iter()
                    .find(|input| input.port == ivec3(3, 0, 0))
                    .unwrap();
                sim.s(controller.qubit);
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(report.discarded, 0);
        assert!(report.all_detectors_constant());
        assert!(report.branch_selectors.contains(&true));
        assert!(report.branch_selectors.contains(&false));
    }
}
