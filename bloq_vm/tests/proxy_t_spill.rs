//! Ideal T preparation needs only its normal surface-code patch.

use bloq_compile::{CompileConfig, CompileContext, CompileError};
use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind};
use bloq_vm::{run_bloq_with_hook, verify::logical_bloch};
use glam::IVec3;
use std::f64::consts::FRAC_1_SQRT_2;

#[test]
fn boxed_t_source_runs_without_cultivation_spill() {
    let mut graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n0: T [0,0,0]\n1: ZXZ [0,0,1]\n2: Port [0,0,2] role=output\n0 -> +Z\n1 -> +Z\n",
    )
    .unwrap();
    for offset in [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y] {
        graph.add_block(Block::new(offset, BlockKind::Cube(CubeKind::ZXZ)));
    }
    let proxy = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true));
    let cultivation = CompileContext::new(CompileConfig::new(3));
    let module = graph.clone().with_inferred_interface().unwrap();
    for source in [&graph, &module] {
        assert!(matches!(
            cultivation.compile(source),
            Err(CompileError::TBlockNeedsFreeNeighbor { pos }) if pos == IVec3::ZERO
        ));
        let bloq = proxy.compile_and_validate(source).unwrap().bloq;
        assert_eq!(bloq.logical_outputs().len(), 1);
        let mut scored = 0;
        let report = run_bloq_with_hook(&bloq, 8, 17, |sim, context| {
            let frame = &context.frames[0];
            let actual = logical_bloch(
                sim,
                &context.outputs[0],
                (frame.x.unwrap(), frame.z.unwrap()),
            )?;
            for (actual, expected) in
                [actual.0, actual.1, actual.2]
                    .into_iter()
                    .zip([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0])
            {
                assert!((actual - expected).abs() < 1e-9);
            }
            scored += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(scored, 8);
        assert_eq!(report.discarded, 0);
        assert!(report.all_detectors_constant());
        assert!(report.max_rank > 1);
    }
}
