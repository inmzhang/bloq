//! Joint T outlets retain logical and physical channels after spatial doglegs.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{
    Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe,
    verify::{BoundaryOrder, LogicalVerifier},
};
use glam::{IVec3, ivec3};
use std::collections::{BTreeMap, BTreeSet};

mod common;
use common::choi::exact::ExactChoi;
use common::parse_qasm;

fn compile_source_channel(
    source: &str,
    qasm: &str,
    inputs: &[IVec3],
    outputs: &[IVec3],
) -> bloq_ir::Bloq {
    let graph = BlockGraph::from_blog_text(source).expect("generated outlet fixture parses");
    let expected = parse_qasm(qasm)
        .expect("independent source QASM parses")
        .to_graph();
    let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
        &graph,
        BoundaryOrder::new(inputs.to_vec(), outputs.to_vec()),
    )
    .expect("outlet interface admits a logical verifier");
    let report = verifier
        .verify(&expected, 32, 7)
        .expect("outlets preserve the source channel");
    assert_eq!(report.verified, 32);
    CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile_and_validate(&graph)
        .expect("outlets compile and validate at distance three")
        .bloq
}

#[rstest::rstest]
#[case(false, include_str!("data/interop_t_outlets_0.blog"))]
#[case(true, include_str!("data/interop_t_outlets_1.blog"))]
fn neighboring_t_resources_have_the_signed_channel(#[case] dagger: bool, #[case] source: &str) {
    let inputs = [ivec3(0, 0, 0)];
    let outputs = [ivec3(0, 0, 4)];
    let qasm = if dagger {
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; t q[0]; tdg q[0];"
    } else {
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; t q[0]; t q[0];"
    };
    let program = compile_source_channel(source, qasm, &inputs, &outputs);
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        if !dagger {
            sim.s(0);
        }
        Ok(())
    });
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    for seed in [0x11, 0x22, 0x33] {
        let report = oracle.run(&program, 8, seed, |_, context| {
            for &(name, value) in context.named_branch_selectors {
                selectors.entry(name.to_owned()).or_default().insert(value);
            }
            Ok(())
        });
        assert!(report.all_detectors_constant());
    }
    assert_eq!(selectors.len(), 2);
    assert!(
        selectors
            .values()
            .all(|values| *values == BTreeSet::from([false, true]))
    );
}

#[rstest::rstest]
#[case::dogleg(false)]
#[case::winding(true)]
fn time_z_spatial_paths_preserve_the_entangled_signed_channel(#[case] winding: bool) {
    let mut graph = BlockGraph::from_blog_text(include_str!("data/interop_t_outlets_time_z.blog"))
        .expect("generated outlet fixture parses");
    if winding {
        let corners = [
            ivec3(1, -2, 2),
            ivec3(0, -2, 2),
            ivec3(0, -4, 2),
            ivec3(4, -4, 2),
            ivec3(4, -2, 2),
            ivec3(2, -2, 2),
        ];
        // Four XY bends replace one identity seam without changing the final
        // signed injection seam, canonical elbow, selector, or feedback.
        let [start, .., end] = corners;
        graph.remove_pipe(start, end).expect("arm seam exists");
        for pair in corners.windows(2) {
            let step = (pair[1] - pair[0]).signum();
            let direction = Direction::try_from(step).expect("arm stays axial");
            let mut previous = pair[0];
            while previous != pair[1] {
                let next = previous + step;
                if next != end {
                    graph.add_block(Block::new(next, BlockKind::Cube(CubeKind::XXZ)));
                }
                graph.add_pipe(Pipe::new(previous, direction));
                previous = next;
            }
        }
    }
    let inputs = [ivec3(0, 0, 0), ivec3(2, 0, 0)];
    let outputs = [ivec3(2, 0, 5), ivec3(0, 0, 5)];
    let program = compile_source_channel(
        &graph
            .to_program_blog_text()
            .expect("outlet graph serializes"),
        include_str!("data/topols/entangled_tdg.qasm"),
        &inputs,
        &outputs,
    );
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        sim.h(0);
        sim.cx(0, 1)?;
        sim.t_dag(1)?;
        sim.cx(1, 0)?;
        Ok(())
    });
    let mut selectors = BTreeSet::new();
    for seed in [0x11, 0x22, 0x33] {
        let report = oracle.run(&program, 8, seed, |_, context| {
            let [(_, value)] = context.named_branch_selectors else {
                panic!("one T resource declares one selector");
            };
            selectors.insert(*value);
            Ok(())
        });
        assert!(report.all_detectors_constant());
        assert!(report.max_rank > 1);
    }
    assert_eq!(selectors, BTreeSet::from([false, true]));
}

#[test]
fn temporal_t_outlet_identity_elbow_preserves_the_full_signed_channel() {
    let inputs = [ivec3(0, -1, 0)];
    let outputs = [ivec3(0, 1, 2)];
    let program = compile_source_channel(
        include_str!("data/interop_temporal_t_elbow.blog"),
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; tdg q[0];",
        &inputs,
        &outputs,
    );
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        sim.t_dag(0)?;
        Ok(())
    });
    let mut selectors = BTreeSet::new();
    for seed in [0x11, 0x22, 0x33] {
        let report = oracle.run(&program, 8, seed, |_, context| {
            let [(_, value)] = context.named_branch_selectors else {
                panic!("one T resource declares one selector");
            };
            selectors.insert(*value);
            Ok(())
        });
        assert!(report.all_detectors_constant());
        assert!(report.max_rank > 1);
    }
    assert_eq!(selectors, BTreeSet::from([false, true]));
}
