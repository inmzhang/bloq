//! Measurement-frame absorption preserves named instruments and magic-state channels.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_1_SQRT_2;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{
    Action, BlockGraph, BlockKind, PauliBasis,
    verify::{LogicalVerifier, QuizxGraph},
};
use bloq_ir::NodeProvenance;
use bloq_vm::{ShotContext, run_bloq_with_io, verify::logical_bloch};
use glam::IVec3;
use quizx::graph::{BasisElem, GraphLike};

mod common;
use common::choi::stabilizer::physical_qubit;
use common::parse_qasm;

fn source(cap: &str, feedback: bool) -> BlockGraph {
    let resolve = if matches!(cap, "XY" | "XZ" | "YZ") {
        "resolve 2 if mzz\n"
    } else {
        ""
    };
    let feedback = if feedback {
        "feedback X 2 if mzz\n"
    } else {
        ""
    };
    BlockGraph::from_blog_text(&format!(
        "BLOG 1.0\n0: Port [0,0,0] role=input\n1: XZX [0,0,1]\n2: {cap} [-1,0,2]\n3: T [1,0,0]\n4: XZX [1,0,1]\n5: YX [1,0,2]\n6: ZXZ [-1,0,1]\n0 -> +Z\n1 -H> -X\n6 -> +Z\n1 -> +X\n3 -> +Z\n5 -> -Z\nmzz = measure 1 -> +X\nresolve 5 if mzz\nm = measure 2\n{resolve}{feedback}"
    )).expect("signed T instrument parses")
}

fn measured_axis(cap: &str, control: bool) -> PauliBasis {
    let kind: BlockKind = cap.parse().expect("measurement cap kind");
    match kind {
        BlockKind::Measurement(basis) => basis.into(),
        BlockKind::Y => PauliBasis::Y,
        BlockKind::Selective(kind) => {
            if control {
                kind.pauli_if_true()
            } else {
                kind.pauli_if_false()
            }
        }
        _ => unreachable!(),
    }
}

fn read(context: &ShotContext<'_>, name: &str) -> bool {
    context
        .named_branch_selectors
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .expect("named measurement is exposed to the shot hook")
        .1
}

#[rstest::rstest]
#[case("X")]
#[case("Z")]
#[case("Y")]
#[case("XY")]
#[case("XZ")]
#[case("YZ")]
fn terminal_measurement_absorption_preserves_the_signed_t_instrument(#[case] cap: &str) {
    for feedback in [false, true] {
        let original = source(cap, feedback);
        let rewritten = original.with_temporal_hadamards().unwrap();
        assert!(
            !rewritten
                .get_pipe(IVec3::new(-1, 0, 1), IVec3::new(-1, 0, 2))
                .unwrap()
                .is_hadamard()
        );
        assert_eq!(rewritten.t_count(), 1);
        assert_eq!(
            rewritten.get_block(IVec3::new(1, 0, 0)).unwrap().kind(),
            BlockKind::T
        );
        for graph in [&original, &rewritten] {
            let verifier = LogicalVerifier::new(graph).unwrap();
            for control in [false, true] {
                let axis = measured_axis(cap, control);
                for outcome in [false, true] {
                    // Independent one-qubit T -> H -> conditional X -> Pauli measurement.
                    let correction = if control && feedback { "x q[0];" } else { "" };
                    let measurement = match axis {
                        PauliBasis::X => "h q[0];",
                        PauliBasis::Y => "sdg q[0]; h q[0];",
                        PauliBasis::Z => "",
                    };
                    let mut expected: QuizxGraph = parse_qasm(&format!(
                        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; t q[0]; h q[0]; {correction} {measurement}"
                    )).unwrap().to_graph();
                    expected.plug_outputs(&[if outcome {
                        BasisElem::Z1
                    } else {
                        BasisElem::Z0
                    }]);
                    verifier
                        .verify_branch(
                            &expected,
                            &BTreeMap::from([("mzz".into(), control), ("m".into(), outcome)]),
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "{cap} feedback={feedback} control={control} m={outcome}: {error}"
                            )
                        });
                }
            }
            let mut program =
                CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
                    .compile_and_validate(graph)
                    .unwrap()
                    .bloq;
            // Expose existing corrected named values to the shot hook without changing execution.
            let actions = graph.actions();
            let ids = program
                .nodes()
                .filter_map(|(id, node)| {
                    let NodeProvenance::Action { ordinal } = node.provenance else {
                        return None;
                    };
                    let Action::Measure { name, .. } = &actions[ordinal as usize] else {
                        return None;
                    };
                    Some((id, name.clone()))
                })
                .collect::<Vec<_>>();
            for (id, name) in ids {
                program.node_mut(id).unwrap().provenance = NodeProvenance::BranchSelector { name };
            }
            let reference = Cell::new(0);
            let mut sectors = BTreeSet::new();
            let report = run_bloq_with_io(&program, 32, 17,
                |sim, context| {
                    assert_eq!(context.inputs.len(), 1);
                    reference.set(sim.num_qubits());
                    sim.cx(context.inputs[0].qubit, reference.get())?;
                    Ok(())
                },
                |sim, context| {
                    let (control, outcome) = (read(context, "mzz"), read(context, "m"));
                    sectors.insert((control, outcome));
                    let axis = measured_axis(cap, control);
                    // Measuring P after T,H leaves its transposed Heisenberg axis on the Bell reference.
                    let expected = match axis {
                        PauliBasis::X => [0.0, 0.0, 1.0],
                        PauliBasis::Y => [-FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0],
                        PauliBasis::Z => [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0],
                    };
                    let sign = if outcome ^ (feedback && control && axis != PauliBasis::X) { -1.0 } else { 1.0 };
                    let reference = physical_qubit(reference.get(), sim.num_qubits());
                    let actual = logical_bloch(sim, &reference, (false, false))?;
                    for (actual, expected) in [actual.0, actual.1, actual.2].into_iter().zip(expected) {
                        assert!((actual - sign * expected).abs() < 1e-8, "{cap} feedback={feedback} control={control} m={outcome}: {actual} != {}", sign * expected);
                    }
                    Ok(())
                }
            ).unwrap();
            assert_eq!(report.discarded, 0);
            assert!(!report.detectors.is_empty());
            assert!(report.all_detectors_constant());
            assert!(report.max_rank > 1);
            assert_eq!(sectors.len(), 4);
        }
    }
}
