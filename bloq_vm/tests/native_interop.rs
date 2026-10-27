//! Small native compiler channels; large benchmarks are compilation-only.

mod common;

use bloq_compile::CompileConfig;
use bloq_interop::{GridOptions, LayoutOptions, Strategy, layout_qasm};
use common::choi::exact::ExactChoi;
use rstest::rstest;

#[rstest]
#[case::t(false, false)]
#[case::tdg(false, true)]
#[case::ht(true, false)]
fn native_single_t_phases(
    #[case] hadamard: bool,
    #[case] inverse: bool,
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    assert_single_t_phase(strategy, hadamard, inverse, true);
}

#[rstest]
fn native_cultivation_gateway_phases(
    #[values(false, true)] inverse: bool,
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    // The full factory's signed X/Y gateway and retry boundary are the subject.
    assert_single_t_phase(strategy, false, inverse, false);
}

fn assert_single_t_phase(strategy: Strategy, hadamard: bool, inverse: bool, ideal_t: bool) {
    let source = format!(
        "OPENQASM 2.0; qreg q[1]; {} {} q[0];",
        if hadamard { "h q[0];" } else { "" },
        if inverse { "tdg" } else { "t" },
    );
    let layout = layout_qasm(
        &source,
        &LayoutOptions {
            strategy,
            ..Default::default()
        },
    )
    .expect("single-T native layout is valid");
    let cap = layout
        .graph
        .blocks()
        .find(|block| matches!(block.kind(), bloq_graph::BlockKind::Selective(_)))
        .expect("single-T layout has a selective cap")
        .pos();
    let selector = bloq_graph::selective_selector_name(cap);
    let compiled = layout
        .compile(CompileConfig::new(3).with_prepare_t_with_mpps(ideal_t))
        .expect("single-T physical compilation passes the IR audit");
    let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
        if hadamard {
            sim.h(0);
        }
        if inverse {
            sim.t_dag(0)?;
        } else {
            sim.t(0)?;
        }
        Ok(())
    });
    let mut outcomes = [false; 2];
    let report = channel.run(&compiled.bloq, 16, 0x1A7E, |_, context| {
        let outcome = context
            .named_branch_selectors
            .iter()
            .find(|(name, _)| *name == selector)
            .expect("native selective outcome")
            .1;
        outcomes[usize::from(outcome)] = true;
        Ok(())
    });
    assert_eq!(outcomes, [true, true]);
    assert!(report.all_detectors_constant());
}

#[rstest]
#[case::signed_phase(1, "t q[0]; h q[0]; tdg q[0]; h q[0]; s q[0];")]
// Adaptive resources follow both Hadamard frames in this entangled channel.
#[case::adaptive_entangled(2, "h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[0]; tdg q[0];")]
#[case::nonlocal(3, "h q[0]; cx q[0],q[2]; cz q[1],q[2]; t q[2];")]
fn native_channels(
    #[case] qubits: usize,
    #[case] gates: &str,
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    let source = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{qubits}]; {gates}");
    let layout = layout_qasm(
        &source,
        &LayoutOptions {
            strategy,
            ..Default::default()
        },
    )
    .expect("native layout is causal and valid");
    assert!(
        layout
            .graph
            .blocks()
            .all(|block| !matches!(block.kind(), bloq_graph::BlockKind::PatchRotation(_)))
    );
    assert!(
        layout
            .graph
            .pipes()
            .all(|pipe| !pipe.is_hadamard() || !pipe.dir().is_spatial())
    );
    let compiled = layout
        .compile(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .expect("small physical program passes the full IR audit");
    let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
        match qubits {
            1 => {
                sim.t(0)?;
                sim.h(0);
                sim.t_dag(0)?;
                sim.h(0);
                sim.s(0);
            }
            2 => {
                sim.h(0);
                sim.cx(0, 1)?;
                sim.t(1)?;
                sim.h(1);
                sim.cx(1, 0)?;
                sim.t_dag(0)?;
            }
            3 => {
                sim.h(0);
                sim.cx(0, 2)?;
                sim.cz(1, 2)?;
                sim.t(2)?;
            }
            _ => unreachable!("three explicit small channel fixtures"),
        }
        Ok(())
    });
    channel.run(&compiled.bloq, 8, 0x1A7E, |_, _| Ok(()));
}

#[rstest]
fn mqt_grover3_preserves_the_algorithm_channel(
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    // Exact MQT Bench 2.3.0 Grover3 lowering; the oracle below uses CCZ directly.
    let source = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
h q[1];
x q[2];
cx q[0],q[2];
tdg q[2];
cx q[1],q[2];
t q[2];
cx q[0],q[2];
tdg q[2];
cx q[1],q[2];
t q[1];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
h q[0];
x q[0];
h q[1];
x q[1];
h q[1];
cx q[0],q[1];
x q[0];
h q[0];
h q[1];
x q[1];
h q[1];
t q[2];
"#;
    let layout = layout_qasm(
        source,
        &LayoutOptions {
            strategy,
            grid: Some(GridOptions {
                columns: 2,
                ..Default::default()
            }),
            measurement_batch_size: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    let compiled = layout
        .compile(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .unwrap();
    let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
        sim.h(0);
        sim.h(1);
        sim.x(2);
        sim.ccz(0, 1, 2)?;
        for q in [0, 1] {
            sim.h(q);
            sim.x(q);
        }
        sim.cz(0, 1)?;
        for q in [0, 1] {
            sim.x(q);
            sim.h(q);
        }
        Ok(())
    });
    let report = channel.run(&compiled.bloq, 8, 0x1A7E, |_, _| Ok(()));
    assert!(report.all_detectors_constant());
}

#[rstest]
fn mqt_qpe3_preserves_the_algorithm_channel(
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    // Exact MQT Bench 2.3.0 QPE3 lowering; the oracle groups its controlled phases.
    let source = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
h q[1];
x q[2];
s q[2];
cx q[2],q[0];
sdg q[0];
cx q[2],q[0];
s q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
h q[0];
tdg q[1];
cx q[1],q[0];
t q[0];
cx q[1],q[0];
tdg q[0];
h q[1];
"#;
    let layout = layout_qasm(
        source,
        &LayoutOptions {
            strategy,
            grid: Some(GridOptions {
                columns: 2,
                ..Default::default()
            }),
            measurement_batch_size: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    let compiled = layout
        .compile(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .unwrap();
    let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
        sim.h(0);
        sim.h(1);
        sim.x(2);
        sim.cz(2, 0)?;
        sim.swap(0, 1);
        sim.h(0);
        // Controlled S† from its parity phase, independently of the source lowering.
        sim.t_dag(0)?;
        sim.t_dag(1)?;
        sim.cx(1, 0)?;
        sim.t(0)?;
        sim.cx(1, 0)?;
        sim.h(1);
        Ok(())
    });
    let report = channel.run(&compiled.bloq, 8, 0x1A7E, |_, _| Ok(()));
    assert!(report.all_detectors_constant());
}

#[rstest]
fn numerical_resource_budgets_preserve_the_channel(
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    let source = "OPENQASM 2.0; qreg q[3]; cz q[1],q[2]; tdg q[2]; cx q[1],q[0]; h q[2]; tdg q[1]; cz q[1],q[2]; h q[2]; t q[1];";
    let layouts = [0, 2].map(|extra_patches| {
        layout_qasm(
            source,
            &LayoutOptions {
                strategy,
                extra_patches,
                grid: Some(GridOptions {
                    columns: 2,
                    ..Default::default()
                }),
                measurement_batch_size: Some(2),
                ..Default::default()
            },
        )
        .unwrap()
    });
    for layout in layouts {
        let program = layout
            .compile(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .unwrap();
        let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
            sim.cz(1, 2)?;
            sim.t_dag(2)?;
            sim.cx(1, 0)?;
            sim.h(2);
            sim.t_dag(1)?;
            sim.cz(1, 2)?;
            sim.h(2);
            sim.t(1)?;
            Ok(())
        });
        channel.run(&program.bloq, 8, 0xA7EA, |_, _| Ok(()));
    }
}

#[test]
fn signed_t_faces_preserve_the_same_magic_phase() {
    for hadamard in [false, true] {
        let source = format!(
            "OPENQASM 2.0; qreg q[1]; {} t q[0];",
            if hadamard { "h q[0];" } else { "" }
        );
        let layout = layout_qasm(&source, &LayoutOptions::default()).unwrap();
        for turns in [0, 2] {
            let graph = layout
                .graph
                .rotate_about_origin(bloq_graph::UDirection::Z, turns)
                .unwrap();
            let program = bloq_compile::CompileContext::new(
                CompileConfig::new(3).with_prepare_t_with_mpps(true),
            )
            .compile_and_validate(&graph)
            .unwrap();
            let channel = ExactChoi::new(&layout.input_ports, &layout.output_ports, |sim| {
                if hadamard {
                    sim.h(0);
                }
                sim.t(0)?;
                Ok(())
            });
            channel.run(&program.bloq, 8, 0xFACE, |_, _| Ok(()));
        }
    }
}
