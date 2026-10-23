//! Small native compiler channels; large benchmarks are compilation-only.

mod common;

use bloq_compile::CompileConfig;
use bloq_interop::{GridOptions, LayoutOptions, Strategy, layout_qasm};
use common::choi::exact::ExactChoi;
use rstest::rstest;

fn save_evidence(
    case: &str,
    source: &str,
    layout: &bloq_interop::Layout,
    program: &bloq_ir::Bloq,
    report: &bloq_vm::verify::VerifyReport,
) {
    let Some(root) = std::env::var_os("BLOQ_INTEROP_EVIDENCE_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root).expect("create evidence root");
    let directory = root.join(case);
    std::fs::create_dir(&directory).expect("evidence directory must be new");
    for (name, bytes) in [
        ("source.qasm", source.as_bytes().to_vec()),
        ("layout.blog", layout.graph.to_blog_text().into_bytes()),
        ("program_d3.bloq", program.to_binary()),
        ("execution.json", serde_json::to_vec_pretty(&serde_json::json!({
            "case": case, "verification": "sampled exact normalized Choi channel",
            "distance": 3, "t_preparation": "ideal_t_proxy", "grid_columns": layout.grid.map(|g| g.columns),
            "shots": report.shots, "seed": 0x1A7E,
            "discarded": report.discarded, "execution_verified": true,
            "all_detectors_constant": report.all_detectors_constant(),
            "input_ports": layout.input_ports.iter().map(glam::IVec3::to_array).collect::<Vec<_>>(),
            "output_ports": layout.output_ports.iter().map(glam::IVec3::to_array).collect::<Vec<_>>(),
        })).expect("serialize execution evidence")),
    ] {
        std::fs::write(directory.join(name), bytes).expect("save exact executed artifacts");
    }
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
    #[values(GridOptions::default(), GridOptions { columns: 2, ..GridOptions::default() })]
    grid: GridOptions,
) {
    let source = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{qubits}]; {gates}");
    let layout = layout_qasm(
        &source,
        &LayoutOptions {
            strategy,
            grid: Some(grid),
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
    let report = channel.run(&compiled.bloq, 8, 0x1A7E, |_, _| Ok(()));
    save_evidence(
        &format!(
            "{strategy:?}-q{qubits}{}",
            if grid.columns == 0 {
                "-auto-grid"
            } else {
                "-grid2"
            }
        ),
        &source,
        &layout,
        &compiled.bloq,
        &report,
    );
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
    save_evidence(
        &format!("{strategy:?}-grover3-grid2"),
        source,
        &layout,
        &compiled.bloq,
        &report,
    );
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
    save_evidence(
        &format!("{strategy:?}-qpe3-grid2"),
        source,
        &layout,
        &compiled.bloq,
        &report,
    );
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
