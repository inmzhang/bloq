//! Small native compiler channels; large benchmarks are compilation-only.

mod common;

use bloq_compile::CompileConfig;
use bloq_interop::{GridOptions, LayoutOptions, LayoutTradeoff, Strategy, layout_qasm};
use common::choi::exact::ExactChoi;
use rstest::rstest;

#[rstest]
#[case::signed_phase(1, "t q[0]; h q[0]; tdg q[0]; h q[0]; s q[0];")]
// Inward T placement reuses a data column between rotations in this channel.
#[case::adaptive_entangled(2, "h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[0]; tdg q[0];")]
#[case::nonlocal(3, "h q[0]; cx q[0],q[2]; cz q[1],q[2]; t q[2];")]
fn native_channels(
    #[case] qubits: usize,
    #[case] gates: &str,
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
    #[values(None, Some(GridOptions { columns: 2, ..GridOptions::default() }))] grid: Option<
        GridOptions,
    >,
) {
    let source = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{qubits}]; {gates}");
    let layout = layout_qasm(
        &source,
        &LayoutOptions {
            strategy,
            grid,
            ..Default::default()
        },
    )
    .expect("native layout is causal and valid");
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
    if let Some(root) = std::env::var_os("BLOQ_INTEROP_EVIDENCE_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).expect("create evidence root");
        let directory = root.join(format!(
            "{strategy:?}-q{qubits}{}",
            if grid.is_some() { "-grid" } else { "" }
        ));
        std::fs::create_dir(&directory).expect("evidence directory must be new");
        for (name, bytes) in [
            ("source.qasm", source.into_bytes()),
            ("layout.blog", layout.graph.to_blog_text().into_bytes()),
            ("program_d3.bloq", compiled.bloq.to_binary()),
            ("execution.json", serde_json::to_vec_pretty(&serde_json::json!({
                "verification": "sampled exact normalized Choi channel",
                "distance": 3, "t_preparation": "ideal_t_proxy", "grid_columns": grid.map(|g| g.columns),
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
}

#[rstest]
fn area_time_tradeoff_preserves_the_channel(
    #[values(Strategy::TopoLs, Strategy::SpiderLs)] strategy: Strategy,
) {
    let source = "OPENQASM 2.0; qreg q[3]; cz q[1],q[2]; tdg q[2]; cx q[1],q[0]; h q[2]; tdg q[1]; cz q[1],q[2]; h q[2]; t q[1];";
    let layouts = [
        LayoutTradeoff::Compact,
        LayoutTradeoff::Depth { extra_patches: 2 },
    ]
    .map(|tradeoff| {
        layout_qasm(
            source,
            &LayoutOptions {
                strategy,
                tradeoff,
                ..Default::default()
            },
        )
        .unwrap()
    });
    let metrics = |layout: &bloq_interop::Layout| {
        let (x, y, z) = layout.graph.spans().unwrap();
        (
            (*x.end() - *x.start() + 1) * (*y.end() - *y.start() + 1),
            *z.end() - *z.start() + 1,
        )
    };
    let (compact, depth) = (metrics(&layouts[0]), metrics(&layouts[1]));
    assert!(depth.1 <= compact.1);
    if strategy == Strategy::TopoLs {
        assert!(depth.0 > compact.0 && depth.1 < compact.1);
    }
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
