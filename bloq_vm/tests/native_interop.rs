//! Small native compiler channels; large benchmarks are compilation-only.

mod common;

use bloq_compile::CompileConfig;
use bloq_interop::{LayoutOptions, Strategy, layout_qasm};
use common::choi::exact::ExactChoi;
use rstest::rstest;

#[rstest]
#[case::signed_phase(1, "t q[0]; h q[0]; tdg q[0]; h q[0]; s q[0];")]
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
