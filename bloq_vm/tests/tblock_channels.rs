//! T-block compiler output runs as its source channel on the physical VM.

mod common;

use bloq_compile::{CompileConfig, compile_with};
use bloq_interop::parse_qasm;
use bloq_tblock::{MachineOptions, Options, TMode, compile};
use common::choi::exact::ExactChoi;
use rstest::rstest;

/// A Toffoli as H on the target around the textbook seven-T CCZ.
const TOFFOLI: &str = "h q[2]; cx q[1],q[2]; tdg q[2]; cx q[0],q[2]; t q[2]; cx q[1],q[2]; \
                       tdg q[2]; cx q[0],q[2]; t q[1]; t q[2]; cx q[0],q[1]; t q[0]; tdg q[1]; \
                       cx q[0],q[1]; h q[2];";

#[rstest]
fn toffoli_channel(#[values(TMode::Gadget, TMode::Block)] t_mode: TMode) {
    let circuit = parse_qasm(&format!(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; {TOFFOLI}"
    ))
    .expect("source parses");
    let options = Options {
        machine: MachineOptions {
            t_mode,
            ..Default::default()
        },
        ..Default::default()
    };
    let built = compile(&circuit, &options).expect("layout compiles");
    // Block mode measures dependent T parities between sites, not on the data.
    assert_eq!(built.checks > 0, t_mode == TMode::Block);
    let compiled = compile_with(
        &built.graph,
        CompileConfig::new(3).with_prepare_t_with_mpps(true),
    )
    .expect("physical program passes the IR audit");
    let channel = ExactChoi::new(&built.inputs, &built.outputs, |sim| {
        sim.h(2);
        sim.cx(1, 2)?;
        sim.t_dag(2)?;
        sim.cx(0, 2)?;
        sim.t(2)?;
        sim.cx(1, 2)?;
        sim.t_dag(2)?;
        sim.cx(0, 2)?;
        sim.t(1)?;
        sim.t(2)?;
        sim.cx(0, 1)?;
        sim.t(0)?;
        sim.t_dag(1)?;
        sim.cx(0, 1)?;
        sim.h(2);
        Ok(())
    });
    channel.run(&compiled.bloq, 8, 0x7B10, |_, _| Ok(()));
}
