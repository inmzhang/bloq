//! Compiled layouts match their source maps under every compiler mode.

use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
use bloq_interop::parse_qasm;
use bloq_tblock::{Deferral, MachineOptions, Options, Placement, Reduction, TMode, compile};

/// Toffolis give T batches whose rank is below their size, so block mode places checks.
const SOURCES: [&str; 3] = [
    "qreg q[3]; h q[2]; ccx q[0],q[1],q[2]; t q[0]; cx q[2],q[1]; h q[0]; tdg q[1];",
    "qreg q[4]; h q[0]; cx q[0],q[3]; ccx q[3],q[1],q[2]; s q[2]; cz q[0],q[2]; h q[3]; t q[3];",
    "qreg q[2]; h q[1]; t q[1]; h q[1]; tdg q[0]; cx q[1],q[0]; h q[0]; t q[0]; x q[1];",
];

#[test]
fn layouts_match_sources_in_every_mode() {
    let grid = Placement::Grid { columns: 0 };
    let modes = [
        (Deferral::Full, Reduction::Raw, TMode::Gadget, grid),
        (
            Deferral::None,
            Reduction::Raw,
            TMode::Gadget,
            Placement::Strip,
        ),
        (Deferral::Full, Reduction::Zx, TMode::Gadget, grid),
        (Deferral::Full, Reduction::Raw, TMode::Block, grid),
    ];
    let mut checks = 0;
    for body in SOURCES {
        let circuit = parse_qasm(&format!("OPENQASM 2.0; include \"qelib1.inc\"; {body}"))
            .expect("source parses");
        for (deferral, reduction, t_mode, placement) in modes {
            let options = Options {
                reduction,
                deferral,
                machine: MachineOptions {
                    t_mode,
                    placement,
                    ..Default::default()
                },
            };
            let built = compile(&circuit, &options).expect("layout compiles");
            built.graph.validate_source().expect("layout is valid");
            LogicalVerifier::with_boundaries_and_internal_measurements(
                &built.graph,
                BoundaryOrder::new(built.inputs.clone(), built.outputs.clone()),
            )
            .expect("verifier accepts the layout")
            .verify(&circuit.to_graph::<QuizxGraph>(), 8, 7)
            .unwrap_or_else(|e| panic!("{body} under {options:?}: {e}"));
            checks += built.checks;
        }
    }
    assert!(checks > 0, "block mode shares at least one T site");
}
