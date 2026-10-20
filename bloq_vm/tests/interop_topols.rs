//! Pinned TopoLS layouts checked against source maps and independent physical channels.

use std::collections::{BTreeMap, BTreeSet};

use bloq_compile::{CompileConfig, CompileContext};
use bloq_vm::verify::{logical_signature, signatures_match};
use bloq_vm::{EnginePauli, EnginePauliString, Simulator};
use serde::Deserialize;

mod common;
use common::choi::exact::ExactChoi;
use common::choi::stabilizer::physical_qubit;
use common::interop::{Fixture, apply_source_qasm, assert_selectors_toggle, record_selectors};

const UPSTREAM_REVISION: &str = "e564d350a596cfea3227e09698733210e2e14dcf";
const SHOTS: usize = 8;
const SEEDS: [u64; 3] = [0x11, 0x22, 0x33];

#[rstest::rstest]
#[case(false)]
#[case(true)]
fn signed_t_native_instrument_has_complete_kraus_branches(#[case] dagger: bool) {
    let mut ideal = Simulator::with_seed(2, 0);
    ideal.h(0);
    ideal.cx(0, 1).unwrap();
    if dagger {
        ideal.t_dag(0).unwrap();
    } else {
        ideal.t(0).unwrap();
    }
    let ideal_outputs = [physical_qubit(0, 2), physical_qubit(1, 2)];
    let expected = logical_signature(
        &ideal,
        &ideal_outputs
            .iter()
            .map(|out| (out, (false, false)))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut total_probability = 0.0;
    for parity in [false, true] {
        for readout in [false, true] {
            // Bell inputs test the entire Kraus operator, rather than one
            // selected state. The resource is always positive |T>.
            let mut sim = Simulator::with_seed(3, 0);
            sim.h(0);
            sim.cx(0, 1).unwrap();
            sim.h(2);
            sim.t(2).unwrap();
            let zz = EnginePauliString::from_terms(3, [(0, EnginePauli::Z), (2, EnginePauli::Z)]);
            let joint = sim.postselect_observable(&zz, parity).unwrap();
            // Positive T: X for even parity, Y for odd. T† exchanges them.
            let terminal = if parity == dagger {
                sim.postselect_x(2, readout)
            } else {
                sim.postselect_y(2, readout)
            }
            .unwrap();
            if readout {
                sim.z(0);
            }
            let probability = joint.probability * terminal.probability;
            assert!(
                (probability - 0.25).abs() < 1e-9,
                "T†={dagger} m={parity} n={readout}"
            );
            total_probability += probability;
            let outputs = [physical_qubit(0, 3), physical_qubit(1, 3)];
            let actual = logical_signature(
                &sim,
                &outputs
                    .iter()
                    .map(|out| (out, (false, false)))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            assert!(signatures_match(&actual, &expected));
        }
    }
    // Every corrected branch is (phase / 2) T±, so each K†K = I/4.
    assert!((total_probability - 1.0).abs() < 1e-9);
}

#[derive(Deserialize)]
struct Metadata {
    upstream_revision: String,
}

fn fixture(name: &str) -> Fixture<Metadata> {
    let fixture: Fixture<Metadata> = common::interop::fixture(&format!("topols/{name}"));
    assert_eq!(fixture.metadata.upstream_revision, UPSTREAM_REVISION);
    fixture
}

#[rstest::rstest]
#[case("s")]
#[case("sdg")]
#[case("t")]
#[case("tdg")]
#[case("thth")]
#[case("paulis")]
#[case("entangled_t")]
#[case("entangled_tdg")]
#[case("entangled_st")]
#[case("branching_lift")]
fn real_topols_layout_has_the_source_channel_at_d3(#[case] name: &str) {
    let fixture = fixture(name);
    let expected = fixture.verify_source();
    if matches!(name, "t" | "tdg") {
        fixture.verify_named_outcomes(&expected);
    }
    let Fixture {
        graph,
        qasm,
        inputs,
        outputs,
        t_gadgets,
        ..
    } = fixture;

    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .expect("imported graph compiles at distance three")
        .bloq;
    bloq.validate().expect("compiled IR validates");
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        apply_source_qasm(sim, &qasm);
        Ok(())
    });
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut frames = BTreeSet::new();
    for seed in SEEDS {
        let report = oracle.run(&bloq, SHOTS, seed, |_, context| {
            record_selectors(&mut selectors, context.named_branch_selectors);
            frames.insert(
                outputs
                    .iter()
                    .map(|port| {
                        let frame = context
                            .frames
                            .iter()
                            .find(|frame| frame.port == *port)
                            .expect("every declared output has a frame");
                        (
                            frame.x.expect("evaluable X frame"),
                            frame.z.expect("evaluable Z frame"),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(())
        });
        assert!(
            report.all_detectors_constant(),
            "{name}: deterministic detectors"
        );
        if t_gadgets > 0 {
            assert!(
                report.max_rank > 1,
                "{name}: non-Clifford physical execution"
            );
        }
    }
    assert_selectors_toggle(&selectors, t_gadgets, name);
    assert!(
        frames.len() > 1,
        "{name}: terminal frames vary across shots"
    );
}

#[rstest::rstest]
#[case("ising4", 8, 4)]
#[case("adder4", 7, 4)]
#[case("adder4_early_t", 7, 4)]
#[case("qae3", 6, 3)]
fn topols_algorithm_has_the_expected_prepared_state_at_d3(
    #[case] name: &str,
    #[case] t_gadgets: usize,
    #[case] qubits: usize,
) {
    let fixture = fixture(name);
    fixture.verify_source();
    assert_eq!(fixture.inputs.len(), qubits);
    assert_eq!(fixture.outputs.len(), qubits);
    assert_eq!(fixture.t_gadgets, t_gadgets);
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("algorithm compiles at distance three")
        .bloq;
    bloq.validate().expect("compiled algorithm IR validates");
    // A known zero input needs no reference halves. The oracle checks every
    // nonidentity joint output Pauli expectation (255 for four qubits).
    let oracle = ExactChoi::new(&[], &fixture.outputs, |sim| {
        apply_source_qasm(sim, &fixture.qasm);
        Ok(())
    });
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    for seed in SEEDS {
        let report = oracle.run_prepared(
            &bloq,
            SHOTS,
            seed,
            &fixture.inputs,
            |sim, context| {
                for input in context.inputs {
                    sim.h(input.qubit);
                }
                Ok(())
            },
            |_, context| {
                record_selectors(&mut selectors, context.named_branch_selectors);
                Ok(())
            },
        );
        assert!(
            report.all_detectors_constant(),
            "{name}: deterministic detectors"
        );
        assert!(
            report.max_rank > 1,
            "{name}: non-Clifford physical execution"
        );
    }
    assert_selectors_toggle(&selectors, t_gadgets, name);
}
