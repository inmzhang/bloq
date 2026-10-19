//! Pinned TopoLS layouts checked against source maps and independent physical channels.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{
    Action, BlockGraph,
    verify::{BoundaryOrder, BranchStatus, LogicalVerifier, QuizxGraph, parse_qasm},
};
use bloq_vm::verify::{logical_signature, signatures_match};
use bloq_vm::{EnginePauli, EnginePauliString, ExecError, Simulator};
use glam::IVec3;
use serde::Deserialize;

mod common;
use common::choi::exact::ExactChoi;
use common::choi::stabilizer::physical_qubit;

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
    input_ports: Vec<[i32; 3]>,
    output_ports: Vec<[i32; 3]>,
    t_gadgets: usize,
    upstream_revision: String,
}

struct Fixture {
    graph: BlockGraph,
    qasm: String,
    inputs: Vec<IVec3>,
    outputs: Vec<IVec3>,
    t_gadgets: usize,
}

fn fixture(name: &str) -> Fixture {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/topols");
    let qasm = std::fs::read_to_string(directory.join(format!("{name}.qasm")))
        .expect("exported source QASM is readable");
    let graph = BlockGraph::from_blog_text(
        &std::fs::read_to_string(directory.join(format!("{name}.blog")))
            .expect("exported BLOG is readable"),
    )
    .expect("exported BLOG parses");
    let metadata: Metadata = serde_json::from_str(
        &std::fs::read_to_string(directory.join(format!("{name}.json")))
            .expect("exported metadata is readable"),
    )
    .expect("exported metadata parses");
    assert_eq!(metadata.upstream_revision, UPSTREAM_REVISION);
    Fixture {
        graph,
        qasm,
        inputs: metadata.input_ports.into_iter().map(IVec3::from).collect(),
        outputs: metadata.output_ports.into_iter().map(IVec3::from).collect(),
        t_gadgets: metadata.t_gadgets,
    }
}

fn verify_source(fixture: &Fixture) -> QuizxGraph {
    // QuiZX parses the original source independently of TopoLS normalization.
    let expected = parse_qasm(&fixture.qasm)
        .expect("source QASM has an independent QuiZX map")
        .to_graph();
    let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
        &fixture.graph,
        BoundaryOrder::new(fixture.inputs.clone(), fixture.outputs.clone()),
    )
    .expect("exported interface admits a logical verifier");
    let report = verifier
        .verify(&expected, 32, 7)
        .expect("every imported branch has the source logical map");
    assert_eq!(report.verified, 32, "sampled logical branches");
    expected
}

fn ideal(name: &str, simulator: &mut Simulator) -> Result<(), ExecError> {
    match name {
        "s" => simulator.s(0),
        "sdg" => simulator.s_dag(0),
        "t" => simulator.t(0)?,
        "tdg" => simulator.t_dag(0)?,
        "thth" => {
            simulator.t(0)?;
            simulator.h(0);
            simulator.t(0)?;
            simulator.h(0);
        }
        "paulis" => {
            simulator.h(0);
            simulator.x(0);
            simulator.z(0);
        }
        "entangled_t" => {
            simulator.h(0);
            simulator.cx(0, 1)?;
            simulator.t(1)?;
        }
        "entangled_tdg" => {
            simulator.h(0);
            simulator.cx(0, 1)?;
            simulator.t_dag(1)?;
            simulator.cx(1, 0)?;
        }
        "entangled_st" => {
            simulator.h(0);
            simulator.s(0);
            simulator.t(0)?;
            simulator.cx(0, 1)?;
        }
        "adder4" => {
            for qubit in 0..3 {
                simulator.h(qubit);
            }
            simulator.h(3);
            simulator.ccz(0, 2, 3)?;
            simulator.h(3);
            simulator.cx(0, 2)?;
            simulator.cx(1, 3)?;
        }
        "ising4" => {
            simulator.x(0);
            simulator.x(2);
            for _ in 0..2 {
                for qubit in 0..3 {
                    simulator.cx(qubit, qubit + 1)?;
                    simulator.s(qubit + 1);
                    simulator.cx(qubit, qubit + 1)?;
                }
                for qubit in 0..4 {
                    simulator.h(qubit);
                    simulator.t(qubit)?;
                    simulator.h(qubit);
                }
            }
        }
        _ => panic!("no independent physical ideal for {name}"),
    }
    Ok(())
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
fn real_topols_layout_has_the_source_channel_at_d3(#[case] name: &str) {
    let fixture = fixture(name);
    let expected = verify_source(&fixture);
    let Fixture {
        graph,
        inputs,
        outputs,
        t_gadgets,
        ..
    } = fixture;
    let boundaries = BoundaryOrder::new(inputs.clone(), outputs.clone());

    if matches!(name, "t" | "tdg") {
        let pure = LogicalVerifier::with_boundaries(&graph, boundaries)
            .expect("single T fixture admits explicit branch verification");
        let measurement = graph
            .actions()
            .into_iter()
            .find_map(|action| match action {
                Action::Measure { name, .. } => Some(name),
                _ => None,
            })
            .expect("single T fixture declares its readout");
        for value in [false, true] {
            let branch = pure
                .verify_branch(&expected, &BTreeMap::from([(measurement.clone(), value)]))
                .expect("each named T outcome preserves the signed channel");
            assert_eq!(branch.status(), BranchStatus::Verified);
        }
    }

    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .expect("imported graph compiles at distance three")
        .bloq;
    bloq.validate().expect("compiled IR validates");
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| ideal(name, sim));
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut frames = BTreeSet::new();
    for seed in SEEDS {
        let report = oracle.run(&bloq, SHOTS, seed, |_, context| {
            for &(selector, value) in context.named_branch_selectors {
                selectors
                    .entry(selector.to_owned())
                    .or_default()
                    .insert(value);
            }
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
        assert_eq!(report.shots, SHOTS);
        assert_eq!(report.discarded, 0, "{name}: no postselection");
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
    assert_eq!(selectors.len(), t_gadgets, "{name}: T selectors");
    for (selector, values) in selectors {
        assert_eq!(
            values,
            BTreeSet::from([false, true]),
            "{name}: selector {selector}"
        );
    }
    assert!(
        frames.len() > 1,
        "{name}: terminal frames vary across shots"
    );
}

#[rstest::rstest]
#[case("ising4", 8)]
#[case("adder4", 7)]
fn topols_algorithm_has_the_expected_prepared_state_at_d3(
    #[case] name: &str,
    #[case] t_gadgets: usize,
) {
    let fixture = fixture(name);
    verify_source(&fixture);
    assert_eq!(fixture.inputs.len(), 4);
    assert_eq!(fixture.outputs.len(), 4);
    assert_eq!(fixture.t_gadgets, t_gadgets);
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("algorithm compiles at distance three")
        .bloq;
    bloq.validate().expect("compiled algorithm IR validates");
    // No reference halves are needed for a known |0000> input: this checks
    // every one of the 255 nonidentity joint output Pauli expectations.
    let oracle = ExactChoi::new(&[], &fixture.outputs, |sim| ideal(name, sim));
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
                for &(selector, value) in context.named_branch_selectors {
                    selectors
                        .entry(selector.to_owned())
                        .or_default()
                        .insert(value);
                }
                Ok(())
            },
        );
        assert_eq!(report.shots, SHOTS);
        assert_eq!(report.discarded, 0, "{name}: no postselection");
        assert!(
            report.all_detectors_constant(),
            "{name}: deterministic detectors"
        );
        assert!(
            report.max_rank > 1,
            "{name}: non-Clifford physical execution"
        );
    }
    assert_eq!(selectors.len(), t_gadgets, "{name}: T selectors");
    for (selector, values) in selectors {
        assert_eq!(
            values,
            BTreeSet::from([false, true]),
            "{name}: selector {selector}"
        );
    }
}
