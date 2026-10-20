//! Pinned SpiderLS geometry checked against independent source and physical channels.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{
    Action, BlockGraph,
    verify::{BoundaryOrder, BranchStatus, LogicalVerifier, QuizxGraph},
};
use bloq_vm::{ExecError, Simulator};
use glam::IVec3;
use serde::Deserialize;

mod common;
use common::choi::exact::ExactChoi;
use common::parse_qasm;

const UPSTREAM_REVISION: &str = "2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea";
const SHOTS: usize = 8;
const SEEDS: [u64; 3] = [0x11, 0x22, 0x33];

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
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/spiderls");
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
    // QuiZX parses the original source independently of SpiderLS/PyZX.
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
        "identity" => {}
        "h" => simulator.h(0),
        "z" => simulator.z(0),
        "hzh" => simulator.x(0),
        "parallel_local" => {
            simulator.x(0);
            simulator.x(1);
        }
        "s" => simulator.s(0),
        "sdg" => simulator.s_dag(0),
        "t" | "passthrough" => simulator.t(0)?,
        "tdg" => simulator.t_dag(0)?,
        "phase3" => {
            simulator.z(0);
            simulator.t_dag(0)?;
        }
        "phase5" => {
            simulator.z(0);
            simulator.t(0)?;
        }
        "thth" => {
            simulator.t(0)?;
            simulator.h(0);
            simulator.t(0)?;
            simulator.h(0);
        }
        "entangled_t" => {
            simulator.h(0);
            simulator.cx(0, 1)?;
            simulator.t(1)?;
        }
        "crossed_h" => {
            simulator.h(0);
            simulator.cx(0, 1)?;
            simulator.cx(1, 0)?;
            simulator.cx(0, 1)?;
        }
        "lifecycle" => {
            simulator.t(0)?;
            simulator.h(0);
            simulator.t(1)?;
            simulator.h(0);
            simulator.cx(1, 0)?;
            simulator.t(0)?;
            simulator.h(1);
            simulator.cx(1, 0)?;
            simulator.t(0)?;
            simulator.h(0);
            simulator.cx(0, 1)?;
            simulator.h(0);
            simulator.cx(1, 0)?;
        }
        "walk3" => {
            for _ in 0..3 {
                // RY(pi/4), then position +/-1 controlled by the coin.
                simulator.s_dag(0);
                simulator.h(0);
                simulator.t(0)?;
                simulator.h(0);
                simulator.s(0);
                simulator.x(2);
                simulator.cx(1, 2)?;
                simulator.cx(0, 2)?;
                simulator.x(1);
            }
        }
        _ => panic!("no independent physical ideal for {name}"),
    }
    Ok(())
}

#[rstest::rstest]
#[case("identity")]
#[case("h")]
#[case("z")]
#[case("hzh")]
#[case("parallel_local")]
#[case("s")]
#[case("sdg")]
#[case("t")]
#[case("tdg")]
#[case("phase3")]
#[case("phase5")]
#[case("thth")]
#[case("entangled_t")]
#[case("passthrough")]
#[case("lifecycle")]
#[case("crossed_h")]
fn spiderls_layout_has_the_source_channel_at_d3(#[case] name: &str) {
    let fixture = fixture(name);
    let expected = verify_source(&fixture);
    if matches!(name, "t" | "tdg" | "phase3" | "phase5") {
        let verifier = LogicalVerifier::with_boundaries(
            &fixture.graph,
            BoundaryOrder::new(fixture.inputs.clone(), fixture.outputs.clone()),
        )
        .expect("single T fixture admits explicit branch verification");
        let measurement = fixture
            .graph
            .actions()
            .into_iter()
            .find_map(|action| match action {
                Action::Measure { name, .. } => Some(name),
                _ => None,
            })
            .expect("single T fixture declares its readout");
        for value in [false, true] {
            let branch = verifier
                .verify_branch(&expected, &BTreeMap::from([(measurement.clone(), value)]))
                .expect("each named T outcome preserves the signed channel");
            assert_eq!(branch.status(), BranchStatus::Verified);
        }
    }
    let program = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("imported graph compiles at distance three")
        .bloq;
    program.validate().expect("compiled IR validates");
    let oracle = ExactChoi::new(&fixture.inputs, &fixture.outputs, |sim| ideal(name, sim));
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    for seed in SEEDS {
        let report = oracle.run(&program, SHOTS, seed, |_, context| {
            for &(selector, value) in context.named_branch_selectors {
                selectors
                    .entry(selector.to_owned())
                    .or_default()
                    .insert(value);
            }
            Ok(())
        });
        assert!(
            report.all_detectors_constant(),
            "{name}: deterministic detectors"
        );
        if fixture.t_gadgets > 0 {
            assert!(
                report.max_rank > 1,
                "{name}: non-Clifford physical execution"
            );
        }
    }
    assert_eq!(selectors.len(), fixture.t_gadgets, "{name}: T selectors");
    for (selector, values) in selectors {
        assert_eq!(
            values,
            BTreeSet::from([false, true]),
            "{name}: selector {selector}"
        );
    }
}

#[test]
fn spiderls_walk_has_the_expected_prepared_state_at_d3() {
    let fixture = fixture("walk3");
    verify_source(&fixture);
    assert_eq!(fixture.inputs.len(), 3);
    assert_eq!(fixture.outputs.len(), 3);
    let program = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("algorithm compiles at distance three")
        .bloq;
    program.validate().expect("compiled algorithm IR validates");
    // Check every joint Pauli expectation of the prepared three-qubit state.
    let oracle = ExactChoi::new(&[], &fixture.outputs, |sim| ideal("walk3", sim));
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    for seed in SEEDS {
        let report = oracle.run_prepared(
            &program,
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
        assert!(
            report.all_detectors_constant(),
            "walk3: deterministic detectors"
        );
        assert!(
            report.max_rank > 1,
            "walk3: non-Clifford physical execution"
        );
    }
    assert_eq!(selectors.len(), fixture.t_gadgets, "walk3: T selectors");
    for (selector, values) in selectors {
        assert_eq!(
            values,
            BTreeSet::from([false, true]),
            "walk3: selector {selector}"
        );
    }
}
