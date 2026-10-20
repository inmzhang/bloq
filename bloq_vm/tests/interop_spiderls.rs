//! Pinned SpiderLS geometry checked against independent source and physical channels.

use std::collections::{BTreeMap, BTreeSet};

use bloq_compile::{CompileConfig, CompileContext};
use serde::Deserialize;

mod common;
use common::choi::exact::ExactChoi;
use common::interop::{Fixture, apply_source_qasm, assert_selectors_toggle, record_selectors};

const UPSTREAM_REVISION: &str = "2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea";
const SHOTS: usize = 8;
const SEEDS: [u64; 3] = [0x11, 0x22, 0x33];

#[derive(Deserialize)]
struct Metadata {
    upstream_revision: String,
}

fn fixture(name: &str) -> Fixture<Metadata> {
    let fixture: Fixture<Metadata> = common::interop::fixture(&format!("spiderls/{name}"));
    assert_eq!(fixture.metadata.upstream_revision, UPSTREAM_REVISION);
    fixture
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
    let expected = fixture.verify_source();
    if matches!(name, "t" | "tdg" | "phase3" | "phase5") {
        fixture.verify_named_outcomes(&expected);
    }
    let program = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("imported graph compiles at distance three")
        .bloq;
    program.validate().expect("compiled IR validates");
    let oracle = ExactChoi::new(&fixture.inputs, &fixture.outputs, |sim| {
        apply_source_qasm(sim, &fixture.qasm);
        Ok(())
    });
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    for seed in SEEDS {
        let report = oracle.run(&program, SHOTS, seed, |_, context| {
            record_selectors(&mut selectors, context.named_branch_selectors);
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
    assert_selectors_toggle(&selectors, fixture.t_gadgets, name);
}

#[test]
fn spiderls_walk_has_the_expected_prepared_state_at_d3() {
    let fixture = fixture("walk3");
    fixture.verify_source();
    assert_eq!(fixture.inputs.len(), 3);
    assert_eq!(fixture.outputs.len(), 3);
    let program = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("algorithm compiles at distance three")
        .bloq;
    program.validate().expect("compiled algorithm IR validates");
    // Check every joint Pauli expectation of the prepared three-qubit state.
    let oracle = ExactChoi::new(&[], &fixture.outputs, |sim| {
        apply_source_qasm(sim, &fixture.qasm);
        Ok(())
    });
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
                record_selectors(&mut selectors, context.named_branch_selectors);
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
    assert_selectors_toggle(&selectors, fixture.t_gadgets, "walk3");
}
