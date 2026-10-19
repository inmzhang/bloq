#![cfg(test)]

//! Sampled, normalized Choi branches for native stabilizer instruments.
//! External magic-resource ports are arbitrary Bell-paired inputs here; native
//! T preparation, postselection probabilities, and consumed outputs need their
//! dedicated physical oracles. Independent QuiZX checks remain separate.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{Action, GalleryCategory, MeasurementObservable, NodeKind, PauliBasis};
use bloq_test::TestFixture;

mod common;

#[test]
#[ignore = "Bell-input physical checks across native fixtures; run with --release --ignored"]
fn native_stabilizer_instruments_preserve_choi_correlations() {
    let mut checked = Vec::new();
    for &fixture in bloq_test::compile_fixtures() {
        let exclusion = if fixture == TestFixture::TenBitAdder {
            Some("three-bit adder covers the physical arithmetic map")
        } else if fixture.in_category(GalleryCategory::AnalysisOnly) {
            Some("documented compilation limitation")
        } else {
            None
        };
        if let Some(reason) = exclusion {
            eprintln!("Choi skip {}: {reason}", fixture.id());
            continue;
        }
        let program = if let Some(item) = fixture.gallery_item() {
            item.build()
        } else {
            let case = bloq_test::select_test_cases_for_fixture(fixture)
                .unwrap()
                .into_iter()
                .find(|case| {
                    case.metadata.origins.iter().any(|origin| {
                        origin.fixture == fixture
                            && origin.fill_variant.is_none()
                            && !origin.flip_xz_basis
                            && origin.rotation.is_none()
                    })
                })
                .unwrap();
            case.build().with_inferred_interface().unwrap()
        };
        let graph = program.flatten().unwrap();
        let exclusion = if graph.t_count() != 0 {
            Some("internal T preparation needs a non-stabilizer oracle")
        } else if graph
            .actions()
            .iter()
            .any(|action| matches!(action, Action::DiscardIf(_)))
        {
            Some("postselection probabilities need a separate oracle")
        } else {
            let zx = bloq_graph::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
            zx.action_graph().ordered_nodes().find_map(|action| {
                let Action::Measure { target, .. } = &action.action else {
                    return None;
                };
                let Some(MeasurementObservable::Concrete(axis @ (PauliBasis::X | PauliBasis::Z))) =
                    action.measurement
                else {
                    return Some("Y/selective named-readout coordinates need a separate oracle");
                };
                let column = zx.measurement_column(target).unwrap();
                zx.nodes().get(column).and_then(|node| {
                    matches!(
                        (node.kind, axis),
                        (NodeKind::X, PauliBasis::X) | (NodeKind::Z, PauliBasis::Z)
                    )
                    .then_some("crossing named-readout coordinates need a separate oracle")
                })
            })
        };
        if let Some(reason) = exclusion {
            eprintln!("Choi skip {}: {reason}", fixture.id());
            continue;
        }
        let shots = if fixture == TestFixture::ThreeBitAdder {
            8
        } else {
            4
        };
        let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .compile(&program)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.id()))
            .bloq;
        eprintln!("Choi check {}: {shots} shots", fixture.id());
        common::choi::assert_channel(&program, &bloq, shots);
        checked.push(fixture);
    }
    assert!(checked.contains(&TestFixture::ThreeBitAdder));
    assert!(checked.contains(&TestFixture::CNOT));
    eprintln!("Choi checked {} native fixtures", checked.len());
}
