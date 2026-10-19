//! Later source layers may supply a control when the held data remains intact.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{
    BlockGraph,
    verify::{BoundaryOrder, LogicalVerifier},
};
use glam::ivec3;
use quizx::circuit::Circuit;

mod common;
use common::choi::exact::{BellOutcomes, ExactChoi};

#[rstest::rstest]
#[case(false)]
#[case(true)]
fn a_t_cap_waits_for_readout_support_above_its_source_layer(#[case] dagger: bool) {
    let source = include_str!("data/delayed_control_t.blog");
    let source = if dagger {
        source
            .replace("resolve 4 if !m", "resolve 4 if m")
            .replace("  feedback Z 1 -> -Y if m\n", "")
    } else {
        source.to_owned()
    };
    let graph = BlockGraph::from_blog_text(&source).unwrap();
    let inputs = [ivec3(0, 0, 4)];
    let outputs = [ivec3(2, 0, 3)];
    let cap = ivec3(1, -1, 2);
    let readouts = graph.stabilizers().unwrap();
    let readout = readouts
        .iter()
        .find(|row| row.measurement_name() == Some("m"))
        .unwrap();
    assert!(readout.stabilizer.port_stabilizer.contains_key(&inputs[0]));
    assert!(
        inputs[0].z > cap.z,
        "the control surface reaches above the cap"
    );
    let mut circuit = Circuit::new(1);
    circuit.add_gate(if dagger { "tdg" } else { "t" }, vec![0]);
    let expected = circuit.to_graph();
    LogicalVerifier::with_boundaries_and_internal_measurements(
        &graph,
        BoundaryOrder::new(inputs.to_vec(), outputs.to_vec()),
    )
    .unwrap()
    .verify(&expected, 16, 7)
    .unwrap();
    let mut bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .unwrap()
        .bloq;
    let bell = BellOutcomes::expose(&mut bloq);
    bloq.validate().unwrap();
    assert!(
        bloq.has_path(
            bloq.node_by_block(inputs[0]).unwrap(),
            bloq.node_by_block(cap).unwrap(),
        ),
        "the source-later readout is scheduled before the adaptive cap"
    );
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        if dagger {
            sim.t_dag(0)?;
        } else {
            sim.t(0)?;
        }
        Ok(())
    });
    let mut branches = std::collections::BTreeSet::new();
    let mut sectors = std::collections::BTreeSet::new();
    for seed in [17, 34, 51] {
        let report = oracle.run(&bloq, 16, seed, |_, context| {
            sectors.insert(bell.read(context));
            branches.extend(
                context
                    .named_branch_selectors
                    .iter()
                    .filter_map(|(name, value)| name.starts_with("selective ").then_some(*value)),
            );
            Ok(())
        });
        assert!(report.all_detectors_constant());
        assert_eq!(report.discarded, 0);
    }
    assert_eq!(branches, std::collections::BTreeSet::from([false, true]));
    assert_eq!(sectors.len(), 4);
}
