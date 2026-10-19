//! Causal controls must preserve the data waiting for their selected measurement.

use bloq_compile::{CompileConfig, CompileContext, CompileError};
use bloq_graph::{
    BlockGraph,
    verify::{BoundaryOrder, LogicalVerifier},
};
use glam::ivec3;
use quizx::circuit::Circuit;

// Unchanged geometry cut from Topologiq's adder: CX(15,14) followed by T(17),
// upstream d010bd3e63686c93c12c922a84a944a3386ac785. The adaptive cap at z4
// needs a syndrome from a z7 component that resets the cap's live data.
const LATE_CONTROL: &str = include_str!("data/late_control_t.blog");

#[test]
fn a_logically_valid_t_channel_with_a_destructive_late_control_is_rejected() {
    let graph = BlockGraph::from_blog_text(LATE_CONTROL).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.add_gate("cx", vec![0, 1]);
    circuit.add_gate("t", vec![1]);
    let expected = circuit.to_graph();
    LogicalVerifier::with_boundaries_and_internal_measurements(
        &graph,
        BoundaryOrder::new(
            vec![ivec3(2, 0, 6), ivec3(0, -1, 3)],
            vec![ivec3(2, 0, 8), ivec3(-1, -1, 1)],
        ),
    )
    .unwrap()
    .verify(&expected, 16, 7)
    .unwrap();

    let error = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .unwrap_err();
    assert!(
        matches!(error, CompileError::LiveQubitReuse { .. }),
        "{error}"
    );
}

#[test]
fn the_same_geometry_with_a_fixed_x_cap_remains_compilable() {
    let source = LATE_CONTROL
        .replace("4: T [", "4: XZX [")
        .replace("5: XY [", "5: X [")
        .replace("  m = measure 0\n", "")
        .replace("  resolve 5 if !m\n", "")
        .replace("  feedback X 2 -> -X if m\n", "");
    let graph = BlockGraph::from_blog_text(&source).unwrap();
    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&graph)
        .unwrap()
        .bloq;
    bloq.validate().unwrap();
}
