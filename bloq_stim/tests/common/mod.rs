//! Shared fixtures, noise model, and distance oracle for Stim integration tests.

#![allow(dead_code, reason = "shared by independent integration-test binaries")]

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::{Bloq, BloqEdge, BloqNodeId};
#[cfg(feature = "verify")]
use stim::{Circuit, noise::UniformDepolarizing};

/// Two ZXZ cubes stacked in time: one temporal seam and one observable.
pub(crate) fn compile_two_cubes(distance: u32) -> Bloq {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n  1: ZXZ [0,0,1]\n  [0,0,0] -> +Z\n",
    )
    .expect("two-cube memory graph parses");
    CompileContext::new(CompileConfig::new(distance))
        .compile(&graph)
        .expect("two-cube graph compiles")
        .bloq
}

/// The two-cube program's single quantum edge, carrying exactly one pipe.
pub(crate) fn quantum_seam(bloq: &Bloq) -> (BloqNodeId, BloqNodeId) {
    let mut seams = bloq.edges().filter_map(|edge| {
        let BloqEdge::Quantum(quantum) = edge.edge else {
            return None;
        };
        let [_seam] = quantum.pipes.as_slice() else {
            panic!("two-cube seam carries exactly one pipe");
        };
        Some((edge.source, edge.target))
    });
    let seam = seams.next().expect("two-cube program has a Quantum seam");
    assert!(seams.next().is_none(), "exactly one Quantum seam expected");
    seam
}

#[cfg(feature = "verify")]
pub(crate) fn noisy_graphlike_distance(bloq: &Bloq) -> usize {
    let noisy: Circuit = bloq_stim::emit_bloq_stim_with_stim_noise(bloq, &uniform_depolarizing())
        .expect("emit noisy Stim")
        .parse()
        .expect("parse noisy Stim");
    graphlike_distance(&noisy).expect("detectors are deterministic and graphlike")
}

/// The physical error rate every distance check in this suite assumes.
#[cfg(feature = "verify")]
pub(crate) fn uniform_depolarizing() -> UniformDepolarizing {
    UniformDepolarizing::new(1e-3).expect("1e-3 is a valid probability")
}

/// The same graphlike search as `Circuit::shortest_graphlike_error`, without
/// reconstructing physical error locations that distance checks never inspect.
#[cfg(feature = "verify")]
pub(crate) fn graphlike_distance(circuit: &Circuit) -> stim::Result<usize> {
    // Match the circuit helper's DEM options, including disjoint-error handling.
    circuit
        .detector_error_model_with_options(false, false, false, 1.0, false, false)?
        .shortest_graphlike_error(true)
        .map(|error| error.len())
}
