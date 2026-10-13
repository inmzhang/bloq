//! Flatten acceptance: `Bloq::flatten` unrolls every `REPEAT` block and
//! rewrites detector side tables so the emitted circuit keeps the same
//! observables, detector count, and graphlike distance as the looped program.
//!
//! The base two-cube compile unrolls its rounds into per-layer instances, so
//! the looped fixture is a decoder-wait splice: its padding node instantiates
//! the recorded looped template (`first round + REPEAT r`), giving the program
//! a genuine repeat body with loop-carried detector state.

use bloq_ir::{Bloq, LevelPath, MemoryRoundTarget};
use bloq_stim::emit_bloq_stim;
use stim::Circuit;

mod common;
use common::{compile_two_cubes, noisy_graphlike_distance, quantum_seam};

const DISTANCE: u32 = 3;
/// Enough wait rounds that the emitter keeps a `REPEAT` block instead of
/// unrolling the tail itself.
const WAIT_ROUNDS: u32 = 5;

/// Two ZXZ cubes stacked in time, with a decoder wait spliced into the seam.
fn compile_waiting_two_cubes() -> Bloq {
    let mut bloq = compile_two_cubes(DISTANCE);
    let (from, to) = quantum_seam(&bloq);
    bloq.insert_memory_rounds(
        MemoryRoundTarget::Edge {
            path: LevelPath::default(),
            from,
            to,
        },
        WAIT_ROUNDS,
    )
    .expect("decoder-wait insertion verifies and applies");
    bloq
}

#[test]
fn flatten_preserves_observables_detectors_and_distance() {
    let mut bloq = compile_waiting_two_cubes();
    let baseline_text = emit_bloq_stim(&bloq).expect("emit looped Stim");
    assert!(
        baseline_text.contains("REPEAT"),
        "test is vacuous unless the looped program actually repeats"
    );
    let baseline: Circuit = baseline_text.parse().expect("parse looped Stim");
    let baseline_distance = noisy_graphlike_distance(&bloq);
    assert_eq!(baseline_distance, DISTANCE as usize);

    bloq.flatten().expect("program flattens");
    bloq.validate().expect("flattened program validates");

    let flat_text = emit_bloq_stim(&bloq).expect("emit flattened Stim");
    assert!(
        !flat_text.contains("REPEAT"),
        "flattened program must emit no REPEAT blocks"
    );
    let flat: Circuit = flat_text.parse().expect("parse flattened Stim");
    assert_eq!(flat.num_observables(), baseline.num_observables());
    assert_eq!(flat.num_detectors(), baseline.num_detectors());
    assert_eq!(
        noisy_graphlike_distance(&bloq),
        baseline_distance,
        "flattening must not change the graphlike distance"
    );
}
