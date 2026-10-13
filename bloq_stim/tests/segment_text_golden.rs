//! Byte-compatibility gate for segmented Stim emission.
//!
//! Sinter strong ids depend on the emitted circuit text, so a silent change
//! can invalidate downstream data identifiers. These snapshots pin the
//! reassembled clean and noisy text of a fixed fixture; restructuring the
//! segment *container* must leave them untouched.

use bloq_circuit::NoiseModel;
use bloq_stim::{
    BloqStimOptions, emit_bloq_stim_segments, emit_bloq_stim_segments_pair,
    emit_bloq_stim_segments_with,
};

mod common;
use common::compile_two_cubes;

const DISTANCE: u32 = 3;
const PHYSICAL_ERROR_PROBABILITY: f64 = 1e-3;

#[test]
fn clean_segment_text_is_unchanged() {
    let segments = emit_bloq_stim_segments(&compile_two_cubes(DISTANCE)).expect("segments emit");

    insta::assert_snapshot!("clean_segments", segments.to_text());
}

/// The two options are independent knobs on one emission, which is the whole
/// reason they share a type: the per-function spelling could not express
/// "noisy *and* checked" at all.
#[test]
fn noise_and_trust_compose_on_one_emission() {
    let program = compile_two_cubes(DISTANCE);
    let noise = NoiseModel::uniform_depolarizing(PHYSICAL_ERROR_PROBABILITY);
    let checked_noisy = BloqStimOptions::new()
        .with_noise(&noise)
        .with_trust(bloq_stim::InputTrust::Checked);

    assert_eq!(
        emit_bloq_stim_segments_with(&program, &checked_noisy)
            .expect("checked noisy emits")
            .to_text(),
        emit_bloq_stim_segments_with(&program, &BloqStimOptions::new().with_noise(&noise))
            .expect("trusted noisy emits")
            .to_text(),
    );
    assert_eq!(
        emit_bloq_stim_segments(&program).unwrap().to_text(),
        emit_bloq_stim_segments_with(&program, &BloqStimOptions::new())
            .unwrap()
            .to_text(),
    );
}

#[test]
fn noisy_segment_text_is_unchanged() {
    let noise = NoiseModel::uniform_depolarizing(PHYSICAL_ERROR_PROBABILITY);
    let segments = emit_bloq_stim_segments_with(
        &compile_two_cubes(DISTANCE),
        &BloqStimOptions::new().with_noise(&noise),
    )
    .expect("segments emit");

    insta::assert_snapshot!("noisy_segments", segments.to_text());
}

/// The pair entry point exists so callers stop emitting twice and asserting
/// alignment afterwards; it must produce exactly what the two single emissions
/// do.
#[test]
fn segment_pair_matches_the_two_single_emissions() {
    let program = compile_two_cubes(DISTANCE);
    let noise = NoiseModel::uniform_depolarizing(PHYSICAL_ERROR_PROBABILITY);

    let (clean, noisy) = emit_bloq_stim_segments_pair(&program, &noise).expect("pair emits");

    assert_eq!(
        clean.to_text(),
        emit_bloq_stim_segments(&program).unwrap().to_text()
    );
    assert_eq!(
        noisy.to_text(),
        emit_bloq_stim_segments_with(&program, &BloqStimOptions::new().with_noise(&noise))
            .unwrap()
            .to_text()
    );
}

/// Each chunk's declared measurement span must match the records its own text
/// actually emits — that is the number consumers used to recover by reparsing
/// every segment. Stim itself is the arbiter of "how many records is that",
/// hence the `verify` gate.
#[cfg(feature = "verify")]
#[test]
fn segment_measurement_spans_tile_the_record_stream() {
    use stim::Circuit;

    let segments = emit_bloq_stim_segments(&compile_two_cubes(DISTANCE)).expect("segments emit");

    let mut expected_start = 0;
    for segment in &segments.segments {
        assert_eq!(segment.measurement_start, expected_start);
        let emitted: Circuit = segment.text.parse().expect("segment text parses as Stim");
        assert_eq!(
            segment.measurement_count as u64,
            emitted.num_measurements(),
            "node {:?} span disagrees with its own text",
            segment.node_id
        );
        expected_start += segment.measurement_count;
    }
    assert_eq!(segments.num_measurements(), expected_start);
    let whole: Circuit = segments.to_text().parse().expect("reassembly parses");
    assert_eq!(whole.num_measurements(), expected_start as u64);
}
