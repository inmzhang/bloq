#![cfg(test)]

//! CCZ factory output verification.
//! Embedded T resources use ideal logical preparation; T cultivation is
//! verified separately in `verify_t_block`.
//!
//! A CCZ factory (eight embedded `T` blocks) outputs three logical qubits at
//! port-terminated worldlines whose joint state is `CCZ|+++⟩`, carrying a
//! per-shot Pauli byproduct. This file verifies membership in the **byproduct
//! orbit** *without* consulting the compiler's terminal frame — a frame-blind
//! oracle, alongside the common exact Choi state-preparation check.
//!
//! Per surviving shot we read the joint logical Pauli signature directly and
//! compare it with the 64-state `P · CCZ|+++⟩` byproduct orbit. This determines
//! the reduced state exactly, without ancillas, measurement, or uncomputation.
//! A state off the orbit fails; multiple distinct members must occur across
//! shots so the byproduct coverage is nonvacuous.
//!
//! The common Choi runner also pins the frame-corrected state to `CCZ|+++⟩`.
//! Larger seeded batches and the teleported factory live in the ignored
//! `verify_exact_fidelity.rs` checks.

use std::collections::HashSet;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_test::{CompileReadyCase, TestFixture, select_test_cases_for_fixture};
use bloq_vm::verify::{ccz_orbit_signatures, logical_signature, signatures_match};

mod common;
use common::choi::exact::ExactChoi;

/// Compile `fixture` at d=3, run `shots` shots, and assert every surviving shot's
/// three-output state lands exactly on the
/// `CCZ|+++⟩` byproduct orbit, with the byproduct varying across survivors.
fn assert_ccz_factory_on_orbit(fixture: TestFixture, shots: usize, seed: u64) {
    let test_case = select_test_cases_for_fixture(fixture)
        .expect("fixture cases")
        .into_iter()
        .next()
        .expect("a fixture case");
    let name = test_case.id().to_string();
    let case = CompileReadyCase::from_test_case(test_case).expect("case builds");
    let ctx = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true));
    let bloq = ctx.compile(&case.graph).expect("factory compiles").bloq;

    let ports = bloq
        .logical_outputs()
        .iter()
        .map(|output| output.port)
        .collect::<Vec<_>>();
    assert_eq!(ports.len(), 3, "three CCZ outputs");
    let oracle = ExactChoi::new(&[], &ports, |sim| {
        for qubit in 0..3 {
            sim.h(qubit);
        }
        sim.ccz(0, 1, 2)?;
        Ok(())
    });

    // The 64-member byproduct orbit is fixture-independent.
    let orbit = ccz_orbit_signatures();

    let mut matched: Vec<usize> = Vec::new();
    let report = oracle.run(&bloq, shots, seed, |sim, ctx| {
        assert_eq!(
            ctx.outputs.len(),
            3,
            "{name}: expected 3 CCZ output logicals, found {}",
            ctx.outputs.len(),
        );
        let outputs: Vec<_> = ctx
            .outputs
            .iter()
            .map(|out| (out, (false, false)))
            .collect();
        let signature = logical_signature(sim, &outputs)?;
        // The reduced 3-qubit state is PURE: for a pure n-qubit state the sum of
        // squared Pauli expectations (including ⟨I⟩ = 1) is 2ⁿ. This rules out
        // "outputs entangled with factory residue" (a mixed reduced state) before
        // the orbit match rules out incorrect magic content — the two together pin
        // the result to a genuine orbit member.
        let purity_sum: f64 = 1.0 + signature.iter().map(|p| p * p).sum::<f64>();
        assert!(
            (purity_sum - 8.0).abs() < 1e-9,
            "{name}: reduced 3-qubit logical state is not pure (Σ⟨P⟩² = {purity_sum}, want 8) \
             — outputs entangled with factory residue?",
        );
        let member = orbit
            .iter()
            .position(|candidate| signatures_match(candidate, &signature));
        match member {
            Some(index) => matched.push(index),
            None => panic!(
                "{name}: CCZ output is off the CCZ|+++> byproduct orbit \
                 (not a magic-state orbit member — residual non-Clifford content?)"
            ),
        }
        Ok(())
    });

    // Noiselessly the `discard if` check bit is deterministically 0, so every
    // shot survives; the `>= 2` bound is what the varied-byproduct check needs.
    assert!(
        matched.len() >= 2,
        "{name}: need >=2 surviving shots, saw {} (of {shots})",
        matched.len(),
    );
    // Byproduct genuinely varies across shots (not vacuously always the same
    // orbit member).
    let distinct: HashSet<usize> = matched.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "{name}: byproduct never varied across {} survivors (matched member {:?})",
        matched.len(),
        distinct,
    );
    // The compiled factory's detectors are constant across shots (existing
    // report contract).
    assert!(
        report.all_detectors_constant(),
        "{name}: a detector varied across shots",
    );
}

#[test]
fn ccz_4x3x6_output_on_byproduct_orbit() {
    assert_ccz_factory_on_orbit(TestFixture::CCZ4x3_6, 8, 0x21);
}
