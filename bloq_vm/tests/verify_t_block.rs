#![cfg(test)]

//! Isolated |T⟩-factory verification.
//! These tests intentionally retain cultivation and escape circuits: their
//! subject is the factory output and its retry-region boundary.
//!
//! The gallery `t_gate` graph is a *gate teleportation*: its magic worldline is
//! consumed by the lattice-surgery merge, so at end-of-shot nothing live remains
//! to score against |T⟩ (the end-to-end logical op is the trivial `T|0⟩ = |0⟩`).
//! To verify the factory itself we compile a standalone graph carrying ONLY the
//! magic worldline — a `T` cube feeding an output `Port` through a temporal pipe
//! — so the output port terminates the live escaped |T⟩ patch. Its output
//! logical operators let the simulator read exact expectations directly, and we
//! score the logical state against |T⟩.
//!
//! # Orbit invariant (this file)
//!
//! The escaped magic is a perfect |T⟩ *up to a Pauli byproduct* fixed by the
//! cultivation/escape measurement record (its terminal frame). A Pauli maps |T⟩
//! within the four-state orbit {|T⟩, X|T⟩, Y|T⟩, Z|T⟩}, all of which share the
//! frame-independent signature `⟨Z⟩ = 0`, `|⟨X⟩| = |⟨Y⟩| = 1/√2`. Asserting that
//! signature verifies the factory outputs a perfect magic state modulo the Pauli
//! frame — without needing the (compiler-side) record→frame binding that exact
//! fidelity requires.
//!
//! # Exact fidelity (this file)
//!
//! That binding now exists: the compiler emits the port's terminal Pauli frame
//! from the escaped patch's tracked X_L/Z_L observables' raw record parity
//! through `registry/readouts.rs`. Cultivation directly delivers the `|T⟩`
//! orbit in bloq's positive logical-Z frame, so no fixed conversion bit
//! is needed. Applying the frame-predicted byproduct recovers a perfect |T⟩
//! every shot, tightening the orbit invariant to `F = 1`.

use std::f64::consts::FRAC_1_SQRT_2;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_ir::{BloqEdge, MemoryRoundTarget};
use bloq_vm::verify::logical_bloch;

mod common;
use common::choi::exact::ExactChoi;

/// A standalone magic factory: `T` cube at the origin, output `Port` one step up
/// the temporal axis, joined by a `+Z` pipe. No data worldline, no merge — so
/// the escaped |T⟩ is still live at the output port.
fn isolated_factory() -> BlockGraph {
    BlockGraph::from_blog_text(
        "BLOG 1.0\n\
         \n\
         \x20 0: T [0, 0, 0]\n\
         \x20 1: Port [0, 0, 1]\n\
         \x20 [0, 0, 0] -> +Z\n",
    )
    .expect("isolated factory blog parses")
    .fix_shadowed_faces()
}

/// Read frames and logical expectations in one pass, checking every byproduct
/// corner and its exact corrected fidelity.
fn assert_exact_fidelity(distance: u32) {
    const TOL: f64 = 1e-9;
    let graph = isolated_factory();
    let ctx = CompileContext::new(CompileConfig::new(distance).with_prepare_t_with_mpps(false));
    let bloq = ctx
        .compile(&graph)
        .unwrap_or_else(|e| panic!("isolated factory failed to compile at d={distance}: {e:?}"))
        .bloq;

    let oracle = ExactChoi::new(&[], &[glam::IVec3::Z], |sim| {
        sim.h(0);
        sim.t(0)?;
        Ok(())
    });
    let mut classes: std::collections::HashSet<(bool, bool)> = std::collections::HashSet::new();
    // Cover the union of the former orbit and fidelity batches without rerunning
    // their shared first eight shots at seed 0x11.
    for (seed, shots) in [(0x11, 12), (0x22, 8), (0x33, 8)] {
        let mut scored = 0;
        let mut signs = std::collections::HashSet::new();
        let report = oracle.run(&bloq, shots, seed, |sim, sctx| {
            let out = sctx
                .outputs
                .first()
                .expect("output port publishes logicals");
            assert_eq!(sctx.frames.len(), 1, "d={distance}: one output frame");
            let frame = &sctx.frames[0];
            let (x_bit, z_bit) = (
                frame.x.expect("evaluable X frame"),
                frame.z.expect("evaluable Z frame"),
            );
            classes.insert((x_bit, z_bit));
            let shot = sctx.shot;

            // Orbit-corner cross-check: the uncorrected ⟨X⟩/⟨Y⟩ signs place the
            // logical state on one corner of the |T⟩ orbit, and the frame bits predict
            // exactly that corner — `⟨X⟩>0 ⟺ ¬z_bit`, `⟨Y⟩>0 ⟺ x_bit==z_bit`
            // (byproduct I,X,Y,Z map to (+,+),(+,−),(−,+),(−,−)).
            let (ex, ey, ez) = logical_bloch(sim, out, (false, false))?;
            signs.insert((ex > 0.0, ey > 0.0));
            // Keep the frame-independent orbit oracle alongside corrected fidelity.
            assert!(ez.abs() < TOL, "d={distance} shot {shot}: ⟨Z⟩={ez}");
            assert!((ex.abs() - FRAC_1_SQRT_2).abs() < TOL);
            assert!((ey.abs() - FRAC_1_SQRT_2).abs() < TOL);
            assert_eq!(
                ex > 0.0,
                !z_bit,
                "d={distance} seed={seed} shot {shot}: ⟨X⟩ sign disagrees with frame",
            );
            assert_eq!(
                ey > 0.0,
                x_bit == z_bit,
                "d={distance} seed={seed} shot {shot}: ⟨Y⟩ sign disagrees with frame",
            );

            scored += 1;
            Ok(())
        });
        assert_eq!(scored, shots, "d={distance}: hook ran every shot");
        assert!(signs.len() > 1, "d={distance}: byproduct never varied");
        assert!(
            report.all_detectors_constant(),
            "d={distance}: varying detector"
        );
        assert!(report.max_rank > 1, "d={distance}: expected genuine T rank");
    }

    assert_eq!(
        classes.len(),
        4,
        "d={distance}: expected all four byproduct classes across the batch, saw {classes:?}",
    );
}

#[test]
fn isolated_factory_t_exact_fidelity_d3() {
    assert_exact_fidelity(3);
}

#[test]
fn isolated_factory_t_exact_fidelity_d5() {
    assert_exact_fidelity(5);
}

#[test]
fn isolated_factory_t_exact_fidelity_d7() {
    assert_exact_fidelity(7);
}

#[test]
fn isolated_factory_supports_escape_latency() {
    let ctx = CompileContext::new(CompileConfig::new(5).with_prepare_t_with_mpps(false));
    let mut bloq = ctx
        .compile(&isolated_factory())
        .expect("isolated factory compiles")
        .bloq;
    assert_eq!(bloq.pipe_padding().count(), 1);

    let (path, escape) = bloq
        .levels()
        .find_map(|(path, level)| {
            (!path.is_top_level()).then(|| {
                level
                    .nodes()
                    .find(|(id, node)| {
                        node.try_quantum().is_some()
                            && !level
                                .outgoing(*id)
                                .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
                    })
                    .map(|(id, _)| (path, id))
            })?
        })
        .expect("RUS body has a terminal escape node");

    let padding = bloq
        .insert_memory_rounds(
            MemoryRoundTarget::After {
                path: path.clone(),
                node: escape,
            },
            10,
        )
        .expect("escape accepts decoder latency");
    bloq.validate().expect("padded factory validates");

    let body = bloq.level_at(&path).expect("RUS body remains present");
    assert_eq!(body[padding].memory_rounds(), Some(10));
    assert!(body.has_path(escape, padding));
}
