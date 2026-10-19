#![cfg(test)]

//! Exact physical Choi checks against independently declared ideal circuits.
//!
//! The common runner Bell-pairs each logical input with a private reference and
//! compares the complete frame-corrected output/reference state on every shot.
//! Zero-input factories use the same runner as state preparations. Branch and
//! frame witnesses remain fixture-specific; detector-only and consumed-output
//! record checks below keep their separate contracts.
//!
//! This verifies normalized sampled branches, not their probabilities. Large
//! stabilizer instruments use the scalable source-correlation Choi runner.
//! Both CCZ factories run via `just fidelity` / `--run-ignored all`.

use std::collections::BTreeSet;

use bloq_compile::{Bloq, CompileConfig, CompileContext};
use bloq_graph::{
    Action, Block, BlockKind, BranchArm, CubeKind, Direction, Expr, GalleryItem, MeasureTarget,
    Pipe,
};
use bloq_vm::run_bloq;
use glam::ivec3;

mod common;
use common::choi::exact::ExactChoi;

/// Seeds and shots for the small-interface Choi checks. Three seeds
/// give ample byproduct variation at `d=3` while keeping the suite in the CI
/// budget; the post-selecting CCZ factories override the shot count below.
const SEEDS: [u64; 3] = [0x11, 0x22, 0x33];
const SHOTS: usize = 6;

// ==============================================================================
// Compilation
// ==============================================================================

fn compile_gallery(item: GalleryItem) -> Bloq {
    CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&item.build())
        .map(|artifacts| artifacts.bloq)
        .unwrap_or_else(|e| panic!("{item:?}: failed to compile: {e:?}"))
}

fn compile_closed_gallery(item: GalleryItem) -> Bloq {
    let graph = item.build().flatten().unwrap();
    let graph = if graph.is_open() {
        graph
            .filled_graphs()
            .unwrap_or_else(|e| panic!("{item:?}: filled_graphs failed: {e:?}"))
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("{item:?}: no closed variant"))
    } else {
        graph
    };
    CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .map(|artifacts| artifacts.bloq)
        .unwrap_or_else(|e| panic!("{item:?}: closed variant failed to compile: {e:?}"))
}

/// Independent declared ideal maps, matching the gallery's QuiZX contracts.
/// Bell inputs test the complete unitary; zero-input cases test preparation.
fn assert_gallery_choi(item: GalleryItem, byproduct_varies: bool, both_arms: bool) {
    let bloq = compile_gallery(item);
    let mut inputs = bloq
        .logical_inputs()
        .iter()
        .map(|input| input.port)
        .collect::<Vec<_>>();
    let mut outputs = bloq
        .logical_outputs()
        .iter()
        .map(|output| output.port)
        .collect::<Vec<_>>();
    if item == GalleryItem::ToffoliFromAndDelayedCZ {
        // The source declares x,y,z; coordinate order is x,z,y.
        inputs = vec![ivec3(4, 0, 0), ivec3(4, 2, 0), ivec3(4, 1, 2)];
        outputs = vec![ivec3(4, 0, 2), ivec3(4, 2, 2), ivec3(4, 1, 4)];
    }
    let arity = match item {
        GalleryItem::CNOT | GalleryItem::CZTemporalH => (2, 2),
        GalleryItem::BellState => (0, 2),
        GalleryItem::GHZ | GalleryItem::GHZPatchRotations | GalleryItem::GHZSlideThenGlide => {
            (0, 4)
        }
        GalleryItem::ToffoliFromAndDelayedCZ => (3, 3),
        _ => (1, 1),
    };
    assert_eq!(
        (inputs.len(), outputs.len()),
        arity,
        "{item:?}: declared interface"
    );
    let oracle = ExactChoi::new(&inputs, &outputs, |sim| {
        match item {
            GalleryItem::CNOT => {
                sim.cx(0, 1)?;
            }
            GalleryItem::CZTemporalH => {
                sim.h(1);
                sim.cx(0, 1)?;
                sim.h(1);
            }
            GalleryItem::S => sim.s(0),
            GalleryItem::T | GalleryItem::TWithPreparedY => {
                sim.t(0)?;
            }
            GalleryItem::THTH => {
                sim.t(0)?;
                sim.h(0);
                sim.t(0)?;
                sim.h(0);
            }
            GalleryItem::BellState => {
                sim.h(0);
                sim.cx(0, 1)?;
            }
            GalleryItem::GHZ | GalleryItem::GHZPatchRotations | GalleryItem::GHZSlideThenGlide => {
                sim.h(0);
                for qubit in 1..4 {
                    sim.cx(0, qubit)?;
                }
                if matches!(
                    item,
                    GalleryItem::GHZPatchRotations | GalleryItem::GHZSlideThenGlide
                ) {
                    for qubit in 0..4 {
                        sim.h(qubit);
                    }
                }
            }
            GalleryItem::MoveRotation => {}
            GalleryItem::ToffoliFromAndDelayedCZ => {
                sim.h(2);
                sim.ccz(0, 1, 2)?;
                sim.h(2);
            }
            _ => panic!("no independent ideal map declared for {item:?}"),
        }
        Ok(())
    });
    let mut frame_combos = BTreeSet::new();
    let mut arms = BTreeSet::new();
    for seed in SEEDS {
        let report = oracle.run(&bloq, SHOTS, seed, |_, ctx| {
            frame_combos.insert(
                ctx.frames
                    .iter()
                    .map(|frame| {
                        (
                            frame.x.expect("evaluable X frame"),
                            frame.z.expect("evaluable Z frame"),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(())
        });
        arms.extend(report.branch_selectors);
    }
    if byproduct_varies {
        assert!(frame_combos.len() >= 2, "{item:?}: byproduct never varied");
    }
    if both_arms {
        assert_eq!(
            arms,
            BTreeSet::from([false, true]),
            "{item:?}: both branch arms must run"
        );
    }
}

// ==============================================================================
// Small unitary maps and state preparations
// ==============================================================================

#[test]
fn cnot_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::CNOT, true, false);
}

#[test]
fn ghz_slide_then_glide_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::GHZSlideThenGlide, true, false);
}

#[test]
fn cz_temporal_h_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::CZTemporalH, false, false);
}

#[test]
fn s_gate_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::S, false, false);
}

#[test]
fn signed_structural_projections_preserve_s_phase() {
    let compile = |value| {
        let mut graph = GalleryItem::S.build().flatten().unwrap();
        let past = ivec3(1, 0, 1);
        let target = ivec3(1, 0, 2);
        let control = ivec3(3, 0, 0);
        graph.remove_block(target).unwrap();
        graph.add_block(Block::new(control, BlockKind::Cube(CubeKind::ZXZ)));
        let arm = || {
            BranchArm::new(
                vec![Block::new(target, BlockKind::Y)],
                vec![Pipe::new(past, Direction::ZPLUS)],
            )
        };
        let branch = graph.try_add_branch_region("b", arm(), arm()).unwrap();
        let mut actions = graph.actions();
        actions.extend([
            Action::Measure {
                target: MeasureTarget::Node(control),
                name: "m".into(),
            },
            Action::Branch {
                target: branch,
                condition: if value {
                    Expr::Not(Box::new(Expr::Var("m".into())))
                } else {
                    Expr::Var("m".into())
                },
            },
        ]);
        graph.set_actions(actions).unwrap();
        CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
            .compile(&graph)
            .unwrap()
            .bloq
    };
    for value in [false, true] {
        let bloq = compile(value);
        let oracle = ExactChoi::new(&[ivec3(0, 0, 0)], &[ivec3(0, 0, 2)], |sim| {
            sim.s(0);
            Ok(())
        });
        let report = oracle.run(&bloq, SHOTS, SEEDS[0], |_, _| Ok(()));
        assert!(
            report
                .branch_selectors
                .iter()
                .all(|&actual| actual == value)
        );
    }
}

#[test]
fn bell_state_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::BellState, true, false);
}

#[test]
fn ghz_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::GHZ, true, false);
}

#[test]
fn ghz_patch_rotations_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::GHZPatchRotations, true, false);
}

#[test]
fn move_rotation_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::MoveRotation, true, false);
}

#[test]
fn t_gate_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::T, false, true);
}

#[test]
fn thth_preserves_the_complete_choi_state() {
    assert_gallery_choi(GalleryItem::THTH, true, true);
}

#[test]
fn t_with_prepared_y_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::TWithPreparedY, false, true);
}

#[test]
fn toffoli_preserves_choi_state() {
    assert_gallery_choi(GalleryItem::ToffoliFromAndDelayedCZ, true, true);
}

// ==============================================================================
// PhaseGradientK4: consumed magic — frame-evaluability + combo coverage only
// ==============================================================================

/// PhaseGradientK4's phase-gradient worldline is measured out (0 output
/// logicals, one frame pair), so there is no output state to score. The one
/// physically observable property is that the output frame stays *evaluable*
/// across the many fill assignments the seeded batch exercises. This is
/// the ONLY frame coverage PGK4 has now that the compile-time P2c differential
/// is deleted: it samples assignments by seed (no selector-forcing hook exists),
/// asserts every X/Z frame bit is `Some` on every shot, and requires the frame
/// bits to have genuinely varied across the sample so the evaluability check is
/// not vacuous.
#[test]
fn phase_gradient_k4_frame_stays_evaluable() {
    let bloq = CompileContext::new(CompileConfig::default().with_prepare_t_with_mpps(true))
        .compile(&GalleryItem::PhaseGradientK4.build())
        .expect("PhaseGradientK4 module compiles")
        .bloq;
    let frames = bloq.output_frames();
    assert_eq!(
        frames.len(),
        1,
        "phase_gradient_k4: expected one output frame"
    );

    // Sample PGK4's 22 selector bits: eight rank-64 shots across three seeds
    // cover 20+ patterns and several terminal byproduct classes.
    let seeds: [u64; 3] = [0x11, 0x22, 0x33];
    let shots = 8usize;

    let mut frame_bit_classes: BTreeSet<(bool, bool)> = BTreeSet::new();
    for &seed in &seeds {
        let report = run_bloq(&bloq, shots, seed).expect("run_bloq");
        let fp = &report.frame_pairs[0];
        assert_eq!(fp.x_bits.len(), shots, "every shot produced a frame bit");
        for s in 0..shots {
            let x = fp.x_bits[s].unwrap_or_else(|| {
                panic!("phase_gradient_k4: X frame bit unevaluable (seed {seed} shot {s})")
            });
            let z = fp.z_bits[s].unwrap_or_else(|| {
                panic!("phase_gradient_k4: Z frame bit unevaluable (seed {seed} shot {s})")
            });
            frame_bit_classes.insert((x, z));
        }
    }
    assert!(
        frame_bit_classes.len() >= 2,
        "phase_gradient_k4: frame bits never varied across the seeded fill \
         sample — the evaluability check is vacuous (saw {frame_bit_classes:?})",
    );
}

// ==============================================================================
// CCZ factories: absolute magic-state target
// ==============================================================================

/// Assert each frame-corrected logical state equals the ideal `CCZ|+++⟩`.
fn assert_ccz_recovers_plus(item: GalleryItem, batches: &[(usize, u64)]) {
    let bloq = compile_gallery(item);
    let outputs = bloq
        .logical_outputs()
        .iter()
        .map(|output| output.port)
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 3, "{item:?}: three CCZ outputs");
    let oracle = ExactChoi::new(&[], &outputs, |sim| {
        for qubit in 0..3 {
            sim.h(qubit);
        }
        sim.ccz(0, 1, 2)?;
        Ok(())
    });
    for &(shots, seed) in batches {
        let report = oracle.run(&bloq, shots, seed, |_, _| Ok(()));
        // Eight T factories share measurements at seams. Preserve the aliasing
        // regression: no MissingRecord, constant and non-vacuous detectors.
        assert!(
            report.all_detectors_constant(),
            "{item:?}: varying detector"
        );
        assert!(
            report
                .detectors
                .iter()
                .any(|detector| !detector.per_shot.is_empty()),
            "{item:?}: no detector was evaluated"
        );
    }
}

#[test]
#[ignore = "heavy physical fidelity (post-selecting CCZ factory, ~2s/shot \
            debug); passes since D5 resolved ledger §8.1 — run via \
            `just fidelity` / --run-ignored all"]
fn ccz_4x3x6_frame_recovers_plus() {
    // Preserve the separate aliasing regression's seed alongside fidelity shots.
    assert_ccz_recovers_plus(GalleryItem::CCZ4x3_6, &[(12, 0x21), (2, 0x9)]);
}

#[test]
#[ignore = "heavy physical fidelity (post-selecting CCZ factory with \
            teleports, ~2s/shot debug); passes since D5 resolved ledger §8.1 — \
            run via `just fidelity` / --run-ignored all"]
fn ccz_4x3x7_tels_frame_recovers_plus() {
    assert_ccz_recovers_plus(GalleryItem::CCZFactoryWithTels, &[(12, 0x21)]);
}

// ==============================================================================
// Closed programs: no output worldlines, so the score is the observables
// themselves
// ==============================================================================

/// The compilable programs whose compiled form terminates no output worldline
/// (spatial-port graphs close via `fill_ports_auto`; the memory/stability
/// experiments measure everything out).
const CLOSED_PROGRAMS: [GalleryItem; 6] = [
    GalleryItem::ThreeCNOTs,
    GalleryItem::SteaneEncoding,
    GalleryItem::XMemory,
    GalleryItem::YMemory,
    GalleryItem::Stability,
    // CZ is the spatial-Hadamard entry: both its `±Y` ports close under
    // `fill_ports_auto`, and so does the temporal worldline through the `XZX`
    // cube, leaving the two crossing rows as the whole readable content.
    GalleryItem::CZSpatialH,
];

/// Every logical observable of a *closed* program is a ZX stabilizer generator
/// of a state with no free worldline, so noiselessly its record parity is
/// pinned — deterministically `false` every shot. Unlike the recovered-state
/// checks above (which score the frame and deliberately leave the absolute
/// logical value to `verify_logical`), this reads the compiled parity itself,
/// so a misrouted observable shows up as a shot-random or constant-`true` row.
///
/// This is the compilation-correctness arbiter for the spatial Hadamard wall:
/// CZ's two rows both cross the domain wall, and it is exactly the wall
/// gateway's records that turn them from shot-random into constant.
#[test]
fn closed_programs_have_deterministic_observables() {
    for item in CLOSED_PROGRAMS {
        let bloq = compile_closed_gallery(item);
        assert!(
            bloq.output_frames().is_empty(),
            "{item:?}: expected no output frames (closed program)",
        );
        let report = run_bloq(&bloq, SHOTS, SEEDS[0]).expect("run_bloq");
        assert!(
            !report.observables.is_empty(),
            "{item:?}: a closed program has at least one observable row",
        );
        for (index, observable) in report.observables.iter().enumerate() {
            assert!(
                observable.per_shot.iter().all(|&value| !value),
                "{item:?}: observable {index} is not deterministically false                  ({:?}) — its compiled record parity does not reproduce the                  stabilizer row",
                observable.per_shot,
            );
        }
    }
}

#[test]
fn exact_choi_rejects_wrong_phase_and_uncorrected_byproduct() {
    common::choi::exact::assert_wrong_phase_and_uncorrected_byproduct_rejected();
}
