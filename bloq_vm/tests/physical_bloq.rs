#![cfg(test)]

//! Whole-program verification of a compiled bloq through `run_bloq`
//! (design spec §3, item 6), covering the structures the corpus sweep in
//! `verify_gallery_physical.rs` cannot reach on its own:
//!
//! * spatial-Hadamard walls — hand-built graphs (hub arm shapes, parallel
//!   walls, the rejected quarter turn) that are not registry fixtures;
//! * measurement-alias resolution across multi-instance lattice surgery;
//! * `T`-block region execution and the exact per-output Pauli frames, incl.
//!   selector-arm signs and non-linear resolve conditions;
//! * `Feedback` row semantics.
//!
//! The plain "compile a fixture, run it, assert detector constancy" tier lives
//! in `verify_gallery_physical.rs`, which sweeps native non-Clifford fixtures at
//! d3; keep additional distances and transformations in focused checks here.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe, UDirection};
use bloq_ir::{ClassicalNode, NodeProvenance};
use bloq_test::{CompileReadyCase, find_test_case};
use bloq_vm::run_bloq;
use bloq_vm::verify::logical_bloch;
use glam::IVec3;

mod common;
use common::choi::exact::ExactChoi;

fn compiled_case(name: &str, distance: u32) -> (CompileReadyCase, bloq_compile::Bloq) {
    let test_case = find_test_case(name).expect("case exists in the registry");
    let case = CompileReadyCase::from_test_case(test_case).expect("case builds");
    let ctx = CompileContext::new(CompileConfig::new(distance));
    let bloq = ctx
        .compile(&case.graph)
        .expect("graph compiles to a Bloq")
        .bloq;
    (case, bloq)
}

#[rstest::rstest]
#[case::plain_quarter_turn(false, 1)]
#[case::prepared_unrotated(true, 0)]
#[case::prepared_quarter_turn(true, 1)]
fn prepared_rotated_yoke_preserves_source_choi_correlations(
    #[case] prepared: bool,
    #[case] quarter_turns: i32,
) {
    use bloq_graph::{
        GalleryItem,
        verify::{BoundaryOrder, LogicalVerifier},
    };
    use bloq_vm::{EnginePauli, EnginePauliString, run_bloq_with_io};
    use quizx::tensor::ToTensor;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, HashMap};

    let mut graph = GalleryItem::OneDYoked.build().flatten().unwrap();
    if prepared {
        graph.set_block_kind(IVec3::ZERO, BlockKind::T).unwrap();
    }
    let graph = graph
        .rotate_about_origin(UDirection::Z, quarter_turns)
        .unwrap()
        .fix_shadowed_faces();
    let position = |x, y, z| {
        if quarter_turns == 0 {
            IVec3::new(x, y, z)
        } else {
            IVec3::new(-y, x, z)
        }
    };
    let inputs = (i32::from(prepared)..6)
        .map(|x| position(x, 0, 0))
        .collect::<Vec<_>>();
    let outputs = (0..6).map(|x| position(x, 1, 9)).collect::<Vec<_>>();
    let support = [
        position(4, 0, 0),
        position(5, 0, 0),
        position(4, 1, 9),
        position(5, 1, 9),
    ];

    // The source Choi map preserves X4 X5 across both global parity checks.
    // Contract the prepared-T source independently of scalar frame derivation.
    let boundaries = inputs.iter().chain(&outputs).copied().collect::<Vec<_>>();
    let verifier =
        LogicalVerifier::with_boundaries(&graph, BoundaryOrder::new(inputs, outputs)).unwrap();
    let (_, diagram) = verifier.instantiate(&BTreeMap::new()).unwrap();
    let mut diagram = diagram.unwrap();
    quizx::simplify::full_simp(&mut diagram);
    let tensor = diagram.to_tensor64().iter().copied().collect::<Vec<_>>();
    let mask = boundaries
        .iter()
        .enumerate()
        .filter(|(_, port)| support.contains(port))
        .fold(0, |mask, (index, _)| {
            mask | (1 << (boundaries.len() - index - 1))
        });
    #[expect(
        clippy::redundant_closure_for_method_calls,
        reason = "naming the foreign method would require a direct num-complex test dependency"
    )]
    let norm = tensor.iter().map(|value| value.norm_sqr()).sum::<f64>();
    assert!(norm > 0.0);
    let expected = tensor
        .iter()
        .enumerate()
        .map(|(index, value)| (value.conj() * tensor[index ^ mask]).re)
        .sum::<f64>()
        / norm;
    assert!((expected - 1.0).abs() < 1e-9);

    let bloq = CompileContext::new(CompileConfig::new(3))
        .compile(&graph)
        .unwrap()
        .bloq;
    if !prepared {
        let program = graph.clone().with_inferred_interface().unwrap();
        common::choi::assert_channel(&program, &bloq, 16);
        return;
    }
    // Native T leaves a non-stabilizer state on eleven Choi qubits. Keep this
    // focused source correlation without pretending it is full tomography.
    let references = RefCell::new(HashMap::new());
    let report = run_bloq_with_io(
        &bloq,
        16,
        0xC401,
        |sim, context| {
            assert_eq!(context.inputs.len(), 6 - usize::from(prepared));
            let first = sim.num_qubits();
            for (index, input) in context.inputs.iter().enumerate() {
                sim.cx(input.qubit, first + index)?;
                references.borrow_mut().insert(input.port, first + index);
            }
            Ok(())
        },
        |sim, context| {
            use common::choi::stabilizer::{assert_pauli_expectation, physical_qubit};
            let refs = [4, 5]
                .map(|y| physical_qubit(references.borrow()[&position(y, 0, 0)], sim.num_qubits()));
            let mut logicals = refs
                .iter()
                .map(|out| (out, (false, false)))
                .collect::<Vec<_>>();
            for y in [4, 5] {
                let port = position(y, 1, 9);
                let output = context.outputs.iter().find(|out| out.port == port).unwrap();
                let frame = context
                    .frames
                    .iter()
                    .find(|frame| frame.port == port)
                    .unwrap();
                logicals.push((output, (frame.x.unwrap(), frame.z.unwrap())));
            }
            let correlation = EnginePauliString::from_terms(4, (0..4).map(|q| (q, EnginePauli::X)));
            assert_pauli_expectation(sim, &logicals, &correlation, expected);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
}

#[test]
fn junction_feedback_uses_the_same_source_wire_in_graph_and_module_frames() {
    let source = "BLOG 1.0
        0: Port [0,0,-1] role=input
        1: ZXZ [0,0,0]
        2: Port [0,0,1] role=output
        3: Port [1,0,0] role=output
        0 -> +Z
        1 -> +Z
        1 -> +X
        feedback Z 1
    ";
    let compiler = CompileContext::new(CompileConfig::new(3));
    for direction in [" -> +Z", ""] {
        let graph = BlockGraph::from_blog_text(
            &source.replace("feedback Z 1", &format!("feedback Z 1{direction}")),
        )
        .unwrap();
        let module = graph.clone().with_inferred_interface().unwrap();
        for artifact in [compiler.compile(&graph), compiler.compile(&module)] {
            let bloq = artifact.unwrap().bloq;
            common::choi::assert_channel_with_seed(&module, &bloq, 4, 0xCAB);
        }
    }
}

fn assert_spatial_hadamard_is_physical(
    graph: &BlockGraph,
    distance: u32,
    seed: u64,
    label: &str,
    expected_observables: Option<usize>,
) {
    let ctx = CompileContext::new(CompileConfig::new(distance));
    let bloq = ctx
        .compile_and_validate(graph)
        .unwrap_or_else(|error| panic!("{label} d{distance}: {error}"))
        .bloq;
    let report =
        run_bloq(&bloq, 4, seed).unwrap_or_else(|error| panic!("{label} d{distance}: {error}"));
    assert!(
        !report.detectors.is_empty() && report.all_detectors_constant(),
        "{label} d{distance}: varying or missing detectors: {:?}",
        report
            .detectors
            .iter()
            .filter(|detector| !detector.constant)
            .map(|detector| &detector.per_shot)
            .collect::<Vec<_>>()
    );
    if let Some(expected) = expected_observables {
        assert_eq!(report.observables.len(), expected, "{label} d{distance}");
    }
}

#[test]
fn named_hadamard_and_spatial_records_survive_storage_reordering() {
    use bloq_graph::{Action, GalleryItem, MeasureTarget};

    for targets in [
        vec![MeasureTarget::Edge {
            src: IVec3::Z,
            dir: Direction::XPLUS,
        }],
        vec![
            MeasureTarget::Node(IVec3::Z),
            MeasureTarget::Node(IVec3::X + IVec3::Z),
        ],
    ] {
        let source = GalleryItem::CZSpatialH.build().flatten().unwrap();
        let actions = targets
            .into_iter()
            .enumerate()
            .map(|(index, target)| Action::Measure {
                target,
                name: format!("m{index}"),
            })
            .collect::<Vec<_>>();
        // The open source's records reach its outputs until a fill closes them.
        let filled = source
            .filled_graphs()
            .unwrap()
            .into_iter()
            .find_map(|mut graph| graph.set_actions(actions.clone()).is_ok().then_some(graph))
            .expect("a fill supports every named record");
        for reversed in [false, true] {
            let mut graph = BlockGraph::new();
            let mut blocks = filled.blocks().cloned().collect::<Vec<_>>();
            if reversed {
                blocks.reverse();
            }
            for block in blocks {
                graph.add_block(block);
            }
            for pipe in filled.pipes() {
                graph.add_pipe(if reversed && pipe.is_hadamard() {
                    Pipe::new(pipe.dst(), pipe.dir().negate()).with_hadamard()
                } else {
                    pipe.clone()
                });
            }
            graph.set_actions(actions.clone()).unwrap();
            let ctx = CompileContext::new(CompileConfig::new(3));
            let module = graph.clone().with_inferred_interface().unwrap();
            for artifacts in [ctx.compile_and_validate(&graph), ctx.compile(&module)] {
                let report = run_bloq(&artifacts.unwrap().bloq, 4, 0xA4).unwrap();
                assert!(report.all_detectors_constant());
                assert!(report.observables.len() >= actions.len());
                assert!(report.observables.iter().all(|observable| {
                    observable.per_shot.len() == 4
                        && observable.per_shot.iter().all(|&value| !value)
                }));
            }
        }
    }
}

#[test]
fn spatial_hadamard_h_graph_transitions_are_physical() {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n\
         0: XZX [0, 0, -1]\n\
         1: XZX [0, 0, 0]\n\
         2: Port [0, 0, 1]\n\
         3: ZXZ [1, 0, -1]\n\
         4: ZXZ [1, 0, 0]\n\
         5: Port [1, 0, 1]\n\
         [0, 0, -1] -> +Z\n\
         [0, 0, 0] -> +Z\n\
         [0, 0, 0] -H> +X\n\
         [1, 0, -1] -> +Z\n\
         [1, 0, 0] -> +Z\n",
    )
    .expect("H graph parses");
    assert_eq!(graph.stabilizers().expect("H graph stabilizers").len(), 2);
    for distance in [3, 5] {
        assert_spatial_hadamard_is_physical(&graph, distance, 0x57, "H graph", Some(2));
    }
}

#[test]
fn all_spatial_hadamard_hub_arm_shapes_are_physical() {
    let cube_kinds = [
        CubeKind::XZZ,
        CubeKind::ZXZ,
        CubeKind::ZZX,
        CubeKind::ZXX,
        CubeKind::XZX,
        CubeKind::XXZ,
    ];
    for (axis, minus_kind, hub_kind, hub_is_minus) in [
        (UDirection::X, CubeKind::XZX, CubeKind::XXZ, false),
        (UDirection::X, CubeKind::ZXZ, CubeKind::ZZX, false),
        (UDirection::Y, CubeKind::XZZ, CubeKind::ZZX, false),
        (UDirection::Y, CubeKind::ZXX, CubeKind::XXZ, false),
        (UDirection::X, CubeKind::ZZX, CubeKind::ZXZ, true),
        (UDirection::X, CubeKind::XXZ, CubeKind::XZX, true),
        (UDirection::Y, CubeKind::ZZX, CubeKind::XZZ, true),
        (UDirection::Y, CubeKind::XXZ, CubeKind::ZXX, true),
    ] {
        let (wall_dir, arm_dirs) = match axis {
            UDirection::X => (Direction::XPLUS, [Direction::YPLUS, Direction::YMINUS]),
            UDirection::Y => (Direction::YPLUS, [Direction::XPLUS, Direction::XMINUS]),
            UDirection::Z => unreachable!(),
        };
        for distance in [3, 5] {
            for mask in 0..4 {
                let step = axis.to_ivec3();
                let mut graph = BlockGraph::new();
                let hub_pos = if hub_is_minus { IVec3::ZERO } else { step };
                graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(minus_kind)));
                graph.add_block(Block::new(step, BlockKind::Cube(hub_kind)));
                graph.add_pipe(Pipe::new(IVec3::ZERO, wall_dir).with_hadamard());
                for (index, arm) in arm_dirs.into_iter().enumerate() {
                    if mask & (1 << index) == 0 {
                        continue;
                    }
                    let pos = hub_pos + arm.to_ivec3();
                    let Some(kind) = cube_kinds.iter().copied().find(|kind| {
                        let mut candidate = graph.clone();
                        candidate.add_block(Block::new(pos, BlockKind::Cube(*kind)));
                        candidate.add_pipe(Pipe::new(hub_pos, arm));
                        candidate.fix_shadowed_faces().validate().is_ok()
                    }) else {
                        panic!("{axis} {minus_kind:?}->{hub_kind:?} mask {mask}: no arm kind");
                    };
                    graph.add_block(Block::new(pos, BlockKind::Cube(kind)));
                    graph.add_pipe(Pipe::new(hub_pos, arm));
                }
                graph = graph.fix_shadowed_faces();
                let label = format!("{axis} {minus_kind:?}->{hub_kind:?} mask {mask}");
                assert_spatial_hadamard_is_physical(&graph, distance, 0x51, &label, None);
            }
        }
    }
}

#[test]
fn parallel_spatial_hadamard_walls_are_physical() {
    let mut graph = BlockGraph::new();
    graph.add_block(Block::new(
        IVec3::new(0, 0, 0),
        BlockKind::Cube(CubeKind::XZX),
    ));
    graph.add_block(Block::new(
        IVec3::new(1, 0, 0),
        BlockKind::Cube(CubeKind::XXZ),
    ));
    graph.add_block(Block::new(
        IVec3::new(2, 0, 0),
        BlockKind::Cube(CubeKind::XZX),
    ));
    graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::XPLUS).with_hadamard());
    graph.add_pipe(Pipe::new(IVec3::new(1, 0, 0), Direction::XPLUS).with_hadamard());
    let graph = graph.fix_shadowed_faces();
    for distance in [3, 5] {
        assert_spatial_hadamard_is_physical(&graph, distance, 0x52, "sandwich", None);
    }
}

/// Regression for the moment-merge measurement-aliasing bug: a `cnot` (two
/// worldlines merged by lattice surgery) has instances whose measurements alias
/// onto shared physical records. Before the fix, `collect_measurements` minted a
/// distinct global id per aliased `InstanceMeasurement`; only one was ever
/// recorded, so detectors naming the others died with `MissingRecord` and
/// `run_bloq` failed outright. This asserts the compiled program runs and its
/// detectors are constant and non-vacuously evaluated.
#[test]
fn cnot_aliased_measurements_resolve() {
    let test_case = bloq_test::select_test_cases_for_fixture(bloq_test::TestFixture::CNOT)
        .expect("cnot cases")
        .into_iter()
        .next()
        .expect("at least one cnot case");
    let name = test_case.id().to_string();
    let case = CompileReadyCase::from_test_case(test_case).expect("case builds");
    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&case.graph).expect("cnot compiles").bloq;

    let shots = 8;
    let report = match run_bloq(&bloq, shots, 0x5) {
        Ok(report) => report,
        Err(err) => panic!("{name}: run_bloq failed (aliasing regression): {err}"),
    };

    assert_eq!(report.shots, shots);
    assert_eq!(
        report.discarded, 0,
        "{name}: noiseless shots are not discarded"
    );
    assert!(
        report.all_detectors_constant(),
        "{name}: a detector varied across shots: {:?}",
        report
            .detectors
            .iter()
            .filter(|d| !d.constant)
            .map(|d| &d.per_shot)
            .collect::<Vec<_>>(),
    );
    // Non-vacuous: at least one detector was evaluated in every shot.
    assert!(
        report.detectors.iter().any(|d| d.per_shot.len() == shots),
        "{name}: no detector evaluated across all {shots} shots (vacuous constancy)",
    );
}

/// The T block: a `RepeatUntilSuccess` region (cultivation + escape) plus a
/// `Branch` feedforward correction and guarded observable classical structure —
/// the dynamic control flow a procedural engine exists for. Under noiseless
/// execution the RUS accepts its first attempt; detectors (incl. post-selected
/// restart parities) must be constant, and the genuine non-Clifford T must
/// grow the stabilizer rank above 1.
#[test]
fn t_block_runs_through_regions() {
    let test_case = find_test_case("t_gate[open]").expect("native T gate case");
    let name = test_case.id().to_string();
    let case = CompileReadyCase::from_test_case(test_case).expect("case builds");
    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&case.graph).expect("t block compiles").bloq;

    let report = match run_bloq(&bloq, 8, 0x7) {
        Ok(report) => report,
        Err(err) => panic!("{name}: run_bloq failed: {err}"),
    };

    assert_eq!(report.shots, 8);
    assert!(
        report.all_detectors_constant(),
        "{name}: a detector varied across shots (RUS accepts first attempt noiselessly)",
    );
    assert!(
        report.max_rank > 1,
        "{name}: expected a genuine T (rank > 1), saw max_rank {}",
        report.max_rank,
    );
    // Coverage guard: at least one detector was evaluated in every shot, so
    // the constancy check above is not vacuous.
    assert!(
        report.detectors.iter().any(|d| d.per_shot.len() == 8),
        "{name}: no detector evaluated across all 8 shots (vacuous constancy)",
    );
    // Feedforward exercised: the Branch selector took both arms across the seeds.
    assert!(
        report.branch_selectors.iter().any(|&b| b) && report.branch_selectors.iter().any(|&b| !b),
        "{name}: expected both Branch arms across seeds, saw {:?}",
        report.branch_selectors,
    );

    // The raw observable is byproduct-dependent by design; corrections live in
    // the terminal Pauli frame. The empirical feedforward invariant we can
    // ground is
    // `raw_obs ⊕ branch_selector`: the Branch selector is the action bit that
    // drives the Z-frame correction (frame.z, node 21), so the raw byproduct XOR
    // that bit is the feedforward-consistent logical value and must be constant
    // across seeds.
    //
    // The fully-grounded decomposition is `raw ⊕ frame.z`; `run_bloq` folds
    // the frame's member `Decode`s natively. We keep the coarser
    // `raw ⊕ branch_selector` check here (the
    // selector is the same action bit frame.z folds) and separately assert
    // frame.z is evaluable below.
    //
    // The raw fold here is record parity plus live `Output`-face boundary Paulis
    // only: the T block's input-boundary anchor contributes nothing
    // physically — it is the magic-state source endpoint, closed internally by the
    // cultivation seam (its records already fold into observable recipes), so
    // `observable_value` skips `Input` faces.
    assert!(
        !report.observables.is_empty(),
        "{name}: expected a logical observable"
    );
    assert_eq!(
        report.branch_selectors.len(),
        8,
        "{name}: expected exactly one Branch selector per shot, saw {:?}",
        report.branch_selectors,
    );
    let raw = &report.observables[0].per_shot;
    assert_eq!(
        raw.len(),
        8,
        "{name}: observable 0 not evaluated in every shot: {raw:?}"
    );
    let corrected: Vec<bool> = raw
        .iter()
        .zip(&report.branch_selectors)
        .map(|(&o, &b)| o ^ b)
        .collect();
    assert!(
        corrected.windows(2).all(|w| w[0] == w[1]),
        "{name}: raw_obs ⊕ branch_selector varied across seeds: raw={raw:?} sel={:?} corrected={corrected:?}",
        report.branch_selectors,
    );

    // Frame bits read member `Decode`s (record-backed corrected parities),
    // so `run_bloq` folds them natively: frame.z must be `Some` every shot. It
    // is legitimately shot-varying (it tracks the teleportation byproduct), so
    // we assert evaluability, NOT constancy.
    assert_eq!(
        report.frame_pairs.len(),
        1,
        "{name}: expected one output frame pair"
    );
    assert_eq!(
        report.frame_pairs[0].z_bits.len(),
        8,
        "{name}: frame z evaluated in every shot"
    );
    assert!(
        report.frame_pairs[0].z_bits.iter().all(Option::is_some),
        "{name}: frame z-bit should be Some (record-backed), saw {:?}",
        report.frame_pairs[0].z_bits,
    );

    // The T block tracks no X byproduct, so its X-frame content is empty and
    // folds to a constant `Some(false)` every shot (the probe-confirmed shape).
    assert!(
        report.frame_pairs[0]
            .x_bits
            .iter()
            .all(|x| *x == Some(false)),
        "{name}: frame x-bit expected empty (Some(false)) every shot, saw {:?}",
        report.frame_pairs[0].x_bits,
    );
}

#[test]
fn thth_frame_recovers_exact_gate_sequence() {
    use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
    use glam::IVec3;

    let mut graph = bloq_graph::GalleryItem::THTH.build().flatten().unwrap();
    graph
        .set_block_kind(IVec3::ZERO, BlockKind::Cube(CubeKind::XZX))
        .expect("replace the input port with an XZX cube");
    graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
    graph.add_pipe(Pipe::new(IVec3::NEG_Z, Direction::ZPLUS).with_hadamard());

    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&graph).expect("THTH compiles").bloq;
    let shots = 3;
    let seed = 0x11;

    // The added input Hadamard precedes the gallery's T;H;T;H circuit.
    let oracle = ExactChoi::new(&[IVec3::ZERO], &[IVec3::new(-1, 2, 3)], |sim| {
        sim.h(0);
        sim.t(0)?;
        sim.h(0);
        sim.t(0)?;
        sim.h(0);
        Ok(())
    });
    let report = oracle.run(&bloq, shots, seed, |_, _| Ok(()));
    assert_eq!(
        report.branch_selectors.len(),
        shots * 2,
        "two selective branches per shot",
    );
    assert!(
        report
            .branch_selectors
            .as_chunks::<2>()
            .0
            .iter()
            .any(|selectors| selectors[0]),
        "the first selector's true arm must be exercised",
    );
    assert_eq!(report.frame_pairs.len(), 1, "one output frame pair");
}

/// A T-gate teleportation fixture whose data input is fed by a second T block:
/// the gadget consumes |T⟩ and outputs `T|T⟩ = S|+⟩ = |+i⟩`, a state every
/// byproduct component moves visibly (⟨Y⟩ flips under X or Z). This is the
/// sharp Z-frame oracle the plain `t.blog` cannot give: its open input runs as
/// |0⟩ and `T|0⟩ = |0⟩` absorbs any Z byproduct, which is how a selector-only
/// frame once survived every physical check.
///
/// `("YX", false)` is `t.blog`'s cap; `("YZ", true)` the Hadamard-capped
/// variant whose crossing rides the H-conjugated arm.
fn t_fed_t_gate_bloqs(cap: &str, h_pipe: bool, resolve: &str) -> [bloq_compile::Bloq; 2] {
    // H conjugates the Y effect to -Y. Explicit source feedback makes this
    // spelling the same S|+> preparation as the ordinary YX cap.
    let feedback = if h_pipe { "feedback Z 2 if mzz" } else { "" };
    let source = format!(
        "BLOG 1.0

  0: T [0, 0, 0]
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2]
  3: T [1, 0, 0]
  4: XZX [1, 0, 1]
  5: {cap} [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 0] -> +Z
  [1, 0, 2] {arrow} -Z

  mzz = measure 1 -> +X
  resolve 5 if {resolve}
  {feedback}
",
        arrow = if h_pipe { "-H>" } else { "->" },
    );
    let graph = bloq_graph::BlockGraph::from_blog_text(&source)
        .expect("t-fed t gate blog parses")
        .fix_shadowed_faces();
    let ctx = CompileContext::new(CompileConfig::new(3));
    let module = graph.clone().with_inferred_interface().unwrap();
    [ctx.compile(&graph), ctx.compile(&module)]
        .map(|artifacts| artifacts.expect("t-fed t gate compiles").bloq)
}

/// The honest, fully-grounded frame oracle: run shots, read logical expectations
/// with the frame-predicted `X^x Z^z` signs, and require the
/// corrected Bloch vector to be exactly `(0, +1, 0)` — the `|+i⟩ = S|+⟩` the
/// T²-fed program produces — on EVERY shot. Absolute, not just shot-constant,
/// so a constant frame error fails too. Also requires the uncorrected state to
/// have varied (the frame did real work) and both `Branch` arms to fire.
fn assert_frame_recovers_s_plus(name: &str, bloq: &bloq_compile::Bloq) {
    let oracle = ExactChoi::new(&[], &[IVec3::new(0, 0, 2)], |sim| {
        sim.h(0);
        sim.s(0);
        Ok(())
    });
    let shots = 12;
    let (mut saw_true, mut saw_false, mut byproduct_varied) = (false, false, false);
    for seed in [0x11u64, 0x22, 0x33] {
        let mut uncorrected_y: Vec<f64> = Vec::new();
        let report = oracle.run(bloq, shots, seed, |sim, ctx| {
            let out = &ctx.outputs[0];
            let (_, ey, _) = logical_bloch(sim, out, (false, false))?;
            uncorrected_y.push(ey);
            assert_eq!(ctx.frames.len(), 1, "{name}: one output frame");
            Ok(())
        });
        saw_true |= report.branch_selectors.iter().any(|&b| b);
        saw_false |= report.branch_selectors.iter().any(|&b| !b);
        byproduct_varied |=
            uncorrected_y.iter().any(|&y| y > 0.5) && uncorrected_y.iter().any(|&y| y < -0.5);
    }
    assert!(
        byproduct_varied,
        "{name}: uncorrected output never varied — the oracle is vacuous",
    );
    assert!(
        saw_true && saw_false,
        "{name}: feedforward did not exercise both Branch arms across seeds",
    );
}

/// The T-gate frame must track the cap outcome, not just the selector: with a
/// |T⟩ input the output is `S|+⟩` and any missed byproduct component leaves
/// the corrected state off `|+i⟩` on some shot. (The old σ frame — selector ⊕
/// polarity, no record reads — fails exactly this.)
#[test]
fn t_gate_frame_recovers_exact_t_squared_state() {
    for bloq in t_fed_t_gate_bloqs("YX", false, "mzz") {
        assert_frame_recovers_s_plus("t_gate[t-fed]", &bloq);
    }
}

/// Change only the crossing logical row's decoder estimate. The quantum
/// circuit and raw records stay fixed; precisely its Z-frame bit must flip.
fn assert_crossing_decoder_flip_changes_z_frame(bloq: &bloq_ir::Bloq) {
    let frame = bloq
        .nodes()
        .find_map(|(id, node)| {
            matches!(
                node.provenance,
                NodeProvenance::OutputFrame {
                    basis: bloq_ir::Basis::Z,
                    ..
                }
            )
            .then_some(id)
        })
        .expect("Z output frame");
    let crossing = bloq
        .top()
        .value_inputs(frame)
        .find_map(|input| {
            matches!(
                bloq[input.producer].try_classical(),
                Some(ClassicalNode::Observable { index: Some(_), .. })
            )
            .then_some(input.producer)
        })
        .expect("the Z frame uses its complete crossing readout");
    let before = run_bloq(bloq, 4, 0xDEC0DE).unwrap();
    let mut flipped = bloq.clone();
    let incoming = flipped
        .incoming(crossing)
        .map(|edge| (edge.source, edge.edge.clone()))
        .collect::<Vec<_>>();
    let outgoing = flipped
        .outgoing(crossing)
        .map(|edge| (edge.target, edge.edge.clone()))
        .collect::<Vec<_>>();
    let readout = flipped.remove_node(crossing).unwrap();
    let readout = flipped.add_node(readout);
    let corrected = flipped.add_node(bloq_ir::BloqNode::classical(ClassicalNode::Compute {
        expr: bloq_ir::ClassicalExpr::Not(Box::new(bloq_ir::ClassicalExpr::In(0))),
    }));
    for (source, edge) in incoming {
        flipped.add_edge(source, readout, edge);
    }
    flipped.add_edge(readout, corrected, bloq_ir::BloqEdge::value(0));
    for (target, edge) in outgoing {
        flipped.add_edge(corrected, target, edge);
    }
    flipped.validate().unwrap();
    let after = run_bloq(&flipped, 4, 0xDEC0DE).unwrap();
    assert_eq!(before.discarded, 0);
    assert_eq!(after.discarded, 0);
    assert_eq!(before.detectors, after.detectors);
    assert_eq!(
        before.observables, after.observables,
        "only the decoder estimate changed"
    );
    assert_eq!(before.frame_pairs.len(), 1);
    assert_eq!(after.frame_pairs.len(), 1);
    assert_eq!(before.frame_pairs[0].x_bits, after.frame_pairs[0].x_bits);
    for (before, after) in before.frame_pairs[0]
        .z_bits
        .iter()
        .zip(&after.frame_pairs[0].z_bits)
    {
        assert_eq!(*after, Some(!before.expect("evaluable Z frame")));
    }
}

#[test]
fn t_block_frame_z_uses_the_crossing_rows_decoder_estimate() {
    let test_case = find_test_case("t_gate[open]").expect("native T gate case");
    let case = CompileReadyCase::from_test_case(test_case).expect("case builds");
    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&case.graph).expect("t block compiles").bloq;
    assert_crossing_decoder_flip_changes_z_frame(&bloq);
}

/// `t.blog` with its ancilla cap Hadamard-conjugated: a temporal `H` pipe
/// below the selective node, whose arms rotate from the original `YX`'s
/// {Y when `mzz`, X otherwise} to {Y when `mzz`, Z otherwise} — the `YZ`
/// spelling. Physically the same T-gate feedforward (post-H `Z` ≡ pre-H `X`,
/// post-H `Y` ≡ pre-H `−Y`).
///
/// Its purpose: the cap-crossing logical threads the temporal H pipe, so the
/// branch content must fold the pipe (realignment) gateway's records
/// (`pipe_crossing_content`), and the selector arm's fixed row crosses
/// the pipe as `Y`, picking up the transversal-H `Y_L → −Y_L` sign. No other
/// corpus program exercises either path.
const HADAMARD_CAPPED_T_BLOG: &str = "BLOG 1.0

  0: Port [0, 0, 0]
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2]
  3: T [1, 0, 0]
  4: XZX [1, 0, 1]
  5: YZ [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 0] -> +Z
  [1, 0, 2] -H> -Z

  mzz = measure 1 -> +X
  resolve 5 if mzz
";

fn hadamard_capped_t_bloq() -> bloq_compile::Bloq {
    let graph = bloq_graph::BlockGraph::from_blog_text(HADAMARD_CAPPED_T_BLOG)
        .expect("hadamard-capped t blog parses")
        .fix_shadowed_faces();
    let ctx = CompileContext::new(CompileConfig::new(3));
    ctx.compile(&graph)
        .expect("hadamard-capped t block compiles")
        .bloq
}

/// The H-conjugated cap needs explicit conditional Z feedback to prepare
/// S|+>. Both the realignment records and that source feedback must survive.
#[test]
fn hadamard_capped_t_frame_recovers_exact_t_squared_state() {
    for bloq in t_fed_t_gate_bloqs("YZ", true, "mzz") {
        assert_frame_recovers_s_plus("hadamard-capped t[t-fed]", &bloq);
    }
}

/// A nonlinear resolve condition (`mzz | mzz`) must not degrade the fixture
/// off the exact tier, and its frame must still pass the absolute oracle.
#[test]
fn nonlinear_resolve_condition_keeps_exact_frames() {
    for bloq in t_fed_t_gate_bloqs("YZ", true, "mzz | mzz") {
        assert_frame_recovers_s_plus("nonlinear hadamard-capped t[t-fed]", &bloq);
    }
}

/// H Y H = -Y changes the source instrument's true branch. Without explicit
/// feedback the corrected state is S†|+> there, as independently checked by ZX.
#[test]
fn hadamard_capped_t_preserves_the_source_branch_sign() {
    assert_crossing_decoder_flip_changes_z_frame(&hadamard_capped_t_bloq());
    let graph = BlockGraph::from_blog_text(&HADAMARD_CAPPED_T_BLOG.replacen("0: Port", "0: T", 1))
        .unwrap()
        .fix_shadowed_faces();
    let module = graph.clone().with_inferred_interface().unwrap();
    let compiler = CompileContext::new(CompileConfig::new(3));
    for artifacts in [compiler.compile(&graph), compiler.compile(&module)] {
        let bloq = artifacts.unwrap().bloq;
        let ideals = [false, true].map(|negative| {
            ExactChoi::new(&[], &[IVec3::new(0, 0, 2)], |sim| {
                sim.h(0);
                sim.s(0);
                if negative {
                    sim.z(0);
                }
                Ok(())
            })
        });
        let mut selected = Vec::new();
        let report = ideals[0].run_selected(
            &bloq,
            12,
            0x11,
            |context| {
                assert_eq!(context.branch_selectors.len(), 1);
                &ideals[usize::from(context.branch_selectors[0])]
            },
            |_, context| {
                selected.push(context.branch_selectors[0]);
                Ok(())
            },
        );
        let readout = &report.observables[0].per_shot;
        assert!(readout.contains(&false) && readout.contains(&true));
        assert_eq!(
            *readout, selected,
            "source readout selects the signed ideal"
        );
        assert_eq!(report.discarded, 0);
        assert!(report.all_detectors_constant());
    }
}

/// The consumed-Output-face read: `run_bloq` measures an observable's output
/// boundary on the TERMINAL simulator state, which is only
/// correct while the face is a live boundary. When a later-emitted node touches
/// the face's support — an output patch whose tiles a later patch reuses — the
/// read moves to the cut just before that node. Construct the shape: compile
/// `cnot[open]` (its observables fold Output faces), then append a quantum node
/// re-instantiating the face's own template at the same offset, ordered after
/// the face's owning node — the shape a real program reaches when one layout
/// column carries two outputs in sequence.
///
/// This pins the *shape* only: `cnot[open]`'s faces are not deterministic, so
/// their values move with the RNG stream. The executor's
/// `repeated_output_captures_use_owner_cuts_without_observables` test pins the
/// captured values on outputs prepared in a definite state.
#[test]
fn consumed_output_face_is_read_at_its_cut() {
    use bloq_ir::lowering::TemplateInstance;
    use bloq_ir::{BloqEdge, BloqNode};

    let (_case, mut bloq) = compiled_case("cnot[open]", 3);

    // This fixture tests diagnostic output captures, without decoder estimates.
    // Route frame consumers through explicit raw projections of the complete readouts.
    let observables = bloq
        .nodes()
        .filter_map(|(id, node)| match node.try_classical() {
            Some(ClassicalNode::Observable { index: Some(_), .. }) => Some(id),
            _ => None,
        })
        .collect::<Vec<_>>();
    for id in observables {
        let incoming = bloq
            .incoming(id)
            .map(|edge| (edge.source, edge.edge.clone()))
            .collect::<Vec<_>>();
        let outgoing = bloq
            .outgoing(id)
            .map(|edge| (edge.target, edge.edge.clone()))
            .collect::<Vec<_>>();
        let readout = bloq.remove_node(id).unwrap();
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let readout = bloq.add_node(readout);
        for (source, edge) in incoming {
            bloq.add_edge(source, readout, edge);
        }
        bloq.add_edge(readout, raw, BloqEdge::compose(0));
        for (target, edge) in outgoing {
            bloq.add_edge(raw, target, edge);
        }
    }

    // Complete readouts whose values are unused retain their diagnostic output faces.
    let face_instance = bloq
        .top()
        .nodes()
        .find_map(|(_, node)| {
            node.try_classical()?
                .operators()
                .iter()
                .find(|operator| operator.face == bloq_ir::BoundaryFace::Output)
                .map(|operator| operator.instance)
        })
        .expect("cnot[open] binds an output face");

    // The node owning the face's instance, plus that instance's template/offset.
    let (owner_id, template_id, offset) = bloq
        .quantum_nodes()
        .find_map(|(id, quantum)| {
            quantum
                .instances
                .iter()
                .find(|instance| instance.id == face_instance)
                .map(|instance| (id, instance.template_id, instance.offset))
        })
        .expect("face instance belongs to a top-level quantum node");

    // Append a consuming node on the face's support, ordered after the owner.
    let fresh_instance = bloq_ir::lowering::TemplateInstanceId(
        bloq.quantum_nodes()
            .flat_map(|(_, quantum)| quantum.instances.iter())
            .map(|instance| instance.id.0)
            .max()
            .expect("program has instances")
            + 1,
    );
    let mut consumer = BloqNode::from_members(Vec::new());
    consumer
        .expect_quantum_mut()
        .instances
        .push(TemplateInstance::new(fresh_instance, template_id, offset));
    let consumer_id = bloq.add_node(consumer);
    bloq.add_edge(owner_id, consumer_id, BloqEdge::Order);

    let shots = 8;
    let report = run_bloq(&bloq, shots, 0xC0FFEE).expect("the face is read at its cut");
    assert!(
        !report.observables.is_empty(),
        "cnot[open] folds observables"
    );
    for (index, observable) in report.observables.iter().enumerate() {
        assert_eq!(
            observable.per_shot.len(),
            shots,
            "observable {index} folds to a value in every shot",
        );
    }
}

// ==============================================================================
// Feedback fixtures F1/F2/F3 for branch-aware frame routing.
// ==============================================================================

/// F1: two |+⟩ worldlines joined by a deterministic XX seam and named X readout, then
/// one `Feedback` whose Z Paulis anticommute the seam's surface at BOTH
/// worldlines. The two parity-frame folds cancel, so the compiled observable
/// report must not change (I2).
const MULTI_TARGET_FEEDBACK_BLOG: &str = "BLOG 1.0

  0: ZXX [0, 0, 0]
  1: ZXX [0, 0, 1]
  2: ZXX [0, 1, 0]
  3: ZXX [0, 1, 1]
  [0, 0, 0] -> +Z
  [0, 1, 0] -> +Z
  [0, 0, 1] -> +Y
";

/// F1 (fill-model-simplification I2): two anticommuting feedback targets cancel.
#[test]
fn feedback_multi_target_even_parity_does_not_flip_observable() {
    use bloq_graph::{Action, FeedbackTarget, MeasureTarget, Pauli, PauliBasis};
    use glam::IVec3;

    let mut graph = bloq_graph::BlockGraph::from_blog_text(MULTI_TARGET_FEEDBACK_BLOG)
        .expect("multi-target feedback blog parses")
        .fix_shadowed_faces();
    graph
        .set_actions(vec![
            Action::Measure {
                target: MeasureTarget::Node(IVec3::new(0, 0, 1)),
                name: "m".into(),
            },
            Action::Feedback {
                targets: vec![
                    FeedbackTarget {
                        pauli: PauliBasis::Z,
                        target: IVec3::new(0, 0, 0),
                        direction: None,
                    },
                    FeedbackTarget {
                        pauli: PauliBasis::Z,
                        target: IVec3::new(0, 1, 0),
                        direction: None,
                    },
                ],
                condition: None,
            },
        ])
        .expect("multi-target feedback derives its measurement surface");

    // Fixture guard: the program tracks exactly one generator — m's canonical
    // surface — and the feedback's Z anticommutes its X support at both
    // targets. If derivation ever reshapes this, the test stops covering the
    // even-count fold and must be revisited.
    let stabilizers = graph.stabilizers().expect("stabilizers derive");
    assert_eq!(stabilizers.len(), 1, "one tracked generator");
    assert!(stabilizers[0].is_measurement());
    for target in [IVec3::new(0, 0, 0), IVec3::new(0, 1, 0)] {
        assert_eq!(
            stabilizers[0].stabilizer.interior_nodes.get(&target),
            Some(&Pauli::X),
            "the seam surface carries X at feedback target {target}",
        );
    }

    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&graph).expect("fixture compiles").bloq;
    let observable_count = bloq
        .top()
        .nodes()
        .filter(|(_, node)| {
            matches!(
                node.try_classical(),
                Some(ClassicalNode::Observable { index: Some(_), .. })
            )
        })
        .count();
    assert_eq!(observable_count, 1, "m's surface is the only observable");

    let shots = 4;
    let report = run_bloq(&bloq, shots, 0xF1).expect("run_bloq");
    assert_eq!(report.discarded, 0);
    let per_shot = &report.observables[0].per_shot;
    assert_eq!(per_shot.len(), shots, "m folded every shot");
    assert!(
        per_shot.iter().all(|&bit| !bit),
        "the deterministic +1 XX seam changed despite even feedback parity: {per_shot:?}",
    );
}

/// F2: a port-to-port main worldline ZZ-merged with two independent |0⟩
/// ancillas, at layer 1 (`m1`) and layer 2 (`m2`), with an unconditioned
/// `feedback X` on the temporal wire after the first merge. Only `m2` flips,
/// so `discard if m1 ^ !m2` accepts every shot (SEM-READ).
const FRAMED_CONDITION_READ_BLOG: &str = "BLOG 1.0

  0: Port [0, 0, 0]
  1: XZX [0, 0, 1]
  2: XZX [0, 0, 2]
  3: Port [0, 0, 3]
  4: XZZ [1, 0, 0]
  5: XZX [1, 0, 1]
  6: XZZ [-1, 0, 1]
  7: XZX [-1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 2] -> +Z
  [1, 0, 0] -> +Z
  [-1, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [0, 0, 2] -> -X

  m1 = measure 1 -> +X
  feedback X [0, 0, 1]
  m2 = measure 2 -> -X
  discard if m1 ^ !m2
";

/// F2b: source order does not stop an unconditional feedback bit from folding
/// into a condition row listed before it.
const READ_LISTED_BEFORE_FEEDBACK_BLOG: &str = "BLOG 1.0

  0: Port [0, 0, 0]
  1: XZX [0, 0, 1]
  2: XZX [0, 0, 2]
  3: Port [0, 0, 3]
  4: XZZ [1, 0, 0]
  5: XZX [1, 0, 1]
  6: XZZ [-1, 0, 1]
  7: XZX [-1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 2] -> +Z
  [1, 0, 0] -> +Z
  [-1, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [0, 0, 2] -> -X

  m1 = measure 1 -> +X
  m2 = measure 2 -> -X
  discard if m1 ^ !m2
  feedback X [0, 0, 1]
";

/// F2c: the F2 shape shifted down one layer so the graph starts at z = −1 and
/// the feedback targets z = 0 — a genuine `AfterLayer(0)` guard, not the
/// Let/DiscardIf `Immediate` guard.
const ZERO_LAYER_FEEDBACK_NEGATIVE_GRAPH_BLOG: &str = "BLOG 1.0

  0: Port [0, 0, -1]
  1: XZX [0, 0, 0]
  2: XZX [0, 0, 1]
  3: Port [0, 0, 2]
  4: XZZ [1, 0, -1]
  5: XZX [1, 0, 0]
  6: XZZ [-1, 0, 0]
  7: XZX [-1, 0, 1]
  [0, 0, -1] -> +Z
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [1, 0, -1] -> +Z
  [-1, 0, 0] -> +Z
  [0, 0, 0] -> +X
  [0, 0, 1] -> -X

  m1 = measure 1 -> +X
  feedback X [0, 0, 0]
  m2 = measure 2 -> -X
  discard if m1 ^ !m2
";

/// A source Pauli between the two ZZ merges changes only the later read.
/// List order and negative layers preserve the same instrument.
#[rstest::rstest]
#[case::between_reads(FRAMED_CONDITION_READ_BLOG, 0xF2)]
#[case::after_discard(READ_LISTED_BEFORE_FEEDBACK_BLOG, 0xF2B)]
#[case::negative_layers(ZERO_LAYER_FEEDBACK_NEGATIVE_GRAPH_BLOG, 0xF2C)]
fn unconditioned_feedback_uses_its_wire_for_condition_reads(
    #[case] source: &str,
    #[case] seed: u64,
) {
    use bloq_graph::{Action, verify::LogicalVerifier};
    use quizx::{fscalar::Zero, tensor::ToTensor};
    use std::collections::BTreeMap;

    let source = BlockGraph::from_blog_text(source)
        .unwrap()
        .fix_shadowed_faces();
    let compiler = CompileContext::new(CompileConfig::new(3));
    for feedback in [false, true] {
        let mut graph = source.clone();
        graph
            .set_actions(
                source
                    .actions()
                    .iter()
                    .filter(|action| feedback || !matches!(action, Action::Feedback { .. }))
                    .cloned()
                    .collect(),
            )
            .unwrap();

        // Contract all four source branches before postselection. The two
        // reads agree without feedback and disagree with X on the middle wire.
        let mut unfiltered = graph.clone();
        unfiltered
            .set_actions(
                graph
                    .actions()
                    .iter()
                    .filter(|action| !matches!(action, Action::DiscardIf(_)))
                    .cloned()
                    .collect(),
            )
            .unwrap();
        let verifier = LogicalVerifier::new(&unfiltered).unwrap();
        for m1 in [false, true] {
            for m2 in [false, true] {
                let assignment = BTreeMap::from([("m1".to_owned(), m1), ("m2".to_owned(), m2)]);
                let (_, diagram) = verifier.instantiate(&assignment).unwrap();
                let mut diagram = diagram.unwrap();
                quizx::simplify::full_simp(&mut diagram);
                assert_eq!(
                    diagram.to_tensorf().iter().any(|value| !value.is_zero()),
                    m1 ^ m2 == feedback,
                    "feedback={feedback}, m1={m1}, m2={m2}",
                );
            }
        }

        let bloq = compiler.compile(&graph).unwrap().bloq;
        let report = run_bloq(&bloq, 4, seed).unwrap();
        assert_eq!(report.discarded, if feedback { 0 } else { 4 });
        if feedback {
            assert!(!report.detectors.is_empty());
            assert!(report.all_detectors_constant());
        }
    }
}

/// F3: a |0⟩-initialized worldline ending at an output port, with an
/// unconditioned mid-wire `Feedback X` and no measurements. The IR never
/// applies feedback physically, so the byproduct must reach the consumer
/// through the output frame (I3: a record-free correction constraint still
/// carries pending-feedback influence).
const FEEDBACK_ONLY_FRAME_BLOG: &str = "BLOG 1.0

  0: ZXZ [0, 0, 0]
  1: Port [0, 0, 1]
  [0, 0, 0] -> +Z

  feedback X [0, 0, 0]
";

/// F3 (fill-model-simplification I3): the frame-corrected output must carry
/// the pending X — state-fidelity gate against the oracle semantics
/// `X|0⟩ = |1⟩` (Bloch (0, 0, −1)).
#[test]
fn feedback_only_frame_content_reports_the_byproduct() {
    let graph = bloq_graph::BlockGraph::from_blog_text(FEEDBACK_ONLY_FRAME_BLOG)
        .expect("feedback-only frame blog parses")
        .fix_shadowed_faces();
    let ctx = CompileContext::new(CompileConfig::new(3));
    let bloq = ctx.compile(&graph).expect("fixture compiles").bloq;
    let shots = 4;
    let seed = 0xF3;
    let oracle = ExactChoi::new(&[], &[IVec3::Z], |sim| {
        sim.x(0);
        Ok(())
    });
    let report = oracle.run(&bloq, shots, seed, |_, _| Ok(()));
    assert_eq!(report.frame_pairs.len(), 1, "one output frame pair");
}
