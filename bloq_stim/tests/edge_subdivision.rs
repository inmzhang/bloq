//! Acceptance: splicing a memory-round padding node into a
//! `Quantum` seam (`Bloq::insert_memory_rounds` instantiating the recorded
//! edge-owned padding provenance) preserves the program's observables and its
//! graphlike distance, and a bad splice is rejected verify-before-apply with
//! the program untouched.

use bloq_compile::{CompileConfig, CompileContext, compile_clifford_proxy};
use bloq_graph::{BlockGraph, GalleryItem};
use bloq_ir::lowering::PaddingInstance;
use bloq_ir::{Bloq, BloqEdge, EditError, LevelPath, MemoryRoundTarget, NodeProvenance};
use bloq_stim::{emit_bloq_stim, emit_bloq_stim_with_stim_noise};
use glam::{IVec2, ivec3};
use stim::Circuit;

mod common;
use common::{compile_two_cubes, noisy_graphlike_distance, quantum_seam, uniform_depolarizing};

const DISTANCE: u32 = 3;

fn parsed_emit(bloq: &Bloq) -> Circuit {
    emit_bloq_stim(bloq)
        .expect("emit Stim")
        .parse()
        .expect("parse emitted Stim")
}

/// Assert every detector of the noiseless program is deterministic.
///
/// Sampling the zero-noise circuit is the direct statement of the property: a
/// seam that composes the wrong measurements still emits detectors, and they
/// fire. The noisy error model is the symbolic counterpart — it refuses to
/// build at all when a detector is not deterministic — so the two together
/// catch both a wrong seam and a seam that is merely unmatchable.
fn assert_detectors_deterministic(bloq: &Bloq, label: &str) {
    let noiseless = parsed_emit(bloq);
    assert!(
        noiseless.num_detectors() > 0,
        "{label}: program has detectors to check"
    );
    let mut sampler = noiseless.compile_detector_sampler();
    let shots = sampler.sample(8);
    assert!(
        shots.iter().all(|fired| !*fired),
        "{label}: a detector fired without noise"
    );

    let noisy: Circuit = emit_bloq_stim_with_stim_noise(bloq, &uniform_depolarizing())
        .expect("emit noisy Stim")
        .parse()
        .expect("parse noisy Stim");
    noisy
        .detector_error_model()
        .unwrap_or_else(|error| panic!("{label}: detector error model must build: {error}"));
}

#[test]
fn memory_padding_preserves_observables_and_distance() {
    // 1 exercises the recorded one-round template, 3 the looped template
    // verbatim, and 2/5 its repetition-count specialization.
    let original = compile_two_cubes(DISTANCE);
    let baseline = parsed_emit(&original);
    let baseline_distance = noisy_graphlike_distance(&original);
    assert_eq!(baseline_distance, DISTANCE as usize);

    let mut detector_counts = [0; 4];
    for (index, rounds) in [1u32, 2, 3, 5].into_iter().enumerate() {
        let mut bloq = original.clone();

        let (from, to) = quantum_seam(&bloq);
        let mid = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from,
                    to,
                },
                rounds,
            )
            .expect("decoder-wait insertion verifies and applies");
        assert_eq!(bloq.deterministic_emit_order().expect("acyclic")[1], mid);
        let mid_node = &bloq[mid];
        assert_eq!(
            mid_node.memory_rounds(),
            Some(rounds),
            "rounds={rounds}: spliced wait is identifiable and readable"
        );
        bloq.validate().expect("spliced program validates");

        let spliced = parsed_emit(&bloq);
        assert_eq!(
            spliced.num_observables(),
            baseline.num_observables(),
            "rounds={rounds}: observables unchanged"
        );
        assert!(
            spliced.num_detectors() > baseline.num_detectors(),
            "rounds={rounds}: padding adds syndrome rounds"
        );
        detector_counts[index] = spliced.num_detectors();
        assert_eq!(
            noisy_graphlike_distance(&bloq),
            baseline_distance,
            "rounds={rounds}: graphlike distance unchanged"
        );
    }

    let [d1, d2, d3, d5] = detector_counts;
    let per_round = d2 - d1;
    assert!(per_round > 0, "each round adds detectors");
    assert_eq!(d3 - d2, per_round, "round 2->3 adds one round's detectors");
    assert_eq!(
        d5 - d3,
        2 * per_round,
        "round 3->5 adds two rounds' detectors"
    );
}

#[test]
fn partial_t_seam_wait_preserves_joint_observables() {
    for pin in [false, true] {
        let mut bloq = compile_clifford_proxy(
            CompileConfig::new(DISTANCE),
            &GalleryItem::T.build(),
            &[pin],
        )
        .expect("T proxy compiles")
        .bloq;
        let target = bloq
            .nodes()
            .find_map(|(id, node)| {
                node.block_members()
                    .iter()
                    .any(|member| member.pos == ivec3(1, 0, 2))
                    .then_some(id)
            })
            .expect("pinned selective node exists");
        let source = bloq
            .incoming(target)
            .find_map(|edge| matches!(edge.edge, BloqEdge::Quantum(_)).then_some(edge.source))
            .expect("pinned selective seam exists");

        bloq.insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from: source,
                to: target,
            },
            10,
        )
        .expect("partial seam accepts exact decoder wait");
        bloq.validate().expect("padded proxy validates");
        let circuit: Circuit = emit_bloq_stim_with_stim_noise(&bloq, &uniform_depolarizing())
            .expect("padded proxy emits")
            .parse()
            .expect("padded proxy Stim parses");
        let dem = circuit
            .detector_error_model()
            .expect("all padded proxy detectors and observables are deterministic");
        assert_eq!(dem.num_observables(), 3);
    }
}

/// Idling before a logical output port.
///
/// A port's incoming wait must instantiate the plain terminal face, not the
/// cube-side memory template: that one reinitializes the boundary ancillas,
/// presenting the port a different plaquette set than the seam it displaced and
/// leaving the port observable non-deterministic. Recomposition rejects it, and
/// the emitted circuit's error model is the backstop.
#[test]
fn wait_before_port_keeps_the_output_observable_deterministic() {
    for pin in [false, true] {
        let mut bloq = compile_clifford_proxy(
            CompileConfig::new(DISTANCE),
            &GalleryItem::T.build(),
            &[pin],
        )
        .expect("T proxy compiles")
        .bloq;
        let port = bloq
            .nodes()
            .find_map(|(id, node)| {
                node.block_members()
                    .iter()
                    .any(|member| member.pos == ivec3(0, 0, 2))
                    .then_some(id)
            })
            .expect("the T gate's logical output port exists");
        let source = bloq
            .incoming(port)
            .find_map(|edge| matches!(edge.edge, BloqEdge::Quantum(_)).then_some(edge.source))
            .expect("the port has a padded parent seam");

        bloq.insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from: source,
                to: port,
            },
            5,
        )
        .expect("port seam accepts a decoder wait");
        bloq.validate().expect("padded proxy validates");
        assert_detectors_deterministic(&bloq, &format!("port wait, pin={pin}"));
    }
}

/// Idling before a resolved Y-basis measurement.
///
/// Each of `ccz_4x3x6`'s four selective branches ends on an `XY` block whose
/// active resolve pairs the seam's plaquettes into composite chains, halving
/// the seam. Splicing padding in restores an ordinary memory seam below it, so
/// the new seam carries twice the detectors of the one it displaced — the
/// `(2,2,4)` branch is the case that exposes this, because its `ZXZ` source is
/// a lone instance rather than part of a merged region.
#[test]
fn wait_before_resolved_y_basis_target_splices() {
    for (label, pins, target_pos) in [
        ("active branch 0", [true; 4], ivec3(1, 0, 4)),
        ("active branch 1", [true; 4], ivec3(2, 0, 4)),
        ("active branch 2", [true; 4], ivec3(1, 2, 4)),
        ("active branch 3", [true; 4], ivec3(2, 2, 4)),
        ("inactive branch", [false; 4], ivec3(2, 2, 4)),
    ] {
        let mut bloq = compile_clifford_proxy(
            CompileConfig::new(DISTANCE),
            &GalleryItem::CCZ4x3_6.build(),
            &pins,
        )
        .expect("CCZ proxy compiles")
        .bloq;
        let baseline = parsed_emit(&bloq);

        let target = bloq
            .nodes()
            .find_map(|(id, node)| {
                node.block_members()
                    .iter()
                    .any(|member| member.pos == target_pos)
                    .then_some(id)
            })
            .expect("resolved Y-basis block exists");
        let source = bloq
            .incoming(target)
            .find_map(|edge| matches!(edge.edge, BloqEdge::Quantum(_)).then_some(edge.source))
            .expect("Y-basis block has a padded parent seam");

        let rounds = 10;
        let padding = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: source,
                    to: target,
                },
                rounds,
            )
            .unwrap_or_else(|error| panic!("{label} accepts a decoder wait: {error}"));
        bloq.validate()
            .expect("padded Y-basis branch program validates");

        assert_eq!(
            bloq[padding].memory_rounds(),
            Some(rounds),
            "{label}: the spliced node carries the requested rounds"
        );
        let spliced = parsed_emit(&bloq);
        assert_eq!(
            spliced.num_observables(),
            baseline.num_observables(),
            "{label}: observables unchanged"
        );
        // A d=3 patch has d*d-1 stabilizers, so every requested round lands.
        assert_eq!(
            spliced.num_detectors() - baseline.num_detectors(),
            u64::from(rounds) * u64::from(DISTANCE * DISTANCE - 1),
            "{label}: padding adds exactly the rounds it was asked for"
        );
        assert_detectors_deterministic(&bloq, label);
        assert_eq!(
            noisy_graphlike_distance(&bloq),
            DISTANCE as usize,
            "{label}: graphlike distance unchanged"
        );
    }
}

/// Idling on a seam between two merged regions of the CCZ factory.
///
/// This is the seam class LIM-019 named: both endpoints are multi-cube merged
/// components, and the seam carries two temporal pipes. Its padding used to be
/// compiled at each cube's *merged* footprint, which reproduced only 14 of the
/// seam's 16 detector chains — one lost per spatial merge direction.
#[test]
fn wait_on_a_merged_factory_seam_preserves_observables_and_distance() {
    let mut bloq = compile_clifford_proxy(
        CompileConfig::new(DISTANCE),
        &GalleryItem::CCZ4x3_6.build(),
        &[true; 4],
    )
    .expect("CCZ proxy compiles")
    .bloq;
    let baseline = parsed_emit(&bloq);
    let baseline_distance = noisy_graphlike_distance(&bloq);

    let node_at = |bloq: &Bloq, pos| bloq.node_by_block(pos).expect("merged region exists");
    let from = node_at(&bloq, ivec3(0, 1, 1));
    let to = node_at(&bloq, ivec3(0, 0, 2));
    let pipes = bloq
        .edges_between(from, to)
        .find_map(|edge| match edge.edge {
            BloqEdge::Quantum(quantum) => Some(quantum.pipes.len()),
            BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
        })
        .expect("the two merged regions share a Quantum seam");
    assert_eq!(pipes, 2, "the seam under test is a multi-pipe one");

    let rounds = 10;
    let padding = bloq
        .insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from,
                to,
            },
            rounds,
        )
        .expect("the merged factory seam accepts a decoder wait");
    bloq.validate().expect("padded factory program validates");
    assert_eq!(bloq[padding].memory_rounds(), Some(rounds));

    let spliced = parsed_emit(&bloq);
    assert_eq!(
        spliced.num_observables(),
        baseline.num_observables(),
        "observables unchanged"
    );
    assert_eq!(
        spliced.num_detectors() - baseline.num_detectors(),
        pipes as u64 * u64::from(rounds) * u64::from(DISTANCE * DISTANCE - 1),
        "each pipe idles for exactly the rounds it was asked for"
    );
    assert_detectors_deterministic(&bloq, "merged factory seam wait");
    assert_eq!(
        noisy_graphlike_distance(&bloq),
        baseline_distance,
        "graphlike distance unchanged"
    );
}

/// A reloaded `.bloq` program accepts decoder waits without its source graph.
#[test]
fn memory_padding_works_on_reloaded_program() {
    let mut fresh = compile_two_cubes(DISTANCE);
    let mut reloaded = Bloq::from_binary(&fresh.to_binary()).expect("binary round trip decodes");

    let (from, to) = quantum_seam(&reloaded);
    reloaded
        .insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from,
                to,
            },
            2,
        )
        .expect("decoder wait splices into the reloaded program");
    reloaded.validate().expect("spliced program validates");

    let (from, to) = quantum_seam(&fresh);
    fresh
        .insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from,
                to,
            },
            2,
        )
        .expect("decoder wait splices into the fresh program");
    assert_eq!(
        emit_bloq_stim(&reloaded).expect("emit reloaded"),
        emit_bloq_stim(&fresh).expect("emit fresh"),
    );
}

#[test]
fn equivalent_seams_share_padding_templates() {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n  1: ZXZ [0,0,1]\n  2: ZXZ [2,0,0]\n  3: ZXZ [2,0,1]\n  [0,0,0] -> +Z\n  [2,0,0] -> +Z\n",
    )
    .expect("two independent memory seams parse");
    let ctx = CompileContext::new(CompileConfig::new(DISTANCE));
    let bloq = ctx.compile(&graph).expect("memory seams compile").bloq;
    let padding: Vec<_> = bloq.pipe_padding().copied().collect();

    assert_eq!(padding.len(), 2);
    assert_eq!(padding[0].one_round, padding[1].one_round);
    assert_eq!(padding[0].looped, padding[1].looped);
    assert_ne!(padding[0].offset, padding[1].offset);
}

#[test]
fn memory_padding_pads_both_sides_of_temporal_hadamard() {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n  0: XZX [0,0,0]\n  1: ZXZ [0,0,1]\n  [0,0,0] -H> +Z\n",
    )
    .expect("temporal-Hadamard graph parses");
    let ctx = CompileContext::new(CompileConfig::new(DISTANCE));
    let mut bloq = ctx
        .compile(&graph)
        .expect("temporal-Hadamard graph compiles")
        .bloq;

    let hadamard = bloq
        .nodes()
        .find_map(|(id, node)| match node.provenance {
            NodeProvenance::TemporalPipe { pipe } if pipe.hadamard => Some(id),
            _ => None,
        })
        .expect("Hadamard pipe lowers to a quantum node");
    let seam = |edge: bloq_ir::BloqEdgeRef<'_>| match edge.edge {
        BloqEdge::Quantum(quantum) => {
            assert_eq!(quantum.pipes.len(), 1);
            assert!(quantum.pipes[0].pipe.hadamard);
            assert!(quantum.pipes[0].padding.is_some());
            Some((edge.source, edge.target))
        }
        BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
    };
    let below = bloq
        .incoming(hadamard)
        .find_map(seam)
        .expect("Hadamard node has a padded parent seam");
    let above = bloq
        .outgoing(hadamard)
        .find_map(seam)
        .expect("Hadamard node has a padded child seam");
    let baseline = parsed_emit(&bloq);

    bloq.insert_memory_rounds(
        MemoryRoundTarget::Edge {
            path: LevelPath::default(),
            from: below.0,
            to: below.1,
        },
        2,
    )
    .expect("padding below the Hadamard verifies");
    bloq.insert_memory_rounds(
        MemoryRoundTarget::Edge {
            path: LevelPath::default(),
            from: above.0,
            to: above.1,
        },
        2,
    )
    .expect("padding above the Hadamard verifies");
    bloq.validate().expect("twice-padded program validates");
    let padded = parsed_emit(&bloq);
    assert_eq!(padded.num_observables(), baseline.num_observables());
    assert!(padded.num_detectors() > baseline.num_detectors());
}

/// Idling on a seam whose endpoints are spatially merged.
///
/// Two columns joined by `+X` pipes in both layers give a two-pipe seam under a
/// merged region. What crosses it is still each column's own unmerged patch —
/// the shared data row between the columns is prepared and read out inside its
/// own layer — so each pipe's padding is compiled at the cube's footprint with
/// the merge closed. Padding at the merged shape instead drives qubits the lower
/// node already measured out; the seam then fails to recompose, which is how
/// this used to be pinned as unspliceable.
#[test]
fn wait_on_a_spatially_merged_seam_preserves_observables_and_distance() {
    let graph = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n  1: ZXZ [1,0,0]\n  2: ZXZ [0,0,1]\n  3: ZXZ [1,0,1]\n  [0,0,0] -> +X\n  [0,0,1] -> +X\n  [0,0,0] -> +Z\n  [1,0,0] -> +Z\n",
    )
    .expect("two merged columns parse");
    let ctx = CompileContext::new(CompileConfig::new(DISTANCE));
    let mut bloq = ctx.compile(&graph).expect("merged columns compile").bloq;
    let baseline = parsed_emit(&bloq);
    let baseline_distance = noisy_graphlike_distance(&bloq);

    let seam = bloq.edges().find_map(|edge| match edge.edge {
        BloqEdge::Quantum(quantum) if quantum.pipes.len() == 2 => Some((edge.source, edge.target)),
        _ => None,
    });
    let (from, to) = seam.expect("merged columns produce a two-pipe Quantum seam");

    let rounds = 4;
    let padding = bloq
        .insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from,
                to,
            },
            rounds,
        )
        .expect("both pipes of the merged seam accept a decoder wait");
    bloq.validate().expect("padded merged program validates");
    assert_eq!(bloq[padding].memory_rounds(), Some(rounds));

    let spliced = parsed_emit(&bloq);
    assert_eq!(
        spliced.num_observables(),
        baseline.num_observables(),
        "observables unchanged"
    );
    // Two d=3 patches idling side by side, each contributing d*d-1 stabilizers.
    assert_eq!(
        spliced.num_detectors() - baseline.num_detectors(),
        2 * u64::from(rounds) * u64::from(DISTANCE * DISTANCE - 1),
        "padding adds exactly the rounds it was asked for, on both columns"
    );
    assert_detectors_deterministic(&bloq, "merged seam wait");
    assert_eq!(
        noisy_graphlike_distance(&bloq),
        baseline_distance,
        "graphlike distance unchanged"
    );
}

#[test]
fn misplaced_padding_is_rejected_and_program_untouched() {
    let mut bloq = compile_two_cubes(DISTANCE);
    let baseline = emit_bloq_stim(&bloq).expect("emit baseline Stim");

    let (from, to) = quantum_seam(&bloq);
    let entry = bloq
        .edges_between(from, to)
        .find_map(|edge| match edge.edge {
            BloqEdge::Quantum(quantum) => quantum.pipes.first().and_then(|seam| seam.padding),
            BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
        })
        .expect("compile recorded padding on the plain seam");
    let error = bloq
        .subdivide_quantum_edge(
            from,
            to,
            &[PaddingInstance {
                template: entry.one_round,
                offset: entry.offset + IVec2::new(1000, 0),
            }],
            1,
        )
        .expect_err("misplaced padding is rejected");
    assert!(
        matches!(error, EditError::InvalidQuantumEdge { .. }),
        "unexpected error: {error:?}"
    );

    assert_eq!(
        emit_bloq_stim(&bloq).expect("emit Stim after rejected splice"),
        baseline
    );
}
