//! Circuit-distance regression checks for native non-Clifford or adaptive
//! gallery entries in the shared corpus, excluding analysis-only graphs and
//! the ten-bit adder scaling case. The three-bit adder runs at d3/d5 only;
//! d7 retains the other gallery compositions without repeating its costly sweep.
//!
//! `compile_clifford_distance` restricts the certified source relation and
//! rewrites every dynamic element into a Clifford
//! stand-in — T blocks become perfect MPP input ports, selective blocks are
//! statically pinned, the action track is dropped — yielding a determined
//! circuit the static backend emits. This measures the **composition**
//! graphlike distance (how the prepared magic state is protected through seams,
//! downstream cubes, and pinned branches). Hyperedge mechanisms are checked in
//! focused backend tests. The cultivation/escape internal distances are owned
//! by the standalone escape harness; graph-level ZX maps remain the
//! logical-correctness oracle.
//!
//! Reachable structural assignments are exhaustive up to 16 combinations.
//! Larger structural domains use hashed samples plus both domain endpoints;
//! selective domains use samples plus reachable uniform corners.
//! The random selective-sample budget is shared across structural projections.
//! The minimum over samples is a regression check, not an exhaustive proof.

use bloq_compile::{CompileConfig, compile_clifford_distance, compile_clifford_proxy};
use bloq_graph::{BlockGraph, BlockKind, GalleryCategory, GalleryItem, PauliBasis};
use bloq_ir::{Bloq, BloqEdge, BloqNode, ClassicalAssignment, ClassicalNode};
use bloq_stim::emit_bloq_stim_with_stim_noise;
use bloq_test::{TestFixture, compile_fixtures, select_test_cases_for_fixture};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use stim::{Circuit, CircuitInstruction, CircuitItem, Flow, Gate};

mod common;
use common::uniform_depolarizing;

fn pin_combos(graph: &BlockGraph, samples: usize, first_seed: u64) -> Vec<Vec<bool>> {
    // Match the compiler's ordered_blocks pin order.
    let mut sites = graph
        .blocks()
        .filter(|block| block.kind().is_selective())
        .collect::<Vec<_>>();
    sites.sort_unstable_by_key(|block| {
        let pos = block.pos();
        (pos.z, pos.x, pos.y)
    });
    let targets = sites.iter().map(|block| block.pos()).collect::<Vec<_>>();
    let actions = graph.action_graph();
    if let Ok(domain) = actions.resolve_value_domain_bounded(&targets, samples) {
        return domain.values().to_vec();
    }
    let mut combos = [vec![false; sites.len()], vec![true; sites.len()]]
        .into_iter()
        .filter(|pins| {
            actions
                .resolve_values_are_reachable(&targets, pins)
                .expect("fixture reachability is within the Boolean budget")
        })
        .collect::<Vec<_>>();
    for seed in first_seed..first_seed + samples as u64 {
        let (_, replacements) = graph
            .randomly_resolve_selectives(seed)
            .expect("flat fixture resolves selective bases");
        let pins = sites
            .iter()
            .map(|block| {
                let BlockKind::Selective(kind) = block.kind() else {
                    unreachable!("sampled sites are selective blocks")
                };
                let chosen = match replacements[*block].kind() {
                    BlockKind::Y => PauliBasis::Y,
                    BlockKind::Measurement(basis) => basis.into(),
                    _ => unreachable!("selective resolves to a measurement"),
                };
                chosen == kind.pauli_if_true()
            })
            .collect::<Vec<_>>();
        if !combos.contains(&pins) {
            combos.push(pins);
        }
    }
    combos
}

/// Minimum graphlike logical-error weight across the swept pin combos.
fn min_proxy_distance(graph: &BlockGraph, distance: u32, samples: usize, first_seed: u64) -> usize {
    pin_combos(
        &graph.flatten().expect("proxy source flattens"),
        samples,
        first_seed,
    )
    .into_iter()
    .map(|pins| {
        let config = CompileConfig::new(distance);
        let bloq = compile_clifford_distance(config, graph, &pins)
            .unwrap_or_else(|error| panic!("d={distance} pins={pins:?}: proxy compiles: {error}"))
            .bloq;
        let noisy: Circuit = emit_bloq_stim_with_stim_noise(&bloq, &uniform_depolarizing())
            .expect("proxy program emits")
            .parse()
            .expect("noisy proxy Stim parses");
        assert!(
            noisy.num_observables() > 0,
            "d={distance} pins={pins:?}: at least one observable \
                 survives the pin (a distance over zero observables is vacuous)"
        );
        common::graphlike_distance(&noisy).unwrap_or_else(|error| {
            panic!("d={distance} pins={pins:?}: graphlike-error search: {error}")
        })
    })
    .min()
    .expect("at least one combo")
}

fn gallery_proxy_distances(distance: u32, samples: usize) {
    let mut checked = 0;
    let mut failures = Vec::new();
    for fixture in compile_fixtures().iter().copied().filter(|fixture| {
        (fixture.in_category(GalleryCategory::NonClifford)
            || fixture.in_category(GalleryCategory::Adaptive))
            && !fixture.in_category(GalleryCategory::AnalysisOnly)
            // The three-bit adder covers this family at d3/d5; its d7 sweep
            // adds costly carry-chain stress beyond those distance checks.
            && *fixture != TestFixture::TenBitAdder
            && !(distance == 7 && *fixture == TestFixture::ThreeBitAdder)
    }) {
        let case = select_test_cases_for_fixture(fixture)
            .expect("fixture cases")
            .into_iter()
            .find(|case| {
                case.metadata.origins.iter().any(|origin| {
                    origin.fixture == fixture
                        && origin.fill_variant.is_none()
                        && !origin.flip_xz_basis
                        && origin.rotation.is_none()
                })
            })
            .expect("native gallery case in shared corpus");
        eprintln!("{} d{distance}", case.id());
        let program = case.build();
        let graph = program.flatten().expect("gallery source flattens");
        // ponytail: enumerate selector tuples, never all projected graphs;
        // sample the Boolean domain directly if galleries outgrow this bound.
        let mut assignments = graph
            .branch_assignments_up_to(usize::MAX)
            .expect("reachable structural assignments");
        if assignments.len() > 16 {
            let first = assignments.remove(0);
            let last = assignments
                .pop()
                .expect("removing the first of more than sixteen assignments leaves a last one");
            assignments.sort_by_cached_key(|assignment| {
                let mut hash = DefaultHasher::new();
                assignment.hash(&mut hash);
                hash.finish()
            });
            assignments.truncate(samples);
            assignments.extend([first, last]);
        }
        let samples_per_projection = samples.div_ceil(assignments.len());
        let actual = assignments
            .iter()
            .enumerate()
            .map(|(index, assignment)| {
                let projection = graph
                    .project_branches_deferred(assignment.iter().copied())
                    .expect("reachable structural projection");
                let projected = if program.has_module_structure() {
                    let mut modules = program
                        .modules()
                        .map(BlockGraph::clone_local_definition)
                        .collect::<Vec<_>>();
                    let root = modules
                        .iter_mut()
                        .find(|module| module.name == program.name)
                        .expect("a graph contains its named root definition");
                    let body = root
                        .project_branches_in_definition(assignment.iter().copied())
                        .expect("gallery structural branches belong to the root");
                    root.replace_local_body(body);
                    BlockGraph::from_definitions(modules)
                        .expect("reachable root projection preserves the module interface")
                } else {
                    projection
                };
                min_proxy_distance(
                    &projected,
                    distance,
                    samples_per_projection,
                    (index * samples_per_projection) as u64,
                )
            })
            .min()
            .expect("at least one structural projection");
        let expected = match fixture {
            // Authored short TELS layers deliberately use ceil(2d/3) rounds.
            TestFixture::CCZFactoryWithTels => (2 * distance).div_ceil(3),
            // Spatial-Hadamard corrections retain the wall's LIM-016 loss.
            TestFixture::CCZGateTeleport => match distance {
                3 => 2,
                5 => 4,
                7 => 5,
                _ => panic!("measure the teleportation proxy at d{distance} first"),
            },
            _ => distance,
        };
        if actual != expected as usize {
            failures.push(format!(
                "{} d{distance}: expected {expected}, measured {actual}",
                case.id()
            ));
        }
        checked += 1;
    }
    assert!(checked >= 13, "only {checked} gallery entries checked");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn non_clifford_and_dynamic_gallery_distance_d3() {
    gallery_proxy_distances(3, 2);
}

#[test]
#[ignore = "slow d5 graphlike distance searches; run with --release --run-ignored all"]
fn non_clifford_and_dynamic_gallery_distance_d5() {
    gallery_proxy_distances(5, 6);
}

#[test]
#[ignore = "slow d7 distance searches; run with --run-ignored all"]
fn non_clifford_and_dynamic_gallery_distance_d7() {
    gallery_proxy_distances(7, 2);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ParityTerm {
    Record(i64),
    Boundary {
        instruction: usize,
        qubit: u32,
        z: bool,
    },
}

type Parity = BTreeSet<ParityTerm>;

fn xor_parity<T: Clone + Ord>(left: &mut BTreeSet<T>, right: &BTreeSet<T>) {
    for term in right {
        if !left.insert(term.clone()) {
            left.remove(term);
        }
    }
}

type ParityBasis = BTreeMap<ParityTerm, (Parity, BTreeSet<usize>)>;

fn parity_span(rows: impl IntoIterator<Item = Parity>) -> ParityBasis {
    let mut basis = ParityBasis::new();
    for (index, mut row) in rows.into_iter().enumerate() {
        let mut coefficients = BTreeSet::from([index]);
        for (pivot, (other, indices)) in &basis {
            if row.contains(pivot) {
                xor_parity(&mut row, other);
                xor_parity(&mut coefficients, indices);
            }
        }
        if let Some(pivot) = row.first().cloned() {
            basis.insert(pivot, (row, coefficients));
        }
    }
    basis
}

fn decompose_parity(mut row: Parity, basis: &ParityBasis) -> BTreeSet<usize> {
    let mut coefficients = BTreeSet::new();
    for (pivot, (other, indices)) in basis {
        if row.contains(pivot) {
            xor_parity(&mut row, other);
            xor_parity(&mut coefficients, indices);
        }
    }
    assert!(
        row.is_empty(),
        "observable is outside the other detector/observable space: {row:?}"
    );
    coefficients
}

struct CircuitRelations {
    physical: Vec<String>,
    detectors: Vec<Parity>,
    observables: Vec<Parity>,
    detector_signs: Vec<bool>,
    observable_signs: Vec<bool>,
}

/// Physical instructions fix record positions and Pauli insertion cuts. Keep
/// reference signs separate: Pauli phases do not obey unsigned row addition.
fn circuit_relations(circuit: &Circuit) -> CircuitRelations {
    let mut physical = Vec::new();
    let mut detectors = Vec::new();
    let mut observables = vec![Parity::new(); circuit.num_observables() as usize];
    let mut measurements = 0i64;
    for item in circuit.flattened() {
        let CircuitItem::Instruction(instruction) = item else {
            unreachable!("flattened circuit has no repeat blocks")
        };
        if matches!(instruction.name(), "DETECTOR" | "OBSERVABLE_INCLUDE") {
            let mut row = Parity::new();
            for &target in instruction.targets() {
                let mut terms = Parity::new();
                if target.is_measurement_record_target() {
                    let record = measurements + i64::from(target.value());
                    assert!(record >= 0 && record < measurements);
                    terms.insert(ParityTerm::Record(record));
                } else {
                    assert!(target.is_pauli_target());
                    let qubit = target.qubit_value().expect("Pauli target names a qubit");
                    for z in [false, true] {
                        if target.is_y_target()
                            || (!z && target.is_x_target())
                            || (z && target.is_z_target())
                        {
                            terms.insert(ParityTerm::Boundary {
                                instruction: physical.len(),
                                qubit,
                                z,
                            });
                        }
                    }
                }
                xor_parity(&mut row, &terms);
            }
            if instruction.name() == "DETECTOR" {
                detectors.push(row);
            } else {
                xor_parity(&mut observables[instruction.gate_args()[0] as usize], &row);
                continue;
            }
        }
        measurements += i64::try_from(instruction.num_measurements())
            .expect("small comparison instruction has an i64 measurement count");
        physical.push(instruction.to_string());
    }
    CircuitRelations {
        physical,
        detectors,
        observables,
        detector_signs: circuit.reference_detector_signs(),
        observable_signs: circuit.reference_observable_signs(),
    }
}

/// Overlay both annotation sets without moving an annotation across a physical
/// instruction. Observable annotations produce no records or quantum operations.
fn common_annotated_circuit(left: &Circuit, right: &Circuit) -> (Circuit, u64) {
    fn split(
        circuit: &Circuit,
        offset: u64,
    ) -> (
        Vec<CircuitInstruction>,
        BTreeMap<usize, Vec<CircuitInstruction>>,
    ) {
        let mut physical = Vec::new();
        let mut annotations = BTreeMap::<usize, Vec<CircuitInstruction>>::new();
        for item in circuit.flattened() {
            let CircuitItem::Instruction(instruction) = item else {
                unreachable!("flattened circuit has no repeat blocks")
            };
            if instruction.name() == "OBSERVABLE_INCLUDE" {
                annotations.entry(physical.len()).or_default().push(
                    CircuitInstruction::new(
                        Gate::OBSERVABLE_INCLUDE,
                        instruction.targets().iter().copied(),
                        [instruction.gate_args()[0] + offset as f64],
                        instruction.tag(),
                    )
                    .expect("reindexing preserves a valid observable annotation"),
                );
            } else {
                physical.push(instruction);
            }
        }
        (physical, annotations)
    }
    let offset = left.num_observables();
    let (physical, mut left_annotations) = split(left, 0);
    let (right_physical, mut right_annotations) = split(right, offset);
    assert_eq!(
        physical, right_physical,
        "comparison must share every physical instruction"
    );
    let mut combined = Circuit::new();
    for cut in 0..=physical.len() {
        for instruction in left_annotations
            .remove(&cut)
            .into_iter()
            .flatten()
            .chain(right_annotations.remove(&cut).into_iter().flatten())
        {
            combined
                .append_operation(instruction)
                .expect("common circuit accepts the reindexed annotation");
        }
        if let Some(instruction) = physical.get(cut) {
            combined
                .append_operation(instruction.clone())
                .expect("common circuit accepts the unchanged physical instruction");
        }
    }
    assert!(left_annotations.is_empty() && right_annotations.is_empty());
    assert_eq!(combined.num_measurements(), left.num_measurements());
    (combined, offset)
}

fn signed_flow(sign: bool, terms: Vec<String>) -> Flow {
    let rhs = if terms.is_empty() {
        "1".to_owned()
    } else {
        terms.join(" xor ")
    };
    Flow::new(&format!("{}1 -> {rhs}", if sign { "-" } else { "" }))
        .expect("comparison terms form a signed Stim flow")
}

/// Gaussian elimination supplies only the support rewrite. Stim checks the
/// resulting signed identity, including Pauli products at different time cuts.
fn observable_rebase_flows(
    source: &CircuitRelations,
    target: &CircuitRelations,
    source_offset: u64,
    target_offset: u64,
    measurements: u64,
) -> Vec<Flow> {
    let basis = parity_span(source.detectors.iter().chain(&source.observables).cloned());
    target
        .observables
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let mut terms = vec![format!("obs[{}]", target_offset + index as u64)];
            let mut sign = target.observable_signs[index];
            for coefficient in decompose_parity(row.clone(), &basis) {
                if coefficient < source.detectors.len() {
                    sign ^= source.detector_signs[coefficient];
                    for term in &source.detectors[coefficient] {
                        let ParityTerm::Record(record) = term else {
                            panic!("detectors contain only classical records")
                        };
                        terms.push(format!(
                            "rec[{}]",
                            record
                                - i64::try_from(measurements).expect(
                                    "small comparison circuit has an i64 measurement count"
                                )
                        ));
                    }
                } else {
                    let observable = coefficient - source.detectors.len();
                    sign ^= source.observable_signs[observable];
                    terms.push(format!("obs[{}]", source_offset + observable as u64));
                }
            }
            signed_flow(sign, terms)
        })
        .collect()
}

struct FrameProbe {
    key: ([i32; 3], bool),
    raw_observable: Option<u32>,
    sign: bool,
    decoder_observables: BTreeSet<u32>,
}

fn noisy_circuit(program: &Bloq) -> Circuit {
    emit_bloq_stim_with_stim_noise(program, &uniform_depolarizing())
        .expect("comparison program emits with uniform depolarizing noise")
        .parse()
        .expect("emitted comparison circuit parses as Stim")
}

fn frame_probes(program: &Bloq) -> (Circuit, Vec<FrameProbe>) {
    let mut probed = program.clone();
    let mut next = program
        .nodes()
        .filter_map(|(_, node)| match node.try_classical() {
            Some(ClassicalNode::Observable { index, .. }) => *index,
            _ => None,
        })
        .max()
        .map_or(0, |index| index + 1);
    let mut probes = Vec::new();
    for frame in program.output_frames() {
        for (z, id) in [(false, frame.x), (true, frame.z)] {
            let resolved = program
                .resolve_classical(id, ClassicalAssignment::Uniform(false))
                .expect("static proxy frame resolves to a physical and decoder parity");
            let raw_observable = if resolved.measurements.is_empty() {
                None
            } else {
                let index = next;
                next += 1;
                let probe = probed.add_node(BloqNode::classical(ClassicalNode::Observable {
                    index: Some(index),
                    measurements: resolved.measurements.into_iter().collect(),
                    operators: Vec::new(),
                }));
                probed.add_edge(id, probe, BloqEdge::Order);
                Some(index)
            };
            probes.push(FrameProbe {
                key: (frame.port.to_array(), z),
                raw_observable,
                sign: resolved.sign,
                decoder_observables: resolved.decoder_observables,
            });
        }
    }
    (noisy_circuit(&probed), probes)
}

/// A decoder flip is the physical sheet relative to its reference value. Raw
/// frame records remain raw; only decoder sheets receive reference offsets.
fn frame_sheet(
    probe: &FrameProbe,
    relations: &CircuitRelations,
    offset: u64,
) -> (Parity, bool, Vec<String>) {
    let mut row = Parity::new();
    let mut sign = probe.sign;
    let mut terms = Vec::new();
    if let Some(index) = probe.raw_observable {
        xor_parity(&mut row, &relations.observables[index as usize]);
        terms.push(format!("obs[{}]", offset + u64::from(index)));
    }
    for &index in &probe.decoder_observables {
        xor_parity(&mut row, &relations.observables[index as usize]);
        sign ^= relations.observable_signs[index as usize];
        terms.push(format!("obs[{}]", offset + u64::from(index)));
    }
    (row, sign, terms)
}

#[test]
fn relation_distance_preserves_legacy_physical_and_observable_spaces() {
    for (gallery, pins) in [
        (GalleryItem::CNOT, &[][..]),
        (GalleryItem::T, &[false][..]),
        (GalleryItem::T, &[true][..]),
        // Signed Y resources and two temporal Hadamards exercise reference offsets.
        (GalleryItem::THTH, &[false, false][..]),
    ] {
        let graph = gallery.build();
        let config = CompileConfig::new(3);
        let legacy = compile_clifford_proxy(config, &graph, pins).unwrap().bloq;
        let relation = compile_clifford_distance(config, &graph, pins)
            .unwrap()
            .bloq;
        let repeated = compile_clifford_distance(config, &graph, pins)
            .unwrap()
            .bloq;
        relation.validate().unwrap();
        assert_eq!(
            relation.to_binary(),
            repeated.to_binary(),
            "{gallery:?} pins={pins:?}"
        );
        let (left_probed, left_frames) = frame_probes(&legacy);
        let (right_probed, right_frames) = frame_probes(&relation);
        let legacy = noisy_circuit(&legacy);
        let relation = noisy_circuit(&relation);
        let left = circuit_relations(&legacy);
        let right = circuit_relations(&relation);
        // Exact detector presentation and noise locations preserve graphlike edges.
        assert_eq!(left.physical, right.physical, "{gallery:?} pins={pins:?}");
        let detector_rank = parity_span(left.detectors.clone()).len();
        let left_span = parity_span(left.detectors.iter().chain(&left.observables).cloned());
        assert!(
            left_span.len() > detector_rank,
            "{gallery:?}: vacuous observable quotient"
        );
        let left_probe_relations = circuit_relations(&left_probed);
        let right_probe_relations = circuit_relations(&right_probed);
        assert_eq!(
            left.physical, left_probe_relations.physical,
            "frame probes must not change the physical program"
        );
        assert_eq!(
            right.physical, right_probe_relations.physical,
            "frame probes must not change the physical program"
        );
        let (common, right_offset) = common_annotated_circuit(&left_probed, &right_probed);
        let mut flows =
            observable_rebase_flows(&left, &right, 0, right_offset, common.num_measurements());
        flows.extend(observable_rebase_flows(
            &right,
            &left,
            right_offset,
            0,
            common.num_measurements(),
        ));
        assert_eq!(left_frames.len(), right_frames.len());
        for (left, right) in left_frames.iter().zip(&right_frames) {
            assert_eq!(left.key, right.key);
            let (left_row, left_sign, mut terms) = frame_sheet(left, &left_probe_relations, 0);
            let (right_row, right_sign, right_terms) =
                frame_sheet(right, &right_probe_relations, right_offset);
            assert_eq!(
                left_row, right_row,
                "{gallery:?} {:?}: corrected frame sheet",
                left.key
            );
            terms.extend(right_terms);
            flows.push(signed_flow(left_sign ^ right_sign, terms));
        }
        assert!(
            common.has_all_flows(&flows, false).unwrap(),
            "{gallery:?} pins={pins:?}: signed observable/frame rewrites: {flows:?}"
        );
        assert!(relation.num_observables() > 0);
        assert_eq!(
            common::graphlike_distance(&legacy).unwrap(),
            common::graphlike_distance(&relation).unwrap()
        );
    }
}

#[test]
fn spatial_cube_diagonal_hooks_preserve_proxy_distance() {
    use bloq_graph::GalleryItem::{CCZInjectedAnd, CCZInjectedMaj};

    // AND and MAJ expose X and Z hook shortcuts on the lower diagonals.
    // The second AND pin also protects the opposite, upper-right junction.
    for (fixture, pins) in [
        (CCZInjectedAnd, [false, true, true, false]),
        (CCZInjectedAnd, [false, true, false, true]),
        (CCZInjectedMaj, [true, false, false, true]),
    ] {
        let bloq = compile_clifford_proxy(CompileConfig::new(5), &fixture.build(), &pins)
            .unwrap()
            .bloq;
        let noisy: Circuit = emit_bloq_stim_with_stim_noise(&bloq, &uniform_depolarizing())
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            common::graphlike_distance(&noisy).unwrap(),
            5,
            "{fixture:?} pins={pins:?}"
        );
    }
}
