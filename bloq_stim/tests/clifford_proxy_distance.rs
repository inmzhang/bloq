//! Circuit-distance regression checks for native non-Clifford or adaptive
//! gallery entries in the shared corpus, excluding analysis-only graphs and
//! the ten-bit adder scaling case. The three-bit adder runs at d3/d5 only;
//! d7 retains the other gallery compositions without repeating its costly sweep.
//!
//! `compile_clifford_proxy` restricts the certified source relation and
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

use bloq_compile::{CompileConfig, compile_clifford_proxy};
use bloq_graph::{BlockGraph, BlockKind, GalleryCategory, PauliBasis};
use bloq_stim::emit_bloq_stim_with_stim_noise;
use bloq_test::{TestFixture, compile_fixtures, select_test_cases_for_fixture};
use std::hash::{DefaultHasher, Hash, Hasher};
use stim::Circuit;

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
        let bloq = compile_clifford_proxy(config, graph, &pins)
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
