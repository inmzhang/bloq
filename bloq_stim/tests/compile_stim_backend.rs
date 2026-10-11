use bloq_compile::{CompileConfig, CompileContext, validate_bloq_qubit_layout_for_source};
use bloq_ir::{Bloq, LevelPath, MemoryRoundTarget};
use bloq_stim::{
    BloqStimOptions, emit_bloq_stim, emit_bloq_stim_with, emit_bloq_stim_with_stim_noise,
};
use bloq_test::{
    CompileReadyCase, TestCaseError, TestFixture, compile_fixtures, select_test_cases_for_fixture,
};
use glam::ivec3;
use stim::{Circuit, DemInstructionTarget, DemInstructionType, DemItem};

mod common;
use common::uniform_depolarizing;

fn make_context(distance: u32) -> CompileContext {
    CompileContext::new(CompileConfig::new(distance))
}

fn validate_backend_case(
    case: &CompileReadyCase,
    distance: u32,
    ctx: &CompileContext,
) -> Result<(), String> {
    let expected_observables = case
        .expected_observables()
        .map_err(|error| error.to_string())?;
    let artifacts = ctx
        .compile(&case.graph)
        .map_err(|error| format!("{}@d{distance}: compile graph: {error}", case.id()))?;
    validate_bloq_case(case, distance, &artifacts.bloq, expected_observables)?;
    let stim_text = emit_bloq_stim(&artifacts.bloq)
        .map_err(|error| format!("{}@d{distance}: emit Stim: {error}", case.id()))?;
    let stim: Circuit = stim_text
        .parse()
        .map_err(|error| format!("{}@d{distance}: parse emitted Stim: {error}", case.id()))?;

    let actual_observables = stim.num_observables() as usize;
    if actual_observables != expected_observables {
        return Err(format!(
            "{}@d{distance}: expected {} observables from Bloq Stim, got {actual_observables}",
            case.id(),
            expected_observables,
        ));
    }

    let noisy_stim_text = emit_bloq_stim_with_stim_noise(&artifacts.bloq, &uniform_depolarizing())
        .map_err(|error| format!("{}@d{distance}: emit noisy Stim: {error}", case.id()))?;
    let noisy_stim: Circuit = noisy_stim_text.parse().map_err(|error| {
        format!(
            "{}@d{distance}: parse noisy emitted Stim: {error}",
            case.id()
        )
    })?;
    validate_matchable_wall_dem(case, distance, &noisy_stim)?;
    let actual = common::graphlike_distance(&noisy_stim).map_err(|error| {
        format!(
            "{}@d{distance}: noisy Stim shortest_graphlike_error: {error}",
            case.id()
        )
    })?;
    // Distance preserving is the rule, not an invariant: the spatial Hadamard
    // wall trades it away, so the corpus supplies the weight each case must
    // show (`TestCase::expected_graphlike_distance`). The comparison stays an
    // equality — a wall case that silently improved has changed just as much as
    // one that regressed, and either way the frozen table is now a lie.
    let expected = usize::try_from(case.test_case.expected_graphlike_distance(distance))
        .map_err(|_| format!("{}@d{distance}: expected distance overflow", case.id()))?;
    if actual != expected {
        return Err(format!(
            "{}@d{distance}: expected noisy Stim distance {expected}, got {actual}",
            case.id()
        ));
    }
    Ok(())
}

fn validate_matchable_wall_dem(
    case: &CompileReadyCase,
    distance: u32,
    circuit: &Circuit,
) -> Result<(), String> {
    if !case.test_case.metadata.origins.iter().any(|origin| {
        matches!(
            origin.fixture,
            TestFixture::CZ | TestFixture::SpatialHadamard
        )
    }) {
        return Ok(());
    }

    let raw = circuit.detector_error_model().map_err(|error| {
        format!(
            "{}@d{distance}: derive spatial-Hadamard DEM: {error}",
            case.id()
        )
    })?;
    for item in raw.flattened() {
        let DemItem::Instruction(instruction) = item else {
            unreachable!("flattened DEM contains no repeat blocks")
        };
        if instruction.r#type() != DemInstructionType::Error {
            continue;
        }
        let degree = instruction
            .target_groups()
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .filter(|target| {
                        matches!(
                            target,
                            DemInstructionTarget::DemTarget(target)
                                if target.is_relative_detector_id()
                        )
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        if degree > 4 {
            return Err(format!(
                "{}@d{distance}: spatial-Hadamard DEM contains a degree-{degree} fault",
                case.id()
            ));
        }
    }
    circuit
        .detector_error_model_with_options(true, false, false, 0.0, false, true)
        .map_err(|error| {
            format!(
                "{}@d{distance}: spatial-Hadamard DEM does not strictly decompose: {error}",
                case.id()
            )
        })?;
    Ok(())
}

fn validate_bloq_case(
    case: &CompileReadyCase,
    distance: u32,
    program: &Bloq,
    expected_observables: usize,
) -> Result<(), String> {
    validate_bloq_qubit_layout_for_source(program, &case.graph)
        .map_err(|error| format!("{}@d{distance}: validate qubit layout: {error}", case.id()))?;
    program.validate().map_err(|error| {
        format!(
            "{}@d{distance}: validate template instance refs: {error}",
            case.id()
        )
    })?;

    let actual_observables = observable_indices(program).len();
    if actual_observables != expected_observables {
        return Err(format!(
            "{}@d{distance}: expected {} observable indices in Bloq, got {actual_observables}",
            case.id(),
            expected_observables,
        ));
    }
    Ok(())
}

fn observable_indices(program: &Bloq) -> std::collections::BTreeSet<u32> {
    // Observables lower to standalone Observable nodes (IR spec §4); collect their
    // indices instead of the removed QuantumNode side table.
    let mut indices = std::collections::BTreeSet::new();
    for (_, node) in program.nodes() {
        if let Some(bloq_ir::ClassicalNode::Observable {
            index: Some(index), ..
        }) = node.try_classical()
        {
            indices.insert(*index);
        }
    }
    indices
}

fn backend_fixture_failures(
    fixture: TestFixture,
    distance: u32,
    seen: &mut std::collections::BTreeSet<String>,
) -> Vec<String> {
    let cases = match backend_integration_cases_for_fixture(fixture) {
        Ok(cases) if !cases.is_empty() => cases,
        Ok(_) => {
            return vec![format!(
                "{}@d{distance}: fixture selected no cases",
                fixture.id()
            )];
        }
        Err(error) => {
            return vec![format!(
                "{}@d{distance}: load fixture cases: {error}",
                fixture.id()
            )];
        }
    };
    let ctx = make_context(distance);
    let mut failures = Vec::new();
    for case in &cases {
        // The corpus merges fixture origins; each canonical case needs the
        // same layout, emission and distance checks only once per sweep.
        if seen.insert(case.id().to_string())
            && let Err(error) = validate_backend_case(case, distance, &ctx)
        {
            failures.push(error);
        }
    }
    failures
}

fn run_backend_fixtures(fixtures: impl IntoIterator<Item = TestFixture>, distance: u32) {
    let mut count = 0;
    let mut failures = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for fixture in fixtures {
        count += 1;
        failures.extend(backend_fixture_failures(fixture, distance, &mut seen));
    }

    assert!(count > 0, "backend fixture selection was empty");
    assert!(
        failures.is_empty(),
        "compile_stim_backend failures at d={distance}:\n{}",
        failures.join("\n")
    );
}

fn backend_integration_cases_for_fixture(
    fixture: TestFixture,
) -> Result<Vec<CompileReadyCase>, TestCaseError> {
    select_test_cases_for_fixture(fixture)?
        .into_iter()
        .map(CompileReadyCase::from_test_case)
        .collect()
}

fn clifford_fixtures() -> impl Iterator<Item = TestFixture> {
    compile_fixtures().iter().copied().filter(|fixture| {
        fixture.is_clifford() && !fixture.in_category(bloq_graph::GalleryCategory::Adaptive)
    })
}

#[test]
fn spatial_split_readout_padding_preserves_distance() {
    let graph = bloq_graph::BlockGraph::from_blog_text(
        "BLOG 1.0
module main {
  0: XXZ [0, 0, 0]
  1: ZXZ [-1, 0, 0]
  2: XZZ [0, 1, 0]
  3: ZXZ [1, 0, 0]
  4: X [1, 0, 1]
  0 -> 1
  0 -> 2
  0 -> 3
  3 -> 4
}",
    )
    .expect("minimal spatial-split graph parses")
    .flatten()
    .expect("minimal module flattens");
    for graph in [graph.clone(), graph.flip_xz_basis().unwrap()] {
        // Counts are extra rounds before the compiler's closing readout round.
        for (distance, extra_rounds) in [(3, 0), (5, 0), (7, 0), (9, 0), (11, 1), (15, 2)] {
            let mut program = make_context(distance).compile(&graph).unwrap().bloq;
            if extra_rounds > 0 {
                let to = program.node_by_block(ivec3(1, 0, 1)).unwrap();
                let from = program.top().quantum_input(to).unwrap();
                program
                    .insert_memory_rounds(
                        MemoryRoundTarget::Edge {
                            path: LevelPath::default(),
                            from,
                            to,
                        },
                        extra_rounds,
                    )
                    .unwrap();
                program.validate().unwrap();
            }
            let noisy: Circuit = emit_bloq_stim_with_stim_noise(&program, &uniform_depolarizing())
                .unwrap()
                .parse()
                .unwrap();
            // The only logical surface joins the left and upper arms through
            // the junction. Before the closing round, its d7 fault weight was 6.
            assert_eq!(noisy.num_observables(), 1);
            assert_eq!(
                common::graphlike_distance(&noisy).unwrap(),
                distance as usize,
                "spatial split followed by X/Z readout at d{distance} with {extra_rounds} extra rounds"
            );
        }
    }
}

#[test]
fn all_clifford_fixtures_d3() {
    run_backend_fixtures(clifford_fixtures(), 3);
}

#[test]
fn ghz_walks_emit_deterministic_aligned_stim() {
    for distance in [3, 5] {
        let ctx = make_context(distance);
        for case in backend_integration_cases_for_fixture(TestFixture::GHZSlideThenGlide).unwrap() {
            for program in [
                ctx.compile(&case.graph).unwrap().bloq,
                bloq_compile::compile_clifford_proxy(
                    CompileConfig::new(distance),
                    &case.graph,
                    &[],
                )
                .unwrap()
                .bloq,
            ] {
                let text =
                    emit_bloq_stim_with(&program, &BloqStimOptions::new().with_align_moments(true))
                        .unwrap();
                let aligned: Circuit = text.parse().unwrap();
                assert_eq!(
                    aligned.num_observables() as usize,
                    case.expected_observables().unwrap()
                );
                aligned.detector_error_model().unwrap();
            }
        }
    }
}

#[test]
#[ignore = "slow d5 Stim corpus distance sweep; run via just test-full"]
fn all_clifford_fixtures_d5() {
    run_backend_fixtures(clifford_fixtures(), 5);
}

// d7 emits the largest physical circuits and runs a full Stim distance search.
#[test]
#[ignore = "slow d7 stim distance search; run with --run-ignored all"]
fn all_clifford_fixtures_d7() {
    run_backend_fixtures(clifford_fixtures(), 7);
}

#[test]
#[ignore = "slow d9 spatial-Hadamard distance search"]
fn spatial_hadamard_graphlike_distance_d9() {
    run_backend_fixtures([TestFixture::CZ, TestFixture::SpatialHadamard], 9);
}

#[test]
#[ignore = "slow d7 undetectable-logical-error search; run with --run-ignored all"]
fn spatial_hadamard_undetectable_distance_d7() {
    let ctx = make_context(7);
    let mut seen = 0;
    for fixture in [TestFixture::CZ, TestFixture::SpatialHadamard] {
        for case in select_test_cases_for_fixture(fixture).expect("fixture cases") {
            if !matches!(
                case.id(),
                "cz[fill:0]"
                    | "cz[fill:1]"
                    | "cz[fill:1][rotate:Y:2]"
                    | "cz[fill:1][rotate:Z:1]"
                    | "cz[fill:1][rotate:Z:3]"
                    | "spatial_hadamard[base]"
                    | "spatial_hadamard[base][rotate:Y:2]"
                    | "spatial_hadamard[base][rotate:Z:1]"
                    | "spatial_hadamard[base][rotate:Z:3]"
            ) {
                continue;
            }
            let case = CompileReadyCase::from_test_case(case).expect("selected wall case builds");
            let bloq = ctx.compile(&case.graph).expect("compile").bloq;
            let noisy: Circuit = emit_bloq_stim_with_stim_noise(&bloq, &uniform_depolarizing())
                .expect("emit noisy Stim")
                .parse()
                .expect("parse noisy Stim");
            let actual = noisy
                .search_for_undetectable_logical_errors(3, 3, false, true)
                .expect("search")
                .len();
            let expected = case.test_case.expected_graphlike_distance(7) as usize;
            assert_eq!(actual, expected, "{}: measured {actual}", case.id());
            seen += 1;
        }
    }
    assert_eq!(seen, 9, "all representative wall profiles were measured");
}
// The data-driven suite also runs each fixture's `[open]` variant when that form is a
// compilable temporal-port graph; the ports compile to noiseless boundary MPPs,
// so the decoder distance still equals `d`. The two spatial Hadamard fixtures
// (`cz`, `spatial_hadamard`) are the exception in the other direction: the wall
// is not distance preserving, and they pin the measured weights the corpus
// carries instead.
