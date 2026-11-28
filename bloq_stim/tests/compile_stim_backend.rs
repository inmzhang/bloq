use bloq_compile::{CompileConfig, CompileContext, validate_bloq_qubit_layout_for_source};
use bloq_graph::BlockGraph;
use bloq_ir::{Bloq, LevelPath, MemoryRoundTarget};
use bloq_stim::{
    BloqStimOptions, emit_bloq_stim, emit_bloq_stim_with, emit_bloq_stim_with_stim_noise,
};
use bloq_test::{
    CompileReadyCase, TestCaseError, TestFixture, compile_fixtures, select_test_cases_for_fixture,
};
use glam::ivec3;
use stim::{Circuit, CircuitItem, DemInstructionTarget, DemInstructionType, DemItem};

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
fn aligned_temporal_hadamard_pads_live_neighbor() {
    let hadamard = "BLOG 1.0\n0: XZX [0,0,0]\n1: ZXZ [0,0,1]\n0 -H> +Z\n";
    let peer = "2: XZX [1,0,0]\n3: XZX [1,0,1]\n2 -> +Z\n";
    let compile = |source: &str| {
        bloq_compile::compile(&BlockGraph::from_blog_text(source).unwrap(), 3).unwrap()
    };
    let emit = |program: &Bloq, align| -> Circuit {
        emit_bloq_stim_with(program, &BloqStimOptions::new().with_align_moments(align))
            .unwrap()
            .parse()
            .unwrap()
    };
    let program = compile(&format!("{hadamard}{peer}"));
    let aligned = emit(&program, true);

    assert_eq!(
        aligned.num_measurements(),
        emit(&program, false).num_measurements() + 8,
        "one d3 memory round on the ordinary temporal seam"
    );
    let hadamard_only = compile(hadamard);
    assert_eq!(aligned.num_ticks(), emit(&hadamard_only, true).num_ticks());

    let coords = program.sorted_layout_coords().unwrap();
    let hadamard_max_x = hadamard_only
        .sorted_layout_coords()
        .unwrap()
        .last()
        .unwrap()
        .x;
    let instructions: Vec<_> = (&aligned)
        .into_iter()
        .map(|item| match item {
            CircuitItem::Instruction(instruction) => instruction,
            CircuitItem::RepeatBlock(_) => panic!("aligned circuit is flat"),
        })
        .collect();
    let moments: Vec<_> = instructions.split(|op| op.name() == "TICK").collect();
    let transition = moments
        .iter()
        .position(|moment| moment.iter().any(|op| op.name() == "H"))
        .expect("the transversal Hadamard identifies the realignment");
    let shared = moments[transition - 5..transition]
        .iter()
        .chain(std::iter::once(&moments[transition + 1]));
    for (moment, gate) in shared.zip(["R", "CX", "CX", "CX", "CX", "M"]) {
        let targets = moment
            .iter()
            .filter(|op| op.name().starts_with(gate))
            .flat_map(stim::CircuitInstruction::targets)
            .filter_map(|target| target.qubit_value())
            .map(|qubit| coords[qubit as usize].x);
        let (left, right): (Vec<_>, Vec<_>) = targets.partition(|&x| x <= hadamard_max_x);
        assert!(
            !left.is_empty() && !right.is_empty(),
            "both patches share {gate}"
        );
    }
    aligned.detector_error_model().unwrap();
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
                    | "cz[fill:0][rotate:Y:2]"
                    | "cz[fill:0][rotate:Z:1]"
                    | "cz[fill:0][rotate:Z:3]"
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

fn mixed_wall_elbow(sx: i32, sy: i32, z: bool, ports: [i32; 2], gap: i32) -> BlockGraph {
    use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
    let (hub, x, y) = if z {
        (CubeKind::ZZX, CubeKind::ZXZ, CubeKind::XZZ)
    } else {
        (CubeKind::XXZ, CubeKind::XZX, CubeKind::ZXX)
    };
    let mut graph = BlockGraph::new();
    let x_hub = ivec3(gap * sx, 0, 0);
    let x_pos = x_hub + ivec3(sx, 0, 0);
    let y_pos = ivec3(0, sy, 0);
    for (pos, kind) in [(ivec3(0, 0, 0), hub), (x_pos, x), (y_pos, y)] {
        graph.add_block(Block::new(pos, BlockKind::Cube(kind)));
    }
    for step in 1..=gap {
        let along = ivec3(step * sx, 0, 0);
        graph.add_block(Block::new(along, BlockKind::Cube(hub)));
        let start = if sx < 0 {
            along
        } else {
            along - ivec3(1, 0, 0)
        };
        graph.add_pipe(Pipe::new(start, Direction::XPLUS));
    }
    for (pos, dz) in [x_pos, y_pos].into_iter().zip(ports) {
        let terminal = pos + ivec3(0, 0, dz);
        graph.add_block(Block::new(terminal, BlockKind::Port));
        graph.add_pipe(Pipe::new(
            if dz < 0 { terminal } else { pos },
            Direction::ZPLUS,
        ));
    }
    graph.add_pipe(Pipe::new(if sx < 0 { x_pos } else { x_hub }, Direction::XPLUS).with_hadamard());
    graph.add_pipe(
        Pipe::new(
            if sy < 0 { y_pos } else { ivec3(0, 0, 0) },
            Direction::YPLUS,
        )
        .with_hadamard(),
    );
    graph
}

/// Check the authored pipes before the instance merger can hide serialization.
fn assert_shared_wall_clock(program: &Bloq, pipe_count: usize) {
    use bloq_circuit::Op;
    let component = program
        .nodes()
        .filter_map(|(_, node)| node.try_quantum())
        .find(|node| {
            node.instances
                .iter()
                .filter(|i| matches!(i.provenance, bloq_ir::InstanceProvenance::Pipe { .. }))
                .count()
                == pipe_count
        })
        .expect("all physical wall pipes in one quantum node");
    let mut used = std::collections::HashSet::new();
    for instance in &component.instances {
        let circuit = &program.templates()[instance.template_id].circuit;
        let per_pipe = matches!(
            instance.provenance,
            bloq_ir::InstanceProvenance::Pipe { .. }
        );
        let mut slots = std::collections::BTreeSet::new();
        let mut moment = 0;
        for op in circuit
            .body(circuit.entry_body())
            .expect("entry body")
            .ops()
        {
            match op {
                Op::Tick => {
                    moment += 1;
                    if moment == 8 {
                        break;
                    }
                }
                Op::Measure { .. } => assert_eq!(moment, 7, "joint end-of-round readout"),
                Op::Gate { gate, qubits } if gate.two_qubit_bases().is_some() => {
                    slots.insert(moment);
                    for pair in qubits.as_chunks::<2>().0 {
                        assert_eq!((pair[0] - pair[1]).abs(), glam::IVec2::ONE, "local coupler");
                        for &q in pair {
                            assert!(
                                used.insert((moment, q + instance.offset)),
                                "two gates use the same qubit"
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if per_pipe {
            assert_eq!(
                slots,
                (1..=6).collect(),
                "each pipe executes in all six slots"
            );
        }
    }
}

#[test]
fn mixed_wall_ports_and_concurrent_slots() {
    for sx in [-1, 1] {
        for sy in [-1, 1] {
            for z in [false, true] {
                let graph = mixed_wall_elbow(sx, sy, z, [1, 1], 0);
                let program = make_context(3).compile_and_validate(&graph).unwrap().bloq;
                assert_shared_wall_clock(&program, 2);
                let text =
                    emit_bloq_stim_with_stim_noise(&program, &uniform_depolarizing()).unwrap();
                for copy in [
                    Bloq::from_text(&program.to_text()).unwrap(),
                    Bloq::from_binary(&program.to_binary()).unwrap(),
                ] {
                    assert_eq!(
                        emit_bloq_stim_with_stim_noise(&copy, &uniform_depolarizing()).unwrap(),
                        text
                    );
                }
                let proxy =
                    bloq_compile::compile_clifford_proxy(CompileConfig::new(3), &graph, &[])
                        .unwrap()
                        .bloq;
                let mut flat = program.clone();
                flat.flatten()
                    .expect("flatten wall rounds and flag records");
                flat.validate().expect("flattened wall program validates");
                for program in [&program, &proxy, &flat] {
                    let circuit: Circuit =
                        emit_bloq_stim_with_stim_noise(program, &uniform_depolarizing())
                            .unwrap()
                            .parse()
                            .unwrap();
                    assert_eq!(circuit.num_observables(), 2);
                    circuit
                        .detector_error_model_with_options(true, false, false, 0.0, false, true)
                        .unwrap();
                    assert_eq!(common::graphlike_distance(&circuit).unwrap(), 3);
                }
            }
        }
    }
}

#[test]
fn mixed_walls_with_input_ports_and_separated_walls() {
    for gap in [0, 1, 4] {
        for ports in [[-1, -1], [-1, 1], [1, -1], [1, 1]] {
            for (sx, sy, z) in [(-1, -1, false), (1, 1, true)] {
                let graph = mixed_wall_elbow(sx, sy, z, ports, gap);
                let ctx = make_context(3);
                let label = format!("gap={gap} ports={ports:?} walls={sx},{sy},{z}");
                let program = ctx
                    .compile_and_validate(&graph)
                    .unwrap_or_else(|e| panic!("{label}: {e}"))
                    .bloq;
                assert_shared_wall_clock(&program, 2);
                let c: Circuit = emit_bloq_stim_with_stim_noise(&program, &uniform_depolarizing())
                    .unwrap()
                    .parse()
                    .unwrap();
                c.detector_error_model()
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(common::graphlike_distance(&c).unwrap(), 3, "{label}");
            }
        }
    }
}

#[test]
fn four_wall_pipes_execute_on_one_clock() {
    use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
    for z in [false, true] {
        let mut graph = mixed_wall_elbow(1, 1, z, [1, 1], 0);
        let arms = [
            (
                ivec3(-1, 0, 0),
                Direction::XPLUS,
                if z { CubeKind::ZXZ } else { CubeKind::XZX },
            ),
            (
                ivec3(0, -1, 0),
                Direction::YPLUS,
                if z { CubeKind::XZZ } else { CubeKind::ZXX },
            ),
        ];
        for (pos, axis, kind) in arms {
            graph.add_block(Block::new(pos, BlockKind::Cube(kind)));
            graph.add_pipe(Pipe::new(pos, axis).with_hadamard());
            graph.add_block(Block::new(pos + ivec3(0, 0, 1), BlockKind::Port));
            graph.add_pipe(Pipe::new(pos, Direction::ZPLUS));
        }
        let program = make_context(3).compile_and_validate(&graph).unwrap().bloq;
        assert_shared_wall_clock(&program, 4);
        let circuit: Circuit = emit_bloq_stim_with_stim_noise(&program, &uniform_depolarizing())
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(circuit.num_observables(), 4);
        circuit
            .detector_error_model_with_options(true, false, false, 0.0, false, true)
            .unwrap();
        assert_eq!(common::graphlike_distance(&circuit).unwrap(), 3);
    }
}

fn three_arm_elbow(sx: i32, sy: i32, z: bool) -> BlockGraph {
    use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
    let mut graph = mixed_wall_elbow(sx, sy, z, [1, 1], 0);
    let third = ivec3(-sx, 0, 0);
    graph.add_block(Block::new(
        third,
        BlockKind::Cube(if z { CubeKind::XZX } else { CubeKind::ZXZ }),
    ));
    graph.add_pipe(Pipe::new(
        if sx == 1 { third } else { ivec3(0, 0, 0) },
        Direction::XPLUS,
    ));
    graph.add_block(Block::new(third + ivec3(0, 0, 1), BlockKind::Port));
    graph.add_pipe(Pipe::new(third, Direction::ZPLUS));
    graph
}

/// Logical surfaces can use one or both walls at a three-way junction.
#[test]
fn two_wall_pipes_at_a_three_arm_junction() {
    for z in [false, true] {
        for sx in [-1, 1] {
            for sy in [-1, 1] {
                let graph = three_arm_elbow(sx, sy, z);
                let program = make_context(3)
                    .compile_and_validate(&graph)
                    .expect("three-arm wall junction compiles")
                    .bloq;
                assert_shared_wall_clock(&program, 2);
                let noisy: Circuit =
                    emit_bloq_stim_with_stim_noise(&program, &uniform_depolarizing())
                        .unwrap()
                        .parse()
                        .unwrap();
                assert_eq!(noisy.num_observables(), 3);
                noisy
                    .detector_error_model()
                    .unwrap_or_else(|e| panic!("walls={sx},{sy},{z}: {e}"));
                assert_eq!(common::graphlike_distance(&noisy).unwrap(), 3);
            }
        }
    }
}
