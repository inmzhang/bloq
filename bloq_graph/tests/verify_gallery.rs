#![cfg(test)]
//! Acceptance tests for sampled, projective ZX-map verification.

use std::collections::BTreeMap;

use bloq_graph::verify::{
    BoundaryOrder, BranchStatus, LogicalVerifier, QuizxGraph, VerifyLogicalError,
};
use bloq_graph::{
    Action, Basis, Block, BlockGraph, BlockKind, CubeKind, Expr, GalleryItem, MeasureTarget,
    UDirection,
};
use glam::IVec3;
use quizx::circuit::Circuit;
use quizx::graph::{BasisElem, GraphLike};
use quizx::tensor::{CompareTensors, TensorF, ToTensor};
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[test]
fn hadamard_selective_cap_preserves_its_source_branch_sign() {
    let positive = prepared_graph(1, &[("h", vec![0]), ("s", vec![0])], &[BasisElem::Z0]);
    let negative = prepared_graph(
        1,
        &[("h", vec![0]), ("s", vec![0]), ("z", vec![0])],
        &[BasisElem::Z0],
    );
    for (cap, arrow, feedback) in [
        ("YX", "->", false),
        ("YZ", "-H>", false),
        ("YZ", "-H>", true),
    ] {
        let correction = if feedback {
            "feedback Z 2 if mzz\n"
        } else {
            ""
        };
        let graph = BlockGraph::from_blog_text(&format!("BLOG 1.0\n0: T [0,0,0]\n1: XZX [0,0,1]\n2: Port [0,0,2]\n3: T [1,0,0]\n4: XZX [1,0,1]\n5: {cap} [1,0,2]\n0 -> +Z\n1 -> +Z\n1 -> +X\n3 -> +Z\n5 {arrow} -Z\nmzz = measure 1 -> +X\nresolve 5 if mzz\n{correction}")).unwrap().fix_shadowed_faces();
        for internal in [false, true] {
            let verifier = if internal {
                LogicalVerifier::with_internal_measurements(&graph)
            } else {
                LogicalVerifier::new(&graph)
            }
            .unwrap();
            for bit in [false, true] {
                let expected = if cap == "YZ" && bit && !feedback {
                    &negative
                } else {
                    &positive
                };
                assert_eq!(
                    verifier
                        .verify_branch(expected, &BTreeMap::from([("mzz".to_owned(), bit)]))
                        .unwrap()
                        .status(),
                    BranchStatus::Verified,
                    "{cap} internal={internal} mzz={bit} feedback={feedback}"
                );
            }
        }
    }
}

fn circuit_graph(qubits: usize, gates: &[(&str, Vec<usize>)]) -> QuizxGraph {
    let mut circuit = Circuit::new(qubits);
    for (gate, operands) in gates {
        circuit.add_gate(gate, operands.clone());
    }
    circuit.to_graph()
}

fn prepared_graph(qubits: usize, gates: &[(&str, Vec<usize>)], inputs: &[BasisElem]) -> QuizxGraph {
    let mut graph = circuit_graph(qubits, gates);
    graph.plug_inputs(inputs);
    graph
}

fn ghz_state(output_hadamards: bool) -> QuizxGraph {
    let mut gates = vec![
        ("h", vec![0]),
        ("cx", vec![0, 1]),
        ("cx", vec![0, 2]),
        ("cx", vec![0, 3]),
    ];
    if output_hadamards {
        for qubit in 0..4 {
            gates.push(("h", vec![qubit]));
        }
    }
    prepared_graph(4, &gates, &[BasisElem::Z0; 4])
}

fn steane_zero_map() -> QuizxGraph {
    let mut graph = prepared_graph(
        7,
        &[
            ("h", vec![0]),
            ("h", vec![1]),
            ("h", vec![2]),
            ("cx", vec![0, 4]),
            ("cx", vec![0, 5]),
            ("cx", vec![0, 6]),
            ("cx", vec![1, 3]),
            ("cx", vec![1, 5]),
            ("cx", vec![1, 6]),
            ("cx", vec![2, 3]),
            ("cx", vec![2, 4]),
            ("cx", vec![2, 6]),
        ],
        &[BasisElem::Z0; 7],
    );
    let ports = graph.outputs().clone();
    graph.set_inputs(ports[..3].to_vec());
    graph.set_outputs(ports[3..].to_vec());
    graph
}

fn one_d_yoked_map() -> QuizxGraph {
    let mut circuit = Circuit::new(8);
    for data in 0..6 {
        circuit.add_gate("cz", vec![6, data]);
    }
    for data in 0..6 {
        circuit.add_gate("cx", vec![7, data]);
    }
    let mut graph: QuizxGraph = circuit.to_graph();
    let ancilla_inputs = [
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::X0,
        BasisElem::X0,
    ];
    graph.plug_inputs(&ancilla_inputs);
    graph.plug_outputs(&ancilla_inputs);
    graph
}

fn phase_gradient_map() -> QuizxGraph {
    let signs = [
        (1, 1),
        (1, 1),
        (-1, -1),
        (-1, -1),
        (-1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];
    let mut circuit = Circuit::new(1);
    for (x, z) in signs {
        circuit.add_gate_with_phase("rx", vec![0], (x, 4));
        circuit.add_gate_with_phase("rz", vec![0], (z, 4));
    }
    circuit.add_gate("h", vec![0]);
    circuit.to_graph()
}

fn and_map() -> QuizxGraph {
    let mut graph = circuit_graph(3, &[("ccx", vec![0, 1, 2])]);
    graph.plug_inputs(&[BasisElem::SKIP, BasisElem::SKIP, BasisElem::Z0]);
    graph
}

fn ccz_injected_and_map() -> QuizxGraph {
    let mut graph = and_map();
    // Spatial boundary order is qi, i, q by coordinate; both controls survive.
    graph.outputs_mut().rotate_right(1);
    graph
}

fn maj_map() -> QuizxGraph {
    let mut graph = circuit_graph(
        5,
        &[
            ("cx", vec![1, 0]),
            ("cx", vec![1, 4]),
            ("ccx", vec![0, 4, 3]),
            ("cx", vec![1, 3]),
            ("cx", vec![3, 2]),
        ],
    );
    graph.plug_inputs(&[
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::Z0,
        BasisElem::Z0,
        BasisElem::SKIP,
    ]);
    let outputs = graph.outputs().clone();
    graph.set_outputs(vec![
        outputs[2], outputs[1], outputs[3], outputs[0], outputs[4],
    ]);
    graph
}

fn uma_map(m_anc: bool, m_ikprime: bool) -> QuizxGraph {
    // Fig. 62 wire order: c_k, c_(k+1), c_k xor i'_k, two |+>, c_k xor t_k.
    let outcome = |basis: BasisElem, value| if value { basis.flipped() } else { basis };
    let cap_basis = if m_anc { BasisElem::Z0 } else { BasisElem::X0 };
    let mut gates = vec![
        ("cx", vec![0, 1]),
        ("cz", vec![3, 4]),
        ("cx", vec![2, 3]),
        ("cx", vec![5, 4]),
        ("cx", vec![0, 2]),
        ("cx", vec![2, 5]),
    ];
    if m_anc ^ m_ikprime {
        gates.push(("z", vec![5]));
    }
    let mut graph = circuit_graph(6, &gates);
    graph.plug_inputs(&[
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::SKIP,
        BasisElem::X0,
        BasisElem::X0,
        BasisElem::SKIP,
    ]);
    graph.plug_outputs(&[
        BasisElem::X0,
        outcome(BasisElem::X0, m_anc),
        outcome(BasisElem::X0, m_ikprime),
        outcome(cap_basis, m_ikprime),
        outcome(cap_basis, m_anc ^ m_ikprime),
        BasisElem::SKIP,
    ]);
    graph
}

fn controlled_adder_map(bits: usize) -> QuizxGraph {
    // Independent unitary reference: compute q & i into clean ancillas,
    // add that register with a Cuccaro MAJ/UMA ripple, then uncompute it.
    let data = 2 * bits + 1;
    let a = |bit| data + bit;
    let b = |bit| 1 + bits + bit;
    let carry = |bit| if bit == 0 { data + bits } else { a(bit - 1) };
    let mut gates = Vec::new();
    for bit in 0..bits {
        gates.push(("ccx", vec![0, 1 + bit, a(bit)]));
    }
    for bit in 0..bits {
        gates.extend([
            ("cx", vec![a(bit), b(bit)]),
            ("cx", vec![a(bit), carry(bit)]),
            ("ccx", vec![carry(bit), b(bit), a(bit)]),
        ]);
    }
    for bit in (0..bits).rev() {
        gates.extend([
            ("ccx", vec![carry(bit), b(bit), a(bit)]),
            ("cx", vec![a(bit), carry(bit)]),
            ("cx", vec![carry(bit), b(bit)]),
        ]);
    }
    for bit in 0..bits {
        gates.push(("ccx", vec![0, 1 + bit, a(bit)]));
    }
    let mut graph = circuit_graph(data + bits + 1, &gates);
    let mut boundary = vec![BasisElem::Z0; data + bits + 1];
    boundary[..data].fill(BasisElem::SKIP);
    graph.plug_inputs(&boundary);
    graph.plug_outputs(&boundary);
    graph
}

#[test]
fn adder_uma_preserves_erase_and_carry_phase() {
    let program = GalleryItem::ThreeBitAdder.build();
    let module = program
        .modules()
        .find(|m| m.name == "BulkUmaParked")
        .unwrap();
    let verifier = LogicalVerifier::with_boundaries(
        module.local_body(),
        BoundaryOrder::new(
            vec![
                IVec3::new(0, 0, -1),
                IVec3::new(0, 2, -1),
                IVec3::new(1, 0, -1),
                IVec3::new(1, 2, -1),
            ],
            vec![IVec3::new(2, 0, 1)],
        ),
    )
    .unwrap();
    for u in [false, true] {
        for e in [false, true] {
            for incoming in [false, true] {
                let (_, map) = verifier
                    .instantiate(&BTreeMap::from([
                        ("m_anc".into(), u),
                        ("m_ikprime".into(), e),
                        ("incoming_z".into(), incoming),
                    ]))
                    .unwrap();
                let map = map.unwrap();
                let mut amplitudes = Vec::new();
                for mask in 0..8 {
                    let c = mask & 1 != 0;
                    let a = mask & 2 != 0;
                    let t = mask & 4 != 0;
                    let basis = |v| if v { BasisElem::Z1 } else { BasisElem::Z0 };
                    let mut selected = map.clone();
                    selected.plug_inputs(&[c, c ^ ((a ^ c) & (t ^ c)), a ^ c, t ^ c].map(basis));
                    selected.plug_outputs(&[basis(a ^ t ^ c)]);
                    let value = contract(&selected).iter().next().unwrap().complex_value();
                    assert!(
                        value.norm() > 0.0,
                        "u={u} e={e} incoming={incoming} mask={mask}"
                    );
                    let phase = ((e ^ u) & (a ^ c)) ^ (incoming & ((a ^ c) & (t ^ c)));
                    amplitudes.push(if phase { -value } else { value });
                }
                for amplitude in &amplitudes {
                    assert!(
                        (*amplitude / amplitudes[0] - 1.0).norm() < 1e-9,
                        "UMA phase: anc={u} erase={e} incoming={incoming}: {amplitudes:?}"
                    );
                }
            }
        }
    }
}

fn action_value(
    expr: &Expr,
    values: &BTreeMap<String, bool>,
    aliases: &BTreeMap<&str, &Expr>,
) -> bool {
    match expr {
        Expr::Var(name) => values
            .get(name)
            .copied()
            .unwrap_or_else(|| action_value(aliases[name.as_str()], values, aliases)),
        Expr::Not(inner) => !action_value(inner, values, aliases),
        Expr::Binary(op, left, right) => {
            let left = action_value(left, values, aliases);
            let right = action_value(right, values, aliases);
            match op {
                bloq_graph::BinaryOp::Xor => left ^ right,
                bloq_graph::BinaryOp::And => left & right,
                bloq_graph::BinaryOp::Or => left | right,
            }
        }
    }
}

fn verify_controlled_adder(gallery: GalleryItem, expected: &QuizxGraph, samples: usize) {
    let bits = (expected.inputs().len() - 1) / 2;
    use bloq_graph::PortDirection;
    let program = gallery.build();
    let graph = program
        .materialize_flat_graph()
        .unwrap()
        .fix_shadowed_faces();
    let ports = &program.root().interface.quantum_ports;
    let port = |name: String| {
        ports
            .iter()
            .find(|port| port.name == name)
            .unwrap()
            .position
    };
    let data = 2 * bits + 1;
    let inputs = std::iter::once(port("q_in".into()))
        .chain((0..bits).map(|bit| port(format!("i{bit}_in"))))
        .chain((0..bits).map(|bit| port(format!("t{bit}_in"))))
        .chain(
            ports
                .iter()
                .filter(|port| {
                    port.resource_type == "ccz" && port.direction == PortDirection::Input
                })
                .map(|port| port.position),
        )
        .collect::<Vec<_>>();
    let outputs = std::iter::once(port("q_out".into()))
        .chain((0..bits).map(|bit| port(format!("i{bit}_out"))))
        .chain((0..bits).map(|bit| port(format!("s{bit}_out"))))
        .collect::<Vec<_>>();
    let mut preparation = Circuit::new(inputs.len());
    for base in (data..inputs.len()).step_by(3) {
        preparation.add_gate("ccz", vec![base, base + 1, base + 2]);
    }
    let mut preparation: QuizxGraph = preparation.to_graph();
    let mut prepared_inputs = vec![BasisElem::X0; inputs.len()];
    prepared_inputs[..data].fill(BasisElem::SKIP);
    preparation.plug_inputs(&prepared_inputs);
    let boundaries = BoundaryOrder::new(inputs, outputs);
    let actions = graph.actions();
    let aliases = actions
        .iter()
        .filter_map(|action| match action {
            Action::Let { name, expr } => Some((name.as_str(), expr)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let normalize = |mut map: QuizxGraph| {
        quizx::simplify::full_simp(&mut map);
        assert!(map.scalar().complex_value().norm() > 0.0);
        *map.scalar_mut() = quizx::fscalar::FScalar::real(1.0);
        map
    };
    let amplitude = |map: &QuizxGraph, input: usize, output: usize| {
        let basis = |mask| {
            (0..data)
                .map(|bit| {
                    if mask & (1 << bit) == 0 {
                        BasisElem::Z0
                    } else {
                        BasisElem::Z1
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut map = map.clone();
        map.plug_inputs(&basis(input));
        map.plug_outputs(&basis(output));
        contract(&map).iter().next().unwrap().complex_value()
    };
    let limit = (1 << bits) - 1;
    for sample in 0..samples {
        // Select one reachable source instrument before deriving its ZX rows.
        // This avoids asking the scalar oracle to normalize a joint domain.
        let mut rng = StdRng::seed_from_u64(0xADD4 + sample as u64);
        let values = actions
            .iter()
            .filter_map(|action| match action {
                Action::Measure { name, .. } => Some((name.clone(), rng.random())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let assignments = actions.iter().filter_map(|action| match action {
            Action::Branch { target, condition } => {
                Some((*target, action_value(condition, &values, &aliases)))
            }
            _ => None,
        });
        let mut selected = graph.project_branches_deferred(assignments).unwrap();
        for action in &actions {
            if let Action::Resolve { target, condition } = action {
                let BlockKind::Selective(kind) = selected.get_block(*target).unwrap().kind() else {
                    unreachable!()
                };
                let basis = if action_value(condition, &values, &aliases) {
                    kind.pauli_if_true()
                } else {
                    kind.pauli_if_false()
                };
                let kind = match basis {
                    bloq_graph::PauliBasis::X => BlockKind::Measurement(Basis::X),
                    bloq_graph::PauliBasis::Y => BlockKind::Y,
                    bloq_graph::PauliBasis::Z => BlockKind::Measurement(Basis::Z),
                };
                selected.set_block_kind(*target, kind).unwrap();
            }
        }
        selected
            .set_actions_deferred(
                actions
                    .iter()
                    .filter(|action| {
                        !matches!(action, Action::Resolve { .. } | Action::Branch { .. })
                    })
                    .cloned()
                    .collect(),
            )
            .unwrap();
        let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
            &selected,
            boundaries.clone(),
        )
        .unwrap();
        let (branch, diagram) = verifier.instantiate(&values).unwrap();
        let mut actual = preparation.clone();
        actual.plug(&diagram.unwrap());
        let actual = normalize(actual);
        // An ideal controlled addition has unit amplitude on each correct
        // output, so every input must share one branch-global scalar.
        let mut ratios = Vec::new();
        for (probe, (q, i, t)) in [
            (0, limit, limit),
            (1, 1, limit),
            (1, limit, 1),
            (1, limit, limit),
            (1, 0, limit),
            (1, 1 << (bits - 1), limit),
            (0, 1, 0),
            (1, limit / 3, 2 * (limit / 3)),
        ]
        .into_iter()
        .enumerate()
        {
            let input = q | (i << 1) | (t << (bits + 1));
            let output = q | (i << 1) | (((t + q * i) & limit) << (bits + 1));
            let a = amplitude(&actual, input, output);
            assert!(
                a.norm() > 0.0,
                "{gallery} sample {sample} input {input}: actual={a:?}"
            );
            let ideal = amplitude(expected, input, output);
            assert!(ideal.norm() > 0.0, "independent adder reference is nonzero");
            ratios.push(a / ideal);
            let wrong = output ^ (1 << ((sample + probe) % data));
            assert!(amplitude(&actual, input, wrong).norm() < a.norm() * 1e-9);
        }
        for ratio in &ratios {
            assert!(
                (*ratio / ratios[0] - 1.0).norm() < 1e-9,
                "{gallery} sample {sample}: relative input phase changed: ratios={ratios:?}; values={:?}",
                branch.classical_state()
            );
        }
    }
}

fn ccz_state() -> QuizxGraph {
    prepared_graph(3, &[("ccz", vec![0, 1, 2])], &[BasisElem::X0; 3])
}

fn prepared_ccz_inputs() -> QuizxGraph {
    let mut graph = circuit_graph(6, &[("ccz", vec![1, 3, 5])]);
    graph.plug_inputs(&[
        BasisElem::SKIP,
        BasisElem::X0,
        BasisElem::SKIP,
        BasisElem::X0,
        BasisElem::SKIP,
        BasisElem::X0,
    ]);
    graph
}

fn prepared_ccz_prefix_inputs(qubits: usize) -> QuizxGraph {
    let mut graph = circuit_graph(qubits, &[("ccz", vec![0, 1, 2])]);
    let mut inputs = vec![BasisElem::SKIP; qubits];
    inputs[..3].fill(BasisElem::X0);
    graph.plug_inputs(&inputs);
    graph
}

fn maj_boundaries() -> BoundaryOrder {
    BoundaryOrder::new(
        vec![
            IVec3::new(-1, 0, 2),
            IVec3::new(-1, 1, 2),
            IVec3::new(-1, 2, 2),
            IVec3::new(0, 1, -1),
            IVec3::new(2, -1, 0),
            IVec3::new(5, 1, 3),
        ],
        vec![
            IVec3::new(2, 3, 0),
            IVec3::new(3, 0, 5),
            IVec3::new(3, 2, 5),
            IVec3::new(4, 0, 5),
            IVec3::new(4, 2, 5),
        ],
    )
}

fn port_z_basis(graph: &BlockGraph, position: IVec3) -> UDirection {
    let pipe = graph
        .pipes()
        .find(|pipe| pipe.src() == position || pipe.dst() == position)
        .expect("gallery port has one pipe");
    let pipe_axis = pipe.dir().as_udirection();
    let bases = graph.infer_pipe_basis_from_endpoint(pipe, position);
    UDirection::iter()
        .find(|axis| *axis != pipe_axis && bases[axis.index()] == Some(Basis::Z))
        .expect("port pipe has one Z face")
}

#[test]
fn maj_ports_are_exterior_and_match_and_uma_frames() {
    let and = GalleryItem::CCZInjectedAnd
        .build()
        .materialize_root_graph()
        .expect("gallery flat projection");
    let maj = GalleryItem::CCZInjectedMaj
        .build()
        .materialize_root_graph()
        .expect("gallery flat projection");
    let uma = GalleryItem::UMA
        .build()
        .materialize_root_graph()
        .expect("gallery flat projection");
    let in_box = |position: IVec3| {
        position.cmpge(IVec3::ZERO).all() && position.cmplt(IVec3::new(5, 3, 5)).all()
    };
    for port in maj.blocks().filter(|block| block.kind().is_port()) {
        let pipe = maj
            .pipes()
            .find(|pipe| pipe.src() == port.pos() || pipe.dst() == port.pos())
            .expect("gallery port has one pipe");
        let neighbor = if pipe.src() == port.pos() {
            pipe.dst()
        } else {
            pipe.src()
        };
        assert!(!in_box(port.pos()) && in_box(neighbor), "{}", port.pos());
    }

    assert_eq!(
        port_z_basis(&maj, IVec3::new(0, 1, -1)),
        port_z_basis(&and, IVec3::new(0, 1, 2)),
    );
    for (maj_port, uma_port) in [
        (IVec3::new(3, 0, 5), IVec3::new(0, 0, -1)),
        (IVec3::new(3, 2, 5), IVec3::new(0, 2, -1)),
        (IVec3::new(4, 0, 5), IVec3::new(1, 0, -1)),
        (IVec3::new(4, 2, 5), IVec3::new(1, 2, -1)),
    ] {
        assert_eq!(port_z_basis(&maj, maj_port), port_z_basis(&uma, uma_port));
    }
}

fn contract(graph: &QuizxGraph) -> TensorF {
    let mut graph = graph.clone();
    quizx::simplify::full_simp(&mut graph);
    graph.to_tensorf()
}

fn verify_prepared_branches(
    verifier: &LogicalVerifier,
    names: &[&str],
    expected: &QuizxGraph,
    mut prepare: impl FnMut() -> QuizxGraph,
    label: &str,
) {
    let ideal = contract(expected);
    for mask in 0..1 << names.len() {
        let assignment = names
            .iter()
            .enumerate()
            .map(|(bit, name)| ((*name).to_owned(), mask & (1 << bit) != 0))
            .collect();
        let (branch, actual) = verifier.instantiate(&assignment).unwrap();
        assert_eq!(
            branch.status(),
            BranchStatus::Instantiated,
            "{label} {mask:b}"
        );
        let mut prepared = prepare();
        prepared.plug(&actual.expect("reachable branch"));
        assert!(
            <TensorF as CompareTensors>::scalar_eq(&contract(&prepared), &ideal),
            "{label} branch {mask:b} differs from its declared map"
        );
    }
}

fn expected_map(gallery: GalleryItem) -> (QuizxGraph, usize) {
    match gallery {
        GalleryItem::CNOT => (circuit_graph(2, &[("cx", vec![0, 1])]), 1),
        GalleryItem::CZSpatialH | GalleryItem::CZTemporalH => {
            (circuit_graph(2, &[("cz", vec![0, 1])]), 1)
        }
        GalleryItem::S => (circuit_graph(1, &[("s", vec![0])]), 1),
        GalleryItem::T | GalleryItem::TWithPreparedY => (circuit_graph(1, &[("t", vec![0])]), 32),
        GalleryItem::TComparison => (QuizxGraph::new(), 64),
        GalleryItem::PhaseGradientK4 => (phase_gradient_map(), 32),
        GalleryItem::And4T => {
            let mut graph = and_map();
            // Spatial boundary order is x, xy, y by coordinate.
            graph.outputs_mut().swap(1, 2);
            (graph, 32)
        }
        GalleryItem::CCZInjectedAnd => (ccz_injected_and_map(), 4),
        GalleryItem::CCZInjectedMaj => (maj_map(), 4),
        GalleryItem::UMA => (uma_map(false, false), 4),
        GalleryItem::ThreeBitAdder => (controlled_adder_map(3), 4),
        GalleryItem::TenBitAdder => (controlled_adder_map(10), 4),
        GalleryItem::ToffoliFromAndDelayedCZ => (circuit_graph(3, &[("ccx", vec![0, 1, 2])]), 128),
        GalleryItem::CCZGateTeleport => (circuit_graph(3, &[("ccz", vec![0, 1, 2])]), 8),
        GalleryItem::CCZFactoryWithTels | GalleryItem::CCZ4x3_6 => (ccz_state(), 128),
        GalleryItem::BellState => (
            prepared_graph(
                2,
                &[("h", vec![0]), ("cx", vec![0, 1])],
                &[BasisElem::Z0; 2],
            ),
            1,
        ),
        GalleryItem::GHZ => (ghz_state(false), 1),
        GalleryItem::GHZSlideThenGlide | GalleryItem::GHZPatchRotations => (ghz_state(true), 1),
        GalleryItem::OneDYoked => (one_d_yoked_map(), 1),
        GalleryItem::THTH => (
            circuit_graph(
                1,
                &[
                    ("t", vec![0]),
                    ("h", vec![0]),
                    ("t", vec![0]),
                    ("h", vec![0]),
                ],
            ),
            32,
        ),
        GalleryItem::ThreeCNOTs => (
            circuit_graph(
                3,
                &[("cx", vec![0, 1]), ("cx", vec![0, 2]), ("cx", vec![1, 2])],
            ),
            1,
        ),
        GalleryItem::SteaneEncoding => (steane_zero_map(), 1),
        GalleryItem::XMemory | GalleryItem::YMemory | GalleryItem::Stability => {
            (QuizxGraph::new(), 1)
        }
        GalleryItem::MoveRotation => (circuit_graph(1, &[]), 1),
    }
}

#[test]
fn every_gallery_map_verifies() {
    for gallery in GalleryItem::iter().filter(|gallery| *gallery != GalleryItem::TenBitAdder) {
        let (expected, samples) = expected_map(gallery);
        if gallery == GalleryItem::ThreeBitAdder {
            verify_controlled_adder(gallery, &expected, samples);
            continue;
        }
        let graph = gallery
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        // Exhaustive named assignments already include every sampled branch.
        if matches!(
            gallery,
            GalleryItem::CNOT
                | GalleryItem::CZSpatialH
                | GalleryItem::CZTemporalH
                | GalleryItem::S
                | GalleryItem::T
                | GalleryItem::TWithPreparedY
                | GalleryItem::THTH
                | GalleryItem::And4T
                | GalleryItem::ToffoliFromAndDelayedCZ
        ) {
            verify_gate_branches(gallery, &graph, &expected);
            continue;
        }
        if gallery == GalleryItem::CCZInjectedAnd {
            let verifier = LogicalVerifier::with_internal_measurements(&graph)
                .expect("CCZ-injected AND verifier builds");
            verify_prepared_branches(
                &verifier,
                &["m1", "m2"],
                &expected,
                || prepared_ccz_prefix_inputs(5),
                "CCZ-injected AND",
            );
            continue;
        }
        if gallery == GalleryItem::CCZInjectedMaj {
            let verifier = LogicalVerifier::with_boundaries(&graph, maj_boundaries())
                .expect("CCZ-injected MAJ verifier builds");
            verify_prepared_branches(
                &verifier,
                &["mt1", "mt2"],
                &expected,
                || prepared_ccz_prefix_inputs(6),
                "CCZ-injected MAJ",
            );
            continue;
        }
        if gallery == GalleryItem::UMA {
            let verifier =
                LogicalVerifier::with_internal_measurements(&graph).expect("UMA verifier builds");
            for mask in 0..4 {
                let m_anc = mask & 1 != 0;
                let m_ikprime = mask & 2 != 0;
                let assignment = [
                    ("m_anc".to_owned(), m_anc),
                    ("m_ikprime".to_owned(), m_ikprime),
                ]
                .into_iter()
                .collect();
                verifier
                    .verify_branch(&uma_map(m_anc, m_ikprime), &assignment)
                    .unwrap_or_else(|error| panic!("UMA branch {mask:02b}: {error}"));
            }
            continue;
        }
        if gallery == GalleryItem::CCZGateTeleport {
            let verifier = LogicalVerifier::with_internal_measurements(&graph)
                .expect("CCZ teleport verifier builds");
            verify_prepared_branches(
                &verifier,
                &["m0x", "m1y", "m2z"],
                &expected,
                prepared_ccz_inputs,
                "CCZ teleport",
            );
            continue;
        }
        let verifier = LogicalVerifier::new(&graph)
            .unwrap_or_else(|error| panic!("{gallery:?}: verifier build failed: {error}"));
        verifier
            .verify(&expected, samples, 0x5eed)
            .unwrap_or_else(|error| panic!("{gallery:?}: {error}"));
    }
}

#[test]
#[ignore = "ten-bit logical-map amplitude stress; run via just test-full"]
fn ten_bit_adder_map_verifies() {
    let (expected, samples) = expected_map(GalleryItem::TenBitAdder);
    verify_controlled_adder(GalleryItem::TenBitAdder, &expected, samples);
}

fn verify_gate_branches(gallery: GalleryItem, graph: &BlockGraph, expected: &QuizxGraph) {
    let names = graph
        .actions()
        .into_iter()
        .filter_map(|action| match action {
            Action::Measure { name, .. } => Some(name),
            _ => None,
        })
        .collect::<Vec<_>>();
    let verifier = LogicalVerifier::new(graph).unwrap();
    let mut verified = 0;
    for bits in 0..1usize << names.len() {
        let assignment = names
            .iter()
            .enumerate()
            .map(|(bit, name)| (name.clone(), bits >> bit & 1 == 1))
            .collect();
        if verifier
            .verify_branch(expected, &assignment)
            .unwrap()
            .status()
            == BranchStatus::Verified
        {
            verified += 1;
        }
    }
    assert!(verified > 0, "{gallery:?}: every assignment was impossible");
}

#[test]
fn wrong_expected_map_is_rejected() {
    let verifier = LogicalVerifier::new(
        &GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection"),
    )
    .unwrap();
    let wrong = circuit_graph(2, &[("cz", vec![0, 1])]);

    let error = verifier.verify(&wrong, 1, 0).unwrap_err();

    assert!(matches!(error, VerifyLogicalError::MapMismatch { .. }));
}

#[test]
fn discard_rejects_after_the_complete_assignment_is_seeded() {
    let first = IVec3::new(0, 0, 0);
    let later = IVec3::new(0, 0, 2);
    let mut graph = BlockGraph::new();
    graph.add_block(Block::new(first, BlockKind::Cube(CubeKind::ZXZ)));
    graph.add_block(Block::new(later, BlockKind::Cube(CubeKind::ZXZ)));
    graph
        .set_actions(vec![
            Action::Measure {
                target: MeasureTarget::Node(first),
                name: "first".to_string(),
            },
            Action::DiscardIf(Expr::Var("first".to_string())),
            Action::Measure {
                target: MeasureTarget::Node(later),
                name: "later".to_string(),
            },
        ])
        .unwrap();
    let verifier = LogicalVerifier::new(&graph).unwrap();
    let assignment = BTreeMap::from([("first".to_string(), true), ("later".to_string(), false)]);

    let (branch, diagram) = verifier.instantiate(&assignment).unwrap();

    assert_eq!(branch.status(), BranchStatus::Rejected);
    assert_eq!(
        branch.classical_state(),
        &BTreeMap::from([("first".to_string(), true), ("later".to_string(), false),])
    );
    assert!(diagram.is_none());
}

#[test]
fn zero_diagram_assignment_is_impossible_not_verified() {
    let position = IVec3::ZERO;
    let mut graph = BlockGraph::new();
    graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
    graph
        .set_actions(vec![Action::Measure {
            target: MeasureTarget::Node(position),
            name: "m".to_string(),
        }])
        .unwrap();
    let verifier = LogicalVerifier::new(&graph).unwrap();

    let branch = verifier
        .verify_branch(
            &QuizxGraph::new(),
            &BTreeMap::from([("m".to_string(), true)]),
        )
        .unwrap();

    assert_eq!(branch.status(), BranchStatus::Impossible);
}

/// A record whose only surface reaches an output is not a valid graph.
///
/// The T gadget on an internally prepared `|+>`: with no input port on the data
/// worldline, `mzz` has no row that closes on the past. Its only surface is the
/// transport row `Z(T) . Z(Out)`, which no physical readout can realize.
#[test]
fn output_anchored_measurement_surface_is_rejected() {
    let error = BlockGraph::from_blog_text(
        "BLOG 1.0\n\n\
         1: XZX [0, 0, 1]\n\
         2: Port [0, 0, 2] <Out>\n\
         3: T [1, 0, 0]\n\
         4: XZX [1, 0, 1]\n\
         5: YX [1, 0, 2]\n\
         [0, 0, 1] -> +Z\n\
         [0, 0, 1] -> +X\n\
         [1, 0, 0] -> +Z\n\
         [1, 0, 2] -> -Z\n\n\
         mzz = measure 1 -> +X\n\
         resolve 5 if mzz\n",
    )
    .unwrap_err();
    let bloq_graph::BlockGraphError::Parse(bloq_graph::ParseError::Graph { source, .. }) = &error
    else {
        panic!("{error:?}");
    };
    assert!(
        matches!(
            source.as_ref(),
            bloq_graph::BlockGraphError::Stabilizer(
                bloq_graph::StabilizerError::UnavailableControlParity { mvar, .. }
            ) if mvar == "mzz"
        ),
        "{error:?}"
    );
}
