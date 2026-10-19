//! Complete normalized input/output correlations for sampled stabilizer branches.
//! Bell-paired inputs cover arbitrary joint inputs, including open resources.
//! Source-derived correlations complement independent ideal-circuit/QuiZX checks;
//! they do not check branch probabilities. Closed boundaries only check records.

#![allow(dead_code, reason = "shared by independent integration-test binaries")]

#[path = "choi/exact.rs"]
pub(crate) mod exact;
#[path = "choi/stabilizer.rs"]
pub(crate) mod stabilizer;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use bloq_graph::{
    Action, BinaryOp, BlockGraph, Expr, ModuleCertificationLimits, NodeKind, Pauli,
    RuntimeStabilizerBasis, ZXGraph,
};
use bloq_vm::{EnginePauli, EnginePauliString, run_bloq_with_io};
use glam::IVec3;

fn evaluate(expr: &Expr, values: &HashMap<String, bool>, aliases: &HashMap<&str, &Expr>) -> bool {
    match expr {
        Expr::Var(name) => values
            .get(name)
            .copied()
            .unwrap_or_else(|| evaluate(aliases[name.as_str()], values, aliases)),
        Expr::Not(inner) => !evaluate(inner, values, aliases),
        Expr::Binary(op, left, right) => match op {
            BinaryOp::Xor => evaluate(left, values, aliases) ^ evaluate(right, values, aliases),
            BinaryOp::And => evaluate(left, values, aliases) & evaluate(right, values, aliases),
            BinaryOp::Or => evaluate(left, values, aliases) | evaluate(right, values, aliases),
        },
    }
}

pub(super) fn widen(pauli: &EnginePauliString, width: usize) -> EnginePauliString {
    let mut result = EnginePauliString::from_terms(
        width,
        (0..pauli.nqubits).map(|qubit| (qubit, pauli.get(qubit))),
    );
    result.set_phase(pauli.phase_exponent());
    result
}

/// Each sampled branch must satisfy a complete linearly independent Choi stabilizer
/// set. By linearity this checks the open map on arbitrary joint inputs,
/// including all of its CCZ resources, without expanding their tensor product.
pub(crate) fn assert_channel(program: &BlockGraph, bloq: &bloq_ir::Bloq, shots: usize) {
    assert_channel_with_seed(program, bloq, shots, 0xC401);
}

pub(crate) fn assert_channel_with_seed(
    program: &BlockGraph,
    bloq: &bloq_ir::Bloq,
    shots: usize,
    seed: u64,
) {
    let graph = program.flatten().expect("Choi source materializes");
    let offset = IVec3::new(
        0,
        0,
        -*graph.spans().expect("Choi source is nonempty").2.start(),
    );
    let graph = graph
        .with_zero_min_z()
        .expect("Choi source normalizes within the coordinate lattice")
        .fix_shadowed_faces();
    let summary = (!program.instances.is_empty()).then(|| {
        program
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .expect("Choi source summary certifies")
    });
    let linked = bloq_graph::flatten_module_definition(program, program, "")
        .expect("Choi source definition flattens");
    let sites = linked
        .sites
        .into_iter()
        .map(|(pos, site)| (pos + offset, site))
        .collect();
    let space = bloq_graph::GuardedSurfaceSpace::new(
        bloq_graph::GuardedTopology::new(&graph, ModuleCertificationLimits::DEFAULT)
            .expect("Choi source topology certifies"),
        &sites,
        ModuleCertificationLimits::DEFAULT,
    )
    .expect("Choi surface space certifies");
    let space = space.plan_readouts().expect("Choi source readouts close");
    // Lowering emits a readout for every source row except an absent output
    // axis. Preserve that source order to bind unnamed input-only parities.
    let mut next_observable = 0;
    let mut logical_readouts = Vec::new();
    for (index, surface) in space.surfaces.iter().enumerate() {
        if matches!(
            surface.kind,
            bloq_graph::GuardedSurfaceKind::OutputFrame { .. }
        ) && space
            .topology
            .diagram
            .witness(surface.activation())
            .is_none()
        {
            continue;
        }
        if surface.kind == bloq_graph::GuardedSurfaceKind::LogicalReadout {
            logical_readouts.push((index, next_observable));
        }
        next_observable += 1;
    }
    let actions = graph.actions();
    let aliases = actions
        .iter()
        .filter_map(|action| match action {
            Action::Let { name, expr } => Some((name.as_str(), expr)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let named_indices = bloq
        .nodes()
        .filter_map(|(_, node)| {
            let bloq_ir::NodeProvenance::Action { ordinal } = node.provenance else {
                return None;
            };
            let Some(bloq_ir::ClassicalNode::Observable {
                index: Some(index), ..
            }) = node.try_classical()
            else {
                return None;
            };
            let Action::Measure { name, .. } = &actions[ordinal as usize] else {
                panic!("readout source is a measurement");
            };
            Some((name.as_str(), *index as usize))
        })
        .collect::<HashMap<_, _>>();
    let references = RefCell::new(HashMap::new());
    let mut states = Vec::new();
    let report = run_bloq_with_io(
        bloq,
        shots,
        seed,
        |sim, ctx| {
            let first = sim.num_qubits();
            for (index, input) in ctx.inputs.iter().enumerate() {
                let reference = first + index;
                sim.cx(input.qubit, reference)?;
                if let Some(previous) = references
                    .borrow_mut()
                    .insert(input.port, (reference, index))
                {
                    assert_eq!(previous, (reference, index), "stable Bell-reference layout");
                }
            }
            Ok(())
        },
        |sim, ctx| {
            states.push((sim.clone(), ctx.outputs.to_vec(), ctx.frames.to_vec()));
            Ok(())
        },
    )
    .expect("physical Choi execution succeeds");
    assert_eq!(report.discarded, 0);
    assert_eq!(report.shots, shots);
    assert_eq!(states.len(), shots, "every shot has a Choi state");
    assert!(
        report
            .detectors
            .iter()
            .all(|detector| detector.constant && detector.value != Some(true))
    );
    assert!(
        report
            .detectors
            .iter()
            .any(|detector| detector.per_shot.len() == shots)
    );
    let observed = bloq
        .top()
        .nodes()
        .filter_map(|(_, node)| match node.try_classical() {
            Some(bloq_ir::ClassicalNode::Observable {
                index: Some(index), ..
            }) => Some(*index as usize),
            _ => None,
        })
        .zip(&report.observables)
        .collect::<HashMap<_, _>>();
    let references = references.into_inner();
    assert_eq!(observed.len(), next_observable);
    for (shot, (sim, outputs, frames)) in states.iter().enumerate() {
        let values = named_indices
            .iter()
            .map(|(&name, &index)| (name.to_owned(), observed[&index].per_shot[shot]))
            .collect::<HashMap<_, _>>();
        let assignments = actions
            .iter()
            .filter_map(|action| match action {
                Action::Branch { target, condition } => {
                    Some((*target, evaluate(condition, &values, &aliases)))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let projected = graph
            .project_branches_deferred(assignments.iter().copied())
            .expect("observed source branch projects");
        let zx = ZXGraph::try_from(&projected).expect("selected source branch has a ZX graph");
        let generators = match &summary {
            // The composed certificate presents the canonical all-true arms.
            Some(summary) if assignments.iter().all(|&(_, shown)| shown) => summary
                .materialize_stabilizers(&zx, offset)
                .expect("module source has a stabilizer certificate"),
            _ => zx
                .stabilizers()
                .expect("selected source has a stabilizer certificate"),
        };
        let basis = RuntimeStabilizerBasis::for_source_readouts(&generators)
            .expect("source certificate has a runtime basis");
        let named_coordinates = zx
            .action_graph()
            .ordered_nodes()
            .filter_map(|action| {
                let Action::Measure { name, target } = &action.action else {
                    return None;
                };
                let Some(bloq_graph::MeasurementObservable::Concrete(axis)) = action.measurement
                else {
                    panic!("Choi source coordinates require concrete X/Z readouts");
                };
                assert!(matches!(
                    axis,
                    bloq_graph::PauliBasis::X | bloq_graph::PauliBasis::Z
                ));
                Some((
                    name.as_str(),
                    zx.measurement_column(target)
                        .expect("source measurement target has an outcome coordinate"),
                    Pauli::from(axis),
                ))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values.len(),
            generators
                .generators
                .iter()
                .filter(|row| row.is_measurement())
                .count()
        );
        let mut fills = Vec::new();
        for action in &actions {
            if let Action::Resolve { target, condition } = action {
                let value = evaluate(condition, &values, &aliases);
                let NodeKind::Selective(kind) = zx
                    .node_at(*target)
                    .expect("source Resolve target is in the selected graph")
                    .kind
                else {
                    unreachable!()
                };
                fills.push((
                    *target,
                    if value {
                        kind.pauli_if_true()
                    } else {
                        kind.pauli_if_false()
                    },
                ));
            }
        }
        let feedback = |surface: &bloq_graph::Stabilizer| {
            actions.iter().fold(false, |value, action| match action {
                Action::Feedback { targets, condition }
                    if condition
                        .as_ref()
                        .is_none_or(|expr| evaluate(expr, &values, &aliases)) =>
                {
                    value ^ surface.odd_anticommutes_feedback(targets, Some(&zx))
                }
                _ => value,
            })
        };
        let mut rank = BTreeMap::<usize, BTreeSet<usize>>::new();
        // A source Multiplex boundary is a Z-copy isometry: X on its graph
        // leg lifts to X on both the input reference and retained output,
        // while Z can use the reference. The copy also preserves Z_ref Z_out.
        let multiplex = references
            .iter()
            .filter(|(port, _)| {
                graph
                    .get_block(**port)
                    .expect("compiled input port is in the source graph")
                    .port_role()
                    == Some(bloq_graph::PortRole::Multiplex)
            })
            .map(|(&port, &(reference, input))| {
                let output = outputs
                    .iter()
                    .position(|output| output.port == port)
                    .expect("Multiplex input retains its output");
                let frame = frames
                    .iter()
                    .find(|frame| frame.port == port)
                    .expect("logical output has its compiled frame pair");
                let reference_z =
                    EnginePauliString::single(sim.num_qubits(), reference, EnginePauli::Z);
                let operator = &reference_z * &widen(&outputs[output].logical_z, sim.num_qubits());
                let expected = if frame.x.expect("output X frame is available") {
                    -1.0
                } else {
                    1.0
                };
                assert_eq!(
                    sim.peek_observable_expectation(&operator)
                        .expect("Choi observable is Hermitian on live qubits"),
                    expected
                );
                let column = 2 * input + 1;
                rank.insert(
                    column,
                    BTreeSet::from([column, 2 * (references.len() + output) + 1]),
                );
                (port, output)
            })
            .collect::<HashMap<_, _>>();
        let mut check = |surface: &bloq_graph::Stabilizer,
                         expected: bool,
                         correct_outputs: bool| {
            let mut operator = EnginePauliString::new(sim.num_qubits());
            let mut flip = false;
            let mut columns = BTreeSet::new();
            for (&port, &pauli) in &surface.port_stabilizer {
                let index = if let Some(&(reference, index)) = references.get(&port) {
                    operator.set(
                        reference,
                        match pauli {
                            Pauli::I => EnginePauli::I,
                            Pauli::X => EnginePauli::X,
                            Pauli::Y => EnginePauli::Y,
                            Pauli::Z => EnginePauli::Z,
                        },
                    );
                    if pauli & Pauli::X
                        && let Some(&output) = multiplex.get(&port)
                    {
                        operator = &operator * &widen(&outputs[output].logical_x, sim.num_qubits());
                        columns.insert(2 * (references.len() + output));
                        if correct_outputs {
                            flip ^= frames
                                .iter()
                                .find(|frame| frame.port == port)
                                .expect("logical output has its compiled frame pair")
                                .z
                                .expect("output Z frame is available");
                        }
                    }
                    index
                } else {
                    let index = outputs
                        .iter()
                        .position(|output| output.port == port)
                        .expect("source boundary port has a logical output");
                    let output = &outputs[index];
                    assert!(!output.consumed);
                    let frame = frames
                        .iter()
                        .find(|frame| frame.port == port)
                        .expect("logical output has its compiled frame pair");
                    if pauli & Pauli::X {
                        operator = &operator * &widen(&output.logical_x, sim.num_qubits());
                        if correct_outputs {
                            flip ^= frame.z.expect("output Z frame is available");
                        }
                    }
                    if pauli & Pauli::Z {
                        operator = &operator * &widen(&output.logical_z, sim.num_qubits());
                        if correct_outputs {
                            flip ^= frame.x.expect("output X frame is available");
                        }
                    }
                    if pauli == Pauli::Y {
                        operator.set_phase(operator.phase_exponent() + 1);
                    }
                    references.len() + index
                };
                if pauli & Pauli::X {
                    columns.insert(2 * index);
                }
                if pauli & Pauli::Z {
                    columns.insert(2 * index + 1);
                }
            }
            let expectation = sim
                .peek_observable_expectation(&operator)
                .expect("Choi observable is Hermitian on live qubits");
            assert!(
                (expectation.abs() - 1.0).abs() < 1e-9,
                "shot {shot}: Choi correlation {:?} is {expectation}, corrected={correct_outputs}",
                surface.port_stabilizer
            );
            assert_eq!(
                (expectation < 0.0) ^ flip,
                expected,
                "shot {shot}: Choi sign on {:?}",
                surface.port_stabilizer
            );
            while let Some(&first) = columns.first() {
                let Some(pivot) = rank.get(&first) else {
                    rank.insert(first, columns);
                    break;
                };
                columns = columns.symmetric_difference(pivot).copied().collect();
            }
        };
        let branch = basis
            .apply_selective_fills(&fills)
            .expect("observed Resolve choices form a valid source branch");
        let filled = branch.zx_graph().clone();
        let mut readouts = HashMap::<String, bloq_graph::PauliString>::new();
        for (index, recipe) in space.surfaces.iter().enumerate() {
            let bloq_graph::GuardedSurfaceKind::Readout { name, folds } = &recipe.kind else {
                continue;
            };
            let surface = space
                .materialize_surface(index, &filled, |name| values[name])
                .expect("asserted named readout is active in the selected source");
            let predecessors = folds
                .iter()
                .filter(|(_, coefficient)| {
                    space
                        .topology
                        .evaluate_source(*coefficient, |name| values[name])
                })
                .map(|(name, _)| name)
                .collect::<Vec<_>>();
            for &(measured, column, axis) in &named_coordinates {
                assert_eq!(
                    surface.paulis.get(column) & axis,
                    measured == name || predecessors.iter().any(|name| name.as_str() == measured),
                    "shot {shot}: frozen {name} coordinate {measured}"
                );
            }
            let mut named = values[name];
            let mut paulis = surface.paulis.clone();
            for predecessor in predecessors {
                named ^= values[predecessor];
                paulis ^= &readouts[predecessor];
            }
            check(&surface, surface.sign ^ named ^ feedback(&surface), false);
            readouts.insert(name.clone(), paulis);
        }
        for (index, recipe) in space.surfaces.iter().enumerate() {
            if let bloq_graph::GuardedSurfaceKind::OutputFrame { port, basis, .. } = &recipe.kind
                && let Some(surface) =
                    space.materialize_surface(index, &filled, |name| values[name])
            {
                for &(name, column, axis) in &named_coordinates {
                    assert!(
                        !(surface.paulis.get(column) & axis),
                        "shot {shot}: terminal surface retains named coordinate {name}"
                    );
                }
                let symbolic_feedback =
                    recipe
                        .feedbacks
                        .iter()
                        .fold(false, |value, (_, coefficient)| {
                            value
                                ^ space
                                    .topology
                                    .evaluate_source(*coefficient, |name| values[name])
                        });
                assert_eq!(
                    symbolic_feedback,
                    feedback(&surface),
                    "shot {shot}: feedback {port:?}/{basis:?}"
                );
                check(&surface, surface.sign ^ feedback(&surface), true);
            }
        }
        for (row, surface) in branch
            .into_output_correction_surfaces(&zx.output_ports())
            .expect("selected output correction surfaces reconstruct")
        {
            let mut paulis = surface.paulis;
            for ordinal in row.readout_ordinals {
                paulis ^= &readouts[generators.generators[ordinal]
                    .measurement_name()
                    .expect("output correction fold names a measurement")];
            }
            let surface = filled.materialize_stabilizer(paulis);
            check(&surface, surface.sign ^ feedback(&surface), true);
        }
        for &(index, observable) in &logical_readouts {
            let surface = space
                .materialize_surface(index, &filled, |name| values[name])
                .expect("asserted logical readout is active in the selected source");
            check(&surface, observed[&observable].per_shot[shot], false);
        }
        assert_eq!(
            rank.len(),
            references.len() + outputs.len(),
            "the correlations fully determine the Choi state"
        );
        eprintln!(
            "Choi branch {}/{shots}: {} independent correlations",
            shot + 1,
            rank.len()
        );
    }
}
