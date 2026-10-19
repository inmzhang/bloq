//! Independent tensors for bending either end of a temporal identity wire.
use std::cell::RefCell;
use std::collections::BTreeSet;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryItem, Pipe, PortRole};
use bloq_vm::verify::{logical_signature, signatures_match};
use bloq_vm::{Simulator, run_bloq_with_io};
use glam::{IVec3, ivec3};

mod common;
use common::choi::{
    exact::{BellOutcomes, ExactChoi},
    stabilizer::physical_qubit,
};

fn compile(graph: &BlockGraph) -> bloq_ir::Bloq {
    CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(graph)
        .expect("temporal Port fixture compiles with ideal T sources")
        .bloq
}

fn wire(
    graph: &mut BlockGraph,
    origin: IVec3,
    kind: CubeKind,
    first: PortRole,
    last: PortRole,
    hadamard: bool,
) {
    for (position, role) in [(origin, first), (origin + 2 * IVec3::Z, last)] {
        graph.add_block(
            Block::new(position, BlockKind::Port)
                .with_port_role(role)
                .expect("Port accepts its declared role"),
        );
    }
    graph.add_block(Block::new(origin + IVec3::Z, BlockKind::Cube(kind)));
    graph.add_pipe(Pipe::new(origin, Direction::ZPLUS));
    let pipe = Pipe::new(origin + IVec3::Z, Direction::ZPLUS);
    graph.add_pipe(if hadamard { pipe.with_hadamard() } else { pipe });
}

#[test]
fn temporal_roles_preserve_unitary_channels_and_bell_instrument_sectors() {
    for kind in [CubeKind::ZXZ, CubeKind::XZX] {
        for hadamard in [false, true] {
            for (first, last, inputs) in [
                (PortRole::Auto, PortRole::Auto, 1),
                (PortRole::Output, PortRole::Auto, 0),
                (PortRole::Auto, PortRole::Input, 2),
                (PortRole::Output, PortRole::Input, 1),
            ] {
                let mut graph = BlockGraph::new();
                wire(&mut graph, IVec3::ZERO, kind, first, last, hadamard);
                let mut bloq = compile(&graph);
                let bell = (last == PortRole::Input).then(|| BellOutcomes::expose(&mut bloq));
                bloq.validate().unwrap();
                let bloq = bloq_ir::Bloq::from_binary(&bloq.to_binary()).unwrap();
                let text = bloq.to_text();
                let bloq = bloq_ir::Bloq::from_text(&text).unwrap();
                assert_eq!(bloq.logical_inputs().len(), inputs);
                assert_eq!(bloq.logical_outputs().len(), 2 - inputs);

                // The source tensor is delta_ij (or H_ij), irrespective of which
                // legs are references. Its normalized vector is a two-qubit Choi state.
                let mut ideal = Simulator::with_seed(2, 0);
                ideal.h(0);
                ideal.cx(0, 1).unwrap();
                if hadamard {
                    ideal.h(1);
                }
                let expected = logical_signature(
                    &ideal,
                    &[
                        (&physical_qubit(0, 2), (false, false)),
                        (&physical_qubit(1, 2), (false, false)),
                    ],
                )
                .unwrap();
                let references = RefCell::new(Vec::new());
                let mut checked = 0;
                let mut sectors = BTreeSet::new();
                let shots = if last == PortRole::Input { 64 } else { 8 };
                let report = run_bloq_with_io(
                    &bloq,
                    shots,
                    0xB311,
                    |sim, context| {
                        references.borrow_mut().clear();
                        for input in context.inputs {
                            let reference = sim.num_qubits();
                            sim.cx(input.qubit, reference)?;
                            references.borrow_mut().push((input.port, reference));
                        }
                        Ok(())
                    },
                    |sim, context| {
                        let sector = bell.as_ref().map(|bell| bell.read(context));
                        sectors.extend(sector);
                        let refs = references
                            .borrow()
                            .iter()
                            .map(|&(port, qubit)| (port, physical_qubit(qubit, sim.num_qubits())))
                            .collect::<Vec<_>>();
                        let mut boundaries = refs
                            .iter()
                            .map(|(port, output)| {
                                // Two inputs form a Bell measurement instrument. Its
                                // reference state depends on the observed sector; this
                                // frame compares that sector with the positive Bell tensor.
                                let frame = if inputs == 2 && *port == 2 * IVec3::Z {
                                    let (x, z) = sector.unwrap();
                                    (z, x)
                                } else {
                                    (false, false)
                                };
                                (*port, output, frame)
                            })
                            .collect::<Vec<_>>();
                        for output in context.outputs {
                            assert!(
                                !output.consumed,
                                "private outputs survive later use of the patch"
                            );
                            let frame = context
                                .frames
                                .iter()
                                .find(|frame| frame.port == output.port)
                                .unwrap();
                            boundaries.push((
                                output.port,
                                output,
                                (frame.x.unwrap(), frame.z.unwrap()),
                            ));
                        }
                        boundaries.sort_by_key(|(port, _, _)| port.to_array());
                        let actual = logical_signature(
                            sim,
                            &boundaries
                                .iter()
                                .map(|(_, output, frame)| (*output, *frame))
                                .collect::<Vec<_>>(),
                        )?;
                        assert!(
                            signatures_match(&actual, &expected),
                            "{kind:?} {first:?}/{last:?}, H={hadamard}"
                        );
                        checked += 1;
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(checked, shots);
                assert_eq!(report.discarded, 0);
                assert!(report.all_detectors_constant());
                if last == PortRole::Input {
                    assert_eq!(sectors.len(), 4, "every Bell measurement outcome");
                }
            }
        }
    }
}

#[test]
fn an_explicit_discard_retains_the_positive_bell_effect() {
    use bloq_ir::{BloqEdge, BloqNode, ClassicalExpr, ClassicalNode};

    let mut graph = BlockGraph::new();
    wire(
        &mut graph,
        IVec3::ZERO,
        CubeKind::ZXZ,
        PortRole::Auto,
        PortRole::Input,
        false,
    );
    let mut bloq = compile(&graph);
    let bell = BellOutcomes::expose(&mut bloq);
    let discard = bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
        condition: ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
    }));
    for (slot, node) in bell.nodes.into_iter().enumerate() {
        bloq.add_edge(node, discard, BloqEdge::value(slot as u32));
    }
    bloq.validate().unwrap();
    let expected = (1..16)
        .map(|word| match word {
            5 | 15 => 1.0,
            10 => -1.0,
            _ => 0.0,
        })
        .collect::<Vec<_>>();
    let references = RefCell::new(Vec::new());
    let mut accepted = 0;
    let report = run_bloq_with_io(
        &bloq,
        64,
        17,
        |sim, context| {
            references.borrow_mut().clear();
            for input in context.inputs {
                let reference = sim.num_qubits();
                sim.cx(input.qubit, reference)?;
                references.borrow_mut().push(reference);
            }
            Ok(())
        },
        |sim, context| {
            assert_eq!(bell.read(context), (false, false));
            let outputs = references
                .borrow()
                .iter()
                .map(|&qubit| physical_qubit(qubit, sim.num_qubits()))
                .collect::<Vec<_>>();
            let actual = logical_signature(
                sim,
                &outputs
                    .iter()
                    .map(|output| (output, (false, false)))
                    .collect::<Vec<_>>(),
            )?;
            assert!(signatures_match(&actual, &expected));
            accepted += 1;
            Ok(())
        },
    )
    .unwrap();
    assert!(accepted > 0 && report.discarded > 0);
    assert_eq!(accepted + report.discarded, report.shots);
    assert!(report.all_detectors_constant());
}

#[test]
fn reversed_non_clifford_map_is_transposed_and_outputs_survive_patch_reuse() {
    let mut graph = GalleryItem::THTH.build().flatten().unwrap();
    let output = ivec3(0, 0, 0);
    let input = ivec3(-1, 2, 2);
    graph.set_port_role(output, PortRole::Output).unwrap();
    graph.set_port_role(input, PortRole::Input).unwrap();
    // Reuse the early output's logical patch while its private output stays live.
    let ordinary_input = ivec3(0, 0, 4);
    let ordinary_output = ivec3(0, 0, 6);
    wire(
        &mut graph,
        ordinary_input,
        CubeKind::ZXZ,
        PortRole::Auto,
        PortRole::Auto,
        false,
    );
    let mut bloq = compile(&graph.with_inferred_interface().unwrap());
    let bell = BellOutcomes::expose(&mut bloq);
    assert_eq!(
        bloq.logical_inputs()
            .iter()
            .map(|port| port.port)
            .collect::<Vec<_>>(),
        [input, ordinary_input]
    );
    assert_eq!(
        bloq.logical_outputs()
            .iter()
            .map(|port| port.port)
            .collect::<Vec<_>>(),
        [output, ordinary_output]
    );
    // Deliberately use a different order from the persisted coordinate order.
    let oracle = ExactChoi::new(
        &[ordinary_input, input],
        &[ordinary_output, output],
        |sim| {
            sim.h(1);
            sim.t(1)?;
            sim.h(1);
            sim.t(1)?;
            Ok(())
        },
    );
    let mut sectors = BTreeSet::new();
    let report = oracle.run(&bloq, 64, 0x7A45, |_, context| {
        sectors.insert(bell.read(context));
        Ok(())
    });
    assert_eq!(sectors.len(), 4);
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
}

#[test]
fn causal_vm_executes_ideal_bell_boundaries_with_feedback() {
    let graph = BlockGraph::from_text(
        "BLOG 1.0
module Wire {
  in late: data = 2
  out early: data = 0
  0: Port [0,0,0] role=output
  2: Port [0,0,2] role=input
  1: ZXZ [0,0,1]
  0 -> +Z
  1 -> +Z
}
module main {
  in late: data = 12
  out early: data = 10
  child: Wire @ [0,0,0]
  10: Port [0,0,0] role=output
  12: Port [0,0,2] role=input
  child.early -> 10
  12 -> child.late
}",
    )
    .unwrap();
    let bloq = compile(&graph);
    let program = bloq_vm::lower(&bloq, &bloq_vm::LoweringConfig::default()).unwrap();
    for seed in 0..64 {
        let result = program
            .run(bloq_vm::RuntimeConfig {
                seed,
                decoder: bloq_vm::decoder::MockDecoderConfig {
                    acceptance_probability: 1.0,
                    accepted_accuracy: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
        assert!(!result.artifact.discarded);
        let (x, y, z) = result.logical_bloch(&program.outputs[0]).unwrap();
        assert!(
            (x - 1.0).abs() < 1e-9 && y.abs() < 1e-9 && z.abs() < 1e-9,
            "seed {seed}: Bloch ({x},{y},{z}), input={:?}, output={:?}",
            program.inputs,
            program.outputs
        );
    }
}

#[test]
fn reversed_temporal_ports_keep_signed_y_correlations() {
    let mut graph = BlockGraph::new();
    graph.add_block(
        Block::new(IVec3::ZERO, BlockKind::Port)
            .with_port_role(PortRole::Output)
            .unwrap(),
    );
    graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
    graph.add_block(Block::new(2 * IVec3::Z, BlockKind::Y));
    graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
    graph.add_pipe(Pipe::new(IVec3::Z, Direction::ZPLUS));
    let bloq = compile(&graph);
    // The future Y effect is <+Y|; transposing it to an output gives |-Y>.
    ExactChoi::new(&[], &[IVec3::ZERO], |sim| {
        sim.h(0);
        sim.s(0);
        sim.z(0);
        Ok(())
    })
    .run(&bloq, 8, 0x51, |_, _| Ok(()));

    // S is symmetric, with imaginary coefficients: S^T=S, whereas S^dagger differs.
    let mut graph = GalleryItem::S.build().flatten().unwrap();
    graph.set_port_role(IVec3::ZERO, PortRole::Output).unwrap();
    graph.set_port_role(2 * IVec3::Z, PortRole::Input).unwrap();
    let mut bloq = compile(&graph);
    let bell = BellOutcomes::expose(&mut bloq);
    let mut sectors = BTreeSet::new();
    let report = ExactChoi::new(&[2 * IVec3::Z], &[IVec3::ZERO], |sim| {
        sim.s(0);
        Ok(())
    })
    .run(&bloq, 64, 0x51, |_, context| {
        sectors.insert(bell.read(context));
        Ok(())
    });
    assert_eq!(sectors.len(), 4);
    assert_eq!(report.discarded, 0);
}
