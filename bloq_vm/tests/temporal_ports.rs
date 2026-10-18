//! Independent tensors for bending either end of a temporal identity wire.
use std::cell::RefCell;

use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryItem, Pipe, PortRole};
use bloq_vm::verify::{logical_signature, signatures_match};
use bloq_vm::{Simulator, run_bloq_with_io};
use glam::{IVec3, ivec3};

mod common;
use common::choi::{exact::ExactChoi, stabilizer::physical_qubit};

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
fn temporal_roles_are_the_same_independent_bell_tensor() {
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
                let bloq = if first == PortRole::Output && last == PortRole::Input && hadamard {
                    bloq_compile::compile_clifford_proxy(
                        bloq_compile::CompileConfig::new(3),
                        &graph,
                        &[],
                    )
                    .unwrap()
                    .bloq
                } else {
                    bloq_compile::compile(&graph, 3).unwrap()
                };
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
                let shots = if last == PortRole::Input { 128 } else { 8 };
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
                        let refs = references
                            .borrow()
                            .iter()
                            .map(|&(port, qubit)| (port, physical_qubit(qubit, sim.num_qubits())))
                            .collect::<Vec<_>>();
                        let mut boundaries = refs
                            .iter()
                            .map(|(port, output)| (*port, output, (false, false)))
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
                assert_eq!(checked, shots - report.discarded);
                assert!(report.all_detectors_constant());
                if last == PortRole::Input {
                    // Two independent positive Bell outcomes: probability 1/4.
                    assert!(
                        (16..=52).contains(&checked),
                        "Bell cap survivors: {checked}/{shots}"
                    );
                } else {
                    assert_eq!(report.discarded, 0);
                }
            }
        }
    }
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
    let bloq = bloq_compile::compile(&graph.with_inferred_interface().unwrap(), 3).unwrap();
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
    )
    .postselected();
    let report = oracle.run(&bloq, 64, 0x7A45, |_, _| Ok(()));
    assert!((8..=26).contains(&(report.shots - report.discarded)));
    assert!(report.all_detectors_constant());
}

#[test]
fn causal_vm_executes_ideal_bell_boundaries_with_postselection() {
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
    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    let program = bloq_vm::lower(&bloq, &bloq_vm::LoweringConfig::default()).unwrap();
    let mut accepted = 0;
    let mut rejected = 0;
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
        if result.artifact.discarded {
            rejected += 1;
        } else {
            accepted += 1;
            let (x, y, z) = result.logical_bloch(&program.outputs[0]).unwrap();
            assert!(
                (x - 1.0).abs() < 1e-9 && y.abs() < 1e-9 && z.abs() < 1e-9,
                "seed {seed}: Bloch ({x},{y},{z}), input={:?}, output={:?}",
                program.inputs,
                program.outputs
            );
        }
    }
    assert!(accepted > 0 && rejected > 0);
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
    let bloq = bloq_compile::compile(&graph, 3).unwrap();
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
    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    ExactChoi::new(&[2 * IVec3::Z], &[IVec3::ZERO], |sim| {
        sim.s(0);
        Ok(())
    })
    .postselected()
    .run(&bloq, 64, 0x51, |_, _| Ok(()));
}
