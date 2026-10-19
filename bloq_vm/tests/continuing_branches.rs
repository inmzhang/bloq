#![cfg(test)]

use bloq_compile::{CompileConfig, CompileContext};
use bloq_ir::{Bloq, NodeProvenance};
use bloq_vm::run_bloq_with_io;
use bloq_vm::{EnginePauli, EnginePauliString};

mod common;
use common::choi::stabilizer::{assert_stabilizer_state, physical_qubit};

fn assert_noiseless(report: &bloq_vm::verify::VerifyReport) {
    assert_eq!(report.discarded, 0);
    assert!(
        report
            .detectors
            .iter()
            .any(|detector| detector.per_shot.len() == report.shots)
    );
    assert!(
        report
            .detectors
            .iter()
            .all(|detector| detector.per_shot.iter().all(|bit| !bit))
    );
}

fn source() -> String {
    let mut source = include_str!("../../docs/fixtures/conditional_cz_strip.blog").to_owned();
    for bit in 0..3 {
        source = source.replace(&format!("in enable{bit}"), &format!(
            "in control{bit}: data = {}\n  {}: Port [4, {bit}, -3] role=input\n  {}: ZXZ [4, {bit}, -2]\n  [4, {bit}, -3] -> +Z\n  enable{bit} = measure {}",
            80 + 2 * bit, 80 + 2 * bit, 81 + 2 * bit, 81 + 2 * bit));
    }
    source.replace(
        "3: Port [0, 0, 1] role=output",
        "3: Port [0, 0, 2] role=output\n  8: XZX [0, 0, 1]\n  [0, 0, 1] -H> +Z",
    )
}

#[test]
fn three_independent_cz_choices_share_members_and_continue() {
    let ast = bloq_graph::parse_blog_program_to_ast(&source()).unwrap();
    let program = bloq_graph::lower_blog_graph_ast_deferred(&ast).unwrap();
    let compiler = CompileContext::new(CompileConfig::new(3));
    let bloq = compiler.compile(&program).unwrap().bloq;
    assert!(
        bloq.quantum_nodes()
            .any(|(_, node)| !node.guards.is_empty())
    );
    assert_eq!(
        bloq.nodes()
            .filter(|(_, node)| matches!(node.provenance, NodeProvenance::BranchSelector { .. }))
            .count(),
        3
    );
    for mask in 0..8 {
        let choices = (0..3)
            .map(|bit| (format!("cz{bit}"), mask & (1 << bit) != 0))
            .collect();
        let pinned = bloq.pin_membership(&choices).unwrap();
        assert!(!pinned.has_conditional_membership());
        bloq_stim::emit_bloq_stim(&pinned).unwrap();
        let noisy = bloq_stim::emit_bloq_stim_with(
            &pinned,
            &bloq_stim::BloqStimOptions::new()
                .with_noise(&bloq_ir::circuit::NoiseModel::uniform_depolarizing(1e-3)),
        )
        .unwrap();
        let circuit: stim::Circuit = noisy.parse().unwrap();
        circuit.detector_error_model().unwrap();
    }
    let bloq = Bloq::from_binary(&bloq.to_binary()).unwrap();
    let bloq = Bloq::from_text(&bloq.to_text()).unwrap();
    bloq.validate().unwrap();
    let mut masks = std::collections::BTreeSet::new();
    let references = std::cell::Cell::new([0usize; 4]);
    let report = run_bloq_with_io(
        &bloq,
        64,
        7,
        |sim, shot| {
            let mut qubits = [0; 4];
            let incoming = shot.shot.wrapping_mul(0x9e37);
            for seed in shot.inputs.iter().filter(|seed| seed.port.x < 2) {
                let wire = if seed.port.x == 1 {
                    0
                } else {
                    seed.port.y as usize + 1
                };
                let reference = sim.num_qubits();
                sim.reset(reference)?;
                sim.cx(seed.qubit, reference)?;
                if incoming & (1 << wire) != 0 {
                    sim.x(seed.qubit);
                }
                if incoming & (1 << (wire + 4)) != 0 {
                    sim.z(seed.qubit);
                }
                qubits[wire] = reference;
            }
            references.set(qubits);
            Ok(())
        },
        |sim, shot| {
            let enabled = (0..3)
                .map(|bit| {
                    shot.named_branch_selectors
                        .iter()
                        .find(|(name, _)| *name == format!("cz{bit}"))
                        .unwrap()
                        .1
                })
                .collect::<Vec<_>>();
            masks.insert(enabled.clone());
            let outputs = (0..4)
                .map(|wire| {
                    shot.outputs
                        .iter()
                        .find(|output| {
                            if wire == 0 {
                                output.port.x == 1
                            } else {
                                output.port.x == 0 && output.port.y == wire - 1
                            }
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert!(outputs.iter().all(|output| !output.consumed));
            let reference_outputs = references
                .get()
                .map(|q| physical_qubit(q, sim.num_qubits()));
            let mut logicals = outputs
                .iter()
                .map(|output| {
                    let frame = shot
                        .frames
                        .iter()
                        .find(|frame| frame.port == output.port)
                        .unwrap();
                    (*output, (frame.x.unwrap(), frame.z.unwrap()))
                })
                .collect::<Vec<_>>();
            logicals.extend(
                reference_outputs
                    .iter()
                    .map(|output| (output, (false, false))),
            );
            // Independent ideal CZ generators, followed by H on wire 1.
            // Incoming X/Z signs retain the frame-transport regression.
            let generators = (0..8)
                .map(|generator| {
                    let origin = generator % 4;
                    let x_generator = generator < 4;
                    let mut product = EnginePauliString::single(
                        8,
                        4 + origin,
                        if x_generator {
                            EnginePauli::X
                        } else {
                            EnginePauli::Z
                        },
                    );
                    let incoming = shot.shot.wrapping_mul(0x9e37);
                    let sign = incoming & (1 << (origin + if x_generator { 4 } else { 0 })) != 0;
                    product.set_phase(2 * i32::from(sign));
                    for wire in 0..4 {
                        let x = x_generator && wire == origin;
                        let z = if !x_generator {
                            wire == origin
                        } else if origin == 0 {
                            wire > 0 && enabled[wire - 1]
                        } else {
                            wire == 0 && enabled[origin - 1]
                        };
                        let (x, z) = if wire == 1 { (z, x) } else { (x, z) };
                        product.set(
                            wire,
                            match (x, z) {
                                (false, false) => EnginePauli::I,
                                (true, false) => EnginePauli::X,
                                (false, true) => EnginePauli::Z,
                                (true, true) => EnginePauli::Y,
                            },
                        );
                    }
                    product
                })
                .collect::<Vec<_>>();
            assert_stabilizer_state(sim, &logicals, &generators);
            Ok(())
        },
    )
    .unwrap();
    assert_noiseless(&report);
    assert_eq!(masks.len(), 8, "every measured selector combination ran");
}

#[test]
fn continuing_membership_keeps_native_t_and_selective_execution() {
    let ast = bloq_graph::parse_blog_program_to_ast(&source()).unwrap();
    let program = bloq_graph::lower_blog_graph_ast_deferred(&ast).unwrap();
    let mut graph = program.flatten().unwrap();
    let t = bloq_graph::GalleryItem::T
        .build()
        .flatten()
        .unwrap()
        .shift_positions(glam::ivec3(8, 0, 0))
        .unwrap();
    for block in t.blocks() {
        graph.try_add_block(block.clone()).unwrap();
    }
    for pipe in t.pipes() {
        graph.try_add_pipe(pipe.clone()).unwrap();
    }
    let mut actions = graph.actions();
    actions.extend(t.actions());
    let y = bloq_graph::GalleryItem::YMemory
        .build()
        .flatten()
        .unwrap()
        .shift_positions(glam::ivec3(16, 0, 0))
        .unwrap();
    for block in y.blocks() {
        graph.try_add_block(block.clone()).unwrap();
    }
    for pipe in y.pipes() {
        graph.try_add_pipe(pipe.clone()).unwrap();
    }
    actions.push(bloq_graph::Action::Measure {
        target: bloq_graph::MeasureTarget::Node(glam::ivec3(16, 0, 2)),
        name: "y_cap".into(),
    });
    for (index, kind) in [
        bloq_graph::SelectiveKind::XY,
        bloq_graph::SelectiveKind::XZ,
        bloq_graph::SelectiveKind::YZ,
    ]
    .into_iter()
    .enumerate()
    {
        let x = 20 + 4 * index as i32;
        let port = glam::ivec3(x, 0, 0);
        let cap = glam::ivec3(x, 0, 2);
        graph
            .try_add_block(
                bloq_graph::Block::new(port, bloq_graph::BlockKind::Port)
                    .with_port_role(bloq_graph::PortRole::Input)
                    .unwrap(),
            )
            .unwrap();
        graph
            .try_add_block(bloq_graph::Block::new(
                glam::ivec3(x, 0, 1),
                bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ))
            .unwrap();
        graph
            .try_add_block(bloq_graph::Block::new(
                cap,
                bloq_graph::BlockKind::Selective(kind),
            ))
            .unwrap();
        graph
            .try_add_pipe(bloq_graph::Pipe::new(port, bloq_graph::Direction::ZPLUS))
            .unwrap();
        graph
            .try_add_pipe(bloq_graph::Pipe::new(
                glam::ivec3(x, 0, 1),
                bloq_graph::Direction::ZPLUS,
            ))
            .unwrap();
        actions.push(bloq_graph::Action::Resolve {
            target: cap,
            condition: bloq_graph::Expr::Var(format!("enable{index}")),
        });
        actions.push(bloq_graph::Action::Measure {
            target: bloq_graph::MeasureTarget::Node(cap),
            name: format!("selective{index}"),
        });
    }
    graph.set_actions_deferred(actions).unwrap();
    let bloq = bloq_compile::compile(&graph, 3).unwrap();
    let report = bloq_vm::run_bloq(&bloq, 4, 11).unwrap();
    assert_noiseless(&report);
}

#[test]
fn modules_with_mixed_continuing_choices_run_noiselessly() {
    let source = "BLOG 1.0
module Gate {
  0: ZXZ [0,0,0]
  1: ZXZ [0,0,2]
  9: ZXZ [2,0,0]
  branch b {
    false {
      2: XZX [0,0,1]
      0 -H> +Z
      [0,0,1] -H> +Z
    }
    true {
      3: ZXZ [0,0,1]
      0 -> +Z
      [0,0,1] -> +Z
    }
  }
  m = measure 9
  resolve b if m
}

module main {
  a: Gate @ [0,0,0]
  b: Gate @ [4,0,0]
}
";
    let program = bloq_graph::BlockGraph::from_text(source).unwrap();
    let compiler = CompileContext::new(CompileConfig::default());
    let bloq = compiler.compile(&program).unwrap().bloq;
    assert_noiseless(&bloq_vm::run_bloq(&bloq, 8, 13).unwrap());
}

#[test]
#[ignore = "heavy coherent adder execution with all CCZ resources; run via just test-full"]
fn adder_preserves_coherent_state_with_adaptive_corrections() {
    let program = bloq_test::benchmark::controlled_adder(3);
    let graph = program.flatten().unwrap();
    let offset = glam::IVec3::new(0, 0, -*graph.spans().unwrap().2.start());
    let resources = program
        .interface
        .quantum_ports
        .iter()
        .filter(|port| port.resource_type == "ccz")
        .map(|port| port.position + offset)
        .collect::<Vec<_>>();
    let (resources, remainder) = resources.as_chunks::<3>();
    assert!(remainder.is_empty());
    let config = CompileConfig::default();
    let compiler = CompileContext::new(config);
    let bloq = compiler.compile(&program).unwrap().bloq;
    assert_eq!(
        bloq.nodes()
            .filter(|(_, node)| matches!(&node.provenance, NodeProvenance::BranchSelector { name } if graph.branch_by_name(name).is_some()))
            .count(),
        3
    );
    bloq.validate().unwrap();
    let report = run_bloq_with_io(
        &bloq,
        4,
        0xC0FFEE,
        |_, ctx| {
            for &group in resources {
                ctx.prepare_ccz(group)?;
            }
            Ok(())
        },
        |sim, ctx| {
            // Controlled addition is a phase-free permutation, so it must
            // preserve |+> on all data wires, for every sampled erase branch.
            assert_eq!(ctx.outputs.len(), 7);
            let logicals = ctx
                .outputs
                .iter()
                .map(|output| {
                    let frame = ctx
                        .frames
                        .iter()
                        .find(|frame| frame.port == output.port)
                        .unwrap();
                    (output, (frame.x.unwrap(), frame.z.unwrap()))
                })
                .collect::<Vec<_>>();
            let generators = (0..7)
                .map(|wire| EnginePauliString::single(7, wire, EnginePauli::X))
                .collect::<Vec<_>>();
            assert_stabilizer_state(sim, &logicals, &generators);
            Ok(())
        },
    )
    .unwrap();
    assert_noiseless(&report);
    assert!(report.max_rank > 1);
}

#[test]
#[ignore = "64-bit compiler scaling; run with --release --ignored"]
fn large_adder_compiles_without_an_implicit_ir_audit() {
    let program = bloq_test::benchmark::controlled_adder(64);
    let artifacts = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap();
    assert!(!artifacts.bloq.to_binary().is_empty());
}

#[test]
#[ignore = "ten-bit registry and four reference compilations; run with --release"]
fn ten_bit_adder_verifies_uniform_and_alternating_choices() {
    let program = bloq_test::benchmark::controlled_adder(10);
    // This budget bounds local patterns. A joint-mask compiler would reject
    // the ten independent structural choices before producing an artifact.
    let limits = bloq_graph::ModuleCertificationLimits {
        max_guarded_domain_size: 256,
        ..bloq_graph::ModuleCertificationLimits::DEFAULT
    };
    let compiler = CompileContext::new(CompileConfig::default().with_certification_limits(limits));
    let artifacts = compiler.compile(&program).unwrap();
    eprintln!(
        "ten-bit shared registry compiled in {:?}",
        artifacts.compile_duration
    );
    let bloq = artifacts.bloq;
    let graph = program.flatten().unwrap();
    assert_eq!(
        bloq.nodes()
            .filter(|(_, node)| matches!(&node.provenance, NodeProvenance::BranchSelector { name } if graph.branch_by_name(name).is_some()))
            .count(),
        10
    );
    // Both arms at every position and all four adjacent-arm combinations.
    for mask in [0u16, 0b11_1111_1111, 0b01_0101_0101, 0b10_1010_1010] {
        let mut choices: std::collections::BTreeMap<_, _> = (0..10)
            .map(|bit| (format!("cz{bit}"), mask & (1 << bit) != 0))
            .collect();
        let selectors = bloq
            .nodes()
            .filter_map(|(id, node)| match &node.provenance {
                NodeProvenance::BranchSelector { name } => Some((id, name)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut pins = selectors
            .iter()
            .filter_map(|&(id, name)| choices.get(name).map(|&value| (id, value)))
            .collect::<Vec<_>>();
        let mut predicates = bloq_ir::lowering::PredicateAnalysis::new(bloq.top());
        assert!(predicates.assignment_reachable(&pins).unwrap());
        // Complete the selected structural path with reachable measurement
        // choices, without enumerating their joint domain.
        for (id, name) in selectors {
            if choices.contains_key(name) {
                continue;
            }
            pins.push((id, false));
            if !predicates.assignment_reachable(&pins).unwrap() {
                pins.last_mut().unwrap().1 = true;
            }
            choices.insert(name.clone(), pins.last().unwrap().1);
        }
        let selected = bloq
            .pin_membership(&choices)
            .unwrap_or_else(|error| panic!("mask {mask:010b}: {error}"));
        selected.validate().unwrap();
        let mut modules = program
            .modules()
            .map(bloq_graph::BlockGraph::clone_local_definition)
            .collect::<Vec<_>>();
        let root = modules
            .iter_mut()
            .find(|module| module.name == "main")
            .unwrap();
        let assignment = root
            .branch_definitions()
            .iter()
            .map(|region| (region.target, choices[&region.name]))
            .collect::<Vec<_>>();
        let body = root.project_branches_in_definition(assignment).unwrap();
        root.replace_local_body(body);
        let reference = compiler
            .compile(&bloq_graph::BlockGraph::from_definitions(modules).unwrap())
            .unwrap()
            .bloq;
        reference.validate().unwrap();
        let interface = |program: &Bloq| {
            program
                .logical_outputs()
                .iter()
                .map(|output| (output.port, output.x.clone(), output.z.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            interface(&selected),
            interface(&reference),
            "mask {mask:010b}"
        );
        // Native T-state simulation of the full adder is exponentially costly.
        // The shared Choi suite and three-bit test exercise physical execution.
    }
}
