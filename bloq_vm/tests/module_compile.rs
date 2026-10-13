#![cfg(test)]

use bloq_compile::{CompileConfig, CompileContext, CompileError, SharedCompileCache};
use bloq_graph::{
    Action, Basis, BitBinding, BitOutput, BitRef, Block, BlockGraph, BlockKind, BranchArm,
    CubeKind, Direction, Expr, GalleryItem, MeasureTarget, ModuleCertificationError,
    ModuleCertificationLimits, ModuleInstance, ModuleInterface, Pipe,
};
use bloq_ir::{BloqEdge, ValueRole};
use glam::ivec3;

mod common;

fn compile_blog(source: &str) -> bloq_ir::Bloq {
    let program = bloq_graph::BlockGraph::from_text(source).unwrap();
    let bloq = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap()
        .bloq;
    bloq.validate().unwrap();
    bloq
}

fn run_noiseless(program: &BlockGraph, bloq: &bloq_ir::Bloq) -> bloq_vm::verify::VerifyReport {
    let graph = program.flatten().unwrap();
    let min_z = graph.spans().map_or(0, |(_, _, z)| *z.start());
    let offset = ivec3(0, 0, min_z.checked_neg().unwrap());
    let ccz_ports = program
        .interface
        .quantum_ports
        .iter()
        .filter(|port| {
            port.direction == bloq_graph::PortDirection::Input
                && port.resource_type.eq_ignore_ascii_case("ccz")
        })
        .map(|port| port.position + offset)
        .collect::<Vec<_>>();
    assert_eq!(ccz_ports.len() % 3, 0, "partial CCZ input");
    bloq_vm::run_bloq_with_io(
        bloq,
        16,
        0xADD,
        |_, ctx| {
            for &ports in ccz_ports.as_chunks::<3>().0 {
                ctx.prepare_ccz(ports)?;
            }
            Ok(())
        },
        |_, _| Ok(()),
    )
    .unwrap()
}

#[test]
fn parent_connector_can_wrap_a_one_block_child() {
    let bloq = compile_blog(ONE_BLOCK_INTERIOR_STAGE);
    assert_eq!(bloq.output_frames().len(), 1);
}

#[test]
fn parent_connector_can_run_before_and_after_a_child() {
    let bloq = compile_blog(TWO_BLOCK_INTERIOR_STAGE);
    assert_eq!(bloq.output_frames().len(), 1);
}

#[test]
fn binding_let_keeps_child_feedback_in_its_output_frame() {
    let bloq = compile_blog(BOUND_FEEDBACK_STAGE);
    let feedback = bloq
        .edges()
        .find_map(|edge| {
            matches!(
                edge.edge,
                BloqEdge::Value {
                    output: bloq_ir::ObservableOutput::Corrected,
                    role: ValueRole::FeedbackFold { action: 1 },
                    ..
                }
            )
            .then_some(edge.source)
        })
        .unwrap();
    let frame = bloq
        .output_frames()
        .into_iter()
        .find(|frame| frame.port == ivec3(0, 0, 2))
        .unwrap();

    assert!(bloq.has_path(feedback, frame.z));
}

#[test]
fn dependency_ordered_binding_keeps_producer_feedback_in_its_output_frame() {
    let bloq = compile_blog(PRODUCER_BOUND_FEEDBACK_STAGE);
    let feedback = bloq
        .edges()
        .find_map(|edge| {
            matches!(
                edge.edge,
                BloqEdge::Value {
                    output: bloq_ir::ObservableOutput::Corrected,
                    role: ValueRole::FeedbackFold { action: 0 },
                    ..
                }
            )
            .then_some(edge.source)
        })
        .unwrap();
    let frame = bloq
        .output_frames()
        .into_iter()
        .find(|frame| frame.port == ivec3(0, 0, 2))
        .unwrap();

    assert!(bloq.has_path(feedback, frame.z));
}

#[test]
fn public_y_seam_selects_the_linked_boundary_basis_variant() {
    let bloq = compile_blog(PUBLIC_Y_SEAM);
    assert_eq!(bloq.output_frames().len(), 1);
}

#[test]
fn spatially_rotated_module_interfaces_compile() {
    for (axis, input, output) in [
        ("X", "[0,1,0]", ivec3(0, -1, 0)),
        ("Y", "[-1,0,0]", ivec3(1, 0, 0)),
    ] {
        let source = format!(
            "BLOG 1.0\n\nmodule Stage {{\n  in q_in: data = 0\n  out q_out: data = 2\n  0: Port [0,0,-1] role=input <q_in>\n  1: ZXZ [0,0,0]\n  2: Port [0,0,1] role=output <q_out>\n  0 -> +Z\n  1 -> +Z\n}}\nmodule main {{\n  in q_in: data = 10\n  out q_out: data = 11\n  s: Stage @ [0,0,0] rotate {axis} 90\n  10: Port {input} role=input <q_in>\n  11: Port {output} role=output <q_out>\n  10 -> s.q_in\n  s.q_out -> 11\n}}\n"
        );
        let bloq = compile_blog(&source);

        assert!(
            bloq.output_frames()
                .into_iter()
                .any(|frame| frame.port == output)
        );
    }
}

#[test]
fn module_compile_uses_configured_certification_limits() {
    let program = bloq_graph::BlockGraph::from_text(ONE_BLOCK_INTERIOR_STAGE).unwrap();
    let config = CompileConfig::default().with_certification_limits(ModuleCertificationLimits {
        max_local_columns: 0,
        ..ModuleCertificationLimits::DEFAULT
    });
    let error = CompileContext::new(config).compile(&program).unwrap_err();

    assert!(matches!(
        error,
        CompileError::ModuleCertification(ModuleCertificationError::ResourceLimited {
            phase: "local ZX columns",
            ..
        })
    ));
}

#[test]
fn module_composition_rejects_measurement_surface_on_output() {
    let program = bloq_graph::BlockGraph::from_text(bloq_test::ONE_BIT_ADDER_SOURCE).unwrap();
    let context = CompileContext::new(CompileConfig::default());

    for error in [
        context
            .compile(&program)
            .map(|_| ())
            .expect_err("composition must fail"),
        context
            .compile_object(&program)
            .map(|_| ())
            .expect_err("object build must fail"),
    ] {
        assert!(
            matches!(
                error,
                CompileError::BlockGraph(bloq_graph::BlockGraphError::Stabilizer(
                    bloq_graph::StabilizerError::UnavailableControlParity { .. }
                ))
            ),
            "{error}"
        );
    }
}

#[test]
fn compact_adders_compile_with_default_branch_limits() {
    for item in [GalleryItem::ThreeBitAdder, GalleryItem::TenBitAdder] {
        let program = item.build();
        let artifacts = CompileContext::new(CompileConfig::default())
            .compile(&program)
            .unwrap_or_else(|error| panic!("{item:?}: {error}"));
        artifacts.bloq.validate().unwrap();
        let binary = artifacts.bloq.to_binary();
        assert_eq!(
            bloq_ir::Bloq::from_binary(&binary).unwrap().to_binary(),
            binary
        );
        let text = artifacts.bloq.to_text();
        assert!(
            text.contains(" readout"),
            "earlier decoded parities stay explicit in IR"
        );
        let restored = bloq_ir::Bloq::from_text(&text).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.to_binary(), binary);
    }
}

#[test]
#[ignore = "heavy guarded adder Choi verification; run via just test-full"]
fn guarded_arithmetic_allows_prepared_and_entangled_outputs() {
    let source = GalleryItem::ThreeBitAdder.build().to_blog_text().replacen(
        "module main {",
        r#"module main {
  out prepared: data = 50001
  out bell_a: data = 50003
  out bell_b: data = 50005
  50000: ZXX [-10,-10,0]
  50001: Port [-10,-10,1] role=output
  50000 -> +Z
  50002: XZZ [-12,-10,0]
  50003: Port [-12,-10,1] role=output
  50004: XZZ [-12,-9,0]
  50005: Port [-12,-9,1] role=output
  50002 -> +Y
  50002 -> +Z
  50004 -> +Z
"#,
        1,
    );
    let program = bloq_graph::BlockGraph::from_text(&source).unwrap();
    let bloq = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap()
        .bloq;
    bloq.validate().unwrap();
    assert_eq!(bloq.output_frames().len(), 10);
    common::choi::assert_channel(&program, &bloq, 4);
}

#[test]
fn module_object_links_repeatably_and_rejects_another_target() {
    for item in [GalleryItem::CNOT, GalleryItem::PhaseGradientK4] {
        let program = item.build();
        let context = CompileContext::new(CompileConfig::default());
        let object = context.compile_object(&program).unwrap();

        assert_eq!(object.config(), CompileConfig::default());
        let mut first = context.link_object(&object).unwrap();
        let second = context.link_object(&object).unwrap();
        let detached = CompileContext::new(CompileConfig::default())
            .link_object(&object)
            .unwrap();
        let bytes = first.bloq.to_binary();
        assert_eq!(bytes, second.bloq.to_binary());
        assert_eq!(bytes, detached.bloq.to_binary());
        let node = first.bloq.node_ids().next().unwrap();
        assert!(std::ptr::eq(
            first.bloq.node(node).unwrap(),
            second.bloq.node(node).unwrap(),
        ));
        first.bloq.remove_node(node).unwrap();
        assert_eq!(bytes, second.bloq.to_binary());
        assert_eq!(
            bytes,
            context.link_object(&object).unwrap().bloq.to_binary()
        );
        second.bloq.validate().unwrap();

        let other_target = CompileContext::new(CompileConfig::new(5));
        assert!(matches!(
            other_target.link_object(&object),
            Err(CompileError::CompiledObjectTargetMismatch { .. })
        ));
    }
}

#[test]
fn cold_and_warm_module_compiles_are_byte_identical() {
    for item in [GalleryItem::CNOT, GalleryItem::PhaseGradientK4] {
        let program = item.build();
        let cold = CompileContext::new(CompileConfig::default())
            .compile(&program)
            .unwrap()
            .bloq
            .to_binary();
        let cache = SharedCompileCache::new();
        CompileContext::with_shared_cache(CompileConfig::default(), &cache)
            .compile(&program)
            .unwrap();
        let warm = CompileContext::with_shared_cache(CompileConfig::default(), &cache)
            .compile(&program)
            .unwrap()
            .bloq
            .to_binary();

        assert_eq!(cold, warm);
    }
}

#[test]
fn module_link_matches_flat_oracle_structure_and_execution_contract() {
    let item = GalleryItem::PhaseGradientK4;
    let program = item.build();
    let flat = CompileContext::new(CompileConfig::default())
        .compile(&item.build().flatten().unwrap())
        .unwrap()
        .bloq;
    let modular = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap()
        .bloq;
    let shape = |bloq: &bloq_ir::Bloq| {
        (
            bloq.quantum_node_count(),
            bloq.measurement_count(),
            bloq.qubit_count().unwrap(),
            bloq.edges()
                .filter(|edge| matches!(edge.edge, bloq_ir::BloqEdge::Quantum(_)))
                .count(),
            bloq.logical_inputs().len(),
            bloq.logical_outputs().len(),
            bloq.output_frames().len(),
            bloq.pipe_padding().count(),
        )
    };

    modular.validate().unwrap();
    assert_eq!(shape(&modular), shape(&flat));
    let execution_contract = |report: &bloq_vm::verify::VerifyReport| {
        (
            report.shots,
            report.discarded,
            report.detectors.len(),
            report.all_detectors_constant(),
            report.max_rank,
            report
                .frame_pairs
                .iter()
                .map(|frame| {
                    (
                        frame.port,
                        frame.x_bits.iter().filter(|value| value.is_some()).count(),
                        frame.z_bits.iter().filter(|value| value.is_some()).count(),
                    )
                })
                .collect::<Vec<_>>(),
            report.branch_selectors.len(),
        )
    };
    let modular = run_noiseless(&program, &modular);
    assert_eq!(modular.discarded, 0, "noiseless module shots must survive");
    assert!(modular.all_detectors_constant());
    let flat = run_noiseless(&program, &flat);
    assert_eq!(execution_contract(&modular), execution_contract(&flat));
}

#[test]
fn composed_readout_includes_upstream_feedback_across_module_seams() {
    for hadamard in [false, true] {
        for feedback in [false, true] {
            let seam = if hadamard { "-H>" } else { "->" };
            let correction = if hadamard { "Z" } else { "X" };
            let action = if feedback {
                format!("feedback {correction} 1")
            } else {
                String::new()
            };
            let source = format!(
                r#"BLOG 1.0
module Wire {{
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0,0,0] role=input <q_in>
  1: ZXZ [0,0,1]
  2: Port [0,0,2] role=output <q_out>
  0 -> +Z
  1 -> +Z
  {action}
}}
module Read {{
  in q_in: data = 0
  out result = m
  0: Port [0,0,0] role=input <q_in>
  1: Z [0,0,1]
  0 -> +Z
  m = measure 1
}}
module Select {{
  in q_in: data = 2
  in enable
  2: Port [0,0,-1] role=input <q_in>
  0: ZXZ [0,0,0]
  1: XZ [0,0,1]
  2 -> +Z
  0 -> +Z
  resolve 1 if enable
}}
module main {{
  in q_in: data = 0
  in aux: data = 10
  0: Port [0,0,0] role=input <q_in>
  10: Port [3,0,2] role=input <aux>
  wire: Wire @ [0,0,0]
  read: Read @ [0,0,1]
  select: Select @ [3,0,3]
  0 -> wire.q_in
  10 -> select.q_in
  wire.q_out {seam} read.q_in
  read.result => select.enable
}}
"#,
            );
            let program = bloq_graph::BlockGraph::from_text(&source).unwrap();
            let context = CompileContext::new(CompileConfig::default());
            let composed = context.compile(&program).unwrap().bloq;
            let flat = context.compile(&program.flatten().unwrap()).unwrap().bloq;
            for bloq in [&composed, &flat] {
                let report = bloq_vm::run_bloq_with_io(
                    bloq,
                    8,
                    0xC105ED,
                    |sim, ctx| {
                        let seed = ctx
                            .inputs
                            .iter()
                            .find(|input| input.port == ivec3(0, 0, 0))
                            .unwrap()
                            .qubit;
                        let bit = ctx.shot % 2 != 0;
                        if hadamard {
                            if bit {
                                sim.z(seed);
                            }
                        } else {
                            sim.h(seed);
                            if bit {
                                sim.x(seed);
                            }
                        }
                        Ok(())
                    },
                    |_, _| Ok(()),
                )
                .unwrap();
                assert_eq!(report.discarded, 0);
                assert!(report.all_detectors_constant());
                assert_eq!(
                    report.branch_selectors,
                    (0..8)
                        .map(|shot| (shot % 2 != 0) ^ feedback)
                        .collect::<Vec<_>>(),
                    "H={hadamard}, feedback={feedback}"
                );
            }
        }
    }
}

fn producer_module() -> BlockGraph {
    let mut producer_body = BlockGraph::new();
    producer_body.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
    producer_body
        .set_actions(vec![Action::Measure {
            target: MeasureTarget::Node(ivec3(0, 0, 0)),
            name: "m".into(),
        }])
        .unwrap();
    BlockGraph::definition(
        "Producer",
        producer_body,
        ModuleInterface {
            bit_outputs: vec![BitOutput {
                name: "fire".into(),
                expr: Expr::Var("m".into()),
            }],
            ..ModuleInterface::default()
        },
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

#[test]
fn child_branch_can_read_a_bound_sibling_export() {
    let producer = producer_module();

    let prefix = ivec3(0, 0, 0);
    let target = ivec3(0, 0, 1);
    let mut consumer_body = BlockGraph::new();
    consumer_body.add_block(Block::new(prefix, BlockKind::Cube(CubeKind::ZXZ)));
    let false_arm = BranchArm::new(
        vec![Block::new(target, BlockKind::Measurement(Basis::Z))],
        vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
    );
    let true_arm = BranchArm::new(
        vec![Block::new(target, BlockKind::Cube(CubeKind::XZX))],
        vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
    );
    let branch = consumer_body
        .try_add_branch_region("choice", false_arm, true_arm)
        .unwrap();
    consumer_body
        .set_actions_with_inputs(
            vec![Action::Branch {
                target: branch,
                condition: Expr::Var("select".into()),
            }],
            ["select".to_string()],
        )
        .unwrap();
    let consumer = BlockGraph::definition(
        "Consumer",
        consumer_body,
        ModuleInterface {
            bit_inputs: vec!["select".into()],
            ..ModuleInterface::default()
        },
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );

    let root = BlockGraph::definition(
        "main",
        BlockGraph::new(),
        ModuleInterface::default(),
        vec![
            ModuleInstance {
                name: "producer".into(),
                definition: "Producer".into(),
                rotation: Default::default(),
                translation: ivec3(0, 0, 0),
            },
            ModuleInstance {
                name: "consumer_a".into(),
                definition: "Consumer".into(),
                rotation: Default::default(),
                translation: ivec3(3, 0, 0),
            },
            ModuleInstance {
                name: "consumer_b".into(),
                definition: "Consumer".into(),
                rotation: Default::default(),
                translation: ivec3(6, 0, 0),
            },
        ],
        Vec::new(),
        ["consumer_a", "consumer_b"]
            .into_iter()
            .map(|consumer| BitBinding {
                source: BitRef {
                    instance: Some("producer".into()),
                    bit: "fire".into(),
                },
                target_instance: consumer.into(),
                target_bit: "select".into(),
            })
            .collect(),
    );
    let program = BlockGraph::from_definitions(vec![producer, consumer, root]).unwrap();
    let artifacts = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap();
    artifacts.bloq.validate().unwrap();
}

#[test]
fn composite_branch_can_read_a_child_export() {
    let producer = producer_module();

    let prefix = ivec3(3, 0, 0);
    let target = ivec3(3, 0, 1);
    let mut body = BlockGraph::new();
    body.add_block(Block::new(prefix, BlockKind::Cube(CubeKind::ZXZ)));
    let branch = body
        .try_add_branch_region(
            "choice",
            BranchArm::new(
                vec![Block::new(target, BlockKind::Measurement(Basis::Z))],
                vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
            ),
            BranchArm::new(
                vec![Block::new(target, BlockKind::Cube(CubeKind::XZX))],
                vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
            ),
        )
        .unwrap();
    body.set_actions_with_inputs(
        vec![Action::Branch {
            target: branch,
            condition: Expr::Var("producer.fire".into()),
        }],
        ["producer.fire".to_string()],
    )
    .unwrap();
    let root = BlockGraph::definition(
        "main",
        body,
        ModuleInterface::default(),
        vec![ModuleInstance {
            name: "producer".into(),
            definition: "Producer".into(),
            rotation: Default::default(),
            translation: ivec3(0, 0, 0),
        }],
        Vec::new(),
        Vec::new(),
    );
    let program = BlockGraph::from_definitions(vec![producer, root]).unwrap();
    let artifacts = CompileContext::new(CompileConfig::default())
        .compile(&program)
        .unwrap();
    artifacts.bloq.validate().unwrap();
}

const ONE_BLOCK_INTERIOR_STAGE: &str = r#"BLOG 1.0

module Stage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, -1] role=input <q_in>
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] role=output <q_out>
  0 -> +Z
  1 -> +Z
}

module main {
  in q_in: data = 100
  out q_out: data = 900
  s: Stage @ [0, 0, 5]
  100: Port [0, 0, 3] role=input <q_in>
  200: ZXZ [0, 0, 4]
  201: ZXZ [0, 0, 6]
  900: Port [0, 0, 7] role=output <q_out>
  100 -> +Z
  200 -> s.q_in
  s.q_out -> 201
  201 -> +Z
}
"#;

const TWO_BLOCK_INTERIOR_STAGE: &str = r#"BLOG 1.0

module Stage {
  in q_in: data = 0
  out q_out: data = 3
  0: Port [0, 0, -1] role=input <q_in>
  1: ZXZ [0, 0, 0]
  2: ZXZ [0, 0, 1]
  3: Port [0, 0, 2] role=output <q_out>
  0 -> +Z
  1 -> +Z
  2 -> +Z
}

module main {
  in q_in: data = 100
  out q_out: data = 900
  s: Stage @ [0, 0, 5]
  100: Port [0, 0, 3] role=input <q_in>
  200: ZXZ [0, 0, 4]
  201: ZXZ [0, 0, 7]
  900: Port [0, 0, 8] role=output <q_out>
  100 -> +Z
  200 -> s.q_in
  s.q_out -> 201
  201 -> +Z
}
"#;

const BOUND_FEEDBACK_STAGE: &str = r#"BLOG 1.0

module Stage {
  in q_in: data = 0
  out q_out: data = 2
  in enable
  0: Port [0, 0, 0] role=input <q_in>
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q_out>
  3: XZX [1, 0, 1]
  4: Y [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 1] -> +Z
  feedback Z 1
}

module main {
  in q_in: data = 0
  out q_out: data = 2
  in enable
  0: Port [0, 0, 0] role=input <q_in>
  2: Port [0, 0, 2] role=output <q_out>
  s: Stage @ [0, 0, 0]
  0 -> s.q_in
  s.q_out -> 2
  enable => s.enable
}
"#;

const PRODUCER_BOUND_FEEDBACK_STAGE: &str = r#"BLOG 1.0

module FrameStage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, 0] role=input <q_in>
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q_out>
  3: XZX [1, 0, 1]
  4: Y [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 1] -> +Z
  feedback Z 1
}

module Producer {
  out fire = t
  0: ZXZ [0, 0, 0]
  m = measure 0
  t = !m
}

module Consumer {
  in enable
  discard if enable
}

module main {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, 0] role=input <q_in>
  2: Port [0, 0, 2] role=output <q_out>
  p: FrameStage @ [0, 0, 0]
  b: Producer @ [5, 0, 0]
  c: Consumer @ [0, 0, 0]
  0 -> p.q_in
  p.q_out -> 2
  b.fire => c.enable
}
"#;

const PUBLIC_Y_SEAM: &str = r#"BLOG 1.0

module YPrep {
  out q: data = 1
  0: Y [0, 0, 0]
  1: Port [0, 0, 1] role=output <q>
  [0, 0, 0] -> +Z
}

module Wire {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, -1] role=input <q_in>
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] role=output <q_out>
  [0, 0, -1] -> +Z
  [0, 0, 0] -> +Z
}

module main {
  out q: data = 10
  y: YPrep @ [0, 0, 0]
  wire: Wire @ [0, 0, 1]
  10: Port [0, 0, 2] role=output <q>
  y.q -> wire.q_in
  wire.q_out -> 10
}
"#;
