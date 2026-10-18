#![cfg(test)]

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::GalleryItem;
use bloq_ir::circuit::{CoordCircuit, DetectorParity, GateType, NoiseModel, Op, PauliBasis};
use bloq_ir::lowering::{
    BloqTemplate, InstanceBoundaryOperator, InstanceMeasurement, TemplateDetectorScope,
    TemplateInstance, TemplateInstanceId,
};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BoundaryFace, ClassicalExpr, ClassicalNode, NodeDetector,
    QuantumNode, TemplateDetector,
};
use bloq_vm::decoder::MockDecoderConfig;
use bloq_vm::instruction::{
    BoolOp, BoundaryBinding, BoundaryFlow, DecodeRequest, DecodeTiming, FrontierInitializer,
    Instruction, MemoryCycle, Moment, MomentKind, Pauli, PauliProduct, Program, QuantumAlternative,
    QuantumOp, QuantumStream, QuantumTask, RecordParity, SourceRole, Stream, Task, TaskFunction,
    TaskOrigin,
};
use bloq_vm::runtime::{
    ExecutionArtifact, ExecutionEvent, MemoryKind, RuntimeConfig, RuntimeError, WaitReason, run,
};
use bloq_vm::{LoweringConfig, lower, run_bloq};
use glam::IVec2;

fn run_dynamic(bloq: &Bloq) -> ExecutionArtifact {
    let program = lower(bloq, &LoweringConfig::default()).unwrap();
    run(
        &program,
        RuntimeConfig {
            decoder: MockDecoderConfig {
                acceptance_probability: 1.0,
                accepted_accuracy: 1.0,
                rejected_accuracy: 1.0,
                acceptance_script: vec![true; 128],
                ..MockDecoderConfig::default()
            },
            seed: 7,
            ..RuntimeConfig::default()
        },
    )
    .unwrap()
    .artifact
}

fn quantum_node(template: bloq_ir::TemplateId, instance: u32) -> BloqNode {
    BloqNode::quantum(QuantumNode {
        instances: vec![TemplateInstance::new(
            TemplateInstanceId(instance),
            template,
            IVec2::ZERO,
        )],
        ..QuantumNode::default()
    })
}

fn empty_memory_cycle(round_duration: f64, moment_duration: Option<f64>) -> MemoryCycle {
    MemoryCycle {
        stream: QuantumStream {
            moments: moment_duration.map_or_else(
                || Vec::new().into_boxed_slice(),
                |duration| {
                    vec![Moment {
                        kind: None,
                        duration,
                        operations: Box::new([]),
                    }]
                    .into_boxed_slice()
                },
            ),
        },
        round_duration,
        detectors: Box::new([]),
        boundary_flows: Box::new([]),
        initializers: Box::new([]),
    }
}

#[test]
fn activation_and_signed_detectors_match_the_verifier() {
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
    let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
    let mut template = BloqTemplate::new(circuit);
    template.detectors.push(TemplateDetector {
        scope: TemplateDetectorScope::TopLevel,
        parity: DetectorParity::from_measurements([measurement]),
        coords: None,
    });

    let mut bloq = Bloq::new();
    let template = bloq.add_template(template);
    let physical = bloq.add_node(quantum_node(template, 0));

    let mut signed = BloqNode::from_members(Vec::new());
    signed.expect_quantum_mut().detectors.push(NodeDetector {
        parity: DetectorParity::default().with_sign(true),
        coords: None,
    });
    let signed = bloq.add_node(signed);
    bloq.add_edge(physical, signed, BloqEdge::Order);

    let disabled = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
        expr: ClassicalExpr::Const(false),
    }));
    let mut inactive = BloqNode::classical(ClassicalNode::Compute {
        expr: ClassicalExpr::Const(true),
    });
    inactive.activation = Some(0);
    let inactive = bloq.add_node(inactive);
    bloq.add_edge(disabled, inactive, BloqEdge::value(0));
    bloq.add_edge(signed, inactive, BloqEdge::Order);

    let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
    bloq.add_edge(inactive, observable, BloqEdge::compose(1));
    bloq.add_edge(inactive, observable, BloqEdge::Order);

    let legacy = run_bloq(&bloq, 1, 7).unwrap();
    let dynamic = run_dynamic(&bloq);
    assert_eq!(
        dynamic
            .detectors
            .iter()
            .filter(|detector| detector.committed)
            .map(|detector| detector.value)
            .collect::<Vec<_>>(),
        legacy
            .detectors
            .iter()
            .flat_map(|detector| detector.per_shot.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        dynamic
            .observables
            .iter()
            .find(|observable| observable.committed && observable.index == 0)
            .unwrap()
            .value,
        Some(legacy.observables[0].per_shot[0])
    );
}

#[test]
fn shared_readout_recipe_preserves_binding_and_activation() {
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
    let mut bloq = Bloq::new();
    let template = bloq.add_template(BloqTemplate::new(circuit));
    let quantum = bloq.add_node(quantum_node(template, 0));
    let include = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
        Vec::new(),
        vec![InstanceBoundaryOperator {
            instance: TemplateInstanceId(0),
            face: BoundaryFace::Output,
            operator: [(IVec2::ZERO, bloq_ir::circuit::Pauli::Z)]
                .into_iter()
                .collect(),
        }],
    )));
    bloq.add_edge(quantum, include, BloqEdge::Order);
    let bit = |bloq: &mut Bloq, value| {
        bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(value),
        }))
    };
    let zero = bit(&mut bloq, false);
    let one = bit(&mut bloq, true);
    let mut recipe =
        BloqNode::classical(ClassicalNode::observable_fragment(Vec::new(), Vec::new()));
    recipe.activation = Some(2);
    let recipe = bloq.add_node(recipe);
    bloq.add_edge(zero, recipe, BloqEdge::value(0));
    bloq.add_edge(include, recipe, BloqEdge::compose(1));
    bloq.add_edge(one, recipe, BloqEdge::value(2));

    let mut enabled = BloqNode::classical(ClassicalNode::observable(0));
    enabled.activation = Some(1);
    let enabled = bloq.add_node(enabled);
    bloq.add_edge(recipe, enabled, BloqEdge::compose(0));
    bloq.add_edge(one, enabled, BloqEdge::value(1));
    let shared = bloq.add_node(BloqNode::classical(ClassicalNode::observable(1)));
    bloq.add_edge(recipe, shared, BloqEdge::compose(0));

    let mut disabled_recipe =
        BloqNode::classical(ClassicalNode::observable_fragment(Vec::new(), Vec::new()));
    disabled_recipe.activation = Some(2);
    let disabled_recipe = bloq.add_node(disabled_recipe);
    bloq.add_edge(one, disabled_recipe, BloqEdge::value(0));
    bloq.add_edge(include, disabled_recipe, BloqEdge::compose(1));
    bloq.add_edge(zero, disabled_recipe, BloqEdge::value(2));
    let disabled = bloq.add_node(BloqNode::classical(ClassicalNode::observable(2)));
    bloq.add_edge(disabled_recipe, disabled, BloqEdge::compose(0));
    for observable in [enabled, shared, disabled] {
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(observable, consumer, BloqEdge::value(0));
    }

    let lowered = lower(&bloq, &LoweringConfig::default()).unwrap();
    assert_eq!(
        lowered
            .instructions()
            .filter(|instruction| matches!(instruction, Instruction::ReadoutRecipe { .. }))
            .count(),
        3
    );
    let dynamic = run_dynamic(&bloq);
    let values: Vec<_> = dynamic
        .observables
        .iter()
        .map(|observable| (observable.index, observable.raw, observable.value))
        .collect();
    assert_eq!(
        values,
        [
            (0, Some(false), Some(true)),
            (1, Some(false), Some(true)),
            (2, Some(false), Some(false))
        ]
    );
    assert_eq!(dynamic.decoder_decisions.len(), 3);
    let physical = run_bloq(&bloq, 1, 7).unwrap();
    assert_eq!(
        physical
            .observables
            .iter()
            .map(|observable| observable.per_shot[0])
            .collect::<Vec<_>>(),
        [true, true, false]
    );
}

#[test]
fn optimized_shared_output_packets_preserve_multiplicity_and_phase() {
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
    let mut bloq = Bloq::new();
    let template = bloq.add_template(BloqTemplate::new(circuit));
    let quantum = bloq.add_node(quantum_node(template, 0));
    let packet = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
        Vec::new(),
        [bloq_ir::circuit::Pauli::X, bloq_ir::circuit::Pauli::Z]
            .into_iter()
            .map(|pauli| InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: [(IVec2::ZERO, pauli)].into_iter().collect(),
            })
            .collect(),
    )));
    bloq.add_edge(quantum, packet, BloqEdge::Order);
    for (index, copies) in [(0, 2), (1, 4)] {
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(index)));
        for slot in 0..copies {
            bloq.add_edge(packet, observable, BloqEdge::compose(slot));
        }
    }
    bloq.validate().unwrap();
    let mut optimized = bloq.clone();
    optimized.optimize().unwrap();
    optimized.validate().unwrap();

    for program in [&bloq, &optimized] {
        // Each packet retains its ordered product: (XZ)^2 = -I, (XZ)^4 = I.
        let physical = run_bloq(program, 4, 7).unwrap();
        assert_eq!(
            physical
                .observables
                .iter()
                .map(|observable| observable.per_shot.clone())
                .collect::<Vec<_>>(),
            [vec![true; 4], vec![false; 4]],
        );
        let dynamic = run_dynamic(program);
        let mut values = dynamic
            .observables
            .iter()
            .map(|observable| (observable.index, observable.raw, observable.value))
            .collect::<Vec<_>>();
        values.sort_unstable_by_key(|&(index, _, _)| index);
        assert_eq!(
            values,
            [(0, Some(false), Some(true)), (1, Some(false), Some(false))],
        );
    }
}

#[test]
fn deep_shared_binding_recipes_keep_multiplicity_and_phase() {
    let mut program = Program {
        qubit_count: 1,
        bit_count: 2,
        ..Program::default()
    };
    let mut push = |dependencies: Vec<u32>, output, instruction| {
        let id = program.tasks.len() as u32;
        program.tasks.push(Task {
            label: String::new(),
            release: 0.0,
            source: None,
            dependencies: dependencies.into_boxed_slice(),
            activation: None,
            output,
            qubits: Box::new([]),
            duration: 0.0,
            instruction,
        });
        id
    };
    let bind = |pauli| {
        Instruction::Bind(
            vec![BoundaryBinding {
                input: false,
                operator: PauliProduct {
                    negative: false,
                    terms: vec![(0, pauli)].into_boxed_slice(),
                },
            }]
            .into_boxed_slice(),
        )
    };
    let x = push(vec![], None, bind(Pauli::X));
    let z = push(vec![], None, bind(Pauli::Z));
    let mut root = z;
    for _ in 0..1024 {
        root = push(
            vec![root],
            None,
            Instruction::ReadoutRecipe {
                bits: Box::new([]),
                bindings: vec![root].into_boxed_slice(),
            },
        );
    }
    for _ in 0..24 {
        root = push(
            vec![root],
            None,
            Instruction::ReadoutRecipe {
                bits: Box::new([]),
                bindings: vec![root, root].into_boxed_slice(),
            },
        );
    }
    let sign = push(
        vec![x, z],
        None,
        Instruction::ReadoutRecipe {
            bits: Box::new([]),
            bindings: vec![x, z, x, z].into_boxed_slice(),
        },
    );
    push(
        vec![root],
        Some(0),
        Instruction::Observable {
            index: 0,
            bits: Box::new([]),
            bindings: vec![root].into_boxed_slice(),
        },
    );
    push(
        vec![sign],
        Some(1),
        Instruction::Observable {
            index: 1,
            bits: Box::new([]),
            bindings: vec![sign].into_boxed_slice(),
        },
    );
    program.entry.tasks = (0..program.tasks.len() as u32).collect();
    let artifact = run(&program, RuntimeConfig::default()).unwrap().artifact;
    let mut values = artifact
        .observables
        .iter()
        .map(|observable| (observable.index, observable.value))
        .collect::<Vec<_>>();
    values.sort_unstable_by_key(|(index, _)| *index);
    assert_eq!(values, [(0, Some(false)), (1, Some(true))]);
}

#[test]
fn rus_declared_result_is_independent_of_its_restart_source() {
    let bloq = Bloq::from_text(
        "BLOQIR 1\ngraph {\n n0 rus in0 source n0 {\n   body {\n     n0 compute 0\n     n1 compute 1\n     n2 compute !in0\n     n3 compute 0\n     n1 -> n2 value 0\n     result n1\n   }\n }\n n1 observable 0\n n0 -> n1 value 0\n}",
    )
    .unwrap();
    let legacy = run_bloq(&bloq, 1, 7).unwrap();
    let dynamic = run_dynamic(&bloq);
    assert_eq!(legacy.observables[0].per_shot, [true]);
    assert_eq!(dynamic.retries.len(), 1);
    assert!(dynamic.retries[0].accepted);
    assert_eq!(
        dynamic
            .observables
            .iter()
            .find(|observable| observable.committed && observable.index == 0)
            .unwrap()
            .value,
        Some(true)
    );
}

#[test]
fn rejected_attempt_records_stay_uncommitted() {
    let task = |dependencies, output, instruction| Task {
        label: String::new(),
        release: 0.0,
        source: None,
        dependencies,
        activation: None,
        output,
        qubits: Box::new([0]),
        duration: 0.0,
        instruction,
    };
    let body = Stream {
        tasks: Box::new([0, 1, 2, 3]),
        value: Some(1),
        bindings: Box::new([]),
    };
    let program = Program {
        qubit_count: 1,
        bit_count: 4,
        record_count: 1,
        tasks: vec![
            task(
                Box::new([]),
                None,
                Instruction::Quantum(QuantumTask {
                    alternatives: Box::new([QuantumAlternative {
                        when: Box::new([]),
                        stream: QuantumStream {
                            moments: Box::new([Moment {
                                kind: None,
                                duration: 1.0,
                                operations: Box::new([QuantumOp::Measure {
                                    observable: PauliProduct {
                                        negative: false,
                                        terms: Box::new([(0, Pauli::Z)]),
                                    },
                                    records: Box::new([0]),
                                    flip_probability: 0.0,
                                }]),
                            }]),
                        },
                        selected_records: Box::new([0]),
                        excluded_records: Box::new([]),
                        detectors: Box::new([]),
                        restarts: Box::new([]),
                    }]),
                }),
            ),
            task(
                Box::new([0]),
                Some(0),
                Instruction::Accumulate(RecordParity {
                    records: [0].into(),
                    constant: false,
                }),
            ),
            task(
                Box::new([1]),
                Some(1),
                Instruction::Observable {
                    index: 0,
                    bits: Box::new([0]),
                    bindings: Box::new([]),
                },
            ),
            task(
                Box::new([2]),
                Some(2),
                Instruction::Decode(DecodeRequest {
                    output: bloq_ir::ObservableOutput::Flip,
                    observable: 2,
                    index: 0,
                    raw: 1,
                    round_duration: 0.0,
                    timing: DecodeTiming::Measurement,
                }),
            ),
            task(
                Box::new([]),
                Some(3),
                Instruction::Rus {
                    body: body.clone(),
                    restart: BoolOp::Copy(2),
                    owned_qubits: Box::new([0]),
                    attempt_bits: Box::new([0, 1, 2]),
                    attempt_records: Box::new([0]),
                    resource: 0,
                    retry_prepare: QuantumStream {
                        moments: Box::new([Moment {
                            kind: Some(MomentKind::Reset),
                            duration: 1.0,
                            operations: Box::new([QuantumOp::Reset {
                                basis: Pauli::Z,
                                qubit: 0,
                            }]),
                        }]),
                    },
                    decoder_hold: Box::new([]),
                    cultivation_exits: Box::new([]),
                },
            ),
        ],
        task_origins: Vec::new(),
        entry: Stream {
            tasks: Box::new([4]),
            value: Some(3),
            bindings: Box::new([]),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        decoder_latency_rounds: 10,
    };
    let result = run(
        &program,
        RuntimeConfig {
            decoder: MockDecoderConfig {
                acceptance_script: vec![false, true],
                accepted_accuracy: 1.0,
                rejected_accuracy: 0.0,
                ..MockDecoderConfig::default()
            },
            ..RuntimeConfig::default()
        },
    )
    .unwrap();

    assert_eq!(
        result
            .artifact
            .retries
            .iter()
            .map(|attempt| attempt.accepted)
            .collect::<Vec<_>>(),
        [false, true]
    );
    assert_eq!(
        result
            .artifact
            .measurements
            .iter()
            .map(|record| record.committed)
            .collect::<Vec<_>>(),
        [false, true]
    );
    assert_eq!(
        result
            .artifact
            .decoder_decisions
            .iter()
            .map(|decision| (decision.decision.accepted, decision.committed))
            .collect::<Vec<_>>(),
        [(false, false), (true, true)]
    );
    let retry_prepare = result
        .artifact
        .events
        .iter()
        .find_map(|event| match event {
            ExecutionEvent::RetryPrepare {
                attempt: 1,
                start,
                end,
                ..
            } => Some((*start, *end)),
            _ => None,
        })
        .expect("the second attempt re-prepares its source");
    assert!((retry_prepare.1 - retry_prepare.0 - 1.0).abs() < 1e-9);
    let body_start = result
        .artifact
        .events
        .iter()
        .find_map(|event| match event {
            ExecutionEvent::RusAttemptStarted {
                attempt: 1, time, ..
            } => Some(*time),
            _ => None,
        })
        .unwrap();
    assert!((body_start - retry_prepare.1).abs() < 1e-9);
}

#[test]
fn delayed_decode_pair_reuses_one_causal_decision() {
    for delayed_flip in [false, true] {
        let mut measurement = CoordCircuit::new();
        let record = measurement.measure(PauliBasis::Z, [IVec2::ZERO])[0];
        let mut delay = CoordCircuit::new();
        for _ in 0..8 {
            delay.do_gate(GateType::H, [IVec2::X]).unwrap();
            delay.tick();
        }
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(measurement));
        let measure = bloq.add_node(quantum_node(template, 0));
        let template = bloq.add_template(BloqTemplate::new(delay));
        let delay = bloq.add_node(quantum_node(template, 1));
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: record,
            }],
            Vec::new(),
        )));
        bloq.add_edge(measure, raw, BloqEdge::Order);
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(raw, observable, BloqEdge::compose(0));
        let decode = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let flip = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(observable, decode, BloqEdge::value(0));
        bloq.add_edge(observable, flip, BloqEdge::flip(0));
        bloq.add_edge(
            delay,
            if delayed_flip { flip } else { decode },
            BloqEdge::Order,
        );
        bloq.validate().unwrap();
        let program = bloq_vm::lower(
            &bloq,
            &bloq_vm::LoweringConfig {
                decoder_latency_rounds: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let result = run(
            &program,
            RuntimeConfig {
                decoder: MockDecoderConfig {
                    acceptance_script: vec![false, true],
                    accepted_accuracy: 1.0,
                    rejected_accuracy: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.artifact.decoder_decisions.len(), 1);
        assert!(!result.artifact.decoder_decisions[0].decision.accepted);
        assert_eq!(result.artifact.decoder_decisions[0].decision.ready_at, 3.0);
        let completed = result
            .artifact
            .timing
            .iter()
            .filter(|task| {
                matches!(
                    program.tasks[task.task as usize].instruction,
                    Instruction::Decode(_)
                )
            })
            .map(|task| task.end)
            .collect::<Vec<_>>();
        assert_eq!(completed, [3.0, 3.0]);
    }
}

#[test]
fn observable_corrected_and_flip_share_a_solve_and_composition_stays_raw() {
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(GateType::X, [IVec2::ZERO]).unwrap();
    let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
    let mut bloq = Bloq::new();
    let template = bloq.add_template(BloqTemplate::new(circuit));
    let measure = bloq.add_node(quantum_node(template, 0));
    let observable = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
        index: Some(0),
        measurements: vec![InstanceMeasurement {
            instance: TemplateInstanceId(0),
            measurement,
        }],
        operators: vec![],
    }));
    bloq.add_edge(measure, observable, BloqEdge::Order);
    let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
        Vec::new(),
        Vec::new(),
    )));
    let flip = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
        expr: ClassicalExpr::In(0),
    }));
    bloq.add_edge(observable, raw, BloqEdge::compose(0));
    bloq.add_edge(observable, flip, BloqEdge::flip(0));
    bloq.validate().unwrap();
    let program = lower(
        &bloq,
        &LoweringConfig {
            decoder_latency_rounds: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let result = run(
        &program,
        RuntimeConfig {
            decoder: MockDecoderConfig {
                acceptance_script: vec![false],
                rejected_accuracy: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.artifact.decoder_decisions.len(), 1);
    assert!(result.artifact.decoder_decisions[0].decision.flip);
    for (instruction, expected_bit, expected_end) in [
        ("decode", false, 4.0),
        ("raw", true, 2.0),
        ("flip", true, 4.0),
    ] {
        let (task_id, task) = program
            .tasks
            .iter()
            .enumerate()
            .find(|(_, task)| match instruction {
                "decode" => matches!(
                    task.instruction,
                    Instruction::Decode(DecodeRequest {
                        output: bloq_ir::ObservableOutput::Corrected,
                        ..
                    })
                ),
                "raw" => matches!(task.instruction, Instruction::ReadoutRecipe { .. }),
                _ => matches!(
                    task.instruction,
                    Instruction::Decode(DecodeRequest {
                        output: bloq_ir::ObservableOutput::Flip,
                        ..
                    })
                ),
            })
            .unwrap();
        assert_eq!(
            result.artifact.final_bits[task.output.unwrap() as usize],
            Some(expected_bit)
        );
        assert_eq!(
            result
                .artifact
                .timing
                .iter()
                .find(|timing| timing.task as usize == task_id)
                .unwrap()
                .end,
            expected_end
        );
    }
}

#[test]
fn inactive_observable_outputs_stay_zero_without_a_solve() {
    let mut bloq = Bloq::new();
    let guard = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
        expr: ClassicalExpr::Const(false),
    }));
    let mut observable = BloqNode::classical(ClassicalNode::observable(0));
    observable.activation = Some(0);
    let observable = bloq.add_node(observable);
    bloq.add_edge(guard, observable, BloqEdge::value(0));
    for edge in [BloqEdge::value(0), BloqEdge::flip(0)] {
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(observable, consumer, edge);
    }
    bloq.validate().unwrap();
    let program = lower(&bloq, &LoweringConfig::default()).unwrap();
    let result = run(&program, RuntimeConfig::default()).unwrap();
    assert!(result.artifact.decoder_decisions.is_empty());
    for task in &program.tasks {
        if let Instruction::Decode(_) = task.instruction {
            assert_eq!(
                result.artifact.final_bits[task.output.unwrap() as usize],
                Some(false)
            );
        }
    }
}

#[test]
fn memory_frontier_moves_across_one_fault_and_the_downstream_seam() {
    let empty_quantum = Instruction::Quantum(QuantumTask {
        alternatives: Box::new([QuantumAlternative {
            stream: QuantumStream {
                moments: Box::new([Moment {
                    kind: None,
                    duration: 3.0,
                    operations: Box::new([]),
                }]),
            },
            ..QuantumAlternative::default()
        }]),
    });
    let memory = MemoryCycle {
        stream: QuantumStream {
            moments: Box::new([Moment {
                kind: None,
                duration: 1.0,
                operations: Box::new([
                    QuantumOp::ConditionalPauli {
                        basis: Pauli::X,
                        qubit: 0,
                        control: 1,
                    },
                    QuantumOp::Measure {
                        observable: PauliProduct {
                            negative: false,
                            terms: Box::new([(0, Pauli::Z)]),
                        },
                        records: Box::new([0]),
                        flip_probability: 0.0,
                    },
                    QuantumOp::Measure {
                        observable: PauliProduct {
                            negative: false,
                            terms: Box::new([(1, Pauli::Z)]),
                        },
                        records: Box::new([1]),
                        flip_probability: 0.0,
                    },
                ]),
            }]),
        },
        round_duration: 1.0,
        detectors: Box::new([]),
        boundary_flows: Box::new([BoundaryFlow {
            start: PauliProduct {
                negative: false,
                terms: Box::new([(0, Pauli::Z)]),
            },
            end: PauliProduct {
                negative: false,
                terms: Box::new([(0, Pauli::Z)]),
            },
            parity: RecordParity {
                records: [0].into(),
                constant: false,
            },
            frontier_record: Some(0),
        }]),
        initializers: Box::new([
            FrontierInitializer {
                record: 0,
                parity: RecordParity::default(),
            },
            FrontierInitializer {
                record: 1,
                parity: RecordParity {
                    records: [].into(),
                    constant: true,
                },
            },
        ]),
    };
    let consumer = Instruction::Quantum(QuantumTask {
        alternatives: Box::new([QuantumAlternative {
            when: Box::new([]),
            stream: QuantumStream {
                moments: Box::new([Moment {
                    kind: None,
                    duration: 1.0,
                    operations: Box::new([QuantumOp::Measure {
                        observable: PauliProduct {
                            negative: false,
                            terms: Box::new([(0, Pauli::Z)]),
                        },
                        records: Box::new([2]),
                        flip_probability: 0.0,
                    }]),
                }]),
            },
            selected_records: Box::new([2]),
            excluded_records: Box::new([]),
            detectors: Box::new([RecordParity {
                records: [0, 2].into(),
                constant: false,
            }]),
            restarts: Box::new([]),
        }]),
    });
    let program = Program {
        qubit_count: 3,
        bit_count: 0,
        record_count: 3,
        tasks: vec![
            Task {
                label: "sibling".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([]),
                activation: None,
                output: None,
                qubits: Box::new([2]),
                duration: 3.0,
                instruction: empty_quantum,
            },
            Task {
                label: "wait".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([]),
                activation: None,
                output: None,
                qubits: Box::new([0, 1]),
                duration: 0.0,
                instruction: Instruction::WaitFor {
                    until: Box::new([0]),
                    memory: Box::new([memory]),
                },
            },
            Task {
                label: "consumer".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([1]),
                activation: None,
                output: None,
                qubits: Box::new([0]),
                duration: 1.0,
                instruction: consumer,
            },
        ],
        task_origins: Vec::new(),
        entry: Stream {
            tasks: Box::new([1, 0, 2]),
            value: None,
            bindings: Box::new([]),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        decoder_latency_rounds: 10,
    };

    let artifact = run(&program, RuntimeConfig::default()).unwrap().artifact;
    assert_eq!(
        artifact
            .detectors
            .iter()
            .map(|detector| detector.value)
            .collect::<Vec<_>>(),
        [true, false, false, false]
    );
}

#[test]
fn wait_runs_one_full_round_then_idles_to_a_known_deadline() {
    let sibling = Instruction::Quantum(QuantumTask {
        alternatives: Box::new([QuantumAlternative {
            stream: QuantumStream {
                moments: Box::new([Moment {
                    kind: None,
                    duration: 10.0,
                    operations: Box::new([]),
                }]),
            },
            ..QuantumAlternative::default()
        }]),
    });
    let program = Program {
        qubit_count: 0,
        bit_count: 0,
        record_count: 0,
        tasks: vec![
            Task {
                label: "sibling".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([]),
                activation: None,
                output: None,
                qubits: Box::new([]),
                duration: 10.0,
                instruction: sibling,
            },
            Task {
                label: "wait".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([]),
                activation: None,
                output: None,
                qubits: Box::new([]),
                duration: 0.0,
                instruction: Instruction::WaitFor {
                    until: Box::new([0]),
                    memory: Box::new([empty_memory_cycle(6.0, Some(6.0))]),
                },
            },
        ],
        task_origins: Vec::new(),
        entry: Stream {
            tasks: Box::new([1, 0]),
            value: None,
            bindings: Box::new([]),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        decoder_latency_rounds: 10,
    };

    let artifact = run(&program, RuntimeConfig::default()).unwrap().artifact;
    let rounds = artifact
        .events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::MemoryRound {
                task,
                round,
                memory_kind,
                wait_reason,
                start,
                end,
            } if *task == 1 => Some((*round, *memory_kind, *wait_reason, *start, *end)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rounds,
        [(
            0,
            MemoryKind::DynamicWait,
            Some(WaitReason::Synchronization),
            0.0,
            6.0,
        )]
    );
    let idles = artifact
        .events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::Idle {
                task: 1,
                wait_reason,
                start,
                end,
            } => Some((*wait_reason, *start, *end)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(idles, [(Some(WaitReason::Synchronization), 6.0, 10.0)]);
}

#[test]
fn wait_sees_a_decoder_deadline_through_a_classical_chain() {
    let task = |label: &str, dependencies, output, qubits, duration, instruction| Task {
        label: label.to_owned(),
        release: 0.0,
        source: None,
        dependencies,
        activation: None,
        output,
        qubits,
        duration,
        instruction,
    };
    let quantum = |moment: Moment| {
        Instruction::Quantum(QuantumTask {
            alternatives: Box::new([QuantumAlternative {
                stream: QuantumStream {
                    moments: Box::new([moment]),
                },
                ..QuantumAlternative::default()
            }]),
        })
    };
    let make_program = |decoder_round_duration, decoder_latency_rounds| {
        let mut origins = vec![TaskOrigin::default(); 7];
        origins[6] = TaskOrigin {
            function: TaskFunction::Memory,
            sites: Box::new([[4, 5]]),
            members: Box::new([[4, 5, 6]]),
        };
        Program {
            qubit_count: 2,
            bit_count: 4,
            record_count: 1,
            tasks: vec![
                task(
                    "measure",
                    Box::new([]),
                    None,
                    Box::new([0]),
                    6.0,
                    quantum(Moment {
                        kind: None,
                        duration: 6.0,
                        operations: Box::new([QuantumOp::Measure {
                            observable: PauliProduct {
                                negative: false,
                                terms: Box::new([(0, Pauli::Z)]),
                            },
                            records: Box::new([0]),
                            flip_probability: 0.0,
                        }]),
                    }),
                ),
                task(
                    "raw",
                    Box::new([0]),
                    Some(0),
                    Box::new([]),
                    0.0,
                    Instruction::Accumulate(RecordParity {
                        records: [0].into(),
                        constant: false,
                    }),
                ),
                task(
                    "observable",
                    Box::new([1]),
                    Some(1),
                    Box::new([]),
                    0.0,
                    Instruction::Observable {
                        index: 0,
                        bits: Box::new([0]),
                        bindings: Box::new([]),
                    },
                ),
                task(
                    "decode",
                    Box::new([2]),
                    Some(2),
                    Box::new([]),
                    0.0,
                    Instruction::Decode(DecodeRequest {
                        output: bloq_ir::ObservableOutput::Corrected,
                        observable: 2,
                        index: 0,
                        raw: 1,
                        round_duration: decoder_round_duration,
                        timing: DecodeTiming::Measurement,
                    }),
                ),
                task(
                    "transparent eval",
                    Box::new([3]),
                    Some(3),
                    Box::new([]),
                    0.0,
                    Instruction::Eval(BoolOp::Copy(2)),
                ),
                task(
                    "zero unit starter",
                    Box::new([]),
                    None,
                    Box::new([1]),
                    0.0,
                    quantum(Moment {
                        kind: None,
                        duration: 0.0,
                        operations: Box::new([]),
                    }),
                ),
                task(
                    "wait",
                    Box::new([5]),
                    None,
                    Box::new([1]),
                    0.0,
                    Instruction::WaitFor {
                        until: Box::new([4]),
                        memory: Box::new([empty_memory_cycle(6.0, Some(6.0))]),
                    },
                ),
            ],
            entry: Stream {
                tasks: Box::new([0, 1, 2, 3, 4, 5, 6]),
                value: None,
                bindings: Box::new([]),
            },
            task_origins: origins,
            decoder_latency_rounds,
            ..Program::default()
        }
    };
    let program = make_program(1.0, 2);

    let artifact = run(&program, RuntimeConfig::default()).unwrap().artifact;
    assert_eq!(artifact.metadata.decoder_latency_rounds, 2);
    let rounds = artifact
        .events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::MemoryRound {
                task: 6,
                round,
                wait_reason,
                start,
                end,
                ..
            } => Some((*round, *wait_reason, *start, *end)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(rounds, [(0, Some(WaitReason::CausalCut), 0.0, 6.0)]);
    assert!(artifact.events.iter().any(|event| matches!(
        event,
        ExecutionEvent::Idle {
            task: 6,
            wait_reason: Some(WaitReason::DecoderLatency),
            start: 6.0,
            end: 8.0,
        }
    )));
    assert!(
        artifact
            .events
            .iter()
            .any(|event| matches!(event, ExecutionEvent::JoinReleased { task: 6, time: 8.0 }))
    );
    assert!(artifact.events.iter().all(|event| !matches!(
        event,
        ExecutionEvent::MemoryRound {
            task: 6,
            start,
            ..
        } | ExecutionEvent::Idle {
            task: 6,
            start,
            ..
        } if *start >= 8.0
    )));
    assert!(artifact.events.iter().any(|event| matches!(
        event,
        ExecutionEvent::DecoderDeadline {
            task: 3,
            measurements_ready_at: Some(6.0),
            requested_at: 6.0,
            ready_at: 8.0,
            ..
        }
    )));
    let wait_metadata = &artifact.metadata.tasks[6];
    assert_eq!(wait_metadata.dependencies, [5]);
    assert_eq!(wait_metadata.wait_until, [4]);
    assert_eq!(wait_metadata.output, None);
    assert_eq!(
        wait_metadata.origin.as_ref().unwrap().function,
        TaskFunction::Memory
    );
    assert_eq!(
        wait_metadata.origin.as_ref().unwrap().sites.as_ref(),
        [[4, 5]]
    );

    let mut without_origins = program.clone();
    without_origins.task_origins.clear();
    let without_origins = run(&without_origins, RuntimeConfig::default())
        .unwrap()
        .artifact;
    assert_eq!(artifact.measurements, without_origins.measurements);
    assert_eq!(artifact.timing, without_origins.timing);
    assert_eq!(artifact.events, without_origins.events);
    assert_eq!(artifact.final_bits, without_origins.final_bits);

    let immediate = run(&make_program(1.0, 0), RuntimeConfig::default())
        .unwrap()
        .artifact;
    assert_eq!(immediate.metadata.decoder_latency_rounds, 0);
    let immediate_rounds = immediate
        .events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::MemoryRound {
                task: 6,
                round,
                wait_reason,
                start,
                end,
                ..
            } => Some((*round, *wait_reason, *start, *end)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        immediate_rounds,
        [(0, Some(WaitReason::CausalCut), 0.0, 6.0)]
    );
    assert!(immediate.events.iter().any(|event| matches!(
        event,
        ExecutionEvent::DecoderDeadline {
            task: 3,
            measurements_ready_at: Some(6.0),
            requested_at: 6.0,
            ready_at: 6.0,
            ..
        }
    )));
    assert!(
        immediate
            .events
            .iter()
            .any(|event| matches!(event, ExecutionEvent::JoinReleased { task: 6, time: 6.0 }))
    );
    assert!(immediate.events.iter().all(|event| !matches!(
        event,
        ExecutionEvent::MemoryRound {
            task: 6,
            start,
            ..
        } | ExecutionEvent::Idle {
            task: 6,
            start,
            ..
        } if *start >= 6.0
    )));

    assert!(matches!(
        run(&make_program(f64::MAX, 2), RuntimeConfig::default()),
        Err(RuntimeError::InvalidTiming)
    ));
}

#[test]
fn malformed_memory_cycle_is_a_typed_program_error() {
    let program = |cycle| Program {
        qubit_count: 0,
        bit_count: 0,
        record_count: 0,
        tasks: vec![Task {
            label: "wait".into(),
            release: 0.0,
            source: None,
            dependencies: Box::new([]),
            activation: None,
            output: None,
            qubits: Box::new([]),
            duration: 0.0,
            instruction: Instruction::WaitFor {
                until: Box::new([]),
                memory: Box::new([cycle]),
            },
        }],
        task_origins: Vec::new(),
        entry: Stream {
            tasks: Box::new([0]),
            value: None,
            bindings: Box::new([]),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        decoder_latency_rounds: 10,
    };

    for cycle in [
        empty_memory_cycle(0.0, None),
        empty_memory_cycle(6.0, Some(5.0)),
    ] {
        assert!(matches!(
            run(&program(cycle), RuntimeConfig::default()),
            Err(RuntimeError::InvalidProgram("invalid memory duration"))
        ));
    }
}

#[test]
fn boundary_observable_is_sampled_at_its_declared_cut() {
    let quantum = |dependencies, operation| Task {
        label: "quantum".into(),
        release: 0.0,
        source: None,
        dependencies,
        activation: None,
        output: None,
        qubits: Box::new([0]),
        duration: 0.0,
        instruction: Instruction::Quantum(QuantumTask {
            alternatives: Box::new([QuantumAlternative {
                when: Box::new([]),
                stream: QuantumStream {
                    moments: Box::new([Moment {
                        kind: None,
                        duration: 0.0,
                        operations: Box::new([operation]),
                    }]),
                },
                selected_records: Box::new([]),
                excluded_records: Box::new([]),
                detectors: Box::new([]),
                restarts: Box::new([]),
            }]),
        }),
    };
    let z = PauliProduct {
        negative: false,
        terms: Box::new([(0, Pauli::Z)]),
    };
    let program = Program {
        qubit_count: 1,
        tasks: vec![
            quantum(
                Box::new([]),
                QuantumOp::Reset {
                    basis: Pauli::Z,
                    qubit: 0,
                },
            ),
            Task {
                label: "bind cut".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([0]),
                activation: None,
                output: None,
                qubits: Box::new([]),
                duration: 0.0,
                instruction: Instruction::Bind(Box::new([BoundaryBinding {
                    input: false,
                    operator: z.clone(),
                }])),
            },
            Task {
                label: "observe cut".into(),
                release: 0.0,
                source: None,
                dependencies: Box::new([1]),
                activation: None,
                output: None,
                qubits: Box::new([]),
                duration: 0.0,
                instruction: Instruction::Observable {
                    index: 0,
                    bits: Box::new([]),
                    bindings: Box::new([1]),
                },
            },
            quantum(
                Box::new([2]),
                QuantumOp::Pauli {
                    basis: Pauli::X,
                    qubit: 0,
                },
            ),
        ],
        entry: Stream {
            tasks: Box::new([0, 1, 2, 3]),
            value: None,
            bindings: Box::new([]),
        },
        ..Program::default()
    };

    let result = run(&program, RuntimeConfig::default()).unwrap();
    assert_eq!(result.artifact.observables[0].value, Some(false));
    assert_eq!(result.simulator.peek_z(0).unwrap(), -1.0);
}

#[test]
fn independent_sources_keep_their_own_moment_lengths() {
    let mut long = CoordCircuit::new();
    long.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
    long.tick();
    long.do_gate(GateType::CX, [IVec2::ZERO, IVec2::X]).unwrap();
    long.tick();
    long.measure(PauliBasis::Z, [IVec2::ZERO]);
    long.tick();
    long.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
    long.tick();
    long.do_gate(GateType::CX, [IVec2::ZERO, IVec2::X]).unwrap();
    long.tick();
    long.measure(PauliBasis::Z, [IVec2::ZERO]);

    let mut short = CoordCircuit::new();
    short.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
    short.tick();
    short.measure(PauliBasis::Z, [IVec2::ZERO]);

    let mut bloq = Bloq::new();
    let long = bloq.add_template(BloqTemplate::new(long));
    let short = bloq.add_template(BloqTemplate::new(short));
    bloq.add_node(quantum_node(long, 0));
    let mut short_node = quantum_node(short, 1);
    short_node.expect_quantum_mut().instances[0].offset = IVec2::new(4, 0);
    bloq.add_node(short_node);

    let noise = NoiseModel {
        p_idle: 0.125,
        ..NoiseModel::uniform_depolarizing(0.0)
    };
    let program = lower(
        &bloq,
        &LoweringConfig {
            noise: Some(&noise),
            ..LoweringConfig::default()
        },
    )
    .unwrap();
    let streams = program
        .tasks
        .iter()
        .filter_map(|task| match &task.instruction {
            Instruction::Quantum(quantum) => Some(&quantum.alternatives[0].stream),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(streams.len(), 2);
    assert!(
        program
            .tasks
            .iter()
            .all(|task| { task.source == Some(SourceRole::Clifford) && task.release == 0.0 })
    );
    let long = streams
        .iter()
        .copied()
        .find(|stream| {
            stream.moments.iter().any(|moment| {
                moment
                    .operations
                    .iter()
                    .any(|operation| matches!(operation, QuantumOp::Gate2 { .. }))
            })
        })
        .unwrap();
    let short = streams
        .iter()
        .copied()
        .find(|stream| *stream != long)
        .unwrap();
    let measurement = |stream: &QuantumStream| {
        stream
            .moments
            .iter()
            .rposition(|moment| moment.kind == Some(MomentKind::Measurement))
            .unwrap()
    };
    assert!(measurement(short) < measurement(long));
    let reset = short
        .moments
        .iter()
        .position(|moment| moment.kind == Some(MomentKind::Reset))
        .unwrap();
    assert_eq!(
        reset, 0,
        "an independent source starts at its release epoch"
    );
}

#[test]
fn standalone_noise_keeps_its_zero_time_stream_and_effect() {
    let mut error = CoordCircuit::new();
    let entry = error.entry_body();
    error
        .body_mut(entry)
        .unwrap()
        .ops_mut()
        .push(Op::PauliError {
            probability: 1.0,
            pauli: PauliBasis::X,
            qubits: vec![IVec2::ZERO],
        });
    let mut measure = CoordCircuit::new();
    measure.measure(PauliBasis::Z, [IVec2::ZERO]);
    let mut independent = CoordCircuit::new();
    independent.do_gate(GateType::H, [IVec2::ZERO]).unwrap();

    let mut bloq = Bloq::new();
    let error = bloq.add_template(BloqTemplate::new(error));
    let measure = bloq.add_template(BloqTemplate::new(measure));
    let independent = bloq.add_template(BloqTemplate::new(independent));
    let error = bloq.add_node(quantum_node(error, 0));
    let measure = bloq.add_node(quantum_node(measure, 1));
    let mut independent_node = quantum_node(independent, 2);
    independent_node.expect_quantum_mut().instances[0].offset = IVec2::new(4, 0);
    bloq.add_node(independent_node);
    bloq.add_edge(error, measure, BloqEdge::Order);

    let program = lower(&bloq, &LoweringConfig::default()).unwrap();
    let error_task = program
        .tasks
        .iter()
        .find(|task| match &task.instruction {
            Instruction::Quantum(quantum) => {
                quantum.alternatives[0].stream.moments.iter().any(|moment| {
                    moment
                        .operations
                        .iter()
                        .any(|operation| matches!(operation, QuantumOp::PauliError { .. }))
                })
            }
            _ => false,
        })
        .unwrap();
    let Instruction::Quantum(quantum) = &error_task.instruction else {
        unreachable!()
    };
    assert_eq!(error_task.duration, 0.0);
    assert_eq!(quantum.alternatives[0].stream.moments.len(), 1);
    let moment = &quantum.alternatives[0].stream.moments[0];
    assert_eq!((moment.kind, moment.duration), (None, 0.0));
    assert!(matches!(
        moment.operations.as_ref(),
        [QuantumOp::PauliError { probability, basis: Pauli::X, .. }] if *probability == 1.0
    ));

    let artifact = run(&program, RuntimeConfig::default()).unwrap().artifact;
    assert_eq!(artifact.measurements.len(), 1);
    assert!(
        artifact.measurements[0].value,
        "X error flips the later Z read"
    );
}

#[test]
fn compiled_clifford_t_and_thth_run_without_noiseless_detector_failures() {
    for item in [GalleryItem::CNOT, GalleryItem::T, GalleryItem::THTH] {
        let bloq = CompileContext::new(CompileConfig::new(3))
            .compile(&item.build())
            .unwrap()
            .bloq;

        let legacy = run_bloq(&bloq, 1, 7).unwrap();
        let dynamic = run_dynamic(&bloq);
        assert_eq!(legacy.discarded, 0, "{item:?}");
        assert!(
            legacy
                .detectors
                .iter()
                .all(|detector| detector.value != Some(true)),
            "{item:?}"
        );
        assert!(!dynamic.discarded, "{item:?}");
        assert!(
            dynamic
                .detectors
                .iter()
                .filter(|detector| detector.committed)
                .all(|detector| !detector.value),
            "{item:?}"
        );
        if !matches!(item, GalleryItem::CNOT) {
            assert!(!dynamic.retries.is_empty(), "{item:?}");
            let lowered = lower(&bloq, &LoweringConfig::default()).unwrap();
            assert!(lowered.instructions().any(|instruction| matches!(
                instruction,
                Instruction::Rus { decoder_hold, .. } if !decoder_hold.is_empty()
            )));
        }
    }
}
