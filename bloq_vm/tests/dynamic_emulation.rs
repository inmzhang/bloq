use std::collections::BTreeSet;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::GalleryItem;
use bloq_ir::circuit::NoiseModel;
use bloq_vm::DEFAULT_DECODER_LATENCY_ROUNDS;
use bloq_vm::decoder::MockDecoderConfig;
use bloq_vm::instruction::{Instruction, Program, QuantumOp, SourceRole, Stream};
use bloq_vm::runtime::{
    ExecutionArtifact, ExecutionEvent, LogicalInputState, MemoryKind, RuntimeConfig,
};
use bloq_vm::{LoweringConfig, Simulator, SourceTiming, lower};

const ROUND_DURATION: f64 = 6.0;

fn lower_thth(distance: u32, noise: Option<&NoiseModel>) -> Program {
    lower_thth_with_timing(distance, noise, SourceTiming::default(), 1.0)
}

fn lower_thth_with_timing(
    distance: u32,
    noise: Option<&NoiseModel>,
    source_timing: SourceTiming,
    gate_duration: f64,
) -> Program {
    // Full cultivation is the subject: authored resets, retries and factory GAP timing.
    let bloq = CompileContext::new(CompileConfig::new(distance).with_prepare_t_with_mpps(false))
        .compile(&GalleryItem::THTH.build())
        .expect("THTH compiles")
        .bloq;
    lower(
        &bloq,
        &LoweringConfig {
            noise,
            source_timing,
            gate_duration,
            ..LoweringConfig::default()
        },
    )
    .expect("full physical THTH lowers")
}

fn runtime_config(seed: u64, acceptance_script: Vec<bool>) -> RuntimeConfig {
    RuntimeConfig {
        seed,
        input_state: LogicalInputState::Plus,
        decoder: MockDecoderConfig {
            acceptance_probability: 1.0,
            accepted_accuracy: 1.0,
            rejected_accuracy: 0.0,
            acceptance_script,
            ..MockDecoderConfig::default()
        },
        ..RuntimeConfig::default()
    }
}

fn event_times(event: &ExecutionEvent) -> Vec<f64> {
    match event {
        ExecutionEvent::SourceReleased { time, .. }
        | ExecutionEvent::InputArrival { time, .. }
        | ExecutionEvent::MomentCompleted { time, .. }
        | ExecutionEvent::EarlyRestart { time, .. }
        | ExecutionEvent::JoinReleased { time, .. }
        | ExecutionEvent::QuantumAlternative { time, .. }
        | ExecutionEvent::ConditionalCorrection { time, .. }
        | ExecutionEvent::RusAttemptStarted { time, .. }
        | ExecutionEvent::RusAttemptRejected { time, .. }
        | ExecutionEvent::RusAttemptAccepted { time, .. }
        | ExecutionEvent::FactoryReady { time, .. }
        | ExecutionEvent::Discarded { time, .. } => vec![*time],
        ExecutionEvent::MomentIssued { start, end, .. }
        | ExecutionEvent::RetryPrepare { start, end, .. }
        | ExecutionEvent::MemoryRound { start, end, .. }
        | ExecutionEvent::Idle { start, end, .. } => vec![*start, *end],
        ExecutionEvent::DecoderDeadline {
            measurements_ready_at,
            requested_at,
            ready_at,
            ..
        } => measurements_ready_at
            .iter()
            .copied()
            .chain([*requested_at, *ready_at])
            .collect(),
    }
}

fn assert_scaled_time(base: f64, scaled: f64, scale: f64) {
    assert!(
        (scaled - base * scale).abs() < 1e-8,
        "time {scaled} != {base} * {scale}"
    );
}

fn assert_source_events(program: &Program, artifact: &ExecutionArtifact) {
    for (task, source) in program.tasks.iter().enumerate() {
        let Some(role) = source.source else {
            continue;
        };
        assert!(artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::SourceReleased {
                task: released,
                role: released_role,
                time,
            } if *released == task as u32
                && released_role == &format!("{role:?}")
                && (*time - source.release).abs() < 1e-9
        )));
    }
    let input = program.inputs.first().expect("one encoded input");
    let release = program.tasks[input.task as usize].release;
    assert!(artifact.events.iter().any(|event| matches!(
        event,
        ExecutionEvent::InputArrival {
            task,
            port,
            state: LogicalInputState::Plus,
            time,
        } if *task == input.task && *port == input.port && *time == release
    )));
}

fn assert_authored_retry_replay(program: &Program, artifact: &ExecutionArtifact) {
    for retry in artifact.retries.iter().filter(|retry| retry.attempt > 0) {
        let previous = artifact
            .retries
            .iter()
            .find(|previous| previous.task == retry.task && previous.attempt + 1 == retry.attempt)
            .expect("retried attempt has predecessor");
        assert_eq!(retry.start, previous.end);
        assert!(!artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::RetryPrepare { task, attempt, .. }
                if *task == retry.task && *attempt == retry.attempt
        )));
        assert!(artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::RusAttemptStarted { task, attempt, time, .. }
                if *task == retry.task
                    && *attempt == retry.attempt
                    && *time == previous.end
        )));

        let Instruction::Rus { body, .. } = &program.tasks[retry.task as usize].instruction else {
            panic!("retry record points at a non-RUS task");
        };
        let mut body_tasks = BTreeSet::new();
        stream_tasks(program, body, &mut body_tasks);
        let reset_moments = body_tasks
            .into_iter()
            .flat_map(|task| match &program.tasks[task as usize].instruction {
                Instruction::Quantum(quantum) => quantum
                    .alternatives
                    .iter()
                    .flat_map(|alternative| {
                        alternative.stream.moments.iter().enumerate().filter_map(
                            move |(moment, value)| {
                                value
                                    .operations
                                    .iter()
                                    .any(|operation| matches!(operation, QuantumOp::Reset { .. }))
                                    .then_some((task, moment as u32))
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<BTreeSet<_>>();
        assert!(
            !reset_moments.is_empty(),
            "factory body has authored resets"
        );
        assert!(artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::MomentIssued {
                task,
                attempt: Some(epoch),
                moment,
                ..
            } if *epoch == retry.epoch && reset_moments.contains(&(*task, *moment))
        )));
    }
}

fn stream_tasks(program: &Program, stream: &Stream, tasks: &mut BTreeSet<u32>) {
    for &task in &stream.tasks {
        tasks.insert(task);
        if let Instruction::Rus { body, .. } = &program.tasks[task as usize].instruction {
            stream_tasks(program, body, tasks);
        }
    }
}

fn assert_factory_gaps(program: &Program, artifact: &ExecutionArtifact) {
    for retry in &artifact.retries {
        let Instruction::Rus {
            body,
            cultivation_exits,
            ..
        } = &program.tasks[retry.task as usize].instruction
        else {
            panic!("factory retry does not point at a RUS task");
        };
        let rounds = artifact
            .events
            .iter()
            .filter_map(|event| match event {
                ExecutionEvent::MemoryRound {
                    task,
                    round,
                    memory_kind: MemoryKind::FactoryGap,
                    start,
                    end,
                    ..
                } if *task == retry.task && *start >= retry.start && *end <= retry.end + 1e-9 => {
                    Some((*round, *start, *end))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if rounds.is_empty() {
            assert!(!retry.accepted, "accepted attempt skipped protected GAP");
            continue;
        }
        assert_eq!(rounds.len(), program.decoder_latency_rounds as usize);
        for (position, (round, start, end)) in rounds.iter().copied().enumerate() {
            assert_eq!(round, position as u32);
            assert_eq!(end - start, ROUND_DURATION);
            if let Some((_, _, previous_end)) = position.checked_sub(1).map(|i| rounds[i]) {
                assert_eq!(start, previous_end);
            }
        }
        let cultivation_end = cultivation_exits
            .iter()
            .map(|exit| {
                artifact
                    .timing
                    .iter()
                    .find(|timing| timing.task == *exit && timing.attempt == Some(retry.epoch))
                    .expect("cultivation exit completed in this attempt")
                    .end
            })
            .fold(0.0, f64::max);
        let first_gap = rounds.first().expect("configured GAP rounds").1;
        let ready_at = rounds.last().expect("configured GAP rounds").2;
        assert_eq!(first_gap, cultivation_end);
        assert_eq!(
            ready_at,
            cultivation_end + ROUND_DURATION * f64::from(program.decoder_latency_rounds)
        );
        assert_eq!(ready_at, retry.end);

        let mut body_tasks = BTreeSet::new();
        stream_tasks(program, body, &mut body_tasks);
        let decodes = body_tasks
            .into_iter()
            .filter(|task| {
                matches!(
                    program.tasks[*task as usize].instruction,
                    Instruction::Decode(_)
                )
            })
            .collect::<Vec<_>>();
        assert!(!decodes.is_empty());
        assert!(
            decodes
                .iter()
                .all(|decode| artifact.events.iter().any(|event| matches!(
                    event,
                    ExecutionEvent::DecoderDeadline {
                        task,
                        factory: true,
                        ready_at: deadline,
                        ..
                    } if task == decode && *deadline == ready_at
                )))
        );
    }
}

fn assert_early_restarts_skip_gap(artifact: &ExecutionArtifact) {
    let early = artifact
        .events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::EarlyRestart {
                task,
                attempt,
                time,
                ..
            } => Some((*task, *attempt, *time)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        !early.is_empty(),
        "the chosen seed must exercise an early restart"
    );
    for (task, attempt, detected_at) in early {
        let retry = artifact
            .retries
            .iter()
            .find(|retry| retry.task == task && retry.attempt == attempt)
            .expect("early restart is retained in attempt history");
        assert!(!retry.accepted);
        assert!(detected_at <= retry.end);
        assert!(!artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::MemoryRound {
                task: gap_task,
                memory_kind: MemoryKind::FactoryGap,
                start,
                ..
            } if *gap_task == task && *start >= retry.start && *start < retry.end
        )));
        let epoch = (u64::from(task) << 32) | (u64::from(attempt) + 1);
        assert!(!artifact.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::MomentIssued {
                attempt: Some(issued_epoch),
                start,
                ..
            } if *issued_epoch == epoch && *start > detected_at
        )));
    }
}

fn assert_terminal_output_is_held(program: &Program, artifact: &ExecutionArtifact) {
    for output in &program.outputs {
        let frame_ready = [output.frame_x, output.frame_z]
            .into_iter()
            .filter_map(|bit| {
                let task = program
                    .tasks
                    .iter()
                    .position(|task| task.output == Some(bit))? as u32;
                artifact
                    .timing
                    .iter()
                    .find(|timing| timing.task == task && timing.attempt.is_none())
                    .map(|timing| timing.end)
            })
            .fold(0.0, f64::max);
        let output_qubits = output
            .x
            .terms
            .iter()
            .chain(&output.z.terms)
            .map(|(qubit, _)| *qubit)
            .collect::<BTreeSet<_>>();
        assert!(
            program.tasks.iter().enumerate().any(|(task, instruction)| {
                if !matches!(&instruction.instruction, Instruction::WaitFor { .. })
                    || !instruction
                        .qubits
                        .iter()
                        .any(|qubit| output_qubits.contains(qubit))
                {
                    return false;
                }
                let memory_end = artifact
                    .events
                    .iter()
                    .filter_map(|event| match event {
                        ExecutionEvent::MemoryRound {
                            task: memory_task,
                            memory_kind: MemoryKind::DynamicWait,
                            end,
                            ..
                        } if *memory_task == task as u32 => Some(*end),
                        _ => None,
                    })
                    .max_by(f64::total_cmp);
                let Some(memory_end) = memory_end else {
                    return false;
                };
                let protected_until_frame = memory_end == frame_ready
                    || artifact.events.iter().any(|event| {
                        matches!(
                            event,
                            ExecutionEvent::Idle {
                                task: idle_task,
                                start,
                                end,
                                ..
                            } if *idle_task == task as u32
                                && *start == memory_end
                                && *end == frame_ready
                        )
                    });
                protected_until_frame
                    && artifact.events.iter().any(|event| {
                        matches!(
                            event,
                            ExecutionEvent::JoinReleased {
                                task: joined_task,
                                time,
                            } if *joined_task == task as u32 && *time == frame_ready
                        )
                    })
            }),
            "terminal output did not run QEC through frame readiness at {frame_ready}"
        );
    }
}

#[test]
fn thth_sources_follow_the_priced_release_schedule() {
    let timing = SourceTiming {
        factory: 7.0,
        input: 11.0,
        clifford: 13.0,
    };
    let program = lower_thth_with_timing(3, None, timing, 1.0);
    let sources = program.sources();
    for (role, count, release) in [
        (SourceRole::Factory, 2, timing.factory),
        (SourceRole::LogicalInput, 1, timing.input),
        (SourceRole::PreparedY, 2, timing.clifford),
        (SourceRole::Clifford, 1, timing.clifford),
    ] {
        let matching = || sources.iter().filter(|source| source.role == role);
        assert_eq!(matching().count(), count);
        if role != SourceRole::PreparedY {
            assert!(matching().all(|source| source.release == release));
        }
    }
    let clifford_duration = sources
        .iter()
        .find(|source| source.role == SourceRole::Clifford)
        .unwrap()
        .duration;
    assert!(
        sources
            .iter()
            .filter(|source| source.role == SourceRole::PreparedY)
            .all(|source| {
                source.release == timing.clifford + clifford_duration - source.duration
                    && source.release + source.duration == timing.clifford + clifford_duration
            })
    );
    assert_eq!(program.inputs.len(), 1);
    assert_eq!(
        program.tasks[program.inputs[0].task as usize].source,
        Some(SourceRole::LogicalInput)
    );
}

#[test]
fn ideal_thth_schedule_scales_with_time_unit() {
    let scale = 0.1;
    let timing = SourceTiming {
        factory: 0.0,
        input: 17.0,
        clifford: 19.0,
    };
    let scaled_timing = SourceTiming {
        factory: timing.factory * scale,
        input: timing.input * scale,
        clifford: timing.clifford * scale,
    };
    let base_program = lower_thth_with_timing(3, None, timing, 1.0);
    let scaled_program = lower_thth_with_timing(3, None, scaled_timing, scale);
    let base = base_program.run(runtime_config(0x11, vec![false])).unwrap();
    let scaled = scaled_program
        .run(runtime_config(0x11, vec![false]))
        .unwrap();

    assert_eq!(base.artifact.discarded, scaled.artifact.discarded);
    assert_eq!(base.artifact.final_bits, scaled.artifact.final_bits);
    assert_eq!(base.artifact.events.len(), scaled.artifact.events.len());
    assert_eq!(
        base.artifact.measurements.len(),
        scaled.artifact.measurements.len()
    );
    assert_eq!(
        base.artifact.detectors.len(),
        scaled.artifact.detectors.len()
    );
    assert_eq!(
        base.artifact.observables.len(),
        scaled.artifact.observables.len()
    );
    assert_eq!(base.artifact.retries.len(), scaled.artifact.retries.len());
    assert_scaled_time(
        base.artifact.finished_at,
        scaled.artifact.finished_at,
        scale,
    );
    for (position, (base_event, scaled_event)) in base
        .artifact
        .events
        .iter()
        .zip(&scaled.artifact.events)
        .enumerate()
    {
        assert_eq!(
            std::mem::discriminant(base_event),
            std::mem::discriminant(scaled_event),
            "event {position}: {base_event:?} != {scaled_event:?}"
        );
        let base_times = event_times(base_event);
        let scaled_times = event_times(scaled_event);
        assert_eq!(base_times.len(), scaled_times.len());
        for (base_time, scaled_time) in base_times.into_iter().zip(scaled_times) {
            assert_scaled_time(base_time, scaled_time, scale);
        }
        match (base_event, scaled_event) {
            (
                ExecutionEvent::MomentIssued {
                    task,
                    attempt,
                    moment,
                    ..
                },
                ExecutionEvent::MomentIssued {
                    task: scaled_task,
                    attempt: scaled_attempt,
                    moment: scaled_moment,
                    ..
                },
            ) => assert_eq!(
                (*task, *attempt, *moment),
                (*scaled_task, *scaled_attempt, *scaled_moment)
            ),
            (
                ExecutionEvent::MemoryRound {
                    task,
                    round,
                    memory_kind,
                    wait_reason,
                    ..
                },
                ExecutionEvent::MemoryRound {
                    task: scaled_task,
                    round: scaled_round,
                    memory_kind: scaled_kind,
                    wait_reason: scaled_reason,
                    ..
                },
            ) => assert_eq!(
                (*task, *round, *memory_kind, *wait_reason),
                (*scaled_task, *scaled_round, *scaled_kind, *scaled_reason)
            ),
            (
                ExecutionEvent::Idle {
                    task, wait_reason, ..
                },
                ExecutionEvent::Idle {
                    task: scaled_task,
                    wait_reason: scaled_reason,
                    ..
                },
            ) => assert_eq!((*task, *wait_reason), (*scaled_task, *scaled_reason)),
            _ => {}
        }
    }
    for (base_task, scaled_task) in base.artifact.timing.iter().zip(&scaled.artifact.timing) {
        assert_eq!(base_task.task, scaled_task.task);
        assert_eq!(base_task.attempt, scaled_task.attempt);
        assert_scaled_time(base_task.start, scaled_task.start, scale);
        assert_scaled_time(base_task.end, scaled_task.end, scale);
    }
    for (base_measurement, scaled_measurement) in base
        .artifact
        .measurements
        .iter()
        .zip(&scaled.artifact.measurements)
    {
        assert_eq!(base_measurement.record, scaled_measurement.record);
        assert_eq!(base_measurement.value, scaled_measurement.value);
        assert_eq!(base_measurement.attempt, scaled_measurement.attempt);
        assert_scaled_time(base_measurement.time, scaled_measurement.time, scale);
    }
    for (base_output, scaled_output) in base_program.outputs.iter().zip(&scaled_program.outputs) {
        assert_eq!(
            base.logical_bloch(base_output).unwrap(),
            scaled.logical_bloch(scaled_output).unwrap()
        );
    }
}

#[test]
fn ideal_thth_runs_physical_factories_gap_and_live_input() {
    let program = lower_thth(3, None);
    let result = program
        .run(runtime_config(0x11, vec![false]))
        .expect("THTH dynamic run");
    let artifact = &result.artifact;

    assert_source_events(&program, artifact);
    assert_authored_retry_replay(&program, artifact);
    assert_factory_gaps(&program, artifact);
    assert_terminal_output_is_held(&program, artifact);
    assert!(
        artifact.retries.iter().any(|retry| {
            !retry.accepted
                && artifact
                    .events
                    .iter()
                    .filter(|event| {
                        matches!(
                            event,
                            ExecutionEvent::MemoryRound {
                                task,
                                memory_kind: MemoryKind::FactoryGap,
                                start,
                                end,
                                ..
                            } if *task == retry.task && *start >= retry.start && *end <= retry.end
                        )
                    })
                    .count()
                    == DEFAULT_DECODER_LATENCY_ROUNDS as usize
        }),
        "fixture must reject one completed GAP and replay its authored reset path"
    );
    let output = program.outputs.first().expect("one THTH output");
    let actual = result
        .logical_bloch(output)
        .expect("corrected output Bloch vector");
    let mut ideal = Simulator::with_seed(1, 0);
    ideal.reset_x(0).expect("prepare |+>");
    ideal.t(0).expect("first T");
    ideal.h(0);
    ideal.t(0).expect("second T");
    ideal.h(0);
    let expected = (
        ideal.peek_x(0).unwrap(),
        ideal.peek_y(0).unwrap(),
        ideal.peek_z(0).unwrap(),
    );
    assert!(
        [actual.0, actual.1, actual.2]
            .into_iter()
            .zip([expected.0, expected.1, expected.2])
            .all(|(actual, expected)| (actual - expected).abs() < 1e-9),
        "corrected THTH state {actual:?}, expected {expected:?}",
    );
}

#[test]
fn factory_gap_uses_configured_nondefault_and_zero_latency() {
    let base = lower_thth(3, None);
    let mut reference_bloch: Option<(f64, f64, f64)> = None;
    for latency in [DEFAULT_DECODER_LATENCY_ROUNDS, 3, 0] {
        let mut program = base.clone();
        program.decoder_latency_rounds = latency;
        let result = program
            .run(runtime_config(0x11, Vec::new()))
            .expect("THTH dynamic run");
        let output = program.outputs.first().expect("one THTH output");
        let bloch = result
            .logical_bloch(output)
            .expect("corrected output Bloch vector");
        if let Some(reference) = reference_bloch {
            assert!(
                [bloch.0, bloch.1, bloch.2]
                    .into_iter()
                    .zip([reference.0, reference.1, reference.2])
                    .all(|(actual, expected)| (actual - expected).abs() < 1e-9),
                "latency {latency} changed corrected THTH state from {reference:?} to {bloch:?}",
            );
        } else {
            reference_bloch = Some(bloch);
        }
        let artifact = &result.artifact;

        assert_eq!(artifact.metadata.decoder_latency_rounds, latency);
        assert!(!artifact.retries.is_empty());
        assert!(artifact.retries.iter().all(|retry| retry.accepted));
        for retry in &artifact.retries {
            let Instruction::Rus {
                body,
                cultivation_exits,
                ..
            } = &program.tasks[retry.task as usize].instruction
            else {
                panic!("factory retry does not point at a RUS task");
            };
            let cultivation_end = cultivation_exits
                .iter()
                .map(|exit| {
                    artifact
                        .timing
                        .iter()
                        .find(|timing| timing.task == *exit && timing.attempt == Some(retry.epoch))
                        .expect("cultivation exit completed in this attempt")
                        .end
                })
                .fold(0.0, f64::max);
            let expected_ready = cultivation_end + ROUND_DURATION * f64::from(latency);
            let gaps = artifact
                .events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        ExecutionEvent::MemoryRound {
                            task,
                            memory_kind: MemoryKind::FactoryGap,
                            start,
                            end,
                            ..
                        } if *task == retry.task
                            && *start >= retry.start
                            && *end <= retry.end + 1e-9
                    )
                })
                .count();
            assert_eq!(gaps, latency as usize);
            assert_eq!(retry.end, expected_ready);

            let mut body_tasks = BTreeSet::new();
            stream_tasks(&program, body, &mut body_tasks);
            let factory_decodes = body_tasks
                .into_iter()
                .filter(|task| {
                    matches!(
                        program.tasks[*task as usize].instruction,
                        Instruction::Decode(_)
                    )
                })
                .collect::<Vec<_>>();
            assert!(!factory_decodes.is_empty());
            assert!(factory_decodes.iter().all(|decode| {
                artifact.events.iter().any(|event| {
                    matches!(
                        event,
                        ExecutionEvent::DecoderDeadline {
                            task,
                            factory: true,
                            ready_at,
                            ..
                        } if task == decode && *ready_at == expected_ready
                    )
                })
            }));
        }
    }
}

#[test]
fn factory_decode_pair_can_query_after_gap_readiness_without_rewinding() {
    let mut base = lower_thth(3, None);
    base.decoder_latency_rounds = 2;
    let reference = base.run(runtime_config(0x11, Vec::new())).unwrap();
    let (observable, ready_at) = reference
        .artifact
        .events
        .iter()
        .find_map(|event| match event {
            ExecutionEvent::DecoderDeadline {
                factory: true,
                observable,
                ready_at,
                ..
            } => Some((*observable, *ready_at)),
            _ => None,
        })
        .unwrap();
    let release = ready_at + ROUND_DURATION / 2.0;
    let expected = reference.logical_bloch(&base.outputs[0]).unwrap();
    let rounds = |artifact: &ExecutionArtifact| {
        artifact
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    ExecutionEvent::MemoryRound {
                        memory_kind: MemoryKind::FactoryGap,
                        ..
                    }
                )
            })
            .count()
    };
    for delayed_flip in [false, true] {
        let mut program = base.clone();
        let delayed = program
            .tasks
            .iter()
            .position(|task| match &task.instruction {
                Instruction::Decode(request) => {
                    request.observable == observable
                        && (request.output == bloq_ir::ObservableOutput::Flip) == delayed_flip
                }
                _ => false,
            })
            .unwrap();
        program.tasks[delayed].release = release;
        let result = program.run(runtime_config(0x11, Vec::new())).unwrap();
        assert!(!result.artifact.discarded);
        let query = result
            .artifact
            .timing
            .iter()
            .find(|task| task.task as usize == delayed)
            .unwrap();
        assert_eq!((query.start, query.end), (release, release));
        let decisions = result
            .artifact
            .decoder_decisions
            .iter()
            .filter(|record| record.decision.key.observable_task == observable)
            .collect::<Vec<_>>();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].decision.ready_at, ready_at);
        assert_eq!(rounds(&result.artifact), rounds(&reference.artifact));
        let actual = result.logical_bloch(&program.outputs[0]).unwrap();
        assert!(
            [actual.0, actual.1, actual.2]
                .into_iter()
                .zip([expected.0, expected.1, expected.2])
                .all(|(actual, expected)| (actual - expected).abs() < 1e-9)
        );
    }
}

#[test]
fn noisy_thth_restarts_as_soon_as_a_physical_check_fires() {
    let noise = NoiseModel::uniform_depolarizing(5e-4);
    let program = lower_thth(3, Some(&noise));
    let mut config = runtime_config(5, Vec::new());
    config.idle_error_rate = 5e-4;
    let result = program.run(config).expect("seeded noisy THTH run");

    assert!(!result.artifact.discarded);
    assert_early_restarts_skip_gap(&result.artifact);
    assert_authored_retry_replay(&program, &result.artifact);
    assert_factory_gaps(&program, &result.artifact);
}
