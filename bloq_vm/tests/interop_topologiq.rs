//! Pinned Topologiq layouts and boundary-import snapshots checked against independent maps.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use bloq_compile::{CompileConfig, CompileContext, CompileError};
use bloq_graph::{
    Action, Expr,
    verify::{BoundaryOrder, LogicalVerifier},
};
use bloq_vm::verify::{logical_signature, signatures_match};
use bloq_vm::{
    EnginePauli, EnginePauliString, ExecError, Simulator, run_bloq_with_io_and_captured_outputs,
};
use serde::Deserialize;

mod common;
use common::choi::exact::{BellOutcomes, ExactChoi};
use common::choi::stabilizer::physical_qubit;
use common::choi::widen;
use common::interop::{Fixture, apply_source_qasm, assert_selectors_toggle, record_selectors};

const UPSTREAM_REVISION: &str = "d010bd3e63686c93c12c922a84a944a3386ac785";
const SHOTS: usize = 8;
const SEEDS: [u64; 3] = [0x11, 0x22, 0x33];

#[derive(Deserialize)]
struct Metadata {
    upstream: Upstream,
    expected_state: Option<Vec<[f64; 2]>>,
}

#[derive(Deserialize)]
struct Upstream {
    repository: String,
    revision: String,
    patch: String,
    patch_sha256: String,
}

fn fixture(name: &str) -> Fixture<Metadata> {
    let fixture: Fixture<Metadata> = common::interop::fixture(&format!("topologiq/{name}"));
    let upstream = &fixture.metadata.upstream;
    assert_eq!(upstream.revision, UPSTREAM_REVISION);
    assert_eq!(upstream.repository, "https://github.com/tqec/topologiq");
    assert_eq!(upstream.patch, "bloq_py/examples/topologiq.patch");
    assert_eq!(upstream.patch_sha256.len(), 64);
    fixture
}

#[rstest::rstest]
#[case("t")]
#[case("tdg")]
#[case("thth")]
#[case("entangled_t")]
#[case("qft2")]
#[case("qft2_x")]
#[case("frames/z_neck_t")]
#[case("frames/z_neck_tdg")]
#[case("frames/split_x_neck_t")]
#[case("frames/split_x_neck_tdg")]
#[case("frames/padded_x_neck_t")]
#[case("frames/padded_x_neck_tdg")]
#[case("frames/direct_twins_input_t")]
#[case("frames/direct_twins_input_tdg")]
#[case("frames/direct_tree_input_t")]
#[case("frames/direct_tree_input_tdg")]
#[case("frames/direct_split_input_t")]
#[case("frames/direct_split_input_tdg")]
#[case("frames/direct_h_feedback_t")]
#[case("frames/direct_h_feedback_tdg")]
#[case("frames/direct_cap_padding_t")]
#[case("frames/direct_cap_padding_tdg")]
#[case("frames/direct_upstream_h_t")]
#[case("frames/direct_upstream_h_tdg")]
fn real_topologiq_layout_has_the_source_channel_at_d3(#[case] name: &str) {
    let fixture = fixture(name);
    let expected = fixture.verify_source();
    if name.starts_with("frames/") {
        fixture.verify_named_outcomes(&expected);
    }

    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("imported graph compiles at distance three")
        .bloq;
    bloq.validate().expect("compiled IR validates");
    let oracle = ExactChoi::new(&fixture.inputs, &fixture.outputs, |sim| {
        apply_source_qasm(sim, &fixture.qasm);
        Ok(())
    });
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut frames = BTreeSet::new();
    for seed in SEEDS {
        let report = oracle.run(&bloq, SHOTS, seed, |_, context| {
            record_selectors(&mut selectors, context.named_branch_selectors);
            frames.insert(
                fixture
                    .outputs
                    .iter()
                    .map(|port| {
                        let frame = context
                            .frames
                            .iter()
                            .find(|frame| frame.port == *port)
                            .expect("every declared output has a frame");
                        (
                            frame.x.expect("evaluable X frame"),
                            frame.z.expect("evaluable Z frame"),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(())
        });
        assert!(
            report.all_detectors_constant(),
            "{name}: deterministic detectors"
        );
        assert!(
            report.max_rank > 1,
            "{name}: non-Clifford physical execution"
        );
    }
    assert_selectors_toggle(&selectors, fixture.t_gadgets, name);
    assert!(
        frames.len() > 1,
        "{name}: terminal frames must vary across shots"
    );
}

fn undo_qpe(sim: &mut Simulator, [a, b, c, d]: [usize; 4]) -> Result<(), ExecError> {
    fn controlled_s(
        sim: &mut Simulator,
        control: usize,
        target: usize,
        dagger: bool,
    ) -> Result<(), ExecError> {
        for qubit in [control, target] {
            if dagger {
                sim.t_dag(qubit)?;
            } else {
                sim.t(qubit)?;
            }
        }
        sim.cx(control, target)?;
        if dagger {
            sim.t(target)?;
        } else {
            sim.t_dag(target)?;
        }
        sim.cx(control, target)?;
        Ok(())
    }

    // Invert the high-level controlled powers and inverse QFT independently
    // of the importer's shared seven-T phase-polynomial decomposition.
    sim.h(b);
    controlled_s(sim, b, a, false)?;
    sim.h(a);
    sim.swap(a, b);
    sim.cz(b, c)?;
    sim.ccz(a, c, d)?;
    controlled_s(sim, a, c, true)?;
    for qubit in [a, b, c, d] {
        sim.h(qubit);
    }
    Ok(())
}

fn assert_bell_pairs(sim: &Simulator, outputs: &[usize], references: &[usize]) {
    assert_eq!(outputs.len(), references.len());
    for (&output, &reference) in outputs.iter().zip(references) {
        for pauli in [EnginePauli::X, EnginePauli::Z] {
            let stabilizer = &EnginePauliString::single(sim.num_qubits(), output, pauli)
                * &EnginePauliString::single(sim.num_qubits(), reference, pauli);
            let value = sim
                .peek_observable_expectation(&stabilizer)
                .expect("output/reference Bell stabilizer has a valid engine observable");
            assert!((value - 1.0).abs() < 1e-9, "Bell {pauli:?}: {value}");
        }
    }
}

fn decode_output(
    sim: &mut Simulator,
    output: &bloq_vm::OutputLogical,
    frame: &bloq_vm::ShotFrame,
) -> Result<usize, ExecError> {
    let qubit = sim.num_qubits();
    let (x, z) = (
        widen(&output.logical_x, qubit + 1),
        widen(&output.logical_z, qubit + 1),
    );
    let ancilla = physical_qubit(qubit, qubit + 1);
    sim.reset(qubit)?;
    // A coherent logical SWAP retains all output/reference correlations.
    sim.controlled_pauli(&z, &ancilla.logical_x)?;
    sim.controlled_pauli(&ancilla.logical_z, &x)?;
    sim.controlled_pauli(&z, &ancilla.logical_x)?;
    if frame.x.expect("output X frame is evaluable") {
        sim.x(qubit);
    }
    if frame.z.expect("output Z frame is evaluable") {
        sim.z(qubit);
    }
    Ok(qubit)
}

#[test]
fn temporal_qpe_preserves_reference_entanglement_in_every_bell_sector() {
    let fixture = fixture("qpe4_temporal");
    assert_eq!(fixture.inputs.len(), 4);
    assert_eq!(fixture.outputs.len(), 4);
    assert_eq!(fixture.t_gadgets, 7);
    let logical = fixture
        .verifier()
        .verify(&fixture.expected(), 32, 7)
        .expect("temporal-only QPE preserves the complete source operator");
    assert_eq!(logical.verified, 32);

    // Check the independently written inverse against the actual source map.
    let mut source = Simulator::with_seed(8, 0);
    for input in 0..4 {
        source.h(input);
        source.cx(input, input + 4).unwrap();
    }
    apply_source_qasm(&mut source, &fixture.qasm);
    undo_qpe(&mut source, [0, 1, 2, 3]).unwrap();
    assert_bell_pairs(&source, &[0, 1, 2, 3], &[4, 5, 6, 7]);

    let mut bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("temporal-only QPE compiles with ideal logical T sources")
        .bloq;
    let bell = BellOutcomes::expose(&mut bloq);
    bloq.validate().unwrap();
    let references = RefCell::new(Vec::new());
    let mut bell_sectors = BTreeSet::new();
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut scored = 0;
    let report = run_bloq_with_io_and_captured_outputs(
        &bloq,
        16,
        17,
        |sim, context| {
            assert_eq!(context.inputs.len(), 4);
            references.borrow_mut().clear();
            for port in &fixture.inputs {
                let input = context
                    .inputs
                    .iter()
                    .find(|input| input.port == *port)
                    .unwrap();
                let reference = sim.num_qubits();
                sim.cx(input.qubit, reference)?;
                references.borrow_mut().push(reference);
            }
            Ok(())
        },
        |sim, context| {
            assert_eq!(
                context.saved_outputs.len(),
                2,
                "QPE reuses two output patches"
            );
            let logicals = context.output_logicals()?;
            let mut decoded = Vec::new();
            for port in &fixture.outputs {
                let output = logicals.iter().find(|output| output.port == *port).unwrap();
                let frame = context
                    .frames
                    .iter()
                    .find(|frame| frame.port == *port)
                    .unwrap();
                decoded.push(decode_output(sim, output, frame)?);
            }
            undo_qpe(sim, decoded.as_slice().try_into().unwrap())?;
            // These eight commuting +1 stabilizers uniquely fix the complete
            // eight-qubit Bell state, hence the arbitrary-input QPE channel.
            assert_bell_pairs(sim, &decoded, &references.borrow());
            bell_sectors.insert(bell.read(context));
            for &(name, value) in context.named_branch_selectors {
                if name.starts_with("selective ") {
                    selectors.entry(name.into()).or_default().insert(value);
                }
            }
            scored += 1;
            Ok(())
        },
    )
    .expect("reference-entangled QPE execution with captured outputs");
    assert_eq!(scored, 16);
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
    assert!(report.max_rank > 1, "physical non-Clifford execution");
    assert_eq!(
        bell_sectors,
        BTreeSet::from([(false, false), (false, true), (true, false), (true, true)])
    );
    assert_selectors_toggle(&selectors, 7, "topologiq");
}

fn expected_qpe_state_signature() -> Vec<f64> {
    // QPE maps each |target0,target1> to counting value
    // target0*(1+2*target1) mod 4. From |++> targets the four amplitudes are +1/2.
    let mut state = [[0.0; 2]; 16];
    for basis in [0, 5, 8, 15] {
        state[basis][0] = 0.5;
    }
    signature_from_state(&state)
}

#[rstest::rstest]
#[case::source_order(false)]
#[case::independent_measurements_reversed(true)]
fn four_qubit_phase_estimation_executes_proxy_t_with_captured_boundaries(
    #[case] reverse_independent_measurements: bool,
) {
    let mut fixture = fixture("qpe4");
    if reverse_independent_measurements {
        let mut actions = fixture.graph.actions();
        let dependencies = fixture
            .graph
            .action_graph()
            .dependencies()
            .collect::<Vec<_>>();
        let mut depth = vec![0; actions.len()];
        for _ in 0..actions.len() {
            let mut changed = false;
            for &(predecessor, consumer, _) in &dependencies {
                if depth[consumer] <= depth[predecessor] {
                    depth[consumer] = depth[predecessor] + 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Equal DAG depth certifies an antichain: none of these measurements
        // depends on another. Move whole declarations, retaining name/target pairs.
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for (index, action) in actions.iter().enumerate() {
            if matches!(action, Action::Measure { .. }) {
                groups.entry(depth[index]).or_default().push(index);
            }
        }
        let mut changed = false;
        for group in groups.values() {
            for pair in 0..group.len() / 2 {
                actions.swap(group[pair], group[group.len() - 1 - pair]);
                changed = true;
            }
        }
        assert!(changed, "QPE contains independent measurement declarations");
        fixture
            .graph
            .set_actions(actions)
            .expect("permuted declarations preserve the action DAG");
        let report = fixture
            .verifier()
            .verify(&fixture.expected(), 32, 7)
            .unwrap();
        assert_eq!(report.verified, 32);
    }
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect("QPE compiles with ideal logical T-source proxies")
        .bloq;
    assert_prepared_output_state(&fixture, &bloq, &expected_qpe_state_signature());
}

fn assert_prepared_output_state(
    fixture: &Fixture<Metadata>,
    bloq: &bloq_ir::Bloq,
    expected: &[f64],
) {
    let mut scored = 0;
    let report = run_bloq_with_io_and_captured_outputs(
        bloq,
        3,
        17,
        |sim, context| {
            assert_eq!(context.inputs.len(), fixture.inputs.len());
            for input in context.inputs {
                sim.h(input.qubit); // Private |+> seed -> algorithm input |0>.
            }
            Ok(())
        },
        |sim, context| {
            assert!(
                !context.saved_outputs.is_empty(),
                "fixture reuses output patches"
            );
            let outputs = context.output_logicals()?;
            let ordered = fixture
                .outputs
                .iter()
                .map(|port| {
                    let output = outputs
                        .iter()
                        .find(|output| output.port == *port)
                        .expect("source-ordered logical output exists");
                    let frame = context
                        .frames
                        .iter()
                        .find(|frame| frame.port == *port)
                        .expect("logical output has a terminal frame");
                    (
                        output,
                        (
                            frame.x.expect("evaluable X frame"),
                            frame.z.expect("evaluable Z frame"),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let actual = logical_signature(sim, &ordered)?;
            assert!(
                signatures_match(&actual, expected),
                "every corrected output Pauli expectation matches the independent algorithm state"
            );
            scored += 1;
            Ok(())
        },
    )
    .expect("proxy T execution with prepared inputs and ideal boundary capture");
    assert_eq!(scored, 3);
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
    assert!(report.max_rank > 1, "non-Clifford T-source execution");
}

fn signature_from_state(state: &[[f64; 2]]) -> Vec<f64> {
    assert!(state.len().is_power_of_two());
    let norm: f64 = state.iter().map(|[r, i]| r * r + i * i).sum();
    assert!((norm - 1.0).abs() < 1e-12);
    let qubits = state.len().ilog2();
    (1..4usize.pow(qubits))
        .map(|word| {
            state
                .iter()
                .enumerate()
                .map(|(basis, &[ar, ai])| {
                    let mut target = basis;
                    let mut phase = 0;
                    for qubit in 0..qubits {
                        let bit = (basis >> qubit) & 1;
                        match (word >> (2 * qubit)) & 3 {
                            1 => target ^= 1 << qubit,
                            2 => {
                                target ^= 1 << qubit;
                                phase += if bit == 0 { 1 } else { 3 };
                            }
                            3 => phase += 2 * bit,
                            _ => {}
                        }
                    }
                    let [br, bi] = state[target];
                    let real = ar * br + ai * bi;
                    let imaginary = ai * br - ar * bi;
                    match phase % 4 {
                        0 => real,
                        1 => -imaginary,
                        2 => -real,
                        _ => imaginary,
                    }
                })
                .sum()
        })
        .collect()
}

#[test]
fn adder_layout_with_destructive_live_qubit_reuse_is_rejected() {
    let fixture = fixture("algorithms/adder4");
    assert_eq!(fixture.graph.block_count(), 165);
    assert_eq!(fixture.graph.actions().len(), 19);
    assert_eq!(fixture.t_gadgets, 7);
    assert_eq!(fixture.inputs.len(), 4);
    assert_eq!(fixture.outputs.len(), 4);
    let error = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&fixture.graph)
        .expect_err("unsupported Topologiq layout must fail before accepting executable IR");
    assert!(
        matches!(error, CompileError::LiveQubitReuse { .. }),
        "{error}"
    );
}

#[test]
fn grover_preserves_signed_source_branches_and_all_63_output_paulis() {
    let fixture = fixture("algorithms/grover3");
    assert_eq!(fixture.graph.block_count(), 405);
    assert_eq!(fixture.graph.actions().len(), 45);
    assert_eq!(fixture.t_gadgets, 14);
    let logical = LogicalVerifier::with_boundaries(
        &fixture.graph,
        BoundaryOrder::new(fixture.inputs.clone(), fixture.outputs.clone()),
    )
    .expect("Grover admits fixed signed named-outcome verification")
    .verify(&fixture.expected(), 32, 7)
    .expect("every checked signed Grover branch preserves the source operator");
    assert_eq!(logical.verified, 32);
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile_and_validate(&fixture.graph)
        .expect("unchanged Grover layout compiles through strict public parsing")
        .bloq;

    // One Grover iteration has amplitude 5/sqrt(32) at 101 and 1/sqrt(32) elsewhere.
    let state = (0..8)
        .map(|basis| [if basis == 5 { 5.0 } else { 1.0 } / 32.0f64.sqrt(), 0.0])
        .collect::<Vec<_>>();
    let expected = signature_from_state(&state);
    assert_eq!(expected.len(), 63);
    assert!(signatures_match(
        &signature_from_state(fixture.metadata.expected_state.as_ref().unwrap()),
        &expected,
    ));
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut scored = 0;
    let report = run_bloq_with_io_and_captured_outputs(
        &bloq,
        16,
        17,
        |sim, context| {
            assert_eq!(context.inputs.len(), 3);
            for input in context.inputs {
                sim.h(input.qubit);
            }
            Ok(())
        },
        |sim, context| {
            assert_eq!(context.saved_outputs.len(), 1);
            let logicals = context.output_logicals()?;
            let ordered = fixture
                .outputs
                .iter()
                .map(|port| {
                    let output = logicals.iter().find(|output| output.port == *port).unwrap();
                    let frame = context
                        .frames
                        .iter()
                        .find(|frame| frame.port == *port)
                        .unwrap();
                    (output, (frame.x.unwrap(), frame.z.unwrap()))
                })
                .collect::<Vec<_>>();
            let actual = logical_signature(sim, &ordered)?;
            assert!(signatures_match(&actual, &expected));
            record_selectors(&mut selectors, context.named_branch_selectors);
            scored += 1;
            Ok(())
        },
    )
    .expect("Grover executes with ideal T sources and captured output boundaries");
    assert_eq!(scored, 16);
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
    assert!(report.max_rank > 1);
    assert_selectors_toggle(&selectors, 14, "topologiq");
}

fn undo_grover(sim: &mut Simulator, outputs: [usize; 3]) -> Result<(), ExecError> {
    // Invert H^3 -> oracle(101) -> diffusion independently of T synthesis.
    for qubit in outputs {
        sim.h(qubit);
        sim.x(qubit);
    }
    sim.ccz(outputs[0], outputs[1], outputs[2])?;
    for qubit in outputs {
        sim.x(qubit);
        sim.h(qubit);
    }
    sim.x(outputs[1]);
    sim.ccz(outputs[0], outputs[1], outputs[2])?;
    sim.x(outputs[1]);
    for qubit in outputs {
        sim.h(qubit);
    }
    Ok(())
}

#[test]
fn grover_preserves_all_three_input_references() {
    let fixture = fixture("algorithms/grover3");
    let mut source = Simulator::with_seed(6, 0);
    for input in 0..3 {
        source.h(input);
        source.cx(input, input + 3).unwrap();
    }
    apply_source_qasm(&mut source, &fixture.qasm);
    undo_grover(&mut source, [0, 1, 2]).unwrap();
    assert_bell_pairs(&source, &[0, 1, 2], &[3, 4, 5]);

    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile_and_validate(&fixture.graph)
        .expect("strictly parsed Grover compiles with ideal logical T sources")
        .bloq;
    let references = RefCell::new(Vec::new());
    let mut selectors = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut scored = 0;
    let report = run_bloq_with_io_and_captured_outputs(
        &bloq,
        16,
        17,
        |sim, context| {
            assert_eq!(context.inputs.len(), 3);
            references.borrow_mut().clear();
            for port in &fixture.inputs {
                let input = context
                    .inputs
                    .iter()
                    .find(|input| input.port == *port)
                    .unwrap();
                let reference = sim.num_qubits();
                sim.cx(input.qubit, reference)?;
                references.borrow_mut().push(reference);
            }
            Ok(())
        },
        |sim, context| {
            assert_eq!(context.saved_outputs.len(), 1);
            let logicals = context.output_logicals()?;
            let mut decoded = Vec::new();
            for port in &fixture.outputs {
                let output = logicals.iter().find(|output| output.port == *port).unwrap();
                let frame = context
                    .frames
                    .iter()
                    .find(|frame| frame.port == *port)
                    .unwrap();
                decoded.push(decode_output(sim, output, frame)?);
            }
            undo_grover(sim, decoded.as_slice().try_into().unwrap())?;
            // Six commuting +1 stabilizers uniquely fix the complete six-qubit Choi state.
            assert_bell_pairs(sim, &decoded, &references.borrow());
            record_selectors(&mut selectors, context.named_branch_selectors);
            scored += 1;
            Ok(())
        },
    )
    .expect("Grover preserves the reference-entangled input channel");
    assert_eq!(scored, 16);
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
    assert!(report.max_rank > 1);
    assert_selectors_toggle(&selectors, 14, "topologiq");
}

#[rstest::rstest]
#[case("t")]
#[case("frames/direct_twins_input_t")]
#[case("frames/direct_tree_input_t")]
#[case("frames/direct_split_input_t")]
#[case("frames/direct_h_feedback_t")]
#[case("frames/direct_cap_padding_t")]
#[case("frames/direct_upstream_h_t")]
fn wrong_selector_and_dagger_sign_fail_the_independent_logical_oracle(#[case] name: &str) {
    let mut positive = fixture(name);
    let ideal_t = positive.expected();
    let mut actions = positive.graph.actions();
    let condition = actions
        .iter_mut()
        .find_map(|action| match action {
            Action::Resolve { condition, .. } => Some(condition),
            _ => None,
        })
        .expect("T fixture contains a selective resolution");
    *condition = Expr::Not(Box::new(condition.clone()));
    positive
        .graph
        .set_actions(actions)
        .expect("inverted selector is structurally valid");
    positive
        .verifier()
        .verify(&ideal_t, 32, 7)
        .expect_err("the logical oracle rejects the inverted selector");

    let dagger = fixture(&format!("{name}dg"));
    dagger
        .verifier()
        .verify(&ideal_t, 32, 7)
        .expect_err("the logical oracle distinguishes T dagger from T");

    if name.starts_with("frames/") {
        let mut missing_correction = fixture(name);
        let actions = missing_correction
            .graph
            .actions()
            .into_iter()
            .filter(|action| !matches!(action, Action::Feedback { .. }))
            .collect();
        missing_correction.graph.set_actions(actions).unwrap();
        missing_correction
            .verifier()
            .verify(&ideal_t, 32, 7)
            .expect_err("the input-transport recipe requires the conditional X correction");
    }
}
