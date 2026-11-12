//! Circuit noise models and their materialization as explicit operations.
//!
//! Applying a [`NoiseModel`] produces an ordinary [`CoordCircuit`] containing
//! depolarizing, Pauli-error, and measurement-flip instructions. Backends,
//! simulators, and decoders can therefore consume the same operation stream.

use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{BodyId, CircuitError, CoordCircuit, GateType, Op, PauliBasis};

/// Circuit-level Pauli noise probabilities materialized by
/// [`NoiseModel::noisy_circuit`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct NoiseModel {
    /// Single-qubit depolarizing probability after each noisy 1q gate.
    pub p1: f64,
    /// Two-qubit depolarizing probability after each 2q gate.
    pub p2: f64,
    /// Measurement result-flip probability.
    pub p_meas: f64,
    /// Orthogonal-state preparation probability after each reset.
    pub p_reset: f64,
    /// Per-tick single-qubit depolarizing probability on idle footprint qubits.
    pub p_idle: f64,
}

impl NoiseModel {
    /// Sets every channel to `probability`.
    #[must_use]
    pub const fn uniform_depolarizing(probability: f64) -> Self {
        Self {
            p1: probability,
            p2: probability,
            p_meas: probability,
            p_reset: probability,
            p_idle: probability,
        }
    }

    /// Whether idle noise needs repeat expansion to account for moments that
    /// cross a repeat boundary. Annotation owners must flatten their side tables
    /// together with the circuit before applying noise in this case.
    pub fn requires_repeat_flattening(&self, circuit: &CoordCircuit) -> bool {
        self.p_idle != 0.0 && idle_body_shape(circuit, circuit.entry_body()).is_none()
    }

    /// Clone `circuit` and materialize this model as explicit noise operations.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError`] if repeat expansion fails.
    pub fn noisy_circuit(&self, circuit: &CoordCircuit) -> Result<CoordCircuit, CircuitError> {
        self.noisy_circuit_excluding(circuit, &[])
    }

    /// [`Self::noisy_circuit`] while keeping `ideal_qubits` free of
    /// generated or pre-existing error channels. A two-qubit channel is
    /// removed when either member of its pair is ideal. Repeats sharing a moment
    /// with their surroundings are flattened; final measurement ids are retained.
    ///
    /// # Errors
    ///
    /// Returns the circuit's flatten error if a repeat needing expansion is
    /// invalid, exhausts measurement ids, or exceeds the expansion allowance.
    /// Returns [`CircuitError::BodyIdOutOfRange`] if a body index exceeds `u32::MAX`.
    pub fn noisy_circuit_excluding(
        &self,
        circuit: &CoordCircuit,
        ideal_qubits: &[IVec2],
    ) -> Result<CoordCircuit, CircuitError> {
        let mut noisy = circuit.clone();
        if self.requires_repeat_flattening(circuit) {
            // ponytail: partial moments expand; use loop peeling if large partial-moment loops matter.
            noisy.flatten()?;
        }
        let footprint = noisy.qubits();
        let ideal_qubits = ideal_qubits.iter().copied().collect::<FxHashSet<_>>();
        for index in 0..noisy.body_count() {
            let body = crate::circuit::checked_body_id(index)?;
            annotate_body(&mut noisy, body, &footprint, &ideal_qubits, self);
        }
        Ok(noisy)
    }

    /// Annotate the operations of one tick-free moment and accumulate its
    /// active qubits. The caller applies idle noise after the aligned tick.
    pub fn noisy_moment(
        &self,
        ops: Vec<Op>,
        active: &mut FxHashSet<IVec2>,
        ideal_qubits: &FxHashSet<IVec2>,
    ) -> Vec<Op> {
        annotate_ops(ops, &FxHashSet::default(), ideal_qubits, active, self)
    }

    /// Build the idle error for a tick from its idle, nonideal qubits.
    pub fn idle_error(&self, qubits: impl IntoIterator<Item = IVec2>) -> Option<Op> {
        if self.p_idle == 0.0 {
            return None;
        }
        let mut qubits: Vec<_> = qubits.into_iter().collect();
        qubits.sort_unstable_by_key(|coord| (coord.x, coord.y));
        (!qubits.is_empty()).then_some(Op::Depolarize1 {
            probability: self.p_idle,
            qubits,
        })
    }
}

// `Some(active)` means body-local noising is sound and reports whether the
// body's last moment is still active. A repeat entered or left inside a moment
// needs expansion: its first iteration may have different idle qubits from later
// iterations, so one shared body cannot hold the right channels for both.
fn idle_body_shape(circuit: &CoordCircuit, entry: BodyId) -> Option<bool> {
    let mut cache = FxHashMap::default();
    let mut visiting = FxHashSet::default();
    visiting.insert(entry);
    let mut pending = vec![(entry, 0usize, false)];
    while let Some((body, next, active)) = pending.last_mut() {
        let Some(op) = circuit.body(*body)?.ops().get(*next) else {
            cache.insert(*body, *active);
            visiting.remove(body);
            pending.pop();
            continue;
        };
        *next += 1;
        match op {
            Op::Repeat { body, repetitions } if *repetitions != 0 => {
                if *active {
                    return None;
                }
                if let Some(&unfinished) = cache.get(body) {
                    if unfinished {
                        return None;
                    }
                } else {
                    if !visiting.insert(*body) {
                        return None;
                    }
                    // Revisit this repeat once its body shape is known.
                    *next -= 1;
                    pending.push((*body, 0, false));
                }
            }
            Op::Tick => *active = false,
            Op::Gate { qubits, .. } | Op::Measure { qubits, .. } => *active |= !qubits.is_empty(),
            Op::MPP { products, .. } => {
                *active |= products.iter().any(|product| !product.is_empty())
            }
            Op::ConditionalPauli(corrections) => *active |= !corrections.is_empty(),
            Op::Repeat { .. }
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => {}
        }
    }
    cache.get(&entry).copied()
}

fn annotate_body(
    circuit: &mut CoordCircuit,
    body: BodyId,
    footprint: &FxHashSet<IVec2>,
    ideal_qubits: &FxHashSet<IVec2>,
    noise: &NoiseModel,
) {
    let body = circuit
        .body_mut(body)
        .expect("body id comes from this circuit's dense body range");
    let ops = body.ops_mut();
    *ops = annotate_ops(
        std::mem::take(ops),
        footprint,
        ideal_qubits,
        &mut FxHashSet::default(),
        noise,
    );
}

fn annotate_ops(
    source: Vec<Op>,
    footprint: &FxHashSet<IVec2>,
    ideal_qubits: &FxHashSet<IVec2>,
    active: &mut FxHashSet<IVec2>,
    noise: &NoiseModel,
) -> Vec<Op> {
    let mut ops = Vec::with_capacity(source.len() * 2);

    for mut op in source {
        match &mut op {
            Op::Gate { gate, qubits } => {
                // Idling qubits count as active either way, so the tick closing
                // the moment cannot stack `p_idle` on top of whichever channel
                // this gate earned.
                active.extend(qubits.iter().copied());
                let error = gate_noise(*gate, qubits, ideal_qubits, noise);
                ops.push(op);
                if let Some(error) = error {
                    ops.push(error);
                }
            }
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability,
            } => {
                active.extend(qubits.iter().copied());
                if qubits.iter().any(|qubit| ideal_qubits.contains(qubit)) {
                    for (&qubit, &measurement) in qubits.iter().zip(measurements.iter()) {
                        ops.push(Op::Measure {
                            basis: *basis,
                            qubits: vec![qubit],
                            measurements: vec![measurement],
                            flip_probability: if ideal_qubits.contains(&qubit) {
                                0.0
                            } else {
                                compose_flip_probabilities(*flip_probability, noise.p_meas)
                            },
                        });
                    }
                } else {
                    *flip_probability = compose_flip_probabilities(*flip_probability, noise.p_meas);
                    ops.push(op);
                }
            }
            Op::MPP { products, .. } => {
                active.extend(
                    products
                        .iter()
                        .flat_map(|product| product.iter().map(|(coord, _)| *coord)),
                );
                ops.push(op);
            }
            Op::ConditionalPauli(corrections) => {
                active.extend(corrections.iter().map(|correction| correction.target));
                ops.push(op);
            }
            Op::Tick => {
                ops.push(op);
                if let Some(error) = noise.idle_error(
                    footprint
                        .difference(active)
                        .filter(|qubit| !ideal_qubits.contains(*qubit))
                        .copied(),
                ) {
                    ops.push(error);
                }
                active.clear();
            }
            Op::Depolarize1 { qubits, .. } | Op::PauliError { qubits, .. } => {
                *qubits = nonideal_qubits(qubits, ideal_qubits);
                if !qubits.is_empty() {
                    ops.push(op);
                }
            }
            Op::Depolarize2 { qubits, .. } => {
                *qubits = nonideal_pairs(qubits, ideal_qubits);
                if !qubits.is_empty() {
                    ops.push(op);
                }
            }
            Op::Repeat { .. } => ops.push(op),
        }
    }

    ops
}

fn gate_noise(
    gate: GateType,
    qubits: &[IVec2],
    ideal_qubits: &FxHashSet<IVec2>,
    noise: &NoiseModel,
) -> Option<Op> {
    if gate.is_reset() {
        let qubits = nonideal_qubits(qubits, ideal_qubits);
        return (noise.p_reset != 0.0 && !qubits.is_empty()).then(|| Op::PauliError {
            probability: noise.p_reset,
            pauli: match gate {
                GateType::RX => PauliBasis::Z,
                GateType::RY | GateType::RZ => PauliBasis::X,
                _ => unreachable!("is_reset is exactly RX/RY/RZ"),
            },
            qubits,
        });
    }
    if gate.is_two_qubit_gate() {
        let qubits = nonideal_pairs(qubits, ideal_qubits);
        return (noise.p2 != 0.0 && !qubits.is_empty()).then(|| Op::Depolarize2 {
            probability: noise.p2,
            qubits,
        });
    }
    let qubits = nonideal_qubits(qubits, ideal_qubits);
    (noise.p1 != 0.0 && !qubits.is_empty()).then(|| Op::Depolarize1 {
        probability: noise.p1,
        qubits,
    })
}

fn nonideal_qubits(qubits: &[IVec2], ideal: &FxHashSet<IVec2>) -> Vec<IVec2> {
    qubits
        .iter()
        .filter(|qubit| !ideal.contains(qubit))
        .copied()
        .collect()
}

fn nonideal_pairs(qubits: &[IVec2], ideal: &FxHashSet<IVec2>) -> Vec<IVec2> {
    qubits
        .as_chunks::<2>()
        .0
        .iter()
        .filter(|pair| pair.iter().all(|qubit| !ideal.contains(qubit)))
        .flatten()
        .copied()
        .collect()
}

fn compose_flip_probabilities(left: f64, right: f64) -> f64 {
    left + right - 2.0 * left * right
}

#[cfg(test)]
mod tests {
    use glam::ivec2;

    use super::*;

    #[test]
    fn idle_noise_propagates_expansion_limits_without_panicking() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(crate::CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::H,
            qubits: vec![IVec2::ZERO],
        }]));
        circuit.push_repeat(body, u32::MAX);
        let noise = NoiseModel::uniform_depolarizing(0.1);
        assert!(matches!(
            noise.noisy_circuit(&circuit),
            Err(CircuitError::FlattenResourceLimit { .. })
        ));
    }

    #[test]
    fn idle_noise_is_unchanged_by_repeat_expansion() {
        let a = ivec2(0, 0);
        let b = ivec2(1, 0);
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };
        for close_body in [false, true] {
            for repetitions in [1, 3] {
                let mut circuit = CoordCircuit::new();
                circuit.do_gate(GateType::H, [a]).unwrap();
                let mut ops = vec![Op::Gate {
                    gate: GateType::H,
                    qubits: vec![b],
                }];
                if close_body {
                    ops.push(Op::Tick);
                }
                let body = circuit.add_body(crate::CircuitBody::from_ops(ops));
                circuit.push_repeat(body, repetitions);
                circuit.tick();
                let mut flat = circuit.clone();
                flat.flatten().unwrap();
                let expected = noise.noisy_circuit(&flat).unwrap();
                let mut actual = noise.noisy_circuit(&circuit).unwrap();
                actual.flatten().unwrap();
                assert_eq!(
                    actual.body(actual.entry_body()).unwrap().ops(),
                    expected.body(expected.entry_body()).unwrap().ops()
                );
            }
        }

        let mut closed = CoordCircuit::new();
        closed.do_gate(GateType::H, [a]).unwrap();
        closed.tick();
        let body = closed.add_body(crate::CircuitBody::from_ops(vec![
            Op::Gate {
                gate: GateType::H,
                qubits: vec![b],
            },
            Op::Tick,
        ]));
        closed.push_repeat(body, 100);
        assert!(!noise.requires_repeat_flattening(&closed));
        assert_eq!(
            noise
                .noisy_circuit(&closed)
                .unwrap()
                .entry_top_level_repeats(),
            [100]
        );
    }

    #[test]
    fn model_materializes_gate_reset_measurement_and_idle_noise() {
        let q0 = ivec2(0, 0);
        let q1 = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [q0]).unwrap();
        circuit.tick();
        circuit.do_gate(GateType::H, [q0]).unwrap();
        circuit.do_gate(GateType::CX, [q0, q1]).unwrap();
        circuit.measure(PauliBasis::Z, [q0]);

        let noisy = NoiseModel {
            p1: 0.1,
            p2: 0.2,
            p_meas: 0.3,
            p_reset: 0.4,
            p_idle: 0.5,
        }
        .noisy_circuit(&circuit)
        .unwrap();
        let ops = noisy.body(noisy.entry_body()).unwrap().ops();

        assert!(matches!(
            &ops[1],
            Op::PauliError { probability, pauli: PauliBasis::X, qubits }
                if *probability == 0.4 && qubits == &[q0]
        ));
        assert!(matches!(
            &ops[3],
            Op::Depolarize1 { probability, qubits }
                if *probability == 0.5 && qubits == &[q1]
        ));
        assert!(ops.iter().any(|op| matches!(
            op,
            Op::Depolarize1 { probability, qubits }
                if *probability == 0.1 && qubits == &[q0]
        )));
        assert!(ops.iter().any(|op| matches!(
            op,
            Op::Depolarize2 { probability, qubits }
                if *probability == 0.2 && qubits == &[q0, q1]
        )));
        assert!(ops.iter().any(|op| matches!(
            op,
            Op::Measure { flip_probability, .. } if *flip_probability == 0.3
        )));
    }

    #[test]
    fn applying_zero_noise_keeps_operation_stream_unchanged() {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
        circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
        let expected = circuit.body(circuit.entry_body()).unwrap().ops().to_vec();

        let noisy = NoiseModel::uniform_depolarizing(0.0)
            .noisy_circuit(&circuit)
            .unwrap();

        assert_eq!(noisy.body(noisy.entry_body()).unwrap().ops(), expected);
    }

    #[test]
    fn excluded_qubits_receive_no_error_channel() {
        let ideal = ivec2(0, 0);
        let real = ivec2(1, 0);
        let other = ivec2(2, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [ideal, real]).unwrap();
        circuit.do_gate(GateType::H, [ideal, real]).unwrap();
        circuit.do_gate(GateType::CX, [ideal, real]).unwrap();
        circuit.do_gate(GateType::CX, [real, other]).unwrap();
        circuit.measure(PauliBasis::Z, [ideal, real]);

        let noisy = NoiseModel::uniform_depolarizing(0.1)
            .noisy_circuit_excluding(&circuit, &[ideal])
            .unwrap();
        let ops = noisy.body(noisy.entry_body()).unwrap().ops();

        for op in ops {
            match op {
                Op::Depolarize1 { qubits, .. } | Op::PauliError { qubits, .. } => {
                    assert!(!qubits.contains(&ideal), "{op:?}");
                }
                Op::Depolarize2 { qubits, .. } => {
                    assert!(
                        qubits
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .all(|pair| !pair.contains(&ideal)),
                        "{op:?}"
                    );
                }
                Op::Measure {
                    qubits,
                    flip_probability,
                    ..
                } if qubits == &[ideal] => assert_eq!(*flip_probability, 0.0),
                _ => {}
            }
        }
        assert!(ops.iter().any(|op| matches!(
            op,
            Op::Depolarize2 { qubits, .. } if qubits == &[real, other]
        )));
        assert!(ops.iter().any(|op| matches!(
            op,
            Op::Measure { qubits, flip_probability, .. }
                if qubits == &[real] && *flip_probability > 0.0
        )));
    }
}
