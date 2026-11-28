//! Resolve delayed flag records on the merged component, after physical noise has
//! been assigned. The correction is a classical Pauli frame and adds no tick.

use bloq_circuit::{BodyId, ConditionalCorrection, CoordCircuit, Op, Pauli, PauliBasis, PauliMap};
use glam::IVec2;

use crate::{FxMap, ReadoutFrame, instantiation::NodeTemplateInstanceMergeError as Error};

type Layer = FxMap<IVec2, (Pauli, IVec2, Pauli)>;

pub(crate) fn resolve(circuit: &mut CoordCircuit, frames: &[ReadoutFrame]) -> Result<(), Error> {
    let mut controls = FxMap::<_, Vec<_>>::default();
    for frame in frames {
        controls.entry(frame.flag).or_default().push(*frame);
    }
    let mut unresolved: crate::FxSet<_> = controls.keys().copied().collect();
    for i in 0..circuit.body_count() {
        let body = circuit.body_mut(BodyId(i as u32)).expect("allocated body");
        if !body.ops().iter().any(|op| matches!(op, Op::Measure {measurements, ..} if measurements.iter().any(|m|controls.contains_key(m)))) {
            continue;
        }
        let source = std::mem::take(body.ops_mut());
        let out = body.ops_mut();
        let mut history = Vec::<Layer>::new();
        for (index, moment) in source.split(|op| matches!(op, Op::Tick)).enumerate() {
            if index != 0 {
                out.push(Op::Tick);
            }
            let mut layer = Layer::default();
            let mut flags = Vec::new();
            for op in moment {
                match op {
                    Op::Gate { gate, qubits } => {
                        if let Some((a, b)) = gate.two_qubit_bases() {
                            for pair in qubits.as_chunks::<2>().0 {
                                layer.insert(pair[0], (Pauli::from(a), pair[1], Pauli::from(b)));
                                layer.insert(pair[1], (Pauli::from(b), pair[0], Pauli::from(a)));
                            }
                        } else {
                            history.clear();
                        }
                    }
                    Op::Measure { measurements, .. } => {
                        flags.extend(
                            measurements
                                .iter()
                                .filter(|m| controls.contains_key(m))
                                .copied(),
                        );
                    }
                    Op::Repeat { .. } | Op::MPP { .. } | Op::ConditionalPauli(_) => history.clear(),
                    _ => {}
                }
            }
            if flags.is_empty() {
                out.extend_from_slice(moment);
            } else {
                let mut corrections = Vec::new();
                for id in flags {
                    unresolved.remove(&id);
                    for frame in &controls[&id] {
                        let propagated = propagate(frame, &history)?;
                        for (&target, &pauli) in propagated.iter() {
                            let pauli = match pauli {
                                Pauli::X => PauliBasis::X,
                                Pauli::Y => PauliBasis::Y,
                                Pauli::Z => PauliBasis::Z,
                                Pauli::I => continue,
                            };
                            corrections.push(ConditionalCorrection {
                                control: id,
                                target,
                                pauli,
                            });
                        }
                    }
                }
                write_readout(moment, &controls, corrections, out)?;
            }
            if moment.iter().any(|op| matches!(op, Op::Measure { .. })) {
                history.clear();
            } else if !layer.is_empty() {
                history.push(layer);
            }
        }
    }
    if let Some(id) = unresolved.into_iter().min() {
        return Err(Error::InvalidReadoutFrame(format!(
            "m{id} must come from a single-qubit measurement"
        )));
    }
    Ok(())
}

/// Conjugate the data Pauli through subsequent disjoint controlled-Pauli layers.
fn propagate(frame: &ReadoutFrame, history: &[Layer]) -> Result<PauliMap, Error> {
    let after = history
        .iter()
        .rposition(|layer| {
            layer
                .get(&frame.target)
                .is_some_and(|(_, q, _)| *q == frame.ancilla)
        })
        .ok_or_else(|| {
            Error::InvalidReadoutFrame(format!(
                "m{} has no preceding data interaction on {:?}",
                frame.flag, frame.target
            ))
        })?;
    let mut propagated = PauliMap::from_unique_entries([(frame.target, Pauli::from(frame.pauli))]);
    for layer in &history[after + 1..] {
        let flips: Vec<_> = propagated
            .iter()
            .filter_map(|(q, p)| {
                let &(basis, neighbor, other) = layer.get(q)?;
                p.anticommutes(basis).then_some((neighbor, other))
            })
            .collect();
        for (q, p) in flips {
            let old = propagated.get(&q).copied().unwrap_or(Pauli::I);
            propagated.insert(q, old ^ p);
        }
    }
    Ok(propagated)
}

/// Read flags before the feedback and all other records after it. Preserve
/// the order of physical noise on each measured qubit, including interleaved
/// noise in editable Bloq templates.
fn write_readout(
    moment: &[Op],
    controls: &FxMap<u32, Vec<ReadoutFrame>>,
    corrections: Vec<ConditionalCorrection>,
    out: &mut Vec<Op>,
) -> Result<(), Error> {
    let first = moment
        .iter()
        .position(|op| matches!(op, Op::Measure { .. }))
        .expect("flag in moment");
    if moment[..first].iter().any(|op| {
        !matches!(
            op,
            Op::Depolarize1 { .. } | Op::Depolarize2 { .. } | Op::PauliError { .. }
        )
    }) {
        return Err(Error::InvalidReadoutFrame(
            "flag readout overlaps a quantum gate".into(),
        ));
    }
    out.extend_from_slice(&moment[..first]);
    let mut before = Vec::new();
    let mut after = Vec::new();
    let mut required = crate::FxSet::default();
    // Starting from the flags, find the noise operations that must precede
    // them. Correlated noise keeps all of its targets on the same side.
    for op in moment[first..].iter().rev() {
        match op {
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability,
            } => {
                for (&q, &id) in qubits.iter().zip(measurements).rev() {
                    let destination = if controls.contains_key(&id) {
                        required.insert(q);
                        &mut before
                    } else if required.contains(&q) {
                        return Err(Error::InvalidReadoutFrame(
                            "noise ordering prevents simultaneous flag readout".into(),
                        ));
                    } else {
                        &mut after
                    };
                    destination.push(Op::Measure {
                        basis: *basis,
                        qubits: vec![q],
                        measurements: vec![id],
                        flip_probability: *flip_probability,
                    });
                }
            }
            Op::Depolarize1 { qubits, .. }
            | Op::Depolarize2 { qubits, .. }
            | Op::PauliError { qubits, .. } => {
                if qubits.iter().any(|q| required.contains(q)) {
                    required.extend(qubits.iter().copied());
                    before.push(op.clone());
                } else {
                    after.push(op.clone());
                }
            }
            _ => {
                return Err(Error::InvalidReadoutFrame(
                    "flag readout overlaps a quantum gate".into(),
                ));
            }
        }
    }
    out.extend(before.into_iter().rev());
    out.push(Op::ConditionalPauli(corrections));
    out.extend(after.into_iter().rev());
    Ok(())
}
