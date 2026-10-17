//! Translate a flattened [`CoordCircuit`] op stream into engine instructions,
//! and replay it.
//!
//! The IR addresses qubits by [`IVec2`] coordinate and measurements by circuit-
//! local id; the engine addresses both by dense index. Nothing about that
//! translation depends on the shot, so it happens once — in [`prepare_ops`] —
//! and every shot replays the result through
//! [`Simulator::apply_batch`](crate::backend::Simulator::apply_batch).

use bloq_ir::circuit::{
    ConditionalCorrection, CoordCircuit, Op, Pauli as BloqPauli, PauliBasis, PauliMap,
};
use bloq_ir::lowering::validate_template_circuit;
use glam::IVec2;
use rustc_hash::FxHashMap;

use super::ExecError;
use super::gate_table::single_qubit_instruction;
use crate::backend::{Instruction, Pauli, PauliBasis as EnginePauliBasis, PauliString, Simulator};

/// The measurement outcomes of one shot, indexed by (global) measurement id.
///
/// Outcome `true` is the `−1` eigenvalue (a stim "1" record bit).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShotRecord {
    records: Vec<Option<bool>>,
    aliases: FxHashMap<u32, u32>,
}

impl ShotRecord {
    /// The outcome of measurement `id`, or `None` if it was not produced.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<bool> {
        let id = self.aliases.get(&id).copied().unwrap_or(id);
        self.records.get(id as usize).copied().flatten()
    }

    /// The outcome of measurement `id`, erroring if it has not been produced.
    ///
    /// # Errors
    /// [`ExecError::MissingRecord`] if `id` is absent.
    pub fn require(&self, id: u32) -> Result<bool, ExecError> {
        self.get(id).ok_or(ExecError::MissingRecord(id))
    }

    pub(crate) fn set(&mut self, id: u32, value: bool) {
        let id = id as usize;
        if id >= self.records.len() {
            self.records.resize(id + 1, None);
        }
        self.records[id] = Some(value);
    }

    pub(crate) fn with_len(len: u32) -> Self {
        Self {
            records: vec![None; len as usize],
            aliases: FxHashMap::default(),
        }
    }

    /// Each selected plan maps virtual instance records directly to one
    /// physical record. RUS snapshots retain this map with the outcomes.
    pub(crate) fn select_aliases(&mut self, aliases: &[(u32, u32)]) {
        self.aliases.extend(aliases.iter().copied());
    }
}

/// Resolve global circuit coordinates to engine indices.
pub(crate) struct LayoutCtx<'a> {
    pub(crate) coord_to_index: &'a FxHashMap<IVec2, u32>,
    pub(crate) qubit_count: usize,
}

impl LayoutCtx<'_> {
    /// Engine index of a global coordinate.
    pub(crate) fn index(&self, coord: IVec2) -> Result<usize, ExecError> {
        self.coord_to_index
            .get(&coord)
            .map(|&i| i as usize)
            .ok_or(ExecError::UnknownCoord(coord))
    }
}

// ==============================================================================
// Translation
// ==============================================================================

/// One op stream in engine form: the instructions, plus the global record id
/// each of its measurements produces.
///
/// The two vectors are parallel by construction — `record_ids[i]` names the
/// `i`-th [`Instruction::Measure`] of `instructions`, which is exactly the
/// indexing [`crate::backend::BatchOutcome::records`] uses.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedOps {
    instructions: Vec<Instruction>,
    record_ids: Vec<u32>,
}

impl PreparedOps {
    /// Replay the stream on `sim`, binding each outcome to its global record
    /// id. Returns the peak stabilizer rank the batch passed through.
    ///
    /// # Errors
    /// Propagates engine errors. A failure aborts the batch where it happened
    /// and no outcome is recorded, which is fine because an engine error ends
    /// the run: nothing downstream ever reads a half-populated record set.
    pub(crate) fn run(
        &self,
        sim: &mut Simulator,
        record: &mut ShotRecord,
    ) -> Result<usize, ExecError> {
        let outcome = sim.apply_batch(&self.instructions)?;
        debug_assert_eq!(
            outcome.records.len(),
            self.record_ids.len(),
            "preparation counted the batch's measurements"
        );
        for (&id, result) in self.record_ids.iter().zip(&outcome.records) {
            record.set(id, result.outcome);
        }
        Ok(outcome.max_rank)
    }

    /// The global record ids this stream produces, in batch order.
    #[cfg(test)]
    pub(crate) fn record_ids(&self) -> &[u32] {
        &self.record_ids
    }
}

/// Translate `ops` into engine instructions. `remap` maps a circuit-local
/// measurement id to the id its outcome is recorded under (identity for a
/// standalone circuit).
///
/// # Errors
/// [`ExecError::UnflattenedRepeat`] if any repeat remains,
/// [`ExecError::UnknownCoord`] if an operand is outside `layout`,
/// [`ExecError::EmptyMppProduct`] for an empty MPP product, or
/// [`ExecError::MissingRecord`] if a conditional correction reads a
/// measurement the same stream has not produced yet.
pub(crate) fn prepare_ops(
    ops: &[Op],
    layout: &LayoutCtx<'_>,
    remap: &impl Fn(u32) -> u32,
) -> Result<PreparedOps, ExecError> {
    let mut prepared = PreparedOps::default();
    // Circuit-local measurement id to its position in the batch's outcome
    // vector, which is how a conditional correction names its control. Only
    // ids this stream has already measured are in here, so a correction that
    // reads ahead of itself is caught during preparation rather than once per
    // shot.
    let mut produced: FxHashMap<u32, usize> = FxHashMap::default();

    for op in ops {
        match op {
            Op::Gate { gate, qubits } => {
                if gate.is_two_qubit_gate() {
                    let (control, target) =
                        gate.two_qubit_bases().expect("two-qubit gate has bases");
                    debug_assert!(qubits.len().is_multiple_of(2), "two-qubit gate arg count");
                    for pair in qubits.as_chunks::<2>().0 {
                        prepared.instructions.push(Instruction::Gate2 {
                            control: sim_basis(control),
                            target: sim_basis(target),
                            control_qubit: layout.index(pair[0])?,
                            target_qubit: layout.index(pair[1])?,
                        });
                    }
                } else {
                    for &coord in qubits {
                        let index = layout.index(coord)?;
                        prepared
                            .instructions
                            .extend(single_qubit_instruction(*gate, index));
                    }
                }
            }
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability,
            } => {
                for (coord, &id) in qubits.iter().zip(measurements) {
                    let obs = basis_pauli(*basis, layout.index(*coord)?, layout.qubit_count);
                    push_measurement(
                        &mut prepared,
                        &mut produced,
                        id,
                        remap(id),
                        obs,
                        *flip_probability,
                    );
                }
            }
            Op::MPP {
                products,
                measurements,
            } => {
                for (product, &id) in products.iter().zip(measurements) {
                    let obs = product_pauli(product, layout)?;
                    push_measurement(&mut prepared, &mut produced, id, remap(id), obs, 0.0);
                }
            }
            Op::ConditionalPauli(corrections) => {
                for correction in corrections {
                    prepared
                        .instructions
                        .push(conditional(correction, layout, remap, &produced)?);
                }
            }
            Op::Depolarize1 {
                probability,
                qubits,
            } => {
                let probabilities = vec![probability / 3.0; 3];
                for &qubit in qubits {
                    let qubit = layout.index(qubit)?;
                    push_random_pauli(
                        &mut prepared,
                        probabilities.clone(),
                        [Pauli::X, Pauli::Y, Pauli::Z]
                            .map(|pauli| PauliString::single(layout.qubit_count, qubit, pauli))
                            .into(),
                    );
                }
            }
            Op::Depolarize2 {
                probability,
                qubits,
            } => {
                debug_assert!(qubits.len().is_multiple_of(2), "two-qubit noise arg count");
                for pair in qubits.as_chunks::<2>().0 {
                    let first = layout.index(pair[0])?;
                    let second = layout.index(pair[1])?;
                    let mut alternatives = Vec::with_capacity(15);
                    for a in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
                        for b in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
                            if a != Pauli::I || b != Pauli::I {
                                alternatives.push(PauliString::from_terms(
                                    layout.qubit_count,
                                    [(first, a), (second, b)],
                                ));
                            }
                        }
                    }
                    push_random_pauli(&mut prepared, vec![probability / 15.0; 15], alternatives);
                }
            }
            Op::PauliError {
                probability,
                pauli,
                qubits,
            } => {
                let pauli: Pauli = sim_basis(*pauli).into();
                for &qubit in qubits {
                    let alternative =
                        PauliString::single(layout.qubit_count, layout.index(qubit)?, pauli);
                    push_random_pauli(&mut prepared, vec![*probability], vec![alternative]);
                }
            }
            Op::Tick => {}
            Op::Repeat { .. } => return Err(ExecError::UnflattenedRepeat),
        }
    }
    Ok(prepared)
}

fn push_measurement(
    prepared: &mut PreparedOps,
    produced: &mut FxHashMap<u32, usize>,
    local: u32,
    global: u32,
    observable: PauliString,
    flip_probability: f64,
) {
    produced.insert(local, prepared.record_ids.len());
    prepared.record_ids.push(global);
    if flip_probability == 0.0 {
        prepared.instructions.push(Instruction::Measure(observable));
    } else {
        prepared
            .instructions
            .push(Instruction::MeasureWithReadoutError {
                observable,
                probability: flip_probability,
            });
    }
}

fn push_random_pauli(
    prepared: &mut PreparedOps,
    probabilities: Vec<f64>,
    alternatives: Vec<PauliString>,
) {
    prepared.instructions.push(Instruction::RandomPauli {
        probabilities,
        alternatives,
        heralded: false,
    });
}

fn conditional(
    correction: &ConditionalCorrection,
    layout: &LayoutCtx<'_>,
    remap: &impl Fn(u32) -> u32,
    produced: &FxHashMap<u32, usize>,
) -> Result<Instruction, ExecError> {
    let control = *produced
        .get(&correction.control)
        .ok_or_else(|| ExecError::MissingRecord(remap(correction.control)))?;
    Ok(Instruction::ConditionalPauli {
        basis: sim_basis(correction.pauli),
        qubit: layout.index(correction.target)?,
        control,
    })
}

/// Require a global identity for every record produced or consumed by `ops`.
pub(crate) fn validate_record_remap(
    ops: &[Op],
    remap: &FxHashMap<u32, u32>,
) -> Result<(), ExecError> {
    for op in ops {
        let records: &[u32] = match op {
            Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => measurements,
            _ => &[],
        };
        for &record in records {
            if !remap.contains_key(&record) {
                return Err(ExecError::MissingRecordRemap(record));
            }
        }
        if let Op::ConditionalPauli(corrections) = op {
            for correction in corrections {
                if !remap.contains_key(&correction.control) {
                    return Err(ExecError::MissingRecordRemap(correction.control));
                }
            }
        }
    }
    Ok(())
}

/// Build the joint Pauli observable of an MPP product.
///
/// The terms sit on distinct coordinates, so this is a set of sites, not a
/// Pauli product: no phase bookkeeping, and a `PauliMap` never carries
/// identity.
pub(super) fn product_pauli(
    product: &PauliMap,
    layout: &LayoutCtx<'_>,
) -> Result<PauliString, ExecError> {
    if product.is_empty() {
        return Err(ExecError::EmptyMppProduct);
    }
    prepare_boundary_operator(product, layout)
}

/// Prepare a positive-Hermitian boundary map once for exact physical folds.
pub(super) fn prepare_boundary_operator(
    product: &PauliMap,
    layout: &LayoutCtx<'_>,
) -> Result<PauliString, ExecError> {
    let mut pauli = PauliString::new(layout.qubit_count);
    for (coord, bloq) in product.iter() {
        pauli.set(layout.index(*coord)?, bloq_pauli(*bloq));
    }
    Ok(pauli)
}

/// Pad an engine operator with trailing identity sites during program
/// preparation. The coefficient is preserved exactly.
pub(super) fn widen_pauli(pauli: &PauliString, nqubits: usize) -> PauliString {
    debug_assert!(nqubits >= pauli.nqubits);
    let mut widened = PauliString::from_terms(
        nqubits,
        (0..pauli.nqubits).map(|qubit| (qubit, pauli.get(qubit))),
    );
    widened.set_phase(pauli.phase_exponent());
    widened
}

// ==============================================================================
// Flat single-circuit executor
// ==============================================================================

/// Replays a flattened [`CoordCircuit`] against a [`Simulator`].
///
/// Qubit coordinates are mapped to dense engine indices once (sorted by
/// `(x, y)`, matching [`CoordCircuit::build_coord_to_index`]); measurement ids
/// are used as-is. For whole-program execution (global layout, per-instance
/// offsets and id remaps) see [`run_bloq`](super::run_bloq).
#[derive(Debug)]
pub struct CircuitExecutor {
    qubit_count: usize,
    prepared: PreparedOps,
}

impl CircuitExecutor {
    /// Build an executor over a **flattened** circuit (no [`Op::Repeat`]),
    /// translating it to engine instructions up front.
    ///
    /// # Errors
    ///
    /// [`ExecError::UnflattenedRepeat`] if any repeat remains,
    /// [`ExecError::InvalidCircuit`] if the circuit fails the same structural
    /// preflight used by Bloq IR template emission, or any
    /// circuit-preparation error.
    ///
    /// # Panics
    ///
    /// Panics only if a validated [`CoordCircuit`] lacks its required entry body.
    pub fn new(circuit: &CoordCircuit) -> Result<Self, ExecError> {
        if circuit.has_repeats() {
            return Err(ExecError::UnflattenedRepeat);
        }
        validate_template_circuit(circuit).map_err(ExecError::InvalidCircuit)?;
        let coord_to_index = circuit.build_coord_to_index();
        let qubit_count = coord_to_index.len();
        let layout = LayoutCtx {
            coord_to_index: &coord_to_index,
            qubit_count,
        };
        let body = circuit
            .body(circuit.entry_body())
            .expect("entry body always exists on a CoordCircuit");
        let prepared = prepare_ops(body.ops(), &layout, &|id| id)?;
        Ok(Self {
            qubit_count,
            prepared,
        })
    }

    /// Number of distinct qubits (engine width needed).
    #[must_use]
    pub fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    /// Run one shot on `sim` (which must have at least [`qubit_count`] qubits),
    /// consuming classical inputs for `Value`-slot conditional corrections.
    ///
    /// # Errors
    /// See [`ExecError`]; propagates engine errors (e.g. rank overflow).
    ///
    /// [`qubit_count`]: Self::qubit_count
    pub fn run_shot(&self, sim: &mut Simulator) -> Result<ShotRecord, ExecError> {
        let mut record = ShotRecord::default();
        self.prepared.run(sim, &mut record)?;
        Ok(record)
    }
}

/// A single-qubit basis observable at engine index `idx`.
fn basis_pauli(basis: PauliBasis, idx: usize, n: usize) -> PauliString {
    PauliString::single(n, idx, sim_basis(basis).into())
}

/// The engine's spelling of a Bloq IR Pauli (never identity in a `PauliMap`).
pub(crate) fn bloq_pauli(pauli: BloqPauli) -> Pauli {
    match pauli {
        BloqPauli::X => Pauli::X,
        BloqPauli::Y => Pauli::Y,
        BloqPauli::Z => Pauli::Z,
        BloqPauli::I => unreachable!("PauliMap omits identity terms"),
    }
}

/// The engine's spelling of a layout basis. Both enums name the same three
/// axes; only the workspace's copy carries the serde derives the IR needs.
pub(super) fn sim_basis(basis: PauliBasis) -> EnginePauliBasis {
    match basis {
        PauliBasis::X => EnginePauliBasis::X,
        PauliBasis::Y => EnginePauliBasis::Y,
        PauliBasis::Z => EnginePauliBasis::Z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_ir::circuit::{GateType, Pauli};
    use bloq_ir::lowering::NodeTemplateInstanceMergeError;

    /// Push a raw op onto the entry body (for ops without a `CoordCircuit`
    /// helper).
    fn push_op(circuit: &mut CoordCircuit, op: Op) {
        let entry = circuit.entry_body();
        circuit
            .body_mut(entry)
            .expect("entry body")
            .ops_mut()
            .push(op);
    }

    fn invalid_executor(op: Op) -> ExecError {
        let mut circuit = CoordCircuit::new();
        push_op(&mut circuit, op);
        match CircuitExecutor::new(&circuit) {
            Err(error) => error,
            Ok(_) => panic!("malformed circuit was accepted"),
        }
    }

    #[test]
    fn rejects_malformed_operation_arities() {
        let q = IVec2::ZERO;
        assert!(matches!(
            invalid_executor(Op::Gate {
                gate: GateType::CX,
                qubits: vec![q],
            }),
            ExecError::InvalidCircuit(NodeTemplateInstanceMergeError::OddTwoQubitTargetCount {
                gate: GateType::CX,
                targets: 1,
            })
        ));
        assert!(matches!(
            invalid_executor(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![q],
                measurements: vec![],
                flip_probability: 0.0,
            }),
            ExecError::InvalidCircuit(NodeTemplateInstanceMergeError::MeasureOutputCountMismatch {
                qubits: 1,
                measurements: 0,
            })
        ));
        assert!(matches!(
            invalid_executor(Op::MPP {
                products: vec![[(q, Pauli::Z)].into_iter().collect()],
                measurements: vec![],
            }),
            ExecError::InvalidCircuit(
                NodeTemplateInstanceMergeError::MppMeasurementCountMismatch {
                    products: 1,
                    measurements: 0,
                },
            )
        ));
        assert!(matches!(
            invalid_executor(Op::Gate {
                gate: GateType::CX,
                qubits: vec![q, q],
            }),
            ExecError::InvalidCircuit(
                NodeTemplateInstanceMergeError::RepeatedTwoQubitGateTarget {
                    gate: GateType::CX,
                    qubit,
                }
            ) if qubit == q
        ));
        assert!(matches!(
            invalid_executor(Op::Depolarize2 {
                probability: 0.1,
                qubits: vec![q, q],
            }),
            ExecError::InvalidCircuit(
                NodeTemplateInstanceMergeError::RepeatedDepolarize2Target(qubit)
            ) if qubit == q
        ));
    }

    #[test]
    fn missing_record_remap_is_typed() {
        let ops = [Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![IVec2::ZERO],
            measurements: vec![7],
            flip_probability: 0.0,
        }];

        assert!(matches!(
            validate_record_remap(&ops, &FxHashMap::default()),
            Err(ExecError::MissingRecordRemap(7))
        ));
    }

    /// A correction whose control has not been measured yet is an out-of-order
    /// stream, and preparation is where that is caught — once, rather than on
    /// every shot that reaches the correction. The reported id is the *global*
    /// one, after applying the circuit-local record remap.
    #[test]
    fn conditional_pauli_reading_ahead_is_rejected_at_preparation() {
        let q = IVec2::ZERO;
        let coord_to_index: FxHashMap<IVec2, u32> = [(q, 0)].into_iter().collect();
        let layout = LayoutCtx {
            coord_to_index: &coord_to_index,
            qubit_count: 1,
        };
        let correction = Op::ConditionalPauli(vec![ConditionalCorrection {
            pauli: PauliBasis::X,
            control: 4,
            target: q,
        }]);
        let measure = Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![q],
            measurements: vec![4],
            flip_probability: 0.0,
        };

        assert!(matches!(
            prepare_ops(&[correction.clone(), measure.clone()], &layout, &|id| id
                + 10),
            Err(ExecError::MissingRecord(14))
        ));
        // The same two ops in emit order translate cleanly.
        let prepared = prepare_ops(&[measure, correction], &layout, &|id| id + 10)
            .expect("a correction after its control resolves");
        assert_eq!(prepared.record_ids(), [14]);
    }

    /// A measurement-controlled Pauli really fires mid-stream: `H`, `MZ`, then
    /// an `X` conditioned on that record puts the qubit back in `|0⟩`, so the
    /// second `MZ` reads `0` whichever way the first one landed.
    #[test]
    fn conditional_pauli_corrects_the_measured_qubit() {
        let data = IVec2::ZERO;
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [data]).expect("H is valid");
        let m0 = circuit.measure(PauliBasis::Z, [data])[0];
        push_op(
            &mut circuit,
            Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: m0,
                target: data,
            }]),
        );
        let m1 = circuit.measure(PauliBasis::Z, [data])[0];

        let executor = CircuitExecutor::new(&circuit).expect("valid circuit");
        let mut corrected_something = false;
        for seed in 0..16 {
            let mut sim = Simulator::with_seed(executor.qubit_count(), seed);
            let record = executor.run_shot(&mut sim).expect("shot runs");
            assert_eq!(record.get(m1), Some(false), "the correction restores |0>");
            corrected_something |= record.get(m0) == Some(true);
        }
        assert!(
            corrected_something,
            "the uncorrected outcome should vary across seeds"
        );
    }

    #[test]
    fn explicit_noise_ops_lower_to_backend_channels() {
        let q0 = IVec2::ZERO;
        let q1 = IVec2::X;
        let coord_to_index = [(q0, 0), (q1, 1)].into_iter().collect();
        let layout = LayoutCtx {
            coord_to_index: &coord_to_index,
            qubit_count: 2,
        };
        let prepared = prepare_ops(
            &[
                Op::Depolarize1 {
                    probability: 0.3,
                    qubits: vec![q0],
                },
                Op::Depolarize2 {
                    probability: 0.15,
                    qubits: vec![q0, q1],
                },
                Op::PauliError {
                    probability: 0.4,
                    pauli: PauliBasis::Y,
                    qubits: vec![q1],
                },
                Op::Measure {
                    basis: PauliBasis::Z,
                    qubits: vec![q0],
                    measurements: vec![0],
                    flip_probability: 0.25,
                },
            ],
            &layout,
            &|id| id,
        )
        .unwrap();

        let channel_sizes: Vec<_> = prepared
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::RandomPauli {
                    probabilities,
                    alternatives,
                    heralded,
                } => Some((probabilities.len(), alternatives.len(), *heralded)),
                _ => None,
            })
            .collect();
        assert_eq!(
            channel_sizes,
            [(3, 3, false), (15, 15, false), (1, 1, false)]
        );
        assert!(matches!(
            prepared.instructions.last(),
            Some(Instruction::MeasureWithReadoutError {
                probability,
                ..
            }) if *probability == 0.25
        ));
        assert_eq!(prepared.record_ids(), [0]);
    }

    #[test]
    fn pauli_error_changes_state_while_readout_error_only_changes_record() {
        let qubit = IVec2::ZERO;
        let mut circuit = CoordCircuit::new();
        push_op(
            &mut circuit,
            Op::PauliError {
                probability: 1.0,
                pauli: PauliBasis::X,
                qubits: vec![qubit],
            },
        );
        let measurement = circuit.measure(PauliBasis::Z, [qubit])[0];
        let entry = circuit.entry_body();
        let Op::Measure {
            flip_probability, ..
        } = circuit
            .body_mut(entry)
            .expect("entry body")
            .ops_mut()
            .last_mut()
            .expect("measurement")
        else {
            panic!("last op is the measurement just appended");
        };
        *flip_probability = 1.0;

        let executor = CircuitExecutor::new(&circuit).expect("valid noisy circuit");
        let mut simulator = Simulator::with_seed(executor.qubit_count(), 7);
        let record = executor.run_shot(&mut simulator).unwrap();
        assert_eq!(record.get(measurement), Some(false));
        assert_eq!(simulator.peek_z(0).unwrap(), -1.0);
    }

    /// `MPP` measures a genuine joint observable rather than its factors: on a
    /// Bell pair `X⊗X` is a stabilizer, so it reads `+1` and agrees with a
    /// repeat, even though each single-qubit `X` is random.
    #[test]
    fn mpp_reads_a_bell_stabilizer() {
        let (d0, d1) = (IVec2::ZERO, IVec2::new(1, 0));
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [d0]).expect("H is valid");
        circuit
            .do_gate(GateType::CX, [d0, d1])
            .expect("CX is valid");
        let xx = || PauliMap::from_iter([(d0, Pauli::X), (d1, Pauli::X)]);
        let first = circuit.measure_pauli_products([xx()]).expect("XX product")[0];
        let again = circuit.measure_pauli_products([xx()]).expect("XX product")[0];

        let executor = CircuitExecutor::new(&circuit).expect("valid circuit");
        let mut sim = Simulator::with_seed(executor.qubit_count(), 7);
        let record = executor.run_shot(&mut sim).expect("shot runs");
        assert_eq!(record.get(first), Some(false), "XX stabilizes a Bell pair");
        assert_eq!(record.get(again), record.get(first));
    }
}
