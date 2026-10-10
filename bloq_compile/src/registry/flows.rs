//! Boundary-flow composition with Boolean coefficients instead of joint masks.

use std::collections::{BTreeMap, BTreeSet};

use bloq_circuit::{FlowMarker, PauliMap};
use bloq_ir::NodeDetectorParity;
use bloq_ir::lowering::InstanceMeasurement;
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, BooleanRow, BooleanRowSpace, DECISION_FALSE as ZERO,
    DECISION_TRUE as ONE, DecisionId,
};

use crate::CompileError;
use bloq_utils::Pauli;

pub(super) struct BoundaryFlow {
    pub guard: DecisionId,
    pub start: PauliMap,
    pub end: PauliMap,
    pub measurements: Vec<InstanceMeasurement>,
    pub sign: bool,
    pub center: Option<[i32; 2]>,
    pub marker: FlowMarker,
    pub skip_unmatched: bool,
}

pub(super) struct GuardedCheck {
    pub guard: DecisionId,
    pub parity: NodeDetectorParity,
    pub contributions: BTreeMap<DecisionId, NodeDetectorParity>,
    pub center: Option<[i32; 2]>,
    pub restart: bool,
}

#[derive(Clone)]
struct FlowRow {
    body: BooleanRow,
    /// Canonical X^x Z^z phase, rather than the Hermitian-Pauli phase.
    phase: [DecisionId; 2],
    parity: BTreeMap<InstanceMeasurement, DecisionId>,
    centers: BTreeMap<[i32; 2], DecisionId>,
    discard: DecisionId,
    restart: DecisionId,
    latent: bool,
}

impl FlowRow {
    fn empty(active: DecisionId) -> Self {
        Self {
            body: BooleanRow::new(active),
            phase: [ZERO; 2],
            parity: BTreeMap::new(),
            centers: BTreeMap::new(),
            discard: ZERO,
            restart: ZERO,
            latent: false,
        }
    }

    fn multiply(
        &mut self,
        other: &Self,
        factor: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), CompileError> {
        if factor == ZERO {
            return Ok(());
        }
        diagram.charge(
            other
                .body
                .terms()
                .len()
                .saturating_add(other.parity.len())
                .saturating_add(other.centers.len()),
        )?;
        let low = diagram.apply(BooleanOp::And, factor, other.phase[0])?;
        let high = diagram.apply(BooleanOp::And, factor, other.phase[1])?;
        let carry = diagram.apply(BooleanOp::And, self.phase[0], low)?;
        self.phase[0] = diagram.apply(BooleanOp::Xor, self.phase[0], low)?;
        self.phase[1] = diagram.apply(BooleanOp::Xor, self.phase[1], high)?;
        self.phase[1] = diagram.apply(BooleanOp::Xor, self.phase[1], carry)?;
        for &(column, x) in other
            .body
            .terms()
            .iter()
            .filter(|(column, _)| *column % 2 == 0)
        {
            let x = diagram.apply(BooleanOp::And, factor, x)?;
            let phase = diagram.apply(BooleanOp::And, self.body.get(column + 1), x)?;
            self.phase[1] = diagram.apply(BooleanOp::Xor, self.phase[1], phase)?;
        }
        self.body.xor_scaled(&other.body, factor, diagram)?;
        self.multiply_annotations(other, factor, diagram)
    }

    fn multiply_annotations(
        &mut self,
        other: &Self,
        factor: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), CompileError> {
        for (&measurement, &coefficient) in &other.parity {
            let delta = diagram.apply(BooleanOp::And, factor, coefficient)?;
            let current = self.parity.get(&measurement).copied().unwrap_or(ZERO);
            let next = diagram.apply(BooleanOp::Xor, current, delta)?;
            if next == ZERO {
                self.parity.remove(&measurement);
            } else {
                self.parity.insert(measurement, next);
            }
        }
        let discarded = diagram.apply(BooleanOp::And, factor, other.discard)?;
        self.discard = diagram.apply(BooleanOp::Or, self.discard, discarded)?;
        let restarted = diagram.apply(BooleanOp::And, factor, other.restart)?;
        self.restart = diagram.apply(BooleanOp::Or, self.restart, restarted)?;
        let mut centered = self
            .centers
            .values()
            .copied()
            .try_fold(ZERO, |sum, bit| diagram.apply(BooleanOp::Or, sum, bit))?;
        for (&center, &coefficient) in &other.centers {
            let absent = diagram.negate(centered)?;
            let selected = diagram.apply(BooleanOp::And, factor, coefficient)?;
            let selected = diagram.apply(BooleanOp::And, absent, selected)?;
            let previous = self.centers.get(&center).copied().unwrap_or(ZERO);
            self.centers
                .insert(center, diagram.apply(BooleanOp::Or, previous, selected)?);
            centered = diagram.apply(BooleanOp::Or, centered, selected)?;
        }
        Ok(())
    }

    fn close_matching(
        source: &Self,
        previous: &Self,
        selected: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, CompileError> {
        // Source is the raw Hermitian PauliMap from source(); its low phase bit
        // is the Y parity. Selected includes same_body, so the bodies cancel
        // and their cross phase is exactly the selected source low bit.
        diagram.charge(
            source
                .parity
                .len()
                .saturating_add(source.centers.len())
                .saturating_add(previous.parity.len())
                .saturating_add(previous.centers.len()),
        )?;
        let mut closed = Self::empty(selected);
        let source_low = diagram.apply(BooleanOp::And, selected, source.phase[0])?;
        let source_high = diagram.apply(BooleanOp::And, selected, source.phase[1])?;
        let previous_low = diagram.apply(BooleanOp::And, selected, previous.phase[0])?;
        let previous_high = diagram.apply(BooleanOp::And, selected, previous.phase[1])?;
        let carry = diagram.apply(BooleanOp::And, source_low, previous_low)?;
        closed.phase[0] = diagram.apply(BooleanOp::Xor, source_low, previous_low)?;
        closed.phase[1] = diagram.apply(BooleanOp::Xor, source_high, previous_high)?;
        closed.phase[1] = diagram.apply(BooleanOp::Xor, closed.phase[1], carry)?;
        closed.phase[1] = diagram.apply(BooleanOp::Xor, closed.phase[1], source_low)?;
        closed.multiply_annotations(source, selected, diagram)?;
        closed.multiply_annotations(previous, selected, diagram)?;
        Ok(closed)
    }

    fn same_body(
        &self,
        other: &Self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<DecisionId, CompileError> {
        diagram.charge(
            self.body
                .terms()
                .len()
                .saturating_add(other.body.terms().len()),
        )?;
        if self
            .body
            .terms()
            .iter()
            .chain(other.body.terms())
            .all(|&(_, bit)| bit == ONE)
        {
            return Ok(if self.body.terms() == other.body.terms() {
                ONE
            } else {
                ZERO
            });
        }
        let mut equal = ONE;
        for &(column, _) in self.body.terms().iter().chain(other.body.terms()) {
            let difference = diagram.apply(
                BooleanOp::Xor,
                self.body.get(column),
                other.body.get(column),
            )?;
            let same = diagram.negate(difference)?;
            equal = diagram.apply(BooleanOp::And, equal, same)?;
            if equal == ZERO {
                break;
            }
        }
        Ok(equal)
    }

    fn merge_disjoint(
        &mut self,
        row: Self,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<(), CompileError> {
        let overlap = diagram.apply(BooleanOp::And, self.body.active, row.body.active)?;
        if diagram.constrain(overlap, domain)? != ZERO {
            return Err(super::assembly("duplicate open boundary flow"));
        }
        let mut selected =
            FlowRow::empty(diagram.apply(BooleanOp::Or, self.body.active, row.body.active)?);
        selected.multiply(self, self.body.active, diagram)?;
        selected.multiply(&row, row.body.active, diagram)?;
        // Equal bodies stay fixed rather than carrying their guard.
        let active = selected.body.active;
        selected.body = self.body.clone();
        selected.body.active = active;
        selected.latent = row.latent;
        *self = selected;
        Ok(())
    }
}

fn witness_terms(row: &BooleanRow, first_column: usize) -> &[(usize, DecisionId)] {
    let terms = row.terms();
    &terms[terms.partition_point(|&(column, _)| column < first_column)..]
}

#[derive(Default)]
pub(super) struct GuardedFlowEngine {
    qubits: crate::FxMap<[i32; 2], usize>,
    rows: Vec<FlowRow>,
    constant: crate::FxMap<Vec<(usize, DecisionId)>, Vec<usize>>,
    conditional: Vec<usize>,
    latent: Vec<usize>,
    vacant: usize,
}

impl GuardedFlowEngine {
    fn source_body(&mut self, paulis: &PauliMap, active: DecisionId) -> FlowRow {
        let mut row = FlowRow::empty(active);
        let mut phase = 0u8;
        for (qubit, pauli) in paulis {
            let next = self.qubits.len();
            let index = *self.qubits.entry(qubit.to_array()).or_insert(next);
            for (axis, pauli_axis) in [Pauli::X, Pauli::Z].into_iter().enumerate() {
                if *pauli & pauli_axis {
                    row.body.set(2 * index + axis, ONE);
                }
            }
            phase = (phase + u8::from(*pauli == Pauli::Y)) % 4;
        }
        row.phase = [
            if phase & 1 != 0 { ONE } else { ZERO },
            if phase & 2 != 0 { ONE } else { ZERO },
        ];
        row
    }

    fn source(
        &mut self,
        flow: &BoundaryFlow,
        output: bool,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<FlowRow, CompileError> {
        let paulis = if output { &flow.end } else { &flow.start };
        diagram.charge(
            paulis
                .len()
                .saturating_add(flow.measurements.len())
                .saturating_add(1),
        )?;
        let mut row = self.source_body(paulis, flow.guard);
        if flow.sign {
            row.phase[1] = diagram.negate(row.phase[1])?;
        }
        for &measurement in &flow.measurements {
            if row.parity.remove(&measurement).is_none() {
                row.parity.insert(measurement, ONE);
            }
        }
        if let Some(center) = flow.center {
            row.centers.insert(center, ONE);
        }
        match flow.marker {
            FlowMarker::Detector => {}
            FlowMarker::Discard => row.discard = ONE,
            FlowMarker::Restart => row.restart = ONE,
        }
        Ok(row)
    }

    pub(super) fn append(
        &mut self,
        flows: &[BoundaryFlow],
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<Vec<GuardedCheck>, CompileError> {
        let mut output = Vec::new();
        if flows.is_empty() {
            return Ok(output);
        }
        let mut active = ZERO;
        let mut incoming = Vec::new();
        let mut creators = Vec::new();
        for flow in flows {
            active = diagram.apply(BooleanOp::Or, active, flow.guard)?;
            if flow.start.is_empty() {
                creators.push(self.source(flow, true, diagram)?);
            } else {
                incoming.push((flow, self.source(flow, false, diagram)?));
            }
        }
        let mut exact = active;
        diagram.charge(self.latent.len())?;
        for &index in &self.latent {
            let absent = diagram.negate(self.rows[index].body.active)?;
            exact = diagram.apply(BooleanOp::And, exact, absent)?;
        }
        let mut incoming_guards = crate::FxMap::default();
        diagram.charge(incoming.len())?;
        let mut matches = Vec::with_capacity(incoming.len());
        for (flow, row) in &incoming {
            diagram.charge(row.body.terms().len())?;
            let guard = incoming_guards.entry(row.body.terms()).or_insert(ZERO);
            let duplicate = diagram.apply(BooleanOp::And, row.body.active, *guard)?;
            if diagram.constrain(duplicate, domain)? != ZERO {
                return Err(super::assembly(
                    "incoming flow boundaries are not independent",
                ));
            }
            *guard = diagram.apply(BooleanOp::Or, *guard, row.body.active)?;
            let constant = self
                .constant
                .get(row.body.terms())
                .map(Vec::as_slice)
                .unwrap_or_default();
            diagram.charge(constant.len().saturating_add(self.conditional.len()))?;
            let mut candidates = constant
                .iter()
                .map(|&index| (index, ONE))
                .collect::<Vec<_>>();
            for &index in &self.conditional {
                let same = self.rows[index].same_body(row, diagram)?;
                if same != ZERO {
                    candidates.push((index, same));
                }
            }
            // Preserve multiplication and detector-center precedence.
            candidates.sort_unstable_by_key(|&(index, _)| index);
            if !flow.skip_unmatched {
                let mut present = ZERO;
                for &(index, same) in &candidates {
                    let old = &self.rows[index];
                    if !old.latent {
                        let matched = diagram.apply(BooleanOp::And, old.body.active, same)?;
                        present = diagram.apply(BooleanOp::Or, present, matched)?;
                    }
                }
                let absent = diagram.negate(row.body.active)?;
                let valid = diagram.apply(BooleanOp::Or, absent, present)?;
                exact = diagram.apply(BooleanOp::And, exact, valid)?;
            }
            matches.push(candidates);
        }
        exact = diagram.constrain(exact, domain)?;
        let not_exact = diagram.negate(exact)?;
        let general = diagram.apply(BooleanOp::And, active, not_exact)?;
        let general = diagram.constrain(general, domain)?;
        // Constant consumers only change matched rows. Keep their slots stable
        // until enough are empty to amortize compaction. Relays and span changes
        // retain the full ordered path below, including detector-center order.
        let incremental = general == ZERO
            && self.conditional.is_empty()
            && self.latent.is_empty()
            && incoming.iter().all(|(flow, _)| flow.end.is_empty());
        let mut old = if incremental {
            Vec::new()
        } else {
            std::mem::take(&mut self.rows)
        };
        if exact != ZERO {
            for ((flow, row), candidates) in incoming.iter().zip(matches) {
                let mut combined = FlowRow::empty(ZERO);
                // Consuming matches changes availability, never their bodies.
                for (index, same) in candidates {
                    let previous = if incremental {
                        &mut self.rows[index]
                    } else {
                        &mut old[index]
                    };
                    let selected = diagram.apply(BooleanOp::And, exact, row.body.active)?;
                    let selected = diagram.apply(BooleanOp::And, selected, previous.body.active)?;
                    let selected = diagram.apply(BooleanOp::And, selected, same)?;
                    if selected == ZERO {
                        continue;
                    }
                    let closed = FlowRow::close_matching(row, previous, selected, diagram)?;
                    if combined.body.active == ZERO {
                        // The closed row's coefficients already imply selected.
                        combined = closed;
                    } else {
                        combined.multiply(&closed, selected, diagram)?;
                        combined.body.active =
                            diagram.apply(BooleanOp::Or, combined.body.active, selected)?;
                    }
                    let unselected = diagram.negate(selected)?;
                    previous.body.active =
                        diagram.apply(BooleanOp::And, previous.body.active, unselected)?;
                    if incremental {
                        previous.body.active = diagram.constrain(previous.body.active, domain)?;
                        if previous.body.active == ZERO {
                            diagram.charge(row.body.terms().len())?;
                            self.constant.remove(row.body.terms());
                            *previous = FlowRow::empty(ZERO);
                            self.vacant += 1;
                        }
                    }
                }
                if combined.body.active == ZERO {
                    continue;
                }
                if flow.end.is_empty() {
                    self.checks(combined, diagram, domain, &mut output)?;
                } else {
                    // The start relation owns measurements, markers, centers,
                    // and flow sign. The end adds only its canonical Y phase.
                    diagram.charge(flow.end.len().saturating_add(1))?;
                    let end = self.source_body(&flow.end, flow.guard);
                    combined.multiply(&end, combined.body.active, diagram)?;
                    self.rows.push(combined);
                }
            }
        }
        let mut transitioning = Vec::new();
        for mut row in old {
            if general != ZERO {
                let selected = diagram.apply(BooleanOp::And, row.body.active, general)?;
                if selected != ZERO {
                    let mut selected_row = row.clone();
                    selected_row.body.active = selected;
                    transitioning.push(selected_row);
                    let keep = diagram.negate(general)?;
                    row.body.active = diagram.apply(BooleanOp::And, row.body.active, keep)?;
                }
            }
            if row.body.active != ZERO {
                self.rows.push(row);
            }
        }
        if general != ZERO {
            let mut measured = Vec::new();
            for (flow, mut row) in incoming {
                row.body.active = diagram.apply(BooleanOp::And, row.body.active, general)?;
                if row.body.active == ZERO {
                    continue;
                }
                if !flow.end.is_empty() {
                    return Err(super::assembly(
                        "stabilizer-span transitions require boundary consumer flows",
                    ));
                }
                if flow.skip_unmatched {
                    let mut matched = ZERO;
                    for old in &transitioning {
                        let same = old.same_body(&row, diagram)?;
                        let same = diagram.apply(BooleanOp::And, same, old.body.active)?;
                        matched = diagram.apply(BooleanOp::Or, matched, same)?;
                    }
                    row.body.active = diagram.apply(BooleanOp::And, row.body.active, matched)?;
                }
                if row.body.active != ZERO {
                    measured.push(row);
                }
            }
            self.transition(transitioning, measured, diagram, domain, &mut output)?;
        }
        for creator in creators {
            if creator.body.terms().is_empty() {
                self.checks(creator, diagram, domain, &mut output)?;
            } else if incremental {
                self.insert_constant(creator, diagram, domain)?;
            } else {
                self.rows.push(creator);
            }
        }
        if !incremental {
            self.compact(diagram, domain)?;
        } else if self.vacant > self.rows.len() / 2 {
            self.compact_constant(diagram, domain)?;
        }
        Ok(output)
    }

    fn insert_constant(
        &mut self,
        mut row: FlowRow,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<(), CompileError> {
        row.body.active = diagram.constrain(row.body.active, domain)?;
        if row.body.active == ZERO {
            return Ok(());
        }
        diagram.charge(row.body.terms().len().saturating_add(1))?;
        if let Some(indices) = self.constant.get(row.body.terms()) {
            self.rows[indices[0]].merge_disjoint(row, diagram, domain)?;
        } else {
            self.constant
                .insert(row.body.terms().to_vec(), vec![self.rows.len()]);
            self.rows.push(row);
        }
        Ok(())
    }

    fn matrix(
        &self,
        sources: &[FlowRow],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<BooleanRow>, CompileError> {
        diagram.charge(sources.len())?;
        diagram.charge(sources.iter().fold(0usize, |terms, source| {
            terms.saturating_add(source.body.terms().len().saturating_add(1))
        }))?;
        let width = self.qubits.len() * 2;
        Ok(sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let mut row = source.body.clone();
                row.set(width + index, ONE);
                row
            })
            .collect())
    }

    fn product(
        &self,
        combination: &BooleanRow,
        sources: &[FlowRow],
        first_source: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<FlowRow, CompileError> {
        let width = self.qubits.len() * 2;
        let witnesses = witness_terms(combination, width);
        diagram.charge(witnesses.len())?;
        let split = witnesses.partition_point(|&(column, _)| column < width + first_source);
        let mut result = FlowRow::empty(combination.active);
        // Preserve the requested cyclic source order while skipping every zero
        // witness. Cross-group Pauli products and center precedence are ordered.
        for &(column, coefficient) in witnesses[split..].iter().chain(&witnesses[..split]) {
            let coefficient = diagram.apply(BooleanOp::And, combination.active, coefficient)?;
            result.multiply(&sources[column - width], coefficient, diagram)?;
        }
        Ok(result)
    }

    fn transition(
        &mut self,
        old: Vec<FlowRow>,
        incoming: Vec<FlowRow>,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
        output: &mut Vec<GuardedCheck>,
    ) -> Result<(), CompileError> {
        let width = self.qubits.len() * 2;
        let old_len = old.len();
        let mut sources = old;
        sources.extend(incoming);
        let mut combined = BooleanRowSpace::new(self.matrix(&sources, diagram)?, diagram)?;
        diagram.charge(combined.columns().len())?;
        let columns = combined
            .columns()
            .filter(|&column| column < width)
            .collect::<BTreeSet<_>>();
        for column in columns {
            combined.eliminate_column(column, diagram)?;
        }
        for dependency in combined.into_rows(diagram)? {
            let witnesses = witness_terms(&dependency, width);
            diagram.charge(witnesses.len())?;
            let mut has_old = ZERO;
            let mut has_new = ZERO;
            for &(column, coefficient) in witnesses {
                let present = if column < width + old_len {
                    &mut has_old
                } else {
                    &mut has_new
                };
                *present = diagram.apply(BooleanOp::Or, *present, coefficient)?;
            }
            let both = diagram.apply(BooleanOp::And, has_old, has_new)?;
            let missing = diagram.negate(both)?;
            let invalid = diagram.apply(BooleanOp::And, dependency.active, missing)?;
            if diagram.constrain(invalid, domain)? != ZERO {
                return Err(super::assembly(
                    "boundary flow generators are not independent",
                ));
            }
            // The measured product precedes the old product, exactly as in the
            // ordinary flow engine. Cross-group generators need not commute.
            let row = self.product(&dependency, &sources, old_len, diagram)?;
            self.checks(row, diagram, domain, output)?;
        }
        let old = &sources[..old_len];
        let mut surviving = BooleanRowSpace::new(self.matrix(old, diagram)?, diagram)?;
        for measured in &sources[old_len..] {
            diagram.charge(measured.body.terms().len())?;
            // Distribute the measurement activation over the symplectic sum:
            // active & XOR(x_i z'_i, z_i x'_i). The guard still gates every term.
            let terms = measured
                .body
                .terms()
                .iter()
                .map(|&(column, value)| {
                    diagram
                        .apply(BooleanOp::And, measured.body.active, value)
                        .map(|weight| (column ^ 1, weight))
                })
                .collect::<Result<Vec<_>, _>>()?;
            surviving.eliminate_linear(&terms, diagram)?;
        }
        let surviving = surviving
            .into_rows(diagram)?
            .iter()
            .map(|row| self.product(row, old, 0, diagram))
            .collect::<Result<Vec<_>, _>>()?;
        sources.drain(..old_len);
        let measured = sources.len();
        sources.extend(surviving);
        let mut rows = BooleanRowSpace::new(self.matrix(&sources, diagram)?, diagram)?;
        diagram.charge(rows.columns().len())?;
        let columns = rows
            .columns()
            .filter(|&column| column < width)
            .collect::<BTreeSet<_>>();
        for column in columns {
            let mut pivot = rows.eliminate_column(column, diagram)?;
            let witnesses = witness_terms(&pivot, width + measured);
            diagram.charge(witnesses.len())?;
            let from_old = witnesses
                .iter()
                .try_fold(ZERO, |value, &(_, coefficient)| {
                    diagram.apply(BooleanOp::Or, value, coefficient)
                })?;
            pivot.active = diagram.apply(BooleanOp::And, pivot.active, from_old)?;
            if pivot.active == ZERO {
                continue;
            }
            let mut row = self.product(&pivot, &sources, 0, diagram)?;
            row.latent = true;
            self.rows.push(row);
        }
        Ok(())
    }

    fn compact_constant(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<(), CompileError> {
        // Incremental appends keep one constant, nonlatent row per key.
        // Constrain first so a resource error cannot leave the index half remapped.
        diagram.charge(self.rows.len())?;
        let actives = self
            .rows
            .iter()
            .map(|row| diagram.constrain(row.body.active, domain))
            .collect::<Result<Vec<_>, _>>()?;
        diagram.charge(self.rows.iter().fold(0usize, |work, row| {
            if row.body.terms().is_empty() {
                work
            } else {
                work.saturating_add(row.body.terms().len().saturating_add(1))
            }
        }))?;
        let mut next = 0;
        let mut actives = actives.into_iter();
        let constant = &mut self.constant;
        self.rows.retain_mut(|row| {
            let active = actives.next().expect("one activity per boundary row");
            if active == ZERO {
                if !row.body.terms().is_empty() {
                    constant.remove(row.body.terms());
                }
                return false;
            }
            row.body.active = active;
            let indices = constant
                .get_mut(row.body.terms())
                .expect("incremental boundary row has a constant key");
            debug_assert_eq!(indices.len(), 1);
            indices[0] = next;
            next += 1;
            true
        });
        self.vacant = 0;
        Ok(())
    }

    fn compact(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<(), CompileError> {
        let mut indices = crate::FxMap::default();
        let mut rows: Vec<FlowRow> = Vec::new();
        diagram.charge(self.rows.len())?;
        for mut row in std::mem::take(&mut self.rows) {
            row.body.active = diagram.constrain(row.body.active, domain)?;
            if row.body.active == ZERO {
                continue;
            }
            diagram.charge(row.body.terms().len())?;
            let key = (row.latent, row.body.terms().to_vec());
            if let Some(&index) = indices.get(&key) {
                let previous: &mut FlowRow = &mut rows[index];
                previous.merge_disjoint(row, diagram, domain)?;
            } else {
                indices.insert(key, rows.len());
                rows.push(row);
            }
        }
        self.rows = rows;
        self.constant.clear();
        self.conditional.clear();
        self.latent.clear();
        self.vacant = 0;
        for (index, row) in self.rows.iter().enumerate() {
            diagram.charge(row.body.terms().len().saturating_add(1))?;
            if row.body.terms().iter().all(|&(_, bit)| bit == ONE) {
                self.constant
                    .entry(row.body.terms().to_vec())
                    .or_default()
                    .push(index);
            } else {
                self.conditional.push(index);
            }
            if row.latent {
                self.latent.push(index);
            }
        }
        Ok(())
    }

    fn checks(
        &self,
        row: FlowRow,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
        output: &mut Vec<GuardedCheck>,
    ) -> Result<(), CompileError> {
        let conflict = diagram.apply(BooleanOp::And, row.discard, row.restart)?;
        let conflict = diagram.apply(BooleanOp::And, row.body.active, conflict)?;
        if diagram.constrain(conflict, domain)? != ZERO {
            return Err(super::assembly(
                "stabilizer-span transition mixes discard and restart components",
            ));
        }
        let keep = diagram.negate(row.discard)?;
        let active = diagram.apply(BooleanOp::And, row.body.active, keep)?;
        let active = diagram.constrain(active, domain)?;
        if active == ZERO {
            return Ok(());
        }
        let imaginary = diagram.apply(BooleanOp::And, active, row.phase[0])?;
        if diagram.constrain(imaginary, domain)? != ZERO {
            return Err(super::assembly("non-Hermitian completed boundary flow"));
        }
        let mut centers = row
            .centers
            .iter()
            .map(|(&center, &guard)| (Some(center), guard))
            .collect::<Vec<_>>();
        let centered = row
            .centers
            .values()
            .copied()
            .try_fold(ZERO, |value, guard| {
                diagram.apply(BooleanOp::Or, value, guard)
            })?;
        centers.push((None, diagram.negate(centered)?));
        for (center, guard) in centers {
            let guard = diagram.apply(BooleanOp::And, active, guard)?;
            for restart in [false, true] {
                let kind = if restart {
                    row.restart
                } else {
                    diagram.negate(row.restart)?
                };
                let guard = diagram.apply(BooleanOp::And, guard, kind)?;
                let guard = diagram.constrain(guard, domain)?;
                if guard == ZERO {
                    continue;
                }
                let mut parity = NodeDetectorParity::default();
                let mut contributions = BTreeMap::<DecisionId, NodeDetectorParity>::new();
                for (&measurement, &coefficient) in &row.parity {
                    let coefficient = diagram.apply(BooleanOp::And, guard, coefficient)?;
                    let coefficient = diagram.constrain(coefficient, domain)?;
                    if coefficient == ZERO {
                        continue;
                    }
                    let term = NodeDetectorParity::from_measurements([measurement]);
                    if coefficient == guard {
                        parity.xor_assign(&term);
                    } else {
                        contributions
                            .entry(coefficient)
                            .or_default()
                            .xor_assign(&term);
                    }
                }
                let sign = diagram.apply(BooleanOp::And, guard, row.phase[1])?;
                let sign = diagram.constrain(sign, domain)?;
                if sign == guard {
                    parity = parity.with_sign(true);
                } else if sign != ZERO {
                    contributions
                        .entry(sign)
                        .or_default()
                        .xor_assign(&NodeDetectorParity::default().with_sign(true));
                }
                if !parity.terms().is_empty() || parity.sign() || !contributions.is_empty() {
                    output.push(GuardedCheck {
                        guard,
                        parity,
                        contributions,
                        center,
                        restart,
                    });
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish(
        self,
        diagram: &mut BooleanDecisionDiagram,
        domain: DecisionId,
    ) -> Result<(), CompileError> {
        for row in &self.rows {
            if !row.latent && diagram.constrain(row.body.active, domain)? != ZERO {
                return Err(super::assembly("unterminated boundary flows"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_ir::lowering::TemplateInstanceId;
    use glam::ivec2;

    fn measurement(index: u32) -> InstanceMeasurement {
        InstanceMeasurement {
            instance: TemplateInstanceId(0),
            measurement: index,
        }
    }

    fn canonical(parities: Vec<NodeDetectorParity>) -> Vec<NodeDetectorParity> {
        let mut basis = BTreeMap::<InstanceMeasurement, NodeDetectorParity>::new();
        for mut parity in parities {
            loop {
                let pivot = parity.measurements().next();
                let Some(pivot) = pivot else {
                    break;
                };
                if let Some(row) = basis.get(&pivot) {
                    parity.xor_assign(row);
                } else {
                    basis.insert(pivot, parity);
                    break;
                }
            }
        }
        let pivots = basis.keys().copied().rev().collect::<Vec<_>>();
        for pivot in pivots {
            let row = basis[&pivot].clone();
            for (_, earlier) in basis.range_mut(..pivot) {
                if earlier.measurements().any(|term| term == pivot) {
                    earlier.xor_assign(&row);
                }
            }
        }
        basis.into_values().collect()
    }

    #[test]
    fn exact_closure_matches_ordered_signed_product() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, ZERO, ONE).unwrap();
        let b = diagram.make_node(1, ZERO, ONE).unwrap();
        let not_a = diagram.negate(a).unwrap();
        let mut engine = GuardedFlowEngine::default();
        let bodies = [
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::X)]),
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::Z)]),
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::Y)]),
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::Y), (ivec2(1, 0), Pauli::Y)]),
        ];
        let phases = [[ZERO, ZERO], [ONE, ZERO], [ZERO, ONE], [ONE, ONE], [a, b]];
        for body in &bodies {
            for source_sign in [false, true] {
                for &previous_phase in &phases {
                    for conditional in [false, true] {
                        let mut source = engine.source_body(body, ONE);
                        let mut previous = source.clone();
                        if source_sign {
                            source.phase[1] = diagram.negate(source.phase[1]).unwrap();
                        }
                        previous.phase = previous_phase;
                        source.parity.insert(measurement(0), ONE);
                        source.parity.insert(measurement(1), a);
                        previous.parity.insert(measurement(0), ONE);
                        previous.parity.insert(measurement(2), b);
                        source.centers.insert([1, 0], a);
                        previous.centers.insert([2, 0], ONE);
                        if conditional {
                            for (column, _) in previous.body.terms().to_vec() {
                                previous.body.set(column, a);
                            }
                            previous.body.set(2 * engine.qubits.len(), not_a);
                        }
                        for (discard, restart) in [(ZERO, ZERO), (a, not_a), (a, b)] {
                            source.discard = discard;
                            previous.restart = restart;
                            let selected = source.same_body(&previous, &mut diagram).unwrap();
                            let mut expected = FlowRow::empty(selected);
                            expected.multiply(&source, selected, &mut diagram).unwrap();
                            expected
                                .multiply(&previous, selected, &mut diagram)
                                .unwrap();
                            let actual =
                                FlowRow::close_matching(&source, &previous, selected, &mut diagram)
                                    .unwrap();
                            assert_eq!(actual.body.active, expected.body.active);
                            assert_eq!(actual.body.terms(), expected.body.terms());
                            assert_eq!(actual.phase, expected.phase);
                            assert_eq!(actual.parity, expected.parity);
                            assert_eq!(actual.centers, expected.centers);
                            assert_eq!(actual.discard, expected.discard);
                            assert_eq!(actual.restart, expected.restart);
                            let mut expected_checks = Vec::new();
                            let mut actual_checks = Vec::new();
                            let expected_error = engine
                                .checks(expected, &mut diagram, ONE, &mut expected_checks)
                                .err()
                                .map(|error| error.to_string());
                            let actual_error = engine
                                .checks(actual, &mut diagram, ONE, &mut actual_checks)
                                .err()
                                .map(|error| error.to_string());
                            assert_eq!(actual_error, expected_error);
                            assert_eq!(actual_checks.len(), expected_checks.len());
                            for (actual, expected) in actual_checks.iter().zip(expected_checks) {
                                assert_eq!(actual.guard, expected.guard);
                                assert_eq!(actual.parity, expected.parity);
                                assert_eq!(actual.contributions, expected.contributions);
                                assert_eq!(actual.center, expected.center);
                                assert_eq!(actual.restart, expected.restart);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn constant_compaction_matches_generic_before_relay_and_gauge() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, ZERO, ONE).unwrap();
        let not_a = diagram.negate(a).unwrap();
        let pair =
            |pauli| PauliMap::from_unique_entries([(ivec2(20, 0), pauli), (ivec2(21, 0), pauli)]);
        let mut fast = GuardedFlowEngine::default();
        let mut bodies = (0..6)
            .map(|index| PauliMap::from_unique_entries([(ivec2(index, 0), Pauli::X)]))
            .collect::<Vec<_>>();
        bodies.extend([
            pair(Pauli::X),
            pair(Pauli::Z),
            PauliMap::from_unique_entries([(ivec2(30, 0), Pauli::X)]),
        ]);
        for (index, body) in bodies.iter().enumerate() {
            let mut row = fast.source_body(body, if index == 8 { not_a } else { ONE });
            row.parity.insert(measurement(index as u32), ONE);
            fast.insert_constant(row, &mut diagram, ONE).unwrap();
        }
        for index in 0..6 {
            let key = fast.rows[index].body.terms().to_vec();
            fast.constant.remove(&key);
            fast.rows[index] = FlowRow::empty(ZERO);
            fast.vacant += 1;
        }
        let mut generic = GuardedFlowEngine {
            qubits: fast.qubits.clone(),
            rows: fast.rows.clone(),
            constant: fast.constant.clone(),
            conditional: fast.conditional.clone(),
            latent: fast.latent.clone(),
            vacant: fast.vacant,
        };
        let mut generic_diagram = diagram.clone();
        fast.compact_constant(&mut diagram, a).unwrap();
        generic.compact(&mut generic_diagram, a).unwrap();
        assert_eq!(fast.constant, generic.constant);
        assert_eq!(fast.vacant, generic.vacant);
        assert_eq!(fast.rows.len(), generic.rows.len());
        for (fast, generic) in fast.rows.iter().zip(&generic.rows) {
            assert_eq!(fast.body.active, generic.body.active);
            assert_eq!(fast.body.terms(), generic.body.terms());
            assert_eq!(fast.phase, generic.phase);
            assert_eq!(fast.parity, generic.parity);
            assert_eq!(fast.centers, generic.centers);
            assert_eq!(
                (fast.discard, fast.restart),
                (generic.discard, generic.restart)
            );
        }
        let flow = |start, end, record| BoundaryFlow {
            guard: a,
            start,
            end,
            measurements: vec![measurement(record)],
            sign: false,
            center: Some([record as i32, 0]),
            marker: FlowMarker::Detector,
            skip_unmatched: false,
        };
        let groups = [
            vec![flow(pair(Pauli::X), pair(Pauli::X), 40)],
            vec![
                flow(pair(Pauli::Y), PauliMap::empty(), 41),
                flow(PauliMap::empty(), pair(Pauli::Y), 42),
            ],
        ];
        for group in &groups {
            let fast_checks = fast.append(group, &mut diagram, a).unwrap();
            let generic_checks = generic.append(group, &mut generic_diagram, a).unwrap();
            assert_eq!(fast_checks.len(), generic_checks.len());
            for (fast, generic) in fast_checks.iter().zip(&generic_checks) {
                assert_eq!(fast.guard, generic.guard);
                assert_eq!(fast.parity, generic.parity);
                assert_eq!(fast.contributions, generic.contributions);
                assert_eq!(fast.center, generic.center);
                assert_eq!(fast.restart, generic.restart);
            }
        }
    }

    #[test]
    fn constant_boundary_appends_scale_with_changed_rows() {
        let run = |count: u32| {
            let mut diagram = BooleanDecisionDiagram::default();
            let a = diagram.make_node(0, ZERO, ONE).unwrap();
            let not_a = diagram.negate(a).unwrap();
            let domain = diagram.make_node(1, ZERO, ONE).unwrap();
            let mut engine = GuardedFlowEngine::default();
            let flow = |index, guard, create, record, sign| {
                let body = PauliMap::from_unique_entries([(ivec2(index as i32, 0), Pauli::Y)]);
                BoundaryFlow {
                    guard,
                    start: if create {
                        PauliMap::empty()
                    } else {
                        body.clone()
                    },
                    end: if create { body } else { PauliMap::empty() },
                    measurements: vec![measurement(record)],
                    sign,
                    center: Some([record as i32, 0]),
                    marker: FlowMarker::Detector,
                    skip_unmatched: false,
                }
            };
            // Disjoint creators merge under one cached body. Each first
            // consumer leaves a guarded remainder for the second pass.
            for index in 0..count {
                for (guard, offset) in [(a, 0), (not_a, count)] {
                    engine
                        .append(
                            &[flow(index, guard, true, index + offset, offset != 0)],
                            &mut diagram,
                            domain,
                        )
                        .unwrap();
                }
            }
            for (guard, offset) in [(a, 0), (not_a, count)] {
                for index in 0..count {
                    let record = 2 * count + index + offset;
                    let checks = engine
                        .append(
                            &[flow(index, guard, false, record, false)],
                            &mut diagram,
                            domain,
                        )
                        .unwrap();
                    assert_eq!(checks.len(), 1);
                    assert_eq!(checks[0].guard, guard);
                    assert_eq!(checks[0].center, Some([record as i32, 0]));
                    assert_eq!(
                        checks[0].parity,
                        NodeDetectorParity::from_measurements([
                            measurement(index + offset),
                            measurement(record),
                        ])
                        .with_sign(offset != 0)
                    );
                    assert!(checks[0].contributions.is_empty());
                }
            }
            engine.finish(&mut diagram, domain).unwrap();
            diagram.steps()
        };
        let small = run(32);
        let large = run(128);
        assert!(large < 5 * small, "append work: 32={small}, 128={large}");
    }

    #[test]
    fn sparse_witness_products_preserve_signed_order_and_center_precedence() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, ZERO, ONE).unwrap();
        let b = diagram.make_node(1, ZERO, ONE).unwrap();
        let mut engine = GuardedFlowEngine::default();
        let sources = (0..100)
            .map(|index| {
                engine
                    .source(
                        &BoundaryFlow {
                            guard: ONE,
                            start: PauliMap::empty(),
                            end: PauliMap::from_unique_entries([(
                                ivec2(0, 0),
                                [Pauli::X, Pauli::Z, Pauli::Y][index as usize % 3],
                            )]),
                            measurements: vec![measurement(index)],
                            sign: index % 2 == 0,
                            center: Some([index as i32, 0]),
                            marker: FlowMarker::Detector,
                            skip_unmatched: false,
                        },
                        true,
                        &mut diagram,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let width = engine.qubits.len() * 2;
        let combination = BooleanRow::from_terms(
            ONE,
            [(0, a), (width + 1, ONE), (width + 47, a), (width + 95, b)],
        );
        for first in [0, 50, sources.len()] {
            let actual = engine
                .product(&combination, &sources, first, &mut diagram)
                .unwrap();
            let mut expected = FlowRow::empty(ONE);
            // The former replay visited every source, including zero witnesses.
            for index in (first..sources.len()).chain(0..first) {
                let factor = diagram
                    .apply(
                        BooleanOp::And,
                        combination.active,
                        combination.get(width + index),
                    )
                    .unwrap();
                expected
                    .multiply(&sources[index], factor, &mut diagram)
                    .unwrap();
            }
            assert_eq!(actual.body.active, expected.body.active);
            assert_eq!(actual.body.terms(), expected.body.terms());
            assert_eq!(actual.phase, expected.phase);
            assert_eq!(actual.parity, expected.parity);
            assert_eq!(actual.centers, expected.centers);
            assert_eq!(
                (actual.discard, actual.restart),
                (expected.discard, expected.restart)
            );
        }
    }

    #[test]
    fn exact_joins_preserve_conditional_bodies_and_reachable_overlap() {
        for axis in [Pauli::X, Pauli::Y] {
            let mut diagram = BooleanDecisionDiagram::default();
            let a = diagram.make_node(0, ZERO, ONE).unwrap();
            let b = diagram.make_node(1, ZERO, ONE).unwrap();
            let overlap = diagram.apply(BooleanOp::And, a, b).unwrap();
            let domain = diagram.negate(overlap).unwrap();
            let x = PauliMap::from_unique_entries([(ivec2(0, 0), axis)]);
            let z = PauliMap::from_unique_entries([(ivec2(1, 0), Pauli::Z)]);
            let flow = |guard, start: &PauliMap, end: &PauliMap, record, sign| BoundaryFlow {
                guard,
                start: start.clone(),
                end: end.clone(),
                measurements: vec![measurement(record)],
                sign,
                center: Some([record as i32, 0]),
                marker: FlowMarker::Detector,
                skip_unmatched: false,
            };
            let empty = PauliMap::empty();
            let mut engine = GuardedFlowEngine::default();
            engine
                .append(
                    &[
                        flow(ONE, &empty, &x, 0, true),
                        flow(ONE, &empty, &z, 1, false),
                    ],
                    &mut diagram,
                    domain,
                )
                .unwrap();
            engine
                .append(
                    &[flow(a, &x, &x, 2, true), flow(b, &z, &z, 3, false)],
                    &mut diagram,
                    domain,
                )
                .unwrap();
            // Relays leave conditional bodies interleaved with constant bodies.
            // The two X consumers overlap only outside the reachable domain.
            let consumers = [
                flow(a, &x, &empty, 4, false),
                flow(b, &x, &empty, 5, true),
                flow(ONE, &z, &empty, 6, true),
            ];
            let mut checks = engine.append(&consumers, &mut diagram, domain).unwrap();
            let mut remainder = flow(ONE, &x, &empty, 7, false);
            remainder.skip_unmatched = true;
            checks.extend(engine.append(&[remainder], &mut diagram, domain).unwrap());
            engine.finish(&mut diagram, domain).unwrap();
            for mask in 0..3 {
                let enabled = |root| diagram.evaluate(root, |index| mask & (1 << index) != 0);
                let actual = checks
                    .iter()
                    .filter(|check| enabled(check.guard))
                    .map(|check| {
                        let mut parity = check.parity.clone();
                        for (&guard, contribution) in &check.contributions {
                            if enabled(guard) {
                                parity.xor_assign(contribution);
                            }
                        }
                        assert!(!check.restart);
                        (parity, check.center)
                    })
                    .collect::<Vec<_>>();
                let (x_records, x_sign, x_center) = match mask {
                    0 => (vec![0, 7], true, 7),
                    1 => (vec![0, 2, 4], false, 4),
                    2 => (vec![0, 5], false, 5),
                    _ => unreachable!(),
                };
                let z_records = if mask == 2 { vec![1, 3, 6] } else { vec![1, 6] };
                let expected = [
                    (
                        NodeDetectorParity::from_measurements(
                            x_records.into_iter().map(measurement),
                        )
                        .with_sign(x_sign),
                        Some([x_center, 0]),
                    ),
                    (
                        NodeDetectorParity::from_measurements(
                            z_records.into_iter().map(measurement),
                        )
                        .with_sign(true),
                        Some([6, 0]),
                    ),
                ];
                assert_eq!(actual.len(), expected.len(), "mask {mask}");
                for check in expected {
                    assert!(actual.contains(&check), "mask {mask}: {actual:?}");
                }
            }
            // The same consumer overlap must be rejected when it is reachable.
            assert!(matches!(
                GuardedFlowEngine::default().append(&consumers, &mut diagram, ONE),
                Err(CompileError::BranchAssembly { reason })
                    if reason == "incoming flow boundaries are not independent"
            ));
        }
    }

    #[test]
    fn guarded_gauge_transitions_match_signed_concrete_flow_spaces() {
        let pair =
            |pauli| PauliMap::from_unique_entries([(ivec2(0, 0), pauli), (ivec2(1, 0), pauli)]);
        let flow =
            |start, end, record| bloq_circuit::Flow::new(start, end).with_measurements([record]);
        let mut diagram = BooleanDecisionDiagram::default();
        let controls = (0..3)
            .map(|index| diagram.make_node(index, ZERO, ONE).unwrap())
            .collect::<Vec<_>>();
        let mut groups = vec![vec![
            (ONE, flow(PauliMap::empty(), pair(Pauli::X), 0)),
            (ONE, flow(PauliMap::empty(), pair(Pauli::Z), 1)),
        ]];
        for (index, &control) in controls.iter().enumerate() {
            let mut group = Vec::new();
            for (enabled, pauli, offset) in [
                (control, Pauli::Y, 0),
                (diagram.negate(control).unwrap(), Pauli::X, 2),
            ] {
                let record = 2 + 4 * index as u32 + offset;
                group.push((enabled, flow(pair(pauli), PauliMap::empty(), record)));
                group.push((enabled, flow(PauliMap::empty(), pair(pauli), record)));
            }
            groups.push(group);
        }
        groups.push(vec![
            (ONE, flow(pair(Pauli::X), PauliMap::empty(), 40)),
            (ONE, flow(pair(Pauli::Z), PauliMap::empty(), 41)),
        ]);
        let mut symbolic = GuardedFlowEngine::default();
        let mut checks = Vec::new();
        for group in &groups {
            let flows = group
                .iter()
                .map(|(guard, flow)| BoundaryFlow {
                    guard: *guard,
                    start: flow.start.clone(),
                    end: flow.end.clone(),
                    measurements: flow.measurements.iter().copied().map(measurement).collect(),
                    sign: flow.sign,
                    center: None,
                    marker: flow.marker,
                    skip_unmatched: false,
                })
                .collect::<Vec<_>>();
            checks.extend(symbolic.append(&flows, &mut diagram, ONE).unwrap());
        }
        symbolic.finish(&mut diagram, ONE).unwrap();
        for mask in 0..8 {
            let enabled = |root| diagram.evaluate(root, |index| mask & (1 << index) != 0);
            let selected = groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .filter(|(guard, _)| enabled(*guard))
                        .map(|(_, flow)| flow.clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut ordinary = bloq_circuit::FlowEngine::new();
            for group in &selected {
                ordinary
                    .append_group_allowing_stabilizer_transition(
                        &[bloq_circuit::OffsetFlows::new(group, glam::IVec2::ZERO)],
                        |_, &index| measurement(index),
                    )
                    .unwrap();
            }
            let expected = ordinary
                .finish()
                .unwrap()
                .into_iter()
                .map(|flow| {
                    NodeDetectorParity::from_measurements(flow.measurements).with_sign(flow.sign)
                })
                .collect();
            let actual = checks
                .iter()
                .filter(|check| enabled(check.guard))
                .map(|check| {
                    let mut parity = check.parity.clone();
                    for (&guard, contribution) in &check.contributions {
                        if enabled(guard) {
                            parity.xor_assign(contribution);
                        }
                    }
                    parity
                })
                .collect();
            assert_eq!(canonical(actual), canonical(expected), "mask {mask}");
        }
    }
}
