//! Symbolic output-correction equations for logical verification.

use std::collections::{BTreeMap, BTreeSet};

use glam::IVec3;
use quizx::fscalar::{FScalar, Zero};
use quizx::graph::{GraphLike, VType};
use quizx::params::{Parity, Var};
use quizx::phase::Phase;
use rustc_hash::FxHashMap;

use crate::zx::{CoeffVec, solve_coeff_combination};
use crate::{
    Action, ActionNode, NodeKind, OutputCorrectionError, Pauli, PauliBasis, RuntimeStabilizerBasis,
    ZXGraph,
};

use super::parity::{phase_constant, phase_value};
use super::{
    MeasurementKey, QuizxGraph, VerifyLogicalError, replay_fill, solve_evolved_correction,
    stabilizer_supports_measurement,
};

/// The readout of every named record in `basis`, by name: a relation row
/// that reads the record's key and no other named key, preferring one that
/// avoids live outputs and then the earliest final layer. Unnamed records may
/// cross it; sampling compensates them. A crossing centre is the OR of its
/// arms, so every condition is linear in the arm components.
pub(super) fn named_readouts(
    basis: &RuntimeStabilizerBasis,
    records: &BTreeMap<String, MeasurementKey>,
) -> Result<BTreeMap<String, crate::Stabilizer>, VerifyLogicalError> {
    let zx = basis.zx_graph();
    let components = records
        .iter()
        .map(|(name, key)| (name.as_str(), key_components(zx, key)))
        .collect::<Vec<_>>();
    let outputs = zx
        .nodes()
        .iter()
        .filter(|node| node.is_output_port(zx))
        .flat_map(|node| {
            let axes: &[Pauli] = if node.role == crate::PortRole::Multiplex {
                &[Pauli::X]
            } else {
                &[Pauli::X, Pauli::Z]
            };
            axes.iter().map(move |&axis| (node.id, axis))
        })
        .collect::<Vec<_>>();
    // Columns by layer, latest first; an edge belongs to its later endpoint.
    let mut layers = BTreeMap::<std::cmp::Reverse<i32>, Vec<(usize, Pauli)>>::new();
    let columns = zx
        .nodes()
        .iter()
        .map(|node| (node.id, node.pos.z))
        .chain(zx.edges().iter().map(|edge| {
            let z = zx.nodes()[edge.n1].pos.z.max(zx.nodes()[edge.n2].pos.z);
            (edge.id, z)
        }));
    for (column, z) in columns {
        layers
            .entry(std::cmp::Reverse(z))
            .or_default()
            .extend([(column, Pauli::X), (column, Pauli::Z)]);
    }
    components
        .iter()
        .map(|(name, target)| {
            let others = components
                .iter()
                .filter(|(other, _)| other != name)
                .flat_map(|(_, components)| components.iter().copied())
                .collect::<Vec<_>>();
            [true, false]
                .into_iter()
                .find_map(|output_free| {
                    let mut constraints = others.clone();
                    if output_free {
                        constraints.extend(outputs.iter().copied());
                    }
                    earliest_reading_row(basis.generators(), &constraints, target, &layers)
                })
                .map(|row| ((*name).to_owned(), zx.materialize_stabilizer(row)))
                .ok_or_else(|| VerifyLogicalError::MeasurementPhaseUnavailable((*name).to_owned()))
        })
        .collect()
}

/// Linear components whose OR is the support a key reads.
fn key_components(zx: &ZXGraph, key: &MeasurementKey) -> Vec<(usize, Pauli)> {
    let (column, pauli) = match *key {
        MeasurementKey::Node { pos, pauli } => {
            (zx.node_at(pos).expect("record node exists").id, pauli)
        }
        MeasurementKey::Edge { src, dst, pauli } => {
            let src = zx.node_at(src).expect("record edge source exists").id;
            let dst = zx.node_at(dst).expect("record edge target exists").id;
            (
                zx.edge_between(src, dst).expect("record edge exists").id,
                pauli,
            )
        }
    };
    pauli
        .iter_xz()
        .flat_map(|axis| zx.readout_components(column, axis))
        .collect()
}

/// A row of the span of `rows` that avoids every constraint component and
/// carries some target component, with no support after the earliest layer
/// at which such a row exists.
fn earliest_reading_row(
    rows: &[crate::PauliString],
    constraints: &[(usize, Pauli)],
    target: &[(usize, Pauli)],
    layers: &BTreeMap<std::cmp::Reverse<i32>, Vec<(usize, Pauli)>>,
) -> Option<crate::PauliString> {
    let reads =
        |row: &crate::PauliString| target.iter().any(|&(column, axis)| row.get(column) & axis);
    let mut kernel = rows.to_vec();
    let restrict = |kernel: &mut Vec<crate::PauliString>, constraints: &[(usize, Pauli)]| {
        let pivots = crate::zx::gaussian_elimination_with_tracking(
            kernel,
            &mut [],
            constraints.iter().copied(),
            |row, &(column, axis)| row.get(column) & axis,
            None,
        );
        kernel.drain(..pivots);
    };
    restrict(&mut kernel, constraints);
    let mut best = kernel.iter().find(|row| reads(row))?.clone();
    for columns in layers.values() {
        restrict(&mut kernel, columns);
        match kernel.iter().find(|row| reads(row)) {
            Some(row) => best = row.clone(),
            None => break,
        }
    }
    Some(best)
}

#[derive(Debug, Clone)]
pub(super) struct InternalCorrection {
    measurement_vars: BTreeMap<MeasurementKey, Parity>,
    sampled: Vec<(MeasurementKey, Var)>,
    initial_basis: RuntimeStabilizerBasis,
    named_vars: BTreeMap<String, Var>,
    records: BTreeMap<String, MeasurementKey>,
}

pub(super) struct ResolvedCorrection {
    pub(super) basis: RuntimeStabilizerBasis,
    pub(super) phases: BTreeMap<MeasurementKey, Parity>,
}

impl InternalCorrection {
    fn parity_columns(
        &self,
        surfaces: &[crate::Stabilizer],
        phases: &BTreeMap<MeasurementKey, Parity>,
    ) -> Vec<CoeffVec> {
        self.sampled_vars()
            .map(|variable| {
                let mut column = CoeffVec::zeros(surfaces.len());
                for (row, surface) in surfaces.iter().enumerate() {
                    column.set_bit(
                        row,
                        phases
                            .iter()
                            .filter(|(_, parity)| parity.iter().any(|value| value == variable))
                            .fold(false, |odd, (key, _)| {
                                odd ^ stabilizer_supports_measurement(surface, key)
                            }),
                    );
                }
                column
            })
            .collect()
    }
    pub(super) fn new(
        zx: &ZXGraph,
        named_phases: &BTreeMap<MeasurementKey, Parity>,
        named_vars: &BTreeMap<String, Var>,
        records: BTreeMap<String, MeasurementKey>,
        next_var: &mut usize,
    ) -> Result<Self, VerifyLogicalError> {
        let keys = measurement_keys(zx);
        let mut measurement_vars = named_phases.clone();
        let mut sampled = Vec::new();
        for key in keys {
            if measurement_vars.contains_key(&key)
                || matches!(&key, MeasurementKey::Node { pos, .. }
                if measurement_vars.keys().any(|named| matches!(named, MeasurementKey::Node { pos: named, .. } if named == pos)))
            {
                continue;
            }
            let variable = fresh_var(next_var)?;
            measurement_vars.insert(key.clone(), Parity::single(variable));
            sampled.push((key, variable));
        }

        let initial_basis = RuntimeStabilizerBasis::from_zx_graph(zx.clone())
            .map_err(|source| VerifyLogicalError::RuntimeBasisInitializationFailed { source })?;
        let anchors = if sampled.is_empty() {
            BTreeMap::new()
        } else {
            named_readouts(&initial_basis, &records)?
        };
        if !sampled.is_empty() && !named_vars.is_empty() {
            // A free native phase can cross a named row. Compensate through the
            // same right-inverse that realizes named outcomes, so the action still
            // reads its supplied outcome after unnamed measurements are sampled.
            for (name, readout) in &anchors {
                let named = named_vars[name];
                for (key, variable) in &sampled {
                    if !stabilizer_supports_measurement(readout, key) {
                        continue;
                    }
                    for (target, parity) in named_phases {
                        if parity.iter().any(|value| value == named) {
                            let target = measurement_vars
                                .get_mut(target)
                                .expect("named phase exists");
                            *target = &*target + &Parity::single(*variable);
                        }
                    }
                }
            }
        }
        Ok(Self {
            measurement_vars,
            sampled,
            initial_basis,
            named_vars: named_vars.clone(),
            records,
        })
    }

    pub(super) fn measurement_vars(&self) -> &BTreeMap<MeasurementKey, Parity> {
        &self.measurement_vars
    }

    pub(super) fn sampled_vars(&self) -> impl Iterator<Item = Var> + '_ {
        self.sampled.iter().map(|(_, variable)| *variable)
    }

    fn resolved_measurement_vars(
        &self,
        selectives: &FxHashMap<IVec3, MeasurementKey>,
        rows: &[crate::Stabilizer],
    ) -> Result<BTreeMap<MeasurementKey, Parity>, VerifyLogicalError> {
        let mut phases = BTreeMap::<MeasurementKey, Parity>::new();
        for (key, parity) in &self.measurement_vars {
            let target = match key {
                MeasurementKey::Node { pos, .. } => selectives.get(pos).unwrap_or(key),
                MeasurementKey::Edge { .. } => key,
            };
            let entry = phases.entry(target.clone()).or_default();
            *entry = &*entry + parity;
        }
        if self.named_vars.is_empty() {
            return Ok(phases);
        }
        let columns = phases
            .keys()
            .map(|key| {
                let mut column = CoeffVec::zeros(rows.len());
                for (row, surface) in rows.iter().enumerate() {
                    column.set_bit(row, stabilizer_supports_measurement(surface, key));
                }
                column
            })
            .collect::<Vec<_>>();
        // A fill can move a later named readout onto different native records.
        // Reparameterize the whole affine fiber, rather than preserving an
        // obsolete pre-fill compensation or losing an anonymous direction.
        let sampled = self.sampled_vars().collect::<BTreeSet<_>>();
        for parity in phases.values_mut() {
            *parity = Parity::new(
                parity
                    .iter()
                    .filter(|variable| !sampled.contains(variable))
                    .collect::<Vec<_>>(),
                phase_constant(parity),
            );
        }
        for (row, (name, &variable)) in self.named_vars.iter().enumerate() {
            let mut rhs = CoeffVec::singleton(row, rows.len());
            for (column, parity) in columns.iter().zip(phases.values()) {
                if parity.iter().any(|value| value == variable) {
                    rhs.xor_assign(column);
                }
            }
            let adjustment = solve_coeff_combination(&columns, &rhs)
                .ok_or_else(|| VerifyLogicalError::MeasurementPhaseUnavailable(name.clone()))?;
            for (index, parity) in phases.values_mut().enumerate() {
                if adjustment.bit(index) {
                    *parity = &*parity + &Parity::single(variable);
                }
            }
        }
        let mut constant = CoeffVec::zeros(rows.len());
        for (column, parity) in columns.iter().zip(phases.values()) {
            if phase_constant(parity) {
                constant.xor_assign(column);
            }
        }
        let adjustment = solve_coeff_combination(&columns, &constant).ok_or_else(|| {
            VerifyLogicalError::MeasurementPhaseUnavailable("affine readout offset".into())
        })?;
        for (index, parity) in phases.values_mut().enumerate() {
            if adjustment.bit(index) {
                *parity = parity.negated();
            }
        }
        let directions = kernel_directions(&columns);
        if directions.len() > self.sampled.len() {
            return Err(VerifyLogicalError::MeasurementPhaseUnavailable(
                "resolved anonymous coordinates".into(),
            ));
        }
        for (direction, (_, variable)) in directions.iter().zip(&self.sampled) {
            for (index, parity) in phases.values_mut().enumerate() {
                if direction.bit(index) {
                    *parity = &*parity + &Parity::single(*variable);
                }
            }
        }
        Ok(phases)
    }

    pub(super) fn zx_graph(&self) -> &ZXGraph {
        self.initial_basis.zx_graph()
    }

    pub(super) fn resolve(
        &self,
        actions: &[ActionNode],
        selectives: &FxHashMap<IVec3, PauliBasis>,
    ) -> Result<ResolvedCorrection, VerifyLogicalError> {
        let mut basis = self.initial_basis.clone();
        let mut keys = FxHashMap::default();
        let mut resolves = actions
            .iter()
            .filter_map(|action| match action.action {
                Action::Resolve { target, .. } => Some((target.z, action.ordinal, target)),
                _ => None,
            })
            .collect::<Vec<_>>();
        resolves.sort_unstable_by_key(|&(z, ordinal, _)| (z, ordinal));
        for (_, _, target) in resolves {
            let (updated, key) = replay_fill(&basis, target, selectives[&target])?;
            basis = updated;
            keys.insert(target, key);
        }
        // A fill moves a selective record onto its chosen native key.
        let records = self
            .records
            .iter()
            .map(|(name, key)| {
                let key = match key {
                    MeasurementKey::Node { pos, .. } => keys.get(pos).unwrap_or(key),
                    MeasurementKey::Edge { .. } => key,
                };
                (name.clone(), key.clone())
            })
            .collect();
        let readouts = named_readouts(&basis, &records)?
            .into_values()
            .collect::<Vec<_>>();
        let phases = self.resolved_measurement_vars(&keys, &readouts)?;
        Ok(ResolvedCorrection { basis, phases })
    }

    pub(super) fn apply(
        &self,
        graph: &mut QuizxGraph,
        outputs: &[IVec3],
        values: &FxHashMap<Var, bool>,
        resolved: &ResolvedCorrection,
        actions: &[ActionNode],
        feedback_values: &FxHashMap<usize, bool>,
    ) -> Result<(), VerifyLogicalError> {
        let ResolvedCorrection {
            basis,
            phases: measurement_vars,
        } = resolved;
        let measured = measurement_vars
            .iter()
            .map(|(key, parity)| {
                (
                    key.clone(),
                    phase_value(parity, |variable| values[&variable]),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let sampled = self.sampled_vars().collect::<BTreeSet<_>>();
        let reference = measurement_vars
            .iter()
            .map(|(key, parity)| {
                let value = phase_value(parity, |variable| {
                    !sampled.contains(&variable) && values[&variable]
                });
                (key.clone(), value)
            })
            .collect::<BTreeMap<_, _>>();

        let Some(reference) =
            self.reference_outcomes(basis, measurement_vars, reference, actions, feedback_values)
        else {
            *graph.scalar_mut() = FScalar::zero();
            return Ok(());
        };
        let (surfaces, correction) = solve_evolved_correction(basis, outputs, |surface| {
            measured
                .keys()
                .any(|key| stabilizer_supports_measurement(surface, key))
        })?;
        let flips = surfaces
            .iter()
            .map(|surface| {
                measured.iter().fold(false, |parity, (key, value)| {
                    parity
                        ^ ((*value ^ reference[key])
                            && stabilizer_supports_measurement(surface, key))
                })
            })
            .collect::<Vec<_>>();
        match correction.evaluate(&flips) {
            Ok(corrections) => insert_output_corrections(graph, outputs, &corrections),
            Err(OutputCorrectionError::Inconsistent) => *graph.scalar_mut() = FScalar::zero(),
            Err(
                error @ (OutputCorrectionError::RhsLengthMismatch { .. }
                | OutputCorrectionError::UnknownOutput { .. }),
            ) => return Err(error.into()),
        }
        Ok(())
    }

    /// Correct within the authored named branch. Its all-zero unnamed outcome
    /// assignment can be impossible: repeated merges must agree with the named
    /// parity. Closed surfaces find a consistent reference with free bits zero.
    fn reference_outcomes(
        &self,
        basis: &RuntimeStabilizerBasis,
        measurement_vars: &BTreeMap<MeasurementKey, Parity>,
        reference: BTreeMap<MeasurementKey, bool>,
        actions: &[ActionNode],
        feedback_values: &FxHashMap<usize, bool>,
    ) -> Option<BTreeMap<MeasurementKey, bool>> {
        let surfaces = basis.closed_stabilizer_surfaces();
        let mut rhs = CoeffVec::zeros(surfaces.len());
        for (index, surface) in surfaces.iter().enumerate() {
            let parity = reference
                .iter()
                .fold(surface.sign, |parity, (key, &value)| {
                    parity ^ (value && stabilizer_supports_measurement(surface, key))
                })
                ^ feedback_sign(surface, basis.zx_graph(), actions, feedback_values);
            rhs.set_bit(index, parity);
        }
        // Each sampled variable includes its compensating native phases.
        // Solving individual phase coordinates would change the named branch.
        let columns = self.parity_columns(&surfaces, measurement_vars);
        let solution = solve_coeff_combination(&columns, &rhs)?;
        Some(self.apply_coordinate(measurement_vars, reference, &solution))
    }

    fn apply_coordinate(
        &self,
        phases: &BTreeMap<MeasurementKey, Parity>,
        mut outcomes: BTreeMap<MeasurementKey, bool>,
        coordinate: &CoeffVec,
    ) -> BTreeMap<MeasurementKey, bool> {
        for (key, value) in &mut outcomes {
            let parity = &phases[key];
            for (index, (_, variable)) in self.sampled.iter().enumerate() {
                *value ^= coordinate.bit(index) && parity.iter().any(|value| value == *variable);
            }
        }
        outcomes
    }
}

fn kernel_directions(columns: &[CoeffVec]) -> Vec<CoeffVec> {
    let mut basis = Vec::<CoeffVec>::new();
    let mut selected = Vec::new();
    let mut kernel = Vec::new();
    for (index, column) in columns.iter().enumerate() {
        if let Some(combination) = solve_coeff_combination(&basis, column) {
            let mut direction = CoeffVec::singleton(index, columns.len());
            for bit in combination.iter_ones() {
                direction.set_bit(selected[bit], true);
            }
            kernel.push(direction);
        } else {
            selected.push(index);
            basis.push(column.clone());
        }
    }
    kernel
}

fn feedback_sign(
    surface: &crate::Stabilizer,
    zx: &ZXGraph,
    actions: &[ActionNode],
    values: &FxHashMap<usize, bool>,
) -> bool {
    actions
        .iter()
        .filter(|action| values.get(&action.ordinal) == Some(&true))
        .fold(false, |odd, action| {
            let Action::Feedback { targets, .. } = &action.action else {
                return odd;
            };
            targets.iter().fold(odd, |odd, target| {
                let Some((column, pauli)) = zx.feedback_column(target) else {
                    return odd;
                };
                odd ^ surface.paulis.get(column).anticommutes(pauli)
            })
        })
}

pub(super) fn fresh_var(next: &mut usize) -> Result<Var, VerifyLogicalError> {
    let variable = Var::try_from(*next).map_err(|_| VerifyLogicalError::TooManySymbols)?;
    *next += 1;
    Ok(variable)
}

fn measurement_keys(zx: &ZXGraph) -> Vec<MeasurementKey> {
    let mut keys = BTreeSet::new();
    for node in zx.nodes() {
        let pauli = match node.kind {
            NodeKind::X => Pauli::Z,
            NodeKind::Y => Pauli::Y,
            NodeKind::Z => Pauli::X,
            NodeKind::Selective(kind) => Pauli::from(kind.pauli_if_false()),
            _ => continue,
        };
        let continues = zx.neighbors(node.id).is_some_and(|neighbors| {
            neighbors.iter().any(|&neighbor| {
                zx.nodes()[neighbor].pos.z > node.pos.z || zx.nodes()[neighbor].is_output_port(zx)
            })
        });
        if !continues {
            keys.insert(MeasurementKey::Node {
                pos: node.pos,
                pauli,
            });
        }
    }
    for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
        let lhs = zx.nodes()[edge.n1];
        let rhs = zx.nodes()[edge.n2];
        if lhs.pos.z != rhs.pos.z
            || !matches!(lhs.kind, NodeKind::X | NodeKind::Z)
            || !matches!(rhs.kind, NodeKind::X | NodeKind::Z)
        {
            continue;
        }
        if (lhs.kind == NodeKind::X) == ((rhs.kind == NodeKind::X) ^ edge.hadamard) {
            keys.insert(MeasurementKey::Edge {
                src: lhs.pos,
                dst: rhs.pos,
                pauli: if lhs.kind == NodeKind::X {
                    Pauli::X
                } else {
                    Pauli::Z
                },
            });
        }
    }
    keys.into_iter().collect()
}

fn insert_output_corrections(
    graph: &mut QuizxGraph,
    outputs: &[IVec3],
    corrections: &[(IVec3, Pauli)],
) {
    for &(position, pauli) in corrections {
        let output = outputs
            .iter()
            .position(|candidate| *candidate == position)
            .expect("correction names a requested output");
        let boundary = graph.outputs()[output];
        let vertex_types: &[VType] = match pauli {
            Pauli::I => &[],
            Pauli::X => &[VType::X],
            Pauli::Y => &[VType::Z, VType::X],
            Pauli::Z => &[VType::Z],
        };
        for &vertex_type in vertex_types {
            let (neighbor, edge_type) = graph
                .incident_edges(boundary)
                .next()
                .expect("an output boundary has one wire");
            graph.remove_edge(boundary, neighbor);
            let correction = graph.add_vertex_with_phase(vertex_type, Phase::from(1_i64));
            graph.add_edge(boundary, correction);
            graph.add_edge_with_type(correction, neighbor, edge_type);
        }
    }
}
