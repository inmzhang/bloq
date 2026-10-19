//! Runtime-selectable stabilizer bases and derived readout surfaces.

use std::collections::HashMap;

use glam::IVec3;
use thiserror::Error;

use super::graph::insert_selective_fill;
use super::output_correction::{OutputCorrectionRow, pivot_output_correction_rows};
use super::stabilizer::{
    CoeffVec, DeadlineProjection, SelectiveConstraint, StabilizerBasis,
    collect_selective_constraints, deadline_projection_with_initial_coeffs,
    normalize_selective_support, reduce_to_basis, solve_coeff_combination,
};
use super::{
    NodeKind, Stabilizer, StabilizerError, StabilizerGenerator, StabilizerGenerators,
    StabilizerRowKind, ZXError, ZXGraph,
};
use crate::{BlockGraph, ModuleCertificationLimits, Pauli, PauliBasis, PauliString, SelectiveKind};

/// An error from deriving or updating a [`RuntimeStabilizerBasis`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum RuntimeBasisError {
    /// A selective site is absent from the live basis.
    #[error("selective node at {0} is not part of the live basis")]
    UnknownSelective(IVec3),
    /// A selective site occurs twice in one assignment.
    #[error("selective node at {0} appears more than once in one assignment")]
    DuplicateSelective(IVec3),
    /// A chosen basis is not allowed at the selective site.
    #[error("basis {chosen} is not allowed at selective {pos} of kind {kind}")]
    InvalidSelectiveBasis {
        /// Selective position.
        pos: IVec3,
        /// Selective kind.
        kind: SelectiveKind,
        /// Chosen basis.
        chosen: PauliBasis,
    },
    /// A measurement variable is absent from the live graph.
    #[error("measurement variable {0:?} is not present in the live ZX graph")]
    UnknownMeasurement(String),
    /// Filling non-Clifford computation is unsupported.
    #[error("selective fill is only supported for Clifford-only computation")]
    NonCliffordFillUnsupported,
    /// An output is absent from the live graph.
    #[error("output port {0} is not present in the live ZX graph")]
    UnknownOutput(IVec3),
    /// No closed frame surface exists for an output Pauli.
    #[error("no closed {pauli} frame surface for output {output}")]
    OutputSurfaceUnavailable {
        /// Output position.
        output: IVec3,
        /// Requested Pauli basis.
        pauli: PauliBasis,
    },
    /// ZX graph update failed.
    #[error("{0}")]
    ZX(#[from] ZXError),
    /// Stabilizer-basis derivation failed.
    #[error("{0}")]
    Stabilizer(#[from] StabilizerError),
}

/// The live stabilizer basis of a ZX graph, tracking the generators and the
/// selective nodes whose runtime basis choice can still update it.
///
/// This is the state that dynamic (selective) fills mutate as each selective is
/// resolved, and the source of derived measurement and output surfaces.
#[derive(Debug, Clone)]
pub struct RuntimeStabilizerBasis {
    zx_graph: ZXGraph,
    // Filling treats native T states as ports, but does not turn prepared
    // resources into arbitrary inputs of the authored map.
    input_columns: Vec<usize>,
    prepared_columns: Vec<usize>,
    basis: StabilizerBasis,
    /// Readable-value provenance of every live row, expressed in the original
    /// generator ordinals. Internal and fixing directions carry the zero vector.
    /// Row operations update these coefficients with Pauli support, so
    /// readability never depends on a mutable row-role label.
    readout_coeffs: Vec<CoeffVec>,
    measurement_ordinals: HashMap<String, usize>,
    selective_kinds: HashMap<IVec3, SelectiveKind>,
}

/// A materialized stabilizer surface with readable-value provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedSurface {
    /// Original public generator ordinals whose readable values XOR to this
    /// surface. Internal and fixing directions carry no readable ordinal.
    /// A frozen [`super::ReadoutPlan`] declares whether these are public-row
    /// coordinates or earlier corrected source outcomes. In source coordinates,
    /// the name being defined is supplied by the recipe's owner.
    pub readout_ordinals: Vec<usize>,
    /// The surface materialized as a [`Stabilizer`].
    pub stabilizer: Stabilizer,
}

impl RuntimeStabilizerBasis {
    /// The live ZX graph backing this basis.
    pub fn zx_graph(&self) -> &ZXGraph {
        &self.zx_graph
    }

    /// The Pauli-string generators of this basis, in row order.
    pub fn generators(&self) -> &[PauliString] {
        &self.basis.rows
    }

    pub(crate) fn output_free_readout_failure(
        &self,
        name: &str,
        ordinal: usize,
        readout_width: usize,
        outputs: &[IVec3],
    ) -> Result<Option<IVec3>, RuntimeBasisError> {
        let mut components = Vec::new();
        for &output in outputs {
            let node = self
                .zx_graph
                .node_at(output)
                .filter(|node| node.is_output_port(&self.zx_graph))
                .ok_or(RuntimeBasisError::UnknownOutput(output))?;
            components.push((output, node.id, Pauli::X));
            if node.role != crate::PortRole::Multiplex {
                components.push((output, node.id, Pauli::Z));
            }
        }

        let width = readout_width + components.len();
        let vectors = self
            .basis
            .rows
            .iter()
            .zip(&self.readout_coeffs)
            .map(|(row, readout)| {
                let mut projected = CoeffVec::zeros(width);
                for bit in readout.iter_ones() {
                    projected.set_bit(bit, true);
                }
                for (offset, (_, col, axis)) in components.iter().enumerate() {
                    projected.set_bit(readout_width + offset, row.get(*col) & *axis);
                }
                projected
            })
            .collect::<Vec<_>>();
        if solve_coeff_combination(&vectors, &CoeffVec::singleton(ordinal, width)).is_some() {
            return Ok(None);
        }

        let witness = solve_coeff_combination(
            &self.readout_coeffs,
            &CoeffVec::singleton(ordinal, readout_width),
        )
        .ok_or_else(|| RuntimeBasisError::UnknownMeasurement(name.into()))?;
        let mut row = PauliString::new(self.zx_graph.total_ids());
        for index in witness.iter_ones() {
            row ^= &self.basis.rows[index];
        }
        for (output, col, axis) in components {
            if row.get(col) & axis {
                return Ok(Some(output));
            }
        }
        unreachable!("an output-free witness would satisfy the projected solve")
    }

    /// Demotes every [`T`](NodeKind::T) node to a plain [`Port`](NodeKind::Port).
    /// Used when selective filling treats prepared T states as open inputs.
    pub fn with_t_nodes_as_ports(mut self) -> Self {
        self.zx_graph.demote_t_nodes_to_ports();
        self
    }

    fn selective_port_map(
        &self,
        fills: &[(IVec3, PauliBasis)],
    ) -> Result<HashMap<IVec3, NodeKind>, RuntimeBasisError> {
        let mut port_map = HashMap::with_capacity(fills.len());
        for &(pos, chosen) in fills {
            let kind = self
                .selective_kinds
                .get(&pos)
                .copied()
                .ok_or(RuntimeBasisError::UnknownSelective(pos))?;
            insert_selective_fill(&mut port_map, pos, kind, chosen)?;
        }
        Ok(port_map)
    }

    /// Consumes this live basis and materializes its current rows against its
    /// current ZX graph.
    pub fn into_generators(self) -> StabilizerGenerators {
        let Self {
            zx_graph,
            mut basis,
            ..
        } = self;
        // Selective pivots XOR rows; cross centers must follow the resulting
        // edge support before those rows become physical observable requests.
        zx_graph.reconstruct_cross_center(&mut basis.rows);
        let auxiliary_rows = basis.rows.split_off(basis.public_len);
        basis.kinds.truncate(basis.public_len);
        let row_phases = zx_graph.stabilizer_row_phases(&basis.rows);
        let generators = basis
            .rows
            .into_iter()
            .zip(basis.kinds)
            .zip(row_phases)
            .map(|((row, kind), row_phase)| {
                StabilizerGenerator::new(
                    zx_graph.pauli_string_to_stabilizer_with_row_phase(row, row_phase),
                    kind,
                )
            })
            .collect();
        StabilizerGenerators::from_parts(zx_graph, generators, auxiliary_rows)
    }

    /// Builds the runtime basis by converting `graph` to a ZX graph first.
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeBasisError`] if conversion or basis derivation fails.
    pub fn from_block_graph(graph: &BlockGraph) -> Result<Self, RuntimeBasisError> {
        let zx_graph = ZXGraph::try_from(graph)?;
        Self::from_zx_graph(zx_graph)
    }

    /// Builds the runtime basis from an existing ZX graph.
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeBasisError`] if the stabilizer basis cannot be derived.
    pub fn from_zx_graph(zx_graph: ZXGraph) -> Result<Self, RuntimeBasisError> {
        Self::from_zx_graph_with_limits(zx_graph, ModuleCertificationLimits::DEFAULT)
    }

    /// Builds a runtime basis under explicit stabilizer-search limits.
    ///
    /// # Errors
    ///
    /// Returns a basis-derivation or resource-limit error.
    pub fn from_zx_graph_with_limits(
        zx_graph: ZXGraph,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, RuntimeBasisError> {
        match zx_graph.stabilizers_with_limits(limits) {
            Ok(stabilizers) => return Ok(Self::from_generators(&stabilizers)),
            Err(
                StabilizerError::SelectiveSupportUnsatisfiable { .. }
                | StabilizerError::MeasurementSurfaceUnavailable { .. },
            ) => {}
            Err(error) => return Err(error.into()),
        }

        // Runtime filling can still use a private all-logical basis when no
        // uniform readable presentation exists before the fills.
        let raw_external = zx_graph.to_external_generator_table();
        let constraints = collect_selective_constraints(&zx_graph);
        let raw_basis = reduce_to_basis(&raw_external, zx_graph.total_ids);
        let generators = reduce_to_basis(
            &normalize_selective_support(&raw_basis, &constraints)?,
            zx_graph.total_ids,
        );
        let basis = StabilizerBasis::all_logical(generators);
        let selective_kinds = selective_kinds_of(&zx_graph);
        let readout_coeffs = initial_readout_coeffs(&basis.kinds, basis.public_len);
        let (input_columns, prepared_columns) = authored_boundary_columns(&zx_graph);
        Ok(Self {
            input_columns,
            prepared_columns,
            zx_graph,
            basis,
            readout_coeffs,
            measurement_ordinals: HashMap::new(),
            selective_kinds,
        })
    }

    /// Builds the runtime basis directly from already-canonicalized
    /// [`StabilizerGenerators`], reusing their rows and row kinds verbatim
    /// instead of re-deriving the canonical stabilizer table from the ZX graph.
    ///
    /// The *n*th runtime generator is the *n*th input generator, so the row
    /// ordinals downstream code keys on stay identity-stable. This exists so a
    /// caller that already holds the graph's `StabilizerGenerators` avoids
    /// paying for another canonicalization pass, which is otherwise what
    /// [`from_zx_graph`](Self::from_zx_graph) would re-run.
    pub fn from_generators(stabilizers: &StabilizerGenerators) -> Self {
        let zx_graph = stabilizers.zx_graph.clone();
        let rows = stabilizers
            .generators
            .iter()
            .map(|generator| generator.stabilizer.paulis.clone())
            .chain(stabilizers.auxiliary_rows().iter().cloned())
            .collect();
        let kinds: Vec<_> = stabilizers
            .generators
            .iter()
            .map(|generator| generator.kind.clone())
            .chain(
                stabilizers
                    .auxiliary_rows()
                    .iter()
                    .map(|_| StabilizerRowKind::Logical),
            )
            .collect();
        let selective_kinds = selective_kinds_of(&zx_graph);
        let measurement_ordinals = stabilizers
            .generators
            .iter()
            .enumerate()
            .filter_map(|(ordinal, generator)| {
                generator
                    .measurement_name()
                    .map(|name| (name.to_owned(), ordinal))
            })
            .collect();
        let public_len = stabilizers.generators.len();
        let readout_coeffs = initial_readout_coeffs(&kinds, public_len);
        let (input_columns, prepared_columns) = authored_boundary_columns(&zx_graph);
        Self {
            input_columns,
            prepared_columns,
            zx_graph,
            basis: StabilizerBasis::with_public_len(rows, kinds, public_len),
            readout_coeffs,
            measurement_ordinals,
            selective_kinds,
        }
    }

    /// Builds linear X/Z source-outcome coordinates on the complete signed basis.
    ///
    /// Public row roles do not define these coordinates: a named anchor can also
    /// occur in a logical or private completion row. This basis is used to freeze
    /// causal readouts and to remove those same named components from frames.
    /// Crossing-node and Y readouts require the ordinary readout convention.
    ///
    /// # Errors
    ///
    /// Returns an error if a named source measurement lacks a concrete readable row.
    pub fn for_source_readouts(
        stabilizers: &StabilizerGenerators,
    ) -> Result<Self, RuntimeBasisError> {
        let mut basis = Self::from_generators(stabilizers).with_t_nodes_as_ports();
        let width = stabilizers.generators.len();
        basis.readout_coeffs.fill(CoeffVec::zeros(width));
        for node in basis.zx_graph.action_graph().ordered_nodes() {
            let crate::Action::Measure { name, target } = &node.action else {
                continue;
            };
            let axis = match node.measurement {
                Some(crate::MeasurementObservable::Concrete(PauliBasis::X)) => Pauli::X,
                Some(crate::MeasurementObservable::Concrete(PauliBasis::Z)) => Pauli::Z,
                _ => {
                    return Err(StabilizerError::MeasurementSurfaceUnavailable {
                        mvar: name.clone(),
                    }
                    .into());
                }
            };
            let col = basis
                .zx_graph
                .measurement_column(target)
                .ok_or_else(|| RuntimeBasisError::UnknownMeasurement(name.clone()))?;
            if basis.zx_graph.nodes().get(col).is_some_and(|node| {
                matches!(node.kind, NodeKind::X | NodeKind::Z) && node.kind.cross_pauli() == axis
            }) {
                return Err(
                    StabilizerError::MeasurementSurfaceUnavailable { mvar: name.clone() }.into(),
                );
            }
            let ordinal = basis.measurement_ordinals[name];
            for (row, coeff) in basis.basis.rows.iter().zip(&mut basis.readout_coeffs) {
                coeff.set_bit(ordinal, row.get(col) & axis);
            }
        }
        Ok(basis)
    }

    /// The composed relation restricted to zero source named-outcome coordinates.
    /// Closed named readouts can remove those coordinates without changing any
    /// output operator, so this kernel retains the full terminal frame space.
    ///
    /// Test-only since the symbolic-frame planner was superseded by
    /// `GuardedSurfaceSpace`: its remaining caller is the terminal-boundary
    /// oracle that cross-checks the live guarded planner.
    #[cfg(test)]
    pub(super) fn for_output_frames(
        stabilizers: &StabilizerGenerators,
    ) -> Result<Self, RuntimeBasisError> {
        let mut basis = Self::for_source_readouts(stabilizers)?;
        let constraints = basis
            .zx_graph
            .action_graph()
            .ordered_nodes()
            .filter_map(|node| {
                let crate::Action::Measure { target, .. } = &node.action else {
                    return None;
                };
                let Some(crate::MeasurementObservable::Concrete(axis)) = node.measurement else {
                    unreachable!("source readout measurements have concrete observables")
                };
                Some((
                    basis
                        .zx_graph
                        .measurement_column(target)
                        .expect("named coordinate"),
                    Pauli::from(axis),
                ))
            })
            .collect::<Vec<_>>();
        basis.restrict_to_zero(&constraints);
        Ok(basis)
    }

    #[cfg(test)]
    fn restrict_to_zero(&mut self, constraints: &[(usize, Pauli)]) {
        let rank = super::stabilizer::gaussian_elimination_with_tracking(
            &mut self.basis.rows,
            &mut self.readout_coeffs,
            constraints.iter().copied(),
            |row, &(col, axis)| row.get(col) & axis,
            None,
        );
        self.basis.rows.drain(..rank);
        self.readout_coeffs.drain(..rank);
        self.basis = StabilizerBasis::all_logical(std::mem::take(&mut self.basis.rows));
    }

    /// Resolves the selective node at `pos` to the `chosen` basis and returns
    /// the evolved basis. Readable provenance stays internal to that basis.
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeBasisError`] if `pos` is not a live selective, the
    /// basis is not allowed for its kind, or the computation is not Clifford.
    pub fn apply_selective_fill(
        &self,
        pos: IVec3,
        chosen: PauliBasis,
    ) -> Result<RuntimeStabilizerBasis, RuntimeBasisError> {
        self.apply_selective_fills(&[(pos, chosen)])
    }

    /// Resolves one known assignment with the usual ordered basis transitions,
    /// but fills the graph only once.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeBasisError`] when a requested fill is invalid.
    ///
    /// # Panics
    ///
    /// Panics if validated selective metadata is internally inconsistent.
    pub fn apply_selective_fills(
        &self,
        fills: &[(IVec3, PauliBasis)],
    ) -> Result<RuntimeStabilizerBasis, RuntimeBasisError> {
        if fills.is_empty() {
            return Ok(self.clone());
        }
        if !self.zx_graph.is_clifford_computation() {
            return Err(RuntimeBasisError::NonCliffordFillUnsupported);
        }

        let port_map = self.selective_port_map(fills)?;

        let mut basis = self.basis.clone();
        let mut readout_coeffs = self.readout_coeffs.clone();
        let mut remaining_constraints = collect_selective_constraints(&self.zx_graph);
        for &(pos, chosen) in fills {
            let kind = self.selective_kinds[&pos];
            let col = self
                .zx_graph
                .node_at(pos)
                .expect("validated selective has a ZX node")
                .id;
            remaining_constraints.retain(|constraint| constraint.pos != pos);
            let constraints = [(col, Pauli::from(chosen))];
            (basis, readout_coeffs) = evolve_selective_basis(
                &basis,
                &readout_coeffs,
                &constraints,
                &remaining_constraints,
                pos,
                kind,
            )?;
        }

        let filled_zx = self.zx_graph.fill_ports(&port_map)?;
        Ok(RuntimeStabilizerBasis {
            input_columns: self.input_columns.clone(),
            prepared_columns: self.prepared_columns.clone(),
            zx_graph: filled_zx,
            basis,
            readout_coeffs,
            measurement_ordinals: self.measurement_ordinals.clone(),
            selective_kinds: remaining_constraints
                .into_iter()
                .map(|constraint| (constraint.pos, constraint.kind))
                .collect(),
        })
    }

    /// Resolves several assignments while sharing their common fill prefixes.
    /// Assignments with different site order use the ordinary independent path.
    ///
    /// # Errors
    ///
    /// Returns the same validation or fill errors as [`Self::apply_selective_fills`].
    ///
    /// # Panics
    ///
    /// Panics if validated selective metadata is internally inconsistent.
    pub fn apply_selective_fill_assignments(
        &self,
        assignments: &[Vec<(IVec3, PauliBasis)>],
    ) -> Result<Vec<RuntimeStabilizerBasis>, RuntimeBasisError> {
        let independently = || {
            assignments
                .iter()
                .map(|fills| self.apply_selective_fills(fills))
                .collect()
        };
        let Some(first) = assignments.first() else {
            return Ok(Vec::new());
        };
        if assignments.iter().any(|assignment| {
            assignment.len() != first.len()
                || assignment
                    .iter()
                    .zip(first)
                    .any(|((pos, _), (first, _))| pos != first)
        }) {
            return independently();
        }
        if first.is_empty() {
            return Ok(vec![self.clone(); assignments.len()]);
        }
        if !self.zx_graph.is_clifford_computation() {
            return Err(RuntimeBasisError::NonCliffordFillUnsupported);
        }
        let Ok(port_maps) = assignments
            .iter()
            .map(|fills| self.selective_port_map(fills))
            .collect::<Result<Vec<_>, _>>()
        else {
            return independently();
        };
        let mut states = vec![None; assignments.len()];
        let remaining = collect_selective_constraints(&self.zx_graph);
        if fill_prefix_states(self, assignments, remaining.clone(), &mut states).is_none() {
            return independently();
        }
        let remaining = first.iter().fold(remaining, |mut remaining, (pos, _)| {
            remaining.retain(|constraint| constraint.pos != *pos);
            remaining
        });
        states
            .into_iter()
            .zip(port_maps)
            .map(|(state, ports)| {
                let (basis, readout_coeffs) = state.expect("every assignment reached a trie leaf");
                Ok(RuntimeStabilizerBasis {
                    input_columns: self.input_columns.clone(),
                    prepared_columns: self.prepared_columns.clone(),
                    zx_graph: self.zx_graph.fill_ports(&ports)?,
                    basis,
                    readout_coeffs,
                    measurement_ordinals: self.measurement_ordinals.clone(),
                    selective_kinds: remaining
                        .iter()
                        .map(|constraint| (constraint.pos, constraint.kind))
                        .collect(),
                })
            })
            .collect()
    }

    /// Derives the stabilizer surface that witnesses measurement variable
    /// `mvar`, projected to what is decidable by time layer `deadline`.
    /// Preserves the evolved readout identity after selective fills and prefers
    /// an output-safe representative when one exists.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeBasisError::UnknownMeasurement`] if `mvar` is not a live
    /// measurement variable.
    pub fn derive_measurement_surface(
        &self,
        mvar: &str,
        deadline: i64,
    ) -> Result<DerivedSurface, RuntimeBasisError> {
        self.derive_measurement_surface_with_constraints(mvar, deadline, &[], &[], false, &[])
    }

    /// Selects an output-free readout using only resolved selective choices.
    ///
    /// # Errors
    ///
    /// Returns an error if `mvar` is unknown or no causal representative exists.
    pub fn derive_causal_measurement_surface(
        &self,
        mvar: &str,
        deadline: i64,
    ) -> Result<DerivedSurface, RuntimeBasisError> {
        self.derive_readout_from_known(mvar, deadline, &[], &[])
    }

    pub(crate) fn derive_readout_from_known(
        &self,
        mvar: &str,
        deadline: i64,
        unavailable: &[Vec<(usize, Pauli)>],
        known: &[usize],
    ) -> Result<DerivedSurface, RuntimeBasisError> {
        let absent = self
            .selective_kinds
            .keys()
            .flat_map(|&pos| {
                let col = self.zx_graph.node_at(pos).expect("live selective node").id;
                [(col, Pauli::X), (col, Pauli::Z)]
            })
            .collect::<Vec<_>>();
        self.derive_measurement_surface_with_constraints(
            mvar,
            deadline,
            &absent,
            unavailable,
            true,
            known,
        )
    }

    fn derive_measurement_surface_with_constraints(
        &self,
        mvar: &str,
        deadline: i64,
        absent: &[(usize, Pauli)],
        unavailable_feedback: &[Vec<(usize, Pauli)>],
        require_output_free: bool,
        known: &[usize],
    ) -> Result<DerivedSurface, RuntimeBasisError> {
        let (derivation_rows, derivation_coeffs) = if require_output_free {
            (self.basis.rows.clone(), self.readout_coeffs.clone())
        } else {
            self.derivation_rows_and_coeffs()
        };
        let DeadlineProjection {
            consumed,
            rows: projected_rows,
            coeffs: projected_coeffs,
        } = deadline_projection_with_initial_coeffs(
            &derivation_rows,
            &derivation_coeffs,
            &self.zx_graph,
            deadline,
        );
        let ordinal = *self
            .measurement_ordinals
            .get(mvar)
            .ok_or_else(|| RuntimeBasisError::UnknownMeasurement(mvar.to_owned()))?;
        let readout_width = self.readout_coeffs.first().map_or(0, CoeffVec::len);
        let rows = &projected_rows[consumed..];
        let outputs = self.zx_graph.output_ports();
        let output_components = outputs
            .iter()
            .flat_map(|&pos| {
                let node = self.zx_graph.node_at(pos).expect("output port is a node");
                [Pauli::X, Pauli::Z]
                    .into_iter()
                    .filter(move |&axis| {
                        node.role != crate::PortRole::Multiplex || axis == Pauli::X
                    })
                    .map(move |axis| (node.id, axis))
            })
            .collect::<Vec<_>>();
        // Selective fills can move a named parity off its original anchor.
        // Preserve its readable identity, not its pre-fill local Pauli.
        // Prefer C0-safe representatives; retain the analysis-only fallback.
        for output_free in [true, false] {
            if !output_free && require_output_free {
                break;
            }
            let components = absent
                .iter()
                .chain(output_components.iter().filter(|_| output_free))
                .copied()
                .collect::<Vec<_>>();
            let width = readout_width + components.len() + unavailable_feedback.len();
            let constraints = rows
                .iter()
                .zip(&projected_coeffs[consumed..])
                .map(|(row, coeff)| {
                    let mut value = CoeffVec::zeros(width);
                    for bit in coeff.iter_ones().filter(|bit| !known.contains(bit)) {
                        value.set_bit(bit, true);
                    }
                    for (offset, &(col, axis)) in components.iter().enumerate() {
                        value.set_bit(readout_width + offset, row.get(col) & axis);
                    }
                    for (offset, targets) in unavailable_feedback.iter().enumerate() {
                        let odd = targets.iter().fold(false, |odd, &(column, pauli)| {
                            odd ^ row.get(column).anticommutes(pauli)
                        });
                        value.set_bit(readout_width + components.len() + offset, odd);
                    }
                    value
                })
                .collect::<Vec<_>>();
            let target = CoeffVec::singleton(ordinal, width);
            if let Some(combination) = solve_coeff_combination(&constraints, &target) {
                let mut row = PauliString::new(self.zx_graph.total_ids());
                let mut coeff = CoeffVec::zeros(readout_width);
                for index in combination.iter_ones() {
                    row ^= &rows[index];
                    coeff.xor_assign(&projected_coeffs[consumed + index]);
                }
                return Ok(materialize_derived_surface(&self.zx_graph, row, coeff));
            }
        }
        Err(StabilizerError::UnavailableControlParity {
            mvar: mvar.to_owned(),
            deadline,
        }
        .into())
    }

    /// Derives the compact terminal output-correction rows consumed by
    /// [`crate::solve_output_correction_symbolic`].
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeBasisError`] if an output is unknown or cannot be
    /// supported.
    pub fn derive_output_correction_rows(
        &self,
        outputs: &[IVec3],
    ) -> Result<Vec<OutputCorrectionRow>, RuntimeBasisError> {
        Ok(self
            .pivoted_output_correction_rows(outputs)?
            .into_iter()
            .map(|(correction, _)| correction)
            .collect())
    }

    /// Derives compact output rows together with their full stabilizer
    /// surfaces. Consumers that must account for effects outside readable
    /// generator provenance, such as feedback on auxiliary directions, use
    /// the materialized surface.
    ///
    /// # Errors
    ///
    /// Returns an error if an output is unknown or cannot be supported.
    pub fn derive_output_correction_surfaces(
        &self,
        outputs: &[IVec3],
    ) -> Result<Vec<(OutputCorrectionRow, Stabilizer)>, RuntimeBasisError> {
        Ok(self
            .pivoted_output_correction_rows(outputs)?
            .into_iter()
            .map(|(correction, row)| (correction, self.zx_graph.materialize_stabilizer(row)))
            .collect())
    }

    /// Closed Clifford surfaces constrain internal outcome assignments without
    /// depending on the state at any open input or output (including T ports).
    #[cfg(feature = "verify")]
    pub(crate) fn closed_stabilizer_surfaces(&self) -> Vec<Stabilizer> {
        let mut rows = self.derivation_rows_and_coeffs().0;
        let constraints = self
            .zx_graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::Port | NodeKind::T))
            .flat_map(|node| [(node.id, Pauli::X), (node.id, Pauli::Z)]);
        let consumed = super::stabilizer::gaussian_elimination_with_tracking(
            &mut rows,
            &mut [],
            constraints,
            |row, &(column, axis)| row.get(column) & axis,
            None,
        );
        rows.into_iter()
            .skip(consumed)
            .map(|row| self.zx_graph.materialize_stabilizer(row))
            .collect()
    }

    /// All resolved rows, including zero-output and private directions.
    #[cfg(feature = "verify")]
    pub(crate) fn resolved_surfaces(&self) -> Vec<Stabilizer> {
        self.derivation_rows_and_coeffs()
            .0
            .into_iter()
            .map(|row| self.zx_graph.materialize_stabilizer(row))
            .collect()
    }

    /// Consumes this basis while deriving output corrections, moving its dense
    /// rows and provenance into the solve instead of cloning them.
    ///
    /// # Errors
    ///
    /// Returns an error if an output is unknown or cannot be supported.
    pub fn into_output_correction_surfaces(
        self,
        outputs: &[IVec3],
    ) -> Result<Vec<(OutputCorrectionRow, Stabilizer)>, RuntimeBasisError> {
        let Self {
            zx_graph,
            input_columns,
            prepared_columns,
            basis,
            readout_coeffs,
            ..
        } = self;
        let (rows, coeffs) = basis
            .rows
            .into_iter()
            .zip(basis.kinds)
            .zip(readout_coeffs)
            .filter(|((_, kind), _)| !kind.is_selective_fixing())
            .map(|((row, _), coeff)| (row, coeff))
            .unzip();
        let output_cols = requested_output_columns(&zx_graph, outputs)?;
        let protected = protected_boundary_columns(&zx_graph, &prepared_columns, &output_cols);
        Ok(pivot_output_correction_rows(
            rows,
            coeffs,
            outputs,
            &output_cols,
            &input_columns,
            &protected,
        )
        .into_iter()
        .map(|(correction, row)| (correction, zx_graph.materialize_stabilizer(row)))
        .collect())
    }

    fn pivoted_output_correction_rows(
        &self,
        outputs: &[IVec3],
    ) -> Result<Vec<(OutputCorrectionRow, PauliString)>, RuntimeBasisError> {
        let projected = self.projected_rows_for_outputs(outputs)?;
        let protected = protected_boundary_columns(
            &self.zx_graph,
            &self.prepared_columns,
            &projected.output_cols,
        );
        Ok(pivot_output_correction_rows(
            projected.rows,
            projected.coeffs,
            outputs,
            &projected.output_cols,
            &self.input_columns,
            &protected,
        ))
    }

    /// Projects the derivation rows past the last layer of the graph and
    /// resolves the requested outputs to their live port nodes.
    fn projected_rows_for_outputs(
        &self,
        outputs: &[IVec3],
    ) -> Result<ProjectedOutputRows, RuntimeBasisError> {
        let (projected_rows, projected_coeffs) = self.derivation_rows_and_coeffs();
        let output_cols = requested_output_columns(&self.zx_graph, outputs)?;
        Ok(ProjectedOutputRows {
            rows: projected_rows,
            coeffs: projected_coeffs,
            output_cols,
        })
    }
    fn derivation_rows_and_coeffs(&self) -> (Vec<PauliString>, Vec<CoeffVec>) {
        self.basis
            .rows
            .iter()
            .zip(&self.basis.kinds)
            .zip(&self.readout_coeffs)
            .filter(|((_, kind), _)| !kind.is_selective_fixing())
            .map(|((row, _), coeff)| (row.clone(), coeff.clone()))
            .unzip()
    }
}

/// Capture boundary ownership before native resources are relaxed for fills.
pub(super) fn authored_boundary_columns(zx: &ZXGraph) -> (Vec<usize>, Vec<usize>) {
    let mut nodes = zx
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::T || node.is_input_port(zx))
        .collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|node| node.pos.to_array());
    let inputs = nodes
        .iter()
        .filter(|node| node.is_input_port(zx))
        .map(|node| node.id)
        .collect();
    let prepared = nodes
        .iter()
        .filter(|node| node.kind == NodeKind::T)
        .map(|node| node.id)
        .collect();
    (inputs, prepared)
}

/// A partial-output query must also preserve every other live output.
fn protected_boundary_columns(zx: &ZXGraph, prepared: &[usize], outputs: &[usize]) -> Vec<usize> {
    let mut columns = prepared
        .iter()
        .copied()
        .chain(
            zx.nodes()
                .iter()
                .filter(|node| node.is_output_port(zx) && !outputs.contains(&node.id))
                .map(|node| node.id),
        )
        .collect::<Vec<_>>();
    columns.sort_unstable_by_key(|&column| zx.nodes()[column].pos.to_array());
    columns.dedup();
    columns
}

struct FillWork {
    indices: Vec<usize>,
    depth: usize,
    basis: StabilizerBasis,
    readout_coeffs: Vec<CoeffVec>,
    remaining: Vec<SelectiveConstraint>,
}

fn fill_prefix_states(
    source: &RuntimeStabilizerBasis,
    assignments: &[Vec<(IVec3, PauliBasis)>],
    remaining: Vec<SelectiveConstraint>,
    states: &mut [Option<(StabilizerBasis, Vec<CoeffVec>)>],
) -> Option<()> {
    let mut stack = vec![FillWork {
        indices: (0..assignments.len()).collect(),
        depth: 0,
        basis: source.basis.clone(),
        readout_coeffs: source.readout_coeffs.clone(),
        remaining,
    }];
    while let Some(mut work) = stack.pop() {
        loop {
            if work.depth == assignments[work.indices[0]].len() {
                let last = work.indices.pop()?;
                for &index in &work.indices {
                    states[index] = Some((work.basis.clone(), work.readout_coeffs.clone()));
                }
                states[last] = Some((work.basis, work.readout_coeffs));
                break;
            }
            let pos = assignments[work.indices[0]][work.depth].0;
            let kind = source.selective_kinds[&pos];
            let col = source
                .zx_graph
                .node_at(pos)
                .expect("validated selective has a ZX node")
                .id;
            work.remaining.retain(|constraint| constraint.pos != pos);
            let mut groups: Vec<(PauliBasis, Vec<usize>)> = Vec::new();
            for &index in &work.indices {
                let chosen = assignments[index][work.depth].1;
                if let Some((_, group)) = groups.iter_mut().find(|(value, _)| *value == chosen) {
                    group.push(index);
                } else {
                    groups.push((chosen, vec![index]));
                }
            }
            if groups.len() == 1 {
                let (chosen, indices) = groups.pop()?;
                (work.basis, work.readout_coeffs) = evolve_selective_basis(
                    &work.basis,
                    &work.readout_coeffs,
                    &[(col, Pauli::from(chosen))],
                    &work.remaining,
                    pos,
                    kind,
                )
                .ok()?;
                work.indices = indices;
                work.depth += 1;
                continue;
            }
            for (chosen, indices) in groups {
                let (basis, readout_coeffs) = evolve_selective_basis(
                    &work.basis,
                    &work.readout_coeffs,
                    &[(col, Pauli::from(chosen))],
                    &work.remaining,
                    pos,
                    kind,
                )
                .ok()?;
                stack.push(FillWork {
                    indices,
                    depth: work.depth + 1,
                    basis,
                    readout_coeffs,
                    remaining: work.remaining.clone(),
                });
            }
            break;
        }
    }
    Some(())
}

fn evolve_selective_basis(
    basis: &StabilizerBasis,
    readout_coeffs: &[CoeffVec],
    constraints: &[(usize, Pauli)],
    remaining_constraints: &[SelectiveConstraint],
    pos: IVec3,
    kind: SelectiveKind,
) -> Result<(StabilizerBasis, Vec<CoeffVec>), RuntimeBasisError> {
    let (basis, coeffs) = if basis
        .kinds
        .iter()
        .any(|row_kind| row_kind.fixes_selective(pos))
    {
        match basis.apply_tagged_selective_fill(constraints, pos, kind) {
            Ok(transition) => transition,
            Err(StabilizerError::SelectiveSupportUnsatisfiable { .. }) => {
                basis.apply_generic_selective_fill(constraints, remaining_constraints)?
            }
            Err(err) => return Err(err.into()),
        }
    } else {
        basis.apply_generic_selective_fill(constraints, remaining_constraints)?
    };
    Ok((basis, compose_coeffs(&coeffs, readout_coeffs)))
}

fn initial_readout_coeffs(kinds: &[StabilizerRowKind], public_len: usize) -> Vec<CoeffVec> {
    let width = kinds.len();
    kinds
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            if index < public_len && kind.is_readout() {
                CoeffVec::singleton(index, width)
            } else {
                CoeffVec::zeros(width)
            }
        })
        .collect()
}

fn compose_coeffs(transform: &[CoeffVec], source: &[CoeffVec]) -> Vec<CoeffVec> {
    let width = source.first().map_or(0, CoeffVec::len);
    transform
        .iter()
        .map(|combination| {
            let mut composed = CoeffVec::zeros(width);
            for source_index in combination.iter_ones() {
                composed.xor_assign(&source[source_index]);
            }
            composed
        })
        .collect()
}

fn selective_kinds_of(zx_graph: &ZXGraph) -> HashMap<IVec3, SelectiveKind> {
    collect_selective_constraints(zx_graph)
        .into_iter()
        .map(|constraint| (constraint.pos, constraint.kind))
        .collect()
}

/// Derivation rows projected past the graph's last layer, plus requested output
/// columns in caller order.
struct ProjectedOutputRows {
    rows: Vec<PauliString>,
    coeffs: Vec<CoeffVec>,
    output_cols: Vec<usize>,
}

fn materialize_derived_surface(
    zx_graph: &ZXGraph,
    row: PauliString,
    coeff: CoeffVec,
) -> DerivedSurface {
    DerivedSurface {
        readout_ordinals: coeff.to_indices(),
        stabilizer: zx_graph.materialize_stabilizer(row),
    }
}

fn requested_output_columns(
    zx_graph: &ZXGraph,
    outputs: &[IVec3],
) -> Result<Vec<usize>, RuntimeBasisError> {
    outputs
        .iter()
        .copied()
        .map(|output| {
            let node_id = zx_graph
                .node_at(output)
                .filter(|node| node.is_output_port(zx_graph))
                .map(|node| node.id)
                .ok_or(RuntimeBasisError::UnknownOutput(output))?;
            Ok(node_id)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use glam::ivec3;

    use crate::MeasurementObservable;
    use crate::{BlockGraph, GalleryItem, Pauli, PauliBasis, PauliString};

    use super::{CoeffVec, RuntimeBasisError, RuntimeStabilizerBasis};

    #[test]
    fn source_readouts_preserve_the_input_instrument_under_a_public_rebase() {
        let mut generators = GalleryItem::CCZInjectedAnd
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection")
            .stabilizers()
            .unwrap();
        let source = RuntimeStabilizerBasis::for_source_readouts(&generators).unwrap();
        let expected = source
            .derive_causal_measurement_surface("m1", i64::MAX)
            .unwrap();
        let named = generators
            .generators
            .iter()
            .position(|g| g.measurement_name() == Some("m1"))
            .unwrap();
        let logical = generators
            .generators
            .iter()
            .position(|g| matches!(g.kind, crate::StabilizerRowKind::Logical))
            .unwrap();
        let mixed = &generators.generators[named].stabilizer.paulis
            ^ &generators.generators[logical].stabilizer.paulis;
        generators.generators[named].stabilizer = generators.zx_graph.materialize_stabilizer(mixed);
        let rebased = RuntimeStabilizerBasis::for_source_readouts(&generators)
            .unwrap()
            .derive_causal_measurement_surface("m1", i64::MAX)
            .unwrap();
        assert_eq!(
            rebased.stabilizer.port_stabilizer,
            expected.stabilizer.port_stabilizer
        );
        assert_eq!(rebased.stabilizer.sign, expected.stabilizer.sign);
    }

    #[test]
    fn joint_fill_provenance_never_names_internal_rows() {
        let graph = GalleryItem::ToffoliFromAndDelayedCZ
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let stabilizers = graph.stabilizers().unwrap();
        let public_rows = stabilizers.generators.len();
        assert!(!stabilizers.auxiliary_rows().is_empty());

        let mut basis =
            RuntimeStabilizerBasis::from_generators(&stabilizers).with_t_nodes_as_ports();
        assert!(
            basis.readout_coeffs[public_rows..]
                .iter()
                .all(CoeffVec::is_zero)
        );

        let resolves = graph
            .actions()
            .iter()
            .filter_map(|action| match action {
                crate::Action::Resolve { target, .. } => Some(*target),
                _ => None,
            })
            .collect::<Vec<_>>();
        for pos in resolves {
            let kind = basis.selective_kinds[&pos];
            basis = basis
                .apply_selective_fill(pos, kind.pauli_if_false())
                .unwrap();
        }

        let outputs = basis.zx_graph.output_ports();
        for correction in basis.derive_output_correction_rows(&outputs).unwrap() {
            assert!(
                correction
                    .readout_ordinals
                    .iter()
                    .all(|&ordinal| ordinal < public_rows),
                "derived readout provenance exposed an internal row"
            );
        }
    }

    #[test]
    fn batched_fills_match_sequential_output_surfaces() {
        for entry in [
            GalleryItem::CCZInjectedAnd,
            GalleryItem::CCZInjectedMaj,
            GalleryItem::ToffoliFromAndDelayedCZ,
            GalleryItem::T,
        ] {
            let graph = entry
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            let stabilizers = graph.stabilizers().unwrap();
            let base =
                RuntimeStabilizerBasis::from_generators(&stabilizers).with_t_nodes_as_ports();
            assert!(base.input_columns.iter().all(|&column| {
                stabilizers.zx_graph.nodes()[column].is_input_port(&stabilizers.zx_graph)
            }));
            assert!(base.prepared_columns.iter().all(|&column| {
                stabilizers.zx_graph.nodes()[column].kind == crate::NodeKind::T
                    && base.zx_graph.nodes()[column].kind == crate::NodeKind::Port
                    && !base.input_columns.contains(&column)
            }));
            let sites = graph
                .actions()
                .iter()
                .filter_map(|action| match action {
                    crate::Action::Resolve { target, .. } => Some(*target),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let outputs = stabilizers.zx_graph.output_ports();

            let assignments = stabilizers
                .zx_graph
                .action_graph()
                .resolve_value_domain(&sites)
                .unwrap()
                .values()
                .iter()
                .map(|bits| {
                    sites
                        .iter()
                        .zip(bits)
                        .map(|(&pos, &bit)| {
                            let kind = base.selective_kinds[&pos];
                            (
                                pos,
                                if bit {
                                    kind.pauli_if_true()
                                } else {
                                    kind.pauli_if_false()
                                },
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let forked = base.apply_selective_fill_assignments(&assignments).unwrap();
            for (fills, forked) in assignments.iter().zip(forked) {
                let batched = base.apply_selective_fills(fills).unwrap();
                let sequential = fills.iter().try_fold(base.clone(), |basis, &(pos, axis)| {
                    basis.apply_selective_fill(pos, axis)
                });

                let sequential = sequential
                    .unwrap()
                    .derive_output_correction_surfaces(&outputs)
                    .unwrap();
                assert_eq!(
                    batched.derive_output_correction_surfaces(&outputs).unwrap(),
                    sequential,
                    "borrowed {entry:?} assignment {fills:?}"
                );
                assert_eq!(
                    batched.into_output_correction_surfaces(&outputs).unwrap(),
                    sequential,
                    "consuming {entry:?} assignment {fills:?}"
                );
                assert_eq!(
                    forked.into_output_correction_surfaces(&outputs).unwrap(),
                    sequential,
                    "forked {entry:?} assignment {fills:?}"
                );
            }

            let mut reordered = assignments[..2].to_vec();
            reordered[0].reverse();
            let fallback = base.apply_selective_fill_assignments(&reordered).unwrap();
            for (fills, fallback) in reordered.iter().zip(fallback) {
                let expected = base
                    .apply_selective_fills(fills)
                    .unwrap()
                    .into_output_correction_surfaces(&outputs)
                    .unwrap();
                assert_eq!(
                    fallback.into_output_correction_surfaces(&outputs).unwrap(),
                    expected,
                    "fallback {entry:?} assignment {fills:?}"
                );
            }

            let repeated = vec![assignments[0].clone(); 3];
            let expected = base
                .apply_selective_fills(&repeated[0])
                .unwrap()
                .into_output_correction_surfaces(&outputs)
                .unwrap();
            for filled in base.apply_selective_fill_assignments(&repeated).unwrap() {
                assert_eq!(
                    filled.into_output_correction_surfaces(&outputs).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn uma_readout_prefers_output_free_surface_after_z_fills() {
        let stabilizers = GalleryItem::UMA
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection")
            .stabilizers()
            .unwrap();
        stabilizers
            .validate_measurements_close_before_outputs()
            .unwrap();
        let basis = RuntimeStabilizerBasis::from_generators(&stabilizers)
            .apply_selective_fills(&[
                (ivec3(2, 1, 2), PauliBasis::Z),
                (ivec3(2, 2, 2), PauliBasis::Z),
            ])
            .unwrap();
        let derived = basis.derive_measurement_surface("m_ikprime", 2).unwrap();
        assert!(
            !derived
                .stabilizer
                .port_stabilizer
                .contains_key(&ivec3(2, 0, 1))
        );
        let ordinal = stabilizers
            .generators
            .iter()
            .position(|row| row.measurement_name() == Some("m_ikprime"))
            .unwrap();
        assert_eq!(derived.readout_ordinals, vec![ordinal]);
    }

    #[test]
    fn measurement_derivation_preserves_observable_and_readout_identity() {
        for (kind, expected) in [("XZX", Pauli::X), ("ZXZ", Pauli::Z)] {
            let graph = BlockGraph::from_blog_text(&format!(
                "BLOG 1.0\n0: Port [0,0,-1]\n1: {kind} [0,0,0]\n2: Port [0,0,1]\n0 -> +Z\n1 -> +Z\nm = measure 1\n"
            )).unwrap();
            let stabilizers = graph.stabilizers().unwrap();
            let ordinal = stabilizers
                .generators
                .iter()
                .position(|generator| generator.measurement_name() == Some("m"))
                .unwrap();
            let basis = RuntimeStabilizerBasis::from_generators(&stabilizers);
            let derived = basis.derive_measurement_surface("m", 2).unwrap();
            assert_eq!(derived.stabilizer.interior_nodes[&ivec3(0, 0, 0)], expected);
            assert_eq!(derived.readout_ordinals, vec![ordinal]);
        }
    }

    #[test]
    fn runtime_basis_uses_measurement_actions() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = crate::ZXGraph::try_from(&graph).unwrap();

        let basis = RuntimeStabilizerBasis::from_zx_graph(zx).unwrap();
        let derived = basis.derive_measurement_surface("mzz", 2).unwrap();

        assert!(!derived.readout_ordinals.is_empty());
    }

    #[test]
    fn apply_selective_fill_refreshes_filled_selective_measurement_metadata() {
        let mut graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: XZ [0,0,1]\n  2: ZXZ [1,0,0]\n  [0,0,0] -> +Z\n",
        )
        .unwrap();
        graph
            .set_actions_lenient(vec![
                crate::Action::Measure {
                    target: crate::MeasureTarget::Node(ivec3(0, 0, 1)),
                    name: "m".into(),
                },
                crate::Action::Measure {
                    target: crate::MeasureTarget::Node(ivec3(1, 0, 0)),
                    name: "m2".into(),
                },
                crate::Action::Resolve {
                    target: ivec3(0, 0, 1),
                    condition: crate::Expr::Var("m2".into()),
                },
            ])
            .unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let basis = RuntimeStabilizerBasis::from_zx_graph(zx).unwrap();

        let filled = basis
            .apply_selective_fill(ivec3(0, 0, 1), PauliBasis::Z)
            .unwrap();
        let zx = &filled.zx_graph;

        assert_eq!(
            zx.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(PauliBasis::Z))
        );
        let ordinal = basis.measurement_ordinals["m"];
        let derived = filled.derive_measurement_surface("m", 2).unwrap();
        assert_eq!(derived.stabilizer.interior_nodes[&ivec3(0, 0, 1)], Pauli::Z);
        assert_eq!(derived.readout_ordinals, vec![ordinal]);
    }

    #[test]
    fn apply_selective_fill_rejects_non_clifford_graphs() {
        let x_basis_graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let x_basis = RuntimeStabilizerBasis::from_block_graph(&x_basis_graph).unwrap();
        let x_err = x_basis
            .apply_selective_fill(ivec3(1, 0, 2), crate::PauliBasis::X)
            .unwrap_err();
        assert!(matches!(
            x_err,
            RuntimeBasisError::NonCliffordFillUnsupported
        ));

        let z_basis_graph = GalleryItem::TWithPreparedY
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let z_basis = RuntimeStabilizerBasis::from_block_graph(&z_basis_graph).unwrap();
        let z_err = z_basis
            .apply_selective_fill(ivec3(2, 1, 2), crate::PauliBasis::Z)
            .unwrap_err();
        assert!(matches!(
            z_err,
            RuntimeBasisError::NonCliffordFillUnsupported
        ));
    }

    #[test]
    fn selective_future_pipe_topology_is_rejected() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: XZ [0,0,1]\n  1: ZXZ [0,0,2]\n  2: Port [0,0,3]\n  [0,0,1] -> +Z\n  [0,0,2] -> +Z\n",
        )
        .unwrap();
        RuntimeStabilizerBasis::from_block_graph(&graph).unwrap_err();
    }

    #[test]
    fn tagged_selective_fill_preserves_remaining_selective_fixing_rows() {
        let filled_pos = ivec3(0, 0, 0);
        let remaining_pos = ivec3(1, 0, 0);
        let remaining_col = 1;
        let remaining_forbidden = Pauli::Z;
        let generators = vec![
            PauliString::try_from("Z_").unwrap(),
            PauliString::try_from("_Z").unwrap(),
            PauliString::try_from("Y_").unwrap(),
            PauliString::try_from("_X").unwrap(),
        ];
        let row_kinds = vec![
            super::super::stabilizer::StabilizerRowKind::SelectiveFixing {
                targets: vec![super::super::stabilizer::SelectiveFixingTarget {
                    pos: filled_pos,
                    forbidden: Pauli::Z,
                }],
            },
            super::super::stabilizer::StabilizerRowKind::SelectiveFixing {
                targets: vec![super::super::stabilizer::SelectiveFixingTarget {
                    pos: remaining_pos,
                    forbidden: remaining_forbidden,
                }],
            },
            super::super::stabilizer::StabilizerRowKind::Logical,
            super::super::stabilizer::StabilizerRowKind::Logical,
        ];

        let basis = super::super::stabilizer::StabilizerBasis::new(generators, row_kinds);
        let (result, _) = basis
            .apply_tagged_selective_fill(&[(0, Pauli::X)], filled_pos, crate::SelectiveKind::XY)
            .unwrap();

        assert!(
            result
                .kinds
                .iter()
                .all(|kind| !kind.fixes_selective(filled_pos))
        );
        assert!(result.rows.iter().zip(&result.kinds).any(|(row, kind)| {
            kind.fixes_selective(remaining_pos) && row.get(remaining_col) == remaining_forbidden
        }));
    }

    #[test]
    fn output_correction_rows_cover_requested_outputs() {
        let graph = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let basis = RuntimeStabilizerBasis::from_block_graph(&graph).unwrap();
        let outputs = [ivec3(0, 0, 3), ivec3(1, 1, 3)];
        let joints = basis.derive_output_correction_rows(&outputs).unwrap();

        assert!(!joints.is_empty());
        for output in outputs {
            assert!(joints.iter().any(|joint| {
                joint
                    .output_support
                    .iter()
                    .any(|&(supported, _)| supported == output)
            }));
        }
        assert!(
            joints
                .iter()
                .all(|joint| !joint.readout_ordinals.is_empty())
        );
    }

    #[test]
    fn output_projection_past_i32_max_layer_is_representable() {
        let graph = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let max_z = *graph.spans().unwrap().2.end();
        let shifted = graph
            .shift_positions(ivec3(0, 0, i32::MAX - max_z))
            .unwrap();
        let zx = crate::ZXGraph::try_from(&shifted).unwrap();
        assert_eq!(
            zx.nodes().iter().map(|node| node.pos.z).max(),
            Some(i32::MAX)
        );
        let outputs = zx.output_ports();
        let basis = RuntimeStabilizerBasis::from_zx_graph(zx).unwrap();

        assert!(
            !basis
                .derive_output_correction_rows(&outputs)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn output_correction_rows_report_unknown_output() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let basis = RuntimeStabilizerBasis::from_block_graph(&graph).unwrap();
        let err = basis
            .derive_output_correction_rows(&[ivec3(99, 99, 99)])
            .expect_err("missing output should error");

        assert!(matches!(err, RuntimeBasisError::UnknownOutput(pos) if pos == ivec3(99, 99, 99)));
    }

    #[test]
    fn ccz_factory_runtime_basis_derives_mx1357_surface() {
        let graph = GalleryItem::CCZFactoryWithTels
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let basis = RuntimeStabilizerBasis::from_block_graph(&graph).unwrap();
        let derived = basis.derive_measurement_surface("mx1357", 7).unwrap();

        assert!(!derived.readout_ordinals.is_empty());
        assert!(
            derived
                .stabilizer
                .interior_nodes
                .contains_key(&ivec3(2, 0, 6))
        );
    }
}
