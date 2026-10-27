//! Runtime-selectable stabilizer bases and their output-correction surfaces.

use std::collections::HashMap;
use std::ops::BitXorAssign;
use std::sync::Arc;

use bloq_utils::boolean::BooleanDecisionDiagram;
use glam::IVec3;
use thiserror::Error;

use super::graph::insert_selective_fill;
use super::output_correction::{OutputCorrectionRow, pivot_output_correction_rows};
use super::reduction::{CanonicalCorrelation, CorrelationReduction, OriginalCoordinates};
use super::stabilizer::{
    CoeffVec, SelectiveConstraint, collect_selective_constraints,
    gaussian_elimination_with_tracking, normalize_selective_support_in_place,
    normalize_selective_support_with, reduce_to_basis,
};
use super::{NodeKind, Stabilizer, StabilizerError, StabilizerGenerators, ZXError, ZXGraph};
use crate::{
    CancellationToken, ModuleCertificationLimits, Pauli, PauliBasis, PauliString, SelectiveKind,
};

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
/// resolved, and the source of output-correction surfaces.
#[derive(Debug, Clone)]
pub struct RuntimeStabilizerBasis {
    zx_graph: ZXGraph,
    // Filling treats native T states as ports, but does not turn prepared
    // resources into arbitrary inputs of the authored map.
    input_columns: Vec<usize>,
    prepared_columns: Vec<usize>,
    rows: Vec<PauliString>,
    // Certified initial equalities survive every runtime row combination and fill.
    coordinates: Option<Arc<OriginalCoordinates>>,
    /// Readable-value provenance of every live row, expressed in the original
    /// public generator ordinals. Rank-completion directions carry the zero
    /// vector. Row operations update these coefficients with Pauli support.
    readout_coeffs: Vec<CoeffVec>,
    selective_kinds: HashMap<IVec3, SelectiveKind>,
}

impl RuntimeStabilizerBasis {
    /// The live ZX graph backing this basis.
    pub fn zx_graph(&self) -> &ZXGraph {
        &self.zx_graph
    }

    /// The Pauli-string generators of this basis, in row order.
    pub fn generators(&self) -> &[PauliString] {
        &self.rows
    }

    /// Demotes every [`T`](NodeKind::T) node to a plain [`Port`](NodeKind::Port).
    /// Used when selective filling treats prepared T states as open inputs.
    pub fn with_t_nodes_as_ports(mut self) -> Self {
        self.zx_graph.demote_t_nodes_to_ports();
        self
    }

    /// A complete basis of the graph's correlation relation, with every row
    /// normalized to permitted selective support.
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeBasisError`] if a selective site admits no permitted
    /// support, computation is cancelled, or the relation size is unrepresentable.
    pub fn from_zx_graph(zx_graph: ZXGraph) -> Result<Self, RuntimeBasisError> {
        let (rows, coordinates) = match Self::reduced_relation(&zx_graph)? {
            Some(relation) => (relation.rows, Some(Arc::new(relation.coordinates))),
            None => (
                reduce_to_basis(&zx_graph.to_external_generator_table(), zx_graph.total_ids),
                None,
            ),
        };
        let mut basis = Self::from_complete_rows(zx_graph, rows)?;
        basis.coordinates = coordinates;
        Ok(basis)
    }

    fn reduced_relation(
        zx_graph: &ZXGraph,
    ) -> Result<Option<CanonicalCorrelation>, StabilizerError> {
        CancellationToken::checkpoint()?;
        if zx_graph.stabilizer_phase_basis.get().is_some() {
            return Ok(None);
        }
        // This public constructor has never imposed certification budgets.
        let limits = ModuleCertificationLimits::UNLIMITED;
        let mut diagram = BooleanDecisionDiagram::with_limits(limits.boolean_limits());
        CorrelationReduction::new(zx_graph, &mut diagram, limits)?
            .map(CorrelationReduction::into_canonical_relation)
            .transpose()
    }

    #[cfg(test)]
    pub(super) fn from_zx_graph_unreduced(zx_graph: ZXGraph) -> Result<Self, RuntimeBasisError> {
        let rows = reduce_to_basis(&zx_graph.to_external_generator_table(), zx_graph.total_ids);
        Self::from_complete_rows(zx_graph, rows)
    }

    fn from_complete_rows(
        zx_graph: ZXGraph,
        mut rows: Vec<PauliString>,
    ) -> Result<Self, RuntimeBasisError> {
        // Normalization adds rows to rows, so they remain an independent basis.
        normalize_selective_support_in_place(
            &mut rows,
            None,
            &collect_selective_constraints(&zx_graph),
        )?;
        let public_len = rows.len();
        Ok(Self::new(zx_graph, rows, public_len))
    }

    /// Builds the runtime basis from a [`StabilizerGenerators`] presentation,
    /// reusing its public rows and then its rank completion verbatim.
    ///
    /// The *n*th runtime generator is the *n*th input generator, so the row
    /// ordinals downstream code keys on stay identity-stable.
    pub fn from_generators(stabilizers: &StabilizerGenerators) -> Self {
        let rows = stabilizers
            .generators
            .iter()
            .map(|generator| generator.stabilizer.paulis.clone())
            .chain(stabilizers.auxiliary_rows().iter().cloned())
            .collect();
        Self::new(
            stabilizers.zx_graph.clone(),
            rows,
            stabilizers.generators.len(),
        )
    }

    fn new(zx_graph: ZXGraph, rows: Vec<PauliString>, public_len: usize) -> Self {
        let width = rows.len();
        let readout_coeffs = (0..width)
            .map(|index| {
                if index < public_len {
                    CoeffVec::singleton(index, width)
                } else {
                    CoeffVec::zeros(width)
                }
            })
            .collect();
        let (input_columns, prepared_columns) = authored_boundary_columns(&zx_graph);
        Self {
            input_columns,
            prepared_columns,
            selective_kinds: collect_selective_constraints(&zx_graph)
                .into_iter()
                .map(|constraint| (constraint.pos, constraint.kind))
                .collect(),
            zx_graph,
            rows,
            coordinates: None,
            readout_coeffs,
        }
    }

    /// A complete basis whose readable coordinates are the source outcomes:
    /// coordinate `i` of a row is its measured component at the `i`th named
    /// measurement in action order. T nodes become ports. Crossing-node and Y
    /// readouts have no linear coordinate.
    ///
    /// # Errors
    ///
    /// Returns an error if a named source measurement lacks a concrete linear
    /// coordinate, or a selective site admits no permitted support.
    pub fn for_source_readouts(zx_graph: ZXGraph) -> Result<Self, RuntimeBasisError> {
        Self::from_zx_graph(zx_graph)?.with_source_readouts()
    }

    fn with_source_readouts(self) -> Result<Self, RuntimeBasisError> {
        let mut basis = self.with_t_nodes_as_ports();
        let measurements = basis
            .zx_graph
            .action_graph()
            .ordered_nodes()
            .filter(|node| matches!(node.action, crate::Action::Measure { .. }))
            .collect::<Vec<_>>();
        let mut readout_coeffs = vec![CoeffVec::zeros(measurements.len()); basis.rows.len()];
        for (ordinal, node) in measurements.into_iter().enumerate() {
            let crate::Action::Measure { name, target } = &node.action else {
                unreachable!("filtered to measurements")
            };
            let unavailable =
                || StabilizerError::MeasurementSurfaceUnavailable { mvar: name.clone() };
            let axis = match node.measurement {
                Some(crate::MeasurementObservable::Concrete(PauliBasis::X)) => Pauli::X,
                Some(crate::MeasurementObservable::Concrete(PauliBasis::Z)) => Pauli::Z,
                _ => return Err(unavailable().into()),
            };
            let col = basis
                .zx_graph
                .measurement_column(target)
                .ok_or_else(|| RuntimeBasisError::UnknownMeasurement(name.clone()))?;
            if basis.zx_graph.nodes().get(col).is_some_and(|node| {
                matches!(node.kind, NodeKind::X | NodeKind::Z) && node.kind.cross_pauli() == axis
            }) {
                return Err(unavailable().into());
            }
            for (row, coeff) in basis.rows.iter().zip(&mut readout_coeffs) {
                coeff.set_bit(ordinal, row.get(col) & axis);
            }
        }
        basis.readout_coeffs = readout_coeffs;
        Ok(basis)
    }

    /// The relation restricted to zero source named-outcome coordinates, with
    /// T nodes as ports. Closed named readouts can remove those coordinates
    /// without changing any output operator, so this kernel retains the full
    /// terminal frame space: the oracle that cross-checks the guarded planner.
    #[cfg(test)]
    pub(super) fn for_output_frames(zx: ZXGraph) -> Result<Self, RuntimeBasisError> {
        let mut basis = Self::from_zx_graph_unreduced(zx)?.with_t_nodes_as_ports();
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
        let rank = gaussian_elimination_with_tracking(
            &mut self.rows,
            &mut self.readout_coeffs,
            constraints.iter().copied(),
            |row, &(col, axis)| row.get(col) & axis,
            None,
        );
        self.rows.drain(..rank);
        self.readout_coeffs.drain(..rank);
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

    /// Resolves one known assignment in order, filling the graph once. Each
    /// fill eliminates the rows that disagree with its chosen axis, then
    /// re-normalizes the remaining selective support.
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

        let mut port_map = HashMap::with_capacity(fills.len());
        for &(pos, chosen) in fills {
            let kind = self
                .selective_kinds
                .get(&pos)
                .copied()
                .ok_or(RuntimeBasisError::UnknownSelective(pos))?;
            insert_selective_fill(&mut port_map, pos, kind, chosen)?;
        }

        let mut remaining = collect_selective_constraints(&self.zx_graph);
        let columns = fills
            .iter()
            .map(|&(pos, chosen)| {
                (
                    self.zx_graph
                        .node_at(pos)
                        .expect("validated selective has a ZX node")
                        .id,
                    Pauli::from(chosen),
                )
            })
            .collect::<Vec<_>>();
        // Single fills avoid repacking the complete basis.
        let (rows, readout_coeffs) = if let Some(coordinates) =
            self.coordinates.as_ref().filter(|_| fills.len() > 1)
        {
            // This temporary namespace names this invocation's input rows,
            // independently of their original readable-value provenance.
            let mut traces = (0..self.rows.len())
                .map(|index| {
                    CancellationToken::checkpoint()?;
                    Ok(CoeffVec::singleton(index, self.rows.len()))
                })
                .collect::<Result<Vec<_>, StabilizerError>>()?;
            let query_bits = remaining.len().saturating_mul(2);
            if query_bits.div_ceil(64) < coordinates.class_count().div_ceil(64) {
                let mut columns = columns;
                let mut rows = project_selective_queries(&self.rows, &mut remaining, &mut columns)?;
                restrict_selective_rows(
                    &mut rows,
                    &mut traces,
                    &mut remaining,
                    &columns,
                    selective_query_pauli,
                )?;
            } else {
                let mut rows = coordinates.pack_rows(&self.rows)?;
                restrict_selective_rows(
                    &mut rows,
                    &mut traces,
                    &mut remaining,
                    &columns,
                    |row, column| coordinates.get(row, column),
                )?;
            }
            self.replay_fill_traces(&traces)?
        } else {
            let mut rows = self.rows.clone();
            let mut readout_coeffs = self.readout_coeffs.clone();
            restrict_selective_rows(
                &mut rows,
                &mut readout_coeffs,
                &mut remaining,
                &columns,
                PauliString::get,
            )?;
            (rows, readout_coeffs)
        };

        Ok(RuntimeStabilizerBasis {
            input_columns: self.input_columns.clone(),
            prepared_columns: self.prepared_columns.clone(),
            zx_graph: self.zx_graph.fill_ports(&port_map)?,
            rows,
            coordinates: self.coordinates.clone(),
            readout_coeffs,
            selective_kinds: remaining
                .into_iter()
                .map(|constraint| (constraint.pos, constraint.kind))
                .collect(),
        })
    }

    fn replay_fill_traces(
        &self,
        traces: &[CoeffVec],
    ) -> Result<(Vec<PauliString>, Vec<CoeffVec>), StabilizerError> {
        let mut rows = Vec::with_capacity(traces.len());
        let mut coefficients = Vec::with_capacity(traces.len());
        for trace in traces {
            CancellationToken::checkpoint()?;
            let mut sources = trace.iter_ones();
            let Some(first) = sources.next() else {
                rows.push(PauliString::new(self.zx_graph.total_ids()));
                coefficients.push(
                    self.readout_coeffs
                        .first()
                        .map_or_else(|| CoeffVec::zeros(0), CoeffVec::zeros_like),
                );
                continue;
            };
            let mut row = self.rows[first].clone();
            let mut coefficient = self.readout_coeffs[first].clone();
            for source in sources {
                CancellationToken::checkpoint()?;
                row ^= &self.rows[source];
                coefficient.xor_assign(&self.readout_coeffs[source]);
            }
            rows.push(row);
            coefficients.push(coefficient);
        }
        Ok((rows, coefficients))
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
        self.clone().into_output_correction_surfaces(outputs)
    }

    /// Closed Clifford surfaces constrain internal outcome assignments without
    /// depending on the state at any open input or output (including T ports).
    #[cfg(feature = "verify")]
    pub(crate) fn closed_stabilizer_surfaces(&self) -> Vec<Stabilizer> {
        let mut rows = self.rows.clone();
        let constraints = self
            .zx_graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::Port | NodeKind::T))
            .flat_map(|node| [(node.id, Pauli::X), (node.id, Pauli::Z)]);
        let consumed = gaussian_elimination_with_tracking(
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
        let (zx, rows) = self.into_output_correction_rows(outputs)?;
        Ok(rows
            .into_iter()
            .map(|(correction, row)| (correction, zx.materialize_stabilizer(row)))
            .collect())
    }

    /// Select the original output rows once; the consumer owns their physical interpretation.
    pub(crate) fn into_output_correction_rows(
        self,
        outputs: &[IVec3],
    ) -> Result<(ZXGraph, Vec<(OutputCorrectionRow, PauliString)>), RuntimeBasisError> {
        let output_cols = requested_output_columns(&self.zx_graph, outputs)?;
        let protected =
            protected_boundary_columns(&self.zx_graph, &self.prepared_columns, &output_cols);
        let rows = pivot_output_correction_rows(
            self.rows,
            self.readout_coeffs,
            outputs,
            &output_cols,
            &self.input_columns,
            &protected,
        );
        Ok((self.zx_graph, rows))
    }
}

/// Retain every row and its trace, even when all selective queries are zero.
fn project_selective_queries(
    rows: &[PauliString],
    remaining: &mut [SelectiveConstraint],
    fills: &mut [(usize, Pauli)],
) -> Result<Vec<CoeffVec>, StabilizerError> {
    for (column, _) in fills {
        *column = remaining
            .iter()
            .position(|constraint| constraint.col == *column)
            .expect("validated fill has a live selective query");
    }
    let width = remaining.len().saturating_mul(2);
    let rows = rows
        .iter()
        .map(|row| {
            CancellationToken::checkpoint()?;
            let mut projected = CoeffVec::zeros(width);
            for (slot, constraint) in remaining.iter().enumerate() {
                let pauli = row.get(constraint.col);
                projected.set_bit(2 * slot, pauli & Pauli::X);
                projected.set_bit(2 * slot + 1, pauli & Pauli::Z);
            }
            Ok(projected)
        })
        .collect::<Result<Vec<_>, StabilizerError>>()?;
    // Slots stay fixed while restrictions remove entries from `remaining`.
    for (slot, constraint) in remaining.iter_mut().enumerate() {
        constraint.col = slot;
    }
    Ok(rows)
}

fn selective_query_pauli(row: &CoeffVec, slot: usize) -> Pauli {
    match (row.bit(2 * slot), row.bit(2 * slot + 1)) {
        (false, false) => Pauli::I,
        (true, false) => Pauli::X,
        (false, true) => Pauli::Z,
        (true, true) => Pauli::Y,
    }
}

fn restrict_selective_rows<R>(
    rows: &mut Vec<R>,
    coefficients: &mut Vec<CoeffVec>,
    remaining: &mut Vec<SelectiveConstraint>,
    fills: &[(usize, Pauli)],
    get: impl Fn(&R, usize) -> Pauli,
) -> Result<(), StabilizerError>
where
    R: Clone + for<'a> BitXorAssign<&'a R>,
{
    for &(column, chosen) in fills {
        CancellationToken::checkpoint()?;
        remaining.retain(|constraint| constraint.col != column);
        let pivots = gaussian_elimination_with_tracking(
            rows,
            coefficients,
            [(column, chosen)],
            |row, &(column, chosen)| {
                let pauli = get(row, column);
                pauli != Pauli::I && pauli != chosen
            },
            None,
        );
        rows.drain(..pivots);
        coefficients.drain(..pivots);
        normalize_selective_support_with(rows, Some(coefficients), remaining, &get)?;
    }
    Ok(())
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

fn requested_output_columns(
    zx_graph: &ZXGraph,
    outputs: &[IVec3],
) -> Result<Vec<usize>, RuntimeBasisError> {
    outputs
        .iter()
        .copied()
        .map(|output| {
            zx_graph
                .node_at(output)
                .filter(|node| node.is_output_port(zx_graph))
                .map(|node| node.id)
                .ok_or(RuntimeBasisError::UnknownOutput(output))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use glam::ivec3;

    use crate::{BlockGraph, GalleryItem, MeasurementObservable, PauliBasis};

    use super::{RuntimeBasisError, RuntimeStabilizerBasis};

    fn runtime_basis(graph: &BlockGraph) -> RuntimeStabilizerBasis {
        RuntimeStabilizerBasis::from_zx_graph(crate::ZXGraph::try_from(graph).unwrap()).unwrap()
    }

    fn assert_same_runtime(actual: &RuntimeStabilizerBasis, expected: &RuntimeStabilizerBasis) {
        assert_eq!(actual.rows, expected.rows);
        assert_eq!(actual.readout_coeffs, expected.readout_coeffs);
        assert_eq!(actual.input_columns, expected.input_columns);
        assert_eq!(actual.prepared_columns, expected.prepared_columns);
        assert_eq!(actual.selective_kinds, expected.selective_kinds);
        let outputs = actual.zx_graph.output_ports();
        for requested in [&outputs[..], &outputs[..outputs.len().min(1)]] {
            assert_eq!(
                actual.derive_output_correction_surfaces(requested).unwrap(),
                expected
                    .derive_output_correction_surfaces(requested)
                    .unwrap()
            );
        }
    }

    #[test]
    fn reduced_runtime_preserves_ordered_rows_provenance_and_output_phases() {
        for gallery in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::T,
            GalleryItem::TWithPreparedY,
            GalleryItem::THTH,
            GalleryItem::YMemory,
            GalleryItem::Stability,
        ] {
            let source = gallery.build().materialize_root_graph().unwrap();
            let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
            let expected = RuntimeStabilizerBasis::from_zx_graph_unreduced(zx.clone()).unwrap();
            let actual = RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
            assert_same_runtime(&actual, &expected);
            assert_same_runtime(
                &RuntimeStabilizerBasis::for_source_readouts(zx).unwrap(),
                &expected.with_source_readouts().unwrap(),
            );
        }
    }

    #[test]
    fn reduced_runtime_preserves_installed_authority_and_unbounded_admission() {
        let source = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let mut signed = zx.to_signed_external_generator_table();
        signed[0].shift_phase(2);
        zx.stabilizer_phase_basis
            .set(std::sync::Arc::new(
                crate::zx::modular::StabilizerPhaseBasis::new(signed),
            ))
            .unwrap();
        let actual = RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
        let expected = RuntimeStabilizerBasis::from_zx_graph_unreduced(zx).unwrap();
        assert_same_runtime(&actual, &expected);

        // An unsupported synthetic width charges more than the default Boolean
        // budget before returning None, without allocating that many columns.
        let mut unsupported = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        unsupported.total_ids = 1 << 22;
        assert!(
            RuntimeStabilizerBasis::reduced_relation(&unsupported)
                .unwrap()
                .is_none()
        );
        unsupported.total_ids = usize::MAX;
        assert!(matches!(
            RuntimeStabilizerBasis::from_zx_graph(unsupported),
            Err(RuntimeBasisError::Stabilizer(
                crate::StabilizerError::ResourceLimited { .. }
            ))
        ));
    }

    fn assert_projected_selective_fills_match(
        original: Vec<crate::PauliString>,
        constraints: Vec<super::SelectiveConstraint>,
        fills: &[(usize, crate::Pauli)],
    ) -> Result<Vec<crate::PauliString>, String> {
        use super::*;
        let width = original.first().unwrap().len();
        let mut expected = original.clone();
        let mut expected_constraints = constraints.clone();
        let mut expected_traces = (0..original.len())
            .map(|index| CoeffVec::singleton(index, original.len()))
            .collect::<Vec<_>>();
        let mut remaining = constraints;
        let mut mapped_fills = fills.to_vec();
        let mut projected =
            project_selective_queries(&original, &mut remaining, &mut mapped_fills).unwrap();
        assert_eq!(
            projected.len(),
            original.len(),
            "zero-query rows are retained"
        );
        let mut traces = expected_traces.clone();
        for (original_fill, mapped_fill) in fills.iter().zip(&mapped_fills) {
            let expected_result = restrict_selective_rows(
                &mut expected,
                &mut expected_traces,
                &mut expected_constraints,
                std::slice::from_ref(original_fill),
                PauliString::get,
            )
            .map_err(|error| error.to_string());
            let actual_result = restrict_selective_rows(
                &mut projected,
                &mut traces,
                &mut remaining,
                std::slice::from_ref(mapped_fill),
                selective_query_pauli,
            )
            .map_err(|error| error.to_string());
            assert_eq!(actual_result, expected_result, "same original error sites");
            assert_eq!(
                traces, expected_traces,
                "ordered input traces after each fill and normalization"
            );
            actual_result?;
        }
        assert_eq!(
            remaining
                .iter()
                .map(|c| (c.pos, c.kind, c.forbidden))
                .collect::<Vec<_>>(),
            expected_constraints
                .iter()
                .map(|c| (c.pos, c.kind, c.forbidden))
                .collect::<Vec<_>>()
        );
        let actual = traces
            .iter()
            .map(|trace| {
                trace
                    .iter_ones()
                    .fold(PauliString::new(width), |mut row, index| {
                        row ^= &original[index];
                        row
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "original row order and hidden support");
        Ok(actual)
    }

    #[test]
    fn projected_batch_keeps_zero_query_rows_and_private_modes() {
        use super::*;
        let rows = vec![
            PauliString::from_terms(6, [(0, Pauli::X), (2, Pauli::X)]),
            PauliString::from_terms(6, [(0, Pauli::X), (3, Pauli::X)]),
            PauliString::from_terms(6, [(1, Pauli::X), (4, Pauli::X)]),
            PauliString::from_terms(6, [(1, Pauli::Y), (4, Pauli::Z)]),
            PauliString::single(6, 5, Pauli::Z),
        ];
        let constraints = [0, 1]
            .map(|col| SelectiveConstraint {
                pos: ivec3(col as i32, 0, 0),
                kind: crate::SelectiveKind::XY,
                col,
                forbidden: Pauli::Z,
            })
            .to_vec();
        let actual = assert_projected_selective_fills_match(
            rows,
            constraints,
            &[(0, Pauli::Y), (1, Pauli::X)],
        )
        .unwrap();
        assert!(actual.contains(&PauliString::from_terms(6, [(2, Pauli::X), (3, Pauli::X)])));
        assert!(actual.contains(&PauliString::single(6, 5, Pauli::Z)));
    }

    #[test]
    fn projected_batch_matches_all_small_queries_and_fill_orders() {
        use super::*;
        let paulis = [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y];
        let constraints = vec![
            SelectiveConstraint {
                pos: ivec3(2, 0, 0),
                kind: crate::SelectiveKind::XY,
                col: 2,
                forbidden: Pauli::Z,
            },
            SelectiveConstraint {
                pos: ivec3(0, 0, 0),
                kind: crate::SelectiveKind::XZ,
                col: 0,
                forbidden: Pauli::Y,
            },
        ];
        for encoded in 0..256 {
            let rows = vec![
                PauliString::from_terms(
                    6,
                    [
                        (0, paulis[encoded & 3]),
                        (2, paulis[(encoded >> 2) & 3]),
                        (3, Pauli::X),
                    ],
                ),
                PauliString::from_terms(
                    6,
                    [
                        (0, paulis[(encoded >> 4) & 3]),
                        (2, paulis[(encoded >> 6) & 3]),
                        (4, Pauli::Z),
                    ],
                ),
                PauliString::single(6, 5, Pauli::Y),
            ];
            for choices in 0..4 {
                let mut fills = [
                    (0, [Pauli::X, Pauli::Z][choices & 1]),
                    (2, [Pauli::X, Pauli::Y][choices >> 1]),
                ];
                for _ in 0..2 {
                    let _ = assert_projected_selective_fills_match(
                        rows.clone(),
                        constraints.clone(),
                        &fills,
                    );
                    fills.reverse();
                }
            }
        }
    }

    #[test]
    fn projected_batch_keeps_unfilled_queries_and_original_error_sites() {
        use super::*;
        let rows = vec![
            PauliString::from_terms(4, [(0, Pauli::X), (1, Pauli::X)]),
            PauliString::from_terms(4, [(0, Pauli::X), (1, Pauli::Z)]),
        ];
        let constraints = vec![
            SelectiveConstraint {
                pos: ivec3(1, 0, 0),
                kind: crate::SelectiveKind::XZ,
                col: 1,
                forbidden: Pauli::Y,
            },
            SelectiveConstraint {
                pos: ivec3(0, 0, 0),
                kind: crate::SelectiveKind::XY,
                col: 0,
                forbidden: Pauli::Z,
            },
            SelectiveConstraint {
                pos: ivec3(2, 0, 0),
                kind: crate::SelectiveKind::XY,
                col: 2,
                forbidden: Pauli::Z,
            },
        ];
        // The first fill creates forbidden Y on unrequested original node one.
        assert_projected_selective_fills_match(rows, constraints, &[(0, Pauli::Y), (2, Pauli::X)])
            .expect_err("an unfilled selective still requires normalization");
    }

    #[test]
    fn reduced_runtime_preserves_batch_fills_and_original_coordinates() {
        use super::*;
        for (private_modes, source_coordinates) in [(0, true), (64, false), (64, true)] {
            let mut source = GalleryItem::THTH.build().materialize_root_graph().unwrap();
            // Independent internal modes widen equality storage without adding queries.
            for index in 0..private_modes {
                source.add_block(crate::Block::new(
                    ivec3(1000 + 2 * index, 1000, 0),
                    crate::BlockKind::Cube(crate::CubeKind::ZXZ),
                ));
            }
            let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
            let actual = RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
            let expected = RuntimeStabilizerBasis::from_zx_graph_unreduced(zx).unwrap();
            let (actual, expected) = if source_coordinates {
                (
                    actual.with_source_readouts().unwrap(),
                    expected.with_source_readouts().unwrap(),
                )
            } else {
                (
                    actual.with_t_nodes_as_ports(),
                    expected.with_t_nodes_as_ports(),
                )
            };
            let sites = collect_selective_constraints(&actual.zx_graph);
            assert_eq!(sites.len(), 2);
            if private_modes != 0 {
                assert!(
                    (2 * sites.len()).div_ceil(64)
                        < actual
                            .coordinates
                            .as_ref()
                            .unwrap()
                            .class_count()
                            .div_ceil(64)
                );
            }
            for choices in 0..4 {
                let mut fills = sites
                    .iter()
                    .enumerate()
                    .map(|(index, site)| {
                        (
                            site.pos,
                            if choices & (1 << index) == 0 {
                                site.kind.pauli_if_false()
                            } else {
                                site.kind.pauli_if_true()
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                for _ in 0..2 {
                    assert_same_runtime(
                        &actual.apply_selective_fills(&fills).unwrap(),
                        &expected.apply_selective_fills(&fills).unwrap(),
                    );
                    let mut actual = actual.clone();
                    let mut expected = expected.clone();
                    for &(pos, axis) in &fills {
                        actual = actual.apply_selective_fill(pos, axis).unwrap();
                        expected = expected.apply_selective_fill(pos, axis).unwrap();
                        assert_same_runtime(&actual, &expected);
                    }
                    fills.reverse();
                }
            }
        }
    }

    #[test]
    fn projected_batch_trace_width_and_empty_projection_preserve_hidden_rows() {
        use super::*;
        let mut rows = (0..65)
            .map(|index| PauliString::single(66, index + 1, Pauli::X))
            .collect::<Vec<_>>();
        rows[2] = PauliString::from_terms(66, [(0, Pauli::Z), (3, Pauli::X)]);
        rows[64] = PauliString::from_terms(66, [(0, Pauli::Z), (65, Pauli::Z)]);
        let constraints = vec![SelectiveConstraint {
            pos: ivec3(0, 0, 0),
            kind: crate::SelectiveKind::XY,
            col: 0,
            forbidden: Pauli::Z,
        }];
        // One payload word, two trace words; trace2 XOR trace64 must retain bit64.
        let actual =
            assert_projected_selective_fills_match(rows, constraints, &[(0, Pauli::X)]).unwrap();
        assert!(actual.contains(&PauliString::from_terms(
            66,
            [(3, Pauli::X), (65, Pauli::Z)]
        )));
        assert_eq!(actual.len(), 64);
        let rows = vec![PauliString::new(4), PauliString::single(4, 2, Pauli::Y)];
        assert_eq!(
            assert_projected_selective_fills_match(rows.clone(), Vec::new(), &[]).unwrap(),
            rows
        );
    }

    #[test]
    fn projected_batch_normalizes_unfilled_queries_before_a_second_batch() {
        use super::*;
        let rows = vec![
            PauliString::from_terms(5, [(0, Pauli::Z), (1, Pauli::X), (2, Pauli::X)]),
            PauliString::from_terms(5, [(0, Pauli::Z), (1, Pauli::Z), (3, Pauli::Z)]),
            PauliString::from_terms(5, [(1, Pauli::X), (4, Pauli::Y)]),
            PauliString::single(5, 1, Pauli::Z),
        ];
        let future = SelectiveConstraint {
            pos: ivec3(1, 0, 0),
            kind: crate::SelectiveKind::XZ,
            col: 1,
            forbidden: Pauli::Y,
        };
        let constraints = vec![
            future,
            SelectiveConstraint {
                pos: ivec3(0, 0, 0),
                kind: crate::SelectiveKind::XY,
                col: 0,
                forbidden: Pauli::Z,
            },
        ];
        let first =
            assert_projected_selective_fills_match(rows, constraints, &[(0, Pauli::X)]).unwrap();
        assert!(first.iter().all(|row| row.get(1) != Pauli::Y));
        // Rebuild the new query map from original ids after full replay.
        assert_projected_selective_fills_match(first, vec![future], &[(1, Pauli::Z)]).unwrap();
    }

    #[test]
    fn fill_traces_compose_input_rows_without_relabeling_source_provenance() {
        let source = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let mut basis = RuntimeStabilizerBasis::from_zx_graph(zx).unwrap();
        assert!(basis.rows.len() >= 2);
        let source_width = basis.rows.len() + 3;
        basis.readout_coeffs = basis
            .rows
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let mut value = super::CoeffVec::singleton(index + 2, source_width);
                value.set_bit(0, index % 2 == 0);
                value
            })
            .collect();
        let mut combined = super::CoeffVec::singleton(0, basis.rows.len());
        combined.set_bit(1, true);
        let traces = vec![
            super::CoeffVec::singleton(1, basis.rows.len()),
            combined,
            super::CoeffVec::zeros(basis.rows.len()),
        ];
        let (rows, coefficients) = basis.replay_fill_traces(&traces).unwrap();
        assert_eq!(rows[0], basis.rows[1]);
        assert_eq!(coefficients[0], basis.readout_coeffs[1]);
        assert_eq!(rows[1], &basis.rows[0] ^ &basis.rows[1]);
        let mut expected = basis.readout_coeffs[0].clone();
        expected.xor_assign(&basis.readout_coeffs[1]);
        assert_eq!(coefficients[1], expected);
        assert_eq!(rows[2], crate::PauliString::new(basis.zx_graph.total_ids()));
        assert_eq!(coefficients[2], super::CoeffVec::zeros(source_width));
        // Zero support is not a reason to discard a distinct readable recipe.
        basis.rows[1] = basis.rows[0].clone();
        let (rows, coefficients) = basis.replay_fill_traces(&traces[1..2]).unwrap();
        assert_eq!(rows[0], crate::PauliString::new(basis.zx_graph.total_ids()));
        assert_eq!(coefficients[0], expected);
        assert!(!coefficients[0].to_indices().is_empty());
    }

    #[test]
    fn compact_runtime_y_fill_preserves_prepared_sectors_and_phase_interpretation() {
        let source = GalleryItem::T.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let actual = RuntimeStabilizerBasis::for_source_readouts(zx.clone()).unwrap();
        let expected = RuntimeStabilizerBasis::from_zx_graph_unreduced(zx)
            .unwrap()
            .with_source_readouts()
            .unwrap();
        let site = super::collect_selective_constraints(actual.zx_graph())[0];
        assert_eq!(site.kind, crate::SelectiveKind::XY);
        for chosen in [PauliBasis::X, PauliBasis::Y] {
            assert_same_runtime(
                &actual.apply_selective_fill(site.pos, chosen).unwrap(),
                &expected.apply_selective_fill(site.pos, chosen).unwrap(),
            );
        }
    }

    #[test]
    fn supplied_runtime_rows_keep_literal_order_and_dense_fill_ownership() {
        let source = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let mut generators = source.stabilizers().unwrap();
        generators.reverse();
        generators.push(generators[0].clone());
        let mut zero = generators[0].clone();
        zero.stabilizer.paulis = crate::PauliString::new(zx.total_ids());
        generators.push(zero);
        // Caller presentation is literal, even when it violates graph aliases.
        generators[0].stabilizer.paulis =
            crate::PauliString::single(zx.total_ids(), 0, crate::Pauli::Y);
        let expected = generators
            .iter()
            .map(|generator| generator.stabilizer.paulis.clone())
            .collect::<Vec<_>>();
        let presentation = crate::StabilizerGenerators::new(zx, generators);
        let actual = RuntimeStabilizerBasis::from_generators(&presentation);
        assert!(actual.coordinates.is_none());
        assert_eq!(actual.generators(), expected);
        for (index, coefficient) in actual.readout_coeffs.iter().enumerate() {
            assert_eq!(coefficient.to_indices(), [index]);
        }
    }

    #[test]
    fn runtime_fill_cancellation_keeps_the_original_rows() {
        let source = GalleryItem::THTH.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let actual = RuntimeStabilizerBasis::for_source_readouts(zx).unwrap();
        assert!(actual.coordinates.is_some());
        let original = actual.generators().to_vec();
        let site = super::collect_selective_constraints(actual.zx_graph())[0];
        let token = crate::CancellationToken::new();
        token.cancel();
        assert!(matches!(
            token.scope(|| actual.apply_selective_fill(site.pos, site.kind.pauli_if_false())),
            Err(RuntimeBasisError::Stabilizer(
                crate::StabilizerError::Cancelled(_)
            ))
        ));
        assert_eq!(actual.generators(), original);
    }

    #[test]
    fn reduced_runtime_propagates_cancellation_without_retry() {
        let source = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let token = crate::CancellationToken::new();
        token.cancel();
        assert!(matches!(
            token.scope(|| RuntimeStabilizerBasis::from_zx_graph(zx)),
            Err(RuntimeBasisError::Stabilizer(
                crate::StabilizerError::Cancelled(_)
            ))
        ));
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
        let filled = RuntimeStabilizerBasis::from_zx_graph(zx)
            .unwrap()
            .apply_selective_fill(ivec3(0, 0, 1), PauliBasis::Z)
            .unwrap();

        assert_eq!(
            filled
                .zx_graph
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement,
            Some(MeasurementObservable::Concrete(PauliBasis::Z))
        );
    }

    #[test]
    fn apply_selective_fill_rejects_non_clifford_graphs() {
        for (item, site, chosen) in [
            (GalleryItem::T, ivec3(1, 0, 2), PauliBasis::X),
            (GalleryItem::TWithPreparedY, ivec3(2, 1, 2), PauliBasis::Z),
        ] {
            let graph = item.build().materialize_root_graph().unwrap();
            assert!(matches!(
                runtime_basis(&graph).apply_selective_fill(site, chosen),
                Err(RuntimeBasisError::NonCliffordFillUnsupported)
            ));
        }
    }

    #[cfg(feature = "verify")]
    #[test]
    fn verifier_output_queries_preserve_parallel_edge_materialization() {
        use crate::verify::{
            MeasurementKey, solve_evolved_correction, stabilizer_supports_measurement,
        };
        use crate::zx::CoeffVec;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        fn oracle(
            basis: &RuntimeStabilizerBasis,
            outputs: &[glam::IVec3],
            keys: &[&MeasurementKey],
        ) -> (Vec<CoeffVec>, crate::SymbolicOutputCorrection) {
            let mut constraints = Vec::new();
            let mut masks = Vec::new();
            for (constraint, surface) in basis.derive_output_correction_surfaces(outputs).unwrap() {
                let mut mask = CoeffVec::zeros(keys.len());
                for (index, key) in keys.iter().enumerate() {
                    mask.set_bit(index, stabilizer_supports_measurement(&surface, key));
                }
                if !mask.is_zero() {
                    constraints.push(constraint);
                    masks.push(mask);
                }
            }
            (
                masks,
                crate::solve_output_correction_symbolic(outputs, &constraints).unwrap(),
            )
        }

        for hadamard in [false, true] {
            let nodes = vec![
                crate::ZXNode::new(0, glam::ivec3(0, 0, 0), crate::NodeKind::Port),
                crate::ZXNode::new(1, glam::ivec3(0, 0, 1), crate::NodeKind::Port),
            ];
            let edges = [(0, 1), (1, 0), (0, 1), (1, 0)]
                .into_iter()
                .enumerate()
                .map(|(index, (n1, n2))| crate::ZXEdge {
                    n1,
                    n2,
                    id: 2 + index,
                    hadamard,
                })
                .collect::<Vec<_>>();
            let zx = crate::ZXGraph {
                adjacency: super::super::graph::CsrAdjacency::from_edges(
                    nodes.len(),
                    &edges
                        .iter()
                        .map(|edge| (edge.n1, edge.n2, edge.id))
                        .collect::<Vec<_>>(),
                ),
                pos_to_node: nodes.iter().map(|node| (node.pos, node.id)).collect(),
                total_ids: nodes.len() + edges.len(),
                nodes,
                edges,
                action_graph: crate::ActionDag::default(),
                cross_incident: Default::default(),
                stabilizer_phase_basis: Default::default(),
            };
            let outputs = zx.output_ports();
            assert_eq!(outputs.len(), 1);
            let keys = [
                MeasurementKey::Node {
                    pos: outputs[0],
                    pauli: crate::Pauli::X,
                },
                MeasurementKey::Edge {
                    src: zx.nodes[0].pos,
                    dst: zx.nodes[1].pos,
                    pauli: crate::Pauli::X,
                },
                MeasurementKey::Edge {
                    src: zx.nodes[1].pos,
                    dst: zx.nodes[0].pos,
                    pauli: crate::Pauli::X,
                },
            ];
            let refs = keys.iter().collect::<Vec<_>>();
            let basis = RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
            assert_eq!(
                solve_evolved_correction(&basis, &outputs, &refs).unwrap(),
                oracle(&basis, &outputs, &refs)
            );

            // The latest forward column is I, but an earlier forward X remains
            // in the physical map. Its reverse reference is X, so maps are valid
            // under the existing debug assertion despite synthetic pair mismatch.
            for last_reverse in [crate::Pauli::X, crate::Pauli::I] {
                let row = crate::PauliString::from_terms(
                    zx.total_ids(),
                    [
                        (1, crate::Pauli::X),
                        (2, crate::Pauli::X),
                        (3, crate::Pauli::X),
                        (5, last_reverse),
                    ],
                );
                let generators =
                    crate::StabilizerGenerators::completed(zx.clone(), Vec::new(), [row]);
                let basis = RuntimeStabilizerBasis::from_generators(&generators);
                let expected = catch_unwind(AssertUnwindSafe(|| oracle(&basis, &outputs, &refs)));
                let actual = catch_unwind(AssertUnwindSafe(|| {
                    solve_evolved_correction(&basis, &outputs, &refs).unwrap()
                }));
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert!(
                            expected.0.iter().any(|mask| mask.bit(1)),
                            "earlier nonzero edge survives"
                        );
                        assert_eq!(actual, expected);
                    }
                    (Err(_), Err(_)) => assert_eq!(last_reverse, crate::Pauli::I),
                    (Ok(_), Err(_)) => {
                        panic!("query path hid the signed materializer's debug failure")
                    }
                    (Err(_), Ok(_)) => panic!("query path introduced a failure"),
                }
            }
        }
        let graph = crate::BlockGraph::from_text(
            "BLOG 1.0\nmodule main {\nin q: data = 0\nout r: data = 1\n0: Port [0,0,0] role=input\n1: Port [0,0,1] role=output\n0 -> +Z\n}\n",
        )
        .unwrap()
        .materialize_root_graph()
        .unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let outputs = zx.output_ports();
        let keys = [
            MeasurementKey::Node {
                pos: outputs[0],
                pauli: crate::Pauli::X,
            },
            MeasurementKey::Edge {
                src: zx.nodes[0].pos,
                dst: zx.nodes[1].pos,
                pauli: crate::Pauli::X,
            },
            MeasurementKey::Edge {
                src: zx.nodes[1].pos,
                dst: zx.nodes[0].pos,
                pauli: crate::Pauli::X,
            },
        ];
        let refs = keys.iter().collect::<Vec<_>>();
        for (width, paired) in [
            (zx.total_ids(), true),
            (zx.total_ids(), false),
            (zx.total_ids() - 1, false),
            (zx.total_ids() + 1, true),
        ] {
            let mut row = crate::PauliString::new(width);
            row.set(0, crate::Pauli::X);
            row.set(1, crate::Pauli::X);
            let forward = zx.edge_id(0, 1).unwrap();
            if forward < width {
                row.set(forward, crate::Pauli::X);
            }
            if paired {
                row.set(zx.edge_id(1, 0).unwrap(), crate::Pauli::X);
            }
            let basis = RuntimeStabilizerBasis::new(zx.clone(), vec![row], 1);
            let expected = catch_unwind(AssertUnwindSafe(|| oracle(&basis, &outputs, &refs)));
            let actual = catch_unwind(AssertUnwindSafe(|| {
                solve_evolved_correction(&basis, &outputs, &refs).unwrap()
            }));
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
                (Err(_), Err(_)) => {}
                (Ok(_), Err(_)) => panic!("query hid a width or paired-arm failure"),
                (Err(_), Ok(_)) => panic!("query introduced a width or paired-arm failure"),
            }
        }
    }

    #[test]
    fn output_correction_rows_cover_requested_outputs() {
        let graph = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let outputs = [ivec3(0, 0, 3), ivec3(1, 1, 3)];
        let joints = runtime_basis(&graph)
            .derive_output_correction_surfaces(&outputs)
            .unwrap();

        for output in outputs {
            assert!(joints.iter().any(|(joint, _)| {
                joint
                    .output_support
                    .iter()
                    .any(|&(supported, _)| supported == output)
            }));
        }
        assert!(
            joints
                .iter()
                .all(|(joint, _)| !joint.readout_ordinals.is_empty())
        );
    }

    #[test]
    fn output_projection_past_i32_max_layer_is_representable() {
        let graph = GalleryItem::CNOT.build().materialize_root_graph().unwrap();
        let max_z = *graph.spans().unwrap().2.end();
        let shifted = graph
            .shift_positions(ivec3(0, 0, i32::MAX - max_z))
            .unwrap();
        let basis = runtime_basis(&shifted);
        let outputs = basis.zx_graph.output_ports();
        assert!(
            basis
                .zx_graph
                .nodes()
                .iter()
                .any(|node| node.pos.z == i32::MAX)
        );
        assert!(
            !basis
                .derive_output_correction_surfaces(&outputs)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn output_correction_rows_report_unknown_output() {
        let graph = GalleryItem::T.build().materialize_root_graph().unwrap();
        let err = runtime_basis(&graph)
            .derive_output_correction_surfaces(&[ivec3(99, 99, 99)])
            .expect_err("missing output should error");

        assert!(matches!(err, RuntimeBasisError::UnknownOutput(pos) if pos == ivec3(99, 99, 99)));
    }
}
