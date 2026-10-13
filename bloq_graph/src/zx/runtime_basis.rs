//! Runtime-selectable stabilizer bases and their output-correction surfaces.

use std::collections::HashMap;

use glam::IVec3;
use thiserror::Error;

use super::graph::insert_selective_fill;
use super::output_correction::{OutputCorrectionRow, pivot_output_correction_rows};
use super::stabilizer::{
    CoeffVec, collect_selective_constraints, gaussian_elimination_with_tracking,
    normalize_selective_support_in_place, reduce_to_basis,
};
use super::{NodeKind, Stabilizer, StabilizerError, StabilizerGenerators, ZXError, ZXGraph};
use crate::{Pauli, PauliBasis, PauliString, SelectiveKind};

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
    /// support.
    pub fn from_zx_graph(zx_graph: ZXGraph) -> Result<Self, RuntimeBasisError> {
        let mut rows = reduce_to_basis(&zx_graph.to_external_generator_table(), zx_graph.total_ids);
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
        let mut basis = Self::from_zx_graph(zx_graph)?.with_t_nodes_as_ports();
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
        let mut basis = Self::from_zx_graph(zx)?.with_t_nodes_as_ports();
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

        let mut rows = self.rows.clone();
        let mut readout_coeffs = self.readout_coeffs.clone();
        let mut remaining = collect_selective_constraints(&self.zx_graph);
        for &(pos, chosen) in fills {
            let col = self
                .zx_graph
                .node_at(pos)
                .expect("validated selective has a ZX node")
                .id;
            remaining.retain(|constraint| constraint.pos != pos);
            let pivots = gaussian_elimination_with_tracking(
                &mut rows,
                &mut readout_coeffs,
                [(col, Pauli::from(chosen))],
                |row, &(col, chosen)| {
                    let pauli = row.get(col);
                    pauli != Pauli::I && pauli != chosen
                },
                None,
            );
            rows.drain(..pivots);
            readout_coeffs.drain(..pivots);
            normalize_selective_support_in_place(&mut rows, Some(&mut readout_coeffs), &remaining)?;
        }

        Ok(RuntimeStabilizerBasis {
            input_columns: self.input_columns.clone(),
            prepared_columns: self.prepared_columns.clone(),
            zx_graph: self.zx_graph.fill_ports(&port_map)?,
            rows,
            readout_coeffs,
            selective_kinds: remaining
                .into_iter()
                .map(|constraint| (constraint.pos, constraint.kind))
                .collect(),
        })
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
        let output_cols = requested_output_columns(&self.zx_graph, outputs)?;
        let protected =
            protected_boundary_columns(&self.zx_graph, &self.prepared_columns, &output_cols);
        Ok(pivot_output_correction_rows(
            self.rows,
            self.readout_coeffs,
            outputs,
            &output_cols,
            &self.input_columns,
            &protected,
        )
        .into_iter()
        .map(|(correction, row)| (correction, self.zx_graph.materialize_stabilizer(row)))
        .collect())
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

    use crate::{BlockGraph, GalleryItem, MeasurementObservable, PauliBasis};

    use super::{RuntimeBasisError, RuntimeStabilizerBasis};

    fn runtime_basis(graph: &BlockGraph) -> RuntimeStabilizerBasis {
        RuntimeStabilizerBasis::from_zx_graph(crate::ZXGraph::try_from(graph).unwrap()).unwrap()
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
