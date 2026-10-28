//! Basis-evolution helpers for internal-measurement correction.

use glam::IVec3;

use crate::{
    Pauli, PauliBasis, RuntimeBasisError, RuntimeStabilizerBasis, SymbolicOutputCorrection,
    solve_output_correction_symbolic,
};

use crate::zx::CoeffVec;

use super::{MeasurementKey, VerifyLogicalError, stabilizer_supports_measurement};

/// One selective-fill replay step: apply `chosen` at `pos` over `basis` (with
/// the T-port fallback) and return the evolved basis together with the
/// measurement key it records.
pub(crate) fn replay_fill(
    basis: &RuntimeStabilizerBasis,
    pos: IVec3,
    chosen: PauliBasis,
) -> Result<(RuntimeStabilizerBasis, MeasurementKey), VerifyLogicalError> {
    let basis = apply_selective_fill_with_fallback(basis, pos, chosen)?;
    let key = MeasurementKey::Node {
        pos,
        pauli: Pauli::from(chosen),
    };
    Ok((basis, key))
}

/// Applies a selective fill at `pos` in the `chosen` basis, retrying with T
/// nodes demoted to ports if the fill would otherwise be non-Clifford.
fn apply_selective_fill_with_fallback(
    basis: &RuntimeStabilizerBasis,
    pos: IVec3,
    chosen: PauliBasis,
) -> Result<RuntimeStabilizerBasis, VerifyLogicalError> {
    match basis.apply_selective_fill(pos, chosen) {
        Ok(transition) => Ok(transition),
        Err(RuntimeBasisError::NonCliffordFillUnsupported) => basis
            .clone()
            .with_t_nodes_as_ports()
            .apply_selective_fill(pos, chosen)
            .map_err(|err| VerifyLogicalError::RuntimeBasisUpdateFailed { pos, source: err }),
        Err(err) => Err(VerifyLogicalError::RuntimeBasisUpdateFailed { pos, source: err }),
    }
}

/// The evolved-basis terminal solve, retaining support on every measured key.
/// Crossing-node predicates are evaluated after the final row combinations.
/// Closed reference constraints retain their separate signed materialization.
pub(crate) fn solve_evolved_correction(
    basis: &RuntimeStabilizerBasis,
    outputs: &[IVec3],
    keys: &[&MeasurementKey],
) -> Result<(Vec<CoeffVec>, SymbolicOutputCorrection), VerifyLogicalError> {
    let (zx, rows) = basis
        .clone()
        .into_output_correction_rows(outputs)
        .map_err(|_| VerifyLogicalError::OutputSurfaceUnavailable(outputs.to_vec()))?;
    // A physical edge map keeps the last nonzero parallel entry, not simply
    // edge_id's last entry. Preserve the existing materializer in that case.
    let unique_edges = !zx.has_parallel_edges();
    let components = keys
        .iter()
        .map(|key| key.components(&zx).unwrap_or_default())
        .collect::<Vec<_>>();
    let mut constraints = Vec::new();
    let mut supports = Vec::new();
    for (constraint, row) in rows {
        let mut support = CoeffVec::zeros(keys.len());
        if unique_edges
            && row.len() == zx.total_ids()
            && zx.row_has_external_edge_pair_support(&row)
        {
            for (index, components) in components.iter().enumerate() {
                support.set_bit(
                    index,
                    components
                        .iter()
                        .any(|&(column, axis)| row.get(column) & axis),
                );
            }
        } else {
            let surface = zx.materialize_stabilizer(row);
            for (index, key) in keys.iter().enumerate() {
                support.set_bit(index, stabilizer_supports_measurement(&surface, key));
            }
        }
        if !support.is_zero() {
            constraints.push(constraint);
            supports.push(support);
        }
    }
    let symbolic = solve_output_correction_symbolic(outputs, &constraints)?;
    Ok((supports, symbolic))
}

#[cfg(test)]
pub(crate) fn signed_output_correction(
    basis: &RuntimeStabilizerBasis,
    outputs: &[IVec3],
    keys: &[&MeasurementKey],
) -> (Vec<CoeffVec>, SymbolicOutputCorrection) {
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
        solve_output_correction_symbolic(outputs, &constraints).unwrap(),
    )
}

#[cfg(test)]
mod output_query_tests {
    use super::*;

    fn all_keys(zx: &crate::ZXGraph) -> Vec<MeasurementKey> {
        let mut keys = Vec::new();
        for node in zx.nodes() {
            for pauli in [Pauli::X, Pauli::Z, Pauli::Y] {
                keys.push(MeasurementKey::Node {
                    pos: node.pos,
                    pauli,
                });
            }
        }
        for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
            let left = zx.nodes()[edge.n1].pos;
            let right = zx.nodes()[edge.n2].pos;
            for (src, dst) in [(left, right), (right, left)] {
                for pauli in [Pauli::X, Pauli::Z, Pauli::Y] {
                    keys.push(MeasurementKey::Edge { src, dst, pauli });
                }
            }
        }
        let absent = glam::ivec3(i32::MAX, 42, 17);
        assert!(zx.node_at(absent).is_none());
        keys.push(MeasurementKey::Node {
            pos: absent,
            pauli: Pauli::Y,
        });
        keys.push(MeasurementKey::Edge {
            src: absent,
            dst: glam::ivec3(i32::MAX - 1, 42, 17),
            pauli: Pauli::Y,
        });
        if let Some(first) = zx.nodes().first()
            && let Some(other) = zx
                .nodes()
                .iter()
                .find(|other| other.id != first.id && zx.edge_id(first.id, other.id).is_none())
        {
            keys.push(MeasurementKey::Edge {
                src: first.pos,
                dst: other.pos,
                pauli: Pauli::X,
            });
        }
        if let Some(first) = keys.first().cloned() {
            keys.push(first); // The supplied key order may contain duplicates.
        }
        keys
    }

    fn assert_matches_signed(
        basis: &RuntimeStabilizerBasis,
        outputs: &[IVec3],
        keys: &[MeasurementKey],
    ) -> usize {
        let refs = keys.iter().collect::<Vec<_>>();
        let (expected_masks, expected) = signed_output_correction(basis, outputs, &refs);
        let (actual_masks, actual) = solve_evolved_correction(basis, outputs, &refs).unwrap();
        assert_eq!(
            actual_masks, expected_masks,
            "ordered physical support masks"
        );
        assert_eq!(
            actual, expected,
            "exact symbolic frames and consistency equations"
        );
        actual_masks.len()
    }

    #[test]
    fn output_queries_match_signed_gallery_surfaces_and_key_order() {
        for gallery in [
            crate::GalleryItem::CNOT,
            crate::GalleryItem::CZSpatialH,
            crate::GalleryItem::CZTemporalH,
            crate::GalleryItem::S,
            crate::GalleryItem::T,
            crate::GalleryItem::TWithPreparedY,
            crate::GalleryItem::THTH,
            crate::GalleryItem::YMemory,
            crate::GalleryItem::Stability,
        ] {
            let graph = gallery.build().materialize_root_graph().unwrap();
            let zx = crate::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
            let basis = RuntimeStabilizerBasis::from_zx_graph(zx)
                .unwrap()
                .with_t_nodes_as_ports();
            let outputs = basis.zx_graph().output_ports();
            let mut keys = all_keys(basis.zx_graph());
            assert_matches_signed(&basis, &outputs, &keys);
            keys.reverse();
            assert_matches_signed(&basis, &outputs[..outputs.len().min(1)], &keys);
            assert_eq!(assert_matches_signed(&basis, &outputs, &[]), 0);
        }
    }

    #[test]
    fn output_queries_preserve_batched_and_sequential_selective_fills() {
        let graph = crate::GalleryItem::THTH
            .build()
            .materialize_root_graph()
            .unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let basis = RuntimeStabilizerBasis::from_zx_graph(zx)
            .unwrap()
            .with_t_nodes_as_ports();
        let sites = basis
            .zx_graph()
            .nodes()
            .iter()
            .filter_map(|node| {
                let crate::NodeKind::Selective(kind) = node.kind else {
                    return None;
                };
                Some((node.pos, kind))
            })
            .collect::<Vec<_>>();
        assert_eq!(sites.len(), 2, "THTH has two adaptive choices");
        let outputs = basis.zx_graph().output_ports();
        for choices in 0..4 {
            let mut fills = sites
                .iter()
                .enumerate()
                .map(|(index, &(pos, kind))| {
                    (
                        pos,
                        if choices & (1 << index) == 0 {
                            kind.pauli_if_false()
                        } else {
                            kind.pauli_if_true()
                        },
                    )
                })
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let batch = basis.apply_selective_fills(&fills).unwrap();
                assert_matches_signed(&batch, &outputs, &all_keys(batch.zx_graph()));
                let mut sequential = basis.clone();
                for &(pos, chosen) in &fills {
                    sequential = sequential.apply_selective_fill(pos, chosen).unwrap();
                    assert_matches_signed(&sequential, &outputs, &all_keys(sequential.zx_graph()));
                }
                fills.reverse();
            }
        }
    }

    #[test]
    fn output_queries_keep_private_support_without_readable_provenance() {
        let graph = crate::GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .unwrap();
        let zx = crate::ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let relation = RuntimeStabilizerBasis::from_zx_graph(zx.clone()).unwrap();
        let generators = crate::StabilizerGenerators::completed(
            zx,
            Vec::new(),
            relation.generators().iter().cloned(),
        );
        assert!(generators.generators.is_empty());
        let basis = RuntimeStabilizerBasis::from_generators(&generators);
        let outputs = basis.zx_graph().output_ports();
        let derived = basis.derive_output_correction_surfaces(&outputs).unwrap();
        assert!(!derived.is_empty());
        assert!(
            derived
                .iter()
                .all(|(row, _)| row.readout_ordinals.is_empty())
        );
        assert!(assert_matches_signed(&basis, &outputs, &all_keys(basis.zx_graph())) > 0);
    }
}
