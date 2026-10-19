//! Exact cap contraction of an already certified, complete signed relation.

use std::collections::HashMap;
use std::sync::Arc;

use glam::IVec3;

use super::graph::insert_selective_fill;
use super::modular::{StabilizerPhaseBasis, try_leading_pivot_elimination_from};
use super::stabilizer::SearchBudget;
use super::{NodeKind, RuntimeBasisError, StabilizerError, ZXGraph};
use crate::{ModuleCertificationLimits, Pauli, PauliBasis, PhasedPauliString};

impl ZXGraph {
    /// Restricts the supplied complete raw relation, including its closed rows.
    /// Source validity and correlated choice reachability belong to the caller's
    /// source certificate; no external table is derived from the filled graph.
    pub(crate) fn restrict_selective_relation_with_limits(
        &self,
        external_basis: &[PhasedPauliString],
        fills: &[(IVec3, PauliBasis)],
        limits: ModuleCertificationLimits,
    ) -> Result<(ZXGraph, Arc<StabilizerPhaseBasis>), RuntimeBasisError> {
        let mut budget = SearchBudget::with_limits(limits);
        // Retain the conservative three-table bound across restriction and handoff.
        let mut checkpoint = || {
            budget.check_matrix(
                external_basis.len().saturating_mul(3),
                self.total_ids(),
                0,
                0,
            )?;
            budget.visit("signed relation restriction states")
        };
        checkpoint()?;
        if external_basis
            .iter()
            .any(|row| row.paulis.len() != self.total_ids() || row.hermitian_sign().is_none())
        {
            return Err(StabilizerError::ComposedPresentationUnsatisfied.into());
        }
        let mut port_map = HashMap::with_capacity(fills.len());
        let mut constraints = Vec::with_capacity(fills.len());
        for &(pos, chosen) in fills {
            checkpoint()?;
            let node = self
                .node_at(pos)
                .ok_or(RuntimeBasisError::UnknownSelective(pos))?;
            let NodeKind::Selective(kind) = node.kind else {
                return Err(RuntimeBasisError::UnknownSelective(pos));
            };
            insert_selective_fill(&mut port_map, pos, kind, chosen)?;
            // Boundary YY has sign -1. An effect Y cap also has sign -1;
            // a preparation Y cap has sign +1. Hadamard signs cancel here.
            let cap_phase = if chosen == PauliBasis::Y
                && self
                    .neighbor_ids(node.id)
                    .iter()
                    .any(|&neighbor| self.nodes()[neighbor].pos.z >= pos.z)
            {
                2
            } else {
                0
            };
            constraints.push((node.id, Pauli::from(chosen), cap_phase));
        }
        constraints.sort_unstable_by_key(|&(column, _, _)| column);

        let mut rows = external_basis.to_vec();
        let mut eliminated = 0;
        for &(column, chosen, _) in &constraints {
            checkpoint()?;
            let forbidden = |row: &PhasedPauliString| {
                let pauli = row.paulis.get(column);
                pauli != Pauli::I && pauli != chosen
            };
            let Some(pivot) = (eliminated..rows.len()).find(|&index| forbidden(&rows[index]))
            else {
                continue;
            };
            rows.swap(eliminated, pivot);
            let (before, pivot_and_after) = rows.split_at_mut(eliminated);
            let (pivot, after) = pivot_and_after.split_first_mut().expect("pivot exists");
            for row in before.iter_mut().chain(after) {
                checkpoint()?;
                if forbidden(row) {
                    row.multiply_assign(pivot);
                }
            }
            eliminated += 1;
        }
        rows.drain(..eliminated);
        for row in &mut rows {
            checkpoint()?;
            for &(column, _, cap_phase) in &constraints {
                if row.paulis.get(column) == Pauli::Y {
                    row.shift_phase(cap_phase);
                }
            }
        }
        let rank = try_leading_pivot_elimination_from(
            &mut rows,
            0,
            |row| {
                row.paulis.iter_support().next().map(|(column, pauli)| {
                    (column, if pauli & Pauli::X { Pauli::X } else { Pauli::Z })
                })
            },
            |row, &(column, axis)| row.paulis.get(column) & axis,
            &mut checkpoint,
        )?;
        if rows[rank..].iter().any(|row| row.phase() != 0) {
            return Err(StabilizerError::ComposedPresentationUnsatisfied.into());
        }
        rows.truncate(rank);

        let mut zx_graph = self.fill_ports(&port_map)?;
        zx_graph.demote_t_nodes_to_ports();
        zx_graph.stabilizer_phase_basis.take();
        checkpoint()?;
        let basis = Arc::new(StabilizerPhaseBasis::new(rows));
        zx_graph
            .stabilizer_phase_basis
            .set(Arc::clone(&basis))
            .expect("the old phase cache was removed");
        Ok((zx_graph, basis))
    }
}

#[cfg(test)]
mod tests {
    use glam::ivec3;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::zx::graph::CsrAdjacency;
    use crate::zx::graph::filled_node_kind;
    use crate::{ActionDag, PauliString, SelectiveKind, ZXEdge, ZXNode};

    fn chain(kinds: [SelectiveKind; 2], spider: NodeKind, hadamards: usize) -> ZXGraph {
        let nodes = vec![
            ZXNode::new(0, ivec3(0, 0, 0), NodeKind::Selective(kinds[0])),
            ZXNode::new(1, ivec3(0, 0, 1), spider),
            ZXNode::new(2, ivec3(0, 0, 2), NodeKind::Selective(kinds[1])),
            // A private direction with no public boundary support.
            ZXNode::new(3, ivec3(2, 0, 0), NodeKind::X),
        ];
        let edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
            .into_iter()
            .enumerate()
            .map(|(index, (n1, n2))| ZXEdge {
                n1,
                n2,
                id: nodes.len() + index,
                hadamard: hadamards & (1 << (index / 2)) != 0,
            })
            .collect::<Vec<_>>();
        ZXGraph {
            adjacency: CsrAdjacency::from_edges(
                nodes.len(),
                &edges
                    .iter()
                    .map(|edge| (edge.n1, edge.n2, edge.id))
                    .collect::<Vec<_>>(),
            ),
            pos_to_node: FxHashMap::from_iter(nodes.iter().map(|node| (node.pos, node.id))),
            total_ids: nodes.len() + edges.len(),
            nodes,
            edges,
            action_graph: ActionDag::from_actions(&[]),
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        }
    }

    fn assert_matches_cap_oracle(
        graph: &ZXGraph,
        source: &[PhasedPauliString],
        fills: &[(IVec3, PauliBasis)],
    ) {
        let (filled, rows) = graph
            .restrict_selective_relation_with_limits(
                source,
                fills,
                ModuleCertificationLimits::default(),
            )
            .unwrap();
        let oracle = graph
            .fill_ports(
                &fills
                    .iter()
                    .map(|&(pos, chosen)| (pos, filled_node_kind(chosen)))
                    .collect::<HashMap<_, _>>(),
            )
            .unwrap();
        let expected = oracle.to_external_generator_table_with_signed().1;
        assert_eq!(
            rows.rows(),
            expected,
            "nodes={:?}, fills={fills:?}",
            graph.nodes,
        );
        assert!(
            rows.rows().iter().any(|row| row.paulis.get(3) == Pauli::Z),
            "private direction was lost"
        );
        for row in rows.rows() {
            assert!(filled.contains_stabilizer_support(&row.paulis));
            assert_eq!(
                filled.stabilizer_row_phases(std::slice::from_ref(&row.paulis)),
                [row.phase()]
            );
        }
    }

    #[test]
    fn signed_restriction_exhaustively_matches_local_caps() {
        for left_kind in [SelectiveKind::XY, SelectiveKind::XZ, SelectiveKind::YZ] {
            for right_kind in [SelectiveKind::XY, SelectiveKind::XZ, SelectiveKind::YZ] {
                for spider in [NodeKind::X, NodeKind::Z] {
                    for hadamards in 0..4 {
                        let graph = chain([left_kind, right_kind], spider, hadamards);
                        let source = graph.to_external_generator_table_with_signed().1;
                        for left in [left_kind.pauli_if_false(), left_kind.pauli_if_true()] {
                            for right in [right_kind.pauli_if_false(), right_kind.pauli_if_true()] {
                                for selected in 0..4 {
                                    let fills =
                                        [(graph.nodes[0].pos, left), (graph.nodes[2].pos, right)]
                                            .into_iter()
                                            .enumerate()
                                            .filter_map(|(index, fill)| {
                                                (selected & (1 << index) != 0).then_some(fill)
                                            })
                                            .collect::<Vec<_>>();
                                    assert_matches_cap_oracle(&graph, &source, &fills);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn signed_restriction_uses_supplied_signs_and_preserves_private_rows() {
        let graph = chain([SelectiveKind::XY; 2], NodeKind::Z, 1);
        let mut source = graph.to_external_generator_table_with_signed().1;
        let private = source
            .iter()
            .position(|row| row.paulis.get(3) == Pauli::Z)
            .unwrap();
        source[private].shift_phase(2);
        // The input certificate need not arrive in its original row presentation.
        let pivot = source[0].clone();
        source[private].multiply_assign(&pivot);
        source.reverse();
        let (filled, rows) = graph
            .restrict_selective_relation_with_limits(
                &source,
                &[(graph.nodes[0].pos, PauliBasis::Y)],
                ModuleCertificationLimits::default(),
            )
            .unwrap();
        let private = rows
            .rows()
            .iter()
            .find(|row| row.paulis.get(3) == Pauli::Z)
            .unwrap();
        assert_eq!(private.phase(), 2);
        assert_eq!(
            filled.stabilizer_row_phases(std::slice::from_ref(&private.paulis)),
            [2]
        );
        assert_eq!(
            filled.external_stabilizer_row_phases(std::slice::from_ref(&private.paulis)),
            [0],
            "local geometry must not replace the supplied sign"
        );
    }

    #[test]
    fn signed_restriction_matches_reversed_y_caps_and_rejects_spatial_source_shapes() {
        for kind in [SelectiveKind::XY, SelectiveKind::YZ] {
            for spider in [NodeKind::X, NodeKind::Z] {
                for hadamards in 0..4 {
                    let mut graph = chain([kind; 2], spider, hadamards);
                    for node in &mut graph.nodes {
                        node.pos.z = -node.pos.z;
                    }
                    graph.pos_to_node =
                        FxHashMap::from_iter(graph.nodes.iter().map(|node| (node.pos, node.id)));
                    let source = graph.to_external_generator_table_with_signed().1;
                    let left = (graph.nodes[0].pos, PauliBasis::Y);
                    let right = (graph.nodes[2].pos, PauliBasis::Y);
                    for fills in [vec![left], vec![right], vec![left, right]] {
                        assert_matches_cap_oracle(&graph, &source, &fills);
                    }
                }
            }
        }
        // Certified Selective/Y caps have one temporal neighbor; equal-z
        // placements are rejected before the restriction API is reached.
        for direction in [ivec3(1, 0, 0), ivec3(0, 1, 0)] {
            let mut graph = chain([SelectiveKind::XY; 2], NodeKind::X, 0);
            for node in &mut graph.nodes[..3] {
                node.pos = direction * node.id as i32;
            }
            graph.nodes[3].pos = ivec3(3, 3, 0);
            graph.pos_to_node =
                FxHashMap::from_iter(graph.nodes.iter().map(|node| (node.pos, node.id)));
            assert!(matches!(
                crate::BlockGraph::try_from(&graph),
                Err(super::super::ZXError::SpecialNodeNotTimeLike { .. })
            ));
        }
    }

    #[test]
    fn signed_restriction_demotes_prepared_t_resources_without_rederivation() {
        let mut graph = chain([SelectiveKind::XY; 2], NodeKind::X, 3);
        graph.nodes[2].kind = NodeKind::T;
        let source = graph.to_external_generator_table_with_signed().1;
        for fills in [vec![], vec![(graph.nodes[0].pos, PauliBasis::Y)]] {
            let (filled, _) = graph
                .restrict_selective_relation_with_limits(
                    &source,
                    &fills,
                    ModuleCertificationLimits::default(),
                )
                .unwrap();
            assert_eq!(filled.nodes[2].kind, NodeKind::Port);
            assert!(filled.is_clifford_computation());
        }
    }

    #[test]
    fn signed_restriction_preserves_signed_queries_at_tight_matrix_limit() {
        let graph = chain([SelectiveKind::XY; 2], NodeKind::Z, 1);
        let mut source = graph.to_external_generator_table_with_signed().1;
        source
            .iter_mut()
            .find(|row| row.paulis.get(3) == Pauli::Z)
            .unwrap()
            .shift_phase(2);
        let fills = [(graph.nodes[0].pos, PauliBasis::Y)];
        // Three simultaneous Pauli tables of the three-row source need 144 words.
        let limits = ModuleCertificationLimits {
            max_matrix_words: 144,
            ..ModuleCertificationLimits::DEFAULT
        };
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(
                &source,
                &fills,
                ModuleCertificationLimits {
                    max_matrix_words: 143,
                    ..limits
                },
            ),
            Err(RuntimeBasisError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "dense matrix words",
                    observed: 144,
                    limit: 143,
                }
            ))
        ));
        let (filled, basis) = graph
            .restrict_selective_relation_with_limits(&source, &fills, limits)
            .unwrap();
        drop(source);
        let mut expected = graph
            .fill_ports(&HashMap::from([(graph.nodes[0].pos, NodeKind::Y)]))
            .unwrap()
            .to_external_generator_table_with_signed()
            .1;
        expected
            .iter_mut()
            .find(|row| row.paulis.get(3) == Pauli::Z)
            .unwrap()
            .shift_phase(2);
        assert_eq!(basis.rows(), expected);
        drop(basis);
        for row in expected {
            assert!(filled.contains_stabilizer_support(&row.paulis));
            assert_eq!(
                filled.stabilizer_row_phases(std::slice::from_ref(&row.paulis)),
                [row.phase()]
            );
        }
    }

    #[test]
    fn signed_restriction_validates_fills_and_resource_bounds() {
        let graph = chain([SelectiveKind::XY; 2], NodeKind::X, 0);
        let source = graph.to_external_generator_table_with_signed().1;
        let pos = graph.nodes[0].pos;
        let limits = ModuleCertificationLimits::default();
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(&source, &[(pos, PauliBasis::Z)], limits),
            Err(RuntimeBasisError::InvalidSelectiveBasis { .. })
        ));
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(
                &source,
                &[(pos, PauliBasis::X), (pos, PauliBasis::Y)],
                limits
            ),
            Err(RuntimeBasisError::DuplicateSelective(_))
        ));
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(
                &source,
                &[(graph.nodes[1].pos, PauliBasis::X)],
                limits
            ),
            Err(RuntimeBasisError::UnknownSelective(_))
        ));
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(
                &source,
                &[],
                ModuleCertificationLimits {
                    max_matrix_words: 0,
                    ..limits
                }
            ),
            Err(RuntimeBasisError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "dense matrix words",
                    ..
                }
            ))
        ));
        for state_limit in [0, 1] {
            assert!(matches!(
                graph.restrict_selective_relation_with_limits(
                    &source,
                    &[],
                    ModuleCertificationLimits {
                        max_normalization_states: state_limit,
                        ..limits
                    }
                ),
                Err(RuntimeBasisError::Stabilizer(StabilizerError::ResourceLimited {
                    phase: "signed relation restriction states",
                    observed,
                    limit,
                })) if limit == state_limit && observed == state_limit + 1
            ));
        }
        let malformed = [PhasedPauliString::positive(PauliString::new(
            graph.total_ids() + 1,
        ))];
        assert!(matches!(
            graph.restrict_selective_relation_with_limits(&malformed, &[], limits),
            Err(RuntimeBasisError::Stabilizer(
                StabilizerError::ComposedPresentationUnsatisfied
            ))
        ));
        let token = crate::CancellationToken::new();
        token.cancel();
        token.scope(|| {
            assert!(matches!(
                graph.restrict_selective_relation_with_limits(&source, &[], limits),
                Err(RuntimeBasisError::Stabilizer(StabilizerError::Cancelled(_)))
            ));
        });
    }
}
