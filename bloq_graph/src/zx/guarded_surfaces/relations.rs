//! Reusable open relations, retaining ordered rows and every interior half-edge.

use std::collections::BTreeSet;

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanRow, BooleanRowSpace, DecisionId, DecisionNode,
};
use rustc_hash::FxHashMap as HashMap;

use super::{ONE, ZERO};
use crate::{BlockGraphError, StabilizerError};

type Row = (DecisionId, Vec<(usize, DecisionId)>);
type Seam = (usize, usize, DecisionId);

// Oversized relations still compose exactly, without cache admission.
const MAX_RETAINED_TERMS: usize = 262_144;

#[derive(Debug, PartialEq, Eq, Hash)]
struct RelationKey {
    rows: Vec<Row>,
    seams: Vec<Seam>,
    decisions: Vec<DecisionNode>,
}

struct RelationTemplate {
    rows: Vec<Row>,
    decisions: Vec<DecisionNode>,
}

#[derive(Default)]
pub(super) struct RelationCache {
    templates: HashMap<RelationKey, RelationTemplate>,
    retained: usize,
    #[cfg(test)]
    pub hits: usize,
    #[cfg(test)]
    pub misses: usize,
}

impl RelationCache {
    pub(super) fn close(
        &mut self,
        rows: Vec<BooleanRow>,
        seams: &[Seam],
        diagram: &mut BooleanDecisionDiagram,
        witness_limit: usize,
    ) -> Result<Vec<BooleanRow>, BlockGraphError> {
        let mut renaming = Renaming::new(&rows, seams, diagram)?;
        let input_rows = renaming.rows(&rows, diagram)?;
        let input_seams = seams
            .iter()
            .map(|&(left, right, guard)| {
                Ok((
                    renaming.columns[&left],
                    renaming.columns[&right],
                    renaming.decision(guard, diagram)?,
                ))
            })
            .collect::<Result<_, BlockGraphError>>()?;
        charge_terms(diagram, renaming.nodes.len())?;
        let variables = renaming
            .nodes
            .iter()
            .map(|node| node.variable)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let key = RelationKey {
            rows: input_rows,
            seams: input_seams,
            decisions: local_nodes(&renaming.nodes, &variables, diagram)?,
        };
        // Cache equality includes the full ordered input, not just its boundary
        // span. This preserves the selected readout/frame reference sector.
        if let Some(template) = self.templates.get(&key) {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            let mut decisions = renaming.original_decisions;
            diagram.charge(template.decisions.len().saturating_add(template.rows.len()))?;
            for node in &template.decisions {
                decisions.push(diagram.make_node(
                    variables[node.variable],
                    decisions[node.low.0],
                    decisions[node.high.0],
                )?);
                check_budget(diagram, witness_limit)?;
            }
            let mut rows = Vec::with_capacity(template.rows.len());
            for (active, terms) in &template.rows {
                charge_terms(diagram, terms.len())?;
                let mut terms = terms
                    .iter()
                    .map(|&(column, guard)| {
                        (
                            2 * renaming.original_columns[column / 2] + column % 2,
                            decisions[guard.0],
                        )
                    })
                    .collect::<Vec<_>>();
                terms.sort_unstable_by_key(|&(column, _)| column);
                rows.push(BooleanRow::from_terms(decisions[active.0], terms));
            }
            check_budget(diagram, witness_limit)?;
            return Ok(rows);
        }
        #[cfg(test)]
        {
            self.misses += 1;
        }
        let rows = close_seams(rows, seams, diagram, witness_limit)?;
        diagram.charge(key.rows.len().saturating_add(rows.len()))?;
        let retained = key
            .rows
            .iter()
            .fold(0usize, |sum, (_, terms)| {
                sum.saturating_add(terms.len().saturating_add(1))
            })
            .saturating_add(rows.iter().fold(0usize, |sum, row| {
                sum.saturating_add(row.terms().len().saturating_add(1))
            }))
            .saturating_add(key.seams.len())
            .saturating_add(key.decisions.len());
        let remaining = MAX_RETAINED_TERMS.saturating_sub(self.retained);
        if retained <= remaining {
            let input_nodes = renaming.nodes.len();
            let normalized = renaming.rows(&rows, diagram)?;
            let retained = retained.saturating_add(renaming.nodes.len() - input_nodes);
            if retained > remaining {
                return Ok(rows);
            }
            let decisions = local_nodes(&renaming.nodes[input_nodes..], &variables, diagram)?;
            self.retained += retained;
            self.templates.insert(
                key,
                RelationTemplate {
                    rows: normalized,
                    decisions,
                },
            );
        }
        Ok(rows)
    }
}

/// Column-pair and decision-node bijections for one instance's open relation.
struct Renaming {
    columns: HashMap<usize, usize>,
    original_columns: Vec<usize>,
    decisions: HashMap<DecisionId, DecisionId>,
    original_decisions: Vec<DecisionId>,
    nodes: Vec<DecisionNode>,
}

impl Renaming {
    fn new(
        rows: &[BooleanRow],
        seams: &[Seam],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, BlockGraphError> {
        let mut columns = HashMap::default();
        let mut original_columns = Vec::new();
        for row in rows {
            diagram.charge(row.terms().len().saturating_add(1))?;
            for &(column, _) in row.terms() {
                columns.entry(column / 2).or_insert_with(|| {
                    original_columns.push(column / 2);
                    original_columns.len() - 1
                });
            }
        }
        diagram.charge(seams.len())?;
        for &(left, right, _) in seams {
            for column in [left, right] {
                columns.entry(column).or_insert_with(|| {
                    original_columns.push(column);
                    original_columns.len() - 1
                });
            }
        }
        Ok(Self {
            columns,
            original_columns,
            decisions: HashMap::from_iter([(ZERO, ZERO), (ONE, ONE)]),
            original_decisions: vec![ZERO, ONE],
            nodes: Vec::new(),
        })
    }

    fn decision(
        &mut self,
        root: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<DecisionId, BlockGraphError> {
        diagram.charge(1)?;
        if root.0 < 2 {
            return Ok(root);
        }
        if let Some(&local) = self.decisions.get(&root) {
            return Ok(local);
        }
        let mut pending = vec![(root, false)];
        while let Some((id, ready)) = pending.pop() {
            diagram.charge(1)?;
            if self.decisions.contains_key(&id) {
                continue;
            }
            let node = diagram.node(id).expect("unmapped nonterminal decision");
            if ready {
                let local = DecisionId(self.nodes.len() + 2);
                self.nodes.push(DecisionNode {
                    variable: node.variable,
                    low: self.decisions[&node.low],
                    high: self.decisions[&node.high],
                });
                self.decisions.insert(id, local);
                self.original_decisions.push(id);
            } else {
                pending.extend([(id, true), (node.high, false), (node.low, false)]);
            }
        }
        Ok(self.decisions[&root])
    }

    fn rows(
        &mut self,
        rows: &[BooleanRow],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<Row>, BlockGraphError> {
        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            charge_terms(diagram, row.terms().len().saturating_add(1))?;
            let active = self.decision(row.active, diagram)?;
            let mut terms = Vec::with_capacity(row.terms().len());
            for &(column, guard) in row.terms() {
                terms.push((
                    2 * self.columns[&(column / 2)] + column % 2,
                    self.decision(guard, diagram)?,
                ));
            }
            terms.sort_unstable_by_key(|&(column, _)| column);
            result.push((active, terms));
        }
        Ok(result)
    }
}

// A monotone variable renaming keeps every imported BDD ordered.
fn local_nodes(
    nodes: &[DecisionNode],
    variables: &[usize],
    diagram: &mut BooleanDecisionDiagram,
) -> Result<Vec<DecisionNode>, BlockGraphError> {
    diagram.charge(
        nodes
            .len()
            .saturating_mul(variables.len().max(1).ilog2() as usize + 1),
    )?;
    Ok(nodes
        .iter()
        .map(|node| DecisionNode {
            variable: variables
                .binary_search(&node.variable)
                .expect("elimination only combines input variables"),
            ..*node
        })
        .collect())
}

fn charge_terms(diagram: &mut BooleanDecisionDiagram, count: usize) -> Result<(), BlockGraphError> {
    diagram.charge(count.saturating_mul(count.max(1).ilog2() as usize + 1))?;
    Ok(())
}

fn check_budget(diagram: &BooleanDecisionDiagram, limit: usize) -> Result<(), BlockGraphError> {
    if diagram.nodes().len() > limit {
        return Err(StabilizerError::ResourceLimited {
            phase: "guarded correlation coefficients",
            observed: diagram.nodes().len(),
            limit,
        }
        .into());
    }
    Ok(())
}

pub(super) fn close_seams(
    mut rows: Vec<BooleanRow>,
    seams: &[Seam],
    diagram: &mut BooleanDecisionDiagram,
    witness_limit: usize,
) -> Result<Vec<BooleanRow>, BlockGraphError> {
    if seams.is_empty() {
        diagram.charge(rows.len())?;
        rows.retain(|row| row.active != ZERO);
        return Ok(rows);
    }
    let mut rows = BooleanRowSpace::new(rows, diagram)?;
    for &(lhs, rhs, hadamard) in seams {
        for axis in 0..2 {
            // A shorter pivot can add a logical readout to an output frame.
            rows.eliminate_linear(
                &[
                    (2 * lhs + axis, ONE),
                    (2 * rhs + axis, ONE),
                    (2 * rhs, hadamard),
                    (2 * rhs + 1, hadamard),
                ],
                diagram,
            )?;
        }
        check_budget(diagram, witness_limit)?;
    }
    Ok(rows.into_rows(diagram)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(
        diagram: &mut BooleanDecisionDiagram,
        columns: [usize; 4],
        variables: [usize; 2],
    ) -> (Vec<BooleanRow>, Vec<Seam>) {
        let guard = diagram.make_node(variables[0], ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let hadamard = diagram.make_node(variables[1], ZERO, ONE).unwrap();
        let [a, b, c, d] = columns;
        (
            vec![
                BooleanRow::from_terms(guard, [(2 * a, ONE), (2 * c, ONE)]),
                BooleanRow::from_terms(absent, [(2 * a + 1, ONE), (2 * c + 1, ONE)]),
                BooleanRow::from_terms(ONE, [(2 * b, ONE), (2 * d, ONE)]),
                BooleanRow::from_terms(ONE, [(2 * b + 1, ONE), (2 * d + 1, ONE)]),
                // A closed observable must survive alongside the open relation.
                BooleanRow::from_terms(ONE, [(2 * c, ONE), (2 * c + 1, ONE)]),
            ],
            vec![(a, b, hadamard)],
        )
    }

    fn explicit(rows: &[BooleanRow]) -> Vec<Row> {
        rows.iter()
            .map(|row| (row.active, row.terms().to_vec()))
            .collect()
    }

    #[test]
    fn scopes_without_seams_only_remove_inactive_rows() {
        let mut diagram = BooleanDecisionDiagram::default();
        let rows = vec![
            BooleanRow::from_terms(ZERO, [(0, ONE)]),
            BooleanRow::new(ONE),
            BooleanRow::from_terms(ONE, [(3, ONE)]),
        ];
        let expected = BooleanRowSpace::new(rows.clone(), &mut diagram)
            .unwrap()
            .into_rows(&mut diagram)
            .unwrap();
        let actual = close_seams(rows, &[], &mut diagram, usize::MAX).unwrap();
        assert_eq!(explicit(&actual), explicit(&expected));
    }

    #[test]
    fn remaps_full_support_and_independent_guarded_hadamards() {
        let mut diagram = BooleanDecisionDiagram::default();
        let mut cache = RelationCache::default();
        let (rows, seams) = inputs(&mut diagram, [0, 1, 2, 3], [0, 1]);
        cache.close(rows, &seams, &mut diagram, usize::MAX).unwrap();
        let (rows, seams) = inputs(&mut diagram, [10, 12, 11, 13], [7, 9]);
        let expected = close_seams(rows.clone(), &seams, &mut diagram, usize::MAX).unwrap();
        let actual = cache.close(rows, &seams, &mut diagram, usize::MAX).unwrap();
        assert_eq!((cache.hits, cache.misses), (1, 1));
        assert_eq!(explicit(&actual), explicit(&expected));
        for row in &actual {
            let variables = std::iter::once(row.active)
                .chain(row.terms().iter().map(|&(_, guard)| guard))
                .flat_map(|root| diagram.variables(root))
                .collect::<BTreeSet<_>>();
            assert!(variables.is_subset(&BTreeSet::from([7, 9])));
            assert!(row.terms().iter().all(|&(column, _)| column >= 20));
        }
    }

    #[test]
    fn different_interior_witnesses_and_variable_aliases_do_not_match() {
        let mut diagram = BooleanDecisionDiagram::default();
        let mut cache = RelationCache::default();
        let (rows, seams) = inputs(&mut diagram, [0, 1, 2, 3], [0, 1]);
        cache.close(rows, &seams, &mut diagram, usize::MAX).unwrap();
        for shared_variables in [false, true] {
            let (mut rows, seams) = inputs(
                &mut diagram,
                [10, 12, 11, 13],
                if shared_variables { [7, 7] } else { [7, 9] },
            );
            if !shared_variables {
                rows[0].set(2 * 13, ONE);
            }
            let expected = close_seams(rows.clone(), &seams, &mut diagram, usize::MAX).unwrap();
            let actual = cache.close(rows, &seams, &mut diagram, usize::MAX).unwrap();
            assert_eq!(explicit(&actual), explicit(&expected));
        }
        assert_eq!((cache.hits, cache.misses), (0, 3));
    }

    #[test]
    fn cache_exhaustion_preserves_results_and_hits_honor_resource_limits() {
        let mut diagram = BooleanDecisionDiagram::default();
        let mut cache = RelationCache::default();
        let (rows, seams) = inputs(&mut diagram, [0, 1, 2, 3], [0, 1]);
        let expected = close_seams(rows.clone(), &seams, &mut diagram, usize::MAX).unwrap();
        cache
            .close(rows.clone(), &seams, &mut diagram, usize::MAX)
            .unwrap();
        assert!(matches!(
            cache.close(rows.clone(), &seams, &mut diagram, 0),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited { limit: 0, .. }
            ))
        ));
        let mut cache = RelationCache {
            retained: usize::MAX,
            ..RelationCache::default()
        };
        let actual = cache.close(rows, &seams, &mut diagram, usize::MAX).unwrap();
        assert!(cache.templates.is_empty());
        assert_eq!(explicit(&actual), explicit(&expected));
    }
}
