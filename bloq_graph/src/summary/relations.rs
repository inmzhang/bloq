//! Complete factored relations shared by summary presentation and production.

use std::collections::BTreeSet;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanRow, BooleanRowSpace, DECISION_FALSE as ZERO,
    DECISION_TRUE as ONE, DecisionId, DecisionNode,
};

use crate::{BlockGraphError, ModuleCertificationLimits, StabilizerError};

type Terms = Vec<(usize, DecisionId)>;

/// Column and Boolean-variable names are local to this exact reduction recipe.
/// Full rows retain interior support; external half-edges are never projected out.
#[derive(Debug, PartialEq, Eq, Hash)]
struct RelationKey {
    nodes: Vec<DecisionNode>,
    rows: Vec<(DecisionId, Terms)>,
    constraints: Vec<Terms>,
}

#[derive(Debug)]
struct ReducedRelation {
    diagram: BooleanDecisionDiagram,
    rows: Vec<BooleanRow>,
}

#[derive(Debug, Default)]
pub(crate) struct ModuleRelations {
    reductions: Mutex<HashMap<RelationKey, Arc<ReducedRelation>>>,
    eliminated: AtomicUsize,
    reused: AtomicUsize,
}

impl ModuleRelations {
    pub(crate) fn work(&self) -> (usize, usize) {
        (
            self.eliminated.load(Ordering::Relaxed),
            self.reused.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn reduce(
        &self,
        rows: Vec<BooleanRow>,
        constraints: Vec<Terms>,
        column_rank: &[usize],
        diagram: &mut BooleanDecisionDiagram,
        limits: ModuleCertificationLimits,
    ) -> Result<Vec<BooleanRow>, BlockGraphError> {
        if constraints.is_empty() {
            return close_seams(rows, &constraints, diagram, limits);
        }
        let terms = rows
            .iter()
            .map(|row| row.terms().len().saturating_add(1))
            .chain(constraints.iter().map(Vec::len))
            .fold(0usize, usize::saturating_add);
        diagram.charge(
            terms
                .saturating_mul(terms.max(1).ilog2() as usize + 1)
                .saturating_mul(3),
        )?;
        let used = rows
            .iter()
            .flat_map(BooleanRow::terms)
            .chain(constraints.iter().flatten())
            .map(|&(column, _)| column / 2)
            .collect::<HashSet<_>>();
        let mut columns = used.into_iter().collect::<Vec<_>>();
        columns.sort_unstable_by_key(|&column| column_rank[column]);
        let column_map = columns
            .iter()
            .enumerate()
            .map(|(local, &source)| (source, local))
            .collect::<HashMap<_, _>>();
        let roots = rows
            .iter()
            .flat_map(|row| {
                std::iter::once(row.active).chain(row.terms().iter().map(|&(_, root)| root))
            })
            .chain(constraints.iter().flatten().map(|&(_, root)| root))
            .collect::<Vec<_>>();
        diagram.charge(roots.len().saturating_add(columns.len()))?;
        let mut reachable = HashSet::default();
        let mut variables = BTreeSet::new();
        let mut pending = roots.clone();
        while let Some(root) = pending.pop() {
            if root == ZERO || root == ONE || !reachable.insert(root) {
                continue;
            }
            diagram.charge(1)?;
            let node = diagram.node(root).expect("live relation coefficient");
            variables.insert(node.variable);
            pending.extend([node.low, node.high]);
        }
        let variables = variables.into_iter().collect::<Vec<_>>();
        let variable_map = variables
            .iter()
            .enumerate()
            .map(|(local, &source)| (source, local))
            .collect::<HashMap<_, _>>();
        let mut local = BooleanDecisionDiagram::with_limits(limits.boolean_limits());
        let mut imported = HashMap::from_iter([(ZERO, ZERO), (ONE, ONE)]);
        // Traverse roots in recipe order, independent of the source arena's
        // allocation history. Only reachable nodes become owned cache roots.
        for root in roots {
            if imported.contains_key(&root) {
                continue;
            }
            let mut pending = vec![(root, false)];
            while let Some((root, visited)) = pending.pop() {
                if imported.contains_key(&root) {
                    continue;
                }
                local.charge(1)?;
                let node = diagram.node(root).expect("live relation coefficient");
                if visited {
                    let value = local.make_node(
                        variable_map[&node.variable],
                        imported[&node.low],
                        imported[&node.high],
                    )?;
                    imported.insert(root, value);
                } else {
                    pending.extend([(root, true), (node.high, false), (node.low, false)]);
                }
            }
        }
        let normalize = |terms: &[(usize, DecisionId)]| {
            let mut result = terms
                .iter()
                .map(|&(column, value)| {
                    (2 * column_map[&(column / 2)] + column % 2, imported[&value])
                })
                .collect::<Terms>();
            result.sort_unstable_by_key(|&(column, _)| column);
            result
        };
        let key = RelationKey {
            nodes: local.nodes().to_vec(),
            rows: rows
                .iter()
                .map(|row| (imported[&row.active], normalize(row.terms())))
                .collect(),
            // Constraint term order retains the source pivot convention.
            constraints: constraints
                .iter()
                .map(|terms| {
                    terms
                        .iter()
                        .map(|&(column, value)| {
                            (2 * column_map[&(column / 2)] + column % 2, imported[&value])
                        })
                        .collect()
                })
                .collect(),
        };
        diagram.charge(local.steps())?;
        let reduced = {
            let mut cache = self
                .reductions
                .lock()
                .expect("module relation cache poisoned");
            if let Some(reduced) = cache.get(&key) {
                self.reused.fetch_add(constraints.len(), Ordering::Relaxed);
                Arc::clone(reduced)
            } else {
                let start = local.steps();
                let mut rows = close_seams(
                    key.rows
                        .iter()
                        .map(|(active, terms)| BooleanRow::from_terms(*active, terms.clone()))
                        .collect(),
                    &key.constraints,
                    &mut local,
                    limits,
                )?;
                local.charge(
                    local.nodes().len().saturating_add(
                        rows.iter()
                            .map(|row| row.terms().len().saturating_add(1))
                            .fold(0usize, usize::saturating_add),
                    ),
                )?;
                local.collect_garbage(rows.iter_mut().flat_map(BooleanRow::decisions_mut));
                diagram.charge(local.steps() - start)?;
                self.eliminated
                    .fetch_add(constraints.len(), Ordering::Relaxed);
                let reduced = Arc::new(ReducedRelation {
                    diagram: local,
                    rows,
                });
                cache.insert(key, Arc::clone(&reduced));
                reduced
            }
        };
        let mut decisions = vec![ZERO, ONE];
        for node in reduced.diagram.nodes() {
            // Canonical variables are a sorted subset of caller variables;
            // this injection preserves ordering even across distinct arenas.
            diagram.charge(1)?;
            decisions.push(diagram.make_node(
                variables[node.variable],
                decisions[node.low.0],
                decisions[node.high.0],
            )?);
        }
        let mut result = Vec::with_capacity(reduced.rows.len());
        for row in &reduced.rows {
            let count = row.terms().len();
            diagram.charge(
                count
                    .saturating_mul(count.max(1).ilog2() as usize + 1)
                    .saturating_add(1),
            )?;
            let mut terms = row
                .terms()
                .iter()
                .map(|&(column, root)| (2 * columns[column / 2] + column % 2, decisions[root.0]))
                .collect::<Terms>();
            terms.sort_unstable_by_key(|&(column, _)| column);
            result.push(BooleanRow::from_terms(decisions[row.active.0], terms));
        }
        if diagram.nodes().len() > limits.max_witness_nodes {
            return Err(StabilizerError::ResourceLimited {
                phase: "guarded correlation coefficients",
                observed: diagram.nodes().len(),
                limit: limits.max_witness_nodes,
            }
            .into());
        }
        Ok(result)
    }
}

pub(crate) fn close_seams(
    rows: Vec<BooleanRow>,
    constraints: &[Terms],
    diagram: &mut BooleanDecisionDiagram,
    limits: ModuleCertificationLimits,
) -> Result<Vec<BooleanRow>, BlockGraphError> {
    let mut rows = BooleanRowSpace::new(rows, diagram)?;
    for terms in constraints {
        // First-row pivot priority preserves authored outcome/frame conventions.
        rows.eliminate_linear(terms, diagram)?;
        if diagram.nodes().len() > limits.max_witness_nodes {
            return Err(StabilizerError::ResourceLimited {
                phase: "guarded correlation coefficients",
                observed: diagram.nodes().len(),
                limit: limits.max_witness_nodes,
            }
            .into());
        }
    }
    Ok(rows.into_rows(diagram)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_full_support_survives_column_variable_and_arena_remapping() {
        let cache = ModuleRelations::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        for (columns, variables) in [([0, 1, 2], [0, 1]), ([2, 0, 1], [4, 9])] {
            let mut diagram = BooleanDecisionDiagram::default();
            let x = diagram.make_node(variables[0], ZERO, ONE).unwrap();
            let y = diagram.make_node(variables[1], ZERO, ONE).unwrap();
            let not_x = diagram.negate(x).unwrap();
            let row = |active, terms: &[(usize, DecisionId)]| {
                let mut terms = terms
                    .iter()
                    .map(|&(column, value)| (2 * columns[column], value))
                    .collect::<Vec<_>>();
                terms.sort_unstable_by_key(|&(column, _)| column);
                BooleanRow::from_terms(active, terms)
            };
            let rows = vec![
                row(x, &[(0, ONE), (1, y)]),
                row(not_x, &[(0, ONE), (2, ONE)]),
                row(ONE, &[(1, ONE), (2, ONE)]),
            ];
            let constraints = vec![vec![(2 * columns[0], ONE), (2 * columns[1], ONE)]];
            let mut rank = vec![0; 3];
            for (index, &column) in columns.iter().enumerate() {
                rank[column] = index;
            }
            let mut expected =
                close_seams(rows.clone(), &constraints, &mut diagram, limits).unwrap();
            let mut actual = cache
                .reduce(rows, constraints, &rank, &mut diagram, limits)
                .unwrap();
            diagram.collect_garbage(
                expected
                    .iter_mut()
                    .chain(&mut actual)
                    .flat_map(BooleanRow::decisions_mut),
            );
            for assignment in 0..4 {
                let values = |rows: &[BooleanRow]| {
                    rows.iter()
                        .map(|row| {
                            let evaluate = |root| {
                                diagram.evaluate(root, |variable| {
                                    let index = variables
                                        .iter()
                                        .position(|&candidate| candidate == variable)
                                        .unwrap();
                                    assignment & (1 << index) != 0
                                })
                            };
                            (
                                evaluate(row.active),
                                row.terms()
                                    .iter()
                                    .map(|&(column, value)| (column, evaluate(value)))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(values(&actual), values(&expected));
            }
        }
        assert_eq!(cache.work(), (1, 1));
    }
}
