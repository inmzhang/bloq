//! Complete factored relations shared by summary presentation and production.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanRow, BooleanRowSpace, DECISION_FALSE as ZERO,
    DECISION_TRUE as ONE, DecisionId, DecisionNode,
};

use crate::{BlockGraphError, ModuleCertificationLimits, StabilizerError};

pub(crate) type Terms = Vec<(usize, DecisionId)>;
pub(crate) type Seam = (usize, usize, DecisionId);

/// Algebraic aliases apply to this completed relation, never its incoming generators.
pub(crate) struct RelationRows {
    pub(crate) rows: Vec<BooleanRow>,
    pub(crate) aliases: Vec<(usize, Option<usize>)>,
}

/// Column and Boolean-variable names are local to this exact reduction recipe.
/// Original interior and exterior queries remain in the caller's alias ledger.
#[derive(Debug, PartialEq, Eq, Hash)]
struct RelationKey {
    nodes: Vec<DecisionNode>,
    rows: Vec<(DecisionId, Terms)>,
    equations: Vec<Terms>,
}

#[derive(Debug)]
struct ReducedRelation {
    nodes: Vec<DecisionNode>,
    rows: Vec<BooleanRow>,
    aliases: Vec<(usize, Option<usize>)>,
}

type ReductionSlot = Arc<Mutex<Option<Arc<ReducedRelation>>>>;

#[derive(Debug, Default)]
pub(crate) struct ModuleRelations {
    reductions: Mutex<HashMap<Arc<RelationKey>, ReductionSlot>>,
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

    fn cached_reduction(
        &self,
        key: RelationKey,
        equations: usize,
        solve: impl FnOnce(&RelationKey) -> Result<ReducedRelation, BlockGraphError>,
    ) -> Result<Arc<ReducedRelation>, BlockGraphError> {
        let key = Arc::new(key);
        let slot = Arc::clone(
            self.reductions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(Arc::clone(&key))
                .or_default(),
        );
        // One caller solves each key while the others wait for it. A failed or
        // panicking solve leaves the slot empty for the next caller.
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(reduced) = slot.as_ref() {
            self.reused.fetch_add(equations, Ordering::Relaxed);
            return Ok(Arc::clone(reduced));
        }
        let reduced = Arc::new(solve(&key)?);
        self.eliminated.fetch_add(equations, Ordering::Relaxed);
        *slot = Some(Arc::clone(&reduced));
        Ok(reduced)
    }

    pub(crate) fn reduce_compact(
        &self,
        rows: Vec<BooleanRow>,
        equations: Vec<Terms>,
        column_rank: &[usize],
        diagram: &mut BooleanDecisionDiagram,
        limits: ModuleCertificationLimits,
    ) -> Result<RelationRows, BlockGraphError> {
        if equations.is_empty() {
            return close_relation(rows, &equations, diagram, limits);
        }
        let terms = rows
            .iter()
            .map(|row| row.terms().len().saturating_add(1))
            .fold(0usize, usize::saturating_add)
            .saturating_add(equations.iter().map(Vec::len).sum::<usize>());
        diagram.charge(
            terms
                .saturating_mul(terms.max(1).ilog2() as usize + 1)
                .saturating_mul(3),
        )?;
        // Omitted original slots stay caller-owned; absence from this scope's
        // algebra makes no zero claim about another scope.
        let used = rows
            .iter()
            .flat_map(BooleanRow::terms)
            .chain(equations.iter().flatten())
            .map(|&(column, _)| column)
            .collect::<HashSet<_>>();
        let mut columns = used.into_iter().collect::<Vec<_>>();
        columns.sort_unstable_by_key(|&column| (column_rank[column / 2], column % 2));
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
            .chain(equations.iter().flatten().map(|&(_, root)| root))
            .collect::<Vec<_>>();
        diagram.charge(roots.len().saturating_add(columns.len()))?;
        let mut imported = HashMap::from_iter([(ZERO, ZERO), (ONE, ONE)]);
        let mut nodes = Vec::new();
        let mut variables = HashSet::default();
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
                diagram.charge(1)?;
                let node = diagram.node(root).expect("live relation coefficient");
                if visited {
                    variables.insert(node.variable);
                    let value = DecisionId(nodes.len() + 2);
                    nodes.push(DecisionNode {
                        variable: node.variable,
                        low: imported[&node.low],
                        high: imported[&node.high],
                    });
                    imported.insert(root, value);
                } else {
                    pending.extend([(root, true), (node.high, false), (node.low, false)]);
                }
            }
        }
        diagram.charge(
            nodes.len().saturating_add(
                variables
                    .len()
                    .saturating_mul(variables.len().max(1).ilog2() as usize + 1),
            ),
        )?;
        let mut variables = variables.into_iter().collect::<Vec<_>>();
        variables.sort_unstable();
        let variable_map = variables
            .iter()
            .enumerate()
            .map(|(local, &source)| (source, local))
            .collect::<HashMap<_, _>>();
        for node in &mut nodes {
            node.variable = variable_map[&node.variable];
        }
        let normalize = |terms: &[(usize, DecisionId)]| {
            let mut result = terms
                .iter()
                .map(|&(column, value)| (column_map[&column], imported[&value]))
                .collect::<Terms>();
            result.sort_unstable_by_key(|&(column, _)| column);
            result
        };
        let key = RelationKey {
            nodes,
            rows: rows
                .iter()
                .map(|row| (imported[&row.active], normalize(row.terms())))
                .collect(),
            // Equation order and repeated query terms retain source pivots and aliases.
            equations: equations.iter().map(|terms| normalize(terms)).collect(),
        };
        let count = equations.len();
        let reduced = self.cached_reduction(key, count, |key| {
            let mut local_limits = limits.boolean_limits();
            local_limits.max_nodes = local_limits.max_nodes.min(diagram.limits().max_nodes);
            local_limits.max_steps = local_limits.max_steps.min(diagram.limits().max_steps);
            let mut local = BooleanDecisionDiagram::with_limits(local_limits);
            // Reserve work already spent by the caller before local solving.
            let start = diagram.steps();
            local.charge(start)?;
            for node in &key.nodes {
                local.charge(1)?;
                local.make_node(node.variable, node.low, node.high)?;
            }
            local.charge(key.rows.iter().fold(0usize, |sum, (_, terms)| {
                sum.saturating_add(
                    terms
                        .len()
                        .saturating_mul(terms.len().max(1).ilog2() as usize + 1)
                        .saturating_add(1),
                )
            }))?;
            let RelationRows { mut rows, aliases } = close_relation(
                key.rows
                    .iter()
                    .map(|(active, terms)| BooleanRow::from_terms(*active, terms.clone()))
                    .collect(),
                &key.equations,
                &mut local,
                limits,
            )?;
            local.charge(
                local.nodes().len().saturating_mul(2).saturating_add(
                    rows.iter()
                        .map(|row| row.terms().len().saturating_add(1))
                        .fold(0usize, usize::saturating_add),
                ),
            )?;
            local.collect_garbage(rows.iter_mut().flat_map(BooleanRow::decisions_mut));
            diagram.charge(local.steps() - start)?;
            Ok(ReducedRelation {
                nodes: local.nodes().to_vec(),
                rows,
                aliases,
            })
        })?;
        let mut decisions = vec![ZERO, ONE];
        for node in &reduced.nodes {
            // Canonical variables are a sorted subset of caller variables;
            // this injection preserves ordering even across distinct arenas.
            diagram.charge(1)?;
            decisions.push(diagram.make_node(
                variables[node.variable],
                decisions[node.low.0],
                decisions[node.high.0],
            )?);
        }
        diagram.charge(reduced.aliases.len().saturating_mul(3))?;
        let mut representatives = vec![usize::MAX; columns.len()];
        for &(column, alias) in &reduced.aliases {
            if let Some(alias) = alias {
                representatives[alias] = representatives[alias].min(columns[column]);
            }
        }
        let aliases = reduced
            .aliases
            .iter()
            .map(|&(column, alias)| (columns[column], alias.map(|alias| representatives[alias])))
            .collect();
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
                .map(|&(column, root)| (representatives[column], decisions[root.0]))
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
        Ok(RelationRows {
            rows: result,
            aliases,
        })
    }
}

pub(crate) fn seam_equations(seams: &[Seam]) -> Vec<Terms> {
    seams
        .iter()
        .flat_map(|&(left, right, hadamard)| {
            [0, 1].map(|axis| {
                vec![
                    (2 * left + axis, ONE),
                    (2 * right + axis, ONE),
                    (2 * right, hadamard),
                    (2 * right + 1, hadamard),
                ]
            })
        })
        .collect()
}

pub(crate) fn coalesce_columns(
    rows: &mut [BooleanRow],
    columns: impl IntoIterator<Item = usize>,
    diagram: &mut BooleanDecisionDiagram,
) -> Result<Vec<(usize, Option<usize>)>, BlockGraphError> {
    let mut columns = columns.into_iter().collect::<Vec<_>>();
    diagram.charge(
        columns
            .len()
            .saturating_mul(columns.len().max(1).ilog2() as usize + 4),
    )?;
    columns.sort_unstable();
    columns.dedup();
    let indices = columns
        .iter()
        .enumerate()
        .map(|(index, &column)| (column, index))
        .collect::<HashMap<_, _>>();
    let mut signatures = vec![Vec::new(); columns.len()];
    for (index, row) in rows.iter().enumerate() {
        crate::CancellationToken::checkpoint()?;
        diagram.charge(row.terms().len().saturating_mul(3))?;
        for &(column, coefficient) in row.terms() {
            // Raw equality preserves off-activation timing and pivot choices.
            signatures[indices[&column]].push((index, coefficient));
        }
    }
    let mut unique = HashMap::default();
    let mut aliases = Vec::with_capacity(columns.len());
    for (&column, signature) in columns.iter().zip(signatures) {
        crate::CancellationToken::checkpoint()?;
        diagram.charge(signature.len().saturating_mul(2).saturating_add(1))?;
        aliases.push((
            column,
            if signature.is_empty() {
                None
            } else {
                Some(*unique.entry(signature).or_insert(column))
            },
        ));
    }
    if unique.len() != columns.len() {
        for row in rows {
            crate::CancellationToken::checkpoint()?;
            let count = row.terms().len();
            diagram.charge(count.saturating_mul(count.max(1).ilog2() as usize + 2))?;
            let active = row.active;
            let terms = std::mem::take(row)
                .into_terms()
                .into_iter()
                .filter(|&(column, _)| aliases[indices[&column]].1 == Some(column));
            *row = BooleanRow::from_terms(active, terms);
        }
    }
    Ok(aliases)
}

pub(crate) fn close_relation(
    mut rows: Vec<BooleanRow>,
    equations: &[Terms],
    diagram: &mut BooleanDecisionDiagram,
    limits: ModuleCertificationLimits,
) -> Result<RelationRows, BlockGraphError> {
    diagram.charge(
        rows.iter()
            .map(|row| row.terms().len())
            .sum::<usize>()
            .saturating_add(equations.iter().map(Vec::len).sum::<usize>()),
    )?;
    let columns = rows
        .iter()
        .flat_map(BooleanRow::terms)
        .chain(equations.iter().flatten())
        .map(|&(column, _)| column)
        .collect::<HashSet<_>>();
    if equations.is_empty() {
        let mut rows = close_equations(rows, equations, diagram, limits)?;
        let aliases = coalesce_columns(&mut rows, columns, diagram)?;
        return Ok(RelationRows { rows, aliases });
    }
    // Only raw equalities already present in every incoming row precede solving.
    let mut aliases = coalesce_columns(&mut rows, columns, diagram)?;
    diagram.charge(
        aliases
            .len()
            .saturating_add(equations.len())
            .saturating_add(equations.iter().map(Vec::len).sum::<usize>()),
    )?;
    let incoming = aliases.iter().copied().collect::<HashMap<_, _>>();
    let mut remaining = BooleanRowSpace::new(rows, diagram)?;
    let mut online = HashMap::default();
    let mut mapped = Vec::new();
    for terms in equations {
        crate::CancellationToken::checkpoint()?;
        mapped.clear();
        // These are weighted XOR occurrences, even after IDs repeat or reorder.
        for &(column, weight) in terms {
            if let Some(column) = incoming[&column] {
                mapped.push((resolve_online_alias(&mut online, column, diagram)?, weight));
            }
        }
        remaining.eliminate_linear(&mapped, diagram)?;
        if let Some((keep, discard)) = raw_singleton_pair(&remaining, &mapped, diagram)? {
            // Only the linear seam path maintains complete raw postings here.
            remaining.erase_column(discard, diagram)?;
            if online.len() == online.capacity() {
                diagram.charge(online.len())?;
            }
            diagram.charge(1)?;
            online.insert(discard, keep);
        }
        if diagram.nodes().len() > limits.max_witness_nodes {
            return Err(StabilizerError::ResourceLimited {
                phase: "guarded correlation coefficients",
                observed: diagram.nodes().len(),
                limit: limits.max_witness_nodes,
            }
            .into());
        }
    }
    drop(incoming);
    drop(mapped);
    let mut rows = remaining.into_rows(diagram)?;
    diagram.charge(aliases.len())?;
    for (_, alias) in &mut aliases {
        crate::CancellationToken::checkpoint()?;
        if let Some(column) = alias {
            *column = resolve_online_alias(&mut online, *column, diagram)?;
        }
    }
    let representatives = aliases
        .iter()
        .filter_map(|&(column, alias)| (alias == Some(column)).then_some(column));
    let completed = coalesce_columns(&mut rows, representatives, diagram)?;
    diagram.charge(completed.len().saturating_add(aliases.len()))?;
    let completed = completed.into_iter().collect::<HashMap<_, _>>();
    for (_, alias) in &mut aliases {
        crate::CancellationToken::checkpoint()?;
        *alias = alias.and_then(|column| completed[&column]);
    }
    Ok(RelationRows { rows, aliases })
}

fn resolve_online_alias(
    aliases: &mut HashMap<usize, usize>,
    mut column: usize,
    diagram: &mut BooleanDecisionDiagram,
) -> Result<usize, BlockGraphError> {
    let mut root = column;
    loop {
        crate::CancellationToken::checkpoint()?;
        diagram.charge(1)?;
        let Some(&next) = aliases.get(&root) else {
            break;
        };
        root = next;
    }
    while column != root {
        crate::CancellationToken::checkpoint()?;
        diagram.charge(2)?;
        let next = aliases[&column];
        aliases.insert(column, root);
        column = next;
    }
    Ok(root)
}

/// A copy certificate for the linear-only seam loop, not an activation equality.
fn raw_singleton_pair(
    rows: &BooleanRowSpace,
    terms: &[(usize, DecisionId)],
    diagram: &mut BooleanDecisionDiagram,
) -> Result<Option<(usize, usize)>, BlockGraphError> {
    crate::CancellationToken::checkpoint()?;
    diagram.charge(1)?;
    if terms.len() > 4 {
        return Ok(None);
    }
    diagram.charge(terms.len().saturating_mul(terms.len().saturating_add(1)))?;
    let mut odd = Vec::with_capacity(terms.len());
    for &term in terms {
        if term.1 == ZERO {
            continue;
        }
        if let Some(index) = odd.iter().position(|&existing| existing == term) {
            odd.swap_remove(index);
        } else {
            odd.push(term);
        }
    }
    // Recognition alone cancels occurrences; elimination keeps the original list.
    let [(left, ONE), (right, ONE)] = odd.as_slice() else {
        return Ok(None);
    };
    let (left, right) = (*left, *right);
    if left == right {
        return Ok(None);
    }
    let mut contributors = [None; 2];
    for (index, column) in [left, right].into_iter().enumerate() {
        let mut count = 0;
        rows.visit_column_rows(column, diagram, |row| {
            count += 1;
            contributors[index] = Some(row);
        })?;
        if count != 1 {
            return Ok(None);
        }
    }
    if contributors[0] != contributors[1] {
        return Ok(None);
    }
    let row = rows
        .row(contributors[0].expect("one raw contributor"))
        .expect("posting identifies a live row");
    diagram.charge((row.terms().len().ilog2() as usize + 1).saturating_mul(2))?;
    Ok((row.get(left) == row.get(right)).then_some((left.min(right), left.max(right))))
}

pub(crate) fn close_seams(
    rows: Vec<BooleanRow>,
    seams: &[Seam],
    diagram: &mut BooleanDecisionDiagram,
    limits: ModuleCertificationLimits,
) -> Result<Vec<BooleanRow>, BlockGraphError> {
    close_equations(rows, &seam_equations(seams), diagram, limits)
}

fn close_equations(
    mut rows: Vec<BooleanRow>,
    equations: &[Terms],
    diagram: &mut BooleanDecisionDiagram,
    limits: ModuleCertificationLimits,
) -> Result<Vec<BooleanRow>, BlockGraphError> {
    if equations.is_empty() {
        diagram.charge(rows.len())?;
        rows.retain(|row| row.active != ZERO);
        return Ok(rows);
    }
    let mut rows = BooleanRowSpace::new(rows, diagram)?;
    for terms in equations {
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

    impl ModuleRelations {
        fn reduce(
            &self,
            rows: Vec<BooleanRow>,
            seams: Vec<Seam>,
            column_rank: &[usize],
            diagram: &mut BooleanDecisionDiagram,
            limits: ModuleCertificationLimits,
        ) -> Result<Vec<BooleanRow>, BlockGraphError> {
            let reduced =
                self.reduce_compact(rows, seam_equations(&seams), column_rank, diagram, limits)?;
            Ok(expand(&reduced))
        }
    }

    fn expand(reduced: &RelationRows) -> Vec<BooleanRow> {
        reduced
            .rows
            .iter()
            .map(|row| {
                BooleanRow::from_terms(
                    row.active,
                    reduced
                        .aliases
                        .iter()
                        .filter_map(|&(column, alias)| alias.map(|alias| (column, row.get(alias)))),
                )
            })
            .collect()
    }

    fn assert_pre_solve_trace(
        rows: &[BooleanRow],
        equations: &[Terms],
        diagram: &mut BooleanDecisionDiagram,
    ) -> RelationRows {
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut result = None;
        for end in 0..=equations.len() {
            let equations = &equations[..end];
            let columns = rows
                .iter()
                .flat_map(BooleanRow::terms)
                .chain(equations.iter().flatten())
                .map(|&(column, _)| column)
                .collect::<HashSet<_>>();
            let raw = close_equations(rows.to_vec(), equations, diagram, limits).unwrap();
            let mut expected = raw.clone();
            let aliases = coalesce_columns(&mut expected, columns, diagram).unwrap();
            let actual = close_relation(rows.to_vec(), equations, diagram, limits).unwrap();
            assert_eq!(actual.aliases, aliases, "aliases after {end} equations");
            assert_eq!(
                explicit(&actual.rows),
                explicit(&expected),
                "compact rows after {end} equations"
            );
            assert_eq!(
                explicit(&expand(&actual)),
                explicit(&raw),
                "raw rows after {end} equations"
            );
            result = Some(actual);
        }
        result.unwrap()
    }

    #[test]
    fn online_alias_chains_preserve_future_queries_and_final_zeros() {
        let mut diagram = BooleanDecisionDiagram::default();
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let rows = vec![
            BooleanRow::from_terms(ONE, [(8, ONE), (9, ONE)]),
            BooleanRow::from_terms(ONE, [(6, ONE)]),
            BooleanRow::from_terms(ONE, [(4, ONE)]),
            BooleanRow::from_terms(ONE, [(2, ONE)]),
            BooleanRow::from_terms(ONE, [(10, ONE)]),
        ];
        let equations = [
            vec![(8, ONE), (6, ONE)],
            vec![(9, ONE), (4, ONE)],
            vec![(6, ONE), (2, ONE)],
            vec![(8, guard), (4, guard)],
        ];
        let merged = assert_pre_solve_trace(&rows, &equations, &mut diagram);
        assert_eq!(
            merged.aliases,
            [
                (2, Some(2)),
                (4, Some(2)),
                (6, Some(2)),
                (8, Some(2)),
                (9, Some(2)),
                (10, Some(10)),
            ]
        );
        let mut equations = equations.to_vec();
        equations.push(vec![(9, guard), (6, absent)]);
        let zeroed = assert_pre_solve_trace(&rows, &equations, &mut diagram);
        assert_eq!(
            zeroed.aliases,
            [
                (2, None),
                (4, None),
                (6, None),
                (8, None),
                (9, None),
                (10, Some(10)),
            ]
        );
        assert_eq!(explicit(&zeroed.rows), explicit(&rows[4..]));
    }

    #[test]
    fn online_singleton_certificate_requires_complete_raw_equality() {
        let mut diagram = BooleanDecisionDiagram::default();
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let source = [0, 1]
            .map(|column| BooleanRow::from_terms(ONE, [(column, ONE)]))
            .to_vec();
        let mut rows = BooleanRowSpace::new(source.clone(), &mut diagram).unwrap();
        rows.eliminate_linear(&[(0, guard)], &mut diagram).unwrap();
        rows.eliminate_linear(&[(0, ONE), (1, ONE)], &mut diagram)
            .unwrap();
        let remaining = rows.rows().next().unwrap();
        assert_eq!(remaining.active, absent);
        assert_eq!(remaining.get(0), absent);
        assert_eq!(remaining.get(1), ONE);
        assert_eq!(
            raw_singleton_pair(&rows, &[(0, ONE), (1, ONE)], &mut diagram).unwrap(),
            None,
            "active equality must not erase the off-activation difference"
        );
        assert_pre_solve_trace(
            &source,
            &[vec![(0, guard)], vec![(0, ONE), (1, ONE)]],
            &mut diagram,
        );

        let equal = BooleanRowSpace::new(
            vec![BooleanRow::from_terms(absent, [(0, guard), (2, guard)])],
            &mut diagram,
        )
        .unwrap();
        let nodes = diagram.nodes().len();
        assert_eq!(
            raw_singleton_pair(
                &equal,
                &[(2, ONE), (0, ONE), (0, guard), (0, guard)],
                &mut diagram,
            )
            .unwrap(),
            Some((0, 2)),
            "cancel duplicate weighted occurrences only for recognition"
        );
        assert_eq!(
            raw_singleton_pair(&equal, &[(0, guard), (2, guard)], &mut diagram).unwrap(),
            None,
            "the narrow rule declines non-unit weights"
        );
        assert_eq!(
            diagram.nodes().len(),
            nodes,
            "recognition creates no BDD roots"
        );

        let second = diagram.make_node(1, ZERO, ONE).unwrap();
        let narrower = diagram
            .apply(bloq_utils::boolean::BooleanOp::And, guard, second)
            .unwrap();
        let source = vec![
            BooleanRow::from_terms(narrower, [(0, guard), (1, guard)]),
            BooleanRow::from_terms(ONE, [(0, ONE), (2, ONE)]),
            BooleanRow::from_terms(ONE, [(4, ONE)]),
        ];
        let completed = assert_pre_solve_trace(
            &source,
            &[vec![(2, ONE)], vec![(0, ONE), (1, ONE)]],
            &mut diagram,
        );
        assert_eq!(
            explicit(&expand(&completed)),
            explicit(&[source[0].clone(), source[2].clone()]),
            "matching nonconstant raw values survive outside the narrower activation"
        );

        // ZERO activation is retired by RowSpace. This retained contributor's
        // x coefficient instead vanishes only on its own nonzero activation.
        let source = vec![
            BooleanRow::from_terms(guard, [(0, ONE), (1, ONE)]),
            BooleanRow::from_terms(absent, [(0, guard), (2, ONE)]),
        ];
        let mut rows = BooleanRowSpace::new(source.clone(), &mut diagram).unwrap();
        rows.eliminate_linear(&[(0, ONE), (1, ONE)], &mut diagram)
            .unwrap();
        assert_eq!(rows.row_ids_with_column(0).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(rows.row_ids_with_column(1).collect::<Vec<_>>(), [0]);
        assert_eq!(
            raw_singleton_pair(&rows, &[(0, ONE), (1, ONE)], &mut diagram).unwrap(),
            None,
            "the inactive raw contribution must remain indexed"
        );
        assert_pre_solve_trace(&source, &[vec![(0, ONE), (1, ONE)]], &mut diagram);
    }

    #[test]
    fn online_alias_survives_contributor_spread_and_retirement() {
        let mut diagram = BooleanDecisionDiagram::default();
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let source = vec![
            BooleanRow::from_terms(ONE, [(0, ONE)]),
            BooleanRow::from_terms(ONE, [(1, ONE)]),
            BooleanRow::from_terms(ONE, [(2, ONE), (3, ONE)]),
            BooleanRow::from_terms(ONE, [(2, ONE), (4, ONE)]),
            BooleanRow::from_terms(ONE, [(5, ONE)]),
        ];
        let equations = [
            vec![(0, ONE), (1, ONE)],
            vec![(1, ONE), (2, ONE)],
            vec![(1, guard), (3, ONE)],
            vec![(3, ONE)],
        ];
        assert_pre_solve_trace(&source, &equations, &mut diagram);
        let mut baseline = BooleanRowSpace::new(source.clone(), &mut diagram).unwrap();
        let mut compact = BooleanRowSpace::new(source, &mut diagram).unwrap();
        let expected = baseline
            .eliminate_linear(&equations[0], &mut diagram)
            .unwrap();
        let actual = compact
            .eliminate_linear(&equations[0], &mut diagram)
            .unwrap();
        assert_eq!(explicit(&[actual]), explicit(&[expected]));
        assert_eq!(
            raw_singleton_pair(&compact, &equations[0], &mut diagram).unwrap(),
            Some((0, 1))
        );
        compact.erase_column(1, &mut diagram).unwrap();
        for (index, query) in equations[1..].iter().enumerate() {
            let mapped = query
                .iter()
                .map(|&(column, weight)| (if column == 1 { 0 } else { column }, weight))
                .collect::<Vec<_>>();
            let expected = baseline.eliminate_linear(query, &mut diagram).unwrap();
            let mut actual = compact.eliminate_linear(&mapped, &mut diagram).unwrap();
            actual.set(1, actual.get(0));
            assert_eq!(explicit(&[actual]), explicit(&[expected]), "pivot {index}");
            let expected = baseline
                .indexed_rows()
                .map(|(id, row)| (id, row.active, row.terms().to_vec()))
                .collect::<Vec<_>>();
            let actual = compact
                .indexed_rows()
                .map(|(id, row)| {
                    let mut row = row.clone();
                    row.set(1, row.get(0));
                    (id, row.active, row.into_terms())
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "raw stable rows after pivot {index}");
            if index == 0 {
                assert!(
                    compact.row(1).is_none(),
                    "the certificate's old row retired"
                );
                assert_eq!(compact.row_ids_with_column(0).collect::<Vec<_>>(), [2, 3]);
            }
        }
        assert!(compact.row(2).is_none());
        assert_eq!(compact.row_ids_with_column(0).collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn online_aliases_do_not_rewrite_historical_source_rows() {
        let mut diagram = BooleanDecisionDiagram::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let cache = ModuleRelations::default();
        let source = [0, 1]
            .map(|column| BooleanRow::from_terms(ONE, [(column, ONE)]))
            .to_vec();
        let correlated = cache
            .reduce_compact(
                source.clone(),
                vec![vec![(0, ONE), (1, ONE)]],
                &[0],
                &mut diagram,
                limits,
            )
            .unwrap();
        assert_eq!(correlated.aliases, [(0, Some(0)), (1, Some(0))]);
        let restricted = cache
            .reduce_compact(
                source.clone(),
                vec![vec![(0, ONE)]],
                &[0],
                &mut diagram,
                limits,
            )
            .unwrap();
        assert_eq!(restricted.aliases, [(0, None), (1, Some(1))]);
        assert_eq!(explicit(&expand(&restricted)), explicit(&source[1..]));
        assert_eq!(cache.work(), (2, 0));
    }

    #[test]
    fn pre_solve_weighted_duplicates_cancel_or_cover_both_guards() {
        let mut diagram = BooleanDecisionDiagram::default();
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let rows = vec![
            BooleanRow::from_terms(guard, [(1, ONE), (4, ONE)]),
            BooleanRow::from_terms(absent, [(1, ONE), (4, ONE)]),
            BooleanRow::from_terms(ONE, [(8, ONE)]),
        ];
        let equal = vec![(4, guard), (1, guard)];
        let cancelled = assert_pre_solve_trace(&rows, std::slice::from_ref(&equal), &mut diagram);
        assert_eq!(explicit(&expand(&cancelled)), explicit(&rows));
        let completed =
            assert_pre_solve_trace(&rows, &[equal, vec![(4, guard), (1, absent)]], &mut diagram);
        assert_eq!(completed.aliases, [(1, None), (4, None), (8, Some(8))]);
        assert_eq!(explicit(&completed.rows), explicit(&rows[2..]));
    }

    #[test]
    fn pre_solve_nonmonotonic_maps_compose_merges_and_zeros() {
        let mut diagram = BooleanDecisionDiagram::default();
        let rows = [[1, 4], [2, 3], [8, 10]]
            .map(|columns| BooleanRow::from_terms(ONE, columns.map(|column| (column, ONE))))
            .to_vec();
        // The sorted source query [3, 4, 9] maps to [2, 1]; 9 is query-only zero.
        let merge = vec![(3, ONE), (4, ONE), (9, ONE)];
        let merged = assert_pre_solve_trace(&rows, std::slice::from_ref(&merge), &mut diagram);
        assert_eq!(
            merged.aliases,
            [
                (1, Some(1)),
                (2, Some(1)),
                (3, Some(1)),
                (4, Some(1)),
                (8, Some(8)),
                (9, None),
                (10, Some(8)),
            ]
        );
        let zeroed = assert_pre_solve_trace(
            &rows,
            &[merge, vec![(3, ONE), (2, ONE)], vec![(4, ONE)]],
            &mut diagram,
        );
        assert_eq!(
            zeroed.aliases,
            [
                (1, None),
                (2, None),
                (3, None),
                (4, None),
                (8, Some(8)),
                (9, None),
                (10, Some(8)),
            ]
        );
        assert_eq!(explicit(&expand(&zeroed)), explicit(&rows[2..]));
    }

    #[test]
    fn pre_solve_symbolic_h_preserves_inactive_raw_distinctions() {
        for shared_axes in [false, true] {
            let mut diagram = BooleanDecisionDiagram::default();
            let guard = diagram.make_node(0, ZERO, ONE).unwrap();
            let absent = diagram.negate(guard).unwrap();
            let hadamard = diagram.make_node(1, ZERO, ONE).unwrap();
            let mut rows = vec![
                BooleanRow::from_terms(guard, [(0, ONE), (2, ONE)]),
                BooleanRow::from_terms(absent, [(1, ONE), (3, ONE)]),
                BooleanRow::from_terms(guard, [(8, ONE), (9, guard), (10, ONE), (11, ONE)]),
                BooleanRow::from_terms(ZERO, [(10, ONE)]),
                BooleanRow::from_terms(ONE, [(12, ONE)]),
            ];
            if shared_axes {
                rows.push(BooleanRow::from_terms(
                    ONE,
                    [(4, ONE), (5, ONE), (6, ONE), (7, ONE)],
                ));
            } else {
                rows.extend([
                    BooleanRow::from_terms(ONE, [(4, ONE), (6, ONE)]),
                    BooleanRow::from_terms(ONE, [(5, ONE), (7, ONE)]),
                ]);
            }
            let mut incoming = rows.clone();
            let aliases = coalesce_columns(&mut incoming, 0..13, &mut diagram).unwrap();
            assert_ne!(aliases[8].1, aliases[9].1, "do not mask by activation");
            assert_ne!(aliases[10].1, aliases[11].1, "inspect inactive raw rows");
            let completed =
                assert_pre_solve_trace(&rows, &seam_equations(&[(1, 2, hadamard)]), &mut diagram);
            assert!(
                expand(&completed)
                    .iter()
                    .any(|row| row.active == ONE && row.get(12) == ONE),
                "retain the closed mode"
            );
        }
    }

    #[test]
    fn pre_solve_cached_children_keep_original_queries_and_fresh_parent_rows() {
        let mut diagram = BooleanDecisionDiagram::default();
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let limits = ModuleCertificationLimits::DEFAULT;
        let cache = ModuleRelations::default();
        let mut original = [0, 2]
            .map(|column| BooleanRow::from_terms(ONE, [(column, ONE)]))
            .to_vec();
        let mut compact = original.clone();
        let mut binding = (0..16).map(Some).collect::<Vec<_>>();
        for (arms, internal) in [([4, 6], 8), ([10, 12], 14)] {
            let rows = vec![
                BooleanRow::from_terms(ONE, arms.map(|column| (column, ONE))),
                BooleanRow::from_terms(ONE, [(internal, ONE)]),
            ];
            original.extend(rows.clone());
            let child = cache
                .reduce_compact(
                    rows,
                    vec![arms.map(|column| (column, ONE)).to_vec()],
                    &[0, 1, 2, 3, 4, 5, 6, 7],
                    &mut diagram,
                    limits,
                )
                .unwrap();
            assert_eq!(child.aliases.len(), 3);
            for (column, alias) in child.aliases {
                binding[column] = alias;
            }
            compact.extend(child.rows);
        }
        assert_eq!(cache.work(), (1, 1));
        let equations = vec![
            vec![(6, guard), (0, guard)],
            vec![(12, absent), (2, absent)],
        ];
        let mapped = equations
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .map(|&(column, weight)| (binding[column].unwrap(), weight))
                    .collect()
            })
            .collect::<Vec<_>>();
        let parent = assert_pre_solve_trace(&compact, &mapped, &mut diagram);
        let mut actual = expand(&parent);
        for row in &mut actual {
            row.set(6, row.get(4));
            row.set(12, row.get(10));
        }
        let expected = close_equations(original, &equations, &mut diagram, limits).unwrap();
        assert_eq!(explicit(&actual), explicit(&expected));
        let columns = [0, 2, 4, 6, 8, 10, 12, 14];
        let mut expected = expected;
        let expected_aliases = coalesce_columns(&mut expected, columns, &mut diagram).unwrap();
        let completed = parent.aliases.into_iter().collect::<HashMap<_, _>>();
        let actual_aliases = columns
            .map(|column| {
                (
                    column,
                    binding[column].and_then(|coordinate| completed[&coordinate]),
                )
            })
            .to_vec();
        assert_eq!(actual_aliases, expected_aliases);
        for internal in [8, 14] {
            assert!(
                actual
                    .iter()
                    .any(|row| row.active == ONE && row.get(internal) == ONE)
            );
        }
    }

    #[test]
    fn completed_aliases_do_not_project_unsolved_source_generators() {
        let mut diagram = BooleanDecisionDiagram::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let rows = [[0, 1], [1, 2], [1, 3], [1, 4]]
            .map(|bits| BooleanRow::from_terms(ONE, bits.map(|bit| (bit, ONE))))
            .to_vec();
        let equation = vec![(0, ONE), (1, ONE)];
        let reduced = close_relation(rows, &[equation], &mut diagram, limits).unwrap();
        assert_eq!(reduced.rows.len(), 3);
        for row in expand(&reduced) {
            assert_eq!(row.get(0), row.get(1));
            assert!(
                ![2, 3, 4]
                    .map(|bit| row.get(bit) == ONE)
                    .into_iter()
                    .fold(false, |sum, bit| sum ^ bit)
            );
        }
    }

    #[test]
    fn compact_cache_preserves_alias_zero_and_hadamard_seam_queries() {
        let limits = ModuleCertificationLimits::DEFAULT;
        let cache = ModuleRelations::default();
        for (columns, variable) in [([0, 1, 2], 0), ([2, 0, 1], 7)] {
            let mut diagram = BooleanDecisionDiagram::default();
            let hadamard = diagram.make_node(variable, ZERO, ONE).unwrap();
            let bit = |local| 2 * columns[local / 2] + local % 2;
            let row = |bits: &[usize]| {
                BooleanRow::from_terms(ONE, bits.iter().map(|&index| (bit(index), ONE)))
            };
            let original = vec![row(&[0, 2]), row(&[4]), row(&[5])];
            let seams = [(columns[1], columns[2], hadamard)];
            let expected = close_seams(original, &seams, &mut diagram, limits).unwrap();
            // The completed child has x2=x0 and z2=z0=0; parent support is disjoint.
            let equations = seam_equations(&seams)
                .into_iter()
                .map(|terms| {
                    terms
                        .into_iter()
                        .filter_map(|(column, value)| {
                            if column == bit(3) {
                                None
                            } else {
                                Some((if column == bit(2) { bit(0) } else { column }, value))
                            }
                        })
                        .collect()
                })
                .collect();
            let mut rank = vec![0; columns.len()];
            for (local, &column) in columns.iter().enumerate() {
                rank[column] = local;
            }
            let reduced = cache
                .reduce_compact(
                    vec![row(&[0]), row(&[4]), row(&[5])],
                    equations,
                    &rank,
                    &mut diagram,
                    limits,
                )
                .unwrap();
            let mut actual = expand(&reduced);
            for row in &mut actual {
                row.set(bit(2), row.get(bit(0)));
            }
            for assignment in [false, true] {
                let evaluate = |rows: &[BooleanRow]| {
                    rows.iter()
                        .map(|row| {
                            (
                                diagram.evaluate(row.active, |_| assignment),
                                (0..6)
                                    .map(|index| {
                                        diagram.evaluate(row.get(bit(index)), |_| assignment)
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(evaluate(&actual), evaluate(&expected));
            }
        }
        assert_eq!(cache.work(), (2, 2));
    }

    #[test]
    fn cache_keys_distinguish_query_alias_partitions() {
        let cache = ModuleRelations::default();
        let mut diagram = BooleanDecisionDiagram::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let rows = [0, 1]
            .map(|bit| BooleanRow::from_terms(ONE, [(bit, ONE)]))
            .to_vec();
        for (query, rank) in [(vec![(0, ONE), (0, ONE)], 2), (vec![(0, ONE), (1, ONE)], 1)] {
            let reduced = cache
                .reduce_compact(rows.clone(), vec![query], &[0], &mut diagram, limits)
                .unwrap();
            assert_eq!(reduced.rows.len(), rank);
        }
        assert_eq!(cache.work(), (2, 0));
    }

    #[test]
    fn cached_children_keep_independent_modes_until_guarded_parent_seam() {
        let cache = ModuleRelations::default();
        let mut diagram = BooleanDecisionDiagram::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let mut compact = Vec::new();
        let mut original = Vec::new();
        let mut binding = (0..9).map(Some).collect::<Vec<_>>();
        for arms in [[0, 2], [4, 6]] {
            let row = BooleanRow::from_terms(ONE, arms.map(|column| (column, ONE)));
            original.push(row.clone());
            let child = cache
                .reduce_compact(
                    vec![row],
                    vec![arms.map(|column| (column, ONE)).to_vec()],
                    &[0, 1, 2, 3, 4],
                    &mut diagram,
                    limits,
                )
                .unwrap();
            for (column, alias) in child.aliases {
                binding[column] = alias;
            }
            compact.extend(child.rows);
        }
        assert_eq!(cache.work(), (1, 1));
        assert_eq!(binding[8], Some(8), "unused zero slot remains caller-owned");
        let query = vec![(2, guard), (6, guard)];
        let expected =
            close_equations(original, std::slice::from_ref(&query), &mut diagram, limits).unwrap();
        let mapped = query
            .into_iter()
            .map(|(column, weight)| (binding[column].unwrap(), weight))
            .collect();
        let parent = close_relation(compact, &[mapped], &mut diagram, limits).unwrap();
        let mut actual = expand(&parent);
        for row in &mut actual {
            row.set(2, row.get(0));
            row.set(6, row.get(4));
            assert_eq!(row.get(8), ZERO);
        }
        assert_eq!(explicit(&actual), explicit(&expected));
        for (enabled, rank) in [(false, 2), (true, 1)] {
            let mut span = HashSet::from_iter([0u16]);
            for row in actual
                .iter()
                .filter(|row| diagram.evaluate(row.active, |_| enabled))
            {
                let word = (0..9).fold(0u16, |word, column| {
                    word | (u16::from(diagram.evaluate(row.get(column), |_| enabled)) << column)
                });
                span.extend(span.clone().into_iter().map(|value| value ^ word));
            }
            assert_eq!(span.len(), 1 << rank, "parent selector {enabled}");
        }
    }

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
            let seams = vec![(columns[0], columns[1], ZERO)];
            let mut rank = vec![0; 3];
            for (index, &column) in columns.iter().enumerate() {
                rank[column] = index;
            }
            let mut expected = close_seams(rows.clone(), &seams, &mut diagram, limits).unwrap();
            let mut actual = cache
                .reduce(rows, seams, &rank, &mut diagram, limits)
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
        assert_eq!(cache.work(), (2, 2));
    }

    fn explicit(rows: &[BooleanRow]) -> Vec<(DecisionId, Terms)> {
        rows.iter()
            .map(|row| (row.active, row.terms().to_vec()))
            .collect()
    }

    fn inputs(
        diagram: &mut BooleanDecisionDiagram,
        variables: [usize; 2],
    ) -> (Vec<BooleanRow>, Vec<Seam>) {
        let guard = diagram.make_node(variables[0], ZERO, ONE).unwrap();
        let absent = diagram.negate(guard).unwrap();
        let hadamard = diagram.make_node(variables[1], ZERO, ONE).unwrap();
        (
            vec![
                BooleanRow::from_terms(guard, [(0, ONE), (4, ONE)]),
                BooleanRow::from_terms(absent, [(1, ONE), (5, ONE)]),
                BooleanRow::from_terms(ONE, [(2, ONE), (6, ONE)]),
                BooleanRow::from_terms(ONE, [(3, ONE), (7, ONE)]),
                BooleanRow::from_terms(ONE, [(4, ONE), (5, ONE)]),
            ],
            vec![(0, 1, hadamard)],
        )
    }

    #[test]
    fn different_interior_support_and_variable_aliases_do_not_match() {
        let cache = ModuleRelations::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut diagram = BooleanDecisionDiagram::default();
        let (rows, seams) = inputs(&mut diagram, [0, 1]);
        cache
            .reduce(rows, seams, &[0, 1, 2, 3], &mut diagram, limits)
            .unwrap();
        for shared in [false, true] {
            let (mut rows, seams) = inputs(&mut diagram, if shared { [7, 7] } else { [7, 9] });
            if !shared {
                rows[0].set(6, ONE);
            }
            let expected = close_seams(rows.clone(), &seams, &mut diagram, limits).unwrap();
            let actual = cache
                .reduce(rows, seams, &[0, 1, 2, 3], &mut diagram, limits)
                .unwrap();
            assert_eq!(explicit(&actual), explicit(&expected));
        }
        assert_eq!(cache.work(), (6, 0));
    }

    #[test]
    fn scopes_without_seams_only_remove_inactive_rows() {
        let limits = ModuleCertificationLimits::DEFAULT;
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
        let actual = ModuleRelations::default()
            .reduce(rows, Vec::new(), &[], &mut diagram, limits)
            .unwrap();
        assert_eq!(explicit(&actual), explicit(&expected));
    }

    #[test]
    fn cache_hits_honor_witness_limits() {
        let cache = ModuleRelations::default();
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut diagram = BooleanDecisionDiagram::default();
        let (rows, seams) = inputs(&mut diagram, [0, 1]);
        cache
            .reduce(
                rows.clone(),
                seams.clone(),
                &[0, 1, 2, 3],
                &mut diagram,
                limits,
            )
            .unwrap();
        assert!(matches!(
            cache.reduce(
                rows,
                seams,
                &[0, 1, 2, 3],
                &mut diagram,
                ModuleCertificationLimits {
                    max_witness_nodes: 0,
                    ..limits
                }
            ),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "guarded correlation coefficients",
                    limit: 0,
                    ..
                }
            ))
        ));
        assert_eq!(cache.work(), (2, 2));
    }

    #[test]
    fn cache_misses_use_only_the_callers_remaining_work() {
        for (caller_limit, core_limit) in [(400, 400), (400, 800), (800, 400)] {
            let limits = ModuleCertificationLimits {
                max_boolean_steps: core_limit,
                ..ModuleCertificationLimits::DEFAULT
            };
            let mut caller_limits = limits.boolean_limits();
            caller_limits.max_steps = caller_limit;
            let mut diagram = BooleanDecisionDiagram::with_limits(caller_limits);
            let x = diagram.make_node(0, ZERO, ONE).unwrap();
            let absent = diagram.negate(x).unwrap();
            let y = diagram.make_node(1, ZERO, ONE).unwrap();
            diagram.charge(200 - diagram.steps()).unwrap();
            let rows = vec![
                BooleanRow::from_terms(x, [(0, ONE), (4, y)]),
                BooleanRow::from_terms(absent, [(0, ONE), (2, ONE)]),
            ];
            let cache = ModuleRelations::default();
            let error = cache
                .reduce(rows, vec![(0, 1, y)], &[0, 1, 2], &mut diagram, limits)
                .unwrap_err();
            let BlockGraphError::Stabilizer(StabilizerError::ResourceLimited {
                phase: "Boolean work steps",
                observed,
                limit,
            }) = error
            else {
                panic!("{error:?}")
            };
            assert_eq!(limit, caller_limit.min(core_limit));
            // Each charge in this small recipe is at most 30 steps. A fresh local
            // quota instead completes the solve before submitting a much larger bill.
            assert!(
                observed <= limit + 30,
                "late budget enforcement: {observed}"
            );
            assert_eq!(cache.work(), (0, 0));
        }
    }
}
