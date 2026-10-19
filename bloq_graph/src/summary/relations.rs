//! Complete factored relations shared by summary presentation and production.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanRow, BooleanRowSpace, DECISION_FALSE as ZERO,
    DECISION_TRUE as ONE, DecisionId, DecisionNode,
};

use crate::{BlockGraphError, ModuleCertificationLimits, StabilizerError};

type Terms = Vec<(usize, DecisionId)>;
pub(crate) type Seam = (usize, usize, DecisionId);

/// Column and Boolean-variable names are local to this exact reduction recipe.
/// Full rows retain interior support; external half-edges are never projected out.
#[derive(Debug, PartialEq, Eq, Hash)]
struct RelationKey {
    nodes: Vec<DecisionNode>,
    rows: Vec<(DecisionId, Terms)>,
    seams: Vec<Seam>,
}

#[derive(Debug)]
struct ReducedRelation {
    nodes: Vec<DecisionNode>,
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

    fn cached_reduction(
        &self,
        key: RelationKey,
        equations: usize,
        solve: impl FnOnce(&RelationKey) -> Result<ReducedRelation, BlockGraphError>,
    ) -> Result<Arc<ReducedRelation>, BlockGraphError> {
        let cached = self
            .reductions
            .lock()
            .expect("module relation cache poisoned")
            .get(&key)
            .cloned();
        if let Some(reduced) = cached {
            self.reused.fetch_add(equations, Ordering::Relaxed);
            return Ok(reduced);
        }
        // Independent callers may solve the same miss; only publication is locked.
        let reduced = Arc::new(solve(&key)?);
        self.eliminated.fetch_add(equations, Ordering::Relaxed);
        let mut cache = self
            .reductions
            .lock()
            .expect("module relation cache poisoned");
        Ok(Arc::clone(cache.entry(key).or_insert(reduced)))
    }

    pub(crate) fn reduce(
        &self,
        rows: Vec<BooleanRow>,
        seams: Vec<Seam>,
        column_rank: &[usize],
        diagram: &mut BooleanDecisionDiagram,
        limits: ModuleCertificationLimits,
    ) -> Result<Vec<BooleanRow>, BlockGraphError> {
        if seams.is_empty() {
            return close_seams(rows, &seams, diagram, limits);
        }
        let terms = rows
            .iter()
            .map(|row| row.terms().len().saturating_add(1))
            .fold(0usize, usize::saturating_add)
            .saturating_add(seams.len().saturating_mul(3));
        diagram.charge(
            terms
                .saturating_mul(terms.max(1).ilog2() as usize + 1)
                .saturating_mul(3),
        )?;
        let used = rows
            .iter()
            .flat_map(BooleanRow::terms)
            .map(|&(column, _)| column / 2)
            .chain(seams.iter().flat_map(|&(left, right, _)| [left, right]))
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
            .chain(seams.iter().map(|&(_, _, root)| root))
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
                .map(|&(column, value)| {
                    (2 * column_map[&(column / 2)] + column % 2, imported[&value])
                })
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
            // Seam and axis order retain the source pivot convention.
            seams: seams
                .iter()
                .map(|&(left, right, value)| {
                    (column_map[&left], column_map[&right], imported[&value])
                })
                .collect(),
        };
        let equations = seams.len().saturating_mul(2);
        let reduced = self.cached_reduction(key, equations, |key| {
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
            let mut rows = close_seams(
                key.rows
                    .iter()
                    .map(|(active, terms)| BooleanRow::from_terms(*active, terms.clone()))
                    .collect(),
                &key.seams,
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
    mut rows: Vec<BooleanRow>,
    seams: &[Seam],
    diagram: &mut BooleanDecisionDiagram,
    limits: ModuleCertificationLimits,
) -> Result<Vec<BooleanRow>, BlockGraphError> {
    if seams.is_empty() {
        diagram.charge(rows.len())?;
        rows.retain(|row| row.active != ZERO);
        return Ok(rows);
    }
    let mut rows = BooleanRowSpace::new(rows, diagram)?;
    for &(left, right, hadamard) in seams {
        for axis in 0..2 {
            // First-row pivot priority preserves authored outcome/frame conventions.
            rows.eliminate_linear(
                &[
                    (2 * left + axis, ONE),
                    (2 * right + axis, ONE),
                    (2 * right, hadamard),
                    (2 * right + 1, hadamard),
                ],
                diagram,
            )?;
            if diagram.nodes().len() > limits.max_witness_nodes {
                return Err(StabilizerError::ResourceLimited {
                    phase: "guarded correlation coefficients",
                    observed: diagram.nodes().len(),
                    limit: limits.max_witness_nodes,
                }
                .into());
            }
        }
    }
    Ok(rows.into_rows(diagram)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_key() -> RelationKey {
        RelationKey {
            nodes: Vec::new(),
            rows: Vec::new(),
            seams: Vec::new(),
        }
    }

    fn empty_relation() -> ReducedRelation {
        ReducedRelation {
            nodes: Vec::new(),
            rows: Vec::new(),
        }
    }

    #[test]
    fn panicking_solver_does_not_poison_or_publish_the_cache() {
        let cache = ModuleRelations::default();
        let panic = std::panic::catch_unwind(|| {
            cache.cached_reduction(empty_key(), 2, |_| {
                drop(cache.reductions.try_lock().unwrap());
                panic!("solver panicked")
            })
        });
        panic.unwrap_err();
        assert!(cache.reductions.lock().unwrap().is_empty());
        assert_eq!(cache.work(), (0, 0));
        cache
            .cached_reduction(empty_key(), 2, |_| Ok(empty_relation()))
            .unwrap();
        assert_eq!(cache.work(), (2, 0));
    }

    #[test]
    fn racing_cache_misses_solve_concurrently_and_publish_one_relation() {
        use std::sync::mpsc::channel;
        use std::time::Duration;

        let cache = ModuleRelations::default();
        let timeout = Duration::from_secs(5);
        std::thread::scope(|scope| {
            let (started, starts) = channel();
            let mut releases = Vec::new();
            let mut workers = Vec::new();
            for _ in 0..2 {
                let (release, released) = channel();
                releases.push(release);
                let started = started.clone();
                let cache = &cache;
                workers.push(scope.spawn(move || {
                    cache
                        .cached_reduction(empty_key(), 2, |_| {
                            started.send(()).unwrap();
                            released.recv_timeout(timeout).unwrap();
                            Ok(empty_relation())
                        })
                        .unwrap()
                }));
            }
            for _ in 0..2 {
                starts.recv_timeout(timeout).unwrap();
            }
            for release in releases {
                release.send(()).unwrap();
            }
            let mut results = workers.into_iter().map(|worker| worker.join().unwrap());
            assert!(Arc::ptr_eq(
                &results.next().unwrap(),
                &results.next().unwrap()
            ));
        });
        assert_eq!(cache.reductions.lock().unwrap().len(), 1);
        // Both misses performed elimination; only subsequent hits reuse work.
        assert_eq!(cache.work(), (4, 0));
        cache
            .cached_reduction(empty_key(), 2, |_| panic!("cache hit must not solve"))
            .unwrap();
        assert_eq!(cache.work(), (4, 2));
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
            assert!(cache.reductions.lock().unwrap().is_empty());
        }
    }
}
