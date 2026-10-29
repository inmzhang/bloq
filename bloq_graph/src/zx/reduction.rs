//! Lossless raw-coordinate quotient for a fixed computational ZX diagram.

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanRow, BooleanRowSpace, DECISION_TRUE as ONE,
};
use bloq_utils::{Pauli, PauliString};
use petgraph::unionfind::UnionFind;
use std::collections::BTreeSet;

use super::modular::try_leading_pivot_elimination_from;
use super::stabilizer::CoeffVec;
use super::{NodeKind, ZXGraph};
use crate::{CancellationToken, ModuleCertificationLimits, StabilizerError};

/// Original tensor coordinates remain queries into a smaller, complete relation.
/// Class ids name individual binary coordinates, not packed Pauli columns.
pub(crate) struct CorrelationReduction {
    native_bits: Vec<[Option<usize>; 2]>,
    reversed_hadamard: Vec<bool>,
    classes: usize,
    pub(crate) rows: Vec<BooleanRow>,
}

/// Lift compact coordinates through each raw column's `[X, Z]` queries.
pub(super) fn lift_queries(
    queries: &[[Option<usize>; 2]],
    component: impl Fn(usize) -> bool,
) -> PauliString {
    PauliString::from_terms(
        queries.len(),
        queries.iter().enumerate().map(|(column, axes)| {
            let [x, z] = axes.map(|bit| bit.is_some_and(&component));
            (column, Pauli::from_xz(x, z))
        }),
    )
}

/// First raw `(column, axis)` query of each compact coordinate.
pub(super) fn raw_query_order(
    queries: &[[Option<usize>; 2]],
    width: usize,
) -> Vec<Option<(usize, Pauli)>> {
    let mut order = vec![None; width];
    for (column, axes) in queries.iter().enumerate() {
        for (axis, query) in [Pauli::X, Pauli::Z].into_iter().zip(axes) {
            if let Some(bit) = *query {
                order[bit].get_or_insert((column, axis));
            }
        }
    }
    order
}

/// Leading-pivot elimination whose pivots follow raw query order; returns the rank.
pub(super) fn eliminate_in_raw_order<E>(
    rows: &mut [CoeffVec],
    queries: &[[Option<usize>; 2]],
    order: &[Option<(usize, Pauli)>],
    checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<usize, E> {
    try_leading_pivot_elimination_from(
        rows,
        0,
        |row| {
            row.iter_ones()
                .filter_map(|bit| order[bit])
                .min_by_key(|&(column, axis)| (column, axis == Pauli::Z))
        },
        |row, &(column, axis)| {
            queries[column][usize::from(axis == Pauli::Z)].is_some_and(|bit| row.bit(bit))
        },
        checkpoint,
    )
}

/// Projection and its inverse tape preserve every solution of the original
/// parity equations, including independent closed/internal modes.
struct ParityReduction {
    eliminated: Vec<bool>,
    equations: Vec<Vec<usize>>,
    recovery: Vec<(usize, Vec<usize>)>,
}

impl ParityReduction {
    fn new(
        zx: &ZXGraph,
        native_bits: &[[Option<usize>; 2]],
        classes: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, StabilizerError> {
        diagram.charge(zx.nodes().len())?;
        let mut equations = Vec::new();
        for node in zx.nodes() {
            CancellationToken::checkpoint()?;
            if !matches!(node.kind, NodeKind::X | NodeKind::Z) {
                continue;
            }
            let cross = usize::from(node.kind == NodeKind::Z);
            let degree = zx.neighbor_ids(node.id).len();
            if degree == 0 || degree == 2 {
                continue;
            }
            diagram.charge(degree.saturating_mul(degree.max(1).ilog2() as usize + 3))?;
            let mut terms = zx
                .neighbor_edges(node.id)
                .filter_map(|(_, edge)| native_bits[edge][cross])
                .collect::<Vec<_>>();
            cancel_repeated_terms(&mut terms);
            equations.push(terms);
        }
        Self::from_equations(equations, classes, diagram)
    }

    fn from_equations(
        equations: Vec<Vec<usize>>,
        classes: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, StabilizerError> {
        diagram.charge(
            equations
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                .saturating_mul(2)
                .saturating_add(equations.len().saturating_mul(6))
                .saturating_add(classes.saturating_mul(6)),
        )?;
        let mut incidence = vec![[usize::MAX; 2]; classes];
        let mut counts = vec![0u8; classes];
        for (node, terms) in equations.iter().enumerate() {
            CancellationToken::checkpoint()?;
            for &class in terms {
                if counts[class] < 2 {
                    incidence[class][usize::from(counts[class])] = node;
                }
                counts[class] = counts[class].saturating_add(1);
            }
        }
        let mut components = UnionFind::new(equations.len());
        let mut adjacent = vec![Vec::new(); equations.len()];
        let mut eliminated = vec![false; classes];
        for class in 0..classes {
            CancellationToken::checkpoint()?;
            let [left, right] = incidence[class];
            // A tree of two-incidence columns has a unique lift in any native frame.
            if counts[class] != 2 || components.equiv(left, right) {
                continue;
            }
            diagram.charge(4)?;
            components.union(left, right);
            eliminated[class] = true;
            adjacent[left].push((right, class));
            adjacent[right].push((left, class));
        }
        let mut visited = vec![false; equations.len()];
        let mut residual = Vec::new();
        let mut recovery = Vec::new();
        for root in 0..equations.len() {
            CancellationToken::checkpoint()?;
            if visited[root] {
                continue;
            }
            visited[root] = true;
            let mut pending = vec![(root, None)];
            let mut order = Vec::new();
            let mut aggregate = Vec::new();
            while let Some((node, parent)) = pending.pop() {
                CancellationToken::checkpoint()?;
                diagram.charge(
                    equations[node]
                        .len()
                        .saturating_add(adjacent[node].len())
                        .saturating_add(2),
                )?;
                order.push((node, parent));
                aggregate.extend_from_slice(&equations[node]);
                for &(neighbor, class) in &adjacent[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        pending.push((neighbor, Some(class)));
                    }
                }
            }
            diagram.charge(
                aggregate
                    .len()
                    .saturating_mul(aggregate.len().max(1).ilog2() as usize + 2),
            )?;
            cancel_repeated_terms(&mut aggregate);
            debug_assert!(aggregate.iter().all(|&class| !eliminated[class]));
            if !aggregate.is_empty() {
                residual.push(aggregate);
            }
            // Child equations recover parent edges after all their descendants.
            // Chords are never selected, preserving the closed/internal modes.
            for (node, parent) in order.into_iter().rev() {
                if let Some(parent) = parent {
                    diagram.charge(equations[node].len())?;
                    recovery.push((
                        parent,
                        equations[node]
                            .iter()
                            .copied()
                            .filter(|&class| class != parent)
                            .collect(),
                    ));
                }
            }
        }
        Ok(Self {
            eliminated,
            equations: residual,
            recovery,
        })
    }

    /// Eliminate local equations when substitution decreases residual occurrences.
    fn simplify(&mut self, diagram: &mut BooleanDecisionDiagram) -> Result<(), StabilizerError> {
        CancellationToken::checkpoint()?;
        self.deduplicate_equations(diagram)?;
        if self.equations.is_empty() {
            return Ok(());
        }
        let mut cofactors = crate::FxHashMap::default();
        for terms in &mut self.equations {
            CancellationToken::checkpoint()?;
            // Match local bialgebra twins without expanding large forest sums.
            if !(3..=4).contains(&terms.len()) {
                continue;
            }
            diagram.charge(terms.len().saturating_mul(terms.len()).saturating_mul(3))?;
            let mut equality = None;
            for &class in terms.iter() {
                let cofactor = terms
                    .iter()
                    .copied()
                    .filter(|&term| term != class)
                    .collect::<Vec<_>>();
                let &mut previous = cofactors.entry(cofactor).or_insert(class);
                if previous != class {
                    equality = Some(vec![previous.min(class), previous.max(class)]);
                    break;
                }
            }
            // Earlier original equations remain in the span of earlier new ones.
            if let Some(equality) = equality {
                *terms = equality;
            }
        }
        drop(cofactors);
        diagram.charge(
            self.eliminated.len().saturating_add(
                self.equations
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>()
                    .saturating_mul(self.equations.len().max(1).ilog2() as usize + 3),
            ),
        )?;
        let mut incidence = vec![BTreeSet::new(); self.eliminated.len()];
        let mut short = Vec::new();
        let mut ternary = Vec::new();
        for (index, terms) in self.equations.iter().enumerate() {
            CancellationToken::checkpoint()?;
            if (1..=2).contains(&terms.len()) {
                short.push(index);
            } else if terms.len() == 3 {
                ternary.push(index);
            }
            for &class in terms {
                incidence[class].insert(index);
            }
        }
        let mut leaves = incidence
            .iter()
            .enumerate()
            .filter_map(|(class, rows)| (rows.len() == 1).then_some(class))
            .collect::<Vec<_>>();
        let incidence_cost = self.equations.len().max(1).ilog2() as usize + 3;
        let mut recovery = Vec::new();
        loop {
            CancellationToken::checkpoint()?;
            diagram.charge(1)?;
            let (source, pivot) = if let Some(index) = short.pop() {
                let terms = &self.equations[index];
                if !(1..=2).contains(&terms.len()) {
                    continue;
                }
                let &pivot = terms
                    .iter()
                    .min_by_key(|&&class| (incidence[class].len(), class))
                    .expect("nonempty short equation");
                (index, pivot)
            } else if let Some(pivot) = leaves.pop() {
                if incidence[pivot].len() != 1 {
                    continue;
                }
                (
                    *incidence[pivot]
                        .first()
                        .expect("degree-one variable has an equation"),
                    pivot,
                )
            } else if let Some(index) = ternary.pop() {
                let terms = &self.equations[index];
                if terms.len() != 3 {
                    continue;
                }
                diagram.charge(3)?;
                let Some(pivot) = terms
                    .iter()
                    .copied()
                    .filter(|&class| incidence[class].len() == 3)
                    .min()
                else {
                    continue;
                };
                // Removing this row and substituting into two others loses at least one term.
                (index, pivot)
            } else {
                break;
            };
            diagram.charge(self.equations[source].len())?;
            let query = self.equations[source]
                .iter()
                .copied()
                .filter(|&class| class != pivot)
                .collect::<Vec<_>>();
            debug_assert!(!self.eliminated[pivot]);
            self.eliminated[pivot] = true;
            for index in std::mem::take(&mut incidence[pivot]) {
                CancellationToken::checkpoint()?;
                let terms = &mut self.equations[index];
                diagram.charge(
                    terms
                        .len()
                        .saturating_mul(if index == source {
                            1
                        } else {
                            query.len().saturating_add(1)
                        })
                        .saturating_add(query.len().saturating_mul(incidence_cost)),
                )?;
                if index == source {
                    terms.clear();
                } else {
                    // Longer leaf queries have no other incidences to substitute.
                    debug_assert!(query.len() <= 2);
                    terms.remove(
                        terms
                            .binary_search(&pivot)
                            .expect("incidence identifies a present variable"),
                    );
                }
                for &class in &query {
                    if index == source {
                        incidence[class].remove(&index);
                    } else {
                        match terms.binary_search(&class) {
                            Ok(offset) => {
                                terms.remove(offset);
                                incidence[class].remove(&index);
                            }
                            Err(offset) => {
                                terms.insert(offset, class);
                                incidence[class].insert(index);
                            }
                        }
                    }
                    if incidence[class].len() == 1 {
                        leaves.push(class);
                    } else if incidence[class].len() == 3 {
                        diagram.charge(3)?;
                        ternary.extend(incidence[class].iter().copied());
                    }
                }
                if (1..=2).contains(&terms.len()) {
                    short.push(index);
                } else if terms.len() == 3 {
                    ternary.push(index);
                }
            }
            recovery.push((pivot, query));
        }
        self.deduplicate_equations(diagram)?;
        recovery.reverse();
        recovery.append(&mut self.recovery);
        self.recovery = recovery;
        Ok(())
    }

    fn deduplicate_equations(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), StabilizerError> {
        let mut unique = rustc_hash::FxHashSet::default();
        for terms in &mut self.equations {
            CancellationToken::checkpoint()?;
            diagram.charge(terms.len().saturating_mul(2).saturating_add(1))?;
            if !unique.insert(terms.clone()) {
                terms.clear();
            }
        }
        self.equations.retain(|terms| !terms.is_empty());
        Ok(())
    }

    /// Consume the local tape before signature interning or planner integration.
    fn lift(
        self,
        rows: &mut [BooleanRow],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), StabilizerError> {
        if self.recovery.is_empty() {
            return Ok(());
        }
        diagram.charge(self.eliminated.len())?;
        let tape = self.recovery.iter().fold(0usize, |sum, (_, query)| {
            sum.saturating_add(query.len() + 1)
        });
        let mut values = vec![false; self.eliminated.len()];
        for row in rows {
            CancellationToken::checkpoint()?;
            diagram.charge(row.terms().len().saturating_add(tape))?;
            for &(class, value) in row.terms() {
                debug_assert_eq!(value, ONE);
                values[class] = true;
            }
            let active = row.active;
            let mut terms = std::mem::take(row).into_terms();
            for (parent, query) in &self.recovery {
                let value = query.iter().fold(false, |sum, &class| sum ^ values[class]);
                debug_assert!(!values[*parent]);
                values[*parent] = value;
                if value {
                    terms.push((*parent, ONE));
                }
            }
            diagram.charge(
                terms
                    .len()
                    .saturating_mul(2 * terms.len().max(1).ilog2() as usize + 3),
            )?;
            for &(class, _) in &terms {
                values[class] = false;
            }
            // Sort once instead of shifting the sparse row for every recovered bit.
            terms.sort_unstable_by_key(|&(class, _)| class);
            *row = BooleanRow::from_terms(active, terms);
        }
        Ok(())
    }
}

pub(super) fn cancel_repeated_terms(terms: &mut Vec<usize>) {
    terms.sort_unstable();
    let mut read = 0;
    let mut write = 0;
    while read < terms.len() {
        let class = terms[read];
        let start = read;
        while read < terms.len() && terms[read] == class {
            read += 1;
        }
        if (read - start) % 2 == 1 {
            terms[write] = class;
            write += 1;
        }
    }
    terms.truncate(write);
}

impl CorrelationReduction {
    /// Quotient tensor equalities, then equal columns of the complete residual relation.
    /// Non-dangling special sites are declined.
    pub(crate) fn new(
        zx: &ZXGraph,
        diagram: &mut BooleanDecisionDiagram,
        limits: ModuleCertificationLimits,
    ) -> Result<Option<Self>, StabilizerError> {
        CancellationToken::checkpoint()?;
        let width = zx.total_ids();
        if width > limits.max_local_columns || width > (usize::MAX - 1) / 2 {
            return Err(StabilizerError::ResourceLimited {
                phase: "correlation reduction columns",
                observed: width,
                limit: limits.max_local_columns.min((usize::MAX - 1) / 2),
            });
        }
        let bits = width * 2;
        diagram.charge(bits.saturating_mul(bits.max(1).ilog2() as usize + 8))?;
        if zx.nodes().len().checked_add(zx.edges().len()) != Some(width)
            || zx.edges().iter().enumerate().any(|(index, edge)| {
                edge.id != zx.nodes().len() + index
                    || edge.n1 >= zx.nodes().len()
                    || edge.n2 >= zx.nodes().len()
                    || edge.n1 == edge.n2
            })
            || zx.nodes().iter().any(|node| {
                !matches!(node.kind, NodeKind::X | NodeKind::Z)
                    && zx.neighbor_ids(node.id).len() != 1
            })
        {
            return Ok(None);
        }

        // Half-edge atoms use their source's native frame; the last atom is zero.
        let zero = bits;
        let mut equal = UnionFind::<usize>::new(bits + 1);
        for edge in zx.edges().iter().filter(|edge| edge.n1 < edge.n2) {
            CancellationToken::checkpoint()?;
            let Some(reverse) = zx.edge_between(edge.n2, edge.n1) else {
                return Ok(None);
            };
            if reverse.hadamard != edge.hadamard {
                return Ok(None);
            }
            for axis in 0..2 {
                equal.union(
                    2 * edge.id + axis,
                    2 * reverse.id + (axis ^ usize::from(edge.hadamard)),
                );
            }
        }
        for node in zx.nodes() {
            CancellationToken::checkpoint()?;
            let mut arms = zx.neighbor_edges(node.id);
            match node.kind {
                NodeKind::X | NodeKind::Z => {
                    let cross = usize::from(node.kind == NodeKind::Z);
                    let broadcast = 1 - cross;
                    equal.union(2 * node.id + cross, zero);
                    if let Some((_, first)) = arms.next() {
                        equal.union(2 * node.id + broadcast, 2 * first + broadcast);
                        for (_, edge) in arms {
                            equal.union(2 * first + broadcast, 2 * edge + broadcast);
                            if zx.neighbor_ids(node.id).len() == 2 {
                                equal.union(2 * first + cross, 2 * edge + cross);
                            }
                        }
                    }
                }
                NodeKind::Y | NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => {
                    let (_, arm) = arms.next().expect("special sites have one arm");
                    for axis in 0..2 {
                        equal.union(2 * node.id + axis, 2 * arm + axis);
                    }
                    if node.kind == NodeKind::Y {
                        equal.union(2 * node.id, 2 * node.id + 1);
                    }
                }
            }
        }

        let zero = equal.find_mut(zero);
        let mut labels = vec![None; bits + 1];
        let mut classes = 0usize;
        let native_bits = (0..width)
            .map(|column| {
                [0, 1].map(|axis| {
                    let root = equal.find_mut(2 * column + axis);
                    if root == zero {
                        None
                    } else {
                        Some(*labels[root].get_or_insert_with(|| {
                            let class = classes;
                            classes += 1;
                            class
                        }))
                    }
                })
            })
            .collect::<Vec<_>>();
        diagram.charge(classes.saturating_mul(4))?;
        let mut reduction = ParityReduction::new(zx, &native_bits, classes, diagram)?;
        reduction.simplify(diagram)?;
        let candidates = (0..classes)
            .filter(|&class| !reduction.eliminated[class])
            .map(|column| BooleanRow::from_terms(ONE, [(column, ONE)]))
            .collect();
        let mut rows = BooleanRowSpace::new(candidates, diagram)?;
        for terms in &reduction.equations {
            CancellationToken::checkpoint()?;
            diagram.charge(terms.len())?;
            let terms = terms.iter().map(|&class| (class, ONE)).collect::<Vec<_>>();
            rows.eliminate_linear(&terms, diagram)?;
        }
        if diagram.nodes().len() > limits.max_witness_nodes {
            return Err(StabilizerError::ResourceLimited {
                phase: "correlation reduction coefficients",
                observed: diagram.nodes().len(),
                limit: limits.max_witness_nodes,
            });
        }
        let reversed_hadamard = std::iter::repeat_n(false, zx.nodes().len())
            .chain(
                zx.edges()
                    .iter()
                    .map(|edge| edge.hadamard && edge.n1 > edge.n2),
            )
            .collect();
        let mut rows = rows.into_rows(diagram)?;
        reduction.lift(&mut rows, diagram)?;
        let mut reduced = Self {
            native_bits,
            reversed_hadamard,
            classes,
            rows,
        };
        reduced.coalesce_equal_coordinates(diagram)?;
        Ok(Some(reduced))
    }

    fn coalesce_equal_coordinates(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), StabilizerError> {
        // Full rank already separates every nonzero coordinate functional.
        if self.rows.len() == self.classes {
            return Ok(());
        }
        diagram.charge(
            self.classes
                .saturating_mul(4)
                .saturating_add(self.rows.len()),
        )?;
        // These aliases apply only to members of this complete raw relation,
        // including its closed/internal modes and certified native gauge rows.
        let mut signatures = vec![Vec::new(); self.classes];
        for (index, row) in self.rows.iter().enumerate() {
            CancellationToken::checkpoint()?;
            diagram.charge(row.terms().len().saturating_mul(3))?;
            for &(column, coefficient) in row.terms() {
                debug_assert_eq!(coefficient, ONE);
                signatures[column].push(index);
            }
        }
        let mut unique = crate::FxHashMap::default();
        let mut aliases = Vec::with_capacity(self.classes);
        let mut representatives = Vec::new();
        for (column, signature) in signatures.into_iter().enumerate() {
            CancellationToken::checkpoint()?;
            diagram.charge(signature.len().saturating_mul(2).saturating_add(1))?;
            let alias = if signature.is_empty() {
                None
            } else {
                let alias = *unique.entry(signature).or_insert_with(|| {
                    representatives.push(column);
                    representatives.len() - 1
                });
                Some(alias)
            };
            aliases.push(alias);
        }
        if unique.len() == self.classes {
            return Ok(());
        }
        self.classes = unique.len();
        drop(unique);
        diagram.charge(self.native_bits.len().saturating_mul(2))?;
        for axes in &mut self.native_bits {
            CancellationToken::checkpoint()?;
            for axis in axes {
                *axis = axis.and_then(|column| aliases[column]);
            }
        }
        for row in &mut self.rows {
            CancellationToken::checkpoint()?;
            diagram.charge(
                row.terms()
                    .len()
                    .saturating_mul(row.terms().len().max(1).ilog2() as usize + 2),
            )?;
            let active = row.active;
            // Equal columns carry one support value; XOR would erase it.
            let terms =
                std::mem::take(row)
                    .into_terms()
                    .into_iter()
                    .filter_map(|(column, coefficient)| {
                        aliases[column]
                            .filter(|&alias| representatives[alias] == column)
                            .map(|alias| (alias, coefficient))
                    });
            *row = BooleanRow::from_terms(active, terms);
        }
        Ok(())
    }

    pub(crate) fn class_count(&self) -> usize {
        self.classes
    }

    /// A directed half-edge query in its source node's native frame.
    pub(crate) fn native_bit(&self, column: usize, axis: Pauli) -> Option<usize> {
        assert!(matches!(axis, Pauli::X | Pauli::Z));
        self.native_bits[column][usize::from(axis == Pauli::Z)]
    }

    /// A raw graph-table query, whose half-edges use the smaller endpoint's frame.
    #[cfg(test)]
    fn original_bit(&self, column: usize, axis: Pauli) -> Option<usize> {
        self.native_bit(
            column,
            if self.reversed_hadamard[column] {
                axis.flip()
            } else {
                axis
            },
        )
    }

    /// Canonicalize the complete relation in original raw query order before
    /// its single dense lift. Closed pivots also reduce earlier boundary rows.
    pub(crate) fn into_canonical_rows(mut self) -> Result<Vec<PauliString>, StabilizerError> {
        CancellationToken::checkpoint()?;
        let words = self.classes.div_ceil(64);
        if self.rows.len().checked_mul(words).is_none() {
            return Err(StabilizerError::ResourceLimited {
                phase: "runtime correlation matrix words",
                observed: usize::MAX,
                limit: usize::MAX,
            });
        }
        // Raw graph-table queries use the smaller endpoint's frame.
        for (axes, reversed) in self.native_bits.iter_mut().zip(&self.reversed_hadamard) {
            if *reversed {
                axes.swap(0, 1);
            }
        }
        let queries = self.native_bits;
        let order = raw_query_order(&queries, self.classes);
        debug_assert!(
            order.iter().all(Option::is_some),
            "every class has an original query"
        );
        let mut rows = self
            .rows
            .into_iter()
            .map(|row| {
                CancellationToken::checkpoint()?;
                debug_assert_eq!(row.active, ONE);
                let mut packed = CoeffVec::zeros(self.classes);
                for (class, value) in row.into_terms() {
                    debug_assert_eq!(value, ONE);
                    packed.set_bit(class, true);
                }
                Ok(packed)
            })
            .collect::<Result<Vec<_>, StabilizerError>>()?;
        let rank =
            eliminate_in_raw_order(&mut rows, &queries, &order, CancellationToken::checkpoint)?;
        rows.truncate(rank);
        rows.iter()
            .map(|row| {
                CancellationToken::checkpoint()?;
                Ok(lift_queries(&queries, |class| row.bit(class)))
            })
            .collect()
    }

    /// Crossing centers belong to the original graph's nonlinear physical
    /// materialization, not this linear raw-coordinate lift.
    #[cfg(test)]
    fn lift(&self, row: &BooleanRow) -> PauliString {
        assert_eq!(row.active, ONE);
        assert!(row.terms().iter().all(|&(_, value)| value == ONE));
        let queries = (0..self.native_bits.len())
            .map(|column| [Pauli::X, Pauli::Z].map(|axis| self.original_bit(column, axis)))
            .collect::<Vec<_>>();
        lift_queries(&queries, |class| row.get(class) == ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zx::graph::CsrAdjacency;
    use crate::zx::modular::canonical_external_basis;
    use crate::{ActionDag, GalleryItem, PhasedPauliString, ZXEdge, ZXNode};
    use glam::ivec3;

    fn graph(kinds: &[NodeKind], links: &[(usize, usize, bool)]) -> ZXGraph {
        let nodes = kinds
            .iter()
            .enumerate()
            .map(|(id, &kind)| ZXNode::new(id, ivec3(id as i32, 0, 0), kind))
            .collect::<Vec<_>>();
        let mut edges = Vec::new();
        let mut directed = Vec::new();
        for &(a, b, hadamard) in links {
            for (n1, n2) in [(a, b), (b, a)] {
                let id = nodes.len() + edges.len();
                edges.push(ZXEdge {
                    n1,
                    n2,
                    hadamard,
                    id,
                });
                directed.push((n1, n2, id));
            }
        }
        ZXGraph {
            adjacency: CsrAdjacency::from_edges(nodes.len(), &directed),
            total_ids: nodes.len() + edges.len(),
            pos_to_node: nodes.iter().map(|node| (node.pos, node.id)).collect(),
            nodes,
            edges,
            action_graph: ActionDag::default(),
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        }
    }

    fn assert_complete_signed_relation(zx: &ZXGraph) -> CorrelationReduction {
        let mut diagram = BooleanDecisionDiagram::default();
        let reduced =
            CorrelationReduction::new(zx, &mut diagram, ModuleCertificationLimits::DEFAULT)
                .unwrap()
                .expect("supported tensor topology");
        let lifted = reduced
            .rows
            .iter()
            .map(|row| reduced.lift(row))
            .collect::<Vec<_>>();
        let phases = zx.external_stabilizer_row_phases(&lifted);
        let actual = canonical_external_basis(
            lifted
                .into_iter()
                .zip(phases)
                .map(|(row, phase)| PhasedPauliString::new(row, phase))
                .collect(),
            zx.total_ids(),
        );
        assert_eq!(actual, zx.to_signed_external_generator_table());
        reduced
    }

    #[test]
    fn quotient_preserves_gallery_signed_relations_and_raw_order() {
        for gallery in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::T,
            GalleryItem::TWithPreparedY,
            GalleryItem::THTH,
            GalleryItem::XMemory,
            GalleryItem::YMemory,
            GalleryItem::Stability,
            GalleryItem::GHZSlideThenGlide,
        ] {
            let source = gallery.build().materialize_root_graph().unwrap();
            let zx = ZXGraph::from_block_graph_for_analysis(&source).unwrap();
            assert_eq!(
                assert_complete_signed_relation(&zx)
                    .into_canonical_rows()
                    .unwrap(),
                zx.to_external_generator_table(),
                "exact raw canonical order for {gallery:?}",
            );
        }
    }

    /// Exhaust every solution and every linear query, including queries on removed coordinates.
    fn assert_complete_parity_relation(
        equations: Vec<Vec<usize>>,
        classes: usize,
        use_forest: bool,
    ) -> (usize, usize) {
        let terms = |mask: usize| {
            (0..classes)
                .filter(|&column| mask & (1 << column) != 0)
                .collect::<Vec<_>>()
        };
        let mask = |terms: &[usize]| terms.iter().map(|&column| 1 << column).sum::<usize>();
        let parity = |query: usize, value: usize| (query & value).count_ones() % 2;
        let original_equations = equations.iter().map(|row| mask(row)).collect::<Vec<_>>();
        let mut diagram = BooleanDecisionDiagram::default();
        let mut reduction = if use_forest {
            ParityReduction::from_equations(equations, classes, &mut diagram).unwrap()
        } else {
            ParityReduction {
                eliminated: vec![false; classes],
                equations,
                recovery: Vec::new(),
            }
        };
        reduction.simplify(&mut diagram).unwrap();
        let retained = (0..classes)
            .filter(|&column| !reduction.eliminated[column])
            .collect::<Vec<_>>();
        let summary = (
            classes - retained.len(),
            reduction.equations.iter().map(Vec::len).sum(),
        );
        let retained_mask = mask(&retained);
        let residual = reduction
            .equations
            .iter()
            .map(|row| mask(row))
            .collect::<Vec<_>>();
        let projected = (0..1usize << classes)
            .filter(|&value| {
                value & !retained_mask == 0
                    && residual.iter().all(|&query| parity(query, value) == 0)
            })
            .collect::<Vec<_>>();
        let mut rows = projected
            .iter()
            .copied()
            .chain(retained.iter().map(|&column| 1 << column))
            .map(|value| {
                BooleanRow::from_terms(ONE, terms(value).into_iter().map(|column| (column, ONE)))
            })
            .collect::<Vec<_>>();
        reduction.lift(&mut rows, &mut diagram).unwrap();
        let lifted = rows
            .iter()
            .map(|row| {
                row.terms()
                    .iter()
                    .map(|&(column, _)| 1 << column)
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        let original = (0..1usize << classes)
            .filter(|&value| {
                original_equations
                    .iter()
                    .all(|&query| parity(query, value) == 0)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(projected.len(), original.len(), "complete kernel bijection");
        assert_eq!(
            lifted[..projected.len()]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            original
        );
        for (&before, &after) in projected.iter().zip(&lifted) {
            assert_eq!(after & retained_mask, before, "projection after lift");
            for query in 0..1usize << classes {
                let pulled = retained
                    .iter()
                    .zip(&lifted[projected.len()..])
                    .filter_map(|(&column, &image)| {
                        (parity(query, image) != 0).then_some(1 << column)
                    })
                    .sum::<usize>();
                assert_eq!(parity(query, after), parity(pulled, before));
            }
        }
        summary
    }

    #[test]
    fn residual_recovery_preserves_every_small_relation_and_query() {
        for encoded in 0..4096 {
            let equations = [encoded & 15, (encoded >> 4) & 15, encoded >> 8]
                .into_iter()
                .map(|mask| (0..4).filter(|&column| mask & (1 << column) != 0).collect())
                .collect();
            assert_complete_parity_relation(equations, 4, true);
        }
    }

    #[test]
    fn ternary_elimination_preserves_closed_modes_and_composes_recovery() {
        // Fano lines have arity/incidence three, and no old local rule can start.
        let fano = vec![
            vec![0, 1, 2],
            vec![0, 3, 4],
            vec![0, 5, 6],
            vec![1, 3, 5],
            vec![1, 4, 6],
            vec![2, 3, 6],
            vec![2, 4, 5],
        ];
        for reverse in [false, true] {
            let mut equations = fano.clone();
            if reverse {
                equations.reverse();
            }
            // Coordinate seven is registered but absent, hence an independent mode.
            assert!(assert_complete_parity_relation(equations.clone(), 8, false).0 > 0);
            // Split one row through a two-incidence variable to install an earlier forest tape.
            let split = equations.pop().unwrap();
            equations.push(vec![split[0], split[1], 8]);
            equations.push(vec![split[2], 8]);
            assert!(assert_complete_parity_relation(equations, 9, true).0 > 1);
        }
        for (from, to) in [(2, 0), (5, 6)] {
            let equations = fano
                .iter()
                .map(|row| {
                    let mut row = row
                        .iter()
                        .map(|&x| if x == from { to } else { x })
                        .collect();
                    cancel_repeated_terms(&mut row);
                    row
                })
                .collect();
            assert_complete_parity_relation(equations, 8, false);
        }
    }

    #[test]
    fn ternary_elimination_preserves_original_hadamard_queries_and_phases() {
        use NodeKind::{Port, Z};
        for open in [false, true] {
            let mut links = BTreeSet::new();
            for node in 0..5 {
                for neighbor in [(node + 1) % 5, node + 5] {
                    links.insert((node.min(neighbor), node.max(neighbor), true));
                }
                let (a, b) = (node + 5, (node + 2) % 5 + 5);
                links.insert((a.min(b), a.max(b), true));
            }
            let mut kinds = vec![Z; 10];
            if open {
                kinds.extend([Port; 4]);
                for (port, node) in [2, 3, 4, 8].into_iter().enumerate() {
                    links.insert((node, 10 + port, false));
                }
            }
            let zx = graph(&kinds, &links.into_iter().collect::<Vec<_>>());
            let reduced = assert_complete_signed_relation(&zx);
            let raw = reduced
                .rows
                .iter()
                .map(|row| reduced.lift(row))
                .collect::<Vec<_>>();
            for selected in 0..1usize << raw.len() {
                let combined = raw.iter().enumerate().fold(
                    PauliString::new(zx.total_ids()),
                    |row, (index, source)| {
                        if selected & (1 << index) != 0 {
                            &row ^ source
                        } else {
                            row
                        }
                    },
                );
                assert!(zx.contains_stabilizer_support(&combined));
                let physical = zx.materialize_stabilizer(combined.clone());
                for node in 0..10 {
                    let cross = zx.neighbor_edges(node).any(|(_, edge)| {
                        let edge = zx.edge_by_id(edge);
                        let arm = combined.get(edge.id);
                        let arm = if edge.hadamard && edge.n1 > edge.n2 {
                            arm.flip()
                        } else {
                            arm
                        };
                        arm & Pauli::Z
                    });
                    let expected = Pauli::from_xz(combined.get(node) & Pauli::X, cross);
                    assert_eq!(
                        physical.paulis.get(node),
                        expected,
                        "original center {node}"
                    );
                }
            }
            assert_eq!(
                reduced.into_canonical_rows().unwrap(),
                zx.to_external_generator_table()
            );
        }
    }

    #[test]
    fn embedded_bialgebra_and_hopf_preserve_full_signed_relations() {
        use NodeKind::{X, Z};
        for decoration in [false, true] {
            let bialgebra = graph(
                &[Z, Z, X, X, Z, X, Z, X],
                &[
                    (0, 2, false),
                    (0, 3, false),
                    (1, 2, false),
                    (1, 3, false),
                    (0, 4, decoration),
                    (1, 5, false),
                    (2, 6, false),
                    (3, 7, false),
                    (4, 5, false),
                    (5, 6, false),
                    (6, 7, false),
                    (7, 4, false),
                ],
            );
            assert_complete_signed_relation(&bialgebra);
            // ZX graphs have one edge per node pair, so the Hopf pair is subdivided.
            let hopf = graph(
                &[Z, X, X, Z, Z, X],
                &[
                    (0, 2, false),
                    (2, 1, false),
                    (0, 3, false),
                    (3, 1, false),
                    (0, 4, decoration),
                    (1, 5, false),
                    (4, 5, false),
                ],
            );
            assert_complete_signed_relation(&hopf);
        }
    }

    #[test]
    fn parity_forests_preserve_mixed_frames_and_hadamard_signs() {
        use NodeKind::{Port, X, Z};
        for colors in 0..16 {
            for hadamards in 0..16 {
                let kinds = (0..4)
                    .map(|node| if colors & (1 << node) == 0 { Z } else { X })
                    .chain([Port; 4])
                    .collect::<Vec<_>>();
                let links = [(0, 1), (0, 2), (1, 3), (2, 3)]
                    .into_iter()
                    .enumerate()
                    .map(|(edge, (a, b))| (a, b, hadamards & (1 << edge) != 0))
                    .chain((0..4).map(|node| (node, node + 4, false)))
                    .collect::<Vec<_>>();
                assert_complete_signed_relation(&graph(&kinds, &links));
            }
        }
    }

    #[test]
    fn unsupported_shapes_and_resource_failures_remain_explicit() {
        use NodeKind::{Port, Z};
        let mut diagram = BooleanDecisionDiagram::default();
        let unsupported = graph(&[Port], &[]);
        assert!(
            CorrelationReduction::new(
                &unsupported,
                &mut diagram,
                ModuleCertificationLimits::DEFAULT
            )
            .unwrap()
            .is_none()
        );
        let actual = crate::RuntimeStabilizerBasis::from_zx_graph(unsupported.clone()).unwrap();
        let expected = crate::RuntimeStabilizerBasis::from_zx_graph_unreduced(unsupported).unwrap();
        assert_eq!(actual.generators(), expected.generators());
        let zx = graph(&[Z], &[]);
        let limits = ModuleCertificationLimits {
            max_local_columns: 0,
            ..ModuleCertificationLimits::DEFAULT
        };
        assert!(matches!(
            CorrelationReduction::new(&zx, &mut diagram, limits),
            Err(StabilizerError::ResourceLimited { .. })
        ));
        let limits = ModuleCertificationLimits {
            max_boolean_steps: 0,
            ..ModuleCertificationLimits::DEFAULT
        };
        let mut exhausted = BooleanDecisionDiagram::with_limits(limits.boolean_limits());
        assert!(CorrelationReduction::new(&zx, &mut exhausted, limits).is_err());
        let token = CancellationToken::new();
        token.cancel();
        assert!(matches!(
            token.scope(|| CorrelationReduction::new(
                &zx,
                &mut diagram,
                ModuleCertificationLimits::DEFAULT
            )),
            Err(StabilizerError::Cancelled(_))
        ));
    }
}
