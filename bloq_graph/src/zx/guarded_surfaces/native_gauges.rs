//! Checked fixed native gauges and their constant causal reference basis.

use super::*;
use std::cell::Cell;

type ColumnPriority = (bool, std::cmp::Reverse<i64>, usize);

#[derive(Clone, Copy, Default)]
enum CandidateSummary {
    #[default]
    Unchecked,
    Blocked,
    Ready(i64),
}

/// Complete constant RREF, including accepted equations and physical kernels.
pub(super) struct ConstantReadoutBasis {
    rows: BooleanRowSpace,
    owners: HashMap<usize, usize>,
    order: Vec<ColumnPriority>,
    pending: Vec<usize>,
    physical: usize,
    // Frontier rows stay borrowed while other summaries are populated.
    summaries: Vec<Cell<CandidateSummary>>,
    queried: Cell<bool>,
}

impl ConstantReadoutBasis {
    pub(super) fn new(
        timing: &ReadoutTiming,
        names: &[(String, usize)],
        physical: usize,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<Self, BooleanResourceError> {
        diagram.charge(
            timing
                .columns
                .len()
                .saturating_mul(2)
                .saturating_add(names.len()),
        )?;
        let mut order = timing
            .columns
            .iter()
            .map(|&(time, original)| (true, std::cmp::Reverse(time), original.0))
            .collect::<Vec<_>>();
        for &(_, column) in names {
            order[column] = (false, std::cmp::Reverse(i64::MIN), column);
        }
        let mut pending = Vec::with_capacity(physical);
        pending.extend(
            timing
                .columns
                .iter()
                .take(physical)
                .enumerate()
                .filter_map(|(column, &(time, _))| (time == i64::MAX).then_some(column)),
        );
        Ok(Self {
            rows: BooleanRowSpace::default(),
            owners: HashMap::new(),
            order,
            pending,
            physical,
            summaries: Vec::new(),
            queried: Cell::new(false),
        })
    }

    pub(super) fn insert(
        &mut self,
        mut row: BooleanRow,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(row.terms().len().saturating_mul(3).saturating_add(1))?;
        assert_eq!(row.active, ONE);
        assert!(row.terms().iter().all(|&(_, bit)| bit == ONE));
        let mut pivots = Vec::with_capacity(row.terms().len());
        pivots.extend(
            row.terms()
                .iter()
                .filter_map(|&(column, _)| self.owners.get(&column).copied()),
        );
        // Complete RREF pivots have zero on every other pivot column.
        for owner in pivots {
            row.xor_scaled(
                self.rows.row(owner).expect("pivot owner exists"),
                ONE,
                diagram,
            )?;
        }
        diagram.charge(row.terms().len())?;
        let Some(&(column, _)) = row
            .terms()
            .iter()
            .min_by_key(|&&(column, _)| self.order[column])
        else {
            return Ok(());
        };
        self.invalidate_column(column, diagram)?;
        self.rows.reduce_column(column, &row, diagram)?;
        let owner = self
            .rows
            .insert_indexed(row, diagram)?
            .expect("nonzero pivot");
        assert!(self.owners.insert(column, owner).is_none());
        diagram.charge(1)?;
        if self.summaries.len() == self.summaries.capacity() {
            diagram.charge(self.summaries.len())?;
        }
        debug_assert_eq!(owner, self.summaries.len());
        self.summaries.push(Cell::default());
        Ok(())
    }

    pub(super) fn make_known(
        &mut self,
        column: usize,
        priority: ColumnTime,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(1)?;
        let order = (true, std::cmp::Reverse(priority.0), priority.1.0);
        if order == self.order[column] {
            return Ok(());
        }
        assert!(
            order >= self.order[column],
            "column priorities only move later"
        );
        self.invalidate_column(column, diagram)?;
        self.order[column] = order;
        let Some(&owner) = self.owners.get(&column) else {
            return Ok(());
        };
        let row = self.rows.row(owner).expect("pivot owner exists");
        diagram.charge(row.terms().len())?;
        let next = row
            .terms()
            .iter()
            .min_by_key(|&&(column, _)| self.order[column])
            .expect("pivot has support")
            .0;
        if next != column {
            let row = self.rows.take(owner, diagram)?.expect("pivot owner exists");
            self.invalidate_column(next, diagram)?;
            self.rows.reduce_column(next, &row, diagram)?;
            self.rows.replace(owner, row, diagram)?;
            self.owners.remove(&column);
            assert!(self.owners.insert(next, owner).is_none());
        }
        Ok(())
    }

    pub(super) fn refresh(
        &mut self,
        timing: &ReadoutTiming,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(self.pending.len())?;
        let mut remaining = 0;
        for index in 0..self.pending.len() {
            let column = self.pending[index];
            self.make_known(column, timing.columns[column], diagram)?;
            if timing.columns[column].0 == i64::MAX {
                self.pending[remaining] = column;
                remaining += 1;
            }
        }
        self.pending.truncate(remaining);
        Ok(())
    }

    pub(super) fn candidate(
        &self,
        wanted: usize,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<Option<&BooleanRow>, BooleanResourceError> {
        diagram.charge(1)?;
        let Some(&owner) = self.owners.get(&wanted) else {
            return Ok(None);
        };
        let row = self.rows.row(owner).expect("pivot owner exists");
        Ok(Some(row))
    }

    fn invalidate_column(
        &self,
        column: usize,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(1)?;
        if !self.queried.get() {
            return Ok(());
        }
        self.rows.visit_column_rows(column, diagram, |owner| {
            self.summaries[owner].set(CandidateSummary::Unchecked);
        })
    }

    pub(super) fn ready_candidate(
        &self,
        wanted: usize,
        unknown: &[bool],
        times: &[ColumnTime],
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<Option<(&BooleanRow, i64)>, BooleanResourceError> {
        let Some(row) = self.candidate(wanted, diagram)? else {
            return Ok(None);
        };
        let summary = &self.summaries[self.owners[&wanted]];
        let time = match summary.get() {
            CandidateSummary::Blocked => return Ok(None),
            CandidateSummary::Ready(time) => time,
            CandidateSummary::Unchecked => {
                diagram.charge(row.terms().len())?;
                self.queried.set(true);
                let mut time = i64::MIN;
                // Certified rows and feedback guards are constant; selector
                // readiness is already included in the cap-X column time.
                for &(column, _) in row.terms() {
                    if column == wanted {
                        continue;
                    }
                    if column >= self.physical && unknown[column - self.physical] {
                        summary.set(CandidateSummary::Blocked);
                        return Ok(None);
                    }
                    time = time.max(times[column].0);
                }
                summary.set(CandidateSummary::Ready(time));
                time
            }
        };
        Ok(Some((row, time)))
    }
}

#[derive(Clone, Debug)]
pub(super) struct NativeGauge {
    pub(super) lambda: usize,
    pub(super) generator: [usize; 4],
}

impl NativeGauge {
    pub(super) fn row(&self, coordinates: &SurfaceCoordinates) -> BooleanRow {
        // Support values coalesce under an equality quotient; query terms XOR.
        let columns = self
            .generator
            .iter()
            .filter_map(|&bit| coordinates.bit(bit))
            .collect::<BTreeSet<_>>();
        BooleanRow::from_terms(ONE, columns.into_iter().map(|bit| (bit, ONE)))
    }
}

impl GuardedSurfaceSpace {
    pub(super) fn certify_native_gauges(
        &mut self,
    ) -> Result<Option<Vec<NativeGauge>>, BlockGraphError> {
        if self.topology.resolves.is_empty() {
            return Ok(None);
        }
        let actions = self.topology.source.actions();
        self.topology.diagram.charge(
            self.local
                .len()
                .saturating_add(self.topology.resolves.len().saturating_mul(1024))
                .saturating_add(actions.len())
                .saturating_add(self.rows.iter().fold(0usize, |work, row| {
                    work.saturating_add(row.terms().len()).saturating_add(1)
                })),
        )?;
        if self.module_sites.values().any(|&owner| owner != 0)
            || self.topology.domain != ONE
            || !self.topology.branches.is_empty()
            || self
                .local
                .values()
                .any(|variants| variants.len() != 1 || variants[0].guard != ONE)
            || self
                .rows
                .iter()
                .any(|row| row.active != ONE || row.terms().iter().any(|&(_, bit)| bit != ONE))
        {
            return Ok(None);
        }
        let mut names = HashMap::new();
        let mut named_edges = BTreeSet::new();
        let mut feedback_columns = BTreeSet::new();
        let edge_key = |a, b| if a < b { (a, b) } else { (b, a) };
        for action in actions {
            match action {
                Action::Measure {
                    name,
                    target: MeasureTarget::Edge { src, dir },
                } => {
                    let key = edge_key(src.to_array(), (src + dir.to_ivec3()).to_array());
                    names.insert(name, key);
                    named_edges.insert(key);
                }
                Action::Measure { .. } => return Ok(None),
                Action::Feedback { targets, condition } => {
                    if condition.is_some() {
                        return Ok(None);
                    }
                    for target in targets {
                        feedback_columns.extend(self.feedback_support(
                            &target,
                            &self.local[&target.target.to_array()][0],
                        )?);
                    }
                }
                _ => {}
            }
        }
        let mut gauges = Vec::new();
        let mut used = BTreeSet::new();
        for &(cap, selector) in &self.topology.resolves {
            crate::CancellationToken::checkpoint()?;
            let cap = cap.to_array();
            let view = &self.local[&cap][0];
            if view.zx.nodes()[view.node].kind != NodeKind::Selective(crate::SelectiveKind::XY) {
                return Ok(None);
            }
            let arms = view.zx.neighbor_edges(view.node).collect::<Vec<_>>();
            let [(neighbor, edge)] = arms.as_slice() else {
                return Ok(None);
            };
            let resource = view.zx.nodes()[*neighbor].pos.to_array();
            let center = &self.local[&resource][0];
            let legs = center.zx.neighbor_edges(center.node).collect::<Vec<_>>();
            if view.zx.edge_by_id(*edge).hadamard
                || center.zx.nodes()[center.node].kind != NodeKind::Z
                || legs.len() != 3
                || legs
                    .iter()
                    .any(|&(_, edge)| center.zx.edge_by_id(edge).hadamard)
            {
                return Ok(None);
            }
            let mut source = None;
            let mut data = None;
            for &(neighbor, _) in &legs {
                let node = center.zx.nodes()[neighbor];
                match node.kind {
                    NodeKind::T => source = Some(node.pos.to_array()),
                    NodeKind::X | NodeKind::Z => data = Some(node.pos.to_array()),
                    _ if node.pos.to_array() == cap => {}
                    _ => return Ok(None),
                }
            }
            let (Some(source), Some(data)) = (source, data) else {
                return Ok(None);
            };
            let source_view = &self.local[&source][0];
            if source_view.zx.neighbor_edges(source_view.node).count() != 1
                || cap[..2] != resource[..2]
                || source[..2] != resource[..2]
                || resource[2].checked_add(1) != Some(cap[2])
                || resource[2].checked_sub(1) != Some(source[2])
                || !used.insert(resource)
                || !used.insert(source)
            {
                return Ok(None);
            }
            let Some(decision) = self.topology.diagram.node(selector) else {
                return Ok(None);
            };
            let crate::GuardedVariable::Outcome(name) = &self.topology.variables[decision.variable]
            else {
                return Ok(None);
            };
            if !matches!((decision.low, decision.high), (ZERO, ONE) | (ONE, ZERO))
                || names.get(name) != Some(&edge_key(data, resource))
                || [edge_key(cap, resource), edge_key(source, resource)]
                    .iter()
                    .any(|edge| named_edges.contains(edge))
            {
                return Ok(None);
            }
            let directed = [
                (cap, resource),
                (resource, cap),
                (source, resource),
                (resource, source),
            ];
            let generator = directed
                .map(|(left, right)| 2 * self.columns[&SurfaceColumn::Edge(left, right)] + 1);
            let Some(lambda) = self.coordinates.bit(generator[0] - 1) else {
                return Ok(None);
            };
            if self.node_components(cap)[0] != [(lambda, ONE)]
                || generator
                    .iter()
                    .filter_map(|&column| self.coordinates.bit(column))
                    .any(|column| feedback_columns.contains(&column))
                || !has_local_z_pair(center, cap, source)
            {
                return Ok(None);
            }
            // The required temporal orientation makes the lambda arm finish last.
            gauges.push(NativeGauge { lambda, generator });
        }
        Ok(Some(gauges))
    }

    pub(super) fn lift_native_surfaces(&mut self) -> Result<(), BlockGraphError> {
        let Some(gauges) = &self.native_gauges else {
            return Ok(());
        };
        self.topology.diagram.charge(gauges.len())?;
        let mut by_lambda = HashMap::<usize, Vec<usize>>::new();
        for (index, gauge) in gauges.iter().enumerate() {
            by_lambda.entry(gauge.lambda).or_default().push(index);
        }
        for index in 0..self.surfaces.len() {
            crate::CancellationToken::checkpoint()?;
            let row = &self.surfaces[index].row;
            self.topology.diagram.charge(row.terms().len())?;
            let mut selected = Vec::new();
            for &(column, bit) in row.terms() {
                if let Some(gauges) = by_lambda.get(&column) {
                    assert_eq!(bit, ONE, "native reference coefficients are constant");
                    self.topology.diagram.charge(gauges.len())?;
                    selected.extend_from_slice(gauges);
                }
            }
            for gauge in selected {
                let selector = self.topology.resolves[gauge].1;
                let flip = self.topology.diagram.negate(selector)?;
                let generator = self.native_gauges.as_ref().expect("native certificate")[gauge]
                    .row(&self.coordinates);
                self.surfaces[index].row.xor_scaled(
                    &generator,
                    flip,
                    &mut self.topology.diagram,
                )?;
            }
            self.check_readout_roots(std::iter::empty())?;
        }
        // All lifted coefficients become roots before physical consumers hold ids.
        self.collect_readout_roots(std::iter::empty())?;
        self.check_budget()
    }
}

fn has_local_z_pair(center: &SurfaceTopology, cap: [i32; 3], source: [i32; 3]) -> bool {
    let zx = &center.zx;
    if zx.total_ids() > 32 {
        return false;
    }
    let bit = |column: usize, pauli: Pauli| {
        let x = u64::from(pauli & Pauli::X) << (2 * column);
        let z = u64::from(pauli & Pauli::Z) << (2 * column + 1);
        x | z
    };
    let target = zx
        .neighbor_edges(center.node)
        .filter(|&(neighbor, _)| [cap, source].contains(&zx.nodes()[neighbor].pos.to_array()))
        .fold(0, |sum, (_, edge)| sum ^ bit(edge, Pauli::Z));
    let generators = zx
        .local_stabilizer_flow_supports_for_node_kind(center.node, NodeKind::Z)
        .into_iter()
        .map(|terms| {
            terms
                .into_iter()
                .fold(0, |sum, (column, pauli)| sum ^ bit(column, pauli))
        })
        .collect::<Vec<_>>();
    (0..1usize << generators.len()).any(|choice| {
        generators.iter().enumerate().fold(0, |sum, (index, &row)| {
            if choice & (1 << index) == 0 {
                sum
            } else {
                sum ^ row
            }
        }) == target
    })
}

#[cfg(test)]
#[test]
fn raw_releases_preserve_canonical_candidates_and_private_kernels() {
    let mut diagram = bloq_utils::boolean::BooleanDecisionDiagram::default();
    let mut timing = ReadoutTiming::new(vec![9, 1, 6, 0, i64::MAX, i64::MAX], vec![], ONE, 0);
    let names = [("first".into(), 4), ("second".into(), 5)];
    let mut raw = ConstantReadoutBasis::new(&timing, &names, 4, &mut diagram).unwrap();
    let mut projected = ConstantReadoutBasis::new(&timing, &names, 4, &mut diagram).unwrap();
    let rows = [[0, 4].as_slice(), &[0, 1], &[1, 2, 5], &[2, 3]]
        .into_iter()
        .map(|columns| BooleanRow::from_terms(ONE, columns.iter().map(|&column| (column, ONE))))
        .collect();
    let constraints = [
        (AvailabilityKey::Feedback(0), vec![(1, ONE)]),
        (AvailabilityKey::Feedback(1), vec![(3, ONE)]),
    ];
    let (mut availability, available) =
        GuardedAvailability::new(rows, &constraints, 6, &[], &mut diagram).unwrap();
    for row in available.into_rows(&mut diagram).unwrap() {
        raw.insert(row.clone(), &mut diagram).unwrap();
        projected.insert(row, &mut diagram).unwrap();
    }
    let first = raw.candidate(4, &mut diagram).unwrap().unwrap().clone();
    let columns = first
        .terms()
        .iter()
        .map(|&(column, _)| column)
        .collect::<Vec<_>>();
    let mut accepted = GuardedPivots::default();
    accepted
        .insert_residual::<false>(first, columns.into_iter(), &mut diagram)
        .unwrap();
    let priority = (9, std::cmp::Reverse(4));
    raw.make_known(4, priority, &mut diagram).unwrap();
    projected.make_known(4, priority, &mut diagram).unwrap();
    timing.columns[4] = priority;
    let mut pending = constraints
        .iter()
        .map(|&(key, _)| key)
        .collect::<BTreeSet<_>>();
    let mut changed_releases = 0;
    for (cut, expected_time) in [9, 6, 1].into_iter().enumerate() {
        for column in 0..6 {
            let left = raw.candidate(column, &mut diagram).unwrap();
            let right = projected.candidate(column, &mut diagram).unwrap();
            assert_eq!(left.map(BooleanRow::terms), right.map(BooleanRow::terms));
        }
        let row = raw.candidate(5, &mut diagram).unwrap().unwrap();
        assert_eq!(timing.row(row, 5, &mut diagram).unwrap(), expected_time);
        for basis in [&raw, &projected] {
            assert_eq!(
                basis
                    .ready_candidate(5, &[false, true], &timing.columns, &mut diagram)
                    .unwrap()
                    .map(|(_, time)| time),
                Some(expected_time),
            );
        }
        if cut < constraints.len() {
            pending.remove(&constraints[cut].0);
            availability
                .release(&pending, &mut diagram, |row, diagram| {
                    raw.insert(row.clone(), diagram)?;
                    let mut row = row;
                    let before = row.terms().to_vec();
                    accepted.project(&mut row, diagram)?;
                    changed_releases += usize::from(row.terms() != before);
                    projected.insert(row, diagram)
                })
                .unwrap();
        }
    }
    assert!(changed_releases > 0);
    diagram
        .charge(diagram.limits().max_steps - diagram.steps())
        .unwrap();
    assert!(raw.insert(BooleanRow::new(ONE), &mut diagram).is_err());
}

#[cfg(test)]
#[test]
fn candidate_cache_tracks_nonpivot_readiness_and_timing() {
    let mut diagram = bloq_utils::boolean::BooleanDecisionDiagram::default();
    let names = [("a".into(), 4), ("b".into(), 5)];
    let timing = ReadoutTiming::new(vec![i64::MAX, 1, 0, 0, i64::MAX, i64::MAX], vec![], ONE, 0);
    let mut times = timing.columns.clone();
    let mut basis = ConstantReadoutBasis::new(&timing, &names, 4, &mut diagram).unwrap();
    basis
        .insert(
            BooleanRow::from_terms(ONE, [(0, ONE), (4, ONE), (5, ONE)]),
            &mut diagram,
        )
        .unwrap();
    assert!(
        basis
            .ready_candidate(4, &[true, true], &times, &mut diagram)
            .unwrap()
            .is_none()
    );
    times[5] = (2, std::cmp::Reverse(5));
    basis.make_known(5, times[5], &mut diagram).unwrap();
    for (column, time, expected) in [(None, 0, i64::MAX), (Some(0), 3, 3)] {
        if let Some(column) = column {
            times[column] = (time, std::cmp::Reverse(column));
            basis
                .make_known(column, times[column], &mut diagram)
                .unwrap();
        }
        let (row, actual) = basis
            .ready_candidate(4, &[true, false], &times, &mut diagram)
            .unwrap()
            .unwrap();
        assert_eq!(row.terms(), [(0, ONE), (4, ONE), (5, ONE)]);
        assert_eq!(actual, expected);
    }
    basis
        .insert(
            BooleanRow::from_terms(ONE, [(0, ONE), (1, ONE)]),
            &mut diagram,
        )
        .unwrap();
    assert_eq!(
        basis
            .ready_candidate(4, &[true, false], &times, &mut diagram)
            .unwrap()
            .unwrap()
            .1,
        2
    );
    // Repeated unchanged queries fit a budget that cannot rescan their support.
    diagram
        .charge(diagram.limits().max_steps - diagram.steps() - 16)
        .unwrap();
    for _ in 0..8 {
        assert_eq!(
            basis
                .ready_candidate(4, &[true, false], &times, &mut diagram)
                .unwrap()
                .unwrap()
                .1,
            2
        );
    }
    diagram
        .charge(diagram.limits().max_steps - diagram.steps())
        .unwrap();
    let error = basis
        .ready_candidate(4, &[true, false], &times, &mut diagram)
        .unwrap_err();
    assert_eq!(error.resource, "Boolean work steps");
}

#[cfg(test)]
#[test]
fn constant_rref_updates_match_fresh_reduction() {
    fn check(
        basis: &ConstantReadoutBasis,
        source: &[BooleanRow],
        timing: &mut ReadoutTiming,
        unknown: &[bool],
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) {
        let mut columns = (0..basis.order.len()).collect::<Vec<_>>();
        columns.sort_unstable_by_key(|&column| basis.order[column]);
        let expected = reduce_boolean_rows(source.to_vec(), columns, &mut Default::default())
            .unwrap()
            .into_iter()
            .map(BooleanRow::into_terms)
            .collect::<Vec<_>>();
        let mut pivots = basis.owners.iter().collect::<Vec<_>>();
        pivots.sort_unstable_by_key(|&(&column, _)| basis.order[column]);
        let actual = pivots
            .into_iter()
            .map(|(_, &owner)| basis.rows.row(owner).unwrap().terms().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        for wanted in (6..9).filter(|&column| unknown[column - 6]) {
            let expected = basis
                .candidate(wanted, diagram)
                .unwrap()
                .filter(|row| {
                    !GuardedSurfaceSpace::named_terms(row, 6, 9)
                        .iter()
                        .any(|&(column, _)| column != wanted && unknown[column - 6])
                })
                .map(|row| timing.row(row, wanted, diagram).unwrap());
            let actual = basis
                .ready_candidate(wanted, unknown, &timing.columns, diagram)
                .unwrap()
                .map(|(_, time)| time);
            assert_eq!(actual, expected);
        }
    }
    for seed in 0..32u64 {
        let mut diagram = bloq_utils::boolean::BooleanDecisionDiagram::default();
        let mut timing = ReadoutTiming::new(vec![i64::MAX; 9], Vec::new(), ONE, 0);
        let names = [("a".into(), 6), ("b".into(), 7), ("c".into(), 8)];
        let mut basis = ConstantReadoutBasis::new(&timing, &names, 6, &mut diagram).unwrap();
        let mut source = Vec::new();
        let mut unknown = [true; 3];
        for (step, column) in [6, 0, 7, 1, 8, 2].into_iter().enumerate() {
            let mask =
                ((seed + 1) * 0x9e37).rotate_left((step * 7) as u32) ^ (0x51a3 * (step as u64 + 1));
            for bits in [mask, mask.rotate_right(3)] {
                let row = BooleanRow::from_terms(
                    ONE,
                    (0..9)
                        .filter_map(|column| (bits & (1 << column) != 0).then_some((column, ONE))),
                );
                basis.insert(row.clone(), &mut diagram).unwrap();
                source.push(row);
                check(&basis, &source, &mut timing, &unknown, &mut diagram);
            }
            let priority = (step as i64, std::cmp::Reverse(column));
            basis.make_known(column, priority, &mut diagram).unwrap();
            timing.columns[column] = priority;
            if column >= 6 {
                unknown[column - 6] = false;
            }
            check(&basis, &source, &mut timing, &unknown, &mut diagram);
        }
    }
}

#[cfg(test)]
#[test]
fn shared_lambda_lifts_every_native_gauge() {
    let source = crate::lower_blog_graph_ast_deferred(
        &crate::parse_blog_program_to_ast(
            r#"BLOG 1.0
module main {
    in q0: data = 0
    out q1: data = 5
    0: Port [0,0,0] role=input
    1: XZX [0,0,1]
    2: XZX [0,0,2]
    3: XZX [0,0,3]
    4: XZX [0,0,4]
    5: Port [0,0,5] role=output
    6: T [1,0,0]
    7: XZX [1,0,1]
    8: XY [1,0,2]
    9: T [1,0,3]
    10: XZX [1,0,4]
    11: XY [1,0,5]
    0 -> +Z
    1 -> +Z
    2 -> +Z
    3 -> +Z
    4 -> +Z
    1 -> +X
    4 -> +X
    6 -> +Z
    7 -> +Z
    9 -> +Z
    10 -> +Z
    t0 = measure 1 -> +X
    resolve 8 if !t0
    t1 = measure 4 -> +X
    resolve 11 if !t1
}"#,
        )
        .unwrap(),
    )
    .unwrap()
    .flatten()
    .unwrap()
    .fix_shadowed_faces();
    source.validate_source().unwrap();
    let limits = ModuleCertificationLimits::DEFAULT;
    let topology = GuardedTopology::new(&source, limits).unwrap();
    let mut actual = GuardedSurfaceSpace::new(topology.clone(), &HashMap::new(), limits).unwrap();
    let gauges = actual.certify_native_gauges().unwrap().unwrap();
    assert_eq!(gauges.len(), 2);
    assert_eq!(gauges[0].lambda, gauges[1].lambda);
    let expected =
        GuardedSurfaceSpace::build::<false>(topology, &HashMap::new(), limits, &HashMap::new())
            .unwrap();
    let mut actual = actual.plan_readouts().unwrap();
    let expected = expected.plan_readouts().unwrap();
    let scalar = super::super::super::RuntimeStabilizerBasis::for_output_frames(
        ZXGraph::from_block_graph_for_analysis(&source).unwrap(),
    )
    .unwrap();
    assert_eq!(actual.topology.variables.len(), 2);
    for values in [[false, false], [false, true], [true, false], [true, true]] {
        let fills = actual
            .topology
            .resolves
            .iter()
            .map(|&(cap, selector)| {
                let chosen = if actual
                    .topology
                    .diagram
                    .evaluate(selector, |bit| values[bit])
                {
                    PauliBasis::X
                } else {
                    PauliBasis::Y
                };
                (cap, chosen)
            })
            .collect::<Vec<_>>();
        let filled = scalar.apply_selective_fills(&fills).unwrap();
        assert_eq!(
            actual.materialize_surfaces(filled.zx_graph(), &values),
            expected.materialize_surfaces(filled.zx_graph(), &values),
            "both physical gauges and their signed branches must survive {values:?}"
        );
        actual.collect_garbage(std::iter::empty());
    }
}
