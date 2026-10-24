//! Non-default unsigned native-gadget experiment. Never emits an IR program.

use super::*;

type ColumnPriority = (bool, std::cmp::Reverse<i64>, usize);

/// Complete constant RREF, including accepted equations and physical kernels.
pub(super) struct ConstantReadoutBasis {
    rows: BooleanRowSpace,
    owners: HashMap<usize, usize>,
    order: Vec<ColumnPriority>,
    pending: Vec<usize>,
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
            .enumerate()
            .map(|(column, &time)| (true, std::cmp::Reverse(time), column))
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
                .filter_map(|(column, &time)| (time == i64::MAX).then_some(column)),
        );
        Ok(Self {
            rows: BooleanRowSpace::default(),
            owners: HashMap::new(),
            order,
            pending,
        })
    }

    pub(super) fn insert(
        &mut self,
        source: &BooleanRow,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(source.terms().len().saturating_mul(3).saturating_add(1))?;
        assert_eq!(source.active, ONE);
        assert!(source.terms().iter().all(|&(_, bit)| bit == ONE));
        let mut row = source.clone();
        let mut pivots = Vec::with_capacity(source.terms().len());
        pivots.extend(
            source
                .terms()
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
        self.rows.reduce_column(column, &row, diagram)?;
        let owner = self
            .rows
            .insert_indexed(row, diagram)?
            .expect("nonzero pivot");
        assert!(self.owners.insert(column, owner).is_none());
        Ok(())
    }

    pub(super) fn make_known(
        &mut self,
        column: usize,
        time: i64,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(1)?;
        let order = (true, std::cmp::Reverse(time), column);
        assert!(
            order >= self.order[column],
            "column priorities only move later"
        );
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
            if timing.columns[column] == i64::MAX {
                self.pending[remaining] = column;
                remaining += 1;
            } else {
                self.make_known(column, timing.columns[column], diagram)?;
            }
        }
        self.pending.truncate(remaining);
        Ok(())
    }

    pub(super) fn candidate(
        &self,
        wanted: usize,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<Option<BooleanRow>, BooleanResourceError> {
        diagram.charge(1)?;
        let Some(&owner) = self.owners.get(&wanted) else {
            return Ok(None);
        };
        let row = self.rows.row(owner).expect("pivot owner exists");
        diagram.charge(row.terms().len())?;
        Ok(Some(row.clone()))
    }
}

#[derive(Clone, Debug)]
pub(super) struct NativeGauge {
    pub(super) lambda: usize,
    generator: [usize; 4],
}

impl GuardedSurfaceSpace {
    pub(super) fn audit_reference_candidate(
        &mut self,
        row: &BooleanRow,
        own: usize,
        reported: i64,
        names: &[(String, usize)],
        known: &HashMap<String, i64>,
    ) -> Result<(), BlockGraphError> {
        let gauges = self.reference_gauges.take().expect("reference experiment");
        let actions = self.topology.source.actions();
        let mut uncharged = self.readout_timing(names, known, &actions)?;
        let mut available = self
            .topology
            .availability(&known.keys().cloned().collect())?;
        let mut lifted = row.clone();
        for (gauge, &(_, selector)) in gauges.iter().zip(&self.topology.resolves) {
            let lambda = row.get(gauge.lambda);
            assert!(lambda == ZERO || lambda == ONE);
            if !available.available(selector, &mut self.topology.diagram)? {
                assert_eq!(lambda, ZERO, "pending cap must be identity");
                assert_eq!(row.get(gauge.lambda + 1), ZERO);
            }
            if lambda == ONE {
                let flip = self.topology.diagram.negate(selector)?;
                let generator =
                    BooleanRow::from_terms(ONE, gauge.generator.map(|column| (column, ONE)));
                lifted.xor_scaled(&generator, flip, &mut self.topology.diagram)?;
            }
        }
        for &(_, bit) in lifted.terms() {
            assert!(available.available(bit, &mut self.topology.diagram)?);
        }
        assert_eq!(
            reported,
            uncharged.row(&lifted, own, &mut self.topology.diagram)?,
            "lift cost at the actual candidate wave"
        );
        self.reference_gauges = Some(gauges);
        Ok(())
    }

    fn certify_native_gauges(&mut self) -> Result<Vec<NativeGauge>, String> {
        self.topology
            .diagram
            .charge(
                self.local
                    .len()
                    .saturating_add(self.topology.resolves.len().saturating_mul(1024)),
            )
            .map_err(|error| error.to_string())?;
        if self.topology.domain != ONE
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
            return Err("probe requires a fixed, constant open relation".into());
        }
        let mut names = HashMap::new();
        let mut named_edges = BTreeSet::new();
        let mut feedback_columns = BTreeSet::new();
        let edge_key = |a, b| if a < b { (a, b) } else { (b, a) };
        for action in self.topology.source.actions() {
            match action {
                Action::Measure {
                    name,
                    target: MeasureTarget::Edge { src, dir },
                } => {
                    let key = edge_key(src.to_array(), (src + dir.to_ivec3()).to_array());
                    names.insert(name, key);
                    named_edges.insert(key);
                }
                Action::Measure { .. } => return Err("probe requires edge-named records".into()),
                Action::Feedback { targets, condition } => {
                    if condition.is_some() {
                        return Err("probe requires unconditional feedback".into());
                    }
                    for target in targets {
                        feedback_columns.extend(
                            self.feedback_support(
                                &target,
                                &self.local[&target.target.to_array()][0],
                            )
                            .map_err(|error| error.to_string())?,
                        );
                    }
                }
                _ => {}
            }
        }
        let mut gauges = Vec::new();
        let mut used = BTreeSet::new();
        for &(cap, selector) in &self.topology.resolves {
            let unsupported = || format!("unsupported native gauge at {cap}");
            let cap = cap.to_array();
            let view = &self.local[&cap][0];
            if view.zx.nodes()[view.node].kind != NodeKind::Selective(crate::SelectiveKind::XY) {
                return Err(unsupported());
            }
            let arms = view.zx.neighbor_edges(view.node).collect::<Vec<_>>();
            let [(neighbor, edge)] = arms.as_slice() else {
                return Err(unsupported());
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
                return Err(unsupported());
            }
            let mut source = None;
            let mut data = None;
            for &(neighbor, _) in &legs {
                let node = center.zx.nodes()[neighbor];
                match node.kind {
                    NodeKind::T => source = Some(node.pos.to_array()),
                    NodeKind::X | NodeKind::Z => data = Some(node.pos.to_array()),
                    _ if node.pos.to_array() == cap => {}
                    _ => return Err(unsupported()),
                }
            }
            let (Some(source), Some(data)) = (source, data) else {
                return Err(unsupported());
            };
            let source_view = &self.local[&source][0];
            if source_view.zx.neighbor_edges(source_view.node).count() != 1
                || IVec3::from_array(cap) != IVec3::from_array(resource) + IVec3::Z
                || IVec3::from_array(source) != IVec3::from_array(resource) - IVec3::Z
                || !used.insert(resource)
                || !used.insert(source)
            {
                return Err(unsupported());
            }
            let decision = self
                .topology
                .diagram
                .node(selector)
                .ok_or_else(unsupported)?;
            let crate::GuardedVariable::Outcome(name) = &self.topology.variables[decision.variable]
            else {
                return Err(unsupported());
            };
            if !matches!((decision.low, decision.high), (ZERO, ONE) | (ONE, ZERO))
                || names.get(name) != Some(&edge_key(data, resource))
                || [edge_key(cap, resource), edge_key(source, resource)]
                    .iter()
                    .any(|edge| named_edges.contains(edge))
            {
                return Err(unsupported());
            }
            let directed = [
                (cap, resource),
                (resource, cap),
                (source, resource),
                (resource, source),
            ];
            let generator = directed
                .map(|(left, right)| 2 * self.columns[&SurfaceColumn::Edge(left, right)] + 1);
            let lambda = generator[0] - 1;
            if self.node_components(cap)[0] != [(lambda, ONE)]
                || generator
                    .iter()
                    .any(|column| feedback_columns.contains(column))
                || !has_local_z_pair(center, cap, source)
            {
                return Err(unsupported());
            }
            // The required temporal orientation makes the lambda arm finish last.
            gauges.push(NativeGauge { lambda, generator });
        }
        Ok(gauges)
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

fn make_space(text: &str) -> GuardedSurfaceSpace {
    let source =
        crate::lower_blog_graph_ast_deferred(&crate::parse_blog_program_to_ast(text).unwrap())
            .unwrap()
            .flatten()
            .unwrap()
            .with_zero_min_z()
            .unwrap()
            .fix_shadowed_faces();
    source.validate_source().unwrap();
    let limits = ModuleCertificationLimits::DEFAULT;
    GuardedSurfaceSpace::new(
        GuardedTopology::new(&source, limits).unwrap(),
        &HashMap::new(),
        limits,
    )
    .unwrap()
}

fn canonical(rows: Vec<BooleanRow>, columns: usize) -> Vec<Vec<(usize, DecisionId)>> {
    reduce_boolean_rows(rows, 0..columns, &mut Default::default())
        .unwrap()
        .into_iter()
        .map(BooleanRow::into_terms)
        .collect()
}

fn assert_same_reference_plan(left: &GuardedSurfaceSpace, right: &GuardedSurfaceSpace) {
    assert_eq!(left.surfaces.len(), right.surfaces.len());
    for (left, right) in left.surfaces.iter().zip(&right.surfaces) {
        assert_eq!(left.kind, right.kind);
        assert_eq!(left.feedbacks, right.feedbacks);
        assert_eq!(left.row.active, right.row.active);
        assert_eq!(left.row.terms(), right.row.terms());
    }
}

#[test]
fn native_reference_branches_preserve_spaces_and_avoid_self_wait() {
    for text in [
        include_str!("fixtures/native_t.blog"),
        include_str!("fixtures/native_tdg.blog"),
    ] {
        let mut actual = make_space(text);
        let gauges = actual.certify_native_gauges().unwrap();
        let [gauge] = gauges.as_slice() else {
            panic!("one T gadget");
        };
        let resolved = actual
            .topology
            .resolves
            .iter()
            .map(|(pos, _)| pos.to_array())
            .collect();
        let mut reference = actual.clone();
        reference.reference_gauges = Some(gauges.clone());
        let mut reference_rows = reference.rows.clone();
        reference
            .fill_selective(&mut reference_rows, &resolved)
            .unwrap();
        assert!(
            reference_rows
                .iter()
                .all(|row| row.terms().iter().all(|&(_, bit)| bit == ONE))
        );
        // Equality of these linear functionals proves the protected-zero kernel is fixed.
        assert!(
            reference_rows
                .iter()
                .all(|row| row.get(gauge.lambda) == row.get(gauge.generator[2] - 1))
        );
        let mut actual_rows = actual.rows.clone();
        actual.fill_selective(&mut actual_rows, &resolved).unwrap();
        for outcome in [false, true] {
            let flip = !actual
                .topology
                .diagram
                .evaluate(actual.topology.resolves[0].1, |_| outcome);
            let evaluated = actual_rows
                .iter()
                .filter(|row| actual.topology.diagram.evaluate(row.active, |_| outcome))
                .map(|row| {
                    BooleanRow::from_terms(
                        ONE,
                        row.terms().iter().filter_map(|&(column, bit)| {
                            actual
                                .topology
                                .diagram
                                .evaluate(bit, |_| outcome)
                                .then_some((column, ONE))
                        }),
                    )
                })
                .collect();
            let mut lifted = reference_rows.clone();
            for row in &mut lifted {
                if flip && row.get(gauge.lambda) == ONE {
                    for &column in &gauge.generator {
                        row.set(column, if row.get(column) == ONE { ZERO } else { ONE });
                    }
                }
            }
            assert_eq!(
                canonical(lifted, 2 * actual.columns.len()),
                canonical(evaluated, 2 * actual.columns.len())
            );
        }
        reference.audit_reference = true;
        let mut incremental = reference.clone();
        incremental.incremental_reference = true;
        incremental.plan_causal_readouts::<true, true>().unwrap();
        reference.plan_causal_readouts::<true, true>().unwrap();
        assert_same_reference_plan(&incremental, &reference);
        let merge = reference
            .surfaces
            .iter()
            .find(|surface| {
                matches!(&surface.kind,
            GuardedSurfaceKind::Readout { name, .. } if name == "t0")
            })
            .unwrap()
            .row
            .clone();
        assert_eq!(merge.get(gauge.lambda), ZERO);
        let actions = reference.topology.source.actions();
        let own = 2 * reference.columns.len();
        let mut timing = reference
            .readout_timing(&[("t0".into(), own)], &HashMap::new(), &actions)
            .unwrap();
        let merge_time = timing
            .row(&merge, own, &mut reference.topology.diagram)
            .unwrap();
        assert!(merge_time < i64::MAX);
        for &column in &gauge.generator {
            timing.columns[column] = i64::MAX;
        }
        assert_eq!(
            timing
                .row(&merge, own, &mut reference.topology.diagram)
                .unwrap(),
            i64::MAX
        );

        let lifted = reference
            .surfaces
            .iter()
            .find(|surface| surface.row.get(gauge.lambda) == ONE)
            .expect("a terminal witness uses the selective cap")
            .row
            .clone();
        let names = [("t0".into(), own)];
        let mut pending = reference
            .readout_timing(&names, &HashMap::new(), &actions)
            .unwrap();
        assert_eq!(
            pending
                .row(&lifted, usize::MAX, &mut reference.topology.diagram)
                .unwrap(),
            i64::MAX
        );
        let known = HashMap::from([("t0".into(), merge_time)]);
        let mut ready = reference.readout_timing(&names, &known, &actions).unwrap();
        let time = ready
            .row(&lifted, usize::MAX, &mut reference.topology.diagram)
            .unwrap();
        assert!((merge_time..i64::MAX).contains(&time));
        reference
            .audit_reference_candidate(&lifted, usize::MAX, time, &names, &known)
            .unwrap();
    }
}

#[test]
fn incremental_reference_preserves_mixed_native_plan() {
    let mut reference = make_space(include_str!("fixtures/native_mixed_t.blog"));
    reference.reference_gauges = Some(reference.certify_native_gauges().unwrap());
    reference.audit_reference = true;
    let mut incremental = reference.clone();
    incremental.incremental_reference = true;
    incremental.plan_causal_readouts::<true, true>().unwrap();
    reference.plan_causal_readouts::<true, true>().unwrap();
    assert_same_reference_plan(&incremental, &reference);
    #[cfg(feature = "verify")]
    {
        let circuit = quizx::circuit::Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2];
             h q[0]; t q[0]; cx q[0],q[1]; tdg q[1]; h q[0]; t q[0];",
        )
        .unwrap();
        crate::verify::LogicalVerifier::new(&reference.topology.source)
            .unwrap()
            .verify(&circuit.to_graph(), 16, 0)
            .unwrap();
    }
}

#[test]
fn constant_rref_updates_match_fresh_reduction() {
    fn check(basis: &ConstantReadoutBasis, source: &[BooleanRow]) {
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
    }
    for seed in 0..32u64 {
        let mut diagram = bloq_utils::boolean::BooleanDecisionDiagram::default();
        let timing = ReadoutTiming::new(vec![i64::MAX; 9], Vec::new(), ONE, 0);
        let names = [("a".into(), 6), ("b".into(), 7), ("c".into(), 8)];
        let mut basis = ConstantReadoutBasis::new(&timing, &names, 6, &mut diagram).unwrap();
        let mut source = Vec::new();
        for (step, column) in [6, 0, 7, 1, 8, 2].into_iter().enumerate() {
            let mask =
                ((seed + 1) * 0x9e37).rotate_left((step * 7) as u32) ^ (0x51a3 * (step as u64 + 1));
            for bits in [mask, mask.rotate_right(3)] {
                let row = BooleanRow::from_terms(
                    ONE,
                    (0..9)
                        .filter_map(|column| (bits & (1 << column) != 0).then_some((column, ONE))),
                );
                basis.insert(&row, &mut diagram).unwrap();
                source.push(row);
                check(&basis, &source);
            }
            basis.make_known(column, step as i64, &mut diagram).unwrap();
            check(&basis, &source);
        }
    }
}

#[test]
fn native_reference_census_rejects_symbolic_relations() {
    let mut space = make_space(include_str!("fixtures/native_t.blog"));
    assert_eq!(space.certify_native_gauges().unwrap().len(), 1);
    let column = space.rows[0].terms()[0].0;
    space.rows[0].set(column, space.topology.resolves[0].1);
    space.certify_native_gauges().unwrap_err();
    assert!(space.reference_gauges.is_none());
}

#[test]
#[ignore = "external frozen BLOG, planning only; requires BLOQ_GAUGE_BLOG"]
fn frozen_reference_probe() {
    let text = std::fs::read_to_string(std::env::var_os("BLOQ_GAUGE_BLOG").expect("frozen input"))
        .unwrap();
    let started = std::time::Instant::now();
    let mut space = make_space(&text);
    let gauges = space.certify_native_gauges().expect("all caps certified");
    println!(
        "GAUGE_PROBE census caps={} rows={} nodes={} seconds={:.6}",
        gauges.len(),
        space.rows.len(),
        space.topology.diagram.nodes().len(),
        started.elapsed().as_secs_f64()
    );
    space.reference_gauges = Some(gauges);
    space.incremental_reference = std::env::var_os("BLOQ_GAUGE_INCREMENTAL").is_some();
    let comparison = std::env::var_os("BLOQ_GAUGE_COMPARE").map(|_| {
        let mut reference = space.clone();
        reference.incremental_reference = false;
        reference
    });
    let planning = std::time::Instant::now();
    space.plan_causal_readouts::<true, true>().unwrap();
    assert!(
        space.surfaces.iter().all(|surface| surface.row.active.0 < 2
            && surface.row.terms().iter().all(|&(_, bit)| bit.0 < 2))
    );
    println!(
        "GAUGE_PROBE reference surfaces={} peak_nodes={} nodes={} steps={} seconds={:.6} compilation_verified=false",
        space.surfaces.len(),
        space.topology.diagram.peak_nodes(),
        space.topology.diagram.nodes().len(),
        space.topology.diagram.steps(),
        planning.elapsed().as_secs_f64()
    );
    if let Some(mut reference) = comparison {
        reference.plan_causal_readouts::<true, true>().unwrap();
        assert_same_reference_plan(&space, &reference);
        println!(
            "GAUGE_PROBE differential exact_raw=true surfaces={}",
            space.surfaces.len()
        );
    }
}
