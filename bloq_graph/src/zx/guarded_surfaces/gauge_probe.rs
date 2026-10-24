//! Unsigned differential checks for the fixed native gauge planner.

use super::*;

impl GuardedSurfaceSpace {
    pub(super) fn audit_reference_candidate(
        &mut self,
        row: &BooleanRow,
        own: usize,
        reported: i64,
        names: &[(String, usize)],
        known: &HashMap<String, i64>,
    ) -> Result<(), BlockGraphError> {
        let gauges = self.native_gauges.take().expect("reference experiment");
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
        self.native_gauges = Some(gauges);
        Ok(())
    }
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
    assert_eq!(left.reference_frontiers, right.reference_frontiers);
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
        include_str!("fixtures/native_mixed_t.blog"),
    ] {
        let mut actual = make_space(text);
        let gauges = actual
            .certify_native_gauges()
            .unwrap()
            .expect("native gauge certificate");
        let resolved = actual
            .topology
            .resolves
            .iter()
            .map(|(pos, _)| pos.to_array())
            .collect();
        let mut reference = actual.clone();
        reference.native_gauges = Some(gauges.clone());
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
        assert!(reference_rows.iter().all(|row| {
            gauges
                .iter()
                .all(|gauge| row.get(gauge.lambda) == row.get(gauge.generator[2] - 1))
        }));
        let mut actual_rows = actual.rows.clone();
        actual.fill_selective(&mut actual_rows, &resolved).unwrap();
        reference.audit_reference = true;
        let mut incremental = reference.clone();
        let mut fallback = reference.clone();
        fallback.incremental_reference = true;
        incremental.incremental_reference = true;
        incremental.plan_causal_readouts::<true, true>().unwrap();
        reference.plan_causal_readouts::<true, true>().unwrap();
        assert_same_reference_plan(&incremental, &reference);
        fallback
            .plan_causal_readouts_with_fast_path::<true, true, true>(0)
            .unwrap();
        assert_same_reference_plan(&fallback, &reference);
        let physical = 2 * actual.columns.len();
        for assignment in 0..1usize << actual.topology.variables.len() {
            let evaluate = |root| {
                actual
                    .topology
                    .diagram
                    .evaluate(root, |variable| assignment & (1 << variable) != 0)
            };
            let evaluated = actual_rows
                .iter()
                .filter(|row| evaluate(row.active))
                .map(|row| {
                    BooleanRow::from_terms(
                        ONE,
                        row.terms()
                            .iter()
                            .filter_map(|&(column, bit)| evaluate(bit).then_some((column, ONE))),
                    )
                })
                .collect::<Vec<_>>();
            let lift = |row: &BooleanRow| {
                let mut lifted = BooleanRow::from_terms(
                    row.active,
                    row.terms()
                        .iter()
                        .copied()
                        .filter(|&(column, _)| column < physical),
                );
                for (gauge, &(_, selector)) in gauges.iter().zip(&actual.topology.resolves) {
                    if !evaluate(selector) && row.get(gauge.lambda) == ONE {
                        for &column in &gauge.generator {
                            lifted.set(column, if lifted.get(column) == ONE { ZERO } else { ONE });
                        }
                    }
                }
                lifted
            };
            let expected = canonical(evaluated.clone(), physical);
            assert_eq!(
                canonical(reference_rows.iter().map(lift).collect(), physical),
                expected
            );
            let mut with_plan = evaluated;
            with_plan.extend(reference.surfaces.iter().map(|surface| lift(&surface.row)));
            assert_eq!(canonical(with_plan, physical), expected);
        }
        let [gauge] = gauges.as_slice() else {
            continue;
        };
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

#[cfg(feature = "verify")]
#[test]
fn mixed_native_fixture_matches_qasm() {
    let space = make_space(include_str!("fixtures/native_mixed_t.blog"));
    let circuit = quizx::circuit::Circuit::from_qasm(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2];
         h q[0]; t q[0]; cx q[0],q[1]; tdg q[1]; h q[0]; t q[0];",
    )
    .unwrap();
    crate::verify::LogicalVerifier::new(&space.topology.source)
        .unwrap()
        .verify(&circuit.to_graph(), 16, 0)
        .unwrap();
}

#[test]
fn native_plan_materializes_y_on_the_flipped_t_boundary() {
    for text in [
        include_str!("fixtures/native_t.blog"),
        include_str!("fixtures/native_tdg.blog"),
    ] {
        let mut space = make_space(text);
        assert!(space.certify_native_gauges().unwrap().is_some());
        let zx = ZXGraph::from_block_graph_for_analysis(&space.topology.source).unwrap();
        let resource = zx
            .nodes()
            .iter()
            .find(|node| node.kind == NodeKind::T)
            .unwrap()
            .pos;
        let scalar = super::super::super::RuntimeStabilizerBasis::for_output_frames(zx).unwrap();
        let mut plan = space.plan_readouts().unwrap();
        for outcome in [false, true] {
            let (cap, selector) = plan.topology.resolves[0];
            let flipped = !plan.topology.evaluate_source(selector, |_| outcome);
            let filled = scalar
                .apply_selective_fill(
                    cap,
                    if flipped {
                        crate::PauliBasis::Y
                    } else {
                        crate::PauliBasis::X
                    },
                )
                .unwrap();
            let rows = plan.materialize_surfaces(filled.zx_graph(), &[outcome]);
            let has_y = rows.iter().any(|row| {
                row.stabilizer
                    .interior_edges
                    .iter()
                    .any(|(&(left, right), &pauli)| {
                        (left == resource || right == resource) && pauli == Pauli::Y
                    })
            });
            assert_eq!(
                has_y, flipped,
                "the signed T-boundary path must be exercised"
            );
            plan.collect_garbage(std::iter::empty());
        }
    }
}

#[test]
fn native_reference_census_rejects_symbolic_relations() {
    let mut space = make_space(include_str!("fixtures/native_t.blog"));
    assert_eq!(
        space
            .certify_native_gauges()
            .unwrap()
            .expect("native gauge certificate")
            .len(),
        1
    );
    let column = space.rows[0].terms()[0].0;
    space.rows[0].set(column, space.topology.resolves[0].1);
    assert!(space.certify_native_gauges().unwrap().is_none());
    assert!(space.native_gauges.is_none());
}

#[test]
fn native_certificate_preserves_work_budget_failures() {
    let mut space = make_space(include_str!("fixtures/native_t.blog"));
    let remaining = space.topology.diagram.limits().max_steps - space.topology.diagram.steps();
    space.topology.diagram.charge(remaining).unwrap();
    assert!(matches!(
        space.certify_native_gauges(),
        Err(BlockGraphError::Stabilizer(
            StabilizerError::ResourceLimited {
                phase: "Boolean work steps",
                ..
            }
        ))
    ));
}

#[test]
#[ignore = "external frozen BLOG, planning only; requires BLOQ_GAUGE_BLOG"]
fn frozen_reference_probe() {
    let text = std::fs::read_to_string(std::env::var_os("BLOQ_GAUGE_BLOG").expect("frozen input"))
        .unwrap();
    let started = std::time::Instant::now();
    let mut space = make_space(&text);
    let gauges = space
        .certify_native_gauges()
        .unwrap()
        .expect("all caps certified");
    println!(
        "GAUGE_PROBE census caps={} rows={} nodes={} seconds={:.6}",
        gauges.len(),
        space.rows.len(),
        space.topology.diagram.nodes().len(),
        started.elapsed().as_secs_f64()
    );
    space.native_gauges = Some(gauges);
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
