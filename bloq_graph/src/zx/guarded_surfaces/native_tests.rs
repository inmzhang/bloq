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
        let gauges = self.native_gauges.take().expect("native gauge certificate");
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
                assert_eq!(self.coordinates.get(row, gauge.generator[0]), ZERO);
            }
            if lambda == ONE {
                let flip = self.topology.diagram.negate(selector)?;
                lifted.xor_scaled(
                    &gauge.row(&self.coordinates),
                    flip,
                    &mut self.topology.diagram,
                )?;
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
    tests::assert_same_raw_plan(left, right);
}

fn single_t_sources() -> [String; 2] {
    let source = include_str!("fixtures/native_t.blog");
    [
        source.into(),
        source.replace("resolve 5 if !t0", "resolve 5 if t0"),
    ]
}

#[test]
fn native_reference_branches_preserve_spaces_and_avoid_self_wait() {
    let [t, tdg] = single_t_sources();
    let feedback = t.replace("resolve 5 if !t0", "resolve 5 if !t0\n  feedback X 1 -> +Z");
    for text in [
        t.as_str(),
        tdg.as_str(),
        include_str!("fixtures/native_mixed_t.blog"),
        feedback.as_str(),
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
            gauges.iter().all(|gauge| {
                row.get(gauge.lambda) == actual.coordinates.get(row, gauge.generator[2] - 1)
            })
        }));
        let mut actual_rows = actual.rows.clone();
        actual.fill_selective(&mut actual_rows, &resolved).unwrap();
        reference.audit_reference = true;
        let mut incremental = reference.clone();
        let mut fallback = reference.clone();
        reference.fresh_reference = true;
        incremental.plan_causal_readouts::<true, true>().unwrap();
        if text == feedback {
            assert!(
                incremental
                    .surfaces
                    .iter()
                    .any(|surface| !surface.feedbacks.is_empty())
            );
        }
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
                        for &(column, _) in gauge.row(&actual.coordinates).terms() {
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
        for &(column, _) in gauge.row(&reference.coordinates).terms() {
            timing.columns[column].0 = i64::MAX;
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
    for text in single_t_sources() {
        let mut space = make_space(&text);
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
fn native_certificate_rejects_symbolic_relations() {
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
fn native_certificate_rejects_node_measurements() {
    let source = include_str!("fixtures/native_t.blog").replace(
        "resolve 5 if !t0",
        "crossing = measure 1\n  resolve 5 if !t0",
    );
    let mut space = make_space(&source);
    assert!(space.certify_native_gauges().unwrap().is_none());
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
