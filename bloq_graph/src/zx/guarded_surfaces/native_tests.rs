//! Unsigned differential checks for the fixed native gauge planner.

use super::*;

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
        tests::assert_same_raw_plan(&incremental, &reference);
        fallback
            .plan_causal_readouts_with_fast_path::<true, true, true>(0)
            .unwrap();
        tests::assert_same_raw_plan(&fallback, &reference);
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
