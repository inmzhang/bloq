use super::*;

fn empty_space() -> GuardedSurfaceSpace {
    let limits = ModuleCertificationLimits::DEFAULT;
    GuardedSurfaceSpace::new(
        GuardedTopology::new(&crate::BlockGraph::new(), limits).unwrap(),
        &HashMap::new(),
        limits,
    )
    .unwrap()
}

fn named_row_from_ids(
    space: &mut GuardedSurfaceSpace,
    rows: &BooleanRowSpace,
    ids: &[usize],
    column: usize,
) -> BooleanRow {
    let rows = BooleanRowSpace::new(
        ids.iter()
            .map(|&id| rows.row(id).unwrap().clone())
            .collect(),
        &mut space.topology.diagram,
    )
    .unwrap();
    let reduced = space
        .reduce_named_rows::<true>(rows, [column].into_iter())
        .unwrap();
    space.named_row(&reduced, column).unwrap()
}

#[test]
fn unrelated_kernel_does_not_reject_an_injective_named_component() {
    let mut space = empty_space();
    space.topology.diagram.make_node(1, ZERO, ONE).unwrap();
    let x = space.topology.diagram.make_node(0, ZERO, ONE).unwrap();
    let names = [("m".to_owned(), 10)];
    let unknown = names.iter().collect::<Vec<_>>();
    let mut rows = BooleanRowSpace::new(
        vec![
            BooleanRow::from_terms(ONE, [(9, ONE)]),
            BooleanRow::from_terms(ONE, [(0, x), (10, ONE)]),
        ],
        &mut space.topology.diagram,
    )
    .unwrap();
    let selected = space.named_component_rows(&rows, &unknown).unwrap();
    assert_eq!(selected, [1]);
    let nodes = space.topology.diagram.nodes().len();
    space.collect_readout_roots(rows.decisions_mut()).unwrap();
    assert!(space.topology.diagram.nodes().len() < nodes);
    assert!(
        space
            .injective_named_candidates(&rows, &[0, 1], &unknown, 10, 11)
            .unwrap()
            .is_none()
    );
    let candidate = space
        .injective_named_candidates(&rows, &selected, &unknown, 10, 11)
        .unwrap()
        .unwrap()
        .remove(0)
        .unwrap();
    let expected = named_row_from_ids(&mut space, &rows, &[0, 1], 10);
    assert_eq!(
        (candidate.active, candidate.terms()),
        (expected.active, expected.terms())
    );
    assert_eq!(
        rows.len(),
        2,
        "selection retains the closed source relation"
    );
}

#[test]
fn component_walk_follows_physical_support_and_known_name_aliases_in_source_order() {
    let mut space = empty_space();
    let names = [("first".to_owned(), 10), ("second".to_owned(), 12)];
    let rows = BooleanRowSpace::new(
        vec![
            BooleanRow::from_terms(ONE, [(9, ONE)]),
            BooleanRow::from_terms(ONE, [(0, ONE), (10, ONE)]),
            BooleanRow::from_terms(ONE, [(0, ONE), (11, ONE)]),
            BooleanRow::from_terms(ONE, [(1, ONE), (11, ONE)]),
            BooleanRow::from_terms(ONE, [(2, ONE), (12, ONE)]),
        ],
        &mut space.topology.diagram,
    )
    .unwrap();
    assert_eq!(
        space
            .named_component_rows(&rows, &names.iter().collect::<Vec<_>>())
            .unwrap(),
        [1, 2, 3, 4]
    );
}

#[test]
fn feedback_parity_couples_disconnected_source_rows_before_selection() {
    let mut space = empty_space();
    let names = [("m".to_owned(), 10)];
    let unknown = names.iter().collect::<Vec<_>>();
    let (_, rows) = GuardedAvailability::new(
        vec![
            BooleanRow::from_terms(ONE, [(0, ONE), (10, ONE)]),
            BooleanRow::from_terms(ONE, [(1, ONE)]),
            BooleanRow::from_terms(ONE, [(9, ONE)]),
        ],
        &[(AvailabilityKey::Feedback(0), vec![(0, ONE), (1, ONE)])],
        11,
        &mut space.topology.diagram,
    )
    .unwrap();
    let selected = space.named_component_rows(&rows, &unknown).unwrap();
    assert_eq!(selected.len(), 1);
    let candidate = named_row_from_ids(&mut space, &rows, &selected, 10);
    let all = rows.indexed_rows().map(|(id, _)| id).collect::<Vec<_>>();
    let expected = named_row_from_ids(&mut space, &rows, &all, 10);
    assert_eq!(candidate.terms(), &[(0, ONE), (1, ONE), (10, ONE)]);
    assert_eq!(
        (candidate.active, candidate.terms()),
        (expected.active, expected.terms())
    );
}

#[test]
fn removed_inactive_postings_preserve_raw_rows_on_a_restricted_domain() {
    let mut space = empty_space();
    let x = space.topology.diagram.make_node(0, ZERO, ONE).unwrap();
    let not_x = space.topology.diagram.negate(x).unwrap();
    space.topology.domain = not_x;
    let names = [("m".to_owned(), 10)];
    let unknown = names.iter().collect::<Vec<_>>();
    let mut rows = BooleanRowSpace::new(
        vec![
            BooleanRow::from_terms(ONE, [(0, x), (2, ONE), (10, ONE)]),
            BooleanRow::from_terms(not_x, [(0, x), (3, ONE)]),
            BooleanRow::from_terms(ONE, [(4, ONE)]),
        ],
        &mut space.topology.diagram,
    )
    .unwrap();
    rows.eliminate_column(0, &mut space.topology.diagram)
        .unwrap();
    assert_eq!(rows.row_ids_with_column(0).count(), 0);
    assert_eq!(rows.row(0).unwrap().get(0), x);
    assert_eq!(rows.row(1).unwrap().get(0), x);
    let selected = space.named_component_rows(&rows, &unknown).unwrap();
    assert_eq!(selected, [0]);
    assert!(
        space
            .injective_named_candidates(&rows, &selected, &unknown, 10, 11)
            .unwrap()
            .is_none()
    );
    let candidate = named_row_from_ids(&mut space, &rows, &selected, 10);
    let expected = named_row_from_ids(&mut space, &rows, &[0, 1, 2], 10);
    assert_eq!(
        (candidate.active, candidate.terms()),
        (expected.active, expected.terms())
    );
    assert_eq!(candidate.active, ONE);
    assert_eq!(candidate.terms(), &[(2, ONE), (10, ONE)]);
}

#[test]
fn component_discovery_obeys_the_boolean_work_budget() {
    let mut space = empty_space();
    let rows = BooleanRowSpace::new(
        vec![BooleanRow::from_terms(ONE, [(0, ONE), (10, ONE)])],
        &mut space.topology.diagram,
    )
    .unwrap();
    let names = [("m".to_owned(), 10)];
    let remaining = space.topology.diagram.limits().max_steps - space.topology.diagram.steps();
    space.topology.diagram.charge(remaining - 1).unwrap();
    let before = space.topology.diagram.steps();
    assert!(matches!(
        space.named_component_rows(&rows, &names.iter().collect::<Vec<_>>()),
        Err(BooleanResourceError {
            resource: "Boolean work steps",
            ..
        })
    ));
    assert!(space.topology.diagram.steps() > before);
}

fn assert_complete_plan_matches_global_reduction(program: &crate::BlockGraph) {
    let linked = crate::flatten_module_definition(program, program.root(), "").unwrap();
    let graph = linked.graph.fix_shadowed_faces();
    let limits = ModuleCertificationLimits::DEFAULT;
    let build = || {
        GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &linked.sites,
            limits,
        )
        .unwrap()
    };
    let mut actual = build();
    actual
        .plan_causal_readouts_with_fast_path::<true, true, true>()
        .unwrap();
    let mut expected = build();
    expected
        .plan_causal_readouts_with_fast_path::<true, true, false>()
        .unwrap();
    super::tests::assert_same_raw_plan(&actual, &expected);
}

#[test]
fn complete_component_plans_match_global_reduction() {
    let mut phase = crate::GalleryItem::PhaseGradientK4.build().to_blog_text();
    phase = phase.replace(
        "module main {",
        "module main {\nclosed0: ComponentMemory @ [1000,0,0]\nclosed1: ComponentMemory @ [1003,0,0] rotate Z 90",
    );
    phase.push_str("\nmodule ComponentMemory {\n0: ZXX [0,0,0]\n}\n");
    let conditional = crate::parse_blog_program_to_ast(include_str!(
        "../../../../docs/fixtures/conditional_cz_strip.blog"
    ))
    .unwrap();
    for program in [
        crate::GalleryItem::PhaseGradientK4.build(),
        crate::BlockGraph::from_text(&phase).unwrap(),
        crate::GalleryItem::TComparison.build(),
        crate::GalleryItem::CCZInjectedAnd.build(),
        crate::GalleryItem::ToffoliFromAndDelayedCZ.build(),
        crate::lower_blog_graph_ast_deferred(&conditional).unwrap(),
    ] {
        assert_complete_plan_matches_global_reduction(&program);
    }
}
