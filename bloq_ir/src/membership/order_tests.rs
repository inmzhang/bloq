use super::*;
use crate::{BloqEdge, BloqNode, ValueRole};

fn input(graph: &mut SubGraph) -> BloqNodeId {
    graph.add_node(BloqNode::classical(ClassicalNode::fragment()))
}

fn compute(graph: &mut SubGraph, expr: ClassicalExpr, inputs: &[ValueRef]) -> BloqNodeId {
    let node = graph.add_node(BloqNode::classical(ClassicalNode::Compute { expr }));
    for (slot, value) in inputs.iter().enumerate() {
        graph.add_edge(
            value.node,
            node,
            BloqEdge::Value {
                slot: slot as u32,
                role: ValueRole::Data,
                output: value.output,
            },
        );
    }
    node
}

fn selected() -> ClassicalExpr {
    ClassicalExpr::Select(Box::new([
        ClassicalExpr::In(0),
        ClassicalExpr::In(1),
        ClassicalExpr::Const(false),
    ]))
}

fn evaluate(graph: &SubGraph, value: ValueRef, assignment: &FxMap<ValueRef, bool>) -> Option<bool> {
    let node = &graph[value.node];
    let read = |slot| {
        graph
            .value_inputs(value.node)
            .find(|input| input.slot == slot)
            .unwrap()
            .value_ref()
            .unwrap()
    };
    let payload = if value.output == ObservableOutput::Corrected
        && let Some(ClassicalNode::Compute { expr }) = node.try_classical()
    {
        expr.eval(&mut |slot| evaluate(graph, read(slot), assignment))
    } else {
        assignment.get(&value).copied()
    };
    Some(
        payload?
            & node
                .activation
                .map_or(Some(true), |slot| evaluate(graph, read(slot), assignment))?,
    )
}

fn check_assignments(analysis: &mut PredicateAnalysis<'_>, roots: &[ValueRef], atoms: &[ValueRef]) {
    let lowered = roots
        .iter()
        .map(|&root| analysis.guard(Some(root)).unwrap())
        .collect::<Vec<_>>();
    for bits in 0..1usize << atoms.len() {
        let assignment = atoms
            .iter()
            .enumerate()
            .map(|(i, &value)| (value, bits & (1 << i) != 0))
            .collect::<FxMap<_, _>>();
        let mut ranked = vec![false; analysis.variables.len()];
        for (value, &rank) in &analysis.variables {
            ranked[rank] = assignment[value];
        }
        for (&root, &lowered) in roots.iter().zip(&lowered) {
            assert_eq!(
                analysis.diagram.evaluate(lowered, |rank| ranked[rank]),
                evaluate(analysis.graph, root, &assignment).unwrap()
            );
        }
    }
}

fn clauses(n: usize) -> (SubGraph, Vec<ValueRef>, ValueRef, ValueRef, ValueRef) {
    use ClassicalExpr::{Const, In, Not, Or, Select};
    let mut graph = SubGraph::new();
    let atoms = (0..2 * n)
        .map(|_| input(&mut graph).into())
        .collect::<Vec<_>>();
    let mut source = compute(&mut graph, Const(true), &[]).into();
    for i in (0..n).rev() {
        let high = compute(
            &mut graph,
            Select(Box::new([In(0), Const(false), In(1)])),
            &[atoms[2 * i + 1], source],
        );
        source = compute(
            &mut graph,
            Select(Box::new([In(0), In(1), In(2)])),
            &[atoms[2 * i], source, high.into()],
        )
        .into();
    }
    let target = compute(
        &mut graph,
        Or(Box::new([Not(Box::new(In(0))), In(1)])),
        &atoms[..2],
    )
    .into();
    let absent = compute(&mut graph, Const(false), &[]).into();
    (graph, atoms, source, target, absent)
}

#[test]
fn recovered_order_preserves_truth_and_bounds_independent_clause_stress() {
    for n in 1..=3 {
        let (graph, atoms, source, target, absent) = clauses(n);
        for recovered in [false, true] {
            let mut analysis = if recovered {
                PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default())
            } else {
                PredicateAnalysis::new(&graph)
            };
            assert!(analysis.implies_values(Some(source), target).unwrap());
            assert!(!analysis.implies_values(Some(source), absent).unwrap());
            check_assignments(&mut analysis, &[source, target], &atoms);
        }
    }
    let (graph, _, source, target, absent) = clauses(12);
    let limits = BooleanLimits {
        max_nodes: 512,
        max_steps: 1_000_000,
    };
    let mut baseline = PredicateAnalysis::with_limits(&graph, limits);
    assert!(
        matches!(baseline.implies_values(Some(source), target), Err(Error::BooleanResource(error)) if error.resource == "Boolean nodes")
    );
    let mut recovered = PredicateAnalysis::for_membership_references(&graph, limits);
    assert!(recovered.implies_values(Some(source), target).unwrap());
    assert!(!recovered.implies_values(Some(source), absent).unwrap());
}

#[test]
fn fallback_and_literal_boundaries_preserve_value_semantics() {
    use ClassicalExpr::{Const, In, Select, Xor};
    let mut graph = SubGraph::new();
    let a = ValueRef::from(input(&mut graph));
    let b = ValueRef::from(input(&mut graph));
    let c = ValueRef::from(input(&mut graph));
    let absent = ValueRef::from(compute(&mut graph, Const(false), &[]));
    let negative = ValueRef::from(compute(&mut graph, ClassicalExpr::parity([0], true), &[a]));
    let repeated = ValueRef::from(compute(&mut graph, selected(), &[negative, a]));
    let ab = ValueRef::from(compute(&mut graph, selected(), &[a, b]));
    let ba = ValueRef::from(compute(&mut graph, selected(), &[b, a]));
    let parity = ValueRef::from(compute(&mut graph, Xor(Box::new([In(0), In(1)])), &[a, b]));
    for roots in [vec![repeated], vec![ab, ba], vec![parity]] {
        let mut analysis =
            PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
        analysis.prepare_membership_order(&roots, absent).unwrap();
        assert!(
            analysis.variables.is_empty(),
            "conflicting or absent constraints keep discovery order"
        );
        assert!(!analysis.membership_order_pending);
        check_assignments(&mut analysis, &roots, &[a, b]);
    }
    let shared = std::sync::Arc::new(ClassicalNode::Compute { expr: selected() });
    let mut shared_roots = Vec::new();
    for inputs in [[a, b], [b, c]] {
        let node = graph.add_node(BloqNode::classical(std::sync::Arc::clone(&shared)));
        for (slot, value) in inputs.into_iter().enumerate() {
            graph.add_edge(value.node, node, BloqEdge::value(slot as u32));
        }
        shared_roots.push(ValueRef::from(node));
    }
    let mut activated = BloqNode::classical(ClassicalNode::fragment());
    activated.activation = Some(0);
    let active = graph.add_node(activated);
    graph.add_edge(a.node, active, BloqEdge::value(0));
    let active_select = ValueRef::from(compute(&mut graph, selected(), &[active.into(), b]));
    let flip = ValueRef {
        node: a.node,
        output: ObservableOutput::Flip,
    };
    let flip_select = ValueRef::from(compute(
        &mut graph,
        Select(Box::new([In(0), Const(false), In(1)])),
        &[flip, b],
    ));
    let mut masked = PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    masked
        .prepare_membership_order(&[active_select], absent)
        .unwrap();
    assert!(
        masked.variables.is_empty(),
        "an activated value is not a literal condition"
    );
    let and = ValueRef::from(compute(
        &mut graph,
        ClassicalExpr::And(Box::new([In(0), In(1)])),
        &[b, c],
    ));
    let mut analysis =
        PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    shared_roots.extend([active_select, flip_select, and]);
    analysis
        .prepare_membership_order(&shared_roots, absent)
        .unwrap();
    assert!(
        analysis.variables[&a] < analysis.variables[&b]
            && analysis.variables[&b] < analysis.variables[&c]
    );
    assert!(analysis.variables[&flip] < analysis.variables[&b]);
    check_assignments(
        &mut analysis,
        &shared_roots,
        &[a, b, c, active.into(), flip],
    );
}

#[test]
fn setup_is_atomic_once_and_private_while_dataflow_errors_remain_errors() {
    let (mut graph, _, source, target, _) = clauses(2);
    let missing = ValueRef::from(compute(&mut graph, ClassicalExpr::In(99), &[]));
    let cycle = compute(&mut graph, ClassicalExpr::In(0), &[]);
    graph.add_edge(cycle, cycle, BloqEdge::value(0));
    let mut complete =
        PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    complete
        .prepare_membership_order(&[source], target)
        .unwrap();
    let setup_steps = complete.steps();
    assert!(setup_steps > 1);
    for max_steps in [0, setup_steps - 1] {
        let mut limited = PredicateAnalysis::for_membership_references(
            &graph,
            BooleanLimits {
                max_nodes: 512,
                max_steps,
            },
        );
        assert!(
            matches!(limited.prepare_membership_order(&[source], target), Err(Error::BooleanResource(error)) if error.resource == "Boolean work steps")
        );
        assert!(
            limited.variables.is_empty()
                && limited.diagram.nodes().is_empty()
                && limited.memo.is_empty()
        );
    }
    complete
        .prepare_membership_order(&[missing], missing)
        .unwrap();
    assert_eq!(
        complete.steps(),
        setup_steps,
        "a completed attempt runs once"
    );
    assert!(matches!(
        complete.implies_values(None, missing),
        Err(Error::MissingMembershipInput(99))
    ));
    let mut plain = PredicateAnalysis::new(&graph);
    plain.prepare_membership_order(&[missing], missing).unwrap();
    assert_eq!(plain.steps(), 0);
    assert!(plain.variables.is_empty());
    let mut malformed =
        PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    malformed
        .prepare_membership_order(&[cycle.into()], cycle.into())
        .unwrap();
    assert!(
        matches!(malformed.guard(Some(cycle.into())), Err(Error::InvalidMembership(message)) if message == "cyclic predicate ancestry")
    );
}

#[test]
fn membership_order_keeps_old_ranks_when_a_later_primitive_arrives_after_collection() {
    let (mut graph, mut atoms, source, target, absent) = clauses(2);
    let late = ValueRef::from(input(&mut graph));
    let late_source = ValueRef::from(compute(
        &mut graph,
        ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        &[late, source],
    ));
    let mut analysis =
        PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    assert!(analysis.implies_values(Some(source), target).unwrap());
    assert!(!analysis.membership_order_pending);
    assert_eq!(analysis.variables.len(), atoms.len());
    assert!(!analysis.variables.contains_key(&late));
    let original_ranks = analysis.variables.clone();
    let generation = analysis.diagram.generation();
    analysis.next_collection = 0;
    // A false target forces the real implication fallback and its normal GC boundary.
    assert!(!analysis.implies_values(Some(late_source), absent).unwrap());
    assert!(analysis.diagram.generation() > generation);
    assert_eq!(analysis.variables[&late], original_ranks.len());
    for (value, rank) in original_ranks {
        assert_eq!(analysis.variables[&value], rank);
    }
    assert!(
        !analysis
            .implies_all_values(&[late, source], absent)
            .unwrap()
    );
    assert!(analysis.implies_values(Some(source), target).unwrap());
    atoms.push(late);
    check_assignments(&mut analysis, &[source, target, late_source], &atoms);
}

#[test]
fn public_limited_enumeration_keeps_its_encounter_order_subset() {
    let (graph, atoms, source, target, _) = clauses(2);
    let outputs = [atoms[1].node, atoms[2].node];
    let encounter_subset = vec![vec![false, true], vec![true, true]];
    let recovered_subset = vec![vec![true, false], vec![true, true]];
    for explicit_limits in [false, true] {
        let mut public = if explicit_limits {
            PredicateAnalysis::with_limits(&graph, BooleanLimits::default())
        } else {
            PredicateAnalysis::new(&graph)
        };
        assert!(!public.membership_order_pending);
        assert!(public.implies_values(Some(source), target).unwrap());
        assert_eq!(public.values_up_to(&outputs, 2).unwrap(), encounter_subset);
    }
    // The same implication changes this subset when private recovery is enabled.
    let mut private =
        PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
    assert!(private.implies_values(Some(source), target).unwrap());
    assert!(!private.membership_order_pending);
    assert_eq!(private.values_up_to(&outputs, 2).unwrap(), recovered_subset);
    assert_ne!(encounter_subset, recovered_subset);
}
