//! Recover the ordering encoded by emitted decisions before auditing implications.
use super::*;

impl<'a> PredicateAnalysis<'a> {
    // Limited public enumeration observes ordering, so only audit owners opt in.
    pub(crate) fn for_membership_references(graph: &'a SubGraph, limits: BooleanLimits) -> Self {
        Self {
            membership_order_pending: true,
            ..Self::with_limits(graph, limits)
        }
    }

    // Polarity affects branch meaning, but either sign has the same precedence atom.
    fn order_literal(
        &mut self,
        owner: BloqNodeId,
        mut expr: &ClassicalExpr,
    ) -> Result<Option<ValueRef>, Error> {
        let slot = loop {
            self.charge(1)?;
            match expr {
                ClassicalExpr::In(slot) => break *slot,
                ClassicalExpr::Not(inner) => expr = inner,
                ClassicalExpr::Parity { inputs, .. } if inputs.len() == 1 => break inputs[0],
                _ => return Ok(None),
            }
        };
        let input = self.input(owner, slot)?;
        let (value, _) = self.normalize_alias(input)?;
        let node = self
            .graph
            .node(value.node)
            .ok_or_else(|| Error::InvalidMembership(format!("unknown predicate {value:?}")))?;
        Ok((node.activation.is_none()
            && !(value.output == ObservableOutput::Corrected
                && matches!(node.try_classical(), Some(ClassicalNode::Compute { .. }))))
        .then_some(value))
    }

    /// Recover the emitted decision order at most once, before the first
    /// implication fallback. A failed attempt is not retried or charged again.
    pub(super) fn prepare_membership_order(
        &mut self,
        roots: &[ValueRef],
        target: ValueRef,
    ) -> Result<(), Error> {
        if !std::mem::take(&mut self.membership_order_pending) {
            return Ok(());
        }
        self.recover_membership_order(roots, target)
    }

    fn recover_membership_order(
        &mut self,
        roots: &[ValueRef],
        target: ValueRef,
    ) -> Result<(), Error> {
        if !self.variables.is_empty() || !self.diagram.nodes().is_empty() {
            return Ok(());
        }
        use petgraph::{algo::toposort, graph::DiGraph};
        enum Task<'a> {
            Value(ValueRef, petgraph::graph::NodeIndex),
            Input(petgraph::graph::NodeIndex, ValueRef),
            Slot(petgraph::graph::NodeIndex, BloqNodeId, u32),
            Expr(BloqNodeId, &'a ClassicalExpr, petgraph::graph::NodeIndex),
        }
        let graph = self.graph;
        let mut order = DiGraph::<Option<ValueRef>, ()>::new();
        self.charge(1)?;
        let root_scope = order.add_node(None);
        let mut scopes = FxMap::default();
        let mut atoms = FxMap::default();
        self.charge(roots.len().saturating_add(1))?;
        let mut pending = vec![Task::Input(root_scope, target)];
        pending.extend(
            roots
                .iter()
                .rev()
                .map(|&root| Task::Input(root_scope, root)),
        );
        // Paths through scopes encode descendant precedence without expanding support sets.
        let mut has_constraints = false;
        while let Some(task) = pending.pop() {
            self.charge(1)?;
            match task {
                Task::Slot(parent, owner, slot) => {
                    pending.push(Task::Input(parent, self.input(owner, slot)?))
                }
                Task::Input(parent, value) => {
                    let scope = if let Some(&scope) = scopes.get(&value) {
                        scope
                    } else {
                        self.charge(1)?;
                        let scope = order.add_node(None);
                        scopes.insert(value, scope);
                        pending.push(Task::Value(value, scope));
                        scope
                    };
                    self.charge(1)?;
                    order.add_edge(parent, scope, ());
                }
                Task::Value(value, scope) => {
                    let node = graph.node(value.node).ok_or_else(|| {
                        Error::InvalidMembership(format!("unknown predicate {value:?}"))
                    })?;
                    if let Some(slot) = node.activation {
                        pending.push(Task::Slot(scope, value.node, slot));
                    }
                    if value.output == ObservableOutput::Corrected
                        && let Some(ClassicalNode::Compute { expr }) = node.try_classical()
                    {
                        pending.push(Task::Expr(value.node, expr, scope));
                    } else {
                        self.charge(2)?;
                        let atom = *atoms
                            .entry(value)
                            .or_insert_with(|| order.add_node(Some(value)));
                        order.add_edge(scope, atom, ());
                    }
                }
                Task::Expr(owner, expr, scope) => match expr {
                    ClassicalExpr::Const(_) => {}
                    ClassicalExpr::In(slot) => pending.push(Task::Slot(scope, owner, *slot)),
                    ClassicalExpr::Parity { inputs, .. } => {
                        self.charge(inputs.len())?;
                        for &slot in inputs.iter().rev() {
                            pending.push(Task::Slot(scope, owner, slot));
                        }
                    }
                    _ => {
                        let operands = expr.operands();
                        let literal = match expr {
                            ClassicalExpr::Select(_) => self.order_literal(owner, &operands[0])?,
                            // These binary forms are Select with one constant arm.
                            ClassicalExpr::And(_) | ClassicalExpr::Or(_) if operands.len() == 2 => {
                                self.order_literal(owner, &operands[0])?
                            }
                            _ => None,
                        };
                        let condition = if let Some(value) = literal {
                            self.charge(1)?;
                            Some(
                                *atoms
                                    .entry(value)
                                    .or_insert_with(|| order.add_node(Some(value))),
                            )
                        } else {
                            None
                        };
                        for (index, operand) in operands.iter().enumerate().rev() {
                            self.charge(2)?;
                            let child = order.add_node(None);
                            order.add_edge(scope, child, ());
                            if index > 0
                                && let Some(condition) = condition
                            {
                                self.charge(1)?;
                                order.add_edge(condition, child, ());
                                has_constraints = true;
                            }
                            pending.push(Task::Expr(owner, operand, child));
                        }
                    }
                },
            }
        }
        if !has_constraints {
            return Ok(());
        }
        self.charge(order.node_count().saturating_add(order.edge_count()))?;
        let Ok(sorted) = toposort(&order, None) else {
            return Ok(());
        };
        self.charge(sorted.len().saturating_add(atoms.len()))?;
        let variables = sorted
            .into_iter()
            .filter_map(|node| order[node])
            .enumerate()
            .map(|(rank, value)| (value, rank))
            .collect::<FxMap<_, _>>();
        assert_eq!(variables.len(), atoms.len());
        assert!(self.variables.is_empty());
        self.variables = variables;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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

    fn evaluate(
        graph: &SubGraph,
        value: ValueRef,
        assignment: &FxMap<ValueRef, bool>,
    ) -> Option<bool> {
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

    fn check_assignments(
        analysis: &mut PredicateAnalysis<'_>,
        roots: &[ValueRef],
        atoms: &[ValueRef],
    ) {
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
        let mut masked =
            PredicateAnalysis::for_membership_references(&graph, BooleanLimits::default());
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
}
