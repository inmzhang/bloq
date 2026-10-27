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
