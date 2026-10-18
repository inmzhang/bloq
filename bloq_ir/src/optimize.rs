//! Classical graph rewrites preserve recipe occurrences and selected output ports.

use std::sync::Arc;

use petgraph::Direction;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeIndexable, EdgeRef, IntoEdgeReferences, NodeIndexable};

use crate::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, ClassicalExpr, ClassicalNode,
    CycleDetected, SubGraph, ValueRole,
};

impl Bloq {
    /// Fuse computations and compatible observable recipes, then keep only the
    /// contributor frontier of observable ordering dependencies per level.
    /// Acyclicity is checked before any rewrite.
    ///
    /// # Errors
    ///
    /// Returns [`CycleDetected`] without rewriting anything if any level has
    /// a cycle.
    pub fn optimize(&mut self) -> Result<(), CycleDetected> {
        if !self.levels().all(|(_, level)| level.is_acyclic()) {
            return Err(CycleDetected);
        }
        self.top_mut().optimize_level(None);
        self.top_mut().share_classical_data();
        Ok(())
    }
}

impl SubGraph {
    /// JSON decodes each nested level separately; do not revisit its children.
    pub(crate) fn share_local_classical_data(&mut self) {
        let mut definitions = crate::FxSet::default();
        for node in self.graph_mut().node_weights_mut() {
            if let BloqNodeKind::Classical(data) = &mut node.kind {
                share_definition(data, &mut definitions);
            }
        }
    }

    /// Share exact classical bodies while retaining each node's own execution,
    /// bindings, activation, and provenance. The pool ends after this traversal;
    /// nodes own the shared definitions and mutable access detaches them.
    pub(crate) fn share_classical_data(&mut self) {
        fn share(level: &mut SubGraph, definitions: &mut crate::FxSet<Arc<ClassicalNode>>) {
            for node in level.graph_mut().node_weights_mut() {
                match &mut node.kind {
                    BloqNodeKind::Classical(data) => {
                        share_definition(data, definitions);
                    }
                    BloqNodeKind::Region(region) => {
                        for (_, body) in region.bodies_mut() {
                            share(body, definitions);
                        }
                    }
                    BloqNodeKind::Quantum(_) => {}
                }
            }
        }
        share(self, &mut crate::FxSet::default());
    }

    /// Run the optimization passes over this level, then recurse into every
    /// region body. Passes run before the recursion so a body is optimized as
    /// its final (post-merge) self, independent of its parent.
    fn optimize_level(&mut self, restart_source: Option<BloqNodeId>) {
        let mut pinned = self.rewrite_anchors(restart_source);
        pinned.extend(self.boundary_outputs());
        self.fuse_single_use_computes(&pinned);
        loop {
            let count = self.graph().node_count();
            self.fuse_local_observable_fragments(&pinned);
            if self.graph().node_count() == count {
                break;
            }
        }
        self.reduce_observable_order_frontiers();
        for node in self.graph_mut().node_weights_mut() {
            let classical = match &mut node.kind {
                BloqNodeKind::Classical(data)
                    if matches!(data.as_ref(),
                        ClassicalNode::Compute { expr } | ClassicalNode::Discard { condition: expr }
                        if expr.needs_affine_compaction()) =>
                {
                    Some(Arc::make_mut(data))
                }
                _ => None,
            };
            if let Some(
                ClassicalNode::Compute { expr } | ClassicalNode::Discard { condition: expr },
            ) = classical
            {
                expr.compact_affine();
            }
            if let BloqNodeKind::Region(region) = &mut node.kind {
                let crate::RegionNode::RepeatUntilSuccess {
                    restart_source,
                    restart_condition,
                    ..
                } = region;
                restart_condition.compact_affine();
                let restart_source = restart_source.map(|source| source.node);
                for (_, body) in region.bodies_mut() {
                    body.optimize_level(restart_source);
                }
            }
        }
        self.share_readout_recipes();
    }

    /// Share ordered readout inputs after payload merging has selected the
    /// final leaves. Discard levels retain their implicit execution barriers.
    fn share_readout_recipes(&mut self) {
        if self
            .nodes()
            .any(|(_, node)| matches!(node.try_classical(), Some(ClassicalNode::Discard { .. })))
        {
            return;
        }
        let graph = self.graph();
        // Binding products follow incoming adjacency, not input-slot order.
        // Batch rebuilding retains original edge-index order; require both
        // orders to agree, even on consumers that this pass will not rewrite.
        for target in graph.node_indices() {
            if !matches!(
                graph[target].try_classical(),
                Some(ClassicalNode::Observable { .. })
            ) {
                continue;
            }
            let mut previous = None;
            for edge in graph.edges_directed(target, Direction::Incoming) {
                if let BloqEdge::Compose { slot, .. } = edge.weight() {
                    let current = (*slot, edge.id());
                    if previous.is_some_and(|(slot, index)| current.0 >= slot || current.1 >= index)
                    {
                        return;
                    }
                    previous = Some(current);
                }
            }
        }
        let mut recipes = ReadoutRecipes::default();
        for target in graph.node_indices() {
            let node = &graph[target];
            if node.activation.is_some()
                || !matches!(node.try_classical(), Some(ClassicalNode::Observable { .. }))
                // Composition slots share a namespace with value operands.
                // Leave mixed consumers intact, including selected output ports.
                || graph.edges_directed(target, Direction::Incoming)
                    .any(|edge| matches!(edge.weight(), BloqEdge::Value { .. }))
            {
                continue;
            }
            let mut inputs = graph
                .edges_directed(target, Direction::Incoming)
                .filter_map(|edge| match edge.weight() {
                    BloqEdge::Compose { slot, role } => {
                        Some((*slot, edge.source(), role.clone(), edge.id()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if inputs.len() < 2 || u32::try_from(inputs.len()).is_err() {
                continue;
            }
            inputs.reverse();
            recipes.add(target, inputs);
        }
        recipes.rewrite(self);
    }

    /// Inline measurement leaves into compatible consumers, or merge siblings
    /// with identical activation and composition uses. Boundary products retain
    /// their packet order: even X*Z*X and X*X*Z have different signs.
    fn fuse_local_observable_fragments(&mut self, pinned: &crate::FxSet<BloqNodeId>) {
        let activation = |level: &Self, id| {
            level[id].activation.map(|slot| {
                level
                    .value_inputs(id)
                    .find(|input| input.slot == slot)
                    .expect("activation has a value input")
                    .value_ref()
                    .expect("runtime input")
            })
        };
        let mut siblings = crate::FxMap::default();
        let candidates = self
            .nodes()
            .filter_map(|(source, node)| {
                if pinned.contains(&source)
                    || !matches!(
                        node.try_classical(),
                        Some(ClassicalNode::Observable { index: None, operators, .. })
                            if operators.is_empty()
                    )
                    || self.data_inputs(source).next().is_some()
                {
                    return None;
                }
                let mut targets = Vec::new();
                for edge in self.outgoing(source) {
                    if !matches!(
                        edge.edge,
                        BloqEdge::Compose {
                            role: ValueRole::Data,
                            ..
                        }
                    ) || !matches!(
                        self[edge.target].try_classical(),
                        Some(ClassicalNode::Observable { .. })
                    ) {
                        return None;
                    }
                    targets.push(edge.target);
                }
                if targets.is_empty() {
                    return None;
                }
                let active = activation(self, source);
                if targets
                    .iter()
                    .all(|&target| activation(self, target) == active)
                {
                    return Some((source, targets));
                }
                // Stamped recipes retain their source identity. Siblings may
                // use different slot numbers but must have equal use counts.
                if node.provenance != crate::NodeProvenance::None {
                    return None;
                }
                targets.sort_unstable();
                match siblings.entry((active, targets)) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(source);
                        None
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        Some((source, vec![*entry.get()]))
                    }
                }
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return;
        }
        let mut orders = self
            .edges()
            .filter(|edge| matches!(edge.edge, BloqEdge::Order))
            .map(|edge| (edge.source, edge.target))
            .collect::<crate::FxSet<_>>();
        let mut removed = crate::FxSet::default();
        let mut rewired = Vec::new();
        for (source, targets) in candidates {
            let Some(ClassicalNode::Observable { measurements, .. }) = self
                .node_mut(source)
                .expect("candidate remains live")
                .try_classical_mut()
            else {
                unreachable!()
            };
            let local_measurements = std::mem::take(measurements);
            for target in targets {
                for edge in self
                    .incoming(source)
                    .filter(|edge| matches!(edge.edge, BloqEdge::Order))
                {
                    if orders.insert((edge.source, target)) {
                        rewired.push((
                            NodeIndex::new(edge.source.0 as usize),
                            NodeIndex::new(target.0 as usize),
                            BloqEdge::Order,
                        ));
                    }
                }
                let Some(ClassicalNode::Observable { measurements, .. }) = self
                    .node_mut(target)
                    .expect("consumer remains live")
                    .try_classical_mut()
                else {
                    unreachable!()
                };
                measurements.extend_from_slice(&local_measurements);
            }
            removed.insert(NodeIndex::new(source.0 as usize));
        }
        let graph = self.graph_mut();
        retain_edges_preserving_adjacency(graph, |_, source, target| {
            !removed.contains(&source) && !removed.contains(&target)
        });
        for &node in &removed {
            graph.remove_node(node);
        }
        for (source, target, edge) in rewired {
            graph.add_edge(source, target, edge);
        }
    }

    /// Direct unconditional successors already enforce their predecessors' order.
    /// Each contributor scans only its outgoing adjacency; guarded execution is
    /// retained because selecting a successor need not execute its predecessor.
    fn reduce_observable_order_frontiers(&mut self) {
        let successors = self
            .quantum_nodes()
            .filter(|(_, q)| q.guards.is_empty())
            .map(|(source, _)| {
                let next = self
                    .outgoing(source)
                    .filter_map(|edge| {
                        (self[edge.target]
                            .try_quantum()
                            .is_some_and(|q| q.guards.is_empty())
                            && match edge.edge {
                                BloqEdge::Order => true,
                                BloqEdge::Quantum(q) => q.guard.is_none(),
                                _ => false,
                            })
                        .then_some(edge.target)
                    })
                    .collect::<Vec<_>>();
                (source, next)
            })
            .collect::<crate::FxMap<_, _>>();
        let mut redundant = crate::FxSet::default();
        for (observable, node) in self.nodes() {
            if !matches!(node.try_classical(), Some(ClassicalNode::Observable { .. })) {
                continue;
            }
            let mut contributors = crate::FxMap::<BloqNodeId, Vec<EdgeIndex>>::default();
            for edge in self
                .graph()
                .edges_directed(NodeIndex::new(observable.0 as usize), Direction::Incoming)
                .filter(|edge| matches!(edge.weight(), BloqEdge::Order))
            {
                contributors
                    .entry(BloqNodeId(edge.source().index() as u32))
                    .or_default()
                    .push(edge.id());
            }
            for (&source, orders) in &contributors {
                if successors
                    .get(&source)
                    .is_some_and(|next| next.iter().any(|target| contributors.contains_key(target)))
                {
                    redundant.extend(orders.iter().copied());
                }
            }
        }
        retain_edges_preserving_adjacency(self.graph_mut(), |edge, _, _| {
            !redundant.contains(&edge)
        });
    }

    /// Contract single-use compute forests, then rebuild adjacency once.
    /// Per-node deletion scans shared neighbors' edge lists quadratically.
    fn fuse_single_use_computes(&mut self, pinned: &crate::FxSet<BloqNodeId>) {
        let mut parents = self
            .graph()
            .node_indices()
            .filter_map(|node| {
                self.fusable_compute_consumer(node, pinned)
                    .map(|consumer| (node, consumer))
            })
            .collect::<crate::FxMap<_, _>>();
        if parents.is_empty() {
            return;
        }
        // A nonlinear expression can read one slot repeatedly. Keep that
        // producer shared instead of duplicating its expression tree.
        let mut uses = parents
            .values()
            .map(|&input| (input, 0usize))
            .collect::<crate::FxMap<_, _>>();
        let mut consumers = uses.keys().map(|&(node, _)| node).collect::<Vec<_>>();
        consumers.sort_unstable();
        consumers.dedup();
        for consumer in consumers {
            let Some(ClassicalNode::Compute { expr }) = self.graph()[consumer].try_classical()
            else {
                unreachable!("candidate consumer is a Compute")
            };
            expr.for_each_input(&mut |slot| {
                if let Some(count) = uses.get_mut(&(consumer, slot)) {
                    *count += 1;
                }
            });
        }
        parents.retain(|_, input| uses[input] == 1);
        if parents.is_empty() {
            return;
        }

        // Accepted links form a forest: each producer has exactly one outgoing
        // edge, and the graph was checked acyclic before these passes. Order
        // only its nodes, with children before their consumer.
        let mut children = crate::FxMap::<NodeIndex, usize>::default();
        for (&producer, &(consumer, _)) in &parents {
            children.entry(producer).or_default();
            *children.entry(consumer).or_default() += 1;
        }
        let mut ready = children
            .iter()
            .filter_map(|(&node, &count)| (count == 0).then_some(node))
            .collect::<Vec<_>>();
        ready.sort_unstable();
        let mut order = Vec::with_capacity(children.len());
        while let Some(node) = ready.pop() {
            order.push(node);
            if let Some(&(consumer, _)) = parents.get(&node) {
                let remaining = children
                    .get_mut(&consumer)
                    .expect("accepted consumer belongs to the forest");
                *remaining -= 1;
                if *remaining == 0 {
                    ready.push(consumer);
                }
            }
        }
        assert_eq!(order.len(), children.len(), "accepted links are acyclic");
        let mut roots = crate::FxMap::default();
        for &node in order.iter().rev() {
            if let Some(&(consumer, _)) = parents.get(&node) {
                let root = roots.get(&consumer).copied().unwrap_or(consumer);
                roots.insert(node, root);
                roots.insert(root, root);
            }
        }

        let mut expressions = crate::FxMap::default();
        let mut next_slots = crate::FxMap::<NodeIndex, u32>::default();
        let mut replaced_inputs = crate::FxSet::default();
        let mut rewired = Vec::new();
        for node in order {
            let Some(&root) = roots.get(&node) else {
                continue;
            };
            let mut inputs = crate::FxMap::default();
            for edge in self.graph().edges_directed(node, Direction::Incoming) {
                match edge.weight() {
                    BloqEdge::Value { slot, role, output } => {
                        inputs.insert(*slot, (edge.source(), role.clone(), *output, None));
                        if node == root {
                            replaced_inputs.insert(edge.id());
                        }
                    }
                    BloqEdge::Order if node != root => {
                        rewired.push((edge.source(), root, BloqEdge::Order));
                    }
                    _ => {}
                }
            }
            let Some(ClassicalNode::Compute { expr }) = self.graph_mut()[node].try_classical_mut()
            else {
                unreachable!("fusion forests contain only Computes")
            };
            let mut expr = std::mem::replace(expr, ClassicalExpr::Const(false));
            expr.replace_inputs(&mut |slot| {
                let (source, role, output, assigned) =
                    inputs.get_mut(&slot).expect("fed expression slot");
                if parents.contains_key(source) {
                    return expressions.remove(source).expect("single expression use");
                }
                let assigned = *assigned.get_or_insert_with(|| {
                    let next = next_slots.entry(root).or_default();
                    let assigned = *next;
                    *next = next
                        .checked_add(1)
                        .expect("IR edge count fits the slot width");
                    rewired.push((
                        *source,
                        root,
                        BloqEdge::Value {
                            slot: assigned,
                            role: role.clone(),
                            output: *output,
                        },
                    ));
                    assigned
                });
                ClassicalExpr::In(assigned)
            });
            expressions.insert(node, expr);
        }
        for (node, expr) in expressions {
            debug_assert!(
                !parents.contains_key(&node),
                "a fusion root cannot also be an inlined child"
            );
            self.graph_mut()[node].kind =
                BloqNodeKind::Classical(ClassicalNode::Compute { expr }.into());
        }
        let graph = self.graph_mut();
        retain_edges_preserving_adjacency(graph, |edge, source, target| {
            !parents.contains_key(&source)
                && !parents.contains_key(&target)
                && !replaced_inputs.contains(&edge)
        });
        for &node in parents.keys() {
            graph.remove_node(node);
        }
        for (source, target, edge) in rewired {
            graph.add_edge(source, target, edge);
        }
    }

    /// Uses outside ordinary graph edges: output frames, named source choices,
    /// quantum seam guards, the declared result, and the enclosing retry predicate (WF-10).
    /// Membership and activation inputs already count as ordinary Value uses.
    fn rewrite_anchors(&self, restart_source: Option<BloqNodeId>) -> crate::FxSet<BloqNodeId> {
        let mut pinned = self
            .graph()
            .node_indices()
            .filter_map(|index| {
                matches!(
                    self.graph()[index].provenance,
                    crate::NodeProvenance::OutputFrame { .. }
                        | crate::NodeProvenance::BranchSelector { .. }
                )
                .then_some(BloqNodeId(index.index() as u32))
            })
            .collect::<crate::FxSet<_>>();
        pinned.extend(restart_source);
        pinned.extend(self.value_output().map(|value| value.node));
        pinned.extend(self.edges().filter_map(|edge| match edge.edge {
            BloqEdge::Quantum(quantum) => quantum.guard.map(|value| value.node),
            _ => None,
        }));
        pinned
    }

    /// The sole `Value` consumer and its slot, when both endpoints are
    /// Computes and the producer is not referenced off-graph.
    fn fusable_compute_consumer(
        &self,
        producer: NodeIndex,
        pinned: &crate::FxSet<BloqNodeId>,
    ) -> Option<(NodeIndex, u32)> {
        let graph = self.graph();
        if !matches!(
            graph.node_weight(producer)?.try_classical(),
            Some(ClassicalNode::Compute { .. })
        ) || pinned.contains(&BloqNodeId(producer.index() as u32))
            || graph[producer].activation.is_some()
        {
            return None;
        }
        let mut outgoing = graph.edges_directed(producer, Direction::Outgoing);
        let (Some(edge), None) = (outgoing.next(), outgoing.next()) else {
            return None;
        };
        let BloqEdge::Value {
            slot,
            role: ValueRole::Data,
            output: crate::ObservableOutput::Corrected,
        } = edge.weight()
        else {
            return None;
        };
        let consumer = edge.target();
        // SEM-ACTIVATE skips operand evaluation, while a plain Compute evaluates
        // every operand. Preserve that boundary, including unavailable inputs.
        (graph[consumer].activation.is_none()
            && matches!(
                graph[consumer].try_classical(),
                Some(ClassicalNode::Compute { .. })
            ))
        .then_some((consumer, *slot))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ReadoutOperand {
    Leaf(NodeIndex, ValueRole),
    Pair(usize),
}

/// Virtual balanced trees have at most one pair per original input edge.
/// Only subtrees that save edges become shared, zero-duration IR recipes.
// ponytail: balanced pairs miss shifted overlaps; broaden matching only when
// measured savings justify the extra compilation work.
#[derive(Default)]
struct ReadoutRecipes {
    pairs: Vec<([ReadoutOperand; 2], usize)>,
    interned: crate::FxMap<[ReadoutOperand; 2], usize>,
    readouts: Vec<(NodeIndex, usize, Vec<EdgeIndex>)>,
}

impl ReadoutRecipes {
    fn add(&mut self, target: NodeIndex, inputs: Vec<(u32, NodeIndex, ValueRole, EdgeIndex)>) {
        let edges = inputs.iter().map(|input| input.3).collect();
        let mut row = inputs
            .into_iter()
            .map(|(_, node, role, _)| ReadoutOperand::Leaf(node, role))
            .collect::<Vec<_>>();
        while row.len() > 1 {
            let mut next = Vec::with_capacity(row.len().div_ceil(2));
            for chunk in row.chunks(2) {
                let [left, right] = chunk else {
                    next.push(chunk[0].clone());
                    continue;
                };
                let key = [left.clone(), right.clone()];
                let id = if let Some(&id) = self.interned.get(&key) {
                    id
                } else {
                    for child in &key {
                        if let ReadoutOperand::Pair(child) = child {
                            self.pairs[*child].1 += 1;
                        }
                    }
                    let id = self.pairs.len();
                    self.pairs.push((key.clone(), 0));
                    self.interned.insert(key, id);
                    id
                };
                next.push(ReadoutOperand::Pair(id));
            }
            row = next;
        }
        let Some(ReadoutOperand::Pair(root)) = row.pop() else {
            unreachable!("at least two input edges form a pair");
        };
        self.pairs[root].1 += 1;
        self.readouts.push((target, root, edges));
    }

    fn rewrite(mut self, level: &mut SubGraph) {
        drop(std::mem::take(&mut self.interned));
        let mut arities = Vec::<usize>::with_capacity(self.pairs.len());
        let mut kept = Vec::<bool>::with_capacity(self.pairs.len());
        for (children, uses) in &self.pairs {
            let Some(arity) = children.iter().try_fold(0usize, |sum, child| {
                sum.checked_add(match child {
                    ReadoutOperand::Pair(id) if !kept[*id] => arities[*id],
                    _ => 1,
                })
            }) else {
                return;
            };
            if u32::try_from(arity).is_err() {
                return;
            }
            // Flattening a parent can increase a child's actual uses. Direct
            // DAG uses therefore retain only conservative, proven savings.
            kept.push(arity.saturating_add(*uses) < arity.saturating_mul(*uses));
            arities.push(arity);
        }
        let new_nodes = kept.iter().filter(|&&keep| keep).count();
        if new_nodes == 0
            || level
                .graph()
                .node_bound()
                .checked_add(new_nodes)
                .is_none_or(|bound| bound > u32::MAX as usize)
        {
            return;
        }
        let rewritten = self
            .readouts
            .iter()
            .enumerate()
            .filter_map(|(index, (_, root, edges))| {
                let arity = if kept[*root] { 1 } else { arities[*root] };
                (arity < edges.len()).then_some((index, arity))
            })
            .collect::<Vec<_>>();
        let Some(new_edges) = arities
            .iter()
            .zip(&kept)
            .filter(|(_, keep)| **keep)
            .map(|(&arity, _)| arity)
            .chain(rewritten.iter().map(|(_, arity)| *arity))
            .try_fold(0usize, usize::checked_add)
        else {
            return;
        };
        let removed = rewritten
            .iter()
            .flat_map(|(index, _)| self.readouts[*index].2.iter().copied())
            .collect::<crate::FxSet<_>>();
        if new_edges >= removed.len() {
            return;
        }
        let graph = level.graph_mut();
        let mut emitted = vec![None; self.pairs.len()];
        for (id, &keep) in kept.iter().enumerate() {
            if keep {
                emitted[id] = Some(graph.add_node(BloqNode::classical(ClassicalNode::fragment())));
            }
        }
        retain_edges_preserving_adjacency(graph, |edge, _, _| !removed.contains(&edge));
        for (id, (children, _)) in self.pairs.iter().enumerate() {
            if let Some(target) = emitted[id] {
                self.wire(graph, target, children, &emitted);
            }
        }
        for (index, _) in rewritten {
            let (target, root, _) = &self.readouts[index];
            self.wire(graph, *target, &[ReadoutOperand::Pair(*root)], &emitted);
        }
    }

    fn wire(
        &self,
        graph: &mut StableDiGraph<BloqNode, BloqEdge>,
        target: NodeIndex,
        inputs: &[ReadoutOperand],
        emitted: &[Option<NodeIndex>],
    ) {
        let mut pending = inputs.iter().rev().collect::<Vec<_>>();
        let mut slot = 0u32;
        while let Some(input) = pending.pop() {
            let (source, role) = match input {
                ReadoutOperand::Leaf(source, role) => (*source, role.clone()),
                ReadoutOperand::Pair(id) => {
                    if let Some(source) = emitted[*id] {
                        (source, ValueRole::Data)
                    } else {
                        pending.extend(self.pairs[*id].0.iter().rev());
                        continue;
                    }
                }
            };
            graph.add_edge(source, target, BloqEdge::Compose { slot, role });
            slot = slot
                .checked_add(1)
                .expect("recipe arity was checked before rewriting");
        }
    }
}

fn share_definition(
    data: &mut Arc<ClassicalNode>,
    definitions: &mut crate::FxSet<Arc<ClassicalNode>>,
) {
    if let Some(shared) = definitions.get(data) {
        *data = Arc::clone(shared);
    } else {
        definitions.insert(Arc::clone(data));
    }
}

/// Batch the usual descending-index adjacency without changing either order.
/// Reused slots can break that order; those graphs retain individual deletion
/// and its degree-dependent cost.
fn retain_edges_preserving_adjacency<F>(
    graph: &mut StableDiGraph<BloqNode, BloqEdge>,
    mut retain: F,
) where
    F: FnMut(EdgeIndex, NodeIndex, NodeIndex) -> bool,
{
    let removed = graph
        .edge_references()
        .filter_map(|edge| (!retain(edge.id(), edge.source(), edge.target())).then_some(edge.id()))
        .collect::<crate::FxSet<_>>();
    if removed.is_empty() {
        return;
    }
    let ordered = graph.node_indices().all(|node| {
        [Direction::Incoming, Direction::Outgoing]
            .into_iter()
            .all(|direction| {
                graph
                    .edges_directed(node, direction)
                    .map(|edge| edge.id())
                    .filter(|edge| !removed.contains(edge))
                    .is_sorted_by(|left, right| left > right)
            })
    });
    if ordered {
        for (source, target, edge) in
            detach_retained_edges(graph, |edge, _, _| !removed.contains(&edge))
        {
            graph.add_edge(source, target, edge);
        }
    } else {
        for edge in removed {
            graph.remove_edge(edge);
        }
    }
}

/// Move retained payloads out, then clear adjacency before batch node removal.
/// Removing the now-isolated nodes is O(1); graph allocations remain reusable.
fn detach_retained_edges<F>(
    graph: &mut StableDiGraph<BloqNode, BloqEdge>,
    mut retain: F,
) -> impl Iterator<Item = (NodeIndex, NodeIndex, BloqEdge)> + use<F>
where
    F: FnMut(EdgeIndex, NodeIndex, NodeIndex) -> bool,
{
    // Most edges are plain Data inputs. Keep their endpoints and slot in
    // twelve bytes instead of copying the full edge enum for every entry.
    // Other payloads retain their ordinal in a sparse side list. Rebuilding
    // in this order leaves dense edge slots, just like decoding the program;
    // incident-only deletion would make later edits reuse different slots.
    let mut retained = Vec::with_capacity(graph.edge_count());
    let mut other = Vec::new();
    for index in 0..graph.edge_bound() {
        let edge = EdgeIndex::new(index);
        let Some((source, target)) = graph.edge_endpoints(edge) else {
            continue;
        };
        if retain(edge, source, target) {
            let weight = std::mem::replace(
                graph
                    .edge_weight_mut(edge)
                    .expect("edge endpoints came from the same live edge"),
                BloqEdge::Order,
            );
            let slot = match weight {
                BloqEdge::Value {
                    slot,
                    role: ValueRole::Data,
                    output: crate::ObservableOutput::Corrected,
                } => slot,
                weight => {
                    other.push((retained.len(), weight));
                    0
                }
            };
            retained.push((source, target, slot));
        }
    }
    graph.clear_edges();
    let mut other = other.into_iter().peekable();
    retained
        .into_iter()
        .enumerate()
        .map(move |(index, (source, target, slot))| {
            let weight = if other.peek().is_some_and(|(ordinal, _)| *ordinal == index) {
                other.next().expect("peeked retained edge").1
            } else {
                BloqEdge::value(slot)
            };
            (source, target, weight)
        })
}

/// Read a Compute expression for regression checks.
#[cfg(test)]
fn compute_expr(kind: &BloqNodeKind) -> ClassicalExpr {
    match kind.try_classical() {
        Some(ClassicalNode::Compute { expr }) => expr.clone(),
        _ => unreachable!("fusion only ever inlines Compute nodes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstanceId};

    fn expanded_readout_inputs(
        bloq: &Bloq,
        target: BloqNodeId,
        by_slot: bool,
    ) -> Vec<(BloqNodeId, ValueRole)> {
        let inputs = |node| {
            let mut inputs = bloq
                .data_inputs(node)
                .filter(|input| input.output.is_none())
                .collect::<Vec<_>>();
            if by_slot {
                inputs.sort_unstable_by_key(|input| input.slot);
            }
            inputs
                .into_iter()
                .map(|input| (input.producer, input.role.clone()))
                .collect::<Vec<_>>()
        };
        let mut pending = inputs(target).into_iter().rev().collect::<Vec<_>>();
        let mut leaves = Vec::new();
        while let Some((node, role)) = pending.pop() {
            if matches!(
                bloq[node].try_classical(),
                Some(ClassicalNode::Observable { index: None, measurements, operators }) if measurements.is_empty() && operators.is_empty()
            ) {
                assert_eq!(role, ValueRole::Data);
                pending.extend(inputs(node).into_iter().rev());
            } else {
                leaves.push((node, role));
            }
        }
        leaves
    }

    #[test]
    fn shared_readout_recipes_preserve_leaves_roles_payloads_and_owners() {
        let mut bloq = Bloq::new();
        let leaves = (0..3)
            .map(|index| bloq.add_node(BloqNode::classical(ClassicalNode::observable(100 + index))))
            .collect::<Vec<_>>();
        bloq.node_mut(leaves[1]).unwrap().activation = Some(0);
        bloq.add_edge(leaves[2], leaves[1], BloqEdge::value(0));
        let inputs = vec![
            (leaves[0], ValueRole::Data),
            (leaves[1], ValueRole::FeedbackFold { action: 7 }),
            (leaves[0], ValueRole::ReadoutFold),
            (leaves[2], ValueRole::Data),
            (leaves[0], ValueRole::Data),
            (leaves[1], ValueRole::FeedbackFold { action: 7 }),
            (leaves[0], ValueRole::ReadoutFold),
        ];
        let mut changed_role = inputs.clone();
        changed_role[1].1 = ValueRole::Data;
        let cases = [
            inputs.clone(),
            inputs.clone(),
            inputs.clone(),
            changed_role,
            Vec::new(),
            vec![inputs[0].clone()],
        ];
        let mut observables = Vec::new();
        for (index, inputs) in cases.iter().enumerate() {
            let node = bloq.add_node(
                BloqNode::classical(ClassicalNode::Observable {
                    index: Some(index as u32),
                    measurements: vec![InstanceMeasurement {
                        instance: TemplateInstanceId(0),
                        measurement: index as u32,
                    }],
                    operators: Vec::new(),
                })
                .with_provenance(crate::NodeProvenance::Generator {
                    ordinal: index as u32,
                }),
            );
            for (slot, (producer, role)) in inputs.iter().enumerate() {
                bloq.add_edge(
                    *producer,
                    node,
                    BloqEdge::Compose {
                        slot: 2 * slot as u32,
                        role: role.clone(),
                    },
                );
            }
            observables.push(node);
        }
        bloq.node_mut(observables[0]).unwrap().activation = Some(99);
        bloq.add_edge(leaves[2], observables[0], BloqEdge::value(99));
        let payloads = observables
            .iter()
            .map(|&node| bloq[node].try_classical().unwrap().clone())
            .collect::<Vec<_>>();
        let edges = bloq.edge_count();
        let activated_slots = value_slots_into(&bloq, observables[0]);
        let adjacency = observables
            .iter()
            .map(|&node| expanded_readout_inputs(&bloq, node, false))
            .collect::<Vec<_>>();
        bloq.top_mut().share_readout_recipes();
        assert!(bloq.edge_count() < edges);
        assert!(
            bloq.nodes().any(|(_, node)| matches!(
                node.try_classical(),
                Some(ClassicalNode::Observable { index: None, measurements, operators }) if measurements.is_empty() && operators.is_empty()
            ))
        );
        assert_eq!(bloq[leaves[1]].activation, Some(0));
        assert_eq!(bloq[observables[0]].activation, Some(99));
        assert_eq!(value_slots_into(&bloq, observables[0]), activated_slots);
        for (index, (&node, expected)) in observables.iter().zip(&cases).enumerate() {
            assert_eq!(bloq[node].try_classical(), Some(&payloads[index]));
            assert_eq!(
                bloq[node].provenance,
                crate::NodeProvenance::Generator {
                    ordinal: index as u32
                }
            );
            assert_eq!(expanded_readout_inputs(&bloq, node, true), *expected);
            assert_eq!(
                expanded_readout_inputs(&bloq, node, false),
                adjacency[index]
            );
        }
    }

    #[test]
    fn shared_readout_recipes_require_savings_and_skip_discard_levels() {
        for (width, uses, discard, expected_recipes) in [
            (1, 4, false, 0),
            (2, 2, false, 0),
            (4, 2, false, 1),
            (4, 3, true, 0),
        ] {
            let mut bloq = Bloq::new();
            let inputs = (0..width)
                .map(|_| bloq.add_node(BloqNode::classical(ClassicalNode::observable(100))))
                .collect::<Vec<_>>();
            for index in 0..uses {
                let node = bloq.add_node(BloqNode::classical(ClassicalNode::observable(index)));
                for (slot, &source) in inputs.iter().enumerate() {
                    bloq.add_edge(source, node, BloqEdge::compose(slot as u32));
                }
            }
            if discard {
                bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
                    condition: ClassicalExpr::Const(false),
                }));
            }
            let before = bloq.to_binary();
            let edges = bloq.edge_count();
            bloq.top_mut().share_readout_recipes();
            assert_eq!(
                bloq.nodes()
                    .filter(|(_, node)| matches!(
                        node.try_classical(),
                        Some(ClassicalNode::Observable { index: None, measurements, operators }) if measurements.is_empty() && operators.is_empty()
                    ))
                    .count(),
                expected_recipes
            );
            assert!(bloq.edge_count() <= edges);
            if expected_recipes == 0 {
                assert_eq!(bloq.to_binary(), before);
            }
        }
    }

    #[test]
    fn optimization_preserves_indexed_boundary_packet_order() {
        for operators in [
            "i0 output X(0,0), i0 output Z(0,0), i0 output X(0,0)",
            "i0 output Z(0,0), i0 output X(0,0), i0 output Z(0,0)",
        ] {
            let mut program = Bloq::from_text(&format!(
                "BLOQIR 1\ntemplate t0 {{\ncircuit {{\nR (0,0)\n}}\n}}\ngraph {{\nn0 quantum {{\ninstance i0 t0 @ (0,0)\n}}\nn1 observable 0 operators {operators}\nn0 -> n1 order\n}}"
            )).unwrap();
            program.validate().unwrap();
            let expected = program[BloqNodeId(1)]
                .try_classical()
                .unwrap()
                .operators()
                .to_vec();
            program.optimize().unwrap();
            program.optimize().unwrap();
            for actual in [
                program.clone(),
                Bloq::from_binary(&program.to_binary()).unwrap(),
                Bloq::from_text(&program.to_text()).unwrap(),
            ] {
                actual.validate().unwrap();
                // XZX=-Z and ZXZ=-X; sorting either vector loses that sign.
                assert_eq!(
                    actual[BloqNodeId(1)].try_classical().unwrap().operators(),
                    expected
                );
            }
        }
    }

    #[test]
    fn repeated_optimization_preserves_shared_signed_boundary_order() {
        for activated in [false, true] {
            let when = if activated { " when v8" } else { "" };
            let guards = if activated {
                "n7 -> n1 value 8 flip\nn7 -> n2 value 8 flip\nn7 -> n3 value 8 flip"
            } else {
                ""
            };
            let mut program = Bloq::from_text(&format!(
                "BLOQIR 1\ntemplate t0 {{\ncircuit {{\nR (0,0)\n}}\n}}\ngraph {{
n0 quantum {{\ninstance i0 t0 @ (0,0)\n}}
n1 observable fragment operators i0 output X(0,0){when}
n2 observable fragment operators i0 output Z(0,0){when}
n3 observable fragment operators i0 output X(0,0){when}
n4 observable 0
n5 observable 1
n6 observable 2
n7 observable 3
n0 -> n1 order
n0 -> n2 order
n0 -> n3 order
n1 -> n4 compose 0
n2 -> n4 compose 1
n3 -> n4 compose 2
n1 -> n5 compose 0
n2 -> n5 compose 1
n3 -> n5 compose 2
n1 -> n6 compose 0
n2 -> n6 compose 1
n3 -> n6 compose 2
{guards}
}}"
            ))
            .unwrap();
            program.validate().unwrap();
            let before = program.clone();
            program.optimize().unwrap();
            let once = program.to_text();
            program.optimize().unwrap();
            assert_eq!(program.to_text(), once, "optimization is idempotent");
            for actual in [
                program.clone(),
                Bloq::from_binary(&program.to_binary()).unwrap(),
                Bloq::from_text(&program.to_text()).unwrap(),
            ] {
                actual.validate().unwrap();
                for target in [4, 5, 6].map(BloqNodeId) {
                    // Adjacency multiplies X*Z*X = -Z. Fusing and sorting
                    // these operands into X*X*Z would produce +Z instead.
                    assert_eq!(
                        expanded_readout_inputs(&actual, target, false),
                        expanded_readout_inputs(&before, target, false)
                    );
                    for selected in [false, true] {
                        let assignment = crate::ClassicalAssignment::Uniform(selected);
                        assert_eq!(
                            actual.resolve_classical(target, assignment),
                            before.resolve_classical(target, assignment)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn readout_sharing_preserves_shuffled_signed_binding_order() {
        let mut bloq = Bloq::new();
        let leaves = [
            bloq_circuit::Pauli::X,
            bloq_circuit::Pauli::X,
            bloq_circuit::Pauli::Z,
        ]
        .into_iter()
        .map(|pauli| {
            bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
                index: None,
                measurements: Vec::new(),
                operators: vec![InstanceBoundaryOperator {
                    instance: TemplateInstanceId(0),
                    face: crate::BoundaryFace::Output,
                    operator: [(glam::IVec2::ZERO, pauli)].into_iter().collect(),
                }],
            }))
        })
        .collect::<Vec<_>>();
        let mut observables = Vec::new();
        for index in 0..2 {
            let node = bloq.add_node(BloqNode::classical(ClassicalNode::observable(index)));
            for (source, slot) in leaves.iter().zip([0, 2, 1]) {
                bloq.add_edge(*source, node, BloqEdge::compose(slot));
            }
            observables.push(node);
        }
        // Adjacency multiplies Z * X * X = Z. Slot-sorted regrouping would
        // multiply X * Z * X = -Z, so this level must remain untouched.
        let expected = vec![
            (leaves[2], ValueRole::Data),
            (leaves[1], ValueRole::Data),
            (leaves[0], ValueRole::Data),
        ];
        let before = bloq.to_binary();
        bloq.top_mut().share_readout_recipes();
        assert_eq!(bloq.to_binary(), before);
        for node in observables {
            assert_eq!(expanded_readout_inputs(&bloq, node, false), expected);
        }
    }

    #[test]
    fn readout_sharing_skips_edge_reuse_on_unmodified_consumers() {
        for activated in [false, true] {
            let mut bloq = Bloq::new();
            let leaves = (0..4)
                .map(|_| bloq.add_node(BloqNode::classical(ClassicalNode::observable(100))))
                .collect::<Vec<_>>();
            let mut untouched = BloqNode::classical(if activated {
                ClassicalNode::observable(0)
            } else {
                ClassicalNode::fragment()
            });
            if activated {
                untouched.activation = Some(99);
            }
            let untouched = bloq.add_node(untouched);
            if activated {
                bloq.add_edge(leaves[0], untouched, BloqEdge::value(99));
            }
            let index = |node: BloqNodeId| NodeIndex::new(node.0 as usize);
            let graph = bloq.top_mut().graph_mut();
            let hole = graph.add_edge(index(leaves[0]), index(leaves[1]), BloqEdge::Order);
            let first = graph.add_edge(index(leaves[0]), index(untouched), BloqEdge::compose(0));
            graph.remove_edge(hole);
            let reused = graph.add_edge(index(leaves[1]), index(untouched), BloqEdge::compose(1));
            assert!(reused < first);
            // These other, canonical observables would otherwise share a recipe.
            for ordinal in 1..4 {
                let node = bloq.add_node(BloqNode::classical(ClassicalNode::observable(ordinal)));
                for (slot, &source) in leaves.iter().enumerate() {
                    bloq.add_edge(source, node, BloqEdge::compose(slot as u32));
                }
            }
            let before_inputs = bloq
                .data_inputs(untouched)
                .map(|input| (input.producer, input.slot))
                .collect::<Vec<_>>();
            assert_eq!(
                before_inputs
                    .iter()
                    .map(|input| input.1)
                    .collect::<Vec<_>>(),
                [1, 0]
            );
            let before = bloq.to_binary();
            bloq.top_mut().share_readout_recipes();
            assert_eq!(bloq.to_binary(), before);
            assert_eq!(
                bloq.data_inputs(untouched)
                    .map(|input| (input.producer, input.slot))
                    .collect::<Vec<_>>(),
                before_inputs
            );
        }
    }

    #[test]
    fn recipe_sharing_preserves_mixed_value_ports_and_composition_slots() {
        let mut program = Bloq::new();
        let leaves = (0..4)
            .map(|index| program.add_node(BloqNode::classical(ClassicalNode::observable(index))))
            .collect::<Vec<_>>();
        let mut targets = Vec::new();
        for index in 4..7 {
            let target = program.add_node(BloqNode::classical(ClassicalNode::observable(index)));
            for (slot, &source) in leaves.iter().enumerate() {
                program.add_edge(source, target, BloqEdge::compose(slot as u32));
            }
            targets.push(target);
        }
        let mixed = targets[2];
        program.add_edge(leaves[0], mixed, BloqEdge::flip(4));
        program.validate().unwrap();
        let before = program.clone();
        let edges = program.edge_count();
        let inputs = program
            .data_inputs(mixed)
            .map(|input| (input.producer, input.slot, input.output))
            .collect::<Vec<_>>();

        program.top_mut().share_readout_recipes();

        program.validate().unwrap();
        assert!(program.edge_count() < edges);
        assert_eq!(
            program
                .data_inputs(mixed)
                .map(|input| (input.producer, input.slot, input.output))
                .collect::<Vec<_>>(),
            inputs
        );
        for target in targets {
            for value in [
                target.into(),
                crate::ValueRef {
                    node: target,
                    output: crate::ObservableOutput::Flip,
                },
            ] {
                let assignment = crate::ClassicalAssignment::Uniform(false);
                assert_eq!(
                    program.resolve_value(value, assignment),
                    before.resolve_value(value, assignment)
                );
                assert_eq!(
                    program.measurement_value_dependencies(value, assignment),
                    before.measurement_value_dependencies(value, assignment)
                );
            }
        }
    }

    #[test]
    fn retained_edge_scratch_preserves_weights_and_dense_future_edge_ids() {
        let mut bloq = Bloq::new();
        let nodes = (0..4)
            .map(|_| {
                bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                }))
            })
            .collect::<Vec<_>>();
        let index = |node: BloqNodeId| NodeIndex::new(node.0 as usize);
        let expected = vec![
            (index(nodes[0]), index(nodes[1]), BloqEdge::value(u32::MAX)),
            (index(nodes[1]), index(nodes[2]), BloqEdge::Order),
            (
                index(nodes[2]),
                index(nodes[3]),
                BloqEdge::Value {
                    slot: u32::MAX - 1,
                    role: ValueRole::FeedbackFold { action: u32::MAX },
                    output: crate::ObservableOutput::Corrected,
                },
            ),
            (
                index(nodes[0]),
                index(nodes[3]),
                BloqEdge::Value {
                    slot: 0,
                    role: ValueRole::ReadoutFold,
                    output: crate::ObservableOutput::Corrected,
                },
            ),
            (
                index(nodes[1]),
                index(nodes[3]),
                BloqEdge::Quantum(Box::new(crate::QuantumEdge {
                    pipes: Vec::new(),
                    guard: Some(nodes[0].into()),
                })),
            ),
        ];
        let graph = bloq.top_mut().graph_mut();
        graph.add_edge(expected[0].0, expected[0].1, expected[0].2.clone());
        let skipped = graph.add_edge(index(nodes[0]), index(nodes[2]), BloqEdge::value(9));
        for (source, target, edge) in &expected[1..] {
            graph.add_edge(*source, *target, edge.clone());
        }
        let hole = graph.add_edge(index(nodes[3]), index(nodes[2]), BloqEdge::Order);
        graph.remove_edge(hole);
        retain_edges_preserving_adjacency(graph, |id, _, _| id != skipped);
        assert_eq!(graph.edge_bound(), expected.len());
        assert_eq!(
            graph
                .edge_references()
                .map(|edge| (edge.source(), edge.target(), edge.weight().clone()))
                .collect::<Vec<_>>(),
            expected
        );

        let mut decoded = Bloq::from_binary(&bloq.to_binary()).unwrap();
        let added = (index(nodes[3]), index(nodes[0]), BloqEdge::value(17));
        let live_id = bloq
            .top_mut()
            .graph_mut()
            .add_edge(added.0, added.1, added.2.clone());
        let decoded_id = decoded
            .top_mut()
            .graph_mut()
            .add_edge(added.0, added.1, added.2);
        assert_eq!(live_id, decoded_id);
        assert_eq!(bloq.to_binary(), decoded.to_binary());
    }

    fn observable(bloq: &mut Bloq) -> BloqNodeId {
        bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)))
    }

    fn fragment(bloq: &mut Bloq, measurement: u32) -> BloqNodeId {
        bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![crate::InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
            Vec::new(),
        )))
    }

    fn value_slots_into(bloq: &Bloq, node: BloqNodeId) -> Vec<u32> {
        let mut slots: Vec<u32> = bloq.value_inputs(node).map(|input| input.slot).collect();
        slots.sort_unstable();
        slots
    }

    #[test]
    fn classical_definitions_share_without_merging_bindings_or_activation() {
        fn definition(bloq: &Bloq, id: u32) -> &Arc<ClassicalNode> {
            let BloqNodeKind::Classical(data) = &bloq[BloqNodeId(id)].kind else {
                panic!("expected a classical node")
            };
            data
        }

        let mut bloq = Bloq::from_text(
            r#"BLOQIR 1
template t0 {
  circuit {
    R (0,0)
    M (0,0):m0
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0
  n2 observable fragment measurements i0:m0 when v0
  n3 observable 0
  n4 compute in0
  n5 compute in0 when v1
  n0 -> n1 order
  n0 -> n2 order
  n3 -> n2 value 0
  n1 -> n4 value 0
  n2 -> n5 value 0
  n3 -> n5 value 1
}
"#,
        )
        .unwrap();
        bloq.validate().unwrap();
        let before = bloq.to_text();

        bloq.optimize().unwrap();

        bloq.validate().unwrap();
        assert_eq!(bloq.to_text(), before);
        assert!(Arc::ptr_eq(definition(&bloq, 1), definition(&bloq, 2)));
        assert!(Arc::ptr_eq(definition(&bloq, 4), definition(&bloq, 5)));
        let original = Arc::clone(definition(&bloq, 5));
        let Some(ClassicalNode::Compute { expr }) =
            bloq.node_mut(BloqNodeId(4)).unwrap().try_classical_mut()
        else {
            panic!("the edited node remains a Compute")
        };
        *expr = ClassicalExpr::Const(true);
        assert!(!Arc::ptr_eq(definition(&bloq, 4), &original));
        assert!(Arc::ptr_eq(definition(&bloq, 5), &original));
        assert_eq!(
            definition(&bloq, 5).as_ref(),
            &ClassicalNode::Compute {
                expr: ClassicalExpr::In(0)
            }
        );
    }

    #[test]
    fn cyclic_program_is_rejected_unchanged() {
        let mut bloq = Bloq::new();
        let a = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let b = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(a, b, BloqEdge::value(0));
        bloq.add_edge(b, a, BloqEdge::value(0));
        assert_eq!(bloq.optimize(), Err(CycleDetected));
        assert_eq!((bloq.top().node_count(), bloq.top().edge_count()), (2, 2));
    }

    #[test]
    fn guarded_rewrites_preserve_recipes_effects_and_external_uses() {
        let mut bloq = Bloq::from_text(
            r#"BLOQIR 1
template t0 {
  circuit {
    R (0,0)
    M (0,0):m0
    M (0,0):m1
  }
}
graph {
  n0 observable 0
  n1 compute in0 from selector choice
  n2 compute in0
  n3 compute in0
  n4 quantum {
    instance i0 t0 @ (0,0)
  }
  n5 quantum {
    instance i1 t0 @ (2,0)
    guard 7 i1
  }
  n6 observable fragment measurements i1:m0 when v0
  n7 observable fragment measurements i1:m1 when v1
  n8 observable fragment measurements i0:m0
  n9 observable fragment measurements i0:m1 when v2
  n10 observable 1 when v90
  n11 observable fragment operators i1 input X(2,0) when v4
  n12 observable fragment operators i1 output Z(2,0) when v5
  n13 compute !in0
  n14 compute !in9
  n15 compute in0 when v1
  n16 compute !in0
  n17 compute !in0
  n18 compute in0 when v1
  n19 observable fragment measurements i0:m1
  n21 rus in0 when v9 {
    body {
      n0 compute 1
      n1 compute !in0
      n0 -> n1 value 0
    }
  }
  n22 compute !in0
  n23 compute in0 ^ in1
  n24 compute in0 & in0
  n0 -> n1 value 0
  n1 -> n2 value 0
  n2 -> n3 value 0
  n3 -> n5 value 7
  n3 -> n6 value 0
  n3 -> n7 value 1
  n3 -> n10 value 90
  n3 -> n11 value 4
  n3 -> n12 value 5
  n3 -> n22 value 0
  n22 -> n9 value 2
  n5 -> n6 order
  n5 -> n7 order
  n5 -> n11 order
  n5 -> n12 order
  n4 -> n8 order
  n4 -> n9 order
  n6 -> n10 compose 2
  n7 -> n10 compose 5
  n8 -> n10 compose 6
  n9 -> n10 compose 7
  n11 -> n10 compose 8
  n12 -> n10 compose 9
  n0 -> n13 value 0
  n13 -> n14 value 9
  n0 -> n15 value 0
  n3 -> n15 value 1
  n15 -> n16 value 0
  n0 -> n17 value 0
  n17 -> n18 value 0
  n3 -> n18 value 1
  n4 -> n19 order
  n19 -> n10 value 12 readout
  n0 -> n21 value 0
  n3 -> n21 value 9
  n0 -> n23 value 0
  n3 -> n23 value 1
  n23 -> n24 value 0
}
"#,
        )
        .unwrap();
        bloq.add_edge(
            BloqNodeId(4),
            BloqNodeId(5),
            BloqEdge::Quantum(Box::new(crate::QuantumEdge {
                pipes: Vec::new(),
                guard: Some(BloqNodeId(2).into()),
            })),
        );
        bloq.validate().unwrap();
        let before = bloq.clone();
        bloq.optimize().expect("acyclic test program");
        bloq.validate().unwrap();

        assert!(bloq.node(BloqNodeId(13)).is_none());
        for retained in [1, 2, 8, 9, 15, 17, 19, 23] {
            assert!(bloq.node(BloqNodeId(retained)).is_some(), "n{retained}");
        }
        assert_eq!(
            bloq[BloqNodeId(23)].try_classical(),
            Some(&ClassicalNode::Compute {
                expr: ClassicalExpr::parity([0, 1], false)
            })
        );
        assert_eq!(
            bloq.top()
                .value_inputs(BloqNodeId(10))
                .find(|input| Some(input.slot) == bloq[BloqNodeId(10)].activation)
                .unwrap()
                .producer,
            BloqNodeId(3)
        );
        assert!(bloq.top().data_inputs(BloqNodeId(10)).any(|input| {
            input.producer == BloqNodeId(19) && *input.role == ValueRole::ReadoutFold
        }));
        let body = bloq[BloqNodeId(21)]
            .try_region()
            .unwrap()
            .bodies()
            .next()
            .unwrap()
            .1;
        assert_eq!(body.node_count(), 1);

        for optimized in [
            Bloq::from_binary(&bloq.to_binary()).unwrap(),
            Bloq::from_text(&bloq.to_text()).unwrap(),
        ] {
            optimized.validate().unwrap();
            for selected in [false, true] {
                let values = [(0, selected)].into();
                let assignment = crate::ClassicalAssignment::Pinned {
                    forced_observables: &values,
                };
                for output in [10, 14, 16, 18, 21] {
                    assert_eq!(
                        optimized.resolve_classical(BloqNodeId(output), assignment),
                        before.resolve_classical(BloqNodeId(output), assignment),
                        "n{output}, choice={selected}"
                    );
                }
                let choices = [("choice".into(), selected)].into();
                let old = before.pin_membership(&choices).unwrap();
                let new = optimized.pin_membership(&choices).unwrap();
                assert!(!new.has_conditional_membership());
                for quantum in [4, 5] {
                    assert_eq!(
                        old[BloqNodeId(quantum)].expect_quantum().instances,
                        new[BloqNodeId(quantum)].expect_quantum().instances,
                    );
                }
            }
        }
    }

    /// Fusing a producer used to leave a hole at its consumer slot
    /// (`Xor(In(2), In(1))`, no slot 0). The renumber sweep must leave the
    /// fused consumer with dense `0..n` edge slots and matching `In` leaves.
    #[test]
    fn fused_consumer_slots_are_dense() {
        let mut bloq = Bloq::new();
        let a = fragment(&mut bloq, 0);
        let b = fragment(&mut bloq, 1);
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        bloq.add_edge(a, producer, BloqEdge::value(0));
        bloq.add_edge(b, producer, BloqEdge::value(1));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(producer, consumer, BloqEdge::value(0));

        bloq.top_mut().fuse_single_use_computes(&Default::default());

        let slots = value_slots_into(&bloq, consumer);
        assert_eq!(slots, vec![0, 1], "edge slots renumber to dense 0..n");
        let Some(ClassicalNode::Compute { expr }) = bloq[consumer].try_classical() else {
            panic!("consumer stays a Compute");
        };
        let mut leaves = Vec::new();
        expr.for_each_input(&mut |slot| leaves.push(slot));
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1], "In leaves reference the dense slots");
    }

    #[test]
    fn fusion_preserves_shared_direct_and_indirect_input() {
        let mut bloq = Bloq::new();
        let source = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(source, producer, BloqEdge::value(0));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        bloq.add_edge(source, consumer, BloqEdge::value(0));
        bloq.add_edge(producer, consumer, BloqEdge::value(1));

        bloq.top_mut().fuse_single_use_computes(&Default::default());

        assert!(bloq.node(producer).is_none());
        assert_eq!(value_slots_into(&bloq, consumer), vec![0, 1]);
        let Some(ClassicalNode::Compute { expr }) = bloq[consumer].try_classical() else {
            panic!("consumer stays a Compute");
        };
        assert_eq!(expr.eval(&mut |_| Some(true)), Some(false));
        bloq.validate().unwrap();
    }

    #[test]
    fn fusion_handles_the_max_consumer_slot_without_overflow() {
        let mut bloq = Bloq::new();
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Xor(Box::new([
                    ClassicalExpr::In(u32::MAX - 1),
                    ClassicalExpr::In(u32::MAX),
                ])),
            ])),
        }));
        let source = observable(&mut bloq);
        bloq.add_edge(source, producer, BloqEdge::value(0));
        bloq.add_edge(source, producer, BloqEdge::value(1));
        bloq.add_edge(source, consumer, BloqEdge::value(0));
        bloq.add_edge(source, consumer, BloqEdge::value(u32::MAX - 1));
        bloq.add_edge(producer, consumer, BloqEdge::value(u32::MAX));
        bloq.validate().expect("max slot is valid IR");

        bloq.optimize().expect("acyclic test program");

        bloq.validate().expect("optimization leaves valid IR");
    }

    #[test]
    fn fusion_compacts_sparse_slots_through_compute_forests() {
        for (graph, expected) in [
            // A constant and a value meet at the maximum input slot.
            (
                "n0 compute in0\nn1 compute 1\nn2 compute in0 ^ in4294967295\nn3 observable 0\nn3 -> n0 value 0\nn0 -> n2 value 0\nn1 -> n2 value 4294967295",
                true,
            ),
            // A forest can cross a sparse intermediate interface.
            (
                "n0 compute in0 ^ in1\nn1 compute in4294967295\nn2 compute in0\nn3 observable 0\nn3 -> n0 value 0\nn3 -> n0 value 1\nn0 -> n1 value 4294967295\nn1 -> n2 value 0",
                false,
            ),
            // An internal constant removes one leaf from the final interface.
            (
                "n0 compute in0 ^ in1\nn1 compute 1\nn2 compute in0 ^ in4294967294\nn3 observable 0\nn1 -> n0 value 0\nn3 -> n0 value 1\nn0 -> n2 value 0\nn3 -> n2 value 4294967294",
                true,
            ),
        ] {
            let mut bloq = Bloq::from_text(&format!("BLOQIR 1\ngraph {{\n{graph}\n}}")).unwrap();
            bloq.validate().unwrap();
            bloq.optimize().expect("acyclic test program");
            bloq.validate().unwrap();
            assert!(bloq.node(BloqNodeId(0)).is_none(), "{graph}");
            assert!(bloq.node(BloqNodeId(1)).is_none(), "{graph}");
            assert_eq!(
                compute_expr(&bloq[BloqNodeId(2)].kind).eval(&mut |_| Some(false)),
                Some(expected),
                "{graph}"
            );
        }
    }

    #[test]
    fn fusion_orders_disconnected_branching_forests_and_keeps_order_dependencies() {
        let mut bloq = Bloq::new();
        let compute = |bloq: &mut Bloq, expr| {
            bloq.add_node(BloqNode::classical(ClassicalNode::Compute { expr }))
        };
        let source = compute(&mut bloq, ClassicalExpr::Const(true));
        let barrier = compute(&mut bloq, ClassicalExpr::Const(false));
        let left = compute(&mut bloq, ClassicalExpr::In(0));
        let right = compute(
            &mut bloq,
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        );
        let joined = compute(
            &mut bloq,
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        );
        let single = compute(&mut bloq, ClassicalExpr::In(0));
        let single_root = compute(
            &mut bloq,
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        );
        let repeated = compute(&mut bloq, ClassicalExpr::In(0));
        let repeated_root = compute(
            &mut bloq,
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(0)])),
        );
        for node in [left, right, single, repeated] {
            bloq.add_edge(source, node, BloqEdge::value(0));
        }
        bloq.add_edge(barrier, left, BloqEdge::Order);
        bloq.add_edge(left, joined, BloqEdge::value(0));
        bloq.add_edge(right, joined, BloqEdge::value(1));
        bloq.add_edge(single, single_root, BloqEdge::value(0));
        bloq.add_edge(repeated, repeated_root, BloqEdge::value(0));
        bloq.validate().unwrap();
        let before = bloq.clone();

        bloq.top_mut().fuse_single_use_computes(&Default::default());

        for removed in [left, right, single] {
            assert!(bloq.node(removed).is_none());
        }
        assert!(bloq.node(repeated).is_some(), "repeated slot blocks fusion");
        assert_eq!(value_slots_into(&bloq, joined), vec![0, 1]);
        assert_eq!(value_slots_into(&bloq, single_root), vec![0]);
        assert!(
            bloq.incoming(joined)
                .any(|edge| { edge.source == barrier && matches!(edge.edge, BloqEdge::Order) })
        );
        bloq.validate().unwrap();
        let values = Default::default();
        let assignment = crate::ClassicalAssignment::Pinned {
            forced_observables: &values,
        };
        for node in [joined, single_root, repeated_root] {
            assert_eq!(
                before.resolve_classical(node, assignment),
                bloq.resolve_classical(node, assignment)
            );
        }
    }

    #[test]
    fn optimize_preserves_every_rus_restart_source_value() {
        for body in [
            "n0 compute 0\nn1 compute !in0\nn0 -> n1 value 0",
            "n0 observable fragment\nn1 observable fragment\nn2 observable 0\nn0 -> n2 compose 0\nn1 -> n2 compose 1",
        ] {
            for source in [0, 1] {
                let mut bloq = Bloq::from_text(&format!(
                    "BLOQIR 1\ngraph {{\nn0 rus in0 source n{source} {{\nbody {{\n{body}\n}}\n}}\n}}"
                ))
                .unwrap();
                bloq.validate().unwrap();
                let path =
                    crate::LevelPath::default().child(BloqNodeId(0), crate::BodySelector::Body);
                let original = bloq.level_at(&path).unwrap()[BloqNodeId(source)]
                    .try_classical()
                    .unwrap()
                    .clone();

                bloq.optimize().expect("acyclic test program");

                bloq.validate().unwrap();
                let current = bloq.level_at(&path).unwrap()[BloqNodeId(source)]
                    .try_classical()
                    .unwrap();
                match (&original, current) {
                    (
                        ClassicalNode::Compute { expr: before },
                        ClassicalNode::Compute { expr: after },
                    ) => assert_eq!(
                        before.eval(&mut |_| Some(false)),
                        after.eval(&mut |_| Some(false))
                    ),
                    _ => assert_eq!(current, &original),
                }
            }
        }
    }

    /// An `OutputFrame`-stamped `Compute` is referenced off-graph (the frame
    /// table derives from stamps), so fusion must never inline it — even when
    /// its only edge is one `Value` into another `Compute`. The derived table
    /// keeps naming the surviving nodes.
    #[test]
    fn output_frame_stamped_compute_survives_fusion() {
        use crate::{Basis, FramePair, NodeProvenance};
        use glam::ivec3;

        let mut bloq = Bloq::new();
        let stamped = |basis| {
            BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })
            .with_provenance(NodeProvenance::OutputFrame {
                port: ivec3(0, 0, 1),
                basis,
            })
        };
        let frame_x = bloq.add_node(stamped(Basis::X));
        let frame_z = bloq.add_node(stamped(Basis::Z));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(frame_x, consumer, BloqEdge::value(0));

        bloq.optimize().expect("acyclic test program");

        assert_eq!(
            bloq.output_frames(),
            vec![FramePair {
                port: ivec3(0, 0, 1),
                x: frame_x,
                z: frame_z,
            }],
            "the stamped frame pair survives fusion and derives intact"
        );
    }

    fn assert_same_ordered_graph(
        actual: &StableDiGraph<BloqNode, BloqEdge>,
        expected: &StableDiGraph<BloqNode, BloqEdge>,
    ) {
        assert_eq!(actual.edge_count(), expected.edge_count());
        assert_eq!(
            actual.node_indices().collect::<Vec<_>>(),
            expected.node_indices().collect::<Vec<_>>()
        );
        for node in expected.node_indices() {
            assert_eq!(actual[node].try_classical(), expected[node].try_classical());
            for direction in [Direction::Incoming, Direction::Outgoing] {
                let edges = |graph: &StableDiGraph<BloqNode, BloqEdge>| {
                    graph
                        .edges_directed(node, direction)
                        .map(|edge| (edge.source(), edge.target(), edge.weight().clone()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(edges(actual), edges(expected), "{node:?} {direction:?}");
            }
        }
    }

    #[test]
    fn batch_edge_removal_matches_individual_removal_with_reused_slots() {
        let mut empty = StableDiGraph::default();
        retain_edges_preserving_adjacency(&mut empty, |_, _, _| true);
        assert_eq!(empty.node_count(), 0);
        for reuse_target in [None, Some(1), Some(4)] {
            let mut graph = StableDiGraph::default();
            let nodes = (0..5)
                .map(|index| graph.add_node(BloqNode::classical(ClassicalNode::observable(index))))
                .collect::<Vec<_>>();
            graph.remove_node(nodes[3]);
            let hole = graph.add_edge(nodes[4], nodes[4], BloqEdge::Order);
            // Include parallel edges, both directions, self-loops, and a node hole.
            for (source, target, edge) in [
                (0, 1, BloqEdge::Order),
                (0, 1, BloqEdge::compose(0)),
                (1, 1, BloqEdge::Order),
                (2, 0, BloqEdge::Order),
                (0, 2, BloqEdge::value(1)),
                (4, 2, BloqEdge::Order),
                (0, 1, BloqEdge::Order),
                (2, 2, BloqEdge::compose(2)),
            ] {
                graph.add_edge(nodes[source], nodes[target], edge);
            }
            if let Some(target) = reuse_target {
                graph.remove_edge(hole);
                // Target 4 has no other retained incoming edge: only its
                // source's outgoing order exposes the reused-slot mismatch.
                let reused = graph.add_edge(nodes[0], nodes[target], BloqEdge::compose(3));
                assert_eq!(reused, hole);
            }
            let orders = graph
                .edge_references()
                .filter(|edge| matches!(edge.weight(), BloqEdge::Order))
                .map(|edge| edge.id())
                .collect::<Vec<_>>();
            // Every subset includes the empty and all-Order removal cases.
            for mask in 0..(1usize << orders.len()) {
                let removed = orders
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| mask & (1usize << *bit) != 0)
                    .map(|(_, &edge)| edge)
                    .collect::<crate::FxSet<_>>();
                let mut expected = graph.clone();
                for &edge in &removed {
                    expected.remove_edge(edge);
                }
                let mut actual = graph.clone();
                retain_edges_preserving_adjacency(&mut actual, |edge, _, _| {
                    !removed.contains(&edge)
                });
                assert_same_ordered_graph(&actual, &expected);
            }
        }
    }

    #[test]
    fn optimization_preserves_signed_compose_order_during_unrelated_fusion() {
        use bloq_circuit::{Pauli, PauliMap};

        let product = |program: &Bloq, observable| {
            let mut paulis = PauliMap::empty();
            let mut phase = 0;
            for (leaf, _) in expanded_readout_inputs(program, observable, false) {
                for operator in program[leaf].try_classical().unwrap().operators() {
                    phase = (phase + paulis.product_phase(&operator.operator)) % 4;
                    paulis = &paulis ^ &operator.operator;
                }
            }
            (paulis, phase)
        };
        for compute in [false, true] {
            for reused in [false, true] {
                let mut program = Bloq::from_text(
                    "BLOQIR 1\ntemplate t0 {\ncircuit {\nR (0,0)\nM (0,0):m0\n}\n}\ngraph {\n\
                 n0 quantum {\ninstance i0 t0 @ (0,0)\n}\n\
                 n1 quantum {\ninstance i1 t0 @ (2,0)\n}\n\
                 n2 observable fragment operators i0 output X(0,0)\n\
                 n3 observable fragment operators i0 output Z(0,0)\n\
                 n4 observable fragment operators i0 output X(0,0)\n\
                 n5 observable 0\n\
                 n0 -> n2 order\nn0 -> n3 order\nn0 -> n4 order\n}",
                )
                .unwrap();
                let index = NodeIndex::new;
                let graph = program.top_mut().graph_mut();
                let hole = graph.add_edge(index(1), index(0), BloqEdge::Order);
                graph.add_edge(index(2), index(5), BloqEdge::compose(0));
                graph.add_edge(index(3), index(5), BloqEdge::compose(1));
                if reused {
                    graph.remove_edge(hole);
                }
                graph.add_edge(index(4), index(5), BloqEdge::compose(2));
                if !reused {
                    graph.remove_edge(hole);
                }
                graph.add_edge(index(0), index(1), BloqEdge::quantum(Vec::new()));
                for _ in 0..2 {
                    graph.add_edge(index(0), index(5), BloqEdge::Order);
                }
                graph.add_edge(index(1), index(5), BloqEdge::Order);
                let removed = program.add_node(BloqNode::classical(if compute {
                    ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(true),
                    }
                } else {
                    ClassicalNode::observable_fragment(
                        vec![crate::test_fixture::instance_measurement(0, 0)],
                        Vec::new(),
                    )
                }));
                let consumer = program.add_node(BloqNode::classical(if compute {
                    ClassicalNode::Compute {
                        expr: ClassicalExpr::In(0),
                    }
                } else {
                    ClassicalNode::observable(1)
                }));
                program.add_edge(
                    removed,
                    consumer,
                    if compute {
                        BloqEdge::value(0)
                    } else {
                        BloqEdge::compose(0)
                    },
                );
                if !compute {
                    program.add_edge(BloqNodeId(0), removed, BloqEdge::Order);
                }
                program.validate().unwrap();
                let observable = BloqNodeId(5);
                let expected_product: (PauliMap, u8) =
                    ([(glam::IVec2::ZERO, Pauli::Z)].into_iter().collect(), 2);
                assert_eq!(product(&program, observable), expected_product, "XZX = -Z");
                let before = program.clone();
                program.optimize().unwrap();
                assert!(program.node(removed).is_none(), "unrelated fusion ran");
                assert!(!program.incoming(observable).any(|edge| {
                    edge.source == BloqNodeId(0) && matches!(edge.edge, BloqEdge::Order)
                }));
                assert_eq!(
                    expanded_readout_inputs(&program, observable, false),
                    expanded_readout_inputs(&before, observable, false),
                );
                assert_eq!(product(&program, observable), expected_product);
                for selected in [false, true] {
                    let assignment = crate::ClassicalAssignment::Uniform(selected);
                    assert_eq!(
                        program.resolve_classical(observable, assignment),
                        before.resolve_classical(observable, assignment)
                    );
                }
                program.validate().unwrap();
            }
        }
    }

    #[test]
    fn observable_order_frontier_keeps_incomparable_and_guarded_contributors() {
        let mut program = Bloq::new();
        let a = program.add_node(BloqNode::from_members(Vec::new()));
        let b = program.add_node(BloqNode::from_members(Vec::new()));
        let c = program.add_node(BloqNode::from_members(Vec::new()));
        let observable = program.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        program.add_edge(a, b, BloqEdge::quantum(Vec::new()));
        for source in [a, a, b, c] {
            program.add_edge(source, observable, BloqEdge::Order);
        }
        program.optimize().unwrap();
        let orders = program
            .incoming(observable)
            .map(|edge| edge.source)
            .collect::<crate::FxSet<_>>();
        assert_eq!(orders, [b, c].into_iter().collect());
        assert!(program.has_path(a, observable));

        let mut guarded = Bloq::new();
        let predicate = guarded.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let a = guarded.add_node(BloqNode::from_members(Vec::new()));
        let b = guarded.add_node(BloqNode::from_members(Vec::new()));
        let mut observable_node = BloqNode::classical(ClassicalNode::observable(0));
        observable_node.activation = Some(8);
        let observable = guarded.add_node(observable_node);
        guarded.add_edge(predicate, observable, BloqEdge::value(8));
        guarded.add_edge(
            a,
            b,
            BloqEdge::Quantum(Box::new(crate::QuantumEdge {
                pipes: Vec::new(),
                guard: Some(predicate.into()),
            })),
        );
        for source in [a, b] {
            guarded.add_edge(source, observable, BloqEdge::Order);
        }
        guarded.optimize().unwrap();
        assert_eq!(
            guarded
                .incoming(observable)
                .filter(|edge| matches!(edge.edge, BloqEdge::Order))
                .count(),
            2
        );
        assert_eq!(guarded[observable].activation, Some(8));
    }

    #[test]
    fn observable_fusion_preserves_measurements_and_ordered_boundary_packets() {
        for payload in [
            "measurements i0:m0*i0:m0*i1:m1",
            "operators i0 input X(0,0), i1 input X(2,0)",
            "operators i0 input X(0,0), i0 input Z(0,0), i0 input X(0,0)",
            "measurements i0:m0*i0:m0*i1:m1 operators i0 input X(0,0), i1 input X(2,0)",
        ] {
            for activated in [false, true] {
                let when = if activated {
                    [" when v7", " when v8", " when v9", " when v10", " when v11"]
                } else {
                    [""; 5]
                };
                let guards = if activated {
                    "n6 -> n1 value 7 flip\nn6 -> n2 value 8 flip\nn6 -> n3 value 9 flip\nn6 -> n4 value 10 flip\nn6 -> n5 value 11 flip"
                } else {
                    ""
                };
                let mut program = Bloq::from_text(&format!(
                    "BLOQIR 1\ntemplate t0 {{\ncircuit {{\nR (0,0)\nM (0,0):m0\nM (0,0):m1\n}}\n}}\ngraph {{\nn0 quantum {{\ninstance i0 t0 @ (0,0)\ninstance i1 t0 @ (2,0)\n}}\nn1 observable fragment {payload}{}\nn2 observable fragment{}\nn3 observable fragment{}\nn4 observable 0{}\nn5 observable 1{}\nn6 observable 2\nn0 -> n1 order\nn1 -> n2 compose 0\nn2 -> n3 compose 0\nn3 -> n4 compose 0\nn3 -> n4 compose 1\nn3 -> n5 compose 0\n{guards}\n}}",
                    when[0], when[1], when[2], when[3], when[4],
                ))
                .unwrap();
                program.validate().unwrap();
                let before = program.clone();
                let leaf = before[BloqNodeId(1)].try_classical().unwrap();
                program.optimize().unwrap();
                if leaf.operators().is_empty() {
                    for removed in [1, 2, 3] {
                        assert!(program.node(BloqNodeId(removed)).is_none());
                    }
                }
                for mut optimized in [
                    program.clone(),
                    Bloq::from_binary(&program.to_binary()).unwrap(),
                    Bloq::from_text(&program.to_text()).unwrap(),
                ] {
                    optimized.validate().unwrap();
                    if !leaf.operators().is_empty() {
                        let shared = optimized[BloqNodeId(1)].try_classical().unwrap();
                        assert_eq!(shared.measurements(), leaf.measurements());
                        assert_eq!(shared.operators(), leaf.operators());
                    }
                    for (target, copies) in [(BloqNodeId(4), 2), (BloqNodeId(5), 1)] {
                        if leaf.operators().is_empty() {
                            assert_eq!(
                                optimized[target].try_classical().unwrap().measurements(),
                                leaf.measurements().repeat(copies)
                            );
                            assert!(expanded_readout_inputs(&optimized, target, true).is_empty());
                        } else {
                            assert_eq!(
                                expanded_readout_inputs(&optimized, target, true),
                                vec![(BloqNodeId(1), ValueRole::Data); copies]
                            );
                        }
                        assert!(optimized.has_path(BloqNodeId(0), target));
                        for selected in [false, true] {
                            let assignment = crate::ClassicalAssignment::Uniform(selected);
                            assert_eq!(
                                optimized.resolve_classical(target, assignment),
                                before.resolve_classical(target, assignment),
                            );
                            assert_eq!(
                                optimized.measurement_dependencies(target, assignment),
                                before.measurement_dependencies(target, assignment),
                            );
                        }
                    }
                    let text = optimized.to_text();
                    optimized.optimize().unwrap();
                    assert_eq!(optimized.to_text(), text, "optimization is idempotent");
                }
            }
        }
    }

    #[test]
    fn shared_observable_leaves_retain_activation_and_external_uses() {
        for extra in [
            "n4 compute in0\nn1 -> n4 value 0",
            "n4 compute 1\nn4 -> n2 value 9",
            "",
        ] {
            let mut program = Bloq::from_text(&format!(
                "BLOQIR 1\ngraph {{\nn1 observable fragment\nn2 observable 0\nn3 observable 1\nn1 -> n2 compose 0\nn1 -> n3 compose 0\n{extra}\n}}"
            ))
            .unwrap();
            if extra.contains("value 9") {
                program.node_mut(BloqNodeId(2)).unwrap().activation = Some(9);
            } else if extra.is_empty() {
                program.top_mut().set_boundary_outputs(vec![BloqNodeId(1)]);
            }
            program.validate().unwrap();
            program.optimize().unwrap();
            program.validate().unwrap();
            assert!(program.node(BloqNodeId(1)).is_some());
        }
    }

    #[test]
    fn sibling_fragments_merge_recipes_with_matching_activation_and_uses() {
        let mut program = Bloq::from_text(
            "BLOQIR 1
template t0 {
  circuit {
    R (0,0)
    M (0,0):m0
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 quantum {
    instance i1 t0 @ (2,0)
  }
  n2 observable 0
  n3 observable fragment measurements i0:m0*i0:m0 when v4
  n4 observable fragment measurements i1:m0 when v9
  n5 observable fragment operators i0 input Z(0,0) when v0
  n6 observable fragment when v0
  n7 observable fragment operators i1 input X(2,0) when v9
  n10 observable 1
  n11 observable 2
  n0 -> n3 order
  n1 -> n4 order
  n1 -> n7 order
  n0 -> n5 order
  n2 -> n3 value 4 flip
  n2 -> n4 value 9 flip
  n2 -> n5 value 0 flip
  n2 -> n6 value 0
  n2 -> n7 value 9 flip
  n3 -> n10 compose 0
  n3 -> n10 compose 1
  n3 -> n11 compose 0
  n4 -> n10 compose 2
  n4 -> n10 compose 3
  n4 -> n11 compose 1
  n5 -> n10 compose 4
  n6 -> n10 compose 5
  n7 -> n10 compose 6
  n7 -> n10 compose 7
  n7 -> n11 compose 2
}",
        )
        .unwrap();
        program.validate().unwrap();
        let before = program.clone();
        program.optimize().unwrap();
        assert!(program.node(BloqNodeId(4)).is_none());
        assert!(
            program.node(BloqNodeId(5)).is_some(),
            "different use counts"
        );
        assert!(
            program.node(BloqNodeId(6)).is_some(),
            "different output port"
        );
        let merged = program[BloqNodeId(3)].try_classical().unwrap();
        assert_eq!(merged.measurements().len(), 3, "retain duplicate terms");
        assert!(merged.operators().is_empty());
        assert_eq!(
            program[BloqNodeId(7)]
                .try_classical()
                .unwrap()
                .operators()
                .len(),
            1
        );
        assert_eq!(program[BloqNodeId(3)].activation, Some(4));
        for source in [BloqNodeId(0), BloqNodeId(1)] {
            assert!(program.has_path(source, BloqNodeId(3)));
        }
        for mut optimized in [
            program.clone(),
            Bloq::from_text(&program.to_text()).unwrap(),
            Bloq::from_binary(&program.to_binary()).unwrap(),
        ] {
            optimized.validate().unwrap();
            for active in [false, true] {
                let assignment = crate::ClassicalAssignment::Uniform(active);
                for target in [BloqNodeId(10), BloqNodeId(11)] {
                    assert_eq!(
                        optimized.resolve_classical(target, assignment),
                        before.resolve_classical(target, assignment)
                    );
                    assert_eq!(
                        optimized.measurement_dependencies(target, assignment),
                        before.measurement_dependencies(target, assignment)
                    );
                }
            }
            let text = optimized.to_text();
            optimized.optimize().unwrap();
            assert_eq!(optimized.to_text(), text, "optimization is idempotent");
        }
    }

    #[test]
    fn compute_fusion_retains_flip_port_and_recipe_occurrences() {
        let mut program = Bloq::new();
        let term = crate::InstanceMeasurement {
            instance: TemplateInstanceId(0),
            measurement: 0,
        };
        let fragment = program.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![term, term],
            Vec::new(),
        )));
        let observable = program.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        program.add_edge(fragment, observable, BloqEdge::compose(2));
        let intermediate = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let result = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        program.add_edge(observable, intermediate, BloqEdge::flip(0));
        program.add_edge(intermediate, result, BloqEdge::value(0));
        let before = program
            .resolve_classical(result, crate::ClassicalAssignment::Uniform(false))
            .unwrap();
        program.optimize().unwrap();
        assert!(program.node(fragment).is_none());
        assert_eq!(
            program[observable].try_classical().unwrap().measurements(),
            [term, term]
        );
        assert_eq!(
            program.data_inputs(result).next().unwrap().output,
            Some(crate::ObservableOutput::Flip)
        );
        assert_eq!(
            program
                .resolve_classical(result, crate::ClassicalAssignment::Uniform(false))
                .unwrap(),
            before
        );
        assert_eq!(program.incoming(observable).count(), 0);
    }
}
