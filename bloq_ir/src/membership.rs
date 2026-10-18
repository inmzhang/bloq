//! Symbolic input domains for conditional component registration.
use crate::{
    Bloq, BloqNodeId, ClassicalExpr, ClassicalNode, FxMap, NodeTemplateInstanceMergeError as Error,
    ObservableOutput, SubGraph, ValueRef,
};
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanLimits, BooleanOp, DECISION_FALSE, DECISION_TRUE, DecisionId,
};

/// Cached Boolean queries over one immutable graph level.
///
/// Shared Compute ancestry preserves correlated guards. The graph must be
/// acyclic and predicates must name existing bit producers. The borrow prevents
/// graph edits from invalidating retained predicates and query results.
#[derive(Debug)]
pub struct PredicateAnalysis<'a> {
    graph: &'a SubGraph,
    diagram: BooleanDecisionDiagram,
    memo: FxMap<ValueRef, DecisionId>,
    variables: FxMap<ValueRef, usize>,
    inputs: FxMap<BloqNodeId, FxMap<u32, ValueRef>>,
    aliases: FxMap<ValueRef, (ValueRef, bool)>,
    implications: FxMap<(Option<ValueRef>, ValueRef), bool>,
    overlaps: FxMap<(Option<ValueRef>, Option<ValueRef>), bool>,
}

/// One exhaustive, mutually exclusive member choice and its known guard inputs.
pub(crate) struct QuantumSelection {
    pub input: u32,
    pub values: FxMap<u32, bool>,
}

impl<'a> PredicateAnalysis<'a> {
    /// Recognize a component whose member guards partition the input domain.
    /// Side-table guards must be fixed by each choice; independent selections
    /// still require pinning before timing or seam edits.
    pub(crate) fn quantum_selections(
        &mut self,
        node: BloqNodeId,
    ) -> Result<Option<Vec<QuantumSelection>>, Error> {
        let quantum = self.graph[node].expect_quantum();
        let mut choices = Vec::new();
        let mut covered = DECISION_FALSE;
        for guard in &quantum.guards {
            if guard.instances.is_empty() {
                continue;
            }
            let value = self.lower(Eval::Input(node, guard.input))?;
            if value == DECISION_FALSE || choices.iter().any(|&(_, old)| old == value) {
                continue;
            }
            if self.diagram.apply(BooleanOp::And, covered, value)? != DECISION_FALSE {
                return Ok(None);
            }
            covered = self.diagram.apply(BooleanOp::Or, covered, value)?;
            choices.push((guard.input, value));
        }
        if covered != DECISION_TRUE {
            return Ok(None);
        }
        let mut selections = Vec::with_capacity(choices.len());
        for (input, choice) in choices {
            let mut values = FxMap::default();
            for guard in &quantum.guards {
                let value = self.lower(Eval::Input(node, guard.input))?;
                let value = match self.diagram.constrain(value, choice)? {
                    DECISION_FALSE => false,
                    DECISION_TRUE => true,
                    _ => return Ok(None),
                };
                values.insert(guard.input, value);
            }
            selections.push(QuantumSelection { input, values });
        }
        Ok(Some(selections))
    }

    /// Start a query cache borrowing this graph level, with finite defaults.
    pub fn new(graph: &'a SubGraph) -> Self {
        Self::with_limits(graph, BooleanLimits::default())
    }

    /// Start a query cache with explicit decision-node and cumulative-work limits.
    pub fn with_limits(graph: &'a SubGraph, limits: BooleanLimits) -> Self {
        Self {
            graph,
            diagram: BooleanDecisionDiagram::with_limits(limits),
            memo: FxMap::default(),
            variables: FxMap::default(),
            inputs: FxMap::default(),
            aliases: FxMap::default(),
            implications: FxMap::default(),
            overlaps: FxMap::default(),
        }
    }

    pub(crate) fn steps(&self) -> usize {
        self.diagram.steps()
    }

    pub(crate) fn charge(&mut self, steps: usize) -> Result<(), Error> {
        Ok(self.diagram.charge(steps)?)
    }

    /// Synchronize graph-local caches with the whole validation's work meter.
    pub(crate) fn using_work<T>(
        &mut self,
        work: &mut BooleanDecisionDiagram,
        query: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.charge(work.steps().saturating_sub(self.steps()))?;
        let value = query(self)?;
        work.charge(self.steps().saturating_sub(work.steps()))?;
        Ok(value)
    }

    /// Whether two guards can both hold; an absent guard is unconditional.
    ///
    /// # Errors
    ///
    /// Returns [`crate::lowering::NodeTemplateInstanceMergeError`] for malformed predicate
    /// dataflow or exhausted Boolean resources.
    pub fn overlap(
        &mut self,
        left: Option<BloqNodeId>,
        right: Option<BloqNodeId>,
    ) -> Result<bool, Error> {
        self.overlap_values(left.map(Into::into), right.map(Into::into))
    }

    /// Whether selected output guards overlap.
    ///
    /// # Errors
    /// Returns an error for malformed predicates or exhausted Boolean resources.
    pub fn overlap_values(
        &mut self,
        left: Option<ValueRef>,
        right: Option<ValueRef>,
    ) -> Result<bool, Error> {
        self.charge(1)?;
        let key = (left.min(right), left.max(right));
        if let Some(&result) = self.overlaps.get(&key) {
            return Ok(result);
        }
        let left = self.guard(left)?;
        let right = self.guard(right)?;
        let result = self.diagram.apply(BooleanOp::And, left, right)? != DECISION_FALSE;
        self.overlaps.insert(key, result);
        Ok(result)
    }

    fn guard(&mut self, node: Option<ValueRef>) -> Result<DecisionId, Error> {
        node.map(|node| self.lower(Eval::Node(node)))
            .transpose()
            .map(|value| value.unwrap_or(DECISION_TRUE))
    }

    #[cfg(test)]
    pub(crate) fn implies(
        &mut self,
        left: Option<BloqNodeId>,
        right: BloqNodeId,
    ) -> Result<bool, Error> {
        self.implies_values(left.map(Into::into), right.into())
    }

    pub(crate) fn implies_values(
        &mut self,
        left: Option<ValueRef>,
        right: ValueRef,
    ) -> Result<bool, Error> {
        self.charge(1)?;
        if let Some(&result) = self.implications.get(&(left, right)) {
            return Ok(result);
        }
        if self.proves_conjunct(left.as_slice(), right)? {
            self.implications.insert((left, right), true);
            return Ok(true);
        }
        let source = self.guard(left)?;
        let target = self.guard(Some(right))?;
        let absent = self.diagram.negate(target)?;
        let result = self.diagram.apply(BooleanOp::And, source, absent)? == DECISION_FALSE;
        self.implications.insert((left, right), result);
        Ok(result)
    }

    pub(crate) fn implies_all_values(
        &mut self,
        left: &[ValueRef],
        right: ValueRef,
    ) -> Result<bool, Error> {
        if left.len() < 2 {
            return self.implies_values(left.first().copied(), right);
        }
        self.charge(left.len())?;
        if self.proves_conjunct(left, right)? {
            return Ok(true);
        }
        let mut source = DECISION_TRUE;
        for &node in left {
            let value = self.guard(Some(node))?;
            source = self.diagram.apply(BooleanOp::And, source, value)?;
        }
        let target = self.guard(Some(right))?;
        let absent = self.diagram.negate(target)?;
        Ok(self.diagram.apply(BooleanOp::And, source, absent)? == DECISION_FALSE)
    }

    // Return the representative's truth when this value is true; compress shared paths.
    fn normalize_alias(&mut self, mut value: ValueRef) -> Result<(ValueRef, bool), Error> {
        let mut truth = true;
        let mut path = FxMap::default();
        let graph = self.graph;
        'aliases: loop {
            self.charge(1)?;
            if let Some(&(canonical, positive)) = self.aliases.get(&value) {
                truth = truth == positive;
                value = canonical;
                break;
            }
            if path.insert(value, truth).is_some() {
                return Err(Error::InvalidMembership("cyclic predicate ancestry".into()));
            }
            let node = graph
                .node(value.node)
                .ok_or_else(|| Error::InvalidMembership(format!("unknown predicate {value:?}")))?;
            if node.activation.is_some() || value.output != ObservableOutput::Corrected {
                break;
            }
            let Some(ClassicalNode::Compute { expr }) = node.try_classical() else {
                break;
            };
            let mut expr = expr;
            let mut inverted = false;
            let slot = loop {
                self.charge(1)?;
                match expr {
                    ClassicalExpr::In(slot) => break *slot,
                    ClassicalExpr::Not(inner) => {
                        inverted = !inverted;
                        expr = inner;
                    }
                    ClassicalExpr::Parity { inputs, constant } if inputs.len() == 1 => {
                        inverted ^= *constant;
                        break inputs[0];
                    }
                    _ => break 'aliases,
                }
            };
            truth ^= inverted;
            value = self.input(value.node, slot)?;
        }
        self.charge(path.len())?;
        self.aliases.extend(
            path.into_iter()
                .map(|(alias, positive)| (alias, (value, truth == positive))),
        );
        Ok((value, truth))
    }

    // Each part records a truth forced by the source conjunction. This proves
    // Boolean implication; full validation still checks every operand's dataflow.
    fn proves_conjunct(&mut self, roots: &[ValueRef], target: ValueRef) -> Result<bool, Error> {
        enum Part<'a> {
            Value(ValueRef, bool),
            Expr(BloqNodeId, &'a ClassicalExpr, bool),
        }
        if roots.is_empty() {
            return Ok(false);
        }
        let (target, target_truth) = self.normalize_alias(target)?;
        self.charge(roots.len())?;
        let mut pending = roots
            .iter()
            .rev()
            .copied()
            .map(|value| Part::Value(value, true))
            .collect::<Vec<_>>();
        let mut seen = crate::FxSet::default();
        let graph = self.graph;
        while let Some(part) = pending.pop() {
            self.charge(1)?;
            match part {
                Part::Value(value, truth) => {
                    let (value, positive) = self.normalize_alias(value)?;
                    let truth = truth == positive;
                    let node = graph.node(value.node).ok_or_else(|| {
                        Error::InvalidMembership(format!("unknown predicate {value:?}"))
                    })?;
                    if value == target && truth == target_truth {
                        return Ok(true);
                    }
                    if !seen.insert((value, truth)) {
                        continue;
                    }
                    // A false activated value need not have a false payload.
                    if node.activation.is_some() && !truth {
                        continue;
                    }
                    if value.output == ObservableOutput::Corrected
                        && let Some(ClassicalNode::Compute { expr }) = node.try_classical()
                    {
                        pending.push(Part::Expr(value.node, expr, truth));
                    }
                    if let Some(slot) = node.activation {
                        pending.push(Part::Value(self.input(value.node, slot)?, true));
                    }
                }
                Part::Expr(node, ClassicalExpr::In(slot), truth) => {
                    pending.push(Part::Value(self.input(node, *slot)?, truth));
                }
                Part::Expr(node, ClassicalExpr::Not(inner), truth) => {
                    pending.push(Part::Expr(node, inner, !truth));
                }
                Part::Expr(node, ClassicalExpr::Parity { inputs, constant }, truth)
                    if inputs.len() == 1 =>
                {
                    pending.push(Part::Value(self.input(node, inputs[0])?, truth ^ *constant));
                }
                Part::Expr(node, ClassicalExpr::And(operands), true) => {
                    self.charge(operands.len())?;
                    pending.extend(
                        operands
                            .iter()
                            .rev()
                            .map(|expr| Part::Expr(node, expr, true)),
                    );
                }
                Part::Expr(node, ClassicalExpr::Or(operands), false) => {
                    self.charge(operands.len())?;
                    pending.extend(
                        operands
                            .iter()
                            .rev()
                            .map(|expr| Part::Expr(node, expr, false)),
                    );
                }
                Part::Expr(node, ClassicalExpr::Select(operands), truth) => {
                    self.charge(2)?;
                    let [condition, when_false, when_true] = operands.as_ref();
                    if matches!(when_true, ClassicalExpr::Const(bit) if *bit != truth) {
                        pending.push(Part::Expr(node, when_false, truth));
                        pending.push(Part::Expr(node, condition, false));
                    } else if matches!(when_false, ClassicalExpr::Const(bit) if *bit != truth) {
                        pending.push(Part::Expr(node, when_true, truth));
                        pending.push(Part::Expr(node, condition, true));
                    }
                }
                Part::Expr(_, ClassicalExpr::Const(bit), truth) if *bit != truth => {
                    return Ok(true);
                }
                Part::Expr(_, _, _) => {}
            }
        }
        Ok(false)
    }

    fn input(&mut self, node: BloqNodeId, slot: u32) -> Result<ValueRef, Error> {
        if !self.inputs.contains_key(&node) {
            let mut inputs = FxMap::default();
            for input in self.graph.value_inputs(node) {
                self.diagram.charge(1)?;
                inputs.insert(input.slot, input.value_ref().expect("runtime input"));
            }
            self.inputs.insert(node, inputs);
        }
        self.inputs[&node]
            .get(&slot)
            .copied()
            .ok_or(Error::MissingMembershipInput(slot))
    }

    // Both expression nesting and shared Compute ancestry use one explicit
    // stack. A long predicate chain cannot consume the native call stack.
    fn lower(&mut self, first: Eval<'_>) -> Result<DecisionId, Error> {
        self.charge(1)?;
        let mut pending = vec![first];
        let mut values = Vec::new();
        let mut visiting = crate::FxSet::default();
        while let Some(next) = pending.pop() {
            self.charge(1)?;
            match next {
                Eval::Node(node) => {
                    if let Some(&value) = self.memo.get(&node) {
                        values.push(value);
                        continue;
                    }
                    if !visiting.insert(node) {
                        return Err(Error::InvalidMembership("cyclic predicate ancestry".into()));
                    }
                    let weight = self.graph.node(node.node).ok_or_else(|| {
                        Error::InvalidMembership(format!("unknown predicate {node:?}"))
                    })?;
                    pending.push(Eval::Store(node));
                    if let Some(slot) = weight.activation {
                        pending.push(Eval::Apply(BooleanOp::And, 2, DECISION_TRUE));
                        pending.push(Eval::Input(node.node, slot));
                    }
                    if node.output == ObservableOutput::Corrected
                        && let Some(ClassicalNode::Compute { expr }) = weight.try_classical()
                    {
                        pending.push(Eval::Expr(node.node, expr));
                    } else {
                        // Keep related predicate inputs together. Registry node ids
                        // follow physical emission and can separate correlated bits,
                        // making a compact predicate exponential in that ordering.
                        let next = self.variables.len();
                        let variable = *self.variables.entry(node).or_insert(next);
                        values.push(self.diagram.make_node(
                            variable,
                            DECISION_FALSE,
                            DECISION_TRUE,
                        )?);
                    }
                }
                Eval::Store(node) => {
                    let value = *values.last().expect("evaluated node");
                    self.memo.insert(node, value);
                    visiting.remove(&node);
                }
                Eval::Input(node, slot) => {
                    let producer = self.input(node, slot)?;
                    pending.push(Eval::Node(producer));
                }
                Eval::Expr(node, expr) => match expr {
                    ClassicalExpr::Const(value) => {
                        values.push(if *value {
                            DECISION_TRUE
                        } else {
                            DECISION_FALSE
                        });
                    }
                    ClassicalExpr::In(slot) => pending.push(Eval::Input(node, *slot)),
                    ClassicalExpr::Not(inner) => {
                        pending.push(Eval::Negate);
                        pending.push(Eval::Expr(node, inner));
                    }
                    ClassicalExpr::Parity { inputs, constant } => {
                        self.charge(inputs.len())?;
                        pending.push(Eval::Apply(
                            BooleanOp::Xor,
                            inputs.len(),
                            if *constant {
                                DECISION_TRUE
                            } else {
                                DECISION_FALSE
                            },
                        ));
                        pending.extend(inputs.iter().rev().map(|&slot| Eval::Input(node, slot)));
                    }
                    ClassicalExpr::Xor(operands)
                    | ClassicalExpr::And(operands)
                    | ClassicalExpr::Or(operands) => {
                        self.charge(operands.len())?;
                        let op = match expr {
                            ClassicalExpr::Xor(..) => BooleanOp::Xor,
                            ClassicalExpr::And(..) => BooleanOp::And,
                            _ => BooleanOp::Or,
                        };
                        pending.push(Eval::Apply(
                            op,
                            operands.len(),
                            if op == BooleanOp::And {
                                DECISION_TRUE
                            } else {
                                DECISION_FALSE
                            },
                        ));
                        pending.extend(operands.iter().rev().map(|expr| Eval::Expr(node, expr)));
                    }
                    ClassicalExpr::Select(operands) => {
                        pending.push(Eval::Select);
                        pending.extend(operands.iter().rev().map(|expr| Eval::Expr(node, expr)));
                    }
                },
                Eval::Apply(op, count, mut value) => {
                    for _ in 0..count {
                        value = self
                            .diagram
                            .apply(op, value, values.pop().expect("operand"))?;
                    }
                    values.push(value);
                }
                Eval::Negate => {
                    let value = values.pop().expect("negation operand");
                    values.push(self.diagram.negate(value)?);
                }
                Eval::Select => {
                    let high = values.pop().expect("true operand");
                    let low = values.pop().expect("false operand");
                    let condition = values.pop().expect("selection predicate");
                    let absent = self.diagram.negate(condition)?;
                    let low = self.diagram.apply(BooleanOp::And, absent, low)?;
                    let high = self.diagram.apply(BooleanOp::And, condition, high)?;
                    values.push(self.diagram.apply(BooleanOp::Or, low, high)?);
                }
            }
        }
        Ok(values.pop().expect("evaluated predicate"))
    }
}

impl Bloq {
    /// Terminal choices share fixed input seams, so their waits can be composed
    /// before selection. Changing seams or continuing membership still needs a
    /// pinned path. Activated quantum regions also retain that requirement.
    pub(crate) fn timing_requires_pins(&self) -> Result<bool, Error> {
        fn guarded(level: &SubGraph, under_activation: bool) -> Result<bool, Error> {
            let mut predicates = PredicateAnalysis::new(level);
            for (id, node) in level.nodes() {
                if let Some(quantum) = node.try_quantum() {
                    if under_activation {
                        return Ok(true);
                    }
                    if !quantum.guards.is_empty()
                        && (level
                            .outgoing(id)
                            .any(|edge| matches!(edge.edge, crate::BloqEdge::Quantum(_)))
                            || predicates.quantum_selections(id)?.is_none())
                    {
                        return Ok(true);
                    }
                }
                if let Some(region) = node.try_region() {
                    for (_, body) in region.bodies() {
                        if guarded(body, under_activation || node.activation.is_some())? {
                            return Ok(true);
                        }
                    }
                }
            }
            Ok(level.edges().any(|edge| {
                matches!(edge.edge, crate::BloqEdge::Quantum(quantum) if quantum.guard.is_some())
            }))
        }
        guarded(self.top(), false)
    }
}

enum Eval<'a> {
    Node(ValueRef),
    Store(ValueRef),
    Input(BloqNodeId, u32),
    Expr(BloqNodeId, &'a ClassicalExpr),
    Apply(BooleanOp, usize, DecisionId),
    Negate,
    Select,
}

/// SEM-MERGE legality is pairwise: aligned tick/repeat shapes, a unique partner
/// for every reused body, and compatible operations on shared qubits. Checking
/// every coexisting pair therefore certifies every selected set without listing
/// its Boolean assignments. Record availability is checked separately by WF-7.
pub(crate) fn validate_merge(
    graph: &SubGraph,
    id: BloqNodeId,
    templates: &crate::BloqTemplatePool,
    options: crate::lowering::InstantiationOptions<'_>,
    predicates: &mut PredicateAnalysis<'_>,
) -> Result<(), Error> {
    let node = &graph[id];
    let quantum = node.expect_quantum();
    predicates.charge(
        quantum
            .instances
            .len()
            .saturating_add(quantum.guards.len())
            .saturating_add(quantum.detector_bundles.len()),
    )?;
    for guard in &quantum.guards {
        predicates.charge(
            guard
                .instances
                .len()
                .saturating_add(guard.detectors.len())
                .saturating_add(guard.detector_bundles.len())
                .saturating_add(guard.restarts.len())
                .saturating_add(guard.detector_parities.len())
                .saturating_add(guard.restart_parities.len()),
        )?;
    }
    for parity in quantum.stored_parities() {
        predicates.charge(parity.terms().len().saturating_add(1))?;
    }
    // Validate registration indices and uniqueness even for unreachable guards.
    node.select_quantum_members(|_| Some(false))?;
    let mut guards = FxMap::default();
    for guard in &quantum.guards {
        let producer = predicates.input(id, guard.input)?;
        let value = predicates.guard(Some(producer))?;
        for &instance in &guard.instances {
            guards.insert(instance, value);
        }
    }
    let mut pair = crate::BloqNode::from_members(Vec::new());
    pair.provenance = node.provenance.clone();
    let mut costs = FxMap::default();
    let mut singleton = crate::FxSet::default();
    let mut groups: FxMap<crate::TemplateId, (DecisionId, DecisionId)> = FxMap::default();
    let mut shapes = crate::BloqTemplatePool::new();
    let mut shape_ids = FxMap::default();
    for instance in &quantum.instances {
        let template = templates
            .get(instance.template_id)
            .ok_or(Error::UnknownTemplate(instance.template_id))?;
        if let std::collections::hash_map::Entry::Vacant(entry) = costs.entry(instance.template_id)
        {
            let cost = crate::instantiation::circuit_work(&template.circuit);
            predicates.charge(cost)?;
            entry.insert(cost);
            shape_ids.insert(
                instance.template_id,
                shapes.insert(crate::BloqTemplate::new(circuit_shape(&template.circuit))),
            );
        }
        // Translation is the only instance-specific singleton check. Reuse the
        // source footprint and materialize each template/provenance class once.
        predicates.charge(template.qubits().len())?;
        for &qubit in template.qubits() {
            bloq_circuit::checked_translate_coordinate(qubit, instance.offset)?;
        }
        if singleton.insert((
            instance.template_id,
            instance.provenance.is_ideal_boundary(),
        )) {
            predicates.charge(costs[&instance.template_id])?;
            pair.expect_quantum_mut().instances = vec![*instance];
            pair.emission_plan_with_options(templates, options)?;
        }
        let value = guards.get(&instance.id).copied().unwrap_or(DECISION_TRUE);
        // Keep the domains with at least one/two selected copies. The latter
        // needs the pair merge's structural limits even at disjoint placements.
        let (one, two) = groups
            .entry(instance.template_id)
            .or_insert((DECISION_FALSE, DECISION_FALSE));
        let another = predicates.diagram.apply(BooleanOp::And, *one, value)?;
        *two = predicates.diagram.apply(BooleanOp::Or, *two, another)?;
        *one = predicates.diagram.apply(BooleanOp::Or, *one, value)?;
    }
    let groups: Vec<_> = groups
        .into_iter()
        .filter(|&(_, (guard, _))| guard != DECISION_FALSE)
        .collect();
    // Coexisting templates must align even when all their placements are
    // disjoint. Strip coordinates, then use the same merge for shape legality.
    for (index, &(left, (left_guard, multiple))) in groups.iter().enumerate() {
        for &(right, (right_guard, _)) in groups.iter().take(index + 1) {
            predicates.charge(1)?;
            let coexisting = if left == right {
                multiple
            } else {
                predicates
                    .diagram
                    .apply(BooleanOp::And, left_guard, right_guard)?
            };
            if coexisting == DECISION_FALSE {
                continue;
            }
            predicates.charge(costs[&left].saturating_add(costs[&right]))?;
            pair.expect_quantum_mut().instances = vec![
                crate::TemplateInstance::new(
                    crate::TemplateInstanceId(0),
                    shape_ids[&left],
                    glam::IVec2::ZERO,
                ),
                crate::TemplateInstance::new(
                    crate::TemplateInstanceId(1),
                    shape_ids[&right],
                    glam::IVec2::ZERO,
                ),
            ];
            pair.emission_plan(&shapes)?;
        }
    }

    let mut occupants: FxMap<glam::IVec2, Vec<usize>> = FxMap::default();
    let mut candidate_pairs = crate::FxSet::default();
    let mut checked = crate::FxSet::default();
    for (index, left) in quantum.instances.iter().enumerate() {
        let left_guard = guards.get(&left.id).copied().unwrap_or(DECISION_TRUE);
        if left_guard == DECISION_FALSE {
            continue;
        }
        let footprint = templates[left.template_id].qubits();
        predicates.charge(footprint.len())?;
        for &qubit in footprint {
            let qubit = bloq_circuit::checked_translate_coordinate(qubit, left.offset)?;
            let others = occupants.entry(qubit).or_default();
            for &other in others.iter() {
                predicates.charge(1)?;
                if !candidate_pairs.insert((index, other)) {
                    continue;
                }
                let right = &quantum.instances[other];
                let right_guard = guards.get(&right.id).copied().unwrap_or(DECISION_TRUE);
                if predicates
                    .diagram
                    .apply(BooleanOp::And, left_guard, right_guard)?
                    == DECISION_FALSE
                {
                    continue;
                }
                let offset = [
                    i64::from(right.offset.x) - i64::from(left.offset.x),
                    i64::from(right.offset.y) - i64::from(left.offset.y),
                ];
                if !checked.insert((
                    left.template_id,
                    right.template_id,
                    offset,
                    left.provenance.is_ideal_boundary(),
                    right.provenance.is_ideal_boundary(),
                )) {
                    continue;
                }
                predicates
                    .charge(costs[&left.template_id].saturating_add(costs[&right.template_id]))?;
                pair.expect_quantum_mut().instances = vec![*left, *right];
                pair.emission_plan_with_options(templates, options)?;
            }
            others.push(index);
        }
    }
    Ok(())
}

/// Keep exactly the tick/repeat structure, including nonempty plain segments.
/// The regular merge remains the sole authority for aligned-body legality.
fn circuit_shape(source: &bloq_circuit::CoordCircuit) -> bloq_circuit::CoordCircuit {
    use bloq_circuit::{BodyId, CircuitBody, CoordCircuit, GateType, Op};
    let mut shape = CoordCircuit::new();
    for index in 0..source.body_count() {
        if index != 0 {
            shape.add_body(CircuitBody::new());
        }
        let id = BodyId(index as u32);
        let ops = shape.body_mut(id).expect("allocated shape body").ops_mut();
        for op in source.body(id).expect("stored source body").ops() {
            match op {
                Op::Tick | Op::Repeat { .. } => ops.push(op.clone()),
                _ if !matches!(ops.last(), Some(Op::Gate { .. })) => {
                    ops.push(Op::Gate {
                        gate: GateType::H,
                        qubits: Vec::new(),
                    });
                }
                _ => {}
            }
        }
    }
    shape
        .set_entry_body(source.entry_body())
        .expect("stored source entry");
    shape
}

impl PredicateAnalysis<'_> {
    /// Enumerate at most `limit` distinct, jointly reachable predicate tuples.
    /// Shared Boolean ancestry preserves equal and complementary inputs without
    /// enumerating all assignments to the underlying bit producers.
    ///
    /// # Errors
    ///
    /// Returns malformed predicate dataflow or exhausted Boolean resources.
    pub fn values_up_to(
        &mut self,
        predicates: &[BloqNodeId],
        limit: usize,
    ) -> Result<Vec<Vec<bool>>, Error> {
        self.charge(predicates.len())?;
        let roots = predicates
            .iter()
            .map(|&node| self.guard(Some(node.into())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.diagram.values_up_to(&roots, limit)?)
    }

    /// Whether a joint assignment is consistent with the Boolean dataflow.
    ///
    /// # Errors
    ///
    /// Returns [`crate::lowering::NodeTemplateInstanceMergeError`] for malformed predicate
    /// dataflow or exhausted Boolean resources.
    pub fn assignment_reachable(
        &mut self,
        predicates: &[(BloqNodeId, bool)],
    ) -> Result<bool, Error> {
        Ok(self.assignment_root(predicates)? != DECISION_FALSE)
    }

    /// Additional predicate values fixed by one reachable assignment.
    /// Unconditional Boolean constants remain for the caller's strict fold:
    /// an unrelated `x ^ x` must still wait on `x` (SEM-EXPR).
    pub(crate) fn values_fixed_by_assignment(
        &mut self,
        predicates: &[(ValueRef, bool)],
        nodes: impl IntoIterator<Item = ValueRef>,
    ) -> Result<Option<FxMap<ValueRef, bool>>, Error> {
        let domain = self.assignment_value_root(predicates)?;
        if domain == DECISION_FALSE {
            return Ok(None);
        }
        let mut known = FxMap::default();
        if domain == DECISION_TRUE {
            return Ok(Some(known));
        }
        let mut fixed = FxMap::default();
        for node in nodes {
            let value = self.guard(Some(node))?;
            if value == DECISION_FALSE || value == DECISION_TRUE {
                continue;
            }
            let bit = if let Some(&bit) = fixed.get(&value) {
                bit
            } else {
                let bit = match self.diagram.constrain(value, domain)? {
                    DECISION_FALSE => Some(false),
                    DECISION_TRUE => Some(true),
                    _ => None,
                };
                fixed.insert(value, bit);
                bit
            };
            if let Some(bit) = bit {
                known.insert(node, bit);
            }
        }
        Ok(Some(known))
    }

    fn assignment_root(&mut self, predicates: &[(BloqNodeId, bool)]) -> Result<DecisionId, Error> {
        self.assignment_value_root(
            &predicates
                .iter()
                .map(|&(node, value)| (node.into(), value))
                .collect::<Vec<_>>(),
        )
    }

    fn assignment_value_root(
        &mut self,
        predicates: &[(ValueRef, bool)],
    ) -> Result<DecisionId, Error> {
        self.charge(predicates.len().saturating_add(1))?;
        let mut root = DECISION_TRUE;
        for &(predicate, value) in predicates {
            let mut condition = self.guard(Some(predicate))?;
            if !value {
                condition = self.diagram.negate(condition)?;
            }
            root = self.diagram.apply(BooleanOp::And, root, condition)?;
            if root == DECISION_FALSE {
                return Ok(root);
            }
        }
        Ok(root)
    }

    /// Whether level-local bit predicates can all be true.
    ///
    /// Private because [`assignment_reachable`](Self::assignment_reachable) is
    /// the public generalization (it takes a value per predicate); this is the
    /// all-true special case, kept for its memoized one- and two-predicate
    /// path through [`overlap`](Self::overlap).
    fn can_coexist(&mut self, predicates: &[BloqNodeId]) -> Result<bool, Error> {
        self.can_coexist_values(
            &predicates
                .iter()
                .copied()
                .map(Into::into)
                .collect::<Vec<_>>(),
        )
    }

    fn can_coexist_values(&mut self, predicates: &[ValueRef]) -> Result<bool, Error> {
        if predicates.len() <= 2 {
            return self.overlap_values(predicates.first().copied(), predicates.get(1).copied());
        }
        self.charge(predicates.len())?;
        let mut root = DECISION_TRUE;
        for &predicate in predicates {
            let value = self.guard(Some(predicate))?;
            root = self.diagram.apply(BooleanOp::And, root, value)?;
            if root == DECISION_FALSE {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl SubGraph {
    /// Whether selected level-local Boolean outputs can all be true. Shared Boolean
    /// ancestry retains equal and complementary controls. The graph must be
    /// acyclic, and every predicate must name an existing bit producer.
    /// Use [`PredicateAnalysis::assignment_reachable`] to reuse derived
    /// predicates across queries, or to ask for values other than all-true.
    ///
    /// # Errors
    ///
    /// Returns [`crate::lowering::NodeTemplateInstanceMergeError`] for malformed predicate
    /// dataflow or exhausted Boolean resources.
    pub fn predicates_can_coexist_values(&self, predicates: &[ValueRef]) -> Result<bool, Error> {
        PredicateAnalysis::new(self).can_coexist_values(predicates)
    }

    /// Whether default corrected outputs can all be true.
    ///
    /// # Errors
    /// Returns an error for malformed predicates or exhausted Boolean resources.
    pub fn predicates_can_coexist(&self, predicates: &[BloqNodeId]) -> Result<bool, Error> {
        PredicateAnalysis::new(self).can_coexist(predicates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BloqEdge, BloqNode, BloqTemplate, BloqValidationError, QuantumGuard, TemplateInstance,
        TemplateInstanceId,
    };
    use bloq_circuit::{CoordCircuit, GateType};
    use glam::ivec2;

    fn input_node(graph: &mut SubGraph) -> BloqNodeId {
        graph.add_node(BloqNode::classical(ClassicalNode::fragment()))
    }

    fn compute_node(
        graph: &mut SubGraph,
        expr: ClassicalExpr,
        inputs: &[BloqNodeId],
    ) -> BloqNodeId {
        let mut slots = std::collections::BTreeSet::new();
        expr.for_each_input(&mut |slot| {
            slots.insert(slot);
        });
        let node = graph.add_node(BloqNode::classical(ClassicalNode::Compute { expr }));
        for slot in slots {
            graph.add_edge(inputs[slot as usize], node, BloqEdge::value(slot));
        }
        node
    }

    #[test]
    fn conjunct_implication_proves_aliases_and_activation_without_boolean_nodes() {
        let mut graph = SubGraph::new();
        let a = input_node(&mut graph);
        let b = input_node(&mut graph);
        let alias = compute_node(&mut graph, ClassicalExpr::In(0), &[a]);
        let both = compute_node(
            &mut graph,
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            &[alias, b],
        );
        let mut activated = BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(1),
        });
        activated.activation = Some(0);
        let activated = graph.add_node(activated);
        graph.add_edge(alias, activated, BloqEdge::value(0));
        graph.add_edge(b, activated, BloqEdge::value(1));
        let mut analysis = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 0,
                max_steps: 1_000,
            },
        );
        for (source, target) in [(a, a), (both, alias), (both, a), (both, b), (activated, a)] {
            assert!(analysis.implies(Some(source), target).unwrap());
        }
        assert!(
            analysis
                .implies_all_values(&[b.into(), both.into()], a.into())
                .unwrap()
        );
    }

    #[test]
    fn conjunct_implication_agrees_with_all_input_assignments() {
        let mut graph = SubGraph::new();
        let a = input_node(&mut graph);
        let b = input_node(&mut graph);
        let c = input_node(&mut graph);
        let alias = compute_node(&mut graph, ClassicalExpr::In(0), &[a]);
        let formulas = [
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
                ClassicalExpr::In(2),
            ])),
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Const(false),
                ClassicalExpr::In(1),
            ])),
            ClassicalExpr::And(Box::new([
                ClassicalExpr::Or(Box::new([ClassicalExpr::In(1), ClassicalExpr::In(2)])),
                ClassicalExpr::In(0),
            ])),
            ClassicalExpr::Not(Box::new(ClassicalExpr::Or(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
            ])))),
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
                ClassicalExpr::Const(false),
            ])),
            ClassicalExpr::Not(Box::new(ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Const(true),
                ClassicalExpr::In(1),
            ])))),
            ClassicalExpr::Const(false),
            ClassicalExpr::Const(true),
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Const(false),
                ClassicalExpr::Const(false),
            ])),
        ];
        let mut expressions = formulas
            .into_iter()
            .map(|expr| compute_node(&mut graph, expr, &[alias, b, c]))
            .collect::<Vec<_>>();
        let not_a = compute_node(&mut graph, ClassicalExpr::parity([0], true), &[alias]);
        let mut activated = BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(1),
        });
        activated.activation = Some(0);
        let activated = graph.add_node(activated);
        graph.add_edge(a, activated, BloqEdge::value(0));
        graph.add_edge(b, activated, BloqEdge::value(1));
        let negative_activated = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
            &[activated],
        );
        expressions.extend([activated, negative_activated]);
        let order = graph.deterministic_emit_order().unwrap();
        for &source in &expressions {
            for target in [a, alias, b, c, not_a] {
                let mut analysis = PredicateAnalysis::new(&graph);
                let actual = analysis.implies(Some(source), target).unwrap();
                let mut exhaustive = true;
                for bits in 0..8 {
                    let mut values = FxMap::default();
                    for &id in &order {
                        let mut value =
                            if let Some(index) = [a, b, c].iter().position(|&raw| raw == id) {
                                bits & (1 << index) != 0
                            } else {
                                let ClassicalNode::Compute { expr } =
                                    graph[id].try_classical().unwrap()
                                else {
                                    unreachable!("only raw inputs and Compute nodes")
                                };
                                expr.eval(&mut |slot| {
                                    let input =
                                        graph.value_inputs(id).find(|input| input.slot == slot)?;
                                    values.get(&input.producer).copied()
                                })
                                .unwrap()
                            };
                        if let Some(slot) = graph[id].activation {
                            let input = graph
                                .value_inputs(id)
                                .find(|input| input.slot == slot)
                                .unwrap();
                            value &= values[&input.producer];
                        }
                        values.insert(id, value);
                    }
                    exhaustive &= !values[&source] || values[&target];
                }
                assert_eq!(
                    actual, exhaustive,
                    "finite valuations: {source:?} implies {target:?}"
                );
            }
        }
        assert!(
            !PredicateAnalysis::new(&graph)
                .implies(Some(negative_activated), not_a)
                .unwrap()
        );
        // These unproved OR/NOT/Select cases retain the exact BDD fallback.
        for &source in &expressions[1..4] {
            let mut limited = PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 0,
                    max_steps: 1_000,
                },
            );
            assert!(
                matches!(limited.implies(Some(source), a), Err(Error::BooleanResource(error))
                if error.resource == "Boolean nodes")
            );
        }
    }

    #[test]
    fn conjunct_implication_charges_deep_shared_dataflow_work() {
        let mut graph = SubGraph::new();
        let raw = input_node(&mut graph);
        let mut tail = raw;
        for _ in 0..12_000 {
            tail = compute_node(&mut graph, ClassicalExpr::In(0), &[tail]);
        }
        for (source, target) in [(tail, raw), (raw, tail)] {
            let mut limited = PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 0,
                    max_steps: 100,
                },
            );
            assert!(
                matches!(limited.implies(Some(source), target), Err(Error::BooleanResource(error))
                if error.resource == "Boolean work steps")
            );
        }
        let mut enough = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 0,
                max_steps: 100_000,
            },
        );
        assert!(enough.implies(Some(tail), raw).unwrap());
        assert!(enough.implies(Some(raw), tail).unwrap());
    }

    #[test]
    fn conjunct_implication_reuses_shared_signed_aliases_in_both_directions() {
        let mut graph = SubGraph::new();
        let raw = input_node(&mut graph);
        let mut aliases = Vec::new();
        let mut tail = raw;
        for _ in 0..512 {
            tail = compute_node(&mut graph, ClassicalExpr::parity([0], true), &[tail]);
            aliases.push(tail);
        }
        let mut analysis = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 0,
                max_steps: 10_000,
            },
        );
        for (index, &alias) in aliases.iter().enumerate().rev() {
            let equivalent = if index % 2 == 0 { aliases[0] } else { raw };
            assert!(analysis.implies(Some(alias), equivalent).unwrap());
            assert!(analysis.implies(Some(equivalent), alias).unwrap());
        }
    }

    #[test]
    fn conjunct_implication_proves_signed_aliases_and_constant_select_arms() {
        let mut graph = SubGraph::new();
        let a = input_node(&mut graph);
        let b = input_node(&mut graph);
        let alias = compute_node(&mut graph, ClassicalExpr::In(0), &[a]);
        let not_a = compute_node(&mut graph, ClassicalExpr::parity([0], true), &[alias]);
        let not_not_a = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
            &[not_a],
        );
        let low = compute_node(
            &mut graph,
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
                ClassicalExpr::Const(false),
            ])),
            &[a, b],
        );
        let high = compute_node(
            &mut graph,
            ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Const(false),
                ClassicalExpr::In(1),
            ])),
            &[a, b],
        );
        let neither = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::Or(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
            ])))),
            &[a, b],
        );
        let negative_low = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::Select(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::Const(true),
                ClassicalExpr::In(1),
            ])))),
            &[a, b],
        );
        let opaque = compute_node(
            &mut graph,
            ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            &[a, b],
        );
        let contains_opaque = compute_node(
            &mut graph,
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            &[opaque, a],
        );
        let opaque_not = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::Or(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1),
            ])))),
            &[a, b],
        );
        let negated_opaque = compute_node(
            &mut graph,
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
            &[opaque_not],
        );
        assert!(
            !PredicateAnalysis::new(&graph)
                .implies(Some(negated_opaque), opaque_not)
                .unwrap()
        );
        let mut analysis = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 0,
                max_steps: 2_000,
            },
        );
        for (source, target) in [
            (low, not_a),
            (low, b),
            (high, a),
            (high, b),
            (neither, not_a),
            (negative_low, a),
            (not_not_a, a),
            (a, not_not_a),
            (contains_opaque, opaque),
            (opaque_not, opaque_not),
        ] {
            assert!(
                analysis.implies(Some(source), target).unwrap(),
                "{source:?} implies {target:?}"
            );
        }
        assert!(
            analysis
                .implies_all_values(&[b.into(), low.into()], not_a.into())
                .unwrap()
        );
    }

    #[test]
    fn conjunct_implication_keeps_corrected_and_flip_outputs_distinct() {
        let mut graph = SubGraph::new();
        let node = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: Vec::new(),
            operators: Vec::new(),
        }));
        let corrected = ValueRef::from(node);
        let flip = ValueRef {
            node,
            output: ObservableOutput::Flip,
        };
        let mut analysis = PredicateAnalysis::new(&graph);
        assert!(!analysis.implies_values(Some(corrected), flip).unwrap());
        assert!(!analysis.implies_values(Some(flip), corrected).unwrap());
        assert!(analysis.implies_values(Some(flip), flip).unwrap());
    }

    #[test]
    fn conjunct_implication_keeps_full_validation_and_conditional_record_checks() {
        let mut bloq = Bloq::new();
        let a = input_node(bloq.top_mut());
        let b = input_node(bloq.top_mut());
        let condition = compute_node(
            bloq.top_mut(),
            ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            &[a, b],
        );
        let mut circuit = CoordCircuit::new();
        circuit.measure(bloq_circuit::PauliBasis::Z, [ivec2(0, 0)]);
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut owner = BloqNode::from_members(Vec::new());
        owner
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        owner.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(0)],
            ..Default::default()
        });
        let owner = bloq.add_node(owner);
        bloq.add_edge(a, owner, BloqEdge::value(0));
        let mut readout = BloqNode::classical(ClassicalNode::observable_fragment(
            vec![crate::test_fixture::instance_measurement(0, 0)],
            Vec::new(),
        ));
        readout.activation = Some(0);
        let readout = bloq.add_node(readout);
        bloq.add_edge(condition, readout, BloqEdge::value(0));
        bloq.add_edge(owner, readout, BloqEdge::Order);
        bloq.validate().unwrap();
        let mut malformed = bloq.clone();
        malformed.node_mut(condition).unwrap().kind = crate::BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(99)])),
            }
            .into(),
        );
        assert!(matches!(malformed.validate(),
            Err(BloqValidationError::MissingClassicalInput { node, slot: 99 }) if node == condition));
        let mut cyclic = bloq.clone();
        cyclic.add_edge(condition, condition, BloqEdge::value(2));
        assert!(matches!(
            cyclic.validate(),
            Err(BloqValidationError::CyclicGraph)
        ));
        bloq.node_mut(condition).unwrap().kind = crate::BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            }
            .into(),
        );
        assert!(matches!(
            bloq.validate(),
            Err(BloqValidationError::ConditionalReferenceUnavailable { .. })
        ));
    }

    #[test]
    fn correlated_predicates_do_not_use_registry_allocation_order() {
        let mut graph = SubGraph::new();
        let groups: Vec<Vec<_>> = (0..2)
            .map(|_| {
                (0..20)
                    .map(|_| {
                        graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                            Vec::new(),
                            Vec::new(),
                        )))
                    })
                    .collect()
            })
            .collect();
        let equalities: Vec<_> = (0..20)
            .map(|index| {
                let node = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::parity([0, 1], true),
                }));
                graph.add_edge(groups[0][index], node, BloqEdge::value(0));
                graph.add_edge(groups[1][index], node, BloqEdge::value(1));
                node
            })
            .collect();
        let all_equal = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::And((0..20).map(ClassicalExpr::In).collect()),
        }));
        for (slot, &equal) in equalities.iter().enumerate() {
            graph.add_edge(equal, all_equal, BloqEdge::value(slot as u32));
        }
        let mut analysis = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 1_000,
                max_steps: 100_000,
            },
        );
        assert!(analysis.implies(Some(all_equal), equalities[0]).unwrap());
        assert!(!analysis.implies(Some(equalities[0]), all_equal).unwrap());
        assert!(
            !analysis
                .assignment_reachable(&[
                    (all_equal, true),
                    (groups[0][0], false),
                    (groups[1][0], true),
                ])
                .unwrap()
        );
        assert!(
            analysis
                .assignment_reachable(&[
                    (all_equal, true),
                    (groups[0][0], true),
                    (groups[1][0], true),
                ])
                .unwrap()
        );
    }

    #[test]
    fn deep_shared_predicates_keep_correlations_and_honor_limits() {
        let mut graph = SubGraph::new();
        let raw = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let mut tail = raw;
        for _ in 0..12_000 {
            let next = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }));
            graph.add_edge(tail, next, BloqEdge::value(0));
            tail = next;
        }
        let different = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(vec![ClassicalExpr::In(0), ClassicalExpr::In(1)].into()),
        }));
        graph.add_edge(raw, different, BloqEdge::value(0));
        graph.add_edge(tail, different, BloqEdge::value(1));
        let mut analysis = PredicateAnalysis::new(&graph);
        assert!(analysis.overlap(Some(raw), Some(tail)).unwrap());
        assert!(!analysis.overlap(None, Some(different)).unwrap());
        assert_eq!(
            analysis.values_up_to(&[raw, tail, different], 3).unwrap(),
            [vec![false, false, false], vec![true, true, false]]
        );
        assert!(matches!(
            PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 4,
                    max_steps: 100
                }
            )
            .values_up_to(&[raw, tail], 3),
            Err(Error::BooleanResource(_))
        ));
        assert!(matches!(
            PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 4,
                    max_steps: 100
                }
            )
            .overlap(None, Some(tail)),
            Err(Error::BooleanResource(_))
        ));
        assert!(matches!(
            PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 4,
                    max_steps: 100
                }
            )
            .values_fixed_by_assignment(&[(raw.into(), true)], [tail.into()]),
            Err(Error::BooleanResource(_))
        ));
        assert!(matches!(
            PredicateAnalysis::with_limits(
                &graph,
                BooleanLimits {
                    max_nodes: 0,
                    max_steps: 100
                }
            )
            .overlap(None, Some(raw)),
            Err(Error::BooleanResource(_))
        ));
        let mut constants = PredicateAnalysis::with_limits(
            &graph,
            BooleanLimits {
                max_nodes: 0,
                max_steps: 2,
            },
        );
        assert!(constants.overlap(None, None).unwrap());
        assert!(constants.overlap(None, None).unwrap());
        assert!(matches!(
            constants.overlap(None, None),
            Err(Error::BooleanResource(_))
        ));
    }

    #[test]
    fn disjoint_constant_members_scale_with_footprint_and_charge_work() {
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let guard = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().instances = (0..1_000)
            .map(|index| {
                TemplateInstance::new(TemplateInstanceId(index), template, ivec2(index as i32, 0))
            })
            .collect();
        node.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: (0..1_000).map(TemplateInstanceId).collect(),
            ..Default::default()
        });
        let node = bloq.add_node(node);
        bloq.add_edge(guard, node, BloqEdge::value(0));
        let options =
            crate::lowering::InstantiationOptions::default().with_boolean_limits(BooleanLimits {
                max_nodes: 0,
                max_steps: 12_000,
            });
        bloq.validate_with_options(&options).unwrap();
        let limited = options.with_boolean_limits(BooleanLimits {
            max_nodes: 0,
            max_steps: 100,
        });
        assert!(matches!(
            bloq.validate_with_options(&limited),
            Err(BloqValidationError::BooleanResource(_)
                | BloqValidationError::InvalidInstanceMergeStructure {
                    source: Error::BooleanResource(_),
                    ..
                },)
        ));
    }

    #[test]
    fn disjoint_templates_still_require_coexisting_tick_shapes() {
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let left = bloq.add_template(BloqTemplate::new(circuit.clone()));
        circuit.tick();
        let right = bloq.add_template(BloqTemplate::new(circuit));
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let inverse = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(raw, inverse, BloqEdge::value(0));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), left, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), right, ivec2(100, 0)),
        ];
        node.expect_quantum_mut().guards = vec![
            QuantumGuard {
                input: 0,
                instances: vec![TemplateInstanceId(0)],
                ..Default::default()
            },
            QuantumGuard {
                input: 1,
                instances: vec![TemplateInstanceId(1)],
                ..Default::default()
            },
        ];
        let node = bloq.add_node(node);
        bloq.add_edge(raw, node, BloqEdge::value(0));
        bloq.add_edge(inverse, node, BloqEdge::value(1));
        bloq.validate().unwrap();
        bloq.node_mut(inverse).unwrap().kind = crate::BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }
            .into(),
        );
        assert!(matches!(
            bloq.validate(),
            Err(BloqValidationError::InvalidInstanceMergeStructure {
                source: Error::TickSegmentCountMismatch,
                ..
            })
        ));
    }

    #[test]
    fn deep_disjoint_copies_merge_when_they_coexist() {
        use bloq_circuit::{CircuitBody, Op};
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        for _ in 0..257 {
            let child = circuit.entry_body();
            let parent = circuit.add_body(CircuitBody::new());
            circuit
                .body_mut(parent)
                .unwrap()
                .ops_mut()
                .push(Op::Repeat {
                    body: child,
                    repetitions: 1,
                });
            circuit.set_entry_body(parent).unwrap();
        }
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let inverse = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(raw, inverse, BloqEdge::value(0));
        let mut node = BloqNode::from_members(Vec::new());
        for index in 0..2 {
            let instance = TemplateInstanceId(index);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    instance,
                    template,
                    ivec2(index as i32, 0),
                ));
            node.expect_quantum_mut().guards.push(QuantumGuard {
                input: index,
                instances: vec![instance],
                ..Default::default()
            });
        }
        let node = bloq.add_node(node);
        bloq.add_edge(raw, node, BloqEdge::value(0));
        bloq.add_edge(inverse, node, BloqEdge::value(1));
        bloq.validate().unwrap();
        bloq.node_mut(inverse).unwrap().kind = crate::BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }
            .into(),
        );
        bloq.validate().unwrap();
    }
}
