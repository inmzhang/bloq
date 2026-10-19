//! Validated classical-action dependencies and reachable value domains.

pub(crate) mod hierarchy;
mod svg;

use std::collections::{BTreeSet, HashMap, HashSet};

use bloq_utils::PauliBasis;
use glam::IVec3;
use petgraph::algo::toposort;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rand::{Rng, RngExt};

use crate::validate::InvalidActionError;
use crate::{
    Action, BlockGraph, BlockGraphError, BlockKind, BranchRegion, Expr, MeasureTarget,
    SelectiveKind, Stabilizer, StabilizerGenerator,
};

/// Distinct jointly reachable values of an ordered set of Boolean action
/// conditions.
///
/// This is the canonical branch domain shared by stabilizer construction,
/// lowering, editor pinning, and public reachability queries. Values use the
/// same target order supplied at construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveValueDomain {
    values: Vec<Vec<bool>>,
}

/// A bounded value query exhausted Boolean work or found too many values.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ResolveDomainError {
    /// Boolean resource limits were exceeded while enumerating values.
    #[error("{0}")]
    Resource(#[from] BooleanResourceError),
    /// The domain exceeded the caller's value-count limit.
    #[error("resolve domain contains at least {observed} values")]
    TooManyValues {
        /// Lower bound on the number of reachable values observed.
        observed: usize,
    },
}

impl ResolveDomainError {}

impl ResolveValueDomain {
    /// Distinct reachable value tuples.
    pub fn values(&self) -> &[Vec<bool>] {
        &self.values
    }

    /// Whether `values` is one reachable tuple in this domain.
    pub fn contains(&self, values: &[bool]) -> bool {
        self.values
            .binary_search_by(|candidate| candidate.as_slice().cmp(values))
            .is_ok()
    }
}

use bloq_utils::boolean::{BooleanDecisionDiagram, BooleanLimits, BooleanOp, BooleanResourceError};
pub(crate) use bloq_utils::boolean::{DECISION_FALSE, DECISION_TRUE, DecisionId};

#[derive(Debug, Default)]
pub(crate) struct ResolveDecisionDiagram {
    pub(crate) diagram: BooleanDecisionDiagram,
    pub(crate) variable_indices: HashMap<String, usize>,
    alias_cache: HashMap<String, DecisionId>,
    pub(crate) roots: Vec<DecisionId>,
}

impl ResolveDecisionDiagram {
    pub(crate) fn build(
        exprs: &[&Expr],
        aliases: &HashMap<&str, &Expr>,
        limits: BooleanLimits,
    ) -> Result<Option<Self>, BooleanResourceError> {
        let mut diagram = Self {
            diagram: BooleanDecisionDiagram::with_limits(limits),
            ..Self::default()
        };
        for expr in exprs {
            let Some(root) = diagram.expression(expr, aliases)? else {
                return Ok(None);
            };
            diagram.roots.push(root);
        }
        Ok(Some(diagram))
    }

    fn expression(
        &mut self,
        expr: &Expr,
        aliases: &HashMap<&str, &Expr>,
    ) -> Result<Option<DecisionId>, BooleanResourceError> {
        enum Step<'a> {
            Expr(&'a Expr),
            Alias(&'a str),
            Not,
            Binary(crate::BinaryOp),
        }
        let mut pending = vec![Step::Expr(expr)];
        let mut values = Vec::new();
        let mut visiting = HashSet::new();
        while let Some(step) = pending.pop() {
            self.diagram.charge(1)?;
            match step {
                Step::Expr(Expr::Var(name)) => {
                    if let Some(&root) = self.alias_cache.get(name) {
                        values.push(root);
                    } else if let Some(&resolved) = aliases.get(name.as_str()) {
                        if !visiting.insert(name.as_str()) {
                            return Ok(None);
                        }
                        pending.extend([Step::Alias(name), Step::Expr(resolved)]);
                    } else {
                        // First-use order keeps neighboring source controls together;
                        // lexical names can interleave unrelated sites (bit10 before bit2).
                        let next = self.variable_indices.len();
                        let variable = *self.variable_indices.entry(name.clone()).or_insert(next);
                        values.push(self.make_node(variable, DECISION_FALSE, DECISION_TRUE)?);
                    }
                }
                Step::Expr(Expr::Not(inner)) => {
                    pending.extend([Step::Not, Step::Expr(inner)]);
                }
                Step::Expr(Expr::Binary(operator, left, right)) => {
                    pending.extend([Step::Binary(*operator), Step::Expr(right), Step::Expr(left)]);
                }
                Step::Alias(name) => {
                    visiting.remove(name);
                    self.alias_cache.insert(
                        name.to_owned(),
                        *values
                            .last()
                            .expect("alias expression is evaluated before caching"),
                    );
                }
                Step::Not => {
                    let value = values.pop().expect("operand is evaluated before negation");
                    values.push(self.negate(value)?);
                }
                Step::Binary(operator) => {
                    let right = values
                        .pop()
                        .expect("right operand is evaluated before its operator");
                    let left = values
                        .pop()
                        .expect("left operand is evaluated before its operator");
                    values.push(self.apply(operator, left, right)?);
                }
            }
        }
        Ok(values.pop())
    }

    pub(crate) fn make_node(
        &mut self,
        variable: usize,
        low: DecisionId,
        high: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        self.diagram.make_node(variable, low, high)
    }
    pub(crate) fn negate(&mut self, id: DecisionId) -> Result<DecisionId, BooleanResourceError> {
        self.diagram.negate(id)
    }
    pub(crate) fn apply(
        &mut self,
        op: crate::BinaryOp,
        left: DecisionId,
        right: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        let op = match op {
            crate::BinaryOp::Xor => BooleanOp::Xor,
            crate::BinaryOp::And => BooleanOp::And,
            crate::BinaryOp::Or => BooleanOp::Or,
        };
        self.diagram.apply(op, left, right)
    }
    fn values(&mut self) -> Result<Vec<Vec<bool>>, BooleanResourceError> {
        self.values_up_to(usize::MAX)
    }
    pub(crate) fn values_up_to(
        &mut self,
        limit: usize,
    ) -> Result<Vec<Vec<bool>>, BooleanResourceError> {
        self.diagram.values_up_to(&self.roots, limit)
    }
    fn contains(&mut self, values: &[bool]) -> Result<bool, BooleanResourceError> {
        if values.len() != self.roots.len() {
            return Ok(false);
        }
        let mut conjunction = DECISION_TRUE;
        for (index, &value) in values.iter().enumerate() {
            let root = self.roots[index];
            let required = if value { root } else { self.negate(root)? };
            conjunction = self.apply(crate::BinaryOp::And, conjunction, required)?;
            if conjunction == DECISION_FALSE {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
/// The Pauli basis a measurement action observes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementObservable {
    /// A fixed Pauli basis.
    Concrete(PauliBasis),
    /// A basis chosen at runtime by resolving a selective block.
    Selective(SelectiveKind),
}

/// Why one action must complete before another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionDependency {
    /// A condition or binding reads a variable produced by the predecessor.
    Classical,
    /// A measurement surface crosses a selective site and waits for its resolve.
    SelectiveSupport,
    /// A measurement surface crosses a terminal conditional region and waits
    /// for its structural branch decision.
    BranchSupport,
    /// Feedback anticommutes with a measurement surface and changes its value.
    FeedbackAnticommutation,
    /// A composed parity reuses an already decoded named readout.
    ReadoutParity,
}

impl ActionDependency {
    /// Whether this dependency was derived from stabilizer support.
    pub const fn is_implicit(self) -> bool {
        !matches!(self, Self::Classical)
    }
}

/// Error from [`ActionDag::set_measurement_observable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub(crate) enum SetMeasurementObservableError {
    /// The requested action ordinal is outside the action list.
    #[error("action ordinal {ordinal} is out of range for {len} actions")]
    OrdinalOutOfRange {
        /// Requested source action ordinal.
        ordinal: usize,
        /// Number of actions in the DAG.
        len: usize,
    },
    /// The requested action is not a measurement.
    #[error("action at ordinal {ordinal} is not a measurement node")]
    NotMeasurementNode {
        /// Source action ordinal.
        ordinal: usize,
    },
}

/// Authored module ownership of an action in a composed dependency graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOwner {
    /// Name of the reusable module definition.
    pub definition: String,
    /// Qualified instance path, separated by `__`. Empty for the root.
    pub instance_path: String,
}

/// A source action and its cached measurement metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionNode {
    /// Position in the standalone source order or composed action list.
    pub ordinal: usize,
    /// The action itself.
    pub action: Action,
    /// Module definition and instance that own this composed action.
    /// Absent on a DAG built from a standalone action list or flat graph.
    /// Interface bindings belong to the receiving instance.
    pub owner: Option<ActionOwner>,
    /// Resolved observable, set for measurement nodes during validation.
    /// Edge records use the stored pipe source in a `BlockGraph` and the smaller
    /// node ID endpoint in a `ZXGraph`; conversion conjugates across H as needed.
    pub measurement: Option<MeasurementObservable>,
    /// Derived stabilizer surface for a measurement node after physical analysis.
    pub measurement_stabilizer: Option<Stabilizer>,
}

/// An ordered dependency graph over a program's [`Action`]s.
///
/// Nodes preserve source order via [`ActionNode::ordinal`]; edges point from a
/// definition (a `let` or `measure` binding, or a resolve of a selective block)
/// to each consumer. Timing is not stored; compiler scheduling also uses graph
/// topology.
#[derive(Debug, Clone, Default)]
pub struct ActionDag {
    graph: StableDiGraph<ActionNode, ActionDependency>,
    ordered: Vec<NodeIndex<u32>>,
    inputs: BTreeSet<String>,
    branch_regions: Vec<BranchRegion>,
    analyzed: bool,
}

impl ActionDag {
    /// Selective fills needed by this named readout, in coordinate order.
    /// Returns `None` when the measurement is not in this action graph.
    pub fn resolve_sites_for_measurement(&self, name: &str) -> Option<Vec<IVec3>> {
        let consumer = self
            .ordered_nodes()
            .find(|node| {
                matches!(&node.action, Action::Measure { name: measured, .. } if measured == name)
            })?
            .ordinal;
        let mut sites = self
            .dependencies()
            .filter_map(|(predecessor, target, dependency)| {
                if target != consumer || dependency != ActionDependency::SelectiveSupport {
                    return None;
                }
                match self.node_by_ordinal(predecessor)?.action {
                    Action::Resolve { target, .. } => Some(target),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        sites.sort_unstable_by_key(IVec3::to_array);
        sites.dedup();
        Some(sites)
    }

    /// Builds the DAG from actions in source order, wiring dependency edges from
    /// each variable, measurement, or resolve definition to its uses.
    pub fn from_actions(actions: &[Action]) -> Self {
        Self::from_actions_with_inputs(actions, std::iter::empty())
    }

    pub(crate) fn from_actions_with_inputs(
        actions: &[Action],
        inputs: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut dag = ActionDag::default();
        dag.inputs.extend(inputs);
        let mut defs = HashMap::<&str, NodeIndex<u32>>::new();
        let mut resolve_targets = HashMap::<IVec3, NodeIndex<u32>>::new();

        for (ordinal, action) in actions.iter().enumerate() {
            let node = dag.graph.add_node(ActionNode {
                ordinal,
                action: action.clone(),
                owner: None,
                measurement: None,
                measurement_stabilizer: None,
            });
            dag.ordered.push(node);

            if let Some(name) = defines_name(action) {
                defs.entry(name).or_insert(node);
            }

            if let Action::Resolve { target, .. } = action {
                resolve_targets.entry(*target).or_insert(node);
            }
        }

        for (ordinal, action) in actions.iter().enumerate() {
            let node = dag.ordered[ordinal];
            for used in action.referenced_names() {
                let Some(&pred) = defs.get(used) else {
                    continue;
                };
                dag.add_dependency(pred, node, ActionDependency::Classical);
            }

            if let Action::Measure {
                target: MeasureTarget::Node(pos),
                ..
            } = action
                && let Some(&pred) = resolve_targets.get(pos)
            {
                dag.add_dependency(pred, node, ActionDependency::SelectiveSupport);
            }
        }

        dag
    }

    /// External Boolean names accepted by this action program.
    pub fn inputs(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(String::as_str)
    }

    /// Iterates over nodes in source order.
    pub fn ordered_nodes(&self) -> impl Iterator<Item = &ActionNode> {
        self.ordered.iter().map(|&idx| &self.graph[idx])
    }

    /// Returns the validated terminal region owned by `target`, if this DAG
    /// was built against a source graph containing that branch action.
    pub(crate) fn branch_region(&self, target: IVec3) -> Option<&BranchRegion> {
        self.branch_regions
            .iter()
            .find(|region| region.target == target)
    }

    /// Iterates over dependency edges as `(predecessor ordinal, consumer ordinal, reason)`.
    pub fn dependencies(&self) -> impl Iterator<Item = (usize, usize, ActionDependency)> + '_ {
        self.graph.edge_references().map(|edge| {
            (
                self.graph[edge.source()].ordinal,
                self.graph[edge.target()].ordinal,
                *edge.weight(),
            )
        })
    }

    /// Whether correlation-derived dependencies are current.
    /// An empty program has nothing to derive.
    pub const fn is_analyzed(&self) -> bool {
        self.analyzed || self.ordered.is_empty()
    }

    pub(crate) fn attach_guarded_dependencies(
        &mut self,
        edges: impl IntoIterator<Item = (usize, usize, ActionDependency)>,
    ) -> Result<(), BlockGraphError> {
        for (from, to, reason) in edges {
            self.add_dependency(self.ordered[from], self.ordered[to], reason);
        }
        self.validate_semantics()?;
        self.analyzed = true;
        Ok(())
    }

    /// Records each named readout's surface on its measurement node.
    pub(crate) fn attach_measurement_surfaces(&mut self, surfaces: Vec<StabilizerGenerator>) {
        let mut surfaces = surfaces
            .into_iter()
            .filter_map(|row| match row.kind {
                crate::StabilizerRowKind::Measurement { name } => Some((name, row.stabilizer)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        for &node in &self.ordered {
            let node = &mut self.graph[node];
            if let Action::Measure { name, .. } = &node.action {
                node.measurement_stabilizer = surfaces.remove(name);
            }
        }
    }

    /// Returns the node at the given source ordinal, if any.
    pub fn node_by_ordinal(&self, ordinal: usize) -> Option<&ActionNode> {
        self.ordered.get(ordinal).map(|&idx| &self.graph[idx])
    }

    /// Records the resolved observable for the measurement node at `ordinal`.
    ///
    /// # Errors
    ///
    /// Returns an error if `ordinal` is out of range or does not name a
    /// measurement node.
    pub(crate) fn set_measurement_observable(
        &mut self,
        ordinal: usize,
        measurement: MeasurementObservable,
    ) -> Result<(), SetMeasurementObservableError> {
        let Some(&idx) = self.ordered.get(ordinal) else {
            return Err(SetMeasurementObservableError::OrdinalOutOfRange {
                ordinal,
                len: self.ordered.len(),
            });
        };
        if !matches!(self.graph[idx].action, Action::Measure { .. }) {
            return Err(SetMeasurementObservableError::NotMeasurementNode { ordinal });
        }
        self.graph[idx].measurement = Some(measurement);
        Ok(())
    }

    /// Returns whether `node` has a direct dependency edge from the action at
    /// `pred_ordinal`.
    pub fn depends_on(&self, node: &ActionNode, pred_ordinal: usize) -> bool {
        let Some(&node_idx) = self.ordered.get(node.ordinal) else {
            return false;
        };
        let Some(&pred_idx) = self.ordered.get(pred_ordinal) else {
            return false;
        };
        self.graph.find_edge(pred_idx, node_idx).is_some()
    }

    fn add_dependency(
        &mut self,
        predecessor: NodeIndex<u32>,
        consumer: NodeIndex<u32>,
        dependency: ActionDependency,
    ) {
        if self
            .graph
            .edges_connecting(predecessor, consumer)
            .any(|edge| *edge.weight() == dependency)
        {
            return;
        }
        self.graph.add_edge(predecessor, consumer, dependency);
    }

    fn validate_semantics(&self) -> Result<(), BlockGraphError> {
        let mut defined_vars = self.inputs.iter().cloned().collect::<HashSet<_>>();
        let mut resolve_targets = HashSet::new();
        let mut branch_targets = HashSet::new();
        let mut measurement_targets = HashSet::new();
        let bucket_measurements = self
            .ordered_nodes()
            .filter_map(|node| match &node.action {
                Action::Measure { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect::<HashSet<_>>();

        for action in self.ordered_nodes().map(|node| &node.action) {
            BlockGraph::validate_action_syntax(action)?;
            match action {
                Action::Measure { target, name } => {
                    if defined_vars.contains(name) {
                        return Err(InvalidActionError::VariableRedefinition(name.clone()).into());
                    }
                    let site_key = measurement_site_key(target)?;
                    if !measurement_targets.insert(site_key) {
                        return Err(InvalidActionError::DuplicateMeasurementTarget(*target).into());
                    }
                    defined_vars.insert(name.clone());
                }
                Action::Let { name, expr } => {
                    if defined_vars.contains(name) {
                        return Err(InvalidActionError::VariableRedefinition(name.clone()).into());
                    }
                    check_expr_vars(expr, &defined_vars, &bucket_measurements)?;
                    defined_vars.insert(name.clone());
                }
                Action::DiscardIf(expr) => {
                    check_expr_vars(expr, &defined_vars, &bucket_measurements)?;
                }
                Action::Resolve { target, condition } => {
                    check_expr_vars(condition, &defined_vars, &bucket_measurements)?;
                    if !resolve_targets.insert(*target) {
                        return Err(InvalidActionError::DuplicateResolveTarget(*target).into());
                    }
                }
                Action::Branch { target, condition } => {
                    check_expr_vars(condition, &defined_vars, &bucket_measurements)?;
                    if !branch_targets.insert(*target) {
                        return Err(InvalidActionError::DuplicateBranchTarget(*target).into());
                    }
                }
                Action::Feedback { condition, .. } => {
                    if let Some(condition) = condition {
                        check_expr_vars(condition, &defined_vars, &bucket_measurements)?;
                    }
                }
            }
        }

        toposort(&self.graph, None).map_err(|cycle| {
            let ordinal = self.graph[cycle.node_id()].ordinal;
            BlockGraphError::InvalidAction(InvalidActionError::DependencyCycle { ordinal })
        })?;

        Ok(())
    }

    fn validate_graph_constraints(&mut self, graph: &BlockGraph) -> Result<(), BlockGraphError> {
        let mut selective_positions = graph
            .blocks()
            .filter(|block| block.kind.is_selective())
            .map(|block| block.pos)
            .collect::<HashSet<_>>();

        for action in self.ordered_nodes().map(|node| &node.action) {
            match action {
                Action::Measure { target, .. } => {
                    ensure_measure_target_is_valid(graph, target)?;
                }
                Action::Resolve { target, .. } => {
                    match graph.get_block(*target) {
                        Some(b) if b.kind.is_selective() => {}
                        _ => {
                            return Err(InvalidActionError::InvalidResolveTarget(*target).into());
                        }
                    }
                    selective_positions.remove(target);
                }
                Action::Branch { target, .. } => {
                    if graph.branch_by_target(*target).is_none() {
                        return Err(InvalidActionError::InvalidBranchTarget(*target).into());
                    }
                }
                Action::Feedback { targets, .. } => {
                    for feedback in targets {
                        ensure_feedback_target_exists(graph, feedback.target)?;
                        if let Some(dir) = feedback.direction {
                            let dst = crate::checked_add_position(feedback.target, dir.to_ivec3())?;
                            if graph.get_pipe(feedback.target, dst).is_none() {
                                return Err(InvalidActionError::InvalidFeedbackTarget(
                                    feedback.target,
                                )
                                .into());
                            }
                        }
                    }
                }
                Action::Let { .. } | Action::DiscardIf(..) => {}
            }
        }

        if let Some(&missing) = selective_positions.iter().min_by_key(|pos| pos.to_array()) {
            return Err(InvalidActionError::MissingResolveForSelective { target: missing }.into());
        }

        let actions = self
            .ordered_nodes()
            .map(|node| node.action.clone())
            .collect::<Vec<_>>();
        self.branch_regions = crate::branch::validate_regions_for_actions(graph, &actions)?;

        Ok(())
    }

    pub(crate) fn validate(
        &mut self,
        source_graph: Option<&BlockGraph>,
    ) -> Result<(), BlockGraphError> {
        self.validate_semantics()?;

        if let Some(graph) = source_graph {
            self.validate_graph_constraints(graph)?;
        }

        Ok(())
    }

    /// Builds the exact distinct resolve-value domain at `targets`, following
    /// `Let` bindings. A reduced shared decision diagram avoids enumerating the
    /// measurement inputs; materializing the output domain can still be
    /// exponential when that many distinct tuples are genuinely reachable.
    /// A missing or cyclic resolve target has an empty domain.
    ///
    /// # Errors
    ///
    /// Returns an error if Boolean construction or enumeration exceeds its budget.
    pub fn resolve_value_domain(
        &self,
        targets: &[IVec3],
    ) -> Result<ResolveValueDomain, BooleanResourceError> {
        let Some(mut diagram) = self.resolve_decision_diagram(targets)? else {
            return Ok(ResolveValueDomain { values: Vec::new() });
        };
        Ok(ResolveValueDomain {
            values: diagram.values()?,
        })
    }

    /// Builds the exact resolve-value domain when it contains at most `limit`
    /// distinct tuples.
    ///
    /// # Errors
    ///
    /// Returns Boolean resource exhaustion or the first observed size above
    /// `limit` instead of truncating the domain.
    pub fn resolve_value_domain_bounded(
        &self,
        targets: &[IVec3],
        limit: usize,
    ) -> Result<ResolveValueDomain, ResolveDomainError> {
        self.resolve_value_domain_bounded_with_limits(targets, limit, BooleanLimits::DEFAULT)
    }

    /// Query a bounded output domain with an explicit Boolean work/allocation budget.
    ///
    /// # Errors
    ///
    /// Returns resource exhaustion or the first domain size above `limit`.
    pub fn resolve_value_domain_bounded_with_limits(
        &self,
        targets: &[IVec3],
        limit: usize,
        boolean_limits: BooleanLimits,
    ) -> Result<ResolveValueDomain, ResolveDomainError> {
        let Some(mut diagram) =
            self.resolve_decision_diagram_with_limits(targets, boolean_limits)?
        else {
            return Ok(ResolveValueDomain { values: Vec::new() });
        };
        let values = diagram.values_up_to(limit.saturating_add(1))?;
        if values.len() > limit {
            Err(ResolveDomainError::TooManyValues {
                observed: values.len(),
            })
        } else {
            Ok(ResolveValueDomain { values })
        }
    }

    /// Whether `values` is jointly reachable for `targets`, decided
    /// symbolically without materializing the full output domain. Missing or
    /// cyclic targets and mismatched slice lengths are not reachable.
    ///
    /// # Errors
    ///
    /// Returns an error if Boolean construction or evaluation exceeds its budget.
    pub fn resolve_values_are_reachable(
        &self,
        targets: &[IVec3],
        values: &[bool],
    ) -> Result<bool, BooleanResourceError> {
        let Some(mut diagram) = self.resolve_decision_diagram(targets)? else {
            return Ok(false);
        };
        diagram.contains(values)
    }

    pub(crate) fn branch_values_up_to(
        &self,
        targets: &[IVec3],
        limit: usize,
    ) -> Result<Vec<Vec<bool>>, BooleanResourceError> {
        self.branch_values_up_to_with_limits(targets, limit, BooleanLimits::DEFAULT)
    }

    pub(crate) fn branch_values_up_to_with_limits(
        &self,
        targets: &[IVec3],
        limit: usize,
        limits: BooleanLimits,
    ) -> Result<Vec<Vec<bool>>, BooleanResourceError> {
        let Some(mut diagram) = self.branch_decision_diagram(targets, limits)? else {
            return Ok(Vec::new());
        };
        diagram.values_up_to(limit)
    }

    pub(crate) fn resolve_decision_diagram(
        &self,
        targets: &[IVec3],
    ) -> Result<Option<ResolveDecisionDiagram>, BooleanResourceError> {
        self.resolve_decision_diagram_with_limits(targets, BooleanLimits::DEFAULT)
    }

    pub(crate) fn resolve_decision_diagram_with_limits(
        &self,
        targets: &[IVec3],
        limits: BooleanLimits,
    ) -> Result<Option<ResolveDecisionDiagram>, BooleanResourceError> {
        let exprs = targets
            .iter()
            .map(|&target| self.resolve_expr(target))
            .collect::<Option<Vec<_>>>();
        match exprs {
            Some(exprs) => ResolveDecisionDiagram::build(&exprs, &self.let_bindings(), limits),
            None => Ok(None),
        }
    }

    fn branch_decision_diagram(
        &self,
        targets: &[IVec3],
        limits: BooleanLimits,
    ) -> Result<Option<ResolveDecisionDiagram>, BooleanResourceError> {
        let exprs = targets
            .iter()
            .map(|&target| self.branch_expr(target))
            .collect::<Option<Vec<_>>>();
        match exprs {
            Some(exprs) => ResolveDecisionDiagram::build(&exprs, &self.let_bindings(), limits),
            None => Ok(None),
        }
    }

    /// Sample one shared assignment of the measurement variables feeding the
    /// requested resolves, then evaluate every available target condition from
    /// that assignment. Targets without a resolve are omitted so structural
    /// graph helpers can retain their independent fallback behavior.
    pub(crate) fn sample_resolve_values(
        &self,
        targets: &[IVec3],
        rng: &mut impl Rng,
    ) -> HashMap<IVec3, bool> {
        let exprs = targets
            .iter()
            .filter_map(|&target| {
                self.ordered_nodes()
                    .find_map(|node| match &node.action {
                        Action::Resolve {
                            target: candidate,
                            condition,
                        } if *candidate == target => Some(condition),
                        _ => None,
                    })
                    .map(|expr| (target, expr))
            })
            .collect::<Vec<_>>();
        let aliases = self.let_bindings();
        let mut variables = BTreeSet::new();
        let mut visited = HashSet::new();
        for (_, expr) in &exprs {
            expand_expr_vars_to_measurements(expr, &aliases, &mut variables, &mut visited);
        }
        let values = variables
            .into_iter()
            .map(|name| (name, rng.random()))
            .collect::<HashMap<_, _>>();

        let mut alias_values = HashMap::new();
        exprs
            .into_iter()
            .map(|(target, expr)| {
                (
                    target,
                    evaluate_expr_with(expr, &aliases, &|name| values[name], &mut alias_values),
                )
            })
            .collect()
    }

    fn resolve_expr(&self, target: IVec3) -> Option<&Expr> {
        self.ordered_nodes().find_map(|node| match &node.action {
            Action::Resolve {
                target: candidate,
                condition,
            } if *candidate == target => Some(condition),
            _ => None,
        })
    }

    fn branch_expr(&self, target: IVec3) -> Option<&Expr> {
        self.ordered_nodes().find_map(|node| match &node.action {
            Action::Branch {
                target: candidate,
                condition,
            } if *candidate == target => Some(condition),
            _ => None,
        })
    }

    fn let_bindings(&self) -> HashMap<&str, &Expr> {
        self.ordered_nodes()
            .filter_map(|node| match &node.action {
                Action::Let { name, expr } => Some((name.as_str(), expr)),
                _ => None,
            })
            .collect()
    }
}

fn evaluate_expr_with(
    expr: &Expr,
    aliases: &HashMap<&str, &Expr>,
    value_of: &impl Fn(&str) -> bool,
    alias_values: &mut HashMap<String, bool>,
) -> bool {
    match expr {
        Expr::Var(name) => {
            if let Some(&value) = alias_values.get(name) {
                return value;
            }
            let Some(expr) = aliases.get(name.as_str()) else {
                return value_of(name);
            };
            let value = evaluate_expr_with(expr, aliases, value_of, alias_values);
            alias_values.insert(name.clone(), value);
            value
        }
        Expr::Not(expr) => !evaluate_expr_with(expr, aliases, value_of, alias_values),
        Expr::Binary(op, left, right) => {
            let left = evaluate_expr_with(left, aliases, value_of, alias_values);
            let right = evaluate_expr_with(right, aliases, value_of, alias_values);
            match op {
                crate::BinaryOp::Xor => left ^ right,
                crate::BinaryOp::And => left & right,
                crate::BinaryOp::Or => left | right,
            }
        }
    }
}

fn defines_name(action: &Action) -> Option<&str> {
    match action {
        Action::Let { name, .. } | Action::Measure { name, .. } => Some(name),
        Action::DiscardIf(_)
        | Action::Resolve { .. }
        | Action::Branch { .. }
        | Action::Feedback { .. } => None,
    }
}

fn expand_expr_vars_to_measurements(
    expr: &Expr,
    aliases: &HashMap<&str, &Expr>,
    out: &mut BTreeSet<String>,
    visited: &mut HashSet<String>,
) {
    match expr {
        Expr::Var(name) => {
            if let Some(resolved) = aliases.get(name.as_str()) {
                if !visited.insert(name.clone()) {
                    return;
                }
                expand_expr_vars_to_measurements(resolved, aliases, out, visited);
            } else {
                out.insert(name.clone());
            }
        }
        Expr::Not(expr) => expand_expr_vars_to_measurements(expr, aliases, out, visited),
        Expr::Binary(_, lhs, rhs) => {
            expand_expr_vars_to_measurements(lhs, aliases, out, visited);
            expand_expr_vars_to_measurements(rhs, aliases, out, visited);
        }
    }
}

fn check_expr_vars(
    expr: &Expr,
    defined: &HashSet<String>,
    bucket_measurements: &HashSet<String>,
) -> Result<(), InvalidActionError> {
    match expr {
        Expr::Var(name) => {
            if !defined.contains(name) && !bucket_measurements.contains(name) {
                return Err(InvalidActionError::UndefinedVariable(name.clone()));
            }
        }
        Expr::Not(expr) => check_expr_vars(expr, defined, bucket_measurements)?,
        Expr::Binary(_, lhs, rhs) => {
            check_expr_vars(lhs, defined, bucket_measurements)?;
            check_expr_vars(rhs, defined, bucket_measurements)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MeasurementSiteKey {
    Node(IVec3),
    Edge { a: IVec3, b: IVec3 },
}

fn measurement_site_key(target: &MeasureTarget) -> Result<MeasurementSiteKey, BlockGraphError> {
    Ok(match target {
        MeasureTarget::Node(pos) => MeasurementSiteKey::Node(*pos),
        MeasureTarget::Edge { src, dir } => {
            let dst = crate::checked_add_position(*src, dir.to_ivec3())?;
            if src.to_array() <= dst.to_array() {
                MeasurementSiteKey::Edge { a: *src, b: dst }
            } else {
                MeasurementSiteKey::Edge { a: dst, b: *src }
            }
        }
    })
}

fn ensure_measure_target_is_valid(
    graph: &BlockGraph,
    target: &MeasureTarget,
) -> Result<(), InvalidActionError> {
    match target {
        MeasureTarget::Node(pos) => match graph.get_block(*pos).map(|block| block.kind) {
            Some(
                BlockKind::Cube(_)
                | BlockKind::Y
                | BlockKind::Measurement(_)
                | BlockKind::Selective(_),
            ) => Ok(()),
            _ => Err(InvalidActionError::InvalidMeasurementNode(*pos)),
        },
        MeasureTarget::Edge { src, dir } if dir.as_udirection() == crate::UDirection::Z => {
            Err(InvalidActionError::TimeLikeMeasurementEdge(*src, *dir))
        }
        MeasureTarget::Edge { .. } => Ok(()),
    }
}

fn ensure_feedback_target_exists(graph: &BlockGraph, pos: IVec3) -> Result<(), InvalidActionError> {
    if graph.get_block(pos).is_none() {
        return Err(InvalidActionError::InvalidFeedbackTarget(pos));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ResolveDomainError, SetMeasurementObservableError};
    use bloq_utils::boolean::{BooleanLimits, BooleanResourceError};
    use glam::ivec3;

    use crate::validate::InvalidActionError;
    use crate::{
        Action, Block, BlockGraph, BlockKind, CubeKind, Direction, Expr, MeasureTarget, Pipe,
    };
    use crate::{ActionDag, BlockGraphError, MeasurementObservable, PauliBasis, SelectiveKind};

    #[test]
    fn shared_aliases_stay_bounded_for_domain_and_sampling() {
        let target = ivec3(0, 0, 1);
        let mut actions = vec![Action::Measure {
            target: MeasureTarget::Node(ivec3(0, 0, 0)),
            name: "m".into(),
        }];
        let mut name = "m".to_owned();
        for index in 0..64 {
            let next = format!("a{index}");
            actions.push(Action::Let {
                name: next.clone(),
                expr: Expr::Binary(
                    crate::BinaryOp::Xor,
                    Box::new(Expr::Var(name.clone())),
                    Box::new(Expr::Var(name)),
                ),
            });
            name = next;
        }
        actions.push(Action::Resolve {
            target,
            condition: Expr::Var(name),
        });
        let dag = ActionDag::from_actions(&actions);
        assert_eq!(
            dag.resolve_value_domain(&[target]).unwrap().values(),
            &[vec![false]]
        );
        assert!(!dag.sample_resolve_values(&[target], &mut rand::rng())[&target]);
    }

    #[test]
    fn resolve_domain_preserves_shared_control_correlation() {
        let first = ivec3(0, 0, 1);
        let second = ivec3(1, 0, 1);
        let dag = ActionDag::from_actions(&[
            Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m".into(),
            },
            Action::Let {
                name: "not_m".into(),
                expr: Expr::Not(Box::new(Expr::Var("m".into()))),
            },
            Action::Resolve {
                target: first,
                condition: Expr::Var("m".into()),
            },
            Action::Resolve {
                target: second,
                condition: Expr::Var("not_m".into()),
            },
        ]);

        let domain = dag.resolve_value_domain(&[first, second]).unwrap();
        assert_eq!(domain.values(), &[vec![false, true], vec![true, false]]);
    }

    #[test]
    fn branch_domain_preserves_shared_control_correlation() {
        let first = ivec3(0, 0, 1);
        let second = ivec3(1, 0, 1);
        let dag = ActionDag::from_actions(&[
            Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m".into(),
            },
            Action::Let {
                name: "not_m".into(),
                expr: Expr::Not(Box::new(Expr::Var("m".into()))),
            },
            Action::Branch {
                target: first,
                condition: Expr::Var("m".into()),
            },
            Action::Branch {
                target: second,
                condition: Expr::Var("not_m".into()),
            },
        ]);

        // The shared control makes the branch domain the two anti-correlated
        // tuples only; (false, false) must not appear.
        let domain = dag
            .branch_values_up_to(&[first, second], usize::MAX)
            .unwrap();
        assert_eq!(domain, [vec![false, true], vec![true, false]]);
    }

    #[test]
    fn symbolic_domain_matches_a_small_truth_table() {
        let targets = [ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(2, 0, 1)];
        let actions = [
            Action::Let {
                name: "parity".into(),
                expr: Expr::Binary(
                    crate::BinaryOp::Xor,
                    Box::new(Expr::Var("a".into())),
                    Box::new(Expr::Var("b".into())),
                ),
            },
            Action::Resolve {
                target: targets[0],
                condition: Expr::Var("parity".into()),
            },
            Action::Resolve {
                target: targets[1],
                condition: Expr::Binary(
                    crate::BinaryOp::And,
                    Box::new(Expr::Var("parity".into())),
                    Box::new(Expr::Var("c".into())),
                ),
            },
            Action::Resolve {
                target: targets[2],
                condition: Expr::Binary(
                    crate::BinaryOp::Or,
                    Box::new(Expr::Not(Box::new(Expr::Var("a".into())))),
                    Box::new(Expr::Var("c".into())),
                ),
            },
        ];
        let dag = ActionDag::from_actions(&actions);
        let aliases = dag.let_bindings();
        let exprs = targets.map(|target| dag.resolve_expr(target).unwrap());
        let mut expected = std::collections::BTreeSet::new();
        for assignment in 0..8 {
            expected.insert(
                exprs
                    .iter()
                    .map(|expr| {
                        super::evaluate_expr_with(
                            expr,
                            &aliases,
                            &|name| {
                                let index = match name {
                                    "a" => 0,
                                    "b" => 1,
                                    "c" => 2,
                                    _ => unreachable!("all input variables are listed"),
                                };
                                assignment >> index & 1 == 1
                            },
                            &mut std::collections::HashMap::new(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let expected = expected.into_iter().collect::<Vec<_>>();
        let domain = dag.resolve_value_domain(&targets).unwrap();
        assert_eq!(domain.values(), expected);
        let bounded = dag
            .resolve_value_domain_bounded(&targets, expected.len())
            .expect("the exact domain fits its measured size");
        assert_eq!(bounded.values(), expected);
        assert_eq!(
            dag.resolve_value_domain_bounded(&targets, expected.len() - 1),
            Err(ResolveDomainError::TooManyValues {
                observed: expected.len()
            })
        );
        for assignment in 0..8 {
            let values = (0..3)
                .map(|index| assignment >> index & 1 == 1)
                .collect::<Vec<_>>();
            assert_eq!(
                dag.resolve_values_are_reachable(&targets, &values).unwrap(),
                domain.contains(&values)
            );
        }
    }

    #[test]
    fn domain_limits_distinguish_alias_work_from_value_count() {
        let target = ivec3(0, 0, 1);
        let mut actions = (0..256)
            .map(|index| Action::Let {
                name: format!("a{index}"),
                expr: Expr::Var(if index == 0 {
                    "input".into()
                } else {
                    format!("a{}", index - 1)
                }),
            })
            .collect::<Vec<_>>();
        actions.push(Action::Resolve {
            target,
            condition: Expr::Var("a255".into()),
        });
        let dag = ActionDag::from_actions(&actions);
        let query = |values, limits| {
            dag.resolve_value_domain_bounded_with_limits(&[target], values, limits)
        };
        assert!(matches!(
            query(
                2,
                BooleanLimits {
                    max_steps: 8,
                    ..BooleanLimits::DEFAULT
                }
            ),
            Err(ResolveDomainError::Resource(BooleanResourceError {
                resource: "Boolean work steps",
                observed: 9,
                limit: 8,
            }))
        ));
        assert!(matches!(
            query(
                2,
                BooleanLimits {
                    max_nodes: 0,
                    ..BooleanLimits::DEFAULT
                }
            ),
            Err(ResolveDomainError::Resource(BooleanResourceError {
                resource: "Boolean nodes",
                observed: 1,
                limit: 0,
            }))
        ));
        assert_eq!(
            query(1, BooleanLimits::DEFAULT),
            Err(ResolveDomainError::TooManyValues { observed: 2 })
        );
        let domain = query(2, BooleanLimits::DEFAULT).unwrap();
        assert_eq!(domain.values(), &[vec![false], vec![true]]);
        assert_eq!(domain, query(2, BooleanLimits::UNLIMITED).unwrap());
    }

    #[test]
    fn symbolic_reachability_handles_wide_independent_controls() {
        let count = usize::BITS as usize + 1;
        let targets = (0..count)
            .map(|index| ivec3(index as i32, 0, 0))
            .collect::<Vec<_>>();
        let actions = targets
            .iter()
            .enumerate()
            .map(|(index, &target)| Action::Resolve {
                target,
                condition: Expr::Var(format!("m{index}")),
            })
            .collect::<Vec<_>>();
        let dag = ActionDag::from_actions(&actions);

        assert!(
            dag.resolve_values_are_reachable(&targets, &vec![false; count])
                .unwrap()
        );
    }

    #[test]
    fn measure_target_with_shift_updates_node_and_edge_geometry() {
        use crate::{Action, Direction, MeasureTarget};
        use glam::ivec3;

        let shifted_node = Action::Measure {
            target: MeasureTarget::Node(ivec3(0, 0, 0)),
            name: "m_node".into(),
        }
        .with_shift(ivec3(2, 3, 4));
        assert_eq!(
            shifted_node,
            Action::Measure {
                target: MeasureTarget::Node(ivec3(2, 3, 4)),
                name: "m_node".into(),
            }
        );

        let shifted_edge = Action::Measure {
            target: MeasureTarget::Edge {
                src: ivec3(1, 0, 0),
                dir: Direction::XPLUS,
            },
            name: "m_edge".into(),
        }
        .with_shift(ivec3(0, 5, -1));
        assert_eq!(
            shifted_edge,
            Action::Measure {
                target: MeasureTarget::Edge {
                    src: ivec3(1, 5, -1),
                    dir: Direction::XPLUS,
                },
                name: "m_edge".into(),
            }
        );
    }

    #[test]
    fn measurement_metadata_requires_an_existing_measurement() {
        use crate::{Action, ActionDag, MeasureTarget};
        use glam::ivec3;

        let mut dag = ActionDag::from_actions(&[Action::Measure {
            target: MeasureTarget::Node(ivec3(0, 0, 0)),
            name: "m0".into(),
        }]);
        let observable = MeasurementObservable::Selective(SelectiveKind::XY);
        dag.set_measurement_observable(0, observable).unwrap();
        assert_eq!(
            dag.node_by_ordinal(0).unwrap().measurement,
            Some(observable)
        );
        assert!(matches!(
            dag.set_measurement_observable(1, observable),
            Err(SetMeasurementObservableError::OrdinalOutOfRange { ordinal: 1, .. })
        ));

        let mut dag = ActionDag::from_actions(&[Action::Let {
            name: "a".into(),
            expr: Expr::Var("m0".into()),
        }]);
        assert!(matches!(
            dag.set_measurement_observable(0, MeasurementObservable::Concrete(PauliBasis::Z)),
            Err(SetMeasurementObservableError::NotMeasurementNode { ordinal: 0 })
        ));
    }

    #[test]
    fn action_dag_adds_dependency_edges_for_variable_uses() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m0".into(),
            })
            .unwrap();
        graph
            .add_action(Action::Let {
                name: "alias".into(),
                expr: Expr::Var("m0".into()),
            })
            .unwrap();
        graph
            .add_action(Action::DiscardIf(Expr::Var("alias".into())))
            .unwrap();

        let dag = graph.action_graph();
        let ordinals = dag
            .ordered_nodes()
            .map(|node| node.ordinal)
            .collect::<Vec<_>>();
        assert_eq!(ordinals, vec![0, 1, 2]);

        let alias_node = dag.node_by_ordinal(1).unwrap();
        let discard_node = dag.node_by_ordinal(2).unwrap();
        assert!(dag.depends_on(alias_node, 0));
        assert!(dag.depends_on(discard_node, 1));
    }

    #[test]
    fn action_dag_requires_resolve_before_selective_measurement() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: XZX [0,0,1]\n  2: XY [0,0,2]\n  3: ZXZ [1,0,1]\n  [0,0,0] -> +Z\n  [0,0,1] -> +Z\n  [0,0,1] -> +X\n\n  resolve 2 if m0\n  m_sel = measure 2\n  m0 = measure 1 -> +X\n",
        )
        .unwrap();

        let dag = graph.action_graph();
        let resolve = dag
            .ordered_nodes()
            .find(|node| matches!(node.action, Action::Resolve { target, .. } if target == glam::ivec3(0, 0, 2)))
            .unwrap();
        let measure = dag
            .ordered_nodes()
            .find(|node| matches!(&node.action, Action::Measure { name, .. } if name == "m_sel"))
            .unwrap();
        assert!(dag.depends_on(measure, resolve.ordinal));
    }

    #[test]
    fn action_dag_rejects_variable_redefinition() {
        let actions = vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m0".into(),
            },
            Action::Let {
                name: "a".into(),
                expr: Expr::Var("m0".into()),
            },
            Action::Let {
                name: "a".into(),
                expr: Expr::Var("m0".into()),
            },
        ];

        let mut dag = ActionDag::from_actions(&actions);
        let err = dag.validate(None).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::VariableRedefinition(ref n))
                    if n == "a"
            ),
            "expected VariableRedefinition, got: {err:?}",
        );
    }

    #[test]
    fn action_dag_rejects_duplicate_resolve_target() {
        let actions = vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, -1)),
                name: "m0".into(),
            },
            Action::Resolve {
                target: glam::ivec3(0, 0, 0),
                condition: Expr::Var("m0".into()),
            },
            Action::Resolve {
                target: glam::ivec3(0, 0, 0),
                condition: Expr::Var("m0".into()),
            },
        ];

        let mut dag = ActionDag::from_actions(&actions);
        let err = dag.validate(None).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::DuplicateResolveTarget(target))
                    if target == glam::ivec3(0, 0, 0)
            ),
            "expected DuplicateResolveTarget, got: {err:?}",
        );
    }

    #[test]
    fn action_dag_rejects_duplicate_measurement_target() {
        let actions = vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m0".into(),
            },
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m1".into(),
            },
        ];

        let mut dag = ActionDag::from_actions(&actions);
        let err = dag.validate(None).unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(InvalidActionError::DuplicateMeasurementTarget(..))
            ),
            "expected DuplicateMeasurementTarget, got: {err:?}",
        );
    }

    #[test]
    fn action_dag_rejects_missing_resolve_for_selective_block() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_block(Block::new(
            glam::ivec3(0, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(glam::ivec3(0, 0, -1), Direction::ZPLUS));

        let err = ActionDag::from_actions(&[])
            .validate(Some(&graph))
            .unwrap_err();
        assert!(
            matches!(
                err,
                BlockGraphError::InvalidAction(
                    InvalidActionError::MissingResolveForSelective { target }
                ) if target == glam::ivec3(0, 0, 0)
            ),
            "expected MissingResolveForSelective, got: {err:?}",
        );
    }

    #[test]
    fn action_dag_rejects_dependency_cycle() {
        let actions = vec![
            Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m".into(),
            },
            Action::Resolve {
                target: glam::ivec3(0, 0, 0),
                condition: Expr::Var("m".into()),
            },
        ];

        let mut dag = ActionDag::from_actions(&actions);
        let err = dag.validate(None).unwrap_err();
        match err {
            BlockGraphError::InvalidAction(InvalidActionError::DependencyCycle { ordinal }) => {
                assert!(ordinal == 0 || ordinal == 1, "got ordinal: {ordinal}");
            }
            other => panic!("expected DependencyCycle, got: {other:?}"),
        }
    }
}
