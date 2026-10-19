//! Resolve the two cuts at which an observable's folded `Output`-face boundary
//! Paulis must still be *live* boundaries.
//!
//! # Why this analysis exists
//!
//! WF-7/WF-18 guarantee that every *bit-carrying* record an observable folds
//! exists before it is read. But a boundary operator carries **no bit**: it is
//! a declarative binding of an `Output`-face Pauli, with no schedule or
//! consumption check of its own. So the physical face an observable folds can be
//! reopened or consumed by a later-scheduled quantum node while the read still
//! refers to it — and the read would silently return the wrong-time operator.
//! `Input` faces are exempt throughout: they are source endpoints closed by
//! composition and are never measured (see `Program::observable_value`).
//!
//! Two reads need protecting, one cut apart:
//!
//! * [`plan_early_output_reads`] — the *terminal* read. A top-level
//!   `Observable`'s faces are measured on the final simulator state, so the cut
//!   is the position of the node owning the face's instance. A face a later node
//!   consumes cannot wait that long, so it is read at its own cut instead: the
//!   plan names the node each such read must go in front of.
//! * [`guard_decode_observable_closure`] — the *decode* read, one cut earlier. A
//!   complete `Observable` requests its solve as of its own scheduled
//!   position, at any nesting level (observables are scope-local). The sole
//!   exception is an attempt-local RUS decode, whose parent continuation may
//!   legitimately consume the accepted RUS output after the GAP cut.
//!
//! Both share the schedule and footprint model below: [`execution_schedule`]
//! flattens the program into one absolute emit order with per-node coordinate
//! footprints, and [`later_touchers`] finds nodes after a cut that act on a
//! face's support on paths that can coexist with it.
//!
//! Neither pass mutates the program; both only read the graph.
//!
//! A consumed `Output` face is a real shape: a layout column can host two
//! program outputs in sequence, the second patch reusing the tiles the first
//! vacated (`and_4t`). Hence the early read. The *decode* cut has no fixture —
//! every corpus `Output` face is closed before its decode — so it stays a
//! guard, turning a silent wrong answer into a typed error.

use bloq_ir::lowering::{InstanceBoundaryOperator, TemplateInstanceId};
use bloq_ir::{
    Bloq, BloqEdge, BloqNodeId, BodySelector, BoundaryFace, ClassicalNode, LevelPath, RegionNode,
    SubGraph, ValueRef,
};
use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{ExecError, MAX_PHYSICAL_BOUNDARY_BINDINGS};

// ============================================================
// Shared schedule / footprint model
// ============================================================

/// One node in recursive execution order, preserving its region scope.
pub(super) struct ScheduledNode {
    pub(super) id: BloqNodeId,
    pub(super) scope: LevelPath,
    pub(super) order: usize,
    pub(super) instances: Vec<TemplateInstanceId>,
    pub(super) footprint: FxHashSet<IVec2>,
    members: Vec<(Option<ValueRef>, FxHashSet<IVec2>)>,
}

pub(super) fn execution_schedule(
    bloq: &Bloq,
    level: &SubGraph,
) -> Result<Vec<ScheduledNode>, ExecError> {
    fn visit(
        bloq: &Bloq,
        level: &SubGraph,
        scope: &LevelPath,
        next: &mut usize,
        out: &mut Vec<ScheduledNode>,
    ) -> Result<(), ExecError> {
        for id in level.deterministic_emit_order()? {
            let node = &level[id];
            let order = *next;
            *next += 1;
            let mut instances = Vec::new();
            let mut footprint = FxHashSet::default();
            let mut members = Vec::new();
            if let Some(quantum) = node.try_quantum() {
                instances.extend(quantum.instances.iter().map(|instance| instance.id));
                footprint = bloq.node_qubits(node)?;
                for instance in &quantum.instances {
                    let guard = quantum
                        .guards
                        .iter()
                        .find(|guard| guard.instances.contains(&instance.id))
                        .map(|guard| {
                            level
                                .value_inputs(id)
                                .find(|input| input.slot == guard.input)
                                .ok_or(ExecError::MalformedGraph(
                                    "quantum guard has no value input",
                                ))
                                .map(|input| ValueRef {
                                    node: input.producer,
                                    output: input.output.expect("value port"),
                                })
                        })
                        .transpose()?;
                    let qubits = bloq
                        .templates()
                        .get(instance.template_id)
                        .ok_or(ExecError::MalformedGraph(
                            "quantum instance has no template",
                        ))?
                        .qubits()
                        .iter()
                        .map(|&qubit| qubit + instance.offset)
                        .collect();
                    members.push((guard, qubits));
                }
            }
            out.push(ScheduledNode {
                id,
                scope: scope.clone(),
                order,
                instances,
                footprint,
                members,
            });
            if let Some(region) = node.try_region() {
                for (selector, body) in region.bodies() {
                    visit(bloq, body, &scope.child(id, selector), next, out)?;
                }
            }
        }
        Ok(())
    }

    let mut schedule = Vec::new();
    visit(bloq, level, &LevelPath::default(), &mut 0, &mut schedule)?;
    Ok(schedule)
}

/// Identity of a boundary binding, including its region gate. Equal physical
/// operators in different gated nodes are distinct observable contributions.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct OutputBinding {
    pub(super) scope: LevelPath,
    node: BloqNodeId,
    index: usize,
    pub(super) guards: Vec<(LevelPath, ValueRef)>,
}

impl OutputBinding {
    pub(super) fn operator<'a>(&self, bloq: &'a Bloq) -> &'a InstanceBoundaryOperator {
        let level = bloq.level_at(&self.scope).expect("binding scope exists");
        &level[self.node]
            .try_classical()
            .expect("binding is classical")
            .operators()[self.index]
    }
}

/// Whether a producer can export an Output face. Memoizing the graph nodes
/// before path expansion keeps Input-only shared recipes linear in graph size.
fn can_export_output(
    level: &SubGraph,
    scope: &LevelPath,
    producer: BloqNodeId,
    reachable: &mut FxHashMap<(LevelPath, BloqNodeId), bool>,
) -> Result<bool, ExecError> {
    let root = (scope.clone(), producer);
    let mut stack = vec![(level, scope.clone(), producer, false)];
    let mut active = FxHashSet::default();
    while let Some((level, scope, producer, finish)) = stack.pop() {
        let key = (scope.clone(), producer);
        if reachable.contains_key(&key) {
            continue;
        }
        let node = level.node(producer).ok_or(ExecError::MalformedGraph(
            "boundary export names missing producer",
        ))?;
        let children = match (node.try_classical(), node.try_region()) {
            (Some(ClassicalNode::Observable { .. }), _) => level
                .data_inputs(producer)
                .filter(|input| input.output.is_none())
                .map(|input| (level, scope.clone(), input.producer))
                .collect::<Vec<_>>(),
            (_, Some(region)) => region
                .bodies()
                .flat_map(|(selector, body)| {
                    let child_scope = scope.child(producer, selector);
                    body.boundary_outputs()
                        .iter()
                        .map(move |&child| (body, child_scope.clone(), child))
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        if !finish {
            if !active.insert(key.clone()) {
                return Err(ExecError::MalformedGraph("cyclic boundary recipe"));
            }
            stack.push((level, scope, producer, true));
            stack.extend(
                children
                    .into_iter()
                    .rev()
                    .map(|(level, scope, child)| (level, scope, child, false)),
            );
        } else {
            let output = match node.try_classical() {
                Some(ClassicalNode::Observable { operators, .. }) => {
                    operators
                        .iter()
                        .any(|operator| operator.face == BoundaryFace::Output)
                        || children.iter().any(|(_, scope, child)| {
                            reachable.get(&(scope.clone(), *child)) == Some(&true)
                        })
                }
                _ => children.iter().any(|(_, scope, child)| {
                    reachable.get(&(scope.clone(), *child)) == Some(&true)
                }),
            };
            reachable.insert(key.clone(), output);
            active.remove(&key);
        }
    }
    Ok(reachable[&root])
}

/// Collect boundary operators a value producer can export. Closed constant
/// predicates narrow the set statically; non-constant regions conservatively
/// include every body that can export. Scheduling still recurses through all
/// bodies independently of this value analysis. Callers first run
/// `execution_schedule`, whose topological walk rejects cycles.
fn collect_exported_boundary_operators<'a>(
    level: &'a SubGraph,
    scope: &LevelPath,
    producer: BloqNodeId,
    guards: &[(LevelPath, ValueRef)],
    out: &mut Vec<(OutputBinding, &'a InstanceBoundaryOperator)>,
    expanded: &mut usize,
    reachable: &mut FxHashMap<(LevelPath, BloqNodeId), bool>,
) -> Result<(), ExecError> {
    let mut stack = vec![(level, scope.clone(), producer, guards.to_vec(), false)];
    while let Some((level, scope, producer, mut guards, nested_recipe)) = stack.pop() {
        if !can_export_output(level, &scope, producer, reachable)? {
            continue;
        }
        if nested_recipe {
            *expanded = expanded.saturating_add(1);
            if *expanded > MAX_PHYSICAL_BOUNDARY_BINDINGS {
                return Err(ExecError::BoundaryBindingExpansionLimit {
                    limit: MAX_PHYSICAL_BOUNDARY_BINDINGS,
                });
            }
        }
        let node = level.node(producer).ok_or(ExecError::MalformedGraph(
            "boundary export names missing producer",
        ))?;
        if let Some(slot) = node.activation {
            guards.push((
                scope.clone(),
                level
                    .value_inputs(producer)
                    .find(|input| input.slot == slot)
                    .map(|input| ValueRef {
                        node: input.producer,
                        output: input.output.expect("value port"),
                    })
                    .ok_or(ExecError::MalformedGraph("activation has no value input"))?,
            ));
        }
        guards.sort_unstable();
        guards.dedup();
        match (node.try_classical(), node.try_region()) {
            (Some(ClassicalNode::Observable { operators, .. }), _) => {
                if nested_recipe {
                    *expanded = expanded.saturating_add(
                        operators
                            .iter()
                            .filter(|operator| operator.face == BoundaryFace::Output)
                            .count(),
                    );
                    if *expanded > MAX_PHYSICAL_BOUNDARY_BINDINGS {
                        return Err(ExecError::BoundaryBindingExpansionLimit {
                            limit: MAX_PHYSICAL_BOUNDARY_BINDINGS,
                        });
                    }
                }
                out.extend(
                    operators
                        .iter()
                        .enumerate()
                        .filter(|(_, operator)| operator.face == BoundaryFace::Output)
                        .map(|(index, operator)| {
                            (
                                OutputBinding {
                                    scope: scope.clone(),
                                    node: producer,
                                    index,
                                    guards: guards.clone(),
                                },
                                operator,
                            )
                        }),
                );
                let children = level
                    .data_inputs(producer)
                    .filter(|input| input.output.is_none())
                    .map(|input| input.producer)
                    .collect::<Vec<_>>();
                stack.extend(
                    children
                        .into_iter()
                        .rev()
                        .map(|child| (level, scope.clone(), child, guards.clone(), true)),
                );
            }
            (_, Some(region)) => {
                let RegionNode::RepeatUntilSuccess { body, .. } = region;
                let child_scope = scope.child(producer, BodySelector::Body);
                stack.extend(body.boundary_outputs().iter().rev().map(|&child| {
                    (
                        body,
                        child_scope.clone(),
                        child,
                        guards.clone(),
                        nested_recipe,
                    )
                }));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The live `Output`-face operators `observable` folds, each with the scope it
/// was exported from. `Input` faces are dropped here — neither read measures
/// them, so neither can be invalidated by reopening one.
fn folded_output_faces<'a>(
    level: &'a SubGraph,
    scope: &LevelPath,
    observable: BloqNodeId,
    reachable: &mut FxHashMap<(LevelPath, BloqNodeId), bool>,
) -> Result<Vec<(OutputBinding, &'a InstanceBoundaryOperator)>, ExecError> {
    let mut operators = Vec::new();
    let mut expanded = 0;
    let guards = level
        .node(observable)
        .ok_or(ExecError::MalformedGraph("observable node is missing"))?
        .activation
        .map(|slot| {
            level
                .value_inputs(observable)
                .find(|input| input.slot == slot)
                .ok_or(ExecError::MalformedGraph(
                    "observable activation has no value input",
                ))
                .map(|input| {
                    (
                        scope.clone(),
                        ValueRef {
                            node: input.producer,
                            output: input.output.expect("value port"),
                        },
                    )
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    operators.extend(
        level[observable]
            .try_classical()
            .expect("observable is classical")
            .operators()
            .iter()
            .enumerate()
            .filter(|(_, operator)| operator.face == BoundaryFace::Output)
            .map(|(index, operator)| {
                (
                    OutputBinding {
                        scope: scope.clone(),
                        node: observable,
                        index,
                        guards: guards.clone(),
                    },
                    operator,
                )
            }),
    );
    for input in level
        .data_inputs(observable)
        .filter(|input| input.output.is_none())
    {
        collect_exported_boundary_operators(
            level,
            scope,
            input.producer,
            &guards,
            &mut operators,
            &mut expanded,
            reachable,
        )?;
    }
    Ok(operators)
}

/// Nodes scheduled after `cut` that act on `operator`'s support from a scope
/// that can coexist with `export_scope`.
///
/// `confine`, when set, restricts the search to that scope's subtree: an
/// attempt-local RUS decode is only threatened from inside its own body, since
/// the parent continuation runs after the attempt is accepted.
fn later_touchers<'a>(
    bloq: &'a Bloq,
    schedule: &'a [ScheduledNode],
    cut: usize,
    binding: &'a OutputBinding,
    operator: &'a InstanceBoundaryOperator,
    confine: Option<&'a LevelPath>,
    limit: usize,
) -> Result<Vec<&'a ScheduledNode>, ExecError> {
    let mut touchers = Vec::new();
    for scheduled in schedule {
        if scheduled.order <= cut
            || confine
                .is_some_and(|scope| !scheduled.scope.segments().starts_with(scope.segments()))
        {
            continue;
        }
        for (guard, qubits) in &scheduled.members {
            if !operator
                .operator
                .iter()
                .any(|(coord, _)| qubits.contains(coord))
            {
                continue;
            }
            let mut predicates = std::collections::BTreeMap::<_, Vec<_>>::new();
            for (scope, predicate) in &binding.guards {
                predicates.entry(scope).or_default().push(*predicate);
            }
            if let Some(guard) = guard {
                predicates.entry(&scheduled.scope).or_default().push(*guard);
            }
            let mut coexist = true;
            for (scope, roots) in predicates {
                let level = bloq.level_at(scope).ok_or(ExecError::MalformedGraph(
                    "scheduled predicate scope is missing",
                ))?;
                if !level
                    .predicates_can_coexist_values(&roots)
                    .map_err(ExecError::InvalidCircuit)?
                {
                    coexist = false;
                    break;
                }
            }
            if coexist {
                touchers.push(scheduled);
                if touchers.len() == limit {
                    return Ok(touchers);
                }
                break;
            }
        }
    }
    Ok(touchers)
}

// ============================================================
// Terminal read
// ============================================================

/// Joint boundary factors of one observable, read before a later node consumes
/// their output cut. Fragment identities preserve conditional binding gates.
pub(super) struct EarlyOutputRead {
    /// The node whose execution destroys the face. The read is taken just before
    /// it runs, in that node's own scope — which may be a region body, so the
    /// scope is part of the key.
    pub(super) before: (LevelPath, BloqNodeId),
    pub(super) observable: BloqNodeId,
    pub(super) bindings: Vec<OutputBinding>,
    /// Other factors are future-owned or gated separately.
    pub(super) incomplete: bool,
}

/// Plan the `Output`-face reads that cannot wait for the terminal state.
///
/// `Program::observable_value` measures a top-level `Observable`'s faces on the
/// TERMINAL simulator state, so the cut is the position of the node owning the
/// face's instance. Any node scheduled after that cut which acts on the face's
/// support destroys it; the plan moves the read to the earliest required cut
/// on each possible path. Faces nothing touches are absent and keep the
/// terminal read. See the [module docs](self).
pub(super) fn plan_early_output_reads(bloq: &Bloq) -> Result<Vec<EarlyOutputRead>, ExecError> {
    let schedule = execution_schedule(bloq, bloq.top())?;
    let mut reachable = FxHashMap::default();
    let mut instance_position: FxHashMap<TemplateInstanceId, usize> = FxHashMap::default();
    for scheduled in &schedule {
        for &instance in &scheduled.instances {
            instance_position.insert(instance, scheduled.order);
        }
    }

    let mut candidates = Vec::new();
    for (obs_id, node) in bloq.top().nodes() {
        if !matches!(
            node.try_classical(),
            Some(ClassicalNode::Observable { index: Some(_), .. })
        ) {
            continue;
        }
        let faces = folded_output_faces(bloq.top(), &LevelPath::default(), obs_id, &mut reachable)?;
        for (binding, operator) in &faces {
            let Some(&owner) = instance_position.get(&operator.instance) else {
                continue;
            };
            for later in
                later_touchers(bloq, &schedule, owner, binding, operator, None, usize::MAX)?
            {
                // Keep a jointly available observable together. Its factors may
                // anticommute with another observable's factors even when the
                // complete observables commute (for example Bell XX and ZZ).
                let bindings = faces
                    .iter()
                    .filter(|(other, face)| {
                        other.scope == binding.scope
                            && other.guards == binding.guards
                            && instance_position
                                .get(&face.instance)
                                .is_some_and(|&position| position < later.order)
                    })
                    .map(|(other, _)| other.clone())
                    .collect::<Vec<_>>();
                candidates.push((
                    later.order,
                    EarlyOutputRead {
                        before: (later.scope.clone(), later.id),
                        observable: obs_id,
                        incomplete: bindings.len() != faces.len(),
                        bindings,
                    },
                ));
            }
        }
    }

    // A later ancestor must not suppress an earlier nested cut.
    candidates.sort_by_key(|(order, _)| *order);
    let mut plan: Vec<EarlyOutputRead> = Vec::new();
    for (_, mut candidate) in candidates {
        // An earlier read in this scope or an ancestor already owns its
        // factors. Distinct nested cuts remain independently applicable.
        let count = candidate.bindings.len();
        candidate.bindings.retain(|binding| {
            !plan.iter().any(|read| {
                read.observable == candidate.observable
                    && read.bindings.contains(binding)
                    && candidate
                        .before
                        .0
                        .segments()
                        .starts_with(read.before.0.segments())
            })
        });
        candidate.incomplete |= candidate.bindings.len() != count;
        if !candidate.bindings.is_empty() {
            plan.push(candidate);
        }
    }
    Ok(plan)
}

/// Potential consumers of each logical output, identified by its owning
/// instance rather than its reusable coordinate support. A branch may have a
/// different first consumer on each arm; execution captures only once.
pub(super) fn logical_output_cuts(
    bloq: &Bloq,
) -> Result<Vec<Vec<(LevelPath, BloqNodeId)>>, ExecError> {
    let schedule = execution_schedule(bloq, bloq.top())?;
    bloq.logical_outputs()
        .iter()
        .map(|output| {
            let owner = schedule
                .iter()
                .find(|node| node.instances.contains(&output.instance))
                .ok_or(ExecError::MalformedGraph(
                    "logical output has no scheduled owner",
                ))?;
            Ok(schedule
                .iter()
                .filter(|node| {
                    node.order > owner.order
                        && output
                            .x
                            .iter()
                            .chain(output.z.iter())
                            .any(|(coord, _)| node.footprint.contains(coord))
                })
                .map(|node| (node.scope.clone(), node.id))
                .collect())
        })
        .collect()
}

// ============================================================
// Decode read
// ============================================================

/// Whether this complete observable's corrected result or completion is used.
/// A raw projection alone leaves the physical diagnostic independent of a solve.
pub(crate) fn needs_correction(
    level: &SubGraph,
    observable: BloqNodeId,
    implicit_demands: &FxHashSet<BloqNodeId>,
) -> bool {
    matches!(
        level.node(observable).and_then(|node| node.try_classical()),
        Some(ClassicalNode::Observable { index: Some(_), .. })
    ) && (implicit_demands.contains(&observable)
        || level
            .value_output()
            .is_some_and(|value| value.node == observable)
        || level
            .outgoing(observable)
            .any(|edge| matches!(edge.edge, BloqEdge::Value { .. } | BloqEdge::Order)))
}

/// Value uses stored outside ordinary edges and the explicit body result.
/// Collect once per graph level so guard discovery stays linear in edge count.
pub(crate) fn implicit_correction_demands(
    level: &SubGraph,
    restart_source: Option<ValueRef>,
) -> FxHashSet<BloqNodeId> {
    let mut demands = restart_source
        .into_iter()
        .map(|value| value.node)
        .collect::<FxHashSet<_>>();
    demands.extend(level.edges().filter_map(|edge| match edge.edge {
        BloqEdge::Quantum(quantum) => quantum.guard.map(|value| value.node),
        BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
    }));
    demands
}

/// Validate that no decoder-backed node's folded `Output` face is reopened by
/// a later-scheduled node. See the [module docs](self).
///
/// # Errors
/// [`ExecError::DecodeObservableNotClosed`] naming the offending decode, its
/// observable, the boundary operator's instance, and the reopening node.
pub(super) fn guard_decode_observable_closure(bloq: &Bloq) -> Result<(), ExecError> {
    let schedule = execution_schedule(bloq, bloq.top())?;
    let mut reachable = FxHashMap::default();
    let positions = schedule
        .iter()
        .map(|scheduled| ((scheduled.scope.clone(), scheduled.id), scheduled.order))
        .collect();
    check_level(
        bloq,
        bloq.top(),
        &LevelPath::default(),
        None,
        None,
        &schedule,
        &positions,
        &mut reachable,
    )
}

/// Run the closure check for one scope, then recurse into region bodies so a
/// complete Observable inside a RUS body is checked against *its own*
/// scope's schedule and footprints (scope-local observables).
#[expect(
    clippy::too_many_arguments,
    reason = "recursive closure analysis carries one region scope and its explicit demand"
)]
fn check_level(
    bloq: &Bloq,
    level: &SubGraph,
    scope: &LevelPath,
    active_rus_scope: Option<&LevelPath>,
    restart_source: Option<ValueRef>,
    schedule: &[ScheduledNode],
    positions: &FxHashMap<(LevelPath, BloqNodeId), usize>,
    reachable: &mut FxHashMap<(LevelPath, BloqNodeId), bool>,
) -> Result<(), ExecError> {
    let implicit_demands = implicit_correction_demands(level, restart_source);
    for obs_id in level.node_ids() {
        if !needs_correction(level, obs_id, &implicit_demands) {
            continue;
        }
        let decode_pos =
            *positions
                .get(&(scope.clone(), obs_id))
                .ok_or(ExecError::MalformedGraph(
                    "observable has no scheduled position",
                ))?;
        for (binding, operator) in folded_output_faces(level, scope, obs_id, reachable)? {
            if let Some(later) = later_touchers(
                bloq,
                schedule,
                decode_pos,
                &binding,
                operator,
                active_rus_scope,
                1,
            )?
            .into_iter()
            .next()
            {
                return Err(ExecError::DecodeObservableNotClosed {
                    decode: obs_id.0,
                    observable: obs_id.0,
                    instance: operator.instance.0,
                    later: later.id.0,
                });
            }
        }
    }

    for (region_id, node) in level.nodes() {
        if let Some(region) = node.try_region() {
            for (selector, body) in region.bodies() {
                let child = scope.child(region_id, selector);
                let RegionNode::RepeatUntilSuccess { restart_source, .. } = region;
                check_level(
                    bloq,
                    body,
                    &child,
                    Some(&child),
                    *restart_source,
                    schedule,
                    positions,
                    reachable,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::run_bloq;
    use super::super::test_support::{guarded_node, one_qubit_template, quantum_node};
    use super::*;
    use bloq_ir::ClassicalExpr;
    use bloq_ir::circuit::{GateType, Pauli as CircuitPauli};
    use bloq_ir::{BloqNode, TemplateId};
    use glam::ivec2;

    /// Where every fixture places its instance. The one-qubit template's
    /// footprint at this offset is exactly the folded face's support, so
    /// "touches the decoded face" is unambiguous.
    const SUPPORT: IVec2 = ivec2(5, 0);

    #[test]
    fn input_only_and_empty_recipe_diamond_skips_output_expansion() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(support_node(template, 0));
        let include = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![InstanceBoundaryOperator {
                face: BoundaryFace::Input,
                ..output_operator()
            }],
        )));
        bloq.add_edge(owner, include, bloq_ir::BloqEdge::Order);
        let empty = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let mut source = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        bloq.add_edge(include, source, BloqEdge::compose(0));
        bloq.add_edge(empty, source, BloqEdge::compose(1));
        for _ in 0..24 {
            let recipe = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                Vec::new(),
            )));
            bloq.add_edge(source, recipe, BloqEdge::compose(0));
            bloq.add_edge(source, recipe, BloqEdge::compose(1));
            source = recipe;
        }
        let bit = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(source, observable, BloqEdge::compose(0));
        bloq.add_edge(bit, observable, BloqEdge::value(1));
        let decode = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(observable, decode, BloqEdge::value(0));
        bloq.validate().unwrap();
        let mut reachable = FxHashMap::default();
        assert!(
            folded_output_faces(
                bloq.top(),
                &LevelPath::default(),
                observable,
                &mut reachable
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(reachable.len(), 27); // 2^24 Input paths, 27 composed producers.
        assert!(plan_early_output_reads(&bloq).unwrap().is_empty());
        guard_decode_observable_closure(&bloq).unwrap();
        assert_eq!(
            run_bloq(&bloq, 1, 7).unwrap().observables[0].per_shot,
            [true]
        );
        let dynamic = crate::lower(&bloq, &crate::LoweringConfig::default()).unwrap();
        let artifact = crate::runtime::run(&dynamic, crate::runtime::RuntimeConfig::default())
            .unwrap()
            .artifact;
        assert_eq!(artifact.observables[0].raw, Some(true));
        assert_eq!(artifact.observables[0].value, Some(true));
    }

    #[test]
    fn output_recipe_diamond_still_hits_typed_lowering_closure_work_limit() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(support_node(template, 0));
        let mut source = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![output_operator()],
        )));
        bloq.add_edge(owner, source, bloq_ir::BloqEdge::Order);
        for _ in 0..24 {
            let recipe = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                Vec::new(),
            )));
            bloq.add_edge(source, recipe, BloqEdge::compose(0));
            bloq.add_edge(source, recipe, BloqEdge::compose(1));
            source = recipe;
        }
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(source, observable, BloqEdge::compose(0));
        assert!(matches!(
            crate::lower(&bloq, &crate::LoweringConfig::default()),
            Err(crate::LowerError::Analysis(
                ExecError::BoundaryBindingExpansionLimit { .. }
            ))
        ));
    }

    /// A quantum node carrying `instance` of `template` at [`SUPPORT`].
    fn support_node(template: TemplateId, instance: u32) -> BloqNode {
        quantum_node(template, instance, SUPPORT)
    }

    /// The `Output`-face boundary operator on instance 0 that every fixture folds.
    fn output_operator() -> InstanceBoundaryOperator {
        InstanceBoundaryOperator {
            instance: TemplateInstanceId(0),
            face: BoundaryFace::Output,
            operator: [(SUPPORT, CircuitPauli::Z)].into_iter().collect(),
        }
    }

    /// The folded shape under test: fragment → complete `Observable(0)`,
    /// returning the binding and the observable's raw/solve cut identity.
    fn add_decoded_observable(level: &mut SubGraph) -> (BloqNodeId, BloqNodeId, BloqNodeId) {
        let include = level.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![output_operator()],
        )));
        let observable = level.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        level.add_edge(include, observable, BloqEdge::compose(0));
        let consumer = level.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        level.add_edge(observable, consumer, BloqEdge::value(0));
        (include, observable, observable)
    }

    /// Build a program with the decoded observable at top level plus a quantum
    /// node whose footprint touches the face support. `after` schedules that
    /// toucher after the decode cut (the unclosed shape) or before it (the
    /// corpus shape).
    ///
    /// Returns the graph plus the observable / decode / toucher node ids.
    fn scaffold(after: bool) -> (Bloq, BloqNodeId, BloqNodeId, BloqNodeId) {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let (_, observable, decode) = add_decoded_observable(bloq.top_mut());
        let toucher = bloq.add_node(support_node(template, 0));
        // An `Order` edge pins the toucher on the intended side of the cut, so
        // the assertion never rides on emit-order tie-breaking.
        if after {
            bloq.add_edge(decode, toucher, BloqEdge::Order);
        } else {
            bloq.add_edge(toucher, decode, BloqEdge::Order);
        }
        (bloq, observable, decode, toucher)
    }

    #[test]
    fn missing_quantum_guard_input_is_an_execution_error() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        bloq.add_node(guarded_node(support_node(template, 0), 0));
        assert!(matches!(
            execution_schedule(&bloq, bloq.top()),
            Err(ExecError::MalformedGraph(
                "quantum guard has no value input"
            ))
        ));
    }

    /// A quantum node reopening the decoded `Output` face *after* the cut is
    /// rejected with the typed error naming decode, observable, instance, node.
    #[test]
    fn later_node_reopening_decoded_output_face_errors() {
        let (bloq, observable, decode, toucher) = scaffold(true);
        let err = guard_decode_observable_closure(&bloq)
            .expect_err("a later node reopens the decoded face");
        assert!(
            matches!(
                err,
                ExecError::DecodeObservableNotClosed {
                    decode: d,
                    observable: o,
                    instance: 0,
                    later: l,
                } if d == decode.0 && o == observable.0 && l == toucher.0
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn direct_observable_operator_retains_its_owner_and_solve_cut() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(support_node(template, 0));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: Vec::new(),
            operators: vec![output_operator()],
        }));
        let later = bloq.add_node(support_node(template, 1));
        bloq.add_edge(owner, observable, BloqEdge::Order);
        bloq.add_edge(observable, later, BloqEdge::Order);
        let faces = folded_output_faces(
            bloq.top(),
            &LevelPath::default(),
            observable,
            &mut FxHashMap::default(),
        )
        .unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].0.operator(&bloq), &output_operator());
        assert!(matches!(
            guard_decode_observable_closure(&bloq),
            Err(ExecError::DecodeObservableNotClosed {
                decode,
                observable: observed,
                instance: 0,
                later: touched,
            }) if decode == observable.0 && observed == observable.0 && touched == later.0
        ));
    }

    #[test]
    fn observable_restart_source_requires_closed_support_without_value_consumers() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let mut body = SubGraph::new();
        let owner = body.add_node(support_node(template, 0));
        let observable = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: Vec::new(),
            operators: vec![output_operator()],
        }));
        let later = body.add_node(support_node(template, 1));
        body.add_edge(owner, observable, BloqEdge::Order);
        body.add_edge(owner, later, BloqEdge::Order);
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(observable.into()),
        }));
        assert!(matches!(
            guard_decode_observable_closure(&bloq),
            Err(ExecError::DecodeObservableNotClosed {
                observable: checked,
                later: touched,
                ..
            }) if checked == observable.0 && touched == later.0
        ));
    }

    #[test]
    fn declared_seam_guard_requests_correction_through_a_raw_only_path() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(support_node(template, 0));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: Vec::new(),
            operators: vec![output_operator()],
        }));
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let later = bloq.add_node(support_node(template, 1));
        bloq.add_edge(owner, observable, BloqEdge::Order);
        bloq.add_edge(observable, raw, BloqEdge::compose(0));
        bloq.add_edge(raw, later, BloqEdge::Order);
        assert!(!needs_correction(
            bloq.top(),
            observable,
            &FxHashSet::default()
        ));
        bloq.add_edge(
            owner,
            later,
            BloqEdge::Quantum(Box::new(bloq_ir::QuantumEdge {
                pipes: Vec::new(),
                guard: Some(observable.into()),
            })),
        );
        bloq.validate().unwrap();
        let demands = implicit_correction_demands(bloq.top(), None);
        assert!(needs_correction(bloq.top(), observable, &demands));
        assert!(matches!(
            guard_decode_observable_closure(&bloq),
            Err(ExecError::DecodeObservableNotClosed {
                observable: checked,
                later: touched,
                ..
            }) if checked == observable.0 && touched == later.0
        ));
    }

    /// The same touching node scheduled *before* the decode is the corpus shape
    /// (produce the readout patch, decode it, nothing reopens it) and validates.
    /// Proves the guard is schedule-aware, not mere face-presence.
    #[test]
    fn node_touching_face_before_decode_is_closed() {
        let (bloq, ..) = scaffold(false);
        guard_decode_observable_closure(&bloq).expect("face closed before the cut");
    }

    #[test]
    fn nested_decode_checks_program_global_ancestor_instance() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let owner = bloq.add_node(support_node(template, 0));

        let mut body = SubGraph::new();
        let (_, _, decode) = add_decoded_observable(&mut body);
        let later = body.add_node(support_node(template, 1));
        body.add_edge(decode, later, BloqEdge::Order);
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));
        bloq.add_edge(owner, region, BloqEdge::Order);

        assert!(matches!(
            guard_decode_observable_closure(&bloq),
            Err(ExecError::DecodeObservableNotClosed {
                instance: 0,
                later: id,
                ..
            }) if id == later.0
        ));
    }

    #[test]
    fn rus_body_decode_allows_parent_continuation() {
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let mut body = SubGraph::new();
        let owner = body.add_node(support_node(template, 0));
        let (include, _, decode) = add_decoded_observable(&mut body);
        body.add_edge(owner, include, BloqEdge::Order);
        let restart = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: bloq_ir::ClassicalExpr::Const(false),
        }));
        body.add_edge(decode, restart, BloqEdge::Order);
        let region = bloq.add_node(BloqNode::region(bloq_ir::RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: bloq_ir::ClassicalExpr::Const(false),
            restart_source: Some(restart.into()),
        }));
        let later = bloq.add_node(support_node(template, 1));
        bloq.add_edge(region, later, BloqEdge::Order);

        guard_decode_observable_closure(&bloq)
            .expect("the parent continuation starts after the accepted RUS attempt");
    }

    #[test]
    fn t_gate_gallery_program_validates() {
        let case = bloq_test::CompileReadyCase::from_test_case(
            bloq_test::find_test_case("t_gate[open]").expect("native T gallery case"),
        )
        .expect("gallery case builds");
        let mut bloq = bloq_compile::CompileContext::new(
            bloq_compile::CompileConfig::new(3).with_prepare_t_with_mpps(true),
        )
        .compile(&case.graph)
        .expect("ideal-source T compiles")
        .bloq;
        bloq.flatten().expect("flatten native T");

        assert!(
            bloq.levels()
                .any(|(_, level)| level.nodes().any(|(_, node)| matches!(
                    node.try_classical(),
                    Some(ClassicalNode::Observable { .. })
                ))),
            "native T must contain observables"
        );
        guard_decode_observable_closure(&bloq).expect("native T closes every decoded face");
    }

    // ---- terminal read: guard_consumed_output_faces ----

    fn constant_untaken_include_program() -> Bloq {
        let coord = IVec2::ZERO;
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let quantum = |instance| quantum_node(template, instance, IVec2::ZERO);
        let owner = bloq.add_node(quantum(0));
        let selector = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let mut fragment = BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: [(coord, CircuitPauli::Z)].into_iter().collect(),
            }],
        ));
        fragment.activation = Some(0);
        let region = bloq.add_node(fragment);
        bloq.add_edge(selector, region, BloqEdge::value(0));
        let later = bloq.add_node(quantum(1));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(owner, region, BloqEdge::Order);
        bloq.add_edge(region, later, BloqEdge::Order);
        bloq.add_edge(region, observable, BloqEdge::compose(0));
        bloq.add_edge(later, observable, BloqEdge::Order);
        bloq
    }

    #[test]
    fn constant_untaken_bindings_plan_no_early_reads() {
        let bloq = constant_untaken_include_program();
        assert_eq!(
            execution_schedule(&bloq, bloq.top()).unwrap().len(),
            5,
            "scheduling retains the inactive recipe"
        );
        let report = run_bloq(&bloq, 1, 0).expect("constant-untaken program executes");
        assert_eq!(report.observables[0].per_shot, [false]);
    }

    #[test]
    fn mutually_exclusive_binding_and_toucher_are_allowed() {
        use bloq_ir::circuit::{CoordCircuit, PauliBasis};
        use bloq_ir::lowering::{BloqTemplate, InstanceMeasurement};

        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
        circuit.do_gate(GateType::H, [IVec2::X]).unwrap();
        let measurement = circuit.measure(PauliBasis::Z, [IVec2::X])[0];
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let owner = bloq.add_node(support_node(template, 0));
        let selector = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
            Vec::new(),
        )));
        let opposite = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(owner, selector, BloqEdge::Order);
        bloq.add_edge(selector, opposite, BloqEdge::value(0));
        let mut include = BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![output_operator()],
        ));
        include.activation = Some(0);
        let include = bloq.add_node(include);
        let flip = one_qubit_template(&mut bloq, GateType::X);
        let toucher = bloq.add_node(guarded_node(support_node(flip, 1), 0));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(owner, include, BloqEdge::Order);
        bloq.add_edge(opposite, include, BloqEdge::value(0));
        bloq.add_edge(selector, toucher, BloqEdge::value(0));
        bloq.add_edge(include, toucher, BloqEdge::Order);
        bloq.add_edge(include, observable, BloqEdge::compose(0));
        bloq.add_edge(toucher, observable, BloqEdge::Order);
        assert!(plan_early_output_reads(&bloq).unwrap().is_empty());
        let report =
            run_bloq(&bloq, 32, 0).expect("mutually exclusive binding and toucher execute");
        assert_eq!(report.observables[0].per_shot, [false; 32]);
    }

    #[test]
    fn consumed_output_face_inside_same_region_is_read_at_its_cut() {
        let coord = IVec2::ZERO;
        let mut bloq = Bloq::new();
        let template = one_qubit_template(&mut bloq, GateType::H);
        let quantum = |instance| quantum_node(template, instance, IVec2::ZERO);

        let mut body = SubGraph::new();
        let owner = body.add_node(quantum(0));
        let include = body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: [(coord, CircuitPauli::Z)].into_iter().collect(),
            }],
        )));
        let later = body.add_node(quantum(1));
        body.add_edge(owner, include, BloqEdge::Order);
        body.add_edge(include, later, BloqEdge::Order);
        body.set_boundary_outputs(vec![include]);

        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        bloq.add_edge(region, observable, BloqEdge::compose(0));

        let plan = plan_early_output_reads(&bloq).expect("the program schedules");
        assert_eq!(plan.len(), 1, "the exported face is consumed once");
        assert_eq!(plan[0].before, (region_scope(region), later));
        assert_eq!(
            plan[0].bindings[0].operator(&bloq).instance,
            TemplateInstanceId(0)
        );
        run_bloq(&bloq, 1, 0).expect("the face is read before its consumer");
    }

    #[test]
    fn activated_exports_are_read_before_the_guarded_consumer() {
        for selected in [false, true] {
            let mut bloq = Bloq::new();
            let reset = one_qubit_template(&mut bloq, GateType::RZ);
            let flip = one_qubit_template(&mut bloq, GateType::X);
            let owner = bloq.add_node(support_node(reset, 0));
            let selector = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(selected),
            }));
            let mut include = BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                vec![output_operator()],
            ));
            include.activation = Some(0);
            let include = bloq.add_node(include);
            let consume = bloq.add_node(guarded_node(support_node(flip, 1), 0));
            let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(owner, include, BloqEdge::Order);
            bloq.add_edge(selector, include, BloqEdge::value(0));
            bloq.add_edge(selector, consume, BloqEdge::value(0));
            bloq.add_edge(include, consume, BloqEdge::Order);
            bloq.add_edge(include, observable, BloqEdge::compose(0));
            bloq.add_edge(consume, observable, BloqEdge::Order);
            let report = run_bloq(&bloq, 1, 0).expect("face is read before selected consumer");
            assert_eq!(report.observables[0].per_shot, [false]);
        }
    }

    #[test]
    fn inactive_activated_read_preserves_bell_correlation() {
        use bloq_ir::circuit::CoordCircuit;
        use bloq_ir::lowering::BloqTemplate;

        for future_gate in [false, true] {
            for gate_source in ["constant", "shared", "record", "future_record"] {
                if gate_source == "future_record" && !future_gate {
                    continue;
                }
                for nested in [false, true] {
                    let mut bloq = Bloq::new();
                    let mut circuit = CoordCircuit::new();
                    circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
                    circuit
                        .do_gate(GateType::CX, [IVec2::ZERO, ivec2(1, 0)])
                        .unwrap();
                    let measurement =
                        circuit.measure(bloq_ir::circuit::PauliBasis::Z, [ivec2(2, 0)])[0];
                    let template = bloq.add_template(BloqTemplate::new(circuit));
                    let owner = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
                    let mut predicate = bloq.add_node(BloqNode::classical(
                        if matches!(gate_source, "constant" | "shared") {
                            ClassicalNode::Compute {
                                expr: ClassicalExpr::Const(false),
                            }
                        } else {
                            ClassicalNode::observable_fragment(
                                vec![bloq_ir::lowering::InstanceMeasurement {
                                    instance: TemplateInstanceId(if gate_source == "record" {
                                        0
                                    } else {
                                        2
                                    }),
                                    measurement,
                                }],
                                Vec::new(),
                            )
                        },
                    ));
                    let predicate_source = predicate;
                    if gate_source == "shared" {
                        // A future shared DAG must cost O(nodes), not O(paths).
                        for _ in 0..32 {
                            let next = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                                expr: ClassicalExpr::Xor(Box::new([
                                    ClassicalExpr::In(0),
                                    ClassicalExpr::In(1),
                                ])),
                            }));
                            bloq.add_edge(predicate, next, BloqEdge::value(0));
                            bloq.add_edge(predicate, next, BloqEdge::value(1));
                            predicate = next;
                        }
                    }
                    let output = |operator| InstanceBoundaryOperator {
                        instance: TemplateInstanceId(0),
                        face: BoundaryFace::Output,
                        operator,
                    };
                    let mut fragment = BloqNode::classical(ClassicalNode::observable_fragment(
                        Vec::new(),
                        vec![output(
                            [(IVec2::ZERO, CircuitPauli::X)].into_iter().collect(),
                        )],
                    ));
                    fragment.activation = Some(0);
                    let mut region = bloq.add_node(fragment);
                    if nested {
                        let next = bloq.add_node(BloqNode::classical(
                            ClassicalNode::observable_fragment(Vec::new(), Vec::new()),
                        ));
                        bloq.add_edge(region, next, BloqEdge::compose(0));
                        bloq.add_edge(predicate, region, BloqEdge::value(0));
                        region = next;
                    }
                    if !nested {
                        bloq.add_edge(predicate, region, BloqEdge::value(0));
                    }
                    bloq.add_edge(owner, region, BloqEdge::Order);
                    let inactive = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
                    bloq.add_edge(region, inactive, BloqEdge::compose(0));
                    let include =
                        bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                            Vec::new(),
                            vec![output(
                                [
                                    (IVec2::ZERO, CircuitPauli::Z),
                                    (ivec2(1, 0), CircuitPauli::Z),
                                ]
                                .into_iter()
                                .collect(),
                            )],
                        )));
                    bloq.add_edge(owner, include, BloqEdge::Order);
                    let bell = bloq.add_node(BloqNode::classical(ClassicalNode::observable(1)));
                    bloq.add_edge(include, bell, BloqEdge::compose(0));
                    let reset = one_qubit_template(&mut bloq, GateType::RZ);
                    let reset = bloq.add_node(quantum_node(reset, 1, IVec2::ZERO));
                    bloq.add_edge(include, reset, BloqEdge::Order);
                    if future_gate {
                        bloq.add_edge(reset, predicate_source, BloqEdge::Order);
                    } else {
                        bloq.add_edge(region, reset, BloqEdge::Order);
                    }
                    bloq.add_edge(reset, inactive, BloqEdge::Order);
                    bloq.add_edge(reset, bell, BloqEdge::Order);
                    if gate_source == "future_record" {
                        let later = bloq.add_node(quantum_node(template, 2, IVec2::ZERO));
                        bloq.add_edge(reset, later, BloqEdge::Order);
                        bloq.add_edge(later, predicate, BloqEdge::Order);
                        assert!(matches!(
                            run_bloq(&bloq, 1, 0),
                            Err(ExecError::EarlyOutputGateUnavailable { .. })
                        ));
                        continue;
                    }
                    bloq.add_edge(owner, predicate_source, BloqEdge::Order);
                    let report = run_bloq(&bloq, 64, 0).unwrap();
                    assert!(
                        report
                            .observables
                            .iter()
                            .all(|observable| observable.per_shot.iter().all(|&bit| !bit)),
                        "inactive X must not disturb Bell ZZ: future_gate={future_gate}, nested={nested}"
                    );
                }
            }
        }
    }

    /// The scope a region body's nodes are scheduled in.
    fn region_scope(region: BloqNodeId) -> LevelPath {
        LevelPath::default().child(region, BodySelector::Body)
    }
}
