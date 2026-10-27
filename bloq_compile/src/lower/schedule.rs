use bloq_ir::{Bloq, BloqEdge, BloqNode};
use smallvec::SmallVec;
use std::collections::{BTreeMap, hash_map::Entry};
use std::rc::Rc;

use super::qubit_layout::{QubitRun, template_qubit_runs};

#[derive(Clone, Copy)]
struct OccupancyEvent {
    y: i32,
    parity: i32,
    x: i64,
    slot: u32,
    start: bool,
}

fn append_run_events(events: &mut Vec<OccupancyEvent>, slot: u32, run: QubitRun) {
    for (x, start) in [(run.start, true), (run.end, false)] {
        events.push(OccupancyEvent {
            y: run.y,
            parity: run.parity,
            x,
            slot,
            start,
        });
    }
}

fn occupancy_pairs(
    mut events: Vec<OccupancyEvent>,
    mut allowed: impl FnMut(u32, u32, &[u32], Option<&crate::FxSet<u32>>, QubitRun) -> bool,
) -> crate::FxSet<(u32, u32)> {
    events.sort_unstable_by_key(|event| (event.y, event.parity, event.x));
    let mut active = BTreeMap::<u32, usize>::new();
    let mut pairs = crate::FxSet::default();
    let mut users = SmallVec::<[u32; 8]>::new();
    let mut first = 0;
    while first < events.len() {
        let event = events[first];
        let mut end = first;
        // Starts and ends at one coordinate change membership together.
        // Counts retain a slot while another of its placements still covers it.
        while end < events.len()
            && (events[end].y, events[end].parity, events[end].x)
                == (event.y, event.parity, event.x)
        {
            let update = events[end];
            if update.start {
                *active.entry(update.slot).or_default() += 1;
            } else {
                let count = active
                    .get_mut(&update.slot)
                    .expect("an ending run is active");
                *count -= 1;
                if *count == 0 {
                    active.remove(&update.slot);
                }
            }
            end += 1;
        }
        if end < events.len()
            && (events[end].y, events[end].parity) == (event.y, event.parity)
            && active.len() > 1
        {
            users.clear();
            users.extend(active.keys().copied());
            let user_slots =
                (users.len() > 8).then(|| users.iter().copied().collect::<crate::FxSet<_>>());
            let span = QubitRun {
                y: event.y,
                parity: event.parity,
                start: event.x,
                end: events[end].x,
            };
            for pair in users.windows(2) {
                if allowed(pair[0], pair[1], &users, user_slots.as_ref(), span) {
                    pairs.insert((pair[0], pair[1]));
                }
            }
        }
        first = end;
    }
    debug_assert!(active.is_empty(), "every footprint run has an end");
    pairs
}

fn neighbor_in_users(
    neighbors: &[usize],
    users: &[u32],
    user_slots: Option<&crate::FxSet<u32>>,
) -> bool {
    // Neighbor slots are sorted. Search the short user list when that costs
    // less than scanning every neighbor (with an inline or hashed lookup).
    let search_steps = (usize::BITS - neighbors.len().leading_zeros()) as usize;
    if users.len().saturating_mul(search_steps) < neighbors.len() {
        users
            .iter()
            .any(|&user| neighbors.binary_search(&(user as usize)).is_ok())
    } else {
        neighbors.iter().any(|&neighbor| {
            let neighbor = neighbor as u32;
            user_slots.map_or_else(
                || users.contains(&neighbor),
                |slots| slots.contains(&neighbor),
            )
        })
    }
}

/// Alternative guards under which a node uses each physical qubit.
type QubitGuards = crate::FxMap<glam::IVec2, Vec<Option<bloq_ir::ValueRef>>>;

#[derive(Default)]
struct LiveQubitUses {
    inputs: QubitGuards,
    outputs: QubitGuards,
    uses: QubitGuards,
}

type GuardedPlacement = (bloq_ir::TemplateId, glam::IVec2, Option<bloq_ir::ValueRef>);

#[derive(Default)]
struct LiveSupportCache {
    nodes: crate::FxMap<bloq_ir::BloqNodeId, Rc<LiveQubitUses>>,
    placements: crate::FxMap<Vec<GuardedPlacement>, Rc<LiveQubitUses>>,
}

struct EraNeighbors {
    predecessors: crate::FxMap<bloq_ir::BloqNodeId, Vec<bloq_ir::BloqNodeId>>,
    successors: crate::FxMap<bloq_ir::BloqNodeId, Vec<bloq_ir::BloqNodeId>>,
}

fn guarded_node(node: &BloqNode) -> bool {
    node.activation.is_some()
        || node
            .try_quantum()
            .is_some_and(|quantum| !quantum.guards.is_empty())
}

fn conditional_footprint(node: &BloqNode) -> bool {
    guarded_node(node)
        || node.try_region().is_some_and(|region| {
            region
                .bodies()
                .any(|(_, body)| body.nodes().any(|(_, node)| conditional_footprint(node)))
        })
}

#[derive(Clone, Copy)]
enum LiveDirection {
    BeforeInputs,
    AfterOutputs,
}

impl LiveDirection {
    fn entry(self, uses: &LiveQubitUses) -> &QubitGuards {
        match self {
            Self::BeforeInputs => &uses.inputs,
            Self::AfterOutputs => &uses.outputs,
        }
    }

    fn exit(self, uses: &LiveQubitUses) -> &QubitGuards {
        match self {
            Self::BeforeInputs => &uses.outputs,
            Self::AfterOutputs => &uses.inputs,
        }
    }

    /// Orders `(near, far)` along the live state's flow.
    fn orient<T>(self, near: T, far: T) -> (T, T) {
        match self {
            Self::BeforeInputs => (far, near),
            Self::AfterOutputs => (near, far),
        }
    }
}

/// Memoized live boundary support of a top-level node.
fn cache_live_uses(
    bloq: &Bloq,
    supports: &mut LiveSupportCache,
    id: bloq_ir::BloqNodeId,
) -> Result<(), crate::CompileError> {
    if let Entry::Vacant(entry) = supports.nodes.entry(id) {
        let support =
            if let Some(quantum) = bloq[id].try_quantum() {
                let guards = instance_guards(bloq.top(), id);
                let key = quantum
                    .instances
                    .iter()
                    .map(|instance| {
                        (
                            instance.template_id,
                            instance.offset,
                            guards.get(&instance.id).copied(),
                        )
                    })
                    .collect::<Vec<_>>();
                // Time and instance IDs do not change a quantum node's physical support.
                match supports.placements.entry(key) {
                    Entry::Occupied(entry) => Rc::clone(entry.get()),
                    Entry::Vacant(entry) => Rc::clone(
                        entry.insert(Rc::new(live_qubit_uses_in_level(bloq, bloq.top(), id, id)?)),
                    ),
                }
            } else {
                Rc::new(live_qubit_uses_in_level(bloq, bloq.top(), id, id)?)
            };
        entry.insert(support);
    }
    Ok(())
}

fn instance_guards(
    level: &bloq_ir::SubGraph,
    id: bloq_ir::BloqNodeId,
) -> crate::FxMap<bloq_ir::lowering::TemplateInstanceId, bloq_ir::ValueRef> {
    let mut guards = crate::FxMap::default();
    for guard in &level[id].expect_quantum().guards {
        let value = level
            .value_inputs(id)
            .find(|input| input.slot == guard.input)
            .and_then(bloq_ir::ValueInput::value_ref)
            .expect("compiled membership input exists");
        guards.extend(guard.instances.iter().map(|&instance| (instance, value)));
    }
    guards
}

fn live_qubit_uses_in_level(
    bloq: &Bloq,
    level: &bloq_ir::SubGraph,
    id: bloq_ir::BloqNodeId,
    owner: bloq_ir::BloqNodeId,
) -> Result<LiveQubitUses, crate::CompileError> {
    let mut result = LiveQubitUses::default();
    if let Some(region) = level[id].try_region() {
        let activation = level[id].activation.map(|slot| {
            level
                .value_inputs(id)
                .find(|input| input.slot == slot)
                .and_then(bloq_ir::ValueInput::value_ref)
                .expect("compiled region activation exists")
        });
        for (_, body) in region.bodies() {
            if body.nodes().any(|(_, node)| guarded_node(node)) {
                return Err(crate::CompileError::ConditionalRegionLivenessUnsupported {
                    node: owner,
                });
            }
            let children = body
                .nodes()
                .map(|(id, _)| Ok((id, live_qubit_uses_in_level(bloq, body, id, owner)?)))
                .collect::<Result<crate::FxMap<_, _>, crate::CompileError>>()?;
            for (&child, support) in &children {
                for &qubit in support.uses.keys() {
                    result.uses.entry(qubit).or_default().push(activation);
                }
                for (incoming, face, external) in [
                    (true, &support.inputs, &mut result.inputs),
                    (false, &support.outputs, &mut result.outputs),
                ] {
                    for &qubit in face.keys() {
                        let internal = if incoming {
                            body.incoming(child).any(|edge| {
                                matches!(edge.edge, BloqEdge::Quantum(_))
                                    && children[&edge.source].outputs.contains_key(&qubit)
                            })
                        } else {
                            body.outgoing(child).any(|edge| {
                                matches!(edge.edge, BloqEdge::Quantum(_))
                                    && children[&edge.target].inputs.contains_key(&qubit)
                            })
                        };
                        if !internal {
                            external.entry(qubit).or_default().push(activation);
                        }
                    }
                }
            }
        }
        return Ok(result);
    }
    let Some(quantum) = level[id].try_quantum() else {
        return Ok(result);
    };
    let guards = instance_guards(level, id);
    for instance in &quantum.instances {
        let guard = guards.get(&instance.id).copied();
        let template = &bloq.templates()[instance.template_id];
        for &qubit in template.qubits() {
            let qubit = bloq_circuit::checked_translate_coordinate(qubit, instance.offset)?;
            result.uses.entry(qubit).or_default().push(guard);
        }
        for flow in &template.boundary_flows {
            for (map, support) in [
                (&flow.start, &mut result.inputs),
                (&flow.end, &mut result.outputs),
            ] {
                for (&qubit, _) in map.iter() {
                    let qubit = bloq_circuit::checked_translate_coordinate(qubit, instance.offset)?;
                    let alternatives = support.entry(qubit).or_default();
                    if !alternatives.contains(&guard) {
                        alternatives.push(guard);
                    }
                }
            }
        }
    }
    Ok(result)
}

fn guarded_era_boundary(
    bloq: &Bloq,
    endpoints: [(bloq_ir::BloqNodeId, bool); 2],
    qubit: glam::IVec2,
    require_live: bool,
    neighbors: &EraNeighbors,
    supports: &mut LiveSupportCache,
    predicates: &mut bloq_ir::lowering::PredicateAnalysis<'_>,
) -> Result<bool, crate::CompileError> {
    let [(left, left_moving), (right, right_moving)] = endpoints;
    let outgoing = &neighbors.successors[&left];
    let incoming = &neighbors.predecessors[&right];
    for id in [left, right]
        .into_iter()
        .chain(outgoing.iter().chain(incoming).copied())
    {
        cache_live_uses(bloq, supports, id)?;
    }
    let mut continuing = Vec::new();
    for (endpoint, adjacent, direction) in [
        (left, outgoing, LiveDirection::AfterOutputs),
        (right, incoming, LiveDirection::BeforeInputs),
    ] {
        let Some(face) = direction.entry(&supports.nodes[&endpoint]).get(&qubit) else {
            continue;
        };
        for next in adjacent {
            if let Some(other) = direction.exit(&supports.nodes[next]).get(&qubit) {
                for first in face {
                    for second in other {
                        continuing.push(first.iter().chain(second).copied().collect::<Vec<_>>());
                    }
                }
            }
        }
    }
    // A moving endpoint orders only a real teardown or initialization, never transit use.
    let left_support = &supports.nodes[&left];
    let right_support = &supports.nodes[&right];
    for (moving, opposite) in [
        (left_moving, &left_support.outputs),
        (right_moving, &right_support.inputs),
    ] {
        if moving {
            continuing.extend(
                opposite
                    .get(&qubit)
                    .into_iter()
                    .flatten()
                    .map(|guard| guard.iter().copied().collect()),
            );
        }
    }
    let left_face = if left_moving {
        &left_support.inputs
    } else {
        &left_support.uses
    };
    let right_face = if right_moving {
        &right_support.outputs
    } else {
        &right_support.uses
    };
    let Some(left_guards) = left_face.get(&qubit) else {
        return Ok(false);
    };
    let Some(right_guards) = right_face.get(&qubit) else {
        return Ok(false);
    };
    for left_guard in left_guards {
        for right_guard in right_guards {
            let mut required = left_guard
                .iter()
                .chain(right_guard)
                .copied()
                .collect::<Vec<_>>();
            // A required continuation rules out an era boundary without lowering its guards.
            if continuing
                .iter()
                .any(|clause| clause.iter().all(|guard| required.contains(guard)))
            {
                continue;
            }
            if !predicates
                .coexist_without(&required, &continuing)
                .map_err(
                    |source| bloq_ir::BloqValidationError::InvalidInstanceMergeStructure {
                        node: left,
                        source,
                    },
                )?
            {
                continue;
            }
            if !require_live {
                return Ok(true);
            }
            for guard in [
                &left_support.inputs,
                &left_support.outputs,
                &right_support.inputs,
                &right_support.outputs,
            ]
            .into_iter()
            .filter_map(|face| face.get(&qubit))
            .flatten()
            {
                let Some(guard) = guard else {
                    return Ok(true);
                };
                required.push(*guard);
                let live =
                    predicates
                        .coexist_without(&required, &continuing)
                        .map_err(|source| {
                            bloq_ir::BloqValidationError::InvalidInstanceMergeStructure {
                                node: left,
                                source,
                            }
                        })?;
                required.pop();
                if live {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

struct EraFrontier {
    node: bloq_ir::BloqNodeId,
    qubit: glam::IVec2,
    required: Vec<bloq_ir::ValueRef>,
    excluded: Vec<Vec<bloq_ir::ValueRef>>,
}

#[derive(Default)]
struct EraPlan {
    edges: crate::FxSet<(bloq_ir::BloqNodeId, bloq_ir::BloqNodeId)>,
    frontiers: Vec<EraFrontier>,
}

enum EraReversal {
    Feasible(Vec<(bloq_ir::BloqNodeId, bloq_ir::BloqNodeId)>),
    Cyclic,
}

fn frontier_domain(
    bloq: &Bloq,
    node: bloq_ir::BloqNodeId,
    required: &[bloq_ir::ValueRef],
    excluded: &[Vec<bloq_ir::ValueRef>],
) -> Result<bool, crate::CompileError> {
    if excluded
        .iter()
        .any(|clause| clause.iter().all(|guard| required.contains(guard)))
    {
        return Ok(false);
    }
    Ok(bloq_ir::lowering::PredicateAnalysis::new(bloq.top())
        .coexist_without(required, excluded)
        .map_err(
            |source| bloq_ir::BloqValidationError::InvalidInstanceMergeStructure { node, source },
        )?)
}

/// Follow one side of a live era, retaining the guards under which its outer boundary ends.
fn walk_live_era(
    bloq: &Bloq,
    endpoint: bloq_ir::BloqNodeId,
    reuser: bloq_ir::BloqNodeId,
    neighbors: &crate::FxMap<bloq_ir::BloqNodeId, Vec<bloq_ir::BloqNodeId>>,
    direction: LiveDirection,
    supports: &mut LiveSupportCache,
    scratch: &mut bloq_ir::PathScratch,
) -> Result<EraPlan, crate::CompileError> {
    use bloq_utils::boolean::BooleanDecisionDiagram;

    for id in [endpoint, reuser] {
        cache_live_uses(bloq, supports, id)?;
    }
    let mut qubits = direction
        .entry(&supports.nodes[&endpoint])
        .keys()
        .filter(|qubit| supports.nodes[&reuser].uses.contains_key(qubit))
        .copied()
        .collect::<Vec<_>>();
    qubits.sort_unstable_by_key(glam::IVec2::to_array);
    let mut work = BooleanDecisionDiagram::default();
    let mut plan = EraPlan::default();
    let mut paths = crate::FxMap::default();
    let mut has_path = |from, to| {
        *paths
            .entry((from, to))
            .or_insert_with(|| bloq.has_path_backwards_with_scratch(from, to, scratch))
    };
    let coexist = |guards: &[bloq_ir::ValueRef]| {
        bloq.top()
            .predicates_can_coexist_values(guards)
            .map_err(
                |source| bloq_ir::BloqValidationError::InvalidInstanceMergeStructure {
                    node: endpoint,
                    source,
                },
            )
    };
    for qubit in qubits {
        let mut pending = Vec::new();
        for input in &direction.entry(&supports.nodes[&endpoint])[&qubit] {
            for usage in &supports.nodes[&reuser].uses[&qubit] {
                pending.push((
                    endpoint,
                    input.iter().chain(usage).copied().collect::<Vec<_>>(),
                ));
            }
        }
        let mut visited = crate::FxSet::default();
        while let Some((current, mut guards)) = pending.pop() {
            work.charge(guards.len().saturating_add(1))?;
            guards.sort_unstable();
            guards.dedup();
            if !visited.insert((current, guards.clone())) || !coexist(&guards)? {
                continue;
            }
            let mut continuing = Vec::new();
            let entries = direction
                .entry(&supports.nodes[&current])
                .get(&qubit)
                .cloned()
                .unwrap_or_default();
            for &next in neighbors.get(&current).into_iter().flatten() {
                cache_live_uses(bloq, supports, next)?;
                let Some(outputs) = direction.exit(&supports.nodes[&next]).get(&qubit) else {
                    continue;
                };
                for input in &entries {
                    for output in outputs {
                        let clause = input.iter().chain(output).copied().collect::<Vec<_>>();
                        let mut path_guards = guards.clone();
                        path_guards.extend(&clause);
                        work.charge(path_guards.len().saturating_add(1))?;
                        // Reaching the opposite endpoint continues this same era, not a frontier.
                        continuing.push(clause);
                        if !coexist(&path_guards)? || next == reuser {
                            continue;
                        }
                        let (from, to) = direction.orient(next, reuser);
                        if has_path(to, from) {
                            let (producer, consumer) = direction.orient(endpoint, next);
                            return Err(crate::CompileError::LiveQubitReuse {
                                producer,
                                consumer,
                                reuser,
                                qubit,
                            });
                        }
                        if !has_path(from, to) {
                            plan.edges.insert((from, to));
                        }
                        pending.push((next, path_guards));
                    }
                }
            }
            if frontier_domain(bloq, endpoint, &guards, &continuing)? {
                plan.frontiers.push(EraFrontier {
                    node: current,
                    qubit,
                    required: guards,
                    excluded: continuing,
                });
            }
        }
    }
    Ok(plan)
}

/// Reverse complete eras atomically: the new terminal frontier must precede the old initial one.
fn plan_live_era_reordering(
    bloq: &Bloq,
    from: bloq_ir::BloqNodeId,
    to: bloq_ir::BloqNodeId,
    neighbors: &EraNeighbors,
    supports: &mut LiveSupportCache,
    scratch: &mut bloq_ir::PathScratch,
) -> Result<EraReversal, crate::CompileError> {
    let mut before = walk_live_era(
        bloq,
        from,
        to,
        &neighbors.predecessors,
        LiveDirection::BeforeInputs,
        supports,
        scratch,
    )?;
    let after = walk_live_era(
        bloq,
        to,
        from,
        &neighbors.successors,
        LiveDirection::AfterOutputs,
        supports,
        scratch,
    )?;
    before.edges.extend(after.edges);
    let mut terminals: crate::FxMap<_, Vec<_>> = crate::FxMap::default();
    for terminal in &after.frontiers {
        terminals.entry(terminal.qubit).or_default().push(terminal);
    }
    let mut work = bloq_utils::boolean::BooleanDecisionDiagram::default();
    for initial in &before.frontiers {
        for terminal in terminals.get(&initial.qubit).into_iter().flatten() {
            work.charge(1)?;
            if initial.node == terminal.node {
                continue;
            }
            let required = initial
                .required
                .iter()
                .chain(&terminal.required)
                .copied()
                .collect::<Vec<_>>();
            let excluded = initial
                .excluded
                .iter()
                .chain(&terminal.excluded)
                .cloned()
                .collect::<Vec<_>>();
            work.charge(required.len().saturating_add(excluded.len()))?;
            if !frontier_domain(bloq, from, &required, &excluded)? {
                continue;
            }
            if bloq.has_path_backwards_with_scratch(initial.node, terminal.node, scratch) {
                return Err(crate::CompileError::LiveQubitReuse {
                    producer: to,
                    consumer: terminal.node,
                    reuser: initial.node,
                    qubit: initial.qubit,
                });
            }
            if !bloq.has_path_backwards_with_scratch(terminal.node, initial.node, scratch) {
                before.edges.insert((terminal.node, initial.node));
            }
        }
    }
    let mut edges = before.edges.into_iter().collect::<Vec<_>>();
    edges.sort_unstable();
    Ok(if edges.is_empty() || bloq.is_acyclic_with_edges(&edges) {
        EraReversal::Feasible(edges)
    } else {
        EraReversal::Cyclic
    })
}

/// Induce occupancy `Order` edges between physical-qubit eras: a terminating
/// use must precede a later source on the same qubit — teardown then reinit.
/// Other qubits may continue through the same nodes' quantum seams.
/// Detector/observable composition has already run over `Temporal` seams
/// only, so these pure-ordering edges only constrain emission order — they
/// never change emitted detectors or observables.
///
/// Only era boundaries (terminal → source) are ordered. Moving patches use
/// intervals bounded by their members and owned temporal seam endpoints; their
/// input-only/output-only live support identifies teardown/initialization.
/// Transit and reset-only qubits do not establish moving era boundaries.
///
/// The Clifford proxy was the first emitter to need this: a T column's
/// source node hovers a layer gap above the previous era's terminating block
/// on the same footprint, and a bare toposort emitted the later era's port
/// MPP before the earlier era's readout of the same qubits.
pub(super) fn induce_occupancy_order_edges(bloq: &mut Bloq) -> Result<(), crate::CompileError> {
    use bloq_ir::{BloqEdgeRef, BloqNodeId};

    struct Footprint {
        id: BloqNodeId,
        z_min: i64,
        z_max: i64,
        placements: SmallVec<[(bloq_ir::TemplateId, glam::IVec2); 4]>,
        moving: bool,
    }

    /// Seam-probe direction, and the neighbour a seam edge reaches in it.
    #[derive(Clone, Copy, PartialEq)]
    enum Dir {
        Incoming,
        Outgoing,
    }
    fn seam_edges<'a>(
        bloq: &'a Bloq,
        of: BloqNodeId,
        direction: Dir,
    ) -> impl Iterator<Item = BloqEdgeRef<'a>> {
        let (incoming, outgoing) = match direction {
            Dir::Incoming => (Some(bloq.incoming(of)), None),
            Dir::Outgoing => (None, Some(bloq.outgoing(of))),
        };
        incoming
            .into_iter()
            .flatten()
            .chain(outgoing.into_iter().flatten())
    }

    // A region occupies its bodies' members. This keeps its block worldlines
    // ordered across eras even though the region node carries no members.
    // Virtual spatial Ports also occupy qubits, just before/after their source
    // layer. Omitting them lets a later input reset a still-live earlier patch.
    // Pipe nodes still carry no footprint.
    fn collect_member_positions(node: &BloqNode, positions: &mut Vec<(glam::IVec3, i64)>) {
        positions.extend(
            node.block_members()
                .iter()
                .map(|m| (m.pos, 2 * i64::from(m.pos.z))),
        );
        if let bloq_ir::NodeProvenance::SpatialPortSubstitution { source, .. } = node.provenance {
            positions.push((source, node.layer()));
        }
        if let Some(region) = node.try_region() {
            for (_, body) in region.bodies() {
                for (_, body_node) in body.nodes() {
                    collect_member_positions(body_node, positions);
                }
            }
        }
    }

    // Keep first-appearance order in `placements`, including across region bodies.
    fn collect_footprint_placements(
        bloq: &Bloq,
        node: &BloqNode,
        placements: &mut SmallVec<[(bloq_ir::TemplateId, glam::IVec2); 4]>,
        seen: &mut Option<crate::FxSet<(bloq_ir::TemplateId, glam::IVec2)>>,
    ) {
        if let Some(region) = node.try_region() {
            for (_, body) in region.bodies() {
                for (_, body_node) in body.nodes() {
                    collect_footprint_placements(bloq, body_node, placements, seen);
                }
            }
            return;
        }
        let Some(quantum) = node.try_quantum() else {
            return;
        };
        for instance in &quantum.instances {
            let placement = (instance.template_id, instance.offset);
            if bloq.templates().get(instance.template_id).is_none() {
                continue;
            }
            if let Some(seen) = seen {
                if seen.insert(placement) {
                    placements.push(placement);
                }
            } else if !placements.contains(&placement) {
                if placements.len() == 4 {
                    *seen = Some(placements.iter().copied().chain([placement]).collect());
                }
                placements.push(placement);
            }
        }
    }

    let mut footprints = Vec::new();
    let mut positions = Vec::new();
    for (id, node) in bloq.nodes() {
        positions.clear();
        collect_member_positions(node, &mut positions);
        let Some(mut z_min) = positions.iter().map(|(_, layer)| *layer).min() else {
            continue;
        };
        let mut z_max = positions
            .iter()
            .map(|(_, layer)| *layer)
            .max()
            .expect("z_min proved the member list is non-empty");
        let member_columns: crate::FxSet<_> = positions.iter().map(|(p, _)| p.truncate()).collect();
        let endpoints = |direction| {
            seam_edges(bloq, id, direction)
                .filter_map(|edge| match edge.edge {
                    BloqEdge::Quantum(edge) => Some(edge.pipes.as_slice()),
                    _ => None,
                })
                .flatten()
                .map(move |seam| match direction {
                    Dir::Incoming => seam.pipe.dst,
                    Dir::Outgoing => seam.pipe.src,
                })
        };
        let endpoints: SmallVec<[glam::IVec3; 4]> = endpoints(Dir::Incoming)
            .chain(endpoints(Dir::Outgoing))
            .collect();
        let moving = endpoints
            .iter()
            .any(|p| !member_columns.contains(&p.truncate()));
        if moving {
            // The displaced output face can outlive the block's origin layer.
            for endpoint in endpoints {
                z_min = z_min.min(2 * i64::from(endpoint.z));
                z_max = z_max.max(2 * i64::from(endpoint.z));
            }
        }
        let mut placements = SmallVec::new();
        collect_footprint_placements(bloq, node, &mut placements, &mut None);
        footprints.push(Footprint {
            id,
            z_min,
            z_max,
            placements,
            moving,
        });
    }
    // Active slots inherit this order, avoiding one sort per interval.
    footprints.sort_unstable_by_key(|footprint| (footprint.z_min, footprint.id.0));

    // Per qubit, order its users by z and constrain each *consecutive* pair:
    // teardown must precede the next era's reinit. A pair already ordered
    // through the IR (a continuous piped column, possibly routing through
    // other columns' seams) needs no edge; what this catches is footprint
    // reuse across an era gap with *no* connecting path — e.g. a T column
    // whose port/region source hovers two layers above the previous era's
    // terminating measurement (the U17g proxy hit this first), or a
    // selective measurement freeing a qubit that an adjacent era's merged
    // patch reinits (the CCZ factories: the merged patch has quantum seams
    // from *other* columns, so any-seam probes wrongly treat it as ordered).
    //
    // Both endpoints are judged per qubit: `lower` must have no outgoing
    // quantum seam carrying *this* qubit (its worldline ends there — a
    // node-level any-seam probe wrongly treats a merged patch that measures
    // one column out while others continue as mid-column), and `upper` must
    // have no incoming quantum seam carrying it (its worldline starts there).
    // The CCZ factories hit both: their merged patches carry seams for other
    // columns across every era boundary.
    //
    // Moving patches include transit qubits. Their era boundaries require
    // actual live input/output support as well as disjoint source intervals.
    let slot_by_id: crate::FxMap<BloqNodeId, usize> = footprints
        .iter()
        .enumerate()
        .map(|(slot, footprint)| (footprint.id, slot))
        .collect();
    let mut template_runs = crate::FxMap::default();
    let mut events = Vec::new();
    for (slot, footprint) in footprints.iter().enumerate() {
        #[cfg(debug_assertions)]
        let mut qubits = crate::FxSet::default();
        for &(template_id, offset) in &footprint.placements {
            let template = &bloq.templates()[template_id];
            let local = template_runs
                .entry(template_id)
                .or_insert_with(|| template_qubit_runs(template.qubits()));
            for &run in local.iter() {
                let run = match run.translated(offset) {
                    Ok(run) => run,
                    Err(_) => {
                        // Run order differs from template qubit order. Earlier
                        // placements passed their endpoint checks, so re-scan
                        // only this placement to retain the first old error.
                        let error = template
                            .qubits()
                            .iter()
                            .find_map(|&qubit| {
                                bloq_circuit::checked_translate_coordinate(qubit, offset).err()
                            })
                            .expect("a run endpoint overflowed");
                        return Err(error.into());
                    }
                };
                append_run_events(&mut events, slot as u32, run);
                #[cfg(debug_assertions)]
                for x in run.start..run.end {
                    qubits.insert(glam::IVec2::new(
                        (2 * x + i64::from(run.parity)) as i32,
                        run.y,
                    ));
                }
            }
        }
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            qubits,
            bloq.node_qubits(
                bloq.node(footprint.id)
                    .expect("footprint node still exists")
            )?,
            "run collection preserves the node's qubit set"
        );
    }
    if events.is_empty() {
        return Ok(());
    }
    drop(template_runs);
    // Per slot and direction, the footprint slots of its quantum neighbours
    // (looking through member-less pipe nodes, which carry no footprint).
    // Resolved once so the per-(qubit, pair) probe below never walks edges.
    let neighbor_slots = |direction: Dir| -> Vec<Vec<usize>> {
        let quantum_neighbors = |of: BloqNodeId| {
            seam_edges(bloq, of, direction)
                .filter(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
                .map(move |edge| match direction {
                    Dir::Incoming => edge.source,
                    Dir::Outgoing => edge.target,
                })
        };
        footprints
            .iter()
            .map(|footprint| {
                let mut slots = Vec::new();
                for next in quantum_neighbors(footprint.id) {
                    match slot_by_id.get(&next) {
                        Some(&slot) => slots.push(slot),
                        None => slots.extend(
                            quantum_neighbors(next).filter_map(|hop| slot_by_id.get(&hop).copied()),
                        ),
                    }
                }
                slots.sort_unstable();
                slots.dedup();
                slots
            })
            .collect()
    };
    let outgoing_neighbor_slots = neighbor_slots(Dir::Outgoing);
    let incoming_neighbor_slots = neighbor_slots(Dir::Incoming);
    // Whether the current qubit's worldline continues across a quantum seam of
    // the slot's node in `direction`: one of the quantum-neighbour footprint
    // slots also appears in the qubit's inverse user list.
    let worldline_continues =
        |slot: usize, users: &[u32], user_slots: Option<&crate::FxSet<u32>>, direction: Dir| {
            let neighbors = match direction {
                Dir::Incoming => &incoming_neighbor_slots[slot],
                Dir::Outgoing => &outgoing_neighbor_slots[slot],
            };
            neighbor_in_users(neighbors, users, user_slots)
        };
    // Membership is constant between run endpoints, so the same qubit filters
    // run once per interval. Sorted slots retain the old per-qubit user order;
    // duplicate slots only added self-windows, which always failed the z-gap.
    let conditional_members = footprints
        .iter()
        .map(|footprint| conditional_footprint(&bloq[footprint.id]))
        .collect::<Vec<_>>();
    let geometric_pair = |lower_slot: u32, upper_slot: u32| {
        let (lower_slot, upper_slot) = (lower_slot as usize, upper_slot as usize);
        let (lower, upper) = (&footprints[lower_slot], &footprints[upper_slot]);
        lower.z_max < upper.z_min
    };
    let neighbor_ids = |slots: &[Vec<usize>]| {
        footprints
            .iter()
            .enumerate()
            .map(|(slot, footprint)| {
                (
                    footprint.id,
                    slots[slot]
                        .iter()
                        .map(|&slot| footprints[slot].id)
                        .collect(),
                )
            })
            .collect()
    };
    let neighbors = EraNeighbors {
        predecessors: neighbor_ids(&incoming_neighbor_slots),
        successors: neighbor_ids(&outgoing_neighbor_slots),
    };
    let mut boundary_checks = Vec::new();
    let mut candidate_slots = occupancy_pairs(events, |lower, upper, users, user_slots, span| {
        let needs_boundary_check = users
            .iter()
            .any(|&slot| conditional_members[slot as usize] || footprints[slot as usize].moving);
        if needs_boundary_check && lower == users[0] {
            // Conditional users can disappear; moving users may only touch transit
            // qubits. Look through these runs so neither hides a later era boundary.
            for (index, &lower) in users.iter().enumerate() {
                for upper_index in index + 1..users.len() {
                    if upper_index > index + 1
                        && !conditional_members[users[upper_index - 1] as usize]
                        && !footprints[users[upper_index - 1] as usize].moving
                    {
                        break;
                    }
                    let upper = users[upper_index];
                    if geometric_pair(lower, upper) {
                        boundary_checks.push((lower, upper, span));
                    }
                }
            }
        }
        !needs_boundary_check
            && geometric_pair(lower, upper)
            && !worldline_continues(lower as usize, users, user_slots, Dir::Outgoing)
            && !worldline_continues(upper as usize, users, user_slots, Dir::Incoming)
    });
    let mut live_supports = LiveSupportCache::default();
    let mut predicates = bloq_ir::lowering::PredicateAnalysis::new(bloq.top());
    for (lower, upper, span) in boundary_checks {
        if candidate_slots.contains(&(lower, upper)) {
            continue;
        }
        for x in span.start..span.end {
            let qubit = glam::IVec2::new((2 * x + i64::from(span.parity)) as i32, span.y);
            if guarded_era_boundary(
                bloq,
                [
                    (
                        footprints[lower as usize].id,
                        footprints[lower as usize].moving,
                    ),
                    (
                        footprints[upper as usize].id,
                        footprints[upper as usize].moving,
                    ),
                ],
                qubit,
                false,
                &neighbors,
                &mut live_supports,
                &mut predicates,
            )? {
                candidate_slots.insert((lower, upper));
                break;
            }
        }
    }

    /// Whether an edge `from -> to` already exists. An existing edge appears
    /// in both `outgoing(from)` and `incoming(to)`, so scanning the two lists
    /// in lockstep and stopping as soon as either runs dry costs
    /// O(min(outdeg(from), indeg(to))) instead of a full out-list walk —
    /// `from` nodes can have enormous out-degree.
    fn has_edge(bloq: &Bloq, from: BloqNodeId, to: BloqNodeId) -> bool {
        let mut outgoing = bloq.outgoing(from);
        let mut incoming = bloq.incoming(to);
        loop {
            match (outgoing.next(), incoming.next()) {
                (Some(edge), _) if edge.target == to => return true,
                (_, Some(edge)) if edge.source == from => return true,
                (Some(_), Some(_)) => {}
                _ => return false,
            }
        }
    }

    // Initial-T control flow can reverse source time; certify the candidate union.
    let mut candidate_pairs: Vec<_> = candidate_slots
        .into_iter()
        .map(|(lower, upper)| (footprints[lower as usize].id, footprints[upper as usize].id))
        .collect();
    candidate_pairs.retain(|&(from, to)| !has_edge(bloq, from, to));
    let initial_order = bloq
        .deterministic_emit_order()
        .expect("lowered IR is acyclic before occupancy ordering");
    let mut rank =
        vec![0usize; initial_order.iter().map(|id| id.0).max().unwrap_or(0) as usize + 1];
    for (index, id) in initial_order.into_iter().enumerate() {
        rank[id.0 as usize] = index;
    }
    // An acyclic union certifies every candidate and every insertion prefix.
    // Check it once instead of walking the graph for each backward candidate.
    // A cyclic union needs individual ordering decisions below.
    let batch_acyclic = candidate_pairs
        .iter()
        .all(|(from, to)| rank[from.0 as usize] < rank[to.0 as usize])
        || bloq.is_acyclic_with_edges(&candidate_pairs);
    // Reset-only overlaps are preferences; settle data-bearing boundaries first.
    let mut pairs = Vec::with_capacity(candidate_pairs.len());
    let mut scratch = bloq.path_scratch();
    for (from, to) in candidate_pairs {
        let live = !batch_acyclic
            && (bloq[from].try_region().is_some() || bloq[to].try_region().is_some() || {
                cache_live_uses(bloq, &mut live_supports, from)?;
                cache_live_uses(bloq, &mut live_supports, to)?;
                let (a, b) = (&live_supports.nodes[&from], &live_supports.nodes[&to]);
                let mut qubits = a
                    .inputs
                    .keys()
                    .chain(a.outputs.keys())
                    .filter(|q| b.uses.contains_key(*q))
                    .chain(
                        b.inputs
                            .keys()
                            .chain(b.outputs.keys())
                            .filter(|q| a.uses.contains_key(*q)),
                    )
                    .copied()
                    .collect::<Vec<_>>();
                qubits.sort_unstable_by_key(glam::IVec2::to_array);
                qubits.dedup();
                // Shared live support alone is not a witness: that qubit may still
                // continue through a seam while only reset ancillas justify this pair.
                let endpoints = [from, to].map(|id| (id, footprints[slot_by_id[&id]].moving));
                let mut live = false;
                for qubit in qubits {
                    if guarded_era_boundary(
                        bloq,
                        endpoints,
                        qubit,
                        true,
                        &neighbors,
                        &mut live_supports,
                        &mut predicates,
                    )? {
                        live = true;
                        break;
                    }
                }
                live
            });
        let forward = rank[from.0 as usize] < rank[to.0 as usize];
        // Prove required orientations before committing unrelated source-time preferences.
        let reversal_blocked = if live && !forward {
            match plan_live_era_reordering(
                bloq,
                from,
                to,
                &neighbors,
                &mut live_supports,
                &mut scratch,
            ) {
                Ok(EraReversal::Feasible(_)) => false,
                Ok(EraReversal::Cyclic) | Err(crate::CompileError::LiveQubitReuse { .. }) => true,
                Err(error) => return Err(error),
            }
        } else {
            false
        };
        pairs.push((!live, !reversal_blocked, !forward, from, to));
    }
    pairs.sort_unstable_by_key(|&(optional, reversible, backward, from, to)| {
        (optional, reversible, backward, from.0, to.0)
    });
    // Forward edges cannot close a cycle until a backward edge is inserted.
    let mut ranked = true;
    for (_, _, backward, from, to) in pairs {
        if has_edge(bloq, from, to) {
            continue;
        }
        if batch_acyclic
            || (ranked && !backward)
            || !bloq.has_path_backwards_with_scratch(to, from, &mut scratch)
        {
            bloq.add_edge(from, to, BloqEdge::Order);
            ranked &= !backward;
        } else {
            ranked = false;
            let EraReversal::Feasible(edges) = plan_live_era_reordering(
                bloq,
                from,
                to,
                &neighbors,
                &mut live_supports,
                &mut scratch,
            )?
            else {
                return Err(bloq_ir::BloqValidationError::CyclicGraph.into());
            };
            for (before, after) in edges {
                bloq.add_edge(before, after, BloqEdge::Order);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CoordCircuit, Flow, GateType, Pauli, PauliMap};
    use bloq_ir::{
        BloqNodeId, ClassicalExpr, ClassicalNode, QuantumGuard, RegionNode, SourceBlockRef,
        SubGraph,
        lowering::{BloqTemplate, TemplateInstance, TemplateInstanceId},
    };
    use glam::{ivec2, ivec3};

    /// A one-qubit block-component node at `pos`, occupying layout qubit `(0,0)`
    /// shifted by `offset` (so two nodes overlap iff their offsets match).
    fn add_block_node(bloq: &mut Bloq, pos: glam::IVec3, offset: glam::IVec2) -> BloqNodeId {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef { pos }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(bloq.node_count() as u32),
                template,
                offset,
            ));
        bloq.add_node(node)
    }

    fn order_edges(bloq: &Bloq) -> Vec<(u32, u32)> {
        bloq.edges()
            .filter(|edge| matches!(edge.edge, BloqEdge::Order))
            .map(|edge| (edge.source.0, edge.target.0))
            .collect()
    }

    fn add_live_node(bloq: &mut Bloq, z: i32, input: bool, output: bool) -> BloqNodeId {
        let mut circuit = CoordCircuit::new();
        if !input {
            circuit.do_gate(GateType::RZ, [ivec2(0, 0)]).unwrap();
            circuit.tick();
        }
        let measurements = if output {
            circuit.do_gate(GateType::Z, [ivec2(0, 0)]).unwrap();
            Vec::new()
        } else {
            circuit.measure(bloq_circuit::PauliBasis::Z, [ivec2(0, 0)])
        };
        let face = |present| {
            if present {
                PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::Z)])
            } else {
                PauliMap::empty()
            }
        };
        let template = bloq.add_template(BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            Vec::new(),
            vec![Flow::new(face(input), face(output)).with_measurements(measurements)],
            Vec::new(),
        ));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, z),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(bloq.node_count() as u32),
                template,
                ivec2(0, 0),
            ));
        bloq.add_node(node)
    }

    fn wrap_in_rus(bloq: &mut Bloq, id: BloqNodeId) {
        let mut body = SubGraph::new();
        body.add_node(bloq[id].clone());
        *bloq.node_mut(id).unwrap() = BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
        });
    }

    fn guard_member(bloq: &mut Bloq, node: BloqNodeId, selector: BloqNodeId) {
        let quantum = bloq.node_mut(node).unwrap().expect_quantum_mut();
        quantum.guards.push(QuantumGuard {
            input: 0,
            instances: vec![quantum.instances[0].id],
            ..Default::default()
        });
        bloq.add_edge(selector, node, BloqEdge::value(0));
    }

    fn opposite_selectors(bloq: &mut Bloq) -> (BloqNodeId, BloqNodeId) {
        let bit = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let inverse = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(bit, inverse, BloqEdge::value(0));
        (bit, inverse)
    }

    #[test]
    fn reversing_source_time_does_not_split_a_continuous_live_era() {
        let mut bloq = Bloq::new();
        let source = add_live_node(&mut bloq, 4, false, true);
        let first = add_live_node(&mut bloq, 5, true, true);
        let second = add_live_node(&mut bloq, 6, true, true);
        let cap = add_live_node(&mut bloq, 0, true, false);
        for (from, to) in [(source, first), (first, second), (second, cap)] {
            bloq.add_edge(from, to, BloqEdge::quantum(vec![]));
        }
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert_eq!(
            bloq.deterministic_emit_order().unwrap(),
            vec![source, first, second, cap]
        );
    }

    #[test]
    fn reversing_two_live_eras_orders_the_new_cap_before_the_old_source() {
        let mut bloq = Bloq::new();
        let old_source = add_live_node(&mut bloq, 0, false, true);
        let old_cap = add_live_node(&mut bloq, 1, true, false);
        let new_source = add_live_node(&mut bloq, 2, false, true);
        let new_cap = add_live_node(&mut bloq, 3, true, false);
        bloq.add_edge(old_source, old_cap, BloqEdge::quantum(vec![]));
        bloq.add_edge(new_source, new_cap, BloqEdge::quantum(vec![]));
        bloq.add_edge(new_source, old_cap, BloqEdge::Order);
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert_eq!(
            bloq.deterministic_emit_order().unwrap(),
            vec![new_source, new_cap, old_source, old_cap]
        );
    }

    #[test]
    fn original_feedback_orders_live_eras_before_unconstrained_forward_boundaries() {
        let mut bloq = Bloq::new();
        // Node order puts the independent x -> y boundary before b -> p.
        let p = add_live_node(&mut bloq, 2, false, true);
        let a = add_live_node(&mut bloq, 0, false, true);
        let d = add_live_node(&mut bloq, 0, false, true);
        let x = add_live_node(&mut bloq, 1, true, false);
        let y = add_live_node(&mut bloq, 2, false, true);
        let z = add_live_node(&mut bloq, 3, true, false);
        let b = add_live_node(&mut bloq, 1, true, false);
        let c = add_live_node(&mut bloq, 3, true, false);
        for id in [d, x, y, z] {
            let node = bloq.node_mut(id).unwrap();
            node.expect_quantum_mut().instances[0].offset = ivec2(10, 0);
            if let bloq_ir::NodeProvenance::BlockComponent { members } = &mut node.provenance {
                members[0].pos.x = 5;
            }
        }
        for (from, to) in [(a, b), (p, c), (d, x), (y, z)] {
            bloq.add_edge(from, to, BloqEdge::quantum(vec![]));
        }
        for (from, to) in [(p, x), (z, b), (b, c)] {
            bloq.add_edge(from, to, BloqEdge::Order);
        }
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert!(bloq.has_path(b, p));
        assert!(bloq.has_path(z, d));
        bloq.deterministic_emit_order().unwrap();
    }

    #[test]
    fn continuing_live_wire_needs_no_boolean_work_to_exclude_an_era_boundary() {
        for guarded in [false, true] {
            let mut bloq = Bloq::new();
            let left = add_live_node(&mut bloq, 0, true, false);
            let source = add_live_node(&mut bloq, 1, false, true);
            let right = add_live_node(&mut bloq, 2, true, false);
            bloq.add_edge(source, right, BloqEdge::quantum(vec![]));
            if guarded {
                let (first, _) = opposite_selectors(&mut bloq);
                let (second, _) = opposite_selectors(&mut bloq);
                guard_member(&mut bloq, left, first);
                guard_member(&mut bloq, right, second);
            }
            let neighbors = EraNeighbors {
                predecessors: [(right, vec![source])].into_iter().collect(),
                successors: [(left, vec![])].into_iter().collect(),
            };
            for require_live in [false, true] {
                let mut predicates = bloq_ir::lowering::PredicateAnalysis::with_limits(
                    bloq.top(),
                    bloq_utils::boolean::BooleanLimits {
                        max_steps: 0,
                        ..Default::default()
                    },
                );
                assert!(
                    !guarded_era_boundary(
                        &bloq,
                        [(left, false), (right, false)],
                        ivec2(0, 0),
                        require_live,
                        &neighbors,
                        &mut LiveSupportCache::default(),
                        &mut predicates,
                    )
                    .unwrap()
                );
            }
        }
    }

    #[test]
    fn shared_live_supports_match_uncached_templates_offsets_and_guards() {
        let mut bloq = Bloq::new();
        let live = add_live_node(&mut bloq, 0, true, true);
        let reset = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));
        let a = bloq[live].expect_quantum().instances[0].template_id;
        let b = bloq[reset].expect_quantum().instances[0].template_id;
        let selector = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        let (_, other) = opposite_selectors(&mut bloq);
        let mut supports = LiveSupportCache::default();
        for templates in [vec![a], vec![b], vec![a, b], vec![b, a]] {
            for offset in [ivec2(0, 0), ivec2(-3, 5)] {
                for guard in [
                    None,
                    Some((selector, BloqEdge::value(0))),
                    Some((selector, BloqEdge::flip(0))),
                    Some((other, BloqEdge::value(0))),
                ] {
                    for z in [2, 3] {
                        let mut node = BloqNode::from_members(vec![SourceBlockRef {
                            pos: ivec3(0, 0, z),
                        }]);
                        node.expect_quantum_mut().instances.extend(
                            templates.iter().enumerate().map(|(index, &template)| {
                                TemplateInstance::new(
                                    TemplateInstanceId((2 * bloq.node_count() + index) as u32),
                                    template,
                                    offset,
                                )
                            }),
                        );
                        if guard.is_some() {
                            let quantum = node.expect_quantum_mut();
                            quantum.guards.push(QuantumGuard {
                                input: 0,
                                instances: vec![quantum.instances[0].id],
                                ..Default::default()
                            });
                        }
                        let id = bloq.add_node(node);
                        if let Some((source, edge)) = &guard {
                            bloq.add_edge(*source, id, edge.clone());
                        }
                        let expected = live_qubit_uses_in_level(&bloq, bloq.top(), id, id).unwrap();
                        cache_live_uses(&bloq, &mut supports, id).unwrap();
                        let actual = &supports.nodes[&id];
                        assert_eq!(
                            (&actual.inputs, &actual.outputs, &actual.uses),
                            (&expected.inputs, &expected.outputs, &expected.uses),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn plain_and_rus_footprints_obey_the_same_live_era_rule() {
        for wrapped in [None, Some(false), Some(true)] {
            let mut bloq = Bloq::new();
            let producer = add_live_node(&mut bloq, 0, false, true);
            let consumer = add_live_node(&mut bloq, 2, true, false);
            let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
            if let Some(region_is_producer) = wrapped {
                wrap_in_rus(
                    &mut bloq,
                    if region_is_producer { producer } else { reuser },
                );
            }
            bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
            bloq.add_edge(producer, reuser, BloqEdge::Order);
            bloq.add_edge(reuser, consumer, BloqEdge::Order);
            assert!(matches!(
                induce_occupancy_order_edges(&mut bloq),
                Err(crate::CompileError::LiveQubitReuse {
                    producer: actual_producer, consumer: actual_consumer,
                    reuser: actual_reuser, qubit,
                }) if (actual_producer, actual_consumer, actual_reuser, qubit)
                    == (producer, consumer, reuser, ivec2(0, 0))
            ));
        }
    }

    #[test]
    fn conditional_rus_body_liveness_is_rejected_without_an_opaque_summary() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        wrap_in_rus(&mut bloq, reuser);
        let bloq_ir::BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { body, .. }) =
            &mut bloq.node_mut(reuser).unwrap().kind
        else {
            unreachable!()
        };
        let child = body.quantum_nodes().next().unwrap().0;
        let selector = body.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let quantum = body.node_mut(child).unwrap().expect_quantum_mut();
        quantum.guards.push(QuantumGuard {
            input: 0,
            instances: vec![quantum.instances[0].id],
            ..Default::default()
        });
        body.add_edge(selector, child, BloqEdge::value(0));
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(producer, reuser, BloqEdge::Order);
        bloq.add_edge(reuser, consumer, BloqEdge::Order);
        assert!(matches!(
            induce_occupancy_order_edges(&mut bloq),
            Err(crate::CompileError::ConditionalRegionLivenessUnsupported { node }) if node == reuser
        ));
    }

    #[test]
    fn optional_ancilla_order_cannot_force_a_reset_inside_a_live_era() {
        let mut bloq = Bloq::new();
        let later_source = add_live_node(&mut bloq, 4, false, true);
        let ancilla = add_block_node(&mut bloq, ivec3(1, 0, 1), ivec2(1, 0));
        let earlier_source = add_live_node(&mut bloq, 2, false, true);
        let earlier_cap = add_live_node(&mut bloq, 3, true, false);
        let later_cap = add_live_node(&mut bloq, 5, true, false);
        let touch = add_block_node(&mut bloq, ivec3(1, 0, 2), ivec2(1, 0));
        let instance = bloq[touch].expect_quantum().instances[0];
        bloq.remove_node(touch).unwrap();
        bloq.node_mut(earlier_source)
            .unwrap()
            .expect_quantum_mut()
            .instances
            .push(instance);
        bloq.add_edge(earlier_source, earlier_cap, BloqEdge::quantum(vec![]));
        bloq.add_edge(later_source, later_cap, BloqEdge::quantum(vec![]));
        bloq.add_edge(later_source, ancilla, BloqEdge::Order);
        bloq.add_edge(earlier_cap, later_cap, BloqEdge::Order);
        // Both complete live eras fit: earlier_source, earlier_cap,
        // later_source, ancilla, later_cap. Only the resettable ancilla's
        // optional source-z ordering needs to change.
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert!(bloq.has_path(earlier_cap, later_source));
        assert!(bloq.has_path(earlier_source, ancilla));
        bloq.deterministic_emit_order().unwrap();
    }

    #[test]
    fn shared_live_support_does_not_promote_an_ancilla_only_order() {
        for guarded in [false, true] {
            let mut bloq = Bloq::new();
            let later_source = add_live_node(&mut bloq, 4, false, true);
            let earlier_source = add_live_node(&mut bloq, 2, false, true);
            let earlier_cap = add_live_node(&mut bloq, 3, true, false);
            let later_cap = add_live_node(&mut bloq, 5, true, false);
            let mut reset = CoordCircuit::new();
            reset.do_gate(GateType::RZ, [ivec2(1, 0)]).unwrap();
            let template = bloq.add_template(BloqTemplate::new(reset));
            for id in [earlier_cap, later_cap] {
                bloq.node_mut(id)
                    .unwrap()
                    .expect_quantum_mut()
                    .instances
                    .push(TemplateInstance::new(
                        TemplateInstanceId(100 + id.0),
                        template,
                        ivec2(0, 0),
                    ));
            }
            if guarded {
                let (bit, _) = opposite_selectors(&mut bloq);
                for id in [later_source, earlier_source, earlier_cap, later_cap] {
                    guard_member(&mut bloq, id, bit);
                }
            }
            bloq.add_edge(earlier_source, earlier_cap, BloqEdge::quantum(vec![]));
            bloq.add_edge(later_source, later_cap, BloqEdge::quantum(vec![]));
            bloq.add_edge(later_source, earlier_cap, BloqEdge::Order);
            // The cap-to-cap candidate is witnessed only by the reset ancilla.
            // Their shared data qubit belongs to two complete, reversible eras.
            induce_occupancy_order_edges(&mut bloq).unwrap();
            assert!(bloq.has_path(later_cap, earlier_source));
            assert!(!bloq.has_path(earlier_cap, later_cap));
            bloq.deterministic_emit_order().unwrap();
        }
    }

    #[test]
    fn a_nominally_earlier_reuser_cannot_interrupt_a_later_live_era() {
        let mut bloq = Bloq::new();
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        let producer = add_live_node(&mut bloq, 2, false, true);
        let consumer = add_live_node(&mut bloq, 4, true, false);
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(producer, reuser, BloqEdge::Order);
        bloq.add_edge(reuser, consumer, BloqEdge::Order);
        assert!(matches!(
            induce_occupancy_order_edges(&mut bloq),
            Err(crate::CompileError::LiveQubitReuse { .. })
        ));
    }

    #[test]
    fn a_disjoint_middle_member_cannot_hide_a_compatible_reuser() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let middle = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 6), ivec2(0, 0));
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let (bit, inverse) = opposite_selectors(&mut bloq);
        guard_member(&mut bloq, consumer, bit);
        guard_member(&mut bloq, middle, inverse);
        guard_member(&mut bloq, reuser, bit);
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(middle, consumer, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert!(bloq.has_path(consumer, reuser));
    }

    #[test]
    fn a_seam_in_the_opposite_arm_cannot_hide_a_reset() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let other_producer = add_live_node(&mut bloq, 3, false, true);
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let passthrough = add_live_node(&mut bloq, 4, true, true);
        let instance = bloq[passthrough].expect_quantum().instances[0];
        bloq.remove_node(passthrough).unwrap();
        bloq.node_mut(reuser)
            .unwrap()
            .expect_quantum_mut()
            .instances
            .push(instance);
        let (bit, inverse) = opposite_selectors(&mut bloq);
        guard_member(&mut bloq, producer, bit);
        guard_member(&mut bloq, consumer, bit);
        guard_member(&mut bloq, other_producer, inverse);
        guard_member(&mut bloq, reuser, bit);
        bloq.node_mut(reuser)
            .unwrap()
            .expect_quantum_mut()
            .guards
            .push(QuantumGuard {
                input: 1,
                instances: vec![instance.id],
                ..Default::default()
            });
        bloq.add_edge(inverse, reuser, BloqEdge::value(1));
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(other_producer, reuser, BloqEdge::quantum(vec![]));

        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert!(bloq.has_path(consumer, reuser));
    }

    #[test]
    fn a_complete_era_can_run_after_a_later_source_layer() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(reuser, producer, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert_eq!(
            bloq.deterministic_emit_order().unwrap(),
            vec![reuser, producer, consumer]
        );
    }

    #[test]
    fn an_incomparable_live_producer_is_scheduled_after_the_reuser() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(reuser, consumer, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert_eq!(
            bloq.deterministic_emit_order().unwrap(),
            vec![reuser, producer, consumer]
        );
    }

    #[test]
    fn disjoint_members_do_not_conflict_with_a_live_era() {
        let mut bloq = Bloq::new();
        let producer = add_live_node(&mut bloq, 0, false, true);
        let consumer = add_live_node(&mut bloq, 2, true, false);
        let reuser = add_block_node(&mut bloq, ivec3(0, 0, 4), ivec2(0, 0));
        let (bit, inverse) = opposite_selectors(&mut bloq);
        guard_member(&mut bloq, consumer, inverse);
        guard_member(&mut bloq, reuser, bit);
        bloq.add_edge(producer, consumer, BloqEdge::quantum(vec![]));
        bloq.add_edge(producer, reuser, BloqEdge::Order);
        bloq.add_edge(reuser, consumer, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();
    }

    #[test]
    fn neighbor_user_intersection_matches_linear_scan() {
        let user_lists = [
            vec![],
            vec![1],
            vec![1, 1, 3],
            vec![10, 11],
            (0..20).collect(),
        ];
        let neighbor_lists = [
            vec![],
            vec![3],
            vec![6, 13],
            (10..50).collect(),
            (0..50).collect(),
        ];
        for users in &user_lists {
            let slots =
                (users.len() > 8).then(|| users.iter().copied().collect::<crate::FxSet<_>>());
            for neighbors in &neighbor_lists {
                let expected = neighbors
                    .iter()
                    .any(|&neighbor| users.contains(&(neighbor as u32)));
                assert_eq!(
                    neighbor_in_users(neighbors, users, slots.as_ref()),
                    expected,
                    "users={users:?}, neighbors={neighbors:?}"
                );
            }
        }
    }

    #[test]
    fn occupancy_sweep_matches_enumerated_qubits_with_overlapping_placements() {
        let row = [-8, -6, -4, 0, 2, 4].map(|x| ivec2(x, -3)).to_vec();
        let mut placements = vec![
            (0, row.clone(), ivec2(-1, 2)),
            (0, row.clone(), ivec2(1, 2)),
            // These distinct placements touch in reduced x; they keep one slot live.
            (0, vec![ivec2(-2, 7), ivec2(0, 7)], ivec2(0, 0)),
            (0, vec![ivec2(2, 7), ivec2(4, 7)], ivec2(0, 0)),
        ];
        for slot in 1..12 {
            placements.push((slot, row.clone(), ivec2(-1, 2)));
        }
        placements.extend([
            (5, row, ivec2(1, 2)),
            (6, vec![ivec2(0, 7), ivec2(2, 7)], ivec2(0, 0)),
            // Touching slots do not share a physical qubit.
            (14, vec![ivec2(8, 5)], ivec2(0, 0)),
            (15, vec![ivec2(10, 5)], ivec2(0, 0)),
            // Both lattice extremes have short runs and an i64 exclusive end.
            (
                12,
                vec![ivec2(i32::MIN, 0), ivec2(i32::MAX, 0)],
                ivec2(0, 0),
            ),
            (
                13,
                vec![ivec2(i32::MIN, 0), ivec2(i32::MAX, 0)],
                ivec2(0, 0),
            ),
            // Equal rows but different x parity cannot intersect.
            (13, vec![ivec2(-8, -1), ivec2(-6, -1)], ivec2(0, 0)),
        ]);
        placements.sort_by_key(|(slot, _, _)| *slot);
        let mut events = Vec::new();
        let mut oracle = BTreeMap::<(i32, i32), Vec<u32>>::new();
        for (slot, qubits, offset) in placements {
            for &qubit in &qubits {
                let global = bloq_circuit::checked_translate_coordinate(qubit, offset).unwrap();
                oracle.entry((global.x, global.y)).or_default().push(slot);
            }
            for run in template_qubit_runs(&qubits) {
                append_run_events(&mut events, slot, run.translated(offset).unwrap());
            }
        }
        let mut outgoing = vec![vec![]; 16];
        let mut incoming = vec![vec![]; 16];
        outgoing[0] = vec![2];
        incoming[4] = vec![1];
        for walking in [None, Some(2)] {
            let allowed =
                |lower: u32, upper: u32, users: &[u32], slots: Option<&crate::FxSet<u32>>| {
                    lower != upper
                        && Some(lower) != walking
                        && Some(upper) != walking
                        && !neighbor_in_users(&outgoing[lower as usize], users, slots)
                        && !neighbor_in_users(&incoming[upper as usize], users, slots)
                };
            let mut expected = crate::FxSet::default();
            for users in oracle.values() {
                let slots = (users.len() > 8).then(|| users.iter().copied().collect());
                for pair in users.windows(2) {
                    if allowed(pair[0], pair[1], users, slots.as_ref()) {
                        expected.insert((pair[0], pair[1]));
                    }
                }
            }
            let actual = occupancy_pairs(events.clone(), |a, b, users, slots, _| {
                allowed(a, b, users, slots)
            });
            assert_eq!(actual, expected, "walking={walking:?}");
            assert!(
                !actual.contains(&(14, 15)),
                "touching endpoints are disjoint"
            );
        }
        assert!(occupancy_pairs(Vec::new(), |_, _, _, _, _| true).is_empty());

        // Isolated cases cannot have a missing or spurious pair hidden by a
        // second shared coordinate elsewhere in the larger fixture.
        let scenarios = [
            vec![
                (0, vec![ivec2(0, 0), ivec2(2, 0), ivec2(4, 0)], ivec2(0, 0)),
                (0, vec![ivec2(0, 0), ivec2(2, 0), ivec2(4, 0)], ivec2(2, 0)),
                (1, vec![ivec2(6, 0)], ivec2(0, 0)),
            ],
            vec![
                (0, vec![ivec2(0, 0)], ivec2(0, 0)),
                (1, vec![ivec2(2, 0)], ivec2(0, 0)),
            ],
            vec![
                (0, vec![ivec2(i32::MAX, 0)], ivec2(0, 0)),
                (1, vec![ivec2(i32::MAX, 0)], ivec2(0, 0)),
            ],
        ];
        for placements in scenarios {
            let mut events = Vec::new();
            let mut oracle = BTreeMap::<(i32, i32), Vec<u32>>::new();
            for (slot, qubits, offset) in placements {
                for &qubit in &qubits {
                    let global = bloq_circuit::checked_translate_coordinate(qubit, offset).unwrap();
                    oracle.entry((global.x, global.y)).or_default().push(slot);
                }
                for run in template_qubit_runs(&qubits) {
                    append_run_events(&mut events, slot, run.translated(offset).unwrap());
                }
            }
            let expected = oracle
                .values()
                .flat_map(|users| users.windows(2))
                .filter(|pair| pair[0] != pair[1])
                .map(|pair| (pair[0], pair[1]))
                .collect::<crate::FxSet<_>>();
            assert_eq!(
                occupancy_pairs(events.clone(), |_, _, _, _, _| true),
                expected
            );
            events.reverse();
            assert_eq!(occupancy_pairs(events, |_, _, _, _, _| true), expected);
        }
    }

    #[test]
    fn occupancy_run_overflow_keeps_footprint_and_template_qubit_order() {
        let mut bloq = Bloq::new();
        let mut add = |z, points: &[glam::IVec2], offset| {
            let mut circuit = CoordCircuit::new();
            for &point in points {
                circuit.do_gate(GateType::H, [point]).unwrap();
            }
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let mut node = BloqNode::from_members(vec![SourceBlockRef {
                pos: ivec3(0, 0, z),
            }]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(bloq.node_count() as u32),
                    template,
                    offset,
                ));
            bloq.add_node(node);
            template
        };
        add(10, &[ivec2(1, 0)], ivec2(i32::MAX, 0));
        let first = add(0, &[ivec2(i32::MAX, 0), ivec2(0, i32::MAX)], ivec2(1, 1));
        let expected = bloq.templates()[first]
            .qubits()
            .iter()
            .find_map(|&qubit| bloq_circuit::checked_translate_coordinate(qubit, ivec2(1, 1)).err())
            .unwrap();
        assert!(matches!(induce_occupancy_order_edges(&mut bloq),
            Err(crate::CompileError::CoordinateOverflow(actual)) if actual == expected));
    }

    fn add_moving_patch(bloq: &mut Bloq, z: i32, returning: bool) -> (BloqNodeId, BloqNodeId) {
        let [source, target] = if returning {
            [ivec2(2, 0), ivec2(0, 0)]
        } else {
            [ivec2(0, 0), ivec2(2, 0)]
        };
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [target]).unwrap();
        circuit.do_gate(GateType::CX, [source, target]).unwrap();
        circuit.do_gate(GateType::CX, [target, source]).unwrap();
        circuit.measure(bloq_circuit::PauliBasis::Z, [source]);
        // A resettable transit ancilla is touched but carries no boundary state.
        circuit.do_gate(GateType::RZ, [ivec2(4, 0)]).unwrap();
        let face = |qubit| PauliMap::from_unique_entries([(qubit, Pauli::Z)]);
        let template = bloq.add_template(BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            Vec::new(),
            vec![Flow::new(face(source), face(target))],
            Vec::new(),
        ));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: (source / 2).extend(z),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(bloq.node_count() as u32),
                template,
                ivec2(0, 0),
            ));
        let moving = bloq.add_node(node);
        let next = add_block_node(bloq, (target / 2).extend(z + 2), target);
        bloq.add_edge(
            moving,
            next,
            BloqEdge::quantum(vec![bloq_ir::TemporalPipeRef {
                src: (target / 2).extend(z + 1),
                dst: (target / 2).extend(z + 2),
                hadamard: false,
            }]),
        );
        (moving, next)
    }

    #[test]
    fn moving_patch_lifetimes_order_both_sides_of_a_reused_column() {
        for guarded in [false, true] {
            let mut bloq = Bloq::new();
            let resource = add_live_node(&mut bloq, 8, false, true);
            let cap = add_live_node(&mut bloq, 10, true, false);
            bloq.add_edge(resource, cap, BloqEdge::quantum(vec![]));
            let (outbound, continuation) = add_moving_patch(&mut bloq, 3, false);
            let (returning, _) = add_moving_patch(&mut bloq, 11, true);
            bloq.add_edge(continuation, returning, BloqEdge::quantum(vec![]));
            if guarded {
                let (bit, _) = opposite_selectors(&mut bloq);
                for node in [resource, cap, outbound, returning] {
                    guard_member(&mut bloq, node, bit);
                }
            }
            induce_occupancy_order_edges(&mut bloq).unwrap();
            assert!(bloq.has_path(outbound, resource));
            assert!(bloq.has_path(cap, returning));
            bloq.deterministic_emit_order().unwrap();
        }
    }

    #[test]
    fn moving_patch_intervals_exclude_transit_overlap_and_disjoint_guards() {
        for (offset, z, disjoint) in [
            (ivec2(0, 0), 4, false), // Still inside the displaced output interval.
            (ivec2(4, 0), 8, false), // Resettable transit qubit, not a live endpoint.
            (ivec2(0, 0), 8, true),
        ] {
            let mut bloq = Bloq::new();
            let reuse = add_block_node(&mut bloq, ivec3(0, 0, z), offset);
            let (moving, _) = add_moving_patch(&mut bloq, 3, false);
            if disjoint {
                let (bit, inverse) = opposite_selectors(&mut bloq);
                guard_member(&mut bloq, moving, bit);
                guard_member(&mut bloq, reuse, inverse);
            }
            induce_occupancy_order_edges(&mut bloq).unwrap();
            assert!(!bloq.has_path(moving, reuse));
            assert!(!bloq.has_path(reuse, moving));
        }
    }

    #[test]
    fn moving_transit_cannot_hide_a_later_qubit_era() {
        let mut bloq = Bloq::new();
        let end = add_live_node(&mut bloq, 1, true, false);
        let start = add_live_node(&mut bloq, 8, false, true);
        for id in [end, start] {
            bloq.node_mut(id).unwrap().expect_quantum_mut().instances[0].offset = ivec2(4, 0);
        }
        let (transit, _) = add_moving_patch(&mut bloq, 3, false);
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert!(bloq.has_path(end, start));
        assert!(!bloq.has_path(end, transit));
        assert!(!bloq.has_path(transit, start));
    }

    #[test]
    fn reused_spatial_ports_wait_for_the_previous_patch() {
        let mut bloq = Bloq::new();
        let lower = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));
        let upper = add_block_node(&mut bloq, ivec3(0, 0, 6), ivec2(0, 0));
        bloq.node_mut(upper).unwrap().provenance =
            bloq_ir::NodeProvenance::SpatialPortSubstitution {
                source: ivec3(0, 0, 6),
                role: bloq_graph::PortRole::Input,
            };
        induce_occupancy_order_edges(&mut bloq).unwrap();
        assert_eq!(order_edges(&bloq), vec![(lower.0, upper.0)]);
    }

    #[test]
    fn adjacent_z_non_overlapping_nodes_stay_unordered() {
        // Independent parallel components (disjoint footprints) get no edge.
        let mut bloq = Bloq::new();
        add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(5, 0));

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert!(order_edges(&bloq).is_empty());
    }

    #[test]
    fn already_piped_nodes_get_no_order_edge() {
        // A temporal seam already orders the pair, so no redundant Order edge.
        let mut bloq = Bloq::new();
        let lower = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        let upper = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));
        bloq.add_edge(lower, upper, BloqEdge::quantum(vec![]));

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert!(order_edges(&bloq).is_empty());
    }

    #[test]
    fn non_adjacent_z_overlap_gets_order_edge() {
        // Footprint reuse across an era gap still needs teardown-before-reinit:
        // a T column's source hovers a layer gap above the previous era's
        // terminating block (U17g's proxy emitter exposed this — the bare
        // toposort emitted the later era's port MPP first).
        let mut bloq = Bloq::new();
        let lower = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        let upper = add_block_node(&mut bloq, ivec3(0, 0, 2), ivec2(0, 0));

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert_eq!(order_edges(&bloq), vec![(lower.0, upper.0)]);
    }

    #[test]
    fn shared_qubit_users_are_ordered_by_z_not_node_id() {
        let mut bloq = Bloq::new();
        let upper = add_block_node(&mut bloq, ivec3(0, 0, 2), ivec2(0, 0));
        let lower = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        let middle = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert_eq!(
            order_edges(&bloq),
            vec![(lower.0, middle.0), (middle.0, upper.0)]
        );
    }

    #[test]
    fn merged_stage_placements_order_a_reused_qubit() {
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut merged = BloqNode::from_members(
            (0..16)
                .map(|x| SourceBlockRef {
                    pos: ivec3(x, 0, 0),
                })
                .collect(),
        );
        for x in 0..16 {
            merged
                .expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(x as u32 + 10_000),
                    template,
                    ivec2(x, 0),
                ));
        }
        let lower = bloq.add_node(merged);
        let upper = add_block_node(&mut bloq, ivec3(15, 0, 1), ivec2(15, 0));

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert_eq!(order_edges(&bloq), vec![(lower.0, upper.0)]);
    }

    #[test]
    fn long_shared_qubit_worldline_only_orders_the_era_gap() {
        let mut bloq = Bloq::new();
        let ids = (0..12)
            .map(|z| add_block_node(&mut bloq, ivec3(0, 0, z), ivec2(0, 0)))
            .collect::<Vec<_>>();
        for (index, pair) in ids.windows(2).enumerate() {
            if index != 5 {
                bloq.add_edge(pair[0], pair[1], BloqEdge::quantum(vec![]));
            }
        }

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert_eq!(order_edges(&bloq), vec![(ids[5].0, ids[6].0)]);
    }

    #[test]
    fn existing_reverse_path_blocks_source_time_order_edge() {
        let mut bloq = Bloq::new();
        let upper = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));
        let lower = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        bloq.add_edge(upper, lower, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();

        assert_eq!(order_edges(&bloq), vec![(upper.0, lower.0)]);
        bloq.deterministic_emit_order().unwrap();
    }

    #[test]
    fn guarded_occupancy_edges_check_paths_added_in_the_same_pass() {
        let mut bloq = Bloq::new();
        let a = add_block_node(&mut bloq, ivec3(0, 0, 0), ivec2(0, 0));
        let b = add_block_node(&mut bloq, ivec3(0, 0, 1), ivec2(0, 0));
        let c = add_block_node(&mut bloq, ivec3(1, 0, 0), ivec2(1, 0));
        let d = add_block_node(&mut bloq, ivec3(1, 0, 1), ivec2(1, 0));
        bloq.add_edge(b, c, BloqEdge::Order);
        bloq.add_edge(d, a, BloqEdge::Order);

        induce_occupancy_order_edges(&mut bloq).unwrap();

        bloq.deterministic_emit_order().unwrap();
        assert!(bloq.has_path(c, d));
        assert!(!bloq.has_path(a, b));
    }

    #[test]
    fn batch_occupancy_order_matches_sequential_cycle_checks() {
        // Every four-node DAG in node-id order, with every possible z order.
        // The shared footprint makes consecutive z nodes occupancy candidates.
        for z in itertools::Itertools::permutations(0..4, 4) {
            for edges in 0..64 {
                let mut bloq = Bloq::new();
                let ids: Vec<_> = z
                    .iter()
                    .map(|&z| add_block_node(&mut bloq, ivec3(0, 0, z), ivec2(0, 0)))
                    .collect();
                let mut bit = 0;
                for from in 0..4 {
                    for to in from + 1..4 {
                        if edges & (1 << bit) != 0 {
                            bloq.add_edge(ids[from], ids[to], BloqEdge::Order);
                        }
                        bit += 1;
                    }
                }
                let mut expected = bloq.clone();
                let order = expected.deterministic_emit_order().unwrap();
                let rank = |id| order.iter().position(|&node| node == id).unwrap();
                let mut by_z = ids.clone();
                by_z.sort_unstable_by_key(|id| z[id.0 as usize]);
                let mut candidates: Vec<_> = by_z.windows(2).map(|w| (w[0], w[1])).collect();
                candidates
                    .sort_unstable_by_key(|&(from, to)| (rank(from) >= rank(to), from.0, to.0));
                for (from, to) in candidates {
                    if !expected.outgoing(from).any(|edge| edge.target == to)
                        && !expected.has_path(to, from)
                    {
                        expected.add_edge(from, to, BloqEdge::Order);
                    }
                }

                induce_occupancy_order_edges(&mut bloq).unwrap();

                assert_eq!(
                    order_edges(&bloq),
                    order_edges(&expected),
                    "z={z:?}, edges={edges}"
                );
                bloq.deterministic_emit_order().unwrap();
            }
        }
    }
}
