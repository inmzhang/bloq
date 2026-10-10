use bloq_ir::{Bloq, BloqEdge, BloqNode};
use smallvec::SmallVec;
use std::collections::BTreeMap;

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
    mut allowed: impl FnMut(u32, u32, &[u32], Option<&crate::FxSet<u32>>) -> bool,
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
            for pair in users.windows(2) {
                if allowed(pair[0], pair[1], &users, user_slots.as_ref()) {
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

/// Induce occupancy `Order` edges: a *terminating* node (no
/// outgoing quantum seam) that shares physical qubits with a later *source*
/// node (no incoming quantum seam) must emit first — teardown then reinit.
/// Detector/observable composition has already run over `Temporal` seams
/// only, so these pure-ordering edges only constrain emission order — they
/// never change emitted detectors or observables.
///
/// Only era boundaries (terminal → source) are ordered. Mid-column walking
/// nodes also touch other columns' qubits (a glide's rotation transits a
/// neighbour cell's corner), but their member z-extent says nothing about
/// *when* within the walk the shared qubit is used, so any z-derived edge
/// between mid-column nodes can invert the real choreography
/// (ghz_slide_then_glide breaks under a naive per-qubit z ordering). At an
/// era boundary the direction is unambiguous.
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
        member_columns: crate::FxSet<glam::IVec2>,
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
        let Some(z_min) = positions.iter().map(|(_, layer)| *layer).min() else {
            continue;
        };
        let z_max = positions
            .iter()
            .map(|(_, layer)| *layer)
            .max()
            .expect("z_min proved the member list is non-empty");
        let mut placements = SmallVec::new();
        collect_footprint_placements(bloq, node, &mut placements, &mut None);
        footprints.push(Footprint {
            id,
            z_min,
            z_max,
            placements,
            member_columns: positions.iter().map(|(p, _)| p.truncate()).collect(),
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
    // Walking nodes are excluded entirely: their layout footprint covers the
    // whole transit, and member z says nothing about when *within* the walk a
    // shared qubit is used. In ghz_slide_then_glide the glide endpoint's
    // patch (member z=5) is live *before* the neighbour arm's slide transit
    // (member z=3) crosses it, so any z-derived edge between them inverts the
    // real choreography. A walk is detected by displacement between a node's
    // member columns and its seam endpoints: a slide/glide block keeps its
    // source member position but hands off (or was handed) the patch at a
    // different xy column.
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
    // Computed once per slot: the probe is pure per-node but was previously
    // re-evaluated for every (qubit, pair) window it appeared in.
    let is_walking: Vec<bool> = (0..footprints.len())
        .map(|slot| {
            let footprint = &footprints[slot];
            let displaced =
                |endpoint: glam::IVec3| !footprint.member_columns.contains(&endpoint.truncate());
            let seam_pipes = |direction: Dir| {
                seam_edges(bloq, footprint.id, direction).flat_map(|edge| match edge.edge {
                    BloqEdge::Quantum(edge) => edge.pipes.as_slice(),
                    BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => &[],
                })
            };
            seam_pipes(Dir::Outgoing).any(|seam| displaced(seam.pipe.src))
                || seam_pipes(Dir::Incoming).any(|seam| displaced(seam.pipe.dst))
        })
        .collect();
    // Membership is constant between run endpoints, so the same qubit filters
    // run once per interval. Sorted slots retain the old per-qubit user order;
    // duplicate slots only added self-windows, which always failed the z-gap.
    let candidate_slots = occupancy_pairs(events, |lower_slot, upper_slot, users, user_slots| {
        let (lower_slot, upper_slot) = (lower_slot as usize, upper_slot as usize);
        let (lower, upper) = (&footprints[lower_slot], &footprints[upper_slot]);
        lower.z_max < upper.z_min
            && !is_walking[lower_slot]
            && !is_walking[upper_slot]
            && !worldline_continues(lower_slot, users, user_slots, Dir::Outgoing)
            && !worldline_continues(upper_slot, users, user_slots, Dir::Incoming)
    });

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

    // One topological order certifies most candidates without a graph walk: an
    // edge that follows the order cannot close a cycle. Add all such edges
    // first, then probe only candidates that oppose the order against the
    // evolving graph. Initial-T control flow can genuinely point from a later-z
    // footprint to an earlier-z one, so source time alone is not a certificate.
    let mut candidate_pairs: Vec<_> = candidate_slots
        .into_iter()
        .map(|(lower, upper)| (footprints[lower as usize].id, footprints[upper as usize].id))
        .collect();
    candidate_pairs.sort_unstable_by_key(|(from, to)| (from.0, to.0));
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
    // If the union cycles, keep the existing per-edge decisions and order.
    let batch_acyclic = candidate_pairs
        .iter()
        .all(|(from, to)| rank[from.0 as usize] < rank[to.0 as usize])
        || bloq.is_acyclic_with_edges(&candidate_pairs);
    let mut scratch = bloq.path_scratch();
    for forward in [true, false] {
        for &(from, to) in &candidate_pairs {
            if (rank[from.0 as usize] < rank[to.0 as usize]) == forward
                && !has_edge(bloq, from, to)
                && (batch_acyclic || forward || !bloq.has_path_with_scratch(to, from, &mut scratch))
            {
                bloq.add_edge(from, to, BloqEdge::Order);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CoordCircuit, GateType};
    use bloq_ir::{
        BloqNodeId, SourceBlockRef,
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
            let actual = occupancy_pairs(events.clone(), allowed);
            assert_eq!(actual, expected, "walking={walking:?}");
            assert!(
                !actual.contains(&(14, 15)),
                "touching endpoints are disjoint"
            );
        }
        assert!(occupancy_pairs(Vec::new(), |_, _, _, _| true).is_empty());

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
            assert_eq!(occupancy_pairs(events.clone(), |_, _, _, _| true), expected);
            events.reverse();
            assert_eq!(occupancy_pairs(events, |_, _, _, _| true), expected);
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
