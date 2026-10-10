use std::collections::BTreeMap;
use std::ops::Bound::Excluded;

use bloq_graph::BlockGraph;
use bloq_ir::lowering::{NodeTemplateInstanceMergeError, PredicateAnalysis, TemplateInstance};
use bloq_ir::{
    Bloq, BloqNode, BloqNodeId, BloqValidationError, NodeProvenance, SubGraph, TemplateId, ValueRef,
};
use glam::{IVec2, IVec3};

use crate::CompileError;

/// Validate same-layer physical-qubit ownership in `bloq`.
///
/// Two nodes may overlap where source adjacency permits it or their predicates
/// are mutually exclusive. Validation runs at every graph level, so nodes inside
/// RUS retry bodies are covered. Each
/// level is checked in isolation: separate `SubGraph`s never cross-check, which
/// is exactly the intended semantics — a region reports its whole footprint as
/// one node at the parent level, and each body has its own execution scope.
/// Hierarchical source graphs are explicitly projected to their flat topology
/// for this audit; native compilation retains its definition ownership.
///
/// # Errors
///
/// Returns [`CompileError::OverlappingNodeQubit`] for the first disallowed overlap.
pub fn validate_bloq_qubit_layout_for_source(
    bloq: &Bloq,
    source: &BlockGraph,
) -> Result<(), CompileError> {
    let flat = source
        .has_module_structure()
        .then(|| source.flatten())
        .transpose()?;
    validate_bloq_qubit_layout_with_limits(
        bloq,
        flat.as_ref().unwrap_or(source),
        bloq_utils::boolean::BooleanLimits::DEFAULT,
    )
}

pub(crate) fn validate_bloq_qubit_layout_with_limits(
    bloq: &Bloq,
    source: &BlockGraph,
    limits: bloq_utils::boolean::BooleanLimits,
) -> Result<(), CompileError> {
    let mut template_runs = crate::FxMap::default();
    for (path, level) in bloq.levels() {
        // Inside a region body, same-layer nodes may legitimately share a
        // physical footprint when they are temporally *sequential* rather than
        // concurrent — a T block's cultivation and escape stages sit at the same
        // layer and qubits, joined by a temporal seam edge. So bodies skip pairs
        // with a directed path between them and flag only genuinely concurrent
        // (no path either way) same-layer overlaps. Top-level Region nodes use
        // the same ordered sharing with adjacent lowering nodes.
        let skip_sequential = !path.segments().is_empty();
        validate_level_qubit_layout(
            bloq,
            source,
            level,
            skip_sequential,
            limits,
            &mut template_runs,
        )?;
    }
    Ok(())
}

/// Runs the same-layer overlap check within a single graph level. When
/// `skip_sequential` is set (region bodies), pairs joined by a directed path are
/// exempt because they occupy the shared footprint at different times.
fn validate_level_qubit_layout(
    bloq: &Bloq,
    source: &BlockGraph,
    level: &SubGraph,
    skip_sequential: bool,
    limits: bloq_utils::boolean::BooleanLimits,
    template_runs: &mut crate::FxMap<TemplateId, Vec<QubitRun>>,
) -> Result<(), CompileError> {
    let mut layers = BTreeMap::<_, Vec<_>>::new();
    let mut guarded = false;
    for (id, node) in level.nodes() {
        if node.try_quantum().is_none() && node.try_region().is_none() {
            continue;
        }
        guarded |= node.activation.is_some()
            || node
                .try_quantum()
                .is_some_and(|quantum| !quantum.guards.is_empty());
        layers.entry(node.layer()).or_default().push((id, node));
    }
    if guarded && !level.is_acyclic() {
        return Err(BloqValidationError::CyclicGraph.into());
    }
    let mut predicates = PredicateAnalysis::with_limits(level, limits);
    let mut allowed_pairs = crate::FxMap::default();
    let mut scratch = level.path_scratch();
    for nodes in layers.values() {
        // One interval index per row and x parity. A stride-2 run stays one
        // interval regardless of code distance.
        let mut occupied = crate::FxMap::<(i32, i32), BTreeMap<i64, OccupiedRun>>::default();
        for &(right_id, right) in nodes {
            let footprints = guarded_footprints(bloq, level, right_id, right, template_runs)?;
            let mut conflict = None;
            for footprint in &footprints {
                for run in &footprint.runs {
                    let Some(row) = occupied.get(&(run.y, run.parity)) else {
                        continue;
                    };
                    let candidates = row
                        .range(..=run.start)
                        .next_back()
                        .into_iter()
                        .chain(row.range((Excluded(run.start), Excluded(run.end))));
                    for (&start, interval) in candidates {
                        let x_index = run.start.max(start);
                        if x_index >= interval.end {
                            continue;
                        }
                        let x = (2 * x_index + i64::from(run.parity)) as i32;
                        for &(left_id, left_guard) in &interval.users {
                            let allowed =
                                *allowed_pairs.entry((left_id, right_id)).or_insert_with(|| {
                                    let left = &level[left_id];
                                    let region_pair =
                                        left.try_region().is_some() || right.try_region().is_some();
                                    allows_same_layer_qubit_overlap(source, left, right)
                                        || ((skip_sequential || region_pair)
                                            && (level.has_path_with_scratch(
                                                left_id,
                                                right_id,
                                                &mut scratch,
                                            ) || level.has_path_with_scratch(
                                                right_id,
                                                left_id,
                                                &mut scratch,
                                            )))
                                });
                            if allowed
                                || !predicates
                                    .overlap_values(left_guard, footprint.guard)
                                    .map_err(|source| match source {
                                        NodeTemplateInstanceMergeError::BooleanResource(
                                            resource,
                                        ) => CompileError::from(resource),
                                        source => {
                                            BloqValidationError::InvalidInstanceMergeStructure {
                                                node: right_id,
                                                source,
                                            }
                                            .into()
                                        }
                                    })?
                            {
                                continue;
                            }
                            let key = (left_id.0, x, run.y);
                            if conflict.is_none_or(|previous| key < previous) {
                                conflict = Some(key);
                            }
                        }
                    }
                }
            }
            if let Some((left, x, y)) = conflict {
                return Err(CompileError::OverlappingNodeQubit {
                    first: BloqNodeId(left),
                    second: right_id,
                    qubit: IVec2::new(x, y),
                });
            }
            for footprint in footprints {
                for run in footprint.runs {
                    insert_occupied(
                        occupied.entry((run.y, run.parity)).or_default(),
                        run,
                        (right_id, footprint.guard),
                    );
                }
            }
        }
    }
    Ok(())
}

struct GuardedFootprint {
    guard: Option<ValueRef>,
    runs: Vec<QubitRun>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QubitRun {
    pub(super) y: i32,
    pub(super) parity: i32,
    pub(super) start: i64,
    pub(super) end: i64,
}

impl QubitRun {
    pub(super) fn translated(
        self,
        offset: IVec2,
    ) -> Result<Self, bloq_circuit::CoordinateOverflowError> {
        let first = IVec2::new((2 * self.start + i64::from(self.parity)) as i32, self.y);
        let last = IVec2::new((2 * (self.end - 1) + i64::from(self.parity)) as i32, self.y);
        let first = bloq_circuit::checked_translate_coordinate(first, offset)?;
        let last = bloq_circuit::checked_translate_coordinate(last, offset)?;
        Ok(Self {
            y: first.y,
            parity: first.x.rem_euclid(2),
            start: i64::from(first.x.div_euclid(2)),
            end: i64::from(last.x.div_euclid(2)) + 1,
        })
    }
}

struct OccupiedRun {
    end: i64,
    users: Vec<(BloqNodeId, Option<ValueRef>)>,
}

fn merge_runs(mut runs: Vec<QubitRun>) -> Vec<QubitRun> {
    runs.sort_unstable_by_key(|run| (run.y, run.parity, run.start, run.end));
    let mut merged: Vec<QubitRun> = Vec::with_capacity(runs.len());
    for run in runs {
        if let Some(last) = merged.last_mut()
            && last.y == run.y
            && last.parity == run.parity
            && run.start <= last.end
        {
            last.end = last.end.max(run.end);
        } else {
            merged.push(run);
        }
    }
    merged
}

pub(super) fn template_qubit_runs(qubits: &[IVec2]) -> Vec<QubitRun> {
    merge_runs(
        qubits
            .iter()
            .map(|qubit| QubitRun {
                y: qubit.y,
                parity: qubit.x.rem_euclid(2),
                start: i64::from(qubit.x.div_euclid(2)),
                end: i64::from(qubit.x.div_euclid(2)) + 1,
            })
            .collect(),
    )
}

fn append_instance_runs(
    bloq: &Bloq,
    node: &BloqNode,
    instance: &TemplateInstance,
    template_runs: &mut crate::FxMap<TemplateId, Vec<QubitRun>>,
    runs: &mut Vec<QubitRun>,
) -> Result<(), CompileError> {
    let Some(template) = bloq.templates().get(instance.template_id) else {
        return Ok(());
    };
    let local = template_runs
        .entry(instance.template_id)
        .or_insert_with(|| template_qubit_runs(template.qubits()));
    for run in local {
        let run = match run.translated(instance.offset) {
            Ok(run) => run,
            Err(_) => {
                // Template qubits are stored in arbitrary order. Re-scan that
                // order to keep the old first-overflow diagnostic exactly.
                return Err(bloq
                    .node_qubits(node)
                    .expect_err("a run endpoint overflowed")
                    .into());
            }
        };
        runs.push(run);
    }
    Ok(())
}

fn append_node_runs(
    bloq: &Bloq,
    node: &BloqNode,
    template_runs: &mut crate::FxMap<TemplateId, Vec<QubitRun>>,
    runs: &mut Vec<QubitRun>,
) -> Result<(), CompileError> {
    if let Some(region) = node.try_region() {
        for (_, body) in region.bodies() {
            for (_, body_node) in body.nodes() {
                append_node_runs(bloq, body_node, template_runs, runs)?;
            }
        }
    } else if let Some(quantum) = node.try_quantum() {
        let mut seen = crate::FxSet::default();
        for instance in &quantum.instances {
            if seen.insert((instance.template_id, instance.offset)) {
                append_instance_runs(bloq, node, instance, template_runs, runs)?;
            }
        }
    }
    Ok(())
}

fn split_occupied_at(row: &mut BTreeMap<i64, OccupiedRun>, x: i64) {
    if let Some((&start, interval)) = row.range(..x).next_back()
        && x < interval.end
    {
        let end = interval.end;
        let users = interval.users.clone();
        row.get_mut(&start).expect("occupied interval exists").end = x;
        row.insert(x, OccupiedRun { end, users });
    }
}

fn insert_occupied(
    row: &mut BTreeMap<i64, OccupiedRun>,
    run: QubitRun,
    user: (BloqNodeId, Option<ValueRef>),
) {
    split_occupied_at(row, run.start);
    split_occupied_at(row, run.end);
    let mut gaps = Vec::new();
    let mut cursor = run.start;
    for (&start, interval) in row.range_mut(run.start..run.end) {
        if cursor < start {
            gaps.push((cursor, start));
        }
        interval.users.push(user);
        cursor = interval.end;
    }
    if cursor < run.end {
        gaps.push((cursor, run.end));
    }
    for (start, end) in gaps {
        row.insert(
            start,
            OccupiedRun {
                end,
                users: vec![user],
            },
        );
    }
}

fn guarded_footprints(
    bloq: &Bloq,
    level: &SubGraph,
    id: BloqNodeId,
    node: &BloqNode,
    template_runs: &mut crate::FxMap<TemplateId, Vec<QubitRun>>,
) -> Result<Vec<GuardedFootprint>, CompileError> {
    let Some(quantum) = node
        .try_quantum()
        .filter(|quantum| !quantum.guards.is_empty())
    else {
        let guard = node
            .activation
            .map(|slot| {
                level
                    .value_inputs(id)
                    .find(|input| input.slot == slot)
                    .map(|input| input.value_ref().expect("value input"))
                    .ok_or(BloqValidationError::MissingClassicalInput { node: id, slot })
            })
            .transpose()?;
        let mut runs = Vec::new();
        append_node_runs(bloq, node, template_runs, &mut runs)?;
        return Ok(vec![GuardedFootprint {
            guard,
            runs: merge_runs(runs),
        }]);
    };
    // Keep the first producer for each slot, as the previous per-guard search did.
    let mut producers = crate::FxMap::default();
    for input in level.value_inputs(id) {
        producers
            .entry(input.slot)
            .or_insert(input.value_ref().expect("value input"));
    }
    let producer = |slot| {
        producers
            .get(&slot)
            .copied()
            .ok_or(BloqValidationError::MissingClassicalInput { node: id, slot })
    };
    let instance_ids = quantum
        .instances
        .iter()
        .map(|instance| instance.id)
        .collect::<crate::FxSet<_>>();
    let mut guards = crate::FxMap::default();
    for guard in &quantum.guards {
        let predicate = producer(guard.input)?;
        for &instance in &guard.instances {
            if !instance_ids.contains(&instance) || guards.insert(instance, predicate).is_some() {
                return Err(BloqValidationError::InvalidInstanceMergeStructure {
                    node: id,
                    source: NodeTemplateInstanceMergeError::InvalidMembership(format!(
                        "instance i{} is missing or registered more than once",
                        instance.0,
                    )),
                }
                .into());
            }
        }
    }
    let mut groups = crate::FxMap::<Option<ValueRef>, Vec<QubitRun>>::default();
    let mut seen = crate::FxSet::default();
    for instance in &quantum.instances {
        let guard = guards.get(&instance.id).copied();
        let runs = groups.entry(guard).or_default();
        if seen.insert((guard, instance.template_id, instance.offset)) {
            append_instance_runs(bloq, node, instance, template_runs, runs)?;
        }
    }
    Ok(groups
        .into_iter()
        .map(|(guard, runs)| GuardedFootprint {
            guard,
            runs: merge_runs(runs),
        })
        .collect())
}

fn allows_same_layer_qubit_overlap(source: &BlockGraph, left: &BloqNode, right: &BloqNode) -> bool {
    let NodeProvenance::BlockComponent {
        members: left_members,
    } = &left.provenance
    else {
        return false;
    };
    let NodeProvenance::BlockComponent {
        members: right_members,
    } = &right.provenance
    else {
        return false;
    };
    if left_members.is_empty() || right_members.is_empty() {
        return false;
    }

    left_members.iter().all(|left| {
        right_members
            .iter()
            .all(|right| walking_soft_corridor_overlap_allowed(source, left.pos, right.pos))
    })
}

fn walking_soft_corridor_overlap_allowed(source: &BlockGraph, left: IVec3, right: IVec3) -> bool {
    if left == right {
        return false;
    }
    let Some(left_block) = source.get_block(left) else {
        return false;
    };
    let Some(right_block) = source.get_block(right) else {
        return false;
    };
    let left_kind = left_block.kind();
    let right_kind = right_block.kind();
    let right_offsets = right_kind.reserved_offsets();
    let mut has_allowed_overlap = false;

    for left_offset in left_kind.reserved_offsets() {
        let overlap_pos = left + left_offset;
        if right_offsets
            .iter()
            .any(|&right_offset| right + right_offset == overlap_pos)
        {
            if !left_kind.allows_reserved_overlap(left, right_kind, right, overlap_pos) {
                return false;
            }
            has_allowed_overlap = true;
        }
    }

    has_allowed_overlap
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CoordCircuit, GateType};
    use bloq_graph::{Block, BlockKind, WalkingBoundaryKind, WalkingKind};
    use bloq_ir::{
        BloqEdge, ClassicalExpr, RegionNode, SourceBlockRef,
        lowering::{BloqTemplate, TemplateInstanceId},
    };
    use glam::{ivec2, ivec3};

    fn single_qubit_node(
        bloq: &mut Bloq,
        pos: glam::IVec3,
        gate: GateType,
        instance_id: TemplateInstanceId,
    ) -> BloqNode {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(gate, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef { pos }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(instance_id, template, ivec2(0, 0)));
        node
    }

    fn same_layer_overlap_bloq(left: glam::IVec3, right: glam::IVec3) -> Bloq {
        let mut bloq = Bloq::new();
        let first = single_qubit_node(&mut bloq, left, GateType::H, TemplateInstanceId(0));
        bloq.add_node(first);
        let second = single_qubit_node(&mut bloq, right, GateType::X, TemplateInstanceId(1));
        bloq.add_node(second);
        bloq
    }

    #[test]
    fn interval_layout_matches_coordinate_oracle() {
        let cases = [
            vec![
                vec![(
                    vec![ivec2(-5, -1), ivec2(-3, -1), ivec2(-1, -1), ivec2(1, -1)],
                    IVec2::ZERO,
                )],
                vec![(vec![ivec2(-5, -1), ivec2(0, -1)], IVec2::X)],
            ],
            vec![
                vec![(vec![ivec2(-5, -2), ivec2(-3, -2), ivec2(-1, -2)], IVec2::X)],
                vec![(vec![ivec2(-5, -2), ivec2(-3, -2)], IVec2::ZERO)],
            ],
            vec![
                vec![
                    (vec![ivec2(0, 0), ivec2(2, 0), ivec2(4, 0)], IVec2::ZERO),
                    (vec![ivec2(0, 0), ivec2(2, 0)], ivec2(2, 0)),
                ],
                vec![(vec![ivec2(6, 0)], IVec2::ZERO)],
            ],
            vec![
                vec![(vec![ivec2(i32::MAX, 1)], IVec2::ZERO)],
                vec![(vec![ivec2(i32::MIN, 1)], IVec2::ZERO)],
                vec![(vec![ivec2(i32::MIN, 1), ivec2(i32::MAX, 1)], IVec2::ZERO)],
            ],
        ];
        for placements in cases {
            let mut bloq = Bloq::new();
            let mut coordinates = Vec::new();
            for (node_index, instances) in placements.into_iter().enumerate() {
                let mut node = BloqNode::from_members(vec![SourceBlockRef {
                    pos: ivec3(node_index as i32, 0, 0),
                }]);
                for (instance_index, (qubits, offset)) in instances.into_iter().enumerate() {
                    let mut circuit = CoordCircuit::new();
                    for qubit in qubits {
                        circuit.do_gate(GateType::H, [qubit]).unwrap();
                    }
                    let template = bloq.add_template(BloqTemplate::new(circuit));
                    node.expect_quantum_mut()
                        .instances
                        .push(TemplateInstance::new(
                            TemplateInstanceId(instance_index as u32),
                            template,
                            offset,
                        ));
                }
                coordinates.push(bloq.node_qubits(&node).unwrap());
                bloq.add_node(node);
            }
            let expected = (0..coordinates.len()).find_map(|right| {
                let mut conflict = None;
                for left in 0..right {
                    for qubit in coordinates[left].intersection(&coordinates[right]) {
                        let key = (left as u32, qubit.x, qubit.y);
                        if conflict.is_none_or(|previous| key < previous) {
                            conflict = Some(key);
                        }
                    }
                }
                conflict
                    .map(|(left, x, y)| (BloqNodeId(left), BloqNodeId(right as u32), ivec2(x, y)))
            });
            let actual = validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new());
            match (expected, actual) {
                (None, Ok(())) => {}
                (
                    Some((first, second, qubit)),
                    Err(CompileError::OverlappingNodeQubit {
                        first: actual_first,
                        second: actual_second,
                        qubit: actual_qubit,
                    }),
                ) => assert_eq!(
                    (actual_first, actual_second, actual_qubit),
                    (first, second, qubit)
                ),
                (expected, actual) => panic!("coordinate oracle {expected:?}, layout {actual:?}"),
            }
        }
    }

    #[test]
    fn interval_layout_preserves_first_coordinate_overflow() {
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(i32::MAX, 0)]).unwrap();
        circuit.do_gate(GateType::H, [ivec2(i32::MIN, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef { pos: IVec3::ZERO }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                IVec2::X,
            ));
        let expected = bloq.node_qubits(&node).unwrap_err();
        bloq.add_node(node);
        assert!(matches!(
            validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()),
            Err(CompileError::CoordinateOverflow(actual)) if actual == expected
        ));
    }

    #[test]
    fn interval_insert_matches_coordinate_users_after_overlaps() {
        let mut row = BTreeMap::new();
        let mut oracle = BTreeMap::<i64, Vec<_>>::new();
        for (id, start, end) in [(0, 0, 8), (1, 2, 6), (2, -2, 3), (3, 5, 10), (4, 0, 8)] {
            let user = (BloqNodeId(id), None);
            insert_occupied(
                &mut row,
                QubitRun {
                    y: 0,
                    parity: 0,
                    start,
                    end,
                },
                user,
            );
            for x in start..end {
                oracle.entry(x).or_default().push(user);
            }
        }
        for x in -3..11 {
            let actual = row
                .range(..=x)
                .next_back()
                .filter(|(_, interval)| x < interval.end)
                .map(|(_, interval)| interval.users.as_slice());
            assert_eq!(actual, oracle.get(&x).map(Vec::as_slice), "x={x}");
        }
    }

    #[test]
    fn guarded_overlap_inserts_both_users_for_later_node() {
        use bloq_ir::{ClassicalNode, QuantumGuard};

        fn row_node(bloq: &mut Bloq, xs: &[i32], id: u32) -> BloqNode {
            let mut circuit = CoordCircuit::new();
            for &x in xs {
                circuit.do_gate(GateType::H, [ivec2(x, 0)]).unwrap();
            }
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let mut node = BloqNode::from_members(vec![SourceBlockRef {
                pos: ivec3(id as i32, 0, 0),
            }]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(id),
                    template,
                    IVec2::ZERO,
                ));
            node
        }

        let mut bloq = Bloq::new();
        let yes = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let no = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let mut first = row_node(&mut bloq, &[0, 2, 4, 6], 0);
        first.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(0)],
            ..Default::default()
        });
        let first = bloq.add_node(first);
        bloq.add_edge(yes, first, BloqEdge::value(0));
        let mut second = row_node(&mut bloq, &[2, 4, 6, 8], 1);
        second.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(1)],
            ..Default::default()
        });
        let second = bloq.add_node(second);
        bloq.add_edge(no, second, BloqEdge::value(0));
        let third_node = row_node(&mut bloq, &[6, 8], 2);
        let third = bloq.add_node(third_node);
        let error = validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()).unwrap_err();
        assert!(matches!(
            error,
            CompileError::OverlappingNodeQubit {
                first: actual_first,
                second: actual_second,
                qubit,
            } if actual_first == first && actual_second == third && qubit == ivec2(6, 0)
        ));
    }

    #[test]
    fn guarded_layout_preserves_member_and_region_availability() {
        use bloq_ir::{ClassicalNode, QuantumGuard};

        let mut bloq = Bloq::new();
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        let guard = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(raw, guard, BloqEdge::value(0));
        let inverse = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(guard, inverse, BloqEdge::value(0));
        let mut left =
            single_qubit_node(&mut bloq, IVec3::ZERO, GateType::H, TemplateInstanceId(0));
        let extra = single_qubit_node(&mut bloq, IVec3::ZERO, GateType::X, TemplateInstanceId(1));
        left.expect_quantum_mut()
            .instances
            .extend(extra.expect_quantum().instances.iter().copied());
        left.expect_quantum_mut().instances[1].offset = IVec2::X;
        left.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(0)],
            ..Default::default()
        });
        let left = bloq.add_node(left);
        bloq.add_edge(guard, left, BloqEdge::value(0));
        let mut right =
            single_qubit_node(&mut bloq, IVec3::ZERO, GateType::H, TemplateInstanceId(2));
        right.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(2)],
            ..Default::default()
        });
        let right = bloq.add_node(right);
        bloq.add_edge(inverse, right, BloqEdge::value(0));

        validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()).unwrap();
        let error = validate_bloq_qubit_layout_with_limits(
            &bloq,
            &BlockGraph::new(),
            bloq_utils::boolean::BooleanLimits {
                max_steps: 0,
                ..bloq_utils::boolean::BooleanLimits::UNLIMITED
            },
        )
        .unwrap_err();
        assert!(matches!(&error, CompileError::BooleanResource(resource) if resource.limit == 0));
        assert!(error.resource_limit_help().is_some());
        bloq.node_mut(left).unwrap().expect_quantum_mut().instances[1].offset = IVec2::ZERO;
        assert!(matches!(
            validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()),
            Err(CompileError::OverlappingNodeQubit { .. })
        ));

        bloq.node_mut(left).unwrap().expect_quantum_mut().instances[1].offset = IVec2::X;
        let mut child = bloq[right].clone();
        child.expect_quantum_mut().guards.clear();
        let mut body = SubGraph::new();
        body.add_node(child);
        let mut region = BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        });
        region.activation = Some(0);
        *bloq.node_mut(right).unwrap() = region;
        validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()).unwrap();

        bloq.node_mut(inverse).unwrap().kind = bloq_ir::BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::Or(Box::new([
                    ClassicalExpr::In(0),
                    ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
                ])),
            }
            .into(),
        );
        assert!(matches!(
            validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new()),
            Err(CompileError::OverlappingNodeQubit { .. })
        ));
    }

    #[test]
    fn guarded_layout_keeps_missing_input_before_membership_errors() {
        use bloq_ir::{ClassicalNode, QuantumGuard};

        let mut bloq = Bloq::new();
        let mut node =
            single_qubit_node(&mut bloq, IVec3::ZERO, GateType::H, TemplateInstanceId(0));
        node.expect_quantum_mut().guards.push(QuantumGuard {
            input: 7,
            instances: vec![TemplateInstanceId(99)],
            ..Default::default()
        });
        let id = bloq.add_node(node);
        assert!(matches!(
            guarded_footprints(&bloq, bloq.top(), id, &bloq[id], &mut crate::FxMap::default()),
            Err(CompileError::BloqValidation(
                BloqValidationError::MissingClassicalInput { node, slot: 7 }
            )) if node == id
        ));

        let predicate = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        bloq.add_edge(predicate, id, BloqEdge::value(7));
        assert!(matches!(
            guarded_footprints(&bloq, bloq.top(), id, &bloq[id], &mut crate::FxMap::default()),
            Err(CompileError::BloqValidation(
                BloqValidationError::InvalidInstanceMergeStructure {
                    node,
                    source: NodeTemplateInstanceMergeError::InvalidMembership(_)
                }
            )) if node == id
        ));

        bloq.node_mut(id).unwrap().expect_quantum_mut().guards[0].instances =
            vec![TemplateInstanceId(0), TemplateInstanceId(0)];
        assert!(matches!(
            guarded_footprints(&bloq, bloq.top(), id, &bloq[id], &mut crate::FxMap::default()),
            Err(CompileError::BloqValidation(
                BloqValidationError::InvalidInstanceMergeStructure {
                    node,
                    source: NodeTemplateInstanceMergeError::InvalidMembership(_)
                }
            )) if node == id
        ));

        bloq.node_mut(id).unwrap().expect_quantum_mut().guards[0].instances =
            vec![TemplateInstanceId(0)];
        let other = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        bloq.add_edge(other, id, BloqEdge::value(7));
        let first = bloq.value_inputs(id).find(|input| input.slot == 7).unwrap();
        let footprints = guarded_footprints(
            &bloq,
            bloq.top(),
            id,
            &bloq[id],
            &mut crate::FxMap::default(),
        )
        .unwrap();
        assert_eq!(footprints.len(), 1);
        assert_eq!(footprints[0].guard, first.value_ref());
    }

    #[test]
    fn source_aware_qubit_layout_allows_walking_soft_corridor_overlap() {
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap();
        let mut source = BlockGraph::new();
        source.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Walking(walking)));
        source.add_block(Block::new(
            ivec3(0, 1, 0),
            BlockKind::Walking(walking.with_boundary(WalkingBoundaryKind::XZX)),
        ));

        let bloq = same_layer_overlap_bloq(ivec3(0, 0, 0), ivec3(0, 1, 0));

        validate_bloq_qubit_layout_for_source(&bloq, &source)
            .expect("parallel walking soft-corridor source overlap is valid");
    }

    #[test]
    fn source_aware_qubit_layout_rejects_unrelated_walking_overlap() {
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::X).unwrap();
        let mut source = BlockGraph::new();
        source.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Walking(walking)));
        source.add_block(Block::new(ivec3(3, 0, 0), BlockKind::Walking(walking)));

        let bloq = same_layer_overlap_bloq(ivec3(0, 0, 0), ivec3(3, 0, 0));

        let error = validate_bloq_qubit_layout_for_source(&bloq, &source)
            .expect_err("distant walking blocks do not justify overlapping qubits");

        assert!(
            error.to_string().contains("overlapping physical qubit"),
            "{error}"
        );
    }

    #[test]
    fn qubit_layout_check_descends_into_region_bodies() {
        // Two genuinely parallel (no connecting edge) same-layer nodes sharing a
        // physical qubit inside a retry body must be rejected — proof the check
        // reaches region-body nodes, which the old top-level-only scan missed.
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let first = single_qubit_node(
            &mut bloq,
            ivec3(0, 0, 0),
            GateType::H,
            TemplateInstanceId(0),
        );
        body.add_node(first);
        let second = single_qubit_node(
            &mut bloq,
            ivec3(0, 0, 0),
            GateType::X,
            TemplateInstanceId(1),
        );
        body.add_node(second);
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));

        // Empty source: no walking soft-corridor exemption, so the identical
        // footprint at the same layer is a genuine overlap.
        let source = BlockGraph::new();
        let error = validate_bloq_qubit_layout_for_source(&bloq, &source)
            .expect_err("parallel overlapping nodes in a region body must be rejected");
        assert!(
            error.to_string().contains("overlapping physical qubit"),
            "{error}"
        );
    }

    #[test]
    fn qubit_layout_check_allows_sequential_region_body_overlap() {
        // Two same-layer nodes sharing a footprint but joined by a temporal edge
        // (the cultivation -> escape shape) are sequential, not concurrent, so
        // the shared qubit is legitimate and must NOT be flagged.
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let first = single_qubit_node(
            &mut bloq,
            ivec3(0, 0, 0),
            GateType::H,
            TemplateInstanceId(0),
        );
        let first_id = body.add_node(first);
        let second = single_qubit_node(
            &mut bloq,
            ivec3(0, 0, 0),
            GateType::X,
            TemplateInstanceId(1),
        );
        let second_id = body.add_node(second);
        body.add_edge(first_id, second_id, BloqEdge::Order);
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));

        let source = BlockGraph::new();
        validate_bloq_qubit_layout_for_source(&bloq, &source)
            .expect("sequential same-layer footprint sharing in a body is valid");
    }

    #[test]
    fn region_footprint_does_not_get_vacuous_block_overlap_exemption() {
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let body_node = single_qubit_node(
            &mut bloq,
            ivec3(0, 0, 0),
            GateType::H,
            TemplateInstanceId(0),
        );
        body.add_node(body_node);
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));
        let sibling = single_qubit_node(
            &mut bloq,
            ivec3(1, 0, 0),
            GateType::X,
            TemplateInstanceId(1),
        );
        bloq.add_node(sibling);

        let error = validate_bloq_qubit_layout_for_source(&bloq, &BlockGraph::new())
            .expect_err("a region and sibling quantum node cannot overlap concurrently");
        assert!(error.to_string().contains("overlapping physical qubit"));
    }
}
