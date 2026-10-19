//! Materialize measurement leaves only after their complete use lists are known.

use std::sync::Arc;

use bloq_graph::GuardedTopology;
use bloq_ir::lowering::InstanceMeasurement;
use bloq_ir::{Bloq, BloqEdge, BloqNode, BloqNodeId, ClassicalNode, NodeProvenance, ValueRole};
use bloq_utils::boolean::{DECISION_TRUE as ONE, DecisionId};

use super::super::assembly::{Assembly, BooleanProgram};
use crate::{CompileError, FxMap, FxSet, check_resource};

#[derive(Clone, Copy, Debug)]
pub(super) enum Input {
    Node(BloqNodeId),
    Measurement(u32),
}

impl From<BloqNodeId> for Input {
    fn from(node: BloqNodeId) -> Self {
        Self::Node(node)
    }
}

struct Measurement {
    records: Arc<[InstanceMeasurement]>,
    predicate: Option<BloqNodeId>,
    owner: BloqNodeId,
}

struct Readout {
    node: BloqNodeId,
    inputs: Vec<(Input, ValueRole)>,
    inline_records: usize,
}

#[derive(Default)]
pub(super) struct ReadoutEmission {
    measurements: Vec<Measurement>,
    readouts: Vec<Readout>,
}

impl ReadoutEmission {
    pub(super) fn measurement(
        &mut self,
        assembly: &mut Assembly,
        boolean: &mut BooleanProgram,
        topology: &GuardedTopology,
        guard: DecisionId,
        records: &Arc<[InstanceMeasurement]>,
        owner: BloqNodeId,
    ) -> Result<Input, CompileError> {
        // Emit at the cache miss so later sibling sharing preserves guard order.
        let predicate =
            (guard != ONE).then(|| boolean.emit(guard, &topology.diagram, &mut assembly.bloq));
        check_resource(
            "deferred measurement fragments",
            self.measurements.len().saturating_add(1),
            u32::MAX as usize,
        )?;
        let id = self.measurements.len() as u32;
        self.measurements.push(Measurement {
            records: Arc::clone(records),
            predicate,
            owner,
        });
        Ok(Input::Measurement(id))
    }

    pub(super) fn readout(
        &mut self,
        assembly: &mut Assembly,
        inputs: Vec<(Input, ValueRole)>,
    ) -> Result<BloqNodeId, CompileError> {
        check_resource("readout input slots", inputs.len(), u32::MAX as usize)?;
        let index = assembly.allocate_observable()?;
        let node = assembly.bloq.add_node(
            BloqNode::classical(ClassicalNode::observable(index))
                .with_provenance(NodeProvenance::Generator { ordinal: index }),
        );
        self.readouts.push(Readout {
            node,
            inputs,
            inline_records: 0,
        });
        Ok(node)
    }

    pub(super) fn finish(mut self, bloq: &mut Bloq) -> Result<(), CompileError> {
        // One compact inverse index, without a separately allocated use vector
        // for each leaf. Readouts arrive in increasing node-id order.
        let mut offsets = vec![0usize; self.measurements.len() + 1];
        for readout in &mut self.readouts {
            for (input, role) in &readout.inputs {
                if let Input::Measurement(id) = input {
                    debug_assert_eq!(*role, ValueRole::Data);
                    offsets[*id as usize + 1] += 1;
                    let measurement = &self.measurements[*id as usize];
                    if measurement.predicate.is_none() {
                        readout.inline_records += measurement.records.len();
                    }
                }
            }
        }
        for index in 1..offsets.len() {
            offsets[index] = offsets[index].saturating_add(offsets[index - 1]);
        }
        let occurrences = offsets.last().copied().unwrap_or(0);
        check_resource(
            "readout composition occurrences",
            occurrences,
            u32::MAX as usize,
        )?;
        let mut consumers = vec![BloqNodeId(0); occurrences];
        let mut next = offsets[..self.measurements.len()].to_vec();
        for readout in &self.readouts {
            for (input, _) in &readout.inputs {
                if let Input::Measurement(id) = input {
                    consumers[next[*id as usize]] = readout.node;
                    next[*id as usize] += 1;
                }
            }
        }
        drop(next);

        struct Group {
            first: usize,
            records: usize,
        }
        let mut siblings = FxMap::default();
        let mut groups = Vec::<Group>::new();
        let mut group_of = vec![u32::MAX; self.measurements.len()];
        for (id, measurement) in self.measurements.iter().enumerate() {
            let uses = &consumers[offsets[id]..offsets[id + 1]];
            // Lowering drops inactive readouts before emission. An unconsumed
            // leaf would only leave its records unused, so release builds continue.
            debug_assert!(
                !uses.is_empty(),
                "a pending measurement has a readout consumer"
            );
            debug_assert!(uses.is_sorted());
            let Some(predicate) = measurement.predicate else {
                continue;
            };
            // The slice retains repeated consumers. Slots remain in readouts;
            // only the first sibling's occurrences survive the existing fuser.
            let group = *siblings.entry((predicate, uses)).or_insert_with(|| {
                let index = groups.len();
                groups.push(Group {
                    first: id,
                    records: 0,
                });
                index
            });
            groups[group].records += measurement.records.len();
            group_of[id] = group as u32;
        }
        drop(siblings);

        let mut nodes = Vec::with_capacity(groups.len());
        for group in &groups {
            let predicate = self.measurements[group.first]
                .predicate
                .expect("conditional siblings have an activation producer");
            let mut fragment = BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::with_capacity(group.records),
                Vec::new(),
            ));
            fragment.activation = Some(0);
            let node = bloq.add_node(fragment);
            bloq.add_edge(predicate, node, BloqEdge::value(0));
            nodes.push(node);
        }
        for readout in &self.readouts {
            measurements_mut(bloq, readout.node).reserve_exact(readout.inline_records);
        }
        let mut orders = FxSet::default();
        for (id, measurement) in self.measurements.into_iter().enumerate() {
            let targets = if group_of[id] == u32::MAX {
                &consumers[offsets[id]..offsets[id + 1]]
            } else {
                std::slice::from_ref(&nodes[group_of[id] as usize])
            };
            for &target in targets {
                measurements_mut(bloq, target).extend_from_slice(&measurement.records);
                if orders.insert((measurement.owner, target)) {
                    bloq.add_edge(measurement.owner, target, BloqEdge::Order);
                }
            }
        }
        drop((orders, consumers, offsets));

        // Replay every retained input in slot order, including boundary packets
        // and semantic Value roles. Never move or combine boundary products.
        for readout in self.readouts {
            for (slot, (input, role)) in readout.inputs.into_iter().enumerate() {
                let producer = match input {
                    Input::Node(node) => node,
                    Input::Measurement(id) => {
                        let group = group_of[id as usize];
                        if group == u32::MAX || groups[group as usize].first != id as usize {
                            continue;
                        }
                        nodes[group as usize]
                    }
                };
                let edge = if role == ValueRole::Data
                    && matches!(
                        bloq[producer].try_classical(),
                        Some(ClassicalNode::Observable { index: None, .. })
                    ) {
                    BloqEdge::compose(slot as u32)
                } else {
                    BloqEdge::Value {
                        slot: slot as u32,
                        role,
                        output: Default::default(),
                    }
                };
                bloq.add_edge(producer, readout.node, edge);
            }
        }
        Ok(())
    }
}

fn measurements_mut(bloq: &mut Bloq, node: BloqNodeId) -> &mut Vec<InstanceMeasurement> {
    let Some(ClassicalNode::Observable { measurements, .. }) = bloq
        .node_mut(node)
        .expect("emitted readout exists")
        .try_classical_mut()
    else {
        unreachable!("readout emission owns only observable nodes")
    };
    measurements
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_ir::lowering::{InstanceBoundaryOperator, TemplateInstanceId};
    use bloq_ir::{BoundaryFace, ClassicalAssignment};
    use glam::IVec2;
    use std::collections::BTreeSet;

    #[test]
    fn deferred_siblings_preserve_parity_activation_boundary_order_and_owners() {
        let mut assembly = Assembly::new();
        assert_eq!(assembly.allocate_observable().unwrap(), 0);
        assembly.bloq = Bloq::from_text("BLOQIR 1\ntemplate t0 {\ncircuit {\nR (0,0) (1,0)\nM (0,0):m0 (1,0):m1\n}\n}\ngraph {\nn0 quantum {\ninstance i0 t0 @ (0,0)\n}\nn1 quantum {\ninstance i1 t0 @ (10,0)\n}\nn2 observable 0 measurements i0:m1\nn3 compute in0\nn0 -> n2 order\nn2 -> n3 value 0\n}\n").unwrap();
        let record = |instance, measurement| InstanceMeasurement {
            instance: TemplateInstanceId(instance),
            measurement,
        };
        let mut emission = ReadoutEmission {
            measurements: [
                (vec![record(0, 0); 2], BloqNodeId(0), Some(BloqNodeId(3))),
                (vec![record(1, 0)], BloqNodeId(1), Some(BloqNodeId(3))),
                (vec![record(1, 1)], BloqNodeId(1), Some(BloqNodeId(3))),
                (vec![record(0, 1)], BloqNodeId(0), None),
            ]
            .map(|(records, owner, predicate)| Measurement {
                records: records.into(),
                owner,
                predicate,
            })
            .into(),
            ..ReadoutEmission::default()
        };
        let [a, b, other_uses, unconditional] = [0, 1, 2, 3].map(Input::Measurement);
        let operators = [
            bloq_circuit::Pauli::X,
            bloq_circuit::Pauli::Z,
            bloq_circuit::Pauli::X,
        ]
        .map(|pauli| InstanceBoundaryOperator {
            instance: TemplateInstanceId(0),
            face: BoundaryFace::Input,
            operator: [(IVec2::ZERO, pauli)].into_iter().collect(),
        });
        let boundaries = operators.clone().map(|operator| {
            let mut fragment = BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                vec![operator],
            ));
            fragment.activation = Some(0);
            let node = assembly.bloq.add_node(fragment);
            assembly
                .bloq
                .add_edge(BloqNodeId(3), node, BloqEdge::value(0));
            Input::Node(node)
        });
        let mut readout = |inputs: Vec<Input>| {
            emission
                .readout(
                    &mut assembly,
                    inputs
                        .into_iter()
                        .map(|input| (input, ValueRole::Data))
                        .collect(),
                )
                .unwrap()
        };
        let first = readout(vec![
            a,
            boundaries[0],
            a,
            boundaries[1],
            b,
            boundaries[2],
            b,
            other_uses,
            unconditional,
            unconditional,
        ]);
        let second = readout(vec![a, b, unconditional]);
        emission.finish(&mut assembly.bloq).unwrap();
        assembly.bloq.validate().unwrap();

        let actual = assembly
            .bloq
            .top()
            .data_inputs(first)
            .filter(|input| input.output.is_none())
            .flat_map(|input| {
                assembly.bloq[input.producer]
                    .try_classical()
                    .unwrap()
                    .operators()
            })
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(actual, operators);
        for (target, expected_occurrences) in [(first, 9), (second, 4)] {
            let occurrences = std::iter::once(target)
                .chain(
                    assembly
                        .bloq
                        .top()
                        .data_inputs(target)
                        .map(|input| input.producer),
                )
                .map(|node| {
                    assembly.bloq[node]
                        .try_classical()
                        .unwrap()
                        .measurements()
                        .len()
                })
                .sum::<usize>();
            assert_eq!(occurrences, expected_occurrences);
            for owner in [BloqNodeId(0), BloqNodeId(1)] {
                assert!(assembly.bloq.has_path(owner, target));
            }
        }
        for enabled in [false, true] {
            let assignment = ClassicalAssignment::Uniform(enabled);
            for (target, remainder, conditional) in [
                (
                    first,
                    record(1, 1),
                    vec![record(0, 0), record(1, 0), record(1, 1)],
                ),
                (second, record(1, 0), vec![record(0, 0), record(1, 0)]),
            ] {
                let parity = assembly.bloq.resolve_classical(target, assignment).unwrap();
                let expected = (target == second)
                    .then_some(record(0, 1))
                    .into_iter()
                    .chain(enabled.then_some(remainder))
                    .collect::<BTreeSet<_>>();
                assert_eq!(parity.measurements, expected);
                let expected = std::iter::once(record(0, 1))
                    .chain(conditional.into_iter().filter(|_| enabled))
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    assembly
                        .bloq
                        .measurement_dependencies(target, assignment)
                        .unwrap(),
                    expected
                );
            }
        }
    }
}
