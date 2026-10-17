//! Shared, immutable preparation for the whole-program walk.

use bloq_ir::circuit::{Op, PauliMap};
use bloq_ir::lowering::{InstanceMeasurement, InstantiationOptions, TemplateInstanceId};
use bloq_ir::{
    Bloq, BloqNode, BloqNodeId, BloqNodeKind, ClassicalNode, LevelPath, QuantumNode, SubGraph,
    TemplateId, ValidatedPlans,
};
use glam::IVec2;
use rustc_hash::FxHashMap;
use std::{cell::RefCell, rc::Rc};

use super::ExecError;
use super::circuit::{
    LayoutCtx, PreparedOps, prepare_boundary_operator, prepare_ops, validate_record_remap,
};
use crate::backend::PauliString;

type ScopedNode = (LevelPath, BloqNodeId);

pub(super) struct SelectedOps {
    pub(super) quantum: QuantumNode,
    pub(super) ops: PreparedOps,
    pub(super) aliases: Vec<(u32, u32)>,
}

type SelectionKey = (ScopedNode, Vec<(u32, bool)>);

/// One quantum node's merged op stream, in emit order, with the map that
/// resolves its measurement records to global record ids. A build-time
/// intermediate: [`PreparedProgramCore`] translates it and keeps only the
/// result.
struct NodeOpStream {
    /// The enclosing region chain (empty at the top level).
    scope: LevelPath,
    /// The quantum node this stream belongs to (level-local id).
    node: BloqNodeId,
    /// The node's merged emission-plan entry-body ops, already translated to
    /// global coordinates (instance offsets applied at lowering).
    ops: Vec<Op>,
    /// Merged-local measurement id to global record id. Preparation proves this
    /// map covers every record produced or consumed by `ops`.
    local_to_global: FxHashMap<u32, u32>,
}

/// Flattened facts the whole-program walk runs on.
pub(super) struct PreparedProgramCore {
    pub(super) bloq: Bloq,
    pub(super) coord_index: FxHashMap<IVec2, u32>,
    pub(super) global_meas: FxHashMap<InstanceMeasurement, u32>,
    pub(super) instance_offset: FxHashMap<TemplateInstanceId, IVec2>,
    pub(super) record_count: u32,
    /// Every quantum node's stream in engine-instruction form, in emit order.
    /// Translating a [`NodeOpStream`] is shot-independent, so it happens once
    /// here and every shot replays the result; the streams themselves are not
    /// retained, since nothing downstream reads them back.
    prepared: Vec<PreparedOps>,
    node_index: FxHashMap<ScopedNode, usize>,
    /// Deterministic SEM-ORD schedule for every level, keyed by its scope.
    emit_orders: FxHashMap<LevelPath, Vec<BloqNodeId>>,
    /// Boundary maps translated to engine indices once for all shots.
    boundary_operators: FxHashMap<PauliMap, PauliString>,
    selected: RefCell<FxHashMap<SelectionKey, Rc<SelectedOps>>>,
}

impl PreparedProgramCore {
    pub(super) fn build(bloq: &Bloq) -> Result<Self, ExecError> {
        // The op streams are built from the flattened program. SEM-MERGE runs
        // on that graph; flattening preserves the scoped node addresses used by
        // `collect_streams`. Full validation is an explicit caller operation.
        let mut bloq = bloq.clone();
        bloq.flatten()?;
        let plans = bloq.emission_plans(&InstantiationOptions::default())?;
        Self::assemble(bloq, &plans)
    }

    /// Assemble the shared core from an already-flattened `bloq`, reading each
    /// quantum node's stream out of its emission plan.
    fn assemble(bloq: Bloq, plans: &ValidatedPlans) -> Result<Self, ExecError> {
        let mut global_meas = FxHashMap::default();
        let mut instance_offset = FxHashMap::default();
        let mut instance_templates = FxHashMap::default();
        let mut record_count = 0;
        let mut streams_by_node = FxHashMap::default();
        collect_streams(
            &bloq,
            bloq.top(),
            &LevelPath::default(),
            plans,
            &mut global_meas,
            &mut instance_offset,
            &mut instance_templates,
            &mut record_count,
            &mut streams_by_node,
        )?;
        for (_, level) in bloq.levels() {
            for (_, quantum) in level.quantum_nodes() {
                bloq.check_detector_bundle_bindings(quantum, |instance| {
                    instance_templates.get(&instance).copied()
                })?;
            }
        }

        let mut node_streams = Vec::with_capacity(streams_by_node.len());
        let mut emit_orders = FxHashMap::default();
        order_streams(
            bloq.top(),
            &LevelPath::default(),
            &mut streams_by_node,
            &mut node_streams,
            &mut emit_orders,
        )?;
        if !streams_by_node.is_empty() {
            return Err(ExecError::MalformedGraph(
                "prepared stream has no scheduled quantum node",
            ));
        }
        let node_index = node_streams
            .iter()
            .enumerate()
            .map(|(index, stream)| ((stream.scope.clone(), stream.node), index))
            .collect();

        let coord_index: FxHashMap<IVec2, u32> = bloq
            .sorted_layout_coords()?
            .iter()
            .enumerate()
            .map(|(index, &coord)| {
                u32::try_from(index)
                    .map(|index| (coord, index))
                    .map_err(|_| ExecError::IdOverflow("qubit"))
            })
            .collect::<Result<_, _>>()?;

        // Instance offsets were already applied at lowering, so the whole
        // program shares one un-shifted layout — the same one every shot runs
        // against, which is why the translation below can be hoisted out of the
        // shot loop entirely.
        let layout = LayoutCtx {
            coord_to_index: &coord_index,
            qubit_count: coord_index.len(),
        };
        let prepared = node_streams
            .iter()
            .map(|stream| {
                // `collect_streams` proved the remap total over this stream's
                // records, so the lookup cannot miss.
                prepare_ops(&stream.ops, &layout, &|local| {
                    stream.local_to_global[&local]
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut boundary_operators = FxHashMap::default();
        prepare_boundary_operators(bloq.top(), &layout, &mut boundary_operators)?;

        Ok(Self {
            bloq,
            coord_index,
            global_meas,
            instance_offset,
            record_count,
            prepared,
            node_index,
            emit_orders,
            boundary_operators,
            selected: RefCell::default(),
        })
    }

    pub(super) fn selected_ops(
        &self,
        scope: &LevelPath,
        id: BloqNodeId,
        node: &BloqNode,
        inputs: &FxHashMap<u32, bool>,
    ) -> Result<Rc<SelectedOps>, ExecError> {
        let mut values = inputs
            .iter()
            .map(|(&slot, &value)| (slot, value))
            .collect::<Vec<_>>();
        values.sort_unstable();
        let key = ((scope.clone(), id), values);
        if let Some(selected) = self.selected.borrow().get(&key) {
            return Ok(Rc::clone(selected));
        }
        let selected = node
            .select_quantum_members(|slot| inputs.get(&slot).copied())
            .map_err(ExecError::InvalidCircuit)?;
        let plan = selected
            .emission_plan(self.bloq.templates())
            .map_err(ExecError::InvalidCircuit)?;
        let mut local_to_global = FxHashMap::default();
        let mut aliases = Vec::new();
        for (local, sources) in plan.grouped_measurements() {
            let canonical = sources
                .iter()
                .map(|source| self.global_meas[source])
                .min()
                .expect("measurement group is nonempty");
            local_to_global.insert(local, canonical);
            aliases.extend(
                sources
                    .iter()
                    .map(|source| (self.global_meas[source], canonical)),
            );
        }
        let ops = plan
            .circuit
            .body(plan.circuit.entry_body())
            .expect("entry body exists")
            .ops();
        validate_record_remap(ops, &local_to_global)?;
        let layout = LayoutCtx {
            coord_to_index: &self.coord_index,
            qubit_count: self.coord_index.len(),
        };
        let prepared = Rc::new(SelectedOps {
            quantum: selected.expect_quantum().clone(),
            ops: prepare_ops(ops, &layout, &|local| local_to_global[&local])?,
            aliases,
        });
        let mut cache = self.selected.borrow_mut();
        // ponytail: 64 plans retained across the program; use LRU if measured
        // mask churn warrants retaining hot plans across an eviction.
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(key, Rc::clone(&prepared));
        Ok(prepared)
    }

    /// The engine-instruction form of a quantum node's stream.
    pub(super) fn prepared_ops(
        &self,
        scope: &LevelPath,
        node: BloqNodeId,
    ) -> Result<&PreparedOps, ExecError> {
        self.node_index
            .get(&(scope.clone(), node))
            .and_then(|&index| self.prepared.get(index))
            .ok_or(ExecError::MalformedGraph(
                "quantum node has no prepared stream",
            ))
    }

    /// The prepared deterministic schedule for one level.
    pub(super) fn emit_order(&self, scope: &LevelPath) -> Result<&[BloqNodeId], ExecError> {
        self.emit_orders
            .get(scope)
            .map(Vec::as_slice)
            .ok_or(ExecError::MalformedGraph(
                "level has no prepared emit order",
            ))
    }

    /// A boundary map already translated to the engine's dense layout.
    pub(super) fn boundary_operator(&self, map: &PauliMap) -> Result<&PauliString, ExecError> {
        self.boundary_operators
            .get(map)
            .ok_or(ExecError::MalformedGraph(
                "boundary map has no prepared engine operator",
            ))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive preparation threads one shared walk state"
)]
fn collect_streams(
    bloq: &Bloq,
    level: &SubGraph,
    scope: &LevelPath,
    plans: &ValidatedPlans,
    global_meas: &mut FxHashMap<InstanceMeasurement, u32>,
    instance_offset: &mut FxHashMap<TemplateInstanceId, IVec2>,
    instance_templates: &mut FxHashMap<TemplateInstanceId, TemplateId>,
    next_record: &mut u32,
    out: &mut FxHashMap<ScopedNode, NodeOpStream>,
) -> Result<(), ExecError> {
    // Node-id order deliberately preserves the established global record layout.
    for (id, node) in level.nodes() {
        match &node.kind {
            BloqNodeKind::Quantum(quantum) => {
                for instance in &quantum.instances {
                    instance_offset.insert(instance.id, instance.offset);
                    instance_templates.insert(instance.id, instance.template_id);
                }
                if !quantum.guards.is_empty() {
                    for instance in &quantum.instances {
                        let template = bloq
                            .templates()
                            .get(instance.template_id)
                            .ok_or(ExecError::MalformedGraph("missing selected template"))?;
                        for record in template.circuit.meas_registry().records() {
                            let global = *next_record;
                            *next_record = global
                                .checked_add(1)
                                .ok_or(ExecError::IdOverflow("record"))?;
                            global_meas.insert(
                                InstanceMeasurement {
                                    instance: instance.id,
                                    measurement: record.id,
                                },
                                global,
                            );
                        }
                    }
                    continue;
                }
                // The plan contains the materialized merge used for execution.
                let plan = plans.get(scope, id).ok_or(ExecError::MalformedGraph(
                    "quantum node has no emission plan",
                ))?;
                // Mint one global record per merged-local id, in the canonical
                // grouping order (ascending local; see `grouped_measurements`), so
                // the established global record layout is preserved exactly.
                let mut local_to_global = FxHashMap::default();
                for (local, sources) in plan.grouped_measurements() {
                    let global = *next_record;
                    *next_record = global
                        .checked_add(1)
                        .ok_or(ExecError::IdOverflow("record"))?;
                    local_to_global.insert(local, global);
                    for measurement in sources {
                        global_meas.insert(measurement, global);
                    }
                }
                let ops = plan
                    .circuit
                    .body(plan.circuit.entry_body())
                    .expect("entry body exists")
                    .ops()
                    .to_vec();
                validate_record_remap(&ops, &local_to_global)?;
                if out
                    .insert(
                        (scope.clone(), id),
                        NodeOpStream {
                            scope: scope.clone(),
                            node: id,
                            ops,
                            local_to_global,
                        },
                    )
                    .is_some()
                {
                    return Err(ExecError::MalformedGraph(
                        "duplicate scoped quantum node identity",
                    ));
                }
            }
            BloqNodeKind::Region(region) => {
                for (selector, body) in region.bodies() {
                    collect_streams(
                        bloq,
                        body,
                        &scope.child(id, selector),
                        plans,
                        global_meas,
                        instance_offset,
                        instance_templates,
                        next_record,
                        out,
                    )?;
                }
            }
            BloqNodeKind::Classical(_) => {}
        }
    }
    Ok(())
}

fn order_streams(
    level: &SubGraph,
    scope: &LevelPath,
    by_node: &mut FxHashMap<ScopedNode, NodeOpStream>,
    out: &mut Vec<NodeOpStream>,
    emit_orders: &mut FxHashMap<LevelPath, Vec<BloqNodeId>>,
) -> Result<(), ExecError> {
    let emit_order = level.deterministic_emit_order()?;
    for &id in &emit_order {
        let node = level.node(id).expect("emit order yields live nodes");
        match &node.kind {
            BloqNodeKind::Quantum(quantum) if quantum.guards.is_empty() => out.push(
                by_node
                    .remove(&(scope.clone(), id))
                    .ok_or(ExecError::MalformedGraph(
                        "scheduled quantum node has no prepared stream",
                    ))?,
            ),
            BloqNodeKind::Region(region) => {
                for (selector, body) in region.bodies() {
                    order_streams(body, &scope.child(id, selector), by_node, out, emit_orders)?;
                }
            }
            BloqNodeKind::Quantum(_) | BloqNodeKind::Classical(_) => {}
        }
    }
    emit_orders.insert(scope.clone(), emit_order);
    Ok(())
}

fn prepare_boundary_operators(
    level: &SubGraph,
    layout: &LayoutCtx<'_>,
    out: &mut FxHashMap<PauliMap, PauliString>,
) -> Result<(), ExecError> {
    for (_, node) in level.nodes() {
        if let Some(ClassicalNode::Observable { operators, .. }) = node.try_classical() {
            for operator in operators {
                if !out.contains_key(&operator.operator) {
                    let prepared = prepare_boundary_operator(&operator.operator, layout)?;
                    out.insert(operator.operator.clone(), prepared);
                }
            }
        }
        if let Some(region) = node.try_region() {
            for (_, body) in region.bodies() {
                prepare_boundary_operators(body, layout, out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::quantum_node;
    use bloq_ir::circuit::{CoordCircuit, PauliBasis};
    use bloq_ir::lowering::BloqTemplate;
    use bloq_ir::{BloqEdge, ClassicalExpr, RegionNode};

    fn nested_bloq() -> (Bloq, BloqNodeId, BloqNodeId, BloqNodeId, u32) {
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));

        let top = bloq.add_node(quantum_node(template, 0, IVec2::ZERO));
        let mut body = SubGraph::new();
        let nested = body.add_node(quantum_node(template, 1, IVec2::new(1, 0)));
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
            body,
        }));
        bloq.add_edge(top, region, BloqEdge::Order);
        (bloq, top, region, nested, measurement)
    }

    /// Two nodes at different levels share a level-local id, so only the scope
    /// tells their streams apart. Each must get its own instructions (their
    /// instances sit at different coordinates) and its own global record.
    #[test]
    fn scoped_node_index_distinguishes_level_local_ids() {
        let (bloq, top, region, nested, _measurement) = nested_bloq();
        assert_eq!(top, nested);
        let core = PreparedProgramCore::build(&bloq).unwrap();
        let top_scope = LevelPath::default();
        let nested_scope = top_scope.child(region, bloq_ir::BodySelector::Body);

        let top_stream = core.prepared_ops(&top_scope, top).unwrap();
        let nested_stream = core.prepared_ops(&nested_scope, nested).unwrap();
        assert_ne!(top_stream, nested_stream);
        assert_eq!(top_stream.record_ids(), [0]);
        assert_eq!(nested_stream.record_ids(), [1]);
        assert_eq!(core.emit_order(&top_scope).unwrap(), [top, region]);
        assert_eq!(core.emit_order(&nested_scope).unwrap(), [nested]);
    }

    #[test]
    fn record_allocation_rejects_exhaustion_without_aliasing() {
        for guarded in [false, true] {
            let (mut bloq, top, ..) = nested_bloq();
            if guarded {
                bloq.top_mut()
                    .node_mut(top)
                    .unwrap()
                    .expect_quantum_mut()
                    .guards
                    .push(bloq_ir::QuantumGuard::default());
            }
            let plans = bloq
                .emission_plans(&InstantiationOptions::default())
                .unwrap();
            let mut next_record = u32::MAX;
            let mut measurements = FxHashMap::default();
            let result = collect_streams(
                &bloq,
                bloq.top(),
                &LevelPath::default(),
                &plans,
                &mut measurements,
                &mut FxHashMap::default(),
                &mut FxHashMap::default(),
                &mut next_record,
                &mut FxHashMap::default(),
            );
            assert!(matches!(result, Err(ExecError::IdOverflow("record"))));
            assert_eq!(next_record, u32::MAX);
            assert!(measurements.is_empty());
        }
    }
}
