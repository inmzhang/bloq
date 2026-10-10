//! Shared physical placement and emission. Signed readouts are lowered separately.

use std::collections::BTreeMap;

use bloq_graph::{BlockGraph, BlockKind};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, NodeDetector, QuantumNode, QuantumTimeline, SubGraph,
    TemporalPipeRef,
    lowering::{NodeRestart, TemplateInstance, TemplateInstanceId},
};
use glam::IVec3;
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use super::{
    BloqLowerContext, ChunkSiteSource, LowerPlan, SpatialPipeRef, TemplateInstanceAllocator,
    TemplatePlan, TemplateRemapper,
    classify::{NodeLowering, classify_nodes},
    detector::attach_temporal_component_detectors,
    place::{PlacedInstances, place_instances},
    region,
};
use crate::{
    BlockLayout, CompileError, add_resource,
    block::{LoweringTemplateId, LoweringTemplatePool},
    compile::CompiledTemplateMap,
    spatial_port::SpatialPortExpansionMap,
};

/// Compiled geometry shared by placement, flow composition, and readout resolution.
#[derive(Clone, Copy)]
pub(crate) struct PhysicalInput<'a> {
    pub graph: &'a BlockGraph,
    pub plan: &'a LowerPlan,
    pub compiled: &'a CompiledTemplateMap,
    pub spatial_port_templates: &'a CompiledTemplateMap,
    pub spatial_ports: &'a SpatialPortExpansionMap,
    pub temporal_templates: &'a crate::FxMap<TemplatePlan, LoweringTemplateId>,
    pub wall_templates: &'a crate::FxMap<SpatialPipeRef, LoweringTemplateId>,
    pub template_pool: &'a LoweringTemplatePool,
    pub layout: BlockLayout,
}

/// The two globally allocated realizations of one selective site.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectiveArmInstances {
    pub when_true: TemplateInstanceId,
    pub when_false: TemplateInstanceId,
}

pub(crate) struct PhysicalPlacements {
    pub instances: crate::FxMap<ChunkSiteSource, TemplateInstanceId>,
    pub selectives: crate::FxMap<IVec3, SelectiveArmInstances>,
    pub regions: crate::FxMap<IVec3, BloqNodeId>,
}

/// Physical nodes and their source ownership, ready for semantic lowering.
pub(crate) struct PhysicalProgram {
    pub bloq: Bloq,
    pub placements: PhysicalPlacements,
    pub(super) node_map: Vec<Option<BloqNodeId>>,
}

/// Owns placed instances and the allocator for region-owned instances.
/// Static flow/readout passes borrow this state; physical emission consumes it.
pub(crate) struct PlacedPlan<'a> {
    input: PhysicalInput<'a>,
    classified: Vec<NodeLowering>,
    lower_order: Vec<NodeIndex>,
    emit_order: Vec<usize>,
    placed: PlacedInstances,
    instance_allocator: TemplateInstanceAllocator,
    node_detectors: Vec<Vec<NodeDetector>>,
    node_restarts: Vec<Vec<NodeRestart>>,
}

impl<'a> PlacedPlan<'a> {
    pub(crate) fn new(input: PhysicalInput<'a>) -> Result<Self, CompileError> {
        let lower_order = input.plan.emission_order();
        let node_count = input.plan.graph().node_count();
        let mut emit_order = vec![0; node_count];
        for (rank, node) in lower_order.iter().enumerate() {
            emit_order[node.index()] = rank;
        }
        let classified = classify_nodes(
            input.plan,
            input.compiled,
            input.spatial_port_templates,
            input.temporal_templates,
            input.wall_templates,
        );
        let mut instance_allocator = TemplateInstanceAllocator::default();
        let placed = place_instances(
            &classified,
            &lower_order,
            input.layout,
            &mut instance_allocator,
        )?;
        Ok(Self {
            input,
            classified,
            lower_order,
            emit_order,
            placed,
            instance_allocator,
            node_detectors: vec![Vec::new(); node_count],
            node_restarts: vec![Vec::new(); node_count],
        })
    }

    pub(super) fn context(&mut self) -> BloqLowerContext<'_> {
        BloqLowerContext {
            input: self.input,
            classified: &self.classified,
            emit_order: &self.emit_order,
            node_instances: &self.placed.node_instances,
            node_detectors: &mut self.node_detectors,
            node_restarts: &mut self.node_restarts,
        }
    }

    pub(super) fn compose_detectors(&mut self) -> Result<(), CompileError> {
        attach_temporal_component_detectors(self.input.plan, &mut self.context())
    }

    /// Emit geometry and local region checks. Ordinary lowering reserves its
    /// semantic observable range before the T regions' auxiliary readouts.
    pub(crate) fn emit(mut self, first_observable: u32) -> Result<PhysicalProgram, CompileError> {
        let PhysicalInput {
            graph,
            plan,
            template_pool,
            layout,
            ..
        } = self.input;
        let node_count = plan.graph().node_count();
        let instance_by_site = self
            .placed
            .node_instances
            .iter()
            .flatten()
            .map(|instance| (instance.site_source, instance.instance.id))
            .collect::<crate::FxMap<_, _>>();
        let mut bloq = Bloq::new();
        let mut program_template_ids = crate::FxMap::default();
        let mut selective_arm_instances: crate::FxMap<IVec3, SelectiveArmInstances> =
            crate::FxMap::default();
        let mut node_map = vec![None; node_count];
        let mut region_node_by_pos: crate::FxMap<IVec3, BloqNodeId> = crate::FxMap::default();
        // GAP observable indices allocate after the generator range (U17 §5.1), two
        // per T block in emission order.
        let mut next_t_observable = first_observable;
        for plan_index in std::mem::take(&mut self.lower_order) {
            match &self.classified[plan_index.index()] {
                // Each Pauli realization keeps its own records. Registry assembly
                // supplies the complementary membership and seam-check guards.
                NodeLowering::Selective {
                    pos,
                    when_true,
                    when_false,
                } => {
                    let mut node =
                        BloqNode::from_members(vec![bloq_ir::SourceBlockRef { pos: *pos }]);
                    let mut remapper =
                        TemplateRemapper::new(&mut bloq, &mut program_template_ids, template_pool);
                    let offset = layout.offset(*pos)?;
                    let [when_true, when_false] = [when_true, when_false].map(|template| {
                        let instance = TemplateInstance::new(
                            self.instance_allocator.allocate(),
                            remapper.remap(*template),
                            offset,
                        );
                        node.expect_quantum_mut().instances.push(instance);
                        instance.id
                    });
                    let id = bloq.add_node(node);
                    node_map[plan_index.index()] = Some(id);
                    selective_arm_instances.insert(
                        *pos,
                        SelectiveArmInstances {
                            when_true,
                            when_false,
                        },
                    );
                }

                // A T block becomes a `RepeatUntilSuccess` region: cultivation +
                // escape quantum nodes, the intra-body seam restarts, and the GAP
                // observable recipes feeding the explicit Flip retry predicate (U17 §5.3).
                NodeLowering::TRegion {
                    pos,
                    cultivation,
                    escape,
                } => {
                    let escape_id = {
                        let taken =
                            std::mem::take(&mut self.placed.node_instances[plan_index.index()]);
                        let [instance] = <[_; 1]>::try_from(taken)
                            .expect("T plan node carries exactly its escape instance");
                        instance.instance.id
                    };
                    let observable_base = next_t_observable;
                    next_t_observable = add_resource(
                        "observable IDs",
                        next_t_observable as usize,
                        2,
                        u32::MAX as usize,
                    )? as u32;
                    let region = region::build_t_region(
                        *pos,
                        region::TStage {
                            template: *cultivation,
                            instance: self.placed.t_cultivation_instances[pos],
                        },
                        region::TStage {
                            template: *escape,
                            instance: escape_id,
                        },
                        observable_base,
                        &mut TemplateRemapper::new(
                            &mut bloq,
                            &mut program_template_ids,
                            template_pool,
                        ),
                        layout,
                    )?;
                    let id = bloq.add_node(BloqNode::region(region));
                    node_map[plan_index.index()] = Some(id);
                    region_node_by_pos.insert(*pos, id);
                }

                NodeLowering::Fixed { .. }
                | NodeLowering::TemporalPipe { .. }
                | NodeLowering::SpatialPort { .. } => {
                    let node = self.take_quantum_node(
                        plan_index,
                        &mut TemplateRemapper::new(
                            &mut bloq,
                            &mut program_template_ids,
                            template_pool,
                        ),
                    )?;
                    let id = bloq.add_node(node);
                    node_map[plan_index.index()] = Some(id);
                }
            }
        }

        // Lowering coverage is total: `lower_order` is a toposort of the whole plan
        // graph and every match arm above records an id for its node, so both
        // endpoints of every plan edge resolve.
        let mut quantum_edges = BTreeMap::<(BloqNodeId, BloqNodeId), Vec<TemporalPipeRef>>::new();
        for edge in plan.graph().edge_references() {
            let endpoints = (
                node_map[edge.source().index()].expect("every plan node was lowered"),
                node_map[edge.target().index()].expect("every plan node was lowered"),
            );
            quantum_edges
                .entry(endpoints)
                .or_default()
                .extend(edge.weight().temporal_pipes());
        }
        for ((source, target), mut pipes) in quantum_edges {
            pipes.sort_unstable_by_key(|pipe| (pipe.src.to_array(), pipe.dst.to_array()));
            bloq.add_edge(source, target, BloqEdge::quantum(pipes));
        }

        induce_walking_choreography_order_edges(graph, plan, &node_map, bloq.top_mut())?;
        // Composed detectors can read across nodes without a physical seam.
        induce_detector_order_edges(bloq.top_mut());

        Ok(PhysicalProgram {
            bloq,
            placements: PhysicalPlacements {
                instances: instance_by_site,
                selectives: selective_arm_instances,
                regions: region_node_by_pos,
            },
            node_map,
        })
    }

    fn take_quantum_node(
        &mut self,
        plan_index: NodeIndex,
        remapper: &mut TemplateRemapper<'_>,
    ) -> Result<BloqNode, CompileError> {
        let PhysicalInput {
            graph,
            plan,
            spatial_ports,
            layout,
            ..
        } = self.input;
        let classified = &self.classified;
        let node_instances = &mut self.placed.node_instances;
        let node_detectors = &mut self.node_detectors;
        let node_restarts = &mut self.node_restarts;
        debug_assert!(matches!(
            classified[plan_index.index()],
            NodeLowering::Fixed { .. }
                | NodeLowering::TemporalPipe { .. }
                | NodeLowering::SpatialPort { .. }
        ));
        let timeline = matches!(classified[plan_index.index()], NodeLowering::Fixed { .. })
            .then(|| block_component_timeline(graph, plan, plan_index, layout.distance()))
            .transpose()?
            .flatten();
        let instances = std::mem::take(&mut node_instances[plan_index.index()])
            .into_iter()
            .map(|instance| bloq_ir::lowering::TemplateInstance {
                id: instance.instance.id,
                template_id: remapper.remap(instance.instance.template),
                offset: instance.instance.offset,
                provenance: match instance.site_source {
                    ChunkSiteSource::Block(source) if spatial_ports.contains_key(&source) => {
                        let port = spatial_ports[&source];
                        bloq_ir::InstanceProvenance::SpatialPortSubstitution {
                            source,
                            role: port.role,
                            part: bloq_ir::SpatialPortPart::Cube,
                        }
                    }
                    ChunkSiteSource::SpatialPort(source) => {
                        let port = spatial_ports[&source];
                        bloq_ir::InstanceProvenance::SpatialPortSubstitution {
                            source,
                            role: port.role,
                            part: bloq_ir::SpatialPortPart::TemporalPort,
                        }
                    }
                    ChunkSiteSource::Block(source) => bloq_ir::InstanceProvenance::Block { source },
                    ChunkSiteSource::TemporalPipe(pipe) => bloq_ir::InstanceProvenance::Pipe {
                        src: pipe.src,
                        dst: pipe.dst,
                    },
                    ChunkSiteSource::SpatialPipe(pipe) => bloq_ir::InstanceProvenance::Pipe {
                        src: pipe.src,
                        dst: pipe.dst,
                    },
                },
            })
            .collect();
        let detectors = std::mem::take(&mut node_detectors[plan_index.index()]);
        let restarts = std::mem::take(&mut node_restarts[plan_index.index()]);
        Ok(BloqNode::quantum(QuantumNode {
            instances,
            detectors,
            detector_bundles: Vec::new(),
            restarts,
            timeline,
            guards: Vec::new(),
        })
        .with_provenance(plan.graph()[plan_index].provenance.clone()))
    }
}

fn block_component_timeline(
    graph: &BlockGraph,
    plan: &LowerPlan,
    node: NodeIndex,
    distance: u32,
) -> Result<Option<QuantumTimeline>, CompileError> {
    let first = plan.graph()[node]
        .block_members()
        .first()
        .expect("fixed plan nodes have block members");
    let block = graph
        .get_block(first.pos)
        .expect("lower-plan block members come from the source graph");
    let layer_round_ends = match block.kind() {
        BlockKind::Walking(_) => vec![distance + 1, 2 * (distance + 1)],
        BlockKind::PatchRotation(_) => vec![distance, 2 * distance],
        BlockKind::Cube(_) if block.height_cells() > 1 => {
            let rounds = crate::signature::cube_rounds(block, distance)?;
            let packs_from_low = plan.graph()[node].incoming_cut;
            cube_layer_round_ends(rounds, block.height_cells(), distance, packs_from_low)
        }
        _ => return Ok(None),
    };
    Ok(Some(QuantumTimeline { layer_round_ends }))
}

/// Add the read-after-measure dependencies whose owners live in the same
/// region arm. Cross-boundary predecessor instances are intentionally absent
/// from `owners`; the parent quantum cut edge orders those before the region.
fn induce_detector_order_edges(body: &mut SubGraph) {
    let owners = body
        .nodes()
        .flat_map(|(node, weight)| {
            weight.try_quantum().into_iter().flat_map(move |quantum| {
                quantum
                    .instances
                    .iter()
                    .map(move |instance| (instance.id, node))
            })
        })
        .collect::<crate::FxMap<_, _>>();
    let mut order = Vec::new();
    for (target, weight) in body.nodes() {
        let Some(quantum) = weight.try_quantum() else {
            continue;
        };
        for parity in quantum
            .detectors
            .iter()
            .map(|detector| &detector.parity)
            .chain(quantum.restarts.iter().map(|restart| &restart.parity))
        {
            for measurement in parity.measurements() {
                if let Some(&owner) = owners.get(&measurement.instance)
                    && owner != target
                {
                    order.push((owner, target));
                }
            }
        }
    }
    order.sort_unstable();
    order.dedup();
    for (owner, target) in order {
        if !body.has_path(owner, target) {
            assert!(
                !body.has_path(target, owner),
                "a detector cannot read a measurement from its causal successor"
            );
            body.add_edge(owner, target, BloqEdge::Order);
        }
    }
}

/// Parallel walking blocks share transit qubits. If one walk enters another
/// walk's source cell, the latter must finish vacating it first; source order is
/// otherwise rotation-dependent and can corrupt the merged physical schedule.
fn induce_walking_choreography_order_edges(
    graph: &BlockGraph,
    plan: &LowerPlan,
    node_map: &[Option<BloqNodeId>],
    body: &mut SubGraph,
) -> Result<(), CompileError> {
    for &position in plan.node_by_block().keys() {
        let block = graph.get_block(position).expect("planned block exists");
        let BlockKind::Walking(kind) = block.kind() else {
            continue;
        };
        let destination_source = kind.try_end_position(block.pos())? - IVec3::Z;
        if !graph
            .get_block(destination_source)
            .is_some_and(|block| matches!(block.kind(), BlockKind::Walking(_)))
        {
            continue;
        }
        let Some(entering) = node_map[plan.node_by_block()[&block.pos()].index()] else {
            continue;
        };
        let Some(&vacating_plan) = plan.node_by_block().get(&destination_source) else {
            continue;
        };
        let Some(vacating) = node_map[vacating_plan.index()] else {
            continue;
        };
        if entering != vacating {
            body.add_edge(vacating, entering, BloqEdge::Order);
        }
    }
    Ok(())
}

/// Split a tall cube's rounds across its occupied source layers. All layers on
/// the anchored side receive at most `distance` rounds; any residual (including
/// overflow beyond that cap) belongs to the far layer.
fn cube_layer_round_ends(rounds: u32, cells: u32, distance: u32, packs_from_low: bool) -> Vec<u32> {
    debug_assert!(cells > 1);
    (1..cells)
        .map(|layer| {
            if packs_from_low {
                rounds.min(layer * distance)
            } else {
                rounds.saturating_sub((cells - layer) * distance)
            }
        })
        .chain([rounds])
        .collect()
}

#[cfg(test)]
mod tests {
    use bloq_circuit::DetectorParity;
    use bloq_ir::TemplateId;
    use bloq_ir::lowering::InstanceMeasurement;
    use glam::IVec2;

    use super::*;

    fn local_quantum(instance: TemplateInstanceId) -> BloqNode {
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(instance, TemplateId(0), IVec2::ZERO));
        node
    }

    #[test]
    fn local_detector_reference_induces_owner_order_inside_branch_arm() {
        let first_instance = TemplateInstanceId(1);
        let second_instance = TemplateInstanceId(2);
        let mut body = SubGraph::new();
        let first = body.add_node(local_quantum(first_instance));
        let mut second_node = local_quantum(second_instance);
        second_node
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: DetectorParity::from_measurements([
                    InstanceMeasurement {
                        instance: first_instance,
                        measurement: 0,
                    },
                    InstanceMeasurement {
                        instance: second_instance,
                        measurement: 0,
                    },
                ]),
                coords: None,
            });
        let second = body.add_node(second_node);

        induce_detector_order_edges(&mut body);

        assert!(body.has_path(first, second));
        assert!(!body.has_path(second, first));
    }
}
