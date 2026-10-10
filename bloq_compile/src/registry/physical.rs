//! Compile local realizations on a common component schedule.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use bloq_graph::{
    Block, BlockGraph, BlockKind, GuardedTopology, ModuleCertificationLimits, UDirection,
};
use bloq_ir::Bloq;
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId,
};
use glam::IVec3;
use petgraph::unionfind::UnionFind;

use crate::block::{LoweringTemplateId, LoweringTemplatePool};
use crate::compile::{CompileContext, CompiledTemplateMap};
use crate::lower::{LinkedPlanInput, LowerPlan, PhysicalInput, PlacedPlan, SpatialPipeRef};
use crate::signature::{LayerSchedule, LayerScheduleMap};
use crate::spatial_port::SpatialPortExpansionMap;
use crate::{BlockLayout, CompileError, check_resource};

#[path = "anchors.rs"]
mod anchors;
pub(super) use anchors::GuardedDynamicAnchors;

/// Port-expanded geometry shared by scheduling, template selection, and local
/// physical gateways. It has no selector or readout semantics.
pub(super) struct PhysicalGeometry {
    pub graph: Arc<BlockGraph>,
    pub spatial_ports: SpatialPortExpansionMap,
    /// Complete sites eligible for emission. Remaining blocks are lookup halo.
    positions: Vec<IVec3>,
}

fn geometry(
    cache: &mut crate::FxMap<(usize, IVec3), Arc<PhysicalGeometry>>,
    source: &Arc<BlockGraph>,
    center: IVec3,
    reference_source: &Arc<BlockGraph>,
    reference: &PhysicalGeometry,
) -> Result<Arc<PhysicalGeometry>, CompileError> {
    // Local source views stay alive in GuardedTopology for this preparation.
    let entry = cache.entry((Arc::as_ptr(source) as usize, center));
    let geometry = match entry {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let mut positions = source
                .neighbors(center)
                .into_iter()
                .map(Block::pos)
                .collect::<Vec<_>>();
            positions.push(center);
            positions.sort_unstable_by_key(glam::IVec3::to_array);
            positions.dedup();
            let (graph, spatial_ports) = if Arc::ptr_eq(source, reference_source) {
                // The source Arc can serve many local cases, including Ports.
                // Expand it once globally, never clone it for each local Port.
                let ports = positions
                    .iter()
                    .filter_map(|position| {
                        reference
                            .spatial_ports
                            .get(position)
                            .map(|port| (*position, *port))
                    })
                    .collect();
                (Arc::clone(&reference.graph), ports)
            } else {
                let (graph, ports) = crate::spatial_port::expand_spatial_port_context(
                    source,
                    &positions,
                    &reference.spatial_ports,
                )?;
                let graph = match graph {
                    std::borrow::Cow::Borrowed(_) => Arc::clone(source),
                    std::borrow::Cow::Owned(graph) => Arc::new(graph),
                };
                (graph, ports)
            };
            entry.insert(Arc::new(PhysicalGeometry {
                graph,
                spatial_ports,
                positions,
            }))
        }
    };
    Ok(Arc::clone(geometry))
}

pub(super) struct ScheduledGeometry {
    geometry: Arc<PhysicalGeometry>,
    schedules: LayerScheduleMap,
    incoming_cuts: crate::FxMap<IVec3, bool>,
}

/// Completed physical planning: local registrations refer to realizations on
/// one common schedule. Temporary expansion and assignment indices end here.
pub(super) struct ScheduledFamily {
    pub cases: Vec<ScheduledGeometry>,
    pub sites: Vec<(IVec3, DecisionId, usize)>,
    pub t_sides: crate::FxMap<IVec3, bloq_graph::Direction>,
    pub reference: Arc<PhysicalGeometry>,
    pub anchors: GuardedDynamicAnchors,
}

fn reference_geometry(
    topology: &mut GuardedTopology,
    distance: u32,
) -> Result<(Arc<BlockGraph>, Arc<PhysicalGeometry>), CompileError> {
    let source = topology.project(ONE)?;
    let (graph, mut spatial_ports) =
        crate::spatial_port::expand_spatial_port_topology(&source, distance)?;
    crate::spatial_port::allocate_multiplex_outputs(
        &topology.source,
        distance,
        &mut spatial_ports,
    )?;
    let graph = match graph {
        std::borrow::Cow::Borrowed(_) => Arc::clone(&source),
        std::borrow::Cow::Owned(graph) => Arc::new(graph),
    };
    let mut positions = graph.positions().collect::<Vec<_>>();
    positions.sort_unstable_by_key(glam::IVec3::to_array);
    Ok((
        source,
        Arc::new(PhysicalGeometry {
            graph,
            spatial_ports,
            positions,
        }),
    ))
}

impl ScheduledFamily {
    pub(super) fn new(
        topology: &mut GuardedTopology,
        distance: u32,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, CompileError> {
        let anchors = GuardedDynamicAnchors::new(topology, limits)?;
        let (source, reference) = reference_geometry(topology, distance)?;
        let mut expanded = crate::FxMap::default();
        let envelope = ScheduleEnvelope::new(
            topology,
            limits.max_witness_nodes,
            &mut expanded,
            &source,
            &reference,
        )?;
        let variables = topology
            .witness(ONE)?
            .expect("source has a reachable reference");
        let (schedules, incoming_cuts) = envelope.select(&reference, |root| {
            topology.diagram.evaluate(root, |index| {
                variables.get(&index).copied().unwrap_or(false)
            })
        });
        // Common declarations are emitted once. Only local deviations need a
        // separate bounded context, never another complete source program.
        let mut cases = vec![ScheduledGeometry {
            geometry: Arc::clone(&reference),
            schedules,
            incoming_cuts,
        }];
        let mut sites = Vec::new();
        let variants = topology
            .sites
            .iter()
            .flat_map(|(&position, variants)| {
                variants.iter().map(move |variant| {
                    (
                        IVec3::from_array(position),
                        variant.guard,
                        Arc::clone(&variant.graph),
                    )
                })
            })
            .collect::<Vec<_>>();
        for (position, guard, local_source) in variants {
            let geometry = geometry(&mut expanded, &local_source, position, &source, &reference)?;
            let roots = envelope.selectors(&geometry);
            for (values, guard) in topology.cases(&roots, guard)? {
                let values = roots
                    .iter()
                    .copied()
                    .zip(values)
                    .collect::<crate::FxMap<_, _>>();
                let (schedules, incoming_cuts) = envelope.select(&geometry, |root| {
                    if root == ZERO {
                        false
                    } else if root == ONE {
                        true
                    } else {
                        values[&root]
                    }
                });
                let case = if same_local_geometry(&reference, &geometry)
                    && schedules.iter().all(|(position, schedule)| {
                        cases[0].schedules.get(position) == Some(schedule)
                    })
                    && incoming_cuts.iter().all(|(position, incoming)| {
                        cases[0].incoming_cuts.get(position) == Some(incoming)
                    }) {
                    0
                } else {
                    let case = cases.len();
                    cases.push(ScheduledGeometry {
                        geometry: Arc::clone(&geometry),
                        schedules,
                        incoming_cuts,
                    });
                    case
                };
                sites.push((position, guard, case));
            }
        }
        Ok(Self {
            cases,
            sites,
            t_sides: envelope.t_sides,
            reference,
            anchors,
        })
    }
}

fn same_local_geometry(reference: &PhysicalGeometry, local: &PhysicalGeometry) -> bool {
    local.positions.iter().all(|&position| {
        local.graph.get_block(position) == reference.graph.get_block(position)
            && local.spatial_ports.get(&position) == reference.spatial_ports.get(&position)
            && local.graph.pipes_at(position).count() == reference.graph.pipes_at(position).count()
            && local
                .graph
                .pipes_at(position)
                .all(|pipe| reference.graph.get_pipe(pipe.src(), pipe.dst()) == Some(pipe))
    })
}

pub(super) struct PhysicalProjection {
    pub geometry: Arc<PhysicalGeometry>,
    pub compiled: CompiledTemplateMap,
    pub spatial_templates: CompiledTemplateMap,
    pub pipe_templates: crate::FxMap<bloq_ir::TemporalPipeRef, LoweringTemplateId>,
    pub wall_templates: crate::FxMap<SpatialPipeRef, LoweringTemplateId>,
    pub pool: LoweringTemplatePool,
    pub placements: crate::lower::PhysicalPlacements,
}

impl PhysicalProjection {
    pub(super) fn new(
        context: &CompileContext,
        scheduled: &ScheduledGeometry,
        t_sides: &crate::FxMap<IVec3, bloq_graph::Direction>,
        definitions: &[Arc<crate::compile::PreparedDefinitionObject>],
        sites: &std::collections::HashMap<IVec3, bloq_graph::MaterializedModuleSite>,
    ) -> Result<(Self, Bloq), CompileError> {
        let PhysicalGeometry {
            graph,
            spatial_ports,
            positions,
        } = scheduled.geometry.as_ref();
        let graph = graph.as_ref();
        let schedules = &scheduled.schedules;
        let mut compiled = if definitions.is_empty() {
            let mut signatures = positions
                .iter()
                .map(|&position| {
                    crate::signature::derive_block_signature(
                        graph,
                        graph.get_block(position).expect("complete site exists"),
                        context.config.code_distance(),
                        schedules[&position],
                        t_sides.get(&position).copied(),
                    )
                    .map(|signature| (position, signature))
                })
                .collect::<Result<crate::signature::BlockSignatureMap, _>>()?;
            for port in spatial_ports.values() {
                let signature = signatures
                    .get_mut(&port.source)
                    .expect("expanded Port cube");
                signature.connectivity = signature.connectivity.with_pipe(port.cube_pipe_dir());
            }
            let blocks = positions
                .iter()
                .map(|&position| {
                    graph
                        .get_block(position)
                        .expect("physical geometry positions belong to the graph")
                })
                .collect::<Vec<_>>();
            context.compile_blocks(graph, &blocks, &signatures)?
        } else {
            let relocations = crate::compile::relocate_definition_sites(
                definitions,
                positions.iter().copied(),
                sites,
            );
            crate::compile::select_module_link_templates_with_schedule(
                context.config,
                graph,
                spatial_ports,
                definitions,
                &relocations,
                schedules,
                t_sides,
            )?
        };
        for port in spatial_ports.values() {
            let info = compiled
                .get_mut(&port.source)
                .expect("compiled geometry contains every expanded Port source");
            info.graph_connectivity = info.graph_connectivity.with_pipe(port.cube_pipe_dir());
        }
        let input =
            LinkedPlanInput::from_module_graph(graph, spatial_ports, positions.iter().copied());
        let spatial_templates = context.compile_spatial_port_templates(spatial_ports)?;
        let temporal_templates =
            context.compile_temporal_template_plans(input.temporal_templates())?;
        let wall_templates =
            context.compile_spatial_template_walls(graph, input.spatial_walls())?;
        let pool = context.cache.pool_snapshot();
        let mut plan = LowerPlan::from_linked_input(&input);
        plan.set_incoming_cuts(&scheduled.incoming_cuts);
        let pipe_templates = plan
            .node_by_temporal_pipe()
            .iter()
            .map(|(&pipe, &node)| {
                (
                    pipe,
                    temporal_templates
                        [&plan.graph()[node].template.expect("temporal pipe template")],
                )
            })
            .collect();
        let physical = PlacedPlan::new(PhysicalInput {
            graph,
            plan: &plan,
            compiled: &compiled,
            spatial_port_templates: &spatial_templates,
            spatial_ports,
            temporal_templates: &temporal_templates,
            wall_templates: &wall_templates,
            template_pool: &pool,
            layout: BlockLayout::new(context.config.code_distance()),
        })?
        .emit(0)?;
        let mut bloq = physical.bloq;
        let placements = physical.placements;
        let padding = context.cache.padding.prepare(
            plan.graph().edge_count() != 0,
            context.config.code_distance(),
        )?;
        crate::padding::record_edge_padding(
            &padding,
            &mut bloq,
            graph,
            context.config.code_distance(),
        )?;
        Ok((
            Self {
                geometry: Arc::clone(&scheduled.geometry),
                compiled,
                spatial_templates,
                pipe_templates,
                wall_templates,
                pool,
                placements,
            },
            bloq,
        ))
    }
}

#[derive(Clone, Copy)]
struct ScheduleChoice {
    low: LayerSchedule,
    high: LayerSchedule,
    selector: DecisionId,
    incoming: DecisionId,
}

/// All realizations of a potential spatial component use a common depth.
/// Axis selection only remains conditional when alternative H walls require
/// different orientations. Symbolic connectivity propagates that demand.
struct ScheduleEnvelope {
    choices: crate::FxMap<IVec3, ScheduleChoice>,
    pub t_sides: crate::FxMap<IVec3, bloq_graph::Direction>,
}

struct ScheduleIncidence {
    positions: Vec<IVec3>,
    indices: crate::FxMap<IVec3, usize>,
    groups: UnionFind<usize>,
    edges: BTreeMap<(usize, usize), DecisionId>,
    demands: Vec<[DecisionId; 4]>,
    tall: Vec<bool>,
}

impl ScheduleIncidence {
    fn new(positions: Vec<IVec3>) -> Self {
        let indices = positions
            .iter()
            .enumerate()
            .map(|(index, &position)| (position, index))
            .collect();
        let count = positions.len();
        Self {
            positions,
            indices,
            groups: UnionFind::new(count),
            edges: BTreeMap::new(),
            demands: vec![[ZERO; 4]; count],
            tall: vec![false; count],
        }
    }

    fn add_variant(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
        position: IVec3,
        guard: DecisionId,
        geometry: &PhysicalGeometry,
    ) -> Result<(), CompileError> {
        let index = self.indices[&position];
        let graph = &geometry.graph;
        let block = graph
            .get_block(position)
            .expect("local block exists after Port expansion");
        self.tall[index] |= block.kind().is_cube() && block.height_cells() > 1;
        if geometry
            .spatial_ports
            .get(&position)
            .is_some_and(|port| port.is_input())
            || graph.pipes_at(position).any(|pipe| {
                !pipe.dir().is_spatial()
                    && [pipe.src(), pipe.dst()].into_iter().any(|endpoint| {
                        graph
                            .get_endpoint_block(endpoint)
                            .is_some_and(|other| other.pos().z < position.z)
                    })
            })
        {
            self.demands[index][3] = diagram.apply(BooleanOp::Or, self.demands[index][3], guard)?;
        }
        if matches!(block.kind(), BlockKind::Cube(kind) if kind.is_spatial()) {
            self.demands[index][0] = diagram.apply(BooleanOp::Or, self.demands[index][0], guard)?;
        }
        for pipe in graph
            .pipes_at(position)
            .filter(|pipe| pipe.dir().is_spatial())
        {
            let source = graph
                .get_endpoint_block(pipe.src())
                .expect("graph pipe source endpoint exists")
                .pos();
            let target = graph
                .get_endpoint_block(pipe.dst())
                .expect("graph pipe target endpoint exists")
                .pos();
            let source = self.indices[&source];
            let target = self.indices[&target];
            self.groups.union(source, target);
            let edge = self
                .edges
                .entry((source.min(target), source.max(target)))
                .or_insert(ZERO);
            *edge = diagram.apply(BooleanOp::Or, *edge, guard)?;
            if pipe.is_hadamard() {
                let axis = match pipe.dir().as_udirection() {
                    UDirection::X => 1,
                    UDirection::Y => 2,
                    UDirection::Z => unreachable!(),
                };
                self.demands[index][axis] =
                    diagram.apply(BooleanOp::Or, self.demands[index][axis], guard)?;
            }
        }
        Ok(())
    }
}

impl ScheduleEnvelope {
    pub(super) fn new(
        topology: &mut GuardedTopology,
        limit: usize,
        expanded: &mut crate::FxMap<(usize, IVec3), Arc<PhysicalGeometry>>,
        reference_source: &Arc<BlockGraph>,
        reference: &PhysicalGeometry,
    ) -> Result<Self, CompileError> {
        let positions = topology
            .sites
            .keys()
            .copied()
            .map(IVec3::from_array)
            .collect::<Vec<_>>();
        let mut incidence = ScheduleIncidence::new(positions);
        for (&position, variants) in &topology.sites {
            let position = IVec3::from_array(position);
            for variant in variants {
                let geometry = geometry(
                    expanded,
                    &variant.graph,
                    position,
                    reference_source,
                    reference,
                )?;
                incidence.add_variant(&mut topology.diagram, position, variant.guard, &geometry)?;
            }
        }
        Self::from_incidence(topology, incidence, limit)
    }

    fn from_incidence(
        topology: &mut GuardedTopology,
        incidence: ScheduleIncidence,
        limit: usize,
    ) -> Result<Self, CompileError> {
        let ScheduleIncidence {
            positions,
            indices: _,
            groups,
            edges,
            mut demands,
            tall,
        } = incidence;
        let mut possible = vec![[false; 3]; positions.len()];
        let mut tall_components = vec![false; positions.len()];
        for (index, demand) in demands.iter().enumerate() {
            let group = groups.find(index);
            tall_components[group] |= tall[index];
            for axis in 0..3 {
                possible[group][axis] |= demand[axis] != ZERO;
            }
        }
        let mut neighbors = vec![Vec::new(); positions.len()];
        for ((source, target), guard) in edges {
            neighbors[source].push((target, guard));
            neighbors[target].push((source, guard));
        }
        let mut pending = (0..positions.len())
            .filter(|&index| {
                let group = groups.find(index);
                (possible[group][1] && possible[group][2]) || tall_components[group]
            })
            .collect::<VecDeque<_>>();
        let mut queued = vec![false; positions.len()];
        for &index in &pending {
            queued[index] = true;
        }
        while let Some(source) = pending.pop_front() {
            queued[source] = false;
            for &(target, edge) in &neighbors[source] {
                let mut changed = false;
                let source_demands = demands[source];
                let group = groups.find(source);
                for axis in 1..4 {
                    if (axis < 3 && !(possible[group][1] && possible[group][2]))
                        || (axis == 3 && !tall_components[group])
                    {
                        continue;
                    }
                    let contribution =
                        topology
                            .diagram
                            .apply(BooleanOp::And, edge, source_demands[axis])?;
                    let next = topology.diagram.apply(
                        BooleanOp::Or,
                        demands[target][axis],
                        contribution,
                    )?;
                    changed |= next != demands[target][axis];
                    demands[target][axis] = next;
                }
                if changed && !std::mem::replace(&mut queued[target], true) {
                    pending.push_back(target);
                }
            }
            check_resource(
                "component schedule Boolean nodes",
                topology.diagram.nodes().len(),
                limit,
            )?;
        }
        let mut choices = crate::FxMap::default();
        for (index, &position) in positions.iter().enumerate() {
            let [padded, x, y] = possible[groups.find(index)];
            let (low, high, selector) = match (x, y) {
                (true, true) => {
                    if topology.diagram.apply(
                        BooleanOp::And,
                        demands[index][1],
                        demands[index][2],
                    )? != ZERO
                    {
                        return Err(CompileError::MixedSpatialHadamardUnsupported {
                            pos: position,
                        });
                    }
                    (
                        LayerSchedule::Extended,
                        LayerSchedule::ExtendedY,
                        demands[index][2],
                    )
                }
                (true, false) => (LayerSchedule::Extended, LayerSchedule::Extended, ZERO),
                (false, true) => (LayerSchedule::ExtendedY, LayerSchedule::ExtendedY, ZERO),
                (false, false) if padded => (LayerSchedule::Padded, LayerSchedule::Padded, ZERO),
                (false, false) => (LayerSchedule::Compact, LayerSchedule::Compact, ZERO),
            };
            choices.insert(
                position,
                ScheduleChoice {
                    low,
                    high,
                    selector,
                    incoming: demands[index][3],
                },
            );
        }
        // A common T region reserves one spill cell valid in both source arms.
        let t_sides = crate::signature::assign_linked_t_surgery_sides(
            topology
                .source
                .blocks()
                .filter(|block| block.kind() == BlockKind::T)
                .map(Block::pos)
                .collect(),
            crate::signature::occupied_branch_positions(&topology.source)?,
        )?;
        Ok(Self { choices, t_sides })
    }

    fn selectors(&self, geometry: &PhysicalGeometry) -> Vec<DecisionId> {
        geometry
            .positions
            .iter()
            .flat_map(|position| {
                let choice = self.choices[position];
                let block = geometry
                    .graph
                    .get_block(*position)
                    .expect("physical geometry positions belong to the graph");
                [
                    (choice.low != choice.high).then_some(choice.selector),
                    (block.kind().is_cube() && block.height_cells() > 1).then_some(choice.incoming),
                ]
            })
            .flatten()
            .filter(|&root| root != ZERO && root != ONE)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn select(
        &self,
        geometry: &PhysicalGeometry,
        mut value: impl FnMut(DecisionId) -> bool,
    ) -> (LayerScheduleMap, crate::FxMap<IVec3, bool>) {
        let mut schedules = LayerScheduleMap::default();
        let mut incoming = crate::FxMap::default();
        for &position in &geometry.positions {
            let choice = self.choices[&position];
            schedules.insert(
                position,
                if value(choice.selector) {
                    choice.high
                } else {
                    choice.low
                },
            );
            let block = geometry
                .graph
                .get_block(position)
                .expect("physical geometry positions belong to the graph");
            if block.kind().is_cube() && block.height_cells() > 1 {
                incoming.insert(position, value(choice.incoming));
            }
        }
        (schedules, incoming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_branches_keep_only_one_full_physical_context() {
        let graph = bloq_test::benchmark::structural_branch_workload(40);
        let mut topology =
            GuardedTopology::new(&graph, bloq_graph::ModuleCertificationLimits::DEFAULT).unwrap();
        let family =
            ScheduledFamily::new(&mut topology, 3, ModuleCertificationLimits::DEFAULT).unwrap();
        assert!(family.cases.len() > 1);
        assert!(family.reference.positions.len() > 40);
        assert!(Arc::ptr_eq(&family.cases[0].geometry, &family.reference));
        for case in &family.cases[1..] {
            assert!(case.geometry.positions.len() <= 7);
            assert!(case.geometry.graph.block_count() <= 31);
        }
    }

    #[test]
    fn shared_reference_contexts_reuse_the_expanded_spatial_port_graph() {
        let graph = bloq_graph::GalleryItem::CCZGateTeleport
            .build()
            .flatten()
            .expect("gallery graph expands");
        let mut topology =
            GuardedTopology::new(&graph, ModuleCertificationLimits::DEFAULT).unwrap();
        let source = topology.project(ONE).unwrap();
        let family =
            ScheduledFamily::new(&mut topology, 3, ModuleCertificationLimits::DEFAULT).unwrap();
        assert!(!family.reference.spatial_ports.is_empty());
        let mut cache = crate::FxMap::default();
        for &position in family.reference.spatial_ports.keys() {
            let local =
                geometry(&mut cache, &source, position, &source, &family.reference).unwrap();
            assert!(Arc::ptr_eq(&local.graph, &family.reference.graph));
            assert!(local.positions.len() <= 7);
            assert!(
                local
                    .spatial_ports
                    .keys()
                    .all(|position| local.positions.contains(position))
            );
            assert_eq!(
                local.spatial_ports[&position],
                family.reference.spatial_ports[&position]
            );
            let repeated =
                geometry(&mut cache, &source, position, &source, &family.reference).unwrap();
            assert!(Arc::ptr_eq(&local, &repeated));
        }
    }
}
