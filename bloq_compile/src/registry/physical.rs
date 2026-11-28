//! Compile local realizations on a common component schedule.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use bloq_graph::{Block, BlockGraph, BlockKind, GuardedTopology};
use bloq_ir::Bloq;
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId,
};
use glam::IVec3;
use petgraph::unionfind::UnionFind;

use crate::block::{LoweringTemplateId, LoweringTemplatePool};
use crate::compile::{CompileContext, CompiledTemplateMap};
use crate::lower::{LinkedPlanInput, LowerPlan, PhysicalInput, PlacedPlan, SpatialPipeRef};
use crate::port::PortExpansionMap;
use crate::signature::{LayerSchedule, LayerScheduleMap};
use crate::{BlockLayout, CompileError, check_resource};

#[path = "anchors.rs"]
mod anchors;
pub(super) use anchors::GuardedDynamicAnchors;

/// Port-expanded geometry shared by scheduling, template selection, and local
/// physical gateways. It has no selector or readout semantics.
pub(super) struct PhysicalGeometry {
    pub graph: Arc<BlockGraph>,
    pub ports: PortExpansionMap,
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
            let (graph, ports) = if Arc::ptr_eq(source, reference_source) {
                // The source Arc can serve many local cases, including Ports.
                // Expand it once globally, never clone it for each local Port.
                let ports = positions
                    .iter()
                    .filter_map(|position| {
                        reference.ports.get(position).map(|port| (*position, *port))
                    })
                    .collect();
                (Arc::clone(&reference.graph), ports)
            } else {
                let (graph, ports) =
                    crate::port::expand_port_context(source, &positions, &reference.ports)?;
                let graph = match graph {
                    std::borrow::Cow::Borrowed(_) => Arc::clone(source),
                    std::borrow::Cow::Owned(graph) => Arc::new(graph),
                };
                (graph, ports)
            };
            entry.insert(Arc::new(PhysicalGeometry {
                graph,
                ports,
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
    let (graph, mut ports) = crate::port::expand_port_topology(&source, distance)?;
    crate::port::allocate_private_port_qubits(&topology.source, distance, &mut ports)?;
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
            ports,
            positions,
        }),
    ))
}

impl ScheduledFamily {
    pub(super) fn new(
        topology: &mut GuardedTopology,
        config: crate::CompileConfig,
    ) -> Result<Self, CompileError> {
        let distance = config.code_distance();
        let limits = config.certification_limits();
        let anchors = GuardedDynamicAnchors::new(topology, limits)?;
        let (source, reference) = reference_geometry(topology, distance)?;
        let mut expanded = crate::FxMap::default();
        let envelope = ScheduleEnvelope::new(
            topology,
            limits.max_witness_nodes,
            &mut expanded,
            &source,
            &reference,
            config.prepare_t_with_mpps(),
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
        // A wall changes the clock of every cube in its spatial component. Resolve
        // its guarded geometry once for all member sites, including the two ends
        // that independently register the same pipe in the physical assembly.
        let mut joint_sites = crate::FxSet::default();
        let mut variants = Vec::new();
        for group in &envelope.wall_groups {
            if group.iter().all(|p| {
                topology.sites[&p.to_array()]
                    .iter()
                    .all(|v| Arc::ptr_eq(&v.graph, &source))
            }) {
                continue;
            }
            joint_sites.extend(group.iter().copied());
            let roots = group
                .iter()
                .flat_map(|p| topology.sites[&p.to_array()].iter().map(|v| v.guard))
                .filter(|&r| r != ONE)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            for (values, guard) in topology.cases(&roots, ONE)? {
                let selected = group
                    .iter()
                    .filter_map(|&p| {
                        topology.sites[&p.to_array()]
                            .iter()
                            .find(|v| {
                                v.guard == ONE
                                    || values
                                        [roots.binary_search(&v.guard).expect("component guard")]
                            })
                            .map(|v| (p, &v.graph))
                    })
                    .collect::<Vec<_>>();
                if !selected.is_empty() {
                    let geometry = wall_geometry(&selected, &source, &reference)?;
                    variants.push((selected.iter().map(|&(p, _)| p).collect(), guard, geometry));
                }
            }
        }
        for (&position, arms) in &topology.sites {
            let position = IVec3::from_array(position);
            if joint_sites.contains(&position) {
                continue;
            }
            for arm in arms {
                let geometry = geometry(&mut expanded, &arm.graph, position, &source, &reference)?;
                variants.push((vec![position], arm.guard, geometry));
            }
        }
        for (positions, guard, geometry) in variants {
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
                sites.extend(positions.iter().map(|&p| (p, guard, case)));
            }
        }
        sites.sort_by_key(|(p, _, _)| p.to_array());
        Ok(Self {
            cases,
            sites,
            t_sides: envelope.t_sides,
            reference,
            anchors,
        })
    }
}

/// Join the complete incidence of each selected site and its immediate neighbors.
fn wall_geometry(
    selected: &[(IVec3, &Arc<BlockGraph>)],
    source: &Arc<BlockGraph>,
    reference: &PhysicalGeometry,
) -> Result<Arc<PhysicalGeometry>, CompileError> {
    let mut complete = BTreeMap::new();
    for &(center, local) in selected {
        for p in local.neighbors(center).into_iter().map(Block::pos) {
            complete.entry(p.to_array()).or_insert(local);
        }
    }
    for &(center, local) in selected {
        complete.insert(center.to_array(), local);
    }
    let positions = complete
        .keys()
        .copied()
        .map(IVec3::from_array)
        .collect::<Vec<_>>();
    if selected
        .iter()
        .all(|&(_, local)| Arc::ptr_eq(local, source))
    {
        let ports = positions
            .iter()
            .filter_map(|p| reference.ports.get(p).map(|v| (*p, *v)))
            .collect();
        return Ok(Arc::new(PhysicalGeometry {
            graph: Arc::clone(&reference.graph),
            positions,
            ports,
        }));
    }
    let mut blocks = BTreeMap::new();
    let mut pipes = BTreeMap::new();
    for (&pos, local) in &complete {
        for pipe in local.pipes_at(IVec3::from_array(pos)) {
            pipes.insert((pipe.src().to_array(), pipe.dst().to_array()), pipe);
            for q in [pipe.src(), pipe.dst()] {
                let block = local.get_endpoint_block(q).expect("incident endpoint");
                blocks.entry(block.pos().to_array()).or_insert(block);
            }
        }
    }
    for (pos, local) in complete {
        blocks.insert(
            pos,
            local
                .get_block(IVec3::from_array(pos))
                .expect("selected block"),
        );
    }
    let mut graph = BlockGraph::new();
    for block in blocks.into_values() {
        graph.try_add_block(block.clone())?;
    }
    for pipe in pipes.into_values() {
        graph.try_add_pipe(pipe.clone())?;
    }
    let (expanded, ports) = crate::port::expand_port_context(&graph, &positions, &reference.ports)?;
    let graph = match expanded {
        std::borrow::Cow::Borrowed(_) => graph,
        std::borrow::Cow::Owned(expanded) => expanded,
    };
    Ok(Arc::new(PhysicalGeometry {
        graph: Arc::new(graph),
        ports,
        positions,
    }))
}

fn same_local_geometry(reference: &PhysicalGeometry, local: &PhysicalGeometry) -> bool {
    local.positions.iter().all(|&position| {
        local.graph.get_block(position) == reference.graph.get_block(position)
            && local.ports.get(&position) == reference.ports.get(&position)
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
    pub port_templates: CompiledTemplateMap,
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
            ports,
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
            for (source, dir) in ports
                .values()
                .filter_map(|port| port.cube_pipe_dir().map(|dir| (port.source, dir)))
            {
                let signature = signatures.get_mut(&source).expect("expanded Port cube");
                signature.connectivity = signature.connectivity.with_pipe(dir);
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
                ports,
                definitions,
                &relocations,
                schedules,
                t_sides,
            )?
        };
        let input = LinkedPlanInput::from_module_graph(graph, ports, positions.iter().copied());
        let port_templates = context.compile_port_templates(ports, &mut compiled)?;
        let temporal_templates =
            context.compile_temporal_template_plans(input.temporal_templates())?;
        let wall_sites = schedules
            .iter()
            .filter_map(|(&pos, &schedule)| (schedule == LayerSchedule::Wall).then_some(pos));
        let (wall_templates, pool) = context.spatial_wall_pool(
            graph,
            ports,
            input.spatial_walls(),
            wall_sites,
            &mut compiled,
        )?;
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
            port_templates: &port_templates,
            ports,
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
                port_templates,
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
    schedule: LayerSchedule,
    incoming: DecisionId,
}

/// All realizations of a potential spatial component use a common depth.
/// Tall cubes propagate the guarded presence of incoming temporal cuts.
struct ScheduleEnvelope {
    choices: crate::FxMap<IVec3, ScheduleChoice>,
    wall_groups: Vec<Vec<IVec3>>,
    pub t_sides: crate::FxMap<IVec3, bloq_graph::Direction>,
}

struct ScheduleIncidence {
    positions: Vec<IVec3>,
    indices: crate::FxMap<IVec3, usize>,
    groups: UnionFind<usize>,
    edges: BTreeMap<(usize, usize), DecisionId>,
    demands: Vec<[DecisionId; 3]>,
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
            demands: vec![[ZERO; 3]; count],
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
            .ports
            .get(&position)
            .is_some_and(|port| port.is_spatial() && port.has_future_pipe())
            || graph.pipes_at(position).any(|pipe| {
                !pipe.dir().is_spatial()
                    && [pipe.src(), pipe.dst()].into_iter().any(|endpoint| {
                        graph
                            .get_endpoint_block(endpoint)
                            .is_some_and(|other| other.pos().z < position.z)
                    })
            })
        {
            self.demands[index][2] = diagram.apply(BooleanOp::Or, self.demands[index][2], guard)?;
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
                self.demands[index][1] =
                    diagram.apply(BooleanOp::Or, self.demands[index][1], guard)?;
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
        prepare_t_with_mpps: bool,
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
        Self::from_incidence(topology, incidence, limit, prepare_t_with_mpps)
    }

    fn from_incidence(
        topology: &mut GuardedTopology,
        incidence: ScheduleIncidence,
        limit: usize,
        prepare_t_with_mpps: bool,
    ) -> Result<Self, CompileError> {
        let ScheduleIncidence {
            positions,
            indices: _,
            groups,
            edges,
            mut demands,
            tall,
        } = incidence;
        let mut possible = vec![[false; 2]; positions.len()];
        let mut tall_components = vec![false; positions.len()];
        for (index, demand) in demands.iter().enumerate() {
            let group = groups.find(index);
            tall_components[group] |= tall[index];
            for axis in 0..2 {
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
                tall_components[group]
            })
            .collect::<VecDeque<_>>();
        let mut queued = vec![false; positions.len()];
        for &index in &pending {
            queued[index] = true;
        }
        while let Some(source) = pending.pop_front() {
            queued[source] = false;
            for &(target, edge) in &neighbors[source] {
                let contribution =
                    topology
                        .diagram
                        .apply(BooleanOp::And, edge, demands[source][2])?;
                let next =
                    topology
                        .diagram
                        .apply(BooleanOp::Or, demands[target][2], contribution)?;
                let changed = next != demands[target][2];
                demands[target][2] = next;
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
        let mut wall_groups = BTreeMap::<usize, Vec<IVec3>>::new();
        for (index, &position) in positions.iter().enumerate() {
            let [padded, wall] = possible[groups.find(index)];
            let schedule = if wall {
                wall_groups
                    .entry(groups.find(index))
                    .or_default()
                    .push(position);
                LayerSchedule::Wall
            } else if padded {
                LayerSchedule::Padded
            } else {
                LayerSchedule::Compact
            };
            choices.insert(
                position,
                ScheduleChoice {
                    schedule,
                    incoming: demands[index][2],
                },
            );
        }
        // Cultivation reserves one spill cell valid in both source arms.
        let t_sides = crate::signature::t_surgery_sides(&topology.source, prepare_t_with_mpps)?;
        Ok(Self {
            choices,
            t_sides,
            wall_groups: wall_groups.into_values().collect(),
        })
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
                (block.kind().is_cube() && block.height_cells() > 1).then_some(choice.incoming)
            })
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
            schedules.insert(position, choice.schedule);
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
    use bloq_graph::ModuleCertificationLimits;

    use super::*;

    #[test]
    fn independent_branches_keep_only_one_full_physical_context() {
        let graph = bloq_test::benchmark::structural_branch_workload(40);
        let mut topology =
            GuardedTopology::new(&graph, bloq_graph::ModuleCertificationLimits::DEFAULT).unwrap();
        let family = ScheduledFamily::new(&mut topology, crate::CompileConfig::new(3)).unwrap();
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
        let family = ScheduledFamily::new(&mut topology, crate::CompileConfig::new(3)).unwrap();
        assert!(!family.reference.ports.is_empty());
        let mut cache = crate::FxMap::default();
        for &position in family.reference.ports.keys() {
            let local =
                geometry(&mut cache, &source, position, &source, &family.reference).unwrap();
            assert!(Arc::ptr_eq(&local.graph, &family.reference.graph));
            assert!(local.positions.len() <= 7);
            assert!(
                local
                    .ports
                    .keys()
                    .all(|position| local.positions.contains(position))
            );
            assert_eq!(local.ports[&position], family.reference.ports[&position]);
            let repeated =
                geometry(&mut cache, &source, position, &source, &family.reference).unwrap();
            assert!(Arc::ptr_eq(&local, &repeated));
        }
    }
}
