//! Intern physical declarations and compose their guarded stage ownership.

use std::collections::BTreeMap;
use std::sync::Arc;

use bloq_graph::{GuardedTopology, GuardedVariable};
use bloq_ir::lowering::{TemplateInstance, TemplateInstanceId};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, ClassicalExpr, ClassicalNode,
    InstanceProvenance, NodeProvenance, PipeSeam, SubGraph, TemplateId,
};
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId,
};
use glam::IVec3;
use petgraph::unionfind::UnionFind;

use super::flow_families::{FamilyCompiler, FlowMember, PlannedBundleUse};
use super::flows::{BoundaryFlow, GuardedCheck, GuardedFlowEngine};
use super::physical::PhysicalProjection;
use super::{InstanceKey, Stage, assembly, merge_provenance, remap_edge, remap_node};
use crate::lower::ChunkSiteSource;
use crate::{CompileError, FxMap as HashMap, add_resource, check_resource};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Reference {
    Quantum(usize),
    Region(usize),
}

pub(super) struct Member {
    pub instance: TemplateInstance,
    pub guard: DecisionId,
    nested: bool,
}

struct Region {
    position: IVec3,
    node: BloqNode,
    fingerprint: Vec<u8>,
    guard: DecisionId,
    members: Vec<usize>,
    source_instance: Option<TemplateInstanceId>,
}

pub(super) struct ImportedProjection {
    pub physical: Arc<PhysicalProjection>,
    pub sites: HashMap<ChunkSiteSource, TemplateInstanceId>,
    pub selectives: HashMap<IVec3, crate::lower::SelectiveArmInstances>,
}

pub(super) struct RegistrationIndex {
    projection: Arc<ImportedProjection>,
    seams: Vec<(Reference, Reference, PipeSeam)>,
    positions: HashMap<IVec3, Registration>,
}

#[derive(Default)]
struct Registration {
    members: Vec<TemplateInstanceId>,
    seams: Vec<usize>,
}

pub(super) struct Profile {
    pub position: IVec3,
    pub guard: DecisionId,
    pub projection: Arc<ImportedProjection>,
}

pub(super) struct Assembly {
    pub bloq: Bloq,
    /// Only identities retained by `bloq.templates()` enter this cache, so an
    /// imported-but-deduplicated Arc cannot leave a recyclable pointer behind.
    templates_by_identity: HashMap<usize, TemplateId>,
    pub members: Vec<Member>,
    instances: HashMap<InstanceKey, usize>,
    groups: UnionFind<usize>,
    stages: BTreeMap<usize, Stage>,
    regions: Vec<Region>,
    region_by_position: HashMap<IVec3, usize>,
    region_by_node: HashMap<BloqNodeId, usize>,
    region_observables: HashMap<(String, u32), u32>,
    edges: HashMap<(Reference, Reference, BloqEdge), DecisionId>,
    quantum_nodes: BTreeMap<usize, BloqNodeId>,
    region_nodes: HashMap<IVec3, BloqNodeId>,
    pub owners: HashMap<TemplateInstanceId, BloqNodeId>,
    pub next_observable: u32,
    // Guards remain roots until choreography consumes these owned sites.
    walking: Vec<(IVec3, IVec3, TemplateInstanceId, DecisionId)>,
}

impl Assembly {
    pub(super) fn allocate_observable(&mut self) -> Result<u32, CompileError> {
        let index = self.next_observable;
        self.next_observable =
            add_resource("observable IDs", index as usize, 1, u32::MAX as usize)? as u32;
        Ok(index)
    }

    pub(super) fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        self.members
            .iter_mut()
            .map(|member| &mut member.guard)
            .chain(self.regions.iter_mut().map(|region| &mut region.guard))
            .chain(self.edges.values_mut())
            .chain(self.walking.iter_mut().map(|walk| &mut walk.3))
    }

    pub(super) fn new() -> Self {
        Self {
            bloq: Bloq::new(),
            templates_by_identity: HashMap::default(),
            members: Vec::new(),
            instances: HashMap::default(),
            groups: UnionFind::new(0),
            stages: BTreeMap::new(),
            regions: Vec::new(),
            region_by_position: HashMap::default(),
            region_by_node: HashMap::default(),
            region_observables: HashMap::default(),
            edges: HashMap::default(),
            quantum_nodes: BTreeMap::new(),
            region_nodes: HashMap::default(),
            owners: HashMap::default(),
            next_observable: 0,
            walking: Vec::new(),
        }
    }

    pub(super) fn register(
        &mut self,
        index: &RegistrationIndex,
        position: IVec3,
        guard: DecisionId,
        selector: Option<DecisionId>,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Profile, CompileError> {
        if let Some(arms) = index.projection.selectives.get(&position) {
            let selector = selector.expect("selective sites have a resolve selector");
            for (instance, selected) in [
                (arms.when_false, diagram.negate(selector)?),
                (arms.when_true, selector),
            ] {
                let active = diagram.apply(BooleanOp::And, guard, selected)?;
                let member = &mut self.members[instance.0 as usize];
                member.guard = diagram.apply(BooleanOp::Or, member.guard, active)?;
            }
        }
        let registration = index.positions.get(&position);
        if let Some(registration) = registration {
            for &instance in &registration.members {
                let member = &mut self.members[instance.0 as usize];
                member.guard = diagram.apply(BooleanOp::Or, member.guard, guard)?;
            }
        }
        if let Some(&region) = self.region_by_position.get(&position) {
            let region = &mut self.regions[region];
            // Flow composition used the first profile at this position.
            region
                .source_instance
                .get_or_insert_with(|| index.projection.sites[&ChunkSiteSource::Block(position)]);
            region.guard = diagram.apply(BooleanOp::Or, region.guard, guard)?;
            for &member in &region.members {
                self.members[member].guard =
                    diagram.apply(BooleanOp::Or, self.members[member].guard, guard)?;
            }
        }
        if let Some(registration) = registration {
            for &seam in &registration.seams {
                let (source, target, pipe) = &index.seams[seam];
                let edge = BloqEdge::Quantum(Box::new(bloq_ir::QuantumEdge {
                    pipes: vec![pipe.clone()],
                    guard: None,
                }));
                let value = self.edges.entry((*source, *target, edge)).or_insert(ZERO);
                *value = diagram.apply(BooleanOp::Or, *value, guard)?;
            }
        }
        if let Some(block) = index.projection.physical.geometry.graph.get_block(position)
            && let bloq_graph::BlockKind::Walking(kind) = block.kind()
            && let Some(&instance) = index
                .projection
                .sites
                .get(&ChunkSiteSource::Block(position))
        {
            self.walking.push((
                position,
                kind.try_end_position(position)? - IVec3::Z,
                instance,
                guard,
            ));
        }
        Ok(Profile {
            position,
            guard,
            projection: Arc::clone(&index.projection),
        })
    }

    pub(super) fn import(
        &mut self,
        physical: PhysicalProjection,
        program: Bloq,
    ) -> Result<RegistrationIndex, CompileError> {
        let physical = Arc::new(physical);
        let mut templates = HashMap::default();
        for (old, template) in program.templates().iter_shared() {
            let identity = Arc::as_ptr(template) as usize;
            let id = self
                .templates_by_identity
                .get(&identity)
                .copied()
                .or_else(|| {
                    self.bloq
                        .templates()
                        .iter_shared()
                        .find(|(_, other)| Arc::ptr_eq(other, template) || *other == template)
                        .map(|(id, _)| id)
                })
                .unwrap_or_else(|| {
                    let id = self.bloq.add_shared_template(Arc::clone(template));
                    self.templates_by_identity.insert(identity, id);
                    id
                });
            templates.insert(old, id);
        }
        let mut instances = HashMap::default();
        self.intern_members(
            program.top(),
            "",
            false,
            &templates,
            &mut instances,
            &mut crate::FxSet::default(),
        )?;
        let region_positions = physical
            .placements
            .regions
            .iter()
            .map(|(&position, &id)| (id, position))
            .collect::<HashMap<_, _>>();
        let mut nodes = HashMap::default();
        for id in program
            .top()
            .deterministic_emit_order()
            .map_err(|_| assembly("physical projection is cyclic"))?
        {
            let node = &program[id];
            let reference = if let Some(quantum) = node.try_quantum() {
                let members = quantum
                    .instances
                    .iter()
                    .map(|instance| instances[&instance.id].0 as usize)
                    .collect::<Vec<_>>();
                let first = *members
                    .first()
                    .ok_or_else(|| assembly("empty physical assembly stage"))?;
                for &member in &members[1..] {
                    self.groups.union(first, member);
                }
                if let Some(stage) = self.stages.get_mut(&first) {
                    append_provenance(&mut stage.provenance, &node.provenance)?;
                    if stage.timeline != quantum.timeline {
                        return Err(assembly("selected timelines have no common stage boundary"));
                    }
                } else {
                    self.stages.insert(
                        first,
                        Stage {
                            member: first,
                            provenance: node.provenance.clone(),
                            timeline: quantum.timeline.clone(),
                        },
                    );
                }
                Reference::Quantum(first)
            } else {
                let position = region_positions[&id];
                let mut candidate = node.clone();
                remap_node(&mut candidate, &templates, &instances);
                self.remap_observables(&mut candidate, &format!("{position:?}"))?;
                let mut fingerprint = Bloq::new();
                fingerprint.add_node(candidate.clone());
                let fingerprint = fingerprint.to_binary();
                let region = if let Some(&region) = self.region_by_position.get(&position) {
                    if self.regions[region].fingerprint != fingerprint {
                        return Err(assembly(format!(
                            "dynamic region at {position} changes its physical realization"
                        )));
                    }
                    region
                } else {
                    let mut members = Vec::new();
                    collect_members(&candidate, &mut members);
                    let region = self.regions.len();
                    self.regions.push(Region {
                        position,
                        node: candidate,
                        fingerprint,
                        guard: ZERO,
                        members,
                        source_instance: None,
                    });
                    self.region_by_position.insert(position, region);
                    region
                };
                Reference::Region(region)
            };
            nodes.insert(id, reference);
        }
        let sites: HashMap<_, _> = physical
            .placements
            .instances
            .iter()
            .map(|(&site, instance)| (site, instances[instance]))
            .collect();
        // Each local profile reuses this projection. Resolve incidence once,
        // preserving member and seam order for subsequent guard updates.
        let mut registration: HashMap<_, Registration> = HashMap::default();
        for (&site, &instance) in &sites {
            for position in incident_positions(site, &physical.geometry.graph) {
                registration
                    .entry(position)
                    .or_default()
                    .members
                    .push(instance);
            }
        }
        let mut seams = Vec::new();
        let instance_owners = program
            .top()
            .quantum_nodes()
            .flat_map(|(node, quantum)| {
                quantum
                    .instances
                    .iter()
                    .map(move |instance| (instance.id, node))
            })
            .collect::<HashMap<_, _>>();
        let mut local_walking_orders = crate::FxSet::default();
        for (&site, instance) in &physical.placements.instances {
            let ChunkSiteSource::Block(position) = site else {
                continue;
            };
            let Some(block) = physical.geometry.graph.get_block(position) else {
                continue;
            };
            let bloq_graph::BlockKind::Walking(kind) = block.kind() else {
                continue;
            };
            let vacated = kind.try_end_position(position)? - IVec3::Z;
            if !physical
                .geometry
                .graph
                .get_block(vacated)
                .is_some_and(|block| matches!(block.kind(), bloq_graph::BlockKind::Walking(_)))
            {
                continue;
            }
            if let Some(vacating) = physical
                .placements
                .instances
                .get(&ChunkSiteSource::Block(vacated))
                && let (Some(&vacating), Some(&entering)) =
                    (instance_owners.get(vacating), instance_owners.get(instance))
                && entering != vacating
            {
                local_walking_orders.insert((vacating, entering));
            }
        }
        for edge in program.edges() {
            let source = nodes[&edge.source];
            let target = nodes[&edge.target];
            let mut kind = edge.edge.clone();
            remap_edge(&mut kind, &templates);
            match kind {
                BloqEdge::Order => {
                    // Context neighbors are not registrations. Rebuild walking
                    // choreography from owned profiles after global assembly;
                    // retain a coincident record dependency independently.
                    if local_walking_orders.contains(&(edge.source, edge.target))
                        && !program[edge.target].try_quantum().is_some_and(|quantum| {
                            quantum
                                .detectors
                                .iter()
                                .map(|row| &row.parity)
                                .chain(quantum.restarts.iter().map(|row| &row.parity))
                                .flat_map(bloq_circuit::DetectorParity::measurements)
                                .any(|term| {
                                    instance_owners.get(&term.instance) == Some(&edge.source)
                                })
                        })
                    {
                        continue;
                    }
                    self.edges.insert((source, target, BloqEdge::Order), ONE);
                }
                BloqEdge::Quantum(quantum) => {
                    for pipe in quantum.pipes {
                        for position in incident_positions(
                            ChunkSiteSource::TemporalPipe(pipe.pipe),
                            &physical.geometry.graph,
                        ) {
                            registration
                                .entry(position)
                                .or_default()
                                .seams
                                .push(seams.len());
                        }
                        seams.push((source, target, pipe));
                    }
                }
                BloqEdge::Value { .. } | BloqEdge::Compose { .. } => {}
            }
        }
        let selectives = physical
            .placements
            .selectives
            .iter()
            .map(|(&position, arms)| {
                (
                    position,
                    crate::lower::SelectiveArmInstances {
                        when_false: instances[&arms.when_false],
                        when_true: instances[&arms.when_true],
                    },
                )
            })
            .collect();
        Ok(RegistrationIndex {
            projection: Arc::new(ImportedProjection {
                physical,
                sites,
                selectives,
            }),
            seams,
            positions: registration,
        })
    }

    fn intern_members(
        &mut self,
        level: &SubGraph,
        scope: &str,
        nested: bool,
        templates: &HashMap<TemplateId, TemplateId>,
        instances: &mut HashMap<TemplateInstanceId, TemplateInstanceId>,
        seen: &mut crate::FxSet<usize>,
    ) -> Result<(), CompileError> {
        let keys = level
            .stable_keys()
            .map_err(|_| assembly("physical level is cyclic"))?
            .into_iter()
            .map(|(key, id)| (id, key))
            .collect::<HashMap<_, _>>();
        for (id, node) in level.nodes() {
            let lexical = keys.get(&id).map_or_else(
                || {
                    format!(
                        "{scope}/{}",
                        super::physical_region_key(node).unwrap_or_else(|| format!("n{}", id.0))
                    )
                },
                |key| format!("{scope}/{key}"),
            );
            if let Some(quantum) = node.try_quantum() {
                for (ordinal, instance) in quantum.instances.iter().enumerate() {
                    let key = InstanceKey {
                        template: templates[&instance.template_id],
                        offset: instance.offset,
                        provenance: instance.provenance,
                        lexical: matches!(instance.provenance, InstanceProvenance::Source)
                            .then(|| format!("{lexical}/{ordinal}")),
                    };
                    let member = if let Some(&member) = self.instances.get(&key) {
                        member
                    } else {
                        check_resource(
                            "physical template instances",
                            self.members.len() + 1,
                            u32::MAX as usize,
                        )?;
                        let member = self.groups.new_set();
                        let mut placed = *instance;
                        placed.id = TemplateInstanceId(member as u32);
                        placed.template_id = key.template;
                        self.members.push(Member {
                            instance: placed,
                            guard: ZERO,
                            nested,
                        });
                        self.instances.insert(key, member);
                        member
                    };
                    if !seen.insert(member) {
                        return Err(assembly(
                            "one physical member is owned twice in a projection",
                        ));
                    }
                    instances.insert(instance.id, TemplateInstanceId(member as u32));
                }
            }
            if let Some(region) = node.try_region() {
                for (body, graph) in region.bodies() {
                    self.intern_members(
                        graph,
                        &format!("{lexical}/{body:?}"),
                        true,
                        templates,
                        instances,
                        seen,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn remap_observables(&mut self, node: &mut BloqNode, scope: &str) -> Result<(), CompileError> {
        let BloqNodeKind::Region(region) = &mut node.kind else {
            return Ok(());
        };
        for (selector, body) in region.bodies_mut() {
            let scope = format!("{scope}/{selector:?}");
            let mut mapping = HashMap::default();
            for (id, node) in body.nodes() {
                if let Some(ClassicalNode::Observable {
                    index: Some(index), ..
                }) = node.try_classical()
                {
                    let old = *index;
                    let key = (scope.clone(), id.0);
                    let index = if let Some(&index) = self.region_observables.get(&key) {
                        index
                    } else {
                        let index = self.allocate_observable()?;
                        self.region_observables.insert(key, index);
                        index
                    };
                    mapping.insert(old, index);
                }
            }
            for id in body.node_ids().collect::<Vec<_>>() {
                let node = body
                    .node_mut(id)
                    .expect("collected region node still exists during observable remap");
                if let BloqNodeKind::Classical(data) = &mut node.kind
                    && matches!(
                        data.as_ref(),
                        ClassicalNode::Observable { index: Some(_), .. }
                    )
                {
                    let ClassicalNode::Observable {
                        index: Some(index), ..
                    } = Arc::make_mut(data)
                    else {
                        unreachable!("only observable references are remapped")
                    };
                    *index = mapping[index];
                }
                if let NodeProvenance::Generator { ordinal } = &mut node.provenance
                    && let Some(&index) = mapping.get(ordinal)
                {
                    *ordinal = index;
                }
                self.remap_observables(node, &format!("{scope}/{}", id.0))?;
            }
        }
        Ok(())
    }

    pub(super) fn build(
        &mut self,
        topology: &mut GuardedTopology,
    ) -> Result<BooleanProgram, CompileError> {
        for (index, member) in self.members.iter_mut().enumerate() {
            member.guard = topology.diagram.constrain(member.guard, topology.domain)?;
            if member.nested || member.guard == ZERO {
                continue;
            }
            let group = self.groups.find(index);
            let node = *self
                .quantum_nodes
                .entry(group)
                .or_insert_with(|| self.bloq.add_node(BloqNode::from_members(Vec::new())));
            self.bloq
                .node_mut(node)
                .expect("newly added quantum assembly node exists")
                .expect_quantum_mut()
                .instances
                .push(member.instance);
            self.owners.insert(member.instance.id, node);
        }
        for stage in self.stages.values() {
            let Some(&id) = self.quantum_nodes.get(&self.groups.find(stage.member)) else {
                continue;
            };
            let node = self
                .bloq
                .node_mut(id)
                .expect("registered quantum assembly node exists");
            append_provenance(&mut node.provenance, &stage.provenance)?;
            let timeline = &mut node.expect_quantum_mut().timeline;
            if timeline.is_some() && *timeline != stage.timeline {
                return Err(assembly("selected timelines have no common stage boundary"));
            }
            *timeline = stage.timeline.clone();
        }
        for &node in self.quantum_nodes.values() {
            if let NodeProvenance::BlockComponent { members } = &mut self
                .bloq
                .node_mut(node)
                .expect("registered quantum assembly node exists")
                .provenance
            {
                members.sort_unstable();
                members.dedup();
            }
        }
        for (index, region) in self.regions.iter_mut().enumerate() {
            region.guard = topology.diagram.constrain(region.guard, topology.domain)?;
            if region.guard != ONE {
                return Err(assembly(format!(
                    "dynamic region at {} is not common to every choice",
                    region.position
                )));
            }
            let id = self.bloq.add_node(region.node.clone());
            self.region_nodes.insert(region.position, id);
            self.region_by_node.insert(id, index);
            for &member in &region.members {
                self.owners.insert(self.members[member].instance.id, id);
            }
        }
        let mut boolean = BooleanProgram::new(topology, &mut self.bloq);
        for member in &self.members {
            if member.nested || member.guard == ZERO || member.guard == ONE {
                continue;
            }
            let node = self.owners[&member.instance.id];
            let guard = boolean.emit(member.guard, &topology.diagram, &mut self.bloq);
            super::registration(&mut self.bloq, node, guard)?
                .instances
                .push(member.instance.id);
        }
        let mut edges = HashMap::<(BloqNodeId, BloqNodeId, BloqEdge), DecisionId>::default();
        for ((source, target, edge), guard) in &self.edges {
            let (Some(source), Some(target)) = (self.node(*source), self.node(*target)) else {
                continue;
            };
            if source == target {
                return Err(assembly(
                    "shared membership would merge sequential source stages",
                ));
            }
            let previous = edges.entry((source, target, edge.clone())).or_insert(ZERO);
            *previous = topology.diagram.apply(BooleanOp::Or, *previous, *guard)?;
        }
        let mut edges = edges.into_iter().collect::<Vec<_>>();
        for (_, guard) in &mut edges {
            *guard = topology.diagram.constrain(*guard, topology.domain)?;
        }
        // Stable sort identical to a key of `(from.0, to.0, guard.0, "{edge:?}")`;
        // the debug strings are only formatted to break full ties, since every
        // comparison would otherwise re-render each edge's pipes.
        edges.sort_by(
            |((from, to, edge), guard), ((prev_from, prev_to, prev_edge), prev_guard)| {
                (from.0, to.0, guard.0)
                    .cmp(&(prev_from.0, prev_to.0, prev_guard.0))
                    .then_with(|| format!("{edge:?}").cmp(&format!("{prev_edge:?}")))
            },
        );
        // One selected face seam owns all parallel pipes and their padding.
        // Distinct guards must stay separate until membership is pinned.
        edges.dedup_by(
            |((from, to, edge), guard), ((prev_from, prev_to, prev), prev_guard)| {
                if (from, to, guard) == (prev_from, prev_to, prev_guard)
                    && let (BloqEdge::Quantum(edge), BloqEdge::Quantum(prev)) = (edge, prev)
                {
                    prev.pipes.append(&mut edge.pipes);
                    true
                } else {
                    false
                }
            },
        );
        for ((source, target, mut edge), guard) in edges {
            if guard == ZERO {
                continue;
            }
            if let BloqEdge::Quantum(quantum) = &mut edge
                && guard != ONE
            {
                let guard = boolean.emit(guard, &topology.diagram, &mut self.bloq);
                quantum.guard = Some(guard.into());
                self.bloq.add_edge(guard, target, BloqEdge::Order);
            }
            self.bloq.add_edge(source, target, edge);
        }
        Ok(boolean)
    }

    /// Restore choreography across local projection boundaries using only
    /// registered source sites, not their imported context neighbors.
    pub(super) fn order_walking_choreography(
        &mut self,
        topology: &mut GuardedTopology,
    ) -> Result<(), CompileError> {
        topology.diagram.charge(self.walking.len())?;
        let mut walks = Vec::new();
        for (position, destination, instance, guard) in std::mem::take(&mut self.walking) {
            let Some(&owner) = self.owners.get(&instance) else {
                continue;
            };
            walks.push((position, destination, owner, guard));
        }
        if walks.is_empty() {
            return Ok(());
        }
        add_walking_order_edges(self.bloq.top_mut(), &walks, topology)
    }

    fn node(&self, reference: Reference) -> Option<BloqNodeId> {
        match reference {
            Reference::Quantum(member) => {
                self.quantum_nodes.get(&self.groups.find(member)).copied()
            }
            Reference::Region(region) => self
                .region_nodes
                .get(&self.regions[region].position)
                .copied(),
        }
    }

    pub(super) fn compose_flows(
        &mut self,
        topology: &mut GuardedTopology,
        boolean: &mut BooleanProgram,
        limit: usize,
        distance: u32,
    ) -> Result<(), CompileError> {
        let order = self
            .bloq
            .top()
            .deterministic_emit_order()
            .map_err(|_| assembly("physical assembly is cyclic"))?;
        let nodes = order
            .into_iter()
            .filter(|id| {
                self.bloq[*id].try_quantum().is_some() || self.bloq[*id].try_region().is_some()
            })
            .collect::<Vec<_>>();
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(index, &node)| (node, index))
            .collect::<HashMap<_, _>>();
        let mut components = UnionFind::new(nodes.len());
        for edge in self.bloq.edges() {
            if matches!(edge.edge, BloqEdge::Quantum(_)) {
                components.union(indices[&edge.source], indices[&edge.target]);
            }
        }
        let mut members = HashMap::default();
        let mut layers = HashMap::default();
        for &node in &nodes {
            let declared = if let Some(quantum) = self.bloq[node].try_quantum() {
                layers.insert(node, self.bloq[node].layer());
                quantum
                    .instances
                    .iter()
                    .map(|instance| (instance.id, false))
                    .collect::<Vec<_>>()
            } else {
                let region = &self.regions[self.region_by_node[&node]];
                layers.insert(node, 2 * i64::from(region.position.z));
                vec![(
                    region
                        .source_instance
                        .expect("registered region has a source instance"),
                    true,
                )]
            };
            members.insert(
                node,
                declared
                    .into_iter()
                    .map(|(id, skip_unmatched)| {
                        let member = &self.members[id.0 as usize];
                        FlowMember {
                            instance: id,
                            template: member.instance.template_id,
                            offset: member.instance.offset,
                            guard: if skip_unmatched { ONE } else { member.guard },
                            skip_unmatched,
                        }
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let mut families = FamilyCompiler::new(self.bloq.templates(), topology, distance)?;
        // Validate real placements before local recipe normalization, including
        // inactive members. Scan original row order only on a failing bound.
        for &node in &nodes {
            for &member in &members[&node] {
                if !families.coordinates_fit(member) {
                    materialize_flows(&[member], self.bloq.templates(), topology)?;
                }
            }
        }
        let mut bundle_ids = HashMap::default();
        let mut detector_owners = HashMap::default();
        let mut detector_edges = crate::FxSet::default();
        let (mut accepted, mut fallback, mut bundle_uses) = (0, 0, 0);
        let mut groups = BTreeMap::<usize, Vec<BloqNodeId>>::new();
        for (index, &node) in nodes.iter().enumerate() {
            groups.entry(components.find(index)).or_default().push(node);
        }
        for mut group in groups.into_values() {
            group.sort_by_key(|node| (layers[node], indices[node]));
            let mut first = 0;
            while first < group.len() {
                let layer = layers[&group[first]];
                let end = first + group[first..].partition_point(|node| layers[node] == layer);
                if !families.order_same_layer(&mut group[first..end], &members, topology)? {
                    let mut sources = HashMap::default();
                    for &node in &group[first..end] {
                        sources.insert(
                            node,
                            materialize_flows(&members[&node], self.bloq.templates(), topology)?,
                        );
                    }
                    order_same_layer(&mut group[first..end], &sources, topology)?;
                }
                first = end;
            }
            if let Some(plan) =
                families.plan_component(&group, &members, self.bloq.templates(), topology, limit)?
            {
                accepted += 1;
                bundle_uses += plan.bundles.len();
                // Nothing public changes until the complete component succeeds.
                for mut use_ in plan.bundles {
                    use_.node = *detector_owners
                        .entry(use_.node)
                        .or_insert_with(|| self.detector_owner(use_.node));
                    self.attach_bundle(
                        use_,
                        &mut bundle_ids,
                        &mut detector_edges,
                        topology,
                        boolean,
                    )?;
                }
            } else {
                fallback += 1;
                // Unsupported planning retains spent Boolean work and replays
                // only this component from its original descriptors.
                let mut engine = GuardedFlowEngine::default();
                for node in group {
                    let source =
                        materialize_flows(&members[&node], self.bloq.templates(), topology)?;
                    let checks = engine.append(&source, &mut topology.diagram, topology.domain)?;
                    self.attach_checks(node, checks, topology, boolean)?;
                    check_resource(
                        "guarded boundary-flow Boolean nodes",
                        topology.diagram.nodes().len(),
                        limit,
                    )?;
                }
                engine.finish(&mut topology.diagram, topology.domain)?;
            }
        }
        families.report(accepted, fallback, bundle_uses);
        Ok(())
    }

    fn attach_bundle(
        &mut self,
        use_: PlannedBundleUse,
        bundle_ids: &mut HashMap<usize, bloq_ir::DetectorBundleId>,
        detector_edges: &mut crate::FxSet<(BloqNodeId, BloqNodeId)>,
        topology: &mut GuardedTopology,
        boolean: &mut BooleanProgram,
    ) -> Result<(), CompileError> {
        let node = use_.node;
        let key = Arc::as_ptr(&use_.bundle) as usize;
        let bundle = *bundle_ids.entry(key).or_insert_with(|| {
            self.bloq
                .add_shared_detector_bundle(Arc::clone(&use_.bundle))
        });
        for &owner in use_
            .bundle
            .used_owners()
            .expect("compiler recipes have valid owner slots")
        {
            let dependency = self.owners[&use_.instances[owner as usize]];
            if dependency != node && detector_edges.insert((dependency, node)) {
                self.bloq.add_edge(dependency, node, BloqEdge::Order);
            }
        }
        let quantum = self
            .bloq
            .node_mut(node)
            .expect("bundle owner exists")
            .expect_quantum_mut();
        let count = add_resource(
            "detector bundle use IDs",
            quantum.detector_bundles.len(),
            1,
            u32::MAX as usize,
        )?;
        let index = (count - 1) as u32;
        quantum.detector_bundles.push(bloq_ir::DetectorBundleUse {
            bundle,
            instances: use_.instances,
            offset: use_.offset,
        });
        if use_.guard != ONE {
            let guard = boolean.emit(use_.guard, &topology.diagram, &mut self.bloq);
            super::registration(&mut self.bloq, node, guard)?
                .detector_bundles
                .push(index);
        }
        Ok(())
    }

    fn detector_owner(&mut self, owner: BloqNodeId) -> BloqNodeId {
        if self.bloq[owner].try_region().is_some() {
            let after = self
                .bloq
                .outgoing(owner)
                .filter(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
                .map(|edge| edge.target)
                .collect::<Vec<_>>();
            let node = self.bloq.add_node(BloqNode::from_members(Vec::new()));
            self.bloq.add_edge(owner, node, BloqEdge::Order);
            for after in after {
                self.bloq.add_edge(node, after, BloqEdge::Order);
            }
            for &member in &self.regions[self.region_by_node[&owner]].members {
                let parent = self
                    .owners
                    .get_mut(&self.members[member].instance.id)
                    .expect("registered region member has an owner");
                if *parent == owner {
                    *parent = node;
                }
            }
            node
        } else {
            owner
        }
    }

    fn attach_checks(
        &mut self,
        owner: BloqNodeId,
        checks: Vec<GuardedCheck>,
        topology: &mut GuardedTopology,
        boolean: &mut BooleanProgram,
    ) -> Result<(), CompileError> {
        if checks.is_empty() {
            return Ok(());
        }
        let node = self.detector_owner(owner);
        let mut dependencies = checks
            .iter()
            .flat_map(|check| {
                check.parity.measurements().chain(
                    check
                        .contributions
                        .values()
                        .flat_map(bloq_circuit::DetectorParity::measurements),
                )
            })
            .map(|term| self.owners[&term.instance])
            .collect::<Vec<_>>();
        dependencies.sort_unstable();
        dependencies.dedup();
        for dependency in dependencies {
            if dependency != node {
                self.bloq.add_edge(dependency, node, BloqEdge::Order);
            }
        }
        for check in checks {
            if check.restart {
                return Err(assembly(
                    "a top-level boundary flow carries a restart syndrome",
                ));
            }
            let quantum = self
                .bloq
                .node_mut(node)
                .expect("detector owner exists in the assembled program")
                .expect_quantum_mut();
            let count = add_resource(
                "inline detector IDs",
                quantum.detectors.len(),
                1,
                u32::MAX as usize,
            )?;
            let index = (count - 1) as u32;
            quantum.detectors.push(bloq_ir::NodeDetector {
                parity: check.parity,
                coords: check
                    .center
                    .map(|center| center.into_iter().map(f64::from).collect()),
            });
            if check.guard != ONE {
                let guard = boolean.emit(check.guard, &topology.diagram, &mut self.bloq);
                super::registration(&mut self.bloq, node, guard)?
                    .detectors
                    .push(index);
            }
            for (guard, parity) in check.contributions {
                let producer = boolean.emit(guard, &topology.diagram, &mut self.bloq);
                super::registration(&mut self.bloq, node, producer)?
                    .detector_parities
                    .push((index, parity));
            }
        }
        Ok(())
    }
}

pub(super) fn materialize_flows(
    members: &[FlowMember],
    templates: &bloq_ir::lowering::BloqTemplatePool,
    topology: &mut GuardedTopology,
) -> Result<Vec<BoundaryFlow>, CompileError> {
    let count = members.iter().fold(0usize, |count, member| {
        count.saturating_add(templates[member.template].boundary_flows.len())
    });
    topology.diagram.charge(count)?;
    let work = members
        .iter()
        .flat_map(|member| &templates[member.template].boundary_flows)
        .fold(0usize, |sum, flow| {
            sum.saturating_add(flow.start.len())
                .saturating_add(flow.end.len())
                .saturating_add(flow.measurements.len())
                .saturating_add(1)
        });
    topology.diagram.charge(work)?;
    let mut flows = Vec::with_capacity(count);
    for &member in members {
        for flow in &templates[member.template].boundary_flows {
            flows.push(BoundaryFlow {
                guard: member.guard,
                start: flow.start.try_translated(member.offset)?,
                end: flow.end.try_translated(member.offset)?,
                measurements: flow
                    .measurements
                    .iter()
                    .map(|&measurement| bloq_ir::lowering::InstanceMeasurement {
                        instance: member.instance,
                        measurement,
                    })
                    .collect(),
                sign: flow.sign,
                center: flow
                    .center
                    .map(|center| {
                        bloq_circuit::checked_translate_coordinate(center, member.offset)
                            .map(|center| center.to_array())
                    })
                    .transpose()?,
                marker: flow.marker,
                skip_unmatched: member.skip_unmatched,
            });
        }
    }
    Ok(flows)
}

// Component provenance is consumed only by build(). Append while importing,
// then sort/deduplicate once per final stage rather than after every local view.
fn append_provenance(
    target: &mut NodeProvenance,
    added: &NodeProvenance,
) -> Result<(), CompileError> {
    if let (
        NodeProvenance::BlockComponent { members },
        NodeProvenance::BlockComponent { members: added },
    ) = (&mut *target, added)
    {
        members.extend(added);
        Ok(())
    } else {
        merge_provenance(target, added)
    }
}

fn add_walking_order_edges(
    body: &mut SubGraph,
    walks: &[(IVec3, IVec3, BloqNodeId, DecisionId)],
    topology: &mut GuardedTopology,
) -> Result<(), CompileError> {
    if walks.is_empty() {
        return Ok(());
    }
    topology
        .diagram
        .charge(walks.len().saturating_add(body.edge_count()))?;
    let mut sites = HashMap::<IVec3, HashMap<(BloqNodeId, IVec3), DecisionId>>::default();
    for &(position, destination, owner, guard) in walks {
        let active = sites
            .entry(position)
            .or_default()
            .entry((owner, destination))
            .or_insert(ZERO);
        *active = topology.diagram.apply(BooleanOp::Or, *active, guard)?;
    }
    let mut existing = body
        .edges()
        .filter(|edge| matches!(edge.edge, BloqEdge::Order))
        .map(|edge| (edge.source, edge.target))
        .collect::<crate::FxSet<_>>();
    let mut orders = Vec::new();
    for variants in sites.values() {
        for (&(entering, destination), &guard) in variants {
            let Some(vacating) = sites.get(&destination) else {
                continue;
            };
            for (&(vacating, _), &other) in vacating {
                topology.diagram.charge(1)?;
                if entering == vacating || existing.contains(&(vacating, entering)) {
                    continue;
                }
                let together = topology.diagram.apply(BooleanOp::And, guard, other)?;
                if topology
                    .diagram
                    .apply(BooleanOp::And, together, topology.domain)?
                    != ZERO
                {
                    existing.insert((vacating, entering));
                    orders.push((vacating, entering));
                }
            }
        }
    }
    orders.sort_unstable();
    for (vacating, entering) in orders {
        body.add_edge(vacating, entering, BloqEdge::Order);
    }
    Ok(())
}

fn order_same_layer(
    nodes: &mut [BloqNodeId],
    sources: &HashMap<BloqNodeId, Vec<BoundaryFlow>>,
    topology: &mut GuardedTopology,
) -> Result<(), CompileError> {
    if nodes.len() < 2 {
        return Ok(());
    }
    let mut consumers = HashMap::<&bloq_circuit::PauliMap, Vec<(usize, DecisionId)>>::default();
    for (index, node) in nodes.iter().enumerate() {
        for flow in sources[node].iter().filter(|flow| !flow.start.is_empty()) {
            consumers
                .entry(&flow.start)
                .or_default()
                .push((index, flow.guard));
        }
    }
    let mut successors = vec![crate::FxSet::default(); nodes.len()];
    let mut indegrees = vec![0; nodes.len()];
    for (producer, node) in nodes.iter().enumerate() {
        for flow in sources[node].iter().filter(|flow| !flow.end.is_empty()) {
            for &(consumer, guard) in consumers.get(&flow.end).into_iter().flatten() {
                if consumer == producer {
                    continue;
                }
                let together = topology.diagram.apply(BooleanOp::And, guard, flow.guard)?;
                if topology.diagram.constrain(together, topology.domain)? != ZERO
                    && successors[consumer].insert(producer)
                {
                    indegrees[producer] += 1;
                }
            }
        }
    }
    let mut ready = indegrees
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    let mut ordered = Vec::new();
    while let Some(index) = ready.pop_first() {
        ordered.push(nodes[index]);
        for &next in &successors[index] {
            indegrees[next] -= 1;
            if indegrees[next] == 0 {
                ready.insert(next);
            }
        }
    }
    if ordered.len() != nodes.len() {
        return Err(assembly("cyclic same-layer boundary-flow dependency"));
    }
    nodes.copy_from_slice(&ordered);
    Ok(())
}

fn collect_members(node: &BloqNode, out: &mut Vec<usize>) {
    if let Some(quantum) = node.try_quantum() {
        out.extend(
            quantum
                .instances
                .iter()
                .map(|instance| instance.id.0 as usize),
        );
    }
    if let Some(region) = node.try_region() {
        for (_, body) in region.bodies() {
            for (_, node) in body.nodes() {
                collect_members(node, out);
            }
        }
    }
}

fn incident_positions(
    site: ChunkSiteSource,
    graph: &bloq_graph::BlockGraph,
) -> impl Iterator<Item = IVec3> {
    let owner = |endpoint| {
        graph
            .get_endpoint_block(endpoint)
            .map_or(endpoint, bloq_graph::Block::pos)
    };
    let positions = match site {
        ChunkSiteSource::Block(source) | ChunkSiteSource::SpatialPort(source) => [source; 2],
        ChunkSiteSource::TemporalPipe(pipe) => [pipe.src, pipe.dst].map(owner),
        ChunkSiteSource::SpatialPipe(pipe) => [pipe.src, pipe.dst].map(owner),
    };
    positions
        .into_iter()
        .take(if positions[0] == positions[1] { 1 } else { 2 })
}

pub(super) struct BooleanProgram {
    pub(super) values: HashMap<DecisionId, BloqNodeId>,
    variables: Vec<Option<BloqNodeId>>,
    definitions: crate::FxSet<Arc<ClassicalNode>>,
}

impl BooleanProgram {
    fn new(topology: &GuardedTopology, bloq: &mut Bloq) -> Self {
        let variables = topology
            .variables
            .iter()
            .map(|variable| match variable {
                GuardedVariable::Branch { name, .. } => Some(
                    bloq.add_node(
                        BloqNode::classical(ClassicalNode::Compute {
                            expr: ClassicalExpr::Const(false),
                        })
                        .with_provenance(NodeProvenance::BranchSelector { name: name.clone() }),
                    ),
                ),
                GuardedVariable::Outcome(_) => None,
            })
            .collect();
        let mut program = Self {
            values: HashMap::default(),
            variables,
            definitions: crate::FxSet::default(),
        };
        for (index, variable) in topology.variables.iter().enumerate() {
            if let GuardedVariable::Branch { condition, .. } = variable {
                let condition = program.emit(*condition, &topology.diagram, bloq);
                let node =
                    program.variables[index].expect("branch variables allocate selector nodes");
                bloq.node_mut(node)
                    .expect("allocated branch selector node exists")
                    .kind = BloqNodeKind::Classical(
                    ClassicalNode::Compute {
                        expr: ClassicalExpr::In(0),
                    }
                    .into(),
                );
                bloq.add_edge(condition, node, BloqEdge::value(0));
            }
        }
        // View pins name the resolved value, without adding independent Boolean
        // coordinates to correlation planning. Guards keep their shared recipes.
        for &(position, condition) in &topology.resolves {
            let condition = program.emit(condition, &topology.diagram, bloq);
            let selector = bloq.add_node(
                BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::In(0),
                })
                .with_provenance(NodeProvenance::BranchSelector {
                    name: bloq_graph::selective_selector_name(position),
                }),
            );
            bloq.add_edge(condition, selector, BloqEdge::value(0));
        }
        program
    }

    pub(super) fn emit(
        &mut self,
        value: DecisionId,
        diagram: &BooleanDecisionDiagram,
        bloq: &mut Bloq,
    ) -> BloqNodeId {
        if let Some(&node) = self.values.get(&value) {
            return node;
        }
        let mut pending = vec![(value, false)];
        while let Some((value, finish)) = pending.pop() {
            if self.values.contains_key(&value) {
                continue;
            }
            let decision = diagram.node(value);
            let complement = decision
                .filter(|node| node.low != ZERO || node.high != ONE)
                .and_then(|_| diagram.cached_complement(value))
                .and_then(|other| self.values.get(&other).copied());
            let node = if let Some(producer) = complement {
                let node =
                    self.emit_compute(ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))), bloq);
                bloq.add_edge(producer, node, BloqEdge::value(0));
                node
            } else if let Some(decision) = decision {
                // Preserve the original allocation order: the variable first,
                // then the low subtree, high subtree, and this selection.
                let variable = *self.variables[decision.variable].get_or_insert_with(|| {
                    bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(false),
                    }))
                });
                if decision.low == ZERO && decision.high == ONE {
                    variable
                } else if !finish {
                    pending.push((value, true));
                    for child in [decision.high, decision.low] {
                        if child != ZERO && child != ONE && !self.values.contains_key(&child) {
                            pending.push((child, false));
                        }
                    }
                    continue;
                } else {
                    let mut inputs = [Some(variable), None, None];
                    let mut operand = |slot: usize, child| {
                        if child == ZERO || child == ONE {
                            ClassicalExpr::Const(child == ONE)
                        } else {
                            inputs[slot] = Some(self.values[&child]);
                            ClassicalExpr::In(slot as u32)
                        }
                    };
                    let expr = ClassicalExpr::select(
                        ClassicalExpr::In(0),
                        operand(1, decision.low),
                        operand(2, decision.high),
                    );
                    let node = self.emit_compute(expr, bloq);
                    for (slot, producer) in inputs.into_iter().enumerate() {
                        if let Some(producer) = producer {
                            bloq.add_edge(producer, node, BloqEdge::value(slot as u32));
                        }
                    }
                    node
                }
            } else {
                self.emit_compute(ClassicalExpr::Const(value == ONE), bloq)
            };
            self.values.insert(value, node);
        }
        self.values[&value]
    }

    fn emit_compute(&mut self, expr: ClassicalExpr, bloq: &mut Bloq) -> BloqNodeId {
        let data = ClassicalNode::Compute { expr };
        let definition = if let Some(shared) = self.definitions.get(&data) {
            Arc::clone(shared)
        } else {
            let shared = Arc::new(data);
            self.definitions.insert(Arc::clone(&shared));
            shared
        };
        bloq.add_node(BloqNode::classical(definition))
    }

    pub(super) fn bind(
        &mut self,
        topology: &GuardedTopology,
        readouts: &HashMap<String, BloqNodeId>,
        bloq: &mut Bloq,
    ) -> Result<(), CompileError> {
        for (variable, node) in topology.variables.iter().zip(&self.variables) {
            let (GuardedVariable::Outcome(name), Some(node)) = (variable, node) else {
                continue;
            };
            let readout = readouts.get(name).ok_or_else(|| {
                assembly(format!("control '{name}' has no corrected named readout"))
            })?;
            bloq.node_mut(*node)
                .expect("allocated outcome selector node exists")
                .kind = BloqNodeKind::Classical(
                ClassicalNode::Compute {
                    expr: ClassicalExpr::In(0),
                }
                .into(),
            );
            bloq.add_edge(*readout, *node, BloqEdge::value(0));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, WalkingBoundaryKind, WalkingKind};
    use bloq_ir::TemporalPipeRef;

    use super::*;

    #[test]
    fn deep_boolean_emission_preserves_order_slots_and_shared_children_on_small_stack() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                const DEPTH: usize = 20_000;
                let mut diagram = BooleanDecisionDiagram::default();
                // Allocate the high leaf first in the BDD; emission must still
                // visit the deep low branch first and reuse this shared high leaf.
                let high = diagram.make_node(DEPTH + 1, ZERO, ONE).unwrap();
                let mut root = diagram.make_node(DEPTH, ZERO, ONE).unwrap();
                for variable in (0..DEPTH).rev() {
                    root = diagram.make_node(variable, root, high).unwrap();
                }
                let mut boolean = BooleanProgram {
                    values: HashMap::default(),
                    variables: vec![None; DEPTH + 2],
                    definitions: crate::FxSet::default(),
                };
                let mut bloq = Bloq::new();
                let emitted = boolean.emit(root, &diagram, &mut bloq);
                assert_eq!(boolean.variables[DEPTH], Some(BloqNodeId(DEPTH as u32)));
                assert_eq!(
                    boolean.variables[DEPTH + 1],
                    Some(BloqNodeId(DEPTH as u32 + 1))
                );
                assert_eq!(bloq.top().node_count(), 2 * DEPTH + 2);
                assert_eq!(boolean.values.len(), diagram.nodes().len());
                assert_eq!(boolean.emit(root, &diagram, &mut bloq), emitted);
                assert_eq!(bloq.top().node_count(), 2 * DEPTH + 2);
                let inputs = bloq
                    .top()
                    .value_inputs(emitted)
                    .map(|input| (input.slot, input.producer))
                    .collect::<BTreeMap<_, _>>();
                let decision = diagram.node(root).unwrap();
                assert_eq!(inputs[&0], boolean.variables[0].unwrap());
                assert_eq!(inputs[&1], boolean.values[&decision.low]);
                assert_eq!(inputs[&2], boolean.values[&high]);
                let Some(ClassicalNode::Compute { expr }) = bloq[emitted].try_classical() else {
                    panic!("a nontrivial decision emits a Compute");
                };
                assert_eq!(expr.eval(&mut |slot| Some(slot == 1)), Some(true));
                assert_eq!(expr.eval(&mut |slot| Some(slot != 2)), Some(false));
                assert_eq!(
                    expr.eval(&mut |slot| (slot != 1).then_some(true)),
                    None,
                    "an unavailable unselected low operand must still propagate"
                );
                let complement = diagram.negate(root).unwrap();
                let before = bloq.top().node_count();
                let negated = boolean.emit(complement, &diagram, &mut bloq);
                assert_eq!(bloq.top().node_count(), before + 1);
                assert_eq!(
                    bloq.top().value_inputs(negated).next().unwrap().producer,
                    emitted
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn cached_complement_emits_exact_strict_not() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, ZERO, ONE).unwrap();
        let b = diagram.make_node(1, ZERO, ONE).unwrap();
        let value = diagram.apply(BooleanOp::And, a, b).unwrap();
        let mut complement = diagram.negate(value).unwrap();
        let mut value = value;
        diagram.collect_garbage([&mut value, &mut complement]);
        let mut boolean = BooleanProgram {
            values: HashMap::default(),
            variables: vec![None; 2],
            definitions: crate::FxSet::default(),
        };
        let mut bloq = Bloq::new();
        let producer = boolean.emit(value, &diagram, &mut bloq);
        let before = bloq.top().node_count();
        let consumer = boolean.emit(complement, &diagram, &mut bloq);
        assert_eq!(bloq.top().node_count(), before + 1);
        let Some(ClassicalNode::Compute { expr }) = bloq[consumer].try_classical() else {
            panic!("cached complement emits a Compute");
        };
        assert!(matches!(expr, ClassicalExpr::Not(_)));
        assert_eq!(
            bloq.top().value_inputs(consumer).next().unwrap().producer,
            producer
        );
        for mask in 0..4 {
            let bit = |variable| mask & (1 << variable) != 0;
            assert_eq!(
                expr.eval(&mut |_| Some(diagram.evaluate(value, bit))),
                Some(diagram.evaluate(complement, bit))
            );
        }
        assert_eq!(expr.eval(&mut |_| None), None);

        let mut diagram = BooleanDecisionDiagram::default();
        let positive = diagram.make_node(0, ZERO, ONE).unwrap();
        let negative = diagram.negate(positive).unwrap();
        let mut boolean = BooleanProgram {
            values: HashMap::default(),
            variables: vec![None],
            definitions: crate::FxSet::default(),
        };
        let mut bloq = Bloq::new();
        boolean.emit(negative, &diagram, &mut bloq);
        let before = bloq.top().node_count();
        assert_eq!(
            boolean.emit(positive, &diagram, &mut bloq),
            boolean.variables[0].unwrap()
        );
        assert_eq!(bloq.top().node_count(), before);
    }

    #[test]
    fn empty_walking_choreography_needs_no_work() {
        let mut topology = GuardedTopology::new(
            &BlockGraph::new(),
            bloq_graph::ModuleCertificationLimits::DEFAULT,
        )
        .unwrap();
        topology.diagram =
            BooleanDecisionDiagram::with_limits(bloq_utils::boolean::BooleanLimits {
                max_steps: 0,
                ..bloq_utils::boolean::BooleanLimits::DEFAULT
            });
        let mut assembly = Assembly::new();
        let from = assembly.bloq.add_node(BloqNode::from_members(Vec::new()));
        let to = assembly.bloq.add_node(BloqNode::from_members(Vec::new()));
        assembly.bloq.add_edge(from, to, BloqEdge::Order);
        assembly.order_walking_choreography(&mut topology).unwrap();
        add_walking_order_edges(assembly.bloq.top_mut(), &[], &mut topology).unwrap();
        assert_eq!(topology.diagram.steps(), 0);
        assert_eq!(assembly.bloq.top().edge_count(), 1);
    }

    #[test]
    fn walking_choreography_joins_owned_sites_only_when_guards_coexist() {
        let mut topology = GuardedTopology::new(
            &BlockGraph::new(),
            bloq_graph::ModuleCertificationLimits::DEFAULT,
        )
        .unwrap();
        let guard = topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = topology.diagram.negate(guard).unwrap();
        let mut body = SubGraph::new();
        let entering = body.add_node(BloqNode::from_members(Vec::new()));
        let vacating = body.add_node(BloqNode::from_members(Vec::new()));
        let placement = |position, owner, guard| {
            (
                position,
                WalkingKind::DEFAULT.try_end_position(position).unwrap() - IVec3::Z,
                owner,
                guard,
            )
        };
        // No pipe joins these sites, and an unregistered context neighbor
        // contributes no placement to this global choreography join.
        let enters = placement(IVec3::ZERO, entering, guard);
        add_walking_order_edges(&mut body, &[enters], &mut topology).unwrap();
        assert_eq!(body.edge_count(), 0);
        let exclusive = placement(IVec3::X, vacating, absent);
        add_walking_order_edges(&mut body, &[enters, exclusive], &mut topology).unwrap();
        assert_eq!(body.edge_count(), 0);
        let together = placement(IVec3::X, vacating, ONE);
        for _ in 0..2 {
            add_walking_order_edges(&mut body, &[enters, enters, together], &mut topology).unwrap();
        }
        assert_eq!(body.edge_count(), 1);
        let edge = body.edges().next().unwrap();
        assert_eq!((edge.source, edge.target), (vacating, entering));
        assert_eq!(edge.edge, &BloqEdge::Order);
    }

    #[test]
    fn incidence_preserves_extended_owners_and_deduplicates_virtual_seams() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().unwrap())
                .unwrap(),
        );
        let walk = IVec3::new(4, 0, 0);
        graph.add_block(Block::new(
            walk,
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::IVec2::ONE).unwrap(),
            ),
        ));
        let top = 2 * IVec3::Z;
        let port = 3 * IVec3::Z;
        for (src, dst, expected) in [
            (top, port, vec![IVec3::ZERO, port]),
            (walk + IVec3::ONE, port, vec![walk, port]),
            (IVec3::ZERO, top, vec![IVec3::ZERO]),
            (port, port, vec![port]),
        ] {
            for site in [
                ChunkSiteSource::TemporalPipe(TemporalPipeRef {
                    src,
                    dst,
                    hadamard: true,
                }),
                ChunkSiteSource::SpatialPipe(crate::lower::SpatialPipeRef { src, dst }),
            ] {
                assert_eq!(
                    incident_positions(site, &graph).collect::<Vec<_>>(),
                    expected
                );
            }
        }
        for site in [
            ChunkSiteSource::Block(top),
            ChunkSiteSource::SpatialPort(top),
        ] {
            assert_eq!(incident_positions(site, &graph).collect::<Vec<_>>(), [top]);
        }
    }
}
