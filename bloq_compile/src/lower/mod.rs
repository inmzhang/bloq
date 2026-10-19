//! Lowering: assembling per-block templates and the temporal/spatial plan into
//! a complete [`Bloq`] program — placing template instances, composing
//! detectors, observables, and output frames, and wiring the classical
//! dataflow for selectives, feedback, and regions.

use bloq_circuit::PauliMap;
use bloq_graph::{Basis, BlockGraph, BlockKind, Pauli, PortRole};
use bloq_ir::{
    Bloq, BloqValidationError, ClassicalNode, LogicalInput, LogicalOutput, NodeDetector,
    TemplateId, TemporalPipeRef, WalkControl,
    lowering::{NodeRestart, TemplateInstanceId},
};
use glam::{IVec2, IVec3};
use petgraph::graph::NodeIndex;

use crate::CompileError;
use crate::block::{LoweringTemplateId, LoweringTemplatePool};
use crate::port::PortExpansionMap;

mod classify;
mod detector;
mod gateway;
pub(crate) use gateway::{BoundDynamicAnchor, LocalSurfaceContext, TSourceGateway};
mod leaves;
mod observable;
pub(crate) use observable::physical_neighbor;
#[cfg(test)]
pub(crate) use observable::{patch_touches_dynamic_block, source_t_block};
mod proxy;
pub(crate) use proxy::lower_clifford_proxy;
mod physical;
pub(crate) use physical::{
    PhysicalInput, PhysicalPlacements, PhysicalProgram, PlacedPlan, SelectiveArmInstances,
};
mod place;
mod plan;
mod qubit_layout;
mod region;
mod schedule;

pub use qubit_layout::validate_bloq_qubit_layout_for_source;
pub(crate) use qubit_layout::validate_bloq_qubit_layout_with_limits;

use classify::NodeLowering;
pub(crate) use plan::{LinkedPlanInput, LowerPlan, SpatialPipeRef, TemplatePlan};
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoweredTemplateInstance {
    pub(crate) site_source: ChunkSiteSource,
    pub(crate) instance: PlacedInstance,
}

/// A placed template instance still in pool space: the `bloq_ir::TemplateInstance`
/// shape, but `template` is the pool-space [`LoweringTemplateId`]. It is resolved
/// to a program `TemplateId` only through [`TemplateRemapper::remap`], so no
/// pass can mistake a pool id for a program id.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlacedInstance {
    pub(crate) id: TemplateInstanceId,
    pub(crate) template: LoweringTemplateId,
    pub(crate) offset: IVec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ChunkSiteSource {
    Block(IVec3),
    TemporalPipe(TemporalPipeRef),
    /// A spatial Hadamard wall. Unlike a temporal pipe it has no plan node of
    /// its own — its instance rides the spatial component node its two endpoint
    /// cubes belong to.
    SpatialPipe(SpatialPipeRef),
    /// Positionless temporal Port substituted for an authored spatial Port.
    SpatialPort(IVec3),
}

#[derive(Debug, Default)]
pub(super) struct TemplateInstanceAllocator {
    next: u32,
}

struct BloqLowerContext<'a> {
    input: PhysicalInput<'a>,
    classified: &'a [NodeLowering],
    /// Emission rank per plan node, indexed by NodeIndex::index().
    emit_order: &'a [usize],
    node_instances: &'a [Vec<LoweredTemplateInstance>],
    node_detectors: &'a mut [Vec<NodeDetector>],
    node_restarts: &'a mut [Vec<NodeRestart>],
}

impl TemplateInstanceAllocator {
    pub(super) fn allocate(&mut self) -> TemplateInstanceId {
        let id = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("template instance ids fit in u32");
        TemplateInstanceId(id)
    }
}

impl BloqLowerContext<'_> {
    fn site_node(&self, source: ChunkSiteSource) -> NodeIndex {
        match source {
            ChunkSiteSource::Block(block_pos) => self.input.plan.node_by_block()[&block_pos],
            ChunkSiteSource::TemporalPipe(pipe) => self.input.plan.node_by_temporal_pipe()[&pipe],
            ChunkSiteSource::SpatialPort(port) => self.input.plan.node_by_spatial_port()[&port],
            // A wall lives on the component node holding both its endpoints.
            ChunkSiteSource::SpatialPipe(pipe) => self.input.plan.node_by_block()[&pipe.src],
        }
    }

    /// The lowered template instance that `source` resolves to on its plan node.
    fn instance_for_site(&self, source: ChunkSiteSource) -> &LoweredTemplateInstance {
        let node = self.site_node(source);
        self.node_instances[node.index()]
            .iter()
            .find(|instance| instance.site_source == source)
            .expect("observable insertion site has a lowered template instance")
    }
}

/// Source cut ownership needed after physical/semantic assembly is released.
/// Operator inference still runs on optimized IR; the expanded source graph,
/// signature maps, readout plan, and Boolean caches are no longer needed then.
type BoundPorts = crate::FxMap<TemplateInstanceId, (IVec3, Option<IVec2>)>;

pub(crate) struct LogicalInterface {
    inputs: BoundPorts,
    outputs: BoundPorts,
    distance: u32,
}

impl LogicalInterface {
    pub(crate) fn bind(
        bloq: &Bloq,
        graph: &BlockGraph,
        instance_by_site: &crate::FxMap<ChunkSiteSource, TemplateInstanceId>,
        ports: &PortExpansionMap,
        distance: u32,
    ) -> Self {
        let frame_ports = bloq
            .output_frames()
            .into_iter()
            .map(|frame| frame.port)
            .collect::<crate::FxSet<_>>();
        let mut inputs = BoundPorts::default();
        let mut outputs = BoundPorts::default();
        for (&site, &instance) in instance_by_site {
            let (position, expansion) = match site {
                ChunkSiteSource::SpatialPort(position) => (position, Some(ports[&position])),
                ChunkSiteSource::Block(position)
                    if graph
                        .get_block(position)
                        .is_some_and(|block| block.kind() == BlockKind::Port) =>
                {
                    (position, ports.get(&position).copied())
                }
                _ => continue,
            };
            let role = expansion.map(|port| port.role).unwrap_or_else(|| {
                let role = graph
                    .get_block(position)
                    .expect("Port exists")
                    .port_role()
                    .expect("Port has role");
                if role != PortRole::Auto {
                    role
                } else if graph
                    .neighbor_positions(position)
                    .iter()
                    .all(|neighbor| neighbor.z > position.z)
                {
                    PortRole::Input
                } else {
                    PortRole::Output
                }
            });
            if role.has_input_boundary() {
                let private = expansion
                    .filter(|port| !port.is_spatial())
                    .and_then(|port| port.private_qubit);
                inputs.insert(instance, (position, private));
            }
            if role.has_output_boundary() && frame_ports.contains(&position) {
                outputs.insert(
                    instance,
                    (position, expansion.and_then(|port| port.private_qubit)),
                );
            }
        }
        Self {
            inputs,
            outputs,
            distance,
        }
    }
}

/// Shared finalization after every physical member and semantic recipe is bound.
pub(crate) fn finish_program(
    bloq: &mut Bloq,
    interface: LogicalInterface,
) -> Result<(), CompileError> {
    // Port-adapter outcomes already enter readouts and frames through their gateways.
    schedule::induce_occupancy_order_edges(bloq)?;
    bloq.optimize()
        .map_err(|_| BloqValidationError::CyclicGraph)?;
    let inputs = logical_boundaries(
        bloq,
        interface.inputs,
        bloq_ir::BoundaryFace::Input,
        interface.distance,
    )?;
    bloq.set_logical_inputs(
        inputs
            .into_iter()
            .map(|(port, instance, x, z)| LogicalInput {
                port,
                instance,
                x,
                z,
            })
            .collect(),
    );
    let outputs = logical_boundaries(
        bloq,
        interface.outputs,
        bloq_ir::BoundaryFace::Output,
        interface.distance,
    )?;
    bloq.set_logical_outputs(
        outputs
            .into_iter()
            .map(|(port, instance, x, z)| LogicalOutput {
                port,
                instance,
                x,
                z,
            })
            .collect(),
    );
    Ok(())
}

/// Persist the complete boundary operators an execution hook may initialize
/// or read, sorted by port.
///
/// Ordinary temporal Ports, reversed temporal Port adapters, and
/// compiler-expanded spatial Ports land here. The bound interface is the
/// authoritative source→instance map; the executor must not reconstruct it
/// from fused-node provenance. Compiler geometry stays here; consumers read
/// only the resulting IR maps.
fn logical_boundaries(
    bloq: &Bloq,
    ports_by_instance: BoundPorts,
    face: bloq_ir::BoundaryFace,
    distance: u32,
) -> Result<Vec<(IVec3, TemplateInstanceId, PauliMap, PauliMap)>, CompileError> {
    let (placements, mut faces) = logical_faces(bloq, face);

    let mut boundaries = Vec::new();
    for (instance, (port, private)) in ports_by_instance {
        let (template, offset) = placements[&instance];
        let pair = if let Some(qubit) = private {
            Some(private_boundary_pair(bloq, instance, port, qubit, offset)?)
        } else {
            boundary_operators(
                &bloq.templates()[template],
                faces.remove(&instance).unwrap_or_default(),
                face,
                offset,
                distance,
            )?
        };
        if let Some((x, z)) = pair {
            boundaries.push((port, instance, x, z));
        }
    }
    boundaries.sort_by_key(|(port, ..)| port.to_array());
    Ok(boundaries)
}

fn private_boundary_pair(
    bloq: &Bloq,
    instance: TemplateInstanceId,
    port: IVec3,
    qubit: IVec2,
    offset: IVec2,
) -> Result<(PauliMap, PauliMap), CompileError> {
    let qubit = bloq_circuit::checked_translate_coordinate(qubit, offset)?;
    for (_, node) in bloq.quantum_nodes() {
        for other in &node.instances {
            if other.id == instance {
                continue;
            }
            for local in bloq.templates()[other.template_id].qubits() {
                if bloq_circuit::checked_translate_coordinate(*local, other.offset)? == qubit {
                    return Err(CompileError::PrivatePortCoordinateUnavailable { port });
                }
            }
        }
    }
    Ok((
        PauliMap::from_unique_entries([(qubit, Pauli::X)]),
        PauliMap::from_unique_entries([(qubit, Pauli::Z)]),
    ))
}

type LogicalFaces = crate::FxMap<TemplateInstanceId, (Option<PauliMap>, Option<PauliMap>)>;

/// Complete observed boundary operators in template coordinates, then place them.
fn boundary_operators(
    template: &bloq_ir::lowering::BloqTemplate,
    observed: (Option<PauliMap>, Option<PauliMap>),
    face: bloq_ir::BoundaryFace,
    offset: IVec2,
    distance: u32,
) -> Result<Option<(PauliMap, PauliMap)>, CompileError> {
    let x = observed.0.map(|map| rebase_map(&map, offset)).transpose()?;
    let z = observed.1.map(|map| rebase_map(&map, offset)).transpose()?;
    complete_logical_pair(x, z)
        .or_else(|| template_logical_pair(template, face, distance))
        .map(|(x, z)| Ok((x.try_translated(offset)?, z.try_translated(offset)?)))
        .transpose()
}

fn logical_faces(
    bloq: &Bloq,
    face: bloq_ir::BoundaryFace,
) -> (
    crate::FxMap<TemplateInstanceId, (TemplateId, IVec2)>,
    LogicalFaces,
) {
    let mut placements = crate::FxMap::default();
    let mut faces: LogicalFaces = crate::FxMap::default();
    bloq.walk(|cx| {
        if let Some(quantum) = cx.node.try_quantum() {
            placements.extend(
                quantum
                    .instances
                    .iter()
                    .map(|instance| (instance.id, (instance.template_id, instance.offset))),
            );
        }
        if let Some(ClassicalNode::Observable { operators, .. }) = cx.node.try_classical() {
            for operator in operators.iter().filter(|operator| operator.face == face) {
                let entry = faces.entry(operator.instance).or_default();
                match homogeneous_pauli(&operator.operator) {
                    Some(bloq_circuit::Pauli::X) => {
                        entry.0.get_or_insert_with(|| operator.operator.clone());
                    }
                    Some(bloq_circuit::Pauli::Z) => {
                        entry.1.get_or_insert_with(|| operator.operator.clone());
                    }
                    _ => {}
                }
            }
        }
        WalkControl::Continue
    });
    (placements, faces)
}

fn template_logical_pair(
    template: &bloq_ir::lowering::BloqTemplate,
    face: bloq_ir::BoundaryFace,
    distance: u32,
) -> Option<(PauliMap, PauliMap)> {
    use crate::block::fixed_bulk::observable::logical_line_operator;

    let maps = template
        .boundary_flows
        .iter()
        .filter_map(|flow| match face {
            bloq_ir::BoundaryFace::Input if flow.start.is_empty() && !flow.end.is_empty() => {
                Some(&flow.end)
            }
            bloq_ir::BoundaryFace::Output if !flow.start.is_empty() && flow.end.is_empty() => {
                Some(&flow.start)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    [Basis::Z, Basis::X].into_iter().find_map(|top_basis| {
        let x = logical_line_operator(distance, Basis::X, top_basis, Pauli::X);
        let z = logical_line_operator(distance, Basis::Z, top_basis, Pauli::Z);
        maps.iter()
            .all(|stabilizer| {
                !pauli_maps_anticommute(&x, stabilizer) && !pauli_maps_anticommute(&z, stabilizer)
            })
            .then_some((x, z))
    })
}

fn pauli_maps_anticommute(a: &PauliMap, b: &PauliMap) -> bool {
    a.iter().fold(false, |odd, (coordinate, pauli)| {
        odd ^ b
            .get(coordinate)
            .is_some_and(|other| pauli.anticommutes(*other))
    })
}

fn homogeneous_pauli(map: &PauliMap) -> Option<bloq_circuit::Pauli> {
    let pauli = *map.iter().next()?.1;
    map.iter()
        .all(|(_, candidate)| *candidate == pauli)
        .then_some(pauli)
}

fn rebase_map(map: &PauliMap, offset: IVec2) -> Result<PauliMap, CompileError> {
    map.iter()
        .map(|(coordinate, pauli)| {
            let x = coordinate.x.checked_sub(offset.x).ok_or(
                CompileError::LogicalBoundaryRebaseOverflow {
                    coordinate: *coordinate,
                    offset,
                },
            )?;
            let y = coordinate.y.checked_sub(offset.y).ok_or(
                CompileError::LogicalBoundaryRebaseOverflow {
                    coordinate: *coordinate,
                    offset,
                },
            )?;
            Ok((IVec2::new(x, y), *pauli))
        })
        .collect()
}

fn complete_logical_pair(x: Option<PauliMap>, z: Option<PauliMap>) -> Option<(PauliMap, PauliMap)> {
    match (x, z) {
        (Some(x), Some(z)) => Some((x, z)),
        (Some(x), None) => perpendicular_line(&x, Pauli::X).map(|z| (x, z)),
        (None, Some(z)) => perpendicular_line(&z, Pauli::Z).map(|x| (x, z)),
        (None, None) => None,
    }
}

fn perpendicular_line(known: &PauliMap, known_pauli: Pauli) -> Option<PauliMap> {
    use crate::block::fixed_bulk::observable::logical_line_operator;

    let distance = known.len() as u32;
    let (known_basis, other_basis, other_pauli) = match known_pauli {
        Pauli::X => (Basis::X, Basis::Z, Pauli::Z),
        _ => (Basis::Z, Basis::X, Pauli::X),
    };
    let top_basis = [Basis::Z, Basis::X]
        .into_iter()
        .find(|&top| &logical_line_operator(distance, known_basis, top, known_pauli) == known)?;
    Some(logical_line_operator(
        distance,
        other_basis,
        top_basis,
        other_pauli,
    ))
}

/// The program under construction, the pool it copies templates from, and the
/// dedup table between them. Bundled because instantiating a template always
/// needs all three, and region building threads them through every layer.
pub(super) struct TemplateRemapper<'a> {
    bloq: &'a mut Bloq,
    program_template_ids: &'a mut crate::FxMap<LoweringTemplateId, TemplateId>,
    pool: &'a LoweringTemplatePool,
}

impl<'a> TemplateRemapper<'a> {
    fn new(
        bloq: &'a mut Bloq,
        program_template_ids: &'a mut crate::FxMap<LoweringTemplateId, TemplateId>,
        pool: &'a LoweringTemplatePool,
    ) -> Self {
        Self {
            bloq,
            program_template_ids,
            pool,
        }
    }

    /// Lazily copy a pool template into the program, dedup'd by
    /// `LoweringTemplateId`, and return its program-local `TemplateId`.
    pub(super) fn remap(&mut self, lowering_template_id: LoweringTemplateId) -> TemplateId {
        let (bloq, pool) = (&mut *self.bloq, self.pool);
        *self
            .program_template_ids
            .entry(lowering_template_id)
            .or_insert_with(|| {
                bloq.add_shared_template(pool[lowering_template_id].program_template.clone())
            })
    }

    /// The pool being copied from, for callers that read a template's side
    /// tables (gateways, seams) rather than instantiate it.
    pub(super) fn pool(&self) -> &LoweringTemplatePool {
        self.pool
    }
}
