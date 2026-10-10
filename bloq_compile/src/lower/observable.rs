use std::collections::BTreeMap;

use bloq_circuit::PauliMap;
use bloq_graph::{
    Block, BlockGraph, Direction, Pauli, Stabilizer, StabilizerGenerator, StabilizerGenerators,
    StabilizerRowKind, SurfaceSupport, checked_add_position,
};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BoundaryFace, ClassicalNode,
    lowering::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstanceId},
};
use glam::IVec3;
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use smallvec::SmallVec;

use crate::CompileError;
use crate::block::gateway::GatewayEntry;
use crate::block::{LocalStabilizer, LoweringTemplateId, LoweringTemplatePool};
use crate::compile::CompiledTemplateMap;
use crate::signature::Connectivity;
use crate::spatial_port::SpatialPortExpansionMap;

use bloq_ir::NodeProvenance;

use super::{BloqLowerContext, ChunkSiteSource};

/// One observable insertion resolved into instance-global references: its
/// measurement parity plus its two boundary-operator faces, tagged with
/// the producing quantum node so [`materialize_observable_nodes`] can attach the
/// read-after-measure `Order` edge. Owned (no borrow of the lowering context) so
/// materialization runs after the context is dropped.
pub(super) struct ResolvedObservable {
    pub(super) index: u32,
    /// The plan node that produced these measurements (the read-after-measure
    /// source). Resolved to a `BloqNodeId` at materialization.
    producer: NodeIndex,
    /// Owning instance for boundary bindings.
    instance: TemplateInstanceId,
    measurements: Vec<InstanceMeasurement>,
    operator_in: PauliMap,
    operator_out: PauliMap,
}

/// Template-local observable include requested by a graph stabilizer.
///
/// Measurements are already in template-id space (the gateway is canonicalized
/// at template-build time), so lowering only relabels them to instance ids.
/// Boundary logical operators in the block's local qubit
/// coordinates, kept distinct per temporal face. An operator-only insertion has
/// empty `measurements`; a measurement-only insertion has empty operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ObservableChunkInsertion {
    pub(super) site: ChunkSiteSource,
    pub(super) observable_index: u32,
    pub(super) measurements: Vec<u32>,
    pub(super) operator_in: PauliMap,
    pub(super) operator_out: PauliMap,
    pub(super) provenance_in: FaceProvenance,
    pub(super) provenance_out: FaceProvenance,
}

/// Whether an insertion's boundary-operator face meets static content across
/// its seam (another fixed instance's operator, compared and cancelled) or
/// dynamic content (a runtime-resolved selective block, which emits no static
/// counterpart — the operator terminates there; its continuation is owned by
/// the resolved measurement). Stamped at construction time, where the
/// block graph and the compiled-template map meet; consumed by
/// [`cancel_seam_operators`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FaceProvenance {
    Static,
    Dynamic,
}

impl ObservableChunkInsertion {
    /// A measurement-only insertion (no boundary operators on either face).
    fn measurements(site: ChunkSiteSource, observable_index: u32, measurements: Vec<u32>) -> Self {
        Self {
            site,
            observable_index,
            measurements,
            operator_in: PauliMap::empty(),
            operator_out: PauliMap::empty(),
            provenance_in: FaceProvenance::Static,
            provenance_out: FaceProvenance::Static,
        }
    }

    /// An operator-only insertion (no measurement records) carrying a block's two
    /// boundary faces, or `None` when both are empty so there is nothing to attach.
    pub(super) fn operators(
        site: ChunkSiteSource,
        observable_index: u32,
        operator_in: PauliMap,
        operator_out: PauliMap,
        provenance_in: FaceProvenance,
        provenance_out: FaceProvenance,
    ) -> Option<Self> {
        if operator_in.is_empty() && operator_out.is_empty() {
            return None;
        }
        Some(Self {
            site,
            observable_index,
            measurements: Vec::new(),
            operator_in,
            operator_out,
            provenance_in,
            provenance_out,
        })
    }
}

/// Push one measurement-only insertion per non-empty chunk of a gateway
/// entry's records, XOR-canonicalized, against the block at `pos`.
fn push_measurement_insertions(
    insertions: &mut Vec<ObservableChunkInsertion>,
    site: ChunkSiteSource,
    chunk_measurements: &[crate::block::gateway::ChunkMeasurements],
    obs_index: usize,
) {
    for chunk_meas in chunk_measurements {
        debug_assert!(
            chunk_meas
                .measurements
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "template gateway measurements are canonical"
        );
        let measurements = chunk_meas.measurements.clone();
        if measurements.is_empty() {
            continue;
        }
        insertions.push(ObservableChunkInsertion::measurements(
            site,
            obs_index as u32,
            measurements,
        ));
    }
}

/// Resolve graph-level stabilizers into template-local observable insertions.
///
/// This pass intentionally does not mutate block templates. Detectors are
/// composed first at Bloq lowering time, then these insertions are
/// mapped to final measurement ids and appended to the corresponding lowered
/// chunk circuit.
#[cfg(test)]
pub(super) fn resolve_observable_insertions(
    graph: &BlockGraph,
    generators: &[StabilizerGenerator],
    compiled_blocks: &CompiledTemplateMap,
    template_pool: &LoweringTemplatePool,
) -> Result<Vec<ObservableChunkInsertion>, CompileError> {
    resolve_observable_insertions_with_spatial_ports(
        graph,
        generators,
        compiled_blocks,
        &Default::default(),
        &Default::default(),
        template_pool,
    )
}

fn resolve_observable_insertions_with_spatial_ports(
    graph: &BlockGraph,
    generators: &[StabilizerGenerator],
    compiled_blocks: &CompiledTemplateMap,
    spatial_ports: &SpatialPortExpansionMap,
    spatial_port_templates: &crate::compile::CompiledTemplateMap,
    template_pool: &LoweringTemplatePool,
) -> Result<Vec<ObservableChunkInsertion>, CompileError> {
    let mut insertions = Vec::new();
    for (obs_index, stabilizer) in generators.iter().enumerate() {
        if !statically_resolved(stabilizer, compiled_blocks) {
            continue;
        }

        // Resolve every block touched by either node or edge support through one
        // gateway path. Edge-only algebraic support is not a different physical
        // category: `build_local_flow_key` reconstructs its junction Pauli.
        let mut support_positions: Vec<IVec3> = stabilizer
            .stabilizer
            .interior_nodes
            .iter()
            .filter_map(|(&pos, &pauli)| (pauli != Pauli::I).then_some(pos))
            .chain(
                stabilizer
                    .stabilizer
                    .interior_edges
                    .iter()
                    .filter(|(_, pauli)| **pauli != Pauli::I)
                    .flat_map(|(&(a, b), _)| [a, b]),
            )
            .chain(
                stabilizer
                    .stabilizer
                    .port_stabilizer
                    .iter()
                    .filter_map(|(&pos, &pauli)| {
                        (pauli != Pauli::I && spatial_ports.contains_key(&pos)).then_some(pos)
                    }),
            )
            .collect();
        support_positions.sort_unstable_by_key(|pos| (pos.x, pos.y, pos.z));
        support_positions.dedup();
        insertions.extend(
            resolve_observable_insertions_at_positions_with_spatial_ports(
                graph,
                &stabilizer.stabilizer,
                &stabilizer.kind,
                compiled_blocks,
                template_pool,
                &support_positions,
                obs_index as u32,
                false,
                spatial_ports,
                None,
                None,
            )?,
        );

        insertions.extend(resolve_spatial_port_observable_insertions(
            &stabilizer.stabilizer,
            spatial_ports,
            spatial_port_templates,
            template_pool,
            &support_positions,
            obs_index as u32,
            false,
        )?);
    }

    Ok(insertions)
}

#[expect(
    clippy::too_many_arguments,
    reason = "observable resolution needs the complete compiled-site context"
)]
pub(super) fn resolve_observable_insertions_at_positions_with_spatial_ports(
    graph: &BlockGraph,
    stabilizer: &impl SurfaceSupport,
    kind: &StabilizerRowKind,
    compiled_blocks: &CompiledTemplateMap,
    template_pool: &LoweringTemplatePool,
    support_positions: &[IVec3],
    observable_index: u32,
    external_faces_only: bool,
    spatial_ports: &SpatialPortExpansionMap,
    selected_template: Option<(IVec3, LoweringTemplateId)>,
    anchor: Option<super::BoundDynamicAnchor<'_>>,
) -> Result<Vec<ObservableChunkInsertion>, CompileError> {
    let mut insertions = Vec::new();
    for &node_pos in support_positions {
        let pauli = stabilizer.node_pauli(node_pos);
        let Some(compiled) = compiled_blocks.get(&node_pos) else {
            continue;
        };
        // Pin only this gateway's physical template. Neighbor connectivity and
        // dynamic anchors are unchanged; no graph-sized template map is copied.
        let Some(template_id) = selected_template
            .filter(|(position, _)| *position == node_pos)
            .map(|(_, template)| template)
            .or_else(|| compiled.template.observable_template())
        else {
            continue;
        };
        let template = template_pool
            .get(template_id)
            .expect("compiled block references an existing template");
        let gateway = &template.observable_gateway;
        let local_flow = build_local_flow_key(
            graph,
            node_pos,
            pauli,
            stabilizer,
            compiled.graph_connectivity,
            spatial_ports
                .get(&node_pos)
                .map(|port| port.cube_pipe_dir()),
        );
        if local_flow == LocalStabilizer::isolated(Pauli::I) {
            continue;
        }
        if needs_dynamic_anchor(compiled.graph_connectivity, local_flow)
            && anchor.map_or_else(
                || patch_touches_dynamic_block(graph, compiled_blocks, node_pos),
                |anchor| anchor.touches_dynamic,
            )
        {
            if !matches!(kind, StabilizerRowKind::Logical) {
                continue;
            }
            let (t_pos, flipped, t_gateway) = if let Some(anchor) = anchor {
                let Some(source) = anchor.source_t else {
                    continue;
                };
                (source.source, source.flipped, source.gateway)
            } else {
                let Some((t_pos, flipped)) = source_t_block(graph, compiled_blocks, node_pos)
                else {
                    continue;
                };
                let t_template_id = compiled_blocks[&t_pos]
                    .template
                    .observable_template()
                    .expect("a T block has an escape observable template");
                let gateway = &template_pool
                    .get(t_template_id)
                    .expect("escape template exists")
                    .observable_gateway;
                (t_pos, flipped, gateway)
            };
            let query = if flipped {
                LocalStabilizer::isolated(local_flow.center_basis().flip())
            } else {
                local_flow
            };
            let Some(entry) = t_gateway.lookup_ref(query) else {
                return Err(CompileError::ObservableMissingGatewayEntry {
                    block_pos: t_pos,
                    observable_index,
                    local_stabilizer: query.to_string(),
                });
            };
            push_measurement_insertions(
                &mut insertions,
                ChunkSiteSource::Block(t_pos),
                &entry.measurements,
                observable_index as usize,
            );
            continue;
        }
        let Some(entry) = gateway.lookup_ref(local_flow) else {
            return Err(CompileError::ObservableMissingGatewayEntry {
                block_pos: node_pos,
                observable_index,
                local_stabilizer: local_flow.to_string(),
            });
        };
        push_measurement_insertions(
            &mut insertions,
            ChunkSiteSource::Block(node_pos),
            &entry.measurements,
            observable_index as usize,
        );
        let operator_in =
            if external_faces_only && compiled.graph_connectivity.has_pipe(Direction::ZMINUS) {
                PauliMap::empty()
            } else {
                entry.operator_in.clone()
            };
        let operator_out =
            if external_faces_only && compiled.graph_connectivity.has_pipe(Direction::ZPLUS) {
                PauliMap::empty()
            } else {
                entry.operator_out.clone()
            };
        insertions.extend(boundary_operator_insertion(
            graph,
            compiled_blocks,
            compiled.graph_connectivity,
            node_pos,
            observable_index,
            operator_in,
            operator_out,
        ));
    }
    Ok(insertions)
}

pub(super) fn resolve_spatial_port_observable_insertions(
    stabilizer: &impl SurfaceSupport,
    spatial_ports: &SpatialPortExpansionMap,
    spatial_port_templates: &crate::compile::CompiledTemplateMap,
    template_pool: &LoweringTemplatePool,
    support_positions: &[IVec3],
    observable_index: u32,
    external_faces_only: bool,
) -> Result<Vec<ObservableChunkInsertion>, CompileError> {
    let mut insertions = Vec::new();
    for &source in support_positions
        .iter()
        .filter(|source| spatial_ports.contains_key(source))
    {
        // SEM-SPATIAL-PORT: the raw Port Pauli is copied to the virtual
        // temporal half; the derived cube half is resolved separately.
        let pauli = stabilizer.port_pauli(source);
        if pauli == Pauli::I {
            continue;
        }
        let port = spatial_ports[&source];
        let compiled = &spatial_port_templates[&source];
        let template_id = compiled
            .template
            .observable_template()
            .expect("derived temporal Ports use fixed templates");
        let gateway = &template_pool[template_id].observable_gateway;
        let key = LocalStabilizer::new(pauli, compiled.graph_connectivity);
        let Some(entry) = gateway.lookup_ref(key) else {
            return Err(CompileError::ObservableMissingGatewayEntry {
                block_pos: source,
                observable_index,
                local_stabilizer: key.to_string(),
            });
        };
        push_measurement_insertions(
            &mut insertions,
            ChunkSiteSource::SpatialPort(source),
            &entry.measurements,
            observable_index as usize,
        );
        let (operator_in, operator_out) = if !external_faces_only {
            (entry.operator_in.clone(), entry.operator_out.clone())
        } else {
            match port.role {
                bloq_graph::PortRole::Input => (entry.operator_in.clone(), PauliMap::empty()),
                bloq_graph::PortRole::Output => (PauliMap::empty(), entry.operator_out.clone()),
                bloq_graph::PortRole::Multiplex => {
                    let output = port
                        .output_qubit
                        .expect("Multiplex expansions allocate an output qubit");
                    let operator_out = entry
                        .operator_out
                        .get(&output)
                        .copied()
                        .map_or_else(PauliMap::empty, |pauli| {
                            PauliMap::from_unique_entries([(output, pauli)])
                        });
                    (entry.operator_in.clone(), operator_out)
                }
                bloq_graph::PortRole::Auto => unreachable!(),
            }
        };
        if let Some(operators) = ObservableChunkInsertion::operators(
            ChunkSiteSource::SpatialPort(source),
            observable_index,
            operator_in,
            operator_out,
            FaceProvenance::Static,
            FaceProvenance::Static,
        ) {
            insertions.push(operators);
        }
    }
    Ok(insertions)
}

/// Resolve every observable generator into owned [`ResolvedObservable`] records.
/// Runs the same insertion resolution + seam cancellation as
/// before, then relabels each surviving insertion into instance-global refs,
/// tagging it with its producing plan node. Returns owned data so the caller can
/// drop the lowering context before materializing nodes.
pub(super) fn resolve_observables(
    stabilizers: &StabilizerGenerators,
    context: &BloqLowerContext<'_>,
) -> Result<(Vec<ResolvedObservable>, crate::FxSet<u32>), CompileError> {
    let generators = &stabilizers.generators;
    let mut insertions = resolve_observable_insertions_with_spatial_ports(
        context.input.graph,
        generators,
        context.input.compiled,
        context.input.spatial_ports,
        context.input.spatial_port_templates,
        context.input.template_pool,
    )?;
    insertions.extend(resolve_pipe_observable_insertions(
        context.input.graph,
        context.input.plan,
        context.input.compiled,
        context.input.temporal_templates,
        context.input.template_pool,
        generators,
    )?);
    let mut reference_offsets = crate::FxSet::default();
    for (index, generator) in generators.iter().enumerate() {
        if !statically_resolved(generator, context.input.compiled) {
            continue;
        }
        let wall_sign = push_wall_observable_insertions(
            context.input.graph,
            context.input.wall_templates,
            context.input.template_pool,
            &generator.stabilizer,
            None,
            index as u32,
            &mut insertions,
        )?;
        if matches!(generator.kind, StabilizerRowKind::Logical)
            && (wall_sign
                ^ temporal_hadamard_observable_sign(context.input.graph, &generator.stabilizer))
        {
            reference_offsets.insert(index as u32);
        }
    }

    cancel_seam_operators(&mut insertions, context)?;

    let resolved = insertions
        .into_iter()
        .map(|insertion| resolve_observable_insertion(&insertion, context))
        .collect::<Result<_, _>>()?;
    Ok((resolved, reference_offsets))
}

/// Constant parity picked up by a logical row crossing temporal Hadamards.
/// Gateways carry phaseless Paulis, so the `Y_L -> -Y_L` phase must ride as a
/// classical bit inside the projection's conditioned observable content.
pub(super) fn temporal_hadamard_observable_sign(
    graph: &BlockGraph,
    stabilizer: &Stabilizer,
) -> bool {
    stabilizer
        .interior_edges
        .iter()
        .filter_map(|(&(src, dst), &edge_pauli)| {
            let pipe = hadamard_temporal_pipe(graph, src, dst)?;
            Some(realignment_observable_pauli(
                graph, &pipe, stabilizer, edge_pauli,
            ))
        })
        .filter(|pauli| *pauli != Pauli::I)
        .fold(false, |sign, pauli| sign ^ (pauli == Pauli::Y))
}

/// Public row values shared by the pinned proxy's frame constraints.
pub(super) struct MaterializedObservables {
    /// Generator ordinal → public indexed observable.
    pub(super) observable_by_index: crate::FxMap<u32, BloqNodeId>,
    /// Known transport offsets in the pinned public basis.
    pub(super) reference_offsets: crate::FxSet<u32>,
}

/// The statically-resolved generator ordinals — every non-selective-crossing
/// `Measurement` or `Logical` row (the rows [`statically_resolved`] owns). The
/// all-rows completeness rule (design §3) materializes an `Observable` for
/// each, so a row whose gateway resolution yields no records or operators still
/// surfaces as a constant-`false` observable (zero bit inputs) rather than
/// silently having no readable value. Selective-crossing rows are excluded
/// here; the proxy has already pinned them. Selective-internal rows never
/// become observables.
pub(super) fn statically_resolved_row_indices(
    generators: &[StabilizerGenerator],
    compiled_blocks: &CompiledTemplateMap,
) -> Vec<u32> {
    generators
        .iter()
        .enumerate()
        .filter(|(_, generator)| statically_resolved(generator, compiled_blocks))
        .map(|(index, _)| index as u32)
        .collect()
}

/// Materialize the pinned public basis, including empty rows. Static insertions
/// share one complete recipe per row; guarded/shared fragments use composition
/// in ordinary lowering.
pub(super) fn materialize_observable_nodes(
    resolved: Vec<ResolvedObservable>,
    all_row_indices: &[u32],
    bloq: &mut Bloq,
    node_map: &[Option<BloqNodeId>],
) -> MaterializedObservables {
    type Row = (
        Vec<Vec<InstanceMeasurement>>,
        Vec<InstanceBoundaryOperator>,
        crate::FxSet<BloqNodeId>,
    );
    let mut rows = BTreeMap::<u32, Row>::new();
    for observable in resolved {
        let (groups, operators, owners) = rows.entry(observable.index).or_default();
        if !observable.measurements.is_empty() {
            groups.push(observable.measurements);
            owners
                .insert(node_map[observable.producer.index()].expect("producing node was lowered"));
        }
        for (face, operator) in [
            (BoundaryFace::Input, observable.operator_in),
            (BoundaryFace::Output, observable.operator_out),
        ] {
            if !operator.is_empty() {
                operators.push(InstanceBoundaryOperator {
                    instance: observable.instance,
                    face,
                    operator,
                });
            }
        }
    }
    for &index in all_row_indices {
        rows.entry(index).or_default();
    }
    let mut observable_by_index = crate::FxMap::default();
    for (index, (mut groups, operators, owners)) in rows {
        // Each template chunk is ordered; sort ranges before concatenation.
        groups.sort_unstable_by_key(|group| group[0]);
        let measurements = groups.into_iter().flatten().collect::<Vec<_>>();
        debug_assert!(measurements.is_sorted());
        let observable = bloq.add_node(
            BloqNode::classical(ClassicalNode::Observable {
                index: Some(index),
                measurements,
                operators,
            })
            .with_provenance(NodeProvenance::Generator { ordinal: index }),
        );
        let mut owners = owners.into_iter().collect::<Vec<_>>();
        owners.sort_unstable();
        for owner in owners {
            bloq.add_edge(owner, observable, BloqEdge::Order);
        }
        observable_by_index.insert(index, observable);
    }
    MaterializedObservables {
        observable_by_index,
        reference_offsets: Default::default(),
    }
}

/// Match and cancel boundary operators at every temporal seam.
///
/// A logical crossing a coherent seam must be continuous: the emission-earlier
/// (lower) site's `+Z operator_out` must **equal** the emission-later (upper)
/// site's `−Z operator_in`. Equal → cancel both (the seam's measurement + detector
/// carry the logical across). Unequal → the logical is discontinuous, a
/// compile-time bug, raised as [`CompileError::ObservableSeamMismatch`].
///
/// The pass walks the **plan graph's edges**, not raw block-graph pipes, so a
/// temporal Hadamard — its own pipe node between two cubes — is a first-class site:
/// its two incident edges give two seams (`source.out ↔ pipe.in`, `pipe.out ↔
/// dest.in`), with the basis flip living inside the pipe node so each seam stays a
/// plain equality. Each edge carries one `TemporalPipeRef` per member pair, so
/// matching is per member. What survives is the port-boundary faces the backend
/// emits directly.
fn cancel_seam_operators(
    insertions: &mut [ObservableChunkInsertion],
    context: &BloqLowerContext<'_>,
) -> Result<(), CompileError> {
    cancel_seam_operators_in_plan(
        insertions,
        context.input.plan,
        context.input.graph,
        |source| Ok(context.instance_for_site(source).instance.offset),
    )
}

/// Shared seam checker for static and branch-conditioned observable lowering.
/// `offset_for` places each block/pipe-local operator in the global qubit frame.
pub(super) fn cancel_seam_operators_in_plan(
    insertions: &mut [ObservableChunkInsertion],
    plan: &super::LowerPlan,
    graph: &BlockGraph,
    mut offset_for: impl FnMut(ChunkSiteSource) -> Result<glam::IVec2, CompileError>,
) -> Result<(), CompileError> {
    use petgraph::visit::EdgeRef;

    // One operator-carrying insertion per (site, observable). Measurement-only
    // insertions hold empty operators and never participate in a seam. The
    // instance offset placing each site in the global frame is cached alongside,
    // because comparison happens in global coordinates (see below). The
    // per-site observable sets bound the seam walk below: an observable with no
    // operator insertion at either seam site compares empty against empty, so
    // skipping it is a pure win (this pass is the heaviest lowering loop).
    let mut by_site: crate::FxMap<(ChunkSiteSource, u32), usize> = crate::FxMap::default();
    let mut offsets: crate::FxMap<ChunkSiteSource, glam::IVec2> = crate::FxMap::default();
    let mut obs_by_site: crate::FxMap<ChunkSiteSource, SmallVec<[u32; 4]>> =
        crate::FxMap::default();
    for (index, insertion) in insertions.iter().enumerate() {
        if insertion.operator_in.is_empty() && insertion.operator_out.is_empty() {
            continue;
        }
        let previous = by_site.insert((insertion.site, insertion.observable_index), index);
        // Each (site, observable) must carry at most one operator insertion —
        // a silent overwrite here would drop one face's operator from the seam
        // comparison and could mask an `ObservableSeamMismatch`.
        assert!(
            previous.is_none(),
            "duplicate operator insertion for observable {} at {:?}",
            insertion.observable_index,
            insertion.site
        );
        if let std::collections::hash_map::Entry::Vacant(entry) = offsets.entry(insertion.site) {
            entry.insert(offset_for(insertion.site)?);
        }
        obs_by_site
            .entry(insertion.site)
            .or_default()
            .push(insertion.observable_index);
    }
    for observables in obs_by_site.values_mut() {
        observables.sort_unstable();
    }

    // The site that `pipe` touches on `plan_node`: the pipe node itself, or the
    // member block owning whichever pipe endpoint lies in this node.
    let site_for = |plan_node: NodeIndex, pipe: &super::TemporalPipeRef| -> ChunkSiteSource {
        match &plan.graph()[plan_node].provenance {
            NodeProvenance::TemporalPipe { pipe } | NodeProvenance::MemoryPadding { pipe, .. } => {
                ChunkSiteSource::TemporalPipe(*pipe)
            }
            NodeProvenance::BlockComponent { members } => {
                let owner = |endpoint| {
                    graph
                        .get_endpoint_block(endpoint)
                        .expect("temporal pipe endpoint is owned by a block")
                        .pos()
                };
                let src_owner = owner(pipe.src);
                if members.iter().any(|member| member.pos == src_owner) {
                    ChunkSiteSource::Block(src_owner)
                } else {
                    ChunkSiteSource::Block(owner(pipe.dst))
                }
            }
            other => unreachable!("plan nodes carry block/pipe provenance, got {other:?}"),
        }
    };
    let seam_pos = |site: ChunkSiteSource| match site {
        ChunkSiteSource::Block(pos) => pos,
        ChunkSiteSource::TemporalPipe(pipe) => pipe.src,
        // `site_for` only ever names block or temporal-pipe sites: a wall has no
        // plan node and no temporal face, so it never bounds a seam.
        ChunkSiteSource::SpatialPipe(pipe) => pipe.src,
        ChunkSiteSource::SpatialPort(port) => port,
    };
    // A plan edge always runs emission-earlier → emission-later, so the source
    // site owns the `+Z operator_out` and the target site the `−Z operator_in`.
    let empty_obs = SmallVec::<[u32; 4]>::new();
    for edge in plan.graph().edge_references() {
        let seams = edge
            .weight()
            .pipes
            .iter()
            .map(|pipe| {
                (
                    site_for(edge.source(), pipe),
                    site_for(edge.target(), pipe),
                    None,
                )
            })
            .chain(edge.weight().spatial_ports.iter().map(|port| {
                let endpoints = if port.is_input() {
                    (
                        ChunkSiteSource::SpatialPort(port.source),
                        ChunkSiteSource::Block(port.source),
                    )
                } else {
                    (
                        ChunkSiteSource::Block(port.source),
                        ChunkSiteSource::SpatialPort(port.source),
                    )
                };
                (endpoints.0, endpoints.1, Some(*port))
            }));
        for (lower, upper, spatial_port) in seams {
            // Only observables with an operator at one of the two seam sites
            // can cancel or mismatch here; iterate their (ascending) union.
            let lower_obs = obs_by_site.get(&lower).unwrap_or(&empty_obs);
            let upper_obs = obs_by_site.get(&upper).unwrap_or(&empty_obs);
            for &obs in lower_obs.iter().merge(upper_obs).dedup() {
                let out_idx = by_site.get(&(lower, obs)).copied();
                let in_idx = by_site.get(&(upper, obs)).copied();
                // A face stamped `Dynamic` meets a selective block, which
                // emits no static counterpart: terminate the operator here —
                // its continuation is runtime content the resolved measurement
                // owns (U15) — instead of comparing against nothing. A
                // selective never carries an insertion itself, so the stamped
                // face is always the seam's only present side.
                let dynamic = out_idx
                    .is_some_and(|i| insertions[i].provenance_out == FaceProvenance::Dynamic)
                    || in_idx
                        .is_some_and(|i| insertions[i].provenance_in == FaceProvenance::Dynamic);
                if dynamic {
                    if let Some(i) = out_idx {
                        insertions[i].operator_out = PauliMap::empty();
                    }
                    if let Some(i) = in_idx {
                        insertions[i].operator_in = PauliMap::empty();
                    }
                    continue;
                }
                // Multiplex cancels the exact shared patch and exposes X_out separately.
                if let Some(port) =
                    spatial_port.filter(|port| port.role == bloq_graph::PortRole::Multiplex)
                {
                    let lower_offset = offset_for(lower)?;
                    let upper_offset = offset_for(upper)?;
                    let mut lower_patch = out_idx
                        .map(|i| insertions[i].operator_out.try_translated(lower_offset))
                        .transpose()?
                        .unwrap_or_default();
                    let upper_patch = in_idx
                        .map(|i| insertions[i].operator_in.try_translated(upper_offset))
                        .transpose()?
                        .unwrap_or_default();
                    let output_qubit = port
                        .output_qubit
                        .expect("Multiplex expansions allocate an output qubit");
                    let output_global =
                        bloq_circuit::checked_translate_coordinate(output_qubit, lower_offset)?;
                    let output_pauli = lower_patch.insert(output_global, bloq_circuit::Pauli::I);
                    if lower_patch != upper_patch
                        || output_pauli.is_some_and(|pauli| pauli != bloq_circuit::Pauli::X)
                    {
                        return Err(CompileError::ObservableSeamMismatch {
                            observable_index: obs,
                            lower: seam_pos(lower),
                            upper: seam_pos(upper),
                        });
                    }
                    if let Some(i) = out_idx {
                        insertions[i].operator_out =
                            output_pauli.map_or_else(PauliMap::empty, |_| {
                                PauliMap::from_unique_entries([(
                                    output_qubit,
                                    bloq_circuit::Pauli::X,
                                )])
                            });
                    }
                    if let Some(i) = in_idx {
                        insertions[i].operator_in = PauliMap::empty();
                    }
                    continue;
                }
                // The two faces of a seam meet at one xy patch, but a block
                // that moves in xy (a patch rotation, a walk) carries its
                // `operator_out` at a different local origin than its
                // `operator_in`. So equality is checked in the global frame:
                // each face is shifted by its own instance offset before
                // comparison. A stationary seam (equal offsets — the common
                // case) compares untranslated: translating both faces by one
                // shared offset is a coordinate bijection, so it cannot
                // change equality, and skipping it avoids two `PauliMap`
                // allocations per comparison in the heaviest lowering loop.
                let lower_out = out_idx.map(|i| &insertions[i].operator_out);
                let upper_in = in_idx.map(|i| &insertions[i].operator_in);
                let seam_cancels = match (lower_out, upper_in) {
                    (None, None) => true,
                    (Some(out), None) => out.is_empty(),
                    (None, Some(in_)) => in_.is_empty(),
                    (Some(out), Some(in_)) if offsets[&lower] == offsets[&upper] => out == in_,
                    (Some(out), Some(in_)) => {
                        out.try_translated(offsets[&lower])?
                            == in_.try_translated(offsets[&upper])?
                    }
                };
                if !seam_cancels {
                    return Err(CompileError::ObservableSeamMismatch {
                        observable_index: obs,
                        lower: seam_pos(lower),
                        upper: seam_pos(upper),
                    });
                }
                if let Some(i) = out_idx {
                    insertions[i].operator_out = PauliMap::empty();
                }
                if let Some(i) = in_idx {
                    insertions[i].operator_in = PauliMap::empty();
                }
            }
        }
    }
    Ok(())
}

fn resolve_observable_insertion(
    insertion: &ObservableChunkInsertion,
    context: &BloqLowerContext<'_>,
) -> Result<ResolvedObservable, CompileError> {
    let producer = context.site_node(insertion.site);
    let instance = context.instance_for_site(insertion.site);
    // Boundary operators are in block-local coordinates; shift them into the
    // instance-global frame so Phase 5 emission resolves them against the global
    // qubit layout (the same offset the instance's circuit qubits carry).
    let offset = instance.instance.offset;
    Ok(ResolvedObservable {
        index: insertion.observable_index,
        producer,
        instance: instance.instance.id,
        measurements: insertion
            .measurements
            .iter()
            .map(|&measurement| InstanceMeasurement {
                instance: instance.instance.id,
                measurement,
            })
            .collect(),
        operator_in: insertion.operator_in.try_translated(offset)?,
        operator_out: insertion.operator_out.try_translated(offset)?,
    })
}

fn resolve_pipe_observable_insertions(
    graph: &BlockGraph,
    plan: &super::LowerPlan,
    compiled: &CompiledTemplateMap,
    temporal_templates: &crate::FxMap<super::TemplatePlan, crate::block::LoweringTemplateId>,
    template_pool: &LoweringTemplatePool,
    generators: &[StabilizerGenerator],
) -> Result<Vec<ObservableChunkInsertion>, CompileError> {
    let mut insertions = Vec::new();
    for (obs_index, stabilizer) in generators.iter().enumerate() {
        // Mirror the block pass's ownership skip exactly, or a U6-owned
        // generator's realignment operator dangles one-sidedly at the H seam
        // (`ObservableSeamMismatch` — phase_gradient's rail).
        if !statically_resolved(stabilizer, compiled) {
            continue;
        }
        let mut hadamard_edges = stabilizer
            .stabilizer
            .interior_edges
            .iter()
            .filter_map(|(&(src, dst), &pauli)| {
                hadamard_temporal_pipe(graph, src, dst).map(|pipe| {
                    let pauli =
                        realignment_observable_pauli(graph, &pipe, &stabilizer.stabilizer, pauli);
                    (pipe, pauli)
                })
            })
            .collect::<Vec<_>>();
        hadamard_edges.sort_by_key(|(pipe, _)| (pipe.src.to_array(), pipe.dst.to_array()));

        for (pipe, pauli) in hadamard_edges {
            let Some(&plan_node) = plan.node_by_temporal_pipe().get(&pipe) else {
                continue;
            };
            let template_plan = plan.graph()[plan_node]
                .template
                .expect("temporal pipe node has a template");
            let template_id = temporal_templates[&template_plan];
            let template = template_pool
                .get(template_id)
                .expect("temporal pipe template exists");
            let local = LocalStabilizer::isolated(pauli);
            let entry = template.observable_gateway.lookup(local).ok_or_else(|| {
                CompileError::ObservableMissingGatewayEntry {
                    block_pos: pipe.src,
                    observable_index: obs_index as u32,
                    local_stabilizer: local.to_string(),
                }
            })?;
            // The pipe node's boundary operators ride a distinct operator-only
            // insertion, mirroring the block path: `cancel_seam_operators` matches
            // its `operator_in` against the source node and its `operator_out`
            // against the destination node (a temporal Hadamard sits as a node
            // between the two cubes). Built before the measurements below move out
            // of `entry`.
            let (lower_endpoint, upper_endpoint) = pipe.endpoints_by_z();
            let endpoint_provenance = |endpoint: IVec3| {
                let selective = graph
                    .get_endpoint_block(endpoint)
                    .and_then(|owner| compiled.get(&owner.pos()))
                    .is_some_and(|compiled| compiled.template.is_selective());
                if selective {
                    FaceProvenance::Dynamic
                } else {
                    FaceProvenance::Static
                }
            };
            insertions.extend(ObservableChunkInsertion::operators(
                ChunkSiteSource::TemporalPipe(pipe),
                obs_index as u32,
                entry.operator_in.clone(),
                entry.operator_out.clone(),
                endpoint_provenance(lower_endpoint),
                endpoint_provenance(upper_endpoint),
            ));
            for chunk_meas in entry.measurements {
                insertions.push(ObservableChunkInsertion::measurements(
                    ChunkSiteSource::TemporalPipe(pipe),
                    obs_index as u32,
                    chunk_meas.measurements,
                ));
            }
        }
    }
    Ok(insertions)
}

/// Route logical crossings through the spatial Hadamard walls a row touches.
///
/// The spatial analogue of [`resolve_pipe_observable_insertions`], and
/// deliberately the smaller of the two: a wall has no plan node and no temporal
/// face, so there are no boundary operators to emit or cancel — only the records
/// that carry the crossing string past the cubes' seam-column resets.
///
/// The query key names both sides of the seam in the pipe's own frame (`XMINUS`
/// the negative-axis cube, `XPLUS` the positive-axis cube), and reads each Pauli
/// through [`incident_edge_pauli`], which already delivers it in that cube's
/// frame — the far side reads the stored edge Pauli flipped across the
/// Hadamard. The wall only ever inserts `P` against `P.flip()`, so a row whose
/// two sides are somehow not flipped misses the gateway and raises
/// [`CompileError::ObservableMissingGatewayEntry`] instead of silently routing
/// a crossing the wall never meant to carry.
pub(super) fn push_wall_observable_insertions(
    graph: &BlockGraph,
    wall_templates: &crate::FxMap<super::SpatialPipeRef, crate::block::LoweringTemplateId>,
    template_pool: &LoweringTemplatePool,
    stabilizer: &impl SurfaceSupport,
    positions: Option<&[IVec3]>,
    observable_index: u32,
    insertions: &mut Vec<ObservableChunkInsertion>,
) -> Result<bool, CompileError> {
    let mut walls: Vec<super::SpatialPipeRef> = stabilizer
        .owned_edges(positions)
        .filter(|(_, pauli)| *pauli != Pauli::I)
        .filter_map(|((src, dst), _)| spatial_hadamard_wall(graph, src, dst))
        .collect();
    walls.sort_unstable_by_key(|wall| (wall.src.to_array(), wall.dst.to_array()));
    walls.dedup();

    let mut sign = false;
    for wall in walls {
        let template_id = *wall_templates
            .get(&wall)
            .expect("spatial Hadamard walls are compiled before lowering");
        let gateway = &template_pool
            .get(template_id)
            .expect("spatial Hadamard wall template exists")
            .observable_gateway;
        let (minus_dir, plus_dir) = wall.arm_directions();
        let arm = |pos: IVec3, dir: Direction| {
            incident_edge_pauli(graph, pos, dir, stabilizer, true).unwrap_or(Pauli::I)
        };
        let minus_pauli = arm(wall.src, plus_dir);
        let plus_pauli = arm(wall.dst, minus_dir);
        let y_crossing = minus_pauli == Pauli::Y && plus_pauli == Pauli::Y;
        let key = LocalStabilizer::isolated(Pauli::I)
            .with_arm(minus_dir, minus_pauli, false)
            .with_arm(plus_dir, plus_pauli, false);
        let crossing = |pauli| {
            LocalStabilizer::isolated(Pauli::I)
                .with_arm(minus_dir, pauli, false)
                .with_arm(plus_dir, pauli.flip(), false)
        };
        let entry = gateway.lookup(key).or_else(|| {
            if !y_crossing {
                return None;
            }
            let x = gateway.lookup(crossing(Pauli::X))?;
            let z = gateway.lookup(crossing(Pauli::Z))?;
            Some(GatewayEntry::xor([&x, &z]))
        });
        let entry = entry.ok_or_else(|| CompileError::ObservableMissingGatewayEntry {
            block_pos: wall.src,
            observable_index,
            local_stabilizer: key.to_string(),
        })?;
        sign ^= y_crossing;
        for chunk_meas in entry.measurements {
            insertions.push(ObservableChunkInsertion::measurements(
                ChunkSiteSource::SpatialPipe(wall),
                observable_index,
                chunk_meas.measurements,
            ));
        }
    }
    Ok(sign)
}

/// The wall standing on the pipe between `src` and `dst`, if that pipe is a
/// spatial Hadamard.
fn spatial_hadamard_wall(
    graph: &BlockGraph,
    src: IVec3,
    dst: IVec3,
) -> Option<super::SpatialPipeRef> {
    let pipe = graph.get_pipe(src, dst)?;
    (pipe.is_hadamard() && pipe.dir().is_spatial()).then(|| {
        let (src, dst) = pipe.endpoints();
        super::SpatialPipeRef::new(src, dst)
    })
}

pub(super) fn realignment_observable_pauli(
    graph: &BlockGraph,
    pipe: &super::TemporalPipeRef,
    stabilizer: &impl SurfaceSupport,
    edge_pauli: Pauli,
) -> Pauli {
    let (lower_endpoint, _) = pipe.endpoints_by_z();
    let lower_owner = graph
        .get_endpoint_block(lower_endpoint)
        .map_or(lower_endpoint, bloq_graph::Block::pos);
    // The seam carries the lower cube's +Z *arm*, read in that cube's frame
    // (`incident_edge_pauli` H-conjugates the stored edge Pauli for us). On a
    // plain worldline that equals the cube's center Pauli, but at a junction the
    // center is the product across several faces — the toffoli's merge cube
    // holds `Y` while only `X` crosses upward — and keying on the center would
    // ask the pipe for a face operator the block never emitted
    // (`ObservableSeamMismatch`). The center Pauli stays as the fallback for a
    // lower endpoint with no arm support of its own.
    let arm = incident_edge_pauli(graph, lower_owner, Direction::ZPLUS, stabilizer, true)
        .unwrap_or(Pauli::I);
    (arm != Pauli::I)
        .then_some(arm)
        .or_else(|| {
            let center = stabilizer.node_pauli(lower_owner);
            (center != Pauli::I).then_some(center)
        })
        .unwrap_or(edge_pauli)
}

pub(super) fn hadamard_temporal_pipe(
    graph: &BlockGraph,
    src: IVec3,
    dst: IVec3,
) -> Option<super::TemporalPipeRef> {
    let pipe = graph
        .get_pipe(src, dst)
        .or_else(|| graph.get_pipe_between_blocks(src, dst))?;
    (pipe.is_hadamard() && !pipe.dir().is_spatial()).then(|| {
        // Must match `plan::temporal_pipe_ref`'s `src.z <= dst.z` orientation:
        // this ref keys `node_by_temporal_pipe` / pipe-template lookups.
        let (src, dst) = pipe.endpoints();
        let (src, dst) = if src.z <= dst.z {
            (src, dst)
        } else {
            (dst, src)
        };
        super::TemporalPipeRef {
            src,
            dst,
            hadamard: true,
        }
    })
}

/// Whether the pinned proxy can resolve this public row through static gateways.
/// Selective-internal rows and deferred readout plans have no static observable;
/// every selective in a retained row's support must already have a fixed template.
fn statically_resolved(
    stabilizer: &StabilizerGenerator,
    compiled_blocks: &CompiledTemplateMap,
) -> bool {
    if !stabilizer.kind.is_readout() || stabilizer.readout_plan().is_some() {
        return false;
    }
    !stabilizer
        .stabilizer
        .interior_nodes
        .iter()
        .any(|(pos, pauli)| {
            *pauli != Pauli::I
                && compiled_blocks
                    .get(pos)
                    .is_some_and(|compiled| compiled.template.is_selective())
        })
}

/// One operator-only insertion carrying a block's boundary operators, with
/// each temporal face's provenance resolved.
fn boundary_operator_insertion(
    graph: &BlockGraph,
    compiled_blocks: &CompiledTemplateMap,
    connectivity: Connectivity,
    node_pos: IVec3,
    obs_index: u32,
    operator_in: PauliMap,
    operator_out: PauliMap,
) -> Option<ObservableChunkInsertion> {
    ObservableChunkInsertion::operators(
        ChunkSiteSource::Block(node_pos),
        obs_index,
        operator_in,
        operator_out,
        face_provenance(
            graph,
            compiled_blocks,
            connectivity,
            node_pos,
            Direction::ZMINUS,
        ),
        face_provenance(
            graph,
            compiled_blocks,
            connectivity,
            node_pos,
            Direction::ZPLUS,
        ),
    )
}

/// Provenance of the block-at-`pos`'s seam face toward `dir`: `Dynamic` when
/// the seam counterparty is a selective block, whose content is
/// runtime-resolved and emits no static insertion to cancel against.
/// A temporal-Hadamard pipe is its own plan node, so a face behind one always
/// meets the static realignment pipe, never the far block. A face without a
/// temporal pipe never joins a seam and stays `Static` (e.g. a T escape's
/// `operator_in`, which must survive and materialize as a boundary fragment).
fn face_provenance(
    graph: &BlockGraph,
    compiled_blocks: &CompiledTemplateMap,
    connectivity: Connectivity,
    pos: IVec3,
    dir: Direction,
) -> FaceProvenance {
    if !connectivity.has_pipe(dir) || connectivity.has_hadamard(dir) {
        return FaceProvenance::Static;
    }
    let selective = graph
        .get_block(pos)
        .and_then(|block| {
            checked_add_position(block.endpoint_for_direction(dir), dir.to_ivec3()).ok()
        })
        .and_then(|neighbor| graph.get_endpoint_block(neighbor))
        .and_then(|neighbor| compiled_blocks.get(&neighbor.pos()))
        .is_some_and(|compiled| compiled.template.is_selective());
    if selective {
        FaceProvenance::Dynamic
    } else {
        FaceProvenance::Static
    }
}

/// A source pipe must own the connection. A compiler-only spatial Port cut
/// cannot lead to an unrelated real block in the adjacent temporal cell.
pub(crate) fn physical_neighbor<'a>(
    graph: &'a BlockGraph,
    block: &Block,
    connectivity: Connectivity,
    direction: Direction,
) -> Option<(&'a Block, bool)> {
    if !connectivity.has_pipe(direction) {
        return None;
    }
    let endpoint = block.endpoint_for_direction(direction);
    let neighbor_endpoint = checked_add_position(endpoint, direction.to_ivec3()).ok()?;
    graph.get_pipe(endpoint, neighbor_endpoint)?;
    graph
        .get_endpoint_block(neighbor_endpoint)
        .map(|neighbor| (neighbor, connectivity.has_hadamard(direction)))
}

/// Whether the pipe-connected patch containing the block at `start` is
/// anchored in dynamic content: some patch member, or a piped neighbor of one,
/// is a selective or T block. The dynamic anchor need not touch the block
/// being resolved — in t_with_prepared_y the armless key sits two spatial
/// merges from both the T and the runtime-measured patch, and in the CCZ
/// factories the magic-state worldline runs two or more *temporal* steps above
/// its T block — so the walk follows spatial and temporal pipes alike (dynamic
/// content flows along the worldline). The anchor need not appear in the
/// generator's row at all (in t.blog the absorbed segment's pipes carry `I`).
/// Unlike [`face_provenance`], block kinds come from the *block graph*, so the
/// answer is invariant under branch-assignment `Fixed` pinning of the compiled
/// map, and T counts: a T's magic state is a legitimate dynamic anchor even
/// though its +Z seam is a static contract.
pub(crate) fn patch_touches_dynamic_block(
    graph: &BlockGraph,
    compiled_blocks: &CompiledTemplateMap,
    start: IVec3,
) -> bool {
    let mut seen = crate::FxSet::default();
    seen.insert(start);
    let mut stack = vec![start];
    while let Some(pos) = stack.pop() {
        let Some(block) = graph.get_block(pos) else {
            continue;
        };
        if block.kind().is_dynamic() {
            return true;
        }
        let Some(compiled) = compiled_blocks.get(&pos) else {
            continue;
        };
        for dir in Direction::iter() {
            let Some((neighbor, _)) =
                physical_neighbor(graph, block, compiled.graph_connectivity, dir)
            else {
                continue;
            };
            if neighbor.kind().is_dynamic() {
                return true;
            }
            if seen.insert(neighbor.pos()) {
                stack.push(neighbor.pos());
            }
        }
    }
    false
}

/// The T block sourcing the escaped-patch worldline that contains `start`.
///
/// An armless crossing on a memory cube above a T carries the T byproduct, but
/// the records live on the T's escape instance, not the cube. The escaped patch
/// flows `+Z` from the T source, so this descends the `-Z` worldline until it
/// reaches the T, tracking whether an odd number of Hadamard pipes were crossed
/// (each flips the escaped patch's `X ↔ Z`, so the caller must query the escape
/// gateway in the flipped basis). Returns `None` when the worldline has no T
/// below — a purely selective-anchored segment, left to the branch-conditioned
/// lowering (U6).
pub(crate) fn source_t_block(
    graph: &BlockGraph,
    compiled_blocks: &CompiledTemplateMap,
    start: IVec3,
) -> Option<(IVec3, bool)> {
    let mut pos = start;
    let mut flipped = false;
    let mut seen = crate::FxSet::default();
    while seen.insert(pos) {
        let block = graph.get_block(pos)?;
        if block.kind().is_t() {
            return Some((pos, flipped));
        }
        let connectivity = compiled_blocks.get(&pos)?.graph_connectivity;
        let (below, hadamard) = physical_neighbor(graph, block, connectivity, Direction::ZMINUS)?;
        flipped ^= hadamard;
        pos = below.pos();
    }
    None
}

/// Convert algebraic stabilizer support into the physical local-flow shape
/// consumed by a block's observable gateway.
///
/// `Stabilizer::phase_free_product` may cancel a node Pauli while leaving edge
/// support. Adding each arm through `LocalStabilizer::with_arm` reconstructs
/// those flowing Pauli components at the block junction without mutating the
/// algebraic stabilizer.
pub(super) fn build_local_flow_key(
    graph: &BlockGraph,
    node_pos: IVec3,
    algebraic_center: Pauli,
    stabilizer: &impl SurfaceSupport,
    block_connectivity: Connectivity,
    spatial_port_dir: Option<Direction>,
) -> LocalStabilizer {
    let mut local = LocalStabilizer::isolated(algebraic_center);
    for dir in Direction::iter() {
        if !block_connectivity.has_pipe(dir) {
            continue;
        }
        let pauli = incident_edge_pauli(
            graph,
            node_pos,
            dir,
            stabilizer,
            block_connectivity.has_hadamard(dir),
        )
        .unwrap_or(Pauli::I);
        if pauli != Pauli::I {
            // Gateway keys never carry Hadamard flags (`gateway_key_connectivity`):
            // `incident_edge_pauli` already delivered the arm Pauli in this
            // block's frame, and the realignment pipe node owns the X↔Z flip.
            local = local.with_arm(dir, pauli, false);
        }
    }
    if let Some(dir) = spatial_port_dir {
        let pauli = stabilizer.port_pauli(node_pos);
        if pauli != Pauli::I {
            debug_assert!(block_connectivity.has_pipe(dir));
            local = local.with_arm(dir, pauli, false);
        }
    }
    local
}

pub(super) fn needs_dynamic_anchor(connectivity: Connectivity, flow: LocalStabilizer) -> bool {
    flow != LocalStabilizer::isolated(Pauli::I)
        && !connectivity.is_isolated()
        && Direction::iter().all(|dir| !flow.has_arm(dir))
}

pub(crate) fn incident_edge_positions(
    graph: &BlockGraph,
    node_pos: IVec3,
    dir: Direction,
) -> Option<(IVec3, IVec3)> {
    let block = graph.get_block(node_pos)?;
    let endpoint = block.endpoint_for_direction(dir);
    let neighbor_endpoint = checked_add_position(endpoint, dir.to_ivec3()).ok()?;
    let neighbor_pos = graph
        .get_endpoint_block(neighbor_endpoint)
        .map_or(neighbor_endpoint, bloq_graph::Block::pos);
    Some((node_pos, neighbor_pos))
}

fn incident_edge_pauli(
    graph: &BlockGraph,
    node_pos: IVec3,
    dir: Direction,
    stabilizer: &impl SurfaceSupport,
    has_hadamard: bool,
) -> Option<Pauli> {
    let (node_pos, neighbor_pos) = incident_edge_positions(graph, node_pos, dir)?;

    if let Some(pauli) = stabilizer.edge_pauli(node_pos, neighbor_pos) {
        // `pauli_string_to_stabilizer` stores the edge Pauli on the first
        // endpoint in the tuple, which corresponds to the smaller ZX-node side.
        return Some(pauli);
    }
    if let Some(pauli_at_small_side) = stabilizer.edge_pauli(neighbor_pos, node_pos) {
        return Some(if has_hadamard {
            pauli_at_small_side.flip()
        } else {
            pauli_at_small_side
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use bloq_graph::{Basis, BlockKind, CubeKind, PatchRotationKind, Pipe};
    use glam::{ivec2, ivec3};

    use super::*;
    use crate::block::compile_fixed_bulk;
    use crate::compile::CompiledTemplateInfo;
    use crate::signature::BlockSignature;

    #[test]
    fn multi_site_observable_merges_into_one_recipe() {
        use bloq_ir::ClassicalNode;

        // One complete row combines records from both sites in stable order.
        let mut bloq = Bloq::new();
        let producer_a = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: bloq_ir::ClassicalExpr::Const(false),
        }));
        let producer_b = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: bloq_ir::ClassicalExpr::Const(false),
        }));
        let node_map = vec![Some(producer_a), Some(producer_b)];

        let insertion = |producer_slot: usize, measurement: u32| ResolvedObservable {
            index: 0,
            producer: NodeIndex::new(producer_slot),
            instance: TemplateInstanceId(0),
            measurements: vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
            operator_in: PauliMap::empty(),
            operator_out: PauliMap::empty(),
        };
        let resolved = vec![insertion(0, 1), insertion(1, 0)];

        let _materialized = materialize_observable_nodes(resolved, &[], &mut bloq, &node_map);

        let records: Vec<_> = bloq
            .nodes()
            .filter_map(|(_, node)| match node.try_classical() {
                Some(ClassicalNode::Observable {
                    index: Some(0),
                    measurements,
                    ..
                }) => Some(
                    measurements
                        .iter()
                        .map(|measurement| measurement.measurement)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .collect();
        assert_eq!(
            records,
            vec![vec![0, 1]],
            "both sites merge into one ordered observable"
        );
    }

    #[test]
    fn hadamard_edge_belongs_only_to_smaller_local_stabilizer() {
        let lower = ivec3(0, 0, 0);
        let upper = ivec3(0, 0, 1);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(lower, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(lower, Direction::ZPLUS).with_hadamard());
        let stabilizer = Stabilizer::from_interior_nodes([(lower, Pauli::Z), (upper, Pauli::X)])
            .with_interior_edges([((lower, upper), Pauli::X)]);

        let lower_local = build_local_flow_key(
            &graph,
            lower,
            Pauli::Z,
            &stabilizer,
            Connectivity::ISOLATED.with_hadamard(Direction::ZPLUS),
            None,
        );
        let upper_local = build_local_flow_key(
            &graph,
            upper,
            Pauli::X,
            &stabilizer,
            Connectivity::ISOLATED.with_hadamard(Direction::ZMINUS),
            None,
        );

        // The stored edge Pauli belongs to the smaller (lower) side; the upper
        // side reads it flipped across the Hadamard edge — arm Paulis are
        // frame-local, so the keys themselves carry no Hadamard flags.
        assert_eq!(lower_local.arm_basis(Direction::ZPLUS), Pauli::X);
        assert_eq!(upper_local.arm_basis(Direction::ZMINUS), Pauli::Z);
        assert!(!lower_local.has_hadamard(Direction::ZPLUS));
        assert!(!upper_local.has_hadamard(Direction::ZMINUS));
    }

    #[test]
    fn projection_observable_sign_tracks_y_temporal_hadamard_crossing() {
        let lower = ivec3(0, 0, 0);
        let upper = ivec3(0, 0, 1);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(lower, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(lower, Direction::ZPLUS).with_hadamard());
        let stabilizer = Stabilizer::from_interior_nodes([(lower, Pauli::Y), (upper, Pauli::Y)])
            .with_interior_edges([((lower, upper), Pauli::Y)]);

        assert!(temporal_hadamard_observable_sign(&graph, &stabilizer));
    }

    #[test]
    fn boundary_local_stabilizer_ignores_unconnected_out_of_range_neighbors() {
        let position = ivec3(i32::MAX, 0, 0);
        let mut graph = BlockGraph::new();
        graph
            .try_add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)))
            .unwrap();
        let stabilizer = Stabilizer::from_interior_nodes([(position, Pauli::Z)]);

        let local = build_local_flow_key(
            &graph,
            position,
            Pauli::Z,
            &stabilizer,
            Connectivity::ISOLATED,
            None,
        );

        for direction in Direction::iter() {
            assert_eq!(local.arm_basis(direction), Pauli::I);
        }
    }

    #[test]
    fn shared_seam_checker_rejects_discontinuous_branch_content() {
        let lower = ivec3(0, 0, 0);
        let upper = ivec3(0, 0, 1);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(lower, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(lower, Direction::ZPLUS));
        let plan = super::super::LowerPlan::from_block_graph(&graph);
        let pauli = |pauli| PauliMap::from_unique_entries([(glam::ivec2(0, 0), pauli)]);
        let mut insertions = vec![
            ObservableChunkInsertion {
                site: super::super::ChunkSiteSource::Block(lower),
                observable_index: 7,
                measurements: Vec::new(),
                operator_in: PauliMap::empty(),
                operator_out: pauli(bloq_circuit::Pauli::X),
                provenance_in: FaceProvenance::Static,
                provenance_out: FaceProvenance::Static,
            },
            ObservableChunkInsertion {
                site: super::super::ChunkSiteSource::Block(upper),
                observable_index: 7,
                measurements: Vec::new(),
                operator_in: pauli(bloq_circuit::Pauli::Z),
                operator_out: PauliMap::empty(),
                provenance_in: FaceProvenance::Static,
                provenance_out: FaceProvenance::Static,
            },
        ];

        let error = cancel_seam_operators_in_plan(&mut insertions, &plan, &graph, |_| {
            Ok(glam::IVec2::ZERO)
        })
        .unwrap_err();

        assert!(matches!(
            error,
            CompileError::ObservableSeamMismatch {
                observable_index: 7,
                lower: error_lower,
                upper: error_upper,
            } if error_lower == lower && error_upper == upper
        ));
    }

    #[test]
    fn scaled_cube_top_edge_maps_to_owner_node_stabilizer_edge() {
        let scaled = ivec3(0, 0, 0);
        let upper = ivec3(0, 0, 2);
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(scaled, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 1), Direction::ZPLUS));
        let stabilizer = Stabilizer::from_interior_nodes([(scaled, Pauli::Z), (upper, Pauli::Z)])
            .with_interior_edges([((scaled, upper), Pauli::X)]);

        let local = build_local_flow_key(
            &graph,
            scaled,
            Pauli::Z,
            &stabilizer,
            Connectivity::ISOLATED.with_pipe(Direction::ZPLUS),
            None,
        );

        assert_eq!(local.arm_basis(Direction::ZPLUS), Pauli::X);
    }

    #[test]
    fn patch_rotation_gateway_resolves_observable_insertions() {
        let block_pos = ivec3(0, 0, 0);
        let kind = PatchRotationKind::new(Basis::X, ivec2(1, 0)).unwrap();
        let end_pos = kind.end_position(block_pos);
        let connectivity = Connectivity::ISOLATED
            .with_pipe(Direction::ZMINUS)
            .with_pipe(Direction::ZPLUS);
        let template = compile_fixed_bulk(
            BlockSignature {
                kind: BlockKind::PatchRotation(kind),
                rounds: None,
                connectivity,
                boundary_basis: None,
                layer_schedule: None,
                surgery_side: None,
            },
            3,
        )
        .expect("patch rotation compiles");
        let mut template_pool = LoweringTemplatePool::default();
        let template = template_pool.insert(template);
        let compiled_blocks = CompiledTemplateMap::from_iter([(
            block_pos,
            CompiledTemplateInfo {
                graph_connectivity: connectivity,
                template: crate::compile::TemplateRef::Fixed(template),
            },
        )]);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(block_pos, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(
            block_pos + Direction::ZMINUS.to_ivec3(),
            BlockKind::Port,
        ));
        graph.add_block(Block::new(
            end_pos + Direction::ZPLUS.to_ivec3(),
            BlockKind::Port,
        ));
        graph.add_pipe(Pipe::new(block_pos, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(end_pos, Direction::ZPLUS));
        let stabilizers = StabilizerGenerators::new(
            bloq_graph::ZXGraph::try_from(&bloq_graph::BlockGraph::new()).expect("empty ZX graph"),
            vec![StabilizerGenerator::new(
                Stabilizer::from_interior_nodes([(block_pos, Pauli::X)]).with_interior_edges([
                    (
                        (block_pos, block_pos + Direction::ZMINUS.to_ivec3()),
                        Pauli::X,
                    ),
                    ((block_pos, end_pos + Direction::ZPLUS.to_ivec3()), Pauli::Z),
                ]),
                StabilizerRowKind::Logical,
            )],
        );

        let insertions = resolve_observable_insertions(
            &graph,
            &stabilizers.generators,
            &compiled_blocks,
            &template_pool,
        )
        .expect("patch rotation gateway resolves mixed local observable");

        assert!(!insertions.is_empty());
        // Every insertion targets the patch-rotation block and carries something —
        // either measurement records or a boundary operator (the latter rides a
        // distinct operator-only insertion with no measurements).
        assert!(insertions.iter().all(|insertion| {
            matches!(insertion.site, ChunkSiteSource::Block(pos) if pos == block_pos)
                && (!insertion.measurements.is_empty()
                    || !insertion.operator_in.is_empty()
                    || !insertion.operator_out.is_empty())
        }));
        // The rotation now contributes its boundary operators (the support-move
        // case), so at least one insertion carries a non-empty temporal face.
        assert!(insertions.iter().any(|insertion| {
            !insertion.operator_in.is_empty() || !insertion.operator_out.is_empty()
        }));
        // An all-Fixed compiled map stamps every face static — this is also why
        // the U6 branch path (which pins selectives to `Fixed`) is unaffected
        // by provenance stamping.
        assert!(insertions.iter().all(|insertion| {
            insertion.provenance_in == FaceProvenance::Static
                && insertion.provenance_out == FaceProvenance::Static
        }));
    }

    /// A piped memory cube whose generator row carries `I` on every pipe
    /// (armless key) in an all-static patch: the absorbed-segment drop must NOT
    /// fire, and the gateway lookup raises `ObservableMissingGatewayEntry`
    /// (the invariant the old unconditional drop left unenforced).
    #[test]
    fn armless_center_on_piped_block_without_dynamic_anchor_errors() {
        let (graph, compiled_blocks, template_pool, cube) =
            piped_cube_fixture(BlockKind::Cube(CubeKind::ZXZ));

        let generators = vec![StabilizerGenerator::new(
            Stabilizer::from_interior_nodes([(cube, Pauli::X)]),
            StabilizerRowKind::Logical,
        )];

        let result =
            resolve_observable_insertions(&graph, &generators, &compiled_blocks, &template_pool);
        assert!(matches!(
            result,
            Err(CompileError::ObservableMissingGatewayEntry { block_pos, .. })
                if block_pos == cube
        ));
    }

    /// Same armless key, but the patch is anchored in dynamic content (a T
    /// neighbor): the row is an absorbed dynamic segment and drops silently.
    #[test]
    fn armless_center_next_to_dynamic_block_absorbed_silently() {
        let (graph, compiled_blocks, template_pool, cube) = piped_cube_fixture(BlockKind::T);

        let generators = vec![StabilizerGenerator::new(
            Stabilizer::from_interior_nodes([(cube, Pauli::X)]),
            StabilizerRowKind::Logical,
        )];

        let insertions =
            resolve_observable_insertions(&graph, &generators, &compiled_blocks, &template_pool)
                .expect("absorbed dynamic segment drops silently");
        assert!(insertions.is_empty());
    }

    #[test]
    fn bounded_gateway_binds_distant_t_records_and_their_hadamard_frame() {
        use crate::block::ObservableGateway;
        use crate::block::gateway::ChunkMeasurements;
        use crate::lower::{BoundDynamicAnchor, LocalSurfaceContext, TSourceGateway};

        let center = IVec3::ZERO;
        let connectivity = Connectivity::ISOLATED
            .with_pipe(Direction::ZMINUS)
            .with_pipe(Direction::ZPLUS);
        let template = compile_fixed_bulk(
            BlockSignature {
                kind: BlockKind::Cube(CubeKind::ZXZ),
                rounds: Some(3),
                connectivity,
                boundary_basis: None,
                layer_schedule: Some(crate::signature::LayerSchedule::Compact),
                surgery_side: None,
            },
            3,
        )
        .unwrap();
        let mut pool = LoweringTemplatePool::default();
        let template = pool.insert(template);
        let compiled = CompiledTemplateMap::from_iter([(
            center,
            CompiledTemplateInfo {
                graph_connectivity: connectivity,
                template: crate::compile::TemplateRef::Fixed(template),
            },
        )]);
        let mut graph = BlockGraph::new();
        for z in -1..=1 {
            graph.add_block(Block::new(z * IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        }
        graph.add_pipe(Pipe::new(-IVec3::Z, Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        let mut t_gateway = ObservableGateway::new();
        for (pauli, measurements) in [(Pauli::X, vec![7, 9]), (Pauli::Z, vec![11])] {
            t_gateway.insert(
                LocalStabilizer::isolated(pauli),
                GatewayEntry {
                    measurements: vec![ChunkMeasurements {
                        chunk_index: 0,
                        measurements,
                    }],
                    ..GatewayEntry::default()
                },
            );
        }
        let context = LocalSurfaceContext {
            graph: &graph,
            compiled: &compiled,
            spatial_ports: &Default::default(),
            spatial_templates: &Default::default(),
            wall_templates: &Default::default(),
            pipe_templates: &Default::default(),
            pool: &pool,
            instances: &Default::default(),
            selectives: &Default::default(),
            layout: crate::BlockLayout::new(3),
        };
        let surface = Stabilizer::from_interior_nodes([(center, Pauli::X)]);
        assert!(context.needs_dynamic_anchor(center, &surface));
        let source = -8 * IVec3::Z;
        assert!(graph.get_block(source).is_none());
        assert!(!compiled.contains_key(&source));
        for (flipped, expected) in [(false, vec![7, 9]), (true, vec![11])] {
            let (records, boundaries, sign) = context
                .resolve(
                    &[center],
                    &surface,
                    None,
                    &StabilizerRowKind::Logical,
                    0,
                    Some(BoundDynamicAnchor {
                        touches_dynamic: true,
                        source_t: Some(TSourceGateway {
                            source,
                            flipped,
                            gateway: &t_gateway,
                            instance: TemplateInstanceId(42),
                        }),
                    }),
                )
                .unwrap();
            assert_eq!(
                records,
                expected
                    .into_iter()
                    .map(|measurement| InstanceMeasurement {
                        instance: TemplateInstanceId(42),
                        measurement,
                    })
                    .collect::<Vec<_>>()
            );
            assert!(boundaries.is_empty());
            assert!(!sign);
        }
    }

    /// A cube whose +Z seam meets a selective block: its operator insertion is
    /// stamped `Dynamic` on the out face and `Static` on the in face, so
    /// `cancel_seam_operators` terminates instead of comparing — without any
    /// neighbor probe at seam-walk time.
    #[test]
    fn operator_face_into_selective_stamped_dynamic() {
        use bloq_graph::SelectiveKind;

        let cube = ivec3(0, 0, 0);
        let selective = ivec3(0, 0, 1);
        let connectivity = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        let template = compile_fixed_bulk(
            BlockSignature {
                kind: BlockKind::Cube(CubeKind::ZXZ),
                rounds: Some(3),
                connectivity,
                boundary_basis: None,
                layer_schedule: Some(crate::signature::LayerSchedule::Compact),
                surgery_side: None,
            },
            3,
        )
        .expect("piped cube compiles");
        let mut template_pool = LoweringTemplatePool::default();
        let template = template_pool.insert(template);
        let compiled_blocks = CompiledTemplateMap::from_iter([
            (
                cube,
                CompiledTemplateInfo {
                    graph_connectivity: connectivity,
                    template: crate::compile::TemplateRef::Fixed(template),
                },
            ),
            (
                selective,
                CompiledTemplateInfo {
                    graph_connectivity: Connectivity::ISOLATED.with_pipe(Direction::ZMINUS),
                    template: crate::compile::TemplateRef::Selective {
                        when_true: template,
                        when_false: template,
                    },
                },
            ),
        ]);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(cube, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            selective,
            BlockKind::Selective(SelectiveKind::XZ),
        ));
        graph.add_pipe(Pipe::new(cube, Direction::ZPLUS));

        let generators = vec![StabilizerGenerator::new(
            Stabilizer::from_interior_nodes([(cube, Pauli::Z)])
                .with_interior_edges([((cube, selective), Pauli::Z)]),
            StabilizerRowKind::Measurement {
                name: "m".to_string(),
            },
        )];

        let insertions =
            resolve_observable_insertions(&graph, &generators, &compiled_blocks, &template_pool)
                .expect("measurement row resolves through the cube gateway");

        let operator = insertions
            .iter()
            .find(|insertion| {
                !insertion.operator_in.is_empty() || !insertion.operator_out.is_empty()
            })
            .expect("cube emits a boundary-operator insertion toward the seam");
        assert_eq!(operator.provenance_out, FaceProvenance::Dynamic);
        assert_eq!(operator.provenance_in, FaceProvenance::Static);
    }

    /// A cube piped +Z to `neighbor_kind`, compiled and mapped; the neighbor
    /// deliberately has no compiled entry (it is outside every generator's
    /// support in these tests).
    fn piped_cube_fixture(
        neighbor_kind: BlockKind,
    ) -> (BlockGraph, CompiledTemplateMap, LoweringTemplatePool, IVec3) {
        let cube = ivec3(0, 0, 0);
        let neighbor = ivec3(0, 0, 1);
        let connectivity = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        let template = compile_fixed_bulk(
            BlockSignature {
                kind: BlockKind::Cube(CubeKind::ZXZ),
                rounds: Some(3),
                connectivity,
                boundary_basis: None,
                layer_schedule: Some(crate::signature::LayerSchedule::Compact),
                surgery_side: None,
            },
            3,
        )
        .expect("piped cube compiles");
        let mut template_pool = LoweringTemplatePool::default();
        let template = template_pool.insert(template);
        let compiled_blocks = CompiledTemplateMap::from_iter([(
            cube,
            CompiledTemplateInfo {
                graph_connectivity: connectivity,
                template: crate::compile::TemplateRef::Fixed(template),
            },
        )]);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(cube, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(neighbor, neighbor_kind));
        graph.add_pipe(Pipe::new(cube, Direction::ZPLUS));
        (graph, compiled_blocks, template_pool, cube)
    }
}
