//! Bind a local correlation surface to physical records and boundary operators.

use bloq_graph::{BlockGraph, Pauli, StabilizerRowKind, SurfaceSupport};
use bloq_ir::lowering::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstanceId};
use bloq_ir::{BoundaryFace, TemporalPipeRef};
use glam::IVec3;

use super::observable::{
    ObservableChunkInsertion, hadamard_temporal_pipe, realignment_observable_pauli,
};
use super::{ChunkSiteSource, SelectiveArmInstances, SpatialPipeRef};
use crate::block::{LocalStabilizer, LoweringTemplateId, LoweringTemplatePool, ObservableGateway};
use crate::compile::{CompiledTemplateMap, TemplateRef};
use crate::spatial_port::SpatialPortExpansionMap;
use crate::{BlockLayout, CompileError, xor_toggle};

/// A distant T escape owns these records even when it is outside local geometry.
#[derive(Clone, Copy)]
pub(crate) struct TSourceGateway<'a> {
    pub source: IVec3,
    pub flipped: bool,
    pub gateway: &'a ObservableGateway,
    pub instance: TemplateInstanceId,
}

/// Exact selected global facts used by an armless local gateway.
#[derive(Clone, Copy)]
pub(crate) struct BoundDynamicAnchor<'a> {
    pub touches_dynamic: bool,
    pub source_t: Option<TSourceGateway<'a>>,
}

pub(crate) struct LocalSurfaceContext<'a> {
    pub graph: &'a BlockGraph,
    pub compiled: &'a CompiledTemplateMap,
    pub spatial_ports: &'a SpatialPortExpansionMap,
    pub spatial_templates: &'a CompiledTemplateMap,
    pub wall_templates: &'a crate::FxMap<SpatialPipeRef, LoweringTemplateId>,
    pub pipe_templates: &'a crate::FxMap<bloq_ir::TemporalPipeRef, LoweringTemplateId>,
    pub pool: &'a LoweringTemplatePool,
    pub instances: &'a crate::FxMap<ChunkSiteSource, TemplateInstanceId>,
    pub selectives: &'a crate::FxMap<IVec3, SelectiveArmInstances>,
    pub layout: BlockLayout,
}

impl LocalSurfaceContext<'_> {
    /// Only armless nonzero flow on a piped block needs global anchor facts.
    pub(crate) fn needs_dynamic_anchor(
        &self,
        position: IVec3,
        surface: &impl SurfaceSupport,
    ) -> bool {
        let Some(compiled) = self.compiled.get(&position) else {
            return false;
        };
        let flow = super::observable::build_local_flow_key(
            self.graph,
            position,
            surface.node_pauli(position),
            surface,
            compiled.graph_connectivity,
            self.spatial_ports
                .get(&position)
                .map(|port| port.cube_pipe_dir()),
        );
        super::observable::needs_dynamic_anchor(compiled.graph_connectivity, flow)
    }

    /// Bounded geometry must supply an anchor when `needs_dynamic_anchor` is
    /// true. Without one, the fallback queries require a complete source graph.
    pub(crate) fn resolve(
        &self,
        positions: &[IVec3],
        surface: &impl SurfaceSupport,
        selected: Option<(IVec3, bool)>,
        kind: &StabilizerRowKind,
        index: u32,
        anchor: Option<BoundDynamicAnchor<'_>>,
    ) -> Result<
        (
            Vec<InstanceMeasurement>,
            Vec<InstanceBoundaryOperator>,
            bool,
        ),
        CompileError,
    > {
        debug_assert!(positions.is_sorted_by_key(glam::IVec3::to_array));
        debug_assert!(anchor.is_none() || positions.len() == 1);
        let mut arms = crate::FxMap::default();
        let selected_template = selected.map(|(position, selected)| {
            debug_assert_eq!(positions, [position]);
            let instances = self.selectives[&position];
            let TemplateRef::Selective {
                when_false,
                when_true,
            } = self.compiled[&position].template
            else {
                unreachable!("selected local gateway is a selective")
            };
            arms.insert(
                position,
                if selected {
                    instances.when_true
                } else {
                    instances.when_false
                },
            );
            (position, if selected { when_true } else { when_false })
        });
        let mut insertions =
            super::observable::resolve_observable_insertions_at_positions_with_spatial_ports(
                self.graph,
                surface,
                kind,
                self.compiled,
                self.pool,
                positions,
                index,
                true,
                self.spatial_ports,
                selected_template,
                anchor,
            )?;
        insertions.extend(
            super::observable::resolve_spatial_port_observable_insertions(
                surface,
                self.spatial_ports,
                self.spatial_templates,
                self.pool,
                positions,
                index,
                true,
            )?,
        );
        // Each edge gateway belongs to its smaller source position. Borrow the
        // complete surface so a proxy can bind every site without cloning it.
        let wall_sign = super::observable::push_wall_observable_insertions(
            self.graph,
            self.wall_templates,
            self.pool,
            surface,
            Some(positions),
            index,
            &mut insertions,
        )?;
        let (mut records, includes) = collect_assignment_content(
            &insertions,
            self.layout,
            &arms,
            self.instances,
            anchor.and_then(|anchor| anchor.source_t),
        )?;
        let (pipe_records, pipe_sign) = pipe_crossing_content(
            self.graph,
            self.pipe_templates,
            self.pool,
            self.instances,
            surface,
            positions,
            index,
        )?;
        records.extend(pipe_records);
        let sign = matches!(kind, StabilizerRowKind::Logical) && (wall_sign ^ pipe_sign);
        Ok((crate::xor_toggled_sorted(records), includes, sign))
    }
}

/// A temporal H crossing contributes its realignment records and the H Y H = -Y offset.
pub(super) fn pipe_crossing_content(
    graph: &BlockGraph,
    pipe_templates: &crate::FxMap<TemporalPipeRef, LoweringTemplateId>,
    template_pool: &LoweringTemplatePool,
    instance_by_site: &crate::FxMap<ChunkSiteSource, TemplateInstanceId>,
    fixed: &impl SurfaceSupport,
    positions: &[IVec3],
    index: u32,
) -> Result<(Vec<InstanceMeasurement>, bool), CompileError> {
    let mut crossings: Vec<(TemporalPipeRef, Pauli)> = fixed
        .owned_edges(Some(positions))
        .filter_map(|((src, dst), edge_pauli)| {
            hadamard_temporal_pipe(graph, src, dst).map(|pipe| {
                let pauli = realignment_observable_pauli(graph, &pipe, fixed, edge_pauli);
                (pipe, pauli)
            })
        })
        .collect();
    crossings.sort_by_key(|(pipe, _)| (pipe.src.to_array(), pipe.dst.to_array()));

    let mut records: crate::FxSet<InstanceMeasurement> = crate::FxSet::default();
    let mut sign = false;
    for (pipe, pauli) in crossings {
        if pauli == Pauli::I {
            continue;
        }
        sign ^= pauli == Pauli::Y;
        let template_id = pipe_templates
            .get(&pipe)
            .expect("crossed temporal pipe has a lowered template");
        let gateway = &template_pool
            .get(*template_id)
            .expect("temporal pipe template exists")
            .observable_gateway;
        let local = LocalStabilizer::isolated(pauli);
        let entry = gateway.lookup_ref(local).ok_or_else(|| {
            CompileError::ObservableMissingGatewayEntry {
                block_pos: pipe.src,
                observable_index: index,
                local_stabilizer: local.to_string(),
            }
        })?;
        let instance = instance_by_site[&ChunkSiteSource::TemporalPipe(pipe)];
        for chunk_meas in &entry.measurements {
            xor_toggle(
                &mut records,
                chunk_meas
                    .measurements
                    .iter()
                    .copied()
                    .map(|measurement| InstanceMeasurement {
                        instance,
                        measurement,
                    }),
            );
        }
    }
    let mut records: Vec<InstanceMeasurement> = records.into_iter().collect();
    records.sort_unstable();
    Ok((records, sign))
}

/// Relabel an assignment's template-local insertions into instance-global observable measurement
/// records and observable boundary operators, naming each site's instance (the resolved arm
/// instance for a selective site, else the block's own instance).
pub(super) fn collect_assignment_content(
    insertions: &[ObservableChunkInsertion],
    layout: BlockLayout,
    arm_instance_by_site: &crate::FxMap<IVec3, TemplateInstanceId>,
    instance_by_site: &crate::FxMap<ChunkSiteSource, TemplateInstanceId>,
    source_t: Option<TSourceGateway<'_>>,
) -> Result<(Vec<InstanceMeasurement>, Vec<InstanceBoundaryOperator>), CompileError> {
    let mut measurement_groups = Vec::new();
    let mut includes: Vec<InstanceBoundaryOperator> = Vec::new();

    for insertion in insertions {
        let instance = match insertion.site {
            ChunkSiteSource::Block(pos) => source_t
                .filter(|source| source.source == pos)
                .map(|source| source.instance)
                .or_else(|| arm_instance_by_site.get(&pos).copied())
                .unwrap_or_else(|| instance_by_site[&insertion.site]),
            ChunkSiteSource::TemporalPipe(_)
            | ChunkSiteSource::SpatialPipe(_)
            | ChunkSiteSource::SpatialPort(_) => instance_by_site[&insertion.site],
        };

        if !insertion.measurements.is_empty() {
            measurement_groups.push((instance, insertion.measurements.as_slice()));
        }

        let off = insertion_offset(layout, insertion.site)?;
        for (face, operator) in [
            (BoundaryFace::Input, &insertion.operator_in),
            (BoundaryFace::Output, &insertion.operator_out),
        ] {
            if operator.is_empty() {
                continue;
            }
            includes.push(InstanceBoundaryOperator {
                instance,
                face,
                operator: operator.try_translated(off)?,
            });
        }
    }

    Ok((
        xor_merge_assignment_measurements(measurement_groups),
        includes,
    ))
}

/// XOR-merge non-empty, non-overlapping sorted ranges without sorting every record.
fn xor_merge_assignment_measurements(
    mut groups: Vec<(TemplateInstanceId, &[u32])>,
) -> Vec<InstanceMeasurement> {
    groups.sort_unstable_by_key(|&(instance, group)| (instance, group[0]));
    let mut merged = Vec::with_capacity(groups.iter().map(|(_, group)| group.len()).sum());
    let mut previous = None;
    for (instance, group) in groups {
        debug_assert!(group.is_sorted());
        for &measurement in group {
            let measurement = InstanceMeasurement {
                instance,
                measurement,
            };
            debug_assert!(previous.is_none_or(|previous| previous <= measurement));
            previous = Some(measurement);
            if merged.last() == Some(&measurement) {
                merged.pop();
            } else {
                merged.push(measurement);
            }
        }
    }
    merged
}

fn insertion_offset(
    layout: BlockLayout,
    source: ChunkSiteSource,
) -> Result<glam::IVec2, CompileError> {
    match source {
        ChunkSiteSource::Block(pos) => layout.offset(pos),
        ChunkSiteSource::TemporalPipe(pipe) => layout.offset(pipe.endpoints_by_z().0),
        // The wall template uses its negative-axis endpoint's coordinates.
        ChunkSiteSource::SpatialPipe(pipe) => layout.offset(pipe.src),
        ChunkSiteSource::SpatialPort(port) => layout.offset(port),
    }
}
