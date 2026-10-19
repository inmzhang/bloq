//! Pinned Clifford oracle: preserve the supplied public rows and omit actions.

use bloq_graph::{Pauli, RuntimeStabilizerBasis, StabilizerGenerators, StabilizerRowKind};
use bloq_ir::{Basis, Bloq, BloqEdge, BloqNode, ClassicalExpr, ClassicalNode, NodeProvenance};

use super::leaves::ExprLeaves;
use super::observable::{
    MaterializedObservables, materialize_observable_nodes, resolve_observables,
};
use super::{
    ChunkSiteSource, LocalSurfaceContext, PhysicalInput, PhysicalPlacements, PhysicalProgram,
    PlacedPlan,
};
use crate::{CompileError, check_resource};

pub(crate) fn lower_clifford_proxy(
    input: PhysicalInput<'_>,
    stabilizers: &StabilizerGenerators,
) -> Result<Bloq, CompileError> {
    check_resource(
        "readout IDs",
        stabilizers.generators.len(),
        u32::MAX as usize,
    )?;
    let mut placed = PlacedPlan::new(input)?;
    placed.compose_detectors()?;
    let (observables, reference_offsets) = resolve_observables(stabilizers, &placed.context())?;
    let PhysicalProgram {
        mut bloq,
        placements,
        node_map,
    } = placed.emit(stabilizers.generators.len() as u32)?;
    debug_assert!(
        placements.selectives.is_empty(),
        "proxy selectives are pinned"
    );
    let mut materialized = materialize_observable_nodes(
        observables,
        stabilizers.generators.len() as u32,
        &mut bloq,
        &node_map,
    );
    materialized.reference_offsets = reference_offsets;
    emit_frames(input, stabilizers, &placements, &materialized, &mut bloq)?;
    let interface = super::LogicalInterface::bind(
        &bloq,
        input.graph,
        &placements.instances,
        input.spatial_ports,
        input.layout.distance(),
    );
    drop((materialized, placements, node_map));
    super::finish_program(&mut bloq, interface)?;
    #[cfg(debug_assertions)]
    super::validate_bloq_qubit_layout_for_source(&bloq, input.graph)?;
    Ok(bloq)
}

/// The proxy retains its pinned public basis for distance tests. Complete
/// surfaces still supply private parity, including rows with no public member.
fn emit_frames(
    input: PhysicalInput<'_>,
    stabilizers: &StabilizerGenerators,
    placements: &PhysicalPlacements,
    materialized: &MaterializedObservables,
    bloq: &mut Bloq,
) -> Result<(), CompileError> {
    let outputs = stabilizers.zx_graph.output_ports();
    let (rows, surfaces): (Vec<_>, Vec<_>) = RuntimeStabilizerBasis::from_generators(stabilizers)
        .with_t_nodes_as_ports()
        .into_output_correction_surfaces(&outputs)?
        .into_iter()
        .unzip();
    let correction = bloq_graph::solve_output_correction_symbolic(&outputs, &rows)?;
    let pipe_templates = input
        .plan
        .graph()
        .node_weights()
        .filter_map(|node| {
            let NodeProvenance::TemporalPipe { pipe } = node.provenance else {
                return None;
            };
            node.template
                .map(|template| (pipe, input.temporal_templates[&template]))
        })
        .collect();
    let context = LocalSurfaceContext {
        graph: input.graph,
        compiled: input.compiled,
        spatial_ports: input.spatial_ports,
        spatial_templates: input.spatial_port_templates,
        wall_templates: input.wall_templates,
        pipe_templates: &pipe_templates,
        pool: input.template_pool,
        instances: &placements.instances,
        selectives: &placements.selectives,
        layout: input.layout,
    };
    let mut owners = bloq
        .nodes()
        .flat_map(|(id, node)| {
            node.try_quantum().into_iter().flat_map(move |quantum| {
                quantum
                    .instances
                    .iter()
                    .map(move |instance| (instance.id, id))
            })
        })
        .collect::<crate::FxMap<_, _>>();
    for (site, node) in &placements.regions {
        if let Some(instance) = placements.instances.get(&ChunkSiteSource::Block(*site)) {
            owners.insert(*instance, *node);
        }
    }
    let mut terms = Vec::with_capacity(rows.len());
    for (index, (row, surface)) in rows.iter().zip(surfaces).enumerate() {
        let mut positions = surface
            .interior_nodes
            .iter()
            .chain(&surface.port_stabilizer)
            .filter_map(|(&pos, &pauli)| (pauli != Pauli::I).then_some(pos))
            .chain(
                surface
                    .interior_edges
                    .iter()
                    .filter(|(_, pauli)| **pauli != Pauli::I)
                    .flat_map(|(&(left, right), _)| [left, right]),
            )
            .collect::<Vec<_>>();
        positions.sort_unstable_by_key(glam::IVec3::to_array);
        positions.dedup();
        let (records, _, mut sign) = context.resolve(
            &positions,
            &surface,
            None,
            &StabilizerRowKind::Logical,
            index as u32,
            None,
        )?;
        let predecessors = records
            .iter()
            .map(|record| owners[&record.instance])
            .collect::<crate::FxSet<_>>();
        let parity = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            records,
            vec![],
        )));
        for owner in predecessors {
            bloq.add_edge(owner, parity, BloqEdge::Order);
        }
        for (slot, &ordinal) in row.readout_ordinals.iter().enumerate() {
            let ordinal = ordinal as u32;
            let observable = materialized
                .observable_by_index
                .get(&ordinal)
                .copied()
                .expect("frame member has a public readout");
            bloq.add_edge(observable, parity, BloqEdge::flip(slot as u32));
            sign ^= materialized.reference_offsets.contains(&ordinal);
        }
        terms.push((parity, sign));
    }
    for pair in correction.frames() {
        for (basis, rows) in [(Basis::X, &pair.x), (Basis::Z, &pair.z)] {
            let mut leaves = ExprLeaves::default();
            let mut sign = false;
            let mut operands = Vec::new();
            for &row in rows {
                let (input, offset) = &terms[row];
                sign ^= offset;
                operands.push(leaves.input(*input));
            }
            let mut expr = ClassicalExpr::xor(operands);
            if sign {
                expr = ClassicalExpr::Not(Box::new(expr));
            }
            let frame = bloq.add_node(
                BloqNode::classical(ClassicalNode::Compute { expr }).with_provenance(
                    NodeProvenance::OutputFrame {
                        port: pair.output,
                        basis,
                    },
                ),
            );
            leaves.wire(bloq, frame);
        }
    }
    Ok(())
}
