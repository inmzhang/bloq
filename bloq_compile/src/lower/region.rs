use bloq_ir::{
    BloqEdge, BloqNode, BoundaryFace, ClassicalExpr, ClassicalNode, NodeProvenance, RegionNode,
    SourceBlockRef, SubGraph,
    lowering::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstance},
};
use glam::IVec3;

use crate::CompileError;

use super::{PlacedInstance, TemplateRemapper, detector};

/// Build the `RepeatUntilSuccess` region for a T block: cultivation
/// and escape quantum nodes joined by the intra-body seam (whose cross-template
/// post-selected parities become the escape node's `NodeRestart`s), plus the two
/// logical observables. Their predicted flips feed an explicit restart predicate;
/// physical postselection parities remain attached to the quantum stages.
pub(super) fn build_t_region(
    pos: IVec3,
    cultivation: PlacedInstance,
    escape: PlacedInstance,
    observable_base: u32,
    remapper: &mut TemplateRemapper<'_>,
) -> Result<RegionNode, CompileError> {
    // The cultivation template's six restart-flagged open Steane chains close
    // against the escape template's first merge round: cross-template restart
    // parities (U17b §4), attached to the escape node inside the body.
    let (seam_detectors, seam_restarts) = detector::compose_instance_seam(&[
        (
            &remapper.pool()[cultivation.template],
            cultivation.offset,
            cultivation.id,
        ),
        (&remapper.pool()[escape.template], escape.offset, escape.id),
    ])?;

    let mut body = SubGraph::new();
    let mut cultivation_node = BloqNode::from_members(vec![SourceBlockRef { pos }]);
    cultivation_node
        .expect_quantum_mut()
        .instances
        .push(TemplateInstance::new(
            cultivation.id,
            remapper.remap(cultivation.template),
            cultivation.offset,
        ));
    let cultivation_node = body.add_node(cultivation_node);

    let mut escape_node = BloqNode::from_members(vec![SourceBlockRef { pos }]);
    {
        let quantum = escape_node.expect_quantum_mut();
        quantum.instances.push(TemplateInstance::new(
            escape.id,
            remapper.remap(escape.template),
            escape.offset,
        ));
        quantum.detectors = seam_detectors;
        quantum.restarts = seam_restarts;
    }
    let escape_node = body.add_node(escape_node);
    body.add_edge(cultivation_node, escape_node, BloqEdge::quantum(vec![]));

    // GAP observables (U17 §5.1): one per escaped-patch logical, keyed in the
    // escape gateway by the template-local Pauli at the +Z face. The
    // Z-keyed entry takes the even slot, the X-keyed the odd one —
    // deterministic regardless of gateway map order or the dual-family swap.
    let escape_template = &remapper.pool()[escape.template];
    let connectivity = crate::Connectivity::ISOLATED.with_pipe(bloq_graph::Direction::ZPLUS);
    let mut observables = Vec::new();
    for (slot, pauli) in [(0u32, bloq_graph::Pauli::Z), (1, bloq_graph::Pauli::X)] {
        let key = crate::block::LocalStabilizer::new(pauli, connectivity);
        let entry = escape_template
            .observable_gateway
            .lookup(key)
            .expect("escape gateway carries both +Z-face logical entries");
        let index = observable_base + slot;

        // XOR-dedup the gateway measurements (duplicated records cancel),
        // mirroring the static observable path's parity fold.
        let measurements = crate::xor_toggled_sorted(
            entry
                .measurements
                .iter()
                .flat_map(|chunk| chunk.measurements.iter().copied()),
        );
        let generator = NodeProvenance::Generator { ordinal: index };
        let observable = body.add_node(
            BloqNode::classical(ClassicalNode::Observable {
                index: Some(index),
                measurements: measurements
                    .into_iter()
                    .map(|measurement| InstanceMeasurement {
                        instance: escape.id,
                        measurement,
                    })
                    .collect(),
                operators: [
                    (BoundaryFace::Input, &entry.operator_in),
                    (BoundaryFace::Output, &entry.operator_out),
                ]
                .into_iter()
                .map(|(face, operator)| {
                    Ok(InstanceBoundaryOperator {
                        instance: escape.id,
                        face,
                        operator: operator.try_translated(escape.offset)?,
                    })
                })
                .collect::<Result<_, CompileError>>()?,
            })
            .with_provenance(generator),
        );
        body.add_edge(escape_node, observable, BloqEdge::Order);
        observables.push(observable);
    }

    let restart = body.add_node(BloqNode::classical(ClassicalNode::Compute {
        expr: ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
    }));
    for (slot, observable) in observables.into_iter().enumerate() {
        body.add_edge(observable, restart, BloqEdge::flip(slot as u32));
    }

    Ok(RegionNode::RepeatUntilSuccess {
        body,
        restart_condition: ClassicalExpr::In(0),
        restart_source: Some(restart.into()),
    })
}
