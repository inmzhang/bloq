use bloq_ir::{NodeProvenance, SourceBlockRef, TemporalPipeRef};
use glam::IVec3;

use crate::block::LoweringTemplateId;
use crate::compile::{CompiledTemplateMap, TemplateRef};

use super::plan::{LowerPlan, SpatialPipeRef, TemplatePlan};

/// One spatial Hadamard wall placed inside a block component's node.
#[derive(Debug, Clone, Copy)]
pub(super) struct WallLowering {
    pub(super) pipe: SpatialPipeRef,
    pub(super) template: LoweringTemplateId,
}

/// How one plan node lowers, computed once by [`classify_nodes`] and matched by
/// the instance-placement, detector, and node-materialization passes. Keeping
/// selective/T dispatch in exactly one place means adding a new
/// dynamically lowered block kind extends this enum and lets the compiler's
/// exhaustiveness check find every pass that must handle it.
#[derive(Debug)]
pub(super) enum NodeLowering {
    /// Fixed block component: one template instance per member, plus
    /// one per spatial Hadamard wall standing between two of them.
    Fixed {
        members: Vec<(SourceBlockRef, LoweringTemplateId)>,
        walls: Vec<WallLowering>,
    },
    /// Temporal Hadamard realignment pipe node: one instance at the lower
    /// endpoint block's offset.
    TemporalPipe {
        pipe: TemporalPipeRef,
        template: LoweringTemplateId,
        origin: IVec3,
    },
    /// Positionless temporal Port derived from an authored spatial Port.
    SpatialPort {
        source: IVec3,
        template: LoweringTemplateId,
    },
    /// Selective measurement block, lowered to two guarded per-basis instances.
    Selective {
        pos: IVec3,
        when_true: LoweringTemplateId,
        when_false: LoweringTemplateId,
    },
    /// T block, lowered to a `RepeatUntilSuccess` region holding the
    /// cultivation + escape stage pair.
    TRegion {
        pos: IVec3,
        cultivation: LoweringTemplateId,
        escape: LoweringTemplateId,
    },
}

/// Classify every plan node, indexed by `NodeIndex::index()`.
///
/// The sole-member invariant for dynamic blocks (their signatures forbid
/// spatial pipes, so they never join a spatial component) is asserted here,
/// once, instead of at each consuming pass.
pub(super) fn classify_nodes(
    plan: &LowerPlan,
    compiled: &CompiledTemplateMap,
    port_templates: &CompiledTemplateMap,
    temporal_templates: &crate::FxMap<TemplatePlan, LoweringTemplateId>,
    wall_templates: &crate::FxMap<SpatialPipeRef, LoweringTemplateId>,
) -> Vec<NodeLowering> {
    plan.graph()
        .node_weights()
        .map(|node| {
            if let NodeProvenance::SpatialPortSubstitution { source, .. } = &node.provenance {
                let source = *source;
                let template = port_templates[&source]
                    .template
                    .observable_template()
                    .expect("derived temporal Ports use fixed templates");
                return NodeLowering::SpatialPort { source, template };
            }
            if let Some(template_plan) = node.template {
                let template = *temporal_templates
                    .get(&template_plan)
                    .expect("temporal template plan was compiled before lowering");
                let TemplatePlan { origin, .. } = template_plan;
                let NodeProvenance::TemporalPipe { pipe } = &node.provenance else {
                    unreachable!("temporal template nodes are backed by temporal pipe sources");
                };
                return NodeLowering::TemporalPipe {
                    pipe: *pipe,
                    template,
                    origin,
                };
            }

            let members = node.block_members();
            if let [member] = members {
                match compiled
                    .get(&member.pos)
                    .expect("plan members come from compiled blocks")
                    .template
                {
                    TemplateRef::Selective {
                        when_true,
                        when_false,
                    } => {
                        return NodeLowering::Selective {
                            pos: member.pos,
                            when_true,
                            when_false,
                        };
                    }
                    TemplateRef::T {
                        cultivation,
                        escape,
                    } => {
                        return NodeLowering::TRegion {
                            pos: member.pos,
                            cultivation,
                            escape,
                        };
                    }
                    TemplateRef::Fixed(_) => {}
                }
            }

            let members = members
                .iter()
                .map(|&member| {
                    let template = match compiled
                        .get(&member.pos)
                        .expect("plan members come from compiled blocks")
                        .template
                    {
                        TemplateRef::Fixed(id) => id,
                        TemplateRef::Selective { .. } | TemplateRef::T { .. } => unreachable!(
                            "a dynamic block never joins a spatial component \
                             (its signature forbids spatial pipes)"
                        ),
                    };
                    (member, template)
                })
                .collect();
            let walls = node
                .walls
                .iter()
                .map(|&pipe| WallLowering {
                    pipe,
                    template: *wall_templates
                        .get(&pipe)
                        .expect("spatial Hadamard walls are compiled before lowering"),
                })
                .collect();
            NodeLowering::Fixed { members, walls }
        })
        .collect()
}
