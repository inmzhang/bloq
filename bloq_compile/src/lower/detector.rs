use bloq_circuit::{DetectorTerm, FlowKey, OffsetFlows};
use bloq_ir::{
    NodeDetector, SourceBlockRef,
    lowering::{InstanceMeasurement, NodeRestart, TemplateInstanceId},
};
use glam::IVec2;
use petgraph::graph::NodeIndex;

use crate::CompileError;
use crate::block::LoweringTemplate;

use super::classify::NodeLowering;
use super::plan::LowerPlan;
use super::{BloqLowerContext, ChunkSiteSource};

type DetectorFlowEngine<'c> = bloq_circuit::FlowEngine<'c, DetectorTerm<InstanceMeasurement>>;

impl<'ctx> BloqLowerContext<'ctx> {
    /// Resolve the templates each node member instantiates, with the offsets
    /// and measurement-map sites needed to read their flows in place.
    ///
    /// The detector pass walks template flows directly instead of per-node
    /// translated copies: the engine's offset-aware keys make that exact, and
    /// it skips materializing two translated `PauliMap`s per flow per node.
    fn detector_node_context(
        &self,
        plan_node: NodeIndex,
    ) -> Result<DetectorNodeContext<'ctx>, CompileError> {
        let node = &self.input.plan.graph()[plan_node];
        let node_instances = self
            .node_instances
            .get(plan_node.index())
            .expect("plan node has lowered template instances");
        let mut sources = Vec::new();
        match &self.classified[plan_node.index()] {
            NodeLowering::Selective { .. } => {
                unreachable!("static detector lowering receives pinned selective templates")
            }
            // A T block lowers to a `RepeatUntilSuccess` region; its escape
            // instance is the block's temporal face toward the +Z neighbour
            // (U8's source-only residual). The escape template's boundary flows
            // also carry the six Steane-side consumers already closed by the
            // intra-body cultivation seam (`compose_instance_seam`) — at top
            // level they have nothing to match, so this source skips unmatched
            // inputs instead of erroring. The end-face creators stay live and
            // the neighbour's consumers close them into ordinary top-level
            // `NodeDetector`s referencing the in-body escape instance.
            NodeLowering::TRegion { pos, escape, .. } => {
                let [instance] = &node_instances[..] else {
                    unreachable!("T plan node carries exactly its escape instance");
                };
                debug_assert_eq!(instance.site_source, ChunkSiteSource::Block(*pos));
                sources.push(MemberFlowSource {
                    template: &self.input.template_pool[*escape],
                    offset: self.input.layout.offset(*pos)?,
                    instance_id: instance.instance.id,
                    skip_unmatched_inputs: true,
                });
            }
            NodeLowering::TemporalPipe {
                pipe,
                template,
                origin,
            } => {
                assert_eq!(
                    node_instances.len(),
                    1,
                    "temporal pipe node has one lowered template instance"
                );
                let instance = &node_instances[0];
                debug_assert_eq!(instance.site_source, ChunkSiteSource::TemporalPipe(*pipe));
                sources.push(MemberFlowSource {
                    template: &self.input.template_pool[*template],
                    offset: self.input.layout.offset(*origin)?,
                    instance_id: instance.instance.id,
                    skip_unmatched_inputs: false,
                });
            }
            NodeLowering::SpatialPort { source, template } => {
                let [instance] = &node_instances[..] else {
                    unreachable!("derived temporal Port has one instance");
                };
                debug_assert_eq!(instance.site_source, ChunkSiteSource::SpatialPort(*source));
                sources.push(MemberFlowSource {
                    template: &self.input.template_pool[*template],
                    offset: self.input.layout.offset(*source)?,
                    instance_id: instance.instance.id,
                    skip_unmatched_inputs: false,
                });
            }
            NodeLowering::Fixed { members, walls } => {
                assert_eq!(
                    node_instances.len(),
                    members.len() + walls.len(),
                    "block node has one lowered template instance per member and wall"
                );
                let (member_instances, wall_instances) = node_instances.split_at(members.len());
                for (&(member, template), instance) in members.iter().zip(member_instances) {
                    debug_assert_eq!(instance.site_source, ChunkSiteSource::Block(member.pos));
                    sources.push(MemberFlowSource {
                        template: &self.input.template_pool[template],
                        offset: self.input.layout.offset(member.pos)?,
                        instance_id: instance.instance.id,
                        skip_unmatched_inputs: false,
                    });
                }
                for (wall, instance) in walls.iter().zip(wall_instances) {
                    debug_assert_eq!(
                        instance.site_source,
                        ChunkSiteSource::SpatialPipe(wall.pipe)
                    );
                    sources.push(MemberFlowSource {
                        template: &self.input.template_pool[wall.template],
                        offset: self.input.layout.offset(wall.pipe.src)?,
                        instance_id: instance.instance.id,
                        skip_unmatched_inputs: false,
                    });
                }
            }
        }
        Ok(DetectorNodeContext {
            plan_node: plan_node.index(),
            layer: node.layer,
            members: node.block_members(),
            sources,
        })
    }
}

struct DetectorNodeContext<'a> {
    plan_node: usize,
    layer: i64,
    members: &'a [SourceBlockRef],
    sources: Vec<MemberFlowSource<'a>>,
}

/// One node member's compiled template, placed at the member's XY offset.
struct MemberFlowSource<'a> {
    template: &'a LoweringTemplate,
    offset: IVec2,
    instance_id: TemplateInstanceId,
    /// Compose with `append_group_skipping_unmatched_inputs`: a consumer flow
    /// with no matching open chain is dropped instead of erroring. Set only for
    /// a T block's escape template at top level, whose Steane-side consumers
    /// were already closed by the intra-body cultivation seam.
    skip_unmatched_inputs: bool,
}

/// Composed boundary detectors grouped by the plan node the chains closed on.
type ComposedDetectors = Vec<(usize, Vec<NodeDetector>)>;
/// Composed restart syndromes grouped by the plan node the chains closed on.
type ComposedRestarts = Vec<(usize, Vec<NodeRestart>)>;

/// Pending side-table entries keyed by the plan node the chain closed on.
struct DetectorLowerState<'c> {
    detectors: ComposedDetectors,
    restarts: ComposedRestarts,
    engine: DetectorFlowEngine<'c>,
}

impl DetectorLowerState<'_> {
    fn drain_node_chains(&mut self, plan_node: usize) {
        let (detectors, restarts) = bloq_ir::lowering::drain_composed_chains(&mut self.engine);
        if !detectors.is_empty() {
            self.detectors.push((plan_node, detectors));
        }
        if !restarts.is_empty() {
            self.restarts.push((plan_node, restarts));
        }
    }
}

// ==============================================================================
// Boundary-only detector composition
// ==============================================================================
//
// Detectors that close inside a single template are recovered downstream by the
// backend (it instantiates `program_template.detectors`/`repeat_states` per
// instance via a cheap offset + measurement relabel). So instance lowering only
// composes the **boundary** detectors — flows the template left open that close
// against a temporally adjacent node.
//
// Those open flows are precomputed per template as `boundary_flows` (see
// `block/compile.rs::drain_boundary_flows`), already fused into mutually
// independent chains. Here we replay only that residual through a per-component
// engine in emission order, so an open created by a lower-z node is consumed by
// the node above it. This pass composes nothing but `TopLevel` `NodeDetector`s
// and allocates no loop state — a loop's recurrence is backend-owned, and its
// sole cross-node round is resolved by the backend's post-`REPEAT` lookback.
pub(super) fn attach_temporal_component_detectors(
    plan: &LowerPlan,
    context: &mut BloqLowerContext<'_>,
) -> Result<(), CompileError> {
    for mut component in plan.temporal_node_components() {
        layer_major_order(&mut component, context)?;
        attach_component_detectors(&component, context)?;
    }
    Ok(())
}

/// Layer-major, emit-order tie-broken order for one temporal component. Every
/// plan edge ascends `layer`, so this is still a topological order — but plain
/// emit order is not enough here: the engine keys open chains by 2D support only,
/// and a bare toposort may interleave same-footprint branches out of z order (two
/// T columns are DAG sources with no path between them, U17e), letting a lower
/// column's creators clobber another column's still-open chains at the same XY
/// offset. Layer-major ordering walks the component in time.
fn layer_major_order(
    nodes: &mut [NodeIndex],
    context: &BloqLowerContext<'_>,
) -> Result<(), CompileError> {
    nodes.sort_by_cached_key(|node| {
        (
            context.input.plan.graph()[*node].layer,
            context.emit_order[node.index()],
        )
    });

    let mut first = 0;
    while first < nodes.len() {
        let layer = context.input.plan.graph()[nodes[first]].layer;
        let end = nodes[first..]
            .partition_point(|node| context.input.plan.graph()[*node].layer == layer)
            + first;
        order_same_layer_flows(&mut nodes[first..end], context, layer)?;
        first = end;
    }
    Ok(())
}

/// A same-layer moving patch may produce a boundary still waiting to be
/// consumed by its neighbor. Replay those consumers first; unrelated nodes
/// retain emission order.
fn order_same_layer_flows(
    nodes: &mut [NodeIndex],
    context: &BloqLowerContext<'_>,
    layer: i64,
) -> Result<(), CompileError> {
    if nodes.len() < 2 {
        return Ok(());
    }
    let mut consumers = crate::FxMap::<FlowKey<'_>, smallvec::SmallVec<[usize; 1]>>::default();
    let mut produced = vec![Vec::new(); nodes.len()];
    for (index, &node) in nodes.iter().enumerate() {
        let node = context.detector_node_context(node)?;
        for source in node.sources {
            for flow in &source.template.program_template.boundary_flows {
                if !flow.start.is_empty() {
                    consumers
                        .entry(FlowKey::try_new(&flow.start, source.offset)?)
                        .or_default()
                        .push(index);
                }
                if !flow.end.is_empty() {
                    produced[index].push(FlowKey::try_new(&flow.end, source.offset)?);
                }
            }
        }
    }

    let mut successors = vec![crate::FxSet::default(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for (producer, boundaries) in produced.iter().enumerate() {
        for boundary in boundaries {
            for &consumer in consumers.get(boundary).into_iter().flatten() {
                if consumer != producer && successors[consumer].insert(producer) {
                    indegree[producer] += 1;
                }
            }
        }
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| (count == 0).then_some(index))
        .collect::<std::collections::BTreeSet<_>>();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(index) = ready.pop_first() {
        order.push(nodes[index]);
        for &next in &successors[index] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.insert(next);
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(CompileError::NodeFlowCompositionFailed {
            layer,
            members: nodes
                .iter()
                .flat_map(|node| context.input.plan.graph()[*node].block_members())
                .map(|member| member.pos)
                .collect(),
            source: bloq_circuit::FlowError::Composition(
                "cyclic same-layer boundary-flow dependency".into(),
            ),
        });
    }
    nodes.copy_from_slice(&order);
    Ok(())
}

/// Compose the already pinned proxy's boundary flows with their allocated ids.
fn attach_component_detectors(
    nodes: &[NodeIndex],
    context: &mut BloqLowerContext<'_>,
) -> Result<(), CompileError> {
    let mut state = DetectorLowerState {
        detectors: Vec::new(),
        restarts: Vec::new(),
        engine: DetectorFlowEngine::new(),
    };
    for &node in nodes {
        let node_context = context.detector_node_context(node)?;
        append_node_boundary_flows(&mut state, &node_context)?;
    }
    let remaining = state
        .engine
        .finish()
        .map_err(|error| CompileError::DetectorFlowCompositionFailed { source: error })?;
    debug_assert!(
        remaining.is_empty(),
        "completed detectors are drained into their closing node"
    );
    for (node, detectors) in state.detectors {
        debug_assert!(context.node_detectors[node].is_empty());
        context.node_detectors[node] = detectors;
    }
    for (node, restarts) in state.restarts {
        debug_assert!(context.node_restarts[node].is_empty());
        context.node_restarts[node] = restarts;
    }
    Ok(())
}

/// Append one node's boundary flows to the component engine, one group per
/// source.
///
/// Each source's `boundary_flows` are already fused into mutually independent
/// chains, so the whole list goes in as a single `TopLevel` group. Sources are
/// appended independently: a node's members lie in one z-layer and expose
/// boundary flows only on their own temporal faces, so two members never collide
/// on a key. There is no loop arm — a top-open loop's residual arrives as a plain
/// creator flow, resolved by the backend's post-`REPEAT` lookback.
fn append_node_boundary_flows<'c>(
    state: &mut DetectorLowerState<'c>,
    node: &DetectorNodeContext<'c>,
) -> Result<(), CompileError> {
    if !node
        .sources
        .iter()
        .any(|source| source.skip_unmatched_inputs)
    {
        let sources: Vec<_> = node
            .sources
            .iter()
            .filter(|source| !source.template.program_template.boundary_flows.is_empty())
            .collect();
        let parts: Vec<_> = sources
            .iter()
            .map(|source| {
                OffsetFlows::new(
                    &source.template.program_template.boundary_flows,
                    source.offset,
                )
            })
            .collect();
        state
            .engine
            .append_group_allowing_stabilizer_transition(&parts, |part, &measurement| {
                DetectorTerm::Measurement(InstanceMeasurement {
                    instance: sources[part].instance_id,
                    measurement,
                })
            })
            .map_err(|error| node_flow_error(node, error))?;
        state.drain_node_chains(node.plan_node);
        return Ok(());
    }

    for source in &node.sources {
        if source.template.program_template.boundary_flows.is_empty() {
            continue;
        }
        let instance = source.instance_id;
        let parts = [OffsetFlows::new(
            &source.template.program_template.boundary_flows,
            source.offset,
        )];
        let term = |_: usize, &measurement: &u32| {
            DetectorTerm::Measurement(InstanceMeasurement {
                instance,
                measurement,
            })
        };
        if source.skip_unmatched_inputs {
            state
                .engine
                .append_group_skipping_unmatched_inputs(&parts, term)
        } else {
            state.engine.append_group_allowing_dangling(&parts, term)
        }
        .map_err(|error| node_flow_error(node, error))?;
    }
    // Classification (Discard drop / Restart -> `NodeRestart` / else
    // `NodeDetector`) is shared with the U11 edit recompose so the marker
    // discipline cannot drift between compile time and post-hoc seam edits.
    state.drain_node_chains(node.plan_node);
    Ok(())
}

/// Compose the seam between template instances of one `RepeatUntilSuccess`
/// body: sources in emission order — cultivation, then escape. All
/// completed chains here are the cultivation template's restart-flagged open
/// Steane stabilizers closed by the escape template's first merge round, so
/// they land as `NodeRestart`s (plus any plain detectors a future body shape
/// might close). Deliberately no `engine.finish()`: the escape template's
/// end-face creators legitimately stay open — the +Z neighbour closes them at
/// top level (with the T source in skip mode dropping the Steane consumers
/// this pass already resolved).
pub(super) fn compose_instance_seam(
    sources: &[(&LoweringTemplate, IVec2, TemplateInstanceId)],
) -> Result<(Vec<NodeDetector>, Vec<NodeRestart>), CompileError> {
    let mut state = DetectorLowerState {
        detectors: Vec::new(),
        restarts: Vec::new(),
        engine: DetectorFlowEngine::new(),
    };
    for &(template, offset, instance_id) in sources {
        let node = DetectorNodeContext {
            plan_node: 0,
            layer: 0,
            members: &[],
            sources: vec![MemberFlowSource {
                template,
                offset,
                instance_id,
                skip_unmatched_inputs: false,
            }],
        };
        append_node_boundary_flows(&mut state, &node)?;
    }
    Ok((
        state
            .detectors
            .into_iter()
            .flat_map(|(_, detectors)| detectors)
            .collect(),
        state
            .restarts
            .into_iter()
            .flat_map(|(_, restarts)| restarts)
            .collect(),
    ))
}

fn node_flow_error(node: &DetectorNodeContext<'_>, error: bloq_circuit::FlowError) -> CompileError {
    CompileError::NodeFlowCompositionFailed {
        layer: node.layer,
        members: node.members.iter().map(|member| member.pos).collect(),
        source: error,
    }
}

#[cfg(test)]
mod tests {
    use bloq_circuit::{
        Chunk, ChunkOrLoop, CoordCircuit, DetectorParity, Flow, FlowMarker, Pauli, PauliBasis,
        PauliMap,
    };
    use glam::ivec2;

    use super::*;
    use crate::block::ObservableGateway;

    /// One-measurement chunk carrying `flow`.
    fn chunk_with_flow(coord: glam::IVec2, flow: Flow) -> ChunkOrLoop {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [coord]);
        ChunkOrLoop::Single(Box::new(Chunk {
            circuit,
            flows: vec![flow],
        }))
    }

    fn template_with_flow(coord: glam::IVec2, flow: Flow) -> LoweringTemplate {
        LoweringTemplate::from_chunks(vec![chunk_with_flow(coord, flow)], ObservableGateway::new())
            .expect("single-flow template compiles")
    }

    /// The cross-template restart composition: a restart-flagged chain left
    /// open by one template instance and closed by the next fuses at the seam
    /// into a `NodeRestart` carrying both instances' measurements — never a
    /// `NodeDetector`.
    #[test]
    fn cross_instance_restart_chain_becomes_node_restart() {
        let seam = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);
        let lower_template = template_with_flow(
            ivec2(0, 0),
            Flow::new(PauliMap::empty(), seam.clone())
                .with_measurements([0])
                .with_marker(FlowMarker::Restart),
        );
        let upper_template = template_with_flow(
            ivec2(0, 0),
            Flow::new(seam, PauliMap::empty()).with_measurements([0]),
        );

        let node_context = |template, plan_node, instance| DetectorNodeContext {
            plan_node,
            layer: 0,
            members: &[],
            sources: vec![MemberFlowSource {
                template,
                offset: IVec2::ZERO,
                instance_id: TemplateInstanceId(instance),
                skip_unmatched_inputs: false,
            }],
        };
        let mut state = DetectorLowerState {
            detectors: Vec::new(),
            restarts: Vec::new(),
            engine: DetectorFlowEngine::new(),
        };
        append_node_boundary_flows(&mut state, &node_context(&lower_template, 0, 0))
            .expect("lower node appends");
        append_node_boundary_flows(&mut state, &node_context(&upper_template, 1, 1))
            .expect("upper node closes the seam");
        assert!(
            state.engine.finish().expect("no open flows").is_empty(),
            "the fused chain drains at the closing node"
        );

        assert!(
            state.detectors.is_empty(),
            "no detector for a restart chain"
        );
        assert_eq!(state.restarts.len(), 1);
        // The chain completes at the upper node, so the restart attaches there.
        let (plan_node, restarts) = &state.restarts[0];
        assert_eq!(*plan_node, 1);
        assert_eq!(restarts.len(), 1);
        assert_eq!(
            restarts[0].parity,
            DetectorParity::from_measurements([
                InstanceMeasurement {
                    instance: TemplateInstanceId(0),
                    measurement: 0,
                },
                InstanceMeasurement {
                    instance: TemplateInstanceId(1),
                    measurement: 0,
                },
            ])
        );
    }
}
