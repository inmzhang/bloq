//! Native compilation from one Boolean-valued correlation space.

#[path = "registry/assembly.rs"]
mod assembly;
#[path = "registry/flow_families.rs"]
mod flow_families;
#[path = "registry/flows.rs"]
mod flows;
#[path = "registry/physical.rs"]
mod physical;
#[path = "registry/readouts.rs"]
mod readouts;

use crate::FxMap as HashMap;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use bloq_graph::{BlockGraph, GuardedSurfaceSpace, GuardedTopology, MaterializedModuleSite};
use bloq_ir::lowering::{InstanceMeasurement, TemplateInstanceId};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, ClassicalNode, InstanceProvenance,
    NodeProvenance, QuantumGuard, QuantumTimeline, TemplateId,
};
use glam::IVec3;

use super::{CompileArtifacts, CompileContext, CompileError};

fn assembly(reason: impl Into<String>) -> CompileError {
    CompileError::BranchAssembly {
        reason: reason.into(),
    }
}

fn needs_local_variants(graph: &BlockGraph) -> bool {
    graph.has_continuing_branches()
        || graph
            .blocks()
            .chain(
                graph
                    .branch_definitions()
                    .iter()
                    .flat_map(|region| region.on_false().blocks().chain(region.on_true().blocks())),
            )
            .any(|block| block.kind().is_selective())
}

fn hierarchy_needs_local_variants(program: &BlockGraph) -> bool {
    let mut pending = vec![program.root()];
    let mut seen = BTreeSet::new();
    while let Some(module) = pending.pop() {
        if !seen.insert(&module.name) {
            continue;
        }
        if needs_local_variants(module.local_body()) {
            return true;
        }
        pending.extend(module.instances.iter().map(|instance| {
            program
                .module(&instance.definition)
                .expect("validated definition")
        }));
    }
    false
}

pub(super) fn compile_graph(
    context: &CompileContext,
    graph: &BlockGraph,
) -> Result<CompileArtifacts, CompileError> {
    compile_native(
        context,
        graph,
        &Default::default(),
        &[],
        bloq_graph::default_module_jobs(),
    )
}

pub(super) fn compile_hierarchy(
    context: &CompileContext,
    program: &BlockGraph,
    jobs: NonZeroUsize,
) -> Result<CompileArtifacts, CompileError> {
    if !hierarchy_needs_local_variants(program) {
        let (linked, definitions) = context.prepare_module_definitions(program, jobs)?;
        return compile_native(context, &linked.graph, &linked.sites, &definitions, jobs);
    }
    // C0 remains a definition contract, including definitions whose parent
    // later consumes all their public outputs.
    let limits = context.config.certification_limits();
    let mut pending = vec![program.root()];
    let mut seen = BTreeSet::new();
    while let Some(module) = pending.pop() {
        if !seen.insert(module.name.clone()) {
            continue;
        }
        pending.extend(module.instances.iter().map(|instance| {
            program
                .module(&instance.definition)
                .expect("validated definition")
        }));
        if module.name == program.root().name {
            continue;
        }
        let linked = bloq_graph::flatten_module_definition(program, module, "")?;
        certify_definition(
            &module.name,
            &linked.graph.fix_shadowed_faces(),
            &linked.sites,
            limits,
        )?;
    }
    let linked = bloq_graph::flatten_module_definition(program, program.root(), "")?;
    compile_native(context, &linked.graph, &linked.sites, &[], jobs)
}

pub(super) fn certify_definition(
    name: &str,
    graph: &BlockGraph,
    sites: &std::collections::HashMap<IVec3, MaterializedModuleSite>,
    limits: bloq_graph::ModuleCertificationLimits,
) -> Result<(), CompileError> {
    GuardedTopology::new(graph, limits)
        .and_then(|topology| GuardedSurfaceSpace::new(topology, sites, limits))
        .and_then(|space| space.plan_readouts().map(drop))
        .map_err(|source| match source {
            bloq_graph::BlockGraphError::Stabilizer(
                bloq_graph::StabilizerError::ResourceLimited {
                    phase,
                    observed,
                    limit,
                },
            ) => bloq_graph::ModuleCertificationError::ResourceLimited {
                module: name.to_owned(),
                phase,
                observed,
                limit,
            }
            .into(),
            source => bloq_graph::ModuleCertificationError::Graph {
                module: name.to_owned(),
                source,
            }
            .into(),
        })
}

fn compile_native(
    context: &CompileContext,
    source: &BlockGraph,
    source_sites: &std::collections::HashMap<IVec3, MaterializedModuleSite>,
    definitions: &[Arc<super::PreparedDefinitionObject>],
    jobs: NonZeroUsize,
) -> Result<CompileArtifacts, CompileError> {
    use bloq_utils::boolean::DECISION_TRUE;
    let started = web_time::Instant::now();
    context.report_progress(super::CompileStage::Validation)?;
    if context.clifford_proxy.is_some() {
        return Err(CompileError::StructuralBranchCliffordProxyUnsupported);
    }
    let limits = context.config.certification_limits();
    let distance = context.config.code_distance();
    source.validate_resource_limits(limits)?;
    // One origin for both arms. Subtracting the origin also handles i32::MIN,
    // whose negation cannot be represented as a translation vector.
    let low = source.spans().map_or(0, |(_, _, z)| *z.start());
    let graph = source.with_zero_min_z()?.fix_shadowed_faces();
    let (bloq, interface, warnings) = {
        let sites = source_sites
            .iter()
            .map(|(&position, site)| {
                let z = position
                    .z
                    .checked_sub(low)
                    .ok_or_else(|| assembly("module source site normalization overflows"))?;
                Ok((IVec3::new(position.x, position.y, z), site.clone()))
            })
            .collect::<Result<_, CompileError>>()?;
        context.report_progress(super::CompileStage::Correlations)?;
        let topology = GuardedTopology::new(&graph, limits)?;
        let relation = GuardedSurfaceSpace::new(topology, &sites, limits)?;
        let mut readout_plan = relation.plan_readouts()?;
        context.report_progress(super::CompileStage::Placement)?;
        let mut family =
            physical::ScheduledFamily::new(&mut readout_plan.topology, distance, limits)?;
        let mut assembly = assembly::Assembly::new();
        let mut imported = Vec::new();
        // Physical template compilation follows the existing worker budget. Signed
        // row composition above ran once; no complete semantic projections are built.
        context.report_progress(super::CompileStage::Templates)?;
        for batch in family.cases.chunks(jobs.get()) {
            for result in bloq_graph::map_jobs(batch, jobs, |scheduled| {
                physical::PhysicalProjection::new(
                    context,
                    scheduled,
                    &family.t_sides,
                    definitions,
                    &sites,
                )
            }) {
                let (physical, program) = result?;
                imported.push(assembly.import(physical, program)?);
            }
        }
        drop(family.cases);
        drop(family.t_sides);
        let mut profiles = Vec::new();
        let selectors = readout_plan
            .topology
            .resolves
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        for (position, guard, case) in family.sites {
            profiles.push(assembly.register(
                &imported[case],
                position,
                guard,
                selectors.get(&position).copied(),
                &mut readout_plan.topology.diagram,
            )?);
        }
        drop((imported, selectors));
        let mut boolean = assembly.build(&mut readout_plan.topology)?;
        assembly.order_walking_choreography(&mut readout_plan.topology)?;
        assembly.compose_flows(
            &mut readout_plan.topology,
            &mut boolean,
            limits.max_witness_nodes,
            distance,
        )?;
        context.report_progress(super::CompileStage::Readouts)?;
        readouts::lower(
            &mut readout_plan,
            &mut profiles,
            &mut assembly,
            &mut boolean,
            distance,
            &mut family.anchors,
            limits.max_witness_nodes,
        )?;
        let interface = {
            // A local projection includes neighboring lookup/emission context.
            // External cuts belong to the registered source site's own instance.
            let mut instances = HashMap::default();
            for profile in &profiles {
                for source in [
                    crate::lower::ChunkSiteSource::Block(profile.position),
                    crate::lower::ChunkSiteSource::SpatialPort(profile.position),
                ] {
                    let Some(&instance) = profile.projection.sites.get(&source) else {
                        continue;
                    };
                    if assembly.members[instance.0 as usize].guard == DECISION_TRUE
                        && let Some(previous) = instances.insert(source, instance)
                        && previous != instance
                    {
                        return Err(self::assembly(format!(
                            "external source site {source:?} has multiple common instances"
                        )));
                    }
                }
            }
            for port in readout_plan
                .topology
                .source
                .blocks()
                .filter(|block| block.kind().is_port())
            {
                let source = if family.reference.spatial_ports.contains_key(&port.pos()) {
                    crate::lower::ChunkSiteSource::SpatialPort(port.pos())
                } else {
                    crate::lower::ChunkSiteSource::Block(port.pos())
                };
                if !instances.contains_key(&source) {
                    // WF-7: an external cut needs an always-selected owner.
                    return Err(self::assembly(format!(
                        "external Port at {} has no common physical instance",
                        port.pos()
                    )));
                }
            }
            crate::lower::LogicalInterface::bind(
                &assembly.bloq,
                &family.reference.graph,
                &instances,
                &family.reference.spatial_ports,
                distance,
            )
        };
        // Physical planning already kept the selected source geometry. Port
        // expansion does not change its pipes, which are all this warning reads.
        let warnings = crate::config::spatial_hadamard_distance_warning(&family.reference.graph)
            .into_iter()
            .collect();
        (assembly.bloq, interface, warnings)
    };
    finish_native(context, started, &graph, bloq, interface, warnings)
}

fn finish_native(
    context: &CompileContext,
    started: web_time::Instant,
    graph: &BlockGraph,
    mut bloq: Bloq,
    interface: crate::lower::LogicalInterface,
    warnings: Vec<&'static str>,
) -> Result<CompileArtifacts, CompileError> {
    context.report_progress(super::CompileStage::Optimization)?;
    let limits = context.config.certification_limits();
    let distance = context.config.code_distance();
    crate::lower::finish_program(&mut bloq, interface)?;
    bloq.insert_metadata(super::CODE_DISTANCE_METADATA_KEY, distance);
    bloq.insert_metadata(super::CONVENTION_METADATA_KEY, "fixed-bulk".to_owned());
    crate::lower::validate_bloq_qubit_layout_with_limits(&bloq, graph, limits.boolean_limits())?;
    Ok(CompileArtifacts {
        bloq,
        warnings,
        compile_duration: started.elapsed(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InstanceKey {
    template: TemplateId,
    offset: glam::IVec2,
    provenance: InstanceProvenance,
    lexical: Option<String>,
}

struct Stage {
    member: usize,
    provenance: NodeProvenance,
    timeline: Option<QuantumTimeline>,
}

fn has_quantum(node: &BloqNode) -> bool {
    node.try_quantum().is_some()
        || node.try_region().is_some_and(|region| {
            region
                .bodies()
                .any(|(_, body)| body.nodes().any(|(_, node)| has_quantum(node)))
        })
}

// T region wrappers have no provenance. Their physical
// children do: use those source sites, never a projection's allocation order,
// to namespace the unchanged body's instances and observable ids.
fn physical_region_key(node: &BloqNode) -> Option<String> {
    if let Some(key) = node.stable_key() {
        return Some(key.to_string());
    }
    let region = node.try_region()?;
    let mut sites = BTreeSet::new();
    for (arm, body) in region.bodies() {
        for (_, child) in body.nodes().filter(|(_, child)| has_quantum(child)) {
            sites.insert(format!("{arm:?}/{}", physical_region_key(child)?));
        }
    }
    (!sites.is_empty()).then(|| format!("region {sites:?}"))
}

fn merge_provenance(
    target: &mut NodeProvenance,
    added: &NodeProvenance,
) -> Result<(), CompileError> {
    match (&mut *target, added) {
        (
            NodeProvenance::BlockComponent { members },
            NodeProvenance::BlockComponent { members: added },
        ) => {
            members.extend(added);
            members.sort_unstable();
            members.dedup();
        }
        (NodeProvenance::BlockComponent { members }, _) if members.is_empty() => {
            *target = added.clone()
        }
        (current, other) if current == other => {}
        _ => {
            return Err(assembly(
                "one shared member belongs to different temporal stages",
            ));
        }
    }
    Ok(())
}

fn remap_edge(edge: &mut BloqEdge, templates: &HashMap<TemplateId, TemplateId>) {
    if let BloqEdge::Quantum(quantum) = edge {
        for seam in &mut quantum.pipes {
            if let Some(padding) = &mut seam.padding {
                padding.one_round = templates[&padding.one_round];
                padding.looped = templates[&padding.looped];
            }
        }
    }
}

fn remap_node(
    node: &mut BloqNode,
    templates: &HashMap<TemplateId, TemplateId>,
    instances: &HashMap<TemplateInstanceId, TemplateInstanceId>,
) {
    let measurement = |mut term: InstanceMeasurement| {
        term.instance = instances[&term.instance];
        term
    };
    match &mut node.kind {
        BloqNodeKind::Quantum(_) => {
            let quantum = node.expect_quantum_mut();
            for instance in &mut quantum.instances {
                instance.id = instances[&instance.id];
                instance.template_id = templates[&instance.template_id];
            }
            for detector in &mut quantum.detectors {
                detector.parity = detector.parity.clone().map_measurements(measurement);
            }
            for restart in &mut quantum.restarts {
                restart.parity = restart.parity.clone().map_measurements(measurement);
            }
        }
        BloqNodeKind::Classical(data) => {
            if data.measurements().is_empty() && data.operators().is_empty() {
                return;
            }
            let (measurements, operators): (&mut [_], &mut [_]) =
                match std::sync::Arc::make_mut(data) {
                    ClassicalNode::Observable {
                        measurements,
                        operators,
                        ..
                    } => (measurements, operators),
                    _ => unreachable!("only physical readout payloads are remapped"),
                };
            for term in measurements {
                *term = measurement(*term);
            }
            for operator in operators {
                operator.instance = instances[&operator.instance];
            }
        }
        BloqNodeKind::Region(region) => {
            for (_, body) in region.bodies_mut() {
                let ids = body.node_ids().collect::<Vec<_>>();
                for id in ids {
                    remap_node(body.node_mut(id).expect("body node"), templates, instances);
                }
                body.for_each_edge_mut(|edge| remap_edge(edge, templates));
            }
        }
    }
}

fn registration(
    bloq: &mut Bloq,
    node: BloqNodeId,
    guard: BloqNodeId,
) -> Result<&mut QuantumGuard, CompileError> {
    let slot = if let Some(input) = bloq
        .top()
        .value_inputs(node)
        .find(|input| input.producer == guard)
    {
        input.slot
    } else {
        let count = crate::add_resource(
            "quantum activation slots",
            bloq.top().value_inputs(node).count(),
            1,
            u32::MAX as usize,
        )?;
        let slot = (count - 1) as u32;
        bloq.add_edge(guard, node, BloqEdge::value(slot));
        slot
    };
    let quantum = bloq
        .node_mut(node)
        .expect("registry node")
        .expect_quantum_mut();
    let index = quantum
        .guards
        .iter()
        .position(|entry| entry.input == slot)
        .unwrap_or_else(|| {
            quantum.guards.push(QuantumGuard {
                input: slot,
                ..Default::default()
            });
            quantum.guards.len() - 1
        });
    Ok(&mut quantum.guards[index])
}

#[cfg(test)]
mod tests {
    #[test]
    fn conditional_external_templates_cannot_silently_drop_logical_cuts() {
        let mut source = include_str!("../../docs/fixtures/conditional_cz_strip.blog")
            .replace("12: ZXZ", "12: XZX");
        for bit in 0..3 {
            source = source.replace(
                &format!("in enable{bit}"),
                &format!(
                    "{}: ZXZ [4, {bit}, -2]\n  enable{bit} = measure {}",
                    80 + bit,
                    80 + bit
                ),
            );
        }
        let program = bloq_graph::lower_blog_graph_ast_deferred(
            &bloq_graph::parse_blog_program_to_ast(&source).unwrap(),
        )
        .unwrap();
        let error = crate::CompileContext::new(crate::CompileConfig::default())
            .compile(&program)
            .unwrap_err();
        assert!(
            matches!(&error, crate::CompileError::BranchAssembly { reason }
                if reason.contains("has no common physical instance")),
            "{error:?}"
        );
    }
}
