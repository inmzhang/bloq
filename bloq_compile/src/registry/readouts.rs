//! One guarded readout DAG shared by selectors, feedback, and terminal frames.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bloq_graph::{
    Action, Expr, GuardedReadoutPlan, GuardedSurfaceKind, Pauli, PauliBasis, StabilizerRowKind,
    SurfaceSupport,
};
use bloq_ir::lowering::{InstanceBoundaryOperator, InstanceMeasurement, TemplateInstanceId};
use bloq_ir::{
    BloqEdge, BloqNode, BloqNodeId, ClassicalExpr, ClassicalNode, NodeProvenance, ValueRole,
};
use bloq_utils::boolean::{BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId};
use glam::IVec3;

use super::assembly::{Assembly, BooleanProgram, Profile};
use super::physical::{GuardedDynamicAnchors, PhysicalProjection};
use crate::block::LoweringTemplateId;
use crate::lower::ChunkSiteSource;
use crate::{BlockLayout, CompileError, FxMap as HashMap, check_resource};

type MeasurementCache = HashMap<(ParityPacket, DecisionId), BloqNodeId>;
type BoundaryCache = HashMap<(InstanceBoundaryOperator, DecisionId), BloqNodeId>;
type PendingPackets = BTreeMap<TemplateInstanceId, HashMap<ParityPacket, DecisionId>>;

/// An exact, sorted parity belonging to one instance. Reusing a packet hashes
/// only its fingerprint; equality still compares content after hash collisions.
#[derive(Clone, Debug)]
struct ParityPacket {
    records: Arc<[InstanceMeasurement]>,
    hash: u64,
}

impl ParityPacket {
    fn new(
        records: &[InstanceMeasurement],
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<Self, CompileError> {
        debug_assert!(!records.is_empty());
        debug_assert!(records.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(
            records
                .iter()
                .all(|record| record.instance == records[0].instance)
        );
        diagram.charge(records.len().saturating_mul(2).saturating_add(1))?;
        let mut hasher = rustc_hash::FxHasher::default();
        records.hash(&mut hasher);
        Ok(Self {
            records: Arc::from(records),
            hash: hasher.finish(),
        })
    }

    fn instance(&self) -> TemplateInstanceId {
        self.records[0].instance
    }
}

impl Hash for ParityPacket {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for ParityPacket {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && (Arc::ptr_eq(&self.records, &other.records) || self.records == other.records)
    }
}

impl Eq for ParityPacket {}

fn accumulate_packet(
    pending: &mut PendingPackets,
    packet: &ParityPacket,
    guard: DecisionId,
    diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
) -> Result<(), CompileError> {
    diagram.charge(1)?;
    let previous = pending
        .entry(packet.instance())
        .or_default()
        .entry(packet.clone())
        .or_insert(ZERO);
    *previous = diagram.apply(BooleanOp::Xor, *previous, guard)?;
    Ok(())
}

/// Complete concrete gateway inputs. Profile identities and selected anchor
/// instances remain fixed when the Boolean arena is collected.
#[derive(PartialEq, Eq, Hash)]
struct LocalBindingKey {
    profile: usize,
    center: Pauli,
    port: Pauli,
    edges: Vec<((IVec3, IVec3), Pauli)>,
    selective: Option<bool>,
    logical: bool,
    anchor: Option<LocalBindingAnchor>,
}

#[derive(PartialEq, Eq, Hash)]
struct LocalBindingAnchor {
    touches_dynamic: bool,
    source_t: Option<(IVec3, bool, TemplateInstanceId)>,
}

/// A complete module contribution to a source query. It retains record owners,
/// boundary operators, guards, and exact gateway transport. Each consumer still
/// constructs its own complete decoder equation and corrected-parity folds.
struct FragmentBinding {
    packets: PendingPackets,
    operators: HashMap<InstanceBoundaryOperator, DecisionId>,
    offset: DecisionId,
}

struct LocalBinding {
    packets: Vec<ParityPacket>,
    boundaries: Vec<InstanceBoundaryOperator>,
    sign: bool,
}

/// Escape-gateway lookup for the T blocks among a profile set, keyed by block
/// position: the owning projection, its observable template, and the template
/// instance carrying the escape measurement.
pub(super) type TSourceGateways = HashMap<
    IVec3,
    (
        Arc<PhysicalProjection>,
        LoweringTemplateId,
        TemplateInstanceId,
    ),
>;

/// Resolve every T block in `profiles` to its escape gateway.
///
/// Only T blocks carry an escape gateway, so a non-T profile contributes
/// nothing. Shared with the binding tests so they bind against the same
/// gateways production lowering does rather than a parallel copy.
pub(super) fn t_source_gateways(profiles: &[Profile]) -> TSourceGateways {
    profiles
        .iter()
        .filter_map(|profile| {
            let physical = &profile.projection.physical;
            let block = physical.geometry.graph.get_block(profile.position)?;
            if !block.kind().is_t() {
                return None;
            }
            let template = physical.compiled[&profile.position]
                .template
                .observable_template()
                .expect("T source has an escape gateway");
            Some((
                profile.position,
                (
                    Arc::clone(physical),
                    template,
                    profile.projection.sites[&ChunkSiteSource::Block(profile.position)],
                ),
            ))
        })
        .collect()
}

#[cfg(test)]
thread_local! {
    static FORCE_COLLECTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static KEEP_QUERY_COEFFICIENTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FORCE_UNCACHED_BINDINGS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FRAGMENT_CACHE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BINDING_CACHE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn lower(
    plan: &mut GuardedReadoutPlan,
    profiles: &mut [Profile],
    assembly: &mut Assembly,
    boolean: &mut BooleanProgram,
    distance: u32,
    anchors: &mut GuardedDynamicAnchors,
    witness_limit: usize,
) -> Result<(), CompileError> {
    check_resource("readout IDs", plan.surfaces.len(), u32::MAX as usize)?;
    let mut profiles_at = HashMap::<_, Vec<_>>::default();
    for (index, profile) in profiles.iter().enumerate() {
        profiles_at
            .entry(profile.position.to_array())
            .or_default()
            .push(index);
    }
    let t_sources = t_source_gateways(profiles);
    let actions = plan.topology.source.actions();
    let named_actions = actions
        .iter()
        .enumerate()
        .filter_map(|(ordinal, action)| match action {
            Action::Measure { name, .. } => Some((name.as_str(), ordinal as u32)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut accumulates = MeasurementCache::default();
    let mut includes = BoundaryCache::default();
    let mut bindings = HashMap::<LocalBindingKey, Arc<LocalBinding>>::default();
    let mut binding_terms = 0usize;
    let mut fragments = HashMap::<usize, Arc<FragmentBinding>>::default();
    let mut fragment_terms = 0usize;
    #[cfg(test)]
    let cache_bindings = !FORCE_UNCACHED_BINDINGS.get();
    #[cfg(not(test))]
    let cache_bindings = true;
    let mut readouts = HashMap::default();
    let mut frames = HashMap::default();
    let mut pending_frames = Vec::new();
    // A representative can outlive its own readout when later queries reuse
    // one of its fragments. Keep it until the last such consumer has finished.
    plan.topology
        .diagram
        .charge(plan.surfaces.len().saturating_mul(3))?;
    let mut last_use = (0..plan.surfaces.len()).collect::<Vec<_>>();
    let mut last_fragment_use = HashMap::default();
    for consumer in 0..plan.surfaces.len() {
        plan.topology
            .diagram
            .charge(plan.query_fragments(consumer).len().saturating_mul(3))?;
        for &fragment in plan.query_fragments(consumer) {
            let representative = plan.query_fragment(fragment).0;
            last_use[representative] = last_use[representative].max(consumer);
            last_fragment_use.insert(fragment, consumer);
        }
    }
    let mut releases = vec![Vec::new(); plan.surfaces.len()];
    for (surface, consumer) in last_use.into_iter().enumerate() {
        releases[consumer].push(surface);
    }
    let collection_floor = (witness_limit / 4).max(1);
    let mut next_collection = collection_floor;
    for (index, finished_queries) in releases.into_iter().enumerate() {
        // Named values and terminal frames compare the same source instrument.
        // Removing H/Y transport here would change that instrument's branch.
        // Only closed logical observables normalize their known offset.
        let normalize_transport = matches!(
            plan.surfaces[index].kind,
            GuardedSurfaceKind::LogicalReadout
        );
        let row_kind = match &plan.surfaces[index].kind {
            GuardedSurfaceKind::Readout { name, .. } => {
                StabilizerRowKind::Measurement { name: name.clone() }
            }
            GuardedSurfaceKind::LogicalReadout | GuardedSurfaceKind::OutputFrame { .. } => {
                StabilizerRowKind::Logical
            }
        };
        let mut packets = PendingPackets::new();
        let mut operators = HashMap::<InstanceBoundaryOperator, DecisionId>::default();
        let mut reference_offset = ZERO;
        #[cfg(test)]
        let queries = if cache_bindings {
            plan.query_fragments(index)
                .iter()
                .map(|&id| (Some(id), plan.query_fragment(id)))
                .collect::<Vec<_>>()
        } else {
            vec![(
                None,
                (
                    index,
                    plan.support_sites(index).into_iter().collect::<Arc<[_]>>(),
                ),
            )]
        };
        #[cfg(not(test))]
        let queries = plan
            .query_fragments(index)
            .iter()
            .map(|&id| (Some(id), plan.query_fragment(id)))
            .collect::<Vec<_>>();
        for (id, (representative, sites)) in queries {
            let fragment = if let Some(fragment) = id.and_then(|id| fragments.get(&id)) {
                #[cfg(test)]
                FRAGMENT_CACHE_HITS.set(FRAGMENT_CACHE_HITS.get() + 1);
                Arc::clone(fragment)
            } else {
                let mut fragment = FragmentBinding {
                    packets: PendingPackets::new(),
                    operators: HashMap::default(),
                    offset: ZERO,
                };
                for &profile_index in sites
                    .iter()
                    .flat_map(|position| profiles_at.get(position).into_iter().flatten())
                {
                    // No returned local cases remain live here. The pending physical
                    // parity is included in the same remap as every stage-owned cache.
                    let collect = plan.topology.diagram.nodes().len() > next_collection;
                    #[cfg(test)]
                    let collect = collect || FORCE_COLLECTION.get();
                    if collect {
                        // Optional images own live guards. Drop them before remapping;
                        // the stable source fragment ids need no Boolean cache keys.
                        fragments.clear();
                        fragment_terms = 0;
                        collect_boolean_garbage(
                            plan,
                            profiles,
                            anchors,
                            assembly,
                            boolean,
                            &mut accumulates,
                            &mut includes,
                            packets
                                .values_mut()
                                .flat_map(HashMap::values_mut)
                                .chain(operators.values_mut())
                                .chain(std::iter::once(&mut reference_offset))
                                .chain(fragment.packets.values_mut().flat_map(HashMap::values_mut))
                                .chain(fragment.operators.values_mut())
                                .chain(std::iter::once(&mut fragment.offset)),
                        )?;
                        let retained = plan.topology.diagram.nodes().len();
                        check_resource(
                            "live physical readout Boolean nodes",
                            retained,
                            witness_limit,
                        )?;
                        next_collection = retained
                            .saturating_add(witness_limit.saturating_sub(retained) / 4)
                            .max(collection_floor);
                    }
                    let profile = &profiles[profile_index];
                    let physical = &profile.projection.physical;
                    let context = crate::lower::LocalSurfaceContext {
                        graph: &physical.geometry.graph,
                        compiled: &physical.compiled,
                        spatial_ports: &physical.geometry.spatial_ports,
                        spatial_templates: &physical.spatial_templates,
                        wall_templates: &physical.wall_templates,
                        pipe_templates: &physical.pipe_templates,
                        pool: &physical.pool,
                        instances: &profile.projection.sites,
                        selectives: &profile.projection.selectives,
                        layout: BlockLayout::new(distance),
                    };
                    for case in plan.local_cases(representative, profile.position, profile.guard)? {
                        let alternatives =
                            if context.needs_dynamic_anchor(profile.position, &case.surface) {
                                anchors
                                    .cases(&mut plan.topology, profile.position, case.guard)?
                                    .into_iter()
                                    .map(|(guard, anchor)| (guard, Some(anchor)))
                                    .collect()
                            } else {
                                vec![(case.guard, None)]
                            };
                        for (guard, anchor) in alternatives {
                            let anchor = anchor.map(|anchor| crate::lower::BoundDynamicAnchor {
                                touches_dynamic: anchor.touches_dynamic,
                                source_t: anchor.source_t.map(|(source, flipped)| {
                                    let (physical, template, instance) = &t_sources[&source];
                                    crate::lower::TSourceGateway {
                                        source,
                                        flipped,
                                        gateway: &physical.pool[*template].observable_gateway,
                                        instance: *instance,
                                    }
                                }),
                            });
                            let edge_count = case.surface.edges().count();
                            plan.topology.diagram.charge(1usize.saturating_add(
                                edge_count.saturating_mul(edge_count.max(1).ilog2() as usize + 3),
                            ))?;
                            let mut edges = case.surface.edges().collect::<Vec<_>>();
                            edges.sort_unstable_by_key(|&((source, target), pauli)| {
                                (source.to_array(), target.to_array(), pauli as u8)
                            });
                            let key = LocalBindingKey {
                                profile: profile_index,
                                center: case.surface.node_pauli(profile.position),
                                port: case.surface.port_pauli(profile.position),
                                edges,
                                selective: case.selective,
                                logical: matches!(row_kind, StabilizerRowKind::Logical),
                                anchor: anchor.map(|anchor| LocalBindingAnchor {
                                    touches_dynamic: anchor.touches_dynamic,
                                    source_t: anchor.source_t.map(|source| {
                                        (source.source, source.flipped, source.instance)
                                    }),
                                }),
                            };
                            let cached = cache_bindings.then(|| bindings.get(&key)).flatten();
                            let binding = if let Some(binding) = cached {
                                #[cfg(test)]
                                BINDING_CACHE_HITS.set(BINDING_CACHE_HITS.get() + 1);
                                Arc::clone(binding)
                            } else {
                                // Cache successes only: errors retain this readout's
                                // original index and measurement name.
                                let (measurements, boundaries, sign) = context.resolve(
                                    &[profile.position],
                                    &case.surface,
                                    case.selective.map(|selected| (profile.position, selected)),
                                    &row_kind,
                                    index as u32,
                                    anchor,
                                )?;
                                plan.topology.diagram.charge(measurements.len())?;
                                let packets = measurements
                                    .chunk_by(|left, right| left.instance == right.instance)
                                    .map(|records| {
                                        ParityPacket::new(records, &mut plan.topology.diagram)
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let binding = Arc::new(LocalBinding {
                                    packets,
                                    boundaries,
                                    sign,
                                });
                                let terms = 1usize
                                    .saturating_add(key.edges.len())
                                    .saturating_add(binding.packets.iter().fold(
                                        0usize,
                                        |sum, packet| {
                                            sum.saturating_add(1)
                                                .saturating_add(packet.records.len())
                                        },
                                    ))
                                    .saturating_add(binding.boundaries.iter().fold(
                                        0usize,
                                        |sum, operator| {
                                            sum.saturating_add(1)
                                                .saturating_add(operator.operator.len())
                                        },
                                    ));
                                plan.topology.diagram.charge(terms)?;
                                // ponytail: bound payload terms; add eviction only if
                                // measured workloads repeatedly fill this cache.
                                const MAX_BINDING_TERMS: usize = 262_144;
                                if cache_bindings && terms <= MAX_BINDING_TERMS {
                                    if binding_terms + terms > MAX_BINDING_TERMS {
                                        plan.topology.diagram.charge(binding_terms)?;
                                        bindings.clear();
                                        binding_terms = 0;
                                    }
                                    if bindings.len() == bindings.capacity() {
                                        plan.topology.diagram.charge(binding_terms)?;
                                    }
                                    bindings.insert(key, Arc::clone(&binding));
                                    binding_terms += terms;
                                }
                                binding
                            };
                            if binding.sign {
                                fragment.offset = plan.topology.diagram.apply(
                                    BooleanOp::Xor,
                                    fragment.offset,
                                    guard,
                                )?;
                            }
                            for packet in &binding.packets {
                                accumulate_packet(
                                    &mut fragment.packets,
                                    packet,
                                    guard,
                                    &mut plan.topology.diagram,
                                )?;
                            }
                            for operator in &binding.boundaries {
                                plan.topology.diagram.charge(operator.operator.len())?;
                                let previous =
                                    fragment.operators.entry(operator.clone()).or_insert(ZERO);
                                *previous = plan.topology.diagram.apply(
                                    BooleanOp::Xor,
                                    *previous,
                                    guard,
                                )?;
                            }
                        }
                    }
                }
                let terms = fragment
                    .packets
                    .values()
                    .flat_map(HashMap::keys)
                    .fold(1usize, |sum, packet| {
                        sum.saturating_add(packet.records.len()).saturating_add(1)
                    })
                    .saturating_add(fragment.operators.keys().fold(0usize, |sum, operator| {
                        sum.saturating_add(operator.operator.len())
                            .saturating_add(1)
                    }));
                plan.topology.diagram.charge(terms)?;
                let fragment = Arc::new(fragment);
                // Shared payloads are optional. Oversized responses bind with
                // the same topology and cumulative resource budget.
                const MAX_FRAGMENT_TERMS: usize = 262_144;
                // A response with no future consumer cannot produce a cache
                // hit. Keep capacity for reusable fragments instead.
                if let Some(id) = id.filter(|id| {
                    cache_bindings && terms <= MAX_FRAGMENT_TERMS && last_fragment_use[id] > index
                }) {
                    if terms > MAX_FRAGMENT_TERMS.saturating_sub(fragment_terms) {
                        plan.topology.diagram.charge(fragment_terms)?;
                        fragments.clear();
                        fragment_terms = 0;
                    }
                    if fragments.len() == fragments.capacity() {
                        plan.topology.diagram.charge(fragment_terms)?;
                    }
                    fragments.insert(id, Arc::clone(&fragment));
                    fragment_terms += terms;
                }
                fragment
            };
            for groups in fragment.packets.values() {
                for (packet, &guard) in groups {
                    accumulate_packet(&mut packets, packet, guard, &mut plan.topology.diagram)?;
                }
            }
            for (operator, &guard) in &fragment.operators {
                plan.topology.diagram.charge(operator.operator.len())?;
                let previous = operators.entry(operator.clone()).or_insert(ZERO);
                *previous = plan
                    .topology
                    .diagram
                    .apply(BooleanOp::Xor, *previous, guard)?;
            }
            if normalize_transport {
                reference_offset = plan.topology.diagram.apply(
                    BooleanOp::Xor,
                    reference_offset,
                    fragment.offset,
                )?;
            }
        }
        // Cache hits can also allocate XOR guards while combining fragments.
        // No fragment image remains borrowed at this query boundary.
        if plan.topology.diagram.nodes().len() > next_collection {
            fragments.clear();
            fragment_terms = 0;
            collect_boolean_garbage(
                plan,
                profiles,
                anchors,
                assembly,
                boolean,
                &mut accumulates,
                &mut includes,
                packets
                    .values_mut()
                    .flat_map(HashMap::values_mut)
                    .chain(operators.values_mut())
                    .chain(std::iter::once(&mut reference_offset)),
            )?;
            let retained = plan.topology.diagram.nodes().len();
            check_resource(
                "live physical readout Boolean nodes",
                retained,
                witness_limit,
            )?;
            next_collection = retained
                .saturating_add(witness_limit.saturating_sub(retained) / 4)
                .max(collection_floor);
        }
        let mut values = emit_accumulates(
            packets,
            &mut accumulates,
            &mut plan.topology,
            assembly,
            boolean,
        )?;
        let mut operators = operators.into_iter().collect::<Vec<_>>();
        operators.sort_by_key(|(operator, guard)| {
            (
                operator.instance.0,
                format!("{:?}", operator.face),
                format!("{}", operator.operator),
                guard.0,
            )
        });
        for (operator, guard) in operators {
            let guard = plan
                .topology
                .diagram
                .constrain(guard, plan.topology.domain)?;
            if guard == ZERO {
                continue;
            }
            let key = (operator.clone(), guard);
            let node = if let Some(&node) = includes.get(&key) {
                node
            } else {
                let node = emit_guarded_value(
                    assembly,
                    boolean,
                    &plan.topology,
                    guard,
                    ClassicalNode::observable_fragment(vec![], vec![operator.clone()]),
                );
                if operator.face == bloq_ir::BoundaryFace::Output {
                    assembly.bloq.add_edge(
                        assembly.owners[&operator.instance],
                        node,
                        BloqEdge::Order,
                    );
                }
                includes.insert(key, node);
                node
            };
            values.push((node, ValueRole::Data));
        }
        let kind = plan.surfaces[index].kind.clone();
        let feedbacks = plan.surfaces[index].feedbacks.clone();
        for (ordinal, guard) in feedbacks {
            let node = boolean.emit(guard, &plan.topology.diagram, &mut assembly.bloq);
            values.push((node, ValueRole::FeedbackFold { action: ordinal }));
        }
        match kind {
            GuardedSurfaceKind::Readout { name, folds } => {
                for (predecessor, guard) in folds {
                    let value = plan.topology.condition(&Expr::Var(predecessor));
                    let value = plan.topology.diagram.apply(BooleanOp::And, guard, value)?;
                    values.push((
                        boolean.emit(value, &plan.topology.diagram, &mut assembly.bloq),
                        ValueRole::ReadoutFold,
                    ));
                }
                let observable = emit_readout(assembly, values)?;
                let ordinal = named_actions[name.as_str()];
                assembly
                    .bloq
                    .node_mut(observable)
                    .expect("newly emitted readout node exists")
                    .provenance = NodeProvenance::Action { ordinal };
                readouts.insert(name, observable);
            }
            GuardedSurfaceKind::OutputFrame { port, basis, .. } => {
                let inputs = if plan.surfaces[index].activation() != ZERO {
                    let observable = emit_readout(assembly, values)?;
                    // The complete observable already supplies the corrected
                    // physical parity of this frame equation.
                    vec![observable]
                } else {
                    // A prepared output can lack an axis relation. Retain its
                    // zero frame without inventing an empty decoder problem.
                    debug_assert!(values.is_empty());
                    Vec::new()
                };
                let expr = ClassicalExpr::parity(0..inputs.len() as u32, false);
                let node = assembly.bloq.add_node(
                    BloqNode::classical(ClassicalNode::Compute { expr }).with_provenance(
                        NodeProvenance::OutputFrame {
                            port,
                            basis: if basis == PauliBasis::X {
                                bloq_ir::Basis::X
                            } else {
                                bloq_ir::Basis::Z
                            },
                        },
                    ),
                );
                frames.insert((port, basis), node);
                pending_frames.push((index, node, inputs.len() as u32));
                for (slot, producer) in inputs.into_iter().enumerate() {
                    assembly
                        .bloq
                        .add_edge(producer, node, BloqEdge::value(slot as u32));
                }
            }
            GuardedSurfaceKind::LogicalReadout => {
                if reference_offset != ZERO {
                    values.push((
                        boolean.emit(reference_offset, &plan.topology.diagram, &mut assembly.bloq),
                        ValueRole::Data,
                    ));
                }
                emit_readout(assembly, values)?;
            }
        }
        #[cfg(test)]
        let release = !KEEP_QUERY_COEFFICIENTS.get();
        #[cfg(not(test))]
        let release = true;
        if release {
            for surface in finished_queries {
                plan.release_query_coefficients(surface)?;
            }
        }
    }
    // Each physical equation retains its own raw/corrected decoder pair.
    // Back-substitute output dependencies as shared classical values instead
    // of expanding their physical witnesses into every preceding equation.
    for (index, node, mut slot) in pending_frames.into_iter().rev() {
        let GuardedSurfaceKind::OutputFrame { folds, .. } = &plan.surfaces[index].kind else {
            unreachable!()
        };
        plan.topology.diagram.charge(folds.len())?;
        let mut terms = vec![ClassicalExpr::parity(0..slot, false)];
        for &(port, basis, guard) in folds {
            let producer = frames[&(port, basis)];
            assembly
                .bloq
                .add_edge(producer, node, BloqEdge::value(slot));
            let value = ClassicalExpr::In(slot);
            slot += 1;
            let value = if guard == ONE {
                value
            } else {
                let predicate = boolean.emit(guard, &plan.topology.diagram, &mut assembly.bloq);
                assembly
                    .bloq
                    .add_edge(predicate, node, BloqEdge::value(slot));
                slot += 1;
                ClassicalExpr::And(Box::new([ClassicalExpr::In(slot - 1), value]))
            };
            terms.push(value);
        }
        let mut expression = ClassicalExpr::xor(terms);
        let active = plan.surfaces[index].activation();
        if active == ZERO {
            expression = ClassicalExpr::Const(false);
        } else if active != ONE {
            let predicate = boolean.emit(active, &plan.topology.diagram, &mut assembly.bloq);
            assembly
                .bloq
                .add_edge(predicate, node, BloqEdge::value(slot));
            expression = ClassicalExpr::And(Box::new([ClassicalExpr::In(slot), expression]));
        }
        assembly
            .bloq
            .node_mut(node)
            .expect("newly emitted frame node exists")
            .kind =
            bloq_ir::BloqNodeKind::Classical(ClassicalNode::Compute { expr: expression }.into());
    }
    for action in &actions {
        if let Action::DiscardIf(expr) = action {
            let value = boolean.emit(
                plan.topology.condition(expr),
                &plan.topology.diagram,
                &mut assembly.bloq,
            );
            let discard = assembly
                .bloq
                .add_node(BloqNode::classical(ClassicalNode::Discard {
                    condition: ClassicalExpr::In(0),
                }));
            assembly.bloq.add_edge(value, discard, BloqEdge::value(0));
        }
    }
    boolean.bind(&plan.topology, &readouts, &mut assembly.bloq)?;
    Ok(())
}

/// The physical readout stage owns every decision handle below. Hash keys must
/// leave their maps during remapping; IR node/instance identities do not change.
#[expect(
    clippy::too_many_arguments,
    reason = "all remapped owners must move through one garbage-collection boundary"
)]
fn collect_boolean_garbage<'a>(
    plan: &mut GuardedReadoutPlan,
    profiles: &mut [Profile],
    anchors: &mut GuardedDynamicAnchors,
    assembly: &mut Assembly,
    boolean: &mut BooleanProgram,
    accumulates: &mut MeasurementCache,
    includes: &mut BoundaryCache,
    pending: impl Iterator<Item = &'a mut DecisionId>,
) -> Result<(), CompileError> {
    #[cfg(test)]
    COLLECTIONS.set(COLLECTIONS.get() + 1);
    let mut pending = pending
        .map(|root| {
            plan.topology.diagram.charge(1)?;
            Ok(root)
        })
        .collect::<Result<Vec<_>, bloq_utils::boolean::BooleanResourceError>>()?;
    let copied_roots = boolean
        .values
        .len()
        .saturating_add(accumulates.len())
        .saturating_add(includes.len())
        .saturating_add(profiles.len())
        .saturating_add(anchors.decisions_mut().count())
        .saturating_add(assembly.decisions_mut().count());
    let owned_work = plan.collection_work();
    plan.topology
        .diagram
        .charge(owned_work.saturating_add(copied_roots))?;
    let mut values = std::mem::take(&mut boolean.values)
        .into_iter()
        .collect::<Vec<_>>();
    let mut accumulates_owned = std::mem::take(accumulates).into_iter().collect::<Vec<_>>();
    let mut includes_owned = std::mem::take(includes).into_iter().collect::<Vec<_>>();
    plan.collect_garbage(
        values
            .iter_mut()
            .map(|(guard, _)| guard)
            .chain(accumulates_owned.iter_mut().map(|((_, guard), _)| guard))
            .chain(includes_owned.iter_mut().map(|((_, guard), _)| guard))
            .chain(profiles.iter_mut().map(|profile| &mut profile.guard))
            .chain(anchors.decisions_mut())
            .chain(assembly.decisions_mut())
            .chain(pending.iter_mut().map(|guard| &mut **guard)),
    );
    boolean.values = values.into_iter().collect();
    *accumulates = accumulates_owned.into_iter().collect();
    *includes = includes_owned.into_iter().collect();
    Ok(())
}

// Equal complete packets combine their guards before touching records. Distinct
// packets from one instance may overlap, so only that instance needs the full
// record-level cancellation. Constrain the final coefficient, after every XOR.
fn normalize_packets(
    pending: PendingPackets,
    topology: &mut bloq_graph::GuardedTopology,
) -> Result<Vec<(ParityPacket, DecisionId)>, CompileError> {
    let mut groups = Vec::new();
    for packets in pending.into_values() {
        topology.diagram.charge(packets.len())?;
        let mut packets = packets
            .into_iter()
            .filter(|(_, guard)| *guard != ZERO)
            .collect::<Vec<_>>();
        #[cfg(test)]
        let reuse_packet = !FORCE_UNCACHED_BINDINGS.get();
        #[cfg(not(test))]
        let reuse_packet = true;
        if reuse_packet && packets.len() == 1 {
            let (packet, guard) = packets.pop().expect("one nonzero packet");
            let guard = topology.diagram.constrain(guard, topology.domain)?;
            if guard != ZERO {
                groups.push((packet, guard));
            }
            continue;
        }
        let mut records = BTreeMap::<InstanceMeasurement, DecisionId>::new();
        for (packet, guard) in packets {
            topology.diagram.charge(packet.records.len())?;
            for &record in packet.records.iter() {
                let previous = records.entry(record).or_insert(ZERO);
                *previous = topology.diagram.apply(BooleanOp::Xor, *previous, guard)?;
            }
        }
        let mut by_guard = BTreeMap::<DecisionId, Vec<InstanceMeasurement>>::new();
        topology.diagram.charge(records.len())?;
        for (record, guard) in records {
            let guard = topology.diagram.constrain(guard, topology.domain)?;
            if guard != ZERO {
                by_guard.entry(guard).or_default().push(record);
            }
        }
        for (guard, records) in by_guard {
            groups.push((ParityPacket::new(&records, &mut topology.diagram)?, guard));
        }
    }
    // Boolean allocation ids change during collection; physical records keep
    // the emitted readout's XOR operand order stable across that remapping.
    topology.diagram.charge(
        groups
            .len()
            .saturating_mul(groups.len().max(1).ilog2() as usize + 1),
    )?;
    groups.sort_unstable_by(|left, right| left.0.records.cmp(&right.0.records));
    Ok(groups)
}

// Share each instance's exact, fully cancelled physical parity across readouts.
fn emit_accumulates(
    packets: PendingPackets,
    accumulates: &mut MeasurementCache,
    topology: &mut bloq_graph::GuardedTopology,
    assembly: &mut Assembly,
    boolean: &mut BooleanProgram,
) -> Result<Vec<(BloqNodeId, ValueRole)>, CompileError> {
    let mut values = Vec::new();
    for (packet, guard) in normalize_packets(packets, topology)? {
        let key = (packet, guard);
        let node = if let Some(&node) = accumulates.get(&key) {
            node
        } else {
            topology.diagram.charge(key.0.records.len())?;
            let node = emit_guarded_value(
                assembly,
                boolean,
                topology,
                guard,
                ClassicalNode::observable_fragment(key.0.records.to_vec(), vec![]),
            );
            let owner = assembly.owners[&key.0.instance()];
            assembly.bloq.add_edge(owner, node, BloqEdge::Order);
            accumulates.insert(key, node);
            node
        };
        values.push((node, ValueRole::Data));
    }
    Ok(values)
}

pub(super) fn emit_readout(
    assembly: &mut Assembly,
    values: Vec<(BloqNodeId, ValueRole)>,
) -> Result<BloqNodeId, CompileError> {
    check_resource("readout input slots", values.len(), u32::MAX as usize)?;
    let index = assembly.allocate_observable()?;
    let observable = assembly.bloq.add_node(
        BloqNode::classical(ClassicalNode::observable(index))
            .with_provenance(NodeProvenance::Generator { ordinal: index }),
    );
    for (slot, (producer, role)) in values.into_iter().enumerate() {
        let edge = if role == ValueRole::Data
            && matches!(
                assembly.bloq[producer].try_classical(),
                Some(ClassicalNode::Observable { index: None, .. })
            ) {
            BloqEdge::compose(slot as u32)
        } else {
            BloqEdge::Value {
                slot: slot as u32,
                role,
                output: Default::default(),
            }
        };
        assembly.bloq.add_edge(producer, observable, edge);
    }
    Ok(observable)
}

fn emit_guarded_value(
    assembly: &mut Assembly,
    boolean: &mut BooleanProgram,
    topology: &bloq_graph::GuardedTopology,
    guard: DecisionId,
    value: ClassicalNode,
) -> BloqNodeId {
    let mut node = BloqNode::classical(value);
    if guard != ONE {
        // SEM-ACTIVATE gates both record reads and boundary bindings.
        node.activation = Some(0);
    }
    let node = assembly.bloq.add_node(node);
    if guard != ONE {
        let predicate = boolean.emit(guard, &topology.diagram, &mut assembly.bloq);
        assembly.bloq.add_edge(predicate, node, BloqEdge::value(0));
    }
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::CoordCircuit;
    use bloq_graph::{BlockGraph, GuardedTopology, GuardedVariable, ModuleCertificationLimits};
    use bloq_ir::lowering::{BloqTemplate, TemplateInstance};
    use glam::ivec2;

    #[test]
    fn unindexed_observable_folds_remain_boolean() {
        for role in [
            ValueRole::ReadoutFold,
            ValueRole::FeedbackFold { action: 0 },
        ] {
            let mut assembly = Assembly::new();
            let fragment = assembly
                .bloq
                .add_node(BloqNode::classical(ClassicalNode::fragment()));
            let observable = emit_readout(&mut assembly, vec![(fragment, role.clone())]).unwrap();
            assembly.bloq.validate().unwrap();
            let input = assembly.bloq.value_inputs(observable).next().unwrap();
            assert_eq!(*input.role, role);
            assert_eq!(input.output, Some(bloq_ir::ObservableOutput::Corrected));
        }
    }

    #[test]
    fn exhausted_observable_ids_fail_before_readout_emission() {
        let mut assembly = Assembly::new();
        assembly.next_observable = u32::MAX - 1;
        let observable = emit_readout(&mut assembly, Vec::new()).unwrap();
        assert!(matches!(
            assembly.bloq[observable].try_classical(),
            Some(ClassicalNode::Observable { index, .. }) if *index == Some(u32::MAX - 1)
        ));
        let nodes = assembly.bloq.node_count();
        assert!(matches!(
            emit_readout(&mut assembly, Vec::new()),
            Err(CompileError::BooleanResource(error))
                if error.resource == "observable IDs" && error.limit == u32::MAX as usize
        ));
        assert_eq!(assembly.next_observable, u32::MAX);
        assert_eq!(assembly.bloq.node_count(), nodes);
    }

    #[test]
    fn output_frames_keep_decode_availability_and_later_folds() {
        let program = bloq_graph::GalleryItem::ThreeBitAdder.build();
        let config = crate::CompileConfig::new(3);
        let bloq = crate::CompileContext::new(config)
            .compile(&program)
            .unwrap()
            .bloq;
        let mut decoded_frames = 0;
        let mut later_folds = 0;
        for (id, node) in bloq.top().nodes() {
            if !matches!(node.provenance, NodeProvenance::OutputFrame { .. }) {
                continue;
            }
            let Some(ClassicalNode::Compute { expr }) = node.try_classical() else {
                panic!("output frame stays a Compute");
            };
            let mut decode_slot = None;
            let mut fold_slots = Vec::new();
            for input in bloq.top().value_inputs(id) {
                match bloq[input.producer].try_classical() {
                    Some(ClassicalNode::Observable { index: Some(_), .. }) => {
                        assert_eq!(input.output, Some(bloq_ir::ObservableOutput::Corrected));
                        assert!(decode_slot.replace(input.slot).is_none());
                    }
                    Some(ClassicalNode::Observable { index: None, .. }) => {
                        panic!("frame reads parity outside its complete Observable")
                    }
                    _ => {}
                }
                if matches!(
                    bloq[input.producer].provenance,
                    NodeProvenance::OutputFrame { .. }
                ) {
                    fold_slots.push(input.slot);
                }
            }
            if let Some(slot) = decode_slot {
                decoded_frames += 1;
                assert_eq!(
                    expr.eval(&mut |input| (input != slot).then_some(false)),
                    None,
                    "a missing Observable stays unavailable with other inputs false"
                );
                let eval = |changed, value| {
                    expr.eval(&mut |input| Some(if input == changed { value } else { true }))
                        .unwrap()
                };
                if eval(slot, false) != eval(slot, true) {
                    for fold in fold_slots {
                        assert_ne!(eval(fold, false), eval(fold, true));
                        later_folds += 1;
                    }
                }
            }
        }
        assert!(decoded_frames > 0);
        assert!(later_folds > 0);
    }

    #[test]
    fn t_gate_merges_each_measurement_branch_into_one_fragment() {
        let program = crate::compile(&bloq_graph::GalleryItem::T.build(), 3).unwrap();
        program.validate().unwrap();
        let fragments = program
            .top()
            .nodes()
            .filter(|(_, node)| {
                matches!(
                    node.try_classical(),
                    Some(ClassicalNode::Observable { index: None, .. })
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(fragments.len(), 2, "one fragment per X/Y branch");
        for (id, node) in fragments {
            let recipe = node.try_classical().unwrap();
            assert!(!recipe.measurements().is_empty());
            assert!(!recipe.operators().is_empty());
            assert!(node.activation.is_some());
            assert_eq!(program.top().compose_consumers(id).count(), 1);
        }
    }

    #[test]
    fn query_retirement_preserves_shared_fragment_replay() {
        let source = bloq_test::benchmark::controlled_adder(10);
        let compile =
            || crate::CompileContext::new(crate::CompileConfig::default()).compile(&source);
        KEEP_QUERY_COEFFICIENTS.set(true);
        let expected = compile();
        KEEP_QUERY_COEFFICIENTS.set(false);
        let actual = compile().unwrap().bloq;
        assert_eq!(actual.to_binary(), expected.unwrap().bloq.to_binary());
    }

    #[test]
    fn collection_between_every_local_binding_preserves_compiled_recipes() {
        BINDING_CACHE_HITS.set(0);
        FRAGMENT_CACHE_HITS.set(0);
        for gallery in [
            bloq_graph::GalleryItem::THTH,
            bloq_graph::GalleryItem::CCZGateTeleport,
            bloq_graph::GalleryItem::ThreeBitAdder,
        ] {
            let program = gallery.build();
            let compile = || {
                let config = crate::CompileConfig::default();
                crate::CompileContext::new(config).compile(&program)
            };
            FORCE_UNCACHED_BINDINGS.set(true);
            let expected = compile();
            FORCE_UNCACHED_BINDINGS.set(false);
            let expected = expected.unwrap().bloq;
            let shared = compile().unwrap().bloq;
            assert!(
                shared.to_binary() == expected.to_binary(),
                "{gallery:?}: fragment sharing changed IR"
            );
            COLLECTIONS.set(0);
            FORCE_COLLECTION.set(true);
            let actual = compile();
            FORCE_COLLECTION.set(false);
            let actual = actual.unwrap().bloq;
            assert!(COLLECTIONS.get() > 2);
            assert!(
                actual.to_binary() == expected.to_binary(),
                "{gallery:?}: collection changed IR"
            );
        }
        assert!(BINDING_CACHE_HITS.get() > 0);
        assert!(FRAGMENT_CACHE_HITS.get() > 0);
    }

    #[test]
    fn overlapping_packets_cancel_exact_guards_and_keep_hash_collisions_distinct() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        for name in ["a", "b"] {
            topology
                .variables
                .push(GuardedVariable::Outcome(name.into()));
        }
        let a = topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let b = topology.diagram.make_node(1, ZERO, ONE).unwrap();
        let record = |instance, measurement| InstanceMeasurement {
            instance: TemplateInstanceId(instance),
            measurement,
        };
        let left = ParityPacket::new(&[record(0, 0), record(0, 1)], &mut topology.diagram).unwrap();
        let mut right =
            ParityPacket::new(&[record(0, 1), record(0, 2)], &mut topology.diagram).unwrap();
        right.hash = left.hash;
        assert_ne!(left, right, "a fingerprint is not packet identity");
        let other_owner =
            ParityPacket::new(&[record(1, 0), record(1, 1)], &mut topology.diagram).unwrap();
        let mut pending = PendingPackets::new();
        // Three equal contributions retain one; the other packet overlaps only
        // at record 1, whose final coefficient must be a XOR b.
        for _ in 0..3 {
            accumulate_packet(&mut pending, &left, a, &mut topology.diagram).unwrap();
        }
        accumulate_packet(&mut pending, &right, b, &mut topology.diagram).unwrap();
        accumulate_packet(&mut pending, &other_owner, a, &mut topology.diagram).unwrap();
        assert_eq!(pending[&TemplateInstanceId(0)].len(), 2);
        let restricted = topology.diagram.apply(BooleanOp::Or, a, b).unwrap();
        for domain in [ONE, restricted] {
            topology.domain = domain;
            let groups = normalize_packets(pending.clone(), &mut topology).unwrap();
            for assignment in 0..4 {
                let evaluate = |root| {
                    topology
                        .diagram
                        .evaluate(root, |variable| assignment & (1 << variable) != 0)
                };
                if !evaluate(domain) {
                    continue;
                }
                let mut actual = BTreeMap::<_, bool>::new();
                for (packet, guard) in &groups {
                    for &record in packet.records.iter() {
                        *actual.entry(record).or_default() ^= evaluate(*guard);
                    }
                }
                for (record, expected) in [
                    (record(0, 0), evaluate(a)),
                    (record(0, 1), evaluate(a) ^ evaluate(b)),
                    (record(0, 2), evaluate(b)),
                    (record(1, 0), evaluate(a)),
                    (record(1, 1), evaluate(a)),
                ] {
                    assert_eq!(actual.get(&record).copied().unwrap_or(false), expected);
                }
            }
        }
    }

    #[test]
    fn readouts_share_instance_parities_after_cancellation_with_exact_activation() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        topology
            .variables
            .push(GuardedVariable::Outcome("enabled".into()));
        let guard = topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let mut assembly = Assembly::new();
        let mut circuit = CoordCircuit::new();
        circuit.measure(bloq_circuit::PauliBasis::Z, [ivec2(0, 0), ivec2(1, 0)]);
        let template = assembly.bloq.add_template(BloqTemplate::new(circuit));
        let mut source = BloqNode::from_members(Vec::new());
        for instance in 0..2 {
            source
                .expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(instance),
                    template,
                    ivec2(instance as i32 * 10, 0),
                ));
        }
        let source = assembly.bloq.add_node(source);
        for instance in 0..2 {
            assembly.owners.insert(TemplateInstanceId(instance), source);
        }
        let enabled = assembly
            .bloq
            .add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            }));
        let mut boolean = assembly.build(&mut topology).unwrap();
        let mut cache = HashMap::default();
        let record = |instance, measurement| InstanceMeasurement {
            instance: TemplateInstanceId(instance),
            measurement,
        };
        let mut observables = Vec::new();
        for last in 0..2 {
            let mut packets = PendingPackets::new();
            for records in [
                vec![record(0, 0), record(0, 1)],
                vec![record(1, 0), record(1, 1)],
                vec![record(1, 1 - last)],
            ] {
                let packet = ParityPacket::new(&records, &mut topology.diagram).unwrap();
                accumulate_packet(&mut packets, &packet, guard, &mut topology.diagram).unwrap();
            }
            let mut values = emit_accumulates(
                packets,
                &mut cache,
                &mut topology,
                &mut assembly,
                &mut boolean,
            )
            .unwrap();
            values.push((enabled, ValueRole::ReadoutFold));
            observables.push(emit_readout(&mut assembly, values).unwrap());
        }
        boolean
            .bind(
                &topology,
                &[("enabled".into(), enabled)].into_iter().collect(),
                &mut assembly.bloq,
            )
            .unwrap();
        assembly.bloq.optimize().expect("acyclic test program");
        assembly.bloq.validate().unwrap();
        let chunks = assembly
            .bloq
            .top()
            .nodes()
            .filter_map(|(id, node)| {
                if let Some(ClassicalNode::Observable {
                    index: None,
                    measurements,
                    ..
                }) = node.try_classical()
                {
                    Some((id, node, measurements))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            chunks
                .iter()
                .map(|(_, _, records)| records.len())
                .sum::<usize>(),
            4
        );
        let &(shared, node, records) = chunks
            .iter()
            .find(|(_, _, records)| records.len() == 2)
            .unwrap();
        assert_eq!(records, &[record(0, 0), record(0, 1)]);
        assert!(node.activation.is_some());
        assert_eq!(
            assembly
                .bloq
                .top()
                .outgoing(shared)
                .filter(|edge| matches!(
                    edge.edge,
                    BloqEdge::Compose {
                        role: ValueRole::Data,
                        ..
                    }
                ))
                .count(),
            2
        );
        for observable in observables {
            assert!(
                assembly
                    .bloq
                    .top()
                    .value_inputs(observable)
                    .any(|input| *input.role == ValueRole::ReadoutFold)
            );
        }
    }
}
