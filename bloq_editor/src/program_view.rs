//! Builds the circuit viewer's presentation model from a compiled `Bloq`:
//! per-node tiles (quantum, classical, and region container nodes), the edges
//! between them, and each quantum node's flattened moment timeline.
//!
//! Region nodes recurse — their body nodes are lifted into the flat view with
//! fresh view ids and a `parent` link so the renderer can box them inside the
//! region tile.

use std::collections::{HashMap, HashSet};

use bloq_circuit::{CoordCircuit, Op, Pauli, PauliBasis};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, ClassicalNode, LevelPath, MomentLane, MomentSegment,
    MomentSegmenter, NodeProvenance, NodeRef, RegionNode, align_moment_lanes,
    aligned_moment_segments,
};
use color_eyre::eyre::{self, ContextCompat, WrapErr};
use glam::{IVec2, IVec3};

use crate::components::GraphElement;
use crate::resources::{
    BloqEdgeKind, BloqEdgeView, BloqNodeCategory, BloqNodeView, FlatMoment, FlatMomentOp,
    LayerCircuitView,
};
use bloq_graph::BlockGraph;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BloqCircuitView {
    pub(crate) nodes: Vec<BloqNodeView>,
    pub(crate) edges: Vec<BloqEdgeView>,
    /// Maps each quantum/classical node's tape-space [`NodeRef`] to its view id.
    /// Region container tiles are excluded (they own no detector slices). The
    /// detector-slice wiring uses this to key the Layer-2 `per_node` map by view
    /// id; only meaningful on a flattened program, whose `NodeRef` space matches
    /// [`bloq_ir::program_detector_slices`].
    pub(crate) node_refs: HashMap<NodeRef, u32>,
}

pub(crate) fn build_bloq_circuit_view(
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
) -> eyre::Result<BloqCircuitView> {
    build_view(program, source_graph, source_offset, true)
}

pub(crate) fn build_detslice_circuit_view(
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
) -> eyre::Result<BloqCircuitView> {
    build_view(program, source_graph, source_offset, false)
}

/// Build source-z concurrent circuit views.
pub(crate) fn build_concurrent_layer_views(
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
) -> (HashMap<i32, LayerCircuitView>, Option<String>) {
    match try_build_concurrent_layer_views(program, source_graph, source_offset) {
        Ok(views) => {
            let reason = views
                .is_empty()
                .then(|| "Program has no concurrent circuits".into());
            (views, reason)
        }
        Err(error) => (HashMap::new(), Some(format!("{error:#}"))),
    }
}

struct ConcurrentNodeCircuit {
    segments: Vec<Option<MomentSegment>>,
    slice_moments: Vec<Option<usize>>,
    qubits: Vec<IVec2>,
}

fn try_build_concurrent_layer_views(
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
) -> eyre::Result<HashMap<i32, LayerCircuitView>> {
    let mut flat = program.clone();
    flat.flatten()
        .wrap_err("flatten program for concurrent ops")?;

    let mut lanes = Vec::new();
    let mut node_circuits = HashMap::new();
    for (id, node) in flat.nodes() {
        // Regions remain in the graph as causal barriers.
        if node.try_quantum().is_none() {
            continue;
        }
        let circuit = node
            .instantiate_circuit(flat.templates())
            .wrap_err_with(|| format!("instantiate concurrent node N{}", id.0))?;
        let body = circuit
            .body(circuit.entry_body())
            .expect("instantiated circuit has its entry body");
        if body
            .ops()
            .iter()
            .any(|op| matches!(op, Op::Gate { gate, .. } if gate.is_non_clifford()))
        {
            // Also covers top-level T sources from `prepare_t_with_mpps`.
            continue;
        }
        let segments = aligned_moment_segments(body.ops())
            .wrap_err_with(|| format!("segment concurrent node N{}", id.0))?;
        let mut slice_moment = 0;
        let slice_moments = segments
            .iter()
            .map(|segment| {
                segment.as_ref()?;
                let current = slice_moment;
                slice_moment += 1;
                Some(current)
            })
            .collect();
        let moments = segments
            .iter()
            .map(|segment| segment.as_ref().map(|segment| segment.kind))
            .collect();
        let qubits = sorted_node_layout_coords(&circuit);
        lanes.push(MomentLane {
            node: id,
            moments,
            qubits: qubits.clone(),
        });
        node_circuits.insert(
            id,
            ConcurrentNodeCircuit {
                segments,
                slice_moments,
                qubits,
            },
        );
    }

    let aligned = align_moment_lanes(flat.top(), lanes).wrap_err("align source-layer moments")?;
    let mut views = HashMap::new();
    for layer in aligned {
        // The picker has source-z layers only; temporal H seams moved above.
        if layer.layer.rem_euclid(2) != 0 {
            continue;
        }
        let normalized_z = i32::try_from(layer.layer.div_euclid(2))
            .wrap_err("aligned source layer exceeds editor coordinate range")?;
        let display_z = normalized_z
            .checked_add(source_offset.z)
            .wrap_err("aligned display layer exceeds editor coordinate range")?;
        let contributors = layer
            .slots
            .iter()
            .flat_map(|slot| slot.entries.iter().map(|entry| entry.node))
            .chain(
                source_graph
                    .layer(normalized_z)
                    .blocks()
                    .filter_map(|block| flat.node_by_block(block.pos())),
            )
            .filter(|id| node_circuits.contains_key(id))
            .collect::<HashSet<_>>();
        let mut layout_coords = contributors
            .iter()
            .flat_map(|id| node_circuits[id].qubits.iter().copied())
            .collect::<Vec<_>>();
        layout_coords.sort_by_key(|coord| (coord.x, coord.y));
        layout_coords.dedup();
        let coord_to_qubit = layout_coords
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coord)| Ok((coord, checked_qubit_index(index)?)))
            .collect::<eyre::Result<HashMap<_, _>>>()?;
        let qubit_coords = layout_coords
            .iter()
            .copied()
            .enumerate()
            .collect::<HashMap<_, _>>();
        let mut moments = Vec::new();
        let mut source_moments = Vec::new();
        for slot in layer.slots {
            let Some(kind) = slot.kind else {
                continue;
            };
            let mut ops = Vec::new();
            let mut sources = Vec::new();
            for entry in &slot.entries {
                let circuit = &node_circuits[&entry.node];
                let segment = circuit
                    .segments
                    .get(entry.moment)
                    .expect("aligned moment references its source segment");
                if let Some(segment) = segment {
                    debug_assert_eq!(segment.kind, kind);
                    ops.extend(ops_to_moment_ops(&segment.ops, &coord_to_qubit));
                    sources.push((
                        entry.node.0,
                        circuit.slice_moments[entry.moment]
                            .expect("visible aligned segment has a slice moment"),
                    ));
                }
            }
            sources.sort_unstable();
            source_moments.push(sources);
            moments.push(FlatMoment {
                kind,
                ops,
                repeat_label: None,
                repeat_span: None,
            });
        }

        let _ = views.insert(
            display_z,
            LayerCircuitView {
                moments,
                source_moments,
                qubit_coords,
                num_qubits: layout_coords.len(),
            },
        );
    }
    Ok(views)
}

fn build_view(
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
    details: bool,
) -> eyre::Result<BloqCircuitView> {
    bloq_compile::validate_bloq_qubit_layout_for_source(program, source_graph)
        .wrap_err("validate Bloq qubit layout")?;

    // Observable lowering injects classical recipe and value
    // nodes that carry no circuit. They are built here so the graph view can show
    // dataflow when the user toggles it on; the renderer hides them by default.
    // Region nodes recurse: their body nodes are lifted into the flat view with
    // a `parent` link so the renderer can box them inside the region tile.
    let ctx = ViewContext {
        program,
        source_graph,
        source_offset,
        details,
    };
    let mut view = BloqCircuitView {
        nodes: Vec::new(),
        edges: Vec::new(),
        node_refs: HashMap::new(),
    };
    // Body nodes get fresh view ids past the top-level range (a `SubGraph`'s
    // node ids restart at 0, so they cannot be used verbatim).
    let mut next_id = program.node_ids().map(|id| id.0 + 1).max().unwrap_or(0);
    for (id, node) in program.nodes() {
        push_node_views(
            id.0,
            id,
            &LevelPath::default(),
            None,
            node,
            &ctx,
            &mut next_id,
            &mut view,
        )?;
    }

    // Output-frame `Compute` nodes have no outgoing edges (their `OutputFrame`
    // provenance is the reference); the label already carries the role via
    // `classical_provenance_suffix`, and this attribute spells it out in the
    // detail panel.
    if details {
        for frame in program.output_frames() {
            for (axis, node_id) in [("X", frame.x), ("Z", frame.z)] {
                if let Some(node) = view.nodes.iter_mut().find(|node| node.id == node_id.0) {
                    node.attributes.push((
                        "role".into(),
                        format!(
                            "{axis}-frame sign for output port ({}, {}, {})",
                            frame.port.x, frame.port.y, frame.port.z
                        ),
                    ));
                }
            }
        }
    }

    // Column placement (including where classical nodes land relative to their
    // quantum anchors) is computed by the graph-view layout pass from the edge
    // structure; here we only need a deterministic node order.
    if details {
        view.nodes.sort_by_key(|node| (node.layer, node.id));

        for edge in program.edges() {
            view.edges.push(BloqEdgeView {
                from: edge.source.0,
                to: edge.target.0,
                kind: edge_kind(edge.edge),
                has_pipes: !edge.edge.pipes().is_empty(),
            });
        }
        view.edges
            .sort_by_key(|edge| (edge.from, edge.to, !edge.has_pipes));
    }

    Ok(view)
}

struct ViewContext<'a> {
    program: &'a Bloq,
    source_graph: &'a BlockGraph,
    source_offset: IVec3,
    details: bool,
}

/// Build the view for one node. A quantum/classical node yields one tile; a
/// region node first lifts its body nodes (recursively, with remapped view ids
/// and `parent` links), then summarizes itself as a container tile.
///
/// `local_id` is the node's id within its own graph level and `path` names the
/// region bodies descended into to reach it: together they form the tape-space
/// [`NodeRef`] that keys the detector-slice data, recorded here against the
/// (possibly remapped) `view_id`.
fn push_node_views(
    view_id: u32,
    local_id: BloqNodeId,
    path: &LevelPath,
    parent: Option<u32>,
    node: &BloqNode,
    ctx: &ViewContext<'_>,
    next_id: &mut u32,
    view: &mut BloqCircuitView,
) -> eyre::Result<()> {
    let Some(region) = node.try_region() else {
        let mut built = build_node_view(
            BloqNodeId(view_id),
            node,
            ctx.program,
            ctx.source_graph,
            ctx.source_offset,
            ctx.details,
        )?;
        built.parent = parent;
        view.nodes.push(built);
        view.node_refs.insert(
            NodeRef {
                path: path.clone(),
                node: local_id,
            },
            view_id,
        );
        return Ok(());
    };

    let start = view.nodes.len();
    let mut body_nodes = 0usize;
    for (selector, body) in region.bodies() {
        // A body node's `NodeRef` descends one more step, naming this region (by
        // its level-local id) and which body — mirroring `bloq_ir::detslice`.
        let child_path = path.child(local_id, selector);
        let id_map: HashMap<BloqNodeId, u32> = body
            .node_ids()
            .map(|body_id| {
                let mapped = *next_id;
                *next_id += 1;
                (body_id, mapped)
            })
            .collect();
        body_nodes += id_map.len();
        for (body_id, body_node) in body.nodes() {
            push_node_views(
                id_map[&body_id],
                body_id,
                &child_path,
                Some(view_id),
                body_node,
                ctx,
                next_id,
                view,
            )?;
        }
        if ctx.details {
            for edge in body.edges() {
                view.edges.push(BloqEdgeView {
                    from: id_map[&edge.source],
                    to: id_map[&edge.target],
                    kind: edge_kind(edge.edge),
                    has_pipes: !edge.edge.pipes().is_empty(),
                });
            }
        }
    }

    // The slice covers the whole subtree (nested regions append before their
    // parent), so quantum content and qubit/source aggregation see all depths.
    let (has_quantum_body, qubits, source_elements) = if ctx.details {
        let descendants = &view.nodes[start..];
        (
            descendants.iter().any(|child| child.category.is_quantum()),
            descendants
                .iter()
                .flat_map(|child| child.qubit_coords.values().copied())
                .collect::<HashSet<_>>(),
            descendants
                .iter()
                .flat_map(|child| child.source_elements.iter().copied())
                .collect::<HashSet<_>>(),
        )
    } else {
        (false, HashSet::new(), HashSet::new())
    };
    let (label, attributes) = if ctx.details {
        (
            format!("N{} · {}", view_id, region.kind_name()),
            region_attributes(region, body_nodes, qubits.len()),
        )
    } else {
        (String::new(), Vec::new())
    };

    view.nodes.push(BloqNodeView {
        id: view_id,
        layer: node.layer() + 2 * i64::from(ctx.source_offset.z),
        label,
        category: BloqNodeCategory::Region,
        attributes,
        operator_table: Vec::new(),
        source_elements,
        moments: Vec::new(),
        qubit_coords: HashMap::new(),
        num_qubits: qubits.len(),
        parent,
        has_quantum_body,
    });
    Ok(())
}

fn region_attributes(
    region: &RegionNode,
    body_nodes: usize,
    body_qubits: usize,
) -> Vec<(String, String)> {
    let RegionNode::RepeatUntilSuccess {
        restart_condition, ..
    } = region;
    vec![
        ("kind".into(), region.kind_name().into()),
        ("restart_condition".into(), format!("{restart_condition:?}")),
        ("body nodes".into(), body_nodes.to_string()),
        ("body qubits".into(), body_qubits.to_string()),
    ]
}

fn edge_kind(edge: &BloqEdge) -> BloqEdgeKind {
    match edge {
        BloqEdge::Quantum(_) => BloqEdgeKind::Quantum,
        BloqEdge::Value {
            output: bloq_ir::ObservableOutput::Flip,
            ..
        } => BloqEdgeKind::Flip,
        BloqEdge::Value { .. } => BloqEdgeKind::Value,
        BloqEdge::Compose { .. } => BloqEdgeKind::Compose,
        BloqEdge::Order => BloqEdgeKind::Order,
    }
}

fn node_category(node: &BloqNode) -> BloqNodeCategory {
    if node.try_region().is_some() {
        return BloqNodeCategory::Region;
    }
    if node.try_quantum().is_some() {
        return match node.provenance {
            NodeProvenance::TemporalPipe { .. } | NodeProvenance::MemoryPadding { .. } => {
                BloqNodeCategory::QuantumPipe
            }
            _ => BloqNodeCategory::QuantumBlock,
        };
    }
    match node.try_classical() {
        Some(ClassicalNode::Observable { .. }) => BloqNodeCategory::Observable,
        // Compute and Discard share the catch-all colour.
        _ => BloqNodeCategory::Other,
    }
}

fn build_node_view(
    id: BloqNodeId,
    node: &BloqNode,
    program: &Bloq,
    source_graph: &BlockGraph,
    source_offset: IVec3,
    details: bool,
) -> eyre::Result<BloqNodeView> {
    // SEM-MEMBERSHIP: alternatives are not one static circuit. The Program
    // graph can show their registry; slice timelines still require selection.
    let conditional = details
        && node
            .try_quantum()
            .is_some_and(|quantum| !quantum.guards.is_empty());
    let circuit = if conditional {
        None
    } else {
        Some(
            node.instantiate_circuit(program.templates())
                .wrap_err_with(|| format!("instantiate circuit for Bloq node N{}", id.0))?,
        )
    };
    let layout_coords = if let Some(circuit) = &circuit {
        sorted_node_layout_coords(circuit)
    } else {
        let mut coords = program.node_qubits(node)?.into_iter().collect::<Vec<_>>();
        coords.sort_by_key(|coord| (coord.x, coord.y));
        coords
    };
    let coord_to_qubit = layout_coords
        .iter()
        .copied()
        .enumerate()
        .map(|(index, coord)| Ok((coord, checked_qubit_index(index)?)))
        .collect::<eyre::Result<HashMap<_, _>>>()?;
    let qubit_coords = layout_coords
        .iter()
        .copied()
        .enumerate()
        .collect::<HashMap<_, _>>();
    let (source_elements, label, attributes, operator_table) = if details {
        (
            node_source_elements(node, source_graph, source_offset),
            node_label(id, node),
            node_attributes(node),
            node_operator_table(node),
        )
    } else {
        (HashSet::new(), String::new(), Vec::new(), Vec::new())
    };
    let moments = circuit
        .as_ref()
        .map(|circuit| node_timeline(circuit, &coord_to_qubit))
        .unwrap_or_default();

    Ok(BloqNodeView {
        id: id.0,
        layer: node.layer() + 2 * i64::from(source_offset.z),
        label,
        category: node_category(node),
        attributes,
        operator_table,
        source_elements,
        moments,
        qubit_coords,
        num_qubits: layout_coords.len(),
        parent: None,
        has_quantum_body: false,
    })
}

fn checked_qubit_index(index: usize) -> eyre::Result<u32> {
    u32::try_from(index).wrap_err("viewer qubit index exceeds u32 range")
}

/// An `Observable` node's boundary operators as
/// `(instance, face, operator)` table rows.
fn node_operator_table(node: &BloqNode) -> Vec<(String, String, String)> {
    let Some(ClassicalNode::Observable { operators, .. }) = node.try_classical() else {
        return Vec::new();
    };
    operators
        .iter()
        .map(|operator| {
            (
                format!("i{}", operator.instance.0),
                format!("{:?}", operator.face),
                operator.operator.to_string(),
            )
        })
        .collect()
}

/// Readable attributes for classical nodes and conditional quantum registries.
fn node_attributes(node: &BloqNode) -> Vec<(String, String)> {
    if let Some(quantum) = node
        .try_quantum()
        .filter(|quantum| !quantum.guards.is_empty())
    {
        let mut attributes = vec![(
            "circuit".into(),
            "Conditional components: a circuit timeline requires selector values.".into(),
        )];
        for instance in &quantum.instances {
            let condition = quantum
                .guards
                .iter()
                .find(|guard| guard.instances.contains(&instance.id))
                .map_or_else(
                    || "always".into(),
                    |guard| format!("input {} is true", guard.input),
                );
            attributes.push((format!("instance i{}", instance.id.0), condition));
        }
        for (index, bundle) in quantum.detector_bundles.iter().enumerate() {
            let condition = quantum
                .guards
                .iter()
                .find(|guard| {
                    guard
                        .detector_bundles
                        .iter()
                        .any(|&entry| entry as usize == index)
                })
                .map_or_else(
                    || "always".into(),
                    |guard| format!("input {} is true", guard.input),
                );
            attributes.push((
                format!("detector bundle {index}"),
                format!(
                    "bundle b{}, {} owners, offset ({}, {}); {condition}",
                    bundle.bundle.0,
                    bundle.instances.len(),
                    bundle.offset.x,
                    bundle.offset.y,
                ),
            ));
        }
        for guard in &quantum.guards {
            attributes.push((
                format!("input {} is true", guard.input),
                format!(
                    "{} detectors, {} detector bundles, {} restarts; {} detector parity updates, {} restart parity updates",
                    guard.detectors.len(),
                    guard.detector_bundles.len(),
                    guard.restarts.len(),
                    guard.detector_parities.len(),
                    guard.restart_parities.len(),
                ),
            ));
        }
        return attributes;
    }
    let Some(classical) = node.try_classical() else {
        return Vec::new();
    };
    match classical {
        ClassicalNode::Observable {
            index,
            measurements,
            operators,
        } => vec![
            ("kind".into(), "Observable".into()),
            (
                "index".into(),
                index.map_or_else(|| "fragment".into(), |index| index.to_string()),
            ),
            ("measurements".into(), measurements.len().to_string()),
            (
                "sites".into(),
                measurements
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            ("operators".into(), operators.len().to_string()),
            (
                "outputs".into(),
                if index.is_some() {
                    "Corrected: decoded parity; Flip: decoder correction"
                } else {
                    "Corrected: assembled parity; composition: records and boundaries"
                }
                .into(),
            ),
        ],
        ClassicalNode::Compute { expr } => vec![
            ("kind".into(), "Compute".into()),
            ("expr".into(), format!("{expr:?}")),
        ],
        ClassicalNode::Discard { condition } => vec![
            ("kind".into(), "Discard".into()),
            ("condition".into(), format!("{condition:?}")),
        ],
    }
}

fn sorted_node_layout_coords(circuit: &CoordCircuit) -> Vec<IVec2> {
    let mut coords = circuit.qubits().into_iter().collect::<Vec<_>>();
    coords.sort_by_key(|coord| (coord.x, coord.y));
    coords.dedup();
    coords
}

fn node_source_elements(
    node: &BloqNode,
    source_graph: &BlockGraph,
    source_offset: IVec3,
) -> HashSet<GraphElement> {
    match &node.provenance {
        NodeProvenance::BlockComponent { members } => {
            let mut elements = members
                .iter()
                .map(|member| GraphElement::Block(member.pos + source_offset))
                .collect::<HashSet<_>>();
            for (lhs_index, lhs) in members.iter().enumerate() {
                for rhs in members.iter().skip(lhs_index + 1) {
                    if source_graph.has_pipe_between(lhs.pos, rhs.pos) {
                        elements.insert(
                            GraphElement::Pipe(lhs.pos + source_offset, rhs.pos + source_offset)
                                .canonical(),
                        );
                    }
                }
            }
            elements
        }
        NodeProvenance::MemoryPadding { pipe, .. } if pipe.src == pipe.dst => {
            HashSet::from([GraphElement::Block(pipe.src + source_offset)])
        }
        NodeProvenance::TemporalPipe { pipe } | NodeProvenance::MemoryPadding { pipe, .. } => {
            HashSet::from([
                GraphElement::Pipe(pipe.src + source_offset, pipe.dst + source_offset).canonical(),
            ])
        }
        // Classical provenance (generator/action/frame) names no source-graph
        // geometry to highlight.
        _ => HashSet::new(),
    }
}

fn node_label(id: BloqNodeId, node: &BloqNode) -> String {
    if let Some(classical) = node.try_classical() {
        let tag = match classical {
            ClassicalNode::Observable {
                index: Some(index), ..
            } => format!("obs {index}"),
            ClassicalNode::Observable { index: None, .. } => "obs fragment".into(),
            ClassicalNode::Compute { .. } => "compute".into(),
            ClassicalNode::Discard { .. } => "discard".to_string(),
        };
        return format!("N{} · {tag}{}", id.0, classical_provenance_suffix(node));
    }
    let Some(quantum) = node.try_quantum() else {
        return format!("N{} · region", id.0);
    };
    // Instance ids link observable measurements and operators to their quantum node.
    let instances = quantum
        .instances
        .iter()
        .map(|instance| format!("i{}", instance.id.0))
        .collect::<Vec<_>>()
        .join(",");
    let instances = if instances.is_empty() {
        String::new()
    } else {
        format!(" · {instances}")
    };
    match &node.provenance {
        NodeProvenance::BlockComponent { members } => {
            format!(
                "N{} · {} block{}{instances}",
                id.0,
                members.len(),
                plural_s(members.len())
            )
        }
        NodeProvenance::TemporalPipe { pipe } => {
            let suffix = if pipe.hadamard { " · H" } else { "" };
            format!("N{} · pipe{}{instances}", id.0, suffix)
        }
        NodeProvenance::MemoryPadding { rounds, .. } => {
            format!(
                "N{} · memory {rounds} round{}{instances}",
                id.0,
                plural_s(*rounds as usize)
            )
        }
        _ => format!("N{} · quantum{instances}", id.0),
    }
}

/// A classical node's provenance rendered as a label suffix: which generator
/// row, source action, or output frame it realizes.
fn classical_provenance_suffix(node: &BloqNode) -> String {
    match &node.provenance {
        NodeProvenance::Generator { ordinal } => format!(" · gen {ordinal}"),
        NodeProvenance::Action { ordinal } => format!(" · action {ordinal}"),
        NodeProvenance::BranchSelector { name } => format!(" · selector {name}"),
        NodeProvenance::OutputFrame { port, basis } => {
            let basis = match basis {
                bloq_ir::Basis::X => "X",
                bloq_ir::Basis::Z => "Z",
            };
            format!(" · {basis}-frame ({},{},{})", port.x, port.y, port.z)
        }
        _ => String::new(),
    }
}

fn plural_s(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

#[derive(Default)]
struct TimelineCollector {
    segmenter: MomentSegmenter,
    moments: Vec<FlatMoment>,
}

fn node_timeline(circuit: &CoordCircuit, coord_to_qubit: &HashMap<IVec2, u32>) -> Vec<FlatMoment> {
    let mut collector = TimelineCollector::default();
    collect_body_timeline(
        circuit,
        circuit.entry_body(),
        coord_to_qubit,
        None,
        &mut collector,
    );
    collector.moments
}

fn collect_body_timeline(
    circuit: &CoordCircuit,
    body: bloq_circuit::BodyId,
    coord_to_qubit: &HashMap<IVec2, u32>,
    active_repeat: Option<u32>,
    collector: &mut TimelineCollector,
) {
    let body_ops = circuit
        .body(body)
        .expect("instantiated circuit contains every referenced body");
    let mut current = Vec::new();
    let mut first_repeat_moment = active_repeat.is_some();
    for op in body_ops.ops() {
        match op {
            &Op::Repeat {
                body, repetitions, ..
            } => {
                flush_ops(
                    &mut current,
                    coord_to_qubit,
                    active_repeat,
                    &mut first_repeat_moment,
                    &mut collector.segmenter,
                    &mut collector.moments,
                );
                let repeat_start = collector.moments.len();
                collect_body_timeline(circuit, body, coord_to_qubit, Some(repetitions), collector);
                let repeat_span = collector.moments.len() - repeat_start;
                if repeat_span > 0
                    && let Some(moment) = collector.moments.get_mut(repeat_start)
                {
                    moment.repeat_span = Some(repeat_span);
                }
            }
            // Keep ticks in the straight-line chunk so feedforward from an
            // all-invisible tick reaches the next tick. The stateful segmenter
            // also carries it through structural repeat recursion.
            Op::Tick
            | Op::Gate { .. }
            | Op::Measure { .. }
            | Op::MPP { .. }
            | Op::ConditionalPauli(_)
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => current.push(op.clone()),
        }
    }
    flush_ops(
        &mut current,
        coord_to_qubit,
        active_repeat,
        &mut first_repeat_moment,
        &mut collector.segmenter,
        &mut collector.moments,
    );
}

fn flush_ops(
    ops: &mut Vec<Op>,
    coord_to_qubit: &HashMap<IVec2, u32>,
    active_repeat: Option<u32>,
    first_repeat_moment: &mut bool,
    segmenter: &mut MomentSegmenter,
    out: &mut Vec<FlatMoment>,
) {
    if ops.is_empty() {
        return;
    }
    for mut moment in moments_from_ops(segmenter, ops, coord_to_qubit) {
        moment.repeat_label = if *first_repeat_moment {
            active_repeat.map(|repetitions| format!("REPEAT {repetitions}"))
        } else {
            None
        };
        *first_repeat_moment = false;
        out.push(moment);
    }
    ops.clear();
}

/// Bucket a straight-line op run into viewer moments using the shared
/// [`MomentSegmenter`] rules, so the timeline and the detector-slice tape cannot
/// drift on how ops split into moments. `circuit_ops` may span ticks; structural
/// repeats delimit chunks while the segmenter carries invisible feedforward.
fn moments_from_ops(
    segmenter: &mut MomentSegmenter,
    circuit_ops: &[Op],
    coord_to_qubit: &HashMap<IVec2, u32>,
) -> Vec<FlatMoment> {
    segmenter
        .extend(circuit_ops)
        .into_iter()
        .map(|segment| FlatMoment {
            kind: segment.kind,
            ops: ops_to_moment_ops(&segment.ops, coord_to_qubit),
            repeat_label: None,
            repeat_span: None,
        })
        .collect()
}

fn ops_to_moment_ops(
    circuit_ops: &[Op],
    coord_to_qubit: &HashMap<IVec2, u32>,
) -> Vec<FlatMomentOp> {
    let mut moment_ops = Vec::new();
    let qubit_indices = |qubits: &[IVec2]| {
        qubits
            .iter()
            .map(|&coord| qubit_index(coord_to_qubit, coord))
            .collect::<Vec<_>>()
    };

    for op in circuit_ops {
        match op {
            Op::Gate { gate, qubits } => {
                moment_ops.push(FlatMomentOp::Gate {
                    gate: *gate,
                    qubits: qubit_indices(qubits),
                });
            }
            Op::Measure { basis, qubits, .. } => {
                moment_ops.push(FlatMomentOp::Measure {
                    basis: *basis,
                    qubits: qubit_indices(qubits),
                });
            }
            // A product measurement: each product's per-qubit Pauli support,
            // resolved to viewer qubit indices, drawn as joined measurement
            // glyphs (Stim `MPP`). Empty products (identity) are skipped.
            Op::MPP { products, .. } => {
                let products = products
                    .iter()
                    .map(|product| {
                        product
                            .iter()
                            .map(|(&coord, &pauli)| {
                                (qubit_index(coord_to_qubit, coord), pauli_basis(pauli))
                            })
                            .collect::<Vec<_>>()
                    })
                    .filter(|product| !product.is_empty())
                    .collect::<Vec<_>>();
                if !products.is_empty() {
                    moment_ops.push(FlatMomentOp::Mpp { products });
                }
            }
            Op::Tick
            | Op::Repeat { .. }
            | Op::ConditionalPauli(_)
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => {}
        }
    }
    moment_ops
}

/// A `PauliMap` entry's [`Pauli`] as a [`PauliBasis`]. `PauliMap` never stores
/// identity, so the `I` arm is unreachable; it falls back to `Z`.
fn pauli_basis(pauli: Pauli) -> PauliBasis {
    match pauli {
        Pauli::X => PauliBasis::X,
        Pauli::Y => PauliBasis::Y,
        Pauli::Z | Pauli::I => PauliBasis::Z,
    }
}

fn qubit_index(coord_to_qubit: &HashMap<IVec2, u32>, coord: IVec2) -> u32 {
    coord_to_qubit
        .get(&coord)
        .copied()
        .expect("instantiated circuit qubit exists in its viewer layout")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CircuitBody, ConditionalCorrection, GateType};
    use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
    use bloq_ir::{
        BoundaryFace, ClassicalExpr, MomentKind, QuantumTimeline, SourceBlockRef, SubGraph,
        TemplateId, TemporalPipeRef,
        lowering::{
            BloqTemplate, InstanceBoundaryOperator, InstanceMeasurement,
            NodeTemplateInstanceMergeError, TemplateInstance, TemplateInstanceId,
        },
    };
    use glam::{ivec2, ivec3};

    #[test]
    fn viewer_qubit_indices_do_not_wrap() {
        let maximum = u32::MAX as usize;
        assert_eq!(checked_qubit_index(maximum).unwrap(), u32::MAX);
        if let Some(index) = maximum.checked_add(1) {
            assert_eq!(
                checked_qubit_index(index).unwrap_err().to_string(),
                "viewer qubit index exceeds u32 range"
            );
        }
    }

    fn push_repeat(circuit: &mut CoordCircuit, body: bloq_circuit::BodyId, repetitions: u32) {
        circuit
            .body_mut(circuit.entry_body())
            .expect("entry body exists")
            .ops_mut()
            .push(Op::Repeat { body, repetitions });
    }

    fn push_measurement_correction(circuit: &mut CoordCircuit, measurement: u32, target: IVec2) {
        circuit
            .body_mut(circuit.entry_body())
            .expect("entry body exists")
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: (measurement),
                target,
            }]));
    }

    #[test]
    fn timeline_matches_tracker_across_an_invisible_feedforward_tick() {
        let control = ivec2(0, 0);
        let target = ivec2(1, 0);
        let spectator = ivec2(2, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.measure(PauliBasis::Z, [control])[0];
        circuit.tick();
        push_measurement_correction(&mut circuit, measurement, target);
        circuit.tick();
        circuit.measure(PauliBasis::Z, [spectator]);
        circuit.do_gate(GateType::RZ, [target]).unwrap();

        let ops = circuit.body(circuit.entry_body()).unwrap().ops();
        let tracker_kinds: Vec<_> = bloq_ir::moment_segments(ops)
            .into_iter()
            .map(|moment| moment.kind)
            .collect();
        let coord_to_qubit = HashMap::from([(control, 0), (target, 1), (spectator, 2)]);
        let viewer_kinds: Vec<_> = node_timeline(&circuit, &coord_to_qubit)
            .into_iter()
            .map(|moment| moment.kind)
            .collect();

        assert_eq!(
            tracker_kinds,
            vec![
                MomentKind::Measurement,
                MomentKind::Measurement,
                MomentKind::Reset,
            ]
        );
        assert_eq!(viewer_kinds, tracker_kinds);
    }

    #[test]
    fn timeline_carries_invisible_feedforward_into_a_repeat_body() {
        let control = ivec2(0, 0);
        let target = ivec2(1, 0);
        let spectator = ivec2(2, 0);
        let mut circuit = CoordCircuit::new();
        let control_measurement = circuit.measure(PauliBasis::Z, [control])[0];
        let spectator_measurement = circuit.reserve_measurement_id(spectator);
        circuit.tick();
        push_measurement_correction(&mut circuit, control_measurement, target);
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![spectator],
                measurements: vec![spectator_measurement],
                flip_probability: 0.0,
            },
            Op::Gate {
                gate: GateType::RZ,
                qubits: vec![target],
            },
        ]));
        push_repeat(&mut circuit, body, 7);

        let entry_ops = circuit.body(circuit.entry_body()).unwrap().ops();
        let repeat_index = entry_ops
            .iter()
            .position(|op| matches!(op, Op::Repeat { .. }))
            .unwrap();
        let mut projected_ops = entry_ops[..repeat_index].to_vec();
        projected_ops.extend_from_slice(circuit.body(body).unwrap().ops());
        projected_ops.extend_from_slice(&entry_ops[repeat_index + 1..]);
        let tracker_kinds: Vec<_> = bloq_ir::moment_segments(&projected_ops)
            .into_iter()
            .map(|moment| moment.kind)
            .collect();
        let coord_to_qubit = HashMap::from([(control, 0), (target, 1), (spectator, 2)]);
        let timeline = node_timeline(&circuit, &coord_to_qubit);
        let viewer_kinds: Vec<_> = timeline.iter().map(|moment| moment.kind).collect();

        assert_eq!(viewer_kinds, tracker_kinds);
        assert_eq!(timeline[1].repeat_label.as_deref(), Some("REPEAT 7"));
        assert_eq!(timeline[1].repeat_span, Some(2));
    }

    #[test]
    fn bloq_view_preserves_repeat_once_with_label() {
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(ivec2(0, 0));
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Gate {
                gate: GateType::H,
                qubits: vec![ivec2(0, 0)],
            },
            Op::Tick,
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![ivec2(0, 0)],
                measurements: vec![measurement],
                flip_probability: 0.0,
            },
        ]));
        push_repeat(&mut circuit, body, 7);
        let mut program = Bloq::new();
        let template = program.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        program.add_node(node);
        let source_graph = BlockGraph::new();

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");
        let moments = &view.nodes[0].moments;

        assert_eq!(moments.len(), 2);
        assert_eq!(moments[0].repeat_label.as_deref(), Some("REPEAT 7"));
        assert_eq!(moments[0].repeat_span, Some(2));
        assert_eq!(moments[1].repeat_label, None);
        assert_eq!(moments[1].repeat_span, None);
    }

    #[test]
    fn bloq_view_instantiates_template_backed_node() {
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut program = Bloq::new();
        let template = program.add_template(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(2, 0),
            ));
        program.add_node(node);
        let source_graph = BlockGraph::new();

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");

        assert_eq!(view.nodes[0].num_qubits, 1);
        assert_eq!(view.nodes[0].qubit_coords[&0], ivec2(2, 0));
        assert_eq!(view.nodes[0].moments.len(), 1);
        assert_eq!(view.nodes[0].moments[0].kind, MomentKind::Measurement);
    }

    #[test]
    fn concurrent_layers_merge_global_qubits_and_project_a_tall_block() {
        let measurement_circuit = |rounds: usize| {
            let mut circuit = CoordCircuit::new();
            for round in 0..rounds {
                if round > 0 {
                    circuit.tick();
                }
                circuit
                    .do_gate(GateType::RZ, [IVec2::ZERO])
                    .expect("reset is valid");
                circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
            }
            circuit
        };
        let tall = ivec3(0, 0, 2);
        let peer = ivec3(1, 0, 3);
        let mut program = Bloq::new();
        let tall_template = program.add_template(BloqTemplate::new(measurement_circuit(2)));
        let peer_template = program.add_template(BloqTemplate::new(measurement_circuit(1)));
        for (instance, template_id, pos, offset, layer_round_ends) in [
            (0, tall_template, tall, IVec2::ZERO, Some(vec![1, 2])),
            (1, peer_template, peer, peer.truncate() * 8, None),
        ] {
            let mut node = BloqNode::from_members(vec![SourceBlockRef { pos }]);
            let quantum = node.expect_quantum_mut();
            quantum.instances.push(TemplateInstance::new(
                TemplateInstanceId(instance),
                template_id,
                offset,
            ));
            quantum.timeline =
                layer_round_ends.map(|layer_round_ends| QuantumTimeline { layer_round_ends });
            program.add_node(node);
        }
        let mut source = BlockGraph::new();
        source.add_block(
            Block::new(tall, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("cube accepts height"),
        );
        source.add_block(Block::new(peer, BlockKind::Cube(CubeKind::XZZ)));

        let (views, reason) = build_concurrent_layer_views(&program, &source, IVec3::ZERO);

        assert!(reason.is_none(), "{reason:?}");
        let upper = &views[&3];
        assert_eq!(
            upper.qubit_coords,
            HashMap::from([(0, ivec2(0, 0)), (1, ivec2(8, 0))])
        );
        assert_eq!(
            upper
                .moments
                .iter()
                .map(|moment| (moment.kind, moment.ops.len()))
                .collect::<Vec<_>>(),
            [(MomentKind::Reset, 2), (MomentKind::Measurement, 2)]
        );
        assert_eq!(
            upper.source_moments,
            [vec![(0, 2), (1, 0)], vec![(0, 3), (1, 1)]]
        );
    }

    #[test]
    fn concurrent_layers_show_temporal_hadamard_with_its_source_layer() {
        let source = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: XZX [0,0,0]\n  1: ZXZ [0,0,1]\n  [0,0,0] -H> +Z\n",
        )
        .expect("temporal H graph parses");
        let program = bloq_compile::compile(&source, 3).expect("temporal H graph compiles");
        let hadamard = program
            .nodes()
            .find_map(|(id, node)| {
                matches!(node.provenance, NodeProvenance::TemporalPipe { .. }).then_some(id.0)
            })
            .expect("realignment node exists");

        let (views, reason) = build_concurrent_layer_views(&program, &source, IVec3::ZERO);

        assert!(reason.is_none(), "{reason:?}");
        assert!(
            views[&0]
                .source_moments
                .iter()
                .flatten()
                .any(|&(node, _)| node == hadamard),
            "realignment is visible on source z=0"
        );
    }

    #[test]
    fn concurrent_layers_build_for_a_compiled_t_graph() {
        let source = bloq_graph::GalleryItem::T.build();
        for prepare_t_with_mpps in [false, true] {
            let program = bloq_compile::compile_with(
                &source,
                bloq_compile::CompileConfig::default()
                    .with_prepare_t_with_mpps(prepare_t_with_mpps),
            )
            .expect("T graph compiles")
            .bloq;

            let choices = program
                .nodes()
                .filter_map(|(_, node)| match &node.provenance {
                    NodeProvenance::BranchSelector { name } => Some((name.clone(), false)),
                    _ => None,
                })
                .collect();
            let program = program.pin_membership(&choices).expect("T choices pin");
            let (views, reason) = build_concurrent_layer_views(&program, &source, IVec3::ZERO);

            assert!(reason.is_none(), "{reason:?}");
            assert!(!views.is_empty(), "Clifford neighbors remain visible");
        }
    }

    #[test]
    fn ccz_teleportation_concurrent_moments_match_the_node_timelines() {
        let source = bloq_graph::GalleryItem::CCZGateTeleport.build();
        let original = bloq_compile::compile(&source, 3).unwrap();
        for selected in [false, true] {
            let choices = original
                .nodes()
                .filter_map(|(_, node)| match &node.provenance {
                    NodeProvenance::BranchSelector { name } => Some((name.clone(), selected)),
                    _ => None,
                })
                .collect();
            let mut program = original.pin_membership(&choices).unwrap();
            let (views, reason) = build_concurrent_layer_views(&program, &source, IVec3::ZERO);
            assert!(reason.is_none(), "{reason:?}");
            assert!(!views.is_empty());
            program.flatten().unwrap();
            let nodes = build_detslice_circuit_view(&program, &source, IVec3::ZERO)
                .unwrap()
                .nodes;
            for view in views.values() {
                assert_eq!(view.source_moments.len(), view.moments.len());
                for &(node, moment) in view.source_moments.iter().flatten() {
                    let node = nodes.iter().find(|view| view.id == node).unwrap();
                    assert!(
                        moment < node.moments.len(),
                        "corrections add no visible moment"
                    );
                }
            }
        }
    }

    #[test]
    fn bloq_view_propagates_template_instantiation_errors() {
        let missing_template = TemplateId(99);
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                missing_template,
                IVec2::ZERO,
            ));
        let mut program = Bloq::new();
        program.add_node(node);

        let error = build_bloq_circuit_view(&program, &BlockGraph::new(), IVec3::ZERO)
            .expect_err("missing template must reject the view");

        assert_eq!(error.to_string(), "instantiate circuit for Bloq node N0");
        assert!(matches!(
            error.downcast_ref::<NodeTemplateInstanceMergeError>(),
            Some(NodeTemplateInstanceMergeError::UnknownTemplate(template))
                if *template == missing_template
        ));
    }

    #[test]
    fn classical_observable_nodes_are_built_with_category_and_attributes() {
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut program = Bloq::new();
        let template = program.add_template(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let quantum = program.add_node(node);
        // Observable lowering appends this classical node; it carries no circuit
        // but must still be built so the graph view can show it on toggle.
        let observable = program.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Input,
                operator: [(ivec2(0, 0), Pauli::Z)].into_iter().collect(),
            }],
        }));
        program.add_edge(quantum, observable, BloqEdge::Order);
        let fragment = program.add_node(BloqNode::classical(ClassicalNode::fragment()));
        program.add_edge(fragment, observable, BloqEdge::compose(0));
        let corrected = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: bloq_ir::ClassicalExpr::In(0),
        }));
        program.add_edge(observable, corrected, BloqEdge::value(0));
        let flip = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: bloq_ir::ClassicalExpr::In(0),
        }));
        program.add_edge(observable, flip, BloqEdge::flip(0));
        let source_graph = BlockGraph::new();

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");

        let quantum_view = view
            .nodes
            .iter()
            .find(|node| node.id == quantum.0)
            .expect("quantum node built");
        let observable_view = view
            .nodes
            .iter()
            .find(|node| node.id == observable.0)
            .expect("observable node built");

        assert_eq!(view.nodes.len(), 5);
        assert_eq!(quantum_view.category, BloqNodeCategory::QuantumBlock);
        assert_eq!(observable_view.category, BloqNodeCategory::Observable);
        assert!(observable_view.moments.is_empty(), "no circuit to show");
        assert_eq!(
            observable_view.attributes,
            vec![
                ("kind".to_string(), "Observable".to_string()),
                ("index".to_string(), "0".to_string()),
                ("measurements".to_string(), "1".to_string()),
                ("sites".to_string(), "i0:m0".to_string()),
                ("operators".to_string(), "1".to_string()),
                (
                    "outputs".to_string(),
                    "Corrected: decoded parity; Flip: decoder correction".to_string()
                ),
            ]
        );
        assert_eq!(observable_view.operator_table.len(), 1);
        assert_eq!(observable_view.operator_table[0].0, "i0");
        assert_eq!(observable_view.operator_table[0].1, "Input");

        let fragment_view = view
            .nodes
            .iter()
            .find(|node| node.id == fragment.0)
            .unwrap();
        assert_eq!(fragment_view.category, BloqNodeCategory::Observable);
        assert!(fragment_view.label.contains("obs fragment"));
        for (from, to, kind) in [
            (fragment, observable, BloqEdgeKind::Compose),
            (observable, corrected, BloqEdgeKind::Value),
            (observable, flip, BloqEdgeKind::Flip),
        ] {
            assert!(
                view.edges
                    .iter()
                    .any(|edge| edge.from == from.0 && edge.to == to.0 && edge.kind == kind)
            );
        }
        let edge = view
            .edges
            .iter()
            .find(|edge| edge.from == quantum.0 && edge.to == observable.0);
        assert_eq!(edge.map(|edge| edge.kind), Some(BloqEdgeKind::Order));
    }

    #[test]
    fn region_body_nodes_are_lifted_with_parent_links_and_region_summary() {
        let mut program = Bloq::new();
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let template = program.add_template(BloqTemplate::new(template_circuit));

        let mut body = SubGraph::new();
        let mut quantum = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        quantum
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let body_quantum = body.add_node(quantum);
        let body_obs = body.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        body.add_edge(body_quantum, body_obs, BloqEdge::value(0));

        let region = program.add_node(BloqNode::region(bloq_ir::RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(body_obs.into()),
        }));
        let source_graph = BlockGraph::new();

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");

        let region_view = view
            .nodes
            .iter()
            .find(|node| node.id == region.0)
            .expect("region built");
        assert_eq!(region_view.category, BloqNodeCategory::Region);
        assert!(region_view.has_quantum_body);
        assert_eq!(
            region_view.label,
            format!("N{} · RepeatUntilSuccess", region.0)
        );
        assert!(
            region_view
                .attributes
                .iter()
                .any(|(field, _)| field == "restart_condition")
        );
        assert_eq!(region_view.num_qubits, 1);

        let children: Vec<_> = view
            .nodes
            .iter()
            .filter(|node| node.parent == Some(region.0))
            .collect();
        assert_eq!(children.len(), 2);
        let child_quantum = children
            .iter()
            .find(|node| node.category.is_quantum())
            .expect("quantum child lifted");
        let child_obs = children
            .iter()
            .find(|node| node.category == BloqNodeCategory::Observable)
            .expect("observable child lifted");
        assert!(
            child_quantum.id > region.0 && child_obs.id > region.0,
            "body view ids allocate past the top-level range"
        );
        assert!(
            !child_quantum.moments.is_empty(),
            "body quantum node keeps its inspectable circuit"
        );
        assert!(!child_obs.has_quantum_body);

        let edge = view
            .edges
            .iter()
            .find(|edge| edge.from == child_quantum.id && edge.to == child_obs.id)
            .expect("body edge remaps to the lifted view ids");
        assert_eq!(edge.kind, BloqEdgeKind::Value);

        let detslice = build_detslice_circuit_view(&program, &source_graph, IVec3::ZERO)
            .expect("build detector-slice view");
        assert!(detslice.edges.is_empty());
    }

    #[test]
    fn temporal_pipe_node_highlights_source_pipe() {
        let pipe = TemporalPipeRef {
            src: ivec3(0, 0, 0),
            dst: ivec3(0, 0, 1),
            hadamard: true,
        };
        let node = BloqNode::from_temporal_pipe(pipe);
        let mut program = Bloq::new();
        program.add_node(node);
        let source_graph = BlockGraph::new();

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");

        assert_eq!(
            view.nodes[0].source_elements,
            HashSet::from([GraphElement::Pipe(pipe.src, pipe.dst).canonical()])
        );
    }

    #[test]
    fn block_component_highlights_member_blocks_and_internal_pipes() {
        let node = BloqNode::from_members(vec![
            SourceBlockRef {
                pos: ivec3(0, 0, 0),
            },
            SourceBlockRef {
                pos: ivec3(1, 0, 0),
            },
        ]);
        let mut program = Bloq::new();
        program.add_node(node);
        let mut source_graph = BlockGraph::new();
        source_graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::XZZ)));
        source_graph.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        source_graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::XPLUS));

        let view =
            build_bloq_circuit_view(&program, &source_graph, IVec3::ZERO).expect("build view");

        assert!(
            view.nodes[0]
                .source_elements
                .contains(&GraphElement::Block(ivec3(0, 0, 0)))
        );
        assert!(
            view.nodes[0]
                .source_elements
                .contains(&GraphElement::Block(ivec3(1, 0, 0)))
        );
        assert!(
            view.nodes[0]
                .source_elements
                .contains(&GraphElement::Pipe(ivec3(0, 0, 0), ivec3(1, 0, 0)).canonical())
        );
    }

    #[test]
    fn source_hover_elements_shift_back_to_editor_coordinates() {
        let shifted_pipe = TemporalPipeRef {
            src: ivec3(0, 0, 0),
            dst: ivec3(0, 0, 1),
            hadamard: true,
        };
        let node = BloqNode::from_temporal_pipe(shifted_pipe);
        let mut program = Bloq::new();
        program.add_node(node);
        let output = TemporalPipeRef {
            dst: shifted_pipe.src,
            ..shifted_pipe
        };
        program.add_node(BloqNode::memory_padding(output, 1));
        let source_graph = BlockGraph::new();
        let editor_offset = ivec3(0, 0, -2);

        let view =
            build_bloq_circuit_view(&program, &source_graph, editor_offset).expect("build view");

        assert_eq!(
            view.nodes[0].source_elements,
            HashSet::from([GraphElement::Pipe(ivec3(0, 0, -2), ivec3(0, 0, -1)).canonical()])
        );
        assert_eq!(view.nodes[0].layer, -3);
        for node in &view.nodes[1..] {
            assert_eq!(
                node.source_elements,
                HashSet::from([GraphElement::Block(editor_offset)])
            );
        }
    }
}
