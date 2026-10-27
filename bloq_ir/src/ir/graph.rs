use std::fmt;
use std::ops::Index;
use std::sync::Arc;

use petgraph::Direction;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, Visitable};

use super::{BloqEdge, BloqNode, BloqNodeId, ClassicalExpr, ClassicalNode, QuantumNode, ValueRole};

mod classical_data;

/// A nested region body.
///
/// This graph uses the **same** global [`crate::BloqTemplatePool`] as the
/// enclosing [`Bloq`](crate::Bloq) — template ids resolve against
/// that shared pool, so a `SubGraph` carries none of its own. Edges do not cross
/// region boundaries. The body's declared result and boundary bindings form
/// its classical interface (WF-17/SEM-RVAL).
/// Clones share immutable storage. Mutable access copies only this graph level
/// when shared; nested bodies remain shared until they too are edited.
#[derive(Debug, Clone, Default)]
pub struct SubGraph {
    data: Arc<SubGraphData>,
}

#[derive(Debug, Clone, Default)]
struct SubGraphData {
    graph: StableDiGraph<BloqNode, BloqEdge>,
    value_output: Option<super::ValueRef>,
    boundary_outputs: Vec<BloqNodeId>,
}

/// Why a serialized graph level could not be rebuilt ([`rebuild_level`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LevelRebuildError {
    DuplicateNode(u32),
    /// The `index`-th edge (in listing order) names a vacant/out-of-range id.
    MissingEndpoint {
        index: usize,
        id: u32,
    },
    /// The id space is far sparser than any node-removing pass produces —
    /// rebuilding would materialize `holes` placeholder slots (cost is
    /// O(largest id), not O(node count)), so a crafted artifact with one huge
    /// id must be rejected here, before [`crate::Bloq::validate`] can run.
    ExcessiveVacancy {
        holes: usize,
        live: usize,
    },
}

impl fmt::Display for LevelRebuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "duplicate node n{id}"),
            Self::MissingEndpoint { id, .. } => {
                write!(f, "edge endpoint n{id} is not a live node")
            }
            Self::ExcessiveVacancy { holes, live } => write!(
                f,
                "node ids are implausibly sparse: {holes} vacant slots for {live} live nodes"
            ),
        }
    }
}

/// Decode-time cap on vacant node slots per level, over and above a 4× live
/// multiple. Holes only ever come from node-removing passes, which delete at
/// most a few times the surviving node count; this bound is far past that
/// while keeping placeholder materialization cheap.
const MAX_EXTRA_VACANCIES: usize = 1 << 16;

/// Rebuild one graph level from `(id, node)` pairs and `(source, target,
/// edge)` triples — the shared materialization step of both exchange formats.
///
/// Node ids are honored exactly: ids missing below the last live one become
/// vacant slots (a placeholder node is added at the slot, then removed).
/// Edges are added in the given order, so live-edge iteration order survives
/// a round trip. Vacancy past the last live node and the free-slot reuse
/// order are *not* representable — non-semantic history that only affects
/// the ids future `add_node`/`add_edge` calls hand out.
///
/// Vacancy is capped ([`LevelRebuildError::ExcessiveVacancy`]): rebuilding
/// costs O(largest id), so an untrusted artifact must not be able to demand
/// billions of placeholder slots with a single huge id.
pub(crate) fn rebuild_level(
    mut nodes: Vec<(u32, BloqNode)>,
    edges: Vec<(u32, u32, BloqEdge)>,
) -> Result<SubGraph, LevelRebuildError> {
    nodes.sort_by_key(|(id, _)| *id);
    if let Some(window) = nodes.windows(2).find(|window| window[0].0 == window[1].0) {
        return Err(LevelRebuildError::DuplicateNode(window[0].0));
    }
    let bound = nodes.last().map_or(0, |(id, _)| u64::from(*id) + 1);
    let live = nodes.len();
    let holes = bound - live as u64;
    let vacancy_error = || LevelRebuildError::ExcessiveVacancy {
        holes: usize::try_from(holes).unwrap_or(usize::MAX),
        live,
    };
    if holes
        > (live as u64)
            .saturating_mul(4)
            .saturating_add(MAX_EXTRA_VACANCIES as u64)
    {
        return Err(vacancy_error());
    }
    let bound = usize::try_from(bound).map_err(|_| vacancy_error())?;

    let mut level = SubGraph::new();
    level.graph_mut().reserve_exact_nodes(bound);
    let mut pending = nodes.into_iter().peekable();
    let mut placeholders = Vec::new();
    for slot in 0..bound {
        if pending.peek().is_some_and(|(id, _)| *id as usize == slot) {
            let (_, node) = pending.next().expect("peek said so");
            level.add_node(node);
        } else {
            placeholders.push(level.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })));
        }
    }
    drop(pending);
    for placeholder in placeholders {
        level
            .graph_mut()
            .remove_node(to_index(placeholder))
            .expect("placeholder was just added");
    }

    for (index, (source, target, _)) in edges.iter().enumerate() {
        check_edge_endpoints(&level, index, *source, *target)?;
    }
    level.graph_mut().reserve_exact_edges(edges.len());
    for (source, target, edge) in edges {
        level.add_edge(BloqNodeId(source), BloqNodeId(target), edge);
    }
    Ok(level)
}

fn check_edge_endpoints(
    level: &SubGraph,
    index: usize,
    source: u32,
    target: u32,
) -> Result<(), LevelRebuildError> {
    for id in [source, target] {
        if level.node(BloqNodeId(id)).is_none() {
            return Err(LevelRebuildError::MissingEndpoint { index, id });
        }
    }
    Ok(())
}

/// Canonical wire shape for a graph level: live nodes with explicit slot ids,
/// live edges in slot order. Deliberately independent of petgraph's own serde
/// layout (which also encodes vacant *edge* slots — non-semantic free-list
/// state that would make equal programs serialize unequally).
#[derive(serde::Serialize)]
struct SubGraphWireRef<'a> {
    #[serde(serialize_with = "serialize_nodes")]
    nodes: &'a SubGraph,
    #[serde(serialize_with = "serialize_edges")]
    edges: &'a SubGraph,
    value_output: Option<super::ValueRef>,
    boundary_outputs: &'a [BloqNodeId],
}

// Stream the known live counts, preserving slot order without temporary
// reference arrays. Iterator size hints include holes and are not exact.
fn serialize_nodes<S: serde::Serializer>(
    graph: &SubGraph,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut sequence = serializer.serialize_seq(Some(graph.node_count()))?;
    for (id, node) in graph.nodes() {
        sequence.serialize_element(&(id.0, node))?;
    }
    sequence.end()
}

fn serialize_edges<S: serde::Serializer>(
    graph: &SubGraph,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    if !serializer.is_human_readable() {
        return edge_runs::serialize(graph, serializer);
    }
    let mut sequence = serializer.serialize_seq(Some(graph.edge_count()))?;
    for edge in graph.edges() {
        sequence.serialize_element(&(edge.source.0, edge.target.0, edge.edge))?;
    }
    sequence.end()
}

#[derive(serde::Deserialize)]
struct SubGraphWire {
    nodes: Vec<(u32, BloqNode)>,
    edges: Vec<(u32, u32, BloqEdge)>,
    value_output: Option<super::ValueRef>,
    boundary_outputs: Vec<BloqNodeId>,
}

/// Binary-only edge packing. The in-memory graph and human-readable triples
/// retain every edge; runs only omit repeated target, role and consecutive slots.
mod edge_runs {
    use super::{BloqEdge, BloqNodeId, SubGraph, ValueRole, check_edge_endpoints};
    use crate::{ObservableOutput, QuantumEdge};
    use serde::de::{DeserializeSeed, EnumAccess, Error, SeqAccess, VariantAccess, Visitor};
    use serde::ser::SerializeSeq;

    #[derive(serde::Serialize)]
    enum Edge<'a> {
        Quantum(u32, u32, &'a QuantumEdge),
        Value(u32, u32, u32, &'a ValueRole, ObservableOutput),
        Compose(u32, u32, u32, &'a ValueRole),
        Order(u32, u32),
        ValueRun(u32, u32, &'a ValueRole, ObservableOutput, &'a [u32]),
    }

    #[derive(serde::Deserialize)]
    #[serde(field_identifier)]
    enum EdgeKind {
        Quantum,
        Value,
        Compose,
        Order,
        ValueRun,
    }

    pub(super) fn serialize<S: serde::Serializer>(
        graph: &SubGraph,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut count = 0usize;
        let mut previous: Option<(crate::BloqNodeId, u32, &ValueRole, ObservableOutput)> = None;
        for edge in graph.edges() {
            let next = match edge.edge {
                BloqEdge::Value { slot, role, output } => Some((edge.target, *slot, role, *output)),
                _ => None,
            };
            let continues = match (previous, next) {
                (
                    Some((target, slot, role, output)),
                    Some((next_target, next_slot, next_role, next_output)),
                ) => {
                    target == next_target
                        && role == next_role
                        && output == next_output
                        && slot.checked_add(1) == Some(next_slot)
                }
                _ => false,
            };
            count += usize::from(!continues);
            previous = next;
        }
        let mut sequence = serializer.serialize_seq(Some(count))?;
        let mut edges = graph.edges().peekable();
        // One reusable run buffer, bounded by the largest consecutive fan-in.
        let mut sources = Vec::new();
        while let Some(edge) = edges.next() {
            let encoded = match edge.edge {
                BloqEdge::Quantum(value) => Edge::Quantum(edge.source.0, edge.target.0, value),
                BloqEdge::Order => Edge::Order(edge.source.0, edge.target.0),
                BloqEdge::Compose { slot, role } => {
                    Edge::Compose(edge.source.0, edge.target.0, *slot, role)
                }
                BloqEdge::Value { slot, role, output } => {
                    sources.clear();
                    sources.push(edge.source.0);
                    while let Some(next) = edges.peek() {
                        let BloqEdge::Value {
                            slot: next_slot,
                            role: next_role,
                            output: next_output,
                        } = next.edge
                        else {
                            break;
                        };
                        if next.target != edge.target
                            || next_role != role
                            || next_output != output
                            || u32::try_from(sources.len())
                                .ok()
                                .and_then(|length| slot.checked_add(length))
                                != Some(*next_slot)
                        {
                            break;
                        }
                        sources.push(edges.next().expect("peeked edge").source.0);
                    }
                    if sources.len() == 1 {
                        Edge::Value(edge.source.0, edge.target.0, *slot, role, *output)
                    } else {
                        Edge::ValueRun(edge.target.0, *slot, role, *output, &sources)
                    }
                }
            };
            sequence.serialize_element(&encoded)?;
        }
        sequence.end()
    }

    fn add_edge<E: Error>(
        level: &mut SubGraph,
        source: u32,
        target: u32,
        edge: BloqEdge,
    ) -> Result<(), E> {
        check_edge_endpoints(level, level.edge_count(), source, target).map_err(E::custom)?;
        level.add_edge(BloqNodeId(source), BloqNodeId(target), edge);
        Ok(())
    }

    pub(super) struct EdgeSequence<'a>(pub(super) &'a mut SubGraph);
    impl<'de> DeserializeSeed<'de> for EdgeSequence<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
            deserializer.deserialize_seq(self)
        }
    }
    impl<'de> Visitor<'de> for EdgeSequence<'_> {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("graph edges and consecutive value-edge runs")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
            // An untrusted count is not permission to allocate the whole graph.
            let capacity = sequence
                .size_hint()
                .unwrap_or(0)
                .min((1 << 20) / size_of::<petgraph::graph::Edge<Option<BloqEdge>>>());
            self.0.graph_mut().reserve_exact_edges(capacity);
            while sequence.next_element_seed(EdgeSeed(self.0))?.is_some() {}
            self.0.graph_mut().shrink_to_fit_edges();
            Ok(())
        }
    }

    struct EdgeSeed<'a>(&'a mut SubGraph);
    impl<'de> DeserializeSeed<'de> for EdgeSeed<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
            deserializer.deserialize_enum(
                "Edge",
                &["Quantum", "Value", "Compose", "Order", "ValueRun"],
                self,
            )
        }
    }
    impl<'de> Visitor<'de> for EdgeSeed<'_> {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("one graph edge or value-edge run")
        }
        fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<(), A::Error> {
            let (kind, value) = data.variant::<EdgeKind>()?;
            match kind {
                EdgeKind::Quantum => {
                    let (source, target, edge) =
                        value.newtype_variant::<(u32, u32, QuantumEdge)>()?;
                    add_edge::<A::Error>(
                        self.0,
                        source,
                        target,
                        BloqEdge::Quantum(Box::new(edge)),
                    )?;
                }
                EdgeKind::Value => {
                    let (source, target, slot, role, output) =
                        value.newtype_variant::<(u32, u32, u32, ValueRole, ObservableOutput)>()?;
                    add_edge::<A::Error>(
                        self.0,
                        source,
                        target,
                        BloqEdge::Value { slot, role, output },
                    )?;
                }
                EdgeKind::Compose => {
                    let (source, target, slot, role) =
                        value.newtype_variant::<(u32, u32, u32, ValueRole)>()?;
                    add_edge::<A::Error>(self.0, source, target, BloqEdge::Compose { slot, role })?;
                }
                EdgeKind::Order => {
                    let (source, target) = value.newtype_variant::<(u32, u32)>()?;
                    add_edge::<A::Error>(self.0, source, target, BloqEdge::Order)?;
                }
                EdgeKind::ValueRun => return value.tuple_variant(5, Run(self.0)),
            }
            Ok(())
        }
    }

    struct Run<'a>(&'a mut SubGraph);
    impl<'de> Visitor<'de> for Run<'_> {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("target, first slot, role, output and source IDs")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
            let target = sequence
                .next_element()?
                .ok_or_else(|| A::Error::custom("value run needs a target"))?;
            let slot = sequence
                .next_element()?
                .ok_or_else(|| A::Error::custom("value run needs a slot"))?;
            let role = sequence
                .next_element()?
                .ok_or_else(|| A::Error::custom("value run needs a role"))?;
            let output = sequence
                .next_element()?
                .ok_or_else(|| A::Error::custom("value run needs an output"))?;
            sequence
                .next_element_seed(Sources {
                    edges: self.0,
                    target,
                    slot,
                    role,
                    output,
                })?
                .ok_or_else(|| A::Error::custom("value run needs source IDs"))
        }
    }

    struct Sources<'a> {
        edges: &'a mut SubGraph,
        target: u32,
        slot: u32,
        role: ValueRole,
        output: ObservableOutput,
    }
    impl<'de> DeserializeSeed<'de> for Sources<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
            deserializer.deserialize_seq(self)
        }
    }
    impl<'de> Visitor<'de> for Sources<'_> {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("at least two source IDs with consecutive slots")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
            let mut count = 0usize;
            while let Some(source) = sequence.next_element::<u32>()? {
                let offset = u32::try_from(count)
                    .map_err(|_| A::Error::custom("value run slot overflow"))?;
                let slot = self
                    .slot
                    .checked_add(offset)
                    .ok_or_else(|| A::Error::custom("value run slot overflow"))?;
                // Expand directly into graph storage, without an intermediate
                // edge or source vector.
                add_edge::<A::Error>(
                    self.edges,
                    source,
                    self.target,
                    BloqEdge::Value {
                        slot,
                        role: self.role.clone(),
                        output: self.output,
                    },
                )?;
                count += 1;
            }
            if count < 2 {
                return Err(A::Error::custom("value run needs at least two sources"));
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{BloqNode, ClassicalExpr, ClassicalNode, NodeProvenance, ValueRef};

        #[derive(Clone, serde::Serialize)]
        enum WireKind {
            // Keep the same variant index as the binary node-kind wire.
            #[allow(
                dead_code,
                reason = "the wire tag must match the production quantum variant"
            )]
            Quantum,
            Classical(u32),
        }

        #[derive(Clone, serde::Serialize)]
        struct WireNode {
            kind: WireKind,
            provenance: NodeProvenance,
            activation: Option<u32>,
        }

        struct Decoded(SubGraph);

        impl<'de> serde::Deserialize<'de> for Decoded {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let mut graph = SubGraph::new();
                for _ in 0..10 {
                    graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(false),
                    }));
                }
                EdgeSequence(&mut graph).deserialize(deserializer)?;
                Ok(Self(graph))
            }
        }

        fn encode(value: &impl serde::Serialize) -> Vec<u8> {
            postcard::to_extend(value, Vec::new()).unwrap()
        }

        #[test]
        fn value_runs_preserve_full_width_slots_duplicates_roles_and_edge_order() {
            let role = ValueRole::FeedbackFold { action: u32::MAX };
            let source = [9, 9];
            let encoded = encode(&vec![
                Edge::Value(1, 2, 7, &ValueRole::Data, ObservableOutput::Corrected),
                Edge::Order(2, 3),
                Edge::ValueRun(9, u32::MAX - 1, &role, ObservableOutput::Flip, &source),
                Edge::Value(
                    3,
                    9,
                    0,
                    &ValueRole::ReadoutFold,
                    ObservableOutput::Corrected,
                ),
            ]);
            let expected = vec![
                (1, 2, BloqEdge::value(7)),
                (2, 3, BloqEdge::Order),
                (
                    9,
                    9,
                    BloqEdge::Value {
                        slot: u32::MAX - 1,
                        role: role.clone(),
                        output: ObservableOutput::Flip,
                    },
                ),
                (
                    9,
                    9,
                    BloqEdge::Value {
                        slot: u32::MAX,
                        role,
                        output: ObservableOutput::Flip,
                    },
                ),
                (
                    3,
                    9,
                    BloqEdge::Value {
                        slot: 0,
                        role: ValueRole::ReadoutFold,
                        output: ObservableOutput::Corrected,
                    },
                ),
            ];
            assert_eq!(
                postcard::from_bytes::<Decoded>(&encoded)
                    .unwrap()
                    .0
                    .edges()
                    .map(|edge| (edge.source.0, edge.target.0, edge.edge.clone()))
                    .collect::<Vec<_>>(),
                expected
            );
            for end in 0..encoded.len() {
                assert!(postcard::from_bytes::<Decoded>(&encoded[..end]).is_err());
            }
        }

        #[test]
        fn value_run_decoder_rejects_invalid_lengths_slots_and_endpoints() {
            for (slot, sources) in [(0, vec![]), (0, vec![0]), (u32::MAX, vec![0, 1])] {
                let encoded = encode(&vec![Edge::ValueRun(
                    0,
                    slot,
                    &ValueRole::Data,
                    ObservableOutput::Flip,
                    &sources,
                )]);
                assert!(postcard::from_bytes::<Decoded>(&encoded).is_err());
            }
            // Declared counts without payload must not reserve those counts.
            for encoded in [
                encode(&usize::MAX),
                encode(&(
                    1usize,
                    4u32,
                    0u32,
                    0u32,
                    ValueRole::Data,
                    ObservableOutput::Flip,
                    usize::MAX,
                )),
                encode(&(1usize, 5u32)),
            ] {
                assert!(postcard::from_bytes::<Decoded>(&encoded).is_err());
            }
            #[derive(serde::Serialize)]
            struct InvalidLevel<'a> {
                classical: Vec<ClassicalNode>,
                nodes: Vec<(u32, WireNode)>,
                edges: Vec<Edge<'a>>,
                value_output: Option<ValueRef>,
                boundary_outputs: Vec<BloqNodeId>,
            }
            let mut wire = InvalidLevel {
                classical: vec![ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                }],
                nodes: vec![(
                    0,
                    WireNode {
                        kind: WireKind::Classical(0),
                        provenance: NodeProvenance::None,
                        activation: None,
                    },
                )],
                edges: vec![Edge::ValueRun(
                    0,
                    0,
                    &ValueRole::Data,
                    ObservableOutput::Flip,
                    &[0, 1],
                )],
                value_output: None,
                boundary_outputs: Vec::new(),
            };
            postcard::from_bytes::<SubGraph>(&encode(&wire)).unwrap_err();
            wire.edges.clear();
            postcard::from_bytes::<SubGraph>(&encode(&wire)).unwrap();
            wire.nodes.push(wire.nodes[0].clone());
            postcard::from_bytes::<SubGraph>(&encode(&wire)).unwrap_err();
            wire.nodes.pop();
            wire.nodes[0].0 = u32::MAX;
            postcard::from_bytes::<SubGraph>(&encode(&wire)).unwrap_err();
        }

        #[test]
        fn binary_graph_accepts_unsorted_nodes_and_checks_every_edge_endpoint() {
            let definitions = vec![ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            }];
            let node = WireNode {
                kind: WireKind::Classical(0),
                provenance: NodeProvenance::None,
                activation: None,
            };
            let nodes = vec![(2u32, node.clone()), (0u32, node)];
            let wire = (
                &definitions,
                &nodes,
                vec![Edge::ValueRun(
                    2,
                    0,
                    &ValueRole::Data,
                    ObservableOutput::Flip,
                    &[0, 0],
                )],
                Some(ValueRef::from(BloqNodeId(2))),
                vec![BloqNodeId(0)],
            );
            let bytes = encode(&wire);
            let mut restored = postcard::from_bytes::<SubGraph>(&bytes).unwrap();
            assert!(restored.node(BloqNodeId(1)).is_none());
            assert_eq!(restored.value_output(), Some(ValueRef::from(BloqNodeId(2))));
            assert_eq!(restored.boundary_outputs(), &[BloqNodeId(0)]);
            let mut reloaded = postcard::from_bytes::<SubGraph>(&encode(&restored)).unwrap();
            for level in [&mut restored, &mut reloaded] {
                level.add_edge(BloqNodeId(2), BloqNodeId(0), BloqEdge::Order);
            }
            assert_eq!(encode(&restored), encode(&reloaded));
            for end in 0..bytes.len() {
                postcard::from_bytes::<SubGraph>(&bytes[..end]).unwrap_err();
            }

            let quantum = QuantumEdge {
                pipes: Vec::new(),
                guard: None,
            };
            for edge in [
                Edge::Quantum(1, 2, &quantum),
                Edge::Value(0, 1, 0, &ValueRole::Data, ObservableOutput::Corrected),
                Edge::Order(1, 2),
                Edge::Compose(0, 1, 0, &ValueRole::Data),
                Edge::ValueRun(2, 0, &ValueRole::Data, ObservableOutput::Flip, &[0, 1]),
            ] {
                let invalid = encode(&(
                    &definitions,
                    &nodes,
                    vec![edge],
                    None::<ValueRef>,
                    Vec::<BloqNodeId>::new(),
                ));
                postcard::from_bytes::<SubGraph>(&invalid).unwrap_err();
            }
        }

        #[test]
        fn graph_runs_keep_node_holes_declarations_and_human_readable_triples() {
            let mut graph = SubGraph::new();
            let nodes = (0..6)
                .map(|_| {
                    graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(false),
                    }))
                })
                .collect::<Vec<_>>();
            graph.remove_node(nodes[1]).unwrap();
            for (source, slot, role) in [
                (0, 0, ValueRole::Data),
                (0, 1, ValueRole::Data),
                (2, 2, ValueRole::FeedbackFold { action: 9 }),
                (3, 3, ValueRole::FeedbackFold { action: 9 }),
                (2, 4, ValueRole::ReadoutFold),
                (2, 5, ValueRole::ReadoutFold),
                (3, u32::MAX, ValueRole::Data),
            ] {
                graph.add_edge(
                    nodes[source],
                    nodes[4],
                    BloqEdge::Value {
                        slot,
                        role,
                        output: if slot % 2 == 0 {
                            ObservableOutput::Flip
                        } else {
                            ObservableOutput::Corrected
                        },
                    },
                );
                if slot == 1 {
                    graph.add_edge(nodes[2], nodes[5], BloqEdge::Order);
                }
            }
            graph.add_edge(nodes[0], nodes[4], BloqEdge::compose(100));
            graph.node_mut(nodes[4]).unwrap().kind = crate::BloqNodeKind::Classical(
                ClassicalNode::Compute {
                    expr: ClassicalExpr::parity([0, 1, 2, 3, 4, 5, u32::MAX], false),
                }
                .into(),
            );
            graph.set_value_output(Some(ValueRef::from(nodes[4])));
            let json = serde_json::to_value(&graph).unwrap();
            let triples = graph
                .edges()
                .map(|edge| (edge.source.0, edge.target.0, edge.edge))
                .collect::<Vec<_>>();
            assert_eq!(json["edges"], serde_json::to_value(triples).unwrap());
            let binary = encode(&graph);
            let mut unsorted_json = json.clone();
            unsorted_json["nodes"].as_array_mut().unwrap().reverse();
            for restored in [
                postcard::from_bytes::<SubGraph>(&binary).unwrap(),
                serde_json::from_value::<SubGraph>(json.clone()).unwrap(),
                serde_json::from_value::<SubGraph>(unsorted_json).unwrap(),
            ] {
                assert!(restored.node(nodes[1]).is_none());
                assert_eq!(restored.value_output(), Some(ValueRef::from(nodes[4])));
                assert_eq!(serde_json::to_value(&restored).unwrap(), json);
                assert_eq!(encode(&restored), binary);
            }
        }
    }
}

impl serde::Serialize for SubGraph {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !serializer.is_human_readable() {
            return classical_data::serialize(self, serializer);
        }
        SubGraphWireRef {
            nodes: self,
            edges: self,
            value_output: self.value_output(),
            boundary_outputs: self.boundary_outputs(),
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SubGraph {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return deserializer.deserialize_struct(
                "SubGraphWire",
                &[
                    "classical",
                    "nodes",
                    "edges",
                    "value_output",
                    "boundary_outputs",
                ],
                BinaryGraph,
            );
        }
        let wire = SubGraphWire::deserialize(deserializer)?;
        let mut level = rebuild_level(wire.nodes, wire.edges).map_err(serde::de::Error::custom)?;
        level.set_value_output(wire.value_output);
        level.set_boundary_outputs(wire.boundary_outputs);
        level.share_local_classical_data();
        Ok(level)
    }
}

struct BinaryGraph;

impl<'de> serde::de::Visitor<'de> for BinaryGraph {
    type Value = SubGraph;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("graph nodes, edges and output declarations")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> Result<SubGraph, A::Error> {
        use serde::de::Error;

        let nodes = classical_data::read_nodes(&mut sequence)?;
        // Reuse the shared node-id and vacancy checks, then release the staged
        // node buffer before allocating any decoded edges.
        let mut level = rebuild_level(nodes, Vec::new()).map_err(A::Error::custom)?;
        sequence
            .next_element_seed(edge_runs::EdgeSequence(&mut level))?
            .ok_or_else(|| A::Error::missing_field("edges"))?;
        level.set_value_output(
            sequence
                .next_element()?
                .ok_or_else(|| A::Error::missing_field("value_output"))?,
        );
        level.set_boundary_outputs(
            sequence
                .next_element()?
                .ok_or_else(|| A::Error::missing_field("boundary_outputs"))?,
        );
        Ok(level)
    }
}

/// A borrowed graph edge with id-space endpoints — the id-native view of one
/// [`BloqEdge`], so traversal never exposes petgraph types.
#[derive(Debug, Clone, Copy)]
pub struct BloqEdgeRef<'a> {
    /// Source node.
    pub source: BloqNodeId,
    /// Target node.
    pub target: BloqNodeId,
    /// Borrowed edge payload.
    pub edge: &'a BloqEdge,
}

/// Reusable visit-map scratch for batched reachability probes
/// ([`SubGraph::has_path_with_scratch`]). Opaque so petgraph stays sealed
/// behind the IR.
#[doc(hidden)]
#[derive(Debug)]
pub struct PathScratch(
    petgraph::algo::DfsSpace<NodeIndex, <StableDiGraph<BloqNode, BloqEdge> as Visitable>::Map>,
);

fn to_index(id: BloqNodeId) -> NodeIndex {
    NodeIndex::new(id.0 as usize)
}

fn to_id(index: NodeIndex) -> BloqNodeId {
    BloqNodeId(index.index() as u32)
}

fn to_edge_ref(edge: petgraph::stable_graph::EdgeReference<'_, BloqEdge>) -> BloqEdgeRef<'_> {
    BloqEdgeRef {
        source: to_id(edge.source()),
        target: to_id(edge.target()),
        edge: edge.weight(),
    }
}

/// One `Value` input of a node.
///
/// This records the producer wired into `slot` and the edge's role.
/// [`ValueRole::FeedbackFold`] marks runtime Pauli-frame bookkeeping;
/// [`ValueRole::ReadoutFold`] carries an earlier corrected parity. Both are
/// omitted from the new decoder query's physical symptoms.
// Spec rule WF-11.
#[derive(Debug, Clone, Copy)]
pub struct ValueInput<'a> {
    /// Consumer input slot.
    pub slot: u32,
    /// Value producer.
    pub producer: BloqNodeId,
    /// Edge provenance.
    pub role: &'a ValueRole,
    /// Selected Boolean output, or None for structural recipe composition.
    pub output: Option<super::ObservableOutput>,
}

impl ValueInput<'_> {
    /// The selected runtime bit, absent for recipe composition.
    pub fn value_ref(self) -> Option<super::ValueRef> {
        self.output.map(|output| super::ValueRef {
            node: self.producer,
            output,
        })
    }
}

impl SubGraph {
    /// Create an empty graph level.
    pub fn new() -> Self {
        Self::default()
    }

    /// The declared bit result, or `None` for constant false (SEM-RVAL).
    /// Local consumers do not affect this binding.
    pub fn value_output(&self) -> Option<super::ValueRef> {
        self.data.value_output
    }

    /// Select the body's bit result. Validation checks that the node exists
    /// and produces a bit (WF-17).
    pub fn set_value_output(&mut self, output: Option<super::ValueRef>) {
        Arc::make_mut(&mut self.data).value_output = output;
    }

    /// Declared boundary-binding producers, independently of the bit result
    /// and any local consumers. Each id names an `Observable` or region (WF-17).
    pub fn boundary_outputs(&self) -> &[BloqNodeId] {
        &self.data.boundary_outputs
    }

    /// Select the body's boundary bindings. Validation checks producer kinds
    /// and rejects repeated ids; operator multiplicity within an Observable stays.
    pub fn set_boundary_outputs(&mut self, outputs: Vec<BloqNodeId>) {
        Arc::make_mut(&mut self.data).boundary_outputs = outputs;
    }

    /// Add a node and return its level-local id.
    pub fn add_node(&mut self, node: BloqNode) -> BloqNodeId {
        to_id(self.graph_mut().add_node(node))
    }

    /// Adds an edge `from -> to`. Panics when either endpoint id is stale
    /// (its node was removed and not reused).
    ///
    /// # Panics
    ///
    /// Panics if either endpoint is stale.
    pub fn add_edge(&mut self, from: BloqNodeId, to: BloqNodeId, edge: BloqEdge) {
        self.graph_mut()
            .add_edge(to_index(from), to_index(to), edge);
    }

    /// The node at `id`, or `None` when the id is stale.
    pub fn node(&self, id: BloqNodeId) -> Option<&BloqNode> {
        self.graph().node_weight(to_index(id))
    }

    /// Mutable access to the node at `id`, or `None` when the id is stale — the
    /// mutable pair of [`Self::node`]. Rewrites a node's kind in place (e.g.
    /// rewriting a region predicate) without disturbing its edges.
    pub fn node_mut(&mut self, id: BloqNodeId) -> Option<&mut BloqNode> {
        self.graph_mut().node_weight_mut(to_index(id))
    }

    /// Remove the node at `id` (and its incident edges), returning it, or `None`
    /// when the id is stale. Other node ids are unchanged (the id space is
    /// stable). Used to excise a classical sink — e.g. a dynamically-corrected
    /// observable the static fold cannot resolve — without renumbering.
    /// Declarations referencing this id are cleared along with incident edges.
    /// Rewrites preserving an output must retarget it before removal.
    pub fn remove_node(&mut self, id: BloqNodeId) -> Option<BloqNode> {
        let data = Arc::make_mut(&mut self.data);
        let node = data.graph.remove_node(to_index(id))?;
        if data.value_output.is_some_and(|value| value.node == id) {
            data.value_output = None;
        }
        data.boundary_outputs.retain(|&output| output != id);
        Some(node)
    }

    /// Every node id at this level, in ascending id order.
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = BloqNodeId> + '_ {
        self.graph().node_indices().map(to_id)
    }

    /// Every `(id, node)` at this level, in ascending id order.
    pub fn nodes(&self) -> impl DoubleEndedIterator<Item = (BloqNodeId, &BloqNode)> + '_ {
        self.graph()
            .node_indices()
            .map(|index| (to_id(index), &self.graph()[index]))
    }

    /// This level's quantum (block-bearing) nodes, skipping the classical and
    /// region nodes that observable lowering and control flow introduce.
    pub fn quantum_nodes(&self) -> impl Iterator<Item = (BloqNodeId, &QuantumNode)> + '_ {
        self.nodes()
            .filter_map(|(id, node)| Some((id, node.try_quantum()?)))
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.graph().node_count()
    }

    /// Number of live edges.
    pub fn edge_count(&self) -> usize {
        self.graph().edge_count()
    }

    /// Every edge at this level.
    pub fn edges(&self) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_ {
        self.graph().edge_references().map(to_edge_ref)
    }

    /// Rewrite edge payloads without changing node identities or connectivity.
    pub fn for_each_edge_mut(&mut self, mut edit: impl FnMut(&mut BloqEdge)) {
        for edge in self.graph_mut().edge_weights_mut() {
            edit(edge);
        }
    }

    /// The edges into `id`. Empty for a stale id.
    pub fn incoming(&self, id: BloqNodeId) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_ {
        self.graph()
            .edges_directed(to_index(id), Direction::Incoming)
            .map(to_edge_ref)
    }

    /// The edges out of `id`. Empty for a stale id.
    pub fn outgoing(&self, id: BloqNodeId) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_ {
        self.graph()
            .edges_directed(to_index(id), Direction::Outgoing)
            .map(to_edge_ref)
    }

    /// Every edge `from -> to` (parallel edges are legal).
    pub fn edges_between(
        &self,
        from: BloqNodeId,
        to: BloqNodeId,
    ) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_ {
        self.graph()
            .edges_connecting(to_index(from), to_index(to))
            .map(to_edge_ref)
    }

    /// Whether a directed path `from -> to` exists (any edge kind).
    pub fn has_path(&self, from: BloqNodeId, to: BloqNodeId) -> bool {
        petgraph::algo::has_path_connecting(self.graph(), to_index(from), to_index(to), None)
    }

    /// Fresh scratch for [`Self::has_path_with_scratch`], sized to this level.
    pub fn path_scratch(&self) -> PathScratch {
        PathScratch(petgraph::algo::DfsSpace::new(self.graph()))
    }

    /// [`Self::has_path`] with caller-provided scratch: a batch of probes shares
    /// one visit-map allocation instead of re-allocating per call. Each probe
    /// still walks the graph (the map is reset, not memoized).
    pub fn has_path_with_scratch(
        &self,
        from: BloqNodeId,
        to: BloqNodeId,
        scratch: &mut PathScratch,
    ) -> bool {
        petgraph::algo::has_path_connecting(
            self.graph(),
            to_index(from),
            to_index(to),
            Some(&mut scratch.0),
        )
    }

    /// Search a consumer's ancestors without walking its producers' other
    /// descendants. Quantum measurement dependencies often precede large
    /// classical fanouts, which cannot help establish the required order.
    /// Returns the same result as [`Self::has_path_with_scratch`].
    pub fn has_path_backwards_with_scratch(
        &self,
        from: BloqNodeId,
        to: BloqNodeId,
        scratch: &mut PathScratch,
    ) -> bool {
        petgraph::algo::has_path_connecting(
            petgraph::visit::Reversed(self.graph()),
            to_index(to),
            to_index(from),
            Some(&mut scratch.0),
        )
    }

    /// The producers wired into `id`'s `Value` slots.
    /// Unordered; collect into a map or sort by slot as needed.
    // Spec rule WF-11.
    pub fn value_inputs(&self, id: BloqNodeId) -> impl Iterator<Item = ValueInput<'_>> + '_ {
        self.incoming(id).filter_map(|edge| match edge.edge {
            BloqEdge::Value { slot, role, output } => Some(ValueInput {
                slot: *slot,
                producer: edge.source,
                role,
                output: Some(*output),
            }),
            _ => None,
        })
    }

    /// Data operands, excluding a node's separate activation slot.
    pub fn data_inputs(&self, id: BloqNodeId) -> impl Iterator<Item = ValueInput<'_>> + '_ {
        self.incoming(id).filter_map(move |edge| {
            let (slot, role, output) = match edge.edge {
                BloqEdge::Value { slot, role, output } => (*slot, role, Some(*output)),
                BloqEdge::Compose { slot, role } => (*slot, role, None),
                _ => return None,
            };
            (Some(slot) != self[id].activation).then_some(ValueInput {
                slot,
                producer: edge.source,
                role,
                output,
            })
        })
    }

    /// The consumers of `id`'s single `Value` output: `(consumer, slot)`.
    pub fn value_consumers(&self, id: BloqNodeId) -> impl Iterator<Item = (BloqNodeId, u32)> + '_ {
        self.outgoing(id).filter_map(|edge| match edge.edge {
            BloqEdge::Value { slot, .. } => Some((edge.target, *slot)),
            _ => None,
        })
    }

    /// Runtime and recipe consumers, sharing the operand-slot namespace.
    pub fn data_consumers(&self, id: BloqNodeId) -> impl Iterator<Item = (BloqNodeId, u32)> + '_ {
        self.outgoing(id).filter_map(|edge| match edge.edge {
            BloqEdge::Value { slot, .. } | BloqEdge::Compose { slot, .. } => {
                Some((edge.target, *slot))
            }
            _ => None,
        })
    }

    /// Consumers including this node's recipe content.
    pub fn compose_consumers(
        &self,
        id: BloqNodeId,
    ) -> impl Iterator<Item = (BloqNodeId, u32)> + '_ {
        self.outgoing(id).filter_map(|edge| match edge.edge {
            BloqEdge::Compose { slot, .. } => Some((edge.target, *slot)),
            _ => None,
        })
    }

    /// This level's open `Value` producers: value-producing classical or
    /// region nodes whose `Value` no in-level consumer reads. This diagnostic
    /// query is independent of the declared outputs (WF-17/SEM-RVAL).
    pub fn open_value_producers(&self) -> impl Iterator<Item = BloqNodeId> + '_ {
        self.nodes()
            .filter(|(id, node)| {
                (node.try_region().is_some()
                    || matches!(
                        node.try_classical(),
                        Some(ClassicalNode::Compute { .. } | ClassicalNode::Observable { .. })
                    ))
                    && self.value_consumers(*id).next().is_none()
            })
            .map(|(id, _)| id)
    }

    /// Open producers with a corrected Boolean output.
    /// This diagnostic query does not select the declared bit result.
    pub fn open_bit_producers(&self) -> impl Iterator<Item = BloqNodeId> + '_ {
        self.open_value_producers()
    }

    /// The body's deterministic Kahn topological order — the same schedule
    /// [`crate::Bloq::deterministic_emit_order`] gives the top level, for
    /// walkers that execute or emit a region body.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CycleDetected`] if the body has a cycle.
    pub fn deterministic_emit_order(&self) -> Result<Vec<BloqNodeId>, crate::CycleDetected> {
        super::program::deterministic_emit_order_of(self.graph())
    }

    /// Whether the body is acyclic — [`Self::deterministic_emit_order`]'s
    /// success condition, without materializing the order.
    pub fn is_acyclic(&self) -> bool {
        super::program::is_acyclic_of(self.graph(), &[])
    }

    /// Whether the body stays acyclic when `extra` directed edges are added
    /// (duplicates and parallel edges are allowed). One O(V+E) answer for
    /// callers probing candidate insertions, instead of reachability walks
    /// or a graph copy per candidate.
    pub fn is_acyclic_with_edges(&self, extra: &[(BloqNodeId, BloqNodeId)]) -> bool {
        super::program::is_acyclic_of(self.graph(), extra)
    }

    pub(crate) fn graph(&self) -> &StableDiGraph<BloqNode, BloqEdge> {
        &self.data.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut StableDiGraph<BloqNode, BloqEdge> {
        &mut Arc::make_mut(&mut self.data).graph
    }
}

/// The panicking twin of [`SubGraph::node`]: panics on a stale id, for passes
/// that hold ids they know are live.
impl Index<BloqNodeId> for SubGraph {
    type Output = BloqNode;

    fn index(&self, id: BloqNodeId) -> &Self::Output {
        self.node(id)
            .expect("BloqNodeId is live in this graph level (spec rule SEM-ID)")
    }
}
