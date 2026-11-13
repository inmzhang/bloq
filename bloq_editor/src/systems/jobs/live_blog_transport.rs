//! Private browser-worker transport. Flat graphs use compact geometry/action
//! DTOs; hierarchy, branches and external inputs use checked semantic snapshots.
//! No BLOG parse or repeated full geometry certification runs on the UI thread.

use bloq_graph::{Block, BlockGraph, BlockKind, Direction, Pipe, PortRole};
use glam::IVec3;
use serde::{Deserialize, Serialize};

mod snapshot;
use snapshot::SnapshotWire;

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(tag = "encoding", content = "data")]
enum GraphWire {
    Geometry {
        blocks: Vec<BlockWire>,
        pipes: Vec<PipeWire>,
    },
    Document {
        geometry: Box<GraphWire>,
        actions: Vec<ActionWire>,
        definition: Option<DefinitionWire>,
    },
    Snapshot(Box<SnapshotWire>),
}

#[derive(Serialize, Deserialize, PartialEq)]
struct BlockWire {
    position: IVec3,
    kind: KindWire,
    height: Option<String>,
    tag: String,
    color: Option<[u8; 3]>,
    role: Option<String>,
}

#[derive(Serialize, Deserialize, PartialEq)]
enum KindWire {
    Plain(String),
    Walking {
        boundary: String,
        movement: glam::IVec2,
    },
    Rotation {
        basis: String,
        movement: glam::IVec2,
    },
}

impl KindWire {
    fn encode(kind: BlockKind) -> Self {
        match kind {
            BlockKind::Walking(kind) => Self::Walking {
                boundary: kind.boundary().to_string(),
                movement: kind.movement(),
            },
            BlockKind::PatchRotation(kind) => Self::Rotation {
                basis: kind.basis().to_string(),
                movement: kind.movement(),
            },
            _ => Self::Plain(kind.to_string()),
        }
    }

    fn decode(self) -> Result<BlockKind, String> {
        match self {
            Self::Plain(kind) => kind
                .parse()
                .map_err(|e: bloq_graph::BlockError| e.to_string()),
            Self::Walking { boundary, movement } => Ok(BlockKind::Walking(
                bloq_graph::WalkingKind::new(
                    boundary
                        .parse::<bloq_graph::WalkingBoundaryKind>()
                        .map_err(|e| e.to_string())?,
                    movement,
                )
                .map_err(|e| e.to_string())?,
            )),
            Self::Rotation { basis, movement } => Ok(BlockKind::PatchRotation(
                bloq_graph::PatchRotationKind::new(
                    basis
                        .parse::<bloq_graph::Basis>()
                        .map_err(|e| e.to_string())?,
                    movement,
                )
                .map_err(|e| e.to_string())?,
            )),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
struct PipeWire {
    source: IVec3,
    direction: String,
    hadamard: bool,
    tag: String,
}

impl BlockWire {
    fn encode(block: &Block) -> Self {
        Self {
            position: block.pos(),
            kind: KindWire::encode(block.kind()),
            height: block.kind().is_cube().then(|| block.height().to_string()),
            tag: block.tag().unwrap_or("").into(),
            color: block.port_color().map(|c| [c.r, c.g, c.b]),
            role: block.port_role().map(|r| r.to_string()),
        }
    }
    fn decode(self) -> Result<Block, String> {
        let mut block = Block::new(self.position, self.kind.decode()?);
        if let Some(height) = self.height {
            block = block
                .with_height(
                    height
                        .parse()
                        .map_err(|e: bloq_graph::BlockError| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
        }
        block = block.with_tag(self.tag).map_err(|e| e.to_string())?;
        if let Some(color) = self.color {
            block = block.with_port_color(color).map_err(|e| e.to_string())?;
        }
        if let Some(role) = self.role {
            block = block
                .with_port_role(role.parse::<PortRole>().map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        }
        Ok(block)
    }
}
impl PipeWire {
    fn encode(pipe: &Pipe) -> Self {
        Self {
            source: pipe.src(),
            direction: pipe.dir().to_string(),
            hadamard: pipe.is_hadamard(),
            tag: pipe.tag().unwrap_or("").into(),
        }
    }
    fn decode(self) -> Result<Pipe, String> {
        let mut pipe = Pipe::new(
            self.source,
            self.direction
                .parse::<Direction>()
                .map_err(|e| e.to_string())?,
        );
        if self.hadamard {
            pipe = pipe.with_hadamard();
        }
        pipe.with_tag(self.tag).map_err(|e| e.to_string())
    }
}

// Editor-local mirrors deliberately carry semantic source fields, not DAG
// caches or validation internals. Only no-external-input, branch-free documents
// are hydrated through the public deferred-action API.
#[derive(Serialize, Deserialize, PartialEq)]
enum ExprWire {
    Var(String),
    Not(Box<ExprWire>),
    Binary(OperatorWire, Box<ExprWire>, Box<ExprWire>),
}

#[derive(Serialize, Deserialize, PartialEq)]
enum OperatorWire {
    Xor,
    And,
    Or,
}

impl ExprWire {
    fn encode(expr: &bloq_graph::Expr) -> Self {
        use bloq_graph::{BinaryOp, Expr};
        match expr {
            Expr::Var(name) => Self::Var(name.clone()),
            Expr::Not(inner) => Self::Not(Box::new(Self::encode(inner))),
            Expr::Binary(op, left, right) => Self::Binary(
                match op {
                    BinaryOp::Xor => OperatorWire::Xor,
                    BinaryOp::And => OperatorWire::And,
                    BinaryOp::Or => OperatorWire::Or,
                },
                Box::new(Self::encode(left)),
                Box::new(Self::encode(right)),
            ),
        }
    }
    fn decode(self) -> bloq_graph::Expr {
        use bloq_graph::{BinaryOp, Expr};
        match self {
            Self::Var(name) => Expr::Var(name),
            Self::Not(inner) => Expr::Not(Box::new(inner.decode())),
            Self::Binary(op, left, right) => Expr::Binary(
                match op {
                    OperatorWire::Xor => BinaryOp::Xor,
                    OperatorWire::And => BinaryOp::And,
                    OperatorWire::Or => BinaryOp::Or,
                },
                Box::new(left.decode()),
                Box::new(right.decode()),
            ),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
enum MeasureWire {
    Node(IVec3),
    Edge(IVec3, String),
}

#[derive(Serialize, Deserialize, PartialEq)]
struct FeedbackWire {
    pauli: String,
    target: IVec3,
    direction: Option<String>,
}

#[derive(Serialize, Deserialize, PartialEq)]
enum ActionWire {
    Let(String, ExprWire),
    Measure(MeasureWire, String),
    Discard(ExprWire),
    Resolve(IVec3, ExprWire),
    Branch(IVec3, ExprWire),
    Feedback(Vec<FeedbackWire>, Option<ExprWire>),
}

impl ActionWire {
    fn encode(action: &bloq_graph::Action) -> Self {
        use bloq_graph::{Action, MeasureTarget};
        match action {
            Action::Let { name, expr } => Self::Let(name.clone(), ExprWire::encode(expr)),
            Action::Measure { target, name } => Self::Measure(
                match target {
                    MeasureTarget::Node(pos) => MeasureWire::Node(*pos),
                    MeasureTarget::Edge { src, dir } => MeasureWire::Edge(*src, dir.to_string()),
                },
                name.clone(),
            ),
            Action::DiscardIf(expr) => Self::Discard(ExprWire::encode(expr)),
            Action::Resolve { target, condition } => {
                Self::Resolve(*target, ExprWire::encode(condition))
            }
            Action::Branch { target, condition } => {
                Self::Branch(*target, ExprWire::encode(condition))
            }
            Action::Feedback { targets, condition } => Self::Feedback(
                targets
                    .iter()
                    .map(|t| FeedbackWire {
                        pauli: t.pauli.to_string(),
                        target: t.target,
                        direction: t.direction.map(|d| d.to_string()),
                    })
                    .collect(),
                condition.as_ref().map(ExprWire::encode),
            ),
        }
    }
    fn decode(self) -> Result<bloq_graph::Action, String> {
        use bloq_graph::{Action, FeedbackTarget, MeasureTarget};
        Ok(match self {
            Self::Let(name, expr) => Action::Let {
                name,
                expr: expr.decode(),
            },
            Self::Measure(target, name) => Action::Measure {
                name,
                target: match target {
                    MeasureWire::Node(pos) => MeasureTarget::Node(pos),
                    MeasureWire::Edge(src, dir) => MeasureTarget::Edge {
                        src,
                        dir: dir.parse::<Direction>().map_err(|e| e.to_string())?,
                    },
                },
            },
            Self::Discard(expr) => Action::DiscardIf(expr.decode()),
            Self::Resolve(target, condition) => Action::Resolve {
                target,
                condition: condition.decode(),
            },
            Self::Branch(target, condition) => Action::Branch {
                target,
                condition: condition.decode(),
            },
            Self::Feedback(targets, condition) => Action::Feedback {
                targets: targets
                    .into_iter()
                    .map(|t| {
                        Ok(FeedbackTarget {
                            pauli: t
                                .pauli
                                .parse::<bloq_graph::PauliBasis>()
                                .map_err(|e| e.to_string())?,
                            target: t.target,
                            direction: t
                                .direction
                                .map(|d| d.parse::<Direction>().map_err(|e| e.to_string()))
                                .transpose()?,
                        })
                    })
                    .collect::<Result<_, String>>()?,
                condition: condition.map(ExprWire::decode),
            },
        })
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
struct QuantumPortWire {
    name: String,
    position: IVec3,
    input: bool,
    resource: String,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct DefinitionWire {
    name: String,
    quantum_ports: Vec<QuantumPortWire>,
    bit_outputs: Vec<(String, ExprWire)>,
}

impl DefinitionWire {
    fn encode(graph: &BlockGraph) -> Self {
        Self {
            name: graph.name.clone(),
            quantum_ports: graph
                .interface
                .quantum_ports
                .iter()
                .map(|port| QuantumPortWire {
                    name: port.name.clone(),
                    position: port.position,
                    input: port.direction == bloq_graph::PortDirection::Input,
                    resource: port.resource_type.clone(),
                })
                .collect(),
            bit_outputs: graph
                .interface
                .bit_outputs
                .iter()
                .map(|output| (output.name.clone(), ExprWire::encode(&output.expr)))
                .collect(),
        }
    }
    fn apply(self, body: BlockGraph) -> BlockGraph {
        use bloq_graph::{BitOutput, ModuleInterface, PortDirection, QuantumPort};
        BlockGraph::definition(
            self.name,
            body,
            ModuleInterface {
                quantum_ports: self
                    .quantum_ports
                    .into_iter()
                    .map(|port| QuantumPort {
                        name: port.name,
                        position: port.position,
                        direction: if port.input {
                            PortDirection::Input
                        } else {
                            PortDirection::Output
                        },
                        resource_type: port.resource,
                    })
                    .collect(),
                bit_inputs: vec![],
                bit_outputs: self
                    .bit_outputs
                    .into_iter()
                    .map(|(name, expr)| BitOutput {
                        name,
                        expr: expr.decode(),
                    })
                    .collect(),
            },
            vec![],
            vec![],
            vec![],
        )
    }
}

impl GraphWire {
    fn encode(graph: &BlockGraph) -> Self {
        if graph.modules().len() != 1
            || !graph.instances.is_empty()
            || !graph.quantum_connections.is_empty()
            || !graph.bit_bindings.is_empty()
            || !graph.interface.bit_inputs.is_empty()
            || !graph.branch_definitions().is_empty()
            || graph.action_graph().inputs().next().is_some()
            || graph
                .actions()
                .iter()
                .any(|action| matches!(action, bloq_graph::Action::Branch { .. }))
        {
            return Self::Snapshot(Box::new(SnapshotWire::encode(graph.source_snapshot())));
        }
        let geometry = Self::Geometry {
            blocks: graph.blocks().map(BlockWire::encode).collect(),
            pipes: graph.pipes().map(PipeWire::encode).collect(),
        };
        if graph.has_actions() || graph.has_module_structure() {
            Self::Document {
                geometry: Box::new(geometry),
                actions: graph.actions().iter().map(ActionWire::encode).collect(),
                definition: graph
                    .has_module_structure()
                    .then(|| DefinitionWire::encode(graph)),
            }
        } else {
            geometry
        }
    }

    fn decode(self) -> Result<BlockGraph, String> {
        match self {
            Self::Snapshot(snapshot) => {
                BlockGraph::from_source_snapshot(snapshot.decode()?).map_err(|e| e.to_string())
            }
            Self::Document {
                geometry,
                actions,
                definition,
            } => {
                let mut graph = geometry.decode()?;
                if let Some(definition) = definition {
                    graph = definition.apply(graph);
                }
                let actions = actions
                    .into_iter()
                    .map(ActionWire::decode)
                    .collect::<Result<Vec<_>, _>>()?;
                if !actions.is_empty() {
                    graph
                        .set_actions_deferred(actions)
                        .map_err(|e| e.to_string())?;
                }
                Ok(graph)
            }
            Self::Geometry { blocks, pipes } => {
                let mut graph = BlockGraph::default();
                for wire in blocks {
                    graph
                        .try_add_block(wire.decode()?)
                        .map_err(|e| e.to_string())?;
                }
                for wire in pipes {
                    graph
                        .try_add_pipe(wire.decode()?)
                        .map_err(|e| e.to_string())?;
                }
                Ok(graph)
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
enum ResultWire {
    Changed(Box<GraphWire>, Option<GraphWire>),
    Unchanged,
}

#[derive(Serialize, Deserialize)]
struct AnalysisWire {
    language: crate::systems::ui::blog_buffer::BlogLanguageState,
    result: Result<ResultWire, String>,
}

pub(super) fn encode(analysis: super::LiveBlogAnalysis) -> Result<String, String> {
    serde_json::to_string(&AnalysisWire {
        language: analysis.language,
        result: analysis.result.map(|result| match result {
            super::LiveBlogResult::Unchanged => ResultWire::Unchanged,
            super::LiveBlogResult::Changed(parsed) => {
                let graph = GraphWire::encode(&parsed.graph);
                let source = GraphWire::encode(&parsed.source_graph);
                let source = (source != graph).then_some(source);
                ResultWire::Changed(Box::new(graph), source)
            }
        }),
    })
    .map_err(|e| e.to_string())
}

pub(super) fn decode(json: &str) -> Result<super::LiveBlogAnalysis, String> {
    let wire: AnalysisWire = serde_json::from_str(json).map_err(|e| e.to_string())?;
    Ok(super::LiveBlogAnalysis {
        language: wire.language,
        result: wire.result.and_then(|result| match result {
            ResultWire::Unchanged => Ok(super::LiveBlogResult::Unchanged),
            ResultWire::Changed(graph, source) => {
                let graph = graph.decode()?;
                let source_graph = match source {
                    Some(source) => source.decode()?,
                    None => graph.clone(),
                };
                Ok(super::LiveBlogResult::Changed(Box::new(
                    super::ParsedBlog {
                        graph,
                        source_graph,
                    },
                )))
            }
        }),
    })
}

/// Checked recovery transport also retains incomplete authored actions.
#[derive(Serialize, Deserialize)]
pub(crate) struct RecoveryGraph(SnapshotWire);

impl RecoveryGraph {
    pub(crate) fn capture(graph: &BlockGraph) -> Self {
        Self(SnapshotWire::encode(graph.source_snapshot()))
    }

    pub(crate) fn restore(self) -> Result<BlockGraph, String> {
        BlockGraph::from_source_snapshot_lenient(self.0.decode()?)
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::{CubeKind, ModuleInterface};

    fn roundtrip(graph: &BlockGraph) -> BlockGraph {
        let json = serde_json::to_string(&GraphWire::encode(graph)).unwrap();
        serde_json::from_str::<GraphWire>(&json)
            .unwrap()
            .decode()
            .unwrap()
    }

    fn hierarchy() -> BlockGraph {
        let gate = bloq_graph::GalleryItem::CNOT
            .build()
            .to_blog_text()
            .replace("module main", "module Gate");
        BlockGraph::from_text(
            &(gate
                + r#"
module main {
  in control_in: data = 0
  out control_out: data = 1
  in target_in: data = 2
  out target_out: data = 3
  gate: Gate @ [0, 0, 0]
  0: Port [0, 0, 0] role=input
  1: Port [0, 0, 3] role=output
  2: Port [1, 1, 0] role=input
  3: Port [1, 1, 3] role=output
  0 -> gate.control_in
  gate.control_out -> 1
  2 -> gate.target_in
  gate.target_out -> 3
}
"#),
        )
        .unwrap()
    }

    #[test]
    fn worker_snapshot_hierarchy_roundtrip_preserves_all_source_fields() {
        let graph = hierarchy();
        assert!(matches!(GraphWire::encode(&graph), GraphWire::Snapshot(_)));
        let restored = roundtrip(&graph);
        assert_eq!(restored.to_blog_text(), graph.to_blog_text());
        assert_eq!(
            restored.flatten().unwrap().to_blog_text(),
            graph.flatten().unwrap().to_blog_text()
        );
    }

    #[test]
    fn worker_analysis_transport_roundtrips_success_and_diagnostics() {
        for source in ["BLOG 1.0\n0: ZXZ [0,0,0]\n", "invalid BLOG"] {
            let analysis = super::super::analyze_live_blog(source, 17);
            let expected_ok = analysis.result.is_ok();
            let encoded = encode(analysis).unwrap();
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded.result.is_ok(), expected_ok);
            assert_eq!(encode(decoded).unwrap(), encoded);
        }
    }

    #[test]
    fn worker_named_helper_requires_executable_main_before_transport() {
        let source = bloq_graph::GalleryItem::CNOT
            .build()
            .to_blog_text()
            .replace("module main", "module Gate");
        // A helper alone is not an executable BLOG document. The parser rejects
        // it before successful analysis or transport, even in definition view.
        let helper = source.as_str();
        super::super::analyze_live_blog(helper, 1)
            .result
            .err()
            .expect("missing main");
        let json = encode(super::super::analyze_live_blog(helper, 1)).unwrap();
        decode(&json)
            .unwrap()
            .result
            .err()
            .expect("transported parse error");
    }

    #[test]
    fn worker_branch_analysis_keeps_authored_snapshot_separate_from_projection() {
        let source = "BLOG 1.0\n8: Port [3,0,-3] role=input\n9: Z [3,0,-2]\n8 -> +Z\n11: Port [0,0,-2] role=input\n10: ZXZ [0,0,-1]\n11 -> +Z\n[0,0,-1] -> +Z\nbranch b {\n false {\n 1: Z [0,0,0]\n }\n true {\n 2: Z [0,0,0]\n }\n}\nm = measure 9\nresolve b if m\n";
        let json = encode(super::super::analyze_live_blog(source, 1)).unwrap();
        let wire: AnalysisWire = serde_json::from_str(&json).unwrap();
        let ResultWire::Changed(graph, source) = wire.result.unwrap() else {
            panic!("changed")
        };
        assert!(
            matches!(*graph, GraphWire::Snapshot(_))
                || matches!(source, Some(GraphWire::Snapshot(_)))
        );
        let super::super::LiveBlogResult::Changed(parsed) = decode(&json).unwrap().result.unwrap()
        else {
            panic!("changed")
        };
        assert_eq!(parsed.source_graph.branch_definitions().len(), 1);
    }

    #[test]
    fn worker_external_input_analysis_preserves_source() {
        let source = "BLOG 1.0\nmodule main {\n in enabled\n in q: data = 0\n out r: data = 2\n 0: Port [0,0,0] role=input\n 1: ZXZ [0,0,1]\n 2: Port [0,0,2] role=output\n 0 -> +Z\n 1 -> +Z\n feedback X 1 if enabled\n}\n";
        let json = encode(super::super::analyze_live_blog(source, 1)).unwrap();
        let super::super::LiveBlogResult::Changed(parsed) = decode(&json).unwrap().result.unwrap()
        else {
            panic!("changed")
        };
        assert_eq!(parsed.source_graph.interface.bit_inputs, ["enabled"]);
    }

    #[test]
    fn worker_unchanged_transport_omits_graphs() {
        let source = "BLOG 1.0\n0: ZXZ [0,0,0]\n";
        let analysis = super::super::analyze_live_blog_against(source, 9, Some(source));
        let json = encode(analysis).unwrap();
        assert!(json.contains("Unchanged"));
        assert!(!json.contains("Geometry"));
        assert!(!json.contains("Snapshot"));
        assert!(matches!(
            decode(&json).unwrap().result.unwrap(),
            super::super::LiveBlogResult::Unchanged
        ));
    }

    #[test]
    #[ignore = "manual branch snapshot reconstruction benchmark"]
    fn worker_snapshot_transport_timing() {
        use std::time::Instant;
        let source = "BLOG 1.0\n8: Port [3,0,-3] role=input\n9: Z [3,0,-2]\n8 -> +Z\n11: Port [0,0,-2] role=input\n10: ZXZ [0,0,-1]\n11 -> +Z\n[0,0,-1] -> +Z\nbranch b {\n false {\n 1: Z [0,0,0]\n }\n true {\n 2: Z [0,0,0]\n }\n}\nm = measure 9\nresolve b if m\n".to_owned()
            + &(0..3000).map(|i| format!("{}: ZXZ [{},0,0]\nm{i} = measure {}\n", i+20,i*2+20,i+20)).collect::<String>();
        let graph = BlockGraph::from_text(&source).unwrap();
        let canonical = graph.to_blog_text();
        let start = Instant::now();
        let old = BlockGraph::from_text(&canonical).unwrap();
        let old_time = start.elapsed();
        let wire = GraphWire::encode(&graph);
        assert!(matches!(wire, GraphWire::Snapshot(_)));
        let json = serde_json::to_string(&wire).unwrap();
        let start = Instant::now();
        let restored = serde_json::from_str::<GraphWire>(&json)
            .unwrap()
            .decode()
            .unwrap();
        let new_time = start.elapsed();
        assert_eq!(restored.to_blog_text(), old.to_blog_text());
        eprintln!(
            "branch + 3000 measured cubes: old parse={old_time:?}; snapshot JSON+restore={new_time:?}; bytes={}",
            json.len()
        );
    }

    #[test]
    #[ignore = "manual transport responsiveness benchmark"]
    fn worker_large_flat_transport_timing() {
        use std::time::Instant;
        let source = "BLOG 1.0\n".to_owned()
            + &(0..12000)
                .map(|i| format!("{i}: ZXZ [{},0,0]\n", i * 2))
                .collect::<String>();
        let start = Instant::now();
        let analysis = super::super::analyze_live_blog(&source, 7);
        let analyze = start.elapsed();
        let start = Instant::now();
        let json = encode(analysis).unwrap();
        let encode_time = start.elapsed();
        let wire: AnalysisWire = serde_json::from_str(&json).unwrap();
        let ResultWire::Changed(graph, original) = wire.result.unwrap() else {
            panic!("expected changed graph")
        };
        assert!(matches!(*graph, GraphWire::Geometry { .. }));
        eprintln!(
            "source graph uses geometry fast path: {}",
            original.is_none() || matches!(original, Some(GraphWire::Geometry { .. }))
        );
        let start = Instant::now();
        let super::super::LiveBlogResult::Changed(decoded) = decode(&json).unwrap().result.unwrap()
        else {
            panic!("expected changed graph")
        };
        let decode_time = start.elapsed();
        assert_eq!(decoded.graph.block_count(), 12000);
        eprintln!(
            "12000 blocks: worker analysis={analyze:?}; encode={encode_time:?}; UI JSON/graph reconstruction={decode_time:?}; payload={} bytes",
            json.len()
        );
    }

    #[test]
    fn worker_geometry_transport_preserves_height_tags_color_role_and_hadamard() {
        let mut graph = BlockGraph::default();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().unwrap())
                .unwrap()
                .with_tag("cube")
                .unwrap(),
        );
        graph.add_block(
            Block::new(2 * IVec3::Z, BlockKind::Port)
                .with_port_color([15, 30, 90])
                .unwrap()
                .with_port_role(PortRole::Output)
                .unwrap()
                .with_tag("out")
                .unwrap(),
        );
        graph.add_pipe(
            Pipe::new(IVec3::Z, Direction::ZPLUS)
                .with_hadamard()
                .with_tag("wire")
                .unwrap(),
        );
        assert!(matches!(
            GraphWire::encode(&graph),
            GraphWire::Geometry { .. }
        ));
        let decoded = roundtrip(&graph);
        assert_eq!(decoded.to_blog_body_text(), graph.to_blog_body_text());
        assert_eq!(decoded.get_block(IVec3::ZERO), graph.get_block(IVec3::ZERO));
        assert_eq!(
            decoded.get_block(2 * IVec3::Z),
            graph.get_block(2 * IVec3::Z)
        );
    }

    #[test]
    fn worker_geometry_transport_preserves_all_block_kinds() {
        for kind in BlockKind::all_kinds() {
            let mut graph = BlockGraph::default();
            graph.add_block(Block::new(IVec3::ZERO, kind));
            assert_eq!(
                roundtrip(&graph).get_block(IVec3::ZERO),
                graph.get_block(IVec3::ZERO),
                "{kind}"
            );
        }
    }

    #[test]
    fn worker_rich_transport_preserves_actions_and_module_metadata() {
        for graph in [
            bloq_graph::GalleryItem::CNOT.build(),
            bloq_graph::GalleryItem::T.build(),
        ] {
            assert!(matches!(
                GraphWire::encode(&graph),
                GraphWire::Document { .. }
            ));
            let decoded = roundtrip(&graph);
            assert_eq!(decoded.to_blog_text(), graph.to_blog_text());
            assert_eq!(decoded.actions(), graph.actions());
            assert_eq!(decoded.interface, graph.interface);
            assert_eq!(decoded.instances, graph.instances);
            assert_eq!(decoded.quantum_connections, graph.quantum_connections);
            assert_eq!(decoded.bit_bindings, graph.bit_bindings);
        }
    }

    #[test]
    fn worker_document_transport_preserves_all_action_and_expression_fields() {
        use bloq_graph::{Action, BinaryOp, Expr, FeedbackTarget, MeasureTarget, PauliBasis};
        let expr = Expr::Binary(
            BinaryOp::Or,
            Box::new(Expr::Not(Box::new(Expr::Var("a".into())))),
            Box::new(Expr::Binary(
                BinaryOp::And,
                Box::new(Expr::Var("b".into())),
                Box::new(Expr::Binary(
                    BinaryOp::Xor,
                    Box::new(Expr::Var("c".into())),
                    Box::new(Expr::Var("d".into())),
                )),
            )),
        );
        let actions = vec![
            Action::Let {
                name: "result".into(),
                expr: expr.clone(),
            },
            Action::Measure {
                name: "node".into(),
                target: MeasureTarget::Node(IVec3::new(3, 4, 5)),
            },
            Action::Measure {
                name: "edge".into(),
                target: MeasureTarget::Edge {
                    src: IVec3::new(-2, 3, 4),
                    dir: Direction::XMINUS,
                },
            },
            Action::DiscardIf(expr.clone()),
            Action::Resolve {
                target: IVec3::Z,
                condition: expr.clone(),
            },
            Action::Feedback {
                targets: vec![
                    FeedbackTarget {
                        pauli: PauliBasis::Y,
                        target: IVec3::X,
                        direction: Some(Direction::ZMINUS),
                    },
                    FeedbackTarget {
                        pauli: PauliBasis::Z,
                        target: IVec3::Y,
                        direction: None,
                    },
                ],
                condition: Some(expr),
            },
        ];
        for action in actions {
            let json = serde_json::to_string(&ActionWire::encode(&action)).unwrap();
            assert_eq!(
                serde_json::from_str::<ActionWire>(&json)
                    .unwrap()
                    .decode()
                    .unwrap(),
                action
            );
        }
    }

    #[test]
    fn worker_document_transport_preserves_classical_outputs_and_resource_ports() {
        use bloq_graph::{BitOutput, Expr};
        let mut graph = bloq_graph::GalleryItem::T.build();
        graph.interface.quantum_ports[0].resource_type = "injected_data".into();
        graph.interface.bit_outputs.push(BitOutput {
            name: "parity".into(),
            expr: Expr::Not(Box::new(Expr::Var("mzz".into()))),
        });
        let decoded = roundtrip(&graph);
        assert_eq!(decoded.to_blog_text(), graph.to_blog_text());
        assert_eq!(decoded.interface, graph.interface);
        assert_eq!(decoded.actions(), graph.actions());
    }

    #[test]
    fn worker_complex_graphs_use_checked_snapshots() {
        use bloq_graph::{Branch, BranchArm, Expr};
        let mut graph =
            BlockGraph::from_text("BLOG 1.0\n0: ZXZ [0,0,0]\n1: ZXZ [2,0,0]\nm = measure 1\n")
                .unwrap();
        let arm = BranchArm::try_from_blocks([([0, 0, 1], "Z")]).unwrap();
        graph
            .try_add_branches(
                [Branch::new("cap", Expr::Var("m".into()), arm.clone(), arm)],
                [Pipe::new([0, 0, 0], Direction::ZPLUS)],
                [],
            )
            .unwrap();
        assert!(matches!(GraphWire::encode(&graph), GraphWire::Snapshot(_)));
        graph.set_shown_branch_arm("cap", false).unwrap();
        let decoded = roundtrip(&graph);
        assert!(!decoded.source_snapshot().root.branches[0].shown_true);
        assert_eq!(decoded.branch_definitions(), graph.branch_definitions());
        assert_eq!(decoded.actions(), graph.actions());
        let mut external = BlockGraph::default();
        external
            .set_actions_with_inputs(
                vec![bloq_graph::Action::DiscardIf(Expr::Var("enabled".into()))],
                ["enabled".into()],
            )
            .unwrap();
        assert!(matches!(
            GraphWire::encode(&external),
            GraphWire::Snapshot(_)
        ));
        let restored = roundtrip(&external);
        assert_eq!(restored.actions(), external.actions());
        assert_eq!(
            restored.source_snapshot().root.action_inputs,
            external.source_snapshot().root.action_inputs
        );
    }

    #[test]
    #[ignore = "manual complex-document transport benchmark"]
    fn worker_action_document_transport_timing() {
        use std::time::Instant;
        let source = "BLOG 1.0\nmodule main {\n".to_owned()
            + &(0..3000)
                .map(|i| format!("{i}: ZXZ [{},0,0]\nm{i} = measure {i}\n", i * 2))
                .collect::<String>()
            + "}\n";
        let graph = BlockGraph::from_text(&source).unwrap();
        let canonical = graph.to_blog_text();
        let start = Instant::now();
        let old = BlockGraph::from_text(&canonical).unwrap();
        let old_time = start.elapsed();
        let wire = GraphWire::encode(&graph);
        assert!(matches!(wire, GraphWire::Document { .. }));
        let json = serde_json::to_string(&wire).unwrap();
        let start = Instant::now();
        let decoded = serde_json::from_str::<GraphWire>(&json)
            .unwrap()
            .decode()
            .unwrap();
        let new_time = start.elapsed();
        assert_eq!(decoded.to_blog_text(), old.to_blog_text());
        assert_eq!(decoded.actions(), old.actions());
        eprintln!(
            "3000 measured cubes + main interface: canonical BLOG parse/lower={old_time:?}; Document JSON/reconstruct={new_time:?}; payload={} bytes; source={} bytes",
            json.len(),
            canonical.len()
        );
    }

    #[test]
    fn worker_graph_transport_does_not_drop_explicit_interfaces() {
        let graph = BlockGraph::definition(
            "main",
            BlockGraph::default(),
            ModuleInterface::default(),
            vec![],
            vec![],
            vec![],
        );
        assert!(matches!(
            GraphWire::encode(&graph),
            GraphWire::Document { .. }
        ));
        assert!(roundtrip(&graph).has_module_structure());
    }
}
