//! Compile coverage for supported gallery entries under the fixed-bulk
//! convention. Analysis-only entries have rejection coverage in module and
//! physical gallery tests. Open temporal/directed spatial ports compile directly.

use bloq_circuit::Pauli;
use bloq_graph::{
    Action, Block, BlockGraph, BlockKind, CubeKind, Direction, GalleryCategory, GalleryItem,
    MeasureTarget, Pipe, PortRole,
};
use bloq_ir::{BoundaryFace, ClassicalNode, InstanceProvenance, NodeProvenance, SpatialPortPart};
use bloq_vm::run_bloq;

#[test]
fn every_supported_gallery_entry_compiles() {
    for gallery in
        GalleryItem::iter().filter(|gallery| !gallery.in_category(GalleryCategory::AnalysisOnly))
    {
        let program = gallery.build();
        bloq_compile::CompileContext::new(bloq_compile::CompileConfig::default())
            .compile(&program)
            .unwrap_or_else(|error| panic!("{gallery:?}: module compile failed: {error:?}"));
    }
}

#[test]
fn spatial_ports_lower_to_one_cube_and_one_virtual_temporal_port() {
    for gallery in [
        GalleryItem::CZSpatialH,
        GalleryItem::ThreeCNOTs,
        GalleryItem::SteaneEncoding,
        GalleryItem::And4T,
    ] {
        let graph = gallery.build();
        let bloq = bloq_compile::compile(&graph, 3)
            .unwrap_or_else(|error| panic!("{gallery:?}: direct compile failed: {error}"));
        let min_z = graph
            .positions()
            .map(|position| position.z)
            .min()
            .unwrap_or(0);

        for port in graph.blocks().filter(|block| {
            block.kind() == bloq_graph::BlockKind::Port
                && graph.pipes().any(|pipe| {
                    let (left, right) = pipe.endpoints();
                    pipe.dir().is_spatial() && (left == block.pos() || right == block.pos())
                })
        }) {
            let source = port.pos() - glam::ivec3(0, 0, min_z);
            let role = port.port_role().expect("Port has a role");
            let mut parts = bloq
                .quantum_nodes()
                .flat_map(|(_, node)| &node.instances)
                .filter_map(|instance| match instance.provenance {
                    InstanceProvenance::SpatialPortSubstitution {
                        source: actual,
                        role: actual_role,
                        part,
                    } if actual == source => {
                        assert_eq!(actual_role, role);
                        Some(part)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            parts.sort_unstable();
            assert_eq!(
                parts,
                [SpatialPortPart::Cube, SpatialPortPart::TemporalPort],
                "{gallery:?} spatial Port at {source}"
            );
            assert_eq!(
                bloq.nodes()
                    .filter(|(_, node)| matches!(
                        node.provenance,
                        NodeProvenance::SpatialPortSubstitution {
                            source: actual,
                            ..
                        } if actual == source
                    ))
                    .count(),
                1,
                "{gallery:?} spatial Port at {source} has one positionless Port node"
            );
        }
    }
}

#[test]
fn virtual_temporal_port_does_not_claim_the_next_lattice_cell() {
    let mut graph = BlockGraph::new();
    graph.add_block(
        Block::new(glam::IVec3::ZERO, BlockKind::Port)
            .with_port_role(PortRole::Output)
            .unwrap(),
    );
    graph.add_block(Block::new(glam::IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
    graph.add_pipe(Pipe::new(glam::IVec3::ZERO, Direction::XPLUS));
    graph.add_block(Block::new(glam::IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));

    let bloq = bloq_compile::compile(&graph, 3).expect("occupied cell above is unrelated");

    assert!(bloq.nodes().any(|(_, node)| matches!(
        node.provenance,
        NodeProvenance::SpatialPortSubstitution {
            source: glam::IVec3::ZERO,
            role: PortRole::Output,
        }
    )));
}

#[test]
fn multiplex_port_lowers_to_a_corrected_physical_output() {
    let mut graph = BlockGraph::new();
    let second = glam::ivec3(0, 4, 0);
    for (port, kind) in [(glam::IVec3::ZERO, CubeKind::ZXZ), (second, CubeKind::XZX)] {
        graph.add_block(
            Block::new(port, BlockKind::Port)
                .with_port_role(PortRole::Multiplex)
                .unwrap(),
        );
        graph.add_block(Block::new(port + glam::IVec3::X, BlockKind::Cube(kind)));
        graph.add_pipe(Pipe::new(port, Direction::XPLUS));
    }
    graph
        .set_actions(vec![Action::Measure {
            target: MeasureTarget::Node(glam::IVec3::X),
            name: "z_at_multiplex".into(),
        }])
        .unwrap();
    let stabilizers = graph.stabilizers().unwrap();
    let measurement = stabilizers
        .iter()
        .find(|generator| generator.measurement_name() == Some("z_at_multiplex"))
        .unwrap();
    assert_eq!(
        measurement
            .stabilizer
            .port_stabilizer
            .get(&glam::IVec3::ZERO),
        Some(&Pauli::Z)
    );

    let bloq = bloq_compile::compile(&graph, 3).expect("Multiplex Z measurement compiles");
    assert_eq!(bloq.logical_outputs().len(), 2);
    assert_eq!(bloq.output_frames().len(), 2);
    let output_qubit = |port| {
        let output = bloq
            .logical_outputs()
            .iter()
            .find(|output| output.port == port)
            .expect("q_out is exported");
        let (&qubit, &pauli) = output.x.iter().next().unwrap();
        assert_eq!((output.x.len(), pauli), (1, Pauli::X));
        assert_eq!(output.z.len(), 1);
        assert_eq!(output.z.get(&qubit), Some(&Pauli::Z));
        qubit
    };
    let x_qubit = output_qubit(glam::IVec3::ZERO);
    let second_qubit = output_qubit(second);
    assert_ne!(x_qubit, second_qubit);
    let second_virtual = bloq
        .quantum_nodes()
        .flat_map(|(_, node)| &node.instances)
        .find(|instance| {
            matches!(
                instance.provenance,
                InstanceProvenance::SpatialPortSubstitution {
                    source,
                    role: PortRole::Multiplex,
                    part: SpatialPortPart::TemporalPort,
                } if source == second
            )
        })
        .expect("virtual temporal input exists");
    let mut split_support = [false; 2];
    for (_, node) in bloq.nodes() {
        let Some(ClassicalNode::Observable { operators, .. }) = node.try_classical() else {
            continue;
        };
        for operator in operators
            .iter()
            .filter(|operator| operator.instance == second_virtual.id)
        {
            split_support[0] |= operator.face == BoundaryFace::Input
                && !operator.operator.is_empty()
                && operator
                    .operator
                    .iter()
                    .all(|(_, pauli)| *pauli == Pauli::X);
            split_support[1] |= operator.face == BoundaryFace::Output
                && operator.operator.len() == 1
                && operator.operator.get(&second_qubit) == Some(&Pauli::X);
        }
    }
    assert_eq!(split_support, [true, true], "X lowers to X_L X_out");

    let stim = bloq_stim::emit_bloq_stim(&bloq).expect("Multiplex Port emits Stim");
    assert!(stim.contains("RX "));
    assert!(stim.contains("CX rec[-1]"));
    assert_eq!(run_bloq(&bloq, 8, 0x5A).unwrap().discarded, 0);

    let error = graph
        .set_actions(vec![Action::Measure {
            target: MeasureTarget::Node(second + glam::IVec3::X),
            name: "x_at_multiplex".into(),
        }])
        .unwrap_err();
    assert!(
        matches!(
            error,
            bloq_graph::BlockGraphError::Stabilizer(
                bloq_graph::StabilizerError::UnavailableControlParity { .. }
            )
        ),
        "{error:?}"
    );
}

#[test]
fn spatial_port_matches_its_manual_cube_and_temporal_port_expansion() {
    let mut spatial = BlockGraph::new();
    spatial.add_block(
        Block::new(glam::IVec3::ZERO, BlockKind::Port)
            .with_port_role(PortRole::Input)
            .unwrap(),
    );
    spatial.add_block(Block::new(glam::IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
    spatial.add_pipe(Pipe::new(glam::IVec3::ZERO, Direction::XPLUS));

    let mut manual = spatial.clone();
    let inferred = manual
        .infer_spatial_port_cube_kind(glam::IVec3::ZERO)
        .expect("spatial Port has one substitution cube kind");
    manual
        .set_block_kind(glam::IVec3::ZERO, BlockKind::Cube(inferred))
        .unwrap();
    manual.add_block(Block::new(-glam::IVec3::Z, BlockKind::Port));
    manual.add_pipe(Pipe::new(-glam::IVec3::Z, Direction::ZPLUS));

    let emit = |graph: &BlockGraph| {
        let bloq = bloq_compile::compile(graph, 3).expect("equivalent graph compiles");
        bloq_stim::emit_bloq_stim(&bloq)
            .expect("equivalent graph emits Stim")
            .parse::<stim::Circuit>()
            .expect("emitted Stim parses")
    };
    assert_eq!(emit(&spatial), emit(&manual));
}

#[test]
fn dynamic_graph_accepts_spatial_port() {
    let bloq = bloq_compile::compile(&GalleryItem::CCZInjectedAnd.build(), 3)
        .expect("dynamic spatial Port compiles");
    assert!(
        bloq.quantum_nodes()
            .flat_map(|(_, node)| &node.instances)
            .any(|instance| matches!(
                instance.provenance,
                InstanceProvenance::SpatialPortSubstitution {
                    source: glam::IVec3 { x: -1, y: 0, z: 0 },
                    role: PortRole::Input,
                    part: SpatialPortPart::TemporalPort,
                }
            ))
    );
}
