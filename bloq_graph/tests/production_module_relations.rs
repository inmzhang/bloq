#![cfg(test)]
//! Production planning preserves complete supports while reusing definition relations.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::num::NonZeroUsize;

use bloq_graph::{
    BitBinding, BitRef, Block, BlockGraph, BlockKind, GalleryItem, GuardedLocalSurface,
    GuardedReadoutPlan, GuardedSurfaceKind, GuardedSurfaceSpace, GuardedTopology, GuardedVariable,
    InstancePort, ModuleCertificationLimits, ModuleInstance, ModuleInterface, ModuleRotation,
    PortDirection, PortRole, QuantumConnection, SurfaceSupport, UDirection, ZXGraph,
    flatten_module_definition,
};
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, DECISION_FALSE, DECISION_TRUE, DecisionId,
};
use glam::IVec3;

fn import(source: &BooleanDecisionDiagram, target: &mut BooleanDecisionDiagram) -> Vec<DecisionId> {
    let mut ids = vec![DECISION_FALSE, DECISION_TRUE];
    for node in source.nodes() {
        ids.push(
            target
                .make_node(node.variable, ids[node.low.0], ids[node.high.0])
                .expect("diagnostic decision import is unbounded"),
        );
    }
    ids
}

fn local_supports(plan: &mut GuardedReadoutPlan) -> Vec<(usize, IVec3, Vec<GuardedLocalSurface>)> {
    let mut result = Vec::new();
    for surface in 0..plan.surfaces.len() {
        for position in plan.support_sites(surface) {
            let position = IVec3::from_array(position);
            result.push((
                surface,
                position,
                plan.local_cases(surface, position, DECISION_TRUE)
                    .expect("fixture surface support has bounded local cases"),
            ));
        }
    }
    result
}

fn assert_same_plan(actual: &mut GuardedReadoutPlan, expected: &mut GuardedReadoutPlan) {
    let variables = |plan: &GuardedReadoutPlan| {
        plan.topology
            .variables
            .iter()
            .map(|variable| match variable {
                GuardedVariable::Branch { name, .. } => ("branch", name.clone()),
                GuardedVariable::Outcome(name) => ("outcome", name.clone()),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(variables(actual), variables(expected));
    let actual_support = local_supports(actual);
    let expected_support = local_supports(expected);
    let mut canonical = BooleanDecisionDiagram::default();
    let actual_ids = import(&actual.topology.diagram, &mut canonical);
    let expected_ids = import(&expected.topology.diagram, &mut canonical);
    let domain = actual_ids[actual.topology.domain.0];
    assert_eq!(domain, expected_ids[expected.topology.domain.0]);
    let mut mapped = |root: DecisionId, ids: &[DecisionId]| {
        canonical
            .apply(BooleanOp::And, domain, ids[root.0])
            .expect("diagnostic guard comparison is unbounded")
    };
    let kind = |kind: &GuardedSurfaceKind,
                ids: &[DecisionId],
                mapped: &mut dyn FnMut(DecisionId, &[DecisionId]) -> DecisionId| {
        let mut kind = kind.clone();
        match &mut kind {
            GuardedSurfaceKind::Readout { folds, .. } => {
                for (_, root) in folds {
                    *root = mapped(*root, ids);
                }
            }
            GuardedSurfaceKind::OutputFrame { folds, .. } => {
                for (_, _, root) in folds {
                    *root = mapped(*root, ids);
                }
            }
            GuardedSurfaceKind::LogicalReadout => {}
        }
        kind
    };
    assert_eq!(actual.surfaces.len(), expected.surfaces.len());
    for (actual, expected) in actual.surfaces.iter().zip(&expected.surfaces) {
        assert_eq!(
            kind(&actual.kind, &actual_ids, &mut mapped),
            kind(&expected.kind, &expected_ids, &mut mapped),
        );
        assert_eq!(
            mapped(actual.activation(), &actual_ids),
            mapped(expected.activation(), &expected_ids)
        );
        let actual_feedbacks = actual
            .feedbacks
            .iter()
            .map(|&(ordinal, root)| (ordinal, mapped(root, &actual_ids)))
            .collect::<Vec<_>>();
        let expected_feedbacks = expected
            .feedbacks
            .iter()
            .map(|&(ordinal, root)| (ordinal, mapped(root, &expected_ids)))
            .collect::<Vec<_>>();
        assert_eq!(actual_feedbacks, expected_feedbacks);
    }
    assert_eq!(actual_support.len(), expected_support.len());
    for (
        (actual_surface, actual_position, actual_cases),
        (expected_surface, expected_position, expected_cases),
    ) in actual_support.into_iter().zip(expected_support)
    {
        assert_eq!(
            (actual_surface, actual_position),
            (expected_surface, expected_position)
        );
        let mut signature = |cases: Vec<GuardedLocalSurface>, ids: &[DecisionId]| {
            cases
                .into_iter()
                .map(|case| {
                    (
                        mapped(case.guard, ids),
                        case.surface.node_pauli(actual_position),
                        case.surface.port_pauli(actual_position),
                        case.surface.edges().collect::<Vec<_>>(),
                        case.selective,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            signature(actual_cases, &actual_ids),
            signature(expected_cases, &expected_ids)
        );
    }
}

fn compare(program: &BlockGraph) -> GuardedReadoutPlan {
    let limits = ModuleCertificationLimits::DEFAULT;
    let relations = program
        .prepare_module_relations_with_cache(limits, NonZeroUsize::MIN, |_| None)
        .expect("fixture definition relations are preparable");
    let linked = flatten_module_definition(program, program.root(), "")
        .expect("fixture module instances can be linked");
    let graph = linked.graph.fix_shadowed_faces();
    let topology = GuardedTopology::new(&graph, limits)
        .expect("fixture linked geometry has a guarded topology");
    let mut expected = GuardedSurfaceSpace::new(topology.clone(), &linked.sites, limits)
        .expect("uncached fixture relation can be constructed")
        .plan_readouts()
        .expect("uncached fixture readouts are causal");
    let mut actual = relations
        .compose(topology, &linked.sites)
        .expect("prepared fixture relation can be composed")
        .plan_readouts()
        .expect("prepared fixture readouts are causal");
    assert_same_plan(&mut actual, &mut expected);
    actual
}

#[test]
fn reusable_relations_preserve_rotated_adaptive_and_continuing_branch_supports() {
    for program in [
        GalleryItem::PhaseGradientK4.build(),
        GalleryItem::ThreeBitAdder.build(),
        repeated_continuing_strips(true),
        repeated_continuing_strips(false),
    ] {
        compare(&program);
    }
}

fn repeated_continuing_strips(shared_inputs: bool) -> BlockGraph {
    let mut child = BlockGraph::from_text(include_str!(
        "../../docs/fixtures/conditional_cz_strip.blog"
    ))
    .expect("continuing CZ strip fixture is valid")
    .clone_local_definition();
    child.name = "Strip".into();
    let mut body = BlockGraph::new();
    let mut interface = ModuleInterface {
        bit_inputs: child.interface.bit_inputs.clone(),
        ..ModuleInterface::default()
    };
    let mut instances = Vec::new();
    let mut connections = Vec::new();
    let mut bindings = Vec::new();
    for index in 0..2 {
        let instance = ModuleInstance {
            name: format!("strip{index}"),
            definition: child.name.clone(),
            translation: IVec3::new(6 * index, 0, 0),
            rotation: ModuleRotation::new(UDirection::Z, index),
        };
        for port in &child.interface.quantum_ports {
            let position = instance
                .try_transform_position(port.position)
                .expect("fixture port transforms fit lattice coordinates");
            let endpoint = InstancePort {
                instance: instance.name.clone(),
                port: port.name.clone(),
            };
            let role = match port.direction {
                PortDirection::Input => {
                    connections.push(QuantumConnection::Input {
                        block: position,
                        input: endpoint,
                        hadamard: false,
                    });
                    PortRole::Input
                }
                PortDirection::Output => {
                    connections.push(QuantumConnection::Output {
                        output: endpoint,
                        block: position,
                        hadamard: index == 1 && port.name == "q_out",
                    });
                    PortRole::Output
                }
            };
            body.try_add_block(
                Block::new(position, BlockKind::Port)
                    .with_port_role(role)
                    .expect("fixture boundary block accepts its declared role"),
            )
            .expect("fixture ports occupy distinct positions");
            let mut root_port = port.clone();
            root_port.position = position;
            root_port.name = format!("strip{index}_{}", port.name);
            interface.quantum_ports.push(root_port);
        }
        for input in &child.interface.bit_inputs {
            let root_input = if index == 1 && !shared_inputs {
                let name = format!("second_{input}");
                interface.bit_inputs.push(name.clone());
                name
            } else {
                input.clone()
            };
            bindings.push(BitBinding {
                source: BitRef {
                    instance: None,
                    bit: root_input,
                },
                target_instance: instance.name.clone(),
                target_bit: input.clone(),
            });
        }
        instances.push(instance);
    }
    let root = BlockGraph::definition("main", body, interface, instances, connections, bindings);
    BlockGraph::from_definitions(vec![child, root])
        .expect("repeated continuing strips form a valid module program")
}

#[test]
fn nested_instances_keep_shared_inputs_and_independent_outcomes() {
    let source = r#"BLOG 1.0
module SelectCap {
  in q: data = 0
  in basis
  0: Port [0,0,0] role=input
  1: XZX [0,0,1]
  2: XZ [0,0,2]
  0 -> +Z
  1 -> +Z
  resolve 2 if basis
}
module Wrapped {
  in q: data = 0
  in basis
  0: Port [0,0,0] role=input
  cap: SelectCap @ [0,0,0]
  0 -> cap.q
  basis => cap.basis
}
module Memory {
  0: Y [0,0,0]
  1: XZX [0,0,1]
  2: Y [0,0,2]
  0 -> +Z
  1 -> +Z
}
module MemoryPair {
  first: Memory @ [0,0,0]
  second: Memory @ [3,0,0] rotate Z 180
}
module main {
  in a: data = 0
  in b: data = 1
  in enable0
  in enable1
  0: Port [0,0,0] role=input
  1: Port [3,0,0] role=input
  first: Wrapped @ [0,0,0]
  second: Wrapped @ [3,0,0] rotate Z 90
  memory0: MemoryPair @ [6,0,0]
  memory1: MemoryPair @ [12,0,0] rotate X 180
  0 -> first.q
  1 -> second.q
  enable0 => first.basis
  enable1 => second.basis
}
"#;
    for source in [
        source.to_owned(),
        source.replace("enable1 => second.basis", "enable0 => second.basis"),
    ] {
        let program = BlockGraph::from_text(&source).unwrap();
        let plan = compare(&program);
        for (x, z) in [(6, 1), (9, 1), (12, -1), (15, -1)] {
            assert!(
                plan.surfaces.iter().enumerate().any(|(index, surface)| {
                    matches!(surface.kind, GuardedSurfaceKind::LogicalReadout)
                        && plan.support_sites(index).contains(&[x, 0, z])
                }),
                "closed memory at x={x}, z={z} must retain its interior support"
            );
        }
        let zx = ZXGraph::try_from(&plan.topology.source).unwrap();
        for mask in 0..4 {
            let outcomes = BTreeMap::from([("enable0", mask & 1 != 0), ("enable1", mask & 2 != 0)]);
            for surface in 0..plan.surfaces.len() {
                let _ = plan.materialize_surface(surface, &zx, |name| outcomes[name]);
            }
        }
    }
}

fn repeated_wire(count: usize) -> BlockGraph {
    let mut instances = String::new();
    let mut seams = String::from("0 -> s0.input\n");
    for index in 0..count {
        writeln!(instances, "s{index}: Wire @ [0,0,{}]", 3 * index)
            .expect("writing fixture text to a String cannot fail");
        if index + 1 < count {
            writeln!(seams, "s{index}.output -> s{}.input", index + 1)
                .expect("writing fixture text to a String cannot fail");
        }
    }
    writeln!(seams, "s{}.output -> 1", count - 1)
        .expect("writing fixture text to a String cannot fail");
    BlockGraph::from_text(&format!(
        r#"BLOG 1.0
module Wire {{
  in input: data = 0
  out output: data = 4
  0: Port [0,0,0] role=input
  1: ZXZ [0,0,1]
  2: ZXZ [0,0,2]
  3: ZXZ [0,0,3]
  4: Port [0,0,4] role=output
  0 -> +Z
  1 -> +Z
  2 -> +Z
  3 -> +Z
}}
module main {{
  in input: data = 0
  out output: data = 1
  0: Port [0,0,0] role=input
  1: Port [0,0,{}] role=output
  {instances}
  {seams}
}}
"#,
        3 * count + 1
    ))
    .expect("repeated wire fixture is a valid module program")
}

#[test]
fn warmed_relations_reuse_internal_elimination_after_translation() {
    let limits = ModuleCertificationLimits::DEFAULT;
    let program = repeated_wire(6);
    let relations = program
        .prepare_module_relations_with_cache(limits, NonZeroUsize::new(1).unwrap(), |_| None)
        .unwrap();
    let linked = flatten_module_definition(&program, program.root(), "").unwrap();
    let graph = linked.graph.fix_shadowed_faces();
    relations
        .compose(GuardedTopology::new(&graph, limits).unwrap(), &linked.sites)
        .unwrap();
    let before = relations.relation_work();
    assert!(before.0 > 0, "the fixture must close internal seams");
    assert!(
        before.1 > 0,
        "repeated instances must reuse internal elimination"
    );
    let offset = IVec3::new(7, -3, 5);
    let shifted = graph.shift_positions(offset).unwrap();
    let sites = linked
        .sites
        .into_iter()
        .map(|(position, site)| (position + offset, site))
        .collect();
    let topology = GuardedTopology::new(&shifted, limits).unwrap();
    let mut actual = relations
        .compose(topology.clone(), &sites)
        .unwrap()
        .plan_readouts()
        .unwrap();
    let after = relations.relation_work();
    assert_eq!(
        after.0, before.0,
        "translation must not eliminate seams again"
    );
    assert!(
        after.1 > before.1,
        "the second composition must reuse the relation"
    );
    let mut expected = GuardedSurfaceSpace::new(topology, &sites, limits)
        .unwrap()
        .plan_readouts()
        .unwrap();
    assert_same_plan(&mut actual, &mut expected);
}

#[test]
fn public_summary_and_prepared_relation_plan_the_same_complete_supports() {
    let limits = ModuleCertificationLimits::DEFAULT;
    let program = repeated_wire(2);
    let (relations, summary) = program
        .prepare_module_summary_with_cache(limits, NonZeroUsize::new(1).unwrap(), |_| None)
        .unwrap();
    let linked = flatten_module_definition(&program, program.root(), "").unwrap();
    let graph = linked.graph.fix_shadowed_faces();
    let topology = GuardedTopology::new(&graph, limits).unwrap();
    let mut actual = summary
        .compose(topology.clone(), &linked.sites)
        .unwrap()
        .plan_readouts()
        .unwrap();
    let mut expected = relations
        .compose(topology, &linked.sites)
        .unwrap()
        .plan_readouts()
        .unwrap();
    assert_same_plan(&mut actual, &mut expected);
    assert_eq!(summary.relation_work(), relations.relation_work());
    let zx = ZXGraph::try_from(&graph).unwrap();
    let stabilizers = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
    for (index, _) in actual.surfaces.iter().enumerate() {
        let surface = actual
            .materialize_surface(index, &zx, |_| unreachable!())
            .unwrap();
        assert!(
            stabilizers
                .generators
                .iter()
                .any(|generator| generator.stabilizer == surface)
        );
    }
}
