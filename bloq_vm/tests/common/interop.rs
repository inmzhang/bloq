//! Pinned upstream layout fixtures checked against their exported source QASM.

#![allow(dead_code, reason = "shared by independent integration-test binaries")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bloq_graph::{
    Action, BlockGraph,
    verify::{BoundaryOrder, BranchStatus, LogicalVerifier, QuizxGraph},
};
use bloq_vm::{EnginePauli, EnginePauliString, Simulator};
use glam::IVec3;
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize)]
struct Interface {
    input_ports: Vec<[i32; 3]>,
    output_ports: Vec<[i32; 3]>,
    t_gadgets: usize,
}

pub(crate) struct Fixture<M> {
    pub(crate) graph: BlockGraph,
    pub(crate) qasm: String,
    pub(crate) inputs: Vec<IVec3>,
    pub(crate) outputs: Vec<IVec3>,
    pub(crate) t_gadgets: usize,
    pub(crate) metadata: M,
}

/// Loads the `.blog`, `.qasm`, and `.json` triple at `tests/data/{path}`.
pub(crate) fn fixture<M: DeserializeOwned>(path: &str) -> Fixture<M> {
    let read = |extension: &str| {
        let file =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{path}.{extension}"));
        std::fs::read_to_string(&file).unwrap_or_else(|error| panic!("{}: {error}", file.display()))
    };
    let json = read("json");
    let interface: Interface = serde_json::from_str(&json).expect("exported interface parses");
    Fixture {
        graph: BlockGraph::from_blog_text(&read("blog")).expect("exported BLOG parses"),
        qasm: read("qasm"),
        inputs: interface.input_ports.into_iter().map(IVec3::from).collect(),
        outputs: interface
            .output_ports
            .into_iter()
            .map(IVec3::from)
            .collect(),
        t_gadgets: interface.t_gadgets,
        metadata: serde_json::from_str(&json).expect("exported metadata parses"),
    }
}

impl<M> Fixture<M> {
    fn boundaries(&self) -> BoundaryOrder {
        BoundaryOrder::new(self.inputs.clone(), self.outputs.clone())
    }

    /// QuiZX parses the source independently of every upstream frontend.
    pub(crate) fn expected(&self) -> QuizxGraph {
        super::parse_qasm(&self.qasm)
            .expect("source QASM has an independent QuiZX map")
            .to_graph()
    }

    pub(crate) fn verifier(&self) -> LogicalVerifier {
        LogicalVerifier::with_boundaries_and_internal_measurements(&self.graph, self.boundaries())
            .expect("exported interface admits a logical verifier")
    }

    /// Checks sampled imported branches against the source map and returns it.
    pub(crate) fn verify_source(&self) -> QuizxGraph {
        let expected = self.expected();
        let report = self
            .verifier()
            .verify(&expected, 32, 7)
            .expect("every imported branch has the source logical map");
        assert_eq!(report.verified, 32, "sampled logical branches");
        expected
    }

    /// Checks both values of a single-gadget fixture's named readout.
    pub(crate) fn verify_named_outcomes(&self, expected: &QuizxGraph) {
        let verifier = LogicalVerifier::with_boundaries(&self.graph, self.boundaries())
            .expect("single T fixture admits explicit branch verification");
        let measurement = self
            .graph
            .actions()
            .into_iter()
            .find_map(|action| match action {
                Action::Measure { name, .. } => Some(name),
                _ => None,
            })
            .expect("single T fixture declares its readout");
        for value in [false, true] {
            let branch = verifier
                .verify_branch(expected, &BTreeMap::from([(measurement.clone(), value)]))
                .expect("each named T outcome preserves the signed channel");
            assert_eq!(branch.status(), BranchStatus::Verified);
        }
    }
}

/// Applies a static Clifford+T fixture source as the ideal physical channel.
pub(crate) fn apply_source_qasm(sim: &mut Simulator, qasm: &str) {
    for gate in super::parse_qasm(qasm)
        .expect("fixture source QASM parses")
        .gates
    {
        use quizx::gate::GType;
        match gate.t {
            GType::HAD => sim.h(gate.qs[0]),
            GType::NOT => sim.x(gate.qs[0]),
            GType::Z => sim.z(gate.qs[0]),
            GType::S => sim.s(gate.qs[0]),
            GType::Sdg => sim.s_dag(gate.qs[0]),
            GType::T => sim.t(gate.qs[0]).expect("source T uses an allocated qubit"),
            GType::Tdg => sim
                .t_dag(gate.qs[0])
                .expect("source T dagger uses an allocated qubit"),
            GType::ZPhase => sim
                .pauli_rotation(
                    &EnginePauliString::single(sim.num_qubits(), gate.qs[0], EnginePauli::Z),
                    gate.phase.to_f64() * std::f64::consts::FRAC_PI_2,
                )
                .expect("source phase rotation uses an allocated qubit"),
            GType::CNOT => sim
                .cx(gate.qs[0], gate.qs[1])
                .expect("source CNOT uses allocated qubits"),
            GType::CZ => sim
                .cz(gate.qs[0], gate.qs[1])
                .expect("source CZ uses allocated qubits"),
            other => panic!("unexpected synthesized source gate {other:?}"),
        }
    }
}

pub(crate) fn record_selectors(
    selectors: &mut BTreeMap<String, BTreeSet<bool>>,
    named: &[(&str, bool)],
) {
    for &(name, value) in named {
        selectors.entry(name.to_owned()).or_default().insert(value);
    }
}

/// Asserts that `count` named selectors each took both values.
pub(crate) fn assert_selectors_toggle(
    selectors: &BTreeMap<String, BTreeSet<bool>>,
    count: usize,
    label: &str,
) {
    assert_eq!(selectors.len(), count, "{label}: T selectors");
    for (selector, values) in selectors {
        assert_eq!(
            *values,
            BTreeSet::from([false, true]),
            "{label}: selector {selector}"
        );
    }
}
