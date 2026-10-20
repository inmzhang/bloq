//! Exact normalized Choi states for small unitary maps and state preparations.
//! The ideal circuit acts on output qubits 0..n; reference halves follow them.
//! Every shot must preserve the corrected channel without discarding Bell outcomes.

#![allow(
    dead_code,
    reason = "shared helper used by independent integration-test binaries"
)]

use std::cell::RefCell;

use bloq_ir::lowering::InstanceMeasurement;
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, ClassicalNode, InstanceProvenance, NodeProvenance,
    PortRole,
};
use bloq_vm::verify::{VerifyReport, logical_signature, signatures_match};
use bloq_vm::{
    ExecError, OutputLogical, PreparationContext, ShotContext, Simulator, run_bloq_with_io,
};
use glam::IVec3;

use super::stabilizer::physical_qubit;

pub(crate) struct ExactChoi {
    inputs: Vec<IVec3>,
    outputs: Vec<IVec3>,
    expected: Vec<f64>,
}

impl ExactChoi {
    pub(crate) fn new(
        inputs: &[IVec3],
        outputs: &[IVec3],
        ideal: impl FnOnce(&mut Simulator) -> Result<(), ExecError>,
    ) -> Self {
        assert!(
            !outputs.is_empty(),
            "an exact Choi check needs live outputs"
        );
        assert!(
            inputs.len() <= outputs.len(),
            "ideal inputs precede fresh output ancillas"
        );
        let width = inputs.len() + outputs.len();
        assert!(
            width <= 6,
            "use the stabilizer oracle for larger interfaces"
        );
        let mut sim = Simulator::with_seed(width, 0);
        for input in 0..inputs.len() {
            sim.h(input);
            sim.cx(input, outputs.len() + input)
                .expect("Bell pair uses distinct allocated qubits");
        }
        ideal(&mut sim).expect("independent ideal circuit");
        assert_eq!(sim.num_qubits(), width, "ideal circuit keeps its interface");
        let logicals: Vec<_> = (0..width).map(|q| physical_qubit(q, width)).collect();
        let expected = logical_signature(
            &sim,
            &logicals
                .iter()
                .map(|out| (out, (false, false)))
                .collect::<Vec<_>>(),
        )
        .expect("six-qubit Choi signature fits the index width");
        Self {
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            expected,
        }
    }

    fn matches(
        &self,
        sim: &Simulator,
        outputs: &[(&OutputLogical, (bool, bool))],
    ) -> Result<bool, ExecError> {
        Ok(signatures_match(
            &logical_signature(sim, outputs)?,
            &self.expected,
        ))
    }

    /// Bell-pair every declared input and compare the corrected output/reference
    /// state. The extra hook retains fixture-specific branch/orbit regressions.
    pub(crate) fn run(
        &self,
        bloq: &Bloq,
        shots: usize,
        seed: u64,
        inspect: impl FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
    ) -> VerifyReport {
        self.run_prepared(bloq, shots, seed, &[], |_, _| Ok(()), inspect)
    }

    /// Fixed resource inputs are prepared separately; all remaining inputs are
    /// Bell-paired. The declared port sets must cover the complete VM interface.
    pub(crate) fn run_prepared(
        &self,
        bloq: &Bloq,
        shots: usize,
        seed: u64,
        prepared_ports: &[IVec3],
        prepare: impl FnMut(&mut Simulator, &PreparationContext<'_>) -> Result<(), ExecError>,
        inspect: impl FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
    ) -> VerifyReport {
        self.run_prepared_selected(
            bloq,
            shots,
            seed,
            (prepared_ports, prepare),
            |_| self,
            inspect,
        )
    }

    /// Compare a normalized source branch against its explicitly chosen ideal.
    /// Every ideal must describe the same ordered input/output interface.
    pub(crate) fn run_selected<'a>(
        &'a self,
        bloq: &Bloq,
        shots: usize,
        seed: u64,
        select: impl FnMut(&ShotContext<'_>) -> &'a Self,
        inspect: impl FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
    ) -> VerifyReport {
        self.run_prepared_selected(bloq, shots, seed, (&[], |_, _| Ok(())), select, inspect)
    }

    fn run_prepared_selected<'a>(
        &'a self,
        bloq: &Bloq,
        shots: usize,
        seed: u64,
        (prepared_ports, mut prepare): (
            &[IVec3],
            impl FnMut(&mut Simulator, &PreparationContext<'_>) -> Result<(), ExecError>,
        ),
        mut select: impl FnMut(&ShotContext<'_>) -> &'a Self,
        mut inspect: impl FnMut(&mut Simulator, &ShotContext<'_>) -> Result<(), ExecError>,
    ) -> VerifyReport {
        let declared = self
            .inputs
            .iter()
            .chain(prepared_ports)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            declared.len(),
            self.inputs.len() + prepared_ports.len(),
            "input roles must not overlap"
        );
        let references = RefCell::new(Vec::new());
        let mut scored = 0;
        let report = run_bloq_with_io(
            bloq,
            shots,
            seed,
            |sim, ctx| {
                assert_eq!(
                    ctx.inputs.len(),
                    declared.len(),
                    "complete Choi input interface"
                );
                assert!(
                    ctx.inputs
                        .iter()
                        .all(|input| declared.contains(&input.port)),
                    "undeclared physical input"
                );
                references.borrow_mut().clear();
                for port in &self.inputs {
                    let input = ctx
                        .inputs
                        .iter()
                        .find(|input| input.port == *port)
                        .unwrap_or_else(|| panic!("missing Choi input {port}"));
                    // VM input seeds start in |+>; a fresh target starts in |0>.
                    let reference = sim.num_qubits();
                    sim.cx(input.qubit, reference)?;
                    references.borrow_mut().push(reference);
                }
                prepare(sim, ctx)
            },
            |sim, ctx| {
                assert_eq!(
                    ctx.outputs.len(),
                    self.outputs.len(),
                    "complete Choi output interface"
                );
                let mut outputs = self
                    .outputs
                    .iter()
                    .map(|port| {
                        let output = ctx
                            .outputs
                            .iter()
                            .find(|output| output.port == *port)
                            .unwrap_or_else(|| panic!("missing Choi output {port}"));
                        assert!(
                            !output.consumed,
                            "Choi output {port} needs capture before reuse"
                        );
                        let frame = ctx
                            .frames
                            .iter()
                            .find(|frame| frame.port == *port)
                            .expect("output frame exists");
                        (
                            output,
                            (
                                frame.x.expect("evaluable X frame"),
                                frame.z.expect("evaluable Z frame"),
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                let reference_outputs = references
                    .borrow()
                    .iter()
                    .map(|&qubit| physical_qubit(qubit, sim.num_qubits()))
                    .collect::<Vec<_>>();
                outputs.extend(reference_outputs.iter().map(|out| (out, (false, false))));
                let ideal = select(ctx);
                assert_eq!(ideal.inputs, self.inputs, "branch ideal input interface");
                assert_eq!(ideal.outputs, self.outputs, "branch ideal output interface");
                assert!(
                    ideal.matches(sim, &outputs)?,
                    "seed={seed} shot={}: corrected Choi state differs from ideal",
                    ctx.shot
                );
                scored += 1;
                inspect(sim, ctx)
            },
        )
        .expect("physical Choi execution");
        assert_eq!(report.discarded, 0);
        assert_eq!(scored, shots, "every noiseless Choi shot must be scored");
        report
    }
}

/// Expose the two existing Bell records without changing physical execution.
pub(crate) struct BellOutcomes {
    pub nodes: [BloqNodeId; 2],
    names: [String; 2],
}

impl BellOutcomes {
    pub(crate) fn expose(bloq: &mut Bloq) -> Self {
        let adapters = bloq
            .quantum_nodes()
            .flat_map(|(id, quantum)| {
                quantum.instances.iter().filter_map(move |instance| {
                    matches!(
                        instance.provenance,
                        InstanceProvenance::IdealPortAdapter {
                            role: PortRole::Input,
                            ..
                        }
                    )
                    .then_some((id, instance.id))
                })
            })
            .collect::<Vec<_>>();
        let [(owner, instance)] = adapters.as_slice() else {
            panic!("fixture must contain exactly one reversed input Bell adapter");
        };
        let names = [0, 1].map(|measurement| format!("Bell {} m{measurement}", instance.0));
        let nodes = std::array::from_fn(|measurement| {
            let node = bloq.add_node(
                BloqNode::classical(ClassicalNode::observable_fragment(
                    vec![InstanceMeasurement {
                        instance: *instance,
                        measurement: measurement as u32,
                    }],
                    Vec::new(),
                ))
                .with_provenance(NodeProvenance::BranchSelector {
                    name: names[measurement].clone(),
                }),
            );
            bloq.add_edge(*owner, node, BloqEdge::Order);
            node
        });
        Self { nodes, names }
    }

    pub(crate) fn read(&self, context: &ShotContext<'_>) -> (bool, bool) {
        let [x, z] = self.names.each_ref().map(|name| {
            context
                .named_branch_selectors
                .iter()
                .find(|(candidate, _)| *candidate == name)
                .expect("the exposed Bell record was evaluated")
                .1
        });
        (x, z)
    }
}

pub(crate) fn assert_wrong_phase_and_uncorrected_byproduct_rejected() {
    let oracle = ExactChoi::new(&[IVec3::ZERO], &[IVec3::Z], |sim| {
        sim.t(0)?;
        Ok(())
    });
    let mut sim = Simulator::with_seed(2, 0);
    sim.h(0);
    sim.cx(0, 1).expect("Bell pair uses two allocated qubits");
    sim.t(0).expect("T acts on an allocated qubit");
    let outputs = [physical_qubit(0, 2), physical_qubit(1, 2)];
    let plain = [(&outputs[0], (false, false)), (&outputs[1], (false, false))];
    assert!(
        oracle
            .matches(&sim, &plain)
            .expect("two-qubit Choi readout is valid")
    );
    sim.z(0);
    assert!(
        !oracle
            .matches(&sim, &plain)
            .expect("two-qubit Choi readout is valid")
    );
    let corrected = [(&outputs[0], (false, true)), (&outputs[1], (false, false))];
    assert!(
        oracle
            .matches(&sim, &corrected)
            .expect("two-qubit Choi readout is valid")
    );
    sim.t(0).expect("T acts on an allocated qubit");
    assert!(
        !oracle
            .matches(&sim, &corrected)
            .expect("two-qubit Choi readout is valid")
    );
}
