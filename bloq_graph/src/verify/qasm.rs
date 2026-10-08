//! Interpretation of the shared QASM instructions as pure ZX branch maps.

use bloq_utils::qasm::{QasmGate, QasmInstruction, QasmProgram};
use quizx::circuit::Circuit;
use quizx::graph::{BasisElem, GraphLike};

use crate::{Basis, PauliBasis};

use super::QuizxGraph;

pub use bloq_utils::qasm::QasmError;

/// Parses the unitary part of an unconditional circuit.
///
/// Resets must precede gates and measurements on each wire, and measurements
/// must be terminal. Both are removed from the returned circuit. Use
/// [`super::LogicalVerifier`] to compare maps with classical branches.
///
/// # Errors
///
/// Returns a parse, boundary-order, or unsupported-conditional error.
pub fn parse_qasm(source: &str) -> Result<Circuit, QasmError> {
    let parsed = QasmCircuit::parse(source)?;
    if parsed.program.instructions().iter().any(|instruction| {
        matches!(instruction, QasmInstruction::Gate { condition, .. } if !condition.is_empty())
    }) {
        return Err(QasmError::Boundary(
            "conditional operations are not supported by unitary circuit parsing".into(),
        ));
    }
    Ok(parsed.unitary(&[]).to_basic_gates())
}

pub(super) struct QasmCircuit {
    pub(super) program: QasmProgram,
    pub(super) reset_inputs: Vec<bool>,
    pub(super) measurements: Vec<Option<usize>>,
}

impl QasmCircuit {
    pub(super) fn parse(source: &str) -> Result<Self, QasmError> {
        let program = bloq_utils::qasm::parse_qasm(source)?;
        let mut touched = vec![false; program.qubits().len()];
        let mut reset_inputs = touched.clone();
        let mut measurements = vec![None; touched.len()];
        for instruction in program.instructions() {
            match instruction {
                QasmInstruction::Reset { qubit } => {
                    if touched[*qubit] {
                        return Err(QasmError::Boundary(format!(
                            "reset must precede gates and measurements on {}",
                            program.qubits()[*qubit]
                        )));
                    }
                    reset_inputs[*qubit] = true;
                }
                QasmInstruction::Measure { qubit, bit } => {
                    if measurements[*qubit].replace(*bit).is_some() {
                        return Err(QasmError::Boundary(format!(
                            "{} is measured more than once",
                            program.qubits()[*qubit]
                        )));
                    }
                    touched[*qubit] = true;
                }
                QasmInstruction::Gate { qubits, .. } => {
                    for &qubit in qubits {
                        if measurements[qubit].is_some() {
                            return Err(QasmError::Boundary(format!(
                                "gate follows measurement of {}",
                                program.qubits()[qubit]
                            )));
                        }
                        touched[qubit] = true;
                    }
                }
            }
        }
        Ok(Self {
            program,
            reset_inputs,
            measurements,
        })
    }

    pub(super) fn input_count(&self) -> usize {
        self.reset_inputs.iter().filter(|reset| !**reset).count()
    }

    pub(super) fn output_count(&self) -> usize {
        self.measurements.iter().filter(|bit| bit.is_none()).count()
    }

    fn unitary(&self, values: &[bool]) -> Circuit {
        let mut circuit = Circuit::new(self.program.qubits().len());
        for instruction in self.program.instructions() {
            let QasmInstruction::Gate {
                gate,
                qubits,
                condition,
            } = instruction
            else {
                continue;
            };
            if !condition
                .iter()
                .all(|&(bit, required)| values[bit] == required)
            {
                continue;
            }
            match *gate {
                QasmGate::Identity => {}
                QasmGate::H => circuit.add_gate("h", qubits.clone()),
                QasmGate::Pauli(PauliBasis::X) => circuit.add_gate("x", qubits.clone()),
                QasmGate::Pauli(PauliBasis::Z) => circuit.add_gate("z", qubits.clone()),
                QasmGate::Pauli(PauliBasis::Y) => {
                    // The omitted i is a global phase within a classical branch.
                    circuit.add_gate("z", qubits.clone());
                    circuit.add_gate("x", qubits.clone());
                }
                QasmGate::Phase { basis, quarters } => circuit.add_gate_with_phase(
                    match basis {
                        Basis::X => "rx",
                        Basis::Z => "rz",
                    },
                    qubits.clone(),
                    (quarters, 4),
                ),
                QasmGate::Cx => circuit.add_gate("cx", qubits.clone()),
                QasmGate::Cz => circuit.add_gate("cz", qubits.clone()),
                QasmGate::Xcx => circuit.add_gate("xcx", qubits.clone()),
                QasmGate::Swap => circuit.add_gate("swap", qubits.clone()),
                QasmGate::Ccx => circuit.add_gate("ccx", qubits.clone()),
                QasmGate::Ccz => circuit.add_gate("ccz", qubits.clone()),
            }
        }
        circuit
    }

    pub(super) fn instantiate(&self, values: &[bool]) -> QuizxGraph {
        let mut graph: QuizxGraph = self.unitary(values).to_graph();
        graph.plug_inputs(
            &self
                .reset_inputs
                .iter()
                .map(|reset| {
                    if *reset {
                        BasisElem::Z0
                    } else {
                        BasisElem::SKIP
                    }
                })
                .collect::<Vec<_>>(),
        );
        graph.plug_outputs(
            &self
                .measurements
                .iter()
                .map(|bit| match bit {
                    Some(bit) if values[*bit] => BasisElem::Z1,
                    Some(_) => BasisElem::Z0,
                    None => BasisElem::SKIP,
                })
                .collect::<Vec<_>>(),
        );
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::tensor::{CompareTensors, TensorF, ToTensor};

    #[test]
    fn unitary_parser_checks_boundary_order_and_rejects_conditionals() {
        let parsed =
            parse_qasm("OPENQASM 2.0; qreg a[1]; qreg b[2]; reset a; barrier a,b; h a; t b[0];")
                .unwrap();
        assert_eq!(parsed.num_qubits(), 3);
        let measured = parse_qasm("OPENQASM 2.0; qreg q[1]; creg c[1]; measure q -> c;").unwrap();
        assert!(measured.gates.is_empty());
        for body in [
            "measure q -> c; x q;",
            "h q; reset q;",
            "measure q -> c; if(c==1) x q;",
            "if(c==0) x q;",
        ] {
            parse_qasm(&format!("OPENQASM 2.0; qreg q[1]; creg c[1]; {body}")).unwrap_err();
        }
    }

    #[test]
    fn ancilla_parity_and_conditional_pauli_keep_both_source_branches() {
        let parsed = QasmCircuit::parse("OPENQASM 2.0; qreg q[2]; creg m[1]; reset q[1]; h q[1]; measure q[1] -> m; if(m==1) x q[0];").unwrap();
        for bit in [false, true] {
            let actual = parsed.instantiate(&[bit]);
            let mut expected = Circuit::new(1);
            if bit {
                expected.add_gate("x", vec![0]);
            }
            assert!(<TensorF as CompareTensors>::scalar_eq(
                &actual.to_tensorf(),
                &expected.to_tensorf()
            ));
        }
    }
}
