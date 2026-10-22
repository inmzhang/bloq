use quizx::circuit::Circuit;
use quizx::gate::{GType, Gate};
use quizx::params::Parity;

pub(crate) fn qasm(text: &str) -> Result<Circuit, String> {
    // openqasm 0.1's conditional lowering indexes a classical name as a
    // quantum register and panics before QuiZX can reject the operation.
    if has_conditional(text) {
        return Err("classically conditional QASM is not supported".into());
    }
    let circuit = Circuit::from_qasm(text)?;
    // QuiZX approximates numeric rotation parameters through f32. Exact
    // rotations remain available through the rational-valued Circuit API.
    if circuit
        .gates
        .iter()
        .any(|gate| matches!(gate.t, GType::XPhase | GType::ZPhase))
    {
        return Err("QASM rotations must be synthesized to discrete Clifford+T gates first".into());
    }
    normalize(&circuit)
}

fn has_conditional(text: &str) -> bool {
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        chars.next();
                    } else if c == '"' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'/') => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        break;
                    }
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut token = String::from(c);
                while chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
                {
                    token.push(chars.next().expect("peeked identifier character"));
                }
                if token == "if" {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

pub(crate) fn normalize(circuit: &Circuit) -> Result<Circuit, String> {
    if circuit.num_qubits() == 0 {
        return Err("the source must have at least one qubit".into());
    }
    let mut result = Circuit::new(circuit.num_qubits());
    for (index, gate) in circuit.gates.iter().enumerate() {
        if !matches!(
            gate.t,
            GType::HAD
                | GType::NOT
                | GType::Z
                | GType::S
                | GType::Sdg
                | GType::T
                | GType::Tdg
                | GType::CNOT
                | GType::CZ
                | GType::SWAP
                | GType::TOFF
                | GType::CCZ
                | GType::XPhase
                | GType::ZPhase
        ) || gate.vars != Parity::default()
        {
            return Err(format!(
                "gate {index} is not a static Clifford+T operation: {:?}",
                gate.t
            ));
        }
        if gate.t.num_qubits() != Some(gate.qs.len())
            || gate.qs.iter().any(|&q| q >= circuit.num_qubits())
            || gate
                .qs
                .iter()
                .enumerate()
                .any(|(i, q)| gate.qs[..i].contains(q))
        {
            return Err(format!("gate {index} has invalid or repeated operands"));
        }
        if 4 % gate.phase.to_rational().denom() != 0 {
            return Err(format!(
                "gate {index} has a non-Clifford+T phase {}",
                gate.phase
            ));
        }
        if gate.t == GType::SWAP {
            // QuiZX 0.3's graph conversion otherwise loses the output permutation.
            let [a, b] = [gate.qs[0], gate.qs[1]];
            for operands in [[a, b], [b, a], [a, b]] {
                result.push(Gate::new(GType::CNOT, operands.to_vec()));
            }
        } else {
            gate.push_basic_gates(&mut result);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::tensor::{CompareTensors, TensorF, ToTensor};
    use quizx::vec_graph::Graph;

    #[test]
    fn swap_keeps_source_wire_order_through_graph_conversion() {
        let original = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; swap q[0],q[1]; h q[0]; z q[1];",
        )
        .unwrap();
        let graph: Graph = normalize(&original).unwrap().to_graph();
        let expected: TensorF = original.to_tensor();
        let actual: TensorF = graph.to_tensor();
        assert!(TensorF::scalar_eq(&actual, &expected));
    }

    #[test]
    fn rejects_dynamic_approximate_and_malformed_sources() {
        for body in [
            "measure q[0] -> c[0];",
            "reset q[0];",
            "if(c==1) x q[0];",
            "include \"ignored//path\"; if(c==1) x q[0];",
            "rz(0.00000001) q[0];",
            "cx q[0],q[0];",
        ] {
            assert!(
                qasm(&format!(
                    "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; {body}"
                ))
                .is_err(),
                "{body}"
            );
        }
        let mut circuit = Circuit::new(1);
        circuit.push(Gate::new_with_phase(GType::ZPhase, vec![0], (1, 8)));
        normalize(&circuit).expect_err("non-quarter phase must be rejected");
        circuit.gates[0].phase = (1, 4).into();
        normalize(&circuit).expect("exact quarter phase is supported");
        qasm("OPENQASM 2.0; qreg q[1]; // if is only a comment\nh q[0];")
            .expect("comments are not dynamic control");
    }
}
