//! Shared VM integration-test fixtures.

pub(crate) mod choi;

/// Parse static fixture QASM while keeping QuiZX's output wires in source order.
#[allow(
    dead_code,
    reason = "common is shared with suites that do not use QASM"
)]
pub(crate) fn parse_qasm(source: &str) -> Result<quizx::circuit::Circuit, String> {
    use quizx::{circuit::Circuit, gate::GType};
    let parsed = Circuit::from_qasm(source)?;
    // QuiZX's parser allocates only wires touched by gates.
    let width = source
        .split(';')
        .filter_map(|statement| {
            let register = statement.trim().strip_prefix("qreg ")?;
            register
                .split_once('[')?
                .1
                .strip_suffix(']')?
                .trim()
                .parse::<usize>()
                .ok()
        })
        .sum::<usize>();
    let mut circuit = Circuit::new(width.max(parsed.num_qubits()));
    for gate in parsed.gates {
        // QuiZX tracks SWAP by relabelling later gates, without permuting outputs.
        if gate.t == GType::SWAP {
            let (a, b) = (gate.qs[0], gate.qs[1]);
            circuit.add_gate("cx", vec![a, b]);
            circuit.add_gate("cx", vec![b, a]);
            circuit.add_gate("cx", vec![a, b]);
        } else {
            circuit.push_back(gate);
        }
    }
    Ok(circuit)
}
