"""Four-qubit phase estimation for U = S(target0) CZ(target0, target1).

Two counting qubits resolve the eigenphases of a two-qubit diagonal unitary.
The targets start in |++>, retaining a coherent superposition of eigenstates.
Run this script to retain QASM, the native graph, validated IR, and VM results.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import CU1Gate
from qiskit.quantum_info import Operator


@dataclass(frozen=True)
class PhaseEstimationCircuit:
    source_qasm: str
    clifford_t_qasm: str
    ideal_unitary: np.ndarray
    projective_error: float
    gate_counts: dict[str, int]


def generate_phase_estimation() -> PhaseEstimationCircuit:
    """Return exact Clifford+T QASM checked against an independent QPE matrix.

    Wires 0 and 1 are the little-endian counting register; 2 and 3 are targets.
    U's eigenphase index is target0 * (1 + 2*target1), modulo four.
    """
    source = QuantumCircuit(4)
    source.h(range(4))
    source.append(CU1Gate(np.pi / 2), [0, 2])
    source.ccz(0, 2, 3)
    source.cz(1, 2)  # U^2 = Z(target0).
    # Inverse QFT on the counting register, including its wire reversal.
    source.cx(0, 1)
    source.cx(1, 0)
    source.cx(0, 1)
    source.h(0)
    source.append(CU1Gate(-np.pi / 2), [1, 0])
    source.h(1)
    circuit = QuantumCircuit(4)
    circuit.h(range(4))
    # Shared phase polynomial for CS(a,b) CCZ(a,b,c), in units of pi/4:
    # 2ab+4abc = 2a+2b+c-2(a^b)-(a^c)-(b^c)+(a^b^c), modulo eight.
    # Combining the two diagonal gates uses four T gates rather than ten.
    a, b, c = 0, 2, 3
    circuit.s(a)
    circuit.s(b)
    circuit.t(c)
    circuit.cx(a, b)
    circuit.sdg(b)
    circuit.cx(a, b)
    for control in (a, b):
        circuit.cx(control, c)
        circuit.tdg(c)
        circuit.cx(control, c)
    circuit.cx(a, c)
    circuit.cx(b, c)
    circuit.t(c)
    circuit.cx(b, c)
    circuit.cx(a, c)
    circuit.cz(1, 2)
    circuit.cx(0, 1)
    circuit.cx(1, 0)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.tdg(1)
    circuit.cx(1, 0)
    circuit.t(0)
    circuit.cx(1, 0)
    circuit.tdg(0)
    circuit.h(1)

    fourier = np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4) / 2
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    preparations = np.kron(np.kron(h, h), np.kron(h, h))
    controlled_powers = np.diag([
        np.exp(2j * np.pi * counting * (target0 * (1 + 2 * target1)) / 4)
        for target1 in range(2) for target0 in range(2) for counting in range(4)
    ])
    ideal = np.kron(np.eye(4), fourier.conj().T) @ controlled_powers @ preparations
    source_qasm, clifford_t_qasm = qasm2.dumps(source), qasm2.dumps(circuit)
    actual = Operator(qasm2.loads(clifford_t_qasm)).data
    if np.linalg.norm(Operator(qasm2.loads(source_qasm)).data - ideal, ord=2) > 1e-12:
        raise RuntimeError("source phase estimation differs from the independent matrix")
    overlap = np.vdot(actual, ideal)
    phase = overlap / abs(overlap) if overlap else 1
    error = float(np.linalg.norm(phase * actual - ideal, ord=2))
    if error > 1e-12:
        raise RuntimeError("Clifford+T conversion changed the phase-estimation channel")
    return PhaseEstimationCircuit(
        source_qasm, clifford_t_qasm, ideal, error, dict(circuit.count_ops()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("topologiq-qpe"))
    parser.add_argument("--shots", type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--synthesis-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    started = time.perf_counter()
    circuit = generate_phase_estimation()
    from _topologiq_demo import run_demo

    summary = dict(
        algorithm="four-qubit phase estimation", counting_qubits=[0, 1], target_qubits=[2, 3],
        unitary="S(target0) CZ(target0, target1)",
        projective_spectral_error=circuit.projective_error, gate_counts=circuit.gate_counts,
        t_count=circuit.gate_counts.get("t", 0) + circuit.gate_counts.get("tdg", 0),
        synthesis_seconds=time.perf_counter() - started,
        expected_counting_probabilities=[0.5, 0.25, 0, 0.25],
    )
    result = run_demo(circuit.source_qasm, circuit.clifford_t_qasm, circuit.ideal_unitary,
                      args.output_dir, summary, shots=args.shots,
                      synthesis_only=args.synthesis_only, compile_only=args.compile_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
