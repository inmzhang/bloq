"""Generate an approximate four-qubit QFT for the Topologiq integration.

Optional dependencies: numpy, qiskit, mpmath, and pygridsynth==2.0.0.
Run this file to compile the logical circuit through Topologiq into Bloq IR
and execute it with the VM. The output retains both QASM inputs, the block
graph, IR, and measured verification results. Use --compile-only to retain IR
without VM execution, or --synthesis-only for the rotation-synthesis check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np
from pygridsynth import gridsynth_circuit
from pygridsynth.quantum_gate import HGate, SGate, SXGate, TGate, WGate
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import CU1Gate
from qiskit.quantum_info import Operator


@dataclass(frozen=True)
class QftSynthesis:
    qubits: int
    epsilon: float
    source_qasm: str
    synthesized_qasm: str
    ideal_unitary: np.ndarray
    synthesized_unitary: np.ndarray
    projective_error: float
    t_count: int
    gate_counts: dict[str, int]


def _projective_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Bound spectral error after aligning global phase by matrix overlap."""
    # QASM 2 omits global phase, which does not change the quantum channel.
    overlap = np.vdot(actual, expected)
    phase = overlap / abs(overlap) if overlap else 1
    return float(np.linalg.norm(phase * actual - expected, ord=2))


def _rotation(denominator: int, epsilon: mp.mpf) -> QuantumCircuit:
    synthesized = gridsynth_circuit(
        mp.pi / denominator, epsilon, seed=0, up_to_phase=True,
    )
    circuit = QuantumCircuit(1)
    circuit.global_phase = float(synthesized.phase)
    # pygridsynth lists matrix factors; Qiskit appends gates in time order.
    for gate in reversed(synthesized):
        if isinstance(gate, HGate):
            circuit.h(0)
        elif isinstance(gate, TGate):
            circuit.t(0)
        elif isinstance(gate, SGate):
            circuit.s(0)
        elif isinstance(gate, SXGate):
            circuit.x(0)
        elif isinstance(gate, WGate):
            circuit.global_phase += np.pi / 4
        else:
            raise TypeError(f"unsupported synthesis gate: {gate!r}")
    actual = Operator(circuit).data
    synthesized_matrix = np.array(synthesized.to_complex_matrix(1).tolist(), dtype=complex)
    if np.linalg.norm(actual - synthesized_matrix, ord=2) > 1e-12:
        raise RuntimeError("rotation conversion changed pygridsynth's matrix")
    angle = float(mp.pi / denominator)
    target = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    if _projective_error(actual, target) > float(epsilon):
        raise RuntimeError("synthesized rotation exceeds its error budget")
    return circuit


def generate_qft(qubits: int = 4, epsilon: float = 0.01) -> QftSynthesis:
    """Synthesize a one- through four-qubit QFT and check its complete matrix.

    Controlled phase rotations use three single-qubit Rz gates and two CNOTs.
    The controlled pi/2 rotations are exact Clifford+T; the remaining rotations
    share the requested error budget. Negative angles invert a synthesized
    positive rotation. The default has 273 T/Tdag gates with pygridsynth 2.0.0.

    The independent oracle is the positive discrete Fourier matrix in integer
    computational-basis order, including the QFT's final wire reversal.
    """
    if isinstance(qubits, bool) or not isinstance(qubits, int) or not 1 <= qubits <= 4:
        raise ValueError("qubits must be an integer between 1 and 4")
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and between zero and one")
    if version("pygridsynth") != "2.0.0":
        raise RuntimeError("this reproducible example requires pygridsynth==2.0.0")

    source = QuantumCircuit(qubits)
    for control in reversed(range(qubits)):
        source.h(control)
        for target in reversed(range(control)):
            source.append(CU1Gate(np.pi / 2 ** (control - target)), [control, target])
    for left in range(qubits // 2):
        right = qubits - 1 - left
        source.cx(left, right)
        source.cx(right, left)
        source.cx(left, right)

    dimension = 1 << qubits
    rows, columns = np.meshgrid(np.arange(dimension), np.arange(dimension), indexing="ij")
    ideal = np.exp(2j * np.pi * rows * columns / dimension) / np.sqrt(dimension)
    if np.linalg.norm(Operator(source).data - ideal, ord=2) > 1e-12:
        raise RuntimeError("source QFT differs from the independent Fourier matrix")

    approximate_rotations = 3 * sum(qubits - step for step in range(2, qubits))
    with mp.workdps(80):
        budget = mp.mpf(str(epsilon)) / max(1, approximate_rotations)
        rotations = {
            2 ** (step + 1): _rotation(2 ** (step + 1), budget)
            for step in range(2, qubits)
        }

    circuit = QuantumCircuit(qubits)

    def rz(denominator: int, wire: int, *, negative: bool = False) -> None:
        if denominator == 4:
            if negative:
                circuit.tdg(wire)
            else:
                circuit.t(wire)
            circuit.global_phase += np.pi / 8 if negative else -np.pi / 8
        else:
            rotation = rotations[denominator]
            circuit.compose(rotation.inverse() if negative else rotation, [wire], inplace=True)

    for control in reversed(range(qubits)):
        circuit.h(control)
        for target in reversed(range(control)):
            denominator = 2 ** (control - target + 1)
            rz(denominator, control)
            circuit.cx(control, target)
            rz(denominator, target, negative=True)
            circuit.cx(control, target)
            rz(denominator, target)
            circuit.global_phase += np.pi / (2 ** (control - target) * 4)
    for left in range(qubits // 2):
        right = qubits - 1 - left
        circuit.cx(left, right)
        circuit.cx(right, left)
        circuit.cx(left, right)

    synthesized_qasm = qasm2.dumps(circuit)
    # Check the exported artifact, including serialization and qubit ordering.
    actual = Operator(qasm2.loads(synthesized_qasm)).data
    error = _projective_error(actual, ideal)
    if error > epsilon:
        raise RuntimeError(f"QFT operator error {error:.6g} exceeds epsilon={epsilon:.6g}")
    counts = dict(circuit.count_ops())
    return QftSynthesis(
        qubits=qubits, epsilon=epsilon,
        source_qasm=qasm2.dumps(source), synthesized_qasm=synthesized_qasm,
        ideal_unitary=ideal, synthesized_unitary=actual, projective_error=error,
        t_count=counts.get("t", 0) + counts.get("tdg", 0), gate_counts=counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("topologiq-qft"))
    parser.add_argument("--shots", type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--synthesis-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true",
                       help="retain validated IR without starting the VM")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    print("Synthesizing and checking the complete QFT matrix...", file=sys.stderr, flush=True)
    started = time.perf_counter()
    result = generate_qft(args.qubits, args.epsilon)
    # F|1> exposes the fine non-Clifford phases in the phase-gradient output.
    def prepare(text: str) -> str:
        circuit = QuantumCircuit(result.qubits)
        circuit.x(0)
        circuit.compose(qasm2.loads(text), inplace=True)
        return qasm2.dumps(circuit)

    source_qasm, compiled_qasm = prepare(result.source_qasm), prepare(result.synthesized_qasm)
    summary = {
        "algorithm": "quantum Fourier transform",
        "qubits": result.qubits, "epsilon": result.epsilon,
        "input_basis_index": 1,
        "projective_spectral_error": result.projective_error,
        "t_count": result.t_count, "gate_counts": result.gate_counts,
        "pygridsynth_version": version("pygridsynth"),
        "synthesis_seconds": time.perf_counter() - started,
    }
    from _topologiq_demo import run_demo

    # The source error is bounded against the Fourier matrix above; execution
    # must reproduce the synthesized circuit, including its X input preparation.
    prepared = result.synthesized_unitary[:, np.arange(1 << result.qubits) ^ 1]
    summary["ideal_qft_state_fidelity"] = float(abs(np.vdot(
        result.ideal_unitary[:, 1], prepared[:, 0],
    ))**2)
    execution = run_demo(source_qasm, compiled_qasm, prepared, args.output_dir, summary,
                         shots=args.shots, synthesis_only=args.synthesis_only,
                         compile_only=args.compile_only)
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
