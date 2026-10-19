"""Exact algorithm circuits with independent matrix oracles for interop demos."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import CU1Gate
from qiskit.quantum_info import Operator, Pauli, Statevector


@dataclass(frozen=True)
class Candidate:
    source_qasm: str
    clifford_t_qasm: str
    ideal_unitary: np.ndarray
    summary: dict


CANDIDATES = {
    "qpe4": "Four-qubit phase estimation",
    "simon6": "Six-qubit Simon algorithm with a nonlinear oracle",
    "grover3": "Three-qubit Grover search for 101, one iteration",
    "qaoa3": "QAOA MaxCut on a triangle, one layer",
    "qaoa4": "QAOA MaxCut on a four-cycle, one layer",
    "qaoa5": "QAOA MaxCut on a five-cycle, one layer",
    "adder4": "Coherent two-bit modular addition",
    "walk3": "Coined walk on a four-cycle, three steps",
    "walk5": "Coined walk on a four-cycle, five steps",
    "qae3": "Three-qubit amplitude estimation",
    "vqe4": "Four-spin Ising variational ansatz, one energy evaluation",
    "ising4": "Four-spin Ising dynamics, two Trotter steps",
    "ising5": "Five-spin Ising dynamics, two Trotter steps",
    "iqp6": "Six-qubit IQP sampling with eight interactions",
}
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]])


def _kron(*matrices):
    result = np.array([[1.0]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def _ry(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]])


def _ry_quarter(circuit, wire, *, inverse=False):
    circuit.sdg(wire)
    circuit.h(wire)
    (circuit.tdg if inverse else circuit.t)(wire)
    circuit.h(wire)
    circuit.s(wire)


def _finish(name, source, compiled, ideal, **metadata):
    source_qasm, compiled_qasm = qasm2.dumps(source), qasm2.dumps(compiled)
    errors = []
    for text in (source_qasm, compiled_qasm):
        actual = Operator(qasm2.loads(text)).data
        overlap = np.vdot(actual, ideal)
        phase = overlap / abs(overlap) if overlap else 1
        error = float(np.linalg.norm(phase * actual - ideal, ord=2))
        if error > 1e-12:
            raise RuntimeError(f"{name}: exported channel differs from independent matrix: {error}")
        errors.append(error)
    state = Statevector(ideal[:, 0].copy())
    # For a pure n-qubit state the Pauli stabilizer subgroup has size 2**n
    # exactly when the state is a stabilizer state (identity included).
    stabilizers = 1
    for word in range(1, 4**source.num_qubits):
        label = "".join("IXYZ"[(word >> (2*i)) & 3] for i in reversed(range(source.num_qubits)))
        stabilizers += abs(abs(state.expectation_value(Pauli(label))) - 1) < 1e-10
    counts = dict(compiled.count_ops())
    summary = dict(candidate=name, algorithm=CANDIDATES[name], qubits=source.num_qubits,
                   candidate_generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   projective_spectral_error=max(errors), gate_counts=counts,
                   t_count=counts.get("t", 0) + counts.get("tdg", 0),
                   expected_probabilities=state.probabilities().tolist(),
                   output_pauli_stabilizer_count=int(stabilizers),
                   non_stabilizer_output=bool(stabilizers < 2**source.num_qubits), **metadata)
    return Candidate(source_qasm, compiled_qasm, ideal, summary)


def check_algorithm_probabilities(summary, probabilities):
    """Check classical algorithm results derived from the VM's logical state."""
    if summary.get("candidate") == "simon6":
        marginal = np.asarray(probabilities).reshape(8, 8).sum(axis=0)
        support = np.flatnonzero(marginal > 1e-7)
        periods = [s for s in range(1, 8)
                   if all((s & int(y)).bit_count() % 2 == 0 for y in support)]
        if periods != [int(summary["hidden_period"], 2)] or not np.allclose(
                marginal, summary["expected_input_probabilities"], atol=1e-7, rtol=0):
            raise RuntimeError("Simon output does not identify the hidden period")
        return dict(input_probabilities=marginal.tolist(), recovered_period=f"{periods[0]:03b}",
                    recovery="unique nonzero period orthogonal to the VM-supported strings")
    if "expected_cut" in summary:
        cuts = [sum(((x >> a) ^ (x >> b)) & 1 for a, b in summary["edges"])
                for x in range(len(probabilities))]
        value = float(np.dot(probabilities, cuts))
        if abs(value - summary["expected_cut"]) > 1e-7:
            raise RuntimeError("QAOA output has the wrong expected cut")
        return dict(expected_cut=value, maximum_cut=max(cuts))
    if summary.get("family") == "IQP sampling":
        even = sum(p for x, p in enumerate(probabilities) if x.bit_count() % 2 == 0)
        zero = float(probabilities[0])
        if abs(even - 1) > 1e-7 or abs(zero - summary["expected_zero_probability"]) > 1e-7:
            raise RuntimeError("IQP output violates its parity or zero-probability check")
        return dict(even_parity_probability=float(even), zero_probability=zero)
    return {}


def generate_candidate(name):
    """Construct exact Clifford+T sources and matrix-based algorithm oracles."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    if name == "qpe4":
        from topologiq_qpe import generate_phase_estimation

        qpe = generate_phase_estimation()
        return _finish(name, qasm2.loads(qpe.source_qasm), qasm2.loads(qpe.clifford_t_qasm),
                       qpe.ideal_unitary, family="phase estimation", counting_qubits=[0, 1])
    if name == "grover3":
        source = QuantumCircuit(3)
        source.h(range(3))
        source.x(1)
        source.ccz(0, 1, 2)
        source.x(1)
        source.h(range(3))
        source.x(range(3))
        source.ccz(0, 1, 2)
        source.x(range(3))
        source.h(range(3))
        compiled = transpile(source, basis_gates=["h", "x", "cx", "t", "tdg"],
                             optimization_level=0, seed_transpiler=17)
        oracle = np.eye(8)
        oracle[5, 5] = -1
        diffusion = np.ones((8, 8)) / 4 - np.eye(8)
        ideal = diffusion @ oracle @ _kron(_H, _H, _H)
        return _finish(name, source, compiled, ideal, family="search", marked_integer=5,
                       expected_success_probability=25/32)
    if name == "simon6":
        source = QuantumCircuit(6)
        source.h(range(3))
        # Compute u on an input wire, then restore it: the oracle XORs f(x)
        # into arbitrary output-register contents, not only into |000>.
        source.cx(2, 0)
        source.cx(0, 3)
        source.cx(1, 4)
        source.ccx(0, 1, 5)
        source.cx(2, 0)
        source.h(range(3))
        compiled = transpile(source, basis_gates=["h", "x", "cx", "t", "tdg"],
                             optimization_level=0, seed_transpiler=17)
        oracle = np.zeros((64, 64))
        for x in range(8):
            u, v = (x & 1) ^ ((x >> 2) & 1), (x >> 1) & 1
            value = u | (v << 1) | ((u & v) << 2)
            for y in range(8):
                oracle[x + ((y ^ value) << 3), x + (y << 3)] = 1
        walsh = _kron(np.eye(8), _H, _H, _H)
        return _finish(name, source, compiled, walsh @ oracle @ walsh,
                       family="hidden-period finding", input_qubits=[0, 1, 2],
                       hidden_period="101", oracle="(x0 xor x2, x1, (x0 xor x2) and x1)",
                       expected_input_probabilities=[1/4, 0, 1/4, 0, 0, 1/4, 0, 1/4])
    if name.startswith("qaoa"):
        qubits = int(name[-1])
        edges = [(i, (i + 1) % qubits) for i in range(qubits)]
        gamma, beta = np.pi/4, np.pi/8
        source, compiled = QuantumCircuit(qubits), QuantumCircuit(qubits)
        source.h(range(qubits))
        compiled.h(range(qubits))
        for a, b in edges:
            source.cx(a, b)
            source.rz(-gamma, b)
            source.cx(a, b)
            compiled.cx(a, b)
            compiled.tdg(b)
            compiled.cx(a, b)
        for wire in range(qubits):
            source.rx(2*beta, wire)
            compiled.h(wire)
            compiled.t(wire)
            compiled.h(wire)
        cuts = np.array([sum(((x >> a) ^ (x >> b)) & 1 for a, b in edges) for x in range(1 << qubits)])
        mixer = np.cos(beta)*np.eye(2) - 1j*np.sin(beta)*_X
        ideal = _kron(*([mixer]*qubits)) @ np.diag(np.exp(-1j*gamma*cuts)) @ _kron(*([_H]*qubits))
        return _finish(name, source, compiled, ideal, family="variational optimization", edges=edges,
                       gamma="pi/4", beta="pi/8", expected_cut=float(np.dot(abs(ideal[:, 0])**2, cuts)))
    if name == "adder4":
        source = QuantumCircuit(4)
        source.h([0, 1, 2])
        source.ccx(0, 2, 3)
        source.cx(0, 2)
        source.cx(1, 3)
        compiled = transpile(source, basis_gates=["h", "x", "cx", "t", "tdg"],
                             optimization_level=0, seed_transpiler=17)
        permutation = np.zeros((16, 16))
        for a in range(4):
            for b in range(4):
                permutation[a + (((a + b) % 4) << 2), a + (b << 2)] = 1
        ideal = permutation @ _kron(np.eye(2), _H, _H, _H)
        return _finish(name, source, compiled, ideal, family="arithmetic", a_qubits=[0, 1], b_qubits=[2, 3],
                       arithmetic="|a,b> -> |a,(a+b) mod 4>")
    if name.startswith("walk"):
        steps = int(name[-1])
        source, compiled = QuantumCircuit(3), QuantumCircuit(3)
        for _ in range(steps):
            source.ry(np.pi/4, 0)
            _ry_quarter(compiled, 0)
            for circuit in (source, compiled):
                # On a four-cycle, the +/-1 carry equals NOT(position0 XOR coin).
                circuit.x(2)
                circuit.cx(1, 2)
                circuit.cx(0, 2)
                circuit.x(1)
        shift = np.zeros((8, 8))
        for coin in range(2):
            for position in range(4):
                following = (position + (1 if coin else -1)) % 4
                shift[coin + 2*following, coin + 2*position] = 1
        ideal = np.linalg.matrix_power(shift @ np.kron(np.eye(4), _ry(np.pi/4)), steps)
        return _finish(name, source, compiled, ideal, family="quantum walk", steps=steps,
                       coin_qubit=0, position_qubits=[1, 2], coin_angle="pi/4")
    if name == "vqe4":
        source, compiled = QuantumCircuit(4), QuantumCircuit(4)
        for wire in range(4):
            source.ry(np.pi/4, wire)
            _ry_quarter(compiled, wire)
        for wire in range(3):
            source.cx(wire, wire+1)
            compiled.cx(wire, wire+1)
        entangler = np.zeros((16, 16))
        for basis in range(16):
            target = basis
            for wire in range(3):
                target ^= ((target >> wire) & 1) << (wire+1)
            entangler[target, basis] = 1
        ideal = entangler @ _kron(*([_ry(np.pi/4)]*4))
        spins = np.array([[1-2*((x >> q) & 1) for q in range(4)] for x in range(16)])
        energy = -np.dot(abs(ideal[:, 0])**2, np.sum(spins[:, :-1]*spins[:, 1:], axis=1))
        for wire in range(4):
            energy -= np.vdot(ideal[:, 0], ideal[np.arange(16) ^ (1 << wire), 0]).real
        return _finish(name, source, compiled, ideal, family="variational energy estimation",
                       hamiltonian="-sum Z_i Z_(i+1) - sum X_i (open chain)",
                       expected_energy=float(energy), optimized=False)
    if name.startswith("ising"):
        qubits = int(name[-1])
        dimension = 1 << qubits
        neel = sum(1 << wire for wire in range(0, qubits, 2))
        source, compiled = QuantumCircuit(qubits), QuantumCircuit(qubits)
        for circuit in (source, compiled):
            circuit.x(range(0, qubits, 2))
        for _ in range(2):
            for wire in range(qubits - 1):
                source.cx(wire, wire+1)
                source.rz(np.pi/2, wire+1)
                source.cx(wire, wire+1)
                compiled.cx(wire, wire+1)
                compiled.s(wire+1)
                compiled.cx(wire, wire+1)
            for wire in range(qubits):
                source.rx(np.pi/4, wire)
                compiled.h(wire)
                compiled.t(wire)
                compiled.h(wire)
        interaction = np.array([sum((1-2*((x >> q) & 1))*(1-2*((x >> (q+1)) & 1))
                                    for q in range(qubits - 1)) for x in range(dimension)])
        x_sum = np.zeros((dimension, dimension))
        for wire in range(qubits):
            x_sum[np.arange(dimension) ^ (1 << wire), np.arange(dimension)] += 1
        delta = np.pi/8
        rotation = np.cos(delta)*np.eye(2) - 1j*np.sin(delta)*_X
        step = _kron(*([rotation]*qubits)) @ np.diag(np.exp(-2j*delta*interaction))
        preparation = np.eye(dimension)[:, np.arange(dimension) ^ neel]
        ideal = step @ step @ preparation
        energies, eigenvectors = np.linalg.eigh(2*np.diag(interaction)+x_sum)
        exact = (eigenvectors * np.exp(-2j*delta*energies)) @ eigenvectors.conj().T @ preparation
        return _finish(name, source, compiled, ideal, family="Hamiltonian simulation", steps=2,
                       hamiltonian="2 sum Z_i Z_(i+1) + sum X_i (open chain)",
                       evolution_time="pi/4", trotter_spectral_error=float(np.linalg.norm(ideal-exact, ord=2)))
    if name == "iqp6":
        edges = [(wire, (wire + 1) % 6) for wire in range(6)] + [(0, 3), (1, 4)]
        source, compiled = QuantumCircuit(6), QuantumCircuit(6)
        source.h(range(6))
        compiled.h(range(6))
        for a, b in edges:
            source.cx(a, b)
            source.rz(np.pi/4, b)
            source.cx(a, b)
            compiled.cx(a, b)
            compiled.t(b)
            compiled.cx(a, b)
        source.h(range(6))
        compiled.h(range(6))
        interaction = np.array([sum((1-2*((x >> a) & 1))*(1-2*((x >> b) & 1))
                                    for a, b in edges) for x in range(64)])
        walsh = _kron(*([_H]*6))
        ideal = walsh @ np.diag(np.exp(-1j*np.pi/8*interaction)) @ walsh
        return _finish(name, source, compiled, ideal, family="IQP sampling", edges=edges,
                       interaction_angle="pi/4", expected_even_parity_probability=1,
                       expected_zero_probability=float((3+2*np.sqrt(2))/16))
    # Canonical amplitude estimation: A=Ry(pi/4), Q=Ry(pi/2).
    # Two counting qubits give a finite-resolution estimate of sin(pi/8)**2.
    source, compiled = QuantumCircuit(3), QuantumCircuit(3)
    for circuit in (source, compiled):
        circuit.h([0, 1])
    source.ry(np.pi/4, 2)
    for control, angle in ((0, np.pi/2), (1, np.pi)):
        source.ry(angle/2, 2)
        source.cx(control, 2)
        source.ry(-angle/2, 2)
        source.cx(control, 2)
    _ry_quarter(compiled, 2)
    _ry_quarter(compiled, 2)
    compiled.cx(0, 2)
    _ry_quarter(compiled, 2, inverse=True)
    compiled.cx(0, 2)
    compiled.sdg(1)
    compiled.sdg(2)
    compiled.cx(1, 2)
    compiled.s(2)
    for circuit in (source, compiled):
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.h(0)
    source.append(CU1Gate(-np.pi/2), [1, 0])
    compiled.tdg(1)
    compiled.cx(1, 0)
    compiled.t(0)
    compiled.cx(1, 0)
    compiled.tdg(0)
    source.h(1)
    compiled.h(1)
    powers = np.zeros((8, 8))
    for counting in range(4):
        powers[np.ix_([counting, counting+4], [counting, counting+4])] = _ry(counting*np.pi/2)
    fourier = np.exp(2j*np.pi*np.outer(np.arange(4), np.arange(4))/4)/2
    ideal = np.kron(np.eye(2), fourier.conj().T) @ powers @ _kron(_ry(np.pi/4), _H, _H)
    return _finish(name, source, compiled, ideal, family="amplitude estimation",
                   target_probability=float(np.sin(np.pi/8)**2), counting_qubits=[0, 1])
