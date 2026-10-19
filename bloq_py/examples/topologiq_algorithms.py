"""Compare small algorithm candidates through Topologiq, Bloq IR, and the VM.

Each candidate has an independent complete-matrix oracle and prepares its input
from zero. Run in the optional, pinned interop environment described in the
Topologiq tutorial. Each subprocess retains artifacts, errors, and timings.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

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
    "grover3": "Three-qubit Grover search for 101, one iteration",
    "qaoa3": "QAOA MaxCut on a triangle, one layer",
    "qaoa4": "QAOA MaxCut on a four-cycle, one layer",
    "adder4": "Coherent two-bit modular addition",
    "walk3": "Coined walk on a four-cycle, three steps",
    "walk5": "Coined walk on a four-cycle, five steps",
    "qae3": "Three-qubit amplitude estimation",
    "vqe4": "Four-spin Ising variational ansatz, one energy evaluation",
    "ising4": "Four-spin Ising dynamics, two Trotter steps",
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
    if name == "ising4":
        source, compiled = QuantumCircuit(4), QuantumCircuit(4)
        for circuit in (source, compiled):
            circuit.x([0, 2])
        for _ in range(2):
            for wire in range(3):
                source.cx(wire, wire+1)
                source.rz(np.pi/2, wire+1)
                source.cx(wire, wire+1)
                compiled.cx(wire, wire+1)
                compiled.s(wire+1)
                compiled.cx(wire, wire+1)
            for wire in range(4):
                source.rx(np.pi/4, wire)
                compiled.h(wire)
                compiled.t(wire)
                compiled.h(wire)
        interaction = np.array([sum((1-2*((x >> q) & 1))*(1-2*((x >> (q+1)) & 1))
                                    for q in range(3)) for x in range(16)])
        x_sum = np.zeros((16, 16))
        for wire in range(4):
            x_sum[np.arange(16) ^ (1 << wire), np.arange(16)] += 1
        delta = np.pi/8
        rotation = np.cos(delta)*np.eye(2) - 1j*np.sin(delta)*_X
        step = _kron(*([rotation]*4)) @ np.diag(np.exp(-2j*delta*interaction))
        preparation = np.eye(16)[:, np.arange(16) ^ 5]
        ideal = step @ step @ preparation
        energies, eigenvectors = np.linalg.eigh(2*np.diag(interaction)+x_sum)
        exact = (eigenvectors * np.exp(-2j*delta*energies)) @ eigenvectors.conj().T @ preparation
        return _finish(name, source, compiled, ideal, family="Hamiltonian simulation", steps=2,
                       hamiltonian="2 sum Z_i Z_(i+1) + sum X_i (open chain)",
                       evolution_time="pi/4", trotter_spectral_error=float(np.linalg.norm(ideal-exact, ord=2)))
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


def _worker(name, args):
    from _topologiq_demo import run_demo

    started = time.perf_counter()
    candidate = generate_candidate(name)
    candidate.summary["synthesis_seconds"] = time.perf_counter() - started
    return run_demo(candidate.source_qasm, candidate.clifford_t_qasm, candidate.ideal_unitary,
                    args.output_dir, candidate.summary, shots=args.shots,
                    synthesis_only=args.synthesis_only, compile_only=args.compile_only,
                    layout_seed=args.layout_seed)


def _comparison_markdown(comparison):
    lines = ["# Topologiq algorithm candidates", "",
             "Noiseless VM execution with ideal T sources and ideal boundaries; requested distance three.",
             "Every accepted demo shot checks all joint Pauli expectations against an independent state.", "",
             "| Candidate | Qubits | T/T† | Blocks | VM seconds | Peak rank | Result |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for result in comparison["candidates"]:
        status = result["status"]
        if status in ("failed", "timeout"):
            status += f" ({result['failed_stage']})"
        values = [result.get("algorithm", result["candidate"]), result.get("qubits", "—"),
                  result.get("t_count", "—"), result.get("layout_blocks", "—"),
                  f"{result['execution_seconds']:.3f}" if "execution_seconds" in result else "—",
                  result.get("peak_rank", "—"), status]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Diagnostics and artifacts", ""]
    for result in comparison["candidates"]:
        name = result["candidate"]
        detail = result.get("error", result.get("vm_verification", "source/compilation artifacts retained"))
        lines.append(f"- [{name}]({name}/result.json): {detail}. [Log]({result['log']})")
    lines += ["", "VM times cover the requested shots on this host; they are not hardware timing estimates.",
              "The routing profile permits Hadamard pipes only on the temporal axis.", ""]
    return "\n".join(lines)


def main():
    from _topologiq_demo import save_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=Path("topologiq-algorithms"))
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--layout-seed", type=int, default=1330)
    parser.add_argument("--timeout", type=float, default=180, help="seconds allowed per candidate")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--synthesis-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shots < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("shots and timeout must be positive and finite")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("candidates must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    if args.worker:
        if len(args.candidates) != 1:
            parser.error("worker requires exactly one candidate")
        try:
            print(json.dumps(_worker(args.candidates[0], args), indent=2))
        except Exception as error:
            result_file = args.output_dir / "result.json"
            if not result_file.exists():
                args.output_dir.mkdir(parents=True, exist_ok=True)
                save_json(result_file, dict(candidate=args.candidates[0], status="failed",
                    failed_stage="synthesis", error_type=type(error).__name__, error=str(error)))
            raise
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = dict(shots_requested=args.shots, timeout_seconds=args.timeout, candidates=[])
    for name in args.candidates:
        output = args.output_dir / name
        output.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--candidates", name,
                   "--shots", str(args.shots), "--layout-seed", str(args.layout_seed),
                   "--output-dir", str(output.resolve())]
        if args.synthesis_only:
            command.append("--synthesis-only")
        if args.compile_only:
            command.append("--compile-only")
        print(f"Trying {name}: {CANDIDATES[name]}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        with (args.output_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "failed" if completed.returncode else None
            except subprocess.TimeoutExpired:
                status = "timeout"
        result_file = output / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else dict(candidate=name)
        if status:
            result.update(status=status, failed_stage=result.get("failed_stage", result.get("stage", "synthesis")))
            save_json(result_file, result)
        result["total_seconds"] = time.perf_counter() - started
        result["log"] = f"{name}.log"
        comparison["candidates"].append(result)
        save_json(args.output_dir / "comparison.json", comparison)
        (args.output_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    if any(c["status"] in ("failed", "timeout") for c in comparison["candidates"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
