"""Pinned MQT Bench sources lowered faithfully to static Clifford+T QASM 2.

MQT Bench supplies algorithm-level circuits. Lowering keeps every Qiskit
rotation angle: multiples of pi/4 become exact Clifford+T words, and other
angles use pygridsynth with an explicit per-rotation budget. Terminal
measurements become the readout specification; any other measurement, reset,
or classical control is rejected instead of being removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
import math

import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator, Statevector

MQT_BENCH_VERSION = "2.3.0"
# MQT Bench assigns uniform random angles from numpy's default_rng(10).
MQT_PARAMETER_SEED = 10
TRANSPILE_SEED = 17
LOWERED_BASIS = ["h", "s", "sdg", "t", "tdg", "x", "z", "cx", "rz"]
# Dense operators are exact checks; statevectors sample a few input states.
DENSE_QUBITS = 10
STATEVECTOR_QUBITS = 20
_DYNAMIC = {"measure", "reset", "if_else", "while_loop", "for_loop", "switch_case", "break_loop",
            "continue_loop", "delay", "store"}
_QUARTER_TURNS = {1: ["t"], 2: ["s"], 3: ["s", "t"], 4: ["z"], 5: ["z", "t"], 6: ["sdg"], 7: ["tdg"]}


class UnsupportedSource(ValueError):
    """An MQT Bench feature that a static Clifford+T lowering cannot preserve."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class Benchmark:
    name: str
    size: int
    tier: str  # "execute" (d=3 VM) or "compile" (layout and d=25 IR only)
    note: str
    compilers: tuple[str, ...] = ("topologiq", "topols", "spiderls")
    options: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.name}_{self.size}"


_ALL = ("topologiq", "topols", "spiderls")
_LARGE = ("topols", "spiderls")
SUITE = [
    Benchmark("half_adder", 3, "execute", "Toffoli-based half adder; exact T gates"),
    Benchmark("grover", 3, "execute", "Grover search; exact pi/4 phases"),
    Benchmark("qpeexact", 3, "execute", "exactly representable phase estimation"),
    Benchmark("full_adder", 4, "execute", "one-bit full adder; exact Toffolis"),
    Benchmark("wstate", 3, "execute", "W-state preparation; two synthesized rotations"),
    Benchmark("qft", 3, "execute", "three-qubit QFT; pi/8 phases synthesized"),
    Benchmark("qwalk", 3, "execute", "quantum walk; 42 exact T gates"),
    Benchmark("qaoa", 3, "execute", "QAOA MaxCut; MQT seed-10 angles synthesized"),
    Benchmark("cdkm_ripple_carry_adder", 6, "compile", "CDKM ripple-carry adder", _ALL),
    Benchmark("vbe_ripple_carry_adder", 7, "compile", "VBE ripple-carry adder", _ALL),
    Benchmark("qpeexact", 6, "compile", "exact phase estimation; pi/2**k phases", _ALL),
    Benchmark("qft", 6, "compile", "six-qubit QFT", _LARGE),
    Benchmark("grover", 5, "compile", "Grover search with multi-controlled phases", _LARGE),
    Benchmark("qwalk", 5, "compile", "quantum walk with multi-controlled X", _LARGE),
    Benchmark("multiplier", 8, "compile", "two-bit multiplier", _LARGE),
    Benchmark("vqe_real_amp", 6, "compile", "real-amplitude VQE ansatz; MQT seed-10 angles", _LARGE),
    Benchmark("qaoa", 8, "compile", "QAOA MaxCut on eight vertices", _LARGE),
    Benchmark("hrs_cumulative_multiplier", 9, "compile", "HRS cumulative multiplier", _LARGE),
    Benchmark("ghz_dynamic", 3, "reject", "mid-circuit measurement with feedforward", ()),
    Benchmark("iqpe", 3, "reject", "iterative phase estimation with classical control", ()),
    Benchmark("dynamic_qft", 3, "reject", "measurement-based QFT", ()),
]


@dataclass(frozen=True)
class Lowered:
    source_qasm: str
    clifford_t_qasm: str
    expected_state: np.ndarray
    summary: dict


def _quarter_turns(angle: float) -> int | None:
    turns = angle / (math.pi / 4)
    nearest = round(turns)
    return nearest % 8 if abs(turns - nearest) <= 1e-9 * max(1, abs(turns)) else None


def _synthesize(angle: float, epsilon: mp.mpf) -> list[str]:
    """Return Rz(angle) up to global phase as time-ordered Clifford+T gate names."""
    from pygridsynth import gridsynth_circuit
    from pygridsynth.quantum_gate import HGate, SGate, SXGate, TGate, WGate

    with mp.workdps(60):
        synthesized = gridsynth_circuit(mp.mpf(angle), epsilon, seed=0, up_to_phase=True)
    names = {HGate: "h", TGate: "t", SGate: "s", SXGate: "x"}
    # pygridsynth lists matrix factors; circuits apply them in reverse order.
    return [names[type(gate)] for gate in reversed(synthesized) if not isinstance(gate, WGate)]


def _projective_distance(actual: np.ndarray, expected: np.ndarray) -> float:
    overlap = np.vdot(actual, expected)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1
    return float(np.linalg.norm(phase * actual - expected, ord=2))


def _static_body(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[int]]:
    """Strip terminal measurements and barriers; reject every dynamic operation."""
    body = circuit.remove_final_measurements(inplace=False)
    measured = sorted({circuit.find_bit(qubit).index for instruction in circuit.data
                       if instruction.operation.name == "measure" for qubit in instruction.qubits})
    for instruction in body.data:
        if instruction.operation.name in _DYNAMIC or instruction.clbits:
            raise UnsupportedSource("dynamic_operation",
                                    f"{instruction.operation.name!r} is not a terminal measurement")
    if body.num_clbits:
        raise UnsupportedSource("dynamic_operation", "classical bits remain after terminal readout")
    return body, measured


def lower(benchmark: Benchmark, *, epsilon: float = 1e-2) -> Lowered:
    """Generate one pinned MQT Bench instance and lower it to Clifford+T.

    `epsilon` bounds the total spectral error over all synthesized rotations,
    shared equally among the rotations that are not multiples of pi/4.
    """
    from mqt.bench import BenchmarkLevel, get_benchmark

    if version("mqt.bench") != MQT_BENCH_VERSION or version("pygridsynth") != "2.0.0":
        raise RuntimeError(f"requires mqt.bench=={MQT_BENCH_VERSION} and pygridsynth==2.0.0")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between zero and one")
    source = get_benchmark(benchmark.name, BenchmarkLevel.ALG, benchmark.size, **benchmark.options)
    source_qasm = qasm3.dumps(source)
    summary = dict(
        benchmark=benchmark.name, circuit_size=benchmark.size, tier=benchmark.tier, note=benchmark.note,
        mqt_bench_version=MQT_BENCH_VERSION, mqt_level="alg", mqt_parameter_seed=MQT_PARAMETER_SEED,
        mqt_options=benchmark.options, source_sha256=sha256(source_qasm.encode()).hexdigest(),
        source_qubits=source.num_qubits, source_gate_counts=dict(source.count_ops()),
        qiskit_version=version("qiskit"), pygridsynth_version=version("pygridsynth"),
        transpile_seed=TRANSPILE_SEED, lowered_basis=LOWERED_BASIS,
    )
    body, measured = _static_body(source)
    summary["terminal_measurements"] = measured
    # Expand MQT's high-level gates by their own definitions: Qiskit synthesis
    # plugins may differ outside an out-of-place gate's specified domain.
    standard = get_standard_gate_name_mapping()
    while custom := {i.operation.name for i in body.data if i.operation.name not in standard}:
        body = body.decompose(gates_to_decompose=sorted(custom))
    basic = transpile(body, basis_gates=LOWERED_BASIS, optimization_level=0, seed_transpiler=TRANSPILE_SEED)
    if basic.num_qubits != body.num_qubits or basic.layout is not None and basic.layout.final_layout:
        raise RuntimeError("transpilation changed the qubit order")
    angles = [float(i.operation.params[0]) for i in basic.data if i.operation.name == "rz"]
    approximate = sorted({a for a in angles if _quarter_turns(a) is None})
    count = sum(_quarter_turns(a) is None for a in angles)
    per_rotation = mp.mpf(epsilon) / max(1, count)
    words = {angle: _synthesize(angle, per_rotation) for angle in approximate}
    lowered = QuantumCircuit(body.num_qubits)
    for instruction in basic.data:
        name = instruction.operation.name
        wires = [basic.find_bit(qubit).index for qubit in instruction.qubits]
        if name != "rz":
            lowered.append(instruction.operation, wires)
            continue
        angle = float(instruction.operation.params[0])
        turns = _quarter_turns(angle)
        for gate in _QUARTER_TURNS.get(turns, []) if turns is not None else words[angle]:
            getattr(lowered, gate)(wires[0])
    clifford_t_qasm = qasm2.dumps(lowered)
    counts = dict(lowered.count_ops())
    bound = float(per_rotation * count)
    summary.update(
        logical_qubits=lowered.num_qubits, lowered_gate_counts=counts,
        t_count=counts.get("t", 0) + counts.get("tdg", 0), rz_rotations=len(angles),
        exact_rotations=len(angles) - count, synthesized_rotations=count,
        per_rotation_epsilon=float(per_rotation) if count else 0.0,
        approximation_bound=bound, exact_lowering=count == 0,
        clifford_t_sha256=sha256((clifford_t_qasm + "\n").encode()).hexdigest(),
    )
    expected = None
    if lowered.num_qubits <= DENSE_QUBITS:
        error = _projective_distance(Operator(lowered).data, Operator(body).data)
        summary.update(lowering_check="dense operator", lowering_error=error)
        expected = Statevector(lowered).data
    elif lowered.num_qubits <= STATEVECTOR_QUBITS:
        rng = np.random.default_rng(TRANSPILE_SEED)
        error = 0.0
        for basis in [0, *rng.integers(1 << lowered.num_qubits, size=3)]:
            start = Statevector.from_int(int(basis), 1 << lowered.num_qubits)
            error = max(error, _projective_distance(start.evolve(lowered).data, start.evolve(body).data))
        summary.update(lowering_check="statevector on 4 basis inputs", lowering_error=error)
    else:
        summary.update(lowering_check="per-rotation synthesis bound only", lowering_error=None)
    if summary["lowering_error"] is not None and summary["lowering_error"] > bound + 1e-8:
        raise RuntimeError(f"{benchmark.slug}: lowering error {summary['lowering_error']} exceeds {bound}")
    return Lowered(source_qasm, clifford_t_qasm, expected, summary)
