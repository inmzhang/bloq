"""MQT circuit provenance and faithful static Clifford+T adaptation."""

from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mqt.bench", reason="opt-in pinned benchmark environment")
pytest.importorskip("pygridsynth")
from qiskit import QuantumCircuit, qasm2, qasm3  # noqa: E402
from qiskit.circuit import Gate  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    return import_module("_mqt_bench")


def _source(monkeypatch, circuit):
    monkeypatch.setattr("mqt.bench.get_benchmark", lambda *args, **kwargs: circuit)


@pytest.mark.parametrize("name,exact", [("half_adder", True), ("wstate", False)])
def test_pinned_mqt_artifacts_and_independent_source_state(bench, name, exact):
    result = bench.lower(bench.Benchmark(name, 3, "execute", "regression"))
    original = qasm3.loads(result.source_qasm).remove_final_measurements(inplace=False)
    artifact = qasm2.loads(result.clifford_t_qasm)
    assert result.summary["exact_lowering"] is exact
    assert result.summary["source_sha256"] == sha256(result.source_qasm.encode()).hexdigest()
    assert result.summary["clifford_t_sha256"] == sha256((result.clifford_t_qasm + "\n").encode()).hexdigest()
    assert result.summary["lowering_error"] <= result.summary["approximation_bound"] + 1e-12
    assert bench._projective_distance(Operator(artifact).data, Operator(original).data) <= 1e-2
    assert result.expected_state == pytest.approx(Statevector(artifact).data)
    assert result.source_state == pytest.approx(Statevector(original).data)
    if not exact:
        assert bench._projective_distance(result.expected_state, result.source_state) > 1e-8


def test_terminal_readout_keeps_classical_mapping_and_order(bench, monkeypatch):
    source = QuantumCircuit(3, 3)
    source.h(0)
    source.measure(2, 0)
    source.measure(0, 2)
    _source(monkeypatch, source)
    result = bench.lower(bench.Benchmark("mapped_readout", 3, "execute", ""))
    assert result.summary["terminal_measurements"] == [
        dict(qubit=2, clbit=0, basis="Z"), dict(qubit=0, clbit=2, basis="Z")]


@pytest.mark.parametrize("name", ["ghz_dynamic", "iqpe", "dynamic_qft"])
def test_dynamic_mqt_sources_reject_and_retain_provenance(bench, name):
    with pytest.raises(bench.UnsupportedSource) as rejected:
        bench.lower(bench.Benchmark(name, 3, "reject", ""))
    assert rejected.value.code == "dynamic_operation"
    assert "measure" in rejected.value.source_qasm
    assert rejected.value.summary["source_sha256"] == sha256(rejected.value.source_qasm.encode()).hexdigest()


def test_near_quarter_turn_is_synthesized_without_rounding(bench, monkeypatch):
    source = QuantumCircuit(1)
    source.h(0)
    source.rz(np.pi / 4 + 1e-10, 0)
    _source(monkeypatch, source)
    result = bench.lower(bench.Benchmark("near_t", 1, "execute", ""), epsilon=1e-4)
    assert not result.summary["exact_lowering"]
    assert result.summary["synthesized_rotations"] == 1
    angle = np.pi / 4 + 1e-10
    assert result.summary["source_gate_parameters"] == [dict(gate="rz", values=[angle], count=1)]
    assert result.summary["rz_angle_counts"] == [dict(radians=angle, count=1, synthesized=True)]
    assert result.summary["lowering_error"] <= 1e-4


def test_synthesis_precision_tracks_small_error_budgets(bench, monkeypatch):
    def synthesize(angle, epsilon, **kwargs):
        assert bench.mp.mp.dps >= 270
        assert abs(angle - bench.mp.mpf(0.123)) < epsilon
        return []

    monkeypatch.setattr("pygridsynth.gridsynth_circuit", synthesize)
    with bench.mp.workdps(100):
        assert bench._synthesize(0.123, bench.mp.mpf("1e-80")) == []


def test_recursive_definition_expansion_is_bounded(bench):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(65):
        nested = QuantumCircuit(1)
        nested.append(circuit.to_gate(), [0])
        circuit = nested
    with pytest.raises(bench.UnsupportedSource, match="definition_depth"):
        bench._expand_definitions(circuit)


def test_opaque_gate_has_an_explicit_rejection(bench):
    circuit = QuantumCircuit(1)
    circuit.append(Gate("opaque", 1, []), [0])
    with pytest.raises(bench.UnsupportedSource, match="opaque_gate"):
        bench._expand_definitions(circuit)


@pytest.mark.parametrize("export", ["qasm2", "qasm3"])
def test_export_corruption_is_detected(bench, monkeypatch, export):
    source = QuantumCircuit(1)
    source.h(0)
    _source(monkeypatch, source)
    module = getattr(bench, export)
    dumps = module.dumps

    def corrupt(circuit, **kwargs):
        damaged = circuit.copy()
        damaged.x(0)
        return dumps(damaged, **kwargs)

    monkeypatch.setattr(module, "dumps", corrupt)
    with pytest.raises(RuntimeError, match="source QASM export changed|lowering error"):
        bench.lower(bench.Benchmark("corrupted", 1, "execute", ""))


def test_coherent_adder_retains_mqt_program_and_verifies_explicit_input_wrapper(bench):
    baseline = bench.lower(bench.Benchmark("full_adder", 4, "execute", ""))
    prepared = bench.lower(next(b for b in bench.SUITE if b.slug == "full_adder_4_coherent"))
    assert prepared.benchmark_qasm == baseline.source_qasm
    assert prepared.summary["benchmark_sha256"] == baseline.summary["source_sha256"]
    assert prepared.summary["source_sha256"] != prepared.summary["benchmark_sha256"]
    assert prepared.summary["source_variant"] == "explicit_input_preparation"
    assert prepared.summary["input_preparation"] == dict(h_qubits=[0, 1, 2])
    algorithm = qasm3.loads(prepared.benchmark_qasm).remove_final_measurements(inplace=False)
    wrapper = QuantumCircuit(4)
    wrapper.h([0, 1, 2])
    expected = Operator(algorithm).data @ Operator(wrapper).data
    augmented = qasm3.loads(prepared.source_qasm).remove_final_measurements(inplace=False)
    assert bench._projective_distance(Operator(augmented).data, expected) < 1e-12
    assert prepared.summary["t_count"] == baseline.summary["t_count"] == 14
    assert not prepared.summary["channel_clifford"]
    assert baseline.summary["output_stabilizer"]
    assert not prepared.summary["output_stabilizer"]
    assert prepared.summary["output_pauli_stabilizer_count"] == 2
    import json
    json.dumps(prepared.summary)


def test_unmodified_entangled_qft_has_a_nonstabilizer_output(bench):
    result = bench.lower(next(b for b in bench.SUITE if b.slug == "qftentangled_2"))
    assert result.benchmark_qasm is None
    assert result.summary["source_variant"] == "unmodified_mqt_benchmark"
    assert result.summary["source_sha256"] == result.summary["benchmark_sha256"]
    assert result.summary["t_count"] == 3
    assert not result.summary["channel_clifford"]
    assert not result.summary["output_stabilizer"]
    assert abs(result.source_state)**2 == pytest.approx([1/2, 1/4, 0, 1/4], abs=1e-12)



def test_mqt_amplitude_estimation_uses_exact_definition_export(bench):
    from mqt.bench import BenchmarkLevel, get_benchmark

    original = get_benchmark("ae", BenchmarkLevel.ALG, 2)
    reference, readout = bench._static_body(original)
    result = bench.lower(bench.Benchmark("ae", 2, "compile", ""))
    exported, actual_readout = bench._static_body(qasm3.loads(result.source_qasm))
    assert result.summary["source_serialization"] == "definition_expanded"
    assert result.benchmark_qasm is None
    assert result.summary["benchmark_sha256"] == sha256(result.source_qasm.encode()).hexdigest()
    assert result.summary["source_sha256"] == sha256(result.source_qasm.encode()).hexdigest()
    assert readout == actual_readout == result.summary["terminal_measurements"]
    assert bench._projective_distance(Operator(exported).data, Operator(reference).data) < 1e-12
    assert result.source_state == pytest.approx(Statevector(reference).data)
    assert result.summary["lowering_error"] <= result.summary["approximation_bound"]
    assert not result.summary["output_stabilizer"]


def test_permutation_export_preserves_complete_channel_and_readout(bench, monkeypatch):
    from qiskit.circuit.library import PermutationGate

    source = QuantumCircuit(3, 2)
    source.x(0)
    source.append(PermutationGate([2, 0, 1]), range(3))
    source.measure(2, 0)
    source.measure(0, 1)
    _source(monkeypatch, source)
    result = bench.lower(bench.Benchmark("permutation", 3, "execute", ""))
    reference = source.remove_final_measurements(inplace=False)
    artifact = qasm2.loads(result.clifford_t_qasm)
    assert bench._projective_distance(Operator(artifact).data, Operator(reference).data) < 1e-12
    assert np.argmax(Statevector(artifact).probabilities()) == 2
    assert result.summary["exact_lowering"]
    assert result.summary["terminal_measurements"] == [
        dict(qubit=2, clbit=0, basis="Z"), dict(qubit=0, clbit=1, basis="Z")]
    assert isinstance(source.data[1].operation, PermutationGate)
