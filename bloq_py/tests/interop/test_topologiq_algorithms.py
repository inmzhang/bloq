"""Algorithm semantics and reproducible experiment artifacts; VM checks live in Rust."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qiskit", reason="opt-in algorithm environment")
EXAMPLE = Path(__file__).parents[2] / "examples/topologiq_algorithms.py"
NAMES = ["qpe4", "simon6", "grover3", "qaoa3", "qaoa4", "qaoa5", "adder4", "walk3", "walk5",
         "qae3", "vqe4", "ising4", "ising5", "iqp6"]


@pytest.fixture
def algorithms(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    return runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize("name", NAMES)
def test_candidate_checks_exported_channels_and_has_non_stabilizer_output(algorithms, name):
    pytest.importorskip("pyzx", reason="opt-in exact QASM frontend")
    from bloq.interop._qasm import parse_qasm

    result = algorithms["generate_candidate"](name)
    # Generation checks both exported channels against the algorithm matrix.
    assert result.summary["projective_spectral_error"] < 1e-12
    assert result.summary["non_stabilizer_output"]
    assert result.summary["t_count"] > 0
    assert np.vdot(result.ideal_unitary[:, 0], result.ideal_unitary[:, 0]) == pytest.approx(1)
    # The actual adapter frontend must accept the exact Clifford+T artifact.
    assert parse_qasm(result.clifford_t_qasm).qubits == result.summary["qubits"]


def test_grover_amplifies_only_the_marked_word(algorithms):
    result = algorithms["generate_candidate"]("grover3")
    probabilities = abs(result.ideal_unitary[:, 0])**2
    expected = np.full(8, 1/32)
    expected[5] = 25/32
    assert probabilities == pytest.approx(expected, abs=1e-12)


def test_modular_addition_preserves_coherent_integer_pairs(algorithms):
    result = algorithms["generate_candidate"]("adder4")
    expected = np.zeros(16)
    for a in range(4):
        for b in range(2):
            expected[a + 4*((a+b) % 4)] = 1/np.sqrt(8)
    assert result.ideal_unitary[:, 0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name,cut", [("qaoa3", 1.875), ("qaoa4", 3.0), ("qaoa5", 3.75)])
def test_qaoa_cost_exceeds_uniform_cut_baseline(algorithms, name, cut):
    result = algorithms["generate_candidate"](name)
    assert result.summary["expected_cut"] == pytest.approx(cut, abs=1e-12)
    assert cut > len(result.summary["edges"])/2


def test_simon_recovers_a_unique_period_from_the_fourier_support(algorithms):
    result = algorithms["generate_candidate"]("simon6")
    probabilities = abs(result.ideal_unitary[:, 0])**2
    marginal = probabilities.reshape(8, 8).sum(axis=0)
    assert marginal == pytest.approx([1/4, 0, 1/4, 0, 0, 1/4, 0, 1/4], abs=1e-12)
    rows = np.flatnonzero(marginal > 1e-12)
    periods = [s for s in range(1, 8) if all((s & int(y)).bit_count() % 2 == 0 for y in rows)]
    assert periods == [0b101]
    assert result.summary["t_count"] == 7


def test_iqp_has_the_independent_zero_probability_and_even_parity(algorithms):
    result = algorithms["generate_candidate"]("iqp6")
    probabilities = abs(result.ideal_unitary[:, 0])**2
    assert probabilities[0] == pytest.approx((3 + 2*np.sqrt(2))/16, abs=1e-12)
    assert sum(p for x, p in enumerate(probabilities) if x.bit_count() % 2) == pytest.approx(0, abs=1e-12)
    assert result.summary["t_count"] == 8


@pytest.mark.parametrize("name", ["simon6", "qaoa5", "iqp6"])
def test_algorithm_results_reject_a_wrong_distribution(algorithms, name):
    from _algorithm_candidates import check_algorithm_probabilities

    result = algorithms["generate_candidate"](name)
    probabilities = abs(result.ideal_unitary[:, 0])**2
    assert check_algorithm_probabilities(result.summary, probabilities)
    with pytest.raises(RuntimeError):
        check_algorithm_probabilities(result.summary, np.full(len(probabilities), 1/len(probabilities)))


def test_amplitude_estimation_matches_finite_resolution_kernel(algorithms):
    from qiskit.quantum_info import Statevector

    result = algorithms["generate_candidate"]("qae3")
    expected = [sum(abs(sum(np.exp(2j*np.pi*k*(phase-y/4)) for k in range(4))/4)**2
                    for phase in (1/8, -1/8))/2 for y in range(4)]
    marginal = Statevector(result.ideal_unitary[:, 0].copy()).probabilities([0, 1])
    assert marginal == pytest.approx(expected, abs=1e-12)


def test_cli_retains_failure_and_continues_to_a_supported_compilation(algorithms, tmp_path, monkeypatch):
    pytest.importorskip("topologiq", reason="opt-in pinned upstream environment")
    import bloq

    command = [sys.executable, str(EXAMPLE), "--candidates", "qaoa3", "walk3",
               "--compile-only", "--output-dir", str(tmp_path)]
    run = subprocess.run

    def candidate_run(command, **kwargs):
        if command[command.index("--candidates") + 1] == "qaoa3":
            output = Path(command[command.index("--output-dir") + 1])
            (output / "result.json").write_text(json.dumps(dict(
                candidate="qaoa3", status="failed", failed_stage="layout",
                error_code="injected_worker_failure", error_nodes=[11, 12],
                error="simulated candidate worker failure",
            )), encoding="utf-8")
            kwargs["stdout"].write("simulated candidate worker failure\n")
            return subprocess.CompletedProcess(command, 1)
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", candidate_run)
    monkeypatch.setattr(sys, "argv", command[1:])
    with pytest.raises(SystemExit) as completed:
        algorithms["main"]()
    assert completed.value.code == 1
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    failed, success = comparison["candidates"]
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "layout"
    assert failed["error_code"] == "injected_worker_failure"
    assert failed["error_nodes"] == [11, 12]
    assert "simulated candidate worker failure" in (tmp_path / failed["log"]).read_text()
    assert success["status"] == "compiled"
    assert success["t_preparation"] == "ideal_t_proxy"
    assert success["layout_options"]["temporal_hadamards_only"]
    assert success["hadamard_pipes_by_axis"]["X"] == success["hadamard_pipes_by_axis"]["Y"] == 0
    assert success["hadamard_pipes_by_axis"]["Z"] > 0
    artifact = tmp_path / "walk3"
    assert success["clifford_t_sha256"] == sha256((artifact / "clifford_t.qasm").read_bytes()).hexdigest()
    bloq.Bloq.load(artifact / "program.bloq").validate()
    repeated = run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "must be empty" in repeated.stderr


def test_timeout_is_recorded_with_the_interrupted_stage(tmp_path):
    command = [sys.executable, str(EXAMPLE), "--candidates", "walk3", "--timeout", "0.000001",
               "--synthesis-only", "--output-dir", str(tmp_path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 1
    result, = json.loads((tmp_path / "comparison.json").read_text())["candidates"]
    assert result["status"] == "timeout"
    assert result["failed_stage"] == "synthesis"


def test_interrupted_checkpoint_keeps_the_previous_valid_result(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    save = runpy.run_path(str(EXAMPLE.with_name("_interop_demo.py")))["save_json"]
    result = tmp_path / "result.json"
    save(result, dict(stage="layout"))
    write_text = Path.write_text

    def interrupted(path, text, **kwargs):
        write_text(path, text[:5], **kwargs)
        raise OSError("simulated interruption during checkpoint")

    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        save(result, dict(stage="compile"))
    assert json.loads(result.read_text()) == dict(stage="layout")
