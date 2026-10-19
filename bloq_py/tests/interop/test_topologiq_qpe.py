"""Independent algorithm and non-Clifford output checks for the QPE example."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qiskit", reason="opt-in algorithm environment")

EXAMPLE = Path(__file__).parents[2] / "examples/topologiq_qpe.py"


def test_four_qubit_phase_estimation_preserves_coherent_eigenphases():
    from qiskit import qasm2
    from qiskit.quantum_info import Operator, Statevector

    result = runpy.run_path(str(EXAMPLE))["generate_phase_estimation"]()
    actual = Operator(qasm2.loads(result.clifford_t_qasm)).data
    assert actual.shape == (16, 16)
    overlap = np.vdot(actual, result.ideal_unitary)
    assert np.linalg.norm(overlap / abs(overlap) * actual - result.ideal_unitary, ord=2) < 1e-12
    assert result.gate_counts["t"] + result.gate_counts["tdg"] == 7
    expected = np.zeros(16)
    expected[[0, 5, 8, 15]] = 0.5
    state = actual[:, 0]
    phase = np.vdot(state, expected)
    assert np.allclose(phase / abs(phase) * state, expected, atol=1e-12, rtol=0)
    assert Statevector(state).probabilities([0, 1]) == pytest.approx([0.5, 0.25, 0, 0.25], abs=1e-12)
    # A stabilizer state's computational support is affine. This four-element
    # support fails closure under XOR, independently witnessing non-Clifford output.
    support = {i for i, amplitude in enumerate(state) if abs(amplitude) > 1e-12}
    assert 0 in support
    assert any(a ^ b not in support for a in support for b in support)


def test_cli_runs_qpe_and_retains_exact_qasm_and_ideal_t_source_ir(tmp_path):
    pytest.importorskip("topologiq", reason="opt-in pinned upstream environment")
    import bloq

    command = [sys.executable, str(EXAMPLE), "--shots", "3", "--output-dir", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "result.json").read_text())
    assert summary["status"] == "executed"
    assert summary["t_preparation"] == "ideal_t_proxy"
    assert len(summary["shots"]) == 3
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["shots"] == summary["attempted_shots"] == 3
    assert summary["discarded_shots"] == execution["discarded"] == 0
    assert len({sample["shot"] for sample in execution["samples"]}) == 3
    assert summary["hadamard_pipes_by_axis"]["X"] == summary["hadamard_pipes_by_axis"]["Y"] == 0
    assert execution["all_detectors_constant"]
    assert summary["t_count"] == 7
    assert summary["projective_spectral_error"] < 1e-12
    assert summary["clifford_t_sha256"] == sha256((tmp_path / "clifford_t.qasm").read_bytes()).hexdigest()
    program = bloq.Bloq.load(tmp_path / "program.bloq")
    program.validate()
    assert program.stats().node_counts.get("RepeatUntilSuccess", 0) == 0
    assert program.stats().node_counts.get("Discard", 0) == 0
    assert len(summary["input_ports"]) == len(program.logical_outputs()) == 4
    bloq.BlockGraph.load(tmp_path / "layout.blog").validate()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "must be empty" in repeated.stderr
