"""Independent matrix and artifact checks for the optional algorithm demo."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qiskit", reason="opt-in algorithm synthesis environment")
pytest.importorskip("pygridsynth", reason="opt-in algorithm synthesis environment")

EXAMPLE = Path(__file__).parents[2] / "examples/topologiq_qft.py"


def test_four_qubit_qft_qasm_has_the_fourier_channel():
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    synthesis = runpy.run_path(str(EXAMPLE))["generate_qft"](4, 0.01)
    rows, columns = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    fourier = np.exp(2j * np.pi * rows * columns / 16) / 4
    source = Operator(qasm2.loads(synthesis.source_qasm)).data
    compiled = Operator(qasm2.loads(synthesis.synthesized_qasm)).data
    assert np.allclose(source, fourier, atol=1e-12, rtol=0)
    phase = np.vdot(compiled, fourier)
    error = np.linalg.norm(phase / abs(phase) * compiled - fourier, ord=2)
    assert error == pytest.approx(synthesis.projective_error, abs=1e-12)
    assert error < 0.01
    assert synthesis.t_count > 3
    assert synthesis.gate_counts["tdg"] > 0


def test_synthesis_cli_retains_preparation_and_valid_qasm(tmp_path):
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    subprocess.run([sys.executable, str(EXAMPLE), "--qubits", "2", "--synthesis-only",
                    "--output-dir", str(tmp_path)], check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "result.json").read_text())
    assert summary["status"] == "synthesized"
    assert summary["input_basis_index"] == 1
    source = Operator(qasm2.loads((tmp_path / "source.qasm").read_text())).data
    compiled = Operator(qasm2.loads((tmp_path / "clifford_t.qasm").read_text())).data
    phase = np.vdot(compiled, source)
    assert np.linalg.norm(phase / abs(phase) * compiled - source, ord=2) < 1e-12
    assert np.allclose(source[:, 0], np.exp(2j * np.pi * np.arange(4) / 4) / 2)
    repeated = subprocess.run([sys.executable, str(EXAMPLE), "--synthesis-only",
                               "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "must be empty" in repeated.stderr


@pytest.mark.parametrize("compile_only", [False, True])
def test_cli_retains_valid_ir_and_optional_execution(tmp_path, compile_only):
    pytest.importorskip("topologiq", reason="opt-in pinned upstream environment")
    import bloq

    command = [sys.executable, str(EXAMPLE), "--qubits", "2", "--shots", "2",
               "--output-dir", str(tmp_path)]
    if compile_only:
        command.append("--compile-only")
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "result.json").read_text())
    assert summary["status"] == ("compiled" if compile_only else "executed")
    assert summary["t_count"] == 3
    assert summary["t_preparation"] == "ideal_t_proxy"
    if not compile_only:
        assert summary["ideal_boundary_capture"]
        assert summary["discarded_shots"] == 0
        assert summary["peak_rank"] > 1
        assert summary["ideal_qft_state_fidelity"] == pytest.approx(1, abs=1e-8)
        assert summary["vm_verification"].startswith("all 15 joint Pauli expectations")
        assert len(summary["shots"]) == 2
        for shot in summary["shots"]:
            assert shot["joint_state_fidelity"] == pytest.approx(1, abs=1e-8)
    program = bloq.Bloq.load(tmp_path / "program.bloq")
    program.validate()
    assert program.stats().node_counts["RepeatUntilSuccess"] == 0
    bloq.BlockGraph.load(tmp_path / "layout.blog").validate()
