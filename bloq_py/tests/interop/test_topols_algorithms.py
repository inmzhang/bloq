"""Shared algorithm artifacts and failure reporting for the TopoLS example."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("qiskit", reason="opt-in algorithm environment")
EXAMPLE = Path(__file__).parents[2] / "examples/topols_algorithms.py"


@pytest.mark.parametrize("candidate", ["adder4", "ising4", "qpe4"])
def test_cli_preserves_checked_algorithm_sources_without_upstream(candidate, tmp_path):
    command = [sys.executable, str(EXAMPLE), "--candidate", candidate,
               "--synthesis-only", "--engine", "python", "--output-dir", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["compiler"] == "topols"
    assert result["candidate"] == candidate
    assert result["status"] == "synthesized"
    assert result["projective_spectral_error"] < 1e-12
    assert result["non_stabilizer_output"]
    assert (tmp_path / "source.qasm").read_text().startswith("OPENQASM 2.0;")
    assert (tmp_path / "clifford_t.qasm").read_text().startswith("OPENQASM 2.0;")


def test_shared_runner_retains_topols_diagnostic_without_node_metadata(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    from bloq.interop import topols

    failure = topols.TopoLSImportError("resource_collision", "T column is occupied")
    options = dict(engine="python", seed=8, iterations=2, time_bound=0.01)

    def fail(*args, **kwargs):
        assert kwargs == options
        raise failure

    monkeypatch.setattr(topols, "from_qasm", fail)
    runner = runpy.run_path(str(EXAMPLE.with_name("_interop_demo.py")))["run_demo"]
    with pytest.raises(topols.TopoLSImportError) as caught:
        runner("source", "compiled", None, tmp_path, {}, compiler="topols", layout_options=options)
    assert caught.value is failure
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == "layout"
    assert result["error_code"] == "resource_collision"
    assert result["error_nodes"] == []
    assert result["layout_options"] == options


def test_compile_only_preserves_interface_with_a_t_source_before_the_input(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    pytest.importorskip("topols")
    pytest.importorskip("topols_core")
    generate = runpy.run_path(str(EXAMPLE.with_name("_algorithm_candidates.py")))["generate_candidate"]
    candidate = generate("adder4")
    runner = runpy.run_path(str(EXAMPLE.with_name("_interop_demo.py")))["run_demo"]
    runner(candidate.source_qasm, candidate.clifford_t_qasm, candidate.ideal_unitary,
           tmp_path, candidate.summary, compiler="topols", compile_only=True,
           layout_options=dict(engine="rust", seed=1))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "compiled"
    # The resource precedes the original input origin, which therefore shifts.
    assert min(p[2] for p in result["input_ports"]) > 0
    import bloq

    graph = bloq.BlockGraph.load(tmp_path / "layout.blog")
    program = bloq.Bloq.load(tmp_path / "program.bloq")
    assert min(block.pos[2] for block in graph.flatten().blocks()) == 0
    assert {output.port for output in program.logical_outputs()} == {
        tuple(port) for port in result["output_ports"]
    }
