"""The shared runner checks the complete source interface and output ordering."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qiskit", reason="opt-in algorithm environment")
import bloq  # noqa: E402

PORTS = ((0, 0, 2), (2, 0, 2))


@pytest.fixture
def demo(monkeypatch):
    run = runpy.run_path(str(Path(__file__).parents[2] / "examples/_interop_demo.py"))["run_demo"]

    def retain(path):
        path.write_text("retained artifact", encoding="utf-8")

    graph = SimpleNamespace(save=retain, export_html_viewer=retain,
                            flatten=lambda: SimpleNamespace(block_count=6, pipes=lambda: []))
    imported = SimpleNamespace(graph=graph, input_ports=PORTS, output_ports=PORTS,
                               layout_options={}, upstream_revision="test")
    adapter = SimpleNamespace(AUDITED_REVISION="test", from_qasm=lambda *args, **kwargs: imported)
    program = SimpleNamespace(save=retain, qubit_count=2, stats=lambda: SimpleNamespace(node_count=2),
                              logical_outputs=lambda: [SimpleNamespace(port=p) for p in reversed(PORTS)])
    monkeypatch.setitem(run.__globals__, "import_module", lambda _: adapter)
    monkeypatch.setattr(bloq, "compile", lambda *args, **kwargs: program)
    return run, program


@pytest.mark.parametrize("ports", [
    (PORTS[0],),
    (PORTS[0], PORTS[0]),
    (PORTS[0], PORTS[1], PORTS[1]),
    (PORTS[0], (4, 0, 2)),
])
def test_compile_only_rejects_missing_duplicate_or_unbound_outputs(demo, ports, tmp_path):
    run, program = demo
    program.logical_outputs = lambda: [SimpleNamespace(port=p) for p in ports]
    with pytest.raises(RuntimeError, match="output interface differs"):
        run("source", "compiled", None, tmp_path, {}, compiler="spiderls", compile_only=True)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == result["failure_layer"] == "compile"
    assert (tmp_path / "layout.blog").is_file()
    assert (tmp_path / "program.bloq").is_file()


def test_source_output_order_controls_expectations_and_report(demo, tmp_path, monkeypatch):
    run, _ = demo
    # Source |01> is |10> in the reversed VM interface. Only ZI, IZ and ZZ
    # have nonzero expectations, with signs +, -, - in VM wire order.
    signature = [0.0] * 15
    signature[2], signature[11], signature[14] = 1.0, -1.0, -1.0
    execution = dict(shots=1, discarded=0, all_detectors_constant=True, peak_rank=1,
                     samples=[dict(shot=0, logical_signature=signature,
                                   logical_bloch=[[0, 0, 1], [0, 0, -1]], captured_output_ports=[])])
    monkeypatch.setattr(bloq, "run_noiseless", lambda *args, **kwargs: execution)
    ideal = np.eye(4)[:, [1, 0, 3, 2]]
    result = run("source", "compiled", ideal, tmp_path, {}, compiler="spiderls", shots=1)
    assert result["status"] == "executed"
    assert result["shots"][0]["joint_state_fidelity"] == pytest.approx(1)
    assert result["shots"][0]["logical_bloch"] == [[0, 0, -1], [0, 0, 1]]
    assert json.loads((tmp_path / "execution.json").read_text()) == execution
