"""Real optional upstream pipeline; no mock compiler or Clifford proxy."""

from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import bloq
from bloq.interop.topologiq import AUDITED_REVISION, TopologiqImportError, from_qasm


def qasm(gates, width=1):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}\n'


@pytest.mark.parametrize('options', [dict(gravity=-1), dict(gravity=True), dict(z_stretch=1.5)])
def test_invalid_routing_weights_reject_before_upstream(options):
    with pytest.raises(ValueError, match='nonnegative integer'):
        from_qasm(qasm('h q[0];'), **options)


def test_compact_qft_profile_reaches_validated_ir():
    require_upstream()
    fixture = Path(__file__).resolve().parents[3] / 'bloq_vm/tests/data/topologiq/qft2.qasm'
    imported = from_qasm(fixture.read_text(), gravity=1, z_stretch=1)
    assert imported.graph.block_count < 74
    program = bloq.compile(imported.graph, distance=3, prepare_t_with_mpps=True, validate=True)
    assert {output.port for output in program.logical_outputs()} == set(imported.output_ports)


@pytest.mark.parametrize("operation", ["measure q[0] -> c[0];", "reset q[0];", "if(c==1) x q[0];"])
def test_classical_operations_reject_before_upstream_ingestion(operation):
    with pytest.raises(TopologiqImportError, match="classical_control"):
        from_qasm(qasm(operation))


def test_other_qasm_versions_reject_before_upstream_ingestion():
    with pytest.raises(TopologiqImportError, match="qasm_version"):
        from_qasm('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nh q[0];')


def require_upstream():
    pytest.importorskip("topologiq", reason="opt-in pinned upstream integration environment")
    pytest.importorskip("pyzx")


def test_frontend_error_preserves_public_error_type_and_message(monkeypatch):
    require_upstream()
    from bloq.interop import _qasm

    monkeypatch.setattr(_qasm, "version", lambda _: "0.0.0")
    with pytest.raises(TopologiqImportError) as caught:
        from_qasm(qasm("h q[0];"))
    assert caught.value.code == "pyzx_version"
    assert str(caught.value) == "pyzx_version: Exact QASM parsing requires pyzx==0.10.7"
    assert isinstance(caught.value.__cause__, _qasm.QASMImportError)


@pytest.mark.parametrize(
    "gates,width,input_state,expected",
    [
        ("h q[0];", 1, "plus", [(0, 0, 1)]),
        ("cx q[0],q[1];", 2, "plus", [(1, 0, 0), (1, 0, 0)]),
        ("t q[0];", 1, "plus", [(math.sqrt(0.5), math.sqrt(0.5), 0)]),
        ("tdg q[0];", 1, "plus", [(math.sqrt(0.5), -math.sqrt(0.5), 0)]),
        ("s q[0];", 1, "plus", [(0, 1, 0)]),
        ("sdg q[0];", 1, "plus", [(0, -1, 0)]),
        ("x q[0];", 1, "zero", [(0, 0, -1)]),
        ("z q[0];", 1, "plus", [(-1, 0, 0)]),
        ("h q[0]; s q[0]; h q[0];", 1, "zero", [(0, -1, 0)]),
        ("h q[0]; sdg q[0]; h q[0];", 1, "zero", [(0, 1, 0)]),
        ("rx(pi/2) q[0];", 1, "zero", [(0, -1, 0)]),
        ("rx(-pi/2) q[0];", 1, "zero", [(0, 1, 0)]),
    ],
)
def test_algorithm_to_ir_and_vm(gates, width, input_state, expected, tmp_path):
    require_upstream()
    source = qasm(gates, width)
    imported = from_qasm(source)
    assert imported.source_qasm == source
    assert len(imported.input_ports) == len(imported.output_ports) == width
    imported.graph.validate()
    path = tmp_path / "imported.blog"
    imported.graph.save(path)
    saved = bloq.BlockGraph.load(path)
    program = bloq.compile(saved, distance=3, prepare_t_with_mpps=True, validate=True)
    # Bloq orders IR ports by coordinates; source wire order is explicit.
    indices = {output.port: i for i, output in enumerate(program.logical_outputs())}
    assert set(indices) == set(imported.output_ports)
    vm = bloq.lower_vm(program, decoder_latency_rounds=0)
    for seed in (17, 34):
        shot = vm.run(seed=seed, input_state=input_state, decoder_acceptance=1,
                      accepted_accuracy=1, rejected_accuracy=1)
        assert not shot.discarded
        for port, ideal in zip(imported.output_ports, expected):
            assert shot.logical_bloch(output=indices[port]) == pytest.approx(ideal, abs=1e-8)


def test_unsupported_rotation_is_not_dropped():
    require_upstream()
    with pytest.raises(TopologiqImportError, match="phase"):
        from_qasm(qasm("rz(pi/8) q[0];"))


def test_native_routing_failure_retains_its_cause_and_failed_nodes(monkeypatch):
    require_upstream()
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    failure = ValueError("ERROR. No path found for STANDARD edge: 11 --> 12")

    def fail_placement(manager):
        manager.curr_src_id, manager.curr_tgt_id = 11, 12
        raise failure

    monkeypatch.setattr(BlockGraphManager, "build", fail_placement)
    with pytest.raises(TopologiqImportError, match="No path found") as caught:
        from_qasm(qasm("h q[0];"))
    assert caught.value.code == "upstream_layout"
    assert caught.value.nodes == (11, 12)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("width,gates", [(1, ""), (2, "h q[0];")])
def test_idle_wire_flow_failure_has_a_public_diagnostic(width, gates):
    require_upstream()
    with pytest.raises(TopologiqImportError) as caught:
        from_qasm(qasm(gates, width))
    assert caught.value.code == "upstream_flow"
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("interface", ["inputs", "outputs"])
def test_routing_cannot_reorder_source_ports(interface, monkeypatch):
    require_upstream()
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    build = BlockGraphManager.build

    def reorder_after_build(manager):
        build(manager)
        setattr(manager, interface, list(reversed(getattr(manager, interface))))

    monkeypatch.setattr(BlockGraphManager, "build", reorder_after_build)
    with pytest.raises(TopologiqImportError, match="boundaries"):
        from_qasm(qasm("cx q[0],q[1];", 2))


@pytest.mark.parametrize("angle", ["pi+1", "1e-1000", "0.2500000000000000000001*pi"])
def test_source_angles_cannot_round_to_a_supported_gate(angle, monkeypatch):
    require_upstream()
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    def no_placement(*args, **kwargs):
        raise AssertionError("invalid angles must fail before placement")

    monkeypatch.setattr(BlockGraphManager, "build", no_placement)
    with pytest.raises(TopologiqImportError, match=r"unsupported_(angle|phase)"):
        from_qasm(qasm(f"rz({angle}) q[0];"))


@pytest.mark.parametrize("bounds,z,accepted", [({"min": 0}, 0, False), ({"min": 0}, 1, True), ({"max": 0}, 0, False), ({"max": 0}, -1, True)])
def test_upstream_zero_time_bound_is_enforced(bounds, z, accepted):
    require_upstream()
    from topologiq.core.blocks import ZXBlockRegistry
    from topologiq.core.pathfinder.pathfinder import PathFinderManager

    block = ZXBlockRegistry.get_create(kind="XZZ")
    target = ((0, 0, z), block)
    search = object.__new__(PathFinderManager)
    search.s = SimpleNamespace(z_bounds=bounds, _kwargs={})
    search.tgt_zx_block = block
    search.tent_tgt_kinds = {"XZZ"}
    search.path = {target: [target]}
    search.valid_paths = {}
    search.tgts_to_fill = 1
    assert search._check_for_success(target) is accepted


def test_import_namespace_does_not_eagerly_load_upstream():
    # A fresh interpreter avoids optional submodules cached by other tests.
    source = qasm("h q[0];")
    script = f'''
import sys
sys.modules["topologiq"] = None
sys.modules["pyzx"] = None
import bloq
from bloq.interop.topologiq import from_qasm
assert "interop" in bloq.__all__
try:
    from_qasm({source!r})
except ImportError as error:
    assert "pinned Topologiq" in str(error)
else:
    raise AssertionError("missing optional dependencies must have a diagnostic")
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_committed_fixture_provenance_is_current():
    root = Path(__file__).resolve().parents[3]
    patch_hash = sha256((root / "bloq_py/examples/topologiq.patch").read_bytes()).hexdigest()
    fixtures = tuple((root / "bloq_vm/tests/data/topologiq").rglob("*.json"))
    assert fixtures
    for path in fixtures:
        provenance = json.loads(path.read_text())
        assert provenance["upstream"]["revision"] == AUDITED_REVISION
        assert provenance["upstream"]["patch_sha256"] == patch_hash
