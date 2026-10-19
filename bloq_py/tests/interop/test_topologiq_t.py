"""Certify real-upstream T readouts and reject corrupted provenance."""

import copy
from fractions import Fraction
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

pytest.importorskip("pyzx", reason="opt-in pinned upstream integration environment")
pytest.importorskip("topologiq", reason="opt-in pinned upstream integration environment")

from bloq.interop._qasm import parse_qasm  # noqa: E402
from bloq.interop._topologiq_t import lower_t_gadgets  # noqa: E402
from bloq.interop.topologiq import TopologiqImportError, _from_qasm, _routed_path, from_qasm  # noqa: E402


@pytest.mark.parametrize("corruption", ["missing", "hadamard", "branched"])
def test_twin_identity_wire_contract(corruption, monkeypatch):
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    build = BlockGraphManager.build

    def corrupt_after_build(manager):
        build(manager)
        root = next(root for root in manager.t_gates if len(manager.twin_trace[root]) > 1)
        path = _routed_path(manager, root, root)
        assert len(path) > 2
        first, router = path[:2]
        if corruption == "missing":
            manager.bgraph.remove_edge(first, router)
        elif corruption == "hadamard":
            manager.bgraph.edges[first, router]["edge_type"] = "HADAMARD"
        else:
            manager.bgraph.add_edge(router, max(manager.bgraph.nodes) + 1)

    monkeypatch.setattr(BlockGraphManager, "build", corrupt_after_build)
    source = Path(__file__).resolve().parents[3] / "bloq_vm/tests/data/topologiq/qft2_x.qasm"
    with pytest.raises(TopologiqImportError, match="t_provenance: Spider twins"):
        # Replay the historical layout containing this subdivided twin wire.
        _from_qasm(source.read_text(), seed=1330, temporal_hadamards_only=False)


@pytest.fixture(scope="module")
def temporal_qpe_layout():
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    repository = Path(__file__).resolve().parents[3]
    generate = runpy.run_path(str(repository / "bloq_py/examples/topologiq_qpe.py"))["generate_phase_estimation"]
    qasm = generate().clifford_t_qasm
    retained, build = [], BlockGraphManager.build

    def retain(manager):
        build(manager)
        retained.append(manager)

    with patch.object(BlockGraphManager, "build", retain):
        imported = from_qasm(qasm)
    manager, = retained
    return manager, parse_qasm(qasm).to_graph(), imported


def test_public_qpe_import_names_the_upstream_h_transport_record(temporal_qpe_layout):
    manager, source, imported = temporal_qpe_layout
    root = 10
    previous = next(node for node in source.neighbors(root) if source.row(node) < source.row(root))
    boundary = next(node for node in source.neighbors(previous) if node != root)
    readout_node = _routed_path(manager, previous, boundary)[1]
    assert imported.layout_options["temporal_hadamards_only"] is True
    measure = next(action for action in imported.graph.actions()
                   if action.kind == "measure" and action.name == f"topologiq_t_{root}")
    cap = manager.t_zx_tracker[root]
    assert not measure.target.is_edge
    assert measure.target.pos == imported.node_positions[readout_node]
    # Coordinate order is not the decoded parity's native execution deadline.
    assert measure.target.pos[2] > imported.node_positions[cap][2]


@pytest.mark.parametrize("corruption", ["web_label", "web_extra_leg", "identity_phase", "hadamard_location", "source_branch"])
def test_upstream_h_input_certificate_rejects_unproved_extensions(temporal_qpe_layout, corruption):
    original, source, _ = temporal_qpe_layout
    manager, source = copy.deepcopy(original), source.copy()
    root = 10
    previous = next(node for node in source.neighbors(root) if source.row(node) < source.row(root))
    boundary = next(node for node in source.neighbors(previous) if node != root)
    path = _routed_path(manager, previous, boundary)
    _, zwebs, _ = manager.aug_zx.pauli_pack
    if corruption == "web_label":
        zwebs[root].half_edges()[boundary, previous] = "Z"
    elif corruption == "web_extra_leg":
        following = next(node for node in source.neighbors(root) if node != previous)
        zwebs[root].half_edges()[root, following] = "Z"
    elif corruption == "identity_phase":
        source.set_phase(previous, Fraction(1, 2))
    elif corruption == "hadamard_location":
        manager.bgraph.edges[path[0], path[1]]["edge_type"] = "SIMPLE"
        manager.bgraph.edges[path[1], path[2]]["edge_type"] = "HADAMARD"
    else:
        branch = source.add_vertex(1)
        source.add_edge((previous, branch))
    manager.t_gates = {root: manager.t_gates[root]}
    manager.t_zx_tracker = {root: manager.t_zx_tracker[root]}
    with pytest.raises(TopologiqImportError, match="Direct resource has no proved input-transport readout") as caught:
        lower_t_gadgets(manager, source, {root: Fraction(1, 4)})
    assert caught.value.code == "t_control_frame"
    assert caught.value.nodes == (root,)
