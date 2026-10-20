"""TopoLS source fidelity, layout validation, and reproducible native import.

Physical non-Clifford channel checks live in bloq_vm/tests/interop_topols.rs.
"""

import copy
from fractions import Fraction
from pathlib import Path
import runpy

import pytest

pytest.importorskip("topols", reason="optional pinned upstream integration")

from bloq.interop._qasm import parse_qasm  # noqa: E402
from bloq.interop.topols import (  # noqa: E402
    TopoLSImportError, _convert, _normalize_qasm, _pipe_diagram, from_qasm,
)


def source(gates, width=1):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}\n'


@pytest.mark.parametrize("phase,kind", [
    (Fraction(1, 4), 5), (Fraction(-1, 4), 5),
    (Fraction(1, 2), 4), (Fraction(-1, 2), 4),
])
@pytest.mark.parametrize("block", [False, True])
def test_patched_classifier_and_fusion_preserve_signed_resource_phase(phase, kind, block):
    import pyzx as zx
    from topols.zx_transform.layering import node_type_convert
    from topols.zx_transform.simplify import zx_optimization, zx_optimization_block

    graph = zx.Graph()
    left = graph.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=0)
    junction = graph.add_vertex(zx.VertexType.Z, qubit=0, row=0)
    resource = graph.add_vertex(zx.VertexType.Z, qubit=0, row=1, phase=phase)
    right = graph.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=1)
    side = graph.add_vertex(zx.VertexType.BOUNDARY, qubit=1, row=0)
    graph.add_edges([(left, junction), (junction, resource), (resource, right), (junction, side)])
    graph.set_inputs((left, side))
    graph.set_outputs((right,))
    assert node_type_convert(graph, resource) == kind
    original = graph.copy()
    if block:
        zx_optimization_block(graph, [0, 1])
    else:
        zx_optimization(graph, None)
    assert graph.num_vertices() == original.num_vertices() - 1
    assert [graph.phase(v) for v in graph.vertices() if graph.phase(v)] == [phase % 2]
    assert zx.compare_tensors(original, graph, preserve_scalar=True)


def test_fixture_exporter_rejects_an_unrelated_upstream_checkout(tmp_path):
    script = Path(__file__).with_name("export_topols_fixtures.py")
    check = runpy.run_path(str(script))["checked_upstream"]
    with pytest.raises(RuntimeError, match="does not come from"):
        check(Path(__file__).resolve().parents[3], tmp_path)


def test_s_measurement_stub_cannot_reenter_a_finished_time_layer():
    from topols.embedding.hadamard import HTable
    from topols.embedding.state import EmbeddingState

    state = EmbeddingState(
        embed_node_pos={14: (1, 2, 6)}, embed_node_ori={14: "i"},
        embed_node_type={14: 0}, embed_path=(), occupied={(1, 2, 6)},
        z_floor=5, x_min_floor=-10, x_max_floor=10, y_min_floor=-10, y_max_floor=10,
        idle_h_track={}, idle_place={}, t_track={}, node_type={17: 4},
        input_connect={17: [14]}, inter_connect=(), output_connect={17: 1},
        order=[17], z_length=1, hadamard_edges=HTable(),
    )
    # The downward incoming wire would put the stub endpoint at z=4.
    assert state.next_state((1, 2, 5)) is None
    legal = state.next_state((1, 2, 7))
    assert legal is not None
    assert all(p[2] >= state.z_floor for path in legal.embed_path for p in path)


@pytest.mark.parametrize("kind", [0, 1, 4, 5])
@pytest.mark.parametrize("outgoing", [1, 2])
def test_ceiling_lift_preserves_branching_spider_color(kind, outgoing):
    from topols.embedding.hadamard import HTable
    from topols.embedding.state import EmbeddingState

    state = EmbeddingState(
        embed_node_pos={0: (0, -1, 0)}, embed_node_ori={0: "j"},
        embed_node_type={0: kind}, embed_path=(), occupied={(0, -1, 0)},
        z_floor=-1, x_min_floor=-10, x_max_floor=10, y_min_floor=-10, y_max_floor=10,
        idle_h_track={}, idle_place={}, t_track={}, node_type={0: kind},
        input_connect={}, inter_connect=(), output_connect={0: outgoing},
        order=[], z_length=1, hadamard_edges=HTable(),
    )
    lifted = state.reward(length=1)[3][0]
    assert lifted["type"] == (kind if outgoing > 1 else int(kind != 1))
    assert lifted["ori"] == ("k" if outgoing > 1 else "i")


@pytest.mark.parametrize("gates", [
    "tdg q[0];", "sdg q[0];", "z q[0];", "x q[0];", "y q[0];",
    "rx(-pi/4) q[0];", "rz(3*pi/4) q[0];", "ry(pi/2) q[0];",
    "cz q[0],q[1];", "ccx q[0],q[1],q[2];",
])
def test_normalization_preserves_signed_channel_and_t_count(gates):
    import numpy as np

    original = source(gates, 3)
    normalized, width = _normalize_qasm(original)
    circuit, canonical = parse_qasm(original), parse_qasm(normalized)
    assert width == 3
    # This checks the frontend identity only. Native channels use independent
    # QuiZX parsing and physical Choi checks in the Rust fixture suite.
    a, b = circuit.to_matrix(), canonical.to_matrix()
    overlap = np.vdot(a, b) / (1 << width)
    assert np.allclose(b, overlap * a, atol=1e-12, rtol=0)
    assert abs(abs(overlap) - 1) < 1e-12
    assert canonical.tcount() == circuit.tcount()
    assert {str(g.phase) for g in canonical.gates if hasattr(g, "phase")} <= {"1/2", "1/4", "-1/2", "-1/4"}


def test_signed_resources_do_not_expand_to_positive_s_gadgets():
    normalized, _ = _normalize_qasm(source("tdg q[0]; sdg q[0];"))
    assert normalized == source("tdg q[0];\nsdg q[0];")


def test_identity_only_source_reports_upstream_interface_limit():
    with pytest.raises(TopoLSImportError, match="identity_layout"):
        from_qasm(source("id q[0];"))


@pytest.mark.parametrize("gates,code", [
    ("rz(pi/8) q[0];", "unsupported_phase"),
    ("rx(pi/8) q[0];", "unsupported_phase"),
    ("rz(0.7853981633974483) q[0];", "unsupported_angle"),
    ("measure q[0] -> c[0];", "classical_control"),
    ("if(c==1) x q[0];", "classical_control"),
    ("reset q[0];", "classical_control"),
    ("h q[1];", "qasm_operand"),
])
def test_unsupported_source_cannot_be_silently_erased(gates, code):
    with pytest.raises(TopoLSImportError) as error:
        _normalize_qasm(source(gates))
    assert error.value.code == code


@pytest.fixture(scope="module")
def t_import():
    return from_qasm(source("t q[0];"), iterations=10, time_bound=0.05)


def test_import_retains_original_and_exact_canonical_inputs(t_import):
    assert t_import.source_qasm == source("t q[0];")
    assert t_import.normalized_qasm == source("t q[0];")
    assert t_import.input_ports == ((0, 0, 0),)
    assert t_import.output_ports == ((0, 0, 4),)
    fixture = Path(__file__).resolve().parents[3] / "bloq_vm/tests/data/topols/t.blog"
    assert t_import.graph.to_text().rstrip() + "\n" == fixture.read_text()


def test_both_layout_engines_preserve_histories_and_interface(t_import):
    pytest.importorskip("topols_core")
    rust = from_qasm(source("t q[0];"), engine="rust", iterations=10, time_bound=0.05)
    assert rust.graph.to_text() == t_import.graph.to_text()
    assert rust.input_ports == t_import.input_ports
    assert rust.output_ports == t_import.output_ports
    assert rust.embedding == t_import.embedding


def test_time_z_dogleg_matches_the_native_channel_fixture(monkeypatch):
    from bloq.interop import _geometry

    original, bent = _geometry._t_placements, []

    def doglegs(nodes, edges, node, **options):
        neighbor = next(b if a == node else a for a, b in edges if node in (a, b))
        p, q = nodes[node]['position'], nodes[neighbor]['position']
        side = next(i for i in (0, 1) if p[i] == q[i])
        for placement in original(nodes, edges, node, **options):
            if nodes[neighbor]['tqec'][2] == 'Z':
                # Exercise the rare time-Z dogleg instead of its shorter straight arm.
                if not any(r[side] != p[side] for name, (r, _) in placement[3].items() if name.startswith('arm_')):
                    continue
                bent.append(placement)
            yield placement

    monkeypatch.setattr(_geometry, '_t_placements', doglegs)
    directory = Path(__file__).resolve().parents[3] / 'bloq_vm/tests/data'
    qasm = (directory / 'topols/entangled_tdg.qasm').read_text()
    imported = from_qasm(qasm, iterations=10, time_bound=0.05)
    assert bent
    assert imported.graph.to_text().rstrip() + '\n' == (directory / 'interop_t_outlets_time_z.blog').read_text()


@pytest.mark.parametrize("mutate,code", [
    (lambda e: e["pos_hist"].update({0: (0.1, 0, 0)}), "coordinates"),
    (lambda e: e["pos_hist"].update({0: (True, 0, 0)}), "coordinates"),
    (lambda e: e["pos_hist"].update({0: (2**31, 0, 0)}), "coordinates"),
    (lambda e: e["pos_hist"].update({0: e["pos_hist"][1]}), "coordinates"),
    (lambda e: e["ori_hist"].update({0: "ij"}), "cube_basis"),
    (lambda e: e["type_hist"].update({0: -1}), "cube_basis"),
    (lambda e: e["path_hist"].append(((0, 0, 1), (2, 0, 1))), "pipe_geometry"),
    (lambda e: e["path_hist"].append(((10, 0, 1), (11, 0, 1))), "incomplete_layout"),
    (lambda e: e["path_hist"].append(((10, 0, 1), (11, 0, 1), (10, 0, 1))), "pipe_geometry"),
])
def test_export_cannot_hide_invalid_raw_geometry(t_import, mutate, code):
    embedding = copy.deepcopy(t_import.embedding)
    mutate(embedding)
    with pytest.raises(TopoLSImportError) as error:
        _pipe_diagram(embedding)
    assert error.value.code == code


def test_declared_source_interface_cannot_lose_a_wire(t_import):
    embedding = copy.deepcopy(t_import.embedding)
    embedding["io_info"].pop(0)
    with pytest.raises(TopoLSImportError, match="boundaries"):
        _convert(embedding, 1)


@pytest.mark.parametrize("options", [
    {"engine": "auto"}, {"iterations": 0}, {"time_bound": float("nan")},
    {"time_bound": float("inf")}, {"seed": True}, {"seed": -1},
    {"seed": 2**64 - 1}, {"iterations": 2**64}, {"time_bound": True},
    {"time_bound": 2**10000}, {"time_bound": "0.1"},
])
def test_invalid_options_reject_before_routing(options):
    with pytest.raises(ValueError):
        from_qasm(source("t q[0];"), **options)
