"""Optional pinned-upstream checks for the certified focused-flow adapter."""

import random

import pytest

zx = pytest.importorskip("pyzx", reason="opt-in pinned Topologiq environment")
pytest.importorskip("topologiq", reason="opt-in pinned Topologiq environment")

from pyzx.linalg import Mat2  # noqa: E402
from pyzx.pauliweb import compute_pauli_webs  # noqa: E402

from bloq.interop import _topologiq_flow as flow  # noqa: E402


def circuit_graph(gates, width=1):
    return zx.Circuit.from_qasm(
        f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}\n'
    ).to_graph()


def canonical_pack(pack):
    order, zwebs, xwebs = pack
    return (
        order,
        {vertex: web.half_edges() for vertex, web in zwebs.items()},
        {vertex: web.half_edges() for vertex, web in xwebs.items()},
    )


@pytest.mark.parametrize(
    "gates,width",
    [
        ("h q[0];", 1),
        ("s q[0];", 1),
        ("t q[0];", 1),
        ("h q[0]; t q[0]; h q[0];", 1),
        ("cx q[0],q[1];", 2),
        ("h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[2]; t q[2];", 3),
    ],
)
def test_native_pauli_packs_match(gates, width):
    graph = circuit_graph(gates, width)
    original = graph.to_json()
    expected = canonical_pack(compute_pauli_webs(graph))
    assert canonical_pack(flow._compute_pauli_webs(graph)) == expected
    assert graph.to_json() == original


def test_incremental_span_matches_independent_binary_solver():
    rng = random.Random(1701)
    for rows in range(1, 7):
        for columns in range(1, 7):
            for _ in range(3):
                span = flow._UnitSpan(rows)
                values = []
                for column in range(columns):
                    values.append(rng.randrange(1 << rows))
                    span.add(values[-1], 1 << column)
                    matrix = Mat2([
                        [int(bool(value & (1 << row))) for value in values]
                        for row in range(rows)
                    ])
                    for row in range(rows):
                        expected = matrix.solve(Mat2.unit_vector(rows, row))
                        assert (row not in span.pending) == (expected is not None)
                        if expected is not None:
                            actual = 0
                            for index, value in enumerate(values):
                                if span.witnesses[row] & (1 << index):
                                    actual ^= value
                            assert actual == 1 << row


def test_preparation_preserves_source_and_upstream_functions():
    import pyzx.pauliweb
    from topologiq.input.zx_manager import AugmentedZXGraph

    graph = circuit_graph("t q[0];")
    original = graph.to_json()
    original_gflow = pyzx.pauliweb.gflow
    original_compute = pyzx.pauliweb.compute_pauli_webs
    augmented = flow.prepare_graph(graph)
    assert isinstance(augmented, AugmentedZXGraph)
    assert canonical_pack(augmented.pauli_pack) == canonical_pack(compute_pauli_webs(graph))
    assert augmented.pauli_pack_reduced is not None
    assert graph.to_json() == original
    assert pyzx.pauliweb.gflow is original_gflow
    assert pyzx.pauliweb.compute_pauli_webs is original_compute


def test_unavailable_reduced_pack_has_a_diagnostic():
    with pytest.warns(RuntimeWarning, match="Pauli webs unavailable for reduced graph"):
        augmented = flow.prepare_graph(circuit_graph("h q[0];"))
    assert augmented.pauli_pack is not None
    assert augmented.pauli_pack_reduced is None
    assert "No edge between" in augmented.pauli_pack_errors["reduced"]


def test_pyzx_version_guard(monkeypatch):
    monkeypatch.setattr(flow, "version", lambda _: "0.0.0")
    with pytest.raises(ValueError, match=r"requires pyzx==0\.10\.7"):
        flow.prepare_graph(circuit_graph("t q[0];"))
