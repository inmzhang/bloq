"""Shared geometry must reject malformed layouts before identity subdivision."""

import pytest

from bloq.interop._geometry import GeometryError, _space_diagram, build_graph


@pytest.fixture
def wire():
    nodes = {
        "in": dict(position=(0, 0, 0), tqec="ZXX", other=dict(type="input", qubit=0)),
        "middle": dict(position=(0, 0, 1), tqec="ZXX", other=None),
        "out": dict(position=(0, 0, 2), tqec="ZXX", other=dict(type="output", qubit=0)),
    }
    edges = {(a, b): (nodes[a]["position"], nodes[b]["position"])
             for a, b in (("in", "middle"), ("middle", "out"))}
    return nodes, edges


def test_identity_subdivision_preserves_declared_source_ports(wire):
    nodes, edges = _space_diagram(*wire)
    graph, inputs, outputs = build_graph(nodes, list(edges), 1, [])
    graph.validate()
    assert inputs == ((0, 0, 0),)
    assert outputs == ((0, 0, 6),)


@pytest.mark.parametrize("position", [(True, 0, 1), (0.0, 0, 1), (2**30, 0, 1), (0, 0, 0)])
def test_subdivision_does_not_coerce_invalid_or_overlapping_coordinates(wire, position):
    nodes, edges = wire
    nodes["middle"]["position"] = position
    with pytest.raises(GeometryError, match="coordinates"):
        _space_diagram(nodes, edges)


@pytest.mark.parametrize("edge,endpoints", [
    (("in", "middle"), ((0, 0, 0), (0, 0, 2))),
    (("in", "missing"), ((0, 0, 0), (0, 0, 1))),
    (("in", "out"), ((0, 0, 0), (0, 0, 2))),
])
def test_subdivision_does_not_hide_stale_or_nonunit_edges(wire, edge, endpoints):
    nodes, _ = wire
    with pytest.raises(GeometryError, match="pipe_geometry"):
        _space_diagram(nodes, {edge: endpoints})
