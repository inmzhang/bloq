"""Shared geometry must reject malformed layouts before identity subdivision."""

import pytest
import bloq

from bloq.interop._geometry import GeometryError, _compact_temporal_layers, _space_diagram, build_graph


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
    assert outputs == ((0, 0, 1),)
    assert graph.flatten().block_count == 2


def temporal_wire(length=8):
    graph = bloq.BlockGraph()
    for z in range(length + 1):
        graph.add_block(bloq.Block((0, 0, z), 'Port' if z in (0, length) else 'ZXX',
                                  role='input' if z == 0 else 'output' if z == length else 'auto'))
    for z in range(length):
        graph.add_pipe(bloq.Pipe((0, 0, z), '+Z'))
    return graph


def test_compaction_remaps_each_feedback_target_and_protects_edge_endpoints():
    graph = temporal_wire()
    graph.set_actions([bloq.Action.feedback([
        bloq.FeedbackTarget('Z', (0, 0, 2), direction='+Z'),
        bloq.FeedbackTarget('X', (0, 0, 6), direction='+Z'),
    ])])
    compact, positions = _compact_temporal_layers(graph)
    assert 1 not in {p[2] for p in positions}
    assert {(0, 0, z) for z in (2, 3, 6, 7)} <= positions.keys()
    targets = compact.actions()[0].feedback_targets
    assert [t.pos for t in targets] == [positions[(0, 0, 2)], positions[(0, 0, 6)]]
    assert [str(t.direction) for t in targets] == ['+Z', '+Z']
    assert compact.to_text() == _compact_temporal_layers(compact)[0].to_text()


@pytest.mark.parametrize('seams', [(1,), (1, 5)])
def test_temporal_hadamards_compose_during_identity_contraction(seams):
    graph = temporal_wire()
    for z in range(1, 8):
        graph.set_block_kind((0, 0, z), 'XZX' if sum(s < z for s in seams) % 2 else 'ZXX')
    for z in seams:
        graph.set_pipe_hadamard((0, 0, z), (0, 0, z + 1), True)
    compact, _ = _compact_temporal_layers(graph)
    assert compact.block_count == 2
    assert [p.hadamard for p in compact.pipes()] == [bool(len(seams) % 2)]


def test_authored_memory_duration_is_not_erased():
    memory = bloq.BlockGraph()
    memory.add_block(bloq.Block((0, 0, 0), 'ZXX', height='2d'))
    assert _compact_temporal_layers(memory)[0].to_text() == memory.to_text()


def test_disconnected_patch_lifetimes_remain_disconnected():
    graph = temporal_wire(2)
    for z in (3, 4, 5):
        graph.add_block(bloq.Block((0, 0, z), 'Port' if z in (3, 5) else 'ZXX',
                                  role='input' if z == 3 else 'output' if z == 5 else 'auto'))
    graph.add_pipe(bloq.Pipe((0, 0, 3), '+Z'))
    graph.add_pipe(bloq.Pipe((0, 0, 4), '+Z'))
    compact, positions = _compact_temporal_layers(graph)
    assert compact.pipe_count == 2
    assert not compact.has_pipe_between(positions[(0, 0, 2)], positions[(0, 0, 3)])


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
