"""Tests for the read-only ZX introspection surface: to_zx_graph and the
ZXGraph/ZXNode/ZXEdge views, checked for consistency against the source
BlockGraph and its stabilizer derivation.
"""

import pytest

import bloq

to_zx_graph = bloq.to_zx_graph


@pytest.fixture
def cnot_zx():
    g = bloq.GalleryItem("cnot").load()
    return g, to_zx_graph(g)


def test_zx_counts_consistent(cnot_zx):
    g, zx = cnot_zx
    assert zx.node_count == len(zx.nodes()) > 0
    assert zx.edge_count == len(zx.edges()) > 0
    assert zx.total_ids == zx.node_count + zx.edge_count
    # Every block contributes at least one ZX node; ports stay boundary nodes.
    assert zx.node_count >= g.block_count
    assert sum(1 for n in zx.nodes() if n.is_port) == g.port_count


def test_zx_node_views(cnot_zx):
    g, zx = cnot_zx
    nodes = zx.nodes()
    for i, n in enumerate(nodes):
        assert n.id == i
        assert n.kind in {"X", "Y", "Z", "Port", "T"} or n.kind.startswith(
            "Selective("
        )
        assert n.is_boundary == (n.is_port or n.is_t or n.kind.startswith("Selective"))
    # Node positions land inside the source graph's occupied volume.
    occupied = set(g.occupied_positions())
    port_positions = {n.pos for n in nodes if n.is_port}
    assert port_positions <= occupied
    for pos in port_positions:
        assert zx.node_at(pos) is not None
    assert zx.node_at((999, 999, 999)) is None


def test_zx_edges_and_neighbors(cnot_zx):
    _, zx = cnot_zx
    nodes, edges = zx.nodes(), zx.edges()
    for e in edges:
        assert e.n1 < len(nodes) and e.n2 < len(nodes)
        assert e.id >= len(nodes)  # edge ids follow node ids in the id space
        # Parallel edges exist (the ZX layer is a multigraph), so lookups by
        # endpoint pair may return a sibling edge — assert pair-consistency,
        # not id identity.
        assert zx.edge_id(e.n1, e.n2) is not None
        back = zx.edge_between(e.n2, e.n1)
        assert back is not None and {back.n1, back.n2} == {e.n1, e.n2}
        assert e.n2 in zx.neighbors(e.n1)
        assert e.n1 in zx.neighbors(e.n2)
    assert zx.neighbors(10**6) is None
    assert zx.edge_between(0, 0) is None


def test_zx_layers_and_ports(cnot_zx):
    _, zx = cnot_zx
    layers = zx.z_layers()
    assert layers == sorted(set(layers))
    assert {n.pos[2] for n in zx.nodes()} == set(layers)
    # cnot's ports are open boundaries, so the graph is open and the output
    # ports (all neighbors strictly in the past) form a subset of the ports.
    assert zx.is_open
    outputs = [n for n in zx.nodes() if zx.is_output_port(n.id)]
    assert all(n.is_port for n in outputs)
    assert {tuple(n.pos) for n in outputs} == set(zx.output_ports())
    with pytest.raises(bloq.InvalidArgumentError, match="unknown node id"):
        zx.is_output_port(10**6)


def test_zx_clifford_and_t():
    zx = to_zx_graph(bloq.GalleryItem("cnot").load())
    assert zx.is_clifford_computation
    tzx = to_zx_graph(bloq.GalleryItem.T_GATE.load())
    assert any(n.is_t for n in tzx.nodes())
    assert not tzx.is_clifford_computation


def test_zx_validate_and_measurement_columns():
    g = bloq.BlockGraph.from_text(
        "BLOG 1.0\n0: ZXZ [0,0,0]\n1: Z [0,0,1]\n0 -> +Z\nm = measure 1\n"
    )
    zx = to_zx_graph(g)
    zx.validate_for_program()
    assert zx.measurement_columns() == {"m": zx.node_at((0, 0, 1)).id}
    edge_zx = to_zx_graph(bloq.GalleryItem.T_GATE.load())
    edge = edge_zx.edge_between(
        edge_zx.node_at((0, 0, 1)).id, edge_zx.node_at((1, 0, 1)).id
    )
    assert edge_zx.measurement_columns() == {"mzz": edge.id}
