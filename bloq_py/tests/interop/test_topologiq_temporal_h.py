"""The opt-in upstream profile keeps Hadamards temporal without global state."""

from copy import deepcopy
from dataclasses import replace

import pytest

nx = pytest.importorskip("networkx")
pytest.importorskip("topologiq", reason="opt-in pinned Topologiq environment")

from topologiq.core.blocks import ZXBlockRegistry  # noqa: E402
from topologiq.core.graph_manager.edge_handlers import add_path_to_bgraph  # noqa: E402
from topologiq.core.graph_manager.first_cube import pick_id_and_kind  # noqa: E402
from topologiq.core.graph_manager.graph_manager import BlockGraphManager  # noqa: E402
from topologiq.core.pathfinder.pathfinder import PathfinderInitState, PathFinderManager  # noqa: E402

from bloq.interop._topologiq_flow import prepare_graph  # noqa: E402
from bloq.interop._qasm import parse_qasm  # noqa: E402
from bloq.interop.topologiq import TopologiqImportError, _routed_path, from_qasm  # noqa: E402


SEARCHES = ("pathfinder_bfs", "pathfinder_a_star", "pathfinder_a_star_multi_target")
KINDS = ("XXZ", "ZZX", "XZZ", "ZXX", "ZXZ", "XZX")


def block(kind):
    return ZXBlockRegistry.get_create(kind=kind)


def add_node(graph, node, coords, kind=None):
    graph.add_node(
        node,
        coords=coords,
        zx_block=block(kind) if kind else ZXBlockRegistry.get_create(zx_type="Z"),
        completions={"degree": 1, "pending": 1},
    )


def search_state(graph, target, *, hadamard=False, cross_edge=True, obstacles=()):
    placed = {
        data["coords"]
        for _, data in graph.nodes(data=True)
        if data["coords"] is not None
    }
    taken = placed | set(obstacles)
    return PathfinderInitState(
        bgraph=graph,
        beams={},
        beams_short={},
        curr_src_id=0,
        curr_tgt_id=1,
        tent_coords=[target],
        cross_edge=cross_edge,
        taken=taken,
        pruned_taken=taken - {graph.nodes[0]["coords"], graph.nodes[1]["coords"]},
        is_hadamard=hadamard,
        z_bounds={"min": None, "max": None},
        graph_bounds=None,
        _kwargs={
            "min_succ_rate": 100,
            "twisted_y": False,
            "temporal_hadamards_only": True,
        },
    )


@pytest.mark.parametrize("kind", KINDS)
def test_move_policy_preserves_simple_steps_and_does_not_poison_shared_caches(kind):
    cube = block(kind)
    for move in cube.get_move_vectors:
        plain = tuple(cube.nxt_kinds(move))
        legacy_h = tuple(cube.nxt_kinds(move, is_hadamard=True))
        assert plain and legacy_h
        restricted_h = cube.nxt_kinds(
            move, is_hadamard=True, temporal_hadamards_only=True
        )
        assert tuple(restricted_h) == (legacy_h if move[2] else ())
        assert tuple(cube.nxt_kinds(move, temporal_hadamards_only=True)) == plain
        # The same registry object is reused by later legacy-profile builds.
        assert block(kind) is cube
        assert tuple(cube.nxt_kinds(move, is_hadamard=True)) == legacy_h


@pytest.mark.parametrize("method", SEARCHES)
@pytest.mark.parametrize("dz", (0, 2, -2))
@pytest.mark.parametrize("source_kind,target_kind", (("XZX", "ZXZ"), ("ZXZ", "XZX")))
def test_committed_hadamard_paths_keep_the_source_edge_and_use_temporal_steps(
    method,
    dz,
    source_kind,
    target_kind,
):
    source, target = (0, 0, 0), (1, 0, dz)
    graph = nx.Graph()
    add_node(graph, 0, source, source_kind)
    add_node(graph, 1, target, target_kind)
    graph.add_edge(0, 1, edge_type="HADAMARD", kind=None)
    # Force both signs across cases; the same-layer case must take a detour.
    obstacles = {(0, 0, -1 if dz > 0 else 1)} if dz else set()
    state = search_state(graph, target, hadamard=True, obstacles=obstacles)
    paths = getattr(PathFinderManager(state), method)()
    assert paths

    for path in paths.values():
        committed, taken, intermediate = add_path_to_bgraph(
            deepcopy(graph),
            state.taken.copy(),
            path.copy(),
            0,
            1,
            True,
            True,
        )
        routed = nx.shortest_path(committed, 0, 1)
        assert [committed.nodes[node]["coords"] for node in routed] == [
            p for p, _ in path
        ]
        assert set(routed[1:-1]) == intermediate
        assert committed.nodes[0]["coords"] == source
        assert committed.nodes[1]["coords"] == target
        assert all(committed.degree(node) == 2 for node in intermediate)
        assert len(committed.edges) == len(routed) - 1
        assert not obstacles.intersection(taken - state.taken)
        h_pipes = [
            (u, v)
            for u, v, edge in committed.edges(data=True)
            if edge["edge_type"] == "HADAMARD"
        ]
        assert len(h_pipes) == 1
        assert set(h_pipes[0]) == {routed[0], routed[1]}
        first, second = (committed.nodes[node]["coords"] for node in routed[:2])
        assert first[:2] == second[:2]
        assert abs(first[2] - second[2]) == 1
        if dz:
            assert (second[2] - first[2]) * dz > 0
        for u, v, edge in committed.edges(data=True):
            if edge["edge_type"] == "HADAMARD":
                assert edge["kind"] in ("XZOH", "ZXOH")
            else:
                assert edge["edge_type"] == "SIMPLE"
                assert not edge["kind"].endswith("H")


@pytest.mark.parametrize("method", SEARCHES)
@pytest.mark.parametrize("block_other_exit", (False, True))
def test_routes_preserve_the_last_free_temporal_exit_of_a_pending_hadamard(
    method, block_other_exit
):
    graph = nx.Graph()
    add_node(graph, 0, (0, 0, 0), "XZX")
    add_node(graph, 1, (2, 0, 0), "XZX")
    add_node(graph, 2, (1, 0, -1), "XZX")
    add_node(graph, 3, None)
    graph.add_edge(0, 1, edge_type="SIMPLE", kind=None)
    graph.add_edge(2, 3, edge_type="HADAMARD", kind=None)
    remaining_exit = (1, 0, 0)
    obstacles = {(1, 0, -2)} if block_other_exit else set()
    state = search_state(graph, (2, 0, 0), obstacles=obstacles)
    paths = getattr(PathFinderManager(state), method)()
    assert paths
    for path in paths.values():
        assert (
            remaining_exit in {coords for coords, _ in path}
        ) is not block_other_exit
        committed, taken, _ = add_path_to_bgraph(
            deepcopy(graph),
            state.taken.copy(),
            path.copy(),
            0,
            1,
            False,
            True,
        )
        assert committed.edges[2, 3]["edge_type"] == "HADAMARD"
        assert committed.edges[2, 3]["kind"] is None
        assert any((1, 0, z) not in taken for z in (-2, 0))


@pytest.mark.parametrize("zx_type", ("X", "Z"))
def test_initial_cube_selection_keeps_a_temporal_exit_without_changing_registry(
    zx_type,
):
    graph = nx.Graph()
    graph.add_node(1, zx_block=ZXBlockRegistry.get_create(zx_type=zx_type))
    add_node(graph, 2, None)
    graph.add_edge(1, 2, edge_type="HADAMARD", kind=None)
    for seed in range(20):
        _, kind, _ = pick_id_and_kind(
            graph,
            "centrality-random",
            {},
            [],
            override_first_cube=(1, None),
            random_seed=seed,
            temporal_hadamards_only=True,
        )
        assert block(kind).get_open_axes[2]
    assert set(graph.nodes[1]["zx_block"].get_fulfillment_kinds) == (
        {"ZXZ", "ZZX", "XZZ"} if zx_type == "X" else {"XZX", "XXZ", "ZXX"}
    )


@pytest.mark.parametrize("method", SEARCHES)
def test_three_pending_hadamards_report_no_route_without_dropping_an_edge(method):
    graph = nx.Graph()
    add_node(graph, 0, (0, 0, 0), "XZX")
    add_node(graph, 1, None)
    graph.add_edge(0, 1, edge_type="SIMPLE", kind=None)
    for node in range(2, 5):
        add_node(graph, node, None)
        graph.add_edge(1, node, edge_type="HADAMARD", kind=None)
    state = search_state(graph, (0, 0, 1), cross_edge=False)
    assert not getattr(PathFinderManager(state), method)()
    assert set(graph.edges) == {(0, 1), (1, 2), (1, 3), (1, 4)}
    assert all(edge["kind"] is None for _, _, edge in graph.edges(data=True))
    legacy = replace(state, _kwargs=dict(state._kwargs, temporal_hadamards_only=False))
    assert getattr(PathFinderManager(legacy), method)()


def test_completed_source_edge_provenance_survives_both_profiles():
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nh q[1];\n'
    for options in ({}, {"temporal_hadamards_only": True}, {}):
        manager = BlockGraphManager(
            prepare_graph(parse_qasm(qasm).to_graph()),
            debug=0,
            animate=False,
            **options,
        )
        source_edges = manager.in_edge_types.copy()
        manager.build()
        for (u, v), edge_type in source_edges.items():
            path = _routed_path(manager, u, v)
            assert path[0] in manager.twin_trace[u]
            assert path[-1] in manager.twin_trace[v]
            pipes = [manager.bgraph.edges[a, b] for a, b in zip(path, path[1:])]
            assert sum(pipe["edge_type"] == "HADAMARD" for pipe in pipes) == (
                edge_type == "HADAMARD"
            )
            if options:
                for a, b in zip(path, path[1:]):
                    if manager.bgraph.edges[a, b]["edge_type"] == "HADAMARD":
                        assert (
                            manager.bgraph.nodes[a]["coords"][:2]
                            == manager.bgraph.nodes[b]["coords"][:2]
                        )


def test_public_importer_enforces_its_profile_when_upstream_returns_a_spatial_hadamard(
    monkeypatch,
):
    build = BlockGraphManager.build
    violating_edge = []

    def inject_spatial_hadamard(manager):
        assert manager._kwargs["temporal_hadamards_only"] is True
        build(manager)
        u, v = next(
            (u, v)
            for u, v, edge in manager.bgraph.edges(data=True)
            if edge["edge_type"] == "HADAMARD"
        )
        x, y, z = manager.bgraph.nodes[u]["coords"]
        manager.bgraph.nodes[v]["coords"] = (x + 1, y, z)
        violating_edge.extend((u, v))

    monkeypatch.setattr(BlockGraphManager, "build", inject_spatial_hadamard)
    with pytest.raises(TopologiqImportError, match="temporal-only") as caught:
        from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n')
    assert caught.value.code == "hadamard_axis"
    assert caught.value.nodes == tuple(violating_edge)
