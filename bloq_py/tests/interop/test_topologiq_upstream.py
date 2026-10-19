"""Behavior preserved by the small patch to the pinned upstream checkout."""

from itertools import product
import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("topologiq", reason="opt-in pinned Topologiq environment")

from topologiq.core.beams import BeamAxisComponent, SingleBeam  # noqa: E402
from topologiq.core.graph_manager.spatial import gen_tent_tgt_coords  # noqa: E402
from topologiq.kwargs import BEAMS_SHORT_LEN  # noqa: E402
from topologiq.utils.classes import GraphBounds  # noqa: E402


def beam(position, direction):
    return SingleBeam(*(
        BeamAxisComponent(start=coordinate, end=coordinate + step * 100, direction=step)
        for coordinate, step in zip(position, direction)
    ))


def numpy_intersects(first, second, short_beams):
    """The original upstream equations, independent of the scalar patch."""
    p1, p2 = first.coords(), second.coords()
    if short_beams and sum(abs(a - b) for a, b in zip(p1, p2)) > BEAMS_SHORT_LEN:
        return False
    d1, d2 = first.direction(), second.direction()
    if np.dot(d1, d2) != 0:
        return False
    sigma, basis = np.subtract(p2, p1), np.subtract(d1, d2)
    return bool(np.all((sigma == 0) | (np.sign(sigma) == np.sign(basis))))


def test_scalar_beam_intersections_match_numpy_equations():
    directions = [
        tuple(sign if component == axis else 0 for component in range(3))
        for axis in range(3) for sign in (-1, 1)
    ]
    for offset in product(range(-2, 3), repeat=3):
        for first_direction, second_direction in product(directions, repeat=2):
            first = beam((0, 0, 0), first_direction)
            second = beam(offset, second_direction)
            for short in (False, True):
                assert first.intersects(second, short_beams=short) == numpy_intersects(first, second, short)

    rng = random.Random(1701)
    for _ in range(2_000):
        first, second = (
            beam(tuple(rng.randrange(-(2**31), 2**31) for _ in range(3)), rng.choice(directions))
            for _ in range(2)
        )
        for short in (False, True):
            assert first.intersects(second, short_beams=short) == numpy_intersects(first, second, short)


def reference_candidate_coords(source, radius, taken, overload, low, high, twin, bounded):
    """Original candidate ordering, with zero treated as an explicit bound."""
    radius = radius + overload if radius == 1 else radius
    if radius <= 3:
        candidates, layer = [], [source]
        moves = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        for distance in range(1, radius + 1):
            layer = [tuple(a + b for a, b in zip(point, move)) for point in layer for move in moves]
            if overload or distance == radius:
                candidates.extend(layer)
    else:
        candidates = []
        for dx in range(radius + 1):
            for dy in range(radius - dx + 1):
                dz = radius - dx - dy
                for delta in product(*([(-d, d) if d else (0,) for d in (dx, dy, dz)])):
                    candidates.append(tuple(a + b for a, b in zip(source, delta)))

    def approved(point):
        if point == source or point in taken:
            return False
        if low is not None and point[2] < low or high is not None and point[2] > high:
            return False
        if twin and bounded:
            # Preserve upstream's upper-bound precedence when both are present.
            if high is not None:
                return point[2] < source[2]
            if low is not None:
                return point[2] > source[2]
        return True

    return [point for point in candidates if approved(point)]


def test_candidate_filter_preserves_order_and_duplicates():
    for source, radius, overload, low, high, twin, bounded, occupied in product(
        ((0, 0, 0), (1, -1, 2)), range(1, 8), (0, 1, 2),
        (None, -2, 0, 2), (None, -2, 0, 2), (False, True), (False, True), (False, True),
    ):
        taken = {
            source, (source[0] + 1, source[1], source[2]),
            (source[0], source[1] + 1, source[2] + 1),
        } if occupied else set()
        expected = reference_candidate_coords(source, radius, taken, overload, low, high, twin, bounded)
        actual = gen_tent_tgt_coords(
            source, radius, taken, overload, {"min": low, "max": high},
            twin, GraphBounds(10, 10) if bounded else None,
        )
        assert actual == expected, (source, radius, overload, low, high, twin, bounded, occupied)


@pytest.mark.parametrize(("source", "radius", "minimum"), [
    ((2, 0, 7), 625, 623), ((0, 0, 0), 750, 748),
])
def test_large_causal_floor_keeps_exact_shell_order(source, radius, minimum):
    # With a positive floor, admissible offsets form this small upper shell cap.
    cap_radius = radius - (minimum - source[2])
    expected = []
    for dx in range(cap_radius + 1):
        for dy in range(cap_radius - dx + 1):
            for x, y in product((-dx, dx) if dx else (0,), (-dy, dy) if dy else (0,)):
                expected.append((source[0] + x, source[1] + y, source[2] + radius - dx - dy))
    assert gen_tent_tgt_coords(source, radius, z_bounds={"min": minimum, "max": None}) == expected


def test_long_conditional_arm_needs_larger_native_search_budget():
    from dataclasses import replace

    import networkx as nx
    from topologiq.core.blocks import ZXBlockRegistry
    from topologiq.core.graph_manager.kwargs import check_assemble_kwargs
    from topologiq.core.pathfinder.pathfinder import PathfinderInitState, PathFinderManager

    source, target = (0, 0, 0), (0, 8, 700)
    graph = nx.Graph()
    graph.add_node(
        0, zx_block=ZXBlockRegistry.get_create(kind="XXZ"), coords=source,
        completions={"degree": 1, "pending": 1},
    )
    graph.add_node(
        1, zx_block=ZXBlockRegistry.get_create(zx_type="XZ"), coords=None,
        completions={"degree": 1, "pending": 1},
    )
    graph.add_edge(0, 1, edge_type="SIMPLE", kind=None, start_coords=None, end_coords=None)
    kwargs = check_assemble_kwargs(
        graph_traverse_mode="tfs", seed=1330, debug=0, animate=False,
    )
    state = PathfinderInitState(
        bgraph=graph, beams={}, beams_short={}, curr_src_id=0, curr_tgt_id=1,
        tent_coords=[target], cross_edge=False,
        # Endpoint coordinates set the workspace extent; there are no interior obstacles.
        taken={source, target}, pruned_taken=set(), is_hadamard=False,
        z_bounds={"min": 699, "max": None}, graph_bounds=None, _kwargs=kwargs,
    )

    short_search = PathFinderManager(replace(
        state, _kwargs=dict(kwargs, a_star_max_visits_standard=3000),
    ))
    assert not short_search.pathfinder_a_star()
    assert short_search.visit_attempts > short_search.max_visits

    long_search = PathFinderManager(replace(
        state, _kwargs=dict(kwargs, a_star_max_visits_standard=20000),
    ))
    paths = long_search.pathfinder_a_star()
    assert paths
    path = next(iter(paths.values()))
    assert path[0][0] == source
    assert path[-1][0] == target
    assert path[-1][1].kind in ("XZ*", "ZX*")
    assert target[2] > state.z_bounds["min"]
    assert long_search.s._kwargs.get("a_star_detour_budget", 8) == 8
