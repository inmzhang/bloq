"""Shared geometry must reject malformed layouts before identity subdivision."""

import pytest
import bloq

import copy

from bloq.interop._geometry import (
    GeometryError, _compact_temporal_layers, _space_diagram, add_t_resources, build_graph,
    push_spatial_hadamards,
)


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


@pytest.mark.parametrize('axis', (0, 1))
def test_spatial_pipes_retain_the_upstream_unit_grid(axis):
    p, q = (2, 3, 4), tuple(v + (i == axis) for i, v in enumerate((2, 3, 4)))
    nodes = {n: dict(position=r, tqec='XZX', other=None) for n, r in [('a', p), ('b', q)]}
    spaced, edges = _space_diagram(nodes, {('a', 'b'): (p, q)})
    assert len(spaced) == 2
    assert edges == {('a', 'b'): ((2, 3, 12), (*q[:2], 12))}


def resource_nodes(frame='XZX'):
    return {'root': dict(position=(0, 0, 0), tqec=frame, other=None),
            'resource': dict(position=(1, 0, 0), tqec=None, other='T')}


def test_t_arm_extends_outward_when_its_temporal_cap_is_occupied():
    nodes = resource_nodes()
    nodes['obstacle'] = dict(position=(1, 0, 1), tqec='ZXX', other=None)
    edges = [('root', 'resource')]
    actions = add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert nodes['resource']['position'] == (2, 0, 0)
    assert nodes['t_arm_0'] == dict(position=(1, 0, 0), tqec='XZX', other=None)
    assert ('root', 't_arm_0') in edges and ('t_arm_0', 'resource') in edges
    assert actions[0].target.pos == (1, 0, 0)
    assert str(actions[0].target.direction) == '+X'


def test_bent_t_arm_can_use_the_opposite_perpendicular_side():
    nodes = resource_nodes('ZXZ')
    nodes['obstacle'] = dict(position=(1, 1, 0), tqec='ZXX', other=None)
    actions = add_t_resources(nodes, [('root', 'resource')], [('resource', '7/4', 't')])
    assert nodes['resource']['position'] == (1, 0, 0)
    assert nodes['t_center']['position'] == (1, -1, 1)
    assert str(actions[0].target.direction) == '-Y'
    assert str(actions[1].condition) == 't'


def test_t_placement_rejects_a_blocked_ray_without_partial_mutation():
    from copy import deepcopy

    nodes = resource_nodes()
    for n, p in [('cap', (1, 0, 1)), ('ray', (2, 0, 0)),
                 ('left', (1, -1, 0)), ('right', (1, 1, 0))]:
        nodes[n] = dict(position=p, tqec='ZXX', other=None)
    edges = [('root', 'resource')]
    before = deepcopy(nodes)
    with pytest.raises(GeometryError) as error:
        add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert error.value.code == 'resource_collision'
    assert nodes == before and edges == [('root', 'resource')]


@pytest.mark.parametrize('axis,frame,lazy_fallback', [(0, 'XZX', False), (1, 'ZXX', False), (1, 'ZXX', True)])
def test_adjacent_t_outlets_use_disjoint_doglegs_with_the_native_channel_fixture(axis, frame, lazy_fallback, monkeypatch):
    from pathlib import Path

    if lazy_fallback:
        from bloq.interop import _geometry

        original = _geometry._t_placements

        def blocked_greedy(*args, **kwargs):
            raise GeometryError('resource_collision', 'Exercise joint placement')

        snapshots = []

        def bounded_candidates(*args, bends=False, **kwargs):
            snapshots.append(kwargs['occupied'])
            assert isinstance(snapshots[0], frozenset)
            assert all(snapshot is snapshots[0] for snapshot in snapshots)
            if not bends:
                return
            for index, candidate in enumerate(original(*args, bends=True, **kwargs)):
                assert index < 4, 'A solved layout must not enumerate unused alternatives'
                yield candidate

        monkeypatch.setattr(_geometry, '_greedy_t_resources', blocked_greedy)
        monkeypatch.setattr(_geometry, '_t_placements', bounded_candidates)

    nodes = {f'w{z}': dict(position=(0, 0, z), tqec=frame,
                          other=dict(type='input' if z == 0 else 'output', qubit=0) if z in (0, 4) else None)
             for z in range(5)}
    edges = [(f'w{z}', f'w{z+1}') for z in range(4)]
    for z in (1, 2):
        nodes[f't{z}'] = dict(position=(int(axis == 0), int(axis == 1), z), tqec=None, other='T')
        edges.append((f'w{z}', f't{z}'))
    actions = add_t_resources(nodes, edges, [('t1', '1/4', 'a'), ('t2', '7/4' if axis else '1/4', 'b')])
    assert len({data['position'] for data in nodes.values()}) == len(nodes)
    graph, inputs, outputs = build_graph(nodes, edges, 1, actions)
    assert inputs == ((0, 0, 0),) and outputs == ((0, 0, 4),)
    # Rust checks the signed logical and full physical channel of this exact output.
    fixture = Path(__file__).resolve().parents[3] / f'bloq_vm/tests/data/interop_t_outlets_{axis}.blog'
    assert graph.to_text().rstrip() + '\n' == fixture.read_text()


@pytest.mark.parametrize('compact', (False, True))
def test_t_factories_do_not_alias_later_resources_or_data_columns(compact, monkeypatch):
    if compact:
        from bloq.interop import _geometry

        def exhausted(*args, **kwargs):
            raise GeometryError('resource_collision', 'Exercise compact placement')

        monkeypatch.setattr(_geometry, '_greedy_t_resources', exhausted)
        monkeypatch.setattr(_geometry, '_search_t_placements', exhausted)
    nodes = resource_nodes()
    nodes.update(root_later=dict(position=(0, 0, 10), tqec='XZX', other=None),
                 resource_later=dict(position=(1, 0, 10), tqec=None, other='T'),
                 data_later=dict(position=(2, 0, 30), tqec='XZX', other=None))
    edges = [('root', 'resource'), ('root_later', 'resource_later')]
    add_t_resources(nodes, edges, [('resource', '1/4', 'a'), ('resource_later', '7/4', 'b')])
    for name, resource in [('a', 'resource'), ('b', 'resource_later')]:
        column = nodes[f'{name}_source']['position'][:2]
        # A cap may wait for a later control: its entire patch column must
        # remain available, regardless of the other blocks' source times.
        assert {n for n, d in nodes.items() if d['position'][:2] == column} == {
            resource, f'{name}_source', f'{name}_cap',
        }


@pytest.mark.parametrize('frame', ('ZXX', 'XZX'))
@pytest.mark.parametrize('dz', (1, -1))
@pytest.mark.parametrize('blocked', (False, True))
def test_temporal_t_outlet_uses_an_identity_elbow_and_existing_signed_recipe(frame, dz, blocked):
    from pathlib import Path

    axis = 1 if frame == 'ZXX' else 0
    nodes = {'root': dict(position=(0, 0, 2), tqec=frame, other=None),
             'resource': dict(position=(0, 0, 2 + dz), tqec=None, other='T')}
    for role, sign in (('input', -1), ('output', 1)):
        p = tuple(sign if i == axis else (2 if i == 2 else 0) for i in range(3))
        nodes[role] = dict(position=p, tqec=frame, other=dict(type=role, qubit=0))
    edges = [('input', 'root'), ('root', 'output'), ('root', 'resource')]
    if blocked:
        p = nodes['output']['position']
        nodes['output_elbow'] = dict(nodes['output'], other=None)
        nodes['obstacle'] = dict(position=(*p[:2], 2 + dz), tqec=frame, other=None)
        nodes['output'] = dict(nodes['output'], position=(*p[:2], 2 + 2*dz))
        edges.remove(('root', 'output'))
        edges.extend([('root', 'output_elbow'), ('output_elbow', 'obstacle'), ('obstacle', 'output')])
    actions = add_t_resources(nodes, edges, [('resource', '7/4', 'magic')])
    assert nodes['magic_temporal_elbow'] == dict(position=(0, 0, 2 + dz), tqec=frame, other=None)
    assert ('root', 'magic_temporal_elbow') in edges
    assert nodes['resource']['position'][axis] < 0 if blocked else nodes['resource']['position'][axis] > 0
    graph, _, _ = build_graph(nodes, edges, 1, actions)
    if frame == 'ZXX' and dz == 1 and blocked:
        fixture = Path(__file__).resolve().parents[3] / 'bloq_vm/tests/data/interop_temporal_t_elbow.blog'
        assert graph.to_text().rstrip() + '\n' == fixture.read_text()


@pytest.mark.parametrize('frame,blocked,code', [('ZXZ', False, 'resource_frame'),
                                               ('XXZ', False, 'resource_frame'),
                                               ('ZXX', True, 'resource_collision')])
def test_temporal_t_normalization_rejects_unproved_or_blocked_frames_atomically(frame, blocked, code):
    nodes = {'root': dict(position=(0, 0, 0), tqec='XZX', other=None),
             'resource': dict(position=(0, 0, 1), tqec=None, other='T'),
             'other_root': dict(position=(10, 0, 0), tqec=frame, other=None),
             'other_resource': dict(position=(10, 0, 1), tqec=None, other='T')}
    if blocked:
        for y in (-1, 1):
            nodes[f'obstacle_{y}'] = dict(position=(10, y, 1), tqec='ZXX', other=None)
    edges = [('root', 'resource'), ('other_root', 'other_resource')]
    before, old_edges = copy.deepcopy(nodes), list(edges)
    with pytest.raises(GeometryError) as error:
        add_t_resources(nodes, edges, [('resource', '1/4', 'a'), ('other_resource', '7/4', 'b')])
    assert error.value.code == code
    assert nodes == before and edges == old_edges


@pytest.mark.parametrize('z', [-(2**31), 2**31 - 1])
def test_t_endpoints_reject_coordinate_overflow_without_mutation(z):
    nodes = resource_nodes()
    nodes = {n: dict(d, position=(*d['position'][:2], z)) for n, d in nodes.items()}
    edges = [('root', 'resource')]
    before = nodes.copy()
    with pytest.raises(GeometryError, match='signed 32-bit range'):
        add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert nodes == before and edges == [('root', 'resource')]


def test_exhausted_candidate_budget_falls_back_to_compact_routing(monkeypatch):
    from bloq.interop import _geometry

    nodes, edges = resource_nodes(), [('root', 'resource')]
    before = nodes.copy()
    placement = next(_geometry._t_placements(nodes, edges, 'resource'))

    def blocked_greedy(*args, **kwargs):
        raise GeometryError('resource_collision', 'Exercise joint placement')

    def candidates(*args, **kwargs):
        for _ in range(100_001):
            yield placement
        pytest.fail('Candidate enumeration exceeded its budget')

    monkeypatch.setattr(_geometry, '_greedy_t_resources', blocked_greedy)
    monkeypatch.setattr(_geometry, '_t_placements', candidates)
    actions = add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert nodes != before and len(actions) == 2
    assert len({data['position'] for data in nodes.values()}) == len(nodes)


def test_joint_backtracking_restores_choices_but_keeps_earlier_constraints():
    from bloq.interop._geometry import _joint_t_placements

    def choice(column, cells, distance=0):
        end = (column, 0, 0)
        return (distance, end, 'XZX', {'source': ((column, 0, -1), 'T')}, frozenset({end, *cells}))

    domains = [
        [choice(10, [(3, 0, 1), (4, 0, 1), (9, 0, 1)]), choice(10, [(9, 0, 1)], 1)],
        [choice(20, [(3, 0, 1)]), choice(20, [(4, 0, 1)]), choice(20, [(8, 0, 2)])],
        [choice(30, [(8, 0, 2), (9, 0, 2)])],
    ]
    # Resource 2 is fixed first. Resource 0's first arm then blocks both
    # surviving choices for resource 1, so its alternate arm must be tried.
    with pytest.raises(GeometryError, match='exceeded 2 choices'):
        _joint_t_placements(domains, max_attempts=2)
    placed = _joint_t_placements(domains)
    assert placed[0] == domains[0][1]
    assert placed[1] in domains[1][:2]
    assert placed[2] == domains[2][0]


@pytest.mark.parametrize('outlets', [((0, 1), (0, 1), (0, 1), (2, 3)),
                                     ((0, 1), (0,), (1, 2), (2, 3))])
def test_joint_outlets_require_a_distinct_factory_column_for_every_resource(outlets):
    from bloq.interop._geometry import _joint_t_placements

    domains = [[(0, (x, 0, i * 10), 'XZX',
                 {'source': ((x, 0, i * 10 - 1), 'T')}, frozenset({(x, 0, i * 10)}))
                for x in choices] for i, choices in enumerate(outlets)]
    if outlets[2] == (0, 1):
        # The global union has four columns, but the first three resources
        # compete for only two. Adding a fourth unrelated outlet cannot help.
        with pytest.raises(GeometryError, match='no distinct factory columns'):
            _joint_t_placements(domains)
    else:
        placed = _joint_t_placements(domains)
        assert len({p[3]['source'][0][:2] for p in placed.values()}) == len(outlets)


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


@pytest.mark.parametrize("position", [(True, 0, 1), (0.0, 0, 1), (2**31, 0, 1), (0, 0, 0)])
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


def turn():
    """A wire whose only spatial pipe carries a removable Hadamard wall."""
    nodes = {
        "in": dict(position=(0, 0, -1), tqec="ZXZ", other=dict(type="input", qubit=0)),
        "a": dict(position=(0, 0, 0), tqec="ZXZ", other=None),
        "b": dict(position=(0, 0, 1), tqec="ZXZ", other=None),
        "c": dict(position=(1, 0, 1), tqec="XZX", other=None),
        "d": dict(position=(1, 0, 2), tqec="XZX", other=None),
        "out": dict(position=(1, 0, 3), tqec="XZX", other=dict(type="output", qubit=0)),
    }
    return nodes, [("in", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "out")]


def hadamard_axes(graph):
    return [str(p.direction)[1] for p in graph.flatten().pipes() if p.hadamard]


@pytest.mark.parametrize(('target', 'pauli'), [(None, None), ('b', 'X'), ('b', 'Z'), ('c', 'X'), ('c', 'Z')])
def test_spatial_walls_move_to_temporal_pipes_without_changing_the_channel(target, pauli):
    nodes, edges = turn()
    source = copy.deepcopy(nodes)
    actions = ([] if target is None else [bloq.Action.feedback([
        bloq.FeedbackTarget(pauli, nodes[target]['position'], direction='+X' if target == 'b' else '+Z')
    ])])
    transported = push_spatial_hadamards(nodes, edges, actions=actions)
    assert {n: d["tqec"] for n, d in nodes.items() if d["tqec"] != source[n]["tqec"]} == {"c": "ZXZ"}
    pushed, _, _ = build_graph(nodes, edges, 1, transported)
    original, _, _ = build_graph(source, edges, 1, actions)
    assert hadamard_axes(original) == ["X"]
    assert hadamard_axes(pushed) == ["Z"]
    for state in ("zero", "plus"):
        expected, actual = (
            bloq.run_noiseless(bloq.compile(graph, distance=3, prepare_t_with_mpps=True),
                               input_state=state)["samples"][0]["logical_bloch"][0]
            for graph in (original, pushed)
        )
        assert actual == pytest.approx(expected, abs=1e-8)
    again = copy.deepcopy(nodes)
    assert push_spatial_hadamards(again, edges, actions=transported) == transported
    assert again == nodes


def test_conflicting_fixed_frames_keep_the_wall():
    nodes, edges = turn()
    source = copy.deepcopy(nodes)
    push_spatial_hadamards(nodes, edges, fixed={"b", "c"})
    assert nodes == source


def test_feedback_transport_preserves_guards_directions_and_unchanged_endpoint_frames():
    nodes, edges = turn()
    guard = bloq.Expr.var('control')
    action = bloq.Action.feedback([
        bloq.FeedbackTarget('Z', nodes['b']['position'], direction='+X'),
        bloq.FeedbackTarget('Z', nodes['c']['position'], direction='+Z'),
    ], guard)
    transported = push_spatial_hadamards(nodes, edges, actions=[action])
    assert transported == [bloq.Action.feedback([
        bloq.FeedbackTarget('Z', nodes['b']['position'], direction='+X'),
        bloq.FeedbackTarget('X', nodes['c']['position'], direction='+Z'),
    ], guard)]


@pytest.mark.parametrize('action', [
    bloq.Action.feedback([bloq.FeedbackTarget('Y', (1, 0, 1), direction='+Z')]),
    bloq.Action.feedback([bloq.FeedbackTarget('X', (1, 0, 1))]),
    bloq.Action.measure(bloq.MeasureTarget.edge((1, 0, 1), '+Z'), 'readout'),
])
def test_unproved_action_frames_stay_fixed_during_hadamard_push(action):
    nodes, edges = turn()
    source = copy.deepcopy(nodes)
    assert push_spatial_hadamards(nodes, edges, actions=[action]) == [action]
    assert nodes == source


@pytest.mark.parametrize("transpose", [False, True])
def test_anchored_mixed_walls_retain_one_spatial_axis_and_the_identity_channel(transpose):
    # Two fixed resource frames prevent removing both walls, but the free
    # corner can move one wall onto its temporal input without changing either.
    frames = [
        ("in", (0, 0, -1), "XZX", dict(type="input", qubit=0)),
        ("idle", (0, 0, 0), "XZX", None),
        ("a", (0, 0, 1), "XZX", None),
        ("b", (-1, 0, 1), "XXZ", None),
        ("c", (-1, 1, 1), "ZXX", None),
        ("d", (-1, 2, 1), "ZXX", None),
        ("out", (-1, 2, 2), "ZXX", dict(type="output", qubit=0)),
    ]
    nodes = {n: dict(position=(p[1], p[0], p[2]) if transpose else p,
                     tqec=kind[1] + kind[0] + kind[2] if transpose else kind, other=other)
             for n, p, kind, other in frames}
    edges = list(zip(["in", "idle", "a", "b", "c", "d"], ["idle", "a", "b", "c", "d", "out"]))
    source = copy.deepcopy(nodes)
    # When every frame is fixed, neither choice of remaining axis is possible.
    push_spatial_hadamards(nodes, edges, fixed={"a", "b", "c", "d"})
    assert nodes == source
    push_spatial_hadamards(nodes, edges, fixed={"b", "d"})
    assert nodes["b"] == source["b"] and nodes["d"] == source["d"]
    graph, _, _ = build_graph(nodes, edges, 1, [])
    assert sorted(hadamard_axes(graph)) == ["X" if transpose else "Y", "Z"]
    program = bloq.compile(graph, distance=3)
    for state, expected in [("zero", [0, 0, 1]), ("plus", [1, 0, 0])]:
        actual = bloq.run_noiseless(program, input_state=state)["samples"][0]["logical_bloch"][0]
        assert actual == pytest.approx(expected, abs=1e-8)
    again = copy.deepcopy(nodes)
    push_spatial_hadamards(again, edges, fixed={"b", "d"})
    assert again == nodes


@pytest.mark.parametrize('frame', ('XZX', 'ZXX', 'ZXZ', 'XZZ'))
@pytest.mark.parametrize('sign', (1, -1))
@pytest.mark.parametrize('blocked', (False, True))
def test_shared_t_occupancy_preserves_candidate_contents_and_order(frame, sign, blocked):
    from itertools import islice
    from bloq.interop._geometry import _t_placements

    axis = 0 if frame in ('XZX', 'ZXZ') else 1
    nodes = resource_nodes(frame)
    nodes['resource']['position'] = tuple(sign if i == axis else 0 for i in range(3))
    # Occupancy at other times still reserves factory columns.
    for x in range(-4, 5):
        for y in range(-4, 5):
            nodes[f'bulk_{x}_{y}'] = dict(position=(x, y, 10), tqec='ZXX', other=None)
    if blocked:
        nodes['ray'] = dict(position=tuple(2*sign if i == axis else 0 for i in range(3)), tqec=frame, other=None)
        nodes['factory'] = dict(position=(2, 2, -10), tqec='T', other=None)
    edges = [('root', 'resource')]
    positions = {n: d['position'] for n, d in nodes.items()}
    expected = list(islice(_t_placements(nodes, edges, 'resource', bends=True), 500))
    assert expected
    actual = list(islice(_t_placements(nodes, edges, 'resource', bends=True,
                                    positions=positions, occupied=frozenset(positions.values())), 500))
    assert actual == expected
