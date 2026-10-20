"""Shared geometry must reject malformed layouts before identity subdivision."""

import pytest
import bloq

from bloq.interop._geometry import GeometryError, _compact_temporal_layers, _space_diagram, add_t_resource, add_t_resources, build_graph


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
    actions = add_t_resource(nodes, edges, 'resource', phase='1/4', name='t')
    assert nodes['resource']['position'] == (2, 0, 0)
    assert nodes['t_arm_0'] == dict(position=(1, 0, 0), tqec='XZX', other=None)
    assert ('root', 't_arm_0') in edges and ('t_arm_0', 'resource') in edges
    assert actions[0].target.pos == (1, 0, 0)
    assert str(actions[0].target.direction) == '+X'


def test_bent_t_arm_can_use_the_opposite_perpendicular_side():
    nodes = resource_nodes('ZXZ')
    nodes['obstacle'] = dict(position=(1, 1, 0), tqec='ZXX', other=None)
    actions = add_t_resource(nodes, [('root', 'resource')], 'resource', phase='7/4', name='t')
    assert nodes['resource']['position'] == (1, 0, 0)
    assert nodes['t_center']['position'] == (1, -1, 1)
    assert str(actions[0].target.direction) == '-Y'
    assert str(actions[1].condition) == 't'


@pytest.mark.parametrize("joint", [False, True])
def test_t_placement_rejects_a_blocked_ray_without_partial_mutation(joint):
    from copy import deepcopy

    nodes = resource_nodes()
    for n, p in [('cap', (1, 0, 1)), ('ray', (2, 0, 0)),
                 ('left', (1, -1, 0)), ('right', (1, 1, 0))]:
        nodes[n] = dict(position=p, tqec='ZXX', other=None)
    edges = [('root', 'resource')]
    before = deepcopy(nodes)
    with pytest.raises(GeometryError) as error:
        if joint:
            add_t_resources(nodes, edges, [('resource', '1/4', 't')])
        else:
            add_t_resource(nodes, edges, 'resource', phase='1/4', name='t')
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

        def bounded_candidates(*args, bends=False, **kwargs):
            if not bends:
                return
            for index, candidate in enumerate(original(*args, bends=True, **kwargs)):
                assert index < 4, 'A solved layout must not enumerate unused alternatives'
                yield candidate

        monkeypatch.setattr(_geometry, 'add_t_resource', blocked_greedy)
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


def test_t_factories_do_not_alias_later_resources_or_data_columns():
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


@pytest.mark.parametrize('z', [-(2**31), 2**31 - 1])
def test_t_endpoints_reject_coordinate_overflow_without_mutation(z):
    nodes = resource_nodes()
    nodes = {n: dict(d, position=(*d['position'][:2], z)) for n, d in nodes.items()}
    edges = [('root', 'resource')]
    before = nodes.copy()
    with pytest.raises(GeometryError, match='signed 32-bit range'):
        add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert nodes == before and edges == [('root', 'resource')]


def test_initial_joint_domains_respect_the_candidate_budget_without_mutation(monkeypatch):
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

    monkeypatch.setattr(_geometry, 'add_t_resource', blocked_greedy)
    monkeypatch.setattr(_geometry, '_t_placements', candidates)
    with pytest.raises(GeometryError, match='T placement exceeded 100000 candidates'):
        add_t_resources(nodes, edges, [('resource', '1/4', 't')])
    assert nodes == before and edges == [('root', 'resource')]


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
