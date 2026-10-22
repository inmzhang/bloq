"""Native-certified resource reuse and failure propagation, without upstream dependencies."""

from pathlib import Path
from importlib import import_module

import bloq
import pytest

from bloq.interop._compile import _Compilation
from bloq.interop._geometry import GeometryError


@pytest.fixture
def two_resources():
    nodes = {
        f'w{z}': dict(position=(0, 0, z), tqec='XZX',
                     other=dict(type='input' if z == 0 else 'output', qubit=0)
                     if z in (0, 7) else None)
        for z in range(8)
    }
    edges = [(f'w{z}', f'w{z + 1}') for z in range(7)]
    resources = []
    for i, z in enumerate((1, 5)):
        nodes[f'r{i}'] = dict(position=(1, 0, z), tqec=None, other='T')
        edges.append((f'w{z}', f'r{i}'))
        resources.append((f'r{i}', '1/4' if i == 0 else '7/4', f't{i}'))
    return nodes, edges, 1, [], resources


@pytest.fixture
def native_error():
    source = Path(__file__).resolve().parents[3] / 'bloq_compile/tests/data/late_control_t.blog'
    graph = bloq.BlockGraph.from_text(source.read_text())
    with pytest.raises(bloq.CompileError, match='reuses live qubit') as caught:
        bloq.compile(graph, distance=3, prepare_t_with_mpps=True)
    return caught.value


def test_native_compiler_accepts_reusing_one_factory_column(two_resources):
    compilation = _Compilation(3, True, None, 2)
    graph, _, outputs = compilation.build(*two_resources)
    sources = [block.pos for block in graph.flatten().blocks() if block.kind.is_t]
    assert len(sources) == 2 and len({p[:2] for p in sources}) == 1
    assert [attempt['status'] for attempt in compilation.attempts] == ['accepted']
    assert {output.port for output in compilation.program.logical_outputs()} == set(outputs)


@pytest.mark.parametrize('compiler', ['topols', 'spiderls'])
@pytest.mark.parametrize('gate', ['h', 't'])
def test_public_compile_preserves_import_when_no_factory_reuse_is_needed(compiler, gate):
    pytest.importorskip('topols' if compiler == 'topols' else 'pyzx_routing')
    from bloq.interop import compile_qasm

    qasm = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n{gate} q[0];\n'
    expected = import_module(f'bloq.interop.{compiler}').from_qasm(qasm)
    result = compile_qasm(qasm, compiler=compiler, prepare_t_with_mpps=True)
    assert result.imported.graph.to_text() == expected.graph.to_text()
    assert {output.port for output in result.program.logical_outputs()} == set(expected.output_ports)


def test_unrelated_native_failure_is_propagated_without_retry(two_resources, native_error):
    compilation = _Compilation(3, True, None, 2)

    class Reject:
        def compile(self, graph, **options):
            raise native_error

    compilation.context = Reject()
    with pytest.raises(bloq.CompileError) as caught:
        compilation.build(*two_resources)
    assert caught.value is native_error
    assert len(compilation.attempts) == 1


def test_resource_ownership_includes_routed_arms_after_time_compaction(
        two_resources, native_error, monkeypatch):
    from bloq.interop import _compile

    build = _compile.build_graph
    mapped = {}

    def capture(nodes, *args, **options):
        result = build(nodes, *args, **options)
        mapped.update({name: options['position_map'][tuple(data['position'])]
                       for name, data in nodes.items()
                       if tuple(data['position']) in options['position_map']})
        return result

    monkeypatch.setattr(_compile, 'build_graph', capture)
    compilation = _Compilation(3, True, None, 2)
    native, calls = compilation.context, []

    class ArmConflict:
        def compile(self, graph, **options):
            calls.append(graph.to_text())
            if len(calls) == 1:
                # This arm lies outside the source/center/cap column. A native
                # witness there must still trigger resource placement refinement.
                arm = next(p for name, p in mapped.items() if str(name).startswith('t0_arm_'))
                assert arm[:2] != mapped['t0_source'][:2]
                native_error.layout_conflict = {
                    'kind': 'live_qubit_reuse', 'qubit': (0, 0),
                    'producer': {'id': 0, 'blocks': [arm]},
                    'consumer': {'id': 1, 'blocks': []},
                    'reuser': {'id': 2, 'blocks': [mapped['t1_source']]},
                }
                raise native_error
            return native.compile(graph, **options)

    compilation.context = ArmConflict()
    compilation.build(*two_resources)
    assert [attempt['status'] for attempt in compilation.attempts] == ['rejected', 'accepted']
    assert len(calls) == 2


@pytest.mark.parametrize('budget,routing_failure,indirect',
                         [(1, False, False), (2, False, False), (2, True, False), (2, False, True)])
def test_structured_conflict_refines_placement_or_preserves_native_cause(
        two_resources, native_error, budget, routing_failure, indirect, monkeypatch):
    from bloq.interop import _compile

    compilation = _Compilation(3, True, None, budget)
    context = compilation.context
    calls = []
    original = _compile._maze_t_placements

    def route(*args, **kwargs):
        if routing_failure and calls:
            raise GeometryError('resource_placement_limit', 'Rerouting exhausted its budget')
        return original(*args, **kwargs)

    monkeypatch.setattr(_compile, '_maze_t_placements', route)
    if routing_failure:
        def no_short_arm(*args):
            raise GeometryError('resource_collision', 'No short arm')
        monkeypatch.setattr(_compile, '_greedy_t_resources', no_short_arm)

    class ConflictOnce:
        def compile(self, graph, **options):
            sources = sorted(block.pos for block in graph.flatten().blocks() if block.kind.is_t)
            calls.append(sources)
            if len(calls) == 1:
                # Model the native certificate for an interrupted first cap.
                # Its producer and consumer belong to one resource, the reset
                # to the other. All coordinates refer to the final graph.
                cap = next(block.pos for block in graph.flatten().blocks()
                           if str(block.kind) == 'XY' and block.pos[:2] == sources[0][:2]
                           and block.pos[2] > sources[0][2])
                native_error.layout_conflict = {
                    'kind': 'live_qubit_reuse', 'qubit': native_error.layout_conflict['qubit'],
                    'producer': {'id': 0, 'blocks': [sources[0]]},
                    'consumer': {'id': 1, 'blocks': [cap]},
                    'reuser': {'id': 2, 'blocks': [] if indirect else [sources[1]]},
                }
                raise native_error
            return context.compile(graph, **options)

    compilation.context = ConflictOnce()
    if budget == 1 or routing_failure:
        with pytest.raises(GeometryError, match='exhausted') as caught:
            compilation.build(*two_resources)
        assert caught.value.__cause__ is native_error
        assert compilation.program is None
    else:
        compilation.build(*two_resources)
        assert [attempt['status'] for attempt in compilation.attempts] == ['rejected', 'accepted']
        assert len({p[:2] for p in calls[1]}) == 2


@pytest.mark.parametrize('budget', [1, 2])
@pytest.mark.parametrize("resource_count", [1, 2])
def test_unactionable_owned_conflict_tries_cached_alternative(
        two_resources, native_error, resource_count, budget):
    args = list(two_resources)
    if resource_count == 1:
        args[0].pop('r1')
        args[1].remove(('w5', 'r1'))
        args[4] = args[4][:1]
    compilation = _Compilation(3, True, None, budget)
    native, calls = compilation.context, []

    class RejectFirstCandidate:
        def compile(self, graph, **options):
            calls.append(graph.to_text())
            if len(calls) == 1:
                source = next(block.pos for block in graph.flatten().blocks() if block.kind.is_t)
                native_error.layout_conflict = {
                    'kind': 'live_qubit_reuse', 'qubit': (0, 0),
                    'producer': {'id': 0, 'blocks': [source]},
                    'consumer': {'id': 1, 'blocks': []},
                    'reuser': {'id': 2, 'blocks': []},
                }
                raise native_error
            return native.compile(graph, **options)

    compilation.context = RejectFirstCandidate()
    if budget == 1:
        with pytest.raises(GeometryError) as caught:
            compilation.build(*args)
        assert caught.value.code == 'resource_placement_limit'
        assert caught.value.__cause__ is native_error
        assert compilation.program is None
        return
    compilation.build(*args)
    assert len(calls) == 2 and calls[0] != calls[1]
    strategies = [attempt['strategy'] for attempt in compilation.attempts]
    assert strategies == (['short_arm', 'reuse'] if resource_count == 1 else ['reuse', 'short_arm'])


@pytest.mark.parametrize('budget,stall', [(2, None), (3, 'routing'), (3, 'aliases'), (2, 'native'), (8, 'native')])
def test_owned_conflict_can_fall_back_to_native_validated_private_columns(
        two_resources, native_error, budget, stall, monkeypatch):
    from bloq.interop import _compile

    compilation = _Compilation(3, True, None, budget)
    native, routes, calls = compilation.context, [], []
    clock = [0.0]
    monkeypatch.setattr(_compile, 'perf_counter', lambda: clock[0])
    original = _compile._maze_t_placements

    def no_short_arm(*args):
        raise GeometryError('resource_collision', 'No short arm')

    def route(*args, **options):
        routes.append(options.get('conflicts'))
        if options.get('conflicts') is None:
            clock[0] += 10
        if stall == 'routing' and calls and options.get('conflicts') is not None:
            raise GeometryError('resource_placement_limit', 'Reuse routing exhausted its budget')
        return original(*args, **options)

    monkeypatch.setattr(_compile, '_greedy_t_resources', no_short_arm)
    monkeypatch.setattr(_compile, '_maze_t_placements', route)

    class RejectReusedColumns:
        def compile(self, graph, **options):
            clock[0] += 1
            sources = sorted(block.pos for block in graph.flatten().blocks() if block.kind.is_t)
            calls.append(sources)
            if len(calls) == 1 or stall == 'native' or (stall == 'aliases' and len(calls) == 2):
                native_error.layout_conflict = {
                    'kind': 'live_qubit_reuse', 'qubit': (0, 0),
                    'producer': {'id': 0, 'blocks': [sources[0]]},
                    'consumer': {'id': 1, 'blocks': []},
                    'reuser': {'id': 2, 'blocks': [sources[1]]},
                }
                raise native_error
            return native.compile(graph, **options)

    compilation.context = RejectReusedColumns()
    if stall == 'native':
        with pytest.raises(GeometryError) as caught:
            compilation.build(*two_resources)
        assert caught.value.code == 'resource_placement_limit'
        assert caught.value.__cause__ is native_error
        assert compilation.program is None
        if budget == 8:
            assert len(calls) < budget
    else:
        compilation.build(*two_resources)
        assert compilation.attempts[-1]['status'] == 'accepted'
        assert compilation.program is not None
    private = next(attempt for attempt in compilation.attempts if attempt['strategy'] == 'private')
    assert private['layout_seconds'] == 10
    assert private['seconds'] == 11
    assert private['native_seconds'] == 1
    assert None in routes
    assert len(calls) <= budget
    assert len({p[:2] for p in calls[-1]}) == 2


def test_persistent_waiter_gets_an_exclusive_outlet_while_other_factories_share(
        two_resources, native_error, monkeypatch):
    from bloq.interop import _compile

    nodes, edges, qubits, actions, resources = two_resources
    output = nodes['w7']['other']
    nodes['w7']['other'] = None
    for z in range(8, 16):
        nodes[f'w{z}'] = dict(position=(0, 0, z), tqec='XZX', other=output if z == 15 else None)
        edges.append((f'w{z - 1}', f'w{z}'))
    for i, z in ((2, 9), (3, 13)):
        nodes[f'r{i}'] = dict(position=(1, 0, z), tqec=None, other='T')
        edges.append((f'w{z}', f'r{i}'))
        resources.append((f'r{i}', '1/4', f't{i}'))

    def no_short_arm(*args):
        raise GeometryError('resource_collision', 'No short arm')

    monkeypatch.setattr(_compile, '_greedy_t_resources', no_short_arm)
    compilation = _Compilation(3, True, None, 4)
    native, sources = compilation.context, []

    class PersistentWait:
        def compile(self, graph, **options):
            current_sources = sorted((b.pos for b in graph.flatten().blocks() if b.kind.is_t), key=lambda p: p[2])
            sources.append({node: p for (node, _, _), p in zip(resources, current_sources)})
            if len(sources) <= 2:
                current, other = sources[-1], f'r{len(sources)}'
                assert current['r0'][:2] == current[other][:2]
                native_error.layout_conflict = {
                    'kind': 'live_qubit_reuse', 'qubit': (0, 0),
                    'producer': {'id': 0, 'blocks': [current['r0']]},
                    'consumer': {'id': 1, 'blocks': []},
                    'reuser': {'id': 2, 'blocks': [current[other]]},
                }
                raise native_error
            return native.compile(graph, **options)

    compilation.context = PersistentWait()
    compilation.build(nodes, edges, qubits, actions, resources)
    assert len(sources) == 3
    assert len({p[:2] for node, p in sources[-1].items() if node != 'r0'}) == 1
    assert sources[-1]['r0'][:2] != sources[-1]['r3'][:2]
    assert compilation.attempts[-1]['strategy'] == 'reuse'
