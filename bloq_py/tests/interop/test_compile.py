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


@pytest.mark.parametrize("resource_count", [1, 2])
def test_unactionable_owned_conflict_tries_cached_alternative(
        two_resources, native_error, resource_count):
    args = list(two_resources)
    if resource_count == 1:
        args[0].pop('r1')
        args[1].remove(('w5', 'r1'))
        args[4] = args[4][:1]
    compilation = _Compilation(3, True, None, 2)
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
    compilation.build(*args)
    assert len(calls) == 2 and calls[0] != calls[1]
    strategies = [attempt['strategy'] for attempt in compilation.attempts]
    assert strategies == (['short_arm', 'reuse'] if resource_count == 1 else ['reuse', 'short_arm'])
