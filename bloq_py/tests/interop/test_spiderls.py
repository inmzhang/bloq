"""SpiderLS's exact provenance and event DAG; native channels live in bloq_vm."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

execution = pytest.importorskip('pyzx_execution_order', reason='optional pinned SpiderLS checkout')

from bloq.interop.spiderls import SpiderLSImportError, _convert, from_qasm  # noqa: E402
from bloq.interop._geometry import GeometryError  # noqa: E402


def source(gates, width=1):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}\n'


@pytest.mark.parametrize('gates,code', [
    ('rz(pi/8) q[0];', 'unsupported_phase'),
    ('rz(0.7853981633974483) q[0];', 'unsupported_angle'),
    ('measure q[0] -> c[0];', 'classical_control'),
    ('reset q[0];', 'classical_control'),
    ('if(c==1) x q[0];', 'classical_control'),
    ('h q[1];', 'qasm_operand'),
])
def test_unsupported_source_rejects_before_reduction(gates, code):
    with pytest.raises(SpiderLSImportError) as error:
        from_qasm(source(gates))
    assert error.value.code == code


def test_unpatched_pi_phase_cannot_be_silently_erased(monkeypatch):
    monkeypatch.setattr(execution, '_phase_gate_label', lambda _: None)
    with pytest.raises(SpiderLSImportError, match='upstream_patch'):
        from_qasm(source('z q[0];'))


def test_zero_time_hzh_uses_dependencies_instead_of_event_array_order():
    imported = from_qasm(source('h q[0]; z q[0]; h q[0];'))
    events = imported.schedule.zero_time_events
    assert {e.after_time for e in events} == {0}
    assert [phase for _, phase in imported.plan.spider_phases] == [Fraction(1)]
    reversed_schedule = replace(imported.schedule, zero_time_events=tuple(reversed(events)))
    graph, inputs, outputs = _convert(imported.reduced_graph, imported.target, reversed_schedule, imported.routing)
    graph.validate()
    assert inputs == imported.input_ports
    assert outputs == imported.output_ports
    # Compilation must preserve the adapter's declared interface.
    import bloq

    program = bloq.compile(graph, distance=3, prepare_t_with_mpps=True, validate=True)
    assert {output.port for output in program.logical_outputs()} == set(outputs)


@pytest.mark.parametrize('phase', ['pi/4', '-pi/4', '3*pi/4', '5*pi/4'])
def test_signed_magic_phase_survives_every_upstream_stage(phase):
    imported = from_qasm(source(f'rz({phase}) q[0];'))
    measurements = [m for slot in imported.schedule.slots for m in slot.measurements]
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.exact_phase == imported.target.instructions[measurement.target_instruction].exact_phase
    assert measurement.exact_phase in dict(imported.plan.spider_phases).values()
    assert sum(str(b.kind) == 'T' for b in imported.graph.flatten().blocks()) == 1


def test_corrupted_magic_provenance_rejects():
    imported = from_qasm(source('t q[0];'))
    slot, = imported.schedule.slots
    measurement, = slot.external
    bad = replace(measurement, exact_phase=Fraction(7, 4))
    schedule = replace(imported.schedule, slots=(replace(slot, external=(bad,)),))
    with pytest.raises(SpiderLSImportError, match='source_provenance'):
        _convert(imported.reduced_graph, imported.target, schedule, imported.routing)


def test_execution_plan_cannot_erase_a_source_phase(monkeypatch):
    extract = execution.extract_execution_order
    monkeypatch.setattr(execution, 'extract_execution_order',
                        lambda graph: replace(extract(graph), spider_phases=(), phase_gates=()))
    with pytest.raises(SpiderLSImportError, match='source_provenance'):
        from_qasm(source('s q[0];'))


@pytest.mark.parametrize('gate', ['s', 't', 'z'])
def test_schedule_cannot_drop_a_phase_instrument(gate):
    imported = from_qasm(source(f'{gate} q[0];'))
    schedule = replace(imported.schedule, slots=(), zero_time_events=tuple(
        e for e in imported.schedule.zero_time_events if e.kind != 'Z'))
    with pytest.raises(SpiderLSImportError, match='source_provenance'):
        _convert(imported.reduced_graph, imported.target, schedule, imported.routing)


def test_magic_parity_cannot_move_to_another_data_patch():
    imported = from_qasm(source('t q[0];'))
    slot, = imported.schedule.slots
    measurement, = slot.external
    bad = replace(measurement, operands=('q1', measurement.operands[1]))
    schedule = replace(imported.schedule, slots=(replace(slot, external=(bad,)),))
    with pytest.raises(SpiderLSImportError, match='source_provenance'):
        _convert(imported.reduced_graph, imported.target, schedule, imported.routing)


@pytest.mark.parametrize('duplicate', [False, True])
def test_each_scheduled_measurement_is_routed_exactly_once(duplicate):
    imported = from_qasm(source('t q[0];'))
    layer, = imported.routing.layers
    connections = (*layer.connections, *layer.connections) if duplicate else ()
    routing = replace(imported.routing, layers=(replace(layer, connections=connections),))
    with pytest.raises(GeometryError, match='routed measurements'):
        _convert(imported.reduced_graph, imported.target, imported.schedule, routing)


def test_zero_time_dependency_cycle_rejects_with_adapter_diagnostic():
    imported = from_qasm(source('h q[0]; z q[0]; h q[0];'))
    events = sorted(imported.schedule.zero_time_events, key=lambda e: e.dag_node_id)
    events[0] = replace(events[0], dependency_node_ids=(events[-1].dag_node_id,))
    schedule = replace(imported.schedule, zero_time_events=tuple(events))
    with pytest.raises(GeometryError, match='dependency cycle'):
        _convert(imported.reduced_graph, imported.target, schedule, imported.routing)


def test_identity_and_hadamard_wires_survive_reduction():
    imported = from_qasm(source('h q[1];', 2))
    assert len(imported.target.ignored_direct_io) == 2
    assert len(imported.input_ports) == len(imported.output_ports) == 2
    assert not imported.routing.layers


def test_reused_physical_patch_preserves_original_source_interface():
    fixture = Path(__file__).resolve().parents[3] / 'bloq_vm/tests/data/spiderls/lifecycle.qasm'
    imported = from_qasm(fixture.read_text())
    plus = [event.patch for event in imported.schedule.zero_time_events if event.kind == 'PREPARE_PLUS']
    assert len(plus) > len(set(plus)), 'fixture must exercise a reused physical patch ID'
    assert len(imported.input_ports) == len(imported.output_ports) == 2
    assert set(imported.input_ports).isdisjoint(imported.output_ports)
