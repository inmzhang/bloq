"""Dynamic VM lowering and execution through the small Python surface."""

import json
import math

import pytest

import bloq


@pytest.fixture(scope="module")
def vm():
    # Ordinary VM behavior uses compact magic preparation; cultivation is tested separately.
    ir = bloq.compile(bloq.GalleryItem.THTH.load(), distance=3, prepare_t_with_mpps=True)
    return bloq.lower_vm(
        ir,
        source_release_time=23,
        factory_release_time=7,
    )


def test_lower_once_run_and_inspect(vm):
    sources = vm.sources
    assert json.dumps(sources) == json.dumps(vm.sources)
    assert {source["role"] for source in sources} >= {
        "logical_input",
        "prepared_y",
        "clifford",
    }
    assert all(source["role"] != "factory" for source in sources)
    assert all(
        source["release"] == 23
        for source in sources
        if source["role"] in {"logical_input", "clifford"}
    )
    lowered = json.loads(vm.to_json())
    assert json.loads(vm.to_json(indent=2)) == lowered
    quantum = [
        task["instruction"]["Quantum"]
        for task in lowered["tasks"]
        if "Quantum" in task["instruction"]
    ]
    assert quantum
    assert any(
        moment["operations"]
        for task in quantum
        for alternative in task["alternatives"]
        for moment in alternative["stream"]["moments"]
    )
    assert any(
        task["source"] == "clifford" and task["release"] == 23
        for task in lowered["tasks"]
    )

    shot = vm.run(
        seed=17,
        input_state="plus",
        decoder_acceptance=1,
        accepted_accuracy=1,
        rejected_accuracy=1,
    )
    trace = shot.trace
    assert not shot.discarded
    assert trace["metadata"]["seed"] == 17
    assert trace["events"] and trace["timing"]
    assert shot.finished_at >= max(span["end"] for span in trace["timing"])
    assert shot.peak_rank > 1
    assert json.loads(shot.to_json()) == trace
    assert json.loads(shot.to_json(indent=2)) == trace
    assert shot.logical_bloch() == pytest.approx((math.sqrt(0.5), 0.5, 0.5))


def test_input_release_and_idle_rate_units():
    ir = bloq.compile(bloq.GalleryItem.THTH.load(), distance=3, prepare_t_with_mpps=True)
    timed = bloq.lower_vm(
        ir,
        noise=0.002,
        gate_duration=0.25,
        decoder_latency_rounds=3,
        source_release_time=23,
        input_release_time=11,
        factory_release_time=7,
    )
    logical_input = next(source for source in timed.sources if source["role"] == "logical_input")
    assert logical_input["release"] == 11
    assert timed.decoder_latency_rounds == 3

    derived = timed.run(seed=17)
    trace = derived.trace
    assert trace["metadata"]["decoder_latency_rounds"] == 3
    assert trace["metadata"]["idle_error_rate"] == pytest.approx(0.008)
    ordinary_deadlines = [
        event
        for event in trace["events"]
        if event["kind"] == "decoder_deadline" and not event["factory"]
    ]
    assert ordinary_deadlines
    assert all(
        event["ready_at"] == pytest.approx(
            max(event["requested_at"], event["measurements_ready_at"] + 4.5)
        )
        for event in ordinary_deadlines
    )
    assert any(
        event["ready_at"] - event["measurements_ready_at"] == pytest.approx(4.5)
        for event in ordinary_deadlines
    )
    overridden = timed.run(seed=18, idle_error_rate=1.25)
    assert overridden.trace["metadata"]["idle_error_rate"] == 1.25


def test_native_cultivation_factory_release_gap_and_retry_discard():
    # This test specifically exercises native cultivation and its RUS protocol.
    ir = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3, prepare_t_with_mpps=False)
    native = bloq.lower_vm(
        ir, gate_duration=0.25, decoder_latency_rounds=3, factory_release_time=7,
    )
    factories = [source for source in native.sources if source["role"] == "factory"]
    assert len(factories) == 1
    assert factories[0]["release"] == 7
    accepted = native.run(seed=17, decoder_acceptance=1, accepted_accuracy=1, rejected_accuracy=1)
    assert not accepted.discarded
    factory_gap = [
        event
        for event in accepted.trace["events"]
        if event["kind"] == "memory_round" and event["memory_kind"] == "factory_gap"
    ]
    assert {event["round"] for event in factory_gap} == {0, 1, 2}
    for task in {event["task"] for event in factory_gap}:
        rounds = [event["round"] for event in factory_gap if event["task"] == task]
        assert rounds == [0, 1, 2] * (len(rounds) // 3)
    discarded = native.run(
        seed=18, accepted_accuracy=0, rejected_accuracy=0, max_attempts=1,
    )
    assert discarded.discarded
    with pytest.raises(bloq.RuntimeError):
        discarded.logical_bloch()


def test_vm_arguments_are_checked(vm):
    ir = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.lower_vm(ir, noise=-0.1)
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.lower_vm(ir, gate_duration=1e-9)
    thth = bloq.compile(bloq.GalleryItem.THTH.load(), distance=3, prepare_t_with_mpps=True)
    with pytest.raises(bloq.LowerError):
        bloq.lower_vm(thth, max_quantum_variants=1)
    with pytest.raises(bloq.InvalidArgumentError):
        vm.run(input_state="one")
    with pytest.raises(bloq.InvalidArgumentError):
        vm.run(max_steps=0)
    with pytest.raises(bloq.RuntimeError):
        vm.run(max_steps=1)
    shot = vm.run()
    with pytest.raises(bloq.InvalidArgumentError):
        shot.to_json(indent=4)
    with pytest.raises(bloq.InvalidArgumentError):
        vm.to_json(indent=4)
    with pytest.raises(bloq.InvalidArgumentError):
        shot.logical_bloch(output=vm.output_count)
