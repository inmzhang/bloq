"""Noiseless physical execution and bounded logical-state inspection."""

import pytest

import bloq


def test_ideal_t_source_returns_corrected_joint_signature():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3, prepare_t_with_mpps=True)
    report = bloq.run_noiseless(program, shots=3, seed=17, input_state="plus", include_signature=True)
    assert report["execution"] == "noiseless"
    assert report["ideal_boundary_capture"]
    assert report["discarded"] == 0
    assert report["all_detectors_constant"]
    assert report["peak_rank"] > 1
    assert len(report["samples"]) == 3
    for sample in report["samples"]:
        assert sample["logical_signature"] == pytest.approx([2**-0.5, 2**-0.5, 0], abs=1e-8)
        assert sample["logical_bloch"][0] == pytest.approx(sample["logical_signature"], abs=1e-8)


def test_noiseless_input_state_and_optional_signature():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3, prepare_t_with_mpps=True)
    report = bloq.run_noiseless(program, input_state="zero")
    assert report["samples"][0]["logical_bloch"][0] == pytest.approx((0, 0, 1), abs=1e-8)
    assert report["samples"][0]["logical_signature"] is None
    with pytest.raises(bloq.InvalidArgumentError, match="positive"):
        bloq.run_noiseless(program, shots=0)
    with pytest.raises(bloq.InvalidArgumentError, match="input_state"):
        bloq.run_noiseless(program, input_state="one")


def test_joint_signature_limit_rejects_before_physical_execution():
    graph = bloq.BlockGraph()
    for wire in range(7):
        for z, kind, role in ((0, "Port", "input"), (1, "ZXX", "auto"), (2, "Port", "output")):
            graph.add_block(bloq.Block((3 * wire, 0, z), kind, role=role))
        for z in (0, 1):
            graph.add_pipe(bloq.Pipe((3 * wire, 0, z), "+Z"))
    program = bloq.compile(graph, distance=3)
    with pytest.raises(bloq.InvalidArgumentError, match="six logical outputs"):
        bloq.run_noiseless(program, include_signature=True)
