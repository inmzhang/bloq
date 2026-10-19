"""Explicit source roles survive import in every straight-wire orientation."""

from types import SimpleNamespace

import pytest

import bloq
from bloq.interop.topologiq import _convert_geometry

nx = pytest.importorskip("networkx")
pytest.importorskip("topologiq", reason="opt-in pinned upstream environment")
from topologiq.core.blocks import ZXBlockRegistry  # noqa: E402


def wire_snapshot(axis, reversed_roles):
    """A minimal completed snapshot using the upstream's native block types."""
    graph = nx.Graph()
    cube = "XZZ" if axis == 1 else "ZXZ"
    for node in range(3):
        position = tuple(node if i == axis else 0 for i in range(3))
        graph.add_node(
            node, coords=position,
            zx_block=ZXBlockRegistry.get_create(kind=cube if node == 1 else "OOO"),
            completions={"degree": 2 if node == 1 else 1, "pending": 0},
        )
    frame = list(cube)
    frame[axis] = "O"
    for node in (0, 1):
        graph.add_edge(
            node, node + 1, kind="".join(frame), edge_type="SIMPLE",
            start_coords=graph.nodes[node]["coords"], end_coords=graph.nodes[node + 1]["coords"],
        )
    return SimpleNamespace(
        bgraph=graph, inputs=[2 if reversed_roles else 0], outputs=[0 if reversed_roles else 2],
    )


@pytest.mark.parametrize("axis", range(3), ids=["x", "y", "z"])
@pytest.mark.parametrize("reversed_roles", [False, True], ids=["forward", "reverse"])
def test_explicit_boundary_roles_preserve_geometry_and_compile(axis, reversed_roles):
    manager = wire_snapshot(axis, reversed_roles)
    graph, positions, inputs, outputs = _convert_geometry(manager, special_kinds={}, actions=[])
    assert positions == {node: data["coords"] for node, data in manager.bgraph.nodes(data=True)}
    assert inputs == (positions[manager.inputs[0]],)
    assert outputs == (positions[manager.outputs[0]],)
    flat = graph.flatten()
    assert flat.block_count == 3
    assert flat.get_block(inputs[0]).role == "input"
    assert flat.get_block(outputs[0]).role == "output"
    program = bloq.compile(graph, distance=3, prepare_t_with_mpps=True, validate=True)
    assert [output.port for output in program.logical_outputs()] == list(outputs)
    report = bloq.run_noiseless(program, shots=16, seed=17, include_signature=True)
    assert report["discarded"] == 0
    assert len(report["samples"]) == 16
    assert report["all_detectors_constant"]
    for sample in report["samples"]:
        assert sample["logical_signature"] == pytest.approx([1, 0, 0], abs=1e-9)
