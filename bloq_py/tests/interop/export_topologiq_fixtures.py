"""Regenerate VM fixtures with the audited, patched real Topologiq checkout.

Run from the repository root with the optional interop environment:
    PYTHONPATH=/path/to/topologiq/src uv run --project bloq_py --group interop \
        python bloq_py/tests/interop/export_topologiq_fixtures.py \
        --upstream-root /path/to/topologiq

The checkout must contain exactly the checked-in patch on the audited revision.
"""

import argparse
import copy
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import runpy
from unittest.mock import patch

import topologiq

from bloq.interop._qasm import parse_qasm
from bloq.interop._topologiq_t import lower_t_gadgets
from bloq.interop.topologiq import AUDITED_REVISION, _CUBES, _convert_geometry, _direction, _routed_path, _from_qasm


def from_qasm(qasm):
    # Historical spatial-H snapshots remain useful signed-gadget regressions.
    return _from_qasm(qasm, seed=1330, temporal_hadamards_only=False)


def fixture_sources() -> dict[str, str]:
    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    circuits = {
        "t": (1, "t q[0];"),
        "tdg": (1, "tdg q[0];"),
        "thth": (1, "t q[0];\nh q[0];\nt q[0];\nh q[0];"),
        "entangled_t": (2, "h q[0];\ncx q[0],q[1];\nt q[1];"),
        "qft2": (
            2,
            "// Prepare |0+>, with q[0] the least significant bit.\n"
            "h q[0];\n"
            "// Positive-phase QFT, including the final bit-order swap.\n"
            "h q[1];\n"
            "t q[0];\nt q[1];\ncx q[0],q[1];\n"
            "tdg q[1];\ncx q[0],q[1];\nh q[0];\n"
            "cx q[0],q[1];\ncx q[1],q[0];\ncx q[0],q[1];",
        ),
        "qft2_x": (
            2,
            "x q[0];\nh q[1];\nt q[1];\ncx q[1],q[0];\n"
            "tdg q[0];\ncx q[1],q[0];\nt q[0];\nh q[0];\n"
            "cx q[0],q[1];\ncx q[1],q[0];\ncx q[0],q[1];",
        ),
    }
    return {
        name: f"{header}qreg q[{width}];\n{gates}\n"
        for name, (width, gates) in circuits.items()
    }


def _cut_t_frame(manager, source, root, frame, *, extend_input=False):
    from topologiq.core.blocks import ZXBlockRegistry

    nodes, originals = manager.bgraph.nodes, set(source.vertices())

    def kind(node):
        return nodes[node]["zx_block"].kind

    resource, cap = [edge[1] for edge in manager.t_gates[root]]
    resource_path, cap_path = [_routed_path(manager, root, end) for end in (resource, cap)]
    family = set(manager.twin_trace[root])
    core = family | set(resource_path) | set(cap_path)
    for first in family:
        for neighbor in manager.bgraph.neighbors(first):
            path = [first, neighbor]
            while path[-1] not in family:
                node = path[-1]
                if node in originals or kind(node) not in _CUBES or manager.bgraph.degree(node) != 2:
                    break
                following = next(v for v in manager.bgraph.neighbors(node) if v != path[-2])
                if following in path:
                    break
                path.append(following)
            else:
                core.update(path)
    previous, following = sorted(source.neighbors(root), key=source.row)
    incoming, outgoing = [_routed_path(manager, root, end) for end in (previous, following)]
    if extend_input:
        boundary = next(node for node in source.neighbors(previous) if node != root)
        if boundary not in source.inputs():
            raise RuntimeError(f"Frame {frame} no longer has the certified original input boundary")
        earlier = _routed_path(manager, previous, boundary)
        core.update(incoming)
        core.update(earlier[:-1])
        input_node = boundary
        input_edges = list(zip(incoming, incoming[1:])) + list(zip(earlier, earlier[1:]))
    else:
        input_node = incoming[1]
        input_edges = [tuple(incoming[:2])]
    ports = (input_node, outgoing[1])
    if set(ports) & core:
        raise RuntimeError(f"Frame {frame} does not admit an independent one-wire cut")
    local = copy.copy(manager)
    local.bgraph = manager.bgraph.subgraph(core | set(ports)).copy()
    local.inputs, local.outputs = [ports[0]], [ports[1]]
    local.t_gates, local.t_zx_tracker = {root: manager.t_gates[root]}, {root: cap}
    for node, attributes in local.bgraph.nodes(data=True):
        attributes["completions"] = {"degree": local.bgraph.degree(node), "pending": 0}
        if node in ports:
            attributes["zx_block"] = ZXBlockRegistry.get_create(kind="OOO")
    input_h = bool(sum(manager.bgraph.edges[edge]["edge_type"] == "HADAMARD" for edge in input_edges) % 2)
    output_h = manager.bgraph.edges[outgoing[0], outgoing[1]]["edge_type"] == "HADAMARD"
    return local, resource, cap, input_h, output_h


def _export_t_frame(manager, source, root, frame, source_name, qasm, destination, common, *, extend_input=False):
    local, resource, cap, input_h, output_h = _cut_t_frame(manager, source, root, frame, extend_input=extend_input)
    for dagger in (False, True):
        phase = Fraction(7 if dagger else 1, 4)
        signed_source = source.copy()
        signed_source.set_phase(root, phase)
        controls = copy.copy(manager)
        controls.t_gates, controls.t_zx_tracker = {root: manager.t_gates[root]}, {root: cap}
        special, actions = lower_t_gadgets(controls, signed_source, {root: phase})
        graph, _, inputs, outputs = _convert_geometry(local, special_kinds=special, actions=actions)
        sign = "tdg" if dagger else "t"
        name = f"{frame}_{sign}"
        gates = (["h"] if input_h else []) + [sign] + (["h"] if output_h else [])
        ideal = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n' + "".join(f"{gate} q[0];\n" for gate in gates)
        (destination / f"{name}.blog").write_text(graph.to_text().rstrip() + "\n")
        (destination / f"{name}.qasm").write_text(ideal)
        metadata = common | {
            "source_qasm": f"{name}.qasm", "graph": f"{name}.blog",
            "input_ports": inputs, "output_ports": outputs, "t_gadgets": 1,
            "origin": {"source": source_name, "source_sha256": sha256(qasm.encode()).hexdigest(), "root": root},
        }
        if extend_input:
            metadata["origin"].update({
                "input_source_node": local.inputs[0],
                "retained_upstream_nodes": sorted(local.bgraph.nodes),
            })
        (destination / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"Exported {name}: unchanged {frame} cut from real QPE layout")


def _route_keeping_manager(importer, qasm):
    from topologiq.core.graph_manager.graph_manager import BlockGraphManager

    retained, build = [], BlockGraphManager.build

    def retain(manager):
        build(manager)
        retained.append(manager)

    with patch.object(BlockGraphManager, "build", retain):
        imported = importer(qasm)
    manager, = retained
    return imported, manager


def export_t_frames(repository: Path, common: dict) -> None:
    """Cut proved one-wire frames from the completed real four-qubit layout."""
    generate = runpy.run_path(str(repository / "bloq_py/examples/topologiq_qpe.py"))["generate_phase_estimation"]
    qasm = generate().clifford_t_qasm
    source = parse_qasm(qasm).to_graph()
    imported, manager = _route_keeping_manager(from_qasm, qasm)
    full_destination = repository / "bloq_vm/tests/data/topologiq"
    (full_destination / "qpe4.qasm").write_text(qasm)
    (full_destination / "qpe4.blog").write_text(imported.graph.to_text().rstrip() + "\n")
    metadata = common | {
        "source_qasm": "qpe4.qasm", "graph": "qpe4.blog",
        "input_ports": imported.input_ports, "output_ports": imported.output_ports,
        "t_gadgets": sum(phase.denominator == 4 for phase in imported.source_phases.values()),
        "source_phases": {str(node): str(phase) for node, phase in imported.source_phases.items()},
    }
    (full_destination / "qpe4.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Exported qpe4: {metadata['t_gadgets']} signed T gadgets")
    nodes = manager.bgraph.nodes
    def position(node):
        return tuple(nodes[node]["coords"])

    def kind(node):
        return nodes[node]["zx_block"].kind
    selected = {}
    for root in sorted(manager.t_gates):
        resource, cap = [edge[1] for edge in manager.t_gates[root]]
        resource_path, cap_path = [_routed_path(manager, root, end) for end in (resource, cap)]
        spatial = [(a, b) for a, b in zip(resource_path[:-2], resource_path[1:-1])
                   if not _direction(position(a), position(b)).endswith("Z")]
        if not spatial:
            continue
        corner = spatial[-1][1]
        if kind(corner) in ("ZXX", "XZX"):
            selected.setdefault("z_neck", root)
        elif kind(corner) in ("XZZ", "ZXZ"):
            if resource_path[0] != cap_path[0]:
                selected.setdefault("split_x_neck", root)
            if corner != resource_path[-2]:
                selected.setdefault("padded_x_neck", root)
    if set(selected) != {"z_neck", "split_x_neck", "padded_x_neck"}:
        raise RuntimeError(f"Upstream layout no longer contains the proved T frames: {selected}")

    destination = repository / "bloq_vm/tests/data/topologiq/frames"
    destination.mkdir(exist_ok=True)
    for frame, root in selected.items():
        _export_t_frame(manager, source, root, frame, "qpe4.qasm", qasm, destination, common)


def export_direct_t_frames(repository: Path, common: dict) -> None:
    generate = runpy.run_path(str(repository / "bloq_py/examples/_algorithm_candidates.py"))["generate_candidate"]
    destination = repository / "bloq_vm/tests/data/topologiq/frames"
    selections = {
        "grover3": {11: "direct_twins_input", 47: "direct_tree_input"},
        "qaoa3": {8: "direct_split_input", 25: "direct_h_feedback"},
        "qaoa4": {35: "direct_cap_padding"},
    }
    for name, selected in selections.items():
        qasm = generate(name).clifford_t_qasm
        source = parse_qasm(qasm).to_graph()
        _, manager = _route_keeping_manager(from_qasm, qasm)
        for root, frame in selected.items():
            if len(_routed_path(manager, root, manager.t_gates[root][0][1])) != 2:
                raise RuntimeError(f"{name}:{root} no longer has the certified direct resource frame")
            _export_t_frame(manager, source, root, frame, f"{name}.qasm", qasm, destination, common)


def export_upstream_h_t_frames(repository: Path, common: dict) -> None:
    """Retain the real input H web for the public temporal-only QPE profile."""
    from bloq.interop.topologiq import from_qasm as public_from_qasm

    generate = runpy.run_path(str(repository / "bloq_py/examples/topologiq_qpe.py"))["generate_phase_estimation"]
    qasm = generate().clifford_t_qasm
    source = parse_qasm(qasm).to_graph()
    imported, manager = _route_keeping_manager(public_from_qasm, qasm)
    if imported.layout_options["temporal_hadamards_only"] is not True:
        raise RuntimeError("The input-H fixture must use the public temporal-only layout")
    full_destination = repository / "bloq_vm/tests/data/topologiq"
    name = "qpe4_temporal"
    (full_destination / f"{name}.blog").write_text(imported.graph.to_text().rstrip() + "\n")
    (full_destination / f"{name}.qasm").write_text(qasm)
    metadata = common | {
        "layout": imported.layout_options,
        "source_qasm": f"{name}.qasm", "graph": f"{name}.blog",
        "input_ports": imported.input_ports, "output_ports": imported.output_ports,
        "t_gadgets": sum(phase.denominator == 4 for phase in imported.source_phases.values()),
        "source_phases": {str(node): str(phase) for node, phase in imported.source_phases.items()},
        "source_sha256": sha256(qasm.encode()).hexdigest(),
    }
    (full_destination / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Exported {name}: complete public temporal-only QPE layout")
    destination = repository / "bloq_vm/tests/data/topologiq/frames"
    destination.mkdir(exist_ok=True)
    source_name = "qpe4_temporal.qasm"
    _export_t_frame(
        manager, source, 10, "direct_upstream_h", source_name, qasm, destination,
        common | {"layout": imported.layout_options}, extend_input=True,
    )


def export_algorithms(
    repository: Path, common: dict, *, names=("grover3", "adder4"),
) -> None:
    from bloq.interop.topologiq import from_qasm

    generate = runpy.run_path(str(repository / "bloq_py/examples/_algorithm_candidates.py"))["generate_candidate"]
    destination = repository / "bloq_vm/tests/data/topologiq/algorithms"
    destination.mkdir(exist_ok=True)
    for name in names:
        candidate = generate(name)
        imported = from_qasm(candidate.clifford_t_qasm)
        (destination / f"{name}.blog").write_text(imported.graph.to_text().rstrip() + "\n")
        (destination / f"{name}.qasm").write_text(candidate.clifford_t_qasm + "\n")
        metadata = common | {
            "layout": imported.layout_options,
            "input_ports": imported.input_ports, "output_ports": imported.output_ports,
            "t_gadgets": sum(phase.denominator == 4 for phase in imported.source_phases.values()),
            "algorithm": candidate.summary,
        }
        if name == "adder4":
            metadata["expected_compile_error"] = "LiveQubitReuse"
        else:
            metadata["expected_state"] = [[float(a.real), float(a.imag)] for a in candidate.ideal_unitary[:, 0]]
        (destination / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if name == "adder4":
            print(f"Exported {name}: unchanged layout retained for LiveQubitReuse rejection")
        else:
            print(f"Exported {name}: independent algorithm state and unchanged Topologiq layout")


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve()
    if Path(topologiq.__file__).resolve() != upstream / "src/topologiq/__init__.py":
        raise RuntimeError("Imported Topologiq does not come from --upstream-root")

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(upstream), *arguments], text=True,
        ).strip()

    revision = git("rev-parse", "HEAD")
    patch = repository / "bloq_py/examples/topologiq.patch"
    if revision != AUDITED_REVISION or git("diff", "HEAD") != patch.read_text().strip():
        raise RuntimeError("Upstream must match the audited revision plus exactly topologiq.patch")
    if git("diff", "--cached"):
        raise RuntimeError("Upstream checkout contains staged changes")

    destination = repository / "bloq_vm/tests/data/topologiq"
    destination.mkdir(parents=True, exist_ok=True)
    for name, qasm in fixture_sources().items():
        imported = from_qasm(qasm)
        imported.graph.validate()
        metadata = {
            "source_qasm": f"{name}.qasm",
            "graph": f"{name}.blog",
            "input_ports": imported.input_ports,
            "output_ports": imported.output_ports,
            "t_gadgets": sum(phase.denominator == 4 for phase in imported.source_phases.values()),
            "source_phases": {str(node): str(phase) for node, phase in imported.source_phases.items()},
            "upstream": {
                "repository": "https://github.com/tqec/topologiq",
                "revision": revision,
                "patch": "bloq_py/examples/topologiq.patch",
                "patch_sha256": sha256(patch.read_bytes()).hexdigest(),
            },
            "frontend": {
                "pyzx": version("pyzx"),
                "flow_module": "bloq.interop._topologiq_flow",
                "parser_module": "bloq.interop._qasm",
            },
            "layout": imported.layout_options,
        }
        (destination / f"{name}.qasm").write_text(qasm)
        (destination / f"{name}.blog").write_text(imported.graph.to_text().rstrip() + "\n")
        (destination / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"Exported {name}: {metadata['t_gadgets']} signed T gadgets")
    export_t_frames(repository, {key: metadata[key] for key in ("upstream", "frontend", "layout")})
    export_direct_t_frames(repository, {key: metadata[key] for key in ("upstream", "frontend", "layout")})
    export_upstream_h_t_frames(repository, {key: metadata[key] for key in ("upstream", "frontend", "layout")})
    export_algorithms(repository, {key: metadata[key] for key in ("upstream", "frontend", "layout")})


if __name__ == "__main__":
    main()
