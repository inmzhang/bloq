#!/usr/bin/env python3
"""Compile small constructions and render representative detector-slice moments.

Run: uv run --project bloq_py --no-sync --with stim python tools/render_construction_slices.py
The complete circuit is retained when drawing selected moments. No rounds are
removed to make a diagram shorter. Y-up coordinates follow the paper's
src/figures/generate_stim_diagrams.py; SVG composition uses the standard library.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import bloq
import stim


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


@dataclass
class Example:
    name: str
    family: str
    body: str
    position: tuple[int, int, int] = (0, 0, 0)
    distance: int | None = None

    @property
    def source(self) -> str:
        return "BLOG 1.0\n\nmodule main {\n" + "\n".join(
            "  " + line for line in self.body.splitlines()
        ) + "\n}\n"


def examples() -> list[Example]:
    cases = []
    for kind in ("ZXZ", "XZX", "XZZ", "ZXX"):
        axis = (1, 0) if kind in {"ZXZ", "XZX"} else (0, 1)
        for spatial, offsets in {"isolated": [], "one-arm": [1], "opposite-arms": [-1, 1]}.items():
            for temporal, (past, future) in {
                "closed": (False, False), "input": (True, False),
                "output": (False, True), "through": (True, True),
            }.items():
                # Keep the source origin at zero so compiler normalization
                # preserves the selected block's coordinates.
                x, y = axis if spatial == "opposite-arms" else (0, 0)
                z = int(past)
                lines = [f"0: {kind} [{x},{y},{z}]"]
                for i, offset in enumerate(offsets, 1):
                    lines += [f"{i}: {kind} [{x + offset*axis[0]},{y + offset*axis[1]},{z}]",
                              f"0 -> {i}"]
                for attached, role, dz in ((past, "in", -1), (future, "out", 1)):
                    if attached:
                        i = len(offsets) + (1 if role == "in" else 1 + int(past))
                        lines += [f"{role} q_{role}: data = {i}",
                                  f"{i}: Port [{x},{y},{z + dz}]",
                                  f"{i} -> 0" if role == "in" else f"0 -> {i}"]
                name = f"regular-{kind.lower()}"
                if spatial != "isolated" or temporal != "closed":
                    name += f"-{spatial}-{temporal}"
                cases.append(Example(name, "regular", "\n".join(lines), (x, y, z)))
    arms = {
        "isolated": [], "one-arm": [(1, 0)],
        "straight": [(1, 0), (-1, 0)],
        "elbow": [(1, 0), (0, -1)],
        "tee": [(1, 0), (-1, 0), (0, -1)],
        "cross": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    }
    for kind in ("ZZX", "XXZ"):
        for label, offsets in arms.items():
            lines = [f"0: {kind} [0,0,0]"]
            for i, (x, y) in enumerate(offsets, 1):
                neighbor = ("XZX" if x else "ZXX") if kind == "ZZX" else ("ZXZ" if x else "XZZ")
                lines += [f"{i}: {neighbor} [{x},{y},0]", f"0 -> {i}"]
            cases.append(Example(f"spatial-{kind.lower()}-{label}", "spatial", "\n".join(lines)))
    for orientation in ("ZXZ", "XZX"):
        for basis in ("X", "Z"):
            cases.append(Example(f"measurement-{basis.lower()}-{orientation.lower()}", "measurement",
                f"0: {orientation} [0,0,0]\n1: {basis} [0,0,1]\n0 -> +Z", (0, 0, 1)))
        cases.append(Example("port-" + orientation.lower(), "port",
            f"in q_in: data = 0\nout q_out: data = 2\n0: Port [0,0,0]\n"
            f"1: {orientation} [0,0,1]\n2: Port [0,0,2]\n0 -> +Z\n1 -> +Z"))
        cases.append(Example("memory-padding-" + orientation.lower(), "padding",
            f"in q_in: data = 0\nout q_out: data = 3\n0: Port [0,0,0]\n"
            f"1: {orientation} [0,0,1]\n2: {orientation} [0,0,2]\n"
            "3: Port [0,0,3]\n0 -> +Z\n1 -> +Z\n2 -> +Z", (0, 0, 2)))
        cases.append(Example("y-initialization-" + orientation.lower(), "y-init",
            f"out q_out: data = 2\n0: Y [0,0,0]\n1: {orientation} [0,0,1]\n"
            "2: Port [0,0,2]\n0 -> +Z\n1 -> +Z"))
        cases.append(Example("y-measurement-" + orientation.lower(), "y-meas",
            f"in q_in: data = 0\n0: Port [0,0,0]\n1: {orientation} [0,0,1]\n"
            "2: Y [0,0,2]\n0 -> +Z\n1 -> +Z", (0, 0, 2)))
        for label, (x, y) in {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1),
                              "northeast": (1, 1), "northwest": (-1, 1),
                              "southeast": (1, -1), "southwest": (-1, -1)}.items():
            motion = "slide" if not x or not y else "glide"
            cases.append(Example(f"walking-{orientation.lower()}-{motion}-{label}", "walking",
                f"in q_in: data = 0\nout q_out: data = 2\n0: Port [0,0,0]\n"
                f"1: walk {orientation} [0,0,1] -> [{x},{y},2]\n"
                f"2: Port [{x},{y},3]\n0 -> +Z\n[{x},{y},2] -> +Z", (0, 0, 1)))
    for basis in ("X", "Z"):
        for label, (x, y) in {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}.items():
            cases.append(Example(f"rotation-{basis.lower()}-{label}", "rotation",
                f"in q_in: data = 0\nout q_out: data = 2\n0: Port [0,0,0]\n"
                f"1: rotate {basis} [0,0,1] -> [{x},{y},2]\n"
                f"2: Port [{x},{y},3]\n0 -> +Z\n[{x},{y},2] -> +Z", (0, 0, 1)))
    cases.append(Example("planar-rotation", "planar-rotation",
        "in q_in: data = 0\nout q_out: data = 2\n0: Port [-1,0,1] role=input\n"
        "1: rotate Z [0,0,1] -> [1,1,1]\n2: Port [1,2,1] role=output\n"
        "0 -> +X\n[1,1,1] -> +Y"))
    cases.append(Example("planar-rotation-temporal-ports", "planar-rotation",
        "in q_in: data = 0\nout q_out: data = 4\n"
        "0: Port [1,2,0] role=input\n1: XZZ [1,2,1]\n"
        "2: rotate Z [1,1,1] -> [0,0,1]\n3: XZX [-1,0,1]\n"
        "4: Port [-1,0,2] role=output\n"
        "0 -> +Z\n1 -> [1,1,1]\n[0,0,1] -> 3\n3 -> +Z"))
    for kind in ("ZXZ", "XZX", "XZZ", "ZXX"):
        other = kind.translate(str.maketrans("XZ", "ZX"))
        cases.append(Example("temporal-hadamard-" + kind.lower(), "temporal-h",
            f"in q_in: data = 0\nout q_out: data = 3\n0: Port [0,0,0]\n"
            f"1: {kind} [0,0,1]\n2: {other} [0,0,2]\n3: Port [0,0,3]\n"
            "0 -> +Z\n1 -H> +Z\n2 -> +Z"))
    for kind in ("ZXZ", "XZX"):
        other = kind.translate(str.maketrans("XZ", "ZX"))
        for axis, (x, y) in {"x": (1, 0), "y": (0, 1)}.items():
            cases.append(Example(f"spatial-hadamard-{axis}-{kind.lower()}", "spatial-h",
                f"in q_in: data = 0\nout q_out: data = 3\n0: Port [0,0,0]\n"
                f"1: {kind} [0,0,1]\n2: {other} [{x},{y},1]\n3: Port [{x},{y},2]\n"
                "0 -> +Z\n1 -H> 2\n2 -> +Z", (0, 0, 1)))
    for suffix, kind in (("", "XZX"), ("-swapped", "ZXZ")):
        cases.append(Example("t-cultivation-and-escape" + suffix, "t",
            f"out q_out: data = 2\n0: T [0,0,0]\n1: {kind} [0,0,1]\n"
            "2: Port [0,0,2]\n0 -> +Z\n1 -> +Z", distance=11))
    assert len({c.name for c in cases}) == len(cases)
    return cases


def tick_count(circuit: bloq.ir.Circuit, body: int | None = None) -> int:
    return sum(1 if isinstance(op, bloq.ir.CircuitOp.Tick) else
               op.repetitions * tick_count(circuit, op.body) if isinstance(op, bloq.ir.CircuitOp.Repeat) else 0
               for op in circuit.ops(body))


def repeat_depth(circuit: bloq.ir.Circuit) -> int:
    repeats = [op for op in circuit.ops() if isinstance(op, bloq.ir.CircuitOp.Repeat)]
    assert repeats, "expected a compiled repeat body"
    depths = {tick_count(circuit, op.body) for op in repeats}
    assert len(depths) == 1, depths
    return depths.pop()


def stage(label: str, start: int, stop: int, repetitions: int = 1, meaning: str = "") -> dict:
    assert start < stop and repetitions >= 1
    return dict(label=label, tick_start=start, tick_stop=stop,
                repetitions=repetitions, repeat_meaning=meaning)


def compiled_stages(case: Example, distance: int):
    program = bloq.compile(bloq.BlockGraph.from_text(case.source), distance=distance)
    limitations = []
    if case.family == "planar-rotation":
        # Align the enlarged patch and its neighbors on their shared clock.
        circuit = bloq.emit_stim(program, align_moments=True)
        assert circuit.num_ticks == 6 * distance + 2
        return circuit, [stage("Round 3, from ancilla reset to readout", 13, 19)], [], {}
    if case.family == "t":
        artifact = bloq.emit_isolated_t_attempts(program, noise=0)
        circuit = stim.Circuit(artifact.companion)
        region = program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
        body = region.body(program)
        plans = [program.emission_plan(n, path=region.path + [(region.node, "body")])
                 for n in body.deterministic_emit_order() if body.node(n).quantum is not None]
        assert len(plans) == 2
        cultivation, escape = plans
        # Compiled boundaries: injection 11, syndrome 10, certification 14.
        assert tick_count(cultivation.circuit) + 1 == 35
        assert repeat_depth(escape.circuit) == 6
        assert tick_count(escape.circuit) == 45
        offset = tick_count(cultivation.circuit) + 1
        assert circuit.num_ticks == offset + tick_count(escape.circuit) + 1
        stages = [stage("Injection (Clifford proxy)", 0, 11),
                  stage("Steane syndrome extraction", 11, 21),
                  stage("Certification (Clifford proxy)", 21, offset)]
        stages += [stage(f"Escape merge round {i+1} (d = 5)", offset + 7*i, offset + 7*(i+1)) for i in range(3)]
        stages += [stage(f"Growth (d = 5 → {distance})" if distance > 5 else "Patch recovery (d = 5)", offset + 21, offset + 27),
                   stage(f"Stabilization (d = {distance})", offset + 27, offset + 33, 3, "Three unchanged surface code rounds")]
        limitations.append("Detector propagation uses the S/Clifford companion of one cultivation attempt; it does not propagate Pauli detectors through honest T gates or model repeat-until-success retries.")
        return circuit, stages, limitations, {}

    edits = {}
    target = program.node_by_block(case.position)
    assert target is not None
    if case.family == "padding":
        target = program.insert_memory_rounds(program.quantum_input(target), target, rounds=3)
        edits = {"operation": "insert_memory_rounds", "position": list(case.position), "rounds": 3}
    segments = bloq.emit_stim_segments(program)
    circuit = stim.Circuit(segments.to_text())
    offset = 0
    nodes = {}
    active_ticks = {}
    for segment in segments.segments:
        node_circuit = stim.Circuit(segment.text)
        depth = node_circuit.num_ticks
        if depth:
            nodes[segment.node_id] = (offset, depth)
            tick = offset
            active = set()
            for op in node_circuit.flattened():
                if op.name == "TICK":
                    tick += 1
                elif op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}:
                    active.add(tick)
            active_ticks[segment.node_id] = sorted(active)
        offset += depth
    assert offset == circuit.num_ticks
    if case.family == "temporal-h":
        candidates = [(n, start, depth) for n, (start, depth) in nodes.items()
                      if depth == 6 and program.node(n).quantum is not None]
        assert len(candidates) == 1, candidates
        target, start, depth = candidates[0]
    else:
        start, depth = nodes[target]
    plan = program.emission_plan(target)
    # Fused components and walking/padding may keep empty boundary moments.
    # Derive diagram bounds from moments containing physical operations rather
    # than assuming a barrier belongs to a round. Retain those empty moments
    # in the full circuit used for detector propagation.
    physical_ticks = active_ticks[target]
    depth = len(physical_ticks)
    start = 0
    family = case.family
    if family in {"regular", "spatial"}:
        step = repeat_depth(plan.circuit)
        assert step in ({6} if family == "regular" else {6, 7})
        assert depth == step * distance, (case.name, depth, step)
        stages = [stage("First syndrome round" if family == "regular" else "Initialization round", start, start + step),
                  stage("Bulk round (d − 2)", start + step, start + 2*step, distance - 2, "d - 2 identical syndrome rounds"),
                  stage("Last syndrome round" if family == "regular" else "Final measurement round", start + (distance-1)*step, start + distance*step)]
        if family == "regular" and "-arm" in case.name:
            limitations.append("The full spatially connected component is shown; temporal Ports close only the selected central patch with ideal simulator boundaries.")
        if family == "spatial":
            limitations.append("The complete minimal connected component is shown, including regular endpoint patches required by the arms.")
            if step == 6:
                limitations.append("This straight-through junction compiles to a compact four-CNOT-slot schedule.")
    elif family == "measurement":
        assert depth == 6
        stages = [stage("Closing syndrome round with transversal data readout", start, start+6)]
    elif family == "port":
        assert depth == 1
        stages = [stage("Input Port: ideal stabilizer MPP closure", start, start+1)]
        last_start, last_depth = nodes[program.node_by_block((0, 0, 2))]
        assert last_depth == 1
        stages.append(stage("Output Port: ideal stabilizer MPP closure", last_start, last_start+1))
        limitations.append("MPP closures are ideal simulation boundaries, not a physical preparation/readout circuit.")
    elif family in {"y-init", "y-meas"}:
        step = repeat_depth(plan.circuit)
        padding = distance // 2
        assert step == 6 and depth == 8 + step * (padding + 1)
        if family == "y-init":
            stages = [stage("Degenerate-patch initialization", start, start+step),
                      stage("Degenerate bulk", start+step, start+2*step, padding, "floor(d/2) syndrome rounds"),
                      stage("Reversed twist transition", start+step*(padding+1), start+depth)]
        else:
            stages = [stage("Twist transition", start, start+8),
                      stage("Degenerate bulk", start+8, start+8+step, padding, "floor(d/2) syndrome rounds"),
                      stage("Mixed-basis final measurement", start+8+step*padding, start+depth)]
    elif family == "rotation":
        step = repeat_depth(plan.circuit)
        assert step == 6 and depth == 2 * distance * step
        stages = [stage("Grow", start, start+step),
                  stage("Grown bulk", start+step, start+2*step, distance-2, "d - 2 unchanged syndrome rounds"),
                  stage("Transfer measurement", start+(distance-1)*step, start+distance*step),
                  stage("Rotated initialization", start+distance*step, start+(distance+1)*step),
                  stage("Rotated bulk", start+(distance+1)*step, start+(distance+2)*step, distance-2, "d - 2 unchanged syndrome rounds"),
                  stage("Shrink", start+(2*distance-1)*step, start+depth)]
    elif family == "walking":
        assert depth == 6 * 2 * (distance+1)
        stages = [stage("One elementary stepping round", start, start+6)]
        limitations.append("The full motion uses 2(d+1) translated elementary rounds. They are not identical fixed-support repetitions; only the first elementary step is displayed.")
    elif family == "temporal-h":
        assert depth == 6
        stages = [stage("Realignment and transversal Hadamard", start, start+depth)]
    elif family == "spatial-h":
        assert depth == 8 * distance
        stages = [stage("Initialization round", start, start+8),
                  stage("Backward wall round", start+8, start+16, distance//2, "Interior backward schedule appears floor(d/2) times"),
                  stage("Forward wall round", start+16, start+24, max(1, distance//2-1), "Interior forward schedule; boundary initialization and measurement are displayed separately"),
                  stage("Final measurement round", start+(distance-1)*8, start+depth)]
        limitations.append("The current spatial Hadamard construction is distance degraded. Alternating rounds are displayed separately; they must not be collapsed into one repeated round.")
    elif family == "padding":
        assert depth == 18
        stages = [stage("Protected memory round", start, start+6, 3, "Three inserted QEC rounds on the temporal seam")]
    else:
        raise ValueError(family)
    for item in stages[:1] if family == "port" else stages:
        local_start, local_stop = item["tick_start"], item["tick_stop"]
        selected = physical_ticks[local_start:local_stop]
        assert len(selected) == local_stop-local_start
        assert selected == list(range(selected[0], selected[-1]+1)), (case.name, item, selected)
        item["tick_start"], item["tick_stop"] = selected[0], selected[-1]+1
    return circuit, stages, limitations, edits


def y_up(circuit: stim.Circuit) -> stim.Circuit:
    """Reflect coordinate annotations only; preserve the entire compiled circuit."""
    out = stim.Circuit()
    for op in circuit.flattened():
        args = op.gate_args_copy()
        if op.name in {"QUBIT_COORDS", "DETECTOR"} and len(args) >= 2:
            args[1] = -args[1]
        out.append(op.name, op.targets_copy(), args, tag=op.tag)
    return out


def compose(circuit: stim.Circuit, stages: list[dict], title: str) -> str:
    """Stack native Stim diagrams, each with a visible stage/repetition label."""
    panels = []
    reflected = y_up(circuit)
    for i, item in enumerate(stages):
        assert item["tick_stop"] <= circuit.num_ticks + 1
        ticks = range(item["tick_start"], item["tick_stop"])
        raw = str(reflected.diagram("detslice-with-ops-svg", tick=ticks, rows=math.ceil(len(ticks)/3)))
        raw = re.sub(r'<text[^>]*>Tick \d+</text>', "", raw)
        # Native Stim uses repeated ids (qubit_dots, tick_borders, clip paths).
        # Prefix them so combining panels cannot change another panel's references.
        root = ET.fromstring(raw)
        ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
        for element in root.iter():
            for key, value in list(element.attrib.items()):
                if key == "id":
                    element.set(key, f"p{i}-{value}")
                else:
                    value = re.sub(r"url\(#([^)]+)\)",
                                   lambda m: f"url(#p{i}-{m[1]})" if m[1] in ids else m[0], value)
                    if value.startswith("#") and value[1:] in ids:
                        value = f"#p{i}-{value[1:]}"
                    element.set(key, value)
        box = list(map(float, root.attrib["viewBox"].split()))
        panels.append((item, root, box))
    width = 1200
    heights = [box[3] * (width-32)/box[2] + 48 for _, _, box in panels]
    height = 44 + sum(heights)
    root = ET.Element(f"{{{SVG}}}svg", {"viewBox": f"0 0 {width} {height:.2f}",
                                      "width": str(width), "height": f"{height:.2f}", "role": "img"})
    ET.SubElement(root, f"{{{SVG}}}title").text = title
    ET.SubElement(root, f"{{{SVG}}}rect", {"width": str(width), "height": f"{height:.2f}", "fill": "white"})
    top = 16
    for (item, panel, box), panel_height in zip(panels, heights):
        label = f"{item['label']} — moments {item['tick_start']}–{item['tick_stop'] - 1}"
        if item["repetitions"] > 1:
            label += f" — repeats {item['repetitions']} times"
        ET.SubElement(root, f"{{{SVG}}}text", {"x": "16", "y": str(top+22), "font-family": "sans-serif", "font-size": "22", "fill": "#111"}).text = label
        inner_height = panel_height - 48
        panel.attrib.update(x="16", y=str(top+36), width=str(width-32), height=f"{inner_height:.2f}")
        panel.insert(0, ET.Element(f"{{{SVG}}}rect", {"x": str(box[0]), "y": str(box[1]), "width": str(box[2]), "height": str(box[3]), "fill": "white"}))
        root.append(panel)
        top += panel_height
    return ET.tostring(root, encoding="unicode") + "\n"


def retain_unselected(manifest: dict, previous: dict, selected: list[Example]) -> None:
    """Keep other diagrams when updating a subset with the same compiler settings."""
    names = {case.name for case in selected}
    old_cases = [case for case in previous["examples"] if case["name"] in names]
    if (any(previous[key] != manifest[key] for key in ("distance", "stim_version"))
            or any(case.get("bloq_version", previous["bloq_version"]) != manifest["bloq_version"]
                   for case in old_cases)):
        raise ValueError("Settings changed; regenerate all constructions without --only")
    # Historical entries inherit the manifest version. A new example records
    # its own compiler version if the retained examples came from an older one.
    manifest["bloq_version"] = previous["bloq_version"]
    for key in ("examples", "errors"):
        manifest[key] = [case for case in previous[key] if case["name"] not in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/_static/constructions"))
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--only", help="Render one family or exact example name")
    args = parser.parse_args()
    if args.distance < 5 or args.distance % 2 == 0:
        parser.error("Use an odd distance of at least 5 to display a repeated bulk round")
    selected = [c for c in examples() if args.only is None or args.only in {c.family, c.name}]
    if not selected:
        parser.error("--only did not match an example")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"distance": args.distance, "stim_version": stim.__version__, "bloq_version": bloq.__version__,
                "description": "Diagrams show selected moments of complete noiseless compiled circuits; all undisplayed rounds remain in detector propagation.",
                "examples": [], "errors": []}
    manifest_path = args.output_dir / "manifest.json"
    if args.only and manifest_path.exists():
        try:
            retain_unselected(manifest, json.loads(manifest_path.read_text()), selected)
        except ValueError as exc:
            parser.error(str(exc))
    for case in selected:
        try:
            distance = case.distance if case.distance is not None else args.distance
            circuit, stages, limitations, edits = compiled_stages(case, distance)
            svg = compose(circuit, stages, case.name)
            ET.fromstring(svg)
            source = args.output_dir / f"{case.name}.blog"
            output = args.output_dir / f"{case.name}.stim"
            figure = args.output_dir / f"{case.name}.svg"
            source.write_text(case.source)
            output.write_text(str(circuit) + "\n")
            figure.write_text(svg)
            manifest["examples"].append(dict(name=case.name, family=case.family, source=source.name,
                stim=output.name, svg=figure.name, source_blog=case.source, ir_edits=edits, distance=distance,
                circuit_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), ticks=circuit.num_ticks,
                measurements=circuit.num_measurements, detectors=circuit.num_detectors,
                stages=stages, limitations=limitations))
            if bloq.__version__ != manifest["bloq_version"]:
                manifest["examples"][-1]["bloq_version"] = bloq.__version__
            print(f"{case.name}: {circuit.num_ticks} ticks, {len(stages)} displayed stages", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            manifest["errors"].append(dict(name=case.name, error=message))
            print(f"ERROR {case.name}: {message}", file=sys.stderr, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    names = {case.name for case in selected}
    rendered = sum(case["name"] in names for case in manifest["examples"])
    print(f"Rendered {rendered}/{len(selected)} constructions")
    return int(bool(manifest["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
