"""Render the distance-five logical strings from the published four-CNOT-layer circuit.

Run from the repository root with Stim installed and rsvg-convert on PATH:
    python tools/render_planar_rotation_logicals.py
"""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import stim

SOURCE = Path("docs/_static/constructions/planar-rotation.stim")
OUTPUT = Path("docs/assets/planar-rotation-logicals.png")
CUTS = (0, 12, 31)  # Stim detecting_regions cuts, after the indicated moment.


def column(x, ymin, ymax):
    return {(x, y) for y in range(ymin, ymax + 1, 2)}


def row(y, xmin, xmax):
    return {(x, y) for x in range(xmin, xmax + 1, 2)}


def checks(basis, xmin, xmax, ymin, ymax):
    return {(x, y) for x in range(xmin, xmax + 1, 2) for y in range(ymin, ymax + 1, 2)
            if (x + y) % 4 == (0 if basis == "X" else 2)}


def verified_flows(circuit):
    """Match closed-form strings and parity records to the circuit, including their signs."""
    assert circuit.num_ticks == 32 and circuit.num_observables == 2
    coords = {q: tuple(map(int, pos)) for q, pos in circuit.get_final_qubit_coordinates().items()}
    data = {(x, y) for x, y in coords.values() if x % 2 and y % 2}
    x_in, x_out = column(-7, 1, 9), column(17, 25, 33)
    z_in, z_out = row(5, -11, -3), row(29, 13, 21)
    x_read = column(17, 1, 23)
    z_read = row(5, -11, 5) | column(5, 5, 21)
    # Each tuple contains input, merged, output, first-round checks, and final data.
    logicals = (
        (x_in, x_out | x_read, x_out,
         checks("X", -6, -2, 0, 10) | checks("X", 0, 16, 0, 16), x_read),
        (z_in, z_out | z_read, z_out,
         checks("Z", 6, 22, 6, 22) | checks("Z", 12, 22, 24, 28), z_read),
    )
    tick = 0
    measurements = []
    phases = [set(), set()]
    physical = stim.Circuit()
    for op in circuit:
        if op.name == "OBSERVABLE_INCLUDE":
            phase = phases[int(op.gate_args_copy()[0])]
            for target in op.targets_copy():
                if target.is_measurement_record_target:
                    phase.symmetric_difference_update([len(measurements) + target.value])
            continue
        physical.append(op)
        if op.name == "TICK":
            tick += 1
        elif op.num_measurements:
            groups = op.target_groups()
            assert len(groups) == op.num_measurements
            measurements.extend((tick, op.name, tuple(coords[q.value] for q in group))
                                for group in groups)

    regions = circuit.detecting_regions(targets=["L"])
    for index, (basis, logical) in enumerate(zip("XZ", logicals)):
        start, mid, end, first_checks, last_data = logical
        regions_at_tick = regions[stim.target_logical_observable_id(index)]
        for at, sites in zip((0, 6, 12, 18, 24, 30, 31), (start, mid, mid, mid, mid, end, end)):
            support = regions_at_tick[at]
            assert sites <= data and support.weight == len(sites)
            assert {coords[q] for q in support.pauli_indices(basis)} == sites
        gate = "MX" if basis == "X" else "M"
        outcomes = {(6, gate, (pos,)) for pos in first_checks} | {
            (30, gate, (pos,)) for pos in last_data}
        records = sorted(phases[index])
        assert {measurements[k] for k in records} == outcomes and len(records) == len(outcomes)
        flow = stim.Flow(input=regions_at_tick[CUTS[0]], output=regions_at_tick[CUTS[-1]],
                         measurements=records)
        assert physical.has_flow(flow, unsigned=False), f"incorrect sign for logical {basis}"
        print(f"{basis} → {basis}: {len(start)} → {len(mid)} → {len(end)} sites, "
              f"{len(first_checks)} ancilla + {len(last_data)} data outcomes")
    return data, logicals


def render(data, logicals):
    figure = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox="0 0 1320 825",
                        width="1320", height="825", attrib={"font-family": "DejaVu Sans, sans-serif"})
    ink, muted = "#203449", "#465f6e"

    def element(tag, **attrs):
        return ET.SubElement(figure, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})

    def label(x, y, text, size=17, weight=400, anchor="start", fill=ink):
        element("text", x=x, y=y, fill=fill, font_size=size, font_weight=weight,
                text_anchor=anchor).text = text

    def line(x1, y1, x2, y2, color, width=1):
        element("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)

    def dot(x, y, r, color, stroke="none"):
        element("circle", cx=x, cy=y, r=r, fill=color, stroke=stroke, stroke_width=.8)

    def square(x, y, half, color):
        element("rect", x=x-half, y=y-half, width=half*2, height=half*2, fill="white",
                stroke=color, stroke_width=1.7)

    element("rect", width=1320, height=825, fill="white")
    label(29, 43, "Logical Pauli transport through a planar patch rotation", 32, 650)
    label(30, 73, "Distance 5   ·   West input to north output   ·   Five rounds, four CNOT layers per round",
          18, fill=muted)
    line(28, 92, 1292, 92, "#bdcbd1")
    centers = (193, 486, 779, 1127)
    for center, heading, detail in zip(
        centers, ("Input boundary", "Merged patch", "Output boundary", "Measurement parity"),
        ("Before round 1  ·  tick 0", "After round 2  ·  tick 12", "After output checks  ·  tick 31",
         "Ancilla at tick 6, data at tick 30"),
    ):
        label(center, 125, heading, 22, 600, "middle")
        label(center, 150, detail, 15, anchor="middle", fill=muted)
    line(948, 105, 948, 720, "#b4c3ca")
    line(28, 440, 1292, 440, "#d3dde1")
    for index, (basis, logical, color) in enumerate(zip("XZ", logicals, ("#ae4133", "#235e9e"))):
        ytop = 168 + 282 * index
        label(28, ytop+118, basis, 30, 650, fill=color)
        label(27, ytop+142, f"L{index}", 17, 500, fill=muted)
        for panel, center in enumerate(centers):
            scale = 6.45

            def point(x, y):
                return center + (x - 5) * scale, ytop + (35 - y) * scale

            for x0, y0, x1, y1 in [(-12, 0, -2, 10), (12, 24, 22, 34),
                                    (-2, 0, 0, 10), (12, 22, 22, 24), (0, 0, 22, 22)]:
                x, y = point(x0, y1)
                element("rect", x=x, y=y, width=(x1-x0)*scale, height=(y1-y0)*scale,
                        fill="#eaf0f2" if x0 == 0 else "#f4f6f6", stroke="#afbdc4")
            for pos in sorted(data):
                dot(*point(*pos), 1.25, "#b5c2c8")
            if panel < 3:
                sites = logical[panel]
                for x, y in sorted(sites):
                    for next_pos in ((x+2, y), (x, y+2)):
                        if next_pos in sites:
                            line(*point(x, y), *point(*next_pos), color, 2.1)
                for pos in sorted(sites):
                    dot(*point(*pos), 3.75, color, "white")
                caption = f"{len(sites)} data qubits"
            else:
                first_checks, last_data = logical[3:]
                for pos in sorted(first_checks):
                    square(*point(*pos), 2.7, color)
                for pos in sorted(last_data):
                    dot(*point(*pos), 3.9, color, "white")
                caption = f"{len(first_checks)} ancilla + {len(last_data)} data outcomes"
            if index == panel == 0:
                label(*point(-12, 12), "input", 14, fill=muted)
                label(*point(10, 28), "output", 14, anchor="end", fill=muted)
            label(center, ytop+254, caption, 17, anchor="middle")
    line(28, 729, 1292, 729, "#bdcbd1")
    label(30, 761, "Each red or blue string marks the data qubits supporting logical X or Z at that tick. North is up.", 18)
    square(35, 793, 4.2, ink)
    dot(56, 793, 4.8, ink)
    label(72, 799, "Odd parity of the marked ancilla (squares) and data (dots) outcomes reverses the sign. Even parity preserves it.", 18)
    subprocess.run(["rsvg-convert", "--width=1980", "--output", str(OUTPUT)],
                   input=ET.tostring(figure), check=True)


if __name__ == "__main__":
    render(*verified_flows(stim.Circuit.from_file(SOURCE).flattened()))
