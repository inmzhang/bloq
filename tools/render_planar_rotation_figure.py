"""Render the first, repeated middle, and final rounds of the planar rotation.

Run from the repository root after render_construction_slices.py --only planar-rotation.
Rendering the indexed PNG requires rsvg-convert (librsvg) and ImageMagick.
"""

from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import stim

from render_construction_slices import SVG, y_up

SOURCE = Path("docs/_static/constructions/planar-rotation.stim")
OUTPUT = Path("docs/assets/planar-rotation-detectors.png")


def syndrome_rounds(circuit):
    tick = 0
    gates = defaultdict(set)
    for op in circuit.flattened():
        if op.name == "TICK":
            tick += 1
        elif op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}:
            gates[tick].add(op.name)
    rounds = []
    for moment, names in gates.items():
        if names == {"MPP"}:
            continue  # Ideal port boundaries surrounding the physical rounds.
        if names == {"R", "RX"}:
            rounds.append([])
        rounds[-1].append(moment)
    expected = [{"R", "RX"}, {"CX"}, {"CX"}, {"CX"}, {"CX"}, {"M", "MX"}]
    assert len(rounds) == 5
    for row in rounds:
        assert [gates[t] for t in row] == expected
    return rounds


def main():
    circuit = stim.Circuit.from_file(SOURCE)
    rounds = syndrome_rounds(circuit)
    display = y_up(circuit)
    ticks = [tick for row in rounds for tick in row]
    regions = circuit.detecting_regions(targets=["D"], ticks=ticks)
    # Detector IDs advance between rounds. Compare the Pauli supports, with multiplicity.
    for repeated_ticks in zip(*rounds[1:4], strict=True):
        first, *rest = [
            Counter(str(cuts[tick]) for cuts in regions.values() if tick in cuts)
            for tick in repeated_ticks
        ]
        assert all(supports == first for supports in rest), repeated_ticks

    qubits = set(circuit.get_final_qubit_coordinates())
    left, col_width, top, row_height = 140, 182, 163, 177
    rows = [("Round 1", rounds[0]), ("Rounds 2–4", rounds[1]), ("Round 5", rounds[4])]
    footer = top + row_height * len(rows)
    width, height = left + 6 * col_width + 16, footer + 65
    figure = ET.Element(f"{{{SVG}}}svg", viewBox=f"0 0 {width} {height}",
                        width=str(width), height=str(height))

    def rect(x, y, w, h, color):
        ET.SubElement(figure, "rect", x=str(x), y=str(y), width=str(w), height=str(h), fill=color)

    def label(x, y, content, size, weight="400", color="#203449", anchor="start"):
        element = ET.SubElement(figure, "text", x=str(x), y=str(y), fill=color, attrib={
            "font-family": "DejaVu Sans, sans-serif", "font-size": str(size),
            "font-weight": weight, "text-anchor": anchor})
        element.text = content

    rect(0, 0, width, height, "white")
    label(24, 41, "Five syndrome rounds of a planar patch rotation", 33, "650")
    label(25, 72, "Distance 5   ·   2 × 2 patch cells   ·   West input to north output   ·   Detector support after each moment", 18)
    for x, color, caption in [(25, "#ff7070", "X sensitivity"), (184, "#70aaff", "Z sensitivity")]:
        rect(x, 93, 14, 14, color)
        label(x + 23, 106, caption, 16)
    label(350, 106, "Dots mark qubits. Physical x increases right and y increases up.", 16, color="#455c6c")

    for col, name in enumerate(["Reset", "CNOT 1", "CNOT 2", "CNOT 3", "CNOT 4", "Readout"]):
        label(left + (col + .5) * col_width, 149, name, 21, "600", anchor="middle")
    for row_index, (name, row) in enumerate(rows):
        row_top = top + row_height * row_index
        rect(24, row_top - 5, width - 48, 1, "#c4d0d6")
        label(25, row_top + 76, name, 19, "650")
        if row_index == 1:
            label(25, row_top + 110, "×3", 28, "650", "#455c6c")
            label(25, row_top + 133, "Round 2 shown", 14, color="#455c6c")
        for col, tick in enumerate(row):
            # Stim's bare diagram counts the cut before the first moment as zero.
            raw = display.diagram("detslice-svg", tick=tick + 1, filter_coords=[()])
            panel = ET.fromstring(str(raw))
            ids = set()
            dots = set()
            for el in panel.iter():
                identifier = el.attrib.pop("id", "")
                if identifier.startswith("slice:"):
                    ids.add(int(identifier.split(":")[1]))
                elif identifier.startswith("qubit_dot:"):
                    dots.add(int(identifier.split(":")[1]))
            assert dots == qubits
            assert ids == {d.val for d, cuts in regions.items() if tick in cuts}
            assert all("url(#" not in v for el in panel.iter() for v in el.attrib.values())
            x = left + col * col_width
            side = row_height - 13
            panel.attrib.update(x=str(x + (col_width-side)/2), y=str(row_top + 2),
                                width=str(side), height=str(side))
            figure.append(panel)
            # The west-input / north-output layout leaves the northwest corner free.
            label(x + 6, row_top + 23, f"t{tick}", 16, "600", "#445969")

    rect(24, footer - 5, width - 48, 1, "#c4d0d6")
    label(25, footer + 25, "Round 1 prepares the data in two bases. Round 5 reads out the data across the perpendicular split.", 16)
    label(25, footer + 49, "Ideal port stabilizer measurements precede round 1 and follow round 5.", 16, color="#455c6c")
    raster = subprocess.check_output(["rsvg-convert", "--width=1836"], input=ET.tostring(figure))
    subprocess.run(["magick", "png:-", "+dither", "-colors", "128", "-strip",
                    "-define", "png:exclude-chunks=date,time", "-define", "png:compression-level=9",
                    f"PNG8:{OUTPUT}"], input=raster, check=True)


if __name__ == "__main__":
    main()
