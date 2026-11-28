"""Render the distance-five rotation walkthrough from its published Stim circuit.

Run from the repository root after render_construction_slices.py --only planar-rotation.
Rendering the PNG requires rsvg-convert (librsvg).
"""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import stim

from render_construction_slices import SVG, y_up

SOURCE = Path("docs/_static/constructions/planar-rotation.stim")
OUTPUT = Path("docs/assets/planar-rotation-detectors.png")


def main():
    circuit = stim.Circuit.from_file(SOURCE)
    display = y_up(circuit)
    coords = circuit.get_final_qubit_coordinates()
    by_pos = {tuple(p): q for q, p in coords.items()}
    figure = ET.Element(f"{{{SVG}}}svg", viewBox="0 0 1180 480", width="1180", height="480")
    ET.SubElement(figure, "rect", width="1180", height="480", fill="white")
    ink = "#203449"
    muted = "#425a6e"

    def label(x, y, content, size=16, weight="400", anchor="start", color=ink):
        el = ET.SubElement(figure, "text", x=str(x), y=str(y), fill=color, attrib={
            "font-family": "DejaVu Sans, sans-serif", "font-size": str(size),
            "font-weight": weight, "text-anchor": anchor})
        el.text = content

    def panel(moment, x, y, w, h, detector=None, center=None):
        # Stim's bare detector-slice diagram numbers cuts from before the first
        # moment. detecting_regions uses the zero-based tick immediately after it.
        raw = str(display.diagram("detslice-svg", tick=moment + 1,
                                  filter_coords=[detector] if detector else [()]))
        svg = ET.fromstring(raw)
        svg.attrib.update(x=str(x), y=str(y), width=str(w), height=str(h), overflow="hidden")
        dots = {int(el.get("id").split(":")[1]): el for el in svg.iter()
                if el.get("id", "").startswith("qubit_dot:")}
        ax, ay = (float(dots[by_pos[(4, 4)]].get(k)) for k in ("cx", "cy"))
        scale = (float(dots[by_pos[(8, 8)]].get("cx")) - ax) / 4

        def point(xx, yy):
            return ax + (xx - 4) * scale, ay - (yy - 4) * scale

        if detector:
            cx, cy = point(*center)
            span = 5.7 * scale
            svg.set("viewBox", f"{cx-span/2} {cy-span/2} {span} {span}")
            support = regions[detector][moment]
            assert support.weight == len(support.pauli_indices("X"))
            for q in support.pauli_indices():
                px, py = point(*coords[q])
                if int(coords[q][0]) % 2:
                    ET.SubElement(svg, "circle", cx=str(px), cy=str(py), r="3.7",
                                  fill="#90273a", stroke="white", attrib={"stroke-width": ".7"})
                else:
                    ET.SubElement(svg, "rect", x=str(px - 3.5), y=str(py - 3.5),
                                  width="7", height="7", fill="white", stroke="#90273a",
                                  attrib={"stroke-width": "1.8"})
        else:
            px, py = point(-.35, 22.35)
            ET.SubElement(svg, "rect", x=str(px), y=str(py), width=str(22.7 * scale),
                          height=str(22.7 * scale), fill="none", stroke=ink,
                          attrib={"stroke-width": "2", "stroke-dasharray": "8 6"})
            for name, pos in [("A", (4, 4)), ("B", (8, 8))]:
                px, py = point(*pos)
                ET.SubElement(svg, "circle", cx=str(px), cy=str(py), r=str(scale), fill=ink)
                badge = ET.SubElement(svg, "text", x=str(px), y=str(py + scale * .46),
                                      fill="white", attrib={"font-family": "sans-serif",
                                      "font-size": str(scale * 1.45), "text-anchor": "middle"})
                badge.text = name
        figure.append(svg)

    label(25, 42, "Four CNOT layers in a planar patch rotation", 32, "650")
    label(26, 71, "Distance 5    ·    West input to north output    ·    Five rounds on a 2 × 2 patch footprint", 17, color=muted)
    label(26, 112, "Full patch after round 2", 21, "600")
    label(26, 137, "Tick 12  ·  ancilla readout", 15, color=muted)
    panel(12, 24, 146, 265, 250)
    label(26, 420, "Dashed square = rotation block", 14, color=muted)
    label(330, 112, "Two X checks during round 3", 21, "600")
    regions = circuit.detecting_regions(targets=["D"], ticks=range(12, 19))
    detector_coords = circuit.get_detector_coordinates()
    cx = {}
    tick = 0
    for op in circuit.flattened():
        if op.name == "TICK":
            tick += 1
        elif op.name == "CX" and 14 <= tick <= 17:
            ts = op.targets_copy()
            for a, b in zip(ts[::2], ts[1::2]):
                px, py = coords[a.value]
                qx, qy = coords[b.value]
                cx[tick, (px, py)] = ("N" if qy > py else "S") + ("E" if qx > px else "W")
    cols = [522, 678, 834, 990]
    for layer, left in enumerate(cols):
        label(left + 62, 145, f"CNOT {layer+1}  ·  tick {layer+14}", 15, "600", "middle")
    for row, (pos, place) in enumerate([((4, 4), "Port triangle"), ((8, 8), "Interior")]):
        det, = [d for d, cuts in regions.items() if 12 in cuts and 17 in cuts and
                tuple(detector_coords[d.val]) == pos]
        top = 155 + 128 * row
        label(330, top + 39, f"{chr(65 + row)}  ·  {place}", 19, "600")
        label(330, top + 66, f"{det} at {pos}", 15, color=muted)
        for layer, left in enumerate(cols):
            tick = 14 + layer
            panel(tick, left, top, 124, 90, det, pos)
            label(left + 62, top + 105, cx[tick, pos], 14, "600", "middle", muted)
    ET.SubElement(figure, "path", d="M24 441 H1156", stroke="#ccd6db")
    label(26, 466, "Red = X sensitivity    ·    Blue = Z sensitivity    ·    Dark red dots = sensitive data    ·    Open squares = sensitive ancillas    ·    North is up", 14)
    subprocess.run(["rsvg-convert", "--width=1652", "--output", str(OUTPUT)],
                   input=ET.tostring(figure), check=True)


if __name__ == "__main__":
    main()
