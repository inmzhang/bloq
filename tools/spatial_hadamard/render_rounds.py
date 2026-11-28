"""Render every cut of the distance-five straight and concurrent X/Y wall rounds."""

import io
import json
from pathlib import Path

import resvg_py
import stim
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/_static/constructions"
CELL = 260


def font(size):
    return ImageFont.load_default(size=size)


def render(name, title, record):
    circuit = stim.Circuit.from_file(ASSETS / record["stim"])
    # Reflect annotations so +Y points north on the drawing.
    y_up = stim.Circuit()
    for op in circuit.flattened():
        args = op.gate_args_copy()
        if op.name in {"QUBIT_COORDS", "DETECTOR"} and len(args) >= 2:
            args[1] = -args[1]
        y_up.append(op.name, op.targets_copy(), args, tag=op.tag)
    d = record["distance"]
    # A repeat-body boundary can contribute an empty TICK between rounds.
    # Locate the actual reset and measurement cuts in the full compiled timeline.
    starts, stops = set(), set()
    tick = 0
    for op in y_up:
        if op.name == "TICK":
            tick += 1
        elif op.name in {"R", "RX", "RY"}:
            starts.add(tick)
        elif op.name in {"M", "MX", "MY"}:
            stops.add(tick)
    starts = sorted(starts)
    assert len(starts) == d == 5 and stops == {t+6 for t in starts}
    canvas = Image.new("RGB", (7*CELL+36, 100+d*(CELL+38)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, font=font(32), fill="#17252c")
    for slot, text in enumerate(["Prepare", *(f"CX/CZ {i}" for i in range(1, 6)), "Read out"]):
        draw.text((18+(slot+.5)*CELL, 71), text, anchor="mm", font=font(23), fill="#17252c")
    for r in range(d):
        top = 96 + r*(CELL+38)
        start = starts[r]
        label = "Initialization" if r == 0 else "Final data readout" if r == d-1 else "Bulk"
        draw.text((18, top), f"Round {r+1}   ·   {label}   ·   ticks {start} to {start+6}",
                  font=font(20), fill="#263d48")
        for slot in range(7):
            tick = start + slot
            raw = str(y_up.diagram("detslice-with-ops-svg", tick=tick))
            png = resvg_py.svg_to_bytes(svg_string=raw, width=CELL-4, background="white")
            panel = Image.open(io.BytesIO(png)).convert("RGB")
            panel.thumbnail((CELL-4, CELL-4))
            left = 18 + slot*CELL
            canvas.paste(panel, (left+(CELL-panel.width)//2, top+32+(CELL-panel.height)//2))
    canvas.save(ROOT / "docs/assets" / f"{name}-rounds.png", optimize=True)


if __name__ == "__main__":
    cases = {r["name"]: r for r in json.loads((ASSETS/"manifest.json").read_text())["examples"]}
    for name, title in [("spatial-hadamard-x-zxz", "Single X wall   ·   distance 5"),
                        ("spatial-hadamard-elbow-zzx", "Concurrent X/Y walls   ·   ZZX hub   ·   distance 5"),
                        ("spatial-hadamard-elbow-xxz", "Concurrent X/Y walls   ·   XXZ hub   ·   distance 5")]:
        render(name, title, cases[name])
