"""Render the guide's four planar pipe layouts with Bloq Editor's depth buffer.

Build the editor with cargo build -p bloq_editor --locked. Run with:
    uv run --project bloq_py --no-sync --with pillow python tools/render_planar_rotation_pipes.py
Requires a graphical session and rsvg-convert. Use --editor to override the binary.
"""

import argparse
import base64
from io import BytesIO
from itertools import groupby, product
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import bloq
from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/_static/constructions/planar-rotation.blog"
OUTPUT = ROOT / "docs/assets/planar-rotation-pipes.png"
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)
INK, MUTED = "#203449", "#496275"
INPUTS = [("West", (-1, 0, 1), "+X"), ("South", (0, -1, 1), "+Y")]
OUTPUTS = [("North", (1, 2, 1), "+Y"), ("East", (2, 1, 1), "+X")]


def element(parent, name, **attrs):
    return ET.SubElement(parent, f"{{{SVG}}}{name}", {k: str(v) for k, v in attrs.items()})


def label(parent, x, y, text, size=17, color=INK, weight=400, anchor="start"):
    element(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "DejaVu Sans, sans-serif", "font-size": size,
        "font-weight": weight, "text-anchor": anchor,
    }).text = text


def capture(editor, rotation, src, dst, tmp):
    in_name, input_pos, incoming = src
    out_name, output_pos, outgoing = dst
    graph = bloq.BlockGraph()
    graph.add_block(rotation)
    graph.add_block(bloq.Block(input_pos, "Port", role="input"))
    graph.add_block(bloq.Block(output_pos, "Port", role="output"))
    graph.add_pipe(bloq.Pipe(input_pos, incoming))
    graph.add_pipe(bloq.Pipe((1, 1, 1), outgoing))
    graph.validate()

    # The editor fits the camera to occupied cells. These two off-panel cubes
    # fix that box around the same slab center for all four captures.
    for position in [(-2, -2, 1), (3, 3, 1)]:
        graph.add_block(bloq.Block(position, "ZXZ"))
    path = tmp / f"{in_name}-{out_name}.blog"
    graph.save(path)
    png = path.with_suffix(".png")
    subprocess.run([str(editor), "--render-blog", str(path), "--output", str(png),
                    "--azimuth", "45", "--elevation", "35.26439"], check=True)
    with Image.open(png) as image:
        return image.convert("RGB")


def objects(image):
    mask = image.convert("L").point(lambda value: 255 if value < 245 else 0)
    boxes = []
    for occupied, group in groupby(enumerate(mask.getprojection()[0]), key=lambda pair: pair[1]):
        if occupied:
            xs = [x for x, _ in group]
            left, right = xs[0], xs[-1] + 1
            _, top, _, bottom = mask.crop((left, 0, right, image.height)).getbbox()
            boxes.append((left, top, right, bottom))
    assert len(boxes) == 3, "Expected the model between two separate camera framing cubes"
    return boxes


def port_tops(image):
    # Leaders terminate on the light grey Port caps, independent of screenshot resolution.
    r, g, b = image.split()
    chroma = ImageChops.lighter(ImageChops.difference(r, g), ImageChops.difference(r, b))
    grey = ImageChops.multiply(chroma.point(lambda c: 255 if c <= 5 else 0),
                               image.convert("L").point(lambda v: 255 if 150 <= v <= 240 else 0))
    # Reject isolated grey antialiasing pixels along the colored pipe edges.
    grey = grey.filter(ImageFilter.MinFilter(5))
    quarter = image.width // 4
    tops = []
    for left, right in [(0, quarter), (image.width - quarter, image.width)]:
        box = grey.crop((left, 0, right, image.height)).getbbox()
        assert box, "Expected a visible grey Port at each end"
        tops.append((left + (box[0] + box[2]) / 2, box[1]))
    return tops


def draw(images):
    boxes = [objects(image) for image in images]
    assert all(left == boxes[0][0] and right == boxes[0][2] for left, _, right in boxes), (
        "The common framing cubes must project to identical pixels in every panel"
    )
    bounds = [box[1] for box in boxes]
    crop = (min(b[0] for b in bounds) - 2, min(b[1] for b in bounds) - 2,
            max(b[2] for b in bounds) + 2, max(b[3] for b in bounds) + 2)
    images = [image.crop(crop) for image in images]
    root = ET.Element(f"{{{SVG}}}svg", width="1100", height="670", viewBox="0 0 1100 670")
    element(root, "rect", width=1100, height=670, fill="white")
    label(root, 40, 48, "Four pipe layouts for a planar patch rotation", 31, weight=650)
    label(root, 41, 80, "rotate Z    (0, 0, 1) → (1, 1, 1)    ·    Opposite corners of a 2 × 2 footprint", 16, MUTED)

    # The camera sees east down-right, north up-right, and positive time up.
    for dx, dy, title, tx, ty in [(29, -17, "N  (+y)", 982, 35), (29, 17, "E  (+x)", 982, 70),
                                   (0, -33, "Time  (+z)", 873, 21)]:
        element(root, "line", x1=943, y1=49, x2=943 + dx, y2=49 + dy, stroke=MUTED, **{"stroke-width": 1.6})
        element(root, "circle", cx=943 + dx, cy=49 + dy, r=2.5, fill=MUTED)
        label(root, tx, ty, title, 14, MUTED)
    for x1, y1, x2, y2 in [(40, 102, 1060, 102), (550, 120, 550, 646), (40, 377, 1060, 377)]:
        element(root, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke="#d3dce2")

    for i, (image, (src, dst)) in enumerate(zip(images, product(INPUTS, OUTPUTS), strict=True)):
        row, col = divmod(i, 2)
        scale = min(470 / image.width, 180 / image.height)
        w, h = image.width * scale, image.height * scale
        left, top = 284 + col * 530 - w / 2, 170 + row * 272
        buf = BytesIO()
        image.save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        element(root, "image", x=left, y=top, width=w, height=h, **{f"{{{XLINK}}}href": uri})
        for (x, y), name in zip(port_tops(image), [f"{src[0]} input", f"{dst[0]} output"], strict=True):
            xpos = left + x * scale
            label(root, xpos, top - 25, name, 18, weight=600, anchor="middle")
            element(root, "line", x1=xpos, y1=top - 17, x2=xpos, y2=top + y * scale - 3, stroke="#8194a1")
    subprocess.run(["rsvg-convert", "--zoom", "1.5", "--output", str(OUTPUT)],
                   input=ET.tostring(root, encoding="utf-8"), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=Path, default=ROOT / "target/debug/bloq_editor")
    args = parser.parse_args()
    canonical = bloq.BlockGraph.load(SOURCE).flatten()
    canonical.validate()
    rotation, = [b for b in canonical.blocks() if b.kind.is_patch_rotation]
    assert rotation.pos == (0, 0, 1) and rotation.kind.movement == (1, 1)
    assert canonical.has_pipe_between((-1, 0, 1), rotation.pos)
    assert canonical.has_pipe_between((1, 1, 1), (1, 2, 1))
    with TemporaryDirectory() as tmp:
        draw([capture(args.editor.resolve(), rotation, src, dst, Path(tmp))
              for src, dst in product(INPUTS, OUTPUTS)])
    print(OUTPUT)


if __name__ == "__main__":
    main()
