"""Regenerate platform icons: uv run --with pillow bloq_editor/assets/icons/generate.py."""

from pathlib import Path

from PIL import Image

icons = Path(__file__).resolve().parent
with Image.open(icons / "bloq.png") as source:
    image = source.convert("RGBA")
    if image.width != image.height:
        raise ValueError("The app icon must be square")
    small = image.resize((256, 256), Image.Resampling.LANCZOS)
    small.save(icons / "bloq-256.png")
    small.save(
        icons / "bloq.ico", sizes=[(size, size) for size in (16, 32, 48, 64, 128, 256)]
    )
    image.resize((1024, 1024), Image.Resampling.LANCZOS).save(icons / "bloq.icns")
