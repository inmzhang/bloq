#!/usr/bin/env bash
# Regenerates the minimal font subsets bundled into the editor binary.
#
# The Zed faces (classic pre-2024 Zed Sans/Mono, Iosevka-derived) come from the
# zed-fonts release; Fantasque Nerd Font supplies only the icon glyphs. See
# NOTICE.md and the adjacent LICENSE-* files for source and license details.
# Each face is subset to exactly what the editor renders, so the wasm binary
# stays small:
#   - Zed faces: printable ASCII plus every non-ASCII character that appears in
#     bloq_editor/src (scanned below).
#   - Fantasque: only referenced Nerd Font icon codepoints (U+F000..U+F252).
# Hinting is stripped: egui (ab_glyph) never reads it.
#
# Requires: curl, sha256sum, unzip, python3, uvx (for fonttools/pyftsubset).
set -euo pipefail

cd "$(dirname "$0")"
src_dir=../../src
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

echo "Downloading source fonts..."
curl -fsSL -o "$work/zed.zip" \
    https://github.com/zed-industries/zed-fonts/releases/download/1.2.0/zed-app-fonts-1.2.0.zip
printf '%s  %s\n' \
    11ced02d43f122fcc9c5edbe248ff3308d015b038acc2d1743f8da776eabad02 \
    "$work/zed.zip" | sha256sum --check -
unzip -q "$work/zed.zip" -d "$work/zed"
curl -fsSL -o "$work/fantasque.zip" \
    https://github.com/ryanoasis/nerd-fonts/releases/download/v3.4.0/FantasqueSansMono.zip
printf '%s  %s\n' \
    29c6fe2420a61fff58a78c689e27d8b984ccef2990d6ed9c1a7f3661136acd41 \
    "$work/fantasque.zip" | sha256sum --check -
unzip -q "$work/fantasque.zip" -d "$work/fantasque"

# Scan the editor sources for the characters each face must cover. Icon
# codepoints (>= U+E000) go to Fantasque; everything else non-ASCII joins the
# ASCII range for the Zed faces.
read -r text_unicodes icon_unicodes < <(python3 - "$src_dir" <<'EOF'
import re, sys, pathlib

chars, icons = set(), set()
for path in pathlib.Path(sys.argv[1]).rglob("*.rs"):
    source = path.read_text()
    for m in re.finditer(r"\\u\{([0-9a-fA-F]+)\}", source):
        cp = int(m.group(1), 16)
        (icons if cp >= 0xE000 else chars).add(cp)
    chars.update(ord(c) for c in source if ord(c) > 0x7E)

text = "U+0020-007E," + ",".join(f"U+{c:04X}" for c in sorted(chars))
icon = ",".join(f"U+{c:04X}" for c in sorted(icons))
print(text, icon)
EOF
)
echo "Text unicodes: $text_unicodes"
echo "Icon unicodes: $icon_unicodes"

subset() {
    uvx --from fonttools==4.63.0 pyftsubset "$1" \
        --unicodes="$2" --no-hinting --name-IDs='*' --output-file="$3"
    ls -la "$3"
}

subset "$work/zed/zed-sans-extended.ttf" "$text_unicodes" ZedSans-Regular.ttf
subset "$work/zed/zed-sans-extendedbold.ttf" "$text_unicodes" ZedSans-Bold.ttf
subset "$work/zed/zed-mono-extended.ttf" "$text_unicodes" ZedMono-Regular.ttf
subset "$work/fantasque/FantasqueSansMNerdFontPropo-Regular.ttf" \
    "$icon_unicodes" FantasqueSansMNerdFontPropo-Regular.ttf
