"""Regenerate Examples pages and thumbnails from the compiler's gallery.

Run with: uv run --project bloq_py --no-sync python tools/render_gallery_examples.py
Build the native editor first with cargo build -p bloq_editor --locked.
Its off-screen gallery renderer supplies the PNG images (--editor overrides its path).
BLOG and viewer geometry are read from the native gallery during the site build.
"""

import argparse
import html
from pathlib import Path
import re
import subprocess

import bloq

EXAMPLES = {
    "cnot": ("CNOT", "The control and target enter through separate Ports and leave through their corresponding outputs. Spatial merge and split geometry implements the controlled X operation without measuring either logical output."),
    "cz_spatial_h": ("CZ with a spatial Hadamard", "A Hadamard on a spatial connection changes the boundary basis at the interaction. The two input wires remain available at the outputs after the controlled Z operation."),
    "cz_temporal_h": ("CZ with a temporal Hadamard", "This alternative CZ layout places the Hadamard on a temporal connection. Compare its pipe orientation with the spatial-Hadamard version to see two layouts for the same gate."),
    "s_gate": ("S gate", "The data patch interacts with an ancilla that ends in a Y-basis measurement. The data input and output stay open, so this component can be placed between other gates."),
    "t_gate": ("T gate", "A T resource couples to the data patch. A named parity readout chooses the resource patch's X or Y measurement, while the data patch continues to the output."),
    "t_gate_structural_branch": ("T gate with a structural branch", "The ZZ parity measurement selects the geometry after T injection. The true arm adds a Y cap and applies Z feedback to the output."),
    "t_with_prepared_y": ("T gate with a prepared Y state", "This variant supplies a prepared Y ancilla for the Clifford correction. Its explicit ancilla geometry makes the correction part of the teleportation layout."),
    "t_comparison": ("T-state comparison", "Two T resources are compared through a joint parity readout and a selective X/Y measurement. A final discard action rejects the shot when its corrected comparison result is one."),
    "phase_gradient": ("Phase-gradient sequence", "Reusable injection modules form a single-qubit chain of 22 signed π/4 rotations about X and Z, followed by a Hadamard. The source shows repeated definitions, rotated instances, and explicit quantum seams."),
    "and_4t": ("Four-T AND", "Two multiplex control inputs remain live while four T injections produce an AND output. This temporary AND component can supply the nonlinear intermediate value needed by a larger arithmetic circuit."),
    "ccz_injected_and": ("CCZ-injected AND", "Three resource Ports receive a joint CCZ state from the caller. Multiplex data controls, selective caps, and conditional feedback produce the temporary AND output for later carry logic."),
    "ccz_injected_maj": ("CCZ-injected MAJ", "The majority carry slice consumes a caller-supplied CCZ state. Its interface exposes the next carry together with the intermediate wires needed by the UMA slice."),
    "uma": ("UMA", "The unmajority-and-add slice consumes the carry-stage intermediate wires and extracts a sum output. Named measurements select the final caps, providing the adaptive uncomputation step."),
    "three_bit_adder": ("Three-bit controlled adder", "A three-bit controlled addition is assembled from reusable carry and uncompute modules. Five externally supplied CCZ states drive the nonlinear operations; measurement-dependent final CZ branches supply the corrections."),
    "ten_bit_adder": ("Ten-bit controlled adder", "This larger controlled addition extends the same module pattern to ten bits and nineteen external CCZ states. Its full source illustrates repeated arithmetic stages, routing, classical bindings, and adaptive corrections."),
    "toffoli_from_and_delayed_cz": ("Toffoli from AND", "A four-T temporary AND is combined with a delayed CZ correction. The target output carries z XOR (x AND y), while the two control wires are retained."),
    "ccz_gate_teleport": ("CCZ gate teleportation", "Three multiplex data wires couple to an externally supplied CCZ state. Three named joint measurements choose explicit correction branches, and pairs of outcomes control the remaining Pauli feedback."),
    "ccz_4x3x7_tels": ("CCZ factory with TELS", "The factory prepares a CCZ resource using T injections and parity checks in a 4-by-3 layout. The time-efficient lattice-surgery (TELS) checks are expressed through cube heights, named values, and a discard condition."),
    "ccz_4x3x6": ("CCZ factory without TELS", "This factory uses the related 4-by-3 construction without the TELS check. Its measured checks and discard action define which shots are retained as resource preparations."),
    "ccz_2x3x16": ("Serial CCZ factory", "The 2×3 core runs for 16 surgery layers. Supply T states at the eight purple ports on one side. The BLOG interface declares the first two as data and the remaining six as T resources. The conditional checks couple directly to the shared comparator. [Latte source](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/testdata/surgery_scripts/ccz_1p2x3_17.lat)"),
    "ccz_4x2x12": ("CCZ factory with a selective parity check", "The 4×2 core runs for 12 surgery layers and takes eight T states at purple ports on one side. Structural branches own the Y preparations for S corrections. A selective Y/Z cap supplies the final comparison readout. [Latte source](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/testdata/surgery_scripts/ccz_1p4x2_12.lat)"),
    "perfect_3x3x13": ("Ten-to-two factory, 3×3", "Ten T states produce two T† states in a 3×3 layout. Latte’s output check applies T before measuring X, which gives the T† convention. [Latte source](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/testdata/surgery_scripts/perfect_3x3_13.lat)"),
    "perfect_2x3x28": ("Ten-to-two factory, 2×3", "Ten T states produce two T† states in a 2×3 layout over 28 surgery layers. Two selective caps supply the discard checks. [Latte source](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/testdata/surgery_scripts/perfect_2x3_28.lat)"),
    "bell_state": ("Bell state", "A connected preparation layout creates a two-qubit Bell state. The open output Ports expose the prepared logical qubits to a caller or to later components."),
    "ghz": ("GHZ state", "A row of connected preparation cubes creates a four-qubit GHZ state. Each output Port retains one member of the entangled state."),
    "ghz_slide_then_glide": ("GHZ with slides and glides", "After preparing a four-qubit GHZ state, walking blocks move the patches through a slide and then a glide. Start and end coordinates record each movement explicitly."),
    "ghz_patch_rotations": ("GHZ with patch rotations", "The four GHZ patches pass through rotation blocks before reaching their output Ports. These blocks change patch boundary orientation during the computation, rather than rotating a whole module instance."),
    "1d-yoked": ("1D yoked layout", "Six logical wires pass through a shared lattice-surgery layout with extended-duration cubes and patch rotations. The graph shows how the yoked arrangement joins patches while preserving its public input and output interface."),
    "thth": ("T–H–T–H sequence", "Two T teleportations and two Hadamards form a single-qubit gate sequence. Its resource and adaptive-measurement stages also make it useful for examining execution dependencies and live-patch waiting."),
    "three_cnots": ("Compressed three CNOTs", "A compact graph combines three CNOT interactions on the a, b, and c wires. Spatial and temporal Ports expose the mixed-orientation interface of this compressed layout."),
    "steane_encoding": ("Steane encoding", "A compressed encoding network joins its logical wires through spatial merge and split geometry. The named Ports identify the input and output boundaries of the encoding component."),
    "x_memory": ("X-basis memory", "A logical patch is initialized, held through a memory cube, and measured in the X basis. This closed experiment is a small starting point for inspecting physical circuits and detector consistency."),
    "y_memory": ("Y-basis memory", "Y initialization and measurement surround a memory cube. The Y boundaries exercise the corresponding preparation and readout constructions without open logical Ports."),
    "move_rotation": ("Rotation through movement", "The logical wire turns through spatial connections before reaching its output. The changing cube orientations rotate its boundary types without using an explicit patch-rotation block."),
    "stability": ("Stability experiment", "An isolated spatial cube forms a closed stability experiment with no open logical Ports. It provides a minimal layout for studying stabilizer extraction and detector consistency."),
}

CATEGORIES = {
    "clifford": "Clifford", "non_clifford": "Non-Clifford",
    "factory": "Factory", "external_resource": "External resource",
    "adaptive": "Adaptive", "arithmetic": "Arithmetic", "addition": "Addition",
}


def has_spatial_hadamard(graph):
    # Canonical flat BLOG resolves child rotations and seams, retains both
    # branch arms, and always writes pipe destinations as signed directions.
    # Inspect it without enumerating runtime outcomes or deriving readouts.
    return re.search(r"-H>\s+[+-][XY]\b", graph.flatten().to_text()) is not None


SPATIAL_HADAMARD_NOTE = """:::{caution}
This computation contains spatial Hadamard pipes. The current construction can
reduce the effective circuit distance below the requested code distance.
Check the emitted circuit's distance before relying on it. Prefer temporal
Hadamards or a redesigned layout when preserving circuit distance is required.
See [the distance boundary](../circuit-constructions.md#distance-boundary).
:::

"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pages = root / "docs" / "gallery"
    images = root / "docs" / "_static" / "gallery"
    pages.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    editor = args.editor or root / "target" / "debug" / "bloq_editor"
    subprocess.run([str(editor), "--export-gallery-thumbnails", str(images)], check=True)
    entries = list(bloq.GalleryItem)
    assert set(EXAMPLES) == {item.id for item in entries}, "Every gallery entry needs an explanation"
    assert set(CATEGORIES) == {category for item in entries for category in item.categories}
    cards, tree = [], []
    for item in entries:
        title, explanation = EXAMPLES[item.id]
        assert (images / f"{item.id}.png").is_file(), item.id
        cards.append(
            f'<a class="bloq-gallery-card" href="{item.id}.html" aria-label="{html.escape(title)}" data-categories="{" ".join(item.categories)}">'
            f'<img src="../_static/gallery/{item.id}.png" alt="{html.escape(title)} block graph" '
            f'width="128" height="104" loading="lazy">'
            f'<span>{html.escape(title)}</span></a>'
        )
        tree.append(item.id)
        note = SPATIAL_HADAMARD_NOTE if has_spatial_hadamard(item.load()) else ""
        (pages / f"{item.id}.md").write_text(
            f"# {title}\n\n{explanation}\n\n"
            + note
            + f"## Block graph\n\n```{{bloq-view}} {item.id}\n```\n\n"
            "## BLOG source\n\n"
            f"```{{gallery-blog}} {item.id}\n```\n\n"
            "See [BLOG Format](../graphs/blog.md) for the syntax or return to\n"
            "[all examples](index.md).\n"
        )
        print(item.id)
    (pages / "index.md").write_text(
        "---\nhide-toc: true\n---\n\n# Examples\n\n"
        "Here we show the construction of some compilable surface code computations.\n\n"
        "```{raw} html\n<div class=\"bloq-gallery-filters\" role=\"group\" aria-label=\"Gallery category\" hidden>\n"
        '<button type="button" data-category="all" aria-pressed="true">All</button>\n'
        + "\n".join(f'<button type="button" data-category="{key}" aria-pressed="false">{label}</button>'
                    for key, label in CATEGORIES.items())
        + '\n</div>\n<p class="bloq-gallery-count" aria-live="polite"></p>\n<div class="bloq-gallery-grid">\n'
        + "\n".join(cards) + "\n</div>\n```\n\n"
        "```{toctree}\n:hidden:\n:maxdepth: 1\n\n"
        + "\n".join(tree) + "\n```\n"
    )
    print(f"Rendered {len(entries)} examples and thumbnails")


if __name__ == "__main__":
    main()
