"""Docstring doctests and docs/api/python.rst drift protection."""

import doctest
import itertools
import json
import math
import re
import runpy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import bloq
import bloq._core
import bloq._gallery
import pytest

API_RST = Path(__file__).parents[2] / "docs" / "api" / "python.rst"


def test_construction_subset_retains_other_diagrams_and_checks_settings():
    pytest.importorskip("stim")
    renderer = runpy.run_path(str(API_RST.parents[2] / "tools/render_construction_slices.py"))
    settings = {"distance": 5, "stim_version": "test", "bloq_version": "test"}
    previous = dict(settings, examples=[{"name": "regular-zxz"}, {"name": "regular-xzx"}],
                    errors=[{"name": "regular-zxz"}, {"name": "other"}])
    manifest = dict(settings, examples=[], errors=[])
    selected = [renderer["Example"]("regular-zxz", "regular", "")]
    renderer["retain_unselected"](manifest, previous, selected)
    assert manifest["examples"] == [{"name": "regular-xzx"}]
    assert manifest["errors"] == [{"name": "other"}]
    for key in settings:
        changed = dict(previous, **{key: "changed"})
        with pytest.raises(ValueError, match="regenerate all constructions"):
            renderer["retain_unselected"](manifest, changed, selected)


def _run_doctests(module):
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0, f"{result.failed} doctest failures in {module.__name__}"
    return result.attempted


def test_facade_doctests():
    assert _run_doctests(bloq) > 0


def test_gallery_doctests():
    assert _run_doctests(bloq._gallery) > 0


def test_ir_doctests():
    assert _run_doctests(bloq.ir) > 0


def test_core_doctests():
    # No minimum count: coverage grows as Examples sections are added to the
    # Rust docstrings; this gate only demands that what exists passes.
    _run_doctests(bloq._core)


@pytest.mark.parametrize("module", [bloq, bloq.ir])
def test_all_lists_exactly_the_public_names(module):
    # `__all__` is hand-written so that linters and type checkers can see the
    # re-exports at all; this is what keeps it in step with the import list.
    bound = {
        name
        for name in vars(module)
        if name == "__version__" or not name.startswith("_")
    }
    assert set(module.__all__) == bound, (
        f"{module.__name__}.__all__ out of sync with the imports: "
        f"missing={sorted(bound - set(module.__all__))} "
        f"stale={sorted(set(module.__all__) - bound)}"
    )
    assert module.__all__ == sorted(module.__all__), "keep __all__ sorted"


def test_api_rst_lists_exactly_the_public_api():
    names = []
    module = "bloq"
    for line in API_RST.read_text().splitlines():
        if line.startswith(".. currentmodule:: "):
            module = line.removeprefix(".. currentmodule:: ")
        elif match := re.fullmatch(r"\.\. auto(?:function|class|data|module):: ([A-Za-z_][A-Za-z0-9_.]*)", line):
            name = match[1]
            names.append(name if name.startswith("bloq.") else f"{module}.{name}")
        elif match := re.fullmatch(r"   ([A-Za-z_][A-Za-z0-9_]*)", line):
            names.append(f"{module}.{match[1]}")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"duplicate inline API descriptions: {duplicates}"
    listed = set(names)
    public = {
        f"{module.__name__}.{name}"
        for module in (bloq, bloq.ir)
        for name in module.__all__
    }
    assert listed == public, (
        f"docs/api/python.rst out of sync with public namespaces: "
        f"missing={sorted(public - listed)} stale={sorted(listed - public)}"
    )


def test_builtin_data_docstrings_are_omitted_without_removing_authored_docs():
    pytest.importorskip("sphinx")
    from sphinx.util.docstrings import prepare_docstring

    hook = runpy.run_path(str(API_RST.parent.parent / "conf.py"))["_strip_builtin_data_docstrings"]
    for value in (bloq.ir.BLOQ_TEXT_EXTENSION, bloq.ir.BLOQ_TEXT_VERSION, True, 0.5, b"bytes"):
        for kind in ("data", "attribute"):
            lines = prepare_docstring(type(value).__doc__)
            assert lines
            hook(None, kind, "bloq.constant", value, None, lines)
            assert not lines

            authored = ["The Bloq format version.", ""]
            hook(None, kind, "bloq.constant", value, None, authored)
            assert authored == ["The Bloq format version.", ""]

    lines = prepare_docstring(str.__doc__)
    hook(None, "class", "str", str, None, lines)
    assert lines == prepare_docstring(str.__doc__)

    # Built-in descriptors can expose a descriptor, rather than text, as __doc__.
    lines = ["The property documentation.", ""]
    hook(None, "attribute", "bloq.property", property(), None, lines)
    assert lines == ["The property documentation.", ""]


def test_guide_chapters_use_active_navigation_parents():
    hook = runpy.run_path(str(API_RST.parent.parent / "conf.py"))["_guide_navigation"]
    cached_parent = SimpleNamespace(active=False)
    preliminaries = SimpleNamespace(
        aria_label="Preliminaries", children=[], parent=cached_parent, current=False,
    )
    tutorial = SimpleNamespace(aria_label="Logical CNOT", children=[], parent=cached_parent)
    chapter = SimpleNamespace(aria_label="Tutorials", children=[tutorial], parent=cached_parent)
    guide = SimpleNamespace(
        aria_label="User Guide", active=True, caption_only=False,
        children=[preliminaries, chapter],
    )
    hook(None, "user-guide", None, {"nav": [guide]}, None)
    # Material's section styling follows the displayed parent, not its stale cache.
    assert chapter.parent.active
    assert tutorial.parent is chapter
    assert guide.children[0].aria_label == "Overview" and guide.children[0].current
    assert not preliminaries.current and guide.caption_only


def test_surface_views_select_boundary_relations_independently_of_row_order():
    select = runpy.run_path(str(API_RST.parent.parent / "conf.py"))["_surface_indices"]
    graph = bloq.GalleryItem.CNOT.load().flatten()
    ports = sorted(block.pos for block in graph.blocks() if block.kind.is_port)
    codes = {"I": 0, "X": 1, "Z": 2, "Y": 3}
    for rows in (graph.stabilizers(), list(reversed(graph.stabilizers()))):
        view = SimpleNamespace(blocks=graph.blocks, stabilizers=lambda: rows)
        for word in ("XXIX", "ZZII", "IIXX", "IZZZ"):
            actual = [0] * len(ports)
            for index in select(view, word):
                support = rows[index].stabilizer.port_stabilizer
                for i, port in enumerate(ports):
                    actual[i] ^= codes[str(support.get(port, "I")).split(".")[-1]]
            assert actual == [codes[letter] for letter in word]
    for invalid in ("XXXX", "III", "IIII", "QQQQ"):
        with pytest.raises(ValueError):
            select(graph, invalid)


def test_mermaid_guard_rejects_keyword_node_ids():
    keyword = runpy.run_path(str(API_RST.parent.parent / "conf.py"))["_MERMAID_KEYWORD_ID"]
    # Browsers report these as "Syntax error in text"; the build must reject them.
    assert keyword.search('start --> end["Rotated patch"]')
    assert keyword.search('  graph["Block graph"] --> compiler')
    for valid in ('start --> finish["Patch"]', 'backend["Stim"]', 'classical["Value"]'):
        assert keyword.search(valid) is None


def test_blog_highlighting_tokens_and_registration():
    from runpy import run_path
    from types import SimpleNamespace

    from pygments.lexers import TextLexer
    from pygments.token import Comment, Error, Keyword, Name, Number, Operator, Punctuation, String

    pytest.importorskip("sphinx_immaterial")
    config = run_path(str(API_RST.parent.parent / "conf.py"))
    registered = {}
    config["setup"](SimpleNamespace(
        add_lexer=registered.__setitem__,
        add_directive=lambda *args: None, connect=lambda *args, **kwargs: None,
    ))
    assert registered["blog"] is config["BlogLexer"]
    assert all(registered[name] is TextLexer for name in ("bloqir", "stim"))
    source = '''BLOG 1.0
import "stage#1.blog" as Stage
module main {
  in q: data = 0
  out bit = child.bit
  0: Port [-1, 0, 0] role=input color=12abEF <q#入口>
  1: ZXZ [0, 0, 1] height=3d/2 + 1
  2: walk XZX [0, 0, 2] -> [1, 0, 3]
  3: rotate Z [1, 0, 3] -> [1, 1, 4]
  child: Stage @ [0, 0, 0] rotate Z -90
  child.q_out -H> other.q_in
  child.bit => other.basis
  branch correction {
    false {
      4: XZ [1, 1, 5]
    }
    true {
      4: YX [1, 1, 5]
    }
  }
  read/path.+- = MeAsUrE 1 -> -X # readout
  combined = !read/path.+- & (child.bit ^ other.bit) | third.bit
  resolve correction if combined
  feedback X 1 -> +Y, Z 2 if !combined
  discard if combined
}
'''
    tokens = list(registered["blog"]().get_tokens(source))
    assert "".join(value for _, value in tokens) == source
    assert not any(kind is Error for kind, _ in tokens)
    for expected in (
        (Keyword, "BLOG"), (Number, "1.0"), (Keyword, "MeAsUrE"),
        (String.Double, '"stage#1.blog"'), (String.Other, "<q#入口>"),
        (Comment.Single, "# readout"), (Name.Attribute, "color"),
        (Number.Hex, "12abEF"), (Number, "3d/2 + 1"), (Number, "-90"),
        (Keyword.Type, "XZ"), (Keyword.Type, "XZX"), (Name.Constant, "-X"),
        (Name, "child.q_out"), (Name, "read/path.+-"), (Punctuation, "{"),
    ):
        assert expected in tokens
    assert {"->", "-H>", "=>", "!", "&", "^", "|"} <= {
        value for kind, value in tokens if kind is Operator
    }
    # Extended names must not be split into keyword/type prefixes.
    names = "module.bit measure/path resolve+flag ZXZ-output input.value discard_count\n"
    assert all(kind is Name for kind, value in registered["blog"]().get_tokens(names)
               if value.strip())


def test_blog_highlighting_preserves_representative_sources():
    from runpy import run_path

    from pygments.token import Error

    docs = API_RST.parent.parent
    lexer = run_path(str(docs / "conf.py"))["BlogLexer"]()
    for path in (
        docs / "examples/quickstart-t.blog",
        docs.parent / "bloq_graph/tests/fixtures/module_classical.blog",
        docs / "modules/two-stages.blog",
        docs / "_static/constructions/walking-xzx-slide-east.blog",
        docs.parent / "bloq_graph/assets/ccz_gate_teleport.blog",
    ):
        source = path.read_text()
        tokens = list(lexer.get_tokens_unprocessed(source))
        assert "".join(value for _, _, value in tokens) == source, path
        assert not any(kind is Error for _, kind, _ in tokens), path


@pytest.mark.parametrize("targets", [[], ["citation-a"], ["citation-a", "citation-b"]])
def test_reference_backlinks_follow_the_text_and_retain_every_citation(targets):
    nodes = pytest.importorskip("docutils.nodes")
    hook = runpy.run_path(str(API_RST.parent.parent / "conf.py"))["_footnote_backlinks"]
    document = nodes.document(None, None)
    footnote = nodes.footnote(backrefs=targets.copy())
    footnote += nodes.label("", "1")
    footnote += nodes.paragraph("", "Paper reference.")
    document += footnote
    app = SimpleNamespace(builder=SimpleNamespace(format="html"))
    hook(app, document, "theory/prerequisites")
    hook(app, document, "theory/prerequisites")
    links = list(footnote.findall(nodes.reference))
    assert footnote["backrefs"] == []
    assert [link["refid"] for link in links] == targets
    assert all(link.parent is footnote[-1] and link.astext() == "↩" for link in links)
    assert footnote[0].astext() == "1"
    assert footnote[-1].astext().startswith("Paper reference.")


def test_zx_illustrations_preserve_cnot_and_local_pauli_identities():
    assets = API_RST.parent.parent / "assets" / "paper"
    svg = ET.parse(assets / "zx-cnot-correspondence.svg")
    topology = json.loads(svg.find('.//*[@id="source-graph"]').text)
    graph = bloq.to_zx_graph(bloq.GalleryItem.CNOT.load())
    assert topology["nodes"] == [
        {"id": node.id, "position": list(node.pos), "basis": node.kind}
        for node in graph.nodes()
    ]
    edges = [tuple(edge) for edge in topology["edges"]]
    assert set(edges) == {(e.n1, e.n2) for e in graph.edges() if e.n1 < e.n2}
    web = {tuple(edge) for edge in topology["web_edges"]}
    assert web <= set(edges)
    matrix = [[0.0] * 4 for _ in range(4)]
    nodes = topology["nodes"]
    incident = {node["id"]: [i for i, edge in enumerate(edges) if node["id"] in edge]
                for node in nodes}
    ports = {tuple(node["position"]): node["id"] for node in nodes if node["basis"] == "Port"}
    boundaries = [ports[pos] for pos in ((0, 0, 0), (1, 1, 0), (0, 0, 3), (1, 1, 3))]
    assert {node for node in boundaries if any(node in edge for edge in web)} == set(boundaries) - {boundaries[1]}
    for bits in itertools.product((0, 1), repeat=len(edges)):
        amplitude = 1.0
        for node in nodes:
            legs = incident[node["id"]]
            support = sum(node["id"] in edge for edge in web)
            values = [bits[i] for i in legs]
            if node["basis"] == "Z":
                assert support in (0, len(legs))
                amplitude *= float(len(set(values)) == 1)
            elif node["basis"] == "X":
                assert support % 2 == 0
                amplitude *= 2 ** (1 - len(legs) / 2) if sum(values) % 2 == 0 else 0
        c_in, t_in, c_out, t_out = [bits[incident[node][0]] for node in boundaries]
        matrix[2 * c_out + t_out][2 * c_in + t_in] += amplitude
    scalar = matrix[0][0]
    assert scalar > 0
    for row, column in itertools.product(range(4), repeat=2):
        expected = scalar if row == (column ^ (column >> 1)) else 0
        assert math.isclose(matrix[row][column], expected, abs_tol=1e-9)

    # Check the displayed local insertions against the four-leg spider tensors.
    rules = ET.parse(assets / "zx-spider-rules.svg")
    for panel in rules.findall('.//*[@data-paulis]'):
        basis, paulis = panel.get("data-basis"), panel.get("data-paulis")
        insertions = panel.findall('.//*[@data-pauli]')
        assert [item.get("data-pauli") for item in insertions] == [p for p in paulis if p != "I"]
        for item in insertions:
            assert item.find("{*}text").text == "π"
            assert item.find("{*}circle").get("fill") == ("#ff7f7f" if item.get("data-pauli") == "X" else "#7396ff")
        vector, transformed = {}, {}
        for bits in itertools.product((0, 1), repeat=4):
            value = float(len(set(bits)) == 1) if basis == "Z" else (0.5 if sum(bits) % 2 == 0 else 0)
            vector[bits] = value
            result = tuple(bit ^ (pauli == "X") for bit, pauli in zip(bits, paulis))
            sign = (-1) ** sum(bit for bit, pauli in zip(bits, paulis) if pauli == "Z")
            transformed[result] = sign * value
        assert transformed == vector
