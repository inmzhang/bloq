"""Sphinx configuration for the bloq website and generated Python reference.

Docs are generated with autodoc from the installed `bloq`
package (docstrings originate in the Rust sources and the pure-Python
facade), grouped by functionality with inline signatures and members.
`api/python.rst` is kept in sync with `bloq` and `bloq.ir` exports by a unit test.

The compiled extension module carries no type annotations at runtime, so
autodoc alone would render bare, untyped signatures. The pyo3-stub-gen
stubs DO carry the full signatures (types inferred from the Rust side),
and they are the single source of truth for typing — so we parse the
`.pyi` files with `ast` and feed every signature to autodoc via the
`autodoc-process-signature` event.
"""

import ast
import copy
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import bloq
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text

project = "bloq"
author = "Yiming Zhang"
copyright = "Bloq contributors"
release = bloq.__version__

# No current cross-reference needs an external intersphinx inventory.
extensions = [
    "sphinx_immaterial",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autodoc_member_order = "groupwise"
object_description_options = [
    ("py:function", {"toc_icon_text": "func"}),
    ("py:class", {"toc_icon_text": "class"}),
    ("py:method", {"toc_icon_text": "meth"}),
    ("py:property", {"toc_icon_text": "prop"}),
    ("py:attribute", {"toc_icon_text": "attr"}),
    ("py:parameter", {"include_in_toc": False}),
    ("py:.*", {"wrap_signatures_column_limit": 60, "include_fields_in_toc": False}),
]
# Bare names everywhere: no `bloq.` prefix in signatures, titles, or sidebar.
add_module_names = False
# Docstrings use single-backtick code spans (rustdoc habit); render them as
# code instead of rst "interpreted text".
default_role = "code"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build", "examples/README.md", "api/_autosummary", "assets/**/*.md",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "html_image", "deflist"]
myst_heading_anchors = 6
myst_fence_as_directive = ["mermaid"]
html_theme = "sphinx_immaterial"
html_domain_indices = False
html_title = "bloq"
html_logo = "assets/logos/bloq-mark-light.png"
html_favicon = "assets/logos/bloq-mark-light.png"
html_static_path = ["_static", "assets/logos"]
html_css_files = ["site.css"]
html_js_files = ["versions.js", "graph-viewer.js"]
html_copy_source = False
html_show_sourcelink = False
html_context = {
    "bloq_docs_version": os.environ.get("BLOQ_DOCS_VERSION", "dev"),
    "bloq_package_version": release,
}
html_theme_options = {
    "repo_url": "https://github.com/inmzhang/bloq",
    "repo_name": "inmzhang/bloq",
    "scope": "/",
    "features": ["navigation.tabs", "navigation.sections", "navigation.top", "search.highlight", "content.tabs.link", "content.code.copy"],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default", "primary": "white", "accent": "custom",
            "toggle": {"icon": "material/brightness-7", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate", "primary": "black", "accent": "custom",
            "toggle": {"icon": "material/brightness-4", "name": "Switch to light mode"},
        },
    ],
    "globaltoc_collapse": False,
    "toc_title": "On this page",
}

# ==============================================================================
# Stub-derived signatures
# ==============================================================================

_PKG = Path(__file__).parent.parent / "bloq_py" / "python" / "bloq"
_STUB_SOURCES = [_PKG / "_core" / "__init__.pyi", _PKG / "_gallery.py"]


def _clean(sig: str) -> str:
    # The stubs fully qualify builtins ("builtins.int") and typing names;
    # strip the prefixes so rendered signatures read like normal Python.
    return sig.replace("builtins.", "").replace("typing.", "")


def _signature_of(fn: ast.FunctionDef, drop_self: bool) -> tuple[str, str | None]:
    args = fn.args
    if drop_self and args.args and args.args[0].arg in ("self", "cls"):
        remaining = len(args.args) - 1
        args = ast.arguments(
            posonlyargs=args.posonlyargs,
            args=args.args[1:],
            vararg=args.vararg,
            kwonlyargs=args.kwonlyargs,
            kw_defaults=args.kw_defaults,
            defaults=args.defaults[-remaining:] if remaining > 0 else [],
            kwarg=args.kwarg,
        )
    sig = f"({_clean(ast.unparse(args))})"
    ret = _clean(ast.unparse(fn.returns)) if fn.returns else None
    return sig, ret


def _collect_stub_signatures() -> dict[str, tuple[str, str | None]]:
    """Map 'Name' / 'Class.method' → (signature, return annotation)."""
    sigs: dict[str, tuple[str, str | None]] = {}
    for path in _STUB_SOURCES:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sigs[node.name] = _signature_of(node, drop_self=False)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sigs[f"{node.name}.{item.name}"] = _signature_of(
                            item, drop_self=True
                        )
                # A class documents itself with its constructor signature.
                for ctor in ("__init__", "__new__"):
                    if (key := f"{node.name}.{ctor}") in sigs:
                        sigs[node.name] = sigs[key]
                        break
    return sigs


_STUB_SIGS = _collect_stub_signatures()


def _process_signature(app, what, name, obj, options, signature, return_annotation):
    # Properties and attributes render without call signatures.
    if what in ("property", "attribute", "data"):
        return None
    # `name` is fully qualified ("bloq.BlockGraph.add_block"); the stub map
    # is keyed by the trailing one or two components.
    parts = name.split(".")
    for key in (".".join(parts[-2:]), parts[-1]):
        if key in _STUB_SIGS:
            sig, ret = _STUB_SIGS[key]
            # A class renders its constructor args; a `-> Class` return
            # annotation there is just noise.
            return sig, None if what == "class" else ret
    return None


def _strip_builtin_data_docstrings(app, what, name, obj, options, lines):
    """Keep authored constant docs, but omit their built-in value-type docs."""
    if what not in ("data", "attribute") or type(obj).__module__ != "builtins":
        return
    from sphinx.util.docstrings import prepare_docstring

    docstring = type(obj).__doc__
    if isinstance(docstring, str) and lines == prepare_docstring(docstring):
        lines.clear()


def _signature_type_context(app, doctree):
    """Resolve constructor type names in their own public namespace."""
    from sphinx import addnodes

    for signature in doctree.findall(addnodes.desc_signature):
        if module := signature.get("module"):
            for reference in signature.findall(addnodes.pending_xref):
                if reference.get("refdomain") == "py":
                    reference["py:module"] = module


def _rewrite_links(app, current, relative_to, source):
    """Keep repository source links valid when Markdown is published as HTML."""
    root = Path(app.srcdir).parent
    destinations = {
        "CONTRIBUTING.md": "development/contributing.md",
        "bloq_py/README.md": "getting-started/installation.md",
    }
    ref = os.environ.get("BLOQ_DOCS_REF", "main")

    def replace(match):
        target = match.group(1)
        url = urlsplit(target)
        if url.scheme or target.startswith(("#", "//", "/")):
            return match.group(0)
        path = (current.parent / url.path).resolve()
        if path.is_relative_to(Path(app.srcdir).resolve()):
            if current == relative_to:
                return match.group(0)
            target = os.path.relpath(path, relative_to.parent)
        elif not path.is_file():
            return match.group(0)
        elif not path.is_relative_to(root.resolve()):
            return match.group(0)
        else:
            relative = path.relative_to(root.resolve()).as_posix()
            if relative in destinations:
                destination = Path(app.srcdir) / destinations[relative]
                target = os.path.relpath(destination, relative_to.parent)
            else:
                target = f"https://github.com/inmzhang/bloq/blob/{ref}/{relative}"
        if url.fragment:
            target += "#" + url.fragment
        return "](" + target + ")"

    source[0] = re.sub(r"\]\(([^\s)]+)\)", replace, source[0])


def _source_links(app, docname, source):
    current = Path(app.env.doc2path(docname))
    _rewrite_links(app, current, current, source)


def _include_links(app, relative_path, parent_docname, source):
    current = Path(app.srcdir) / relative_path
    relative_to = Path(app.env.doc2path(app.env.docname))
    _rewrite_links(app, current, relative_to, source)


def _footnote_backlinks(app, doctree, docname):
    """Put return arrows after each reference, retaining every citation target."""
    if app.builder.format != "html":
        return
    from docutils import nodes

    for footnote in doctree.findall(nodes.footnote):
        backrefs = footnote.get("backrefs", [])
        if not backrefs:
            continue
        footnote["backrefs"] = []
        if not isinstance(footnote[-1], nodes.paragraph):
            footnote += nodes.paragraph()
        for index, target in enumerate(backrefs, 1):
            footnote[-1] += nodes.Text(" ")
            footnote[-1] += nodes.reference(
                "", "↩", refid=target, classes=["footnote-backlink"],
                reftitle=f"Back to citation {index}",
            )


def _api_cards(app, doctree, docname):
    """Keep native Python targets and cross-references, with separate API titles."""
    if docname != "api/python":
        return
    from docutils import nodes
    from sphinx import addnodes

    kinds = {"function": "func", "class": "class", "exception": "class",
             "method": "meth", "property": "prop", "attribute": "attr", "data": "data"}
    for description in doctree.findall(addnodes.desc):
        if description.get("domain") != "py":
            continue
        for signature in list(description.children):
            if not isinstance(signature, addnodes.desc_signature) or not signature.get("ids"):
                continue
            title = nodes.rubric(classes=["bloq-api-title"])
            kind = kinds.get(description.get("objtype"), "data")
            title += nodes.inline("", kind, classes=["bloq-api-kind", kind])
            title += nodes.Text(" ")
            title += nodes.literal("", signature["ids"][0])
            description.insert(description.index(signature), title)
            signature["classes"].append("bloq-api-signature")


def _api_aliases(app, exception):
    """Retain old per-object URLs while the canonical reference lives inline."""
    if exception is not None or app.builder.format != "html":
        return
    directory = Path(app.outdir) / "api" / "_autosummary"
    directory.mkdir(parents=True, exist_ok=True)
    aliases = {f"bloq.{name}": f"bloq.{name}" for name in bloq.__all__}
    aliases["bloq.ir"] = "module-bloq.ir"
    aliases["bloq.StimDialect"] = "bloq.emit_plan_stim"
    for name in bloq.ir.__all__:
        aliases[f"bloq.{name}"] = f"bloq.ir.{name}"
        aliases[f"bloq.ir.{name}"] = f"bloq.ir.{name}"
    for old, current in aliases.items():
        target = f"../python.html#{current}"
        (directory / f"{old}.html").write_text(
            '<!doctype html><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0; url={html.escape(target, quote=True)}">'
            f'<script>const target = new URL({json.dumps(target)}, location.href);'
            f'if (location.hash) target.hash = location.hash.replace({json.dumps(old)}, {json.dumps(current)});'
            'location.replace(target.href);</script>'
            f'<a href="{html.escape(target, quote=True)}">{html.escape(old)}</a>\n'
        )


def _guide_navigation(app, pagename, templatename, context, doctree):
    """Keep the guide's chapter groups in native Material section navigation."""
    def reparent(entry):
        if entry.aria_label == "Tutorials":
            entry.caption_only = True
        if entry.aria_label == "Backend Emission":
            entry.children = []  # Detailed workflow pages stay linked from the chapter.
        for child in entry.children:
            child.parent = entry
            reparent(child)

    for entry in context.get("nav", []):
        if entry.aria_label != "User Guide":
            continue
        # The theme copies entries but leaves parent links pointing at its cache.
        reparent(entry)
        entry.caption_only = True
        # Sphinx prunes Overview <self> as local content; keep it as a chapter link.
        overview = copy.copy(entry)
        overview.title = "<span>Overview</span>"
        overview.aria_label = "Overview"
        overview.children = []
        overview.parent = entry
        overview.caption_only = False
        overview.current = overview.active = pagename == "user-guide"
        entry.children.insert(0, overview)


def _surface_indices(graph, word):
    """Select logical correlation support in lexicographic Port-position order."""
    ports = sorted(block.pos for block in graph.blocks() if block.kind.is_port)
    paulis = {"I": 0, "X": 1, "Z": 2, "Y": 3}
    if len(word) != len(ports) or any(letter not in paulis for letter in word) or set(word) <= {"I"}:
        raise ValueError("surface must be a nonempty Pauli word over the graph's sorted Ports")
    target = sum(paulis[letter] << (2 * i) for i, letter in enumerate(word))
    basis = {}
    for index, row in enumerate(graph.stabilizers()):
        if row.kind != "logical":
            continue
        support = row.stabilizer.port_stabilizer
        vector = sum(paulis[str(support.get(port, "I")).split(".")[-1]] << (2 * i)
                     for i, port in enumerate(ports))
        mask = 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector, mask
                break
            vector ^= basis[pivot][0]
            mask ^= basis[pivot][1]
    mask = 0
    while target:
        pivot = target.bit_length() - 1
        if pivot not in basis:
            raise ValueError(f"surface {word} is outside the graph's logical correlation span")
        vector, rows = basis[pivot]
        target ^= vector
        mask ^= rows
    return [index for index in range(mask.bit_length()) if mask & (1 << index)]


# Flowchart keywords that break parsing when used as a node id, e.g. `end["..."]`.
_MERMAID_KEYWORD_ID = re.compile(
    r"(?<![\w-])(end|graph|subgraph|flowchart|style|class|classDef|click|linkStyle|direction)\s*[\[({>]"
)


class BlogLexer(RegexLexer):
    """Highlight BLOG source with the site's existing Pygments theme."""

    name = "BLOG"
    aliases = ["blog"]
    filenames = ["*.blog"]
    flags = re.MULTILINE | re.IGNORECASE
    # Names may contain slashes, dots, pluses, and hyphens, including after keywords.
    _name_end = r"(?![A-Za-z0-9_/.+-])"
    tokens = {"root": [
        (r"\s+", Text),
        (r"#[^\r\n]*", Comment.Single),
        (r'"[^"\r\n]*"', String.Double),
        (r"<[^\s<>]+>", String.Other),
        (r"-H>|->|=>|[=!&^|@]", Operator),
        (r"[+-][XYZ]" + _name_end, Name.Constant),
        (words(("BLOG", "module", "import", "as", "in", "out", "branch", "false", "true",
                "measure", "resolve", "feedback", "discard", "if", "walk", "rotate"),
               suffix=_name_end), Keyword),
        (words(("XZZ", "ZXZ", "ZZX", "ZXX", "XZX", "XXZ", "Port", "X", "Y", "Z", "T",
                "XY", "YX", "XZ", "ZX", "YZ", "ZY"), suffix=_name_end), Keyword.Type),
        (r"(color)([ \t]*)(=)([ \t]*)([0-9a-f]{6})" + _name_end,
         bygroups(Name.Attribute, Text, Operator, Text, Number.Hex)),
        (words(("height", "role"), suffix=r"(?=[ \t]*=)"), Name.Attribute),
        (words(("auto", "input", "output", "multiplex"), suffix=_name_end), Name.Builtin),
        (r"\d*d(?:[ \t]*/[ \t]*\d+)?(?:[ \t]*[+-][ \t]*\d+)?" + _name_end, Number),
        (r"-?\d+(?:\.\d+)?", Number),
        (r"[A-Za-z_][A-Za-z0-9_/.+-]*", Name),
        (r"[{}\[\]():,]", Punctuation),
    ]}


def setup(app):
    from docutils import nodes
    from docutils.parsers.rst import directives
    from pygments.lexers import TextLexer
    from pygments.lexers import RustLexer
    from sphinx_immaterial.mermaid_diagrams import MermaidDirective
    from sphinx.util import relative_uri
    from sphinx.util.docutils import SphinxDirective

    class BlockGraphView(SphinxDirective):
        """Embed the public source or module-view exporter."""
        required_arguments = 1
        option_spec = {"modules": directives.flag, "source": directives.path,
                       "surface": directives.unchanged, "measurement": directives.unchanged,
                       "pop-faces": directives.unchanged}

        def run(self):
            name = self.arguments[0]
            if "source" in self.options:
                source = (Path(app.srcdir) / self.options["source"]).resolve()
                self.env.note_dependency(str(source))
                graph = bloq.BlockGraph.load(source)
            else:
                graph = bloq.GalleryItem(name).load()
            modules = "modules" in self.options
            surface = self.options.get("surface")
            measurement = self.options.get("measurement")
            if sum((modules, bool(surface), bool(measurement))) > 1:
                raise self.error("Choose one of modules, surface, or measurement")
            filename = f"{name}{'-modules' if modules else ''}{'-' + (surface or measurement) if surface or measurement else ''}.html"
            destination = Path(app.outdir) / "_static" / "block-graphs" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if modules:
                graph.export_html_viewer(destination, module_view=True)
            else:
                flat = graph.flatten()
                if measurement:
                    branches = [(action.branch_target, True) for action in flat.actions()
                                if action.kind == "branch"]
                    if branches:
                        flat = flat.project_branches(branches)
                    selected = next((row for row in flat.stabilizers()
                                     if row.measurement_name == measurement), None)
                    if selected is None:
                        raise self.error(f"Unknown named measurement {measurement}")
                else:
                    selected = _surface_indices(flat, surface) if surface else None
                flat.export_html_viewer(destination,
                    stabilizer=selected,
                    pop_faces_at_directions=self.options.get("pop-faces", "").split())
            url = relative_uri(f"{self.env.docname}.html", f"_static/block-graphs/{filename}")
            view = f"correlation surface {surface or measurement}" if surface or measurement else "module view" if modules else "block graph"
            title = html.escape(f"Interactive {name.replace('_', ' ')} {view}", quote=True)
            return [nodes.raw("", (
                f'<iframe class="bloq-block-view" src="{url}" title="{title}" '
                'loading="lazy" allowfullscreen></iframe>'
            ), format="html")]

    class GalleryBlog(SphinxDirective):
        """Print the complete resolved source from the native gallery."""
        required_arguments = 1

        def run(self):
            text = bloq.GalleryItem(self.arguments[0]).load().to_text()
            source = nodes.literal_block(text, text)
            source["language"] = "blog"
            return [source]

    class DetectorSlices(SphinxDirective):
        """Use the existing SVG canvas for a compiled construction diagram."""
        required_arguments = 1

        def run(self):
            name = self.arguments[0]
            manifest = Path(app.srcdir) / "_static" / "constructions" / "manifest.json"
            self.env.note_dependency(str(manifest))
            case = next(item for item in json.loads(manifest.read_text())["examples"] if item["name"] == name)
            urls = {key: relative_uri(f"{self.env.docname}.html", f"_static/constructions/{case[key]}")
                    for key in ("svg", "source", "stim")}
            label = html.escape(name.replace("-", " "), quote=True)
            content = (f'<div class="bloq-ir-graph" role="region" aria-label="{label} detector slices">'
                       f'<img loading="lazy" src="{urls["svg"]}" alt="{label}: representative detector-slice moments" /></div>'
                       f'<p><a href="{urls["source"]}">BLOG source</a> · <a href="{urls["stim"]}">Complete Stim circuit</a></p>')
            return [nodes.raw("", content, format="html")]

    class CheckedMermaid(MermaidDirective):
        """Reject keyword node ids, which Mermaid only reports in the browser."""

        def run(self):
            # Labels may contain any word; only bare identifiers are node ids.
            code = re.sub(r'"[^"]*"|\|[^|]*\|', "", "\n".join(self.content))
            keyword = re.search(_MERMAID_KEYWORD_ID, code)
            if keyword:
                raise self.error(f"Mermaid keyword {keyword.group(1)!r} cannot be a node id; rename the node")
            return super().run()

    app.add_directive("mermaid", CheckedMermaid)
    app.add_directive("bloq-view", BlockGraphView)
    app.add_directive("gallery-blog", GalleryBlog)
    app.add_directive("detector-slices", DetectorSlices)

    app.add_lexer("blog", BlogLexer)
    for language in ("bloqir", "stim"):
        app.add_lexer(language, TextLexer)
    for language in ("rust,ignore", "rust,no_run"):
        app.add_lexer(language, RustLexer)
    app.connect("autodoc-process-signature", _process_signature)
    app.connect("autodoc-process-docstring", _strip_builtin_data_docstrings, priority=100)
    app.connect("doctree-read", _signature_type_context)
    app.connect("source-read", _source_links)
    app.connect("include-read", _include_links)
    app.connect("doctree-resolved", _api_cards)
    app.connect("doctree-resolved", _footnote_backlinks)
    app.connect("build-finished", _api_aliases)
    app.connect("html-page-context", _guide_navigation, priority=600)
