"""Integration tests for the bloq bindings: exercise the real pipeline
(gallery/BLOG -> BlockGraph -> compile -> Bloq IR -> stim) rather than
mocking any layer. Distances stay small so the suite runs in seconds.
"""

import copy
import errno
import pickle
import sys
import warnings
from collections.abc import Hashable

import pytest
import stim

import bloq
import bloq._core


def test_ir_svg_preserves_dynamic_structure_and_filters_classical_nodes():
    from xml.etree import ElementTree

    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    before = program.to_binary()
    full = program.to_svg()
    quantum = program.to_svg(include_classical=False)
    assert ElementTree.fromstring(full).tag == "{http://www.w3.org/2000/svg}svg"
    assert ElementTree.fromstring(quantum).tag == "{http://www.w3.org/2000/svg}svg"
    assert "RepeatUntilSuccess" in full and "RepeatUntilSuccess" in quantum
    assert "Observable" in full and "Observable" not in quantum
    assert "Output frame" in full and "Output frame" not in quantum
    assert program.to_binary() == before


def test_stats_inventory_dynamic_ir_without_flattening():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    before = program.to_binary()
    stats = program.stats()
    assert isinstance(stats, bloq.ir.BloqStats)
    assert stats.node_count == len(program.walk()) > program.node_count
    assert stats.edge_count == sum(level.edge_count for _, level in program.levels())
    assert sum(stats.node_counts.values()) == stats.node_count
    assert sum(stats.edge_counts.values()) == stats.edge_count
    assert stats.template_count == program.template_count
    assert stats.node_counts["RepeatUntilSuccess"] > 0
    assert not stats.is_static
    assert str(stats).startswith("Bloq IR\n├── Compiled Templates: ")
    assert str(stats).endswith("└── Static: No")
    assert bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3).stats().is_static
    with pytest.raises(AttributeError):
        stats.node_count = 0
    assert program.to_binary() == before


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------


def test_gallery_ids_and_load():
    ids = [g.id for g in bloq.GalleryItem]
    assert "x_memory" in ids and "cnot" in ids
    g = bloq.GalleryItem("cnot").load()
    assert isinstance(g, bloq.BlockGraph)
    assert g.block_count == 10 and g.port_count == 4


def test_gallery_unknown_id_raises():
    with pytest.raises(ValueError, match="not a valid GalleryItem"):
        bloq.GalleryItem("no_such_entry")
    # The `_core` plumbing rejects unknown ids too, for the enum's own lookups.
    with pytest.raises(bloq.InvalidArgumentError, match="unknown"):
        bloq._core.gallery_load("no_such_entry")


def test_gallery_enum_covers_every_core_entry():
    """`GalleryItem` is hand-written; `_core` is the compiled-in truth.

    The enum is the only public spelling of the gallery, so nothing else would
    notice a Rust-side entry it never grew a member for.
    """
    entries = bloq._core.gallery_entries()
    assert {"id", "description", "categories"} <= entries[0].keys()
    assert {e["id"] for e in entries} == {g.id for g in bloq.GalleryItem}
    assert bloq._core.gallery_source("cnot") == bloq.GalleryItem.CNOT.source()
    assert (
        bloq._core.gallery_load("cnot").to_text()
        == bloq.GalleryItem.CNOT.load().to_text()
    )


# ---------------------------------------------------------------------------
# BLOG parse round-trip
# ---------------------------------------------------------------------------


def test_blog_round_trip():
    g = bloq.GalleryItem("bell_state").load()
    text = g.to_text()
    g2 = bloq.BlockGraph.from_text(text)
    assert g2.to_text() == text
    assert g2.block_count == g.block_count


@pytest.mark.parametrize(
    "entry", [bloq.GalleryItem.X_MEMORY, bloq.GalleryItem.CNOT, bloq.GalleryItem.BELL_STATE, bloq.GalleryItem.T_GATE]
)
def test_gallery_executable_blog_parses_to_its_block_graph(entry):
    parsed = bloq.BlockGraph.from_text(entry.source())
    assert parsed == entry.load()


def test_inline_module_hierarchy_is_retained_and_compiles_as_one_input(tmp_path):
    parsed = bloq.BlockGraph.from_text("""BLOG 1.0
module Leaf {
  0: ZXZ [0,0,0]
}
module Twice {
  left: Leaf @ [0,0,0]
  right: Leaf @ [2,0,0]
}
module main {
  first: Twice @ [0,0,0]
  second: Twice @ [4,0,0]
}
""")
    assert parsed.has_module_structure
    assert set(parsed.module_names) == {"main", "Leaf", "Twice"}
    assert parsed.positions() == []
    assert not parsed.is_empty
    with pytest.raises(KeyError, match="absent"):
        parsed.module("absent")
    leaf = parsed.module("Leaf")
    assert leaf.positions() == [(0, 0, 0)]
    leaf.add_block(bloq.Block((1, 0, 0), "ZXZ"))
    assert parsed.module("Leaf").block_count == 1

    flat = parsed.flatten()
    assert not flat.has_module_structure
    assert set(flat.positions()) == {(0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0)}
    assert len(parsed.stabilizers()) == len(flat.stabilizers())
    with pytest.raises(bloq.BlockGraphError, match="expanded blocks limit"):
        parsed.flatten(limits={"max_expanded_blocks": 1})
    assert parsed.flatten(limits={"max_expanded_blocks": None}) == flat
    assert "first: Twice" in parsed.to_text()
    for copied in [copy.copy(parsed), copy.deepcopy(parsed), pickle.loads(pickle.dumps(parsed))]:
        assert copied == parsed
        assert copied.module_names == parsed.module_names

    program = bloq.compile(parsed, distance=3)
    program.validate()
    assert program.qubit_count == bloq.compile(flat, distance=3).qubit_count

    child = parsed.module("Twice")
    with pytest.raises(bloq.BlockGraphError, match="expanded blocks limit"):
        parsed.module("Twice", limits={"max_expanded_blocks": 1})
    assert set(child.module_names) == {"main", "Leaf"}
    assert set(child.flatten().positions()) == {(0, 0, 0), (2, 0, 0)}
    path = tmp_path / "child.blog"
    child.save(path)
    assert bloq.BlockGraph.load(path) == child
    assert pickle.loads(pickle.dumps(child)) == child
    bloq.compile(child, distance=3).validate()


def test_module_parse_errors_keep_the_module_diagnostic():
    with pytest.raises(bloq.ParseError, match="duplicate module"):
        bloq.BlockGraph.from_text("BLOG 1.0\nmodule main {\n}\nmodule main {\n}\n")


def test_graph_load_save_and_relative_imports(tmp_path):
    graph = bloq.GalleryItem.CNOT.load()
    path = tmp_path / "saved.blog"
    graph.save(path)
    assert bloq.BlockGraph.load(path) == graph
    assert bloq.BlockGraph.load(str(path)) == graph

    child = tmp_path / "child.blog"
    child.write_text("BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n}\n")
    path.write_text(
        'BLOG 1.0\nimport "child.blog" as Child\n'
        "module main {\nfirst: Child @ [2,0,0]\n}\n"
    )
    imported = bloq.BlockGraph.load(path)
    assert imported.has_module_structure
    assert "Child" in imported.module_names
    assert imported.flatten().positions() == [(2, 0, 0)]
    imported.save(path)
    assert bloq.BlockGraph.load(path) == imported
    path.write_text(
        'BLOG 1.0\nimport "child.blog" as Child\n'
        "module main {\nfirst: Child @ [2,0,0]\n}\n"
    )
    child.unlink()
    with pytest.raises(FileNotFoundError) as caught:
        bloq.BlockGraph.load(path)
    assert caught.value.filename == str(child)
    assert caught.value.errno == errno.ENOENT
    path.unlink()
    with pytest.raises(FileNotFoundError) as caught:
        bloq.BlockGraph.load(path)
    assert caught.value.filename == str(path)
    path.write_text("not BLOG")
    with pytest.raises(bloq.ParseError, match="syntax error"):
        bloq.BlockGraph.load(path)
    for source, excerpt in [
        ("BLOG 1.0\n0: INVALID [0,0,0]\n", "INVALID [0,0,0]"),
        ("BLOG 1.0\nmodule main {\n0: INVALID [0,0,0]\n}\n", "INVALID [0,0,0]"),
        ("BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n0: ZXZ [2,0,0]\n}\n", "ZXZ [2,0,0]"),
    ]:
        path.write_text(source)
        with pytest.raises(bloq.ParseError) as loaded:
            bloq.BlockGraph.load(path)
        with pytest.raises(bloq.ParseError) as parsed:
            bloq.BlockGraph.from_text(source)
        assert str(loaded.value) == str(parsed.value)
        assert excerpt in str(loaded.value)
        assert "here" in str(loaded.value)


@pytest.mark.parametrize(
    "duplicate", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_pauli_string_copy_and_pickle_keep_independent_storage(duplicate):
    original = bloq.PauliString("XZ_Y")
    copied = duplicate(original)
    assert copied == original
    copied[0] = bloq.Pauli.Z
    assert str(original) == "XZ_Y"
    assert str(copied) == "ZZ_Y"


def test_parse_error_plain_message():
    with pytest.raises(bloq.ParseError) as ei:
        bloq.BlockGraph.from_text("not a blog file")
    msg = str(ei.value)
    assert "syntax error" in msg
    assert "\x1b" not in msg  # ANSI stripped for log-friendly exceptions


# ---------------------------------------------------------------------------
# BlockGraph CRUD, validation, errors
# ---------------------------------------------------------------------------


def test_graph_crud_and_validate():
    g = bloq.BlockGraph()
    pos = g.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    assert pos == (0, 0, 0) and len(g) == 1
    g.add_block(bloq.Block((0, 0, 1), bloq.BlockKind.cube("ZXZ")))
    g.add_pipe(bloq.Pipe((0, 0, 0), "+Z"))
    assert g.has_pipe_between((0, 0, 0), (0, 0, 1))
    g.validate()

    # Direction strings are case-insensitive, like every other
    # enum-or-string argument.
    assert bloq.Pipe((0, 0, 0), "+z") == bloq.Pipe((0, 0, 0), "+Z")
    assert bloq.Direction.parse("-x") == bloq.Direction.X_MINUS

    assert g.get_block((5, 5, 5)) is None
    assert g.remove_block((5, 5, 5)) is None
    removed = g.remove_block((0, 0, 1))
    assert removed is not None and removed.pos == (0, 0, 1)


@pytest.mark.parametrize("coordinate,direction,step", [(2**31 - 1, "+X", 1), (-2**31, "-X", -1)])
def test_pipe_rejects_overflowing_destination(coordinate, direction, step):
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.Pipe((coordinate, 0, 0), direction)
    pipe = bloq.Pipe((coordinate - step, 0, 0), direction)
    assert pipe.dst == (coordinate, 0, 0)


def test_graph_errors_are_bloq_errors():
    g = bloq.BlockGraph()
    g.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    with pytest.raises(bloq.BlockGraphError):
        g.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    assert issubclass(bloq.BlockGraphError, bloq.BloqError)


def test_spatial_port_roles_are_explicit():
    port = bloq.Block((0, 0, 0), "Port", role="input")
    assert port.role == "input"
    assert bloq.Block((0, 0, 0), "ZXZ").role is None

    graph = bloq.BlockGraph()
    graph.add_block(port)
    graph.set_port_role((0, 0, 0), "output")
    assert graph.get_block((0, 0, 0)).role == "output"
    graph.set_port_role((0, 0, 0), "multiplex")
    assert graph.get_block((0, 0, 0)).role == "multiplex"


@pytest.mark.parametrize("method", ["save", "export_gltf", "export_html_viewer"])
def test_graph_filesystem_errors_are_os_errors(tmp_path, method):
    graph = bloq.GalleryItem("x_memory").load()
    with pytest.raises(OSError) as caught:
        getattr(graph, method)(tmp_path)
    error = caught.value
    assert error.errno is not None
    assert error.filename == str(tmp_path)
    assert isinstance(error, (IsADirectoryError, PermissionError))
    if error.errno == errno.EISDIR:
        assert isinstance(error, IsADirectoryError)


@pytest.mark.parametrize("method", ["export_gltf", "export_html_viewer"])
def test_module_view_export_preserves_hierarchy_and_shared_colors(tmp_path, method):
    import base64
    import json

    graph = bloq.GalleryItem.THREE_BIT_ADDER.load()
    before = graph.to_text()
    output = tmp_path / ("modules.html" if method == "export_html_viewer" else "modules.gltf")
    getattr(graph, method)(output, module_view=True)
    assert graph.to_text() == before
    text = output.read_text()
    if method == "export_html_viewer":
        assert "InjectedAnd" in text and "3 instances" in text
        assert "Highlight main" not in text
        encoded = text.split("data:model/gltf+json;base64,", 1)[1].split('"', 1)[0]
        model = json.loads(base64.b64decode(encoded))
    else:
        model = json.loads(text)
    # Child definitions use the same linearized ownership palette as the editor.
    child_color = [c / 255 for c in (0x44, 0x77, 0xAA)]
    expected = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in child_color]
    materials = model["materials"]
    assert any(m["pbrMetallicRoughness"]["baseColorFactor"][:3] == pytest.approx(expected)
               for m in materials)
    assert all(m.get("alphaMode", "OPAQUE") == "OPAQUE" for m in materials
               if m["pbrMetallicRoughness"]["baseColorFactor"][:3] == pytest.approx(expected))



def test_root_only_module_export_keeps_normal_rendering(tmp_path):
    graph = bloq.GalleryItem.CNOT.load()
    ordinary = tmp_path / "ordinary.gltf"
    modules = tmp_path / "modules.gltf"
    graph.export_gltf(ordinary)
    graph.export_gltf(modules, module_view=True)
    assert modules.read_bytes() == ordinary.read_bytes()
    viewer = tmp_path / "modules.html"
    graph.export_html_viewer(viewer, module_view=True)
    assert '<details class="module-legend"' not in viewer.read_text()


def test_module_view_rejects_a_stabilizer_overlay_before_writing(tmp_path):
    graph = bloq.GalleryItem.CNOT.load().flatten()
    generator = graph.stabilizers()[0]
    output = tmp_path / "invalid.html"
    with pytest.raises(bloq.InvalidArgumentError, match="stabilizer overlay"):
        graph.export_html_viewer(output, stabilizer=generator, module_view=True)
    assert not output.exists()


@pytest.mark.parametrize("method", ["export_gltf", "export_html_viewer"])
def test_correlation_view_accepts_logical_generator_products(tmp_path, method):
    graph = bloq.GalleryItem.CNOT.load().flatten()
    rows = graph.stabilizers()
    before = graph.to_text()
    single, indexed, product = (tmp_path / name for name in ("single", "indexed", "product"))
    export = getattr(graph, method)
    export(single, stabilizer=rows[0])
    export(indexed, stabilizer=[0])
    assert single.read_bytes() == indexed.read_bytes()
    export(product, stabilizer=[1, 2])
    assert product.stat().st_size > 0
    assert product.read_bytes() != single.read_bytes()
    assert graph.to_text() == before
    for selection in ([], [len(rows)]):
        output = tmp_path / "invalid"
        with pytest.raises(bloq.InvalidArgumentError):
            export(output, stabilizer=selection)
        assert not output.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OSError contract")
def test_graph_windows_errors_expose_winerror(tmp_path):
    graph = bloq.GalleryItem("x_memory").load()
    output = tmp_path / "missing" / "graph.blog"
    with pytest.raises(FileNotFoundError) as caught:
        graph.save(output)

    error = caught.value
    assert error.filename == str(output)
    assert error.errno == errno.ENOENT
    assert error.winerror == 3  # ERROR_PATH_NOT_FOUND


def test_exception_hierarchy():
    for exc in (
        bloq.ParseError,
        bloq.BlockGraphError,
        bloq.CompileError,
        bloq.BloqValidationError,
        bloq.TextParseError,
        bloq.BinaryDecodeError,
        bloq.StimEmissionError,
    ):
        assert issubclass(exc, bloq.BloqError)
    assert issubclass(bloq.BloqError, Exception)


# ---------------------------------------------------------------------------
# BlockKind
# ---------------------------------------------------------------------------


def test_block_kind_constructors_and_parse():
    zxz = bloq.BlockKind.cube("ZXZ")
    assert zxz.is_cube and not zxz.is_t
    assert bloq.BlockKind.parse("ZXZ") == zxz
    assert str(zxz) == "ZXZ"
    assert bloq.BlockKind.t().is_t
    assert bloq.BlockKind.y().is_y
    assert bloq.BlockKind.port().is_port
    assert bloq.BlockKind.selective("XY").is_selective
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.BlockKind.parse("BOGUS")


def test_measurement_block_kind():
    mx = bloq.BlockKind.measurement("x")
    assert mx.is_measurement and mx.measurement_basis == bloq.Basis.X
    assert str(mx) == "X" and bloq.BlockKind.parse("X") == mx
    assert bloq.BlockKind.measurement(bloq.Basis.Z).measurement_basis == bloq.Basis.Z
    assert bloq.BlockKind.t().measurement_basis is None
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.BlockKind.measurement("Y")


# ---------------------------------------------------------------------------
# stabilizers / fill_ports_auto / transforms / selectives
# ---------------------------------------------------------------------------


def test_project_branches_exposes_static_false_arm():
    graph = bloq.GalleryItem.CCZ_GATE_TELEPORT.load().flatten()
    targets = [action.branch_target for action in graph.actions() if action.kind == "branch"]
    projected = graph.project_branches([(target, False) for target in targets])

    assert len(targets) == 3
    assert all(
        projected.get_block(target).kind == bloq.BlockKind.measurement("X")
        for target in targets
    )
    assert all(action.kind != "branch" for action in projected.actions())
    projected.stabilizers()


def test_stabilizers():
    gens = bloq.GalleryItem("x_memory").load().stabilizers()
    assert len(gens) == 1
    (gen,) = gens
    assert gen.kind == "logical"
    assert not gen.is_measurement
    assert gen.measurement_name is None
    assert str(gen.stabilizer.paulis) == "X"

    measurement = next(
        gen
        for gen in bloq.GalleryItem("t_gate").load().stabilizers()
        if gen.is_measurement
    )
    assert measurement.kind == "measurement"
    assert measurement.measurement_name == "mzz"


def test_fill_ports_auto():
    bell = bloq.GalleryItem("bell_state").load().flatten()
    assert bell.is_open
    fills = bell.fill_ports_auto()
    assert fills
    filled, stabs = fills[0]
    assert not filled.is_open
    assert filled.port_count == 0
    assert stabs
    assert all(isinstance(stab, bloq.StabilizerGenerator) for stab in stabs)
    filled.validate()


def test_transforms():
    g = bloq.GalleryItem("bell_state").load().flatten()
    shifted = g.shift_positions((1, 2, 3))
    assert sorted(shifted.positions()) == sorted(
        (x + 1, y + 2, z + 3) for (x, y, z) in g.positions()
    )
    assert g.rotate_about_origin(bloq.UDirection.Z, 1).block_count == g.block_count
    # Axis also accepted as a string, like every other direction argument.
    assert g.rotate_about_origin("z", 1).to_text() == g.rotate_about_origin(
        bloq.UDirection.Z, 1
    ).to_text()
    assert g.flip_xz_basis().block_count == g.block_count


def test_coordinate_transforms_reject_i32_overflow():
    max_coord = 2**31 - 1
    min_coord = -(2**31)

    high = bloq.BlockGraph()
    high.add_block(bloq.Block((max_coord, 0, 0), "ZXZ"))
    with pytest.raises(bloq.InvalidArgumentError, match="coordinate range"):
        high.shift_positions((1, 0, 0))

    low = bloq.BlockGraph()
    low.add_block(bloq.Block((min_coord, 0, min_coord), "ZXZ"))
    with pytest.raises(bloq.InvalidArgumentError, match="coordinate range"):
        low.rotate_about_origin("y", 1)

    normalized = low.with_zero_min_z()
    assert normalized.positions() == [(min_coord, 0, 0)]

    wide = bloq.BlockGraph()
    wide.add_block(bloq.Block((0, 0, -2_000_000_000), "ZXZ"))
    wide.add_block(bloq.Block((1, 0, 2_000_000_000), "ZXZ"))
    with pytest.raises(bloq.InvalidArgumentError, match="coordinate range"):
        wide.with_zero_min_z()


def test_pauli_string_unhashable():
    ps = bloq.PauliString("XZ")
    assert ps == bloq.PauliString("XZ")
    assert not isinstance(ps, Hashable)
    with pytest.raises(TypeError):
        hash(ps)


def test_randomly_resolve_selectives_seeded():
    source = bloq.GalleryItem("t_gate").load()
    with pytest.raises(bloq.BlockGraphError, match="flatten"):
        source.randomly_resolve_selectives(seed=1)
    tg = source.flatten()
    assert tg.selective_count == 1
    resolved, replacements = tg.randomly_resolve_selectives(seed=1)
    assert resolved.selective_count == 0
    assert len(replacements) == 1
    again, _ = tg.randomly_resolve_selectives(seed=1)
    assert again.to_text() == resolved.to_text()  # deterministic under a seed


# ---------------------------------------------------------------------------
# compile -> Bloq IR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def y_memory_program():
    g = bloq.GalleryItem("y_memory").load()
    art = bloq.compile(g, distance=3)
    return g, art


def test_compile_returns_the_program(y_memory_program):
    _, p = y_memory_program
    assert isinstance(p, bloq.Bloq)
    assert p.quantum_node_count == 3
    assert p.qubit_count > 0 and p.measurement_count > 0
    p.validate()


def test_compile_context_reuse():
    ctx = bloq.CompileContext(distance=3)
    g = bloq.GalleryItem("x_memory").load()
    p = ctx.compile(g, validate=True)
    assert p == ctx.compile(g)
    assert p == bloq.compile(g, validate=True)
    assert repr(ctx) == "<CompileContext distance=3>"
    assert ctx.distance == 3 and ctx.prepare_t_with_mpps is False
    configured = bloq.CompileContext(distance=5, prepare_t_with_mpps=True)
    assert configured.distance == 5 and configured.prepare_t_with_mpps is True


def test_compiler_options_are_keyword_only():
    graph = bloq.GalleryItem.X_MEMORY.load()
    for call in (
        lambda: bloq.CompileContext(3, True),
        lambda: bloq.compile(graph, 3, True),
        lambda: bloq.CompileContext(3).compile(graph, True),
        lambda: bloq.compile_to_stim(graph, 3, True),
        lambda: bloq.compile_clifford_proxy(graph, [], 3, True),
        lambda: bloq.compile_random_clifford_proxy(graph, 3, True),
    ):
        with pytest.raises(TypeError):
            call()
    assert bloq.compile(graph, 3, prepare_t_with_mpps=True).node_count > 0


def test_compiler_limit_overrides_and_argument_errors():
    graph = bloq.GalleryItem.X_MEMORY.load()
    compile_paths = [
        lambda limits: bloq.compile(graph, limits=limits),
        lambda limits: bloq.compile_to_stim(graph, limits=limits),
        lambda limits: bloq.compile_clifford_proxy(graph, [], limits=limits),
        lambda limits: bloq.compile_random_clifford_proxy(graph, seed=17, limits=limits),
        lambda limits: bloq.CompileContext(limits=limits).compile(graph),
        lambda limits: bloq.CompileContext(limits=limits).compile(graph, validate=True),
        lambda limits: bloq.compile(graph, validate=True, limits=limits),
    ]
    for compile_graph in compile_paths:
        with pytest.raises(bloq.CompileError, match="> 0") as refused:
            compile_graph({"max_expanded_blocks": 0})
        assert "limits=" in str(refused.value)
        compile_graph({"max_expanded_blocks": None})

    with pytest.raises(bloq.CompileError, match="Boolean work steps.*> 0"):
        bloq.compile(graph, limits={"max_boolean_steps": 0})

    field_names = [
        "max_expanded_blocks", "max_expanded_instances", "max_occupied_cells",
        "max_local_columns", "max_boolean_nodes", "max_boolean_steps",
        "max_matrix_words", "max_frontier_width", "max_witness_nodes",
        "max_normalization_states", "max_guarded_domain_size",
    ]
    bloq.CompileContext(limits=dict.fromkeys(field_names)).compile(graph)
    with pytest.raises(bloq.InvalidArgumentError, match="unknown.*not_a_limit"):
        bloq.CompileContext(limits={"not_a_limit": 1})
    with pytest.raises(OverflowError, match="negative"):
        bloq.CompileContext(limits={"max_boolean_steps": -1})
    with pytest.raises(OverflowError):
        bloq.CompileContext(limits={"max_boolean_steps": 1 << 200})
    with pytest.raises(TypeError):
        bloq.CompileContext(limits={"max_boolean_steps": 1.5})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bloq.compile(graph)
    assert not caught


def test_source_limit_overrides_apply_to_graph_bodies_modules_and_imports(tmp_path):
    body = "BLOG 1.0\n0: ZXZ [0,0,0]\n"
    module = "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n}\n"
    for source in [body, module]:
        with pytest.raises(bloq.ParseError, match="> 0") as refused:
            bloq.BlockGraph.from_text(source, limits={"max_expanded_blocks": 0})
        assert "limits=" in str(refused.value)
        assert bloq.BlockGraph.from_text(
            source, limits={"max_expanded_blocks": None}
        ).block_count == 1

    (tmp_path / "child.blog").write_text(module)
    root = tmp_path / "root.blog"
    root.write_text('BLOG 1.0\nimport "child.blog" as child\nmodule main {\n}\n')
    with pytest.raises(bloq.ParseError, match="> 0") as refused:
        bloq.BlockGraph.load(root, limits={"max_expanded_blocks": 0})
    assert "limits=" in str(refused.value)
    assert bloq.BlockGraph.load(root, limits={"max_expanded_blocks": None}).is_empty


@pytest.mark.parametrize("validate", [False, True])
def test_spatial_hadamard_compile_warns_about_distance_loss(validate):
    graph = bloq.BlockGraph()
    graph.add_block(bloq.Block((0, 0, 0), "XZX"))
    graph.add_block(bloq.Block((1, 0, 0), "XXZ"))
    graph.add_pipe(bloq.Pipe((0, 0, 0), "+X", hadamard=True))

    with pytest.warns(bloq.CompileWarning, match="fixed-bulk spatial Hadamard"):
        bloq.compile(graph, distance=3, validate=validate)

    context = bloq.CompileContext(distance=3)
    with pytest.warns(bloq.CompileWarning, match="fixed-bulk spatial Hadamard"):
        context.compile(graph, validate=validate)

    # Filterable on its own category, and blamed on the caller's line rather
    # than on the extension.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bloq.compile(graph, distance=3, validate=validate)
    assert [w.category for w in caught] == [bloq.CompileWarning]
    assert caught[0].filename == __file__


def test_compile_clifford_proxy():
    tg = bloq.GalleryItem("t_gate").load()
    art = bloq.compile_clifford_proxy(tg, pins=[True], distance=3)
    art.validate()


def test_compile_random_clifford_proxy_is_seeded_and_alignable():
    graph = bloq.GalleryItem("t_gate").load()
    proxy = bloq.compile_random_clifford_proxy(graph, distance=3, seed=17)

    assert proxy.metadata[bloq.ir.CLIFFORD_PROXY_SEED_METADATA_KEY] == 17
    assert "TICK" in str(bloq.emit_stim(proxy, align_moments=True))


def test_noisy_stim_segments_preserve_node_boundaries():
    program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    clean = bloq.emit_stim_segments(program, noise=0)
    noisy = bloq.emit_stim_segments(program, noise=1e-3)

    assert stim.Circuit(clean.to_text()) == bloq.emit_stim(program)
    assert len(noisy.segments) == len(clean.segments)
    assert "DEPOLARIZE" in "".join(s.text for s in noisy.segments)
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.emit_stim_segments(program, noise=float("nan"))


def test_noise_and_validation_compose_on_one_emission():
    """The two emission knobs are independent, so both can apply at once."""
    program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    reloaded = bloq.Bloq.from_binary(program.to_binary())

    circuit = bloq.emit_stim(reloaded, noise=1e-3, validate=True)

    assert "DEPOLARIZE" in str(circuit)
    assert circuit == stim.Circuit(bloq.emit_stim_segments(program, noise=1e-3).to_text())


def test_stim_segment_spans_partition_the_program():
    program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    segments = bloq.emit_stim_segments(program)

    assert isinstance(segments, bloq.StimSegments)
    assert len(segments) == len(segments.segments)
    assert [s.node_id for s in segments.segments] == program.deterministic_emit_order()
    column = 0
    for segment in segments.segments:
        assert segment.measurement_start == column
        column += segment.measurement_count
    assert column == segments.num_measurements


def test_emit_stim_segments_pair_agrees_with_emitting_twice():
    program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    clean, noisy = bloq.emit_stim_segments_pair(program, noise=1e-3)

    assert stim.Circuit(clean.to_text()) == bloq.emit_stim(program)
    assert [(s.node_id, s.measurement_start) for s in noisy.segments] == [
        (s.node_id, s.measurement_start) for s in clean.segments
    ]
    assert "DEPOLARIZE" in noisy.to_text()
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.emit_stim_segments_pair(program, noise=2.0)


def test_compile_to_stim_matches_the_two_step_pipeline():
    graph = bloq.GalleryItem.X_MEMORY.load()
    assert isinstance(bloq.compile_to_stim(graph), stim.Circuit)
    assert bloq.compile_to_stim(graph) == bloq.compile_to_stim(graph, 3)
    assert bloq.compile_to_stim(graph) == bloq.emit_stim(bloq.compile(graph))
    assert bloq.compile_to_stim(graph, noise=0.001) == bloq.emit_stim(
        bloq.compile(graph), noise=0.001
    )
    with pytest.raises(bloq.InvalidArgumentError, match="noise must be finite and in"):
        bloq.compile_to_stim(graph, noise=float("nan"))


def test_is_valid_distance_matches_what_compile_accepts():
    graph = bloq.GalleryItem.X_MEMORY.load()
    candidates = (-10**100, -1, 1, 2, 3, 4, 5, 256, 1000, 10**100)
    assert [d for d in candidates if bloq.is_valid_distance(d)] == [3, 5]
    for invalid in ("3", 3.0, None):
        with pytest.raises(TypeError):
            bloq.is_valid_distance(invalid)
    for rejected in (1, 2, 4, 1000):
        with pytest.raises(bloq.InvalidArgumentError):
            bloq.compile(graph, distance=rejected)


def test_bloq_text_binary_round_trip(y_memory_program, tmp_path):
    _, art = y_memory_program
    p = art
    assert bloq.Bloq.from_text(p.to_text()).to_text() == p.to_text()
    assert bloq.Bloq.from_binary(p.to_binary()).to_text() == p.to_text()

    text_path = tmp_path / f"prog.{bloq.ir.BLOQ_TEXT_EXTENSION}"
    bin_path = tmp_path / f"prog.{bloq.ir.BLOQ_BINARY_EXTENSION}"
    p.save(text_path)
    p.save(bin_path)
    assert bloq.Bloq.load(text_path).to_text() == p.to_text()
    assert bloq.Bloq.load(bin_path).to_text() == p.to_text()

    with pytest.raises(bloq.TextParseError):
        bloq.Bloq.from_text("not bloqir")


def test_ir_introspection(y_memory_program):
    _, art = y_memory_program
    p = art
    nodes = p.nodes()
    assert [nid for nid, _ in nodes] == p.node_ids()
    for nid, node in nodes:
        assert isinstance(node.kind, bloq.ir.BloqNodeKind)
        assert isinstance(node.provenance, bloq.ir.NodeProvenance)
        # isinstance-dispatch over the kind tree
        assert (
            isinstance(node.kind, bloq.ir.BloqNodeKind.Quantum) == node.is_quantum()
        )
    assert len(p.quantum_nodes()) == p.quantum_node_count
    edges = p.edges()
    assert len(edges) == p.edge_count
    for ref in edges:
        assert isinstance(ref.edge, bloq.ir.BloqEdge)
        assert (ref.source, ref.target) in {
            (e.source, e.target) for e in p.outgoing(ref.source)
        }
    assert p.sorted_layout_coords()
    assert p.deterministic_emit_order()

    tall = bloq.BlockGraph()
    tall.add_block(bloq.Block((0, 0, 0), "ZXZ", height="3d/2"))
    _, node = bloq.compile(tall, distance=5).quantum_nodes()[0]
    assert node.timeline == [3, 8]


def test_insert_memory_rounds_bad_endpoint(y_memory_program):
    g, _ = y_memory_program
    p = bloq.compile(g, distance=3)
    # Edge (2, 4) is an Order edge: memory rounds splice a quantum seam, and
    # only a quantum seam carries the padding provenance they are measured
    # against.
    with pytest.raises(bloq.BloqError, match="is not a single non-empty seam"):
        p.insert_memory_rounds(2, 4, rounds=2)


def test_insert_memory_rounds_on_a_y_block_seam(y_memory_program):
    g, _ = y_memory_program
    p = bloq.compile(g, distance=3)
    # Edge (0, 1) leaves the Y block. A non-cube endpoint waits on the plain
    # terminal face of its cube end, so the seam is padded like any other.
    new_id = p.insert_memory_rounds(0, 1, rounds=2)
    assert p.node(new_id).memory_rounds() == 2
    p.validate()


# ---------------------------------------------------------------------------
# backend-authoring read surface
# ---------------------------------------------------------------------------


def test_template_and_circuit_reads(y_memory_program):
    _, art = y_memory_program
    p = art
    assert p.template_count > 0
    template = p.template(0)
    assert isinstance(template, bloq.ir.Template)
    circuit = template.circuit
    assert circuit.num_qubits == len(circuit.qubits())
    # Measurement record ids are all in range and the side tables reference
    # only in-range measurements — the invariant a backend relies on.
    assert all(mid < circuit.num_measurements for mid, _ in circuit.measurement_records())
    assert all(
        m < circuit.num_measurements
        for d in template.detectors
        for m in d.parity.measurements
    )
    # Ops default to the entry body; every op is a CircuitOp variant.
    ops = circuit.ops()
    variants = (
        bloq.ir.CircuitOp.Gate,
        bloq.ir.CircuitOp.Measure,
        bloq.ir.CircuitOp.Mpp,
        bloq.ir.CircuitOp.Tick,
        bloq.ir.CircuitOp.Repeat,
        bloq.ir.CircuitOp.ConditionalPauli,
    )
    assert all(isinstance(op, variants) for op in ops)
    with pytest.raises(bloq.InvalidArgumentError, match="unknown template id"):
        p.template(p.template_count)
    with pytest.raises(bloq.InvalidArgumentError, match="unknown circuit body"):
        circuit.ops(circuit.body_count)


def test_template_exposes_compiler_authored_boundary_flows():
    p = bloq.compile_clifford_proxy(
        bloq.GalleryItem("t_gate").load(), pins=[False], distance=3
    )
    output_port = p.logical_outputs()[0].port
    output_id, output = next(
        (node_id, node)
        for node_id, node in p.nodes()
        if node.is_quantum() and output_port in node.block_members()
    )
    instance = output.kind.node.instances[0]
    flows = p.template(instance.template_id).boundary_flows
    terminal = [
        flow
        for flow in flows
        if flow.start and not flow.end and flow.marker == "detector"
    ]
    plan = p.emission_plan(output_id)

    assert len(terminal) == 3**2 - 1
    assert all(flow.measurements for flow in terminal)
    assert all(
        (instance.id, measurement) in plan.measurements
        for flow in terminal
        for measurement in flow.measurements
    )


def test_emission_plan_top_level(y_memory_program):
    _, art = y_memory_program
    p = art
    # Emitting a top-level quantum node yields a merged circuit whose
    # instance-space maps cover every instance the node places.
    quantum_ids = [nid for nid, _ in p.quantum_nodes()]
    plan = p.emission_plan(quantum_ids[0])
    assert isinstance(plan, bloq.ir.EmissionPlan)
    assert plan.circuit.num_measurements > 0
    assert len(plan.measurements) == plan.circuit.num_measurements


@pytest.mark.parametrize("noise", [None, 0.001])
def test_emission_plan_nested_region(noise):
    # A T-gate program lowers to a RepeatUntilSuccess region; its body holds
    # quantum nodes reachable only through a non-empty WalkNode.path.
    p = bloq.compile(bloq.GalleryItem("t_gate").load(), distance=3)
    nested = [w for w in p.walk() if w.path and w.node.is_quantum()]
    assert nested, "t_gate should lower to nested quantum nodes"
    w = nested[0]
    plan = p.emission_plan(w.id, w.path, noise=noise)
    assert isinstance(plan, bloq.ir.EmissionPlan)
    assert plan.circuit.num_measurements > 0


def test_emission_plan_materializes_optional_noise(y_memory_program):
    _, art = y_memory_program
    p = art
    node_id = next(iter(p.quantum_nodes()))[0]

    assert str(p.emission_plan(node_id, noise=0).circuit) == str(
        p.emission_plan(node_id).circuit
    )
    circuit = p.emission_plan(node_id, noise=0.001).circuit
    assert any(
        isinstance(
            op,
            (
                bloq.ir.CircuitOp.Depolarize1,
                bloq.ir.CircuitOp.Depolarize2,
                bloq.ir.CircuitOp.PauliError,
            ),
        )
        for body in range(circuit.body_count)
        for op in circuit.ops(body)
    )
    for noise in (-0.001, 1.001, float("inf"), float("nan")):
        with pytest.raises(bloq.InvalidArgumentError, match="noise must be finite and in"):
            p.emission_plan(node_id, noise=noise)


def test_emission_plan_bad_path():
    p = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    region_ids = [
        nid for nid, _ in p.nodes() if isinstance(p.node(nid).kind, bloq.ir.BloqNodeKind.Region)
    ]
    quantum_ids = [nid for nid, _ in p.quantum_nodes()]
    with pytest.raises(bloq.InvalidArgumentError, match="no node .* at the given level"):
        p.emission_plan(9999)
    assert region_ids
    with pytest.raises(bloq.InvalidArgumentError, match="unknown body selector"):
        p.emission_plan(0, [(region_ids[0], "not_a_body")])
    # A non-region node cannot appear as a path hop.
    with pytest.raises(bloq.InvalidArgumentError, match="is not a region node"):
        p.emission_plan(0, [(quantum_ids[0], "body")])


def test_pipe_padding_provenance(y_memory_program):
    _, art = y_memory_program
    p = art
    padding = p.pipe_padding()
    assert padding, "a compiled program with temporal pipes records padding"
    edge_padding = [
        seam.padding
        for edge in p.edges()
        if isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
        for seam in edge.edge.pipes
        if seam.padding is not None
    ]
    assert len(edge_padding) == len(padding)
    entry = padding[0]
    assert isinstance(entry, bloq.ir.PipePadding)
    # The recorded template ids are valid pool lookups a backend can resolve.
    assert isinstance(p.template(entry.one_round), bloq.ir.Template)
    assert isinstance(p.template(entry.looped), bloq.ir.Template)


# ---------------------------------------------------------------------------
# stim emission
# ---------------------------------------------------------------------------


def test_emit_isolated_t_attempts():
    graph = bloq.BlockGraph.from_text(
        "BLOG 1.0\n\n"
        "0: T [0, 0, 0]\n"
        "1: Port [0, 0, 1]\n"
        "0 -> +Z\n"
    )
    program = bloq.compile(graph, distance=3)
    path, body = next(
        (path, level)
        for path, level in program.levels()
        if len(path) == 1
        and sum(node.is_quantum() for _, node in level.nodes()) == 2
    )
    escape = next(
        node_id
        for node_id, node in body.nodes()
        if node.is_quantum()
        and not any(
            isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
            for edge in body.outgoing(node_id)
        )
    )
    program.insert_memory_rounds_after(escape, rounds=2, path=path)

    with pytest.raises(bloq.InvalidArgumentError, match="finite and in"):
        bloq.emit_isolated_t_attempts(program, noise=-1.0)

    artifacts = bloq.emit_isolated_t_attempts(program, noise=0.001)
    manifest = artifacts.manifest
    assert isinstance(artifacts, bloq.IsolatedTAttemptArtifacts)
    assert isinstance(manifest, bloq.IsolatedTAttemptManifest)
    assert manifest.rus_path == path
    assert artifacts.physical_s != artifacts.physical_t
    for physical in (artifacts.physical_s, artifacts.physical_t):
        assert sum(line.startswith("EXP_VAL ") for line in physical.splitlines()) == 3
        assert sum(line.startswith("DETECTOR") for line in physical.splitlines()) == len(
            manifest.detector_signs
        )
        assert "OBSERVABLE_INCLUDE" not in physical
    assert "EXP_VAL " not in artifacts.companion
    assert "EXP_VAL " not in artifacts.sheets
    assert manifest.frame_x_gap_observable == manifest.gap_z_observable
    assert manifest.frame_z_gap_observable == manifest.gap_x_observable
    assert manifest.frame_x_measurements and manifest.frame_z_measurements
    frame = program.output_frames()[0]
    assert manifest.frame_x_sign == program.resolve_classical(frame.x, pins=False).sign
    assert manifest.frame_z_sign == program.resolve_classical(frame.z, pins=False).sign
    assert manifest.postselection_detectors
    assert max(manifest.postselection_detectors) < len(manifest.detector_signs)
    assert len(manifest.frontier_observables) == len(manifest.frontier_bases)
    assert len(manifest.frontier_observables) == len(manifest.frontier_signs)
    assert manifest.frontier_observables
    assert set(manifest.exp_val_signs) <= {-1, 1}

    companion = stim.Circuit(artifacts.companion)
    sheets = stim.Circuit(artifacts.companion + artifacts.sheets)
    assert companion.num_detectors == len(manifest.detector_signs)
    assert sheets.num_detectors == companion.num_detectors
    # Frontier ids follow the full program's observables, including ids
    # absent from the companion. Stim counts slots through the largest id.
    assert sheets.num_observables == max(manifest.frontier_observables) + 1


def test_emit_stim(y_memory_program, tmp_path):
    graph, art = y_memory_program
    circuit = bloq.emit_stim(art)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert bloq.emit_stim(art) == circuit  # deterministic
    path = tmp_path / "memory.stim"
    circuit.to_file(path)
    assert stim.Circuit.from_file(path) == circuit

    padded = bloq.compile(graph, distance=3)
    padded.insert_memory_rounds(1, 2, rounds=3)
    assert "REPEAT" in str(bloq.emit_stim(padded))
    assert "REPEAT" not in str(bloq.emit_stim(padded, align_moments=True))


def test_checked_stim_emission_preserves_validation_error_type():
    program = bloq.Bloq.from_text("BLOQIR 1\ngraph {\n  n0 compute in0\n}\n")
    for emit in (bloq.emit_stim, bloq.emit_stim_segments):
        with pytest.raises(bloq.BloqValidationError, match="WF-11"):
            emit(program, validate=True)


# ---------------------------------------------------------------------------
# emission-plan lowering and the Stim dialects
# ---------------------------------------------------------------------------


def test_emit_plan_stim_lowers_a_node_against_a_caller_layout(y_memory_program):
    _, program = y_memory_program
    layout = {coord: index for index, coord in enumerate(program.sorted_layout_coords())}
    node = program.deterministic_emit_order()[0]
    plan = program.emission_plan(node)

    lowered = bloq.emit_plan_stim(plan, layout)
    assert isinstance(lowered, bloq.PlanStim)
    assert lowered.text
    # Every measurement the plan names lands in a column the text produces.
    assert set(lowered.measurement_columns) == set(plan.measurements)
    assert all(0 <= column < lowered.measurement_count
               for column in lowered.measurement_columns.values())
    assert lowered.measurement_count == program.node_measurement_count(node)


def test_emit_plan_stim_dialects_differ_only_in_the_non_clifford_gates():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    layout = {coord: index for index, coord in enumerate(program.sorted_layout_coords())}
    node = program.deterministic_emit_order()[0]
    plan = program.emission_plan(node)

    default = bloq.emit_plan_stim(plan, layout)
    stim_text = bloq.emit_plan_stim(plan, layout, dialect="STIM")
    clifft = bloq.emit_plan_stim(plan, layout, dialect="clifft")

    assert default.text == stim_text.text  # Stim is the default dialect
    assert clifft.measurement_columns == stim_text.measurement_columns
    assert bloq.stim_to_clifft_text(stim_text.text) == clifft.text

    with pytest.raises(bloq.InvalidArgumentError, match="unknown Stim dialect"):
        bloq.emit_plan_stim(plan, layout, dialect="bogus")


def test_emit_plan_stim_reports_a_missing_layout_coordinate(y_memory_program):
    _, program = y_memory_program
    node = program.deterministic_emit_order()[0]
    with pytest.raises(bloq.StimEmissionError):
        bloq.emit_plan_stim(program.emission_plan(node), {})


def test_emit_plan_stim_tags_the_gates_it_emits(y_memory_program):
    _, program = y_memory_program
    layout = {coord: index for index, coord in enumerate(program.sorted_layout_coords())}
    node = program.deterministic_emit_order()[0]
    tagged = bloq.emit_plan_stim(program.emission_plan(node), layout, tag="probe")
    assert "[probe]" in tagged.text


def test_dialect_codecs_round_trip_the_honest_t_tag():
    assert bloq.clifft_to_stim_text("T 0") == f"S[{bloq.HONEST_T_TAG}] 0"
    assert bloq.stim_to_clifft_text(f"S[{bloq.HONEST_T_TAG}] 0") == "T 0"
    assert bloq.stim_to_clifft_text("S 0") == "S 0"  # untagged S is a real S


def test_expanded_measurement_columns_follow_the_unrolled_circuit(y_memory_program):
    _, base = y_memory_program
    program = bloq.compile(bloq.GalleryItem("y_memory").load(), distance=3)
    edge = next(
        (e.source, e.target)
        for e in program.edges()
        if isinstance(e.edge, bloq.ir.BloqEdge.Quantum) and e.source == 1
    )
    padded = program.insert_memory_rounds(*edge, rounds=5)
    circuit = program.emission_plan(padded).circuit
    columns = circuit.expanded_measurement_columns()

    # Unrolling a REPEAT gives more columns than the circuit has measurements.
    assert columns.count > len(columns)
    assert len(columns) == len(columns.as_dict())
    assert dict(columns.as_dict()) == {m: columns.column(m) for m in columns.as_dict()}
    for measurement in columns.as_dict():
        assert measurement in columns
        assert columns[measurement] == columns.column(measurement)
    assert columns.column(10_000) is None
    assert 10_000 not in columns
    with pytest.raises(KeyError):
        columns[10_000]

    # A loop-free node reports one column per measurement.
    flat = base.emission_plan(0).circuit.expanded_measurement_columns()
    assert flat.count == len(flat)


# ---------------------------------------------------------------------------
# dunders: pickling, copying, equality, iteration
# ---------------------------------------------------------------------------


def test_bloq_pickles_through_its_binary_codec(y_memory_program):
    _, program = y_memory_program
    revived = pickle.loads(pickle.dumps(program))
    assert revived == program
    assert revived.to_binary() == program.to_binary()
    assert bloq.emit_stim(revived) == bloq.emit_stim(program)


def test_block_graph_pickles_through_its_text_codec():
    graph = bloq.GalleryItem.CNOT.load()
    revived = pickle.loads(pickle.dumps(graph))
    assert revived == graph
    assert revived.to_text() == graph.to_text()


def test_copies_are_independent(y_memory_program):
    graph, program = y_memory_program
    fresh = bloq.compile(graph, distance=3)
    deep = copy.deepcopy(fresh)
    assert copy.copy(fresh) == fresh and deep == fresh

    edge = next(
        (e.source, e.target)
        for e in fresh.edges()
        if isinstance(e.edge, bloq.ir.BloqEdge.Quantum) and e.source == 1
    )
    deep.insert_memory_rounds(*edge, rounds=2)
    assert deep != fresh, "mutating a deepcopy changed the original"
    assert copy.deepcopy(graph) == graph


def test_equality_is_structural():
    assert bloq.GalleryItem.CNOT.load() == bloq.GalleryItem.CNOT.load()
    assert bloq.GalleryItem.CNOT.load() != bloq.GalleryItem.BELL_STATE.load()
    assert bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3) == bloq.compile(
        bloq.GalleryItem.X_MEMORY.load(), distance=3
    )
    assert bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3) != bloq.compile(
        bloq.GalleryItem.X_MEMORY.load(), distance=5
    )
    graph = bloq.GalleryItem.X_MEMORY.load()
    for value in (graph, bloq.compile(graph, distance=3)):
        other = object()
        assert value.__eq__(other) is NotImplemented
        assert value != other
    stabilizers = bloq.GalleryItem.X_MEMORY.load().stabilizers()
    assert stabilizers[0] == bloq.GalleryItem.X_MEMORY.load().stabilizers()[0]
    # A row's kind is part of its identity, not just the operator it carries.
    measurement = next(
        row for row in bloq.GalleryItem.T_GATE.load().stabilizers() if row.is_measurement
    )
    assert measurement != stabilizers[0]


def test_block_graph_iterates_its_blocks():
    graph = bloq.GalleryItem.CNOT.load()
    assert [block.pos for block in graph] == [block.pos for block in graph.blocks()]
    assert len(list(graph)) == len(graph)


def test_reprs_are_concise_and_never_raise(y_memory_program):
    _, program = y_memory_program
    # The CNOT has open ports, so it is the one with terminal frames.
    open_program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    subjects = [
        program,
        program.level_at([]),
        program.node(0),
        program.nodes()[0][1].kind,
        program.nodes()[0][1].provenance,
        program.edges()[0],
        program.walk()[0],
        program.template(0),
        program.template(0).circuit,
        program.template(0).circuit.ops()[0],
        program.emission_plan(0),
        open_program.output_frames()[0],
        open_program.logical_outputs()[0],
        bloq.GalleryItem.CNOT.load(),
        bloq.CompileContext(distance=3),
        bloq.emit_stim_segments(program),
        bloq.emit_stim_segments(program).segments[0],
    ]
    for subject in subjects:
        text = repr(subject)
        assert "object at 0x" not in text, f"{type(subject).__name__} has no __repr__"
        assert "true" not in text and "false" not in text, f"Rust bool in {text}"
        # Rust `Debug` output would leak binding-internal type names and glam
        # vectors into a public repr.
        assert "IVec" not in text and " { " not in text, f"Rust Debug in {text}"
        assert len(text) < 200, f"repr too long: {text}"


def test_every_exposed_class_has_a_custom_repr():
    """Exhaustive, so a new binding cannot ship pyo3's default object repr.

    The instance-level checks above can only cover the types this test file
    knows how to build; this one covers the whole export list.
    """
    missing = [
        name
        for name in bloq.__all__
        if isinstance(value := getattr(bloq, name), type)
        # Exceptions inherit a perfectly good repr from `BaseException`.
        and not issubclass(value, BaseException)
        and value.__repr__ is object.__repr__
    ]
    assert missing == []


@pytest.mark.parametrize(
    "text,skip,measurements,source,destination",
    [
        ("M 0\nM 0", set(), {}, 0, 2**63 - 1),
        ("M 0\nM 0", set(), None, 2**63 - 1, 0),
        ("M 0", {"M"}, None, 2**63 - 1, 0),
        ("DETECTOR rec[-1]", set(), {0: 2**63 - 1}, 1, -(2**63)),
        ("DETECTOR rec[-1]", set(), {}, -(2**63), 0),
    ],
)
def test_stim_remapping_rejects_measurement_offset_overflow(
    text, skip, measurements, source, destination
):
    with pytest.raises(bloq.InvalidArgumentError, match="overflow"):
        bloq.remap_stim_circuit(
            text,
            {0: 0},
            skip=skip,
            measurement_map=measurements,
            source_measurement_start=source,
            destination_measurement_start=destination,
        )


def test_remap_stim_observable_index_does_not_saturate_to_u64_max():
    with pytest.raises(bloq.InvalidArgumentError, match="fitting u64"):
        bloq.remap_stim_circuit(
            "OBSERVABLE_INCLUDE(18446744073709551616)",
            {},
            observable_map={2**64 - 1: [2]},
        )
    largest_float_below_limit = 2**64 - 2048
    remapped, _ = bloq.remap_stim_circuit(
        f"OBSERVABLE_INCLUDE({largest_float_below_limit})",
        {},
        observable_map={largest_float_below_limit: [3]},
    )
    assert remapped == "OBSERVABLE_INCLUDE(3)"


def test_remap_stim_circuit_skips_nothing_by_default():
    text = "QUBIT_COORDS(0, 0) 0\nM 0"
    remapped, _ = bloq.remap_stim_circuit(text, {0: 1})
    assert remapped == "QUBIT_COORDS(0,0) 1\nM 1"
    dropped, _ = bloq.remap_stim_circuit(text, {0: 1}, skip={"QUBIT_COORDS"})
    assert dropped == "M 1"


# ---------------------------------------------------------------------------
# Block-graph read and mutation accessors
# ---------------------------------------------------------------------------


def test_block_lookup_accessors_agree_with_the_iteration_accessors():
    graph = bloq.GalleryItem.THREE_CNOTS.load()

    for block in graph.blocks():
        assert graph.has_block_at(block.pos)
        assert graph.get_block(block.pos) == block
        # Ids are assigned in the same order `.blog` files number them.
        assert graph.get_block_id(block.pos) is not None
        # `degree` counts pipes, so it must agree with an independent count
        # over the pipe list.
        incident = sum(block.pos in (pipe.src, pipe.dst) for pipe in graph.pipes())
        assert graph.degree(block.pos) == incident

    absent = (99, 99, 99)
    assert not graph.has_block_at(absent)
    assert graph.get_block_id(absent) is None
    assert graph.degree(absent) == 0


def test_pipe_lookup_and_mutation_accessors_round_trip():
    graph = bloq.GalleryItem.THREE_CNOTS.load()
    pipe = graph.pipes()[0]
    u, v = pipe.src, pipe.dst

    assert graph.get_pipe(u, v) == pipe
    assert graph.get_pipe(u, (99, 99, 99)) is None

    graph.set_pipe_tag(u, v, "marked")
    graph.set_pipe_hadamard(u, v, not pipe.hadamard)
    changed = graph.get_pipe(u, v)
    assert changed.tag == "marked"
    assert changed.hadamard is not pipe.hadamard

    assert graph.remove_pipe(u, v) == changed
    assert graph.get_pipe(u, v) is None
    assert graph.remove_pipe(u, v) is None
    assert graph.pipe_count == len(graph.pipes())


def test_block_mutation_accessors_retarget_the_stored_block():
    graph = bloq.BlockGraph()
    graph.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    assert graph.can_place_block(bloq.Block((0, 0, 1), "ZXZ"))
    assert not graph.can_place_block(bloq.Block((0, 0, 0), "ZXZ"))

    graph.set_block_kind((0, 0, 0), "PORT")
    graph.set_block_tag((0, 0, 0), "named")
    stored = graph.get_block((0, 0, 0))
    assert stored.kind.is_port
    assert stored.tag == "named"

    with pytest.raises(bloq.BlockGraphError):
        graph.set_block_kind((9, 9, 9), "PORT")
    with pytest.raises(bloq.BlockGraphError):
        graph.set_block_tag((9, 9, 9), "named")


def test_graph_counters_and_predicates_agree_with_the_block_list():
    def counted(graph, predicate):
        return sum(predicate(block.kind) for block in graph.blocks())

    for item in (
        bloq.GalleryItem.T_GATE,
        bloq.GalleryItem.Y_MEMORY,
        bloq.GalleryItem.GHZ_SLIDE_THEN_GLIDE,
        bloq.GalleryItem.GHZ_PATCH_ROTATIONS,
        bloq.GalleryItem.T_COMPARISON,
    ):
        graph = item.load()
        assert graph.t_count == counted(graph, lambda k: k.is_t)
        assert graph.y_count == counted(graph, lambda k: k.is_y)
        assert graph.walking_count == counted(graph, lambda k: k.is_walking)
        assert graph.patch_rotation_count == counted(
            graph, lambda k: k.is_patch_rotation
        )
        assert graph.selective_count == counted(graph, lambda k: k.is_selective)
        assert graph.is_clifford == (graph.t_count == 0)
        assert graph.is_rigid == (graph.selective_count == 0)
        # `validate_structure` is the structural half of `validate`, so every
        # gallery entry must pass it.
        graph.validate_structure()


def test_clear_actions_leaves_the_structure_intact():
    graph = bloq.GalleryItem.T_GATE.load()
    assert graph.has_actions()
    blocks, pipes = graph.blocks(), graph.pipes()

    graph.clear_actions()

    assert not graph.has_actions()
    assert graph.actions() == []
    assert (graph.blocks(), graph.pipes()) == (blocks, pipes)


def test_moving_block_kinds_expose_their_movement():
    walking = bloq.BlockKind.walking("ZXZ", (1, 0))
    rotation = bloq.BlockKind.patch_rotation("X", (0, 1))

    assert (walking.is_walking, walking.movement) == (True, (1, 0))
    assert (rotation.is_patch_rotation, rotation.movement) == (True, (0, 1))
    # The repr carries the movement, since the BLOG spelling alone does not.
    assert repr(walking) == "<BlockKind Walking movement=(1, 0)>"
    assert repr(rotation) == "<BlockKind PatchRotation movement=(0, 1)>"

    # `is_dynamic` is about runtime resolution, not motion.
    assert not walking.is_dynamic and not rotation.is_dynamic
    assert bloq.BlockKind.t().is_dynamic
    assert bloq.BlockKind.selective("XY").is_dynamic
    assert bloq.BlockKind.cube("ZXZ").movement is None


def test_direction_is_spatial_splits_the_time_axis_off():
    spatial = (
        bloq.Direction.X_PLUS,
        bloq.Direction.X_MINUS,
        bloq.Direction.Y_PLUS,
        bloq.Direction.Y_MINUS,
    )
    assert all(d.is_spatial() for d in spatial)
    assert not bloq.Direction.Z_PLUS.is_spatial()
    assert not bloq.Direction.Z_MINUS.is_spatial()


def test_stabilizer_interiors_and_port_support_are_consistent():
    graph = bloq.GalleryItem.CNOT.load()
    generator = graph.stabilizers()[0]
    stabilizer = generator.stabilizer

    # Every interior edge joins two interior nodes.
    for (u, v) in stabilizer.interior_edges:
        assert u in stabilizer.interior_nodes
        assert v in stabilizer.interior_nodes
    # The port support is the part of the interior sitting on port blocks.
    ports = {block.pos for block in graph.blocks() if block.kind.is_port}
    assert set(stabilizer.port_stabilizer) <= ports
    assert set(stabilizer.port_stabilizer) <= set(stabilizer.interior_nodes)

    # Support maps iterate in ascending key order, so printed rows are reproducible.
    for row in graph.stabilizers():
        for support in (row.stabilizer.port_stabilizer, row.stabilizer.interior_nodes,
                        row.stabilizer.interior_edges):
            assert list(support) == sorted(support)


def test_negative_bulk_adder_is_not_in_public_gallery():
    assert "ONE_BIT_ADDER" not in bloq.GalleryItem.__members__
    assert "one_bit_adder" not in {entry.id for entry in bloq.GalleryItem}
    with pytest.raises(ValueError):
        bloq.GalleryItem("one_bit_adder")
    with pytest.raises(bloq.InvalidArgumentError):
        bloq._core.gallery_load("one_bit_adder")
