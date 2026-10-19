"""Tests for structured action authoring (Action / Expr / MeasureTarget /
FeedbackTarget) and their round-trip equivalence with the BLOG-text path.
"""

from collections.abc import Hashable
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import pytest

import bloq


def test_branch_authoring_matches_ccz_source_in_every_choice():
    example = Path(__file__).parents[1] / "examples/action_ccz_teleport.py"
    graph = runpy.run_path(str(example))["build_graph"]()
    expected = bloq.GalleryItem.CCZ_GATE_TELEPORT.load().flatten()
    targets = [action.branch_target for action in graph.actions() if action.kind == "branch"]
    assert len(targets) == 3
    for choices in range(8):
        assignments = [(target, bool(choices & (1 << index))) for index, target in enumerate(targets)]
        assert graph.project_branches(assignments).to_text() == expected.project_branches(assignments).to_text()
    assert bloq.BlockGraph.from_text(graph.to_text()).to_text() == graph.to_text()
    bloq.compile(graph, distance=3).validate()


def test_branch_objects_and_failed_edit_are_independent():
    block = bloq.Block((0, 0, 1), "Z", tag="cap")
    arm = bloq.BranchArm([block, ((2, 0, 1), bloq.BlockKind.cube("ZXZ"))],
                         pipes=[bloq.Pipe((0, 0, 0), "+Z")])
    branch = bloq.Branch("b", "missing", on_false=arm, on_true=arm)
    assert branch.name == "b" and branch.condition == bloq.Expr.var("missing")
    assert branch.on_false.blocks()[0].tag == "cap"
    assert branch.on_true.pipes()[0].src == (0, 0, 0)
    graph = bloq.BlockGraph()
    graph.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    before = graph.to_text()
    with pytest.raises(bloq.BlockGraphError):
        graph.add_branches([branch])
    assert graph.to_text() == before
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.BranchArm([((0, 0, 1), "invalid")])


def test_expr_var_and_introspection():
    e = bloq.Expr.var("m1")
    assert e.kind == "var" and e.name == "m1" and e.op is None
    assert e.operands() == []
    assert str(e) == "m1"


def test_expr_operators_map_to_blog_syntax():
    a, b, c = bloq.Expr.var("a"), bloq.Expr.var("b"), bloq.Expr.var("c")
    assert str(a ^ b) == "a ^ b"
    assert str(a & b) == "a & b"
    assert str(a | b) == "a | b"
    assert str(~a) == "!a"
    # Precedence-aware rendering: `|` binds looser than `&`.
    assert str((a | b) & c) == "(a | b) & c"
    assert str(~(a ^ b)) == "!(a ^ b)"


def test_expr_str_promotion():
    a = bloq.Expr.var("a")
    assert (a ^ "b") == (a ^ bloq.Expr.var("b"))
    assert ("b" ^ a) == (bloq.Expr.var("b") ^ a)
    assert ("x" & a).operands()[0] == bloq.Expr.var("x")


def test_symbolic_expr_rejects_python_truth_testing():
    with pytest.raises(TypeError, match="symbolic Expr has no truth value"):
        bool(bloq.Expr.var("measurement"))


def test_expr_eq_and_unhashable():
    a1 = bloq.Expr.var("a") ^ "b"
    a2 = bloq.Expr.var("a") ^ "b"
    assert a1 == a2
    assert a1 != (bloq.Expr.var("a") & "b")
    assert not isinstance(a1, Hashable)
    with pytest.raises(TypeError, match="unhashable"):
        hash(a1)


def test_expr_nested_introspection():
    e = (bloq.Expr.var("a") ^ "b") & ~bloq.Expr.var("c")
    assert e.kind == "binary" and e.op == "and"
    lhs, rhs = e.operands()
    assert lhs.op == "xor"
    assert rhs.kind == "unary" and rhs.op == "not"
    assert rhs.operands()[0].name == "c"


def test_measure_target_node_and_edge():
    n = bloq.MeasureTarget.node((0, 0, 1))
    assert not n.is_edge and n.pos == (0, 0, 1) and n.direction is None

    e = bloq.MeasureTarget.edge((0, 0, 1), "+X")
    assert e.is_edge and e.pos == (0, 0, 1)
    assert e.direction == bloq.Direction.X_PLUS
    assert e == bloq.MeasureTarget.edge((0, 0, 1), bloq.Direction.X_PLUS)
    assert hash(e) == hash(bloq.MeasureTarget.edge((0, 0, 1), "+X"))

    with pytest.raises(bloq.InvalidArgumentError):
        bloq.MeasureTarget.edge((0, 0, 1), "sideways")


def test_feedback_target():
    t = bloq.FeedbackTarget("Z", (1, 0, 2))
    assert t.pauli == bloq.PauliBasis.Z and t.pos == (1, 0, 2)
    assert t == bloq.FeedbackTarget(bloq.PauliBasis.Z, (1, 0, 2))
    with pytest.raises(bloq.InvalidArgumentError):
        bloq.FeedbackTarget("Q", (0, 0, 0))


def test_action_builders_display():
    a = bloq.Expr.var("a")
    assert str(bloq.Action.let("r", a ^ "b")) == "r = a ^ b"
    assert (
        str(bloq.Action.measure(bloq.MeasureTarget.edge((0, 0, 1), "+X"), "mzz"))
        == "mzz = measure [0, 0, 1] -> +X"
    )
    assert str(bloq.Action.discard_if("m")) == "discard if m"
    assert str(bloq.Action.resolve((1, 0, 2), "mzz")) == "resolve [1, 0, 2] if mzz"
    assert str(bloq.Action.branch((2, 0, 3), a & "b")) == "resolve [2, 0, 3] if a & b"
    assert str(bloq.Action.feedback([("Z", (0, 0, 2))])) == "feedback Z [0, 0, 2]"
    assert (
        str(bloq.Action.feedback([("X", (0, 0, 2)), ("Z", (1, 0, 2))], condition="m"))
        == "feedback X [0, 0, 2], Z [1, 0, 2] if m"
    )


def test_action_introspection():
    m = bloq.Action.measure((0, 0, 1), "m")
    assert m.kind == "measure" and m.name == "m" and not m.target.is_edge
    assert m.expr is None and m.condition is None

    let = bloq.Action.let("r", "a")
    assert let.kind == "let" and let.name == "r" and let.expr.name == "a"

    r = bloq.Action.resolve((1, 0, 2), "m")
    assert r.kind == "resolve" and r.resolve_target == (1, 0, 2)
    assert r.condition.name == "m"

    b = bloq.Action.branch((2, 0, 3), "m")
    assert b.kind == "branch" and b.branch_target == (2, 0, 3)
    assert b.condition.name == "m"

    f = bloq.Action.feedback([("Z", (0, 0, 2))], condition="m")
    assert f.kind == "feedback"
    assert f.feedback_targets == [bloq.FeedbackTarget("Z", (0, 0, 2))]
    assert f.condition.name == "m"

    d = bloq.Action.discard_if("m")
    assert d.kind == "discard_if" and d.condition.name == "m"

    assert not isinstance(m, Hashable)
    with pytest.raises(TypeError, match="unhashable"):
        hash(m)


def test_action_with_shift():
    m = bloq.Action.measure(bloq.MeasureTarget.edge((0, 0, 1), "+X"), "m")
    shifted = m.with_shift((1, 2, 3))
    assert shifted.target.pos == (1, 2, 4)
    assert shifted.target.direction == bloq.Direction.X_PLUS
    # Position-free actions are unchanged.
    let = bloq.Action.let("r", "a")
    assert let.with_shift((5, 5, 5)) == let
    assert bloq.Action.branch((2, 0, 3), "m").with_shift((1, 2, 3)).branch_target == (
        3,
        2,
        6,
    )
    with pytest.raises(bloq.InvalidArgumentError, match="coordinate range"):
        bloq.Action.resolve((2**31 - 1, 0, 0), "m").with_shift((1, 0, 0))
    with pytest.raises(bloq.InvalidArgumentError, match="coordinate range"):
        bloq.Action.measure(
            bloq.MeasureTarget.edge((2**31 - 2, 0, 0), "+X"), "m"
        ).with_shift((1, 0, 0))


def structured_t_actions():
    """Structured equivalents of the gallery `t_gate` action block. The gallery
    build differs from the raw `.blog` asset: its resolve condition is `!mzz`.
    """
    return [
        bloq.Action.measure(bloq.MeasureTarget.edge((0, 0, 1), "+X"), "mzz"),
        bloq.Action.resolve((1, 0, 2), ~bloq.Expr.var("mzz")),
    ]


def test_structured_equals_parsed_gallery_actions():
    g = bloq.GalleryItem("t_gate").load()
    assert g.actions() == structured_t_actions()


def test_set_actions_round_trips_to_text():
    g = bloq.GalleryItem("t_gate").load()
    original = g.to_text()
    g.set_actions(structured_t_actions())
    assert g.to_text() == original


def test_set_actions_matches_text_path():
    item = bloq.GalleryItem("t_gate")
    structured = item.load()
    structured.set_actions([])
    structured.set_actions(
        [
            bloq.Action.measure(bloq.MeasureTarget.edge((0, 0, 1), "+X"), "mzz"),
            bloq.Action.resolve((1, 0, 2), "mzz"),
        ]
    )

    text = item.load()
    text.set_actions([])
    text.add_actions_from_text("mzz = measure [0, 0, 1] -> +X\nresolve [1, 0, 2] if mzz")

    assert structured.actions() == text.actions()
    assert structured.to_text() == text.to_text()
    assert structured.has_actions()


def test_add_action_appends_and_validates():
    # A Z cap on a Z-initialized patch closes the record before any output.
    g = bloq.BlockGraph.from_text(
        "BLOG 1.0\n0: ZXZ [0,0,0]\n1: Z [0,0,1]\n[0,0,0] -> +Z\n"
    )
    g.add_action(bloq.Action.measure((0, 0, 1), "m"))
    assert [a.kind for a in g.actions()] == ["measure"]
    g.add_action(bloq.Action.discard_if("m"))
    assert [a.kind for a in g.actions()] == ["measure", "discard_if"]


def test_actions_validate_against_graph():
    g = bloq.GalleryItem("bell_state").load()
    # Measuring a block that does not exist must fail DAG validation.
    with pytest.raises(bloq.BlockGraphError):
        g.add_action(bloq.Action.measure((9, 9, 9), "m"))
    # An undefined variable reference must fail too.
    with pytest.raises(bloq.BlockGraphError):
        g.add_action(bloq.Action.discard_if("undefined_var"))


def test_structured_actions_compile_end_to_end():
    g = bloq.GalleryItem("s_gate").load().flatten()
    parsed = g.actions()
    assert [a.kind for a in parsed] == ["feedback"]

    # Rebuild the same action structurally and compile the filled graph.
    g.set_actions([bloq.Action.feedback([("Z", (0, 0, 2))])])
    assert g.actions() == parsed

    filled, _stabilizers = g.fill_ports_auto()[0]
    program = bloq.compile(filled, distance=3)
    program.validate()
    assert bloq.emit_stim(program)


def test_analyzed_action_dag_svg_and_snapshot():
    flat = bloq.GalleryItem.T_GATE.load().flatten()
    original = flat.to_text()
    dag = flat.analyze_action_graph()
    assert isinstance(dag, bloq.ActionDag)
    assert dag.is_analyzed
    assert dag.actions() == flat.actions()
    assert dag.inputs() == []
    assert dag.dependencies() == [(0, 1, "Classical")]
    assert flat.to_text() == original

    svg = dag.to_svg()
    root = ET.fromstring(svg)
    assert len(root.findall(".//*[@data-node]")) == len(dag) == 2
    assert [p.attrib["data-dependency"] for p in root.findall(".//*[@data-dependency]")] == [
        "Classical"
    ]
    assert svg == dag.to_svg()
    flat.add_action(bloq.Action.discard_if("mzz"))
    assert len(flat.action_graph()) == 3
    assert len(dag) == 2  # The returned snapshot stays independent of later edits.


def test_hierarchical_action_analysis_preserves_source_and_honors_limits():
    source = bloq.GalleryItem.THREE_BIT_ADDER.load()
    original = source.to_text()
    dag = source.analyze_action_graph()
    assert dag.is_analyzed and len(dag) == 51
    assert len(dag.owners()) == len(dag)
    assert all(owner is not None for owner in dag.owners())
    assert any(path for _, path in dag.owners())
    assert source.to_text() == original
    with pytest.raises(bloq.BlockGraphError):
        source.analyze_action_graph(limits={"max_expanded_instances": 0})
    with pytest.raises(bloq.BlockGraphError):
        source.analyze_action_graph(limits={"max_local_columns": 0})
    with pytest.raises(bloq.InvalidArgumentError):
        source.analyze_action_graph(limits={"unknown_limit": 1})


def test_three_bit_adder_action_svg_contains_every_node_and_edge(tmp_path, monkeypatch):
    example = Path(__file__).parents[1] / "examples/action_dag.py"
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(example))
    dag = result["dag"]
    assert dag.is_analyzed and len(dag) == 51
    root = ET.parse(tmp_path / "three-bit-adder-action-dag.svg").getroot()
    assert len(root.findall(".//*[@data-node]")) == len(dag)
    assert len(root.findall(".//*[@data-definition]")) == len(dag)
    assert len(root.findall(".//*[@data-dependency]")) == len(dag.dependencies())
    assert {reason for _, _, reason in dag.dependencies()} >= {
        "Classical", "SelectiveSupport", "ReadoutParity", "FeedbackAnticommutation"
    }
    assert all(0 <= a < len(dag) and 0 <= b < len(dag) for a, b, _ in dag.dependencies())
