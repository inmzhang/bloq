"""IR binding tests: recorded compile provenance and structural parity terms."""

import collections
import copy

import bloq
import pytest


@pytest.fixture(scope="module")
def y_memory():
    g = bloq.GalleryItem("y_memory").load()
    return g, bloq.compile(g, distance=3)


def test_compiled_instance_provenance_is_public(y_memory):
    from bloq.ir import InstanceProvenance

    _, program = y_memory
    provenance = program.quantum_nodes()[0][1].instances[0].provenance
    assert bloq.ir.InstanceProvenance is InstanceProvenance
    assert isinstance(provenance, bloq.ir.InstanceProvenance.Block)


def test_compile_records_metadata_and_preserves_it_through_codecs(y_memory):
    _, program = y_memory
    for distance, compiled in [
        (3, program),
        (5, bloq.compile(bloq.GalleryItem.CNOT.load(), distance=5)),
    ]:
        assert compiled.metadata[bloq.ir.CODE_DISTANCE_METADATA_KEY] == distance
        assert compiled.metadata[bloq.ir.CONVENTION_METADATA_KEY] == "fixed-bulk"
        for restored in [
            bloq.Bloq.from_text(compiled.to_text()),
            bloq.Bloq.from_binary(compiled.to_binary()),
        ]:
            assert restored.metadata == compiled.metadata


def test_hand_built_program_has_no_metadata():
    assert bloq.Bloq.from_text("BLOQIR 1\ngraph {\n}\n").metadata == {}


def test_observable_composition_and_output_ports_round_trip():
    program = bloq.Bloq.from_text("""BLOQIR 1
template t0 {
  circuit {
    R (0,0)
    M (0,0):m0
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable 7 measurements i0:m0*i0:m0 operators i0 input X(0,0), i0 output Z(0,0)
  n2 observable 8
  n3 compute in0^in1
  n0 -> n1 order
  n1 -> n2 compose 0
  n1 -> n3 value 0 flip
  n2 -> n3 value 1
  result n1:flip
}
""")
    for restored in [program, bloq.Bloq.from_text(program.to_text()),
                     bloq.Bloq.from_binary(program.to_binary())]:
        restored.validate()
        observable = restored.nodes()[1][1].classical
        assert isinstance(observable, bloq.ir.ClassicalNode.Observable)
        assert observable.index == 7
        assert len(observable.measurements) == 2
        assert len(observable.operators) == 2
        assert any(edge.is_compose() for edge in restored.edges())
        assert restored.data_inputs(2)[0].output is None
        outputs = {0: bloq.ir.ObservableOutput.Flip, 1: bloq.ir.ObservableOutput.Corrected}
        assert {value.slot: value.output for value in restored.value_inputs(3)} == outputs
        assert {edge.edge.slot: edge.edge.output for edge in restored.edges() if edge.is_value()} == outputs
        assert restored.value_output == bloq.ir.ValueRef(1, bloq.ir.ObservableOutput.Flip)
        assert restored.resolve_classical(1).decoder_observables == [7]
        assert restored.resolve_classical(1, output=bloq.ir.ObservableOutput.Flip).decoder_observables == [7]
        assert restored.resolve_classical(2).decoder_observables == [8]


def test_classical_expression_operand_lists_and_selection_are_public():
    program = bloq.Bloq.from_text(
        "BLOQIR 1\ngraph {\n"
        "n0 compute select(in0, xor(in1, in2, in3), and())\n}\n"
    )
    expr = program.nodes()[0][1].classical.expr
    condition, when_false, when_true = expr.operands()
    assert expr.kind == "select" and not expr.is_linear()
    assert condition.slot == 0
    assert when_false.kind == "xor"
    assert [operand.slot for operand in when_false.operands()] == [1, 2, 3]
    assert when_true.kind == "and" and when_true.operands() == []
    assert str(expr) == "select(in(0), xor(in(1), in(2), in(3)), and())"
    for symbolic in (expr, condition, when_true):
        with pytest.raises(TypeError, match="symbolic ClassicalExpr has no truth value"):
            bool(symbolic)
    parity_program = bloq.Bloq.from_text(
        "BLOQIR 1\ngraph {\nn0 compute parity(1, in2, in0, in2)\n}\n"
    )
    parity = parity_program.nodes()[0][1].classical.expr
    assert parity.kind == "parity" and parity.is_linear()
    assert parity.input_slots == [2, 0, 2] and parity.value is True
    assert parity.operands() == []
    assert str(parity) == "parity(true, in(2), in(0), in(2))"
    restored = bloq.Bloq.from_binary(parity_program.to_binary())
    assert restored.nodes()[0][1].classical.expr == parity


def test_clifford_proxy_has_output_frames():
    proxy = bloq.compile_clifford_proxy(bloq.GalleryItem.X_MEMORY.load(), [])
    assert len(proxy.output_frames()) == len(proxy.logical_outputs())


def test_logical_output_owner_survives_exchange_roundtrips():
    program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    outputs = program.logical_outputs()
    instances = {
        instance.id
        for _, node in program.nodes()
        if node.is_quantum()
        for instance in node.quantum.instances
    }
    assert outputs and all(output.instance in instances for output in outputs)
    assert bloq.Bloq.from_text(program.to_text()).logical_outputs() == outputs
    assert bloq.Bloq.from_binary(program.to_binary()).logical_outputs() == outputs


def test_noisy_plan_exposes_expanded_source_annotations():
    program = bloq.Bloq.from_text("""BLOQIR 1
template t0 {
  circuit {
    R (0,0)
    REPEAT 3 b1
  }
  body b1 {
    M (1,0):m0
    TICK
  }
  detector body(b1) m0
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
}
""")
    program.validate()
    assert program.emission_plan(0).normalized_template(0) is None
    plan = program.emission_plan(0, noise=0.01)
    normalized = plan.normalized_template(0)
    assert normalized is not None
    assert len(normalized.detectors) == 3
    assert len(program.template(0).detectors) == 1
    assert normalized.circuit.body_count == 1
    assert len(plan.measurements) == 3


def test_insert_memory_rounds_uses_recorded_padding_provenance(y_memory):
    g, _ = y_memory
    p = bloq.compile(g, distance=3)
    # No source graph, no distance: the recorded edge padding provenance
    # alone drives the splice — including on a program reloaded from binary.
    reloaded = bloq.Bloq.from_binary(p.to_binary())
    for program in (p, reloaded):
        assert any(
            (edge.source, edge.target) == (1, 2)
            and isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
            for edge in program.edges()
        )
        before = program.node_count
        padding = program.insert_memory_rounds(1, 2, rounds=2)
        node = program.node(padding)
        assert program.node_count > before
        assert node.is_quantum()
        assert node.memory_rounds() == 2
        assert isinstance(node.provenance, bloq.ir.NodeProvenance.MemoryPadding)
        program.validate()
    # Live ids survive the codec; the free-slot reuse order is not serialized.
    # Padding must still produce exactly the same physical instructions.
    assert bloq.emit_stim(p) == bloq.emit_stim(reloaded)


def test_insert_memory_rounds_without_quantum_edge_raises():
    # No Quantum edge between the named nodes (an empty graph has neither node).
    # The seam-with-a-real-edge-but-no-recorded-provenance path is covered by
    # test_pipeline's bad-endpoint case (a Y-block seam the compiler skips).
    p = bloq.Bloq.from_text("BLOQIR 1\ngraph {\n}\n")
    with pytest.raises(bloq.BloqError, match="not a single non-empty seam"):
        p.insert_memory_rounds(0, 1, rounds=2)


def test_insert_memory_rounds_after_rus_escape():
    compiled = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    p = bloq.Bloq.from_binary(compiled.to_binary())
    path, body = next(
        (path, level)
        for path, level in p.levels()
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

    padding = p.insert_memory_rounds_after(escape, rounds=2, path=path)
    p.validate()

    body = next(level for candidate, level in p.levels() if candidate == path)
    assert body.node(padding).memory_rounds() == 2
    assert body.has_path(escape, padding)


def _looped_two_cube_program():
    # Two ZXZ cubes in time; the decoder wait splices the looped padding
    # template, giving the program a REPEAT body with loop-carried state.
    g = bloq.BlockGraph.from_text(
        "BLOG 1.0\n\n0: ZXZ [0,0,0]\n1: ZXZ [0,0,1]\n0 -> +Z\n"
    )
    p = bloq.compile(g, distance=3)
    seam = next(e for e in p.edges() if isinstance(e.edge, bloq.ir.BloqEdge.Quantum))
    p.insert_memory_rounds(seam.source, seam.target, rounds=5)
    return p


def test_flatten_unrolls_repeats_preserving_detectors_and_observables():
    p = _looped_two_cube_program()
    looped = bloq.emit_stim(p)
    assert "REPEAT" in str(looped)

    p.flatten()
    p.validate()

    flat = bloq.emit_stim(p)
    assert "REPEAT" not in str(flat)
    assert flat.num_detectors == looped.num_detectors
    assert flat.num_observables == looped.num_observables


# Node-level loop-state terms are invalid without a recurrence table, but the
# codec remains structural and can expose one for parity inspection.
LOOP_TEXT = """BLOQIR 1

template t0 {
  circuit {
    M (0,0):m0
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
    detector i0:m0*s0
  }
}
"""


def test_parity_terms_structural():
    p = bloq.Bloq.from_text(LOOP_TEXT)
    ((_, qn),) = p.quantum_nodes()
    parity = qn.detectors[0].parity

    measurement, loop = parity.terms()
    assert isinstance(measurement, bloq.ir.DetectorTerm.Measurement)
    assert str(measurement.measurement) == "i0:m0"
    assert isinstance(loop, bloq.ir.DetectorTerm.LoopState)
    assert loop.state == 0

    assert parity.loop_states() == [0]
    assert [str(m) for m in parity.measurements()] == ["i0:m0"]


def test_parity_terms_measurement_only(y_memory):
    _, art = y_memory
    quantum_nodes = [qn for _, qn in art.quantum_nodes()]
    parities = [
        det.parity for qn in quantum_nodes for det in qn.detectors
    ]
    for parity in parities:
        assert parity.loop_states() == []
        assert len(parity.terms()) == len(parity.measurements())

    instances = {
        instance.id: instance
        for qn in quantum_nodes
        for instance in qn.instances
    }
    bundle_rows = 0
    for qn in quantum_nodes:
        for use in qn.detector_bundles:
            bundle = art.detector_bundle(use.bundle)
            assert len(use.instances) == len(bundle.owner_templates)
            for instance_id, template_id in zip(use.instances, bundle.owner_templates):
                assert instances[instance_id].template_id == template_id
            for detector in bundle.detectors:
                bundle_rows += 1
                for term in detector.terms:
                    assert isinstance(term, bloq.ir.BundleDetectorTerm.Measurement)
                    instance = instances[use.instances[term.owner]]
                    assert (
                        term.measurement
                        < art.template(instance.template_id).circuit.num_measurements
                    )
    assert bundle_rows, "compiled boundary detectors must exercise shared bundle rows"


@pytest.fixture(scope="module")
def t_gate():
    """A gallery program with region bodies (nested graph levels)."""
    g = bloq.GalleryItem("t_gate").load()
    return bloq.compile(g, distance=3)


def test_walk_top_level_matches_flat_reads(y_memory):
    _, art = y_memory
    p = art
    walk = p.walk()
    # A flat (region-free) program visits exactly its top-level nodes.
    assert len(walk) >= p.node_count
    top = [v for v in walk if not v.path]
    assert [v.id for v in top] == p.node_ids()


def test_walk_descends_region_bodies(t_gate):
    walk = t_gate.walk()
    # Regions mean more visited nodes than the top level alone.
    assert len(walk) > t_gate.node_count
    nested = [v for v in walk if v.path]
    assert nested, "expected at least one node inside a region body"
    # Every path hop names a real body selector and an owning region node.
    for visit in nested:
        for owner, selector in visit.path:
            assert selector == "body"
            assert isinstance(owner, int)
        # top_level_ancestor resolves to a real top-level node id.
        assert t_gate.node(visit.top_level_ancestor()) is not None


def test_walk_top_level_ancestor_is_self_at_top(t_gate):
    for visit in t_gate.walk():
        if not visit.path:
            assert visit.top_level_ancestor() == visit.id


def test_levels_first_is_the_top_level(y_memory):
    _, art = y_memory
    p = art
    path, level = p.levels()[0]
    assert path == []
    assert level.node_count == p.node_count
    assert level.node_ids() == p.node_ids()


def test_levels_cover_every_walked_node(t_gate):
    levels = t_gate.levels()
    assert len(levels) > 1  # top level plus region bodies
    walked = sum(level.node_count for _, level in levels)
    assert walked == len(t_gate.walk())


def test_level_at_is_the_targeted_read_levels_enumerates(t_gate):
    """One path in, one level out — no scan over every level to find it."""
    for path, level in t_gate.levels():
        found = t_gate.level_at(path)
        assert found.node_ids() == level.node_ids()
        assert found.node_count == level.node_count


def test_level_at_rejects_a_path_that_names_nothing(t_gate):
    rus = t_gate.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
    quantum = next(nid for nid, node in t_gate.nodes() if node.is_quantum())
    with pytest.raises(bloq.InvalidArgumentError):
        t_gate.level_at([(999, "body")])
    with pytest.raises(bloq.InvalidArgumentError):
        t_gate.level_at([(quantum, "body")])
    with pytest.raises(bloq.InvalidArgumentError):
        t_gate.level_at([(rus.node, "on_true")])


# ---------------------------------------------------------------------------
# node/edge payload accessors
# ---------------------------------------------------------------------------


def test_node_payload_accessors_agree_with_the_predicates(t_gate):
    """`node.<payload>` unwraps exactly what the matching predicate reports."""
    seen = set()
    for _, node in t_gate.nodes():
        payloads = [node.quantum, node.classical, node.region]
        assert sum(p is not None for p in payloads) == 1
        assert (node.quantum is not None) == node.is_quantum()
        assert (node.classical is not None) == node.is_classical()
        assert (node.region is not None) == node.is_region()
        seen.add(node.is_region())
    assert seen == {True, False}  # the fixture has both, so both arms ran


def test_region_kind_selects_a_variant_without_isinstance(t_gate):
    by_kind = collections.Counter(
        node.region.kind for _, node in t_gate.nodes() if node.is_region()
    )
    assert by_kind[bloq.ir.RegionKind.RepeatUntilSuccess] == 1
    assert by_kind[bloq.ir.RegionKind.RepeatUntilSuccess] == sum(
        isinstance(node.kind, bloq.ir.BloqNodeKind.Region)
        and isinstance(node.kind.region, bloq.ir.RegionNode.RepeatUntilSuccess)
        for _, node in t_gate.nodes()
    )


def test_edge_predicates_partition_every_edge(t_gate):
    for edge in t_gate.edges():
        assert [edge.is_quantum(), edge.is_value(), edge.is_compose(), edge.is_order()].count(True) == 1
        assert edge.is_quantum() == isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)


def test_subgraph_quantum_nodes_mirror_the_program(t_gate):
    # `QuantumNode` snapshots have no equality, so compare the shape they carry.
    def shape(pairs):
        return [(nid, len(node.instances)) for nid, node in pairs]

    assert shape(t_gate.level_at([]).quantum_nodes()) == shape(t_gate.quantum_nodes())
    body = next(
        node.region.body
        for _, node in t_gate.nodes()
        if node.is_region() and node.region.kind is bloq.ir.RegionKind.RepeatUntilSuccess
    )
    assert shape(body.quantum_nodes()) == shape(
        (nid, node.quantum) for nid, node in body.nodes() if node.is_quantum()
    )


def test_open_value_producers_mirror_between_receivers(t_gate):
    assert t_gate.open_value_producers() == t_gate.level_at([]).open_value_producers()


def test_declared_region_outputs_round_trip_independently_of_consumers():
    program = bloq.Bloq.from_text("""BLOQIR 1
graph {
  n0 rus 0 {
    body {
      n0 observable fragment
      n1 compute 1
      n2 compute !in0
      n1 -> n2 value 0
      result n1
      bindings n0
    }
  }
  result n0
  bindings n0
}
""")
    for restored in [program, bloq.Bloq.from_text(program.to_text()),
                     bloq.Bloq.from_binary(program.to_binary())]:
        assert restored.value_output == restored.level_at([]).value_output == bloq.ir.ValueRef(0)
        assert restored.boundary_outputs == restored.level_at([]).boundary_outputs == [0]
        body = restored.node(0).kind.region.body
        assert body.value_output == bloq.ir.ValueRef(1)
        assert body.boundary_outputs == [0]
        assert 1 not in body.open_value_producers()


# ---------------------------------------------------------------------------
# structural queries, seam-locating padding, resolver, stable identity
# ---------------------------------------------------------------------------


def test_regions_of_finds_every_nesting_level():
    program = bloq.Bloq.from_text("""BLOQIR 1
graph {
  n0 rus in0 source n0 {
    body {
      n0 compute 0
      n1 rus 0 {
        body {
          n0 rus 0 {
            body {
            }
          }
        }
      }
    }
  }
}
""")
    rus = program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)
    assert len(rus) == 3

    # Every reference resolves in the program it came from, and its path is a
    # real `path=` argument: the region is reachable at the level it names.
    for ref in rus:
        assert isinstance(ref.resolve(program), bloq.ir.RegionNode)
        level = program.level_at([])
        for owner, selector in ref.path:
            level = getattr(level.node(owner).kind.region, selector)
        assert level.node(ref.node) is not None

    # The walk agrees on which nodes are regions and where they live.
    walked = {(tuple(map(tuple, visit.path)), visit.id)
              for visit in program.walk() if visit.node.is_region()}
    assert {
        (tuple(map(tuple, ref.path)), ref.node) for ref in rus
    } == walked


def test_selection_seams_name_the_edge_feeding_each_selection(t_gate):
    seams = t_gate.selection_seams()
    selection_ids = {nid for nid, node in t_gate.quantum_nodes() if node.guards}
    assert selection_ids
    assert {target for _, target in seams} == selection_ids
    for source, target in seams:
        assert (source, target) in {
            (edge.source, edge.target)
            for edge in t_gate.edges()
            if isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
        }


def test_selection_seams_include_every_input_of_one_selection():
    program = bloq.Bloq.from_text(
        """BLOQIR 1

template t0 {
  circuit {
    H (0,0)
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 quantum {
    instance i1 t0 @ (4,0)
  }
  n2 quantum {
    instance i2 t0 @ (0,0)
    guard 0 i2
  }
  n3 compute 1
  n3 -> n2 value 0
  n0 -> n2 quantum (0,0,0)>(0,0,1)
  n1 -> n2 quantum (1,0,0)>(1,0,1)
}
"""
    )

    assert program.selection_seams() == [(0, 2), (1, 2)]


def test_node_by_block_locates_source_members(y_memory):
    graph, program = y_memory
    for pos in sorted(graph.positions()):
        node = program.node_by_block(pos)
        assert node is not None
        assert program.node(node).is_quantum()
        assert pos in program.node(node).block_members()
    assert program.node_by_block((99, 99, 99)) is None


def test_quantum_tail_is_the_body_node_with_no_quantum_successor(t_gate):
    ref = t_gate.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
    body = ref.resolve(t_gate).body
    escape = body.quantum_tail()
    assert not [
        edge
        for edge in body.outgoing(escape)
        if isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
    ]


def test_quantum_tail_is_generic_and_requires_one_quantum_terminal():
    program = bloq.Bloq.from_text("""BLOQIR 1
graph {
  n0 quantum {
  }
  n1 compute 0
  n0 -> n1 order
}
""")
    assert program.quantum_tail() == 0
    for nodes, count in [("", 0), ("n0 quantum {\n}\nn1 quantum {\n}\n", 2)]:
        program = bloq.Bloq.from_text(f"BLOQIR 1\ngraph {{\n{nodes}}}\n")
        with pytest.raises(bloq.BloqError, match=f"{count} quantum tails"):
            program.quantum_tail()


def test_memory_round_batch_pads_every_selection_seam():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    before = program.node_count
    targets = [bloq.ir.MemoryRoundTarget.Edge(*seam)
               for seam in program.selection_seams()]
    padded = program.insert_memory_rounds_batch(targets, rounds=2)
    assert padded
    assert program.node_count == before + len(padded)
    program.validate()
    # A pinned Clifford proxy has no selections; the empty batch is a no-op.
    proxy = bloq.compile_clifford_proxy(bloq.GalleryItem.T_GATE.load(), [True], distance=3)
    assert proxy.insert_memory_rounds_batch(
        [bloq.ir.MemoryRoundTarget.Edge(*seam) for seam in proxy.selection_seams()], 2
    ) == []


def test_memory_round_batch_mixes_edges_and_region_terminals():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    targets = [bloq.ir.MemoryRoundTarget.Edge(*seam)
               for seam in program.selection_seams()]
    region = program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
    path = region.path + [(region.node, "body")]
    escape = program.level_at(path).quantum_tail()
    targets.append(bloq.ir.MemoryRoundTarget.After(escape, path=path))
    sequential = copy.copy(program)
    for target in targets:
        if isinstance(target, bloq.ir.MemoryRoundTarget.Edge):
            sequential.insert_memory_rounds(target.from_node, target.to_node, 2,
                                            path=target.path)
        else:
            sequential.insert_memory_rounds_after(target.node, 2, path=target.path)
    padded = program.insert_memory_rounds_batch(targets, 2)
    assert len(padded) == len(targets)
    assert program.level_at(path).quantum_tail() == padded[-1]
    assert program.level_at(path).node(padded[-1]).memory_rounds() == 2
    assert program.to_binary() == sequential.to_binary()
    program.validate()


def test_memory_round_batch_rolls_back_after_a_later_failure():
    program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    seam = next(edge for edge in program.edges() if edge.is_quantum())
    valid = bloq.ir.MemoryRoundTarget.Edge(seam.source, seam.target)
    before = program.to_binary()
    with pytest.raises(bloq.BloqError):
        program.insert_memory_rounds_batch(
            [valid, bloq.ir.MemoryRoundTarget.Edge(99999, 99998)], 2
        )
    assert program.to_binary() == before
    with pytest.raises(bloq.BloqError):
        program.insert_memory_rounds_batch([valid, valid], 2)
    assert program.to_binary() == before


def test_memory_round_batch_resolves_all_paths_before_editing():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    valid = bloq.ir.MemoryRoundTarget.Edge(*program.selection_seams()[0])
    region = program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
    before = program.to_binary()
    for path in [[(99999, "body")], [(region.node, "unknown")]]:
        with pytest.raises(bloq.InvalidArgumentError):
            program.insert_memory_rounds_batch(
                [valid, bloq.ir.MemoryRoundTarget.After(0, path=path)], 2
            )
        assert program.to_binary() == before


def test_memory_round_batch_rejects_zero_rounds_and_wrong_target_types():
    program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    before = program.to_binary()
    with pytest.raises(bloq.BloqError):
        program.insert_memory_rounds_batch([], 0)
    with pytest.raises(TypeError):
        program.insert_memory_rounds_batch([(0, 1)], 2)
    assert program.to_binary() == before


def test_resolve_classical_folds_the_dataflow():
    # CNOT has affine readouts. Native T output frames can be nonlinear and
    # intentionally have no single measurement-parity recipe.
    program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    classical = [nid for nid, node in program.nodes() if node.is_classical()]
    assert classical
    resolved = [program.resolve_classical(nid, pins=False) for nid in classical]
    assert any(res.measurements for res in resolved)
    for res in resolved:
        assert isinstance(res, bloq.ir.ClassicalResolution)
        # Sites are plain tuples, keyed the way the emission-plan and plan-Stim
        # column maps are, so a recipe indexes them with no conversion.
        assert all(isinstance(m, tuple) and len(m) == 2 for m in res.measurements)
        # The set is already XOR-cancelled, so no site appears twice.
        assert len(set(res.measurements)) == len(res.measurements)

    # Every resolved site is a key of some emission plan's measurement map,
    # which is what makes a recipe usable against the lowered columns without
    # a conversion step. Region bodies hold instances too, so the union is
    # taken over every level.
    plan_measurements = {
        site
        for step in program.walk()
        for site in program.emission_plan(step.id, path=step.path).measurements
    }
    assert plan_measurements >= {site for res in resolved for site in res.measurements}


def test_resolve_classical_accepts_only_one_predicate_assignment(t_gate):
    node = next(nid for nid, n in t_gate.nodes() if n.is_classical())
    # A uniform `pins` already fixes every predicate, so nothing is left for
    # `forced_observables` to say.
    with pytest.raises(bloq.InvalidArgumentError, match="forced_observables"):
        t_gate.resolve_classical(node, pins=True, forced_observables={0: True})
    assert t_gate.resolve_classical(node, forced_observables={})
    assert t_gate.resolve_classical(node, pins=True) == t_gate.resolve_classical(node)
    with pytest.raises(TypeError):
        t_gate.resolve_classical(node, pins={})


def test_resolve_classical_rejects_a_quantum_node(t_gate):
    node = next(nid for nid, n in t_gate.nodes() if n.is_quantum())
    with pytest.raises(bloq.BloqError):
        t_gate.resolve_classical(node, pins=False)


def _selection_predicate(program):
    node_id, quantum = next((nid, q) for nid, q in program.quantum_nodes() if q.guards)
    return next(
        edge.source for edge in program.incoming(node_id)
        if edge.is_value() and edge.edge.slot == quantum.guards[0].input
    )


def test_classical_value_folds_a_selection_predicate(t_gate):
    selector = _selection_predicate(t_gate)
    observables = sorted(
        node.kind.node.index
        for _, node in t_gate.nodes()
        if node.is_classical() and isinstance(node.kind.node, bloq.ir.ClassicalNode.Observable)
        and node.kind.node.index is not None
    )
    assert observables

    values = {
        decoded: t_gate.classical_value(
            selector, forced_observables=dict.fromkeys(observables, decoded)
        )
        for decoded in (False, True)
    }
    assert values[False] != values[True]


def test_classical_value_rejects_what_the_path_does_not_fix(t_gate):
    selector = _selection_predicate(t_gate)
    with pytest.raises(bloq.BloqError, match="not fixed by the resolved path"):
        t_gate.classical_value(selector, forced_observables={})

    quantum = next(nid for nid, node in t_gate.nodes() if node.is_quantum())
    with pytest.raises(bloq.BloqError):
        t_gate.classical_value(quantum, pins=False)

    # A repeat-until-success restarts on per-attempt decoder acceptance, which
    # no assignment names.
    (rus,) = t_gate.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)
    with pytest.raises(bloq.BloqError):
        t_gate.classical_value(rus.node, pins=False)

    with pytest.raises(bloq.InvalidArgumentError, match="forced_observables"):
        t_gate.classical_value(selector, pins=True, forced_observables={0: True})


def _level_shape(level):
    """A subgraph's identity as comparable data (nodes carry no `__eq__`)."""
    return ([(nid, repr(node)) for nid, node in level.nodes()], level.stable_keys())


def test_region_ref_body_reads_the_executed_subgraph(t_gate):
    (rus,) = t_gate.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)
    assert _level_shape(rus.body(t_gate)) == _level_shape(rus.resolve(t_gate).body)


def test_stable_keys_agree_across_a_clifford_proxy():
    graph = bloq.GalleryItem.T_GATE.load()
    program = bloq.compile(graph, distance=3)
    proxy = bloq.compile_clifford_proxy(graph, [True], distance=3)

    keys = program.stable_key_map()
    proxy_keys = proxy.stable_key_map()
    shared = set(keys) & set(proxy_keys)
    assert shared, "no node kept its identity across the proxy compile"
    for key in shared:
        assert program.node(keys[key]).block_members() == proxy.node(proxy_keys[key]).block_members()

    # Only nodes with a keyable provenance are in the map, and every id in it
    # is real.
    assert 0 < len(keys) <= program.node_count
    assert all(program.node(node_id) is not None for node_id in keys.values())

    # Keys are usable as dict keys and print without leaking their internals.
    key = next(iter(keys))
    assert isinstance(hash(key), int)
    assert repr(key).startswith("<NodeKey ")
    assert {key: "tagged"}[key] == "tagged"


def test_node_snapshot_carries_its_stable_key(y_memory):
    _, program = y_memory
    by_key = program.stable_key_map()
    own = {nid: node.stable_key for nid, node in program.nodes() if node.stable_key}
    assert own

    # A node reports its key before collision disambiguation, so it always has
    # ordinal 0; the program map is what assigns ordinals. The two agree for
    # every node whose key is unique to begin with.
    counts = collections.Counter(own.values())
    for node_id, key in own.items():
        assert key.ordinal == 0
        if counts[key] == 1:
            assert by_key[key] == node_id
    assert set(own) == set(by_key.values())


def test_subgraph_stable_keys_match_the_program_map(y_memory):
    _, program = y_memory
    assert dict(program.level_at([]).stable_keys()) == program.stable_key_map()


def test_set_template_repetitions_retunes_the_loop(y_memory):
    graph, _ = y_memory
    program = bloq.compile(graph, distance=3)
    edge = next(
        (edge.source, edge.target)
        for edge in program.edges()
        if isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
        and edge.source == 1
    )
    padding_id = program.insert_memory_rounds(*edge, rounds=5)
    # The pool holds looped templates the program never instantiates, so take
    # the one the new padding node actually placed.
    padding = program.node(padding_id).kind.node.instances
    looped = [
        instance.template_id
        for instance in padding
        if any(
            isinstance(op, bloq.ir.CircuitOp.Repeat)
            for op in program.template(instance.template_id).circuit.ops()
        )
    ]
    assert looped, "memory padding should have placed a looped template"

    before = bloq.emit_stim(program)
    program.set_template_repetitions(looped[0], 9)
    assert bloq.emit_stim(program) != before
    assert "REPEAT 9" in str(bloq.emit_stim(program))

    with pytest.raises(bloq.BloqError, match="unknown template"):
        program.set_template_repetitions(program.template_count, 2)


def test_continuing_membership_snapshots_and_pins():
    graph = bloq.BlockGraph.from_text("""BLOG 1.0
0: ZXZ [0,0,0]
1: ZXZ [0,0,2]
9: ZXZ [2,0,0]
branch b {
  false {
    2: XZX [0,0,1]
    0 -H> +Z
    [0,0,1] -H> +Z
  }
  true {
    3: ZXZ [0,0,1]
    0 -> +Z
    [0,0,1] -> +Z
  }
}
m = measure 9
resolve b if m
""")
    program = bloq.compile(graph, distance=3)
    assert program.has_conditional_membership()
    assert any(q.guards for _, q in program.quantum_nodes())
    assert all(isinstance(guard, bloq.ir.QuantumGuard)
               for _, q in program.quantum_nodes() for guard in q.guards)
    assert any(isinstance(n.provenance, bloq.ir.NodeProvenance.BranchSelector)
               for _, n in program.nodes())
    saved = program.to_binary()
    for selected in (False, True):
        pinned = program.pin_membership({"b": selected})
        pinned.validate()
        assert not pinned.has_conditional_membership()
        assert bloq.emit_stim(pinned)
    assert program.to_binary() == saved
    with pytest.raises(bloq.BloqError, match="no pin"):
        program.pin_membership({})


# ---------------------------------------------------------------------------
# Structural read accessors shared by `Bloq` and `SubGraph`
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cnot():
    return bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)


def endpoints(refs):
    """`(source, target)` pairs; `BloqEdgeRef` is deliberately not comparable."""
    return [(ref.source, ref.target) for ref in refs]


def test_edges_between_selects_exactly_the_edges_joining_two_nodes(cnot):
    pairs = endpoints(cnot.edges())
    for source, target in set(pairs):
        between = cnot.edges_between(source, target)
        # Parallel edges of different kinds are legal, so this counts them
        # rather than assuming one.
        assert endpoints(between) == [(source, target)] * len(between)
        assert len(between) == pairs.count((source, target))
    assert cnot.edges_between(0, 9999) == []
    assert cnot.edges_between(9999, 0) == []


def test_quantum_input_names_the_single_quantum_predecessor(cnot):
    resolved = 0
    for node_id in cnot.node_ids():
        sources = {ref.source for ref in cnot.incoming(node_id) if ref.is_quantum()}
        if len(sources) == 1:
            resolved += 1
            assert cnot.quantum_input(node_id) == sources.pop()
        else:
            with pytest.raises(bloq.BloqError):
                cnot.quantum_input(node_id)
    assert resolved > 0


def test_value_inputs_and_value_consumers_are_two_views_of_one_edge(cnot):
    seen = 0
    node_ids = cnot.node_ids()
    for consumer in node_ids:
        for value_input in cnot.value_inputs(consumer):
            seen += 1
            assert (consumer, value_input.slot) in cnot.value_consumers(
                value_input.producer
            )
            assert value_input.producer in node_ids
            assert isinstance(value_input.role, bloq.ir.ValueRole.Data)
    assert seen > 0
    assert cnot.value_inputs(9999) == []
    assert cnot.value_consumers(9999) == []


def test_sub_graph_read_accessors_match_the_top_level_spelling(t_gate):
    """The region-body reads delegate to the same helpers as `Bloq`'s."""
    body = next(
        level for path, level in t_gate.levels() if path and level.node_count > 1
    )
    for source, target in set(endpoints(body.edges())):
        between = body.edges_between(source, target)
        assert endpoints(between) == [(source, target)] * len(between)

    paired = 0
    for level in (body, t_gate.level_at([])):
        for consumer in level.node_ids():
            for value_input in level.value_inputs(consumer):
                paired += 1
                assert (consumer, value_input.slot) in level.value_consumers(
                    value_input.producer
                )
    assert paired > 0

    for node_id in body.node_ids():
        sources = {ref.source for ref in body.incoming(node_id) if ref.is_quantum()}
        if len(sources) == 1:
            assert body.quantum_input(node_id) == sources.pop()


def test_node_qubits_are_the_nodes_footprint_in_the_program_layout(cnot):
    layout = set(cnot.sorted_layout_coords())
    quantum = 0
    for node_id, node in cnot.nodes():
        qubits = cnot.node_qubits(node_id)
        if not node.is_quantum():
            assert qubits == []
            continue
        quantum += 1
        assert qubits
        assert len(set(qubits)) == len(qubits)
        assert set(qubits) <= layout
    assert quantum > 0


def test_subdivide_quantum_edge_splices_one_padding_node(cnot):
    program = bloq.Bloq.from_binary(cnot.to_binary())
    reference, padding = next(
        (ref, seam.padding)
        for ref in program.edges()
        if ref.is_quantum()
        for seam in ref.edge.pipes
        if seam.padding is not None
    )
    source, target = reference.source, reference.target
    before = program.node_count

    spliced = program.subdivide_quantum_edge(
        source,
        target,
        padding=[(padding.one_round, padding.offset)],
        rounds=1,
    )

    assert program.node_count == before + 1
    assert spliced not in (source, target)
    # The quantum edge is gone, replaced by the pair through the padding node.
    # The `Order` edge between the two endpoints is untouched.
    assert not any(
        ref.is_quantum() for ref in program.edges_between(source, target)
    )
    assert endpoints(program.outgoing(spliced)) == [(spliced, target)]
    assert (source, spliced) in endpoints(program.incoming(spliced))
    program.validate()


# ---------------------------------------------------------------------------
# Side tables: detector bundles, guards, templates, resolutions
# ---------------------------------------------------------------------------


def program_instances(program):
    """Every template instance in the program, keyed by program-global id."""
    return {
        instance.id: instance
        for _, level in program.levels()
        for _, node in level.nodes()
        if node.is_quantum()
        for instance in node.quantum.instances
    }


def test_detector_bundle_uses_bind_instances_matching_the_owner_templates(t_gate):
    assert t_gate.detector_bundle_count > 0
    instances = program_instances(t_gate)
    uses = 0
    for _, level in t_gate.levels():
        for _, node in level.nodes():
            if not node.is_quantum():
                continue
            for use in node.quantum.detector_bundles:
                uses += 1
                bundle = t_gate.detector_bundle(use.bundle)
                # A bundle use may span a seam, so the bound instances need not
                # all belong to the using node — but each must match the
                # template its owner slot expects.
                assert [
                    instances[bound].template_id for bound in use.instances
                ] == bundle.owner_templates
                assert len(use.offset) == 2
                for detector in bundle.detectors:
                    assert detector.terms
                    for term in detector.terms:
                        if isinstance(term, bloq.ir.BundleDetectorTerm.Measurement):
                            assert term.owner < len(bundle.owner_templates)
                    if detector.coords is not None:
                        assert len(detector.coords) >= 2
    assert uses > 0


def test_quantum_guards_index_their_owning_nodes_side_tables(t_gate):
    guarded = 0
    for _, level in t_gate.levels():
        for node_id, node in level.nodes():
            if not node.is_quantum():
                continue
            quantum = node.quantum
            slots = {value_input.slot for value_input in level.value_inputs(node_id)}
            owned = {instance.id for instance in quantum.instances}
            for guard in quantum.guards:
                guarded += 1
                assert guard.input in slots
                assert set(guard.instances) <= owned
                assert all(row < len(quantum.detectors) for row in guard.detectors)
                assert all(row < len(quantum.restarts) for row in guard.restarts)
                assert all(
                    row < len(quantum.detector_bundles)
                    for row in guard.detector_bundles
                )
                # Conditional XOR contributions name rows of those same tables.
                for row, _ in guard.detector_parities:
                    assert row < len(quantum.detectors)
                for row, _ in guard.restart_parities:
                    assert row < len(quantum.restarts)
    assert guarded > 0


def test_template_instance_offsets_place_a_pooled_template(cnot):
    placed = 0
    for _, node in cnot.nodes():
        if not node.is_quantum():
            continue
        for instance in node.quantum.instances:
            placed += 1
            assert 0 <= instance.template_id < cnot.template_count
            assert len(instance.offset) == 2
            for detector in cnot.template(instance.template_id).detectors:
                if detector.coords is not None:
                    assert len(detector.coords) >= 2
    assert placed > 0


def test_repeat_states_appear_exactly_for_templates_with_a_repeat_body(t_gate):
    looping = 0
    for template_id in range(t_gate.template_count):
        template = t_gate.template(template_id)
        circuit = template.circuit
        assert 0 <= circuit.entry_body < circuit.body_count
        if not template.repeat_states:
            continue
        looping += 1
        assert circuit.body_count > 1
        for state in template.repeat_states:
            assert state.body < circuit.body_count
            # The carried state is a parity in both the entry and loop rows.
            assert isinstance(state.initial, bloq.ir.TemplateParity)
            assert isinstance(state.next, bloq.ir.TemplateParity)
    assert looping > 0


def test_boundary_flow_endpoints_and_center_stay_in_template_space(cnot):
    flows = 0
    for template_id in range(cnot.template_count):
        template = cnot.template(template_id)
        qubits = set(template.circuit.qubits())
        for flow in template.boundary_flows:
            flows += 1
            # `start` / `end` are Pauli supports over the template's own
            # qubits, one of which may be empty for an opening or closing flow.
            for support in (flow.start, flow.end):
                assert {coord for coord, _ in support} <= qubits
            assert flow.start or flow.end
            if flow.center is not None:
                assert len(flow.center) == 2
            assert isinstance(flow.sign, bool)
    assert flows > 0


def test_resolved_classical_values_report_sorted_decoder_observables(t_gate):
    resolved = 0
    for node_id, node in t_gate.nodes():
        if not node.is_classical():
            continue
        resolution = t_gate.resolve_classical(node_id, pins=True)
        resolved += 1
        # Both term lists are documented as ascending, and a resolution is
        # looked up by those keys, so duplicates would be a defect.
        assert resolution.decoder_observables == sorted(
            set(resolution.decoder_observables)
        )
        assert resolution.measurements == sorted(set(resolution.measurements))
        assert isinstance(resolution.sign, bool)
    assert resolved > 0


def test_node_activation_names_a_guarding_value_producer(t_gate):
    activated = 0
    for _, level in t_gate.levels():
        node_ids = set(level.node_ids())
        for _, node in level.nodes():
            if node.activation is None:
                continue
            activated += 1
            # Classical activation gates the record independently of quantum
            # work, and its producer lives at the same level.
            assert node.activation in node_ids
    assert activated > 0


def test_instance_measurements_sort_into_instance_then_column_order(t_gate):
    """`InstanceMeasurement` is ordered, so a gathered set sorts usefully."""
    sites = {
        site
        for _, level in t_gate.levels()
        for _, node in level.nodes()
        if node.is_quantum()
        for restart in node.quantum.restarts
        for site in restart.parity.measurements()
    }
    assert len(sites) > 1
    assert [(site.instance, site.measurement) for site in sorted(sites)] == sorted(
        (site.instance, site.measurement) for site in sites
    )


def test_conditional_corrections_gate_on_a_real_measurement_record():
    """`and_4t` is the gallery entry whose templates carry feedforward."""
    program = bloq.compile(bloq.GalleryItem.AND_4T.load(), distance=3)
    corrections = 0
    for template_id in range(program.template_count):
        circuit = program.template(template_id).circuit
        for body in range(circuit.body_count):
            for op in circuit.ops(body):
                if not isinstance(op, bloq.ir.CircuitOp.ConditionalPauli):
                    continue
                for correction in op.corrections:
                    corrections += 1
                    # The control names a measurement this circuit produces.
                    assert 0 <= correction.control < circuit.num_measurements
                    assert correction.target in set(circuit.qubits())
                    assert correction.pauli in (
                        bloq.PauliBasis.X,
                        bloq.PauliBasis.Y,
                        bloq.PauliBasis.Z,
                    )
    assert corrections > 0
