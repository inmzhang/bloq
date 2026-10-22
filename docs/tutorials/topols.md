# Compile quantum algorithms through TopoLS

Use the [MQT Bench suite](mqt-bench.md) for the primary algorithm comparisons.
The custom circuits below remain regression examples.

The optional [TopoLS](https://github.com/tqec/TopoLS) adapter provides:

```text
Logical circuit → Clifford+T OpenQASM 2 → TopoLS → BlockGraph → Bloq IR → VM
```

## Install

Use a Bloq source checkout and Python 3.11–3.14. Install the headless dependencies
and the pinned upstream with its integration patch:

```sh
uv sync --project bloq_py --locked --group dev --group interop --inexact
git clone https://github.com/tqec/TopoLS ../topols
git -C ../topols checkout e564d350a596cfea3227e09698733210e2e14dcf
git -C ../topols apply "$PWD/bloq_py/examples/topols.patch"
git -C ../topols add -N .
uv pip install --python bloq_py/.venv/bin/python --no-deps --editable ../topols
```

This headless path uses Bloq's pinned PyZX 0.10.7; upstream declares 0.10.6.
The adapter's tests exercise 0.10.7 with both layout engines. Base Bloq does not
import or require TopoLS. Use `--no-sync` for subsequent commands to retain the
separately installed checkout.

The Python engine needs no upstream extension. For faster algorithm routing,
build TopoLS's matching Rust engine in the same environment:

```sh
VIRTUAL_ENV="$PWD/bloq_py/.venv" uvx --from 'maturin>=1.15,<2' maturin develop \
    --release --manifest-path ../topols/rust/topols_core/Cargo.toml
```

## Import or execute

```python
import bloq
from bloq.interop import compile_qasm

compiled = compile_qasm('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
tdg q[0];
''', compiler="topols", engine="python", distance=3, prepare_t_with_mpps=True)
imported, program = compiled.imported, compiled.program
imported.graph.save("layout.blog")
program.save("program.bloq")
report = bloq.run_noiseless(program, shots=3, input_state="zero", include_signature=True)
```

`TopoLSImport` retains original and normalized QASM, the compact embedding,
exact signed resource phases, layout options, and ports in source-qubit order.
VM outputs use coordinate order; match them by `port`. `TopoLSImportError.code`
identifies source, provenance, and geometry failures; native compile errors remain
authoritative.

`compile_qasm` routes upstream once, searches reusable T-resource patches, and
returns only natively validated IR. `placement_attempts` records footprint and
timing for accepted and rejected candidates. Use `from_qasm` from
`bloq.interop.topols` for pure graph import with conservative private T columns.

Run four-spin Ising dynamics with two Trotter steps and eight T gates:

```sh
uv run --project bloq_py --no-sync python bloq_py/examples/topols_algorithms.py \
    --candidate ising4 --shots 3 --output-dir target/topols-ising
```

The same runner supports phase estimation, Grover search, QAOA, quantum walks,
amplitude estimation, a variational ansatz, and coherent addition. Add
`--engine python` to use the reference engine. Each run needs an empty output
directory and retains source/normalized
QASM, BLOG, a viewer, validated IR, and JSON results. Independent matrix oracles
check synthesis; execution checks every joint output Pauli expectation.

## Semantic bridge and limits

- Sources are static Clifford+T unitary QASM 2. Exact rational multiples of `pi`
  with denominator at most four are lowered; synthesize other angles first.
  Measurements, resets, classical control, and identity-only sources are rejected.
- The upstream patch recognizes signed S/S† and T/T† and preserves their exact
  phases through spider fusion. It also retains only live output boundaries and
  keeps S-resource stubs above finished layers and clear of routed inputs.
  Layer lifts preserve the color of branching spiders, including continuations
  with a temporal normal; recoloring a junction would change its logical map.
  Bloq binds each routed resource to its prepared or fallback source vertex.
  T† uses one T resource, with no extra S gadgets.
- Gate-by-gate fallback reserves its inherited frontier even after Hadamard
  dissolution, so real gates are not skipped. Initial T outlets need a nonzero
  arm. Supported temporal-X endpoints gain an identity elbow and a spatial
  resource leg; other temporal frames still reject.
- S endpoints become signed Y preparations/measurements. T endpoints receive
  Bloq's positive T source and adaptive X/Y measurement. For corrected ZZ bit
  `b`, T reads X when `b=0` and Y when `b=1`; T† exchanges them. Bloq computes
  the corrected readouts, dependencies, and output frames from the imported graph.
- Upstream spatial coordinates are preserved. Only time is initially spaced by
  three to fit endpoint gadgets; complete idle slices are then contracted while
  protecting resources, controls, ports, and tags. Temporal H seams compose by
  parity. Signed Y caps are reserved before T attachments. `grid_spacing` records
  `(1, 1, 3)` before temporal contraction; the original embedding is retained.
  Endpoint gadgets still contribute to the imported volume.
- Pure imports try short arms, joint doglegs, and bounded maze routing with
  private T columns. `compile_qasm` instead reuses columns and learns conflicting
  resource assignments from native physical-lifetime witnesses, including a
  bounded batch from each failed compile. Persistent waits trigger exclusive
  outlets for implicated resources while the remaining factories keep sharing.
  Narrow exits route first. Source Z
  separation never certifies reuse: every accepted layout passes full native
  validation at the requested distance and a binary round trip. Intermediate
  construction defers readout analysis until this final check. Identity arms
  preserve the signed injection recipe. Search is bounded and not optimal;
  stalled refinement can try private columns within the same attempt budget.
  The fallback still requires full native validation; exhausted searches reject.
- Before T attachment, the importer moves spatial Hadamard walls onto temporal
  pipes, conjugating explicit X/Z edge feedback with its source cube's frame.
  Ports, resources, and unproved action frames remain fixed. If complete
  elimination is impossible, it tries retaining walls on one spatial axis.
  Irreducible mixed-axis components still reject; supported remaining walls
  retain Bloq's distance warning.
- Routing success does not guarantee compatibility. Unsupported frames,
  inconsistent interfaces, mixed spatial Hadamard-wall axes, exhausted
  placement budgets, and destructive patch reuse reject.
  The adapter performs no lifetime-aware upstream rerouting.
- VM checks use ideal T-source proxies, noiseless physical circuits, ideal input
  preparation, and ideal boundary capture. They do not claim cultivation
  acceptance, hardware timing, or fault-tolerant distance.

Run adapter checks with `pytest bloq_py/tests/interop/test_topols.py` and independent
logical/physical checks with `cargo test --locked -p bloq_vm --test interop_topols`.
Regenerate fixtures with `bloq_py/tests/interop/export_topols_fixtures.py
--upstream-root ../topols`; this checks the checkout and both layout engines.
