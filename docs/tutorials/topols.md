# Compile quantum algorithms through TopoLS

The optional [TopoLS](https://github.com/tqec/TopoLS) adapter provides:

```text
Logical circuit → Clifford+T OpenQASM 2 → TopoLS → BlockGraph → Bloq IR → VM
```

## Install

Use a Bloq source checkout and Python 3.11–3.14. Install the headless dependencies
and the pinned upstream with its signed-phase patch:

```sh
uv sync --project bloq_py --locked --group dev --group interop --inexact
git clone https://github.com/tqec/TopoLS ../topols
git -C ../topols checkout e564d350a596cfea3227e09698733210e2e14dcf
git -C ../topols apply "$PWD/bloq_py/examples/topols.patch"
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
from bloq.interop.topols import from_qasm

imported = from_qasm('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
tdg q[0];
''', engine="python")
imported.graph.save("layout.blog")
program = bloq.compile(imported.graph, distance=3, prepare_t_with_mpps=True, validate=True)
program.save("program.bloq")
report = bloq.run_noiseless(program, shots=3, input_state="zero", include_signature=True)
```

`TopoLSImport` retains original and normalized QASM, the compact embedding,
exact signed resource phases, layout options, and ports in source-qubit order.
VM outputs use coordinate order; match them by `port`. `TopoLSImportError.code`
identifies source, provenance, and geometry failures; native compile errors remain
authoritative.

Run four-spin Ising dynamics with two Trotter steps and eight T gates:

```sh
uv run --project bloq_py --no-sync python bloq_py/examples/topols_algorithms.py \
    --candidate ising4 --shots 3 --output-dir target/topols-ising
```

The same runner supports phase estimation, Grover search, QAOA, quantum walks,
amplitude estimation, a variational ansatz, and coherent addition. For addition,
use `--candidate adder4 --layout-seed 2`. Add `--engine python` to use the reference
engine. Each run needs an empty output directory and retains source/normalized
QASM, BLOG, a viewer, validated IR, and JSON results. Independent matrix oracles
check synthesis; execution checks every joint output Pauli expectation.

## Semantic bridge and limits

- Sources are static Clifford+T unitary QASM 2. Exact rational multiples of `pi`
  with denominator at most four are lowered; synthesize other angles first.
  Measurements, resets, classical control, and identity-only sources are rejected.
- The upstream patch recognizes signed S/S† and T/T† and preserves their exact
  phases through spider fusion. Bloq binds each routed resource to its prepared
  or fallback source vertex. T† uses one T resource, with no extra S gadgets.
- S endpoints become signed Y preparations/measurements. T endpoints receive
  Bloq's positive T source and adaptive X/Y measurement. For corrected ZZ bit
  `b`, T reads X when `b=0` and Y when `b=1`; T† exchanges them. Bloq computes
  the corrected readouts, dependencies, and output frames from the imported graph.
- Imported coordinates are uniformly spaced by three; identity cubes subdivide
  pipes and preserve their Hadamard parity. This leaves room for endpoint gadgets
  and enlarges the layout. The original compact embedding remains available for
  provenance; adapted volume is not TopoLS's optimized volume.
- Routing success does not guarantee compatibility. Unsupported frames,
  inconsistent interfaces, mixed-axis spatial Hadamard walls, or destructive patch
  reuse fail explicitly. In particular, adder seed zero currently produces
  duplicate output metadata, and amplitude-estimation seed zero exports overlapping
  nodes. The example also checks the compiled output interface before execution.
  The adapter performs no lifetime-aware rerouting.
- VM checks use ideal T-source proxies, noiseless physical circuits, ideal input
  preparation, and ideal boundary capture. They do not claim cultivation
  acceptance, hardware timing, or fault-tolerant distance.

Run adapter checks with `pytest bloq_py/tests/interop/test_topols.py` and independent
logical/physical checks with `cargo test --locked -p bloq_vm --test interop_topols`.
Regenerate fixtures with `bloq_py/tests/interop/export_topols_fixtures.py
--upstream-root ../topols`; this checks the checkout and both layout engines.
