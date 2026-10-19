# Compile quantum algorithms through SpiderLS

The optional [SpiderLS](https://github.com/fluorite42/SpiderLS) integration runs
static Clifford+T OpenQASM 2 through ZX reduction, execution planning,
parallel scheduling, routing, Bloq block graphs, validated IR, and the VM.

## Install and run

From a Bloq checkout, use Python 3.11–3.14 and the audited upstream revision:

```sh
uv sync --project bloq_py --locked --group dev --group interop --inexact
git clone https://github.com/fluorite42/SpiderLS ../spiderls
git -C ../spiderls checkout 2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea
git -C ../spiderls apply "$PWD/bloq_py/examples/spiderls.patch"
export PYTHONPATH="$PWD/../spiderls/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --project bloq_py --no-sync python bloq_py/examples/spiderls_algorithms.py \
    --candidate walk3 --shots 3 --output-dir target/spiderls-walk
```

SpiderLS is an unpackaged source checkout; only its `src` directory needs to be
on `PYTHONPATH`. The headless integration uses Bloq's pinned PyZX 0.10.7 rather
than upstream's declared 0.10.2. Base Bloq does not import or require SpiderLS.

The example also executes `walk5` and `qaoa4`. Each run requires an empty output
directory and retains source and Clifford+T QASM, `layout.blog`, an HTML viewer,
validated `program.bloq`, and JSON results. Independent matrix oracles check
synthesis and every joint output Pauli expectation. Other candidates are useful
compatibility probes and may reject.

For direct use:

```python
import bloq
from bloq.interop.spiderls import from_qasm

imported = from_qasm('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
tdg q[0];
''')
program = bloq.compile(imported.graph, distance=3, prepare_t_with_mpps=True, validate=True)
report = bloq.run_noiseless(program, shots=3, input_state="zero", include_signature=True)
```

## Semantics and limits

- The patch preserves π phases as ordered Z events. The adapter enables input
  and output Hadamards, retains exact S/S† and T/T† signs, and lowers 3π/4 and
  5π/4 with one signed T and a residual Pauli. No extra S gadgets replace T†.
  Missing phase stages and incomplete or duplicate routing histories reject.
- Original ZX boundaries determine port order. Patch IDs can be reused;
  preparation, H, Z, and release events receive ordered time slices. A release
  is an X effect. Separate lifetimes are never joined. Reduced identity/H wires
  are preserved even when SpiderLS omits them from target code.
- Spacing by three leaves room for the shared, channel-checked resource gadgets.
  Added phase-zero cubes preserve the tensor map. The imported volume therefore
  differs from SpiderLS's optimized volume. `SpiderLSImport` retains the reduced
  graph, plan, target, schedule, routing, options, revision, and ordered ports.
  Bloq derives readouts and adaptive dependencies from the imported graph.
  Saved BLOG and viewer files show that source layout; compilation may change
  cube frames and Hadamard placement while retaining its positions and topology.
- Unsupported angles, dynamic source control, missing provenance, collisions,
  unsupported frames, and destructive patch reuse fail explicitly. Normal Bloq
  compilation moves spatial Hadamards through ordinary cubes while preserving
  protected resource and control frames. When complete removal is impossible,
  each spatial component retains walls on at most one axis. Conflicting protected
  frames still reject with an anchored-path diagnostic; see
  [compilation](../theory/compilation.md).
- VM checks use ideal T-source proxies, noiseless physical circuits, ideal input
  preparation, and boundary capture. Spatial Hadamard pipes retain Bloq's
  distance warning; these demonstrations make no fault-tolerant distance claim.

Run adapter checks with `pytest bloq_py/tests/interop/test_spiderls.py`. Independent
QuiZX and physical channel checks use
`cargo test --locked -p bloq_vm --test interop_spiderls`. Regenerate their pinned
fixtures with `bloq_py/tests/interop/export_spiderls_fixtures.py --upstream-root
../spiderls` in the same environment.
