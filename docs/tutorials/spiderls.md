# Compile quantum algorithms through SpiderLS

The optional [SpiderLS](https://github.com/fluorite42/SpiderLS) integration runs
static Clifford+T OpenQASM 2 through ZX reduction, execution planning,
parallel scheduling, routing, Bloq block graphs, validated IR, and the VM.

Use the [MQT Bench suite](mqt-bench.md) for the primary algorithm comparisons.
The custom circuits below remain regression examples.

## Install and run

From a Bloq checkout, use Python 3.11–3.14 and the audited upstream revision:

```sh
uv sync --project bloq_py --locked --group dev --group interop --inexact
git clone https://github.com/fluorite42/SpiderLS ../spiderls
git -C ../spiderls checkout 2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea
git -C ../spiderls apply "$PWD/bloq_py/examples/spiderls.patch"
git -C ../spiderls add -N .
export PYTHONPATH="$PWD/../spiderls/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --project bloq_py --no-sync python bloq_py/examples/spiderls_algorithms.py \
    --candidate walk3 --shots 3 --output-dir target/spiderls-walk
```

SpiderLS is an unpackaged source checkout; only its `src` directory needs to be
on `PYTHONPATH`. The headless integration uses Bloq's pinned PyZX 0.10.7 rather
than upstream's declared 0.10.2. Base Bloq does not import or require SpiderLS.

The example also executes larger candidates:

| Candidate | Logical qubits | Source T/T† gates | Algorithm check |
| --- | ---: | ---: | --- |
| `simon6` | 6 | 7 | Recover period `101` from the input register's Fourier support |
| `qaoa5` | 5 | 10 | Five-cycle expected cut 3.75; maximum cut 4 |
| `ising5` | 5 | 10 | Reproduce two Trotter steps for a five-spin chain |
| `iqp6` | 6 | 8 | Even output parity and the analytic all-zero probability |

For example, replace `--candidate walk3` above with `--candidate simon6`.
Use `--spacing 2 --routing-margin 1 --placement-margin 1` to try a tighter
upstream routing canvas. The same options are available on `from_qasm` as
`spacing`, `routing_margin`, and `placement_margin`. Smaller margins can
increase routing depth or cause failure; compare validated layouts rather than
assuming a smaller canvas reduces occupied spacetime volume. Bloq places its
resource endpoints after upstream routing.

The patched router scales its finite layer budget with logical schedule depth.
Pass `max_physical_layers` to `from_qasm` for an explicit hard cap; the effective
budget is recorded in `layout_options`.
Simon uses three input and three output qubits with the nonlinear oracle
`f(x) = (x0 xor x2, x1, (x0 xor x2) and x1)`. Its input marginal is uniform on
`000`, `010`, `101` and `111`. The report recovers the unique nonzero period
orthogonal to this support. This recovery uses probabilities derived from the
VM state, rather than a finite set of computational-basis samples.

Each run requires an empty output
directory and retains source and Clifford+T QASM, `layout.blog`, an HTML viewer,
validated `program.bloq`, and JSON results. Independent matrix oracles check
synthesis and every joint output Pauli expectation: 1,023 for five outputs and
4,095 for six. Each shot also records source-ordered `logical_probabilities`
and applicable `algorithm_checks`. These noiseless state checks are distinct
from statistical estimates from sampled algorithm outputs. Other candidates
are useful compatibility probes and may reject.

For Ising dynamics, fidelity is checked against the Trotterized circuit.
`trotter_spectral_error` separately reports its operator-norm error against
continuous Hamiltonian evolution. Two coarse steps are a compilation demo,
not an accurate continuous-evolution approximation.

For direct use:

```python
import bloq
from bloq.interop import compile_qasm

compiled = compile_qasm('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
tdg q[0];
''', compiler="spiderls", distance=3, prepare_t_with_mpps=True)
imported, program = compiled.imported, compiled.program
report = bloq.run_noiseless(program, shots=3, input_state="zero", include_signature=True)
```

This entry point searches compact reusable T patches using native lifetime
conflicts and returns validated IR. Its `placement_attempts` records footprint
and timing. `from_qasm` in `bloq.interop.spiderls` remains a pure graph importer
with private T columns; the routing options above apply to both entry points.

## Semantics and limits

- The patch preserves π phases as ordered Z events. The adapter enables input
  and output Hadamards, retains exact S/S† and T/T† signs, and lowers 3π/4 and
  5π/4 with one signed T and a residual Pauli. No extra S gadgets replace T†.
  Missing phase stages and incomplete or duplicate routing histories reject.
- Original ZX boundaries determine port order. The event DAG validates every
  dependency and patch lifetime. H events become pipe Hadamards; Z events become
  Pauli feedback, conjugated through any intervening H gates. Preparation and
  release remain endpoint cubes; a release is an X effect. Separate lifetimes
  are never joined. Reduced identity/H wires are preserved even when SpiderLS
  omits them from target code.
- Routed layers start on the upstream unit grid. Only endpoint collisions add
  temporal gaps, preserving the order at each spatial cell. `grid_spacing`
  records `(1, 1, 1)`; endpoints, local gaps, and T gadgets contribute to imported
  volume. Idle slices are contracted while protecting resources, controls,
  ports, and tags; temporal H seams compose by parity.
- T attachments and spatial Hadamard walls use the shared handling described
  for [TopoLS](topols.md#semantic-bridge-and-limits).
- `SpiderLSImport` retains the reduced graph, plan, target, schedule, routing,
  options, revision, and ordered ports.
  Bloq derives readouts and adaptive dependencies from the imported graph.
- Unsupported angles, dynamic source control, missing provenance, collisions,
  unsupported frames, exhausted placement budgets, and destructive patch reuse
  reject. Try different upstream routing options when placement fails.
- VM checks use ideal T-source proxies, noiseless physical circuits, ideal input
  preparation, and boundary capture. Spatial Hadamard pipes retain Bloq's
  distance warning; these demonstrations make no fault-tolerant distance claim.

Run adapter checks with `pytest bloq_py/tests/interop/test_spiderls.py`. Independent
QuiZX and physical channel checks use
`cargo test --locked -p bloq_vm --test interop_spiderls`. Regenerate their pinned
fixtures with `bloq_py/tests/interop/export_spiderls_fixtures.py --upstream-root
../spiderls` in the same environment.
