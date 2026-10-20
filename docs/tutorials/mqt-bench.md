# Benchmark compiler integrations with MQT Bench

The primary algorithm suite uses [MQT Bench](https://github.com/munich-quantum-toolkit/bench)
2.3.0. Each algorithm-level circuit is lowered to Clifford+T; the identical
lowered QASM is passed to Topologiq, TopoLS, and SpiderLS. Custom algorithm
examples remain available as focused regressions.

## Run

Build Bloq's Python extension, install the `interop` dependency group, and
install the patched upstreams from the [Topologiq](topologiq.md),
[TopoLS](topols.md), and [SpiderLS](spiderls.md) tutorials. Use Python 3.13 for
the pinned experiment environment and keep SpiderLS's `src` on `PYTHONPATH`.
The experiment runner uses POSIX process groups and resource limits. Build the
independent graph checker once:

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo build --release --locked \
    -p bloq_vm --example verify_interop_layout
rtk proxy uv run --project bloq_py --no-sync python \
    bloq_py/examples/mqt_bench_experiments.py --cases full_adder_4_coherent grover_3 qpeexact_3 \
    --output-dir target/mqt-small --jobs 2
```

Omit `--cases` for the complete suite. `--compilers topols spiderls` limits
compiler comparisons. Each run requires an empty output directory and retains
stage logs and failures. Layout/IR, graph checks, and VM execution have separate
budgets; see `--help` for available limits.

Generate a fresh report:

```sh
rtk proxy uv run --project bloq_py --no-sync python \
    bloq_py/examples/mqt_bench_report.py target/mqt-small
```

Reports, viewers, source circuits, and binary programs stay under the output
directory. The manifest records package versions, source hashes, upstream
revisions and patches, and the Bloq revision and working-tree snapshot.

## Semantics and evidence

- Unmodified cases retain the MQT program in `source.qasm`. The separately
  labeled `full_adder_4_coherent` demonstration also retains the original in
  `benchmark.qasm` and prepares its three inputs with Hadamards before the
  unchanged adder; `source.qasm` and its own hash include that wrapper.
  This produces a nonstabilizer output without changing the adder's angles or
  operations. Unmodified benchmarks remain the primary comparison suite.
- Source QASM 3 retains MQT's angles and ordered terminal measurement mapping.
  Static Clifford+T QASM 2 is the upstream input. Terminal measurements become
  the recorded readout specification; resets, internal measurements, and dynamic
  control are explicitly rejected.
- Exact Clifford+T identities are used where available. Other rotations use
  pinned `pygridsynth` with a total operator-norm synthesis budget divided over
  their occurrences. Parameters are never rounded to convenient angles.
- Small lowering checks compare complete numerical operators after reloading
  both artifacts. Larger instances sample basis inputs or retain only the local
  synthesis bound. That bound alone does not verify the entire adaptation.
- QuiZX checks the imported graph against the lowered input on sampled adaptive
  branches. Timeouts and inconclusive checks remain distinct from mismatches.
- Retained IR is compiled at **d=25**, with compiler validation and exact binary
  round trips. Small VM checks run separately at **d=3**, with ideal T sources,
  ideal input preparation, and boundary capture. Every joint Pauli expectation
  is compared with the lowered circuit; the original MQT state is checked within
  its synthesis budget.
- Numerical and sampled checks are not formal exhaustive channel proofs.
  Compilation and a chosen code distance do not establish fault tolerance or
  cultivation acceptance. Upstream routing, endpoint placement, fixed-frame,
  and patch-lifetime restrictions still apply.
- T counts alone establish neither a non-Clifford channel nor a nonstabilizer
  output. Small sources record numerical classifications; routed resource counts
  reveal upstream cancellation. VM rank limits and search exhaustion are recorded
  as inconclusive checks rather than output mismatches.
