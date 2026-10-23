# Rust API Reference

The generated <a href="rust/bloq/index.html">Rust API reference</a> documents the
`bloq` facade, including item signatures, typed errors, and executable examples.
The website builds it from the same source as this documentation version. Use
this page to choose an entry point, then follow the generated reference for its
arguments and contracts.

## Namespaces

Start with `bloq::prelude::*` for graph authoring, compilation, and ordinary Stim
emission. Use explicit modules for specialized types. The facade re-exports the
workspace crates, so importing a type through `bloq::ir`, for example, does not
create a wrapper around `bloq_ir`.

| Namespace | Responsibility |
| --- | --- |
| `bloq::graph` | Source blocks, pipes, actions, module interfaces, BLOG I/O, and logical analysis |
| `bloq::compile` | Validated configuration, contexts, progress, template caches, and compiled objects |
| `bloq::ir` | Physical programs, graph traversal and edits, templates, regions, readouts, and codecs |
| `bloq::circuit` | Physical operations, coordinates, noise models, and Clifford flows |
| `bloq::stim` | Static emission, segments, and Stim/clifft text dialects |
| `bloq::utils` | Pauli algebra, geometry vocabulary, and shared supporting types |
| `bloq::vm` | Dynamic physical execution and verification; requires `vm` |

The [feature reference](../guide.md#cargo-feature-flags) lists the optional
capabilities. Ordinary graph authoring, compilation, IR access, and Stim text
emission require no features.

## Native circuit layouts

The separate `bloq-interop` crate (Rust import `bloq_interop`) constructs block
graphs from static Clifford+T circuits using QuiZX. `layout_qasm` accepts discrete
OpenQASM 2 gates; `layout` also accepts exact rational quarter-turn rotations in
a QuiZX `Circuit`. Neither entry point synthesizes approximate rotations.

`Strategy::SpiderLs` uses full ZX reduction with a recorded flow-preserving
fallback. `Strategy::TopoLs` uses bounded fusion and breadth-first topology
placement. It compares single-pass and fixed-point fusion before compilation.
Both use a causal XY-gflow scheduler and direct Bloq constructions.
These are native strategies inspired by the two compilers, with a shared
deterministic router; they do not reproduce upstream stochastic placement or
CZ-star grouping. Layouts retain source-ordered inputs and outputs, signed
phases, adaptive T measurements, and resource lifetime reservations.
Idle spans use native tall-memory blocks: this reduces graph nodes while
preserving occupied cells and syndrome rounds. The block budget and candidate
cost count graph nodes; occupied cells, footprint, and temporal extent remain
separate layout metrics.

`LayoutOptions::tradeoff` controls spatial reuse and candidate selection.
`LayoutTradeoff::Compact` is the default: reuse patch locations and choose fewer
blocks, then smaller footprint, then shorter temporal extent.
`LayoutTradeoff::Depth { extra_patches: N }` also tries placement with up to `N`
extra patch locations beyond the plan's peak live frontier. It assigns each new
lifetime to the free location with the earliest availability, and selects shorter
temporal extent, then smaller footprint, then fewer blocks. Compact placement
remains a candidate, so enabling depth mode cannot lengthen the selected layout.
The extra-patch budget bounds data lanes, not resource columns or total footprint.
This is a bounded heuristic: a larger budget need not improve time, and the fixed
routing corridor can still serialize operations.

```rust
use bloq_interop::{LayoutOptions, LayoutTradeoff, layout_qasm};

let options = LayoutOptions {
    tradeoff: LayoutTradeoff::Depth { extra_patches: 4 },
    ..Default::default()
};
let layout = layout_qasm("OPENQASM 2.0; qreg q[1]; t q[0];", &options)
    .expect("valid causal layout");
```

`Layout::compile` takes the normal `CompileConfig` and always performs the full
IR audit. Layout construction, successful compilation, logical branch checks,
and physical execution are separate claims. Small VM regressions use ideal
T-source proxies. Unsupported causal flows and exhausted budgets are errors.

To reproduce compilation measurements from a frozen synthesized circuit:

```sh
cargo build --release -p bloq-interop --example compile_qasm --locked
target/release/examples/compile_qasm circuit.qasm spiderls new-result-directory 25
# Depth mode with at most four extra patch locations:
target/release/examples/compile_qasm circuit.qasm spiderls new-depth-directory 25 4
```

The output includes BLOG, validated binary IR, persisted-file round-trip
checks, compilation times, separate block/column/footprint/time metrics, and
the reduction actually selected. The result directory must not already exist.
The example compiles the persisted BLOG after exact canonical checks, including
after flattening; full source analysis and IR validation run on that graph.
`matched_pipeline_seconds` covers source-file reading through the validated IR
round trip, before result metadata. `pipeline_seconds` includes that metadata
and starts after the source-file read.
Measure process wall time and peak RSS with an external isolated runner under
the same CPU, memory, and timeout limits for both implementations. Column and
footprint comparisons must consistently include extended blocks' reserved cells;
`anchor_columns` separately supports comparisons with older anchor-only reports.

## Local API documentation

From a Cargo project that depends on `bloq`, build its enabled API with:

```sh
cargo doc -p bloq --no-deps --open
```

Enable the features you use in that project's dependency declaration before
building the docs. In the Bloq source tree, `just docs bloq` builds the facade
reference with its default features. To include every optional API from the
source tree, run `cargo doc -p bloq --all-features --no-deps --open`.
