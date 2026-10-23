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

`Strategy::SpiderLs` uses full ZX reduction with a recorded conservative
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

`LayoutOptions` exposes numerical placement and scheduling controls. By default,
patch locations form a near-square 2D grid with cell-aware orthogonal routing.
Set `grid.columns` to a positive integer to choose the maximum locations per row;
zero resolves to the ceiling of the square root of the allocated locations.
Two or fewer locations can naturally occupy a single row. `grid: None` explicitly
selects the legacy row router.
The grid reserves local space for routing outlets and T/Y resources,
including future patch lifetimes and untouched wires. Routes can detour around
busy cells or wait for their release. The final footprint includes used geometry,
not the entire search canvas. `Layout.grid` records the resolved numerical
settings, `Layout.patch_locations` lists allocated homes, and
`Layout.routing_bounds` records the inclusive routing canvas. Allocated homes
include spare sites and untouched wires; they are distinct from occupied cells.

`extra_patches` adds up to the requested number of locations beyond the plan's
peak live frontier, limited by the number of preparations that can use them.
With a positive budget, new lifetimes choose the free location available earliest.
The requested budget applies to every candidate; the compiler does not silently
replace it with a zero-budget candidate. Extra locations can reduce waiting but
do not guarantee shorter layouts. They do not bound resource cells or footprint.

For T gadgets the grid also tries a placement using the earlier available
compatible face (`±X` or `±Y` according to the patch frame). Equal times prefer
an already occupied column, then the original positive face. Source, junction
and adaptive-cap lifetimes remain reserved together. The physical occupancy
scheduler orders teardown and initialization when resources reuse qubits.
Y/S resources keep their existing placement. The original fixed-face layout
remains a candidate and wins equal-cost ties.

Data patches stay at their allocated homes. CZ routes connect to the compatible
boundary in either Hadamard frame, without emitting `PatchRotation` blocks.
An exact ZX color-change pass moves spatial Hadamards onto temporal connections,
then cancels even Hadamard parity through uninterrupted temporal memory. It
preserves cube heights, occupied cells, and storage rounds. Resource and action
sites retain their frames; cancellation does not cross an interaction or an
adaptive measurement. Spatial parity conflicts are errors. One deterministic
component sweep reduces the remaining temporal transitions without claiming a
global minimum.
Complete, source-validated candidates use one deterministic ordering:
footprint times temporal extent, then footprint, then blocks.
There is no public area/depth objective selector
and no optimality guarantee. `Layout::compile` still
performs the full IR audit.
This bounded search tries at most two embeddings per plan when T gadgets are
present; physical lowering still runs only on the selected layout.

| Control | Default | Meaning |
| --- | --- | --- |
| `grid` | `Some(GridOptions::default())` | 2D placement and cell-aware routing |
| `grid.columns` | 0 (automatic) | Maximum locations per row; automatic resolves from allocated locations |
| `grid.spacing` | 4 | Pitch on both axes in graph cells; must be at least 4 for the Bloq gadgets |
| `grid.routing_margin` | 2 | Positive extra routing canvas around the patch islands |
| `grid.max_route_expansions` | 100,000 | Expanded A* states per route search; exhaustion is an error |
| `measurement_batch_size` | `None` | Split causal layers into ordered batches; smaller batches can reduce the live frontier, but this is not an exact ancilla cap |
| `extra_patches` | 0 | Additional reusable patch locations beyond the peak live frontier |
| `spider_reduction` | `Auto` | Full simplification with conservative fallback, `Conservative` only, or strict `Full` without fallback |
| `topology.fusion` | `true` | Enable bounded fusion candidates; disabling retains graph normalization |
| `topology.block_rows` | 20 | Source ZX rows per fusion partition, unrelated to physical grid rows |
| `max_temporal_extent` | `None` | Failure limit in graph cells, including boundaries; not an execution-time target |
| `max_blocks` | 1,000,000 | Failure limit on emitted graph nodes |

The reduction control applies to SpiderLS; fusion controls apply to TopoLS.
Every reduction recomputes and requires causal XY gflow. Conservative
simplification is not a claim of universal flow preservation. These deterministic
native planners have no upstream MCTS, stochastic seed, or CZ-star controls.

```rust
use bloq_interop::{GridOptions, LayoutOptions, layout_qasm};

let options = LayoutOptions {
    grid: Some(GridOptions { columns: 2, ..Default::default() }),
    measurement_batch_size: Some(2),
    extra_patches: 4,
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
# Two columns, four extra patch locations, batches of two measurements:
target/release/examples/compile_qasm circuit.qasm spiderls new-grid-directory 25 --extra-patches 4 --columns 2 --batch 2
# A wider grid, larger routing canvas, and explicit topology controls:
target/release/examples/compile_qasm circuit.qasm topols new-wide-directory 25 --columns 4 --routing-margin 3 --fusion off --block-rows 10
```

The output includes BLOG, validated binary IR, persisted-file round-trip
checks, compilation times, separate block/column/footprint/time metrics, and
all requested and resolved configuration values and the reduction actually
selected. The result directory must not already exist.
The example compiles the persisted BLOG after exact canonical checks, including
after flattening; full source analysis and IR validation run on that graph.
`matched_pipeline_seconds` covers source-file reading through the validated IR
round trip, before result metadata. `pipeline_seconds` includes that metadata
and starts after the source-file read.
Measure process wall time and peak RSS with an external isolated runner under
the same CPU, memory, and timeout limits for both implementations. Column and
footprint comparisons must consistently include extended blocks' reserved cells;
`anchor_columns` separately supports comparisons with older anchor-only reports.

Migration: the unreleased `LayoutTradeoff` enum, `LayoutOptions.tradeoff`, and
`--objective` CLI flag have been removed. Set numerical budgets and grid dimensions
directly. Omitted grid settings now use automatic 2D placement rather than a row.

## Local API documentation

From a Cargo project that depends on `bloq`, build its enabled API with:

```sh
cargo doc -p bloq --no-deps --open
```

Enable the features you use in that project's dependency declaration before
building the docs. In the Bloq source tree, `just docs bloq` builds the facade
reference with its default features. To include every optional API from the
source tree, run `cargo doc -p bloq --all-features --no-deps --open`.
