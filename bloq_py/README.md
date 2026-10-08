# bloq

CLI and Python bindings for the [bloq](https://github.com/inmzhang/bloq) workspace:
compile block graphs for fault-tolerant lattice-surgery quantum circuits,
emit [Stim](https://github.com/quantumlib/Stim) circuits, and verify dynamic
programs with a simple simulation VM.

Requires Python 3.10 or later. The PyPI distribution is `bloq-py`; the import
and CLI are `bloq`.

## Install

Install the CLI from a released wheel with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install "bloq-py==0.1.1"
bloq --help
bloq compile --gallery cnot -d 3 -o cnot.stim
```

For the Python API, use `uv add "bloq-py==0.1.1"` in your project or
`pip install "bloq-py==0.1.1"`.
Wheels include the same Rust CLI used by
`cargo install bloq-cli --version 0.1.1 --locked`. No Rust installation is needed
on platforms with a matching wheel. Source installs
require the workspace Rust toolchain.

## Install from source

From the repository root, with the workspace Rust toolchain installed:

```sh
uv tool install ./bloq_py    # CLI in an isolated environment
pip install ./bloq_py        # CLI and API in the current environment
```

Development recipes use `uv` and `just`:

```sh
just py-develop   # uv-managed editable install in bloq_py/.venv
just py-test      # pytest suite
just py-build     # release wheel
```

## Quick example

```python
import bloq

graph = bloq.GalleryItem.CNOT.load()
program = bloq.compile(graph, distance=3)     # Bloq IR

circuit = bloq.emit_stim(program)           # returns stim.Circuit
circuit.to_file("cnot.stim")
assert circuit == bloq.compile_to_stim(graph, 3)

vm = bloq.lower_vm(program)
shot = vm.run(seed=94)
print(shot.trace)
```

Compiler options such as `prepare_t_with_mpps` and `limits` are keyword-only.
`compile_to_stim(graph, noise=0.001)` handles the common noisy Stim workflow.
For an explicit full IR audit, use `compile(graph, validate=True)` or
`context.compile(graph, validate=True)`; both return the audited program and
apply the compilation's Boolean limits to the audit.
Programs, graphs, and Pauli strings support copying and pickling for
`multiprocessing` workers.

`BlockGraph.load(path)` reads `.blog` files and resolves relative module imports.
Like `from_text`, it retains module definitions, instances, and interfaces in
the graph. `save(path)` preserves that hierarchy. `module_names` lists the
definitions; `module(name)` returns an independent graph rooted at that
definition, with its reachable helpers and an executable `main` root. Local block
accessors inspect the owning definition. Use `flatten()` to assemble a separate
flat graph for geometry inspection. The same `compile` API accepts both forms.

Symbolic `Expr` and `ClassicalExpr` objects reject Python truth testing. Build
source expressions with `&`, `|`, `^`, and `~`. Use `program.classical_value(...)`
to evaluate IR nodes.

Dynamic quantum choices use `QuantumNode.guards`. Retry regions use
`RepeatUntilSuccess`. `program.selection_seams()` locates incoming
selection seams. Pass their `bloq.ir.MemoryRoundTarget.Edge` targets to
`program.insert_memory_rounds_batch(targets, rounds)` to add memory atomically.
Import detailed inspection types from `bloq.ir`, for example
`from bloq.ir import CircuitOp, RegionKind`. Custom backends use
`program.emission_plan(node, path=None, noise=0.001)` and
`emit_plan_stim(plan, layout, dialect="stim")`; noise is optional and dialects
are strings. See [Bloq IR](https://bloqec.com/docs/dev/backends/ir.html) for inspection and editing APIs.

VM decoding uses a mock policy model. Its traces do not estimate calibrated
logical error rates. Graph-level and physical verification remain Rust-only.
Python callers can check emitted Clifford circuits with the PyPI `stim` package.

Domain failures derive from `BloqError`. Python protocols and I/O retain
standard exceptions such as `IndexError`, `TypeError`, and `OSError`.

See the [Python guide](https://bloqec.com/docs/dev/api/python.html)
for the API and limits, and the [VM contract](https://bloqec.com/docs/dev/backends/vm.html)
for timing and decoding semantics.
