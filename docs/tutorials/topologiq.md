# Compile a quantum algorithm through Topologiq

The optional Python adapter imports [Topologiq](https://github.com/tqec/topologiq)
layouts into Bloq without changing their routed geometry:

```text
Logical circuit → Clifford+T OpenQASM 2 → Topologiq → BlockGraph → Bloq IR → VM
```

## Install

Use a Bloq source checkout and Python 3.11–3.14. Topologiq requires the pinned
checkout and routing patch below; base Bloq does not require these dependencies.

```sh
uv sync --project bloq_py --locked --group dev --group interop --inexact
git clone https://github.com/tqec/topologiq ../topologiq
git -C ../topologiq checkout d010bd3e63686c93c12c922a84a944a3386ac785
git -C ../topologiq apply "$PWD/bloq_py/examples/topologiq.patch"
uv pip install --python bloq_py/.venv/bin/python --no-deps --editable ../topologiq
```

The patch keeps Hadamard transitions on temporal pipes, preserves pending
temporal exits, and fixes zero-valued routing bounds. Subsequent commands use `--no-sync`
to retain the separately installed upstream checkout. Use `uv sync --inexact`
when updating the environment.

## Import OpenQASM

```python
import bloq
from bloq.interop.topologiq import from_qasm

imported = from_qasm('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
tdg q[0];
''')
imported.graph.save("layout.blog")
program = bloq.compile(imported.graph, distance=3, prepare_t_with_mpps=True, validate=True)
program.save("program.bloq")
report = bloq.run_noiseless(program, shots=3, input_state="zero", include_signature=True)
```

`TopologiqImport` retains source QASM and phase signs, the upstream-node position
map, routing options, and input/output ports in source-qubit order. Match
`program.logical_outputs()` by `port` when interpreting VM outputs, whose order
is coordinate-based. `TopologiqImportError` includes a diagnostic `code` and
available upstream `nodes`; native compilation errors remain authoritative.
Use a nonzero layout seed; the pinned upstream treats zero as unseeded.

## Run the examples

Four-qubit phase estimation uses two counting and two target qubits to estimate
the eigenphases of `U = S(target0) CZ(target0, target1)`. Its seven T/T† gates
produce a coherent non-stabilizer state:

```sh
uv run --project bloq_py --no-sync python bloq_py/examples/topologiq_qpe.py \
    --shots 3 --output-dir target/topologiq-qpe
```

The broader corpus includes Grover search, QAOA MaxCut on three and four qubits,
three- and five-step quantum walks, amplitude estimation, modular addition,
a variational ansatz, and Trotter dynamics:

```sh
uv run --project bloq_py --no-sync python bloq_py/examples/topologiq_algorithms.py \
    --shots 3 --timeout 90 --output-dir target/topologiq-algorithms
```

Select a subset with `--candidates qpe4 grover3 qaoa3 walk3`. The sweep retains
failures, continues to later candidates, and exits nonzero if any candidate
fails or times out. The variational example evaluates one ansatz without an
optimizer. The dynamics oracle checks the stated Trotter approximation.

The QFT example synthesizes rotations before import. A two-qubit smoke test
executes quickly; the four-qubit example has 273 T/T† gates and is intended for
synthesis checks:

```sh
uv run --project bloq_py --no-sync python bloq_py/examples/topologiq_qft.py \
    --qubits 2 --shots 3 --output-dir target/topologiq-qft
```

All examples require an empty output directory and retain source/synthesized
QASM, layout and viewer, validated IR, and JSON results. `--synthesis-only` and
`--compile-only` stop at those stages. Independent matrix oracles check source
conversion; every executed shot checks all joint output Pauli expectations.
QFT execution checks the synthesized circuit and separately reports its
approximation error against the Fourier transform.

## Supported behavior and limits

- Sources are static unitary OpenQASM 2 circuits with Clifford gates and
  Z-axis T/T† phases. Rotation parameters must be exact rational multiples of
  `pi`; synthesize other angles first. Measurements, resets, classical control,
  unsupported phases, and unproved gadget frames are rejected.
- The adapter preserves signed T/T† controls and verifies routed provenance.
  It does not transport complete upstream Pauli webs into Bloq. Bloq derives
  its own [corrected readouts](../theory/correlation-surfaces.md#step-3-causal-readouts).
- Spatial and reversed temporal ports use Bloq's ideal boundary adapters.
  Bell outcome signs enter controls and output frames without postselection.
  [Port constructions](../circuit-constructions.md#port) describe their semantics.
- Upstream routing success does not guarantee native executability. The current
  adder layout fails with `LiveQubitReuse`; the Trotter example exhausts the
  temporal-only H routing budget. Bloq rejects incompatible results without rerouting
  or adding mixed-axis spatial Hadamard walls.
- VM examples use ideal T-source proxies, ideal decoding, and ideal boundary
  capture. They verify noiseless compiled circuits, not cultivation acceptance,
  hardware timing, or fault-tolerant distance. The
  [noiseless executor](../backends/vm.md#check-a-physical-output-state-without-timing)
  documents these boundaries.

Run adapter checks with `uv run --project bloq_py --no-sync pytest
bloq_py/tests/interop` and native signed-channel checks with `cargo test -p
bloq_vm --test interop_topologiq --locked`. Regenerate native fixtures with
`bloq_py/tests/interop/export_topologiq_fixtures.py --upstream-root ../topologiq`.
