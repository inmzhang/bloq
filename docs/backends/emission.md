# Backend Emission

Compilation produces Bloq IR. An **emitter** translates that program into what
a particular backend executes. Bloq provides:

- a **static** emitter that writes one Stim circuit for a program without
  runtime control, and
- the **Bloq VM**, a simple verification simulator for adaptive choices,
  mock decoder queries, retries, and waiting.

```{mermaid}
flowchart LR
  ir["Bloq IR"] --> stim["Stim emitter"] --> circuit["Stim circuit<br/>sampling, DEM"]
  ir --> vm["VM lowering"] --> run["Verification simulator<br/>mock decoder + timing model"]
  ir --> custom["Custom dynamic backend"] --> hw["Device controller + decoder"]
```

The VM checks compiled programs. Practical execution requires a separate
backend with device-specific instructions, a real decoder, and calibrated
timing.

## Stim

Stim emits static Clifford block-graph compilations and selected Clifford
proxies of non-Clifford graphs. It supports linear measurement parities and
record-controlled Pauli corrections.

Compile and emit a gallery CNOT:

::::{md-tab-set}
:::{md-tab-item} Python
```python
import bloq

program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
circuit = bloq.emit_stim(program)
circuit.to_file("cnot.stim")
noisy = bloq.emit_stim(program, noise=0.001)
```
:::
:::{md-tab-item} Rust
```rust
use bloq::prelude::*;
use bloq::stim::emit_bloq_stim;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let program = compile(&GalleryItem::CNOT.build(), 3)?;
    std::fs::write("cnot.stim", emit_bloq_stim(&program)?)?;
    Ok(())
}
```
:::
:::{md-tab-item} CLI
```sh
bloq compile --gallery cnot -d 3 --backend ir-text -o cnot.bloqir
bloq emit cnot.bloqir -o cnot.stim
```
:::
::::

The circuit contains qubit coordinates, physical gates and measurements,
detectors, and logical observables. `noise=p` applies a uniform circuit noise
model. The [Logical CNOT tutorial](../tutorials/cnot.md#sample-and-decode-with-noise)
walks through sampling, detector error models, and decoding.

The emitter builds the circuit in five steps:

1. **Plan each node.** For every quantum node, instantiate its templates at
   their offsets, merge them into one node-local circuit, and attach the node's
   detectors and bundle uses.
2. **Annotate noise.** When `noise=p` is requested, apply the noise model to
   each merged node circuit. Add supported gate and reset errors, measurement
   flips, and idle errors on the node's own qubits. This happens before
   concatenation, so other nodes do not introduce extra idle noise.
3. **Order the nodes.** Visit nodes in `deterministic_emit_order()`, a
   topological order of the dependency graph.
4. **Concatenate.** Append each node's circuit in that order. Resolve every
   detector and observable to Stim `rec[-k]` references in one global
   measurement record.
5. **Write observables.** Each complete `Observable` becomes an
   `OBSERVABLE_INCLUDE` of its raw parity. Linear classical feedback folds
   into these parities.

:::{admonition} Segmented and aligned emission
:class: important

Node circuits are concatenated without aligning `TICK`s. This preserves
simulation results when all noise is explicitly annotated in the Stim circuit
before concatenation, as in Bloq's per-node noise model. `TICK` counts are not
hardware timings, and external idle-noise models may over-count errors.

Segmented emission stores a shared header and node chunks with one global
measurement record. Sample the assembled circuit.

Aligned emission (`align_moments=True`) merges compatible moments in noiseless
whole-program emission. It adds no memory rounds or hardware durations.
:::

## Bloq VM

Bloq VM simulates the execution of compiled Clifford and non-Clifford programs
for verification, without physical noise by default. It models adaptive
measurements, resource retries, mock decoder queries, and QEC during waits to
check control flow and physical states. Circuit and idle noise are optional.

Lower the IR once, then run verification shots:

::::{md-tab-set}
:::{md-tab-item} Python
```python
import bloq

ir = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
program = bloq.lower_vm(ir, decoder_latency_rounds=3)
result = program.run(seed=17, input_state="plus")
if not result.discarded:
    print(result.finished_at, result.logical_bloch())
```
:::
:::{md-tab-item} Rust
```rust
use bloq::prelude::*;
use bloq::vm::{LoweringConfig, lower};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = compile(&GalleryItem::T.build(), 3)?;
    let options = LoweringConfig { decoder_latency_rounds: 3, ..Default::default() };
    let program = lower(&ir, &options)?;
    let result = program.run(options.runtime_config(17))?;
    println!("discarded: {}", result.artifact.discarded);
    Ok(())
}
```
:::
::::

Rust applications enable the VM with `cargo add bloq@=0.1.1 --features vm`.

Lowering runs once per program. It resolves the IR into the VM's internal
instruction stream: dense task, bit, qubit, and record
ids, physical gate streams, and structured control. Each task records its
dependencies, earliest release time, physical qubits, and nominal duration.

| Bloq IR | VM instruction | Runtime behavior |
| --- | --- | --- |
| Quantum node | `Quantum` | Runs the alternative selected by its guard values |
| `MemoryPadding` node | `MemoryRounds` | Runs a fixed number of QEC rounds |
| Join or retry output seam | `WaitFor` | Runs QEC rounds until its producers are ready |
| Unindexed `Observable` | Parity and binding tasks | Shared recipe fragment; no decoder request |
| Indexed `Observable` | `Observable` + `Decode` | One solve exposes `Corrected` parity and the `Flip` prediction |
| `Compute` | `Eval` | Boolean bytecode over bit registers |
| `Discard` | `Discard` | Rejects the shot |
| `RepeatUntilSuccess` | `Rus`, `SignalReady` | Retries the body and publishes the accepted resource |

The runtime then advances one causal clock. A task starts when its
dependencies are complete, its classical inputs are available, and its qubits
are free. Measurements go to the simulator, decoder queries go to a mock
streaming decoder, and waiting patches keep running QEC rounds.
[Synchronization](synchronization.md) explains these timing rules.

:::{important}
The default decoder is a stochastic policy model, not a syndrome decoder. VM
runs test control behavior and timing. They do not estimate a calibrated
logical error rate.
:::

## Custom Backend

A custom backend consumes Bloq IR and implements its own device integration.
The VM's instruction JSON and traces support inspection and verification.
They are not hardware-controller instructions.

| Requirement | A practical backend must provide |
| --- | --- |
| Physical operations | Device-specific gates, measurements, and qubit mapping |
| Decoder decisions | A real decoder connected to live measurement results |
| Synchronization | Calibrated device timing, classical latency, and QEC during waits |

```{toctree}
:hidden:
:maxdepth: 1

VM Verification <vm>
```
