# VM Verification

The Bloq VM is a simple tool for verifying compiled physical IR. It simulates
non-Clifford gates, classical choices, resource retries, and causal timing
with `ticit`. Its decoder and timing models are for testing, not practical
execution. Compile and lower once, then reuse the simulation across shots.

The [T–H–T–H example](thth.md) combines these features in one small computation.

## Lower once, run many shots

Python's `bloq` package includes the VM. Rust applications enable it with
`cargo add bloq@=0.1.1 --features vm`.

::::{md-tab-set}
:::{md-tab-item} Python

```{literalinclude} ../examples/vm_intro.py
:language: python
```

:::
:::{md-tab-item} Rust

```{literalinclude} ../examples/rust/vm_intro.rs
:language: rust
```

:::
::::

Run the [Python example](../examples/vm_intro.py) with `python vm_intro.py`, or
use `cargo run --locked --release --manifest-path docs/examples/Cargo.toml --bin vm-intro`
in the source distribution. The three shots write the reusable instructions
to `cnot.instructions.json` and the last shot's trace to `cnot.trace.json`.

Each run starts a new shot. A discarded shot has no logical output. For an
accepted shot, the X/Y/Z expectations include the output's Pauli frame correction.

:::{important}
The default decoder is a stochastic policy model, not a syndrome decoder.
Its acceptance and residual-accuracy settings help test control behavior.
They do not establish a calibrated logical error rate.
:::

## Set simulation timing and noise

| Setting | Meaning | Default |
| --- | --- | --- |
| `gate_duration` | Duration of one nonempty physical moment | `1.0` |
| `decoder_latency_rounds` | Decoder latency in local memory rounds | `10` |
| Source release epochs | Earliest eligibility of factories, inputs, and Clifford sources | `0.0` |
| Uniform circuit noise | Noise applied after physical instances are merged | None |

Use the same time unit for releases and durations. `gate_duration` must
exceed the clock-comparison tolerance `1e-9`. This uniform-duration model does
not independently calibrate reset, measurement, and two-qubit gates.

Clifford sources, input arrivals, and factories can have separate release
times. Releases determine earliest eligibility; dependencies or occupied
qubits can delay execution.

Uniform circuit noise also supplies an idle error rate to the runtime.
Dynamic idle depolarization grows with wait duration, capped at probability
0.75. The idle error rate is measured per time unit.

## Readiness determines execution time

The VM advances one causal clock. Same-time completions settle before new
work starts. Independent moments on disjoint qubits can overlap; a join waits
for its participating patches, resources, and classical values.

| Behavior | Runtime consequence |
| --- | --- |
| Unavailable classical input | Dependent work waits, including expressions with an unselected arm |
| Decoder solve in progress | Readout-dependent work waits for the result |
| Live patch waiting | QEC continues in whole rounds; idle fills shorter residual gaps |
| Output frame pending | The output remains protected until its frame is ready |
| Failed resource attempt | Unissued work is canceled, then authored preparation is replayed |
| Accepted cultivation | Physical completion is followed by the configured GAP memory rounds |

Retries do not rewind physical time or add a redundant reset block. Failed
attempts remain in the trace but do not publish accepted record aliases.
Earlier factory measurements do not shorten the completion-based GAP hold.

## Inspect instructions and traces

| Artifact | Contains | Use |
| --- | --- | --- |
| Instruction JSON | Tasks, physical operations, dependencies, choices, releases, and logical boundaries | Inspect the lowered program |
| Execution trace | Simulated intervals, measurements, detectors, retries, decoder deadlines, waits, and frames | Inspect one simulated shot |
| Text or binary IR | The compiled program before VM lowering | Save and exchange programs |

Keep trace metadata with timing measurements. The JSON inspection layouts may
change with the software version; use the [IR codecs](ir.md#bloq-ir-format)
for program exchange. A completed trace describes one run, rather than a
schedule that can be replayed for every future measurement outcome.

## Separate modeling from verification

Physical verification uses ideal decoder decisions, supplied logical inputs,
and exact expectations. It executes explicit noise in the supplied circuit
but adds none itself. Source
verification with QuiZX is a separate logical check and does not execute IR.

Dynamic execution has work and attempt limits. Attempt exhaustion discards
the shot with its history; other execution failures return errors. Retry
regions require an isolated source; nested retries and reused output cuts are
unsupported. Guarded-stage lowering also has a quantum-variant cap.

See the [Python API](../api/python.rst) or
[Rust API](../api/rust.md) for configuration options.
