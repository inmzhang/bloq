# Runnable documentation examples

These maintained examples accompany the [homepage](../index.md), [Quick Start](../getting-started/quickstart.md), [IR](../backends/ir.md),
[Logical CNOT](../tutorials/cnot.md), and [THTH](../backends/thth.md) tutorials,
including runtime checks. Run examples in an empty working
directory because they write the artifacts described below.

## Python

Install the released packages:

```sh
pip install "bloq-py==0.1.1"
```

Run the downloaded scripts:

```sh
python quickstart.py
python ir_roundtrip.py
python stim_memory.py
python vm_intro.py
python thth.py
python t_gate.py
```

`quickstart.py` loads the gallery CNOT and T gate at distance 3, saves their
Bloq IR, emits the static CNOT to Stim, and checks exchange round-trips and
the static/dynamic distinction. The full captured outputs (`quickstart-*`)
appear in scrollable panels in Quick Start. Its graph images are rendered with
Bloq Editor, using azimuth 45 for CNOT, 315 for T, and elevation 35.264,
with ports visible.

Save the homepage's BLOG definition as `t-gate.blog` alongside these examples.
`t_gate.py` loads that graph, compiles at distance 11, writes `t-gate.bloqir`,
prints a structural summary, and checks that the summary survives an IR round-trip.
The matching CLI example is `sh t_gate.sh`. Its output and the Python/Rust
output are recorded in `t_gate_stats.txt` and displayed on the homepage.

`ir_roundtrip.py` writes `memory.bloqir` and `memory.bloq`, performs explicit IR
validation, and checks both codec round-trips. `stim_memory.py` writes
`memory.stim`, checks 32 noiseless detector shots, and builds a noisy detector
error model. `vm_intro.py` runs three CNOT shots and saves instructions and the
last trace. `thth.py` checks the exact corrected output and writes instruction
JSON, trace JSON, a timeline CSV, and an illustrative controller envelope.

`tutorial_t_cultivation.py` calibrates a distance-11 T source with Clifft and
GAP-enabled PyMatching. Follow the [installation instructions](../tutorials/t-cultivation.md#cultivated-state-calibration),
then run `uv run python tutorial_t_cultivation.py --shots 64` for a small check.
The default run uses 10,000 attempts and reports each GAP threshold separately.

`tutorial_t_gate_simulation.py` performs offline decoding and path-based
postselection on the retained distance-9 T gate circuits. Download its ZIP
bundle and install the dependencies from the [Logical T Gate tutorial](../tutorials/t-gate.md#simulation).
Use `--shots 16` for a small check of all six inputs and the Choi case.

The CSV is an event inventory: memory intervals can enclose physical-moment
intervals. The controller envelope is an integration example, not a device
driver. Neither should be interpreted as a ready-to-run hardware schedule.

## Rust

For a new application, use `cargo add bloq@=0.1.1`; add `--features vm`
for both VM examples.
Copy the selected `rust/*.rs` file to `src/main.rs` and run `cargo run`.

To check the maintained source-distribution examples against their matching
checkout, use:

```sh
cargo run --locked --manifest-path docs/examples/Cargo.toml --bin quickstart
cargo run --locked --manifest-path docs/examples/Cargo.toml --bin ir-roundtrip
cargo run --locked --manifest-path docs/examples/Cargo.toml --bin t-gate
cargo run --locked --manifest-path docs/examples/Cargo.toml --bin stim-memory
cargo run --locked --release --manifest-path docs/examples/Cargo.toml --bin vm-intro
cargo run --locked --release --manifest-path docs/examples/Cargo.toml --bin thth
```

The example package uses the checkout's facade through a relative dependency.
The lockfile fixes its resolved dependencies. Rust IR and THTH examples check
the same invariants as Python. The Rust Stim example emits circuit files;
sampling them is demonstrated with the public Python Stim API in the tutorial.
