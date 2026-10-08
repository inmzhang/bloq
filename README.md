<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logos/bloq-logo-dark.png">
  <img src="docs/assets/logos/bloq-logo-light.png" alt="Bloq logo: interlocking coral b and blue q above the Bloq wordmark" width="180">
</picture>

# Bloq

<!-- bloq-intro-start -->

Turning a logical algorithm into an executable surface code program requires
several compilation stages:

![Compilation pipeline from quantum algorithm to circuit, ZX diagram, BlockGraph, physical program, and backend target. Bloq compiles the BlockGraph into the physical program.](docs/assets/compilation-toolchain.svg)

Bloq compiles surface code spacetime layouts into physical programs with the
circuit, decoding, and classical-control information needed for simulation and
hardware execution. It connects the logical layout to a physical instruction
stream that combines quantum operations with classical control.

The input is a `BlockGraph`: a 3D spacetime layout composed of primitive logical
blocks and actions such as measurement-controlled branches and logical Pauli
feedback. The same graph can own reusable definitions, placed instances, and
their quantum and classical interfaces. Loading and saving retain that
hierarchy; one compile API handles both local and composed graphs.
The output is **Bloq IR**, which records physical gates, detectors, logical
observables, classical control, and source provenance.

Bloq provides Rust and Python APIs, a CLI, and an interactive editor.

<!-- bloq-intro-end -->

Start with the [documentation](https://bloqec.com/docs/dev/) or
[open the browser editor](https://bloqec.com/editor/).

## Feature highlights

- **Surface Code Building Blocks** Compose cubes for memory and lattice surgery,
  patch rotations, walking patches (sliding or gliding), Y-basis initialization
  and measurement, logical Hadamards, and T-state cultivation.
- **Dynamic Quantum-Classical Computation** Express logical measurements,
  classical bindings, Pauli feedback, measurement-controlled branch selection,
  repeat-until-success regions, and shot discard.
- **Reusable Modules** Define logical components such as AND, MAJ, and UMA
  independently, then compose and link them into larger computations such as
  adders, retaining their quantum and classical interfaces.
- **Correlation Surfaces** A shared logical model tracks measurement
  signs, byproduct operators, and authored feedback across modules and reachable
  branches, deriving corrected readouts and output Pauli frames together.
- **Physical Circuits And Decoding Information** Block and pipe constructions
  have physical circuit realizations with detector and logical-observable
  information for downstream decoding and correction.
- **An Editable Physical IR** Rust and Python APIs expose physical operations,
  control dependencies, and provenance. Use the IR for backend
  emission, causal timing and synchronization, and memory-round padding.

## Relationship with TQEC

Bloq grew out of work on [TQEC](https://github.com/tqec/tqec) and is a substantial
Rust rewrite and extension of its block-graph compilation approach. Its design
focuses on reusable modules, explicit physical-program semantics, and dynamic
classical control.

## Installation

Install the CLI with Cargo:

```sh
cargo install bloq-cli --version 0.1.1 --locked
```

Or install the CLI in an isolated Python environment with `uv`
(Python 3.10 or newer):

```sh
uv tool install "bloq-py==0.1.1"
```

For a Rust project:

```sh
cargo add bloq@=0.1.1
```

For a Python project (Python 3.10 or newer):

```sh
pip install "bloq-py==0.1.1"
```

The Python distribution is `bloq-py`; the import and CLI remain `bloq`.

See [Contributing](CONTRIBUTING.md) and the [installation guide](https://bloqec.com/docs/dev/getting-started/installation.html)
for source builds, development setup, and editor prerequisites.

## Usage

This example implements a logical T gate. The purple block represents
T-state cultivation, then lattice surgery implements the MZZ parity measurement
between the T state and the data qubit. The MZZ outcome selects whether we measure
the consumed T state in the X or Y basis.

[Explore the interactive T-gate block graph](https://bloqec.com/docs/dev/getting-started/quickstart.html#dynamic-logical-t-gate).

Save the following BLOG source as `t-gate.blog`:

```blog
BLOG 1.0

module main {
  in q_in: data = 0
  out q_out: data = 2

  0: Port [0, 0, 0]
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2]
  3: T [1, 0, 0]
  4: XZX [1, 0, 1]
  5: YX [1, 0, 2]
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 0] -> +Z
  [1, 0, 2] -> -Z

  mzz = measure 1 -> +X
  resolve 5 if mzz
}
```

The examples below compile this layout at code distance 11 and save Bloq IR,
which retains its dynamic control flow.

### CLI

```sh
bloq compile t-gate.blog -d 11 --backend ir-text -o t-gate.bloqir
bloq validate t-gate.bloqir
bloq view t-gate.blog --html
```

### Rust

```rust,no_run
use bloq::prelude::*;

fn main() -> Result {
    let graph = BlockGraph::load("t-gate.blog")?;
    let program = compile(&graph, 11)?;
    std::fs::write("t-gate.bloqir", program.to_text())?;
    Ok(())
}
```

### Python

```python
import bloq

graph = bloq.BlockGraph.load("t-gate.blog")
program = bloq.compile(graph, distance=11)
program.save("t-gate.bloqir")
```

Compiled Bloq IR records physical gates, detectors, logical observables,
classical control, and source provenance. This T-gate program also retains
basis selection, output Pauli frames, and the cultivation retry region.

[![Compiled T-gate Bloq IR showing quantum nodes, classical dependencies, and the nested repeat-until-success cultivation region.](docs/assets/t-gate-program.svg)](docs/assets/t-gate-program.svg)

## Documentation

The documentation is hosted on the [Bloq website](https://bloqec.com/). Some quick links are shown below:

- [Getting started](https://bloqec.com/docs/dev/getting-started/quickstart.html)
- [User Guide](https://bloqec.com/docs/dev/user-guide.html)
- [CLI reference](https://bloqec.com/docs/dev/reference/cli.html)
- [Python API](https://bloqec.com/docs/dev/api/python.html)
- [Rust API](https://bloqec.com/docs/dev/api/rust.html)
- [Browser editor](https://bloqec.com/editor/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and verification.

## Citation

If you use Bloq in your work, please cite the software repository:

```bibtex
@misc{zhang2026bloq,
  author       = {Zhang, Yiming},
  title        = {{Bloq}: Compiling dynamic Clifford+T surface code computations to fault-tolerant programs.},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/inmzhang/bloq}
}
```

## AI Acknowledgement

Generative AI tools assist Bloq's software development and documentation,
including drafting and reviewing code, analyzing implementations, and editing
prose. Human contributors remain responsible for design, correctness,
validation, and release decisions. See the
[AI contribution policy](CONTRIBUTING.md#ai-assisted-contributions) for review
and disclosure requirements.

Licensed under [Apache-2.0](LICENSE).
