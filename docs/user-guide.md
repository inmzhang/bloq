# User Guide

Bloq takes a surface code layout of a logical computation and compiles it into
a physical program: the quantum circuit and the classical decisions needed to
run it. This guide follows that path from source to execution.

```{mermaid}
flowchart LR
  source["Block graph<br/>geometry + actions"] --> compile["Compilation<br/>logical correlations + physical blocks"]
  compile --> ir["Bloq IR<br/>operations + dependencies"]
  ir --> stim["Stim<br/>static circuit"]
  ir --> vm["VM<br/>verification simulator"]
```

The chapters build on each other, so read them in order the first time.

**1. Describe the computation**

[Preliminaries](theory/prerequisites.md) reviews the basic concepts the
rest of the guide relies on. [Existing Works](theory/existing-works.md) compares
Bloq with related compilers and tools. [Blocks and Pipes](graphs/concepts.md) introduces
the block graph: blocks placed in spacetime and joined by pipes.
[Actions](graphs/actions.md) adds named measurements and the decisions that
depend on them. [Modules](modules/index.md) packages a graph behind a reusable
quantum and classical interface. [BLOG Format](graphs/blog.md) shows how to
write and exchange the complete source as text.

**2. Understand compilation**

[Correlation Surfaces](theory/correlation-surfaces.md) explains how Bloq tracks
the signed Pauli relations behind every logical readout and output correction.
[Patch Alignment](patch-alignment.md) introduces the shared qubit lattice and
stabilizer placement conventions.
[Circuit Constructions](circuit-constructions.md) explains the fault-tolerant
physical circuit design for each block and pipe. [Compilation](theory/compilation.md) ties
these ingredients together and produces Bloq IR.

**3. Inspect and execute the result**

[Bloq IR](backends/ir.md) is the main output of compilation and the input to
downstream backend compilers. It keeps circuit templates and a dependency graph
together. Bloq provides convenient APIs to visit and edit the IR, inspect and
draw it, and save it as text or binary.

[Backend Emission](backends/emission.md) translates the IR for execution. Bloq
provides a default [Stim emitter](backends/emission.md#stim) for static Clifford
computations and a simple [verification VM](backends/vm.md) for checking
dynamic compilation outputs. Practical execution needs a custom backend:

| Program | Execution path | What it preserves |
| --- | --- | --- |
| Fixed Clifford computation | [Stim emission](backends/emission.md#stim) | Physical gates, measurement records, detectors, and observables |
| Verify an adaptive measurement or retry protocol | [VM verification](backends/vm.md) | Simulated choices, records, retries, and timing |

[Synchronization](backends/synchronization.md) covers the explicit
synchronization across logical patches and instructions needed for real hardware
emulation. It explains how resource preparation, decoding, and classical
dependencies determine when instructions can execute and where live patches
must wait.

**4. Walk through real examples**

Building on the topics above, a series of tutorials provides real examples of
compiling and simulating fault-tolerant surface code computations.

```{toctree}
:hidden:
:maxdepth: 1

Overview <self>
theory/prerequisites
theory/existing-works
graphs/concepts
graphs/actions
modules/index
graphs/blog
theory/correlation-surfaces
patch-alignment
circuit-constructions
theory/compilation
backends/ir
backends/emission
backends/synchronization
```

## Tutorials

| Tutorial | Focus |
| --- | --- |
| [Logical CNOT](tutorials/cnot.md) | API authoring, correlations, filling, Stim emission, and decoding |
| [T State Cultivation](tutorials/t-cultivation.md) | Preparing a T resource state and its acceptance–quality tradeoff |
| [Logical T Gate](tutorials/t-gate.md) | Gate teleportation, adaptive measurement, and retries |
| [MQT Bench Compiler Comparisons](tutorials/mqt-bench.md) | Faithful benchmark lowering, validated IR, and sampled execution |
| [Compile Phase Estimation through Topologiq](tutorials/topologiq.md) | Logical QASM, upstream layout, and native non-Clifford controls |
| [Compile Algorithms through TopoLS](tutorials/topols.md) | Signed magic resources, addition, and spin dynamics |
| [Compile Algorithms through SpiderLS](tutorials/spiderls.md) | Routed ZX schedules, quantum walks, and QAOA |
| [8T-1CCZ Distillation Factory](tutorials/ccz-factory.md) | Resource errors, circuit noise, and factory response |
| [Emulate Hardware Synchronization](backends/thth.md) | Backend timing controls, decoder waits, and live-patch protection with THTH |

```{toctree}
:hidden:
:maxdepth: 1

tutorials/cnot
tutorials/t-cultivation
tutorials/t-gate
tutorials/mqt-bench
tutorials/topologiq
tutorials/topols
tutorials/spiderls
tutorials/ccz-factory
backends/thth
```
