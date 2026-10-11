# Bloq IR

Bloq IR stores the compiled fault-tolerant physical circuits, their
placements, detector checks, readouts, and classical control. Backends consume
it without the source block graph. Templates, loops, and retries remain
structured.

This chapter introduces the data structure, its text and binary formats, and
the APIs for reading, querying, editing, and checking a program.

## Design in one picture

Bloq IR is a dependency graph over two shared pools:

```{mermaid}
flowchart LR
  pool["Template pool<br/>circuits + flows"] --> q["Quantum nodes<br/>placed instances"]
  bundles["Detector bundle pool<br/>shared checks"] --> q
  q -->|Order| c["Classical nodes<br/>readouts + logic"]
  c -->|Value| c2["Classical nodes"]
  c2 -->|Value guard| q2["Quantum nodes"]
  q -->|Quantum| q2
```

Edges record dependencies. Node ids and coordinates do not specify execution
times.

| Repeated work | IR representation |
| --- | --- |
| Fixed circuit repetitions | Named `REPEAT` body, repetition count, and loop-carried detector state |
| Dynamic retries | `RepeatUntilSuccess` region with a body and restart predicate |

Circuits therefore cost storage once per distinct template, not once per use.
A 128-bit controlled adder at distance 3 stores 11,372 quantum nodes over
90 templates. Classical definitions also share storage, while each invocation
retains its own bindings and activation.

Storing each body once eliminates duplicated code. Downstream compilers can
optimize the body once and consume the loop structure directly.

## Data structure

### Program

A `Bloq` program contains:

| Part | Contents |
| --- | --- |
| Metadata | Key–value pairs, such as `bloq_compile.code_distance` |
| Template pool | Circuit templates, indexed by `TemplateId` |
| Detector bundle pool | Shared detector recipes, indexed by `DetectorBundleId` |
| Logical inputs and outputs | Boundary operators at their owning instances |
| Top-level graph | Nodes, edges, and nested retry bodies |

### Templates and instances

A template holds a physical circuit on relative qubit coordinates and
everything derived from it once:

| Field | Meaning |
| --- | --- |
| `circuit` | Gates, measurements, `TICK`s, and named `REPEAT` bodies |
| `detectors` | Checks closed inside the template |
| `repeat_states` | Loop-carried detector state for `REPEAT` bodies |
| `restarts` | Post-selection parities for T-state retries |
| `boundary_flows` | Stabilizer flows the template leaves open for its neighbors |

An **instance** places a template at a qubit offset and records its source
block. Instances share circuit data and own separate measurement records.

| Record address | Example |
| --- | --- |
| Template-local measurement | `m2` |
| Measurement at a specific instance | `i3:m2` |

### Nodes

Every node has a kind and a **provenance** that links it back to the source.

| Kind | Variants | Purpose |
| --- | --- | --- |
| Quantum | — | Placed instances, cross-instance detectors, bundle uses, restarts, and guards |
| Classical | `Observable` | Readout recipe built from measurements and boundary operators |
| | `Compute` | Boolean expression over input slots |
| | `Discard` | Rejects the shot when its condition holds |
| Region | `RepeatUntilSuccess` | Retries its body until a restart predicate accepts |

| Provenance | Node it marks |
| --- | --- |
| `BlockComponent` | Quantum node for source blocks sharing physical qubits on a layer |
| `TemporalPipe` | Realignment layer of a temporal Hadamard pipe |
| `SpatialPortSubstitution` | Compiler-inserted replacement of a spatial Port |
| `MemoryPadding` | Memory rounds inserted after compilation, with their count |
| `Generator` | Readout for a correlation surface |
| `Action` | Lowered source action, such as a named measurement |
| `BranchSelector` | Resolved selector of a branch or selective cap |
| `OutputFrame` | X or Z frame bit for an output Port |

### Guards and activation

A **guard** is a Boolean condition that decides which work participates in an
execution. A measurement or `Compute` node supplies the bit through a `Value`
edge. Compound conditions, such as `NOT s` or `a AND b`, are computed before
they reach the guarded node.

| Control | When true | When false |
| --- | --- | --- |
| Quantum membership (`QuantumNode.guards`) | Includes the listed instances, detectors, bundle uses, and restart checks | Omits them. Skipped instances produce no gates or measurement records |
| Classical activation (`BloqNode.activation`) | Evaluates the node normally | Exports zero and no boundary bindings, without reading records, querying a decoder, or discarding the shot |
| Retry-region activation | Executes the retry body | Skips the body and exports zero with no boundary bindings |
| Quantum-edge guard | Enables the patch connection | Omits that connection |

Quantum members absent from every guard are common to all choices. Classical
nodes without activation always evaluate. Guard bits must be available before
the controlled work starts. An activation bit enables evaluation; it is not
XORed into the readout parity.

For example, a selective cap can choose between X and Z readout using `s`:

| `s` | Quantum membership | Classical activation |
| --- | --- | --- |
| False | `NOT s` enables the X-measurement instance | The X-readout fragment evaluates; the Z fragment contributes zero |
| True | `s` enables the Z-measurement instance | The Z-readout fragment evaluates; the X fragment contributes zero |

Each readout fragment uses the same condition as its measurement instance, so
it never reads records from an unselected alternative. An inactive fragment's
zero means no contribution, not a measured logical zero.

In text IR, `guard 0 i5` refers to **input slot 0**, whose Boolean value controls
instance `i5`. A classical node's `when v0` uses slot 0 for activation.

### Edges

| Edge | Payload | Meaning |
| --- | --- | --- |
| `Quantum` | Temporal pipes, padding templates, optional guard | A patch passes from one node to the next |
| `Value` | Input slot, output port, and role | `Corrected` or `Flip` feeds a Boolean input |
| `Compose` | Input slot and role | Includes raw parity and boundary bindings from an observable fragment |
| `Order` | — | The target waits for the source to finish |

Quantum edges retain one-round and looped memory templates for their seams.
[Synchronization](synchronization.md) uses them to insert memory
rounds without the source graph.

### Detectors, observables, and frames

**Detectors** check measurement parities at three levels:

| Stored in | Checks |
| --- | --- |
| Template | Records within one circuit |
| Quantum node | Records across template instances |
| Detector bundle | A shared recipe whose owner slots bind to concrete instances at each use |

**Observables** combine measurement parities with boundary operators:

| Form | Purpose | Decoder query |
| --- | --- | --- |
| Indexed `Observable` | Complete readout | One query shared by both output ports |
| Unindexed `Observable` | Reusable readout fragment | None |

A `Compose` edge adds a fragment's raw parity and boundary bindings to its
parent. It adds no separate decoder correction. Shared fragments keep their
own activation and instance owners. The optimizer merges compatible leaves
or inlines them into their parents.

```{mermaid}
flowchart LR
  fragments["Readout fragments"] -->|Compose| observable["Indexed Observable<br/>one decoder query"]
  observable --> corrected["Corrected<br/>raw parity XOR Flip"]
  observable --> flip["Flip<br/>decoder estimate"]
```

| Reference | Carries |
| --- | --- |
| `ValueRef(node, Corrected)` | Indexed observable's corrected bit, or an ordinary producer's Boolean result |
| `ValueRef(node, Flip)` | Indexed observable's decoder estimate |
| `Compose` edge | Raw parity and boundary bindings. Boolean `Value` edges carry no boundary bindings |

**Output frames** describe corrections for live logical outputs:

| Property | Representation |
| --- | --- |
| Frame bit | `Compute` node with `OutputFrame` provenance |
| X/Z pair per output Port | Returned by `output_frames()` |
| Exported result | Retained even when its node has no outgoing edges |

**T-state retries** use a body-local `Compute` node to form
`restart = Flip₀ OR Flip₁`. Either predicted flip triggers a retry.
Physical post-selection failures also trigger retries.

## Bloq IR Format

| Format | Extension | Use |
| --- | --- | --- |
| Text IR | `.bloqir` | Inspection, review, and diffs |
| Binary IR | `.bloq` | Compact program exchange |

### Text Bloq IR

Text IR starts with `BLOQIR 1` and lists the program's parts in order. Lines
starting with `#` and blank lines are ignored. Parse errors report one-based
line numbers. The excerpts below come from a compiled logical T gate.

**Header and boundaries.** Metadata, then logical inputs and outputs with
their owning instance and X/Z operators:

```text
BLOQIR 1
metadata bloq_compile.code_distance u64 3
metadata bloq_compile.convention string fixed-bulk
logical-input (0,0,0) instance i2 x X(1,3)*X(3,3)*X(5,3) z Z(3,1)*Z(3,3)*Z(3,5)
```

**Templates.** A circuit on relative coordinates, then its detectors, loop
states, and boundary flows. `REPEAT 2 b1` runs the named body `b1` twice:

```text
template t9 {
  circuit {
    R (2,0) (4,2) (2,4) (4,6)
    ...
    REPEAT 2 b1
  }
  body b1 { ... }
  detector body(b1) m8*s0 @ (2,0)
  loop(b1,s0) init m0 next m8
  flow Z(1,1)*Z(3,1) -> _ meas m0 center (2,0)
}
```

**Detector bundles.** Owner slots `o0`, `o1` are bound at each use:

```text
bundle b0 owners t2 t3 {
  detector o0:m0*o1:m0 @ (2,0)
}
```

**Graph.** Quantum nodes list instances, bundle uses, and guards. The
selective cap's node enables instance `i5` on guard slot 0 and `i6` on slot 1:

```text
graph {
  n2 quantum {
    instance i5 t5 @ (8,0)
    instance i6 t6 @ (8,0)
    use b3 i4 i5 @ (8,0)
    use b4 i4 i6 @ (8,0)
    guard 0 i5 b0
    guard 1 i6 b1
    from blocks (1,0,2)
  }
```

The T-state factory stores its body in a retry region. Its restart expression
ORs the two observable `Flip` outputs:

```text
  n4 rus in0 source n4 {
    body {
      n1 quantum { ... restart i1:m11*i1:m13 ... }
      n2 observable 0 measurements i1:m15*... operators i1 input ..., i1 output ...
      n3 observable 1 measurements i1:m16*... operators i1 input ..., i1 output ...
      n4 compute in0 | in1
      n2 -> n4 value 0 flip
      n3 -> n4 value 1 flip
    }
  }
```

Classical nodes name their records, operators, expressions, activation
(`when v0`), and provenance:

```text
  n13 observable 2 measurements i1:m15*...*i4:m4 operators i1 input ... from action 0
  n7 compute in0 from selector selective%20%5B1%2C0%2C2%5D
  n15 observable fragment measurements i1:m15*...*i6:m7 operators i1 input ... when v0
  n24 compute in0 from frame z (0,0,2)
```

Edges come last. A quantum edge names its pipe and its padding templates:

```text
  n1 -> n2 quantum (1,0,1)>(1,0,2) padding offset (8,0) one t8 loop t9
  n12 -> n13 compose 2
  n13 -> n24 value 0
  n13 -> n25 value 0 flip
  n1 -> n3 order
}
```

### Binary Bloq IR

Binary IR stores the same program compactly. Text and binary convert into each
other exactly. The CLI converts binary IR to text with:

```sh
bloq emit program.bloq --backend ir-text -o program.bloqir
```

### Save, load, and convert

::::{md-tab-set}
:::{md-tab-item} Python

```{literalinclude} ../examples/ir_roundtrip.py
:language: python
:start-after: "# [example-start]"
:end-before: "# [example-end]"
```

:::
:::{md-tab-item} Rust

```{literalinclude} ../examples/rust/ir_roundtrip.rs
:language: rust
:start-after: "// [example-start]"
:end-before: "// [example-end]"
:dedent: 4
```

:::
::::

`save` and `load` choose the codec from the file extension. Decoders reject
malformed structure, such as dangling endpoints or duplicate ids, but do not
check every semantic rule. Call `validate()` for that.

## APIs

The tables list the Python names. Rust provides the same operations on
`bloq::ir::Bloq`. The [Rust API](../api/rust.md) gives their exact
signatures.

### Read and traverse

| Operation | API |
| --- | --- |
| Program counts | `node_count`, `edge_count`, `quantum_node_count`, `qubit_count`, `measurement_count`, `stats()` |
| Top-level nodes and edges | `nodes()`, `node(id)`, `quantum_nodes()`, `edges()`, `incoming(id)`, `outgoing(id)` |
| Every level, including region bodies | `walk()`, `levels()`, `level_at(path)` |
| Classical dataflow | `value_inputs(id)`, `data_inputs(id)`, `value_consumers(id)`, `has_path(a, b)` |
| Templates and bundles | `template(id)`, `detector_bundle(id)` |
| Physical layout | `node_qubits(id)`, `sorted_layout_coords()`, `node_measurement_count(id)` |
| A schedule that respects every edge | `deterministic_emit_order()` |

Node ids are local to one graph level. `walk()` reports each node with its
`path`, the chain of `(region node, body)` hops from the top level:

```python
for entry in program.walk():
    kind = entry.node.kind
    if isinstance(kind, bloq.ir.BloqNodeKind.Classical):
        print(entry.path, entry.id, kind.node)
```

```{literalinclude} ../examples/ir_walk.txt
:language: text
```

Templates are reached through instances:

```python
node_id, quantum = program.quantum_nodes()[0]
instance = quantum.instances[0]
template = program.template(instance.template_id)
```

### Query structure

These queries name the landmarks a backend usually needs. Each fails with a
structural error if the program does not have the expected shape, rather than
returning a plausible but wrong node.

| Question | API |
| --- | --- |
| Output frame bits per output Port | `output_frames()` |
| Logical outputs and their owning instances | `logical_outputs()` |
| Seams feeding guarded quantum alternatives | `selection_seams()` |
| Region nodes of a kind | `regions_of(RegionKind.RepeatUntilSuccess)` |
| Unique quantum node with no outgoing quantum edge | `quantum_tail()` on the program or a body |
| Node realizing a source block | `node_by_block(pos)` |
| Node identity that survives recompilation | `stable_key_map()` |

### Resolve classical values

| API | Result |
| --- | --- |
| `resolve_classical(node)` | Measurement parity, constant sign, and decoder observables |
| `classical_value(node, ...)` | Boolean value under a fixed assignment |

For nonlinear choices, supply one assignment method:

| Argument | Assignment |
| --- | --- |
| `pins=True` or `pins=False` | Gives every predicate leaf the same Boolean value |
| `forced_observables={index: bit}` | Fixes individual corrected observable values |

```python
resolution = program.resolve_classical(13)
# <ClassicalResolution measurements=13 sign=False observables=[2]>
```

### Edit and specialize

| Operation | API |
| --- | --- |
| Fix structural choices, keeping the original | `pin_membership({selector: value})` |
| Check for unresolved guarded alternatives | `has_conditional_membership()` |
| Insert memory rounds | `insert_memory_rounds`, `insert_memory_rounds_after`, `insert_memory_rounds_batch` |
| Change a template's round count | `set_template_repetitions(template, n)` |
| Unroll every `REPEAT` body | `flatten()` |

Edits are transactional: a failed edit leaves the program unchanged.
[Synchronization](synchronization.md) explains the memory-round
edits with examples.

### Check and visualize

| Operation | API |
| --- | --- |
| Full semantic audit | `validate()` |
| Per-node circuit that a backend would emit | `emission_plan(node)` |
| Dependency graph as SVG | `to_svg(include_classical=True)` |

Compilation, emission, and VM lowering perform their own local checks.
`validate()` is the separate whole-program audit. Use it for IR from another
producer or IR edited by hand.

## Visualize the graph

Export the nested dependency view used by Bloq Editor as a self-contained SVG:

::::{md-tab-set}
:::{md-tab-item} Python
```python
from pathlib import Path

Path("t-gate-ir.svg").write_text(program.to_svg(include_classical=True), encoding="utf-8")
```
:::
:::{md-tab-item} Rust
```rust
std::fs::write("t-gate-ir.svg", program.to_svg(true))?;
```
:::
:::{md-tab-item} CLI
```sh
bloq view t-gate.bloqir --svg --include-classical -o t-gate-ir.svg
```
:::
::::

| Shown | Not shown |
| --- | --- |
| Nested graphs, dependencies, and activation conditions | Runtime retry attempts and execution times |

## What a consumer must preserve

| Information | Required interpretation |
| --- | --- |
| Loop structure | Preserve repetition counts, loop-carried detector state, and retry conditions when transforming loop bodies |
| Detector sign | Evaluate the stored constant together with the records |
| `Corrected` and `Flip` | Corrected logical parity versus the decoder correction alone |
| `Compose` | Includes recipe parity and bindings without adding the child's decoder flip |
| Fragment activation | Inactive contributions produce no record reads or boundary bindings |
| Expression operands | Every declared operand must be available, including unselected `Select` arms |
| Retry records | Records and accepted exports belong to their attempt |
| Logical output | Keep its owning instance, X/Z operators, and frame pair |

Equal physical support at two times does not identify the same logical output.
A consumer must keep the boundary's owning instance and dependencies.

Continue with [Backend Emission](emission.md) to turn a program into a Stim
circuit or verify dynamic control in simulation. The [Python API](../api/python.rst)
and [Rust API](../api/rust.md) document every method.
