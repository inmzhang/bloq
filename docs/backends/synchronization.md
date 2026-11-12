# Synchronization

Bloq IR orders work by dependencies, not by clock times. A circuit simulator
can run that order directly. Hardware cannot: patches finish at different
times, decoders take time to answer, and resource factories succeed after a
random number of attempts. Configure the backend's timing and waiting
behavior to specify **how long** patches wait and **what they do** meanwhile.

Long waits need QEC rounds to protect the logical patch. These physical
circuits have their own measurements and detectors. Short gaps can instead
use physical idle intervals. Use Bloq IR's memory-round edits to reserve QEC
rounds, and set the backend's timing controls for moment durations and source
release times.

## Where waits come from

```{mermaid}
flowchart LR
  acc["Resource preparation<br/>accepted resource ready"]
  inp["Input arrival"] --> data["Data patch ready"]
  meas["Logical measurement"] --> dec["Corrected value ready"]
  acc --> join["Join or selected operation"]
  data --> join
  dec --> join
  data --> qec["QEC or short idle while waiting"]
  qec --> join
```

| Cause | Example | What the waiting patch needs |
| --- | --- | --- |
| Join | A partner patch arrives later, or a T-state factory retries until accepted | Protect the waiting patch until the partner or resource is ready |
| Decoder latency | An adaptive cap or output Pauli frame needs a decoded value | Protect the patch until the decoder answers |

How these waits are realized depends on what the hardware controller can do.

| Controller capability | Synchronization model | Bloq mechanism |
| --- | --- | --- |
| Fixed instruction schedule, no runtime waiting | **Static**: every wait is a fixed number of rounds chosen in advance | Insert memory rounds into the IR |
| Runtime clock, readiness checks, and repeatable QEC loops | **Dynamic**: wait for actual readiness | Repeat QEC rounds and use physical idle for short gaps to known deadlines |

Both models use the same IR support, described next.

## IR support for waiting

### Padding templates on every seam

During compilation, each supported quantum edge records memory templates
matching its patch's X/Z boundary orientations at the seam: one template for
a single round and a looped template for several. The text IR shows them on
the edge:

```text
n1 -> n2 quantum (1,0,1)>(1,0,2) padding offset (8,0) one t8 loop t9
```

Waits can therefore be inserted into a compiled or loaded program without the
source graph or the compiler.

[Aligned Stim emission](emission.md#stim) uses these templates at temporal
Hadamard layers. Live patches on ordinary temporal seams run a memory round
alongside the Hadamard realignment.

### Memory padding nodes

An insertion splits a quantum edge with a new quantum node that has
`MemoryPadding` provenance and records its round count. Bloq recomposes the
detectors across the new seams, so the edited program keeps a complete,
deterministic set of checks. Every edit is transactional: on error the
program is unchanged.

### Edit APIs

| API | Inserts memory rounds |
| --- | --- |
| `insert_memory_rounds(from, to, rounds, path=...)` | On one quantum edge, at any graph level |
| `insert_memory_rounds_after(node, rounds, path=...)` | After the terminal quantum node of a region body |
| `insert_memory_rounds_batch(targets, rounds)` | At several `MemoryRoundTarget.Edge` or `.After` targets, atomically |

Use structural queries to choose targets. `selection_seams()` locates edges
feeding guarded alternatives. `regions_of()` and `quantum_tail()` locate retry
outputs. `node_by_block()` locates a node by its source block coordinates.
If any batch insertion fails, none of the targets are changed.

The round count is always supplied by the caller. Bloq does not know a
device's decoder latency or classical delay; choose it from the hardware's
timing.

## Example: fixed memory on one seam

Insert four memory rounds between the first two nodes of a CNOT:

::::{md-tab-set}
:::{md-tab-item} Python
```python
import bloq

program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
seam = next(
    (edge.source, edge.target)
    for edge in program.edges()
    if isinstance(edge.edge, bloq.ir.BloqEdge.Quantum)
)
padding = program.insert_memory_rounds(*seam, rounds=4)
program.validate()
print(program.node(padding).provenance)  # NodeProvenance.MemoryPadding(rounds=4)
```
:::
:::{md-tab-item} Rust
```rust
use bloq::ir::{BloqNodeId, LevelPath, MemoryRoundTarget};
use bloq::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut program = compile(&GalleryItem::CNOT.build(), 3)?;
    let (from, to) = (BloqNodeId(0), BloqNodeId(1));
    let target = MemoryRoundTarget::Edge { path: LevelPath::default(), from, to };
    let padding = program.insert_memory_rounds(target, 4)?;
    program.validate()?;
    println!("inserted {padding:?}");
    Ok(())
}
```
:::
::::

## Example: wait for decoder before an adaptive choice

In the logical T gate, the corrected value of `mzz` selects the basis of the
resource patch's final measurement. The decoder needs time to produce that
value. A controller without runtime waiting must reserve that time in advance:

```python
import bloq
from bloq.ir import MemoryRoundTarget, RegionKind

ir = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
padded = bloq.Bloq.from_binary(ir.to_binary())  # independent copy
targets = [MemoryRoundTarget.Edge(source, target)
           for source, target in padded.selection_seams()]
for region in padded.regions_of(RegionKind.RepeatUntilSuccess):
    if region.path:  # only top-level factories
        continue
    path = region.path + [(region.node, "body")]
    tail = padded.level_at(path).quantum_tail()
    targets.append(MemoryRoundTarget.After(tail, path=path))
padded.insert_memory_rounds_batch(targets, rounds=10)
padded.validate()
```

The edge target holds the resource patch before its adaptive cap. The `After`
target holds the accepted T state at the factory output. One batch reserves
ten rounds at both targets.

See [Emulate Hardware Synchronization](thth.md) for a complete example of
synchronization with explicit backend timing controls.
