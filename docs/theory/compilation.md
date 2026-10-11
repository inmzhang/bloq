# Compilation Pipeline

Compilation turns a block graph into a physical program. Bloq plans which
logical relations each measurement and output correction needs, builds the
fault-tolerant physical circuits for every block, and binds the two together with
the classical logic that runs between them. The result is
[Bloq IR](../backends/ir.md). Backend emission is a separate step.

This chapter walks through each compilation stage. The logical planning stage
is explained in [Correlation Surfaces](correlation-surfaces.md), and the
physical circuits in [Circuit Constructions](../circuit-constructions.md).

## Compile a graph

The same entry point accepts a local `BlockGraph` or a graph containing module
definitions and placed instances. Here we compile a gallery CNOT at distance 3:

::::{md-tab-set}
:::{md-tab-item} Python
```python
import bloq

graph = bloq.GalleryItem.CNOT.load()
program = bloq.compile(graph, distance=3)
program.save("cnot.bloqir")
```
:::
:::{md-tab-item} Rust
```rust
use bloq::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = GalleryItem::CNOT.build();
    let program = compile(&graph, 3)?;
    std::fs::write("cnot.bloqir", program.to_text())?;
    Ok(())
}
```
:::
:::{md-tab-item} CLI
```sh
bloq compile --gallery cnot -d 3 --backend ir-text -o cnot.bloqir
```
:::
::::

## Stages at a glance

```{mermaid}
flowchart LR
  src["Block graph"] --> val["1 Validation"]
  val --> cert["2 Certification"]
  cert --> corr["3 Correlations"]
  corr --> place["4 Placement"]
  place --> tmpl["5 Templates"]
  tmpl --> read["6 Readouts"]
  read --> fin["7 Finalization"]
  fin --> ir["Bloq IR"]
```

| Stage | Operation | Question it answers | Result |
| --- | --- | --- | --- |
| 1. Validation | Validating source | Is the source well formed and within size limits? | Checked source |
| 2. Certification | Certifying module | Do module definitions keep their interface contracts? | Linked definitions, or a cached artifact |
| 3. Correlations | Planning correlations | Which logical relations recover each value and output frame? | Readout plan |
| 4. Placement | Placing physical blocks | Which extraction schedule does each patch use, and where? | Scheduled physical cases |
| 5. Templates | Binding physical program | Which circuits realize the blocks, and which checks join them? | Placed instances and detectors |
| 6. Readouts | Lowering readouts | Which physical records realize each planned relation? | Observables and classical program |
| 7. Finalization | Finalizing program | How do the pieces form one ordered, checked program? | Optimized Bloq IR |

Stages 1 to 3 read only the block graph and do not depend on code distance.
Stages 4 to 7 build physical circuits whose size grows with distance.

## 1. Validation

Bloq first checks that the source describes a valid computation:

- Pipes join compatible faces, and Ports, T resources, and caps have their
  required connections.
- Module instances match their definitions' quantum and classical interfaces.
- Action names are unique, targets exist, and the action dependency graph is
  acyclic.
- The expanded graph fits the configured counts of blocks, instances, and
  occupied cells.

The size check counts the expanded graph from the module hierarchy before any
expansion. A small definition graph can expand exponentially, so Bloq refuses
an oversized source before spending memory on it.

## 2. Certification

Bloq computes a canonical key from the source's resolved BLOG text. If the
compile cache already holds a finished artifact for that key and
configuration, compilation reuses it.

On a cache miss, a hierarchical graph is prepared definition by definition:

1. **Reuse.** A definition variant compiled earlier for the same configuration
   and orientation is reused, including across different root programs.
2. **Link.** The root geometry is linked once. Only missing definition variants
   materialize descendant geometry, join connected ports, and qualify action
   names by instance path.
3. **Certify.** Each child definition's public interface is checked: every
   named readout must close before its live outputs. The root is checked again
   after composition, because a new connection can make a locally valid
   readout unsafe.

A local graph without modules skips the first two steps. The composition
algorithm is still evolving. [Modules](../modules/index.md) describes the
authoring model it implements.

## 3. Correlations

This stage computes the logical side of the program. It first builds the
**guarded topology**: the reachable assignments of every selector and
structural branch, and the local shape of each block under each assignment.
Reachability here describes Boolean control, not the probability of a quantum
outcome. If one selector is $a$ and another is $\neg a$, the planner does not
treat all four joint choices as reachable.

The general composed-surface planner builds the correlation space by seam
elimination with Boolean-guarded coefficients, plans named readouts in causal
waves, and derives terminal frame equations. It produces these recipes:

| Recipe | Contents |
| --- | --- |
| Named readout | A surface for the value, earlier corrected values it folds in, and its feedback terms |
| Terminal frame | One equation per output axis, with references to later frame bits |
| Logical readout | A closed surface, such as a memory or stability observable |

Planning finishes before any physical work starts. Later stages consume this
plan and never revisit the correlation space. [Correlation
Surfaces](correlation-surfaces.md) explains the algorithm.

## 4. Placement

Placement decides how each patch is realized physically:

- **Schedules.** Patches joined by spatial pipes need compatible syndrome
  extraction schedules. Bloq groups each spatial component and selects
  compatible compact, padded, or wall schedules for it.
- **Walks.** Adjacent walks on the same source layer with the same movement
  share one quantum node. Their templates merge tick by tick, so slides and
  glides run together while preserving each patch's measurement records.
- **Reservations.** T-resource spill cells are reserved against the union of
  all alternatives that can occupy them.
- **Spatial Ports.** A spatial Port expands into an inferred cube and a
  temporal Port, with explicit provenance for the substituted qubits.
- **Coordinates.** A block at $(x,y)$ is placed at physical offset
  $(x,y)\cdot(2d+2)$. Coordinate overflow is a typed error.

Bloq places each distinct local case once. Joint assignments of independent
choices are not expanded, so placement work follows the number of distinct
local shapes rather than $2^s$ for $s$ selectors.

## 5. Templates

This stage builds and places the physical circuits.

**Build templates once.** Each block has a **signature**: its kind, round
count, connectivity, boundary bases, schedule, and T-surgery side. Bloq
compiles one **template** per distinct signature and caches it. A template
holds:

- the fault-tolerant physical circuit;
- its internal detectors, derived from the circuit's own stabilizer flows;
- its **boundary flows**, the flows it leaves open for neighbors to close;
- its **observable gateway**, which maps a local Pauli pattern to the
  template's measurement records.

A selective cap has one template per basis. A T block has cultivation and
escape templates, or the configured ideal preparation.

**Place instances.** An instance places a template at an offset for one source
block. Two instances of the same template share circuit data but own distinct
measurement records. Structural branches become guarded instance membership
rather than separate subgraphs.

**Compose checks.** Bloq composes the boundary flows of adjacent instances.
Each flow that closes becomes a detector, or a retry check inside a T retry
region. Detector recipes that repeat across the program are stored once in a
shared bundle and referenced by each use. Template-internal detectors are not
recomposed per instance.

The number of templates depends on the variety of blocks, not on distance or
program size. Each template's circuit grows
with distance: a patch has $O(d^2)$ qubits and an ordinary cube runs $d$
rounds.

## 6. Readouts

This stage binds each planned recipe to physical records. For every block
that a planned surface touches, Bloq reconstructs the surface's local Pauli
pattern and asks that instance's gateway for the records that realize it. The
records of all blocks combine into one complete `Observable` node per named
readout or output axis. That node carries both the raw parity and the
corrected value. The decoder estimates the difference between them.

The source's classical behavior becomes IR nodes:

| Source behavior | IR representation |
| --- | --- |
| Named logical measurement | Raw `Observable` parity and corrected `Observable` value |
| Earlier corrected value in a recipe | `ReadoutFold` dependency |
| Pauli feedback | `FeedbackFold` contributions to affected readouts and output frames |
| Boolean expression | `Compute` node |
| Measurement-basis selection | Guarded quantum alternatives, selected by corrected values |
| Shot rejection | `Discard` effect |
| Retried T preparation | `RepeatUntilSuccess` region with accepted exports |

A record in an alternative that was not taken is unavailable, and only an
accepted retry attempt exports its records and resource patch. The
compiler additionally checks that the emitted dependency graph is
acyclic before accepting its result.

## 7. Finalization

Finalization turns the assembled pieces into one program:

1. **Bind the logical interface.** Record which instances own each input and
   output cut. Planning state is released after this step.
2. **Order occupancy.** Add ordering edges so that a qubit is reused only after
   its previous owner finishes. Alternatives may share qubits only when their
   guards cannot hold together.
3. **Optimize.** Simplify the classical dependency graph while preserving
   activation guards, selectors, retry roots, and fold roles. Infer each cut's
   logical operators from the optimized IR.
4. **Check the layout.** Confirm that no two coexisting instances claim the
   same physical qubit at the same time.

The result records the code distance and construction convention as metadata.
