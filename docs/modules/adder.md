# Controlled Addition

The `three_bit_adder` gallery entry combines reusable arithmetic stages into
controlled addition. This example shows quantum composition, classical
corrections, and an explicit resource interface. Start with
[reusable modules](index.md).

## Goal and interfaces

For little-endian three-bit registers $i$ and $t$, the corrected logical map is

$$
|q\rangle|i\rangle|t\rangle
\longmapsto |q\rangle|i\rangle|t+qi\pmod 8\rangle.
$$

For example, $q=1$, $i=3$, and $t=2$ give sum $5$. With $q=0$, the target
register stays $2$. The operation preserves the control and addend, including
their coherent superpositions. It is not a classical integer calculator.

| Boundary | Names | Meaning |
| --- | --- | --- |
| Control | `q_in`, `q_out` | One preserved data qubit |
| Addend | `i0_in` … `i2_in`, `i0_out` … `i2_out` | Three preserved data qubits, least significant bit first |
| Target and sum | `t0_in` … `t2_in`, `s0_out` … `s2_out` | Target input and corrected sum output |
| Resources | `bit0_and_ccz_*`, `bit0_maj_ccz_*`, `bit1_and_ccz_*`, `bit1_maj_ccz_*`, `bit2_and_ccz_*` | Fifteen quantum ports, grouped into five three-qubit CCZ states |
| Classical results | `raw_erase_0`, `raw_erase_1`, `raw_erase_2` | Exported erase readouts; internal actions use their corrected recipes |

:::{important}
Executing this adder requires five externally prepared CCZ states. Supply
$\mathrm{CCZ}|+\rangle^{\otimes 3}$ at each resource triple. The `ccz` type checks
connections; it does not prepare that state. The VM's preparation hooks can
inject these resources.
:::

::::{md-tab-set}
:::{md-tab-item} Block graph
```{bloq-view} three_bit_adder
```
:::
:::{md-tab-item} Modules
```{bloq-view} three_bit_adder
:modules:
```
:::
::::

## Inspect the reusable construction

The source contains six helper definitions and `main`. The root places eight
instances:

```{literalinclude} ../../bloq_graph/assets/three_bit_adder.blog
:language: text
:start-at:   bit0_and: InjectedAnd
:end-at:   bit2_tail: TailParity
```

| Definition | Role |
| --- | --- |
| `InjectedAnd` | Form the temporary $q\land i_k$ using a CCZ resource |
| `HeadMaj`, `BulkMaj` | Propagate carries using CCZ resources |
| `HeadUmaParked`, `BulkUmaParked` | Uncompute carries and return sum bits |
| `TailParity` | Handle the highest bit without another majority stage |

```{mermaid}
flowchart LR
  q["q and addend i"] --> injected["3 InjectedAnd instances"]
  injected --> head["HeadMaj: bit 0"] --> bulk["BulkMaj: bit 1"] --> tail["TailParity: bit 2"]
  tail -. corrected carry bit .-> uma1["BulkUmaParked: bit 1"]
  uma1 -. corrected carry bit .-> uma0["HeadUmaParked: bit 0"]
  tail --> s2["s2"]
  uma1 --> s1["s1"]
  uma0 --> s0["s0"]
```

This diagram shows logical dependencies. The BLOG file also authors the physical
routing cubes, ports, and Hadamard seams. Those are part of the construction.

For example, the first stage connects majority outputs to its uncompute stage:

```text
bit0_maj.c_k_plus_1_duplicate -> bit0_uma.c_k_plus_1
bit0_maj.c_k_xor_i_prime_k -> bit0_uma.i_prime_k
bit0_maj.c_k_xor_t_k -H> bit0_uma.t_k
```

The final corrections are structural CZ choices controlled by the erase
readouts. Classical bindings carry phase information backward:

```{literalinclude} ../../bloq_graph/assets/three_bit_adder.blog
:language: text
:start-at:   bit1_uma.carry_z =>
:end-at:   bit2_tail.m_ikprime =>
```

Removing these bindings or selecting final CZ arms independently changes the
coherent channel. A truth table on computational-basis inputs alone would not
detect every resulting phase error.

## Load and compile

Compile at code distance three and save text Bloq IR:

::::{md-tab-set}
:::{md-tab-item} Python
```python
from pathlib import Path
import bloq

example = bloq.GalleryItem.THREE_BIT_ADDER
Path("three-bit-adder.blog").write_text(example.source(), encoding="utf-8")
graph = example.load()
assert graph.flatten().block_count == 532
program = bloq.compile(graph, distance=3)
program.validate()
program.save("three-bit-adder.bloqir")
```
:::
:::{md-tab-item} Rust
```rust
use bloq::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = GalleryItem::ThreeBitAdder.build();
    assert_eq!(source.instances.len(), 8);
    std::fs::write("three-bit-adder.blog", source.to_blog_text())?;
    let program = compile(&source, 3)?;
    program.validate()?;
    std::fs::write("three-bit-adder.bloqir", program.to_text())?;
    Ok(())
}
```
:::
::::

Compilation uses the hierarchy directly. The saved BLOG retains its reusable
definitions and instances.

The CLI can compile the gallery directly:

```bash
bloq compile --gallery three_bit_adder -d 3 --backend ir-text -o three-bit-adder.bloqir
bloq validate three-bit-adder.bloqir
```

Or download {download}`the full three-bit adder BLOG
<../../bloq_graph/assets/three_bit_adder.blog>` and compile that file. Open it in
the [editor](https://bloqec.com/editor/), select **Modules**, then open a linked
definition tab to inspect a stage's implementation.

## Verify the adaptive result

The IR retains the adaptive final CZ choices and classical dependencies. Use
the [verification VM](../backends/vm.md) for simulation with the supplied CCZ
states. A static Stim circuit requires a supported pinned realization and does
not execute these runtime choices.

| Gallery entry | Purpose | External CCZ states |
| --- | --- | --- |
| `three_bit_adder` | Executable controlled addition example | 5 |
| `ten_bit_adder` | Larger compilation and scaling example | 19 |

The AND/MAJ/UMA resource constructions draw on the arithmetic layouts in
[Low et al., *A Denser Planar Surface Code*](https://arxiv.org/abs/2605.30455).
The authored BLOG supplies the actual routing, readout, and correction behavior.
Continue with [phase-gradient rotations](phase-gradient.md) for repeated
single-qubit resource injection, or [compilation](../theory/compilation.md) for
the steps that produce these circuits and dependencies.
