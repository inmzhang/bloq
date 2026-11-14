# Actions

Blocks and pipes describe the geometry layout of the surface code computation.
Actions name logical measurements, combine their results, and use those values
to choose a measurement basis, apply a Pauli correction, or keep a shot.
Bloq derives the physical readout recipes and their dependencies during compilation.

## Action kinds

| Action | BLOG example | Effect |
| --- | --- | --- |
| Named measurement | `mzz = measure 1 -> +X` | Name a logical observable on a block or spatial pipe |
| Binding | `odd = m1 ^ m2` | Name a Boolean expression |
| Resolve | `resolve 5 if odd` or `resolve correction if odd` | Select a measurement basis or a branch geometry |
| Pauli feedback | `feedback Z 1 -> +Z if odd` | Apply a logical correction at a block or wire target |
| Discard | `discard if bad` | Reject the shot when the condition is true |

Targets refer to the authored spacetime layout. A block reference can be its
ID or coordinate, such as `[0, 0, 1]`. A pipe reference adds a direction from
one endpoint, such as `[0, 0, 1] -> +X`. These coordinates identify logical
geometry, while the compiler determines physical qubits and execution times.

### Named measurement

`name = measure target` names a logical observable already measured by the
geometry. Its value is a bit: zero for eigenvalue $+1$ and one for eigenvalue
$-1$.

For a **block target**, the patch terminates at a measured top face, facing
forward in time. The observable has a correlation surface whose support at
that face agrees with its measurement basis. For example, an `XZX` cube has
an X top face, so `mx = measure block_id` names its logical X measurement.
Fixed `X` and `Z` caps and a terminal `Y` block supply their corresponding
bases. A selective cap supplies the basis chosen by its resolve action.
A block with a continuing temporal wire does not become a terminal measurement
merely because a `measure` statement names it. Ports and T resources are not
block measurement targets.

For a **pipe target**, the pipe must be spatial, along X or Y. It represents
lattice surgery between patches. The named observable is the joint parity
whose correlation surface crosses that pipe in the surgery measurement basis.
For an ordinary pipe, this is the Pauli basis complementary to its time face.
For example, the spatial pipe between the two `XZX` cubes in the T example
below measures $Z\otimes Z$, even though its direction is `+X`. The direction
locates the pipe and does not specify the Pauli basis. Hadamard pipes exchange
the endpoint frames, so their support must be interpreted in each local frame.
Temporal pipes continue a patch and cannot be named parity measurement targets.

The named value combines physical records, signs, frame terms, and the decoder's
flip estimate. Conditions use this **corrected logical value**. Bloq must find
an output safe readout in every reachable choice. Naming a relation that touches
a live output does not authorize measuring that output. See
[Correlation Surfaces](../theory/correlation-surfaces.md) for the readout closure rule.

### Binding

`odd = m1 ^ m2` gives a name to a Boolean expression. It has no block or pipe
target and adds no quantum operation. Its inputs can be named measurements,
other bindings, or declared classical module inputs.

| Operator, from highest to lowest precedence | Meaning |
| --- | --- |
| `!` | NOT |
| `&` | AND |
| `^` | XOR |
| `\|` | OR |

Parentheses make grouping explicit, as in `odd = (m1 ^ m2) & !bad`.
All operands must be available. An OR expression still waits for both inputs
when one is already true.

### Resolve

`resolve target if condition` selects one of two authored alternatives.

For a **selective measurement block**, the target is the cap's position or ID.
The condition chooses its Pauli basis. In `XY`, true selects X and false
selects Y. In `XZ`, true selects X and false selects Z. In `YZ`, true selects
Y and false selects Z. Reversed spellings reverse this assignment and are
canonicalized when saved. The choice acts at the cap's time boundary and must
wait for its corrected selector before the measurement can execute.

For a **structural branch**, the target is its name, such as `correction`.
True selects the `true` geometry and false selects the `false` geometry.
The arms share an external interface, and only the selected arm executes.
Each arm component must connect to that interface. An arm may prepare an
ancilla and add a spatial surgery pipe to a shared cube. Arms cannot contain
nested branches, Ports, T resources, or selective caps.

### Pauli feedback

`feedback Z target if odd` applies a logical Pauli correction when the
condition is true. Its target locates where the logical operator acts in the
spacetime layout.

For a **block target**, such as `feedback Z block_id if odd`, Bloq selects
an outgoing temporal wire at the block, then an incoming temporal wire,
then a sole spatial wire. A multiplex Port selects its retained logical
output. The block reference therefore places the correction on a particular
logical continuation.

For a **wire target**, such as `feedback Z block_id -> +Z if odd`, the
direction explicitly selects the incident pipe from that block's endpoint.
Use this form to distinguish wires when several meet at a block. The Pauli
is expressed in the selected endpoint's frame. Across a Hadamard pipe,
X and Z exchange, so moving the correction to the other end requires changing
its basis.

Interpret either target through the [correlation surfaces](../theory/correlation-surfaces.md)
that cross it. At the selected wire, compare the feedback Pauli with each
surface's local Pauli support. Anticommuting support flips the sign of that
relation and hence the logical observable it carries. For example, Z feedback
flips an X observable whose surface crosses the target with X support. The
observable's bit is XORed with `odd`. Z or identity support commutes with the
same feedback and contributes no flip. A geometric intersection alone does
not determine the effect, since the Pauli labels and endpoint frames matter.
Bloq uses these crossings to propagate feedback into named readouts and output
frame relations.

Feedback can list several targets separated by commas. Omitting `if` makes
it unconditional. For each correlation surface, the contributions combine by
XOR, including repeated targets. An odd number of anticommuting crossings
flips its observable, while two identical corrections cancel.

### Discard

`discard if bad` rejects the current shot when its condition is true.
It has no geometric target and waits for the values in its condition.
The runtime cancels work that has not yet issued once rejection is known.
It does not undo operations that have already executed.

Discard is postselection on the computation. A T resource's cultivation retry
protocol instead repeats that resource preparation inside its compiled region.
Discard does not request a retry of the whole computation.

## Action DAG

Actions form a directed acyclic graph, or **DAG**. Each node is an action and
each arrow points from a prerequisite to its consumer. Named values give
explicit dependencies, such as a measurement feeding a resolve. Analysis also
finds dependencies from correlation surfaces: a readout can wait for a selected
cap, a branch containing its support, an earlier corrected parity, or feedback
that changes its sign.

::::{md-tab-set}
:::{md-tab-item} Python
```{literalinclude} ../../bloq_py/examples/action_dag.py
:language: python
:start-after: "[dag-start]"
:end-before: "[dag-end]"
```
:::
:::{md-tab-item} Rust
```{literalinclude} ../../bloq/examples/action_dag.rs
:language: rust
:start-after: "[dag-start]"
:end-before: "[dag-end]"
```
:::
::::

The complete three bit adder graph below contains 51 actions.

```{raw} html
<div class="bloq-output bloq-ir-graph" role="region" aria-label="Three bit adder action DAG" tabindex="0">
```
```{image} ../assets/three-bit-adder-action-dag.svg
:alt: Complete analyzed source action DAG for the three bit adder with measurements, bindings, branch resolves, feedback and their dependencies.
```
```{raw} html
</div>
```

Source line order does not set start times. Independent actions can proceed
concurrently, while a cycle means the required values cannot be produced in a
causal order.

## Examples

### Logical T gate

The data patch runs from `[0, 0, 0]` to `[0, 0, 2]`. A cultivated T resource
enters beside it at `[1, 0, 0]`. Two `XZX` cubes join through a spatial pipe,
and an `XY` cap terminates the resource patch. Build the layout and its two
actions directly:

::::{md-tab-set}
:::{md-tab-item} Python
```{literalinclude} ../../bloq_py/examples/action_t_gate.py
:language: python
:start-after: "[build-start]"
:end-before: "[build-end]"
```
:::
:::{md-tab-item} Rust
```{literalinclude} ../../bloq/examples/action_t_gate.rs
:language: rust
:start-after: "[build-start]"
:end-before: "[build-end]"
```
:::
:::{md-tab-item} BLOG
```{literalinclude} ../examples/viewers/action-t-gate.blog
:language: blog
```
:::
::::

```{bloq-view} action_t_gate
:source: examples/viewers/action-t-gate.blog
```

The same graph with the `mzz` correlation surface:

```{bloq-view} action_t_gate
:source: examples/viewers/action-t-gate.blog
:measurement: mzz
```

`mzz` names the $Z\otimes Z$ parity measured by the spatial pipe at
`[0, 0, 1] -> +X`. The Bloq compiler constructs a corresponding correlation
surface for this measurement action and uses it to derive the physical
readout recipe for `mzz`. Its corrected value controls the resource cap.
`resolve [1, 0, 2] if !mzz` selects X when `mzz` is zero and Y when it is one.
The compiler derives the remaining Pauli frame for the surviving data output.
The [logical T gate tutorial](../tutorials/t-gate.md) continues with physical
execution and output checks.

### CCZ gate teleportation

This layout consumes an externally supplied three qubit $|CCZ\rangle$ state
through the temporal input Ports at `[0, 2, 0]`, `[1, 2, 0]`, and `[2, 2, 0]`.
Three multiplex Ports at `[0, 0, 1]`, `[1, 0, 1]`, and `[2, 0, 1]` couple the
data to the resource while retaining the data outputs. Three correction regions
choose either X measurement caps or a geometry implementing a CZ correction.
The source supplies the teleportation protocol, without preparing the CCZ state.

::::{md-tab-set}
:::{md-tab-item} Python
```{literalinclude} ../../bloq_py/examples/action_ccz_teleport.py
:language: python
:start-after: "[build-start]"
:end-before: "[build-end]"
```
:::
:::{md-tab-item} Rust
```{literalinclude} ../../bloq/examples/action_ccz_teleport.rs
:language: rust
:start-after: "[build-start]"
:end-before: "[build-end]"
```
:::
:::{md-tab-item} BLOG
```{literalinclude} ../examples/viewers/action-ccz-teleport.blog
:language: blog
```
:::
::::

```{bloq-view} action_ccz_teleport
:source: examples/viewers/action-ccz-teleport.blog
```

The three measurement correlation surfaces are shown with all correction
branches projected to their true arms:

::::{md-tab-set}
:::{md-tab-item} m0x
```{bloq-view} action_ccz_teleport
:source: examples/viewers/action-ccz-teleport.blog
:measurement: m0x
```
:::
:::{md-tab-item} m1y
```{bloq-view} action_ccz_teleport
:source: examples/viewers/action-ccz-teleport.blog
:measurement: m1y
```
:::
:::{md-tab-item} m2z
```{bloq-view} action_ccz_teleport
:source: examples/viewers/action-ccz-teleport.blog
:measurement: m2z
```
:::
::::

`m0x`, `m1y`, and `m2z` name the three joint measurements at the multiplex
connections. The suffixes label the data wires. Each outcome selects one CZ
correction region: `b0`, `b1`, or `b2`. False selects the pair of X caps,
while true selects the connected correction geometry. The remaining Pauli
corrections depend on pairs of outcomes:

| Retained data output | Z feedback condition |
| --- | --- |
| `z` at `[2, 0, 1]` | `m0x & m1y` |
| `y` at `[1, 0, 1]` | `m0x & m2z` |
| `x` at `[0, 0, 1]` | `m1y & m2z` |

These nonlinear conditions and the three branch selectors remain separate
actions in the DAG. They consume corrected values from the same signed
correlation relation.

Continue with [Modules](../modules/index.md) to compose geometry and classical
interfaces, then [BLOG Format](blog.md) to write the complete source as text.
