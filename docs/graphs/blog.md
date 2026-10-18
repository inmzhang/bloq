# BLOG Format

BLOG is the standard input format of the Bloq compiler. It represents a block
graph as text: blocks and pipes describe the spacetime layout, actions control
the computation, and modules compose reusable components.

This chapter gives the syntax for the concepts introduced in
[Blocks and Pipes](concepts.md), [Actions](actions.md), and
[Modules](../modules/index.md).

## BLOG and BlockGraph

BLOG is the text representation; `BlockGraph` is the in-memory source data
structure. Loading BLOG creates a `BlockGraph`. Building a graph through the
Python or Rust API creates the same source structure, which can be saved as
BLOG. The compiler consumes that `BlockGraph`, including its module hierarchy.

Parsing and saving retain definitions, interfaces, instances, quantum
connections, classical bindings, and actions. Saving writes canonical
formatting, module-local IDs, and equivalent spellings; it does not retain the
original comments or whitespace. Compilation accepts the hierarchy directly.
An explicit `flatten()` creates an independent assembled projection.

| Operation | Python | Rust |
| --- | --- | --- |
| Parse inline source | `BlockGraph.from_text(text)` | `BlockGraph::from_text(text)` |
| Load source and resolve imports | `BlockGraph.load(path)` | `BlockGraph::load(path)` |
| Serialize the complete source | `graph.to_text()` | `graph.to_blog_text()` |
| Save the complete source | `graph.save(path)` | `graph.to_file(path)` |

Loading resolves imports; saving that graph writes all resolved definitions
together, so the resulting file is self-contained.

## Source structure

An executable BLOG file starts with `BLOG 1.0` and defines `module main`, the
root of the computation. Other definitions form its reusable module library.
Optional imports appear before the definitions.

```blog
BLOG 1.0

module main {
  # Quantum and classical interfaces
  # Local blocks and pipes
  # Child module instances and their connections
  # Actions
}
```

Ordinary statements occupy one line. Opening braces follow module or branch
names; closing braces occupy their own lines. Indentation and blank lines are
optional. A `#` starts a comment that runs to the line end.

## Syntax reference

### Statement kinds

| Statement | Example | Meaning |
| --- | --- | --- |
| Block | `1: ZXZ [0, 0, 0]` | Place a block with a module-local ID |
| Pipe | `0 -> +Z` | Connect a block to its neighbor in the given direction |
| Quantum interface | `in q_in: data = 0` | Name a Port and its resource annotation |
| Classical interface | `out bit = mz` | Export a Boolean value |
| Instance | `child0: CNOT @ [0, 0, 0]` | Place a child module instance |
| Quantum connection | `child0.control_out -> child1.control_in` | Join child module quantum boundaries |
| Classical binding | `child0.result => child1.flip` | Supply a child module's classical input |
| Action | `mzz = measure 1 -> +X` | Name a logical readout |
| Branch region | `branch correction { ... }` | Group false and true geometries for a resolve target |

### Blocks and pipes

| Statement | Form |
| --- | --- |
| Block | `id: KIND [x, y, z] [attribute=value ...] [<tag>]` |
| Walking block | `id: walk BOUNDARY [start] -> [end] [<tag>]` |
| Patch rotation | `id: rotate BASIS [start] -> [end] [<tag>]` |
| Pipe | `source -> destination [<tag>]` or `source -H> destination [<tag>]` |

Brackets around coordinates are literal; other brackets in the forms above
indicate optional parts. IDs are unsigned and **local to the containing
module**. Coordinates are signed integers in that module's spacetime layout.
A reference is an ID or `[x, y, z]`. A pipe destination may also be `+X`, `-X`,
`+Y`, `-Y`, `+Z`, or `-Z`. For adjacent blocks, `0 -> 1`, `0 -> +Z`, and
`[0, 0, 0] -> +Z` can describe the same pipe. `-H>` carries a Hadamard and
exchanges endpoint Pauli frames. Neither arrow performs automatic routing.

Cube kinds are `XZZ`, `ZXZ`, `ZZX`, `ZXX`, `XZX`, and `XXZ`. Other kinds are
`Port`, `Y`, `T`, fixed measurement caps `X` and `Z`, and selective pairs
`XY`/`YX`, `XZ`/`ZX`, and `YZ`/`ZY`. The first selective basis is chosen for
true and the second for false; writers canonicalize equivalent reverse
spellings and adjust the selector. See [Blocks and Pipes](concepts.md#special-variants)
for walking boundaries and rotation bases.

| Attribute | Applies to | Example |
| --- | --- | --- |
| Height | Cubes | `height=2d` |
| Role | Ports | `role=auto`, `role=input`, `role=output`, `role=multiplex` |
| Color | Ports | `color=7396ff` |
| Tag | Blocks and pipes | `<control_in>` |

Height expressions include `d`, `2d`, `d/2`, and `3d+2`. For coefficient
$k$ and offset $n$, a cube occupies $\lceil k\rceil$ logical time cells and
runs $\lceil kd\rceil+n$ syndrome rounds. Offsets change rounds, not cell
occupancy. Spatially connected cubes share a height; contradictory declarations
fail. See [Blocks and Pipes](concepts.md) for connection rules.

Port colors are six hexadecimal RGB digits without `#` and affect only the
rendering. Temporal Ports infer their role from pipe geometry with `role=auto`;
explicit input/output roles can reverse it. Spatial roles are explicit, and
`role=multiplex` requires a spatial pipe. Tags are nonempty and contain no
whitespace, controls, `<`, or `>`; other Unicode characters are allowed.
For example:

```blog
0: Port [0, 0, 0] role=input color=7396ff <control_in>
1: ZXZ [0, 0, 1] height=2d
0 -> +Z <input_wire>
```

### Interfaces, instances, and connections

| Statement | Form |
| --- | --- |
| Quantum input or output | `in name: resource = id` or `out name: resource = id` |
| Classical input | `in name` |
| Classical output | `out name = expression` |
| Instance | `name: Definition @ [x, y, z] [rotate X\|Y\|Z degrees]` |
| Quantum connection | `endpoint -> endpoint` or `endpoint -H> endpoint` |
| Classical binding | `value-name => child.input` |

A quantum interface names a module-local Port block. Resource names such as
`data` and `ccz` are annotations checked for connection compatibility; they do
not prepare states or change physical compilation. A child endpoint is written
as `child.port`. A parent endpoint in a module connection is a local block ID.
A classical binding names a parent value or child output; bind an expression
to a name first if it needs to be passed to a child.

An instance's optional rotation uses a multiple of 90 degrees and acts before
translation. For example, `child1: CNOT @ [2, 1, 4] rotate Z 180` rotates the
complete module. A `rotate BASIS` block instead changes a patch's boundary
orientation during the computation. See [Modules](../modules/index.md) for
port matching, supported rotations, and how connections affect child actions.
BLOG records the authored placement and connections; it adds no inferred
routing, Hadamard, or correction.

Module, instance, and interface names match `[A-Za-z_][A-Za-z0-9_+-]*`.
Reserved keywords and `__` are not allowed.

### Actions and expressions

| Action | Form |
| --- | --- |
| Measure | `name = measure ref [-> direction]` |
| Bind a value | `name = expression` |
| Resolve | `resolve ref-or-branch-name if expression` |
| Feedback | `feedback Pauli ref [-> direction] [, Pauli ref ...] [if expression]` |
| Discard | `discard if expression` |

Expressions accept names, `!`, `&`, `^`, `|`, and parentheses.
Precedence is `!`, then `&`, then `^`, then `|`. For example,
`bad = (m0x ^ m1y) | (m2z & !m0x)` binds a Boolean value from three readouts.
Action references match `[A-Za-z_][A-Za-z0-9_/.+-]*`; `.` separates a child
instance from its output. Locally declared action names cannot contain `.` or
`__`, or be reserved keywords. `__` is reserved for linked instance paths.

Actions use their containing module's names and targets. Feedback requires at
least one target, uses `X`, `Y`, or `Z`, and acts in the selected wire endpoint
frame. An omitted condition means unconditional feedback. A direction selects
an edge endpoint for measurement or feedback. Named readouts require an
output-safe correlation. See [Actions](actions.md) for these semantics.
Dependencies determine execution order; source line order does not assign
physical execution times.

### Branch regions

```blog
branch correction {
  false {
    # Blocks and pipes for outcome false
  }
  true {
    # Blocks and pipes for outcome true
  }
}
resolve correction if result
```

A branch has one `false` arm and one `true` arm sharing a common external
interface. Arms contain only block and pipe statements, with no nested branches.
Both arms remain part of the source until selection. They belong to the
containing module. See [Resolve](actions.md#resolve) for selection rules.
The [CCZ teleportation example](../gallery/ccz_gate_teleport.md) contains
complete branch geometries.

### Imports

`import "cnot.blog" as CNOT` imports that file's `main` definition under the
name `CNOT`. Helper definitions are namespaced under the alias. Paths resolve
relative to the importing file. Each imported file defines its own `main`.
Imports and module references must not form cycles.

For complete BLOG sources, visit [Examples](../gallery/index.md). Each gallery
entry explains its functionality, shows the block graph, and includes its full
BLOG source.

Continue with [Correlation Surfaces](../theory/correlation-surfaces.md) to see
how Bloq derives logical readouts and output corrections from these sources.
