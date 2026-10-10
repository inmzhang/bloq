# Rust Feature Flags

The `bloq` facade has no default features. Graph authoring, compilation, IR
inspection and editing, and ordinary Stim text emission are available in the
base library. Enable a feature when your application needs the corresponding
API; enabling one does not make compilation run extra verification automatically.

## Cargo feature flags

| Feature | Capability and scope |
| --- | --- |
| `gltf` | Export source geometry as glTF or an HTML viewer through `bloq::graph` |
| `graph-verify` | Verify source logical maps and infer feedback using QuiZX, without native Stim bindings |
| `verify` | Verify physical Clifford flows through `bloq::stim`; adds the native Stim dependency |
| `vm` | Lower and execute physical IR through `bloq::vm`, including adaptive control and non-Clifford verification |

For example, add the physical verification simulator to a Cargo project:

```sh
cargo add bloq@=0.1.1 --features vm
```

Combine features with a comma-separated list, such as `--features gltf,vm`.
Features are additive: enabling `vm` does not select it as a default backend,
and `verify` is not required to emit Stim text. Use `bloq::prelude::*` for common
graph and compilation types, and explicit modules such as `bloq::vm` for
specialized workflows. The [Rust API](api/rust.md) describes the entry points
and links to generated item documentation.

## Select the verification layer

Choose `graph-verify` when comparing an authored graph with an expected logical
map before physical compilation. Choose `verify` when checking Clifford flows in
physical circuits with native Stim. Choose `vm` when measurement-dependent
execution, retries, or corrected non-Clifford output states are the question.
These layers check different properties; none alone establishes noisy decoding
performance.

A full IR well-formedness audit through `Bloq::validate` is available without
these features. It checks the compiled representation rather than an expected
logical map or sampled execution. See
[logical correlations](theory/correlation-surfaces.md) and
[VM verification](backends/vm.md) for the verification models and assumptions.

## CLI and editor

These feature names belong to the Rust facade. The separate `bloq-cli` package
includes its compilation and viewing commands without additional
feature flags. The editor's normal build provides graph viewing, compilation,
and Stim text export. Its browser build uses WebGPU and does not include the
native Stim verification bindings or VM engine.

Use the [benchmarking guide](development/benchmarks.md) for reproducible
measurements and profiling with perf or samply.
