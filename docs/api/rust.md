# Rust API Reference

The generated <a href="rust/bloq/index.html">Rust API reference</a> documents the
`bloq` facade, including item signatures, typed errors, and executable examples.
The website builds it from the same source as this documentation version. Use
this page to choose an entry point, then follow the generated reference for its
arguments and contracts.

## Namespaces

Start with `bloq::prelude::*` for graph authoring, compilation, and ordinary Stim
emission. Use explicit modules for specialized types. The facade re-exports the
workspace crates, so importing a type through `bloq::ir`, for example, does not
create a wrapper around `bloq_ir`.

| Namespace | Responsibility |
| --- | --- |
| `bloq::graph` | Source blocks, pipes, actions, module interfaces, BLOG I/O, and logical analysis |
| `bloq::compile` | Validated configuration, contexts, progress, template caches, and compiled objects |
| `bloq::ir` | Physical programs, graph traversal and edits, templates, regions, readouts, and codecs |
| `bloq::circuit` | Physical operations, coordinates, noise models, and Clifford flows |
| `bloq::stim` | Static emission, segments, and Stim/clifft text dialects |
| `bloq::utils` | Pauli algebra, geometry vocabulary, and shared supporting types |
| `bloq::vm` | Dynamic physical execution and verification; requires `vm` |

The [feature reference](../guide.md#cargo-feature-flags) lists the optional
capabilities. Ordinary graph authoring, compilation, IR access, and Stim text
emission require no features.

## Local API documentation

From a Cargo project that depends on `bloq`, build its enabled API with:

```sh
cargo doc -p bloq --no-deps --open
```

Enable the features you use in that project's dependency declaration before
building the docs. In the Bloq source tree, `just docs bloq` builds the facade
reference with its default features. To include every optional API from the
source tree, run `cargo doc -p bloq --all-features --no-deps --open`.
