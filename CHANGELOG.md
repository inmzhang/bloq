# Changelog

## [0.1.1]

- Remove the `bloq_lassynth` crate, the optional `bloq::synth` API, and the
  `bloq synth` command. Graph construction, compilation, and VM execution remain
  available.
- Remove native SAT solver dependencies that blocked Linux Python wheels and
  Windows builds.
- Enable deterministic C/C++ path remapping on Windows and check Windows
  CLI/editor builds before release.

## [0.1.0]

First public release of Bloq, a dynamic Clifford+T surface code compiler. APIs and
artifact layouts may change before `1.0.0`. This release includes:

- `bloq` rust crates
- `bloq` python package
- `bloq` CLI
- `bloq-editor` desktop binaries and web builds
