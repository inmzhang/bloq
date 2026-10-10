# Changelog

## [0.2.0] - 2026-10-10

### Changed

- **BREAKING:** Migrate editor to Bevy 0.20 ([#11](https://github.com/inmzhang/bloq/pull/11))
- Streamline checks and automate development bumps ([#5](https://github.com/inmzhang/bloq/pull/5))
- **BREAKING:** Remove Hotpath profiling ([#12](https://github.com/inmzhang/bloq/pull/12))


### Fixed

- Prevent middle-click paste ([#8](https://github.com/inmzhang/bloq/pull/8))


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
