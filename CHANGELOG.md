# Changelog

## [0.2.0] - 2026-10-11

### Dependencies

#### Changed

- **BREAKING:** Migrate editor to Bevy 0.20 ([#11](https://github.com/inmzhang/bloq/pull/11)) ([a42f838](https://github.com/inmzhang/bloq/commit/a42f8387d4eff11491e38fc3cecaab617bb802cc))
  Source builds require Rust 1.98.0, and public Rust
  geometry types use glam 0.33. Documentation builds require Python
  3.11 or newer; the Python package still supports Python 3.10.

### Documentation

#### Changed

- Require concrete AI model attribution ([#10](https://github.com/inmzhang/bloq/pull/10)) ([c612adb](https://github.com/inmzhang/bloq/commit/c612adb1fc9166ca3419190a0af08138f35bdb53))
- Reserve changelog updates for release-plz ([#14](https://github.com/inmzhang/bloq/pull/14)) ([092615d](https://github.com/inmzhang/bloq/commit/092615d50bf1147451b1b4d64b198fa93ad426a5))

### Editor

#### Fixed

- Prevent middle-click paste ([#8](https://github.com/inmzhang/bloq/pull/8)) ([aefd730](https://github.com/inmzhang/bloq/commit/aefd73018d298906c1e4c2ec5b7887f91461e573))
- Restore WebGPU and refresh dependencies ([#17](https://github.com/inmzhang/bloq/pull/17)) ([bdaea0d](https://github.com/inmzhang/bloq/commit/bdaea0d5514d7ca97dca80002812b9d0f46be072))

### General

#### Changed

- **BREAKING:** Remove Hotpath profiling ([#12](https://github.com/inmzhang/bloq/pull/12)) ([919b15b](https://github.com/inmzhang/bloq/commit/919b15b014209232ff55605117ad978c5131b204))
  Remove the hotpath and hotpath-alloc Cargo features.
  Use the existing perf or samply workflows for sampling profiles.

### Rust API and compiler

#### Fixed

- Merge adjacent walking stages ([#19](https://github.com/inmzhang/bloq/pull/19)) ([be30ed8](https://github.com/inmzhang/bloq/commit/be30ed8d2714bf3cdb45994df04f63a5a955abb5))

### Workflows and packaging

#### Changed

- Streamline checks and automate development bumps ([#5](https://github.com/inmzhang/bloq/pull/5)) ([80cdaa9](https://github.com/inmzhang/bloq/commit/80cdaa91edbd673a3a0c599e52c159718720df3e))

#### Fixed

- Publish development docs on main ([#7](https://github.com/inmzhang/bloq/pull/7)) ([e80f8e3](https://github.com/inmzhang/bloq/commit/e80f8e30086d3494d202c09cd15c094062f04dbb))
- Include all commits in release notes ([#18](https://github.com/inmzhang/bloq/pull/18)) ([f7d3395](https://github.com/inmzhang/bloq/commit/f7d3395fe3e0255304ed8c611decff8e7ffce1c0))


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
