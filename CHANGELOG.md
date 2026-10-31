# Changelog

## [0.2.0] - 2026-10-31

### Dependencies

#### Changed

- **BREAKING:** Migrate editor to Bevy 0.20 ([#11](https://github.com/inmzhang/bloq/pull/11)) ([a42f838](https://github.com/inmzhang/bloq/commit/a42f8387d4eff11491e38fc3cecaab617bb802cc))
  Source builds require Rust 1.98.0, and public Rust
  geometry types use glam 0.33. Documentation builds require Python
  3.11 or newer; the Python package still supports Python 3.10.
- Add update workflow and refresh dependencies ([#22](https://github.com/inmzhang/bloq/pull/22)) ([ab714a2](https://github.com/inmzhang/bloq/commit/ab714a21710efd9072c5d28c80d5b4191a823b99))

### Documentation

#### Changed

- Require concrete AI model attribution ([#10](https://github.com/inmzhang/bloq/pull/10)) ([c612adb](https://github.com/inmzhang/bloq/commit/c612adb1fc9166ca3419190a0af08138f35bdb53))
- Reserve changelog updates for release-plz ([#14](https://github.com/inmzhang/bloq/pull/14)) ([092615d](https://github.com/inmzhang/bloq/commit/092615d50bf1147451b1b4d64b198fa93ad426a5))
- Require runtime model attribution before publishing ([#21](https://github.com/inmzhang/bloq/pull/21)) ([20cee05](https://github.com/inmzhang/bloq/commit/20cee0546e0d855313da632dec29b496e88d8017))

#### Fixed

- Restore search with compatible Sphinx ([#35](https://github.com/inmzhang/bloq/pull/35)) ([423f45b](https://github.com/inmzhang/bloq/commit/423f45b69d176717355d4f2b86860887bccfd4e9))
- Discover Rustdoc link targets from generated pages ([#36](https://github.com/inmzhang/bloq/pull/36)) ([733f1a8](https://github.com/inmzhang/bloq/commit/733f1a88aedcd2b3eea68fc0053c7981d7d60718))

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

#### Added

- **BREAKING:** Support reversed temporal ports and signed Bell outcomes ([#27](https://github.com/inmzhang/bloq/pull/27)) ([3593899](https://github.com/inmzhang/bloq/commit/3593899fae6574352d03ee306a6bcbc57e89486b))
  InstanceProvenance includes IdealPortAdapter. Downstream
  exhaustive matches must handle ideal temporal boundary instances.
- Compose noiseless VM execution and output capture ([#29](https://github.com/inmzhang/bloq/pull/29)) ([8ad7f8b](https://github.com/inmzhang/bloq/commit/8ad7f8bf5701218c2768b10ab82afe96ab256e36))

#### Changed

- **BREAKING:** Plan every correlation query and retire the flat stabilizer engine ([#31](https://github.com/inmzhang/bloq/pull/31)) ([11aad7c](https://github.com/inmzhang/bloq/commit/11aad7c5a852be6cbef7ad28eaa9664d8c0983e3))
  BlockGraph::analyze_actions and analyze_actions_with_limits
  return the analyzed graph instead of a (graph, StabilizerGenerators)
  pair. BlockGraph::analyze_actions_with_projections, AnalyzedBranchProjections,
  and InvalidActionError::{BranchDependentMeasurementSurface,
  GuardedAnalysisRequired} are removed. Validation rejects named records
  whose only correlation surface reaches a live output.
- **BREAKING:** Remove feedback inference and trim internals ([#26](https://github.com/inmzhang/bloq/pull/26)) ([aaec756](https://github.com/inmzhang/bloq/commit/aaec756636fadc6b732058f4d3c0a584698fe763))
  Remove infer_feedback, infer_feedback_with,
  FeedbackOptions, FeedbackInferenceError, bloq_utils::qasm, and the QASM
  parsing exports from bloq_graph::verify. Author Action::Feedback explicitly
  and pass expected QuiZX graphs to LogicalVerifier or verify_logical.
- Link planned surfaces to the public branch projection ([#33](https://github.com/inmzhang/bloq/pull/33)) ([394900a](https://github.com/inmzhang/bloq/commit/394900a62b800b994f4ee62a646d70e626d12aea))

#### Fixed

- Merge adjacent walking stages ([#19](https://github.com/inmzhang/bloq/pull/19)) ([be30ed8](https://github.com/inmzhang/bloq/commit/be30ed8d2714bf3cdb45994df04f63a5a955abb5))
- Reject X/Z basis flips of multiplex Ports ([#30](https://github.com/inmzhang/bloq/pull/30)) ([1502648](https://github.com/inmzhang/bloq/commit/150264850227d0c1395129cbcdef5f98ee1808cb))
- **BREAKING:** Read named merges before later unnamed merges ([#32](https://github.com/inmzhang/bloq/pull/32)) ([bdbbd66](https://github.com/inmzhang/bloq/commit/bdbbd66b98577e7f5e098cf7461a41edb6fc4441))
  BlockGraph::analyze_actions and analyze_actions_with_limits
  return the analyzed graph instead of a (graph, StabilizerGenerators)
  pair. BlockGraph::analyze_actions_with_projections, AnalyzedBranchProjections,
  and InvalidActionError::{BranchDependentMeasurementSurface,
  GuardedAnalysisRequired} are removed. Validation rejects named records
  whose only correlation surface reaches a live output.
- **BREAKING:** Certify causal readouts and protect live patch state ([#28](https://github.com/inmzhang/bloq/pull/28)) ([31ebfd4](https://github.com/inmzhang/bloq/commit/31ebfd460165d30d9675abdc2410aafc23a0d5b6))
  InstanceProvenance includes IdealPortAdapter. Downstream
  exhaustive matches must handle ideal temporal boundary instances.
- Preserve causal scheduling across guarded qubit lifetimes ([#37](https://github.com/inmzhang/bloq/pull/37)) ([4af85fc](https://github.com/inmzhang/bloq/commit/4af85fcdcfeaee931893b11d6dfa5fdffa198fdf))

#### Performance

- Compact flow and detector buffers ([#23](https://github.com/inmzhang/bloq/pull/23)) ([efe9d71](https://github.com/inmzhang/bloq/commit/efe9d719cc45cde75da3817ce584e8e17d20b13e))
- Scale module relations and readout compilation ([#25](https://github.com/inmzhang/bloq/pull/25)) ([90e40bb](https://github.com/inmzhang/bloq/commit/90e40bb7ecb4d88c6c2fde4979e660f71fff6cbc))
- Reduce repeated correlation-surface planning ([#38](https://github.com/inmzhang/bloq/pull/38)) ([2e29921](https://github.com/inmzhang/bloq/commit/2e299218fb56ce48a9199e2e588eb2f6e1cb3632))

### Workflows and packaging

#### Changed

- Streamline checks and automate development bumps ([#5](https://github.com/inmzhang/bloq/pull/5)) ([80cdaa9](https://github.com/inmzhang/bloq/commit/80cdaa91edbd673a3a0c599e52c159718720df3e))

#### Fixed

- Publish development docs on main ([#7](https://github.com/inmzhang/bloq/pull/7)) ([e80f8e3](https://github.com/inmzhang/bloq/commit/e80f8e30086d3494d202c09cd15c094062f04dbb))
- Include all commits in release notes ([#18](https://github.com/inmzhang/bloq/pull/18)) ([f7d3395](https://github.com/inmzhang/bloq/commit/f7d3395fe3e0255304ed8c611decff8e7ffce1c0))

#### Performance

- Reduce repeated builds and batch website publication ([#20](https://github.com/inmzhang/bloq/pull/20)) ([7e776ad](https://github.com/inmzhang/bloq/commit/7e776ad006302eeb1e9cddf9ded83186515691b0))


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
