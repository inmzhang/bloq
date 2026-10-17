# Benchmarking

Measure compilation or Stim emission with the shared workloads before adding a
new benchmark. Workloads live in crate `benches/` and `examples/` directories
and `bloq_test::benchmark`; measurements and profiles stay local.

## Choose a workload

| Command | Measures |
| --- | --- |
| `just bench` | Combined compilation suite |
| `just bench-stim-emit` | Combined Stim emission suite |
| `just bench-cases cube_line` | One matching fixture in both stages |
| `just bench-case-list` | Available fixture slugs |
| `just bench-all` | All Criterion groups |

```sh
just bench-case-list
just bench-cases cube_line
```

:::{tip}
Criterion output can contain old workloads. Use a fresh `CARGO_TARGET_DIR` for
an isolated run.
:::

## Compare a change

Keep the workload, code distance, feature flags, and resource settings fixed.
Use separate Cargo target directories for different revisions so one build
does not replace the other.

| Record | Why it matters |
| --- | --- |
| Source revision and executable hash | Identifies the implementation actually timed |
| OS, CPU, and memory | Makes the host conditions explicit |
| Workload, width, distance, and limits | Defines the measured problem |
| Cold or warm compilation | Distinguishes initial work from cache reuse |
| Sample variation | Shows whether a difference exceeds measurement noise |

A structural node count can confirm that a workload ran. It does not prove two
physical programs implement the same instrument. Preserve semantic tests when
optimizing compilation.

## Profile a slow case

```sh
# Linux: requires perf and flamegraph
just profile-case cube_line compile 11

# Requires samply
just profile-case-samply cube_line stim 11
```

Profiles are written under `target/benchmark/profiles/<stage>/<case>-d<distance>/`.
Use function profiles to locate a cost: a compilation progress stage can contain
several operations.

## Sweep adder size and distance

```sh
env BLOQ_ADDER_BITS=3,10,32 BLOQ_ADDER_DISTANCES=3,5,9 \
  cargo bench -p bloq_compile --bench adder_scaling
```

Large cases may reach a compiler resource cap. Keep
those refusals in the results instead of dropping difficult samples.
