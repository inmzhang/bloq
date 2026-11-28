# Reproduce the spatial Hadamard results

The four inputs are the straight X and Y walls with a ZXZ input and the
perpendicular X/Y elbows with ZZX and XXZ hubs. Each carries one logical qubit
between temporal ports. All experiments compile the BLOG source with the public
Bloq Python API. The construction and detector slices are in
[the circuit guide](../../docs/circuit-constructions.md#spatial-hadamard).

## Monte Carlo

Run from the repository root. The published sweep used Python 3.13.18, Stim
1.16.0, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, and the PyMatching 2.4.0
revision pinned below. The tools and their dependencies are only needed to
reproduce the figures and certificates.

```sh
just py-develop
uv run --project bloq_py --with stim==1.16.0 --with numpy==2.5.1 --with scipy==1.18.0 --with matplotlib==3.11.1 \
  --with 'pymatching @ git+https://github.com/inmzhang/PyMatching.git@de4bb3e0796c1c9873d3ed5d1704364db10592cb' \
  python tools/spatial_hadamard/simulate.py --jobs 4
```

The command samples 216 circuits and writes
[the CSV and figure](../../docs/assets/experiments). Both ordinary and correlated
matching score the same shots. The CSV records the failure mask counts for both
port observables, fixed shot budget, seed, 95% Wilson limits and SHA-256 of the
native Stim text. Stim sampling can depend on the version, CPU instruction set,
and batch size. Use `--plot-only` to redraw the published data and `--output` to
choose another results directory.

## Exact circuit distance

Check out [Tropidist](https://github.com/Fanerst/tropidist) and expose its
`cdist` package to the verifier. The default command recompiles all 24 layouts
and distances at physical noise 0.001 and replays the committed
[certificates](../../docs/assets/experiments/spatial-hadamard-certificates.zip).

```sh
git clone https://github.com/Fanerst/tropidist.git target/tropidist
git -C target/tropidist checkout b03ec2d4077f287f89f97b5c2c16ce5ce17c2c58
PYTHONPATH="$PWD/target/tropidist" uv run --project bloq_py --with stim==1.16.0 --with numpy==2.5.1 \
  python tools/spatial_hadamard/verify_corpus.py --jobs 4 --sweep docs/assets/experiments/spatial-hadamard-rates.csv
```

The verifier also checks the physical resets, five entangling moments, nearest-neighbor
couplers and readout in each round, the four-detector raw fault bound, and strict matching decomposition. With `--sweep` it compares the compiler's Stim
output with the recorded circuit hashes across the simulation grid.

Pass `--generate --archive /path/to/new.zip` to discover certificates afresh.
For each observable, the verifier combines detector parity checks and adds a
detector parity to the logical bit. This preserves every undetected error that
flips that bit. Tropidist's rank-trellis dynamic program computes a lower bound
on this relaxed problem using every column of the original undecomposed detector
error model. A graphlike search proposes a physical witness. The verifier checks
its original fault columns, injects those faults into the native circuit, and
requires zero detection events, a logical flip, and equal lower and upper bounds.

### Signed walls and junctions

The [extended corpus](../../docs/assets/experiments/spatial-hadamard-junctions.tar.xz)
contains 117 BLOG inputs and 468 noisy Stim circuits with individual Tropidist
certificates. It covers signed elbows, separated walls, temporal input/output
variants, three-arm junctions, four-wall crosses and rotated gallery circuits.
Each input is compiled at distances 3, 5, 7 and 9. The Rust integration harness
uses `emit_bloq_stim_with_stim_noise` and Stim's `UniformDepolarizing` at 0.001.
The manifest records the number of concurrent pipes and the hash of each circuit.

```sh
mkdir -p target/spatial-wall-junctions
tar -xJf docs/assets/experiments/spatial-hadamard-junctions.tar.xz -C target/spatial-wall-junctions
PYTHONPATH="$PWD/target/tropidist" uv run --project bloq_py --with stim==1.16.0 --with numpy==2.5.1 \
  python tools/spatial_hadamard/verify_corpus.py --junctions target/spatial-wall-junctions
```

## Detector slices

The panels use Stim's detector slices with operations on the complete noiseless
distance-five circuits. Each row shows one syndrome round from preparation through
readout. The mixed circuit has an empty loop-boundary tick before its final round.

```sh
uv run --project bloq_py --with stim==1.16.0 python tools/render_construction_slices.py --only spatial-h
uv run --project bloq_py --with stim==1.16.0 python tools/render_construction_slices.py --only mixed-h
uv run --no-project --python 3.13 --with resvg-py --with pillow --with stim==1.16.0 python tools/spatial_hadamard/render_rounds.py
```
