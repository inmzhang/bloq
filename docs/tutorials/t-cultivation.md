# T State Cultivation

This tutorial shows how to compile an isolated T block and assess its prepared
state. Use the state in a [Logical T Gate](t-gate.md) for gate teleportation.

## Build and compile

The source needs one T block and one output port. Its pipe carries the prepared
patch forward in time:

```{literalinclude} ../examples/tutorial_t_cultivation.py
:language: python
:start-at: import bloq
:end-at: return program
```

```{bloq-view} t-source
:source: examples/viewers/t-source.blog
```

The IR retains the output patch and a cultivation retry region. Physical
postselection failures or either logical observable’s decoder `Flip` trigger a
retry. Whole-program Stim emission cannot represent this non-Clifford protocol.

## Cultivated state calibration

Add ten memory rounds after escape and growth to represent the decoding latency.
Clifft samples the physical T circuit, while its Clifford companion supplies PyMatching's detector error
model. Ideal terminal stabilizer measurements close the decoding problem.

We compute the corrected logical expectations for the accepted-state infidelity:

$$
q = \frac12\left(1-
\frac{\langle\overline X\rangle+\langle\overline Y\rangle}{\sqrt2}\right).
$$

Use Python 3.12 or newer and install Clifft and the GAP-enabled PyMatching build:

```sh
uv add "bloq-py==0.1.1" clifft "pymatching @ git+https://github.com/inmzhang/PyMatching.git@de4bb3e0796c1c9873d3ed5d1704364db10592cb"
```

The full script samples in batches and reports acceptance and infidelity for
each GAP threshold:

```{literalinclude} ../examples/tutorial_t_cultivation.py
:language: python
```

Download {download}`the script <../examples/tutorial_t_cultivation.py>` and run:

```sh
uv run python tutorial_t_cultivation.py
```

The default is 10,000 attempts. Use `--shots` to change the sample size.

## Simulation Results

At physical error probability $p=0.001$, each point below used 20 million
attempts. Infidelities are conditioned on acceptance. Uncertainties are nominal
pointwise 95% half-widths, in units of $10^{-6}$.

| Distance | GAP threshold | Acceptance | Accepted infidelity $(\times10^{-6})$ |
| --- | ---: | ---: | ---: |
| 5 | 12 | 31.319105% | $6.386 \pm 1.517$ |
| 7 | 11 | 39.447890% | $7.985 \pm 1.561$ |
| 9 | 11 | 40.051925% | $7.865 \pm 1.528$ |
| 11 | 11 | 40.120605% | $6.730 \pm 1.403$ |

```{figure} ../assets/experiments/t-cultivation-multi-p.svg
:alt: T-state acceptance versus accepted infidelity for five physical error rates and four distances.
```

For a [Logical T Gate](t-gate.md) or [CCZ factory](ccz-factory.md), retain the
source’s acceptance and full error model. Infidelity alone is insufficient.
