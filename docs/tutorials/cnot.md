# Logical CNOT

This tutorial shows how to represent and compile a logical CNOT gate with Bloq,
then simulate and decode it with Stim and PyMatching.

Install the tutorial's sampling dependencies in your Python project:

```sh
uv add "bloq-py==0.1.1" stim pymatching
```

## Build the block graph

```{figure} ../assets/paper/preliminary-cnot-physical.svg
:alt: CNOT parity-measurement circuit with a plus-state ancilla, conditional X/Z corrections, and four lattice-surgery patch snapshots.
:width: 100%

Logical CNOT via $ZZ$ and $XX$ parity measurements, ancilla $Z$ readout,
and measurement-dependent Pauli corrections.
```

```{literalinclude} ../examples/tutorial_cnot.py
:language: python
:start-after: "# [build-start]"
:end-before: "# [build-end]"
```

```{bloq-view} cnot
```

## Inspect the correlation surfaces

For control $c$ and target $t$, CNOT transports the Pauli operators as follows:

| Input | Output |
| --- | --- |
| $X_c$ | $X_cX_t$ |
| $Z_c$ | $Z_c$ |
| $X_t$ | $X_t$ |
| $Z_t$ | $Z_cZ_t$ |

Compute a generating basis and inspect its external support:

```{literalinclude} ../examples/tutorial_cnot.py
:language: python
:start-after: "# [correlations-start]"
:end-before: "# [correlations-end]"
```

Generators can be combined, so a returned row need not match one table row
verbatim. The complete relation determines the map.

::::{md-tab-set}
:::{md-tab-item} X control → X control · X target
```{bloq-view} cnot
:surface: XXIX
```
:::
:::{md-tab-item} Z control → Z control
```{bloq-view} cnot
:surface: ZZII
```
:::
:::{md-tab-item} X target → X target
```{bloq-view} cnot
:surface: IIXX
```
:::
:::{md-tab-item} Z target → Z control · Z target
```{bloq-view} cnot
:surface: IZZZ
```
:::
::::

Read [Correlation Surfaces](../theory/correlation-surfaces.md) for the signed
algebra and decoder-corrected readout interpretation.

## Compile and emit

```{literalinclude} ../examples/tutorial_cnot.py
:language: python
:start-after: "# [compile-start]"
:end-before: "# [compile-end]"
```

`cnot.bloqir` retains the physical program; `cnot.svg` shows its dependencies.
`cnot.stim` is the static backend circuit. Its Ports are still ideal open
boundaries: compilation has not chosen physical input states or measurements.

The CLI can perform the corresponding gallery workflow:

```sh
bloq compile --gallery cnot -d 3 --backend ir-text -o cnot.bloqir
bloq emit cnot.bloqir -o cnot.stim
```

## Auto-fill a closed experiment

A block graph with open Ports represents a logical channel that can be simulated
with ideal boundaries. To run a complete experiment on quantum hardware, prepare
the input states and measure the outputs in chosen logical bases.

Automatic filling chooses compatible preparation/readout boundaries from the
source correlation relation. It returns closed graphs and the relations each
experiment measures. Select the first compatible variant here:

```{literalinclude} ../examples/tutorial_cnot.py
:language: python
:start-after: "# [fill-start]"
:end-before: "# [fill-end]"
```

## Sample and decode with noise

Apply a uniform depolarizing noise model with strength $p=10^{-3}$, then
sample detector events and decode with PyMatching:

```{literalinclude} ../examples/tutorial_cnot.py
:language: python
:start-after: "# [noise-start]"
:end-before: "# [noise-end]"
```

Download {download}`the complete runnable example <../examples/tutorial_cnot.py>`.
See [Stim's API reference](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)
for further sampling and detector-model options.

## Simulation Results

The plots show simulation results for four independent logical observables of
the open-port CNOT, labeled by their correlation surfaces.

```{figure} ../assets/experiments/cnot-fill0-L0.svg
:alt: Logical error probability for X control to X control times X target.
```

```{figure} ../assets/experiments/cnot-fill0-L1.svg
:alt: Logical error probability for X target to X target.
```

```{figure} ../assets/experiments/cnot-fill1-L0.svg
:alt: Logical error probability for Z control times Z target to Z target.
```

```{figure} ../assets/experiments/cnot-fill1-L1.svg
:alt: Logical error probability for Z target to Z control times Z target.
```

Continue with [Logical T Gate](t-gate.md) to add resource retries and
measurement-dependent basis selection.
