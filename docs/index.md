---
hide-navigation: true
---

# bloq

```{raw} html
<img class="bloq-intro-logo" src="_static/bloq-logo-light.png" alt="bloq: interlocking coral b and blue q above the wordmark" width="220" height="220">
```

<p class="bloq-intro-tagline">A compiler for fault-tolerant surface code computation.</p>

---

## What is bloq?

```{include} ../README.md
:start-after: <!-- bloq-intro-start -->
:end-before: <!-- bloq-intro-end -->
```

## Key features

:::{container} grid cards
```{include} ../README.md
:start-after: "## Feature highlights"
:end-before: "## Relationship with TQEC"
```
:::

## Quick example

Here we use bloq to compile a surface code logical T gate to Bloq IR.

Define its blocks, connections, and adaptive measurement in BLOG, and save this
source as `t-gate.blog`:

::::{container} bloq-example-layout
```{literalinclude} examples/t-gate.blog
:language: blog
```

:::{container} bloq-example-render
```{bloq-view} home-t-gate
:source: examples/t-gate.blog
```
:::
::::

Compile that graph at code distance 11 and inspect the resulting IR:

::::{md-tab-set}
:::{md-tab-item} Python
```{literalinclude} examples/t_gate.py
:language: python
:start-after: "# [example-start]"
:end-before: "# [example-end]"
```
:::
:::{md-tab-item} Rust
```{literalinclude} examples/rust/t_gate.rs
:language: rust
```
:::
:::{md-tab-item} CLI
```{literalinclude} examples/t_gate.sh
:language: sh
:start-after: "# [example-start]"
:end-before: "# [example-end]"
```
:::
::::

We get the following output:

```{literalinclude} examples/t_gate_stats.txt
:language: text
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Home

Home <self>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 3

User Guide <user-guide>
```

```{toctree}
:hidden:
:maxdepth: 2

Examples <gallery/index>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/cli
Python API <api/python>
api/rust
Rust Feature Flags <guide>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/contributing
development/benchmarks
releasing
site-maintenance
```
