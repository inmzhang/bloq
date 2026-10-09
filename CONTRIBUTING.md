# Contributing

Start with the [source setup](docs/getting-started/installation.md#source-and-development-setup)
for the pinned Rust toolchain, Python environment, and development tools.

```sh
just check
just py-develop
```

Describe the behavior changed and the checks used to verify it. Breaking APIs
require coordinated changes to callers, examples, documentation, and release
notes. Preserve each release's documentation and API reference.

## Issues and pull requests

Use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md),
[feature request](.github/ISSUE_TEMPLATE/feature_request.md), or
[question](.github/ISSUE_TEMPLATE/question.md) template when opening an issue.
Pull requests use the [PR template](.github/pull_request_template.md); describe
the change and record the checks you ran. These templates are adapted from
[TQEC's issue templates](https://github.com/tqec/tqec/tree/main/.github/ISSUE_TEMPLATE)
and [PR template](https://github.com/tqec/tqec/blob/main/.github/pull_request_template.md).

## Contributor agreement

By submitting an issue or pull request, you confirm that you created the
contribution or have the right to submit it, and license it under this project's
[Apache-2.0 license](LICENSE). Check the contributor agreement in the template.

## AI-assisted contributions

AI-assisted contributions are welcome. Review all generated code, documentation,
issue text, and pull request text before submitting. You remain responsible for
the contribution's correctness, security, and quality. Disclose the tool and
model used, and how they assisted, in the issue or pull request.

Select exactly one AI acknowledgement option in the template. If you did not use
AI tools, select `No`; no AI disclosure details or `Assisted-by:` trailers are
required. If you used AI tools, complete the disclosure and review the generated
content before selecting `Yes`.

For each AI-assisted commit, include an `Assisted-by:` git trailer identifying
the tool and model:

```text
Assisted-by: <tool> (<model>)
```

Keep the human contributor as the commit author; credit AI tools with
`Assisted-by:`, rather than `Co-authored-by:`. Complete the AI contribution
acknowledgement in the template. This policy is adapted from
[Clifft's contribution guidelines](https://github.com/unitaryfoundation/clifft/blob/main/docs/development/contributing.md).

## Documentation scope

Document supported behavior, APIs, usage, and reproducible benchmarks.
Keep investigation notes and experiment reports out of the public documentation.

## Verification

```sh
just ci-fast       # formatting, Python lint, and script tests; no project build
just fmt-check
just test <crate>  # clippy, nextest, and doctests; all features, locked
just py-test      # Python bindings
just py-lint      # Ruff: package, tests, examples, and repository tools
just py-typecheck # mypy: Python package and generated native stubs
just ci-python    # Python lint, types, tests, and docs, one extension build
just ci           # all native CI suites, including MSRV and Python docs
```

`ticit` is optimized in dev/test builds to keep physical execution checks fast.
Debug assertions and integer overflow checks remain enabled.

PRs run quick checks automatically. Maintainers approve the full checks selected
for the changed files; the required `CI` status passes when those checks succeed.
Manual CI runs check every suite. For website publication, see
[Website maintenance](docs/site-maintenance.md#publish-a-reviewed-build).

Use `just test-full` for ignored or slow tests in release mode, and `just fidelity`
for both physical Choi suites. Graph-level logical checks use QuiZX;
non-Clifford physical execution checks belong in `bloq_vm/tests`.
Prose-only edits need a diff and link review; changed Rust API examples need
doctests. Keep regression tests that protect behavior rather than private layout.

Python stubs are generated from Rust annotations with `just py-stub`; do not
edit them by hand. Build the Python reference with `just py-docs`.
Ruff and mypy use locked dependencies managed by uv. Ruff runs without building
the Python extension.

## Extending Bloq

| Change | Start here |
| --- | --- |
| Source geometry, actions, or modules | [Block graphs](docs/graphs/concepts.md), [BLOG](docs/graphs/blog.md), and [modules](docs/modules/index.md) |
| Readout or compiler behavior | [Correlation surfaces](docs/theory/correlation-surfaces.md) and [compilation](docs/theory/compilation.md) |
| IR, emission, or execution | [Bloq IR](docs/backends/ir.md), [Backend Emission](docs/backends/emission.md), and [VM](docs/backends/vm.md) |
| Performance changes | [Benchmarking](docs/development/benchmarks.md) |

New gallery entries need an expected QuiZX map in
`bloq_graph/tests/verify_gallery.rs` and a shared fixture in `bloq_test`.
Physical and Stim coverage remains data-driven from that corpus. Keep
paper-specific experiment settings outside compiler and VM defaults.

Use Conventional Commits when making a commit. Publication is a separate
maintainer action covered by [releasing](docs/releasing.md).
