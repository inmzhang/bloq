# Bloq Agent Instructions

This file records repository-specific constraints that are easy to miss from
the code alone. Use [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
and `just --list` for orientation, setup, and routine checks.

## Source of Truth and Architecture Changes

Use the workspace crates, their tests, `docs/`, and `README.md` as the source
of truth. Inspect the current implementation and tests before resolving
architectural ambiguity.

If a proposed change contradicts an architectural invariant or the current
architecture cannot support it:

1. Stop instead of implementing a workaround or rewriting architecture-facing
   documentation to justify the change.
2. Explain the discrepancy and propose an explicit architectural change.
3. Wait for human confirmation before implementation.

## Source and Test Rules

- For `bloq_editor` changes, check both desktop and web targets.
- Generate Python stubs from Rust annotations with `just py-stub`; never edit
  generated stubs by hand.
- New gallery entries need an expected QuiZX map in
  `bloq_graph/tests/verify_gallery.rs` and a shared fixture in `bloq_test`.
  Keep physical and Stim coverage driven by that corpus.
- Use QuiZX for graph-level logical checks. Non-Clifford physical execution
  checks belong in `bloq_vm/tests`.
- Test observable behavior rather than private layout. Follow the verification
  commands in [CONTRIBUTING.md](CONTRIBUTING.md#verification); prose-only edits
  need a diff and link review, while changed Rust API examples need doctests.

## Contributions

- Never commit directly to `main`; use a feature branch named with a
  Conventional Commit type, such as `docs/update-guide`, `feat/add-backend`,
  or `fix/readout-order`.
- Coordinate breaking API changes across callers, examples, documentation,
  and release notes. Preserve released documentation and API references.
- Use Conventional Commits. Follow the repository's
  [AI contribution policy](CONTRIBUTING.md#ai-assisted-contributions), including
  tool/model disclosure and an `Assisted-by: <tool> (<model>)` commit trailer.
  Keep the human contributor as author; do not use AI `Co-authored-by:` credit.
- Keep the aggregate `CI` check required. Release and website publication are
  separate maintainer actions; follow [Releasing](docs/releasing.md) and
  [Website maintenance](docs/site-maintenance.md).
