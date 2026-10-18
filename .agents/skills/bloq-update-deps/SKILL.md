---
name: bloq-update-deps
description: Perform routine dependency updates for the Bloq project using its Renovate dashboard, upstream migration guides, focused code improvements, and compatibility checks. Use when asked to refresh or upgrade Bloq dependencies.
---

# Bloq dependency updates

Work from the current Bloq checkout root. Follow the checkout's `AGENTS.md`, `CONTRIBUTING.md`, and current `justfile`/CI workflows.

Never patch, vendor, fork, or backport third-party dependencies. Resolve dependency issues only through released upstream updates or supported configuration changes; otherwise defer the update and report the blocker. Do not suppress security findings merely to make CI pass.

For audit failures, inspect the advisory, resolved features on every supported target, and affected upstream call sites. If evidence shows the vulnerable path is unreachable, use a temporary advisory-specific exception in `.cargo/audit.toml`, documenting reviewed versions/features, rationale, and the upstream removal condition. Keep this configuration-only; do not add dedicated exception-checking scripts. If safety is uncertain, defer or revert the affected update instead. Report pre-existing advisories separately: reverting an unrelated update will not fix them. Reassess every exception during dependency or feature changes, and run `just audit`.

1. **Refresh the queue.** First read [Dependency Dashboard #9](https://github.com/inmzhang/bloq/issues/9), including comments and linked update PRs: `gh issue view 9 --repo inmzhang/bloq --json title,body,comments,url`. Compare candidates with current manifests, lockfiles, toolchain, and open PRs; skip already-applied updates and avoid duplicating existing work. If the dashboard is unavailable, report the blocker instead of treating cached versions as current. Preserve local edits and work on a `chore/` or other repository-approved feature branch, never directly on `main`.

2. **Review each update before editing.** Read official release notes/changelogs and migration guides across the installed-to-target version range, including breaking pre-1.0 releases. If release notes are absent, review upstream commits or published-source diffs. Inspect affected callers, features, target restrictions, MSRV, and companion dependencies. Record source links, required migrations, and useful new capabilities. Keep compatible families together: Bevy, `bevy_egui`, `egui_extras`, text/render plugins, and Bevy's `glam`; likewise PyO3 and its stub generator. Defer incompatible candidates with a concrete reason. If an update requires changing an architectural invariant, explain the proposed architecture change and wait for human confirmation.

3. **Apply the smallest complete update.** Process independent dependencies or compatible groups separately. Update manifests and lockfiles with targeted package-manager commands; inspect transitive churn. Migrate callers, tests, examples, configuration, and generated artifacts as needed. Adopt new APIs when they directly simplify existing code, fix a problem, or improve existing behavior; list larger opportunities separately. Preserve supported behavior and add focused regression coverage for nontrivial migrations. Generate Python stubs with `just py-stub`; never hand-edit them or `CHANGELOG.md`.

4. **Verify each group, then the combined result.** Use current repository recipes and locked resolution. Run `just audit`; report unresolved advisories explicitly. Run affected crate tests via `just test <crate>` and relevant downstream tests; run `just ci-fast` and applicable native CI suites before finishing. Check MSRV for Rust dependency/toolchain changes and license inventory for lockfile changes (`just licenses-check`; regenerate with the repository tool when needed). Use `just ci-python` for Python/PyO3 changes. For editor-related dependencies, require all three:

   - `just test bloq_editor`
   - `cargo build --locked --release -p bloq_editor`
   - `bevy build --locked --release --yes -p bloq_editor --bin bloq_editor web --bundle` (follow current `tools/site.py` and site CI flags/tooling).

   Also smoke-test affected rendering, input, file handling, and compile-worker behavior on desktop and browser when available. Build success alone does not establish runtime compatibility. Report unavailable platforms/checks as unverified; do not silently waive them.

5. **Close with evidence.** Review the final diff for unrelated changes. Summarize old→new versions, upstream links, migrations/improvements, exact checks and outcomes, and deferred updates or remaining risks. Stop when the selected updates pass their applicable checks or have explicit blockers. If publishing a PR, inspect its checks and distinguish failures from pending maintainer approvals. Follow repository contribution rules if committing or opening a PR; merging, publication, dashboard edits, and scheduling are separate requests.
