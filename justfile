default:
    @just --list --justfile {{ justfile() }}

[group('security')]
audit:
    cargo audit

[group('development')]
build crate="": (lint crate)
    cargo build {{ if crate != "" { "-p " + crate } else { "" } }}

[group('development')]
check crate="":
    cargo check {{ if crate != "" { "-p " + crate } else { "--workspace" } }} --locked

[group('development')]
check-no-default:
    cargo check -p bloq --no-default-features --locked

# Reject undocumented or broken public Rust API docs.
[group('development')]
doc-check:
    RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc --workspace --all-features --no-deps --locked

# Check the inherited workspace MSRV by selecting the facade by name.
[group('development')]
msrv:
    cargo +"$(just _msrv-version)" check --workspace --all-targets --all-features --locked

# Read the workspace MSRV inherited by the facade.
[group('development')]
_msrv-version:
    @cargo metadata --no-deps --format-version 1 | jq --raw-output '.packages[] | select(.name == "bloq") | .rust_version'

[group('development')]
clean:
    cargo clean

# show test coverage (requires https://lib.rs/crates/cargo-llvm-cov)
[group('development')]
coverage:
    PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')" cargo llvm-cov nextest --open

# show dependencies of this project
[group('development')]
deps:
    cargo tree

# combined-suite compile benchmarks; the default filter keeps the fine-grained
# `compile-case/*` groups out (run those via `just bench-cases` or `bench-all`)
[group('development')]
bench filter="^compile/" case_query="":
    BLOQ_TEST_CASE_QUERY="{{ case_query }}" cargo bench -p bloq_compile --bench bench {{ if filter != "" { "-- " + filter } else { "" } }}

[group('development')]
bench-stim-emit filter="^backend/" case_query="bench-core":
    BLOQ_TEST_CASE_QUERY="{{ case_query }}" cargo bench -p bloq_stim --bench bench {{ if filter != "" { "-- " + filter } else { "" } }}

[group('development')]
bench-list filter="" case_query="":
    cargo run -q -p xtask -- list-scenarios {{ if filter != "" { "--filter " + filter } else { "" } }} {{ if case_query != "" { "--case-query " + case_query } else { "" } }}

# run the fine-grained per-case benchmark groups (one group per fixture);
# narrow with e.g. `just bench-cases cube_line`
[group('benchmark')]
bench-cases fixture="":
    cargo bench -p bloq_compile --bench bench -- "compile-case/{{ fixture }}"
    cargo bench -p bloq_stim --bench bench -- "backend-case/{{ fixture }}"

# run everything: combined suites plus all per-case groups
[group('benchmark')]
bench-all:
    cargo bench -p bloq_compile --bench bench
    cargo bench -p bloq_stim --bench bench

# list the case slugs used in per-case benchmark IDs and profile paths
[group('benchmark')]
bench-case-list:
    cargo run -q -p xtask -- list-cases --compile-ready --slugs

# perf-record one case (slug from `just bench-case-list`, or `all`) and emit
# target/benchmark/profiles/<stage>/<case>-d<distance>/{flamegraph.svg,report.txt}
[group('benchmark')]
profile-case case="all" stage="compile" distance="11" duration="5" warmup="2":
    RUSTFLAGS="-Cforce-frame-pointers=yes" cargo build --profile profiling -p bloq_stim --example profile_case
    mkdir -p target/benchmark/profiles/{{ stage }}/{{ case }}-d{{ distance }} tmp/perf
    perf record -e cpu-clock -F 999 -g --call-graph fp -o tmp/perf/{{ stage }}-{{ case }}-d{{ distance }}.perf -- target/profiling/examples/profile_case --stage {{ stage }} --case {{ case }} --distance {{ distance }} --duration-secs {{ duration }} --warmup {{ warmup }}
    flamegraph --perfdata tmp/perf/{{ stage }}-{{ case }}-d{{ distance }}.perf -o target/benchmark/profiles/{{ stage }}/{{ case }}-d{{ distance }}/flamegraph.svg --title "{{ stage }} {{ case }} d{{ distance }}"
    perf report --stdio --no-children --sort=symbol -i tmp/perf/{{ stage }}-{{ case }}-d{{ distance }}.perf --percent-limit 0.5 > target/benchmark/profiles/{{ stage }}/{{ case }}-d{{ distance }}/report.txt

# record a samply profile for one case (view with `samply load <file>`)
[group('benchmark')]
profile-case-samply case="all" stage="compile" distance="11" duration="5" warmup="2":
    RUSTFLAGS="-Cforce-frame-pointers=yes" cargo build --profile profiling -p bloq_stim --example profile_case
    mkdir -p target/benchmark/profiles/{{ stage }}/{{ case }}-d{{ distance }}
    samply record --save-only -o target/benchmark/profiles/{{ stage }}/{{ case }}-d{{ distance }}/profile.json.gz -- target/profiling/examples/profile_case --stage {{ stage }} --case {{ case }} --distance {{ distance }} --duration-secs {{ duration }} --warmup {{ warmup }}

# profile every per-case workload plus the combined suite for one stage
[group('benchmark')]
profile-all stage="compile" distance="11" duration="5":
    #!/usr/bin/env bash
    set -euo pipefail
    for slug in $(cargo run -q -p xtask -- list-cases --compile-ready --slugs) all; do
        just profile-case "$slug" {{ stage }} {{ distance }} {{ duration }}
    done

# regenerate target/benchmark/index.html and target/benchmark/data.json from Criterion
# output and recorded profiles
[group('benchmark')]
bench-report:
    cargo run -q -p xtask -- bench-report

# full pipeline: all benchmarks, all profiles (both stages), then the report
[group('benchmark')]
bench-full: bench-all (profile-all "compile") (profile-all "stim") bench-report

[group('development')]
test-case-list filter="":
    cargo run -q -p xtask -- list-cases {{ if filter != "" { "--filter " + filter } else { "" } }}

[group('development')]
docs crate="":
    cargo doc --no-deps --open {{ if crate != "" { "-p " + crate } else { "--workspace --exclude bloq_editor" } }}

[group('development')]
fmt:
    cargo fmt

[group('development')]
fmt-check:
    cargo fmt --all -- --check

# lint the sources
[group('development')]
lint crate="":
    cargo clippy {{ if crate != "" { "-p " + crate } else { "--workspace" } }} --all-targets --all-features --tests --locked -- -D warnings

[group('development')]
lint-autocorrect crate="":
    cargo clippy --fix {{ if crate != "" { "-p " + crate } else { "" } }} --all-targets --all-features --tests --allow-dirty "$@"

# detect undefined behavior with miri (requires https://github.com/rust-lang/miri)
[group('security')]
miri:
    cargo +nightly miri test

# check, full test matrix, lint, miri
[group('development')]
pre-release: check test-full lint audit miri

[group('production')]
release: pre-release
    cargo build --release

# lint + nextest + doctests, all features and locked. Test bloq_py separately
# to avoid sibling profiling features entering its cdylib through unification.
# Its rustdoc is Python docstring source; the empty doctest pass detects accidental
# Rust code blocks. Binary-only crates have no doctest target.
[group('development')]
test crate="": (lint crate)
    PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')" cargo nextest run {{ if crate != "" { "-p " + crate } else { "--workspace --exclude bloq_py" } }} --all-features --locked --no-fail-fast
    {{ if crate == "" { "PYTHONHOME=\"$(python3 -c 'import sys; print(sys.base_prefix)')\" cargo nextest run -p bloq_py --all-features --locked --no-fail-fast" } else { ":" } }}
    {{ if crate == "bloq_editor" { "echo 'skipping Rust doctests (binary-only crate)'" } else if crate == "xtask" { "echo 'skipping Rust doctests (binary-only crate)'" } else if crate == "" { "cargo test --doc --workspace --exclude bloq_py --all-features --locked" } else { "cargo test --doc -p " + crate + " --all-features --locked" } }}
    {{ if crate == "" { "cargo test --doc -p bloq_py --all-features --locked" } else { ":" } }}

# Run every native CI suite locally; Actions gives each its own job and budget.
[group('development')]
ci: ci-rust msrv ci-python

[group('development')]
ci-rust: fmt-check licenses-check editor-package-check release-build-check check check-no-default doc-check test

# Tests and docs share one editable extension build and Python environment.
[group('python')]
ci-python: py-lint py-typecheck py-docs
    uv run --project bloq_py --no-sync pytest bloq_py/tests

# release build for the full matrix, including slow Stim distance searches
[group('development')]
test-full:
    PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')" cargo nextest run --release --workspace --exclude bloq_py --all-features --locked --run-ignored all --no-fail-fast
    PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')" cargo nextest run --release -p bloq_py --all-features --locked --run-ignored all --no-fail-fast

# Exact Choi checks for small logical maps and state preparations.
# Larger stabilizer instruments live in verify_choi; graph oracles remain independent.
# Run both physical Choi suites, including ignored CCZ factories and the
# large native instrument sweep.
[group('development')]
fidelity:
    cargo nextest run --release -p bloq_vm --test verify_exact_fidelity --all-features --locked --run-ignored all --no-fail-fast
    cargo nextest run --release -p bloq_vm --test verify_choi --all-features --locked --run-ignored all --no-fail-fast

[group('development')]
upgrade:
    cargo upgrade

[group('development')]
machete:
    cargo machete

[group('development')]
cloc:
    cloc bloq bloq_cli bloq_circuit bloq_compile bloq_vm bloq_editor bloq_graph bloq_ir bloq_stim bloq_test bloq_utils bloq_py xtask docs --exclude-ext svg --exclude-dir=.venv

# build + install bloq (editable) into the uv-managed bloq_py/.venv,
# with locked dev dependencies
[group('python')]
py-develop:
    uv sync --project bloq_py --group dev --locked

[group('python')]
py-lint: py-develop
    uv run --project bloq_py --no-sync ruff check --config bloq_py/pyproject.toml bloq_py/python bloq_py/tests bloq_py/examples tools bloq_editor/desktop

[group('python')]
py-typecheck: py-develop
    uv run --project bloq_py --no-sync mypy --config-file bloq_py/pyproject.toml

[group('python')]
py-build:
    uv run --no-project python tools/release_build.py uv build bloq_py --wheel --out-dir target/wheels -C maturin.build-args=--locked

[group('python')]
py-sdist:
    uv build bloq_py --sdist --out-dir target/wheels

# Preview only; release-plz writes CHANGELOG.md in the release PR.
[group('production')]
changelog:
    uvx --from git-cliff git-cliff --config cliff.toml --unreleased

# Generate annotated stubs and strip trailing whitespace. PYTHONHOME locates
# stdlib on relocatable Python installs; perl supports both BSD and GNU hosts.
[group('python')]
py-stub:
    PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')" cargo run -p bloq_py --bin stub_gen
    find bloq_py/python/bloq -name '*.pyi' -exec perl -pi -e 's/[ \t]+$//' {} +

[group('python')]
py-test: py-develop
    uv run --project bloq_py --no-sync pytest bloq_py/tests

# build the Python API reference: reread autodoc after rebuilding Rust docstrings
[group('python')]
py-docs:
    uv sync --project bloq_py --group docs --locked
    uv run --project bloq_py --no-sync sphinx-build -E -W --keep-going -d target/doctrees -b html docs target/site/docs/dev

# Assemble the homepage, versioned docs, both APIs, and the browser editor.
[group('website')]
site *args:
    python3 tools/site.py build {{ args }}

# Fast documentation preview; full `site` is the publication gate.
[group('website')]
site-docs:
    python3 tools/site.py build --docs-only

[group('website')]
site-check:
    python3 tools/test_site.py
    python3 tools/site.py check

[group('website')]
site-preview:
    python3 -m http.server 8000 --directory target/site

[group('editor')]
editor:
    cargo run -p bloq_editor

# Package the native editor with its desktop icon (macOS .app or Linux launcher).
[group('editor')]
editor-package:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 tools/release_build.py cargo build -p bloq_editor --release --locked
    editor_target=$(rustc -vV | sed -n 's/^host: //p')
    python3 bloq_editor/desktop/package.py build target/release/bloq_editor "$editor_target"

[group('editor')]
editor-icons:
    uv run --with pillow bloq_editor/assets/icons/generate.py

[group('editor')]
editor-package-check:
    python3 bloq_editor/desktop/package.py check

[group('development')]
release-build-check:
    uv run --no-project python tools/test_release_build.py
    python3 tools/post_release.py --check

# Run on a reviewed chore/ branch after all uploads for the stable tag succeed.
[group('production')]
start-dev:
    python3 tools/workspace_version.py dev

# Adjust a release candidate without rewriting previously released changelogs.
[group('production')]
set-version version:
    python3 tools/workspace_version.py set {{ quote(version) }}

[group('development')]
licenses-check:
    cargo fetch --locked
    uv run --no-project python tools/third_party_licenses.py --check

[group('editor')]
web:
    bevy run -p bloq_editor --bin bloq_editor web --bundle --open
