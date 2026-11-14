[group('production')]
release: pre-release
    cargo build --release

# Preview only; release-plz writes CHANGELOG.md in the release PR.
[group('production')]
changelog:
    uvx --from git-cliff==2.14.2 git-cliff --config cliff.toml --unreleased

# Run on a reviewed chore/ branch after all uploads for the stable tag succeed.
[group('production')]
start-dev:
    python3 tools/workspace_version.py dev

# Adjust a release candidate without rewriting previously released changelogs.
[group('production')]
set-version version:
    python3 tools/workspace_version.py set {{ quote(version) }}
