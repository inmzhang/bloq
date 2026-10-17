# Temporary Bevy image security backport

Copied from the published `bevy_image 0.20.0` crate, whose SHA-256 is
`bb818dd8d705809d54adea08e346edce81a51ddbf06406ddf12a72ee467a3855`.
The Rust sources and licenses are unchanged. Only the normalized `Cargo.toml`
changes: `ruzstd` requires `0.9.1` instead of `0.8.0` to fix
[RUSTSEC-2026-0344](https://rustsec.org/advisories/RUSTSEC-2026-0344.html).

This follows [Bevy's accepted dependency bump](https://github.com/bevyengine/bevy/pull/25753),
with the minimum raised to the fixed patch release. Cargo packaging metadata and
the crate's own lockfile are omitted.

Remove this directory, the workspace exclusion, and the `.cargo/config.toml` patch
entry when a compatible Bevy release permits `ruzstd >=0.9.1`.
The upstream backport is tracked in [Bevy #26206](https://github.com/bevyengine/bevy/issues/26206).
