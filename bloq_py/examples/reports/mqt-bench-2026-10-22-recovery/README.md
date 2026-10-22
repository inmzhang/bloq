# MQT Bench compilation recovery catalog — 2026-10-22

**Private experiment evidence:** the supporting bundles are retained locally and
are not publicly hosted.

This catalog contains 71 validated compiler–circuit pairs across 35 cases and
24 families: SpiderLS 35, TopoLS 28, Topologiq 8. Programs compile at requested
d=25 with ideal T-source proxies, native validation and binary round trip.
Exactly 26 selected artifacts retain sampled QuiZX + d=3 VM checks; 3 more
retain sampled VM passes with inconclusive logical checks. Fresh recovery
layouts are compilation-only. Spatial-Hadamard warnings can indicate lower
effective distance than requested.

The PDF and 13 evidence archives are local-only and supplied separately.
Git retains this README, `.gitignore`, and [SHA256SUMS](SHA256SUMS).
The machine-local backup inventory records where the supplied bundles are stored.

The 70-page `report.pdf` links raw/lowered QASM, BLOG, d=25 IR, 3D views and
provenance. Tall layouts include a true-proportion overview and a labeled
magnified detail with equal axes. Interactive viewers load their runtime from
the Google CDN. The packaged `report-summary.json` and `artifacts.json` are
available after extraction. Frozen R1/R2 cohorts retain separate sources,
audits and outcomes; semantic evidence never transfers across changed BLOG hashes.

## Verify and restore the local evidence

From the repository root, set the supplied bundle location and verify every
checksum before extraction:

```sh
report_checksums="$PWD/bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/SHA256SUMS"
report_bundle="/path/to/supplied/mqt-bench-2026-10-22-recovery"
(cd "$report_bundle" && sha256sum -c "$report_checksums")
```

Extract all 13 parts into a separate, empty evidence directory. Its original
packaged README belongs to the frozen `artifacts.json` hash set and must remain
unchanged; this repository README is a local-bundle index. This catalog is
self-contained and does not require the 2026-10-21 archives.

```sh
report_evidence="/path/to/separate/mqt-bench-2026-10-22-recovery"
mkdir -p "$report_evidence"
for report_archive in "$report_bundle"/artifacts-*.tar.zst; do
    tar --zstd -xf "$report_archive" -C "$report_evidence"
done
python3 bloq_py/examples/mqt_bench_report.py "$report_evidence" --verify
```

Open `report.pdf` inside the extracted evidence directory so its relative links
resolve. Packaging was independently checked: 1,764 artifact hashes, 1,765 local
paths and 598 PDF URIs. These are integrity checks, not large-circuit execution
or full-channel proofs. `pipeline_seconds` begins after Clifford+T lowering
and includes routing through validated IR and binary round trip; source generation
and synthesis are recorded separately. No optimal-packing claim is made.
