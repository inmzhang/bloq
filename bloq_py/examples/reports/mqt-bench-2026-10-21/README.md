# MQT Bench compilation catalog — 2026-10-21

**Private experiment evidence:** the supporting bundles are retained locally and
are not publicly hosted.

This historical catalog contains 65 accepted compiler–circuit pairs at requested
distance 25 across 31 cases and 24 families: SpiderLS 31, TopoLS 26, Topologiq 8.
It retains 26 sampled QuiZX + distance-3 VM checks and 3 additional VM passes
with inconclusive logical checks. Checks use ideal T sources; compilation and
sampled execution remain separate claims. Spatial-Hadamard warnings can indicate
lower effective distance than requested.

The PDFs and five evidence archives are local-only and supplied separately.
Git retains this README, `.gitignore`, and [SHA256SUMS](SHA256SUMS).
The machine-local backup inventory records where the supplied bundles are stored.

The main PDF is `report.pdf`; `native-reuse/comparison.pdf` records the reuse
comparison, and `compact-placement/report.pdf` retains the earlier addendum.
The bundle includes raw/lowered QASM, BLOG, d=25 IR, viewers, logs and provenance.
Its `report-summary.json` and `artifacts.json` are available after extraction.

## Verify and restore the local evidence

From the repository root, set the supplied bundle location and verify every
checksum before extraction:

```sh
report_checksums="$PWD/bloq_py/examples/reports/mqt-bench-2026-10-21/SHA256SUMS"
report_bundle="/path/to/supplied/mqt-bench-2026-10-21"
(cd "$report_bundle" && sha256sum -c "$report_checksums")
```

Extract into a separate, empty evidence directory, in the order below. Its
original packaged README belongs to the frozen `artifacts.json` hash set and
must remain unchanged; this repository README is a local-bundle index.

```sh
report_evidence="/path/to/separate/mqt-bench-2026-10-21"
mkdir -p "$report_evidence"
for report_archive in artifacts.tar.zst compact-artifacts.tar.zst \
    native-reuse-artifacts-01.tar.zst native-reuse-artifacts-02.tar.zst \
    native-reuse-artifacts-03.tar.zst; do
    tar --zstd -xf "$report_bundle/$report_archive" -C "$report_evidence"
done
python3 bloq_py/examples/mqt_bench_report.py "$report_evidence" --verify
```

Open `report.pdf` inside the extracted evidence directory so its relative
QASM, BLOG, IR and viewer links resolve. Verification checks packaged hashes
and paths; it does not add algorithm-equivalence evidence. For new experiments,
see the [MQT Bench tutorial](../../../../docs/tutorials/mqt-bench.md).
