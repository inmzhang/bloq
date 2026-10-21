# MQT Bench compilation catalog — 2026-10-21

[Read the report](report.pdf). It records 63 successful distance-25 compilations
across 30 circuit cases and 23 algorithm families: SpiderLS 30, TopoLS 25, and
Topologiq 8. Twenty-six runs also pass sampled QuiZX and distance-3 VM checks
with ideal T sources. Large runs, including the 32-qubit CDKM adder, are
compilation-only results. The report retains unsuccessful and inconclusive
outcomes separately.

The [compact-placement addendum](compact-placement/report.pdf) records ten new
compilations and compares their geometry with the rejected exterior-bank
prototype. QAOA3/SpiderLS uses 73.5% less bounding area and 31% fewer occupied
spatial columns. These compilation-only additions retain private factory
columns; they do not claim optimal packing or VM execution.

The PDFs are available directly. Their relative links become usable after extracting
`artifacts.tar.zst` and `compact-artifacts.tar.zst` into this directory. The archive retains the Typst source,
ledgers, provenance, QASM, BLOG, binary IR, previews, interactive viewers, and
stage logs. Unpacked evidence is ignored by Git.

Run from the repository root, with `tar` and `zstd` installed:

```sh
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/artifacts.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/compact-artifacts.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy python3 bloq_py/examples/mqt_bench_report.py \
    bloq_py/examples/reports/mqt-bench-2026-10-21 --verify
```

Verification checks every retained artifact hash and local file link. Open the
extracted `layout.html` files through the report's viewer links. The recorded
manifests identify the source revisions and patches used for the experiments;
subsequent cleanup commits do not alter that evidence.

For new experiments, follow [the MQT Bench tutorial](../../../../docs/tutorials/mqt-bench.md).
