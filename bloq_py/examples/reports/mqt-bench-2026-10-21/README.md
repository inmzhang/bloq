# MQT Bench compilation catalog — 2026-10-21

[Read the report](report.pdf). It records 53 successful distance-25 compilations
across 26 circuit cases and 21 algorithm families: SpiderLS 26, TopoLS 19, and
Topologiq 8. Twenty-six runs also pass sampled QuiZX and distance-3 VM checks
with ideal T sources. Large runs, including the 32-qubit CDKM adder, are
compilation-only results. The report retains unsuccessful and inconclusive
outcomes separately.

The PDF is available directly. Its relative links become usable after extracting
`artifacts.tar.zst` into this directory. The archive retains the Typst source,
ledgers, provenance, QASM, BLOG, binary IR, previews, interactive viewers, and
stage logs. Unpacked evidence occupies about 1.6 GB and is ignored by Git.

Run from the repository root, with `tar` and `zstd` installed:

```sh
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/artifacts.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy python3 bloq_py/examples/mqt_bench_report.py \
    bloq_py/examples/reports/mqt-bench-2026-10-21 --verify
```

Verification checks every retained artifact hash and local file link. Open the
extracted `layout.html` files through the report's viewer links. The recorded
manifests identify the source revisions and patches used for the experiments;
subsequent cleanup commits do not alter that evidence.

For new experiments, follow [the MQT Bench tutorial](../../../../docs/tutorials/mqt-bench.md).
