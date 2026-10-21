# MQT Bench compilation catalog — 2026-10-21

[Read the accepted catalog](report.pdf). It selects 65 successful
compilations at requested distance 25 across 31 circuit cases and
24 algorithm families: SpiderLS 31,
TopoLS 26, and Topologiq 8.
26 selected runs also have sampled QuiZX and distance-3 VM checks
with ideal T sources. Compilation, synthesis approximation, partial or inconclusive
checks, and sampled execution remain separate claims.
Another 3 selected runs pass VM checks
while their bounded logical checks remain inconclusive; they carry no logical-channel proof.
Requested distances are nominal: retained spatial-Hadamard warnings can indicate
lower effective circuit distance. No measured-distance guarantee is claimed.

The [native T-patch reuse comparison](native-reuse/comparison.pdf) records
26 strict footprint improvements and 2 new compiler–circuit
pairs. Equal or worse footprints and unsuccessful replays retain accepted baseline
artifacts. Replacements increase none of bounding XY area, occupied columns, blocks,
or temporal extent. All 65 replay outcomes and every historical result remain in
their original ledgers. New layouts receive semantic evidence only through exact
BLOG, lowered-QASM, and program hashes.

The [earlier compact-placement addendum](compact-placement/report.pdf) and the
historical catalog retained in the archives document the prior private-factory-column
experiments. Current reuse candidates instead pass native lifetime and dependency
validation. These results do not claim optimal packing or large-circuit execution.

The PDFs are available directly. Extract all evidence layers below to enable their
relative links to QASM, BLOG, distance-25 IR, previews, viewers, logs, and provenance.
The two historical archives remain byte-for-byte unchanged; the native-reuse layer
adds the reviewed selection and retains older evidence. Unpacked files are ignored
by Git.

Run from the repository root, with `tar` and `zstd` installed:

```sh
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/artifacts.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/compact-artifacts.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/native-reuse-artifacts-01.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/native-reuse-artifacts-02.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-21/native-reuse-artifacts-03.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-21
rtk proxy python3 bloq_py/examples/mqt_bench_report.py \
    bloq_py/examples/reports/mqt-bench-2026-10-21 --verify
```

Verification checks retained artifact hashes and local links. The manifests record
source revisions, patches, exact inputs, seeds, and source snapshots; later cleanup
commits do not change that evidence. For new runs, follow the
[MQT Bench tutorial](../../../../docs/tutorials/mqt-bench.md).
