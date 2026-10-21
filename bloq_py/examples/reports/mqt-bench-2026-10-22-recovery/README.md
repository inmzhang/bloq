# MQT Bench → Bloq: compilation recovery catalog

[Read the fresh catalog](report.pdf). It selects **71 validated compiler–circuit pairs**, 35 cases and 24 families: SpiderLS 35, TopoLS 28, Topologiq 8.
All selected programs compile at requested d=25 with ideal T-source proxies, native validation and binary round trip. Exactly 26 selected artifacts retain sampled QuiZX and d=3 VM checks; 3 more retain sampled VM passes with inconclusive logical checks. Fresh recovery layouts are compilation-only. Requested d=25 is nominal: retained spatial-Hadamard warnings may indicate lower effective distance.

The [portable ledger](report-summary.json) distinguishes exact lowering, synthesis bounds, compilation and sampled execution. Each case links raw/lowered QASM, BLOG, d=25 IR, rendered 3D views, original outcomes and provenance. Very tall layouts retain a true-proportion overview plus a labeled magnified central detail with equal axes. Interactive viewers embed geometry but require internet access to load the Google CDN runtime. No large physical-execution or optimal-packing claim is made.

R1 and R2 frozen cohorts retain separate source patches, file ledgers, native audits and all bounded outcomes. No semantic evidence transfers across changed BLOG hashes. Primary replacements require no increase in bounding XY area, occupied XY columns, blocks or temporal extent, and strict improvement in at least one; validated tradeoffs remain alternates. Timings retain their actual run configurations; pipeline_seconds begins after Clifford+T lowering and includes routing through validated IR and binary round trip. The original 2026-10-21 catalog and its five archives are byte-identical historical references, and do not need to be overlaid onto this fresh directory.

Extract every part into this directory to enable portable PDF links. From the repository root:

```sh
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-01.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-02.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-03.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-04.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-05.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-06.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-07.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-08.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-09.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-10.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-11.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-12.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy tar --zstd -xf bloq_py/examples/reports/mqt-bench-2026-10-22-recovery/artifacts-13.tar.zst \
    -C bloq_py/examples/reports/mqt-bench-2026-10-22-recovery
rtk proxy python3 bloq_py/examples/mqt_bench_report.py     bloq_py/examples/reports/mqt-bench-2026-10-22-recovery --verify
```

The [artifact manifest](artifacts.json) checks retained bytes and local paths. Verification additionally inspected actual PDF URIs and an independent extraction with identical documentation and metadata. The archives contain this README, ignore rules, Typst source, complete report data, provenance, and artifacts. Unpacked evidence is ignored by Git.
