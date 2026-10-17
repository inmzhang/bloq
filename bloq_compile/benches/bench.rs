use std::fmt::Write;
use std::hint::black_box;
use std::time::Duration;

use bloq_compile::{CompileConfig, CompileContext, SharedCompileCache};
use bloq_graph::{BlockGraph, ModuleCertificationLimits};
use bloq_test::benchmark::{
    DEFAULT_COMPILE_DISTANCES, benchmark_case_query_from_env, benchmark_value,
    benchmark_workload_cases, case_bench_distances, fixture_case_groups,
    structural_branch_workload,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

fn make_context(distance: u32) -> CompileContext {
    CompileContext::new(CompileConfig::new(distance))
}

fn compile_workload(ctx: &CompileContext, workloads: &[BlockGraph]) -> (usize, u64) {
    let mut total_nodes = 0usize;
    let mut total_measurements = 0u64;
    for graph in workloads {
        let program = ctx.compile(graph);
        let program = program.expect("compile benchmark workload").bloq;
        total_nodes += program.quantum_node_count();
        total_measurements += program.measurement_count() as u64;
    }
    (total_nodes, total_measurements)
}

fn bench_compile_scenarios(c: &mut Criterion) {
    let case_query = benchmark_case_query_from_env();
    let mut group = c.benchmark_group("compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));

    for &distance in DEFAULT_COMPILE_DISTANCES {
        let value = benchmark_value(case_query.as_deref(), distance);
        group.bench_with_input(
            BenchmarkId::new("runtime", &value),
            &distance,
            |b, distance| {
                b.iter_batched(
                    || {
                        let workloads = benchmark_workload_cases(case_query.as_deref())
                            .expect("runtime setup")
                            .into_iter()
                            .map(|case| case.build())
                            .collect::<Vec<_>>();
                        (make_context(*distance), workloads)
                    },
                    |(ctx, workloads)| black_box(compile_workload(&ctx, &workloads)),
                    BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

// Fine-grained benchmarks: one Criterion group per fixture, one benchmark per
// case variant (fills, rotations) and distance. Benchmark IDs look like
// `compile-case/<fixture>/<case-slug>/d<distance>`; case slugs also name profiles
// under `target/benchmark/profiles/`.
fn bench_compile_cases(c: &mut Criterion) {
    let case_query = benchmark_case_query_from_env();
    let groups = fixture_case_groups(case_query.as_deref()).expect("enumerate per-case specs");
    let distances =
        case_bench_distances().expect("BLOQ_BENCH_CASE_DISTANCES must contain unsigned integers");

    for fixture_group in groups {
        let mut group = c.benchmark_group(format!("compile-case/{}", fixture_group.fixture_id));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));

        for spec in &fixture_group.specs {
            for &distance in &distances {
                group.bench_with_input(
                    BenchmarkId::new(&spec.case_slug, format!("d{distance}")),
                    spec,
                    |b, spec| {
                        // Build inside the closure so filtered-out benchmarks
                        // do not pay graph construction.
                        let graph = spec.case.build();
                        b.iter_batched(
                            || make_context(distance),
                            |ctx| {
                                let compiled = ctx.compile(&graph);
                                black_box(compiled.unwrap_or_else(|error| {
                                    panic!("compile {}: {error}", spec.case_name)
                                }))
                            },
                            BatchSize::PerIteration,
                        );
                    },
                );
            }
        }

        group.finish();
    }
}

fn bench_structural_branch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("structural-branches");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    for branches in 1..=5 {
        let graph = structural_branch_workload(branches);
        group.bench_with_input(BenchmarkId::new("compile", branches), &graph, |b, graph| {
            b.iter_batched(
                || make_context(3),
                |ctx| black_box(ctx.compile(graph).expect("compile branch workload")),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn measured_chain(stages: usize) -> BlockGraph {
    const STAGE: &str = r#"module MeasuredStage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, 0] role=input <q_in>
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q_out>
  3: ZXZ [1, 0, 0]
  4: Z [1, 0, 1]
  0 -> +Z
  1 -> +Z
  3 -> +Z
  m = measure 4
}
"#;
    let mut instances = String::new();
    let mut connectors = String::new();
    let mut connections = String::from("  100 -> s0.q_in\n");
    for index in 0..stages {
        writeln!(
            instances,
            "  s{index}: MeasuredStage @ [0, 0, {}]",
            index * 2
        )
        .expect("writing to a String cannot fail");
        if index + 1 < stages {
            writeln!(
                connectors,
                "  {}: XZX [0, 0, {}]",
                200 + index,
                (index + 1) * 2
            )
            .expect("writing to a String cannot fail");
            writeln!(connections, "  s{index}.q_out -> {}", 200 + index)
                .expect("writing to a String cannot fail");
            writeln!(connections, "  {} -> s{}.q_in", 200 + index, index + 1)
                .expect("writing to a String cannot fail");
        }
    }
    writeln!(connections, "  s{}.q_out -> 101", stages - 1)
        .expect("writing to a String cannot fail");
    BlockGraph::from_text(&format!(
        "BLOG 1.0\n\n{STAGE}\nmodule main {{\n  in q_in: data = 100\n  out q_out: data = 101\n{instances}  100: Port [0, 0, 0] role=input <q_in>\n{connectors}  101: Port [0, 0, {}] role=output <q_out>\n{connections}}}\n",
        stages * 2
    ))
    .expect("measured-chain benchmark is valid")
}

fn bench_module_summary_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("module-summary-scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    for stages in [8, 16, 32] {
        let program = measured_chain(stages);
        group.bench_with_input(
            BenchmarkId::new("measured-chain", stages),
            &program,
            |b, program| {
                b.iter(|| {
                    black_box(
                        program
                            .summarize_root(ModuleCertificationLimits::UNLIMITED)
                            .expect("summarize measured chain"),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_module_objects(c: &mut Criterion) {
    let mut group = c.benchmark_group("module-objects");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(1));
    for (name, program) in [
        ("cnot", bloq_graph::GalleryItem::CNOT.build()),
        ("measured-chain-8", measured_chain(8)),
        ("measured-chain-32", measured_chain(32)),
        (
            "phase-gradient-k4",
            bloq_graph::GalleryItem::PhaseGradientK4.build(),
        ),
    ] {
        group.bench_function(BenchmarkId::new("object-cold", name), |b| {
            b.iter_batched(
                || make_context(3),
                |ctx| {
                    black_box(
                        ctx.compile_object(&program)
                            .expect("compile benchmark module object"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("compile-cold", name), |b| {
            b.iter_batched(
                || make_context(3),
                |ctx| black_box(ctx.compile(&program).expect("compile benchmark module")),
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("link", name), |b| {
            let context = make_context(3);
            let object = context
                .compile_object(&program)
                .expect("compile benchmark module object");
            b.iter_batched(
                || (),
                |()| {
                    black_box(
                        context
                            .link_object(&object)
                            .expect("link benchmark module object"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("compile-warm", name), |b| {
            let context = make_context(3);
            context
                .compile_object(&program)
                .expect("compile benchmark module object");
            b.iter_batched(
                || (),
                |()| black_box(context.compile(&program).expect("compile benchmark module")),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn selective_modules(count: usize, inverted: bool) -> BlockGraph {
    let mut source = String::from(
        "BLOG 1.0\nmodule Cap {\n  in basis\n  0: ZXZ [0,0,0]\n  1: XY [0,0,1]\n  0 -> +Z\n  resolve 1 if basis\n}\nmodule main {\n  0: ZXZ [-3,0,-1]\n  source = measure 0\n",
    );
    writeln!(
        source,
        "  out control = {}source",
        if inverted { "!" } else { "" }
    )
    .expect("writing to a String cannot fail");
    for index in 0..count {
        writeln!(
            source,
            "  cap{index}: Cap @ [{},0,0] rotate Z {}\n  control => cap{index}.basis",
            index * 3,
            90 * (index % 4)
        )
        .expect("writing to a String cannot fail");
    }
    source.push_str("}\n");
    BlockGraph::from_text(&source).expect("selective module workload")
}

fn bench_selective_definition_inventory(c: &mut Criterion) {
    let mut group = c.benchmark_group("selective-definition-inventory");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(1));
    for count in [8, 32] {
        let first = selective_modules(count, false);
        let rebound = selective_modules(count, true);
        group.bench_with_input(BenchmarkId::new("cold", count), &rebound, |b, program| {
            b.iter_batched(
                || make_context(3),
                |context| black_box(context.compile(program).expect("compile selective modules")),
                BatchSize::PerIteration,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("templates-warm", count),
            &rebound,
            |b, program| {
                b.iter_batched(
                    || {
                        let cache = SharedCompileCache::new();
                        let context =
                            CompileContext::with_shared_cache(CompileConfig::new(3), &cache);
                        context.compile(&first).expect("warm block templates");
                        // Existing contexts retain block templates when object caches clear.
                        cache.clear();
                        context
                    },
                    |context| {
                        black_box(
                            context
                                .compile(program)
                                .expect("rebuild selective inventory"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rebound", count),
            &rebound,
            |b, program| {
                b.iter_batched(
                    || {
                        let cache = SharedCompileCache::new();
                        let context =
                            CompileContext::with_shared_cache(CompileConfig::new(3), &cache);
                        context.compile(&first).expect("warm definition inventory");
                        CompileContext::with_shared_cache(CompileConfig::new(3), &cache)
                    },
                    |context| black_box(context.compile(program).expect("compile rebound modules")),
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compile_scenarios,
    bench_compile_cases,
    bench_structural_branch_scaling,
    bench_module_summary_scaling,
    bench_module_objects,
    bench_selective_definition_inventory
);
criterion_main!(benches);
