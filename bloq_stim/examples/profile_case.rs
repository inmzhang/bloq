// --- Per-Case Profile Harness ---
//
// Runs exactly one compile-ready case (or the combined benchmark suite) in a
// tight loop so that `perf record` captures a profile attributable to that
// case. Run through `just profile-case`.
//
// Usage:
//   RUSTFLAGS="-Cforce-frame-pointers=yes" \
//     cargo build --profile profiling -p bloq_stim --example profile_case
//   perf record ... -- target/profiling/examples/profile_case [OPTIONS]
//
// Options:
//   --stage STAGE        `compile` (graph -> Bloq) or `stim` (Bloq -> Stim text)
//                        (default: compile)
//   --case NAME          Exact case name, alias, or case slug as used in
//                        benchmark IDs (e.g. `cube-line-base-rotate-z-1`), or
//                        `all` for the combined `bench-core` suite
//                        (default: all)
//   --distance N         Code distance (default: 11)
//   --duration-secs N    Keep iterating until N measured seconds have elapsed,
//                        so workloads get a comparable number of perf
//                        samples regardless of its size (default: 5)
//   --iterations N       Run exactly N measured iterations instead of
//                        `--duration-secs`
//   --warmup N           Unmeasured warmup iterations (default: 2)
//   --describe           Compile once, validate, print JSON IR counts, then exit
//   --dump-ir DIR        Also write each described program to DIR/<index>.bloq

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use bloq_stim::emit_bloq_stim;
use bloq_test::benchmark::{
    case_bench_specs, case_slug, controlled_adder, structural_branch_workload,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Compile,
    Stim,
}

enum StopCondition {
    Duration(Duration),
    Iterations(usize),
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut stage = Stage::Compile;
    let mut case: String = "all".to_string();
    let mut distance: u32 = 11;
    let mut stop = StopCondition::Duration(Duration::from_secs(5));
    let mut warmup: usize = 2;
    let mut describe = false;
    let mut dump_ir = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--stage" => {
                i += 1;
                stage = match args[i].as_str() {
                    "compile" => Stage::Compile,
                    "stim" => Stage::Stim,
                    other => {
                        eprintln!("unknown stage {other:?} (expected `compile` or `stim`)");
                        std::process::exit(1);
                    }
                };
            }
            "--case" => {
                i += 1;
                case = args[i].clone();
            }
            "--distance" => {
                i += 1;
                distance = args[i].parse().expect("invalid distance");
            }
            "--duration-secs" => {
                i += 1;
                stop = StopCondition::Duration(Duration::from_secs_f64(
                    args[i].parse().expect("invalid duration"),
                ));
            }
            "--iterations" => {
                i += 1;
                stop = StopCondition::Iterations(args[i].parse().expect("invalid iterations"));
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().expect("invalid warmup");
            }
            "--describe" => describe = true,
            "--dump-ir" => {
                i += 1;
                dump_ir = Some(std::path::PathBuf::from(&args[i]));
                describe = true;
            }
            "--help" => {
                eprintln!("Per-case profile harness; see the header of this example for options.");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // ---- Setup phase: resolve cases and build graphs once ----
    let workloads = resolve_case_workloads(&case);
    let stage_name = match stage {
        Stage::Compile => "compile",
        Stage::Stim => "stim",
    };
    eprintln!(
        "profiling stage={stage_name} case={case} d={distance} ({} graph(s))",
        workloads.len()
    );

    // For the stim stage the compilation itself is setup, not workload.
    let programs = match stage {
        Stage::Compile if !describe => Vec::new(),
        Stage::Compile | Stage::Stim => workloads
            .iter()
            .map(|graph| {
                make_context(distance)
                    .compile(graph)
                    .expect("compile stim-stage workload")
                    .bloq
            })
            .collect::<Vec<_>>(),
    };

    if describe {
        if let Some(directory) = &dump_ir {
            std::fs::create_dir_all(directory).expect("create IR dump directory");
        }
        for (index, program) in programs.iter().enumerate() {
            program.validate().expect("valid compiled IR");
            let binary = program.to_binary();
            if let Some(directory) = &dump_ir {
                std::fs::write(directory.join(format!("{index}.bloq")), &binary)
                    .expect("write compiled IR");
            }
            println!(
                "{{\"index\":{index},\"distance\":{distance},\"quantum_nodes\":{},\"templates\":{},\"measurements\":{},\"binary_bytes\":{}}}",
                program.quantum_node_count(),
                program.templates().len(),
                program.measurement_count(),
                binary.len(),
            );
        }
        return;
    }

    let run_once = || match stage {
        Stage::Compile => {
            let ctx = make_context(distance);
            let mut total = 0usize;
            for graph in &workloads {
                total += ctx
                    .compile(graph)
                    .expect("compile workload")
                    .bloq
                    .quantum_node_count();
            }
            total
        }
        Stage::Stim => programs
            .iter()
            .map(|program| emit_bloq_stim(program).expect("emit Stim workload").len())
            .sum(),
    };

    // ---- Warmup phase: not measured ----
    for _ in 0..warmup {
        black_box(run_once());
    }
    eprintln!("warmup done ({warmup} iterations)");

    // ---- Measured phase: this is what perf should capture ----
    let start = Instant::now();
    let mut iterations = 0usize;
    loop {
        black_box(run_once());
        iterations += 1;
        match stop {
            StopCondition::Duration(duration) => {
                if start.elapsed() >= duration {
                    break;
                }
            }
            StopCondition::Iterations(count) => {
                if iterations >= count {
                    break;
                }
            }
        }
    }
    let elapsed = start.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    eprintln!(
        "done: {iterations} iterations in {:.1}s ({per_iter_ms:.3} ms/iter)",
        elapsed.as_secs_f64()
    );
}

fn resolve_case_workloads(case: &str) -> Vec<BlockGraph> {
    if let Some(bits) = case.strip_prefix("adder-") {
        let program = controlled_adder(bits.parse().expect("adder bit count"));
        return vec![program];
    }
    if let Some(branches) = case
        .strip_prefix("branch-scale-")
        .and_then(|value| value.parse().ok())
    {
        return vec![structural_branch_workload(branches)];
    }
    let specs =
        case_bench_specs((case != "all").then_some(case)).expect("enumerate compile-ready cases");
    if case == "all" {
        return specs.iter().map(|spec| spec.case.build()).collect();
    }

    let wanted_slug = case_slug(case);
    let Some(spec) = specs.iter().find(|spec| {
        spec.case_slug == wanted_slug
            || spec.case_name == case
            || spec.case.aliases().iter().any(|alias| alias == case)
    }) else {
        eprintln!("no compile-ready case matched {case:?}; available slugs:");
        for spec in &specs {
            eprintln!("  {}", spec.case_slug);
        }
        std::process::exit(1);
    };
    eprintln!("resolved case {:?} -> {}", case, spec.case_name);
    vec![spec.case.build()]
}

fn make_context(distance: u32) -> CompileContext {
    CompileContext::new(CompileConfig::new(distance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_named_cases_with_module_provenance() {
        assert!(!resolve_case_workloads("x-memory-base-rotate-z-1")[0].has_module_structure());
        assert!(resolve_case_workloads("phase-gradient-k4-open")[0].has_module_structure());
    }

    #[test]
    fn resolves_structural_branch_scaling_case() {
        assert_eq!(
            resolve_case_workloads("branch-scale-3")[0]
                .branch_projections()
                .unwrap()
                .len(),
            8
        );
    }
}
