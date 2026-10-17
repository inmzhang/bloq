//! Benchmark workloads over the shared case registry.
//!
//! Case selection reuses the registry's grep queries, defaulting to the
//! [`crate::COMPILE_BENCH_CORE_ALIAS`] subset.

mod adder;
pub use adder::controlled_adder;

use std::env;

use bloq_graph::{
    Action, Basis, Block, BlockGraph, BlockKind, BranchArm, CubeKind, Direction, Expr,
    MeasureTarget, Pipe,
};
use glam::IVec3;

use crate::{
    COMPILE_BENCH_CORE_ALIAS, TestCase, TestCaseError, compile_fixtures, compile_ready_test_cases,
    required_compile_ready_test_cases,
};

/// Default code-distance ladder for aggregate benchmarks.
pub const DEFAULT_COMPILE_DISTANCES: &[u32] = &[3, 7, 11, 15, 21];

/// Reads a benchmark case query from the `BLOQ_TEST_CASE_QUERY` environment
/// variable, ignoring blank values.
pub fn benchmark_case_query_from_env() -> Option<String> {
    env::var("BLOQ_TEST_CASE_QUERY")
        .ok()
        .map(|query| query.trim().to_string())
        .filter(|query| !query.is_empty())
}

fn benchmark_case_query(case_query: Option<&str>) -> &str {
    case_query
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .unwrap_or(COMPILE_BENCH_CORE_ALIAS)
}

/// Criterion value for an aggregate workload at `distance`.
pub fn benchmark_value(case_query: Option<&str>, distance: u32) -> String {
    format!(
        "cases-{}-d{distance}",
        case_slug(benchmark_case_query(case_query))
    )
}

/// Selects the cases for a benchmark workload, applying the default core subset
/// when `case_query` is empty.
///
/// # Errors
///
/// Returns a [`TestCaseError`] if the query selects no compile-ready cases.
pub fn benchmark_workload_cases(case_query: Option<&str>) -> Result<Vec<TestCase>, TestCaseError> {
    required_compile_ready_test_cases(Some(benchmark_case_query(case_query)))
}

/// Independent terminal branches for projection-scaling benchmarks.
///
/// # Panics
///
/// Panics if `branches` is zero.
pub fn structural_branch_workload(branches: usize) -> BlockGraph {
    assert!(branches > 0, "branch workload must contain a branch");
    let mut graph = BlockGraph::new();
    let mut actions = Vec::with_capacity(2 * branches);
    for index in 0..branches {
        let x = i32::try_from(3 * index).expect("benchmark width fits i32");
        let past = IVec3::new(x, 0, 0);
        let target = past + IVec3::Z;
        let controller = IVec3::new(x, 2, 0);
        graph.add_block(Block::new(past, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(controller, BlockKind::Cube(CubeKind::ZXZ)));
        let arm = |kind| {
            BranchArm::new(
                vec![Block::new(target, kind)],
                vec![Pipe::new(past, Direction::ZPLUS)],
            )
        };
        let target = graph
            .try_add_branch_region(
                format!("b{index}"),
                arm(BlockKind::Measurement(Basis::X)),
                arm(BlockKind::Cube(CubeKind::ZXZ)),
            )
            .expect("benchmark branch interface");
        let name = format!("m{index}");
        actions.push(Action::Measure {
            target: MeasureTarget::Node(controller),
            name: name.clone(),
        });
        actions.push(Action::Branch {
            target,
            condition: Expr::Var(name),
        });
    }
    // The compiler owns semantic analysis; eager setup enumerates all branch tuples.
    graph
        .set_actions_deferred(actions)
        .expect("benchmark actions");
    graph
}

/// A wider version of the gallery's one-dimensional yoked memory.
///
/// # Panics
///
/// Panics if `width` is less than two, exceeds `i32`, or the gallery geometry is invalid.
pub fn yoked_memory(width: usize) -> BlockGraph {
    assert!(width >= 2, "yoked memory needs at least two columns");
    let width = i32::try_from(width).expect("benchmark width fits i32");
    let seed = bloq_graph::GalleryItem::OneDYoked
        .build()
        .materialize_root_graph()
        .expect("yoked gallery geometry");
    let mut graph = BlockGraph::new();
    for x in 0..width {
        let offset = IVec3::new(x - 1, 0, 0);
        for block in seed.blocks().filter(|block| block.pos().x == 1) {
            graph.add_block(block.try_with_shift(offset).expect("shifted yoked column"));
        }
    }
    for x in 0..width {
        let offset = IVec3::new(x - 1, 0, 0);
        for pipe in seed.pipes() {
            let endpoints = (pipe.src().x, pipe.dst().x);
            if endpoints == (1, 1) || (x + 1 < width && matches!(endpoints, (1, 2) | (2, 1))) {
                graph.add_pipe(pipe.try_with_shift(offset).expect("shifted yoked pipe"));
            }
        }
    }
    graph
}

// --- Per-Case Benchmark Specs ---

/// A single compile-ready test case prepared for per-case benchmarking and
/// profiling.
#[derive(Debug, Clone)]
pub struct CaseBenchSpec {
    /// Identifier of the fixture this case was derived from (group key).
    pub fixture_id: &'static str,
    /// Canonical (human-readable) case name, e.g. `cube_line[base][rotate:Z:1]`.
    pub case_name: String,
    /// Filesystem- and Criterion-safe slug shared by benchmark IDs and profile paths.
    pub case_slug: String,
    /// The underlying test case (use [`case.build()`](TestCase::build) to obtain the graph).
    pub case: TestCase,
}

/// All per-case benchmark specs of one fixture, benchmarked as one Criterion
/// group.
#[derive(Debug, Clone)]
pub struct FixtureCaseGroup {
    /// Id of the fixture all specs in this group derive from.
    pub fixture_id: &'static str,
    /// Per-case specs, sorted by case slug.
    pub specs: Vec<CaseBenchSpec>,
}

/// Default code-distance ladder for per-case benchmarks.
const DEFAULT_CASE_BENCH_DISTANCES: &[u32] = &[7, 15, 21];

/// Code distances for per-case benchmarks, overridable via the
/// `BLOQ_BENCH_CASE_DISTANCES` environment variable (comma-separated, e.g.
/// `"7,21"`).
///
/// # Errors
///
/// Returns an error if an entry is not an unsigned integer.
pub fn case_bench_distances() -> Result<Vec<u32>, TestCaseError> {
    let Ok(raw) = env::var("BLOQ_BENCH_CASE_DISTANCES") else {
        return Ok(DEFAULT_CASE_BENCH_DISTANCES.to_vec());
    };
    let distances = raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            entry.parse().map_err(|error| {
                TestCaseError(format!(
                    "invalid BLOQ_BENCH_CASE_DISTANCES entry {entry:?}: {error}"
                ))
            })
        })
        .collect::<Result<Vec<u32>, _>>()?;
    Ok(if distances.is_empty() {
        DEFAULT_CASE_BENCH_DISTANCES.to_vec()
    } else {
        distances
    })
}

/// Benchmark cases grouped by source fixture, in fixture declaration order.
/// Cases whose graphs deduplicated across fixtures are attributed to the
/// fixture of their first recorded origin.
///
/// # Errors
///
/// Returns an error if the case query cannot be resolved.
///
/// # Panics
///
/// Panics if the shared case registry violates its construction invariants.
pub fn fixture_case_groups(
    case_query: Option<&str>,
) -> Result<Vec<FixtureCaseGroup>, TestCaseError> {
    let mut groups = compile_fixtures()
        .iter()
        .map(|fixture| FixtureCaseGroup {
            fixture_id: fixture.id(),
            specs: Vec::new(),
        })
        .collect::<Vec<_>>();

    for case in compile_ready_test_cases(Some(benchmark_case_query(case_query)))? {
        let fixture = case
            .metadata
            .origins
            .first()
            .expect("every test case records at least one origin")
            .fixture;
        let group = groups
            .iter_mut()
            .find(|group| group.fixture_id == fixture.id())
            .expect("compile-ready cases only derive from compile fixtures");
        group.specs.push(CaseBenchSpec {
            fixture_id: fixture.id(),
            case_name: case.id().to_string(),
            case_slug: case_slug(case.id()),
            case,
        });
    }

    groups.retain(|group| !group.specs.is_empty());
    for group in &mut groups {
        group.specs.sort_by(|a, b| a.case_slug.cmp(&b.case_slug));
    }
    Ok(groups)
}

/// Flat list of all per-case benchmark specs (all fixtures concatenated).
///
/// # Errors
///
/// Returns an error if the case query cannot be resolved.
pub fn case_bench_specs(case_query: Option<&str>) -> Result<Vec<CaseBenchSpec>, TestCaseError> {
    Ok(fixture_case_groups(case_query)?
        .into_iter()
        .flat_map(|group| group.specs)
        .collect())
}

/// Slugify a case name for use in benchmark IDs and profile paths, e.g.
/// `cube_line[base][rotate:Z:1]` becomes `cube-line-base-rotate-z-1`.
pub fn case_slug(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for char in name.chars() {
        if char.is_ascii_alphanumeric() {
            slug.push(char.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    if slug.ends_with('-') {
        slug.pop();
    }
    slug
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_value_preserves_aggregate_ids() {
        assert_eq!(benchmark_value(None, 7), "cases-bench-core-d7");
        assert_eq!(
            case_slug("cube_line[base][rotate:Z:1]"),
            "cube-line-base-rotate-z-1"
        );
    }

    #[test]
    fn structural_branch_workload_has_full_projection_domain() {
        let graph = structural_branch_workload(4);
        assert_eq!(graph.branch_projections().unwrap().len(), 16);
        assert!(
            graph
                .analyze_actions()
                .unwrap()
                .action_graph()
                .is_analyzed()
        );
    }

    #[test]
    fn yoked_memory_preserves_gallery_geometry_and_scales() {
        let seed = bloq_graph::GalleryItem::OneDYoked
            .build()
            .materialize_root_graph()
            .unwrap();
        let generated = yoked_memory(6);
        let geometry = |graph: &BlockGraph| {
            let mut blocks = graph.blocks().cloned().collect::<Vec<_>>();
            blocks.sort_by_key(|block| block.pos().to_array());
            let mut pipes = graph
                .pipes()
                .map(|pipe| {
                    let mut endpoints = [pipe.src().to_array(), pipe.dst().to_array()];
                    endpoints.sort_unstable();
                    (endpoints, pipe.is_hadamard())
                })
                .collect::<Vec<_>>();
            pipes.sort_unstable();
            (blocks, pipes)
        };
        // Block equality includes port roles, rotations and custom heights.
        assert_eq!(geometry(&generated), geometry(&seed));
        for width in [2, 64] {
            let graph = yoked_memory(width);
            assert_eq!(graph.block_count(), seed.block_count() / 6 * width);
            assert_eq!(
                graph
                    .blocks()
                    .filter(|block| block.port_role().is_some())
                    .count(),
                2 * width
            );
            graph.validate().expect("valid yoked memory source");
        }
    }

    #[test]
    fn case_slugs_are_unique_join_keys() {
        let specs = case_bench_specs(None).expect("enumerate per-case specs");
        assert!(!specs.is_empty());
        let mut slugs = specs
            .iter()
            .map(|spec| spec.case_slug.clone())
            .collect::<Vec<_>>();
        let total = slugs.len();
        slugs.sort();
        slugs.dedup();
        assert_eq!(slugs.len(), total);
        assert!(slugs.iter().all(|slug| {
            slug.chars()
                .all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '-')
        }));
    }

    #[test]
    fn fixture_case_groups_default_to_base_benchmark_cases() {
        let groups = fixture_case_groups(None).expect("group per-case specs");
        for (fixture, case, slug) in [
            ("x_memory", "x_memory[base]", "x-memory-base"),
            ("stability", "stability[base]", "stability-base"),
            ("cube_line", "cube_line[base]", "cube-line-base"),
        ] {
            let group = groups
                .iter()
                .find(|group| group.fixture_id == fixture)
                .unwrap();
            let spec = group
                .specs
                .iter()
                .find(|spec| spec.case_name == case)
                .unwrap();
            assert_eq!(spec.fixture_id, fixture);
            assert_eq!(spec.case_slug, slug);
        }
        assert!(groups.iter().all(|group| {
            group
                .specs
                .windows(2)
                .all(|pair| pair[0].case_slug < pair[1].case_slug)
        }));
    }
}
