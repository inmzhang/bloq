//! Shared test-case registry and fixture generation for the `bloq` workspace.
//!
//! A small set of named [`TestFixture`]s (gallery entries and embedded `.blog`
//! assets) is expanded by filling open ports, flipping the X/Z basis, and
//! rotating, then deduplicated into a flat list of [`TestCase`]s.
//! Downstream crates and benchmarks select cases through this registry so they
//! share one deduplicated corpus rather than each maintaining their own. The [`benchmark`]
//! module prepares Criterion workloads from it.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use bloq_graph::{BlockGraph, Direction, GalleryCategory, GalleryItem, Pipe, UDirection};

pub mod benchmark;

/// Error from test case construction, transformation, lookup, or configuration.
///
/// This deliberately carries a pre-rendered message instead of a source error:
/// test harnesses only print it, and the message includes fixture context that
/// no single source error carries.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct TestCaseError(String);

/// Grep alias matching the core benchmark subset (base and fill cases, no
/// flips or rotations).
pub const COMPILE_BENCH_CORE_ALIAS: &str = "bench-core";

/// Negative module-composition fixture; it is deliberately absent from the gallery.
pub const ONE_BIT_ADDER_SOURCE: &str = include_str!("../assets/one_bit_adder.blog");

// One catalog owns the public variants, stable IDs, gallery mappings and order.
macro_rules! fixtures {
    ($( $(#[$doc:meta])* $variant:ident => ($id:literal, $gallery:expr), )*) => {
        /// A named block graph fixture from which test cases are generated.
        ///
        /// Each fixture expands through filling, flipping, and rotation.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum TestFixture { $( $(#[$doc])* $variant, )* }

        const COMPILE_FIXTURES: &[TestFixture] = &[$(TestFixture::$variant),*];

        impl TestFixture {
            /// Returns the stable lowercase id used in case names and slugs.
            pub const fn id(self) -> &'static str {
                match self { $(Self::$variant => $id),* }
            }

            /// Returns the native gallery item, when this fixture has one.
            #[must_use]
            pub fn gallery_item(self) -> Option<GalleryItem> {
                match self { $(Self::$variant => $gallery),* }
            }
        }
    };
}

fixtures! {
    /// Bell state preparation.
    BellState => ("bell_state", Some(GalleryItem::BellState)),
    /// GHZ state preparation.
    GHZ => ("ghz", Some(GalleryItem::GHZ)),
    /// Alternating T and Hadamard gates.
    THTH => ("thth", Some(GalleryItem::THTH)),
    /// T implementation comparison.
    TComparison => ("t_comparison", Some(GalleryItem::TComparison)),
    /// Toffoli composed from AND and delayed CZ correction.
    ToffoliFromAndDelayedCZ => ("toffoli_from_and_delayed_cz", Some(GalleryItem::ToffoliFromAndDelayedCZ)),
    /// CNOT gate (open, requires fill).
    CNOT => ("cnot", Some(GalleryItem::CNOT)),
    /// CZ gate across a spatial Hadamard wall (open, requires fill).
    CZ => ("cz", Some(GalleryItem::CZSpatialH)),
    /// CZ gate with temporal Hadamard pipes.
    CZTemporalH => ("cz_temporal_h", Some(GalleryItem::CZTemporalH)),
    /// S gate (open, requires fill).
    S => ("s", Some(GalleryItem::S)),
    /// Three cascaded CNOT gates.
    ThreeCnots => ("three_cnots", Some(GalleryItem::ThreeCNOTs)),
    /// Steane code encoding circuit.
    SteaneEncoding => ("steane_encoding", Some(GalleryItem::SteaneEncoding)),
    /// X-basis memory experiment.
    XMemory => ("x_memory", Some(GalleryItem::XMemory)),
    /// Y-basis memory experiment.
    YMemory => ("y_memory", Some(GalleryItem::YMemory)),
    /// Move-then-rotate pattern.
    MoveRotation => ("move_rotation", Some(GalleryItem::MoveRotation)),
    /// 1D yoked lattice-surgery layout with patch rotation blocks.
    OneDYoked => ("1d_yoked", Some(GalleryItem::OneDYoked)),
    /// GHZ state preparation using sliding then gliding walking blocks.
    GHZSlideThenGlide => ("ghz_slide_then_glide", Some(GalleryItem::GHZSlideThenGlide)),
    /// GHZ state preparation with patch rotations.
    GHZPatchRotations => ("ghz_patch_rotations", Some(GalleryItem::GHZPatchRotations)),
    /// Stability experiment.
    Stability => ("stability", Some(GalleryItem::Stability)),
    /// T gate teleportation (non-Clifford).
    TGate => ("t_gate", Some(GalleryItem::T)),
    /// T gate teleportation with a prepared Y for Clifford fix (non-Clifford).
    TWithPreparedY => ("t_with_prepared_y", Some(GalleryItem::TWithPreparedY)),
    /// k=4 phase-gradient Rz synthesis (non-Clifford).
    PhaseGradientK4 => ("phase_gradient_k4", Some(GalleryItem::PhaseGradientK4)),
    /// Four-T AND gadget `(x, y) -> (x, y, x AND y)` (non-Clifford).
    And4T => ("and_4t", Some(GalleryItem::And4T)),
    /// CCZ-injected temporary-AND effect with an explicit prepared resource.
    CCZInjectedAnd => ("ccz_injected_and", Some(GalleryItem::CCZInjectedAnd)),
    /// CCZ-injected MAJ carry slice with an explicit prepared resource.
    CCZInjectedMaj => ("ccz_injected_maj", Some(GalleryItem::CCZInjectedMaj)),
    /// Adaptive UMA carry-uncompute and sum-extraction slice.
    UMA => ("uma", Some(GalleryItem::UMA)),
    /// Test-only bulk adder with output-supported readouts that must be rejected.
    OneBitAdder => ("one_bit_adder", None),
    /// Three-bit controlled-adder layout with compact routing.
    ThreeBitAdder => ("three_bit_adder", Some(GalleryItem::ThreeBitAdder)),
    /// Ten-bit controlled-adder layout with compact routing.
    TenBitAdder => ("ten_bit_adder", Some(GalleryItem::TenBitAdder)),
    /// CCZ gate teleportation with explicit correction branches.
    CCZGateTeleport => ("ccz_gate_teleport", Some(GalleryItem::CCZGateTeleport)),
    /// CCZ state factory, 4x3 layout, without the TELS check (non-Clifford).
    CCZ4x3_6 => ("ccz_4x3x6", Some(GalleryItem::CCZ4x3_6)),
    /// CCZ state factory with TELS check (non-Clifford).
    CCZFactoryWithTels => ("ccz_4x3x7_tels", Some(GalleryItem::CCZFactoryWithTels)),
    /// Linear chain of cubes (test-only).
    CubeLine => ("cube_line", None),
    /// H-shaped junction (test-only).
    HShapeJunction => ("h_shape_junction", None),
    /// Bell-state graph with a temporal Hadamard edge (test-only).
    HadamardBellState => ("hadamard_bell_state", None),
    /// Closed spatial Hadamard wall between a cube and a two-armed spatial hub
    /// (test-only).
    SpatialHadamard => ("spatial_hadamard", None),
    /// Stability experiment with a linear chain (test-only).
    StabilityLine => ("stability_line", None),
    /// T-shaped junction (test-only).
    TShapeJunction => ("t_shape_junction", None),
    /// Temporal Hadamard (test-only).
    TemporalHadamard => ("temporal_hadamard", None),
    /// X-shaped junction (test-only).
    XShapeJunction => ("x_shape_junction", None),
}

impl TestFixture {
    /// Returns whether Stim can execute this fixture.
    ///
    /// Gallery-backed fixtures use their gallery category. Test-only geometry
    /// fixtures are Clifford; the negative bulk-adder fixture is non-Clifford.
    pub fn is_clifford(self) -> bool {
        self != Self::OneBitAdder
            && self
                .gallery_item()
                .is_none_or(|item| item.in_category(GalleryCategory::Clifford))
    }

    /// Returns whether this fixture carries `category`.
    pub fn in_category(self, category: GalleryCategory) -> bool {
        if self == Self::OneBitAdder {
            return matches!(
                category,
                GalleryCategory::ExternalResource
                    | GalleryCategory::NonClifford
                    | GalleryCategory::Adaptive
                    | GalleryCategory::Arithmetic
                    | GalleryCategory::Addition
                    | GalleryCategory::AnalysisOnly
            );
        }
        self.gallery_item()
            .is_some_and(|item| item.in_category(category))
    }
}

/// A spatial rotation applied to a test case graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RotationSpec {
    /// Axis of rotation.
    pub axis: UDirection,
    /// Number of 90-degree turns (1–3).
    pub quarter_turns: i32,
}

/// Records how a particular test case was derived from a fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestCaseOrigin {
    /// Human-readable name encoding the derivation chain.
    pub name: String,
    /// Source fixture this case was generated from.
    pub fixture: TestFixture,
    /// Index of the fill variant, if the fixture had open ports.
    pub fill_variant: Option<usize>,
    /// Whether the X/Z basis was flipped.
    pub flip_xz_basis: bool,
    /// Optional rotation applied after filling and flipping.
    pub rotation: Option<RotationSpec>,
}

/// Deduplicated metadata for a test case, potentially merging multiple derivation paths
/// that produced structurally identical graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestCaseMetadata {
    /// Canonical name (from the first origin that produced this graph).
    pub name: String,
    /// All names that map to this graph (including `name`).
    pub aliases: Vec<String>,
    /// All derivation paths that produced this graph.
    pub origins: Vec<TestCaseOrigin>,
}

impl TestCaseMetadata {
    /// Returns the sorted, deduplicated fixture ids across all origins.
    pub fn fixture_ids(&self) -> Vec<&'static str> {
        self.origins
            .iter()
            .map(|origin| origin.fixture)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(TestFixture::id)
            .collect()
    }

    fn merge_origin(&mut self, origin: TestCaseOrigin) {
        if !self.aliases.iter().any(|alias| alias == &origin.name) {
            self.aliases.push(origin.name.clone());
        }
        if !self.origins.iter().any(|existing| existing == &origin) {
            self.origins.push(origin);
        }
    }
}

/// A single registry entry: dedup metadata plus its graph.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Canonical name, aliases, and derivation origins for this case.
    pub metadata: TestCaseMetadata,
    graph: Arc<BlockGraph>,
}

/// A test case paired with its compiler input, retaining authored hierarchy.
/// Static oracle queries are separate from preparing a native compilation.
#[derive(Debug, Clone)]
pub struct CompileReadyCase {
    /// The originating registry case.
    pub test_case: TestCase,
    /// The graph built from `test_case`.
    pub graph: BlockGraph,
}

impl CompileReadyCase {
    /// Returns the canonical case name.
    pub fn id(&self) -> &str {
        self.test_case.id()
    }

    /// Build and validate the source without requiring a static projection.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid geometry or source actions.
    pub fn from_test_case(test_case: TestCase) -> Result<Self, TestCaseError> {
        let graph = test_case.build();
        graph
            .validate_source()
            .map_err(|error| TestCaseError(format!("build: {}: {error}", test_case.id())))?;
        Ok(Self { test_case, graph })
    }

    /// Observable count of the planned correlation surfaces: one per named
    /// readout, output frame, and closed logical surface.
    ///
    /// # Errors
    /// Returns an error when this source has no causal readout plan.
    pub fn expected_observables(&self) -> Result<usize, TestCaseError> {
        self.graph
            .stabilizers()
            .map(|rows| rows.len())
            .map_err(|error| {
                TestCaseError(format!(
                    "build: {}: failed to plan correlation surfaces: {error}",
                    self.id()
                ))
            })
    }
}

impl TestCase {
    /// Returns compiler input with authored hierarchy, or a transformed flat graph.
    pub fn build(&self) -> BlockGraph {
        (*self.graph).clone()
    }

    /// Returns the canonical case name.
    pub fn id(&self) -> &str {
        &self.metadata.name
    }

    /// Returns all names (canonical plus aliases) that resolve to this case.
    pub fn aliases(&self) -> &[String] {
        &self.metadata.aliases
    }

    /// Returns the `shortest_graphlike_error` weight a correct compile of this
    /// case must produce at `distance`.
    ///
    /// Every ordinary case is distance preserving, so the answer is `distance`
    /// itself. The spatial Hadamard wall is not:
    /// its GHZ chain alternates direction but still permits sub-distance
    /// mechanisms. Those cases therefore pin their **measured** weight from
    /// `WALL_GRAPHLIKE_DISTANCES`, which the backend suite asserts by equality
    /// in both directions — a wall that silently got better is as much a change
    /// to explain as one that got worse.
    pub fn expected_graphlike_distance(&self, distance: u32) -> u32 {
        self.metadata
            .origins
            .iter()
            .find_map(|origin| wall_graphlike_distance(origin, distance))
            .unwrap_or(distance)
    }

    /// Returns whether any alias matches `query`, treating an empty query as a
    /// match-all. Matching is fuzzy: case- and separator-insensitive substring
    /// or subsequence.
    fn matches_grep(&self, query: &str) -> bool {
        if query.trim().is_empty() {
            return true;
        }
        self.metadata
            .aliases
            .iter()
            .any(|alias| fuzzy_match(alias, query))
    }

    fn from_graph(metadata: TestCaseMetadata, graph: BlockGraph) -> Self {
        Self {
            metadata,
            graph: Arc::new(graph),
        }
    }
}

#[derive(Debug)]
struct Candidate {
    origin: TestCaseOrigin,
    graph: BlockGraph,
}

/// Returns the fixtures the compile and benchmark surfaces are generated from.
pub fn compile_fixtures() -> &'static [TestFixture] {
    COMPILE_FIXTURES
}

/// Returns the full deduplicated case corpus, built once and memoized.
///
/// # Errors
///
/// Returns a [`TestCaseError`] if any fixture fails to build, parse, or
/// transform during the one-time construction.
pub fn all_test_cases() -> Result<Vec<TestCase>, TestCaseError> {
    Ok(cached_test_cases()?.to_vec())
}

fn cached_test_cases() -> Result<&'static [TestCase], TestCaseError> {
    static ALL_CASES: OnceLock<Result<Vec<TestCase>, TestCaseError>> = OnceLock::new();
    ALL_CASES
        .get_or_init(build_all_test_cases)
        .as_deref()
        .map_err(Clone::clone)
}

/// Returns every case with at least one origin derived from `fixture`.
///
/// # Errors
///
/// Returns an error if the shared case corpus cannot be built.
pub fn select_test_cases_for_fixture(fixture: TestFixture) -> Result<Vec<TestCase>, TestCaseError> {
    Ok(cached_test_cases()?
        .iter()
        .filter(|case| {
            case.metadata
                .origins
                .iter()
                .any(|origin| origin.fixture == fixture)
        })
        .cloned()
        .collect())
}

/// Returns compilation cases whose aliases fuzzy-match `query`.
///
/// An empty or absent query selects every case. Fixtures tagged
/// `AnalysisOnly` remain in [`all_test_cases`] but are excluded here.
///
/// # Errors
///
/// Returns an error if the shared case corpus cannot be built.
pub fn compile_ready_test_cases(query: Option<&str>) -> Result<Vec<TestCase>, TestCaseError> {
    Ok(cached_test_cases()?
        .iter()
        .filter(|case| {
            case.metadata
                .origins
                .iter()
                .any(|origin| !origin.fixture.in_category(GalleryCategory::AnalysisOnly))
        })
        .filter(|case| query.is_none_or(|query| case.matches_grep(query)))
        .cloned()
        .collect())
}

/// Like [`compile_ready_test_cases`], but errors instead of returning an empty
/// selection.
///
/// # Errors
///
/// Returns an error if `query` selects no compile-ready cases, so a mistyped
/// benchmark query fails loudly rather than silently running nothing.
pub fn required_compile_ready_test_cases(
    query: Option<&str>,
) -> Result<Vec<TestCase>, TestCaseError> {
    let cases = compile_ready_test_cases(query)?;
    if cases.is_empty() {
        let query = query
            .map(str::trim)
            .filter(|query| !query.is_empty())
            .unwrap_or("<all>");
        return Err(TestCaseError(format!(
            "not found: compile-ready case query {query:?} selected no cases"
        )));
    }
    Ok(cases)
}

/// Looks up a single case by an exact alias (ASCII case-insensitive).
///
/// # Errors
///
/// Returns an error if no alias matches `name`.
pub fn find_test_case(name: &str) -> Result<TestCase, TestCaseError> {
    cached_test_cases()?
        .iter()
        .find(|case| {
            case.aliases()
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
        })
        .cloned()
        .ok_or_else(|| {
            TestCaseError(format!(
                "not found: no test case matched exact name {name:?}"
            ))
        })
}

fn allowed_rotations(fixture: TestFixture, graph: &BlockGraph) -> Vec<RotationSpec> {
    // Rotation is unsupported for graphs with dynamic (T/selective) blocks.
    if graph.t_count() > 0 || graph.selective_count() > 0 {
        return Vec::new();
    }
    let turns: &[i32] = if graph.y_count() > 0 {
        &[2]
    } else {
        &[1, 2, 3]
    };
    // Named readouts and their consumers keep the physical time axis.
    if graph.has_actions() {
        return turns
            .iter()
            .map(|&quarter_turns| RotationSpec {
                axis: UDirection::Z,
                quarter_turns,
            })
            .collect();
    }
    // A spatial Hadamard wall keeps half turns about X/Y (odd turns move the
    // wall into time) and all turns about Z (which swap the wall axes).
    if has_spatial_hadamard_pipe(graph) {
        return [UDirection::X, UDirection::Y]
            .into_iter()
            .map(|axis| RotationSpec {
                axis,
                quarter_turns: 2,
            })
            .chain((1..=3).map(|quarter_turns| RotationSpec {
                axis: UDirection::Z,
                quarter_turns,
            }))
            .collect();
    }
    let mut rotations = Vec::with_capacity(turns.len() * 3);
    let axes: &[UDirection] = if graph.walking_count() > 0
        || graph.patch_rotation_count() > 0
        || graph.blocks().any(|block| !block.height().is_default())
    {
        &[UDirection::Z]
    } else {
        &[UDirection::X, UDirection::Y, UDirection::Z]
    };
    for &axis in axes {
        for &quarter_turns in turns {
            if matches!(
                fixture,
                TestFixture::CZTemporalH
                    | TestFixture::TemporalHadamard
                    | TestFixture::HadamardBellState
            ) && matches!(axis, UDirection::X | UDirection::Y)
                && quarter_turns.rem_euclid(2) != 0
            {
                continue;
            }
            rotations.push(RotationSpec {
                axis,
                quarter_turns,
            });
        }
    }
    rotations
}

/// Measured `shortest_graphlike_error` weights for the spatial Hadamard wall
/// fixtures, per code distance, under uniform depolarizing noise at `p = 1e-3`.
///
/// Five measured profiles. The `perpendicular` profile is CZ's first auto-filled
/// variant and keeps the full code distance. Parallel crossings are split by wall axis
/// and sign because the oriented GHZ schedule is not rotation invariant.
///
/// These are observations, not a formula: nothing rules out the perpendicular
/// profile dropping below `d` at a size no one has run, so an unmeasured
/// distance panics rather than extrapolating.
const WALL_GRAPHLIKE_DISTANCES: WallDistanceProfiles = WallDistanceProfiles {
    perpendicular: &[(3, 3), (5, 5), (7, 7), (9, 9)],
    parallel_x_positive: &[(3, 2), (5, 4), (7, 6), (9, 7)],
    parallel_x_negative: &[(3, 2), (5, 4), (7, 5), (9, 7)],
    parallel_y_positive: &[(3, 2), (5, 4), (7, 6), (9, 7)],
    parallel_y_negative: &[(3, 2), (5, 4), (7, 5), (9, 7)],
};

/// The measured profiles of [`WALL_GRAPHLIKE_DISTANCES`], each a
/// `(code distance, graphlike weight)` table.
struct WallDistanceProfiles {
    perpendicular: &'static [(u32, u32)],
    parallel_x_positive: &'static [(u32, u32)],
    parallel_x_negative: &'static [(u32, u32)],
    parallel_y_positive: &'static [(u32, u32)],
    parallel_y_negative: &'static [(u32, u32)],
}

/// The frozen graphlike weight for a wall case, or `None` when `origin` is not
/// a wall case (leaving the code distance to apply).
fn wall_graphlike_distance(origin: &TestCaseOrigin, distance: u32) -> Option<u32> {
    let profile = match origin.fixture {
        TestFixture::CZ if origin.fill_variant == Some(1) => WALL_GRAPHLIKE_DISTANCES.perpendicular,
        TestFixture::CZ | TestFixture::SpatialHadamard
            if origin.rotation.is_some_and(|rotation| {
                rotation.axis == UDirection::Y && rotation.quarter_turns.rem_euclid(4) == 2
            }) =>
        {
            WALL_GRAPHLIKE_DISTANCES.parallel_x_negative
        }
        TestFixture::CZ | TestFixture::SpatialHadamard
            if origin.rotation.is_some_and(|rotation| {
                rotation.axis == UDirection::Z && rotation.quarter_turns.rem_euclid(4) == 1
            }) =>
        {
            WALL_GRAPHLIKE_DISTANCES.parallel_y_positive
        }
        TestFixture::CZ | TestFixture::SpatialHadamard
            if origin.rotation.is_some_and(|rotation| {
                rotation.axis == UDirection::Z && rotation.quarter_turns.rem_euclid(4) == 2
            }) =>
        {
            WALL_GRAPHLIKE_DISTANCES.parallel_x_negative
        }
        TestFixture::CZ | TestFixture::SpatialHadamard
            if origin.rotation.is_some_and(|rotation| {
                rotation.axis == UDirection::Z && rotation.quarter_turns.rem_euclid(4) == 3
            }) =>
        {
            WALL_GRAPHLIKE_DISTANCES.parallel_y_negative
        }
        TestFixture::CZ | TestFixture::SpatialHadamard => {
            WALL_GRAPHLIKE_DISTANCES.parallel_x_positive
        }
        _ => return None,
    };
    let measured = profile
        .iter()
        .find_map(|&(code_distance, weight)| (code_distance == distance).then_some(weight))
        .unwrap_or_else(|| {
            panic!(
                "{}: the spatial Hadamard wall's graphlike distance has never been \
                 measured at d={distance}; measure it and freeze it in \
                 WALL_GRAPHLIKE_DISTANCES rather than extrapolating",
                origin.name
            )
        });
    Some(measured)
}

/// Whether `graph` carries a spatial Hadamard pipe (a domain wall), which
/// constrains how the case may be rotated (see [`allowed_rotations`]).
fn has_spatial_hadamard_pipe(graph: &BlockGraph) -> bool {
    graph
        .pipes()
        .any(|pipe| pipe.is_hadamard() && pipe.dir().is_spatial())
}

fn build_all_test_cases() -> Result<Vec<TestCase>, TestCaseError> {
    let (mut all_candidates, open_candidates) = build_fixture_cases()?;
    let flipped = all_candidates
        .iter()
        .map(flip_candidate)
        .collect::<Result<Vec<_>, _>>()?;
    all_candidates.extend(flipped);
    let rotated = all_candidates
        .iter()
        .flat_map(|candidate| {
            allowed_rotations(candidate.origin.fixture, &candidate.graph)
                .into_iter()
                .map(move |rotation| rotate_candidate(candidate, rotation))
        })
        .collect::<Result<Vec<_>, _>>()?;
    all_candidates.extend(rotated);

    // Open variants stay unrotated because rotation can invalidate `Auto` roles.
    all_candidates.extend(open_candidates);

    Ok(deduplicate_candidates(all_candidates))
}

fn build_fixture_cases() -> Result<(Vec<Candidate>, Vec<Candidate>), TestCaseError> {
    let mut closed = Vec::new();
    let mut filled = Vec::new();
    let mut open = Vec::new();
    for &fixture in compile_fixtures() {
        let graph = build_fixture_graph(fixture)?;
        if fixture.in_category(GalleryCategory::AnalysisOnly)
            || fixture
                .gallery_item()
                .is_some_and(|item| !item.build().instances.is_empty())
        {
            // Module cases retain their authored hierarchy. Flat auto-fill and
            // rotations belong to the primitive and focused transform suites.
            open.push(Candidate {
                origin: TestCaseOrigin {
                    name: format!(
                        "{}[{}]",
                        fixture.id(),
                        if graph.is_open() { "open" } else { "base" }
                    ),
                    fixture,
                    fill_variant: None,
                    flip_xz_basis: false,
                    rotation: None,
                },
                graph,
            });
            continue;
        }
        if !graph.is_open() {
            closed.push(Candidate {
                origin: TestCaseOrigin {
                    name: format!("{}[base]", fixture.id()),
                    fixture,
                    fill_variant: None,
                    flip_xz_basis: false,
                    rotation: None,
                },
                graph,
            });
            continue;
        }
        let compilable = is_compilable_open_graph(&graph);
        let filled_variants = match graph.fill_ports_auto() {
            Ok(variants) if variants.is_empty() => {
                return Err(TestCaseError(format!(
                    "build: {}: fill_ports_auto returned no closed variants",
                    fixture.id()
                )));
            }
            Ok(variants) => variants,
            // An unsupported static fill can still have a compilable open variant.
            Err(_) if compilable => Vec::new(),
            Err(error) => {
                return Err(TestCaseError(format!(
                    "build: {}: fill_ports_auto failed: {error}",
                    fixture.id()
                )));
            }
        };

        for (variant_index, (filled_graph, _)) in filled_variants.into_iter().enumerate() {
            if filled_graph.is_open() {
                return Err(TestCaseError(format!(
                    "build: {}[fill:{variant_index}]: filled graph remained open",
                    fixture.id()
                )));
            }
            filled.push(Candidate {
                origin: TestCaseOrigin {
                    name: format!("{}[fill:{variant_index}]", fixture.id()),
                    fixture,
                    fill_variant: Some(variant_index),
                    flip_xz_basis: false,
                    rotation: None,
                },
                graph: filled_graph,
            });
        }
        if compilable {
            open.push(Candidate {
                origin: TestCaseOrigin {
                    name: format!("{}[open]", fixture.id()),
                    fixture,
                    fill_variant: None,
                    flip_xz_basis: false,
                    rotation: None,
                },
                graph,
            });
        }
    }
    closed.extend(filled);
    Ok((closed, open))
}

fn flip_candidate(candidate: &Candidate) -> Result<Candidate, TestCaseError> {
    let graph = candidate.graph.flip_xz_basis().map_err(|error| {
        TestCaseError(format!(
            "transform: {}: flip_xz_basis failed: {error}",
            candidate.origin.name
        ))
    })?;
    Ok(Candidate {
        origin: TestCaseOrigin {
            name: format!("{}[flip-xz]", candidate.origin.name),
            fixture: candidate.origin.fixture,
            fill_variant: candidate.origin.fill_variant,
            flip_xz_basis: true,
            rotation: None,
        },
        graph,
    })
}

fn rotate_candidate(
    candidate: &Candidate,
    rotation: RotationSpec,
) -> Result<Candidate, TestCaseError> {
    let graph = candidate
        .graph
        .rotate_about_origin(rotation.axis, rotation.quarter_turns)
        .map_err(|error| {
            TestCaseError(format!(
                "transform: {}: rotation about {:?} by {} quarter turns failed: {error}",
                candidate.origin.name, rotation.axis, rotation.quarter_turns
            ))
        })?;
    Ok(Candidate {
        origin: TestCaseOrigin {
            name: format!(
                "{}[rotate:{:?}:{}]",
                candidate.origin.name, rotation.axis, rotation.quarter_turns
            ),
            fixture: candidate.origin.fixture,
            fill_variant: candidate.origin.fill_variant,
            flip_xz_basis: candidate.origin.flip_xz_basis,
            rotation: Some(rotation),
        },
        graph,
    })
}

fn deduplicate_candidates(candidates: Vec<Candidate>) -> Vec<TestCase> {
    let mut index_by_key = HashMap::<String, usize>::new();
    let mut cases = Vec::<TestCase>::new();

    for candidate in candidates {
        let key = canonical_graph_key(&candidate.graph);
        if let Some(&existing_index) = index_by_key.get(&key) {
            cases[existing_index]
                .metadata
                .merge_origin(candidate.origin);
            continue;
        }

        let name = candidate.origin.name.clone();
        index_by_key.insert(key, cases.len());
        cases.push(TestCase::from_graph(
            TestCaseMetadata {
                name: name.clone(),
                aliases: vec![name],
                origins: vec![candidate.origin],
            },
            candidate.graph,
        ));
    }

    for case in &mut cases {
        if let Some(item) = case.metadata.origins.iter().find_map(|origin| {
            (origin.fill_variant.is_none() && !origin.flip_xz_basis && origin.rotation.is_none())
                .then(|| origin.fixture.gallery_item())
                .flatten()
        }) {
            case.graph = Arc::new(item.build());
        }
        if is_compile_bench_core_case(&case.metadata) {
            case.metadata
                .aliases
                .push(COMPILE_BENCH_CORE_ALIAS.to_string());
        }
    }
    cases
}

fn canonical_graph_key(graph: &BlockGraph) -> String {
    let graph = graph
        .canonical_true_branch_view()
        .expect("stored branch arms rematerialize");
    let mut key = String::new();
    let mut blocks = graph.blocks().cloned().collect::<Vec<_>>();
    blocks.sort_unstable_by_key(|block| block.pos().to_array());
    for block in blocks {
        key.push_str(&block.to_string());
        key.push('\n');
    }

    let mut pipes = graph.pipes().cloned().collect::<Vec<_>>();
    pipes.sort_unstable_by(|left, right| {
        canonical_pipe_sort_key(left).cmp(&canonical_pipe_sort_key(right))
    });
    for pipe in pipes {
        key.push_str(&canonical_pipe_string(&pipe));
        key.push('\n');
    }

    for branch in graph.branch_definitions() {
        key.push_str("branch ");
        key.push_str(&branch.name);
        key.push('\n');
        for (label, arm) in [("false", branch.on_false()), ("true", branch.on_true())] {
            key.push_str(label);
            key.push('\n');
            for block in arm.blocks() {
                key.push_str(&block.to_string());
                key.push('\n');
            }
            for pipe in arm.pipes() {
                key.push_str(&canonical_pipe_string(pipe));
                key.push('\n');
            }
        }
    }

    for node in graph.action_graph().ordered_nodes() {
        key.push_str(&node.action.to_string());
        key.push('\n');
    }
    key
}

fn canonical_pipe_sort_key(pipe: &Pipe) -> ([i32; 3], [i32; 3], bool, &str) {
    let (src, dst) = canonical_pipe_endpoints(pipe);
    (
        src.to_array(),
        dst.to_array(),
        pipe.is_hadamard(),
        pipe.tag().unwrap_or(""),
    )
}

fn canonical_pipe_string(pipe: &Pipe) -> String {
    let (src, dst) = canonical_pipe_endpoints(pipe);
    let dir = Direction::try_from(dst - src).expect("pipe endpoints remain adjacent");
    format!(
        "pipe {}{} {}{}",
        if pipe.is_hadamard() { "H " } else { "" },
        src,
        dir,
        pipe.tag().map_or(String::new(), |tag| format!(" <{tag}>"))
    )
}

fn canonical_pipe_endpoints(pipe: &Pipe) -> (glam::IVec3, glam::IVec3) {
    let (src, dst) = pipe.endpoints();
    if src.to_array() <= dst.to_array() {
        (src, dst)
    } else {
        (dst, src)
    }
}

fn fuzzy_match(text: &str, query: &str) -> bool {
    let normalized_text = normalize_for_match(text);
    let normalized_query = normalize_for_match(query);
    if normalized_query.is_empty() {
        return true;
    }
    if normalized_text.contains(&normalized_query) {
        return true;
    }

    let mut text_chars = normalized_text.chars();
    for query_char in normalized_query.chars() {
        if text_chars
            .find(|text_char| *text_char == query_char)
            .is_none()
        {
            return false;
        }
    }
    true
}

fn normalize_for_match(text: &str) -> String {
    text.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|char| char.to_ascii_lowercase())
        .collect()
}

fn build_fixture_graph(fixture: TestFixture) -> Result<BlockGraph, TestCaseError> {
    if let Some(item) = fixture.gallery_item() {
        return item
            .build()
            .flatten()
            .map_err(|error| TestCaseError(format!("build: {}: {error}", fixture.id())));
    }
    if fixture == TestFixture::OneBitAdder {
        return BlockGraph::from_text(ONE_BIT_ADDER_SOURCE).map_err(|error| {
            TestCaseError(format!("parse: negative bulk-adder fixture: {error}"))
        });
    }
    let source = match fixture {
        TestFixture::CubeLine => include_str!("../assets/cube_line.blog"),
        TestFixture::HShapeJunction => include_str!("../assets/h_shape_junction.blog"),
        TestFixture::HadamardBellState => include_str!("../assets/hadamard_bell_state.blog"),
        TestFixture::SpatialHadamard => include_str!("../assets/spatial_hadamard.blog"),
        TestFixture::StabilityLine => include_str!("../assets/stability_line.blog"),
        TestFixture::TShapeJunction => include_str!("../assets/t_shape_junction.blog"),
        TestFixture::TemporalHadamard => include_str!("../assets/temporal_hadamard.blog"),
        TestFixture::XShapeJunction => include_str!("../assets/x_shape_junction.blog"),
        _ => unreachable!("gallery fixtures returned above"),
    };
    build_embedded_graph(fixture.id(), source)
}

/// Mirrors the compiler's open-graph check without creating a dependency cycle.
fn is_compilable_open_graph(graph: &BlockGraph) -> bool {
    use bloq_graph::BlockKind;

    if !graph.is_open() {
        return false;
    }
    graph
        .blocks()
        .filter(|block| block.kind() == BlockKind::Port)
        .all(|port| {
            // A leaf port has exactly one incident pipe.
            let mut pipe_dir = None;
            for dir in Direction::iter() {
                let endpoint = port.endpoint_for_direction(dir);
                if graph
                    .get_pipe(endpoint, endpoint + dir.to_ivec3())
                    .is_some()
                {
                    if pipe_dir.is_some() {
                        return false;
                    }
                    pipe_dir = Some(dir);
                }
            }
            let Some(dir) = pipe_dir else {
                return false;
            };
            if dir.is_spatial() && port.port_role() == Some(bloq_graph::PortRole::Auto) {
                return false;
            }
            let neighbor = port.endpoint_for_direction(dir) + dir.to_ivec3();
            graph.get_endpoint_block(neighbor).is_some_and(|neighbor| {
                matches!(neighbor.kind(), BlockKind::Cube(_))
                    && (!dir.is_spatial() || neighbor.height_cells() == 1)
            })
        })
}

fn build_embedded_graph(id: &str, blog_text: &str) -> Result<BlockGraph, TestCaseError> {
    BlockGraph::from_blog_text(blog_text)
        .map_err(|error| TestCaseError(format!("parse: embedded fixture {id}: {error}")))
        .map(|graph| graph.fix_shadowed_faces())
}

// `any`, not `all`: deduplication merges a symmetric case's rotated and
// flipped duplicates into the same metadata (e.g. `x_memory[base]` also
// carries its own `[rotate:X:1]` origin), and those extra origins must not
// disqualify the plain base/fill case from the core benchmark set.
fn is_compile_bench_core_case(metadata: &TestCaseMetadata) -> bool {
    metadata
        .origins
        .iter()
        .any(|origin| !origin.flip_xz_basis && origin.rotation.is_none())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_gallery_item_has_a_shared_fixture() {
        for item in GalleryItem::iter() {
            assert_eq!(
                compile_fixtures()
                    .iter()
                    .filter(|fixture| fixture.gallery_item() == Some(item))
                    .count(),
                1,
                "{item}"
            );
        }
    }

    #[test]
    fn analysis_only_fixture_stays_in_corpus_but_out_of_compilation() {
        assert_eq!(TestFixture::OneBitAdder.gallery_item(), None);
        assert!(!TestFixture::OneBitAdder.is_clifford());
        for fixture in compile_fixtures()
            .iter()
            .copied()
            .filter(|fixture| fixture.in_category(GalleryCategory::AnalysisOnly))
        {
            let cases = select_test_cases_for_fixture(fixture).unwrap();
            assert_eq!(cases.len(), 1, "analysis entries retain one native case");
            assert!(cases[0].build().has_module_structure());
            assert!(
                compile_ready_test_cases(Some(fixture.id()))
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn authored_cases_retain_hierarchy_in_their_compiler_input() {
        let case = find_test_case("phase_gradient_k4[open]").expect("gallery case exists");
        let graph = case.build();
        assert!(!graph.instances.is_empty());
        assert!(graph.modules().len() > 1);
        let flat = graph.flatten().expect("oracle projection expands");
        assert!(flat.instances.is_empty());
        assert!(!flat.has_module_structure());
        assert!(flat.block_count() > graph.block_count());
    }

    #[test]
    fn open_variants_exist_and_compile() {
        // CNOT's open form (before filling) has only temporal ports onto cubes, so
        // it must surface as a `cnot[open]` variant within its own fixture group —
        // exactly the "open case as a variant" the backend suite exercises.
        let open = select_test_cases_for_fixture(TestFixture::CNOT)
            .expect("select CNOT cases")
            .into_iter()
            .find(|case| case.id().ends_with("[open]"))
            .expect("CNOT has an open variant");
        let graph = open.build();
        assert!(graph.is_open(), "the [open] variant must stay open");
        assert!(is_compilable_open_graph(&graph));
    }

    #[test]
    fn all_test_cases_are_closed_or_compilable_open() {
        let cases = all_test_cases().expect("build test cases");
        assert!(!cases.is_empty());
        for case in &cases {
            let graph = case.build().flatten().unwrap();
            if graph.has_actions() {
                assert!(
                    case.metadata.origins.iter().all(|origin| origin
                        .rotation
                        .is_none_or(|rotation| rotation.axis == UDirection::Z)),
                    "{} rotates action time",
                    case.id()
                );
            }
            assert!(
                !graph.is_open() || is_compilable_open_graph(&graph),
                "{} is open but not compilable",
                case.id()
            );
        }
    }

    #[test]
    fn basis_flips_include_native_t_postselection_with_its_resource_frames() {
        let comparison = find_test_case("t_comparison[base]")
            .unwrap()
            .build()
            .flatten()
            .unwrap();
        let expected = comparison.flip_xz_basis().unwrap().to_blog_text();
        let cases = select_test_cases_for_fixture(TestFixture::TComparison).unwrap();
        let flipped = cases
            .iter()
            .find(|case| {
                case.metadata.origins.iter().any(|origin| {
                    origin.fill_variant.is_none()
                        && origin.flip_xz_basis
                        && origin.rotation.is_none()
                })
            })
            .expect("native postselected T graph has a flipped case")
            .build();
        assert_eq!(flipped.to_blog_text(), expected);
        assert_eq!(flipped.t_count(), comparison.t_count());
        assert!(
            flipped
                .actions()
                .iter()
                .any(|action| matches!(action, bloq_graph::Action::DiscardIf(_)))
        );
    }

    #[test]
    fn temporal_hadamard_rotation_expansion_skips_odd_x_and_y_turns() {
        let temporal_hadamard_cases = compile_ready_test_cases(Some("temporal_hadamard"))
            .expect("select temporal Hadamard cases");
        assert!(!temporal_hadamard_cases.is_empty());
        assert!(
            temporal_hadamard_cases.iter().all(|case| {
                case.metadata.origins.iter().all(|origin| {
                    origin.rotation.is_none_or(|rotation| {
                        !matches!(rotation.axis, UDirection::X | UDirection::Y)
                            || rotation.quarter_turns.rem_euclid(2) == 0
                    })
                })
            }),
            "temporal_hadamard should not expose odd X/Y quarter-turn rotations"
        );
    }

    #[test]
    fn spatial_hadamard_rotation_expansion_keeps_spatial_axes() {
        for fixture in [TestFixture::CZ, TestFixture::SpatialHadamard] {
            let cases = select_test_cases_for_fixture(fixture).expect("select wall cases");
            assert!(!cases.is_empty(), "{} produced no cases", fixture.id());
            for case in &cases {
                for origin in &case.metadata.origins {
                    let Some(rotation) = origin.rotation else {
                        continue;
                    };
                    assert!(
                        rotation.axis == UDirection::Z || rotation.quarter_turns.rem_euclid(2) == 0,
                        "{}: only Z quarter-turns may be odd",
                        origin.name
                    );
                }
            }
        }
    }

    /// The wall cases pin measured weights; everything else expects `d`.
    #[test]
    fn expected_graphlike_distance_is_the_code_distance_off_the_wall() {
        for case in all_test_cases().expect("build test cases") {
            let is_wall = case.metadata.origins.iter().any(|origin| {
                matches!(
                    origin.fixture,
                    TestFixture::CZ | TestFixture::SpatialHadamard
                )
            });
            for distance in [3, 5, 7, 9] {
                let expected = case.expected_graphlike_distance(distance);
                if is_wall {
                    assert!(
                        expected <= distance,
                        "{}: the wall never exceeds the code distance",
                        case.id()
                    );
                } else {
                    assert_eq!(expected, distance, "{}", case.id());
                }
            }
        }
    }

    /// Pin the four measured wall profiles literally.
    #[test]
    fn wall_cases_pin_their_measured_graphlike_distances() {
        let perpendicular = find_test_case("cz[fill:1]").expect("CZ variant 1");
        let x_positive = find_test_case("cz[fill:0]").expect("CZ variant 0");
        let x_negative = find_test_case("cz[fill:0][rotate:Y:2]").expect("rotated CZ variant 0");
        let wall = find_test_case("spatial_hadamard[base]").expect("wall fixture");
        let y_positive =
            find_test_case("spatial_hadamard[base][rotate:Z:1]").expect("Y wall aligned");
        let y_negative =
            find_test_case("spatial_hadamard[base][rotate:Z:3]").expect("Y wall flipped");
        for (distance, full, positive, negative) in
            [(3, 3, 2, 2), (5, 5, 4, 4), (7, 7, 6, 5), (9, 9, 7, 7)]
        {
            assert_eq!(perpendicular.expected_graphlike_distance(distance), full);
            assert_eq!(x_positive.expected_graphlike_distance(distance), positive);
            assert_eq!(x_negative.expected_graphlike_distance(distance), negative);
            assert_eq!(wall.expected_graphlike_distance(distance), positive);
            assert_eq!(y_positive.expected_graphlike_distance(distance), positive);
            assert_eq!(y_negative.expected_graphlike_distance(distance), negative);
        }
    }

    #[test]
    fn grep_matches_subsequence_and_exact_aliases() {
        assert!(
            !compile_ready_test_cases(Some("cube_line"))
                .unwrap()
                .is_empty()
        );
        assert!(!compile_ready_test_cases(Some("clb")).unwrap().is_empty());
        let mut selected = find_test_case("cube_line[base]").expect("exact aliases resolve");
        let expected = selected.metadata.clone();
        selected.metadata.aliases.clear();
        selected.metadata.origins.clear();
        assert_eq!(
            find_test_case("CUBE_LINE[BASE]").unwrap().metadata,
            expected
        );
    }

    #[test]
    fn deduplicated_case_names_are_unique() {
        let cases = all_test_cases().expect("build test cases");
        let mut names = cases
            .iter()
            .map(|case| case.id().to_string())
            .collect::<Vec<_>>();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn canonical_key_ignores_displayed_branch_arm() {
        let mut graph = GalleryItem::CCZGateTeleport.build().flatten().unwrap();
        let expected = canonical_graph_key(&graph);
        let name = graph.branch_definitions()[0].name.clone();
        graph.set_shown_branch_arm(&name, false).unwrap();
        assert_eq!(canonical_graph_key(&graph), expected);
    }

    #[test]
    fn compile_bench_core_alias_selects_unrotated_compile_ready_cases() {
        let cases = compile_ready_test_cases(Some(COMPILE_BENCH_CORE_ALIAS))
            .expect("load compile-ready benchmark-core cases");
        assert!(
            !cases.is_empty(),
            "benchmark-core subset should not be empty"
        );
        assert!(cases.iter().all(|case| {
            case.aliases()
                .contains(&COMPILE_BENCH_CORE_ALIAS.to_string())
        }));
        assert!(cases.iter().all(|case| {
            case.metadata
                .origins
                .iter()
                .any(|origin| !origin.flip_xz_basis && origin.rotation.is_none())
        }));
        // Symmetric base cases whose graphs also match rotated origins stay
        // in the core set.
        for expected in ["x_memory[base]", "stability[base]", "cube_line[base]"] {
            assert!(
                cases.iter().any(|case| case.id() == expected),
                "bench-core should include {expected}"
            );
        }
    }
}
