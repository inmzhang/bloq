use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bloq_graph::{
    Basis, Block, BlockGraph, BlockKind, ModuleCertificationError, ModuleCertificationLimits,
    ModuleSummary, PortRole, RuntimeStabilizerBasis, StabilizerGenerators, default_module_jobs,
};
use glam::{IVec2, IVec3};
use web_time::Instant;

use crate::block::{
    LoweringTemplateId, SelectiveTemplates, SpatialHadamardKey, WallSide, compile_fixed_bulk,
    compile_multiplex_port, compile_realignment, compile_selective, compile_spatial_hadamard,
    validate_fixed_bulk,
};
use crate::config::CompileConfig;
use crate::lower::{PhysicalInput, lower_clifford_proxy};
use crate::signature::{
    BlockSignature, BlockSignatureMap, block_signatures, derive_block_signature,
    join_component_layer_schedules, relocatable_block_signatures,
};
use crate::spatial_port::{SpatialPortExpansionMap, expand_spatial_ports};
use crate::{Bloq, CompileError, Connectivity};

#[path = "registry.rs"]
mod registry;

/// Compiled template info for every block, keyed by its grid position.
pub(crate) type CompiledTemplateMap = crate::FxMap<IVec3, CompiledTemplateInfo>;
type CompiledDefinitionVariantMap =
    crate::FxMap<IVec3, Vec<(BlockSignature, CompiledTemplateInfo)>>;
/// Validated semantic input and signatures for the pinned Clifford oracle.
struct ProxyPreflight {
    graph: BlockGraph,
    signatures: BlockSignatureMap,
    stabilizers: StabilizerGenerators,
    spatial_ports: SpatialPortExpansionMap,
}

/// Metadata key for the code distance used to compile a [`Bloq`].
pub const CODE_DISTANCE_METADATA_KEY: &str = "bloq_compile.code_distance";
/// Metadata key for the layout convention used to compile a [`Bloq`].
pub const CONVENTION_METADATA_KEY: &str = "bloq_compile.convention";
/// Metadata key for the seed used to sample a random Clifford proxy path.
pub const CLIFFORD_PROXY_SEED_METADATA_KEY: &str = "bloq_compile.clifford_proxy_seed";

/// The `z` shift that `BlockGraph::with_zero_min_z` applies to `graph`.
///
/// Module summaries and site relocations are both expressed in the source
/// module's coordinates, so both must be moved by exactly this offset before
/// they line up with the normalized graph.
fn module_normalization_offset(
    graph: &BlockGraph,
    module: &ModuleSummary,
) -> Result<IVec3, CompileError> {
    let min_z = graph.spans().map_or(0, |(_, _, span)| *span.start());
    let z = i32::try_from(-i64::from(min_z)).map_err(|_| {
        ModuleCertificationError::InvalidConnection {
            module: module.name().to_string(),
            message: "module normalization exceeds i32 coordinates".to_string(),
        }
    })?;
    Ok(IVec3::new(0, 0, z))
}

/// Qubit-layout offset of the block at grid position `block_xy`:
/// `block_xy · (2·distance + 2)`.
pub(crate) fn block_xy_offset(block_xy: IVec2, distance: u32) -> Result<IVec2, CompileError> {
    let stride = 2 * i128::from(distance) + 2;
    let x = i32::try_from(i128::from(block_xy.x) * stride)
        .map_err(|_| CompileError::BlockLayoutCoordinateOverflow { block_xy, distance })?;
    let y = i32::try_from(i128::from(block_xy.y) * stride)
        .map_err(|_| CompileError::BlockLayoutCoordinateOverflow { block_xy, distance })?;
    Ok(IVec2::new(x, y))
}

/// Block-grid → qubit-layout mapping: a block at grid `pos` places its template
/// at `offset(pos) = pos.xy * (2·distance + 2)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockLayout {
    distance: u32,
}

impl BlockLayout {
    pub(crate) fn new(distance: u32) -> Self {
        Self { distance }
    }

    pub(crate) fn distance(self) -> u32 {
        self.distance
    }

    pub(crate) fn offset(self, pos: IVec3) -> Result<IVec2, CompileError> {
        block_xy_offset(pos.truncate(), self.distance)
    }
}

/// The output of a successful compilation: the lowered [`Bloq`] program, how
/// long the compile took, and any advisories the compiler raised about it.
#[derive(Debug, Clone)]
pub struct CompileArtifacts {
    /// The compiled backend IR program.
    pub bloq: Bloq,
    /// Wall-clock time spent in the compile or link call returning this artifact.
    pub compile_duration: Duration,
    /// Advisories about the compiled program: constructions that compile but
    /// whose circuit does not deliver everything the configuration asks for
    /// (currently only [`spatial_hadamard_distance_warning`]).
    ///
    /// In-band so a front end cannot silently forget to ask — the free
    /// functions stay available for pre-flight, before a compile exists.
    ///
    /// [`spatial_hadamard_distance_warning`]: crate::spatial_hadamard_distance_warning
    pub warnings: Vec<&'static str>,
}

/// A compiled block graph for one physical target, ready for repeated linking.
///
/// It shares the finished IR and advisories. Linking returns an independently
/// editable snapshot without repeating certification, template compilation, or
/// lowering.
pub struct CompiledObject {
    config: CompileConfig,
    artifacts: Arc<CompileArtifacts>,
}

impl std::fmt::Debug for CompiledObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledObject")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CompiledObject {
    /// Physical target this object was built for.
    #[must_use]
    pub fn config(&self) -> CompileConfig {
        self.config
    }
}

/// Pinned Clifford input to the physical oracle's lowering.
struct PreparedProxy {
    graph: BlockGraph,
    stabilizers: StabilizerGenerators,
    spatial_ports: SpatialPortExpansionMap,
    compiled: CompiledTemplateMap,
    spatial_port_templates: CompiledTemplateMap,
    plan: crate::lower::LinkedPlanInput,
    temporal_templates: crate::FxMap<crate::lower::TemplatePlan, LoweringTemplateId>,
    wall_templates: crate::FxMap<crate::lower::SpatialPipeRef, LoweringTemplateId>,
    padding_templates: crate::padding::PreparedPaddingTemplates,
    template_pool: crate::block::LoweringTemplatePool,
}

pub(crate) struct PreparedDefinitionObject {
    name: String,
    dependencies: Vec<Arc<PreparedDefinitionObject>>,
    variants: CompiledDefinitionVariants,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DefinitionVariantKey {
    local_position: IVec3,
    signature: BlockSignature,
    graph_connectivity: Connectivity,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DefinitionObjectCacheKey {
    implementation: String,
    orientations: Vec<bloq_graph::ModuleOrientation>,
}

#[derive(Debug, Clone)]
struct PreparedSiteRelocation {
    position: IVec3,
    definition: usize,
    local_position: IVec3,
}

type CompiledDefinitionVariants = crate::FxMap<DefinitionVariantKey, CompiledTemplateInfo>;
fn build_definition_objects(
    cache: &crate::cache::TemplateCacheShard,
    names: &[String],
    mut parts_by_name: std::collections::HashMap<String, CompiledDefinitionVariants>,
    mut built: std::collections::HashMap<String, Arc<PreparedDefinitionObject>>,
    definition_keys: &std::collections::HashMap<String, String>,
    orientations_by_name: &std::collections::HashMap<String, Vec<bloq_graph::ModuleOrientation>>,
    program: &BlockGraph,
) -> Vec<Arc<PreparedDefinitionObject>> {
    while built.len() < names.len() {
        let before = built.len();
        for name in names {
            if built.contains_key(name) {
                continue;
            }
            let definition = program
                .module(name)
                .expect("reachable cache keys name validated definitions");
            let mut dependency_names = definition
                .instances
                .iter()
                .map(|instance| instance.definition.as_str())
                .collect::<Vec<_>>();
            dependency_names.sort_unstable();
            dependency_names.dedup();
            if dependency_names
                .iter()
                .any(|dependency| !built.contains_key(*dependency))
            {
                continue;
            }
            let dependencies = dependency_names
                .into_iter()
                .map(|dependency| Arc::clone(&built[dependency]))
                .collect();
            let variants = parts_by_name
                .remove(name)
                .expect("every definition has a retained object or compiled parts");
            let object = PreparedDefinitionObject {
                name: name.clone(),
                dependencies,
                variants,
            };
            let object = cache.insert_definition_object(
                DefinitionObjectCacheKey {
                    implementation: definition_keys[name].clone(),
                    orientations: orientations_by_name[name].clone(),
                },
                Arc::new(object),
            );
            built.insert(name.clone(), object);
        }
        assert!(
            built.len() > before,
            "validated module dependencies are acyclic"
        );
    }
    let definitions = names
        .iter()
        .map(|name| Arc::clone(&built[name]))
        .collect::<Vec<_>>();
    debug_assert!(definitions.iter().all(|object| {
        object
            .dependencies
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name)
    }));
    definitions
}

fn relocate_definition_sites(
    definitions: &[Arc<PreparedDefinitionObject>],
    linked_positions: impl IntoIterator<Item = IVec3>,
    module_sites: &std::collections::HashMap<IVec3, bloq_graph::MaterializedModuleSite>,
) -> Vec<PreparedSiteRelocation> {
    let definition_by_name = definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| (definition.name.as_str(), index))
        .collect::<std::collections::HashMap<_, _>>();
    let mut relocations = linked_positions
        .into_iter()
        .map(|position| {
            let site = &module_sites[&position];
            PreparedSiteRelocation {
                position,
                definition: definition_by_name[site.definition.as_str()],
                local_position: site.local_position,
            }
        })
        .collect::<Vec<_>>();
    relocations.sort_unstable_by_key(|site| site.position.to_array());
    relocations
}

fn select_link_signatures(
    config: CompileConfig,
    graph: &BlockGraph,
    spatial_ports: &SpatialPortExpansionMap,
    #[cfg(test)] builds: &std::sync::atomic::AtomicUsize,
) -> Result<BlockSignatureMap, CompileError> {
    #[cfg(test)]
    builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let schedules = join_component_layer_schedules(graph)?;
    let mut signatures = block_signatures(graph, &schedules, config.code_distance())?;
    for expansion in spatial_ports.values() {
        let signature = signatures
            .get_mut(&expansion.source)
            .expect("signature map contains the derived cube");
        signature.connectivity = signature.connectivity.with_pipe(expansion.cube_pipe_dir());
    }
    for &signature in signatures.values() {
        validate_fixed_bulk(signature)?;
    }
    Ok(signatures)
}

fn select_module_link_templates_with_schedule(
    config: CompileConfig,
    graph: &BlockGraph,
    spatial_ports: &SpatialPortExpansionMap,
    definitions: &[Arc<PreparedDefinitionObject>],
    sites: &[PreparedSiteRelocation],
    schedules: &crate::signature::LayerScheduleMap,
    t_sides: &crate::FxMap<IVec3, bloq_graph::Direction>,
) -> Result<CompiledTemplateMap, CompileError> {
    sites
        .iter()
        .map(|site| {
            let definition = &definitions[site.definition];
            let block = graph
                .get_block(site.position)
                .expect("module relocation names a linked block");
            let mut graph_connectivity = BlockSignature::graph_connectivity(block, graph);
            if let Some(expansion) = spatial_ports.get(&site.position) {
                graph_connectivity = graph_connectivity.with_pipe(expansion.cube_pipe_dir());
            }
            let surgery_side = (block.kind() == BlockKind::T).then(|| t_sides[&site.position]);
            let mut linked_signature = derive_block_signature(
                graph,
                block,
                config.code_distance(),
                crate::signature::layer_schedule_at(schedules, site.position),
                surgery_side,
            )?;
            if let Some(expansion) = spatial_ports.get(&site.position) {
                linked_signature.connectivity = linked_signature
                    .connectivity
                    .with_pipe(expansion.cube_pipe_dir());
            }
            let variant = DefinitionVariantKey {
                local_position: site.local_position,
                signature: linked_signature,
                graph_connectivity,
            };
            let Some(info) = definition.variants.get(&variant) else {
                return Err(CompileError::MissingModulePhysicalVariant {
                    module: definition.name.clone(),
                    local_position: site.local_position,
                    linked_position: site.position,
                    signature: format!(
                        "signature={linked_signature:?}, graph={graph_connectivity:?}"
                    ),
                });
            };
            validate_fixed_bulk(variant.signature)?;
            Ok((site.position, *info))
        })
        .collect()
}

/// Compile `graph` at `distance`, returning the lowered program.
///
/// The one-shot for the common case: a fresh private template cache, no
/// artifacts wrapper, and a validated distance (no panic on user input). Reach
/// for [`CompileContext`] when several compilations should share compiled
/// templates, and for [`compile_with`] when the full [`CompileArtifacts`] are
/// wanted.
///
/// # Errors
///
/// Returns [`CompileError::InvalidDistance`] if `distance` is not odd and in
/// `3..=`[`MAX_CODE_DISTANCE`](crate::MAX_CODE_DISTANCE), and any compilation
/// error the graph provokes.
///
/// # Examples
///
/// ```
/// use bloq_compile::compile;
/// use bloq_graph::GalleryItem;
///
/// let bloq = compile(&GalleryItem::CNOT.build(), 3)?;
/// assert!(bloq.quantum_node_count() > 0);
/// # Ok::<(), bloq_compile::CompileError>(())
/// ```
pub fn compile(graph: &BlockGraph, distance: u32) -> Result<Bloq, CompileError> {
    let config = CompileConfig::try_new(distance)?;
    Ok(compile_with(graph, config)?.bloq)
}

/// Compile `graph` under `config` in a fresh context, returning the full
/// [`CompileArtifacts`].
///
/// # Errors
///
/// Any compilation error the graph provokes under `config`.
///
/// # Examples
///
/// ```
/// use bloq_compile::{CompileConfig, compile_with};
/// use bloq_graph::GalleryItem;
///
/// let artifacts = compile_with(&GalleryItem::CNOT.build(), CompileConfig::default())?;
/// assert!(artifacts.warnings.is_empty());
/// # Ok::<(), bloq_compile::CompileError>(())
/// ```
pub fn compile_with(
    graph: &BlockGraph,
    config: CompileConfig,
) -> Result<CompileArtifacts, CompileError> {
    CompileContext::new(config).compile(graph)
}

/// The code distance `bloq` was compiled at, or `None` if it carries no
/// [`CODE_DISTANCE_METADATA_KEY`] entry (it was not produced by this compiler,
/// or predates the key).
#[must_use]
pub fn compiled_distance(bloq: &Bloq) -> Option<u32> {
    match bloq.metadata().get(CODE_DISTANCE_METADATA_KEY)? {
        bloq_ir::MetadataValue::U64(distance) => u32::try_from(*distance).ok(),
        bloq_ir::MetadataValue::String(_) => None,
    }
}

/// A compiled block's [`TemplateRef`] together with the graph connectivity it
/// was compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompiledTemplateInfo {
    pub(crate) graph_connectivity: Connectivity,
    pub(crate) template: TemplateRef,
}

/// How a compiled block's template(s) live in the pool.
///
/// A fixed block has one template. A selective block has two measurement arms —
/// one per resolved Pauli basis — in one guarded component.
/// Lowering classifies each plan node once (`lower::classify_nodes`), so the
/// emission passes never re-derive this.
/// A T block has its cultivation + escape stage pair; lowering turns
/// these into a `RepeatUntilSuccess` region whose escape instance is the
/// block's observable-facing template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemplateRef {
    Fixed(LoweringTemplateId),
    Selective {
        when_true: LoweringTemplateId,
        when_false: LoweringTemplateId,
    },
    T {
        cultivation: LoweringTemplateId,
        escape: LoweringTemplateId,
    },
}

impl TemplateRef {
    /// Whether this block lowers to a guarded selective pair. Static observable
    /// lookup skips these; guarded lookup resolves each arm separately.
    pub(crate) fn is_selective(self) -> bool {
        matches!(self, TemplateRef::Selective { .. })
    }

    /// The template the block's observable gateway lives on: the fixed template,
    /// or a T block's escape template (its output face is the escaped patch).
    /// `None` for selectives, whose observables resolve per pinned arm.
    pub(crate) fn observable_template(self) -> Option<LoweringTemplateId> {
        match self {
            TemplateRef::Fixed(id) => Some(id),
            TemplateRef::T { escape, .. } => Some(escape),
            TemplateRef::Selective { .. } => None,
        }
    }
}

/// A stage entered by a compilation. Stages can repeat if a planner retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileStage {
    /// Validate the source and target limits.
    Validation,
    /// Certify module definitions.
    Certification,
    /// Build the logical correlation model.
    Correlations,
    /// Schedule the physical geometry.
    Placement,
    /// Compile templates, bind physical instances, and compose flows.
    Templates,
    /// Bind physical measurements to logical readouts.
    Readouts,
    /// Add occupancy order, optimize IR, bind logical cuts, and check layout.
    Optimization,
    /// Use a previously compiled module object.
    CacheReuse,
    /// Compilation succeeded.
    Complete,
}

impl std::fmt::Display for CompileStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Validation => "Validating source",
            Self::Certification => "Certifying module",
            Self::Correlations => "Planning correlations",
            Self::Placement => "Placing physical blocks",
            Self::Templates => "Binding physical program",
            Self::Readouts => "Lowering readouts",
            Self::Optimization => "Finalizing program",
            Self::CacheReuse => "Reusing cached module",
            Self::Complete => "Compilation complete",
        })
    }
}

struct ProgressObserver(Box<dyn Fn(CompileStage) + Send + Sync>);

impl std::fmt::Debug for ProgressObserver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ProgressObserver(..)")
    }
}

/// The main compilation context. Orchestrates the full pipeline:
/// BlockGraph -> Bloq.
///
/// Compiled circuit templates live in the context's cache shard, which
/// [`new`](Self::new) keeps private and
/// [`with_shared_cache`](Self::with_shared_cache) shares with other contexts —
/// see [`SharedCompileCache`](crate::SharedCompileCache).
#[derive(Debug)]
pub struct CompileContext {
    config: CompileConfig,
    cache: Arc<crate::cache::TemplateCacheShard>,
    summaries: Arc<crate::cache::ModuleSummaryCache>,
    clifford_proxy: Option<CliffordProxyPins>,
    progress_observer: Option<ProgressObserver>,
    cancellation: Option<bloq_graph::CancellationToken>,
    #[cfg(test)]
    flat_signature_builds: std::sync::atomic::AtomicUsize,
}

/// Per-site branch pins for the Clifford-proxy distance oracle:
/// one bit per selective block, consumed in [`ordered_blocks`] order.
#[derive(Debug)]
struct CliffordProxyPins {
    pins: Vec<bool>,
    purpose: CliffordProxyPurpose,
    source_summary: Option<Arc<ModuleSummary>>,
}

/// Which consumer requested a pinned Clifford artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliffordProxyPurpose {
    /// Static distance oracle: replace a T region with a perfect input port.
    Distance,
    /// Detector-slice viewer: retain the T region so its physical timeline is
    /// inspectable; the slice tracker applies each T gate's Clifford proxy.
    Detslice,
}

/// Compile `graph` with every dynamic element replaced by a Clifford stand-in.
///
/// This is a **distance oracle, not a semantics oracle**. Exact graph-level ZX
/// maps remain the logical-correctness oracle:
///
/// - Every T block becomes a perfect MPP **input port** on its escaped-patch
///   footprint: a noiseless preparation of the `d×d` patch (one MPP record
///   per stabilizer, closing the +Z neighbour's seam like any memory face)
///   whose passthrough gateway anchors a terminating logical at the past
///   boundary via the ordinary boundary `OBSERVABLE_INCLUDE` MPP. A logical
///   `Y` request resolves through the gateway's X·Z composition — msc-ls's
///   S-for-T substitution lifted from per-gate to whole-block; the magic
///   state's phase does not affect circuit distance. The cultivation/escape
///   circuits are dropped wholesale (the standalone harness owns their
///   internal distance).
/// - Every selective block is statically pinned to one arm per `pins` bit
///   (deterministic `ordered_blocks` order), eliminating the dynamic choice
///   dispatch; sweeping pin combos covers the branch space.
/// - Structural `Branch` actions are rejected: these pins describe selective
///   arms only and cannot choose an authored graph region.
/// - The action track (feedback, conditional-S corrections) is dropped —
///   Clifford frame corrections cannot change circuit distance.
/// - Hierarchical inputs reuse their composed source certificate; topology is
///   expanded only for this physical oracle.
/// - Source stabilizer rows are reused and resolved through the pinned fills,
///   without re-deriving a basis from the rewritten graph. Output frames use
///   those resolved rows without replaying source actions.
///
/// The result is a fully determined program the static Stim backend accepts,
/// suitable for `shortest_graphlike_error` / undetectable-error searches.
///
/// # Errors
///
/// Returns [`CompileError::CliffordProxyPinCount`] unless `pins` has exactly one
/// bit per selective block in the expanded graph, and any compilation error the
/// rewritten graph provokes. Pins follow expanded `(z, x, y)` block order.
///
/// # Examples
///
/// ```
/// use bloq_compile::{CompileConfig, compile_clifford_proxy};
/// use bloq_graph::GalleryItem;
///
/// // One pin per selective block; the T gallery graph has one.
/// let proxy = compile_clifford_proxy(CompileConfig::default(), &GalleryItem::T.build(), &[false])?;
/// // Fully Clifford, so the static Stim backend accepts it.
/// bloq_stim::emit_bloq_stim(&proxy.bloq).expect("Clifford proxy emits static Stim");
/// # Ok::<(), bloq_compile::CompileError>(())
/// ```
pub fn compile_clifford_proxy(
    config: CompileConfig,
    graph: &BlockGraph,
    pins: &[bool],
) -> Result<CompileArtifacts, CompileError> {
    compile_proxy(config, graph, pins, CliffordProxyPurpose::Distance)
}

/// Compile one seeded random static Clifford proxy path.
///
/// Shared selective controls are sampled jointly. The seed is stored under
/// [`CLIFFORD_PROXY_SEED_METADATA_KEY`]; T blocks use the usual proxy ports.
/// Structural branches are rejected before random selective resolution drops
/// the source action track.
///
/// # Errors
///
/// Returns graph validation, proxy selection, or compilation errors.
pub fn compile_random_clifford_proxy(
    config: CompileConfig,
    graph: &BlockGraph,
    seed: u64,
) -> Result<CompileArtifacts, CompileError> {
    let flat = flat_proxy_graph(config, graph)?;
    let source = flat.as_ref().unwrap_or(graph);
    let pins = random_proxy_pins(source, seed)?;
    let mut artifacts = compile_pinned_clifford_proxy(
        config,
        source,
        &pins,
        CliffordProxyPurpose::Distance,
        graph.has_module_structure().then_some(graph),
    )?;
    artifacts.bloq.insert_metadata(
        CLIFFORD_PROXY_SEED_METADATA_KEY,
        bloq_ir::MetadataValue::U64(seed),
    );
    Ok(artifacts)
}

fn random_proxy_pins(graph: &BlockGraph, seed: u64) -> Result<Vec<bool>, CompileError> {
    // Random projection drops every action, so reject unsupported structural
    // composition before that rewrite can hide the Branch source.
    reject_structural_proxy(graph)?;
    let (_, replacements) = graph.randomly_resolve_selectives(seed)?;
    let pins = ordered_blocks(graph)
        .into_iter()
        .filter_map(|block| {
            let BlockKind::Selective(kind) = block.kind() else {
                return None;
            };
            let resolved = replacements
                .get(block)
                .expect("random resolution replaces every selective block");
            let chosen = match resolved.kind() {
                BlockKind::Y => bloq_graph::PauliBasis::Y,
                BlockKind::Measurement(basis) => basis.into(),
                _ => unreachable!("a resolved selective is a Y or measurement block"),
            };
            Some(chosen == kind.pauli_if_true())
        })
        .collect::<Vec<_>>();
    Ok(pins)
}

/// Compile a pinned detector-slice proxy.
///
/// Selectives resolve the source stabilizer basis, while T regions remain for
/// display and are tracked with the slice engine's T-to-S substitution.
///
/// # Errors
///
/// Returns [`CompileError::CliffordProxyPinCount`] unless `pins` has exactly one
/// bit per selective block in the expanded graph, and any compilation error the
/// rewritten graph provokes. Pins follow expanded `(z, x, y)` block order.
pub fn compile_detslice_proxy(
    config: CompileConfig,
    graph: &BlockGraph,
    pins: &[bool],
) -> Result<CompileArtifacts, CompileError> {
    compile_proxy(config, graph, pins, CliffordProxyPurpose::Detslice)
}

fn flat_proxy_graph(
    config: CompileConfig,
    graph: &BlockGraph,
) -> Result<Option<BlockGraph>, CompileError> {
    if !graph.has_module_structure() {
        return Ok(None);
    }
    Ok(Some(
        graph.flatten_with_limits(config.certification_limits())?,
    ))
}

fn compile_proxy(
    config: CompileConfig,
    graph: &BlockGraph,
    pins: &[bool],
    purpose: CliffordProxyPurpose,
) -> Result<CompileArtifacts, CompileError> {
    let flat = flat_proxy_graph(config, graph)?;
    compile_pinned_clifford_proxy(
        config,
        flat.as_ref().unwrap_or(graph),
        pins,
        purpose,
        graph.has_module_structure().then_some(graph),
    )
}

fn compile_pinned_clifford_proxy(
    config: CompileConfig,
    graph: &BlockGraph,
    pins: &[bool],
    purpose: CliffordProxyPurpose,
    program: Option<&BlockGraph>,
) -> Result<CompileArtifacts, CompileError> {
    let expected = graph
        .blocks()
        .filter(|block| block.kind().is_selective())
        .count();
    if pins.len() != expected {
        return Err(CompileError::CliffordProxyPinCount {
            expected,
            actual: pins.len(),
        });
    }
    static CACHE: OnceLock<crate::SharedCompileCache> = OnceLock::new();
    // Keep proxy templates separate from real compilations. Pinning only
    // selects one arm of a cached selective template; T proxy ports use their
    // ordinary Port signature below, so no cached entry is site-specific.
    let mut ctx = CompileContext::with_shared_cache(
        config,
        CACHE.get_or_init(crate::SharedCompileCache::new),
    );
    let source_summary = program
        .map(|program| ctx.summarize(program, config.certification_limits()))
        .transpose()?;
    ctx.clifford_proxy = Some(CliffordProxyPins {
        pins: pins.to_vec(),
        purpose,
        source_summary,
    });
    ctx.compile(graph)
}

/// Cache key for the unified signature→template map: a spatial block signature,
/// a temporal Hadamard realignment keyed by its top basis, or a spatial Hadamard
/// wall keyed by its two endpoint cubes' contributions. Collapsing them into one
/// enum lets `sig_to_template` back every template kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Signature {
    Block(BlockSignature),
    MultiplexPort {
        boundary_basis: Basis,
        output_qubit: IVec2,
    },
    Realignment {
        top_basis: Basis,
    },
    SpatialHadamardPipe(SpatialHadamardKey),
}

impl CompileContext {
    /// Creates a context with a private template cache. Templates it compiles
    /// are reused by its own later compilations and by nothing else.
    #[must_use]
    pub fn new(config: CompileConfig) -> Self {
        Self {
            config,
            cache: Arc::new(crate::cache::TemplateCacheShard::default()),
            summaries: Arc::new(crate::cache::ModuleSummaryCache::default()),
            clifford_proxy: None,
            progress_observer: None,
            cancellation: None,
            #[cfg(test)]
            flat_signature_builds: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Creates a context backed by `cache`, so it reuses (and contributes)
    /// templates compiled by every other context sharing that cache at this
    /// [`CompileConfig`]. Contexts sharing a cache may compile concurrently.
    ///
    /// Sharing does not change compiled output: see [`SharedCompileCache`].
    ///
    /// [`SharedCompileCache`]: crate::SharedCompileCache
    #[must_use]
    pub fn with_shared_cache(config: CompileConfig, cache: &crate::SharedCompileCache) -> Self {
        Self {
            config,
            cache: cache.shard(config),
            summaries: cache.summary_cache(),
            clifford_proxy: None,
            progress_observer: None,
            cancellation: None,
            #[cfg(test)]
            flat_signature_builds: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The configuration this context compiles at.
    ///
    /// Front ends that display or re-derive a context's settings read them
    /// here instead of keeping their own copy of what was passed to
    /// [`new`](Self::new).
    #[must_use]
    pub fn config(&self) -> CompileConfig {
        self.config
    }

    /// Observe stage starts for compilations made with this context.
    /// The default context emits no events. Retries may repeat stages; these
    /// events are not a completion fraction. The callback runs synchronously
    /// and should stay cheap. Concurrent calls on one context may interleave.
    #[must_use]
    pub fn with_progress_observer(
        mut self,
        observer: impl Fn(CompileStage) + Send + Sync + 'static,
    ) -> Self {
        self.progress_observer = Some(ProgressObserver(Box::new(observer)));
        self
    }

    /// Makes this context's compilations cooperatively cancellable.
    ///
    /// Cancellation is scoped to this context's request, including native module
    /// workers. Checks occur at stage, template, and normalization boundaries;
    /// no partial artifact is returned. Cached complete templates remain reusable.
    /// Use a new token for a later request.
    #[must_use]
    pub fn with_cancellation(mut self, token: bloq_graph::CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    fn check_cancellation(&self) -> Result<(), CompileError> {
        if let Some(token) = &self.cancellation {
            token.check()?;
        }
        Ok(())
    }

    fn cancellable<T>(
        &self,
        f: impl FnOnce() -> Result<T, CompileError>,
    ) -> Result<T, CompileError> {
        match &self.cancellation {
            Some(token) => token.run(f),
            None => f(),
        }
    }

    pub(crate) fn report_progress(&self, stage: CompileStage) -> Result<(), CompileError> {
        self.check_cancellation()?;
        if let Some(observer) = &self.progress_observer {
            (observer.0)(stage);
        }
        self.check_cancellation()
    }

    /// Certifies and summarizes a module hierarchy, reusing this context's
    /// logical cache independently of physical compile configuration.
    ///
    /// # Errors
    ///
    /// Returns module validation, resource-limit, or certification errors.
    pub fn summarize(
        &self,
        program: &BlockGraph,
        limits: ModuleCertificationLimits,
    ) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
        self.summaries.get_or_summarize(program, limits, None)
    }

    /// [`summarize`](Self::summarize) with an explicit native
    /// module-worker limit. WebAssembly remains serial.
    ///
    /// # Errors
    ///
    /// Returns module validation, resource-limit, or certification errors.
    pub fn summarize_with_jobs(
        &self,
        program: &BlockGraph,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
    ) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
        self.summaries.get_or_summarize(program, limits, Some(jobs))
    }

    /// Compile a target-specific block-graph object for repeated linking.
    /// Both fixed and symbolic inputs cache their finished IR. Exact
    /// definition variants remain reusable across different root programs.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when module validation, certification, or
    /// lowering fails.
    pub fn compile_object(&self, program: &BlockGraph) -> Result<CompiledObject, CompileError> {
        let object = self.compile_object_with_jobs(program, default_module_jobs())?;
        self.report_progress(CompileStage::Complete)?;
        Ok(object)
    }

    fn compile_object_with_jobs(
        &self,
        program: &BlockGraph,
        jobs: NonZeroUsize,
    ) -> Result<CompiledObject, CompileError> {
        self.cancellable(|| {
            self.report_progress(CompileStage::Validation)?;
            let hierarchical = program.has_module_structure();
            if hierarchical {
                program.validate_with_limits(self.config.certification_limits())?;
            } else {
                program.validate_resource_limits(self.config.certification_limits())?;
            }
            let key = (hierarchical, program.to_blog_text());
            let artifacts = if let Some(cached) = self.cache.module_object(&key) {
                self.report_progress(CompileStage::CacheReuse)?;
                cached
            } else {
                self.report_progress(CompileStage::Certification)?;
                let artifacts = if hierarchical {
                    registry::compile_hierarchy(self, program, jobs)?
                } else {
                    registry::compile_graph(self, program)?
                };
                self.check_cancellation()?;
                self.cache.insert_module_object(key, Arc::new(artifacts))
            };
            Ok(CompiledObject {
                config: self.config,
                artifacts,
            })
        })
    }

    fn prepare_module_definitions(
        &self,
        program: &BlockGraph,
        jobs: NonZeroUsize,
    ) -> Result<
        (
            bloq_graph::LinkedModuleDefinition,
            Vec<Arc<PreparedDefinitionObject>>,
        ),
        CompileError,
    > {
        let (definition_keys, seam_faces) = (
            program.module_definition_cache_keys(),
            program.module_public_seam_faces(),
        );
        let mut linked = bloq_graph::flatten_module_definition(program, program.root(), "")
            .map_err(|source| ModuleCertificationError::Graph {
                module: program.root().name.clone(),
                source,
            })?;
        linked.graph = linked.graph.fix_shadowed_faces();
        let definitions = self.compile_module_definition_objects(
            program,
            &linked,
            &definition_keys,
            &seam_faces,
            jobs,
        )?;
        Ok((linked, definitions))
    }

    /// Link a compiled object to an independently editable Bloq snapshot.
    ///
    /// # Errors
    ///
    /// Returns a target mismatch. Compilation errors are reported when building
    /// the object.
    pub fn link_object(&self, object: &CompiledObject) -> Result<CompileArtifacts, CompileError> {
        self.check_cancellation()?;
        if object.config != self.config {
            return Err(CompileError::CompiledObjectTargetMismatch {
                object: Box::new(object.config),
                linker: Box::new(self.config),
            });
        }
        let started_at = Instant::now();
        let mut artifacts = object.artifacts.as_ref().clone();
        artifacts.compile_duration = started_at.elapsed();
        self.check_cancellation()?;
        Ok(artifacts)
    }

    /// Compile `graph`, then explicitly audit the resulting Bloq IR.
    /// The audit inherits this context's Boolean limits; its other resource
    /// limits retain the IR validator's defaults.
    ///
    /// # Errors
    ///
    /// Returns a compilation or full IR validation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_compile::{CompileConfig, CompileContext};
    /// use bloq_graph::GalleryItem;
    ///
    /// let context = CompileContext::new(CompileConfig::default());
    /// let artifacts = context.compile_and_validate(&GalleryItem::CNOT.build())?;
    /// assert!(artifacts.bloq.quantum_node_count() > 0);
    /// # Ok::<(), bloq_compile::CompileError>(())
    /// ```
    pub fn compile_and_validate(
        &self,
        graph: &BlockGraph,
    ) -> Result<CompileArtifacts, CompileError> {
        let artifacts = self.compile(graph)?;
        let options = bloq_ir::lowering::InstantiationOptions::default()
            .with_boolean_limits(self.config.certification_limits().boolean_limits());
        artifacts
            .bloq
            .validate_with_options(&options)
            .map_err(CompileError::from_verification_audit)?;
        Ok(artifacts)
    }

    /// Validate the pinned oracle's source before omitting its action track.
    fn preflight_proxy(&self, graph: &BlockGraph) -> Result<ProxyPreflight, CompileError> {
        graph.validate_resource_limits(self.config.certification_limits())?;
        let proxy = self
            .clifford_proxy
            .as_ref()
            .expect("Clifford proxy context");
        let module_summary = proxy.source_summary.as_deref();
        let graph = graph.canonical_true_branch_view()?;
        graph.validate_source()?;
        reject_structural_proxy(&graph)?;
        let linked_offset = module_summary
            .map(|summary| module_normalization_offset(&graph, summary))
            .transpose()?
            .unwrap_or(IVec3::ZERO);
        let graph = graph.with_zero_min_z()?.fix_shadowed_faces();
        let (source_graph, mut stabilizers) = if let Some(summary) = module_summary {
            let zx = bloq_graph::ZXGraph::from_block_graph_for_analysis(&graph)
                .map_err(bloq_graph::RuntimeBasisError::from)?;
            let stabilizers = summary.materialize_stabilizers(&zx, linked_offset)?;
            (graph.with_analyzed_action_graph(&stabilizers)?, stabilizers)
        } else {
            graph.analyze_actions_with_limits(self.config.certification_limits())?
        };
        let (graph, spatial_ports) =
            expand_spatial_ports(&source_graph, self.config.code_distance(), &stabilizers)?;
        // Module materialization already certifies this on its composed rows.
        if module_summary.is_none() {
            stabilizers.validate_measurements_close_before_outputs_with_limits(
                self.config.certification_limits(),
            )?;
        }
        let changed = stabilizers.prepare_readouts(
            source_graph.action_graph(),
            self.config.certification_limits(),
        )?;
        let graph = if changed {
            graph.with_analyzed_action_graph(&stabilizers)?
        } else {
            graph
        };
        let stabilizers = self.proxy_stabilizers(&graph, &stabilizers)?;
        let signatures = select_link_signatures(
            self.config,
            &graph,
            &spatial_ports,
            #[cfg(test)]
            &self.flat_signature_builds,
        )?;
        Ok(ProxyPreflight {
            graph,
            signatures,
            stabilizers,
            spatial_ports,
        })
    }

    /// Compile a block graph, including its module hierarchy, into Bloq IR.
    ///
    /// Takes `&self`: the template cache is internally synchronized, so
    /// contexts sharing a [`SharedCompileCache`](crate::SharedCompileCache) may
    /// compile concurrently from `&`-borrows.
    ///
    /// # Errors
    ///
    /// Any structural, per-block, or lowering failure `graph` provokes under
    /// this context's [`CompileConfig`].
    pub fn compile(&self, graph: &BlockGraph) -> Result<CompileArtifacts, CompileError> {
        self.cancellable(|| {
            let artifacts = if self.clifford_proxy.is_none() {
                if graph.has_module_structure() {
                    let started_at = Instant::now();
                    let object = self.compile_object_with_jobs(graph, default_module_jobs())?;
                    let mut artifacts = self.link_object(&object)?;
                    artifacts.compile_duration = started_at.elapsed();
                    artifacts
                } else {
                    registry::compile_graph(self, graph)?
                }
            } else {
                let started_at = Instant::now();
                self.report_progress(CompileStage::Validation)?;
                let prepared = self.prepare_proxy(graph)?;
                self.report_progress(CompileStage::Readouts)?;
                let mut artifacts = Self::lower_prepared_proxy(self.config, prepared)?;
                artifacts.compile_duration = started_at.elapsed();
                artifacts
            };
            self.report_progress(CompileStage::Complete)?;
            Ok(artifacts)
        })
    }

    fn compile_module_definition_objects(
        &self,
        program: &BlockGraph,
        linked_root: &bloq_graph::LinkedModuleDefinition,
        definition_keys: &std::collections::HashMap<String, String>,
        seam_faces: &std::collections::HashMap<(String, IVec3), Vec<bloq_graph::Direction>>,
        jobs: NonZeroUsize,
    ) -> Result<Vec<Arc<PreparedDefinitionObject>>, CompileError> {
        let orientations_by_name = program.definition_orientations();
        let mut names = definition_keys.keys().cloned().collect::<Vec<_>>();
        names.sort_unstable();
        // Retain hits until assembly: another context may clear the shared
        // cache while missing definitions are compiled.
        let mut cached = std::collections::HashMap::new();
        let mut missing_names = Vec::new();
        for name in &names {
            let key = DefinitionObjectCacheKey {
                implementation: definition_keys[name].clone(),
                orientations: orientations_by_name[name].clone(),
            };
            if let Some(object) = self.cache.definition_object(&key) {
                cached.insert(name.clone(), object);
            } else {
                missing_names.push(name);
            }
        }
        let compile_one = |name: &String| -> Result<CompiledDefinitionVariants, CompileError> {
            self.check_cancellation()?;
            // Cache hits retain their complete definition object. Materialize
            // descendant geometry only for a missing definition, once per build.
            let mut linked_child;
            let linked = if name == BlockGraph::ENTRY_MODULE {
                linked_root
            } else {
                let definition = program.module(name).expect("validated definition");
                linked_child = bloq_graph::flatten_module_definition(program, definition, "")
                    .map_err(|source| ModuleCertificationError::Graph {
                        module: program.root().name.clone(),
                        source,
                    })?;
                linked_child.graph = linked_child.graph.fix_shadowed_faces();
                &linked_child
            };
            let mut variants = crate::FxMap::default();
            for &orientation in &orientations_by_name[name] {
                self.check_cancellation()?;
                let oriented_graph = linked.graph.with_orientation_lenient(orientation)?;
                let oriented_sites = linked
                    .sites
                    .iter()
                    .map(|(&position, site)| {
                        let mut site = site.clone();
                        site.orientation = orientation.then_orientation(site.orientation);
                        orientation
                            .try_rotate_position(position)
                            .map(|position| (position, site))
                    })
                    .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
                // Native lowering plans the root's readouts again. Child
                // interfaces still need their own C0 proof before linking.
                if name != BlockGraph::ENTRY_MODULE {
                    registry::certify_definition(
                        name,
                        &oriented_graph,
                        &oriented_sites,
                        self.config.certification_limits(),
                    )?;
                }
                let graph = oriented_graph.canonical_true_branch_view()?;
                let graph = graph.fix_shadowed_faces();
                let (graph, spatial_ports) = crate::spatial_port::expand_spatial_port_topology(
                    &graph,
                    self.config.code_distance(),
                )?;
                let local_positions = oriented_sites
                    .iter()
                    .filter(|(_, site)| site.definition == *name && site.instance_path.is_empty())
                    .map(|(&position, _)| position)
                    .collect::<crate::FxSet<_>>();
                let signatures = relocatable_block_signatures(
                    &graph,
                    &local_positions,
                    self.config.code_distance(),
                )?;
                let mut inventories = vec![self.compile_definition_variants(
                    &graph,
                    &signatures,
                    &oriented_sites,
                    seam_faces,
                    &spatial_ports,
                )?];
                let regions = graph.branch_regions()?;
                if !regions.is_empty() {
                    let projected = graph.project_branches_deferred(
                        regions.iter().map(|region| (region.target, false)),
                    )?;
                    let signatures = relocatable_block_signatures(
                        &projected,
                        &local_positions,
                        self.config.code_distance(),
                    )?;
                    inventories.push(self.compile_definition_variants(
                        &projected,
                        &signatures,
                        &oriented_sites,
                        seam_faces,
                        &spatial_ports,
                    )?);
                }

                for inventory in inventories {
                    for (position, entries) in inventory {
                        let site = &oriented_sites[&position];
                        debug_assert_eq!(&site.definition, name);
                        debug_assert!(site.instance_path.is_empty());
                        for (signature, info) in entries {
                            let key = DefinitionVariantKey {
                                local_position: site.local_position,
                                signature,
                                graph_connectivity: info.graph_connectivity,
                            };
                            if let Some(previous) = variants.insert(key, info) {
                                debug_assert_eq!(previous, info);
                            }
                        }
                    }
                }
            }
            Ok(variants)
        };

        let results = bloq_graph::map_jobs(&missing_names, jobs, |&name| compile_one(name));
        let parts_by_name = missing_names
            .into_iter()
            .zip(results)
            .map(|(name, result)| result.map(|parts| (name.clone(), parts)))
            .collect::<Result<_, _>>()?;

        Ok(build_definition_objects(
            &self.cache,
            &names,
            parts_by_name,
            cached,
            definition_keys,
            &orientations_by_name,
            program,
        ))
    }

    fn prepare_proxy(&self, graph: &BlockGraph) -> Result<PreparedProxy, CompileError> {
        let ProxyPreflight {
            graph,
            signatures,
            stabilizers,
            spatial_ports,
        } = self.preflight_proxy(graph)?;
        self.report_progress(CompileStage::Templates)?;
        let mut compiled = self.compile_blocks(&graph, &ordered_blocks(&graph), &signatures)?;
        for expansion in spatial_ports.values() {
            let derived = compiled
                .get_mut(&expansion.source)
                .expect("derived cube was compiled");
            derived.graph_connectivity = derived
                .graph_connectivity
                .with_pipe(expansion.cube_pipe_dir());
        }
        let spatial_port_templates = self.compile_spatial_port_templates(&spatial_ports)?;
        let plan = crate::lower::LinkedPlanInput::from_module_graph(
            &graph,
            &spatial_ports,
            compiled.keys().copied(),
        );
        let temporal_templates = self.compile_temporal_template_plans(plan.temporal_templates())?;
        let wall_templates = self.compile_spatial_template_walls(&graph, plan.spatial_walls())?;
        let needs_padding =
            !spatial_ports.is_empty() || graph.pipes().any(|pipe| !pipe.dir().is_spatial());
        let padding_templates = self
            .cache
            .padding
            .prepare(needs_padding, self.config.code_distance())?;
        let template_pool = self.cache.pool_snapshot();
        Ok(PreparedProxy {
            graph,
            stabilizers,
            spatial_ports,
            compiled,
            spatial_port_templates,
            plan,
            temporal_templates,
            wall_templates,
            padding_templates,
            template_pool,
        })
    }

    fn lower_prepared_proxy(
        config: CompileConfig,
        prepared: PreparedProxy,
    ) -> Result<CompileArtifacts, CompileError> {
        let PreparedProxy {
            graph,
            stabilizers,
            spatial_ports,
            compiled,
            spatial_port_templates,
            plan,
            temporal_templates,
            wall_templates,
            padding_templates,
            template_pool,
        } = prepared;
        let plan = crate::lower::LowerPlan::from_linked_input(&plan);
        let mut bloq = lower_clifford_proxy(
            PhysicalInput {
                graph: &graph,
                plan: &plan,
                compiled: &compiled,
                spatial_port_templates: &spatial_port_templates,
                spatial_ports: &spatial_ports,
                temporal_templates: &temporal_templates,
                wall_templates: &wall_templates,
                template_pool: &template_pool,
                layout: BlockLayout::new(config.code_distance()),
            },
            &stabilizers,
        )?;
        bloq.insert_metadata(CODE_DISTANCE_METADATA_KEY, config.code_distance());
        bloq.insert_metadata(CONVENTION_METADATA_KEY, "fixed-bulk".to_owned());
        // Memory-padding provenance: edge-owned padding templates, so IR-level
        // `Bloq::insert_memory_rounds` works on the saved program alone.
        crate::padding::record_edge_padding(
            &padding_templates,
            &mut bloq,
            &graph,
            config.code_distance(),
        )?;
        Ok(CompileArtifacts {
            bloq,
            compile_duration: Duration::ZERO,
            warnings: crate::config::spatial_hadamard_distance_warning(&graph)
                .into_iter()
                .collect(),
        })
    }

    fn compile_definition_variants(
        &self,
        graph: &BlockGraph,
        signatures: &BlockSignatureMap,
        module_sites: &std::collections::HashMap<IVec3, bloq_graph::MaterializedModuleSite>,
        seam_faces: &std::collections::HashMap<(String, IVec3), Vec<bloq_graph::Direction>>,
        spatial_ports: &SpatialPortExpansionMap,
    ) -> Result<CompiledDefinitionVariantMap, CompileError> {
        let mut inventory = CompiledDefinitionVariantMap::default();
        for (&position, &signature) in signatures {
            let block = graph
                .get_block(position)
                .expect("compiled site belongs to the prepared graph");
            let mut signature = signature;
            let mut exact_graph_connectivity = BlockSignature::graph_connectivity(block, graph);
            if let Some(expansion) = spatial_ports.get(&position) {
                exact_graph_connectivity =
                    exact_graph_connectivity.with_pipe(expansion.cube_pipe_dir());
                signature.connectivity =
                    signature.connectivity.with_pipe(expansion.cube_pipe_dir());
            }
            let site = module_sites
                .get(&position)
                .expect("module materialization records every compiled block");
            let mutable_faces = seam_faces
                .get(&(site.definition.clone(), site.local_position))
                .map(|directions| {
                    directions
                        .iter()
                        .map(|&direction| site.orientation.rotate_direction(direction))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut connectivity = vec![(signature.connectivity, exact_graph_connectivity)];
            for direction in &mutable_faces {
                let direction = *direction;
                let mut next = Vec::with_capacity(connectivity.len() * 2);
                for (template, graph) in connectivity {
                    next.push((
                        if direction.is_spatial() {
                            template.with_plain_pipe(direction)
                        } else {
                            template
                        },
                        graph.with_plain_pipe(direction),
                    ));
                    next.push((
                        if direction.is_spatial() {
                            template.with_hadamard(direction)
                        } else {
                            template
                        },
                        graph.with_hadamard(direction),
                    ));
                }
                connectivity = next;
            }

            let mut candidates = crate::FxSet::default();
            for (template_connectivity, graph_connectivity) in connectivity {
                let schedules: &[crate::signature::LayerSchedule] =
                    if matches!(signature.kind, BlockKind::Cube(_)) {
                        match template_connectivity.hadamard_wall_axes() {
                            (true, true) => continue,
                            (true, false) => &[crate::signature::LayerSchedule::Extended],
                            (false, true) => &[crate::signature::LayerSchedule::ExtendedY],
                            (false, false) => &[
                                crate::signature::LayerSchedule::Compact,
                                crate::signature::LayerSchedule::Padded,
                                crate::signature::LayerSchedule::Extended,
                                crate::signature::LayerSchedule::ExtendedY,
                            ],
                        }
                    } else {
                        &[]
                    };
                if schedules.is_empty() {
                    let mut candidate = signature;
                    candidate.connectivity = template_connectivity;
                    if candidate.kind == BlockKind::T {
                        for side in [
                            bloq_graph::Direction::YMINUS,
                            bloq_graph::Direction::XPLUS,
                            bloq_graph::Direction::YPLUS,
                            bloq_graph::Direction::XMINUS,
                        ] {
                            candidate.surgery_side = Some(side);
                            candidates.insert((candidate, graph_connectivity));
                        }
                    } else {
                        candidates.insert((candidate, graph_connectivity));
                    }
                } else {
                    for &schedule in schedules {
                        let mut candidate = signature;
                        candidate.connectivity = template_connectivity;
                        candidate.layer_schedule = Some(schedule);
                        candidates.insert((candidate, graph_connectivity));
                    }
                }
            }
            if matches!(
                signature.kind,
                BlockKind::Y
                    | BlockKind::Measurement(_)
                    | BlockKind::Port
                    | BlockKind::Selective(_)
                    | BlockKind::T
            ) && mutable_faces
                .iter()
                .any(|direction| !direction.is_spatial())
            {
                candidates = candidates
                    .drain()
                    .flat_map(|(candidate, graph_connectivity)| {
                        [Basis::X, Basis::Z].map(move |basis| {
                            (
                                BlockSignature {
                                    boundary_basis: Some(basis),
                                    ..candidate
                                },
                                graph_connectivity,
                            )
                        })
                    })
                    .collect();
            }
            candidates.insert((signature, exact_graph_connectivity));
            let mut variants = Vec::with_capacity(candidates.len());
            for (candidate, graph_connectivity) in candidates {
                let valid = validate_fixed_bulk(candidate).is_ok();
                if !valid && candidate != signature {
                    continue;
                }
                let template = self.compile_block_signature(candidate)?;
                variants.push((
                    candidate,
                    CompiledTemplateInfo {
                        graph_connectivity,
                        template,
                    },
                ));
            }
            inventory.insert(position, variants);
        }
        Ok(inventory)
    }

    fn compile_blocks(
        &self,
        graph: &BlockGraph,
        blocks: &[&Block],
        signatures: &BlockSignatureMap,
    ) -> Result<CompiledTemplateMap, CompileError> {
        let mut compiled = crate::FxMap::with_capacity_and_hasher(blocks.len(), Default::default());
        let mut proxy_pins = self
            .clifford_proxy
            .as_ref()
            .into_iter()
            .flat_map(|proxy| proxy.pins.iter().copied());

        for &block in blocks {
            let signature = *signatures
                .get(&block.pos())
                .expect("signature map includes every ordered block");
            // Clifford-proxy rewrites are per *site*, not per signature (two
            // same-signature selectives may pin to different arms), so they
            // select an arm after the signature cache.
            let template = if self.clifford_proxy.is_some() {
                let pin = signature.kind.is_selective().then(|| {
                    proxy_pins
                        .next()
                        .expect("proxy pin count was validated before compilation")
                });
                self.proxy_template_ref(signature, pin)?
            } else {
                self.compile_block_signature(signature)?
            };
            compiled.insert(
                block.pos(),
                CompiledTemplateInfo {
                    graph_connectivity: BlockSignature::graph_connectivity(block, graph),
                    template,
                },
            );
        }

        Ok(compiled)
    }

    fn compile_spatial_port_templates(
        &self,
        spatial_ports: &SpatialPortExpansionMap,
    ) -> Result<CompiledTemplateMap, CompileError> {
        let mut compiled = CompiledTemplateMap::default();
        for expansion in spatial_ports.values().copied() {
            self.check_cancellation()?;
            let connectivity = Connectivity::ISOLATED.with_pipe(expansion.port_pipe_dir());
            let template = if expansion.role == PortRole::Multiplex {
                let output_qubit = expansion
                    .output_qubit
                    .expect("Multiplex expansions allocate an output qubit");
                TemplateRef::Fixed(self.cache.get_or_compile_fixed(
                    Signature::MultiplexPort {
                        boundary_basis: expansion.boundary_basis(),
                        output_qubit,
                    },
                    || {
                        compile_multiplex_port(
                            expansion.boundary_basis(),
                            self.config.code_distance(),
                            output_qubit,
                        )
                    },
                )?)
            } else {
                self.compile_block_signature(BlockSignature {
                    kind: BlockKind::Port,
                    rounds: None,
                    connectivity,
                    boundary_basis: Some(expansion.boundary_basis()),
                    layer_schedule: None,
                    surgery_side: None,
                })?
            };
            compiled.insert(
                expansion.source,
                CompiledTemplateInfo {
                    graph_connectivity: connectivity,
                    template,
                },
            );
        }
        Ok(compiled)
    }

    fn compile_block_signature(
        &self,
        signature: BlockSignature,
    ) -> Result<TemplateRef, CompileError> {
        self.check_cancellation()?;
        let template = self.cache.get_or_compile(Signature::Block(signature), || {
            compile_signature(
                signature,
                self.config.code_distance(),
                self.config.prepare_t_with_mpps(),
            )
        })?;
        self.check_cancellation()?;
        Ok(template)
    }

    /// Rewrite one site's template reference under the Clifford proxy
    /// ([`compile_clifford_proxy`]): a selective pins to the arm its next pin
    /// bit selects; a T block becomes a perfect MPP input port on its
    /// escaped-patch face. Fixed blocks pass through.
    fn proxy_template_ref(
        &self,
        signature: BlockSignature,
        pin: Option<bool>,
    ) -> Result<TemplateRef, CompileError> {
        let proxy = self
            .clifford_proxy
            .as_ref()
            .expect("caller checked the proxy is active");
        if signature.kind == BlockKind::T && proxy.purpose == CliffordProxyPurpose::Distance {
            return self.compile_block_signature(BlockSignature {
                kind: BlockKind::Port,
                rounds: None,
                connectivity: Connectivity::ISOLATED.with_pipe(bloq_graph::Direction::ZPLUS),
                boundary_basis: signature.boundary_basis,
                layer_schedule: None,
                surgery_side: None,
            });
        }
        let template = self.compile_block_signature(signature)?;
        match template {
            TemplateRef::Fixed(_) => Ok(template),
            TemplateRef::Selective {
                when_true,
                when_false,
            } => Ok(TemplateRef::Fixed(
                if pin.expect("selective has a proxy pin") {
                    when_true
                } else {
                    when_false
                },
            )),
            t @ TemplateRef::T { .. } if proxy.purpose == CliffordProxyPurpose::Detslice => Ok(t),
            TemplateRef::T { .. } => unreachable!("distance proxies return above"),
        }
    }

    /// Stabilizers of the fixed Clifford program emitted by the proxy: T nodes
    /// are ideal input ports and each selective is filled from its site pin.
    fn proxy_stabilizers(
        &self,
        graph: &BlockGraph,
        stabilizers: &StabilizerGenerators,
    ) -> Result<StabilizerGenerators, CompileError> {
        let proxy = self
            .clifford_proxy
            .as_ref()
            .expect("caller checked the proxy is active");
        let basis = RuntimeStabilizerBasis::from_generators(stabilizers).with_t_nodes_as_ports();
        let mut pins = proxy.pins.iter().copied();
        let mut fills = Vec::new();
        for block in ordered_blocks(graph) {
            let BlockKind::Selective(kind) = block.kind() else {
                continue;
            };
            let pin = pins
                .next()
                .expect("compile_clifford_proxy validated the selective pin count");
            let chosen = if pin {
                kind.pauli_if_true()
            } else {
                kind.pauli_if_false()
            };
            fills.push((block.pos(), chosen));
        }
        debug_assert!(pins.next().is_none());

        // Resolve the source rows in the selected arms; re-solving the pinned
        // graph would replace the observable basis being tested.
        Ok(basis.apply_selective_fills(&fills)?.into_generators())
    }

    fn compile_temporal_template_plans(
        &self,
        plans: impl IntoIterator<Item = crate::lower::TemplatePlan>,
    ) -> Result<crate::FxMap<crate::lower::TemplatePlan, LoweringTemplateId>, CompileError> {
        let mut templates = crate::FxMap::default();
        for template_plan in plans {
            let crate::lower::TemplatePlan { top_basis, .. } = template_plan;
            let template = self.temporal_template(top_basis)?;
            templates.insert(template_plan, template);
        }
        Ok(templates)
    }

    fn compile_spatial_template_walls(
        &self,
        graph: &BlockGraph,
        walls: impl IntoIterator<Item = crate::lower::SpatialPipeRef>,
    ) -> Result<crate::FxMap<crate::lower::SpatialPipeRef, LoweringTemplateId>, CompileError> {
        let mut templates = crate::FxMap::default();
        for wall in walls {
            self.check_cancellation()?;
            let distance = self.config.code_distance();
            let key = spatial_hadamard_key(graph, wall, distance)?;
            let template = self
                .cache
                .get_or_compile_fixed(Signature::SpatialHadamardPipe(key), || {
                    compile_spatial_hadamard(distance, key)
                })?;
            templates.insert(wall, template);
        }
        Ok(templates)
    }

    pub(crate) fn temporal_template(
        &self,
        top_basis: Basis,
    ) -> Result<LoweringTemplateId, CompileError> {
        self.check_cancellation()?;
        let distance = self.config.code_distance();
        let template = self
            .cache
            .get_or_compile_fixed(Signature::Realignment { top_basis }, || {
                compile_realignment(distance, top_basis)
            })?;
        self.check_cancellation()?;
        Ok(template)
    }
}

/// Derive a wall's cache key from its two endpoint cubes.
///
/// The graph validator has already established that the transverse bases flip
/// across the pipe, so the wall may take the negative-axis cube's perpendicular
/// face basis as the boundary basis for both halves. The endpoint check below is
/// defensive because `Walking` signatures omit spatial faces.
fn spatial_hadamard_key(
    graph: &BlockGraph,
    wall: crate::lower::SpatialPipeRef,
    distance: u32,
) -> Result<SpatialHadamardKey, CompileError> {
    let endpoint = |pos: IVec3| -> Result<(&Block, bloq_graph::CubeKind), CompileError> {
        let block = graph
            .get_endpoint_block(pos)
            .ok_or(CompileError::SpatialHadamardUnsupportedEndpoint { pos })?;
        match block.kind() {
            BlockKind::Cube(kind) => Ok((block, kind)),
            _ => Err(CompileError::SpatialHadamardUnsupportedEndpoint { pos }),
        }
    };
    let (minus_block, minus_kind) = endpoint(wall.src)?;
    let (plus_block, plus_kind) = endpoint(wall.dst)?;
    let axis = wall.axis();
    let boundary_basis = match axis {
        bloq_graph::UDirection::X => minus_kind.y(),
        bloq_graph::UDirection::Y => minus_kind.x(),
        bloq_graph::UDirection::Z => unreachable!("spatial Hadamard walls are not temporal"),
    };
    let make_side = |block: &Block, kind: bloq_graph::CubeKind| {
        let connectivity = BlockSignature::template_connectivity(block, graph);
        WallSide {
            temporal_basis: kind.z(),
            connectivity,
        }
    };
    Ok(SpatialHadamardKey {
        axis,
        // The wall runs one round per cube round. Its two endpoints are cubes
        // in one z-layer joined by a spacelike pipe, so they are in the same
        // spatial component and the validator has already established that they
        // share a height — the `max` is belt and braces, not a real choice.
        rounds: crate::signature::cube_rounds(minus_block, distance)?
            .max(crate::signature::cube_rounds(plus_block, distance)?),
        boundary_basis,
        minus: make_side(minus_block, minus_kind),
        plus: make_side(plus_block, plus_kind),
    })
}

/// Compile one block signature into its circuit template(s). A selective block
/// becomes two per-basis templates (one per resolved Pauli), turned into a
/// guarded component by lowering; a T block becomes its cultivation + escape stage
/// pair, turned into a `RepeatUntilSuccess` region. Everything else is a single
/// fixed template.
///
/// Free-standing rather than a method because the cache runs it outside its
/// lock: it takes only the config values it needs, never the context.
fn compile_signature(
    signature: BlockSignature,
    distance: u32,
    prepare_t_with_mpps: bool,
) -> Result<crate::cache::CompiledTemplates, CompileError> {
    use crate::cache::CompiledTemplates;

    if let BlockKind::Selective(kind) = signature.kind {
        let boundary_basis = signature
            .boundary_basis
            .expect("validate ensures a selective block has a boundary basis");
        let SelectiveTemplates {
            when_true,
            when_false,
        } = compile_selective(signature.connectivity, distance, boundary_basis, kind)?;
        return Ok(CompiledTemplates::Selective {
            when_true,
            when_false,
        });
    }
    if signature.kind == BlockKind::T {
        let boundary_basis = signature
            .boundary_basis
            .expect("validate ensures a T block has a boundary basis");
        let side = signature
            .surgery_side
            .expect("block_signatures picks a surgery side for every T block");
        // `boundary_basis` is the escaped patch's y-face (top) basis, exactly
        // as `compile_y` consumes it; the x-face basis is its flip.
        let layout = crate::block::fixed_bulk::t::SurgeryLayout::new(
            side,
            boundary_basis.flip(),
            boundary_basis,
            distance,
        );
        if prepare_t_with_mpps {
            return Ok(CompiledTemplates::Fixed(std::sync::Arc::new(
                crate::block::fixed_bulk::t::prepare_t_with_mpps(&layout)?,
            )));
        }
        let cultivation = crate::block::fixed_bulk::t::steane::cultivation_template(&layout)?;
        let escape = crate::block::fixed_bulk::t::escape::escape_template(&layout)?;
        return Ok(CompiledTemplates::T {
            cultivation: std::sync::Arc::new(cultivation),
            escape: std::sync::Arc::new(escape),
        });
    }
    let template = compile_fixed_bulk(signature, distance)?;
    Ok(CompiledTemplates::Fixed(template))
}

fn ordered_blocks(graph: &BlockGraph) -> Vec<&Block> {
    let mut blocks: Vec<_> = graph.blocks().collect();
    blocks.sort_unstable_by_key(|block| {
        let pos = block.pos();
        (pos.z, pos.x, pos.y)
    });
    blocks
}

/// Clifford proxies pin only selective measurements, so they cannot erase structural
/// runtime branches before dropping the action track.
fn reject_structural_proxy(graph: &BlockGraph) -> Result<(), CompileError> {
    if !graph.branch_regions()?.is_empty() {
        return Err(CompileError::StructuralBranchCliffordProxyUnsupported);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bloq_ir::{LevelPath, MemoryRoundTarget};

    use super::*;
    use crate::signature::LayerSchedule;
    use bloq_graph::{
        Action, BranchArm, CubeKind, Direction, GalleryItem, MeasureTarget, PatchRotationKind,
        Pipe, WalkingBoundaryKind, WalkingKind,
    };
    use glam::ivec3;
    use rstest::rstest;
    // Structural checks that mean "how many blocks lowered" use
    // `Bloq::quantum_node_count`: observables lower to extra classical nodes,
    // so total `node_count()` does not equal the number of blocks.

    #[test]
    fn compilation_entry_points_honor_resource_limits() {
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("gallery graph expands");
        let module = graph.clone().with_inferred_interface().unwrap();
        let config =
            CompileConfig::default().with_certification_limits(ModuleCertificationLimits {
                max_occupied_cells: 0,
                ..ModuleCertificationLimits::DEFAULT
            });
        let context = CompileContext::new(config);
        for error in [
            context.compile(&graph).unwrap_err(),
            context.compile(&module).unwrap_err(),
            compile_clifford_proxy(config, &graph, &[]).unwrap_err(),
        ] {
            assert!(error.is_resource_limited(), "{error}");
            assert!(
                error.to_string().contains("occupied footprint cells"),
                "{error}"
            );
        }

        let config =
            CompileConfig::default().with_certification_limits(ModuleCertificationLimits {
                max_boolean_nodes: 0,
                ..ModuleCertificationLimits::DEFAULT
            });
        let error = CompileContext::new(config)
            .compile(&two_structural_branch_graph())
            .unwrap_err();
        assert!(error.to_string().contains("Boolean nodes"), "{error}");
        assert!(error.is_resource_limited());

        let limits = ModuleCertificationLimits {
            max_boolean_steps: 0,
            ..ModuleCertificationLimits::DEFAULT
        };
        let error = CompileContext::new(CompileConfig::default().with_certification_limits(limits))
            .compile(&graph)
            .unwrap_err();
        assert!(error.is_resource_limited(), "{error}");
        assert_eq!(
            error.resource_limit_help(),
            Some(ModuleCertificationLimits::RESOURCE_LIMIT_HELP)
        );
        assert!(
            error.to_string().contains("with_certification_limits"),
            "{error}"
        );
        CompileContext::new(CompileConfig::default())
            .compile(&graph)
            .unwrap();

        let exhausted = bloq_utils::boolean::BooleanResourceError {
            resource: "Boolean work steps",
            observed: 2,
            limit: 1,
        };
        let parse_error = bloq_graph::ParseError::Graph {
            source: Box::new(bloq_graph::BlockGraphError::from(exhausted)),
            span: None,
        };
        for (error, certification_help) in [
            (CompileError::from(exhausted), true),
            (
                CompileError::from(bloq_ir::BloqValidationError::from(exhausted)),
                false,
            ),
            (
                CompileError::from(bloq_circuit::CircuitError::FlattenResourceLimit {
                    observed: 2,
                    limit: 1,
                }),
                false,
            ),
            (
                CompileError::from(bloq_graph::RuntimeBasisError::from(exhausted)),
                true,
            ),
            (
                CompileError::from(bloq_graph::BlockGraphError::Zx(bloq_graph::ZXError::Graph(
                    Box::new(bloq_graph::BlockGraphError::from(exhausted)),
                ))),
                true,
            ),
            (
                CompileError::from(bloq_graph::ModuleError::Parse(parse_error.clone())),
                true,
            ),
            (
                CompileError::from(bloq_graph::BlockGraphError::Parse(parse_error)),
                true,
            ),
        ] {
            assert!(error.is_resource_limited(), "{error:?}");
            assert_eq!(
                error.resource_limit_help().is_some(),
                certification_help,
                "{error:?}"
            );
        }
        assert!(!CompileError::InvalidDistance(2).is_resource_limited());
        assert!(
            CompileError::InvalidDistance(2)
                .resource_limit_help()
                .is_none()
        );
    }

    #[test]
    fn verification_audit_hints_only_for_inherited_boolean_limits() {
        use bloq_ir::lowering::NodeTemplateInstanceMergeError;
        use bloq_ir::{BloqNodeId, BloqValidationError, TemplateId};

        let exhausted = bloq_utils::boolean::BooleanResourceError {
            resource: "Boolean work steps",
            observed: 2,
            limit: 1,
        };
        for error in [
            BloqValidationError::BooleanResource(exhausted),
            BloqValidationError::InvalidInstanceMergeStructure {
                node: BloqNodeId(3),
                source: NodeTemplateInstanceMergeError::BooleanResource(exhausted),
            },
            BloqValidationError::InvalidTemplateCircuit {
                template: TemplateId(4),
                source: NodeTemplateInstanceMergeError::BooleanResource(exhausted),
            },
        ] {
            let error = CompileError::from_verification_audit(error);
            assert!(
                matches!(&error, CompileError::BooleanResource(resource) if *resource == exhausted)
            );
            assert_eq!(
                error.resource_limit_help(),
                Some(ModuleCertificationLimits::RESOURCE_LIMIT_HELP)
            );
        }
        for (error, resource_limited) in [
            (
                BloqValidationError::InvalidInstanceMergeStructure {
                    node: BloqNodeId(3),
                    source: NodeTemplateInstanceMergeError::BodyCountOverflow,
                },
                true,
            ),
            (
                BloqValidationError::InvalidTemplateCircuit {
                    template: TemplateId(4),
                    source: NodeTemplateInstanceMergeError::NoiseRepeatExpansion(
                        bloq_ir::FlattenError::Circuit(
                            bloq_circuit::CircuitError::FlattenResourceLimit {
                                observed: 2,
                                limit: 1,
                            },
                        ),
                    ),
                },
                true,
            ),
            (BloqValidationError::CyclicGraph, false),
        ] {
            let converted = CompileError::from_verification_audit(error.clone());
            assert!(matches!(&converted, CompileError::BloqValidation(source) if source == &error));
            assert_eq!(converted.is_resource_limited(), resource_limited);
            assert!(converted.resource_limit_help().is_none());
        }
    }

    #[test]
    fn clifford_proxy_preserves_source_generator_selection_and_order() {
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("gallery graph expands");
        let mut source = graph.stabilizers().unwrap();
        source.generators.reverse();
        source.generators.pop().expect("CNOT has generators");
        let ctx = CompileContext {
            clifford_proxy: Some(CliffordProxyPins {
                pins: Vec::new(),
                purpose: CliffordProxyPurpose::Distance,
                source_summary: None,
            }),
            ..CompileContext::new(CompileConfig::default())
        };
        let resolved = ctx.proxy_stabilizers(&graph, &source).unwrap();
        assert_eq!(resolved.generators, source.generators);
    }

    fn single_cube_runtime_graph() -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(bloq_graph::Block::new(
            ivec3(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "skip-derive".to_string(),
            })
            .expect("add skip-derive measurement action");
        graph
    }

    fn structural_branch_graph() -> BlockGraph {
        let prefix = ivec3(0, 0, 0);
        let target = ivec3(0, 0, 1);
        let reader = ivec3(2, 0, 0);
        let mut graph = BlockGraph::new();
        for pos in [prefix, reader] {
            graph.add_block(Block::new(pos, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let false_arm = BranchArm::new(
            vec![Block::new(target, BlockKind::Measurement(Basis::Z))],
            vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
        );
        let true_arm = BranchArm::new(
            vec![Block::new(target, BlockKind::Cube(CubeKind::XZX))],
            vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
        );
        let target = graph
            .try_add_branch_region("b0", false_arm, true_arm)
            .unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(reader),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: bloq_graph::Expr::Var("m".into()),
                },
            ])
            .unwrap();
        graph
    }

    fn mixed_structural_dynamic_graph() -> BlockGraph {
        let mut graph = GalleryItem::T
            .build()
            .flatten()
            .expect("gallery graph expands");
        let prefix = ivec3(4, 0, 0);
        let target = prefix + IVec3::Z;
        graph.add_block(Block::new(prefix, BlockKind::Cube(CubeKind::ZXZ)));
        let false_arm = BranchArm::new(
            vec![Block::new(target, BlockKind::Measurement(Basis::Z))],
            vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
        );
        let true_arm = BranchArm::new(
            vec![Block::new(target, BlockKind::Cube(CubeKind::XZX))],
            vec![Pipe::new(prefix, Direction::ZPLUS).with_hadamard()],
        );
        let target = graph
            .try_add_branch_region("b0", false_arm, true_arm)
            .unwrap();
        let mut actions = graph.actions();
        actions.push(Action::Branch {
            target,
            condition: bloq_graph::Expr::Var("mzz".into()),
        });
        graph.set_actions(actions).unwrap();
        graph
    }

    fn two_structural_branch_graph() -> BlockGraph {
        let mut graph = GalleryItem::GHZ
            .build()
            .flatten()
            .expect("gallery graph expands");
        let mut actions = Vec::new();
        for (index, (past, kind)) in [
            (ivec3(0, -1, 1), CubeKind::XZZ),
            (ivec3(-1, 0, 1), CubeKind::ZXZ),
        ]
        .into_iter()
        .enumerate()
        {
            let target = past + IVec3::Z;
            let reader = ivec3(20 + index as i32, 0, 0);
            graph.set_block_kind(past, BlockKind::Cube(kind)).unwrap();
            graph.add_block(Block::new(reader - IVec3::Z, BlockKind::Port));
            graph.add_block(Block::new(reader, BlockKind::Cube(CubeKind::XZX)));
            graph.add_pipe(Pipe::new(reader - IVec3::Z, Direction::ZPLUS));
            let arm = |kind| {
                BranchArm::new(
                    vec![Block::new(target, kind)],
                    vec![Pipe::new(past, Direction::ZPLUS)],
                )
            };
            let target = graph
                .try_add_branch_region(
                    format!("b{index}"),
                    arm(BlockKind::Measurement(Basis::Z)),
                    arm(BlockKind::Cube(kind)),
                )
                .unwrap();
            let measurement = format!("m{index}");
            actions.push(Action::Measure {
                target: MeasureTarget::Node(reader),
                name: measurement.clone(),
            });
            actions.push(Action::Branch {
                target,
                condition: bloq_graph::Expr::Var(measurement),
            });
        }
        graph.set_actions(actions).unwrap();
        graph
    }

    #[test]
    fn audited_structural_branch_retains_compiled_artifacts() {
        let graph = structural_branch_graph();
        let context = CompileContext::new(CompileConfig::new(3));

        let artifacts = context.compile_and_validate(&graph).unwrap();
        assert_eq!(compiled_distance(&artifacts.bloq), Some(3));
        assert!(artifacts.bloq.quantum_node_count() > 0);
        assert!(artifacts.bloq.has_conditional_membership());
    }

    #[test]
    fn repeated_branch_variables_obey_classical_slot_exactness() {
        for bit in [false, true] {
            for operation in [
                bloq_graph::BinaryOp::Xor,
                bloq_graph::BinaryOp::And,
                bloq_graph::BinaryOp::Or,
            ] {
                let mut graph = structural_branch_graph();
                let mut actions = graph.actions();
                let Action::Branch { condition, .. } = actions.last_mut().unwrap() else {
                    unreachable!()
                };
                let leaf = if bit {
                    bloq_graph::Expr::Not(Box::new(condition.clone()))
                } else {
                    condition.clone()
                };
                *condition =
                    bloq_graph::Expr::Binary(operation, Box::new(leaf.clone()), Box::new(leaf));
                graph.set_actions(actions).unwrap();
                let program = compile(&graph, 3).unwrap();
                program.validate().unwrap();
                let report = bloq_vm::run_bloq(&program, 2, 0).unwrap();
                let expected = operation != bloq_graph::BinaryOp::Xor && bit;
                assert_eq!(report.branch_selectors, [expected; 2]);
                assert_eq!(report.discarded, 0);
                assert!(report.all_detectors_constant());
            }
        }
    }

    #[test]
    fn structural_branch_composes_with_t_and_selective_components() {
        use bloq_ir::{BloqNodeKind, RegionNode};

        let graph = mixed_structural_dynamic_graph();
        let context = CompileContext::new(CompileConfig::new(3));

        let program = context.compile_and_validate(&graph).unwrap().bloq;
        let regions = program
            .nodes()
            .filter(|(_, node)| {
                matches!(
                    node.kind,
                    BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { .. })
                )
            })
            .count();
        assert_eq!(regions, 1);
        assert!(program.has_conditional_membership());
        assert!(program.nodes().any(|(_, node)| matches!(
            node.provenance,
            bloq_ir::NodeProvenance::BranchSelector { .. }
        )));

        let report = bloq_vm::run_bloq(&program, 8, 7).unwrap();
        assert_eq!(report.discarded, 0);
        assert!(report.all_detectors_constant());
        assert!(report.max_rank > 1);
        assert_eq!(report.frame_pairs.len(), 1);
        assert!(report.branch_selectors.contains(&false));
        assert!(report.branch_selectors.contains(&true));
    }

    #[test]
    fn compile_and_validate_checks_false_arm_physical_signatures() {
        let mut graph = BlockGraph::new();
        let prefix = ivec3(0, 0, 0);
        let a = ivec3(0, 0, 1);
        let b = ivec3(1, 0, 1);
        let c = ivec3(1, 1, 1);
        let reader = ivec3(4, 0, 0);
        graph.add_block(Block::new(prefix, BlockKind::Cube(CubeKind::XZX)));
        graph.add_block(Block::new(reader, BlockKind::Cube(CubeKind::ZXZ)));
        let target = graph
            .try_add_branch_region(
                "mixed_wall",
                BranchArm::new(
                    vec![
                        Block::new(a, BlockKind::Cube(CubeKind::XZX)),
                        Block::new(b, BlockKind::Cube(CubeKind::XXZ)),
                        Block::new(c, BlockKind::Cube(CubeKind::ZXX)),
                    ],
                    vec![
                        Pipe::new(prefix, Direction::ZPLUS),
                        Pipe::new(a, Direction::XPLUS).with_hadamard(),
                        Pipe::new(b, Direction::YPLUS).with_hadamard(),
                    ],
                ),
                BranchArm::new(
                    vec![Block::new(a, BlockKind::Cube(CubeKind::XZX))],
                    vec![Pipe::new(prefix, Direction::ZPLUS)],
                ),
            )
            .unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(reader),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: bloq_graph::Expr::Var("m".into()),
                },
            ])
            .unwrap();
        let context = CompileContext::new(CompileConfig::new(3));
        assert!(matches!(
            context.compile_and_validate(&graph),
            Err(CompileError::MixedSpatialHadamardUnsupported { .. })
        ));
        assert!(matches!(
            context.compile(&graph),
            Err(CompileError::MixedSpatialHadamardUnsupported { .. })
        ));
    }

    #[test]
    fn t_spill_reserves_both_structural_arms() {
        let mut graph = BlockGraph::new();
        let t = ivec3(0, 0, 1);
        let prefix = ivec3(1, -1, 0);
        let arm = ivec3(1, -1, 1);
        let spill = ivec3(0, -1, 1);
        let reader = ivec3(4, 0, 0);
        graph.add_block(Block::new(t, BlockKind::T));
        graph.add_block(Block::new([0, 0, 2], BlockKind::Port));
        graph.add_pipe(Pipe::new(t, Direction::ZPLUS));
        for position in [prefix, reader] {
            graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let target = graph
            .try_add_branch_region(
                "spill",
                BranchArm::new(
                    vec![
                        Block::new(arm, BlockKind::Cube(CubeKind::ZXZ)),
                        Block::new(spill, BlockKind::Cube(CubeKind::ZXZ)),
                    ],
                    vec![
                        Pipe::new(prefix, Direction::ZPLUS),
                        Pipe::new(arm, Direction::XMINUS),
                    ],
                ),
                BranchArm::new(
                    vec![Block::new(arm, BlockKind::Cube(CubeKind::ZXZ))],
                    vec![Pipe::new(prefix, Direction::ZPLUS)],
                ),
            )
            .unwrap();
        for selected in [false, true] {
            let condition = bloq_graph::Expr::Var("m".into());
            graph
                .set_actions(vec![
                    Action::Measure {
                        target: MeasureTarget::Node(reader),
                        name: "m".into(),
                    },
                    Action::Branch {
                        target,
                        condition: if selected {
                            bloq_graph::Expr::Not(Box::new(condition))
                        } else {
                            condition
                        },
                    },
                ])
                .unwrap();
            let context = CompileContext::new(CompileConfig::new(3));
            for module in [false, true] {
                let program = if module {
                    context.compile(&graph.clone().with_inferred_interface().unwrap())
                } else {
                    context.compile_and_validate(&graph)
                }
                .unwrap()
                .bloq;
                crate::validate_bloq_qubit_layout_for_source(&program, &graph).unwrap();
                let report = bloq_vm::run_bloq(&program, 2, 7).unwrap();
                assert_eq!(report.discarded, 0);
                assert!(
                    report
                        .detectors
                        .iter()
                        .all(|detector| detector.per_shot.iter().all(|&bit| !bit))
                );
                assert_eq!(report.branch_selectors, [selected; 2]);
            }
        }
        for position in [ivec3(1, 0, 1), ivec3(0, 1, 1), ivec3(-1, 0, 1)] {
            graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let context = CompileContext::new(CompileConfig::new(3));
        assert!(matches!(
            context.compile_and_validate(&graph),
            Err(CompileError::TBlockNeedsFreeNeighbor { .. })
        ));
        assert!(matches!(
            context.compile(&graph.with_inferred_interface().unwrap()),
            Err(CompileError::TBlockNeedsFreeNeighbor { .. })
        ));
    }

    #[test]
    fn independent_structural_deltas_execute_all_joint_choices() {
        let module = two_structural_branch_graph()
            .with_inferred_interface()
            .unwrap();
        let context = CompileContext::new(CompileConfig::new(3));
        let program = context.compile(&module).unwrap().bloq;
        program.validate().unwrap();

        let report = bloq_vm::run_bloq(&program, 64, 0xB12A).unwrap();
        assert_eq!(report.discarded, 0);
        assert!(report.all_detectors_constant());
        let choices = report
            .branch_selectors
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bits| (bits[0], bits[1]))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(choices.len(), 4);
        assert_eq!(report.frame_pairs.len(), 2);
    }

    #[test]
    fn clifford_proxy_entrypoints_reject_structural_branch() {
        let graph = structural_branch_graph();
        let rejected = |result: Result<CompileArtifacts, CompileError>| {
            assert!(matches!(
                result,
                Err(CompileError::StructuralBranchCliffordProxyUnsupported)
            ));
        };

        rejected(compile_clifford_proxy(
            CompileConfig::default(),
            &graph,
            &[],
        ));
        rejected(compile_random_clifford_proxy(
            CompileConfig::default(),
            &graph,
            7,
        ));
        rejected(compile_detslice_proxy(
            CompileConfig::default(),
            &graph,
            &[],
        ));
    }

    fn source_timeline(program: &Bloq, pos: IVec3) -> Option<&bloq_ir::QuantumTimeline> {
        program.nodes().find_map(|(_, node)| {
            node.block_members()
                .iter()
                .any(|member| member.pos == pos)
                .then(|| node.expect_quantum().timeline.as_ref())
                .flatten()
        })
    }

    #[test]
    fn compilation_rejects_measurement_surface_on_output_port() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n\
             1: XZX [0,0,1]\n\
             2: Port [0,0,2]\n\
             3: T [1,0,0]\n\
             4: XZX [1,0,1]\n\
             5: YX [1,0,2]\n\
             [0,0,1] -> +Z\n\
             [0,0,1] -> +X\n\
             [1,0,0] -> +Z\n\
             [1,0,2] -> -Z\n\n\
             mzz = measure 1 -> +X\n\
             resolve 5 if mzz\n",
        )
        .unwrap();
        let stabilizers = graph.stabilizers().unwrap();
        let measurement = stabilizers
            .generators
            .iter()
            .find(|row| row.measurement_name() == Some("mzz"))
            .expect("the merge supplies the named measurement");
        assert_eq!(
            measurement.stabilizer.port_stabilizer.get(&ivec3(0, 0, 2)),
            Some(&bloq_graph::Pauli::Z),
        );
        assert!(matches!(
            stabilizers.validate_measurements_close_before_outputs(),
            Err(bloq_graph::RuntimeBasisError::Stabilizer(
                bloq_graph::StabilizerError::MeasurementSurfaceTouchesOutputPort {
                    ref name,
                    port,
                }
            )) if name == "mzz" && port == ivec3(0, 0, 2)
        ));
        assert!(matches!(
            compile_clifford_proxy(CompileConfig::default(), &graph, &[false]),
            Err(CompileError::RuntimeBasis(
                bloq_graph::RuntimeBasisError::Stabilizer(
                    bloq_graph::StabilizerError::MeasurementSurfaceTouchesOutputPort {
                        ref name,
                        port,
                    }
                )
            )) if name == "mzz" && port == ivec3(0, 0, 2)
        ));
        let context = CompileContext::new(CompileConfig::default());

        for error in [
            context.compile_and_validate(&graph).unwrap_err(),
            context.compile(&graph).unwrap_err(),
        ] {
            assert!(matches!(
                error,
                CompileError::BlockGraph(bloq_graph::BlockGraphError::Stabilizer(
                    bloq_graph::StabilizerError::UnavailableControlParity { ref mvar, .. }
                )) if mvar == "mzz"
            ));
        }
    }

    #[test]
    fn t_region_observable_ids_fail_before_wrapping() {
        let context = CompileContext {
            clifford_proxy: Some(CliffordProxyPins {
                pins: vec![false],
                purpose: CliffordProxyPurpose::Detslice,
                source_summary: None,
            }),
            ..CompileContext::new(CompileConfig::default())
        };
        let prepared = context
            .prepare_proxy(
                &GalleryItem::T
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
            )
            .unwrap();
        let plan = crate::lower::LowerPlan::from_linked_input(&prepared.plan);
        let placed = crate::lower::PlacedPlan::new(PhysicalInput {
            graph: &prepared.graph,
            plan: &plan,
            compiled: &prepared.compiled,
            spatial_port_templates: &prepared.spatial_port_templates,
            spatial_ports: &prepared.spatial_ports,
            temporal_templates: &prepared.temporal_templates,
            wall_templates: &prepared.wall_templates,
            template_pool: &prepared.template_pool,
            layout: BlockLayout::new(3),
        })
        .unwrap();
        assert!(matches!(
            placed.emit(u32::MAX - 1),
            Err(CompileError::BooleanResource(error)) if error.resource == "observable IDs"
        ));
    }

    #[test]
    fn compile_and_validate_normalizes_a_small_span_starting_at_i32_min() {
        let port = ivec3(0, 0, i32::MIN);
        let cube = ivec3(0, 0, i32::MIN + 1);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(port, BlockKind::Port));
        graph.add_block(Block::new(cube, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(bloq_graph::Pipe::new(port, bloq_graph::Direction::ZPLUS));
        let ctx = CompileContext::new(CompileConfig::new(3));

        ctx.compile_and_validate(&graph)
            .expect("the normalized z span is 0..=1");
    }

    #[test]
    fn compile_propagates_block_layout_overflow() {
        assert!(matches!(
            block_xy_offset(IVec2::MAX, u32::MAX),
            Err(CompileError::BlockLayoutCoordinateOverflow { .. })
        ));

        let graph = single_cube_runtime_graph()
            .shift_positions(ivec3(300_000_000, 0, 0))
            .expect("source coordinate fits");
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&stages);
        let ctx = CompileContext::new(CompileConfig::new(3))
            .with_progress_observer(move |stage| observed.lock().unwrap().push(stage));

        assert!(matches!(
            ctx.compile(&graph),
            Err(CompileError::BlockLayoutCoordinateOverflow {
                block_xy,
                distance: 3,
            }) if block_xy == glam::ivec2(300_000_000, 0)
        ));
        assert!(!stages.lock().unwrap().contains(&CompileStage::Complete));
    }

    #[test]
    fn boundary_observable_compile_returns_layout_overflow_without_panicking() {
        let graph = single_cube_runtime_graph()
            .shift_positions(ivec3(i32::MAX, 0, 0))
            .expect("boundary source coordinate fits");
        let ctx = CompileContext::new(CompileConfig::new(3));

        assert!(matches!(
            ctx.compile(&graph),
            Err(CompileError::BlockLayoutCoordinateOverflow {
                block_xy,
                distance: 3,
            }) if block_xy == glam::ivec2(i32::MAX, 0)
        ));
    }

    #[test]
    fn compile_context_reuses_templates_across_compile_calls() {
        let first_graph = single_cube_runtime_graph();
        let second_graph = single_cube_runtime_graph();
        let ctx = CompileContext::new(CompileConfig::new(3));

        let first = ctx.compile(&first_graph).expect("compile first graph");
        let second = ctx.compile(&second_graph).expect("compile second graph");

        assert_eq!(
            ctx.cache.pool_len(),
            1,
            "matching block signatures should reuse a cached template across compiles"
        );
        assert!(
            std::ptr::eq(
                first.bloq.templates().iter().next().unwrap().1,
                second.bloq.templates().iter().next().unwrap().1,
            ),
            "emitted programs retain the immutable cached template"
        );
    }

    #[test]
    fn cancellation_stops_an_active_stage_and_the_next_request_can_use_the_cache() {
        let graph = GalleryItem::CNOT.build();
        let cache = crate::SharedCompileCache::new();
        for _ in 0..2 {
            let token = bloq_graph::CancellationToken::new();
            let cancel = token.clone();
            let context = CompileContext::with_shared_cache(CompileConfig::default(), &cache)
                .with_cancellation(token)
                .with_progress_observer(move |stage| {
                    if stage == CompileStage::Readouts {
                        cancel.cancel();
                    }
                });
            let error = context.compile(&graph).unwrap_err();
            assert!(error.is_cancelled());
            assert!(!error.is_resource_limited());
        }
        let fresh = CompileContext::with_shared_cache(CompileConfig::default(), &cache)
            .with_cancellation(bloq_graph::CancellationToken::new());
        fresh.compile(&graph).unwrap();
    }

    #[test]
    fn cancelled_module_request_does_not_publish_a_partial_cached_object() {
        let program = GalleryItem::CNOT.build().with_inferred_interface().unwrap();
        let cache = crate::SharedCompileCache::new();
        let token = bloq_graph::CancellationToken::new();
        let cancel = token.clone();
        let context = CompileContext::with_shared_cache(CompileConfig::default(), &cache)
            .with_cancellation(token)
            .with_progress_observer(move |stage| {
                if stage == CompileStage::Readouts {
                    cancel.cancel();
                }
            });
        assert!(context.compile_object(&program).unwrap_err().is_cancelled());
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&stages);
        let fresh = CompileContext::with_shared_cache(CompileConfig::default(), &cache)
            .with_progress_observer(move |stage| observed.lock().unwrap().push(stage));
        fresh.compile_object(&program).unwrap();
        assert!(
            stages
                .lock()
                .unwrap()
                .contains(&CompileStage::Certification)
        );
        assert!(!stages.lock().unwrap().contains(&CompileStage::CacheReuse));
    }

    #[test]
    fn progress_observer_reports_native_stages_and_completion() {
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&stages);
        let context = CompileContext::new(CompileConfig::default())
            .with_progress_observer(move |stage| observed.lock().unwrap().push(stage));
        context
            .compile(
                &GalleryItem::CNOT
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
            )
            .unwrap();
        assert_eq!(
            *stages.lock().unwrap(),
            [
                CompileStage::Validation,
                CompileStage::Correlations,
                CompileStage::Placement,
                CompileStage::Templates,
                CompileStage::Readouts,
                CompileStage::Optimization,
                CompileStage::Complete,
            ]
        );
    }

    #[test]
    fn progress_observer_reports_module_cache_reuse() {
        let module = single_cube_runtime_graph()
            .with_inferred_interface()
            .unwrap();
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&stages);
        let context = CompileContext::new(CompileConfig::default())
            .with_progress_observer(move |stage| observed.lock().unwrap().push(stage));
        context.compile(&module).unwrap();
        assert_eq!(
            stages
                .lock()
                .unwrap()
                .iter()
                .filter(|&&stage| stage == CompileStage::Complete)
                .count(),
            1
        );
        stages.lock().unwrap().clear();
        context.compile(&module).unwrap();
        assert_eq!(
            *stages.lock().unwrap(),
            [
                CompileStage::Validation,
                CompileStage::CacheReuse,
                CompileStage::Complete,
            ]
        );
    }

    #[test]
    fn fixed_measurement_lowers_without_a_region_and_emits_stim() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Measurement(Basis::X)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        let program = CompileContext::new(CompileConfig::new(3))
            .compile(&graph)
            .expect("fixed measurement graph compiles")
            .bloq;

        program.validate().expect("compiled Bloq validates");
        assert!(program.nodes().all(|(_, node)| node.try_region().is_none()));
        let stim = bloq_stim::emit_bloq_stim(&program).expect("fixed measurement emits Stim");
        assert!(
            stim.contains("MX "),
            "expected transversal X readout:\n{stim}"
        );
    }

    /// Two ZXZ cubes stacked along z, joined by one temporal pipe (a two-round
    /// memory). `pipe` fixes the temporal pipe's authored orientation.
    fn stacked_memory_graph(pipe: bloq_graph::Pipe) -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(0, 0, 1), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(pipe);
        graph
    }

    /// CP-04: a temporal pipe authored `+Z` from the lower cube and one authored
    /// `−Z` from the upper cube describe the same seam. Canonical pipe
    /// orientation must erase that authoring difference, so both compile to a
    /// byte-identical artifact.
    #[test]
    fn reversed_temporal_pipe_orientation_compiles_identically() {
        let forward = stacked_memory_graph(bloq_graph::Pipe::new(
            ivec3(0, 0, 0),
            bloq_graph::Direction::ZPLUS,
        ));
        let reversed = stacked_memory_graph(bloq_graph::Pipe::new(
            ivec3(0, 0, 1),
            bloq_graph::Direction::ZMINUS,
        ));

        let compile = |graph: &BlockGraph| {
            CompileContext::new(CompileConfig::new(3))
                .compile(graph)
                .expect("orientation compiles")
                .bloq
                .to_binary()
        };
        let forward_bytes = compile(&forward);
        let reversed_bytes = compile(&reversed);

        assert_eq!(
            forward_bytes, reversed_bytes,
            "reversed temporal-pipe authoring must not change the compiled artifact"
        );
    }

    #[test]
    fn basisless_temporal_hadamard_compiles_in_both_orientations() {
        let compile = |pipe| {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
            graph.add_block(Block::new(IVec3::Z, BlockKind::Port));
            graph.add_pipe(pipe);

            let bloq = super::compile(&graph, 3).expect("temporal Hadamard compiles");
            bloq.validate().expect("compiled Bloq validates");
            bloq.to_binary()
        };
        let forward = compile(Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_hadamard());
        let reversed = compile(Pipe::new(IVec3::Z, Direction::ZMINUS).with_hadamard());

        assert_eq!(forward, reversed);
    }

    #[test]
    fn clifford_proxy_derives_exact_output_frames() {
        for (gallery, pins) in [(GalleryItem::CNOT, &[][..]), (GalleryItem::T, &[false][..])] {
            let bloq = compile_clifford_proxy(
                CompileConfig::default(),
                &gallery.build().flatten().expect("gallery graph expands"),
                pins,
            )
            .unwrap_or_else(|error| panic!("{gallery:?} proxy compiles: {error}"))
            .bloq;
            assert_eq!(
                bloq.output_frames().len(),
                bloq.logical_outputs().len(),
                "{gallery:?} proxy emits one frame pair per logical output",
            );
            bloq.validate()
                .expect("proxy ports and recipe owners validate");
            let flips = bloq
                .edges()
                .filter(|edge| {
                    matches!(
                        edge.edge,
                        bloq_ir::BloqEdge::Value {
                            output: bloq_ir::ObservableOutput::Flip,
                            ..
                        }
                    )
                })
                .collect::<Vec<_>>();
            assert!(
                !flips.is_empty(),
                "proxy frames consume public decoder flips"
            );
            for edge in flips {
                assert!(matches!(
                    bloq[edge.source].try_classical(),
                    Some(bloq_ir::ClassicalNode::Observable { index: Some(_), .. })
                ));
                assert!(matches!(
                    bloq[edge.target].try_classical(),
                    Some(bloq_ir::ClassicalNode::Observable { index: None, .. })
                ));
            }
        }
    }

    #[test]
    fn random_clifford_proxy_is_seeded_and_records_its_seed() {
        let hierarchical = GalleryItem::T.build();
        for graph in [hierarchical.flatten().unwrap(), hierarchical] {
            let compile = || {
                compile_random_clifford_proxy(CompileConfig::default(), &graph, 0xC10F_F0AD)
                    .expect("random proxy compiles")
                    .bloq
            };
            let first = compile();
            let second = compile();
            assert_eq!(first.to_binary(), second.to_binary());
            assert_eq!(
                first.metadata().get(CLIFFORD_PROXY_SEED_METADATA_KEY),
                Some(&bloq_ir::MetadataValue::U64(0xC10F_F0AD))
            );
            bloq_stim::emit_bloq_stim(&first).expect("random proxy is static Stim");
        }
    }

    #[test]
    fn distance_proxy_replaces_t_even_when_mpp_preparation_is_enabled() {
        let bloq = compile_clifford_proxy(
            CompileConfig::default().with_prepare_t_with_mpps(true),
            &GalleryItem::T
                .build()
                .flatten()
                .expect("gallery graph expands"),
            &[false],
        )
        .expect("MPP-configured T proxy compiles")
        .bloq;

        assert!(bloq.nodes().all(|(_, node)| node.try_region().is_none()));
        bloq_stim::emit_bloq_stim(&bloq).expect("MPP-configured T proxy is static Stim");
    }

    #[test]
    fn clifford_proxy_accepts_partial_padding() {
        let mut bloq = compile_clifford_proxy(
            CompileConfig::default(),
            &GalleryItem::T
                .build()
                .flatten()
                .expect("gallery graph expands"),
            &[false],
        )
        .expect("T proxy compiles")
        .bloq;
        let target = bloq
            .nodes()
            .find_map(|(id, node)| {
                node.block_members()
                    .iter()
                    .any(|member| member.pos == ivec3(1, 0, 2))
                    .then_some(id)
            })
            .expect("pinned selective node exists");
        let source = bloq
            .incoming(target)
            .find_map(|edge| {
                matches!(edge.edge, bloq_ir::BloqEdge::Quantum(_)).then_some(edge.source)
            })
            .expect("pinned selective seam exists");

        let padding = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: source,
                    to: target,
                },
                10,
            )
            .expect("pinned selective seam accepts decoder wait");

        assert_eq!(bloq[padding].memory_rounds(), Some(10));
        bloq.validate().expect("padded proxy validates");
        bloq_stim::emit_bloq_stim(&bloq).expect("padded proxy emits");
    }

    #[test]
    fn clifford_proxy_rejects_wrong_selective_pin_count() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Selective(bloq_graph::SelectiveKind::XY),
        ));

        assert!(matches!(
            compile_clifford_proxy(CompileConfig::default(), &graph, &[]),
            Err(CompileError::CliffordProxyPinCount {
                expected: 1,
                actual: 0
            })
        ));
        assert!(matches!(
            compile_clifford_proxy(CompileConfig::default(), &graph, &[false, true]),
            Err(CompileError::CliffordProxyPinCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    /// A graph with T blocks at `positions`, each under its own ZXZ cube.
    fn t_blocks_graph(positions: &[IVec3]) -> BlockGraph {
        let mut graph = BlockGraph::new();
        for &pos in positions {
            graph.add_block(Block::new(pos, BlockKind::T));
            graph.add_block(Block::new(pos + IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            graph.add_pipe(bloq_graph::Pipe::new(pos, bloq_graph::Direction::ZPLUS));
        }
        graph
    }

    /// Compile a graph's block templates only (no lowering), returning the pool
    /// size — the Step-3 surface while T lowering lands separately.
    fn compiled_template_pool_len(graph: &BlockGraph) -> usize {
        let ctx = CompileContext::new(CompileConfig::new(3));
        let signatures = block_signatures(
            graph,
            &join_component_layer_schedules(graph).expect("no mixed spatial Hadamard walls"),
            ctx.config.code_distance(),
        )
        .expect("valid graph");
        let blocks = ordered_blocks(graph);
        ctx.compile_blocks(graph, &blocks, &signatures)
            .expect("compile blocks");
        ctx.cache.pool_len()
    }

    #[test]
    fn t_blocks_with_matching_signatures_share_cached_templates() {
        // Two far-apart T blocks: same connectivity, boundary basis, and spill
        // side => one cultivation + escape pair, plus the shared cube template.
        let graph = t_blocks_graph(&[ivec3(0, 0, 0), ivec3(5, 0, 0)]);
        assert_eq!(compiled_template_pool_len(&graph), 3);
    }

    #[test]
    fn t_blocks_with_differing_surgery_sides_compile_distinct_templates() {
        // Occupy the second T's -y neighbor so its spill side falls through to
        // +x: a different signature, so a second cultivation + escape pair.
        // Pool: piped cube + isolated boxing cube + 2 × (cultivation, escape).
        let mut graph = t_blocks_graph(&[ivec3(0, 0, 0), ivec3(5, 0, 0)]);
        graph.add_block(Block::new(ivec3(5, -1, 0), BlockKind::Cube(CubeKind::ZXZ)));
        assert_eq!(compiled_template_pool_len(&graph), 6);
    }

    #[test]
    fn t_region_body_carries_seam_restarts_and_gap_wiring() {
        use bloq_ir::{BloqNodeKind, ClassicalExpr, ClassicalNode, ObservableOutput, RegionNode};

        let graph = t_blocks_graph(&[ivec3(0, 0, 0)]);
        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx.compile(&graph).expect("T graph compiles").bloq;
        program.validate().expect("retry wiring is valid IR");

        let (body, condition, source) = program
            .nodes()
            .find_map(|(_, node)| match &node.kind {
                BloqNodeKind::Region(RegionNode::RepeatUntilSuccess {
                    body,
                    restart_condition,
                    restart_source,
                }) => Some((
                    body,
                    restart_condition,
                    restart_source.expect("explicit body retry predicate"),
                )),
                _ => None,
            })
            .expect("the T block lowers to one RUS region");

        // Two quantum stage nodes: cultivation (restart-free side tables — its
        // template-local restarts ride the template) then escape, whose node
        // carries the intra-body seam restarts. At least one seam restart spans
        // both stage instances (the cultivation-closure chains).
        let quantum: Vec<_> = body.quantum_nodes().map(|(_, node)| node).collect();
        assert_eq!(quantum.len(), 2, "cultivation + escape stage nodes");
        let (cultivation, escape) = (quantum[0], quantum[1]);
        let cultivation_id = cultivation.instances[0].id;
        let escape_id = escape.instances[0].id;
        assert!(cultivation.restarts.is_empty());
        assert!(!escape.restarts.is_empty(), "intra-body seam restarts");
        assert!(
            escape.restarts.iter().any(|restart| {
                let instances: crate::FxSet<_> = restart
                    .parity
                    .measurements()
                    .map(|measurement| measurement.instance)
                    .collect();
                instances.contains(&cultivation_id) && instances.contains(&escape_id)
            }),
            "a cross-template restart chain spans both stage instances"
        );

        let mut indices = body
            .nodes()
            .filter_map(|(_, node)| match node.try_classical() {
                Some(ClassicalNode::Observable {
                    index: Some(index),
                    measurements,
                    operators,
                }) => {
                    assert!(
                        measurements
                            .iter()
                            .all(|record| record.instance == escape_id)
                    );
                    assert!(
                        operators
                            .iter()
                            .all(|operator| operator.instance == escape_id)
                    );
                    assert!(!measurements.is_empty());
                    Some(*index)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices.len(), 2, "two GAP decoding problems");
        assert_ne!(indices[0], indices[1]);
        let computes = body
            .nodes()
            .filter_map(|(id, node)| match node.try_classical() {
                Some(ClassicalNode::Compute { expr }) => Some((id, expr)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(computes.len(), 1, "one explicit retry predicate");
        let (retry, expr) = computes[0];
        assert_eq!(source.node, retry);
        assert_eq!(source.output, ObservableOutput::Corrected);
        assert_ne!(
            body.value_output(),
            Some(source),
            "retry is not the body result"
        );
        assert_eq!(*condition, ClassicalExpr::In(0));
        let mut inputs = body.value_inputs(retry).collect::<Vec<_>>();
        inputs.sort_unstable_by_key(|input| input.slot);
        assert_eq!(
            inputs.iter().map(|input| input.slot).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_ne!(inputs[0].producer, inputs[1].producer);
        for input in inputs {
            assert_eq!(input.output, Some(ObservableOutput::Flip));
            assert!(matches!(
                body[input.producer].try_classical(),
                Some(ClassicalNode::Observable { index: Some(_), .. })
            ));
        }
        for (flips, expected) in [
            ([false, false], false),
            ([false, true], true),
            ([true, false], true),
            ([true, true], true),
        ] {
            let retry = expr
                .eval(&mut |slot| flips.get(slot as usize).copied())
                .unwrap();
            assert_eq!(retry, expected, "decoder flips {flips:?}");
            assert_eq!(
                condition.eval(&mut |slot| (slot == 0).then_some(retry)),
                Some(expected)
            );
        }

        // The downstream +Z neighbour closes the escape end face: its top-level
        // detectors reference the in-body escape instance, and nothing at top
        // level restarts (that would trip RestartOutsideRepeatUntilSuccess).
        let top_quantum: Vec<_> = program.quantum_nodes().map(|(_, node)| node).collect();
        assert!(top_quantum.iter().all(|node| node.restarts.is_empty()));
        assert!(
            top_quantum.iter().any(|node| {
                program.node_detectors(node).unwrap().any(|detector| {
                    detector
                        .measurements()
                        .any(|measurement| measurement.instance == escape_id)
                })
            }),
            "the neighbour's seam detectors reference the in-body escape instance"
        );
    }

    #[test]
    fn boxed_t_block_needs_free_neighbor_end_to_end() {
        let mut graph = t_blocks_graph(&[ivec3(0, 0, 0)]);
        for neighbor in [
            ivec3(0, -1, 0),
            ivec3(1, 0, 0),
            ivec3(0, 1, 0),
            ivec3(-1, 0, 0),
        ] {
            graph.add_block(Block::new(neighbor, BlockKind::Cube(CubeKind::ZXZ)));
        }
        let ctx = CompileContext::new(CompileConfig::new(3));
        assert!(matches!(
            ctx.compile_and_validate(&graph).unwrap_err(),
            CompileError::TBlockNeedsFreeNeighbor { .. }
        ));
        assert!(matches!(
            ctx.compile(&graph).unwrap_err(),
            CompileError::TBlockNeedsFreeNeighbor { .. }
        ));
    }

    #[test]
    fn compile_t_block_lowers_to_repeat_until_success_region() {
        use bloq_ir::{BloqNodeKind, RegionNode};

        let graph = t_blocks_graph(&[ivec3(0, 0, 0)]);
        let ctx = CompileContext::new(CompileConfig::new(3));

        let program = ctx.compile(&graph).expect("T graph compiles").bloq;

        let rus_count = program
            .nodes()
            .filter(|(_, node)| {
                matches!(
                    node.kind,
                    BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { .. })
                )
            })
            .count();
        assert_eq!(rus_count, 1, "the T block is one RepeatUntilSuccess region");

        // Audit the complete lowered IR explicitly, including the RUS body.
        program.validate().expect("T program validates recursively");

        // The static Stim backend rejects the region by name; the native
        // dynamic runtime executes it.
        let error = bloq_stim::emit_bloq_stim(&program).expect_err("RUS is backend-rejected");
        assert!(
            matches!(error, bloq_stim::StimEmissionError::UnsupportedNode(reason) if reason.contains("RepeatUntilSuccess")),
            "expected RepeatUntilSuccess rejection, got: {error:?}",
        );
    }

    #[test]
    fn registry_closed_readouts_keep_their_code_distance() {
        for gallery in [
            GalleryItem::XMemory,
            GalleryItem::YMemory,
            GalleryItem::Stability,
        ] {
            let context = CompileContext::new(CompileConfig::default());
            let bloq = registry::compile_graph(
                &context,
                &gallery.build().flatten().expect("gallery graph expands"),
            )
            .unwrap()
            .bloq;
            let noise = stim::noise::UniformDepolarizing::new(1e-3).unwrap();
            let circuit: stim::Circuit = bloq_stim::emit_bloq_stim_with_stim_noise(&bloq, &noise)
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(circuit.num_observables(), 1, "{gallery:?}");
            assert_eq!(
                circuit.shortest_graphlike_error().unwrap().len(),
                3,
                "{gallery:?}"
            );
        }
    }

    #[test]
    fn registry_preparations_do_not_decode_unavailable_output_axes() {
        for gallery in [GalleryItem::BellState, GalleryItem::GHZ] {
            let graph = gallery.build().flatten().expect("gallery graph expands");
            let expected = bloq_graph::ZXGraph::from_block_graph_for_analysis(&graph)
                .unwrap()
                .stabilizers()
                .unwrap()
                .generators
                .len();
            let context = CompileContext::new(CompileConfig::default());
            for bloq in [
                registry::compile_graph(&context, &graph).unwrap().bloq,
                registry::compile_hierarchy(&context, &gallery.build(), NonZeroUsize::MIN)
                    .unwrap()
                    .bloq,
            ] {
                let noise = stim::noise::UniformDepolarizing::new(1e-3).unwrap();
                let circuit: stim::Circuit =
                    bloq_stim::emit_bloq_stim_with_stim_noise(&bloq, &noise)
                        .unwrap()
                        .parse()
                        .unwrap();
                assert_eq!(circuit.num_observables(), expected as u64, "{gallery:?}");
                assert_eq!(circuit.shortest_graphlike_error().unwrap().len(), 3);
            }
        }
    }

    #[test]
    fn empty_native_programs_compile_and_validate() {
        let context = CompileContext::new(CompileConfig::default());
        let graph = BlockGraph::new();
        for bloq in [
            context.compile_and_validate(&graph).unwrap().bloq,
            context
                .compile(&graph.with_inferred_interface().unwrap())
                .unwrap()
                .bloq,
        ] {
            bloq.validate().unwrap();
            assert_eq!(bloq.node_count(), 0);
        }
    }

    #[test]
    fn t_gallery_programs_compile_validate_and_are_backend_rejected() {
        // The U17 §7 acceptance matrix across the three d_int regimes (d=3
        // surgery-at-3, d=5 surgery-at-5, d=7 first 5→7 expansion).
        // The gallery graphs compile as-is: their In/Out `Port` blocks lower as
        // ideal temporal seams (`fill_ports_auto` is wrong here — it would fill
        // the selective/T port-like positions too and break the resolve actions).
        for gallery in [
            GalleryItem::T,
            GalleryItem::TWithPreparedY,
            GalleryItem::PhaseGradientK4,
            GalleryItem::THTH,
        ] {
            let graph = gallery.build().flatten().expect("gallery graph expands");
            // Backend-rejection is distance-invariant (RUS and dynamic membership are
            // rejected regardless of code distance); distance coverage lives in
            // the proxy-distance tests. One distance suffices here.
            {
                let code_distance = 3;
                let ctx = CompileContext::new(CompileConfig::new(code_distance));
                let program = ctx
                    .compile(&graph)
                    .unwrap_or_else(|error| {
                        panic!("{gallery:?} d={code_distance} compiles: {error}")
                    })
                    .bloq;
                program.validate().unwrap_or_else(|error| {
                    panic!("{gallery:?} d={code_distance} validates: {error}")
                });
                let error = bloq_stim::emit_bloq_stim(&program)
                    .expect_err("T programs are backend-rejected");
                assert!(
                    matches!(&error, bloq_stim::StimEmissionError::UnsupportedNode(_)),
                    "{gallery:?} d={code_distance}: expected region rejection, got {error:?}",
                );
            }
        }
    }

    #[test]
    fn phase_gradient_records_padding_before_selective_components() {
        use bloq_ir::BloqEdge;

        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx
            .compile(
                &GalleryItem::PhaseGradientK4
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
            )
            .expect("phase gradient compiles")
            .bloq;
        let mut seams = 0;
        for (branch, node) in program.nodes() {
            if !node
                .try_quantum()
                .is_some_and(|quantum| !quantum.guards.is_empty())
            {
                continue;
            }
            for edge in program.incoming(branch) {
                let BloqEdge::Quantum(quantum) = edge.edge else {
                    continue;
                };
                seams += 1;
                assert!(
                    quantum.pipes.iter().all(|seam| seam.padding.is_some()),
                    "selective input {edge:?} must persist a padding template per pipe",
                );
            }
        }
        assert!(seams > 0, "phase gradient contains adaptive quantum seams");
    }

    /// Exchange-format round trips on real compiled programs:
    /// both codecs must reproduce the program exactly — checked as binary
    /// byte equality (full structural identity, node-id holes included), a
    /// text fixpoint, a passing re-validation, and byte-identical Stim
    /// emission where the static backend supports the program.
    #[rstest]
    #[case(GalleryItem::CNOT)]
    #[case(GalleryItem::S)]
    #[case(GalleryItem::T)]
    fn exchange_formats_round_trip(#[case] gallery: GalleryItem) {
        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx
            .compile(&gallery.build().flatten().expect("gallery graph expands"))
            .expect("compiles")
            .bloq;

        let bytes = program.to_binary();
        let from_binary = bloq_ir::Bloq::from_binary(&bytes).expect("binary decodes");
        assert_eq!(
            bytes,
            from_binary.to_binary(),
            "{gallery:?}: binary fixpoint"
        );

        let text = program.to_text();
        let from_text = bloq_ir::Bloq::from_text(&text)
            .unwrap_or_else(|error| panic!("{gallery:?}: text parses: {error}"));
        assert_eq!(
            bytes,
            from_text.to_binary(),
            "{gallery:?}: text round trip is structurally lossless"
        );
        assert_eq!(text, from_text.to_text(), "{gallery:?}: text fixpoint");

        from_text
            .validate()
            .unwrap_or_else(|error| panic!("{gallery:?}: restored program validates: {error}"));

        let expected = bloq_stim::emit_bloq_stim(&program);
        let restored = bloq_stim::emit_bloq_stim(&from_text);
        if gallery == GalleryItem::T {
            for result in [expected, restored] {
                assert!(matches!(
                    result,
                    Err(bloq_stim::StimEmissionError::UnsupportedNode(_))
                ));
            }
        } else {
            assert_eq!(
                expected.expect("static original emits"),
                restored.expect("static restored program emits"),
                "{gallery:?}: identical Stim emission"
            );
        }
    }

    /// Feedback frames consume corrected observables and retain physical record
    /// dependencies. Action-free programs use the same frame construction.
    #[test]
    fn s_gallery_lowers_output_frames_from_symbolic_solve() {
        use bloq_ir::ClassicalNode;

        let ctx = CompileContext::new(CompileConfig::default());
        let program = ctx
            .compile(
                &GalleryItem::S
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
            )
            .unwrap()
            .bloq;
        let frames = program.output_frames();
        assert_eq!(frames.len(), 1, "S teleport has one output port");

        let classical = |node: bloq_ir::BloqNodeId| {
            program[node]
                .try_classical()
                .expect("frame nodes are classical")
                .clone()
        };
        assert!(matches!(
            classical(frames[0].x),
            ClassicalNode::Compute { .. }
        ));
        assert!(matches!(
            classical(frames[0].z),
            ClassicalNode::Compute { .. }
        ));

        // The Z frame reaches a complete observable and its records, through
        // any remaining shared fragments or fused computations.
        let mut stack = vec![frames[0].z];
        let mut seen = std::collections::HashSet::new();
        let (mut saw_records, mut saw_observable) = (false, false);
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if let Some(classical) = program[node].try_classical() {
                saw_records |= !classical.measurements().is_empty();
                saw_observable |= matches!(classical, ClassicalNode::Observable { .. });
            }
            stack.extend(program.top().data_inputs(node).map(|input| input.producer));
        }
        assert!(
            saw_observable,
            "Z frame reads a member's corrected observable"
        );
        assert!(
            saw_records,
            "the observable recipe contains physical records"
        );

        let ctx = CompileContext::new(CompileConfig::default());
        let action_free = ctx
            .compile(
                &GalleryItem::CNOT
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
            )
            .unwrap()
            .bloq;
        assert_eq!(
            action_free.output_frames().len(),
            2,
            "action-free CNOT gets one FramePair per output port"
        );
    }

    #[test]
    fn output_frame_ports_match_logical_outputs() {
        let compile = |graph: &BlockGraph| {
            CompileContext::new(CompileConfig::new(3))
                .compile(graph)
                .expect("gallery graph compiles")
                .bloq
        };

        let cnot = compile(
            &GalleryItem::CNOT
                .build()
                .flatten()
                .expect("gallery graph expands"),
        );
        let frame_outputs: Vec<_> = cnot
            .output_frames()
            .into_iter()
            .map(|frame| frame.port)
            .collect();
        let logical_outputs: Vec<_> = cnot
            .logical_outputs()
            .iter()
            .map(|output| output.port)
            .collect();
        assert_eq!(frame_outputs, logical_outputs);
        assert!(frame_outputs.contains(&glam::ivec3(1, 1, 3)));

        let shifted_t = GalleryItem::T
            .build()
            .flatten()
            .expect("gallery graph expands")
            .shift_positions(IVec3::X)
            .expect("shift fits");
        let t = compile(&shifted_t);
        assert_eq!(
            t.output_frames()
                .into_iter()
                .map(|frame| frame.port)
                .collect::<Vec<_>>(),
            t.logical_outputs()
                .iter()
                .map(|output| output.port)
                .collect::<Vec<_>>(),
            "selective-fill frames use the same source output ports"
        );
    }

    #[test]
    fn compile_graph_lowers_single_cube() {
        let graph = single_cube_runtime_graph();
        let ctx = CompileContext::new(CompileConfig::new(3));

        let artifacts = ctx.compile(&graph).unwrap();
        let program = artifacts.bloq;

        assert_eq!(program.quantum_node_count(), 1);
        assert_eq!(program.templates().len(), 1);
        let node = program.quantum_nodes().next().unwrap().1;
        assert_eq!(node.instances.len(), 1);
        assert_eq!(node.instances[0].id.0, 0);
        assert_eq!(node.instances[0].template_id.0, 0);
        assert_eq!(node.timeline, None);
        // The cube's logical observable lowered to a classical Observable node.
        assert!(program.nodes().any(|(_, node)| matches!(
            node.try_classical(),
            Some(bloq_ir::ClassicalNode::Observable { .. })
        )));
    }

    /// Minimal selective fixture; its input port makes both cap arms reachable.
    fn selective_measurement_graph() -> BlockGraph {
        BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,-1]\n  1: ZXZ [0,0,0]\n  2: XZ [0,0,1]\n  3: ZXZ [1,0,0]\n  [0,0,-1] -> +Z\n  [0,0,0] -> +Z\n\n  m2 = measure 3\n  resolve 2 if m2\n",
        )
        .expect("selective measurement graph parses")
    }

    #[test]
    fn compile_selective_block_lowers_to_guarded_component() {
        let graph = selective_measurement_graph();
        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx.compile(&graph).expect("selective graph compiles").bloq;

        assert_eq!(program.quantum_node_count(), 4);
        assert!(program.nodes().all(|(_, node)| node.try_region().is_none()));
        assert!(ctx.cache.pool_len() >= 3);
        program.validate().unwrap();
        assert!(matches!(
            bloq_stim::emit_bloq_stim(&program),
            Err(bloq_stim::StimEmissionError::UnsupportedNode(_))
        ));

        let selector_name = bloq_graph::selective_selector_name(ivec3(0, 0, 2));
        let mut arm_ids = Vec::new();
        for choice in [false, true] {
            let pinned = program
                .pin_membership(&std::collections::BTreeMap::from([(
                    selector_name.clone(),
                    choice,
                )]))
                .expect("both selective choices are reachable");
            assert!(!pinned.has_conditional_membership());
            let (owner, quantum) = pinned
                .quantum_nodes()
                .find(|(owner, _)| {
                    pinned[*owner]
                        .block_members()
                        .iter()
                        .any(|member| member.pos == ivec3(0, 0, 2))
                })
                .unwrap();
            assert_eq!(quantum.instances.len(), 1);
            arm_ids.push(quantum.instances[0].id);
            pinned[owner]
                .instantiate_circuit(pinned.templates())
                .unwrap();

            // Each pinned seam reads its predecessor and the selected arm's records.
            let arm = quantum.instances[0].id;
            assert!(pinned.node_detector_count(quantum).unwrap() > 0);
            for detector in pinned.node_detectors(quantum).unwrap() {
                let instances = detector
                    .measurements()
                    .map(|record| record.instance)
                    .collect::<Vec<_>>();
                assert!(instances.contains(&arm));
                assert!(instances.iter().any(|&instance| instance != arm));
            }
        }
        assert_ne!(arm_ids[0], arm_ids[1]);
    }

    #[test]
    fn selective_pins_preserve_shared_compound_controls() {
        use std::collections::BTreeMap;

        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: XZ [0,0,1]\n  2: Port [2,0,0]\n  3: XZ [2,0,1]\n  4: ZXZ [4,0,0]\n  5: ZXZ [6,0,0]\n  [0,0,0] -> +Z\n  [2,0,0] -> +Z\n\n  a = measure 4\n  b = measure 5\n  resolve 1 if a ^ b\n  resolve 3 if a ^ b\n",
        ).unwrap();
        let program = compile(&graph, 3).unwrap();
        let names = [ivec3(0, 0, 1), ivec3(2, 0, 1)].map(bloq_graph::selective_selector_name);
        for choice in [false, true] {
            let pinned = program
                .pin_membership(&BTreeMap::from([
                    (names[0].clone(), choice),
                    (names[1].clone(), choice),
                ]))
                .unwrap();
            assert!(!pinned.has_conditional_membership());
            pinned.validate().unwrap();
        }
        assert!(matches!(
            program.pin_membership(&BTreeMap::from([
                (names[0].clone(), false),
                (names[1].clone(), true),
            ])),
            Err(bloq_ir::MembershipPinError::UnreachableAssignment)
        ));
    }

    /// A logical operator threaded through a selective measurement: a `-Z`-boundary
    /// port initializes it, the selective `XZ` block measures it, and `m2` (a side
    /// cube) drives the resolve. Unlike [`selective_measurement_graph`] (whose only
    /// generator, `m2`, does not touch the selective site), here a `Logical`
    /// generator crosses the selective block, exercising the branch-conditioned
    /// observable lowering.
    fn selective_crossing_graph() -> BlockGraph {
        BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: XZ [0,0,1]\n  2: ZXZ [1,0,0]\n  [0,0,0] -> +Z\n\n  m2 = measure 2\n  resolve 1 if m2\n",
        )
        .expect("selective crossing graph parses")
    }

    fn value_with_decode(program: &Bloq, id: bloq_ir::BloqNodeId, control: bool) -> bool {
        use bloq_ir::ClassicalNode;

        match program[id].try_classical().expect("classical fold") {
            ClassicalNode::Observable { .. } => control,
            ClassicalNode::Compute { expr } => expr
                .eval(&mut |slot| {
                    let input = program
                        .top()
                        .value_inputs(id)
                        .find(|input| input.slot == slot)?;
                    Some(value_with_decode(program, input.producer, control))
                })
                .expect("all fold inputs are available"),
            other => panic!("unexpected fold producer: {other:?}"),
        }
    }

    #[test]
    fn feedback_folding_respects_the_resolve_condition() {
        use bloq_ir::{ClassicalNode, ValueRole};

        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0,0,0]\n  1: XZ [0,0,1]\n  2: ZXZ [1,0,0]\n  [0,0,0] -> +Z\n\n  m2 = measure 2\n  resolve 1 if m2\n  feedback X 1 if m2\n",
        )
        .expect("delta-only feedback graph parses");
        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx
            .compile(&graph)
            .expect("delta-only feedback compiles")
            .bloq;
        let folds: Vec<_> = program.nodes()
            .filter(|(_, node)| matches!(node.try_classical(), Some(ClassicalNode::Observable { index, .. }) if *index != Some(0)))
            .flat_map(|(observable, _)| program.incoming(observable))
            .filter(|edge| matches!(edge.edge, bloq_ir::BloqEdge::Value { role: ValueRole::FeedbackFold { .. }, .. }))
            .collect();

        // False selects Z, but disables feedback. True enables X feedback on
        // the X arm, where it commutes. The total fold is zero in both cases.
        for control in [false, true] {
            assert!(!folds.iter().fold(false, |odd, edge| {
                odd ^ value_with_decode(&program, edge.source, control)
            }));
        }
    }

    /// Both selector values retain the selected input operator and read only
    /// the active arm's records, independent of how the Boolean rows are split.
    #[test]
    fn selective_crossing_observable_reads_selected_arm() {
        use bloq_graph::Pauli;
        use bloq_ir::{BloqEdge, ClassicalNode};
        use std::collections::BTreeSet;

        let program = CompileContext::new(CompileConfig::default())
            .compile(&selective_crossing_graph())
            .unwrap()
            .bloq;
        program.validate().unwrap();
        let (branch, quantum) = program
            .quantum_nodes()
            .find(|(_, quantum)| quantum.instances.len() == 2 && !quantum.guards.is_empty())
            .unwrap();
        let arms = [false, true].map(|control| {
            let selected = program[branch]
                .select_quantum_members(|slot| {
                    let input = program
                        .value_inputs(branch)
                        .find(|input| input.slot == slot)?;
                    Some(value_with_decode(&program, input.producer, control))
                })
                .unwrap();
            selected.expect_quantum().instances[0].id
        });
        assert_eq!(quantum.instances.len(), arms.len());
        for control in [false, true] {
            let mut selected = 0;
            for (observable, _) in program.nodes().filter(|(_, node)| {
                matches!(node.try_classical(), Some(ClassicalNode::Observable { index, .. }) if *index != Some(0))
            }) {
                let mut bases = Vec::new();
                let mut records = BTreeSet::new();
                for input in program.top().data_inputs(observable).filter(|input| input.output.is_none()) {
                    let producer = input.producer;
                    let node = &program[producer];
                    if let Some(slot) = node.activation {
                        let guard = program.value_inputs(producer).find(|input| input.slot == slot).unwrap().producer;
                        if !value_with_decode(&program, guard, control) { continue; }
                    }
                    if let Some(ClassicalNode::Observable { measurements, operators, .. }) = node.try_classical() {
                        bases.extend(operators.iter().flat_map(|operator| operator.operator.iter().map(|(_, pauli)| *pauli)));
                        records.extend(measurements.iter().map(|record| record.instance));
                        if measurements.iter().any(|record| arms.contains(&record.instance)) {
                            assert!(program.incoming(producer).any(|edge| edge.source == branch && *edge.edge == BloqEdge::Order));
                        }
                    }
                }
                if bases.is_empty() { continue; }
                selected += 1;
                assert!(bases.iter().all(|&basis| basis == if control { Pauli::X } else { Pauli::Z }));
                assert!(records.contains(&arms[usize::from(control)]));
                assert!(!records.contains(&arms[usize::from(!control)]));
            }
            assert_eq!(selected, 1, "one selected input correlation per arm");
        }
        assert!(matches!(
            bloq_stim::emit_bloq_stim(&program),
            Err(bloq_stim::StimEmissionError::UnsupportedNode(_))
        ));
    }

    #[test]
    fn compile_lowers_observable_to_classical_nodes() {
        use bloq_ir::{BloqEdge, ClassicalNode};

        let graph = single_cube_runtime_graph();
        let ctx = CompileContext::new(CompileConfig::new(3));
        let program = ctx.compile(&graph).unwrap().bloq;
        let g = &program;

        assert_eq!(program.quantum_node_count(), 1);

        let observables = g
            .nodes()
            .filter_map(|(id, node)| match node.try_classical() {
                Some(ClassicalNode::Observable { measurements, .. })
                    if !measurements.is_empty() =>
                {
                    Some(id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            !observables.is_empty(),
            "records belong directly to complete observables"
        );
        let order = program.deterministic_emit_order().expect("acyclic");
        let position = |id| order.iter().position(|&node| node == id).unwrap();
        for observable in observables {
            let owners = g
                .incoming(observable)
                .filter(|edge| {
                    matches!(edge.edge, BloqEdge::Order) && g[edge.source].try_quantum().is_some()
                })
                .collect::<Vec<_>>();
            assert!(
                !owners.is_empty(),
                "the complete readout waits for producing quantum work"
            );
            for edge in owners {
                assert!(position(edge.source) < position(observable));
            }
        }
    }

    #[test]
    fn try_new_rejects_invalid_distance() {
        // Illegal distances fail at the API boundary, not deep in block
        // compilation: an even or sub-3 distance never yields a `CompileConfig`.
        assert_eq!(CompileConfig::try_new(2), Err(crate::InvalidDistance(2)));
        assert_eq!(CompileConfig::try_new(1), Err(crate::InvalidDistance(1)));
        CompileConfig::try_new(3).unwrap();
    }

    #[test]
    fn compile_preserves_scaled_cube_top_temporal_pipe_ref() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            bloq_graph::Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(bloq_graph::Block::new(
            ivec3(0, 0, 2),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(bloq_graph::Pipe::new(
            ivec3(0, 0, 1),
            bloq_graph::Direction::ZPLUS,
        ));
        let ctx = CompileContext::new(CompileConfig::new(3));

        let artifacts = ctx.compile(&graph).expect("scaled temporal graph compiles");
        let pipes = artifacts
            .bloq
            .edges()
            .flat_map(|edge| edge.edge.pipes().iter())
            .collect::<Vec<_>>();

        assert_eq!(artifacts.bloq.quantum_node_count(), 2);
        assert_eq!(pipes.len(), 1);
        assert_eq!(pipes[0].pipe.src, ivec3(0, 0, 1));
        assert_eq!(pipes[0].pipe.dst, ivec3(0, 0, 2));
        assert_eq!(
            source_timeline(&artifacts.bloq, ivec3(0, 0, 0)),
            Some(&bloq_ir::QuantumTimeline {
                layer_round_ends: vec![3, 6],
            })
        );
    }

    #[test]
    fn tall_cube_timeline_packs_from_the_connected_side() {
        let tall = |pos| {
            Block::new(pos, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d/2".parse().expect("valid height"))
                .expect("height fits")
        };

        let mut high_anchored = BlockGraph::new();
        high_anchored.add_block(tall(IVec3::ZERO));
        let high = compile(&high_anchored, 5).expect("isolated tall cube compiles");
        assert_eq!(
            source_timeline(&high, IVec3::ZERO),
            Some(&bloq_ir::QuantumTimeline {
                layer_round_ends: vec![3, 8],
            })
        );

        let mut low_anchored = BlockGraph::new();
        low_anchored.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        low_anchored.add_block(tall(IVec3::Z));
        low_anchored.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        let low = compile(&low_anchored, 5).expect("connected tall cube compiles");
        assert_eq!(
            source_timeline(&low, IVec3::Z),
            Some(&bloq_ir::QuantumTimeline {
                layer_round_ends: vec![5, 8],
            })
        );
    }

    #[test]
    fn compile_graph_lowers_temporal_y_pair_with_inferred_basis() {
        let mut graph = BlockGraph::new();
        graph.add_block(bloq_graph::Block::new(ivec3(0, 0, 0), BlockKind::Y));
        graph.add_block(bloq_graph::Block::new(ivec3(0, 0, 1), BlockKind::Y));
        graph.add_pipe(bloq_graph::Pipe::new(
            ivec3(0, 0, 0),
            bloq_graph::Direction::ZPLUS,
        ));
        let ctx = CompileContext::new(CompileConfig::new(3));

        let artifacts = ctx.compile(&graph).unwrap();

        assert_eq!(artifacts.bloq.quantum_node_count(), 2);
    }

    #[test]
    fn compile_context_outputs_only_graph_local_templates() {
        let unit_graph = single_cube_runtime_graph();
        let mut scaled_graph = BlockGraph::new();
        scaled_graph.add_block(
            bloq_graph::Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );
        let ctx = CompileContext::new(CompileConfig::new(3));

        ctx.compile(&unit_graph).expect("compile unit cube");
        let artifacts = ctx.compile(&scaled_graph).expect("compile scaled cube");

        assert_eq!(ctx.cache.pool_len(), 2);
        assert_eq!(artifacts.bloq.templates().len(), 1);
        let (_, node) = artifacts.bloq.nodes().next().unwrap();
        assert_eq!(node.expect_quantum().instances[0].template_id.0, 0);
    }

    /// Gallery graphs spanning the template kinds a shared cache must handle:
    /// fixed cubes, a temporal Hadamard, a selective arm pair, and a T stage
    /// pair.
    const SHARED_CACHE_CASES: [GalleryItem; 4] = [
        GalleryItem::CNOT,
        GalleryItem::XMemory,
        GalleryItem::S,
        GalleryItem::T,
    ];

    fn compile_to_binary(ctx: &CompileContext, graph: &BlockGraph) -> Vec<u8> {
        ctx.compile(graph)
            .expect("gallery graph compiles")
            .bloq
            .to_binary()
    }

    /// Sharing a cache reorders pool ids (a context sees templates it did not
    /// compile) but must not reach the output: program-local template ids are
    /// assigned per `Bloq` in lowering order.
    #[test]
    fn shared_cache_compiles_byte_identically_to_a_private_context() {
        let config = CompileConfig::new(3);
        let cache = crate::SharedCompileCache::new();

        for item in SHARED_CACHE_CASES {
            let graph = item.build().flatten().expect("gallery graph expands");
            let private = compile_to_binary(&CompileContext::new(config), &graph);
            // A second context on the same cache reads the first's templates.
            let warming = CompileContext::with_shared_cache(config, &cache);
            compile_to_binary(&warming, &graph);
            let shared =
                compile_to_binary(&CompileContext::with_shared_cache(config, &cache), &graph);

            assert_eq!(private, shared, "{item:?} compiles identically when shared");
        }
    }

    // ==========================================================================
    // Cube height in the template cache key
    // ==========================================================================

    /// A cube of `height` at `(0, 0, 0)`, capped by a port on its top temporal
    /// endpoint. The port sits at `cells` above the anchor, so both graphs here
    /// present the same `ZPLUS` connectivity from different footprints.
    fn capped_cube(height: &str, cells: i32) -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(
            bloq_graph::Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ))
                .with_height(height.parse().expect("valid height"))
                .expect("cube accepts a height"),
        );
        graph.add_block(bloq_graph::Block::new(ivec3(0, 0, cells), BlockKind::Port));
        graph.add_pipe(bloq_graph::Pipe::new(
            ivec3(0, 0, cells - 1),
            bloq_graph::Direction::ZPLUS,
        ));
        graph
    }

    /// Cubes with different graph heights but the same resolved rounds share a
    /// template. Reusing it must not affect graph-specific lowering.
    #[test]
    fn cube_templates_may_share_rounds_across_different_footprints() {
        const DISTANCE: u32 = 5;
        let config = CompileConfig::new(DISTANCE);
        let two_cell = capped_cube("2d", 2);
        let one_cell = capped_cube("d+5", 1);
        let signature_of = |graph: &BlockGraph| {
            let schedules =
                join_component_layer_schedules(graph).expect("no mixed spatial Hadamard walls");
            block_signatures(graph, &schedules, DISTANCE).expect("valid graph")[&ivec3(0, 0, 0)]
        };
        assert_eq!(signature_of(&two_cell), signature_of(&one_cell));

        let private = compile_to_binary(&CompileContext::new(config), &one_cell);
        let cache = crate::SharedCompileCache::new();
        compile_to_binary(
            &CompileContext::with_shared_cache(config, &cache),
            &two_cell,
        );
        let shared = CompileContext::with_shared_cache(config, &cache);
        let warmed = shared.cache.pool_len();

        assert_eq!(compile_to_binary(&shared, &one_cell), private);
        assert_eq!(shared.cache.pool_len(), warmed);
    }

    /// Clearing detaches template shards without changing what a later
    /// compilation produces.
    #[test]
    fn clearing_a_shared_cache_releases_templates_and_recompiles_identically() {
        let config = CompileConfig::new(3);
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("gallery graph expands");
        let cache = crate::SharedCompileCache::new();

        let warm = CompileContext::with_shared_cache(config, &cache);
        let expected = compile_to_binary(&warm, &graph);
        assert!(warm.cache.pool_len() > 0, "the warm shard holds templates");

        cache.clear();

        // New contexts start from an empty shard; `warm` retains its immutable
        // templates, so nothing in flight breaks.
        let cold = CompileContext::with_shared_cache(config, &cache);
        assert_eq!(cold.cache.pool_len(), 0);
        assert_eq!(compile_to_binary(&cold, &graph), expected);
    }

    /// Contexts sharing one cache may compile concurrently; every output must
    /// match what a private context produces alone.
    #[test]
    fn concurrent_shared_cache_compiles_match_sequential_output() {
        let config = CompileConfig::new(3);
        let graphs: Vec<BlockGraph> = SHARED_CACHE_CASES
            .iter()
            .map(|item| item.build().flatten().expect("gallery graph expands"))
            .collect();
        let sequential: Vec<Vec<u8>> = graphs
            .iter()
            .map(|graph| compile_to_binary(&CompileContext::new(config), graph))
            .collect();

        let cache = crate::SharedCompileCache::new();
        let concurrent: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = graphs
                .iter()
                .map(|graph| {
                    let cache = &cache;
                    scope.spawn(move || {
                        let ctx = CompileContext::with_shared_cache(config, &cache.clone());
                        // Twice: the second pass races the other threads' inserts
                        // against warm hits on this thread.
                        compile_to_binary(&ctx, graph);
                        compile_to_binary(&ctx, graph)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("compile thread does not panic"))
                .collect()
        });

        assert_eq!(concurrent, sequential);
    }

    #[test]
    fn compile_graph_lowers_isolated_diagonal_walking_block() {
        let mut graph = BlockGraph::new();
        graph.add_block(bloq_graph::Block::new(
            ivec3(0, 0, 0),
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 1)).unwrap(),
            ),
        ));
        let ctx = CompileContext::new(CompileConfig::new(3));

        let artifacts = ctx.compile(&graph).unwrap();

        assert_eq!(artifacts.bloq.quantum_node_count(), 1);
    }

    #[test]
    fn compile_graph_lowers_isolated_axial_walking_block() {
        let mut graph = BlockGraph::new();
        graph.add_block(bloq_graph::Block::new(
            ivec3(0, 0, 0),
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 0)).unwrap(),
            ),
        ));
        let ctx = CompileContext::new(CompileConfig::new(3));

        let artifacts = ctx.compile(&graph).unwrap();

        assert_eq!(artifacts.bloq.quantum_node_count(), 1);
        assert_eq!(
            source_timeline(&artifacts.bloq, IVec3::ZERO),
            Some(&bloq_ir::QuantumTimeline {
                layer_round_ends: vec![4, 8],
            })
        );
    }

    #[test]
    fn compile_graph_sets_patch_rotation_timeline() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, glam::IVec2::X).expect("valid rotation");
        let start = IVec3::Z;
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(start, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(ivec3(1, 0, 3), BlockKind::Port));
        graph.add_pipe(Pipe::new(start, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(start), Direction::ZPLUS));

        let program = compile(&graph, 3).expect("patch rotation compiles");
        assert_eq!(
            source_timeline(&program, start),
            Some(&bloq_ir::QuantumTimeline {
                layer_round_ends: vec![3, 6],
            })
        );
    }

    #[test]
    fn compile_graph_lowers_ghz_slide_then_glide_gallery_asset() {
        let graph = GalleryItem::GHZSlideThenGlide
            .build()
            .flatten()
            .expect("gallery graph expands");
        let ctx = CompileContext::new(CompileConfig::new(3));
        let variants = graph
            .fill_ports_auto()
            .expect("GHZ slide-then-glide ports fill");
        assert!(!variants.is_empty());

        for (variant, (filled_graph, _)) in variants.iter().enumerate() {
            let artifacts = ctx.compile(filled_graph).unwrap_or_else(|error| {
                panic!("GHZ slide-then-glide fill variant {variant} compiles: {error}")
            });

            artifacts
                .bloq
                .validate()
                .expect("GHZ slide-then-glide template instance refs validate");
        }
    }

    /// The point of `compile(&self)`: one context, many threads, no `&mut`
    /// juggling — and identical output to compiling each graph alone.
    #[test]
    fn one_context_compiles_concurrently_from_shared_borrows() {
        let graphs: Vec<BlockGraph> = SHARED_CACHE_CASES
            .iter()
            .map(|item| item.build().flatten().expect("gallery graph expands"))
            .collect();
        let expected: Vec<Vec<u8>> = graphs
            .iter()
            .map(|graph| compile_to_binary(&CompileContext::new(CompileConfig::default()), graph))
            .collect();

        let ctx = CompileContext::new(CompileConfig::default());
        let concurrent: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = graphs
                .iter()
                .map(|graph| {
                    let ctx = &ctx;
                    scope.spawn(move || compile_to_binary(ctx, graph))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("compile thread does not panic"))
                .collect()
        });

        assert_eq!(concurrent, expected);
    }

    #[test]
    fn one_shot_compile_matches_the_context_path() {
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("gallery graph expands");
        let expected = CompileContext::new(CompileConfig::new(3))
            .compile(&graph)
            .expect("context compiles")
            .bloq;

        assert_eq!(
            compile(&graph, 3).expect("one-shot compiles").to_binary(),
            expected.to_binary()
        );
        assert_eq!(
            compile_with(&graph, CompileConfig::default())
                .expect("configured one-shot compiles")
                .bloq
                .to_binary(),
            expected.to_binary()
        );
    }

    /// The one-shot validates its distance instead of panicking like
    /// `CompileConfig::new` — it is the path user input reaches.
    #[test]
    fn one_shot_compile_rejects_an_even_distance() {
        assert!(matches!(
            compile(
                &GalleryItem::CNOT
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
                4
            ),
            Err(CompileError::InvalidDistance(4))
        ));
    }

    #[test]
    fn temporal_hadamard_from_port_uses_the_pipe_basis() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_hadamard());

        compile(&graph, 3)
            .expect("Port -> H -> ZXZ compiles")
            .validate()
            .expect("compiled Bloq validates");
    }

    #[test]
    fn temporal_hadamard_from_tall_cube_uses_its_top_endpoint() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("cube accepts height"),
        );
        graph.add_block(Block::new(IVec3::new(0, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::Z, Direction::ZPLUS).with_hadamard());

        compile(&graph, 3)
            .expect("tall ZXZ -> H -> Port compiles")
            .validate()
            .expect("compiled Bloq validates");
    }

    #[test]
    fn compiled_metadata_round_trips_through_the_program() {
        let bloq = compile(
            &GalleryItem::CNOT
                .build()
                .flatten()
                .expect("gallery graph expands"),
            5,
        )
        .expect("compiles");

        assert_eq!(compiled_distance(&bloq), Some(5));
        assert_eq!(compiled_distance(&Bloq::default()), None);
    }

    fn cached_module_definitions(
        context: &CompileContext,
        program: &BlockGraph,
    ) -> std::collections::HashMap<String, Arc<PreparedDefinitionObject>> {
        let orientations = program.definition_orientations();
        program
            .module_definition_cache_keys()
            .into_iter()
            .map(|(name, implementation)| {
                let object = context
                    .cache
                    .definition_object(&DefinitionObjectCacheKey {
                        implementation,
                        orientations: orientations[&name].clone(),
                    })
                    .expect("compiled definition is cached");
                (name, object)
            })
            .collect()
    }

    #[test]
    fn shared_cache_reuses_the_compiled_module_object() {
        let cache = crate::SharedCompileCache::new();
        let config = CompileConfig::default();
        let first = CompileContext::with_shared_cache(config, &cache)
            .compile_object(&GalleryItem::PhaseGradientK4.build())
            .unwrap();
        let second = CompileContext::with_shared_cache(config, &cache)
            .compile_object(&GalleryItem::PhaseGradientK4.build())
            .unwrap();

        assert!(Arc::ptr_eq(&first.artifacts, &second.artifacts));
    }

    #[test]
    fn module_link_reports_a_missing_physical_variant() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        let definitions = [Arc::new(PreparedDefinitionObject {
            name: "Leaf".into(),
            dependencies: Vec::new(),
            variants: crate::FxMap::default(),
        })];
        let sites = [PreparedSiteRelocation {
            position: IVec3::ZERO,
            definition: 0,
            local_position: IVec3::ZERO,
        }];

        assert!(matches!(
            select_module_link_templates_with_schedule(
                CompileConfig::default(),
                &graph,
                &SpatialPortExpansionMap::default(),
                &definitions,
                &sites,
                &crate::signature::join_component_layer_schedules(&graph).unwrap(),
                &crate::FxMap::default(),
            ),
            Err(CompileError::MissingModulePhysicalVariant { .. })
        ));
    }

    #[test]
    fn clearing_shared_cache_evicts_objects_from_live_contexts() {
        let cache = crate::SharedCompileCache::new();
        let context = CompileContext::with_shared_cache(CompileConfig::default(), &cache);
        let program = GalleryItem::PhaseGradientK4.build();
        let first = context.compile_object(&program).unwrap();

        cache.clear();
        let second = context.compile_object(&program).unwrap();

        assert!(!Arc::ptr_eq(&first.artifacts, &second.artifacts));
    }

    #[test]
    fn flat_oracle_artifacts_do_not_replace_hierarchical_objects() {
        let flat = GalleryItem::CNOT.build().flatten().unwrap();
        let structured = flat.clone().with_inferred_interface().unwrap();
        assert_eq!(flat.to_blog_text(), structured.to_blog_text());
        let context = CompileContext::new(CompileConfig::default());
        let oracle = context.compile_object(&flat).unwrap();
        let hierarchy = context.compile_object(&structured).unwrap();
        assert!(!Arc::ptr_eq(&oracle.artifacts, &hierarchy.artifacts));
        assert!(!cached_module_definitions(&context, &structured).is_empty());
    }

    #[test]
    fn unified_compile_revalidates_edited_hierarchy_before_cache_reuse() {
        let mut graph = BlockGraph::from_text(
            "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\n\
             module main {\nchild: Leaf @ [0,0,0]\n}\n",
        )
        .unwrap();
        let context = CompileContext::new(CompileConfig::default());
        context.compile(&graph).unwrap();
        context
            .summarize(&graph, ModuleCertificationLimits::DEFAULT)
            .unwrap();
        graph.instances[0].definition = "Missing".to_string();
        let error = context.compile(&graph).unwrap_err();
        assert!(matches!(error, CompileError::Module(_)), "{error:?}");
        assert!(error.to_string().contains("Missing"), "{error}");
        let summary_error = context
            .summarize(&graph, ModuleCertificationLimits::DEFAULT)
            .unwrap_err();
        assert!(
            matches!(summary_error, ModuleCertificationError::Graph { .. }),
            "{summary_error:?}"
        );
        assert!(
            summary_error.to_string().contains("Missing"),
            "{summary_error}"
        );

        graph.instances[0].definition = BlockGraph::ENTRY_MODULE.to_string();
        let summary_error = context
            .summarize(&graph, ModuleCertificationLimits::DEFAULT)
            .unwrap_err();
        assert!(
            matches!(summary_error, ModuleCertificationError::Graph { .. }),
            "{summary_error:?}"
        );
        assert!(
            summary_error.to_string().contains("cycle"),
            "{summary_error}"
        );
    }

    #[test]
    fn unified_compile_checks_interfaces_added_to_a_plain_graph() {
        let mut graph = single_cube_runtime_graph();
        graph.interface.quantum_ports.push(bloq_graph::QuantumPort {
            name: "missing_port".to_string(),
            position: IVec3::X,
            direction: bloq_graph::PortDirection::Input,
            resource_type: "data".to_string(),
        });
        assert!(graph.has_module_structure());
        let error = CompileContext::new(CompileConfig::default())
            .compile(&graph)
            .unwrap_err();
        assert!(matches!(error, CompileError::Module(_)), "{error:?}");
        assert!(error.to_string().contains("missing_port"), "{error}");
    }

    #[test]
    fn module_assembly_retains_cache_hits_across_a_clear() {
        let program = BlockGraph::from_text(
            "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\n\
             module main {\nleaf: Leaf @ [0,0,0]\n}\n",
        )
        .unwrap();
        let cache = crate::SharedCompileCache::new();
        let context = CompileContext::with_shared_cache(CompileConfig::default(), &cache);
        context.compile_object(&program).unwrap();
        let mut retained = cached_module_definitions(&context, &program);
        let root = retained.remove(BlockGraph::ENTRY_MODULE).unwrap();
        let parts = std::collections::HashMap::from([(root.name.clone(), root.variants.clone())]);
        let mut names = program
            .module_definition_cache_keys()
            .into_keys()
            .collect::<Vec<_>>();
        names.sort_unstable();

        // A deterministic interleaving: eviction after lookups but before
        // assembling a new root from a cached child and newly compiled parts.
        cache.clear();
        let definitions = build_definition_objects(
            &context.cache,
            &names,
            parts,
            retained,
            &program.module_definition_cache_keys(),
            &program.definition_orientations(),
            &program,
        );

        let rebuilt_root = definitions
            .iter()
            .find(|definition| definition.name == "main")
            .unwrap();
        assert!(Arc::ptr_eq(
            &rebuilt_root.dependencies[0],
            &root.dependencies[0]
        ));
    }

    #[test]
    fn serial_and_parallel_module_builds_link_identically() {
        let continuing = BlockGraph::from_text(
            "BLOG 1.0
module main {
  0: ZXZ [0,0,0]
  1: ZXZ [0,0,2]
  9: ZXZ [2,0,0]
  branch b {
    false {
      2: XZX [0,0,1]
      0 -H> +Z
      [0,0,1] -H> +Z
    }
    true {
      3: ZXZ [0,0,1]
      0 -> +Z
      [0,0,1] -> +Z
    }
  }
  m = measure 9
  resolve b if m
}
",
        )
        .unwrap();
        for (label, program) in [
            ("phase gradient", GalleryItem::PhaseGradientK4.build()),
            ("CCZ", GalleryItem::CCZGateTeleport.build()),
            ("continuing", continuing),
        ] {
            let serial_context = CompileContext::new(CompileConfig::default());
            let serial = serial_context
                .compile_object_with_jobs(&program, NonZeroUsize::MIN)
                .and_then(|object| serial_context.link_object(&object))
                .unwrap();
            let parallel_context = CompileContext::new(CompileConfig::default());
            let parallel = parallel_context
                .compile_object_with_jobs(&program, NonZeroUsize::new(4).expect("four is nonzero"))
                .and_then(|object| parallel_context.link_object(&object))
                .unwrap();

            assert_eq!(
                serial.bloq.to_binary(),
                parallel.bloq.to_binary(),
                "{label}"
            );
        }
    }

    #[test]
    fn definition_objects_survive_a_different_root_archive_key() {
        let cache = crate::SharedCompileCache::new();
        let config = CompileConfig::default();
        let program = BlockGraph::from_text(
            "BLOG 1.0\nmodule Leaf {\n0: ZXZ [0,0,0]\n}\n\
             module main {\na: Leaf @ [0,0,0]\nb: Leaf @ [2,0,0]\n}\n",
        )
        .unwrap();
        let first_context = CompileContext::with_shared_cache(config, &cache);
        let first = first_context.compile_object(&program).unwrap();
        let first_definitions = cached_module_definitions(&first_context, &program);

        let mut modules = program
            .modules()
            .map(BlockGraph::clone_local_definition)
            .collect::<Vec<_>>();
        let mut unused = BlockGraph::from_text(include_str!(
            "../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap()
        .clone_local_definition();
        unused.name = "Unused".into();
        modules.push(unused);
        let with_unused = bloq_graph::BlockGraph::from_definitions(modules)
            .expect("unused definition keeps the root valid");
        let second_context = CompileContext::with_shared_cache(config, &cache);
        let second = second_context.compile_object(&with_unused).unwrap();
        let second_definitions = cached_module_definitions(&second_context, &with_unused);

        assert!(!Arc::ptr_eq(&first.artifacts, &second.artifacts));
        assert_eq!(first_definitions.len(), second_definitions.len());
        for (name, left) in first_definitions {
            assert!(
                Arc::ptr_eq(&left, &second_definitions[&name]),
                "{name} was rebuilt"
            );
        }
    }

    #[test]
    fn spatial_module_seam_promotes_both_sides_before_template_selection() {
        let program = BlockGraph::from_text(
            "BLOG 1.0\n\n\
             module A {\n\
               out q: data = 1\n\
               0: XXZ [0,0,0]\n\
               1: Port [1,0,0] role=output <q>\n\
               0 -> +X\n\
             }\n\n\
             module B {\n\
               in q: data = 0\n\
               0: Port [-1,0,0] role=input <q>\n\
               1: ZXZ [0,0,0]\n\
               0 -> +X\n\
             }\n\n\
             module main {\n\
               a: A @ [0,0,0]\n\
               b: B @ [1,0,0]\n\
               a.q -> b.q\n\
             }\n",
        )
        .unwrap();
        let context = CompileContext::new(CompileConfig::default());
        let linked = context.compile(&program).unwrap();
        let graph = program.flatten().unwrap().fix_shadowed_faces();
        let signatures = select_link_signatures(
            context.config,
            &graph,
            &SpatialPortExpansionMap::default(),
            &context.flat_signature_builds,
        )
        .unwrap();

        for pos in [IVec3::ZERO, IVec3::X] {
            let signature = &signatures[&pos];
            assert_eq!(signature.layer_schedule, Some(LayerSchedule::Padded));
        }
        linked.bloq.validate().unwrap();
        let component = linked
            .bloq
            .nodes()
            .find_map(|(_, node)| (node.block_members().len() == 2).then_some(node))
            .expect("the spatial seam fuses both module cubes");
        assert_eq!(component.expect_quantum().instances.len(), 2);
    }

    #[test]
    fn fat_definition_object_reuses_a_cube_across_compact_and_padded_seams() {
        let pair = |neighbor: &str| {
            BlockGraph::from_text(&format!(
                "BLOG 1.0\n\n\
                 module A {{\n\
                   out q: data = 1\n\
                   0: ZXZ [0,0,0]\n\
                   1: Port [1,0,0] role=output <q>\n\
                   0 -> +X\n\
                 }}\n\n\
                 module B {{\n\
                   in q: data = 0\n\
                   0: Port [-1,0,0] role=input <q>\n\
                   1: {neighbor} [0,0,0]\n\
                   0 -> +X\n\
                 }}\n\n\
                 module main {{\n\
                   a: A @ [0,0,0]\n\
                   b: B @ [1,0,0]\n\
                   a.q -> b.q\n\
                 }}\n"
            ))
            .unwrap()
        };
        let cache = crate::SharedCompileCache::new();
        let config = CompileConfig::default();
        let compact_program = pair("ZXZ");
        let padded_program = pair("XXZ");
        let compact_context = CompileContext::with_shared_cache(config, &cache);
        let compact = compact_context.compile(&compact_program).unwrap();
        let compact_definition = cached_module_definitions(&compact_context, &compact_program)
            .remove("A")
            .unwrap();
        let padded_context = CompileContext::with_shared_cache(config, &cache);
        let padded = padded_context.compile(&padded_program).unwrap();
        let padded_definition = cached_module_definitions(&padded_context, &padded_program)
            .remove("A")
            .unwrap();

        assert!(Arc::ptr_eq(&compact_definition, &padded_definition));
        let schedules = compact_definition
            .variants
            .keys()
            .filter(|variant| variant.local_position == IVec3::ZERO)
            .filter_map(|variant| variant.signature.layer_schedule)
            .collect::<crate::FxSet<_>>();
        assert!(schedules.contains(&LayerSchedule::Compact));
        assert!(schedules.contains(&LayerSchedule::Padded));
        let selected = |program: &BlockGraph| {
            let context = CompileContext::new(config);
            let graph = program.flatten().unwrap().fix_shadowed_faces();
            select_link_signatures(
                context.config,
                &graph,
                &SpatialPortExpansionMap::default(),
                &context.flat_signature_builds,
            )
            .unwrap()[&IVec3::ZERO]
                .layer_schedule
                .unwrap()
        };
        assert_eq!(selected(&compact_program), LayerSchedule::Compact);
        assert_eq!(selected(&padded_program), LayerSchedule::Padded);
        compact.bloq.validate().unwrap();
        padded.bloq.validate().unwrap();
    }

    #[test]
    fn flat_signature_guard_observes_worker_threads() {
        let builds = std::sync::atomic::AtomicUsize::new(0);
        let graph = GalleryItem::CNOT
            .build()
            .flatten()
            .expect("gallery graph expands");
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    select_link_signatures(
                        CompileConfig::default(),
                        &graph,
                        &SpatialPortExpansionMap::default(),
                        &builds,
                    )
                    .unwrap();
                })
                .join()
                .unwrap();
        });

        assert_eq!(builds.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn module_link_never_discovers_flat_signatures() {
        let context = CompileContext::new(CompileConfig::default());
        context
            .compile(&GalleryItem::PhaseGradientK4.build())
            .unwrap();
        let branch = BlockGraph::from_definitions(vec![BlockGraph::definition(
            BlockGraph::ENTRY_MODULE,
            structural_branch_graph(),
            bloq_graph::ModuleInterface::default(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )])
        .unwrap();
        context.compile(&branch).unwrap();

        assert_eq!(
            context
                .flat_signature_builds
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    /// A spatial Hadamard wall compiles, but its effective distance may fall
    /// below `d` — the artifact carries that advisory so a front end cannot
    /// forget to ask for it (LIM-016).
    #[test]
    fn artifacts_carry_the_spatial_hadamard_distance_warning() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::XZX)));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::XXZ)));
        graph.add_pipe(
            bloq_graph::Pipe::new(IVec3::ZERO, bloq_graph::Direction::XPLUS).with_hadamard(),
        );

        let artifacts = compile_with(&graph, CompileConfig::default()).expect("wall compiles");

        assert_eq!(
            artifacts.warnings,
            crate::spatial_hadamard_distance_warning(&graph)
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(!artifacts.warnings.is_empty());
        assert!(
            compile_with(
                &GalleryItem::CNOT
                    .build()
                    .flatten()
                    .expect("gallery graph expands"),
                CompileConfig::default()
            )
            .expect("CNOT compiles")
            .warnings
            .is_empty()
        );
    }
}
