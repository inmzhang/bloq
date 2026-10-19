//! Shared template, module-summary, definition, and compiled-artifact caches.
//!
//! Template keys contain block signatures. Config-sharded caches keep physical
//! targets separate; logical summaries are target-independent. Immutable
//! templates and artifacts use `Arc`, so contexts may reuse them concurrently.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bloq_graph::{
    BlockGraph, ModuleCertificationError, ModuleCertificationLimits, ModuleSummary,
    PreparedModuleRelations,
};

use crate::block::{LoweringTemplate, LoweringTemplateId, LoweringTemplatePool};
use crate::compile::{Signature, TemplateRef};
use crate::config::CompileConfig;
use crate::error::CompileError;

/// A compile cache shareable by several [`CompileContext`](crate::CompileContext)s, including across
/// threads.
///
/// Clones share one cache. Hand it to
/// [`CompileContext::with_shared_cache`](crate::CompileContext::with_shared_cache)
/// to build contexts that pool their compiled templates:
///
/// ```
/// use bloq_compile::{CompileConfig, CompileContext, SharedCompileCache};
/// use bloq_graph::GalleryItem;
///
/// let cache = SharedCompileCache::new();
/// let config = CompileConfig::new(3);
/// let first = CompileContext::with_shared_cache(config, &cache);
/// let second = CompileContext::with_shared_cache(config, &cache);
///
/// first.compile(&GalleryItem::CNOT.build()).expect("CNOT compiles");
/// // Reuses the templates the first context compiled.
/// second.compile(&GalleryItem::CNOT.build()).expect("CNOT compiles");
/// ```
///
/// Sharing never changes what a compilation produces: program-local template
/// ids are assigned per output [`Bloq`](crate::Bloq) in lowering order, so a
/// warm cache and a cold one emit byte-identical programs.
#[derive(Debug, Clone, Default)]
pub struct SharedCompileCache {
    shards: Arc<Mutex<crate::FxMap<CompileConfig, Arc<TemplateCacheShard>>>>,
    summaries: Arc<ModuleSummaryCache>,
}

impl SharedCompileCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The shard backing `config`, created on first use.
    pub(crate) fn shard(&self, config: CompileConfig) -> Arc<TemplateCacheShard> {
        Arc::clone(
            lock(&self.shards)
                .entry(config)
                .or_insert_with(|| Arc::new(TemplateCacheShard::default())),
        )
    }

    pub(crate) fn summary_cache(&self) -> Arc<ModuleSummaryCache> {
        Arc::clone(&self.summaries)
    }

    /// Clears module artifacts, definition objects, and logical summaries,
    /// then detaches every template shard from this cache.
    ///
    /// The cache is append-only — nothing is ever evicted — so it grows with
    /// every distinct config, signature, and block graph it sees.
    ///
    /// Existing contexts retain their immutable template pools and padding
    /// caches, so clearing never invalidates an in-flight compilation. Their
    /// object maps are cleared before the shards are detached.
    pub fn clear(&self) {
        let mut shards = lock(&self.shards);
        for shard in shards.values() {
            shard.clear_objects();
        }
        shards.clear();
        self.summaries.clear();
    }
}

#[derive(Debug, Default)]
pub(crate) struct ModuleSummaryCache {
    // ponytail: keep collision-free text keys; hash only if key memory profiles hot.
    entries: Mutex<crate::FxMap<(String, ModuleCertificationLimits), Arc<ModuleSummary>>>,
    definitions:
        Mutex<crate::FxMap<(String, ModuleCertificationLimits), Arc<PreparedModuleRelations>>>,
}

impl ModuleSummaryCache {
    pub(crate) fn get_or_summarize(
        &self,
        program: &BlockGraph,
        limits: ModuleCertificationLimits,
        jobs: Option<NonZeroUsize>,
    ) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
        program.validate_for_preparation(limits)?;
        let key = (program.to_blog_text(), limits);
        if let Some(summary) = lock(&self.entries).get(&key) {
            return Ok(Arc::clone(summary));
        }
        let jobs = jobs.unwrap_or_else(bloq_graph::default_module_jobs);
        let (prepared, summary) =
            program.prepare_module_summary_with_cache(limits, jobs, |key| {
                self.definition(key, limits)
            })?;
        self.retain_definitions(&prepared, limits);
        Ok(Arc::clone(
            lock(&self.entries).entry(key).or_insert(summary),
        ))
    }

    /// Prepares source that already passed `validate_with_limits` under `limits`.
    pub(crate) fn prepare_validated(
        &self,
        program: &BlockGraph,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
    ) -> Result<Arc<PreparedModuleRelations>, ModuleCertificationError> {
        let prepared = program
            .prepare_validated_module_relations(limits, jobs, |key| self.definition(key, limits))?;
        self.retain_definitions(&prepared, limits);
        Ok(prepared)
    }

    fn definition(
        &self,
        key: &str,
        limits: ModuleCertificationLimits,
    ) -> Option<Arc<PreparedModuleRelations>> {
        lock(&self.definitions)
            .get(&(key.to_owned(), limits))
            .cloned()
    }

    fn retain_definitions(
        &self,
        prepared: &Arc<PreparedModuleRelations>,
        limits: ModuleCertificationLimits,
    ) {
        let mut definitions = lock(&self.definitions);
        for definition in prepared.definitions() {
            // The entry is never instantiated. Finished objects already cache
            // its output, so retain only reusable child geometry and relations.
            if definition.name() != BlockGraph::ENTRY_MODULE {
                let key = (definition.implementation_key().to_owned(), limits);
                definitions.entry(key).or_insert(definition);
            }
        }
    }

    fn clear(&self) {
        lock(&self.entries).clear();
        lock(&self.definitions).clear();
    }
}

/// Templates and artifacts for one [`CompileConfig`].
///
/// The signature map and pool share one mutex: a published reference must resolve
/// every template it names. Compilation happens outside that lock. Padding has
/// its own cache because it is prepared in a separate pass.
#[derive(Debug, Default)]
pub(crate) struct TemplateCacheShard {
    templates: Mutex<ShardTemplates>,
    pub(crate) padding: crate::padding::PaddingTemplateCache,
    module_objects: ObjectCache<(bool, String), crate::CompileArtifacts>,
    definition_objects: ObjectCache<
        crate::compile::DefinitionObjectCacheKey,
        crate::compile::PreparedDefinitionObject,
    >,
}

/// Append-only immutable objects. Racing inserts retain the first object.
struct ObjectCache<K, V> {
    entries: Mutex<crate::FxMap<K, Arc<V>>>,
}

// Deriving Default would needlessly require K: Default and V: Default.
impl<K, V> Default for ObjectCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
        }
    }
}

impl<K: std::fmt::Debug, V> std::fmt::Debug for ObjectCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectCache")
            .field("entries", &lock(&self.entries).len())
            .finish()
    }
}

impl<K: Eq + std::hash::Hash, V> ObjectCache<K, V> {
    fn get<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + std::hash::Hash + ?Sized,
    {
        lock(&self.entries).get(key).cloned()
    }

    fn insert(&self, key: K, object: Arc<V>) -> Arc<V> {
        Arc::clone(lock(&self.entries).entry(key).or_insert(object))
    }

    fn clear(&self) {
        lock(&self.entries).clear();
    }
}

impl TemplateCacheShard {
    fn clear_objects(&self) {
        self.module_objects.clear();
        self.definition_objects.clear();
    }

    pub(crate) fn module_object(
        &self,
        key: &(bool, String),
    ) -> Option<Arc<crate::CompileArtifacts>> {
        self.module_objects.get(key)
    }

    pub(crate) fn insert_module_object(
        &self,
        key: (bool, String),
        object: Arc<crate::CompileArtifacts>,
    ) -> Arc<crate::CompileArtifacts> {
        self.module_objects.insert(key, object)
    }

    pub(crate) fn insert_definition_object(
        &self,
        key: crate::compile::DefinitionObjectCacheKey,
        object: Arc<crate::compile::PreparedDefinitionObject>,
    ) -> Arc<crate::compile::PreparedDefinitionObject> {
        self.definition_objects.insert(key, object)
    }

    pub(crate) fn definition_object(
        &self,
        key: &crate::compile::DefinitionObjectCacheKey,
    ) -> Option<Arc<crate::compile::PreparedDefinitionObject>> {
        self.definition_objects.get(key)
    }
}

#[derive(Debug, Default)]
struct ShardTemplates {
    sig_to_template: crate::FxMap<Signature, TemplateRef>,
    pool: LoweringTemplatePool,
}

/// Freshly compiled templates on their way into the pool: the shape of a
/// [`TemplateRef`] before pool ids exist. Compilation produces one of these
/// outside the lock; the shard assigns ids when it inserts.
pub(crate) enum CompiledTemplates {
    Fixed(Arc<LoweringTemplate>),
    Selective {
        when_true: Arc<LoweringTemplate>,
        when_false: Arc<LoweringTemplate>,
    },
    T {
        cultivation: Arc<LoweringTemplate>,
        escape: Arc<LoweringTemplate>,
    },
}

impl TemplateCacheShard {
    /// The reference for `key`, compiling it with `compile` on a miss.
    ///
    /// Compile misses outside the lock. Racing inserts adopt the first reference,
    /// preserving pool ids for existing readers.
    pub(crate) fn get_or_compile(
        &self,
        key: Signature,
        compile: impl FnOnce() -> Result<CompiledTemplates, CompileError>,
    ) -> Result<TemplateRef, CompileError> {
        if let Some(&template) = lock(&self.templates).sig_to_template.get(&key) {
            return Ok(template);
        }
        let compiled = compile()?;
        let mut templates = lock(&self.templates);
        if let Some(&template) = templates.sig_to_template.get(&key) {
            return Ok(template);
        }
        let pool = &mut templates.pool;
        let template = match compiled {
            CompiledTemplates::Fixed(template) => TemplateRef::Fixed(pool.insert(template)),
            CompiledTemplates::Selective {
                when_true,
                when_false,
            } => TemplateRef::Selective {
                when_true: pool.insert(when_true),
                when_false: pool.insert(when_false),
            },
            CompiledTemplates::T {
                cultivation,
                escape,
            } => TemplateRef::T {
                cultivation: pool.insert(cultivation),
                escape: pool.insert(escape),
            },
        };
        templates.sig_to_template.insert(key, template);
        Ok(template)
    }

    /// [`get_or_compile`](Self::get_or_compile) for the keys only ever inserted
    /// as a single fixed template: temporal realignments and spatial Hadamard
    /// walls.
    pub(crate) fn get_or_compile_fixed(
        &self,
        key: Signature,
        compile: impl FnOnce() -> Result<Arc<LoweringTemplate>, CompileError>,
    ) -> Result<LoweringTemplateId, CompileError> {
        match self.get_or_compile(key, || compile().map(CompiledTemplates::Fixed))? {
            TemplateRef::Fixed(id) => Ok(id),
            TemplateRef::Selective { .. } | TemplateRef::T { .. } => {
                unreachable!("realignment and wall keys are only ever inserted as fixed templates")
            }
        }
    }

    /// Snapshot the append-only pool without holding its lock during lowering.
    ///
    /// Take the snapshot after this compilation's last insertion so every held
    /// id resolves. Later insertions cannot invalidate those entries.
    pub(crate) fn pool_snapshot(&self) -> LoweringTemplatePool {
        lock(&self.templates).pool.clone()
    }

    #[cfg(test)]
    pub(crate) fn pool_len(&self) -> usize {
        lock(&self.templates).pool.len()
    }
}

/// Lock `mutex`, recovering from poisoning.
///
/// Keys are inserted last, so an interrupted insertion may leave an unreferenced
/// template but never a key naming a missing template.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileContext;

    #[test]
    fn preparation_retains_children_without_retaining_expanded_roots() {
        let cache = ModuleSummaryCache::default();
        let program = bloq_graph::GalleryItem::PhaseGradientK4.build();
        let limits = ModuleCertificationLimits::DEFAULT;
        let prepared = cache
            .prepare_validated(&program, limits, NonZeroUsize::MIN)
            .unwrap();
        let child = prepared
            .definitions()
            .into_iter()
            .find(|definition| definition.name() != BlockGraph::ENTRY_MODULE)
            .expect("the fixture contains reusable definitions");
        let name = child.name().to_owned();
        let root = Arc::downgrade(&prepared);
        let weak_child = Arc::downgrade(&child);
        drop(child);
        drop(prepared);
        assert!(root.upgrade().is_none());
        let retained = weak_child.upgrade().expect("child remains reusable");
        let next = cache
            .prepare_validated(&program, limits, NonZeroUsize::MIN)
            .unwrap();
        let reused = next
            .definitions()
            .into_iter()
            .find(|definition| definition.name() == name)
            .unwrap();
        assert!(Arc::ptr_eq(&retained, &reused));
        drop((next, retained, reused));
        cache.clear();
        assert!(weak_child.upgrade().is_none());
    }

    #[test]
    fn procedural_graph_summary_infers_open_port_interface() {
        use bloq_graph::{Block, BlockKind, CubeKind, Direction, Pipe};
        use glam::IVec3;
        let mut graph = BlockGraph::new();
        for (position, kind) in [
            (-IVec3::Z, BlockKind::Port),
            (IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)),
            (IVec3::Z, BlockKind::Port),
        ] {
            graph.add_block(Block::new(position, kind));
        }
        graph.add_pipe(Pipe::new(-IVec3::Z, Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        let inferred = graph.clone().with_inferred_interface().unwrap();
        let context = CompileContext::new(CompileConfig::default());
        let limits = ModuleCertificationLimits::DEFAULT;
        let implicit_summary = context.summarize(&graph, limits).unwrap();
        assert_eq!(implicit_summary.quantum_ports().len(), 2);
        let explicit_summary = context.summarize(&inferred, limits).unwrap();
        assert!(Arc::ptr_eq(&implicit_summary, &explicit_summary));
    }

    #[test]
    fn module_summaries_are_shared_across_configs_and_cleared() {
        let program = BlockGraph::from_text(
            r#"BLOG 1.0
module main {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, -1] <q_in>
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] <q_out>
  [0, 0, -1] -> +Z
  [0, 0, 0] -> +Z
}
"#,
        )
        .unwrap();
        let cache = SharedCompileCache::new();
        let first = CompileContext::with_shared_cache(CompileConfig::new(3), &cache);
        let second = CompileContext::with_shared_cache(CompileConfig::new(5), &cache);
        let limits = bloq_graph::ModuleCertificationLimits::UNLIMITED;
        let a = first
            .summarize_with_jobs(&program, limits, NonZeroUsize::new(2).unwrap())
            .unwrap();
        let b = second.summarize(&program, limits).unwrap();
        assert!(Arc::ptr_eq(&a, &b));

        cache.clear();
        let c = second.summarize(&program, limits).unwrap();
        assert!(!Arc::ptr_eq(&a, &c));
    }
}
