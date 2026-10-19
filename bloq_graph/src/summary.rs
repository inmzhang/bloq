//! Reusable module certificates, summaries, and composed correlation supports.

mod proxy;
pub(crate) mod relations;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};

use glam::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::program::{projection_limit, qualified_name, resource_limit};
use crate::zx::{FlowWitness, ProjectionLimits, ZXGraph};
use crate::{
    Basis, Block, BlockGraph, BlockKind, Direction, InstancePort, ModuleCertificationError,
    ModuleCertificationLimits, ModuleError, ModuleInstance, ModuleOrientation, Pauli, PauliString,
    PhasedPauliString, Pipe, QuantumConnection, QuantumPort, StabilizerGenerators, UDirection,
    checked_add_position,
};

/// Geometry and basis data recorded when a module Port is removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModulePortSignature {
    /// Direction from the removed port toward its interior neighbor.
    pub direction: Direction,
    /// Whether the removed port seam carries a Hadamard basis change.
    pub hadamard: bool,
    /// Interior block's boundary basis on each lattice axis.
    pub interior_bases: [Option<Basis>; 3],
}

impl ModulePortSignature {
    /// Returns this port geometry in `orientation`.
    pub(crate) fn with_orientation(self, orientation: ModuleOrientation) -> Self {
        Self {
            direction: orientation.rotate_direction(self.direction),
            hadamard: self.hadamard,
            interior_bases: orientation.rotate_axis_values(self.interior_bases),
        }
    }
}

/// Definition-owned factored correlation relations shared by production and public summaries.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedModuleRelations {
    name: String,
    implementation_key: String,
    limits: ModuleCertificationLimits,
    relation: Arc<relations::ModuleRelations>,
    relations: OnceLock<HashMap<String, Arc<relations::ModuleRelations>>>,
    interface_certified: OnceLock<()>,
    dependencies: HashMap<String, Arc<PreparedModuleRelations>>,
    definition: BlockGraph,
    linked: crate::LinkedModuleDefinition,
}

impl PreparedModuleRelations {
    /// Compose a production correlation space from the same definition relations
    /// used to construct this summary's signed presentation.
    ///
    /// # Errors
    ///
    /// Returns an error if geometry, Boolean work, or configured limits fail.
    #[doc(hidden)]
    pub fn compose(
        &self,
        topology: crate::GuardedTopology,
        sites: &HashMap<IVec3, crate::MaterializedModuleSite>,
    ) -> Result<crate::GuardedSurfaceSpace, crate::BlockGraphError> {
        crate::GuardedSurfaceSpace::with_relations(topology, sites, self.limits, self.relations())
    }

    /// Checks that `graph`, an orientation of this definition, closes its
    /// readouts before outputs (C0) using this definition's relations.
    ///
    /// # Errors
    ///
    /// Returns an error if composition, readout planning, or a limit fails.
    #[doc(hidden)]
    pub fn certify_interface(
        &self,
        graph: &BlockGraph,
        sites: &HashMap<IVec3, crate::MaterializedModuleSite>,
    ) -> Result<(), ModuleCertificationError> {
        crate::GuardedTopology::new(graph, self.limits)
            .and_then(|topology| self.compose(topology, sites))
            .and_then(crate::GuardedSurfaceSpace::plan_readouts)
            .map(drop)
            .map_err(|source| certification_error(&self.name, source))
    }

    fn relations(&self) -> &HashMap<String, Arc<relations::ModuleRelations>> {
        self.relations.get_or_init(|| self.collect_relations())
    }

    fn collect_relations(&self) -> HashMap<String, Arc<relations::ModuleRelations>> {
        let mut result = HashMap::new();
        let mut pending = vec![(String::new(), self)];
        while let Some((path, summary)) = pending.pop() {
            // The entry module is never instantiated; its seams belong to
            // global composition rather than the reusable definition inventory.
            if summary.name != BlockGraph::ENTRY_MODULE {
                result.insert(path.clone(), Arc::clone(&summary.relation));
            }
            pending.extend(summary.definition.instances.iter().map(|instance| {
                (
                    qualified_name(&path, &instance.name),
                    summary.dependencies[&instance.definition].as_ref(),
                )
            }));
        }
        result
    }

    /// Counts seam equations eliminated and reused across the definition tree.
    #[doc(hidden)]
    pub fn relation_work(&self) -> (usize, usize) {
        let mut seen = FxHashSet::default();
        let mut pending = vec![self];
        let mut total = (0, 0);
        while let Some(summary) = pending.pop() {
            if seen.insert(Arc::as_ptr(&summary.relation)) {
                pending.extend(summary.dependencies.values().map(Arc::as_ref));
                if summary.name != BlockGraph::ENTRY_MODULE {
                    let work = summary.relation.work();
                    total.0 += work.0;
                    total.1 += work.1;
                }
            }
        }
        total
    }

    /// Linked geometry in this definition's authored coordinates, before a
    /// parent consumes its public ports. Shared by planning and physical preparation.
    #[doc(hidden)]
    pub fn linked(&self) -> &crate::LinkedModuleDefinition {
        &self.linked
    }

    /// Returns the summarized module definition name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Exact implementation key used when preparing this definition.
    #[doc(hidden)]
    pub fn implementation_key(&self) -> &str {
        &self.implementation_key
    }

    /// Returns this summary and each reachable definition exactly once.
    #[doc(hidden)]
    pub fn definitions(self: &Arc<Self>) -> Vec<Arc<Self>> {
        let mut definitions = Vec::new();
        let mut seen = FxHashSet::default();
        let mut pending = vec![Arc::clone(self)];
        while let Some(summary) = pending.pop() {
            if seen.insert(summary.name.clone()) {
                pending.extend(summary.dependencies.values().cloned());
                definitions.push(summary);
            }
        }
        definitions
    }
}

/// Canonical signed Pauli relation over one module's declared Ports.
#[derive(Debug, Clone)]
pub struct ModuleSummary {
    name: String,
    limits: ModuleCertificationLimits,
    prepared: Arc<PreparedModuleRelations>,
    dependencies: HashMap<String, Arc<ModuleSummary>>,
    ports: Vec<QuantumPort>,
    /// Public boundary rows precede the closed rows in the complete certificate.
    boundary_len: usize,
    /// Complete signed rows retain boundary-zero support and exact local witnesses.
    certificate_rows: Vec<SupportedRow>,
}

impl ModuleSummary {
    /// Compose a production correlation space from the same definition relations
    /// used to construct this summary's signed presentation.
    ///
    /// # Errors
    ///
    /// Returns an error if geometry, Boolean work, or configured limits fail.
    #[doc(hidden)]
    pub fn compose(
        &self,
        topology: crate::GuardedTopology,
        sites: &HashMap<IVec3, crate::MaterializedModuleSite>,
    ) -> Result<crate::GuardedSurfaceSpace, crate::BlockGraphError> {
        self.prepared.compose(topology, sites)
    }

    /// Counts seam equations eliminated and reused across the shared definition tree.
    #[doc(hidden)]
    pub fn relation_work(&self) -> (usize, usize) {
        self.prepared.relation_work()
    }

    /// Returns the summarized module definition name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Finds a definition summary already built as part of this root summary.
    #[doc(hidden)]
    pub fn definition(&self, name: &str) -> Option<&ModuleSummary> {
        let mut pending = vec![self];
        let mut seen = FxHashSet::default();
        while let Some(summary) = pending.pop() {
            if summary.name == name {
                return Some(summary);
            }
            if seen.insert(summary.name.as_str()) {
                pending.extend(summary.dependencies.values().map(Arc::as_ref));
            }
        }
        None
    }

    /// Returns public quantum ports in interface declaration order.
    pub fn quantum_ports(&self) -> &[QuantumPort] {
        &self.ports
    }

    /// Iterates over signed public boundary relations.
    pub fn boundary_rows(&self) -> impl ExactSizeIterator<Item = &PhasedPauliString> {
        self.certificate_rows[..self.boundary_len]
            .iter()
            .map(|row| &row.signed)
    }

    /// Materialize and present the complete composed certificate without
    /// regenerating the linked graph's external stabilizer table. Rejects
    /// named measurements without output-safe branch representatives (C0).
    #[doc(hidden)]
    pub fn materialize_stabilizers(
        &self,
        zx: &ZXGraph,
        offset: IVec3,
    ) -> Result<StabilizerGenerators, ModuleCertificationError> {
        let external_basis = self
            .materialize_certificate_rows(zx, |position| checked_add_position(position, offset))?;
        zx.stabilizers_from_composed_basis(&external_basis, self.limits)
            .map_err(crate::RuntimeBasisError::from)
            .and_then(|stabilizers| {
                // C0 / LIM-027: leaf certification alone does not establish
                // closure after composition. Check the presented readouts,
                // including selective fills, before using them for decoding.
                stabilizers.validate_measurements_close_before_outputs_with_limits(self.limits)?;
                Ok(stabilizers)
            })
            .map_err(|source| self.runtime_error(source))
    }

    fn materialize_certificate_rows(
        &self,
        zx: &ZXGraph,
        transform: impl Fn(IVec3) -> Result<IVec3, crate::BlockGraphError>,
    ) -> Result<Vec<PhasedPauliString>, ModuleCertificationError> {
        // PauliString holds two BitVecs, each rounded to a 512-bit block.
        let words = self
            .certificate_rows
            .len()
            .checked_mul(zx.total_ids().div_ceil(512))
            .and_then(|blocks| blocks.checked_mul(16));
        if words.is_none_or(|words| words > self.limits.max_matrix_words) {
            return Err(resource_limit(
                &self.name,
                "dense matrix words",
                words.unwrap_or(usize::MAX),
                self.limits.max_matrix_words,
            ));
        }
        self.certificate_rows
            .iter()
            .map(|row| {
                let paulis = materialize_summary_row(
                    &row.signed.paulis,
                    &row.support,
                    &self.ports,
                    zx,
                    &transform,
                    &self.name,
                )?;
                let witness = row
                    .witness
                    .transformed(ModuleOrientation::IDENTITY, &transform)
                    .map_err(|source| ModuleCertificationError::Graph {
                        module: self.name.clone(),
                        source,
                    })?;
                let witnessed =
                    zx.materialize_flow_witness(&witness, &paulis)
                        .ok_or_else(|| {
                            invalid(
                                &self.name,
                                "composed certificate witness does not fit the linked graph",
                            )
                        })?;
                if witnessed.paulis != paulis {
                    return Err(invalid(
                        &self.name,
                        "composed certificate witness does not reproduce its positioned support",
                    ));
                }
                Ok(witnessed)
            })
            .collect()
    }

    fn runtime_error(&self, source: crate::RuntimeBasisError) -> ModuleCertificationError {
        match source {
            crate::RuntimeBasisError::Stabilizer(crate::StabilizerError::ResourceLimited {
                phase,
                observed,
                limit,
            }) => resource_limit(&self.name, phase, observed, limit),
            source => ModuleCertificationError::Runtime {
                module: self.name.clone(),
                source,
            },
        }
    }
}

fn materialize_summary_row(
    boundary: &PauliString,
    support: &PositionedSupport,
    ports: &[QuantumPort],
    zx: &ZXGraph,
    transform: impl Fn(IVec3) -> Result<IVec3, crate::BlockGraphError>,
    module: &str,
) -> Result<PauliString, ModuleCertificationError> {
    let mut support =
        support
            .transformed(&transform)
            .map_err(|source| ModuleCertificationError::Graph {
                module: module.to_string(),
                source,
            })?;
    for (column, port) in ports.iter().enumerate() {
        let pauli = boundary.get(column);
        if pauli == Pauli::I {
            continue;
        }
        let position =
            transform(port.position).map_err(|source| ModuleCertificationError::Graph {
                module: module.to_string(),
                source,
            })?;
        if let Some(existing) = support.nodes.insert(position, pauli)
            && existing != pauli
        {
            return Err(invalid(
                module,
                &format!(
                    "certificate support repeats public Port {position}: interior {existing}, boundary {pauli}"
                ),
            ));
        }
    }
    support.materialize(zx).ok_or_else(|| {
        invalid(
            module,
            "composed certificate support does not fit the linked graph",
        )
    })
}

#[derive(Clone)]
struct BindSeam {
    block: IVec3,
    signature: ModulePortSignature,
}

#[derive(Debug, Clone)]
struct SupportedRow {
    signed: PhasedPauliString,
    support: PositionedSupport,
    witness: FlowWitness,
}

/// Sparse full-surface support. Keeping positions rather than a hierarchy-wide
/// dense row makes instance transforms O(surface weight), and connected Port
/// support can be removed before same-position parent blocks are merged.
#[derive(Debug, Clone, Default)]
struct PositionedSupport {
    nodes: FxHashMap<IVec3, Pauli>,
    /// Directed edge support in the first endpoint's Pauli frame.
    edges: FxHashMap<(IVec3, IVec3), Pauli>,
}

impl PositionedSupport {
    fn from_paulis(mut paulis: PauliString, zx: &ZXGraph) -> Self {
        // Cross centers are ORs of incident support, not independent GF(2) bits.
        zx.clear_cross_centers(std::slice::from_mut(&mut paulis));
        let mut support = Self::default();
        for (column, pauli) in paulis.iter_support() {
            if column < zx.nodes().len() {
                let node = zx.nodes()[column];
                if !node.kind.is_port() {
                    support.nodes.insert(node.pos, pauli);
                }
            } else {
                let edge = zx.edges()[column - zx.nodes().len()];
                if edge.n1 <= edge.n2 {
                    debug_assert_eq!(
                        pauli,
                        paulis.get(zx.edge_id(edge.n2, edge.n1).expect("paired edge"))
                    );
                    support
                        .edges
                        .insert((zx.nodes()[edge.n1].pos, zx.nodes()[edge.n2].pos), pauli);
                }
            }
        }
        support
    }

    fn transformed(
        &self,
        transform: impl Fn(IVec3) -> Result<IVec3, crate::BlockGraphError>,
    ) -> Result<Self, crate::BlockGraphError> {
        Ok(Self {
            nodes: self
                .nodes
                .iter()
                .map(|(&position, &pauli)| Ok((transform(position)?, pauli)))
                .collect::<Result<_, crate::BlockGraphError>>()?,
            edges: self
                .edges
                .iter()
                .map(|(&(left, right), &pauli)| Ok(((transform(left)?, transform(right)?), pauli)))
                .collect::<Result<_, crate::BlockGraphError>>()?,
        })
    }

    fn materialize(&self, zx: &ZXGraph) -> Option<PauliString> {
        let mut row = PauliString::new(zx.total_ids());
        for (&position, &pauli) in &self.nodes {
            row.set(zx.node_at(position)?.id, pauli);
        }
        for (&(source, target), &pauli_at_source) in &self.edges {
            let source_id = zx.node_at(source)?.id;
            let target_id = zx.node_at(target)?.id;
            let forward = zx.edge_id(source_id, target_id)?;
            let reverse = zx.edge_id(target_id, source_id)?;
            let edge = &zx.edges()[forward - zx.nodes().len()];
            let mut stored = pauli_at_source;
            if source_id > target_id && edge.hadamard {
                stored = stored.flip();
            }
            row.set(forward, stored);
            row.set(reverse, stored);
        }
        // A spatial rotation can recolor a degree-two identity spider, changing
        // which component was stored as raw node support. Recover its native
        // component from the broadcast on all incident arms. Isolated nodes
        // retain their explicit raw support.
        for node in zx
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, crate::NodeKind::X | crate::NodeKind::Z))
        {
            let Some(neighbors) = zx
                .neighbors(node.id)
                .filter(|neighbors| !neighbors.is_empty())
            else {
                continue;
            };
            let native = node.kind.cross_pauli().flip();
            let broadcast = neighbors.iter().all(|&neighbor| {
                let edge = zx.edge_between(node.id, neighbor).expect("incident edge");
                let axis = if edge.hadamard && node.id > neighbor {
                    native.flip()
                } else {
                    native
                };
                row.get(edge.id) & axis
            });
            let current = row.get(node.id);
            if (current & native) != broadcast {
                row.set(node.id, current ^ native);
            }
        }
        // Witness replay owns the phase, including changes of endpoint frame.
        zx.clear_cross_centers(std::slice::from_mut(&mut row));
        Some(row)
    }
}

#[derive(Clone)]
struct ModuleGeometry {
    ports: Vec<QuantumPort>,
    signatures: Vec<ModulePortSignature>,
}

#[derive(Clone, Copy)]
struct PortGeometry<'a> {
    instance: &'a ModuleInstance,
    port: &'a QuantumPort,
    signature: ModulePortSignature,
}

impl BlockGraph {
    /// Builds the root's signed presentation from reusable definition relations.
    /// Native concurrency defaults to the available cores capped at four.
    ///
    /// # Errors
    ///
    /// Returns an error if validation, certification, or a resource limit fails.
    pub fn summarize_root(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
        self.summarize_root_with_jobs(limits, default_module_jobs())
    }

    /// Builds the root summary using at most `jobs` concurrent module workers.
    ///
    /// Definitions at the same dependency depth run in parallel on native
    /// targets. WebAssembly remains serial.
    ///
    /// # Errors
    ///
    /// Returns an error if validation, certification, or a resource limit fails.
    pub fn summarize_root_with_jobs(
        &self,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
    ) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
        self.validate_for_preparation(limits)?;
        self.prepare_module_summary_with_cache(limits, jobs, |_| None)
            .map(|(_, summary)| summary)
    }

    /// Prepare reusable factored relations without constructing dense signed presentations.
    /// Source is validated before `cached` is consulted by exact implementation key;
    /// reused definitions must use the same limits. Lookups may run concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error if validation, child certification, or a resource limit fails.
    #[doc(hidden)]
    pub fn prepare_module_relations_with_cache(
        &self,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
        cached: impl Fn(&str) -> Option<Arc<PreparedModuleRelations>> + Sync,
    ) -> Result<Arc<PreparedModuleRelations>, ModuleCertificationError> {
        self.validate_for_preparation(limits)?;
        self.prepare_validated_module_relations(limits, jobs, cached)
    }

    /// [`Self::prepare_module_relations_with_cache`] for source that already passed
    /// [`Self::validate_with_limits`] under the same limits.
    ///
    /// # Errors
    ///
    /// Returns an error if child certification or a resource limit fails.
    #[doc(hidden)]
    pub fn prepare_validated_module_relations(
        &self,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
        cached: impl Fn(&str) -> Option<Arc<PreparedModuleRelations>> + Sync,
    ) -> Result<Arc<PreparedModuleRelations>, ModuleCertificationError> {
        self.prepare_relations(limits, jobs, &cached, true)
    }

    /// Prepare relations and their signed presentation, reusing each composed
    /// space for child-interface certification. Source must already have
    /// passed [`Self::validate_for_preparation`] under the same limits.
    ///
    /// # Errors
    ///
    /// Returns an error if certification or presentation fails.
    #[doc(hidden)]
    pub fn prepare_module_summary_with_cache(
        &self,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
        cached: impl Fn(&str) -> Option<Arc<PreparedModuleRelations>> + Sync,
    ) -> Result<(Arc<PreparedModuleRelations>, Arc<ModuleSummary>), ModuleCertificationError> {
        let prepared = self.prepare_relations(limits, jobs, &cached, false)?;
        let summary = summarize_prepared(&prepared, jobs)?;
        Ok((prepared, summary))
    }

    /// Validates source, or a plain graph's local body, before preparation.
    /// Malformed source must not reach cache keys or preparation.
    ///
    /// # Errors
    ///
    /// Returns an error if validation or a resource limit fails.
    #[doc(hidden)]
    pub fn validate_for_preparation(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), ModuleCertificationError> {
        if self.has_module_structure() {
            self.validate_with_limits(limits)
                .map_err(|error| self.module_source_error(error))
        } else {
            self.validate_resource_limits(limits)
                .and_then(|()| self.validate_source())
                .map_err(|source| ModuleCertificationError::Graph {
                    module: self.name.clone(),
                    source,
                })
        }
    }

    fn module_source_error(&self, error: ModuleError) -> ModuleCertificationError {
        ModuleCertificationError::Graph {
            module: self.name.clone(),
            source: crate::BlockGraphError::ModuleSource(Arc::new(error)),
        }
    }

    fn prepare_relations(
        &self,
        limits: ModuleCertificationLimits,
        jobs: NonZeroUsize,
        cached: &(impl Fn(&str) -> Option<Arc<PreparedModuleRelations>> + Sync),
        certify: bool,
    ) -> Result<Arc<PreparedModuleRelations>, ModuleCertificationError> {
        let inferred;
        let program = if self.has_module_structure() {
            self
        } else {
            inferred = self
                .clone_local_definition()
                .with_inferred_interface()
                .map_err(|error| self.module_source_error(error))?;
            inferred
                .validate_with_limits(limits)
                .map_err(|error| self.module_source_error(error))?;
            &inferred
        };
        prepare_validated_program(program, limits, jobs, cached, certify)
    }

    /// Physical endpoints of direct child-to-child pipes in declaration order.
    ///
    /// # Panics
    ///
    /// Panics if validated module geometry is internally inconsistent.
    pub fn direct_pipe_endpoints(&self, module: &str) -> Option<Vec<(IVec3, IVec3)>> {
        let definition = self.module(module)?;
        let mut memo = HashMap::new();
        module_geometry(self, module, &mut memo).expect("validated module geometry remains valid");
        Some(direct_pipe_endpoints(definition, &memo))
    }

    /// Physical direct-pipe endpoints for every module, sharing geometry work.
    ///
    /// # Panics
    ///
    /// Panics if validated module geometry is internally inconsistent.
    pub fn direct_pipe_endpoints_by_module(&self) -> HashMap<String, Vec<(IVec3, IVec3)>> {
        let mut memo = HashMap::new();
        for module in self.modules() {
            module_geometry(self, &module.name, &mut memo)
                .expect("validated module geometry remains valid");
        }
        self.modules()
            .map(|module| (module.name.clone(), direct_pipe_endpoints(module, &memo)))
            .collect()
    }
}

fn direct_pipe_endpoints(
    definition: &BlockGraph,
    geometry: &HashMap<String, ModuleGeometry>,
) -> Vec<(IVec3, IVec3)> {
    let children = definition
        .instances
        .iter()
        .map(|instance| {
            (
                instance.name.as_str(),
                (instance, geometry[&instance.definition].clone()),
            )
        })
        .collect::<HashMap<_, _>>();
    let position = |endpoint| {
        let port = validated_port(&children, endpoint);
        exposed_position(port.instance, port.port, port.signature)
            .expect("validated direct endpoint position fits")
    };
    definition
        .quantum_connections
        .iter()
        .filter_map(|connection| match connection {
            QuantumConnection::Pipe { output, input, .. } => {
                Some((position(output), position(input)))
            }
            _ => None,
        })
        .collect()
}

impl BlockGraph {
    /// Checks the whole source hierarchy's conservative expansion budgets.
    ///
    /// # Errors
    ///
    /// Returns typed declaration, cycle, graph, or resource-limit errors.
    pub fn validate_resource_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), crate::BlockGraphError> {
        if self.has_module_structure() {
            return self
                .validate_hierarchy_resource_limits(limits)
                .map_err(|error| crate::BlockGraphError::ModuleSource(Arc::new(error)));
        }
        self.validate_local_resource_limits(limits)
    }

    /// Checks authored geometry size before cloning or enumerating occupied cells.
    /// Both branch arms count; cube footprints use their cached height in cells.
    /// This checks `max_expanded_blocks` and `max_occupied_cells` only.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::StabilizerError::ResourceLimited`] wrapped in
    /// [`crate::BlockGraphError::Stabilizer`] when a count exceeds its limit or
    /// cannot fit in `usize`.
    pub fn validate_local_resource_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), crate::BlockGraphError> {
        source_expansion_size(self, limits)
            .map(|_| ())
            .map_err(Into::into)
    }
}

fn add_expansion_size(
    size: &mut [usize; 3],
    amount: [usize; 3],
    limits: ModuleCertificationLimits,
) -> Result<(), crate::StabilizerError> {
    for ((current, amount), (phase, limit)) in size.iter_mut().zip(amount).zip([
        ("expanded blocks", limits.max_expanded_blocks),
        ("occupied footprint cells", limits.max_occupied_cells),
        ("expanded module instances", limits.max_expanded_instances),
    ]) {
        let next = current.checked_add(amount);
        if next.is_none_or(|next| next > limit) {
            return Err(crate::StabilizerError::ResourceLimited {
                phase,
                observed: next.unwrap_or(usize::MAX),
                limit,
            });
        }
        *current = next.expect("checked size fits");
    }
    Ok(())
}

fn source_expansion_size(
    source: &BlockGraph,
    limits: ModuleCertificationLimits,
) -> Result<[usize; 3], crate::StabilizerError> {
    let mut size = [0usize; 3];
    // The shown arm already lives in the body. Count its authored blocks only
    // when they differ from that body, so even an unvalidated source is bounded.
    let authored = source.branch_definitions().iter().flat_map(|region| {
        region.arm(!region.shown_true()).blocks().chain(
            region
                .shown_arm()
                .blocks()
                .filter(|block| source.get_block(block.pos()) != Some(*block)),
        )
    });
    for block in source.blocks().chain(authored) {
        let cells = match block.kind() {
            BlockKind::Walking(kind) => Some(kind.movement_3d()),
            BlockKind::PatchRotation(kind) => Some(kind.movement_3d()),
            _ => None,
        }
        .map_or_else(
            || block.height_cells() as usize,
            |movement| {
                // Moving blocks reserve their axis-aligned start/end bounding box.
                movement
                    .to_array()
                    .into_iter()
                    .map(|delta| delta.unsigned_abs() as usize + 1)
                    .product()
            },
        );
        add_expansion_size(&mut size, [1, cells, 0], limits)?;
    }
    Ok(size)
}

/// Count each definition once, then add its cached size for each instance edge.
/// Both branch arms and Ports consumed by seams count toward the allocation bound.
pub(crate) fn preflight_program_expansion(
    program: &BlockGraph,
    limits: ModuleCertificationLimits,
) -> Result<(), ModuleError> {
    let mut sizes = HashMap::<&str, [usize; 3]>::new();
    for module in module_postorder(program) {
        let error = |source| {
            let crate::StabilizerError::ResourceLimited {
                phase,
                observed,
                limit,
            } = source
            else {
                unreachable!("size counting only reports resource limits")
            };
            program_geometry_error(
                &module.name,
                resource_limit(&module.name, phase, observed, limit),
            )
        };
        let mut size = source_expansion_size(module.local_body(), limits).map_err(error)?;
        add_expansion_size(&mut size, [0, 0, 1], limits).map_err(error)?;
        for instance in &module.instances {
            add_expansion_size(&mut size, sizes[instance.definition.as_str()], limits)
                .map_err(error)?;
        }
        sizes.insert(&module.name, size);
    }
    Ok(())
}

pub(crate) fn validate_program_geometry(program: &BlockGraph) -> Result<(), ModuleError> {
    for module in module_postorder(program)
        .into_iter()
        .filter(|module| !module.instances.is_empty())
    {
        validate_composite_geometry(program, module)
            .map_err(|error| program_geometry_error(&module.name, error))?;
    }
    let mut memo = HashMap::new();
    for module in program.modules() {
        module_geometry(program, &module.name, &mut memo)?;
    }
    Ok(())
}

fn module_postorder(program: &BlockGraph) -> Vec<&BlockGraph> {
    module_postorder_from(
        program,
        program.modules().map(|module| module.name.as_str()),
    )
}

fn module_postorder_from<'a>(
    program: &'a BlockGraph,
    roots: impl IntoIterator<Item = &'a str>,
) -> Vec<&'a BlockGraph> {
    let definitions = program
        .modules()
        .map(|module| (module.name.as_str(), module))
        .collect::<HashMap<_, _>>();
    let mut seen = FxHashSet::default();
    let mut ordered = Vec::new();
    for root in roots {
        let mut pending = vec![(root, false)];
        while let Some((name, exiting)) = pending.pop() {
            let module = definitions[name];
            if exiting {
                ordered.push(module);
            } else if seen.insert(name) {
                pending.push((name, true));
                pending.extend(
                    module
                        .instances
                        .iter()
                        .rev()
                        .map(|instance| (instance.definition.as_str(), false)),
                );
            }
        }
    }
    ordered
}

fn module_geometry(
    program: &BlockGraph,
    name: &str,
    memo: &mut HashMap<String, ModuleGeometry>,
) -> Result<ModuleGeometry, ModuleError> {
    if let Some(geometry) = memo.get(name) {
        return Ok(geometry.clone());
    }
    for module in module_postorder_from(program, [name]) {
        if memo.contains_key(&module.name) {
            continue;
        }
        let name = module.name.as_str();
        let geometry = if module.instances.is_empty() {
            let body = module.copy_local_geometry().fix_shadowed_faces();
            let signatures = module
                .interface
                .quantum_ports
                .iter()
                .map(|port| port_signature(name, &body, port.position))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| program_geometry_error(name, error))?;
            ModuleGeometry {
                ports: module.interface.quantum_ports.clone(),
                signatures,
            }
        } else {
            let mut children = HashMap::new();
            for instance in &module.instances {
                let geometry = memo[&instance.definition].clone();
                children.insert(instance.name.as_str(), (instance, geometry));
            }

            let mut binds = Vec::new();
            for connection in &module.quantum_connections {
                match connection {
                    QuantumConnection::Input {
                        block,
                        input,
                        hadamard,
                    } => {
                        binds.push(geometry_bind_seam(input, *block, *hadamard, &children));
                    }
                    QuantumConnection::Output {
                        output,
                        block,
                        hadamard,
                    } => {
                        binds.push(geometry_bind_seam(output, *block, *hadamard, &children));
                    }
                    QuantumConnection::Pipe {
                        output,
                        input,
                        hadamard,
                    } => {
                        geometry_direct_seam(output, input, *hadamard, &children, name)
                            .map_err(|error| program_geometry_error(name, error))?;
                    }
                }
            }
            validate_bind_faces(module, &binds)
                .map_err(|error| program_geometry_error(name, error))?;
            let extra_pipes = binds
                .iter()
                .map(|bind| (bind.block, bind.signature.direction.as_udirection()))
                .collect::<Vec<_>>();
            let body = module
                .copy_local_geometry()
                .fix_shadowed_faces_with(&extra_pipes);
            validate_bind_bases(module, &body, &binds)
                .map_err(|error| program_geometry_error(name, error))?;
            ModuleGeometry {
                ports: module.interface.quantum_ports.clone(),
                signatures: parent_signatures(module, &body, &binds)
                    .map_err(|error| program_geometry_error(name, error))?,
            }
        };
        memo.insert(name.to_string(), geometry);
    }
    Ok(memo[name].clone())
}

fn program_geometry_error(module: &str, error: ModuleCertificationError) -> ModuleError {
    ModuleError::InvalidGeometry {
        module: module.to_string(),
        source: error,
        span: None,
    }
}

/// Default worker count for summarizing or compiling a module hierarchy:
/// the machine's parallelism capped at 4, and always 1 on wasm.
pub fn default_module_jobs() -> NonZeroUsize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        const MAX_JOBS: NonZeroUsize = NonZeroUsize::new(4).expect("4 is nonzero");
        std::thread::available_parallelism().map_or(NonZeroUsize::MIN, |jobs| jobs.min(MAX_JOBS))
    }
    #[cfg(target_arch = "wasm32")]
    {
        NonZeroUsize::MIN
    }
}

/// Maps `f` over `items` across at most `jobs` scoped worker threads, keeping
/// the results in input order.
///
/// Chunked rather than work-stealing: the callers
/// map over module definitions, whose per-item cost is uneven but whose counts
/// are small.
///
/// wasm has no threads, so it always runs serially.
///
/// # Panics
///
/// Panics if `f` panics in a worker thread.
///
// ponytail: scoped workers are enough; add a pool only if spawn profiles hot.
pub fn map_jobs<T: Sync, R: Send>(
    items: &[T],
    jobs: NonZeroUsize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    #[cfg(target_arch = "wasm32")]
    let _ = jobs;
    #[cfg(not(target_arch = "wasm32"))]
    if items.len() > 1 && jobs.get() > 1 {
        let worker_count = jobs.get().min(items.len());
        let f = &f;
        let cancellation = crate::CancellationToken::current();
        return std::thread::scope(|scope| {
            let workers = (0..worker_count)
                .map(|worker| {
                    let start = items.len() * worker / worker_count;
                    let end = items.len() * (worker + 1) / worker_count;
                    let chunk = &items[start..end];
                    let cancellation = cancellation.clone();
                    scope.spawn(move || {
                        let run = || chunk.iter().map(f).collect::<Vec<_>>();
                        match cancellation {
                            Some(token) => token.scope(run),
                            None => run(),
                        }
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("map_jobs worker does not panic"))
                .collect()
        });
    }
    items.iter().map(f).collect()
}

fn certification_error(module: &str, source: crate::BlockGraphError) -> ModuleCertificationError {
    match source {
        crate::BlockGraphError::Stabilizer(crate::StabilizerError::ResourceLimited {
            phase,
            observed,
            limit,
        }) => resource_limit(module, phase, observed, limit),
        source => ModuleCertificationError::Graph {
            module: module.to_owned(),
            source,
        },
    }
}

fn prepare_validated_program(
    program: &BlockGraph,
    limits: ModuleCertificationLimits,
    jobs: NonZeroUsize,
    cached: &(impl Fn(&str) -> Option<Arc<PreparedModuleRelations>> + Sync),
    certify: bool,
) -> Result<Arc<PreparedModuleRelations>, ModuleCertificationError> {
    let mut depths = HashMap::new();
    let mut levels: Vec<Vec<&BlockGraph>> = Vec::new();
    for module in module_postorder_from(program, [program.root().name.as_str()]) {
        let depth = module
            .instances
            .iter()
            .map(|instance| depths[instance.definition.as_str()] + 1)
            .max()
            .unwrap_or(0);
        levels.resize_with(levels.len().max(depth + 1), Vec::new);
        levels[depth].push(module);
        depths.insert(module.name.as_str(), depth);
    }
    let keys = program.module_definition_cache_keys();
    let mut prepared = HashMap::new();
    for level in levels {
        let results = map_jobs(&level, jobs, |&module| {
            if let Some(summary) = cached(&keys[&module.name]).filter(|summary| {
                summary.limits == limits && summary.implementation_key == keys[&module.name]
            }) {
                return Ok(summary);
            }
            let graph_error = |source| certification_error(&module.name, source);
            let mut linked =
                crate::flatten_module_definition(program, module, "").map_err(graph_error)?;
            linked.graph = linked.graph.fix_shadowed_faces();
            let summary = Arc::new(PreparedModuleRelations {
                name: module.name.clone(),
                implementation_key: keys[&module.name].clone(),
                limits,
                relation: Arc::default(),
                relations: OnceLock::new(),
                interface_certified: OnceLock::new(),
                dependencies: module
                    .instances
                    .iter()
                    .map(|instance| {
                        (
                            instance.definition.clone(),
                            Arc::clone(&prepared[&instance.definition]),
                        )
                    })
                    .collect(),
                definition: module.clone_local_definition(),
                linked,
            });
            // A parent consuming outputs must not conceal an invalid child interface.
            if certify && module.name != program.root().name {
                summary.certify_interface(&summary.linked.graph, &summary.linked.sites)?;
                summary.interface_certified.get_or_init(|| ());
            }
            Ok(summary)
        });
        for (module, summary) in level.iter().zip(results) {
            prepared.insert(module.name.clone(), summary?);
        }
    }
    Ok(Arc::clone(&prepared[&program.root().name]))
}

fn summarize_prepared(
    prepared: &Arc<PreparedModuleRelations>,
    jobs: NonZeroUsize,
) -> Result<Arc<ModuleSummary>, ModuleCertificationError> {
    let mut pending = vec![(Arc::clone(prepared), false)];
    let mut seen = FxHashSet::default();
    let mut depths = HashMap::new();
    let mut levels: Vec<Vec<Arc<PreparedModuleRelations>>> = Vec::new();
    while let Some((definition, exiting)) = pending.pop() {
        if exiting {
            let depth = definition
                .dependencies
                .keys()
                .map(|name| depths[name] + 1)
                .max()
                .unwrap_or(0);
            levels.resize_with(levels.len().max(depth + 1), Vec::new);
            depths.insert(definition.name.clone(), depth);
            levels[depth].push(definition);
        } else if seen.insert(definition.name.clone()) {
            pending.push((Arc::clone(&definition), true));
            pending.extend(
                definition
                    .dependencies
                    .values()
                    .cloned()
                    .map(|child| (child, false)),
            );
        }
    }
    let mut summaries = HashMap::new();
    for level in levels {
        let results = map_jobs(&level, jobs, |definition| {
            // A parent consuming outputs must not conceal an invalid child interface.
            let certify =
                definition.name != prepared.name && definition.interface_certified.get().is_none();
            summarize_definition(definition, &summaries, certify)
        });
        for (definition, summary) in level.iter().zip(results) {
            summaries.insert(definition.name.clone(), Arc::new(summary?));
        }
    }
    Ok(Arc::clone(&summaries[&prepared.name]))
}

fn summarize_definition(
    prepared: &Arc<PreparedModuleRelations>,
    summaries: &HashMap<String, Arc<ModuleSummary>>,
    certify: bool,
) -> Result<ModuleSummary, ModuleCertificationError> {
    let module = &prepared.definition;
    let definition = &prepared.linked.graph;
    let limits = prepared.limits;
    let graph_error = |source| ModuleCertificationError::Graph {
        module: module.name.clone(),
        source,
    };
    let composition_error = |source| certification_error(&module.name, source);
    let mut space = prepared
        .compose(
            crate::GuardedTopology::new(definition, limits).map_err(composition_error)?,
            &prepared.linked.sites,
        )
        .map_err(composition_error)?;
    if certify {
        // Planning consumes its space; presentation keeps only the relation rows.
        let view = space.relation_view();
        std::mem::replace(&mut space, view)
            .plan_readouts()
            .map_err(composition_error)?;
        prepared.interface_certified.get_or_init(|| ());
    }
    let targets = definition
        .branch_regions()
        .map_err(graph_error)?
        .into_iter()
        .map(|region| region.target)
        .collect::<Vec<_>>();
    let mut canonical = definition.clone_local_definition();
    canonical.replace_local_body(
        definition
            .canonical_true_branch_view()
            .map_err(graph_error)?,
    );
    let canonical_space;
    let source_space = if space
        .topology
        .diagram
        .evaluate(space.topology.domain, |_| true)
    {
        &space
    } else {
        // An unreachable all-true presentation needs a static, branchless space.
        canonical.replace_local_body(
            canonical
                .local_body()
                .project_branches_in_definition(targets.iter().map(|&target| (target, true)))
                .map_err(graph_error)?,
        );
        canonical_space = prepared
            .compose(
                crate::GuardedTopology::new(&canonical, limits).map_err(graph_error)?,
                &prepared.linked.sites,
            )
            .map_err(graph_error)?;
        &canonical_space
    };
    let mut summary = summarize_space_view(prepared, module, &canonical, source_space, limits)?;
    summary.dependencies = prepared
        .dependencies
        .keys()
        .map(|name| (name.clone(), Arc::clone(&summaries[name])))
        .collect();
    Ok(summary)
}

fn summarize_space_view(
    prepared: &Arc<PreparedModuleRelations>,
    module: &BlockGraph,
    source: &BlockGraph,
    space: &crate::GuardedSurfaceSpace,
    limits: ModuleCertificationLimits,
) -> Result<ModuleSummary, ModuleCertificationError> {
    let graph_error = |source| ModuleCertificationError::Graph {
        module: module.name.clone(),
        source,
    };
    let body = source.fix_shadowed_faces();
    let zx = ZXGraph::from_block_graph_for_analysis(&body)
        .map_err(|source| graph_error(source.into()))?;
    let values = space
        .topology
        .branch_selection_values(&FxHashSet::default())
        .map_err(|source| graph_error(source.into()))?
        .ok_or_else(|| invalid(&module.name, "presentation selects an unreachable branch"))?;
    let evaluate = |root| space.topology.diagram.evaluate(root, |index| values[index]);
    let words = space
        .rows
        .iter()
        .filter(|row| evaluate(row.active))
        .count()
        .checked_mul(zx.total_ids().div_ceil(512))
        .and_then(|blocks| blocks.checked_mul(16));
    if words.is_none_or(|words| words > limits.max_matrix_words) {
        return Err(resource_limit(
            &module.name,
            "dense matrix words",
            words.unwrap_or(usize::MAX),
            limits.max_matrix_words,
        ));
    }
    let rows = space.materialize_relation(&zx, evaluate);
    let phases = zx.external_stabilizer_row_phases(&rows);
    let rows = rows
        .into_iter()
        .zip(phases)
        .map(|(row, phase)| PhasedPauliString::new(row, phase))
        .collect();
    let columns = module
        .interface
        .quantum_ports
        .iter()
        .map(|port| zx.node_at(port.position).expect("validated module port").id)
        .collect::<Vec<_>>();
    let table = zx
        .projected_external_table_from_rows(
            &columns,
            rows,
            ProjectionLimits {
                max_frontier_width: limits.max_frontier_width,
                max_witness_nodes: limits.max_witness_nodes,
            },
        )
        .map_err(|error| projection_limit(&module.name, error))?
        .ok_or_else(|| {
            invalid(
                &module.name,
                "composed relation has invalid local tensor support",
            )
        })?;
    let mut certificate_rows = table
        .boundary_rows
        .iter()
        .map(|row| {
            let full = table.materialize(row);
            SupportedRow {
                signed: row.signed.clone(),
                support: PositionedSupport::from_paulis(full.paulis, &zx),
                witness: row.flow_witness.clone(),
            }
        })
        .collect::<Vec<_>>();
    let boundary_len = certificate_rows.len();
    certificate_rows.extend(table.closed_rows.iter().map(|row| SupportedRow {
        signed: PhasedPauliString::new(PauliString::new(columns.len()), row.signed.phase()),
        support: PositionedSupport::from_paulis(row.signed.paulis.clone(), &zx),
        witness: row.flow_witness.clone(),
    }));
    Ok(ModuleSummary {
        name: module.name.clone(),
        limits,
        prepared: Arc::clone(prepared),
        dependencies: HashMap::new(),
        ports: module.interface.quantum_ports.clone(),
        boundary_len,
        certificate_rows,
    })
}

fn validate_composite_geometry(
    program: &BlockGraph,
    module: &BlockGraph,
) -> Result<(), ModuleCertificationError> {
    let mut footprint = CompositeFootprint::default();
    let mut next_branch = 0;
    add_module_footprint(
        program,
        module,
        (ModuleOrientation::IDENTITY, IVec3::ZERO),
        "",
        true,
        &mut footprint,
        &mut next_branch,
    )
    .map_err(|source| ModuleCertificationError::Graph {
        module: module.name.clone(),
        source,
    })
}

struct FootprintOrigin {
    definition: String,
    instance_path: String,
    orientation: ModuleOrientation,
}

#[derive(Clone)]
struct FootprintBlock {
    position: IVec3,
    kind: BlockKind,
    branch: Option<usize>,
    origin: Arc<FootprintOrigin>,
    local_position: IVec3,
}

impl FootprintBlock {
    fn materialized_site(&self) -> Arc<crate::MaterializedModuleSite> {
        Arc::new(crate::MaterializedModuleSite {
            definition: self.origin.definition.clone(),
            instance_path: self.origin.instance_path.clone(),
            local_position: self.local_position,
            orientation: self.origin.orientation,
        })
    }
}

#[derive(Default)]
struct CompositeFootprint {
    occupied: HashMap<IVec3, Vec<FootprintBlock>>,
}

impl CompositeFootprint {
    fn add(
        &mut self,
        block: Block,
        branch: Option<usize>,
        origin: &Arc<FootprintOrigin>,
        local_position: IVec3,
    ) -> Result<(), crate::BlockGraphError> {
        let candidate = FootprintBlock {
            position: block.pos(),
            kind: block.kind(),
            branch,
            origin: Arc::clone(origin),
            local_position,
        };
        if let Some(other) = self
            .occupied
            .get(&block.pos())
            .into_iter()
            .flatten()
            .find(|other| {
                other.position == block.pos() && !mutually_exclusive(branch, other.branch)
            })
        {
            return Err(crate::BlockGraphError::ModuleBlockOverlap {
                position: block.pos(),
                first: other.materialized_site(),
                second: candidate.materialized_site(),
            });
        }
        let positions = block.checked_reserved_positions()?;
        for &position in &positions {
            if let Some(other) = self
                .occupied
                .get(&position)
                .into_iter()
                .flatten()
                .find(|other| {
                    !mutually_exclusive(branch, other.branch)
                        && !candidate.kind.allows_reserved_overlap(
                            candidate.position,
                            other.kind,
                            other.position,
                            position,
                        )
                })
            {
                return Err(crate::BlockGraphError::ModuleBlockOverlap {
                    position,
                    first: other.materialized_site(),
                    second: candidate.materialized_site(),
                });
            }
        }
        for position in positions {
            self.occupied
                .entry(position)
                .or_default()
                .push(candidate.clone());
        }
        Ok(())
    }
}

fn mutually_exclusive(left: Option<usize>, right: Option<usize>) -> bool {
    left.is_some() && left == right
}

fn add_module_footprint(
    program: &BlockGraph,
    definition: &BlockGraph,
    transform: (ModuleOrientation, IVec3),
    instance_path: &str,
    include_ports: bool,
    footprint: &mut CompositeFootprint,
    next_branch: &mut usize,
) -> Result<(), crate::BlockGraphError> {
    let definitions = program
        .modules()
        .map(|module| (module.name.as_str(), module))
        .collect::<HashMap<_, _>>();
    let mut pending = vec![(
        definition,
        transform,
        instance_path.to_string(),
        include_ports,
    )];
    while let Some((definition, (orientation, translation), instance_path, include_ports)) =
        pending.pop()
    {
        let origin = Arc::new(FootprintOrigin {
            definition: definition.name.clone(),
            instance_path: instance_path.to_string(),
            orientation,
        });
        definition
            .copy_local_geometry()
            .with_orientation_lenient(orientation)?;
        let regions = definition.local_body().branch_regions()?;
        let shown = regions
            .iter()
            .flat_map(|region| region.shown_arm().blocks().map(Block::pos))
            .collect::<FxHashSet<_>>();
        for block in definition
            .local_body()
            .blocks()
            .filter(|block| !shown.contains(&block.pos()))
            .filter(|block| include_ports || !block.kind().is_port())
        {
            footprint.add(
                crate::graph::oriented_block(block, orientation)?.try_with_shift(translation)?,
                None,
                &origin,
                block.pos(),
            )?;
        }
        for region in regions {
            let branch = *next_branch;
            *next_branch += 1;
            for block in region.on_false().blocks().chain(region.on_true().blocks()) {
                footprint.add(
                    crate::graph::oriented_block(block, orientation)?
                        .try_with_shift(translation)?,
                    Some(branch),
                    &origin,
                    block.pos(),
                )?;
            }
        }
        for instance in definition.instances.iter().rev() {
            let child_translation =
                orientation.try_transform_position(instance.translation, translation)?;
            let child_orientation = orientation.then(instance.rotation);
            let child_path = qualified_name(&instance_path, &instance.name);
            pending.push((
                definitions[instance.definition.as_str()],
                (child_orientation, child_translation),
                child_path,
                false,
            ));
        }
    }
    Ok(())
}

fn geometry_bind_seam(
    endpoint: &InstancePort,
    block: IVec3,
    hadamard: bool,
    children: &HashMap<&str, (&ModuleInstance, ModuleGeometry)>,
) -> BindSeam {
    let mut signature = validated_port(children, endpoint).signature;
    signature.hadamard ^= hadamard;
    BindSeam { block, signature }
}

fn geometry_direct_seam(
    output: &InstancePort,
    input: &InstancePort,
    hadamard: bool,
    children: &HashMap<&str, (&ModuleInstance, ModuleGeometry)>,
    module: &str,
) -> Result<bool, ModuleCertificationError> {
    direct_seam_hadamard_parts(
        validated_port(children, output),
        validated_port(children, input),
        hadamard,
        module,
    )
}

fn direct_seam_hadamard_parts(
    output: PortGeometry<'_>,
    input: PortGeometry<'_>,
    hadamard: bool,
    module: &str,
) -> Result<bool, ModuleCertificationError> {
    let output_pos = exposed_position(output.instance, output.port, output.signature)?;
    let input_pos = exposed_position(input.instance, input.port, input.signature)?;
    let direction = Direction::iter()
        .find(|direction| {
            checked_add_position(output_pos, direction.to_ivec3()).ok() == Some(input_pos)
        })
        .ok_or_else(|| invalid(module, "direct pipe endpoints are not adjacent"))?;
    if direction != output.signature.direction.negate() || direction != input.signature.direction {
        return Err(invalid(
            module,
            "direct pipe faces point in different directions",
        ));
    }
    if !compatible_bases(
        output.signature.interior_bases,
        input.signature.interior_bases,
        direction.as_udirection(),
        hadamard,
    ) {
        return Err(invalid(module, "direct pipe face bases are incompatible"));
    }
    Ok(output.signature.hadamard ^ hadamard ^ input.signature.hadamard)
}

fn validated_port<'a>(
    children: &'a HashMap<&str, (&ModuleInstance, ModuleGeometry)>,
    endpoint: &InstancePort,
) -> PortGeometry<'a> {
    let (instance, geometry) = children
        .get(endpoint.instance.as_str())
        .map(|(instance, geometry)| (*instance, geometry))
        .expect("program validation resolved the child instance");
    let index = geometry
        .ports
        .iter()
        .position(|port| port.name == endpoint.port)
        .expect("program validation resolved the child port");
    PortGeometry {
        instance,
        port: &geometry.ports[index],
        signature: geometry.signatures[index].with_orientation(instance.orientation()),
    }
}

fn exposed_position(
    instance: &ModuleInstance,
    port: &QuantumPort,
    signature: ModulePortSignature,
) -> Result<IVec3, ModuleCertificationError> {
    instance
        .try_transform_position(port.position)
        .and_then(|position| checked_add_position(position, signature.direction.to_ivec3()))
        .map_err(|error| ModuleCertificationError::Graph {
            module: instance.name.clone(),
            source: error,
        })
}

fn validate_bind_faces(
    module: &BlockGraph,
    binds: &[BindSeam],
) -> Result<(), ModuleCertificationError> {
    let mut used = FxHashSet::default();
    for bind in binds {
        let neighbor = checked_add_position(bind.block, bind.signature.direction.to_ivec3())
            .map_err(|source| ModuleCertificationError::Graph {
                module: module.name.clone(),
                source,
            })?;
        if module.local_body().has_pipe_between(bind.block, neighbor) {
            return Err(invalid(&module.name, "bind reuses an authored pipe face"));
        }
        if !used.insert((bind.block, bind.signature.direction)) {
            return Err(invalid(&module.name, "two binds reuse one block face"));
        }
    }
    Ok(())
}

fn validate_bind_bases(
    module: &BlockGraph,
    body: &BlockGraph,
    binds: &[BindSeam],
) -> Result<(), ModuleCertificationError> {
    for bind in binds {
        let mut pipe = Pipe::new(bind.block, bind.signature.direction);
        if bind.signature.hadamard {
            pipe = pipe.with_hadamard();
        }
        let parent_bases = body.infer_pipe_endpoint_face_bases(&pipe, bind.block);
        if !compatible_bases(
            parent_bases,
            bind.signature.interior_bases,
            bind.signature.direction.as_udirection(),
            bind.signature.hadamard,
        ) {
            return Err(invalid(&module.name, "bind face bases are incompatible"));
        }
    }
    Ok(())
}

fn parent_signatures(
    module: &BlockGraph,
    body: &BlockGraph,
    binds: &[BindSeam],
) -> Result<Vec<ModulePortSignature>, ModuleCertificationError> {
    module
        .interface
        .quantum_ports
        .iter()
        .map(|port| {
            let local = module.local_body().neighbor_positions(port.position);
            let bound = binds
                .iter()
                .filter(|bind| bind.block == port.position)
                .collect::<Vec<_>>();
            match (local.as_slice(), bound.as_slice()) {
                ([_], []) => port_signature(&module.name, body, port.position),
                ([], [bind]) => Ok(bind.signature),
                _ => Err(invalid(
                    &module.name,
                    "each exported Port must have one local or bound pipe",
                )),
            }
        })
        .collect()
}

fn port_signature(
    module: &str,
    body: &BlockGraph,
    position: IVec3,
) -> Result<ModulePortSignature, ModuleCertificationError> {
    let neighbors = body.neighbor_positions(position);
    let [neighbor] = neighbors.as_slice() else {
        return Err(invalid(module, "module Port must have one pipe"));
    };
    let pipe = body
        .get_pipe(position, *neighbor)
        .expect("neighbor position names the incident pipe");
    let direction = if pipe.src() == position {
        pipe.dir()
    } else {
        pipe.dir().negate()
    };
    Ok(ModulePortSignature {
        direction,
        hadamard: pipe.is_hadamard(),
        interior_bases: body.infer_pipe_endpoint_face_bases(pipe, *neighbor),
    })
}

fn compatible_bases(
    left: [Option<Basis>; 3],
    right: [Option<Basis>; 3],
    pipe_axis: UDirection,
    hadamard: bool,
) -> bool {
    (0..3)
        .filter(|&axis| axis != pipe_axis.index())
        .all(|axis| {
            !matches!(
                (left[axis], right[axis]),
                (Some(left), Some(right)) if (left != right) ^ hadamard
            )
        })
}

fn invalid(module: &str, message: &str) -> ModuleCertificationError {
    ModuleCertificationError::InvalidConnection {
        module: module.to_string(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::*;
    use crate::{GalleryItem, StabilizerRowKind};

    #[test]
    fn flat_source_limit_counts_cube_height_before_geometry_work() {
        let graph = crate::parse_blog_to_graph("BLOG 1.0\n0: ZXZ [0,0,0] height=4d\n").unwrap();
        let limits = ModuleCertificationLimits {
            max_expanded_blocks: 1,
            max_occupied_cells: 3,
            ..ModuleCertificationLimits::UNLIMITED
        };
        assert!(matches!(
            graph.validate_resource_limits(limits),
            Err(crate::BlockGraphError::Stabilizer(
                crate::StabilizerError::ResourceLimited {
                    phase: "occupied footprint cells",
                    observed: 4,
                    limit: 3,
                }
            ))
        ));
        graph
            .validate_resource_limits(ModuleCertificationLimits {
                max_occupied_cells: 4,
                ..limits
            })
            .unwrap();
    }

    fn identity_wrapper(program: &BlockGraph) -> BlockGraph {
        let mut child = program.root().clone();
        child.name = "Child".into();
        let mut body = BlockGraph::new();
        let quantum_connections = child
            .interface
            .quantum_ports
            .iter()
            .map(|port| {
                let role = match port.direction {
                    crate::PortDirection::Input => crate::PortRole::Input,
                    crate::PortDirection::Output => crate::PortRole::Output,
                };
                body.try_add_block(
                    Block::new(port.position, BlockKind::Port)
                        .with_port_role(role)
                        .unwrap(),
                )
                .unwrap();
                let endpoint = InstancePort {
                    instance: "child".into(),
                    port: port.name.clone(),
                };
                match port.direction {
                    crate::PortDirection::Input => QuantumConnection::Input {
                        block: port.position,
                        input: endpoint,
                        hadamard: false,
                    },
                    crate::PortDirection::Output => QuantumConnection::Output {
                        output: endpoint,
                        block: port.position,
                        hadamard: false,
                    },
                }
            })
            .collect();
        let root = BlockGraph::definition(
            "main",
            body,
            child.interface.clone(),
            vec![ModuleInstance {
                name: "child".into(),
                definition: child.name.clone(),
                rotation: crate::ModuleRotation::IDENTITY,
                translation: IVec3::ZERO,
            }],
            quantum_connections,
            Vec::new(),
        );
        BlockGraph::from_definitions(vec![child, root]).unwrap()
    }

    #[test]
    fn shallower_sibling_preserves_queued_dependency_levels() {
        let wire = "in input: data = 0\nout output: data = 2\n
            0: Port [0,0,-1] role=input\n1: XZZ [0,0,0]\n2: Port [0,0,1] role=output\n
            0 -> +Z\n1 -> +Z\n";
        let program = crate::parse_inline_graph(&format!(
            "BLOG 1.0\nmodule Wire {{\n{wire}}}\nmodule Late {{\n{wire}}}\n
            module Nested {{
                in input: data = 0\nout output: data = 2
                0: Port [0,0,-1] role=input\n2: Port [0,0,1] role=output
                wire: Wire @ [0,0,0]
                0 -> wire.input\nwire.output -> 2
            }}
            module main {{
                in input: data = 0\nout output: data = 2
                0: Port [0,0,-1] role=input\n2: Port [0,0,2] role=output
                nested: Nested @ [0,0,0]
                late: Late @ [0,0,1]
                0 -> nested.input\nnested.output -> late.input\nlate.output -> 2
            }}"
        ))
        .unwrap();
        for jobs in [1, 2] {
            let summary = program
                .summarize_root_with_jobs(
                    ModuleCertificationLimits::DEFAULT,
                    NonZeroUsize::new(jobs).unwrap(),
                )
                .unwrap();
            assert_eq!(summary.boundary_rows().len(), 2);
            assert!(summary.definition("Nested").is_some());
            assert!(summary.definition("Late").is_some());
        }
    }

    #[test]
    fn parent_readout_survives_child_composition() {
        let program = crate::parse_inline_graph(
            "BLOG 1.0\n\nmodule Wire {\n\
             in input: data = 0\nout output: data = 2\n\
             0: Port [0,0,-1]\n1: ZXZ [0,0,0]\n2: Port [0,0,1]\n\
             0 -> +Z\n1 -> +Z\n}\n\nmodule main {\n\
             in input: data = 10\nout result = m\n\
             wire: Wire @ [0,0,0]\n\
             10: Port [0,0,-1] role=input\n11: X [0,0,1]\n\
             10 -> wire.input\nwire.output -> 11\nm = measure 11\n}\n",
        )
        .unwrap();
        let summary = program
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap();
        let graph = program
            .materialize_root_graph()
            .unwrap()
            .fix_shadowed_faces();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let composed = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
        let flat = graph.analyze_actions().unwrap().1;
        assert_eq!(composed.generators.len(), flat.generators.len());
        for (composed, flat) in composed.generators.iter().zip(&flat.generators) {
            assert_eq!(composed.kind, flat.kind);
            assert_eq!(composed.stabilizer.paulis, flat.stabilizer.paulis);
            assert_eq!(composed.stabilizer.sign, flat.stabilizer.sign);
        }
    }

    #[test]
    fn classical_parent_action_keeps_independent_connectors_within_frontier_limit() {
        let program = crate::parse_inline_graph(
            "BLOG 1.0\nmodule Wire {\n\
             in input: data = 0\nout output: data = 2\n\
             0: Port [0,0,-1]\n1: ZXZ [0,0,0]\n2: Port [0,0,1]\n\
             0 -> +Z\n1 -> +Z\n}\nmodule main {\n\
             in a: data = 10\nout a_out: data = 11\n\
             in b: data = 12\nout b_out: data = 13\nin enable\nout done = flag\n\
             a: Wire @ [0,0,0]\nb: Wire @ [2,0,0]\n\
             10: Port [0,0,-1] role=input\n11: Port [0,0,1] role=output\n\
             12: Port [2,0,-1] role=input\n13: Port [2,0,1] role=output\n\
             10 -> a.input\na.output -> 11\n12 -> b.input\nb.output -> 13\n\
             flag = enable | !enable\n}\n",
        )
        .unwrap();
        let summary = program
            .summarize_root(ModuleCertificationLimits {
                max_frontier_width: 6,
                ..ModuleCertificationLimits::DEFAULT
            })
            .unwrap();
        assert_eq!(summary.quantum_ports().len(), 4);
    }

    #[test]
    fn identity_wrapper_preserves_materialized_stabilizers() {
        let program = identity_wrapper(&GalleryItem::CCZInjectedMaj.build());
        let summary = program
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        let graph = program.materialize_root_graph().unwrap();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let composed = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
        let flat = graph.analyze_actions().unwrap().1;

        assert_eq!(composed.generators.len(), flat.generators.len());
        for (composed, flat) in composed.generators.iter().zip(&flat.generators) {
            assert_eq!(composed.kind, flat.kind);
            assert_eq!(composed.stabilizer.paulis, flat.stabilizer.paulis);
            assert_eq!(composed.stabilizer.sign, flat.stabilizer.sign);
        }
    }

    #[test]
    fn composed_phase_cache_matches_flat_derivation_and_tracks_t_conversion() {
        for item in [
            GalleryItem::CNOT,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::T,
            GalleryItem::CCZInjectedMaj,
            GalleryItem::PhaseGradientK4,
        ] {
            let program = item.build();
            let summary = program
                .summarize_root(ModuleCertificationLimits::UNLIMITED)
                .unwrap();
            let graph = program.materialize_root_graph().unwrap();
            let zx = ZXGraph::try_from(&graph).unwrap();
            let composed = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
            let cached = composed.zx_graph.stabilizer_phase_basis.get().unwrap();
            let mut queries = composed
                .generators
                .iter()
                .map(|row| row.stabilizer.paulis.clone())
                .chain(composed.auxiliary_rows().iter().cloned())
                .collect::<Vec<_>>();
            let products = queries
                .windows(2)
                .map(|rows| {
                    let mut product = rows[0].clone();
                    product ^= &rows[1];
                    product
                })
                .collect::<Vec<_>>();
            queries.extend(products);
            let flat = ZXGraph::try_from(&graph).unwrap();
            let mut raw_queries = queries.clone();
            flat.clear_cross_centers(&mut raw_queries);
            assert_eq!(
                composed.zx_graph.stabilizer_row_phases(&queries),
                flat.external_stabilizer_row_phases(&raw_queries),
                "{item:?}"
            );
            let runtime =
                crate::RuntimeStabilizerBasis::from_generators(&composed).with_t_nodes_as_ports();
            let retained = runtime.zx_graph().stabilizer_phase_basis.get().unwrap();
            assert!(Arc::ptr_eq(cached, retained), "{item:?}");
            let mut fresh_demoted = runtime.zx_graph().clone();
            fresh_demoted.stabilizer_phase_basis.take();
            assert_eq!(
                runtime.zx_graph().stabilizer_row_phases(&queries),
                fresh_demoted.stabilizer_row_phases(&queries),
                "{item:?} after T conversion"
            );
        }
    }

    #[test]
    fn rotated_phase_gradient_summary_preserves_flat_nonlogical_rows() {
        let program = GalleryItem::PhaseGradientK4.build();
        let summary = program
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap();
        let graph = program.materialize_root_graph().unwrap();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let composed = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
        let flat = graph.analyze_actions().unwrap().1;

        assert_eq!(composed.generators.len(), flat.generators.len());
        for (composed, flat) in composed.generators.iter().zip(&flat.generators) {
            assert_eq!(composed.kind, flat.kind);
            // Composition preserves child readout representatives; flat analysis
            // may choose a different logical basis for the same graph.
            if matches!(composed.kind, StabilizerRowKind::Logical) {
                continue;
            }
            assert_eq!(
                composed.stabilizer.paulis, flat.stabilizer.paulis,
                "{:?}",
                composed.kind
            );
            assert_eq!(
                composed.stabilizer.sign, flat.stabilizer.sign,
                "{:?}",
                composed.kind
            );
        }
    }

    #[test]
    fn leaf_certificate_preserves_signed_rows() {
        let mut phases = FxHashSet::default();
        for item in [
            GalleryItem::CCZInjectedAnd,
            GalleryItem::CCZInjectedMaj,
            GalleryItem::UMA,
        ] {
            let program = item.build();
            let summary = program
                .summarize_root(ModuleCertificationLimits::UNLIMITED)
                .unwrap();
            let graph = program.materialize_root_graph().unwrap();
            let zx = ZXGraph::try_from(&graph).unwrap();
            for row in &summary.certificate_rows {
                let paulis = materialize_summary_row(
                    &row.signed.paulis,
                    &row.support,
                    &summary.ports,
                    &zx,
                    Ok,
                    summary.name(),
                )
                .unwrap();
                let witnessed = zx.materialize_flow_witness(&row.witness, &paulis).unwrap();
                assert_eq!(witnessed.phase(), row.signed.phase(), "{item:?} witness");
                assert_eq!(
                    witnessed.phase(),
                    zx.external_stabilizer_row_phases(&[paulis])[0],
                    "{item:?} certificate"
                );
                phases.insert(witnessed.phase());
            }
            let composed = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();
            let flat = graph.analyze_actions().unwrap().1;
            assert_eq!(composed.generators.len(), flat.generators.len());
            for (composed, flat) in composed.generators.iter().zip(&flat.generators) {
                assert_eq!(composed.kind, flat.kind);
                assert_eq!(composed.stabilizer.paulis, flat.stabilizer.paulis);
                assert_eq!(composed.stabilizer.sign, flat.stabilizer.sign);
            }
        }
        assert_eq!(phases, FxHashSet::from_iter([0, 2]));
    }

    #[test]
    fn composed_adder_rejects_unavailable_causal_readout() {
        let program = crate::BlockGraph::from_text(bloq_test::ONE_BIT_ADDER_SOURCE).unwrap();
        let summary = program
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        let graph = program
            .materialize_root_graph()
            .unwrap()
            .canonical_true_branch_view()
            .unwrap();
        let offset = IVec3::new(0, 0, -*graph.spans().unwrap().2.start());
        let graph = graph.with_zero_min_z().unwrap().fix_shadowed_faces();
        let zx = ZXGraph::try_from(&graph).unwrap();

        let result = summary.materialize_stabilizers(&zx, offset).map(|_| ());
        assert!(
            matches!(
                result,
                Err(ModuleCertificationError::Runtime {
                    source: crate::RuntimeBasisError::Stabilizer(
                        crate::StabilizerError::UnavailableControlParity { ref mvar, .. }
                    ),
                    ..
                }) if mvar == "uma__m_ikprime"
            ),
            "{result:?}"
        );
    }

    #[test]
    fn composed_adder_certificate_matches_flat_row_space() {
        let program = crate::BlockGraph::from_text(bloq_test::ONE_BIT_ADDER_SOURCE).unwrap();
        let summary = program
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        let graph = program
            .materialize_root_graph()
            .unwrap()
            .canonical_true_branch_view()
            .unwrap();
        let offset = IVec3::new(0, 0, -*graph.spans().unwrap().2.start());
        let graph = graph.with_zero_min_z().unwrap().fix_shadowed_faces();
        let zx = ZXGraph::try_from(&graph).unwrap();

        let composed = summary
            .certificate_rows
            .iter()
            .map(|row| {
                materialize_summary_row(
                    &row.signed.paulis,
                    &row.support,
                    &summary.ports,
                    &zx,
                    |position| checked_add_position(position, offset),
                    summary.name(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let flat = zx.to_external_generator_table();
        let composed_rank = crate::zx::reduce_to_basis(&composed, zx.total_ids()).len();
        let flat_rank = crate::zx::reduce_to_basis(&flat, zx.total_ids()).len();
        let mut combined = composed;
        combined.extend(flat);
        let combined_rank = crate::zx::reduce_to_basis(&combined, zx.total_ids()).len();

        assert_eq!(composed_rank, flat_rank);
        assert_eq!(combined_rank, flat_rank);
    }

    #[test]
    fn adaptive_summary_does_not_require_global_readout_presentation() {
        let summary = GalleryItem::ThreeBitAdder
            .build()
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap();
        assert!(summary.boundary_rows().len() > 0);
    }

    #[test]
    fn preparation_validates_before_consulting_shared_definition_cache() {
        let mut program = crate::parse_inline_graph(
            "BLOG 1.0\nmodule Child {\n0: ZXZ [0,0,0]\n}\n\
             module main {\nchild: Child @ [0,0,0]\n}\n",
        )
        .unwrap();
        program.instances[0].definition = "Missing".into();
        let error = program
            .prepare_module_relations_with_cache(
                ModuleCertificationLimits::DEFAULT,
                NonZeroUsize::MIN,
                |_| panic!("invalid source must not reach the cache"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("Missing"), "{error}");
    }

    #[test]
    fn mismatched_prepared_definition_cannot_hide_invalid_child_readout() {
        let program = |actions| {
            crate::parse_inline_graph(&format!(
                "BLOG 1.0\nmodule Child {{
                    in enable
                    out q: data = 2
                    1: XZX [0,0,1]
                    2: Port [0,0,2] role=output
                    3: T [1,0,0]
                    4: XZX [1,0,1]
                    5: YX [1,0,2]
                    1 -> +Z\n1 -> +X\n3 -> +Z\n5 -> -Z
                    {actions}
                }}
                module main {{
                    in enable
                    child: Child @ [0,0,0]
                    enable => child.enable
                    10: Z [0,0,2]
                    child.q -> 10
                }}"
            ))
            .unwrap()
        };
        let limits = ModuleCertificationLimits::DEFAULT;
        let valid = program("resolve 5 if enable");
        let cached = valid
            .prepare_module_relations_with_cache(limits, NonZeroUsize::MIN, |_| None)
            .unwrap();
        let child = cached
            .definitions()
            .into_iter()
            .find(|definition| definition.name() == "Child")
            .unwrap();
        let invalid = program("mzz = measure 1 -> +X\nresolve 5 if mzz");
        let key = invalid.module_definition_cache_keys()["Child"].clone();
        let reused = HashMap::from([(key, child)]);
        assert!(matches!(
            invalid.prepare_module_relations_with_cache(limits, NonZeroUsize::MIN, |key| {
                reused.get(key).cloned()
            }),
            Err(ModuleCertificationError::Graph { module, .. }) if module == "Child"
        ));
        assert!(matches!(
            invalid.prepare_module_summary_with_cache(limits, NonZeroUsize::MIN, |key| {
                reused.get(key).cloned()
            }),
            Err(ModuleCertificationError::Graph { module, .. }) if module == "Child"
        ));
    }
}
