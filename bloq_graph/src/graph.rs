//! The editable block graph and its topology-preserving operations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::RangeInclusive;
use std::str::FromStr;
use std::sync::Arc;

use bloq_utils::{Basis, PauliBasis, UDirection};
use glam::IVec3;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableUnGraph};
use petgraph::visit::EdgeRef;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use rustc_hash::FxBuildHasher;
use smallvec::SmallVec;

use crate::ZXGraph;
#[cfg(feature = "gltf")]
use crate::gltf::{
    GltfData, GltfFaceSelector, POPPED_X_FACE_COLOR, POPPED_Y_FACE_COLOR, POPPED_Z_FACE_COLOR,
    block_graph_as_gltf_data_with_modules, block_graph_as_gltf_data_with_popped_faces,
    module_html_geometry, stabilizer_as_gltf_data,
};
use crate::validate::InvalidActionError;
use crate::zx::{StabilizerGenerator, fill_ports_auto as fill_ports_auto_impl};
use crate::{
    Action, ActionDag, Block, BlockGraphError, BlockKind, CubeHeight, CubeKind, Direction,
    MeasureTarget, MeasurementObservable, PatchRotationKind, Pipe, WalkingBoundaryKind,
    WalkingKind,
};

#[cfg(feature = "gltf")]
const OUTER_FACE_OPACITY_SCALE: f32 = 0.55;
#[cfg(feature = "gltf")]
fn emphasize_popped_faces(mut data: GltfData) -> GltfData {
    // Scoped here rather than at the top of the file: this is the crate's only
    // use of either, and a featureless build would warn on a top-level import.
    use bloq_utils::RGBA;

    let mut triangles: BTreeMap<RGBA, Vec<[glam::Vec3; 3]>> = BTreeMap::new();
    for (color, faces) in std::mem::take(&mut data.triangles) {
        let color = match color {
            RGBA::X_RED => POPPED_X_FACE_COLOR,
            RGBA::Y_GREEN => POPPED_Y_FACE_COLOR,
            RGBA::Z_BLUE => POPPED_Z_FACE_COLOR,
            _ => RGBA { a: 255, ..color },
        };
        triangles.entry(color).or_default().extend(faces);
    }
    data.triangles = triangles;
    data
}

/// A read-only view of a single time (`z`) layer of a [`BlockGraph`].
///
/// Obtained via [`BlockGraph::layer`]; use [`into_graph`](Self::into_graph) to
/// materialize the layer as a standalone graph.
#[derive(Debug, Clone, Copy)]
pub struct BlockLayerView<'a> {
    graph: &'a BlockGraph,
    z: i32,
}

/// Branch-arm footprints retained only while a module linker bulk-adds child
/// blocks. The editable graph continues to use its ordinary collision path.
#[derive(Default)]
pub(crate) struct LinkBranchOccupancy {
    by_position: HashMap<IVec3, Vec<(IVec3, BlockKind)>>,
}

impl LinkBranchOccupancy {
    pub(crate) fn add_region(
        &mut self,
        region: &crate::BranchRegion,
    ) -> Result<(), BlockGraphError> {
        for block in region.on_false().blocks().chain(region.on_true().blocks()) {
            for position in block.checked_reserved_positions()? {
                self.by_position
                    .entry(position)
                    .or_default()
                    .push((block.pos(), block.kind()));
            }
        }
        Ok(())
    }
}

/// A graph of surface code blocks connected by lattice-surgery pipes.
///
/// This is the source-level, authoring representation: blocks sit at integer 3D
/// lattice positions (the `z` axis is time), pipes connect adjacent endpoints,
/// and classical control flow lives in a cached [`ActionDag`]. Structural edits
/// refresh that DAG; metadata-only tag edits leave it intact.
/// [`validate`](Self::validate) rebuilds and checks both structure and action
/// semantics.
#[derive(Debug, Clone)]
pub struct BlockGraph {
    inner: StableUnGraph<Block, Pipe>,
    block_id: HashMap<IVec3, u32, FxBuildHasher>,
    reserved_by_position: HashMap<IVec3, SmallVec<[IVec3; 1]>, FxBuildHasher>,
    pub(crate) branches: Vec<crate::BranchRegion>,
    action_graph: ActionDag,
    action_inputs: BTreeSet<String>,
    /// Error from the last post-mutation DAG rebuild, if it failed. While set,
    /// [`action_graph`](Self::action_graph) holds an unvalidated best-effort DAG.
    action_graph_error: Option<BlockGraphError>,
    /// Definition name, `main` for the executable root.
    pub name: String,
    /// Public quantum and classical interface.
    pub interface: crate::ModuleInterface,
    /// Rotated and translated references to other graph definitions.
    pub instances: Vec<crate::ModuleInstance>,
    /// Authored quantum seams between local blocks and child instances.
    pub quantum_connections: Vec<crate::QuantumConnection>,
    /// Authored bindings to child classical inputs.
    pub bit_bindings: Vec<crate::BitBinding>,
    pub(crate) definitions: Vec<BlockGraph>,
    pub(crate) interface_declared: bool,
}

impl Default for BlockGraph {
    fn default() -> Self {
        Self {
            inner: StableUnGraph::default(),
            block_id: HashMap::default(),
            reserved_by_position: HashMap::default(),
            branches: Vec::new(),
            action_graph: ActionDag::default(),
            action_inputs: BTreeSet::new(),
            action_graph_error: None,
            name: Self::ENTRY_MODULE.to_string(),
            interface: crate::ModuleInterface::default(),
            instances: Vec::new(),
            quantum_connections: Vec::new(),
            bit_bindings: Vec::new(),
            definitions: Vec::new(),
            interface_declared: false,
        }
    }
}

impl BlockGraph {
    /// Replaces local geometry and actions while retaining interfaces and definitions.
    pub fn replace_local_body(&mut self, body: Self) {
        self.inner = body.inner;
        self.block_id = body.block_id;
        self.reserved_by_position = body.reserved_by_position;
        self.branches = body.branches;
        self.action_graph = body.action_graph;
        self.action_inputs = body.action_inputs;
        self.action_graph_error = body.action_graph_error;
    }

    pub(crate) fn copy_local_geometry(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            block_id: self.block_id.clone(),
            reserved_by_position: self.reserved_by_position.clone(),
            branches: self.branches.clone(),
            action_graph: self.action_graph.clone(),
            action_inputs: self.action_inputs.clone(),
            action_graph_error: self.action_graph_error.clone(),
            ..Self::default()
        }
    }
}

impl<'a> BlockLayerView<'a> {
    /// Returns the time layer this view is fixed to.
    pub fn z(&self) -> i32 {
        self.z
    }

    /// Returns the underlying graph.
    pub fn graph(&self) -> &BlockGraph {
        self.graph
    }

    /// Iterates over the blocks occupying this layer.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> + '_ {
        self.graph
            .blocks()
            .filter(move |block| block.occupies_layer(self.z))
    }

    /// Iterates over the pipes lying within this layer, projecting spatial pipes
    /// of multi-layer blocks onto the layer's `z`.
    pub fn spacelike_pipes(&self) -> impl Iterator<Item = Pipe> + '_ {
        self.graph.pipes().filter_map(move |pipe| {
            let p1 = pipe.src;
            let p2 = pipe.dst();
            if p1.z == self.z && p2.z == self.z {
                return Some(pipe.clone());
            }
            if !pipe.dir.is_spatial() {
                return None;
            }
            let src_block = self.graph.get_endpoint_block(p1)?;
            let dst_block = self.graph.get_endpoint_block(p2)?;
            if src_block.occupies_layer(self.z) && dst_block.occupies_layer(self.z) {
                let mut projected = pipe.clone();
                projected.src.z = self.z;
                Some(projected)
            } else {
                None
            }
        })
    }

    /// Iterates over the temporal pipes crossing into the layer above or below.
    pub fn boundary_pipes(&self) -> impl Iterator<Item = &Pipe> + '_ {
        self.graph.pipes().filter(move |pipe| {
            let p1 = pipe.src;
            let p2 = pipe.dst();
            let (p1, p2) = if p1.z <= p2.z { (p1, p2) } else { (p2, p1) };
            (p1.z == self.z && p2.z > self.z) || (p1.z < self.z && p2.z == self.z)
        })
    }

    /// Materializes the layer as a standalone [`BlockGraph`], replacing temporal
    /// pipes into adjacent layers with connections to virtual
    /// [`BlockKind::Port`] blocks. Actions are dropped.
    ///
    /// # Panics
    ///
    /// Panics if the source graph's internal block or pipe indexes are inconsistent.
    pub fn into_graph(self) -> BlockGraph {
        let mut layer = BlockGraph::new();
        for block in self.blocks() {
            let mut layer_block = block.clone();
            if layer_block.kind.is_cube() && layer_block.height_cells() != 1 {
                layer_block = Block::new(IVec3::new(block.pos.x, block.pos.y, self.z), block.kind);
                if let Some(tag) = block.tag() {
                    layer_block = layer_block
                        .with_tag(tag)
                        .expect("existing block tag is valid");
                }
            }
            layer.add_block(layer_block);
        }
        for p in self.spacelike_pipes() {
            layer.add_pipe(p);
        }
        for p in self.boundary_pipes() {
            let p1 = p.src;
            let p2 = p.dst();
            let (p1, p2) = if p1.z <= p2.z { (p1, p2) } else { (p2, p1) };
            if p1.z == self.z && p2.z > self.z {
                layer.add_block(Block::new(p2, BlockKind::Port));
                layer.add_pipe(p.clone());
            } else if p1.z < self.z && p2.z == self.z {
                layer.add_block(Block::new(p1, BlockKind::Port));
                layer.add_pipe(p.clone());
            }
        }
        layer.clear_actions();
        layer
    }
}

impl BlockGraph {
    /// Creates an empty block graph.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn require_flat_hierarchy(
        &self,
        operation: &'static str,
    ) -> Result<(), BlockGraphError> {
        if self.instances.is_empty() {
            Ok(())
        } else {
            Err(BlockGraphError::HierarchyRequiresFlatten { operation })
        }
    }

    pub(crate) fn require_flat_source(
        &self,
        operation: &'static str,
    ) -> Result<(), BlockGraphError> {
        if self.has_module_structure() {
            Err(BlockGraphError::HierarchyRequiresFlatten { operation })
        } else {
            Ok(())
        }
    }

    /// Infers the X/Y/Z basis assignment for each face of a pipe from `pipe.src`.
    pub fn infer_pipe_basis(&self, pipe: &Pipe) -> [Option<Basis>; 3] {
        self.infer_pipe_basis_from_endpoint(pipe, pipe.src)
    }

    /// Infers the X/Y/Z basis assignment for each face of a pipe from one endpoint.
    pub fn infer_pipe_basis_from_endpoint(
        &self,
        pipe: &Pipe,
        endpoint: IVec3,
    ) -> [Option<Basis>; 3] {
        let Ok(dst) = pipe.try_dst() else {
            return [None; 3];
        };
        if endpoint != pipe.src && endpoint != dst {
            return [None; 3];
        }

        let mut canonical = if pipe.src.to_array() < dst.to_array() {
            pipe.clone()
        } else {
            Pipe::new(dst, pipe.dir.negate())
        };
        if pipe.hadamard {
            canonical = canonical.with_hadamard();
        }
        let mut bases = self.infer_pipe_basis_from_source(&canonical);
        if endpoint != canonical.src && pipe.hadamard {
            for basis in bases.iter_mut().flatten() {
                *basis = basis.flip();
            }
        }
        bases
    }

    /// Infers the unique non-spatial cube that continues a spatial Port into a
    /// plain temporal pipe.
    ///
    /// The authored spatial pipe fixes the two bases perpendicular to its
    /// axis. The missing basis equals the temporal basis, leaving the other
    /// spatial axis as the cube normal.
    pub fn infer_spatial_port_cube_kind(&self, position: IVec3) -> Option<crate::CubeKind> {
        let block = self.get_block(position)?;
        if !block.kind().is_port() {
            return None;
        }
        let pipe = self
            .inner
            .edges(self.try_get_id(position).ok()?)
            .filter(|edge| edge.weight().src() == position || edge.weight().dst() == position)
            .min_by_key(|edge| edge.id().index())?
            .weight();
        if !pipe.dir().is_spatial() {
            return None;
        }
        let mut bases = self.infer_pipe_basis_from_endpoint(pipe, position);
        let temporal = bases[crate::UDirection::Z.index()]?;
        bases[pipe.dir().as_udirection().index()] = Some(temporal);
        let kind = crate::CubeKind::try_from([bases[0]?, bases[1]?, bases[2]?]).ok()?;
        (!kind.is_spatial()).then_some(kind)
    }

    fn infer_pipe_basis_from_source(&self, pipe: &Pipe) -> [Option<Basis>; 3] {
        let Ok(dst) = pipe.try_dst() else {
            return [None; 3];
        };
        let mut bases = [None; 3];
        let src_bases = self.infer_pipe_endpoint_face_bases(pipe, pipe.src);
        let dst_bases = self.infer_pipe_endpoint_face_bases(pipe, dst);

        for (i, dir) in UDirection::iter().enumerate() {
            if dir == pipe.dir.as_udirection() {
                continue;
            }
            if let Some(basis) = src_bases[i] {
                bases[i] = Some(basis);
            } else if let Some(b) = dst_bases[i] {
                bases[i] = Some(if pipe.hadamard { b.flip() } else { b });
            }
        }
        let pipe_dir_id = pipe.dir.as_udirection().index();
        // If no bases could be inferred, default the two perpendicular axes to
        // complementary X/Z bases
        let none_count = bases.iter().filter(|b| b.is_none()).count();
        if none_count == 3 {
            bases[(pipe_dir_id + 1) % 3] = Some(Basis::X);
            bases[(pipe_dir_id + 2) % 3] = Some(Basis::Z);
        } else if none_count == 2 {
            let Some(resolved_id) = bases.iter().position(Option::is_some) else {
                return bases;
            };
            let other_id = 3 - pipe_dir_id - resolved_id;
            if let Some(resolved) = bases[resolved_id] {
                bases[other_id] = Some(resolved.flip());
            }
        }
        bases
    }

    pub(crate) fn infer_pipe_endpoint_face_bases(
        &self,
        pipe: &Pipe,
        endpoint: IVec3,
    ) -> [Option<Basis>; 3] {
        let Ok(pipe_dst) = pipe.try_dst() else {
            return [None; 3];
        };
        if endpoint != pipe.src && endpoint != pipe_dst {
            return [None; 3];
        }
        let Some(block) = self.get_endpoint_block(endpoint) else {
            return [None; 3];
        };
        let endpoint_bases = match block.kind() {
            BlockKind::PatchRotation(kind) => {
                kind.pipe_face_bases_at_endpoint(block.pos(), endpoint)
            }
            _ => block.bases_at_endpoint(endpoint),
        };
        let Some(kind_bases) = endpoint_bases else {
            return [None; 3];
        };

        let mut bases = [None; 3];
        for (i, dir) in UDirection::iter().enumerate() {
            if dir == pipe.dir.as_udirection() {
                continue;
            }
            let offset = dir.to_ivec3();
            let has_opposite_pair = crate::checked_add_position(endpoint, offset)
                .is_ok_and(|neighbor| self.has_pipe_between(endpoint, neighbor))
                && crate::checked_add_position(endpoint, -offset)
                    .is_ok_and(|neighbor| self.has_pipe_between(endpoint, neighbor));
            if !has_opposite_pair {
                bases[i] = Some(kind_bases[i]);
            }
        }
        bases
    }

    /// Returns a read-only view of the time layer at `z`.
    pub fn layer(&self, z: i32) -> BlockLayerView<'_> {
        BlockLayerView { graph: self, z }
    }

    /// The `+X`/`+Y` connected components of each time layer's cubes, keyed by
    /// the layer the cube is *anchored* in, each component sorted and the
    /// components ordered by their lowest member.
    ///
    /// Spatial connectivity is the unit of shared time: merged patches drive the
    /// same data qubits, so anything schedule-like belongs to a whole component
    /// — the height propagated here, the CX-slot depth `bloq_compile` derives.
    /// `+Z` pipes are temporal and do not join components.
    pub fn cube_layer_components(&self) -> BTreeMap<i32, Vec<Vec<IVec3>>> {
        let mut layers = BTreeMap::<i32, Vec<IVec3>>::new();
        for block in self.blocks().filter(|block| block.kind().is_cube()) {
            layers.entry(block.pos().z).or_default().push(block.pos());
        }

        layers
            .into_iter()
            .map(|(z, mut positions)| {
                positions.sort_by_key(IVec3::to_array);
                let index: HashMap<IVec3, usize> = positions
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, pos)| (pos, i))
                    .collect();
                let mut union = petgraph::unionfind::UnionFind::new(positions.len());
                for pipe in self.layer(z).spacelike_pipes() {
                    let (src, dst) = pipe.endpoints();
                    let (Some(src), Some(dst)) =
                        (self.get_endpoint_block(src), self.get_endpoint_block(dst))
                    else {
                        continue;
                    };
                    // A cube anchored lower merely overlaps this layer; it
                    // belongs to its own layer's partition, not this one's.
                    let (Some(&src), Some(&dst)) = (index.get(&src.pos()), index.get(&dst.pos()))
                    else {
                        continue;
                    };
                    union.union(src, dst);
                }

                let mut groups: Vec<Vec<IVec3>> = Vec::new();
                let mut slot_of_root = HashMap::<usize, usize>::new();
                for (i, &pos) in positions.iter().enumerate() {
                    let slot = *slot_of_root.entry(union.find(i)).or_insert_with(|| {
                        groups.push(Vec::new());
                        groups.len() - 1
                    });
                    groups[slot].push(pos);
                }
                (z, groups)
            })
            .collect()
    }

    /// Returns the current action DAG.
    ///
    /// Structural and action-affecting edits refresh this DAG; tag-only edits
    /// leave it intact.
    pub fn action_graph(&self) -> &ActionDag {
        &self.action_graph
    }

    /// Returns the error from the last failed DAG rebuild; `Some` means the
    /// cached DAG is an unvalidated best effort. Cleared by the next
    /// successful rebuild.
    pub fn action_graph_error(&self) -> Option<&BlockGraphError> {
        self.action_graph_error.as_ref()
    }

    /// Derives an independent action DAG, including correlation dependencies.
    ///
    /// This inspection query does not change the source or compile physical IR.
    /// Authored hierarchies use the compiler's module linker and correlation
    /// composition, retaining definition and instance ownership in each node.
    /// Continuing branches use the guarded causal readout planner.
    /// Composed and continuing-branch DAGs retain dependencies without assigning
    /// a single measurement stabilizer to each node.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid sources, causal cycles, unavailable readouts,
    /// or exhausted analysis budgets.
    pub fn analyze_action_graph(&self) -> Result<ActionDag, BlockGraphError> {
        self.analyze_action_graph_with_limits(crate::ModuleCertificationLimits::DEFAULT)
    }

    /// Derives an action DAG with explicit source-analysis resource limits.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::analyze_action_graph`], including
    /// exhaustion of the supplied budgets.
    pub fn analyze_action_graph_with_limits(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<ActionDag, BlockGraphError> {
        if self.has_module_structure() {
            return crate::action_dag::hierarchy::analyze(self, limits);
        }
        self.analyzed_local_action_graph(limits)
    }

    /// Causal readout plan of this graph as one definition, ignoring any
    /// module hierarchy around it.
    pub(crate) fn plan_local_readouts(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<crate::GuardedReadoutPlan, BlockGraphError> {
        let topology = crate::GuardedTopology::new(self, limits)?;
        crate::GuardedSurfaceSpace::new(topology, &Default::default(), limits)?.plan_readouts()
    }

    /// Plans causal readouts over this source's correlation relation. A flat
    /// source is one definition; a hierarchy composes its definition relations.
    fn plan_correlation_readouts(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<crate::GuardedReadoutPlan, BlockGraphError> {
        if self.has_module_structure() {
            return crate::action_dag::hierarchy::plan_readouts(self, limits).map(|(_, plan)| plan);
        }
        self.plan_local_readouts(limits)
    }

    /// Builds this definition's action DAG with planned readout dependencies
    /// and the measurement surfaces of its displayed branch arms.
    fn analyzed_local_action_graph(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<ActionDag, BlockGraphError> {
        let mut plan = self.plan_local_readouts(limits)?;
        let mut dag = self.build_action_graph(&self.actions())?;
        dag.attach_guarded_dependencies(plan.action_dependencies()?)?;
        // A displayed selection the guards cannot reach has no surfaces to show.
        if let Some(surfaces) = shown_surfaces(&plan, &self.shown_branch_arms())? {
            dag.attach_measurement_surfaces(surfaces);
        }
        Ok(dag)
    }

    /// Correlation surfaces of one displayed branch selection, from the same
    /// causal readout plan used by compilation and action analysis.
    ///
    /// Named readouts are measurement rows; output frames and closed logical
    /// surfaces are logical rows. `assignments` selects structural branch arms
    /// as in [`Self::project_branches_deferred`]; omitted branches show their
    /// true arm. The surfaces are positioned in [`Self::flatten`] coordinates.
    ///
    /// ```
    /// use bloq_graph::{GalleryItem, ModuleCertificationLimits};
    ///
    /// let adder = GalleryItem::ThreeBitAdder.build();
    /// let surfaces = adder.planned_surfaces(&[], ModuleCertificationLimits::DEFAULT)?;
    /// assert!(
    ///     surfaces
    ///         .iter()
    ///         .any(|row| row.measurement_name() == Some("bit0_uma__m_ikprime"))
    /// );
    /// # Ok::<(), bloq_graph::BlockGraphError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error for invalid sources, an unreachable arm selection,
    /// unavailable readouts, or exhausted analysis budgets.
    pub fn planned_surfaces(
        &self,
        assignments: &[(IVec3, bool)],
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Vec<StabilizerGenerator>, BlockGraphError> {
        let plan = self.plan_correlation_readouts(limits)?;
        shown_surfaces(&plan, assignments)?
            .ok_or_else(|| InvalidActionError::UnreachableBranchSelection.into())
    }

    /// Returns the actions in source order.
    pub fn actions(&self) -> Vec<Action> {
        self.action_graph
            .ordered_nodes()
            .map(|node| node.action.clone())
            .collect()
    }

    /// Returns `true` if the graph has any actions.
    pub fn has_actions(&self) -> bool {
        self.action_graph.ordered_nodes().next().is_some()
    }

    /// Returns the internal node ID for the block at the given position.
    pub fn get_block_id(&self, pos: impl Into<IVec3>) -> Option<u32> {
        self.block_id.get(&pos.into()).copied()
    }

    pub(crate) fn blog_block_ids(&self) -> Vec<(IVec3, u32)> {
        let mut blocks = self
            .block_id
            .iter()
            .map(|(&position, &id)| (position, id))
            .collect::<Vec<_>>();
        blocks.sort_by_key(|(_, id)| *id);
        if self.branches.is_empty() {
            return blocks;
        }
        let shown = self
            .branches
            .iter()
            .flat_map(|region| region.shown_arm().blocks().map(Block::pos))
            .collect::<HashSet<_>>();
        blocks
            .into_iter()
            .filter(|(position, _)| !shown.contains(position))
            .enumerate()
            .map(|(id, (position, _))| {
                (
                    position,
                    u32::try_from(id).expect("BlockGraph ids fit in u32"),
                )
            })
            .collect()
    }

    /// Returns the stored block ID for a connectable endpoint position.
    ///
    /// Ordinary blocks have one endpoint at their stored position. Extended
    /// blocks also expose their shifted end position as a connectable endpoint
    /// that resolves to the same stored graph node.
    pub(crate) fn get_endpoint_block_id(&self, pos: impl Into<IVec3>) -> Option<u32> {
        self.try_get_endpoint_id(pos)
            .ok()
            .map(|id| id.index() as u32)
    }

    /// Converts this static block graph to a ZX-calculus graph.
    ///
    /// A graph with structural Branch actions must first be expanded with
    /// [`BlockGraph::branch_projections`]; optional topology has no single ZX
    /// graph representation.
    ///
    /// # Errors
    ///
    /// Returns an error if graph validation or ZX conversion fails.
    pub fn to_zx_graph(&self) -> Result<ZXGraph, BlockGraphError> {
        self.require_flat_hierarchy("ZX conversion")?;
        ZXGraph::try_from(self).map_err(Into::into)
    }

    /// Correlation surfaces of the displayed branch arms, from the causal
    /// readout plan used by compilation. See [`Self::planned_surfaces`].
    ///
    /// # Errors
    ///
    /// Returns an error for invalid sources, unavailable readouts, or exhausted
    /// analysis budgets.
    pub fn stabilizers(&self) -> Result<Vec<StabilizerGenerator>, BlockGraphError> {
        self.planned_surfaces(
            &self.shown_branch_arms(),
            crate::ModuleCertificationLimits::DEFAULT,
        )
    }

    /// The displayed arm of every branch region.
    fn shown_branch_arms(&self) -> Vec<(IVec3, bool)> {
        self.branches
            .iter()
            .map(|branch| (branch.target, branch.shown_true()))
            .collect()
    }

    /// Installs the planned action analysis: readout dependencies and the
    /// displayed arms' measurement surfaces.
    ///
    /// # Errors
    ///
    /// Returns an error for hierarchies, invalid sources, causal cycles,
    /// unavailable readouts, or exhausted analysis budgets.
    pub fn analyze_actions(self) -> Result<Self, BlockGraphError> {
        self.analyze_actions_with_limits(crate::ModuleCertificationLimits::DEFAULT)
    }

    /// Analyze source actions without replacing explicit resource limits with defaults.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::analyze_actions`] within `limits`.
    pub fn analyze_actions_with_limits(
        mut self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Self, BlockGraphError> {
        self.require_flat_hierarchy("action analysis")?;
        self.validate_resource_limits(limits)?;
        self.analyze_installed_actions(limits)?;
        Ok(self)
    }

    /// Replaces the installed DAG with its planned analysis.
    fn analyze_installed_actions(
        &mut self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<(), BlockGraphError> {
        let dag = self.analyzed_local_action_graph(limits)?;
        self.install_action_graph(dag, None);
        Ok(())
    }

    /// Fills port blocks using compatible external stabilizer support and
    /// returns each closed graph with the generators that selected its fill.
    /// Static variants retain selected measurement records and their aliases,
    /// removing feedback and resolves for filled caps. Returns an error if
    /// postselection or a structural branch needs an unselected record.
    ///
    /// # Errors
    ///
    /// Returns an error if automatic filling or postselection validation fails.
    pub fn fill_ports_auto(
        &self,
    ) -> Result<Vec<(BlockGraph, Vec<StabilizerGenerator>)>, BlockGraphError> {
        self.fill_ports_auto_with_limits(crate::ModuleCertificationLimits::DEFAULT)
    }

    /// Fills ports with explicit limits for source and filled-graph analysis.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`fill_ports_auto`](Self::fill_ports_auto),
    /// including exhaustion of a caller's certification budget.
    pub fn fill_ports_auto_with_limits(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Vec<(BlockGraph, Vec<StabilizerGenerator>)>, BlockGraphError> {
        self.require_flat_source("boundary filling")?;
        fill_ports_auto_impl(self, limits)
    }

    /// The closed graphs [`fill_ports_auto`](Self::fill_ports_auto) produces,
    /// without the generators that selected each fill.
    ///
    /// # Errors
    ///
    /// Returns the same automatic-filling errors as [`Self::fill_ports_auto`].
    pub fn filled_graphs(&self) -> Result<Vec<BlockGraph>, BlockGraphError> {
        self.filled_graphs_with_limits(crate::ModuleCertificationLimits::DEFAULT)
    }

    /// The graphs from [`fill_ports_auto_with_limits`](Self::fill_ports_auto_with_limits),
    /// without their selecting stabilizer generators.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::fill_ports_auto_with_limits`].
    pub fn filled_graphs_with_limits(
        &self,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Vec<BlockGraph>, BlockGraphError> {
        Ok(self
            .fill_ports_auto_with_limits(limits)?
            .into_iter()
            .map(|(graph, _)| graph)
            .collect())
    }

    /// Returns the inclusive coordinate span of the graph on each axis,
    /// including both stored arms of every branch, or `None` when empty.
    #[must_use]
    pub fn spans(
        &self,
    ) -> Option<(
        RangeInclusive<i32>,
        RangeInclusive<i32>,
        RangeInclusive<i32>,
    )> {
        let boundary_positions = |block: &Block| {
            // A tall cube's extrema are its two endpoint cells. Interior cells
            // cannot enlarge the box and need not be materialized or visited.
            let offsets = if block.kind().is_cube() {
                block.connectable_offsets()
            } else {
                block.reserved_offsets()
            };
            let position = block.pos();
            offsets.into_iter().map(move |offset| position + offset)
        };
        let visible = self.blocks().flat_map(boundary_positions);
        let branch_arms = self
            .branches
            .iter()
            .flat_map(|region| region.on_false().blocks().chain(region.on_true().blocks()))
            .flat_map(boundary_positions);
        let mut positions = visible.chain(branch_arms);
        let first_pos = positions.next()?;

        let (mut min_x, mut max_x) = (first_pos.x, first_pos.x);
        let (mut min_y, mut max_y) = (first_pos.y, first_pos.y);
        let (mut min_z, mut max_z) = (first_pos.z, first_pos.z);

        for pos in positions {
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
            min_z = min_z.min(pos.z);
            max_z = max_z.max(pos.z);
        }

        Some((min_x..=max_x, min_y..=max_y, min_z..=max_z))
    }

    /// Resolve all selective blocks from one random assignment of their action
    /// measurement variables. Resolve expressions that share a variable are
    /// therefore sampled jointly; sites without actions retain independent
    /// random choices. Returns the resolved graph and a map from original to
    /// resolved blocks. Resolution is deterministic in `seed`.
    /// Authored module sources must be explicitly flattened first.
    ///
    /// If at least one selective is resolved, the result is structure-only and
    /// all actions are removed, including structural branches. Project branches
    /// first when a concrete conditional topology is required.
    ///
    /// # Panics
    ///
    /// Panics if internally cached resolve metadata names a non-selective block.
    ///
    /// # Errors
    ///
    /// Returns a typed hierarchy error unless child instances are explicitly flattened.
    pub fn randomly_resolve_selectives(
        &self,
        seed: u64,
    ) -> Result<(BlockGraph, HashMap<Block, Block>), BlockGraphError> {
        self.require_flat_source("selective resolution")?;
        let mut rng = StdRng::seed_from_u64(seed);
        let selective_blocks: Vec<Block> = self
            .blocks()
            .filter(|block| block.kind.is_selective())
            .cloned()
            .collect();
        let mut resolved_graph = self.clone();
        let mut replacements = HashMap::with_capacity(selective_blocks.len());
        let has_selectives = !selective_blocks.is_empty();
        let targets = selective_blocks
            .iter()
            .map(|block| block.pos)
            .collect::<Vec<_>>();
        let sampled_resolves = self.action_graph.sample_resolve_values(&targets, &mut rng);

        for original in selective_blocks {
            let resolved_kind = sampled_resolves.get(&original.pos).map_or_else(
                || self.randomly_resolved_selective_kind(&original, &mut rng),
                |&value| {
                    let BlockKind::Selective(kind) = original.kind else {
                        unreachable!("selective_blocks contains only selectives")
                    };
                    let chosen = if value {
                        kind.pauli_if_true()
                    } else {
                        kind.pauli_if_false()
                    };
                    self.resolved_selective_kind_for_basis(chosen)
                },
            );
            let resolved_block = {
                let block = resolved_graph
                    .get_block_mut(original.pos)
                    .expect("resolved graph cloned from source must still contain selective block");
                // Selective and every resolved measurement kind reserve the same
                // single cell, so the occupancy index stays correct.
                block.kind = resolved_kind;
                block.clone()
            };
            replacements.insert(original, resolved_block);
        }

        // Resolve actions and selective-measurement metadata no longer match a
        // statically filled graph, so return structure only.
        if has_selectives {
            resolved_graph.clear_actions();
        }

        Ok((resolved_graph, replacements))
    }

    /// Appends an action, revalidating and re-analyzing the whole action set
    /// against the graph. See [`set_actions`](Self::set_actions).
    ///
    /// # Errors
    ///
    /// Returns a [`BlockGraphError`] if the resulting action set fails
    /// validation or physical analysis.
    pub fn add_action(&mut self, action: Action) -> Result<(), BlockGraphError> {
        let mut ordered = self
            .action_graph
            .ordered_nodes()
            .map(|node| node.action.clone())
            .collect::<Vec<_>>();
        ordered.push(action);
        self.set_actions(ordered)
    }

    pub(crate) fn resolve_measurement_observable(
        &self,
        target: &MeasureTarget,
    ) -> Result<MeasurementObservable, BlockGraphError> {
        match target {
            MeasureTarget::Node(pos) => match self
                .get_block(*pos)
                .ok_or(BlockGraphError::BlockNotFound(*pos))?
                .kind
            {
                BlockKind::Y => Ok(MeasurementObservable::Concrete(PauliBasis::Y)),
                BlockKind::Measurement(basis) => Ok(MeasurementObservable::Concrete(basis.into())),
                BlockKind::Selective(kind) => Ok(MeasurementObservable::Selective(kind)),
                BlockKind::Cube(kind) => Ok(MeasurementObservable::Concrete(kind.z().into())),
                BlockKind::Walking(kind) => {
                    Ok(MeasurementObservable::Concrete(kind.boundary().z().into()))
                }
                BlockKind::PatchRotation(kind) => {
                    Ok(MeasurementObservable::Concrete(kind.basis().into()))
                }
                BlockKind::Port | BlockKind::T => Err(BlockGraphError::InvalidAction(
                    InvalidActionError::InvalidMeasurementNode(*pos),
                )),
            },
            MeasureTarget::Edge { src, dir } => {
                let dst = crate::checked_add_position(*src, dir.to_ivec3())?;
                let pipe = self
                    .get_pipe(*src, dst)
                    .ok_or(BlockGraphError::PipeNotFound(*src, dst))?;
                let bases = self.infer_pipe_basis(pipe);
                // Must stay an `Err`, not a panic: the best-effort DAG rebuild
                // after mutation calls this on possibly-invalid graphs and
                // swallows resolution failures by design.
                let basis =
                    bases[2].ok_or(BlockGraphError::MeasurementEdgeMissingTimeBasis(*src, dst))?;
                Ok(MeasurementObservable::Concrete(basis.flip().into()))
            }
        }
    }

    pub(crate) fn build_action_graph(
        &self,
        actions: &[Action],
    ) -> Result<ActionDag, BlockGraphError> {
        if actions.is_empty() {
            return Ok(self.new_action_graph(actions));
        }
        let mut dag = self.new_action_graph(actions);
        dag.validate(Some(self))?;
        self.attach_measurement_observables(&mut dag, actions.len(), true)?;
        Ok(dag)
    }

    fn new_action_graph(&self, actions: &[Action]) -> ActionDag {
        ActionDag::from_actions_with_inputs(actions, self.action_inputs.iter().cloned())
    }

    /// Attaches the resolved measurement observable to every `Measure` node.
    ///
    /// When `strict`, the first resolution failure aborts and is returned; this
    /// is what [`build_action_graph`](Self::build_action_graph) wants. When not
    /// strict, unresolvable nodes are simply left without metadata and the rest
    /// are still populated — used by the post-mutation refresh so consumers do
    /// not see spurious `None` for nodes that *can* resolve.
    fn attach_measurement_observables(
        &self,
        dag: &mut ActionDag,
        len: usize,
        strict: bool,
    ) -> Result<(), BlockGraphError> {
        for ordinal in 0..len {
            let target = match dag.node_by_ordinal(ordinal).map(|node| &node.action) {
                Some(Action::Measure { target, .. }) => *target,
                _ => continue,
            };
            match self.resolve_measurement_observable(&target) {
                Ok(observable) => dag
                    .set_measurement_observable(ordinal, observable)
                    .expect("ordinals 0..len index the DAG built from these same actions"),
                Err(err) if strict => return Err(err),
                Err(_) => {}
            }
        }
        Ok(())
    }

    /// Single point of installation for the cached DAG, keeping
    /// `action_graph_error` in lockstep with `action_graph`.
    pub(crate) fn install_action_graph(&mut self, dag: ActionDag, error: Option<BlockGraphError>) {
        self.action_graph = dag;
        self.action_graph_error = error;
    }

    /// Builds an unvalidated DAG, attaching whatever measurement metadata
    /// still resolves. Used when a strict [`build_action_graph`](Self::build_action_graph)
    /// has failed but the actions must be kept.
    fn build_action_graph_best_effort(&self, actions: &[Action]) -> ActionDag {
        let mut dag = self.new_action_graph(actions);
        let _ = self.attach_measurement_observables(&mut dag, actions.len(), false);
        dag
    }

    /// Rebuilds and installs the DAG from `actions`; on failure installs a
    /// best-effort DAG and records the error in `action_graph_error`.
    pub(crate) fn rebuild_action_graph_lenient(&mut self, actions: Vec<Action>) {
        match self.build_action_graph(&actions) {
            Ok(dag) => self.install_action_graph(dag, None),
            Err(err) => {
                let dag = self.build_action_graph_best_effort(&actions);
                self.install_action_graph(dag, Some(err));
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_actions_unchecked(&mut self, actions: Vec<Action>) {
        let dag = self.new_action_graph(&actions);
        self.install_action_graph(dag, None);
    }

    /// Replaces all actions, validating the new set against the graph and, once
    /// the topology is structurally complete, deriving each measurement's
    /// stabilizer surface and its implicit dependency edges.
    ///
    /// # Errors
    ///
    /// Returns a [`BlockGraphError`] if the new action set fails semantic or
    /// graph-constraint validation, a measurement surface is unavailable, or
    /// the derived edges close a dependency cycle; the previous actions are
    /// left in place.
    pub fn set_actions(&mut self, actions: Vec<Action>) -> Result<(), BlockGraphError> {
        self.set_actions_with_limits(actions, crate::ModuleCertificationLimits::DEFAULT)
    }

    pub(crate) fn set_actions_with_limits(
        &mut self,
        actions: Vec<Action>,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<(), BlockGraphError> {
        let dag = self.build_action_graph(&actions)?;
        if actions.is_empty() || crate::validate::validate_structure(self).is_err() {
            // Construction may attach actions before topology is complete.
            // Full validation/an editor refresh analyzes them once structure holds.
            self.install_action_graph(dag, None);
            return Ok(());
        }

        // Physical analysis needs the candidate installed, so swap it in and
        // roll back if it turns out to be unanalyzable.
        let previous = std::mem::replace(&mut self.action_graph, dag);
        let previous_error = self.action_graph_error.take();
        if let Err(err) = self.analyze_installed_actions(limits) {
            self.install_action_graph(previous, previous_error);
            return Err(err);
        }
        Ok(())
    }

    /// Replaces actions while declaring external Boolean inputs.
    ///
    /// Module bodies use these names for values bound by their parent. A
    /// standalone graph compiler has no producer for them.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`set_actions`](Self::set_actions), plus
    /// [`InvalidActionError::InvalidActionName`] for an invalid input name.
    pub fn set_actions_with_inputs(
        &mut self,
        actions: Vec<Action>,
        inputs: impl IntoIterator<Item = String>,
    ) -> Result<(), BlockGraphError> {
        self.set_actions_with_inputs_and_limits(
            actions,
            inputs,
            crate::ModuleCertificationLimits::DEFAULT,
        )
    }

    pub(crate) fn set_actions_with_inputs_and_limits(
        &mut self,
        actions: Vec<Action>,
        inputs: impl IntoIterator<Item = String>,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<(), BlockGraphError> {
        self.set_actions_with_inputs_impl(actions, inputs, false, limits)
    }

    pub(crate) fn set_actions_deferred_with_inputs(
        &mut self,
        actions: Vec<Action>,
        inputs: impl IntoIterator<Item = String>,
    ) -> Result<(), BlockGraphError> {
        self.set_actions_with_inputs_impl(
            actions,
            inputs,
            true,
            crate::ModuleCertificationLimits::DEFAULT,
        )
    }

    fn set_actions_with_inputs_impl(
        &mut self,
        actions: Vec<Action>,
        inputs: impl IntoIterator<Item = String>,
        deferred: bool,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<(), BlockGraphError> {
        let inputs = inputs.into_iter().collect::<BTreeSet<_>>();
        if let Some(name) = inputs
            .iter()
            .find(|name| !crate::parser::is_valid_identifier(name))
        {
            return Err(InvalidActionError::InvalidActionName(name.clone()).into());
        }
        let previous = std::mem::replace(&mut self.action_inputs, inputs);
        let result = if deferred {
            self.set_actions_deferred(actions)
        } else {
            self.set_actions_with_limits(actions, limits)
        };
        if let Err(err) = result {
            self.action_inputs = previous;
            return Err(err);
        }
        Ok(())
    }

    /// Replaces actions after source validation, deferring stabilizer surfaces
    /// and implicit dependencies to [`analyze_actions`](Self::analyze_actions).
    ///
    /// # Errors
    ///
    /// Returns a [`BlockGraphError`] for invalid source semantics without
    /// changing the previous actions.
    pub fn set_actions_deferred(&mut self, actions: Vec<Action>) -> Result<(), BlockGraphError> {
        let dag = self.build_action_graph(&actions)?;
        self.install_action_graph(dag, None);
        Ok(())
    }

    /// Replaces all actions without failing on a graph-constraint violation,
    /// recording it in [`action_graph_error`](Self::action_graph_error) instead.
    ///
    /// This supports incremental editing, where an action list can be temporarily
    /// incomplete. Names and nonempty feedback targets remain strict so actions
    /// can survive a BLOG round-trip.
    ///
    /// Callers that must not install a broken action set use
    /// [`set_actions`](Self::set_actions).
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError`] if a `Measure`/`Let` name is invalid or feedback
    /// has no targets; the previous actions are left in place.
    pub fn set_actions_lenient(&mut self, actions: Vec<Action>) -> Result<(), BlockGraphError> {
        for action in &actions {
            Self::validate_action_syntax(action)?;
        }
        self.rebuild_action_graph_lenient(actions);
        Ok(())
    }

    /// Rejects action syntax that cannot survive a BLOG round-trip.
    ///
    /// `Measure`/`Let` names are emitted verbatim by the writer, so a name
    /// outside the BLOG identifier grammar (or a reserved keyword) would fail to
    /// reparse — the same round-trip hazard guarded for tags. Variable
    /// *references* need no check here: an undefined one is caught by the DAG
    /// build, and a defined one is validated at its `Measure`/`Let` site.
    pub(crate) fn validate_action_syntax(action: &Action) -> Result<(), BlockGraphError> {
        if matches!(action, Action::Feedback { targets, .. } if targets.is_empty()) {
            return Err(InvalidActionError::EmptyFeedbackTargets.into());
        }
        let name = match action {
            Action::Measure { name, .. } | Action::Let { name, .. } => name,
            _ => return Ok(()),
        };
        if crate::parser::is_valid_identifier(name) {
            Ok(())
        } else {
            Err(InvalidActionError::InvalidActionName(name.clone()).into())
        }
    }

    /// Removes all actions.
    pub fn clear_actions(&mut self) {
        self.action_inputs.clear();
        self.install_action_graph(ActionDag::default(), None);
    }

    fn refresh_action_graph_after_mutation(&mut self) {
        // A mutation can leave the actions inconsistent with the new topology.
        // These `()`-returning mutators cannot surface that error, so the
        // lenient rebuild keeps the actions and records the failure in
        // `action_graph_error` for callers to inspect.
        let actions = self.actions();
        self.rebuild_action_graph_lenient(actions);
    }

    /// Returns a new [`BlockGraph`] shifted by the given offset.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if any shifted block
    /// footprint, pipe endpoint, or action position exceeds the coordinate
    /// range.
    pub fn shift_positions(&self, offset: IVec3) -> Result<Self, BlockGraphError> {
        if offset == IVec3::ZERO {
            return Ok(self.clone());
        }
        self.require_flat_source("position shifting")?;
        let mut g = Self::new();
        g.action_inputs = self.action_inputs.clone();
        for block in self.blocks() {
            g.try_add_block(block.try_with_shift(offset)?)?;
        }
        for pipe in self.pipes() {
            g.try_add_pipe(pipe.try_with_shift(offset)?)?;
        }
        g.branches =
            self.transformed_branches(|position| crate::checked_add_position(position, offset))?;
        let shifted_actions = self
            .action_graph
            .ordered_nodes()
            .map(|node| node.action.try_with_shift(offset))
            .collect::<Result<Vec<_>, _>>()?;
        // A degraded source graph (see `action_graph_error`) shifts to an
        // equally degraded result instead of panicking.
        g.rebuild_action_graph_lenient(shifted_actions);
        Ok(g)
    }

    /// Returns a new [`BlockGraph`] rotated around the origin along the given axis.
    ///
    /// Rotation preserves blocks, pipes, branches, and actions. Rotations that
    /// cannot represent a block's time-oriented geometry return a typed error.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-orientation, coordinate, or validation error.
    pub fn rotate_about_origin(
        &self,
        axis: UDirection,
        quarter_turns: i32,
    ) -> Result<Self, BlockGraphError> {
        let rotated = self.rotate_about_origin_lenient(axis, quarter_turns)?;
        rotated.validate()?;
        Ok(rotated)
    }

    /// Rotates the graph without validating the result.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-orientation or coordinate-overflow error.
    pub fn rotate_about_origin_lenient(
        &self,
        axis: UDirection,
        quarter_turns: i32,
    ) -> Result<Self, BlockGraphError> {
        self.with_orientation_lenient(crate::ModuleRotation::new(axis, quarter_turns).orientation())
    }

    /// Applies a composed cubic orientation without validating action semantics.
    #[doc(hidden)]
    pub fn with_orientation_lenient(
        &self,
        orientation: crate::ModuleOrientation,
    ) -> Result<Self, BlockGraphError> {
        if orientation != crate::ModuleOrientation::IDENTITY {
            self.require_flat_source("rotation")?;
        }
        validate_block_orientation(
            self.blocks().chain(
                self.branches
                    .iter()
                    .flat_map(|branch| branch.on_false().blocks().chain(branch.on_true().blocks())),
            ),
            orientation,
        )?;
        if orientation == crate::ModuleOrientation::IDENTITY {
            return Ok(self.clone());
        }
        let mut rotated = Self::new();
        rotated.action_inputs = self.action_inputs.clone();

        for block in self.blocks() {
            rotated.try_add_block(oriented_block(block, orientation)?)?;
        }

        for pipe in self.pipes() {
            rotated.try_add_pipe(oriented_pipe(pipe, orientation)?)?;
        }

        rotated.branches = self
            .branches
            .iter()
            .map(|branch| branch.try_with_orientation(orientation))
            .collect::<Result<_, _>>()?;
        let actions = self
            .action_graph
            .ordered_nodes()
            .map(|node| node.action.try_with_orientation(orientation))
            .collect::<Result<Vec<_>, _>>()?;
        rotated.rebuild_action_graph_lenient(actions);
        Ok(rotated)
    }

    /// Returns a new [`BlockGraph`] with X/Z bases flipped.
    ///
    /// Swaps block and feedback bases and negates selective resolve conditions
    /// to preserve their ordered choices in the canonical XY/XZ/YZ representation.
    /// Fixed T/Y resources retain their kind; each incident pipe toggles its
    /// Hadamard. A selective's Y arm also needs its outcome sign flipped because
    /// H conjugation sends Y to -Y; Pauli feedback along the other arm's axis
    /// supplies that sign without changing its X/Z outcome. Positions, tags,
    /// and structural branch conditions are retained.
    ///
    /// # Errors
    ///
    /// Returns a flatten-required error for an authored module source, and
    /// [`BlockGraphError::BasisFlipUnsupportedMultiplexPort`] for a multiplex
    /// Port: its split stays Z while the flipped computation would meet it in X.
    pub fn flip_xz_basis_lenient(&self) -> Result<Self, BlockGraphError> {
        self.require_flat_source("basis flipping")?;
        if let Some(port) = self
            .blocks()
            .filter(|block| block.port_role() == Some(crate::PortRole::Multiplex))
            .map(Block::pos)
            .min_by_key(IVec3::to_array)
        {
            return Err(BlockGraphError::BasisFlipUnsupportedMultiplexPort(port));
        }
        let mut flipped = self.clone();
        for block in flipped.inner.node_weights_mut() {
            block.kind = block.kind.flip_xz_basis();
        }
        for pipe in flipped.inner.edge_weights_mut() {
            flip_pipe_xz_basis(pipe, |pos| {
                self.get_block(pos)
                    .is_some_and(|block| matches!(block.kind(), BlockKind::T | BlockKind::Y))
            });
        }
        flipped.branches = self
            .branches
            .iter()
            .map(|branch| branch.flip_xz_basis(self))
            .collect();
        let mut actions = self.actions();
        for action in &mut actions {
            if let Action::Resolve { condition, .. } = action {
                *condition = condition.clone().negated();
            }
            if let Action::Feedback { targets, .. } = action {
                for target in targets {
                    target.pauli = match target.pauli {
                        PauliBasis::X => PauliBasis::Z,
                        PauliBasis::Y => PauliBasis::Y,
                        PauliBasis::Z => PauliBasis::X,
                    };
                }
            }
        }
        let mut y_signs = flipped
            .blocks()
            .filter_map(|block| {
                let pauli = match block.kind() {
                    BlockKind::Selective(crate::SelectiveKind::XY) => PauliBasis::X,
                    BlockKind::Selective(crate::SelectiveKind::YZ) => PauliBasis::Z,
                    _ => return None,
                };
                Some(crate::FeedbackTarget {
                    pauli,
                    target: block.pos(),
                    direction: None,
                })
            })
            .collect::<Vec<_>>();
        y_signs.sort_unstable_by_key(|target| target.target.to_array());
        if !y_signs.is_empty() {
            let correction = Action::Feedback {
                targets: y_signs,
                condition: None,
            };
            // Adjacent identical Pauli operations cancel. In particular, a
            // second basis flip cancels the first flip's Y-sign correction.
            if actions.last() == Some(&correction) {
                actions.pop();
            } else {
                actions.push(correction);
            }
        }
        flipped.rebuild_action_graph_lenient(actions);
        Ok(flipped)
    }

    /// Returns a validated copy with X/Z bases flipped, including the Hadamard
    /// frame on every fixed T/Y resource's incident pipe and selective Y signs.
    ///
    /// # Errors
    ///
    /// Returns an error if the flipped graph or its actions are invalid.
    pub fn flip_xz_basis(&self) -> Result<Self, BlockGraphError> {
        fn is_missing_resolve(err: &BlockGraphError) -> bool {
            matches!(
                err,
                BlockGraphError::InvalidAction(
                    InvalidActionError::MissingResolveForSelective { .. }
                )
            )
        }

        let mut flipped = self.flip_xz_basis_lenient()?;
        let actions = flipped.actions();
        match flipped
            .set_actions(actions)
            .and_then(|()| flipped.validate())
        {
            Ok(()) => Ok(flipped),
            Err(err) if is_missing_resolve(&err) => {
                flipped.clear_actions();
                flipped.validate()?;
                Ok(flipped)
            }
            Err(err) => Err(err),
        }
    }

    /// Returns a new [`BlockGraph`] with all blocks shifted in the z-axis
    /// so that the minimum z-coordinate is zero.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateNormalizationOverflow`] if the
    /// normalized z span cannot be represented by `i32` coordinates.
    pub fn with_zero_min_z(&self) -> Result<Self, BlockGraphError> {
        self.require_flat_source("time normalization")?;
        let Some((_, _, z_span)) = self.spans() else {
            return Ok(self.clone());
        };
        let min_z = *z_span.start();
        let max_z = *z_span.end();
        if min_z == 0 {
            return Ok(self.clone());
        }
        if i64::from(max_z) - i64::from(min_z) > i64::from(i32::MAX) {
            return Err(BlockGraphError::CoordinateNormalizationOverflow { min_z, max_z });
        }

        let normalize = |position| normalize_z_position(position, min_z, max_z);
        let mut normalized = Self::new();
        normalized.action_inputs = self.action_inputs.clone();
        for block in self.blocks() {
            let mut block = block.clone();
            block.pos = normalize(block.pos)?;
            normalized.try_add_block(block)?;
        }
        for pipe in self.pipes() {
            let mut pipe = pipe.clone();
            pipe.src = normalize(pipe.src)?;
            normalized.try_add_pipe(pipe)?;
        }
        normalized.branches = self.transformed_branches(normalize)?;
        let actions = self
            .action_graph
            .ordered_nodes()
            .map(|node| normalize_action_z(&node.action, min_z, max_z))
            .collect::<Result<Vec<_>, _>>()?;
        normalized.rebuild_action_graph_lenient(actions);
        Ok(normalized)
    }

    fn all_reachable_definitions(&self, predicate: impl Fn(&Self) -> bool) -> bool {
        let mut pending = vec![self];
        let mut seen = HashSet::new();
        while let Some(graph) = pending.pop() {
            if !seen.insert(graph.name.as_str()) {
                continue;
            }
            if !predicate(graph) {
                return false;
            }
            for instance in &graph.instances {
                let Some(child) = self.module(&instance.definition) else {
                    return false;
                };
                pending.push(child);
            }
        }
        true
    }

    fn local_blocks_with_branch_arms(&self) -> impl Iterator<Item = &Block> {
        self.blocks().chain(
            self.branches
                .iter()
                .flat_map(|branch| branch.on_false().blocks().chain(branch.on_true().blocks())),
        )
    }

    /// Returns `true` if this graph and every reachable definition contain no blocks.
    /// Both authored branch arms count. An unresolved definition returns `false`.
    pub fn is_empty(&self) -> bool {
        self.all_reachable_definitions(|graph| {
            graph.local_blocks_with_branch_arms().next().is_none()
        })
    }

    /// Returns the number of blocks in this definition's local geometry.
    pub fn block_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of pipes in this definition's local geometry.
    pub fn pipe_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn count_kind(&self, pred: impl Fn(&BlockKind) -> bool) -> usize {
        self.inner.node_weights().filter(|b| pred(&b.kind)).count()
    }

    /// Returns the number of walking blocks in this definition's local geometry.
    pub fn walking_count(&self) -> usize {
        self.count_kind(BlockKind::is_walking)
    }

    /// Returns the number of patch rotation blocks in this definition's local geometry.
    pub fn patch_rotation_count(&self) -> usize {
        self.count_kind(BlockKind::is_patch_rotation)
    }

    /// Returns the number of Port blocks in this definition's local geometry.
    pub fn port_count(&self) -> usize {
        self.count_kind(BlockKind::is_port)
    }

    /// Returns the number of Y-basis blocks in this definition's local geometry.
    pub fn y_count(&self) -> usize {
        self.count_kind(BlockKind::is_y)
    }

    /// Returns the number of T blocks in this definition's local geometry.
    pub fn t_count(&self) -> usize {
        self.count_kind(BlockKind::is_t)
    }

    /// Returns the number of selective blocks in this definition's local geometry.
    pub fn selective_count(&self) -> usize {
        self.count_kind(BlockKind::is_selective)
    }

    /// Returns this definition's local blocks, without expanding child instances.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.inner.node_weights()
    }

    /// Returns this definition's local pipes, without expanding child instances.
    pub fn pipes(&self) -> impl Iterator<Item = &Pipe> {
        self.inner.edge_weights()
    }

    /// Pipes incident to the block anchored at `position`, including pipes at
    /// an extended block's other endpoints. A missing anchor yields no pipes.
    /// Visits only the block's adjacency, without scanning the graph.
    pub fn pipes_at(&self, position: IVec3) -> impl Iterator<Item = &Pipe> {
        self.block_id
            .get(&position)
            .into_iter()
            .flat_map(|&id| self.inner.edges(NodeIndex::new(id as usize)))
            .map(|edge| edge.weight())
    }

    /// Returns local block positions, without expanding child instances.
    pub fn positions(&self) -> impl Iterator<Item = IVec3> {
        self.inner.node_weights().map(|b| b.pos)
    }

    /// Returns `true` if every reachable definition and branch arm has no T blocks.
    /// An unresolved definition returns `false`.
    pub fn is_clifford(&self) -> bool {
        self.all_reachable_definitions(|graph| {
            graph
                .local_blocks_with_branch_arms()
                .all(|block| !block.kind.is_t())
        })
    }

    /// Returns `true` if every reachable definition and branch arm has a
    /// statically known block kind. An unresolved definition returns `false`.
    pub fn is_rigid(&self) -> bool {
        self.all_reachable_definitions(|graph| {
            graph
                .local_blocks_with_branch_arms()
                .all(|block| !block.kind.is_dynamic())
        })
    }

    /// Returns `true` if the root has a public quantum boundary.
    /// Child Ports consumed by module seams do not make the root open.
    pub fn is_open(&self) -> bool {
        !self.interface.quantum_ports.is_empty()
            || self
                .local_blocks_with_branch_arms()
                .any(|block| block.kind.is_port())
    }

    fn try_get_id(&self, pos: impl Into<IVec3>) -> Result<NodeIndex<u32>, BlockGraphError> {
        let pos = pos.into();
        self.block_id
            .get(&pos)
            .copied()
            .map(NodeIndex::from)
            .ok_or(BlockGraphError::BlockNotFound(pos))
    }

    fn try_get_endpoint_id(
        &self,
        pos: impl Into<IVec3>,
    ) -> Result<NodeIndex<u32>, BlockGraphError> {
        let pos = pos.into();
        if let Some(id) = self.block_id.get(&pos).copied() {
            return Ok(NodeIndex::from(id));
        }

        self.reserved_by_position
            .get(&pos)
            .into_iter()
            .flatten()
            .find_map(|anchor| {
                let id = self
                    .block_id
                    .get(anchor)
                    .copied()
                    .expect("reserved-position index references an existing block");
                let block = self
                    .inner
                    .node_weight(NodeIndex::from(id))
                    .expect("block_id references an existing block");
                block
                    .connectable_offsets()
                    .into_iter()
                    .filter(|offset| *offset != IVec3::ZERO)
                    .any(|offset| block.pos + offset == pos)
                    .then_some(id)
            })
            .map(NodeIndex::from)
            .ok_or(BlockGraphError::BlockNotFound(pos))
    }

    /// Returns a reference to the block at the given position, if it exists.
    pub fn get_block(&self, position: impl Into<IVec3>) -> Option<&Block> {
        self.try_get_id(position)
            .ok()
            .and_then(|id| self.inner.node_weight(id))
    }

    /// Returns the owning block for a connectable endpoint position.
    ///
    /// For ordinary blocks this is the block at `position`; for extended
    /// blocks this also resolves the virtual end position back to the block
    /// stored at its start position.
    pub fn get_endpoint_block(&self, position: impl Into<IVec3>) -> Option<&Block> {
        self.try_get_endpoint_id(position)
            .ok()
            .and_then(|id| self.inner.node_weight(id))
    }

    /// Returns `true` if a block exists at the given position.
    pub fn has_block_at(&self, position: impl Into<IVec3>) -> bool {
        self.block_id.contains_key(&position.into())
    }

    /// Returns `true` if a connectable block endpoint exists at the given position.
    pub fn has_endpoint_at(&self, position: impl Into<IVec3>) -> bool {
        self.try_get_endpoint_id(position).is_ok()
    }

    /// Iterates over every lattice cell reserved by some block.
    pub fn occupied_positions(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.blocks().flat_map(|block| {
            block
                .reserved_offsets()
                .into_iter()
                .map(move |offset| block.pos + offset)
        })
    }

    /// Checks whether `block` can be placed without an illegal footprint overlap.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::BlockPositionOccupied`] if a reserved cell
    /// collides with an existing block that does not permit the overlap.
    pub fn can_place_block(&self, block: &Block) -> Result<(), BlockGraphError> {
        self.can_place_block_except(block, None)
    }

    fn can_place_block_except(
        &self,
        block: &Block,
        ignored_anchor: Option<IVec3>,
    ) -> Result<(), BlockGraphError> {
        let edited_region = ignored_anchor.and_then(|position| {
            self.branches
                .iter()
                .position(|region| region.shown_arm().contains_block(position))
        });
        for occupied in block.checked_reserved_positions()? {
            self.check_visible_occupancy(block, occupied, ignored_anchor)?;
            for (index, region) in self.branches.iter().enumerate() {
                if Some(index) == edited_region {
                    continue;
                }
                for existing in region.on_false().blocks().chain(region.on_true().blocks()) {
                    let overlaps = existing
                        .reserved_offsets()
                        .into_iter()
                        .any(|offset| existing.pos() + offset == occupied);
                    if overlaps
                        && (edited_region.is_some()
                            || !block.kind.allows_reserved_overlap(
                                block.pos,
                                existing.kind(),
                                existing.pos(),
                                occupied,
                            ))
                    {
                        return Err(BlockGraphError::BlockPositionOccupied(occupied));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_visible_occupancy(
        &self,
        block: &Block,
        occupied: IVec3,
        ignored_anchor: Option<IVec3>,
    ) -> Result<(), BlockGraphError> {
        for &anchor in self
            .reserved_by_position
            .get(&occupied)
            .into_iter()
            .flatten()
            .filter(|anchor| Some(**anchor) != ignored_anchor)
        {
            let existing = self
                .get_block(anchor)
                .expect("reserved-position index references an existing block");
            if !block
                .kind
                .allows_reserved_overlap(block.pos, existing.kind, existing.pos, occupied)
            {
                return Err(BlockGraphError::BlockPositionOccupied(occupied));
            }
        }
        Ok(())
    }

    fn index_reserved_positions(&mut self, block: &Block) {
        for position in block
            .checked_reserved_positions()
            .expect("placed block footprint fits the coordinate range")
        {
            self.reserved_by_position
                .entry(position)
                .or_default()
                .push(block.pos);
        }
    }

    fn unindex_reserved_positions(&mut self, block: &Block) {
        for position in block
            .checked_reserved_positions()
            .expect("placed block footprint fits the coordinate range")
        {
            let Some(anchors) = self.reserved_by_position.get_mut(&position) else {
                continue;
            };
            anchors.retain(|anchor| *anchor != block.pos);
            if anchors.is_empty() {
                self.reserved_by_position.remove(&position);
            }
        }
    }

    /// Sets the cube height at `position`, re-checking the resulting footprint.
    ///
    /// Height propagation can grow a cube from one cell to several, so the new
    /// footprint has to clear the same occupancy check a fresh placement would.
    ///
    /// # Errors
    ///
    /// [`BlockGraphError::BlockNotFound`] if nothing is placed at `position`,
    /// [`BlockGraphError::Block`] if the block is not a cube or the footprint
    /// leaves the coordinate range, or
    /// [`BlockGraphError::BlockPositionOccupied`] if the grown footprint
    /// collides.
    ///
    /// # Panics
    ///
    /// Panics if graph indexes change after the initial existence check.
    pub fn set_cube_height(
        &mut self,
        position: IVec3,
        height: CubeHeight,
    ) -> Result<(), BlockGraphError> {
        let existing = self
            .get_block(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?
            .clone();
        let candidate = existing.clone().with_height(height)?;
        self.can_place_block_except(&candidate, Some(position))?;
        self.check_preserved_endpoints(&candidate)?;
        self.unindex_reserved_positions(&existing);
        self.index_reserved_positions(&candidate);
        *self
            .get_block_mut(position)
            .expect("block existence checked above") = candidate;
        self.sync_shown_branch_arm_data();
        self.refresh_action_graph_after_mutation();
        Ok(())
    }

    /// Returns `true` if a pipe exists between the two positions.
    pub fn has_pipe_between(&self, u: impl Into<IVec3>, v: impl Into<IVec3>) -> bool {
        self.get_pipe(u, v).is_some()
    }

    pub(crate) fn get_block_mut(&mut self, position: impl Into<IVec3>) -> Option<&mut Block> {
        self.try_get_id(position)
            .ok()
            .and_then(|index| self.inner.node_weight_mut(index))
    }

    fn check_preserved_endpoints(&self, candidate: &Block) -> Result<(), BlockGraphError> {
        let position = candidate.pos();
        let endpoints = candidate.connectable_offsets();
        for pipe in self.inner.edges(self.try_get_id(position)?) {
            for endpoint in [pipe.weight().src(), pipe.weight().dst()] {
                if self
                    .get_endpoint_block(endpoint)
                    .is_some_and(|owner| owner.pos() == position)
                    && !endpoints
                        .iter()
                        .any(|offset| position + *offset == endpoint)
                {
                    return Err(BlockGraphError::BlockNotFound(endpoint));
                }
            }
        }
        Ok(())
    }

    /// Update a block kind while preserving graph indexes and refreshing action metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the block is missing or the new kind violates geometry or branch rules.
    ///
    /// # Panics
    ///
    /// Panics if graph indexes or an existing cube height are internally invalid.
    pub fn set_block_kind(
        &mut self,
        position: impl Into<IVec3>,
        kind: BlockKind,
    ) -> Result<(), BlockGraphError> {
        let position = position.into();
        if self.get_block(position).is_none() {
            return Err(BlockGraphError::BlockNotFound(position));
        }
        if let Some(region) = self.shown_branch_at(position)
            && (kind.is_dynamic() || kind.is_port())
        {
            return Err(InvalidActionError::InvalidBranchInterface {
                name: region.name.clone(),
                reason: format!("arms cannot contain {kind} blocks"),
            }
            .into());
        }
        let mut candidate = Block::new(position, kind);
        if kind.is_cube() {
            let height = self
                .get_block(position)
                .expect("block existence checked")
                .height();
            candidate = candidate
                .with_height(height)
                .expect("existing cube height is valid");
        }
        self.can_place_block_except(&candidate, Some(position))?;
        self.check_preserved_endpoints(&candidate)?;
        let existing = self
            .get_block(position)
            .expect("block existence checked")
            .clone();
        self.unindex_reserved_positions(&existing);
        let block = self
            .get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?;
        block.set_kind(kind);
        let block = block.clone();
        self.index_reserved_positions(&block);
        self.sync_shown_branch_arm_data();
        self.refresh_action_graph_after_mutation();
        Ok(())
    }

    /// Update a block tag while preserving graph indexes and action metadata.
    /// An empty `tag` clears the tag.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::BlockNotFound`] if no block occupies
    /// `position`, or [`BlockGraphError::InvalidTag`] if `tag` is malformed.
    pub fn set_block_tag(
        &mut self,
        position: impl Into<IVec3>,
        tag: impl Into<String>,
    ) -> Result<(), BlockGraphError> {
        let position = position.into();
        let block = self
            .get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?;
        block.set_tag(tag)?;
        self.sync_shown_branch_arm_data();
        Ok(())
    }

    /// Updates a Port's RGB display color. Alpha remains fixed; default gray is
    /// omitted from serialization.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::BlockNotFound`] if no block occupies
    /// `position`, or [`BlockError::PortColorOnNonPort`](crate::BlockError) if
    /// that block is not a Port.
    pub fn set_port_color(
        &mut self,
        position: impl Into<IVec3>,
        rgb: [u8; 3],
    ) -> Result<(), BlockGraphError> {
        let position = position.into();
        self.get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?
            .set_port_color_checked(rgb)?;
        Ok(())
    }

    /// Assigns the role of a Port block.
    ///
    /// # Errors
    ///
    /// Returns an error if the block is missing or is not a Port.
    pub fn set_port_role(
        &mut self,
        position: impl Into<IVec3>,
        role: crate::PortRole,
    ) -> Result<(), BlockGraphError> {
        let position = position.into();
        self.get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?
            .set_port_role_checked(role)?;
        Ok(())
    }

    /// Add a block to the graph, returning its position.
    ///
    /// This mutates topology and refreshes cached action metadata for any
    /// existing actions.
    ///
    /// # Panics
    ///
    /// Panics if the block's position is already occupied. Use
    /// [`try_add_block`](Self::try_add_block) for a fallible alternative.
    pub fn add_block(&mut self, block: Block) -> IVec3 {
        self.try_add_block(block)
            .unwrap_or_else(|err| panic!("Failed to add block: {err}"))
    }

    /// Adds a block to the graph, returning its position or an error if the position is occupied.
    ///
    /// This mutates topology and refreshes cached action metadata for any
    /// existing actions.
    ///
    /// # Errors
    ///
    /// Returns an error for an occupied or unrepresentable block footprint.
    pub fn try_add_block(&mut self, block: Block) -> Result<IVec3, BlockGraphError> {
        self.try_add_block_with_branch_occupancy(block, None)
    }

    pub(crate) fn try_add_block_with_branch_occupancy(
        &mut self,
        block: Block,
        branch_occupancy: Option<&LinkBranchOccupancy>,
    ) -> Result<IVec3, BlockGraphError> {
        let pos = block.pos;
        if self.block_id.contains_key(&pos) {
            return Err(BlockGraphError::BlockExists(pos));
        }
        if let Some(index) = branch_occupancy {
            self.can_place_block_for_link(&block, index)?;
        } else {
            self.can_place_block(&block)?;
        }
        let indexed = block.clone();
        let index = self.inner.add_node(block);
        self.block_id.insert(pos, index.index() as u32);
        self.index_reserved_positions(&indexed);
        self.refresh_action_graph_after_mutation();
        Ok(pos)
    }

    fn can_place_block_for_link(
        &self,
        block: &Block,
        branch_occupancy: &LinkBranchOccupancy,
    ) -> Result<(), BlockGraphError> {
        for occupied in block.checked_reserved_positions()? {
            self.check_visible_occupancy(block, occupied, None)?;
            for &(existing_pos, existing_kind) in branch_occupancy
                .by_position
                .get(&occupied)
                .into_iter()
                .flatten()
            {
                if !block.kind.allows_reserved_overlap(
                    block.pos,
                    existing_kind,
                    existing_pos,
                    occupied,
                ) {
                    return Err(BlockGraphError::BlockPositionOccupied(occupied));
                }
            }
        }
        Ok(())
    }

    /// Removes and returns the block at the given position, if it exists.
    ///
    /// This mutates topology and refreshes cached action metadata for any
    /// existing actions.
    pub fn remove_block(&mut self, pos: impl Into<IVec3>) -> Option<Block> {
        let id = self.block_id.remove(&pos.into())?;
        let block = self.inner.remove_node(id.into())?;
        self.unindex_reserved_positions(&block);
        self.sync_shown_branch_arm_data();
        self.refresh_action_graph_after_mutation();
        Some(block)
    }

    /// Add a pipe connecting two blocks.
    ///
    /// This mutates topology and refreshes cached action metadata for any
    /// existing actions.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint does not exist or if the pipe already exists.
    /// Use [`try_add_pipe`](Self::try_add_pipe) for a fallible alternative.
    pub fn add_pipe(&mut self, pipe: Pipe) {
        self.try_add_pipe(pipe)
            .unwrap_or_else(|err| panic!("Failed to add pipe: {err}"));
    }

    /// Adds a pipe connecting two blocks, returning an error if either endpoint is missing or the pipe already exists.
    ///
    /// This mutates topology and refreshes cached action metadata for any
    /// existing actions.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid endpoints, missing blocks, or a duplicate pipe.
    pub fn try_add_pipe(&mut self, pipe: Pipe) -> Result<(), BlockGraphError> {
        self.try_add_pipe_with_branch_candidates(pipe, None)
    }

    /// Use a linker's transient ownership index while keeping the same pipe
    /// validation and insertion path as ordinary edits.
    pub(crate) fn try_add_pipe_with_branch_candidates(
        &mut self,
        pipe: Pipe,
        branch_candidates: Option<&[usize]>,
    ) -> Result<(), BlockGraphError> {
        let (u, v) = pipe.try_endpoints()?;
        let u_index = self.try_get_endpoint_id(u)?;
        let v_index = self.try_get_endpoint_id(v)?;
        if self.has_pipe_between(u, v) {
            return Err(BlockGraphError::PipeExists(u, v));
        }
        if let Some(indices) = branch_candidates {
            self.record_added_branch_pipe_for(&pipe, indices.iter().copied());
        } else {
            self.record_added_branch_pipe(&pipe);
        }
        self.inner.add_edge(u_index, v_index, pipe);
        self.refresh_action_graph_after_mutation();
        Ok(())
    }

    /// Finds the graph edge whose pipe endpoints match `u` and `v` in either order.
    fn find_pipe_edge(&self, u: IVec3, v: IVec3) -> Option<EdgeIndex> {
        let u_index = self.try_get_endpoint_id(u).ok()?;
        let v_index = self.try_get_endpoint_id(v).ok()?;
        self.inner
            .edges_connecting(u_index, v_index)
            .find_map(|edge| {
                let pipe = edge.weight();
                let endpoints_match =
                    (pipe.src() == u && pipe.dst() == v) || (pipe.src() == v && pipe.dst() == u);
                endpoints_match.then_some(edge.id())
            })
    }

    /// Removes and returns the pipe between the two positions, if it exists.
    pub fn remove_pipe(&mut self, u: impl Into<IVec3>, v: impl Into<IVec3>) -> Option<Pipe> {
        let removed = self
            .find_pipe_edge(u.into(), v.into())
            .and_then(|edge_index| self.inner.remove_edge(edge_index));
        if removed.is_some() {
            self.sync_shown_branch_arm_data();
            self.refresh_action_graph_after_mutation();
        }
        removed
    }

    /// Returns a reference to the pipe between the two positions, if it exists.
    pub fn get_pipe(&self, u: impl Into<IVec3>, v: impl Into<IVec3>) -> Option<&Pipe> {
        let edge_index = self.find_pipe_edge(u.into(), v.into())?;
        self.inner.edge_weight(edge_index)
    }

    /// Returns the pipe connecting the blocks anchored at `u` and `v`.
    ///
    /// Unlike [`get_pipe`](Self::get_pipe), this accepts block positions rather
    /// than physical pipe endpoints. This matters for multi-cell blocks whose
    /// connected endpoint is offset from the stored block position.
    pub fn get_pipe_between_blocks(
        &self,
        u: impl Into<IVec3>,
        v: impl Into<IVec3>,
    ) -> Option<&Pipe> {
        let u_index = self.try_get_id(u).ok()?;
        let v_index = self.try_get_id(v).ok()?;
        self.inner
            .edges_connecting(u_index, v_index)
            .min_by_key(EdgeRef::id)
            .map(|edge| edge.weight())
    }

    pub(crate) fn get_pipe_mut(
        &mut self,
        u: impl Into<IVec3>,
        v: impl Into<IVec3>,
    ) -> Option<&mut Pipe> {
        let edge_index = self.find_pipe_edge(u.into(), v.into())?;
        self.inner.edge_weight_mut(edge_index)
    }

    /// Update a pipe tag while preserving graph topology and action metadata.
    /// An empty `tag` clears the tag.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::PipeNotFound`] if no pipe connects `u` and
    /// `v`, or [`BlockGraphError::InvalidTag`] if `tag` is malformed.
    pub fn set_pipe_tag(
        &mut self,
        u: impl Into<IVec3>,
        v: impl Into<IVec3>,
        tag: impl Into<String>,
    ) -> Result<(), BlockGraphError> {
        let u = u.into();
        let v = v.into();
        let pipe = self
            .get_pipe_mut(u, v)
            .ok_or(BlockGraphError::PipeNotFound(u, v))?;
        pipe.set_tag(tag)?;
        self.sync_shown_branch_arm_data();
        Ok(())
    }

    /// Update a pipe Hadamard flag and refresh action metadata.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::PipeNotFound`] if no pipe connects `u` and `v`.
    pub fn set_pipe_hadamard(
        &mut self,
        u: impl Into<IVec3>,
        v: impl Into<IVec3>,
        hadamard: bool,
    ) -> Result<(), BlockGraphError> {
        let u = u.into();
        let v = v.into();
        let key = {
            let pipe = self
                .get_pipe_mut(u, v)
                .ok_or(BlockGraphError::PipeNotFound(u, v))?;
            pipe.hadamard = hadamard;
            crate::branch::pipe_key(pipe)
        };
        for region in &mut self.branches {
            region.set_cut_hadamard(key, hadamard);
        }
        self.sync_shown_branch_arm_data();
        self.refresh_action_graph_after_mutation();
        Ok(())
    }

    /// Returns all blocks directly connected to the block owning the given endpoint.
    pub fn neighbors(&self, position: impl Into<IVec3>) -> Vec<&Block> {
        if let Ok(index) = self.try_get_endpoint_id(position.into()) {
            return self
                .inner
                .neighbors(index)
                .filter_map(|neighbor_index| self.inner.node_weight(neighbor_index))
                .collect();
        }
        vec![]
    }

    /// Returns the physical neighbor endpoint positions for pipes attached to
    /// the given connectable endpoint.
    pub fn neighbor_positions(&self, position: impl Into<IVec3>) -> Vec<IVec3> {
        let position = position.into();
        let Ok(index) = self.try_get_endpoint_id(position) else {
            return Vec::new();
        };
        self.inner
            .edges(index)
            .filter_map(|edge| {
                let pipe = edge.weight();
                let dst = pipe.dst();
                if pipe.src == position {
                    Some(dst)
                } else if dst == position {
                    Some(pipe.src)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the number of pipes attached to the given connectable endpoint.
    pub fn degree(&self, position: impl Into<IVec3>) -> usize {
        self.neighbor_positions(position).len()
    }

    /// Returns an iterator over `(block_u, block_v, pipe)` triples for every pipe in the graph.
    pub fn block_pairs_with_pipe(&self) -> impl Iterator<Item = (&Block, &Block, &Pipe)> {
        self.inner.edge_indices().filter_map(|edge_index| {
            let (u_index, v_index) = self.inner.edge_endpoints(edge_index)?;
            let u = self.inner.node_weight(u_index)?;
            let v = self.inner.node_weight(v_index)?;
            let pipe = self.inner.edge_weight(edge_index)?;
            Some((u, v, pipe))
        })
    }

    /// Returns physical endpoint positions and owner blocks for every pipe.
    pub fn pipe_endpoints_with_blocks(
        &self,
    ) -> impl Iterator<Item = (IVec3, IVec3, &Block, &Block, &Pipe)> {
        self.block_pairs_with_pipe().filter_map(|(u, v, pipe)| {
            let src_block = self.get_endpoint_block(pipe.src)?;
            let dst = pipe.dst();
            let dst_block = self.get_endpoint_block(dst)?;

            if src_block.pos == u.pos && dst_block.pos == v.pos {
                Some((pipe.src, dst, u, v, pipe))
            } else if src_block.pos == v.pos && dst_block.pos == u.pos {
                Some((pipe.src, dst, v, u, pipe))
            } else {
                None
            }
        })
    }

    /// Audits structure and source actions, including signed dependencies.
    /// Hierarchical graphs use composed certificates and their C0 contract;
    /// leaves retain the local stabilizer/action audit.
    ///
    /// # Errors
    ///
    /// Returns the first structural, action, or stabilizer-derived validation error.
    pub fn validate(&self) -> Result<(), BlockGraphError> {
        if self.has_module_structure() {
            self.validate_source()?;
            if !self.instances.is_empty() {
                return self
                    .summarize_root(crate::ModuleCertificationLimits::DEFAULT)
                    .map(drop)
                    .map_err(|source| {
                        BlockGraphError::ModuleSource(Arc::new(
                            crate::ModuleError::InvalidGeometry {
                                module: self.name.clone(),
                                source,
                                span: None,
                            },
                        ))
                    });
            }
            return crate::validate::validate(&self.copy_local_geometry().fix_shadowed_faces());
        }
        crate::validate::validate(self)
    }

    /// Validates graph structural constraints without validating action semantics.
    ///
    /// # Errors
    ///
    /// Returns the first structural graph error.
    pub fn validate_structure(&self) -> Result<(), BlockGraphError> {
        crate::validate::validate_structure(self)
    }

    /// Validates graph structure and action-source semantics without deriving
    /// stabilizer surfaces or implicit physical dependencies.
    ///
    /// # Errors
    ///
    /// Returns the first structural or source-action error.
    pub fn validate_source(&self) -> Result<(), BlockGraphError> {
        if self.has_module_structure() {
            self.validate_with_limits(crate::ModuleCertificationLimits::DEFAULT)
                .map_err(|error| BlockGraphError::ModuleSource(Arc::new(error)))?;
            for definition in self.modules() {
                let local = definition.copy_local_geometry().fix_shadowed_faces();
                if definition.instances.is_empty() {
                    crate::validate::validate_source(&local)?;
                } else {
                    // Child seams supply some local block incidence. Their
                    // geometry was checked above; only the owned DAG remains.
                    let mut dag = local.build_action_graph(&local.actions())?;
                    dag.validate(Some(&local))?;
                }
            }
            return Ok(());
        }
        crate::validate::validate_source(self).map(|_| ())
    }

    /// Lists every structural violation rather than stopping at the first, for
    /// callers that need to tell a defect an edit introduced from one the graph
    /// already carried. An empty result means the same as
    /// [`BlockGraph::validate_structure`] returning `Ok`.
    ///
    /// Violations are complete per validation phase, not across phases: an
    /// overlapping footprint or an unresolvable pipe endpoint suppresses the
    /// later geometry checks, which assume those hold.
    pub fn structural_errors(&self) -> Vec<BlockGraphError> {
        crate::validate::structural_errors(self)
    }
}

#[cfg(feature = "gltf")]
impl BlockGraph {
    /// Builds the export glTF scene, optionally overlaying a stabilizer generator.
    fn build_export_gltf(
        &self,
        pipe_length: f32,
        stabilizer: Option<&StabilizerGenerator>,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<GltfData, BlockGraphError> {
        self.require_flat_hierarchy("glTF export")?;
        let mut data = block_graph_as_gltf_data_with_popped_faces(self, pipe_length, popped_faces);
        if !popped_faces.is_empty() {
            data = emphasize_popped_faces(data);
        }
        let editor_world = |point: glam::Vec3| glam::vec3(point.x, point.z, -point.y);
        data = data.map_points(editor_world);
        if let Some(stabilizer) = stabilizer {
            if popped_faces.is_empty() {
                data = data.scale_face_opacity(OUTER_FACE_OPACITY_SCALE);
            }
            let stabilizer_data =
                stabilizer_as_gltf_data(stabilizer, self, pipe_length)?.map_points(editor_world);
            data.extend(stabilizer_data);
        }
        Ok(data)
    }

    fn build_module_export_gltf(
        &self,
        pipe_length: f32,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<(GltfData, Option<crate::ModuleView>), BlockGraphError> {
        let flat = self
            .flatten()
            .map_err(|error| BlockGraphError::ModuleMaterialization(Arc::new(error)))?;
        let modules = crate::ModuleView::from_graph(self, &flat);
        let data = block_graph_as_gltf_data_with_modules(
            &flat,
            pipe_length,
            popped_faces,
            modules.as_ref(),
        )
        .map_points(|point| glam::vec3(point.x, point.z, -point.y));
        Ok((data, modules))
    }

    /// Exports an independent rendering projection colored by module definition.
    ///
    /// Uses the editor's opaque ownership colors and retains ordinary colors on
    /// cross-definition pipes. The authored hierarchy is unchanged.
    ///
    /// # Errors
    ///
    /// Returns a materialization, serialization, or file-I/O error.
    pub fn write_module_gltf_file(
        &self,
        pipe_length: f32,
        path: impl AsRef<std::path::Path>,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<(), BlockGraphError> {
        self.build_module_export_gltf(pipe_length, popped_faces)?
            .0
            .to_file(path)
    }

    /// Exports a module ownership view with independently toggled definition highlights.
    ///
    /// Retains the authored graph and creates an independent flat rendering
    /// projection. The HTML embeds its model and loads model-viewer from a CDN.
    ///
    /// # Errors
    ///
    /// Returns a materialization, serialization, or file-I/O error.
    pub fn write_module_html_viewer(
        &self,
        pipe_length: f32,
        path: impl AsRef<std::path::Path>,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<(), BlockGraphError> {
        let flat = self
            .flatten()
            .map_err(|error| BlockGraphError::ModuleMaterialization(Arc::new(error)))?;
        let modules = crate::ModuleView::from_graph(self, &flat);
        let (data, ranges) =
            module_html_geometry(&flat, pipe_length, popped_faces, modules.as_ref());
        let data = data.map_points(|point| glam::vec3(point.x, point.z, -point.y));
        let mut legend = String::new();
        if let Some(modules) = modules {
            legend.push_str("<details class=\"module-legend\" open><summary>Modules</summary><ul>");
            for (index, module) in modules.modules().iter().enumerate() {
                let color = crate::module_color(index);
                let name = module
                    .name
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");
                let instance_suffix = if module.instance_count == 1 { "" } else { "s" };
                let block_suffix = if module.block_count == 1 { "" } else { "s" };
                legend.push_str(&format!("<li><label><input type=\"checkbox\" data-module=\"{index}\" checked disabled aria-label=\"Highlight {name}\"><i style=\"background:#{:02x}{:02x}{:02x}\"></i>{name}<br>{} instance{instance_suffix} · {} block{block_suffix}</label></li>",
                    color.r, color.g, color.b, module.instance_count, module.block_count));
            }
            legend.push_str("</ul></details>");
        }
        std::fs::write(&path, data.to_module_html_str(&legend, &ranges)?).map_err(|source| {
            BlockGraphError::Io {
                path: path.as_ref().to_path_buf(),
                source: Arc::new(source),
            }
        })
    }

    /// Exports the graph as a `.gltf` file, optionally overlaying a stabilizer generator.
    ///
    /// Uses the editor's orientation and baked diffuse lighting in unlit materials.
    ///
    /// # Errors
    ///
    /// Returns a graph, serialization, or file-I/O error.
    pub fn write_to_gltf_file(
        &self,
        pipe_length: f32,
        path: impl AsRef<std::path::Path>,
        stabilizer: Option<&StabilizerGenerator>,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<(), BlockGraphError> {
        self.build_export_gltf(pipe_length, stabilizer, popped_faces)?
            .to_file(path)
    }

    /// Exports the graph as an HTML viewer with an embedded glTF scene.
    ///
    /// The model data is embedded as a base64 data URI, but the `model-viewer`
    /// renderer is loaded from a CDN, so the page requires network access.
    ///
    /// # Errors
    ///
    /// Returns a graph, serialization, or file-I/O error.
    pub fn write_to_gltf_html_viewer(
        &self,
        pipe_length: f32,
        path: impl AsRef<std::path::Path>,
        stabilizer: Option<&StabilizerGenerator>,
        popped_faces: &[GltfFaceSelector],
    ) -> Result<(), BlockGraphError> {
        let gltf_data = self.build_export_gltf(pipe_length, stabilizer, popped_faces)?;
        let html_str = gltf_data.to_html_str()?;
        std::fs::write(&path, html_str).map_err(|e| BlockGraphError::Io {
            path: path.as_ref().to_path_buf(),
            source: Arc::new(e),
        })
    }
}

impl BlockGraph {
    /// Serializes the complete executable hierarchy as BLOG.
    ///
    /// Procedural graphs infer their interfaces; authored interfaces and helper
    /// definitions are retained exactly. Use `to_blog_body_text` explicitly for
    /// a local geometry/actions projection.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a procedural graph's interface cannot be inferred.
    pub fn to_program_blog_text(&self) -> Result<String, crate::ModuleError> {
        if self.has_module_structure() {
            return Ok(self.to_blog_text());
        }
        self.with_local_export_names()
            .with_inferred_interface()
            .map(|program| program.to_blog_text())
    }

    fn with_local_export_names(&self) -> Self {
        fn rename_expr(expr: &mut crate::Expr, names: &HashMap<String, String>) {
            match expr {
                crate::Expr::Var(name) => {
                    if let Some(local) = names.get(name) {
                        *name = local.clone();
                    }
                }
                crate::Expr::Not(inner) => rename_expr(inner, names),
                crate::Expr::Binary(_, left, right) => {
                    rename_expr(left, names);
                    rename_expr(right, names);
                }
            }
        }

        let mut actions = self.actions();
        let names = self
            .action_inputs
            .iter()
            .chain(actions.iter().filter_map(|action| match action {
                Action::Let { name, .. } | Action::Measure { name, .. } => Some(name),
                _ => None,
            }))
            .chain(self.branches.iter().map(|branch| &branch.name))
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut next = 0;
        let replacements = names
            .iter()
            .filter(|name| name.contains("__") || !crate::parser::is_valid_simple_identifier(name))
            .map(|name| {
                let local = loop {
                    let candidate = format!("flat{next}");
                    next += 1;
                    if !names.contains(&candidate) {
                        break candidate;
                    }
                };
                (name.clone(), local)
            })
            .collect::<HashMap<_, _>>();
        let rename = |name: &String| replacements.get(name).unwrap_or(name).clone();
        let mut graph = self.clone();
        graph.action_inputs = self.action_inputs.iter().map(rename).collect();
        for branch in &mut graph.branches {
            branch.name = rename(&branch.name);
        }
        for action in &mut actions {
            let expr = match action {
                Action::Let { name, expr } => {
                    *name = rename(name);
                    Some(expr)
                }
                Action::Measure { name, .. } => {
                    *name = rename(name);
                    None
                }
                Action::DiscardIf(expr) => Some(expr),
                Action::Resolve { condition, .. } | Action::Branch { condition, .. } => {
                    Some(condition)
                }
                Action::Feedback { condition, .. } => condition.as_mut(),
            };
            if let Some(expr) = expr {
                rename_expr(expr, &replacements);
            }
        }
        graph.rebuild_action_graph_lenient(actions);
        graph
    }

    /// Serializes the graph to the `.blog` text format (BLOG 1.0).
    ///
    /// # Panics
    ///
    /// Panics if internal block, pipe, branch, or action indexes are inconsistent.
    pub fn to_blog_body_text(&self) -> String {
        use std::fmt::Write as _;

        if !self.branches.is_empty() {
            return self.to_blog_text_with_branches();
        }

        let mut blog = String::from("BLOG 1.0\n\n");

        let sorted_id_pos = self.blog_block_ids();

        let has_data = !sorted_id_pos.is_empty() || self.pipes().next().is_some();
        if has_data {
            for (pos, id) in sorted_id_pos {
                let block = self.get_block(pos).expect("block exists in block_id");
                writeln!(blog, "{id}: {block}").expect("writing to String cannot fail");
            }

            let mut sorted_pipes: Vec<Pipe> = self.pipes().cloned().collect();
            sorted_pipes.sort_by_key(|pipe| {
                let src_id = self
                    .get_endpoint_block_id(pipe.src)
                    .expect("pipe src has block_id");
                let dst_pos = pipe.src + pipe.dir.to_ivec3();
                let dst_id = self
                    .get_endpoint_block_id(dst_pos)
                    .expect("pipe dst has block_id");
                (src_id, dst_id)
            });
            for pipe in sorted_pipes {
                writeln!(
                    blog,
                    "{}",
                    pipe.display_with_ids(|position| self.block_id.get(&position).copied())
                )
                .expect("writing to String cannot fail");
            }
        }

        if self.has_actions() {
            if has_data {
                blog.push('\n');
            }
            for node in self.action_graph.ordered_nodes() {
                let action_str = node
                    .action
                    .display_with_ids(|position| self.block_id.get(&position).copied());
                writeln!(blog, "{action_str}").expect("writing to String cannot fail");
            }
        }
        blog
    }

    fn to_blog_text_with_branches(&self) -> String {
        use std::fmt::Write as _;

        let mut blog = String::from("BLOG 1.0\n\n");
        let shown_pipes = self
            .branches
            .iter()
            .flat_map(|region| region.shown_arm().pipes())
            .map(crate::branch::pipe_key)
            .collect::<HashSet<_>>();
        let shared_pipes = self
            .branches
            .iter()
            .flat_map(|region| {
                region
                    .on_false()
                    .pipes()
                    .filter(|pipe| region.on_true().pipes().any(|candidate| candidate == *pipe))
            })
            .cloned()
            .collect::<HashSet<_>>();

        let common = self.blog_block_ids();
        let mut action_ids = HashMap::new();
        let mut next_id = u32::try_from(common.len()).expect("BlockGraph ids fit in u32");
        for (position, id) in common {
            let block = self
                .get_block(position)
                .expect("common block exists in block_id");
            writeln!(blog, "{id}: {block}").expect("writing to String cannot fail");
            action_ids.insert(position, id);
        }

        let mut common_pipes = self
            .pipes()
            .filter(|pipe| !shown_pipes.contains(&crate::branch::pipe_key(pipe)))
            .cloned()
            .collect::<Vec<_>>();
        common_pipes.extend(shared_pipes.iter().cloned());
        common_pipes.sort_by_key(|pipe| {
            let (src, dst) = pipe.endpoints();
            (src.to_array(), dst.to_array())
        });
        for pipe in common_pipes {
            writeln!(
                blog,
                "{}",
                pipe.display_with_ids(|position| action_ids.get(&position).copied())
            )
            .expect("writing to String cannot fail");
        }

        for region in &self.branches {
            writeln!(blog, "branch {} {{", region.name).expect("writing to String cannot fail");
            for (label, arm) in [("false", region.on_false()), ("true", region.on_true())] {
                writeln!(blog, "  {label} {{").expect("writing to String cannot fail");
                let mut arm_ids = HashMap::new();
                for block in arm.blocks() {
                    writeln!(blog, "    {next_id}: {block}")
                        .expect("writing to String cannot fail");
                    arm_ids.insert(block.pos(), next_id);
                    next_id += 1;
                }
                for pipe in arm.pipes().filter(|pipe| !shared_pipes.contains(*pipe)) {
                    writeln!(
                        blog,
                        "    {}",
                        pipe.display_with_ids(|position| {
                            arm_ids
                                .get(&position)
                                .or_else(|| action_ids.get(&position))
                                .copied()
                        })
                    )
                    .expect("writing to String cannot fail");
                }
                writeln!(blog, "  }}").expect("writing to String cannot fail");
            }
            writeln!(blog, "}}").expect("writing to String cannot fail");
        }

        if self.has_actions() {
            blog.push('\n');
            for node in self.action_graph.ordered_nodes() {
                match &node.action {
                    Action::Branch { target, condition } => {
                        let name = &self
                            .branch_by_target(*target)
                            .expect("validated branch action has a named region")
                            .name;
                        writeln!(blog, "resolve {name} if {condition}")
                            .expect("writing to String cannot fail");
                    }
                    action => {
                        let action =
                            action.display_with_ids(|position| action_ids.get(&position).copied());
                        writeln!(blog, "{action}").expect("writing to String cannot fail");
                    }
                }
            }
        }
        blog
    }

    /// Writes the graph to a file in `.blog` text format.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::Io`] if the file cannot be written.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), BlockGraphError> {
        std::fs::write(&path, self.to_blog_text()).map_err(|e| BlockGraphError::Io {
            path: path.as_ref().to_path_buf(),
            source: Arc::new(e),
        })
    }

    /// Parses a block graph from `.blog` text format.
    ///
    /// BLOG is an authoring format, so the result may be structurally
    /// incomplete. Call [`Self::validate`] when a complete graph is required.
    ///
    /// # Errors
    ///
    /// Returns a parse or graph-construction error for invalid BLOG input.
    pub fn from_blog_text(blog_text: &str) -> Result<Self, BlockGraphError> {
        crate::parser::parse_blog_to_graph(blog_text)
    }

    /// Parses executable BLOG or graph-body text, retaining module hierarchy.
    ///
    /// This is the same input model as Python's `BlockGraph.from_text` and the
    /// native compiler. Geometry, authored interfaces, child instances, seams,
    /// bindings, and helper definitions are owned by the returned graph.
    /// Imports require [`Self::load`].
    ///
    /// # Errors
    ///
    /// Returns typed syntax, module, graph-construction, or resource-limit errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_graph::BlockGraph;
    /// let graph = BlockGraph::from_text("BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n}\n")?;
    /// assert_eq!(graph.block_count(), 1);
    /// # Ok::<(), bloq_graph::BlockGraphError>(())
    /// ```
    pub fn from_text(text: &str) -> Result<Self, BlockGraphError> {
        Self::from_text_with_limits(text, crate::ModuleCertificationLimits::DEFAULT)
    }

    /// Parses executable BLOG or graph-body text with explicit source budgets.
    ///
    /// The same limits govern action analysis and hierarchy expansion.
    ///
    /// # Errors
    ///
    /// Returns typed syntax, module, graph-construction, or resource-limit errors.
    pub fn from_text_with_limits(
        text: &str,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Self, BlockGraphError> {
        crate::parser::parse_graph_input(text, limits)
    }

    /// Loads BLOG as a block graph, resolving imports relative to each file.
    ///
    /// This matches Python's `BlockGraph.load`. Module instances and interfaces remain authored; call [`Self::flatten`]
    /// explicitly when a consumer needs one expanded geometry projection.
    ///
    /// # Errors
    ///
    /// Returns typed I/O, parse, import-resolution, graph, or resource-limit errors.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, BlockGraphError> {
        Self::load_with_limits(path, crate::ModuleCertificationLimits::DEFAULT)
    }

    /// Loads BLOG as a graph with explicit source-analysis and expansion budgets.
    ///
    /// The same limits apply to every imported hierarchy.
    ///
    /// # Errors
    ///
    /// Returns typed I/O, parse, import-resolution, graph, or resource-limit errors.
    pub fn load_with_limits(
        path: impl AsRef<std::path::Path>,
        limits: crate::ModuleCertificationLimits,
    ) -> Result<Self, BlockGraphError> {
        crate::parser::load_graph_input(path.as_ref(), limits)
    }

    /// Returns a new graph with cube kinds adjusted so that shadowed pipe faces have consistent basis assignments.
    #[must_use]
    pub fn fix_shadowed_faces(&self) -> Self {
        self.fix_shadowed_faces_with(&[])
    }

    pub(crate) fn fix_shadowed_faces_with(&self, extra_pipes: &[(IVec3, UDirection)]) -> Self {
        let mut extras = HashMap::<IVec3, [usize; 3]>::new();
        for &(position, direction) in extra_pipes {
            extras.entry(position).or_default()[direction.index()] += 1;
        }
        let mut updates: Vec<(IVec3, CubeKind)> = Vec::new();
        for block in self.blocks() {
            let BlockKind::Cube(current_kind) = block.kind else {
                continue;
            };
            // Count pipes per direction
            let mut counts = extras.get(&block.pos()).copied().unwrap_or_default();
            for edge in self.inner.edges(
                self.try_get_id(block.pos)
                    .expect("block from self has valid id"),
            ) {
                counts[edge.weight().dir.as_udirection().index()] += 1;
            }

            let new_kind = shadowed_cube_kind(current_kind, counts);
            if new_kind != current_kind {
                updates.push((block.pos, new_kind));
            }
        }

        let mut new_graph = self.clone();
        let changed = !updates.is_empty();
        for (pos, kind) in updates {
            if let Some(block) = new_graph.get_block_mut(pos) {
                // Cube to cube keeps the height, and so the reserved
                // footprint `reserved_by_position` indexes.
                block.kind = BlockKind::Cube(kind);
            }
        }
        if changed {
            new_graph.sync_shown_branch_arm_data();
            new_graph.refresh_action_graph_after_mutation();
        }
        new_graph
    }

    fn randomly_resolved_selective_kind(&self, block: &Block, rng: &mut impl Rng) -> BlockKind {
        let BlockKind::Selective(selective_kind) = block.kind else {
            return block.kind;
        };
        let chosen_basis = if rng.random::<bool>() {
            selective_kind.pauli_if_true()
        } else {
            selective_kind.pauli_if_false()
        };
        self.resolved_selective_kind_for_basis(chosen_basis)
    }

    fn resolved_selective_kind_for_basis(&self, chosen_basis: PauliBasis) -> BlockKind {
        match chosen_basis {
            PauliBasis::X => BlockKind::Measurement(Basis::X),
            PauliBasis::Y => BlockKind::Y,
            PauliBasis::Z => BlockKind::Measurement(Basis::Z),
        }
    }
}

/// Materializes `plan` under one branch arm selection, or `None` when its
/// guards cannot reach that selection.
fn shown_surfaces(
    plan: &crate::GuardedReadoutPlan,
    assignments: &[(IVec3, bool)],
) -> Result<Option<Vec<StabilizerGenerator>>, BlockGraphError> {
    let source = &plan.topology.source;
    let absent = assignments
        .iter()
        .filter(|(_, present)| !present)
        .map(|&(target, _)| target)
        .collect();
    let Some(values) = plan.topology.branch_selection_values(&absent)? else {
        return Ok(None);
    };
    let shown = if source.branch_definitions().is_empty() {
        source.clone()
    } else {
        // A local plan may belong to a module definition; its source is still
        // one flat body.
        source.project_branches_impl(
            source
                .branch_definitions()
                .iter()
                .map(|branch| (branch.target, !absent.contains(&branch.target))),
            true,
            crate::branch::ProjectionContext::Graph,
        )?
    };
    let zx = ZXGraph::from_block_graph_for_analysis(&shown)?;
    Ok(Some(plan.materialize_surfaces(&zx, &values)))
}

impl std::fmt::Display for BlockGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_blog_text())
    }
}

impl FromStr for BlockGraph {
    type Err = BlockGraphError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_text(s)
    }
}

/// A fixed native resource needs one Hadamard for each flipped endpoint frame.
pub(crate) fn flip_pipe_xz_basis(pipe: &mut Pipe, is_fixed_resource: impl Fn(IVec3) -> bool) {
    pipe.hadamard ^= is_fixed_resource(pipe.src()) ^ is_fixed_resource(pipe.dst());
}

pub(crate) fn validate_block_orientation<'a>(
    blocks: impl IntoIterator<Item = &'a Block>,
    orientation: crate::ModuleOrientation,
) -> Result<(), BlockGraphError> {
    let (mut has_dynamic, mut has_walking_or_patch, mut has_custom_height, mut has_y) =
        (false, false, false, false);
    for block in blocks {
        has_dynamic |= block.kind.is_t() || block.kind.is_selective();
        has_walking_or_patch |= block.kind.is_walking() || block.kind.is_patch_rotation();
        // Any non-default height, not just a multi-cell one: height is a
        // property of the *time* axis, so it stops meaning anything once a
        // rotation swaps time with a spatial axis.
        has_custom_height |= !block.height().is_default();
        has_y |= block.kind.is_y();
    }
    if !orientation.preserves_time_direction() {
        if has_dynamic {
            return Err(BlockGraphError::RotationUnsupportedDynamicBlocks);
        }
        if has_walking_or_patch {
            return Err(BlockGraphError::RotationWalkingRequiresTimeAxis);
        }
        if has_custom_height {
            return Err(BlockGraphError::RotationCubeHeightRequiresTimeAxis);
        }
    }
    if has_y && !orientation.is_coordinate_half_turn() {
        return Err(BlockGraphError::RotationRequiresHalfTurns);
    }
    Ok(())
}

/// Normalize one cube using its complete incident-axis counts. Local geometry
/// contexts use the same rule without normalizing their incomplete outer halo.
pub(crate) fn shadowed_cube_kind(current_kind: CubeKind, counts: [usize; 3]) -> CubeKind {
    let mut new_kind = current_kind;
    match counts.iter().filter(|&&c| c > 0).count() {
        1 if current_kind.is_spatial() => {
            let active_dim = counts
                .iter()
                .position(|&c| c > 0)
                .expect("exactly one active dimension in this match arm");
            // Spatial pass-through: fix the kind to make it not a spatial kind
            if active_dim != 2 && counts[active_dim] == 2 {
                let mut bases = new_kind.bases();
                bases[active_dim] = bases[active_dim].flip();
                new_kind =
                    CubeKind::try_from(bases).expect("flipped basis produces valid CubeKind");
            }
        }
        2 => {
            let mut active_dims = (0..3).filter(|&i| counts[i] > 0).take(2);
            let dim1 = active_dims
                .next()
                .expect("two active dims in this match arm");
            let dim2 = active_dims
                .next()
                .expect("two active dims in this match arm");
            // X-shape/T-shape: use normal basis to decide the block kind
            let mut bases = current_kind.bases();
            let normal_dim = counts
                .iter()
                .position(|c| *c == 0)
                .expect("one inactive dim in two-active match arm");
            let normal_basis = bases[normal_dim];
            let side_basis = normal_basis.flip();
            bases[dim1] = side_basis;
            bases[dim2] = side_basis;
            new_kind = CubeKind::try_from(bases).expect("valid bases produce valid CubeKind");
        }
        _ => {}
    }
    new_kind
}

pub(crate) fn oriented_block(
    block: &Block,
    orientation: crate::ModuleOrientation,
) -> Result<Block, BlockGraphError> {
    let mut rotated = block.clone();
    rotated.pos = orientation.try_rotate_position(block.pos)?;
    rotated.kind = oriented_block_kind(block.kind, orientation);
    for offset in rotated.reserved_offsets() {
        crate::checked_add_position(rotated.pos, offset)?;
    }
    Ok(rotated)
}

pub(crate) fn oriented_pipe(
    pipe: &Pipe,
    orientation: crate::ModuleOrientation,
) -> Result<Pipe, BlockGraphError> {
    let src = orientation.try_rotate_position(pipe.src)?;
    let dir = orientation.rotate_direction(pipe.dir);
    crate::checked_add_position(src, dir.to_ivec3())?;
    Ok(Pipe {
        src,
        dir,
        hadamard: pipe.hadamard,
        tag: pipe.tag.clone(),
    })
}

fn oriented_block_kind(kind: BlockKind, orientation: crate::ModuleOrientation) -> BlockKind {
    match kind {
        BlockKind::Cube(kind) => BlockKind::Cube(
            CubeKind::try_from(orientation.rotate_axis_values(kind.bases()))
                .expect("a cubic orientation permutes valid cube bases"),
        ),
        BlockKind::Walking(kind) => {
            let movement = orientation
                .try_rotate_position(kind.movement_3d())
                .expect("small walking movement remains representable");
            let boundary =
                CubeKind::try_from(orientation.rotate_axis_values(kind.boundary().bases()))
                    .and_then(WalkingBoundaryKind::try_from)
                    .expect("a time-preserving orientation keeps walking boundaries temporal");
            BlockKind::Walking(
                WalkingKind::new(boundary, movement.truncate())
                    .expect("a time-preserving orientation keeps walking movement valid"),
            )
        }
        BlockKind::PatchRotation(kind) => {
            let movement = orientation
                .try_rotate_position(kind.movement_3d())
                .expect("small patch rotation movement remains representable");
            let basis = if orientation
                .rotate_direction(Direction::XPLUS)
                .as_udirection()
                == UDirection::X
            {
                kind.basis()
            } else {
                kind.basis().flip()
            };
            BlockKind::PatchRotation(
                PatchRotationKind::new(basis, movement.truncate())
                    .expect("a time-preserving orientation keeps patch movement valid"),
            )
        }
        _ => kind,
    }
}

pub(crate) fn normalize_z_position(
    position: IVec3,
    min_z: i32,
    max_z: i32,
) -> Result<IVec3, BlockGraphError> {
    let z = i64::from(position.z) - i64::from(min_z);
    let z = i32::try_from(z)
        .map_err(|_| BlockGraphError::CoordinateNormalizationOverflow { min_z, max_z })?;
    Ok(IVec3::new(position.x, position.y, z))
}

fn normalize_action_z(action: &Action, min_z: i32, max_z: i32) -> Result<Action, BlockGraphError> {
    let normalize = |position| normalize_z_position(position, min_z, max_z);
    Ok(match action {
        Action::Measure { target, name } => Action::Measure {
            target: match target {
                MeasureTarget::Node(position) => MeasureTarget::Node(normalize(*position)?),
                MeasureTarget::Edge { src, dir } => {
                    let src = normalize(*src)?;
                    crate::checked_add_position(src, dir.to_ivec3())?;
                    MeasureTarget::Edge { src, dir: *dir }
                }
            },
            name: name.clone(),
        },
        Action::Resolve { target, condition } => Action::Resolve {
            target: normalize(*target)?,
            condition: condition.clone(),
        },
        Action::Branch { target, condition } => Action::Branch {
            target: normalize(*target)?,
            condition: condition.clone(),
        },
        Action::Feedback { targets, condition } => Action::Feedback {
            targets: targets
                .iter()
                .map(|target| {
                    Ok(crate::FeedbackTarget {
                        pauli: target.pauli,
                        target: normalize(target.target)?,
                        direction: target.direction,
                    })
                })
                .collect::<Result<_, BlockGraphError>>()?,
            condition: condition.clone(),
        },
        Action::Let { .. } | Action::DiscardIf(..) => action.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BranchArm, Expr, GalleryItem, SelectiveKind, StabilizerError};

    #[test]
    fn cube_height_edits_preserve_connected_endpoints() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().unwrap())
                .unwrap(),
        );
        graph.add_block(Block::new(2 * IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::Z, Direction::ZPLUS));
        graph.validate().unwrap();
        let before = graph.to_blog_text();
        assert!(matches!(
            graph.set_cube_height(IVec3::ZERO, CubeHeight::DEFAULT),
            Err(BlockGraphError::BlockNotFound(endpoint)) if endpoint == IVec3::Z
        ));
        assert_eq!(graph.to_blog_text(), before);
        graph.to_zx_graph().unwrap();

        // Even an internally corrupted endpoint must fail validation before
        // a conversion indexes the missing endpoint.
        let existing = graph.get_block(IVec3::ZERO).unwrap().clone();
        graph.unindex_reserved_positions(&existing);
        let short = existing.with_height(CubeHeight::DEFAULT).unwrap();
        graph.index_reserved_positions(&short);
        *graph.get_block_mut(IVec3::ZERO).unwrap() = short;
        assert!(matches!(
            graph.validate_structure(),
            Err(BlockGraphError::BlockNotFound(endpoint)) if endpoint == IVec3::Z
        ));
        graph.to_zx_graph().unwrap_err();
    }

    #[test]
    fn executable_flat_exports_round_trip_linked_symbols_without_collisions() {
        for source in [
            GalleryItem::PhaseGradientK4.build(),
            GalleryItem::ThreeBitAdder.build(),
        ] {
            let graph = source
                .materialize_root_graph()
                .expect("gallery flat projection");
            let original_actions = graph.actions();
            let text = graph.to_program_blog_text().unwrap();
            let reparsed = crate::parse_inline_graph(&text)
                .unwrap()
                .materialize_flat_graph()
                .unwrap();
            assert_eq!(reparsed.block_count(), graph.block_count());
            assert_eq!(reparsed.actions().len(), graph.actions().len());
            assert_eq!(reparsed.to_program_blog_text().unwrap(), text);
            assert_eq!(graph.actions(), original_actions);
        }

        let mut graph = BlockGraph::new();
        graph
            .set_actions_with_inputs(
                vec![
                    Action::Let {
                        name: "flat0".into(),
                        expr: crate::Expr::Var("child__input".into()),
                    },
                    Action::Let {
                        name: "child__value".into(),
                        expr: crate::Expr::Not(Box::new(crate::Expr::Var("flat0".into()))),
                    },
                    Action::DiscardIf(crate::Expr::Var("child__value".into())),
                ],
                ["child__input".into()],
            )
            .unwrap();
        let text = graph.to_program_blog_text().unwrap();
        let program = crate::parse_inline_graph(&text).unwrap();
        assert!(text.contains("flat0 = flat1"), "{text}");
        assert!(text.contains("flat2 = !flat0"), "{text}");
        assert_eq!(program.root().interface.bit_inputs, ["flat1"]);
    }

    #[test]
    fn executable_exports_keep_boolean_and_quantum_names_disjoint() {
        for input in ["q0", "source/value", "module"] {
            let mut graph = GalleryItem::CNOT
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            graph
                .set_actions_with_inputs(
                    vec![Action::Let {
                        name: "readout".into(),
                        expr: crate::Expr::Var(input.into()),
                    }],
                    [input.into()],
                )
                .unwrap();
            let exported = graph.to_program_blog_text().unwrap();
            let program = crate::parse_inline_graph(&exported).unwrap();
            let [name] = program.root().interface.bit_inputs.as_slice() else {
                panic!("one external bit survives export");
            };
            assert!(
                program
                    .root()
                    .interface
                    .quantum_ports
                    .iter()
                    .all(|port| port.name != *name)
            );
            assert_eq!(
                program.root().local_body().actions(),
                vec![Action::Let {
                    name: "readout".into(),
                    expr: crate::Expr::Var(name.clone()),
                }]
            );
            assert_eq!(graph.action_graph().inputs().collect::<Vec<_>>(), [input]);
        }
    }

    /// A selective block needs a `resolve`, and that `resolve` needs the
    /// `measure` its condition names, so an interactively built action list is
    /// invalid at every intermediate step. The lenient install has to carry that
    /// violation rather than refuse the edit, and clear it once the list is
    /// complete.
    #[test]
    fn lenient_action_install_carries_an_incomplete_list_then_clears_it() {
        let mut graph = BlockGraph::default();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::X,
            BlockKind::Selective(SelectiveKind::XY),
        ));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let measure = Action::Measure {
            target: MeasureTarget::Node(IVec3::ZERO),
            name: "m0".into(),
        };
        let resolve = Action::Resolve {
            target: IVec3::X,
            condition: crate::Expr::Var("m0".into()),
        };

        assert!(
            graph.set_actions(vec![measure.clone()]).is_err(),
            "the strict install is what forces the lenient one to exist"
        );

        graph
            .set_actions_lenient(vec![measure.clone()])
            .expect("a valid identifier installs even while the list is incomplete");
        assert_eq!(graph.actions(), vec![measure.clone()]);
        assert!(
            graph.action_graph_error().is_some(),
            "the unmet constraint is carried, not dropped"
        );

        graph
            .set_actions_lenient(vec![measure, resolve])
            .expect("the completed list installs");
        assert!(
            graph.action_graph_error().is_none(),
            "completing the list clears the carried violation"
        );
    }

    /// An unusable name is a typo, not a stage of construction, so it is refused
    /// by both installs.
    #[test]
    fn lenient_action_install_still_rejects_an_unroundtrippable_name() {
        let mut graph = BlockGraph::default();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));

        let rejected = graph.set_actions_lenient(vec![Action::Measure {
            target: MeasureTarget::Node(IVec3::ZERO),
            name: "not an identifier".into(),
        }]);

        assert!(rejected.is_err());
        assert!(graph.actions().is_empty());
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn export_geometry_uses_editor_orientation_without_reflecting_faces() {
        let mut graph = BlockGraph::new();
        let position = glam::ivec3(1, 2, 3);
        graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        let source = block_graph_as_gltf_data_with_popped_faces(&graph, 2.0, &[]);
        let exported = graph.build_export_gltf(2.0, None, &[]).unwrap();
        let editor_world = |point: glam::Vec3| glam::vec3(point.x, point.z, -point.y);
        for (color, triangles) in source.triangles {
            let expected: Vec<_> = triangles
                .into_iter()
                .map(|triangle| triangle.map(editor_world))
                .collect();
            assert_eq!(exported.triangles[&color], expected);
        }
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn popped_stabilizer_view_emphasizes_remaining_faces() {
        let graph = GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let stabilizers = graph.stabilizers().expect("CNOT stabilizers");
        let stabilizer = &stabilizers[0];

        let transparent = graph
            .build_export_gltf(2.0, Some(stabilizer), &[])
            .expect("transparent stabilizer view");
        let popped = graph
            .build_export_gltf(
                2.0,
                Some(stabilizer),
                &[GltfFaceSelector::All(Direction::YPLUS)],
            )
            .expect("popped stabilizer view");

        assert!(transparent.triangles.keys().any(|color| color.a < 255));
        assert!(popped.triangles.keys().all(|color| color.a == 255));
        assert!(popped.triangles.contains_key(&POPPED_X_FACE_COLOR));
        assert!(popped.triangles.contains_key(&POPPED_Z_FACE_COLOR));
    }

    #[test]
    fn to_blog_text_serializes_action_dag_in_ordinal_order() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
            .set_actions_lenient(vec![Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();

        let text = graph.to_blog_text();
        assert!(text.starts_with("BLOG 1.0\n"));
        assert!(text.contains("m0 = measure 0\n"));
        assert!(!text.contains("debug"));
        assert!(!text.contains("set_time"));
    }

    #[test]
    fn branch_blog_pipes_use_common_and_arm_local_ids() {
        let past = IVec3::ZERO;
        let inside = IVec3::Z;
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(past, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(
            IVec3::new(2, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        let arm = |tag| {
            BranchArm::new(
                vec![
                    Block::new(inside, BlockKind::Cube(CubeKind::ZXZ)),
                    Block::new(2 * inside, BlockKind::Cube(CubeKind::ZXZ)),
                ],
                vec![
                    Pipe::new(past, Direction::ZPLUS).with_tag(tag).unwrap(),
                    Pipe::new(inside, Direction::ZPLUS).with_tag(tag).unwrap(),
                ],
            )
        };
        let target = graph
            .try_add_branch_region("b", arm("false"), arm("true"))
            .unwrap();
        graph
            .set_actions_deferred(vec![
                Action::Measure {
                    target: MeasureTarget::Node(IVec3::new(2, 0, 0)),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: Expr::Var("m".into()),
                },
            ])
            .unwrap();

        let text = graph.to_blog_body_text();
        assert!(text.contains("  false {\n    2: ZXZ [0, 0, 1]\n    3: ZXZ [0, 0, 2]\n    0 -> +Z <false>\n    2 -> +Z <false>\n  }"));
        assert!(text.contains("  true {\n    4: ZXZ [0, 0, 1]\n    5: ZXZ [0, 0, 2]\n    0 -> +Z <true>\n    4 -> +Z <true>\n  }"));
        assert_eq!(
            BlockGraph::from_blog_text(&text)
                .unwrap()
                .to_blog_body_text(),
            text
        );
        let full = graph.to_blog_text();
        assert!(full.contains("module main {"));
        assert_eq!(BlockGraph::from_text(&full).unwrap().to_blog_text(), full);
    }

    #[test]
    fn to_blog_text_serializes_walking_blocks_with_explicit_end_position() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(1, 0, 3),
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(0, 1)).unwrap(),
            ),
        ));
        graph.add_block(Block::new(glam::ivec3(1, 1, 5), BlockKind::Port));
        graph.add_pipe(Pipe::new(glam::ivec3(1, 1, 4), Direction::ZPLUS));

        let text = graph.to_blog_text();

        assert!(text.contains("0: walk ZXZ [1, 0, 3] -> [1, 1, 4]"));
        assert!(text.contains("[1, 1, 4] -> +Z"));
        assert!(!text.contains("WZXZ"));
    }

    #[test]
    fn tags_survive_blog_text_round_trip() {
        let source = concat!(
            "BLOG 1.0\n\n",
            "0: ZXZ [0, 0, 0] <if>\n",
            "1: ZXZ [1, 0, 0] <R_Z(π/8)^†>\n",
            "0 -> +X <(X⊗X)^(1/2)>\n",
        );
        let graph = BlockGraph::from_blog_text(source).unwrap();
        assert_eq!(graph.to_blog_body_text(), source);
        let full = graph.to_blog_text();
        let restored = BlockGraph::from_text(&full).unwrap();
        assert_eq!(restored.to_blog_body_text(), source);
        assert_eq!(restored.to_blog_text(), full);
    }

    #[test]
    fn tag_edits_preserve_analyzed_actions() {
        let mut graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let snapshot = |graph: &BlockGraph| {
            (
                graph.action_graph().is_analyzed(),
                graph
                    .action_graph()
                    .ordered_nodes()
                    .cloned()
                    .collect::<Vec<_>>(),
                graph.action_graph().dependencies().collect::<Vec<_>>(),
            )
        };
        let actions = snapshot(&graph);
        let block = graph
            .blocks()
            .next()
            .expect("gallery graph has blocks")
            .pos();
        let (src, dst) = graph
            .pipes()
            .next()
            .expect("gallery graph has pipes")
            .endpoints();

        graph.set_block_tag(block, "block-tag").unwrap();
        graph.set_pipe_tag(src, dst, "pipe-tag").unwrap();

        assert_eq!(snapshot(&graph), actions);
    }

    #[test]
    fn shift_positions_preserves_action_dag_contents() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(0, 0, 0)),
                name: "m0".into(),
            })
            .unwrap();

        let shifted = graph
            .shift_positions(glam::ivec3(3, 0, 1))
            .expect("shift fits");
        let ordered = shifted
            .action_graph()
            .ordered_nodes()
            .map(|node| node.action.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![Action::Measure {
                target: MeasureTarget::Node(glam::ivec3(3, 0, 1)),
                name: "m0".into(),
            }]
        );
    }

    #[test]
    fn position_transforms_preserve_external_action_inputs() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            glam::ivec3(0, 0, 3),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph
            .set_actions_with_inputs(
                vec![Action::DiscardIf(crate::Expr::Var("enabled".into()))],
                ["enabled".to_string()],
            )
            .unwrap();

        for transformed in [
            graph.shift_positions(glam::ivec3(1, 0, 0)).unwrap(),
            graph.with_zero_min_z().unwrap(),
        ] {
            assert!(transformed.action_graph_error().is_none());
            assert_eq!(
                transformed.action_graph().inputs().collect::<Vec<_>>(),
                ["enabled"]
            );
            transformed.validate().unwrap();
        }
    }

    #[test]
    fn set_actions_with_inputs_replaces_declared_inputs() {
        let mut graph = BlockGraph::new();
        graph
            .set_actions_with_inputs(Vec::new(), ["old".to_string()])
            .unwrap();

        assert!(matches!(
            graph.set_actions_with_inputs(
                vec![Action::DiscardIf(crate::Expr::Var("old".into()))],
                ["new".to_string()],
            ),
            Err(BlockGraphError::InvalidAction(
                InvalidActionError::UndefinedVariable(name)
            )) if name == "old"
        ));
    }

    #[test]
    fn limited_action_install_restores_actions_and_inputs_after_exhaustion() {
        let continuing = crate::parse_inline_graph(include_str!(
            "../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap()
        .flatten()
        .unwrap();
        assert!(continuing.has_continuing_branches());
        for mut graph in [
            GalleryItem::CNOT
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            continuing,
        ] {
            let previous_actions = graph.actions();
            let previous_nodes = graph
                .action_graph()
                .ordered_nodes()
                .cloned()
                .collect::<Vec<_>>();
            let previous_inputs = graph.action_inputs.clone();
            let mut actions = previous_actions.clone();
            actions.push(Action::Let {
                name: "new_value".into(),
                expr: crate::Expr::Var("new_input".into()),
            });
            let mut inputs = previous_inputs.clone();
            inputs.insert("new_input".into());
            let limits = crate::ModuleCertificationLimits {
                max_local_columns: 0,
                max_boolean_steps: 0,
                ..crate::ModuleCertificationLimits::DEFAULT
            };
            let error = graph
                .set_actions_with_inputs_and_limits(actions.clone(), inputs.clone(), limits)
                .unwrap_err();
            assert!(matches!(
                error,
                BlockGraphError::Stabilizer(crate::StabilizerError::ResourceLimited { .. })
            ));
            assert_eq!(graph.actions(), previous_actions);
            assert_eq!(graph.action_inputs, previous_inputs);
            assert_eq!(
                graph
                    .action_graph()
                    .ordered_nodes()
                    .cloned()
                    .collect::<Vec<_>>(),
                previous_nodes
            );
            assert!(graph.action_graph_error().is_none());
            graph
                .set_actions_with_inputs_and_limits(
                    actions.clone(),
                    inputs.clone(),
                    crate::ModuleCertificationLimits::DEFAULT,
                )
                .unwrap();
            assert_eq!(graph.actions(), actions);
            assert_eq!(graph.action_inputs, inputs);
        }
    }

    #[test]
    fn block_graph_set_actions_enriches_measurements_before_conversion() {
        use crate::{MeasureTarget, MeasurementObservable};
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();

        assert_eq!(
            graph.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::Z))
        );
        assert!(graph.action_graph().is_analyzed());
        assert!(
            graph
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement_stabilizer
                .is_some()
        );
    }

    #[test]
    fn deferred_actions_wait_for_physical_analysis() {
        let mut graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let actions = graph.actions();
        graph.clear_actions();

        graph.set_actions_deferred(actions).unwrap();
        graph.validate_source().unwrap();
        assert!(!graph.action_graph().is_analyzed());

        let graph = graph.analyze_actions().unwrap();
        assert!(graph.action_graph().is_analyzed());
    }

    #[test]
    fn strict_action_install_keeps_previous_list_when_surface_is_impossible() {
        let mut graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let actions = graph.actions();
        for pos in [glam::ivec3(0, 0, 0), glam::ivec3(0, 0, 2)] {
            graph.get_block_mut(pos).unwrap().kind = BlockKind::Cube(CubeKind::XZX);
        }
        graph.clear_actions();

        let err = graph.set_actions(actions).unwrap_err();

        assert!(matches!(
            err,
            BlockGraphError::Stabilizer(StabilizerError::UnavailableControlParity {
                ref mvar,
                ..
            }) if mvar == "mzz"
        ));
        assert!(graph.actions().is_empty());
    }

    #[test]
    fn failed_refresh_after_mutation_records_action_graph_error() {
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();
        assert!(graph.action_graph_error().is_none());

        // Removing the measured block invalidates the action against the new
        // topology; the refresh keeps a best-effort DAG and records the error.
        graph.remove_block(ivec3(0, 0, 0));
        assert!(graph.action_graph_error().is_some());
        assert!(graph.has_actions());

        graph.clear_actions();
        assert!(graph.action_graph_error().is_none());
    }

    #[test]
    fn shift_positions_carries_degradation_instead_of_panicking() {
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(5, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();
        graph.remove_block(ivec3(0, 0, 0));
        assert!(graph.action_graph_error().is_some());

        let shifted = graph.shift_positions(ivec3(1, 0, 0)).expect("shift fits");
        assert!(shifted.action_graph_error().is_some());
        assert!(shifted.has_actions());
    }

    #[test]
    fn shift_positions_translates_actions_in_an_empty_degraded_graph() {
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();
        graph.remove_block(ivec3(0, 0, 0));
        assert!(graph.is_empty());

        let shifted = graph.shift_positions(ivec3(4, 0, 0)).expect("shift fits");
        assert_eq!(
            shifted.actions(),
            [Action::Measure {
                target: MeasureTarget::Node(ivec3(4, 0, 0)),
                name: "m0".into(),
            }]
        );
        assert!(shifted.action_graph_error().is_some());
    }

    #[test]
    fn shift_positions_keeps_lookups_valid_after_removing_lower_block() {
        use glam::ivec3;

        // Removing the lowest-indexed block leaves the source's petgraph node
        // ids non-contiguous; the shift must rebuild lookups from the compacted
        // node ids rather than reusing the stale source ids.
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(2, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.remove_block(ivec3(0, 0, 0));

        let mut shifted = graph.shift_positions(ivec3(10, 0, 0)).expect("shift fits");

        assert_eq!(
            shifted.get_block(ivec3(11, 0, 0)).unwrap().pos,
            ivec3(11, 0, 0)
        );
        assert_eq!(
            shifted.get_block(ivec3(12, 0, 0)).unwrap().pos,
            ivec3(12, 0, 0)
        );
        assert!(shifted.get_block(ivec3(13, 0, 0)).is_none());

        shifted
            .try_add_pipe(Pipe::new(ivec3(11, 0, 0), Direction::XPLUS))
            .unwrap();
        assert!(shifted.has_pipe_between(ivec3(11, 0, 0), ivec3(12, 0, 0)));
    }

    #[test]
    fn validate_and_to_zx_graph_rebuild_measurements_after_kind_mutation() {
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m0".into(),
            }])
            .unwrap();

        graph
            .set_block_kind(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::XZX))
            .unwrap();

        assert_eq!(
            graph.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::X))
        );
        graph.validate().unwrap();
        let zx = graph.to_zx_graph().unwrap();

        assert_eq!(
            zx.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::X))
        );
    }

    #[test]
    fn validate_structure_ignores_action_semantic_errors() {
        use glam::ivec3;

        let measured = ivec3(0, 0, 0);
        let src = ivec3(2, 0, 0);
        let dst = ivec3(3, 0, 0);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(measured, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(src, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(dst, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(src, Direction::XPLUS));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(measured),
                name: "m0".into(),
            }])
            .unwrap();

        graph.remove_block(measured);

        graph.validate_structure().unwrap();
        assert!(graph.validate().is_err());
    }

    #[test]
    fn set_pipe_hadamard_refreshes_cached_edge_measurement_observable() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, -1), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::XPLUS));
        graph
            .set_actions_lenient(vec![Action::Measure {
                target: MeasureTarget::Edge {
                    src: IVec3::new(0, 0, 0),
                    dir: Direction::XPLUS,
                },
                name: "mx".into(),
            }])
            .unwrap();

        assert_eq!(
            graph.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::X))
        );

        graph
            .set_pipe_hadamard(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), true)
            .expect("pipe exists");

        assert_eq!(
            graph.action_graph().node_by_ordinal(0).unwrap().measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::Z))
        );
    }

    #[test]
    fn test_randomly_resolve_selectives_returns_replacement_map() {
        let graph = GalleryItem::CCZFactoryWithTels
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let (resolved, replacements) = graph.randomly_resolve_selectives(rand::random()).unwrap();

        assert!(resolved.blocks().all(|block| !block.kind.is_selective()));
        assert!(!resolved.has_actions());
        assert!(!replacements.is_empty());
        for (original, current) in replacements {
            assert!(original.kind.is_selective());
            assert!(!current.kind.is_selective());
            assert_eq!(original.pos, current.pos);
            assert_eq!(original.tag, current.tag);
            assert_eq!(resolved.get_block(current.pos), Some(&current));
        }
    }

    #[test]
    fn randomly_resolve_selectives_samples_joint_controls_once() {
        let graph = GalleryItem::ToffoliFromAndDelayedCZ
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let joint = [IVec3::new(0, 1, 5), IVec3::new(1, 0, 5)];
        let mut seen = HashSet::new();

        let limit = 2usize
            .saturating_pow(u32::try_from(joint.len()).unwrap_or(u32::MAX))
            .min(crate::ModuleCertificationLimits::DEFAULT.max_guarded_domain_size);
        let domain = graph
            .action_graph()
            .resolve_value_domain_bounded(&joint, limit)
            .expect("two joint sites have at most four resolve tuples");
        assert_eq!(domain.values(), &[vec![false, false], vec![true, true]]);

        for seed in 0..64 {
            let (_, replacements) = graph.randomly_resolve_selectives(seed).unwrap();
            let choices = joint.map(|pos| {
                let original = graph.get_block(pos).expect("joint selective exists");
                let resolved = &replacements[original];
                let BlockKind::Selective(kind) = original.kind else {
                    panic!("joint site is selective");
                };
                resolved.kind == graph.resolved_selective_kind_for_basis(kind.pauli_if_true())
            });
            assert_eq!(
                choices[0], choices[1],
                "seed {seed} split sites controlled by the same mxy value",
            );
            seen.insert(choices[0]);
        }

        assert_eq!(seen, HashSet::from([false, true]));
    }

    #[test]
    fn layer_graph_discards_actions() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(IVec3::new(0, 0, 1)),
                name: "m1".into(),
            })
            .unwrap();

        let layer = graph.layer(0).into_graph();
        assert!(!layer.has_actions());
        assert!(layer.block_count() > 0);
    }

    #[test]
    fn block_layer_view_materializes_boundary_ports() {
        let graph = hand_built_layer_graph();
        let view_graph = graph.layer(0).into_graph();

        let expected_blocks = vec![
            (IVec3::new(0, 0, -1), BlockKind::Port),
            (IVec3::new(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)),
            (IVec3::new(1, 0, 0), BlockKind::Cube(CubeKind::ZXZ)),
            (IVec3::new(1, 0, 1), BlockKind::Port),
        ];
        let expected_pipes = vec![
            (IVec3::new(0, 0, -1), IVec3::new(0, 0, 0), false),
            (IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), false),
            (IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), false),
        ];

        assert_eq!(view_graph.block_count(), 4);
        assert_eq!(view_graph.pipe_count(), 3);
        assert_eq!(block_signatures(&view_graph), expected_blocks);
        assert_eq!(pipe_signatures(&view_graph), expected_pipes);
    }

    #[test]
    fn scaled_cube_occupies_multiple_time_cells_and_exposes_top_endpoint() {
        let mut graph = BlockGraph::new();
        let block = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
            .with_height("3d".parse().expect("valid height"))
            .unwrap();
        graph.add_block(block);

        assert_eq!(
            graph.occupied_positions().collect::<Vec<_>>(),
            vec![
                IVec3::new(0, 0, 0),
                IVec3::new(0, 0, 1),
                IVec3::new(0, 0, 2)
            ]
        );
        assert!(graph.has_endpoint_at(IVec3::new(0, 0, 2)));
        assert_eq!(
            graph
                .get_block(IVec3::ZERO)
                .expect("scaled cube")
                .endpoint_for_direction(Direction::ZPLUS),
            IVec3::new(0, 0, 2)
        );
        assert_eq!(graph.spans().expect("non-empty graph has spans").2, 0..=2);
        assert_eq!(graph.layer(1).blocks().count(), 1);
    }

    #[test]
    fn scaled_cube_footprint_blocks_overlapping_placement() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .unwrap(),
        );

        assert!(matches!(
            graph.try_add_block(Block::new(
                IVec3::new(0, 0, 1),
                BlockKind::Cube(CubeKind::ZXZ)
            )),
            Err(BlockGraphError::BlockPositionOccupied(pos))
                if pos == IVec3::new(0, 0, 1)
        ));
    }

    #[test]
    fn max_height_cube_placement_is_linear_in_footprint_size() {
        let mut graph = BlockGraph::new();
        graph
            .try_add_block(
                Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                    .with_height(CubeHeight::new(crate::MAX_CUBE_HEIGHT_CELLS, 1, 0).unwrap())
                    .unwrap(),
            )
            .unwrap();

        graph
            .try_add_block(
                Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ))
                    .with_height(CubeHeight::new(crate::MAX_CUBE_HEIGHT_CELLS, 1, 0).unwrap())
                    .unwrap(),
            )
            .expect("disjoint maximum-height footprints place without a quadratic scan");
    }

    #[test]
    fn graph_ingress_rejects_coordinate_overflow() {
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 0)).unwrap();
        let mut graph = BlockGraph::new();
        let position = IVec3::new(0, 0, i32::MAX);
        assert!(matches!(
            graph.try_add_block(Block::new(position, BlockKind::Walking(walking))),
            Err(BlockGraphError::CoordinateOverflow { position: p, .. }) if p == position
        ));

        let pipe = Pipe::new(IVec3::new(i32::MAX, 0, 0), Direction::XPLUS);
        assert!(matches!(
            graph.try_add_pipe(pipe),
            Err(BlockGraphError::CoordinateOverflow { .. })
        ));

        assert!(matches!(
            graph.add_action(Action::Measure {
                target: MeasureTarget::Edge {
                    src: IVec3::new(i32::MAX, 0, 0),
                    dir: Direction::XPLUS,
                },
                name: "overflow".into(),
            }),
            Err(BlockGraphError::CoordinateOverflow { .. })
        ));
    }

    #[test]
    fn graph_transforms_report_coordinate_overflow() {
        let mut graph = BlockGraph::new();
        graph
            .try_add_block(Block::new(IVec3::new(i32::MAX, 0, 0), BlockKind::Port))
            .unwrap();
        assert!(matches!(
            graph.shift_positions(IVec3::X),
            Err(BlockGraphError::CoordinateOverflow { .. })
        ));

        let mut rotation = BlockGraph::new();
        rotation
            .try_add_block(Block::new(IVec3::new(i32::MIN, 0, 0), BlockKind::Port))
            .unwrap();
        assert!(matches!(
            rotation.rotate_about_origin(UDirection::Y, 1),
            Err(BlockGraphError::CoordinateRotationOverflow { .. })
        ));
    }

    #[test]
    fn zero_min_z_normalizes_i32_min_and_preserves_graph_content() {
        let mut at_min = BlockGraph::new();
        at_min
            .try_add_block(Block::new(IVec3::new(0, 0, i32::MIN), BlockKind::Port))
            .unwrap();
        at_min
            .try_add_block(Block::new(
                IVec3::new(0, 0, i32::MIN + 1),
                BlockKind::Cube(CubeKind::ZXZ),
            ))
            .unwrap();
        at_min
            .try_add_pipe(Pipe::new(IVec3::new(0, 0, i32::MIN), Direction::ZPLUS))
            .unwrap();
        at_min
            .add_action(Action::Measure {
                target: MeasureTarget::Node(IVec3::new(0, 0, i32::MIN + 1)),
                name: "m".into(),
            })
            .unwrap();

        let normalized = at_min.with_zero_min_z().expect("two-layer span fits");
        assert_eq!(
            normalized.positions().collect::<HashSet<_>>(),
            HashSet::from([IVec3::ZERO, IVec3::Z])
        );
        assert!(normalized.has_pipe_between(IVec3::ZERO, IVec3::Z));
        assert_eq!(
            normalized.actions(),
            [Action::Measure {
                target: MeasureTarget::Node(IVec3::Z),
                name: "m".into(),
            }]
        );
    }

    #[test]
    fn zero_min_z_rejects_unrepresentable_span() {
        let mut too_wide = BlockGraph::new();
        too_wide
            .try_add_block(Block::new(
                IVec3::new(0, 0, -2_000_000_000),
                BlockKind::Cube(CubeKind::ZXZ),
            ))
            .unwrap();
        too_wide
            .try_add_block(Block::new(
                IVec3::new(0, 0, 2_000_000_000),
                BlockKind::Cube(CubeKind::ZXZ),
            ))
            .unwrap();
        assert!(matches!(
            too_wide.with_zero_min_z(),
            Err(BlockGraphError::CoordinateNormalizationOverflow {
                min_z: -2_000_000_000,
                max_z: 2_000_000_000,
            })
        ));
    }

    #[test]
    fn three_quarter_turns_do_not_require_unrepresentable_intermediate_turns() {
        let rotate = |axis, position| {
            crate::ModuleRotation::new(axis, 3)
                .try_rotate_position(position)
                .unwrap()
        };
        assert_eq!(
            rotate(UDirection::X, IVec3::new(0, 0, i32::MIN)),
            IVec3::new(0, i32::MIN, 0)
        );
        assert_eq!(
            rotate(UDirection::Y, IVec3::new(i32::MIN, 0, 0)),
            IVec3::new(0, 0, i32::MIN)
        );
        assert_eq!(
            rotate(UDirection::Z, IVec3::new(0, i32::MIN, 0)),
            IVec3::new(i32::MIN, 0, 0)
        );
    }

    #[test]
    fn layer_graph_materializes_scaled_cube_slice_on_requested_layer() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(Block::new(IVec3::new(0, 0, 3), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 2), Direction::ZPLUS));

        let layer = graph.layer(2).into_graph();
        let block = layer
            .get_block(IVec3::new(0, 0, 2))
            .expect("scaled cube slice appears on occupied layer");

        assert_eq!(block.height_cells(), 1);
        assert_eq!(
            layer.occupied_positions().collect::<Vec<_>>(),
            vec![IVec3::new(0, 0, 2), IVec3::new(0, 0, 3)]
        );
        assert!(layer.has_pipe_between(IVec3::new(0, 0, 2), IVec3::new(0, 0, 3)));
    }

    #[test]
    fn extreme_layer_queries_do_not_overflow() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        assert_eq!(graph.layer(i32::MIN).boundary_pipes().count(), 0);
        assert_eq!(graph.layer(i32::MAX).boundary_pipes().count(), 0);
    }

    #[test]
    fn boundary_pipe_basis_inference_does_not_wrap_perpendicular_neighbors() {
        let lower = IVec3::new(i32::MAX, 0, 0);
        let upper = IVec3::new(i32::MAX, 0, 1);
        let mut graph = BlockGraph::new();
        graph
            .try_add_block(Block::new(lower, BlockKind::Port))
            .unwrap();
        graph
            .try_add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)))
            .unwrap();
        graph
            .try_add_pipe(Pipe::new(lower, Direction::ZPLUS))
            .unwrap();

        graph.validate().expect("boundary pipe remains valid");
        let pipe = graph.pipes().next().unwrap();
        assert!(
            graph
                .infer_pipe_endpoint_face_bases(pipe, upper)
                .into_iter()
                .any(|basis| basis.is_some())
        );

        let invalid = Pipe::new(lower, Direction::XPLUS);
        assert_eq!(graph.infer_pipe_basis(&invalid), [None; 3]);
        assert_eq!(
            graph.infer_pipe_basis_from_endpoint(&invalid, lower),
            [None; 3]
        );
    }

    #[test]
    fn incident_pipes_follow_extended_block_ownership() {
        let mut graph = hand_built_layer_graph();
        let anchor = IVec3::new(4, 0, 0);
        graph.add_block(
            Block::new(anchor, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().unwrap())
                .unwrap(),
        );
        graph.add_block(Block::new(
            anchor + 3 * IVec3::Z,
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(anchor + 2 * IVec3::Z, Direction::ZPLUS));
        for position in graph.positions().chain([IVec3::new(99, 0, 0)]) {
            let mut expected: Vec<_> = graph
                .pipes()
                .filter(|pipe| {
                    [pipe.src(), pipe.dst()].into_iter().any(|endpoint| {
                        graph
                            .get_endpoint_block(endpoint)
                            .is_some_and(|block| block.pos() == position)
                    })
                })
                .map(|pipe| (pipe.src().to_array(), pipe.dst().to_array()))
                .collect();
            let mut actual: Vec<_> = graph
                .pipes_at(position)
                .map(|pipe| (pipe.src().to_array(), pipe.dst().to_array()))
                .collect();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn spatial_port_inference_matches_first_live_pipe_scan() {
        let mut graph = BlockGraph::new();
        let port = IVec3::ZERO;
        graph.add_block(Block::new(port, BlockKind::Port));
        for position in [IVec3::X, -IVec3::X, IVec3::Z, 10 * IVec3::X, 11 * IVec3::X] {
            graph.add_block(Block::new(position, BlockKind::Cube(CubeKind::ZXZ)));
        }
        graph.add_block(Block::new(IVec3::Y, BlockKind::Cube(CubeKind::XZZ)));
        graph.add_pipe(Pipe::new(10 * IVec3::X, Direction::XPLUS));
        graph.add_pipe(Pipe::new(port, Direction::XPLUS).with_hadamard());
        graph.add_pipe(Pipe::new(port, Direction::ZPLUS));
        graph.add_pipe(Pipe::new(-IVec3::X, Direction::XPLUS));
        graph.add_pipe(Pipe::new(port, Direction::YPLUS));

        let check = |graph: &BlockGraph, expected| {
            assert_eq!(graph.infer_spatial_port_cube_kind(port), expected);
        };

        check(&graph, Some(CubeKind::XZX));
        graph.remove_pipe(port, IVec3::X).unwrap();
        check(&graph, None);
        graph.remove_pipe(port, IVec3::Z).unwrap();
        check(&graph, Some(CubeKind::ZXZ));
        graph.remove_pipe(port, -IVec3::X).unwrap();
        check(&graph, Some(CubeKind::XZZ));
        graph.remove_pipe(port, IVec3::Y).unwrap();
        check(&graph, None);
    }

    #[test]
    fn layer_graph_projects_scaled_cube_spatial_pipe_to_requested_layer() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_block(
            Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ))
                .with_height("3d".parse().expect("valid height"))
                .unwrap(),
        );
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let layer = graph.layer(2).into_graph();

        assert!(layer.has_pipe_between(IVec3::new(0, 0, 2), IVec3::new(1, 0, 2)));
    }

    fn hand_built_layer_graph() -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, -1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 1),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, -1), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::XPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 0), Direction::ZPLUS));
        graph
    }

    fn block_signatures(graph: &BlockGraph) -> Vec<(IVec3, BlockKind)> {
        let mut blocks = graph
            .blocks()
            .map(|block| (block.pos, block.kind))
            .collect::<Vec<_>>();
        blocks.sort_by_key(|(pos, kind)| (pos.to_array(), *kind));
        blocks
    }

    fn pipe_signatures(graph: &BlockGraph) -> Vec<(IVec3, IVec3, bool)> {
        let mut pipes = graph
            .block_pairs_with_pipe()
            .map(|(_, _, pipe)| {
                let (src, dst) = pipe.endpoints();
                let (src, dst) = if src.to_array() <= dst.to_array() {
                    (src, dst)
                } else {
                    (dst, src)
                };
                (src, dst, pipe.hadamard)
            })
            .collect::<Vec<_>>();
        pipes.sort_by_key(|(src, dst, hadamard)| (src.to_array(), dst.to_array(), *hadamard));
        pipes
    }

    #[test]
    fn test_selective_resolution_uses_selected_basis() {
        let graph = BlockGraph::new();
        for (basis, expected) in [
            (PauliBasis::X, BlockKind::Measurement(Basis::X)),
            (PauliBasis::Y, BlockKind::Y),
            (PauliBasis::Z, BlockKind::Measurement(Basis::Z)),
        ] {
            assert_eq!(graph.resolved_selective_kind_for_basis(basis), expected);
        }
    }

    #[test]
    fn fixed_measurement_actions_use_the_declared_basis() {
        for basis in [Basis::X, Basis::Z] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, BlockKind::Measurement(basis)));

            assert_eq!(
                graph
                    .resolve_measurement_observable(&MeasureTarget::Node(IVec3::ZERO))
                    .unwrap(),
                MeasurementObservable::Concrete(basis.into()),
            );
        }
    }

    #[test]
    fn rotate_about_origin_lenient_rotates_graph_and_actions() {
        let mut graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0]\n  1: ZXZ [0, 0, 1]\n  [0, 0, 0] -> +Z\n",
        )
        .expect("inline test graph is valid");
        graph
            .set_actions_lenient(vec![Action::Measure {
                target: MeasureTarget::Node(IVec3::Z),
                name: "m0".to_string(),
            }])
            .unwrap();

        let rotated = graph
            .rotate_about_origin_lenient(UDirection::X, 1)
            .expect("rotation succeeds");

        assert!(rotated.has_block_at(IVec3::new(0, 0, 0)));
        assert!(rotated.has_block_at(IVec3::new(0, -1, 0)));
        assert!(rotated.has_pipe_between(IVec3::new(0, 0, 0), IVec3::new(0, -1, 0)));
        assert_eq!(
            rotated
                .get_block(IVec3::new(0, 0, 0))
                .map(|block| block.kind),
            Some(BlockKind::Cube(CubeKind::ZZX))
        );
        assert_eq!(
            rotated
                .get_block(IVec3::new(0, -1, 0))
                .map(|block| block.kind),
            Some(BlockKind::Cube(CubeKind::ZZX))
        );
        assert_eq!(
            rotated.actions(),
            &[Action::Measure {
                target: MeasureTarget::Node(IVec3::NEG_Y),
                name: "m0".to_string(),
            }]
        );
    }

    #[test]
    fn rotate_about_origin_rotates_walking_boundary_around_time_axis() {
        let graph = GalleryItem::GHZSlideThenGlide
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");

        let rotated = graph
            .rotate_about_origin(UDirection::Z, 1)
            .expect("walking gallery should rotate around time axis");

        assert_eq!(
            rotated
                .get_block(IVec3::new(0, 0, 1))
                .map(|block| block.kind),
            Some(BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::XZZ, glam::ivec2(0, 1)).unwrap()
            ))
        );
        assert!(rotated.has_pipe_between(IVec3::new(0, 0, 0), IVec3::new(0, 0, 1)));
    }

    #[test]
    fn rotate_about_origin_rotates_patch_rotation_construction_basis() {
        let kind = PatchRotationKind::new(Basis::X, glam::ivec2(0, 1)).unwrap();
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(0, 1, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(IVec3::ZERO), Direction::ZPLUS));

        let rotated = graph
            .rotate_about_origin(UDirection::Z, 1)
            .expect("patch rotation should rotate around time axis");

        let rotated_kind = match rotated
            .get_block(IVec3::ZERO)
            .expect("rotated patch rotation remains at origin")
            .kind()
        {
            BlockKind::PatchRotation(rotated_kind) => rotated_kind,
            other => panic!("expected patch rotation, got {other:?}"),
        };
        assert_eq!(rotated_kind.movement(), glam::ivec2(-1, 0));
        assert_eq!(rotated_kind.basis(), kind.basis().flip());
    }

    #[test]
    fn rotate_about_origin_requires_half_turns_for_y_graphs() {
        let graph = GalleryItem::YMemory
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");

        assert!(matches!(
            graph.rotate_about_origin(UDirection::X, 1),
            Err(BlockGraphError::RotationRequiresHalfTurns)
        ));
        graph.rotate_about_origin(UDirection::X, 2).unwrap();
    }

    #[test]
    fn rotate_about_origin_only_rotates_dynamic_graphs_around_time() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");

        graph.rotate_about_origin(UDirection::Z, 1).unwrap();
        assert!(matches!(
            graph.rotate_about_origin(UDirection::X, 1),
            Err(BlockGraphError::RotationUnsupportedDynamicBlocks)
        ));
    }

    /// The input port makes both selective arms reachable before and after X/Z flip.
    #[test]
    fn flip_xz_basis_preserves_metadata_and_flips_selective_basis() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Port [0, 0, 0] <source>\n\
             1: XY [0, 0, 1] <selective>\n\
             2: ZXZ [2, 0, 0]\n\
             3: ZXZ [2, 0, 1]\n\
             [0, 0, 0] -H> +Z <link>\n\
             [2, 0, 0] -> +Z\n\n  m0 = measure 3\n  resolve 1 if m0\n",
        )
        .unwrap();

        let flipped = graph.flip_xz_basis().expect("basis flip succeeds");

        assert_eq!(
            flipped
                .get_block(IVec3::new(0, 0, 0))
                .map(|block| (&block.kind, block.tag.as_str(),)),
            Some((&BlockKind::Port, "source"))
        );
        assert_eq!(
            flipped
                .get_block(IVec3::new(0, 0, 1))
                .map(|block| (&block.kind, block.tag.as_str(),)),
            Some((&BlockKind::Selective(SelectiveKind::YZ), "selective",))
        );
        assert_eq!(
            flipped
                .get_pipe(IVec3::new(0, 0, 0), IVec3::new(0, 0, 1))
                .map(|pipe| (pipe.hadamard, pipe.tag.as_str())),
            Some((true, "link"))
        );
        let mut expected_actions = graph.actions();
        if let Action::Resolve { condition, .. } = &mut expected_actions[1] {
            *condition = condition.clone().negated();
        }
        expected_actions.push(Action::Feedback {
            targets: vec![crate::FeedbackTarget {
                pauli: PauliBasis::Z,
                target: IVec3::new(0, 0, 1),
                direction: None,
            }],
            condition: None,
        });
        assert_eq!(flipped.actions(), expected_actions);

        let roundtrip = flipped.flip_xz_basis().expect("second flip succeeds");
        assert_eq!(roundtrip.actions(), graph.actions());
        assert_eq!(
            roundtrip
                .get_block(IVec3::new(0, 0, 1))
                .map(|block| block.kind),
            Some(BlockKind::Selective(SelectiveKind::XY))
        );
    }

    #[test]
    fn basis_flip_swaps_both_selective_choices_but_keeps_branch_conditions() {
        use crate::Expr;
        for kind in [SelectiveKind::XY, SelectiveKind::XZ, SelectiveKind::YZ] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, BlockKind::Selective(kind)));
            let actions = vec![
                Action::Resolve {
                    target: IVec3::ZERO,
                    condition: Expr::Var("control".into()),
                },
                Action::Branch {
                    target: IVec3::X,
                    condition: Expr::Var("control".into()),
                },
            ];
            graph.set_actions_lenient(actions.clone()).unwrap();
            let flipped = graph.flip_xz_basis_lenient().unwrap();
            assert_eq!(flipped.actions()[1], actions[1]);
            assert_eq!(
                flipped.actions()[0],
                Action::Resolve {
                    target: IVec3::ZERO,
                    condition: Expr::Not(Box::new(Expr::Var("control".into())))
                }
            );
            let BlockKind::Selective(flipped_kind) = flipped.get_block(IVec3::ZERO).unwrap().kind
            else {
                unreachable!()
            };
            for control in [false, true] {
                let before = if control {
                    kind.pauli_if_true()
                } else {
                    kind.pauli_if_false()
                };
                let after = if !control {
                    flipped_kind.pauli_if_true()
                } else {
                    flipped_kind.pauli_if_false()
                };
                assert_eq!(
                    after,
                    match before {
                        PauliBasis::X => PauliBasis::Z,
                        PauliBasis::Y => PauliBasis::Y,
                        PauliBasis::Z => PauliBasis::X,
                    }
                );
            }
            assert_eq!(flipped.flip_xz_basis_lenient().unwrap().actions(), actions);
        }
    }

    #[test]
    fn lenient_basis_flip_defers_action_analysis() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let flipped = graph.flip_xz_basis_lenient().unwrap();

        assert_eq!(
            flipped.flip_xz_basis_lenient().unwrap().actions(),
            graph.actions()
        );
        assert!(!flipped.action_graph().is_analyzed());
    }

    #[test]
    fn flip_xz_basis_rebuilds_measurement_metadata_from_flipped_structure() {
        use glam::ivec3;

        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0]\n  1: ZXZ [0, 0, 1]\n  [0, 0, 0] -> +Z\n\n  m0 = measure 1\n",
        )
        .unwrap();

        let flipped = graph.flip_xz_basis().expect("basis flip succeeds");

        assert_eq!(
            flipped.get_block(ivec3(0, 0, 0)).map(|block| block.kind),
            Some(BlockKind::Cube(CubeKind::XZX))
        );
        assert_eq!(
            flipped
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement,
            Some(MeasurementObservable::Concrete(crate::PauliBasis::X))
        );
    }

    #[test]
    fn flip_xz_basis_handles_walking_end_pipe_without_zx_roundtrip_adjacency_error() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n\
             0: XZZ [0, 0, 0]\n\
             1: walk XZX [0, 0, 1] -> [1, 0, 2]\n\
             2: Port [1, 0, 3]\n\
             [0, 0, 0] -> +Z\n\
             [1, 0, 2] -> +Z\n",
        )
        .expect("walking graph should parse");

        let flipped = graph.flip_xz_basis().expect("basis flip succeeds");

        assert_eq!(
            flipped
                .get_block(IVec3::new(0, 0, 1))
                .map(|block| block.kind),
            Some(BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 0)).unwrap()
            ))
        );
        assert!(flipped.has_pipe_between(IVec3::new(1, 0, 2), IVec3::new(1, 0, 3)));
    }

    #[test]
    fn spans_cover_each_axis_inclusively() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::new(-2, 3, 5), BlockKind::Port));
        graph.add_block(Block::new(
            IVec3::new(1, -1, 8),
            BlockKind::Cube(CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(0, 2, 4),
            BlockKind::Cube(CubeKind::XZX),
        ));

        let (x_span, y_span, z_span) = graph.spans().expect("non-empty graph has spans");

        assert_eq!(x_span, -2..=1);
        assert_eq!(y_span, -1..=3);
        assert_eq!(z_span, 4..=8);

        graph
            .set_cube_height(IVec3::new(1, -1, 8), "65535d".parse().unwrap())
            .unwrap();
        let spans = graph.spans().unwrap();
        assert_eq!(spans, (-2..=1, -1..=3, 4..=65_542));
    }
}
