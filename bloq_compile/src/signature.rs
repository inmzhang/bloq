use std::fmt;

use bloq_graph::{
    Basis, Block, BlockGraph, BlockKind, Direction, UDirection, checked_add_position,
};

use crate::CompileError;
use glam::IVec3;

pub(crate) type BlockSignatureMap = crate::FxMap<IVec3, BlockSignature>;

/// Connected faces reported by [`CompileError::InvalidConnectivity`].
///
/// Inspect which faces have pipes and Hadamard walls. Author connections using
/// [`BlockGraph`] and [`bloq_graph::Pipe`]; the compiler derives this diagnostic.
// Each face uses two bits in Direction::index() order: pipe, then Hadamard.
// A Hadamard wall always implies a pipe.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Connectivity(u16);

impl Connectivity {
    /// No connections on any face.
    pub(crate) const ISOLATED: Self = Self(0);

    /// Whether the given face has a pipe.
    pub const fn has_pipe(self, dir: Direction) -> bool {
        self.0 & (1 << (dir.index() * 2)) != 0
    }

    /// Whether the given face has a hadamard pipe.
    /// Only meaningful when [`has_pipe(dir)`](Self::has_pipe) is true.
    pub const fn has_hadamard(self, dir: Direction) -> bool {
        self.0 & (2 << (dir.index() * 2)) != 0
    }

    /// Set a pipe on the given face.
    pub(crate) const fn with_pipe(self, dir: Direction) -> Self {
        Self(self.0 | (1 << (dir.index() * 2)))
    }

    /// Set a plain pipe and clear any Hadamard bit on the face.
    pub(crate) const fn with_plain_pipe(self, dir: Direction) -> Self {
        let shift = dir.index() * 2;
        Self((self.0 & !(0b11 << shift)) | (1 << shift))
    }

    /// Which spatial axes carry a Hadamard wall, as `(x, y)`. A wall on both
    /// axes has no layer schedule that serves it (LIM-017), so callers reject
    /// or skip that combination rather than pick one.
    pub(crate) fn hadamard_wall_axes(self) -> (bool, bool) {
        let on = |axis: [Direction; 2]| axis.into_iter().any(|dir| self.has_hadamard(dir));
        (
            on([Direction::XPLUS, Direction::XMINUS]),
            on([Direction::YPLUS, Direction::YMINUS]),
        )
    }

    /// Set a hadamard pipe on the given face (implies pipe).
    pub(crate) const fn with_hadamard(self, dir: Direction) -> Self {
        Self(self.0 | (0b11 << (dir.index() * 2)))
    }

    /// Iterate over spatial directions that have a pipe connection.
    pub fn spatial_pipes(self) -> impl Iterator<Item = Direction> {
        Direction::iter().filter(move |d| d.is_spatial() && self.has_pipe(*d))
    }

    /// Iterate over all directions that have a pipe connection.
    pub fn pipe_dirs(self) -> impl Iterator<Item = Direction> {
        Direction::iter().filter(move |d| self.has_pipe(*d))
    }

    /// Whether this block has no connections on any face.
    pub const fn is_isolated(self) -> bool {
        self.0 == 0
    }

    /// Whether the given face continues the block's own lattice.
    ///
    /// A spatial Hadamard *cedes* its face to the wall pipe, which owns the
    /// seam-facing cell column outright, so the cube must treat that face as
    /// closed everywhere a plain pipe would open it.
    pub(crate) const fn has_open_edge(self, dir: Direction) -> bool {
        self.has_pipe(dir) && !self.has_hadamard(dir)
    }
}

impl fmt::Debug for Connectivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::ISOLATED {
            return write!(f, "Connectivity(ISOLATED)");
        }
        write!(f, "Connectivity(")?;
        let mut first = true;
        for dir in Direction::iter() {
            if self.has_pipe(dir) {
                if !first {
                    write!(f, " ")?;
                }
                first = false;
                if self.has_hadamard(dir) {
                    write!(f, "{dir}H")?;
                } else {
                    write!(f, "{dir}")?;
                }
            }
        }
        write!(f, ")")
    }
}

/// The CX-slot depth shared by every cube in one z-layer spatial component.
///
/// Depth is a *component-wide* property: merged patches drive shared data
/// qubits, so their syndrome rounds have to line up moment for moment. A
/// component therefore takes the deepest schedule any of its members demands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum LayerSchedule {
    /// 4 slots — the compact schedule, for components of regular cubes only.
    #[default]
    Compact,
    /// 5 slots — the padded schedule a spatial cube's quadrant hooks need.
    Padded,
    /// 6 slots — room for the GHZ bracket of a spatial Hadamard wall's extended
    /// stabilizers on the X axis. Ordinary tiles idle through the bracket
    /// slots.
    Extended,
    /// 6 slots — the diagonal transpose of [`Extended`](Self::Extended), for a
    /// Y-axis spatial Hadamard wall.
    ExtendedY,
}

impl LayerSchedule {
    pub(crate) const fn is_extended(self) -> bool {
        matches!(self, Self::Extended | Self::ExtendedY)
    }
}

/// A block's kind plus its face connectivity, forming a unique compilation signature.
///
/// Two blocks with the same `BlockSignature` produce identical compiled output,
/// enabling caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BlockSignature {
    pub kind: BlockKind,
    /// For cube blocks: the syndrome-extraction rounds the cube's
    /// [`CubeHeight`](bloq_graph::CubeHeight) resolves to at this compilation's
    /// distance. `None` for every other kind.
    ///
    /// The *resolved* count, not the symbolic height: two heights resolving
    /// alike at one distance compile to one template. The footprint half
    /// (`cells`) is deliberately absent — a cube's template is one patch
    /// replayed in time, and which faces are open is already in `connectivity`.
    /// `cube_templates_may_share_rounds_across_different_footprints` pins that.
    pub rounds: Option<u32>,
    pub connectivity: Connectivity,
    /// For Y, Measurement, Port, Selective, and T blocks: the top boundary basis
    /// of the neighboring cube's surface code patch. Determined by the adjacent
    /// cube's y-face basis. `None` for block kinds that do not attach to a
    /// temporal cube seam.
    pub boundary_basis: Option<Basis>,
    /// For cube blocks: the CX-slot depth of the same-layer spatial component
    /// this cube belongs to. `None` for every other kind.
    pub layer_schedule: Option<LayerSchedule>,
    /// For T blocks: the spatial side whose neighbor cell is reserved for the
    /// Steane cultivation spill — picked deterministically from graph
    /// occupancy by `t_surgery_sides`, so it is part of the signature (and thus
    /// the template cache key: the surgery geometry is a rigid transform per
    /// side). Ideal T sources use the canonical side without reserving a cell.
    /// `None` for every other kind.
    pub surgery_side: Option<Direction>,
}

impl BlockSignature {
    /// Patch boundary basis for a Y, Measurement, Port, Selective, or T block,
    /// inherited from its single temporal cube neighbour's top-boundary basis
    /// (`kind.y()`), flipped across a Hadamard pipe.
    pub(crate) fn temporal_neighbor_boundary_basis(
        block: &Block,
        graph: &BlockGraph,
    ) -> Option<Basis> {
        let block_pos = block.pos();
        for dir in [Direction::ZPLUS, Direction::ZMINUS] {
            let Ok(neighbor_endpoint) = checked_add_position(block_pos, dir.to_ivec3()) else {
                continue;
            };
            let pipe = graph.get_pipe(block_pos, neighbor_endpoint);
            if let Some(neighbor) = graph.get_endpoint_block(neighbor_endpoint)
                && let BlockKind::Cube(kind) = neighbor.kind()
                && let Some(pipe) = pipe
            {
                let basis = kind.y();
                return Some(if pipe.is_hadamard() {
                    basis.flip()
                } else {
                    basis
                });
            }
            if let Some(pipe) = pipe {
                return graph.infer_pipe_basis_from_endpoint(pipe, block_pos)[1];
            }
        }
        None
    }

    pub(crate) fn template_connectivity(block: &Block, graph: &BlockGraph) -> Connectivity {
        Self::connectivity(block, graph, false)
    }

    pub(crate) fn graph_connectivity(block: &Block, graph: &BlockGraph) -> Connectivity {
        Self::connectivity(block, graph, true)
    }

    fn connectivity(block: &Block, graph: &BlockGraph, preserve_hadamard: bool) -> Connectivity {
        if let BlockKind::Walking(kind) = block.kind() {
            return Self::two_layer_endpoint_connectivity(
                block.pos(),
                kind.end_position(block.pos()),
                graph,
                preserve_hadamard,
            );
        }
        if let BlockKind::PatchRotation(kind) = block.kind() {
            return Self::two_layer_endpoint_connectivity(
                block.pos(),
                kind.end_position(block.pos()),
                graph,
                preserve_hadamard,
            );
        }

        let mut connectivity = Connectivity::ISOLATED;
        for dir in Direction::iter() {
            let endpoint = block.endpoint_for_direction(dir);
            let Ok(neighbor_pos) = checked_add_position(endpoint, dir.to_ivec3()) else {
                continue;
            };
            if let Some(pipe) = graph.get_pipe(endpoint, neighbor_pos) {
                // Template signatures strip *temporal* Hadamard flags — the
                // realignment pipe node owns that flip, so H-adjacent blocks
                // compile (and cache) as plain. A spatial Hadamard changes its
                // endpoint cube: the cube cedes its seam column to the wall, so
                // that flag must survive in the block signature.
                connectivity = if pipe.is_hadamard() && (preserve_hadamard || dir.is_spatial()) {
                    connectivity.with_hadamard(dir)
                } else {
                    connectivity.with_pipe(dir)
                };
            }
        }
        connectivity
    }

    fn two_layer_endpoint_connectivity(
        start: IVec3,
        end: IVec3,
        graph: &BlockGraph,
        preserve_hadamard: bool,
    ) -> Connectivity {
        let mut connectivity = Connectivity::ISOLATED;
        for (endpoint, dir) in [(start, Direction::ZMINUS), (end, Direction::ZPLUS)] {
            let Ok(neighbor_pos) = checked_add_position(endpoint, dir.to_ivec3()) else {
                continue;
            };
            if let Some(pipe) = graph.get_pipe(endpoint, neighbor_pos) {
                connectivity = if preserve_hadamard && pipe.is_hadamard() {
                    connectivity.with_hadamard(dir)
                } else {
                    connectivity.with_pipe(dir)
                };
            }
        }
        connectivity
    }
}

/// The syndrome rounds a cube compiles to at `distance`, rejecting a height that
/// cannot produce a usable cube.
///
/// A cube's template is an initialization stage, a bulk loop and a measurement
/// stage, so it needs at least two rounds before the loop is even empty; the
/// bulk repetition count is `rounds - 2`. A height like `height=d/2` at `d = 3`
/// clears that, but `height=d-3` at `d = 3` does not.
pub(crate) fn cube_rounds(block: &Block, distance: u32) -> Result<u32, CompileError> {
    let height = block.height();
    let rounds = height.rounds(distance);
    match u32::try_from(rounds) {
        Ok(rounds) if rounds >= 2 => Ok(rounds),
        _ => Err(CompileError::CubeHeightTooShort {
            pos: block.pos(),
            height,
            distance,
            rounds,
        }),
    }
}

pub(crate) fn derive_block_signature(
    graph: &BlockGraph,
    block: &Block,
    distance: u32,
    layer_schedule: LayerSchedule,
    surgery_side: Option<Direction>,
) -> Result<BlockSignature, CompileError> {
    let kind = block.kind();
    let is_cube = matches!(kind, BlockKind::Cube(_));
    let boundary_basis = matches!(
        kind,
        BlockKind::Y
            | BlockKind::Measurement(_)
            | BlockKind::Port
            | BlockKind::Selective(_)
            | BlockKind::T
    )
    .then(|| BlockSignature::temporal_neighbor_boundary_basis(block, graph))
    .flatten();
    Ok(BlockSignature {
        kind,
        rounds: is_cube.then(|| cube_rounds(block, distance)).transpose()?,
        connectivity: BlockSignature::template_connectivity(block, graph),
        boundary_basis,
        layer_schedule: is_cube.then_some(layer_schedule),
        surgery_side,
    })
}

pub(crate) fn block_signatures(
    graph: &BlockGraph,
    layer_schedules: &LayerScheduleMap,
    config: crate::CompileConfig,
) -> Result<BlockSignatureMap, CompileError> {
    let t_surgery_sides = t_surgery_sides(graph, config.prepare_t_with_mpps())?;

    graph
        .blocks()
        .map(|block| {
            let signature = derive_block_signature(
                graph,
                block,
                config.code_distance(),
                layer_schedule_at(layer_schedules, block.pos()),
                t_surgery_sides.get(&block.pos()).copied(),
            )?;
            Ok((block.pos(), signature))
        })
        .collect()
}

/// Per-site signatures used to build relocatable definition objects before global
/// component schedules and T-spill allocation are known. The module compiler
/// expands these local seeds into every finite public-context variant; the
/// linker later selects an exact signature with [`block_signatures`].
pub(crate) fn relocatable_block_signatures(
    graph: &BlockGraph,
    positions: &crate::FxSet<IVec3>,
    distance: u32,
) -> Result<BlockSignatureMap, CompileError> {
    graph
        .blocks()
        .filter(|block| positions.contains(&block.pos()))
        .map(|block| {
            let kind = block.kind();
            let mut signature = derive_block_signature(
                graph,
                block,
                distance,
                LayerSchedule::default(),
                (kind == BlockKind::T)
                    .then_some(crate::block::fixed_bulk::t::SurgeryLayout::SIDE_PRIORITY[0]),
            )?;
            signature.layer_schedule = match kind {
                BlockKind::Cube(cube) => Some(match signature.connectivity.hadamard_wall_axes() {
                    (true, _) => LayerSchedule::Extended,
                    (false, true) => LayerSchedule::ExtendedY,
                    (false, false) if cube.is_spatial() => LayerSchedule::Padded,
                    (false, false) => LayerSchedule::Compact,
                }),
                _ => None,
            };
            Ok((block.pos(), signature))
        })
        .collect()
}

/// Pick each T block's Steane-spill side: a side in
/// [`SurgeryLayout::SIDE_PRIORITY`] whose neighbor cell is neither occupied by a
/// block nor claimed as another T block's spill. The whole neighbor cell is
/// reserved — conservative (the spill is only a margin strip), but it keeps two
/// adjacent T blocks from silently sharing spill space.
/// Ideal preparations need no spill and use one canonical signature side.
///
/// Assignment is deterministic bipartite matching. An augmenting path may move
/// an earlier T off its first choice when a later T needs that cell; unlike
/// backtracking, work stays polynomial in the linked layout size.
pub(crate) fn t_surgery_sides(
    graph: &BlockGraph,
    prepare_t_with_mpps: bool,
) -> Result<crate::FxMap<IVec3, Direction>, CompileError> {
    let t_positions: Vec<IVec3> = graph
        .blocks()
        .filter(|block| block.kind() == BlockKind::T)
        .map(Block::pos)
        .collect();
    if prepare_t_with_mpps {
        // Compact preparation occupies only the normal patch; the side is a cache identity.
        return Ok(t_positions
            .into_iter()
            .map(|position| {
                (
                    position,
                    crate::block::fixed_bulk::t::SurgeryLayout::SIDE_PRIORITY[0],
                )
            })
            .collect());
    }
    assign_linked_t_surgery_sides(t_positions, occupied_branch_positions(graph)?)
}

/// Shared templates must leave room for every authored branch arm.
pub(crate) fn occupied_branch_positions(
    graph: &BlockGraph,
) -> Result<crate::FxSet<IVec3>, CompileError> {
    let mut occupied = graph.occupied_positions().collect::<crate::FxSet<_>>();
    for region in graph.branch_regions()? {
        for block in region.on_false().blocks().chain(region.on_true().blocks()) {
            occupied.extend(
                block
                    .reserved_offsets()
                    .into_iter()
                    .map(|offset| block.pos() + offset),
            );
        }
    }
    Ok(occupied)
}

pub(crate) fn assign_linked_t_surgery_sides(
    mut t_positions: Vec<IVec3>,
    reserved: crate::FxSet<IVec3>,
) -> Result<crate::FxMap<IVec3, Direction>, CompileError> {
    if t_positions.is_empty() {
        return Ok(crate::FxMap::default());
    }
    t_positions.sort_by_key(|pos| (pos.z, pos.x, pos.y));

    let candidates = t_positions
        .iter()
        .map(|&position| {
            crate::block::fixed_bulk::t::SurgeryLayout::SIDE_PRIORITY
                .into_iter()
                .filter_map(|side| {
                    let cell = checked_add_position(position, side.to_ivec3()).ok()?;
                    (!reserved.contains(&cell)).then_some((cell, side))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut owner = crate::FxMap::<IVec3, usize>::default();
    for (index, &position) in t_positions.iter().enumerate() {
        if !augment_surgery_matching(index, &candidates, &mut crate::FxSet::default(), &mut owner) {
            return Err(CompileError::TBlockNeedsFreeNeighbor { pos: position });
        }
    }
    Ok(owner
        .into_iter()
        .map(|(cell, index)| {
            let side = candidates[index]
                .iter()
                .find_map(|&(candidate, side)| (candidate == cell).then_some(side))
                .expect("matched spill cell belongs to its T candidate set");
            (t_positions[index], side)
        })
        .collect())
}

fn augment_surgery_matching(
    index: usize,
    candidates: &[Vec<(IVec3, Direction)>],
    visited: &mut crate::FxSet<IVec3>,
    owner: &mut crate::FxMap<IVec3, usize>,
) -> bool {
    for &(cell, _) in &candidates[index] {
        if !visited.insert(cell) {
            continue;
        }
        let previous = owner.get(&cell).copied();
        if previous
            .is_some_and(|previous| !augment_surgery_matching(previous, candidates, visited, owner))
        {
            continue;
        }
        owner.insert(cell, index);
        return true;
    }
    false
}

pub(crate) type LayerScheduleMap = crate::FxMap<IVec3, LayerSchedule>;

/// The schedule of the cube at `pos`; cubes outside the map are compact.
pub(crate) fn layer_schedule_at(schedules: &LayerScheduleMap, pos: IVec3) -> LayerSchedule {
    schedules.get(&pos).copied().unwrap_or_default()
}

/// Per-cube CX-slot depth, derived per z-layer spatial component.
///
/// Only the components that demand more than [`LayerSchedule::Compact`] are
/// recorded, so the map stays empty for the common case. A component carrying
/// walls on both spatial axes has no schedule at all (LIM-017), so it is
/// rejected here rather than represented.
pub(crate) fn join_component_layer_schedules(
    graph: &BlockGraph,
) -> Result<LayerScheduleMap, CompileError> {
    // Without a spatial cube or Hadamard wall, every component is compact.
    if !graph
        .blocks()
        .any(|block| matches!(block.kind(), BlockKind::Cube(kind) if kind.is_spatial()))
        && !graph
            .pipes()
            .any(|pipe| pipe.is_hadamard() && pipe.dir().is_spatial())
    {
        return Ok(LayerScheduleMap::default());
    }
    let mut schedules = LayerScheduleMap::default();
    let mut mixed = Vec::new();

    // The z-layer spatial partition is `bloq_graph`'s, shared with the cube
    // height propagation that runs at lowering time: both answer "which cubes
    // drive the same data qubits, and therefore share a schedule". A component
    // is named by its lowest member, which is also the position we blame.
    for (layer, components) in graph.cube_layer_components() {
        let root: crate::FxMap<IVec3, IVec3> = components
            .iter()
            .flat_map(|members| members.iter().map(|&pos| (pos, members[0])))
            .collect();

        // A spatial Hadamard wall outranks a spatial cube: its extended
        // stabilizers need the GHZ bracket the padded depth lacks.
        let mut demanded = crate::FxMap::<IVec3, LayerSchedule>::default();
        let mut demand = |component: IVec3, schedule: LayerSchedule| {
            let slot = demanded.entry(component).or_default();
            *slot = (*slot).max(schedule);
        };
        for (&pos, &component) in &root {
            if matches!(
                graph.get_block(pos).map(Block::kind),
                Some(BlockKind::Cube(kind)) if kind.is_spatial()
            ) {
                demand(component, LayerSchedule::Padded);
            }
        }

        let mut wall_axes = crate::FxMap::<IVec3, (bool, bool)>::default();
        for pipe in graph.layer(layer).spacelike_pipes() {
            if !pipe.is_hadamard() {
                continue;
            }
            let (src, dst) = pipe.endpoints();
            let (Some(src), Some(dst)) =
                (graph.get_endpoint_block(src), graph.get_endpoint_block(dst))
            else {
                continue;
            };
            let (Some(&component), Some(_)) = (root.get(&src.pos()), root.get(&dst.pos())) else {
                continue;
            };
            let axes = wall_axes.entry(component).or_default();
            match pipe.dir().as_udirection() {
                UDirection::X => axes.0 = true,
                UDirection::Y => axes.1 = true,
                UDirection::Z => unreachable!("spatial Hadamard walls are not temporal"),
            }
        }
        for (component, (has_x, has_y)) in wall_axes {
            match (has_x, has_y) {
                (true, true) => mixed.push(component),
                (true, false) => demand(component, LayerSchedule::Extended),
                (false, true) => demand(component, LayerSchedule::ExtendedY),
                (false, false) => unreachable!("wall axis map is non-empty"),
            }
        }

        for (&pos, component) in &root {
            if let Some(&schedule) = demanded.get(component) {
                schedules.insert(pos, schedule);
            }
        }
    }

    match mixed.into_iter().min_by_key(glam::IVec3::to_array) {
        Some(pos) => Err(CompileError::MixedSpatialHadamardUnsupported { pos }),
        None => Ok(schedules),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::{Pipe, WalkingBoundaryKind, WalkingKind};
    use rstest::rstest;

    use crate::{CompileConfig, CompileContext};

    /// A hub whose spatial Hadamard walls run on *both* in-plane axes has no
    /// single [`LayerSchedule`], so the compiler must reject it — blaming the
    /// component's lowest coordinate — rather than silently picking one axis.
    /// Compilation with and without the explicit audit must agree.
    #[test]
    fn quarter_turn_spatial_hadamard_is_rejected() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n\
             0: ZZX [0, 0, 0]\n\
             1: XZZ [0, 1, 0]\n\
             2: ZXZ [1, 0, 0]\n\
             3: Port [0, 1, 1]\n\
             4: Port [1, 0, 1]\n\
             [0, 0, 0] -H> +Y\n\
             [0, 0, 0] -H> +X\n\
             [0, 1, 0] -> +Z\n\
             [1, 0, 0] -> +Z\n",
        )
        .expect("quarter-turn graph parses");
        let ctx = CompileContext::new(CompileConfig::new(3));

        assert!(matches!(
            ctx.compile_and_validate(&graph),
            Err(CompileError::MixedSpatialHadamardUnsupported { pos }) if pos == IVec3::ZERO
        ));
        assert!(matches!(
            ctx.compile(&graph),
            Err(CompileError::MixedSpatialHadamardUnsupported { pos }) if pos == IVec3::ZERO
        ));
    }

    #[test]
    fn spatial_cube_pads_its_whole_component() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(
            IVec3::new(1, 0, 0),
            BlockKind::Cube(bloq_graph::CubeKind::ZZX),
        ));
        graph.add_block(Block::new(IVec3::new(2, 0, 0), BlockKind::Y));
        graph.add_block(Block::new(
            IVec3::new(1, 1, 0),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 0), Direction::YPLUS));

        let regular = IVec3::new(0, 0, 0);
        let spatial = IVec3::new(1, 0, 0);
        let connected_regular = IVec3::new(1, 1, 0);
        let y = IVec3::new(2, 0, 0);

        // Exercise the production signature path rather than a per-block helper,
        // so the test covers the same spatial-component set the compiler uses.
        let signatures = block_signatures(
            &graph,
            &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph");

        assert_eq!(
            signatures
                .get(&regular)
                .expect("regular cube signature")
                .layer_schedule,
            Some(LayerSchedule::Compact)
        );
        assert_eq!(
            signatures
                .get(&spatial)
                .expect("spatial cube signature")
                .layer_schedule,
            Some(LayerSchedule::Padded)
        );
        assert_eq!(
            signatures
                .get(&connected_regular)
                .expect("spatial-connected regular cube signature")
                .layer_schedule,
            Some(LayerSchedule::Padded)
        );
        assert_eq!(
            signatures
                .get(&y)
                .expect("Y block signature")
                .layer_schedule,
            None
        );
    }

    /// A cube at `pos` in the z = 0 layer.
    fn cube(pos: IVec3, kind: bloq_graph::CubeKind) -> Block {
        Block::new(pos, BlockKind::Cube(kind))
    }

    #[test]
    fn spatial_hadamard_extends_its_whole_component() {
        use bloq_graph::CubeKind::{XZX, ZXZ};

        let mut graph = BlockGraph::new();
        // A wall between two regular cubes, one of which merges on further.
        graph.add_block(cube(IVec3::ZERO, XZX));
        graph.add_block(cube(IVec3::new(1, 0, 0), XZX));
        graph.add_block(cube(IVec3::new(1, 1, 0), ZXZ));
        // A second component in the same layer, out of the wall's reach.
        graph.add_block(cube(IVec3::new(5, 0, 0), ZXZ));
        graph.add_pipe(Pipe::new(IVec3::new(1, 0, 0), Direction::YPLUS));
        assert!(join_component_layer_schedules(&graph).unwrap().is_empty());
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS).with_hadamard());

        let schedules =
            join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls");
        for pos in [IVec3::ZERO, IVec3::new(1, 0, 0), IVec3::new(1, 1, 0)] {
            assert_eq!(
                layer_schedule_at(&schedules, pos),
                LayerSchedule::Extended,
                "{pos} shares the wall's component"
            );
        }
        assert_eq!(
            layer_schedule_at(&schedules, IVec3::new(5, 0, 0)),
            LayerSchedule::Compact
        );
    }

    #[test]
    fn spatial_hadamard_outranks_a_spatial_cube_in_the_same_component() {
        use bloq_graph::CubeKind::{XXZ, XZX};

        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO, XZX));
        graph.add_block(cube(IVec3::new(1, 0, 0), XXZ));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS).with_hadamard());

        let schedules =
            join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls");
        assert_eq!(
            layer_schedule_at(&schedules, IVec3::new(1, 0, 0)),
            LayerSchedule::Extended,
            "the spatial cube's padded demand must not win over the wall's"
        );
        assert_eq!(
            layer_schedule_at(&schedules, IVec3::ZERO),
            LayerSchedule::Extended
        );
    }

    /// Rounds resolve as `ceil(k*d) + n` on integers, and a cube that lands
    /// below two rounds is rejected by name rather than compiled into a
    /// template with no measurement stage.
    #[rstest]
    #[case("d/2", 3, 2)]
    #[case("d/2", 7, 4)]
    #[case("3d/2", 7, 11)]
    #[case("3d+2", 7, 23)]
    #[case("d-1", 7, 6)]
    fn cube_rounds_resolve_the_height_at_the_code_distance(
        #[case] height: &str,
        #[case] distance: u32,
        #[case] expected: u32,
    ) {
        let cube = Block::new(IVec3::ZERO, BlockKind::Cube(bloq_graph::CubeKind::ZXZ))
            .with_height(height.parse().expect("valid height"))
            .expect("cube accepts a height");

        assert_eq!(
            cube_rounds(&cube, distance).expect("height resolves"),
            expected
        );
    }

    #[rstest]
    #[case("d-2", 3)]
    #[case("d/2-1", 3)]
    #[case("d-9", 7)]
    fn cube_rounds_reject_a_height_below_two_rounds(#[case] height: &str, #[case] distance: u32) {
        let cube = Block::new(
            IVec3::new(1, 2, 3),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        )
        .with_height(height.parse().expect("valid height"))
        .expect("cube accepts a height");

        let error = cube_rounds(&cube, distance).expect_err("too short to compile");
        let CompileError::CubeHeightTooShort { pos, rounds, .. } = error else {
            panic!("expected a too-short height error, got {error}");
        };
        assert_eq!(pos, IVec3::new(1, 2, 3));
        assert!(rounds < 2, "{rounds}");
    }

    #[test]
    fn tall_cube_signature_checks_top_temporal_endpoint() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(
                IVec3::new(0, 0, 0),
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            )
            .with_height("2d".parse().expect("valid height"))
            .unwrap(),
        );
        graph.add_block(Block::new(IVec3::new(0, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 1), Direction::ZPLUS));

        let signature = block_signatures(
            &graph,
            &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph")[&IVec3::ZERO];

        assert_eq!(signature.rounds, Some(6));
        assert!(signature.connectivity.has_pipe(Direction::ZPLUS));
    }

    #[test]
    fn tall_spatial_cube_does_not_mark_unconnected_overlapping_layer_cube() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_block(
            Block::new(
                IVec3::new(1, 0, 0),
                BlockKind::Cube(bloq_graph::CubeKind::ZZX),
            )
            .with_height("2d".parse().expect("valid height"))
            .unwrap(),
        );

        let signatures = block_signatures(
            &graph,
            &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph");

        assert_eq!(
            signatures[&IVec3::new(0, 0, 1)].layer_schedule,
            Some(LayerSchedule::Compact)
        );
        assert_eq!(
            signatures[&IVec3::new(1, 0, 0)].layer_schedule,
            Some(LayerSchedule::Padded)
        );
    }

    #[test]
    fn y_boundary_basis_flips_across_hadamard_edge() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::new(0, 0, 0), BlockKind::Y));
        graph.add_block(Block::new(
            IVec3::new(0, 0, 1),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZPLUS).with_hadamard());

        assert_eq!(
            block_signatures(
                &graph,
                &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::new(0, 0, 0)]
                .boundary_basis,
            Some(Basis::Z)
        );
    }

    #[test]
    fn measurement_signature_inherits_boundary_basis_across_hadamard_edge() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Measurement(Basis::X)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_hadamard());

        let signatures = block_signatures(
            &graph,
            &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph");
        assert_eq!(signatures[&IVec3::Z].boundary_basis, Some(Basis::Z));
        assert!(
            signatures[&IVec3::Z]
                .connectivity
                .has_pipe(Direction::ZMINUS)
        );
    }

    #[test]
    fn y_boundary_basis_resolves_tall_cube_top_endpoint() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(
                IVec3::new(0, 0, 0),
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            )
            .with_height("2d".parse().expect("valid height"))
            .unwrap(),
        );
        graph.add_block(Block::new(IVec3::new(0, 0, 2), BlockKind::Y));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 1), Direction::ZPLUS));

        assert_eq!(
            block_signatures(
                &graph,
                &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::new(0, 0, 2)]
                .boundary_basis,
            Some(Basis::X)
        );
    }

    #[test]
    fn y_boundary_basis_ignores_adjacent_unpiped_cube() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Y));
        graph.add_block(Block::new(
            IVec3::Z,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));

        assert_eq!(
            block_signatures(
                &graph,
                &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::ZERO]
                .boundary_basis,
            None
        );
    }

    #[test]
    fn y_boundary_basis_falls_back_to_graph_pipe_inference() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Y));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Y));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        assert_eq!(
            graph.infer_pipe_basis(graph.pipes().next().unwrap())[1],
            Some(Basis::Z)
        );
        assert_eq!(
            block_signatures(
                &graph,
                &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::ZERO]
                .boundary_basis,
            Some(Basis::Z)
        );
    }

    #[test]
    fn walking_connectivity_checks_virtual_end_endpoint() {
        let mut graph = BlockGraph::new();
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 1)).unwrap();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Walking(walking)));
        graph.add_block(Block::new(IVec3::new(1, 1, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(
            walking.end_position(IVec3::ZERO),
            Direction::ZPLUS,
        ));

        let signature = block_signatures(
            &graph,
            &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph")[&IVec3::ZERO];

        assert!(signature.connectivity.has_pipe(Direction::ZPLUS));
        assert!(!signature.connectivity.has_pipe(Direction::ZMINUS));
    }

    /// A T block at `pos` under a ZXZ cube, optionally with same-layer neighbor
    /// cubes boxing it in.
    fn t_graph(pos: IVec3, neighbors: &[IVec3]) -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(pos, BlockKind::T));
        graph.add_block(Block::new(
            pos + IVec3::Z,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(pos, Direction::ZPLUS));
        for &neighbor in neighbors {
            graph.add_block(Block::new(
                neighbor,
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ));
        }
        graph
    }

    fn t_side(graph: &BlockGraph, pos: IVec3) -> Option<Direction> {
        block_signatures(
            graph,
            &join_component_layer_schedules(graph).expect("no mixed spatial Hadamard walls"),
            CompileConfig::new(3),
        )
        .expect("valid graph")[&pos]
            .surgery_side
    }

    #[test]
    fn t_surgery_side_picks_first_free_in_priority_order() {
        let pos = IVec3::ZERO;
        // All four sides free: -y wins.
        assert_eq!(t_side(&t_graph(pos, &[]), pos), Some(Direction::YMINUS));
        // -y occupied: falls through to +x.
        assert_eq!(
            t_side(&t_graph(pos, &[IVec3::new(0, -1, 0)]), pos),
            Some(Direction::XPLUS)
        );
        // -y and +x occupied: +y.
        assert_eq!(
            t_side(
                &t_graph(pos, &[IVec3::new(0, -1, 0), IVec3::new(1, 0, 0)]),
                pos
            ),
            Some(Direction::YPLUS)
        );
    }

    #[test]
    fn signature_neighbor_steps_do_not_wrap_at_coordinate_limits() {
        let pos = IVec3::new(i32::MAX, i32::MIN, 0);
        let graph = t_graph(pos, &[]);

        assert_eq!(
            t_side(&graph, pos),
            Some(Direction::YPLUS),
            "-Y and +X overflow, so +Y is the first available spill side"
        );

        let isolated = Block::new(IVec3::new(i32::MAX, 0, 0), BlockKind::Port);
        assert!(BlockSignature::template_connectivity(&isolated, &BlockGraph::new()).is_isolated());
    }

    #[test]
    fn t_surgery_side_reservation_is_exclusive_between_t_blocks() {
        // Two T blocks whose only mutually free side is the single cell between
        // them: the left T (first in sorted (z, x, y) order) reserves it as its
        // +x spill, so the right T finds every side occupied or reserved.
        let mut graph = t_graph(IVec3::ZERO, &[IVec3::new(0, -1, 0), IVec3::new(2, -1, 0)]);
        graph.add_block(Block::new(IVec3::new(2, 0, 0), BlockKind::T));
        graph.add_block(Block::new(
            IVec3::new(2, 0, 1),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(2, 0, 0), Direction::ZPLUS));
        // Box the left T's remaining sides except +x, and the right T's +x/+y,
        // so the shared cell (1, 0, 0) is the decisive one.
        for boxed in [
            IVec3::new(0, 1, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(3, 0, 0),
            IVec3::new(2, 1, 0),
        ] {
            graph.add_block(Block::new(
                boxed,
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ));
        }

        assert!(matches!(
            block_signatures(&graph, &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"), CompileConfig::new(3)).unwrap_err(),
            CompileError::TBlockNeedsFreeNeighbor { pos } if pos == IVec3::new(2, 0, 0)
        ));
    }

    #[test]
    fn t_surgery_matching_reassigns_for_a_later_blocks_only_option() {
        // The left T has two free sides (+x, +y); the right T's only free side
        // is the shared middle cell (1, 0, 0) — the left T's +x. A greedy
        // first-fit would grab it for the left T (priority: -y blocked, then
        // +x) and reject the graph; an augmenting path moves the left T to +y
        // and satisfies both.
        let mut graph = t_graph(IVec3::ZERO, &[IVec3::new(0, -1, 0)]);
        graph.add_block(Block::new(IVec3::new(2, 0, 0), BlockKind::T));
        graph.add_block(Block::new(
            IVec3::new(2, 0, 1),
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        graph.add_pipe(Pipe::new(IVec3::new(2, 0, 0), Direction::ZPLUS));
        for boxed in [
            IVec3::new(2, -1, 0),
            IVec3::new(3, 0, 0),
            IVec3::new(2, 1, 0),
        ] {
            graph.add_block(Block::new(
                boxed,
                BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ));
        }

        assert_eq!(t_side(&graph, IVec3::ZERO), Some(Direction::YPLUS));
        assert_eq!(t_side(&graph, IVec3::new(2, 0, 0)), Some(Direction::XMINUS));
    }

    #[test]
    fn t_block_boxed_on_all_sides_needs_free_neighbor() {
        let pos = IVec3::ZERO;
        let graph = t_graph(
            pos,
            &[
                IVec3::new(0, -1, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(-1, 0, 0),
            ],
        );
        assert!(matches!(
            block_signatures(&graph, &join_component_layer_schedules(&graph).expect("no mixed spatial Hadamard walls"), CompileConfig::new(3)).unwrap_err(),
            CompileError::TBlockNeedsFreeNeighbor { pos: p } if p == pos
        ));
    }

    #[test]
    fn t_boundary_basis_inherits_and_flips_across_hadamard_edge() {
        let plain = t_graph(IVec3::ZERO, &[]);
        assert_eq!(
            block_signatures(
                &plain,
                &join_component_layer_schedules(&plain).expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::ZERO]
                .boundary_basis,
            Some(Basis::X) // ZXZ cube's y-face basis
        );

        let mut hadamard = BlockGraph::new();
        hadamard.add_block(Block::new(IVec3::ZERO, BlockKind::T));
        hadamard.add_block(Block::new(
            IVec3::Z,
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        hadamard.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_hadamard());
        assert_eq!(
            block_signatures(
                &hadamard,
                &join_component_layer_schedules(&hadamard)
                    .expect("no mixed spatial Hadamard walls"),
                CompileConfig::new(3)
            )
            .expect("valid graph")[&IVec3::ZERO]
                .boundary_basis,
            Some(Basis::Z)
        );
    }
}
