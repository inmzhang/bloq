//! Conditional graph regions, projections, and branch-aware edits.

use std::collections::{HashMap, HashSet, VecDeque};

use bloq_utils::boolean::BooleanLimits;
use glam::IVec3;

use crate::{
    Action, Block, BlockGraph, BlockGraphError, Expr, InvalidActionError, MeasureTarget, Pipe,
};

/// Blocks and pipes authored as one arm of a named branch region.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BranchArm {
    blocks: Vec<Block>,
    pipes: Vec<Pipe>,
}

impl BranchArm {
    /// Creates an arm. Canonical ordering makes equality and BLOG output stable.
    pub fn new(mut blocks: Vec<Block>, mut pipes: Vec<Pipe>) -> Self {
        blocks.sort_by_key(|block| block.pos().to_array());
        pipes.sort_by_key(pipe_sort_key);
        Self { blocks, pipes }
    }

    /// Creates an arm from position and BLOG block-kind pairs.
    /// Use [`Self::new`] when blocks need tags, heights, or other properties.
    ///
    /// # Errors
    ///
    /// Returns an error if a block kind is invalid. Geometry is checked when
    /// the arm is installed in its parent graph.
    pub fn try_from_blocks<P: Into<IVec3>, K: AsRef<str>>(
        blocks: impl IntoIterator<Item = (P, K)>,
    ) -> Result<Self, BlockGraphError> {
        let blocks = blocks
            .into_iter()
            .map(|(position, kind)| {
                Ok(Block::new(
                    position,
                    kind.as_ref().parse::<crate::BlockKind>()?,
                ))
            })
            .collect::<Result<_, BlockGraphError>>()?;
        Ok(Self::new(blocks, Vec::new()))
    }

    /// Adds explicitly authored pipes to this arm, retaining canonical order.
    #[must_use]
    pub fn with_pipes(mut self, pipes: impl IntoIterator<Item = Pipe>) -> Self {
        self.pipes.extend(pipes);
        self.pipes.sort_by_key(pipe_sort_key);
        self
    }

    /// Iterates over arm blocks in canonical position order.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Iterates over arm pipes in canonical endpoint order.
    pub fn pipes(&self) -> impl Iterator<Item = &Pipe> {
        self.pipes.iter()
    }

    pub(crate) fn push_pipe(&mut self, pipe: Pipe) {
        self.pipes.push(pipe);
        self.pipes.sort_by_key(pipe_sort_key);
    }

    /// Returns whether the arm contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns whether the arm owns a block anchored at `position`.
    pub fn contains_block(&self, position: IVec3) -> bool {
        self.blocks
            .binary_search_by_key(&position.to_array(), |block| block.pos().to_array())
            .is_ok()
    }

    fn reserved_positions(&self) -> Result<HashSet<IVec3>, BlockGraphError> {
        self.blocks
            .iter()
            .try_fold(HashSet::new(), |mut occupied, block| {
                occupied.extend(block.checked_reserved_positions()?);
                Ok(occupied)
            })
    }

    /// Translate an unmaterialized arm, including its external cut endpoints.
    /// Returns an error if a block footprint or pipe endpoint overflows.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] for an unrepresentable shift.
    pub fn try_with_shift(&self, offset: IVec3) -> Result<Self, BlockGraphError> {
        self.try_map_positions(|position| crate::checked_add_position(position, offset))
    }

    fn try_map_positions(
        &self,
        map: impl Fn(IVec3) -> Result<IVec3, BlockGraphError> + Copy,
    ) -> Result<Self, BlockGraphError> {
        Ok(Self::new(
            self.blocks
                .iter()
                .map(|block| {
                    let mut block = block.clone();
                    block.pos = map(block.pos)?;
                    block.checked_reserved_positions()?;
                    Ok::<Block, BlockGraphError>(block)
                })
                .collect::<Result<_, _>>()?,
            self.pipes
                .iter()
                .map(|pipe| {
                    let mut pipe = pipe.clone();
                    pipe.src = map(pipe.src)?;
                    pipe.try_endpoints()?;
                    Ok::<Pipe, BlockGraphError>(pipe)
                })
                .collect::<Result<_, _>>()?,
        ))
    }

    /// Rotate an unmaterialized arm with the graph's block-orientation rules.
    /// Returns an error for unsupported orientations or coordinate overflow.
    ///
    /// # Errors
    ///
    /// Returns an orientation or coordinate error if any arm element cannot rotate.
    pub fn try_with_orientation(
        &self,
        orientation: crate::ModuleOrientation,
    ) -> Result<Self, BlockGraphError> {
        crate::graph::validate_block_orientation(self.blocks(), orientation)?;
        Ok(Self::new(
            self.blocks
                .iter()
                .map(|block| crate::graph::oriented_block(block, orientation))
                .collect::<Result<_, _>>()?,
            self.pipes
                .iter()
                .map(|pipe| crate::graph::oriented_pipe(pipe, orientation))
                .collect::<Result<_, _>>()?,
        ))
    }

    /// Swap X/Z bases, including Hadamards at fixed T/Y endpoints. The source
    /// graph supplies block kinds at external cut endpoints outside this arm.
    #[must_use]
    pub fn flip_xz_basis(&self, source: &BlockGraph) -> Self {
        Self::new(
            self.blocks
                .iter()
                .cloned()
                .map(|mut block| {
                    block.kind = block.kind.flip_xz_basis();
                    block
                })
                .collect(),
            self.pipes
                .iter()
                .cloned()
                .map(|mut pipe| {
                    crate::graph::flip_pipe_xz_basis(&mut pipe, |pos| {
                        self.is_fixed_resource_at(pos, source)
                    });
                    pipe
                })
                .collect(),
        )
    }

    fn is_fixed_resource_at(&self, pos: IVec3, source: &BlockGraph) -> bool {
        self.blocks
            .binary_search_by_key(&pos.to_array(), |block| block.pos().to_array())
            .map(|index| &self.blocks[index])
            .ok()
            .or_else(|| source.get_block(pos))
            .is_some_and(|block| matches!(block.kind(), crate::BlockKind::T | crate::BlockKind::Y))
    }
}

/// A branch ready to attach to a parent graph, including its resolve condition.
/// Both arms use coordinates in the parent graph. External seams are supplied
/// once to [`BlockGraph::try_add_branches`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    /// Public name of the branch.
    pub name: String,
    /// Corrected classical expression selecting the true arm.
    pub condition: Expr,
    /// Geometry selected when the condition is false.
    pub on_false: BranchArm,
    /// Geometry selected when the condition is true.
    pub on_true: BranchArm,
}

impl Branch {
    /// Creates a branch with explicit false and true arms.
    pub fn new(
        name: impl Into<String>,
        condition: Expr,
        on_false: BranchArm,
        on_true: BranchArm,
    ) -> Self {
        Self {
            name: name.into(),
            condition,
            on_false,
            on_true,
        }
    }
}

pub(crate) fn pipe_key(pipe: &Pipe) -> (IVec3, IVec3) {
    let (src, dst) = pipe.endpoints();
    if src.to_array() <= dst.to_array() {
        (src, dst)
    } else {
        (dst, src)
    }
}

fn pipe_sort_key(pipe: &Pipe) -> ([i32; 3], [i32; 3], bool) {
    let (low, high) = pipe_key(pipe);
    (low.to_array(), high.to_array(), pipe.is_hadamard())
}

/// One temporal pipe crossing from the unconditional prefix into an arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchCut {
    /// Endpoint owned by the unconditional prefix.
    pub past: IVec3,
    /// Endpoint owned by the conditional arm.
    pub inside: IVec3,
    /// The source pipe exactly as authored, including orientation, Hadamard,
    /// and tag metadata.
    pub pipe: Pipe,
}

/// A named branch region with two explicitly authored static arms.
///
/// Exactly one arm is materialized in the ordinary [`BlockGraph`] at a time.
/// `incoming` retains the canonical true-arm cut used by lowering; the editor
/// may switch the displayed arm without changing branch semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchRegion {
    /// Public branch name used by BLOG and action expressions.
    pub name: String,
    /// Stable internal anchor, initially the true-arm block owning the first
    /// cut. Transforms preserve the anchor even when cut sorting changes.
    pub target: IVec3,
    /// Complete true-arm boundary attachments. The historical name includes
    /// outgoing and spatial cuts as well as incoming temporal cuts.
    pub incoming: Vec<BranchCut>,
    on_false: BranchArm,
    on_true: BranchArm,
    false_incoming: Vec<BranchCut>,
    shown_true: bool,
}

impl BranchRegion {
    #[cfg(test)]
    pub(crate) fn test_region(target: IVec3, blocks: Vec<IVec3>, incoming: Vec<BranchCut>) -> Self {
        let on_true = BranchArm::new(
            blocks
                .into_iter()
                .map(|position| Block::new(position, crate::BlockKind::default()))
                .collect(),
            Vec::new(),
        );
        Self {
            name: "test_branch".into(),
            target,
            incoming: incoming.clone(),
            on_false: BranchArm::default(),
            on_true,
            false_incoming: incoming,
            shown_true: true,
        }
    }

    /// Returns the arm selected when the branch condition is false.
    pub fn on_false(&self) -> &BranchArm {
        &self.on_false
    }

    /// Returns the arm selected when the branch condition is true.
    pub fn on_true(&self) -> &BranchArm {
        &self.on_true
    }

    /// Returns whether the true arm is currently materialized for display.
    pub fn shown_true(&self) -> bool {
        self.shown_true
    }

    /// Returns the arm currently materialized for display.
    pub fn shown_arm(&self) -> &BranchArm {
        if self.shown_true {
            &self.on_true
        } else {
            &self.on_false
        }
    }

    /// Returns the arm selected by `value`.
    pub fn arm(&self, value: bool) -> &BranchArm {
        if value { &self.on_true } else { &self.on_false }
    }

    /// Returns the boundary cuts for the arm selected by `value`.
    pub fn arm_incoming(&self, value: bool) -> &[BranchCut] {
        if value {
            &self.incoming
        } else {
            &self.false_incoming
        }
    }

    /// Whether either arm owns a block anchored at `position`.
    pub fn contains_any_block(&self, position: IVec3) -> bool {
        self.on_true.contains_block(position) || self.on_false.contains_block(position)
    }

    /// Whether the canonical true arm owns the block at `position`.
    pub fn contains_block(&self, position: IVec3) -> bool {
        self.on_true.contains_block(position)
    }

    fn contains_any_endpoint(&self, position: IVec3) -> bool {
        self.on_false
            .blocks()
            .chain(self.on_true.blocks())
            .any(|block| {
                block
                    .connectable_offsets()
                    .into_iter()
                    .any(|offset| block.pos() + offset == position)
            })
    }

    fn try_map_positions(
        &self,
        map: impl Fn(IVec3) -> Result<IVec3, BlockGraphError> + Copy,
    ) -> Result<Self, BlockGraphError> {
        let map_cut = |cut: &BranchCut| -> Result<BranchCut, BlockGraphError> {
            let mut pipe = cut.pipe.clone();
            pipe.src = map(pipe.src)?;
            Ok(BranchCut {
                past: map(cut.past)?,
                inside: map(cut.inside)?,
                pipe,
            })
        };
        Ok(Self {
            name: self.name.clone(),
            target: map(self.target)?,
            incoming: self
                .incoming
                .iter()
                .map(map_cut)
                .collect::<Result<_, _>>()?,
            on_false: self.on_false.try_map_positions(map)?,
            on_true: self.on_true.try_map_positions(map)?,
            false_incoming: self
                .false_incoming
                .iter()
                .map(map_cut)
                .collect::<Result<_, _>>()?,
            shown_true: self.shown_true,
        })
    }

    pub(crate) fn try_with_orientation(
        &self,
        orientation: crate::ModuleOrientation,
    ) -> Result<Self, BlockGraphError> {
        let map_cut = |cut: &BranchCut| -> Result<BranchCut, BlockGraphError> {
            Ok(BranchCut {
                past: orientation.try_rotate_position(cut.past)?,
                inside: orientation.try_rotate_position(cut.inside)?,
                pipe: crate::graph::oriented_pipe(&cut.pipe, orientation)?,
            })
        };
        Ok(Self {
            name: self.name.clone(),
            target: orientation.try_rotate_position(self.target)?,
            incoming: self
                .incoming
                .iter()
                .map(map_cut)
                .collect::<Result<_, _>>()?,
            on_false: self.on_false.try_with_orientation(orientation)?,
            on_true: self.on_true.try_with_orientation(orientation)?,
            false_incoming: self
                .false_incoming
                .iter()
                .map(map_cut)
                .collect::<Result<_, _>>()?,
            shown_true: self.shown_true,
        })
    }

    pub(crate) fn flip_xz_basis(&self, source: &BlockGraph) -> Self {
        let flip_cut = |cut: &BranchCut, arm: &BranchArm| {
            let mut cut = cut.clone();
            crate::graph::flip_pipe_xz_basis(&mut cut.pipe, |pos| {
                arm.is_fixed_resource_at(pos, source)
            });
            cut
        };
        Self {
            name: self.name.clone(),
            target: self.target,
            incoming: self
                .incoming
                .iter()
                .map(|cut| flip_cut(cut, &self.on_true))
                .collect(),
            on_false: self.on_false.flip_xz_basis(source),
            on_true: self.on_true.flip_xz_basis(source),
            false_incoming: self
                .false_incoming
                .iter()
                .map(|cut| flip_cut(cut, &self.on_false))
                .collect(),
            shown_true: self.shown_true,
        }
    }

    pub(crate) fn set_cut_hadamard(&mut self, key: (IVec3, IVec3), hadamard: bool) {
        let Some(previous) = self
            .arm_incoming(self.shown_true)
            .iter()
            .find(|cut| pipe_key(&cut.pipe) == key)
            .map(|cut| cut.pipe.is_hadamard())
        else {
            return;
        };
        let flip = previous ^ hadamard;
        for pipe in self
            .on_false
            .pipes
            .iter_mut()
            .chain(&mut self.on_true.pipes)
            .filter(|pipe| pipe_key(pipe) == key)
        {
            pipe.hadamard ^= flip;
        }
        for cut in self.incoming.iter_mut().chain(&mut self.false_incoming) {
            if pipe_key(&cut.pipe) == key {
                cut.pipe.hadamard ^= flip;
            }
        }
    }
}

/// One jointly reachable structural-branch assignment and its projected
/// ordinary static graph.
#[derive(Debug, Clone)]
pub struct BranchProjection {
    assignments: Vec<(IVec3, bool)>,
    graph: BlockGraph,
}

impl BranchProjection {
    /// Branch values in canonical resolve-action order.
    pub fn assignments(&self) -> &[(IVec3, bool)] {
        &self.assignments
    }

    /// Returns the projected ordinary block graph.
    pub fn graph(&self) -> &BlockGraph {
        &self.graph
    }

    /// Consumes the projection and returns its ordinary block graph.
    pub fn into_graph(self) -> BlockGraph {
        self.graph
    }
}

#[derive(Debug)]
struct ValidatedArm {
    incoming: Vec<BranchCut>,
    target: IVec3,
}

fn materialize_arm(graph: &mut BlockGraph, arm: &BranchArm) -> Result<(), BlockGraphError> {
    for block in arm.blocks() {
        graph.try_add_block(block.clone())?;
    }
    for pipe in arm.pipes() {
        graph.try_add_pipe(pipe.clone())?;
    }
    Ok(())
}

fn remove_arm(graph: &mut BlockGraph, arm: &BranchArm) {
    for pipe in arm.pipes() {
        let (src, dst) = pipe.endpoints();
        graph.remove_pipe(src, dst);
    }
    for block in arm.blocks() {
        graph.remove_block(block.pos());
    }
}

fn branchless_projection_base(graph: &BlockGraph) -> BlockGraph {
    let mut base = graph.copy_local_geometry();
    base.set_actions_deferred(Vec::new())
        .expect("an empty action list is valid");
    for region in std::mem::take(&mut base.branches) {
        remove_arm(&mut base, region.shown_arm());
    }
    base
}

#[derive(Clone, Copy)]
pub(crate) enum ProjectionContext {
    Graph,
    Definition,
}

fn materialize_projection<'a>(
    mut graph: BlockGraph,
    shared_pipes: &[Pipe],
    arms: impl IntoIterator<Item = (&'a BranchRegion, bool)>,
    actions: &[Action],
    defer_action_analysis: bool,
    context: ProjectionContext,
) -> Result<BlockGraph, BlockGraphError> {
    let arms = arms
        .into_iter()
        .map(|(region, value)| region.arm(value))
        .collect::<Vec<_>>();
    for arm in &arms {
        for block in arm.blocks() {
            graph.try_add_block(block.clone())?;
        }
    }
    for arm in arms {
        for pipe in arm.pipes() {
            graph.try_add_pipe(pipe.clone())?;
        }
    }
    for pipe in shared_pipes {
        if graph.get_pipe(pipe.src(), pipe.dst()).is_none() {
            graph.try_add_pipe(pipe.clone())?;
        }
    }
    graph = graph.fix_shadowed_faces();
    if matches!(context, ProjectionContext::Graph) {
        graph.validate_structure()?;
    }
    if defer_action_analysis {
        graph.set_actions_deferred(actions.to_vec())?;
    } else {
        graph.set_actions(actions.to_vec())?;
    }
    Ok(graph)
}

fn invalid_interface(name: &str, reason: impl Into<String>) -> BlockGraphError {
    InvalidActionError::InvalidBranchInterface {
        name: name.to_owned(),
        reason: reason.into(),
    }
    .into()
}

fn validate_arm(
    candidate: &mut BlockGraph,
    name: &str,
    arm: &BranchArm,
) -> Result<ValidatedArm, BlockGraphError> {
    let members = arm.blocks().map(Block::pos).collect::<HashSet<_>>();
    for block in arm.blocks() {
        if block.kind().is_port() {
            return Err(invalid_interface(name, "arms cannot contain Port blocks"));
        }
        if block.kind().is_dynamic() {
            return Err(invalid_interface(
                name,
                format!("arms cannot contain dynamic {} blocks", block.kind()),
            ));
        }
    }

    materialize_arm(candidate, arm)?;

    let mut incoming = Vec::new();
    let mut forward = HashMap::<IVec3, Vec<IVec3>>::new();
    for pipe in arm.pipes() {
        let (src, dst) = pipe.try_endpoints()?;
        let src_owner = candidate
            .get_endpoint_block(src)
            .ok_or(BlockGraphError::BlockNotFound(src))?
            .pos();
        let dst_owner = candidate
            .get_endpoint_block(dst)
            .ok_or(BlockGraphError::BlockNotFound(dst))?
            .pos();
        let src_inside = members.contains(&src_owner);
        let dst_inside = members.contains(&dst_owner);
        match (src_inside, dst_inside) {
            (false, false) => {
                return Err(invalid_interface(
                    name,
                    "an arm pipe must touch at least one arm block",
                ));
            }
            (true, true) => {
                if pipe.dir().is_spatial() || src.z == dst.z {
                    forward.entry(src_owner).or_default().push(dst_owner);
                    forward.entry(dst_owner).or_default().push(src_owner);
                } else {
                    let (past, future) = if src.z < dst.z {
                        (src_owner, dst_owner)
                    } else {
                        (dst_owner, src_owner)
                    };
                    forward.entry(past).or_default().push(future);
                }
            }
            _ => {
                let (past, inside) = if src_inside { (dst, src) } else { (src, dst) };
                incoming.push(BranchCut {
                    past,
                    inside,
                    pipe: pipe.clone(),
                });
            }
        }
    }
    incoming.sort_by_key(|cut| {
        (
            cut.past.to_array(),
            cut.inside.to_array(),
            pipe_sort_key(&cut.pipe),
        )
    });
    if incoming.is_empty() {
        return Err(invalid_interface(name, "an arm has no external interface"));
    }

    let mut reached = HashSet::new();
    let mut queue = VecDeque::new();
    for cut in &incoming {
        let owner = candidate
            .get_endpoint_block(cut.inside)
            .expect("validated cut endpoint has an arm owner")
            .pos();
        if reached.insert(owner) {
            queue.push_back(owner);
        }
    }
    while let Some(block) = queue.pop_front() {
        for &next in forward.get(&block).into_iter().flatten() {
            if reached.insert(next) {
                queue.push_back(next);
            }
        }
    }
    if let Some(unreachable) = members.iter().find(|position| !reached.contains(position)) {
        return Err(invalid_interface(
            name,
            format!("block {unreachable} is not reachable from an incoming cut"),
        ));
    }

    let target = candidate
        .get_endpoint_block(incoming[0].inside)
        .expect("validated cut endpoint has an arm owner")
        .pos();
    remove_arm(candidate, arm);
    Ok(ValidatedArm { incoming, target })
}

// Keep all source geometry in one scratch graph. Each successful arm check
// removes only what it added, and each region restores the displayed geometry.
struct BranchArmValidation {
    candidate: BlockGraph,
    shared: Vec<Pipe>,
    shared_at: HashMap<IVec3, Vec<usize>>,
}

impl BranchArmValidation {
    fn new(source: &BlockGraph) -> Self {
        let shared = source.branch_shared_pipes();
        let mut shared_at = HashMap::<_, Vec<_>>::new();
        for (index, pipe) in shared.iter().enumerate() {
            for endpoint in [pipe.src(), pipe.dst()] {
                shared_at.entry(endpoint).or_default().push(index);
            }
        }
        let mut candidate = source.clone();
        candidate.branches.clear();
        candidate.clear_actions();
        Self {
            candidate,
            shared,
            shared_at,
        }
    }

    fn with_seams(&self, arm: &BranchArm) -> Result<BranchArm, BlockGraphError> {
        let positions = arm.reserved_positions()?;
        let mut seams = positions
            .iter()
            .flat_map(|position| self.shared_at.get(position).into_iter().flatten().copied())
            .collect::<Vec<_>>();
        seams.sort_unstable();
        seams.dedup();
        let mut arm = arm.clone();
        if !seams.is_empty() {
            arm.pipes
                .extend(seams.into_iter().map(|index| self.shared[index].clone()));
            arm.pipes.sort_by_key(pipe_sort_key);
        }
        Ok(arm)
    }

    fn validate(
        &mut self,
        region: &BranchRegion,
    ) -> Result<(ValidatedArm, ValidatedArm), BlockGraphError> {
        // Restore the actual displayed graph, including common seams. Reading
        // adjacency keeps this proportional to the arm, not the complete source.
        let mut restore = BranchArm::default();
        let mut pipes = HashSet::new();
        for block in region.shown_arm().blocks() {
            if let Some(block) = self.candidate.get_block(block.pos()) {
                restore.blocks.push(block.clone());
                for pipe in self.candidate.pipes_at(block.pos()) {
                    if pipes.insert(pipe_key(pipe)) {
                        restore.pipes.push(pipe.clone());
                    }
                }
            }
        }
        remove_arm(&mut self.candidate, region.shown_arm());
        let on_false = self.with_seams(region.on_false())?;
        let on_false = validate_arm(&mut self.candidate, &region.name, &on_false)?;
        let on_true = self.with_seams(region.on_true())?;
        let on_true = validate_arm(&mut self.candidate, &region.name, &on_true)?;
        if !on_false
            .incoming
            .iter()
            .map(cut_interface)
            .eq(on_true.incoming.iter().map(cut_interface))
        {
            return Err(invalid_interface(
                &region.name,
                "false and true arms have different external interfaces",
            ));
        }
        materialize_arm(&mut self.candidate, &restore)?;
        Ok((on_false, on_true))
    }
}

fn cut_interface(cut: &BranchCut) -> ([i32; 3], [i32; 3]) {
    // SEM-TBRANCH: the shared cut precedes any arm-owned realignment.
    // Its Hadamard is part of the chosen arm, not the common interface.
    (cut.past.to_array(), cut.inside.to_array())
}

fn validate_region_action(
    graph: &BlockGraph,
    actions: &[Action],
    condition: &Expr,
    region: &BranchRegion,
    forbidden: Option<usize>,
) -> Result<(), BlockGraphError> {
    let Some(ordinal) = forbidden else {
        // Every controller is a Measure action in this same list. If no
        // action target touches the region, no controller can touch it.
        return Ok(());
    };
    for controller in controlling_measurements(condition, actions) {
        if measurement_touches_region(graph, controller, region) {
            return Err(InvalidActionError::BranchControllerInside {
                target: region.target,
                controller,
            }
            .into());
        }
    }
    Err(InvalidActionError::BranchContainsActionTarget {
        target: region.target,
        ordinal,
    }
    .into())
}

struct RegionTargetIndex {
    blocks: crate::FxHashMap<IVec3, Vec<usize>>,
    endpoints: crate::FxHashMap<IVec3, Vec<usize>>,
}

impl RegionTargetIndex {
    fn new(regions: &[BranchRegion]) -> Self {
        let mut blocks = crate::FxHashMap::<_, Vec<_>>::default();
        let mut endpoints = crate::FxHashMap::<_, Vec<_>>::default();
        for (region, definition) in regions.iter().enumerate() {
            for block in definition
                .on_false
                .blocks()
                .chain(definition.on_true.blocks())
            {
                blocks.entry(block.pos()).or_default().push(region);
                for offset in block.connectable_offsets() {
                    // Arm validation checked the footprint first; keep the
                    // index total even for a directly queried malformed arm.
                    if let Some(endpoint) = block.pos().checked_add(offset) {
                        endpoints.entry(endpoint).or_default().push(region);
                    }
                }
            }
        }
        Self { blocks, endpoints }
    }

    fn mark(
        postings: &crate::FxHashMap<IVec3, Vec<usize>>,
        position: IVec3,
        ordinal: usize,
        earliest: &mut [Option<usize>],
    ) {
        for &region in postings.get(&position).into_iter().flatten() {
            earliest[region].get_or_insert(ordinal);
        }
    }

    fn mark_measurement(
        &self,
        graph: &BlockGraph,
        target: MeasureTarget,
        ordinal: usize,
        earliest: &mut [Option<usize>],
    ) {
        match target {
            MeasureTarget::Node(position) => {
                if let Some(owner) = graph.get_endpoint_block(position) {
                    Self::mark(&self.blocks, owner.pos(), ordinal, earliest);
                }
                Self::mark(&self.blocks, position, ordinal, earliest);
            }
            MeasureTarget::Edge { src, dir } => {
                let Some(dst) = src.checked_add(dir.to_ivec3()) else {
                    return;
                };
                for endpoint in [src, dst] {
                    if let Some(owner) = graph.get_endpoint_block(endpoint) {
                        Self::mark(&self.blocks, owner.pos(), ordinal, earliest);
                    }
                    Self::mark(&self.endpoints, endpoint, ordinal, earliest);
                }
            }
        }
    }
}

fn earliest_region_action_targets(graph: &BlockGraph, actions: &[Action]) -> Vec<Option<usize>> {
    let index = RegionTargetIndex::new(&graph.branches);
    let mut earliest = vec![None; graph.branches.len()];
    for (ordinal, action) in actions.iter().enumerate() {
        match action {
            Action::Measure { target, .. } => {
                index.mark_measurement(graph, *target, ordinal, &mut earliest);
            }
            Action::Resolve { target, .. } => {
                RegionTargetIndex::mark(&index.blocks, *target, ordinal, &mut earliest);
            }
            Action::Feedback { targets, .. } => {
                for target in targets {
                    let measurement =
                        target
                            .direction
                            .map_or(MeasureTarget::Node(target.target), |dir| {
                                MeasureTarget::Edge {
                                    src: target.target,
                                    dir,
                                }
                            });
                    index.mark_measurement(graph, measurement, ordinal, &mut earliest);
                }
            }
            Action::Let { .. } | Action::DiscardIf(_) | Action::Branch { .. } => {}
        }
    }
    earliest
}

fn controlling_measurements<'a>(condition: &'a Expr, actions: &'a [Action]) -> Vec<MeasureTarget> {
    let definitions = actions
        .iter()
        .filter_map(|action| match action {
            Action::Measure { name, .. } | Action::Let { name, .. } => {
                Some((name.as_str(), action))
            }
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut names = Vec::new();
    collect_expr_names(condition, &mut names);
    let mut seen = HashSet::new();
    let mut measurements = Vec::new();
    while let Some(name) = names.pop() {
        if !seen.insert(name) {
            continue;
        }
        match definitions.get(name).copied() {
            Some(Action::Measure { target, .. }) => measurements.push(*target),
            Some(Action::Let { expr, .. }) => collect_expr_names(expr, &mut names),
            _ => {}
        }
    }
    measurements
}

fn collect_expr_names<'a>(expr: &'a Expr, names: &mut Vec<&'a str>) {
    match expr {
        Expr::Var(name) => names.push(name),
        Expr::Not(inner) => collect_expr_names(inner, names),
        Expr::Binary(_, lhs, rhs) => {
            collect_expr_names(lhs, names);
            collect_expr_names(rhs, names);
        }
    }
}

fn measurement_touches_region(
    graph: &BlockGraph,
    target: MeasureTarget,
    region: &BranchRegion,
) -> bool {
    match target {
        MeasureTarget::Node(position) => {
            graph
                .get_endpoint_block(position)
                .is_some_and(|block| region.contains_any_block(block.pos()))
                || region.contains_any_block(position)
        }
        MeasureTarget::Edge { src, dir } => {
            let Some(dst) = src.checked_add(dir.to_ivec3()) else {
                return false;
            };
            [src, dst].into_iter().any(|endpoint| {
                graph
                    .get_endpoint_block(endpoint)
                    .is_some_and(|block| region.contains_any_block(block.pos()))
                    || region.contains_any_endpoint(endpoint)
            })
        }
    }
}

/// Validates explicit branch definitions against a candidate action list and
/// returns them in resolve-action order.
pub(crate) fn validate_regions_for_actions(
    graph: &BlockGraph,
    actions: &[Action],
) -> Result<Vec<BranchRegion>, BlockGraphError> {
    if !graph.branches.is_empty() {
        let mut validation = BranchArmValidation::new(graph);
        for region in &graph.branches {
            validation.validate(region)?;
            if !region.on_true.contains_block(region.target) {
                return Err(invalid_interface(
                    &region.name,
                    "the true-arm anchor block was removed",
                ));
            }
        }
    }
    let by_target = graph
        .branches
        .iter()
        .enumerate()
        .map(|(index, region)| (region.target, index))
        .collect::<HashMap<_, _>>();
    let mut regions = Vec::new();
    let mut seen = HashSet::new();
    let mut earliest = None;
    for action in actions {
        let Action::Branch { target, condition } = action else {
            continue;
        };
        let &index = by_target
            .get(target)
            .ok_or(InvalidActionError::InvalidBranchTarget(*target))?;
        if !seen.insert(*target) {
            return Err(InvalidActionError::DuplicateBranchTarget(*target).into());
        }
        let region = &graph.branches[index];
        let earliest =
            earliest.get_or_insert_with(|| earliest_region_action_targets(graph, actions));
        validate_region_action(graph, actions, condition, region, earliest[index])?;
        regions.push(region.clone());
    }
    if let Some(region) = graph
        .branches
        .iter()
        .find(|region| !seen.contains(&region.target))
    {
        return Err(InvalidActionError::MissingResolveForBranch(region.name.clone()).into());
    }
    Ok(regions)
}

impl BlockGraph {
    /// Attaches branches, shared seams, and additional actions in one edit.
    /// Resolve actions are created from each branch's condition. Existing
    /// actions are retained. All geometry, names, interfaces, and action
    /// dependencies are checked before changing the graph.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid geometry or actions, leaving `self` unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_graph::{BlockGraph, Branch, BranchArm, Expr, Pipe, Direction};
    /// let mut graph = BlockGraph::from_text("BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\n1: ZXZ [2,0,0]\nm = measure 1\n}")?;
    /// let arm = BranchArm::try_from_blocks([([0, 0, 1], "Z")])?;
    /// graph.try_add_branches(
    ///     [Branch::new("cap", Expr::Var("m".into()), arm.clone(), arm)],
    ///     [Pipe::new([0, 0, 0], Direction::ZPLUS)],
    ///     [],
    /// )?;
    /// assert_eq!(graph.branch_regions()?.len(), 1);
    /// # Ok::<(), bloq_graph::BlockGraphError>(())
    /// ```
    pub fn try_add_branches(
        &mut self,
        branches: impl IntoIterator<Item = Branch>,
        shared_pipes: impl IntoIterator<Item = Pipe>,
        actions: impl IntoIterator<Item = Action>,
    ) -> Result<(), BlockGraphError> {
        let mut candidate = self.clone();
        let mut combined = candidate.actions();
        combined.extend(actions);
        let (regions, conditions): (Vec<_>, Vec<_>) = branches
            .into_iter()
            .map(|branch| {
                (
                    (branch.name, branch.on_false, branch.on_true),
                    branch.condition,
                )
            })
            .unzip();
        let targets =
            candidate.try_add_branch_regions(regions, shared_pipes.into_iter().collect())?;
        combined.extend(
            targets
                .into_iter()
                .zip(conditions)
                .map(|(target, condition)| Action::Branch { target, condition }),
        );
        candidate.set_actions(combined)?;
        *self = candidate;
        Ok(())
    }

    /// Adds a named explicit terminal region and shows its true arm.
    ///
    /// # Errors
    ///
    /// Returns an error if the region name, geometry, seams, or actions are invalid.
    pub fn try_add_branch_region(
        &mut self,
        name: impl Into<String>,
        on_false: BranchArm,
        on_true: BranchArm,
    ) -> Result<IVec3, BlockGraphError> {
        let targets =
            self.try_add_branch_regions(vec![(name.into(), on_false, on_true)], Vec::new())?;
        Ok(targets[0])
    }

    /// Install disjoint region owners together, then resolve their common seams.
    /// Shared pipes remain in the common graph and are never copied into arms.
    ///
    /// # Errors
    ///
    /// Returns an error if any region, shared pipe, or resulting action graph is invalid.
    pub fn try_add_branch_regions(
        &mut self,
        regions: Vec<(String, BranchArm, BranchArm)>,
        shared_pipes: Vec<Pipe>,
    ) -> Result<Vec<IVec3>, BlockGraphError> {
        let mut candidate = self.clone();
        let actions = candidate.actions();
        candidate.rebuild_action_graph_lenient(Vec::new());
        let mut occupied = candidate
            .branches
            .iter()
            .map(|region| {
                let mut positions = region.on_false.reserved_positions()?;
                positions.extend(region.on_true.reserved_positions()?);
                Ok((region.target, positions))
            })
            .collect::<Result<Vec<_>, BlockGraphError>>()?;
        let start = candidate.branches.len();
        for (name, on_false, on_true) in regions {
            if !crate::parser::is_valid_identifier(&name) {
                return Err(InvalidActionError::InvalidActionName(name).into());
            }
            if candidate.branches.iter().any(|region| region.name == name) {
                return Err(InvalidActionError::DuplicateBranchName(name).into());
            }
            let Some(first) = on_true.blocks().next() else {
                return Err(invalid_interface(&name, "an arm has no blocks"));
            };
            if on_false.is_empty() {
                return Err(invalid_interface(&name, "an arm has no blocks"));
            }
            let target = first.pos();
            let mut positions = on_false.reserved_positions()?;
            positions.extend(on_true.reserved_positions()?);
            for (other, other_positions) in &occupied {
                if let Some(&block) = positions.intersection(other_positions).next() {
                    return Err(InvalidActionError::OverlappingBranchRegions {
                        first: *other,
                        second: target,
                        block,
                    }
                    .into());
                }
            }
            occupied.push((target, positions));
            for block in on_true.blocks() {
                candidate.try_add_block(block.clone())?;
            }
            candidate.branches.push(BranchRegion {
                name,
                target,
                incoming: Vec::new(),
                false_incoming: Vec::new(),
                on_false,
                on_true,
                shown_true: true,
            });
        }
        // Keep edit hooks out of installation: each pipe's owner is already known.
        let branches = std::mem::take(&mut candidate.branches);
        for region in &branches[start..] {
            for pipe in region.on_true.pipes() {
                candidate.try_add_pipe(pipe.clone())?;
            }
        }
        for pipe in shared_pipes {
            candidate.try_add_pipe(pipe)?;
        }
        candidate.branches = branches;
        if start < candidate.branches.len() {
            let mut validation = BranchArmValidation::new(&candidate);
            for index in start..candidate.branches.len() {
                let (on_false, on_true) = validation.validate(&candidate.branches[index])?;
                let region = &mut candidate.branches[index];
                region.target = on_true.target;
                region.incoming = on_true.incoming;
                region.false_incoming = on_false.incoming;
            }
        }
        let targets = candidate.branches[start..]
            .iter()
            .map(|region| region.target)
            .collect();
        candidate.rebuild_action_graph_lenient(actions);
        *self = candidate;
        Ok(targets)
    }

    pub(crate) fn branch_shared_pipes(&self) -> Vec<Pipe> {
        let owned = self
            .branches
            .iter()
            .flat_map(|region| region.shown_arm().pipes().map(pipe_key))
            .collect::<HashSet<_>>();
        self.pipes()
            .filter(|pipe| !owned.contains(&pipe_key(pipe)))
            .cloned()
            .collect()
    }

    /// Whether any region reconnects to spatial neighbors or later common code.
    pub fn has_continuing_branches(&self) -> bool {
        self.branches.iter().any(|region| {
            region
                .incoming
                .iter()
                .any(|cut| cut.pipe.dir().is_spatial() || cut.past.z >= cut.inside.z)
        })
    }

    /// Captures selected blocks and their explicitly selected incident pipes,
    /// removing them from the currently visible graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a selected element is missing or the selection is not a closed arm.
    pub fn take_branch_arm(
        &mut self,
        positions: impl IntoIterator<Item = IVec3>,
        pipe_endpoints: impl IntoIterator<Item = (IVec3, IVec3)>,
    ) -> Result<BranchArm, BlockGraphError> {
        let positions = positions.into_iter().collect::<HashSet<_>>();
        let selected_pipes = pipe_endpoints
            .into_iter()
            .map(|(src, dst)| {
                self.get_pipe(src, dst)
                    .cloned()
                    .ok_or(BlockGraphError::PipeNotFound(src, dst))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if positions.is_empty() {
            return if selected_pipes.is_empty() {
                Ok(BranchArm::default())
            } else {
                Err(invalid_interface(
                    "new branch",
                    "selected pipes require a selected block",
                ))
            };
        }
        if let Some((region, position)) = self.branches.iter().find_map(|region| {
            positions
                .iter()
                .find(|position| region.shown_arm().contains_block(**position))
                .map(|position| (region, *position))
        }) {
            return Err(invalid_interface(
                &region.name,
                format!("block {position} already belongs to this branch"),
            ));
        }
        let blocks = positions
            .iter()
            .map(|position| {
                self.get_block(*position)
                    .cloned()
                    .ok_or(BlockGraphError::BlockNotFound(*position))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let incident_pipes = self
            .pipes()
            .filter(|pipe| {
                [pipe.src(), pipe.dst()].into_iter().any(|endpoint| {
                    self.get_endpoint_block(endpoint)
                        .is_some_and(|block| positions.contains(&block.pos()))
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        if incident_pipes.iter().map(pipe_key).collect::<HashSet<_>>()
            != selected_pipes.iter().map(pipe_key).collect()
        {
            return Err(invalid_interface(
                "new branch",
                "select every pipe incident to the arm blocks",
            ));
        }
        let arm = BranchArm::new(blocks, selected_pipes);
        let actions = self.actions();
        self.rebuild_action_graph_lenient(Vec::new());
        remove_arm(self, &arm);
        self.rebuild_action_graph_lenient(actions);
        Ok(arm)
    }

    /// Restores a previously captured loose arm to the visible graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the arm conflicts with existing geometry or actions.
    pub fn restore_branch_arm(&mut self, arm: &BranchArm) -> Result<(), BlockGraphError> {
        let actions = self.actions();
        let mut candidate = self.clone();
        candidate.rebuild_action_graph_lenient(Vec::new());
        materialize_arm(&mut candidate, arm)?;
        candidate.rebuild_action_graph_lenient(actions);
        *self = candidate;
        Ok(())
    }

    /// Returns explicit branch definitions in declaration order.
    pub fn branch_definitions(&self) -> &[BranchRegion] {
        &self.branches
    }

    /// Finds an explicit branch definition by public name.
    pub fn branch_by_name(&self, name: &str) -> Option<&BranchRegion> {
        self.branches.iter().find(|region| region.name == name)
    }

    /// Finds an explicit branch definition by stable internal anchor.
    pub fn branch_by_target(&self, target: IVec3) -> Option<&BranchRegion> {
        self.branches.iter().find(|region| region.target == target)
    }

    /// Returns the displayed region owning `position`, for scene picking.
    pub fn shown_branch_at(&self, position: IVec3) -> Option<&BranchRegion> {
        self.branches
            .iter()
            .find(|region| region.shown_arm().contains_block(position))
    }

    /// Switches which arm is materialized for display.
    ///
    /// # Errors
    ///
    /// Returns an error if the branch is unknown or the selected arm cannot be materialized.
    pub fn set_shown_branch_arm(
        &mut self,
        name: &str,
        show_true: bool,
    ) -> Result<(), BlockGraphError> {
        let Some(index) = self.branches.iter().position(|region| region.name == name) else {
            return Err(InvalidActionError::UnknownBranchName(name.to_owned()).into());
        };
        if self.branches[index].shown_true == show_true {
            return Ok(());
        }
        let actions = self.actions();
        let mut candidate = self.clone();
        candidate.rebuild_action_graph_lenient(Vec::new());
        // Detach the region while changing its display materialization: these
        // removals must not delete the authored arm.
        let shared = candidate.branch_shared_pipes();
        let mut branches = std::mem::take(&mut candidate.branches);
        let region = &mut branches[index];
        remove_arm(&mut candidate, region.shown_arm());
        materialize_arm(&mut candidate, region.arm(show_true))?;
        for pipe in shared {
            if candidate.get_pipe(pipe.src(), pipe.dst()).is_none() {
                candidate.try_add_pipe(pipe)?;
            }
        }
        region.shown_true = show_true;
        candidate.branches = branches;
        candidate.rebuild_action_graph_lenient(actions);
        *self = candidate;
        Ok(())
    }

    #[doc(hidden)]
    pub fn canonical_true_branch_view(&self) -> Result<Self, BlockGraphError> {
        let mut graph = self.clone();
        let hidden_true = graph
            .branches
            .iter()
            .filter(|region| !region.shown_true)
            .map(|region| region.name.clone())
            .collect::<Vec<_>>();
        for name in hidden_true {
            graph.set_shown_branch_arm(&name, true)?;
        }
        Ok(graph)
    }

    /// Returns explicit regions in resolve-action order.
    ///
    /// # Errors
    ///
    /// Returns an error if branch actions and stored regions disagree.
    pub fn branch_regions(&self) -> Result<Vec<BranchRegion>, BlockGraphError> {
        validate_regions_for_actions(self, &self.actions())
    }

    /// Projects every reachable explicit branch assignment.
    ///
    /// # Errors
    ///
    /// Returns an error if branch validation, enumeration, or materialization fails.
    pub fn branch_projections(&self) -> Result<Vec<BranchProjection>, BlockGraphError> {
        self.require_flat_source("branch projection")?;
        self.branch_projections_up_to(usize::MAX)
    }

    /// Enumerate only the reachable selector tuples, without retaining projected graphs.
    #[doc(hidden)]
    pub fn branch_assignments_up_to(
        &self,
        limit: usize,
    ) -> Result<Vec<Vec<(IVec3, bool)>>, BlockGraphError> {
        let regions = self.branch_regions()?;
        let actions = self.actions();
        let dag = self.build_action_graph(&actions)?;
        let targets = regions
            .iter()
            .map(|region| region.target)
            .collect::<Vec<_>>();
        Ok(dag
            .branch_values_up_to(&targets, limit)?
            .into_iter()
            .map(|values| targets.iter().copied().zip(values).collect())
            .collect())
    }

    pub(crate) fn branch_projections_up_to(
        &self,
        limit: usize,
    ) -> Result<Vec<BranchProjection>, BlockGraphError> {
        self.branch_projections_impl(limit, false, BooleanLimits::DEFAULT)
    }

    /// Returns reachable static projections without deriving stabilizers.
    #[cfg(test)]
    pub(crate) fn branch_projections_for_analysis(
        &self,
    ) -> Result<Vec<BranchProjection>, BlockGraphError> {
        self.branch_projections_impl(usize::MAX, true, BooleanLimits::DEFAULT)
    }

    fn branch_projections_impl(
        &self,
        limit: usize,
        defer_action_analysis: bool,
        boolean_limits: BooleanLimits,
    ) -> Result<Vec<BranchProjection>, BlockGraphError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let regions = self.branch_regions()?;
        if regions.is_empty() {
            return Ok(vec![BranchProjection {
                assignments: Vec::new(),
                graph: self.clone(),
            }]);
        }
        let actions = self.actions();
        let dag = self.build_action_graph(&actions)?;
        let remaining_actions = actions
            .iter()
            .filter(|action| !matches!(action, Action::Branch { .. }))
            .cloned()
            .collect::<Vec<_>>();
        let base = branchless_projection_base(self);
        let targets = regions
            .iter()
            .map(|region| region.target)
            .collect::<Vec<_>>();
        let shared_pipes = self.branch_shared_pipes();
        dag.branch_values_up_to_with_limits(&targets, limit, boolean_limits)?
            .iter()
            .map(|values| {
                let assignments = targets
                    .iter()
                    .copied()
                    .zip(values.iter().copied())
                    .collect::<Vec<_>>();
                let graph = materialize_projection(
                    base.clone(),
                    &shared_pipes,
                    regions
                        .iter()
                        .zip(values)
                        .map(|(region, &value)| (region, value)),
                    &remaining_actions,
                    defer_action_analysis,
                    ProjectionContext::Graph,
                )?;
                Ok(BranchProjection { assignments, graph })
            })
            .collect()
    }

    /// Projects every explicit branch to an ordinary static graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing assignments or an invalid projected graph.
    pub fn project_branches(
        &self,
        assignments: impl IntoIterator<Item = (IVec3, bool)>,
    ) -> Result<Self, BlockGraphError> {
        self.require_flat_source("branch projection")?;
        self.project_branches_impl(assignments, false, ProjectionContext::Graph)
    }

    #[doc(hidden)]
    pub fn project_branches_deferred(
        &self,
        assignments: impl IntoIterator<Item = (IVec3, bool)>,
    ) -> Result<Self, BlockGraphError> {
        self.require_flat_source("branch projection")?;
        self.project_branches_impl(assignments, true, ProjectionContext::Graph)
    }

    /// Project a definition's owned graph before module bindings install its
    /// external pipes. The complete linked projection still needs validation.
    #[doc(hidden)]
    pub fn project_branches_in_definition(
        &self,
        assignments: impl IntoIterator<Item = (IVec3, bool)>,
    ) -> Result<Self, BlockGraphError> {
        self.project_branches_impl(assignments, true, ProjectionContext::Definition)
    }

    pub(crate) fn project_branches_impl(
        &self,
        assignments: impl IntoIterator<Item = (IVec3, bool)>,
        defer_action_analysis: bool,
        context: ProjectionContext,
    ) -> Result<Self, BlockGraphError> {
        let regions = self.branch_regions()?;
        let known_targets = regions
            .iter()
            .map(|region| region.target)
            .collect::<HashSet<_>>();
        let mut values = HashMap::new();
        for (target, value) in assignments {
            if !known_targets.contains(&target) {
                return Err(InvalidActionError::UnknownBranchValue(target).into());
            }
            if values.insert(target, value).is_some() {
                return Err(InvalidActionError::DuplicateBranchValue(target).into());
            }
        }
        for region in &regions {
            if !values.contains_key(&region.target) {
                return Err(InvalidActionError::MissingBranchValue(region.target).into());
            }
        }

        let remaining_actions = self
            .actions()
            .into_iter()
            .filter(|action| !matches!(action, Action::Branch { .. }))
            .collect::<Vec<_>>();
        materialize_projection(
            branchless_projection_base(self),
            &self.branch_shared_pipes(),
            regions
                .iter()
                .map(|region| (region, values[&region.target])),
            &remaining_actions,
            defer_action_analysis,
            context,
        )
    }

    /// Materialize an already validated branch description. The caller owns the
    /// region order and supplies one value per region; public projection APIs
    /// continue to validate untrusted assignments before reaching materialization.
    pub(crate) fn project_validated_branches_deferred(
        &self,
        regions: &[BranchRegion],
        shared_pipes: &[Pipe],
        values: &[bool],
    ) -> Result<Self, BlockGraphError> {
        assert_eq!(regions.len(), values.len());
        let actions = self
            .actions()
            .into_iter()
            .filter(|action| !matches!(action, Action::Branch { .. }))
            .collect::<Vec<_>>();
        materialize_projection(
            branchless_projection_base(self),
            shared_pipes,
            regions
                .iter()
                .zip(values)
                .map(|(region, &value)| (region, value)),
            &actions,
            true,
            ProjectionContext::Graph,
        )
    }

    pub(crate) fn transformed_branches(
        &self,
        map: impl Fn(IVec3) -> Result<IVec3, BlockGraphError> + Copy,
    ) -> Result<Vec<BranchRegion>, BlockGraphError> {
        self.branches
            .iter()
            .map(|region| region.try_map_positions(map))
            .collect()
    }

    /// A pipe touching a shown arm belongs to that arm, including a repaired cut.
    pub(crate) fn record_added_branch_pipe(&mut self, pipe: &Pipe) {
        self.record_added_branch_pipe_for(pipe, 0..self.branches.len());
    }

    /// Linkers may preselect the regions touched by this pipe. Indices remain
    /// in stored region order, preserving cut updates and error precedence.
    pub(crate) fn record_added_branch_pipe_for(
        &mut self,
        pipe: &Pipe,
        regions: impl IntoIterator<Item = usize>,
    ) {
        if self.branches.is_empty() {
            return;
        }
        let (src, dst) = pipe.endpoints();
        let src_owner = self
            .get_endpoint_block(src)
            .expect("added pipe source exists")
            .pos();
        let dst_owner = self
            .get_endpoint_block(dst)
            .expect("added pipe destination exists")
            .pos();
        let key = pipe_key(pipe);
        for index in regions {
            let region = &mut self.branches[index];
            let owned = region
                .on_false
                .pipes()
                .chain(region.on_true.pipes())
                .any(|stored| pipe_key(stored) == key);
            region.set_cut_hadamard(key, pipe.is_hadamard());
            let (shown, incoming) = if region.shown_true {
                (&mut region.on_true, &mut region.incoming)
            } else {
                (&mut region.on_false, &mut region.false_incoming)
            };
            let src_inside = shown.contains_block(src_owner);
            let dst_inside = shown.contains_block(dst_owner);
            if !src_inside && !dst_inside {
                continue;
            }
            if src_inside != dst_inside && !owned {
                // External seams live once in the common graph, including
                // seams connecting two independently selected region owners.
                let (past, inside) = if src_inside { (dst, src) } else { (src, dst) };
                for cuts in [&mut region.incoming, &mut region.false_incoming] {
                    cuts.retain(|cut| pipe_key(&cut.pipe) != key);
                    cuts.push(BranchCut {
                        past,
                        inside,
                        pipe: pipe.clone(),
                    });
                    cuts.sort_by_key(|cut| {
                        (
                            cut.past.to_array(),
                            cut.inside.to_array(),
                            pipe_sort_key(&cut.pipe),
                        )
                    });
                }
                continue;
            }
            if let Some(stored) = shown
                .pipes
                .iter_mut()
                .find(|stored| pipe_key(stored) == key)
            {
                *stored = pipe.clone();
            } else {
                shown.push_pipe(pipe.clone());
            }
            if src_inside != dst_inside {
                let (past, inside) = if src_inside { (dst, src) } else { (src, dst) };
                incoming.retain(|cut| pipe_key(&cut.pipe) != key);
                incoming.push(BranchCut {
                    past,
                    inside,
                    pipe: pipe.clone(),
                });
                incoming.sort_by_key(|cut| {
                    (
                        cut.past.to_array(),
                        cut.inside.to_array(),
                        pipe_sort_key(&cut.pipe),
                    )
                });
            }
        }
    }

    pub(crate) fn sync_shown_branch_arm_data(&mut self) {
        if self.branches.is_empty() {
            return;
        }
        let blocks = self
            .blocks()
            .map(|block| (block.pos(), block.clone()))
            .collect::<HashMap<_, _>>();
        let pipes = self
            .pipes()
            .map(|pipe| (pipe_key(pipe), pipe.clone()))
            .collect::<HashMap<_, _>>();
        for region in &mut self.branches {
            let (shown, incoming) = if region.shown_true {
                (&mut region.on_true, &mut region.incoming)
            } else {
                (&mut region.on_false, &mut region.false_incoming)
            };
            shown.blocks.retain_mut(|block| {
                if let Some(updated) = blocks.get(&block.pos()) {
                    *block = updated.clone();
                    true
                } else {
                    false
                }
            });
            shown.pipes.retain_mut(|pipe| {
                if let Some(updated) = pipes.get(&pipe_key(pipe)) {
                    *pipe = updated.clone();
                    true
                } else {
                    false
                }
            });
            incoming.retain_mut(|cut| {
                if let Some(updated) = pipes.get(&pipe_key(&cut.pipe)) {
                    cut.pipe = updated.clone();
                    true
                } else {
                    false
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::ivec3;

    use super::*;
    use crate::{Basis, BlockKind, CubeKind, Direction, SelectiveKind};

    #[test]
    fn reused_arm_validation_matches_independent_full_graph_copies() {
        // Previous implementation, retained as an independent scratch-lifetime
        // oracle: every region and each arm start from their own complete copy.
        fn cloned(
            source: &BlockGraph,
            region: &BranchRegion,
        ) -> Result<(ValidatedArm, ValidatedArm), BlockGraphError> {
            let shared = source.branch_shared_pipes();
            let mut common = source.clone();
            common.branches.clear();
            common.clear_actions();
            remove_arm(&mut common, region.shown_arm());
            let check = |arm: &BranchArm| {
                let positions = arm.reserved_positions()?;
                let mut arm = arm.clone();
                for pipe in &shared {
                    if positions.contains(&pipe.src()) || positions.contains(&pipe.dst()) {
                        arm.push_pipe(pipe.clone());
                    }
                }
                validate_arm(&mut common.clone(), &region.name, &arm)
            };
            let on_false = check(region.on_false())?;
            let on_true = check(region.on_true())?;
            if !on_false
                .incoming
                .iter()
                .map(cut_interface)
                .eq(on_true.incoming.iter().map(cut_interface))
            {
                return Err(invalid_interface(
                    &region.name,
                    "false and true arms have different external interfaces",
                ));
            }
            Ok((on_false, on_true))
        }
        let ast = crate::parse_blog_program_to_ast(include_str!(
            "../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap();
        let base = crate::lower_blog_graph_ast_deferred(&ast)
            .unwrap()
            .root()
            .local_body()
            .clone();
        let mut cases = Vec::new();
        for mask in 0..8 {
            let mut graph = base.clone();
            for bit in 0..3 {
                graph
                    .set_shown_branch_arm(&format!("cz{bit}"), mask & (1 << bit) != 0)
                    .unwrap();
            }
            cases.push(graph);
        }
        for mutation in 0..4 {
            let mut graph = base.clone();
            let arm = &mut graph.branches[1].on_false;
            match mutation {
                0 => arm.blocks.push(arm.blocks[0].clone()),
                1 => arm.blocks.push(cube(ivec3(100, 100, 0))),
                2 => arm.blocks.push(cube(ivec3(1, -1, 0))), // Occupied by a common Port.
                _ => arm
                    .pipes
                    .push(Pipe::new(ivec3(100, 100, 0), Direction::ZPLUS)),
            }
            cases.push(graph);
        }
        let (mut missing_cut, _) = explicit_branch();
        missing_cut.branches[0].on_false.pipes.clear();
        cases.push(missing_cut);
        let summarize = |result: Result<(ValidatedArm, ValidatedArm), BlockGraphError>| {
            result
                .map(|(a, b)| ((a.incoming, a.target), (b.incoming, b.target)))
                .map_err(|error| error.to_string())
        };
        for graph in cases {
            let mut validation = BranchArmValidation::new(&graph);
            let mut valid = true;
            for region in &graph.branches {
                let expected = summarize(cloned(&graph, region));
                let actual = summarize(validation.validate(region));
                assert_eq!(actual, expected);
                if actual.is_err() {
                    valid = false;
                    break;
                }
            }
            if valid {
                let geometry = |graph: &BlockGraph| {
                    let mut blocks = graph.blocks().cloned().collect::<Vec<_>>();
                    let mut pipes = graph.pipes().cloned().collect::<Vec<_>>();
                    blocks.sort_by_key(|block| block.pos().to_array());
                    pipes.sort_by_key(pipe_sort_key);
                    (blocks, pipes)
                };
                assert_eq!(geometry(&validation.candidate), geometry(&graph));
            }
        }
    }

    #[test]
    fn continuing_regions_keep_shared_seams_across_every_projection_and_preview() {
        let source = include_str!("../../docs/fixtures/conditional_cz_strip.blog");
        let ast = crate::parse_blog_program_to_ast(source).unwrap();
        let program = crate::lower_blog_graph_ast_deferred(&ast).unwrap();
        let mut graph = program.flatten().unwrap();
        assert!(graph.has_continuing_branches());
        assert_eq!(graph.branch_shared_pipes().len(), 4);
        let projections = graph.branch_projections_for_analysis().unwrap();
        assert_eq!(projections.len(), 8);
        for projection in projections {
            let enabled = projection
                .assignments()
                .iter()
                .filter(|(_, value)| *value)
                .count();
            assert_eq!(
                projection
                    .graph()
                    .pipes()
                    .filter(|pipe| pipe.is_hadamard())
                    .count(),
                2 * enabled
            );
            projection.graph().validate_structure().unwrap();
        }
        for mask in 0..8 {
            for bit in 0..3 {
                graph
                    .set_shown_branch_arm(&format!("cz{bit}"), mask & (1 << bit) != 0)
                    .unwrap();
            }
            assert_eq!(graph.branch_shared_pipes().len(), 4);
            assert_eq!(graph.branch_regions().unwrap().len(), 3);
        }
        for transformed in [
            graph.shift_positions(ivec3(3, -2, 4)).unwrap(),
            graph
                .rotate_about_origin_lenient(crate::UDirection::Z, 1)
                .unwrap(),
            graph.flip_xz_basis().unwrap(),
        ] {
            assert_eq!(transformed.branch_shared_pipes().len(), 4);
            let projections = transformed.branch_projections_for_analysis().unwrap();
            assert_eq!(projections.len(), 8);
            for projection in projections {
                projection.graph().validate_structure().unwrap();
            }
        }
        let text = program.to_blog_text();
        let ast = crate::parse_blog_program_to_ast(&text).unwrap();
        let reparsed = crate::lower_blog_graph_ast_deferred(&ast).unwrap();
        assert_eq!(
            reparsed
                .root()
                .local_body()
                .branch_projections_for_analysis()
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn module_bindings_preserve_private_boundary_hadamards() {
        let source = r#"BLOG 1.0
module Gate {
  in input: data = 0
  out output: data = 1
  0: Port [0,0,0] role=input
  1: Port [0,0,2] role=output
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
module main {
  gate: Gate @ [0,0,0]
  10: ZXZ [0,0,0]
  11: ZXZ [0,0,2]
  10 -> gate.input
  gate.output -> 11
}
"#;
        let program = crate::parse_inline_graph(source).unwrap();
        let graph = program.materialize_flat_graph().unwrap();
        let region = graph.branch_by_name("gate__b").unwrap();
        assert!(region.on_false().pipes().all(Pipe::is_hadamard));
        assert!(region.on_true().pipes().all(|pipe| !pipe.is_hadamard()));
        for selected in [false, true] {
            graph
                .project_branches([(region.target, selected)])
                .unwrap()
                .validate_structure()
                .unwrap();
        }
    }

    fn cube(position: IVec3) -> Block {
        Block::new(position, BlockKind::Cube(CubeKind::ZXZ))
    }

    fn explicit_branch() -> (BlockGraph, IVec3) {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(ivec3(0, 0, 0)));
        graph.add_block(cube(ivec3(3, 0, 0)));
        let false_arm = BranchArm::new(
            vec![Block::new(ivec3(0, 0, 1), BlockKind::Measurement(Basis::X))],
            vec![Pipe::new(ivec3(0, 0, 0), Direction::ZPLUS)],
        );
        let true_arm = BranchArm::new(
            vec![cube(ivec3(0, 0, 1)), cube(ivec3(0, 0, 2))],
            vec![
                Pipe::new(ivec3(0, 0, 0), Direction::ZPLUS),
                Pipe::new(ivec3(0, 0, 1), Direction::ZPLUS),
            ],
        );
        let target = graph
            .try_add_branch_region("b0", false_arm, true_arm)
            .expect("matching explicit arms");
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(ivec3(3, 0, 0)),
                    name: "m".into(),
                },
                Action::Branch {
                    target,
                    condition: Expr::Var("m".into()),
                },
            ])
            .expect("branch resolve validates");
        (graph, target)
    }

    #[test]
    fn authored_branch_installs_resolve_and_preserves_both_projections() {
        let (expected, target) = explicit_branch();
        let mut graph = BlockGraph::new();
        graph.add_block(cube(ivec3(0, 0, 0)));
        graph.add_block(cube(ivec3(3, 0, 0)));
        graph
            .set_actions(vec![Action::Measure {
                target: MeasureTarget::Node(ivec3(3, 0, 0)),
                name: "m".into(),
            }])
            .unwrap();
        let branch = Branch::new(
            "b0",
            Expr::Var("m".into()),
            BranchArm::try_from_blocks([([0, 0, 1], "X")]).unwrap(),
            BranchArm::try_from_blocks([([0, 0, 1], "ZXZ"), ([0, 0, 2], "ZXZ")])
                .unwrap()
                .with_pipes([Pipe::new([0, 0, 1], Direction::ZPLUS)]),
        );
        graph
            .try_add_branches([branch], [Pipe::new([0, 0, 0], Direction::ZPLUS)], [])
            .unwrap();
        assert_eq!(graph.actions(), expected.actions());
        for value in [false, true] {
            assert_eq!(
                graph
                    .project_branches([(target, value)])
                    .unwrap()
                    .to_blog_text(),
                expected
                    .project_branches([(target, value)])
                    .unwrap()
                    .to_blog_text(),
            );
        }
        let reparsed = BlockGraph::from_text(&graph.to_blog_text()).unwrap();
        assert_eq!(reparsed.to_blog_text(), graph.to_blog_text());
    }

    #[test]
    fn authored_branches_install_shared_seams_between_owners() {
        let mut graph = BlockGraph::new();
        for x in [0, 1, 3] {
            graph.add_block(cube(ivec3(x, 0, 0)));
        }
        let branches = [0, 1].map(|x| {
            let arm = BranchArm::try_from_blocks([([x, 0, 1], "ZXZ")]).unwrap();
            Branch::new(format!("b{x}"), Expr::Var("m".into()), arm.clone(), arm)
        });
        graph
            .try_add_branches(
                branches,
                [
                    Pipe::new([0, 0, 0], Direction::ZPLUS),
                    Pipe::new([1, 0, 0], Direction::ZPLUS),
                    Pipe::new([0, 0, 1], Direction::XPLUS),
                ],
                [Action::Measure {
                    target: MeasureTarget::Node(ivec3(3, 0, 0)),
                    name: "m".into(),
                }],
            )
            .unwrap();
        let regions = graph.branch_regions().unwrap();
        assert_eq!(regions.len(), 2);
        for region in &regions {
            assert!(region.on_true().pipes().next().is_none());
            assert!(region.on_false().pipes().next().is_none());
        }
        for first in [false, true] {
            for second in [false, true] {
                graph
                    .project_branches([(regions[0].target, first), (regions[1].target, second)])
                    .unwrap()
                    .validate_structure()
                    .unwrap();
            }
        }
    }

    #[test]
    fn authored_branch_errors_leave_geometry_and_actions_unchanged() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(ivec3(0, 0, 0)));
        graph.add_block(cube(ivec3(3, 0, 0)));
        let before = graph.to_blog_text();
        let arm = BranchArm::try_from_blocks([([0, 0, 1], "Z")]).unwrap();
        for branch in [
            Branch::new("bad name", Expr::Var("m".into()), arm.clone(), arm.clone()),
            Branch::new("b", Expr::Var("missing".into()), arm.clone(), arm.clone()),
            Branch::new(
                "b",
                Expr::Var("m".into()),
                BranchArm::default(),
                arm.clone(),
            ),
            Branch::new(
                "b",
                Expr::Var("m".into()),
                arm.clone(),
                BranchArm::try_from_blocks([([1, 0, 1], "Z")]).unwrap(),
            ),
        ] {
            assert!(
                graph
                    .try_add_branches(
                        [branch],
                        [Pipe::new([0, 0, 0], Direction::ZPLUS)],
                        [Action::Measure {
                            target: MeasureTarget::Node(ivec3(3, 0, 0)),
                            name: "m".into()
                        }]
                    )
                    .is_err()
            );
            assert_eq!(graph.to_blog_text(), before);
            assert!(!graph.has_actions());
        }
        assert!(matches!(
            BranchArm::try_from_blocks([([0, 0, 1], "invalid")]),
            Err(BlockGraphError::Block(crate::BlockError::InvalidBlockKind(
                _
            )))
        ));
    }

    // The previous per-region scan is an independent oracle for the transient
    // inverse index, including malformed action lists accepted by this helper.
    fn scalar_first_target(
        graph: &BlockGraph,
        actions: &[Action],
        region: &BranchRegion,
    ) -> Option<usize> {
        actions.iter().position(|action| match action {
            Action::Measure { target, .. } => measurement_touches_region(graph, *target, region),
            Action::Resolve { target, .. } => region.contains_any_block(*target),
            Action::Feedback { targets, .. } => targets.iter().any(|target| {
                let measurement =
                    target
                        .direction
                        .map_or(MeasureTarget::Node(target.target), |dir| {
                            MeasureTarget::Edge {
                                src: target.target,
                                dir,
                            }
                        });
                measurement_touches_region(graph, measurement, region)
            }),
            Action::Let { .. } | Action::DiscardIf(_) | Action::Branch { .. } => false,
        })
    }

    #[test]
    fn region_target_index_matches_scalar_for_hidden_and_extended_endpoints() {
        let visible = cube(ivec3(0, 0, 0))
            .with_height("3d".parse().unwrap())
            .unwrap();
        let hidden = cube(ivec3(10, 0, 0))
            .with_height("3d".parse().unwrap())
            .unwrap();
        let virtual_offset = visible
            .connectable_offsets()
            .into_iter()
            .find(|&offset| offset != IVec3::ZERO)
            .unwrap();
        let mut graph = BlockGraph::new();
        graph.add_block(visible.clone());
        let mut first = BranchRegion::test_region(visible.pos(), vec![visible.pos()], Vec::new());
        first.on_true = BranchArm::new(vec![visible.clone()], Vec::new());
        first.on_false = BranchArm::new(vec![hidden.clone()], Vec::new());
        let mut second =
            BranchRegion::test_region(ivec3(20, 0, 0), vec![ivec3(20, 0, 0)], Vec::new());
        second.on_false = BranchArm::new(vec![hidden.clone()], Vec::new());
        graph.branches.extend([first, second]);
        let actions = [
            Action::Measure {
                target: MeasureTarget::Node(visible.pos() + virtual_offset),
                name: "visible".into(),
            },
            Action::Measure {
                target: MeasureTarget::Node(hidden.pos() + virtual_offset),
                name: "hidden_virtual_node".into(),
            },
            Action::Measure {
                target: MeasureTarget::Edge {
                    src: hidden.pos() + virtual_offset,
                    dir: Direction::XPLUS,
                },
                name: "hidden_edge".into(),
            },
            Action::Feedback {
                targets: vec![crate::FeedbackTarget {
                    target: hidden.pos(),
                    direction: Some(Direction::ZPLUS),
                    pauli: bloq_utils::PauliBasis::X,
                }],
                condition: None,
            },
            Action::Resolve {
                target: hidden.pos(),
                condition: Expr::Var("visible".into()),
            },
            Action::Measure {
                target: MeasureTarget::Node(ivec3(100, 0, 0)),
                name: "outside".into(),
            },
        ];
        for action in &actions {
            let indexed = earliest_region_action_targets(&graph, std::slice::from_ref(action));
            for (region, definition) in graph.branches.iter().enumerate() {
                assert_eq!(
                    indexed[region],
                    scalar_first_target(&graph, std::slice::from_ref(action), definition),
                    "{action:?}, region {region}"
                );
            }
        }
        let indexed = earliest_region_action_targets(&graph, &actions);
        for (region, definition) in graph.branches.iter().enumerate() {
            assert_eq!(
                indexed[region],
                scalar_first_target(&graph, &actions, definition)
            );
        }
        assert_eq!(indexed, [Some(0), Some(2)]);
    }

    #[test]
    fn region_action_index_preserves_controller_and_action_error_order() {
        let (graph, target) = explicit_branch();
        let inside = graph.branches[0].on_true.blocks().next().unwrap().pos();
        let outside = ivec3(3, 0, 0);
        let measure = |target, name: &str| Action::Measure {
            target: MeasureTarget::Node(target),
            name: name.into(),
        };
        let branch = |name: &str| Action::Branch {
            target,
            condition: Expr::Var(name.into()),
        };
        let actions = [
            measure(inside, "m"),
            Action::Let {
                name: "alias".into(),
                expr: Expr::Var("m".into()),
            },
            branch("alias"),
        ];
        assert!(matches!(
            validate_regions_for_actions(&graph, &actions),
            Err(BlockGraphError::InvalidAction(InvalidActionError::BranchControllerInside {
                target: actual,
                controller: MeasureTarget::Node(site),
            })) if actual == target && site == inside
        ));
        let duplicate_alias = [
            measure(inside, "m"),
            measure(outside, "m"),
            Action::Let {
                name: "alias".into(),
                expr: Expr::Var("m".into()),
            },
            branch("alias"),
        ];
        assert!(matches!(
            validate_regions_for_actions(&graph, &duplicate_alias),
            Err(BlockGraphError::InvalidAction(InvalidActionError::BranchContainsActionTarget {
                target: actual,
                ordinal: 0,
            })) if actual == target
        ));
        let cyclic_alias = [
            measure(inside, "m"),
            Action::Let {
                name: "a".into(),
                expr: Expr::Var("b".into()),
            },
            Action::Let {
                name: "b".into(),
                expr: Expr::Var("a".into()),
            },
            branch("a"),
        ];
        assert!(matches!(
            validate_regions_for_actions(&graph, &cyclic_alias),
            Err(BlockGraphError::InvalidAction(InvalidActionError::BranchContainsActionTarget {
                target: actual,
                ordinal: 0,
            })) if actual == target
        ));
        assert!(matches!(
            validate_regions_for_actions(&graph, &[branch("m"), branch("m")]),
            Err(BlockGraphError::InvalidAction(InvalidActionError::DuplicateBranchTarget(actual)))
                if actual == target
        ));
        assert!(matches!(
            validate_regions_for_actions(&graph, &[]),
            Err(BlockGraphError::InvalidAction(
                InvalidActionError::MissingResolveForBranch(_)
            ))
        ));
        assert!(matches!(
            validate_regions_for_actions(
                &graph,
                &[Action::Branch {
                    target: ivec3(100, 0, 0),
                    condition: Expr::Var("m".into()),
                }],
            ),
            Err(BlockGraphError::InvalidAction(InvalidActionError::InvalidBranchTarget(site)))
                if site == ivec3(100, 0, 0)
        ));
    }

    #[test]
    fn region_action_index_keeps_branch_order_before_action_order() {
        let ast = crate::parse_blog_program_to_ast(include_str!(
            "../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap();
        let graph = crate::lower_blog_graph_ast_deferred(&ast)
            .unwrap()
            .root()
            .local_body()
            .clone();
        let first = &graph.branches[0];
        let second = &graph.branches[1];
        let actions = [
            Action::Measure {
                target: MeasureTarget::Node(second.on_true.blocks().next().unwrap().pos()),
                name: "second".into(),
            },
            Action::Measure {
                target: MeasureTarget::Node(first.on_true.blocks().next().unwrap().pos()),
                name: "first".into(),
            },
            Action::Branch {
                target: first.target,
                condition: Expr::Var("external".into()),
            },
            Action::Branch {
                target: second.target,
                condition: Expr::Var("external".into()),
            },
        ];
        assert_eq!(
            earliest_region_action_targets(&graph, &actions)[..2],
            [Some(1), Some(0)]
        );
        assert!(matches!(
            validate_regions_for_actions(&graph, &actions),
            Err(BlockGraphError::InvalidAction(InvalidActionError::BranchContainsActionTarget {
                target,
                ordinal: 1,
            })) if target == first.target
        ));
    }

    #[test]
    fn analysis_branch_projections_honor_boolean_limits() {
        let (graph, target) = explicit_branch();
        for (limits, resource) in [
            (
                BooleanLimits {
                    max_nodes: 0,
                    ..BooleanLimits::UNLIMITED
                },
                "Boolean nodes",
            ),
            (
                BooleanLimits {
                    max_steps: 0,
                    ..BooleanLimits::UNLIMITED
                },
                "Boolean work steps",
            ),
        ] {
            assert!(matches!(
                graph.branch_projections_impl(2, true, limits),
                Err(BlockGraphError::Stabilizer(crate::StabilizerError::ResourceLimited {
                    phase, limit: 0, ..
                })) if phase == resource
            ));
        }
        let projections = graph
            .branch_projections_impl(2, true, BooleanLimits::UNLIMITED)
            .unwrap();
        assert_eq!(
            projections
                .iter()
                .map(|projection| projection.assignments().to_vec())
                .collect::<Vec<_>>(),
            [vec![(target, false)], vec![(target, true)]]
        );
    }

    #[test]
    fn terminal_branch_action_analysis_checks_source_limits() {
        use crate::{ModuleCertificationLimits, StabilizerError};

        let (mut graph, _) = explicit_branch();
        assert!(!graph.has_continuing_branches());
        for shown_true in [true, false] {
            graph.set_shown_branch_arm("b0", shown_true).unwrap();
            for (limits, phase) in [
                (
                    ModuleCertificationLimits {
                        max_expanded_blocks: 0,
                        ..ModuleCertificationLimits::DEFAULT
                    },
                    "expanded blocks",
                ),
                (
                    ModuleCertificationLimits {
                        max_occupied_cells: 0,
                        ..ModuleCertificationLimits::DEFAULT
                    },
                    "occupied footprint cells",
                ),
                (
                    ModuleCertificationLimits {
                        max_local_columns: 0,
                        ..ModuleCertificationLimits::DEFAULT
                    },
                    "local ZX columns",
                ),
            ] {
                assert!(matches!(
                    graph.clone().analyze_actions_with_limits(limits),
                    Err(BlockGraphError::Stabilizer(StabilizerError::ResourceLimited {
                        phase: actual_phase,
                        observed: 1,
                        limit: 0,
                    })) if actual_phase == phase
                ));
            }
            graph
                .clone()
                .analyze_actions_with_limits(ModuleCertificationLimits::DEFAULT)
                .unwrap();
        }
    }

    #[test]
    fn arms_may_overlap_and_project_to_authored_contents() {
        let (graph, target) = explicit_branch();
        let false_graph = graph.project_branches([(target, false)]).unwrap();
        let true_graph = graph.project_branches([(target, true)]).unwrap();
        assert_eq!(
            false_graph.get_block(ivec3(0, 0, 1)).map(Block::kind),
            Some(BlockKind::Measurement(Basis::X))
        );
        assert!(false_graph.get_block(ivec3(0, 0, 2)).is_none());
        assert!(true_graph.get_block(ivec3(0, 0, 2)).is_some());
    }

    #[test]
    fn deferred_projection_does_not_run_flat_action_analysis() {
        let (graph, target) = explicit_branch();
        let projected = graph.project_branches_deferred([(target, false)]).unwrap();

        assert!(!projected.action_graph().is_analyzed());
    }

    #[test]
    fn batched_projections_match_individual_projection() {
        let (graph, _) = explicit_branch();
        for projection in graph.branch_projections().unwrap() {
            let individual = graph
                .project_branches(projection.assignments().iter().copied())
                .unwrap();
            assert_eq!(projection.graph().to_blog_text(), individual.to_blog_text());
        }
    }

    #[test]
    fn analysis_projections_include_joint_branch_choices() {
        let (mut graph, first) = explicit_branch();
        let prefix = ivec3(6, 0, 0);
        let target = ivec3(6, 0, 1);
        let reader = ivec3(9, 0, 0);
        graph.add_block(cube(prefix));
        graph.add_block(cube(reader));
        let second = graph
            .try_add_branch_region(
                "b1",
                BranchArm::new(
                    vec![Block::new(target, BlockKind::Measurement(Basis::X))],
                    vec![Pipe::new(prefix, Direction::ZPLUS)],
                ),
                BranchArm::new(
                    vec![cube(target)],
                    vec![Pipe::new(prefix, Direction::ZPLUS)],
                ),
            )
            .unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(ivec3(3, 0, 0)),
                    name: "m".into(),
                },
                Action::Branch {
                    target: first,
                    condition: Expr::Var("m".into()),
                },
                Action::Measure {
                    target: MeasureTarget::Node(reader),
                    name: "n".into(),
                },
                Action::Branch {
                    target: second,
                    condition: Expr::Var("n".into()),
                },
            ])
            .unwrap();

        assert_eq!(graph.branch_projections_for_analysis().unwrap().len(), 4);
    }

    #[test]
    fn projections_keep_external_action_inputs() {
        let (mut graph, target) = explicit_branch();
        graph
            .set_actions_with_inputs(
                vec![Action::Branch {
                    target,
                    condition: Expr::Var("enabled".into()),
                }],
                ["enabled".to_string()],
            )
            .unwrap();

        for projection in graph.branch_projections().unwrap() {
            assert_eq!(
                projection
                    .graph()
                    .action_graph()
                    .inputs()
                    .collect::<Vec<_>>(),
                ["enabled"]
            );
        }
    }

    #[test]
    fn scene_can_switch_the_visible_arm() {
        let (mut graph, _) = explicit_branch();
        graph.set_shown_branch_arm("b0", false).unwrap();
        assert_eq!(
            graph.get_block(ivec3(0, 0, 1)).map(Block::kind),
            Some(BlockKind::Measurement(Basis::X))
        );
        assert!(!graph.branch_by_name("b0").unwrap().shown_true());
        assert!(matches!(
            graph.try_add_block(cube(ivec3(0, 0, 2))),
            Err(BlockGraphError::BlockPositionOccupied(_))
        ));
    }

    #[test]
    fn visible_arm_edits_survive_toggle() {
        let (mut graph, _) = explicit_branch();
        let pos = ivec3(0, 0, 2);
        graph
            .set_block_kind(pos, BlockKind::Cube(CubeKind::XZX))
            .unwrap();
        graph.set_block_tag(pos, "edited").unwrap();
        graph.set_cube_height(pos, "2d".parse().unwrap()).unwrap();
        graph.set_pipe_hadamard(ivec3(0, 0, 1), pos, true).unwrap();
        graph
            .set_pipe_hadamard(ivec3(0, 0, 0), ivec3(0, 0, 1), true)
            .unwrap();
        graph.set_shown_branch_arm("b0", false).unwrap();
        assert!(
            graph
                .get_pipe(ivec3(0, 0, 0), ivec3(0, 0, 1))
                .unwrap()
                .is_hadamard()
        );
        graph.set_shown_branch_arm("b0", true).unwrap();

        let block = graph.get_block(pos).unwrap();
        assert_eq!(block.kind(), BlockKind::Cube(CubeKind::XZX));
        assert_eq!(block.tag(), Some("edited"));
        assert_eq!(block.height(), "2d".parse().unwrap());
        assert!(graph.get_pipe(ivec3(0, 0, 1), pos).unwrap().is_hadamard());
    }

    #[test]
    fn captured_arm_transforms_keep_cuts_and_check_hidden_footprints() {
        let (graph, _) = explicit_branch();
        let arm = graph.branch_by_name("b0").unwrap().on_true();
        let offset = ivec3(3, -2, 4);
        let shifted = arm.try_with_shift(offset).unwrap();
        assert_eq!(shifted.try_with_shift(-offset).unwrap(), *arm);
        for (original, shifted) in arm.pipes().zip(shifted.pipes()) {
            assert_eq!(shifted.src(), original.src() + offset);
            assert_eq!(shifted.dst(), original.dst() + offset);
        }
        let rotation = crate::ModuleRotation::new(crate::UDirection::Z, 1).orientation();
        let rotated = arm.try_with_orientation(rotation).unwrap();
        let rotated_graph = graph.with_orientation_lenient(rotation).unwrap();
        assert_eq!(
            rotated,
            *rotated_graph.branch_by_name("b0").unwrap().on_true()
        );
        assert_eq!(arm.flip_xz_basis(&graph).flip_xz_basis(&graph), *arm);

        let tall = BranchArm::new(
            vec![
                cube(ivec3(0, 0, i32::MAX - 1))
                    .with_height("2d".parse().unwrap())
                    .unwrap(),
            ],
            Vec::new(),
        );
        assert!(matches!(
            tall.try_with_shift(IVec3::Z),
            Err(BlockGraphError::CoordinateOverflow { .. })
        ));
        assert!(matches!(
            tall.try_with_orientation(
                crate::ModuleRotation::new(crate::UDirection::X, 1).orientation(),
            ),
            Err(BlockGraphError::RotationCubeHeightRequiresTimeAxis)
        ));
        let edge = BranchArm::new(
            vec![cube(ivec3(0, 0, i32::MAX - 1))],
            vec![Pipe::new(ivec3(0, 0, i32::MAX - 1), Direction::ZPLUS)],
        );
        assert!(matches!(
            edge.try_with_shift(IVec3::Z),
            Err(BlockGraphError::CoordinateOverflow { .. })
        ));
    }

    #[test]
    fn visible_arm_deletions_survive_toggle_and_serialization() {
        let (mut graph, _) = explicit_branch();
        let terminal = ivec3(0, 0, 2);
        graph.remove_block(terminal).unwrap();
        graph.set_shown_branch_arm("b0", false).unwrap();
        graph.set_shown_branch_arm("b0", true).unwrap();
        assert!(!graph.has_block_at(terminal));
        assert!(!graph.has_pipe_between(IVec3::Z, terminal));
        let reparsed = BlockGraph::from_blog_text(&graph.to_blog_text()).unwrap();
        assert!(!reparsed.has_block_at(terminal));
        reparsed.validate().unwrap();

        graph.remove_pipe(IVec3::ZERO, IVec3::Z).unwrap();
        assert!(graph.branch_regions().is_err(), "a removed cut is invalid");
        graph.set_shown_branch_arm("b0", false).unwrap();
        graph.set_shown_branch_arm("b0", true).unwrap();
        assert!(!graph.has_pipe_between(IVec3::ZERO, IVec3::Z));
        assert!(graph.branch_by_name("b0").unwrap().incoming.is_empty());
        BlockGraph::from_blog_text(&graph.to_blog_text()).unwrap_err();
    }

    #[test]
    fn readding_branch_pipes_repairs_internal_edges_and_incoming_cuts() {
        for (shown_true, src) in [(true, IVec3::Z), (true, IVec3::ZERO), (false, IVec3::ZERO)] {
            let (mut graph, _) = explicit_branch();
            graph.set_shown_branch_arm("b0", shown_true).unwrap();
            let dst = src + IVec3::Z;
            let pipe = graph.remove_pipe(src, dst).unwrap();
            graph.branch_regions().unwrap_err();

            graph.try_add_pipe(pipe).unwrap();
            graph.validate().unwrap();
            graph.set_shown_branch_arm("b0", !shown_true).unwrap();
            graph.set_shown_branch_arm("b0", shown_true).unwrap();
            assert!(graph.has_pipe_between(src, dst));
            let reparsed = BlockGraph::from_blog_text(&graph.to_blog_text()).unwrap();
            reparsed.validate().unwrap();
        }
    }

    #[test]
    fn editor_can_capture_one_arm_then_build_the_other_in_place() {
        let past = ivec3(0, 0, 0);
        let inside = ivec3(0, 0, 1);
        let mut graph = BlockGraph::new();
        graph.add_block(cube(past));
        graph.add_block(cube(inside));
        graph.add_pipe(Pipe::new(past, Direction::ZPLUS));
        graph
            .set_actions_with_inputs(Vec::new(), ["enabled".into()])
            .unwrap();

        graph.take_branch_arm([inside], []).unwrap_err();
        assert!(
            graph.get_block(inside).is_some(),
            "failed capture is atomic"
        );
        assert!(graph.get_pipe(past, inside).is_some());

        let on_true = graph.take_branch_arm([inside], [(past, inside)]).unwrap();
        assert!(graph.get_block(inside).is_none(), "captured arm is hidden");
        graph.restore_branch_arm(&on_true).unwrap();
        let on_true = graph.take_branch_arm([inside], [(past, inside)]).unwrap();
        graph.add_block(Block::new(inside, BlockKind::Measurement(Basis::Z)));
        graph.add_pipe(Pipe::new(past, Direction::ZPLUS).with_tag("false").unwrap());
        let on_false = graph.take_branch_arm([inside], [(past, inside)]).unwrap();

        let target = graph
            .try_add_branch_region("b0", on_false, on_true)
            .unwrap();
        assert_eq!(
            graph.action_graph().inputs().collect::<Vec<_>>(),
            ["enabled"]
        );
        graph
            .set_actions(vec![Action::Branch {
                target,
                condition: Expr::Var("enabled".into()),
            }])
            .unwrap();
        assert_eq!(
            graph.get_block(inside).map(Block::kind),
            Some(BlockKind::Cube(CubeKind::ZXZ)),
            "completed region initially shows its true arm"
        );
    }

    #[test]
    fn branch_arms_reject_ports_t_and_selectives() {
        let past = ivec3(0, 0, 0);
        let inside = ivec3(0, 0, 1);
        let valid = BranchArm::new(vec![cube(inside)], vec![Pipe::new(past, Direction::ZPLUS)]);
        let mut graph = BlockGraph::new();
        graph.add_block(cube(past));

        for kind in [
            BlockKind::Port,
            BlockKind::T,
            BlockKind::Selective(SelectiveKind::XY),
        ] {
            let invalid = BranchArm::new(
                vec![Block::new(inside, kind)],
                vec![Pipe::new(past, Direction::ZPLUS)],
            );
            assert!(matches!(
                graph.try_add_branch_region("b0", invalid, valid.clone()),
                Err(BlockGraphError::InvalidAction(
                    InvalidActionError::InvalidBranchInterface { .. }
                ))
            ));
        }
    }

    #[test]
    fn basis_flip_updates_native_resource_cut_pipes_in_both_arms() {
        let (mut graph, target) = explicit_branch();
        graph.set_block_kind(IVec3::ZERO, BlockKind::T).unwrap();
        for hadamard in [false, true] {
            graph
                .set_pipe_hadamard(IVec3::ZERO, IVec3::Z, hadamard)
                .unwrap();
            let flipped = graph.flip_xz_basis().unwrap();
            for selected in [false, true] {
                let branch = flipped.branch_by_name("b0").unwrap();
                assert_eq!(
                    branch.arm_incoming(selected)[0].pipe.is_hadamard(),
                    !hadamard
                );
                assert_eq!(
                    graph
                        .branch_by_name("b0")
                        .unwrap()
                        .arm(selected)
                        .flip_xz_basis(&graph),
                    *branch.arm(selected),
                    "captured-arm and whole-graph flips share the external T frame"
                );
                let projected = flipped.project_branches([(target, selected)]).unwrap();
                assert_eq!(
                    projected
                        .get_pipe(IVec3::ZERO, IVec3::Z)
                        .unwrap()
                        .is_hadamard(),
                    !hadamard
                );
            }
            assert_eq!(
                flipped.flip_xz_basis().unwrap().to_blog_text(),
                graph.to_blog_text()
            );
        }
    }

    #[test]
    fn basis_flip_keeps_a_hidden_y_arm_and_its_own_hadamard_cut() {
        let mut graph = BlockGraph::new();
        graph.add_block(cube(IVec3::ZERO));
        graph.add_block(cube(ivec3(3, 0, 0)));
        let pipe = Pipe::new(IVec3::ZERO, Direction::ZPLUS);
        let target = graph
            .try_add_branch_region(
                "b0",
                BranchArm::new(vec![Block::new(IVec3::Z, BlockKind::Y)], vec![pipe.clone()]),
                BranchArm::new(vec![cube(IVec3::Z)], vec![pipe]),
            )
            .unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    name: "m".into(),
                    target: MeasureTarget::Node(ivec3(3, 0, 0)),
                },
                Action::Branch {
                    target,
                    condition: Expr::Var("m".into()),
                },
            ])
            .unwrap();
        let flipped = graph.flip_xz_basis().unwrap();
        let region = flipped.branch_by_name("b0").unwrap();
        assert!(region.arm_incoming(false)[0].pipe.is_hadamard());
        assert!(!region.arm_incoming(true)[0].pipe.is_hadamard());
        assert_eq!(
            graph
                .branch_by_name("b0")
                .unwrap()
                .on_false()
                .flip_xz_basis(&graph),
            *region.on_false()
        );
        for selected in [false, true] {
            let expected = graph
                .project_branches([(target, selected)])
                .unwrap()
                .flip_xz_basis()
                .unwrap();
            let actual = flipped.project_branches([(target, selected)]).unwrap();
            assert_eq!(actual.to_blog_text(), expected.to_blog_text());
        }
        assert_eq!(
            flipped.flip_xz_basis().unwrap().to_blog_text(),
            graph.to_blog_text()
        );
    }

    #[test]
    fn different_regions_cannot_overlap_in_either_arm() {
        let (mut graph, _) = explicit_branch();
        let arm = BranchArm::new(
            vec![Block::new(ivec3(0, 0, 1), BlockKind::Measurement(Basis::X))],
            vec![Pipe::new(ivec3(0, 0, 0), Direction::ZPLUS)],
        );
        assert!(matches!(
            graph.try_add_branch_region("b1", arm.clone(), arm),
            Err(BlockGraphError::InvalidAction(
                InvalidActionError::OverlappingBranchRegions { .. }
            ))
        ));
    }

    #[test]
    fn xz_flip_updates_both_stored_arms() {
        let (graph, _) = explicit_branch();
        let flipped = graph.flip_xz_basis().unwrap();
        let branch = flipped.branch_by_name("b0").unwrap();
        assert_eq!(
            branch.on_false().blocks().next().map(Block::kind),
            Some(BlockKind::Measurement(Basis::Z))
        );
        assert_eq!(
            branch.on_true().blocks().next().map(Block::kind),
            Some(BlockKind::Cube(CubeKind::XZX))
        );
    }

    #[test]
    fn rotation_updates_both_stored_arms_and_cut() {
        let (graph, _) = explicit_branch();
        let rotated = graph
            .rotate_about_origin_lenient(crate::UDirection::X, 1)
            .unwrap();
        let branch = rotated.branch_by_name("b0").unwrap();

        assert!(branch.on_false().contains_block(ivec3(0, -1, 0)));
        assert!(branch.on_true().contains_block(ivec3(0, -2, 0)));
        assert_eq!(branch.incoming[0].inside, ivec3(0, -1, 0));
        assert_eq!(branch.incoming[0].pipe.dir(), Direction::YMINUS);
    }

    #[test]
    fn blog_parses_and_round_trips_named_region_and_resolve() {
        let source = r#"BLOG 1.0

  0: ZXZ [0, 0, 0]
  1: ZXZ [3, 0, 0]
  5: ZXZ [2, 0, 0]
  [0, 0, 0] -> +Z
  [2, 0, 0] -> +Z
  branch b0 {
    false {
      2: ZXZ [0, 0, 1] height=2d
      6: ZXZ [1, 0, 1]
      7: X [2, 0, 1]
      [0, 0, 1] -> +X
    }
    true {
      3: ZXZ [0, 0, 1]
      4: ZXZ [0, 0, 2]
      8: Z [2, 0, 1]
      [0, 0, 1] -> +Z
    }
  }

  m = measure 1
  resolve b0 if m
"#;
        let graph = BlockGraph::from_blog_text(source).expect("generic branch BLOG parses");
        assert_eq!(graph.branch_definitions().len(), 1);
        let mut false_cubes = graph.branch_definitions()[0]
            .on_false()
            .blocks()
            .filter(|block| block.kind().is_cube());
        assert!(false_cubes.all(|block| block.height() == "2d".parse().unwrap()));
        let written = graph.to_blog_text();
        assert!(written.contains("branch b0 {"));
        assert!(written.contains("resolve b0 if m"));
        assert_eq!(written.matches("0 -> +Z").count(), 1);
        assert_eq!(written.matches("2 -> +Z").count(), 1);
        assert!(written.contains(": X [2, 0, 1]"));
        assert!(written.contains(": Z [2, 0, 1]"));
        assert_eq!(
            BlockGraph::from_blog_text(&written)
                .expect("canonical generic BLOG reparses")
                .to_blog_text(),
            written
        );
    }
}
