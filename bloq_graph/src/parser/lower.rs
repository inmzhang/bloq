//! Checked lowering from parsed BLOG syntax into block graphs.

use std::collections::{HashMap, HashSet};

use crate::action::{Action, Expr as ActionExpr, FeedbackTarget, MeasureTarget};
use crate::parser::ParseError;
use crate::parser::ast::{
    ActionStmt, BlockDef, BlockKindAst, BranchArmStmt, DataStmt, Expr, FeedbackDef, LetDef,
    MeasureDef, MeasureTargetAst, PipeDef, PipeDst, Ref, ResolveDef, ResolveTargetDef, SourceFile,
    Span, Spanned,
};
use crate::{
    Block, BlockError, BlockGraph, BlockKind, BranchArm, CubeHeight, Direction, InvalidActionError,
    PatchRotationKind, Pipe, WalkingKind,
};
use glam::{IVec2, IVec3};

pub(super) fn lower_with_inputs_and_limits(
    source: &SourceFile,
    inputs: impl IntoIterator<Item = String>,
    limits: crate::ModuleCertificationLimits,
) -> Result<BlockGraph, ParseError> {
    lower_impl(source, inputs, ActionValidation::Full(limits))
}

pub(super) fn lower_deferred_with_inputs(
    source: &SourceFile,
    inputs: impl IntoIterator<Item = String>,
) -> Result<BlockGraph, ParseError> {
    lower_impl(source, inputs, ActionValidation::Deferred)
}

pub(super) fn lower_lenient_with_inputs(
    source: &SourceFile,
    inputs: impl IntoIterator<Item = String>,
) -> Result<BlockGraph, ParseError> {
    lower_impl(source, inputs, ActionValidation::Lenient)
}

enum ActionValidation {
    Full(crate::ModuleCertificationLimits),
    Deferred,
    Lenient,
}

fn lower_impl(
    source: &SourceFile,
    inputs: impl IntoIterator<Item = String>,
    validation: ActionValidation,
) -> Result<BlockGraph, ParseError> {
    let (major, minor) = source.version.node;
    if major != 1 || minor != 0 {
        return Err(ParseError::UnsupportedVersion {
            major,
            minor,
            span: source.version.span,
        });
    }

    let mut graph = BlockGraph::new();
    let mut all_ids: HashMap<u32, Block> = HashMap::new();
    let mut reversed_selectives: HashSet<IVec3> = HashSet::new();
    let mut explicit_heights: HashMap<IVec3, Spanned<CubeHeight>> = HashMap::new();
    let mut branch_targets = HashMap::new();
    let mut lowered_actions = Vec::new();

    // Pass 1: collect every globally unique block ID. Arm blocks are not added
    // to the ordinary graph until their named region is installed.
    {
        let mut register = |def: &BlockDef| -> Result<(), ParseError> {
            if all_ids.contains_key(&def.id.node) {
                return Err(ParseError::DuplicateId {
                    id: def.id.node,
                    span: def.id.span,
                });
            }
            all_ids.insert(def.id.node, lower_block_def(def)?);
            Ok(())
        };
        for stmt in &source.data_stmts {
            match &stmt.node {
                DataStmt::Block(def) => register(def)?,
                DataStmt::Branch(branch) => {
                    for stmt in branch.on_false.iter().chain(&branch.on_true) {
                        if let BranchArmStmt::Block(def) = &stmt.node {
                            register(def)?;
                        }
                    }
                }
                DataStmt::Pipe(_) => {}
            }
        }
    }

    // Collect explicit heights before resolving endpoints. Spatial components
    // can make an unannotated cube tall.
    for stmt in &source.data_stmts {
        if let DataStmt::Block(def) = &stmt.node
            && let Some(height) = &def.height
        {
            explicit_heights.insert(all_ids[&def.id.node].pos(), height.clone());
        }
    }

    let mut false_heights = explicit_heights.clone();
    let mut true_heights = explicit_heights.clone();
    let mut branch_names = HashSet::new();
    for stmt in &source.data_stmts {
        let DataStmt::Branch(def) = &stmt.node else {
            continue;
        };
        if !branch_names.insert(def.name.node.clone()) {
            return Err(ParseError::DuplicateBranchName {
                name: def.name.node.clone(),
                span: def.name.span,
            });
        }
        record_arm_heights(&def.on_false, &all_ids, &mut false_heights);
        record_arm_heights(&def.on_true, &all_ids, &mut true_heights);
    }

    let true_view = id_map_with_inherited_heights(source, &all_ids, &true_heights, true)?;
    let false_view = if branch_names.is_empty() {
        None
    } else {
        Some(id_map_with_inherited_heights(
            source,
            &all_ids,
            &false_heights,
            false,
        )?)
    };
    let true_common = common_id_map(source, &true_view);
    let false_common = false_view
        .as_ref()
        .map(|resolved| common_id_map(source, resolved));

    // Pass 1b: unconditional blocks.
    for stmt in &source.data_stmts {
        if let DataStmt::Block(def) = &stmt.node {
            let block = true_common[&def.id.node].clone();
            if def.selective_order_reversed {
                reversed_selectives.insert(block.pos());
            }
            graph.try_add_block(block).map_err(|e| ParseError::Graph {
                source: Box::new(e),
                span: Some(stmt.span),
            })?;
        }
    }

    let mut lowered_branches = Vec::new();
    for stmt in &source.data_stmts {
        let DataStmt::Branch(def) = &stmt.node else {
            continue;
        };
        let false_local = arm_id_map(&def.on_false, false_view.as_ref().unwrap_or(&true_view));
        let true_local = arm_id_map(&def.on_true, &true_view);
        let on_false = lower_branch_arm(
            &def.on_false,
            &BlockScope {
                common: false_common.as_ref().unwrap_or(&true_common),
                local: Some(&false_local),
            },
        )?;
        let on_true = lower_branch_arm(
            &def.on_true,
            &BlockScope {
                common: &true_common,
                local: Some(&true_local),
            },
        )?;
        lowered_branches.push((def.name.node.clone(), stmt.span, on_false, on_true));
    }

    // Resolve all owners before installing shared seams between them.
    let mut shared_pipes = Vec::new();
    for stmt in &source.data_stmts {
        if let DataStmt::Pipe(def) = &stmt.node {
            shared_pipes.push(lower_pipe_def(def, &BlockScope::common(&true_common))?);
        }
    }
    let names = lowered_branches
        .iter()
        .map(|(name, _, _, _)| name.clone())
        .collect::<Vec<_>>();
    let regions = lowered_branches
        .into_iter()
        .map(|(name, _, on_false, on_true)| (name, on_false, on_true))
        .collect();
    let targets = graph
        .try_add_branch_regions(regions, shared_pipes)
        .map_err(|e| ParseError::Graph {
            source: Box::new(e),
            span: None,
        })?;
    branch_targets.extend(names.into_iter().zip(targets));

    // The ID map hands out endpoints to Pass 3 (`Block::endpoint_for_direction`
    // on a multi-cell cube), so update its unconditional blocks.
    let mut action_ids = true_common;
    for stmt in &source.data_stmts {
        if let DataStmt::Block(def) = &stmt.node {
            let position = action_ids[&def.id.node].pos();
            action_ids.insert(
                def.id.node,
                graph
                    .get_block(position)
                    .expect("unconditional block exists")
                    .clone(),
            );
        }
    }

    // Pass 3: actions — flat list, no time dimension. The block-id lookup mirrors
    // the editor path's callback: `None` yields the anchor, `Some(dir)` the
    // exposed endpoint via `Block::endpoint_for_direction`.
    let i2p = |id: u32, dir: Option<Direction>| -> Option<IVec3> {
        let block = action_ids.get(&id)?;
        Some(dir.map_or_else(|| block.pos(), |dir| block.endpoint_for_direction(dir)))
    };
    for stmt in &source.action_stmts {
        let action = lower_action_stmt(&stmt.node, &i2p, &reversed_selectives, &branch_targets)?;
        lowered_actions.push(action);
    }

    let result = match validation {
        ActionValidation::Full(limits) => graph.validate_resource_limits(limits).and_then(|()| {
            graph.set_actions_with_inputs_and_limits(lowered_actions, inputs, limits)
        }),
        ActionValidation::Deferred => {
            graph.set_actions_deferred_with_inputs(lowered_actions, inputs)
        }
        ActionValidation::Lenient => graph
            .set_actions_deferred_with_inputs(Vec::new(), inputs)
            .and_then(|()| graph.set_actions_lenient(lowered_actions)),
    };
    result.map_err(|e| ParseError::Graph {
        span: action_error_span(
            &e,
            &source.action_stmts,
            &source.data_stmts,
            &i2p,
            &branch_targets,
        ),
        source: Box::new(e),
    })?;

    Ok(graph)
}

pub(super) fn lower_actions(
    stmts: &[Spanned<ActionStmt>],
    i2p: impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<Vec<Action>, ParseError> {
    // Standalone action snippets carry no block definitions, so no
    // reversed-selective canonicalization applies on this path.
    let reversed_selectives = HashSet::new();
    let branch_targets = HashMap::new();
    stmts
        .iter()
        .map(|stmt| lower_action_stmt(&stmt.node, &i2p, &reversed_selectives, &branch_targets))
        .collect()
}

fn lower_branch_arm(
    stmts: &[Spanned<BranchArmStmt>],
    id_map: &BlockScope<'_>,
) -> Result<BranchArm, ParseError> {
    let mut blocks = Vec::new();
    let mut pipes = Vec::new();
    for stmt in stmts {
        match &stmt.node {
            BranchArmStmt::Block(def) => blocks.push(
                id_map
                    .get(def.id.node)
                    .expect("arm scope contains its block")
                    .clone(),
            ),
            BranchArmStmt::Pipe(def) => pipes.push(lower_pipe_def(def, id_map)?),
        }
    }
    Ok(BranchArm::new(blocks, pipes))
}

fn record_arm_heights(
    stmts: &[Spanned<BranchArmStmt>],
    id_map: &HashMap<u32, Block>,
    heights: &mut HashMap<IVec3, Spanned<CubeHeight>>,
) {
    for stmt in stmts {
        if let BranchArmStmt::Block(block) = &stmt.node
            && let Some(height) = &block.height
        {
            heights.insert(id_map[&block.id.node].pos(), height.clone());
        }
    }
}

fn common_id_map(source: &SourceFile, resolved: &HashMap<u32, Block>) -> HashMap<u32, Block> {
    source
        .data_stmts
        .iter()
        .filter_map(|stmt| match &stmt.node {
            DataStmt::Block(def) => Some((def.id.node, resolved[&def.id.node].clone())),
            DataStmt::Pipe(_) | DataStmt::Branch(_) => None,
        })
        .collect()
}

fn arm_id_map(
    arm: &[Spanned<BranchArmStmt>],
    resolved: &HashMap<u32, Block>,
) -> HashMap<u32, Block> {
    arm.iter()
        .filter_map(|stmt| match &stmt.node {
            BranchArmStmt::Block(def) => Some((def.id.node, resolved[&def.id.node].clone())),
            BranchArmStmt::Pipe(_) => None,
        })
        .collect()
}

fn id_map_with_inherited_heights(
    source: &SourceFile,
    id_map: &HashMap<u32, Block>,
    explicit: &HashMap<IVec3, Spanned<CubeHeight>>,
    true_arm: bool,
) -> Result<HashMap<u32, Block>, ParseError> {
    let mut resolved = HashMap::new();
    let mut pipes = Vec::new();

    for stmt in &source.data_stmts {
        match &stmt.node {
            DataStmt::Block(def) => {
                resolved.insert(def.id.node, id_map[&def.id.node].clone());
            }
            DataStmt::Branch(branch) => {
                let arm = if true_arm {
                    &branch.on_true
                } else {
                    &branch.on_false
                };
                for stmt in arm {
                    match &stmt.node {
                        BranchArmStmt::Block(def) => {
                            resolved.insert(def.id.node, id_map[&def.id.node].clone());
                        }
                        BranchArmStmt::Pipe(def) => pipes.push(def),
                    }
                }
            }
            DataStmt::Pipe(def) => pipes.push(def),
        }
    }
    if explicit.is_empty() {
        return Ok(resolved);
    }

    let anchors = resolved
        .values()
        .filter(|block| block.kind().is_cube())
        .map(Block::pos)
        .collect::<HashSet<_>>();
    let mut adjacency = HashMap::<IVec3, Vec<IVec3>>::new();
    for def in pipes {
        let Some((src, dst)) = height_spatial_endpoints(def, &resolved)? else {
            continue;
        };
        if anchors.contains(&src) && anchors.contains(&dst) {
            adjacency.entry(src).or_default().push(dst);
            adjacency.entry(dst).or_default().push(src);
        }
    }

    // Flood explicit heights instead of rescanning every pipe after each growth.
    let mut sources = explicit.iter().collect::<Vec<_>>();
    sources.sort_by_key(|(pos, _)| pos.to_array());
    let mut heights = HashMap::<IVec3, (IVec3, &Spanned<CubeHeight>)>::new();
    for (&source_pos, height) in sources {
        let mut pending = vec![source_pos];
        while let Some(pos) = pending.pop() {
            if let Some((other_pos, other)) = heights.get(&pos) {
                if other.node != height.node {
                    return Err(ParseError::ConflictingCubeHeights {
                        pos: source_pos,
                        height: height.node,
                        other_pos: *other_pos,
                        other_height: other.node,
                        span: height.span,
                    });
                }
                continue;
            }
            heights.insert(pos, (source_pos, height));
            pending.extend(adjacency.get(&pos).into_iter().flatten().copied());
        }
    }
    for block in resolved.values_mut().filter(|block| block.kind().is_cube()) {
        let Some((_, height)) = heights.get(&block.pos()) else {
            continue;
        };
        if block.height() != height.node {
            *block =
                block
                    .clone()
                    .with_height(height.node)
                    .map_err(|source| ParseError::Graph {
                        source: Box::new(source.into()),
                        span: Some(height.span),
                    })?;
        }
    }
    Ok(resolved)
}

fn height_spatial_endpoints(
    def: &PipeDef,
    id_map: &HashMap<u32, Block>,
) -> Result<Option<(IVec3, IVec3)>, ParseError> {
    let position = |reference: &Ref| match reference {
        Ref::Pos(pos) => Some(*pos),
        Ref::Id(id) => id_map.get(id).map(Block::pos),
    };
    Ok(match &def.dst.node {
        PipeDst::Dir(dir) if dir.is_spatial() => position(&def.src.node)
            .map(|src| checked_step(src, *dir, def.dst.span).map(|dst| (src, dst)))
            .transpose()?,
        PipeDst::Ref(dst) => position(&def.src.node)
            .zip(position(dst))
            .and_then(|(src, dst)| {
                Direction::iter()
                    .filter(|dir| dir.is_spatial())
                    .find(|dir| checked_step(src, *dir, def.dst.span).is_ok_and(|next| next == dst))
                    .map(|_| (src, dst))
            }),
        PipeDst::Dir(_) => None,
    })
}

// --- Internal helpers ---

struct BlockScope<'a> {
    common: &'a HashMap<u32, Block>,
    local: Option<&'a HashMap<u32, Block>>,
}

impl<'a> BlockScope<'a> {
    fn common(common: &'a HashMap<u32, Block>) -> Self {
        Self {
            common,
            local: None,
        }
    }

    fn get(&self, id: u32) -> Option<&Block> {
        self.local
            .and_then(|local| local.get(&id))
            .or_else(|| self.common.get(&id))
    }
}

/// Resolves an unresolved AST `Ref` to a concrete position through the block-id
/// lookup `i2p`. `None` selects the block anchor, so both lowering paths share
/// one closure instead of a `Resolver` trait with per-path implementations.
fn resolve_ref(
    r: &Ref,
    span: Span,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<IVec3, ParseError> {
    match r {
        Ref::Pos(pos) => Ok(*pos),
        Ref::Id(id) => i2p(*id, None).ok_or(ParseError::UndefinedId { id: *id, span }),
    }
}

/// Resolves a measure-edge source `Ref`, selecting the endpoint facing `dir`
/// (via `Some(dir)`) so a measure edge on a multi-cell block (walking, scaled
/// cube) picks the exposed face rather than the anchor.
fn resolve_edge_src_ref(
    r: &Ref,
    span: Span,
    dir: Direction,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<IVec3, ParseError> {
    match r {
        Ref::Pos(pos) => Ok(*pos),
        Ref::Id(id) => i2p(*id, Some(dir)).ok_or(ParseError::UndefinedId { id: *id, span }),
    }
}

fn lower_block_def(def: &BlockDef) -> Result<Block, ParseError> {
    let kind = match def.kind.node {
        BlockKindAst::Concrete(kind) => {
            if let Some(end) = &def.end {
                return Err(ParseError::Syntax {
                    message: format!(
                        "block kind {:?} does not take an end position",
                        def.kind.node
                    ),
                    span: end.span,
                });
            }
            kind
        }
        BlockKindAst::WalkingBoundary(boundary) => {
            let Some(end) = &def.end else {
                return Err(ParseError::Syntax {
                    message: format!("walking block {:?} requires an end position", def.kind.node),
                    span: def.kind.span,
                });
            };
            let delta = checked_delta(def.pos.node, end.node, end.span)?;
            let walking_kind =
                WalkingKind::new(boundary, IVec2::new(delta.x, delta.y)).map_err(|e| {
                    ParseError::InvalidWalkingBlock {
                        start: def.pos.node,
                        end: end.node,
                        span: end.span,
                        message: e.to_string(),
                    }
                })?;
            if delta.z != 1 {
                return Err(ParseError::InvalidWalkingBlock {
                    start: def.pos.node,
                    end: end.node,
                    span: end.span,
                    message: "walking end must be exactly one time layer above start".to_string(),
                });
            }
            BlockKind::Walking(walking_kind)
        }
        BlockKindAst::PatchRotationBasis(basis) => {
            let Some(end) = &def.end else {
                return Err(ParseError::Syntax {
                    message: format!(
                        "patch rotation block {:?} requires an end position",
                        def.kind.node
                    ),
                    span: def.kind.span,
                });
            };
            let delta = checked_delta(def.pos.node, end.node, end.span)?;
            if delta.z != 1 {
                return Err(ParseError::InvalidPatchRotationBlock {
                    start: def.pos.node,
                    end: end.node,
                    span: end.span,
                    message: "patch rotation end must be exactly one time layer above start"
                        .to_string(),
                });
            }
            let kind =
                PatchRotationKind::new(basis, IVec2::new(delta.x, delta.y)).map_err(|e| {
                    ParseError::InvalidPatchRotationBlock {
                        start: def.pos.node,
                        end: end.node,
                        span: end.span,
                        message: e.to_string(),
                    }
                })?;
            BlockKind::PatchRotation(kind)
        }
    };
    let mut block = Block::new(def.pos.node, kind);
    if let Some(height) = &def.height {
        if !kind.is_cube() {
            return Err(ParseError::Syntax {
                message: "cube height modifier `height=` is only valid for cube blocks".to_string(),
                span: height.span,
            });
        }
        block = match block.with_height(height.node) {
            Ok(block) => block,
            Err(BlockError::CubeHeightCoordinateOverflow { .. }) => {
                return Err(ParseError::CoordinateOverflow {
                    span: height.span,
                    message: format!(
                        "cube at {} with height {} exceeds the graph coordinate range",
                        def.pos.node, height.node
                    ),
                });
            }
            Err(error) => {
                return Err(ParseError::Syntax {
                    message: error.to_string(),
                    span: height.span,
                });
            }
        };
    }
    if let Some(color) = &def.color {
        block = block
            .with_port_color(color.node)
            .map_err(|error| ParseError::Syntax {
                message: error.to_string(),
                span: color.span,
            })?;
    }
    if let Some(role) = &def.role {
        block = block
            .with_port_role(role.node)
            .map_err(|error| ParseError::Syntax {
                message: error.to_string(),
                span: role.span,
            })?;
    }
    if let Some(tag) = &def.tag {
        block = block
            .with_tag(tag.node.clone())
            .expect("parser tag grammar guarantees a valid tag");
    }
    Ok(block)
}

fn lower_pipe_def(def: &PipeDef, id_map: &BlockScope<'_>) -> Result<Pipe, ParseError> {
    let (src_pos, direction) = match &def.dst.node {
        PipeDst::Dir(dir) => (
            resolve_pipe_src_endpoint(&def.src.node, def.src.span, *dir, id_map)?,
            *dir,
        ),
        PipeDst::Ref(r) => {
            let (src_pos, dst_pos) =
                resolve_pipe_ref_endpoint_pair(&def.src, r, def.dst.span, id_map)?;
            let direction = Direction::try_from(checked_delta(src_pos, dst_pos, def.dst.span)?)
                .map_err(|e| ParseError::InvalidDirection {
                    from: src_pos,
                    to: dst_pos,
                    span: def.dst.span,
                    message: e.to_string(),
                })?;
            (src_pos, direction)
        }
    };
    checked_step(src_pos, direction, def.dst.span)?;
    let mut pipe = Pipe::new(src_pos, direction);
    if def.hadamard {
        pipe = pipe.with_hadamard();
    }
    if let Some(tag) = &def.tag {
        pipe = pipe
            .with_tag(tag.node.clone())
            .expect("parser tag grammar guarantees a valid tag");
    }
    Ok(pipe)
}

fn resolve_pipe_src_endpoint(
    src: &Ref,
    span: Span,
    dir: Direction,
    id_map: &BlockScope<'_>,
) -> Result<IVec3, ParseError> {
    match src {
        Ref::Pos(pos) => Ok(*pos),
        Ref::Id(id) => {
            let block = resolve_block_id(*id, span, id_map)?;
            Ok(block.endpoint_for_direction(dir))
        }
    }
}

fn resolve_pipe_ref_endpoint_pair(
    src: &Spanned<Ref>,
    dst: &Ref,
    dst_span: Span,
    id_map: &BlockScope<'_>,
) -> Result<(IVec3, IVec3), ParseError> {
    let src_candidates = pipe_ref_endpoint_candidates(&src.node, src.span, id_map)?;
    let dst_candidates = pipe_ref_endpoint_candidates(dst, dst_span, id_map)?;
    let mut adjacent = None;
    let mut rejected_spatial = false;
    for src_pos in &src_candidates {
        for dst_pos in &dst_candidates {
            if u64::from(src_pos.x.abs_diff(dst_pos.x))
                + u64::from(src_pos.y.abs_diff(dst_pos.y))
                + u64::from(src_pos.z.abs_diff(dst_pos.z))
                != 1
            {
                continue;
            }
            let delta = *dst_pos - *src_pos;
            let direction = Direction::try_from(delta).expect("unit delta is a direction");
            if direction.is_spatial()
                && (!spatial_cube_ref_uses_anchor(&src.node, *src_pos, id_map)
                    || !spatial_cube_ref_uses_anchor(dst, *dst_pos, id_map))
            {
                rejected_spatial = true;
                continue;
            }
            if adjacent.replace((*src_pos, *dst_pos)).is_some() {
                return Err(ParseError::Syntax {
                    message: concat!(
                        "pipe references expose multiple adjacent endpoint pairs; ",
                        "use a position or direction"
                    )
                    .to_string(),
                    span: Span {
                        start: src.span.start,
                        end: dst_span.end,
                    },
                });
            }
        }
    }
    if let Some(pair) = adjacent {
        return Ok(pair);
    }
    if rejected_spatial {
        return Err(ParseError::Syntax {
            message: "spatial pipes attach to a cube's anchor".to_string(),
            span: Span {
                start: src.span.start,
                end: dst_span.end,
            },
        });
    }
    // A referenced block that exposes no connectable face yields no candidates;
    // report that distinctly rather than as a not-adjacent error between
    // sentinel coordinates.
    let (Some(&from), Some(&to)) = (src_candidates.first(), dst_candidates.first()) else {
        let (span, side) = if src_candidates.is_empty() {
            (src.span, "source")
        } else {
            (dst_span, "destination")
        };
        return Err(ParseError::Syntax {
            message: format!("pipe {side} block exposes no connectable face"),
            span,
        });
    };
    // Both blocks have candidate faces but none are one step apart, so they
    // cannot be joined by a unit pipe.
    Err(ParseError::InvalidDirection {
        from,
        to,
        span: Span {
            start: src.span.start,
            end: dst_span.end,
        },
        message: "pipe endpoints are not adjacent".to_string(),
    })
}

fn spatial_cube_ref_uses_anchor(reference: &Ref, endpoint: IVec3, id_map: &BlockScope<'_>) -> bool {
    let Ref::Id(id) = reference else {
        return true;
    };
    id_map
        .get(*id)
        .is_none_or(|block| !block.kind().is_cube() || endpoint == block.pos())
}

fn pipe_ref_endpoint_candidates(
    r: &Ref,
    span: Span,
    id_map: &BlockScope<'_>,
) -> Result<Vec<IVec3>, ParseError> {
    match r {
        Ref::Pos(pos) => Ok(vec![*pos]),
        Ref::Id(id) => {
            let block = resolve_block_id(*id, span, id_map)?;
            Ok(block
                .connectable_offsets()
                .into_iter()
                .map(|offset| block.pos() + offset)
                .collect())
        }
    }
}

fn checked_delta(from: IVec3, to: IVec3, span: Span) -> Result<IVec3, ParseError> {
    let Some(x) = to.x.checked_sub(from.x) else {
        return Err(coordinate_overflow(from, to, span));
    };
    let Some(y) = to.y.checked_sub(from.y) else {
        return Err(coordinate_overflow(from, to, span));
    };
    let Some(z) = to.z.checked_sub(from.z) else {
        return Err(coordinate_overflow(from, to, span));
    };
    Ok(IVec3::new(x, y, z))
}

fn checked_step(pos: IVec3, direction: Direction, span: Span) -> Result<IVec3, ParseError> {
    crate::checked_add_position(pos, direction.to_ivec3()).map_err(|_| {
        ParseError::CoordinateOverflow {
            span,
            message: format!("stepping from {pos} toward {direction} exceeds the graph range"),
        }
    })
}

fn coordinate_overflow(from: IVec3, to: IVec3, span: Span) -> ParseError {
    ParseError::CoordinateOverflow {
        span,
        message: format!("difference from {from} to {to} exceeds the graph coordinate range"),
    }
}

fn resolve_block_id<'a>(
    id: u32,
    span: Span,
    id_map: &'a BlockScope<'_>,
) -> Result<&'a Block, ParseError> {
    id_map.get(id).ok_or(ParseError::UndefinedId { id, span })
}

fn action_error_span(
    error: &crate::BlockGraphError,
    statements: &[Spanned<ActionStmt>],
    data: &[Spanned<DataStmt>],
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
    branches: &HashMap<String, IVec3>,
) -> Option<Span> {
    let action = match error {
        crate::BlockGraphError::InvalidAction(error) => error,
        crate::BlockGraphError::Stabilizer(
            crate::StabilizerError::MeasurementSurfaceUnavailable { mvar },
        ) => return find_measure_name_span(statements, mvar),
        crate::BlockGraphError::PipeNotFound(src, dst)
        | crate::BlockGraphError::MeasurementEdgeMissingTimeBasis(src, dst) => {
            return find_action_span(statements, 0, |statement| {
                resolved_measure_target(statement, i2p).is_some_and(|target| match target {
                    MeasureTarget::Edge { src: from, dir } => {
                        from == *src
                            && crate::checked_add_position(from, dir.to_ivec3()).ok() == Some(*dst)
                    }
                    MeasureTarget::Node(_) => false,
                })
            });
        }
        _ => return None,
    };
    match action {
        InvalidActionError::DependencyCycle { ordinal }
        | InvalidActionError::BranchContainsActionTarget { ordinal, .. } => {
            statements.get(*ordinal).map(|statement| statement.span)
        }
        InvalidActionError::InvalidActionName(name) => find_action_span(statements, 0, |action| {
            action_name(action) == Some(name.as_str())
        }),
        InvalidActionError::VariableRedefinition(name) => {
            find_action_span(statements, 1, |action| {
                action_name(action) == Some(name.as_str())
            })
            .or_else(|| {
                find_action_span(statements, 0, |action| {
                    action_name(action) == Some(name.as_str())
                })
            })
        }
        InvalidActionError::UndefinedVariable(name) => find_action_span(statements, 0, |action| {
            action_expression(action).is_some_and(|expr| expr_mentions(expr, name))
        }),
        InvalidActionError::TimeLikeMeasurementEdge(src, dir) => find_measure_span(
            statements,
            MeasureTarget::Edge {
                src: *src,
                dir: *dir,
            },
            i2p,
            0,
        ),
        InvalidActionError::InvalidMeasurementNode(position) => {
            find_measure_span(statements, MeasureTarget::Node(*position), i2p, 0)
        }
        InvalidActionError::DuplicateMeasurementTarget(target) => {
            find_measure_span(statements, *target, i2p, 1)
        }
        InvalidActionError::InvalidResolveTarget(position)
        | InvalidActionError::InvalidBranchTarget(position)
        | InvalidActionError::BranchControllerInside {
            target: position, ..
        } => find_resolve_span(statements, *position, i2p, branches, 0),
        InvalidActionError::DuplicateResolveTarget(position)
        | InvalidActionError::DuplicateBranchTarget(position) => {
            find_resolve_span(statements, *position, i2p, branches, 1)
        }
        InvalidActionError::InvalidFeedbackTarget(position) => {
            find_action_span(statements, 0, |action| match action {
                ActionStmt::Feedback(feedback) => feedback.targets.iter().any(|target| {
                    resolve_ref(&target.target.node, target.target.span, i2p).ok()
                        == Some(*position)
                }),
                _ => false,
            })
        }
        InvalidActionError::MissingResolveForSelective { target } => find_block_span(data, *target),
        InvalidActionError::MissingResolveForBranch(name) => find_branch_span(data, name),
        _ => None,
    }
}

fn find_block_span(statements: &[Spanned<DataStmt>], target: IVec3) -> Option<Span> {
    statements
        .iter()
        .find_map(|statement| match &statement.node {
            DataStmt::Block(block) if block.pos.node == target => Some(statement.span),
            DataStmt::Branch(branch) => {
                branch
                    .on_false
                    .iter()
                    .chain(&branch.on_true)
                    .find_map(|statement| match &statement.node {
                        BranchArmStmt::Block(block) if block.pos.node == target => {
                            Some(statement.span)
                        }
                        _ => None,
                    })
            }
            _ => None,
        })
}

fn find_branch_span(statements: &[Spanned<DataStmt>], name: &str) -> Option<Span> {
    statements
        .iter()
        .find_map(|statement| match &statement.node {
            DataStmt::Branch(branch) if branch.name.node == name => Some(statement.span),
            _ => None,
        })
}

fn find_measure_name_span(statements: &[Spanned<ActionStmt>], name: &str) -> Option<Span> {
    find_action_span(
        statements,
        0,
        |action| matches!(action, ActionStmt::Measure(definition) if definition.name.node == name),
    )
}

fn find_action_span(
    statements: &[Spanned<ActionStmt>],
    nth: usize,
    mut matches: impl FnMut(&ActionStmt) -> bool,
) -> Option<Span> {
    statements
        .iter()
        .filter(|statement| matches(&statement.node))
        .nth(nth)
        .map(|statement| statement.span)
}

fn action_name(action: &ActionStmt) -> Option<&str> {
    match action {
        ActionStmt::Measure(definition) => Some(&definition.name.node),
        ActionStmt::Let(definition) => Some(&definition.name.node),
        _ => None,
    }
}

fn action_expression(action: &ActionStmt) -> Option<&Expr> {
    match action {
        ActionStmt::Let(definition) => Some(&definition.expr.node),
        ActionStmt::DiscardIf(expression) => Some(&expression.node),
        ActionStmt::Resolve(definition) => Some(&definition.condition.node),
        ActionStmt::Feedback(definition) => definition
            .condition
            .as_ref()
            .map(|expression| &expression.node),
        ActionStmt::Measure(_) => None,
    }
}

fn expr_mentions(expression: &Expr, name: &str) -> bool {
    match expression {
        Expr::Var(variable) => variable == name,
        Expr::Not(inner) => expr_mentions(&inner.node, name),
        Expr::Binary(_, left, right) => {
            expr_mentions(&left.node, name) || expr_mentions(&right.node, name)
        }
    }
}

fn resolved_measure_target(
    action: &ActionStmt,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Option<MeasureTarget> {
    let ActionStmt::Measure(definition) = action else {
        return None;
    };
    match &definition.target.node {
        MeasureTargetAst::Node(target) => resolve_ref(target, definition.target.span, i2p)
            .ok()
            .map(MeasureTarget::Node),
        MeasureTargetAst::Edge(target, dir) => {
            resolve_edge_src_ref(target, definition.target.span, *dir, i2p)
                .ok()
                .map(|src| MeasureTarget::Edge { src, dir: *dir })
        }
    }
}

fn find_measure_span(
    statements: &[Spanned<ActionStmt>],
    target: MeasureTarget,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
    nth: usize,
) -> Option<Span> {
    find_action_span(statements, nth, |action| {
        resolved_measure_target(action, i2p) == Some(target)
    })
}

fn find_resolve_span(
    statements: &[Spanned<ActionStmt>],
    target: IVec3,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
    branches: &HashMap<String, IVec3>,
    nth: usize,
) -> Option<Span> {
    find_action_span(statements, nth, |action| {
        let ActionStmt::Resolve(definition) = action else {
            return false;
        };
        match &definition.target.node {
            ResolveTargetDef::Ref(reference) => {
                resolve_ref(reference, definition.target.span, i2p).ok() == Some(target)
            }
            ResolveTargetDef::Branch(name) => branches.get(name) == Some(&target),
        }
    })
}

// --- Action lowering ---

fn lower_action_stmt(
    stmt: &ActionStmt,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
    reversed_selectives: &HashSet<IVec3>,
    branch_targets: &HashMap<String, IVec3>,
) -> Result<Action, ParseError> {
    match stmt {
        ActionStmt::Measure(def) => lower_measure_def(def, i2p),
        ActionStmt::Let(def) => Ok(lower_let_def(def)),
        ActionStmt::DiscardIf(expr) => Ok(Action::DiscardIf(lower_expr(&expr.node))),
        ActionStmt::Resolve(def) => {
            lower_resolve_def(def, i2p, reversed_selectives, branch_targets)
        }
        ActionStmt::Feedback(def) => lower_feedback_def(def, i2p),
    }
}

fn lower_measure_def(
    def: &MeasureDef,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<Action, ParseError> {
    let target = match &def.target.node {
        MeasureTargetAst::Node(r) => MeasureTarget::Node(resolve_ref(r, def.target.span, i2p)?),
        MeasureTargetAst::Edge(r, dir) => {
            let src = resolve_edge_src_ref(r, def.target.span, *dir, i2p)?;
            checked_step(src, *dir, def.target.span)?;
            MeasureTarget::Edge { src, dir: *dir }
        }
    };
    Ok(Action::Measure {
        target,
        name: def.name.node.clone(),
    })
}

fn lower_let_def(def: &LetDef) -> Action {
    Action::Let {
        name: def.name.node.clone(),
        expr: lower_expr(&def.expr.node),
    }
}

pub(super) fn lower_expr(expr: &Expr) -> ActionExpr {
    match expr {
        Expr::Var(name) => ActionExpr::Var(name.clone()),
        Expr::Not(inner) => ActionExpr::Not(Box::new(lower_expr(&inner.node))),
        Expr::Binary(op, lhs, rhs) => ActionExpr::Binary(
            *op,
            Box::new(lower_expr(&lhs.node)),
            Box::new(lower_expr(&rhs.node)),
        ),
    }
}

fn lower_resolve_def(
    def: &ResolveDef,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
    reversed_selectives: &HashSet<IVec3>,
    branch_targets: &HashMap<String, IVec3>,
) -> Result<Action, ParseError> {
    match &def.target.node {
        ResolveTargetDef::Branch(name) => Ok(Action::Branch {
            target: branch_targets.get(name).copied().ok_or_else(|| {
                ParseError::UndefinedBranchName {
                    name: name.clone(),
                    span: def.target.span,
                }
            })?,
            condition: lower_expr(&def.condition.node),
        }),
        ResolveTargetDef::Ref(reference) => {
            let target = resolve_ref(reference, def.target.span, i2p)?;
            let mut condition = lower_expr(&def.condition.node);
            if reversed_selectives.contains(&target) {
                condition = condition.negated();
            }
            Ok(Action::Resolve { target, condition })
        }
    }
}

fn lower_feedback_def(
    def: &FeedbackDef,
    i2p: &impl Fn(u32, Option<Direction>) -> Option<IVec3>,
) -> Result<Action, ParseError> {
    let targets = def
        .targets
        .iter()
        .map(|t| {
            let direction = t.direction.as_ref().map(|direction| direction.node);
            let target = if let Some(direction) = direction {
                let src = resolve_edge_src_ref(&t.target.node, t.target.span, direction, i2p)?;
                checked_step(src, direction, t.target.span)?;
                src
            } else {
                resolve_ref(&t.target.node, t.target.span, i2p)?
            };
            Ok(FeedbackTarget {
                pauli: t.pauli.node,
                target,
                direction,
            })
        })
        .collect::<Result<Vec<_>, ParseError>>()?;
    let condition = def.condition.as_ref().map(|c| lower_expr(&c.node));
    Ok(Action::Feedback { targets, condition })
}

#[cfg(test)]
mod tests {
    use super::action_error_span;
    use crate::action::{BinaryOp, Expr};
    use crate::parser::{ParseError, parse_actions, parse_blog_body as parse_blog_to_graph};
    use crate::{
        Action, Basis, Block, BlockKind, CubeKind, Direction, MeasureTarget, PatchRotationKind,
        SelectiveKind, WalkingBoundaryKind, WalkingKind,
    };
    use glam::IVec3;
    use std::collections::HashMap;

    #[test]
    fn test_operator_precedence_xor_between_and_or() {
        // a | b ^ c & d should parse as a | (b ^ (c & d))
        let input = "BLOG 1.0\n\n0: XZZ [0,0,0]\n1: XZZ [1,0,0]\n2: XZZ [2,0,0]\n3: XZZ [3,0,0]\n\na = measure 0\nb = measure 1\nc = measure 2\nd = measure 3\nx = a | b ^ c & d";
        let graph = parse_blog_to_graph(input).expect("xor precedence test should parse");
        let actions = graph.actions();
        let Action::Let { expr, .. } = actions.last().expect("let exists") else {
            panic!("expected Let");
        };
        let Expr::Binary(BinaryOp::Or, lhs, rhs) = expr else {
            panic!("expected Or at top level");
        };
        assert_eq!(**lhs, Expr::Var("a".to_string()));
        let Expr::Binary(BinaryOp::Xor, xor_lhs, xor_rhs) = rhs.as_ref() else {
            panic!("expected Xor on rhs");
        };
        assert_eq!(**xor_lhs, Expr::Var("b".to_string()));
        assert!(matches!(
            xor_rhs.as_ref(),
            Expr::Binary(BinaryOp::And, _, _)
        ));
    }

    #[test]
    fn test_undefined_id_error() {
        let input = "BLOG 1.0\n\n0: XZZ [0,0,0]\n0 -> 99";
        let err = parse_blog_to_graph(input).expect_err("undefined ID 99 should produce error");
        assert!(matches!(err, ParseError::UndefinedId { id: 99, .. }));
        assert!(err.span().is_some());
    }

    #[test]
    fn action_validation_error_keeps_its_statement_span() {
        let source = "BLOG 1.0\n\n0: XZZ [0,0,0]\nm = measure 0\nbad = missing\n";
        let error = parse_blog_to_graph(source).expect_err("missing variable is invalid");
        let span = error.span().expect("action error has a source span");

        assert_eq!(
            &source[span.start as usize..span.end as usize],
            "bad = missing"
        );
    }

    #[test]
    fn physical_measurement_error_keeps_the_measure_span() {
        let source = "BLOG 1.0\n\n0: XZZ [0,0,0]\nm = measure 0\n";
        let ast = crate::parse_blog_to_ast(source).unwrap();
        let error = crate::BlockGraphError::Stabilizer(
            crate::StabilizerError::MeasurementSurfaceUnavailable {
                mvar: "m".to_string(),
            },
        );
        let span = action_error_span(
            &error,
            &ast.action_stmts,
            &ast.data_stmts,
            &|_, _| None,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            &source[span.start as usize..span.end as usize],
            "m = measure 0"
        );
    }

    #[test]
    fn missing_resolve_errors_keep_the_target_span() {
        for (source, expected) in [
            (
                "BLOG 1.0\n\n0: ZXZ [0,0,0]\n1: XY [0,0,1]\n2: Z [2,0,0]\n0 -> +Z\nm = measure 2\n",
                "1: XY [0,0,1]",
            ),
            (
                "BLOG 1.0\n\n0: ZXZ [0,0,0]\n9: Z [3,0,0]\nbranch b {\n  false {\n    1: ZXZ [0,0,1]\n    0 -> +Z\n  }\n  true {\n    2: ZXZ [0,0,1]\n    0 -> +Z\n  }\n}\nm = measure 9\n",
                "branch b {\n  false {\n    1: ZXZ [0,0,1]\n    0 -> +Z\n  }\n  true {\n    2: ZXZ [0,0,1]\n    0 -> +Z\n  }\n}",
            ),
        ] {
            let error = parse_blog_to_graph(source).expect_err("resolve is required");
            let span = error.span().unwrap();
            assert_eq!(&source[span.start as usize..span.end as usize], expected);
        }
    }

    #[test]
    fn test_duplicate_id_error() {
        for input in [
            "BLOG 1.0\n\n  0: XZZ [0,0,0]\n  0: ZXZ [1,0,0]",
            "BLOG 1.0\n\nbranch b {\n  false {\n    0: XZZ [0,0,0]\n  }\n  true {\n    0: ZXZ [1,0,0]\n  }\n}",
        ] {
            let err = parse_blog_to_graph(input).expect_err("duplicate ID 0 should produce error");
            assert!(matches!(err, ParseError::DuplicateId { id: 0, .. }));
        }
    }

    #[test]
    fn test_diagnostic_rendering() {
        let input = "BLOG 1.0\n\n  0: INVALID [0,0,0]";
        let err = parse_blog_to_graph(input).expect_err("invalid block kind should produce error");
        let diagnostic = err.render_diagnostic("<test>", input);
        assert!(diagnostic.contains("<test>"));
        assert!(diagnostic.contains("Error"));

        let plain = err.render_diagnostic_plain("<test>", input);
        assert!(plain.contains("<test>"));
        assert!(plain.contains("Error"));
        assert!(
            !plain.contains('\x1b'),
            "plain render must not contain ANSI escapes"
        );
    }

    #[test]
    fn test_graph_error_carries_span() {
        // Two blocks with distinct IDs share position [0,0,0]: the duplicate-ID
        // check passes (the IDs differ), so lowering fails later in
        // `try_add_block`, surfacing a graph error attributed to the second
        // block statement.
        let input = "BLOG 1.0\n\n  0: XZZ [0,0,0]\n  1: ZXZ [0,0,0]";
        let err = parse_blog_to_graph(input).expect_err("colliding block positions should error");

        assert!(
            matches!(err, ParseError::Graph { .. }),
            "expected Graph error, got: {err:?}",
        );
        assert!(
            err.span().is_some(),
            "graph error should carry the offending statement span",
        );

        let plain = err.render_diagnostic_plain("<test>", input);
        assert!(
            plain.contains("<test>"),
            "diagnostic should carry a source label:\n{plain}",
        );
        assert!(
            plain.contains("1: ZXZ"),
            "diagnostic should show the offending source line:\n{plain}",
        );
    }

    #[test]
    fn test_resolve_condition_is_inverted_with_parenthesized_binary_expr() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  1: XZZ [0,0,1]\n  2: XZZ [1,0,1]\n  5: ZX [0,0,0]\n\n  a = measure 1\n  b = measure 2\n  resolve 5 if a ^ b",
        )
        .unwrap();

        let actions = graph.actions();
        let Action::Resolve { condition, .. } = actions.last().unwrap() else {
            panic!("expected resolve");
        };
        assert_eq!(condition.to_string(), "!(a ^ b)");
    }

    #[test]
    fn test_roundtrip_text_is_canonical_after_reversed_selective_parse() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  5: YX [2,0,0]\n  1: XZZ [2,0,1]\n\n  m = measure 1\n  resolve 5 if m",
        )
        .unwrap();
        let actions = graph.actions();
        let Action::Resolve { target, condition } = actions.last().unwrap() else {
            panic!("expected resolve");
        };
        assert_eq!(*target, glam::ivec3(2, 0, 0));
        assert_eq!(condition, &Expr::Not(Box::new(Expr::Var("m".to_string()))));
        assert_eq!(
            graph.to_blog_body_text(),
            "BLOG 1.0\n\n0: XY [2, 0, 0]\n1: XZZ [2, 0, 1]\n\nm = measure 1\nresolve 0 if !m\n"
        );
    }

    #[test]
    fn test_parse_actions_standalone_rejects_debug() {
        let input = "x = a ^ b\ndebug a b";
        let err = parse_actions(input, |_, _| None).expect_err("debug action should be rejected");

        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    #[test]
    fn parse_actions_resolves_multicell_edge_source_by_direction() {
        // The editor path (`parse_actions`) must resolve a measure-edge source
        // on a multi-cell block through `Block::endpoint_for_direction` — the
        // same resolver the file-parse path uses — rather than collapsing to the
        // block anchor. Regression for the direction-blind editor callback.
        let block = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
            .with_height("2d".parse().expect("valid height"))
            .expect("scaled cube");
        let expected_src = block.endpoint_for_direction(Direction::ZPLUS);
        // The +Z endpoint of a height=2d cube is its top cell, distinct from the anchor.
        assert_eq!(expected_src, IVec3::new(0, 0, 1));
        assert_ne!(expected_src, block.pos());

        let resolve = |id: u32, dir: Option<Direction>| -> Option<IVec3> {
            (id == 0).then(|| match dir {
                Some(dir) => block.endpoint_for_direction(dir),
                None => block.pos(),
            })
        };
        let actions = parse_actions("m = measure 0 -> +Z\nfeedback X 0 -> +Z if m", resolve)
            .expect("editor action parse");

        assert!(matches!(
            actions.as_slice(),
            [Action::Measure { target: MeasureTarget::Edge { src, dir }, name },
             Action::Feedback { targets, .. }]
                if *src == expected_src && *dir == Direction::ZPLUS && name == "m"
                && targets.len() == 1 && targets[0].target == expected_src
                && targets[0].direction == Some(Direction::ZPLUS)
        ));
    }

    #[test]
    fn test_unsupported_version_rejected() {
        let input = "BLOG 2.0\n\n  0: XZZ [0,0,0]";
        let err = parse_blog_to_graph(input).expect_err("version 2.0 should be rejected");
        assert!(
            matches!(
                err,
                ParseError::UnsupportedVersion {
                    major: 2,
                    minor: 0,
                    ..
                }
            ),
            "expected UnsupportedVersion, got: {err:?}",
        );
        assert!(err.span().is_some());
    }

    #[test]
    fn test_parse_blog_to_graph_accepts_reversed_selective_pairs_after_lowering_support() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: YX [0,0,0]")
            .expect("graph lowering accepts reversed pairs");
        assert_eq!(
            graph.get_block(glam::IVec3::ZERO).map(|block| block.kind),
            Some(BlockKind::Selective(SelectiveKind::XY))
        );
    }

    #[test]
    fn fixed_measurement_kinds_round_trip_through_blog() {
        let source = "BLOG 1.0\n\n0: X [0, 0, 0]\n1: Z [1, 0, 0]\n";
        let graph = parse_blog_to_graph(source).expect("fixed measurements parse");

        assert_eq!(
            graph.get_block(IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Measurement(Basis::X)),
        );
        assert_eq!(
            graph.get_block(IVec3::X).map(Block::kind),
            Some(BlockKind::Measurement(Basis::Z)),
        );
        assert_eq!(graph.to_blog_body_text(), source);
    }

    #[test]
    fn test_reversed_selective_resolve_cancels_explicit_negation() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: YX [0,0,0]\n  1: XZX [0,0,1]\n\n  m0 = measure 1\n  resolve 0 if !m0\n",
        )
        .expect("reversed selective resolve should parse");

        assert_eq!(
            graph.actions(),
            vec![
                crate::Action::Measure {
                    target: crate::MeasureTarget::Node(glam::ivec3(0, 0, 1)),
                    name: "m0".to_string(),
                },
                crate::Action::Resolve {
                    target: glam::IVec3::ZERO,
                    condition: crate::Expr::Var("m0".to_string()),
                },
            ]
        );
    }

    #[test]
    fn parse_blog_to_graph_accepts_explicit_walking_end_position() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: walk XZX [0, 0, 0] -> [-1, 1, 1]\n")
            .expect("walking block should parse");

        assert_eq!(
            graph.get_block(glam::IVec3::ZERO).map(Block::kind),
            Some(BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::XZX, glam::ivec2(-1, 1)).unwrap()
            ))
        );
    }

    #[test]
    fn parse_blog_to_graph_rejects_legacy_walking_movement_suffix() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: WXZX-X+Y [0,0,0]\n")
            .expect_err("legacy walking token should be rejected");

        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    #[test]
    fn parse_blog_to_graph_rejects_invalid_explicit_walking_end_position() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: walk XZX [0, 0, 0] -> [2, 0, 1]\n")
            .expect_err("invalid walking movement should be rejected");

        assert!(matches!(err, ParseError::InvalidWalkingBlock { .. }));
    }

    #[test]
    fn parse_blog_to_graph_accepts_explicit_patch_rotation_end_position() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: rotate Z [0, 0, 0] -> [0, -1, 1]\n")
            .expect("patch rotation block should parse");

        assert_eq!(
            graph.get_block(glam::IVec3::ZERO).map(Block::kind),
            Some(BlockKind::PatchRotation(
                PatchRotationKind::new(Basis::Z, glam::ivec2(0, -1)).unwrap()
            ))
        );
        assert_eq!(
            graph.to_blog_body_text(),
            "BLOG 1.0\n\n0: rotate Z [0, 0, 0] -> [0, -1, 1]\n"
        );
    }

    #[test]
    fn parse_blog_to_graph_rejects_invalid_patch_rotation_end_position() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: rotate X [0, 0, 0] -> [1, 1, 1]\n")
            .expect_err("diagonal patch rotation movement should be rejected");

        assert!(matches!(err, ParseError::InvalidPatchRotationBlock { .. }));
    }

    #[test]
    fn parse_blog_to_graph_resolves_tall_cube_directional_pipe_to_top_endpoint() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=2d\n  1: Port [0, 0, 2]\n  0 -> +Z\n",
        )
        .expect("tall cube top pipe should parse");

        assert!(graph.has_pipe_between(glam::IVec3::new(0, 0, 1), glam::IVec3::new(0, 0, 2)));
    }

    #[test]
    fn inherited_height_resolves_id_temporal_pipe_from_top() {
        let source = "BLOG 1.0\n\n0: ZXZ [0,0,0] height=2d\n1: ZXZ [1,0,0]\n2: Port [1,0,2]\n0 -> +X\n1 -> +Z\n";
        let graph = parse_blog_to_graph(source).expect("height propagates before ID resolution");

        graph.validate_structure().expect("lowered graph is valid");
        assert!(graph.has_pipe_between(glam::ivec3(1, 0, 1), glam::ivec3(1, 0, 2)));
        assert_eq!(
            parse_blog_to_graph(&source.replace("1 -> +Z", "[1,0,1] -> +Z"))
                .expect("explicit endpoint parses")
                .to_blog_text(),
            graph.to_blog_text()
        );
    }

    #[test]
    fn branch_heights_resolve_arm_id_pipes_before_shared_cut() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n100: ZXZ [0,0,0]\n101: ZXZ [3,0,0]\n[0,0,0] -> +Z\nbranch b {\n  false {\n    0: ZXZ [0,0,1] height=2d\n    1: ZXZ [1,0,1]\n    2: Z [1,0,3]\n    0 -> +X\n    1 -> +Z\n  }\n  true {\n    3: ZXZ [0,0,1] height=2d\n    4: ZXZ [1,0,1]\n    5: Z [1,0,3]\n    3 -> +X\n    4 -> +Z\n  }\n}\nm = measure 101\nresolve b if m\n",
        )
        .expect("each arm inherits height before its pipes resolve");

        graph.validate_structure().expect("lowered graph is valid");
        for arm in [
            graph.branch_definitions()[0].on_false(),
            graph.branch_definitions()[0].on_true(),
        ] {
            assert!(arm.pipes().any(|pipe| {
                pipe.src() == glam::ivec3(1, 0, 2) && pipe.dir() == Direction::ZPLUS
            }));
        }
    }

    #[test]
    fn branch_ids_are_visible_only_in_their_own_arm() {
        for true_pos in ["[0,0,1]", "[1,0,1]"] {
            let source = format!(
                "BLOG 1.0\n\n0: ZXZ [0,0,0]\n9: Z [3,0,0]\nbranch b {{\n  false {{\n    1: ZXZ [0,0,1]\n    0 -> 1\n  }}\n  true {{\n    2: ZXZ {true_pos}\n    0 -> 1\n  }}\n}}\nm = measure 9\nresolve b if m\n"
            );
            let result = parse_blog_to_graph(&source);
            assert!(
                matches!(result, Err(ParseError::UndefinedId { id: 1, .. })),
                "{true_pos}: {result:?}"
            );
        }
    }

    #[test]
    fn top_level_statements_cannot_name_branch_ids() {
        for statements in ["0 -> 1\nm = measure 9", "m = measure 1"] {
            let source = format!(
                "BLOG 1.0\n\n0: ZXZ [0,0,0]\n9: Z [3,0,0]\nbranch b {{\n  false {{\n    1: ZXZ [0,0,1]\n    0 -> 1\n  }}\n  true {{\n    2: ZXZ [0,0,1]\n    0 -> 2\n  }}\n}}\n{statements}\nresolve b if m\n"
            );

            let result = parse_blog_to_graph(&source);
            assert!(
                matches!(result, Err(ParseError::UndefinedId { id: 1, .. })),
                "{statements}: {result:?}"
            );
        }
    }

    #[test]
    fn interior_spatial_pipe_does_not_merge_height_components() {
        let error = parse_blog_to_graph(
            "BLOG 1.0\n\n0: ZXZ [0,0,0] height=3d\n1: ZXZ [1,0,0] height=2d\n[0,0,1] -> [1,0,1]\n",
        )
        .expect_err("a cube interior is not a pipe endpoint");

        assert!(matches!(
            error,
            ParseError::Graph { source, .. }
                if matches!(*source, crate::BlockGraphError::BlockNotFound(_))
        ));
    }

    #[test]
    fn spatial_ref_pipe_uses_cube_anchors() {
        let source = "BLOG 1.0\n\n0: ZXZ [0,0,0] height=2d\n1: ZXZ [1,0,0] height=2d\n0 -> 1\n";
        let graph = parse_blog_to_graph(source).expect("anchor pair is unambiguous");

        assert!(graph.has_pipe_between(IVec3::ZERO, IVec3::X));
        graph.validate_structure().expect("anchor pipe is valid");
    }

    #[test]
    fn distant_max_height_refs_remain_bounded() {
        let error = parse_blog_to_graph(
            "BLOG 1.0\n\n0: ZXZ [0,0,0] height=65535d\n1: ZXZ [2,0,0] height=65535d\n0 -> 1\n",
        )
        .expect_err("distant blocks are not adjacent");

        assert!(matches!(error, ParseError::InvalidDirection { .. }));
    }

    #[test]
    fn parse_actions_resolves_tall_cube_spatial_edge_measure_to_anchor_endpoint() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=2d\n  1: ZXZ [1, 0, 0] height=2d\n  0 -> +X\n",
        )
        .expect("tall cube should parse");
        let actions = parse_actions("m = measure 0 -> +X", |id, direction| {
            let pos = match id {
                0 => glam::IVec3::ZERO,
                1 => glam::ivec3(1, 0, 0),
                _ => return None,
            };
            let block = graph.get_block(pos)?;
            Some(direction.map_or(pos, |direction| block.endpoint_for_direction(direction)))
        })
        .expect("edge measure should parse");

        assert!(matches!(
            actions.as_slice(),
            [Action::Measure {
                target: crate::MeasureTarget::Edge { src, dir },
                name
            }] if *src == glam::IVec3::ZERO
                && *dir == crate::Direction::XPLUS
                && name == "m"
        ));
    }

    #[test]
    fn parse_blog_to_graph_resolves_patch_rotation_directional_pipes_to_endpoints() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: rotate X [0, 0, 0] -> [1, 0, 1]\n  1: Port [0, 0, -1]\n  2: Port [1, 0, 2]\n  0 -> -Z\n  0 -> +Z\n",
        )
        .expect("patch rotation endpoint pipes should parse");

        assert!(graph.has_pipe_between(glam::IVec3::ZERO, glam::IVec3::new(0, 0, -1)));
        assert!(graph.has_pipe_between(glam::IVec3::new(1, 0, 1), glam::IVec3::new(1, 0, 2)));
    }

    #[test]
    fn patch_rotation_temporal_pipes_use_endpoint_frame_for_face_bases() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: rotate X [0, 0, 0] -> [1, 0, 1]\n  1: Port [0, 0, -1]\n  2: Port [1, 0, 2]\n  [0, 0, 0] -> -Z\n  [1, 0, 1] -> +Z\n",
        )
        .expect("patch rotation endpoint pipes should parse");

        let cases = [
            (glam::IVec3::ZERO, Direction::ZMINUS, Basis::X, Basis::Z),
            (
                glam::IVec3::new(1, 0, 1),
                Direction::ZPLUS,
                Basis::Z,
                Basis::X,
            ),
        ];
        for (src, dir, x_basis, y_basis) in cases {
            let pipe = graph
                .get_pipe(src, src + dir.to_ivec3())
                .expect("patch rotation temporal pipe exists");
            let bases = graph.infer_pipe_basis(pipe);
            assert_eq!(bases[crate::UDirection::X.index()], Some(x_basis));
            assert_eq!(bases[crate::UDirection::Y.index()], Some(y_basis));
        }
    }

    #[test]
    fn parse_blog_to_graph_resolves_walking_directional_pipes_to_endpoints() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: walk XZZ [0, 0, 0] -> [1, 1, 1]\n  1: Port [0, 0, -1]\n  2: Port [1, 1, 2]\n  0 -> -Z\n  0 -> +Z\n",
        )
        .expect("walking endpoint pipes should parse");

        assert!(graph.has_pipe_between(glam::IVec3::ZERO, glam::IVec3::new(0, 0, -1)));
        assert!(graph.has_pipe_between(glam::IVec3::new(1, 1, 1), glam::IVec3::new(1, 1, 2)));
    }

    #[test]
    fn parse_blog_to_graph_omits_default_cube_height_on_roundtrip() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=2d/2\n")
            .expect("default-valued cube height should parse");

        assert_eq!(graph.to_blog_body_text(), "BLOG 1.0\n\n0: ZXZ [0, 0, 0]\n");
    }

    #[test]
    fn parse_blog_to_graph_roundtrips_port_color() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: Port [0, 0, 0] color=Eb4034\n")
            .expect("port color should parse");
        assert_eq!(
            graph.to_blog_text(),
            "BLOG 1.0\n\n0: Port [0, 0, 0] color=eb4034\n"
        );
    }

    #[test]
    fn parse_blog_to_graph_rejects_non_positive_cube_height() {
        for modifier in ["height=0d", "height=d/0"] {
            let err = parse_blog_to_graph(&format!("BLOG 1.0\n\n  0: ZXZ [0, 0, 0] {modifier}\n"))
                .expect_err("non-positive cube height should be rejected");

            assert!(matches!(err, ParseError::Syntax { .. }), "{modifier}");
        }
    }

    #[test]
    fn parse_blog_to_graph_rejects_cube_height_on_non_cube() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: Y [0, 0, 0] height=2d\n")
            .expect_err("cube height on a non-cube should be rejected");

        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    #[test]
    fn parse_blog_to_graph_rejects_color_on_non_port() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: ZXZ [0, 0, 0] color=eb4034\n")
            .expect_err("color on a non-Port should be rejected");

        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    #[test]
    fn parse_blog_to_graph_allows_incomplete_authoring_graphs() {
        let graph = parse_blog_to_graph("BLOG 1.0\n\n  0: Port [0, 0, 0]\n")
            .expect("BLOG authoring syntax permits incomplete graphs");

        assert!(graph.validate().is_err());
    }

    #[test]
    fn parse_blog_to_graph_rejects_coordinate_overflow() {
        let walking = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: walk XZX [-2147483648, 0, 0] -> [2147483647, 0, 1]\n",
        )
        .expect_err("walking delta must not overflow");
        assert!(matches!(walking, ParseError::CoordinateOverflow { .. }));

        let pipe = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [2147483647, 0, 0]\n  [2147483647, 0, 0] -> +X\n",
        )
        .expect_err("pipe destination must not overflow");
        assert!(matches!(pipe, ParseError::CoordinateOverflow { .. }));

        let tall = parse_blog_to_graph("BLOG 1.0\n\n  0: ZXZ [0, 0, 2147483647] height=2d\n")
            .expect_err("tall cube endpoint must not overflow");
        assert!(matches!(tall, ParseError::CoordinateOverflow { .. }));

        let distant_refs = parse_blog_to_graph(
            "BLOG 1.0\n\n0: Port [-2147483648, -2147483648, 0]\n1: Port [2147483647, 2147483647, 0]\n0 -> 1\n",
        )
        .expect_err("distant block references are not adjacent");
        assert!(matches!(distant_refs, ParseError::InvalidDirection { .. }));
    }

    #[test]
    fn parse_blog_to_graph_caps_cube_height_before_footprint_allocation() {
        let err = parse_blog_to_graph("BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=65536d\n")
            .expect_err("oversized parsed cube height must be rejected");
        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    /// One modifier is enough for the component it lands in: spatially merged
    /// cubes share their syndrome rounds, so the height is a property of the
    /// component, not of the cube the author happened to write it on.
    #[test]
    fn parse_blog_to_graph_propagates_one_height_across_a_spatial_component() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0]\n  1: ZXZ [1, 0, 0] height=d/2\n  2: ZXZ [2, 0, 0]\n  [0, 0, 0] -> +X\n  [1, 0, 0] -> +X\n",
        )
        .expect("one modifier carries the component");

        graph.validate_structure().expect("lowered graph is valid");
        let half: crate::CubeHeight = "d/2".parse().expect("valid height");
        for pos in [
            glam::IVec3::new(0, 0, 0),
            glam::IVec3::new(1, 0, 0),
            glam::IVec3::new(2, 0, 0),
        ] {
            assert_eq!(
                graph.get_block(pos).expect("cube exists").height(),
                half,
                "{pos}"
            );
        }
        // Every member is written out, so a re-parse of the emitted text sees
        // the same graph without leaning on propagation a second time.
        assert_eq!(graph.to_blog_text().matches("height=d/2").count(), 3);
    }

    /// Annotating the same height on several members of a component is how a
    /// hand-edited asset reads after propagation has been written back, so it
    /// must stay legal.
    #[test]
    fn parse_blog_to_graph_accepts_repeated_equal_heights_in_one_component() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=d/2\n  1: ZXZ [1, 0, 0] height=d/2\n  [0, 0, 0] -> +X\n",
        )
        .expect("equal modifiers agree");

        graph.validate_structure().expect("lowered graph is valid");
        assert_eq!(
            graph
                .get_block(glam::IVec3::new(1, 0, 0))
                .expect("cube exists")
                .height(),
            "d/2".parse().expect("valid height")
        );
    }

    #[test]
    fn parse_blog_to_graph_rejects_conflicting_heights_in_one_component() {
        let err = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=d/2\n  1: ZXZ [1, 0, 0] height=2d\n  [0, 0, 0] -> +X\n",
        )
        .expect_err("two different explicit heights in one component conflict");

        let ParseError::ConflictingCubeHeights {
            pos,
            height,
            other_pos,
            other_height,
            ..
        } = err
        else {
            panic!("expected a conflicting-height error, got {err}");
        };
        assert_eq!(other_pos, glam::IVec3::new(0, 0, 0));
        assert_eq!(other_height, "d/2".parse().expect("valid height"));
        assert_eq!(pos, glam::IVec3::new(1, 0, 0));
        assert_eq!(height, "2d".parse().expect("valid height"));
    }

    /// A `+Z` pipe stacks cubes in time rather than merging their patches, so
    /// the two ends keep independent heights. This is the whole reason the unit
    /// of propagation is a z-layer component and not a connected component.
    #[test]
    fn parse_blog_to_graph_does_not_propagate_height_through_temporal_pipes() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: ZXZ [0, 0, 0] height=d/2\n  1: ZXZ [0, 0, 1]\n  [0, 0, 0] -> +Z\n",
        )
        .expect("temporal neighbours keep their own heights");

        assert_eq!(
            graph
                .get_block(glam::IVec3::new(0, 0, 0))
                .expect("lower cube")
                .height(),
            "d/2".parse().expect("valid height")
        );
        assert_eq!(
            graph
                .get_block(glam::IVec3::new(0, 0, 1))
                .expect("upper cube")
                .height(),
            crate::CubeHeight::DEFAULT
        );
    }

    /// Round trip on a graph mixing every height shape the grammar accepts,
    /// including a negative offset and a component that keeps the default.
    #[test]
    fn parse_blog_to_graph_round_trips_mixed_cube_heights() {
        let source = "BLOG 1.0\n\n0: ZXZ [0, 0, 0] height=d/2\n1: ZXZ [2, 0, 0] height=3d/2\n2: ZXZ [4, 0, 0] height=2d-1\n3: ZXZ [6, 0, 0]\n";
        let graph = parse_blog_to_graph(source).expect("mixed heights parse");

        assert_eq!(graph.to_blog_body_text(), source);
    }

    #[test]
    fn gallery_height_assets_keep_authored_heights_through_roundtrip() {
        for (source, cube_count, uniform_height) in [
            (include_str!("../../assets/1d_yoked.blog"), 60, None),
            (
                include_str!("../../assets/ccz_4x3x7_tels.blog"),
                63,
                Some("2d/3"),
            ),
        ] {
            let program = crate::parse_inline_graph(source).expect("gallery asset parses");
            let emitted = program.to_blog_text();
            let reparsed = crate::parse_inline_graph(&emitted).expect("emitted text reparses");
            assert_eq!(reparsed.to_blog_text(), emitted);
            for program in [program, reparsed] {
                let graph = program
                    .materialize_root_graph()
                    .expect("gallery flat projection");
                let cubes: Vec<_> = graph
                    .blocks()
                    .filter(|block| block.kind().is_cube())
                    .collect();
                assert_eq!(cubes.len(), cube_count);
                for block in cubes {
                    let pos = block.pos();
                    let expected = uniform_height.unwrap_or(
                        if (pos.y == 1 && pos.z == 1) || (pos.y == 0 && pos.z == 7) {
                            "2d"
                        } else {
                            "d"
                        },
                    );
                    assert_eq!(block.height(), expected.parse().unwrap(), "{pos}");
                }
            }
        }
    }
}
