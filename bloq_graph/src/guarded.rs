//! Local topology alternatives and their exact Boolean applicability.
//!
//! A block signature reads its own block, incident pipes, and neighboring block
//! kinds. Enumerate only that neighborhood, keeping distant branch choices as
//! shared Boolean functions. Component-wide facts can refine these guards later.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, BooleanResourceError, DECISION_FALSE, DECISION_TRUE,
    DecisionId,
};
use glam::IVec3;

use crate::{
    Action, Block, BlockGraph, BlockGraphError, BlockKind, BranchRegion, Expr,
    ModuleCertificationLimits, Pipe, StabilizerError,
};

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuardedProjection {
    pub guard: DecisionId,
    /// Geometry context for this entry's center (the key in `GuardedTopology::sites`).
    /// The center and its immediate neighbors have their exact normalized kinds
    /// and complete selected pipe incidence. Outer halo blocks only resolve those
    /// pipe endpoints; their kinds/incidence are not a complete physical program.
    /// Unchanged contexts share one complete reference graph, which can carry
    /// source actions; deviations contain geometry only. Consumers read only
    /// this center's geometry. Never re-normalize a context or emit its halo.
    pub graph: Arc<BlockGraph>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum GuardedVariable {
    Branch { name: String, condition: DecisionId },
    Outcome(String),
}

/// The source-selector name used to pin a selective measurement at `position`.
///
/// Coordinates are those of the normalized graph supplied to compilation.
/// The punctuation keeps these generated names separate from authored branch names.
#[must_use]
pub fn selective_selector_name(position: IVec3) -> String {
    format!("selective [{},{},{}]", position.x, position.y, position.z)
}

/// Availability as known outcomes grow. Refresh after collection before querying.
pub(crate) struct AvailabilityQuery {
    // None is an unknown outcome. A known outcome needs no further input;
    // a structural variable instead requires its source condition.
    requirements: Vec<Option<DecisionId>>,
    outcomes: crate::FxHashMap<String, usize>,
    generation: u64,
    values: crate::FxHashMap<DecisionId, bool>,
    pending: Vec<DecisionId>,
}

impl AvailabilityQuery {
    pub(crate) fn learn<'a>(
        &mut self,
        names: impl IntoIterator<Item = &'a str>,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        debug_assert_eq!(self.generation, diagram.generation());
        let mut indices = Vec::new();
        for name in names {
            diagram.charge(1)?;
            if let Some(&index) = self.outcomes.get(name)
                && self.requirements[index].is_none()
            {
                indices.push(index);
            }
        }
        if !indices.is_empty() {
            diagram.charge(self.values.len())?;
            self.values.retain(|_, available| *available);
            for index in indices {
                self.requirements[index] = Some(DECISION_TRUE);
            }
        }
        Ok(())
    }

    fn cached(&self, root: DecisionId) -> Option<bool> {
        if root.0 < 2 {
            Some(true)
        } else {
            self.values.get(&root).copied()
        }
    }

    pub(crate) fn available(
        &mut self,
        root: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<bool, BooleanResourceError> {
        debug_assert_eq!(self.generation, diagram.generation());
        if let Some(value) = self.cached(root) {
            return Ok(value);
        }
        self.pending.clear();
        self.pending.push(root);
        while let Some(&root) = self.pending.last() {
            diagram.charge(1)?;
            if self.cached(root).is_some() {
                self.pending.pop();
                continue;
            }
            let available = if let Some(node) = diagram.node(root) {
                if let Some(condition) = self.requirements[node.variable] {
                    let inputs = [condition, node.low, node.high];
                    if inputs
                        .iter()
                        .any(|&input| self.cached(input) == Some(false))
                    {
                        false
                    } else if let Some(input) = inputs
                        .into_iter()
                        .find(|&input| self.cached(input).is_none())
                    {
                        self.pending.push(input);
                        continue;
                    } else {
                        true
                    }
                } else {
                    false
                }
            } else {
                true
            };
            self.values.insert(root, available);
            self.pending.pop();
        }
        Ok(self
            .cached(root)
            .expect("the requested function was visited"))
    }
}

/// Symbolic image of source Boolean functions. Disjoint source-variable
/// supports factor, even when their output coordinates are interleaved.
fn boolean_image(
    source: &BooleanDecisionDiagram,
    output: &mut BooleanDecisionDiagram,
    roots: Vec<(usize, DecisionId)>,
    limit: usize,
) -> Result<DecisionId, BlockGraphError> {
    // One traversal of the shared source DAG: a repeated node or source
    // variable joins the roots that use it. Constant roots stay isolated.
    output.charge(roots.len())?;
    let mut components = petgraph::unionfind::UnionFind::new(roots.len());
    let mut node_owner = HashMap::new();
    let mut variable_owner = HashMap::new();
    let mut pending = Vec::new();
    for (root_index, &(_, root)) in roots.iter().enumerate() {
        pending.push(root);
        while let Some(id) = pending.pop() {
            output.charge(1)?;
            let Some(node) = source.node(id) else {
                continue;
            };
            if let Some(&prior) = node_owner.get(&id) {
                output.charge(1)?;
                components.union(root_index, prior);
                continue;
            }
            node_owner.insert(id, root_index);
            if let Some(&prior) = variable_owner.get(&node.variable) {
                output.charge(1)?;
                components.union(root_index, prior);
            } else {
                variable_owner.insert(node.variable, root_index);
            }
            pending.extend([node.low, node.high]);
        }
    }

    output.charge(roots.len())?;
    let mut groups = Vec::<Vec<(usize, DecisionId)>>::new();
    let mut group_by_component = HashMap::new();
    for (root_index, root) in roots.into_iter().enumerate() {
        let component = components.find(root_index);
        let group = *group_by_component.entry(component).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(root);
    }

    let mut visited = 0;
    let mut image = DECISION_TRUE;
    for group in groups {
        output.charge(1)?;
        let factor = match group.as_slice() {
            [(index, root)] if *root == DECISION_TRUE => {
                output.make_node(*index, DECISION_FALSE, DECISION_TRUE)?
            }
            [(index, root)] if *root == DECISION_FALSE => {
                output.make_node(*index, DECISION_TRUE, DECISION_FALSE)?
            }
            [(_, _)] => {
                // A nonconstant reduced BDD reaches both truth values. One
                // output bit alone therefore has an unconstrained image.
                visit_image_state(&mut visited, limit)?;
                DECISION_TRUE
            }
            _ => boolean_image_component(source, output, group, &mut visited, limit)?,
        };
        image = output.apply(BooleanOp::And, image, factor)?;
    }
    Ok(image)
}

fn visit_image_state(visited: &mut usize, limit: usize) -> Result<(), BlockGraphError> {
    let next = visited.checked_add(1);
    *visited = next.unwrap_or(usize::MAX);
    if next.is_none() || *visited > limit {
        return Err(StabilizerError::ResourceLimited {
            phase: "structural selector image",
            observed: *visited,
            limit,
        }
        .into());
    }
    Ok(())
}

/// Existing residual-tuple search, scoped to one correlated support component.
/// `visited` is shared across all components, so partitioning cannot bypass
/// the structural witness budget.
fn boolean_image_component(
    source: &BooleanDecisionDiagram,
    output: &mut BooleanDecisionDiagram,
    roots: Vec<(usize, DecisionId)>,
    visited: &mut usize,
    limit: usize,
) -> Result<DecisionId, BlockGraphError> {
    enum Step {
        Visit(Vec<(usize, DecisionId)>),
        Join {
            remaining: Vec<(usize, DecisionId)>,
            fixed: DecisionId,
        },
    }
    let mut pending = vec![Step::Visit(roots)];
    let mut values = Vec::new();
    let mut memo = HashMap::new();
    while let Some(step) = pending.pop() {
        let roots = match step {
            Step::Visit(roots) => roots,
            Step::Join { remaining, fixed } => {
                let high = values.pop().expect("visited high image");
                let low = values.pop().expect("visited low image");
                let image = output.apply(BooleanOp::Or, low, high)?;
                memo.insert(remaining, image);
                values.push(output.apply(BooleanOp::And, fixed, image)?);
                continue;
            }
        };
        output.charge(roots.len())?;
        let mut fixed = DECISION_TRUE;
        let mut remaining = Vec::new();
        for (index, root) in roots {
            if root.0 < 2 {
                let value = if root == DECISION_TRUE {
                    output.make_node(index, DECISION_FALSE, DECISION_TRUE)?
                } else {
                    output.make_node(index, DECISION_TRUE, DECISION_FALSE)?
                };
                fixed = output.apply(BooleanOp::And, fixed, value)?;
            } else {
                remaining.push((index, root));
            }
        }
        if remaining.is_empty() {
            values.push(fixed);
            continue;
        }
        if let Some(&image) = memo.get(&remaining) {
            values.push(output.apply(BooleanOp::And, fixed, image)?);
            continue;
        }
        visit_image_state(visited, limit)?;
        let variable = remaining
            .iter()
            .map(|(_, root)| {
                source
                    .node(*root)
                    .expect("nonterminal root belongs to the source diagram")
                    .variable
            })
            .min()
            .expect("remaining source roots are nonempty");
        output.charge(remaining.len().saturating_mul(2))?;
        let branch = |high| {
            remaining
                .iter()
                .map(|&(index, root)| {
                    let node = source
                        .node(root)
                        .expect("nonterminal root belongs to the source diagram");
                    (
                        index,
                        if node.variable != variable {
                            root
                        } else if high {
                            node.high
                        } else {
                            node.low
                        },
                    )
                })
                .collect()
        };
        let low = branch(false);
        let high = branch(true);
        pending.extend([
            Step::Join { remaining, fixed },
            Step::Visit(high),
            Step::Visit(low),
        ]);
    }
    Ok(values.pop().expect("the source image produces one value"))
}

fn block_footprint_size<'a>(blocks: impl Iterator<Item = &'a crate::Block>) -> [usize; 2] {
    blocks.fold([0usize; 2], |[count, cells], block| {
        let footprint = if block.kind().is_cube() {
            block.height_cells() as usize
        } else {
            block.kind().reserved_offsets().len()
        };
        [count.saturating_add(1), cells.saturating_add(footprint)]
    })
}

type LocalChoices = HashMap<DecisionId, bool>;

struct LocalGeometry {
    blocks: Vec<Block>,
    pipes: Vec<Pipe>,
    primary_incidence: Vec<(IVec3, usize)>,
}

impl LocalGeometry {
    fn matches_reference(&self, reference: &BlockGraph) -> bool {
        self.primary_incidence.iter().all(|&(position, count)| {
            let block = self
                .blocks
                .binary_search_by_key(&position.to_array(), |block| block.pos().to_array())
                .map(|index| &self.blocks[index])
                .expect("local primary block is retained");
            reference.get_block(position) == Some(block)
                && reference.pipes_at(position).count() == count
        }) && self
            .pipes
            .iter()
            .all(|pipe| reference.get_pipe(pipe.src(), pipe.dst()) == Some(pipe))
    }
}

#[derive(Debug)]
struct Authored<T> {
    choice: Option<(DecisionId, bool)>,
    value: T,
}

impl<T> Authored<T> {
    fn selected(&self, choices: &LocalChoices) -> bool {
        self.choice
            .is_none_or(|(root, value)| choices.get(&root).copied().unwrap_or(false) == value)
    }
}

/// One source-wide index. Local materialization only visits incident records;
/// distant branch choices are irrelevant to the center and its neighbors.
#[derive(Debug)]
struct AuthoredGeometry {
    blocks: Vec<Authored<Block>>,
    pipes: Vec<Authored<Pipe>>,
    blocks_at: BTreeMap<[i32; 3], Vec<usize>>,
    endpoint_owners: crate::FxHashMap<IVec3, Vec<usize>>,
    incidence: crate::FxHashMap<IVec3, Vec<usize>>,
    owners: crate::FxHashMap<IVec3, BTreeSet<DecisionId>>,
    neighbors: crate::FxHashMap<IVec3, BTreeSet<[i32; 3]>>,
}

impl AuthoredGeometry {
    fn new(
        source: &BlockGraph,
        regions: &[BranchRegion],
        shared_pipes: &[Pipe],
        branch_roots: &HashMap<IVec3, DecisionId>,
    ) -> Result<Self, BlockGraphError> {
        let shown = regions
            .iter()
            .flat_map(|region| region.shown_arm().blocks().map(Block::pos))
            .collect::<std::collections::HashSet<_>>();
        let mut blocks = source
            .blocks()
            .filter(|block| !shown.contains(&block.pos()))
            .cloned()
            .map(|value| Authored {
                choice: None,
                value,
            })
            .collect::<Vec<_>>();
        let mut pipes = Vec::new();
        for region in regions {
            for value in [false, true] {
                let choice = Some((branch_roots[&region.target], value));
                blocks.extend(
                    region
                        .arm(value)
                        .blocks()
                        .cloned()
                        .map(|value| Authored { choice, value }),
                );
                pipes.extend(
                    region
                        .arm(value)
                        .pipes()
                        .cloned()
                        .map(|value| Authored { choice, value }),
                );
            }
        }
        // A selected arm pipe takes precedence over an identical common seam,
        // matching materialize_projection's add-arms-then-shared-pipes order.
        pipes.extend(shared_pipes.iter().cloned().map(|value| Authored {
            choice: None,
            value,
        }));
        let mut index = Self {
            blocks,
            pipes,
            blocks_at: BTreeMap::new(),
            endpoint_owners: crate::FxHashMap::default(),
            incidence: crate::FxHashMap::default(),
            owners: crate::FxHashMap::default(),
            neighbors: crate::FxHashMap::default(),
        };
        for (id, record) in index.blocks.iter().enumerate() {
            let position = record.value.pos();
            index
                .blocks_at
                .entry(position.to_array())
                .or_default()
                .push(id);
            if let Some((root, _)) = record.choice {
                index.owners.entry(position).or_default().insert(root);
            }
            for offset in record.value.connectable_offsets() {
                let endpoint = crate::checked_add_position(position, offset)?;
                index.endpoint_owners.entry(endpoint).or_default().push(id);
            }
        }
        for (id, record) in index.pipes.iter().enumerate() {
            let (left, right) = record.value.try_endpoints()?;
            let left = index
                .endpoint_owners
                .get(&left)
                .into_iter()
                .flatten()
                .map(|&block| index.blocks[block].value.pos())
                .collect::<std::collections::HashSet<_>>();
            let right = index
                .endpoint_owners
                .get(&right)
                .into_iter()
                .flatten()
                .map(|&block| index.blocks[block].value.pos())
                .collect::<std::collections::HashSet<_>>();
            for &position in left.iter().chain(&right) {
                index.incidence.entry(position).or_default().push(id);
                if let Some((root, _)) = record.choice {
                    index.owners.entry(position).or_default().insert(root);
                }
            }
            for &left in &left {
                for &right in &right {
                    index
                        .neighbors
                        .entry(left)
                        .or_default()
                        .insert(right.to_array());
                    index
                        .neighbors
                        .entry(right)
                        .or_default()
                        .insert(left.to_array());
                }
            }
        }
        for records in index.incidence.values_mut() {
            records.sort_unstable();
            records.dedup();
        }
        Ok(index)
    }

    fn block_at(
        &self,
        position: IVec3,
        choices: &LocalChoices,
    ) -> Result<Option<&Block>, BlockGraphError> {
        let mut selected = self
            .blocks_at
            .get(&position.to_array())
            .into_iter()
            .flatten()
            .map(|&id| &self.blocks[id])
            .filter(|record| record.selected(choices));
        let block = selected.next().map(|record| &record.value);
        if selected.next().is_some() {
            return Err(BlockGraphError::BlockExists(position));
        }
        Ok(block)
    }

    fn endpoint_owner(
        &self,
        endpoint: IVec3,
        choices: &LocalChoices,
    ) -> Result<&Block, BlockGraphError> {
        self.endpoint_owners
            .get(&endpoint)
            .into_iter()
            .flatten()
            .map(|&id| &self.blocks[id])
            .find(|record| record.selected(choices))
            .map(|record| &record.value)
            .ok_or(BlockGraphError::BlockNotFound(endpoint))
    }

    fn pipes_at(
        &self,
        position: IVec3,
        choices: &LocalChoices,
    ) -> Result<Vec<&Pipe>, BlockGraphError> {
        let mut selected = BTreeMap::<([i32; 3], [i32; 3]), usize>::new();
        for &id in self.incidence.get(&position).into_iter().flatten() {
            let record = &self.pipes[id];
            if !record.selected(choices) {
                continue;
            }
            let (left, right) = record.value.endpoints();
            if self.endpoint_owner(left, choices)?.pos() != position
                && self.endpoint_owner(right, choices)?.pos() != position
            {
                continue;
            }
            let (left, right) = crate::branch::pipe_key(&record.value);
            let key = (left.to_array(), right.to_array());
            if let Some(&previous) = selected.get(&key) {
                if record.choice.is_some() && self.pipes[previous].choice.is_some() {
                    return Err(BlockGraphError::PipeExists(left, right));
                }
            } else {
                selected.insert(key, id);
            }
        }
        Ok(selected
            .into_values()
            .map(|id| &self.pipes[id].value)
            .collect())
    }

    fn roots(&self, position: IVec3) -> Vec<DecisionId> {
        std::iter::once(position.to_array())
            .chain(self.neighbors.get(&position).into_iter().flatten().copied())
            .flat_map(|position| {
                self.owners
                    .get(&IVec3::from_array(position))
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn local(
        &self,
        center: IVec3,
        choices: &LocalChoices,
    ) -> Result<Option<LocalGeometry>, BlockGraphError> {
        if self.block_at(center, choices)?.is_none() {
            return Ok(None);
        }
        let mut primary = BTreeSet::from([center.to_array()]);
        for pipe in self.pipes_at(center, choices)? {
            for endpoint in [pipe.src(), pipe.dst()] {
                primary.insert(self.endpoint_owner(endpoint, choices)?.pos().to_array());
            }
        }
        let mut blocks = BTreeMap::new();
        let mut pipes = BTreeMap::new();
        let mut primary_incidence = Vec::with_capacity(primary.len());
        for &position in &primary {
            let position = IVec3::from_array(position);
            let mut block = self
                .block_at(position, choices)?
                .expect("selected neighbor exists")
                .clone();
            let mut counts = [0usize; 3];
            for pipe in self.pipes_at(position, choices)? {
                counts[pipe.dir().as_udirection().index()] += 1;
                for endpoint in [pipe.src(), pipe.dst()] {
                    // Unmentioned controls only choose lookup-only halo blocks;
                    // primary kinds/pipes depend entirely on the local case roots.
                    let neighbor = self.endpoint_owner(endpoint, choices)?;
                    blocks
                        .entry(neighbor.pos().to_array())
                        .or_insert_with(|| neighbor.clone());
                }
                let (left, right) = crate::branch::pipe_key(pipe);
                pipes.insert((left.to_array(), right.to_array()), pipe.clone());
            }
            if let BlockKind::Cube(kind) = block.kind() {
                block.kind = BlockKind::Cube(crate::graph::shadowed_cube_kind(kind, counts));
            }
            primary_incidence.push((position, counts.iter().sum()));
            blocks.insert(position.to_array(), block);
        }
        Ok(Some(LocalGeometry {
            blocks: blocks.into_values().collect(),
            pipes: pipes.into_values().collect(),
            primary_incidence,
        }))
    }
}

/// A shared decision diagram over source measurement variables, with local
/// structural alternatives. It never lists the full joint branch domain.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuardedTopology {
    pub source: BlockGraph,
    pub diagram: BooleanDecisionDiagram,
    pub variables: Vec<GuardedVariable>,
    /// Reachable branch-vector image of source outcome assignments. This does
    /// not tie each Branch variable to its condition for a particular Outcome
    /// assignment; [`Self::evaluate_source`] supplies those branch values.
    pub domain: DecisionId,
    pub branches: Vec<(String, IVec3, DecisionId)>,
    pub resolves: Vec<(IVec3, DecisionId)>,
    pub sites: BTreeMap<[i32; 3], Vec<GuardedProjection>>,
    projections: HashMap<Vec<bool>, Arc<BlockGraph>>,
    regions: Vec<BranchRegion>,
    shared_pipes: Vec<Pipe>,
    conditions: Vec<(Arc<Expr>, DecisionId)>,
    condition_slots: HashMap<Arc<Expr>, usize>,
    named_conditions: HashMap<String, usize>,
    limits: ModuleCertificationLimits,
    common_projection_size: [usize; 2],
    retained_projection_size: [usize; 2],
    deferred_sites: bool,
}

impl GuardedTopology {
    pub(crate) fn decision_root_count(&self) -> usize {
        1 + self.conditions.len()
            + self.branches.len()
            + self.resolves.len()
            + self.sites.values().map(Vec::len).sum::<usize>()
            + self
                .variables
                .iter()
                .filter(|variable| matches!(variable, GuardedVariable::Branch { .. }))
                .count()
    }

    /// Solve one complete variable assignment that selects every branch arm
    /// except `absent`. Returns `None` when that arm selection is unreachable.
    pub(crate) fn branch_selection_values(
        &self,
        absent: &rustc_hash::FxHashSet<IVec3>,
    ) -> Result<Option<Vec<bool>>, BooleanResourceError> {
        if self.branches.is_empty() {
            return Ok(Some(vec![false; self.variables.len()]));
        }
        let mut diagram = self.diagram.clone();
        diagram.charge(self.variables.len().saturating_add(self.branches.len()))?;
        let branch_values = self
            .branches
            .iter()
            .map(|(name, position, _)| (name.as_str(), !absent.contains(position)))
            .collect::<HashMap<_, _>>();
        // The branch-vector domain does not link each branch to its source outcomes.
        let conditions = self
            .variables
            .iter()
            .filter_map(|variable| match variable {
                GuardedVariable::Branch { name, condition } => {
                    Some(if branch_values[name.as_str()] {
                        Ok(*condition)
                    } else {
                        diagram.negate(*condition)
                    })
                }
                GuardedVariable::Outcome(_) => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let Some(outcomes) = diagram.conjunction_witness(&conditions)? else {
            return Ok(None);
        };
        let outcome = |index| outcomes.get(&index).copied().unwrap_or(false);
        Ok(Some(
            self.variables
                .iter()
                .enumerate()
                .map(|(index, variable)| match variable {
                    GuardedVariable::Branch { condition, .. } => {
                        self.diagram.evaluate(*condition, outcome)
                    }
                    GuardedVariable::Outcome(_) => outcome(index),
                })
                .collect(),
        ))
    }

    /// Collect only while no external stage retains unlisted decision ids.
    pub(crate) fn collect_garbage<'a>(
        &'a mut self,
        additional: impl IntoIterator<Item = &'a mut DecisionId>,
    ) -> usize {
        assert!(
            !self.deferred_sites,
            "deferred authored-geometry roots must be consumed before Boolean collection"
        );
        let roots = std::iter::once(&mut self.domain)
            .chain(self.conditions.iter_mut().map(|(_, root)| root))
            .chain(self.branches.iter_mut().map(|(_, _, root)| root))
            .chain(self.resolves.iter_mut().map(|(_, root)| root))
            .chain(
                self.sites
                    .values_mut()
                    .flatten()
                    .map(|site| &mut site.guard),
            )
            .chain(
                self.variables
                    .iter_mut()
                    .filter_map(|variable| match variable {
                        GuardedVariable::Branch { condition, .. } => Some(condition),
                        GuardedVariable::Outcome(_) => None,
                    }),
            );
        self.diagram.collect_garbage(roots.chain(additional))
    }

    /// Builds the factored guarded topology for a source graph.
    ///
    /// # Errors
    ///
    /// Returns an error if source validation, projection, or resource accounting fails.
    ///
    /// # Panics
    ///
    /// Panics if validated branch actions do not name their stored regions.
    pub fn new(
        source: &BlockGraph,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, BlockGraphError> {
        let (mut topology, sites) = Self::new_with_geometry(source, limits)?;
        topology.populate_sites(sites)?;
        Ok(topology)
    }

    fn new_with_geometry(
        source: &BlockGraph,
        limits: ModuleCertificationLimits,
    ) -> Result<(Self, AuthoredGeometry), BlockGraphError> {
        source.require_flat_hierarchy("guarded topology")?;
        source.validate_resource_limits(limits)?;
        crate::validate::validate_source(source)?;
        let actions = source.actions();
        // validate_source already checked every arm and its action binding.
        // Recover that resolve order without rebuilding candidate arm graphs.
        let by_target = source
            .branch_definitions()
            .iter()
            .map(|region| (region.target, region))
            .collect::<HashMap<_, _>>();
        let regions = actions
            .iter()
            .filter_map(|action| match action {
                Action::Branch { target, .. } => Some((*by_target[target]).clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let shown = regions
            .iter()
            .flat_map(|region| {
                region
                    .shown_arm()
                    .blocks()
                    .map(|block| block.pos().to_array())
            })
            .collect::<BTreeSet<_>>();
        let common_projection_size = block_footprint_size(
            source
                .blocks()
                .filter(|block| !shown.contains(&block.pos().to_array())),
        );
        let expressions = actions
            .iter()
            .filter_map(|action| match action {
                Action::Branch { condition, .. }
                | Action::Resolve { condition, .. }
                | Action::Feedback {
                    condition: Some(condition),
                    ..
                }
                | Action::DiscardIf(condition) => Some(condition.clone()),
                Action::Let { expr, .. } => Some(expr.clone()),
                Action::Measure { name, .. } => Some(Expr::Var(name.clone())),
                _ => None,
            })
            .map(Arc::new)
            .collect::<Vec<_>>();
        // Slots stay fixed when roots are remapped. Keep the first exact
        // occurrence, matching the ordered condition search.
        let mut condition_slots = HashMap::new();
        for (slot, expression) in expressions.iter().enumerate() {
            condition_slots
                .entry(Arc::clone(expression))
                .or_insert(slot);
        }
        let aliases = actions
            .iter()
            .filter_map(|action| match action {
                Action::Let { name, expr } => Some((name.as_str(), expr)),
                _ => None,
            })
            .collect();
        let decisions = crate::action_dag::ResolveDecisionDiagram::build(
            &expressions.iter().map(Arc::as_ref).collect::<Vec<_>>(),
            &aliases,
            limits.boolean_limits(),
        )?
        .expect("validated source expressions");
        let mut diagram = BooleanDecisionDiagram::with_limits(limits.boolean_limits());
        diagram.charge(decisions.diagram.steps())?;
        let mut outcomes = vec![String::new(); decisions.variable_indices.len()];
        for (name, index) in decisions.variable_indices {
            outcomes[index] = name;
        }
        let named_source_conditions = expressions
            .iter()
            .zip(&decisions.roots)
            .filter_map(|(expression, root)| match expression.as_ref() {
                Expr::Var(name) => Some((name.as_str(), *root)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let source_condition = |expr: &Expr| {
            if let Expr::Var(name) = expr {
                return named_source_conditions[name.as_str()];
            }
            decisions.roots[*condition_slots
                .get(expr)
                .expect("source conditions were indexed before lowering")]
        };
        let branch_sources = actions
            .iter()
            .filter_map(|action| match action {
                Action::Branch { target, condition } => Some((
                    by_target[target].name.clone(),
                    *target,
                    source_condition(condition),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        // Grouping every branch before every outcome makes a sum of local
        // branch/outcome products exponential in an ordered decision diagram.
        // Preserve source outcome order, placing each branch next to its last
        // condition dependency. Source-node remapping then stays monotone.
        diagram.charge(
            decisions
                .diagram
                .nodes()
                .len()
                .saturating_add(outcomes.len())
                .saturating_add(branch_sources.len()),
        )?;
        let mut last_dependency = vec![None, None];
        for node in decisions.diagram.nodes() {
            let last = std::iter::once(node.variable)
                .chain(last_dependency[node.low.0])
                .chain(last_dependency[node.high.0])
                .max();
            last_dependency.push(last);
        }
        let mut after = vec![Vec::new(); outcomes.len() + 1];
        for (branch, (_, _, condition)) in branch_sources.iter().enumerate() {
            after[last_dependency[condition.0].map_or(0, |index| index + 1)].push(branch);
        }
        let mut outcome_indices = vec![0; outcomes.len()];
        let mut branch_indices = vec![0; branch_sources.len()];
        let mut variables = Vec::with_capacity(outcomes.len() + branch_sources.len());
        for (slot, outcome) in outcomes.into_iter().map(Some).chain([None]).enumerate() {
            for &branch in &after[slot] {
                branch_indices[branch] = variables.len();
                let (name, _, condition) = &branch_sources[branch];
                variables.push(GuardedVariable::Branch {
                    name: name.clone(),
                    condition: *condition,
                });
            }
            if let Some(name) = outcome {
                outcome_indices[slot] = variables.len();
                variables.push(GuardedVariable::Outcome(name));
            }
        }
        let mut remap = vec![DECISION_FALSE, DECISION_TRUE];
        for node in decisions.diagram.nodes() {
            remap.push(diagram.make_node(
                outcome_indices[node.variable],
                remap[node.low.0],
                remap[node.high.0],
            )?);
        }
        for variable in &mut variables {
            if let GuardedVariable::Branch { condition, .. } = variable {
                *condition = remap[condition.0];
            }
        }
        let conditions = expressions
            .iter()
            .cloned()
            .zip(decisions.roots.iter().map(|root| remap[root.0]))
            .collect::<Vec<_>>();
        let named_conditions = conditions
            .iter()
            .enumerate()
            .filter_map(|(index, (expression, _))| match expression.as_ref() {
                Expr::Var(name) => Some((name.clone(), index)),
                _ => None,
            })
            .collect();
        let mut branch_functions = Vec::new();
        let mut branches = Vec::new();
        for ((name, target, condition), index) in branch_sources.into_iter().zip(branch_indices) {
            let variable = diagram.make_node(index, DECISION_FALSE, DECISION_TRUE)?;
            branch_functions.push((index, condition));
            branches.push((name, target, variable));
        }
        let mut resolves = Vec::new();
        for action in &actions {
            if let Action::Resolve { target, condition } = action {
                resolves.push((*target, remap[source_condition(condition).0]));
            }
        }
        let domain = boolean_image(
            &decisions.diagram,
            &mut diagram,
            branch_functions,
            limits.max_witness_nodes,
        )?;
        drop(decisions.diagram);
        let branch_root = branches
            .iter()
            .map(|(_, target, root)| (*target, *root))
            .collect::<HashMap<_, _>>();
        let shared_pipes = source.branch_shared_pipes();
        let geometry = AuthoredGeometry::new(source, &regions, &shared_pipes, &branch_root)?;
        let mut topology = Self {
            source: source.clone(),
            diagram,
            variables,
            domain,
            branches,
            resolves,
            sites: BTreeMap::new(),
            projections: HashMap::new(),
            regions,
            shared_pipes,
            conditions,
            condition_slots,
            named_conditions,
            limits,
            common_projection_size,
            retained_projection_size: [0; 2],
            deferred_sites: true,
        };
        // The reference retains authored action order and output roles.
        topology.project(DECISION_TRUE)?;
        Ok((topology, geometry))
    }

    /// Materialize the guarded site views from authored local incidence.
    ///
    /// # Errors
    /// Returns an error if local geometry or its configured budget fails.
    fn populate_sites(&mut self, geometry: AuthoredGeometry) -> Result<(), BlockGraphError> {
        // Global planning needs this one reference anyway. Unchanged local
        // contexts borrow it instead of retaining another copy of every halo.
        let reference = self.project(DECISION_TRUE)?;
        for &position in geometry.blocks_at.keys() {
            let pos = IVec3::from_array(position);
            let variants = self.site_variants(&geometry, pos, &reference)?;
            self.sites.insert(position, variants);
        }
        self.deferred_sites = false;
        Ok(())
    }

    fn site_variants(
        &mut self,
        geometry: &AuthoredGeometry,
        position: IVec3,
        reference: &Arc<BlockGraph>,
    ) -> Result<Vec<GuardedProjection>, BlockGraphError> {
        let roots = geometry.roots(position);
        let cases = self.cases(&roots, DECISION_TRUE)?;
        let mut variants = Vec::new();
        for (values, guard) in cases {
            let choices = roots.iter().copied().zip(values).collect::<LocalChoices>();
            let Some(local) = geometry.local(position, &choices)? else {
                continue;
            };
            let graph = if local.matches_reference(reference) {
                Arc::clone(reference)
            } else {
                self.reserve_geometry(block_footprint_size(local.blocks.iter()))?;
                let mut graph = BlockGraph::new();
                for block in local.blocks {
                    graph.try_add_block(block)?;
                }
                for pipe in local.pipes {
                    graph.try_add_pipe(pipe)?;
                }
                Arc::new(graph)
            };
            variants.push(GuardedProjection { guard, graph });
        }
        Ok(variants)
    }

    /// Returns the decision root for a source expression.
    ///
    /// # Panics
    ///
    /// Panics if `expr` was not part of the source used to build this topology.
    pub fn condition(&self, expr: &Expr) -> DecisionId {
        if let Expr::Var(name) = expr {
            return self.conditions[self.named_conditions[name]].1;
        }
        self.conditions[*self.condition_slots.get(expr).expect("source expression")].1
    }

    /// Distinct values of these local functions, retaining each exact guard.
    ///
    /// # Errors
    ///
    /// Returns an error if Boolean work or the guarded domain exceeds its limit.
    pub fn cases(
        &mut self,
        roots: &[DecisionId],
        enabled: DecisionId,
    ) -> Result<Vec<(Vec<bool>, DecisionId)>, BlockGraphError> {
        let limit = self.limits.max_guarded_domain_size;
        self.diagram.charge(roots.len().saturating_add(1))?;
        let enabled = self.diagram.apply(BooleanOp::And, enabled, self.domain)?;
        if enabled == DECISION_FALSE {
            return Ok(Vec::new());
        }
        // Refine the exact guards directly. Enumerating a joint decision
        // image first revisits residual tuples, then rebuilds these same guards.
        let mut cases = vec![(Vec::with_capacity(roots.len()), enabled)];
        for &root in roots {
            let mut next = Vec::new();
            for (mut values, guard) in cases {
                self.diagram.charge(1)?;
                let high = self.diagram.apply(BooleanOp::And, guard, root)?;
                let low = self.diagram.apply(BooleanOp::Xor, guard, high)?;
                let count =
                    usize::from(low != DECISION_FALSE) + usize::from(high != DECISION_FALSE);
                if next.len().saturating_add(count) > limit {
                    return Err(StabilizerError::ResourceLimited {
                        phase: "local structural patterns",
                        observed: next.len().saturating_add(count),
                        limit,
                    }
                    .into());
                }
                if high == DECISION_FALSE {
                    values.push(false);
                    next.push((values, low));
                } else if low == DECISION_FALSE {
                    values.push(true);
                    next.push((values, high));
                } else {
                    self.diagram.charge(values.len().saturating_add(1))?;
                    let mut lower = values.clone();
                    lower.push(false);
                    next.push((lower, low));
                    values.push(true);
                    next.push((values, high));
                }
            }
            cases = next;
        }
        for (_, guard) in &mut cases {
            self.diagram.charge(1)?;
            *guard = self.diagram.constrain(*guard, self.domain)?;
        }
        if cases.len() > limit {
            return Err(StabilizerError::ResourceLimited {
                phase: "local structural patterns",
                observed: cases.len(),
                limit,
            }
            .into());
        }
        Ok(cases)
    }

    /// Returns one source-outcome witness satisfying `guard`.
    ///
    /// # Errors
    ///
    /// Returns an error if Boolean witness search exceeds its budget.
    pub fn witness(
        &mut self,
        guard: DecisionId,
    ) -> Result<Option<BTreeMap<usize, bool>>, BooleanResourceError> {
        self.diagram.conjunction_witness(&[self.domain, guard])
    }

    pub fn evaluate_source(&self, root: DecisionId, outcome: impl Fn(&str) -> bool) -> bool {
        self.diagram
            .evaluate(root, |index| match &self.variables[index] {
                GuardedVariable::Outcome(name) => outcome(name),
                GuardedVariable::Branch { condition, .. } => {
                    self.diagram.evaluate(*condition, |index| {
                        let GuardedVariable::Outcome(name) = &self.variables[index] else {
                            unreachable!("branch conditions contain only outcome variables")
                        };
                        outcome(name)
                    })
                }
            })
    }

    pub(crate) fn availability(
        &mut self,
        known: &BTreeSet<String>,
    ) -> Result<AvailabilityQuery, BooleanResourceError> {
        self.diagram.charge(self.variables.len())?;
        Ok(AvailabilityQuery {
            requirements: self
                .variables
                .iter()
                .map(|variable| match variable {
                    GuardedVariable::Outcome(name) => known.contains(name).then_some(DECISION_TRUE),
                    GuardedVariable::Branch { condition, .. } => Some(*condition),
                })
                .collect(),
            outcomes: self
                .variables
                .iter()
                .enumerate()
                .filter_map(|(index, variable)| match variable {
                    GuardedVariable::Outcome(name) => Some((name.clone(), index)),
                    GuardedVariable::Branch { .. } => None,
                })
                .collect(),
            generation: self.diagram.generation(),
            values: Default::default(),
            pending: Vec::new(),
        })
    }

    pub(crate) fn refresh_availability(
        &mut self,
        query: &mut AvailabilityQuery,
        known: &BTreeSet<String>,
    ) -> Result<(), BooleanResourceError> {
        if query.generation != self.diagram.generation() {
            *query = self.availability(known)?;
        }
        Ok(())
    }

    /// Returns one complete branch assignment satisfying `guard`.
    ///
    /// # Errors
    ///
    /// Returns an error if witness search exceeds its Boolean budget.
    pub fn assignment(
        &mut self,
        guard: DecisionId,
    ) -> Result<Option<Vec<(IVec3, bool)>>, BooleanResourceError> {
        let Some(variables) = self.witness(guard)? else {
            return Ok(None);
        };
        Ok(Some(
            self.branches
                .iter()
                .map(|(_, target, root)| {
                    (
                        *target,
                        self.diagram.evaluate(*root, |index| {
                            variables.get(&index).copied().unwrap_or(false)
                        }),
                    )
                })
                .collect(),
        ))
    }

    fn reserve_geometry(&mut self, size: [usize; 2]) -> Result<(), BlockGraphError> {
        let mut retained = self.retained_projection_size;
        for (index, (phase, limit)) in [
            (
                "retained projection blocks",
                self.limits.max_expanded_blocks,
            ),
            (
                "retained projection footprint cells",
                self.limits.max_occupied_cells,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let next = retained[index].checked_add(size[index]);
            if next.is_none_or(|next| next > limit) {
                return Err(StabilizerError::ResourceLimited {
                    phase,
                    observed: next.unwrap_or(usize::MAX),
                    limit,
                }
                .into());
            }
            retained[index] = next.expect("checked retained projection size");
        }
        self.retained_projection_size = retained;
        Ok(())
    }

    /// Materializes and caches the branch projection selected by `guard`.
    ///
    /// # Errors
    ///
    /// Returns an error if witness search, resource accounting, or projection fails.
    ///
    /// # Panics
    ///
    /// Panics if `guard` does not select any reachable local case.
    pub fn project(&mut self, guard: DecisionId) -> Result<Arc<BlockGraph>, BlockGraphError> {
        let assignment = self.assignment(guard)?.expect("a local case is reachable");
        let key = assignment
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        if let Some(graph) = self.projections.get(&key) {
            return Ok(Arc::clone(graph));
        }
        let mut size = self.common_projection_size;
        for (region, &value) in self.regions.iter().zip(&key) {
            for (count, amount) in size
                .iter_mut()
                .zip(block_footprint_size(region.arm(value).blocks()))
            {
                *count = count
                    .checked_add(amount)
                    .expect("a projection is bounded by validated source geometry");
            }
        }
        self.reserve_geometry(size)?;
        let graph = Arc::new(self.source.project_validated_branches_deferred(
            &self.regions,
            &self.shared_pipes,
            &key,
        )?);
        self.projections.insert(key, Arc::clone(&graph));
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn branch_selection_values_keep_shared_branch_and_selective_outcomes_consistent() {
        for condition in ["enable", "!enable", "enable ^ toggle"] {
            let program = crate::BlockGraph::from_text(&format!(
                r"BLOG 1.0
module main {{
  in enable
  in toggle
  in q_in: data = 0
  out q_out: data = 1
  in probe: data = 4
  0: Port [0,0,0] role=input
  1: Port [0,0,2] role=output
  4: Port [3,0,0] role=input
  5: XZX [3,0,1]
  6: XZ [3,0,2]
  4 -> +Z
  5 -> +Z
  branch b {{
    false {{
      2: XZX [0,0,1]
      0 -H> +Z
      [0,0,1] -H> +Z
    }}
    true {{
      3: ZXZ [0,0,1]
      0 -> +Z
      [0,0,1] -> +Z
    }}
  }}
  selected = {condition}
  resolve b if selected
  resolve 6 if selected
}}"
            ))
            .unwrap();
            let graph = program
                .materialize_root_graph()
                .unwrap()
                .fix_shadowed_faces();
            let topology =
                crate::GuardedTopology::new(&graph, crate::ModuleCertificationLimits::DEFAULT)
                    .unwrap();
            for selected in [true, false] {
                let absent = topology
                    .branches
                    .iter()
                    .filter_map(|(_, position, _)| (!selected).then_some(*position))
                    .collect();
                let values = topology.branch_selection_values(&absent).unwrap().unwrap();
                let evaluate = |root| topology.diagram.evaluate(root, |index| values[index]);
                assert_eq!(evaluate(topology.branches[0].2), selected);
                assert_eq!(evaluate(topology.resolves[0].1), selected, "{condition}");
                let outcomes = topology
                    .variables
                    .iter()
                    .zip(&values)
                    .filter_map(|(variable, &value)| match variable {
                        crate::GuardedVariable::Outcome(name) => Some((name.as_str(), value)),
                        crate::GuardedVariable::Branch { .. } => None,
                    })
                    .collect::<std::collections::HashMap<_, _>>();
                assert_eq!(
                    topology.evaluate_source(topology.branches[0].2, |name| outcomes[name]),
                    selected,
                    "{condition}"
                );
            }
        }
    }

    #[test]
    fn branch_selection_values_solve_correlated_conditions_within_the_boolean_budget() {
        let source = include_str!("../../docs/fixtures/conditional_cz_strip.blog")
            .replace("resolve cz1 if enable1", "resolve cz1 if !enable0");
        let program = crate::BlockGraph::from_text(&source).unwrap();
        let graph = program.root().local_body().fix_shadowed_faces();
        let limits = crate::ModuleCertificationLimits {
            max_boolean_steps: 1_000_000,
            ..crate::ModuleCertificationLimits::DEFAULT
        };
        let mut topology = crate::GuardedTopology::new(&graph, limits).unwrap();
        for mask in 0..8 {
            let absent = topology
                .branches
                .iter()
                .enumerate()
                .filter_map(|(index, (_, position, _))| {
                    (mask & (1 << index) == 0).then_some(*position)
                })
                .collect();
            let values = topology.branch_selection_values(&absent).unwrap();
            assert_eq!(values.is_some(), (mask & 1 != 0) != (mask & 2 != 0));
            if let Some(values) = values {
                for (index, (_, _, root)) in topology.branches.iter().enumerate() {
                    assert_eq!(
                        topology.diagram.evaluate(*root, |index| values[index]),
                        mask & (1 << index) != 0
                    );
                }
            }
        }
        // An unreachable all-true view still uses the existing static fallback.
        program.summarize_root(limits).unwrap();
        topology
            .diagram
            .charge(limits.max_boolean_steps - topology.diagram.steps())
            .unwrap();
        assert_eq!(
            topology
                .branch_selection_values(&rustc_hash::FxHashSet::default())
                .unwrap_err()
                .resource,
            "Boolean work steps"
        );
    }

    #[test]
    fn selector_image_counter_overflow_is_a_resource_failure() {
        let mut visited = usize::MAX - 1;
        visit_image_state(&mut visited, usize::MAX).expect("last representable state is allowed");
        assert!(matches!(
            visit_image_state(&mut visited, usize::MAX),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "structural selector image",
                    observed: usize::MAX,
                    limit: usize::MAX,
                }
            ))
        ));
        assert_eq!(visited, usize::MAX);
    }

    use super::*;

    #[test]
    fn availability_queries_preserve_dependencies_across_waves_and_collection() {
        fn reference(
            topology: &GuardedTopology,
            root: DecisionId,
            known: &BTreeSet<String>,
        ) -> bool {
            topology
                .diagram
                .variables(root)
                .iter()
                .all(|&index| match &topology.variables[index] {
                    GuardedVariable::Outcome(name) => known.contains(name),
                    GuardedVariable::Branch { condition, .. } => {
                        reference(topology, *condition, known)
                    }
                })
        }

        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        let outcomes = (0..3)
            .map(|index| {
                topology
                    .diagram
                    .make_node(index, DECISION_FALSE, DECISION_TRUE)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        topology.variables = ["a", "b", "c"]
            .map(|name| GuardedVariable::Outcome(name.into()))
            .into();
        let condition = topology
            .diagram
            .apply(BooleanOp::Xor, outcomes[0], outcomes[1])
            .unwrap();
        topology.variables.push(GuardedVariable::Branch {
            name: "choice".into(),
            condition,
        });
        let branch = topology
            .diagram
            .make_node(3, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let mut roots = vec![DECISION_FALSE, DECISION_TRUE, condition, branch];
        roots.extend(outcomes);
        for index in 0..48 {
            let op = [BooleanOp::And, BooleanOp::Or, BooleanOp::Xor][index % 3];
            let root = topology
                .diagram
                .apply(
                    op,
                    roots[index % roots.len()],
                    roots[(index * 7 + 3) % roots.len()],
                )
                .unwrap();
            roots.push(root);
        }
        for mask in 0..8 {
            let known = ["a", "b", "c"]
                .into_iter()
                .enumerate()
                .filter(|&(index, _)| mask & (1 << index) != 0)
                .map(|(_, name)| name.to_owned())
                .collect();
            let mut query = topology.availability(&known).unwrap();
            for &root in &roots {
                let expected = reference(&topology, root, &known);
                assert_eq!(
                    query.available(root, &mut topology.diagram).unwrap(),
                    expected
                );
            }
            let spent = topology.diagram.steps();
            for &root in roots.iter().rev() {
                query.available(root, &mut topology.diagram).unwrap();
            }
            assert_eq!(
                topology.diagram.steps(),
                spent,
                "shared roots need no second traversal"
            );
            let appended = topology
                .diagram
                .apply(BooleanOp::Xor, roots[mask + 2], roots[mask + 7])
                .unwrap();
            let expected = reference(&topology, appended, &known);
            assert_eq!(
                query.available(appended, &mut topology.diagram).unwrap(),
                expected
            );
            roots.push(appended);
            drop(query);
            topology.collect_garbage(roots.iter_mut());
        }
    }

    #[test]
    fn availability_learns_outcomes_and_refreshes_after_collection() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        topology
            .diagram
            .make_node(99, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let a = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let b = topology
            .diagram
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let condition = topology.diagram.apply(BooleanOp::Xor, a, b).unwrap();
        topology.variables = vec![
            GuardedVariable::Outcome("a".into()),
            GuardedVariable::Outcome("b".into()),
            GuardedVariable::Branch {
                name: "choice".into(),
                condition,
            },
        ];
        let branch = topology
            .diagram
            .make_node(2, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let mut roots = [a, b, branch];
        let known = BTreeSet::from(["a".into()]);
        let mut query = topology.availability(&known).unwrap();
        assert!(query.available(a, &mut topology.diagram).unwrap());
        assert!(!query.available(b, &mut topology.diagram).unwrap());
        assert!(!query.available(branch, &mut topology.diagram).unwrap());
        assert!(topology.collect_garbage(roots.iter_mut()) > 0);
        topology.refresh_availability(&mut query, &known).unwrap();
        assert!(query.available(roots[0], &mut topology.diagram).unwrap());
        assert!(!query.available(roots[1], &mut topology.diagram).unwrap());
        assert!(!query.available(roots[2], &mut topology.diagram).unwrap());
        query.learn(["b"], &mut topology.diagram).unwrap();
        let spent = topology.diagram.steps();
        assert!(query.available(roots[0], &mut topology.diagram).unwrap());
        assert_eq!(topology.diagram.steps(), spent, "known roots remain cached");
        assert!(query.available(roots[1], &mut topology.diagram).unwrap());
        assert!(query.available(roots[2], &mut topology.diagram).unwrap());
    }

    #[test]
    fn uncached_availability_checks_spend_the_boolean_work_budget() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        topology.diagram =
            BooleanDecisionDiagram::with_limits(bloq_utils::boolean::BooleanLimits {
                max_nodes: 4,
                max_steps: 16,
            });
        topology.variables = vec![GuardedVariable::Outcome("a".into())];
        let root = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let mut query = topology
            .availability(&BTreeSet::from(["a".into()]))
            .unwrap();
        let remaining = topology.diagram.limits().max_steps - topology.diagram.steps();
        topology.diagram.charge(remaining).unwrap();
        assert!(matches!(
            query.available(root, &mut topology.diagram),
            Err(BooleanResourceError {
                resource: "Boolean work steps",
                observed: 17,
                limit: 16,
            })
        ));
    }

    #[test]
    fn selector_image_preserves_correlations_with_interleaved_variable_indices() {
        let mut source = BooleanDecisionDiagram::default();
        let a = source.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = source.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let parity = source.apply(BooleanOp::Xor, a, b).unwrap();
        let mut output = BooleanDecisionDiagram::default();
        let image = boolean_image(
            &source,
            &mut output,
            vec![(6, a), (1, parity), (4, b), (8, DECISION_TRUE)],
            100,
        )
        .unwrap();
        for values in 0..16 {
            let selected = |slot: usize| values & (1 << slot) != 0;
            let actual = output.evaluate(image, |index| match index {
                6 => selected(0),
                1 => selected(1),
                4 => selected(2),
                8 => selected(3),
                _ => false,
            });
            assert_eq!(
                actual,
                selected(3) && selected(1) == (selected(0) ^ selected(2))
            );
        }
    }

    #[test]
    fn selector_image_factors_two_interleaved_correlations_and_a_constant() {
        let mut source = BooleanDecisionDiagram::default();
        let a = source.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = source.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let not_a = source.negate(a).unwrap();
        let not_b = source.negate(b).unwrap();
        let mut output = BooleanDecisionDiagram::default();
        let image = boolean_image(
            &source,
            &mut output,
            vec![
                (6, a),
                (1, not_a),
                (4, b),
                (8, not_b),
                (3, DECISION_TRUE),
                (5, DECISION_FALSE),
            ],
            2,
        )
        .unwrap();
        for values in 0..64 {
            let selected = |slot: usize| values & (1 << slot) != 0;
            let actual = output.evaluate(image, |index| match index {
                6 => selected(0),
                1 => selected(1),
                4 => selected(2),
                8 => selected(3),
                3 => selected(4),
                5 => selected(5),
                _ => false,
            });
            assert_eq!(
                actual,
                selected(4)
                    && !selected(5)
                    && selected(0) != selected(1)
                    && selected(2) != selected(3)
            );
        }
    }

    #[test]
    fn selector_image_shares_witness_and_work_limits_across_components() {
        let mut source = BooleanDecisionDiagram::default();
        let a = source.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = source.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let not_a = source.negate(a).unwrap();
        let not_b = source.negate(b).unwrap();
        for pair in [[(6, a), (1, not_a)], [(4, b), (8, not_b)]] {
            boolean_image(
                &source,
                &mut BooleanDecisionDiagram::default(),
                pair.into(),
                1,
            )
            .unwrap();
        }
        assert!(matches!(
            boolean_image(
                &source,
                &mut BooleanDecisionDiagram::default(),
                vec![(6, a), (1, not_a), (4, b), (8, not_b)],
                1,
            ),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "structural selector image",
                    observed: 2,
                    limit: 1,
                }
            ))
        ));

        let parity = source.apply(BooleanOp::Xor, a, b).unwrap();
        assert_eq!(
            boolean_image(
                &source,
                &mut BooleanDecisionDiagram::default(),
                vec![(9, parity)],
                1,
            )
            .unwrap(),
            DECISION_TRUE
        );
        assert!(matches!(
            boolean_image(
                &source,
                &mut BooleanDecisionDiagram::default(),
                vec![(9, parity)],
                0,
            ),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "structural selector image",
                    observed: 1,
                    limit: 0,
                }
            ))
        ));
        let mut no_work = BooleanDecisionDiagram::with_limits(bloq_utils::boolean::BooleanLimits {
            max_nodes: 1,
            max_steps: 0,
        });
        assert!(matches!(
            boolean_image(&source, &mut no_work, vec![(9, parity)], 1),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "Boolean work steps",
                    ..
                }
            ))
        ));
    }

    #[test]
    fn constant_case_coordinates_spend_work_before_enumeration() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        topology.diagram =
            BooleanDecisionDiagram::with_limits(bloq_utils::boolean::BooleanLimits {
                max_nodes: 0,
                max_steps: 1,
            });
        // Constant coordinates count even when no Boolean node is created.
        assert!(matches!(
            topology.cases(&[DECISION_TRUE], DECISION_TRUE),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "Boolean work steps",
                    observed: 2,
                    limit: 1,
                }
            ))
        ));
    }

    #[test]
    fn local_cases_partition_enabled_correlated_values_exactly() {
        let mut topology =
            GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::DEFAULT).unwrap();
        let a = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let b = topology
            .diagram
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let c = topology
            .diagram
            .make_node(2, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let parity = topology.diagram.apply(BooleanOp::Xor, a, b).unwrap();
        let product = topology.diagram.apply(BooleanOp::And, b, c).unwrap();
        topology.domain = topology.diagram.apply(BooleanOp::Or, a, c).unwrap();
        let roots = [DECISION_TRUE, parity, product, a, parity];
        let cases = topology.cases(&roots, b).unwrap();
        for assignment in 0..8 {
            let evaluate = |root| {
                topology
                    .diagram
                    .evaluate(root, |index| assignment & (1 << index) != 0)
            };
            let enabled = evaluate(b) && evaluate(topology.domain);
            let selected = cases
                .iter()
                .filter(|(_, guard)| evaluate(*guard))
                .collect::<Vec<_>>();
            if evaluate(topology.domain) {
                assert_eq!(selected.len(), usize::from(enabled));
            }
            if enabled {
                assert_eq!(selected[0].0, roots.map(evaluate));
            }
        }
        topology.limits.max_guarded_domain_size = cases.len() - 1;
        assert!(matches!(
            topology.cases(&roots, b),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "local structural patterns",
                    ..
                }
            ))
        ));
        assert!(topology.cases(&roots, DECISION_FALSE).unwrap().is_empty());
    }

    #[test]
    fn independent_neighborhoods_do_not_form_a_joint_projection_domain() {
        let mut source = String::from("BLOG 1.0\nmodule main {\n");
        for index in 0..40 {
            let x = index * 3;
            source.push_str(&format!("in s{index}\n{}: ZXZ [{x},0,0]\n{}: ZXZ [{x},0,2]\nbranch b{index} {{\nfalse {{\n{}: XZX [{x},0,1]\n[{x},0,0] -H> +Z\n[{x},0,1] -H> +Z\n}}\ntrue {{\n{}: ZXZ [{x},0,1]\n[{x},0,0] -> +Z\n[{x},0,1] -> +Z\n}}\n}}\nresolve b{index} if s{index}\n", index*4, index*4+1, index*4+2, index*4+3));
        }
        source.push_str("}\n");
        let ast = crate::parse_blog_program_to_ast(&source).unwrap();
        let program = crate::lower_blog_graph_ast_deferred(&ast).unwrap();
        let graph = program.materialize_flat_graph().unwrap();
        assert!(matches!(
            GuardedTopology::new(
                &graph,
                ModuleCertificationLimits {
                    max_boolean_nodes: 0,
                    ..ModuleCertificationLimits::DEFAULT
                }
            ),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited {
                    phase: "Boolean nodes",
                    observed: 1,
                    limit: 0,
                }
            ))
        ));
        assert!(matches!(
            GuardedTopology::new(&graph, ModuleCertificationLimits {
                    max_expanded_blocks: 2 * graph.block_count() - 1,
                ..ModuleCertificationLimits::DEFAULT
            }),
            Err(BlockGraphError::Stabilizer(StabilizerError::ResourceLimited {
                phase: "retained projection blocks", observed, limit,
            })) if observed > limit
        ));
        let mut topology = GuardedTopology::new(
            &graph,
            ModuleCertificationLimits {
                max_guarded_domain_size: 4,
                ..ModuleCertificationLimits::DEFAULT
            },
        )
        .unwrap();
        assert_eq!(topology.branches.len(), 40);
        let mut products = DECISION_FALSE;
        for index in 0..12 {
            let name = format!("s{index}");
            let outcome = topology.condition(&Expr::Var(name));
            let branch = topology.branches[index].2;
            assert_eq!(
                topology.diagram.node(branch).unwrap().variable,
                topology.diagram.node(outcome).unwrap().variable + 1,
            );
            let term = topology
                .diagram
                .apply(BooleanOp::And, outcome, branch)
                .unwrap();
            products = topology
                .diagram
                .apply(BooleanOp::Xor, products, term)
                .unwrap();
        }
        let mut compact = topology.diagram.clone();
        compact.collect_garbage([&mut products]);
        // Grouped outcome/branch variables need 10,237 live nodes here.
        assert_eq!(compact.nodes().len(), 46);
        assert_eq!(topology.projections.len(), 1);
        let reference = topology.project(DECISION_TRUE).unwrap();
        assert_eq!(
            topology.retained_projection_size[0],
            4 * graph.block_count()
        );
        for variants in topology.sites.values() {
            assert_eq!(variants.len(), 2);
            assert_eq!(
                variants
                    .iter()
                    .filter(|variant| Arc::ptr_eq(&variant.graph, &reference))
                    .count(),
                1
            );
            for variant in variants {
                if !Arc::ptr_eq(&variant.graph, &reference) {
                    assert_eq!(variant.graph.block_count(), 3);
                }
            }
        }
        assert_eq!(reference.block_count(), 120);
        assert!(Arc::ptr_eq(
            &reference,
            &topology.project(DECISION_TRUE).unwrap()
        ));
        assert_eq!(topology.projections.len(), 1);
        for _ in 0..2 {
            for index in 0..40 {
                let name = format!("s{index}");
                let condition = topology.condition(&Expr::Var(name.clone()));
                assert!(topology.evaluate_source(condition, |candidate: &str| candidate == name));
                assert!(!topology.evaluate_source(condition, |_: &str| false));
            }
            topology.collect_garbage(std::iter::empty());
        }
    }

    #[test]
    fn compound_conditions_preserve_aliases_and_exact_shape_after_collection() {
        let var = |name: &str| Expr::Var(name.to_owned());
        let alias = Expr::Binary(crate::BinaryOp::Xor, Box::new(var("a")), Box::new(var("b")));
        let condition = Expr::Binary(
            crate::BinaryOp::Xor,
            Box::new(var("alias")),
            Box::new(Expr::Not(Box::new(var("a")))),
        );
        let mut graph = BlockGraph::new();
        graph
            .set_actions_with_inputs(
                vec![
                    Action::Let {
                        name: "alias".into(),
                        expr: alias,
                    },
                    Action::DiscardIf(condition.clone()),
                    Action::DiscardIf(condition.clone()),
                    Action::DiscardIf(var("alias")),
                    Action::DiscardIf(var("alias")),
                ],
                ["a", "b"].map(str::to_owned),
            )
            .unwrap();
        let mut topology =
            GuardedTopology::new(&graph, ModuleCertificationLimits::DEFAULT).unwrap();
        let before = topology.condition(&condition);
        for round in 0..2 {
            for values in 0..4 {
                let a = values & 1 != 0;
                let b = values & 2 != 0;
                let outcome = |name: &str| match name {
                    "a" => a,
                    "b" => b,
                    _ => panic!("unexpected source outcome {name}"),
                };
                assert_eq!(
                    topology.evaluate_source(topology.condition(&condition.clone()), outcome),
                    (a ^ b) ^ !a,
                );
                assert_eq!(
                    topology.evaluate_source(topology.condition(&var("alias")), outcome),
                    a ^ b,
                );
            }
            let removed = topology.collect_garbage(std::iter::empty());
            if round == 0 {
                assert!(removed > 0, "unused expression intermediates removed");
                assert_ne!(topology.condition(&condition), before, "root remapped");
            }
        }
        let absent = Expr::Binary(
            crate::BinaryOp::Xor,
            Box::new(Expr::Not(Box::new(var("a")))),
            Box::new(var("alias")),
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| topology.condition(&absent)))
                .is_err(),
            "equivalent reordered operands are not a stored source expression",
        );
    }

    #[test]
    fn local_geometry_preserves_primary_kinds_and_complete_incidence() {
        for item in [
            crate::GalleryItem::CNOT,
            crate::GalleryItem::CZSpatialH,
            crate::GalleryItem::CCZGateTeleport,
            crate::GalleryItem::GHZPatchRotations,
            crate::GalleryItem::MoveRotation,
        ] {
            let source = item
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            let mut topology =
                GuardedTopology::new(&source, ModuleCertificationLimits::DEFAULT).unwrap();
            assert_eq!(topology.projections.len(), 1);
            let reference = topology.project(DECISION_TRUE).unwrap();
            let mut checked_assignments = std::collections::HashSet::new();
            for (center, variants) in topology.sites.clone() {
                let center = IVec3::from_array(center);
                for variant in variants {
                    let full = topology.project(variant.guard).unwrap();
                    for block in std::iter::once(full.get_block(center).unwrap())
                        .chain(full.neighbors(center))
                    {
                        assert_eq!(
                            variant.graph.get_block(block.pos()),
                            Some(block),
                            "{item}: center {center}, block {}",
                            block.pos(),
                        );
                        let pipes = |graph: &BlockGraph| {
                            graph
                                .pipes_at(block.pos())
                                .cloned()
                                .collect::<std::collections::HashSet<_>>()
                        };
                        assert_eq!(pipes(&variant.graph), pipes(&full), "{item}: {center}");
                    }
                    if !Arc::ptr_eq(&variant.graph, &reference) {
                        assert!(variant.graph.actions().is_empty());
                    }
                    assert!(variant.graph.branch_definitions().is_empty());
                    let assignment = topology.assignment(variant.guard).unwrap().unwrap();
                    if checked_assignments.insert(assignment.clone()) {
                        let reference = source.project_branches_deferred(assignment).unwrap();
                        assert_eq!(
                            full.blocks().collect::<std::collections::HashSet<_>>(),
                            reference.blocks().collect(),
                            "{item}",
                        );
                        assert_eq!(
                            full.pipes().collect::<std::collections::HashSet<_>>(),
                            reference.pipes().collect(),
                            "{item}",
                        );
                        assert_eq!(full.actions(), reference.actions(), "{item}");
                    }
                }
            }
        }
    }

    #[test]
    fn full_projections_charge_selected_arms_before_materialization() {
        let source = "BLOG 1.0\nmodule main {\nin s\n0: ZXZ [0,0,0]\nbranch b {\nfalse {\n1: X [0,0,1]\n[0,0,0] -> +Z\n}\ntrue {\n2: ZXZ [0,0,1]\n3: ZXZ [0,0,2]\n[0,0,0] -> +Z\n[0,0,1] -> +Z\n}\n}\nresolve b if s\n}\n";
        let ast = crate::parse_blog_program_to_ast(source).unwrap();
        let program = crate::lower_blog_graph_ast_deferred(&ast).unwrap();
        let source = program.materialize_flat_graph().unwrap();
        for (blocks, cells, refused) in [
            (5, 5, None),
            (4, 5, Some("retained projection blocks")),
            (5, 4, Some("retained projection footprint cells")),
        ] {
            let limits = ModuleCertificationLimits {
                max_expanded_blocks: blocks,
                max_occupied_cells: cells,
                ..ModuleCertificationLimits::DEFAULT
            };
            let (mut topology, _) = GuardedTopology::new_with_geometry(&source, limits).unwrap();
            let reference = topology.project(DECISION_TRUE).unwrap();
            assert_eq!(reference.block_count(), 2);
            assert_eq!(topology.retained_projection_size, [2, 2]);
            let guard = topology.branches[0].2;
            if let Some(expected) = refused {
                assert!(matches!(
                    topology.project(guard),
                    Err(BlockGraphError::Stabilizer(StabilizerError::ResourceLimited {
                        phase,
                        observed: 5,
                        limit: 4,
                    })) if phase == expected
                ));
                assert_eq!(topology.projections.len(), 1);
                assert_eq!(topology.retained_projection_size, [2, 2]);
            } else {
                let selected = topology.project(guard).unwrap();
                assert_eq!(selected.block_count(), 3);
                assert_eq!(topology.retained_projection_size, [5, 5]);
                assert!(Arc::ptr_eq(&selected, &topology.project(guard).unwrap()));
                assert_eq!(topology.retained_projection_size, [5, 5]);
                assert_eq!(topology.projections.len(), 2);
            }
        }
    }

    #[test]
    fn unchanged_sites_share_one_reference_including_source_actions() {
        let source = crate::GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let mut topology =
            GuardedTopology::new(&source, ModuleCertificationLimits::DEFAULT).unwrap();
        let reference = topology.project(DECISION_TRUE).unwrap();
        assert!(!reference.actions().is_empty());
        assert_eq!(topology.projections.len(), 1);
        assert_eq!(
            topology.retained_projection_size,
            topology.common_projection_size
        );
        for (position, variants) in &topology.sites {
            for variant in variants {
                assert!(Arc::ptr_eq(&variant.graph, &reference));
                let local = crate::ZXGraph::from_local_block_neighborhood(
                    &variant.graph,
                    IVec3::from_array(*position),
                )
                .unwrap();
                assert!(local.actions().is_empty());
            }
        }
    }
}
