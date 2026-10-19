//! Correlation surfaces composed over Boolean-valued Pauli coefficients.
//!
//! Each local tensor contributes its two-arm alternatives once. Compatible
//! instances reuse their definition's relation after internal seam reduction;
//! parent composition closes the connecting seams. Coefficients retain shared
//! Boolean ancestry throughout, including products of independent selectors.

#[cfg(test)]
mod component_tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use bloq_utils::boolean::{
    BooleanOp, BooleanResourceError, BooleanRow, BooleanRowSpace, DECISION_FALSE as ZERO,
    DECISION_TRUE as ONE, DecisionId, eliminate_boolean_rows, reduce_boolean_rows,
};
use glam::IVec3;

use super::guarded_readouts::{GuardedReadoutPlan, LocalArm, LocalTopology};
use super::runtime_basis::authored_boundary_columns;
use super::{NodeKind, ZXGraph};
use crate::{
    Action, BlockGraphError, GuardedTopology, MaterializedModuleSite, MeasureTarget,
    MeasurementObservable, ModuleCertificationLimits, Pauli, PauliBasis, StabilizerError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SurfaceColumn {
    /// Independent support of a site with no incident pipe in an active variant.
    Node([i32; 3]),
    /// A directed half-edge in its source node's native Pauli frame.
    Edge([i32; 3], [i32; 3]),
}

#[derive(Debug, Clone)]
pub(crate) struct SurfaceTopology {
    pub guard: DecisionId,
    pub zx: Arc<ZXGraph>,
    pub node: usize,
}

/// One factored, complete module correlation space. No row-space product is
/// formed for combinations of choices in different regions or child modules.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuardedSurfaceSpace {
    pub topology: GuardedTopology,
    pub(crate) columns: BTreeMap<SurfaceColumn, usize>,
    pub(crate) local: BTreeMap<[i32; 3], Vec<SurfaceTopology>>,
    pub(crate) rows: Vec<BooleanRow>,
    surfaces: Vec<GuardedSurface>,
    limits: ModuleCertificationLimits,
    module_sites: BTreeMap<[i32; 3], usize>,
    next_readout_collection: usize,
}

// Reclamation stays finite when callers raise or disable the witness quota.
const READOUT_COLLECTION_TARGET: usize = 1_000_000;

fn next_readout_collection(retained: usize, limit: usize) -> usize {
    let growth = (READOUT_COLLECTION_TARGET / 4).max(retained / 4);
    retained
        .saturating_add(growth)
        .max(READOUT_COLLECTION_TARGET)
        .min(limit)
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardedSurfaceKind {
    Readout {
        name: String,
        folds: Vec<(String, DecisionId)>,
    },
    LogicalReadout,
    OutputFrame {
        port: IVec3,
        basis: PauliBasis,
        /// Later output-frame bits in this row's triangular correction equation.
        /// Each guard includes this row's activation; absent axes remain zero.
        folds: Vec<(IVec3, PauliBasis, DecisionId)>,
    },
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuardedSurface {
    pub kind: GuardedSurfaceKind,
    /// Conditions already include the corresponding source feedback expression.
    pub feedbacks: Vec<(u32, DecisionId)>,
    pub(super) row: BooleanRow,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AvailabilityKey {
    Site([i32; 3]),
    Feedback(usize),
}

struct ModuleScope<'a> {
    parent: Option<usize>,
    depth: usize,
    path: &'a str,
    children: BTreeMap<&'a str, usize>,
}

struct ModuleScopes<'a> {
    nodes: Vec<ModuleScope<'a>>,
}

impl<'a> ModuleScopes<'a> {
    fn new() -> Self {
        Self {
            nodes: vec![ModuleScope {
                parent: None,
                depth: 0,
                path: "",
                children: BTreeMap::new(),
            }],
        }
    }

    fn insert(&mut self, path: &'a str) -> usize {
        let mut owner = 0;
        if path.is_empty() {
            return owner;
        }
        let mut end = 0;
        for component in path.split("__") {
            end += component.len();
            owner = if let Some(&child) = self.nodes[owner].children.get(component) {
                child
            } else {
                let child = self.nodes.len();
                let depth = self.nodes[owner].depth + 1;
                self.nodes.push(ModuleScope {
                    parent: Some(owner),
                    depth,
                    path: &path[..end],
                    children: BTreeMap::new(),
                });
                self.nodes[owner].children.insert(component, child);
                child
            };
            end += 2;
        }
        owner
    }

    fn common_ancestor(&self, mut left: usize, mut right: usize) -> usize {
        if left == 0 || right == 0 {
            return 0;
        }
        while self.nodes[left].depth > self.nodes[right].depth {
            left = self.nodes[left]
                .parent
                .expect("non-root scope has a parent");
        }
        while self.nodes[right].depth > self.nodes[left].depth {
            right = self.nodes[right]
                .parent
                .expect("non-root scope has a parent");
        }
        while left != right {
            left = self.nodes[left]
                .parent
                .expect("distinct peers are non-root");
            right = self.nodes[right]
                .parent
                .expect("distinct peers are non-root");
        }
        left
    }

    fn lexical_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut pending = vec![0];
        while let Some(scope) = pending.pop() {
            order.push(scope);
            pending.extend(self.nodes[scope].children.values().rev().copied());
        }
        order
    }
}

#[derive(Default)]
struct GuardedPivots {
    rows: BooleanRowSpace,
    owners: BTreeMap<usize, usize>,
}

impl GuardedPivots {
    /// Insert a row already reduced against the existing pivots on their
    /// activation. New pivot pieces are disjoint from the existing owner.
    /// Availability pivots require `REDUCE_ABOVE` because their columns are
    /// later released; accepted readouts only substitute into future rows.
    fn insert_residual<const REDUCE_ABOVE: bool>(
        &mut self,
        mut row: BooleanRow,
        columns: impl Iterator<Item = usize>,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(BooleanRow, Vec<(usize, DecisionId)>), BooleanResourceError> {
        let mut added = Vec::new();
        for column in columns {
            diagram.charge(1)?;
            let selected = diagram.apply(BooleanOp::And, row.active, row.get(column))?;
            if selected == ZERO {
                continue;
            }
            let mut pivot = BooleanRow::new(selected);
            pivot.xor_scaled(&row, selected, diagram)?;
            let owner = self.owners.get(&column).copied();
            let previous = owner
                .map(|owner| self.rows.take(owner, diagram))
                .transpose()?
                .flatten();
            if REDUCE_ABOVE {
                self.rows.reduce_column(column, &pivot, diagram)?;
            }
            if let Some(mut previous) = previous {
                previous.xor_scaled(&pivot, ONE, diagram)?;
                previous.active = diagram.apply(BooleanOp::Or, previous.active, selected)?;
                self.rows.replace(
                    owner.expect("a previous row is taken only from an existing owner"),
                    previous,
                    diagram,
                )?;
            } else {
                let owner = self
                    .rows
                    .insert_indexed(pivot, diagram)?
                    .expect("selected pivot is active and has nonzero support");
                self.owners.insert(column, owner);
            }
            added.push((column, selected));
            let absent = diagram.negate(selected)?;
            row.active = diagram.apply(BooleanOp::And, row.active, absent)?;
        }
        Ok((row, added))
    }

    fn project(
        &self,
        row: &mut BooleanRow,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(row.terms().len())?;
        // Forward-echelon rows can contain later pivots. Clearing an earlier
        // pivot may introduce one that was absent from the original support.
        // Resume on the changed support, rather than visiting every pivot in
        // the enclosing graph for each sparse local row.
        let mut after = None;
        loop {
            diagram.charge(row.terms().len().max(1).ilog2() as usize + 1)?;
            let start = after.map_or(0, |last| {
                row.terms().partition_point(|&(column, _)| column <= last)
            });
            let mut selected = None;
            for &(column, value) in &row.terms()[start..] {
                diagram.charge(1)?;
                if let Some(&owner) = self.owners.get(&column) {
                    selected = Some((column, value, owner));
                    break;
                }
            }
            let Some((column, value, owner)) = selected else {
                break;
            };
            let factor = diagram.apply(BooleanOp::And, row.active, value)?;
            row.xor_scaled(
                self.rows
                    .row(owner)
                    .expect("pivot owners retain their indexed rows"),
                factor,
                diagram,
            )?;
            after = Some(column);
        }
        Ok(())
    }
}

struct GuardedAvailability {
    pivots: GuardedPivots,
    groups: BTreeMap<AvailabilityKey, Vec<usize>>,
    active: BTreeSet<usize>,
    auxiliary_start: usize,
}

impl GuardedAvailability {
    fn new(
        mut rows: Vec<BooleanRow>,
        constraints: &[(AvailabilityKey, Vec<(usize, DecisionId)>)],
        auxiliary_start: usize,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(Self, BooleanRowSpace), BooleanResourceError> {
        let mut groups = BTreeMap::<_, Vec<_>>::new();
        let mut columns = HashMap::<usize, Vec<_>>::new();
        diagram.charge(constraints.len())?;
        for (index, (key, terms)) in constraints.iter().enumerate() {
            let tag = auxiliary_start + index;
            groups.entry(*key).or_default().push(tag);
            diagram.charge(terms.len())?;
            for &(column, guard) in terms {
                columns.entry(column).or_default().push((tag, guard));
            }
        }
        for row in &mut rows {
            diagram.charge(row.terms().len())?;
            let mut tags = BTreeMap::new();
            for &(column, coefficient) in row.terms() {
                if let Some(aliases) = columns.get(&column) {
                    diagram.charge(aliases.len())?;
                    for &(tag, guard) in aliases {
                        let value = tags.entry(tag).or_insert(ZERO);
                        let term = diagram.apply(BooleanOp::And, guard, coefficient)?;
                        *value = diagram.apply(BooleanOp::Xor, *value, term)?;
                    }
                }
            }
            for (tag, coefficient) in tags {
                row.set(tag, coefficient);
            }
        }
        let mut remaining = BooleanRowSpace::new(rows, diagram)?;
        let mut pivots = GuardedPivots::default();
        let active =
            (auxiliary_start..auxiliary_start + constraints.len()).collect::<BTreeSet<_>>();
        for &column in &active {
            let pivot = remaining.eliminate_column_sparse(column, diagram)?;
            if pivot.active != ZERO {
                pivots.rows.reduce_column(column, &pivot, diagram)?;
                let owner = pivots
                    .rows
                    .insert_indexed(pivot, diagram)?
                    .expect("active elimination pivot retains its column");
                pivots.owners.insert(column, owner);
            }
        }
        let rows = remaining.into_rows(diagram)?;
        let mut available = BooleanRowSpace::default();
        for mut row in rows {
            row.truncate_columns(auxiliary_start);
            let active = row.active;
            diagram.charge(row.terms().len())?;
            row.map_coefficients(|value| diagram.apply(BooleanOp::And, active, value))?;
            if !row.terms().is_empty() {
                available.push(row, diagram)?;
            }
        }
        Ok((
            Self {
                pivots,
                groups,
                active,
                auxiliary_start,
            },
            available,
        ))
    }

    fn release(
        &mut self,
        pending: &BTreeSet<AvailabilityKey>,
        readouts: &GuardedPivots,
        available: &mut BooleanRowSpace,
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(self.groups.len())?;
        for (_, columns) in self.groups.extract_if(.., |key, _| !pending.contains(key)) {
            for column in columns {
                self.active.remove(&column);
                let row = self
                    .pivots
                    .owners
                    .remove(&column)
                    .map(|owner| self.pivots.rows.take(owner, diagram))
                    .transpose()?
                    .flatten();
                self.pivots.rows.erase_column(column, diagram)?;
                let Some(mut row) = row else {
                    continue;
                };
                row.set(column, ZERO);
                diagram.charge(row.terms().len())?;
                let columns = row
                    .terms()
                    .iter()
                    .filter_map(|&(column, _)| self.active.contains(&column).then_some(column))
                    .collect::<Vec<_>>();
                let (mut row, _) =
                    self.pivots
                        .insert_residual::<true>(row, columns.into_iter(), diagram)?;
                if row.active != ZERO {
                    row.truncate_columns(self.auxiliary_start);
                    let active = row.active;
                    diagram.charge(row.terms().len())?;
                    row.map_coefficients(|value| diagram.apply(BooleanOp::And, active, value))?;
                    readouts.project(&mut row, diagram)?;
                    if !row.terms().is_empty() {
                        available.push(row, diagram)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl GuardedSurface {
    /// Predicate under which this surface supplies a correlation row.
    /// A false output-axis predicate requires no readout or decoder solve.
    pub fn activation(&self) -> DecisionId {
        self.row.active
    }

    pub(super) fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        let (readouts, outputs) = match &mut self.kind {
            GuardedSurfaceKind::Readout { folds, .. } => (Some(folds), None),
            GuardedSurfaceKind::OutputFrame { folds, .. } => (None, Some(folds)),
            GuardedSurfaceKind::LogicalReadout => (None, None),
        };
        self.row
            .decisions_mut()
            .chain(self.feedbacks.iter_mut().map(|(_, root)| root))
            .chain(readouts.into_iter().flatten().map(|(_, root)| root))
            .chain(outputs.into_iter().flatten().map(|(_, _, root)| root))
    }
}

impl GuardedSurfaceSpace {
    /// Relation rows without the per-site inputs that only readout planning uses.
    pub(crate) fn relation_view(&self) -> Self {
        Self {
            topology: self.topology.clone(),
            columns: self.columns.clone(),
            local: BTreeMap::new(),
            rows: self.rows.clone(),
            surfaces: Vec::new(),
            limits: self.limits,
            module_sites: BTreeMap::new(),
            next_readout_collection: self.next_readout_collection,
        }
    }

    /// Builds a factored Boolean row space for one guarded topology.
    ///
    /// # Errors
    ///
    /// Returns an error if local geometry, Boolean work, or configured limits fail.
    ///
    /// # Panics
    ///
    /// Panics if validated topology metadata references a missing local column.
    pub fn new(
        topology: GuardedTopology,
        sites: &HashMap<IVec3, MaterializedModuleSite>,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, BlockGraphError> {
        Self::with_relations(topology, sites, limits, &HashMap::new())
    }

    pub(crate) fn with_relations(
        mut topology: GuardedTopology,
        sites: &HashMap<IVec3, MaterializedModuleSite>,
        limits: ModuleCertificationLimits,
        relations: &HashMap<String, Arc<crate::summary::relations::ModuleRelations>>,
    ) -> Result<Self, BlockGraphError> {
        // Intern components once, including empty ancestors. Numeric parents
        // avoid cloning and comparing every shared prefix during composition.
        let mut scopes = ModuleScopes::new();
        let owners = topology
            .sites
            .keys()
            .map(|&position| {
                let path = sites
                    .get(&IVec3::from_array(position))
                    .map_or("", |site| site.instance_path.as_str());
                (position, scopes.insert(path))
            })
            .collect::<BTreeMap<_, _>>();
        // A source block has at most six incident directions. Most compilations
        // fit the limit even under this allocation-free global upper bound.
        let upper_bound = topology.sites.values().fold(0usize, |sum, variants| {
            sum.saturating_add(variants.len().saturating_mul(6).saturating_add(1))
        });
        if upper_bound > limits.max_local_columns {
            let mut dimensions = vec![0usize; scopes.nodes.len()];
            for (&position, variants) in &topology.sites {
                let neighbors = variants
                    .iter()
                    .flat_map(|variant| {
                        variant
                            .graph
                            .neighbors(IVec3::from_array(position))
                            .into_iter()
                            .map(|block| block.pos().to_array())
                    })
                    .collect::<BTreeSet<_>>();
                let observed = &mut dimensions[owners[&position]];
                *observed += neighbors.len()
                    + usize::from(variants.iter().any(|variant| {
                        variant
                            .graph
                            .neighbors(IVec3::from_array(position))
                            .is_empty()
                    }));
                if *observed > limits.max_local_columns {
                    return Err(StabilizerError::ResourceLimited {
                        phase: "local ZX columns",
                        observed: *observed,
                        limit: limits.max_local_columns,
                    }
                    .into());
                }
            }
        }
        // Keep authored instance boundaries after consuming the local ZX views.
        // Instance ids are local to this source plan; they never identify equal
        // physical owners in different instances.
        let mut modules = HashMap::new();
        let module_sites = topology
            .sites
            .keys()
            .map(|&position| {
                let next = modules.len();
                (position, *modules.entry(owners[&position]).or_insert(next))
            })
            .collect();
        let mut columns = BTreeMap::new();
        let mut groups = vec![Vec::<BooleanRow>::new(); scopes.nodes.len()];
        let mut seams = vec![BTreeMap::new(); scopes.nodes.len()];
        let mut local = BTreeMap::new();
        for (&position, variants) in &topology.sites {
            let owner = owners[&position];
            let mut tensors = BTreeMap::<Vec<(usize, DecisionId)>, DecisionId>::new();
            let mut local_variants = Vec::new();
            let mut local_keys = HashMap::new();
            for variant in variants {
                let zx = Arc::new(ZXGraph::from_local_block_neighborhood(
                    &variant.graph,
                    IVec3::from_array(position),
                )?);
                let node = *zx
                    .node_at(IVec3::from_array(position))
                    .expect("local variant includes its block");
                for terms in zx.local_stabilizer_flow_supports_for_node_kind(node.id, node.kind) {
                    let mut row = BooleanRow::default();
                    for (column, mut pauli) in terms {
                        let key = if column < zx.nodes().len() {
                            if zx.neighbor_edges(node.id).next().is_some() {
                                continue;
                            }
                            SurfaceColumn::Node(position)
                        } else {
                            let edge = &zx.edges()[column - zx.nodes().len()];
                            debug_assert_eq!(edge.n1, node.id);
                            if edge.hadamard && edge.n1 > edge.n2 {
                                pauli = pauli.flip();
                            }
                            SurfaceColumn::Edge(position, zx.nodes()[edge.n2].pos.to_array())
                        };
                        let next = columns.len();
                        let column = *columns.entry(key).or_insert(next);
                        for (index, axis) in [Pauli::X, Pauli::Z].into_iter().enumerate() {
                            if pauli & axis {
                                row.set(2 * column + index, ONE);
                            }
                        }
                    }
                    let active = tensors.entry(row.into_terms()).or_insert(ZERO);
                    *active = topology
                        .diagram
                        .apply(BooleanOp::Or, *active, variant.guard)?;
                }
                for (neighbor, edge_id) in zx.neighbor_edges(node.id) {
                    let other = zx.nodes()[neighbor].pos.to_array();
                    if position >= other {
                        continue;
                    }
                    let parent = scopes.common_ancestor(owner, owners[&other]);
                    let hadamard = seams[parent].entry((position, other)).or_insert(ZERO);
                    if zx.edge_by_id(edge_id).hadamard {
                        *hadamard =
                            topology
                                .diagram
                                .apply(BooleanOp::Or, *hadamard, variant.guard)?;
                    }
                }
                let mut neighbors = zx
                    .neighbor_edges(node.id)
                    .map(|(neighbor, edge)| {
                        (
                            zx.nodes()[neighbor].pos.to_array(),
                            zx.edge_by_id(edge).hadamard,
                        )
                    })
                    .collect::<Vec<_>>();
                neighbors.sort_unstable();
                let key = (node.kind, node.role, neighbors);
                if let Some(&index) = local_keys.get(&key) {
                    let previous: &mut SurfaceTopology = &mut local_variants[index];
                    previous.guard =
                        topology
                            .diagram
                            .apply(BooleanOp::Or, previous.guard, variant.guard)?;
                } else {
                    local_keys.insert(key, local_variants.len());
                    local_variants.push(SurfaceTopology {
                        guard: variant.guard,
                        zx,
                        node: node.id,
                    });
                }
            }
            if local_variants.iter().any(|variant: &SurfaceTopology| {
                variant.zx.neighbor_edges(variant.node).next().is_none()
            }) {
                let next = columns.len();
                columns.entry(SurfaceColumn::Node(position)).or_insert(next);
            }
            groups[owner].extend(
                tensors
                    .into_iter()
                    .map(|(bits, active)| BooleanRow::from_terms(active, bits)),
            );
            local.insert(position, local_variants);
        }
        // Keep source tensor and seam row priority, but sweep physical columns
        // along the longest extent during later readout reduction.
        topology.diagram.charge(upper_bound)?;
        let mut axes = [0, 1, 2];
        if let Some((x, y, z)) = topology.source.spans() {
            let spans = [x, y, z];
            axes.sort_by_key(|&axis| {
                (
                    std::cmp::Reverse(
                        i64::from(*spans[axis].end()) - i64::from(*spans[axis].start()),
                    ),
                    axis,
                )
            });
        }
        topology.diagram.charge(
            columns
                .len()
                .saturating_mul(columns.len().max(1).ilog2() as usize + 1),
        )?;
        let mut ordered = columns.iter_mut().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(key, old)| {
            let position = match key {
                SurfaceColumn::Node(position) | SurfaceColumn::Edge(position, _) => position,
            };
            (axes.map(|axis| position[axis]), **old)
        });
        let mut remap = vec![0; ordered.len()];
        for (new, (_, old)) in ordered.into_iter().enumerate() {
            remap[*old] = new;
            *old = new;
        }
        let mut scope_order = scopes.lexical_order();
        for &scope in &scope_order {
            for row in &mut groups[scope] {
                let count = row.terms().len();
                topology
                    .diagram
                    .charge(count.saturating_mul(count.max(1).ilog2() as usize + 1))?;
                let active = row.active;
                let mut terms = std::mem::take(row).into_terms();
                for (column, _) in &mut terms {
                    *column = 2 * remap[*column / 2] + *column % 2;
                }
                terms.sort_unstable_by_key(|&(column, _)| column);
                *row = BooleanRow::from_terms(active, terms);
            }
        }
        // Stable sorting retains component-wise lexical order among peers.
        // Whole path strings differ here: "a__z" sorts after "a0__z".
        scope_order.sort_by_key(|&scope| std::cmp::Reverse(scopes.nodes[scope].depth));
        let mut column_rank = vec![0; columns.len()];
        for (rank, &column) in columns.values().enumerate() {
            column_rank[column] = rank;
        }
        let mut root = Vec::new();
        for owner in scope_order {
            let rows = std::mem::take(&mut groups[owner]);
            let mut edges = std::mem::take(&mut seams[owner])
                .into_iter()
                .collect::<Vec<_>>();
            // Close lower layers first, as in the ordinary module projector.
            edges.sort_by_key(|((a, b), _)| (a[2].max(b[2]), *a, *b));
            let seams = edges
                .into_iter()
                .map(|((left, right), hadamard)| {
                    let lhs = columns[&SurfaceColumn::Edge(left, right)];
                    let rhs = columns[&SurfaceColumn::Edge(right, left)];
                    (lhs, rhs, hadamard)
                })
                .collect::<Vec<_>>();
            let cache = relations.get(scopes.nodes[owner].path);
            let rows = if let Some(cache) = cache {
                cache.reduce(rows, seams, &column_rank, &mut topology.diagram, limits)?
            } else {
                crate::summary::relations::close_seams(rows, &seams, &mut topology.diagram, limits)?
            };
            if let Some(parent) = scopes.nodes[owner].parent {
                groups[parent].extend(rows);
            } else {
                root = rows;
            }
        }
        Ok(Self {
            topology,
            columns,
            local,
            rows: root,
            surfaces: Vec::new(),
            limits,
            module_sites,
            next_readout_collection: next_readout_collection(0, limits.max_witness_nodes),
        })
    }

    /// Freeze causal named readouts, then derive terminal corrections from that
    /// same source-outcome coordinate system. Earlier corrected parities remain
    /// explicit folds; their physical member rows are never expanded again.
    /// Consumes the working relation before sharing decision ids with physical
    /// lowering. The result retains local incidence instead of full ZX graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if causal readout planning or resource accounting fails.
    pub fn plan_readouts(mut self) -> Result<GuardedReadoutPlan, BlockGraphError> {
        self.plan_causal_readouts::<true, true>()?;
        self.into_readout_plan()
    }

    fn into_readout_plan(self) -> Result<GuardedReadoutPlan, BlockGraphError> {
        let selectors = self
            .topology
            .resolves
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let local = self
            .local
            .into_iter()
            .map(|(position, variants)| {
                let selector = selectors.get(&IVec3::from_array(position)).copied();
                let variants = variants
                    .into_iter()
                    .map(|variant| {
                        let arms: Box<[LocalArm]> = variant
                            .zx
                            .neighbor_edges(variant.node)
                            .map(|(neighbor, edge_id)| {
                                let target = variant.zx.nodes()[neighbor].pos;
                                LocalArm {
                                    column: self.columns
                                        [&SurfaceColumn::Edge(position, target.to_array())],
                                    target,
                                    reversed: variant.node > neighbor,
                                    hadamard: variant.zx.edge_by_id(edge_id).hadamard,
                                }
                            })
                            .collect();
                        LocalTopology {
                            guard: variant.guard,
                            kind: variant.zx.nodes()[variant.node].kind,
                            center: arms
                                .is_empty()
                                .then(|| self.columns[&SurfaceColumn::Node(position)]),
                            arms,
                            selector,
                        }
                    })
                    .collect();
                (position, variants)
            })
            .collect();
        let mut columns = vec![SurfaceColumn::Node([0; 3]); self.columns.len()];
        for (column, index) in self.columns {
            columns[index] = column;
        }
        let mut plan = GuardedReadoutPlan {
            topology: self.topology,
            surfaces: self.surfaces,
            columns,
            local,
            max_witness_nodes: self.limits.max_witness_nodes,
            query_cache: Default::default(),
            query_cache_terms: 0,
            fragments: Vec::new(),
            fragment_uses: Vec::new(),
        };
        plan.share_module_queries(&self.module_sites)?;
        Ok(plan)
    }

    // The unreclaimed instantiation is used only by the differential test.
    fn plan_causal_readouts<const RECLAIM: bool, const DEFER_WITNESSES: bool>(
        &mut self,
    ) -> Result<(), BlockGraphError> {
        self.plan_causal_readouts_with_fast_path::<RECLAIM, DEFER_WITNESSES, true>()
    }

    fn plan_causal_readouts_with_fast_path<
        const RECLAIM: bool,
        const DEFER_WITNESSES: bool,
        const FACTORED: bool,
    >(
        &mut self,
    ) -> Result<(), BlockGraphError> {
        let actions = self.topology.source.actions();
        let feedback_columns = self.index_feedbacks(&actions)?;
        let representative = self.topology.project(ONE)?;
        let zx = ZXGraph::from_block_graph_for_analysis(&representative)?;
        let mut names = Vec::new();
        let mut crossing = None;
        let mut basis = std::mem::take(&mut self.rows);
        let physical_columns = self.columns.len() * 2;
        let mut named_coordinates = HashMap::<usize, Vec<(usize, DecisionId)>>::new();
        for action in zx.action_graph().ordered_nodes() {
            self.topology.diagram.charge(1)?;
            let Action::Measure { name, target } = &action.action else {
                continue;
            };
            let axis = match action.measurement {
                // A Y cap's tensor equates its X and Z bits. Selective fills
                // likewise restrict support to I or the selected axis: use a
                // linear coordinate that is one on either permitted axis.
                Some(MeasurementObservable::Concrete(PauliBasis::X | PauliBasis::Y))
                | Some(MeasurementObservable::Selective(crate::SelectiveKind::XY)) => Pauli::X,
                Some(MeasurementObservable::Concrete(PauliBasis::Z))
                | Some(MeasurementObservable::Selective(crate::SelectiveKind::YZ)) => Pauli::Z,
                Some(MeasurementObservable::Selective(crate::SelectiveKind::XZ)) => Pauli::Y,
                None => {
                    return Err(StabilizerError::MeasurementSurfaceUnavailable {
                        mvar: name.clone(),
                    }
                    .into());
                }
            };
            if let MeasureTarget::Node(pos) = target {
                let variants = &self.local[&pos.to_array()];
                self.topology.diagram.charge(variants.len())?;
                if variants.iter().any(|variant| {
                    matches!(
                        variant.zx.nodes()[variant.node].kind,
                        NodeKind::X | NodeKind::Z
                    ) && variant.zx.nodes()[variant.node].kind.cross_pauli() == axis
                }) {
                    crossing = Some(name.clone());
                }
            }
            let target = self.measurement_column(*target, &zx)?;
            let coordinate = physical_columns + names.len();
            for axis in axis.iter_xz() {
                for (column, weight) in self.component_terms(target, axis) {
                    named_coordinates
                        .entry(column)
                        .or_default()
                        .push((coordinate, weight));
                }
            }
            names.push((name.clone(), coordinate));
        }
        for row in &mut basis {
            self.topology.diagram.charge(row.terms().len())?;
            let mut values = BTreeMap::new();
            for &(column, bit) in row.terms() {
                if let Some(coordinates) = named_coordinates.get(&column) {
                    self.topology.diagram.charge(coordinates.len())?;
                    for &(coordinate, weight) in coordinates {
                        let value = values.entry(coordinate).or_insert(ZERO);
                        let bit = self.topology.diagram.apply(BooleanOp::And, bit, weight)?;
                        *value = self.topology.diagram.apply(BooleanOp::Xor, *value, bit)?;
                    }
                }
            }
            for (coordinate, value) in values {
                row.set(coordinate, value);
            }
        }
        if let Some(name) = crossing {
            basis = self.crossing_readout_coordinates(basis, &zx, &names, &name)?;
        }
        let all_resolved = self
            .topology
            .resolves
            .iter()
            .map(|(position, _)| position.to_array())
            .collect();
        // Output exclusion is permanent. Selective fills accumulate as their
        // controls become known; a previously unknown site was required to be
        // identity, which already satisfies either future fill. Keep the source
        // relation separately for terminal and remaining logical readouts.
        let mut causal_basis = if names.is_empty() {
            Vec::new()
        } else {
            self.charge_row_copy(&basis)?;
            let mut rows = basis.clone();
            self.exclude_outputs(&mut rows, &zx)?;
            if RECLAIM {
                // An unresolved site is later restricted to identity, which
                // already satisfies either selector-dependent fill. Apply all
                // fills once; subsequent availability constraints only relax.
                self.fill_selective(&mut rows, &all_resolved)?;
            }
            rows
        };
        self.topology.diagram.charge(self.columns.len())?;
        let mut columns_by_site = HashMap::<[i32; 3], Vec<usize>>::new();
        for (&column, &index) in &self.columns {
            let position = match column {
                SurfaceColumn::Node(pos) | SurfaceColumn::Edge(pos, _) => pos,
            };
            columns_by_site.entry(position).or_default().push(index);
        }
        let names_set = names
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();
        let mut known = self
            .topology
            .variables
            .iter()
            .filter_map(|variable| match variable {
                crate::GuardedVariable::Outcome(name) if !names_set.contains(name) => {
                    Some(name.clone())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut availability: Option<GuardedAvailability> = None;
        let mut available_values = None;
        let mut pending_resolves = Vec::new();
        let mut pending_sites = Vec::new();
        let mut pending_feedbacks = Vec::new();
        let mut available_rows = BooleanRowSpace::default();
        let mut readout_pivots = GuardedPivots::default();
        while names.iter().any(|(name, _)| !known.contains(name)) {
            if available_values.is_none() {
                self.topology.diagram.charge(
                    self.topology
                        .resolves
                        .len()
                        .saturating_add(self.local.len())
                        .saturating_add(actions.len()),
                )?;
                pending_resolves.extend(0..self.topology.resolves.len());
                pending_sites.extend(
                    self.local.iter().filter_map(|(&position, variants)| {
                        (variants.len() > 1).then_some(position)
                    }),
                );
                pending_feedbacks.extend(actions.iter().enumerate().filter_map(
                    |(index, action)| {
                        matches!(
                            action,
                            Action::Feedback {
                                condition: Some(_),
                                ..
                            }
                        )
                        .then_some(index)
                    },
                ));
                available_values = Some(self.topology.availability(&known)?);
            }
            let available_values = available_values.as_mut().expect("availability initialized");
            self.topology
                .refresh_availability(available_values, &known)?;
            self.topology
                .diagram
                .charge(pending_resolves.len() + pending_sites.len() + pending_feedbacks.len())?;
            let mut newly_resolved = BTreeSet::new();
            let mut pending_count = 0;
            for index in 0..pending_resolves.len() {
                let ordinal = pending_resolves[index];
                let (position, root) = self.topology.resolves[ordinal];
                if available_values.available(root, &mut self.topology.diagram)? {
                    if !RECLAIM {
                        newly_resolved.insert(position.to_array());
                    }
                } else {
                    pending_resolves[pending_count] = ordinal;
                    pending_count += 1;
                }
            }
            pending_resolves.truncate(pending_count);
            let mut pending_count = 0;
            for index in 0..pending_sites.len() {
                let position = pending_sites[index];
                let mut ready = true;
                for variant in &self.local[&position] {
                    if !available_values.available(variant.guard, &mut self.topology.diagram)? {
                        ready = false;
                        break;
                    }
                }
                if !ready {
                    pending_sites[pending_count] = position;
                    pending_count += 1;
                }
            }
            pending_sites.truncate(pending_count);
            let unavailable = pending_sites
                .iter()
                .copied()
                .chain(
                    pending_resolves
                        .iter()
                        .map(|&index| self.topology.resolves[index].0.to_array()),
                )
                .collect::<BTreeSet<_>>();
            let mut pending_count = 0;
            for index in 0..pending_feedbacks.len() {
                let ordinal = pending_feedbacks[index];
                let Action::Feedback {
                    condition: Some(condition),
                    ..
                } = &actions[ordinal]
                else {
                    unreachable!("pending feedbacks have a condition")
                };
                let root = self.topology.condition(condition);
                if !available_values.available(root, &mut self.topology.diagram)? {
                    pending_feedbacks[pending_count] = ordinal;
                    pending_count += 1;
                }
            }
            pending_feedbacks.truncate(pending_count);
            let feedback_absent = &pending_feedbacks;
            if !RECLAIM {
                self.fill_selective(&mut causal_basis, &newly_resolved)?;
            }
            let pending = unavailable
                .iter()
                .copied()
                .map(AvailabilityKey::Site)
                .chain(
                    feedback_absent
                        .iter()
                        .copied()
                        .map(AvailabilityKey::Feedback),
                )
                .collect::<BTreeSet<_>>();
            let mut constraints = Vec::new();
            if !RECLAIM || availability.is_none() {
                let mut blocked_columns = Vec::new();
                for position in unavailable {
                    let columns = &columns_by_site[&position];
                    self.topology.diagram.charge(columns.len())?;
                    blocked_columns.extend(columns.iter().map(|&index| (index, position)));
                }
                blocked_columns.sort_unstable_by_key(|&(index, _)| index);
                for (index, position) in blocked_columns {
                    for axis in 0..2 {
                        // Multiplex permits Z on its live output, but its
                        // physical arms still require known local geometry.
                        constraints.push((
                            AvailabilityKey::Site(position),
                            vec![(2 * index + axis, ONE)],
                        ));
                    }
                }
                for &ordinal in feedback_absent {
                    let Action::Feedback { targets, .. } = &actions[ordinal] else {
                        unreachable!("selector columns name selective nodes")
                    };
                    self.topology.diagram.charge(targets.len())?;
                    let mut terms = Vec::new();
                    for target in targets {
                        self.topology
                            .diagram
                            .charge(self.local[&target.target.to_array()].len())?;
                        for variant in &self.local[&target.target.to_array()] {
                            for index in self.feedback_support(target, variant)? {
                                terms.push((index, variant.guard));
                            }
                        }
                    }
                    // A source Pauli product can commute through even target
                    // overlaps. Preserve every occurrence in its one linear
                    // functional instead of blocking each component alone.
                    constraints.push((AvailabilityKey::Feedback(ordinal), terms));
                }
            }
            if RECLAIM {
                if let Some(availability) = &mut availability {
                    availability.release(
                        &pending,
                        &readout_pivots,
                        &mut available_rows,
                        &mut self.topology.diagram,
                    )?;
                } else {
                    let (state, rows) = GuardedAvailability::new(
                        std::mem::take(&mut causal_basis),
                        &constraints,
                        physical_columns + names.len(),
                        &mut self.topology.diagram,
                    )?;
                    availability = Some(state);
                    available_rows = rows;
                }
            }
            let unknown = names
                .iter()
                .filter(|(name, _)| !known.contains(name))
                .collect::<Vec<_>>();
            let mut unknown_columns = vec![false; names.len()];
            for (_, column) in &unknown {
                unknown_columns[*column - physical_columns] = true;
            }
            let relevant = if RECLAIM && FACTORED {
                Some(self.named_component_rows(&available_rows, &unknown)?)
            } else {
                None
            };
            let mut factored = if let Some(ids) = &relevant {
                self.injective_named_candidates(
                    &available_rows,
                    ids,
                    &unknown,
                    physical_columns,
                    physical_columns + names.len(),
                )?
            } else {
                None
            };
            let reduced = if factored.is_some() {
                BooleanRowSpace::default()
            } else {
                let allowed = if RECLAIM {
                    let rows = if let Some(ids) = &relevant {
                        let mut rows = Vec::with_capacity(ids.len());
                        for &id in ids {
                            let row = available_rows.row(id).expect("selected row exists");
                            self.topology.diagram.charge(row.terms().len() + 1)?;
                            rows.push(row.clone());
                        }
                        rows
                    } else {
                        self.topology.diagram.charge(available_rows.len())?;
                        for row in available_rows.rows() {
                            self.topology.diagram.charge(row.terms().len())?;
                        }
                        available_rows.rows().cloned().collect()
                    };
                    BooleanRowSpace::new(rows, &mut self.topology.diagram)?
                } else {
                    self.charge_row_copy(&causal_basis)?;
                    let mut allowed =
                        BooleanRowSpace::new(causal_basis.clone(), &mut self.topology.diagram)?;
                    for (_, terms) in constraints {
                        allowed.eliminate_linear(&terms, &mut self.topology.diagram)?;
                    }
                    allowed
                };
                // Keep only named pivots. Physical kernel pivots normalize
                // those rows, but their own forms are never used here.
                self.reduce_named_rows::<RECLAIM>(
                    allowed,
                    unknown.iter().map(|(_, column)| *column),
                )?
            };
            let mut candidates = Vec::new();
            for (index, (name, wanted)) in unknown.iter().enumerate() {
                self.topology.diagram.charge(1)?;
                let row = if let Some(factored) = &mut factored {
                    let Some(row) = factored[index].take() else {
                        continue;
                    };
                    row
                } else {
                    self.named_row(&reduced, *wanted)?
                };
                let named =
                    Self::named_terms(&row, physical_columns, physical_columns + names.len());
                if row.active != ONE
                    || named.iter().any(|&(column, _)| {
                        column != *wanted && unknown_columns[column - physical_columns]
                    })
                {
                    continue;
                }
                let mut ready = true;
                for &(_, bit) in row.terms() {
                    if !available_values.available(bit, &mut self.topology.diagram)? {
                        ready = false;
                        break;
                    }
                }
                if !ready {
                    continue;
                }
                let folds = named
                    .iter()
                    .filter(|&&(column, _)| !unknown_columns[column - physical_columns])
                    .map(|&(column, bit)| (names[column - physical_columns].0.clone(), bit))
                    .collect();
                let feedbacks = self.feedbacks::<RECLAIM>(&row, &actions, &feedback_columns)?;
                candidates.push((
                    name.clone(),
                    GuardedSurface {
                        kind: GuardedSurfaceKind::Readout {
                            name: name.clone(),
                            folds,
                        },
                        feedbacks,
                        row,
                    },
                ));
            }
            // The final budget check may remap the diagram. Query images and
            // unselected local candidate rows cannot retain unlisted roots.
            drop((factored, reduced));
            if candidates.is_empty() {
                return Err(StabilizerError::UnavailableControlParity {
                    mvar: unknown[0].0.clone(),
                    deadline: i64::MAX,
                }
                .into());
            }
            available_values.learn(
                candidates.iter().map(|(name, _)| name.as_str()),
                &mut self.topology.diagram,
            )?;
            let reclaim = RECLAIM && candidates.len() < unknown.len();
            let mut reclaimed = Vec::new();
            for (name, surface) in candidates {
                known.insert(name);
                if reclaim {
                    self.topology.diagram.charge(surface.row.terms().len())?;
                    let mut row = surface.row.clone();
                    // Candidate coefficients were simplified on the source
                    // domain. They represent a source relation only there.
                    row.active = self.topology.domain;
                    reclaimed.push(row);
                }
                self.surfaces.push(surface);
            }
            if reclaim {
                self.reclaim_readouts(&mut available_rows, &mut readout_pivots, reclaimed)?;
            }
            self.check_readout_roots(
                basis
                    .iter_mut()
                    .chain(causal_basis.iter_mut())
                    .flat_map(BooleanRow::decisions_mut)
                    .chain(available_rows.decisions_mut())
                    .chain(readout_pivots.rows.decisions_mut())
                    .chain(
                        availability
                            .iter_mut()
                            .flat_map(|state| state.pivots.rows.decisions_mut()),
                    ),
            )?;
        }
        drop((
            causal_basis,
            availability,
            available_rows,
            readout_pivots,
            available_values,
        ));
        let (input_nodes, resource_nodes) = authored_boundary_columns(&zx);
        let mut output_ports = zx.output_ports();
        if DEFER_WITNESSES {
            // Stable terminal choices inspect only these explicit constraints.
            // Reconstruct raw physical support before feedback and final kernel
            // reduction; a zero-core private row must remain live until then.
            self.topology.diagram.charge(
                zx.nodes()
                    .len()
                    .saturating_add(self.topology.resolves.len()),
            )?;
            // Output pivots and selective fills never query input or protected
            // resource columns. Keep their exact payload on the witness tape
            // until input normalization actually needs it.
            let positions = output_ports
                .iter()
                .copied()
                .chain(
                    zx.nodes()
                        .iter()
                        .filter(|node| node.role == crate::PortRole::Multiplex)
                        .map(|node| node.pos),
                )
                .chain(self.topology.resolves.iter().map(|&(position, _)| position));
            let core = positions
                .flat_map(|position| {
                    self.node_components(position.to_array())
                        .into_iter()
                        .flatten()
                        .map(|(column, _)| column)
                })
                .chain(names.iter().map(|(_, column)| *column))
                .collect::<Vec<_>>();
            self.topology
                .diagram
                .start_witness_tape(core, self.limits.max_witness_nodes)?;
            for row in &mut basis {
                row.defer_coefficients(&mut self.topology.diagram)?;
            }
        }
        // Remove the named coordinates before resolving terminal measurements.
        // This is the kernel used by RuntimeStabilizerBasis::for_output_frames.
        let mut basis = BooleanRowSpace::new(basis, &mut self.topology.diagram)?;
        for (_, coordinate) in &names {
            basis.eliminate_column(*coordinate, &mut self.topology.diagram)?;
        }
        self.check_readout_roots(basis.decisions_mut())?;
        self.topology.diagram.charge(self.topology.resolves.len())?;
        for index in 0..self.topology.resolves.len() {
            self.fill_selective_site(&mut basis, index)?;
            // Terminal planning owns every live row here. Reclaim temporary
            // fill products before the next site can exhaust the arena.
            self.check_readout_roots(basis.decisions_mut())?;
        }
        self.check_readout_roots(basis.decisions_mut())?;
        let mut output_rows = BooleanRowSpace::default();
        let mut output_axes = Vec::new();
        let mut output_gates = HashMap::new();
        if RECLAIM {
            // Output order may change the chosen free axes under conditional
            // rank. Preserve the authored pivot activations using only the
            // small output projection, then reorder the physical witnesses.
            let mut projections = Vec::new();
            for &port in &output_ports {
                let axes = if zx
                    .node_at(port)
                    .expect("output ports come from this ZX graph")
                    .role
                    == crate::PortRole::Multiplex
                {
                    1
                } else {
                    2
                };
                projections.extend(
                    [Pauli::X, Pauli::Z]
                        .into_iter()
                        .take(axes)
                        .map(|axis| (port, axis)),
                );
            }
            let output_columns = projections
                .iter()
                .flat_map(|&(port, axis)| {
                    self.component_terms(SurfaceColumn::Node(port.to_array()), axis)
                        .into_iter()
                        .map(|(column, _)| column)
                })
                .collect::<BTreeSet<_>>();
            let mut projected = BooleanRowSpace::default();
            for row in basis.rows() {
                self.topology.diagram.charge(row.terms().len())?;
                projected.push(
                    BooleanRow::from_terms(
                        row.active,
                        row.terms()
                            .iter()
                            .copied()
                            .filter(|(column, _)| output_columns.contains(column)),
                    ),
                    &mut self.topology.diagram,
                )?;
            }
            for (port, axis) in projections {
                let terms = self.component_terms(SurfaceColumn::Node(port.to_array()), axis);
                let pivot = projected.eliminate_linear(&terms, &mut self.topology.diagram)?;
                output_gates.insert((port, axis), pivot.active);
                self.check_readout_roots(
                    basis
                        .decisions_mut()
                        .chain(projected.decisions_mut())
                        .chain(output_gates.values_mut()),
                )?;
            }
            self.topology.diagram.charge(
                output_ports
                    .len()
                    .saturating_mul(output_ports.len().max(1).ilog2() as usize + 1),
            )?;
            output_ports.sort_unstable_by_key(|port| {
                std::cmp::Reverse(
                    self.node_components(port.to_array())
                        .into_iter()
                        .flatten()
                        .map(|(column, _)| column)
                        .min(),
                )
            });
        }
        for port in output_ports {
            for (axis, component) in [Pauli::X, Pauli::Z].into_iter().enumerate() {
                if axis == 1
                    && zx
                        .node_at(port)
                        .expect("output ports come from this ZX graph")
                        .role
                        == crate::PortRole::Multiplex
                {
                    output_axes.push((port, axis, false));
                    continue;
                }
                let gate = output_gates.get(&(port, component)).copied().unwrap_or(ONE);
                let terms = self.component_terms(SurfaceColumn::Node(port.to_array()), component);
                let gated = terms
                    .iter()
                    .map(|&(column, weight)| {
                        Ok((
                            column,
                            self.topology.diagram.apply(BooleanOp::And, weight, gate)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, BooleanResourceError>>()?;
                let pivot = basis.eliminate_linear(&gated, &mut self.topology.diagram)?;
                // Keep the physical witnesses triangular. Expanding each
                // earlier row through every later pivot can make its Boolean
                // transport coefficients exponential. The emitted frame DAG
                // performs this same back-substitution on decoded residuals.
                if !RECLAIM {
                    output_rows.reduce_linear(&terms, &pivot, &mut self.topology.diagram)?;
                }
                output_axes.push((port, axis, pivot.active != ZERO));
                output_rows.push(pivot, &mut self.topology.diagram)?;
                drop((terms, gated));
                self.check_readout_roots(
                    basis
                        .decisions_mut()
                        .chain(output_rows.decisions_mut())
                        .chain(output_gates.values_mut()),
                )?;
            }
        }
        drop(output_gates);
        let (inputs, resources) = (input_nodes, resource_nodes);
        let components = |node: usize| {
            [Pauli::X, Pauli::Z]
                .map(|axis| (SurfaceColumn::Node(zx.nodes()[node].pos.to_array()), axis))
        };
        let inputs = inputs.into_iter().flat_map(components).collect::<Vec<_>>();
        // Source T resources and public Ports cannot belong to optional arms.
        // Multiplex Z was deliberately not pivoted as an output above.
        let protected = resources
            .into_iter()
            .flat_map(components)
            .chain(
                zx.nodes()
                    .iter()
                    .filter(|node| node.role == crate::PortRole::Multiplex)
                    .map(|node| components(node.id)[1]),
            )
            .collect::<Vec<_>>();
        let mut basis = basis.into_rows(&mut self.topology.diagram)?;
        if DEFER_WITNESSES && !basis.is_empty() {
            // The global kernel needs input and protected-resource projections,
            // not every physical coefficient. Keep the same source-ordered solve
            // while its row operations compose the remaining passive transfers.
            self.topology
                .diagram
                .charge(inputs.len().saturating_add(protected.len()))?;
            let mut front = Vec::new();
            for &(column, axis) in inputs.iter().chain(&protected) {
                let terms = self.component_terms(column, axis);
                self.topology.diagram.charge(terms.len())?;
                front.extend(terms.into_iter().map(|(column, _)| column));
            }
            let mut outputs = output_rows.into_rows(&mut self.topology.diagram)?;
            self.topology
                .diagram
                .charge(basis.len().saturating_add(outputs.len()).saturating_mul(2))?;
            let remaining = basis.len();
            basis.append(&mut outputs);
            self.topology
                .diagram
                .extend_witness_front(&front, &mut basis)?;
            outputs = basis.split_off(remaining);
            output_rows = BooleanRowSpace::new(outputs, &mut self.topology.diagram)?;
            self.check_readout_roots(
                basis
                    .iter_mut()
                    .flat_map(BooleanRow::decisions_mut)
                    .chain(output_rows.decisions_mut()),
            )?;
        }
        self.normalize_output_inputs(&mut basis, &mut output_rows, &inputs, &protected)?;
        let mut output_rows = output_rows.into_rows(&mut self.topology.diagram)?;
        if DEFER_WITNESSES {
            for index in 0..basis.len() {
                basis[index].expand_coefficients(&mut self.topology.diagram)?;
                self.check_readout_roots(
                    basis
                        .iter_mut()
                        .chain(output_rows.iter_mut())
                        .flat_map(BooleanRow::decisions_mut),
                )?;
            }
            for index in 0..output_rows.len() {
                output_rows[index].expand_coefficients(&mut self.topology.diagram)?;
                self.check_readout_roots(
                    basis
                        .iter_mut()
                        .chain(output_rows.iter_mut())
                        .flat_map(BooleanRow::decisions_mut),
                )?;
            }
            self.topology.diagram.finish_witness_tape();
            self.check_readout_roots(
                basis
                    .iter_mut()
                    .chain(output_rows.iter_mut())
                    .flat_map(BooleanRow::decisions_mut),
            )?;
        }
        let output_columns = if RECLAIM {
            self.index_output_axes(&output_axes)?
        } else {
            BTreeMap::new()
        };
        let mut output_rows = output_rows.into_iter();
        for (index, &(port, axis, active)) in output_axes.iter().enumerate() {
            let row = if active {
                output_rows
                    .next()
                    .expect("each active output retains its row")
            } else {
                BooleanRow::new(ZERO)
            };
            let folds = if RECLAIM && active {
                self.output_folds(&row, index, &output_axes, &output_columns)?
            } else {
                Vec::new()
            };
            let feedbacks = self.feedbacks::<RECLAIM>(&row, &actions, &feedback_columns)?;
            self.surfaces.push(GuardedSurface {
                kind: GuardedSurfaceKind::OutputFrame {
                    port,
                    basis: if axis == 0 {
                        PauliBasis::Z
                    } else {
                        PauliBasis::X
                    },
                    folds,
                },
                feedbacks,
                row,
            });
        }
        drop(output_columns);
        let input_columns = inputs
            .iter()
            .flat_map(|&(column, axis)| {
                self.component_terms(column, axis)
                    .into_iter()
                    .map(|(column, _)| column)
            })
            .collect::<Vec<_>>();
        let remaining = reduce_boolean_rows(
            basis,
            input_columns.into_iter().chain(0..physical_columns),
            &mut self.topology.diagram,
        )?;
        // The kernel also contains closed logical surfaces: memory and
        // stability readouts have no input or output Port support.
        for row in remaining {
            let feedbacks = self.feedbacks::<RECLAIM>(&row, &actions, &feedback_columns)?;
            self.surfaces.push(GuardedSurface {
                kind: GuardedSurfaceKind::LogicalReadout,
                feedbacks,
                row,
            });
        }
        // Physical lowering retains decision ids in its own caches. Finish the
        // planning stage with only live functions before those ids escape.
        self.collect_readout_roots(std::iter::empty())?;
        self.check_budget()
    }

    /// Normalize erased input operators within the selected resource class.
    /// Only zero-output relations that preserve prepared T resources and the
    /// unpivoted Multiplex Z direction can change the input representative.
    /// The scalar planner uses this same restricted input normalization.
    fn normalize_output_inputs(
        &mut self,
        remaining: &mut [BooleanRow],
        outputs: &mut BooleanRowSpace,
        inputs: &[(SurfaceColumn, Pauli)],
        protected: &[(SurfaceColumn, Pauli)],
    ) -> Result<(), BlockGraphError> {
        if outputs.is_empty() || inputs.is_empty() {
            return Ok(());
        }
        self.topology.diagram.charge(inputs.len())?;
        let input_columns = inputs
            .iter()
            .flat_map(|&(column, axis)| {
                self.component_terms(column, axis)
                    .into_iter()
                    .map(|(column, _)| column)
            })
            .collect::<BTreeSet<_>>();
        let mut input_support = false;
        for row in remaining.iter() {
            self.topology.diagram.charge(row.terms().len())?;
            if row
                .terms()
                .iter()
                .any(|&(column, _)| input_columns.contains(&column))
            {
                input_support = true;
                break;
            }
        }
        if !input_support {
            return Ok(());
        }
        self.charge_row_copy(remaining)?;
        let mut kernel = BooleanRowSpace::new(remaining.to_vec(), &mut self.topology.diagram)?;
        for &(column, axis) in protected {
            let terms = self.component_terms(column, axis);
            kernel.eliminate_linear(&terms, &mut self.topology.diagram)?;
            self.check_readout_roots(
                remaining
                    .iter_mut()
                    .flat_map(BooleanRow::decisions_mut)
                    .chain(outputs.decisions_mut())
                    .chain(kernel.decisions_mut()),
            )?;
        }
        for &(column, axis) in inputs {
            let terms = self.component_terms(column, axis);
            let pivot = kernel.eliminate_linear(&terms, &mut self.topology.diagram)?;
            outputs.reduce_linear(&terms, &pivot, &mut self.topology.diagram)?;
            self.check_readout_roots(
                remaining
                    .iter_mut()
                    .flat_map(BooleanRow::decisions_mut)
                    .chain(outputs.decisions_mut())
                    .chain(kernel.decisions_mut()),
            )?;
        }
        Ok(())
    }

    fn charge_row_copy(&mut self, rows: &[BooleanRow]) -> Result<(), BooleanResourceError> {
        self.topology.diagram.charge(rows.len())?;
        for row in rows {
            self.topology.diagram.charge(row.terms().len())?;
        }
        Ok(())
    }

    fn exclude_outputs(
        &mut self,
        rows: &mut Vec<BooleanRow>,
        zx: &ZXGraph,
    ) -> Result<(), BooleanResourceError> {
        let mut remaining = BooleanRowSpace::new(std::mem::take(rows), &mut self.topology.diagram)?;
        for port in zx.output_ports() {
            let axes = if zx
                .node_at(port)
                .expect("output ports come from this ZX graph")
                .role
                == crate::PortRole::Multiplex
            {
                1
            } else {
                2
            };
            for terms in self.node_components(port.to_array()).iter().take(axes) {
                remaining.eliminate_linear(terms, &mut self.topology.diagram)?;
            }
        }
        *rows = remaining.into_rows(&mut self.topology.diagram)?;
        Ok(())
    }

    fn reduce_named_rows<const SPARSE: bool>(
        &mut self,
        mut remaining: BooleanRowSpace,
        names: impl Iterator<Item = usize>,
    ) -> Result<BooleanRowSpace, BooleanResourceError> {
        let mut named = BooleanRowSpace::default();
        for column in names {
            let pivot = if SPARSE {
                remaining.eliminate_column_sparse(column, &mut self.topology.diagram)?
            } else {
                remaining.eliminate_column(column, &mut self.topology.diagram)?
            };
            if pivot.active != ZERO {
                named.reduce_column(column, &pivot, &mut self.topology.diagram)?;
                named.push(pivot, &mut self.topology.diagram)?;
            }
        }
        // Unknown named columns are gone. Physical coordinates precede the
        // remaining, known-name coordinates numerically, so sorted support has
        // exactly the original physical-then-known priority without a scan of
        // the enclosing graph's entire column range.
        self.topology.diagram.charge(remaining.columns().len())?;
        let mut physical_and_known = remaining.columns().collect::<Vec<_>>();
        physical_and_known.sort_unstable();
        for column in physical_and_known {
            if remaining.is_empty() {
                break;
            }
            let pivot = if SPARSE {
                remaining.eliminate_column_sparse(column, &mut self.topology.diagram)?
            } else {
                remaining.eliminate_column(column, &mut self.topology.diagram)?
            };
            named.reduce_column(column, &pivot, &mut self.topology.diagram)?;
        }
        Ok(named)
    }

    /// Select algebraic components after constraints have coupled their rows.
    /// Keep every physical and known-name column, including private kernels;
    /// only the temporary named solve omits components with no unknown name.
    fn named_component_rows(
        &mut self,
        available: &BooleanRowSpace,
        unknown: &[&(String, usize)],
    ) -> Result<Vec<usize>, BooleanResourceError> {
        let mut columns = HashSet::new();
        let mut rows = HashSet::new();
        let mut pending = Vec::new();
        // Seed every name first so injective spaces need no physical walk.
        for (_, column) in unknown {
            self.topology.diagram.charge(1)?;
            columns.insert(*column);
            for id in available.row_ids_with_column(*column) {
                self.topology.diagram.charge(1)?;
                if rows.insert(id) {
                    pending.push(id);
                }
            }
        }
        while rows.len() < available.len() {
            let Some(id) = pending.pop() else { break };
            let row = available.row(id).expect("indexed row exists");
            self.topology.diagram.charge(row.terms().len())?;
            for &(column, _) in row.terms() {
                if columns.insert(column) {
                    // Removed postings have zero support on their activation.
                    for id in available.row_ids_with_column(column) {
                        self.topology.diagram.charge(1)?;
                        if rows.insert(id) {
                            pending.push(id);
                        }
                    }
                }
            }
        }
        self.topology.diagram.charge(
            rows.len()
                .saturating_mul(rows.len().max(1).ilog2() as usize + 1),
        )?;
        let mut rows = rows.into_iter().collect::<Vec<_>>();
        rows.sort_unstable();
        Ok(rows)
    }

    fn named_terms(row: &BooleanRow, first: usize, end: usize) -> &[(usize, DecisionId)] {
        let terms = row.terms();
        &terms[terms.partition_point(|&(column, _)| column < first)
            ..terms.partition_point(|&(column, _)| column < end)]
    }

    /// Try a named-only reduction. Private tags record each source row's exact
    /// combination. An empty residual certifies that named projection is
    /// injective, so each full witness is the unique lift of its named row.
    /// Restricted source domains retain the full reducer until their
    /// off-domain coefficient convention is factored.
    fn injective_named_candidates(
        &mut self,
        available: &BooleanRowSpace,
        source_ids: &[usize],
        unknown: &[&(String, usize)],
        physical_columns: usize,
        first_tag: usize,
    ) -> Result<Option<Vec<Option<BooleanRow>>>, BooleanResourceError> {
        if self.topology.domain != ONE {
            return Ok(None);
        }
        self.topology
            .diagram
            .charge(source_ids.len() + first_tag - physical_columns)?;
        let mut unknown_columns = vec![false; first_tag - physical_columns];
        for (_, column) in unknown {
            unknown_columns[*column - physical_columns] = true;
        }
        let mut skeleton = Vec::with_capacity(source_ids.len());
        for &id in source_ids {
            let source = available.row(id).expect("selected row exists");
            let terms = Self::named_terms(source, physical_columns, first_tag);
            self.topology.diagram.charge(terms.len() + 1)?;
            let mut row = BooleanRow::from_terms(source.active, terms.iter().copied());
            row.set(first_tag + id, ONE);
            skeleton.push(row);
        }
        let mut remaining = BooleanRowSpace::new(skeleton, &mut self.topology.diagram)?;
        let mut named = BooleanRowSpace::default();
        for (_, column) in unknown {
            let pivot = remaining.eliminate_column_sparse(*column, &mut self.topology.diagram)?;
            if pivot.active != ZERO {
                named.reduce_column(*column, &pivot, &mut self.topology.diagram)?;
                named.push(pivot, &mut self.topology.diagram)?;
            }
        }
        if !remaining.is_empty() {
            return Ok(None);
        }
        let mut result = Vec::with_capacity(unknown.len());
        for (_, wanted) in unknown {
            let row = self.named_row(&named, *wanted)?;
            if row.active != ONE
                || Self::named_terms(&row, physical_columns, first_tag)
                    .iter()
                    .any(|&(column, _)| {
                        column != *wanted && unknown_columns[column - physical_columns]
                    })
            {
                result.push(None);
                continue;
            }
            let diagram = &mut self.topology.diagram;
            let mut full = BooleanRow::new(ONE);
            diagram.charge(row.terms().len())?;
            for &(column, scale) in row.terms() {
                if column >= first_tag {
                    full.xor_scaled(
                        available
                            .row(column - first_tag)
                            .expect("tagged row exists"),
                        scale,
                        diagram,
                    )?;
                }
            }
            // Exact lifts must retain raw named coefficients, including off-domain values.
            let full_named = Self::named_terms(&full, physical_columns, first_tag);
            let expected_named = Self::named_terms(&row, physical_columns, first_tag);
            diagram.charge(full_named.len() + expected_named.len())?;
            if full_named != expected_named {
                return Ok(None);
            }
            result.push(Some(full));
        }
        Ok(Some(result))
    }

    fn named_row(
        &mut self,
        named: &BooleanRowSpace,
        wanted: usize,
    ) -> Result<BooleanRow, BooleanResourceError> {
        let mut rows = Vec::new();
        for row in named.rows_with_column(wanted) {
            self.topology
                .diagram
                .charge(row.terms().len().saturating_add(1))?;
            rows.push(row.clone());
        }
        let mut row = eliminate_boolean_rows(&mut rows, &mut self.topology.diagram, |row, _| {
            Ok(row.get(wanted))
        })?;
        row.active = self
            .topology
            .diagram
            .constrain(row.active, self.topology.domain)?;
        self.topology.diagram.charge(row.terms().len())?;
        row.map_coefficients(|coefficient| {
            self.topology
                .diagram
                .constrain(coefficient, self.topology.domain)
        })?;
        Ok(row)
    }

    fn reclaim_readouts(
        &mut self,
        remaining: &mut BooleanRowSpace,
        pivots: &mut GuardedPivots,
        readouts: Vec<BooleanRow>,
    ) -> Result<(), BooleanResourceError> {
        // Each accepted relation satisfies every later availability constraint.
        // Quotienting along its physical-then-known-name pivots therefore
        // commutes with those constraints. Retain its row to project witnesses
        // promoted by future constraint releases into the same named folds.
        for mut row in readouts {
            pivots.project(&mut row, &mut self.topology.diagram)?;
            self.topology.diagram.charge(row.terms().len())?;
            let columns = row
                .terms()
                .iter()
                .map(|&(column, _)| column)
                .collect::<Vec<_>>();
            // Only forward substitution consumes this basis; later insertions
            // need not rewrite older pivots.
            let (_, added) = pivots.insert_residual::<false>(
                row,
                columns.into_iter(),
                &mut self.topology.diagram,
            )?;
            for (column, active) in added {
                remaining.eliminate_column_when_sparse(
                    column,
                    active,
                    &mut self.topology.diagram,
                )?;
            }
        }
        Ok(())
    }

    /// A crossing centre is an OR of arm support, not a linear Pauli column.
    /// Select independent readable representatives from the composed relation,
    /// then use their dual coordinates through the same causal planner.
    fn crossing_readout_coordinates(
        &mut self,
        mut rows: Vec<BooleanRow>,
        zx: &ZXGraph,
        names: &[(String, usize)],
        crossing: &str,
    ) -> Result<Vec<BooleanRow>, BlockGraphError> {
        let physical_columns = self.columns.len() * 2;
        for row in &mut rows {
            row.truncate_columns(physical_columns);
        }
        // The existing affine chooser handles a constant correlation table.
        // Conditional crossing presentations still require a symbolic chooser;
        // never replace their coefficients with one representative branch.
        if self.local.values().any(|variants| variants.len() != 1)
            || rows
                .iter()
                .any(|row| row.active != ONE || row.terms().iter().any(|&(_, bit)| bit != ONE))
        {
            return Err(StabilizerError::MeasurementSurfaceUnavailable {
                mvar: crossing.to_owned(),
            }
            .into());
        }
        rows = reduce_boolean_rows(rows, 0..physical_columns, &mut self.topology.diagram)?;
        zx.check_external_table_limits(self.limits)?;
        let raw = rows
            .iter()
            .map(|row| self.materialize_row(row, zx, |bit| bit == ONE))
            .collect::<Vec<_>>();
        let phases = zx.stabilizer_row_phases(&raw);
        let signed = raw
            .into_iter()
            .zip(phases)
            .map(|(row, phase)| crate::PhasedPauliString::new(row, phase))
            .collect::<Vec<_>>();
        let presentation = zx.stabilizers_from_external_basis_with_limits(&signed, self.limits)?;
        let mut prefix = Vec::new();
        for generator in &presentation.generators {
            self.topology.diagram.charge(1)?;
            let Some(name) = generator.measurement_name() else {
                continue;
            };
            let mut paulis = generator.stabilizer.paulis.clone();
            zx.clear_cross_centers(std::slice::from_mut(&mut paulis));
            let mut row = BooleanRow::default();
            self.topology.diagram.charge(self.columns.len())?;
            for (&column, &index) in &self.columns {
                let pauli = match column {
                    SurfaceColumn::Node(position) => paulis.get(
                        zx.node_at(IVec3::from_array(position))
                            .expect("constant topology columns belong to the reference ZX graph")
                            .id,
                    ),
                    SurfaceColumn::Edge(source, target) => {
                        let source = zx
                            .node_at(IVec3::from_array(source))
                            .expect("constant edge source belongs to the reference ZX graph")
                            .id;
                        let target = zx
                            .node_at(IVec3::from_array(target))
                            .expect("constant edge target belongs to the reference ZX graph")
                            .id;
                        let edge = zx
                            .edge_between(source, target)
                            .expect("constant edge column belongs to the reference ZX graph");
                        let value = paulis.get(edge.id);
                        if edge.hadamard && source > target {
                            value.flip()
                        } else {
                            value
                        }
                    }
                };
                for (axis, component) in [Pauli::X, Pauli::Z].into_iter().enumerate() {
                    if pauli & component {
                        row.set(2 * index + axis, ONE);
                    }
                }
            }
            // Named representatives must belong to the actual relation, even
            // when the presentation also contains synthetic selective fixings.
            self.topology.diagram.charge(row.terms().len())?;
            let mut remainder = row.clone();
            self.topology.diagram.charge(rows.len())?;
            for basis in &rows {
                let column = basis
                    .terms()
                    .first()
                    .expect("reduced physical basis excludes zero rows")
                    .0;
                remainder.xor_scaled(basis, remainder.get(column), &mut self.topology.diagram)?;
            }
            if !remainder.terms().is_empty() {
                return Err(StabilizerError::MeasurementSurfaceUnavailable {
                    mvar: name.to_owned(),
                }
                .into());
            }
            let coordinate = names
                .iter()
                .find(|(other, _)| other == name)
                .expect("presentation measurements come from the indexed source actions")
                .1;
            row.set(coordinate, ONE);
            prefix.push(row);
        }
        prefix = reduce_boolean_rows(prefix, 0..physical_columns, &mut self.topology.diagram)?;
        // Complete with physical directions independent of the named prefix.
        // These new basis vectors carry no original readable-value identity.
        self.topology.diagram.charge(rows.len())?;
        for row in &mut rows {
            self.topology.diagram.charge(prefix.len())?;
            for named in &prefix {
                let column = named
                    .terms()
                    .first()
                    .expect("reduced named prefix excludes zero rows")
                    .0;
                row.xor_scaled(named, row.get(column), &mut self.topology.diagram)?;
            }
            row.truncate_columns(physical_columns);
        }
        prefix.extend(reduce_boolean_rows(
            rows,
            0..physical_columns,
            &mut self.topology.diagram,
        )?);
        Ok(prefix)
    }

    #[cfg(test)]
    fn check_readout_budget<'a>(
        &mut self,
        pending: impl IntoIterator<Item = &'a mut BooleanRow>,
    ) -> Result<(), BlockGraphError> {
        self.check_readout_roots(pending.into_iter().flat_map(BooleanRow::decisions_mut))
    }

    fn check_readout_roots<'a>(
        &mut self,
        pending: impl IntoIterator<Item = &'a mut DecisionId>,
    ) -> Result<(), BlockGraphError> {
        let allocated = self.topology.diagram.nodes().len();
        if allocated > self.next_readout_collection || allocated > self.limits.max_witness_nodes {
            self.collect_readout_roots(pending)?;
        }
        self.check_budget()
    }

    fn collect_readout_roots<'a>(
        &mut self,
        pending: impl IntoIterator<Item = &'a mut DecisionId>,
    ) -> Result<(), BlockGraphError> {
        let mut pending = pending
            .into_iter()
            .map(|root| {
                self.topology.diagram.charge(1)?;
                Ok(root)
            })
            .collect::<Result<Vec<_>, BooleanResourceError>>()?;
        let rows = self
            .rows
            .iter()
            .map(|row| row.terms().len() + 1)
            .sum::<usize>();
        let surfaces = self
            .surfaces
            .iter()
            .map(|surface| {
                let folds = match &surface.kind {
                    GuardedSurfaceKind::Readout { folds, .. } => folds.len(),
                    GuardedSurfaceKind::OutputFrame { folds, .. } => folds.len(),
                    GuardedSurfaceKind::LogicalReadout => 0,
                };
                surface.row.terms().len() + 1 + surface.feedbacks.len() + folds
            })
            .sum::<usize>();
        let local = self.local.values().map(Vec::len).sum::<usize>();
        self.topology.diagram.charge(
            self.topology
                .diagram
                .nodes()
                .len()
                .saturating_add(self.topology.decision_root_count())
                .saturating_add(rows)
                .saturating_add(surfaces)
                .saturating_add(self.topology.diagram.witness_collection_work())
                .saturating_add(local),
        )?;
        let coefficients = self
            .rows
            .iter_mut()
            .flat_map(BooleanRow::decisions_mut)
            .chain(pending.iter_mut().map(|root| &mut **root));
        let surfaces = self
            .surfaces
            .iter_mut()
            .flat_map(GuardedSurface::decisions_mut);
        let local = self
            .local
            .values_mut()
            .flatten()
            .map(|local| &mut local.guard);
        self.topology
            .collect_garbage(coefficients.chain(surfaces).chain(local));
        self.next_readout_collection = next_readout_collection(
            self.topology.diagram.nodes().len(),
            self.limits.max_witness_nodes,
        );
        Ok(())
    }

    fn check_budget(&self) -> Result<(), BlockGraphError> {
        let observed = self.topology.diagram.nodes().len();
        if observed > self.limits.max_witness_nodes {
            Err(StabilizerError::ResourceLimited {
                phase: "guarded readout coefficients",
                observed,
                limit: self.limits.max_witness_nodes,
            }
            .into())
        } else {
            Ok(())
        }
    }

    fn measurement_column(
        &self,
        target: MeasureTarget,
        zx: &ZXGraph,
    ) -> Result<SurfaceColumn, BlockGraphError> {
        let column = zx.measurement_column(&target).ok_or_else(|| {
            StabilizerError::MeasurementSurfaceUnavailable {
                mvar: format!("{target:?}"),
            }
        })?;
        Ok(if column < zx.nodes().len() {
            SurfaceColumn::Node(zx.nodes()[column].pos.to_array())
        } else {
            let edge = &zx.edges()[column - zx.nodes().len()];
            SurfaceColumn::Edge(
                zx.nodes()[edge.n1].pos.to_array(),
                zx.nodes()[edge.n2].pos.to_array(),
            )
        })
    }

    /// The raw center is a linear projection of an incident arm. Only a
    /// degree-zero variant owns an independent center coordinate. Crossing
    /// support is reconstructed after evaluating the arms, never XORed here.
    fn component_terms(&self, column: SurfaceColumn, axis: Pauli) -> Vec<(usize, DecisionId)> {
        let offset = usize::from(axis == Pauli::Z);
        let SurfaceColumn::Node(position) = column else {
            return vec![(2 * self.columns[&column] + offset, ONE)];
        };
        self.local[&position]
            .iter()
            .filter_map(|variant| {
                let node = variant.zx.nodes()[variant.node];
                let arm = variant.zx.neighbor_edges(variant.node).next();
                if arm.is_some()
                    && matches!(node.kind, NodeKind::X | NodeKind::Z)
                    && node.kind.cross_pauli() == axis
                {
                    return None;
                }
                let key = arm.map_or(SurfaceColumn::Node(position), |(neighbor, _)| {
                    SurfaceColumn::Edge(position, variant.zx.nodes()[neighbor].pos.to_array())
                });
                Some((2 * self.columns[&key] + offset, variant.guard))
            })
            .collect()
    }

    fn node_components(&self, position: [i32; 3]) -> [Vec<(usize, DecisionId)>; 2] {
        [Pauli::X, Pauli::Z].map(|axis| self.component_terms(SurfaceColumn::Node(position), axis))
    }

    fn component(&self, row: &BooleanRow, column: SurfaceColumn, axis: Pauli) -> DecisionId {
        let index = self.columns[&column];
        row.get(2 * index + usize::from(axis == Pauli::Z))
    }

    #[cfg(test)]
    fn linear_value(
        row: &BooleanRow,
        terms: &[(usize, DecisionId)],
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) -> Result<DecisionId, BooleanResourceError> {
        diagram.charge(terms.len())?;
        terms.iter().try_fold(ZERO, |value, &(column, weight)| {
            let term = diagram.apply(BooleanOp::And, weight, row.get(column))?;
            diagram.apply(BooleanOp::Xor, value, term)
        })
    }

    fn index_output_axes(
        &mut self,
        outputs: &[(IVec3, usize, bool)],
    ) -> Result<BTreeMap<usize, Vec<(usize, DecisionId)>>, BooleanResourceError> {
        self.topology.diagram.charge(outputs.len())?;
        let mut columns = BTreeMap::<usize, Vec<(usize, DecisionId)>>::new();
        for (index, &(port, axis, active)) in outputs.iter().enumerate() {
            if !active {
                continue;
            }
            self.topology
                .diagram
                .charge(self.local[&port.to_array()].len())?;
            let terms = self.component_terms(
                SurfaceColumn::Node(port.to_array()),
                [Pauli::X, Pauli::Z][axis],
            );
            self.topology.diagram.charge(terms.len())?;
            for (column, weight) in terms {
                if weight != ZERO {
                    columns.entry(column).or_default().push((index, weight));
                }
            }
        }
        Ok(columns)
    }

    fn output_folds(
        &mut self,
        row: &BooleanRow,
        index: usize,
        outputs: &[(IVec3, usize, bool)],
        columns: &BTreeMap<usize, Vec<(usize, DecisionId)>>,
    ) -> Result<Vec<(IVec3, PauliBasis, DecisionId)>, BooleanResourceError> {
        if index + 1 == outputs.len() {
            return Ok(Vec::new());
        }
        // Probe the smaller support, keeping column order on either path.
        let scan_row = row.terms().len() <= columns.len();
        let probes = if scan_row {
            row.terms().len()
        } else {
            columns.len()
        };
        let width = row.terms().len().max(columns.len()).max(1);
        self.topology
            .diagram
            .charge(probes.saturating_mul(width.ilog2() as usize + 1))?;
        let mut values = BTreeMap::new();
        let mut accumulate = |coefficient, candidates: &[(usize, DecisionId)]| {
            self.topology.diagram.charge(candidates.len())?;
            for &(later, weight) in candidates {
                if later <= index {
                    continue;
                }
                let term = self
                    .topology
                    .diagram
                    .apply(BooleanOp::And, weight, coefficient)?;
                if term == ZERO {
                    continue;
                }
                self.topology
                    .diagram
                    .charge(values.len().max(1).ilog2() as usize + 1)?;
                let value = values.entry(later).or_insert(ZERO);
                *value = self.topology.diagram.apply(BooleanOp::Xor, *value, term)?;
            }
            Ok::<_, BooleanResourceError>(())
        };
        if scan_row {
            for &(column, coefficient) in row.terms() {
                if let Some(candidates) = columns.get(&column) {
                    accumulate(coefficient, candidates)?;
                }
            }
        } else {
            for (&column, candidates) in columns {
                let coefficient = row.get(column);
                if coefficient != ZERO {
                    accumulate(coefficient, candidates)?;
                }
            }
        }
        self.topology.diagram.charge(values.len())?;
        let mut folds = Vec::new();
        for (later, value) in values {
            let guard = self
                .topology
                .diagram
                .apply(BooleanOp::And, row.active, value)?;
            if guard != ZERO {
                let (port, axis, _) = outputs[later];
                folds.push((
                    port,
                    if axis == 0 {
                        PauliBasis::Z
                    } else {
                        PauliBasis::X
                    },
                    guard,
                ));
            }
        }
        Ok(folds)
    }

    fn fill_selective(
        &mut self,
        rows: &mut Vec<BooleanRow>,
        resolved: &BTreeSet<[i32; 3]>,
    ) -> Result<(), BooleanResourceError> {
        if resolved.is_empty() {
            return Ok(());
        }
        let mut indexed = BooleanRowSpace::new(std::mem::take(rows), &mut self.topology.diagram)?;
        self.fill_selective_indexed(&mut indexed, resolved)?;
        *rows = indexed.into_rows(&mut self.topology.diagram)?;
        Ok(())
    }

    fn fill_selective_indexed(
        &mut self,
        indexed: &mut BooleanRowSpace,
        resolved: &BTreeSet<[i32; 3]>,
    ) -> Result<(), BooleanResourceError> {
        self.topology.diagram.charge(self.topology.resolves.len())?;
        for index in 0..self.topology.resolves.len() {
            let (position, _) = self.topology.resolves[index];
            if !resolved.contains(&position.to_array()) {
                continue;
            }
            self.fill_selective_site(indexed, index)?;
        }
        Ok(())
    }

    fn fill_selective_site(
        &mut self,
        indexed: &mut BooleanRowSpace,
        index: usize,
    ) -> Result<(), BooleanResourceError> {
        let (position, selector) = self.topology.resolves[index];
        let NodeKind::Selective(kind) = self.local[&position.to_array()][0]
            .zx
            .node_at(position)
            .expect("local selective view retains its center node")
            .kind
        else {
            unreachable!("selective fill positions name selective nodes")
        };
        let low = Pauli::from(kind.pauli_if_false());
        let high = Pauli::from(kind.pauli_if_true());
        let mut weights = [ZERO; 2];
        for (weight, axis) in weights.iter_mut().zip([Pauli::Z, Pauli::X]) {
            *weight = match (low & axis, high & axis) {
                (false, false) => ZERO,
                (true, true) => ONE,
                (false, true) => selector,
                (true, false) => self.topology.diagram.negate(selector)?,
            };
        }
        // This also fills the terminal source basis. Preserve its reference
        // sector, just as for seam composition; resource rows are not neutral.
        let mut terms = Vec::new();
        for (components, weight) in self
            .node_components(position.to_array())
            .into_iter()
            .zip(weights)
        {
            for (column, guard) in components {
                terms.push((
                    column,
                    self.topology.diagram.apply(BooleanOp::And, guard, weight)?,
                ));
            }
        }
        indexed.eliminate_linear(&terms, &mut self.topology.diagram)?;
        Ok(())
    }

    fn feedback_coefficient(
        &mut self,
        row: &BooleanRow,
        targets: &[crate::FeedbackTarget],
    ) -> Result<DecisionId, BooleanResourceError> {
        self.topology.diagram.charge(targets.len())?;
        let mut result = ZERO;
        for target in targets {
            self.topology
                .diagram
                .charge(self.local[&target.target.to_array()].len())?;
            let mut value = ZERO;
            for variant in &self.local[&target.target.to_array()] {
                let edge = variant
                    .zx
                    .feedback_edge(target)
                    .expect("validated feedback wire");
                let column = SurfaceColumn::Edge(
                    variant.zx.nodes()[edge.n1].pos.to_array(),
                    variant.zx.nodes()[edge.n2].pos.to_array(),
                );
                let bits = [
                    self.component(row, column, Pauli::X),
                    self.component(row, column, Pauli::Z),
                ];
                let pauli = Pauli::from(target.pauli);
                let x = if pauli & Pauli::Z { bits[0] } else { ZERO };
                let z = if pauli & Pauli::X { bits[1] } else { ZERO };
                let term = self.topology.diagram.apply(BooleanOp::Xor, x, z)?;
                let term = self
                    .topology
                    .diagram
                    .apply(BooleanOp::And, variant.guard, term)?;
                value = self.topology.diagram.apply(BooleanOp::Or, value, term)?;
            }
            result = self.topology.diagram.apply(BooleanOp::Xor, result, value)?;
        }
        self.topology
            .diagram
            .apply(BooleanOp::And, row.active, result)
    }

    fn feedback_support(
        &self,
        target: &crate::FeedbackTarget,
        variant: &SurfaceTopology,
    ) -> Result<Vec<usize>, BlockGraphError> {
        let edge =
            variant
                .zx
                .feedback_edge(target)
                .ok_or(StabilizerError::FeedbackTargetWithoutWire {
                    target: target.target,
                    pauli: target.pauli,
                })?;
        let column = SurfaceColumn::Edge(
            variant.zx.nodes()[edge.n1].pos.to_array(),
            variant.zx.nodes()[edge.n2].pos.to_array(),
        );
        Ok(Pauli::from(target.pauli)
            .flip()
            .iter_xz()
            .map(|axis| 2 * self.columns[&column] + usize::from(axis == Pauli::Z))
            .collect())
    }

    fn index_feedbacks(
        &mut self,
        actions: &[Action],
    ) -> Result<HashMap<usize, Vec<usize>>, BlockGraphError> {
        self.topology.diagram.charge(actions.len())?;
        let mut columns = HashMap::<usize, Vec<usize>>::new();
        for (ordinal, action) in actions.iter().enumerate() {
            let Action::Feedback { targets, .. } = action else {
                continue;
            };
            self.topology.diagram.charge(targets.len())?;
            for target in targets {
                let variants = &self.local[&target.target.to_array()];
                self.topology.diagram.charge(variants.len())?;
                for variant in variants {
                    let support = self.feedback_support(target, variant)?;
                    self.topology.diagram.charge(support.len())?;
                    for column in support {
                        let ordinals = columns.entry(column).or_default();
                        if ordinals.last() != Some(&ordinal) {
                            ordinals.push(ordinal);
                        }
                    }
                }
            }
        }
        Ok(columns)
    }

    fn feedbacks<const INDEXED: bool>(
        &mut self,
        row: &BooleanRow,
        actions: &[Action],
        columns: &HashMap<usize, Vec<usize>>,
    ) -> Result<Vec<(u32, DecisionId)>, BooleanResourceError> {
        let ordinals = if INDEXED {
            self.topology.diagram.charge(row.terms().len())?;
            let mut ordinals = BTreeSet::new();
            for &(column, _) in row.terms() {
                if let Some(candidates) = columns.get(&column) {
                    self.topology.diagram.charge(candidates.len())?;
                    ordinals.extend(candidates.iter().copied());
                }
            }
            ordinals.into_iter().collect::<Vec<_>>()
        } else {
            // The unreclaimed planner retains the full action scan as an
            // independent oracle for physical feedback incidence.
            self.topology.diagram.charge(actions.len())?;
            (0..actions.len()).collect()
        };
        let mut feedbacks = Vec::new();
        for ordinal in ordinals {
            let Action::Feedback { targets, condition } = &actions[ordinal] else {
                continue;
            };
            let coefficient = self.feedback_coefficient(row, targets)?;
            let condition = condition
                .as_ref()
                .map_or(ONE, |expr| self.topology.condition(expr));
            let coefficient =
                self.topology
                    .diagram
                    .apply(BooleanOp::And, coefficient, condition)?;
            if coefficient != ZERO {
                feedbacks.push((ordinal as u32, coefficient));
            }
        }
        Ok(feedbacks)
    }

    fn materialize_row(
        &self,
        row: &BooleanRow,
        zx: &ZXGraph,
        evaluate: impl Fn(DecisionId) -> bool,
    ) -> crate::PauliString {
        materialize_row(
            self.columns.iter().map(|(&column, &index)| (column, index)),
            row,
            zx,
            evaluate,
        )
    }

    pub(crate) fn materialize_relation(
        &self,
        zx: &ZXGraph,
        evaluate: impl Fn(DecisionId) -> bool,
    ) -> Vec<crate::PauliString> {
        self.rows
            .iter()
            .filter(|row| evaluate(row.active))
            .map(|row| self.materialize_row(row, zx, &evaluate))
            .filter(|row| !row.is_identity())
            .collect()
    }

    #[cfg(test)]
    fn evaluate(
        &self,
        zx: &ZXGraph,
        variables: &HashMap<String, bool>,
    ) -> Vec<crate::PhasedPauliString> {
        let evaluate = |bit| self.topology.evaluate_source(bit, |name| variables[name]);
        self.rows
            .iter()
            .filter(|row| evaluate(row.active))
            .map(|row| {
                let paulis = self.materialize_row(row, zx, evaluate);
                let phase = zx.stabilizer_row_phases(std::slice::from_ref(&paulis))[0];
                crate::PhasedPauliString::new(paulis, phase)
            })
            .filter(|row| !row.paulis.is_identity())
            .collect()
    }
}

pub(super) fn materialize_row(
    columns: impl IntoIterator<Item = (SurfaceColumn, usize)>,
    row: &BooleanRow,
    zx: &ZXGraph,
    evaluate: impl Fn(DecisionId) -> bool,
) -> crate::PauliString {
    let mut paulis = crate::PauliString::new(zx.total_ids());
    for (column, index) in columns {
        let value = match (
            evaluate(row.get(2 * index)),
            evaluate(row.get(2 * index + 1)),
        ) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
        };
        if value == Pauli::I {
            continue;
        }
        let (id, value) = match column {
            SurfaceColumn::Node(position) => (
                zx.node_at(IVec3::from_array(position))
                    .expect("active surface node belongs to the supplied ZX graph")
                    .id,
                value,
            ),
            SurfaceColumn::Edge(source, target) => {
                let source = zx
                    .node_at(IVec3::from_array(source))
                    .expect("active surface edge source belongs to the supplied ZX graph")
                    .id;
                let target = zx
                    .node_at(IVec3::from_array(target))
                    .expect("active surface edge target belongs to the supplied ZX graph")
                    .id;
                let edge = zx
                    .edge_between(source, target)
                    .expect("active surface edge belongs to the supplied ZX graph");
                (
                    edge.id,
                    if edge.hadamard && source > target {
                        value.flip()
                    } else {
                        value
                    },
                )
            }
        };
        paulis.set(id, value);
    }
    zx.reconstruct_raw_centers(&mut paulis);
    paulis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gc_policy_graph_remaps_pending_roots_and_keeps_explicit_quota_checks() {
        for limit in [2 * READOUT_COLLECTION_TARGET, usize::MAX] {
            let limits = ModuleCertificationLimits {
                max_witness_nodes: limit,
                ..ModuleCertificationLimits::DEFAULT
            };
            let mut space = GuardedSurfaceSpace::new(
                GuardedTopology::new(&crate::BlockGraph::new(), limits).unwrap(),
                &HashMap::new(),
                limits,
            )
            .unwrap();
            space.topology.diagram.make_node(9, ZERO, ONE).unwrap();
            let live = space.topology.diagram.make_node(0, ZERO, ONE).unwrap();
            let mut pending = BooleanRow::from_terms(live, [(0, live)]);
            space.next_readout_collection = 0;
            space.check_readout_budget([&mut pending]).unwrap();
            assert_eq!(space.topology.diagram.nodes().len(), 1);
            assert_ne!(pending.active, live, "the dead predecessor is removed");
            assert_eq!(pending.get(0), pending.active);
            for value in [false, true] {
                assert_eq!(
                    space.topology.diagram.evaluate(pending.active, |_| value),
                    value
                );
            }
            let spent = space.topology.diagram.steps();
            space.check_readout_budget([&mut pending]).unwrap();
            assert_eq!(
                space.topology.diagram.steps(),
                spent,
                "unchanged live data does not recollect"
            );

            space.limits.max_witness_nodes = 0;
            assert!(matches!(
                space.check_readout_budget([&mut pending]),
                Err(BlockGraphError::Stabilizer(
                    StabilizerError::ResourceLimited {
                        phase: "guarded readout coefficients",
                        observed: 1,
                        limit: 0,
                    }
                ))
            ));
        }
    }

    #[track_caller]
    pub(super) fn assert_same_raw_plan(
        actual: &GuardedSurfaceSpace,
        expected: &GuardedSurfaceSpace,
    ) {
        // Both plans use the same source variable order; compare raw functions
        // on every assignment, including outside activation and source domain.
        let mut canonical = expected.topology.diagram.clone();
        let mut remap = vec![ZERO, ONE];
        for node in actual.topology.diagram.nodes() {
            remap.push(
                canonical
                    .make_node(node.variable, remap[node.low.0], remap[node.high.0])
                    .unwrap(),
            );
        }
        assert_eq!(actual.columns, expected.columns);
        assert_eq!(remap[actual.topology.domain.0], expected.topology.domain);
        assert_eq!(actual.surfaces.len(), expected.surfaces.len());
        for (actual, expected) in actual.surfaces.iter().zip(&expected.surfaces) {
            assert_eq!(remap[actual.row.active.0], expected.row.active);
            assert_eq!(
                actual
                    .row
                    .terms()
                    .iter()
                    .map(|&(column, value)| (column, remap[value.0]))
                    .collect::<Vec<_>>(),
                expected.row.terms(),
                "complete raw witness, including off-domain coefficients"
            );
            assert_eq!(
                actual
                    .feedbacks
                    .iter()
                    .map(|&(ordinal, value)| (ordinal, remap[value.0]))
                    .collect::<Vec<_>>(),
                expected.feedbacks
            );
            let mut kind = actual.kind.clone();
            match &mut kind {
                GuardedSurfaceKind::Readout { folds, .. } => {
                    for (_, value) in folds {
                        *value = remap[value.0];
                    }
                }
                GuardedSurfaceKind::OutputFrame { folds, .. } => {
                    for (_, _, value) in folds {
                        *value = remap[value.0];
                    }
                }
                GuardedSurfaceKind::LogicalReadout => {}
            }
            assert_eq!(kind, expected.kind, "ordered role, label, and folds");
        }
    }

    #[test]
    fn correlation_columns_are_edges_except_for_isolated_variants() {
        for gallery in [
            crate::GalleryItem::XMemory,
            crate::GalleryItem::YMemory,
            crate::GalleryItem::CNOT,
            crate::GalleryItem::T,
            crate::GalleryItem::ThreeBitAdder,
        ] {
            let graph = gallery
                .build()
                .materialize_root_graph()
                .unwrap()
                .fix_shadowed_faces();
            let limits = ModuleCertificationLimits::DEFAULT;
            let space = GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &HashMap::new(),
                limits,
            )
            .unwrap();
            let isolated = space
                .local
                .iter()
                .filter(|(_, variants)| {
                    variants
                        .iter()
                        .any(|variant| variant.zx.neighbor_edges(variant.node).next().is_none())
                })
                .map(|(&position, _)| position)
                .collect::<BTreeSet<_>>();
            let centers = space
                .columns
                .keys()
                .filter_map(|column| match column {
                    SurfaceColumn::Node(position) => Some(*position),
                    SurfaceColumn::Edge(_, _) => None,
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(centers, isolated, "{gallery:?}");
            let plan = space.plan_readouts().unwrap();
            for (position, variants) in &plan.local {
                for variant in variants {
                    assert_eq!(variant.center.is_some(), variant.arms.is_empty());
                    assert!(variant.center.is_none() || isolated.contains(position));
                }
            }
        }
    }

    #[test]
    fn a_conditionally_isolated_site_keeps_its_memory_direction() {
        // Exercise local guarded algebra directly. Authored branch interfaces
        // independently reject disconnected blocks inside an arm.
        let isolated = crate::GalleryItem::XMemory
            .build()
            .materialize_root_graph()
            .unwrap();
        let mut connected = isolated.clone();
        connected.add_block(crate::Block::new(
            IVec3::Z,
            crate::BlockKind::Cube(crate::CubeKind::ZXX),
        ));
        connected.add_pipe(crate::Pipe::new(IVec3::ZERO, crate::Direction::ZPLUS));
        let connected = Arc::new(connected.fix_shadowed_faces());
        let isolated = Arc::new(isolated.fix_shadowed_faces());
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut topology = GuardedTopology::new(&connected, limits).unwrap();
        topology
            .variables
            .push(crate::GuardedVariable::Outcome("enable".to_owned()));
        let enabled = topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let disabled = topology.diagram.negate(enabled).unwrap();
        topology.sites.insert(
            [0; 3],
            vec![
                crate::GuardedProjection {
                    guard: disabled,
                    graph: Arc::clone(&isolated),
                },
                crate::GuardedProjection {
                    guard: enabled,
                    graph: Arc::clone(&connected),
                },
            ],
        );
        topology.sites.get_mut(&IVec3::Z.to_array()).unwrap()[0].guard = enabled;
        let space = GuardedSurfaceSpace::new(topology, &HashMap::new(), limits).unwrap();
        assert!(space.columns.contains_key(&SurfaceColumn::Node([0; 3])));
        for enabled in [false, true] {
            let graph = if enabled { &connected } else { &isolated };
            let zx = ZXGraph::from_block_graph_for_analysis(graph).unwrap();
            let actual = super::super::modular::canonical_external_basis(
                space.evaluate(&zx, &HashMap::from([("enable".to_owned(), enabled)])),
                zx.total_ids(),
            );
            assert_eq!(actual, zx.to_external_generator_table_with_signed().1);
        }
    }

    #[test]
    fn sparse_readout_projection_visits_pivots_introduced_by_substitution() {
        let mut diagram = bloq_utils::boolean::BooleanDecisionDiagram::with_limits(
            bloq_utils::boolean::BooleanLimits::UNLIMITED,
        );
        let guard = diagram.make_node(0, ZERO, ONE).unwrap();
        let mut pivots = GuardedPivots::default();
        pivots
            .insert_residual::<false>(
                BooleanRow::from_terms(ONE, [(5, ONE), (10, guard), (70, ONE)]),
                [5].into_iter(),
                &mut diagram,
            )
            .unwrap();
        pivots
            .insert_residual::<false>(
                BooleanRow::from_terms(ONE, [(10, ONE), (90, ONE)]),
                [10].into_iter(),
                &mut diagram,
            )
            .unwrap();
        let mut row = BooleanRow::from_terms(ONE, [(5, ONE)]);
        pivots.project(&mut row, &mut diagram).unwrap();
        assert_eq!(row.terms(), &[(70, ONE), (90, guard)]);
    }

    #[test]
    fn deferred_terminal_witnesses_preserve_raw_reference_rows_and_folds() {
        let mut prepared = crate::GalleryItem::OneDYoked
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        prepared
            .set_block_kind(IVec3::ZERO, crate::BlockKind::T)
            .unwrap();
        let cases = [
            crate::GalleryItem::XMemory
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::T
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::TWithPreparedY
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::CCZGateTeleport
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::CZSpatialH
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::CZTemporalH
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            crate::GalleryItem::ThreeBitAdder
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            prepared.clone(),
            prepared.flip_xz_basis().unwrap(),
            crate::parse_inline_graph(&bloq_test::benchmark::yoked_memory(32).to_blog_text())
                .unwrap(),
        ];
        for graph in cases {
            let graph = graph.fix_shadowed_faces();
            let limits = ModuleCertificationLimits::DEFAULT;
            let build = || {
                GuardedSurfaceSpace::new(
                    GuardedTopology::new(&graph, limits).unwrap(),
                    &HashMap::new(),
                    limits,
                )
                .unwrap()
            };
            let mut expanded = build();
            expanded.plan_causal_readouts::<true, false>().unwrap();
            let mut taped = build();
            taped.plan_causal_readouts::<true, true>().unwrap();
            assert_same_raw_plan(&taped, &expanded);
            assert_output_folds_match_dense(&mut taped);
        }
    }

    fn assert_output_folds_match_dense(space: &mut GuardedSurfaceSpace) {
        let frames = space
            .surfaces
            .iter()
            .filter_map(|surface| {
                let GuardedSurfaceKind::OutputFrame { port, basis, folds } = &surface.kind else {
                    return None;
                };
                Some((
                    *port,
                    usize::from(*basis == PauliBasis::X),
                    surface.row.clone(),
                    folds.clone(),
                ))
            })
            .collect::<Vec<_>>();
        let outputs = frames
            .iter()
            .map(|(port, axis, row, _)| (*port, *axis, row.active != ZERO))
            .collect::<Vec<_>>();
        for (index, (_, _, row, actual)) in frames.iter().enumerate() {
            let mut expected = Vec::new();
            for &(port, axis, active) in &outputs[index + 1..] {
                if !active {
                    continue;
                }
                let terms = space.component_terms(
                    SurfaceColumn::Node(port.to_array()),
                    [Pauli::X, Pauli::Z][axis],
                );
                let value =
                    GuardedSurfaceSpace::linear_value(row, &terms, &mut space.topology.diagram)
                        .unwrap();
                let guard = space
                    .topology
                    .diagram
                    .apply(BooleanOp::And, row.active, value)
                    .unwrap();
                if guard != ZERO {
                    expected.push((
                        port,
                        if axis == 0 {
                            PauliBasis::Z
                        } else {
                            PauliBasis::X
                        },
                        guard,
                    ));
                }
            }
            assert_eq!(*actual, expected, "output fold order and Boolean guards");
        }
    }

    #[test]
    fn independent_outputs_have_no_cross_folds() {
        let mut graph = crate::BlockGraph::new();
        for x in 0..32 {
            for (z, kind) in [
                (0, crate::BlockKind::Port),
                (1, crate::BlockKind::Cube(crate::CubeKind::ZXZ)),
                (2, crate::BlockKind::Port),
            ] {
                graph.add_block(crate::Block::new(IVec3::new(x, 0, z), kind));
            }
            for z in 0..2 {
                graph.add_pipe(crate::Pipe::new(
                    IVec3::new(x, 0, z),
                    crate::Direction::ZPLUS,
                ));
            }
        }
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        space.plan_causal_readouts::<true, true>().unwrap();
        assert_output_folds_match_dense(&mut space);
        let mut outputs = Vec::new();
        for surface in &space.surfaces {
            if let GuardedSurfaceKind::OutputFrame { port, basis, folds } = &surface.kind {
                assert!(
                    folds.is_empty(),
                    "independent wires have no cross-output fold"
                );
                outputs.push((
                    *port,
                    usize::from(*basis == PauliBasis::X),
                    surface.row.active != ZERO,
                ));
            }
        }
        assert_eq!(outputs.len(), 64);
    }

    #[test]
    fn output_folds_probe_only_the_smaller_support() {
        let limits = ModuleCertificationLimits {
            max_boolean_steps: 100,
            ..ModuleCertificationLimits::DEFAULT
        };
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&crate::BlockGraph::new(), limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let row = BooleanRow::from_terms(ONE, (0..4096).map(|column| (column, ONE)));
        let outputs = [(IVec3::ZERO, 0, true), (IVec3::Z, 0, true)];
        let columns = BTreeMap::from([(0, vec![(1, ONE)])]);
        assert_eq!(
            space.output_folds(&row, 0, &outputs, &columns).unwrap(),
            [(IVec3::Z, PauliBasis::Z, ONE)]
        );
    }

    fn expand_output_folds(
        surfaces: &mut [GuardedSurface],
        diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    ) {
        let mut later = HashMap::<_, usize>::new();
        for index in (0..surfaces.len()).rev() {
            let GuardedSurfaceKind::OutputFrame { port, basis, folds } = &mut surfaces[index].kind
            else {
                continue;
            };
            let key = (*port, *basis);
            for (port, basis, guard) in std::mem::take(folds) {
                let &target = later
                    .get(&(port, basis))
                    .expect("fold refers to a later output");
                let row = surfaces[target].row.clone();
                let feedbacks = surfaces[target].feedbacks.clone();
                surfaces[index]
                    .row
                    .xor_scaled(&row, guard, diagram)
                    .unwrap();
                let mut combined = surfaces[index]
                    .feedbacks
                    .iter()
                    .copied()
                    .collect::<BTreeMap<_, _>>();
                for (ordinal, value) in feedbacks {
                    let value = diagram.apply(BooleanOp::And, guard, value).unwrap();
                    let previous = combined.entry(ordinal).or_insert(ZERO);
                    *previous = diagram.apply(BooleanOp::Xor, *previous, value).unwrap();
                }
                surfaces[index].feedbacks = combined
                    .into_iter()
                    .filter(|(_, value)| *value != ZERO)
                    .collect();
            }
            later.insert(key, index);
        }
    }

    #[test]
    fn spatial_column_order_preserves_seam_witnesses() {
        let mut graph = crate::GalleryItem::OneDYoked
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection")
            .fix_shadowed_faces();
        graph
            .set_block_kind(IVec3::ZERO, crate::BlockKind::T)
            .unwrap();
        let positions = graph
            .blocks()
            .map(|block| block.pos().to_array())
            .collect::<BTreeSet<_>>();
        let mut extended = graph.clone();
        for (z, kind) in [
            (0, crate::BlockKind::Port),
            (1, crate::BlockKind::Cube(crate::CubeKind::ZXZ)),
            (2, crate::BlockKind::Port),
        ] {
            extended.add_block(crate::Block::new(IVec3::new(64, 0, z), kind));
        }
        for z in 0..2 {
            extended.add_pipe(crate::Pipe::new(
                IVec3::new(64, 0, z),
                crate::Direction::ZPLUS,
            ));
        }
        extended
            .set_port_role([64, 0, 0], crate::PortRole::Input)
            .unwrap();
        extended
            .set_port_role([64, 0, 2], crate::PortRole::Output)
            .unwrap();
        let build = |source: &crate::BlockGraph| {
            let limits = ModuleCertificationLimits::DEFAULT;
            GuardedSurfaceSpace::new(
                GuardedTopology::new(source, limits).unwrap(),
                &HashMap::new(),
                limits,
            )
            .unwrap()
        };
        let original = build(&graph);
        let extended = build(&extended);
        // The disconnected wire changes the dominant extent from Z to X.
        assert!(
            original
                .columns
                .iter()
                .any(|(column, index)| extended.columns[column] != *index)
        );
        let witnesses = |space: &GuardedSurfaceSpace| {
            // Both sources are unguarded, so decision IDs are shared constants.
            assert!(space.topology.diagram.nodes().is_empty());
            let columns = space
                .columns
                .iter()
                .map(|(&column, &index)| (index, column))
                .collect::<BTreeMap<_, _>>();
            space
                .rows
                .iter()
                .filter_map(|row| {
                    let terms = row
                        .terms()
                        .iter()
                        .map(|&(column, value)| ((columns[&(column / 2)], column % 2), value))
                        .collect::<BTreeMap<_, _>>();
                    let original_terms = terms
                        .keys()
                        .filter(|(column, _)| {
                            let position = match column {
                                SurfaceColumn::Node(position)
                                | SurfaceColumn::Edge(position, _) => position,
                            };
                            positions.contains(position)
                        })
                        .count();
                    assert!(
                        original_terms == 0 || original_terms == terms.len(),
                        "a seam witness must not mix disconnected components"
                    );
                    (original_terms != 0).then_some((row.active, terms))
                })
                .collect::<Vec<_>>()
        };
        // Keep exact row order and all physical coordinates, including the
        // prepared resource. Equality of the correlation span is insufficient.
        assert_eq!(witnesses(&original), witnesses(&extended));
    }

    #[test]
    fn selective_fill_preserves_terminal_resource_reference() {
        let graph = crate::GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection")
            .fix_shadowed_faces();
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let column = |position| space.node_components(position)[0][0].0;
        // Z anticommutes with both choices of this X/Y selective site.
        let selective = column(space.topology.resolves[0].0.to_array()) + 1;
        let output = column([0, 0, 2]);
        let resource = column([1, 0, 0]);
        let input = column([0, 0, 0]);
        let interior = column([0, 0, 1]);
        let row = |columns: &[usize]| {
            BooleanRow::from_terms(ONE, columns.iter().map(|&column| (column, ONE)))
        };
        let mut rows = BooleanRowSpace::new(
            vec![
                row(&[selective, output, resource]),
                row(&[selective, input, interior]),
                row(&[selective, output]),
            ],
            &mut space.topology.diagram,
        )
        .unwrap();
        space.fill_selective_site(&mut rows, 0).unwrap();
        let frame = rows
            .eliminate_column(output, &mut space.topology.diagram)
            .unwrap();
        // The shortest fill pivot would instead leave output+input+interior,
        // changing the reference by the surviving prepared-resource readout.
        assert_eq!(frame.active, ONE);
        assert_eq!(
            frame.terms(),
            row(&[output, resource, input, interior]).terms()
        );
        let remaining = rows.into_rows(&mut space.topology.diagram).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].active, ONE);
        assert_eq!(remaining[0].terms(), row(&[resource]).terms());
    }

    #[test]
    fn availability_releases_preserve_conditional_kernels_and_readout_quotients() {
        use bloq_utils::boolean::BooleanDecisionDiagram;

        fn canonical<'a>(
            rows: impl Iterator<Item = &'a BooleanRow>,
            diagram: &BooleanDecisionDiagram,
            assignment: usize,
        ) -> [u16; 14] {
            let evaluate =
                |root| diagram.evaluate(root, |variable| assignment & (1 << variable) != 0);
            let mut basis = [0u16; 14];
            for row in rows.filter(|row| evaluate(row.active)) {
                let mut bits = 0u16;
                for &(column, coefficient) in row.terms() {
                    assert!(
                        column < basis.len(),
                        "availability tag escaped into a witness"
                    );
                    if evaluate(coefficient) {
                        bits |= 1 << column;
                    }
                }
                while bits != 0 {
                    let pivot = bits.trailing_zeros() as usize;
                    if basis[pivot] == 0 {
                        basis[pivot] = bits;
                        break;
                    }
                    bits ^= basis[pivot];
                }
            }
            for pivot in (0..basis.len()).rev() {
                for previous in 0..pivot {
                    if basis[previous] & (1 << pivot) != 0 {
                        basis[previous] ^= basis[pivot];
                    }
                }
            }
            basis
        }

        let mut orders = Vec::new();
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let order = [a, b, c, d];
                        if order.into_iter().collect::<BTreeSet<_>>().len() == 4 {
                            orders.push(order);
                        }
                    }
                }
            }
        }
        for case in 0..12 {
            let mut original = BooleanDecisionDiagram::default();
            let a = original.make_node(0, ZERO, ONE).unwrap();
            let b = original.make_node(1, ZERO, ONE).unwrap();
            let xor = original.apply(BooleanOp::Xor, a, b).unwrap();
            let not_a = original.negate(a).unwrap();
            let coefficients = [ZERO, ONE, a, b, xor, not_a];
            let mut seed = case as u64 + 17;
            let rows = (0..6)
                .map(|index| {
                    let mut row = BooleanRow::new([ONE, a, b, xor][(case + index) % 4]);
                    for column in 0..8 {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        row.set(
                            column,
                            coefficients[(seed >> 32) as usize % coefficients.len()],
                        );
                    }
                    // Independent source witnesses protect the full relation,
                    // including cancellation invisible in physical support.
                    row.set(8 + index, ONE);
                    row
                })
                .collect::<Vec<_>>();
            let constraints = [
                (AvailabilityKey::Feedback(0), vec![(0, ONE), (3, a)]),
                (AvailabilityKey::Feedback(1), vec![(1, a), (1, b)]),
                (AvailabilityKey::Feedback(2), vec![(2, ONE), (2, ONE)]),
                (AvailabilityKey::Feedback(3), vec![(0, b)]),
            ];
            for order in &orders {
                let mut diagram = original.clone();
                let (mut state, mut available) =
                    GuardedAvailability::new(rows.clone(), &constraints, 14, &mut diagram).unwrap();
                let mut pending = (0..4)
                    .map(AvailabilityKey::Feedback)
                    .collect::<BTreeSet<_>>();
                let mut readouts = GuardedPivots::default();
                for cut in 0..=4 {
                    let mut expected = rows.clone();
                    for (key, terms) in &constraints {
                        if pending.contains(key) {
                            eliminate_boolean_rows(&mut expected, &mut diagram, |row, diagram| {
                                terms.iter().try_fold(ZERO, |sum, &(column, gate)| {
                                    let term =
                                        diagram.apply(BooleanOp::And, gate, row.get(column))?;
                                    diagram.apply(BooleanOp::Xor, sum, term)
                                })
                            })
                            .unwrap();
                        }
                    }
                    for (&column, &owner) in &readouts.owners {
                        let gate = readouts.rows.row(owner).unwrap().active;
                        eliminate_boolean_rows(&mut expected, &mut diagram, |row, diagram| {
                            diagram.apply(BooleanOp::And, gate, row.get(column))
                        })
                        .unwrap();
                    }
                    for assignment in 0..4 {
                        assert_eq!(
                            canonical(available.rows(), &diagram, assignment),
                            canonical(expected.iter(), &diagram, assignment),
                            "case={case}, order={order:?}, cut={cut}, assignment={assignment}",
                        );
                    }
                    if cut == 1 {
                        // A previously available relation may be reclaimed
                        // while unreleased constraint pivots still retain it.
                        let row = available.rows().next().cloned();
                        if let Some(row) = row {
                            let columns = row
                                .terms()
                                .iter()
                                .map(|&(column, _)| column)
                                .collect::<Vec<_>>();
                            let (_, added) = readouts
                                .insert_residual::<false>(row, columns.into_iter(), &mut diagram)
                                .unwrap();
                            for (column, gate) in added {
                                available
                                    .eliminate_column_when_sparse(column, gate, &mut diagram)
                                    .unwrap();
                            }
                        }
                    }
                    if cut < 4 {
                        pending.remove(&AvailabilityKey::Feedback(order[cut]));
                        state
                            .release(&pending, &readouts, &mut available, &mut diagram)
                            .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn forward_readout_pivots_project_like_full_reduction() {
        use bloq_utils::boolean::BooleanDecisionDiagram;

        let mut diagram = BooleanDecisionDiagram::default();
        diagram.make_node(0, ZERO, ONE).unwrap();
        let x = diagram.make_node(1, ZERO, ONE).unwrap();
        let not_x = diagram.negate(x).unwrap();
        let rows = [
            BooleanRow::from_terms(ONE, [(0, ONE), (2, ONE)]),
            BooleanRow::from_terms(ONE, [(2, ONE)]),
            BooleanRow::from_terms(x, [(1, ONE), (3, ONE)]),
            BooleanRow::from_terms(not_x, [(1, ONE), (4, ONE)]),
            BooleanRow::from_terms(ONE, [(3, ONE), (4, ONE)]),
            BooleanRow::from_terms(x, [(5, ONE)]),
        ];
        let mut full = GuardedPivots::default();
        let mut forward = GuardedPivots::default();
        for source in rows {
            let mut a = source.clone();
            let mut b = source;
            full.project(&mut a, &mut diagram).unwrap();
            forward.project(&mut b, &mut diagram).unwrap();
            assert_eq!((a.active, a.terms()), (b.active, b.terms()));
            let columns = a
                .terms()
                .iter()
                .map(|&(column, _)| column)
                .collect::<Vec<_>>();
            let (_, added_a) = full
                .insert_residual::<true>(a, columns.iter().copied(), &mut diagram)
                .unwrap();
            let (_, added_b) = forward
                .insert_residual::<false>(b, columns.into_iter(), &mut diagram)
                .unwrap();
            assert_eq!(added_a, added_b);
        }
        assert_eq!(full.rows.row(full.owners[&0]).unwrap().get(2), ZERO);
        assert_eq!(forward.rows.row(forward.owners[&0]).unwrap().get(2), ONE);
        for active in [ONE, x, not_x] {
            for bits in 0..1usize << 6 {
                let query = BooleanRow::from_terms(
                    active,
                    (0..6)
                        .filter(|column| bits & (1 << column) != 0)
                        .map(|column| (column, ONE)),
                );
                let mut a = query.clone();
                let mut b = query;
                full.project(&mut a, &mut diagram).unwrap();
                forward.project(&mut b, &mut diagram).unwrap();
                assert_eq!(
                    (a.active, a.terms()),
                    (b.active, b.terms()),
                    "active={active:?}, bits={bits}"
                );
            }
        }
        let before = full
            .rows
            .decisions_mut()
            .map(|root| *root)
            .collect::<Vec<_>>();
        assert!(
            diagram.collect_garbage(
                full.rows
                    .decisions_mut()
                    .chain(forward.rows.decisions_mut())
            ) > 0
        );
        let after = full
            .rows
            .decisions_mut()
            .map(|root| *root)
            .collect::<Vec<_>>();
        assert!(before.iter().zip(after).any(|(old, new)| *old != new));
        for bits in 0..1usize << 6 {
            let query = BooleanRow::from_terms(
                ONE,
                (0..6)
                    .filter(|column| bits & (1 << column) != 0)
                    .map(|column| (column, ONE)),
            );
            let mut a = query.clone();
            let mut b = query;
            full.project(&mut a, &mut diagram).unwrap();
            forward.project(&mut b, &mut diagram).unwrap();
            assert_eq!(
                (a.active, a.terms()),
                (b.active, b.terms()),
                "after collection, bits={bits}"
            );
        }
    }

    #[test]
    fn unavailable_feedback_preserves_even_target_cancellations() {
        let graph = crate::GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let ports = graph
            .blocks()
            .filter(|block| block.kind().is_port())
            .take(2)
            .map(|block| crate::FeedbackTarget {
                pauli: PauliBasis::X,
                target: block.pos(),
                direction: None,
            })
            .collect::<Vec<_>>();
        let source = space.rows.clone();
        let mut paths = BTreeMap::new();
        for row in source {
            let left = space.feedback_coefficient(&row, &ports[..1]).unwrap() == ONE;
            let right = space.feedback_coefficient(&row, &ports[1..]).unwrap() == ONE;
            paths.entry((left, right)).or_insert(row);
        }
        let commuting = paths.remove(&(true, true)).unwrap_or_else(|| {
            let mut row = paths.remove(&(true, false)).unwrap();
            row.xor_scaled(&paths[&(false, true)], ONE, &mut space.topology.diagram)
                .unwrap();
            row
        });
        assert_eq!(
            space.feedback_coefficient(&commuting, &ports[..1]).unwrap(),
            ONE
        );
        for targets in [&ports[..], &[ports[0], ports[0]][..]] {
            assert_eq!(
                space.feedback_coefficient(&commuting, targets).unwrap(),
                ZERO
            );
            let mut terms = Vec::new();
            for target in targets {
                for variant in &space.local[&target.target.to_array()] {
                    terms.extend(
                        space
                            .feedback_support(target, variant)
                            .unwrap()
                            .into_iter()
                            .map(|column| (column, variant.guard)),
                    );
                }
            }
            let (_, available) = GuardedAvailability::new(
                vec![commuting.clone()],
                &[(AvailabilityKey::Feedback(0), terms)],
                2 * space.columns.len(),
                &mut space.topology.diagram,
            )
            .unwrap();
            assert_eq!(available.len(), 1);
            assert_eq!(available.rows().next().unwrap().terms(), commuting.terms());
        }
    }

    #[test]
    fn readout_reclamation_uses_known_names_under_partial_activation() {
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&crate::BlockGraph::new(), limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let guard = space.topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let absent = space.topology.diagram.negate(guard).unwrap();
        // This readout has no physical support: its only valid pivot names an
        // earlier corrected parity. Outside its activation retain that parity.
        let readout = BooleanRow::from_terms(guard, [(4, ONE), (5, ONE)]);
        let rows = vec![
            BooleanRow::from_terms(ONE, [(4, ONE), (5, ONE)]),
            BooleanRow::from_terms(ONE, [(0, ONE), (4, ONE), (6, ONE)]),
        ];
        let mut rows = BooleanRowSpace::new(rows, &mut space.topology.diagram).unwrap();
        space
            .reclaim_readouts(&mut rows, &mut GuardedPivots::default(), vec![readout])
            .unwrap();
        let next = rows.rows().find(|row| row.get(0) == ONE).unwrap();
        assert_eq!(next.get(4), absent);
        assert_eq!(next.get(5), guard);
        assert_eq!(next.get(6), ONE);
    }

    #[test]
    fn reclaimed_readouts_match_full_planner_on_every_reachable_selection() {
        use bloq_utils::boolean::BooleanDecisionDiagram;

        fn import(
            source: &BooleanDecisionDiagram,
            target: &mut BooleanDecisionDiagram,
        ) -> Vec<DecisionId> {
            let mut roots = vec![ZERO, ONE];
            for node in source.nodes() {
                roots.push(
                    target
                        .make_node(node.variable, roots[node.low.0], roots[node.high.0])
                        .unwrap(),
                );
            }
            roots
        }

        let conditional = crate::parse_blog_program_to_ast(include_str!(
            "../../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap();
        let programs = [
            crate::GalleryItem::THTH,
            crate::GalleryItem::TComparison,
            crate::GalleryItem::CCZInjectedAnd,
            crate::GalleryItem::CCZInjectedMaj,
            crate::GalleryItem::CCZGateTeleport,
            crate::GalleryItem::ToffoliFromAndDelayedCZ,
            crate::GalleryItem::ThreeBitAdder,
            crate::GalleryItem::PhaseGradientK4,
            crate::GalleryItem::OneDYoked,
            crate::GalleryItem::YMemory,
        ]
        .map(crate::GalleryItem::build)
        .into_iter()
        .chain([crate::lower_blog_graph_ast_deferred(&conditional).unwrap()]);
        for program in programs {
            let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
            let graph = linked.graph.fix_shadowed_faces();
            let limits = ModuleCertificationLimits::DEFAULT;
            let new_space = || {
                GuardedSurfaceSpace::new(
                    GuardedTopology::new(&graph, limits).unwrap(),
                    &linked.sites,
                    limits,
                )
                .unwrap()
            };
            let mut actual = new_space();
            actual.plan_causal_readouts::<true, true>().unwrap();
            expand_output_folds(&mut actual.surfaces, &mut actual.topology.diagram);
            let mut expected = new_space();
            expected.plan_causal_readouts::<false, false>().unwrap();
            assert_eq!(actual.columns, expected.columns);
            assert_eq!(actual.surfaces.len(), expected.surfaces.len());
            let mut diagram = BooleanDecisionDiagram::default();
            let actual_roots = import(&actual.topology.diagram, &mut diagram);
            let expected_roots = import(&expected.topology.diagram, &mut diagram);
            let domain = actual_roots[actual.topology.domain.0];
            assert_eq!(domain, expected_roots[expected.topology.domain.0]);
            let mut same = |left: DecisionId, right: DecisionId, label: &str| {
                let delta = diagram
                    .apply(
                        BooleanOp::Xor,
                        actual_roots[left.0],
                        expected_roots[right.0],
                    )
                    .unwrap();
                let reachable = diagram.apply(BooleanOp::And, domain, delta).unwrap();
                assert_eq!(reachable, ZERO, "{}: {label}", program.root().name);
            };
            for (actual, ordered_expected) in actual.surfaces.iter().zip(&expected.surfaces) {
                let expected = if let GuardedSurfaceKind::OutputFrame { port, basis, .. } =
                    &actual.kind
                {
                    expected.surfaces.iter().find(|surface| matches!(&surface.kind, GuardedSurfaceKind::OutputFrame { port: other, basis: other_basis, .. } if port == other && basis == other_basis)).unwrap()
                } else {
                    ordered_expected
                };
                match (&actual.kind, &expected.kind) {
                    (
                        GuardedSurfaceKind::Readout { name, folds },
                        GuardedSurfaceKind::Readout {
                            name: expected_name,
                            folds: expected_folds,
                        },
                    ) => {
                        assert_eq!(name, expected_name);
                        let folds = folds.iter().cloned().collect::<BTreeMap<_, _>>();
                        let expected_folds =
                            expected_folds.iter().cloned().collect::<BTreeMap<_, _>>();
                        for name in folds.keys().chain(expected_folds.keys()) {
                            same(
                                folds.get(name).copied().unwrap_or(ZERO),
                                expected_folds.get(name).copied().unwrap_or(ZERO),
                                "corrected readout fold",
                            );
                        }
                    }
                    (
                        GuardedSurfaceKind::OutputFrame { port, basis, .. },
                        GuardedSurfaceKind::OutputFrame {
                            port: expected_port,
                            basis: expected_basis,
                            ..
                        },
                    ) => assert_eq!((port, basis), (expected_port, expected_basis)),
                    (GuardedSurfaceKind::LogicalReadout, GuardedSurfaceKind::LogicalReadout) => {}
                    _ => panic!("{}: surface role changed", program.root().name),
                }
                same(actual.row.active, expected.row.active, "activation");
                for &(column, _) in actual.row.terms().iter().chain(expected.row.terms()) {
                    same(
                        actual.row.get(column),
                        expected.row.get(column),
                        "physical/named coefficient",
                    );
                }
                let feedbacks = actual.feedbacks.iter().copied().collect::<BTreeMap<_, _>>();
                let expected_feedbacks = expected
                    .feedbacks
                    .iter()
                    .copied()
                    .collect::<BTreeMap<_, _>>();
                for ordinal in feedbacks.keys().chain(expected_feedbacks.keys()) {
                    same(
                        feedbacks.get(ordinal).copied().unwrap_or(ZERO),
                        expected_feedbacks.get(ordinal).copied().unwrap_or(ZERO),
                        "source feedback coefficient",
                    );
                }
            }
        }
    }

    fn assert_injective_named_lift_matches_full_causal_adder_rows(bits: usize) {
        // The fixture crate depends on this crate; a BLOG round trip
        // brings its program into this test's crate instance.
        let source = bloq_test::benchmark::controlled_adder(bits);
        let program = crate::parse_inline_graph(&source.to_blog_text()).unwrap();
        let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
        let graph = linked.graph.fix_shadowed_faces();
        let limits = ModuleCertificationLimits::DEFAULT;
        let new_space = || {
            GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &linked.sites,
                limits,
            )
            .unwrap()
        };
        let mut actual = new_space();
        actual
            .plan_causal_readouts_with_fast_path::<true, true, true>()
            .unwrap();
        let mut expected = new_space();
        expected
            .plan_causal_readouts_with_fast_path::<true, true, false>()
            .unwrap();
        assert_eq!(actual.topology.domain, ONE, "{bits} bits");
        assert_eq!(expected.topology.domain, ONE, "{bits} bits");
        assert_same_raw_plan(&actual, &expected);
    }

    #[test]
    fn injective_named_lifts_match_full_causal_adder_rows() {
        for bits in [3, 10] {
            assert_injective_named_lift_matches_full_causal_adder_rows(bits);
        }
    }

    #[test]
    #[ignore = "larger exact causal row proof sweep"]
    fn injective_named_lifts_match_full_causal_adder_rows_large() {
        for bits in [20, 40] {
            assert_injective_named_lift_matches_full_causal_adder_rows(bits);
        }
    }

    #[test]
    fn injective_named_certificate_handles_guards_and_rejects_kernel() {
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&crate::BlockGraph::new(), limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        assert_eq!(space.topology.domain, ONE);
        let x = space.topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let not_x = space.topology.diagram.negate(x).unwrap();
        let names = [("m".to_owned(), 2)];
        let unknown = names.iter().collect::<Vec<_>>();
        let guarded = BooleanRowSpace::new(
            vec![
                BooleanRow::from_terms(x, [(0, ONE), (2, ONE)]),
                BooleanRow::from_terms(not_x, [(1, ONE), (2, ONE)]),
            ],
            &mut space.topology.diagram,
        )
        .unwrap();
        let fast = space
            .injective_named_candidates(&guarded, &[0, 1], &unknown, 2, 3)
            .unwrap()
            .unwrap()
            .remove(0)
            .unwrap();
        let full = space
            .reduce_named_rows::<true>(guarded, [2].into_iter())
            .unwrap();
        let expected = space.named_row(&full, 2).unwrap();
        assert_eq!(
            (fast.active, fast.terms()),
            (expected.active, expected.terms())
        );
        assert_eq!((fast.get(0), fast.get(1), fast.get(2)), (x, not_x, ONE));

        let kernel = BooleanRowSpace::new(
            vec![
                BooleanRow::from_terms(ONE, [(0, ONE), (2, ONE)]),
                BooleanRow::from_terms(ONE, [(1, ONE), (2, ONE)]),
            ],
            &mut space.topology.diagram,
        )
        .unwrap();
        assert!(
            space
                .injective_named_candidates(&kernel, &[0, 1], &unknown, 2, 3)
                .unwrap()
                .is_none()
        );

        let remaining = space.topology.diagram.limits().max_steps - space.topology.diagram.steps();
        space.topology.diagram.charge(remaining - 7).unwrap();
        let before = space.topology.diagram.steps();
        assert!(matches!(
            space.injective_named_candidates(&kernel, &[0, 1], &unknown, 2, 3),
            Err(BooleanResourceError {
                resource: "Boolean work steps",
                ..
            })
        ));
        assert!(space.topology.diagram.steps() > before);
    }

    #[test]
    fn persistent_causal_constraints_match_fresh_projection() {
        for program in [
            crate::GalleryItem::THTH.build(),
            crate::GalleryItem::CCZInjectedAnd.build(),
            crate::GalleryItem::ThreeBitAdder.build(),
        ] {
            let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
            let graph = linked.graph.fix_shadowed_faces();
            let limits = ModuleCertificationLimits::DEFAULT;
            let mut space = GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &linked.sites,
                limits,
            )
            .unwrap();
            let representative = space.topology.project(ONE).unwrap();
            let zx = ZXGraph::from_block_graph_for_analysis(&representative).unwrap();
            let physical_columns = 2 * space.columns.len();
            let mut source = space.rows.clone();
            // Independent witnesses compare the exact row transformations, not
            // merely their physical spans; named readout coordinates are linear
            // functions of these witnesses, including the crossing presentation.
            for (index, row) in source.iter_mut().enumerate() {
                row.set(physical_columns + index, ONE);
            }
            let columns = physical_columns + source.len();
            let selectors = space
                .topology
                .resolves
                .iter()
                .rev()
                .map(|(position, _)| position.to_array())
                .collect::<Vec<_>>();
            let mut persistent = source.clone();
            space.exclude_outputs(&mut persistent, &zx).unwrap();
            let mut filled = BTreeSet::new();
            for end in [0, selectors.len() / 2, selectors.len()] {
                let resolved = selectors[..end].iter().copied().collect::<BTreeSet<_>>();
                let newly_resolved = resolved.difference(&filled).copied().collect();
                space
                    .fill_selective(&mut persistent, &newly_resolved)
                    .unwrap();
                filled = resolved.clone();
                let mut fresh = source.clone();
                space.fill_selective(&mut fresh, &resolved).unwrap();
                space.exclude_outputs(&mut fresh, &zx).unwrap();
                let actual = reduce_boolean_rows(
                    persistent.clone(),
                    0..columns,
                    &mut space.topology.diagram,
                )
                .unwrap();
                let expected =
                    reduce_boolean_rows(fresh.clone(), 0..columns, &mut space.topology.diagram)
                        .unwrap();
                let rows = |rows: Vec<BooleanRow>| {
                    rows.into_iter()
                        .map(|row| (row.active, row.into_terms()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    rows(actual),
                    rows(expected),
                    "{}: {end} known selectors",
                    program.root().name
                );
                let names = (physical_columns..columns).step_by(3).collect::<Vec<_>>();
                let expected_named = reduce_boolean_rows(
                    fresh,
                    names.iter().copied().chain(0..columns),
                    &mut space.topology.diagram,
                )
                .unwrap()
                .into_iter()
                .filter(|row| names.iter().any(|&column| row.get(column) != ZERO))
                .collect();
                let remaining =
                    BooleanRowSpace::new(persistent.clone(), &mut space.topology.diagram).unwrap();
                let actual_named = space
                    .reduce_named_rows::<true>(remaining, names.iter().copied())
                    .unwrap()
                    .into_rows(&mut space.topology.diagram)
                    .unwrap();
                assert_eq!(
                    rows(actual_named),
                    rows(expected_named),
                    "{}: {end} selectors, named pivots",
                    program.root().name
                );
            }
        }
    }

    #[test]
    fn junction_node_feedback_matches_source_wire_on_every_crossing_path() {
        let source = crate::BlockGraph::from_blog_text(
            "BLOG 1.0
            0: Port [0,0,-1] role=input
            1: ZXZ [0,0,0]
            2: Port [0,0,1] role=output
            3: Port [1,0,0] role=output
            0 -> +Z
            1 -> +Z
            1 -> +X
            feedback Z 1
            ",
        )
        .unwrap();
        let mut directed = source.clone();
        let mut actions = directed.actions();
        let Action::Feedback { targets, .. } = &mut actions[0] else {
            unreachable!()
        };
        targets[0].direction = Some(crate::Direction::ZPLUS);
        directed.set_actions(actions).unwrap();
        #[cfg(feature = "verify")]
        {
            // The independent source interpreter inserts the Pauli on the
            // outgoing temporal wire. It never reads a reconstructed centre.
            let (_, expected) = crate::verify::LogicalVerifier::new(&directed)
                .unwrap()
                .instantiate(&BTreeMap::new())
                .unwrap();
            crate::verify::LogicalVerifier::new(&source)
                .unwrap()
                .verify(&expected.unwrap(), 1, 0)
                .unwrap();
        }
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&source, limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let zx = ZXGraph::from_block_graph_for_analysis(&source).unwrap();
        let node = zx.node_at(IVec3::ZERO).unwrap();
        assert_eq!(node.kind, NodeKind::X);
        let target = crate::FeedbackTarget {
            target: IVec3::ZERO,
            pauli: PauliBasis::Z,
            direction: None,
        };
        let mut coefficients = Vec::new();
        for pair in [
            [IVec3::Z, IVec3::X],
            [IVec3::X, -IVec3::Z],
            [-IVec3::Z, IVec3::Z],
        ] {
            let mut row = BooleanRow::default();
            for port in pair {
                for column in [
                    SurfaceColumn::Edge([0; 3], port.to_array()),
                    SurfaceColumn::Edge(port.to_array(), [0; 3]),
                ] {
                    row.set(2 * space.columns[&column], ONE);
                }
            }
            let materialized = space.materialize_row(&row, &zx, |root| root == ONE);
            assert!(zx.contains_stabilizer_support(&materialized));
            let expected = DecisionId(usize::from(pair.contains(&IVec3::Z)));
            let actual = space.feedback_coefficient(&row, &[target]).unwrap();
            assert_eq!(actual, expected, "crossing pair {pair:?}");
            assert_eq!(
                space
                    .feedback_coefficient(
                        &row,
                        &[crate::FeedbackTarget {
                            direction: Some(crate::Direction::ZPLUS),
                            ..target
                        }],
                    )
                    .unwrap(),
                expected,
            );
            coefficients.push(actual);
        }
        assert_eq!(
            space
                .topology
                .diagram
                .apply(BooleanOp::Xor, coefficients[0], coefficients[1])
                .unwrap(),
            coefficients[2],
        );
    }

    #[test]
    fn constant_feedback_work_respects_an_exhausted_budget() {
        let graph = crate::GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut space = GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &HashMap::new(),
            limits,
        )
        .unwrap();
        let target = graph
            .blocks()
            .find(|block| block.kind().is_port())
            .unwrap()
            .pos();
        let targets = [crate::FeedbackTarget {
            pauli: PauliBasis::X,
            target,
            direction: None,
        }];
        let mut row = BooleanRow::default();
        let support = space
            .feedback_support(&targets[0], &space.local[&target.to_array()][0])
            .unwrap();
        row.set(support[0], ONE);
        let actions = [Action::Feedback {
            targets: targets.to_vec(),
            condition: None,
        }];
        let columns = space.index_feedbacks(&actions).unwrap();
        assert_eq!(
            space.feedbacks::<true>(&row, &actions, &columns).unwrap(),
            vec![(0, ONE)]
        );
        assert!(space.topology.diagram.nodes().is_empty());
        let remaining = space.topology.diagram.limits().max_steps - space.topology.diagram.steps();
        space.topology.diagram.charge(remaining).unwrap();
        assert!(matches!(
            space.feedbacks::<true>(&row, &actions, &columns),
            Err(BooleanResourceError {
                resource: "Boolean work steps",
                ..
            })
        ));
        assert!(matches!(
            space.feedback_coefficient(&row, &targets),
            Err(BooleanResourceError {
                resource: "Boolean work steps",
                ..
            })
        ));
    }

    fn assert_terminal_boundaries(
        plan: &mut GuardedReadoutPlan,
        zx: &ZXGraph,
        outcome: bool,
        label: &str,
    ) {
        expand_output_folds(&mut plan.surfaces, &mut plan.topology.diagram);
        let scalar =
            super::super::RuntimeStabilizerBasis::for_output_frames(&zx.stabilizers().unwrap())
                .unwrap();
        let fills = plan
            .topology
            .resolves
            .iter()
            .map(|&(position, root)| {
                let NodeKind::Selective(kind) = zx.node_at(position).unwrap().kind else {
                    unreachable!()
                };
                let basis = if plan.topology.evaluate_source(root, |_| outcome) {
                    kind.pauli_if_true()
                } else {
                    kind.pauli_if_false()
                };
                (position, basis)
            })
            .collect::<Vec<_>>();
        let filled = scalar.apply_selective_fills(&fills).unwrap();
        let expected = filled
            .derive_output_correction_surfaces(&zx.output_ports())
            .unwrap();
        let actual = plan
            .surfaces
            .iter()
            .enumerate()
            .filter(|(_, recipe)| matches!(recipe.kind, GuardedSurfaceKind::OutputFrame { .. }))
            .filter_map(|(index, _)| {
                plan.materialize_surface(index, filled.zx_graph(), |_| outcome)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len(), "{label}, outcome={outcome}");
        for surface in actual {
            assert!(
                expected
                    .iter()
                    .any(|(_, reference)| reference.port_stabilizer == surface.port_stabilizer),
                "{label}, outcome={outcome}: terminal boundary {:?}",
                surface.port_stabilizer,
            );
        }
    }

    #[test]
    fn terminal_witnesses_match_scalar_input_and_resource_operators() {
        let conditional = crate::parse_blog_program_to_ast(
            r#"
BLOG 1.0
module main {
  in enable
  in a: data = 0
  in b: data = 1
  out a_out: data = 2
  out b_out: data = 3
  0: Port [0, 0, 0]
  1: Port [1, 0, 0]
  2: Port [0, 0, 2]
  3: Port [1, 0, 2]
  branch yoke {
    false {
      10: XZX [0, 0, 1]
      11: XZX [1, 0, 1]
      [0, 0, 0] -> +Z
      [0, 0, 1] -> +Z
      [1, 0, 0] -> +Z
      [1, 0, 1] -> +Z
    }
    true {
      12: XZX [0, 0, 1]
      13: XZX [1, 0, 1]
      [0, 0, 0] -> +Z
      [0, 0, 1] -> +Z
      [1, 0, 0] -> +Z
      [1, 0, 1] -> +Z
      [0, 0, 1] -> +X
    }
  }
  resolve yoke if enable
}
"#,
        )
        .unwrap();
        let conditional = crate::lower_blog_graph_ast_deferred(&conditional).unwrap();
        let conditional = crate::flatten_module_definition(&conditional, conditional.root(), "")
            .unwrap()
            .graph;
        for (label, graph) in [
            (
                "yoked",
                crate::GalleryItem::OneDYoked
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ),
            ("conditional yoke", conditional),
            (
                "T",
                crate::GalleryItem::T
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ),
            (
                "T with Y",
                crate::GalleryItem::TWithPreparedY
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection"),
            ),
        ] {
            let graph = graph.fix_shadowed_faces();
            let limits = ModuleCertificationLimits::DEFAULT;
            let mut space = GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &HashMap::new(),
                limits,
            )
            .unwrap();
            space.plan_causal_readouts::<true, true>().unwrap();
            assert_output_folds_match_dense(&mut space);
            let mut plan = space.into_readout_plan().unwrap();
            for outcome in [false, true] {
                let selected = graph
                    .project_branches_deferred(plan.topology.branches.iter().map(
                        |(_, target, root)| {
                            (*target, plan.topology.evaluate_source(*root, |_| outcome))
                        },
                    ))
                    .unwrap();
                let zx = ZXGraph::from_block_graph_for_analysis(&selected).unwrap();
                assert_terminal_boundaries(&mut plan, &zx, outcome, label);
            }
        }
    }

    #[test]
    fn input_normalization_preserves_prepared_resource_operators() {
        let graph = crate::GalleryItem::OneDYoked
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection")
            .fix_shadowed_faces();
        let mut prepared = graph.clone();
        prepared
            .set_block_kind(IVec3::ZERO, crate::BlockKind::T)
            .unwrap();
        let mut all_prepared = prepared.clone();
        for x in 1..6 {
            all_prepared
                .set_block_kind(IVec3::new(x, 0, 0), crate::BlockKind::T)
                .unwrap();
        }
        for turns in 0..4 {
            let rotate = |source: &crate::BlockGraph| {
                source
                    .rotate_about_origin(crate::UDirection::Z, turns)
                    .unwrap()
            };
            let graph = rotate(&graph);
            let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
            let boundaries = |source: &crate::BlockGraph| {
                let limits = ModuleCertificationLimits::DEFAULT;
                let mut plan = GuardedSurfaceSpace::new(
                    GuardedTopology::new(source, limits).unwrap(),
                    &HashMap::new(),
                    limits,
                )
                .unwrap()
                .plan_readouts()
                .unwrap();
                expand_output_folds(&mut plan.surfaces, &mut plan.topology.diagram);
                plan.surfaces
                    .iter()
                    .enumerate()
                    .filter(|(_, recipe)| {
                        matches!(recipe.kind, GuardedSurfaceKind::OutputFrame { .. })
                    })
                    .filter_map(|(index, _)| {
                        plan.materialize_surface(index, &zx, |_| unreachable!())
                    })
                    .map(|surface| surface.port_stabilizer)
                    .collect::<Vec<_>>()
            };
            let ordinary = boundaries(&graph);
            let protected = boundaries(&rotate(&prepared));
            // T and Port have identical local flow rows. With all inputs prepared,
            // no input normalization occurs. Both erased-input relations touch the
            // first resource, so preparing only that input must keep those same
            // witnesses. Reducing through either relation would change its operator.
            let unnormalized = boundaries(&rotate(&all_prepared));
            assert_eq!(protected.len(), 10, "{turns} quarter turns");
            assert_eq!(protected, unnormalized, "{turns} quarter turns");
            assert_ne!(ordinary, unnormalized, "{turns} quarter turns");
        }
    }

    #[test]
    fn guarded_planner_preserves_closed_logical_readouts() {
        for gallery in [
            crate::GalleryItem::XMemory,
            crate::GalleryItem::YMemory,
            crate::GalleryItem::Stability,
        ] {
            let source = gallery
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            for graph in [source.clone(), source.flip_xz_basis().unwrap()] {
                let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
                let expected = zx.stabilizers().unwrap();
                assert_eq!(expected.generators.len(), 1);
                let limits = ModuleCertificationLimits::DEFAULT;
                let topology = GuardedTopology::new(&graph, limits).unwrap();
                let space = GuardedSurfaceSpace::new(topology, &HashMap::new(), limits).unwrap();
                let space = space.plan_readouts().unwrap();
                assert_eq!(space.surfaces.len(), 1, "{gallery:?}");
                assert!(matches!(
                    space.surfaces[0].kind,
                    GuardedSurfaceKind::LogicalReadout
                ));
                let actual = space
                    .materialize_surface(0, &zx, |_| unreachable!())
                    .unwrap();
                assert_eq!(actual, expected.generators[0].stabilizer, "{gallery:?}");
            }
        }
    }

    #[test]
    fn conditional_composition_preserves_every_cz_strip_correlation() {
        let ast = crate::parse_blog_program_to_ast(include_str!(
            "../../../docs/fixtures/conditional_cz_strip.blog"
        ))
        .unwrap();
        let program = crate::lower_blog_graph_ast_deferred(&ast).unwrap();
        let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
        let topology = GuardedTopology::new(
            &linked.graph,
            ModuleCertificationLimits {
                max_guarded_domain_size: 64,
                ..ModuleCertificationLimits::DEFAULT
            },
        )
        .unwrap();
        let mut space =
            GuardedSurfaceSpace::new(topology, &linked.sites, ModuleCertificationLimits::DEFAULT)
                .unwrap();
        let mut branches = Vec::new();
        for mask in 0..8 {
            let variables = (0..3)
                .map(|index| (format!("enable{index}"), mask & (1 << index) != 0))
                .collect();
            let assignments = space
                .topology
                .branches
                .iter()
                .enumerate()
                .map(|(index, (_, target, _))| (*target, mask & (1 << index) != 0));
            let graph = linked.graph.project_branches_deferred(assignments).unwrap();
            let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
            let actual = super::super::modular::canonical_external_basis(
                space.evaluate(&zx, &variables),
                zx.total_ids(),
            );
            let expected = zx.to_external_generator_table_with_signed().1;
            assert_eq!(actual, expected, "mask {mask}");
            branches.push((zx, variables));
        }
        space.plan_causal_readouts::<true, false>().unwrap();
        assert_eq!(space.surfaces.len(), 8);
        assert!(
            space.rows.is_empty(),
            "planning consumes its working relation"
        );
        let before = space.topology.diagram.nodes().len();
        space.limits.max_witness_nodes = before;
        space
            .topology
            .diagram
            .make_node(usize::MAX, ZERO, ONE)
            .unwrap();
        space.check_readout_budget(std::iter::empty()).unwrap();
        assert_eq!(space.topology.diagram.nodes().len(), before);
        let space = space.into_readout_plan().unwrap();
        for (zx, variables) in branches {
            for surface in 0..space.surfaces.len() {
                assert!(
                    space
                        .materialize_surface(surface, &zx, |name| variables[name])
                        .is_some()
                );
            }
        }
    }

    #[test]
    fn guarded_planner_preserves_the_adders_causal_readout_order() {
        let program = crate::GalleryItem::ThreeBitAdder.build();
        let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
        let topology = GuardedTopology::new(
            &linked.graph,
            ModuleCertificationLimits {
                max_guarded_domain_size: 64,
                ..ModuleCertificationLimits::DEFAULT
            },
        )
        .unwrap();
        let space =
            GuardedSurfaceSpace::new(topology, &linked.sites, ModuleCertificationLimits::DEFAULT)
                .unwrap();
        let space = space.plan_readouts().unwrap();
        let mut known = BTreeSet::new();
        for surface in &space.surfaces {
            if let GuardedSurfaceKind::Readout { name, folds } = &surface.kind {
                assert!(folds.iter().all(|(name, _)| known.contains(name)));
                assert!(known.insert(name.clone()));
            }
        }
        assert_eq!(
            known.len(),
            linked
                .graph
                .actions()
                .iter()
                .filter(|action| matches!(action, Action::Measure { .. }))
                .count()
        );

        // Force collection inside planning, while the working basis and output
        // pivots are still live. Compare exact Boolean functions, not samples.
        const WITNESS_LIMIT: usize = 512;
        let mut collected = GuardedSurfaceSpace::new(
            GuardedTopology::new(&linked.graph, ModuleCertificationLimits::DEFAULT).unwrap(),
            &linked.sites,
            ModuleCertificationLimits {
                max_witness_nodes: WITNESS_LIMIT,
                ..ModuleCertificationLimits::DEFAULT
            },
        )
        .unwrap();
        for variable in 1000..1000 + 2 * WITNESS_LIMIT {
            collected
                .topology
                .diagram
                .make_node(variable, ZERO, ONE)
                .unwrap();
        }
        let mut collected = collected.plan_readouts().unwrap();
        assert!(collected.topology.diagram.nodes().len() <= WITNESS_LIMIT);
        assert_eq!(collected.columns, space.columns);
        let mut canonical = bloq_utils::boolean::BooleanDecisionDiagram::default();
        let mut import = |space: &GuardedReadoutPlan| {
            let mut ids = vec![ZERO, ONE];
            for node in space.topology.diagram.nodes() {
                ids.push(
                    canonical
                        .make_node(node.variable, ids[node.low.0], ids[node.high.0])
                        .unwrap(),
                );
            }
            ids
        };
        let before = import(&space);
        let after = import(&collected);
        let row = |row: &BooleanRow, ids: &[DecisionId]| {
            (
                ids[row.active.0],
                row.terms()
                    .iter()
                    .map(|&(col, id)| (col, ids[id.0]))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(space.surfaces.len(), collected.surfaces.len());
        for (left, right) in space.surfaces.iter().zip(&collected.surfaces) {
            assert_eq!(row(&left.row, &before), row(&right.row, &after));
            assert_eq!(
                left.feedbacks
                    .iter()
                    .map(|(ordinal, id)| (ordinal, before[id.0]))
                    .collect::<Vec<_>>(),
                right
                    .feedbacks
                    .iter()
                    .map(|(ordinal, id)| (ordinal, after[id.0]))
                    .collect::<Vec<_>>()
            );
            match (&left.kind, &right.kind) {
                (
                    GuardedSurfaceKind::Readout { name: a, folds: af },
                    GuardedSurfaceKind::Readout { name: b, folds: bf },
                ) => {
                    assert_eq!(a, b);
                    assert_eq!(
                        af.iter()
                            .map(|(name, id)| (name, before[id.0]))
                            .collect::<Vec<_>>(),
                        bf.iter()
                            .map(|(name, id)| (name, after[id.0]))
                            .collect::<Vec<_>>()
                    );
                }
                (
                    GuardedSurfaceKind::OutputFrame {
                        port: a,
                        basis: aa,
                        folds: af,
                    },
                    GuardedSurfaceKind::OutputFrame {
                        port: b,
                        basis: ba,
                        folds: bf,
                    },
                ) => {
                    assert_eq!((a, aa), (b, ba));
                    assert_eq!(
                        af.iter()
                            .map(|(port, basis, id)| (port, basis, before[id.0]))
                            .collect::<Vec<_>>(),
                        bf.iter()
                            .map(|(port, basis, id)| (port, basis, after[id.0]))
                            .collect::<Vec<_>>(),
                    );
                }
                (GuardedSurfaceKind::LogicalReadout, GuardedSurfaceKind::LogicalReadout) => {}
                _ => panic!("collection changed surface kind"),
            }
        }
        collected.max_witness_nodes = 0;
        let position = IVec3::from_array(*collected.local.first_key_value().unwrap().0);
        assert!(matches!(
            collected.local_cases(0, position, ONE),
            Err(BlockGraphError::Stabilizer(
                StabilizerError::ResourceLimited { limit: 0, .. }
            ))
        ));
    }
}
