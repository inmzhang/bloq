//! Frozen causal recipes and bounded local views for physical lowering.
//!
//! Planning owns the working relation and may collect Boolean decisions.
//! This stage retains only recipes and local incidence; physical consumers may
//! now hold decision ids. Dense ZX rows are materialized only for diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bloq_utils::boolean::{BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId};
use glam::IVec3;

use super::guarded_surfaces::{GuardedSurface, GuardedSurfaceKind, SurfaceColumn};
use super::{NodeKind, Stabilizer, StabilizerGenerator, StabilizerRowKind, ZXGraph};
use crate::{BlockGraphError, GuardedTopology, Pauli, StabilizerError};

pub(super) type LocalQueryCache =
    crate::FxHashMap<(DecisionId, DecisionId, Vec<DecisionId>), Arc<[(Vec<bool>, DecisionId)]>>;

#[derive(Clone, Copy)]
enum LocalBit {
    Constant(bool),
    Query(usize),
}

impl GuardedReadoutPlan {
    /// Derive source action prerequisites from the frozen causal recipes.
    /// Local cases preserve support and endpoint frames without listing joint
    /// branch assignments. This diagnostic consumes the same bounded queries as
    /// physical binding and never collects while its decision ids are live.
    pub(crate) fn action_dependencies(
        &mut self,
    ) -> Result<Vec<(usize, usize, crate::ActionDependency)>, BlockGraphError> {
        use crate::{Action, ActionDependency, GuardedVariable};

        let actions = self.topology.source.actions();
        let names = actions
            .iter()
            .enumerate()
            .filter_map(|(ordinal, action)| match action {
                Action::Measure { name, .. } | Action::Let { name, .. } => {
                    Some((name.clone(), ordinal))
                }
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let resolves = actions
            .iter()
            .enumerate()
            .filter_map(|(ordinal, action)| match action {
                Action::Resolve { target, .. } => Some((target.to_array(), ordinal)),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let branches = self.topology.branches.iter().filter_map(|(name, target, _)| {
            actions.iter().position(|action| matches!(action, Action::Branch { target: site, .. } if site == target))
                .map(|ordinal| (name.clone(), ordinal))
        }).collect::<BTreeMap<_, _>>();
        let mut owners = BTreeMap::new();
        for region in self.topology.source.branch_regions()? {
            for block in region.on_true().blocks().chain(region.on_false().blocks()) {
                owners.insert(block.pos().to_array(), branches[&region.name]);
            }
        }
        let mut edges = Vec::new();
        for index in 0..self.surfaces.len() {
            let GuardedSurfaceKind::Readout { name, folds } = self.surfaces[index].kind.clone()
            else {
                continue;
            };
            let consumer = names[&name];
            let mut roots = BTreeSet::from([self.surfaces[index].row.active]);
            self.topology
                .diagram
                .charge(self.surfaces[index].row.terms().len())?;
            roots.extend(
                self.surfaces[index]
                    .row
                    .terms()
                    .iter()
                    .map(|&(_, root)| root),
            );
            for (predecessor, guard) in folds {
                if self
                    .topology
                    .diagram
                    .constrain(guard, self.topology.domain)?
                    != ZERO
                {
                    edges.push((
                        names[&predecessor],
                        consumer,
                        ActionDependency::ReadoutParity,
                    ));
                    roots.insert(guard);
                }
            }
            for (ordinal, guard) in self.surfaces[index].feedbacks.clone() {
                if self
                    .topology
                    .diagram
                    .constrain(guard, self.topology.domain)?
                    != ZERO
                {
                    edges.push((
                        ordinal as usize,
                        consumer,
                        ActionDependency::FeedbackAnticommutation,
                    ));
                    roots.insert(guard);
                }
            }
            for position in self.support_sites(index) {
                for case in self.local_cases(index, IVec3::from_array(position), ONE)? {
                    roots.insert(case.guard);
                    // Even a constant selector must choose the physical owner
                    // before a record in that branch can be read.
                    if let Some(&predecessor) = owners.get(&position) {
                        edges.push((predecessor, consumer, ActionDependency::BranchSupport));
                    }
                    if case.selective.is_some()
                        && let Some(&predecessor) = resolves.get(&position)
                    {
                        edges.push((predecessor, consumer, ActionDependency::SelectiveSupport));
                    }
                }
            }
            for root in roots {
                self.topology.diagram.charge(1)?;
                if root == ZERO || root == ONE {
                    continue;
                }
                let root = self
                    .topology
                    .diagram
                    .constrain(root, self.topology.domain)?;
                if root == ZERO || root == ONE {
                    continue;
                }
                self.topology
                    .diagram
                    .charge(self.topology.diagram.nodes().len())?;
                for variable in self.topology.diagram.variables(root) {
                    let (predecessor, reason) = match &self.topology.variables[variable] {
                        GuardedVariable::Branch { name, .. } => {
                            (branches.get(name), ActionDependency::BranchSupport)
                        }
                        GuardedVariable::Outcome(name) => {
                            (names.get(name), ActionDependency::ReadoutParity)
                        }
                    };
                    if let Some(&predecessor) = predecessor {
                        edges.push((predecessor, consumer, reason));
                    }
                }
            }
        }
        edges.sort_by_key(|&(from, to, reason)| (from, to, reason as u8));
        edges.dedup();
        Ok(edges)
    }
}

/// Position-based Pauli support consumed by physical gateways. Edge values use
/// the frame of their first endpoint; a reverse lookup is deliberately distinct.
/// Each directed endpoint pair occurs once.
/// Neither a dense ZX id space nor a global stabilizer sign is required here.
#[doc(hidden)]
pub trait SurfaceSupport {
    fn node_pauli(&self, position: IVec3) -> Pauli;
    fn port_pauli(&self, position: IVec3) -> Pauli;
    fn edge_pauli(&self, source: IVec3, target: IVec3) -> Option<Pauli>;
    fn edges(&self) -> impl Iterator<Item = ((IVec3, IVec3), Pauli)>;

    /// Directed edges owned by sorted source positions, or every edge if absent.
    fn owned_edges(
        &self,
        positions: Option<&[IVec3]>,
    ) -> impl Iterator<Item = ((IVec3, IVec3), Pauli)> {
        self.edges().filter(move |((source, _), _)| {
            positions.is_none_or(|positions| {
                positions
                    .binary_search_by_key(&source.to_array(), IVec3::to_array)
                    .is_ok()
            })
        })
    }
}

impl SurfaceSupport for Stabilizer {
    fn node_pauli(&self, position: IVec3) -> Pauli {
        self.interior_nodes
            .get(&position)
            .copied()
            .unwrap_or(Pauli::I)
    }

    fn port_pauli(&self, position: IVec3) -> Pauli {
        self.port_stabilizer
            .get(&position)
            .copied()
            .unwrap_or(Pauli::I)
    }

    fn edge_pauli(&self, source: IVec3, target: IVec3) -> Option<Pauli> {
        self.interior_edges.get(&(source, target)).copied()
    }

    fn edges(&self) -> impl Iterator<Item = ((IVec3, IVec3), Pauli)> {
        self.interior_edges
            .iter()
            .map(|(&edge, &pauli)| (edge, pauli))
    }
}

/// One center and its incident edges, with no allocation proportional to the
/// enclosing graph. The center's nonlinear crossing component is reconstructed
/// from these arms after evaluating the final symbolic row.
#[doc(hidden)]
#[derive(Debug)]
pub struct LocalPauliSurface {
    position: IVec3,
    center: Pauli,
    port: bool,
    edges: Vec<((IVec3, IVec3), Pauli)>,
}

impl SurfaceSupport for LocalPauliSurface {
    fn node_pauli(&self, position: IVec3) -> Pauli {
        if position == self.position {
            self.center
        } else {
            Pauli::I
        }
    }

    fn port_pauli(&self, position: IVec3) -> Pauli {
        if self.port {
            self.node_pauli(position)
        } else {
            Pauli::I
        }
    }

    fn edge_pauli(&self, source: IVec3, target: IVec3) -> Option<Pauli> {
        self.edges
            .iter()
            .find(|(edge, _)| *edge == (source, target))
            .map(|(_, pauli)| *pauli)
    }

    fn edges(&self) -> impl Iterator<Item = ((IVec3, IVec3), Pauli)> {
        self.edges.iter().copied()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LocalArm {
    pub column: usize,
    pub target: IVec3,
    pub reversed: bool,
    pub hadamard: bool,
}

#[derive(Debug)]
pub(super) struct LocalTopology {
    pub guard: DecisionId,
    pub kind: NodeKind,
    /// Only an armless variant needs an independent center coordinate.
    pub center: Option<usize>,
    pub arms: Box<[LocalArm]>,
    pub selector: Option<DecisionId>,
}

impl LocalTopology {
    fn columns(&self) -> impl Iterator<Item = usize> + '_ {
        self.center
            .into_iter()
            .chain(self.arms.iter().map(|arm| arm.column))
    }
}

/// A completed readout plan. Construction consumes the correlation space, so
/// an unplanned relation cannot reach physical binding or be planned twice.
#[doc(hidden)]
#[derive(Debug)]
pub struct GuardedReadoutPlan {
    pub topology: GuardedTopology,
    pub surfaces: Vec<GuardedSurface>,
    pub(super) columns: Vec<SurfaceColumn>,
    pub(super) local: BTreeMap<[i32; 3], Vec<LocalTopology>>,
    pub(super) max_witness_nodes: usize,
    // Cleared whenever the owning physical stage jointly remaps its live ids.
    pub(super) query_cache: LocalQueryCache,
    pub(super) query_cache_terms: usize,
    // Stable ids reference complete source rows, so collection needs no second
    // coefficient inventory. Interner keys are discarded before binding.
    pub(super) fragments: Vec<(usize, Arc<[[i32; 3]]>)>,
    pub(super) fragment_uses: Vec<Vec<usize>>,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct GuardedLocalSurface {
    pub guard: DecisionId,
    pub surface: LocalPauliSurface,
    pub selective: Option<bool>,
}

impl GuardedReadoutPlan {
    pub(super) fn share_module_queries(
        &mut self,
        module_sites: &BTreeMap<[i32; 3], usize>,
    ) -> Result<(), BlockGraphError> {
        self.topology.diagram.charge(self.columns.len())?;
        let column_modules = self
            .columns
            .iter()
            .map(|column| {
                let (SurfaceColumn::Node(position) | SurfaceColumn::Edge(position, _)) = column;
                module_sites[position]
            })
            .collect::<Vec<_>>();
        let mut shared = crate::FxHashMap::default();
        let mut retained = 0usize;
        for (surface, query) in self.surfaces.iter().enumerate() {
            self.topology.diagram.charge(query.row.terms().len())?;
            let mut modules = crate::FxHashMap::<usize, Vec<_>>::default();
            if query.row.active != ZERO {
                for &(column, coefficient) in query.row.terms() {
                    let Some(&owner) = column_modules.get(column / 2) else {
                        continue;
                    };
                    modules
                        .entry(owner)
                        .or_default()
                        .push((column, coefficient));
                }
            }
            let mut uses = Vec::new();
            let mut modules = modules.into_iter().collect::<Vec<_>>();
            modules.sort_unstable_by_key(|(owner, _)| *owner);
            for (_, terms) in modules {
                // Include activation and gateway role, all native half-edge
                // coefficients, and the actual instance's columns. Missing
                // coefficients are zero. Never equate output-only summaries.
                let key = (
                    query.row.active,
                    matches!(query.kind, super::GuardedSurfaceKind::Readout { .. }),
                    terms,
                );
                self.topology.diagram.charge(key.2.len())?;
                let id = if let Some(&id) = shared.get(&key) {
                    id
                } else {
                    let positions =
                        key.2
                            .iter()
                            .map(|&(column, _)| match self.columns[column / 2] {
                                SurfaceColumn::Node(position)
                                | SurfaceColumn::Edge(position, _) => position,
                            })
                            .collect::<BTreeSet<_>>();
                    self.topology.diagram.charge(positions.len())?;
                    let id = self.fragments.len();
                    self.fragments
                        .push((surface, positions.into_iter().collect()));
                    // ponytail: cap optional interning; uncommon oversized
                    // fragments still bind exactly, without cache admission.
                    const MAX_SHARED_TERMS: usize = 262_144;
                    let terms = key.2.len().saturating_add(1);
                    if terms <= MAX_SHARED_TERMS.saturating_sub(retained) {
                        retained += terms;
                        shared.insert(key, id);
                    }
                    id
                };
                uses.push(id);
            }
            self.fragment_uses.push(uses);
        }
        Ok(())
    }

    /// Shared module fragments of this complete source query. Each id retains
    /// exact coefficients, activation, and physical instance identity.
    ///
    /// # Panics
    ///
    /// Panics if `surface` is not part of this plan.
    pub fn query_fragments(&self, surface: usize) -> &[usize] {
        &self.fragment_uses[surface]
    }

    /// Representative source query and supported sites of one shared fragment.
    /// Corrected-parity folds remain on the complete query; physical binding
    /// supplies any gateway transport offsets.
    ///
    /// # Panics
    ///
    /// Panics if `fragment` is not part of this plan.
    pub fn query_fragment(&self, fragment: usize) -> (usize, Arc<[[i32; 3]]>) {
        self.fragments[fragment].clone()
    }

    /// Release raw coefficients after the physical consumer finishes every
    /// direct and fragment use. Activation and decoder metadata remain live.
    ///
    /// # Errors
    /// Returns an error if accounting exceeds its cumulative work limit.
    ///
    /// # Panics
    /// Panics if `surface` is not part of this plan.
    pub fn release_query_coefficients(&mut self, surface: usize) -> Result<(), BlockGraphError> {
        let row = &mut self.surfaces[surface].row;
        self.topology
            .diagram
            .charge(row.terms().len().saturating_add(1))?;
        *row = bloq_utils::boolean::BooleanRow::new(row.active);
        Ok(())
    }

    /// Charged collection work for this plan's arena, tape and owned handles.
    /// Caller-owned handles must be charged separately before collection.
    pub fn collection_work(&mut self) -> usize {
        self.topology
            .diagram
            .nodes()
            .len()
            .saturating_add(self.topology.diagram.witness_collection_work())
            .saturating_add(self.topology.decision_root_count())
            .saturating_add(
                self.surfaces
                    .iter_mut()
                    .flat_map(GuardedSurface::decisions_mut)
                    .count(),
            )
            .saturating_add(
                self.local
                    .values()
                    .flatten()
                    .map(|local| 1 + usize::from(local.selector.is_some()))
                    .sum::<usize>(),
            )
            .saturating_add(self.query_cache_terms)
    }

    /// Remap frozen recipes together with every caller-owned decision id.
    /// Call only between local queries, after their returned cases are consumed.
    /// Optional query images are discarded rather than retaining stale cache keys.
    pub fn collect_garbage<'a>(
        &'a mut self,
        additional: impl IntoIterator<Item = &'a mut DecisionId>,
    ) -> usize {
        self.query_cache.clear();
        self.query_cache_terms = 0;
        let mut roots = Vec::new();
        for surface in &mut self.surfaces {
            roots.extend(surface.decisions_mut());
        }
        for local in self.local.values_mut().flatten() {
            roots.push(&mut local.guard);
            roots.extend(local.selector.iter_mut());
        }
        self.topology
            .collect_garbage(roots.into_iter().chain(additional))
    }

    /// Only the requested site's Pauli pattern is materialized. Guards may
    /// still depend on any number of source outcomes. Uniform raw support
    /// excludes its identity branch before partitioning.
    ///
    /// # Errors
    ///
    /// Returns an error if Boolean evaluation or local materialization exceeds its limits.
    ///
    /// # Panics
    ///
    /// Panics if `surface` or `position` is not part of this plan.
    pub fn local_cases(
        &mut self,
        surface: usize,
        position: IVec3,
        enabled: DecisionId,
    ) -> Result<Vec<GuardedLocalSurface>, BlockGraphError> {
        let row = &self.surfaces[surface].row;
        let mut output = Vec::new();
        for variant in &self.local[&position.to_array()] {
            let mut gate = self
                .topology
                .diagram
                .apply(BooleanOp::And, enabled, variant.guard)?;
            gate = self
                .topology
                .diagram
                .apply(BooleanOp::And, gate, row.active)?;
            if gate == ZERO {
                continue;
            }
            let column_count = usize::from(variant.center.is_some()) + variant.arms.len();
            self.topology
                .diagram
                .charge(2 * column_count + usize::from(variant.selector.is_some()))?;
            let mut support = ZERO;
            let mut roots = Vec::new();
            let mut coordinates =
                Vec::with_capacity(2 * column_count + usize::from(variant.selector.is_some()));
            let mut push_root = |root| {
                let bit = match root {
                    ZERO => LocalBit::Constant(false),
                    ONE => LocalBit::Constant(true),
                    root => {
                        let index = roots
                            .iter()
                            .position(|&existing| existing == root)
                            .unwrap_or_else(|| {
                                roots.push(root);
                                roots.len() - 1
                            });
                        LocalBit::Query(index)
                    }
                };
                coordinates.push(bit);
            };
            for column in variant.columns() {
                for root in [row.get(2 * column), row.get(2 * column + 1)] {
                    if root != ZERO && support != ONE {
                        support = if support == ZERO || support == root {
                            root
                        } else {
                            ONE
                        };
                    }
                    push_root(root);
                }
            }
            // An all-I local surface contributes no physical records, boundary
            // operators, or transport sign. A shared raw root gates every
            // nonzero Pauli bit; mixed roots keep their original query.
            if support == ZERO {
                continue;
            }
            if support != ONE {
                gate = self.topology.diagram.apply(BooleanOp::And, gate, support)?;
                if gate == ZERO {
                    continue;
                }
            }
            if let Some(selector) = variant.selector {
                push_root(selector);
            }
            let key = (self.topology.domain, gate, roots);
            let cases = if let Some(cases) = self.query_cache.get(&key) {
                Arc::clone(cases)
            } else {
                let cases: Arc<[(Vec<bool>, DecisionId)]> =
                    self.topology.cases(&key.2, gate)?.into();
                let terms = key.2.len()
                    + cases
                        .iter()
                        .map(|(values, _)| values.len() + 1)
                        .sum::<usize>();
                // ponytail: bound retained coordinates; clear-on-full can be
                // replaced by eviction if measured query churn warrants it.
                const MAX_CACHED_TERMS: usize = 262_144;
                if terms <= MAX_CACHED_TERMS {
                    if self.query_cache_terms + terms > MAX_CACHED_TERMS {
                        self.query_cache.clear();
                        self.query_cache_terms = 0;
                    }
                    self.query_cache.insert(key, Arc::clone(&cases));
                    self.query_cache_terms += terms;
                }
                cases
            };
            self.topology.diagram.charge(
                cases
                    .len()
                    .saturating_mul(coordinates.len().saturating_add(1)),
            )?;
            for (values, guard) in cases.iter() {
                let bit = |index| match coordinates[index] {
                    LocalBit::Constant(value) => value,
                    LocalBit::Query(index) => values[index],
                };
                // The first arm determines the linear center component. An
                // isolated variant instead queries its independent coordinate.
                let mut center = pauli(bit(0), bit(1));
                let paulis = (usize::from(variant.center.is_some())..column_count)
                    .map(|index| pauli(bit(2 * index), bit(2 * index + 1)));
                let cross = match variant.kind {
                    NodeKind::X | NodeKind::Z => Some(variant.kind.cross_pauli()),
                    _ => None,
                };
                let mut crosses = false;
                let mut edges = Vec::with_capacity(variant.arms.len());
                for (arm, mut value) in variant.arms.iter().zip(paulis) {
                    crosses |= cross.is_some_and(|cross| value & cross);
                    if value == Pauli::I {
                        continue;
                    }
                    let endpoints = if arm.reversed {
                        if arm.hadamard {
                            value = value.flip();
                        }
                        (arm.target, position)
                    } else {
                        (position, arm.target)
                    };
                    // Parallel adjacency repeats the same resolved edge and
                    // coefficient. Match the full stabilizer's unique edge map.
                    if !edges.iter().any(|(edge, _)| *edge == endpoints) {
                        edges.push((endpoints, value));
                    }
                }
                if let Some(cross) = cross {
                    // Match exact ZX reconstruction: clear stale crossing
                    // centers even when the last supported arm cancelled.
                    if center & cross {
                        center = center ^ cross;
                    }
                    if crosses {
                        center = center | cross;
                    }
                }
                output.push(GuardedLocalSurface {
                    guard: *guard,
                    surface: LocalPauliSurface {
                        position,
                        center,
                        port: variant.kind.is_port(),
                        edges,
                    },
                    selective: variant.selector.map(|_| bit(2 * column_count)),
                });
            }
        }
        let observed = self.topology.diagram.nodes().len();
        if observed > self.max_witness_nodes {
            return Err(StabilizerError::ResourceLimited {
                phase: "guarded readout coefficients",
                observed,
                limit: self.max_witness_nodes,
            }
            .into());
        }
        Ok(output)
    }

    /// Visit nonzero row coefficients instead of scanning the global column
    /// table. Auxiliary named-readout coordinates are not physical support.
    pub fn support_sites(&self, surface: usize) -> BTreeSet<[i32; 3]> {
        let row = &self.surfaces[surface].row;
        if row.active == ZERO {
            return BTreeSet::new();
        }
        row.terms()
            .iter()
            .filter_map(|&(column, _)| {
                self.columns.get(column / 2).map(|column| match *column {
                    SurfaceColumn::Node(position) | SurfaceColumn::Edge(position, _) => position,
                })
            })
            .collect()
    }

    /// Materialize a frozen recipe for an observed source branch. Diagnostic
    /// callers supply the independent ZX graph and its signed relation.
    ///
    /// # Panics
    ///
    /// Panics if `surface` is out of range or `zx` does not match this plan.
    pub fn materialize_surface(
        &self,
        surface: usize,
        zx: &ZXGraph,
        outcome: impl Fn(&str) -> bool,
    ) -> Option<Stabilizer> {
        self.materialize_surface_with(surface, zx, |bit| {
            self.topology.evaluate_source(bit, &outcome)
        })
    }

    /// Materialize every surface active under one complete assignment of the
    /// topology variables, such as [`GuardedTopology::branch_selection_values`].
    /// Named readouts keep their names; output frames and closed surfaces are
    /// logical rows. Identity rows are omitted.
    pub(crate) fn materialize_surfaces(
        &self,
        zx: &ZXGraph,
        values: &[bool],
    ) -> Vec<StabilizerGenerator> {
        let evaluate = |bit| self.topology.diagram.evaluate(bit, |index| values[index]);
        (0..self.surfaces.len())
            .filter_map(|surface| {
                let stabilizer = self.materialize_surface_with(surface, zx, evaluate)?;
                let kind = match &self.surfaces[surface].kind {
                    GuardedSurfaceKind::Readout { name, .. } => {
                        StabilizerRowKind::Measurement { name: name.clone() }
                    }
                    GuardedSurfaceKind::LogicalReadout | GuardedSurfaceKind::OutputFrame { .. } => {
                        StabilizerRowKind::Logical
                    }
                };
                (!stabilizer.paulis.is_identity())
                    .then(|| StabilizerGenerator::new(stabilizer, kind))
            })
            .collect()
    }

    fn materialize_surface_with(
        &self,
        surface: usize,
        zx: &ZXGraph,
        evaluate: impl Fn(DecisionId) -> bool,
    ) -> Option<Stabilizer> {
        let row = &self.surfaces[surface].row;
        evaluate(row.active).then(|| {
            let paulis = super::guarded_surfaces::materialize_row(
                self.columns
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, column)| (column, index)),
                row,
                zx,
                evaluate,
            );
            assert!(
                zx.contains_stabilizer_support(&paulis),
                "frozen surface {surface} must belong to the signed branch relation"
            );
            zx.materialize_stabilizer(paulis)
        })
    }
}

fn pauli(x: bool, z: bool) -> Pauli {
    match (x, z) {
        (false, false) => Pauli::I,
        (true, false) => Pauli::X,
        (false, true) => Pauli::Z,
        (true, true) => Pauli::Y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PauliString;
    use crate::{GalleryItem, GuardedSurfaceSpace, ModuleCertificationLimits};
    use bloq_utils::boolean::DECISION_TRUE as ONE;

    #[test]
    fn constant_action_dependencies_do_not_scan_unrelated_boolean_nodes() {
        let source = crate::BlockGraph::from_text(
            "BLOG 1.0\nmodule main {\n0: ZXZ [0,0,0]\nm = measure 0\n}",
        )
        .unwrap()
        .flatten()
        .unwrap();
        let limits = ModuleCertificationLimits::DEFAULT;
        let plan = || {
            GuardedSurfaceSpace::new(
                GuardedTopology::new(&source, limits).unwrap(),
                &Default::default(),
                limits,
            )
            .unwrap()
            .plan_readouts()
            .unwrap()
        };
        let mut small = plan();
        let mut large = plan();
        for variable in 100..200 {
            large
                .topology
                .diagram
                .make_node(variable, ZERO, ONE)
                .unwrap();
        }
        let before_small = small.topology.diagram.steps();
        let before_large = large.topology.diagram.steps();
        assert_eq!(
            small.action_dependencies().unwrap(),
            large.action_dependencies().unwrap()
        );
        assert_eq!(
            small.topology.diagram.steps() - before_small,
            large.topology.diagram.steps() - before_large,
        );
    }

    #[test]
    fn uniform_support_gate_only_removes_raw_identity_cases() {
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut pruned = 0;
        let mut repeated = 0;
        let mut constants = 0;
        for item in [GalleryItem::THTH, GalleryItem::ThreeBitAdder] {
            let program = item.build();
            let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
            let graph = linked.graph.fix_shadowed_faces();
            let mut plan = GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &linked.sites,
                limits,
            )
            .unwrap()
            .plan_readouts()
            .unwrap();
            for surface in 0..plan.surfaces.len() {
                for position in plan
                    .support_sites(surface)
                    .into_iter()
                    .map(IVec3::from_array)
                {
                    let row = &plan.surfaces[surface].row;
                    let mut expected = Vec::new();
                    for variant in &plan.local[&position.to_array()] {
                        let gate = plan
                            .topology
                            .diagram
                            .apply(BooleanOp::And, ONE, variant.guard)
                            .and_then(|gate| {
                                plan.topology
                                    .diagram
                                    .apply(BooleanOp::And, gate, row.active)
                            })
                            .unwrap();
                        if gate == ZERO {
                            continue;
                        }
                        let mut roots = variant
                            .columns()
                            .flat_map(|column| [row.get(2 * column), row.get(2 * column + 1)])
                            .collect::<Vec<_>>();
                        let raw_len = roots.len();
                        repeated += usize::from(roots.iter().enumerate().any(|(index, root)| {
                            *root != ZERO && *root != ONE && roots[..index].contains(root)
                        }));
                        constants += usize::from(roots.contains(&ZERO) || roots.contains(&ONE));
                        let support = roots.iter().copied().find(|&root| root != ZERO);
                        let uniform = support.is_none_or(|support| {
                            roots.iter().all(|&root| root == ZERO || root == support)
                        });
                        roots.extend(variant.selector);
                        for (values, guard) in plan.topology.cases(&roots, gate).unwrap() {
                            if uniform && !values[..raw_len].contains(&true) {
                                pruned += 1;
                            } else {
                                expected.push(guard);
                            }
                        }
                    }
                    expected.sort_unstable();
                    let mut actual = plan
                        .local_cases(surface, position, ONE)
                        .unwrap()
                        .into_iter()
                        .map(|case| case.guard)
                        .collect::<Vec<_>>();
                    actual.sort_unstable();
                    assert_eq!(actual, expected);
                }
            }
        }
        assert!(pruned > 0);
        assert!(repeated > 0);
        assert!(constants > 0);
    }

    #[test]
    fn repeated_local_queries_share_images_and_keep_activation_separate() {
        let program = GalleryItem::THTH.build();
        let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
        let graph = linked.graph.fix_shadowed_faces();
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut plan = GuardedSurfaceSpace::new(
            GuardedTopology::new(&graph, limits).unwrap(),
            &linked.sites,
            limits,
        )
        .unwrap()
        .plan_readouts()
        .unwrap();
        let (surface, position) = (0..plan.surfaces.len())
            .find_map(|surface| {
                plan.support_sites(surface)
                    .into_iter()
                    .next()
                    .map(|position| (surface, IVec3::from_array(position)))
            })
            .unwrap();
        let signature = |cases: Vec<GuardedLocalSurface>| {
            cases
                .into_iter()
                .map(|case| {
                    (
                        case.guard,
                        case.surface.center,
                        case.surface.port,
                        case.surface.edges,
                        case.selective,
                    )
                })
                .collect::<Vec<_>>()
        };
        let before = plan.topology.diagram.steps();
        let cold = signature(plan.local_cases(surface, position, ONE).unwrap());
        let cold_work = plan.topology.diagram.steps() - before;
        let before = plan.topology.diagram.steps();
        let warm = signature(plan.local_cases(surface, position, ONE).unwrap());
        assert_eq!(cold, warm);
        assert!(plan.topology.diagram.steps() - before < cold_work);
        assert!(!plan.query_cache.is_empty());

        assert!(!plan.topology.variables.is_empty());
        let selected = plan.topology.diagram.make_node(0, ZERO, ONE).unwrap();
        let unselected = plan.topology.diagram.negate(selected).unwrap();
        for (enabled, excluded) in [(selected, unselected), (unselected, selected)] {
            for case in plan.local_cases(surface, position, enabled).unwrap() {
                assert_eq!(
                    plan.topology
                        .diagram
                        .apply(BooleanOp::And, case.guard, excluded)
                        .unwrap(),
                    ZERO
                );
            }
        }
        assert_eq!(
            cold,
            signature(plan.local_cases(surface, position, ONE).unwrap())
        );
    }

    #[test]
    fn local_views_match_dense_crossing_reconstruction() {
        let conditional = crate::lower_blog_graph_ast_deferred(
            &crate::parse_blog_program_to_ast(include_str!(
                "../../../docs/fixtures/conditional_cz_strip.blog"
            ))
            .unwrap(),
        )
        .unwrap();
        let programs = [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::THTH,
            GalleryItem::GHZSlideThenGlide,
            GalleryItem::GHZPatchRotations,
            GalleryItem::CCZInjectedAnd,
            GalleryItem::Stability,
        ]
        .into_iter()
        .map(GalleryItem::build)
        .chain([conditional]);
        let limits = ModuleCertificationLimits::DEFAULT;
        for program in programs {
            let linked = crate::flatten_module_definition(&program, program.root(), "").unwrap();
            let graph = linked.graph.fix_shadowed_faces();
            let relation = GuardedSurfaceSpace::new(
                GuardedTopology::new(&graph, limits).unwrap(),
                &linked.sites,
                limits,
            )
            .unwrap();
            let columns = relation.columns.clone();
            let local = relation.local.clone();
            let mut plan = relation.plan_readouts().unwrap();
            // CSR adjacency may repeat an endpoint; every occurrence resolves
            // to that pair's last edge. The visible surface still has one edge.
            for topology in plan.local.values_mut().flatten() {
                topology.arms = topology
                    .arms
                    .iter()
                    .chain(topology.arms.first())
                    .copied()
                    .collect();
            }
            for surface in 0..plan.surfaces.len() {
                for position in plan
                    .support_sites(surface)
                    .into_iter()
                    .map(IVec3::from_array)
                {
                    for case in plan.local_cases(surface, position, ONE).unwrap() {
                        let witness = plan.topology.witness(case.guard).unwrap().unwrap();
                        let evaluate = |root| {
                            plan.topology.diagram.evaluate(root, |variable| {
                                witness.get(&variable).copied().unwrap_or(false)
                            })
                        };
                        let variant = local[&position.to_array()]
                            .iter()
                            .find(|variant| evaluate(variant.guard))
                            .unwrap();
                        let zx = &variant.zx;
                        let row = &plan.surfaces[surface].row;
                        let mut dense = PauliString::new(zx.total_ids());
                        for (&column, &index) in &columns {
                            let value = pauli(
                                evaluate(row.get(2 * index)),
                                evaluate(row.get(2 * index + 1)),
                            );
                            match column {
                                SurfaceColumn::Node(at) if at == position.to_array() => {
                                    dense.set(variant.node, value)
                                }
                                SurfaceColumn::Edge(source, target)
                                    if source == position.to_array() =>
                                {
                                    // A column can belong to another guarded local topology.
                                    let Some(target) = zx.node_at(IVec3::from_array(target)) else {
                                        continue;
                                    };
                                    let target = target.id;
                                    let Some(edge) = zx.edge_between(variant.node, target) else {
                                        continue;
                                    };
                                    let value = if edge.hadamard && variant.node > target {
                                        value.flip()
                                    } else {
                                        value
                                    };
                                    dense.set(edge.id, value);
                                    dense.set(zx.edge_id(target, variant.node).unwrap(), value);
                                }
                                _ => {}
                            }
                        }
                        zx.reconstruct_raw_centers(&mut dense);
                        let expected = zx.materialize_stabilizer_with_sign(dense, false);
                        assert_eq!(
                            case.surface.node_pauli(position),
                            expected.node_pauli(position)
                        );
                        assert_eq!(
                            case.surface.port_pauli(position),
                            expected.port_pauli(position)
                        );
                        assert_eq!(case.surface.edges.len(), expected.interior_edges.len());
                        assert_eq!(
                            case.surface.edges().collect::<crate::FxHashMap<_, _>>(),
                            expected.interior_edges
                        );
                        let local_variant = plan.local[&position.to_array()]
                            .iter()
                            .find(|variant| evaluate(variant.guard))
                            .unwrap();
                        assert_eq!(case.selective, local_variant.selector.map(evaluate));
                        assert!(case.surface.edges.len() <= 6);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod fragment_tests {
    use super::*;
    use crate::{GalleryItem, GuardedSurfaceSpace, ModuleCertificationLimits};

    #[test]
    fn shared_fragments_cover_complete_native_coefficients_after_collection() {
        for gallery in [
            GalleryItem::PhaseGradientK4,
            GalleryItem::CZSpatialH,
            GalleryItem::ThreeBitAdder,
        ] {
            let source = gallery.build();
            let linked = crate::flatten_module_definition(&source, source.root(), "").unwrap();
            let limits = ModuleCertificationLimits::DEFAULT;
            let mut plan = GuardedSurfaceSpace::new(
                GuardedTopology::new(&linked.graph.fix_shadowed_faces(), limits).unwrap(),
                &linked.sites,
                limits,
            )
            .unwrap()
            .plan_readouts()
            .unwrap();
            let mut visited = BTreeSet::new();
            let mut original_sites = 0;
            for _ in 0..2 {
                for index in 0..plan.surfaces.len() {
                    let mut support = BTreeSet::new();
                    for &id in plan.query_fragments(index) {
                        visited.insert(id);
                        let (representative, positions) = plan.query_fragment(id);
                        assert_eq!(
                            plan.surfaces[index].activation(),
                            plan.surfaces[representative].activation()
                        );
                        let named = |query: usize| {
                            matches!(
                                plan.surfaces[query].kind,
                                GuardedSurfaceKind::Readout { .. }
                            )
                        };
                        assert_eq!(named(index), named(representative));
                        for &position in positions.iter() {
                            assert!(support.insert(position), "fragments overlap");
                            for column in plan.local[&position]
                                .iter()
                                .flat_map(LocalTopology::columns)
                            {
                                for bit in [2 * column, 2 * column + 1] {
                                    assert_eq!(
                                        plan.surfaces[index].row.get(bit),
                                        plan.surfaces[representative].row.get(bit),
                                        "{gallery:?}: native coefficient {bit}"
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(support, plan.support_sites(index));
                    original_sites += support.len();
                }
                plan.collect_garbage(std::iter::empty());
            }
            if gallery == GalleryItem::ThreeBitAdder {
                let shared_sites = visited
                    .into_iter()
                    .map(|id| plan.query_fragment(id).1.len())
                    .sum::<usize>();
                assert!(
                    shared_sites * 2 < original_sites,
                    "complete query overlap is reused"
                );
            }
        }
    }
}
