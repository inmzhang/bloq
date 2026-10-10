//! Shared template-flow support and exact placed-template joins.

use std::sync::Arc;

use bloq_circuit::{DetectorParity, PauliMap};
use bloq_graph::GuardedTopology;
use bloq_ir::lowering::{BloqTemplatePool, TemplateInstanceId};
use bloq_ir::{BloqNodeId, BundleDetector, BundleMeasurement, DetectorBundle, TemplateId};
use bloq_utils::Pauli;
use bloq_utils::boolean::{BooleanOp, DECISION_FALSE, DECISION_TRUE, DecisionId};
use glam::IVec2;

use crate::{CompileError, FxMap as HashMap, FxSet as HashSet, check_resource};

use super::assembly;
use super::flows::{BoundaryFlow, GuardedFlowEngine};

const MAX_CELLS: usize = 256;

#[derive(Clone, Copy)]
pub(super) struct FlowMember {
    pub instance: TemplateInstanceId,
    pub template: TemplateId,
    pub offset: IVec2,
    pub guard: DecisionId,
    pub skip_unmatched: bool,
}

#[derive(Clone, Copy)]
struct Bounds {
    low: [i32; 2],
    high: [i32; 2],
}

impl Bounds {
    fn include(&mut self, point: [i32; 2]) {
        for (axis, point) in point.into_iter().enumerate() {
            self.low[axis] = self.low[axis].min(point);
            self.high[axis] = self.high[axis].max(point);
        }
    }

    fn union(self, other: Self) -> Self {
        let mut union = self;
        union.include(other.low);
        union.include(other.high);
        union
    }

    fn fits(self, offset: IVec2) -> bool {
        let offset = offset.to_array();
        (0..2).all(|axis| {
            let low = i64::from(self.low[axis]) + i64::from(offset[axis]);
            let high = i64::from(self.high[axis]) + i64::from(offset[axis]);
            low >= i64::from(i32::MIN) && high <= i64::from(i32::MAX)
        })
    }

    fn cells(self, offset: IVec2, cell: i64) -> Option<Vec<(i64, i64)>> {
        let x = i64::from(offset.x);
        let y = i64::from(offset.y);
        let low_x = (i64::from(self.low[0]) + x).div_euclid(cell);
        let high_x = (i64::from(self.high[0]) + x).div_euclid(cell);
        let low_y = (i64::from(self.low[1]) + y).div_euclid(cell);
        let high_y = (i64::from(self.high[1]) + y).div_euclid(cell);
        let width = usize::try_from(high_x - low_x + 1).ok()?;
        let height = usize::try_from(high_y - low_y + 1).ok()?;
        let area = width.checked_mul(height)?;
        if area > MAX_CELLS {
            return None;
        }
        let mut cells = Vec::with_capacity(area);
        for x in low_x..=high_x {
            for y in low_y..=high_y {
                cells.push((x, y));
            }
        }
        Some(cells)
    }
}

#[derive(Default)]
struct Support {
    entries: Vec<([i32; 2], Pauli)>,
    bounds: Option<Bounds>,
}

impl Support {
    fn new(paulis: &bloq_circuit::PauliMap) -> Self {
        let mut support = Self::default();
        for (point, &pauli) in paulis {
            let point = point.to_array();
            support.entries.push((point, pauli));
            if let Some(bounds) = &mut support.bounds {
                bounds.include(point);
            } else {
                support.bounds = Some(Bounds {
                    low: point,
                    high: point,
                });
            }
        }
        support
    }

    fn shifted(&self, delta: [i64; 2]) -> Vec<([i64; 2], Pauli)> {
        self.entries
            .iter()
            .map(|&(point, pauli)| {
                (
                    [
                        i64::from(point[0]) + delta[0],
                        i64::from(point[1]) + delta[1],
                    ],
                    pauli,
                )
            })
            .collect()
    }
}

pub(super) struct TemplateFlowIndex {
    rows: Vec<TemplateRows>,
    cell: i64,
}

struct TemplateRows {
    starts: Vec<Support>,
    ends: Vec<Support>,
    creators: Vec<usize>,
    consumers: Vec<usize>,
    supported: bool,
    bounds: Option<Bounds>,
    start_bounds: Option<Bounds>,
    end_bounds: Option<Bounds>,
}

impl TemplateFlowIndex {
    pub(super) fn new(templates: &BloqTemplatePool, distance: u32) -> Self {
        let rows = templates
            .iter()
            .map(|(_, template)| {
                let mut starts = Vec::with_capacity(template.boundary_flows.len());
                let mut ends = Vec::with_capacity(template.boundary_flows.len());
                let mut creators = Vec::new();
                let mut consumers = Vec::new();
                let mut bounds = None;
                let mut start_bounds = None;
                let mut end_bounds = None;
                for (row, flow) in template.boundary_flows.iter().enumerate() {
                    let start = Support::new(&flow.start);
                    let end = Support::new(&flow.end);
                    if let Some(span) = start.bounds {
                        join_bounds(&mut bounds, span);
                        join_bounds(&mut start_bounds, span);
                    }
                    if let Some(span) = end.bounds {
                        join_bounds(&mut bounds, span);
                        join_bounds(&mut end_bounds, span);
                    }
                    if let Some(center) = flow.center {
                        join_bounds(
                            &mut bounds,
                            Bounds {
                                low: center.to_array(),
                                high: center.to_array(),
                            },
                        );
                    }
                    starts.push(start);
                    ends.push(end);
                    if flow.start.is_empty() {
                        if !flow.end.is_empty() {
                            creators.push(row);
                        }
                    } else {
                        consumers.push(row);
                    }
                }
                TemplateRows {
                    starts,
                    ends,
                    creators,
                    consumers,
                    supported: template
                        .boundary_flows
                        .iter()
                        .all(|flow| flow.start.is_empty() != flow.end.is_empty()),
                    bounds,
                    start_bounds,
                    end_bounds,
                }
            })
            .collect();
        Self {
            rows,
            cell: 2 * i64::from(distance) + 2,
        }
    }

    pub(super) fn coordinates_fit(&self, member: FlowMember) -> bool {
        self.rows[member.template.0 as usize]
            .bounds
            .is_none_or(|bounds| bounds.fits(member.offset))
    }
}

fn join_bounds(target: &mut Option<Bounds>, added: Bounds) {
    *target = Some(target.map_or(added, |bounds| bounds.union(added)));
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PairKey {
    producer: TemplateId,
    consumer: TemplateId,
    delta: [i64; 2],
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct RecipeKey {
    pair: PairKey,
    producer: SubsetId,
    consumer: SubsetId,
    alias: bool,
}

struct Recipe {
    bundle: Arc<DetectorBundle>,
}

#[derive(Clone, Copy)]
struct Cohort {
    root: usize,
    consumer: SubsetId,
    member: FlowMember,
    node: BloqNodeId,
    old_guard: DecisionId,
}

pub(super) struct PlannedBundleUse {
    pub node: BloqNodeId,
    pub bundle: Arc<DetectorBundle>,
    pub instances: Vec<TemplateInstanceId>,
    pub offset: IVec2,
    pub guard: DecisionId,
}

#[derive(Default)]
pub(super) struct ComponentPlan {
    pub bundles: Vec<PlannedBundleUse>,
}

pub(super) struct FamilyCompiler {
    index: TemplateFlowIndex,
    pairs: PairJoinCache,
    subsets: RowSubsetPool,
    full_creators: Vec<SubsetId>,
    full_consumers: Vec<SubsetId>,
    recipes: HashMap<RecipeKey, Arc<Recipe>>,
    coverage: HashMap<(SubsetId, Vec<SubsetId>), Vec<Vec<usize>>>,
}

impl FamilyCompiler {
    pub(super) fn report(&self, accepted: usize, fallback: usize, uses: usize) {
        if std::env::var_os("BLOQ_FLOW_FAMILY_DIAG").is_some() {
            let rows = self
                .recipes
                .values()
                .map(|recipe| recipe.bundle.detectors().len())
                .sum::<usize>();
            eprintln!(
                "flow_family_diag accepted={accepted} fallback={fallback} recipes={} stored_rows={rows} uses={uses} pair_joins={} subsets={}",
                self.recipes.len(),
                self.pairs.joins.len(),
                self.subsets.rows.len()
            );
        }
    }
    pub(super) fn new(
        templates: &BloqTemplatePool,
        topology: &mut GuardedTopology,
        distance: u32,
    ) -> Result<Self, CompileError> {
        topology.diagram.charge(templates.len())?;
        for (_, template) in templates.iter() {
            topology.diagram.charge(template.boundary_flows.len())?;
            let work = template.boundary_flows.iter().fold(0usize, |sum, flow| {
                sum.saturating_add(flow.start.len())
                    .saturating_add(flow.end.len())
                    .saturating_add(1)
            });
            topology.diagram.charge(work)?;
        }
        let index = TemplateFlowIndex::new(templates, distance);
        let mut subsets = RowSubsetPool::default();
        let full_creators = index
            .rows
            .iter()
            .enumerate()
            .map(|(id, rows)| subsets.intern(TemplateId(id as u32), rows.creators.clone()))
            .collect();
        let full_consumers = index
            .rows
            .iter()
            .enumerate()
            .map(|(id, rows)| subsets.intern(TemplateId(id as u32), rows.consumers.clone()))
            .collect();
        Ok(Self {
            index,
            pairs: PairJoinCache::default(),
            subsets,
            full_creators,
            full_consumers,
            recipes: HashMap::default(),
            coverage: HashMap::default(),
        })
    }

    pub(super) fn coordinates_fit(&self, member: FlowMember) -> bool {
        self.index.coordinates_fit(member)
    }

    pub(super) fn order_same_layer(
        &mut self,
        nodes: &mut [BloqNodeId],
        members: &HashMap<BloqNodeId, Vec<FlowMember>>,
        topology: &mut GuardedTopology,
    ) -> Result<bool, CompileError> {
        order_same_layer(nodes, members, &self.index, &mut self.pairs, topology)
    }

    fn recipe(
        &mut self,
        producer: FlowMember,
        consumer: FlowMember,
        producer_subset: SubsetId,
        consumer_subset: SubsetId,
        templates: &BloqTemplatePool,
        topology: &mut GuardedTopology,
    ) -> Result<Option<Arc<Recipe>>, CompileError> {
        let key = RecipeKey {
            pair: PairJoinCache::key(producer, consumer),
            producer: producer_subset,
            consumer: consumer_subset,
            alias: producer.instance == consumer.instance,
        };
        if let Some(recipe) = self.recipes.get(&key) {
            // A hit binds owner slots; it never materializes the cached rows.
            topology
                .diagram
                .charge(recipe.bundle.owner_templates().len().saturating_add(1))?;
            return Ok(Some(Arc::clone(recipe)));
        }
        let work = self
            .subsets
            .rows(producer_subset)
            .iter()
            .map(|&row| &templates[producer.template].boundary_flows[row])
            .chain(
                self.subsets
                    .rows(consumer_subset)
                    .iter()
                    .map(|&row| &templates[consumer.template].boundary_flows[row]),
            )
            .fold(0usize, |sum, flow| {
                sum.saturating_add(flow.start.len())
                    .saturating_add(flow.end.len())
                    .saturating_add(flow.measurements.len())
                    .saturating_add(1)
            });
        topology.diagram.charge(work)?;
        let relative = IVec2::new(
            match i32::try_from(key.pair.delta[0]) {
                Ok(value) => value,
                Err(_) => return Ok(None),
            },
            match i32::try_from(key.pair.delta[1]) {
                Ok(value) => value,
                Err(_) => return Ok(None),
            },
        );
        let mut creators = Vec::new();
        for &row in self.subsets.rows(producer_subset) {
            let flow = &templates[producer.template].boundary_flows[row];
            creators.push(BoundaryFlow {
                guard: DECISION_TRUE,
                start: PauliMap::empty(),
                end: flow.end.clone(),
                measurements: flow
                    .measurements
                    .iter()
                    .map(|&measurement| bloq_ir::lowering::InstanceMeasurement {
                        instance: TemplateInstanceId(0),
                        measurement,
                    })
                    .collect(),
                sign: flow.sign,
                center: flow.center.map(|center| center.to_array()),
                marker: flow.marker,
                skip_unmatched: false,
            });
        }
        let consumer_owner = if key.alias { 0 } else { 1 };
        let mut consumers = Vec::new();
        for &row in self.subsets.rows(consumer_subset) {
            let flow = &templates[consumer.template].boundary_flows[row];
            if !flow.end.is_empty() {
                return Ok(None);
            }
            let Some(start) = flow.start.try_translated(relative).ok() else {
                return Ok(None);
            };
            let Some(center) = flow
                .center
                .map(|center| bloq_circuit::checked_translate_coordinate(center, relative))
                .transpose()
                .ok()
            else {
                return Ok(None);
            };
            consumers.push(BoundaryFlow {
                guard: DECISION_TRUE,
                start,
                end: PauliMap::empty(),
                measurements: flow
                    .measurements
                    .iter()
                    .map(|&measurement| bloq_ir::lowering::InstanceMeasurement {
                        instance: TemplateInstanceId(consumer_owner),
                        measurement,
                    })
                    .collect(),
                sign: flow.sign,
                center: center.map(|center| center.to_array()),
                marker: flow.marker,
                skip_unmatched: false,
            });
        }
        let mut engine = GuardedFlowEngine::default();
        let created = engine.append(&creators, &mut topology.diagram, DECISION_TRUE)?;
        if !created.is_empty() {
            return Ok(None);
        }
        let checks = engine.append(&consumers, &mut topology.diagram, DECISION_TRUE)?;
        engine.finish(&mut topology.diagram, DECISION_TRUE)?;
        topology.diagram.charge(checks.len())?;
        topology
            .diagram
            .charge(checks.iter().fold(0usize, |sum, check| {
                sum.saturating_add(check.parity.terms().len())
                    .saturating_add(1)
            }))?;
        let mut detectors = Vec::with_capacity(checks.len());
        for check in checks {
            if check.restart {
                return Err(assembly(
                    "a top-level boundary flow carries a restart syndrome",
                ));
            }
            if check.guard != DECISION_TRUE || !check.contributions.is_empty() {
                return Ok(None);
            }
            detectors.push(BundleDetector {
                parity: DetectorParity::from_measurements(check.parity.measurements().map(
                    |measurement| BundleMeasurement {
                        owner: measurement.instance.0,
                        measurement: measurement.measurement,
                    },
                ))
                .with_sign(check.parity.sign()),
                coords: check
                    .center
                    .map(|center| center.into_iter().map(f64::from).collect()),
            });
        }
        let owners = if key.alias {
            vec![producer.template]
        } else {
            vec![producer.template, consumer.template]
        };
        let recipe = Arc::new(Recipe {
            bundle: Arc::new(DetectorBundle::new(owners, detectors)),
        });
        self.recipes.insert(key, Arc::clone(&recipe));
        Ok(Some(recipe))
    }

    pub(super) fn plan_component(
        &mut self,
        group: &[BloqNodeId],
        members: &HashMap<BloqNodeId, Vec<FlowMember>>,
        templates: &BloqTemplatePool,
        topology: &mut GuardedTopology,
        limit: usize,
    ) -> Result<Option<ComponentPlan>, CompileError> {
        let mut plan = ComponentPlan::default();
        let mut frontier = Frontier::default();
        let mut source_order = 0;
        for &node in group {
            let node_members = &members[&node];
            for &member in node_members {
                // Relays and zero-boundary rows need the general flow engine.
                if !self.index.rows[member.template.0 as usize].supported {
                    return Ok(None);
                }
                topology.diagram.charge(1)?;
            }
            let mut incoming_grid = HashMap::<(i64, i64), Vec<usize>>::default();
            for (position, &member) in node_members.iter().enumerate() {
                let Some(bounds) = self.index.rows[member.template.0 as usize].start_bounds else {
                    continue;
                };
                let Some(cells) = bounds.cells(member.offset, self.index.cell) else {
                    return Ok(None);
                };
                topology.diagram.charge(cells.len().saturating_add(1))?;
                let subset = self.full_consumers[member.template.0 as usize];
                if topology.diagram.constrain(member.guard, topology.domain)? != DECISION_FALSE
                    && self.pairs.has_duplicate(
                        &self.index,
                        member,
                        member,
                        true,
                        true,
                        subset,
                        subset,
                        &self.subsets,
                        topology,
                    )?
                {
                    return Err(assembly("incoming flow boundaries are not independent"));
                }
                let mut seen = HashSet::default();
                for cell in &cells {
                    for &previous in incoming_grid.get(cell).into_iter().flatten() {
                        topology.diagram.charge(1)?;
                        if !seen.insert(previous) {
                            continue;
                        }
                        let old = node_members[previous];
                        let overlap =
                            topology
                                .diagram
                                .apply(BooleanOp::And, old.guard, member.guard)?;
                        if topology.diagram.constrain(overlap, topology.domain)? == DECISION_FALSE {
                            continue;
                        }
                        if self.pairs.has_duplicate(
                            &self.index,
                            old,
                            member,
                            true,
                            false,
                            self.full_consumers[old.template.0 as usize],
                            subset,
                            &self.subsets,
                            topology,
                        )? {
                            return Err(assembly("incoming flow boundaries are not independent"));
                        }
                    }
                }
                for cell in cells {
                    incoming_grid.entry(cell).or_default().push(position);
                }
            }
            let mut cohorts = Vec::new();
            for &member in node_members {
                let incoming = self.full_consumers[member.template.0 as usize];
                if self.subsets.rows(incoming).is_empty() {
                    continue;
                }
                let Some(candidates) = frontier.candidates(member, &self.index, false, topology)?
                else {
                    return Ok(None);
                };
                let mut local = Vec::new();
                for old in candidates {
                    let piece = frontier.pieces[old].expect("live frontier candidate");
                    let selected =
                        topology
                            .diagram
                            .apply(BooleanOp::And, piece.member.guard, member.guard)?;
                    if topology.diagram.constrain(selected, topology.domain)? == DECISION_FALSE {
                        continue;
                    }
                    let Some((_, consumer)) = self.pairs.subset_join(
                        &self.index,
                        &mut self.subsets,
                        piece.member,
                        member,
                        piece.subset,
                        incoming,
                        topology,
                    )?
                    else {
                        continue;
                    };
                    local.push(Cohort {
                        root: piece.root,
                        consumer,
                        member,
                        node,
                        old_guard: piece.member.guard,
                    });
                }
                if !member.skip_unmatched {
                    let key = (
                        incoming,
                        local
                            .iter()
                            .map(|cohort| cohort.consumer)
                            .collect::<Vec<_>>(),
                    );
                    topology.diagram.charge(local.len().saturating_add(1))?;
                    if !self.coverage.contains_key(&key) {
                        topology.diagram.charge(
                            self.subsets
                                .rows(incoming)
                                .len()
                                .saturating_mul(local.len().saturating_add(1)),
                        )?;
                        let mut patterns = HashSet::default();
                        for &row in self.subsets.rows(incoming) {
                            patterns.insert(
                                local
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(role, cohort)| {
                                        self.subsets
                                            .rows(cohort.consumer)
                                            .binary_search(&row)
                                            .is_ok()
                                            .then_some(role)
                                    })
                                    .collect::<Vec<_>>(),
                            );
                        }
                        let mut patterns = patterns.into_iter().collect::<Vec<_>>();
                        patterns.sort_unstable();
                        self.coverage.insert(key.clone(), patterns);
                    }
                    for roles in &self.coverage[&key] {
                        topology.diagram.charge(roles.len().saturating_add(1))?;
                        let mut present = DECISION_FALSE;
                        for &role in roles {
                            present = topology.diagram.apply(
                                BooleanOp::Or,
                                present,
                                local[role].old_guard,
                            )?;
                        }
                        let absent = topology.diagram.negate(present)?;
                        let missing =
                            topology
                                .diagram
                                .apply(BooleanOp::And, member.guard, absent)?;
                        if topology.diagram.constrain(missing, topology.domain)? != DECISION_FALSE {
                            return Ok(None);
                        }
                    }
                }
                cohorts.extend(local);
            }
            // Duplicate creator checks guarantee at most one equal-body
            // predecessor on a reachable assignment. Keep alternatives as
            // separately guarded uses instead of expanding conditional parity.
            for cohort in cohorts {
                let pieces = frontier.lineage(cohort.root).collect::<Vec<_>>();
                for (id, piece) in pieces {
                    let Some((producer, consumer)) = self.pairs.subset_join(
                        &self.index,
                        &mut self.subsets,
                        piece.member,
                        cohort.member,
                        piece.subset,
                        cohort.consumer,
                        topology,
                    )?
                    else {
                        continue;
                    };
                    let selected = topology.diagram.apply(
                        BooleanOp::And,
                        piece.member.guard,
                        cohort.member.guard,
                    )?;
                    let selected = topology.diagram.constrain(selected, topology.domain)?;
                    if selected == DECISION_FALSE {
                        continue;
                    }
                    let Some(recipe) = self.recipe(
                        piece.member,
                        cohort.member,
                        producer,
                        consumer,
                        templates,
                        topology,
                    )?
                    else {
                        return Ok(None);
                    };
                    if !recipe.bundle.detectors().is_empty() {
                        plan.bundles.push(PlannedBundleUse {
                            node: cohort.node,
                            bundle: Arc::clone(&recipe.bundle),
                            instances: if piece.member.instance == cohort.member.instance {
                                vec![piece.member.instance]
                            } else {
                                vec![piece.member.instance, cohort.member.instance]
                            },
                            offset: piece.member.offset,
                            guard: selected,
                        });
                    }
                    let untouched = self.subsets.difference(piece.subset, producer, topology)?;
                    let remaining_guard = topology.diagram.negate(cohort.member.guard)?;
                    let remaining_guard = topology.diagram.apply(
                        BooleanOp::And,
                        piece.member.guard,
                        remaining_guard,
                    )?;
                    let remaining_guard = topology
                        .diagram
                        .constrain(remaining_guard, topology.domain)?;
                    frontier.remove(id);
                    if !self.subsets.rows(untouched).is_empty()
                        && !frontier.add(
                            FamilyPiece {
                                subset: untouched,
                                ..piece
                            },
                            &self.index,
                        )
                    {
                        return Ok(None);
                    }
                    if remaining_guard != DECISION_FALSE
                        && !frontier.add(
                            FamilyPiece {
                                member: FlowMember {
                                    guard: remaining_guard,
                                    ..piece.member
                                },
                                subset: producer,
                                ..piece
                            },
                            &self.index,
                        )
                    {
                        return Ok(None);
                    }
                }
            }
            for &member in node_members {
                let subset = self.full_creators[member.template.0 as usize];
                if self.subsets.rows(subset).is_empty() {
                    continue;
                }
                let Some(candidates) = frontier.candidates(member, &self.index, true, topology)?
                else {
                    return Ok(None);
                };
                for old in candidates {
                    let piece = frontier.pieces[old].expect("live frontier candidate");
                    let both =
                        topology
                            .diagram
                            .apply(BooleanOp::And, piece.member.guard, member.guard)?;
                    if topology.diagram.constrain(both, topology.domain)? == DECISION_FALSE {
                        continue;
                    }
                    if self.pairs.has_duplicate(
                        &self.index,
                        piece.member,
                        member,
                        false,
                        false,
                        piece.subset,
                        subset,
                        &self.subsets,
                        topology,
                    )? {
                        return Err(assembly("duplicate open boundary flow"));
                    }
                }
                let member_guard = topology.diagram.constrain(member.guard, topology.domain)?;
                if member_guard == DECISION_FALSE {
                    continue;
                }
                if self.pairs.has_duplicate(
                    &self.index,
                    member,
                    member,
                    false,
                    true,
                    subset,
                    subset,
                    &self.subsets,
                    topology,
                )? {
                    return Err(assembly("duplicate open boundary flow"));
                }
                if !frontier.add(
                    FamilyPiece {
                        member: FlowMember {
                            guard: member_guard,
                            ..member
                        },
                        subset,
                        order: source_order,
                        root: source_order,
                    },
                    &self.index,
                ) {
                    return Ok(None);
                }
                source_order += 1;
            }
            if frontier.stale > frontier.live {
                topology.diagram.charge(frontier.pieces.len())?;
            }
            if !frontier.rebuild(&self.index) {
                return Ok(None);
            }
            check_resource(
                "guarded boundary-flow Boolean nodes",
                topology.diagram.nodes().len(),
                limit,
            )?;
        }
        if frontier.live != 0 {
            return Err(assembly("unterminated boundary flows"));
        }
        Ok(Some(plan))
    }
}

#[derive(Default)]
pub(super) struct PairJoinCache {
    joins: HashMap<PairKey, Vec<(usize, usize)>>,
    same_starts: HashMap<PairKey, Vec<(usize, usize)>>,
    same_ends: HashMap<PairKey, Vec<(usize, usize)>>,
    subsets: HashMap<(PairKey, SubsetId, SubsetId), Option<(SubsetId, SubsetId)>>,
    duplicates: HashMap<(PairKey, bool, SubsetId, SubsetId, bool), bool>,
}

impl PairJoinCache {
    #[expect(
        clippy::too_many_arguments,
        reason = "the exact duplicate query separates immutable support/subsets from the mutable cache and Boolean budget"
    )]
    fn has_duplicate(
        &mut self,
        index: &TemplateFlowIndex,
        left: FlowMember,
        right: FlowMember,
        starts: bool,
        same: bool,
        left_subset: SubsetId,
        right_subset: SubsetId,
        pool: &RowSubsetPool,
        topology: &mut GuardedTopology,
    ) -> Result<bool, CompileError> {
        let key = (
            Self::key(left, right),
            starts,
            left_subset,
            right_subset,
            same,
        );
        topology.diagram.charge(1)?;
        if let Some(&answer) = self.duplicates.get(&key) {
            return Ok(answer);
        }
        let pairs = self.duplicate_pairs(index, left, right, starts, topology)?;
        topology.diagram.charge(pairs.len())?;
        let answer = pairs.iter().any(|&(a, b)| {
            (!same || a < b)
                && pool.rows(left_subset).binary_search(&a).is_ok()
                && pool.rows(right_subset).binary_search(&b).is_ok()
        });
        self.duplicates.insert(key, answer);
        Ok(answer)
    }
    fn key(producer: FlowMember, consumer: FlowMember) -> PairKey {
        PairKey {
            producer: producer.template,
            consumer: consumer.template,
            delta: [
                i64::from(consumer.offset.x) - i64::from(producer.offset.x),
                i64::from(consumer.offset.y) - i64::from(producer.offset.y),
            ],
        }
    }

    pub(super) fn joins<'a>(
        &'a mut self,
        index: &TemplateFlowIndex,
        producer: FlowMember,
        consumer: FlowMember,
        topology: &mut GuardedTopology,
    ) -> Result<&'a [(usize, usize)], CompileError> {
        let key = Self::key(producer, consumer);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.joins.entry(key) {
            let left = &index.rows[key.producer.0 as usize].ends;
            let right = &index.rows[key.consumer.0 as usize].starts;
            entry.insert(match_supports(left, right, key.delta, topology)?);
        }
        Ok(&self.joins[&key])
    }

    fn duplicate_pairs<'a>(
        &'a mut self,
        index: &TemplateFlowIndex,
        left: FlowMember,
        right: FlowMember,
        starts: bool,
        topology: &mut GuardedTopology,
    ) -> Result<&'a [(usize, usize)], CompileError> {
        let key = Self::key(left, right);
        let cache = if starts {
            &mut self.same_starts
        } else {
            &mut self.same_ends
        };
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
            let left = &index.rows[key.producer.0 as usize];
            let right = &index.rows[key.consumer.0 as usize];
            let (left, right) = if starts {
                (&left.starts, &right.starts)
            } else {
                (&left.ends, &right.ends)
            };
            entry.insert(match_supports(left, right, key.delta, topology)?);
        }
        Ok(&cache[&key])
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the placed-pair query borrows immutable support and mutable subset/cache/Boolean state separately"
    )]
    fn subset_join(
        &mut self,
        index: &TemplateFlowIndex,
        pool: &mut RowSubsetPool,
        producer: FlowMember,
        consumer: FlowMember,
        old: SubsetId,
        incoming: SubsetId,
        topology: &mut GuardedTopology,
    ) -> Result<Option<(SubsetId, SubsetId)>, CompileError> {
        let key = (Self::key(producer, consumer), old, incoming);
        if let Some(&result) = self.subsets.get(&key) {
            return Ok(result);
        }
        let old_rows = pool.rows(old);
        let new_rows = pool.rows(incoming);
        topology
            .diagram
            .charge(old_rows.len().saturating_add(new_rows.len()))?;
        let old_set = old_rows.iter().copied().collect::<HashSet<_>>();
        let new_set = new_rows.iter().copied().collect::<HashSet<_>>();
        let joins = self.joins(index, producer, consumer, topology)?;
        topology.diagram.charge(joins.len())?;
        let matched = joins
            .iter()
            .copied()
            .filter(|(left, right)| old_set.contains(left) && new_set.contains(right))
            .collect::<Vec<_>>();
        topology.diagram.charge(matched.len())?;
        let result = if matched.is_empty() {
            None
        } else {
            let mut left = matched.iter().map(|&(row, _)| row).collect::<Vec<_>>();
            let mut right = matched.iter().map(|&(_, row)| row).collect::<Vec<_>>();
            left.sort_unstable();
            left.dedup();
            right.sort_unstable();
            right.dedup();
            Some((
                pool.intern(producer.template, left),
                pool.intern(consumer.template, right),
            ))
        };
        self.subsets.insert(key, result);
        Ok(result)
    }
}

fn match_supports(
    left: &[Support],
    right: &[Support],
    delta: [i64; 2],
    topology: &mut GuardedTopology,
) -> Result<Vec<(usize, usize)>, CompileError> {
    let work = left.iter().chain(right).fold(0usize, |sum, support| {
        sum.saturating_add(support.entries.len().saturating_add(1))
    });
    topology.diagram.charge(work)?;
    let mut by_shape = HashMap::<Vec<([i64; 2], Pauli)>, Vec<usize>>::default();
    for (row, support) in left.iter().enumerate() {
        if !support.entries.is_empty() {
            by_shape
                .entry(support.shifted([0, 0]))
                .or_default()
                .push(row);
        }
    }
    let mut matches = Vec::new();
    for (right_row, support) in right.iter().enumerate() {
        if !support.entries.is_empty() {
            for &left_row in by_shape.get(&support.shifted(delta)).into_iter().flatten() {
                topology.diagram.charge(1)?;
                matches.push((left_row, right_row));
            }
        }
    }
    matches.sort_unstable();
    Ok(matches)
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SubsetId(usize);

struct RowSubset {
    template: TemplateId,
    rows: Vec<usize>,
}

#[derive(Default)]
struct RowSubsetPool {
    rows: Vec<RowSubset>,
    ids: HashMap<(TemplateId, Vec<usize>), SubsetId>,
    differences: HashMap<(SubsetId, SubsetId), SubsetId>,
}

#[derive(Clone, Copy)]
struct FamilyPiece {
    member: FlowMember,
    subset: SubsetId,
    order: usize,
    root: usize,
}

#[derive(Default)]
struct Frontier {
    pieces: Vec<Option<FamilyPiece>>,
    grid: HashMap<(i64, i64), Vec<usize>>,
    lineages: HashMap<usize, Vec<usize>>,
    live: usize,
    stale: usize,
}

impl Frontier {
    fn add(&mut self, piece: FamilyPiece, index: &TemplateFlowIndex) -> bool {
        let Some(bounds) = index.rows[piece.member.template.0 as usize].end_bounds else {
            return false;
        };
        let Some(cells) = bounds.cells(piece.member.offset, index.cell) else {
            return false;
        };
        let id = self.pieces.len();
        self.lineages.entry(piece.root).or_default().push(id);
        self.pieces.push(Some(piece));
        self.live += 1;
        for cell in cells {
            self.grid.entry(cell).or_default().push(id);
        }
        true
    }

    fn remove(&mut self, id: usize) {
        if let Some(piece) = self.pieces[id].take() {
            if let Some(ids) = self.lineages.get_mut(&piece.root) {
                ids.retain(|&other| other != id);
            }
            self.live -= 1;
            self.stale += 1;
        }
    }

    fn lineage(&self, root: usize) -> impl Iterator<Item = (usize, FamilyPiece)> + '_ {
        self.lineages
            .get(&root)
            .into_iter()
            .flatten()
            .filter_map(|&id| self.pieces[id].map(|piece| (id, piece)))
    }

    fn candidates(
        &self,
        member: FlowMember,
        index: &TemplateFlowIndex,
        ends: bool,
        topology: &mut GuardedTopology,
    ) -> Result<Option<Vec<usize>>, CompileError> {
        let rows = &index.rows[member.template.0 as usize];
        let Some(bounds) = (if ends {
            rows.end_bounds
        } else {
            rows.start_bounds
        }) else {
            return Ok(Some(Vec::new()));
        };
        let Some(cells) = bounds.cells(member.offset, index.cell) else {
            return Ok(None);
        };
        let mut seen = HashSet::default();
        let mut result = Vec::new();
        for cell in cells {
            for &id in self.grid.get(&cell).into_iter().flatten() {
                topology.diagram.charge(1)?;
                if self.pieces[id].is_some() && seen.insert(id) {
                    result.push(id);
                    if result.len() > 16_384 {
                        return Ok(None);
                    }
                }
            }
        }
        result.sort_unstable_by_key(|&id| self.pieces[id].expect("live candidate").order);
        Ok(Some(result))
    }

    fn rebuild(&mut self, index: &TemplateFlowIndex) -> bool {
        if self.stale <= self.live {
            return true;
        }
        self.pieces = std::mem::take(&mut self.pieces)
            .into_iter()
            .flatten()
            .map(Some)
            .collect();
        self.lineages.clear();
        self.grid.clear();
        for (id, piece) in self.pieces.iter().enumerate() {
            let Some(piece) = piece else { continue };
            let Some(bounds) = index.rows[piece.member.template.0 as usize].end_bounds else {
                return false;
            };
            let Some(cells) = bounds.cells(piece.member.offset, index.cell) else {
                return false;
            };
            self.lineages.entry(piece.root).or_default().push(id);
            for cell in cells {
                self.grid.entry(cell).or_default().push(id);
            }
        }
        self.stale = 0;
        true
    }
}

impl RowSubsetPool {
    fn intern(&mut self, template: TemplateId, rows: Vec<usize>) -> SubsetId {
        if let Some(&id) = self.ids.get(&(template, rows.clone())) {
            return id;
        }
        let id = SubsetId(self.rows.len());
        self.ids.insert((template, rows.clone()), id);
        self.rows.push(RowSubset { template, rows });
        id
    }

    fn rows(&self, id: SubsetId) -> &[usize] {
        &self.rows[id.0].rows
    }

    fn difference(
        &mut self,
        old: SubsetId,
        consumed: SubsetId,
        topology: &mut GuardedTopology,
    ) -> Result<SubsetId, CompileError> {
        if let Some(&id) = self.differences.get(&(old, consumed)) {
            return Ok(id);
        }
        let left = self.rows(old);
        let right = self.rows(consumed);
        topology
            .diagram
            .charge(left.len().saturating_add(right.len()))?;
        let mut remaining = Vec::new();
        let mut cursor = 0;
        for &row in left {
            while cursor < right.len() && right[cursor] < row {
                cursor += 1;
            }
            if right.get(cursor) != Some(&row) {
                remaining.push(row);
            }
        }
        let id = self.intern(self.rows[old.0].template, remaining);
        self.differences.insert((old, consumed), id);
        Ok(id)
    }
}

/// Returns false only when a template spans too many grid cells; the caller
/// can use the ordinary same-layer ordering for that layer.
pub(super) fn order_same_layer(
    nodes: &mut [BloqNodeId],
    members: &HashMap<BloqNodeId, Vec<FlowMember>>,
    index: &TemplateFlowIndex,
    pairs: &mut PairJoinCache,
    topology: &mut GuardedTopology,
) -> Result<bool, CompileError> {
    if nodes.len() < 2 {
        return Ok(true);
    }
    let consumers = nodes
        .iter()
        .enumerate()
        .flat_map(|(node, id)| {
            members[id]
                .iter()
                .copied()
                .map(move |member| (node, member))
        })
        .filter(|(_, member)| {
            index.rows[member.template.0 as usize]
                .start_bounds
                .is_some()
        })
        .collect::<Vec<_>>();
    let mut grid = HashMap::<(i64, i64), Vec<usize>>::default();
    for (index_in_list, &(_, member)) in consumers.iter().enumerate() {
        let bounds = index.rows[member.template.0 as usize]
            .start_bounds
            .expect("filtered consumer has support");
        let Some(cells) = bounds.cells(member.offset, index.cell) else {
            return Ok(false);
        };
        topology.diagram.charge(cells.len())?;
        for cell in cells {
            grid.entry(cell).or_default().push(index_in_list);
        }
    }
    let mut successors = vec![HashSet::default(); nodes.len()];
    let mut indegrees = vec![0; nodes.len()];
    for (producer, id) in nodes.iter().enumerate() {
        for &producer_member in &members[id] {
            let Some(bounds) = index.rows[producer_member.template.0 as usize].end_bounds else {
                continue;
            };
            let Some(cells) = bounds.cells(producer_member.offset, index.cell) else {
                return Ok(false);
            };
            topology.diagram.charge(cells.len())?;
            let mut seen = HashSet::default();
            for cell in cells {
                for &candidate in grid.get(&cell).into_iter().flatten() {
                    if !seen.insert(candidate) {
                        continue;
                    }
                    let (consumer, consumer_member) = consumers[candidate];
                    if consumer == producer {
                        continue;
                    }
                    topology.diagram.charge(1)?;
                    if pairs
                        .joins(index, producer_member, consumer_member, topology)?
                        .is_empty()
                    {
                        continue;
                    }
                    let both = topology.diagram.apply(
                        bloq_utils::boolean::BooleanOp::And,
                        producer_member.guard,
                        consumer_member.guard,
                    )?;
                    if topology.diagram.constrain(both, topology.domain)? != DECISION_FALSE
                        && successors[consumer].insert(producer)
                    {
                        indegrees[producer] += 1;
                    }
                }
            }
        }
    }
    let mut ready = indegrees
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(nodes.len());
    while let Some(next) = ready.pop_first() {
        ordered.push(nodes[next]);
        for &successor in &successors[next] {
            indegrees[successor] -= 1;
            if indegrees[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    if ordered.len() != nodes.len() {
        return Err(assembly("cyclic same-layer boundary-flow dependency"));
    }
    nodes.copy_from_slice(&ordered);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CoordCircuit, Flow};
    use bloq_graph::{BlockGraph, ModuleCertificationLimits};
    use bloq_ir::lowering::{BloqTemplate, InstanceMeasurement};
    use bloq_ir::{Bloq, BloqNode, NodeDetectorParity};
    use bloq_utils::boolean::{BooleanDecisionDiagram, BooleanLimits};

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn subset_pool_indices_do_not_narrow_to_u32() {
        let index = u64::from(u32::MAX) + 1;
        let id = SubsetId(index.try_into().expect("subset indices cover usize"));
        assert_eq!(u64::try_from(id.0).unwrap(), index);
    }

    fn topology() -> GuardedTopology {
        GuardedTopology::new(&BlockGraph::new(), ModuleCertificationLimits::default()).unwrap()
    }

    fn template(pool: &mut BloqTemplatePool, flows: Vec<Flow>) -> TemplateId {
        pool.insert(BloqTemplate::with_parts(
            CoordCircuit::default(),
            vec![],
            vec![],
            flows,
            vec![],
        ))
    }

    fn source(template: TemplateId, instance: u32, guard: DecisionId) -> FlowMember {
        FlowMember {
            template,
            instance: TemplateInstanceId(instance),
            offset: IVec2::new(8, -4),
            guard,
            skip_unmatched: false,
        }
    }

    fn compare(
        group: &[BloqNodeId],
        members: &HashMap<BloqNodeId, Vec<FlowMember>>,
        templates: &BloqTemplatePool,
        topology: &mut GuardedTopology,
    ) -> FamilyCompiler {
        let mut compiler = FamilyCompiler::new(templates, topology, 3).unwrap();
        let plan = compiler
            .plan_component(group, members, templates, topology, usize::MAX)
            .unwrap()
            .expect("fixed-support component is supported");
        let mut engine = GuardedFlowEngine::default();
        let mut checks = Vec::new();
        for node in group {
            let flows =
                super::super::assembly::materialize_flows(&members[node], templates, topology)
                    .unwrap();
            checks.extend(
                engine
                    .append(&flows, &mut topology.diagram, topology.domain)
                    .unwrap(),
            );
        }
        engine
            .finish(&mut topology.diagram, topology.domain)
            .unwrap();
        for mask in 0..4 {
            let enabled = |guard| {
                topology
                    .diagram
                    .evaluate(guard, |var| mask & (1 << var) != 0)
            };
            if !enabled(topology.domain) {
                continue;
            }
            let mut expected = checks
                .iter()
                .filter(|check| enabled(check.guard))
                .map(|check| {
                    let mut parity = check.parity.clone();
                    for (&guard, contribution) in &check.contributions {
                        if enabled(guard) {
                            parity.xor_assign(contribution);
                        }
                    }
                    (parity, check.center)
                })
                .collect::<Vec<_>>();
            let actual = plan
                .bundles
                .iter()
                .filter(|use_| enabled(use_.guard))
                .flat_map(|use_| {
                    use_.bundle.detectors().iter().map(move |detector| {
                        let parity = NodeDetectorParity::from_measurements(
                            detector.parity.measurements().map(|m| InstanceMeasurement {
                                instance: use_.instances[m.owner as usize],
                                measurement: m.measurement,
                            }),
                        )
                        .with_sign(detector.parity.sign());
                        let center = detector.coords.as_ref().map(|coords| {
                            [
                                coords[0] as i32 + use_.offset.x,
                                coords[1] as i32 + use_.offset.y,
                            ]
                        });
                        (parity, center)
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), expected.len(), "mask {mask}");
            for check in actual {
                let found = expected
                    .iter()
                    .position(|row| *row == check)
                    .unwrap_or_else(|| panic!("unexpected check at mask {mask}: {check:?}"));
                expected.swap_remove(found);
            }
        }
        compiler
    }

    #[test]
    fn guarded_families_match_expansion_and_reuse_partial_subsets() {
        let mut topology = topology();
        let a = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let not_a = topology.diagram.negate(a).unwrap();
        let y = PauliMap::from_unique_entries([(IVec2::ZERO, Pauli::Y)]);
        let x = PauliMap::from_unique_entries([(IVec2::new(4, 0), Pauli::X)]);
        let mut templates = BloqTemplatePool::default();
        let creator = template(
            &mut templates,
            vec![
                Flow::new(PauliMap::empty(), y.clone())
                    .with_measurements([0])
                    .with_sign(true)
                    .with_center(IVec2::new(1, 0)),
                Flow::new(PauliMap::empty(), x.clone()).with_measurements([1]),
            ],
        );
        let y_end = template(
            &mut templates,
            vec![
                Flow::new(y, PauliMap::empty())
                    .with_measurements([0])
                    .with_center(IVec2::new(2, 0)),
            ],
        );
        let x_end = template(
            &mut templates,
            vec![Flow::new(x, PauliMap::empty()).with_measurements([1])],
        );
        let mut bloq = Bloq::new();
        let group = (0..4)
            .map(|_| bloq.add_node(BloqNode::from_members(vec![])))
            .collect::<Vec<_>>();
        let members = HashMap::from_iter([
            (group[0], vec![source(creator, 0, DECISION_TRUE)]),
            (group[1], vec![source(y_end, 1, a)]),
            (group[2], vec![source(y_end, 2, not_a)]),
            (group[3], vec![source(x_end, 3, DECISION_TRUE)]),
        ]);
        let mut compiler = compare(&group, &members, &templates, &mut topology);
        let count = compiler.recipes.len();
        compiler
            .plan_component(&group, &members, &templates, &mut topology, usize::MAX)
            .unwrap()
            .unwrap();
        assert_eq!(compiler.recipes.len(), count);
    }

    #[test]
    fn disjoint_owner_alternatives_aliases_and_duplicate_inputs() {
        let mut topology = topology();
        let a = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let not_a = topology.diagram.negate(a).unwrap();
        let z = PauliMap::from_unique_entries([(IVec2::ZERO, Pauli::Z)]);
        let mut templates = BloqTemplatePool::default();
        let creator = template(
            &mut templates,
            vec![Flow::new(PauliMap::empty(), z.clone()).with_measurements([0])],
        );
        let consumer = template(
            &mut templates,
            vec![Flow::new(z, PauliMap::empty()).with_measurements([0])],
        );
        let mut bloq = Bloq::new();
        let group = (0..2)
            .map(|_| bloq.add_node(BloqNode::from_members(vec![])))
            .collect::<Vec<_>>();
        let members = HashMap::from_iter([
            (
                group[0],
                vec![source(creator, 0, a), source(creator, 1, not_a)],
            ),
            (group[1], vec![source(consumer, 2, DECISION_TRUE)]),
        ]);
        compare(&group, &members, &templates, &mut topology);
        let b = topology
            .diagram
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let overlap = topology.diagram.apply(BooleanOp::And, a, b).unwrap();
        topology.domain = topology.diagram.negate(overlap).unwrap();
        let present = topology.diagram.apply(BooleanOp::Or, a, b).unwrap();
        let reachable_alternatives = HashMap::from_iter([
            (group[0], vec![source(creator, 0, a), source(creator, 1, b)]),
            (group[1], vec![source(consumer, 2, present)]),
        ]);
        compare(&group, &reachable_alternatives, &templates, &mut topology);
        topology.domain = DECISION_TRUE;
        let aliases = HashMap::from_iter([
            (group[0], vec![source(creator, 0, DECISION_TRUE)]),
            (group[1], vec![source(consumer, 0, DECISION_TRUE)]),
        ]);
        compare(&group, &aliases, &templates, &mut topology);
        let duplicates = HashMap::from_iter([
            (group[0], vec![source(creator, 0, DECISION_TRUE)]),
            (
                group[1],
                vec![
                    source(consumer, 1, DECISION_TRUE),
                    source(consumer, 2, DECISION_TRUE),
                ],
            ),
        ]);
        let mut compiler = FamilyCompiler::new(&templates, &mut topology, 3).unwrap();
        assert!(
            matches!(compiler.plan_component(&group, &duplicates, &templates, &mut topology, usize::MAX),
            Err(CompileError::BranchAssembly { reason }) if reason == "incoming flow boundaries are not independent")
        );
        let aliased_duplicates = HashMap::from_iter([
            (group[0], vec![source(creator, 0, DECISION_TRUE)]),
            (
                group[1],
                vec![
                    source(consumer, 1, DECISION_TRUE),
                    source(consumer, 1, DECISION_TRUE),
                ],
            ),
        ]);
        assert!(
            matches!(compiler.plan_component(&group, &aliased_duplicates, &templates, &mut topology, usize::MAX),
            Err(CompileError::BranchAssembly { reason }) if reason == "incoming flow boundaries are not independent")
        );
    }

    #[test]
    fn unsupported_recipe_keeps_spent_work_and_limits_stay_typed() {
        let mut topology = topology();
        let z = PauliMap::from_unique_entries([(IVec2::ZERO, Pauli::Z)]);
        let mut templates = BloqTemplatePool::default();
        let relay = template(&mut templates, vec![Flow::new(z.clone(), z.clone())]);
        let creator = template(
            &mut templates,
            vec![Flow::new(PauliMap::empty(), z.clone()).with_measurements([0])],
        );
        let consumer = template(
            &mut templates,
            vec![Flow::new(z, PauliMap::empty()).with_measurements([1])],
        );
        let mut bloq = Bloq::new();
        let group = (0..5)
            .map(|_| bloq.add_node(BloqNode::from_members(vec![])))
            .collect::<Vec<_>>();
        let members = HashMap::from_iter([
            (group[0], vec![source(creator, 0, DECISION_TRUE)]),
            (group[1], vec![source(consumer, 1, DECISION_TRUE)]),
            (group[2], vec![source(creator, 2, DECISION_TRUE)]),
            (group[3], vec![source(relay, 3, DECISION_TRUE)]),
            (group[4], vec![source(consumer, 4, DECISION_TRUE)]),
        ]);
        let before = topology.diagram.steps();
        let mut compiler = FamilyCompiler::new(&templates, &mut topology, 3).unwrap();
        assert!(
            compiler
                .plan_component(&group, &members, &templates, &mut topology, usize::MAX)
                .unwrap()
                .is_none()
        );
        assert!(topology.diagram.steps() > before);
        assert!(
            !compiler.recipes.is_empty(),
            "a planned closure precedes fallback"
        );
        let spent = topology.diagram.steps();
        let mut engine = GuardedFlowEngine::default();
        let mut checks = Vec::new();
        for node in &group {
            let flows = super::super::assembly::materialize_flows(
                &members[node],
                &templates,
                &mut topology,
            )
            .unwrap();
            checks.extend(
                engine
                    .append(&flows, &mut topology.diagram, topology.domain)
                    .unwrap(),
            );
        }
        engine
            .finish(&mut topology.diagram, topology.domain)
            .unwrap();
        assert_eq!(checks.len(), 2);
        assert!(topology.diagram.steps() > spent);
        topology.diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_steps: 0,
            ..BooleanLimits::default()
        });
        assert!(FamilyCompiler::new(&templates, &mut topology, 3).is_err());
    }

    #[test]
    fn ordering_guards_skip_and_extreme_normalization_keep_their_contracts() {
        let mut topology = topology();
        let a = topology
            .diagram
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let not_a = topology.diagram.negate(a).unwrap();
        let mut templates = BloqTemplatePool::default();
        let z = PauliMap::from_unique_entries([(IVec2::ZERO, Pauli::Z)]);
        let creator = template(
            &mut templates,
            vec![Flow::new(PauliMap::empty(), z.clone())],
        );
        let consumer = template(&mut templates, vec![Flow::new(z, PauliMap::empty())]);
        let mut bloq = Bloq::new();
        let producer_node = bloq.add_node(BloqNode::from_members(vec![]));
        let consumer_node = bloq.add_node(BloqNode::from_members(vec![]));
        let members = HashMap::from_iter([
            (producer_node, vec![source(creator, 0, a)]),
            (consumer_node, vec![source(consumer, 1, a)]),
        ]);
        let mut compiler = FamilyCompiler::new(&templates, &mut topology, 3).unwrap();
        let mut nodes = [producer_node, consumer_node];
        assert!(
            compiler
                .order_same_layer(&mut nodes, &members, &mut topology)
                .unwrap()
        );
        assert_eq!(nodes, [consumer_node, producer_node]);
        let disjoint = HashMap::from_iter([
            (producer_node, vec![source(creator, 0, a)]),
            (consumer_node, vec![source(consumer, 1, not_a)]),
        ]);
        let mut nodes = [producer_node, consumer_node];
        compiler
            .order_same_layer(&mut nodes, &disjoint, &mut topology)
            .unwrap();
        assert_eq!(nodes, [producer_node, consumer_node]);
        let skipping = HashMap::from_iter([(
            consumer_node,
            vec![FlowMember {
                skip_unmatched: true,
                ..source(consumer, 1, DECISION_TRUE)
            }],
        )]);
        assert!(
            compiler
                .plan_component(
                    &[consumer_node],
                    &skipping,
                    &templates,
                    &mut topology,
                    usize::MAX
                )
                .unwrap()
                .unwrap()
                .bundles
                .is_empty()
        );

        let partial = template(
            &mut templates,
            vec![
                Flow::new(
                    PauliMap::from_unique_entries([(IVec2::ZERO, Pauli::Z)]),
                    PauliMap::empty(),
                )
                .with_measurements([0]),
                Flow::new(
                    PauliMap::from_unique_entries([(IVec2::new(3, 0), Pauli::X)]),
                    PauliMap::empty(),
                )
                .with_measurements([1]),
            ],
        );
        let partial_skip = HashMap::from_iter([
            (producer_node, vec![source(creator, 0, DECISION_TRUE)]),
            (
                consumer_node,
                vec![FlowMember {
                    skip_unmatched: true,
                    ..source(partial, 1, DECISION_TRUE)
                }],
            ),
        ]);
        compare(
            &[producer_node, consumer_node],
            &partial_skip,
            &templates,
            &mut topology,
        );

        let mut extreme_templates = BloqTemplatePool::default();
        let left = template(
            &mut extreme_templates,
            vec![Flow::new(
                PauliMap::empty(),
                PauliMap::from_unique_entries([(IVec2::new(i32::MAX, 0), Pauli::Z)]),
            )],
        );
        let right = template(
            &mut extreme_templates,
            vec![Flow::new(
                PauliMap::from_unique_entries([(IVec2::new(i32::MIN, 0), Pauli::Z)]),
                PauliMap::empty(),
            )],
        );
        let left = FlowMember {
            offset: IVec2::new(i32::MIN, 0),
            ..source(left, 0, DECISION_TRUE)
        };
        let right = FlowMember {
            offset: IVec2::new(i32::MAX, 0),
            ..source(right, 1, DECISION_TRUE)
        };
        let extreme =
            HashMap::from_iter([(producer_node, vec![left]), (consumer_node, vec![right])]);
        let mut compiler = FamilyCompiler::new(&extreme_templates, &mut topology, 3).unwrap();
        assert!(compiler.coordinates_fit(left) && compiler.coordinates_fit(right));
        let spent = topology.diagram.steps();
        assert!(
            compiler
                .plan_component(
                    &[producer_node, consumer_node],
                    &extreme,
                    &extreme_templates,
                    &mut topology,
                    usize::MAX
                )
                .unwrap()
                .is_none()
        );
        assert!(topology.diagram.steps() > spent);
    }

    #[test]
    fn distance_grid_preserves_order_across_negative_and_positive_cells() {
        for distance in [3, 21] {
            let mut topology = topology();
            let stride = 2 * distance + 2;
            let radius = distance as i32;
            let support = PauliMap::from_unique_entries([
                (IVec2::new(-radius, -radius), Pauli::X),
                (IVec2::new(radius, radius), Pauli::X),
            ]);
            let mut templates = BloqTemplatePool::default();
            let creator = template(
                &mut templates,
                vec![Flow::new(PauliMap::empty(), support.clone())],
            );
            let consumer = template(&mut templates, vec![Flow::new(support, PauliMap::empty())]);
            let mut compiler = FamilyCompiler::new(&templates, &mut topology, distance).unwrap();
            assert_eq!(compiler.index.cell, i64::from(stride));
            let mut bloq = Bloq::new();
            let nodes = (0..4)
                .map(|_| bloq.add_node(BloqNode::from_members(vec![])))
                .collect::<Vec<_>>();
            let left = IVec2::splat(-(stride as i32));
            let right = IVec2::splat(stride as i32);
            let members = HashMap::from_iter([
                (
                    nodes[0],
                    vec![FlowMember {
                        offset: left,
                        ..source(creator, 0, DECISION_TRUE)
                    }],
                ),
                (
                    nodes[1],
                    vec![FlowMember {
                        offset: right,
                        ..source(creator, 1, DECISION_TRUE)
                    }],
                ),
                (
                    nodes[2],
                    vec![FlowMember {
                        offset: left,
                        ..source(consumer, 2, DECISION_TRUE)
                    }],
                ),
                (
                    nodes[3],
                    vec![FlowMember {
                        offset: right,
                        ..source(consumer, 3, DECISION_TRUE)
                    }],
                ),
            ]);
            let mut ordered = nodes.clone();
            assert!(
                compiler
                    .order_same_layer(&mut ordered, &members, &mut topology)
                    .unwrap()
            );
            assert_eq!(ordered, [nodes[2], nodes[0], nodes[3], nodes[1]]);
            assert!(
                Bounds {
                    low: [i32::MIN; 2],
                    high: [i32::MAX; 2]
                }
                .cells(IVec2::ZERO, compiler.index.cell)
                .is_none()
            );
        }
    }
}
