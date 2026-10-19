//! Reduced ordered Boolean decision diagrams shared by graph and IR guards.
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::hash::{Hash, Hasher};

/// A commutative binary operation over Boolean decision functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BooleanOp {
    /// Exclusive OR.
    Xor,
    /// Conjunction.
    And,
    /// Disjunction.
    Or,
}

/// Stable handle to a function in a [`BooleanDecisionDiagram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DecisionId(pub usize);
/// The constant-false decision.
pub const DECISION_FALSE: DecisionId = DecisionId(0);
/// The constant-true decision.
pub const DECISION_TRUE: DecisionId = DecisionId(1);

/// One nonterminal node in a reduced ordered decision diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecisionNode {
    /// Variable tested by this node.
    pub variable: usize,
    /// Function selected when the variable is false.
    pub low: DecisionId,
    /// Function selected when the variable is true.
    pub high: DecisionId,
}

/// Bounds on live decision nodes and cumulative Boolean/row work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanLimits {
    /// Nonterminal arena entries, including intermediates until collection.
    pub max_nodes: usize,
    /// Uncached visits, new nodes, and processed row/query coordinates.
    pub max_steps: usize,
}

impl BooleanLimits {
    /// Default production limits.
    pub const DEFAULT: Self = Self {
        max_nodes: 4_000_000,
        max_steps: 64_000_000,
    };
    /// Limits that accept every representable allocation and work count.
    pub const UNLIMITED: Self = Self {
        max_nodes: usize::MAX,
        max_steps: usize::MAX,
    };
}

impl Default for BooleanLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A Boolean operation exceeded its configured resource allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{resource} requires {observed}, limit is {limit}")]
pub struct BooleanResourceError {
    /// Resource whose allowance was exceeded.
    pub resource: &'static str,
    /// Required or observed resource count.
    pub observed: usize,
    /// Configured maximum count.
    pub limit: usize,
}

// Caches may forget results; exhaustion must never change Boolean semantics.
const MAX_CACHE_ENTRIES: usize = 262_144;
type ComputedKey = (BooleanOp, DecisionId, DecisionId);

#[derive(Clone, Copy)]
struct ComputedEntry {
    left: DecisionId,
    right: DecisionId,
    result: DecisionId,
    operator: BooleanOp,
}

impl ComputedEntry {
    const EMPTY: Self = Self {
        left: DecisionId(usize::MAX),
        right: DECISION_FALSE,
        result: DECISION_FALSE,
        operator: BooleanOp::Xor,
    };

    fn key(self) -> ComputedKey {
        (self.operator, self.left, self.right)
    }
}

#[derive(Clone, Default)]
struct ComputedTable {
    entries: Vec<ComputedEntry>,
}

impl std::fmt::Debug for ComputedTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputedTable")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl ComputedTable {
    fn slot((operator, left, right): ComputedKey, mask: usize) -> usize {
        let mut hash = rustc_hash::FxHasher::default();
        (operator as u8, left.0, right.0).hash(&mut hash);
        hash.finish() as usize & mask
    }

    fn get(&self, key: ComputedKey) -> Option<DecisionId> {
        let mask = self.entries.len().checked_sub(1)?;
        let entry = self.entries[Self::slot(key, mask)];
        (entry.key() == key).then_some(entry.result)
    }

    fn store(&mut self, entry: ComputedEntry) {
        let slot = Self::slot(entry.key(), self.entries.len() - 1);
        self.entries[slot] = entry;
    }

    fn insert(&mut self, key: ComputedKey, result: DecisionId, nodes: usize) {
        let capacity = nodes
            .saturating_mul(2)
            .clamp(16, MAX_CACHE_ENTRIES)
            .next_power_of_two();
        if capacity > self.entries.len() {
            let old = std::mem::replace(&mut self.entries, vec![ComputedEntry::EMPTY; capacity]);
            for entry in old {
                if entry.left != ComputedEntry::EMPTY.left {
                    self.store(entry);
                }
            }
        }
        self.store(ComputedEntry {
            operator: key.0,
            left: key.1,
            right: key.2,
            result,
        });
    }

    fn remap(&mut self, mapping: &[DecisionId]) {
        let capacity = self.entries.len();
        let old = std::mem::replace(&mut self.entries, vec![ComputedEntry::EMPTY; capacity]);
        let survives = |id: DecisionId| id.0 < 2 || mapping[id.0] != DECISION_FALSE;
        for entry in old {
            if entry.left != ComputedEntry::EMPTY.left
                && [entry.left, entry.right, entry.result]
                    .into_iter()
                    .all(survives)
            {
                self.store(ComputedEntry {
                    operator: entry.operator,
                    left: mapping[entry.left.0],
                    right: mapping[entry.right.0],
                    result: mapping[entry.result.0],
                });
            }
        }
    }
}

#[derive(Debug, Clone)]
enum ApplyTask {
    Visit(DecisionId, DecisionId),
    Finish(ComputedKey, usize),
}

/// Reduced ordered Boolean decision diagram with bounded construction work.
#[derive(Debug, Clone, Default)]
pub struct BooleanDecisionDiagram {
    nodes: Vec<DecisionNode>,
    unique: FxHashMap<DecisionNode, DecisionId>,
    apply_cache: ComputedTable,
    apply_tasks: Vec<ApplyTask>,
    apply_results: Vec<DecisionId>,
    negate_tasks: Vec<(DecisionId, bool)>,
    not_cache: Vec<DecisionId>,
    limits: BooleanLimits,
    steps: usize,
    generation: u64,
    row_witnesses: Option<RowWitnessTape>,
}

impl BooleanDecisionDiagram {
    /// Creates an empty diagram with explicit resource limits.
    pub fn with_limits(limits: BooleanLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Returns the configured resource limits.
    pub fn limits(&self) -> BooleanLimits {
        self.limits
    }

    /// Returns cumulative charged work.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Per-arena collection counter for invalidating decision-id caches.
    /// Every collection advances it, even if no nodes are removed.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Charge before traversal or allocation. Collection and cloning preserve
    /// consumed work; an error never stands in for a Boolean result.
    /// Cheap constant/cache returns do not consume steps. Callers must charge
    /// their own loops, including loops over constant Boolean expressions.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if charging `steps` exceeds the work limit.
    pub fn charge(&mut self, steps: usize) -> Result<(), BooleanResourceError> {
        let observed = self.steps.saturating_add(steps);
        if steps > self.limits.max_steps.saturating_sub(self.steps) {
            return Err(BooleanResourceError {
                resource: "Boolean work steps",
                observed,
                limit: self.limits.max_steps,
            });
        }
        self.steps = observed;
        Ok(())
    }

    /// Rebuild the diagram from all live roots, removing dead intermediate
    /// functions and retaining cache entries whose operands and results survive.
    /// Variable indices do not change.
    ///
    /// Every surviving `DecisionId` must be passed here: ids are renumbered,
    /// and any unlisted external ids become invalid. Collect only at a boundary
    /// where the caller owns the complete root set.
    pub fn collect_garbage<'a>(
        &mut self,
        roots: impl IntoIterator<Item = &'a mut DecisionId>,
    ) -> usize {
        self.generation = self.generation.wrapping_add(1);
        let mut roots = roots.into_iter().collect::<Vec<_>>();
        if let Some(tape) = &mut self.row_witnesses {
            roots.extend(tape.decisions_mut());
        }
        let mut live = vec![false; self.nodes.len()];
        let mut pending = Vec::new();
        for root in &roots {
            pending.push(**root);
            while let Some(root) = pending.pop() {
                let Some(index) = root.0.checked_sub(2) else {
                    continue;
                };
                if !std::mem::replace(&mut live[index], true) {
                    let node = self.nodes[index];
                    pending.extend([node.low, node.high]);
                }
            }
        }
        let retained = live.iter().filter(|&&live| live).count();
        let removed = self.nodes.len() - retained;
        if removed == 0 {
            return 0;
        }
        let mut mapping = vec![DECISION_FALSE; self.nodes.len() + 2];
        mapping[1] = DECISION_TRUE;
        let mut compact = Self {
            nodes: Vec::with_capacity(retained),
            unique: FxHashMap::with_capacity_and_hasher(retained, Default::default()),
            not_cache: vec![DECISION_FALSE; retained],
            limits: self.limits,
            steps: self.steps,
            generation: self.generation,
            ..Self::default()
        };
        // A node is allocated after its children. Retaining this order also
        // preserves the relative order of live ids used by deterministic passes.
        for (index, node) in self
            .nodes
            .iter()
            .enumerate()
            .filter(|(index, _)| live[*index])
        {
            let node = DecisionNode {
                variable: node.variable,
                low: mapping[node.low.0],
                high: mapping[node.high.0],
            };
            let id = DecisionId(compact.nodes.len() + 2);
            mapping[index + 2] = id;
            compact.nodes.push(node);
            compact.unique.insert(node, id);
        }
        // A cached negation is usable only when both functions survived.
        // The cache does not keep either endpoint alive on its own.
        for (index, &negated) in self.not_cache.iter().enumerate() {
            if live[index]
                && let Some(other) = negated.0.checked_sub(2)
                && live[other]
            {
                compact.not_cache[mapping[index + 2].0 - 2] = mapping[negated.0];
            }
        }
        for root in roots {
            *root = mapping[root.0];
        }
        compact.row_witnesses = self.row_witnesses.take();
        compact.apply_cache = std::mem::take(&mut self.apply_cache);
        compact.apply_cache.remap(&mapping);
        compact.apply_tasks = std::mem::take(&mut self.apply_tasks);
        compact.apply_results = std::mem::take(&mut self.apply_results);
        compact.negate_tasks = std::mem::take(&mut self.negate_tasks);
        *self = compact;
        removed
    }

    /// Evaluate one Boolean function without traversing unrelated choices.
    pub fn evaluate(&self, mut root: DecisionId, mut variable: impl FnMut(usize) -> bool) -> bool {
        while let Some(node) = self.node(root) {
            root = if variable(node.variable) {
                node.high
            } else {
                node.low
            };
        }
        root == DECISION_TRUE
    }

    /// One satisfying variable assignment; omitted variables are don't-cares.
    pub fn witness(&self, mut root: DecisionId) -> Option<BTreeMap<usize, bool>> {
        if root == DECISION_FALSE {
            return None;
        }
        let mut values = BTreeMap::new();
        while let Some(node) = self.node(root) {
            let high = node.low == DECISION_FALSE;
            values.insert(node.variable, high);
            root = if high { node.high } else { node.low };
        }
        Some(values)
    }

    /// Finds one assignment satisfying every supplied decision.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if the search exceeds the work limit.
    pub fn conjunction_witness(
        &mut self,
        roots: &[DecisionId],
    ) -> Result<Option<BTreeMap<usize, bool>>, BooleanResourceError> {
        enum Task {
            Visit(Vec<DecisionId>),
            Retry(Vec<DecisionId>, usize),
            Finish(Vec<DecisionId>, usize),
        }
        self.charge(roots.len().max(1))?;
        let mut pending = vec![Task::Visit(roots.to_vec())];
        let mut values = BTreeMap::new();
        let mut dead = FxHashSet::default();
        let mut cached_terms = 0;
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit(roots) => {
                    self.charge(roots.len().max(1))?;
                    if roots.contains(&DECISION_FALSE) || dead.contains(&roots) {
                        continue;
                    }
                    let Some(variable) = roots
                        .iter()
                        .filter_map(|&root| self.node(root).map(|node| node.variable))
                        .min()
                    else {
                        return Ok(Some(values));
                    };
                    self.charge(roots.len().max(1))?;
                    let next = self.cofactors(&roots, variable, false);
                    values.insert(variable, false);
                    pending.push(Task::Retry(roots, variable));
                    pending.push(Task::Visit(next));
                }
                Task::Retry(roots, variable) => {
                    self.charge(roots.len().max(1))?;
                    let next = self.cofactors(&roots, variable, true);
                    values.insert(variable, true);
                    pending.push(Task::Finish(roots, variable));
                    pending.push(Task::Visit(next));
                }
                Task::Finish(roots, variable) => {
                    values.remove(&variable);
                    self.charge(roots.len().max(1))?;
                    if cached_terms + roots.len() > MAX_CACHE_ENTRIES {
                        dead.clear();
                        cached_terms = 0;
                    }
                    if roots.len() <= MAX_CACHE_ENTRIES {
                        cached_terms += roots.len();
                        dead.insert(roots);
                    }
                }
            }
        }
        Ok(None)
    }

    fn cofactors(&self, roots: &[DecisionId], variable: usize, high: bool) -> Vec<DecisionId> {
        roots
            .iter()
            .map(|&root| {
                self.node(root)
                    .filter(|node| node.variable == variable)
                    .map_or(root, |node| if high { node.high } else { node.low })
            })
            .collect()
    }

    /// Returns every variable reachable from `root`.
    pub fn variables(&self, root: DecisionId) -> BTreeSet<usize> {
        let mut variables = BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut pending = vec![root];
        while let Some(value) = pending.pop() {
            if visited.insert(value)
                && let Some(node) = self.node(value)
            {
                variables.insert(node.variable);
                pending.extend([node.low, node.high]);
            }
        }
        variables
    }

    /// Simplify a function outside its care set. The returned function equals
    /// `root` wherever `care` holds; impossible selector combinations are free.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if construction exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if either decision does not belong to this diagram.
    pub fn constrain(
        &mut self,
        root: DecisionId,
        care: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        if care == DECISION_FALSE {
            return Ok(DECISION_FALSE);
        }
        if root.0 < 2 || care == DECISION_TRUE {
            return Ok(root);
        }
        self.charge(1)?;
        enum Task {
            Visit(DecisionId, DecisionId),
            Finish((DecisionId, DecisionId), Option<usize>),
        }
        let mut pending = vec![Task::Visit(root, care)];
        let mut results = Vec::new();
        let mut memo = FxHashMap::default();
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit(root, care) => {
                    if care == DECISION_FALSE {
                        results.push(DECISION_FALSE);
                        continue;
                    }
                    if root.0 < 2 || care == DECISION_TRUE {
                        results.push(root);
                        continue;
                    }
                    if let Some(&value) = memo.get(&(root, care)) {
                        results.push(value);
                        continue;
                    }
                    self.charge(1)?;
                    let a = self.node(root).expect("nonterminal root has a node");
                    let b = self.node(care).expect("nonterminal care has a node");
                    let variable = a.variable.min(b.variable);
                    let (al, ah) = Self::branches(a, root, variable);
                    let (bl, bh) = Self::branches(b, care, variable);
                    if bl == DECISION_FALSE {
                        pending.push(Task::Finish((root, care), None));
                        pending.push(Task::Visit(ah, bh));
                    } else if bh == DECISION_FALSE {
                        pending.push(Task::Finish((root, care), None));
                        pending.push(Task::Visit(al, bl));
                    } else {
                        pending.push(Task::Finish((root, care), Some(variable)));
                        pending.push(Task::Visit(ah, bh));
                        pending.push(Task::Visit(al, bl));
                    }
                }
                Task::Finish(key, variable) => {
                    let high = results.pop().expect("visited child has a result");
                    let value = if let Some(variable) = variable {
                        let low = results.pop().expect("visited child has a result");
                        self.make_node(variable, low, high)?
                    } else {
                        high
                    };
                    if memo.len() == MAX_CACHE_ENTRIES {
                        memo.clear();
                    }
                    memo.insert(key, value);
                    results.push(value);
                }
            }
        }
        Ok(results.pop().expect("root has a result"))
    }

    fn branches(node: DecisionNode, root: DecisionId, variable: usize) -> (DecisionId, DecisionId) {
        if node.variable == variable {
            (node.low, node.high)
        } else {
            (root, root)
        }
    }

    /// Returns the nonterminal nodes in decision-id order.
    pub fn nodes(&self) -> &[DecisionNode] {
        &self.nodes
    }
    /// Returns the reduced node for `variable`, `low`, and `high`.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if allocation exceeds configured limits.
    pub fn make_node(
        &mut self,
        variable: usize,
        low: DecisionId,
        high: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        if low == high {
            return Ok(low);
        }
        let node = DecisionNode {
            variable,
            low,
            high,
        };
        if let Some(&id) = self.unique.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= self.limits.max_nodes {
            return Err(BooleanResourceError {
                resource: "Boolean nodes",
                observed: self.nodes.len().saturating_add(1),
                limit: self.limits.max_nodes,
            });
        }
        self.charge(1)?;
        let id = DecisionId(self.nodes.len() + 2);
        self.nodes.push(node);
        self.not_cache.push(DECISION_FALSE);
        self.unique.insert(node, id);
        Ok(id)
    }

    /// Returns the nonterminal node named by `id`, or `None` for a terminal.
    ///
    /// # Panics
    ///
    /// Panics if `id` names a nonterminal outside this diagram.
    pub fn node(&self, id: DecisionId) -> Option<DecisionNode> {
        id.0.checked_sub(2).map(|index| self.nodes[index])
    }

    /// Returns an already known exact complement without constructing nodes.
    /// Constants always have known complements.
    ///
    /// # Panics
    ///
    /// Panics if `root` does not belong to this diagram.
    pub fn cached_complement(&self, root: DecisionId) -> Option<DecisionId> {
        match root {
            DECISION_FALSE => Some(DECISION_TRUE),
            DECISION_TRUE => Some(DECISION_FALSE),
            _ => {
                let result = self.not_cache[root.0 - 2];
                (result != DECISION_FALSE).then_some(result)
            }
        }
    }

    /// Returns the logical negation of `root`.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if construction exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if `root` does not belong to this diagram.
    pub fn negate(&mut self, root: DecisionId) -> Result<DecisionId, BooleanResourceError> {
        if let Some(result) = self.cached_complement(root) {
            return Ok(result);
        }
        self.charge(1)?;
        let mut pending = std::mem::take(&mut self.negate_tasks);
        pending.push((root, false));
        let result = (|| {
            while let Some((root, finish)) = pending.pop() {
                if self.cached_complement(root).is_some() {
                    continue;
                }
                let node = self.node(root).expect("nonterminal decision has a node");
                if finish {
                    let low = self
                        .cached_complement(node.low)
                        .expect("visited child is negated");
                    let high = self
                        .cached_complement(node.high)
                        .expect("visited child is negated");
                    let negated = self.make_node(node.variable, low, high)?;
                    self.not_cache[root.0 - 2] = negated;
                    self.not_cache[negated.0 - 2] = root;
                } else {
                    self.charge(1)?;
                    pending.push((root, true));
                    pending.push((node.high, false));
                    pending.push((node.low, false));
                }
            }
            Ok(self.cached_complement(root).expect("root is negated"))
        })();
        pending.clear();
        self.negate_tasks = pending;
        result
    }

    /// Fold constants before visiting decision pairs. All operators commute.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if construction exceeds configured limits.
    #[inline]
    pub fn apply(
        &mut self,
        operator: BooleanOp,
        mut left: DecisionId,
        mut right: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        if left > right {
            std::mem::swap(&mut left, &mut right);
        }
        if let Some(value) = self.simple_apply(operator, left, right)? {
            return Ok(value);
        }
        self.apply_nodes(operator, left, right)
    }

    #[inline]
    fn simple_apply(
        &mut self,
        operator: BooleanOp,
        left: DecisionId,
        right: DecisionId,
    ) -> Result<Option<DecisionId>, BooleanResourceError> {
        use BooleanOp::{And, Or, Xor};
        Ok(match operator {
            Xor if left == DECISION_FALSE => Some(right),
            Xor if left == right => Some(DECISION_FALSE),
            Xor if left == DECISION_TRUE => Some(self.negate(right)?),
            And if left == DECISION_FALSE || left == right => Some(left),
            And if left == DECISION_TRUE => Some(right),
            Or if left == DECISION_FALSE || left == right => Some(right),
            Or if left == DECISION_TRUE => Some(left),
            Xor | And | Or => None,
        })
    }

    fn apply_nodes(
        &mut self,
        operator: BooleanOp,
        left: DecisionId,
        right: DecisionId,
    ) -> Result<DecisionId, BooleanResourceError> {
        if let Some(result) = self.apply_cache.get((operator, left, right)) {
            return Ok(result);
        }
        self.charge(1)?;
        let mut pending = std::mem::take(&mut self.apply_tasks);
        let mut results = std::mem::take(&mut self.apply_results);
        pending.push(ApplyTask::Visit(left, right));
        let result = (|| {
            while let Some(task) = pending.pop() {
                match task {
                    ApplyTask::Visit(mut left, mut right) => {
                        if left > right {
                            std::mem::swap(&mut left, &mut right);
                        }
                        if let Some(value) = self.simple_apply(operator, left, right)? {
                            results.push(value);
                            continue;
                        }
                        let key = (operator, left, right);
                        if let Some(value) = self.apply_cache.get(key) {
                            results.push(value);
                            continue;
                        }
                        self.charge(1)?;
                        let a = self.node(left).expect("terminal cases were simplified");
                        let b = self.node(right).expect("terminal cases were simplified");
                        let variable = a.variable.min(b.variable);
                        let (al, ah) = Self::branches(a, left, variable);
                        let (bl, bh) = Self::branches(b, right, variable);
                        pending.push(ApplyTask::Finish(key, variable));
                        pending.push(ApplyTask::Visit(ah, bh));
                        pending.push(ApplyTask::Visit(al, bl));
                    }
                    ApplyTask::Finish(key, variable) => {
                        let high = results.pop().expect("visited child has a result");
                        let low = results.pop().expect("visited child has a result");
                        let result = self.make_node(variable, low, high)?;
                        self.apply_cache.insert(key, result, self.nodes.len());
                        results.push(result);
                    }
                }
            }
            Ok(results.pop().expect("root has a result"))
        })();
        pending.clear();
        results.clear();
        self.apply_tasks = pending;
        self.apply_results = results;
        result
    }

    /// Enumerate at most `limit` distinct output tuples, charging residual
    /// states and their coordinates before allocating them.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if enumeration exceeds the work limit.
    pub fn values_up_to(
        &mut self,
        roots: &[DecisionId],
        limit: usize,
    ) -> Result<Vec<Vec<bool>>, BooleanResourceError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.charge(roots.len().max(1))?;
        let mut reachable = BTreeSet::new();
        let mut seen = FxHashSet::default();
        let mut cached_terms = 0;
        let mut pending = vec![roots.to_vec()];
        while let Some(state) = pending.pop() {
            self.charge(state.len().max(1))?;
            if seen.contains(&state) {
                continue;
            }
            if cached_terms + state.len() > MAX_CACHE_ENTRIES {
                seen.clear();
                cached_terms = 0;
            }
            if state.len() <= MAX_CACHE_ENTRIES {
                self.charge(state.len().max(1))?;
                cached_terms += state.len();
                seen.insert(state.clone());
            }
            let variable = state
                .iter()
                .filter_map(|&root| self.node(root).map(|node| node.variable))
                .min();
            let Some(variable) = variable else {
                self.charge(state.len().max(1))?;
                reachable.insert(
                    state
                        .into_iter()
                        .map(|root| root == DECISION_TRUE)
                        .collect(),
                );
                if reachable.len() == limit {
                    break;
                }
                continue;
            };
            for high in [false, true] {
                self.charge(state.len().max(1))?;
                pending.push(self.cofactors(&state, variable, high));
            }
        }
        Ok(reachable.into_iter().collect())
    }
}

/// A sparse GF(2) row whose entries and availability are Boolean functions.
/// Extra columns can carry witnesses through exactly the same row operations.
#[derive(Debug, Clone)]
pub struct BooleanRow {
    /// Decision under which this row is available.
    pub active: DecisionId,
    // Sorted, unique columns with nonzero coefficients. Row algebra merges
    // contiguous terms instead of allocating and searching a tree per term.
    bits: Vec<(usize, DecisionId)>,
    witness: RowWitness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowWitness {
    Explicit,
    Deferred(Option<usize>),
}

#[derive(Debug, Clone)]
enum RowWitnessOp {
    Leaf(Vec<(usize, DecisionId)>),
    Xor {
        left: Option<usize>,
        right: usize,
        factor: DecisionId,
    },
}

#[derive(Debug, Clone)]
struct RowWitnessNode {
    op: RowWitnessOp,
    // Conservative inclusive bounds; cancellation may shrink actual support.
    columns: (usize, usize),
}

#[derive(Debug, Clone, Default)]
struct WitnessExpansionScratch {
    weights: Vec<DecisionId>,
    stamps: Vec<u32>,
    epoch: u32,
    pending: BinaryHeap<usize>,
    coefficients: Vec<DecisionId>,
    column_stamps: Vec<u32>,
    touched: Vec<usize>,
}

#[derive(Debug, Clone)]
struct RowWitnessTape {
    explicit_columns: FxHashSet<usize>,
    initial_explicit_columns: usize,
    ops: Vec<RowWitnessNode>,
    limit: usize,
    roots: usize,
    expansion: WitnessExpansionScratch,
}

impl RowWitnessTape {
    fn insertion_work(&self) -> Result<usize, BooleanResourceError> {
        if self.ops.len() >= self.limit {
            return Err(BooleanResourceError {
                resource: "row witness operations",
                observed: self.ops.len().saturating_add(1),
                limit: self.limit,
            });
        }
        Ok(1 + if self.ops.len() == self.ops.capacity() {
            self.ops.len()
        } else {
            0
        })
    }

    // Callers reserve work and check the operation bound before changing a row.
    fn push(&mut self, op: RowWitnessOp) -> usize {
        debug_assert!(self.ops.len() < self.limit);
        let columns = match &op {
            RowWitnessOp::Leaf(terms) => (
                terms.first().expect("witness leaves are nonempty").0,
                terms.last().expect("witness leaves are nonempty").0,
            ),
            RowWitnessOp::Xor { left, right, .. } => {
                let (first, last) = self.ops[*right].columns;
                left.map_or((first, last), |left| {
                    let (a, b) = self.ops[left].columns;
                    (a.min(first), b.max(last))
                })
            }
        };
        self.roots += match &op {
            RowWitnessOp::Leaf(terms) => terms.len(),
            RowWitnessOp::Xor { .. } => 1,
        };
        let id = self.ops.len();
        self.ops.push(RowWitnessNode { op, columns });
        id
    }

    fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        self.ops.iter_mut().flat_map(|node| {
            let (terms, factor) = match &mut node.op {
                RowWitnessOp::Leaf(terms) => (Some(terms), None),
                RowWitnessOp::Xor { factor, .. } => (None, Some(factor)),
            };
            terms
                .into_iter()
                .flatten()
                .map(|(_, root)| root)
                .chain(factor)
        })
    }

    fn reconstruct_columns(
        &self,
        root: usize,
        requested: &[usize],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<(usize, DecisionId)>, BooleanResourceError> {
        diagram.charge(1)?;
        let extended = self.explicit_columns.len() != self.initial_explicit_columns;
        // Parents are newer than children. Coalesce every path's scale before
        // visiting a shared child; never materialize intermediate witness rows.
        let mut pending = BTreeMap::from([(root, DECISION_TRUE)]);
        let mut result = FxHashMap::<usize, DecisionId>::default();
        while let Some((id, weight)) = pending.pop_last() {
            diagram.charge(1)?;
            if weight == DECISION_FALSE {
                continue;
            }
            let node = &self.ops[id];
            diagram.charge(2 * (requested.len().max(1).ilog2() as usize + 1))?;
            let first = requested.partition_point(|&column| column < node.columns.0);
            let last = requested.partition_point(|&column| column <= node.columns.1);
            if first == last {
                continue;
            }
            let requested = &requested[first..last];
            match &node.op {
                RowWitnessOp::Leaf(terms) => {
                    diagram.charge(
                        requested
                            .len()
                            .saturating_mul(terms.len().ilog2() as usize + 1),
                    )?;
                    for &column in requested {
                        if extended && self.explicit_columns.contains(&column) {
                            continue;
                        }
                        if let Ok(index) = terms.binary_search_by_key(&column, |term| term.0) {
                            let value = diagram.apply(BooleanOp::And, weight, terms[index].1)?;
                            let previous = result.entry(column).or_insert(DECISION_FALSE);
                            *previous = diagram.apply(BooleanOp::Xor, *previous, value)?;
                        }
                    }
                }
                RowWitnessOp::Xor {
                    left,
                    right,
                    factor,
                } => {
                    let scaled = diagram.apply(BooleanOp::And, weight, *factor)?;
                    for (child, value) in left
                        .map(|id| (id, weight))
                        .into_iter()
                        .chain(std::iter::once((*right, scaled)))
                    {
                        if value == DECISION_FALSE {
                            continue;
                        }
                        diagram.charge(1)?;
                        let previous = pending.entry(child).or_insert(DECISION_FALSE);
                        *previous = diagram.apply(BooleanOp::Xor, *previous, value)?;
                    }
                }
            }
        }
        diagram.charge(result.len())?;
        let mut terms = Vec::with_capacity(result.len());
        terms.extend(
            result
                .into_iter()
                .filter(|(_, value)| *value != DECISION_FALSE),
        );
        diagram.charge(
            terms
                .len()
                .saturating_mul(terms.len().max(1).ilog2() as usize + 1),
        )?;
        terms.sort_unstable_by_key(|&(column, _)| column);
        Ok(terms)
    }

    fn reconstruct_all(
        &mut self,
        root: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<(usize, DecisionId)>, BooleanResourceError> {
        let extended = self.explicit_columns.len() != self.initial_explicit_columns;
        // Sparse coordinate domains must not allocate up to their largest column.
        let dense = self.ops[root]
            .columns
            .1
            .checked_add(1)
            .is_some_and(|width| width <= self.roots);
        let scratch = &mut self.expansion;
        let result = (|| {
            diagram.charge(1)?;
            if scratch.weights.len() <= root {
                diagram.charge((root + 1 - scratch.weights.len()).saturating_mul(2))?;
                scratch.weights.resize(root + 1, DECISION_FALSE);
                scratch.stamps.resize(root + 1, 0);
            }
            let mut epoch = scratch.epoch.wrapping_add(1);
            if epoch == 0 {
                diagram.charge(
                    scratch
                        .stamps
                        .len()
                        .saturating_add(scratch.column_stamps.len()),
                )?;
                scratch.stamps.fill(0);
                scratch.column_stamps.fill(0);
                epoch = 1;
            }
            scratch.epoch = epoch;
            scratch.stamps[root] = epoch;
            scratch.weights[root] = DECISION_TRUE;
            scratch.pending.push(root);
            let mut sparse = FxHashMap::<usize, DecisionId>::default();
            while let Some(id) = scratch.pending.pop() {
                diagram.charge(1)?;
                let weight = scratch.weights[id];
                if weight == DECISION_FALSE {
                    continue;
                }
                match &self.ops[id].op {
                    RowWitnessOp::Leaf(terms) => {
                        diagram.charge(terms.len())?;
                        for &(column, value) in terms {
                            if extended && self.explicit_columns.contains(&column) {
                                continue;
                            }
                            let value = diagram.apply(BooleanOp::And, weight, value)?;
                            let previous = if dense {
                                if scratch.coefficients.len() <= column {
                                    diagram.charge(
                                        (column + 1 - scratch.coefficients.len()).saturating_mul(2),
                                    )?;
                                    scratch.coefficients.resize(column + 1, DECISION_FALSE);
                                    scratch.column_stamps.resize(column + 1, 0);
                                }
                                if scratch.column_stamps[column] != epoch {
                                    scratch.column_stamps[column] = epoch;
                                    scratch.coefficients[column] = DECISION_FALSE;
                                    scratch.touched.push(column);
                                }
                                &mut scratch.coefficients[column]
                            } else {
                                sparse.entry(column).or_insert(DECISION_FALSE)
                            };
                            *previous = diagram.apply(BooleanOp::Xor, *previous, value)?;
                        }
                    }
                    RowWitnessOp::Xor {
                        left,
                        right,
                        factor,
                    } => {
                        let scaled = diagram.apply(BooleanOp::And, weight, *factor)?;
                        for (child, value) in left
                            .map(|id| (id, weight))
                            .into_iter()
                            .chain(std::iter::once((*right, scaled)))
                        {
                            if value == DECISION_FALSE {
                                continue;
                            }
                            diagram.charge(1)?;
                            if scratch.stamps[child] != epoch {
                                scratch.stamps[child] = epoch;
                                scratch.weights[child] = DECISION_FALSE;
                                scratch.pending.push(child);
                            }
                            scratch.weights[child] =
                                diagram.apply(BooleanOp::Xor, scratch.weights[child], value)?;
                        }
                    }
                }
            }
            let count = if dense {
                scratch.touched.len()
            } else {
                sparse.len()
            };
            diagram.charge(count)?;
            let mut terms = Vec::with_capacity(count);
            if dense {
                terms.extend(scratch.touched.iter().filter_map(|&column| {
                    let value = scratch.coefficients[column];
                    (value != DECISION_FALSE).then_some((column, value))
                }));
            } else {
                terms.extend(
                    sparse
                        .into_iter()
                        .filter(|&(_, value)| value != DECISION_FALSE),
                );
            }
            diagram.charge(
                terms
                    .len()
                    .saturating_mul(terms.len().max(1).ilog2() as usize + 1),
            )?;
            terms.sort_unstable_by_key(|&(column, _)| column);
            Ok(terms)
        })();
        // Every replay advances the stamps; stale scratch handles are never GC roots.
        scratch.pending.clear();
        scratch.touched.clear();
        result
    }

    /// Propagate each requested coordinate through shared transfer uses once.
    /// Only nonzero contributions enter the current coordinate's worklist;
    /// independent components do not form a row-by-column scan.
    fn project_roots(
        &self,
        roots: &[Option<usize>],
        columns: &[usize],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<Vec<(usize, DecisionId)>>, BooleanResourceError> {
        diagram.charge(
            self.ops
                .len()
                .saturating_mul(4)
                .saturating_add(roots.len().saturating_mul(2)),
        )?;
        let mut needed = vec![false; self.ops.len()];
        let mut subscribers = FxHashMap::<usize, Vec<usize>>::default();
        for (index, &root) in roots.iter().enumerate() {
            if let Some(root) = root {
                needed[root] = true;
                diagram.charge(1)?;
                subscribers.entry(root).or_default().push(index);
            }
        }
        let mut parents = vec![Vec::new(); self.ops.len()];
        for (id, node) in self.ops.iter().enumerate().rev() {
            if needed[id]
                && let RowWitnessOp::Xor {
                    left,
                    right,
                    factor,
                } = node.op
            {
                for (child, scale) in left
                    .map(|left| (left, DECISION_TRUE))
                    .into_iter()
                    .chain(std::iter::once((right, factor)))
                {
                    diagram.charge(
                        1 + if parents[child].len() == parents[child].capacity() {
                            parents[child].len()
                        } else {
                            0
                        },
                    )?;
                    needed[child] = true;
                    parents[child].push((id, scale));
                }
            }
        }
        let mut seeds = BTreeMap::<usize, Vec<(usize, DecisionId)>>::new();
        for (id, node) in self.ops.iter().enumerate() {
            let RowWitnessOp::Leaf(terms) = &node.op else {
                continue;
            };
            if !needed[id] {
                continue;
            }
            diagram.charge(2 * (columns.len().max(1).ilog2() as usize + 1))?;
            let first = columns.partition_point(|&column| column < node.columns.0);
            let last = columns.partition_point(|&column| column <= node.columns.1);
            let requested = &columns[first..last];
            let mut seed = |column, value, diagram: &mut BooleanDecisionDiagram| {
                diagram.charge(seeds.len().max(1).ilog2() as usize + 1)?;
                let entries = seeds.entry(column).or_default();
                diagram.charge(
                    1 + if entries.len() == entries.capacity() {
                        entries.len()
                    } else {
                        0
                    },
                )?;
                entries.push((id, value));
                Ok::<_, BooleanResourceError>(())
            };
            if requested.len() <= terms.len() {
                diagram.charge(
                    requested
                        .len()
                        .saturating_mul(terms.len().ilog2() as usize + 1),
                )?;
                for &column in requested {
                    if let Ok(index) = terms.binary_search_by_key(&column, |term| term.0) {
                        seed(column, terms[index].1, diagram)?;
                    }
                }
            } else {
                diagram.charge(
                    terms
                        .len()
                        .saturating_mul(requested.len().ilog2() as usize + 1),
                )?;
                for &(column, value) in terms {
                    if requested.binary_search(&column).is_ok() {
                        seed(column, value, diagram)?;
                    }
                }
            }
        }
        let mut result = vec![Vec::new(); roots.len()];
        for (column, leaves) in seeds {
            diagram.charge(
                leaves
                    .len()
                    .saturating_mul(leaves.len().max(1).ilog2() as usize + 1),
            )?;
            let mut pending = leaves.into_iter().collect::<BTreeMap<_, _>>();
            while !pending.is_empty() {
                diagram.charge(pending.len().ilog2() as usize + 1)?;
                let (id, value) = pending.pop_first().expect("nonempty query worklist");
                if value == DECISION_FALSE {
                    continue;
                }
                if let Some(rows) = subscribers.get(&id) {
                    diagram.charge(rows.len())?;
                    for &index in rows {
                        let front = &mut result[index];
                        diagram.charge(
                            1 + if front.len() == front.capacity() {
                                front.len()
                            } else {
                                0
                            },
                        )?;
                        front.push((column, value));
                    }
                }
                diagram.charge(parents[id].len())?;
                for &(parent, scale) in &parents[id] {
                    let contribution = diagram.apply(BooleanOp::And, value, scale)?;
                    if contribution != DECISION_FALSE {
                        diagram.charge(pending.len().max(1).ilog2() as usize + 1)?;
                        let previous = pending.entry(parent).or_insert(DECISION_FALSE);
                        *previous = diagram.apply(BooleanOp::Xor, *previous, contribution)?;
                    }
                }
            }
        }
        Ok(result)
    }
}

impl BooleanDecisionDiagram {
    /// Start an owned tape for passive row coefficients. Every deferred row
    /// shares this explicit-column partition; keep every row-space constraint
    /// column explicit until all rows are expanded. Passive coordinates can be
    /// read with [`BooleanRow::probe_coefficients`]. Operation bounds include
    /// dead history, and construction/reconstruction spend the same work meter.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if indexing columns exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if a witness tape is already active.
    pub fn start_witness_tape(
        &mut self,
        explicit_columns: impl IntoIterator<Item = usize>,
        max_operations: usize,
    ) -> Result<(), BooleanResourceError> {
        assert!(
            self.row_witnesses.is_none(),
            "one tape owns this arena's rows"
        );
        let mut columns = FxHashSet::default();
        for column in explicit_columns {
            self.charge(1)?;
            if columns.len() == columns.capacity() && !columns.contains(&column) {
                self.charge(columns.len())?;
            }
            columns.insert(column);
        }
        self.row_witnesses = Some(RowWitnessTape {
            initial_explicit_columns: columns.len(),
            explicit_columns: columns,
            ops: Vec::new(),
            limit: max_operations,
            roots: 0,
            expansion: WitnessExpansionScratch::default(),
        });
        Ok(())
    }

    /// Promote coordinates into the shared explicit row front without expanding
    /// other coefficients. The caller must supply every surviving deferred row
    /// owned by this tape, including rows retained for later queries or replay.
    /// No deferred clone outside `rows` may be used after this boundary. Explicit
    /// rows may also be supplied and remain unchanged.
    ///
    /// Raw coefficients, including off-activation values, are queried under the
    /// old partition. Row fronts and the partition change only after all work
    /// succeeds; a failure retains the tape and rows without refunding spent work.
    /// Activation and witness handles never change, and columns are never demoted.
    /// Fixed roots share one topological evaluation per new column; intermediate
    /// values for different columns are never stored together.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if querying or staging exceeds resource limits.
    ///
    /// # Panics
    ///
    /// Panics if no witness tape is active.
    pub fn extend_witness_front(
        &mut self,
        columns: &[usize],
        rows: &mut [BooleanRow],
    ) -> Result<(), BooleanResourceError> {
        self.row_witnesses
            .as_ref()
            .expect("row witness tape started");
        self.charge(
            columns
                .len()
                .saturating_mul(columns.len().max(1).ilog2() as usize + 2)
                .saturating_add(rows.len()),
        )?;
        let mut columns = columns.to_vec();
        columns.sort_unstable();
        columns.dedup();
        let tape = self.row_witnesses.as_ref().expect("tape remains active");
        columns.retain(|column| !tape.explicit_columns.contains(column));
        if columns.is_empty() {
            return Ok(());
        }

        self.charge(rows.len())?;
        let roots = rows
            .iter()
            .map(|row| match row.witness {
                RowWitness::Explicit => None,
                RowWitness::Deferred(root) => root,
            })
            .collect::<Vec<_>>();
        let tape = self.row_witnesses.take().expect("tape remains active");
        let projections = tape.project_roots(&roots, &columns, self);
        self.row_witnesses = Some(tape);
        let projections = projections?;
        let mut staged = Vec::with_capacity(rows.len());
        for (row, values) in rows.iter().zip(projections) {
            if row.witness == RowWitness::Explicit {
                staged.push(None);
                continue;
            }
            let capacity = row.bits.len().saturating_add(values.len());
            self.charge(capacity)?;
            let mut merged = Vec::with_capacity(capacity);
            let mut previous = row.bits.iter().copied().peekable();
            for (column, value) in values {
                while previous.peek().is_some_and(|&(old, _)| old < column) {
                    merged.push(previous.next().expect("peek found a preceding term"));
                }
                debug_assert!(previous.peek().is_none_or(|&(old, _)| old != column));
                merged.push((column, value));
            }
            merged.extend(previous);
            staged.push(Some(merged));
        }

        let capacity = self
            .row_witnesses
            .as_ref()
            .expect("queries retain the witness tape")
            .explicit_columns
            .len()
            .saturating_add(columns.len());
        self.charge(capacity)?;
        let tape = self.row_witnesses.as_mut().expect("tape remains active");
        let mut partition = FxHashSet::with_capacity_and_hasher(capacity, Default::default());
        partition.extend(tape.explicit_columns.iter().copied());
        partition.extend(columns);
        tape.explicit_columns = partition;
        for (row, front) in rows.iter_mut().zip(staged) {
            if let Some(front) = front {
                row.bits = front;
            }
        }
        Ok(())
    }

    /// Additional owned tape nodes and decision handles walked by collection.
    /// Charge this together with the arena and external roots before collecting.
    pub fn witness_collection_work(&self) -> usize {
        self.row_witnesses
            .as_ref()
            .map_or(0, |tape| tape.ops.len().saturating_add(tape.roots))
    }

    /// Release the tape after every retained deferred row has been expanded.
    pub fn finish_witness_tape(&mut self) {
        self.row_witnesses = None;
    }

    fn xor_witness(
        &mut self,
        left: Option<usize>,
        right: Option<usize>,
        factor: DecisionId,
    ) -> Result<Option<usize>, BooleanResourceError> {
        self.charge(1)?;
        let tape = self
            .row_witnesses
            .as_ref()
            .expect("deferred row has an owned tape");
        let Some(right) = right else { return Ok(left) };
        if factor == DECISION_TRUE {
            if left == Some(right) {
                return Ok(None);
            }
            if left.is_none() {
                return Ok(Some(right));
            }
        }
        let work = tape.insertion_work()?;
        self.charge(work)?;
        let tape = self
            .row_witnesses
            .as_mut()
            .expect("charging work retains the owned witness tape");
        Ok(Some(tape.push(RowWitnessOp::Xor {
            left,
            right,
            factor,
        })))
    }
}

impl BooleanRow {
    /// Query raw coefficients without expanding a deferred row. Results retain
    /// request order and duplicates, with zero for absent columns. Activation
    /// does not mask the values, including values outside the reachable domain.
    /// The row and its composed witness remain unchanged.
    ///
    /// Returned decisions belong to `diagram`; retain them as roots if collecting
    /// before consuming them. No query cache or per-row expanded image is kept.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if querying exceeds cumulative work limits.
    ///
    /// # Panics
    ///
    /// Panics if a deferred row has no active witness tape.
    pub fn probe_coefficients(
        &self,
        columns: &[usize],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<DecisionId>, BooleanResourceError> {
        diagram.charge(
            columns
                .len()
                .saturating_mul(self.bits.len().max(1).ilog2() as usize + 1)
                .saturating_add(1),
        )?;
        let mut result = columns
            .iter()
            .map(|&column| self.get(column))
            .collect::<Vec<_>>();
        let RowWitness::Deferred(Some(root)) = self.witness else {
            return Ok(result);
        };
        if columns.is_empty() {
            return Ok(result);
        }
        diagram.charge(
            columns
                .len()
                .saturating_mul(columns.len().ilog2() as usize + 2),
        )?;
        let mut requested = columns.to_vec();
        requested.sort_unstable();
        requested.dedup();
        diagram.charge(requested.len())?;
        let tape = diagram
            .row_witnesses
            .as_ref()
            .expect("deferred row has an owned witness tape");
        requested.retain(|column| !tape.explicit_columns.contains(column));
        if requested.is_empty() {
            return Ok(result);
        }
        let tape = diagram
            .row_witnesses
            .take()
            .expect("deferred row has an owned witness tape");
        let passive = tape.reconstruct_columns(root, &requested, diagram);
        diagram.row_witnesses = Some(tape);
        let passive = passive?;
        diagram.charge(
            columns
                .len()
                .saturating_mul(passive.len().max(1).ilog2() as usize + 1),
        )?;
        for (&column, value) in columns.iter().zip(&mut result) {
            if let Ok(index) = passive.binary_search_by_key(&column, |term| term.0) {
                debug_assert_eq!(
                    *value, DECISION_FALSE,
                    "explicit and passive columns are disjoint"
                );
                *value = passive[index].1;
            }
        }
        Ok(result)
    }

    /// Move coefficients outside the tape's common explicit partition into an
    /// immutable witness leaf. This preserves the raw off-activation values.
    /// A fresh core allocation releases the former full-support capacity.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if recording the witness exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if the row is already deferred or no witness tape is active.
    pub fn defer_coefficients(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        assert_eq!(
            self.witness,
            RowWitness::Explicit,
            "row is already deferred"
        );
        diagram.charge(self.bits.len().saturating_mul(2).saturating_add(1))?;
        let tape = diagram
            .row_witnesses
            .as_ref()
            .expect("row witness tape started");
        let passive_len = self
            .bits
            .iter()
            .filter(|(column, _)| !tape.explicit_columns.contains(column))
            .count();
        if passive_len != 0 {
            let work = tape.insertion_work()?;
            diagram.charge(work)?;
        }
        let mut core = Vec::with_capacity(self.bits.len() - passive_len);
        let mut passive = Vec::with_capacity(passive_len);
        let tape = diagram
            .row_witnesses
            .as_mut()
            .expect("charging work retains the started witness tape");
        for &term in &self.bits {
            if tape.explicit_columns.contains(&term.0) {
                core.push(term);
            } else {
                passive.push(term);
            }
        }
        let witness = if passive.is_empty() {
            None
        } else {
            Some(tape.push(RowWitnessOp::Leaf(passive)))
        };
        self.bits = core;
        self.witness = RowWitness::Deferred(witness);
        Ok(())
    }

    /// Restore every deferred coefficient before full-support iteration,
    /// mapping, or physical binding. No intermediate witness rows are expanded;
    /// use [`Self::probe_coefficients`] to query a subset without changing the row.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if reconstruction exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if a deferred row has no active witness tape.
    pub fn expand_coefficients(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        let RowWitness::Deferred(root) = self.witness else {
            return Ok(());
        };
        let Some(root) = root else {
            self.witness = RowWitness::Explicit;
            return Ok(());
        };
        let mut tape = diagram
            .row_witnesses
            .take()
            .expect("row witness tape started");
        let expanded = tape.reconstruct_all(root, diagram);
        diagram.row_witnesses = Some(tape);
        let passive = expanded?;
        diagram.charge(self.bits.len().saturating_add(passive.len()))?;
        let mut merged = Vec::with_capacity(self.bits.len() + passive.len());
        let mut core = std::mem::take(&mut self.bits).into_iter().peekable();
        for term in passive {
            while core.peek().is_some_and(|&(column, _)| column < term.0) {
                merged.push(core.next().expect("peek found a preceding core term"));
            }
            debug_assert!(core.peek().is_none_or(|&(column, _)| column != term.0));
            merged.push(term);
        }
        merged.extend(core);
        self.bits = merged;
        self.witness = RowWitness::Explicit;
        Ok(())
    }
}

impl Default for BooleanRow {
    fn default() -> Self {
        Self::new(DECISION_TRUE)
    }
}

impl BooleanRow {
    /// Creates an empty row available under `active`.
    pub fn new(active: DecisionId) -> Self {
        Self {
            active,
            bits: Vec::new(),
            witness: RowWitness::Explicit,
        }
    }

    /// Build a row; repeated columns keep their last value, like `set`.
    pub fn from_terms(
        active: DecisionId,
        terms: impl IntoIterator<Item = (usize, DecisionId)>,
    ) -> Self {
        let mut row = Self::new(active);
        for (column, value) in terms {
            row.set(column, value);
        }
        row
    }

    /// Explicit nonzero coefficients in strictly increasing column order.
    /// Deferred witness coefficients are restored by `expand_coefficients`.
    pub fn terms(&self) -> &[(usize, DecisionId)] {
        &self.bits
    }

    /// Consumes the row and returns its explicit coefficients.
    ///
    /// # Panics
    ///
    /// Panics if deferred coefficients have not been expanded.
    pub fn into_terms(self) -> Vec<(usize, DecisionId)> {
        assert_eq!(
            self.witness,
            RowWitness::Explicit,
            "expand deferred coefficients before consuming terms"
        );
        self.bits
    }

    /// Replace coefficients and remove terms that become zero.
    ///
    /// # Errors
    ///
    /// Returns the first [`BooleanResourceError`] produced by `map`.
    ///
    /// # Panics
    ///
    /// Panics if deferred coefficients have not been expanded.
    pub fn map_coefficients(
        &mut self,
        mut map: impl FnMut(DecisionId) -> Result<DecisionId, BooleanResourceError>,
    ) -> Result<(), BooleanResourceError> {
        assert_eq!(
            self.witness,
            RowWitness::Explicit,
            "expand deferred coefficients before mapping"
        );
        let mut result = Ok(());
        self.bits.retain_mut(|(_, value)| {
            if result.is_ok() {
                match map(*value) {
                    Ok(mapped) => *value = mapped,
                    Err(error) => result = Err(error),
                }
            }
            *value != DECISION_FALSE
        });
        result
    }

    /// Discard auxiliary columns at and above `end`.
    ///
    /// # Panics
    ///
    /// Panics if deferred coefficients have not been expanded.
    pub fn truncate_columns(&mut self, end: usize) {
        assert_eq!(
            self.witness,
            RowWitness::Explicit,
            "expand deferred coefficients before truncation"
        );
        self.bits
            .truncate(self.bits.partition_point(|&(column, _)| column < end));
    }

    /// Every decision owned by this row, for arena lifetime management.
    pub fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        std::iter::once(&mut self.active).chain(self.bits.iter_mut().map(|(_, value)| value))
    }

    /// Query an explicit coefficient; deferred columns must not be constraints.
    pub fn get(&self, column: usize) -> DecisionId {
        self.bits
            .binary_search_by_key(&column, |&(column, _)| column)
            .map_or(DECISION_FALSE, |index| self.bits[index].1)
    }

    /// Sets one explicit coefficient, removing it when `value` is false.
    ///
    /// # Panics
    ///
    /// Panics if the row has deferred coefficients.
    pub fn set(&mut self, column: usize, value: DecisionId) {
        assert_eq!(
            self.witness,
            RowWitness::Explicit,
            "deferred rows use row operations only"
        );
        if self.bits.last().is_none_or(|&(last, _)| last < column) {
            if value != DECISION_FALSE {
                self.bits.push((column, value));
            }
            return;
        }
        match self
            .bits
            .binary_search_by_key(&column, |&(column, _)| column)
        {
            Ok(index) if value == DECISION_FALSE => {
                self.bits.remove(index);
            }
            Ok(index) => self.bits[index].1 = value,
            Err(index) if value != DECISION_FALSE => self.bits.insert(index, (column, value)),
            Err(_) => {}
        }
    }

    /// XORs `other` into this row under the Boolean `factor`.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if row algebra exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if explicit coefficients are mixed with deferred witness rows.
    pub fn xor_scaled(
        &mut self,
        other: &Self,
        factor: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        if factor == DECISION_FALSE {
            return Ok(());
        }
        let witness = match (self.witness, other.witness) {
            (RowWitness::Explicit, RowWitness::Explicit) => RowWitness::Explicit,
            (RowWitness::Explicit, RowWitness::Deferred(right)) => {
                assert!(
                    self.bits.is_empty(),
                    "cannot mix untracked coefficients with a deferred row"
                );
                RowWitness::Deferred(diagram.xor_witness(None, right, factor)?)
            }
            (RowWitness::Deferred(left), RowWitness::Explicit) => {
                assert!(
                    other.bits.is_empty(),
                    "cannot mix untracked coefficients with a deferred row"
                );
                RowWitness::Deferred(left)
            }
            (RowWitness::Deferred(left), RowWitness::Deferred(right)) => {
                RowWitness::Deferred(diagram.xor_witness(left, right, factor)?)
            }
        };
        if other.bits.is_empty() {
            self.witness = witness;
            return Ok(());
        }
        if self.bits.is_empty() && factor == DECISION_TRUE {
            diagram.charge(other.bits.len())?;
            self.bits.clone_from(&other.bits);
            self.witness = witness;
            return Ok(());
        }
        // Preserve untouched prefixes/suffixes and their allocation. Copying
        // untouched growing witnesses at each local seam adds quadratic work.
        let first = other
            .bits
            .first()
            .expect("empty source rows returned above")
            .0;
        let last = other
            .bits
            .last()
            .expect("empty source rows returned above")
            .0;
        let start = self.bits.partition_point(|&(column, _)| column < first);
        let end = self.bits.partition_point(|&(column, _)| column <= last);
        diagram.charge(other.bits.len().saturating_add(end - start))?;
        let mut merged = Vec::with_capacity((end - start).max(other.bits.len()));
        let mut left = self.bits[start..end].iter().peekable();
        for &(column, value) in &other.bits {
            while left.peek().is_some_and(|&&(current, _)| current < column) {
                merged.push(*left.next().expect("peek found a preceding left term"));
            }
            let current = if left.peek().is_some_and(|&&(current, _)| current == column) {
                left.next().expect("peek found the matching left term").1
            } else {
                DECISION_FALSE
            };
            let term = diagram.apply(BooleanOp::And, factor, value)?;
            let value = diagram.apply(BooleanOp::Xor, current, term)?;
            if value != DECISION_FALSE {
                merged.push((column, value));
            }
        }
        merged.extend(left);
        if merged.len() != end - start {
            diagram.charge(self.bits.len() - end)?;
            if self.bits.len() - (end - start) + merged.len() > self.bits.capacity() {
                diagram.charge(self.bits.len())?;
            }
        }
        drop(self.bits.splice(start..end, merged));
        self.witness = witness;
        Ok(())
    }
}

/// A sparse conditional row space with stable source order and column incidence.
///
/// Row slots stay fixed while pivots remove rows. Default methods visit only
/// possible contributors in source order; explicit sparse variants prefer
/// shorter rows when individual representatives may change.
#[derive(Debug, Default)]
pub struct BooleanRowSpace {
    rows: Vec<Option<BooleanRow>>,
    columns: FxHashMap<usize, BTreeSet<usize>>,
    live: usize,
}

impl BooleanRowSpace {
    /// Builds an indexed row space in source order.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if indexing exceeds configured limits.
    pub fn new(
        rows: Vec<BooleanRow>,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, BooleanResourceError> {
        let mut space = Self::default();
        for row in rows {
            space.push(row, diagram)?;
        }
        Ok(space)
    }

    /// Returns the number of live rows.
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns whether the space contains no live rows.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Roots for diagram collection. Only renumber the same truth functions;
    /// changing coefficients or activation here would invalidate the index.
    pub fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        self.rows
            .iter_mut()
            .filter_map(Option::as_mut)
            .flat_map(BooleanRow::decisions_mut)
    }

    /// Iterates live rows in source order.
    pub fn rows(&self) -> impl Iterator<Item = &BooleanRow> {
        self.rows.iter().filter_map(Option::as_ref)
    }

    /// Live rows with their stable insertion IDs, in source order.
    pub fn indexed_rows(&self) -> impl Iterator<Item = (usize, &BooleanRow)> {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| row.as_ref().map(|row| (index, row)))
    }

    /// Slots traversed by `indexed_rows`, including removed row IDs.
    pub fn indexed_slot_count(&self) -> usize {
        self.rows.len()
    }

    /// A row by its stable insertion ID, or `None` if it has been removed.
    pub fn row(&self, index: usize) -> Option<&BooleanRow> {
        self.rows.get(index).and_then(Option::as_ref)
    }

    /// Possible support columns, in unspecified order.
    pub fn columns(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.columns.keys().copied()
    }

    /// Rows whose active support may contain `column`, in source order.
    pub fn rows_with_column(&self, column: usize) -> impl Iterator<Item = &BooleanRow> {
        self.row_ids_with_column(column)
            .filter_map(|index| self.row(index))
    }

    /// Stable IDs whose active support may contain `column`, in source order.
    pub fn row_ids_with_column(&self, column: usize) -> impl Iterator<Item = usize> + '_ {
        self.columns
            .get(&column)
            .into_iter()
            .flat_map(|rows| rows.iter())
            .copied()
    }

    /// Append a row after all existing rows in the pivot priority order.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if indexing exceeds configured limits.
    pub fn push(
        &mut self,
        row: BooleanRow,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        self.insert_indexed(row, diagram).map(|_| ())
    }

    /// Append a row and return its stable ID. Inactive rows are discarded.
    /// New insertions never reuse removed IDs; `replace` can restore a known ID.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if indexing exceeds configured limits.
    pub fn insert_indexed(
        &mut self,
        row: BooleanRow,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Option<usize>, BooleanResourceError> {
        diagram.charge(1)?;
        if row.active == DECISION_FALSE {
            return Ok(None);
        }
        diagram.charge(row.bits.len())?;
        if self.rows.len() == self.rows.capacity() {
            diagram.charge(self.rows.len())?;
        }
        let index = self.rows.len();
        for &(column, _) in &row.bits {
            self.columns.entry(column).or_default().insert(index);
        }
        self.rows.push(Some(row));
        self.live += 1;
        Ok(Some(index))
    }

    /// Remove a row and its postings, retaining its slot for explicit replacement.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if accounting exceeds configured limits.
    pub fn take(
        &mut self,
        index: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Option<BooleanRow>, BooleanResourceError> {
        diagram.charge(1)?;
        let Some(row) = self.row(index) else {
            return Ok(None);
        };
        diagram.charge(row.bits.len())?;
        Ok(self.take_row(index))
    }

    /// Replace an existing slot without changing its pivot priority or ID.
    /// An inactive replacement removes the row. Returns the displaced row.
    ///
    /// # Panics
    /// Panics if `index` was never returned by `insert_indexed` in this space.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if accounting exceeds configured limits.
    pub fn replace(
        &mut self,
        index: usize,
        row: BooleanRow,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Option<BooleanRow>, BooleanResourceError> {
        assert!(index < self.rows.len(), "row ID belongs to this space");
        let old_terms = self.row(index).map_or(0, |row| row.bits.len());
        let new_terms = if row.active == DECISION_FALSE {
            0
        } else {
            row.bits.len()
        };
        diagram.charge(1usize.saturating_add(old_terms).saturating_add(new_terms))?;
        let previous = self.take_row(index);
        if row.active != DECISION_FALSE {
            for &(column, _) in &row.bits {
                self.columns.entry(column).or_default().insert(index);
            }
            self.rows[index] = Some(row);
            self.live += 1;
        }
        Ok(previous)
    }

    /// Erase a column from every indexed row without changing activation.
    /// After unconditional elimination, off-activation coefficients may remain
    /// unindexed; this guarantees removal from active support. Spaces maintained
    /// by insertion, replacement, and reduction retain full raw postings.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if accounting exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if the internal row-to-column index is inconsistent.
    pub fn erase_column(
        &mut self,
        column: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        diagram.charge(1)?;
        let Some(indices) = self.columns.get(&column) else {
            return Ok(());
        };
        diagram.charge(indices.len())?;
        let work = indices.iter().try_fold(0usize, |work, &index| {
            let row = self.rows[index].as_ref().expect("indexed row exists");
            let search = row.bits.len().ilog2() as usize + 1;
            diagram.charge(search)?;
            let suffix =
                row.bits.len() - row.bits.partition_point(|&(current, _)| current < column);
            Ok::<_, BooleanResourceError>(work.saturating_add(search).saturating_add(suffix))
        })?;
        // Removing a sorted coefficient can shift its suffix. Charge all rows
        // before changing any of them or taking ownership of the postings.
        diagram.charge(work)?;
        for index in self.columns.remove(&column).expect("indexed column exists") {
            self.rows[index]
                .as_mut()
                .expect("indexed row exists")
                .set(column, DECISION_FALSE);
        }
        Ok(())
    }

    /// Consumes the space and returns its live rows in source order.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if accounting exceeds configured limits.
    pub fn into_rows(
        self,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<BooleanRow>, BooleanResourceError> {
        diagram.charge(self.rows.len())?;
        Ok(self.rows.into_iter().flatten().collect())
    }

    fn candidates(
        &self,
        column: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Vec<usize>, BooleanResourceError> {
        diagram.charge(1)?;
        let Some(indices) = self.columns.get(&column) else {
            return Ok(Vec::new());
        };
        diagram.charge(indices.len())?;
        Ok(indices.iter().copied().collect())
    }

    fn remove_column_row(&mut self, column: usize, index: usize) {
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.columns.entry(column) {
            entry.get_mut().remove(&index);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
    }

    fn take_row(&mut self, index: usize) -> Option<BooleanRow> {
        let row = self.rows[index].take()?;
        self.live -= 1;
        for &(column, _) in &row.bits {
            self.remove_column_row(column, index);
        }
        Some(row)
    }

    fn xor_row(
        &mut self,
        index: usize,
        pivot: &BooleanRow,
        factor: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        if factor == DECISION_FALSE {
            return Ok(());
        }
        // Only pivot columns can change membership. Charge index maintenance
        // before mutating the row, so exhaustion cannot leave stale postings.
        diagram.charge(pivot.bits.len())?;
        self.rows[index]
            .as_mut()
            .expect("indexed row exists")
            .xor_scaled(pivot, factor, diagram)?;
        for &(column, _) in &pivot.bits {
            if self.rows[index]
                .as_ref()
                .expect("indexed row exists")
                .get(column)
                == DECISION_FALSE
            {
                self.remove_column_row(column, index);
            } else {
                self.columns.entry(column).or_default().insert(index);
            }
        }
        Ok(())
    }

    /// Eliminate one column pointwise, choosing the first available row on each
    /// guard. The returned pivot carries every auxiliary witness coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
    pub fn eliminate_column(
        &mut self,
        column: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        self.eliminate_column_when(column, DECISION_TRUE, diagram)
    }

    /// Eliminate one column using shortest-support pivots, with source order
    /// breaking ties. Preserves the conditional row space, not its representatives.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
    pub fn eliminate_column_sparse(
        &mut self,
        column: usize,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        self.eliminate_column_when_sparse(column, DECISION_TRUE, diagram)
    }

    /// Eliminate `gate & coefficient`, retaining the column outside the gate.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
    pub fn eliminate_column_when(
        &mut self,
        column: usize,
        gate: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        self.eliminate_column_ordered(column, gate, false, diagram)
    }

    /// Guarded column elimination with shortest-support pivots. Individual
    /// representatives may change; support outside the gate remains available.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
    pub fn eliminate_column_when_sparse(
        &mut self,
        column: usize,
        gate: DecisionId,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        self.eliminate_column_ordered(column, gate, true, diagram)
    }

    fn eliminate_column_ordered(
        &mut self,
        column: usize,
        gate: DecisionId,
        sparse: bool,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        if gate == DECISION_FALSE {
            return Ok(BooleanRow::new(DECISION_FALSE));
        }
        let mut candidates = self.candidates(column, diagram)?;
        if sparse {
            self.sort_sparse_candidates(&mut candidates, diagram)?;
        }
        let pivot = self.eliminate_candidates(candidates, diagram, |row, diagram| {
            diagram.apply(BooleanOp::And, gate, row.get(column))
        })?;
        // Any surviving coefficient is outside its row's activation. Later
        // pivots mask those coefficients out, so this column stays eliminated.
        if gate == DECISION_TRUE {
            self.columns.remove(&column);
        }
        Ok(pivot)
    }

    /// Eliminate an ordered XOR of weighted column coefficients. Repeated
    /// columns retain every occurrence; columns themselves need not become zero.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
    pub fn eliminate_linear(
        &mut self,
        terms: &[(usize, DecisionId)],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<BooleanRow, BooleanResourceError> {
        diagram.charge(terms.len())?;
        let mut candidates = BTreeSet::new();
        for &(column, weight) in terms {
            if weight != DECISION_FALSE
                && let Some(indices) = self.columns.get(&column)
            {
                diagram.charge(indices.len())?;
                candidates.extend(indices.iter().copied());
            }
        }
        diagram.charge(candidates.len())?;
        let candidates = candidates.into_iter().collect::<Vec<_>>();
        self.eliminate_candidates(candidates, diagram, |row, diagram| {
            diagram.charge(terms.len())?;
            terms
                .iter()
                .try_fold(DECISION_FALSE, |sum, &(column, weight)| {
                    let term = diagram.apply(BooleanOp::And, weight, row.get(column))?;
                    diagram.apply(BooleanOp::Xor, sum, term)
                })
        })
    }

    fn sort_sparse_candidates(
        &self,
        candidates: &mut [usize],
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        if candidates.len() < 2 {
            return Ok(());
        }
        // Account for comparison-sort work before sorting or changing any row.
        diagram.charge(
            candidates
                .len()
                .saturating_mul(candidates.len().ilog2() as usize + 1),
        )?;
        candidates.sort_unstable_by_key(|&index| {
            (
                self.rows[index]
                    .as_ref()
                    .expect("indexed row exists")
                    .bits
                    .len(),
                index,
            )
        });
        Ok(())
    }

    fn eliminate_candidates(
        &mut self,
        candidates: Vec<usize>,
        diagram: &mut BooleanDecisionDiagram,
        mut coefficient: impl FnMut(
            &BooleanRow,
            &mut BooleanDecisionDiagram,
        ) -> Result<DecisionId, BooleanResourceError>,
    ) -> Result<BooleanRow, BooleanResourceError> {
        let mut pivot = BooleanRow::new(DECISION_FALSE);
        for &index in &candidates {
            diagram.charge(1)?;
            let row = self.rows[index].as_mut().expect("indexed row exists");
            let coefficient = coefficient(row, diagram)?;
            select_into_pivot(&mut pivot, row, coefficient, diagram)?;
            if row.active == DECISION_FALSE {
                self.take(index, diagram)?;
            }
        }
        for index in candidates {
            diagram.charge(1)?;
            let Some(row) = &self.rows[index] else {
                continue;
            };
            let coefficient = coefficient(row, diagram)?;
            let factor = diagram.apply(BooleanOp::And, row.active, coefficient)?;
            self.xor_row(index, &pivot, factor, diagram)?;
        }
        Ok(pivot)
    }

    /// Reduce only rows that may carry this pivot column. Unlike elimination,
    /// this preserves their activation and retains their other pivot choices.
    ///
    /// # Errors
    ///
    /// Returns [`BooleanResourceError`] if reduction exceeds configured limits.
    ///
    /// # Panics
    ///
    /// Panics if the internal row-to-column index is inconsistent.
    pub fn reduce_column(
        &mut self,
        column: usize,
        pivot: &BooleanRow,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        if pivot.active == DECISION_FALSE {
            return Ok(());
        }
        for index in self.candidates(column, diagram)? {
            diagram.charge(1)?;
            let row = self.rows[index].as_ref().expect("indexed row exists");
            let factor = diagram.apply(BooleanOp::And, row.active, row.get(column))?;
            self.xor_row(index, pivot, factor, diagram)?;
        }
        Ok(())
    }

    /// Reduce an ordered XOR of weighted columns against a supplied pivot.
    /// The pivot must have unit projection on its activation.
    ///
    /// # Errors
    /// Returns [`BooleanResourceError`] if reduction exceeds configured limits.
    ///
    /// # Panics
    /// Panics if the internal row-to-column index is inconsistent.
    pub fn reduce_linear(
        &mut self,
        terms: &[(usize, DecisionId)],
        pivot: &BooleanRow,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<(), BooleanResourceError> {
        if pivot.active == DECISION_FALSE {
            return Ok(());
        }
        diagram.charge(terms.len())?;
        let mut candidates = BTreeSet::new();
        for &(column, weight) in terms {
            if weight != DECISION_FALSE {
                candidates.extend(self.candidates(column, diagram)?);
            }
        }
        for index in candidates {
            let row = self.rows[index].as_ref().expect("indexed row exists");
            diagram.charge(terms.len())?;
            let coefficient = terms
                .iter()
                .try_fold(DECISION_FALSE, |sum, &(column, weight)| {
                    let term = diagram.apply(BooleanOp::And, weight, row.get(column))?;
                    diagram.apply(BooleanOp::Xor, sum, term)
                })?;
            let factor = diagram.apply(BooleanOp::And, row.active, coefficient)?;
            self.xor_row(index, pivot, factor, diagram)?;
        }
        Ok(())
    }
}

/// Splice `row` into `pivot` on the guards where `pivot` is not yet available,
/// and withdraw those guards from `row`. The pivot therefore takes the first
/// available row pointwise, and the guards the two carry stay disjoint.
///
/// Shared by the dense [`eliminate_boolean_rows`] and the indexed
/// [`BooleanRowSpace::eliminate_candidates`] so their pivot choice, sign
/// handling, and charged work cannot drift apart.
fn select_into_pivot(
    pivot: &mut BooleanRow,
    row: &mut BooleanRow,
    coefficient: DecisionId,
    diagram: &mut BooleanDecisionDiagram,
) -> Result<(), BooleanResourceError> {
    let candidate = diagram.apply(BooleanOp::And, row.active, coefficient)?;
    let absent = diagram.negate(pivot.active)?;
    let selected = diagram.apply(BooleanOp::And, candidate, absent)?;
    pivot.xor_scaled(row, selected, diagram)?;
    pivot.active = diagram.apply(BooleanOp::Or, pivot.active, candidate)?;
    let remaining = diagram.negate(selected)?;
    row.active = diagram.apply(BooleanOp::And, row.active, remaining)?;
    Ok(())
}

/// Eliminate one linear constraint pointwise, returning its conditional pivot.
///
/// The first available pivot is selected by disjoint guards, so no assignment
/// enumeration or product of alternative row spaces is needed.
///
/// # Errors
///
/// Returns [`BooleanResourceError`] if elimination exceeds configured limits.
pub fn eliminate_boolean_rows(
    rows: &mut Vec<BooleanRow>,
    diagram: &mut BooleanDecisionDiagram,
    mut constraint: impl FnMut(
        &BooleanRow,
        &mut BooleanDecisionDiagram,
    ) -> Result<DecisionId, BooleanResourceError>,
) -> Result<BooleanRow, BooleanResourceError> {
    diagram.charge(rows.len().saturating_mul(2))?;
    let mut pivot = BooleanRow::new(DECISION_FALSE);
    let mut factors = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        let coefficient = constraint(row, diagram)?;
        factors.push(coefficient);
        select_into_pivot(&mut pivot, row, coefficient, diagram)?;
    }
    for (row, coefficient) in rows.iter_mut().zip(factors) {
        let factor = diagram.apply(BooleanOp::And, row.active, coefficient)?;
        row.xor_scaled(&pivot, factor, diagram)?;
    }
    rows.retain(|row| row.active != DECISION_FALSE);
    Ok(pivot)
}

/// Canonical reduced rows over the requested columns, with conditional rank.
///
/// # Errors
///
/// Returns [`BooleanResourceError`] if reduction exceeds configured limits.
pub fn reduce_boolean_rows(
    rows: Vec<BooleanRow>,
    columns: impl IntoIterator<Item = usize>,
    diagram: &mut BooleanDecisionDiagram,
) -> Result<Vec<BooleanRow>, BooleanResourceError> {
    let mut remaining = BooleanRowSpace::new(rows, diagram)?;
    let mut reduced = BooleanRowSpace::default();
    for column in columns {
        if remaining.is_empty() || remaining.columns.is_empty() {
            break;
        }
        let pivot = remaining.eliminate_column(column, diagram)?;
        if pivot.active != DECISION_FALSE {
            reduced.reduce_column(column, &pivot, diagram)?;
            reduced.push(pivot, diagram)?;
        }
    }
    reduced.into_rows(diagram)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_reduction_matches_an_explicit_guarded_query_coordinate() {
        let mut diagram = BooleanDecisionDiagram::default();
        let selector = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let other = diagram.negate(selector).unwrap();
        let terms = [(0, selector), (1, other), (2, selector), (2, selector)];
        let pivot = BooleanRow::from_terms(
            DECISION_TRUE,
            [(0, DECISION_TRUE), (1, DECISION_TRUE), (5, selector)],
        );
        let rows = vec![
            BooleanRow::from_terms(
                DECISION_TRUE,
                [
                    (0, other),
                    (1, selector),
                    (2, DECISION_TRUE),
                    (6, DECISION_TRUE),
                ],
            ),
            BooleanRow::from_terms(
                selector,
                [(0, DECISION_TRUE), (2, DECISION_TRUE), (7, DECISION_TRUE)],
            ),
        ];
        let mut expected = rows.clone();
        // The first row's query cancels on both branches; the second is
        // selected exactly when its activation is true. Column 2 cancels twice.
        for (row, query) in expected.iter_mut().zip([DECISION_FALSE, selector]) {
            row.set(10, query);
        }
        let mut tagged = BooleanRowSpace::new(expected, &mut diagram).unwrap();
        let mut tagged_pivot = pivot.clone();
        tagged_pivot.set(10, DECISION_TRUE);
        tagged
            .reduce_column(10, &tagged_pivot, &mut diagram)
            .unwrap();
        let mut compact = BooleanRowSpace::new(rows, &mut diagram).unwrap();
        compact.reduce_linear(&terms, &pivot, &mut diagram).unwrap();
        let mut expected = tagged.into_rows(&mut diagram).unwrap();
        for row in &mut expected {
            row.set(10, DECISION_FALSE);
        }
        let compact = compact.into_rows(&mut diagram).unwrap();
        assert_eq!(compact.len(), expected.len());
        for (actual, expected) in compact.iter().zip(expected) {
            assert_eq!(actual.active, expected.active);
            assert_eq!(actual.terms(), expected.terms());
        }
    }

    #[test]
    fn deferred_witness_failures_leave_rows_unchanged() {
        for exhaust_core_work in [false, true] {
            let mut diagram = BooleanDecisionDiagram::default();
            diagram
                .start_witness_tape([0], if exhaust_core_work { 16 } else { 2 })
                .unwrap();
            let mut left =
                BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (10, DECISION_TRUE)]);
            let mut right =
                BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (11, DECISION_TRUE)]);
            left.bits.reserve(1_024);
            left.defer_coefficients(&mut diagram).unwrap();
            right.defer_coefficients(&mut diagram).unwrap();
            assert!(
                left.bits.capacity() < 1_024,
                "split released passive capacity"
            );
            let before = left.clone();
            if exhaust_core_work {
                // The tape operation fits; the following explicit merge does not.
                diagram.limits.max_steps = diagram.steps() + 2;
            }
            let error = left
                .xor_scaled(&right, DECISION_TRUE, &mut diagram)
                .unwrap_err();
            assert_eq!(
                error.resource,
                if exhaust_core_work {
                    "Boolean work steps"
                } else {
                    "row witness operations"
                },
            );
            assert_eq!(left.active, before.active);
            assert_eq!(left.bits, before.bits);
            assert_eq!(left.witness, before.witness);
        }
        let mut diagram = BooleanDecisionDiagram::default();
        diagram.start_witness_tape([0], 8).unwrap();
        let mut core_only = BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE)]);
        core_only.defer_coefficients(&mut diagram).unwrap();
        assert_eq!(core_only.witness, RowWitness::Deferred(None));
        let untracked = BooleanRow::from_terms(DECISION_TRUE, [(10, DECISION_TRUE)]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                core_only
                    .xor_scaled(&untracked, DECISION_TRUE, &mut diagram)
                    .unwrap();
            }))
            .is_err()
        );
        std::panic::catch_unwind(|| core_only.into_terms()).unwrap_err();
    }

    #[test]
    fn deferred_witnesses_preserve_raw_payload_through_pivots_and_collection() {
        let mut original = BooleanDecisionDiagram::default();
        original
            .make_node(99, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let a = original
            .make_node(0, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let b = original
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let absent = original.negate(a).unwrap();
        let rows = vec![
            BooleanRow::from_terms(DECISION_TRUE, [(0, a), (1, b), (100, DECISION_TRUE)]),
            BooleanRow::from_terms(a, [(0, DECISION_TRUE), (1, a), (101, b)]),
            BooleanRow::from_terms(absent, [(0, DECISION_TRUE), (1, b), (102, a)]),
        ];
        let mut expanded_diagram = original.clone();
        let mut expanded = BooleanRowSpace::new(rows.clone(), &mut expanded_diagram).unwrap();
        let mut tape_diagram = original;
        tape_diagram.start_witness_tape([0, 1], 100).unwrap();
        let mut core = rows;
        for row in &mut core {
            row.defer_coefficients(&mut tape_diagram).unwrap();
        }
        let mut core = BooleanRowSpace::new(core, &mut tape_diagram).unwrap();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        for column in [0, 1] {
            expected.push(
                expanded
                    .eliminate_column(column, &mut expanded_diagram)
                    .unwrap(),
            );
            actual.push(core.eliminate_column(column, &mut tape_diagram).unwrap());
        }
        expected.extend(expanded.into_rows(&mut expanded_diagram).unwrap());
        actual.extend(core.into_rows(&mut tape_diagram).unwrap());
        assert!(tape_diagram.witness_collection_work() > 0);
        assert!(
            tape_diagram.collect_garbage(actual.iter_mut().flat_map(BooleanRow::decisions_mut)) > 0
        );
        expanded_diagram.collect_garbage(expected.iter_mut().flat_map(BooleanRow::decisions_mut));
        let columns = [102, 0, 101, 1, 100, 101, 99, usize::MAX];
        let mut probes = actual
            .iter()
            .map(|row| row.probe_coefficients(&columns, &mut tape_diagram).unwrap())
            .collect::<Vec<_>>();
        tape_diagram.collect_garbage(
            actual
                .iter_mut()
                .flat_map(BooleanRow::decisions_mut)
                .chain(probes.iter_mut().flatten()),
        );
        for (row, probe) in actual.iter_mut().zip(probes) {
            row.expand_coefficients(&mut tape_diagram).unwrap();
            assert_eq!(probe, columns.map(|column| row.get(column)));
        }
        tape_diagram.finish_witness_tape();
        let mut remap = vec![DECISION_FALSE, DECISION_TRUE];
        for node in tape_diagram.nodes() {
            remap.push(
                expanded_diagram
                    .make_node(node.variable, remap[node.low.0], remap[node.high.0])
                    .unwrap(),
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(remap[actual.active.0], expected.active);
            assert_eq!(
                actual
                    .terms()
                    .iter()
                    .map(|&(col, value)| (col, remap[value.0]))
                    .collect::<Vec<_>>(),
                expected.terms()
            );
        }
        let mut limited = BooleanDecisionDiagram::default();
        limited.start_witness_tape([], 0).unwrap();
        assert!(matches!(
            BooleanRow::from_terms(DECISION_TRUE, [(10, DECISION_TRUE)])
                .defer_coefficients(&mut limited),
            Err(BooleanResourceError {
                resource: "row witness operations",
                observed: 1,
                limit: 0
            })
        ));
    }

    #[test]
    fn composed_queries_cancel_shared_paths_and_keep_raw_inactive_values() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        diagram.start_witness_tape([0], 20).unwrap();
        let mut leaf = BooleanRow::from_terms(a, [(0, b), (20, DECISION_TRUE), (22, a)]);
        leaf.defer_coefficients(&mut diagram).unwrap();
        let mut combined = BooleanRow::new(DECISION_FALSE);
        combined.xor_scaled(&leaf, a, &mut diagram).unwrap();
        combined.xor_scaled(&leaf, b, &mut diagram).unwrap();
        let columns = [22, 20, 0, 21, 20, usize::MAX];
        let probe = combined.probe_coefficients(&columns, &mut diagram).unwrap();
        let mut eager = combined.clone();
        eager.expand_coefficients(&mut diagram).unwrap();
        assert_eq!(probe, columns.map(|column| eager.get(column)));
        assert_eq!(combined.active, DECISION_FALSE);
        assert_ne!(
            probe[1], DECISION_FALSE,
            "raw query must not apply activation"
        );
        assert_eq!(probe[3], DECISION_FALSE, "interval holes remain zero");
        assert!(matches!(combined.witness, RowWitness::Deferred(Some(_))));
        let factor = diagram.apply(BooleanOp::Xor, a, b).unwrap();
        combined.xor_scaled(&leaf, factor, &mut diagram).unwrap();
        assert_eq!(
            combined.probe_coefficients(&columns, &mut diagram).unwrap(),
            vec![DECISION_FALSE; columns.len()],
        );
    }

    #[test]
    fn dense_witness_replay_preserves_raw_rows_across_errors_and_collection() {
        for high in [1, usize::MAX] {
            let mut diagram = BooleanDecisionDiagram::default();
            diagram
                .make_node(99, DECISION_FALSE, DECISION_TRUE)
                .unwrap();
            let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
            let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
            let mut row = BooleanRow::from_terms(DECISION_FALSE, [(0, a), (high, b)]);
            let mut other = BooleanRow::from_terms(DECISION_TRUE, [(0, b), (high, DECISION_TRUE)]);
            let mut expected = row.clone();
            expected.xor_scaled(&other, a, &mut diagram).unwrap();
            diagram.start_witness_tape([], 100).unwrap();
            row.defer_coefficients(&mut diagram).unwrap();
            other.defer_coefficients(&mut diagram).unwrap();
            row.xor_scaled(&other, a, &mut diagram).unwrap();
            let mut warm = row.clone();
            warm.expand_coefficients(&mut diagram).unwrap();
            assert_eq!(warm.terms(), expected.terms());
            diagram.limits.max_steps = diagram.steps() + 3;
            let error = row.expand_coefficients(&mut diagram).unwrap_err();
            assert_eq!(error.resource, "Boolean work steps");
            drop((warm, other));
            diagram.collect_garbage(row.decisions_mut().chain(expected.decisions_mut()));
            diagram.limits.max_steps = usize::MAX;
            row.expand_coefficients(&mut diagram).unwrap();
            assert_eq!(row.active, DECISION_FALSE);
            assert_eq!(row.terms(), expected.terms());
        }
    }

    #[test]
    fn composed_queries_prune_disjoint_history_and_restore_tape_on_exhaustion() {
        let mut diagram = BooleanDecisionDiagram::default();
        diagram.start_witness_tape([], 4096).unwrap();
        let mut row = BooleanRow::default();
        let mut costs = Vec::new();
        for column in 0..1024 {
            let mut leaf = BooleanRow::from_terms(DECISION_TRUE, [(column, DECISION_TRUE)]);
            leaf.defer_coefficients(&mut diagram).unwrap();
            row.xor_scaled(&leaf, DECISION_TRUE, &mut diagram).unwrap();
            if [63, 127, 255, 511, 1023].contains(&column) {
                let start = diagram.steps();
                assert_eq!(
                    row.probe_coefficients(&[column], &mut diagram).unwrap(),
                    [DECISION_TRUE],
                );
                costs.push(diagram.steps() - start);
            }
        }
        assert!(costs.windows(2).all(|pair| pair[0] == pair[1]), "{costs:?}");
        assert!(row.terms().is_empty(), "querying never expands the row");
        let before = row.clone();
        let limit = diagram.steps() + 7;
        diagram.limits.max_steps = limit;
        let error = row.probe_coefficients(&[1023], &mut diagram).unwrap_err();
        assert_eq!(error.resource, "Boolean work steps");
        assert_eq!(error.limit, limit);
        assert_eq!(row.witness, before.witness);
        assert_eq!(row.terms(), before.terms());
        assert!(diagram.row_witnesses.is_some());
        let spent = diagram.steps();
        diagram.collect_garbage(row.decisions_mut());
        assert_eq!(diagram.steps(), spent);
        diagram.limits.max_steps = usize::MAX;
        assert_eq!(
            row.probe_coefficients(&[1023], &mut diagram).unwrap(),
            [DECISION_TRUE],
        );
    }

    #[test]
    fn batched_witness_fronts_share_overlapping_history() {
        for width in [64, 128, 256, 512, 1024] {
            let mut diagram = BooleanDecisionDiagram::default();
            diagram.start_witness_tape([], 2 * width).unwrap();
            let mut prefix = BooleanRow::default();
            let mut rows = Vec::new();
            for column in 0..width {
                let mut leaf = BooleanRow::from_terms(DECISION_TRUE, [(column, DECISION_TRUE)]);
                leaf.defer_coefficients(&mut diagram).unwrap();
                prefix
                    .xor_scaled(&leaf, DECISION_TRUE, &mut diagram)
                    .unwrap();
                rows.push(prefix.clone());
            }
            let before = diagram.steps();
            diagram.extend_witness_front(&[0], &mut rows).unwrap();
            let work = diagram.steps() - before;
            assert!(
                work < 32 * width,
                "{width} overlapping roots spent {work} steps"
            );
            assert!(rows.iter().all(|row| row.terms() == [(0, DECISION_TRUE)]));
            let last = rows.last_mut().unwrap();
            last.expand_coefficients(&mut diagram).unwrap();
            assert_eq!(last.terms().len(), width);
        }
    }

    #[test]
    fn batched_witness_fronts_do_not_cross_independent_rows() {
        let mut previous = None;
        for width in [64, 128, 256, 512, 1024] {
            let mut diagram = BooleanDecisionDiagram::default();
            diagram.start_witness_tape([], 3 * width).unwrap();
            let mut rows = Vec::new();
            for index in 0..width {
                let mut left = BooleanRow::from_terms(DECISION_TRUE, [(2 * index, DECISION_TRUE)]);
                let mut right =
                    BooleanRow::from_terms(DECISION_TRUE, [(2 * index + 1, DECISION_TRUE)]);
                left.defer_coefficients(&mut diagram).unwrap();
                right.defer_coefficients(&mut diagram).unwrap();
                left.xor_scaled(&right, DECISION_TRUE, &mut diagram)
                    .unwrap();
                rows.push(left);
            }
            let columns = (0..width).map(|index| 2 * index).collect::<Vec<_>>();
            let before = diagram.steps();
            diagram.extend_witness_front(&columns, &mut rows).unwrap();
            let work = diagram.steps() - before;
            if let Some(previous) = previous {
                assert!(
                    work < 3 * previous,
                    "{width}: projection became a dense cross product"
                );
            }
            previous = Some(work);
            for (index, row) in rows.iter().enumerate() {
                assert_eq!(row.terms(), [(2 * index, DECISION_TRUE)]);
                assert_eq!(row.bits.capacity(), 1);
            }
        }
    }

    #[test]
    fn promoted_witness_fronts_preserve_raw_rows_through_algebra_and_collection() {
        let mut diagram = BooleanDecisionDiagram::default();
        diagram
            .make_node(99, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut expected = vec![
            BooleanRow::from_terms(DECISION_FALSE, [(0, a), (20, b), (22, DECISION_TRUE)]),
            BooleanRow::from_terms(DECISION_TRUE, [(0, b), (21, a), (31, b)]),
            BooleanRow::from_terms(a, [(20, DECISION_TRUE), (21, b), (22, a)]),
        ];
        let mut actual = expected.clone();
        diagram.start_witness_tape([0], 100).unwrap();
        for row in &mut actual {
            row.defer_coefficients(&mut diagram).unwrap();
        }
        let handles = actual.iter().map(|row| row.witness).collect::<Vec<_>>();
        diagram
            .extend_witness_front(&[21, 20, 20, 0, 99], &mut actual)
            .unwrap();
        for (row, handle) in actual.iter().zip(handles) {
            assert_eq!(row.witness, handle);
            assert_eq!(
                row.bits.capacity(),
                row.bits.len(),
                "absent columns reserve no slots"
            );
        }
        assert_eq!(actual[0].active, DECISION_FALSE);
        assert_eq!(
            actual[0].get(20),
            b,
            "promotion does not mask inactive rows"
        );
        assert_eq!(
            actual[0].get(22),
            DECISION_FALSE,
            "unqueried support stays passive"
        );
        let columns = [31, 22, 20, 0, 21, 20, 99];
        for (row, eager) in actual.iter().zip(&expected) {
            assert_eq!(
                row.probe_coefficients(&columns, &mut diagram).unwrap(),
                columns.map(|column| eager.get(column)),
            );
        }
        for rows in [&mut actual, &mut expected] {
            let (left, right) = rows.split_at_mut(1);
            left[0].xor_scaled(&right[1], b, &mut diagram).unwrap();
            let source = right[1].clone();
            right[0].xor_scaled(&source, a, &mut diagram).unwrap();
        }
        diagram
            .extend_witness_front(&[31, 20], &mut actual)
            .unwrap();
        let fresh = BooleanRow::from_terms(DECISION_TRUE, [(20, a), (50, b)]);
        expected.push(fresh.clone());
        let mut fresh = fresh;
        fresh.defer_coefficients(&mut diagram).unwrap();
        assert_eq!(fresh.get(20), a, "new leaves use the extended front");
        actual.push(fresh);

        let pivot = |rows, diagram: &mut BooleanDecisionDiagram| {
            let mut space = BooleanRowSpace::new(rows, diagram).unwrap();
            let first = space
                .eliminate_linear(&[(20, b), (21, DECISION_TRUE)], diagram)
                .unwrap();
            let second = space.eliminate_column(31, diagram).unwrap();
            let mut rows = vec![first, second];
            rows.extend(space.into_rows(diagram).unwrap());
            rows
        };
        let mut actual = pivot(actual, &mut diagram);
        let mut expected = pivot(expected, &mut diagram);
        assert!(
            diagram.collect_garbage(
                actual
                    .iter_mut()
                    .chain(&mut expected)
                    .flat_map(BooleanRow::decisions_mut)
            ) > 0
        );
        for (row, eager) in actual.iter_mut().zip(&expected) {
            row.expand_coefficients(&mut diagram).unwrap();
            assert_eq!(row.active, eager.active);
            assert_eq!(row.terms(), eager.terms());
        }
        assert_eq!(actual.len(), expected.len());
    }

    #[test]
    fn witness_front_promotion_failures_retain_partition_rows_and_spent_work() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        diagram.start_witness_tape([0], 20).unwrap();
        let mut leaf = BooleanRow::from_terms(DECISION_TRUE, [(20, b), (30, DECISION_TRUE)]);
        leaf.defer_coefficients(&mut diagram).unwrap();
        let mut combined = BooleanRow::new(DECISION_FALSE);
        combined.xor_scaled(&leaf, a, &mut diagram).unwrap();
        let rows = vec![leaf, combined, BooleanRow::from_terms(a, [(20, b)])];
        let columns = [30, 20, 30];
        let before_steps = diagram.steps();
        let mut completed = diagram.clone();
        completed
            .extend_witness_front(&columns, &mut rows.clone())
            .unwrap();
        let work = completed.steps() - before_steps;

        // Every work-limit cut, including failure after staged row construction.
        for allowance in 0..=work {
            let mut limited = diagram.clone();
            let mut pending = rows.clone();
            if allowance == work {
                limited.limits.max_nodes = limited.nodes().len();
            } else {
                limited.limits.max_steps = before_steps + allowance;
            }
            let error = limited
                .extend_witness_front(&columns, &mut pending)
                .unwrap_err();
            assert_eq!(
                error.resource,
                if allowance == work {
                    "Boolean nodes"
                } else {
                    "Boolean work steps"
                }
            );
            for (actual, expected) in pending.iter().zip(&rows) {
                assert_eq!(actual.active, expected.active);
                assert_eq!(actual.witness, expected.witness);
                assert_eq!(actual.terms(), expected.terms());
            }
            let tape = limited.row_witnesses.as_ref().unwrap();
            assert_eq!(tape.explicit_columns, [0].into_iter().collect());
            assert_eq!(tape.initial_explicit_columns, 1);
            assert_eq!(
                tape.ops.len(),
                diagram.row_witnesses.as_ref().unwrap().ops.len()
            );
            assert!(limited.steps() >= before_steps);
            let spent = limited.steps();
            limited.collect_garbage(pending.iter_mut().flat_map(BooleanRow::decisions_mut));
            assert_eq!(limited.steps(), spent);
            limited.limits = BooleanLimits::UNLIMITED;
            limited
                .extend_witness_front(&columns, &mut pending)
                .unwrap();
            for row in &mut pending {
                row.expand_coefficients(&mut limited).unwrap();
            }
            assert!(
                pending
                    .iter()
                    .all(|row| row.witness == RowWitness::Explicit)
            );
        }
    }

    #[test]
    fn limits_stop_allocations_and_survive_collection_and_cloning() {
        let mut diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_nodes: 1,
            max_steps: 2,
        });
        let mut root = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        assert_eq!(
            diagram.make_node(0, DECISION_FALSE, DECISION_TRUE),
            Ok(root)
        );
        let error = diagram
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap_err();
        assert_eq!(
            error,
            BooleanResourceError {
                resource: "Boolean nodes",
                observed: 2,
                limit: 1,
            }
        );
        assert_eq!(diagram.nodes().len(), 1);
        let limits = diagram.limits();
        diagram.collect_garbage([&mut root]);
        diagram.collect_garbage(std::iter::empty());
        assert_eq!(diagram.limits(), limits);
        assert_eq!(diagram.steps(), 1);
        let mut cloned = diagram.clone();
        cloned.charge(1).unwrap();
        let error = cloned
            .make_node(1, DECISION_FALSE, DECISION_TRUE)
            .unwrap_err();
        assert_eq!(
            error,
            BooleanResourceError {
                resource: "Boolean work steps",
                observed: 3,
                limit: 2,
            }
        );
        assert!(cloned.nodes().is_empty());
        assert_eq!(cloned.steps(), 2);

        let mut diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_nodes: 10,
            max_steps: 4,
        });
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        diagram.apply(BooleanOp::And, a, b).unwrap_err();
        assert_eq!(diagram.steps(), 4);
        assert_eq!(diagram.nodes().len(), 2);
        diagram.negate(a).unwrap_err();
        diagram.constrain(a, b).unwrap_err();
        assert_eq!(diagram.nodes().len(), 2);
    }

    #[test]
    fn collection_keeps_only_live_cached_complements() {
        let mut diagram = BooleanDecisionDiagram::default();
        let _garbage = diagram.make_node(8, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut value = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut complement = diagram.negate(value).unwrap();
        assert_eq!(
            diagram.cached_complement(DECISION_FALSE),
            Some(DECISION_TRUE)
        );
        assert_eq!(
            diagram.cached_complement(DECISION_TRUE),
            Some(DECISION_FALSE)
        );

        assert_eq!(diagram.collect_garbage([&mut value, &mut complement]), 1);
        assert_eq!(diagram.cached_complement(value), Some(complement));
        assert_eq!(diagram.cached_complement(complement), Some(value));
        assert_eq!(diagram.collect_garbage([&mut value, &mut complement]), 0);
        assert_eq!(diagram.cached_complement(value), Some(complement));

        assert_eq!(diagram.collect_garbage([&mut value]), 1);
        assert_eq!(diagram.cached_complement(value), None);
        assert_eq!(diagram.collect_garbage(std::iter::empty()), 1);
        assert_eq!(
            diagram.cached_complement(DECISION_FALSE),
            Some(DECISION_TRUE)
        );
        assert_eq!(
            diagram.cached_complement(DECISION_TRUE),
            Some(DECISION_FALSE)
        );
    }

    #[test]
    fn constant_rows_and_queries_spend_work_before_materializing() {
        let mut diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_nodes: 0,
            max_steps: 3,
        });
        let other = BooleanRow::from_terms(DECISION_TRUE, (0..4).map(|i| (i, DECISION_TRUE)));
        let mut row = BooleanRow::default();
        assert!(row.xor_scaled(&other, DECISION_TRUE, &mut diagram).is_err());
        assert!(row.terms().is_empty());
        assert!(diagram.nodes().is_empty());
        diagram.values_up_to(&[DECISION_TRUE; 4], 1).unwrap_err();
        diagram
            .conjunction_witness(&[DECISION_TRUE; 4])
            .unwrap_err();
        assert_eq!(diagram.steps(), 0);
        let reduced = reduce_boolean_rows(vec![other], 0..4, &mut diagram);
        assert!(reduced.is_err(), "constant pivots still consume work");
        assert!(diagram.nodes().is_empty());
    }

    #[test]
    fn iterative_operations_preserve_exact_functions_on_deep_chains() {
        const DEPTH: usize = 20_000;
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram
            .make_node(DEPTH, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let b = diagram
            .make_node(DEPTH + 1, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let (mut left, mut right) = (a, b);
        for variable in (0..DEPTH).rev() {
            left = diagram.make_node(variable, DECISION_FALSE, left).unwrap();
            right = diagram.make_node(variable, DECISION_FALSE, right).unwrap();
        }
        let negated = diagram.negate(right).unwrap();
        let joined = diagram.apply(BooleanOp::And, left, negated).unwrap();
        assert!(diagram.evaluate(joined, |variable| variable != DEPTH + 1));
        assert!(!diagram.evaluate(joined, |_| true));
        assert_eq!(diagram.constrain(left, right).unwrap(), a);
        let witness = diagram
            .conjunction_witness(&[left, negated])
            .unwrap()
            .unwrap();
        assert_eq!(witness.len(), DEPTH + 2);
        assert!(!witness[&(DEPTH + 1)]);
        assert_eq!(
            diagram.conjunction_witness(&[right, negated]).unwrap(),
            None
        );
        assert_eq!(
            diagram.values_up_to(&[left, right], 5).unwrap(),
            vec![
                vec![false, false],
                vec![false, true],
                vec![true, false],
                vec![true, true],
            ]
        );
    }

    #[test]
    fn bounded_queries_and_operations_match_truth_tables() {
        let mut diagram = BooleanDecisionDiagram::default();
        let variables = (0..3)
            .map(|i| diagram.make_node(i, DECISION_FALSE, DECISION_TRUE).unwrap())
            .collect::<Vec<_>>();
        let mut roots = vec![DECISION_FALSE, DECISION_TRUE];
        roots.extend(&variables);
        for operator in [BooleanOp::Xor, BooleanOp::And, BooleanOp::Or] {
            roots.push(diagram.apply(operator, variables[0], variables[1]).unwrap());
        }
        roots.push(diagram.negate(variables[2]).unwrap());
        for &left in &roots {
            for &right in &roots {
                let operations = [BooleanOp::Xor, BooleanOp::And, BooleanOp::Or]
                    .map(|operator| diagram.apply(operator, left, right).unwrap());
                let constrained = diagram.constrain(left, right).unwrap();
                let witness = diagram.conjunction_witness(&[left, right]).unwrap();
                let expected = (0..8)
                    .filter(|&assignment| {
                        let value = |bit| assignment & (1 << bit) != 0;
                        diagram.evaluate(left, value) && diagram.evaluate(right, value)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(witness.is_some(), !expected.is_empty());
                if let Some(witness) = witness {
                    let value = |bit| witness.get(&bit).copied().unwrap_or(false);
                    assert!(diagram.evaluate(left, value) && diagram.evaluate(right, value));
                }
                for assignment in 0..8 {
                    let value = |bit| assignment & (1 << bit) != 0;
                    let a = diagram.evaluate(left, value);
                    let b = diagram.evaluate(right, value);
                    assert_eq!(
                        operations.map(|root| diagram.evaluate(root, value)),
                        [a ^ b, a & b, a | b]
                    );
                    if b {
                        assert_eq!(
                            diagram.evaluate(left, value),
                            diagram.evaluate(constrained, value)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cached_results_survive_clone_and_collection() {
        let mut diagram = BooleanDecisionDiagram::default();
        diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut left = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut right = diagram.make_node(2, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut result = diagram.apply(BooleanOp::And, left, right).unwrap();

        let mut cloned = diagram.clone();
        let steps = cloned.steps();
        assert_eq!(cloned.apply(BooleanOp::And, left, right).unwrap(), result);
        assert_eq!(cloned.steps(), steps);

        assert_eq!(
            diagram.collect_garbage([&mut left, &mut right, &mut result]),
            1
        );
        let steps = diagram.steps();
        assert_eq!(diagram.apply(BooleanOp::And, left, right).unwrap(), result);
        assert_eq!(diagram.steps(), steps);
    }

    #[test]
    fn computed_table_collisions_preserve_every_three_variable_function() {
        fn function(
            diagram: &mut BooleanDecisionDiagram,
            variable: usize,
            truth: u8,
        ) -> DecisionId {
            if variable == 3 {
                return if truth & 1 == 0 {
                    DECISION_FALSE
                } else {
                    DECISION_TRUE
                };
            }
            let low = function(diagram, variable + 1, truth);
            let high = function(diagram, variable + 1, truth >> (1 << (2 - variable)));
            diagram.make_node(variable, low, high).unwrap()
        }

        let mut diagram = BooleanDecisionDiagram::default();
        let functions = (0..=u8::MAX)
            .map(|truth| function(&mut diagram, 0, truth))
            .collect::<Vec<_>>();
        for (truth, &root) in functions.iter().enumerate() {
            for assignment in 0..8 {
                assert_eq!(
                    diagram.evaluate(root, |variable| assignment & (1 << (2 - variable)) != 0),
                    truth & (1 << assignment) != 0,
                );
            }
        }
        for (a, &left) in functions.iter().enumerate() {
            for (b, &right) in functions.iter().enumerate().skip(a + 1) {
                for (operator, truth) in [
                    (BooleanOp::Xor, a ^ b),
                    (BooleanOp::And, a & b),
                    (BooleanOp::Or, a | b),
                ] {
                    assert_eq!(
                        diagram.apply(operator, left, right).unwrap(),
                        functions[truth]
                    );
                }
            }
        }
    }

    #[test]
    fn boolean_stacks_can_be_reused_after_exhaustion_and_collection() {
        for negate in [false, true] {
            let mut diagram = BooleanDecisionDiagram::default();
            let variables = (0..4)
                .map(|variable| {
                    diagram
                        .make_node(variable, DECISION_FALSE, DECISION_TRUE)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let mut roots = [
                diagram
                    .apply(BooleanOp::And, variables[0], variables[2])
                    .unwrap(),
                diagram
                    .apply(BooleanOp::Or, variables[1], variables[3])
                    .unwrap(),
            ];
            diagram.limits.max_steps = diagram.steps() + 2;
            let error = if negate {
                diagram.negate(roots[0])
            } else {
                diagram.apply(BooleanOp::And, roots[0], roots[1])
            }
            .unwrap_err();
            assert_eq!(error.resource, "Boolean work steps");
            diagram.collect_garbage(&mut roots);
            diagram.limits.max_steps = usize::MAX;
            let result = if negate {
                diagram.negate(roots[0])
            } else {
                diagram.apply(BooleanOp::And, roots[0], roots[1])
            }
            .unwrap();
            for assignment in 0..16 {
                let value = |bit| assignment & (1 << bit) != 0;
                let left = value(0) && value(2);
                assert_eq!(
                    diagram.evaluate(result, value),
                    if negate {
                        !left
                    } else {
                        left && (value(1) || value(3))
                    },
                );
            }
        }
    }

    #[test]
    fn collection_advances_the_handle_generation_even_without_removing_nodes() {
        let mut diagram = BooleanDecisionDiagram::default();
        let mut root = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let before = diagram.generation();
        assert_eq!(diagram.collect_garbage([&mut root]), 0);
        assert_ne!(diagram.generation(), before);
        let before = diagram.generation();
        diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        assert_eq!(diagram.collect_garbage([&mut root]), 1);
        assert_ne!(diagram.generation(), before);
        assert!(diagram.evaluate(root, |_| true));
        assert!(!diagram.evaluate(root, |_| false));
    }

    #[test]
    fn collection_preserves_live_functions_and_rebuilds_caches() {
        let mut diagram = BooleanDecisionDiagram::default();
        let variables = (0..8)
            .map(|variable| {
                diagram
                    .make_node(variable, DECISION_FALSE, DECISION_TRUE)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let left = diagram
            .apply(BooleanOp::And, variables[0], variables[1])
            .unwrap();
        let right = diagram
            .apply(BooleanOp::Xor, variables[2], variables[3])
            .unwrap();
        let readout = diagram.apply(BooleanOp::Or, left, right).unwrap();
        let complement = diagram.negate(readout).unwrap();
        let mut garbage = variables[7];
        for &variable in variables.iter().rev() {
            garbage = diagram.apply(BooleanOp::Xor, variable, garbage).unwrap();
        }
        let mut roots = [
            readout,
            complement,
            left,
            variables[0],
            DECISION_FALSE,
            DECISION_TRUE,
            readout,
        ];
        let expected = (0..16)
            .map(|mask| roots.map(|root| diagram.evaluate(root, |bit| mask & (1 << bit) != 0)))
            .collect::<Vec<_>>();
        let before = diagram.nodes().len();
        let removed = diagram.collect_garbage(roots.iter_mut());
        assert!(removed > 0);
        assert_eq!(diagram.nodes().len() + removed, before);
        assert_eq!(roots[0], roots[6]);
        for (mask, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                roots.map(|root| diagram.evaluate(root, |bit| mask & (1 << bit) != 0)),
                expected
            );
        }
        for root in roots {
            let complement = diagram.negate(root).unwrap();
            assert_eq!(diagram.negate(complement).unwrap(), root);
            assert_eq!(
                diagram.apply(BooleanOp::Xor, root, complement).unwrap(),
                DECISION_TRUE
            );
        }
        let before = diagram.nodes().len();
        assert_eq!(diagram.collect_garbage(std::iter::empty()), before);
        assert!(diagram.nodes().is_empty());
        let variable = diagram.make_node(4, DECISION_FALSE, DECISION_TRUE).unwrap();
        assert_eq!(variable, DecisionId(2));
        assert!(diagram.evaluate(variable, |bit| bit == 4));
    }

    #[test]
    fn sparse_row_updates_preserve_boolean_coefficients() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let roots = [
            DECISION_FALSE,
            DECISION_TRUE,
            a,
            b,
            diagram.negate(a).unwrap(),
        ];
        for seed in 0..32 {
            let terms = (0..80)
                .map(|index| {
                    (
                        (index * 17 + seed) % 101,
                        roots[(index + seed) % roots.len()],
                    )
                })
                .collect::<Vec<_>>();
            let mut row = BooleanRow::from_terms(a, terms.iter().copied());
            let original = terms.into_iter().collect::<BTreeMap<_, _>>();
            let other = BooleanRow::from_terms(
                b,
                (0..80).map(|index| {
                    (
                        (index * 23 + seed) % 101,
                        roots[(index * 3 + seed) % roots.len()],
                    )
                }),
            );
            let factor = roots[seed % roots.len()];
            row.xor_scaled(&other, factor, &mut diagram).unwrap();
            assert_eq!(row.active, a, "row operations preserve availability");
            assert!(row.terms().windows(2).all(|pair| pair[0].0 < pair[1].0));
            assert!(
                row.terms()
                    .iter()
                    .all(|&(_, value)| value != DECISION_FALSE)
            );
            for assignment in 0..4 {
                let eval = |root| diagram.evaluate(root, |bit| assignment & (1 << bit) != 0);
                for column in 0..105 {
                    let left = original.get(&column).copied().unwrap_or(DECISION_FALSE);
                    assert_eq!(
                        eval(row.get(column)),
                        eval(left) ^ (eval(factor) & eval(other.get(column)))
                    );
                }
            }
            let before = row.clone();
            row.map_coefficients(|value| diagram.apply(BooleanOp::And, value, a))
                .unwrap();
            row.truncate_columns(50);
            for column in 0..105 {
                let expected = if column < 50 {
                    diagram
                        .apply(BooleanOp::And, before.get(column), a)
                        .unwrap()
                } else {
                    DECISION_FALSE
                };
                assert_eq!(row.get(column), expected);
            }
            assert!(
                row.terms()
                    .iter()
                    .all(|&(column, value)| column < 50 && value != DECISION_FALSE)
            );
        }
    }

    #[test]
    fn indexed_pivots_match_full_scans_with_conditional_witnesses() {
        fn evaluated<'a>(
            rows: impl IntoIterator<Item = &'a BooleanRow>,
            diagram: &BooleanDecisionDiagram,
        ) -> Vec<Vec<bool>> {
            rows.into_iter()
                .map(|row| {
                    (0..4)
                        .flat_map(|assignment| {
                            std::iter::once(row.active)
                                .chain((0..28).map(|column| row.get(column)))
                                .map(move |root| {
                                    diagram.evaluate(root, |bit| assignment & (1 << bit) != 0)
                                })
                        })
                        .collect()
                })
                .collect()
        }
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let roots = [
            DECISION_FALSE,
            DECISION_TRUE,
            a,
            b,
            diagram.negate(a).unwrap(),
            diagram.apply(BooleanOp::Xor, a, b).unwrap(),
        ];
        // Every seed-dependent coefficient repeats modulo roots.len().
        for seed in 0..roots.len() {
            let mut expected = (0..8)
                .map(|index| {
                    let mut row = BooleanRow::from_terms(
                        roots[(index + seed) % roots.len()],
                        (0..6).map(|column| {
                            (
                                column,
                                roots[(index * 7 + column * 3 + seed * (column + 1)) % roots.len()],
                            )
                        }),
                    );
                    row.set(20 + index, DECISION_TRUE);
                    row
                })
                .collect::<Vec<_>>();
            let mut reference_diagram = diagram.clone();
            let mut indexed_diagram = diagram.clone();
            let mut indexed = BooleanRowSpace::new(expected.clone(), &mut indexed_diagram).unwrap();
            for terms in [
                vec![(0, a), (1, b), (0, b), (3, DECISION_FALSE)],
                vec![(4, DECISION_TRUE), (2, a)],
                vec![(1, b), (1, b)],
            ] {
                let expected_pivot = eliminate_boolean_rows(
                    &mut expected,
                    &mut reference_diagram,
                    |row, diagram| {
                        terms
                            .iter()
                            .try_fold(DECISION_FALSE, |sum, &(column, weight)| {
                                let term =
                                    diagram.apply(BooleanOp::And, weight, row.get(column))?;
                                diagram.apply(BooleanOp::Xor, sum, term)
                            })
                    },
                )
                .unwrap();
                let pivot = indexed
                    .eliminate_linear(&terms, &mut indexed_diagram)
                    .unwrap();
                assert_eq!(
                    evaluated([&pivot], &indexed_diagram),
                    evaluated([&expected_pivot], &reference_diagram),
                    "linear pivot, seed {seed}"
                );
                assert_eq!(
                    evaluated(indexed.rows(), &indexed_diagram),
                    evaluated(&expected, &reference_diagram),
                    "linear remainder, seed {seed}"
                );
            }
            let mut expected_retained = Vec::<BooleanRow>::new();
            let mut retained = BooleanRowSpace::default();
            for (column, gate) in [
                (4, a),
                (1, DECISION_TRUE),
                (4, DECISION_TRUE),
                (0, b),
                (5, DECISION_FALSE),
                (5, DECISION_TRUE),
                (3, DECISION_TRUE),
                (2, DECISION_TRUE),
                (0, DECISION_TRUE),
                (5, DECISION_TRUE),
            ] {
                let expected_pivot = eliminate_boolean_rows(
                    &mut expected,
                    &mut reference_diagram,
                    |row, diagram| diagram.apply(BooleanOp::And, gate, row.get(column)),
                )
                .unwrap();
                let pivot = indexed
                    .eliminate_column_when(column, gate, &mut indexed_diagram)
                    .unwrap();
                assert_eq!(
                    evaluated([&pivot], &indexed_diagram),
                    evaluated([&expected_pivot], &reference_diagram),
                    "pivot {column}, seed {seed}"
                );
                assert_eq!(
                    evaluated(indexed.rows(), &indexed_diagram),
                    evaluated(&expected, &reference_diagram),
                    "remaining {column}, seed {seed}"
                );
                if expected_pivot.active != DECISION_FALSE {
                    for row in &mut expected_retained {
                        let factor = reference_diagram
                            .apply(BooleanOp::And, row.active, row.get(column))
                            .unwrap();
                        row.xor_scaled(&expected_pivot, factor, &mut reference_diagram)
                            .unwrap();
                    }
                    retained
                        .reduce_column(column, &pivot, &mut indexed_diagram)
                        .unwrap();
                    // Retaining only named pivots must give the same named rows
                    // as full scans while physical/kernel pivots are discarded.
                    if [4, 1].contains(&column) {
                        expected_retained.push(expected_pivot);
                        retained.push(pivot, &mut indexed_diagram).unwrap();
                    }
                }
                assert_eq!(
                    evaluated(retained.rows(), &indexed_diagram),
                    evaluated(&expected_retained, &reference_diagram),
                    "retained {column}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn indexed_mutations_match_rebuilt_conditional_weighted_spaces() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let roots = [
            DECISION_FALSE,
            DECISION_TRUE,
            a,
            b,
            diagram.negate(a).unwrap(),
            diagram.apply(BooleanOp::Xor, a, b).unwrap(),
        ];
        for seed in 0..24 {
            let mut indexed = BooleanRowSpace::default();
            let mut expected = Vec::<Option<BooleanRow>>::new();
            for index in 0..8 {
                let mut row = BooleanRow::from_terms(
                    roots[(index + seed) % roots.len()],
                    (0..6).map(|column| {
                        (
                            column,
                            roots[(index * 7 + column * 3 + seed * (column + 1)) % roots.len()],
                        )
                    }),
                );
                row.set(20 + index, DECISION_TRUE);
                let id = indexed.insert_indexed(row.clone(), &mut diagram).unwrap();
                if row.active != DECISION_FALSE {
                    assert_eq!(id, Some(expected.len()));
                    expected.push(Some(row));
                } else {
                    assert_eq!(id, None);
                }
            }
            for step in 0..6 {
                let id = (seed + step) % expected.len();
                let erased = (step * 3 + seed) % 6;
                let taken = indexed.take(id, &mut diagram).unwrap();
                let mut row = expected[id].take().unwrap();
                assert_eq!(taken.as_ref().unwrap().terms(), row.terms());
                assert!(indexed.row(id).is_none());
                assert!(indexed.take(id, &mut diagram).unwrap().is_none());
                indexed.erase_column(erased, &mut diagram).unwrap();
                for row in expected.iter_mut().flatten() {
                    row.set(erased, DECISION_FALSE);
                }
                row.set(erased, DECISION_FALSE);
                // A conditional update reintroduces an erased coordinate and
                // changes source witnesses before restoring the same row ID.
                let addition = BooleanRow::from_terms(
                    DECISION_TRUE,
                    [(erased, roots[2 + step % 4]), (40 + step, DECISION_TRUE)],
                );
                row.xor_scaled(&addition, roots[2 + (seed + step) % 4], &mut diagram)
                    .unwrap();
                assert!(
                    indexed
                        .replace(id, row.clone(), &mut diagram)
                        .unwrap()
                        .is_none()
                );
                expected[id] = Some(row.clone());
                // Replacement of a live row must drop old postings, add new
                // ones, and keep its place ahead of later insertions.
                row.set(50 + step, a);
                let old = indexed
                    .replace(id, row.clone(), &mut diagram)
                    .unwrap()
                    .unwrap();
                assert_eq!(old.terms(), expected[id].as_ref().unwrap().terms());
                expected[id] = Some(row);
                let appended = BooleanRow::from_terms(b, [(erased, a), (60 + step, DECISION_TRUE)]);
                assert_eq!(
                    indexed
                        .insert_indexed(appended.clone(), &mut diagram)
                        .unwrap(),
                    Some(expected.len())
                );
                expected.push(Some(appended));
                assert_eq!(indexed.len(), expected.len());
                for (id, row) in expected.iter().enumerate() {
                    let actual = indexed.row(id).unwrap();
                    let row = row.as_ref().unwrap();
                    assert_eq!((actual.active, actual.terms()), (row.active, row.terms()));
                }
                for column in 0..66 {
                    assert_eq!(
                        indexed.row_ids_with_column(column).collect::<Vec<_>>(),
                        expected
                            .iter()
                            .enumerate()
                            .filter_map(|(id, row)| {
                                (row.as_ref().unwrap().get(column) != DECISION_FALSE).then_some(id)
                            })
                            .collect::<Vec<_>>()
                    );
                }
            }
            // Drop one slot through replacement, then append without recycling
            // its ID. A fresh space has the same source order despite the hole.
            let id = seed % expected.len();
            let old = indexed
                .replace(id, BooleanRow::new(DECISION_FALSE), &mut diagram)
                .unwrap()
                .unwrap();
            assert_eq!(old.terms(), expected[id].take().unwrap().terms());
            let appended = BooleanRow::from_terms(a, [(0, b), (69, DECISION_TRUE)]);
            assert_eq!(
                indexed
                    .insert_indexed(appended.clone(), &mut diagram)
                    .unwrap(),
                Some(expected.len())
            );
            expected.push(Some(appended));
            let mut rebuilt =
                BooleanRowSpace::new(expected.into_iter().flatten().collect(), &mut diagram)
                    .unwrap();
            for terms in [
                vec![(0, a), (1, b), (0, b), (3, DECISION_FALSE)],
                vec![(4, DECISION_TRUE), (2, a)],
                vec![(1, b), (1, b)],
                vec![(0, DECISION_TRUE), (5, DECISION_TRUE)],
            ] {
                let actual = indexed.eliminate_linear(&terms, &mut diagram).unwrap();
                let expected = rebuilt.eliminate_linear(&terms, &mut diagram).unwrap();
                assert_eq!(
                    (actual.active, actual.terms()),
                    (expected.active, expected.terms())
                );
                assert_eq!(
                    indexed
                        .rows()
                        .map(|row| (row.active, row.terms()))
                        .collect::<Vec<_>>(),
                    rebuilt
                        .rows()
                        .map(|row| (row.active, row.terms()))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn indexed_mutation_limits_leave_rows_and_postings_unchanged() {
        for operation in 0..4 {
            let mut diagram = BooleanDecisionDiagram::default();
            let rows = vec![
                BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (2, DECISION_TRUE)]),
                BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (3, DECISION_TRUE)]),
            ];
            let mut indexed = BooleanRowSpace::new(rows.clone(), &mut diagram).unwrap();
            // Enough for lookup/candidate counting, but not the actual mutation.
            diagram.limits.max_steps = diagram.steps() + if operation == 0 { 2 } else { 3 };
            let error = match operation {
                0 => indexed.take(0, &mut diagram),
                1 => indexed.replace(0, rows[1].clone(), &mut diagram),
                2 => indexed.erase_column(0, &mut diagram).map(|()| None),
                _ => indexed
                    .insert_indexed(
                        BooleanRow::from_terms(
                            DECISION_TRUE,
                            (0..4).map(|column| (column, DECISION_TRUE)),
                        ),
                        &mut diagram,
                    )
                    .map(|_| None),
            };
            error.unwrap_err();
            assert_eq!(indexed.len(), rows.len());
            for (id, row) in rows.iter().enumerate() {
                assert_eq!(indexed.row(id).unwrap().terms(), row.terms());
            }
            assert_eq!(
                indexed.row_ids_with_column(0).collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(indexed.row_ids_with_column(2).collect::<Vec<_>>(), vec![0]);
            assert_eq!(indexed.row_ids_with_column(3).collect::<Vec<_>>(), vec![1]);
        }
    }

    #[test]
    fn sparse_pivots_preserve_conditional_augmented_spans() {
        fn span<'a>(
            rows: impl IntoIterator<Item = &'a BooleanRow>,
            diagram: &BooleanDecisionDiagram,
            assignment: usize,
        ) -> BTreeSet<u32> {
            let evaluate = |root| diagram.evaluate(root, |bit| assignment & (1 << bit) != 0);
            let mut span = BTreeSet::from([0]);
            for row in rows.into_iter().filter(|row| evaluate(row.active)) {
                let value = (0..24).fold(0, |value, column| {
                    value | (u32::from(evaluate(row.get(column))) << column)
                });
                span.extend(span.iter().map(|prior| prior ^ value).collect::<Vec<_>>());
            }
            span
        }
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let roots = [
            DECISION_FALSE,
            DECISION_TRUE,
            a,
            b,
            diagram.negate(a).unwrap(),
            diagram.apply(BooleanOp::Xor, a, b).unwrap(),
        ];
        // Every seed-dependent coefficient repeats modulo roots.len().
        for seed in 0..roots.len() {
            let rows = (0..8)
                .map(|index| {
                    let mut row = BooleanRow::from_terms(
                        roots[(index + seed) % roots.len()],
                        (0..6).map(|column| {
                            (
                                column,
                                roots[(index * 7 + column * 3 + seed * (column + 1)) % roots.len()],
                            )
                        }),
                    );
                    // Source coordinates make this stronger than a physical-only
                    // span check: every reconstruction coefficient is retained.
                    row.set(16 + index, DECISION_TRUE);
                    row
                })
                .collect::<Vec<_>>();
            let mut stable = BooleanRowSpace::new(rows.clone(), &mut diagram).unwrap();
            let mut sparse = BooleanRowSpace::new(rows, &mut diagram).unwrap();
            for (column, terms) in [
                (None, vec![(0, a), (1, b), (0, b), (3, DECISION_FALSE)]),
                (Some(4), vec![(4, a)]),
                (None, vec![(5, DECISION_TRUE), (2, b)]),
                (Some(0), vec![(0, DECISION_TRUE)]),
                (Some(4), vec![(4, DECISION_TRUE)]),
                (None, vec![(1, b), (1, b)]),
            ] {
                let before = (0..4)
                    .map(|assignment| span(stable.rows(), &diagram, assignment))
                    .collect::<Vec<_>>();
                let (expected, actual) = if let Some(column) = column {
                    let gate = terms[0].1;
                    if gate == DECISION_TRUE {
                        (
                            stable.eliminate_column(column, &mut diagram).unwrap(),
                            sparse
                                .eliminate_column_sparse(column, &mut diagram)
                                .unwrap(),
                        )
                    } else {
                        (
                            stable
                                .eliminate_column_when(column, gate, &mut diagram)
                                .unwrap(),
                            sparse
                                .eliminate_column_when_sparse(column, gate, &mut diagram)
                                .unwrap(),
                        )
                    }
                } else {
                    (
                        stable.eliminate_linear(&terms, &mut diagram).unwrap(),
                        sparse.eliminate_linear(&terms, &mut diagram).unwrap(),
                    )
                };
                assert_eq!(actual.active, expected.active);
                for (assignment, before) in before.into_iter().enumerate() {
                    assert_eq!(
                        span(sparse.rows(), &diagram, assignment),
                        span(stable.rows(), &diagram, assignment),
                        "kernel seed {seed}, assignment {assignment}"
                    );
                    assert_eq!(
                        span(sparse.rows().chain([&actual]), &diagram, assignment),
                        before,
                        "augmented span seed {seed}, assignment {assignment}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_pivot_ordering_is_charged_and_source_stable_on_ties() {
        let rows = vec![
            BooleanRow::from_terms(
                DECISION_TRUE,
                [
                    (0, DECISION_TRUE),
                    (1, DECISION_TRUE),
                    (2, DECISION_TRUE),
                    (10, DECISION_TRUE),
                ],
            ),
            BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (11, DECISION_TRUE)]),
            BooleanRow::from_terms(DECISION_TRUE, [(0, DECISION_TRUE), (12, DECISION_TRUE)]),
        ];
        let mut limited = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_nodes: 0,
            max_steps: 15,
        });
        let mut space = BooleanRowSpace::new(rows.clone(), &mut limited).unwrap();
        let error = space.eliminate_column_sparse(0, &mut limited).unwrap_err();
        assert_eq!(error.resource, "Boolean work steps");
        assert_eq!(error.limit, 15);
        assert_eq!(
            space.rows().map(BooleanRow::terms).collect::<Vec<_>>(),
            rows.iter().map(BooleanRow::terms).collect::<Vec<_>>()
        );
        let mut diagram = BooleanDecisionDiagram::default();
        let mut stable = BooleanRowSpace::new(rows.clone(), &mut diagram).unwrap();
        let mut sparse = BooleanRowSpace::new(rows.clone(), &mut diagram).unwrap();
        assert_eq!(
            stable.eliminate_column(0, &mut diagram).unwrap().terms(),
            rows[0].terms()
        );
        assert_eq!(
            sparse
                .eliminate_column_sparse(0, &mut diagram)
                .unwrap()
                .terms(),
            rows[1].terms()
        );
    }

    #[test]
    fn indexed_columns_survive_decision_collection() {
        let mut diagram = BooleanDecisionDiagram::default();
        diagram
            .make_node(99, DECISION_FALSE, DECISION_TRUE)
            .unwrap();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let mut indexed = BooleanRowSpace::new(
            vec![
                BooleanRow::from_terms(DECISION_TRUE, [(2, a), (7, DECISION_TRUE)]),
                BooleanRow::from_terms(a, [(2, DECISION_TRUE), (9, a)]),
            ],
            &mut diagram,
        )
        .unwrap();
        let before = diagram.steps();
        assert_eq!(diagram.collect_garbage(indexed.decisions_mut()), 1);
        assert_eq!(diagram.steps(), before);
        assert_eq!(indexed.len(), 2);
        assert_eq!(indexed.rows_with_column(2).count(), 2);
        assert_eq!(indexed.rows_with_column(7).count(), 1);
        let mut reference_diagram = diagram.clone();
        let mut reference = indexed.rows().cloned().collect::<Vec<_>>();
        let expected = eliminate_boolean_rows(&mut reference, &mut reference_diagram, |row, _| {
            Ok(row.get(2))
        })
        .unwrap();
        let pivot = indexed.eliminate_column(2, &mut diagram).unwrap();
        for value in [false, true] {
            let evaluate = |row: &BooleanRow, diagram: &BooleanDecisionDiagram| {
                std::iter::once(row.active)
                    .chain((0..10).map(|column| row.get(column)))
                    .map(|root| diagram.evaluate(root, |_| value))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                evaluate(&pivot, &diagram),
                evaluate(&expected, &reference_diagram)
            );
            assert_eq!(
                indexed
                    .rows()
                    .map(|row| evaluate(row, &diagram))
                    .collect::<Vec<_>>(),
                reference
                    .iter()
                    .map(|row| evaluate(row, &reference_diagram))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn indexed_reduction_scales_with_sparse_incidence() {
        let rows = (0..128)
            .map(|column| BooleanRow::from_terms(DECISION_TRUE, [(column, DECISION_TRUE)]))
            .collect::<Vec<_>>();
        let mut diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_nodes: 0,
            max_steps: 2_000,
        });
        let reduced = reduce_boolean_rows(rows, 0..128, &mut diagram).unwrap();
        assert_eq!(reduced.len(), 128);
        for (column, row) in reduced.iter().enumerate() {
            assert_eq!(row.active, DECISION_TRUE);
            assert_eq!(row.terms(), [(column, DECISION_TRUE)]);
        }
        // Full-row forward/backward scans need quadratic work even though all
        // rows are disjoint. Incidence construction and each pivot are linear.
        assert!(diagram.steps() < 2_000);
    }

    #[test]
    fn sparse_reduction_skips_absent_and_repeated_columns_in_requested_order() {
        let mut diagram = BooleanDecisionDiagram::with_limits(BooleanLimits {
            max_steps: 11_000,
            ..BooleanLimits::DEFAULT
        });
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let not_a = diagram.negate(a).unwrap();
        let (low, middle, high) = (10_000, 3, 10_007);
        let rows = vec![
            BooleanRow::from_terms(a, [(low, DECISION_TRUE), (high, DECISION_TRUE)]),
            BooleanRow::from_terms(not_a, [(middle, DECISION_TRUE), (high, DECISION_TRUE)]),
            BooleanRow::from_terms(DECISION_TRUE, [(low, DECISION_TRUE), (middle, b)]),
        ];
        let mut compact_diagram = diagram.clone();
        let expected =
            reduce_boolean_rows(rows.clone(), [high, middle, low], &mut compact_diagram).unwrap();
        let reduced = reduce_boolean_rows(
            rows.clone(),
            [high, middle, high].into_iter().chain(0..=high),
            &mut diagram,
        )
        .unwrap();
        assert_eq!(reduced.len(), 3);
        assert_eq!(
            reduced[0].get(high),
            DECISION_TRUE,
            "high column has first priority"
        );
        assert_eq!(
            reduced
                .iter()
                .map(|row| (row.active, row.terms()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|row| (row.active, row.terms()))
                .collect::<Vec<_>>(),
        );
        // The wide range still costs one lookup per requested column, but no
        // whole-row scans or factor vectors for its ten thousand empty columns.
        assert!(diagram.steps() < 10_200);
        for assignment in 0..4 {
            let span = |rows: &[BooleanRow]| {
                let evaluate = |root| diagram.evaluate(root, |bit| assignment & (1 << bit) != 0);
                let mut span = BTreeSet::from([0_u8]);
                for row in rows.iter().filter(|row| evaluate(row.active)) {
                    let value = [high, middle, low]
                        .into_iter()
                        .enumerate()
                        .fold(0, |value, (bit, column)| {
                            value | (u8::from(evaluate(row.get(column))) << bit)
                        });
                    span.extend(span.iter().map(|prior| prior ^ value).collect::<Vec<_>>());
                }
                span
            };
            assert_eq!(span(&reduced), span(&rows));
        }
    }

    #[test]
    fn reduction_preserves_every_concrete_span() {
        let mut diagram = BooleanDecisionDiagram::default();
        let a = diagram.make_node(0, DECISION_FALSE, DECISION_TRUE).unwrap();
        let b = diagram.make_node(1, DECISION_FALSE, DECISION_TRUE).unwrap();
        let roots = [
            DECISION_FALSE,
            DECISION_TRUE,
            a,
            b,
            diagram.negate(a).unwrap(),
            diagram.apply(BooleanOp::And, a, b).unwrap(),
            diagram.apply(BooleanOp::Xor, a, b).unwrap(),
        ];
        for seed in 0..roots.len() {
            let rows = (0..5)
                .map(|row| BooleanRow {
                    active: roots[(seed + row * 3) % roots.len()],
                    bits: (0..4)
                        .map(|col| (col, roots[(seed * (col + 1) + row * 2) % roots.len()]))
                        .filter(|(_, value)| *value != DECISION_FALSE)
                        .collect(),
                    witness: RowWitness::Explicit,
                })
                .collect::<Vec<_>>();
            let reduced = reduce_boolean_rows(rows.clone(), 0..4, &mut diagram).unwrap();
            for assignment in 0..4 {
                let span = |rows: &[BooleanRow]| {
                    let evaluate =
                        |root| diagram.evaluate(root, |bit| assignment & (1 << bit) != 0);
                    let mut span = BTreeSet::from([0_u8]);
                    for row in rows.iter().filter(|row| evaluate(row.active)) {
                        let value = (0..4).fold(0, |value, col| {
                            value | (u8::from(evaluate(row.get(col))) << col)
                        });
                        span.extend(span.iter().map(|prior| prior ^ value).collect::<Vec<_>>());
                    }
                    span
                };
                assert_eq!(
                    span(&rows),
                    span(&reduced),
                    "seed {seed}, assignment {assignment}"
                );
            }
        }
    }
}
