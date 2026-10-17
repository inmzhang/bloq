//! Static evaluation of the classical dataflow graph under a fixed path.
//!
//! A compiled program's classical nodes describe *how* an observable's parity
//! is assembled from measurement records, not what it evaluates to — the bits
//! only exist per shot. What an offline consumer needs is the recipe: which
//! measurement sites XOR into a given node's value, and with what sign. That
//! is a fold over [`ClassicalNode`]/[`ClassicalExpr`], and every consumer that
//! needed it reimplemented the same recursion.
//!
//! Folding requires a *fixed path* for activation and nonlinear expressions.
//! [`ClassicalAssignment`] supplies their predicate values;
//! resolution fails rather than guessing when the path does not determine it.
//!
//! The XOR-expressible fragment resolves directly (SEM-FOLD). On a fixed path,
//! `Select` resolves through its selected arm and a fully fixed `And`/`Or`
//! resolves to a constant; unresolved nonlinear values are rejected instead of
//! approximated.
//!
//! [`Bloq::measurement_dependencies`] asks the same fold a weaker question —
//! *which measurements does this value wait on* rather than *what parity is it*
//! — so it traces Flip queries and unions nonlinear operands.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{
    Bloq, BloqNodeId, BloqNodeKind, ClassicalExpr, ClassicalNode, InstanceMeasurement,
    NodeDetectorParity, ObservableOutput, RegionNode, SubGraph, ValueRef,
};

/// The fixed path a [`Bloq::resolve_classical`] fold evaluates under.
///
/// Predicate inputs are fixed uniformly or through known observable values.
#[derive(Debug, Clone, Copy)]
pub enum ClassicalAssignment<'a> {
    /// Every predicate input and observable leaf folds to the same bit.
    Uniform(bool),
    /// Observable values fixing activation and expressions through their inputs.
    /// A condition that reaches an unknown value is an error, not a default.
    Pinned {
        /// Values for [`ClassicalNode::Observable`] indices this path fixes,
        /// keyed by observable index. These are corrected values; they do not
        /// fix a decoder flip.
        forced_observables: &'a BTreeMap<u32, bool>,
    },
}

/// Whether a fold derives an XOR recipe or only measurement dependencies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
enum FoldMode {
    #[default]
    Parity,
    MeasurementDependencies,
}

/// What a classical node's value is made of, under one fixed path.
///
/// `measurements` and the named decoder flips XORed together, then negated when
/// `sign` is set, reproduce the node's per-shot bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassicalResolution {
    /// The measurement sites whose parity the node's value is.
    pub measurements: BTreeSet<InstanceMeasurement>,
    /// The constant the parity is XORed with: `true` inverts.
    pub sign: bool,
    /// Decoder-flip observables XORed into this value, in ascending order.
    /// Repeated occurrences cancel like repeated measurements.
    pub decoder_observables: BTreeSet<u32>,
}

/// Why a classical fold could not be completed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ResolveError {
    /// The requested node is missing.
    #[error("node {node:?} does not exist at the level being resolved")]
    MissingNode {
        /// Missing node.
        node: BloqNodeId,
    },
    /// A fold reached a quantum node.
    #[error("classical fold reached quantum node {node:?}")]
    NotClassical {
        /// Quantum node.
        node: BloqNodeId,
    },
    /// A classical node has no statically foldable value.
    #[error("classical node {node:?} has no value a static fold can produce")]
    UnsupportedNode {
        /// Unsupported node.
        node: BloqNodeId,
    },
    /// An expression input slot is unwired.
    #[error("node {node:?} reads slot {slot}, which has no producer")]
    UnwiredSlot {
        /// Consumer node.
        node: BloqNodeId,
        /// Missing slot.
        slot: u32,
    },
    /// An expression has no affine measurement recipe.
    #[error("node {node:?} folds a non-linear expression, which has no measurement recipe")]
    NonLinearExpr {
        /// Nonlinear node.
        node: BloqNodeId,
    },
    /// The selected path does not fix a condition.
    #[error("condition at node {node:?} is not fixed by the resolved path")]
    ConditionUnresolved {
        /// Unresolved node.
        node: BloqNodeId,
    },
    /// Classical value dependencies are cyclic.
    #[error("value graph cycles through node {node:?}")]
    CyclicValueGraph {
        /// Node closing the cycle.
        node: BloqNodeId,
    },
}

/// A node's affine value: physical measurement parity plus decoder-flip parity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Value {
    parity: NodeDetectorParity,
    decoder_observables: BTreeSet<u32>,
}

impl Value {
    fn from_measurements(measurements: impl IntoIterator<Item = InstanceMeasurement>) -> Self {
        Self {
            parity: NodeDetectorParity::from_measurements(measurements),
            decoder_observables: BTreeSet::new(),
        }
    }

    fn measurements(&self) -> impl Iterator<Item = InstanceMeasurement> + '_ {
        self.parity.measurements()
    }

    fn sign(&self) -> bool {
        self.parity.sign()
    }

    fn with_sign(mut self, sign: bool) -> Self {
        self.parity = self.parity.with_sign(sign);
        self
    }

    fn xor_assign(&mut self, other: &Self) {
        self.parity.xor_assign(&other.parity);
        for &observable in &other.decoder_observables {
            if !self.decoder_observables.insert(observable) {
                self.decoder_observables.remove(&observable);
            }
        }
    }

    fn with_decoder_observable(mut self, observable: u32) -> Self {
        if !self.decoder_observables.insert(observable) {
            self.decoder_observables.remove(&observable);
        }
        self
    }
}

/// The producers wired into `node`'s value slots, keyed by slot.
///
/// WF-2 makes a slot's producer unique, so the map is total over the fed
/// slots.
fn value_inputs_by_slot(level: &SubGraph, node: BloqNodeId) -> BTreeMap<u32, ValueRef> {
    level
        .value_inputs(node)
        .map(|input| (input.slot, input.value_ref().expect("runtime input")))
        .collect()
}

fn check_output(
    level: &SubGraph,
    node: BloqNodeId,
    output: Option<ObservableOutput>,
) -> Result<(), ResolveError> {
    if output == Some(ObservableOutput::Flip) {
        let payload = level.node(node).ok_or(ResolveError::MissingNode { node })?;
        if payload.try_quantum().is_some() {
            return Err(ResolveError::NotClassical { node });
        }
        if !matches!(
            payload.try_classical(),
            Some(ClassicalNode::Observable { index: Some(_), .. })
        ) {
            return Err(ResolveError::UnsupportedNode { node });
        }
    }
    Ok(())
}

impl SubGraph {
    /// Resolve a level-local node's affine physical/decoder recipe under a fixed path.
    ///
    /// Uses the same semantics and errors as [`Bloq::resolve_classical`].
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] when the path cannot resolve the node to an
    /// affine physical/decoder parity.
    pub fn resolve_classical(
        &self,
        node: BloqNodeId,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<ClassicalResolution, ResolveError> {
        self.resolve_value(node.into(), assignment)
    }

    /// Resolve a selected runtime output under a fixed path.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::resolve_classical`].
    pub fn resolve_value(
        &self,
        value: ValueRef,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<ClassicalResolution, ResolveError> {
        let mut resolver = Resolver::new(assignment, FoldMode::Parity);
        let value = resolver.node_value(self, value)?;
        Ok(ClassicalResolution {
            measurements: value.measurements().collect(),
            sign: value.sign(),
            decoder_observables: value.decoder_observables,
        })
    }
}

impl Bloq {
    /// The physical-measurement and decoder-flip recipe behind a top-level
    /// classical node's value, folded under a fixed path.
    ///
    /// Follows the node's `Value` inputs recursively, descending into a region
    /// under `assignment` and out through that body's declared
    /// bit result (SEM-RVAL). `Observable` fragments and complete observables
    /// contribute measurements; `Const` leaves contribute sign. An observable
    /// also contributes its named decoder flip on its Corrected output;
    /// composition includes only its recipe.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] when the path does not determine a region or
    /// nonlinear cofactor, or when the fold reaches a node with no static value
    /// (a quantum node or a `Discard`).
    pub fn resolve_classical(
        &self,
        node: BloqNodeId,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<ClassicalResolution, ResolveError> {
        self.top().resolve_classical(node, assignment)
    }

    /// Resolve a selected runtime output under a fixed path.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::resolve_classical`].
    pub fn resolve_value(
        &self,
        value: ValueRef,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<ClassicalResolution, ResolveError> {
        self.top().resolve_value(value, assignment)
    }

    /// Which measurement records `node`'s value waits on, rather than what
    /// parity it is.
    ///
    /// Unlike the parity fold, this conservatively unions all record dependencies,
    /// including records needed for Flip outputs and nonlinear expressions.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::resolve_classical`].
    pub fn measurement_dependencies(
        &self,
        node: BloqNodeId,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<BTreeSet<InstanceMeasurement>, ResolveError> {
        self.measurement_value_dependencies(node.into(), assignment)
    }

    /// Measurement dependencies of a selected runtime output, including Flip.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::measurement_dependencies`].
    pub fn measurement_value_dependencies(
        &self,
        value: ValueRef,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<BTreeSet<InstanceMeasurement>, ResolveError> {
        let mut resolver = Resolver::new(assignment, FoldMode::MeasurementDependencies);
        Ok(resolver
            .node_value(self.top(), value)?
            .measurements()
            .collect())
    }

    /// The bit a top-level node already takes on a fixed path.
    ///
    /// Where [`Self::resolve_classical`] answers *which measurements* a value
    /// is assembled from, this answers *what that value is* — for the nodes a
    /// path determines offline, with no per-shot record involved. A
    /// `Observable` reads its corrected bit from the path's
    /// `forced_observables`, and a `Compute` folds its own expression over
    /// those leaves.
    ///
    /// Under [`ClassicalAssignment::Uniform`] every predicate leaf folds to
    /// the uniform bit, matching how a uniform path fixes predicate inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] when the node is quantum, when it is a
    /// `RepeatUntilSuccess` region (whose restart predicate is per-attempt
    /// decoder state, not a path constant), or when the fold reaches a node
    /// the path does not fix — a fragment's per-shot record, or an
    /// observable with no forced value.
    pub fn classical_value(
        &self,
        node: BloqNodeId,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<bool, ResolveError> {
        self.classical_ref_value(node.into(), assignment)
    }

    /// A selected output's bit when fixed by the supplied path.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::classical_value`].
    pub fn classical_ref_value(
        &self,
        value: ValueRef,
        assignment: ClassicalAssignment<'_>,
    ) -> Result<bool, ResolveError> {
        let node = value.node;
        let mut resolver = Resolver::new(assignment, FoldMode::Parity);
        let level = self.top();
        match &level
            .node(node)
            .ok_or(ResolveError::MissingNode { node })?
            .kind
        {
            BloqNodeKind::Region(_) => Err(ResolveError::UnsupportedNode { node }),
            BloqNodeKind::Quantum(_) => Err(ResolveError::NotClassical { node }),
            BloqNodeKind::Classical(_) => resolver.condition_producer(level, value),
        }
    }
}

struct Resolver<'a> {
    assignment: ClassicalAssignment<'a>,
    mode: FoldMode,
    /// The nodes currently being folded, as `(level, node, output)`. Levels are
    /// compared by address because a [`BloqNodeId`] only identifies a node
    /// within its own level. Guards against a cycle in a hand-built or
    /// text-loaded program, which validation would reject (WF-1) but this fold
    /// may run before.
    active: BTreeSet<(*const SubGraph, BloqNodeId, Option<ObservableOutput>)>,
    values: BTreeMap<
        (*const SubGraph, BloqNodeId, Option<ObservableOutput>),
        Result<Value, ResolveError>,
    >,
    conditions: BTreeMap<(*const SubGraph, ValueRef), Result<bool, ResolveError>>,
}

impl<'a> Resolver<'a> {
    fn new(assignment: ClassicalAssignment<'a>, mode: FoldMode) -> Self {
        Self {
            assignment,
            mode,
            active: BTreeSet::new(),
            values: BTreeMap::new(),
            conditions: BTreeMap::new(),
        }
    }
    fn combine(&self, mut left: Value, right: &Value) -> Value {
        if self.mode != FoldMode::Parity {
            let mut measurements: BTreeSet<_> = left.measurements().collect();
            measurements.extend(right.measurements());
            Value::from_measurements(measurements)
        } else {
            left.xor_assign(right);
            left
        }
    }

    fn node_value(
        &mut self,
        level: &SubGraph,
        value: impl Into<ValueRef>,
    ) -> Result<Value, ResolveError> {
        let value = value.into();
        self.node_value_at(level, value.node, Some(value.output))
    }

    fn node_value_at(
        &mut self,
        level: &SubGraph,
        node: BloqNodeId,
        output: Option<ObservableOutput>,
    ) -> Result<Value, ResolveError> {
        let frame = (std::ptr::from_ref(level), node, output);
        if let Some(value) = self.values.get(&frame) {
            return value.clone();
        }
        if !self.active.insert(frame) {
            return Err(ResolveError::CyclicValueGraph { node });
        }
        let value = (|| {
            check_output(level, node, output)?;
            if !self.node_active(level, node)? {
                return Ok(Value::default());
            }
            self.node_value_uncycled(level, node, output)
        })();
        self.active.remove(&frame);
        self.values.insert(frame, value.clone());
        value
    }

    fn node_active(&mut self, level: &SubGraph, node: BloqNodeId) -> Result<bool, ResolveError> {
        let activation = level
            .node(node)
            .ok_or(ResolveError::MissingNode { node })?
            .activation;
        let Some(slot) = activation else {
            return Ok(true);
        };
        let guard = level
            .value_inputs(node)
            .find(|input| input.slot == slot)
            .map(|input| input.value_ref().expect("runtime input"))
            .ok_or(ResolveError::UnwiredSlot { node, slot })?;
        self.condition_producer(level, guard)
    }

    fn measurement_value(&self, measurements: &[InstanceMeasurement]) -> Value {
        if self.mode == FoldMode::Parity {
            Value::from_measurements(measurements.iter().copied())
        } else {
            Value::from_measurements(measurements.iter().copied().collect::<BTreeSet<_>>())
        }
    }

    fn node_value_uncycled(
        &mut self,
        level: &SubGraph,
        node: BloqNodeId,
        output: Option<ObservableOutput>,
    ) -> Result<Value, ResolveError> {
        let payload = match &level
            .node(node)
            .ok_or(ResolveError::MissingNode { node })?
            .kind
        {
            BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { body, .. }) => {
                // Only the accepted attempt contributes its declared result.
                return body.value_output().map_or_else(
                    || Ok(Value::default()),
                    |producer| self.node_value(body, producer),
                );
            }
            BloqNodeKind::Quantum(_) => return Err(ResolveError::NotClassical { node }),
            BloqNodeKind::Classical(payload) => payload.as_ref(),
        };
        match payload {
            ClassicalNode::Compute { expr } => {
                let inputs = value_inputs_by_slot(level, node);
                self.expression(level, node, expr, &inputs)
            }
            ClassicalNode::Observable {
                index,
                measurements,
                ..
            } => {
                let mut value = self.measurement_value(measurements);
                for input in level.data_inputs(node) {
                    let input = self.node_value_at(level, input.producer, input.output)?;
                    value = self.combine(value, &input);
                }
                if self.mode == FoldMode::Parity {
                    match (output, index) {
                        (Some(ObservableOutput::Corrected), Some(index)) => {
                            value = value.with_decoder_observable(*index)
                        }
                        (Some(ObservableOutput::Flip), Some(index)) => {
                            value = Value::default().with_decoder_observable(*index)
                        }
                        _ => {}
                    }
                }
                Ok(value)
            }
            ClassicalNode::Discard { .. } => Err(ResolveError::UnsupportedNode { node }),
        }
    }

    fn expression(
        &mut self,
        level: &SubGraph,
        node: BloqNodeId,
        expr: &ClassicalExpr,
        inputs: &BTreeMap<u32, ValueRef>,
    ) -> Result<Value, ResolveError> {
        match expr {
            ClassicalExpr::In(slot) => {
                let producer = *inputs
                    .get(slot)
                    .ok_or(ResolveError::UnwiredSlot { node, slot: *slot })?;
                self.node_value(level, producer)
            }
            ClassicalExpr::Const(bit) => Ok(Value::default().with_sign(*bit)),
            ClassicalExpr::Parity {
                inputs: slots,
                constant,
            } => {
                let mut value = Value::default().with_sign(*constant);
                for &slot in slots {
                    let operand = self.expression(level, node, &ClassicalExpr::In(slot), inputs)?;
                    value = self.combine(value, &operand);
                }
                Ok(value)
            }
            ClassicalExpr::Not(inner) => {
                let value = self.expression(level, node, inner, inputs)?;
                let sign = !value.sign();
                Ok(value.with_sign(sign))
            }
            ClassicalExpr::Select(operands) if self.mode == FoldMode::Parity => {
                let [condition, when_false, when_true] = operands.as_ref();
                let selected = if self.fixed_condition(level, node, condition)? {
                    when_true
                } else {
                    when_false
                };
                self.expression(level, node, selected, inputs)
            }
            ClassicalExpr::And(..) | ClassicalExpr::Or(..) if self.mode == FoldMode::Parity => {
                let value = self.fixed_condition(level, node, expr)?;
                Ok(Value::default().with_sign(value))
            }
            ClassicalExpr::Xor(_)
            | ClassicalExpr::And(_)
            | ClassicalExpr::Or(_)
            | ClassicalExpr::Select(_) => {
                let mut value = Value::default();
                for operand in expr.operands() {
                    let operand = self.expression(level, node, operand, inputs)?;
                    value = self.combine(value, &operand);
                }
                Ok(value)
            }
        }
    }

    /// Fold a region predicate to the bit it takes on the resolved path.
    ///
    /// Unlike [`Self::expression`] this yields a plain bit, so an unresolved
    /// condition is an error.
    fn condition(
        &mut self,
        level: &SubGraph,
        node: BloqNodeId,
        expr: &ClassicalExpr,
    ) -> Result<bool, ResolveError> {
        match expr {
            ClassicalExpr::Const(bit) => Ok(*bit),
            ClassicalExpr::Parity { inputs, constant } => {
                inputs.iter().try_fold(*constant, |value, &slot| {
                    Ok(value ^ self.condition(level, node, &ClassicalExpr::In(slot))?)
                })
            }
            ClassicalExpr::In(slot) => match self.assignment {
                // A uniform path fixes every condition directly; there is no
                // observable table to resolve the producer against.
                ClassicalAssignment::Uniform(pin) => Ok(pin),
                ClassicalAssignment::Pinned { .. } => {
                    let producer = level
                        .value_inputs(node)
                        .find(|input| input.slot == *slot)
                        .map(|input| input.value_ref().expect("runtime input"))
                        .ok_or(ResolveError::UnwiredSlot { node, slot: *slot })?;
                    self.condition_producer(level, producer)
                }
            },
            ClassicalExpr::Not(inner) => Ok(!self.condition(level, node, inner)?),
            ClassicalExpr::Xor(operands) => operands.iter().try_fold(false, |value, operand| {
                Ok(value ^ self.condition(level, node, operand)?)
            }),
            ClassicalExpr::And(operands) => operands.iter().try_fold(true, |value, operand| {
                Ok(value & self.condition(level, node, operand)?)
            }),
            ClassicalExpr::Or(operands) => operands.iter().try_fold(false, |value, operand| {
                Ok(value | self.condition(level, node, operand)?)
            }),
            ClassicalExpr::Select(operands) => {
                let [condition, when_false, when_true] = operands.as_ref();
                let condition = self.condition(level, node, condition)?;
                let when_false = self.condition(level, node, when_false)?;
                let when_true = self.condition(level, node, when_true)?;
                Ok(if condition { when_true } else { when_false })
            }
        }
    }

    /// A Boolean cofactor needed to turn a nonlinear expression into an exact
    /// fixed-path parity. An unavailable predicate means the nonlinear value
    /// still has no affine measurement recipe.
    fn fixed_condition(
        &mut self,
        level: &SubGraph,
        node: BloqNodeId,
        expr: &ClassicalExpr,
    ) -> Result<bool, ResolveError> {
        self.condition(level, node, expr)
            .map_err(|error| match error {
                ResolveError::ConditionUnresolved { .. } => ResolveError::NonLinearExpr { node },
                error => error,
            })
    }

    /// Fold the value a predicate reads out of one classical producer.
    ///
    /// The pinned path reaches this through a condition's `In` slot; a caller
    /// asking [`Bloq::classical_value`] for a node's own bit enters here
    /// directly, which is why the uniform path is handled rather than
    /// unreachable.
    fn condition_producer(
        &mut self,
        level: &SubGraph,
        value: impl Into<ValueRef>,
    ) -> Result<bool, ResolveError> {
        let value = value.into();
        let node = value.node;
        let frame = (std::ptr::from_ref(level), value);
        if let Some(value) = self.conditions.get(&frame) {
            return value.clone();
        }
        if !self.active.insert((frame.0, node, Some(value.output))) {
            return Err(ResolveError::CyclicValueGraph { node });
        }
        let result = self.condition_producer_uncycled(level, value);
        self.active.remove(&(frame.0, node, Some(value.output)));
        self.conditions.insert(frame, result.clone());
        result
    }

    fn condition_producer_uncycled(
        &mut self,
        level: &SubGraph,
        value: ValueRef,
    ) -> Result<bool, ResolveError> {
        let node = value.node;
        check_output(level, node, Some(value.output))?;
        let payload = level.node(node).ok_or(ResolveError::MissingNode { node })?;
        if let Some(slot) = payload.activation {
            let guard = level
                .value_inputs(node)
                .find(|input| input.slot == slot)
                .ok_or(ResolveError::UnwiredSlot { node, slot })?;
            let guard = guard.value_ref().expect("runtime input");
            if !self.condition_producer(level, guard)? {
                return Ok(false);
            }
        }
        let classical = payload
            .try_classical()
            .ok_or(ResolveError::NotClassical { node })?;
        let observable = match classical {
            ClassicalNode::Observable { index, .. }
                if value.output == ObservableOutput::Corrected =>
            {
                index.as_ref()
            }
            ClassicalNode::Compute { expr } => return self.condition(level, node, expr),
            _ => None,
        };
        let observable = observable.ok_or(ResolveError::ConditionUnresolved { node })?;
        match self.assignment {
            // A uniform path fixes every predicate leaf to the same bit, and
            // carries no observable table to look one up in.
            ClassicalAssignment::Uniform(pin) => Ok(pin),
            ClassicalAssignment::Pinned { forced_observables } => forced_observables
                .get(observable)
                .copied()
                .ok_or(ResolveError::ConditionUnresolved { node }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(instance: u32, measurement: u32) -> InstanceMeasurement {
        InstanceMeasurement {
            instance: crate::TemplateInstanceId(instance),
            measurement,
        }
    }

    #[test]
    fn unsupported_flip_ports_fail_in_value_queries_even_when_inactive() {
        for classical in [
            ClassicalNode::Compute {
                expr: ClassicalExpr::Const(true),
            },
            ClassicalNode::fragment(),
        ] {
            for active in [None, Some(false), Some(true)] {
                let mut program = Bloq::new();
                let mut producer = crate::BloqNode::classical(classical.clone());
                producer.activation = active.map(|_| 0);
                let node = program.add_node(producer);
                if let Some(active) = active {
                    let guard =
                        program.add_node(crate::BloqNode::classical(ClassicalNode::Compute {
                            expr: ClassicalExpr::Const(active),
                        }));
                    program.add_edge(guard, node, crate::BloqEdge::value(0));
                }
                let value = ValueRef {
                    node,
                    output: ObservableOutput::Flip,
                };
                let assignment = ClassicalAssignment::Uniform(false);
                let expected = ResolveError::UnsupportedNode { node };
                assert_eq!(
                    program.classical_ref_value(value, assignment),
                    Err(expected.clone())
                );
                assert_eq!(
                    program.resolve_value(value, assignment),
                    Err(expected.clone())
                );
                assert_eq!(
                    program.measurement_value_dependencies(value, assignment),
                    Err(expected)
                );
            }
        }
    }

    /// Two fragments feed an observable whose corrected parity is inverted by
    /// a frame Compute. A retry region exports a third fragment from its accepted attempt.
    fn program() -> Bloq {
        Bloq::from_text(
            "\
BLOQIR 1

template t0 {
  circuit {
    MPP X(0,0):m0
    MPP X(2,0):m1
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0 from generator 0
  n2 observable fragment measurements i0:m1 from generator 1
  n3 observable 0
  n4 compute in0
  n5 compute !in0 from frame x (0,0,1)
  n6 rus in0 {
    body {
      n0 observable fragment measurements i0:m0 from generator 2
      result n0
    }
  }
  n7 compute in0 ^ in1
  n1 -> n3 compose 0
  n2 -> n3 compose 1
  n3 -> n4 value 0
  n4 -> n5 value 0
  n4 -> n6 value 0
  n5 -> n7 value 0
  n6 -> n7 value 1
}
",
        )
        .expect("valid .bloqir text")
    }

    #[test]
    fn a_region_result_is_independent_of_internal_consumers_and_unused_values() {
        let mut program = Bloq::from_text(
            "BLOQIR 1
 graph {
   n0 rus 0 {
     body {
       n0 compute 1
       n1 observable 0
       n2 compute !in0
       n0 -> n1 value 0
       n1 -> n2 value 0
       result n1
     }
   }
   n1 compute in0
   n0 -> n1 value 0
 }
",
        )
        .unwrap();
        program.validate().unwrap();
        assert!(
            program
                .resolve_classical(BloqNodeId(1), ClassicalAssignment::Uniform(true))
                .unwrap()
                .sign
        );
        let BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { body, .. }) =
            &mut program.top_mut().node_mut(BloqNodeId(0)).unwrap().kind
        else {
            unreachable!()
        };
        body.add_node(crate::BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        program.validate().unwrap();
        assert!(
            program
                .resolve_classical(BloqNodeId(1), ClassicalAssignment::Uniform(true))
                .unwrap()
                .sign
        );
    }

    #[test]
    fn resolution_folds_accumulates_through_a_decoded_frame() {
        let resolution = program()
            .resolve_classical(BloqNodeId(5), ClassicalAssignment::Uniform(false))
            .expect("the frame compute resolves");

        assert_eq!(
            resolution.measurements,
            BTreeSet::from([measurement(0, 0), measurement(0, 1)])
        );
        assert!(resolution.sign, "the frame compute negates its input");
        assert_eq!(resolution.decoder_observables, BTreeSet::from([0]));
    }

    #[test]
    fn composition_preserves_prior_folds_without_the_childs_decoder_flip() {
        let mut program = Bloq::new();
        let earlier = program.add_node(crate::BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: vec![measurement(0, 0)],
            operators: Vec::new(),
        }));
        let child = program.add_node(crate::BloqNode::classical(ClassicalNode::Observable {
            index: Some(1),
            measurements: vec![measurement(0, 1)],
            operators: Vec::new(),
        }));
        program.add_edge(
            earlier,
            child,
            crate::BloqEdge::Value {
                slot: 0,
                role: crate::ValueRole::ReadoutFold,
                output: ObservableOutput::Corrected,
            },
        );
        let parent = program.add_node(crate::BloqNode::classical(ClassicalNode::observable(2)));
        program.add_edge(
            child,
            parent,
            crate::BloqEdge::Compose {
                slot: 0,
                role: crate::ValueRole::Data,
            },
        );
        let assignment = ClassicalAssignment::Uniform(false);
        let resolution = program.resolve_classical(parent, assignment).unwrap();
        assert_eq!(
            resolution.measurements,
            BTreeSet::from([measurement(0, 0), measurement(0, 1)])
        );
        assert_eq!(resolution.decoder_observables, BTreeSet::from([0, 2]));
        let flip = ValueRef {
            node: parent,
            output: ObservableOutput::Flip,
        };
        let resolution = program.resolve_value(flip, assignment).unwrap();
        assert!(resolution.measurements.is_empty());
        assert_eq!(resolution.decoder_observables, BTreeSet::from([2]));
        assert_eq!(
            program
                .measurement_value_dependencies(flip, assignment)
                .unwrap(),
            BTreeSet::from([measurement(0, 0), measurement(0, 1)])
        );
        let forced_observables = BTreeMap::from([(2, true)]);
        let pinned = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };
        assert_eq!(program.classical_value(parent, pinned), Ok(true));
        assert_eq!(
            program.classical_ref_value(flip, pinned),
            Err(ResolveError::ConditionUnresolved { node: parent })
        );
    }

    #[test]
    fn composition_preserves_the_regions_selected_result_port() {
        for output in [ObservableOutput::Corrected, ObservableOutput::Flip] {
            let mut program = Bloq::new();
            let mut body = SubGraph::new();
            let child = body.add_node(crate::BloqNode::classical(ClassicalNode::Observable {
                index: Some(0),
                measurements: vec![measurement(0, 0)],
                operators: Vec::new(),
            }));
            body.set_value_output(Some(ValueRef {
                node: child,
                output,
            }));
            let region =
                program.add_node(crate::BloqNode::region(RegionNode::RepeatUntilSuccess {
                    restart_source: None,
                    restart_condition: ClassicalExpr::Const(false),
                    body,
                }));
            let parent = program.add_node(crate::BloqNode::classical(ClassicalNode::observable(1)));
            program.add_edge(region, parent, crate::BloqEdge::compose(0));
            let assignment = ClassicalAssignment::Uniform(false);
            let value = program.resolve_classical(parent, assignment).unwrap();
            assert_eq!(
                value.measurements,
                if output == ObservableOutput::Flip {
                    BTreeSet::new()
                } else {
                    BTreeSet::from([measurement(0, 0)])
                }
            );
            assert_eq!(value.decoder_observables, BTreeSet::from([0, 1]));
            assert_eq!(
                program
                    .measurement_dependencies(parent, assignment)
                    .unwrap(),
                BTreeSet::from([measurement(0, 0)])
            );
        }
    }

    #[test]
    fn inactive_observable_has_no_prediction_or_flip() {
        let mut program = Bloq::new();
        let guard = program.add_node(crate::BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let mut observable = crate::BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: vec![measurement(0, 0)],
            operators: Vec::new(),
        });
        observable.activation = Some(0);
        let node = program.add_node(observable);
        program.add_edge(guard, node, crate::BloqEdge::value(0));
        for output in [ObservableOutput::Corrected, ObservableOutput::Flip] {
            let value = ValueRef { node, output };
            let resolution = program
                .resolve_value(value, ClassicalAssignment::Uniform(true))
                .unwrap();
            assert!(resolution.measurements.is_empty());
            assert!(resolution.decoder_observables.is_empty());
            assert_eq!(
                program.classical_ref_value(value, ClassicalAssignment::Uniform(true)),
                Ok(false)
            );
        }
    }

    /// The XOR fold cancels: the frame and the retry body share `i0:m0`, so
    /// resolving their XOR sees neither. A resolver that deduplicated visited
    /// nodes instead of re-folding them would keep the record and be wrong.
    #[test]
    fn shared_producers_cancel_under_xor() {
        let forced_observables = BTreeMap::from([(0, true)]);

        let resolution = program()
            .resolve_classical(
                BloqNodeId(7),
                ClassicalAssignment::Pinned {
                    forced_observables: &forced_observables,
                },
            )
            .expect("the xor resolves");

        assert_eq!(resolution.measurements, BTreeSet::from([measurement(0, 1)]));
        assert!(resolution.sign);
    }

    /// Forced observables also determine downstream conditions and frames.
    #[test]
    fn forced_observables_determine_downstream_conditions_and_frames() {
        let program = program();

        for decoded in [false, true] {
            let forced_observables = BTreeMap::from([(0, decoded)]);
            let assignment = ClassicalAssignment::Pinned {
                forced_observables: &forced_observables,
            };

            assert_eq!(
                program.classical_value(BloqNodeId(4), assignment),
                Ok(decoded)
            );
            assert_eq!(
                program.classical_value(BloqNodeId(5), assignment),
                Ok(!decoded),
                "the frame compute negates what it reads"
            );
        }
    }

    #[test]
    fn a_value_the_path_does_not_fix_is_an_error_not_a_guess() {
        let program = program();
        let forced_observables = BTreeMap::new();
        let assignment = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };

        // No value for observable 0, so nothing downstream of the observable is
        // fixed.
        assert_eq!(
            program.classical_value(BloqNodeId(4), assignment),
            Err(ResolveError::ConditionUnresolved {
                node: BloqNodeId(3)
            })
        );
        assert_eq!(
            program.classical_value(BloqNodeId(0), assignment),
            Err(ResolveError::NotClassical {
                node: BloqNodeId(0)
            })
        );
        // n7 reads the retry region's exported value, which is a measurement
        // recipe rather than a bit: that is `resolve_classical`'s question.
        let forced_observables = BTreeMap::from([(0, true)]);
        assert_eq!(
            program.classical_value(
                BloqNodeId(7),
                ClassicalAssignment::Pinned {
                    forced_observables: &forced_observables,
                },
            ),
            Err(ResolveError::NotClassical {
                node: BloqNodeId(6)
            })
        );
    }

    #[test]
    fn a_uniform_path_fixes_every_predicate_leaf_to_the_same_bit() {
        let program = program();

        for pin in [false, true] {
            assert_eq!(
                program.classical_value(BloqNodeId(5), ClassicalAssignment::Uniform(pin)),
                Ok(!pin)
            );
            assert_eq!(
                program.classical_value(BloqNodeId(4), ClassicalAssignment::Uniform(pin)),
                Ok(pin)
            );
        }
    }

    #[test]
    fn source_free_retry_region_preserves_its_body_result_without_a_dummy_value() {
        let program = Bloq::from_text(
            "BLOQIR 1
 graph {
   n0 rus 0 {
     body {
       n0 observable 0
       result n0
     }
   }
   result n0
 }",
        )
        .unwrap();
        program.validate().unwrap();
        let pinned = program.pin_membership(&BTreeMap::new()).unwrap();
        pinned.validate().unwrap();
        let BloqNodeKind::Region(RegionNode::RepeatUntilSuccess {
            body,
            restart_source,
            ..
        }) = &pinned[BloqNodeId(0)].kind
        else {
            panic!("retry region retained")
        };
        assert!(restart_source.is_none());
        assert_eq!(body.nodes().count(), 1);
        let assignment = ClassicalAssignment::Uniform(false);
        let resolved = pinned.resolve_classical(BloqNodeId(0), assignment).unwrap();
        assert!(resolved.measurements.is_empty());
        assert_eq!(resolved.decoder_observables, BTreeSet::from([0]));
        assert!(
            pinned
                .measurement_dependencies(BloqNodeId(0), assignment)
                .unwrap()
                .is_empty()
        );
    }

    /// A retry region's predicate is independent of its exported value.
    #[test]
    fn a_restart_predicate_is_not_a_path_constant() {
        let program = Bloq::from_text(
            "\
BLOQIR 1

graph {
  n0 compute 1
  n1 rus in0 source n0 {
    body {
      n0 compute 0
    }
  }
  n0 -> n1 value 0
}
",
        )
        .expect("valid .bloqir text");

        assert_eq!(
            program.classical_value(BloqNodeId(1), ClassicalAssignment::Uniform(false)),
            Err(ResolveError::UnsupportedNode {
                node: BloqNodeId(1)
            })
        );
        assert_eq!(
            program.classical_value(BloqNodeId(0), ClassicalAssignment::Uniform(false)),
            Ok(true),
            "a constant compute folds without consulting the path"
        );
    }

    #[test]
    fn shared_dags_cache_recipes_and_predicate_values() {
        let mut program = program();
        let mut recipe = BloqNodeId(1);
        let mut predicate = program.add_node(crate::BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        for _ in 0..30 {
            for previous in [&mut recipe, &mut predicate] {
                let next = program.add_node(crate::BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Xor(Box::new([
                        ClassicalExpr::In(0),
                        ClassicalExpr::In(1),
                    ])),
                }));
                program.add_edge(*previous, next, crate::BloqEdge::value(0));
                program.add_edge(*previous, next, crate::BloqEdge::value(1));
                *previous = next;
            }
        }
        let resolution = program
            .resolve_classical(recipe, ClassicalAssignment::Uniform(false))
            .unwrap();
        assert!(resolution.measurements.is_empty());
        assert!(!resolution.sign);
        assert_eq!(
            program.classical_value(
                predicate,
                ClassicalAssignment::Pinned {
                    forced_observables: &BTreeMap::new(),
                }
            ),
            Ok(false)
        );
    }

    #[test]
    fn predicate_cycles_return_the_same_typed_error_as_record_cycles() {
        let program = Bloq::from_text(
            "BLOQIR 1
 graph {
   n0 compute in0
   n1 compute in0
   n2 rus 0 when v0 {
     body {
     }
   }
   n0 -> n1 value 0
   n1 -> n0 value 0
   n0 -> n2 value 0
 }
",
        )
        .unwrap();
        let forced_observables = BTreeMap::new();
        let assignment = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };
        let expected = Err(ResolveError::CyclicValueGraph {
            node: BloqNodeId(0),
        });
        assert_eq!(program.classical_value(BloqNodeId(0), assignment), expected);
        for node in [BloqNodeId(0), BloqNodeId(2)] {
            assert_eq!(
                program.resolve_classical(node, assignment).map(|_| false),
                expected
            );
        }
    }

    #[test]
    fn dependency_folds_never_cancel_shared_measurements() {
        let mut program = Bloq::from_text(
            "BLOQIR 1
 template t0 {
   circuit {
     M (0,0):m0
     M (1,0):m1
   }
 }
 graph {
   n0 quantum {
     instance i0 t0 @ (0,0)
   }
   n1 observable fragment measurements i0:m0
   n2 observable fragment measurements i0:m1
   n3 compute (in0 & in1) ^ in0
   n4 observable 0
   n0 -> n1 order
   n0 -> n2 order
   n1 -> n3 value 0
   n2 -> n3 value 1
   n3 -> n4 value 0
   n2 -> n4 compose 1
 }
",
        )
        .unwrap();
        program.validate().unwrap();
        for expr in [
            program[BloqNodeId(3)].try_classical().unwrap().clone(),
            ClassicalNode::Compute {
                expr: ClassicalExpr::select(
                    ClassicalExpr::In(0),
                    ClassicalExpr::In(1),
                    ClassicalExpr::In(0),
                ),
            },
            ClassicalNode::Compute {
                expr: ClassicalExpr::And(Box::new([
                    ClassicalExpr::In(0),
                    ClassicalExpr::In(1),
                    ClassicalExpr::In(0),
                ])),
            },
        ] {
            program.node_mut(BloqNodeId(3)).unwrap().kind = BloqNodeKind::Classical(expr.into());
            program.validate().unwrap();
            for node in [BloqNodeId(3), BloqNodeId(4)] {
                let expected = BTreeSet::from([measurement(0, 0), measurement(0, 1)]);
                assert_eq!(
                    program
                        .measurement_dependencies(node, ClassicalAssignment::Uniform(false))
                        .unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn fixed_path_cofactors_select_and_fully_fixed_boolean_ops() {
        let program = Bloq::from_text(
            "\
BLOQIR 1

template t0 {
  circuit {
    M (0,0):m0
    M (1,0):m1
    M (2,0):m2
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0
  n2 observable 0
  n3 compute in0
  n4 observable fragment measurements i0:m1
  n5 observable 1
  n6 compute in0
  n7 observable fragment measurements i0:m2
  n8 compute select(in0, in1, in2)
  n9 compute in0 & in1
  n10 compute in0 | in1
  n11 compute in0 ^ in1
  n12 compute in0 ^ in1
  n1 -> n2 compose 0
  n2 -> n3 value 0
  n4 -> n5 compose 0
  n5 -> n6 value 0
  n3 -> n8 value 0
  n7 -> n8 value 1
  n6 -> n8 value 2
  n3 -> n9 value 0
  n6 -> n9 value 1
  n3 -> n10 value 0
  n6 -> n10 value 1
  n3 -> n11 value 0
  n3 -> n11 value 1
  n3 -> n12 value 0
  n6 -> n12 value 1
}
",
        )
        .expect("valid .bloqir text");
        let forced_observables = BTreeMap::from([(0, true), (1, false)]);
        let assignment = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };

        let selected = program
            .resolve_classical(BloqNodeId(8), assignment)
            .unwrap();
        assert_eq!(selected.measurements, BTreeSet::from([measurement(0, 1)]));
        assert!(!selected.sign);
        assert_eq!(selected.decoder_observables, BTreeSet::from([1]));
        for (node, value) in [(BloqNodeId(9), false), (BloqNodeId(10), true)] {
            let resolution = program.resolve_classical(node, assignment).unwrap();
            assert!(resolution.measurements.is_empty());
            assert_eq!(resolution.sign, value);
            assert!(resolution.decoder_observables.is_empty());
            assert_eq!(program.classical_value(node, assignment), Ok(value));
        }
        let cancelled = program
            .resolve_classical(BloqNodeId(11), assignment)
            .unwrap();
        assert!(cancelled.measurements.is_empty());
        assert!(cancelled.decoder_observables.is_empty());
        let multiple = program
            .resolve_classical(BloqNodeId(12), assignment)
            .unwrap();
        assert_eq!(
            multiple.measurements,
            BTreeSet::from([measurement(0, 0), measurement(0, 1)])
        );
        assert_eq!(multiple.decoder_observables, BTreeSet::from([0, 1]));

        // Value evaluation remains strict: even the unselected raw-record arm
        // must be available, while a fixed-path parity reads only its selected
        // arm above.
        assert_eq!(
            program.classical_value(BloqNodeId(8), assignment),
            Err(ResolveError::ConditionUnresolved {
                node: BloqNodeId(7)
            })
        );
        assert_eq!(
            program
                .measurement_dependencies(BloqNodeId(8), assignment)
                .unwrap(),
            BTreeSet::from([measurement(0, 0), measurement(0, 1), measurement(0, 2),])
        );
        for node in [BloqNodeId(9), BloqNodeId(10)] {
            assert_eq!(
                program.measurement_dependencies(node, assignment).unwrap(),
                BTreeSet::from([measurement(0, 0), measurement(0, 1)])
            );
        }

        // Boolean evaluation stays strict instead of short-circuiting around
        // an unavailable operand.
        for (node, fixed) in [(BloqNodeId(9), false), (BloqNodeId(10), true)] {
            let forced_observables = BTreeMap::from([(0, fixed)]);
            let partial = ClassicalAssignment::Pinned {
                forced_observables: &forced_observables,
            };
            assert_eq!(
                program.resolve_classical(node, partial),
                Err(ResolveError::NonLinearExpr { node })
            );
            assert_eq!(
                program.classical_value(node, partial),
                Err(ResolveError::ConditionUnresolved {
                    node: BloqNodeId(5)
                })
            );
        }

        let forced_observables = BTreeMap::new();
        let unresolved = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };
        for node in [BloqNodeId(8), BloqNodeId(9), BloqNodeId(10)] {
            assert_eq!(
                program.resolve_classical(node, unresolved),
                Err(ResolveError::NonLinearExpr { node })
            );
        }
    }

    #[test]
    fn an_unfixed_non_linear_expression_has_no_measurement_recipe() {
        let program = Bloq::from_text(
            "\
BLOQIR 1

template t0 {
  circuit {
    MPP X(0,0):m0
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0
  n2 compute 1
  n3 compute in0 & in1
  n1 -> n3 value 0
  n2 -> n3 value 1
}
",
        )
        .expect("valid .bloqir text");

        let forced_observables = BTreeMap::new();
        let assignment = ClassicalAssignment::Pinned {
            forced_observables: &forced_observables,
        };
        assert_eq!(
            program.resolve_classical(BloqNodeId(3), assignment),
            Err(ResolveError::NonLinearExpr {
                node: BloqNodeId(3)
            })
        );
        assert_eq!(
            program.classical_value(BloqNodeId(3), assignment),
            Err(ResolveError::ConditionUnresolved {
                node: BloqNodeId(1)
            })
        );
        assert_eq!(
            program
                .measurement_dependencies(BloqNodeId(3), assignment)
                .unwrap(),
            BTreeSet::from([measurement(0, 0)])
        );
    }
}
