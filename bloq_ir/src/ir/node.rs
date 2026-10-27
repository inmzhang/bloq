use std::fmt;
use std::sync::Arc;

use bloq_circuit::{Basis, PauliMap};
use glam::IVec3;

use super::{
    DetectorBundleUse, InstanceMeasurement, NodeDetector, NodeRestart, SourceBlockRef, SubGraph,
    TemplateInstance, TemplateInstanceId, TemporalPipeRef,
};

/// What source-level construct an IR node realizes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NodeProvenance {
    /// A quantum node realizing one connected component of source blocks (its
    /// `members`), fused because they share physical qubits on a layer.
    BlockComponent {
        /// Source blocks in the component.
        members: Vec<SourceBlockRef>,
    },
    /// A quantum node realizing one temporal pipe: the seam layer between two
    /// block layers. Normally a temporal pipe will not compile to real quantum
    /// circuits, the currently only use case if for temporal hadamard pipe
    /// which includes a realignment layer.
    TemporalPipe {
        /// Realized temporal pipe.
        pipe: TemporalPipeRef,
    },
    /// Compiler-only replacement of a source spatial Port. The virtual
    /// temporal-Port node has no lattice cell of its own; `role` places it just
    /// before or after `source` on the doubled-z schedule.
    SpatialPortSubstitution {
        /// Source port position.
        source: IVec3,
        /// Source port role.
        role: bloq_utils::PortRole,
    },
    /// Memory-round padding inserted at a `Quantum` seam or region terminal
    /// post-compile ([`crate::Bloq::insert_memory_rounds`]). Like
    /// [`Self::TemporalPipe`] it lives on the pipe's layer, but it carries the
    /// wait duration so consumers can find padding nodes and read
    /// their round count without inspecting the template's repeat structure.
    MemoryPadding {
        /// Padded pipe.
        pipe: TemporalPipeRef,
        /// Inserted memory rounds.
        rounds: u32,
    },
    /// A classical node realizing stabilizer generator `ordinal`'s readout
    /// recipe: its complete `Observable`, any shared fragments, and the
    /// corrected-value `Compute`s derived from them.
    Generator {
        /// Source generator ordinal.
        ordinal: u32,
    },
    /// A classical node lowered from the source action at `ordinal` (program
    /// action order): a named readout's complete `Observable`, a
    /// condition/selector `Compute`, or a `Discard` sink.
    Action {
        /// Source action ordinal.
        ordinal: u32,
    },
    /// A source branch's resolved selector, shared by every assembly stage.
    BranchSelector {
        /// Source selector name.
        name: String,
    },
    /// A frame-sign `Compute`: the `basis`-frame correction bit for source
    /// output port `port`. The stamp **is** the frame
    /// table: [`crate::Bloq::output_frames`] derives its [`crate::FramePair`]s from
    /// these stamps, so there is no side table to drift. Validation enforces
    /// that a stamp sits on a top-level `Compute` and that each output carries
    /// exactly one X and one Z stamp.
    // Spec rule WF-15.
    OutputFrame {
        /// Source output port.
        port: IVec3,
        /// Corrected Pauli basis.
        basis: Basis,
    },
    /// No provenance: hand-built or synthetic (e.g. a constant producer).
    None,
}

/// Human-readable one-line description, e.g. `blocks (0,0,0) (1,0,0)` or
/// `pipe (0,0,0)>(0,0,1)`. Coordinate/pipe spelling matches the `.bloq` text
/// format so the two read alike; consumers are the program view and the Stim
/// backend's per-node provenance comments.
impl fmt::Display for NodeProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let coord3 = |f: &mut fmt::Formatter<'_>, p: IVec3| write!(f, "({},{},{})", p.x, p.y, p.z);
        match self {
            NodeProvenance::None => write!(f, "synthetic"),
            NodeProvenance::BlockComponent { members } => {
                write!(f, "blocks")?;
                for member in members {
                    write!(f, " ")?;
                    coord3(f, member.pos)?;
                }
                Ok(())
            }
            NodeProvenance::TemporalPipe { pipe } => {
                write!(f, "pipe ")?;
                coord3(f, pipe.src)?;
                write!(f, ">")?;
                coord3(f, pipe.dst)?;
                if pipe.hadamard {
                    write!(f, " H")?;
                }
                Ok(())
            }
            NodeProvenance::SpatialPortSubstitution { source, role } => {
                write!(f, "spatial-port {} ", role.as_str())?;
                coord3(f, *source)
            }
            NodeProvenance::MemoryPadding { pipe, rounds } => {
                write!(f, "padding ")?;
                coord3(f, pipe.src)?;
                write!(f, ">")?;
                coord3(f, pipe.dst)?;
                write!(f, " rounds {rounds}")
            }
            NodeProvenance::Generator { ordinal } => write!(f, "generator {ordinal}"),
            NodeProvenance::Action { ordinal } => write!(f, "action {ordinal}"),
            NodeProvenance::BranchSelector { name } => write!(f, "selector {name}"),
            NodeProvenance::OutputFrame { port, basis } => {
                let basis = match basis {
                    Basis::X => "x",
                    Basis::Z => "z",
                };
                write!(f, "frame {basis} ")?;
                coord3(f, *port)
            }
        }
    }
}

/// A graph node, tagged by what kind of work it represents.
///
/// `kind` is what the node computes; `provenance` is why it exists — its tie
/// back to the source block graph, a stabilizer generator, a source action, or
/// an output frame ([`NodeProvenance`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BloqNode {
    /// Node payload.
    pub kind: BloqNodeKind,
    /// Source provenance.
    pub provenance: NodeProvenance,
    /// Optional activation `Value` slot for a classical or region node. When false,
    /// bit results are zero, boundary bindings are empty, and no read, decoder
    /// request, region-body execution, or discard occurs. It is not a data operand.
    pub activation: Option<u32>,
}

/// What a [`BloqNode`] computes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BloqNodeKind {
    /// Shared until quantum data is edited. Large physical payloads do not
    /// inflate classical node slots or copy during unrelated graph edits.
    Quantum(#[serde(deserialize_with = "deserialize_quantum")] Arc<QuantumNode>),
    /// Shared immutable classical definition. Activation, inputs and provenance
    /// belong to this node's invocation; editing detaches only its definition.
    Classical(Arc<ClassicalNode>),
    /// Structured control-flow region.
    Region(RegionNode),
}

impl BloqNodeKind {
    /// Borrow a classical definition without changing its sharing.
    pub fn try_classical(&self) -> Option<&ClassicalNode> {
        match self {
            Self::Classical(data) => Some(data),
            Self::Quantum(_) | Self::Region(_) => None,
        }
    }

    /// Detach a shared classical definition before editing it.
    pub fn try_classical_mut(&mut self) -> Option<&mut ClassicalNode> {
        match self {
            Self::Classical(data) => Some(Arc::make_mut(data)),
            Self::Quantum(_) | Self::Region(_) => None,
        }
    }
}

pub(super) fn deserialize_quantum<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Arc<QuantumNode>, D::Error> {
    // Serde's generic Arc decoder goes through Box<T>. Allocate the final
    // shared payload directly instead of boxing and reallocating each node.
    <QuantumNode as serde::Deserialize<'de>>::deserialize(deserializer).map(Arc::new)
}

/// Classical logic and observable recipes. Composition edges include a child's
/// measurements and boundary bindings; value edges select a Boolean output.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClassicalNode {
    /// Combinational logic over incoming value slots.
    Compute {
        /// Boolean expression.
        expr: ClassicalExpr,
    },
    /// Observable recipe. Indexed recipes expose corrected values and
    /// predicted decoder flips. Unindexed fragments expose their assembled parity and
    /// compose without an independent decoder request.
    Observable {
        /// Graph-local decoder query identity; absent for recipe fragments.
        index: Option<u32>,
        /// Local records, preserving order and occurrences.
        #[serde(with = "crate::binary::instance_measurements")]
        measurements: Vec<InstanceMeasurement>,
        /// Boundary bindings with physical ownership.
        operators: Vec<InstanceBoundaryOperator>,
    },
    /// Reject the shot when the condition holds.
    Discard {
        /// Rejection condition.
        condition: ClassicalExpr,
    },
}

impl ClassicalNode {
    /// A complete observable with an empty recipe.
    pub fn observable(index: u32) -> Self {
        Self::Observable {
            index: Some(index),
            measurements: Vec::new(),
            operators: Vec::new(),
        }
    }

    /// A composable recipe without its own decoder query.
    pub fn observable_fragment(
        measurements: Vec<InstanceMeasurement>,
        operators: Vec<InstanceBoundaryOperator>,
    ) -> Self {
        Self::Observable {
            index: None,
            measurements,
            operators,
        }
    }

    /// An empty composable recipe.
    pub fn fragment() -> Self {
        Self::observable_fragment(Vec::new(), Vec::new())
    }

    /// Record terms owned directly by this producer.
    pub fn measurements(&self) -> &[InstanceMeasurement] {
        match self {
            Self::Observable { measurements, .. } => measurements,
            _ => &[],
        }
    }

    /// Boundary bindings owned directly by this producer.
    pub fn operators(&self) -> &[InstanceBoundaryOperator] {
        match self {
            Self::Observable { operators, .. } => operators,
            _ => &[],
        }
    }
}

/// Combinational logic over a classical node's incoming `Value` edges.
///
/// [`ClassicalExpr::In`] reads the edge whose `slot` equals its index
/// (SSA-by-slot): the slot is carried on the edge, so the binding survives edge
/// reordering under edits rather than depending on iteration order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClassicalExpr {
    /// Read one incoming edge slot.
    In(u32),
    /// A compile-time-constant bit — an input-less producer. Carries a
    /// deterministic feedback sign (constant-sign folds to no op).
    Const(bool),
    /// Boolean negation.
    Not(Box<ClassicalExpr>),
    /// Ordered parity operands; the empty parity is false.
    Xor(Box<[ClassicalExpr]>),
    /// Ordered conjunction operands; the empty conjunction is true.
    And(Box<[ClassicalExpr]>),
    /// Ordered disjunction operands; the empty disjunction is false.
    Or(Box<[ClassicalExpr]>),
    /// `[condition, when_false, when_true]`. All three operands must be known,
    /// including the unselected value; this is combinational dataflow (SEM-EXPR).
    Select(Box<[ClassicalExpr; 3]>),
    /// Affine parity of ordered input slots and a constant XOR term.
    /// Repeated slots stay present: every input must be known even when its
    /// Boolean contributions cancel (SEM-EXPR). No expression nodes per slot.
    Parity {
        /// Ordered input slots.
        inputs: Box<[u32]>,
        /// Constant XOR term.
        constant: bool,
    },
}

impl ClassicalExpr {
    /// Build an affine parity, retaining input order and multiplicity.
    #[must_use]
    pub fn parity(inputs: impl IntoIterator<Item = u32>, constant: bool) -> Self {
        let inputs: Box<[_]> = inputs.into_iter().collect();
        match inputs.as_ref() {
            [] => Self::Const(constant),
            [slot] if !constant => Self::In(*slot),
            _ => Self::Parity { inputs, constant },
        }
    }

    /// Build a parity list, collapsing empty and singleton lists.
    /// Preserves operand order, duplicates, and nested expressions.
    ///
    /// # Panics
    ///
    /// Panics only if the collected operand count changes before extraction.
    #[must_use]
    pub fn xor(operands: impl IntoIterator<Item = Self>) -> Self {
        let mut operands: Vec<_> = operands.into_iter().collect();
        match operands.len() {
            0 => Self::Const(false),
            1 => operands
                .pop()
                .expect("the single operand was counted above"),
            _ => Self::Xor(operands.into_boxed_slice()),
        }
    }

    /// Select a value with explicit false/true operand order (SEM-EXPR).
    /// Literal branches use smaller operators without dropping unknown inputs.
    #[must_use]
    pub fn select(condition: Self, when_false: Self, when_true: Self) -> Self {
        match (when_false, when_true) {
            (Self::Const(false), Self::Const(true)) => condition,
            (Self::Const(true), Self::Const(false)) => Self::Not(Box::new(condition)),
            (Self::Const(false), when_true) => Self::And(Box::new([condition, when_true])),
            (when_false, Self::Const(true)) => Self::Or(Box::new([condition, when_false])),
            // A negated gate needs two boxes; its selection needs only one.
            (when_false, when_true) => Self::Select(Box::new([condition, when_false, when_true])),
        }
    }

    /// Direct expression children in evaluation order. Compact [`Self::Parity`]
    /// stores slots directly; use [`Self::for_each_input`] to visit all inputs.
    pub fn operands(&self) -> &[Self] {
        match self {
            Self::In(_) | Self::Const(_) | Self::Parity { .. } => &[],
            Self::Not(inner) => std::slice::from_ref(inner),
            Self::Xor(operands) | Self::And(operands) | Self::Or(operands) => operands,
            Self::Select(operands) => operands.as_ref(),
        }
    }

    /// Fold the expression over three-valued inputs: `None` (an input that is
    /// unevaluable in the caller's context) propagates. The reference
    /// evaluator every consumer shares.
    pub fn eval(&self, input: &mut impl FnMut(u32) -> Option<bool>) -> Option<bool> {
        match self {
            Self::In(slot) => input(*slot),
            Self::Const(bit) => Some(*bit),
            Self::Parity { inputs, constant } => inputs
                .iter()
                .try_fold(*constant, |value, &slot| Some(value ^ input(slot)?)),
            Self::Not(inner) => inner.eval(input).map(|bit| !bit),
            Self::Xor(operands) => operands
                .iter()
                .try_fold(false, |value, expr| Some(value ^ expr.eval(input)?)),
            Self::And(operands) => operands
                .iter()
                .try_fold(true, |value, expr| Some(value & expr.eval(input)?)),
            Self::Or(operands) => operands
                .iter()
                .try_fold(false, |value, expr| Some(value | expr.eval(input)?)),
            Self::Select(operands) => {
                let [condition, when_false, when_true] = operands.as_ref();
                let condition = condition.eval(input)?;
                let when_false = when_false.eval(input)?;
                let when_true = when_true.eval(input)?;
                Some(if condition { when_true } else { when_false })
            }
        }
    }

    /// Combine evaluated operands under Kleene logic: unlike [`Self::eval`], a
    /// known `false` decides `And` and a known `true` decides `Or` beside
    /// unknown operands, and `Select` with an unknown condition is known when
    /// both arms agree. `operands` holds the input values for `In` and
    /// `Parity`, and the operand values otherwise.
    #[must_use]
    pub fn combine_kleene(&self, operands: &[Option<bool>]) -> Option<bool> {
        match self {
            Self::In(_) => operands[0],
            Self::Const(bit) => Some(*bit),
            Self::Not(_) => operands[0].map(|bit| !bit),
            Self::And(_) => {
                if operands.contains(&Some(false)) {
                    Some(false)
                } else {
                    (!operands.contains(&None)).then_some(true)
                }
            }
            Self::Or(_) => {
                if operands.contains(&Some(true)) {
                    Some(true)
                } else {
                    (!operands.contains(&None)).then_some(false)
                }
            }
            Self::Xor(_) => operands
                .iter()
                .try_fold(false, |value, &bit| Some(value ^ bit?)),
            Self::Parity { constant, .. } => operands
                .iter()
                .try_fold(*constant, |value, &bit| Some(value ^ bit?)),
            Self::Select(_) => match operands[0] {
                Some(condition) => operands[1 + usize::from(condition)],
                None if operands[1] == operands[2] => operands[1],
                None => None,
            },
        }
    }

    /// Visit every `In(slot)` leaf, left to right.
    pub fn for_each_input(&self, f: &mut impl FnMut(u32)) {
        match self {
            Self::In(slot) => f(*slot),
            Self::Parity { inputs, .. } => inputs.iter().for_each(|&slot| f(slot)),
            _ => {}
        }
        for operand in self.operands() {
            operand.for_each_input(f);
        }
    }

    /// Replace original leaves in order, moving replacement subtrees into
    /// place: slot remapping (`f` returns another `In`) and producer inlining
    /// (`f` returns a subtree) in one operation.
    ///
    /// Newly inserted leaves are not visited again.
    pub(crate) fn replace_inputs(&mut self, f: &mut impl FnMut(u32) -> ClassicalExpr) {
        match self {
            Self::In(slot) => *self = f(*slot),
            Self::Const(_) => {}
            Self::Parity { inputs, constant } => {
                let mut operands = Vec::with_capacity(inputs.len());
                for &slot in inputs.iter() {
                    operands.push(f(slot));
                }
                if operands
                    .iter()
                    .all(|operand| matches!(operand, Self::In(_)))
                {
                    for (slot, operand) in inputs.iter_mut().zip(operands) {
                        let Self::In(mapped) = operand else {
                            unreachable!("all replacement operands were checked as inputs")
                        };
                        *slot = mapped;
                    }
                } else {
                    let expr = Self::xor(operands);
                    *self = if *constant {
                        Self::Not(Box::new(expr))
                    } else {
                        expr
                    };
                }
            }
            Self::Not(inner) => inner.replace_inputs(f),
            Self::Xor(operands) | Self::And(operands) | Self::Or(operands) => {
                for operand in operands {
                    operand.replace_inputs(f);
                }
            }
            Self::Select(operands) => {
                for operand in operands.as_mut() {
                    operand.replace_inputs(f);
                }
            }
        }
    }

    /// Whether the expression is XOR-expressible (`In`/`Const`/`Not`/`Xor`/`Parity`
    /// only). A static backend folds a linear condition into
    /// `OBSERVABLE_INCLUDE`; a nonlinear (`And`/`Or`/`Select`) one needs a
    /// dynamic backend.
    pub fn is_linear(&self) -> bool {
        match self {
            Self::In(_) | Self::Const(_) | Self::Parity { .. } => true,
            Self::Not(inner) => inner.is_linear(),
            Self::Xor(operands) => operands.iter().all(Self::is_linear),
            Self::And(..) | Self::Or(..) | Self::Select(..) => false,
        }
    }

    /// Compact affine subexpressions after graph rewrites. Never cancel input
    /// slots: Boolean equality alone does not preserve unavailable inputs.
    pub(crate) fn needs_affine_compaction(&self) -> bool {
        if matches!(self, Self::Not(_) | Self::Xor(_)) && self.is_linear() {
            return true;
        }
        self.operands().iter().any(Self::needs_affine_compaction)
    }

    /// Compact affine subexpressions while preserving ordered input occurrences.
    pub(crate) fn compact_affine(&mut self) {
        if matches!(self, Self::In(_) | Self::Const(_) | Self::Parity { .. }) {
            return;
        }
        if self.is_linear() {
            let constant = self
                .eval(&mut |_| Some(false))
                .expect("constant assignment");
            let mut inputs = Vec::new();
            self.for_each_input(&mut |slot| inputs.push(slot));
            *self = Self::parity(inputs, constant);
            return;
        }
        match self {
            Self::Not(inner) => inner.compact_affine(),
            Self::Xor(operands) | Self::And(operands) | Self::Or(operands) => {
                for operand in operands {
                    operand.compact_affine();
                }
            }
            Self::Select(operands) => {
                for operand in operands.as_mut() {
                    operand.compact_affine();
                }
            }
            _ => unreachable!("leaves returned above"),
        }
    }
}

/// Which temporal face of an instance a boundary operator binds.
///
/// The face fixes emission position: an `Input` (−Z) operator reads the qubit
/// state entering the instance, so it emits *before* the instance circuit; an
/// `Output` (+Z) operator reads on exit, so it emits *after*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BoundaryFace {
    /// State entering the instance.
    Input,
    /// State leaving the instance.
    Output,
}

/// A symbolic logical operator at one template-instance boundary.
///
/// This is one binding payload element of [`ClassicalNode::Observable`]. It names the
/// operator by instance reference (never per-shot bits), in instance-global
/// qubit coordinates — a translated [`PauliMap`] at one temporal `face`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct InstanceBoundaryOperator {
    /// Owning template instance.
    pub instance: TemplateInstanceId,
    /// Bound temporal face.
    pub face: BoundaryFace,
    /// Logical Pauli operator.
    pub operator: PauliMap,
}

/// A structured control-flow region.
///
/// Its interface is the body's declared bit result and boundary bindings. Its predicate arrives on a
/// `Value` edge; its body is a [`SubGraph`] over the same global template pool.
#[derive(Debug, Clone)]
pub enum RegionNode {
    /// Region carrying a postselection/restart predicate. Execution retries a
    /// sandboxed body until its restart predicate is false, subject to a
    /// configured execution cap. Decoder results affect retries only through
    /// the explicit predicate; physical restart parities can abort an attempt early.
    ///
    /// `restart_condition` reads incoming `Value` slots; unfed slots read the
    /// optional body-local `restart_source`. Constant predicates need no source.
    /// The body's declared result remains independent of the restart predicate.
    RepeatUntilSuccess {
        /// Retried body.
        body: SubGraph,
        /// Restart predicate.
        restart_condition: ClassicalExpr,
        /// Optional body-local value feeding the predicate's unfed slots.
        /// Absent when every predicate input is supplied by incoming edges;
        /// constant predicates need no source.
        // Spec rule WF-10.
        restart_source: Option<super::ValueRef>,
    },
}

// Keep the existing RUS binary discriminant (1). The removed variant (0)
// must be rejected rather than interpreted as a retry region.
#[derive(serde::Serialize, serde::Deserialize)]
enum RegionWire<B, E> {
    Removed,
    RepeatUntilSuccess {
        body: B,
        restart_condition: E,
        restart_source: Option<super::ValueRef>,
    },
}

impl serde::Serialize for RegionNode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self::RepeatUntilSuccess {
            body,
            restart_condition,
            restart_source,
        } = self;
        RegionWire::RepeatUntilSuccess {
            body,
            restart_condition,
            restart_source: *restart_source,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for RegionNode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match RegionWire::deserialize(deserializer)? {
            RegionWire::Removed => Err(serde::de::Error::custom("removed region variant")),
            RegionWire::RepeatUntilSuccess {
                body,
                restart_condition,
                restart_source,
            } => Ok(Self::RepeatUntilSuccess {
                body,
                restart_condition,
                restart_source,
            }),
        }
    }
}

/// The body of a [`RegionNode`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BodySelector {
    /// The region's sole body.
    Body,
}

impl RegionNode {
    /// The region's body subgraph.
    pub fn bodies(&self) -> impl Iterator<Item = (BodySelector, &SubGraph)> {
        let Self::RepeatUntilSuccess { body, .. } = self;
        std::iter::once((BodySelector::Body, body))
    }

    /// The mutable mirror of [`Self::bodies`].
    pub fn bodies_mut(&mut self) -> impl Iterator<Item = (BodySelector, &mut SubGraph)> {
        let Self::RepeatUntilSuccess { body, .. } = self;
        std::iter::once((BodySelector::Body, body))
    }

    /// The region variant's display name.
    pub fn kind_name(&self) -> &'static str {
        "RepeatUntilSuccess"
    }
}

/// Cumulative round cuts projecting a multi-layer block onto source z layers.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuantumTimeline {
    /// Cumulative logical-round ends, one entry per occupied source z layer.
    /// The first entry belongs to the node's lowest source layer, derived from
    /// [`BloqNode::layer`]. Equal adjacent entries represent a layer containing
    /// no circuit rounds.
    pub layer_round_ends: Vec<u32>,
}

/// A template-backed quantum node: its ops come from instantiating the
/// referenced templates from the pool, not from any per-node circuit.
///
/// Instances must have merge-compatible template entry bodies: aligned
/// tick/repeat segments, matching repeat counts, and no same-segment qubit
/// overlap except duplicate measurements or duplicate same-basis resets.
/// [`Bloq::validate`](crate::Bloq::validate) enforces this for debug
/// compilation, so instantiation and backends may assume it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct QuantumNode {
    /// Placed circuit templates.
    pub instances: Vec<TemplateInstance>,
    /// Cross-template detectors.
    pub detectors: Vec<NodeDetector>,
    /// Shared cross-template detector rows bound to this node's instances.
    pub detector_bundles: Vec<DetectorBundleUse>,
    /// Cross-instance restart syndromes; see [`NodeRestart`].
    pub restarts: Vec<NodeRestart>,
    /// Source-layer projection for a multi-z block component. Single-layer and
    /// synthetic nodes need no timeline.
    pub timeline: Option<QuantumTimeline>,
    /// Conditional registrations in this component. Unlisted members and side
    /// tables are common to every selection. All inputs must be ready before
    /// the component starts; registration itself performs no quantum work.
    pub guards: Vec<QuantumGuard>,
}

/// Members and side tables enabled by one quantum node's `Value` input.
///
/// Each instance/detector/restart may occur in at most one registration; a
/// compound condition is an ordinary `Compute` producer feeding `input`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuantumGuard {
    /// Incoming selector slot.
    pub input: u32,
    /// Conditionally enabled instances.
    pub instances: Vec<TemplateInstanceId>,
    /// Conditionally enabled detector indices.
    pub detectors: Vec<u32>,
    /// Conditionally enabled detector bundle use indices.
    pub detector_bundles: Vec<u32>,
    /// Conditionally enabled restart indices.
    pub restarts: Vec<u32>,
    /// Parity contributions XORed into selected detector rows by this input.
    pub detector_parities: Vec<(u32, crate::NodeDetectorParity)>,
    /// Parity contributions XORed into selected restart rows by this input.
    pub restart_parities: Vec<(u32, crate::NodeDetectorParity)>,
}

impl QuantumNode {
    /// Every parity stored on this node: the inline detector rows, then the
    /// restart rows, then each guard's side-table contributions.
    ///
    /// Pooled [`DetectorBundleUse`] rows are deliberately *not* included — they
    /// live in the program's bundle pool, so a caller that needs whole-node
    /// detector coverage walks [`Self::detector_bundles`] as well.
    pub(crate) fn stored_parities(&self) -> impl Iterator<Item = &crate::NodeDetectorParity> {
        self.detectors
            .iter()
            .map(|detector| &detector.parity)
            .chain(self.restarts.iter().map(|restart| &restart.parity))
            .chain(self.guards.iter().flat_map(|guard| {
                guard
                    .detector_parities
                    .iter()
                    .chain(&guard.restart_parities)
                    .map(|(_, parity)| parity)
            }))
    }
}

impl BloqNode {
    pub(crate) fn kind_name(&self) -> &'static str {
        match &self.kind {
            BloqNodeKind::Quantum(_) => "Quantum",
            BloqNodeKind::Region(region) => region.kind_name(),
            BloqNodeKind::Classical(classical) => match classical.as_ref() {
                ClassicalNode::Compute { .. } => "Compute",
                ClassicalNode::Observable { .. } => "Observable",
                ClassicalNode::Discard { .. } => "Discard",
            },
        }
    }

    /// Create a quantum node for a source block component.
    pub fn from_members(members: Vec<SourceBlockRef>) -> Self {
        Self::new_quantum(NodeProvenance::BlockComponent { members })
    }

    /// Create a quantum node for a temporal pipe.
    pub fn from_temporal_pipe(pipe: TemporalPipeRef) -> Self {
        Self::new_quantum(NodeProvenance::TemporalPipe { pipe })
    }

    /// A memory-round padding node on `pipe`'s seam, waiting `rounds`
    /// syndrome-extraction rounds.
    pub fn memory_padding(pipe: TemporalPipeRef, rounds: u32) -> Self {
        Self::new_quantum(NodeProvenance::MemoryPadding { pipe, rounds })
    }

    fn new_quantum(provenance: NodeProvenance) -> Self {
        Self::quantum(QuantumNode::default()).with_provenance(provenance)
    }

    /// A template-backed quantum node with no provenance. Its payload shares
    /// across snapshots until [`Self::expect_quantum_mut`] edits it.
    pub fn quantum(quantum: QuantumNode) -> Self {
        Self {
            kind: BloqNodeKind::Quantum(quantum.into()),
            provenance: NodeProvenance::None,
            activation: None,
        }
    }

    /// A control-flow region node ([`RegionNode`]), with no provenance.
    pub fn region(region: RegionNode) -> Self {
        Self {
            kind: BloqNodeKind::Region(region),
            provenance: NodeProvenance::None,
            activation: None,
        }
    }

    /// A classical-dataflow node ([`ClassicalNode`]), with no provenance.
    /// Lowering stamps the real provenance ([`NodeProvenance::Generator`] /
    /// [`NodeProvenance::Action`] / [`NodeProvenance::OutputFrame`]) via
    /// [`Self::with_provenance`] where it knows the origin.
    pub fn classical(node: impl Into<Arc<ClassicalNode>>) -> Self {
        Self {
            kind: BloqNodeKind::Classical(node.into()),
            provenance: NodeProvenance::None,
            activation: None,
        }
    }

    /// The same node with its provenance replaced — builder-style, for lowering
    /// sites that know what a classical node realizes.
    pub fn with_provenance(mut self, provenance: NodeProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// The node's quantum payload, **panicking** on a non-quantum node (the
    /// rustc `expect_*` convention for panicking accessors). Most passes only
    /// ever see quantum nodes (lowering produces no others on the static
    /// path), so they call this directly; a pass that must tolerate other
    /// kinds uses [`Self::try_quantum`] instead.
    ///
    /// # Panics
    ///
    /// Panics when this node is classical or a region.
    pub fn expect_quantum(&self) -> &QuantumNode {
        self.try_quantum()
            .expect("expect_quantum() called on a non-quantum BloqNode")
    }

    /// Mutable twin of [`Self::expect_quantum`]; detaches shared quantum data.
    /// Panics on a non-quantum node.
    ///
    /// # Panics
    ///
    /// Panics when this node is classical or a region.
    pub fn expect_quantum_mut(&mut self) -> &mut QuantumNode {
        match &mut self.kind {
            BloqNodeKind::Quantum(quantum) => Arc::make_mut(quantum),
            BloqNodeKind::Classical(_) | BloqNodeKind::Region(_) => {
                panic!("expect_quantum_mut() called on a non-quantum BloqNode")
            }
        }
    }

    /// The node's quantum payload, or `None` for a non-quantum (classical or
    /// region) node.
    pub fn try_quantum(&self) -> Option<&QuantumNode> {
        match &self.kind {
            BloqNodeKind::Quantum(quantum) => Some(quantum),
            BloqNodeKind::Classical(_) | BloqNodeKind::Region(_) => None,
        }
    }

    /// The node's classical payload, or `None` for a non-classical node.
    pub fn try_classical(&self) -> Option<&ClassicalNode> {
        self.kind.try_classical()
    }

    /// Mutable classical payload, detaching a shared definition before editing.
    pub fn try_classical_mut(&mut self) -> Option<&mut ClassicalNode> {
        self.kind.try_classical_mut()
    }

    /// The node's region payload, or `None` for a non-region node.
    pub fn try_region(&self) -> Option<&RegionNode> {
        match &self.kind {
            BloqNodeKind::Region(region) => Some(region),
            BloqNodeKind::Quantum(_) | BloqNodeKind::Classical(_) => None,
        }
    }

    /// Source block members, or an empty slice for another provenance kind.
    pub fn block_members(&self) -> &[SourceBlockRef] {
        match &self.provenance {
            NodeProvenance::BlockComponent { members } => members,
            _ => &[],
        }
    }

    /// The spliced node's wait duration in syndrome-extraction rounds, or
    /// `None` for anything that is not a wait on a seam.
    ///
    pub fn memory_rounds(&self) -> Option<u32> {
        match &self.provenance {
            NodeProvenance::MemoryPadding { rounds, .. } => Some(*rounds),
            _ => None,
        }
    }

    /// The node's spatial layer on the doubled-z axis: block layers land on
    /// even values (`2z`) and the pipe seams between them on odd values
    /// (`2z + 1`), so temporally adjacent blocks and their connecting seam
    /// stay strictly ordered. Classical, frame, and action nodes have no
    /// spatial layer and report `0`.
    pub fn layer(&self) -> i64 {
        match &self.provenance {
            NodeProvenance::BlockComponent { members } => {
                2 * i64::from(members.iter().map(|member| member.pos.z).min().unwrap_or(0))
            }
            NodeProvenance::TemporalPipe { pipe } | NodeProvenance::MemoryPadding { pipe, .. } => {
                2 * i64::from(pipe.src.z.min(pipe.dst.z)) + 1
            }
            // Spec rule SEM-SPATIAL-PORT: positionless boundary layer.
            NodeProvenance::SpatialPortSubstitution { source, role } => {
                2 * i64::from(source.z) + if role.has_input_boundary() { -1 } else { 1 }
            }
            // Classical/frame/action nodes have no spatial layer of their own.
            NodeProvenance::Generator { .. }
            | NodeProvenance::Action { .. }
            | NodeProvenance::BranchSelector { .. }
            | NodeProvenance::OutputFrame { .. }
            | NodeProvenance::None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::ivec3;

    use super::*;

    /// [`ClassicalExpr::replace_inputs`] on a copy, so an assertion can compare
    /// the mapped expression against the original.
    fn map_inputs(expr: &ClassicalExpr, f: &mut impl FnMut(u32) -> ClassicalExpr) -> ClassicalExpr {
        let mut mapped = expr.clone();
        mapped.replace_inputs(f);
        mapped
    }

    #[test]
    fn affine_compaction_preserves_unknowns_order_and_substitution() {
        use ClassicalExpr::{Const, In, Not, Parity, Xor};
        let tree = Xor(Box::new([In(2), Not(Box::new(In(0))), Const(true), In(2)]));
        let mut compact = tree.clone();
        compact.compact_affine();
        assert_eq!(
            compact,
            Parity {
                inputs: Box::new([2, 0, 2]),
                constant: false
            }
        );
        assert_eq!(map_inputs(&compact, &mut In), compact);
        for a in [None, Some(false), Some(true)] {
            for b in [None, Some(false), Some(true)] {
                let mut input = |slot| if slot == 0 { a } else { b };
                assert_eq!(compact.eval(&mut input), tree.eval(&mut input));
                for expr in [&compact, &tree] {
                    let mut visited = Vec::new();
                    let mapped = map_inputs(expr, &mut |slot| {
                        visited.push(slot);
                        Not(Box::new(In(slot)))
                    });
                    assert_eq!(visited, [2, 0, 2]);
                    assert_eq!(
                        mapped.eval(&mut input),
                        tree.eval(&mut |slot| input(slot).map(|bit| !bit))
                    );
                }
            }
        }
    }

    #[test]
    fn doubled_layers_cover_the_full_source_coordinate_range() {
        let block = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, i32::MAX),
        }]);
        let pipe = BloqNode::from_temporal_pipe(TemporalPipeRef {
            src: ivec3(0, 0, i32::MAX - 1),
            dst: ivec3(0, 0, i32::MAX),
            hadamard: true,
        });

        assert_eq!(block.layer(), 2 * i64::from(i32::MAX));
        assert_eq!(pipe.layer(), 2 * i64::from(i32::MAX - 1) + 1);
    }
    #[test]
    fn operand_lists_and_selection_preserve_strict_unknowns_and_input_order() {
        use ClassicalExpr::{And, In, Or, Select, Xor};
        let expressions = [
            Xor(Box::new([In(0), In(1), In(2)])),
            And(Box::new([In(0), In(1), In(2)])),
            Or(Box::new([In(0), In(1), In(2)])),
            Select(Box::new([In(0), In(1), In(2)])),
        ];
        for a in [None, Some(false), Some(true)] {
            for b in [None, Some(false), Some(true)] {
                for c in [None, Some(false), Some(true)] {
                    let expected = a.zip(b).zip(c).map(|((a, b), c)| {
                        [a ^ b ^ c, a & b & c, a | b | c, if a { c } else { b }]
                    });
                    for (index, expr) in expressions.iter().enumerate() {
                        assert_eq!(
                            expr.eval(&mut |slot| [a, b, c][slot as usize]),
                            expected.map(|bits| bits[index])
                        );
                    }
                    for literal in 0..27 {
                        let operands: [_; 3] = std::array::from_fn(|slot| {
                            match literal / 3usize.pow(slot as u32) % 3 {
                                0 => In(slot as u32),
                                value => ClassicalExpr::Const(value == 2),
                            }
                        });
                        let raw = Select(Box::new(operands.clone()));
                        let [condition, low, high] = operands;
                        let compact = ClassicalExpr::select(condition, low, high);
                        assert_eq!(
                            compact.eval(&mut |slot| [a, b, c][slot as usize]),
                            raw.eval(&mut |slot| [a, b, c][slot as usize])
                        );
                    }
                }
            }
        }
        for expr in &expressions {
            let mut visited = Vec::new();
            let mapped = map_inputs(expr, &mut |slot| {
                visited.push(slot);
                ClassicalExpr::Not(Box::new(In(2 - slot)))
            });
            assert_eq!(visited, [0, 1, 2]);
            visited.clear();
            mapped.for_each_input(&mut |slot| visited.push(slot));
            assert_eq!(visited, [2, 1, 0]);
        }
        assert!(expressions[0].is_linear());
        assert!(expressions[1..].iter().all(|expr| !expr.is_linear()));
        for (expr, expected) in [
            (Xor(Box::new([])), false),
            (And(Box::new([])), true),
            (Or(Box::new([])), false),
        ] {
            assert_eq!(
                expr.eval(&mut |_| panic!("empty expression reads no input")),
                Some(expected)
            );
        }
        for expr in [
            Xor(Box::new([In(0)])),
            And(Box::new([In(0)])),
            Or(Box::new([In(0)])),
        ] {
            for bit in [None, Some(false), Some(true)] {
                assert_eq!(expr.eval(&mut |_| bit), bit);
            }
        }
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<ClassicalExpr>(), 24);
    }
}
