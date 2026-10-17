//! Structured action bindings: `Action`, `Expr`, `MeasureTarget`,
//! `FeedbackTarget`, and the structured `BlockGraph` action methods that
//! complement the BLOG-text authoring path (`add_actions_from_text`).
//!
//! `Expr` is recursive (`Box<Expr>` upstream), which rules out a pyo3 complex
//! enum (fields must be `Clone` pyclasses, and `Py<T>` recursion breaks that),
//! so it binds as a wrapper struct with staticmethod/operator constructors and
//! `kind`-dispatched accessors — the same fallback `ir.rs` uses for
//! `ClassicalExpr`. Operators surface as name strings
//! (`"not"`, `"xor"`, `"and"`, `"or"`) rather than dedicated classes.

use std::str::FromStr;

use bloq_graph::{Action, ActionDag, Expr, FeedbackTarget, MeasureTarget};
use bloq_utils::PauliBasis;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::errors;
use crate::primitives::{DirectionLike, PyDirection, PyPauliBasis};

type PosTuple = (i32, i32, i32);

// ==============================================================================
// Argument unions
// ==============================================================================

/// Accepts an `Expr` or a variable name (`"m1"` becomes `Expr.var("m1")`).
#[derive(FromPyObject)]
pub(crate) enum ExprLike {
    Expr(PyExpr),
    Var(String),
}

pyo3_stub_gen::impl_stub_type!(ExprLike = PyExpr | String);

impl From<ExprLike> for Expr {
    fn from(value: ExprLike) -> Self {
        match value {
            ExprLike::Expr(e) => e.0,
            ExprLike::Var(name) => Expr::Var(name),
        }
    }
}

/// Accepts a `MeasureTarget` or a bare position tuple (treated as a node).
#[derive(FromPyObject)]
pub(crate) enum MeasureTargetLike {
    Target(PyMeasureTarget),
    Node(PosTuple),
}

pyo3_stub_gen::impl_stub_type!(MeasureTargetLike = PyMeasureTarget | PosTuple);

impl From<MeasureTargetLike> for MeasureTarget {
    fn from(value: MeasureTargetLike) -> Self {
        match value {
            MeasureTargetLike::Target(t) => t.0,
            MeasureTargetLike::Node(pos) => MeasureTarget::Node(pos.into()),
        }
    }
}

/// Accepts a `PauliBasis` or its letter (`"X"` / `"Y"` / `"Z"`).
#[derive(FromPyObject)]
pub(crate) enum PauliBasisLike {
    Basis(PyPauliBasis),
    Text(String),
}

pyo3_stub_gen::impl_stub_type!(PauliBasisLike = PyPauliBasis | String);

impl TryFrom<PauliBasisLike> for PauliBasis {
    type Error = PyErr;

    fn try_from(value: PauliBasisLike) -> Result<Self, Self::Error> {
        match value {
            PauliBasisLike::Basis(b) => Ok(b.into()),
            PauliBasisLike::Text(s) => {
                PauliBasis::from_str(&s.to_uppercase()).map_err(errors::invalid_argument)
            }
        }
    }
}

/// Accepts a `FeedbackTarget` or a `(pauli, pos)` tuple.
#[derive(FromPyObject)]
pub(crate) enum FeedbackTargetLike {
    Target(PyFeedbackTarget),
    Pair((PauliBasisLike, PosTuple)),
}

pyo3_stub_gen::impl_stub_type!(FeedbackTargetLike = PyFeedbackTarget | (PauliBasisLike, PosTuple));

impl TryFrom<FeedbackTargetLike> for FeedbackTarget {
    type Error = PyErr;

    fn try_from(value: FeedbackTargetLike) -> Result<Self, Self::Error> {
        match value {
            FeedbackTargetLike::Target(t) => Ok(t.0),
            FeedbackTargetLike::Pair((pauli, pos)) => Ok(FeedbackTarget {
                pauli: pauli.try_into()?,
                target: pos.into(),
                direction: None,
            }),
        }
    }
}

// ==============================================================================
// Expr
// ==============================================================================

/// A boolean expression over measurement outcome variables.
///
/// Build with `Expr.var("m")` (or pass a bare `str` wherever an expression is
/// expected) and compose with `^` (xor), `&` (and), `|` (or), `~` (not) —
/// each maps 1:1 to the BLOG expression operators. Python truth testing
/// (`bool`, `if`, `and`, `or`, `not`) raises `TypeError`: outcomes are unknown
/// until execution. Use the bitwise operators to build expressions.
///
/// Examples:
///     >>> from bloq import Expr
///     >>> e = Expr.var("m1") ^ Expr.var("m2")
///     >>> e.kind, e.op
///     ('binary', 'xor')
///     >>> [str(o) for o in e.operands()]
///     ['m1', 'm2']
#[gen_stub_pyclass]
#[pyclass(name = "Expr", module = "bloq._core", eq, frozen, from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyExpr(pub(crate) Expr);

impl PyExpr {
    fn binary(op: bloq_graph::BinaryOp, lhs: Expr, rhs: Expr) -> Self {
        PyExpr(Expr::Binary(op, Box::new(lhs), Box::new(rhs)))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyExpr {
    /// A named variable reference (a measurement outcome or `let` name).
    ///
    /// Args:
    ///     name: The variable name (a measurement outcome or `let` name).
    #[staticmethod]
    fn var(name: &str) -> Self {
        PyExpr(Expr::Var(name.to_string()))
    }

    /// `"var"`, `"unary"`, or `"binary"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.0 {
            Expr::Var(..) => "var",
            Expr::Not(_) => "unary",
            Expr::Binary(..) => "binary",
        }
    }

    /// Variable name (`kind == "var"` only).
    #[getter]
    fn name(&self) -> Option<&str> {
        match &self.0 {
            Expr::Var(name) => Some(name),
            _ => None,
        }
    }

    /// Operator name — `"not"`, `"xor"`, `"and"`, or `"or"` — for unary and
    /// binary expressions.
    #[getter]
    fn op(&self) -> Option<&'static str> {
        match &self.0 {
            Expr::Var(..) => None,
            Expr::Not(_) => Some("not"),
            Expr::Binary(op, ..) => Some(match op {
                bloq_graph::BinaryOp::Xor => "xor",
                bloq_graph::BinaryOp::And => "and",
                bloq_graph::BinaryOp::Or => "or",
            }),
        }
    }

    /// Direct sub-expressions (empty for `kind == "var"`).
    fn operands(&self) -> Vec<PyExpr> {
        match &self.0 {
            Expr::Var(..) => vec![],
            Expr::Not(e) => vec![PyExpr((**e).clone())],
            Expr::Binary(_, lhs, rhs) => {
                vec![PyExpr((**lhs).clone()), PyExpr((**rhs).clone())]
            }
        }
    }

    fn __xor__(&self, rhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::Xor, self.0.clone(), rhs.into())
    }

    fn __rxor__(&self, lhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::Xor, lhs.into(), self.0.clone())
    }

    fn __and__(&self, rhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::And, self.0.clone(), rhs.into())
    }

    fn __rand__(&self, lhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::And, lhs.into(), self.0.clone())
    }

    fn __or__(&self, rhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::Or, self.0.clone(), rhs.into())
    }

    fn __ror__(&self, lhs: ExprLike) -> Self {
        Self::binary(bloq_graph::BinaryOp::Or, lhs.into(), self.0.clone())
    }

    fn __invert__(&self) -> Self {
        PyExpr(Expr::Not(Box::new(self.0.clone())))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "symbolic Expr has no truth value; use &, |, ^, and ~ to build expressions",
        ))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Expr {}>", self.0)
    }
}

// ==============================================================================
// MeasureTarget / FeedbackTarget
// ==============================================================================

/// The target of a measurement action: a node position, or an edge identified
/// by its source position and direction.
///
/// Examples:
///     >>> from bloq import MeasureTarget
///     >>> node = MeasureTarget.node((0, 0, 0))
///     >>> node.is_edge
///     False
///     >>> edge = MeasureTarget.edge((0, 0, 0), "+Z")
///     >>> edge.is_edge, str(edge.direction)
///     (True, '+Z')
#[gen_stub_pyclass]
#[pyclass(
    name = "MeasureTarget",
    module = "bloq._core",
    eq,
    frozen,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyMeasureTarget(pub(crate) MeasureTarget);

#[gen_stub_pymethods]
#[pymethods]
impl PyMeasureTarget {
    /// Measurement of the block at `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` block position to measure.
    #[staticmethod]
    fn node(pos: PosTuple) -> Self {
        PyMeasureTarget(MeasureTarget::Node(pos.into()))
    }

    /// Measurement of the pipe leaving `src` in `direction`.
    ///
    /// Args:
    ///     src: The `(x, y, z)` source block position.
    ///     direction: A `Direction` or its BLOG spelling (`"+Z"`, ...).
    ///
    /// Raises:
    ///     InvalidArgumentError: If `direction` is not a valid spelling.
    #[staticmethod]
    fn edge(src: PosTuple, direction: DirectionLike) -> PyResult<Self> {
        Ok(PyMeasureTarget(MeasureTarget::Edge {
            src: src.into(),
            dir: direction.try_into()?,
        }))
    }

    /// Node position, or edge source position.
    #[getter]
    fn pos(&self) -> PosTuple {
        match self.0 {
            MeasureTarget::Node(pos) | MeasureTarget::Edge { src: pos, .. } => pos.into(),
        }
    }

    /// Edge direction; `None` for node targets.
    #[getter]
    fn direction(&self) -> Option<PyDirection> {
        match self.0 {
            MeasureTarget::Node(..) => None,
            MeasureTarget::Edge { dir, .. } => Some(dir.into()),
        }
    }

    /// `True` for edge targets, `False` for node targets.
    #[getter]
    fn is_edge(&self) -> bool {
        matches!(self.0, MeasureTarget::Edge { .. })
    }

    fn __str__(&self) -> String {
        match self.0 {
            MeasureTarget::Node(pos) => format!("{pos}"),
            MeasureTarget::Edge { src, dir } => format!("{src} -> {dir}"),
        }
    }

    fn __repr__(&self) -> String {
        format!("<MeasureTarget {}>", self.__str__())
    }
}

/// A Pauli feedback applied to the block at `pos`.
///
/// The `pauli` argument accepts a `PauliBasis` or its `"X"` / `"Y"` / `"Z"`
/// letter.
///
/// Examples:
///     >>> from bloq import FeedbackTarget
///     >>> ft = FeedbackTarget("X", (0, 0, 0))
///     >>> str(ft.pauli), ft.pos
///     ('X', (0, 0, 0))
#[gen_stub_pyclass]
#[pyclass(
    name = "FeedbackTarget",
    module = "bloq._core",
    eq,
    frozen,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyFeedbackTarget(pub(crate) FeedbackTarget);

#[gen_stub_pymethods]
#[pymethods]
impl PyFeedbackTarget {
    /// `FeedbackTarget("X", (0, 0, 0))` — a Pauli feedback (`pauli` accepts a
    /// `PauliBasis` or its `"X"` / `"Y"` / `"Z"` letter) on the block at `pos`.
    ///
    /// Args:
    ///     pauli: A `PauliBasis` or its `"X"` / `"Y"` / `"Z"` letter.
    ///     pos: The `(x, y, z)` target block position.
    ///     direction: Optional outgoing wire direction, such as `"+Z"`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `pauli` is not a valid basis letter.
    #[new]
    #[pyo3(signature = (pauli, pos, *, direction=None))]
    fn new(
        pauli: PauliBasisLike,
        pos: PosTuple,
        direction: Option<DirectionLike>,
    ) -> PyResult<Self> {
        Ok(PyFeedbackTarget(FeedbackTarget {
            pauli: pauli.try_into()?,
            target: pos.into(),
            direction: direction.map(TryInto::try_into).transpose()?,
        }))
    }

    /// The feedback Pauli basis.
    #[getter]
    fn pauli(&self) -> PyPauliBasis {
        self.0.pauli.into()
    }

    /// The target block position `(x, y, z)`.
    #[getter]
    fn pos(&self) -> PosTuple {
        self.0.target.into()
    }

    /// The outgoing wire direction, or `None` for a node correction.
    #[getter]
    fn direction(&self) -> Option<PyDirection> {
        self.0.direction.map(Into::into)
    }

    fn __str__(&self) -> String {
        match self.0.direction {
            Some(dir) => format!("{}{} -> {}", self.0.pauli, self.0.target, dir),
            None => format!("{}{}", self.0.pauli, self.0.target),
        }
    }

    fn __repr__(&self) -> String {
        format!("<FeedbackTarget {}>", self.__str__())
    }
}

// ==============================================================================
// Action
// ==============================================================================

/// A classical action between time steps of a block-graph program.
///
/// Construct via the staticmethod builders; most mirror BLOG action statements.
/// `branch` is the graph-dependent internal form of a named branch resolve.
/// `str(action)` renders a coordinate form (with positions, not block ids).
///
/// Examples:
///     >>> from bloq import Action, Expr
///     >>> a = Action.measure((0, 0, 0), "m")
///     >>> a.kind, a.name
///     ('measure', 'm')
///     >>> let = Action.let("q", Expr.var("m1") ^ Expr.var("m2"))
///     >>> str(let)
///     'q = m1 ^ m2'
#[gen_stub_pyclass]
#[pyclass(name = "Action", module = "bloq._core", eq, frozen, from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyAction(pub(crate) Action);

#[gen_stub_pymethods]
#[pymethods]
impl PyAction {
    /// `let <name> = <expr>` — define a named boolean variable.
    ///
    /// Args:
    ///     name: The variable name to define.
    ///     expr: The defining expression; an `Expr` or a variable-name `str`.
    #[staticmethod]
    #[pyo3(name = "let")]
    fn let_(name: &str, expr: ExprLike) -> Self {
        PyAction(Action::Let {
            name: name.to_string(),
            expr: expr.into(),
        })
    }

    /// `<name> = measure <target>` — bind a measurement outcome variable.
    ///
    /// Args:
    ///     target: A `MeasureTarget` or a bare `(x, y, z)` position (a node).
    ///     name: The outcome variable name to bind.
    #[staticmethod]
    fn measure(target: MeasureTargetLike, name: &str) -> Self {
        PyAction(Action::Measure {
            target: target.into(),
            name: name.to_string(),
        })
    }

    /// `discard if <expr>` — post-select the shot away when `expr` is true.
    ///
    /// Args:
    ///     expr: The predicate; an `Expr` or a variable-name `str`.
    #[staticmethod]
    fn discard_if(expr: ExprLike) -> Self {
        PyAction(Action::DiscardIf(expr.into()))
    }

    /// `resolve <target> if <condition>` — resolve a selective block.
    ///
    /// Args:
    ///     target: The `(x, y, z)` selective block position.
    ///     condition: The resolving condition; an `Expr` or a variable-name
    ///         `str`.
    #[staticmethod]
    fn resolve(target: PosTuple, condition: ExprLike) -> Self {
        PyAction(Action::Resolve {
            target: target.into(),
            condition: condition.into(),
        })
    }

    /// Internal structural resolve action for an existing named branch region.
    /// Prefer BLOG `resolve <branch-name> if <condition>` when authoring one.
    ///
    /// Args:
    ///     target: The region's internal `(x, y, z)` true-arm key.
    ///     condition: The branch condition; an `Expr` or a variable-name
    ///         `str`.
    #[staticmethod]
    fn branch(target: PosTuple, condition: ExprLike) -> Self {
        PyAction(Action::Branch {
            target: target.into(),
            condition: condition.into(),
        })
    }

    /// `feedback <targets> [if <condition>]` — apply Pauli feedback, optionally
    /// gated. Targets accept `FeedbackTarget` instances or `(pauli, pos)`
    /// tuples.
    ///
    /// Args:
    ///     targets: The feedback targets; `FeedbackTarget` instances or
    ///         `(pauli, pos)` tuples.
    ///     condition: Optional gate expression; an `Expr` or a variable-name
    ///         `str`. Defaults to `None` (always applied).
    ///
    /// Raises:
    ///     InvalidArgumentError: If a target's Pauli letter is invalid.
    #[staticmethod]
    #[pyo3(signature = (targets, condition = None))]
    fn feedback(targets: Vec<FeedbackTargetLike>, condition: Option<ExprLike>) -> PyResult<Self> {
        Ok(PyAction(Action::Feedback {
            targets: targets
                .into_iter()
                .map(TryInto::try_into)
                .collect::<PyResult<_>>()?,
            condition: condition.map(Into::into),
        }))
    }

    /// `"let"`, `"measure"`, `"discard_if"`, `"resolve"`, `"branch"`, or
    /// `"feedback"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.0 {
            Action::Let { .. } => "let",
            Action::Measure { .. } => "measure",
            Action::DiscardIf(..) => "discard_if",
            Action::Resolve { .. } => "resolve",
            Action::Branch { .. } => "branch",
            Action::Feedback { .. } => "feedback",
        }
    }

    /// Variable name for `let` and `measure` actions.
    #[getter]
    fn name(&self) -> Option<&str> {
        match &self.0 {
            Action::Let { name, .. } | Action::Measure { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Defining expression of a `let` action.
    #[getter]
    fn expr(&self) -> Option<PyExpr> {
        match &self.0 {
            Action::Let { expr, .. } => Some(PyExpr(expr.clone())),
            _ => None,
        }
    }

    /// Guard expression: the `discard if` predicate, the `resolve` or `branch`
    /// condition, or the optional `feedback` gate.
    #[getter]
    fn condition(&self) -> Option<PyExpr> {
        match &self.0 {
            Action::DiscardIf(expr)
            | Action::Resolve {
                condition: expr, ..
            }
            | Action::Branch {
                condition: expr, ..
            } => Some(PyExpr(expr.clone())),
            Action::Feedback { condition, .. } => condition.clone().map(PyExpr),
            _ => None,
        }
    }

    /// Measurement target of a `measure` action.
    #[getter]
    fn target(&self) -> Option<PyMeasureTarget> {
        match &self.0 {
            Action::Measure { target, .. } => Some(PyMeasureTarget(*target)),
            _ => None,
        }
    }

    /// Selective block position of a `resolve` action.
    #[getter]
    fn resolve_target(&self) -> Option<PosTuple> {
        match &self.0 {
            Action::Resolve { target, .. } => Some((*target).into()),
            _ => None,
        }
    }

    /// Internal true-arm key of a structural branch action.
    #[getter]
    fn branch_target(&self) -> Option<PosTuple> {
        match &self.0 {
            Action::Branch { target, .. } => Some((*target).into()),
            _ => None,
        }
    }

    /// Targets of a `feedback` action.
    #[getter]
    fn feedback_targets(&self) -> Option<Vec<PyFeedbackTarget>> {
        match &self.0 {
            Action::Feedback { targets, .. } => {
                Some(targets.iter().copied().map(PyFeedbackTarget).collect())
            }
            _ => None,
        }
    }

    /// This action with every position shifted by `offset`.
    ///
    /// Args:
    ///     offset: The `(x, y, z)` amount to shift every position by.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a shifted position exceeds the 32-bit
    ///         coordinate range.
    fn with_shift(&self, offset: PosTuple) -> PyResult<Self> {
        self.0
            .try_with_shift(offset)
            .map(PyAction)
            .map_err(|error| {
                errors::InvalidArgumentError::new_err(format!(
                    "shift exceeds the i32 coordinate range: {error}"
                ))
            })
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Action {}>", self.0)
    }
}

// ==============================================================================
// ActionDag
// ==============================================================================

/// A snapshot of source action dependencies.
///
/// Returned by `BlockGraph.analyze_action_graph()`, or by
/// `BlockGraph.action_graph()` to inspect cached dependencies. Node ids are
/// DAG ordinals indexing `actions()`. Rendering alone does not analyze the graph.
#[gen_stub_pyclass]
#[pyclass(name = "ActionDag", module = "bloq._core", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyActionDag(pub(crate) ActionDag);

#[gen_stub_pymethods]
#[pymethods]
impl PyActionDag {
    /// Whether correlation support dependencies have been analyzed.
    /// An empty action program has nothing to derive.
    #[getter]
    fn is_analyzed(&self) -> bool {
        self.0.is_analyzed()
    }

    /// Returns the actions in DAG ordinal order.
    fn actions(&self) -> Vec<PyAction> {
        self.0
            .ordered_nodes()
            .map(|node| PyAction(node.action.clone()))
            .collect()
    }

    /// Returns `(definition_name, instance_path)` ownership for each action.
    ///
    /// Paths use `__` between instances and are empty for the root.
    /// Interface bindings belong to their receiving instance. Entries are
    /// `None` for standalone flat graphs without authored module ownership.
    /// The list has the same ordinal order as `actions()`.
    fn owners(&self) -> Vec<Option<(String, String)>> {
        self.0
            .ordered_nodes()
            .map(|node| {
                node.owner
                    .as_ref()
                    .map(|owner| (owner.definition.clone(), owner.instance_path.clone()))
            })
            .collect()
    }

    /// Returns `(predecessor, consumer, reason)` edges in deterministic order.
    ///
    /// Reasons are `Classical`, `SelectiveSupport`, `BranchSupport`,
    /// `FeedbackAnticommutation`, or `ReadoutParity`. Ordinals index `actions()`.
    fn dependencies(&self) -> Vec<(usize, usize, String)> {
        let mut edges = self
            .0
            .dependencies()
            .map(|(from, to, reason)| (from, to, format!("{reason:?}")))
            .collect::<Vec<_>>();
        edges.sort();
        edges
    }

    /// Returns the declared external Boolean input names.
    fn inputs(&self) -> Vec<String> {
        self.0.inputs().map(str::to_owned).collect()
    }

    /// Renders a self-contained SVG of the stored dependencies.
    ///
    /// Solid arrows show variable dependencies. Dashed arrows show dependencies
    /// inferred from correlation support. Full action text and dependency reasons
    /// are retained. Rendering does not derive missing dependencies or set times.
    ///
    /// Raises:
    ///     BlockGraphError: If action syntax, names or dependency cycles are invalid.
    fn to_svg(&self, py: Python<'_>) -> PyResult<String> {
        let dag = &self.0;
        py.detach(move || dag.to_svg())
            .map_err(|error| errors::BlockGraphError::new_err(error.to_string()))
    }

    fn __len__(&self) -> usize {
        self.0.ordered_nodes().count()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ActionDag actions={} analyzed={}>",
            self.__len__(),
            self.is_analyzed()
        )
    }
}

// ==============================================================================
// Registration
// ==============================================================================

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PyMeasureTarget>()?;
    m.add_class::<PyFeedbackTarget>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyActionDag>()?;
    Ok(())
}
