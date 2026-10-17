//! Leaf types shared across the binding surface, plus conversion helpers.
//!
//! This file establishes the patterns every other binding file copies:
//!
//! - Wrapper types named `Py<Name>` with `#[pyclass(name = "<Name>",
//!   module = "bloq._core", ...)]` and `From` conversions both ways.
//! - `#[gen_stub_pyclass]` / `#[gen_stub_pyclass_enum]` /
//!   `#[gen_stub_pymethods]` stacked *above* the pyo3 macro so
//!   `just py-stub` picks the item up for `.pyi` generation.
//! - `glam::IVec3`/`IVec2` never cross the boundary as classes — positions
//!   are plain `(int, int, int)` / `(int, int)` tuples.
//! - Fallible upstream calls raise the exceptions in [`crate::errors`]; invalid
//!   *arguments* (bad literals) raise `InvalidArgumentError` from that same
//!   hierarchy. The Python protocols keep their builtins: sequence indexing
//!   raises `IndexError`, and mutable containers raise `TypeError` for hashing.

use std::str::FromStr;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use crate::errors;

// ==============================================================================
// Conversion helpers (crate-internal, not exposed to Python)
// ==============================================================================

/// A `bool` spelled the way Python spells it, for `__repr__` bodies: Rust's
/// `Display` writes `true`, which reads as a bug in Python output.
pub(crate) fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// An optional value spelled the way Python spells it, for `__repr__` bodies:
/// `Debug` on an `Option` writes `Some(3)`, which is not Python.
pub(crate) fn py_option<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| value.to_string())
}

// ==============================================================================
// Basis
// ==============================================================================

/// A measurement basis: the Pauli `X` or `Z` axis.
///
/// Examples:
///     >>> from bloq import Basis
///     >>> str(Basis.X)
///     'X'
///     >>> str(Basis.X.flip())
///     'Z'
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "Basis",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyBasis {
    X,
    Z,
}

impl From<bloq_utils::Basis> for PyBasis {
    fn from(b: bloq_utils::Basis) -> Self {
        match b {
            bloq_utils::Basis::X => PyBasis::X,
            bloq_utils::Basis::Z => PyBasis::Z,
        }
    }
}

impl From<PyBasis> for bloq_utils::Basis {
    fn from(b: PyBasis) -> Self {
        match b {
            PyBasis::X => bloq_utils::Basis::X,
            PyBasis::Z => bloq_utils::Basis::Z,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBasis {
    /// Returns the complementary basis (`X` <-> `Z`).
    fn flip(&self) -> PyBasis {
        bloq_utils::Basis::from(*self).flip().into()
    }

    fn __str__(&self) -> String {
        bloq_utils::Basis::from(*self).to_string()
    }
}

// ==============================================================================
// Pauli / PauliBasis
// ==============================================================================

/// A single-qubit Pauli operator (`I`, `X`, `Y`, or `Z`).
///
/// Examples:
///     >>> from bloq import Pauli
///     >>> Pauli.X.anticommutes(Pauli.Z)
///     True
///     >>> Pauli.X.anticommutes(Pauli.X)
///     False
///     >>> str(Pauli.X.flip())
///     'Z'
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "Pauli",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyPauli {
    I,
    X,
    Z,
    Y,
}

impl From<bloq_utils::Pauli> for PyPauli {
    fn from(p: bloq_utils::Pauli) -> Self {
        match p {
            bloq_utils::Pauli::I => PyPauli::I,
            bloq_utils::Pauli::X => PyPauli::X,
            bloq_utils::Pauli::Z => PyPauli::Z,
            bloq_utils::Pauli::Y => PyPauli::Y,
        }
    }
}

impl From<PyPauli> for bloq_utils::Pauli {
    fn from(p: PyPauli) -> Self {
        match p {
            PyPauli::I => bloq_utils::Pauli::I,
            PyPauli::X => bloq_utils::Pauli::X,
            PyPauli::Z => bloq_utils::Pauli::Z,
            PyPauli::Y => bloq_utils::Pauli::Y,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPauli {
    /// Swaps `X` and `Z`; leaves `I` and `Y` unchanged.
    fn flip(&self) -> PyPauli {
        bloq_utils::Pauli::from(*self).flip().into()
    }

    /// Two Paulis anticommute iff both are non-identity and differ.
    ///
    /// Args:
    ///     other: The Pauli to test against.
    fn anticommutes(&self, other: PyPauli) -> bool {
        bloq_utils::Pauli::from(*self).anticommutes(other.into())
    }

    fn __str__(&self) -> String {
        bloq_utils::Pauli::from(*self).to_string()
    }
}

/// A single-qubit Pauli basis: `X`, `Y`, or `Z` (excludes identity).
///
/// A non-identity Pauli axis — unlike `Pauli`, which also has `I`. Use it
/// where an axis must be definite: measurement bases in actions, feedback
/// targets (`FeedbackTarget.pauli`), and compiled-circuit operations. APIs
/// taking one usually also accept its `"X"` / `"Y"` / `"Z"` letter.
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "PauliBasis",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyPauliBasis {
    X,
    Y,
    Z,
}

impl From<bloq_utils::PauliBasis> for PyPauliBasis {
    fn from(p: bloq_utils::PauliBasis) -> Self {
        match p {
            bloq_utils::PauliBasis::X => PyPauliBasis::X,
            bloq_utils::PauliBasis::Y => PyPauliBasis::Y,
            bloq_utils::PauliBasis::Z => PyPauliBasis::Z,
        }
    }
}

impl From<PyPauliBasis> for bloq_utils::PauliBasis {
    fn from(p: PyPauliBasis) -> Self {
        match p {
            PyPauliBasis::X => bloq_utils::PauliBasis::X,
            PyPauliBasis::Y => bloq_utils::PauliBasis::Y,
            PyPauliBasis::Z => bloq_utils::PauliBasis::Z,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPauliBasis {
    fn __str__(&self) -> String {
        bloq_utils::PauliBasis::from(*self).to_string()
    }
}

// ==============================================================================
// Direction / UDirection
// ==============================================================================

/// A signed axis direction in the block-graph lattice (`+X`, `-X`, ..., `-Z`).
///
/// Python variant names are `X_PLUS`-style; `str(d)` and
/// `Direction.parse(...)` use the BLOG spelling (`"+X"`, `"-Z"`, ...).
///
/// Examples:
///     >>> from bloq import Direction
///     >>> str(Direction.parse("+X"))
///     '+X'
///     >>> str(Direction.X_PLUS.negate())
///     '-X'
///     >>> Direction.X_PLUS.vector()
///     (1, 0, 0)
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "Direction",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyDirection {
    #[pyo3(name = "X_PLUS")]
    XPlus,
    #[pyo3(name = "X_MINUS")]
    XMinus,
    #[pyo3(name = "Y_PLUS")]
    YPlus,
    #[pyo3(name = "Y_MINUS")]
    YMinus,
    #[pyo3(name = "Z_PLUS")]
    ZPlus,
    #[pyo3(name = "Z_MINUS")]
    ZMinus,
}

impl From<bloq_utils::Direction> for PyDirection {
    fn from(d: bloq_utils::Direction) -> Self {
        match d {
            bloq_utils::Direction::XPLUS => PyDirection::XPlus,
            bloq_utils::Direction::XMINUS => PyDirection::XMinus,
            bloq_utils::Direction::YPLUS => PyDirection::YPlus,
            bloq_utils::Direction::YMINUS => PyDirection::YMinus,
            bloq_utils::Direction::ZPLUS => PyDirection::ZPlus,
            bloq_utils::Direction::ZMINUS => PyDirection::ZMinus,
        }
    }
}

impl From<PyDirection> for bloq_utils::Direction {
    fn from(d: PyDirection) -> Self {
        match d {
            PyDirection::XPlus => bloq_utils::Direction::XPLUS,
            PyDirection::XMinus => bloq_utils::Direction::XMINUS,
            PyDirection::YPlus => bloq_utils::Direction::YPLUS,
            PyDirection::YMinus => bloq_utils::Direction::YMINUS,
            PyDirection::ZPlus => bloq_utils::Direction::ZPLUS,
            PyDirection::ZMinus => bloq_utils::Direction::ZMINUS,
        }
    }
}

/// Accepts either a `Direction` or its BLOG spelling (`"+X"`) in APIs.
///
/// Worker pattern: take `DirectionLike` as the argument type wherever the
/// Rust API wants a `Direction`, then `.try_into()?` it.
#[derive(FromPyObject)]
pub(crate) enum DirectionLike {
    Direction(PyDirection),
    Text(String),
}

// Union types used as arguments need a manual stub mapping so the generated
// `.pyi` shows `Direction | str` instead of failing to compile.
pyo3_stub_gen::impl_stub_type!(DirectionLike = PyDirection | String);

impl TryFrom<DirectionLike> for bloq_utils::Direction {
    type Error = PyErr;

    fn try_from(value: DirectionLike) -> Result<Self, Self::Error> {
        match value {
            DirectionLike::Direction(d) => Ok(d.into()),
            DirectionLike::Text(s) => parse_direction(&s),
        }
    }
}

/// Parses a BLOG direction spelling like `"+X"` into a `Direction`.
/// Case-insensitive, matching the other enum-or-string arguments (`Basis`,
/// `BlockKind`, …); the sign prefix survives `to_uppercase` unchanged.
pub(crate) fn parse_direction(s: &str) -> PyResult<bloq_utils::Direction> {
    bloq_utils::Direction::from_str(&s.to_uppercase()).map_err(errors::invalid_argument)
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDirection {
    /// Parses a BLOG direction spelling like `"+X"` or `"-Z"`.
    ///
    /// Args:
    ///     text: A BLOG direction spelling, e.g. `"+X"` or `"-Z"`
    ///         (case-insensitive).
    ///
    /// Raises:
    ///     InvalidArgumentError: If `text` is not a valid direction spelling.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<PyDirection> {
        parse_direction(text).map(Into::into)
    }

    /// Returns the opposite direction along the same axis.
    fn negate(&self) -> PyDirection {
        bloq_utils::Direction::from(*self).negate().into()
    }

    /// Returns `True` for the `X`/`Y` (spatial) directions.
    fn is_spatial(&self) -> bool {
        bloq_utils::Direction::from(*self).is_spatial()
    }

    /// Returns the unit axis vector `(x, y, z)` pointing along this direction.
    fn vector(&self) -> (i32, i32, i32) {
        bloq_utils::Direction::from(*self).to_ivec3().into()
    }

    fn __str__(&self) -> String {
        bloq_utils::Direction::from(*self).to_string()
    }
}

/// An unsigned axis (`X`, `Y`, or `Z`).
///
/// A lattice axis with no orientation — a `Direction` with its sign dropped.
/// Use it where only the axis matters, such as the rotation axis of
/// `BlockGraph.rotate_about_origin`. APIs taking one usually also accept its
/// `"X"` / `"Y"` / `"Z"` letter.
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "UDirection",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyUDirection {
    X,
    Y,
    Z,
}

impl From<bloq_utils::UDirection> for PyUDirection {
    fn from(d: bloq_utils::UDirection) -> Self {
        match d {
            bloq_utils::UDirection::X => PyUDirection::X,
            bloq_utils::UDirection::Y => PyUDirection::Y,
            bloq_utils::UDirection::Z => PyUDirection::Z,
        }
    }
}

impl From<PyUDirection> for bloq_utils::UDirection {
    fn from(d: PyUDirection) -> Self {
        match d {
            PyUDirection::X => bloq_utils::UDirection::X,
            PyUDirection::Y => bloq_utils::UDirection::Y,
            PyUDirection::Z => bloq_utils::UDirection::Z,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUDirection {
    fn __str__(&self) -> String {
        bloq_utils::UDirection::from(*self).to_string()
    }
}

// ==============================================================================
// PauliString
// ==============================================================================

/// Constructor argument for `PauliString`: qubit count or a literal.
#[derive(FromPyObject)]
enum PauliStringInit {
    /// Identity string over this many qubits.
    Len(usize),
    /// Literal like `"XZ_Y"` (`_` or `I` for identity).
    Text(String),
}

pyo3_stub_gen::impl_stub_type!(PauliStringInit = usize | String);

/// A dense Pauli operator over a fixed number of qubits.
///
/// Build from a qubit count (identity string) or a literal like `"XZ_Y"`
/// (`_` or `I` marks identity). Indexable and mutable like a list of `Pauli`.
///
/// Examples:
///     >>> from bloq import PauliString
///     >>> ps = PauliString("XZ_Y")
///     >>> len(ps), ps.weight()
///     (4, 3)
///     >>> str(ps[0])
///     'X'
///     >>> str(PauliString(4))
///     '____'
#[gen_stub_pyclass]
#[pyclass(name = "PauliString", module = "bloq._core", eq, from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyPauliString(pub(crate) bloq_utils::PauliString);

impl From<bloq_utils::PauliString> for PyPauliString {
    fn from(ps: bloq_utils::PauliString) -> Self {
        PyPauliString(ps)
    }
}

impl PyPauliString {
    /// Converts a possibly-negative Python index into a checked offset.
    fn checked_index(&self, index: isize) -> PyResult<usize> {
        let len = self.0.len() as isize;
        let resolved = if index < 0 { index + len } else { index };
        if (0..len).contains(&resolved) {
            Ok(resolved as usize)
        } else {
            Err(PyIndexError::new_err(format!(
                "index {index} out of range for PauliString of length {len}"
            )))
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPauliString {
    /// `PauliString(4)` builds the identity over 4 qubits;
    /// `PauliString("XZ_Y")` parses a literal (`_` or `I` for identity).
    ///
    /// Args:
    ///     init: An `int` qubit count (identity string) or a literal like
    ///         `"XZ_Y"` (`_` or `I` marks identity).
    ///
    /// Raises:
    ///     InvalidArgumentError: If a string `init` is not a valid Pauli literal.
    #[new]
    fn new(init: PauliStringInit) -> PyResult<Self> {
        match init {
            PauliStringInit::Len(n) => Ok(bloq_utils::PauliString::new(n).into()),
            PauliStringInit::Text(s) => bloq_utils::PauliString::try_from(s.as_str())
                .map(Into::into)
                .map_err(errors::invalid_argument),
        }
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// The `Pauli` at `index` (supports negative indexing).
    ///
    /// Args:
    ///     index: Position in the string; negative values index from the end.
    ///
    /// Raises:
    ///     IndexError: If `index` is out of range.
    fn __getitem__(&self, index: isize) -> PyResult<PyPauli> {
        Ok(self.0.get(self.checked_index(index)?).into())
    }

    /// Sets the `Pauli` at `index` (supports negative indexing).
    ///
    /// Args:
    ///     index: Position in the string; negative values index from the end.
    ///     pauli: The `Pauli` to store at `index`.
    ///
    /// Raises:
    ///     IndexError: If `index` is out of range.
    fn __setitem__(&mut self, index: isize, pauli: PyPauli) -> PyResult<()> {
        let index = self.checked_index(index)?;
        self.0.set(index, pauli.into());
        Ok(())
    }

    /// Returns the number of non-identity Paulis.
    fn weight(&self) -> usize {
        self.0.weight()
    }

    /// Copies and pickles through the Pauli literal, preserving independent storage.
    fn __reduce__(&self, py: Python<'_>) -> (Py<PyAny>, (String,)) {
        (
            py.get_type::<Self>().into_any().unbind(),
            (self.0.to_string(),),
        )
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("PauliString(\"{}\")", self.0)
    }
}

// ==============================================================================
// Registration
// ==============================================================================

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasis>()?;
    m.add_class::<PyPauli>()?;
    m.add_class::<PyPauliBasis>()?;
    m.add_class::<PyDirection>()?;
    m.add_class::<PyUDirection>()?;
    m.add_class::<PyPauliString>()?;
    Ok(())
}
