//! Block-graph bindings: `BlockGraph`, `Block`, `BlockKind`, `Pipe`,
//! `Stabilizer`, `StabilizerGenerator`, and `parse_blog`.
//!
//! Mutating CRUD binds the fallible `try_` variants under the plain names and
//! raises `BlockGraphError`; the panicking twins are deliberately not exposed.
//! Actions can be authored as BLOG action-statement text or as structured
//! `Action` values (see `actions.rs`).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::str::FromStr;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::actions::{ExprLike, PyAction, PyActionDag, PyExpr};
use crate::errors;
use crate::primitives::{
    DirectionLike, PyBasis, PyDirection, PyPauli, PyPauliString, PyUDirection,
};

/// A position tuple as it crosses the Python boundary.
type PosTuple = (i32, i32, i32);

/// A parent-local block or a named quantum port on a child instance.
#[derive(FromPyObject)]
enum ConnectionEndpoint {
    Block(PosTuple),
    Port(String),
}
pyo3_stub_gen::impl_stub_type!(ConnectionEndpoint = PosTuple | String);

fn instance_port(endpoint: &str) -> PyResult<bloq_graph::InstancePort> {
    let (instance, port) = endpoint.split_once('.').ok_or_else(|| {
        errors::InvalidArgumentError::new_err("expected an endpoint of the form 'instance.port'")
    })?;
    if instance.is_empty() || port.is_empty() || port.contains('.') {
        return Err(errors::InvalidArgumentError::new_err(
            "expected an endpoint of the form 'instance.port'",
        ));
    }
    Ok(bloq_graph::InstancePort {
        instance: instance.into(),
        port: port.into(),
    })
}

/// One derived row or a phase-free product of logical rows from this graph.
#[derive(FromPyObject)]
pub(crate) enum StabilizerLike {
    Row(PyStabilizerGenerator),
    Indices(Vec<usize>),
}
pyo3_stub_gen::impl_stub_type!(StabilizerLike = PyStabilizerGenerator | Vec<usize>);

impl StabilizerLike {
    fn resolve(self, graph: &bloq_graph::BlockGraph) -> PyResult<bloq_graph::StabilizerGenerator> {
        match self {
            Self::Row(row) => Ok(row.0),
            Self::Indices(indices) => {
                if indices.is_empty() {
                    return Err(errors::InvalidArgumentError::new_err(
                        "stabilizer indices cannot be empty",
                    ));
                }
                let rows = graph.stabilizers().map_err(graph_err)?;
                let mut selected = indices.into_iter().map(|index| {
                    let row = rows.get(index).ok_or_else(|| {
                        errors::InvalidArgumentError::new_err(format!(
                            "stabilizer index {index} out of range ({} generators)",
                            rows.len()
                        ))
                    })?;
                    if row.kind != bloq_graph::StabilizerRowKind::Logical {
                        return Err(errors::InvalidArgumentError::new_err(
                            "stabilizer products require logical rows",
                        ));
                    }
                    Ok(row)
                });
                let mut surface = selected
                    .next()
                    .expect("nonempty selection")?
                    .stabilizer
                    .clone();
                for row in selected {
                    surface.phase_free_mul_assign(&row?.stabilizer);
                }
                Ok(bloq_graph::StabilizerGenerator::new(
                    surface,
                    bloq_graph::StabilizerRowKind::Logical,
                ))
            }
        }
    }
}

fn gltf_face_selectors(
    directions: Option<Vec<DirectionLike>>,
    blocks: Option<Vec<(PosTuple, DirectionLike)>>,
    pipes: Option<Vec<(PosTuple, PosTuple, DirectionLike)>>,
) -> PyResult<Vec<bloq_graph::GltfFaceSelector>> {
    let mut selectors = Vec::new();
    for direction in directions.unwrap_or_default() {
        selectors.push(bloq_graph::GltfFaceSelector::All(direction.try_into()?));
    }
    for (position, face) in blocks.unwrap_or_default() {
        selectors.push(bloq_graph::GltfFaceSelector::Block {
            position: position.into(),
            face: face.try_into()?,
        });
    }
    for (u, v, face) in pipes.unwrap_or_default() {
        selectors.push(bloq_graph::GltfFaceSelector::Pipe {
            u: u.into(),
            v: v.into(),
            face: face.try_into()?,
        });
    }
    Ok(selectors)
}

// ==============================================================================
// Error mapping
// ==============================================================================

fn graph_err(e: bloq_graph::BlockGraphError) -> PyErr {
    match e {
        bloq_graph::BlockGraphError::Io { path, source } => errors::io_error(path, &source),
        other => errors::BlockGraphError::new_err(other.to_string()),
    }
}

fn graph_transform_err(e: bloq_graph::BlockGraphError) -> PyErr {
    match e {
        error @ (bloq_graph::BlockGraphError::CoordinateOverflow { .. }
        | bloq_graph::BlockGraphError::CoordinateNormalizationOverflow { .. }
        | bloq_graph::BlockGraphError::CoordinateRotationOverflow { .. }) => {
            errors::InvalidArgumentError::new_err(format!(
                "coordinate transform exceeds the i32 coordinate range: {error}"
            ))
        }
        other => graph_err(other),
    }
}

/// Parse failures carry the full ariadne-rendered diagnostic (needs the
/// original source text, hence a dedicated helper instead of plain `Display`).
/// Exception messages must be plain text, so use the color-free render.
fn parse_err(e: &bloq_graph::ParseError, source: &str) -> PyErr {
    let mut message = e.render_diagnostic_plain("<blog>", source);
    let error: bloq_compile::CompileError = bloq_graph::ModuleError::Parse(e.clone()).into();
    if error.resource_limit_help().is_some() {
        message.push('\n');
        message.push_str(crate::compile::LIMIT_OVERRIDE_HELP);
    }
    errors::ParseError::new_err(message)
}

fn graph_input_err(error: bloq_graph::BlockGraphError, source: Option<&str>) -> PyErr {
    if let Some(source) = source {
        match &error {
            bloq_graph::BlockGraphError::Parse(error) => return parse_err(error, source),
            bloq_graph::BlockGraphError::ModuleSource(error) => {
                if let bloq_graph::ModuleError::Parse(error) = error.as_ref() {
                    return parse_err(error, source);
                }
            }
            _ => {}
        }
    }
    match error {
        error @ (bloq_graph::BlockGraphError::Parse(_)
        | bloq_graph::BlockGraphError::ModuleSource(_)) => {
            errors::ParseError::new_err(crate::compile::compile_error_message(&error.into()))
        }
        error => graph_err(error),
    }
}

// ==============================================================================
// Argument unions
// ==============================================================================

/// Accepts a `BlockKind` or its BLOG spelling (`"XZZ"`, `"T"`, ...).
#[derive(FromPyObject)]
pub(crate) enum BlockKindLike {
    Kind(PyBlockKind),
    Text(String),
}

pyo3_stub_gen::impl_stub_type!(BlockKindLike = PyBlockKind | String);

impl TryFrom<BlockKindLike> for bloq_graph::BlockKind {
    type Error = PyErr;

    fn try_from(value: BlockKindLike) -> Result<Self, Self::Error> {
        match value {
            BlockKindLike::Kind(k) => Ok(k.0),
            BlockKindLike::Text(s) => s.parse().map_err(errors::invalid_argument),
        }
    }
}

/// Accepts a `Basis` or its spelling (`"X"` / `"Z"`).
#[derive(FromPyObject)]
pub(crate) enum BasisLike {
    Basis(PyBasis),
    Text(String),
}

pyo3_stub_gen::impl_stub_type!(BasisLike = PyBasis | String);

impl TryFrom<BasisLike> for bloq_utils::Basis {
    type Error = PyErr;

    fn try_from(value: BasisLike) -> Result<Self, Self::Error> {
        match value {
            BasisLike::Basis(b) => Ok(b.into()),
            BasisLike::Text(s) => {
                bloq_utils::Basis::from_str(&s.to_uppercase()).map_err(errors::invalid_argument)
            }
        }
    }
}

/// Accepts a `UDirection` or its axis letter (`"X"` / `"Y"` / `"Z"`).
#[derive(FromPyObject)]
pub(crate) enum UDirectionLike {
    Axis(PyUDirection),
    Text(String),
}

pyo3_stub_gen::impl_stub_type!(UDirectionLike = PyUDirection | String);

impl TryFrom<UDirectionLike> for bloq_utils::UDirection {
    type Error = PyErr;

    fn try_from(value: UDirectionLike) -> Result<Self, Self::Error> {
        match value {
            UDirectionLike::Axis(a) => Ok(a.into()),
            UDirectionLike::Text(s) => bloq_utils::UDirection::from_str(&s.to_uppercase())
                .map_err(errors::invalid_argument),
        }
    }
}

// ==============================================================================
// BlockKind
// ==============================================================================

/// The kind of a block: a cube, a moving block, `Y`, a fixed measurement,
/// `Port`, `T`, or a selective block. Data-carrying kinds are built via the
/// static constructors.
///
/// Examples:
///     >>> from bloq import BlockKind
///     >>> k = BlockKind.cube("ZXZ")
///     >>> k.is_cube
///     True
///     >>> tuple(str(b) for b in k.bases)
///     ('Z', 'X', 'Z')
///     >>> BlockKind.t().is_clifford
///     False
#[gen_stub_pyclass]
#[pyclass(
    name = "BlockKind",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyBlockKind(pub(crate) bloq_graph::BlockKind);

impl From<bloq_graph::BlockKind> for PyBlockKind {
    fn from(k: bloq_graph::BlockKind) -> Self {
        PyBlockKind(k)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlockKind {
    /// A cube block from its face-basis spelling, e.g. `"XZZ"`.
    ///
    /// Args:
    ///     spec: The three face bases as an `X`/`Z` string, e.g. `"XZZ"`
    ///         (case-insensitive).
    ///
    /// Raises:
    ///     InvalidArgumentError: If `spec` is not a valid cube spelling.
    #[staticmethod]
    fn cube(spec: &str) -> PyResult<Self> {
        bloq_graph::CubeKind::from_str(&spec.to_uppercase())
            .map(|k| bloq_graph::BlockKind::Cube(k).into())
            .map_err(errors::invalid_argument)
    }

    /// A walking block from its boundary spelling (e.g. `"ZXZ"`) and an
    /// `(x, y)` movement of `(+/-1, 0)`, `(0, +/-1)`, or `(+/-1, +/-1)`.
    ///
    /// Args:
    ///     boundary: The boundary basis spelling, e.g. `"ZXZ"`
    ///         (case-insensitive).
    ///     movement: The `(x, y)` step: `(+/-1, 0)`, `(0, +/-1)`, or
    ///         `(+/-1, +/-1)`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `boundary` or `movement` is invalid.
    #[staticmethod]
    fn walking(boundary: &str, movement: (i32, i32)) -> PyResult<Self> {
        let boundary = bloq_graph::WalkingBoundaryKind::from_str(&boundary.to_uppercase())
            .map_err(errors::invalid_argument)?;
        bloq_graph::WalkingKind::new(boundary, movement.into())
            .map(|k| bloq_graph::BlockKind::Walking(k).into())
            .map_err(errors::invalid_argument)
    }

    /// A patch-rotation block from a basis and an `(x, y)` movement of
    /// `(+/-1, 0)` or `(0, +/-1)`.
    ///
    /// Args:
    ///     basis: A `Basis` or its `"X"` / `"Z"` letter (case-insensitive).
    ///     movement: The `(x, y)` step: `(+/-1, 0)` or `(0, +/-1)`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `basis` or `movement` is invalid.
    #[staticmethod]
    fn patch_rotation(basis: BasisLike, movement: (i32, i32)) -> PyResult<Self> {
        bloq_graph::PatchRotationKind::new(basis.try_into()?, movement.into())
            .map(|k| bloq_graph::BlockKind::PatchRotation(k).into())
            .map_err(errors::invalid_argument)
    }

    /// The Y-basis block.
    #[staticmethod]
    fn y() -> Self {
        bloq_graph::BlockKind::Y.into()
    }

    /// A fixed terminal transversal measurement in the X or Z basis.
    ///
    /// Args:
    ///     basis: A `Basis` or its `"X"` / `"Z"` letter (case-insensitive).
    ///
    /// Raises:
    ///     InvalidArgumentError: If `basis` is not X or Z.
    #[staticmethod]
    fn measurement(basis: BasisLike) -> PyResult<Self> {
        Ok(bloq_graph::BlockKind::Measurement(basis.try_into()?).into())
    }

    /// An open port block.
    #[staticmethod]
    fn port() -> Self {
        bloq_graph::BlockKind::Port.into()
    }

    /// The non-Clifford T block.
    #[staticmethod]
    fn t() -> Self {
        bloq_graph::BlockKind::T.into()
    }

    /// A selective block from its spelling (`"XY"`, `"XZ"`, or `"YZ"`).
    ///
    /// Args:
    ///     spec: The selective spelling: `"XY"`, `"XZ"`, or `"YZ"`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `spec` is not a valid selective spelling.
    #[staticmethod]
    fn selective(spec: &str) -> PyResult<Self> {
        bloq_graph::SelectiveKind::from_str(spec)
            .map(|k| bloq_graph::BlockKind::Selective(k).into())
            .map_err(errors::invalid_argument)
    }

    /// Parses a BLOG kind spelling (`"XZZ"`, `"X"`, `"Y"`, `"Z"`, `"PORT"`,
    /// `"T"`, `"XY"`, ...).
    ///
    /// Walking and patch-rotation kinds carry movement data and cannot be
    /// spelled as a single token; build them via `walking` / `patch_rotation`.
    ///
    /// Args:
    ///     text: A BLOG kind token, e.g. `"XZZ"`, `"X"`, `"Y"`, `"Z"`,
    ///         `"PORT"`, `"T"`, or `"XY"`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `text` is not a valid single-token kind spelling.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        bloq_graph::BlockKind::from_str(text)
            .map(Into::into)
            .map_err(errors::invalid_argument)
    }

    /// `True` if this is a cube block.
    #[getter]
    fn is_cube(&self) -> bool {
        self.0.is_cube()
    }

    /// `True` if this is a walking (moving) block.
    #[getter]
    fn is_walking(&self) -> bool {
        self.0.is_walking()
    }

    /// `True` if this is a patch-rotation block.
    #[getter]
    fn is_patch_rotation(&self) -> bool {
        self.0.is_patch_rotation()
    }

    /// `True` if this is the Y-basis block.
    #[getter]
    fn is_y(&self) -> bool {
        self.0.is_y()
    }

    /// `True` if this is a fixed terminal measurement block.
    #[getter]
    fn is_measurement(&self) -> bool {
        self.0.is_measurement()
    }

    /// `True` if this is an open port block.
    #[getter]
    fn is_port(&self) -> bool {
        self.0.is_port()
    }

    /// `True` if this is a T block.
    #[getter]
    fn is_t(&self) -> bool {
        self.0.is_t()
    }

    /// `True` if this is a selective block.
    #[getter]
    fn is_selective(&self) -> bool {
        self.0.is_selective()
    }

    /// `True` for kinds whose behavior is resolved at runtime (T, selective).
    #[getter]
    fn is_dynamic(&self) -> bool {
        self.0.is_dynamic()
    }

    /// `True` if this kind is Clifford (everything except T).
    #[getter]
    fn is_clifford(&self) -> bool {
        self.0.is_clifford()
    }

    /// The `(x, y, z)` face bases, when this kind has a fixed basis assignment.
    #[getter]
    fn bases(&self) -> Option<(PyBasis, PyBasis, PyBasis)> {
        self.0
            .bases()
            .map(|[x, y, z]| (x.into(), y.into(), z.into()))
    }

    /// The `(x, y)` in-plane step of a walking or patch-rotation kind, as
    /// passed to `walking` / `patch_rotation`; `None` for kinds that do not
    /// move.
    ///
    /// Examples:
    ///     >>> from bloq import BlockKind
    ///     >>> BlockKind.walking("ZXZ", (1, 0)).movement
    ///     (1, 0)
    ///     >>> BlockKind.cube("ZXZ").movement is None
    ///     True
    #[getter]
    fn movement(&self) -> Option<(i32, i32)> {
        match self.0 {
            bloq_graph::BlockKind::Walking(kind) => Some(kind.movement().into()),
            bloq_graph::BlockKind::PatchRotation(kind) => Some(kind.movement().into()),
            _ => None,
        }
    }

    /// The X or Z basis of a fixed measurement block; `None` for other kinds.
    #[getter]
    fn measurement_basis(&self) -> Option<PyBasis> {
        match self.0 {
            bloq_graph::BlockKind::Measurement(basis) => Some(basis.into()),
            _ => None,
        }
    }

    /// Returns a new kind with X/Z bases flipped.
    fn flip_xz_basis(&self) -> Self {
        self.0.flip_xz_basis().into()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// Built from the kind's BLOG spelling plus the movement of a moving
    /// kind, whose spelling alone is just `Walking` / `PatchRotation`. The
    /// Rust `Debug` would instead leak `WalkingKind` and `IVec2` into a
    /// public repr; read the boundary bases back with `bases`.
    fn __repr__(&self) -> String {
        match self.movement() {
            Some((x, y)) => format!("<BlockKind {} movement=({x}, {y})>", self.0),
            None => format!("<BlockKind {}>", self.0),
        }
    }
}

// ==============================================================================
// Block
// ==============================================================================

/// A block placed at a 3D lattice position.
///
/// The `kind` argument accepts a `BlockKind` or its BLOG spelling (`"ZXZ"`,
/// `"T"`, ...).
///
/// Examples:
///     >>> from bloq import Block
///     >>> b = Block((0, 1, 2), "ZXZ")
///     >>> b.pos
///     (0, 1, 2)
///     >>> str(b.kind)
///     'ZXZ'
#[gen_stub_pyclass]
#[pyclass(
    name = "Block",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyBlock(pub(crate) bloq_graph::Block);

impl From<bloq_graph::Block> for PyBlock {
    fn from(b: bloq_graph::Block) -> Self {
        PyBlock(b)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlock {
    /// `Block((0, 0, 0), "ZXZ")` — kind accepts a `BlockKind` or its BLOG
    /// spelling. `height` is only valid for cube kinds; spatial Ports must set
    /// `role` to `"input"`, `"output"`, or `"multiplex"` before validation.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    ///     kind: A `BlockKind` or its BLOG spelling (`"ZXZ"`, `"T"`, ...).
    ///     height: The cube's height as a `k*d + n` expression of the code
    ///         distance, in the same spelling BLOG's `height=` takes (`"d"`,
    ///         `"2d"`, `"d/2"`, `"3d+2"`). Only valid for cube kinds.
    ///         Defaults to `"d"`.
    ///     tag: An optional tag without whitespace, controls, `<`, or `>`;
    ///         an empty string means no tag.
    ///     role: A Port's `"auto"`, `"input"`, `"output"`, or `"multiplex"` role
    ///         (case-insensitive). Defaults to `"auto"`; non-Ports only accept
    ///         that default.
    ///
    /// Raises:
    ///     InvalidArgumentError: If an argument is invalid for the block kind.
    #[new]
    #[pyo3(signature = (pos, kind, height="d", tag=None, role="auto"))]
    fn new(
        pos: (i32, i32, i32),
        kind: BlockKindLike,
        height: &str,
        tag: Option<String>,
        role: &str,
    ) -> PyResult<Self> {
        let mut block = bloq_graph::Block::new(pos, bloq_graph::BlockKind::try_from(kind)?);
        let height: bloq_graph::CubeHeight = height.parse().map_err(errors::invalid_argument)?;
        if !height.is_default() {
            block = block
                .with_height(height)
                .map_err(errors::invalid_argument)?;
        }
        if let Some(tag) = tag {
            block = block.with_tag(tag).map_err(errors::invalid_argument)?;
        }
        let role = bloq_graph::PortRole::from_str(role).map_err(errors::invalid_argument)?;
        if role != bloq_graph::PortRole::Auto {
            block = block
                .with_port_role(role)
                .map_err(errors::invalid_argument)?;
        }
        Ok(PyBlock(block))
    }

    /// The block's `(x, y, z)` lattice position.
    #[getter]
    fn pos(&self) -> (i32, i32, i32) {
        self.0.pos().into()
    }

    /// The block's kind.
    #[getter]
    fn kind(&self) -> PyBlockKind {
        self.0.kind().into()
    }

    /// The block's height as a `k*d + n` expression of the code distance, in
    /// the canonical BLOG spelling (`"d"`, `"2d"`, `"d/2"`, `"3d+2"`).
    #[getter]
    fn height(&self) -> String {
        self.0.height().to_string()
    }

    /// The block's tag string, or `None`.
    #[getter]
    fn tag(&self) -> Option<String> {
        self.0.tag().map(str::to_owned)
    }

    /// The Port's `"auto"`, `"input"`, `"output"`, or `"multiplex"` role; `None` for
    /// non-Ports.
    #[getter]
    fn role(&self) -> Option<String> {
        self.0.port_role().map(|role| role.as_str().to_owned())
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Block {}>", self.0)
    }
}

// ==============================================================================
// Pipe
// ==============================================================================

/// A pipe (lattice-surgery connection) from a source block along a direction.
///
/// The destination is one lattice step from `src` along `direction`, so a pipe
/// is fully described by its source and direction. The `direction` argument
/// accepts a `Direction` or its BLOG spelling (`"+Z"`, ...).
///
/// Examples:
///     >>> from bloq import Pipe
///     >>> p = Pipe((0, 0, 0), "+Z")
///     >>> p.src, p.dst
///     ((0, 0, 0), (0, 0, 1))
///     >>> str(p.direction)
///     '+Z'
#[gen_stub_pyclass]
#[pyclass(name = "Pipe", module = "bloq._core", frozen, eq, hash, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyPipe(pub(crate) bloq_graph::Pipe);

impl From<bloq_graph::Pipe> for PyPipe {
    fn from(p: bloq_graph::Pipe) -> Self {
        PyPipe(p)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipe {
    /// `Pipe((0, 0, 0), "+Z")` — direction accepts a `Direction` or its BLOG
    /// spelling.
    ///
    /// Args:
    ///     src: The source block position `(x, y, z)`.
    ///     direction: A `Direction` or its BLOG spelling (`"+Z"`, ...); the
    ///         destination is one lattice step from `src` along it.
    ///     hadamard: Whether the pipe applies a Hadamard across the seam.
    ///         Defaults to `False`.
    ///     tag: An optional tag without whitespace, controls, `<`, or `>`;
    ///         an empty string means no tag.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the destination overflows, `direction` is
    ///         invalid, or `tag` is malformed.
    #[new]
    #[pyo3(signature = (src, direction, hadamard=false, tag=None))]
    fn new(
        src: (i32, i32, i32),
        direction: DirectionLike,
        hadamard: bool,
        tag: Option<String>,
    ) -> PyResult<Self> {
        let mut pipe = bloq_graph::Pipe::new(src, bloq_utils::Direction::try_from(direction)?);
        pipe.try_dst().map_err(errors::invalid_argument)?;
        if hadamard {
            pipe = pipe.with_hadamard();
        }
        if let Some(tag) = tag {
            pipe = pipe.with_tag(tag).map_err(errors::invalid_argument)?;
        }
        Ok(PyPipe(pipe))
    }

    /// The source block position `(x, y, z)`.
    #[getter]
    fn src(&self) -> (i32, i32, i32) {
        self.0.src().into()
    }

    /// The destination block position `(x, y, z)`.
    #[getter]
    fn dst(&self) -> (i32, i32, i32) {
        self.0.dst().into()
    }

    /// The direction the pipe leaves its source block.
    #[getter]
    fn direction(&self) -> PyDirection {
        self.0.dir().into()
    }

    /// `True` if the pipe applies a Hadamard across the seam.
    #[getter]
    fn hadamard(&self) -> bool {
        self.0.is_hadamard()
    }

    /// The pipe's tag string, or `None`.
    #[getter]
    fn tag(&self) -> Option<String> {
        self.0.tag().map(str::to_owned)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Pipe {}>", self.0)
    }
}

// ==============================================================================
// Stabilizer / StabilizerGenerator (read-only views)
// ==============================================================================

/// Accepts an existing block or a compact position/kind pair.
#[derive(FromPyObject)]
pub(crate) enum ArmBlockLike {
    Block(PyBlock),
    Pair((PosTuple, BlockKindLike)),
}
pyo3_stub_gen::impl_stub_type!(ArmBlockLike = PyBlock | (PosTuple, BlockKindLike));

/// Geometry for one branch arm, using coordinates in the parent graph.
/// Blocks accept `bloq.Block` objects or `((x, y, z), kind)` pairs. Pipes use
/// `bloq.Pipe` objects. External seams are passed once to `add_branches`.
/// Geometry is checked when attached to its parent.
///
/// Examples:
///     >>> import bloq
///     >>> arm = bloq.BranchArm([((0, 0, 1), "Z")])
///     >>> arm.blocks()[0].pos
///     (0, 0, 1)
#[gen_stub_pyclass]
#[pyclass(name = "BranchArm", module = "bloq._core", frozen, eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyBranchArm(pub(crate) bloq_graph::BranchArm);

#[gen_stub_pymethods]
#[pymethods]
impl PyBranchArm {
    /// Creates an arm from blocks and optional internal pipes.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a block-kind spelling is invalid.
    #[new]
    #[pyo3(signature = (blocks, *, pipes=None))]
    fn new(blocks: Vec<ArmBlockLike>, pipes: Option<Vec<PyPipe>>) -> PyResult<Self> {
        let blocks = blocks
            .into_iter()
            .map(|block| match block {
                ArmBlockLike::Block(block) => Ok(block.0),
                ArmBlockLike::Pair((position, kind)) => Ok(bloq_graph::Block::new(
                    position,
                    bloq_graph::BlockKind::try_from(kind)?,
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self(bloq_graph::BranchArm::new(
            blocks,
            pipes
                .unwrap_or_default()
                .into_iter()
                .map(|pipe| pipe.0)
                .collect(),
        )))
    }

    /// Arm blocks in canonical position order.
    fn blocks(&self) -> Vec<PyBlock> {
        self.0.blocks().cloned().map(PyBlock).collect()
    }

    /// Arm pipes in canonical endpoint order.
    fn pipes(&self) -> Vec<PyPipe> {
        self.0.pipes().cloned().map(PyPipe).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<BranchArm {} blocks, {} pipes>",
            self.0.blocks().count(),
            self.0.pipes().count()
        )
    }
}

/// A named branch with a corrected selector and two explicitly authored arms.
/// Use `BlockGraph.add_branches` to attach it with its shared seams and resolve.
#[gen_stub_pyclass]
#[pyclass(name = "Branch", module = "bloq._core", frozen, eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyBranch(pub(crate) bloq_graph::Branch);

#[gen_stub_pymethods]
#[pymethods]
impl PyBranch {
    /// Creates a branch. A string condition names an existing classical value.
    #[new]
    #[pyo3(signature = (name, condition, *, on_false, on_true))]
    fn new(name: String, condition: ExprLike, on_false: PyBranchArm, on_true: PyBranchArm) -> Self {
        Self(bloq_graph::Branch::new(
            name,
            condition.into(),
            on_false.0,
            on_true.0,
        ))
    }

    /// Public branch name.
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    /// Corrected expression selecting the true arm.
    #[getter]
    fn condition(&self) -> PyExpr {
        PyExpr(self.0.condition.clone())
    }

    /// Geometry selected when the condition is false.
    #[getter]
    fn on_false(&self) -> PyBranchArm {
        PyBranchArm(self.0.on_false.clone())
    }

    /// Geometry selected when the condition is true.
    #[getter]
    fn on_true(&self) -> PyBranchArm {
        PyBranchArm(self.0.on_true.clone())
    }

    fn __repr__(&self) -> String {
        format!("<Branch {} if {}>", self.0.name, self.0.condition)
    }
}

/// A stabilizer of the block graph's ZX diagram (read-only view).
///
/// Rows come from `BlockGraph.stabilizers()` (as `StabilizerGenerator.stabilizer`)
/// and `fill_ports_auto()`. The operator itself is `paulis`, a dense Pauli
/// string over the ZX graph's combined node+edge id space; the same support is
/// also broken out by graph element: `port_stabilizer` maps open-port positions
/// to Paulis (how the stabilizer meets the outside world), while
/// `interior_nodes` / `interior_edges` cover spiders and edges inside the
/// diagram. Row role and measurement identity live on `StabilizerGenerator`.
#[gen_stub_pyclass]
#[pyclass(name = "Stabilizer", module = "bloq._core", frozen, eq, from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyStabilizer(pub(crate) bloq_graph::Stabilizer);

impl From<bloq_graph::Stabilizer> for PyStabilizer {
    fn from(s: bloq_graph::Stabilizer) -> Self {
        PyStabilizer(s)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStabilizer {
    /// The dense Pauli string over the ZX graph's combined node+edge id space.
    #[getter]
    fn paulis(&self) -> PyPauliString {
        self.0.paulis.clone().into()
    }

    /// Pauli support on external port positions, in ascending position order.
    #[getter]
    fn port_stabilizer(&self) -> BTreeMap<(i32, i32, i32), PyPauli> {
        self.0
            .port_stabilizer
            .iter()
            .map(|(pos, p)| ((*pos).into(), PyPauli::from(*p)))
            .collect()
    }

    /// Pauli support on interior node positions, in ascending position order.
    #[getter]
    fn interior_nodes(&self) -> BTreeMap<(i32, i32, i32), PyPauli> {
        self.0
            .interior_nodes
            .iter()
            .map(|(pos, p)| ((*pos).into(), PyPauli::from(*p)))
            .collect()
    }

    /// Pauli support on interior edges, keyed by endpoint-position pairs in
    /// ascending order.
    #[getter]
    fn interior_edges(&self) -> BTreeMap<(PosTuple, PosTuple), PyPauli> {
        self.0
            .interior_edges
            .iter()
            .map(|((u, v), p)| (((*u).into(), (*v).into()), PyPauli::from(*p)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("<Stabilizer paulis=\"{}\">", self.0.paulis)
    }
}

/// A stabilizer generator row: a stabilizer plus its role.
///
/// One correlation surface returned by `BlockGraph.stabilizers()` or
/// `fill_ports_auto()`. The `kind` says what the row does: `"measurement"`
/// rows read a named measurement variable, and `"logical"` rows are output
/// frames or closed logical surfaces.
#[gen_stub_pyclass]
#[pyclass(
    name = "StabilizerGenerator",
    module = "bloq._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyStabilizerGenerator(pub(crate) bloq_graph::StabilizerGenerator);

impl From<bloq_graph::StabilizerGenerator> for PyStabilizerGenerator {
    fn from(g: bloq_graph::StabilizerGenerator) -> Self {
        PyStabilizerGenerator(g)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStabilizerGenerator {
    /// The underlying stabilizer.
    #[getter]
    fn stabilizer(&self) -> PyStabilizer {
        self.0.stabilizer.clone().into()
    }

    /// The row's role: `"measurement"` or `"logical"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0.kind {
            bloq_graph::StabilizerRowKind::Measurement { .. } => "measurement",
            bloq_graph::StabilizerRowKind::Logical => "logical",
        }
    }

    /// `True` if this row witnesses a measurement variable.
    #[getter]
    fn is_measurement(&self) -> bool {
        self.0.is_measurement()
    }

    /// The measurement variable name, for `"measurement"` rows.
    #[getter]
    fn measurement_name(&self) -> Option<String> {
        match &self.0.kind {
            bloq_graph::StabilizerRowKind::Measurement { name } => Some(name.clone()),
            bloq_graph::StabilizerRowKind::Logical => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<StabilizerGenerator kind=\"{}\" paulis=\"{}\">",
            self.kind(),
            self.0.stabilizer.paulis
        )
    }
}

// ==============================================================================
// BlockGraph
// ==============================================================================

/// A graph of surface code blocks connected by pipes.
///
/// Blocks sit at integer lattice positions; pipes are lattice-surgery
/// connections between adjacent blocks. Build one incrementally with
/// `add_block` / `add_pipe`, load a ready-made graph from `GalleryItem`, or parse
/// `.blog` text with `parse_blog`.
///
/// Examples:
///     >>> from bloq import BlockGraph, Block, Pipe
///     >>> graph = BlockGraph()
///     >>> graph.add_block(Block((0, 0, 0), "ZXZ"))
///     (0, 0, 0)
///     >>> graph.add_block(Block((0, 0, 1), "ZXZ"))
///     (0, 0, 1)
///     >>> graph.add_pipe(Pipe((0, 0, 0), "+Z"))
///     >>> graph.block_count, graph.pipe_count
///     (2, 1)
///     >>> graph.validate()
#[gen_stub_pyclass]
#[pyclass(name = "BlockGraph", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyBlockGraph(pub(crate) bloq_graph::BlockGraph);

impl From<bloq_graph::BlockGraph> for PyBlockGraph {
    fn from(g: bloq_graph::BlockGraph) -> Self {
        PyBlockGraph(g)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlockGraph {
    /// Creates an empty block graph.
    ///
    /// Examples:
    ///     >>> from bloq import BlockGraph, Block, Pipe
    ///     >>> graph = BlockGraph()
    ///     >>> graph.is_empty
    ///     True
    ///     >>> graph.add_block(Block((0, 0, 0), "ZXZ"))
    ///     (0, 0, 0)
    ///     >>> graph.add_block(Block((0, 0, 1), "ZXZ"))
    ///     (0, 0, 1)
    ///     >>> graph.add_pipe(Pipe((0, 0, 0), "+Z"))
    ///     >>> graph.block_count
    ///     2
    #[new]
    fn new() -> Self {
        PyBlockGraph(bloq_graph::BlockGraph::new())
    }

    /// Copies a graph into a reusable module definition.
    /// Omitted interface arguments retain the body's existing declarations.
    /// Quantum port dictionaries retain insertion order; positions name local
    /// Port blocks. Supplying inputs or outputs replaces the quantum interface.
    /// Child instances and connections are retained. Validate the completed
    /// hierarchy with `from_definitions`.
    ///
    /// Args:
    ///     name: Definition name; use `main` for the executable root.
    ///     body: Graph containing local geometry, actions, and child instances.
    ///     inputs: Names and positions of quantum input Ports.
    ///     outputs: Names and positions of quantum output Ports.
    ///     bit_inputs: Classical names supplied by the parent.
    ///     bit_outputs: Exported expressions or local/child value names.
    ///     actions: Optional replacement actions, using the declared bit inputs.
    ///     resources: Quantum port resource types; unspecified ports use `data`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a resource names an undeclared port.
    ///     BlockGraphError: If classical input declarations are invalid.
    #[staticmethod]
    #[pyo3(signature = (name, body, *, inputs=None, outputs=None, bit_inputs=None, bit_outputs=None, actions=None, resources=None))]
    #[allow(clippy::too_many_arguments, reason = "named Python interface fields")]
    fn definition(
        name: String,
        body: Self,
        #[gen_stub(override_type(type_repr = "dict[str, tuple[int, int, int]] | None"))]
        inputs: Option<&Bound<'_, PyDict>>,
        #[gen_stub(override_type(type_repr = "dict[str, tuple[int, int, int]] | None"))]
        outputs: Option<&Bound<'_, PyDict>>,
        bit_inputs: Option<Vec<String>>,
        #[gen_stub(override_type(type_repr = "dict[str, Expr | str] | None"))] bit_outputs: Option<
            &Bound<'_, PyDict>,
        >,
        actions: Option<Vec<PyAction>>,
        resources: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        use bloq_graph::{BitOutput, PortDirection, QuantumPort};
        let mut body = body.0;
        let mut interface = body.interface.clone();
        if inputs.is_some() || outputs.is_some() {
            interface.quantum_ports.clear();
            for (ports, direction) in [
                (inputs, PortDirection::Input),
                (outputs, PortDirection::Output),
            ] {
                if let Some(ports) = ports {
                    for (name, position) in ports.iter() {
                        interface.quantum_ports.push(QuantumPort {
                            name: name.extract()?,
                            position: position.extract::<(i32, i32, i32)>()?.into(),
                            direction,
                            resource_type: "data".into(),
                        });
                    }
                }
            }
        }
        for (name, resource) in resources.unwrap_or_default() {
            let port = interface
                .quantum_ports
                .iter_mut()
                .find(|port| port.name == name)
                .ok_or_else(|| {
                    errors::InvalidArgumentError::new_err(format!("unknown quantum port '{name}'"))
                })?;
            port.resource_type = resource;
        }
        if let Some(inputs) = bit_inputs {
            interface.bit_inputs = inputs;
        }
        if let Some(outputs) = bit_outputs {
            interface.bit_outputs = outputs
                .iter()
                .map(|(name, expr)| {
                    Ok(BitOutput {
                        name: name.extract()?,
                        expr: expr.extract::<ExprLike>()?.into(),
                    })
                })
                .collect::<PyResult<_>>()?;
        }
        // A copied definition is analyzed as a standalone local body before
        // receiving its reusable name, just like extract_definition.
        body.name = bloq_graph::BlockGraph::ENTRY_MODULE.into();
        let actions = actions.map_or_else(
            || body.actions(),
            |actions| actions.into_iter().map(|action| action.0).collect(),
        );
        let mut action_inputs = interface.bit_inputs.clone();
        action_inputs.extend(actions.iter().flat_map(|action| {
            action
                .referenced_names()
                .into_iter()
                .filter(|name| name.contains('.'))
                .map(str::to_owned)
        }));
        body.set_actions_with_inputs(actions, action_inputs)
            .map_err(graph_err)?;
        let instances = std::mem::take(&mut body.instances);
        let connections = std::mem::take(&mut body.quantum_connections);
        let bindings = std::mem::take(&mut body.bit_bindings);
        Ok(bloq_graph::BlockGraph::definition(
            name,
            body,
            interface,
            instances,
            connections,
            bindings,
        )
        .into())
    }

    /// Assembles module definitions into a validated hierarchy rooted at `main`.
    /// Each definition is stored once; input graphs remain independent.
    /// Definitions must be local graphs without their own helper libraries.
    ///
    /// Args:
    ///     definitions: Reusable definitions and one executable `main`.
    ///     limits: Source-analysis and expansion budget overrides.
    ///
    /// Raises:
    ///     BlockGraphError: If names, interfaces, seams, geometry, or dependencies are invalid.
    #[staticmethod]
    #[pyo3(signature = (definitions, *, limits=None))]
    fn from_definitions(
        definitions: Vec<Self>,
        limits: Option<HashMap<String, Option<usize>>>,
    ) -> PyResult<Self> {
        bloq_graph::BlockGraph::from_definitions_with_limits(
            definitions
                .into_iter()
                .map(|definition| definition.0)
                .collect(),
            crate::compile::resolve_limits(limits)?,
        )
        .map(Into::into)
        .map_err(|error| graph_err(bloq_graph::BlockGraphError::ModuleSource(error.into())))
    }

    /// Places a named child definition, rotating before translating.
    /// The completed hierarchy checks names, geometry, and supported orientations.
    ///
    /// Args:
    ///     name: Instance name in this parent.
    ///     definition: Reusable definition name.
    ///     translation: Placement in the parent's lattice coordinates.
    ///     rotation: Optional axis and angle in degrees, such as `("Z", 180)`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the axis or rotation angle is invalid.
    #[pyo3(signature = (name, definition, translation=(0, 0, 0), *, rotation=None))]
    fn add_instance(
        &mut self,
        name: String,
        definition: String,
        translation: PosTuple,
        rotation: Option<(UDirectionLike, i32)>,
    ) -> PyResult<()> {
        let rotation = match rotation {
            Some((axis, degrees)) => bloq_graph::ModuleRotation::from_degrees(
                axis.try_into()?,
                degrees,
            )
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err("rotation must be a multiple of 90 degrees")
            })?,
            None => bloq_graph::ModuleRotation::IDENTITY,
        };
        self.0.instances.push(bloq_graph::ModuleInstance {
            name,
            definition,
            rotation,
            translation: translation.into(),
        });
        Ok(())
    }

    /// Places a module instance and exposes its quantum and classical interface.
    /// With no translation, places it beside the current geometry.
    /// The graph must contain the named definition; input graphs remain independent.
    ///
    /// Args:
    ///     name: New instance name.
    ///     definition: Definition already stored in this graph.
    ///     translation: Optional explicit lattice placement.
    ///     rotation: Optional axis and angle, such as `("Z", 180)`.
    ///
    /// Returns:
    ///     The instance's translation.
    ///
    /// Raises:
    ///     BlockGraphError: If placement, names, or interfaces are invalid; the graph is unchanged.
    #[pyo3(signature = (name, definition, translation=None, *, rotation=None))]
    fn place_module(
        &mut self,
        name: &str,
        definition: &str,
        translation: Option<PosTuple>,
        rotation: Option<(UDirectionLike, i32)>,
    ) -> PyResult<PosTuple> {
        let rotation = match rotation {
            Some((axis, degrees)) => bloq_graph::ModuleRotation::from_degrees(
                axis.try_into()?,
                degrees,
            )
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err("rotation must be a multiple of 90 degrees")
            })?,
            None => bloq_graph::ModuleRotation::IDENTITY,
        };
        self.0
            .place_module_with(name, definition, translation.map(Into::into), rotation)
            .map(Into::into)
            .map_err(graph_err)
    }

    /// Aligns and docks named child ports, joining all other touching compatible ports.
    /// The input instance's connected group moves together. Endpoint Hadamards
    /// are preserved; obstructed compaction retains connection cubes.
    ///
    /// Args:
    ///     output: Child output name, such as `child0.control_out`.
    ///     input: Child input name, such as `child1.control_in`.
    ///     align: Align the input group's selected port to the output.
    ///     compact: Prefer direct seams over parent connection cubes.
    ///     hadamard: Extra basis change at the selected seam.
    ///
    /// Returns:
    ///     Number of port pairs joined.
    ///
    /// Raises:
    ///     BlockGraphError: If ports, types, faces, or placement are invalid; the graph is unchanged.
    #[pyo3(signature = (output, input, *, align=true, compact=true, hadamard=false))]
    fn connect_modules(
        &mut self,
        output: &str,
        input: &str,
        align: bool,
        compact: bool,
        hadamard: bool,
    ) -> PyResult<usize> {
        self.0
            .connect_modules_with(
                output,
                input,
                bloq_graph::ModuleJoinOptions {
                    align,
                    compact,
                    hadamard,
                },
            )
            .map_err(graph_err)
    }

    /// Binds a parent value or child bit output to a child's classical input.
    /// Replaces the current binding and removes an unused exposed parent input.
    ///
    /// Raises:
    ///     BlockGraphError: If values or dependencies are invalid; the graph is unchanged.
    fn bind_modules(&mut self, source: &str, target: &str) -> PyResult<()> {
        self.0.bind_modules(source, target).map_err(graph_err)
    }

    /// Translates a connected group and joins touching compatible free ports.
    /// Returns the applied offset, including any successful compaction.
    ///
    /// Raises:
    ///     BlockGraphError: If geometry or interfaces are invalid; the graph is unchanged.
    #[pyo3(signature = (name, offset, *, compact=true))]
    fn translate_module(
        &mut self,
        name: &str,
        offset: PosTuple,
        compact: bool,
    ) -> PyResult<PosTuple> {
        self.0
            .translate_module(name, offset.into(), compact)
            .map(Into::into)
            .map_err(graph_err)
    }

    /// Connects a child quantum output to an input or a parent-local block.
    /// Endpoints are local position tuples or strings such as `gate.q_out`.
    /// At least one endpoint must belong to an instance. Connections do not route
    /// geometry; the completed hierarchy checks placement and boundary bases.
    ///
    /// Raises:
    ///     InvalidArgumentError: If endpoint syntax is invalid or both endpoints are local.
    #[pyo3(signature = (source, target, *, hadamard=false))]
    fn connect(
        &mut self,
        source: ConnectionEndpoint,
        target: ConnectionEndpoint,
        hadamard: bool,
    ) -> PyResult<()> {
        use bloq_graph::QuantumConnection;
        let connection = match (source, target) {
            (ConnectionEndpoint::Block(block), ConnectionEndpoint::Port(input)) => {
                QuantumConnection::Input {
                    block: block.into(),
                    input: instance_port(&input)?,
                    hadamard,
                }
            }
            (ConnectionEndpoint::Port(output), ConnectionEndpoint::Block(block)) => {
                QuantumConnection::Output {
                    output: instance_port(&output)?,
                    block: block.into(),
                    hadamard,
                }
            }
            (ConnectionEndpoint::Port(output), ConnectionEndpoint::Port(input)) => {
                QuantumConnection::Pipe {
                    output: instance_port(&output)?,
                    input: instance_port(&input)?,
                    hadamard,
                }
            }
            _ => {
                return Err(errors::InvalidArgumentError::new_err(
                    "use add_pipe for two parent-local blocks",
                ));
            }
        };
        self.0.quantum_connections.push(connection);
        Ok(())
    }

    /// Supplies a parent value or child bit output to a child classical input.
    /// For example, `bind("read.result", "correct.flip")` creates a dependency
    /// between those instances. The completed hierarchy validates the binding.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a qualified endpoint is malformed.
    fn bind(&mut self, source: String, target: String) -> PyResult<()> {
        let target = instance_port(&target)?;
        let source = if source.contains('.') {
            let source = instance_port(&source)?;
            bloq_graph::BitRef {
                instance: Some(source.instance),
                bit: source.port,
            }
        } else {
            bloq_graph::BitRef {
                instance: None,
                bit: source,
            }
        };
        self.0.bit_bindings.push(bloq_graph::BitBinding {
            source,
            target_instance: target.instance,
            target_bit: target.port,
        });
        Ok(())
    }

    /// Attaches branches, shared seams, and additional actions in one edit.
    /// Each branch supplies its own resolve condition. Existing actions remain.
    /// The graph is unchanged if geometry or action validation fails.
    ///
    /// Args:
    ///     branches: Named branches with explicit false and true arms.
    ///     pipes: Shared pipes joining arm boundaries to the common graph or
    ///         other branches. They are stored once, outside the arms.
    ///     actions: Additional actions, including measurements used by selectors.
    ///
    /// Raises:
    ///     BlockGraphError: If geometry, interfaces, names, or actions are invalid.
    #[pyo3(signature = (branches, *, pipes=None, actions=None))]
    fn add_branches(
        &mut self,
        branches: Vec<PyBranch>,
        pipes: Option<Vec<PyPipe>>,
        actions: Option<Vec<PyAction>>,
    ) -> PyResult<()> {
        self.0
            .try_add_branches(
                branches.into_iter().map(|branch| branch.0),
                pipes.unwrap_or_default().into_iter().map(|pipe| pipe.0),
                actions
                    .unwrap_or_default()
                    .into_iter()
                    .map(|action| action.0),
            )
            .map_err(graph_err)
    }

    /// Parses `.blog` text into a block graph, retaining module definitions,
    /// instances, interfaces, and connections. This is the same native
    /// constructor as Rust's `BlockGraph::from_text`.
    /// Use `load()` to resolve imports relative to a file.
    ///
    /// Args:
    ///     text: The `.blog` source text.
    ///     limits: Source-analysis and module-expansion budget overrides.
    ///         Use non-negative counts or `None` for unlimited; compilation
    ///         overrides are passed separately to `compile`.
    ///
    /// Raises:
    ///     ParseError: If `text` is not valid `.blog`.
    ///     InvalidArgumentError: If a limit field is unknown.
    ///     OverflowError: If a limit count is negative or exceeds `usize`.
    #[staticmethod]
    #[pyo3(signature = (text, *, limits=None))]
    fn from_text(text: &str, limits: Option<HashMap<String, Option<usize>>>) -> PyResult<Self> {
        parse_blog(text, crate::compile::resolve_limits(limits)?)
    }

    /// Loads a `.blog` file through Rust's native `BlockGraph::load` constructor,
    /// resolving module imports relative to each file.
    ///
    /// Like `from_text`, retains the module hierarchy. Both simple and
    /// hierarchical graphs are accepted by `compile`.
    ///
    /// Args:
    ///     path: A filesystem path, including `pathlib.Path` objects.
    ///     limits: Source-analysis and module-expansion budget overrides.
    ///         Use non-negative counts or `None` for unlimited; the same
    ///         limits apply to each imported hierarchy.
    ///
    /// Raises:
    ///     OSError: If the root file or an imported file cannot be read.
    ///     ParseError: If BLOG parsing or module resolution fails.
    ///     BlockGraphError: If graph structure is invalid.
    ///     InvalidArgumentError: If a limit field is unknown.
    ///     OverflowError: If a limit count is negative or exceeds `usize`.
    #[staticmethod]
    #[pyo3(signature = (path, *, limits=None))]
    fn load(path: PathBuf, limits: Option<HashMap<String, Option<usize>>>) -> PyResult<Self> {
        let limits = crate::compile::resolve_limits(limits)?;
        bloq_graph::BlockGraph::load_with_limits(&path, limits)
            .map(Into::into)
            .map_err(|error| {
                // Root syntax/lowering failures retain source spans. Read text
                // only on that failure path to render its original diagnostic;
                // imported source errors retain their resolver diagnostics.
                let source = matches!(error, bloq_graph::BlockGraphError::Parse(_))
                    .then(|| std::fs::read_to_string(&path).ok())
                    .flatten();
                graph_input_err(error, source.as_deref())
            })
    }

    /// Serializes the graph to `.blog` text (BLOG 1.0).
    fn to_text(&self) -> String {
        self.0.to_blog_text()
    }

    /// Writes the graph to a `.blog` file.
    ///
    /// Args:
    ///     path: Filesystem path to write the `.blog` file to.
    ///
    /// Raises:
    ///     OSError: If the file cannot be written.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.0.to_file(path).map_err(graph_err)
    }

    /// Whether the graph declares module interfaces, instances, or definitions.
    /// A leaf with an explicit quantum or classical interface also returns true.
    #[getter]
    fn has_module_structure(&self) -> bool {
        self.0.has_module_structure()
    }

    /// Names of all graph definitions, including the root.
    #[getter]
    fn module_names(&self) -> Vec<String> {
        self.0.modules().map(|module| module.name.clone()).collect()
    }

    /// Extracts a named definition as an independent graph, retaining its
    /// reachable helper definitions and naming its executable root `main`.
    /// The copy can be edited, saved, pickled, and compiled independently.
    ///
    /// Args:
    ///     name: The definition name.
    ///     limits: Source-analysis and expansion budget overrides.
    ///
    /// Raises:
    ///     KeyError: If the definition is absent.
    ///     BlockGraphError: If its independent hierarchy cannot be constructed.
    #[pyo3(signature = (name, *, limits=None))]
    fn module(&self, name: &str, limits: Option<HashMap<String, Option<usize>>>) -> PyResult<Self> {
        if self.0.module(name).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(name.to_owned()));
        }
        let limits = crate::compile::resolve_limits(limits)?;
        self.0
            .extract_definition_with_limits(name, limits)
            .map(Into::into)
            .map_err(|error| {
                errors::BlockGraphError::new_err(crate::compile::compile_error_message(
                    &error.into(),
                ))
            })
    }

    /// Expands module instances into one independent graph for geometry,
    /// boundary filling, or flat source analysis. Compilation preserves the
    /// hierarchy and does not require this conversion.
    ///
    /// Args:
    ///     limits: Source-expansion budget overrides. Use non-negative counts
    ///         or `None` for unlimited; omitted fields retain their defaults.
    ///
    /// Raises:
    ///     BlockGraphError: If expansion cannot produce a valid flat graph.
    #[pyo3(signature = (*, limits=None))]
    fn flatten(
        &self,
        py: Python<'_>,
        limits: Option<HashMap<String, Option<usize>>>,
    ) -> PyResult<Self> {
        let limits = crate::compile::resolve_limits(limits)?;
        let graph = &self.0;
        py.detach(move || graph.flatten_with_limits(limits))
            .map(Into::into)
            .map_err(|error| {
                errors::BlockGraphError::new_err(crate::compile::compile_error_message(
                    &error.into(),
                ))
            })
    }

    /// Validates structural constraints and action semantics.
    ///
    /// Raises:
    ///     BlockGraphError: If a structural or action-semantic constraint is
    ///         violated.
    fn validate(&self) -> PyResult<()> {
        self.0.validate().map_err(graph_err)
    }

    /// Validates structural constraints only (skips action semantics).
    ///
    /// Raises:
    ///     BlockGraphError: If a structural constraint is violated.
    fn validate_structure(&self) -> PyResult<()> {
        self.0.validate_structure().map_err(graph_err)
    }

    /// Projects every structural branch to a static true or false arm.
    ///
    /// The returned graph has no `branch` actions and can be passed to
    /// `stabilizers()`. Every branch target must appear exactly once.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Args:
    ///     assignments: `(target_position, value)` pairs for every branch.
    ///
    /// Raises:
    ///     BlockGraphError: If a target is unknown, duplicated, or omitted, or
    ///         the branch regions are invalid.
    fn project_branches(&self, assignments: Vec<(PosTuple, bool)>) -> PyResult<Self> {
        self.0
            .project_branches(
                assignments
                    .into_iter()
                    .map(|(target, value)| (target.into(), value)),
            )
            .map(PyBlockGraph)
            .map_err(graph_err)
    }

    /// Returns an independent snapshot of the current source action DAG.
    ///
    /// Local structural edits refresh the cached DAG. `is_analyzed` says whether
    /// correlation support dependencies are present. For child instances, call
    /// `flatten()` explicitly to inspect the combined action program.
    fn action_graph(&self) -> PyActionDag {
        PyActionDag(self.0.action_graph().clone())
    }

    /// Derives an independent DAG including correlation support dependencies.
    ///
    /// Accepts local graphs and authored module hierarchies without changing
    /// the source. Hierarchies use the compiler's module linker and correlation
    /// composition. `owners()` identifies each action's definition and instance.
    /// Continuing branches use guarded causal readout planning.
    ///
    /// Args:
    ///     limits: Source-analysis budget overrides, as accepted by `from_text`.
    ///
    /// Raises:
    ///     BlockGraphError: If module validation, readout analysis or resource limits fail.
    ///     InvalidArgumentError: If a limit field is unknown.
    ///     OverflowError: If a limit count is negative or exceeds `usize`.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> source = bloq.GalleryItem.T_GATE.load()
    ///     >>> dag = source.analyze_action_graph()
    ///     >>> dag.is_analyzed
    ///     True
    #[pyo3(signature = (*, limits=None))]
    fn analyze_action_graph(
        &self,
        py: Python<'_>,
        limits: Option<HashMap<String, Option<usize>>>,
    ) -> PyResult<PyActionDag> {
        let limits = crate::compile::resolve_limits(limits)?;
        let source = &self.0;
        py.detach(move || source.analyze_action_graph_with_limits(limits))
            .map(PyActionDag)
            .map_err(graph_err)
    }

    /// The correlation surfaces of the displayed branch arms, from the causal
    /// readout plan that compilation uses: one row per named readout, output
    /// frame, and closed logical surface. Module hierarchies are composed.
    ///
    /// Raises:
    ///     BlockGraphError: If the source is invalid or a readout is unavailable.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> graph = bloq.GalleryItem.X_MEMORY.load()
    ///     >>> graph.stabilizers()[0].kind
    ///     'logical'
    fn stabilizers(&self, py: Python<'_>) -> PyResult<Vec<PyStabilizerGenerator>> {
        let source = &self.0;
        Ok(py
            .detach(move || source.stabilizers())
            .map_err(graph_err)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Builds compatible static boundary fills, returning
    /// `(closed_graph, stabilizer_generators)` pairs.
    ///
    /// Port, T, and Selective sites use full compatible boundary support.
    /// Variants retain selected measurement records and aliases, and remove
    /// feedback and resolves for filled caps.
    ///
    /// `filled_graphs()` is the same thing without the generators, for the
    /// common case of only wanting the closed graphs.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Raises:
    ///     BlockGraphError: If a boundary cannot be filled, or postselection
    ///         or a structural branch needs an omitted measurement record.
    fn fill_ports_auto(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(PyBlockGraph, Vec<PyStabilizerGenerator>)>> {
        let source = &self.0;
        Ok(py
            .detach(move || source.fill_ports_auto())
            .map_err(graph_err)?
            .into_iter()
            .map(|(g, generators)| {
                (
                    PyBlockGraph(g),
                    generators.into_iter().map(Into::into).collect(),
                )
            })
            .collect())
    }

    /// The graphs from `fill_ports_auto()`, without their selecting generators.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Raises:
    ///     BlockGraphError: If the ports cannot be filled.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> filled = bloq.GalleryItem.BELL_STATE.load().flatten().filled_graphs()
    ///     >>> all(not graph.is_open for graph in filled)
    ///     True
    fn filled_graphs(&self, py: Python<'_>) -> PyResult<Vec<PyBlockGraph>> {
        let source = &self.0;
        Ok(py
            .detach(move || source.filled_graphs())
            .map_err(graph_err)?
            .into_iter()
            .map(PyBlockGraph)
            .collect())
    }

    /// Resolves all selective blocks by randomly choosing a measurement basis.
    /// Returns `(resolved_graph, replacements)` where `replacements` maps each
    /// original selective block to its resolved block. Deterministic when
    /// `seed` is given.
    ///
    /// If any selective is resolved, the returned graph is structure-only and
    /// has no actions, including structural branches. Project branches first
    /// when a concrete conditional topology is required.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Args:
    ///     seed: Optional RNG seed; when given, resolution is deterministic.
    ///         Defaults to `None` (nondeterministic).
    #[pyo3(signature = (seed = None))]
    fn randomly_resolve_selectives(
        &self,
        seed: Option<u64>,
    ) -> PyResult<(PyBlockGraph, HashMap<PyBlock, PyBlock>)> {
        let seed = seed.unwrap_or_else(rand::random);
        let (resolved, replacements) = self
            .0
            .randomly_resolve_selectives(seed)
            .map_err(graph_err)?;
        Ok((
            PyBlockGraph(resolved),
            replacements
                .into_iter()
                .map(|(from, to)| (PyBlock(from), PyBlock(to)))
                .collect(),
        ))
    }

    // --- block CRUD -----------------------------------------------------------

    /// Adds a block, returning its position. Raises `BlockGraphError` if the
    /// position is occupied.
    ///
    /// Args:
    ///     block: The block to add.
    fn add_block(&mut self, block: PyBlock) -> PyResult<(i32, i32, i32)> {
        self.0
            .try_add_block(block.0)
            .map(Into::into)
            .map_err(graph_err)
    }

    /// Removes and returns the block at `pos`, or `None` if absent.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn remove_block(&mut self, pos: (i32, i32, i32)) -> Option<PyBlock> {
        self.0.remove_block(pos).map(Into::into)
    }

    /// Returns the block at `pos`, or `None` if absent.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn get_block(&self, pos: (i32, i32, i32)) -> Option<PyBlock> {
        self.0.get_block(pos).cloned().map(Into::into)
    }

    /// Returns the internal node ID for the block at `pos`, or `None`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn get_block_id(&self, pos: (i32, i32, i32)) -> Option<u32> {
        self.0.get_block_id(pos)
    }

    /// Returns `True` when `block` can be placed without overlap.
    ///
    /// Args:
    ///     block: The candidate block to test.
    fn can_place_block(&self, block: PyRef<'_, PyBlock>) -> bool {
        self.0.can_place_block(&block.0).is_ok()
    }

    /// Why `block` cannot be placed, or `None` when it can.
    ///
    /// The message `add_block` would have raised, without raising it — for
    /// callers reporting the reason rather than just refusing.
    ///
    /// Args:
    ///     block: The candidate block to test.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> graph = bloq.BlockGraph()
    ///     >>> graph.add_block(bloq.Block((0, 0, 0), "ZXZ"))
    ///     (0, 0, 0)
    ///     >>> graph.placement_error(bloq.Block((0, 0, 1), "ZXZ")) is None
    ///     True
    ///     >>> "occupied" in graph.placement_error(bloq.Block((0, 0, 0), "ZXZ"))
    ///     True
    fn placement_error(&self, block: PyRef<'_, PyBlock>) -> Option<String> {
        self.0
            .can_place_block(&block.0)
            .err()
            .map(|e| e.to_string())
    }

    /// `True` if a block occupies `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn has_block_at(&self, pos: (i32, i32, i32)) -> bool {
        self.0.has_block_at(pos)
    }

    /// Replaces the kind of the block at `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    ///     kind: A `BlockKind` or its BLOG spelling.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `kind` is an invalid spelling.
    ///     BlockGraphError: If no block occupies `pos`, or the new kind is
    ///         incompatible.
    fn set_block_kind(&mut self, pos: (i32, i32, i32), kind: BlockKindLike) -> PyResult<()> {
        self.0
            .set_block_kind(pos, bloq_graph::BlockKind::try_from(kind)?)
            .map_err(graph_err)
    }

    /// Assigns a Port's `"auto"`, `"input"`, `"output"`, or `"multiplex"` role.
    fn set_port_role(&mut self, pos: (i32, i32, i32), role: &str) -> PyResult<()> {
        let role = bloq_graph::PortRole::from_str(role).map_err(errors::invalid_argument)?;
        self.0.set_port_role(pos, role).map_err(graph_err)
    }

    /// Replaces the tag of the block at `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    ///     tag: The new tag string.
    ///
    /// Raises:
    ///     BlockGraphError: If no block occupies `pos`.
    fn set_block_tag(&mut self, pos: (i32, i32, i32), tag: &str) -> PyResult<()> {
        self.0.set_block_tag(pos, tag).map_err(graph_err)
    }

    // --- pipe CRUD ------------------------------------------------------------

    /// Adds a pipe. Raises `BlockGraphError` if an endpoint is missing or the
    /// pipe already exists.
    ///
    /// Args:
    ///     pipe: The pipe to add.
    fn add_pipe(&mut self, pipe: PyPipe) -> PyResult<()> {
        self.0.try_add_pipe(pipe.0).map_err(graph_err)
    }

    /// Removes and returns the pipe between `u` and `v`, or `None` if absent.
    ///
    /// Args:
    ///     u: One endpoint position `(x, y, z)`.
    ///     v: The other endpoint position `(x, y, z)`.
    fn remove_pipe(&mut self, u: (i32, i32, i32), v: (i32, i32, i32)) -> Option<PyPipe> {
        self.0.remove_pipe(u, v).map(Into::into)
    }

    /// Returns the pipe between `u` and `v`, or `None` if absent.
    ///
    /// Args:
    ///     u: One endpoint position `(x, y, z)`.
    ///     v: The other endpoint position `(x, y, z)`.
    fn get_pipe(&self, u: (i32, i32, i32), v: (i32, i32, i32)) -> Option<PyPipe> {
        self.0.get_pipe(u, v).cloned().map(Into::into)
    }

    /// `True` if a pipe connects `u` and `v`.
    ///
    /// Args:
    ///     u: One endpoint position `(x, y, z)`.
    ///     v: The other endpoint position `(x, y, z)`.
    fn has_pipe_between(&self, u: (i32, i32, i32), v: (i32, i32, i32)) -> bool {
        self.0.has_pipe_between(u, v)
    }

    /// Replaces the tag of the pipe between `u` and `v`.
    ///
    /// Args:
    ///     u: One endpoint position `(x, y, z)`.
    ///     v: The other endpoint position `(x, y, z)`.
    ///     tag: The new tag string.
    ///
    /// Raises:
    ///     BlockGraphError: If no pipe connects `u` and `v`.
    fn set_pipe_tag(&mut self, u: (i32, i32, i32), v: (i32, i32, i32), tag: &str) -> PyResult<()> {
        self.0.set_pipe_tag(u, v, tag).map_err(graph_err)
    }

    /// Sets whether the pipe between `u` and `v` carries a Hadamard.
    ///
    /// Args:
    ///     u: One endpoint position `(x, y, z)`.
    ///     v: The other endpoint position `(x, y, z)`.
    ///     hadamard: Whether the pipe carries a Hadamard across the seam.
    ///
    /// Raises:
    ///     BlockGraphError: If no pipe connects `u` and `v`.
    fn set_pipe_hadamard(
        &mut self,
        u: (i32, i32, i32),
        v: (i32, i32, i32),
        hadamard: bool,
    ) -> PyResult<()> {
        self.0.set_pipe_hadamard(u, v, hadamard).map_err(graph_err)
    }

    // --- iteration ------------------------------------------------------------

    /// Blocks authored in this definition's local body, excluding child instances.
    fn blocks(&self) -> Vec<PyBlock> {
        self.0.blocks().cloned().map(Into::into).collect()
    }

    /// Pipes authored in this definition's local body, excluding child instances.
    fn pipes(&self) -> Vec<PyPipe> {
        self.0.pipes().cloned().map(Into::into).collect()
    }

    /// Positions of this definition's local blocks, excluding child instances.
    fn positions(&self) -> Vec<(i32, i32, i32)> {
        self.0.positions().map(Into::into).collect()
    }

    /// Local lattice cells reserved by blocks (multi-cell blocks contribute
    /// their full footprint).
    fn occupied_positions(&self) -> Vec<(i32, i32, i32)> {
        self.0.occupied_positions().map(Into::into).collect()
    }

    /// All blocks directly connected to the block at `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn neighbors(&self, pos: (i32, i32, i32)) -> Vec<PyBlock> {
        self.0
            .neighbors(pos)
            .into_iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// The number of pipes attached to the block at `pos`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` lattice position.
    fn degree(&self, pos: (i32, i32, i32)) -> usize {
        self.0.degree(pos)
    }

    // --- counts and predicates --------------------------------------------------

    /// Number of blocks in this definition's local body, excluding child instances.
    #[getter]
    fn block_count(&self) -> usize {
        self.0.block_count()
    }

    /// Number of pipes in this definition's local body, excluding child instances.
    #[getter]
    fn pipe_count(&self) -> usize {
        self.0.pipe_count()
    }

    /// Number of port blocks in this definition's local body.
    #[getter]
    fn port_count(&self) -> usize {
        self.0.port_count()
    }

    /// Number of T blocks in this definition's local body.
    #[getter]
    fn t_count(&self) -> usize {
        self.0.t_count()
    }

    /// Number of selective blocks in this definition's local body.
    #[getter]
    fn selective_count(&self) -> usize {
        self.0.selective_count()
    }

    /// Number of Y blocks in this definition's local body.
    #[getter]
    fn y_count(&self) -> usize {
        self.0.y_count()
    }

    /// Number of walking blocks in this definition's local body.
    #[getter]
    fn walking_count(&self) -> usize {
        self.0.walking_count()
    }

    /// Number of patch-rotation blocks in this definition's local body.
    #[getter]
    fn patch_rotation_count(&self) -> usize {
        self.0.patch_rotation_count()
    }

    /// `True` if the graph and its reachable instances have no blocks.
    #[getter]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `True` when the graph and its reachable instances contain no T blocks.
    #[getter]
    fn is_clifford(&self) -> bool {
        self.0.is_clifford()
    }

    /// `True` when every block, including reachable instances, has a statically
    /// known kind (no selectives).
    #[getter]
    fn is_rigid(&self) -> bool {
        self.0.is_rigid()
    }

    /// `True` when the graph contains at least one port.
    #[getter]
    fn is_open(&self) -> bool {
        self.0.is_open()
    }

    // --- transforms (return new graphs) ----------------------------------------

    /// Returns a new graph shifted by `offset`.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Args:
    ///     offset: The `(x, y, z)` amount to shift every position by.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a shifted position exceeds the 32-bit
    ///         coordinate range.
    ///     BlockGraphError: If the graph retains module structure.
    fn shift_positions(&self, offset: (i32, i32, i32)) -> PyResult<Self> {
        self.0
            .shift_positions(offset.into())
            .map(Into::into)
            .map_err(graph_transform_err)
    }

    /// Returns a new graph rotated about the origin around `axis` by
    /// `quarter_turns` 90-degree steps. Discards actions.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Args:
    ///     axis: A `UDirection` or its axis letter (`"X"` / `"Y"` / `"Z"`).
    ///     quarter_turns: Number of 90-degree steps; may be negative.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `axis` is invalid or a rotated position
    ///         exceeds the 32-bit coordinate range.
    ///     BlockGraphError: If the rotation cannot be applied.
    fn rotate_about_origin(&self, axis: UDirectionLike, quarter_turns: i32) -> PyResult<Self> {
        let axis = bloq_utils::UDirection::try_from(axis)?;
        self.0
            .rotate_about_origin(axis, quarter_turns)
            .map(Into::into)
            .map_err(graph_transform_err)
    }

    /// Returns a new graph with X/Z bases flipped (topology-preserving).
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Raises:
    ///     BlockGraphError: If the basis flip cannot be applied.
    fn flip_xz_basis(&self) -> PyResult<Self> {
        self.0.flip_xz_basis().map(Into::into).map_err(graph_err)
    }

    /// Returns a new graph shifted so the minimum z-coordinate is zero.
    /// Call `flatten()` first for authored module interfaces or child instances.
    ///
    /// Raises:
    ///     InvalidArgumentError: If normalization exceeds the 32-bit
    ///         coordinate range.
    ///     BlockGraphError: If the graph retains module structure.
    fn with_zero_min_z(&self) -> PyResult<Self> {
        self.0
            .with_zero_min_z()
            .map(Into::into)
            .map_err(graph_transform_err)
    }

    // --- export -----------------------------------------------------------------

    /// Exports the graph as a `.gltf` scene, optionally overlaying a stabilizer row.
    /// Set `module_view=True` to color an authored hierarchy by definition.
    ///
    /// Args:
    ///     path: Filesystem path to write the `.gltf` file to.
    ///     pipe_length: Rendered pipe length in scene units. Defaults to 2.0.
    ///     module_view: Color by module definition without changing the hierarchy.
    ///         Cannot be combined with a stabilizer overlay.
    ///     stabilizer: Optional generator, or indices of logical generators
    ///         whose phase-free product supplies the displayed correlation support.
    ///     pop_faces_at_directions: Signed lattice directions whose outward
    ///         faces are removed, for example `["-Y"]`.
    ///     pop_faces_at_blocks: `(position, face_direction)` pairs selecting
    ///         individual block faces.
    ///     pop_faces_at_pipes: `(u, v, face_direction)` triples selecting
    ///         individual pipe faces.
    ///
    /// Raises:
    ///     OSError: If the file cannot be written.
    #[pyo3(signature = (path, pipe_length=2.0, stabilizer=None, pop_faces_at_directions=None, pop_faces_at_blocks=None, pop_faces_at_pipes=None, module_view=false))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword arguments remain explicit"
    )]
    fn export_gltf(
        &self,
        path: PathBuf,
        pipe_length: f32,
        stabilizer: Option<StabilizerLike>,
        pop_faces_at_directions: Option<Vec<DirectionLike>>,
        pop_faces_at_blocks: Option<Vec<(PosTuple, DirectionLike)>>,
        pop_faces_at_pipes: Option<Vec<(PosTuple, PosTuple, DirectionLike)>>,
        module_view: bool,
    ) -> PyResult<()> {
        let popped_faces = gltf_face_selectors(
            pop_faces_at_directions,
            pop_faces_at_blocks,
            pop_faces_at_pipes,
        )?;
        if module_view {
            if stabilizer.is_some() {
                return Err(errors::InvalidArgumentError::new_err(
                    "module_view does not accept a stabilizer overlay",
                ));
            }
            return self
                .0
                .write_module_gltf_file(pipe_length, path, &popped_faces)
                .map_err(graph_err);
        }
        let stabilizer = stabilizer
            .map(|selection| selection.resolve(&self.0))
            .transpose()?;
        self.0
            .write_to_gltf_file(pipe_length, path, stabilizer.as_ref(), &popped_faces)
            .map_err(graph_err)
    }

    /// Exports the graph as a self-contained HTML viewer page (the
    /// `model-viewer` renderer is loaded from a CDN, so viewing needs network).
    /// Set `module_view=True` for the editor's definition colors and module legend.
    ///
    /// Args:
    ///     path: Filesystem path to write the HTML file to.
    ///     pipe_length: Rendered pipe length in scene units. Defaults to 2.0.
    ///     module_view: Color by module definition without changing the hierarchy.
    ///         Cannot be combined with a stabilizer overlay.
    ///     stabilizer: Optional generator, or indices of logical generators
    ///         whose phase-free product supplies the displayed correlation support.
    ///     pop_faces_at_directions: Signed lattice directions whose outward
    ///         faces are removed, for example `["-Y"]`.
    ///     pop_faces_at_blocks: `(position, face_direction)` pairs selecting
    ///         individual block faces.
    ///     pop_faces_at_pipes: `(u, v, face_direction)` triples selecting
    ///         individual pipe faces.
    ///
    /// Raises:
    ///     OSError: If the file cannot be written.
    #[pyo3(signature = (path, pipe_length=2.0, stabilizer=None, pop_faces_at_directions=None, pop_faces_at_blocks=None, pop_faces_at_pipes=None, module_view=false))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword arguments remain explicit"
    )]
    fn export_html_viewer(
        &self,
        path: PathBuf,
        pipe_length: f32,
        stabilizer: Option<StabilizerLike>,
        pop_faces_at_directions: Option<Vec<DirectionLike>>,
        pop_faces_at_blocks: Option<Vec<(PosTuple, DirectionLike)>>,
        pop_faces_at_pipes: Option<Vec<(PosTuple, PosTuple, DirectionLike)>>,
        module_view: bool,
    ) -> PyResult<()> {
        let popped_faces = gltf_face_selectors(
            pop_faces_at_directions,
            pop_faces_at_blocks,
            pop_faces_at_pipes,
        )?;
        if module_view {
            if stabilizer.is_some() {
                return Err(errors::InvalidArgumentError::new_err(
                    "module_view does not accept a stabilizer overlay",
                ));
            }
            return self
                .0
                .write_module_html_viewer(pipe_length, path, &popped_faces)
                .map_err(graph_err);
        }
        let stabilizer = stabilizer
            .map(|selection| selection.resolve(&self.0))
            .transpose()?;
        self.0
            .write_to_gltf_html_viewer(pipe_length, path, stabilizer.as_ref(), &popped_faces)
            .map_err(graph_err)
    }

    // --- actions ------------------------------------------------------------------

    /// Appends actions parsed from BLOG action-statement text, e.g.
    /// `"m = measure 0 -> +Z"`. Block IDs refer to `get_block_id` values.
    ///
    /// Args:
    ///     text: BLOG action-statement text; numeric block IDs refer to
    ///         `get_block_id` values.
    ///
    /// Raises:
    ///     ParseError: If `text` is not valid action-statement syntax.
    ///     BlockGraphError: If action semantics, measurement-surface derivation,
    ///         or the resulting dependency DAG is invalid.
    ///
    /// Examples:
    ///     >>> from bloq import BlockGraph, Block
    ///     >>> graph = BlockGraph()
    ///     >>> graph.add_block(Block((0, 0, 0), "ZXZ"))
    ///     (0, 0, 0)
    ///     >>> graph.add_actions_from_text("m = measure 0")
    ///     >>> [str(a) for a in graph.actions()]
    ///     ['m = measure [0, 0, 0]']
    fn add_actions_from_text(&mut self, text: &str) -> PyResult<()> {
        // parse_actions resolves numeric block IDs through this map, mirroring
        // how ids are assigned in `.blog` files.
        let id_to_pos: HashMap<_, _> = self
            .0
            .positions()
            .filter_map(|pos| self.0.get_block_id(pos).map(|id| (id, pos)))
            .collect();
        let parsed = bloq_graph::parse_actions(text, |id, dir| {
            let pos = id_to_pos.get(&id).copied()?;
            // A measure-edge source (`Some(dir)`) resolves through the block's
            // exposed endpoint so multi-cell blocks match the file-parse path;
            // node targets (`None`) stay at the anchor.
            match dir {
                Some(dir) => self
                    .0
                    .get_block(pos)
                    .map(|block| block.endpoint_for_direction(dir)),
                None => Some(pos),
            }
        })
        .map_err(|e| parse_err(&e, text))?;
        let mut actions = self.0.actions();
        actions.extend(parsed);
        self.0.set_actions(actions).map_err(graph_err)
    }

    /// Appends one structured action, rederiving measurement surfaces and the action DAG.
    ///
    /// Args:
    ///     action: The structured action to append.
    ///
    /// Raises:
    ///     BlockGraphError: If action semantics, measurement-surface derivation,
    ///         or the resulting dependency DAG is invalid.
    fn add_action(&mut self, action: PyAction) -> PyResult<()> {
        self.0.add_action(action.0).map_err(graph_err)
    }

    /// Replaces all actions, rederiving measurement surfaces and the action DAG.
    ///
    /// Args:
    ///     actions: The structured actions to install, replacing any existing.
    ///
    /// Raises:
    ///     BlockGraphError: If action semantics, measurement-surface derivation,
    ///         or the resulting dependency DAG is invalid.
    fn set_actions(&mut self, actions: Vec<PyAction>) -> PyResult<()> {
        self.0
            .set_actions(actions.into_iter().map(|a| a.0).collect())
            .map_err(graph_err)
    }

    /// The graph's actions as structured `Action` values; `str(action)`
    /// renders the BLOG statement (positions, not block ids).
    fn actions(&self) -> Vec<PyAction> {
        self.0.actions().into_iter().map(PyAction).collect()
    }

    /// `True` if the graph carries any actions.
    fn has_actions(&self) -> bool {
        self.0.has_actions()
    }

    /// Removes all actions from the graph.
    fn clear_actions(&mut self) {
        self.0.clear_actions();
    }

    // --- dunder -------------------------------------------------------------------

    fn __len__(&self) -> usize {
        self.0.block_count()
    }

    /// Iterates the graph's blocks, so `for block in graph` works and
    /// `list(graph)` matches `graph.blocks()`.
    ///
    /// Eager, like `blocks()`: the whole list is materialized up front, so
    /// mutating the graph mid-iteration does not invalidate the iterator.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> graph = bloq.GalleryItem.BELL_STATE.load()
    ///     >>> [block.pos for block in graph] == [block.pos for block in graph.blocks()]
    ///     True
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[Block]", imports = ("typing")))]
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyList::new(py, self.blocks())?.try_iter()
    }

    /// Graph equality is structural: same blocks, pipes, actions, and hierarchy —
    /// compared through the canonical `.blog` text, which is exactly the state
    /// the codecs round-trip.
    fn __eq__(
        &self,
        #[gen_stub(override_type(type_repr = "builtins.object"))] other: &Self,
    ) -> bool {
        self.0.to_blog_text() == other.0.to_blog_text()
    }

    /// Pickles through the `.blog` text codec, which makes a graph shippable
    /// to a `multiprocessing` worker (sinter's model, chiefly).
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let constructor = py.get_type::<Self>().getattr("from_text")?.unbind();
        Ok((constructor, (self.0.to_blog_text(),)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// A graph owns no shared substructure Python can see, so a deep copy is a
    /// plain clone; `memo` is accepted to satisfy the protocol and ignored.
    #[pyo3(signature = (memo=None))]
    fn __deepcopy__(&self, memo: Option<Bound<'_, PyAny>>) -> Self {
        let _ = memo;
        self.clone()
    }

    fn __str__(&self) -> String {
        self.0.to_blog_text()
    }

    fn __repr__(&self) -> String {
        format!(
            "<BlockGraph blocks={} pipes={}>",
            self.0.block_count(),
            self.0.pipe_count()
        )
    }
}

// ==============================================================================
// Module functions and registration
// ==============================================================================

/// Parses a block graph from `.blog` text, raising `ParseError` with a
/// rendered diagnostic on failure.
///
/// Not itself bound: `BlockGraph.from_text` is the public spelling, so that
/// parsing pairs with `BlockGraph.to_text` the way `Bloq.from_text` pairs with
/// `Bloq.to_text` rather than being a second, free-function way to say it.
pub(crate) fn parse_blog(
    text: &str,
    limits: bloq_graph::ModuleCertificationLimits,
) -> PyResult<PyBlockGraph> {
    bloq_graph::BlockGraph::from_text_with_limits(text, limits)
        .map(Into::into)
        .map_err(|error| graph_input_err(error, Some(text)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBranchArm>()?;
    m.add_class::<PyBranch>()?;
    m.add_class::<PyBlockKind>()?;
    m.add_class::<PyBlock>()?;
    m.add_class::<PyPipe>()?;
    m.add_class::<PyStabilizer>()?;
    m.add_class::<PyStabilizerGenerator>()?;
    m.add_class::<PyBlockGraph>()?;
    Ok(())
}
