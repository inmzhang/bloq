//! Block, pipe, and moving-block geometry primitives.

use std::cmp::Ordering;

use bloq_utils::{Basis, Direction, PauliBasis, PortRole, RGBA, UDirection};
use glam::{IVec2, IVec3};
use smallvec::{SmallVec, smallvec};
use strum::{Display, EnumString};

use crate::{BlockGraphError, CubeHeight};

/// Reserved/connectable cell offsets; ordinary single-cell blocks stay on the
/// stack so hot placement and validation loops do not allocate.
pub(crate) type Offsets = SmallVec<[IVec3; 2]>;

/// Errors produced when constructing or converting block kinds.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum BlockError {
    /// The three axis bases do not encode a supported cube orientation.
    #[error("invalid bases for CubeKind: {0:?}")]
    InvalidCubeBases([Basis; 3]),
    /// A spatial cube orientation was used as a walking boundary.
    #[error("spatial cube kind {0} is not a valid Walking boundary kind")]
    SpatialCubeNotWalkingBoundary(CubeKind),
    /// A walking displacement is zero or leaves the adjacent 3-by-3 plane.
    #[error(
        "invalid Walking movement vector {0}; expected one of (+/-1,0), (0,+/-1), or (+/-1,+/-1)"
    )]
    InvalidWalkingMovement(IVec2),
    /// A patch-rotation displacement is zero or leaves the adjacent 3-by-3 plane.
    #[error(
        "invalid PatchRotation movement vector {0}; expected one of (+/-1,0), (0,+/-1), or (+/-1,+/-1)"
    )]
    InvalidPatchRotationMovement(IVec2),
    /// Text does not name a supported block kind.
    #[error("invalid BlockKind: {0}")]
    InvalidBlockKind(String),
    /// A cube height was assigned to a non-cube block.
    #[error("cube height can only be set on cube blocks, got {0}")]
    CubeHeightOnNonCube(BlockKind),
    /// A display color was assigned to a non-port block.
    #[error("port color can only be set on Port blocks, got {0}")]
    PortColorOnNonPort(BlockKind),
    /// A port role was assigned to a non-port block.
    #[error("port role can only be set on Port blocks, got {0}")]
    PortRoleOnNonPort(BlockKind),
    /// A symbolic cube height is zero or negative.
    #[error("cube height {numerator}d/{denominator} is not positive")]
    NonPositiveCubeHeight {
        /// Unsigned magnitude of the invalid height numerator.
        numerator: u32,
        /// Denominator of the invalid height.
        denominator: u32,
    },
    /// A cube height needs more cells than the graph format supports.
    #[error("cube height occupies {cells} cells, exceeding supported maximum 65535")]
    CubeHeightExceedsRange {
        /// Required number of time-axis cells.
        cells: u32,
    },
    /// A cube footprint extends beyond the coordinate range.
    #[error("cube at z={z} occupying {cells} cells exceeds the graph coordinate range")]
    CubeHeightCoordinateOverflow {
        /// Cube's starting time coordinate.
        z: i32,
        /// Number of cells in the cube footprint.
        cells: u32,
    },
    /// Text does not encode a valid symbolic cube height.
    #[error("invalid cube height expression `{0}`; expected `k*d + n`, e.g. `d`, `2d`, `3d/2-1`")]
    InvalidCubeHeight(String),
}

/// A cube block's boundary orientation, named by the Pauli basis (`X` or `Z`)
/// assigned to each of the three axes in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString, Display)]
pub enum CubeKind {
    /// X boundary on x; Z boundaries on y and z.
    XZZ,
    /// Z boundary on x; X boundary on y; Z boundary on z.
    ZXZ,
    /// Z boundaries on x and y; X boundary on z.
    ZZX,
    /// Z boundary on x; X boundaries on y and z.
    ZXX,
    /// X boundaries on x and z; Z boundary on y.
    XZX,
    /// X boundaries on x and y; Z boundary on z.
    XXZ,
}

impl CubeKind {
    /// Returns the per-axis boundary bases in `[x, y, z]` order.
    pub const fn bases(&self) -> [Basis; 3] {
        match self {
            CubeKind::XZZ => [Basis::X, Basis::Z, Basis::Z],
            CubeKind::ZXZ => [Basis::Z, Basis::X, Basis::Z],
            CubeKind::ZZX => [Basis::Z, Basis::Z, Basis::X],
            CubeKind::ZXX => [Basis::Z, Basis::X, Basis::X],
            CubeKind::XZX => [Basis::X, Basis::Z, Basis::X],
            CubeKind::XXZ => [Basis::X, Basis::X, Basis::Z],
        }
    }

    /// Returns the axis whose basis differs from the other two (the correlation
    /// surface normal).
    pub const fn normal_direction(&self) -> UDirection {
        match self {
            CubeKind::XZZ | CubeKind::ZXX => UDirection::X,
            CubeKind::ZXZ | CubeKind::XZX => UDirection::Y,
            CubeKind::ZZX | CubeKind::XXZ => UDirection::Z,
        }
    }

    /// Returns the basis on the [`normal_direction`](Self::normal_direction) axis.
    pub const fn normal_basis(&self) -> Basis {
        self.bases()[self.normal_direction().index()]
    }

    /// Returns whether the normal-axis boundary is X-type.
    pub const fn is_x_type(&self) -> bool {
        matches!(self.normal_basis(), Basis::X)
    }

    /// Returns the boundary basis on the x axis.
    pub const fn x(&self) -> Basis {
        self.bases()[0]
    }

    /// Returns the boundary basis on the y axis.
    pub const fn y(&self) -> Basis {
        self.bases()[1]
    }

    /// Returns the boundary basis on the z axis.
    pub const fn z(&self) -> Basis {
        self.bases()[2]
    }

    /// Returns whether the cube's normal lies along the time (`Z`) axis, making
    /// it a spatial rather than temporal boundary.
    pub const fn is_spatial(&self) -> bool {
        matches!(self, CubeKind::ZZX | CubeKind::XXZ)
    }
}

impl TryFrom<[Basis; 3]> for CubeKind {
    type Error = BlockError;

    fn try_from(bases: [Basis; 3]) -> Result<Self, Self::Error> {
        match bases {
            [Basis::X, Basis::Z, Basis::Z] => Ok(CubeKind::XZZ),
            [Basis::Z, Basis::X, Basis::Z] => Ok(CubeKind::ZXZ),
            [Basis::Z, Basis::Z, Basis::X] => Ok(CubeKind::ZZX),
            [Basis::Z, Basis::X, Basis::X] => Ok(CubeKind::ZXX),
            [Basis::X, Basis::Z, Basis::X] => Ok(CubeKind::XZX),
            [Basis::X, Basis::X, Basis::Z] => Ok(CubeKind::XXZ),
            _ => Err(BlockError::InvalidCubeBases(bases)),
        }
    }
}

/// Boundary orientation of a walking block, restricted to the four temporal
/// (non-spatial) cube orientations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString, Display)]
pub enum WalkingBoundaryKind {
    /// X boundary on x; Z boundaries on y and z.
    XZZ,
    /// Z boundary on x; X boundary on y; Z boundary on z.
    ZXZ,
    /// Z boundary on x; X boundaries on y and z.
    ZXX,
    /// X boundaries on x and z; Z boundary on y.
    XZX,
}

impl WalkingBoundaryKind {
    /// All four boundary kinds.
    pub const ALL: [Self; 4] = [Self::ZXZ, Self::ZXX, Self::XZZ, Self::XZX];

    /// Returns the per-axis boundary bases in `[x, y, z]` order.
    ///
    /// Delegates to the equivalent [`CubeKind`] rather than re-tabulating the
    /// rows (not `const`: `From` cannot be called in a const context).
    pub fn bases(&self) -> [Basis; 3] {
        CubeKind::from(*self).bases()
    }

    /// Returns the boundary basis on the x axis.
    pub fn x(&self) -> Basis {
        self.bases()[0]
    }

    /// Returns the boundary basis on the y axis.
    pub fn y(&self) -> Basis {
        self.bases()[1]
    }

    /// Returns the boundary basis on the z axis.
    pub fn z(&self) -> Basis {
        self.bases()[2]
    }

    /// Returns the kind with every `X` and `Z` boundary basis swapped.
    pub const fn flip_xz_basis(&self) -> Self {
        match self {
            WalkingBoundaryKind::XZZ => WalkingBoundaryKind::ZXX,
            WalkingBoundaryKind::ZXZ => WalkingBoundaryKind::XZX,
            WalkingBoundaryKind::ZXX => WalkingBoundaryKind::XZZ,
            WalkingBoundaryKind::XZX => WalkingBoundaryKind::ZXZ,
        }
    }
}

impl TryFrom<CubeKind> for WalkingBoundaryKind {
    type Error = BlockError;

    fn try_from(kind: CubeKind) -> Result<Self, Self::Error> {
        match kind {
            CubeKind::XZZ => Ok(Self::XZZ),
            CubeKind::ZXZ => Ok(Self::ZXZ),
            CubeKind::ZXX => Ok(Self::ZXX),
            CubeKind::XZX => Ok(Self::XZX),
            CubeKind::ZZX | CubeKind::XXZ => Err(BlockError::SpatialCubeNotWalkingBoundary(kind)),
        }
    }
}

impl From<WalkingBoundaryKind> for CubeKind {
    fn from(kind: WalkingBoundaryKind) -> Self {
        match kind {
            WalkingBoundaryKind::XZZ => CubeKind::XZZ,
            WalkingBoundaryKind::ZXZ => CubeKind::ZXZ,
            WalkingBoundaryKind::ZXX => CubeKind::ZXX,
            WalkingBoundaryKind::XZX => CubeKind::XZX,
        }
    }
}

/// A walking block: a boundary orientation plus a diagonal in-plane movement
/// that advances one time layer while shifting the patch by `movement`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WalkingKind {
    boundary: WalkingBoundaryKind,
    movement: IVec2,
}

impl WalkingKind {
    /// The default `ZXZ` boundary moving one cell in `+x`.
    pub const DEFAULT: Self = Self {
        boundary: WalkingBoundaryKind::ZXZ,
        movement: IVec2::new(1, 0),
    };

    /// Constructs a walking kind, validating the movement vector.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::InvalidWalkingMovement`] unless each component of
    /// `movement` is in `-1..=1` and the vector is nonzero.
    pub fn new(boundary: WalkingBoundaryKind, movement: IVec2) -> Result<Self, BlockError> {
        if !is_neighboring_step(movement) {
            return Err(BlockError::InvalidWalkingMovement(movement));
        }
        Ok(Self { boundary, movement })
    }

    /// Returns the walking patch's boundary orientation.
    pub const fn boundary(self) -> WalkingBoundaryKind {
        self.boundary
    }

    /// Returns the in-plane displacement per time layer.
    pub const fn movement(self) -> IVec2 {
        self.movement
    }

    /// Returns the movement as a 3D vector with `+1` on the time axis.
    pub const fn movement_3d(self) -> IVec3 {
        self.movement.extend(1)
    }

    /// Returns a copy with a different boundary orientation.
    pub const fn with_boundary(self, boundary: WalkingBoundaryKind) -> Self {
        Self {
            boundary,
            movement: self.movement,
        }
    }

    /// Returns the end cell reached from `start`.
    ///
    /// # Panics
    ///
    /// Panics if the endpoint exceeds the coordinate range. Use
    /// [`try_end_position`](Self::try_end_position) for untrusted positions.
    pub fn end_position(self, start: IVec3) -> IVec3 {
        self.try_end_position(start)
            .expect("walking endpoint must fit in the i32 coordinate range")
    }

    /// Returns the end cell reached from `start`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if the endpoint exceeds
    /// the coordinate range.
    pub fn try_end_position(self, start: IVec3) -> Result<IVec3, BlockGraphError> {
        crate::checked_add_position(start, self.movement_3d())
    }

    /// Returns the two exposed connection endpoints (start and end) as offsets.
    pub fn connectable_offsets(self) -> [IVec3; 2] {
        [IVec3::ZERO, self.movement_3d()]
    }

    /// Returns every cell the block reserves, as offsets from its position.
    pub fn reserved_offsets(self) -> Vec<IVec3> {
        movement_footprint(self.movement_3d())
    }
}

const fn is_neighboring_step(movement: IVec2) -> bool {
    matches!(movement.x, -1..=1)
        && matches!(movement.y, -1..=1)
        && (movement.x != 0 || movement.y != 0)
}

/// Orders two movement steps componentwise. `IVec2` has no `Ord`, which is why
/// both moving-block kinds hand-write `Ord` instead of deriving it.
fn cmp_movement(left: IVec2, right: IVec2) -> Ordering {
    left.x.cmp(&right.x).then_with(|| left.y.cmp(&right.y))
}

/// Reserves the bounding box of the start and end cells, inclusive.
fn movement_footprint(movement: IVec3) -> Vec<IVec3> {
    let x_min = 0.min(movement.x);
    let x_max = 0.max(movement.x);
    let y_min = 0.min(movement.y);
    let y_max = 0.max(movement.y);
    let mut offsets = Vec::new();
    for x in x_min..=x_max {
        for y in y_min..=y_max {
            for z in 0..=movement.z {
                offsets.push(IVec3::new(x, y, z));
            }
        }
    }
    offsets
}

impl Default for WalkingKind {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl PartialOrd for WalkingKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WalkingKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.boundary
            .cmp(&other.boundary)
            .then_with(|| cmp_movement(self.movement, other.movement))
    }
}

/// Rotates a patch by 90 degrees. An axial step spans two time cells.
/// A diagonal step crosses a 2-by-2 footprint in one time cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchRotationKind {
    basis: Basis,
    movement: IVec2,
}

impl PatchRotationKind {
    /// The default `X`-basis rotation moving one cell in `+x`.
    pub const DEFAULT: Self = Self {
        basis: Basis::X,
        movement: IVec2::new(1, 0),
    };

    /// Constructs a patch-rotation kind, validating the movement vector.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::InvalidPatchRotationMovement`] unless `movement` is
    /// nonzero with both components in `-1..=1`.
    pub fn new(basis: Basis, movement: IVec2) -> Result<Self, BlockError> {
        if !is_neighboring_step(movement) {
            return Err(BlockError::InvalidPatchRotationMovement(movement));
        }
        Ok(Self { basis, movement })
    }

    /// Returns the construction basis, the start corner's time-face basis for a planar rotation.
    pub const fn basis(self) -> Basis {
        self.basis
    }

    /// Returns the boundary basis on the `x` axis at the start corner.
    pub const fn x_axis_boundary_basis(self) -> Basis {
        if self.movement.x != 0 {
            self.basis.flip()
        } else {
            self.basis
        }
    }

    /// Returns the in-plane displacement from input to output.
    pub const fn movement(self) -> IVec2 {
        self.movement
    }

    /// Returns whether the rotation crosses opposite corners in a single time cell.
    pub const fn is_planar(self) -> bool {
        self.movement.x != 0 && self.movement.y != 0
    }

    /// Returns the corner exposing a given spatial face of the planar footprint.
    /// Both temporal faces are closed.
    pub fn planar_endpoint_offset(self, dir: Direction) -> Option<IVec3> {
        if !self.is_planar() || !dir.is_spatial() {
            return None;
        }
        let end = self.movement_3d();
        Some(if end.dot(dir.to_ivec3()) > 0 {
            end
        } else {
            IVec3::ZERO
        })
    }

    /// Returns the displacement between the input and output cells.
    pub const fn movement_3d(self) -> IVec3 {
        self.movement.extend(if self.is_planar() { 0 } else { 1 })
    }

    /// Returns a copy with a different basis.
    pub const fn with_basis(self, basis: Basis) -> Self {
        Self {
            basis,
            movement: self.movement,
        }
    }

    /// Returns the end cell reached from `start`.
    ///
    /// # Panics
    ///
    /// Panics if the endpoint exceeds the coordinate range. Use
    /// [`try_end_position`](Self::try_end_position) for untrusted positions.
    pub fn end_position(self, start: IVec3) -> IVec3 {
        self.try_end_position(start)
            .expect("patch-rotation endpoint must fit in the i32 coordinate range")
    }

    /// Returns the end cell reached from `start`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if the endpoint exceeds
    /// the coordinate range.
    pub fn try_end_position(self, start: IVec3) -> Result<IVec3, BlockGraphError> {
        crate::checked_add_position(start, self.movement_3d())
    }

    /// Returns the two exposed connection endpoints (start and end) as offsets.
    pub fn connectable_offsets(self) -> [IVec3; 2] {
        [IVec3::ZERO, self.movement_3d()]
    }

    /// Returns every cell the block reserves, as offsets from its position.
    pub fn reserved_offsets(self) -> Vec<IVec3> {
        movement_footprint(self.movement_3d())
    }

    /// Returns the boundary bases exposed at `endpoint`, or `None` if `endpoint`
    /// is neither the start nor end cell. The bases flip between the two ends.
    pub fn bases_at_endpoint(self, start: IVec3, endpoint: IVec3) -> Option<[Basis; 3]> {
        let basis = self.x_axis_boundary_basis();
        let x_basis = if endpoint == start {
            basis
        } else if self
            .try_end_position(start)
            .is_ok_and(|end| endpoint == end)
        {
            basis.flip()
        } else {
            return None;
        };
        Some(if self.is_planar() {
            [x_basis, x_basis, x_basis.flip()]
        } else {
            [x_basis, x_basis.flip(), x_basis]
        })
    }

    /// Returns the bases on the pipe face at `endpoint`, or `None` if `endpoint`
    /// is neither the start nor end cell.
    ///
    /// Axial rotations expose the construction basis in the pipe's frame.
    /// A planar rotation exposes its local spatial and time-face bases.
    pub fn pipe_face_bases_at_endpoint(self, start: IVec3, endpoint: IVec3) -> Option<[Basis; 3]> {
        if self.is_planar() {
            return self.bases_at_endpoint(start, endpoint);
        }
        if endpoint == start {
            Some([self.basis, self.basis.flip(), self.basis])
        } else if self
            .try_end_position(start)
            .is_ok_and(|end| endpoint == end)
        {
            let basis = self.basis.flip();
            Some([basis, basis.flip(), basis])
        } else {
            None
        }
    }
}

impl Default for PatchRotationKind {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl PartialOrd for PatchRotationKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PatchRotationKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.basis
            .cmp(&other.basis)
            .then_with(|| cmp_movement(self.movement, other.movement))
    }
}

/// The two Pauli bases a selective block chooses between when resolved. The
/// resolve condition being true selects the first named basis, false the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Display, EnumString)]
#[strum(
    ascii_case_insensitive,
    parse_err_ty = String,
    parse_err_fn = invalid_selective_kind
)]
pub enum SelectiveKind {
    /// Select between X when true and Y when false.
    XY,
    /// Select between X when true and Z when false.
    XZ,
    /// Select between Y when true and Z when false.
    YZ,
}

impl SelectiveKind {
    /// Returns the Pauli basis chosen when the resolve condition is true.
    pub const fn pauli_if_true(&self) -> PauliBasis {
        match self {
            SelectiveKind::XY | SelectiveKind::XZ => PauliBasis::X,
            SelectiveKind::YZ => PauliBasis::Y,
        }
    }

    /// Returns the Pauli basis chosen when the resolve condition is false.
    pub const fn pauli_if_false(&self) -> PauliBasis {
        match self {
            SelectiveKind::XY => PauliBasis::Y,
            SelectiveKind::XZ | SelectiveKind::YZ => PauliBasis::Z,
        }
    }

    /// Returns the kind with its `X` and `Z` roles swapped.
    pub const fn flip_xz_basis(&self) -> Self {
        match self {
            SelectiveKind::XY => SelectiveKind::YZ,
            SelectiveKind::XZ => SelectiveKind::XZ,
            SelectiveKind::YZ => SelectiveKind::XY,
        }
    }
}

/// The kind of a [`Block`], determining its surface code semantics and lattice
/// footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Display, Default)]
pub enum BlockKind {
    /// A single-cell cube with a fixed boundary orientation.
    #[strum(to_string = "{0}")]
    Cube(CubeKind),
    /// A patch that walks diagonally by one cell over one time layer.
    #[strum(to_string = "Walking")]
    Walking(WalkingKind),
    /// A patch that rotates 90 degrees over one time layer.
    #[strum(to_string = "PatchRotation")]
    PatchRotation(PatchRotationKind),
    /// A Y-basis state injection block.
    Y,
    /// A fixed terminal transversal measurement in the X or Z basis.
    #[strum(to_string = "{0}")]
    Measurement(Basis),
    /// An open connection endpoint (graph input or output).
    #[default]
    Port,
    /// A T-state (magic state) block, lowered via cultivation.
    T,
    /// A selective measurement/preparation block resolved at runtime.
    #[strum(to_string = "{0}")]
    Selective(SelectiveKind),
}

impl BlockKind {
    /// Returns whether this is a cube block.
    pub const fn is_cube(&self) -> bool {
        matches!(self, BlockKind::Cube(_))
    }

    /// Returns whether this is a walking block.
    pub const fn is_walking(&self) -> bool {
        matches!(self, BlockKind::Walking(_))
    }

    /// Returns whether this is a patch-rotation block.
    pub const fn is_patch_rotation(&self) -> bool {
        matches!(self, BlockKind::PatchRotation(_))
    }

    /// Returns whether this block rotates a patch within one spatial layer.
    pub const fn is_planar_rotation(&self) -> bool {
        matches!(self, BlockKind::PatchRotation(kind) if kind.is_planar())
    }

    /// Returns whether this is an open port.
    pub const fn is_port(&self) -> bool {
        matches!(self, BlockKind::Port)
    }

    /// Returns whether this is a Y-state block.
    pub const fn is_y(&self) -> bool {
        matches!(self, BlockKind::Y)
    }

    /// Returns whether this is a fixed terminal measurement.
    pub const fn is_measurement(&self) -> bool {
        matches!(self, BlockKind::Measurement(_))
    }

    /// Returns whether this is a T-state block.
    pub const fn is_t(&self) -> bool {
        matches!(self, BlockKind::T)
    }

    /// Returns whether this is a runtime-selective block.
    pub const fn is_selective(&self) -> bool {
        matches!(self, BlockKind::Selective(_))
    }

    /// Returns whether the block's behavior is resolved at runtime (selective or
    /// T blocks) rather than fixed at compile time.
    pub const fn is_dynamic(&self) -> bool {
        matches!(self, BlockKind::Selective(_) | BlockKind::T)
    }

    /// Returns a copy with every `X` and `Z` boundary basis swapped.
    ///
    /// # Panics
    ///
    /// Panics if an internally stored cube kind violates its basis invariant.
    pub fn flip_xz_basis(self) -> Self {
        match self {
            BlockKind::Cube(kind) => {
                let bases = kind.bases().map(Basis::flip);
                BlockKind::Cube(
                    CubeKind::try_from(bases)
                        .expect("flipping X/Z bases preserves cube kind validity"),
                )
            }
            BlockKind::Walking(kind) => {
                BlockKind::Walking(kind.with_boundary(kind.boundary().flip_xz_basis()))
            }
            BlockKind::PatchRotation(kind) => {
                BlockKind::PatchRotation(kind.with_basis(kind.basis().flip()))
            }
            BlockKind::Measurement(basis) => BlockKind::Measurement(basis.flip()),
            BlockKind::Selective(kind) => BlockKind::Selective(kind.flip_xz_basis()),
            BlockKind::Y | BlockKind::Port | BlockKind::T => self,
        }
    }

    /// Returns whether the block is Clifford (everything except the T block).
    pub const fn is_clifford(&self) -> bool {
        !self.is_t()
    }

    /// Returns one representative of every block kind, using default parameters
    /// for the parameterized kinds.
    pub const fn all_kinds() -> [BlockKind; 16] {
        [
            BlockKind::Cube(CubeKind::XZZ),
            BlockKind::Cube(CubeKind::ZXZ),
            BlockKind::Cube(CubeKind::ZZX),
            BlockKind::Cube(CubeKind::ZXX),
            BlockKind::Cube(CubeKind::XZX),
            BlockKind::Cube(CubeKind::XXZ),
            BlockKind::Y,
            BlockKind::Measurement(Basis::X),
            BlockKind::Measurement(Basis::Z),
            BlockKind::T,
            BlockKind::Port,
            BlockKind::Walking(WalkingKind::DEFAULT),
            BlockKind::PatchRotation(PatchRotationKind::DEFAULT),
            BlockKind::Selective(SelectiveKind::XY),
            BlockKind::Selective(SelectiveKind::XZ),
            BlockKind::Selective(SelectiveKind::YZ),
        ]
    }

    /// Returns the cells the kind reserves, as offsets from its position.
    ///
    /// Cube height is applied per-block by [`Block::reserved_offsets`], so the
    /// single-cell kinds here reserve only the origin.
    pub fn reserved_offsets(self) -> Offsets {
        match self {
            BlockKind::Walking(kind) => kind.reserved_offsets().into(),
            BlockKind::PatchRotation(kind) => kind.reserved_offsets().into(),
            _ => smallvec![IVec3::ZERO],
        }
    }

    /// Returns whether two blocks may both reserve `overlap_pos`.
    ///
    /// Only walking blocks permit shared reserved cells, and never at a hard
    /// endpoint of either block.
    pub fn allows_reserved_overlap(
        self,
        self_pos: IVec3,
        other: BlockKind,
        other_pos: IVec3,
        overlap_pos: IVec3,
    ) -> bool {
        match (self, other) {
            (BlockKind::Walking(kind), BlockKind::Walking(other_kind)) => {
                walking_walking_overlap_allowed(kind, self_pos, other_kind, other_pos, overlap_pos)
            }
            (BlockKind::Walking(kind), other) => {
                walking_other_overlap_allowed(kind, self_pos, other, other_pos, overlap_pos)
            }
            (other, BlockKind::Walking(kind)) => {
                walking_other_overlap_allowed(kind, other_pos, other, self_pos, overlap_pos)
            }
            _ => false,
        }
    }

    /// Returns the cells that can host pipe connections, as offsets from the
    /// block position.
    pub fn connectable_offsets(self) -> Offsets {
        match self {
            BlockKind::Walking(kind) => kind.connectable_offsets().into(),
            BlockKind::PatchRotation(kind) => kind.connectable_offsets().into(),
            _ => smallvec![IVec3::ZERO],
        }
    }

    /// Returns the block's per-axis boundary bases, or `None` for kinds without
    /// a fixed boundary frame (`Y`, measurement, `Port`, `T`, and selective blocks).
    pub fn bases(self) -> Option<[Basis; 3]> {
        match self {
            BlockKind::Cube(kind) => Some(kind.bases()),
            BlockKind::Walking(kind) => Some(kind.boundary().bases()),
            BlockKind::PatchRotation(kind) => kind.bases_at_endpoint(IVec3::ZERO, IVec3::ZERO),
            BlockKind::Y
            | BlockKind::Measurement(_)
            | BlockKind::Port
            | BlockKind::T
            | BlockKind::Selective(_) => None,
        }
    }

    /// Returns the boundary bases exposed at `endpoint`, or `None` if the block
    /// exposes no connectable endpoint there.
    pub fn bases_at_endpoint(self, block_pos: IVec3, endpoint: IVec3) -> Option<[Basis; 3]> {
        match self {
            BlockKind::Cube(kind) => (endpoint == block_pos).then_some(kind.bases()),
            BlockKind::Walking(kind) => kind
                .connectable_offsets()
                .into_iter()
                .filter_map(|offset| crate::checked_add_position(block_pos, offset).ok())
                .any(|position| position == endpoint)
                .then_some(kind.boundary().bases()),
            BlockKind::PatchRotation(kind) => kind.bases_at_endpoint(block_pos, endpoint),
            BlockKind::Y
            | BlockKind::Measurement(_)
            | BlockKind::Port
            | BlockKind::T
            | BlockKind::Selective(_) => None,
        }
    }

    /// Returns the bases on the pipe face at `endpoint`.
    ///
    /// Matches [`bases_at_endpoint`](Self::bases_at_endpoint) except for patch
    /// rotation, whose pipe face uses the construction basis directly.
    pub fn pipe_face_bases_at_endpoint(
        self,
        block_pos: IVec3,
        endpoint: IVec3,
    ) -> Option<[Basis; 3]> {
        match self {
            BlockKind::PatchRotation(kind) => kind.pipe_face_bases_at_endpoint(block_pos, endpoint),
            kind => kind.bases_at_endpoint(block_pos, endpoint),
        }
    }
}

fn walking_other_overlap_allowed(
    kind: WalkingKind,
    walking_pos: IVec3,
    other: BlockKind,
    other_pos: IVec3,
    overlap_pos: IVec3,
) -> bool {
    let Ok(walking_end) = kind.try_end_position(walking_pos) else {
        return false;
    };
    if overlap_pos == walking_pos || overlap_pos == walking_end {
        return false;
    }
    match other {
        BlockKind::Port => other_pos.z == walking_pos.z,
        _ => false,
    }
}

fn walking_walking_overlap_allowed(
    kind: WalkingKind,
    pos: IVec3,
    other_kind: WalkingKind,
    other_pos: IVec3,
    overlap_pos: IVec3,
) -> bool {
    if kind.movement() != other_kind.movement() || pos.z != other_pos.z {
        return false;
    }
    let (Ok(end), Ok(other_end)) = (
        kind.try_end_position(pos),
        other_kind.try_end_position(other_pos),
    ) else {
        return false;
    };
    let hard_endpoint = overlap_pos == pos || overlap_pos == end;
    let other_hard_endpoint = overlap_pos == other_pos || overlap_pos == other_end;
    !(hard_endpoint && other_hard_endpoint)
}

impl std::str::FromStr for BlockKind {
    type Err = BlockError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_uppercase();
        if let Ok(cube_kind) = s.parse::<CubeKind>() {
            Ok(BlockKind::Cube(cube_kind))
        } else if s == "Y" {
            Ok(BlockKind::Y)
        } else if s == "X" {
            Ok(BlockKind::Measurement(Basis::X))
        } else if s == "Z" {
            Ok(BlockKind::Measurement(Basis::Z))
        } else if s == "PORT" {
            Ok(BlockKind::Port)
        } else if s == "T" {
            Ok(BlockKind::T)
        } else if let Ok(selective_kind) = s.parse::<SelectiveKind>() {
            Ok(BlockKind::Selective(selective_kind))
        } else {
            Err(BlockError::InvalidBlockKind(s))
        }
    }
}

fn invalid_selective_kind(value: &str) -> String {
    format!("Invalid SelectiveKind: {}", value.to_uppercase())
}

/// Returns whether `tag` is a legal block/pipe tag.
///
/// Tags are non-empty and contain no whitespace, controls, `<`, or `>`.
pub fn is_valid_tag(tag: &str) -> bool {
    !tag.is_empty() && tag.chars().all(is_tag_char)
}

pub(crate) fn is_tag_char(c: char) -> bool {
    !c.is_whitespace() && !c.is_control() && !matches!(c, '<' | '>')
}

/// A single block placed at a 3D lattice position in a [`BlockGraph`](crate::BlockGraph).
///
/// # Examples
///
/// ```
/// use bloq_graph::{Block, BlockKind, CubeKind};
/// use glam::IVec3;
///
/// let block = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)).with_tag("entry")?;
/// assert_eq!(block.pos(), IVec3::ZERO);
/// assert_eq!(block.kind(), BlockKind::Cube(CubeKind::ZXZ));
/// assert_eq!(block.tag(), Some("entry"));
/// # Ok::<(), bloq_graph::BlockGraphError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Block {
    pub(crate) pos: IVec3,
    pub(crate) kind: BlockKind,
    pub(crate) height: CubeHeight,
    pub(crate) port_color: Option<RGBA>,
    pub(crate) port_role: PortRole,
    pub(crate) tag: String,
}

impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BlockKind::Walking(kind) => {
                write!(f, "walk {} {} ->", kind.boundary(), self.pos)?;
                match kind.try_end_position(self.pos) {
                    Ok(end) => write!(f, " {end}")?,
                    Err(_) => write!(f, " [endpoint overflows i32]")?,
                }
            }
            BlockKind::PatchRotation(kind) => {
                write!(f, "rotate {} {} ->", kind.basis(), self.pos)?;
                match kind.try_end_position(self.pos) {
                    Ok(end) => write!(f, " {end}")?,
                    Err(_) => write!(f, " [endpoint overflows i32]")?,
                }
            }
            kind => {
                write!(f, "{} {}", kind, self.pos)?;
                if !self.height.is_default() && kind.is_cube() {
                    write!(f, " height={}", self.height)?;
                }
                if let Some(color) = self.port_color {
                    write!(f, " color={:02x}{:02x}{:02x}", color.r, color.g, color.b)?;
                }
                if self.port_role != PortRole::Auto {
                    write!(f, " role={}", self.port_role.as_str())?;
                }
            }
        }
        if !self.tag.is_empty() {
            write!(f, " <{}>", self.tag)?;
        }
        Ok(())
    }
}

impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Block {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag
            .is_empty()
            .cmp(&other.tag.is_empty())
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| self.pos.to_array().cmp(&other.pos.to_array()))
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.height.cmp(&other.height))
            .then_with(|| self.port_color.cmp(&other.port_color))
            .then_with(|| self.port_role.cmp(&other.port_role))
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new(IVec3::ZERO, BlockKind::default())
    }
}

impl Block {
    /// Create an untagged block.
    pub fn new(pos: impl Into<IVec3>, kind: impl Into<BlockKind>) -> Self {
        Self {
            pos: pos.into(),
            kind: kind.into(),
            height: CubeHeight::DEFAULT,
            port_color: None,
            port_role: PortRole::Auto,
            tag: String::new(),
        }
    }

    /// Returns the block's anchor position.
    pub fn pos(&self) -> IVec3 {
        self.pos
    }

    /// Returns the block kind.
    pub fn kind(&self) -> BlockKind {
        self.kind
    }

    /// Returns the symbolic cube height, always [`CubeHeight::DEFAULT`] for
    /// non-cube kinds.
    pub fn height(&self) -> CubeHeight {
        if self.kind.is_cube() {
            self.height
        } else {
            CubeHeight::DEFAULT
        }
    }

    /// Syndrome duration that must agree across a spatial connection.
    pub(crate) fn spatial_height(&self) -> Option<CubeHeight> {
        (self.kind.is_cube() || self.kind.is_planar_rotation()).then(|| self.height())
    }

    /// Returns how many lattice cells the block occupies along the time axis:
    /// the *footprint* half of [`height`](Self::height), driving positioning,
    /// occupancy, connectivity and visualization. Not the compiled round count
    /// ([`CubeHeight::rounds`](crate::CubeHeight::rounds)) — `height=d/2` occupies
    /// one cell but compiles to `ceil(d/2)` rounds.
    pub fn height_cells(&self) -> u32 {
        self.height().cells()
    }

    /// Returns the block with its cube height set to `height`.
    ///
    /// # Errors
    ///
    /// Returns a [`BlockError`] if the block is not a cube, or if the
    /// resulting footprint leaves the coordinate range.
    pub fn with_height(mut self, height: CubeHeight) -> Result<Self, BlockError> {
        self.set_height_checked(height)?;
        Ok(self)
    }

    /// Returns this Port's display color, or `None` for non-Ports.
    pub fn port_color(&self) -> Option<RGBA> {
        self.kind
            .is_port()
            .then_some(self.port_color.unwrap_or(RGBA::PORT_GRAY))
    }

    /// Sets this Port's RGB display color. Alpha stays fixed and default gray is
    /// omitted from serialization.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::PortColorOnNonPort`] for a non-Port block.
    pub fn with_port_color(mut self, rgb: [u8; 3]) -> Result<Self, BlockError> {
        self.set_port_color_checked(rgb)?;
        Ok(self)
    }

    /// Returns this Port's assigned direction, or `None` for non-Ports.
    pub fn port_role(&self) -> Option<PortRole> {
        self.kind.is_port().then_some(self.port_role)
    }

    /// Assigns this Port's role.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::PortRoleOnNonPort`] if this is not a Port block.
    pub fn with_port_role(mut self, role: PortRole) -> Result<Self, BlockError> {
        self.set_port_role_checked(role)?;
        Ok(self)
    }

    /// Returns the nonempty user tag, if present.
    pub fn tag(&self) -> Option<&str> {
        (!self.tag.is_empty()).then_some(self.tag.as_str())
    }

    /// Returns the block tagged with `tag`. An empty `tag` clears the tag.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::InvalidTag`] unless `tag` is empty or
    /// satisfies [`is_valid_tag`]; an invalid tag would not survive a
    /// [`to_blog_text`](crate::BlockGraph::to_blog_text) round-trip.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Result<Self, BlockGraphError> {
        self.set_tag(tag)?;
        Ok(self)
    }

    /// Returns a copy of the block translated by `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the translated footprint exceeds the coordinate range. Use
    /// [`try_with_shift`](Self::try_with_shift) for untrusted offsets.
    pub fn with_shift(&self, offset: impl Into<IVec3>) -> Self {
        self.try_with_shift(offset)
            .expect("shifted block footprint must fit in the i32 coordinate range")
    }

    /// Returns a copy of the block translated by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if any translated
    /// footprint cell exceeds the coordinate range.
    pub fn try_with_shift(&self, offset: impl Into<IVec3>) -> Result<Self, BlockGraphError> {
        let pos = crate::checked_add_position(self.pos, offset.into())?;
        for relative in self.reserved_offsets() {
            crate::checked_add_position(pos, relative)?;
        }
        let mut shifted = self.clone();
        shifted.pos = pos;
        Ok(shifted)
    }

    /// Returns every cell the block occupies, as offsets from its position.
    ///
    /// A cube reserves one cell per height cell; other kinds defer to
    /// [`BlockKind::reserved_offsets`].
    pub fn reserved_offsets(&self) -> Offsets {
        match self.kind {
            BlockKind::Cube(_) => (0..self.height_cells_i32())
                .map(|z| IVec3::new(0, 0, z))
                .collect(),
            kind => kind.reserved_offsets(),
        }
    }

    pub(crate) fn checked_reserved_positions(&self) -> Result<Vec<IVec3>, BlockGraphError> {
        self.reserved_offsets()
            .into_iter()
            .map(|offset| crate::checked_add_position(self.pos, offset))
            .collect()
    }

    /// Returns the cells that can host pipe connections, as offsets from the
    /// block position.
    ///
    /// A multi-cell cube exposes its bottom and top cells; other kinds defer to
    /// [`BlockKind::connectable_offsets`].
    pub fn connectable_offsets(&self) -> Offsets {
        match self.kind {
            BlockKind::Cube(_) if self.height_cells() > 1 => {
                smallvec![IVec3::ZERO, IVec3::new(0, 0, self.height_cells_i32() - 1)]
            }
            kind => kind.connectable_offsets(),
        }
    }

    /// Iterates over the absolute cells the block reserves, skipping any that
    /// would leave the `i32` lattice.
    fn reserved_positions(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.reserved_offsets()
            .into_iter()
            .filter_map(|offset| crate::checked_add_position(self.pos, offset).ok())
    }

    /// Iterates over the absolute cells the block exposes for pipe connections,
    /// skipping any that would leave the `i32` lattice.
    fn connectable_positions(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.connectable_offsets()
            .into_iter()
            .filter_map(|offset| crate::checked_add_position(self.pos, offset).ok())
    }

    /// Returns whether the block reserves the cell at `position`.
    pub(crate) fn occupies_position(&self, position: IVec3) -> bool {
        self.reserved_positions()
            .any(|occupied| occupied == position)
    }

    /// Returns whether the block reserves any cell in time layer `z`.
    pub fn occupies_layer(&self, z: i32) -> bool {
        self.reserved_offsets()
            .into_iter()
            .filter_map(|offset| self.pos.z.checked_add(offset.z))
            .any(|occupied_z| occupied_z == z)
    }

    /// Returns the boundary bases exposed at the connectable `endpoint`, or
    /// `None` if the block exposes no endpoint there.
    pub fn bases_at_endpoint(&self, endpoint: IVec3) -> Option<[Basis; 3]> {
        match self.kind {
            BlockKind::Cube(kind) => self
                .connectable_positions()
                .any(|position| position == endpoint)
                .then_some(kind.bases()),
            kind => kind.bases_at_endpoint(self.pos, endpoint),
        }
    }

    /// Returns the exposed connectable endpoint facing `dir`.
    ///
    /// Multi-span blocks can reserve interior cells that are not valid external
    /// endpoints; for example a multi-cell cube resolves `+Z` to its top cell.
    pub fn endpoint_for_direction(&self, dir: Direction) -> IVec3 {
        if let BlockKind::PatchRotation(kind) = self.kind
            && let Some(offset) = kind.planar_endpoint_offset(dir)
        {
            return crate::checked_add_position(self.pos, offset).unwrap_or(self.pos);
        }
        self.connectable_positions()
            .find(|endpoint| {
                crate::checked_add_position(*endpoint, dir.to_ivec3())
                    .ok()
                    .is_none_or(|neighbor| !self.occupies_position(neighbor))
            })
            .unwrap_or(self.pos)
    }

    pub(crate) fn set_kind(&mut self, kind: BlockKind) {
        self.kind = kind;
        if !self.kind.is_cube() {
            self.height = CubeHeight::DEFAULT;
        }
        if !self.kind.is_port() {
            self.port_color = None;
            self.port_role = PortRole::Auto;
        }
    }

    // An empty tag means "untagged" (the writer omits it), so it clears rather
    // than errors — the editor clears tags by submitting an empty string.
    pub(crate) fn set_tag(&mut self, tag: impl Into<String>) -> Result<(), BlockGraphError> {
        let tag = tag.into();
        if !tag.is_empty() && !is_valid_tag(&tag) {
            return Err(BlockGraphError::InvalidTag { tag });
        }
        self.tag = tag;
        Ok(())
    }

    fn set_height_checked(&mut self, height: CubeHeight) -> Result<(), BlockError> {
        if !self.kind.is_cube() {
            return Err(BlockError::CubeHeightOnNonCube(self.kind));
        }
        // `CubeHeight::new` is the only constructor and already caps cells at
        // `MAX_CUBE_HEIGHT_CELLS`, so `cells - 1` always fits an `i32`.
        let cells = height.cells();
        if self.pos.z.checked_add(cells as i32 - 1).is_none() {
            return Err(BlockError::CubeHeightCoordinateOverflow {
                z: self.pos.z,
                cells,
            });
        }
        self.height = height;
        Ok(())
    }

    pub(crate) fn set_port_color_checked(&mut self, rgb: [u8; 3]) -> Result<(), BlockError> {
        if !self.kind.is_port() {
            return Err(BlockError::PortColorOnNonPort(self.kind));
        }
        let color = RGBA {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2],
            a: RGBA::PORT_GRAY.a,
        };
        self.port_color = (color != RGBA::PORT_GRAY).then_some(color);
        Ok(())
    }

    pub(crate) fn set_port_role_checked(&mut self, role: PortRole) -> Result<(), BlockError> {
        if !self.kind.is_port() {
            return Err(BlockError::PortRoleOnNonPort(self.kind));
        }
        self.port_role = role;
        Ok(())
    }

    fn height_cells_i32(&self) -> i32 {
        i32::try_from(self.height_cells()).expect("cube height cells are bounded by 65535")
    }
}

/// A directed pipe connecting one block to an adjacent block.
///
/// # Examples
///
/// ```
/// use bloq_graph::{Direction, Pipe};
/// use glam::IVec3;
///
/// let pipe = Pipe::new(IVec3::ZERO, Direction::ZPLUS)
///     .with_hadamard()
///     .with_tag("bridge")?;
/// assert_eq!(pipe.src(), IVec3::ZERO);
/// assert_eq!(pipe.dir(), Direction::ZPLUS);
/// assert!(pipe.is_hadamard());
/// assert_eq!(pipe.tag(), Some("bridge"));
/// # Ok::<(), bloq_graph::BlockGraphError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Pipe {
    pub(crate) src: IVec3,
    pub(crate) dir: Direction,
    pub(crate) hadamard: bool,
    pub(crate) tag: String,
}

impl std::fmt::Display for Pipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display_with_ids(|_| None))
    }
}

impl Pipe {
    pub(crate) fn display_with_ids(&self, p2i: impl Fn(IVec3) -> Option<u32>) -> String {
        let source = p2i(self.src)
            .map(|id| id.to_string())
            .unwrap_or_else(|| self.src.to_string());
        let tag = if self.tag.is_empty() {
            String::new()
        } else {
            format!(" <{}>", self.tag)
        };
        format!(
            "{source} {} {}{tag}",
            if self.hadamard { "-H>" } else { "->" },
            self.dir
        )
    }

    /// Create an untagged, non-Hadamard pipe.
    pub fn new(src: impl Into<IVec3>, dir: impl Into<Direction>) -> Self {
        Self {
            src: src.into(),
            dir: dir.into(),
            hadamard: false,
            tag: String::new(),
        }
    }

    /// Returns the pipe's canonical source position.
    pub fn src(&self) -> IVec3 {
        self.src
    }

    /// Returns the direction from source to destination.
    pub fn dir(&self) -> Direction {
        self.dir
    }

    /// Returns whether the pipe carries a Hadamard basis change.
    pub fn is_hadamard(&self) -> bool {
        self.hadamard
    }

    /// Returns the nonempty user tag, if present.
    pub fn tag(&self) -> Option<&str> {
        (!self.tag.is_empty()).then_some(self.tag.as_str())
    }

    /// Returns this pipe marked as a Hadamard pipe.
    pub fn with_hadamard(mut self) -> Self {
        self.hadamard = true;
        self
    }

    /// Returns the pipe tagged with `tag`. An empty `tag` clears the tag.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::InvalidTag`] unless `tag` is empty or
    /// satisfies [`is_valid_tag`]; an invalid tag would not survive a
    /// [`to_blog_text`](crate::BlockGraph::to_blog_text) round-trip.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Result<Self, BlockGraphError> {
        self.set_tag(tag)?;
        Ok(self)
    }

    /// Returns the source and destination positions as a `(src, dst)` pair.
    ///
    /// # Panics
    ///
    /// Panics if stepping from the source exceeds the coordinate range. Use
    /// [`try_endpoints`](Self::try_endpoints) for untrusted pipes.
    pub fn endpoints(&self) -> (IVec3, IVec3) {
        self.try_endpoints()
            .expect("pipe destination must fit in the i32 coordinate range")
    }

    /// Returns the source and destination positions as a `(src, dst)` pair.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if stepping from the
    /// source exceeds the coordinate range.
    pub fn try_endpoints(&self) -> Result<(IVec3, IVec3), BlockGraphError> {
        Ok((self.src, self.try_dst()?))
    }

    /// Returns the destination position (`src` stepped one cell along `dir`).
    ///
    /// # Panics
    ///
    /// Panics if stepping from the source exceeds the coordinate range. Use
    /// [`try_dst`](Self::try_dst) for untrusted pipes.
    pub fn dst(&self) -> IVec3 {
        self.try_dst()
            .expect("pipe destination must fit in the i32 coordinate range")
    }

    /// Returns the destination position (`src` stepped one cell along `dir`).
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if stepping from the
    /// source exceeds the coordinate range.
    pub fn try_dst(&self) -> Result<IVec3, BlockGraphError> {
        crate::checked_add_position(self.src, self.dir.to_ivec3())
    }

    /// Returns a copy of the pipe translated by `offset`.
    ///
    /// # Panics
    ///
    /// Panics if either translated endpoint exceeds the coordinate range. Use
    /// [`try_with_shift`](Self::try_with_shift) for untrusted offsets.
    pub fn with_shift(&self, offset: impl Into<IVec3>) -> Self {
        self.try_with_shift(offset)
            .expect("shifted pipe endpoints must fit in the i32 coordinate range")
    }

    /// Returns a copy of the pipe translated by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::CoordinateOverflow`] if either translated
    /// endpoint exceeds the coordinate range.
    pub fn try_with_shift(&self, offset: impl Into<IVec3>) -> Result<Self, BlockGraphError> {
        let offset = offset.into();
        let src = crate::checked_add_position(self.src, offset)?;
        crate::checked_add_position(src, self.dir.to_ivec3())?;
        Ok(Self {
            src,
            dir: self.dir,
            hadamard: self.hadamard,
            tag: self.tag.clone(),
        })
    }

    // See `Block::set_tag`: an empty tag clears rather than errors.
    pub(crate) fn set_tag(&mut self, tag: impl Into<String>) -> Result<(), BlockGraphError> {
        let tag = tag.into();
        if !tag.is_empty() && !is_valid_tag(&tag) {
            return Err(BlockGraphError::InvalidTag { tag });
        }
        self.tag = tag;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_order_prefers_tagged_blocks_then_tag_then_position() {
        let mut blocks = [
            Block::new(IVec3::new(2, 0, 0), BlockKind::Port),
            Block::new(IVec3::new(1, 0, 0), BlockKind::Port)
                .with_tag("b")
                .expect("valid tag"),
            Block::new(IVec3::new(0, 0, 0), BlockKind::Port)
                .with_tag("a")
                .expect("valid tag"),
        ];

        blocks.sort();

        assert_eq!(
            blocks.iter().map(|block| block.pos).collect::<Vec<_>>(),
            vec![
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(2, 0, 0)
            ]
        );
    }

    #[test]
    fn test_block_order_breaks_ties_by_kind() {
        let mut blocks = [
            Block::new(IVec3::ZERO, BlockKind::Y)
                .with_tag("tag")
                .expect("valid tag"),
            Block::new(IVec3::ZERO, BlockKind::Port)
                .with_tag("tag")
                .expect("valid tag"),
        ];

        blocks.sort();

        assert_eq!(blocks[0].kind, BlockKind::Y);
        assert_eq!(blocks[1].kind, BlockKind::Port);
    }

    #[test]
    fn patch_rotation_x_axis_boundary_basis_matches_construction_frame() {
        for (basis, movement, expected) in [
            (Basis::X, IVec2::NEG_X, Basis::Z),
            (Basis::X, IVec2::X, Basis::Z),
            (Basis::X, IVec2::NEG_Y, Basis::X),
            (Basis::X, IVec2::Y, Basis::X),
            (Basis::Z, IVec2::NEG_X, Basis::X),
            (Basis::Z, IVec2::X, Basis::X),
            (Basis::Z, IVec2::NEG_Y, Basis::Z),
            (Basis::Z, IVec2::Y, Basis::Z),
        ] {
            assert_eq!(
                PatchRotationKind::new(basis, movement)
                    .unwrap()
                    .x_axis_boundary_basis(),
                expected
            );
        }
    }

    #[test]
    fn test_block_kind_from_str_stays_canonical_for_selectives() {
        assert_eq!(
            "YZ".parse::<BlockKind>().unwrap(),
            BlockKind::Selective(SelectiveKind::YZ)
        );
        "ZY".parse::<BlockKind>().unwrap_err();
    }

    #[test]
    fn measurement_kind_round_trips_and_flips() {
        for (text, basis) in [("X", Basis::X), ("Z", Basis::Z)] {
            let kind = BlockKind::Measurement(basis);
            assert_eq!(text.parse::<BlockKind>().unwrap(), kind);
            assert_eq!(kind.to_string(), text);
            assert_eq!(kind.flip_xz_basis(), BlockKind::Measurement(basis.flip()));
        }
        for old in ["MX", "MZ", "TransversalX", "TransversalZ"] {
            old.parse::<BlockKind>().unwrap_err();
        }
    }

    #[test]
    fn with_tag_rejects_tags_the_parser_cannot_read_back() {
        for bad in ["has space", "angle>bracket", "angle<bracket", "nul\0byte"] {
            let block = Block::new(IVec3::ZERO, BlockKind::Port).with_tag(bad);
            assert!(
                matches!(block, Err(BlockGraphError::InvalidTag { .. })),
                "tag {bad:?} should be rejected"
            );
            let pipe = Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_tag(bad);
            assert!(
                matches!(pipe, Err(BlockGraphError::InvalidTag { .. })),
                "tag {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn empty_tag_clears_instead_of_erroring() {
        // The editor clears a tag by submitting an empty string; internally an
        // empty tag already means "untagged" and the writer omits it.
        let block = Block::new(IVec3::ZERO, BlockKind::Port)
            .with_tag("kept")
            .expect("valid tag")
            .with_tag("")
            .expect("empty tag clears");
        assert_eq!(block.tag(), None);

        let pipe = Pipe::new(IVec3::ZERO, Direction::ZPLUS)
            .with_tag("kept")
            .expect("valid tag")
            .with_tag("")
            .expect("empty tag clears");
        assert_eq!(pipe.tag(), None);
    }

    #[test]
    fn with_tag_accepts_symbols() {
        for tag in ["if", "0", "^", "()", "π"] {
            let block = Block::new(IVec3::ZERO, BlockKind::Port)
                .with_tag(tag)
                .unwrap();
            assert_eq!(block.tag(), Some(tag));
        }
    }

    #[test]
    fn cube_height_footprint_is_bounded_before_reserved_offset_allocation() {
        let near_top = Block::new(IVec3::new(0, 0, i32::MAX), BlockKind::Cube(CubeKind::ZXZ));
        assert!(matches!(
            near_top
                .clone()
                .with_height("2d".parse().expect("valid height")),
            Err(BlockError::CubeHeightCoordinateOverflow { .. })
        ));
        // A sub-unit height stays single-cell, so it fits where `2d` does not.
        near_top
            .with_height("d/2".parse().expect("valid height"))
            .unwrap();
    }

    #[test]
    fn cube_height_is_cube_only_and_resets_with_the_kind() {
        let port = Block::new(IVec3::ZERO, BlockKind::Port);
        assert!(matches!(
            port.with_height("2d".parse().expect("valid height")),
            Err(BlockError::CubeHeightOnNonCube(BlockKind::Port))
        ));

        let mut cube = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
            .with_height("2d".parse().expect("valid height"))
            .expect("cube accepts a height");
        assert_eq!(cube.height_cells(), 2);
        cube.set_kind(BlockKind::Y);
        assert_eq!(cube.height(), CubeHeight::DEFAULT);
        assert_eq!(cube.height_cells(), 1);
    }

    #[test]
    fn port_color_is_port_only_and_keeps_default_alpha() {
        let mut port = Block::new(IVec3::ZERO, BlockKind::Port)
            .with_port_color([0xeb, 0x40, 0x34])
            .expect("Port accepts a color");
        assert_eq!(
            port.port_color(),
            Some(RGBA::from_hex(0xeb4034, RGBA::PORT_GRAY.a))
        );
        assert_eq!(port.to_string(), "Port [0, 0, 0] color=eb4034");

        port.set_kind(BlockKind::Y);
        assert_eq!(port.port_color(), None);
        assert!(matches!(
            port.with_port_color([0, 0, 0]),
            Err(BlockError::PortColorOnNonPort(BlockKind::Y))
        ));
    }

    #[test]
    fn cube_height_drives_cells_not_rounds() {
        let half = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ))
            .with_height("d/2".parse().expect("valid height"))
            .expect("cube accepts a height");
        // One cell, exactly like an ordinary cube: no extra reserved offsets, no
        // second connectable endpoint.
        assert_eq!(half.reserved_offsets().as_slice(), [IVec3::ZERO]);
        assert_eq!(half.connectable_offsets().as_slice(), [IVec3::ZERO]);
        assert_eq!(half.height().rounds(7), 4);

        let tall = half
            .with_height("3d/2".parse().expect("valid height"))
            .expect("cube accepts a height");
        assert_eq!(
            tall.reserved_offsets().as_slice(),
            [IVec3::ZERO, IVec3::new(0, 0, 1)]
        );
        assert_eq!(tall.height().rounds(7), 11);
    }

    #[test]
    fn display_emits_the_canonical_height_only_for_non_default_cubes() {
        let plain = Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ));
        assert_eq!(plain.to_string(), "ZXZ [0, 0, 0]");

        let scaled = plain
            .with_height("2d/4".parse().expect("valid height"))
            .expect("cube accepts a height");
        assert_eq!(scaled.to_string(), "ZXZ [0, 0, 0] height=d/2");
    }

    #[test]
    fn pipe_shift_validates_the_translated_endpoints() {
        let shifted = Pipe::new(IVec3::new(i32::MAX, 0, 0), Direction::XPLUS)
            .try_with_shift(IVec3::NEG_X)
            .expect("translation brings both endpoints into range");

        assert_eq!(
            shifted.try_endpoints().unwrap(),
            (IVec3::new(i32::MAX - 1, 0, 0), IVec3::new(i32::MAX, 0, 0),)
        );
    }

    #[test]
    fn moving_block_queries_handle_unrepresentable_endpoints() {
        let start = IVec3::splat(i32::MAX);
        let wrapped_endpoint = IVec3::splat(i32::MIN);
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::ONE).unwrap();
        let rotation = PatchRotationKind::new(Basis::X, IVec2::X).unwrap();

        walking.try_end_position(start).unwrap_err();
        rotation.try_end_position(start).unwrap_err();

        for kind in [
            BlockKind::Walking(walking),
            BlockKind::PatchRotation(rotation),
        ] {
            let block = Block::new(start, kind);
            assert!(!block.occupies_position(wrapped_endpoint));
            assert!(!block.occupies_layer(i32::MIN));
            assert!(block.bases_at_endpoint(wrapped_endpoint).is_none());
            assert!(kind.bases_at_endpoint(start, wrapped_endpoint).is_none());
            assert!(
                kind.pipe_face_bases_at_endpoint(start, wrapped_endpoint)
                    .is_none()
            );
            assert!(block.to_string().contains("[endpoint overflows i32]"));
        }
    }
}
