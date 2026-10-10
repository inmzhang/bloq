//! Block graph hierarchy, interfaces, linking, and BLOG emission.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use glam::IVec3;
use thiserror::Error;

use crate::ast::Span;
use crate::graph::LinkBranchOccupancy;
use crate::zx::{
    ProjectedExternalTable, ProjectionError, ProjectionLimits, RuntimeBasisError, StabilizerError,
    ZXGraph,
};
use crate::{
    Action, Block, BlockGraph, BlockKind, Direction, Expr, PhasedPauliString, Pipe, PortRole,
    UDirection, checked_add_position,
};

/// Direction of a quantum module port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortDirection {
    /// Quantum data enters the module.
    Input,
    /// Quantum data leaves the module.
    Output,
}

impl PortDirection {
    fn keyword(self) -> &'static str {
        match self {
            Self::Input => "in",
            Self::Output => "out",
        }
    }
}

/// A child-module quarter turn applied before translation.
///
/// BLOG writes `rotate <axis> <degrees>`; validation accepts only multiples of
/// 90 degrees and block kinds that support the resulting orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleRotation {
    axis: UDirection,
    quarter_turns: u8,
}

impl ModuleRotation {
    /// Rotation that leaves every lattice axis unchanged.
    pub const IDENTITY: Self = Self {
        axis: UDirection::Z,
        quarter_turns: 0,
    };

    /// Constructs a rotation, reducing `quarter_turns` modulo four.
    pub const fn new(axis: UDirection, quarter_turns: i32) -> Self {
        let quarter_turns = quarter_turns.rem_euclid(4) as u8;
        Self {
            axis: if quarter_turns == 0 {
                UDirection::Z
            } else {
                axis
            },
            quarter_turns,
        }
    }

    /// Constructs a rotation from degrees, or returns `None` for a non-quarter-turn angle.
    pub const fn from_degrees(axis: UDirection, degrees: i32) -> Option<Self> {
        if degrees % 90 == 0 {
            Some(Self::new(axis, degrees / 90))
        } else {
            None
        }
    }

    /// Returns the unsigned rotation axis.
    pub const fn axis(self) -> UDirection {
        self.axis
    }

    /// Returns the clockwise quarter-turn count modulo four.
    pub const fn quarter_turns(self) -> u8 {
        self.quarter_turns
    }

    /// Returns the normalized angle in degrees.
    pub const fn degrees(self) -> u16 {
        self.quarter_turns as u16 * 90
    }

    /// Returns whether this rotation leaves the lattice unchanged.
    pub const fn is_identity(self) -> bool {
        self.quarter_turns == 0
    }

    /// The full cubic orientation represented by this axis rotation.
    pub fn orientation(self) -> ModuleOrientation {
        ModuleOrientation {
            axes: [
                self.rotate_direction(Direction::XPLUS),
                self.rotate_direction(Direction::YPLUS),
                self.rotate_direction(Direction::ZPLUS),
            ],
        }
    }

    /// Rotates a lattice position around the origin.
    ///
    /// # Errors
    ///
    /// Returns a coordinate-overflow error if the rotated position is unrepresentable.
    pub fn try_rotate_position(self, position: IVec3) -> Result<IVec3, crate::BlockGraphError> {
        self.orientation().try_rotate_position(position)
    }

    /// Rotates an axis-aligned direction.
    ///
    /// # Panics
    ///
    /// Panics if an internally constructed cubic rotation stops being axis-aligned.
    pub fn rotate_direction(self, direction: Direction) -> Direction {
        let mut position = direction.to_ivec3();
        for _ in 0..self.quarter_turns {
            position = match self.axis {
                UDirection::X => IVec3::new(position.x, -position.z, position.y),
                UDirection::Y => IVec3::new(position.z, position.y, -position.x),
                UDirection::Z => IVec3::new(-position.y, position.x, position.z),
            };
        }
        Direction::try_from(position).expect("a cubic rotation preserves axis directions")
    }
}

impl Default for ModuleRotation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Display for ModuleRotation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.axis, self.degrees())
    }
}

/// A composed proper orientation of the cubic lattice.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleOrientation {
    /// Images of the local positive X, Y, and Z axes.
    axes: [Direction; 3],
}

impl ModuleOrientation {
    pub const IDENTITY: Self = Self {
        axes: [Direction::XPLUS, Direction::YPLUS, Direction::ZPLUS],
    };

    /// Composes this parent orientation with one child-local rotation.
    pub fn then(self, rotation: ModuleRotation) -> Self {
        self.then_orientation(rotation.orientation())
    }

    /// Composes this parent orientation with an already-composed child orientation.
    pub fn then_orientation(self, child: Self) -> Self {
        Self {
            axes: child.axes.map(|axis| self.rotate_direction(axis)),
        }
    }

    /// Rotates a lattice position with this orientation.
    ///
    /// # Errors
    ///
    /// Returns a coordinate-overflow error if a negated component is unrepresentable.
    pub fn try_rotate_position(self, position: IVec3) -> Result<IVec3, crate::BlockGraphError> {
        let mut rotated = [0; 3];
        for (component, direction) in position.to_array().into_iter().zip(self.axes) {
            let value = if matches!(
                direction,
                Direction::XMINUS | Direction::YMINUS | Direction::ZMINUS
            ) {
                component.checked_neg().ok_or(
                    crate::BlockGraphError::CoordinateRotationOverflow {
                        position,
                        axis: direction.as_udirection(),
                    },
                )?
            } else {
                component
            };
            rotated[direction.as_udirection().index()] = value;
        }
        Ok(IVec3::from_array(rotated))
    }

    /// Rotates, then translates, a lattice position.
    ///
    /// # Errors
    ///
    /// Returns a coordinate-overflow error if either operation is unrepresentable.
    pub fn try_transform_position(
        self,
        position: IVec3,
        translation: IVec3,
    ) -> Result<IVec3, crate::BlockGraphError> {
        checked_add_position(self.try_rotate_position(position)?, translation)
    }

    pub fn rotate_direction(self, direction: Direction) -> Direction {
        let mapped = self.axes[direction.as_udirection().index()];
        if matches!(
            direction,
            Direction::XMINUS | Direction::YMINUS | Direction::ZMINUS
        ) {
            mapped.negate()
        } else {
            mapped
        }
    }

    pub(crate) fn rotate_axis_values<T: Copy>(self, values: [T; 3]) -> [T; 3] {
        let mut rotated = values;
        for (local, direction) in self.axes.into_iter().enumerate() {
            rotated[direction.as_udirection().index()] = values[local];
        }
        rotated
    }

    pub(crate) const fn preserves_time_direction(self) -> bool {
        matches!(self.axes[2], Direction::ZPLUS)
    }

    pub(crate) fn is_coordinate_half_turn(self) -> bool {
        self.axes
            .iter()
            .enumerate()
            .all(|(axis, direction)| direction.as_udirection().index() == axis)
    }

    fn sort_key(self) -> [usize; 3] {
        self.axes.map(Direction::index)
    }
}

impl Default for ModuleOrientation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// One named quantum boundary of a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantumPort {
    /// Public port name.
    pub name: String,
    /// Definition-local port-block position.
    pub position: IVec3,
    /// Whether quantum data enters or leaves the module.
    pub direction: PortDirection,
    /// Logical resource type carried by the port.
    pub resource_type: String,
}

/// One exported Boolean expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitOutput {
    /// Public output name.
    pub name: String,
    /// Definition-local Boolean expression being exported.
    pub expr: Expr,
}

/// Public quantum and classical interface of a module.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleInterface {
    /// Quantum ports in declaration order.
    pub quantum_ports: Vec<QuantumPort>,
    /// Classical bit-input names in declaration order.
    pub bit_inputs: Vec<String>,
    /// Classical bit outputs in declaration order.
    pub bit_outputs: Vec<BitOutput>,
}

/// One rotated, then translated, use of another module definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleInstance {
    /// Parent-local instance name.
    pub name: String,
    /// Referenced module definition name.
    pub definition: String,
    /// Rotation applied before translation.
    pub rotation: ModuleRotation,
    /// Translation from definition-local into parent coordinates.
    pub translation: IVec3,
}

impl ModuleInstance {
    /// Returns the instance's cubic lattice orientation.
    pub fn orientation(&self) -> ModuleOrientation {
        self.rotation.orientation()
    }

    /// Maps a definition-local position into this instance.
    ///
    /// # Errors
    ///
    /// Returns a coordinate-overflow error if the transformed position is unrepresentable.
    pub fn try_transform_position(&self, position: IVec3) -> Result<IVec3, crate::BlockGraphError> {
        self.orientation()
            .try_transform_position(position, self.translation)
    }
}

/// A named port on one child instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstancePort {
    /// Parent-local child instance name.
    pub instance: String,
    /// Public port name on that child definition.
    pub port: String,
}

/// Explicit quantum composition authored by the parent module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantumConnection {
    /// Connects a parent block to a child quantum input.
    Input {
        /// Parent-local port-block position.
        block: IVec3,
        /// Child input endpoint.
        input: InstancePort,
        /// Whether the seam changes basis by Hadamard.
        hadamard: bool,
    },
    /// Connects a child quantum output to a parent block.
    Output {
        /// Child output endpoint.
        output: InstancePort,
        /// Parent-local port-block position.
        block: IVec3,
        /// Whether the seam changes basis by Hadamard.
        hadamard: bool,
    },
    /// Connects one child output directly to another child input.
    Pipe {
        /// Child output endpoint.
        output: InstancePort,
        /// Child input endpoint.
        input: InstancePort,
        /// Whether the seam changes basis by Hadamard.
        hadamard: bool,
    },
}

/// A Boolean value in the parent or exported by a child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitRef {
    /// Child instance qualifier, or `None` for a parent-local value.
    pub instance: Option<String>,
    /// Classical bit name.
    pub bit: String,
}

/// One explicit binding to a child Boolean input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitBinding {
    /// Parent-local or child-exported source value.
    pub source: BitRef,
    /// Child instance receiving the value.
    pub target_instance: String,
    /// Classical input name on the target child.
    pub target_bit: String,
}

/// Origin of one block in a materialized module hierarchy.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedModuleSite {
    /// Owning module definition name.
    pub definition: String,
    /// Dot-qualified path from the root instance.
    pub instance_path: String,
    /// Block position inside its owning definition.
    pub local_position: IVec3,
    /// Composed definition-to-root orientation.
    pub orientation: ModuleOrientation,
}

impl std::fmt::Display for MaterializedModuleSite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.instance_path.is_empty() {
            write!(
                formatter,
                "definition '{}' local block {}",
                self.definition, self.local_position
            )
        } else {
            write!(
                formatter,
                "instance '{}' (definition '{}') local block {}",
                self.instance_path, self.definition, self.local_position
            )
        }
    }
}

/// Limits applied before and during leaf-module certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleCertificationLimits {
    /// Maximum blocks in one expanded definition, including both branch arms
    /// and child Ports before seam removal. Also caps the aggregate blocks
    /// retained by one guarded topology's projection cache and local views.
    pub max_expanded_blocks: usize,
    /// Maximum module instances in one expanded definition, including its root.
    pub max_expanded_instances: usize,
    /// Maximum sum of occupied footprint cells in one expanded definition.
    /// Overlapping branch alternatives count separately. Also caps the aggregate
    /// retained footprints in one guarded topology's projection cache and local views.
    pub max_occupied_cells: usize,
    /// Maximum ZX columns in one leaf definition.
    pub max_local_columns: usize,
    /// Maximum allocated nodes in one Boolean arena, including dead intermediates
    /// until explicit garbage collection.
    pub max_boolean_nodes: usize,
    /// Maximum cumulative Boolean, row, and query work in one decision arena.
    /// Unlimited by default; set a finite value for an explicit work budget.
    pub max_boolean_steps: usize,
    /// Maximum accounted dense matrix storage, in 64-bit words.
    pub max_matrix_words: usize,
    /// Maximum public interface or live composition width. Independent parent
    /// connectors are projected and joined separately under the same limit.
    pub max_frontier_width: usize,
    /// Maximum nodes in one correlation-witness proof arena.
    /// Guarded readout planning reclaims dead functions at reduction boundaries
    /// before checking this limit; transient operations may allocate more.
    pub max_witness_nodes: usize,
    /// Maximum attempts in one normalization pass. Internal prefixes, rejected
    /// candidates, and terminal solves are charged before affine work.
    pub max_normalization_states: usize,
    /// Maximum distinct patterns in one guarded local topology/Pauli query.
    /// Continuing branches never enumerate the joint selector assignment space.
    pub max_guarded_domain_size: usize,
}

impl ModuleCertificationLimits {
    /// Production limits. Cumulative Boolean work is unlimited; allocation,
    /// materialization, and normalization-search limits remain finite.
    pub const DEFAULT: Self = Self {
        max_expanded_blocks: 1_000_000,
        max_expanded_instances: 1_000_000,
        max_occupied_cells: 4_000_000,
        max_local_columns: 1_000_000,
        max_boolean_nodes: 4_000_000,
        max_boolean_steps: usize::MAX,
        max_matrix_words: 16_777_216,
        max_frontier_width: 4_096,
        max_witness_nodes: 1_000_000,
        max_normalization_states: 1_000_000,
        max_guarded_domain_size: 4_096,
    };

    /// Disables all explicit certification limits.
    pub const UNLIMITED: Self = Self {
        max_expanded_blocks: usize::MAX,
        max_expanded_instances: usize::MAX,
        max_occupied_cells: usize::MAX,
        max_local_columns: usize::MAX,
        max_boolean_nodes: usize::MAX,
        max_boolean_steps: usize::MAX,
        max_matrix_words: usize::MAX,
        max_frontier_width: usize::MAX,
        max_witness_nodes: usize::MAX,
        max_normalization_states: usize::MAX,
        max_guarded_domain_size: usize::MAX,
    };

    /// Field names accepted by [`set`](Self::set).
    pub const FIELD_NAMES: [&'static str; 11] = [
        "max_expanded_blocks",
        "max_expanded_instances",
        "max_occupied_cells",
        "max_local_columns",
        "max_boolean_nodes",
        "max_boolean_steps",
        "max_matrix_words",
        "max_frontier_width",
        "max_witness_nodes",
        "max_normalization_states",
        "max_guarded_domain_size",
    ];

    /// Configuration guidance included in certification resource diagnostics.
    pub const RESOURCE_LIMIT_HELP: &'static str = "increase the matching ModuleCertificationLimits field (or set it to usize::MAX); compilation accepts these through CompileConfig::with_certification_limits; see https://bloqec.com/docs/dev/api/rust/bloq_graph/struct.ModuleCertificationLimits.html";

    /// Sets a certification limit by its public field name.
    ///
    /// `usize::MAX` disables that cap; zero refuses any use of that resource.
    ///
    /// # Errors
    ///
    /// Returns [`UnknownCertificationLimit`] for a name outside [`FIELD_NAMES`](Self::FIELD_NAMES),
    /// without changing any limit.
    pub fn set(&mut self, field: &str, value: usize) -> Result<(), UnknownCertificationLimit> {
        let target = match field {
            "max_expanded_blocks" => &mut self.max_expanded_blocks,
            "max_expanded_instances" => &mut self.max_expanded_instances,
            "max_occupied_cells" => &mut self.max_occupied_cells,
            "max_local_columns" => &mut self.max_local_columns,
            "max_boolean_nodes" => &mut self.max_boolean_nodes,
            "max_boolean_steps" => &mut self.max_boolean_steps,
            "max_matrix_words" => &mut self.max_matrix_words,
            "max_frontier_width" => &mut self.max_frontier_width,
            "max_witness_nodes" => &mut self.max_witness_nodes,
            "max_normalization_states" => &mut self.max_normalization_states,
            "max_guarded_domain_size" => &mut self.max_guarded_domain_size,
            _ => return Err(UnknownCertificationLimit(field.to_owned())),
        };
        *target = value;
        Ok(())
    }

    /// Returns the Boolean arena limits used during certification.
    pub const fn boolean_limits(self) -> bloq_utils::boolean::BooleanLimits {
        bloq_utils::boolean::BooleanLimits {
            max_nodes: self.max_boolean_nodes,
            max_steps: self.max_boolean_steps,
        }
    }
}

impl Default for ModuleCertificationLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A configuration names no field in [`ModuleCertificationLimits`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "unknown compilation limit '{0}'; expected one of {expected}",
    expected = ModuleCertificationLimits::FIELD_NAMES.join(", ")
)]
pub struct UnknownCertificationLimit(
    /// Unrecognized limit field name.
    pub String,
);

/// A certified signed boundary projection for one leaf module.
#[derive(Debug, Clone)]
pub struct LeafModuleCertificate {
    module: String,
    pub(crate) table: ProjectedExternalTable,
    pub(crate) stabilizers: crate::StabilizerGenerators,
}

impl LeafModuleCertificate {
    /// Returns the certified module definition name.
    pub fn module(&self) -> &str {
        &self.module
    }

    /// Exact Pauli rows over quantum ports in interface declaration order.
    pub fn boundary_rows(&self) -> impl ExactSizeIterator<Item = &PhasedPauliString> {
        self.table.boundary_rows.iter().map(|row| &row.signed)
    }

    /// The independently certified public stabilizers for this definition.
    ///
    /// Module lowering reuses the named measurement rows instead of asking a
    /// flattened parent graph to choose new representatives across the child
    /// boundary.
    pub fn stabilizers(&self) -> &crate::StabilizerGenerators {
        &self.stabilizers
    }

    /// Expands one retained witness into the leaf ZX graph's dense column order.
    pub fn materialize_boundary_row(&self, index: usize) -> Option<PhasedPauliString> {
        let row = self.table.boundary_rows.get(index)?;
        Some(self.table.materialize(row))
    }
}

/// Error from certifying one independently authored module.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModuleCertificationError {
    /// The requested definition does not exist.
    #[error("unknown module '{0}'")]
    UnknownModule(String),
    /// Leaf certification was requested for a definition with children.
    #[error("module '{0}' contains child instances")]
    NotLeaf(String),
    /// A definition uses a feature unsupported by leaf certification.
    #[error("module '{module}' uses unsupported {feature}")]
    Unsupported {
        /// Definition being certified.
        module: String,
        /// Unsupported feature name.
        feature: &'static str,
    },
    /// An authored quantum connection is invalid.
    #[error("module '{module}': {message}")]
    InvalidConnection {
        /// Definition containing the connection.
        module: String,
        /// Connection validation diagnostic.
        message: String,
    },
    /// Definition-local graph construction failed.
    #[error("module '{module}': {source}")]
    Graph {
        /// Definition being certified.
        module: String,
        /// Underlying graph failure.
        #[source]
        source: crate::BlockGraphError,
    },
    /// Runtime-basis projection failed.
    #[error("module '{module}': {source}")]
    Runtime {
        /// Definition being certified.
        module: String,
        /// Underlying runtime-basis failure.
        #[source]
        source: RuntimeBasisError,
    },
    /// Certification exceeded an explicit resource limit.
    #[error(
        "module '{module}' exceeded {phase} limit: {observed} > {limit}; {help}",
        help = ModuleCertificationLimits::RESOURCE_LIMIT_HELP
    )]
    ResourceLimited {
        /// Definition being certified.
        module: String,
        /// Certification phase or resource name.
        phase: &'static str,
        /// Resource amount required or observed.
        observed: usize,
        /// Configured maximum.
        limit: usize,
    },
}

/// Error while loading or validating a module program.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModuleError {
    /// BLOG parsing or graph lowering failed.
    #[error("{0}")]
    Parse(#[from] crate::ParseError),
    /// An imported source could not be loaded.
    #[error("cannot load {path}: {message}")]
    Load {
        /// Import path that could not be loaded.
        path: String,
        /// Loader diagnostic.
        message: String,
    },
    /// Imports form a recursive load cycle.
    #[error("import cycle at {path}")]
    ImportCycle {
        /// Path that closes the cycle.
        path: String,
    },
    /// Source with imports was parsed without a loader.
    #[error("imports require BlockGraph::load or an explicit source resolver")]
    ImportsRequireLoader,
    /// Two definitions use the same module name.
    #[error("duplicate module '{0}'")]
    DuplicateModule(String),
    /// The program does not define its required `main` entry module.
    #[error("program must define `module main`")]
    MissingMain,
    /// A module declaration, interface, or connection is invalid.
    #[error("module '{module}': {message}")]
    InvalidModule {
        /// Definition containing the error.
        module: String,
        /// Validation diagnostic.
        message: String,
        /// Responsible source range, when known.
        span: Option<Span>,
    },
    /// Module geometry or certification is invalid.
    #[error("{source}")]
    InvalidGeometry {
        /// Definition containing the error.
        module: String,
        /// Underlying certification failure.
        #[source]
        source: ModuleCertificationError,
        /// Responsible source range, when known.
        span: Option<Span>,
    },
}

impl ModuleError {
    /// Source range responsible for this error, when parsed from BLOG text.
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::Parse(error) => error.span(),
            Self::InvalidModule { span, .. } | Self::InvalidGeometry { span, .. } => *span,
            _ => None,
        }
    }

    pub(crate) fn with_module_span(mut self, module: &str, span: Span) -> Self {
        match &mut self {
            Self::InvalidModule {
                module: name,
                span: current,
                ..
            }
            | Self::InvalidGeometry {
                module: name,
                span: current,
                ..
            } if name == module && current.is_none() => *current = Some(span),
            _ => {}
        }
        self
    }
}

impl BlockGraph {
    /// Required name of the executable root module.
    pub const ENTRY_MODULE: &'static str = "main";

    /// Installs one definition's interface and child composition on its local graph.
    pub fn definition(
        name: impl Into<String>,
        mut body: Self,
        interface: ModuleInterface,
        instances: Vec<ModuleInstance>,
        quantum_connections: Vec<QuantumConnection>,
        bit_bindings: Vec<BitBinding>,
    ) -> Self {
        body.name = name.into();
        body.interface = interface;
        body.instances = instances;
        body.quantum_connections = quantum_connections;
        body.bit_bindings = bit_bindings;
        body.interface_declared = true;
        body
    }

    /// Adds inferred quantum and classical ports to a procedurally authored graph.
    ///
    /// # Errors
    ///
    /// Returns a typed interface error when Port directions cannot be inferred.
    ///
    /// # Panics
    ///
    /// Panics only if the generated Port-name sequence is internally exhausted.
    pub fn with_inferred_interface(mut self) -> Result<Self, ModuleError> {
        if self.has_module_structure() {
            return Ok(self);
        }
        let bit_inputs = self
            .action_graph()
            .inputs()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mut port_names = (0..)
            .map(|index| format!("q{index}"))
            .filter(|name| !bit_inputs.contains(name));
        let quantum_ports = self
            .blocks()
            .filter(|block| block.kind() == BlockKind::Port)
            .map(|block| {
                Ok(QuantumPort {
                    name: port_names
                        .next()
                        .expect("finite bit names leave fresh port names"),
                    position: block.pos,
                    direction: port_direction(&self, block.pos).map_err(|message| {
                        ModuleError::InvalidModule {
                            module: self.name.clone(),
                            message,
                            span: None,
                        }
                    })?,
                    resource_type: "data".to_string(),
                })
            })
            .collect::<Result<Vec<_>, ModuleError>>()?;
        self.interface = ModuleInterface {
            quantum_ports,
            bit_inputs,
            bit_outputs: Vec::new(),
        };
        self.interface_declared = true;
        Ok(self)
    }

    /// Constructs a hierarchy rooted at `main`, preserving each definition once.
    ///
    /// # Errors
    ///
    /// Returns a typed declaration, geometry, dependency, or resource-limit error.
    pub fn from_definitions(definitions: Vec<Self>) -> Result<Self, ModuleError> {
        Self::from_definitions_with_limits(definitions, ModuleCertificationLimits::DEFAULT)
    }

    /// Constructs a hierarchy with explicit source and expansion limits.
    ///
    /// # Errors
    ///
    /// Returns a typed declaration, geometry, dependency, or resource-limit error.
    pub fn from_definitions_with_limits(
        mut definitions: Vec<Self>,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, ModuleError> {
        let mut names = HashSet::new();
        for definition in &definitions {
            if !definition.definitions.is_empty() {
                return Err(ModuleError::InvalidModule { module: definition.name.clone(), message: "from_definitions requires local definitions without nested helper libraries; use clone_local_definition on each source.modules() entry".to_string(), span: None });
            }
            if !names.insert(definition.name.as_str()) {
                return Err(ModuleError::DuplicateModule(definition.name.clone()));
            }
        }
        let entry = definitions
            .iter()
            .position(|definition| definition.name == Self::ENTRY_MODULE)
            .ok_or(ModuleError::MissingMain)?;
        let mut root = definitions.remove(entry);
        root.definitions.extend(definitions);
        root.validate_with_limits(limits)?;
        Ok(root)
    }

    /// Borrows every named definition, including this root, without copying topology.
    pub fn modules(&self) -> impl ExactSizeIterator<Item = &Self> + DoubleEndedIterator + Clone {
        (0..self.definitions.len() + 1).map(move |index| {
            if index == self.definitions.len() {
                self
            } else {
                &self.definitions[index]
            }
        })
    }

    /// Mutably borrows one named definition for edits checked by validation.
    pub fn module_mut(&mut self, name: &str) -> Option<&mut Self> {
        if self.name == name {
            Some(self)
        } else {
            self.definitions
                .iter_mut()
                .find(|module| module.name == name)
        }
    }

    /// Returns the executable root; its local topology is stored directly here.
    pub fn root(&self) -> &Self {
        self
    }

    /// Borrows one named definition, including the root.
    pub fn module(&self, name: &str) -> Option<&Self> {
        self.modules().find(|module| module.name == name)
    }

    /// Borrows this definition's local topology and actions.
    pub fn local_body(&self) -> &Self {
        self
    }

    /// Copies one definition without copying the root's helper library.
    pub fn clone_local_definition(&self) -> Self {
        let mut graph = self.copy_local_geometry();
        graph.name = self.name.clone();
        graph.interface = self.interface.clone();
        graph.instances = self.instances.clone();
        graph.quantum_connections = self.quantum_connections.clone();
        graph.bit_bindings = self.bit_bindings.clone();
        graph.interface_declared = self.interface_declared;
        graph
    }

    /// Whether explicitly authored interfaces or child composition are present.
    pub fn has_module_structure(&self) -> bool {
        self.interface_declared
            || self.name != Self::ENTRY_MODULE
            || self.interface != ModuleInterface::default()
            || !self.instances.is_empty()
            || !self.definitions.is_empty()
            || !self.quantum_connections.is_empty()
            || !self.bit_bindings.is_empty()
    }

    /// Explicitly materializes the authored hierarchy as one flat projection.
    ///
    /// # Errors
    ///
    /// Returns an expansion or graph-construction error.
    pub fn flatten(&self) -> Result<Self, ModuleCertificationError> {
        self.flatten_with_limits(ModuleCertificationLimits::DEFAULT)
    }

    /// Materializes one flat projection without resetting explicit source budgets.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-hierarchy, graph, or resource-limit error.
    pub fn flatten_with_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, ModuleCertificationError> {
        self.validate_with_limits(limits)
            .map_err(|error| ModuleCertificationError::Graph {
                module: self.name.clone(),
                source: crate::BlockGraphError::ModuleSource(std::sync::Arc::new(error)),
            })?;
        self.materialize_root_graph()
    }

    /// Materializes the root hierarchy as one transient block graph.
    ///
    /// This is the topology bridge for consumers whose output remains flat.
    /// Definitions are expanded once per instance path, child actions are
    /// namespaced, connected child Ports are removed, and explicit connectors
    /// become ordinary pipes. Logical module certification remains on the
    /// summary path; this method does not derive stabilizers.
    ///
    /// Classical bindings become explicit namespaced `Let` actions. Child-owned
    /// structural branches retain both arms with names qualified by instance
    /// path.
    ///
    /// # Errors
    ///
    /// Returns [`ModuleCertificationError::Graph`] when expansion cannot
    /// construct a valid flat topology.
    #[doc(hidden)]
    pub fn materialize_root_graph(&self) -> Result<BlockGraph, ModuleCertificationError> {
        self.materialize_flat_graph()
            .map(|graph| graph.fix_shadowed_faces())
            .map_err(|source| ModuleCertificationError::Graph {
                module: self.name.clone(),
                source,
            })
    }

    /// Materializes a module hierarchy for consumers that still need one flat graph.
    #[doc(hidden)]
    pub fn materialize_flat_graph(&self) -> Result<BlockGraph, crate::BlockGraphError> {
        if !self.has_module_structure() {
            return Ok(self.copy_local_geometry());
        }
        self.validate_definition_structure()
            .map_err(|error| crate::BlockGraphError::ModuleSource(std::sync::Arc::new(error)))?;
        flatten_module_definition(self, self.root(), "").map(|module| module.graph)
    }

    /// Checks expanded definition sizes without materializing or revalidating geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when an expanded resource count exceeds `limits`.
    pub(crate) fn validate_hierarchy_resource_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), ModuleError> {
        self.validate_definition_structure()?;
        crate::summary::preflight_program_expansion(self, limits)
    }

    fn validate_definition_structure(&self) -> Result<HashMap<&str, &Self>, ModuleError> {
        let mut definitions = HashMap::new();
        for module in self.modules() {
            if !module.definitions.is_empty() && !std::ptr::eq(module, self) {
                return Err(ModuleError::InvalidModule {
                    module: module.name.clone(),
                    message: "nested helper libraries must be resolved into the executable root"
                        .to_string(),
                    span: None,
                });
            }
            if definitions.insert(module.name.as_str(), module).is_some() {
                return Err(ModuleError::DuplicateModule(module.name.clone()));
            }
        }
        if self.name != Self::ENTRY_MODULE {
            return Err(ModuleError::MissingMain);
        }
        for module in self.modules() {
            validate_module(module, &definitions)?;
        }
        validate_acyclic(&self.modules().collect::<Vec<_>>(), &definitions)?;
        Ok(definitions)
    }

    /// Validates all authored definitions, interfaces, seams, and dependency cycles.
    ///
    /// # Errors
    ///
    /// Returns typed source, module, geometry, or resource-limit errors.
    pub fn validate_with_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), ModuleError> {
        if !self.has_module_structure() {
            self.validate_local_resource_limits(limits)
                .and_then(|()| self.validate_source())
                .map_err(|source| ModuleError::InvalidGeometry {
                    module: self.name.clone(),
                    source: ModuleCertificationError::Graph {
                        module: self.name.clone(),
                        source,
                    },
                    span: None,
                })?;
            return Ok(());
        }
        let definitions = self.validate_definition_structure()?;
        crate::summary::preflight_program_expansion(self, limits)?;
        validate_classical_cycles(&self.modules().collect::<Vec<_>>(), &definitions)?;
        crate::summary::validate_program_geometry(self)
    }

    /// Extracts an independently compilable graph rooted at one named definition.
    ///
    /// Includes its transitive helper definitions and renames the selected root
    /// to `main`, resolving any collision with a reachable original root.
    ///
    /// # Errors
    ///
    /// Returns a typed missing-definition, invalid-hierarchy, or resource error.
    pub fn extract_definition(&self, name: &str) -> Result<Self, ModuleError> {
        self.extract_definition_with_limits(name, ModuleCertificationLimits::DEFAULT)
    }

    /// Extracts one definition and its dependencies with explicit source budgets.
    ///
    /// # Errors
    ///
    /// Returns a typed missing-definition, invalid-hierarchy, or resource error.
    ///
    /// # Panics
    ///
    /// Panics if validated dependency bookkeeping is internally inconsistent.
    pub fn extract_definition_with_limits(
        &self,
        name: &str,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, ModuleError> {
        self.validate_with_limits(limits)?;
        if self.module(name).is_none() {
            return Err(ModuleError::InvalidModule {
                module: name.to_string(),
                message: "unknown graph definition".to_string(),
                span: None,
            });
        }
        if !self.has_module_structure() {
            return self.clone_local_definition().with_inferred_interface();
        }
        let mut reachable = BTreeSet::new();
        let mut pending = vec![name];
        while let Some(definition) = pending.pop() {
            if reachable.insert(definition) {
                pending.extend(
                    self.module(definition)
                        .expect("checked definition exists")
                        .instances
                        .iter()
                        .map(|instance| instance.definition.as_str()),
                );
            }
        }
        let mut names = reachable
            .iter()
            .map(|name| ((*name).to_string(), (*name).to_string()))
            .collect::<HashMap<_, _>>();
        if name != Self::ENTRY_MODULE && reachable.contains(Self::ENTRY_MODULE) {
            let mut fresh = "source_main".to_string();
            let mut index = 0;
            while reachable.contains(fresh.as_str()) {
                index += 1;
                fresh = format!("source_main_{index}");
            }
            names.insert(Self::ENTRY_MODULE.to_string(), fresh);
        }
        names.insert(name.to_string(), Self::ENTRY_MODULE.to_string());
        let definitions = self
            .modules()
            .filter(|module| reachable.contains(module.name.as_str()))
            .map(|module| {
                let mut graph = module.clone_local_definition();
                graph.name = names[&module.name].clone();
                for instance in &mut graph.instances {
                    instance.definition = names[&instance.definition].clone();
                }
                graph
            })
            .collect();
        Self::from_definitions_with_limits(definitions, limits)
    }

    /// Certifies one definition without flattening its ZX rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the definition is missing or cannot be certified within `limits`.
    pub fn certify_leaf(
        &self,
        name: &str,
        limits: ModuleCertificationLimits,
    ) -> Result<LeafModuleCertificate, ModuleCertificationError> {
        if !self.has_module_structure() {
            let inferred = self
                .clone_local_definition()
                .with_inferred_interface()
                .map_err(|error| ModuleCertificationError::Graph {
                    module: self.name.clone(),
                    source: crate::BlockGraphError::ModuleSource(std::sync::Arc::new(error)),
                })?;
            return inferred.certify_leaf(name, limits);
        }
        self.validate_with_limits(limits)
            .map_err(|error| ModuleCertificationError::Graph {
                module: self.name.clone(),
                source: crate::BlockGraphError::ModuleSource(std::sync::Arc::new(error)),
            })?;

        let module = self
            .module(name)
            .ok_or_else(|| ModuleCertificationError::UnknownModule(name.to_string()))?;
        if !module.instances.is_empty() {
            return Err(ModuleCertificationError::NotLeaf(name.to_string()));
        }

        certify_leaf_body(
            name,
            module.local_body(),
            &module.interface.quantum_ports,
            limits,
        )
    }
}

pub(crate) fn certify_leaf_body(
    name: &str,
    source: &BlockGraph,
    ports: &[QuantumPort],
    limits: ModuleCertificationLimits,
) -> Result<LeafModuleCertificate, ModuleCertificationError> {
    let graph_error = |source| ModuleCertificationError::Graph {
        module: name.to_string(),
        source,
    };
    let body = source.fix_shadowed_faces();
    let zx = ZXGraph::from_block_graph_for_analysis(&body)
        .map_err(|source| graph_error(source.into()))?;
    if zx.total_ids() > limits.max_local_columns {
        return Err(resource_limit(
            name,
            "local ZX columns",
            zx.total_ids(),
            limits.max_local_columns,
        ));
    }

    let columns = ports
        .iter()
        .map(|port| {
            zx.node_at(port.position)
                .expect("validated module port is a ZX node")
                .id
        })
        .collect::<Vec<_>>();
    let table = zx
        .projected_external_table(
            &columns,
            ProjectionLimits {
                max_frontier_width: limits.max_frontier_width,
                max_witness_nodes: limits.max_witness_nodes,
            },
        )
        .map_err(|error| projection_limit(name, error))?;
    let external_basis = table.flow_witness_basis().to_external_basis();
    let stabilizers = zx
        .stabilizers_from_external_basis_with_limits(&external_basis, limits)
        .map_err(|source| match source {
            StabilizerError::ResourceLimited {
                phase,
                observed,
                limit,
            } => resource_limit(name, phase, observed, limit),
            source => graph_error(source.into()),
        })?;
    body.with_analyzed_action_graph(&stabilizers)
        .map_err(graph_error)?;
    stabilizers
        .validate_measurements_close_before_outputs_with_limits(limits)
        .map_err(|source| match source {
            RuntimeBasisError::Stabilizer(StabilizerError::ResourceLimited {
                phase,
                observed,
                limit,
            }) => resource_limit(name, phase, observed, limit),
            source => ModuleCertificationError::Runtime {
                module: name.to_string(),
                source,
            },
        })?;
    Ok(LeafModuleCertificate {
        module: name.to_string(),
        table,
        stabilizers,
    })
}

impl BlockGraph {
    /// Canonical BLOG 1.0 text. Resolved imports are written inline.
    pub fn to_blog_text(&self) -> String {
        if !self.has_module_structure() {
            return self
                .to_program_blog_text()
                .unwrap_or_else(|_| self.to_blog_body_text());
        }
        let mut out = String::from("BLOG 1.0\n\n");
        for module in self.modules() {
            write_definition(&mut out, module);
        }
        out
    }

    /// Collision-free implementation keys for independently cacheable
    /// definitions. Each key includes every transitive dependency once in
    /// canonical order, so changing a child invalidates its parents without
    /// expanding shared definitions into an exponential tree of strings.
    #[doc(hidden)]
    pub fn module_definition_cache_keys(&self) -> HashMap<String, String> {
        fn collect<'a>(program: &'a BlockGraph, name: &'a str, names: &mut BTreeSet<&'a str>) {
            let mut pending = vec![name];
            while let Some(name) = pending.pop() {
                if names.insert(name) {
                    let module = program
                        .module(name)
                        .expect("validated instance names a definition");
                    pending.extend(
                        module
                            .instances
                            .iter()
                            .map(|instance| instance.definition.as_str()),
                    );
                }
            }
        }

        let mut reachable = BTreeSet::new();
        collect(self, Self::ENTRY_MODULE, &mut reachable);
        reachable
            .into_iter()
            .map(|name| {
                let mut definitions = BTreeSet::new();
                collect(self, name, &mut definitions);
                let mut key = format!("root {name}\n");
                for definition in definitions {
                    write_definition(
                        &mut key,
                        self.module(definition)
                            .expect("reachable definition exists"),
                    );
                }
                (name.to_string(), key)
            })
            .collect()
    }

    /// Global orientations in which each reachable definition is instantiated.
    #[doc(hidden)]
    pub fn definition_orientations(&self) -> HashMap<String, Vec<ModuleOrientation>> {
        let mut seen = HashSet::new();
        let mut orientations = HashMap::<String, Vec<ModuleOrientation>>::new();
        let mut pending = vec![(Self::ENTRY_MODULE, ModuleOrientation::IDENTITY)];
        while let Some((name, orientation)) = pending.pop() {
            if seen.insert((name, orientation)) {
                orientations
                    .entry(name.to_string())
                    .or_default()
                    .push(orientation);
                let module = self
                    .module(name)
                    .expect("validated instance names a definition");
                pending.extend(module.instances.iter().rev().map(|instance| {
                    (
                        instance.definition.as_str(),
                        orientation.then(instance.rotation),
                    )
                }));
            }
        }
        for values in orientations.values_mut() {
            values.sort_unstable_by_key(|orientation| orientation.sort_key());
        }
        orientations
    }

    /// Definition-local block faces whose Hadamard bit is supplied by a
    /// consumer-side direct module seam. Physical objects carry both variants
    /// for these faces and let the linker select one.
    #[doc(hidden)]
    pub fn module_public_seam_faces(&self) -> HashMap<(String, IVec3), Vec<Direction>> {
        let mut faces = HashMap::<(String, IVec3), Vec<Direction>>::new();
        for module in self.modules() {
            let body = module.copy_local_geometry().fix_shadowed_faces();
            for port in &module.interface.quantum_ports {
                let neighbors = body.neighbor_positions(port.position);
                let [neighbor] = neighbors.as_slice() else {
                    // A composite may forward this public Port directly into a
                    // child. The descendant definition owns the mutable face.
                    continue;
                };
                let block = body
                    .get_endpoint_block(*neighbor)
                    .expect("validated module Port neighbour belongs to a block");
                let direction = Direction::try_from(port.position - *neighbor)
                    .expect("validated module Port pipe is axial and unit length");
                faces
                    .entry((module.name.clone(), block.pos()))
                    .or_default()
                    .push(direction);
            }
        }
        for directions in faces.values_mut() {
            directions.sort_unstable_by_key(|direction| direction.index());
            directions.dedup();
        }
        faces
    }
}

fn write_definition(out: &mut String, module: &BlockGraph) {
    let ids = module.local_body().blog_block_ids().into_iter().collect();
    writeln!(out, "module {} {{", module.name).expect("writing to a String cannot fail");
    write_interface(out, module, &ids);
    write_instances(out, module);
    write_body(out, module.local_body());
    write_connections(out, module, &ids);
    out.push_str("}\n\n");
}

/// One linked definition instance before its parent consumes public ports.
/// Physical object linkers use this record to compose a definition DAG without
/// invoking whole-program materialization.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct LinkedModuleDefinition {
    pub graph: BlockGraph,
    /// Fully qualified module scope for each linked action, in action order.
    pub action_scopes: Vec<String>,
    pub bit_outputs: HashMap<String, Expr>,
    pub sites: HashMap<IVec3, MaterializedModuleSite>,
    pub quantum_ports: Vec<QuantumPort>,
}

/// Links `module` and every descendant instance into one flat
/// graph, qualifying names under `prefix`.
#[doc(hidden)]
pub fn flatten_module_definition(
    program: &BlockGraph,
    module: &BlockGraph,
    prefix: &str,
) -> Result<LinkedModuleDefinition, crate::BlockGraphError> {
    struct Frame<'a> {
        module: &'a BlockGraph,
        prefix: String,
        children: Vec<LinkedModuleDefinition>,
    }
    let definitions = program
        .modules()
        .map(|module| (module.name.as_str(), module))
        .collect::<HashMap<_, _>>();
    let mut pending = vec![Frame {
        module,
        prefix: prefix.to_string(),
        children: Vec::new(),
    }];
    loop {
        let frame = pending.last_mut().expect("root remains until returned");
        if let Some(instance) = frame.module.instances.get(frame.children.len()) {
            let prefix = qualified_name(&frame.prefix, &instance.name);
            pending.push(Frame {
                module: definitions[instance.definition.as_str()],
                prefix,
                children: Vec::new(),
            });
            continue;
        }
        let frame = pending.pop().expect("finished frame exists");
        let mut children = frame.children.into_iter();
        let linked = link_module_definition_with(frame.module, &frame.prefix, |_, _| {
            Ok(children
                .next()
                .expect("each child was linked in instance order"))
        })?;
        match pending.last_mut() {
            Some(parent) => parent.children.push(linked),
            None => return Ok(linked),
        }
    }
}

/// A transient linker index for the regions whose arm block or cut a new pipe
/// can update. Ordinary graph edits keep the full branch scan.
#[derive(Default)]
struct LinkBranchPipeIndex {
    by_shown_block: HashMap<IVec3, Vec<usize>>,
    by_cut: HashMap<(IVec3, IVec3), Vec<usize>>,
}

impl LinkBranchPipeIndex {
    fn add_region(&mut self, index: usize, region: &crate::BranchRegion) {
        for block in region.shown_arm().blocks() {
            self.by_shown_block
                .entry(block.pos())
                .or_default()
                .push(index);
        }
        // A new cut made during linking touches an indexed shown-arm block;
        // arm membership does not change here. Existing cut keys still need
        // indexing for Hadamard repair even without a shown-arm endpoint.
        for cut in region
            .arm_incoming(false)
            .iter()
            .chain(region.arm_incoming(true))
        {
            self.by_cut
                .entry(crate::branch::pipe_key(&cut.pipe))
                .or_default()
                .push(index);
        }
    }

    fn candidates(&self, graph: &BlockGraph, pipe: &Pipe) -> Option<Vec<usize>> {
        let (src, dst) = pipe.try_endpoints().ok()?;
        let src_owner = graph.get_endpoint_block(src)?.pos();
        let dst_owner = graph.get_endpoint_block(dst)?.pos();
        let mut candidates = Vec::new();
        for owner in [src_owner, dst_owner] {
            if let Some(regions) = self.by_shown_block.get(&owner) {
                candidates.extend(regions);
            }
        }
        if let Some(regions) = self.by_cut.get(&crate::branch::pipe_key(pipe)) {
            candidates.extend(regions);
        }
        candidates.sort_unstable();
        candidates.dedup();
        Some(candidates)
    }
}

fn add_link_pipe(
    graph: &mut BlockGraph,
    index: &LinkBranchPipeIndex,
    pipe: Pipe,
) -> Result<(), crate::BlockGraphError> {
    if let Some(candidates) = index.candidates(graph, &pipe) {
        graph.try_add_pipe_with_branch_candidates(pipe, Some(&candidates))
    } else {
        // Keep the ordinary typed error for an invalid or missing endpoint.
        graph.try_add_pipe(pipe)
    }
}

/// Deterministically link one definition from resolved child object records.
/// No stabilizer solve, signature discovery, or template compilation occurs.
fn link_module_definition_with(
    module: &BlockGraph,
    prefix: &str,
    mut link_child: impl FnMut(
        &ModuleInstance,
        &str,
    ) -> Result<LinkedModuleDefinition, crate::BlockGraphError>,
) -> Result<LinkedModuleDefinition, crate::BlockGraphError> {
    let mut graph = module.copy_local_geometry();
    let mut sites = module_sites(module, prefix);
    for branch in &mut graph.branches {
        branch.name = qualified_name(prefix, &branch.name);
    }
    let mut branch_pipe_index = LinkBranchPipeIndex::default();
    let mut branch_occupancy = LinkBranchOccupancy::default();
    for (index, branch) in graph.branches.iter().enumerate() {
        branch_pipe_index.add_region(index, branch);
        branch_occupancy.add_region(branch)?;
    }
    let local_actions = graph.actions();
    graph.clear_actions();
    let mut exposed = HashMap::<InstancePort, IVec3>::new();
    let mut child_actions = Vec::new();
    let mut child_outputs = HashMap::<(String, String), Expr>::new();
    let mut connections_by_port = HashMap::<&InstancePort, &QuantumConnection>::new();
    for connection in &module.quantum_connections {
        match connection {
            QuantumConnection::Input { input, .. } => {
                connections_by_port.entry(input).or_insert(connection);
            }
            QuantumConnection::Output { output, .. } => {
                connections_by_port.entry(output).or_insert(connection);
            }
            QuantumConnection::Pipe { output, input, .. } => {
                connections_by_port.entry(output).or_insert(connection);
                connections_by_port.entry(input).or_insert(connection);
            }
        }
    }

    for instance in &module.instances {
        let child_prefix = qualified_name(prefix, &instance.name);
        let flattened = link_child(instance, &child_prefix)?;
        let mut quantum_ports = flattened.quantum_ports.clone();
        for port in &mut quantum_ports {
            port.position = instance.try_transform_position(port.position)?;
        }
        for (name, expr) in flattened.bit_outputs {
            child_outputs.insert((instance.name.clone(), name), expr);
        }
        let orientation = instance.orientation();
        let mut child = flattened
            .graph
            .with_orientation_lenient(orientation)?
            .shift_positions(instance.translation)?;
        let mut child_sites = flattened
            .sites
            .into_iter()
            .map(|(position, mut site)| {
                site.orientation = orientation.then_orientation(site.orientation);
                instance
                    .try_transform_position(position)
                    .map(|position| (position, site))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let mut actions = child.actions();
        child.clear_actions();
        let mut bound_pipes = Vec::new();

        let child_ports = quantum_ports
            .iter()
            .map(|port| {
                let endpoint = InstancePort {
                    instance: instance.name.clone(),
                    port: port.name.clone(),
                };
                let position = port.position;
                let neighbors = child.neighbor_positions(position);
                let [neighbor] = neighbors.as_slice() else {
                    unreachable!("validated module port has one incident pipe")
                };
                let pipe = child
                    .get_pipe(position, *neighbor)
                    .expect("module port neighbor has a pipe")
                    .clone();
                (endpoint, position, *neighbor, pipe)
            })
            .collect::<Vec<_>>();

        // Consumed Ports are replaced by parent bindings below. Their pipes
        // may be branch cuts, so this temporary removal must retain the arms.
        let child_branches = std::mem::take(&mut child.branches);
        for (endpoint, position, neighbor, pipe) in child_ports {
            // A consumed Port has one wire, selected by either target syntax.
            // Move its correction to the retained endpoint in the source frame,
            // before the parent binding can add another Hadamard.
            for action in &mut actions {
                if let Action::Feedback { targets, .. } = action {
                    for target in targets
                        .iter_mut()
                        .filter(|target| target.target == position)
                    {
                        target.target = neighbor;
                        target.direction = Some(if pipe.src == neighbor {
                            pipe.dir()
                        } else {
                            pipe.dir().negate()
                        });
                        if pipe.hadamard {
                            target.pauli = match target.pauli {
                                crate::PauliBasis::X => crate::PauliBasis::Z,
                                crate::PauliBasis::Z => crate::PauliBasis::X,
                                crate::PauliBasis::Y => crate::PauliBasis::Y,
                            };
                        }
                    }
                }
            }
            let connection = connections_by_port
                .get(&endpoint)
                .copied()
                .expect("validated child port is connected");
            child.remove_block(position);
            child_sites.remove(&position);
            match connection {
                QuantumConnection::Pipe { .. } => {
                    exposed.insert(endpoint, neighbor);
                }
                QuantumConnection::Input { hadamard, .. }
                | QuantumConnection::Output { hadamard, .. } => {
                    // A bound parent block occupies the removed Port position.
                    let mut pipe = pipe;
                    pipe.hadamard ^= *hadamard;
                    bound_pipes.push(pipe);
                }
            }
        }
        child_actions.push((instance.name.clone(), actions, flattened.action_scopes));
        child.branches = child_branches;

        for block in child.blocks() {
            graph.try_add_block_with_branch_occupancy(block.clone(), Some(&branch_occupancy))?;
        }
        for pipe in child.pipes() {
            add_link_pipe(&mut graph, &branch_pipe_index, pipe.clone())?;
        }
        sites.extend(child_sites);
        let first_branch = graph.branches.len();
        graph.branches.extend(child.branches.iter().cloned());
        for (index, branch) in graph.branches.iter().enumerate().skip(first_branch) {
            branch_pipe_index.add_region(index, branch);
            branch_occupancy.add_region(branch)?;
        }
        for pipe in bound_pipes {
            add_link_pipe(&mut graph, &branch_pipe_index, pipe)?;
        }
    }

    for connection in &module.quantum_connections {
        let QuantumConnection::Pipe {
            output,
            input,
            hadamard,
        } = connection
        else {
            continue;
        };
        let output = exposed[output];
        let input = exposed[input];
        let direction = Direction::iter()
            .find(|direction| {
                checked_add_position(output, direction.to_ivec3()).ok() == Some(input)
            })
            .expect("validated module seam endpoints are adjacent");
        let pipe = Pipe::new(output, direction);
        add_link_pipe(
            &mut graph,
            &branch_pipe_index,
            if *hadamard {
                pipe.with_hadamard()
            } else {
                pipe
            },
        )?;
    }

    let bindings = module
        .bit_bindings
        .iter()
        .map(|binding| {
            let child_prefix = qualified_name(prefix, &binding.target_instance);
            (
                binding.target_instance.clone(),
                Action::Let {
                    name: qualified_name(&child_prefix, &binding.target_bit),
                    expr: resolve_bit_source(&binding.source, module, prefix, &child_outputs),
                },
            )
        })
        .collect::<Vec<_>>();
    let (actions, action_scopes) = order_linked_actions(
        bindings,
        child_actions,
        rename_module_actions(local_actions, prefix, &child_outputs),
        prefix,
    )?;
    graph.set_actions_deferred_with_inputs(
        actions,
        module
            .interface
            .bit_inputs
            .iter()
            .map(|name| qualified_name(prefix, name)),
    )?;
    let bit_outputs = module
        .interface
        .bit_outputs
        .iter()
        .map(|output| {
            (
                output.name.clone(),
                resolve_module_expr(output.expr.clone(), prefix, &child_outputs),
            )
        })
        .collect();
    Ok(LinkedModuleDefinition {
        graph,
        action_scopes,
        bit_outputs,
        sites,
        quantum_ports: module.interface.quantum_ports.clone(),
    })
}

fn module_sites(module: &BlockGraph, prefix: &str) -> HashMap<IVec3, MaterializedModuleSite> {
    let positions = module
        .local_body()
        .blocks()
        .map(Block::pos)
        .chain(
            module
                .local_body()
                .branch_definitions()
                .iter()
                .flat_map(|region| {
                    region
                        .on_false()
                        .blocks()
                        .chain(region.on_true().blocks())
                        .map(Block::pos)
                }),
        )
        .collect::<HashSet<_>>();
    positions
        .into_iter()
        .map(|local_position| {
            (
                local_position,
                MaterializedModuleSite {
                    definition: module.name.clone(),
                    instance_path: prefix.to_string(),
                    local_position,
                    orientation: ModuleOrientation::IDENTITY,
                },
            )
        })
        .collect()
}

fn rename_module_actions(
    actions: Vec<Action>,
    prefix: &str,
    child_outputs: &HashMap<(String, String), Expr>,
) -> Vec<Action> {
    let rename = |name: String| qualified_name(prefix, &name);
    actions
        .into_iter()
        .map(|action| match action {
            Action::Let { name, expr } => Action::Let {
                name: rename(name),
                expr: resolve_module_expr(expr, prefix, child_outputs),
            },
            Action::Measure { target, name } => Action::Measure {
                target,
                name: rename(name),
            },
            Action::DiscardIf(expr) => {
                Action::DiscardIf(resolve_module_expr(expr, prefix, child_outputs))
            }
            Action::Resolve { target, condition } => Action::Resolve {
                target,
                condition: resolve_module_expr(condition, prefix, child_outputs),
            },
            Action::Branch { target, condition } => Action::Branch {
                target,
                condition: resolve_module_expr(condition, prefix, child_outputs),
            },
            Action::Feedback { targets, condition } => Action::Feedback {
                targets,
                condition: condition.map(|expr| resolve_module_expr(expr, prefix, child_outputs)),
            },
        })
        .collect()
}

fn order_linked_actions(
    bindings: Vec<(String, Action)>,
    child_actions: Vec<(String, Vec<Action>, Vec<String>)>,
    local_actions: Vec<Action>,
    prefix: &str,
) -> Result<(Vec<Action>, Vec<String>), crate::BlockGraphError> {
    let mut bindings_by_target = HashMap::<String, Vec<Action>>::new();
    for (target, binding) in bindings {
        bindings_by_target.entry(target).or_default().push(binding);
    }

    let mut actions = Vec::new();
    let mut action_scopes = Vec::new();
    let mut ordered_ranges = Vec::new();
    for (instance, child, child_scopes) in child_actions {
        let bindings = bindings_by_target.remove(&instance).unwrap_or_default();
        action_scopes.extend(std::iter::repeat_n(
            qualified_name(prefix, &instance),
            bindings.len(),
        ));
        actions.extend(bindings);
        let start = actions.len();
        debug_assert_eq!(child.len(), child_scopes.len());
        actions.extend(child);
        action_scopes.extend(child_scopes);
        ordered_ranges.push(start..actions.len());
    }
    debug_assert!(bindings_by_target.is_empty());
    let start = actions.len();
    action_scopes.extend(std::iter::repeat_n(prefix.to_string(), local_actions.len()));
    actions.extend(local_actions);
    ordered_ranges.push(start..actions.len());

    let mut successors = vec![BTreeSet::new(); actions.len()];
    for (predecessor, consumer, _) in crate::ActionDag::from_actions(&actions).dependencies() {
        successors[predecessor].insert(consumer);
    }
    for range in ordered_ranges {
        for (predecessor, consumers) in successors
            .iter_mut()
            .enumerate()
            .take(range.end.saturating_sub(1))
            .skip(range.start)
        {
            consumers.insert(predecessor + 1);
        }
    }
    let mut incoming = vec![0usize; actions.len()];
    for consumer in successors.iter().flatten() {
        incoming[*consumer] += 1;
    }
    let mut ready = incoming
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| (count == 0).then_some(index))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(actions.len());
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for &consumer in &successors[index] {
            incoming[consumer] -= 1;
            if incoming[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }
    if order.len() != actions.len() {
        let ordinal = incoming
            .iter()
            .position(|&count| count != 0)
            .expect("unsorted actions retain an incoming dependency");
        return Err(crate::InvalidActionError::DependencyCycle { ordinal }.into());
    }
    Ok(order
        .into_iter()
        .map(|index| (actions[index].clone(), action_scopes[index].clone()))
        .unzip())
}

fn resolve_module_expr(
    expr: Expr,
    prefix: &str,
    child_outputs: &HashMap<(String, String), Expr>,
) -> Expr {
    match expr {
        Expr::Var(name) => split_member(&name)
            .and_then(|(instance, bit)| {
                child_outputs
                    .get(&(instance.to_string(), bit.to_string()))
                    .cloned()
            })
            .unwrap_or_else(|| Expr::Var(qualified_name(prefix, &name))),
        Expr::Not(expr) => Expr::Not(Box::new(resolve_module_expr(*expr, prefix, child_outputs))),
        Expr::Binary(op, lhs, rhs) => Expr::Binary(
            op,
            Box::new(resolve_module_expr(*lhs, prefix, child_outputs)),
            Box::new(resolve_module_expr(*rhs, prefix, child_outputs)),
        ),
    }
}

fn resolve_bit_source(
    source: &BitRef,
    module: &BlockGraph,
    prefix: &str,
    child_outputs: &HashMap<(String, String), Expr>,
) -> Expr {
    if let Some(instance) = &source.instance {
        return child_outputs[&(instance.clone(), source.bit.clone())].clone();
    }
    if module.interface.bit_inputs.contains(&source.bit) {
        return Expr::Var(qualified_name(prefix, &source.bit));
    }
    let output = module
        .interface
        .bit_outputs
        .iter()
        .find(|output| output.name == source.bit)
        .expect("validated bit source names a local export");
    resolve_module_expr(output.expr.clone(), prefix, child_outputs)
}

/// Joins a module instance prefix and a local name with the `__` separator
/// used throughout module linking. An empty prefix returns `name` unchanged.
pub fn qualified_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}

pub(crate) fn resource_limit(
    module: &str,
    phase: &'static str,
    observed: usize,
    limit: usize,
) -> ModuleCertificationError {
    ModuleCertificationError::ResourceLimited {
        module: module.to_string(),
        phase,
        observed,
        limit,
    }
}

pub(crate) fn projection_limit(module: &str, error: ProjectionError) -> ModuleCertificationError {
    match error {
        ProjectionError::FrontierLimit { observed, limit } => {
            resource_limit(module, "frontier columns", observed, limit)
        }
        ProjectionError::WitnessLimit { observed, limit } => {
            resource_limit(module, "witness nodes", observed, limit)
        }
    }
}

fn validate_module(
    module: &BlockGraph,
    definitions: &HashMap<&str, &BlockGraph>,
) -> Result<(), ModuleError> {
    let fail = |message| ModuleError::InvalidModule {
        module: module.name.clone(),
        message,
        span: None,
    };
    if !crate::parser::is_valid_simple_identifier(&module.name) {
        return Err(fail("invalid module name".to_string()));
    }
    if let Some(branch) = module
        .local_body()
        .branch_definitions()
        .iter()
        .find(|branch| branch.name.contains("__"))
    {
        return Err(fail(format!("invalid branch name '{}'", branch.name)));
    }

    let mut names = HashSet::new();
    let mut port_positions = HashSet::new();
    for port in &module.interface.quantum_ports {
        if !crate::parser::is_valid_simple_identifier(&port.name) {
            return Err(fail(format!("invalid port name '{}'", port.name)));
        }
        if !names.insert(port.name.as_str()) {
            return Err(fail(format!("duplicate interface name '{}'", port.name)));
        }
        if !crate::parser::is_valid_resource_type(&port.resource_type) {
            return Err(fail(format!(
                "invalid resource type '{}'",
                port.resource_type
            )));
        }
        if !port_positions.insert(port.position) {
            return Err(fail(format!(
                "multiple interface ports use {}",
                port.position
            )));
        }
        let actual = port_direction(module.local_body(), port.position)
            .map_err(|message| fail(format!("port '{}': {message}", port.name)))?;
        if actual != port.direction {
            return Err(fail(format!(
                "port '{}' is {:?}, declared {:?}",
                port.name, actual, port.direction
            )));
        }
    }
    for name in &module.interface.bit_inputs {
        if !crate::parser::is_valid_simple_identifier(name) || name.contains("__") {
            return Err(fail(format!("invalid in name '{name}'")));
        }
        if !names.insert(name) {
            return Err(fail(format!("duplicate interface name '{name}'")));
        }
    }
    for output in &module.interface.bit_outputs {
        if !crate::parser::is_valid_simple_identifier(&output.name) || output.name.contains("__") {
            return Err(fail(format!("invalid bit output name '{}'", output.name)));
        }
        if !names.insert(&output.name) {
            return Err(fail(format!("duplicate interface name '{}'", output.name)));
        }
    }
    for block in module
        .local_body()
        .blocks()
        .filter(|block| block.kind().is_port())
    {
        if !port_positions.contains(&block.pos()) {
            return Err(fail(format!("undeclared Port at {}", block.pos())));
        }
    }

    let mut instances = HashMap::new();
    for instance in &module.instances {
        if !crate::parser::is_valid_simple_identifier(&instance.name)
            || instance.name.contains("__")
        {
            return Err(fail(format!("invalid instance name '{}'", instance.name)));
        }
        if instances.insert(instance.name.as_str(), instance).is_some() {
            return Err(fail(format!("duplicate instance '{}'", instance.name)));
        }
        if !definitions.contains_key(instance.definition.as_str()) {
            return Err(fail(format!(
                "instance '{}' uses unknown module '{}'",
                instance.name, instance.definition
            )));
        }
    }

    validate_quantum_connections(module, definitions, &instances, &fail)?;
    validate_bits(module, definitions, &instances, &fail)
}

fn validate_quantum_connections(
    module: &BlockGraph,
    definitions: &HashMap<&str, &BlockGraph>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    // Index only definitions used by this module. Repeated instance ports
    // then require one lookup rather than another scan of the child interface.
    let mut ports = HashMap::new();
    let mut indexed = HashSet::new();
    for instance in instances.values() {
        let definition = instance.definition.as_str();
        if indexed.insert(definition) {
            for port in &definitions[definition].interface.quantum_ports {
                ports
                    .entry((definition, port.name.as_str(), port.direction))
                    .or_insert(port);
            }
        }
    }
    let mut used = HashSet::<(&str, &str)>::new();
    let mut block_types = HashMap::<IVec3, &str>::new();
    let common_blocks = module
        .local_body()
        .blog_block_ids()
        .into_iter()
        .map(|(position, _)| position)
        .collect::<HashSet<_>>();
    for connection in &module.quantum_connections {
        let bind = match connection {
            QuantumConnection::Input { block, input, .. } => {
                Some((*block, input, PortDirection::Input))
            }
            QuantumConnection::Output { output, block, .. } => {
                Some((*block, output, PortDirection::Output))
            }
            QuantumConnection::Pipe { output, input, .. } => {
                let output_port =
                    child_port(output, PortDirection::Output, &ports, instances, fail)?;
                let input_port = child_port(input, PortDirection::Input, &ports, instances, fail)?;
                if output_port.resource_type != input_port.resource_type {
                    return Err(fail(format!(
                        "pipe joins resource types '{}' and '{}'",
                        output_port.resource_type, input_port.resource_type
                    )));
                }
                use_port(output, &mut used, fail)?;
                use_port(input, &mut used, fail)?;
                None
            }
        };
        let Some((block, endpoint, direction)) = bind else {
            continue;
        };
        let port = child_port(endpoint, direction, &ports, instances, fail)?;
        validate_bind(
            module,
            block,
            endpoint,
            port,
            instances,
            &common_blocks,
            fail,
        )?;
        use_port(endpoint, &mut used, fail)?;
        if let Some(resource_type) = block_types.insert(block, &port.resource_type)
            && resource_type != port.resource_type
        {
            return Err(fail(format!(
                "block {block} joins resource types '{resource_type}' and '{}'",
                port.resource_type
            )));
        }
    }

    for instance in &module.instances {
        let child = definitions[instance.definition.as_str()];
        for port in &child.interface.quantum_ports {
            if !used.contains(&(instance.name.as_str(), port.name.as_str())) {
                return Err(fail(format!(
                    "unconnected port '{}.{}'",
                    instance.name, port.name
                )));
            }
        }
    }
    Ok(())
}

fn child_port<'a>(
    endpoint: &InstancePort,
    direction: PortDirection,
    ports: &HashMap<(&str, &str, PortDirection), &'a QuantumPort>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<&'a QuantumPort, ModuleError> {
    let instance = instances
        .get(endpoint.instance.as_str())
        .ok_or_else(|| fail(format!("unknown instance '{}'", endpoint.instance)))?;
    ports
        .get(&(
            instance.definition.as_str(),
            endpoint.port.as_str(),
            direction,
        ))
        .copied()
        .ok_or_else(|| {
            fail(format!(
                "unknown {:?} port '{}.{}'",
                direction, endpoint.instance, endpoint.port
            ))
        })
}

fn validate_bind(
    module: &BlockGraph,
    block: IVec3,
    endpoint: &InstancePort,
    port: &QuantumPort,
    instances: &HashMap<&str, &ModuleInstance>,
    common_blocks: &HashSet<IVec3>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    if !common_blocks.contains(&block) {
        return Err(fail(format!(
            "bind references missing or branch-local block {block}"
        )));
    }
    let instance = instances[endpoint.instance.as_str()];
    let translated = instance
        .try_transform_position(port.position)
        .map_err(|error| fail(error.to_string()))?;
    if translated != block {
        return Err(fail(format!(
            "port '{}.{}' is at {translated}, not {block}",
            endpoint.instance, endpoint.port
        )));
    }
    if let Some(parent_port) = module
        .interface
        .quantum_ports
        .iter()
        .find(|candidate| candidate.position == block)
        && (parent_port.direction != port.direction
            || parent_port.resource_type != port.resource_type)
    {
        return Err(fail(format!(
            "bound parent port '{}' has a different interface",
            parent_port.name
        )));
    }
    Ok(())
}

fn use_port<'a>(
    port: &'a InstancePort,
    used: &mut HashSet<(&'a str, &'a str)>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    if used.insert((port.instance.as_str(), port.port.as_str())) {
        Ok(())
    } else {
        Err(fail(format!(
            "port '{}.{}' is connected more than once",
            port.instance, port.port
        )))
    }
}

fn validate_bits(
    module: &BlockGraph,
    definitions: &HashMap<&str, &BlockGraph>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    let local_actions = module
        .local_body()
        .action_graph()
        .ordered_nodes()
        .filter_map(|node| match &node.action {
            Action::Measure { name, .. } | Action::Let { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    if let Some(name) = local_actions.iter().find(|name| {
        !crate::parser::is_valid_identifier(name) || name.contains('.') || name.contains("__")
    }) {
        return Err(fail(format!("invalid local action name '{name}'")));
    }
    let bit_inputs = module
        .interface
        .bit_inputs
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    if let Some(name) = bit_inputs.intersection(&local_actions).next() {
        return Err(fail(format!("bit input '{name}' is also a local action")));
    }
    let exported_bits = bit_inputs
        .iter()
        .copied()
        .chain(
            module
                .interface
                .bit_outputs
                .iter()
                .map(|output| output.name.as_str()),
        )
        .collect::<HashSet<_>>();

    for output in &module.interface.bit_outputs {
        for name in expr_names(&output.expr) {
            validate_bit_ref(
                &name,
                &bit_inputs,
                &local_actions,
                definitions,
                instances,
                fail,
            )?;
        }
    }
    for name in module.local_body().action_graph().inputs() {
        validate_bit_ref(
            name,
            &bit_inputs,
            &local_actions,
            definitions,
            instances,
            fail,
        )?;
    }

    let mut bound = HashSet::new();
    for binding in &module.bit_bindings {
        validate_bit_source(
            &binding.source,
            &exported_bits,
            definitions,
            instances,
            fail,
        )?;
        let target = instances
            .get(binding.target_instance.as_str())
            .ok_or_else(|| fail(format!("unknown instance '{}'", binding.target_instance)))?;
        let child = definitions[target.definition.as_str()];
        if !child.interface.bit_inputs.contains(&binding.target_bit) {
            return Err(fail(format!(
                "unknown bit input '{}.{}'",
                binding.target_instance, binding.target_bit
            )));
        }
        if !bound.insert((
            binding.target_instance.as_str(),
            binding.target_bit.as_str(),
        )) {
            return Err(fail(format!(
                "bit input '{}.{}' is bound more than once",
                binding.target_instance, binding.target_bit
            )));
        }
    }
    for instance in &module.instances {
        for input in &definitions[instance.definition.as_str()]
            .interface
            .bit_inputs
        {
            if !bound.contains(&(instance.name.as_str(), input.as_str())) {
                return Err(fail(format!(
                    "unbound bit input '{}.{}'",
                    instance.name, input
                )));
            }
        }
    }
    Ok(())
}

fn validate_bit_ref(
    name: &str,
    local_bits: &HashSet<&str>,
    local_actions: &HashSet<&str>,
    definitions: &HashMap<&str, &BlockGraph>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    if let Some((instance, bit)) = split_member(name) {
        return validate_child_bit_output(instance, bit, definitions, instances, fail);
    }
    if local_bits.contains(name) || local_actions.contains(name) {
        Ok(())
    } else {
        Err(fail(format!("unknown bit '{name}'")))
    }
}

fn validate_bit_source(
    source: &BitRef,
    local_bits: &HashSet<&str>,
    definitions: &HashMap<&str, &BlockGraph>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    match &source.instance {
        Some(instance) => {
            validate_child_bit_output(instance, &source.bit, definitions, instances, fail)
        }
        None if local_bits.contains(source.bit.as_str()) => Ok(()),
        None => Err(fail(format!("unknown exported bit '{}'", source.bit))),
    }
}

fn validate_child_bit_output(
    instance_name: &str,
    bit: &str,
    definitions: &HashMap<&str, &BlockGraph>,
    instances: &HashMap<&str, &ModuleInstance>,
    fail: &impl Fn(String) -> ModuleError,
) -> Result<(), ModuleError> {
    let instance = instances
        .get(instance_name)
        .ok_or_else(|| fail(format!("unknown instance '{instance_name}'")))?;
    if definitions[instance.definition.as_str()]
        .interface
        .bit_outputs
        .iter()
        .any(|output| output.name == bit)
    {
        Ok(())
    } else {
        Err(fail(format!("unknown bit output '{instance_name}.{bit}'")))
    }
}

fn port_direction(graph: &BlockGraph, position: IVec3) -> Result<PortDirection, String> {
    let block = graph
        .get_block(position)
        .ok_or_else(|| format!("no block at {position}"))?;
    if block.kind() != BlockKind::Port {
        return Err("interface block is not a Port".to_string());
    }
    let role = block.port_role().expect("Port has a role");
    let neighbors = graph.neighbor_positions(position);
    if role == PortRole::Multiplex {
        return match neighbors.as_slice() {
            // A parent Port can inherit its pipe from a bound child interface.
            // Definition geometry validation checks that complete connection.
            [] => Ok(PortDirection::Input),
            [neighbor] if neighbor.z == position.z => Ok(PortDirection::Input),
            _ => Err("multiplex Port must have one spatial pipe".to_string()),
        };
    }
    if neighbors.is_empty() {
        return match role {
            PortRole::Input => Ok(PortDirection::Input),
            PortRole::Output => Ok(PortDirection::Output),
            PortRole::Auto => Err("automatic Port must have one temporal pipe".to_string()),
            PortRole::Multiplex => unreachable!("multiplex ports returned above"),
        };
    }
    let [neighbor] = neighbors.as_slice() else {
        return Err("module Port must have one pipe".to_string());
    };
    if neighbor.z == position.z {
        return match role {
            PortRole::Input => Ok(PortDirection::Input),
            PortRole::Output => Ok(PortDirection::Output),
            PortRole::Auto => Err("spatial Port needs an explicit role".to_string()),
            PortRole::Multiplex => unreachable!("multiplex ports returned above"),
        };
    }
    let inferred = if neighbor.z > position.z {
        PortDirection::Input
    } else {
        PortDirection::Output
    };
    match role {
        PortRole::Auto => Ok(inferred),
        PortRole::Input if inferred == PortDirection::Input => Ok(inferred),
        PortRole::Output if inferred == PortDirection::Output => Ok(inferred),
        PortRole::Input | PortRole::Output => {
            Err("explicit role disagrees with temporal pipe".to_string())
        }
        PortRole::Multiplex => unreachable!("multiplex ports returned above"),
    }
}

fn validate_acyclic(
    modules: &[&BlockGraph],
    definitions: &HashMap<&str, &BlockGraph>,
) -> Result<(), ModuleError> {
    let mut active = HashSet::new();
    let mut done = HashSet::new();
    for module in modules {
        let mut pending = vec![(module.name.as_str(), false)];
        while let Some((name, exiting)) = pending.pop() {
            if exiting {
                active.remove(name);
                done.insert(name);
            } else if !done.contains(name) {
                if !active.insert(name) {
                    return Err(ModuleError::InvalidModule {
                        module: name.to_string(),
                        message: "module dependency cycle".to_string(),
                        span: None,
                    });
                }
                pending.push((name, true));
                pending.extend(
                    definitions[name]
                        .instances
                        .iter()
                        .rev()
                        .map(|instance| (instance.definition.as_str(), false)),
                );
            }
        }
    }
    Ok(())
}

type BitDependencies = HashMap<String, HashSet<String>>;

fn validate_classical_cycles(
    modules: &[&BlockGraph],
    definitions: &HashMap<&str, &BlockGraph>,
) -> Result<(), ModuleError> {
    let mut memo = HashMap::new();
    let mut seen = HashSet::new();
    for module in modules {
        let mut pending = vec![(*module, false)];
        while let Some((module, exiting)) = pending.pop() {
            if exiting {
                module_bit_dependencies(&module.name, definitions, &mut memo)?;
            } else if seen.insert(module.name.as_str()) {
                pending.push((module, true));
                pending.extend(
                    module
                        .instances
                        .iter()
                        .rev()
                        .map(|instance| (definitions[instance.definition.as_str()], false)),
                );
            }
        }
    }
    Ok(())
}

fn module_bit_dependencies(
    name: &str,
    definitions: &HashMap<&str, &BlockGraph>,
    memo: &mut HashMap<String, BitDependencies>,
) -> Result<BitDependencies, ModuleError> {
    if let Some(dependencies) = memo.get(name) {
        return Ok(dependencies.clone());
    }
    let module = definitions[name];
    let mut child_dependencies = HashMap::new();
    for instance in &module.instances {
        child_dependencies.insert(instance.name.as_str(), memo[&instance.definition].clone());
    }

    let mut dependencies = BitDependencies::new();
    let actions = module.local_body().action_graph();
    for node in actions.ordered_nodes() {
        let action = action_node(node.ordinal);
        dependencies.insert(
            action.clone(),
            action_expression(&node.action)
                .map(expr_names)
                .unwrap_or_default()
                .into_iter()
                .collect(),
        );
        if let Action::Let { name, .. } | Action::Measure { name, .. } = &node.action {
            dependencies.insert(name.clone(), HashSet::from([action]));
        }
    }
    for (predecessor, consumer, _) in actions.dependencies() {
        dependencies
            .entry(action_node(consumer))
            .or_default()
            .insert(action_node(predecessor));
    }
    for output in &module.interface.bit_outputs {
        dependencies.insert(
            output_node(&output.name),
            expr_names(&output.expr).into_iter().collect(),
        );
    }
    for instance in &module.instances {
        for (output, inputs) in &child_dependencies[instance.name.as_str()] {
            dependencies.insert(
                format!("{}.{}", instance.name, output),
                inputs
                    .iter()
                    .map(|input| format!("{}.{}", instance.name, input))
                    .collect(),
            );
        }
    }
    for binding in &module.bit_bindings {
        let source = match &binding.source.instance {
            Some(instance) => format!("{instance}.{}", binding.source.bit),
            None if module
                .interface
                .bit_outputs
                .iter()
                .any(|output| output.name == binding.source.bit) =>
            {
                output_node(&binding.source.bit)
            }
            None => binding.source.bit.clone(),
        };
        dependencies.insert(
            format!("{}.{}", binding.target_instance, binding.target_bit),
            HashSet::from([source]),
        );
    }

    let mut active = HashSet::new();
    let mut done = HashSet::new();
    let mut nodes = dependencies.keys().cloned().collect::<Vec<_>>();
    nodes.sort();
    for node in nodes {
        if let Some(cycle) = dependency_cycle(&node, &dependencies, &mut active, &mut done) {
            return Err(ModuleError::InvalidModule {
                module: module.name.clone(),
                message: format!(
                    "classical binding cycle at '{}'",
                    cycle.strip_prefix("@output:").unwrap_or(&cycle)
                ),
                span: None,
            });
        }
    }

    let inputs = module
        .interface
        .bit_inputs
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut output_dependencies = BitDependencies::new();
    for output in &module.interface.bit_outputs {
        let mut found = HashSet::new();
        collect_input_dependencies(
            &output_node(&output.name),
            &dependencies,
            &inputs,
            &mut HashSet::new(),
            &mut found,
        );
        output_dependencies.insert(output.name.clone(), found);
    }
    memo.insert(name.to_string(), output_dependencies.clone());
    Ok(output_dependencies)
}

fn output_node(name: &str) -> String {
    format!("@output:{name}")
}

fn action_node(ordinal: usize) -> String {
    format!("@action:{ordinal}")
}

fn action_expression(action: &Action) -> Option<&Expr> {
    match action {
        Action::Let { expr, .. } | Action::DiscardIf(expr) => Some(expr),
        Action::Resolve { condition, .. } | Action::Branch { condition, .. } => Some(condition),
        Action::Feedback { condition, .. } => condition.as_ref(),
        Action::Measure { .. } => None,
    }
}

fn dependency_cycle(
    node: &str,
    dependencies: &BitDependencies,
    active: &mut HashSet<String>,
    done: &mut HashSet<String>,
) -> Option<String> {
    if done.contains(node) {
        return None;
    }
    if !active.insert(node.to_string()) {
        return Some(node.to_string());
    }
    if let Some(next) = dependencies.get(node) {
        let mut next = next.iter().collect::<Vec<_>>();
        next.sort();
        for dependency in next {
            if let Some(cycle) = dependency_cycle(dependency, dependencies, active, done) {
                return Some(cycle);
            }
        }
    }
    active.remove(node);
    done.insert(node.to_string());
    None
}

fn collect_input_dependencies(
    node: &str,
    dependencies: &BitDependencies,
    inputs: &HashSet<String>,
    seen: &mut HashSet<String>,
    found: &mut HashSet<String>,
) {
    if inputs.contains(node) {
        found.insert(node.to_string());
        return;
    }
    if !seen.insert(node.to_string()) {
        return;
    }
    if let Some(next) = dependencies.get(node) {
        for dependency in next {
            collect_input_dependencies(dependency, dependencies, inputs, seen, found);
        }
    }
}

fn expr_names(expr: &Expr) -> BTreeSet<String> {
    fn collect(expr: &Expr, names: &mut BTreeSet<String>) {
        match expr {
            Expr::Var(name) => {
                names.insert(name.clone());
            }
            Expr::Not(inner) => collect(inner, names),
            Expr::Binary(_, left, right) => {
                collect(left, names);
                collect(right, names);
            }
        }
    }
    let mut names = BTreeSet::new();
    collect(expr, &mut names);
    names
}

pub(crate) fn split_member(name: &str) -> Option<(&str, &str)> {
    let (instance, member) = name.split_once('.')?;
    (!instance.is_empty() && !member.is_empty() && !member.contains('.'))
        .then_some((instance, member))
}

fn write_interface(out: &mut String, module: &BlockGraph, ids: &HashMap<IVec3, u32>) {
    if module.interface == ModuleInterface::default() {
        return;
    }
    for port in &module.interface.quantum_ports {
        let id = ids[&port.position];
        writeln!(
            out,
            "  {} {}: {} = {}",
            port.direction.keyword(),
            port.name,
            port.resource_type,
            id
        )
        .expect("writing to a String cannot fail");
    }
    for input in &module.interface.bit_inputs {
        writeln!(out, "  in {input}").expect("writing to a String cannot fail");
    }
    for output in &module.interface.bit_outputs {
        writeln!(out, "  out {} = {}", output.name, output.expr)
            .expect("writing to a String cannot fail");
    }
    out.push('\n');
}

fn write_instances(out: &mut String, module: &BlockGraph) {
    if module.instances.is_empty() {
        return;
    }
    for instance in &module.instances {
        write!(
            out,
            "  {}: {} @ {}",
            instance.name, instance.definition, instance.translation
        )
        .expect("writing to a String cannot fail");
        if !instance.rotation.is_identity() {
            write!(out, " rotate {}", instance.rotation).expect("writing to a String cannot fail");
        }
        out.push('\n');
    }
    out.push('\n');
}

fn write_body(out: &mut String, body: &BlockGraph) {
    let text = body.to_blog_body_text();
    let body = text
        .strip_prefix("BLOG 1.0\n\n")
        .expect("BlockGraph writer emits BLOG 1.0")
        .trim_end();
    if body.is_empty() {
        return;
    }
    for line in body.lines() {
        if line.is_empty() {
            out.push('\n');
        } else {
            writeln!(out, "  {line}").expect("writing to a String cannot fail");
        }
    }
    out.push('\n');
}

fn write_connections(out: &mut String, module: &BlockGraph, ids: &HashMap<IVec3, u32>) {
    if module.quantum_connections.is_empty() && module.bit_bindings.is_empty() {
        return;
    }
    let id = |block: &IVec3| {
        ids.get(block)
            .expect("validated connection block has a BLOG ID")
    };
    let arrow = |hadamard| if hadamard { "-H>" } else { "->" };
    for connection in &module.quantum_connections {
        match connection {
            QuantumConnection::Input {
                block,
                input,
                hadamard,
            } => writeln!(
                out,
                "  {} {} {}.{}",
                id(block),
                arrow(*hadamard),
                input.instance,
                input.port
            )
            .expect("writing to a String cannot fail"),
            QuantumConnection::Output {
                output,
                block,
                hadamard,
            } => writeln!(
                out,
                "  {}.{} {} {}",
                output.instance,
                output.port,
                arrow(*hadamard),
                id(block)
            )
            .expect("writing to a String cannot fail"),
            QuantumConnection::Pipe {
                output,
                input,
                hadamard,
            } => writeln!(
                out,
                "  {}.{} {} {}.{}",
                output.instance,
                output.port,
                arrow(*hadamard),
                input.instance,
                input.port
            )
            .expect("writing to a String cannot fail"),
        }
    }
    for binding in &module.bit_bindings {
        let source = match &binding.source.instance {
            Some(instance) => format!("{instance}.{}", binding.source.bit),
            None => binding.source.bit.clone(),
        };
        writeln!(
            out,
            "  {source} => {}.{}",
            binding.target_instance, binding.target_bit
        )
        .expect("writing to a String cannot fail");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CubeKind, FeedbackTarget, GalleryItem, Pauli, PauliBasis, SelectiveFixingTarget,
        SelectiveKind, Stabilizer, StabilizerGenerator, StabilizerGenerators, StabilizerRowKind,
        parse_inline_graph,
    };

    #[test]
    fn boolean_work_default_is_unlimited_but_explicit_limits_still_apply() {
        use bloq_utils::boolean::{BooleanDecisionDiagram, BooleanLimits};

        let defaults = ModuleCertificationLimits::DEFAULT;
        assert_eq!(defaults.max_boolean_steps, usize::MAX);
        assert_eq!(defaults.max_boolean_nodes, 4_000_000);
        assert_eq!(BooleanLimits::DEFAULT.max_steps, 64_000_000);
        let mut diagram = BooleanDecisionDiagram::with_limits(defaults.boolean_limits());
        diagram.charge(512_000_001).unwrap();
        assert_eq!(diagram.steps(), 512_000_001);

        let mut limits = defaults;
        limits.set("max_boolean_steps", 512_000_000).unwrap();
        let mut diagram = BooleanDecisionDiagram::with_limits(limits.boolean_limits());
        let error = diagram.charge(512_000_001).unwrap_err();
        assert_eq!((error.observed, error.limit), (512_000_001, 512_000_000));
        assert_eq!(diagram.steps(), 0);
    }

    #[test]
    fn named_limit_overrides_select_each_field_and_reject_unknown_names() {
        let mut limits = ModuleCertificationLimits::DEFAULT;
        for (value, field) in ModuleCertificationLimits::FIELD_NAMES
            .into_iter()
            .enumerate()
        {
            limits.set(field, value).unwrap();
        }
        assert_eq!(
            limits,
            ModuleCertificationLimits {
                max_expanded_blocks: 0,
                max_expanded_instances: 1,
                max_occupied_cells: 2,
                max_local_columns: 3,
                max_boolean_nodes: 4,
                max_boolean_steps: 5,
                max_matrix_words: 6,
                max_frontier_width: 7,
                max_witness_nodes: 8,
                max_normalization_states: 9,
                max_guarded_domain_size: 10,
            }
        );
        let previous = limits;
        let error = limits.set("max_booleen_steps", 0).unwrap_err();
        assert_eq!(error, UnknownCertificationLimit("max_booleen_steps".into()));
        assert_eq!(limits, previous);
        assert!(error.to_string().contains("max_boolean_steps"));
    }

    #[test]
    fn linked_port_feedback_preserves_its_wire_and_hadamard_frame() {
        let source = "BLOG 1.0
            module Wire {
                in input: data = 0
                out output: data = 2
                0: Port [0,0,-1] role=input
                1: ZXZ [0,0,0]
                2: Port [0,0,1] role=output
                0 -> +Z
                1 -H> +Z
                feedback X 2
            }
            module main {
                in input: data = 0
                out a: data = 2
                out b: data = 3
                0: Port [0,0,-1] role=input
                1: XZX [0,0,1]
                2: Port [0,0,2] role=output
                3: Port [1,0,1] role=output
                wire: Wire @ [0,0,0]
                0 -> wire.input
                wire.output -> 1
                1 -> +Z
                1 -> +X
            }";
        for direction in ["", " -> -Z"] {
            for binding_hadamard in [false, true] {
                let source = source.replace("feedback X 2", &format!("feedback X 2{direction}"));
                let source = if binding_hadamard {
                    source
                        .replace("wire.output -> 1", "wire.output -H> 1")
                        .replace("1: XZX [0,0,1]", "1: ZXZ [0,0,1]")
                } else {
                    source
                };
                let program = parse_inline_graph(&source).unwrap();
                let graph = program.materialize_flat_graph().unwrap();
                let expected = Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::Z,
                        target: IVec3::ZERO,
                        direction: Some(Direction::ZPLUS),
                    }],
                    condition: None,
                };
                // The child output X crosses its original H to become Z at the
                // retained endpoint. The later binding H must not change this action.
                assert_eq!(
                    graph.actions(),
                    vec![expected.clone()],
                    "direction={direction:?}, binding_hadamard={binding_hadamard}",
                );
                let round_trip = crate::parse_blog_to_graph(&graph.to_blog_text()).unwrap();
                assert_eq!(round_trip.actions(), vec![expected]);
                let zx = crate::ZXGraph::try_from(&graph).unwrap();
                let center = zx.node_at(IVec3::Z).unwrap();
                let mut row = crate::PauliString::new(zx.total_ids());
                for pos in [IVec3::new(0, 0, 2), IVec3::new(1, 0, 1)] {
                    let output = zx.node_at(pos).unwrap();
                    row.set(output.id, Pauli::Z);
                    row.set(zx.edge_id(center.id, output.id).unwrap(), Pauli::Z);
                    row.set(zx.edge_id(output.id, center.id).unwrap(), Pauli::Z);
                }
                let surface = zx.materialize_stabilizer_with_sign(row, false);
                let Action::Feedback { targets, .. } = &graph.actions()[0] else {
                    unreachable!()
                };
                assert!(
                    !surface.odd_anticommutes_feedback(targets, Some(&zx)),
                    "X on the incoming wire preserves the two-output ZZ relation"
                );
                assert!(
                    surface.odd_anticommutes_feedback(
                        &[FeedbackTarget {
                            pauli: PauliBasis::X,
                            target: center.pos,
                            direction: None
                        }],
                        Some(&zx)
                    ),
                    "the former junction target has different semantics"
                );
            }
        }
    }

    const STAGE: &str = r#"module Stage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, -1] <q_in>
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] <q_out>
  [0, 0, -1] -> +Z
  [0, 0, 0] -> +Z
}
"#;

    const MEASURED_STAGE: &str = r#"module MeasuredStage {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, 0] role=input <q_in>
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <q_out>
  3: ZXZ [1, 0, 0]
  4: Z [1, 0, 1]
  0 -> +Z
  1 -> +Z
  3 -> +Z
  m = measure 4
}
"#;

    const PAIR: &str = r#"module main {
  in q_in: data = 100
  out q_out: data = 103
  lower: Stage @ [0, 0, 0]
  upper: Stage @ [0, 0, 2]
  100: Port [0, 0, -1] role=input <q_in>
  101: ZXZ [0, 0, 1]
  103: Port [0, 0, 3] role=output <q_out>
  100 -> lower.q_in
  lower.q_out -> 101
  101 -> upper.q_in
  upper.q_out -> 103
}
"#;

    const DIRECT_PAIR: &str = r#"module main {
  in q_in: data = 100
  out q_out: data = 101
  lower: Stage @ [0, 0, 0]
  upper: StageDual @ [0, 0, 1]
  100: Port [0, 0, -1] role=input <q_in>
  101: Port [0, 0, 2] role=output <q_out>
  100 -> lower.q_in
  lower.q_out -H> upper.q_in
  upper.q_out -> 101
}
"#;

    fn pair_source(stage: &str) -> String {
        format!("BLOG 1.0\n\n{stage}\n{PAIR}")
    }

    fn direct_pair_source() -> String {
        let dual = STAGE
            .replacen("module Stage", "module StageDual", 1)
            .replacen("1: ZXZ", "1: XZX", 1);
        format!("BLOG 1.0\n\n{STAGE}\n{dual}\n{DIRECT_PAIR}")
    }

    fn child_loop(body: BlockGraph) -> Result<BlockGraph, ModuleError> {
        let child = BlockGraph::definition(
            "Child".to_string(),
            body,
            ModuleInterface {
                bit_inputs: vec!["i".to_string()],
                bit_outputs: vec![BitOutput {
                    name: "o".to_string(),
                    expr: Expr::Var("m".to_string()),
                }],
                ..ModuleInterface::default()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let root = BlockGraph::definition(
            "main".to_string(),
            BlockGraph::new(),
            ModuleInterface::default(),
            vec![ModuleInstance {
                name: "child".to_string(),
                definition: "Child".to_string(),
                rotation: ModuleRotation::IDENTITY,
                translation: IVec3::ZERO,
            }],
            Vec::new(),
            vec![BitBinding {
                source: BitRef {
                    instance: Some("child".to_string()),
                    bit: "o".to_string(),
                },
                target_instance: "child".to_string(),
                target_bit: "i".to_string(),
            }],
        );
        BlockGraph::from_definitions(vec![child, root])
    }

    fn analyzed_body(
        mut body: BlockGraph,
        actions: Vec<Action>,
        generators: Vec<StabilizerGenerator>,
    ) -> BlockGraph {
        body.set_actions_deferred_with_inputs(actions, ["i".to_string()])
            .unwrap();
        let zx = body.to_zx_graph().unwrap();
        body.with_analyzed_action_graph(&StabilizerGenerators::new(zx, generators))
            .unwrap()
    }

    fn y_seam_source() -> String {
        r#"BLOG 1.0

module YCap {
  in q: data = 0
  0: Port [0, 0, 0] role=input <q>
  1: Y [0, 0, 1]
  [0, 0, 0] -> +Z
}

module main {
  in q: data = 10
  y: YCap @ [0, 0, 0]
  10: Port [0, 0, 0] role=input <q>
  10 -> y.q
}

"#
        .to_string()
    }

    fn direct_y_seam_source() -> String {
        r#"BLOG 1.0

module YPrep {
  out q: data = 1
  0: Y [0, 0, 0]
  1: Port [0, 0, 1] role=output <q>
  [0, 0, 0] -> +Z
}

module Wire {
  in q_in: data = 0
  out q_out: data = 2
  0: Port [0, 0, -1] role=input <q_in>
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] role=output <q_out>
  [0, 0, -1] -> +Z
  [0, 0, 0] -> +Z
}

module main {
  out q: data = 10
  y: YPrep @ [0, 0, 0]
  wire: Wire @ [0, 0, 1]
  10: Port [0, 0, 2] role=output <q>
  y.q -> wire.q_in
  wire.q_out -> 10
}

module FlatY {
  out q: data = 10
  0: Y [0, 0, 0]
  1: ZXZ [0, 0, 1]
  10: Port [0, 0, 2] role=output <q>
  [0, 0, 0] -> +Z
  [0, 0, 1] -> +Z
}

"#
        .to_string()
    }

    fn direct_chain_source(len: usize) -> String {
        let mut instances = String::new();
        let mut connections = String::from("  100 -> s0.q_in\n");
        for index in 0..len {
            writeln!(instances, "  s{index}: Stage @ [0, 0, {index}]").unwrap();
            if index + 1 < len {
                writeln!(connections, "  s{index}.q_out -> s{}.q_in", index + 1).unwrap();
            }
        }
        writeln!(connections, "  s{}.q_out -> 101", len - 1).unwrap();
        format!(
            "BLOG 1.0\n\n{STAGE}\nmodule main {{\n  in q_in: data = 100\n  out q_out: data = 101\n{instances}  100: Port [0, 0, -1] role=input <q_in>\n  101: Port [0, 0, {len}] role=output <q_out>\n{connections}}}\n"
        )
    }

    fn measured_chain_source(len: usize) -> String {
        let mut instances = String::new();
        let mut connectors = String::new();
        let mut connections = String::from("  100 -> s0.q_in\n");
        for index in 0..len {
            writeln!(
                instances,
                "  s{index}: MeasuredStage @ [0, 0, {}]",
                index * 2
            )
            .unwrap();
            if index + 1 < len {
                writeln!(
                    connectors,
                    "  {}: XZX [0, 0, {}]",
                    200 + index,
                    (index + 1) * 2
                )
                .unwrap();
                writeln!(connections, "  s{index}.q_out -> {}", 200 + index).unwrap();
                writeln!(connections, "  {} -> s{}.q_in", 200 + index, index + 1).unwrap();
            }
        }
        writeln!(connections, "  s{}.q_out -> 101", len - 1).unwrap();
        format!(
            "BLOG 1.0\n\n{MEASURED_STAGE}\nmodule main {{\n  in q_in: data = 100\n  out q_out: data = 101\n{instances}  100: Port [0, 0, 0] role=input <q_in>\n{connectors}  101: Port [0, 0, {}] role=output <q_out>\n{connections}}}\n",
            len * 2
        )
    }

    #[test]
    fn module_program_round_trips_shared_replacement_block() {
        let program = parse_inline_graph(&pair_source(STAGE)).unwrap();
        assert_eq!(program.modules().len(), 2);
        assert_eq!(program.root().name, "main");

        let written = program.to_blog_text();
        let reparsed = parse_inline_graph(&written).unwrap();
        assert_eq!(reparsed.root().quantum_connections.len(), 4);
        assert_eq!(reparsed.to_blog_text(), written);
    }

    #[test]
    fn linking_a_branch_input_port_preserves_its_authored_cuts() {
        let source = r"BLOG 1.0
module Child {
    in q: data = 0
    in enabled
    0: Port [0, 0, 0] role=input
    branch choice {
        false {
            1: X [0, 0, 1]
            0 -> 1
        }
        true {
            2: ZXZ [0, 0, 1]
            0 -> 2
        }
    }
    resolve choice if enabled
}
module main {
    in enabled
    0: ZXZ [0, 0, 0]
    child: Child @ [0, 0, 0]
    0 -> child.q
    enabled => child.enabled
}
";
        for hadamard in [false, true] {
            let source = if hadamard {
                source
                    .replace("0 -> child.q", "0 -H> child.q")
                    .replace("0: ZXZ [0, 0, 0]", "0: XZX [0, 0, 0]")
            } else {
                source.to_string()
            };
            let program = parse_inline_graph(&source).unwrap();
            let mut graph = program.materialize_flat_graph().unwrap();
            let region = graph.branch_by_name("child__choice").unwrap();
            for arm in [region.on_true(), region.on_false()] {
                let pipes = arm.pipes().collect::<Vec<_>>();
                assert_eq!(pipes.len(), 1);
                assert_eq!(pipes[0].is_hadamard(), hadamard);
            }
            graph.validate().unwrap();
            graph.set_shown_branch_arm("child__choice", false).unwrap();
            assert_eq!(graph.pipes().next().unwrap().is_hadamard(), hadamard);
            let pipe = graph.pipes().next().unwrap().clone();
            let (src, dst) = pipe.endpoints();
            let mut ordinary = graph.clone();
            let mut indexed = graph.clone();
            ordinary.remove_pipe(src, dst).unwrap();
            indexed.remove_pipe(src, dst).unwrap();
            let mut index = LinkBranchPipeIndex::default();
            for (ordinal, region) in indexed.branches.iter().enumerate() {
                index.add_region(ordinal, region);
            }
            ordinary.try_add_pipe(pipe.clone()).unwrap();
            add_link_pipe(&mut indexed, &index, pipe).unwrap();
            assert_eq!(indexed.branch_definitions(), ordinary.branch_definitions());
            let exported = graph.to_program_blog_text().unwrap();
            let reparsed = parse_inline_graph(&exported).unwrap();
            reparsed
                .materialize_flat_graph()
                .unwrap()
                .validate()
                .unwrap();
            assert_eq!(reparsed.to_blog_text(), exported);
        }
    }

    #[test]
    fn linked_occupancy_rejects_a_hidden_arm_footprint() {
        let source = r#"BLOG 1.0
module main {
  0: ZXZ [0,0,0]
  9: Z [2,0,0]
  branch b {
    false {
      1: ZXZ [0,0,1]
      2: ZXZ [1,0,1]
      0 -> +Z
      1 -> +X
    }
    true {
      3: ZXZ [0,0,1]
      4: ZXZ [-1,0,1]
      0 -> +Z
      3 -> -X
    }
  }
  m = measure 9
  resolve b if m
}
"#;
        let graph = parse_inline_graph(source)
            .unwrap()
            .root()
            .copy_local_geometry();
        let hidden = IVec3::new(1, 0, 1);
        assert!(graph.get_block(hidden).is_none());
        let mut index = LinkBranchOccupancy::default();
        for region in graph.branch_definitions() {
            index.add_region(region).unwrap();
        }
        let block = Block::new(hidden, BlockKind::Cube(CubeKind::ZXZ));
        let ordinary = graph.clone().try_add_block(block.clone()).unwrap_err();
        let indexed = graph
            .clone()
            .try_add_block_with_branch_occupancy(block, Some(&index))
            .unwrap_err();
        assert_eq!(ordinary.to_string(), indexed.to_string());
        assert!(matches!(
            indexed,
            crate::BlockGraphError::BlockPositionOccupied(position) if position == hidden
        ));
    }

    #[test]
    fn definition_cache_keys_keep_shared_dependencies_once() {
        let mut modules = Vec::new();
        for index in 0_usize..21 {
            modules.push(BlockGraph::definition(
                if index == 20 {
                    "main".into()
                } else {
                    format!("m{index}")
                },
                BlockGraph::new(),
                ModuleInterface::default(),
                (index.saturating_sub(2)..index)
                    .map(|child| ModuleInstance {
                        name: format!("c{child}"),
                        definition: format!("m{child}"),
                        rotation: ModuleRotation::IDENTITY,
                        translation: IVec3::ZERO,
                    })
                    .collect(),
                Vec::new(),
                Vec::new(),
            ));
        }
        let program = BlockGraph::from_definitions(modules).unwrap();
        let before = program.module_definition_cache_keys();
        assert_eq!(before.len(), 21);
        assert_eq!(before["main"].matches("module m0 {").count(), 1);
        assert!(before["main"].len() <= program.to_blog_text().len() + 20);

        let mut changed = program;
        changed.module_mut("m1").unwrap().instances[0].rotation =
            ModuleRotation::new(UDirection::Z, 1);
        changed.validate().unwrap();
        let after = changed.module_definition_cache_keys();
        assert_eq!(before["m0"], after["m0"]);
        for name in ["m1", "m2", "m19", "main"] {
            assert_ne!(
                before[name], after[name],
                "a changed shared dependency must invalidate {name}"
            );
        }
        changed.definitions.reverse();
        assert_eq!(after, changed.module_definition_cache_keys());
    }

    #[test]
    fn module_rotations_round_trip_and_transform_all_axes() {
        let source = "BLOG 1.0\n\nmodule Cell {\n  0: XZZ [0,1,2]\n}\nmodule main {\n  x: Cell @ [10,0,0] rotate X 90\n  y: Cell @ [20,0,0] rotate Y -90\n  z: Cell @ [30,0,0] rotate Z 180\n}\n";
        let program = parse_inline_graph(source).unwrap();
        let graph = program.materialize_flat_graph().unwrap();

        assert!(graph.has_block_at(IVec3::new(10, -2, 1)));
        assert!(graph.has_block_at(IVec3::new(18, 1, 0)));
        assert!(graph.has_block_at(IVec3::new(30, -1, 2)));
        let written = program.to_blog_text();
        assert!(written.contains("rotate X 90"));
        assert!(written.contains("rotate Y 270"));
        assert!(written.contains("rotate Z 180"));
        assert_eq!(
            parse_inline_graph(&written).unwrap().to_blog_text(),
            written
        );

        let nested = ModuleRotation::new(UDirection::Y, -1)
            .orientation()
            .then(ModuleRotation::new(UDirection::X, 1));
        assert_eq!(
            nested.try_rotate_position(IVec3::new(1, 2, 3)).unwrap(),
            IVec3::new(-2, -3, 1)
        );
    }

    #[test]
    fn module_rotation_constraints_are_checked_during_validation() {
        let source = |axis, degrees| {
            format!(
                "BLOG 1.0\n\nmodule Magic {{\n  0: T [0,0,0]\n}}\nmodule main {{\n  magic: Magic @ [0,0,0] rotate {axis} {degrees}\n}}\n"
            )
        };

        parse_inline_graph(&source("Z", 90)).expect("time-axis rotation is valid");
        for axis in ["X", "Y"] {
            assert!(matches!(
                parse_inline_graph(&source(axis, 90)),
                Err(ModuleError::InvalidGeometry {
                    source: ModuleCertificationError::Graph {
                        source: crate::BlockGraphError::RotationUnsupportedDynamicBlocks,
                        ..
                    },
                    ..
                })
            ));
        }
        assert!(matches!(
            parse_inline_graph(&source("Z", 45)),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn module_program_requires_main() {
        assert!(matches!(
            parse_inline_graph("BLOG 1.0\nmodule Helper {\n}\n"),
            Err(ModuleError::MissingMain)
        ));
    }

    #[test]
    fn graph_program_serializes_main() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::NEG_Z, BlockKind::Port));
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(crate::Pipe::new(IVec3::NEG_Z, crate::Direction::ZPLUS));

        let text = (graph).with_inferred_interface().unwrap().to_blog_text();

        assert!(text.contains("module main {"));
        assert!(text.contains("in q0: data"));
        parse_inline_graph(&text).unwrap();
    }

    #[test]
    fn module_bit_bindings_round_trip() {
        let source = "BLOG 1.0\n\nmodule Child {\n  in enable\n  out done = enable\n}\n\nmodule main {\n  in enable\n  out done = child.done\n  child: Child @ [0,0,0]\n  enable => child.enable\n}\n";
        let program = parse_inline_graph(source).unwrap();

        assert_eq!(program.root().bit_bindings.len(), 1);
        let written = program.to_blog_text();
        assert_eq!(
            parse_inline_graph(&written).unwrap().to_blog_text(),
            written
        );
    }

    #[test]
    fn explicit_temporal_port_role_must_match_geometry() {
        let source = "BLOG 1.0\n\nmodule main {\n  in q: data = 0\n  0: Port [0,0,1] role=input <q>\n  1: ZXZ [0,0,0]\n  1 -> +Z\n}\n";

        assert!(matches!(
            parse_inline_graph(source),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn explicit_spatial_port_role_is_valid() {
        let source = "BLOG 1.0\n\nmodule main {\n  in q: data = 0\n  0: Port [0,0,0] role=input <q>\n  1: ZXZ [1,0,0]\n  0 -> +X\n}\n";

        parse_inline_graph(source).expect("spatial interface has explicit direction");
    }

    #[test]
    fn explicit_port_role_still_requires_one_local_pipe() {
        let source = "BLOG 1.0\n\nmodule main {\n  in q: data = 0\n  0: Port [0,0,0] role=input <q>\n  1: ZXZ [1,0,0]\n  2: ZXZ [0,1,0]\n  0 -> +X\n  0 -> +Y\n}\n";

        assert!(matches!(
            parse_inline_graph(source),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn multiplex_interface_can_bind_a_child_port() {
        let source = "BLOG 1.0\nmodule Wire {\n\
            in q: data = 0\nout result: data = 2\n\
            0: Port [1,0,0] role=multiplex\n1: ZXZ [0,0,0]\n2: Port [0,0,1]\n\
            1 -> +X\n1 -> +Z\n}\nmodule main {\n\
            in q: data = 10\nout result: data = 11\n\
            wire: Wire @ [0,0,0]\n\
            10: Port [1,0,0] role=multiplex\n11: Port [0,0,1] role=output\n\
            10 -> wire.q\nwire.result -> 11\n}\n";
        let program = parse_inline_graph(source).unwrap();
        let graph = program.materialize_root_graph().unwrap();
        graph.validate().unwrap();
        assert_eq!(
            graph.get_block(IVec3::X).unwrap().port_role(),
            Some(PortRole::Multiplex)
        );
        parse_inline_graph(&source.replace("10 -> wire.q\n", "")).unwrap_err();
        parse_inline_graph(
            "BLOG 1.0\nmodule main {\nin q: data = 0\n\
                 0: Port [0,0,0] role=multiplex\n}\n",
        )
        .unwrap_err();
    }

    #[test]
    fn direct_seams_are_geometry_validated_during_parse() {
        let distant = direct_pair_source().replace(
            "upper: StageDual @ [0, 0, 1]",
            "upper: StageDual @ [0, 0, 10]",
        );
        assert!(matches!(
            parse_inline_graph(&distant),
            Err(ModuleError::InvalidModule { .. })
        ));

        let wrong_basis =
            direct_pair_source().replace("lower.q_out -H> upper.q_in", "lower.q_out -> upper.q_in");
        let error = parse_inline_graph(&wrong_basis).unwrap_err();
        assert!(matches!(&error, ModuleError::InvalidGeometry { .. }));
        let span = error.span().expect("geometry error keeps its module span");
        assert!(wrong_basis[span.start as usize..span.end as usize].starts_with("module main"));
    }

    #[test]
    fn module_bind_rejects_branch_local_blocks_before_writing() {
        let mut child = parse_inline_graph(
            "BLOG 1.0\n\nmodule main {\n  in q: data = 0\n  0: Port [0,0,0] role=input <q>\n  1: ZXZ [0,0,1]\n  0 -> +Z\n}\n",
        )
        .unwrap()
        .root()
        .clone();
        child.name = "Child".to_string();
        let body = crate::parse_blog_to_graph(
            "BLOG 1.0\n\n9: Z [3,0,0]\n10: ZXZ [0,0,-1]\n[0,0,-1] -> +Z\nbranch b {\n  false {\n    1: ZXZ [0,0,0]\n  }\n  true {\n    2: ZXZ [0,0,0]\n  }\n}\nm = measure 9\nresolve b if m\n",
        )
        .unwrap();
        let parent = BlockGraph::definition(
            "main".to_string(),
            body,
            ModuleInterface::default(),
            vec![ModuleInstance {
                name: "child".to_string(),
                definition: "Child".to_string(),
                rotation: ModuleRotation::IDENTITY,
                translation: IVec3::ZERO,
            }],
            vec![QuantumConnection::Input {
                block: IVec3::ZERO,
                input: InstancePort {
                    instance: "child".to_string(),
                    port: "q".to_string(),
                },
                hadamard: false,
            }],
            Vec::new(),
        );

        assert!(matches!(
            BlockGraph::from_definitions(vec![child, parent]),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn classical_parent_output_cycle_is_rejected() {
        let source = "BLOG 1.0\n\nmodule Invert {\n  in i\n  out o = !i\n}\n\nmodule main {\n  out x = child.o\n  child: Invert @ [0,0,0]\n  x => child.i\n}\n";

        assert!(matches!(
            parse_inline_graph(source),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn classical_sibling_cycle_is_rejected() {
        let source = "BLOG 1.0\n\nmodule Pass {\n  in i\n  out o = i\n}\n\nmodule main {\n  a: Pass @ [0,0,0]\n  b: Pass @ [0,0,0]\n  a.o => b.i\n  b.o => a.i\n}\n";

        assert!(matches!(
            parse_inline_graph(source),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn input_independent_child_output_does_not_create_a_cycle() {
        let source = "BLOG 1.0\n\nmodule Independent {\n  in i\n  out o = m\n  0: Z [0,0,0]\n  m = measure 0\n}\n\nmodule main {\n  child: Independent @ [0,0,0]\n  child.o => child.i\n}\n";

        parse_inline_graph(source).expect("child output does not depend on its input");
    }

    #[test]
    fn feedback_dependency_closes_a_child_binding_cycle() {
        let target = IVec3::ZERO;
        let mut body = BlockGraph::new();
        body.add_block(Block::new(target, BlockKind::Cube(CubeKind::ZXZ)));
        body.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        body.add_pipe(crate::Pipe::new(target, crate::Direction::ZPLUS));
        let zx = body.to_zx_graph().unwrap();
        let row = crate::PauliString::from_terms(
            zx.total_ids(),
            (0..zx.total_ids()).map(|column| (column, Pauli::Z)),
        );
        let surface = zx.materialize_stabilizer(row);
        let body = analyzed_body(
            body,
            vec![
                Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::X,
                        target,
                        direction: None,
                    }],
                    condition: Some(Expr::Var("i".to_string())),
                },
                Action::Measure {
                    target: crate::MeasureTarget::Node(target),
                    name: "m".to_string(),
                },
            ],
            vec![StabilizerGenerator::new(
                surface,
                StabilizerRowKind::Measurement {
                    name: "m".to_string(),
                },
            )],
        );

        assert!(matches!(
            child_loop(body),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn selective_dependency_closes_a_child_binding_cycle() {
        let target = IVec3::ZERO;
        let mut body = BlockGraph::new();
        body.add_block(Block::new(IVec3::NEG_Z, BlockKind::Cube(CubeKind::ZXZ)));
        body.add_block(Block::new(target, BlockKind::Selective(SelectiveKind::XY)));
        body.add_pipe(crate::Pipe::new(IVec3::NEG_Z, crate::Direction::ZPLUS));
        let body = analyzed_body(
            body,
            vec![
                Action::Resolve {
                    target,
                    condition: Expr::Var("i".to_string()),
                },
                Action::Measure {
                    target: crate::MeasureTarget::Node(target),
                    name: "m".to_string(),
                },
            ],
            vec![
                StabilizerGenerator::new(
                    Stabilizer::from_interior_nodes([(target, Pauli::X)]),
                    StabilizerRowKind::Measurement {
                        name: "m".to_string(),
                    },
                ),
                StabilizerGenerator::new(
                    Stabilizer::from_interior_nodes([(target, Pauli::Y)]),
                    StabilizerRowKind::SelectiveFixing {
                        targets: vec![SelectiveFixingTarget {
                            pos: target,
                            forbidden: Pauli::Y,
                        }],
                    },
                ),
            ],
        );

        assert!(matches!(
            child_loop(body),
            Err(ModuleError::InvalidModule { .. })
        ));
    }

    #[test]
    fn output_can_export_a_same_named_local_action() {
        let source = "BLOG 1.0\n\nmodule main {\n  out m = m\n  0: Z [0,0,0]\n  m = measure 0\n}\n";

        parse_inline_graph(source).expect("output and local action have separate nodes");
    }

    #[test]
    fn programmatic_bit_input_cannot_shadow_a_local_action() {
        let body = crate::parse_blog_to_graph("BLOG 1.0\n\n0: Z [0,0,0]\nm = measure 0\n").unwrap();
        let module = BlockGraph::definition(
            "main".to_string(),
            body,
            ModuleInterface {
                bit_inputs: vec!["m".to_string()],
                ..ModuleInterface::default()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        assert!(matches!(
            BlockGraph::from_definitions(vec![module]),
            Err(ModuleError::InvalidModule { message, .. })
                if message.contains("also a local action")
        ));
    }

    #[test]
    fn module_local_actions_use_contextual_identifier_grammar() {
        let source =
            "BLOG 1.0\n\nmodule main {\n  in value\n  module = value\n  path/name = module\n}\n";

        parse_inline_graph(source).expect("contextual and slash action names are valid");
    }

    #[test]
    fn module_namespaces_reserve_the_instance_path_separator() {
        for source in [
            "BLOG 1.0\n\nmodule Leaf {\n}\nmodule main {\n  bad__child: Leaf @ [0,0,0]\n}\n",
            "BLOG 1.0\n\nmodule main {\n  in bad__input\n}\n",
            "BLOG 1.0\n\nmodule main {\n  in value\n  out bad__output = value\n}\n",
            "BLOG 1.0\n\nmodule main {\n  in value\n  bad__local = value\n}\n",
        ] {
            let error = parse_inline_graph(source).unwrap_err();
            assert!(error.to_string().contains("invalid"), "{error}");
        }

        let mut program = GalleryItem::CCZGateTeleport.build();
        program.branches.first_mut().unwrap().name = "bad__branch".into();
        let error = program.validate().unwrap_err();
        assert!(error.to_string().contains("invalid branch name"), "{error}");
    }

    #[test]
    fn leaf_certification_projects_rows_and_reports_limits() {
        let program = parse_inline_graph(&pair_source(STAGE)).unwrap();
        let certificate = program
            .certify_leaf("Stage", ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        assert_eq!(certificate.boundary_rows().len(), 2);
        let row = certificate.materialize_boundary_row(0).unwrap();
        assert!(row.paulis.len() > 2);

        let error = program
            .certify_leaf(
                "Stage",
                ModuleCertificationLimits {
                    max_local_columns: 0,
                    ..ModuleCertificationLimits::UNLIMITED
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ModuleCertificationError::ResourceLimited {
                phase: "local ZX columns",
                ..
            }
        ));
    }

    #[test]
    fn leaf_certification_bounds_closure_searches() {
        let source = r#"BLOG 1.0

module main {
  out out: data = 2
  1: XZX [0, 0, 1]
  2: Port [0, 0, 2] role=output <out>
  3: T [1, 0, 0]
  4: XZX [1, 0, 1]
  5: YX [1, 0, 2]
  [0, 0, 1] -> +Z
  [0, 0, 1] -> +X
  [1, 0, 0] -> +Z
  [1, 0, 2] -> -Z
  mzz = measure 1 -> +X
  resolve 5 if mzz
}

"#;
        let program = parse_inline_graph(source).unwrap();
        let error = program
            .certify_leaf(
                "main",
                ModuleCertificationLimits {
                    max_guarded_domain_size: 1,
                    ..ModuleCertificationLimits::UNLIMITED
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ModuleCertificationError::ResourceLimited {
                phase: "guarded-domain branches",
                observed: 2,
                limit: 1,
                ..
            }
        ));

        let error = program
            .certify_leaf(
                "main",
                ModuleCertificationLimits {
                    max_normalization_states: 0,
                    ..ModuleCertificationLimits::UNLIMITED
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ModuleCertificationError::ResourceLimited {
                phase: "selective fixing setup states",
                observed: 1,
                limit: 0,
                ..
            }
        ));
    }

    #[test]
    fn child_summaries_compose_bind_and_direct_seams() {
        let summary = parse_inline_graph(&pair_source(STAGE))
            .unwrap()
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        assert_eq!(summary.quantum_ports().len(), 2);
        assert_eq!(summary.boundary_rows().len(), 2);
        for row in summary.boundary_rows() {
            assert_eq!(row.phase(), 0);
            assert_eq!(row.paulis.get(0), row.paulis.get(1));
            assert_ne!(row.paulis.get(0), Pauli::I);
        }

        let direct = parse_inline_graph(&direct_pair_source())
            .unwrap()
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap();
        let rows = direct
            .boundary_rows()
            .map(|row| ((row.paulis.get(0), row.paulis.get(1)), row.phase()))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![((Pauli::X, Pauli::Z), 0), ((Pauli::Z, Pauli::X), 0)]
        );
    }

    #[test]
    fn module_summary_jobs_preserve_output() {
        let program = parse_inline_graph(&direct_pair_source()).unwrap();
        let rows = |jobs| {
            program
                .summarize_root_with_jobs(ModuleCertificationLimits::UNLIMITED, jobs)
                .unwrap()
                .boundary_rows()
                .cloned()
                .collect::<Vec<_>>()
        };

        assert_eq!(
            rows(std::num::NonZeroUsize::MIN),
            rows(std::num::NonZeroUsize::new(4).unwrap())
        );
    }

    #[test]
    fn y_supported_seams_preserve_flattened_signs() {
        let bound = parse_inline_graph(&y_seam_source()).unwrap();
        let leaf = bound
            .certify_leaf("YCap", ModuleCertificationLimits::UNLIMITED)
            .unwrap()
            .boundary_rows()
            .cloned()
            .collect::<Vec<_>>();
        let composed = bound
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap()
            .boundary_rows()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(composed, leaf);

        let direct = parse_inline_graph(&direct_y_seam_source()).unwrap();
        let composed = direct
            .summarize_root(ModuleCertificationLimits::UNLIMITED)
            .unwrap()
            .boundary_rows()
            .cloned()
            .collect::<Vec<_>>();
        let flat = direct
            .certify_leaf("FlatY", ModuleCertificationLimits::UNLIMITED)
            .unwrap()
            .boundary_rows()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(composed, flat);
    }

    #[test]
    fn module_program_rejects_overlapping_translated_child_bodies() {
        const CLOSED: &str = "module Closed {\n  0: ZXZ [0, 0, 0]\n}";
        let root = "module main {\n  child: Closed @ [0, 0, 0]\n  0: ZXZ [0, 0, 0]\n}";
        let source = format!("BLOG 1.0\n\n{CLOSED}\n{root}\n");
        let error = parse_inline_graph(&source).unwrap_err();
        assert!(matches!(
            error,
            ModuleError::InvalidGeometry {
                source: ModuleCertificationError::Graph {
                    source: crate::BlockGraphError::ModuleBlockOverlap { .. },
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn module_overlap_error_attributes_both_instances() {
        let source = "BLOG 1.0\n\nmodule Closed {\n  0: ZXZ [0, 0, 0]\n}\nmodule main {\n  a: Closed @ [0, 0, 0]\n  b: Closed @ [0, 0, 0]\n}\n";
        let error = parse_inline_graph(source).unwrap_err();
        let message = error.to_string();
        let ModuleError::InvalidGeometry {
            source:
                ModuleCertificationError::Graph {
                    source:
                        crate::BlockGraphError::ModuleBlockOverlap {
                            position,
                            first,
                            second,
                        },
                    ..
                },
            ..
        } = error
        else {
            panic!("unexpected error: {message}");
        };
        let mut paths = [first.instance_path.as_str(), second.instance_path.as_str()];
        paths.sort_unstable();

        assert_eq!(position, IVec3::ZERO);
        assert_eq!(paths, ["a", "b"]);
        assert_eq!(first.definition, "Closed");
        assert_eq!(second.definition, "Closed");
        assert_eq!(first.local_position, IVec3::ZERO);
        assert_eq!(second.local_position, IVec3::ZERO);
        assert!(message.contains("instance 'a'"), "{message}");
        assert!(message.contains("instance 'b'"), "{message}");
    }

    #[test]
    fn module_reserved_overlap_names_the_footprint_cell() {
        let source = "BLOG 1.0\n\nmodule Tall {\n  0: ZXZ [0, 0, 0] height=3d\n}\nmodule main {\n  a: Tall @ [0, 0, 0]\n  b: Tall @ [0, 0, 2]\n}\n";
        let error = parse_inline_graph(source).unwrap_err();
        let message = error.to_string();

        assert!(matches!(
            error,
            ModuleError::InvalidGeometry {
                source: ModuleCertificationError::Graph {
                    source: crate::BlockGraphError::ModuleBlockOverlap { position, .. },
                    ..
                },
                ..
            } if position == IVec3::new(0, 0, 2)
        ));
        assert!(
            message.contains("block footprints conflict at cell [0, 0, 2]"),
            "{message}"
        );
    }

    #[test]
    fn nested_geometry_error_names_the_invalid_child() {
        let source = "BLOG 1.0\n\nmodule main {\n  child: Child @ [0,0,0]\n}\n\nmodule Child {\n  a: Leaf @ [0,0,0]\n  b: Leaf @ [0,0,0]\n}\n\nmodule Leaf {\n  0: ZXZ [0,0,0]\n}\n";
        let error = parse_inline_graph(source).unwrap_err();

        assert!(matches!(
            &error,
            ModuleError::InvalidGeometry { module, .. } if module == "Child"
        ));
        let span = error.span().unwrap();
        assert!(source[span.start as usize..span.end as usize].starts_with("module Child"));
    }

    #[test]
    fn composite_geometry_checks_hidden_branch_arms() {
        let source = r#"BLOG 1.0

module Branched {
  0: ZXZ [0,0,0]
  9: Z [2,0,0]
  branch b {
    false {
      1: ZXZ [0,0,1]
      2: ZXZ [1,0,1]
      0 -> +Z
      1 -> +X
    }
    true {
      3: ZXZ [0,0,1]
      4: ZXZ [-1,0,1]
      0 -> +Z
      3 -> -X
    }
  }
  m = measure 9
  resolve b if m
}

module main {
  child: Branched @ [0,0,0]
  0: ZXZ [1,0,1]
}

"#;

        assert!(matches!(
            parse_inline_graph(source),
            Err(ModuleError::InvalidGeometry {
                source: ModuleCertificationError::Graph {
                    source: crate::BlockGraphError::ModuleBlockOverlap { position, .. },
                    ..
                },
                ..
            }) if position == IVec3::new(1, 0, 1)
        ));
    }

    #[test]
    fn direct_chains_keep_a_fixed_composition_frontier() {
        for len in [10, 100, 1_000] {
            let summary = parse_inline_graph(&direct_chain_source(len))
                .unwrap()
                .summarize_root(ModuleCertificationLimits {
                    max_frontier_width: 4,
                    ..ModuleCertificationLimits::UNLIMITED
                })
                .unwrap();
            assert_eq!(summary.boundary_rows().len(), 2);
            assert!(
                summary
                    .boundary_rows()
                    .all(|row| row.phase() == 0 && row.paulis.get(0) == row.paulis.get(1))
            );
        }
    }

    #[test]
    fn binding_from_child_let_is_emitted_after_its_definition() {
        let source = r#"BLOG 1.0

module Producer {
  out fire = t
  0: ZXZ [0, 0, 0]
  m = measure 0
  t = !m
}

module Consumer {
  in enable
  discard if enable
}

module main {
  p: Producer @ [0, 0, 0]
  c: Consumer @ [0, 0, 0]
  p.fire => c.enable
}
"#;
        let program = parse_inline_graph(source).unwrap();
        let linked = flatten_module_definition(&program, program.root(), "").unwrap();
        let actions = linked.graph.actions();
        assert!(matches!(&actions[0], Action::Measure { name, .. } if name == "p__m"));
        assert!(matches!(&actions[1], Action::Let { name, .. } if name == "p__t"));
        assert!(matches!(&actions[2], Action::Let { name, .. } if name == "c__enable"));
        assert!(matches!(&actions[3], Action::DiscardIf(_)));
        assert_eq!(linked.action_scopes, ["p", "p", "c", "c"]);
    }

    #[test]
    fn discharged_measurements_still_materialize() {
        let program = parse_inline_graph(&measured_chain_source(32)).unwrap();
        let summary = program
            .summarize_root(ModuleCertificationLimits {
                max_frontier_width: 6,
                ..ModuleCertificationLimits::UNLIMITED
            })
            .unwrap();
        let graph = program
            .materialize_root_graph()
            .unwrap()
            .fix_shadowed_faces();
        let zx = crate::ZXGraph::try_from(&graph).unwrap();
        let stabilizers = summary.materialize_stabilizers(&zx, IVec3::ZERO).unwrap();

        assert_eq!(
            stabilizers
                .generators
                .iter()
                .filter(|generator| generator.measurement_name().is_some())
                .count(),
            32
        );
    }

    #[test]
    fn independent_connectors_preserve_named_parent_ports_at_the_public_width_limit() {
        let source = format!(
            "BLOG 1.0\n{STAGE}\n{}",
            r#"module main {
  out o1: data = 3
  in i0: data = 0
  out o0: data = 1
  in i1: data = 2
  a: Stage @ [0,0,0]
  b: Stage @ [3,0,0]
  0: Port [0,0,-1] role=input
  1: Port [0,0,1] role=output
  2: Port [3,0,-1] role=input
  3: Port [3,0,1] role=output
  b.q_out -> 3
  0 -> a.q_in
  a.q_out -> 1
  2 -> b.q_in
}"#
        );
        let summary = parse_inline_graph(&source)
            .unwrap()
            .summarize_root(ModuleCertificationLimits {
                max_frontier_width: 4,
                ..ModuleCertificationLimits::UNLIMITED
            })
            .unwrap();
        let column = |name| {
            summary
                .quantum_ports()
                .iter()
                .position(|port| port.name == name)
                .unwrap()
        };
        let expected = [("i0", "o0"), ("i1", "o1")]
            .into_iter()
            .flat_map(|(input, output)| {
                [crate::Pauli::X, crate::Pauli::Z].map(|pauli| {
                    crate::PhasedPauliString::positive(crate::PauliString::from_terms(
                        4,
                        [(column(input), pauli), (column(output), pauli)],
                    ))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(summary.boundary_rows().len(), expected.len());
        assert!(
            expected
                .iter()
                .all(|row| summary.boundary_rows().any(|actual| actual == row))
        );
    }

    #[test]
    fn bit_input_is_an_external_action_variable() {
        let source = r#"BLOG 1.0

module main {
  in enabled
  0: ZXZ [0, 0, 0]
  1: ZXZ [0, 0, 1]
  0 -> +Z
  feedback X 0 if enabled
}

"#;
        let program = parse_inline_graph(source).unwrap();
        assert_eq!(
            program
                .root()
                .local_body()
                .action_graph()
                .inputs()
                .collect::<Vec<_>>(),
            ["enabled"]
        );
    }

    #[test]
    fn loader_resolves_relative_import_root() {
        let stage = format!(
            "BLOG 1.0\n\n{}",
            STAGE.replacen("module Stage", "module main", 1)
        );
        let root = pair_source("import \"stage.blog\" as Stage");
        let program = crate::load_graph_with_resolver_and_limits(
            "root.blog",
            |path| match path.to_str() {
                Some("root.blog") => Ok(root.clone()),
                Some("stage.blog") => Ok(stage.to_string()),
                _ => Err("missing"),
            },
            ModuleCertificationLimits::DEFAULT,
        )
        .unwrap();
        assert!(program.module("Stage").is_some());
        assert_eq!(program.root().name, "main");
    }

    #[test]
    fn loader_rejects_import_cycle() {
        let error = crate::load_graph_with_resolver_and_limits(
            "a.blog",
            |path| match path.to_str() {
                Some("a.blog") => {
                    Ok("BLOG 1.0\nimport \"b.blog\" as B\nmodule main {\n}\n".to_string())
                }
                Some("b.blog") => {
                    Ok("BLOG 1.0\nimport \"./a.blog\" as A\nmodule main {\n}\n".to_string())
                }
                _ => Err("missing"),
            },
            ModuleCertificationLimits::DEFAULT,
        )
        .unwrap_err();
        assert!(matches!(error, ModuleError::ImportCycle { .. }));
    }

    #[test]
    fn compact_hierarchy_is_bounded_before_geometry_expansion() {
        let make_modules = |body: &str, depth| {
            let leaf = crate::parse_blog_to_graph(body).unwrap();
            (0..=depth)
                .map(|level| {
                    BlockGraph::definition(
                        if level == depth {
                            "main".into()
                        } else {
                            format!("L{level}")
                        },
                        if level == 0 {
                            leaf.clone()
                        } else {
                            BlockGraph::default()
                        },
                        ModuleInterface::default(),
                        if level == 0 {
                            Vec::new()
                        } else {
                            ["a", "b"]
                                .map(|name| ModuleInstance {
                                    name: name.into(),
                                    definition: format!("L{}", level - 1),
                                    rotation: ModuleRotation::default(),
                                    translation: IVec3::ZERO,
                                })
                                .to_vec()
                        },
                        Vec::new(),
                        Vec::new(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (body, limits, phase, expected) in [
            (
                "BLOG 1.0\n0: ZXZ [0,0,0]\n",
                ModuleCertificationLimits {
                    max_expanded_blocks: 1_000_000,
                    ..ModuleCertificationLimits::UNLIMITED
                },
                "expanded blocks",
                1_048_576,
            ),
            (
                "BLOG 1.0\n0: ZXZ [0,0,0] height=4d\n",
                ModuleCertificationLimits {
                    max_occupied_cells: 4_000_000,
                    ..ModuleCertificationLimits::UNLIMITED
                },
                "occupied footprint cells",
                4_194_304,
            ),
            (
                "BLOG 1.0\n",
                ModuleCertificationLimits::DEFAULT,
                "expanded module instances",
                1_048_575,
            ),
        ] {
            let error = BlockGraph::from_definitions_with_limits(make_modules(body, 24), limits)
                .unwrap_err();
            assert!(matches!(error, ModuleError::InvalidGeometry {
                source: ModuleCertificationError::ResourceLimited { phase: actual, observed, .. }, ..
            } if actual == phase && observed == expected));
        }
        let overflow = BlockGraph::from_definitions_with_limits(
            make_modules("BLOG 1.0\n", 80),
            ModuleCertificationLimits::UNLIMITED,
        )
        .unwrap_err();
        assert!(matches!(
            overflow,
            ModuleError::InvalidGeometry {
                source: ModuleCertificationError::ResourceLimited {
                    phase: "expanded module instances",
                    observed: usize::MAX,
                    limit: usize::MAX,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn module_instances_must_be_acyclic() {
        let source = "BLOG 1.0\nmodule main {\n  b: B @ [0,0,0]\n}\n\
                      module B {\n  a: main @ [0,0,0]\n}\n";
        let error = parse_inline_graph(source).unwrap_err();
        assert!(matches!(
            error,
            ModuleError::InvalidModule { message, .. }
                if message == "module dependency cycle"
        ));
    }
}
