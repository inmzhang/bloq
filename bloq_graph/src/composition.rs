//! Validated module placement and docking shared with the editor.
use crate::{
    Basis, BitBinding, BitOutput, BitRef, Block, BlockGraph, BlockGraphError, BlockKind, Expr,
    InstancePort, ModuleCertificationError, ModuleCertificationLimits, ModuleError, ModuleInstance,
    ModuleRotation, PortDirection, PortRole, QuantumConnection, QuantumPort, checked_add_position,
    flatten_module_definition,
};
use glam::IVec3;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

fn invalid(message: impl Into<String>) -> BlockGraphError {
    source_error(ModuleError::InvalidModule {
        module: BlockGraph::ENTRY_MODULE.into(),
        message: message.into(),
        span: None,
    })
}
fn source_error(error: ModuleError) -> BlockGraphError {
    BlockGraphError::ModuleSource(Arc::new(error))
}
fn materialization(error: ModuleCertificationError) -> BlockGraphError {
    BlockGraphError::ModuleMaterialization(Arc::new(error))
}
macro_rules! fail { ($($arg:tt)*) => { return Err(invalid(format!($($arg)*))) }; }
macro_rules! ensure { ($condition:expr, $($arg:tt)*) => { if !$condition { fail!($($arg)*); } }; }

/// Docking preferences for module composition.
#[derive(Debug, Clone, Copy)]
pub struct ModuleJoinOptions {
    /// Align the input instance's connected group to the output port.
    pub align: bool,
    /// Prefer direct seams; retain connection cubes when compaction is obstructed.
    pub compact: bool,
    /// Apply an extra Hadamard at the explicitly selected seam.
    pub hadamard: bool,
}
impl Default for ModuleJoinOptions {
    fn default() -> Self {
        Self {
            align: true,
            compact: true,
            hadamard: false,
        }
    }
}

impl BlockGraph {
    /// Places a child beside the current geometry and exposes its public interface.
    /// Quantum ports, classical inputs, and classical outputs receive instance-qualified names.
    /// The graph is unchanged on failure.
    ///
    /// # Errors
    /// Returns a declaration, geometry, coordinate, or certification error.
    pub fn place_module(&mut self, name: &str, definition: &str) -> Result<IVec3, BlockGraphError> {
        self.place_module_with(name, definition, None, ModuleRotation::IDENTITY)
    }

    /// Places a child with optional explicit translation and a local rotation.
    /// When translation is absent, the child is placed beside the current geometry.
    ///
    /// # Errors
    /// Returns a declaration, geometry, coordinate, or certification error without changing the graph.
    pub fn place_module_with(
        &mut self,
        name: &str,
        definition: &str,
        translation: Option<IVec3>,
        rotation: ModuleRotation,
    ) -> Result<IVec3, BlockGraphError> {
        let (next, position) = place_module(self, name, definition, translation, rotation)?;
        *self = next;
        Ok(position)
    }

    /// Aligns and docks two named child ports, joining all other touching compatible ports.
    /// Names use `instance.port`. Connected input-side instances move together.
    /// Direct seams preserve the endpoint Hadamards; obstructed compaction retains connection cubes.
    ///
    /// # Errors
    /// Returns a declaration, geometry, coordinate, or certification error without changing the graph.
    pub fn connect_modules(&mut self, output: &str, input: &str) -> Result<usize, BlockGraphError> {
        self.connect_modules_with(output, input, ModuleJoinOptions::default())
    }

    /// Docks named child ports with explicit alignment, compaction, and Hadamard preferences.
    ///
    /// # Errors
    /// Returns a declaration, geometry, coordinate, or certification error without changing the graph.
    pub fn connect_modules_with(
        &mut self,
        output: &str,
        input: &str,
        options: ModuleJoinOptions,
    ) -> Result<usize, BlockGraphError> {
        self.validate_with_limits(ModuleCertificationLimits::DEFAULT)
            .map_err(source_error)?;
        let result = connect_ports(
            self,
            endpoint(output)?,
            endpoint(input)?,
            options.hadamard,
            options.align,
            options.compact,
        )?;
        let count = result.connections;
        *self = result.program;
        Ok(count)
    }

    /// Binds a parent value or child bit output to a child's classical input.
    /// Replaces its existing binding and removes an unused automatically exposed parent input.
    ///
    /// # Errors
    /// Returns a declaration, dependency, or certification error without changing the graph.
    pub fn bind_modules(&mut self, source: &str, target: &str) -> Result<(), BlockGraphError> {
        self.validate_with_limits(ModuleCertificationLimits::DEFAULT)
            .map_err(source_error)?;
        let target = endpoint(target)?;
        let source = if source.contains('.') {
            let source = endpoint(source)?;
            BitRef {
                instance: Some(source.instance),
                bit: source.port,
            }
        } else {
            BitRef {
                instance: None,
                bit: source.into(),
            }
        };
        let mut modules = self
            .modules()
            .map(BlockGraph::clone_local_definition)
            .collect::<Vec<_>>();
        let root = modules
            .iter_mut()
            .find(|m| m.name == Self::ENTRY_MODULE)
            .expect("validated program has main");
        let binding = root
            .bit_bindings
            .iter_mut()
            .find(|b| b.target_instance == target.instance && b.target_bit == target.port)
            .ok_or_else(|| invalid("unknown child classical input"))?;
        let old = std::mem::replace(&mut binding.source, source);
        prune_unused_input(root, &old)?;
        *self = BlockGraph::from_definitions(modules).map_err(source_error)?;
        Ok(())
    }

    /// Translates an instance's connected group and joins touching compatible free ports.
    /// Returns the applied offset, including any successful seam compaction.
    ///
    /// # Errors
    /// Returns a declaration, geometry, coordinate, or certification error without changing the graph.
    pub fn translate_module(
        &mut self,
        name: &str,
        offset: IVec3,
        compact: bool,
    ) -> Result<IVec3, BlockGraphError> {
        self.validate_with_limits(ModuleCertificationLimits::DEFAULT)
            .map_err(source_error)?;
        let result = translate_module(self, name, offset, compact)?;
        let offset = result.offset;
        *self = result.program;
        Ok(offset)
    }
}
fn endpoint(text: &str) -> Result<InstancePort, BlockGraphError> {
    let (instance, port) = text
        .split_once('.')
        .filter(|(instance, port)| !instance.is_empty() && !port.is_empty() && !port.contains('.'))
        .ok_or_else(|| invalid("expected a quantum port name 'instance.port'"))?;
    Ok(InstancePort {
        instance: instance.into(),
        port: port.into(),
    })
}

pub fn place_module(
    program: &BlockGraph,
    name: &str,
    definition: &str,
    translation: Option<IVec3>,
    rotation: ModuleRotation,
) -> Result<(BlockGraph, IVec3), BlockGraphError> {
    program
        .validate_with_limits(ModuleCertificationLimits::DEFAULT)
        .map_err(source_error)?;
    ensure!(
        definition != BlockGraph::ENTRY_MODULE,
        "A module cannot contain itself"
    );
    let mut modules = program
        .modules()
        .map(BlockGraph::clone_local_definition)
        .collect::<Vec<_>>();
    let child = program
        .module(definition)
        .ok_or_else(|| invalid("Module no longer exists"))?;
    let child_graph = flatten_module_definition(program, child, "")?.graph;
    ensure!(
        !child_graph.is_empty(),
        "Build some geometry before placing this module"
    );
    let translation = match translation {
        Some(position) => position,
        None => position_beside(
            &program.flatten().map_err(materialization)?,
            &child_graph,
            rotation,
        )?,
    };
    let root = modules
        .iter_mut()
        .find(|m| m.name == BlockGraph::ENTRY_MODULE)
        .expect("validated program has main");
    root.interface_declared = true;
    ensure!(
        !root.instances.iter().any(|i| i.name == name),
        "Instance '{name}' already exists"
    );
    let instance = ModuleInstance {
        name: name.into(),
        definition: definition.into(),
        translation,
        rotation,
    };
    for port in &child.interface.quantum_ports {
        expose_port(program, root, &instance, port)?;
    }
    for input in &child.interface.bit_inputs {
        let name = fresh_interface_name(root, &format!("{}_{}", instance.name, input));
        root.interface.bit_inputs.push(name.clone());
        root.bit_bindings.push(BitBinding {
            source: BitRef {
                instance: None,
                bit: name,
            },
            target_instance: instance.name.clone(),
            target_bit: input.clone(),
        });
    }
    for output in &child.interface.bit_outputs {
        root.interface.bit_outputs.push(BitOutput {
            name: fresh_interface_name(root, &format!("{}_{}", instance.name, output.name)),
            expr: Expr::Var(format!("{}.{}", instance.name, output.name)),
        });
    }
    root.instances.push(instance);

    Ok((
        BlockGraph::from_definitions(modules).map_err(source_error)?,
        translation,
    ))
}

#[derive(Debug)]
pub struct ModuleTranslation {
    pub program: BlockGraph,
    pub offset: IVec3,
    pub connections: usize,
    pub compacted: bool,
}

#[derive(Clone, Debug)]
pub struct ConnectionSeam {
    pub output: InstancePort,
    pub input: InstancePort,
    pub hadamard: bool,
    pub cube: Option<IVec3>,
}

/// Recognize ordinary direct seams and isolated parent cubes joining two ports.
/// Authored junctions and local pipes are deliberately left to the definition editor.
pub fn connection_seams(root: &BlockGraph) -> Vec<ConnectionSeam> {
    let mut seams = Vec::new();
    for connection in &root.quantum_connections {
        match connection {
            QuantumConnection::Pipe {
                output,
                input,
                hadamard,
            } => seams.push(ConnectionSeam {
                output: output.clone(),
                input: input.clone(),
                hadamard: *hadamard,
                cube: None,
            }),
            QuantumConnection::Output {
                output,
                block,
                hadamard,
            } => {
                if !root
                    .get_block(*block)
                    .is_some_and(|block| block.kind().is_cube())
                    || !root.neighbor_positions(*block).is_empty()
                    || root.shown_branch_at(*block).is_some()
                {
                    continue;
                }
                let bindings = root
                    .quantum_connections
                    .iter()
                    .filter(|connection| bound_block(connection) == Some(*block))
                    .collect::<Vec<_>>();
                if bindings.len() == 2
                    && let Some((input, input_h)) =
                        bindings.iter().find_map(|binding| match binding {
                            QuantumConnection::Input {
                                input, hadamard, ..
                            } => Some((input, *hadamard)),
                            _ => None,
                        })
                {
                    seams.push(ConnectionSeam {
                        output: output.clone(),
                        input: input.clone(),
                        hadamard: *hadamard ^ input_h,
                        cube: Some(*block),
                    });
                }
            }
            _ => {}
        }
    }
    seams
}

pub fn connected_instances(root: &BlockGraph, start: &str) -> BTreeSet<String> {
    let seams = connection_seams(root);
    let mut group = BTreeSet::from([start.to_owned()]);
    let mut pending = vec![start.to_owned()];
    while let Some(name) = pending.pop() {
        for seam in &seams {
            let other = if seam.output.instance == name {
                &seam.input.instance
            } else if seam.input.instance == name {
                &seam.output.instance
            } else {
                continue;
            };
            if group.insert(other.clone()) {
                pending.push(other.clone());
            }
        }
    }
    group
}

struct OpenPort {
    endpoint: InstancePort,
    port: QuantumPort,
    neighbor: IVec3,
    bases: [Option<Basis>; 3],
    hadamard: bool,
}

struct PortJoin<'a> {
    output: &'a OpenPort,
    input: &'a OpenPort,
    position: IVec3,
    compact_step: Option<IVec3>,
    hadamard: bool,
}

impl PortJoin<'_> {
    fn cube_kind(&self) -> Result<BlockKind, BlockGraphError> {
        for kind in BlockKind::all_kinds() {
            let BlockKind::Cube(cube) = kind else {
                continue;
            };
            let matches = [(self.output, false), (self.input, self.hadamard)]
                .into_iter()
                .all(|(port, flip)| {
                    port.bases
                        .iter()
                        .zip(cube.bases())
                        .all(|(face, cube_face)| {
                            face.is_none_or(
                                |face| if flip { face.flip() } else { face } == cube_face,
                            )
                        })
                });
            if matches {
                return Ok(kind);
            }
        }
        fail!("Port faces do not match any connection cube; rotate the module or choose Hadamard")
    }
}

fn open_ports(program: &BlockGraph) -> Result<Vec<OpenPort>, BlockGraphError> {
    let graph = program.flatten().map_err(materialization)?;
    let mut ports = Vec::new();
    for (endpoint, public) in exposed_ports(program) {
        let position = public.position;
        let neighbors = graph.neighbor_positions(position);
        let [neighbor] = neighbors.as_slice() else {
            fail!("Port needs one incident pipe")
        };
        let pipe = graph
            .get_pipe(position, *neighbor)
            .expect("the neighbor came from this position's incident pipes");
        ports.push(OpenPort {
            endpoint,
            port: public.clone(),
            neighbor: *neighbor,
            bases: graph.infer_pipe_basis_from_endpoint(pipe, position),
            hadamard: pipe.is_hadamard(),
        });
    }
    Ok(ports)
}

pub fn translate_module(
    program: &BlockGraph,
    name: &str,
    offset: IVec3,
    compact: bool,
) -> Result<ModuleTranslation, BlockGraphError> {
    translate_and_connect(program, name, offset, compact, None)
}

pub fn connect_ports(
    program: &BlockGraph,
    output: InstancePort,
    input: InstancePort,
    hadamard: bool,
    align: bool,
    compact: bool,
) -> Result<ModuleTranslation, BlockGraphError> {
    ensure!(
        output.instance != input.instance,
        "Choose ports on different instances"
    );
    let output_port = exposed_port(program.root(), &output)
        .ok_or_else(|| invalid("Output is already connected"))?;
    let input_port = exposed_port(program.root(), &input)
        .ok_or_else(|| invalid("Input is already connected"))?;
    ensure!(
        output_port.direction == PortDirection::Output
            && input_port.direction == PortDirection::Input,
        "Connect an output to an input"
    );
    let offset = if align {
        checked_difference(output_port.position, input_port.position)?
    } else {
        IVec3::ZERO
    };
    let name = input.instance.clone();
    translate_and_connect(
        program,
        &name,
        offset,
        compact,
        Some((output, input, hadamard)),
    )
}

fn translate_and_connect(
    program: &BlockGraph,
    name: &str,
    offset: IVec3,
    compact: bool,
    requested: Option<(InstancePort, InstancePort, bool)>,
) -> Result<ModuleTranslation, BlockGraphError> {
    ensure!(
        program
            .root()
            .instances
            .iter()
            .any(|instance| instance.name == name),
        "Instance no longer exists"
    );
    let group = connected_instances(program.root(), name);
    let ports = open_ports(program)?;
    let mut pairs = Vec::new();
    for moving in ports
        .iter()
        .filter(|port| group.contains(&port.endpoint.instance))
    {
        let position = checked_add_position(moving.port.position, offset)?;
        for fixed in ports
            .iter()
            .filter(|port| !group.contains(&port.endpoint.instance))
        {
            if position != fixed.port.position {
                continue;
            }
            ensure!(
                moving.port.direction != fixed.port.direction,
                "Ports '{}.{}' and '{}.{}' must join an output to an input",
                moving.endpoint.instance,
                moving.endpoint.port,
                fixed.endpoint.instance,
                fixed.endpoint.port
            );
            ensure!(
                moving.port.resource_type == fixed.port.resource_type,
                "Touching ports have different resource types"
            );
            let (output, input) = if moving.port.direction == PortDirection::Output {
                (moving, fixed)
            } else {
                (fixed, moving)
            };
            let hadamard = requested
                .as_ref()
                .is_some_and(|(a, b, h)| *h && a == &output.endpoint && b == &input.endpoint);
            let step = checked_difference(moving.port.position, moving.neighbor)?;
            pairs.push(PortJoin {
                output,
                input,
                hadamard,
                position: fixed.port.position,
                compact_step: (fixed.neighbor - fixed.port.position == step).then_some(step),
            });
        }
    }
    if let Some((output, input, _)) = requested {
        ensure!(
            pairs
                .iter()
                .any(|join| join.output.endpoint == output && join.input.endpoint == input),
            "Ports do not meet; enable Align input instance or move them together first"
        );
    }
    let build = |delta: IVec3, compacted: bool| -> Result<BlockGraph, BlockGraphError> {
        let mut modules = program
            .modules()
            .map(BlockGraph::clone_local_definition)
            .collect::<Vec<_>>();
        let root = modules
            .iter_mut()
            .find(|module| module.name == BlockGraph::ENTRY_MODULE)
            .expect("validated program contains the entry module");
        for join in &pairs {
            consume_exposed_port(root, &join.output.endpoint)?;
            consume_exposed_port(root, &join.input.endpoint)?;
        }
        translate_group(root, &group, delta)?;
        for join in &pairs {
            let PortJoin {
                output,
                input,
                hadamard,
                ..
            } = join;
            if compacted {
                root.quantum_connections.push(QuantumConnection::Pipe {
                    output: output.endpoint.clone(),
                    input: input.endpoint.clone(),
                    hadamard: output.hadamard ^ input.hadamard ^ *hadamard,
                });
            } else {
                let block = join.position;
                root.try_add_block(Block::new(block, join.cube_kind()?))?;
                root.quantum_connections.extend([
                    QuantumConnection::Output {
                        output: output.endpoint.clone(),
                        block,
                        hadamard: false,
                    },
                    QuantumConnection::Input {
                        block,
                        input: input.endpoint.clone(),
                        hadamard: *hadamard,
                    },
                ]);
            }
        }
        ensure!(
            root.actions() == program.root().local_body().actions()
                && root.branch_definitions() == program.root().local_body().branch_definitions(),
            "Move would change parent actions or branches; edit the owning definition instead"
        );
        BlockGraph::from_definitions(modules).map_err(source_error)
    };
    if compact
        && let Some(step) = pairs.first().and_then(|join| join.compact_step)
        && pairs.iter().all(|join| join.compact_step == Some(step))
        && let Ok(compact_offset) = checked_add_position(offset, step)
    {
        match build(compact_offset, true) {
            Ok(program) => {
                return Ok(ModuleTranslation {
                    program,
                    offset: compact_offset,
                    connections: pairs.len(),
                    compacted: true,
                });
            }
            Err(error) => {
                let mut cause: Option<&dyn std::error::Error> = Some(&error);
                while let Some(current) = cause {
                    if matches!(
                        current.downcast_ref::<ModuleCertificationError>(),
                        Some(ModuleCertificationError::ResourceLimited { .. })
                    ) {
                        return Err(error);
                    }
                    cause = current.source();
                }
            }
        }
    }
    Ok(ModuleTranslation {
        program: build(offset, false)?,
        offset,
        connections: pairs.len(),
        compacted: false,
    })
}

/// Move a connected group and its free public ports/shared connector cubes in
/// one transaction, so old positions cannot collide with each other mid-move.
///
/// The graph is unchanged if a shifted coordinate or placement is invalid.
pub fn translate_group(
    graph: &mut BlockGraph,
    group: &BTreeSet<String>,
    offset: IVec3,
) -> Result<(), BlockGraphError> {
    let mut root = graph.clone();
    let mut positions = connection_seams(&root)
        .iter()
        .filter(|seam| {
            group.contains(&seam.output.instance) && group.contains(&seam.input.instance)
        })
        .filter_map(|seam| seam.cube)
        .collect::<HashSet<_>>();
    for connection in &root.quantum_connections {
        let endpoint = match connection {
            QuantumConnection::Input {
                input: endpoint, ..
            }
            | QuantumConnection::Output {
                output: endpoint, ..
            } => endpoint,
            _ => continue,
        };
        if group.contains(&endpoint.instance)
            && let Some(port) = exposed_port(&root, endpoint)
        {
            positions.insert(port.position);
        }
    }
    let mut blocks = positions
        .iter()
        .map(|position| {
            root.get_block(*position)
                .expect("validated interface ports and binds sit on body blocks")
                .clone()
        })
        .collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.pos().to_array());
    for position in &positions {
        root.remove_block(*position);
    }
    for block in blocks {
        root.try_add_block(block.try_with_shift(offset)?)?;
    }
    for port in &mut root.interface.quantum_ports {
        if positions.contains(&port.position) {
            port.position = checked_add_position(port.position, offset)?;
        }
    }
    for connection in &mut root.quantum_connections {
        if let QuantumConnection::Input { block, .. } | QuantumConnection::Output { block, .. } =
            connection
            && positions.contains(block)
        {
            *block = checked_add_position(*block, offset)?;
        }
    }
    for instance in &mut root.instances {
        if group.contains(&instance.name) {
            instance.translation = checked_add_position(instance.translation, offset)?;
        }
    }
    *graph = root;
    Ok(())
}

pub fn unique_name(base: &str, used: &HashSet<String>) -> String {
    if !used.contains(base) {
        return base.to_owned();
    }
    (2..)
        .map(|i| format!("{base}_{i}"))
        .find(|name| !used.contains(name))
        .expect("an unbounded counter always yields a name outside a finite set")
}

pub fn fresh_interface_name(module: &BlockGraph, base: &str) -> String {
    let used = module
        .interface
        .quantum_ports
        .iter()
        .map(|p| p.name.clone())
        .chain(module.interface.bit_inputs.iter().cloned())
        .chain(module.interface.bit_outputs.iter().map(|p| p.name.clone()))
        .collect();
    unique_name(base, &used)
}

pub fn expose_port(
    program: &BlockGraph,
    root: &mut BlockGraph,
    instance: &ModuleInstance,
    port: &QuantumPort,
) -> Result<(), BlockGraphError> {
    let mut public = port.clone();
    public.position = instance.try_transform_position(port.position)?;
    public.name = fresh_interface_name(root, &format!("{}_{}", instance.name, port.name));
    let prototype = program
        .module(&instance.definition)
        .expect("validated program resolves every instance definition")
        .get_block(port.position)
        .expect("validated interface port sits on a body block");
    let role = if prototype.port_role() == Some(PortRole::Multiplex) {
        PortRole::Multiplex
    } else {
        match port.direction {
            PortDirection::Input => PortRole::Input,
            PortDirection::Output => PortRole::Output,
        }
    };
    root.try_add_block(
        prototype
            .try_with_shift(checked_difference(public.position, port.position)?)?
            .with_port_role(role)?
            .with_tag(public.name.clone())?,
    )?;
    let endpoint = InstancePort {
        instance: instance.name.clone(),
        port: port.name.clone(),
    };
    root.quantum_connections.push(match port.direction {
        PortDirection::Input => QuantumConnection::Input {
            block: public.position,
            input: endpoint,
            hadamard: false,
        },
        PortDirection::Output => QuantumConnection::Output {
            block: public.position,
            output: endpoint,
            hadamard: false,
        },
    });
    root.interface.quantum_ports.push(public);
    Ok(())
}

pub fn exposed_port<'a>(root: &'a BlockGraph, endpoint: &InstancePort) -> Option<&'a QuantumPort> {
    let position = root
        .quantum_connections
        .iter()
        .find_map(|connection| match connection {
            QuantumConnection::Input {
                block,
                input: port,
                hadamard: false,
            }
            | QuantumConnection::Output {
                block,
                output: port,
                hadamard: false,
            } if port == endpoint => Some(*block),
            _ => None,
        })?;
    (root.neighbor_positions(position).is_empty()
        && root
            .quantum_connections
            .iter()
            .filter(|c| bound_block(c) == Some(position))
            .count()
            == 1)
        .then(|| {
            root.interface
                .quantum_ports
                .iter()
                .find(|p| p.position == position)
        })
        .flatten()
}

/// Free instance ports in declaration order, paired with their public interface.
pub fn exposed_ports(program: &BlockGraph) -> impl Iterator<Item = (InstancePort, &QuantumPort)> {
    program.root().instances.iter().flat_map(move |instance| {
        program
            .module(&instance.definition)
            .expect("validated program resolves every instance definition")
            .interface
            .quantum_ports
            .iter()
            .filter_map(move |port| {
                let endpoint = InstancePort {
                    instance: instance.name.clone(),
                    port: port.name.clone(),
                };
                exposed_port(program.root(), &endpoint).map(|port| (endpoint, port))
            })
    })
}

fn consume_exposed_port(
    root: &mut BlockGraph,
    endpoint: &InstancePort,
) -> Result<(), BlockGraphError> {
    let position = exposed_port(root, endpoint)
        .ok_or_else(|| invalid("Port is already connected or bound to local geometry"))?
        .position;
    root.quantum_connections
        .retain(|c| bound_block(c) != Some(position));
    remove_public_port(root, position);
    Ok(())
}

pub fn remove_public_port(root: &mut BlockGraph, position: IVec3) {
    if root
        .interface
        .quantum_ports
        .iter()
        .any(|p| p.position == position)
    {
        root.interface
            .quantum_ports
            .retain(|p| p.position != position);
        root.remove_block(position);
    }
}

pub fn bound_block(connection: &QuantumConnection) -> Option<IVec3> {
    match connection {
        QuantumConnection::Input { block, .. } | QuantumConnection::Output { block, .. } => {
            Some(*block)
        }
        QuantumConnection::Pipe { .. } => None,
    }
}

fn checked_difference(a: IVec3, b: IVec3) -> Result<IVec3, BlockGraphError> {
    let error = || BlockGraphError::CoordinateOverflow {
        position: a,
        offset: b,
    };
    Ok(IVec3::new(
        a.x.checked_sub(b.x).ok_or_else(error)?,
        a.y.checked_sub(b.y).ok_or_else(error)?,
        a.z.checked_sub(b.z).ok_or_else(error)?,
    ))
}

fn position_beside(
    graph: &BlockGraph,
    child: &BlockGraph,
    rotation: ModuleRotation,
) -> Result<IVec3, BlockGraphError> {
    let Some((x, _, _)) = graph.spans() else {
        return Ok(IVec3::ZERO);
    };
    let (child_x, child_y, child_z) = child
        .spans()
        .ok_or_else(|| invalid("Module has no geometry"))?;
    let mut min_x = i32::MAX;
    for cx in [*child_x.start(), *child_x.end()] {
        for cy in [*child_y.start(), *child_y.end()] {
            for cz in [*child_z.start(), *child_z.end()] {
                min_x = min_x.min(rotation.try_rotate_position(IVec3::new(cx, cy, cz))?.x);
            }
        }
    }
    let x = x
        .end()
        .checked_sub(min_x)
        .and_then(|x| x.checked_add(3))
        .ok_or_else(|| invalid("No coordinate space beside this composition"))?;
    Ok(IVec3::new(x, 0, 0))
}

pub fn prune_unused_input(root: &mut BlockGraph, source: &BitRef) -> Result<(), BlockGraphError> {
    if source.instance.is_none()
        && !root.bit_bindings.iter().any(|b| b.source == *source)
        && !root
            .actions()
            .iter()
            .any(|action| action.referenced_names().contains(source.bit.as_str()))
        && !root
            .interface
            .bit_outputs
            .iter()
            .any(|output| output.expr.names().any(|name| name == source.bit))
    {
        let inputs = root
            .action_graph()
            .inputs()
            .filter(|name| *name != source.bit)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        root.set_actions_deferred_with_inputs(root.actions(), inputs)?;
        root.interface.bit_inputs.retain(|name| name != &source.bit);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GalleryItem, UDirection};

    fn cnot_program() -> BlockGraph {
        let mut definition = GalleryItem::CNOT.build();
        definition.name = "CNOT".into();
        BlockGraph::from_definitions(vec![definition, BlockGraph::new()]).unwrap()
    }

    #[test]
    fn placement_docking_and_group_translation_preserve_the_library() {
        let mut graph = cnot_program();
        let original = graph.module("CNOT").unwrap().to_blog_text();
        for index in 0..3 {
            graph
                .place_module(&format!("child{index}"), "CNOT")
                .unwrap();
        }
        assert_eq!(
            graph
                .connect_modules("child0.control_out", "child1.control_in")
                .unwrap(),
            2
        );
        assert_eq!(
            graph
                .connect_modules("child1.control_out", "child2.control_in")
                .unwrap(),
            2
        );
        assert_eq!(graph.interface.quantum_ports.len(), 4);
        assert_eq!(
            graph
                .instances
                .iter()
                .map(|i| i.translation)
                .collect::<Vec<_>>(),
            [IVec3::ZERO, IVec3::new(0, 0, 2), IVec3::new(0, 0, 4)]
        );
        assert_eq!(graph.module("CNOT").unwrap().to_blog_text(), original);
        assert_eq!(
            graph
                .translate_module("child0", IVec3::Y * 3, true)
                .unwrap(),
            IVec3::Y * 3
        );
        assert!(graph.instances.iter().all(|i| i.translation.y == 3));
        assert_eq!(graph.module("CNOT").unwrap().to_blog_text(), original);
        let restored = BlockGraph::from_text(&graph.to_blog_text()).unwrap();
        assert_eq!(restored.interface, graph.interface);
        assert_eq!(restored.instances, graph.instances);
        assert_eq!(restored.quantum_connections, graph.quantum_connections);
        for block in graph.blocks() {
            assert_eq!(restored.get_block(block.pos()), Some(block));
        }
    }

    #[test]
    fn failed_group_translation_preserves_the_graph() {
        let mut graph = cnot_program();
        graph.place_module("child", "CNOT").unwrap();
        graph.add_block(Block::new(IVec3::X * 10, BlockKind::Port));
        let before = graph.to_blog_text();
        let group = BTreeSet::from(["child".to_owned()]);
        for offset in [IVec3::splat(i32::MAX), IVec3::X * 10] {
            translate_group(&mut graph, &group, offset)
                .expect_err("shifted boundary coordinates overflow or collide");
            assert_eq!(graph.to_blog_text(), before);
        }
    }

    #[test]
    fn failed_edits_are_atomic_and_rotated_placement_stays_beside_the_graph() {
        let mut graph = cnot_program();
        graph.place_module("child0", "CNOT").unwrap();
        graph
            .place_module_with(
                "child1",
                "CNOT",
                None,
                ModuleRotation::new(UDirection::Z, 2),
            )
            .unwrap();
        let before = graph.to_blog_text();
        for (a, b) in [
            ("child0.control_in", "child1.control_in"),
            ("child0.control_out", "missing.control_in"),
            ("child0", "child1.control_in"),
        ] {
            graph
                .connect_modules(a, b)
                .expect_err("invalid endpoints must fail");
            assert_eq!(graph.to_blog_text(), before);
        }
        graph
            .place_module("child0", "CNOT")
            .expect_err("duplicate instance must fail");
        graph
            .place_module("new", "missing")
            .expect_err("missing definition must fail");
        graph
            .place_module_with(
                "new",
                "CNOT",
                Some(IVec3::splat(i32::MAX)),
                ModuleRotation::IDENTITY,
            )
            .expect_err("coordinate overflow must fail");
        assert_eq!(graph.to_blog_text(), before);
    }
}
