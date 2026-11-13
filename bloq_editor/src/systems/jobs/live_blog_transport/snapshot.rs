//! Lossless source snapshot wire fields. The core checked restore API, not this
//! deserializer, owns topology/branch/hierarchy validation and cache rebuilding.
use super::*;
use bloq_graph::{
    BitBinding, BitRef, BranchArm, BranchSnapshot, DefinitionSnapshot, InstancePort,
    ModuleInstance, ModuleRotation, QuantumConnection, SourceSnapshot, UDirection,
};

#[derive(Serialize, Deserialize, PartialEq)]
pub(super) struct SnapshotWire {
    root: LocalWire,
    definitions: Vec<LocalWire>,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct LocalWire {
    definition: DefinitionWire,
    declared: bool,
    bit_inputs: Vec<String>,
    instances: Vec<InstanceWire>,
    connections: Vec<ConnectionWire>,
    bindings: Vec<BindingWire>,
    geometry: GeometryWire,
    branches: Vec<BranchWire>,
    actions: Vec<ActionWire>,
    action_inputs: Vec<String>,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct InstanceWire {
    name: String,
    definition: String,
    axis: usize,
    turns: u8,
    translation: IVec3,
}
#[derive(Serialize, Deserialize, PartialEq)]
struct PortWire {
    instance: String,
    port: String,
}
#[derive(Serialize, Deserialize, PartialEq)]
enum ConnectionWire {
    Input {
        block: IVec3,
        input: PortWire,
        hadamard: bool,
    },
    Output {
        output: PortWire,
        block: IVec3,
        hadamard: bool,
    },
    Pipe {
        output: PortWire,
        input: PortWire,
        hadamard: bool,
    },
}
#[derive(Serialize, Deserialize, PartialEq)]
struct BindingWire {
    source_instance: Option<String>,
    source_bit: String,
    target_instance: String,
    target_bit: String,
}
#[derive(Serialize, Deserialize, PartialEq)]
struct BranchWire {
    name: String,
    target: IVec3,
    on_false: GeometryWire,
    on_true: GeometryWire,
    shown_true: bool,
}
#[derive(Serialize, Deserialize, PartialEq)]
struct GeometryWire {
    blocks: Vec<BlockWire>,
    pipes: Vec<PipeWire>,
}

impl From<InstancePort> for PortWire {
    fn from(port: InstancePort) -> Self {
        Self {
            instance: port.instance,
            port: port.port,
        }
    }
}
impl From<PortWire> for InstancePort {
    fn from(port: PortWire) -> Self {
        Self {
            instance: port.instance,
            port: port.port,
        }
    }
}
impl GeometryWire {
    fn encode(blocks: Vec<Block>, pipes: Vec<Pipe>) -> Self {
        Self {
            blocks: blocks.iter().map(BlockWire::encode).collect(),
            pipes: pipes.iter().map(PipeWire::encode).collect(),
        }
    }
    fn decode(self) -> Result<(Vec<Block>, Vec<Pipe>), String> {
        Ok((
            self.blocks
                .into_iter()
                .map(BlockWire::decode)
                .collect::<Result<_, _>>()?,
            self.pipes
                .into_iter()
                .map(PipeWire::decode)
                .collect::<Result<_, _>>()?,
        ))
    }
    fn arm(arm: BranchArm) -> Self {
        Self::encode(
            arm.blocks().cloned().collect(),
            arm.pipes().cloned().collect(),
        )
    }
    fn decode_arm(self) -> Result<BranchArm, String> {
        let (blocks, pipes) = self.decode()?;
        Ok(BranchArm::new(blocks, pipes))
    }
}

impl LocalWire {
    fn encode(local: DefinitionSnapshot) -> Self {
        let definition = DefinitionWire {
            name: local.name,
            quantum_ports: local
                .interface
                .quantum_ports
                .into_iter()
                .map(|p| QuantumPortWire {
                    name: p.name,
                    position: p.position,
                    input: p.direction == bloq_graph::PortDirection::Input,
                    resource: p.resource_type,
                })
                .collect(),
            bit_outputs: local
                .interface
                .bit_outputs
                .iter()
                .map(|o| (o.name.clone(), ExprWire::encode(&o.expr)))
                .collect(),
        };
        Self {
            definition,
            declared: local.interface_declared,
            bit_inputs: local.interface.bit_inputs,
            instances: local
                .instances
                .into_iter()
                .map(|i| InstanceWire {
                    name: i.name,
                    definition: i.definition,
                    axis: i.rotation.axis().index(),
                    turns: i.rotation.quarter_turns(),
                    translation: i.translation,
                })
                .collect(),
            connections: local
                .quantum_connections
                .into_iter()
                .map(|c| match c {
                    QuantumConnection::Input {
                        block,
                        input,
                        hadamard,
                    } => ConnectionWire::Input {
                        block,
                        input: input.into(),
                        hadamard,
                    },
                    QuantumConnection::Output {
                        output,
                        block,
                        hadamard,
                    } => ConnectionWire::Output {
                        output: output.into(),
                        block,
                        hadamard,
                    },
                    QuantumConnection::Pipe {
                        output,
                        input,
                        hadamard,
                    } => ConnectionWire::Pipe {
                        output: output.into(),
                        input: input.into(),
                        hadamard,
                    },
                })
                .collect(),
            bindings: local
                .bit_bindings
                .into_iter()
                .map(|b| BindingWire {
                    source_instance: b.source.instance,
                    source_bit: b.source.bit,
                    target_instance: b.target_instance,
                    target_bit: b.target_bit,
                })
                .collect(),
            geometry: GeometryWire::encode(local.blocks, local.pipes),
            branches: local
                .branches
                .into_iter()
                .map(|b| BranchWire {
                    name: b.name,
                    target: b.target,
                    on_false: GeometryWire::arm(b.on_false),
                    on_true: GeometryWire::arm(b.on_true),
                    shown_true: b.shown_true,
                })
                .collect(),
            actions: local.actions.iter().map(ActionWire::encode).collect(),
            action_inputs: local.action_inputs,
        }
    }
    fn decode(self) -> Result<DefinitionSnapshot, String> {
        let empty = self.definition.apply(BlockGraph::default());
        let (blocks, pipes) = self.geometry.decode()?;
        let mut interface = empty.interface;
        interface.bit_inputs = self.bit_inputs;
        Ok(DefinitionSnapshot {
            name: empty.name,
            interface_declared: self.declared,
            interface,
            instances: self
                .instances
                .into_iter()
                .map(|i| {
                    let axis = *[UDirection::X, UDirection::Y, UDirection::Z]
                        .get(i.axis)
                        .ok_or("invalid instance rotation axis")?;
                    if i.turns > 3 {
                        return Err("invalid instance rotation turns".into());
                    }
                    Ok(ModuleInstance {
                        name: i.name,
                        definition: i.definition,
                        rotation: ModuleRotation::new(axis, i32::from(i.turns)),
                        translation: i.translation,
                    })
                })
                .collect::<Result<_, String>>()?,
            quantum_connections: self
                .connections
                .into_iter()
                .map(|c| match c {
                    ConnectionWire::Input {
                        block,
                        input,
                        hadamard,
                    } => QuantumConnection::Input {
                        block,
                        input: input.into(),
                        hadamard,
                    },
                    ConnectionWire::Output {
                        output,
                        block,
                        hadamard,
                    } => QuantumConnection::Output {
                        output: output.into(),
                        block,
                        hadamard,
                    },
                    ConnectionWire::Pipe {
                        output,
                        input,
                        hadamard,
                    } => QuantumConnection::Pipe {
                        output: output.into(),
                        input: input.into(),
                        hadamard,
                    },
                })
                .collect(),
            bit_bindings: self
                .bindings
                .into_iter()
                .map(|b| BitBinding {
                    source: BitRef {
                        instance: b.source_instance,
                        bit: b.source_bit,
                    },
                    target_instance: b.target_instance,
                    target_bit: b.target_bit,
                })
                .collect(),
            blocks,
            pipes,
            branches: self
                .branches
                .into_iter()
                .map(|b| {
                    Ok(BranchSnapshot {
                        name: b.name,
                        target: b.target,
                        on_false: b.on_false.decode_arm()?,
                        on_true: b.on_true.decode_arm()?,
                        shown_true: b.shown_true,
                    })
                })
                .collect::<Result<_, String>>()?,
            actions: self
                .actions
                .into_iter()
                .map(ActionWire::decode)
                .collect::<Result<_, _>>()?,
            action_inputs: self.action_inputs,
        })
    }
}
impl SnapshotWire {
    pub(super) fn encode(snapshot: SourceSnapshot) -> Self {
        Self {
            root: LocalWire::encode(snapshot.root),
            definitions: snapshot
                .definitions
                .into_iter()
                .map(LocalWire::encode)
                .collect(),
        }
    }
    pub(super) fn decode(self) -> Result<SourceSnapshot, String> {
        Ok(SourceSnapshot {
            root: self.root.decode()?,
            definitions: self
                .definitions
                .into_iter()
                .map(LocalWire::decode)
                .collect::<Result<_, _>>()?,
        })
    }
}
