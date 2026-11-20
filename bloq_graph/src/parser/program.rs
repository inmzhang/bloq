//! Lowering parsed module declarations into a hierarchical block graph.

use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};

use crate::parser::ast::{
    ActionStmt, ConnectEndpointAst, ConnectStmt, DataStmt, Expr as AstExpr, InterfaceStmt,
    ModularSourceFile, ModuleDef, PortDirectionAst, SourceFile, Span,
};
use crate::program::{
    BitBinding, BitOutput, BitRef, InstancePort, ModuleCertificationLimits, ModuleError,
    ModuleInstance, ModuleInterface, ModuleRotation, PortDirection, QuantumConnection, QuantumPort,
    split_member,
};
use crate::{BlockGraph, BlockGraphError, ParseError};

use super::{module_source_error, read_blog};

pub(super) fn lower(
    source: &ModularSourceFile,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, ModuleError> {
    lower_impl(source, false, limits)
}

pub(super) fn lower_deferred(source: &ModularSourceFile) -> Result<BlockGraph, ModuleError> {
    lower_impl(source, true, ModuleCertificationLimits::DEFAULT)
}

fn lower_impl(
    source: &ModularSourceFile,
    deferred: bool,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, ModuleError> {
    if !source.imports.is_empty() {
        return Err(ModuleError::ImportsRequireLoader);
    }
    let modules = lower_modules(source, deferred, limits)?;
    with_module_span(
        BlockGraph::from_definitions_with_limits(modules, limits),
        source,
    )
}

fn lower_modules(
    source: &ModularSourceFile,
    deferred: bool,
    limits: ModuleCertificationLimits,
) -> Result<Vec<BlockGraph>, ModuleError> {
    check_version(source)?;
    source
        .modules
        .iter()
        .map(|module| lower_module(source, &module.node, deferred, limits))
        .collect()
}

pub(super) fn load(
    root: &Path,
    source: ModularSourceFile,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, BlockGraphError> {
    load_inner(
        root,
        Some(source),
        &mut Vec::new(),
        &mut HashMap::new(),
        limits,
    )
}

fn load_inner(
    path: &Path,
    root_source: Option<ModularSourceFile>,
    stack: &mut Vec<PathBuf>,
    cache: &mut HashMap<PathBuf, BlockGraph>,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, BlockGraphError> {
    let path = normalize_path(path);
    if stack.contains(&path) {
        return Err(module_source_error(ModuleError::ImportCycle {
            path: path.display().to_string(),
        }));
    }
    if let Some(program) = cache.get(&path) {
        return Ok(program.clone());
    }
    stack.push(path.clone());
    let parsed = match root_source {
        Some(source) => source,
        None => super::parse::parse_program(&read_blog(&path)?)
            .map_err(|error| module_source_error(ModuleError::Parse(error)))?,
    };
    let mut modules = lower_modules(&parsed, false, limits).map_err(module_source_error)?;
    for import in &parsed.imports {
        let import_path = Path::new(&import.node.path.node);
        if import_path.as_os_str().is_empty() || import_path.is_absolute() {
            return Err(module_source_error(ModuleError::InvalidModule {
                module: "main".to_string(),
                message: format!("import path '{}' must be relative", import.node.path.node),
                span: Some(import.span),
            }));
        }
        let child_path = path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(import_path);
        let child = load_inner(&child_path, None, stack, cache, limits)?;
        modules.extend(prefix_modules(&child, &import.node.alias.node));
    }
    let program = with_module_span(
        BlockGraph::from_definitions_with_limits(modules, limits),
        &parsed,
    )
    .map_err(module_source_error)?;
    stack.pop();
    cache.insert(path, program.clone());
    Ok(program)
}

fn with_module_span(
    result: Result<BlockGraph, ModuleError>,
    source: &ModularSourceFile,
) -> Result<BlockGraph, ModuleError> {
    result.map_err(|error| {
        source.modules.iter().fold(error, |error, module| {
            error.with_module_span(&module.node.name.node, module.span)
        })
    })
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) =>
            {
                normalized.pop();
            }
            Component::ParentDir if normalized.has_root() => {}
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn prefix_modules(program: &BlockGraph, alias: &str) -> Vec<BlockGraph> {
    let root = &program.root().name;
    let names = program
        .modules()
        .map(|module| {
            let prefixed = if module.name == *root {
                alias.to_string()
            } else {
                format!("{alias}__{}", module.name)
            };
            (module.name.clone(), prefixed)
        })
        .collect::<HashMap<_, _>>();
    program
        .modules()
        .map(BlockGraph::clone_local_definition)
        .map(|mut module| {
            module.name = names[&module.name].clone();
            for instance in &mut module.instances {
                instance.definition = names[&instance.definition].clone();
            }
            module
        })
        .collect()
}

fn check_version(source: &ModularSourceFile) -> Result<(), ParseError> {
    let (major, minor) = source.version.node;
    if (major, minor) == (1, 0) {
        Ok(())
    } else {
        Err(ParseError::UnsupportedVersion {
            major,
            minor,
            span: source.version.span,
        })
    }
}

fn lower_module(
    source: &ModularSourceFile,
    module: &ModuleDef,
    deferred: bool,
    limits: ModuleCertificationLimits,
) -> Result<BlockGraph, ModuleError> {
    let mut bit_inputs = Vec::new();
    for statement in &module.interface_stmts {
        if let InterfaceStmt::BitInput(name) = &statement.node {
            bit_inputs.push(name.node.clone());
        }
    }
    let mut external_names = bit_inputs.iter().cloned().collect::<BTreeSet<_>>();
    for action in &module.action_stmts {
        collect_action_expr_names(&action.node, &mut external_names);
    }
    external_names.retain(|name| name.contains('.'));
    external_names.extend(bit_inputs.iter().cloned());

    let body_source = SourceFile {
        version: source.version.clone(),
        data_stmts: module.data_stmts.clone(),
        action_stmts: module.action_stmts.clone(),
    };
    let body = if deferred {
        super::lower::lower_deferred_with_inputs(&body_source, external_names)?
    } else {
        super::lower::lower_with_inputs_and_limits(&body_source, external_names, limits)?
    };
    let ids = block_positions(module)?;
    let interface = lower_interface(module, &ids)?;
    let instances = module
        .instance_stmts
        .iter()
        .map(|instance| {
            let rotation = match &instance.node.rotation {
                Some(rotation) => ModuleRotation::from_degrees(rotation.node.0, rotation.node.1)
                    .ok_or_else(|| ModuleError::InvalidModule {
                        module: module.name.node.clone(),
                        message: format!(
                            "instance '{}' rotation must be a multiple of 90 degrees",
                            instance.node.name.node
                        ),
                        span: Some(rotation.span),
                    })?,
                None => ModuleRotation::IDENTITY,
            };
            Ok(ModuleInstance {
                name: instance.node.name.node.clone(),
                definition: instance.node.definition.node.clone(),
                rotation,
                translation: instance.node.translation.node,
            })
        })
        .collect::<Result<_, ModuleError>>()?;
    let (quantum_connections, bit_bindings) = lower_connections(module, &ids)?;

    Ok(BlockGraph::definition(
        module.name.node.clone(),
        body,
        interface,
        instances,
        quantum_connections,
        bit_bindings,
    ))
}

fn block_positions(module: &ModuleDef) -> Result<HashMap<u32, glam::IVec3>, ParseError> {
    let mut ids = HashMap::new();
    for statement in &module.data_stmts {
        if let DataStmt::Block(block) = &statement.node
            && ids.insert(block.id.node, block.pos.node).is_some()
        {
            return Err(ParseError::DuplicateId {
                id: block.id.node,
                span: block.id.span,
            });
        }
    }
    Ok(ids)
}

fn lower_interface(
    module: &ModuleDef,
    ids: &HashMap<u32, glam::IVec3>,
) -> Result<ModuleInterface, ParseError> {
    let mut interface = ModuleInterface::default();
    for statement in &module.interface_stmts {
        match &statement.node {
            InterfaceStmt::Quantum(port) => {
                let position =
                    ids.get(&port.block_id.node)
                        .copied()
                        .ok_or(ParseError::UndefinedId {
                            id: port.block_id.node,
                            span: port.block_id.span,
                        })?;
                interface.quantum_ports.push(QuantumPort {
                    name: port.name.node.clone(),
                    position,
                    direction: match port.direction.node {
                        PortDirectionAst::Input => PortDirection::Input,
                        PortDirectionAst::Output => PortDirection::Output,
                    },
                    resource_type: port.resource_type.node.clone(),
                });
            }
            InterfaceStmt::BitInput(name) => interface.bit_inputs.push(name.node.clone()),
            InterfaceStmt::BitOutput(output) => interface.bit_outputs.push(BitOutput {
                name: output.name.node.clone(),
                expr: super::lower::lower_expr(&output.expr.node),
            }),
        }
    }
    Ok(interface)
}

fn lower_connections(
    module: &ModuleDef,
    ids: &HashMap<u32, glam::IVec3>,
) -> Result<(Vec<QuantumConnection>, Vec<BitBinding>), ParseError> {
    let mut quantum = Vec::new();
    let mut bits = Vec::new();
    for statement in &module.connect_stmts {
        match &statement.node {
            ConnectStmt::Pipe {
                hadamard,
                output,
                input,
            } => quantum.push(QuantumConnection::Pipe {
                output: instance_port(&output.node, output.span)?,
                input: instance_port(&input.node, input.span)?,
                hadamard: *hadamard,
            }),
            ConnectStmt::Bind {
                hadamard,
                source,
                target,
            } => match (&source.node, &target.node) {
                (ConnectEndpointAst::Block(id), ConnectEndpointAst::Name(name)) => {
                    quantum.push(QuantumConnection::Input {
                        block: position(*id, source.span, ids)?,
                        input: instance_port(name, target.span)?,
                        hadamard: *hadamard,
                    });
                }
                (ConnectEndpointAst::Name(name), ConnectEndpointAst::Block(id)) => {
                    quantum.push(QuantumConnection::Output {
                        output: instance_port(name, source.span)?,
                        block: position(*id, target.span, ids)?,
                        hadamard: *hadamard,
                    });
                }
                (ConnectEndpointAst::Name(source_name), ConnectEndpointAst::Name(target_name)) => {
                    let (target_instance, target_bit) =
                        split_member(target_name).ok_or_else(|| {
                            syntax(target.span, "bit binding target must be instance.bit")
                        })?;
                    let source = match split_member(source_name) {
                        Some((instance, bit)) => BitRef {
                            instance: Some(instance.to_string()),
                            bit: bit.to_string(),
                        },
                        None => BitRef {
                            instance: None,
                            bit: source_name.clone(),
                        },
                    };
                    bits.push(BitBinding {
                        source,
                        target_instance: target_instance.to_string(),
                        target_bit: target_bit.to_string(),
                    });
                }
                (ConnectEndpointAst::Block(_), ConnectEndpointAst::Block(_)) => {
                    return Err(syntax(statement.span, "bind requires one named endpoint"));
                }
            },
        }
    }
    Ok((quantum, bits))
}

fn position(
    id: u32,
    span: Span,
    ids: &HashMap<u32, glam::IVec3>,
) -> Result<glam::IVec3, ParseError> {
    ids.get(&id)
        .copied()
        .ok_or(ParseError::UndefinedId { id, span })
}

fn instance_port(name: &str, span: Span) -> Result<InstancePort, ParseError> {
    let (instance, port) =
        split_member(name).ok_or_else(|| syntax(span, "port must be instance.port"))?;
    Ok(InstancePort {
        instance: instance.to_string(),
        port: port.to_string(),
    })
}

fn syntax(span: Span, message: &str) -> ParseError {
    ParseError::Syntax {
        message: message.to_string(),
        span,
    }
}

fn collect_action_expr_names(action: &ActionStmt, names: &mut BTreeSet<String>) {
    let expr = match action {
        ActionStmt::Let(definition) => Some(&definition.expr.node),
        ActionStmt::DiscardIf(expr) => Some(&expr.node),
        ActionStmt::Resolve(definition) => Some(&definition.condition.node),
        ActionStmt::Feedback(definition) => definition.condition.as_ref().map(|expr| &expr.node),
        ActionStmt::Measure(_) => None,
    };
    if let Some(expr) = expr {
        collect_expr_names(expr, names);
    }
}

fn collect_expr_names(expr: &AstExpr, names: &mut BTreeSet<String>) {
    match expr {
        AstExpr::Var(name) => {
            names.insert(name.clone());
        }
        AstExpr::Not(inner) => collect_expr_names(&inner.node, names),
        AstExpr::Binary(_, left, right) => {
            collect_expr_names(&left.node, names);
            collect_expr_names(&right.node, names);
        }
    }
}
