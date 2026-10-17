//! Transactional module authoring. Every operation builds and validates a new
//! block graph before the editor replaces its document or its preview.

use std::collections::{HashMap, HashSet};

use bloq_graph::{
    BitOutput, BitRef, BlockGraph, Expr, InstancePort, ModuleInstance, ModuleRotation,
    QuantumConnection, QuantumPort, flatten_module_definition,
};
use color_eyre::eyre::{self, ContextCompat, ensure};
use glam::IVec3;

mod connections;
pub(crate) use connections::{PortSnap, module_elements, module_instance_at, module_snap_points};

#[derive(Clone, Debug)]
pub(crate) enum ModuleEdit {
    Import {
        name: String,
        program: Box<BlockGraph>,
    },
    Replace {
        name: String,
        program: Box<BlockGraph>,
    },
    RemoveDefinition(String),
    AddInstance {
        definition: String,
        name: String,
    },
    Transform {
        name: String,
        translation: IVec3,
        rotation: ModuleRotation,
    },
    RemoveInstance(String),
    Connect {
        output: InstancePort,
        input: InstancePort,
        hadamard: bool,
        align: bool,
        compact: bool,
    },
    Disconnect {
        output: InstancePort,
        input: InstancePort,
    },
    RenamePort {
        name: String,
        replacement: String,
        resource_type: String,
    },
    BindBit {
        index: usize,
        source: BitRef,
    },
    ExportBit {
        name: String,
        value: String,
    },
}

pub(crate) fn edit_graph(program: &BlockGraph, edit: ModuleEdit) -> eyre::Result<BlockGraph> {
    if let ModuleEdit::AddInstance { definition, name } = edit {
        return Ok(bloq_graph::composition::place_module(
            program,
            &name,
            &definition,
            None,
            ModuleRotation::IDENTITY,
        )?
        .0);
    }
    if let ModuleEdit::Connect {
        output,
        input,
        hadamard,
        align,
        compact,
    } = edit
    {
        return Ok(connect_ports(program, output, input, hadamard, align, compact)?.program);
    }
    if let ModuleEdit::RemoveInstance(name) = &edit {
        let seam = connection_seams(program.root())
            .into_iter()
            .find(|seam| seam.output.instance == *name || seam.input.instance == *name);
        if let Some(seam) = seam {
            let next = edit_graph(
                program,
                ModuleEdit::Disconnect {
                    output: seam.output,
                    input: seam.input,
                },
            )?;
            return edit_graph(&next, edit);
        }
    }
    let mut modules = program
        .modules()
        .map(BlockGraph::clone_local_definition)
        .collect::<Vec<_>>();
    let root_index = modules
        .iter()
        .position(|module| module.name == BlockGraph::ENTRY_MODULE)
        .expect("validated program contains the entry module");
    match edit {
        ModuleEdit::Import {
            name,
            program: imported,
        } => {
            ensure!(
                name != BlockGraph::ENTRY_MODULE,
                "main is the composition entry; choose another module name"
            );
            ensure!(
                program.module(&name).is_none(),
                "Module '{name}' already exists"
            );
            import_definitions(&mut modules, &name, &imported);
        }
        ModuleEdit::Replace {
            name,
            program: imported,
        } => {
            ensure!(name != BlockGraph::ENTRY_MODULE, "Edit main in its own tab");
            ensure!(
                program.module(&name).is_some(),
                "Module '{name}' no longer exists"
            );
            let mut owned = HashSet::new();
            collect_definitions(program, &name, &mut owned)?;
            owned.remove(&name);
            // Helpers shared with another definition stay in the library.
            loop {
                let shared = modules
                    .iter()
                    .filter(|m| m.name != name && !owned.contains(&m.name))
                    .flat_map(|m| &m.instances)
                    .filter(|i| owned.contains(&i.definition))
                    .map(|i| i.definition.clone())
                    .collect::<Vec<_>>();
                if shared.is_empty() {
                    break;
                }
                for helper in shared {
                    owned.remove(&helper);
                }
            }
            modules.retain(|module| module.name != name && !owned.contains(&module.name));
            import_definitions(&mut modules, &name, &imported);
        }
        ModuleEdit::RemoveDefinition(name) => {
            ensure!(
                name != BlockGraph::ENTRY_MODULE,
                "Cannot remove the entry module"
            );
            ensure!(
                !modules
                    .iter()
                    .any(|module| module.instances.iter().any(|i| i.definition == name)),
                "Remove instances of '{name}' before removing its definition"
            );
            modules.retain(|module| module.name != name);
        }
        ModuleEdit::AddInstance { .. } => unreachable!("placement handled above"),
        ModuleEdit::Transform {
            name,
            translation,
            rotation,
        } => {
            transform_instance(&mut modules[root_index], &name, translation, rotation)?;
        }
        ModuleEdit::Connect { .. } => unreachable!("connections handled above"),
        ModuleEdit::Disconnect { output, input } => {
            let root = &mut modules[root_index];
            let seams = connection_seams(root);
            ensure!(
                seams
                    .iter()
                    .any(|seam| seam.output == output && seam.input == input),
                "Connection no longer exists"
            );
            // An interface can carry several ports (MAJ → UMA has four). Split
            // the complete interface, otherwise its remaining seams pin both sides.
            let seams = seams
                .into_iter()
                .filter(|seam| {
                    (seam.output.instance == output.instance
                        && seam.input.instance == input.instance)
                        || (seam.output.instance == input.instance
                            && seam.input.instance == output.instance)
                })
                .collect::<Vec<_>>();
            for seam in &seams {
                root.quantum_connections
                    .retain(|connection| match connection {
                        QuantumConnection::Pipe { output, input, .. } => {
                            output != &seam.output || input != &seam.input
                        }
                        _ => seam.cube.is_none() || bound_block(connection) != seam.cube,
                    });
                if let Some(cube) = seam.cube {
                    root.remove_block(cube);
                }
            }
            let group = connected_instances(root, &input.instance);
            ensure!(
                !group.contains(&output.instance),
                "This seam is part of a loop; rearrange it in the definition or BLOG source"
            );
            // Separate the downstream connected group, keeping its other seams intact.
            let graph = program.flatten()?;
            let max_x = graph.spans().map_or(0, |(x, _, _)| *x.end());
            let mut min_x = i32::MAX;
            for name in &group {
                let instance = root
                    .instances
                    .iter()
                    .find(|i| &i.name == name)
                    .expect("connected_instances names root instances");
                let child = program
                    .module(&instance.definition)
                    .expect("validated program resolves every instance definition");
                let child_graph = flatten_module_definition(program, child, "")?.graph;
                for block in child_graph.blocks() {
                    min_x = min_x.min(instance.try_transform_position(block.pos())?.x);
                }
            }
            let shift = max_x
                .checked_sub(min_x)
                .and_then(|x| x.checked_add(3))
                .wrap_err("No coordinate space to separate these instances")?;
            translate_group(root, &group, IVec3::new(shift, 0, 0))?;
            for endpoint in seams.iter().flat_map(|seam| [&seam.output, &seam.input]) {
                let instance = root
                    .instances
                    .iter()
                    .find(|i| i.name == endpoint.instance)
                    .expect("connection_seams names root instances")
                    .clone();
                expose_port(program, root, &instance, instance_port(program, endpoint)?)?;
            }
        }
        ModuleEdit::RemoveInstance(name) => {
            let root = &mut modules[root_index];
            let removed = root
                .instances
                .iter()
                .find(|i| i.name == name)
                .wrap_err("Instance no longer exists")?
                .clone();
            let connections = std::mem::take(&mut root.quantum_connections);
            for connection in connections {
                match &connection {
                    QuantumConnection::Input {
                        input: endpoint,
                        block,
                        ..
                    }
                    | QuantumConnection::Output {
                        output: endpoint,
                        block,
                        ..
                    } if endpoint.instance == name => remove_public_port(root, *block),
                    _ => root.quantum_connections.push(connection),
                }
            }
            root.instances.retain(|i| i.name != name);
            let previous_inputs = root
                .bit_bindings
                .iter()
                .filter(|binding| binding.target_instance == name)
                .map(|binding| binding.source.clone())
                .collect::<Vec<_>>();
            root.bit_bindings
                .retain(|binding| binding.target_instance != name);
            let child = program
                .module(&removed.definition)
                .expect("validated program resolves every instance definition");
            root.interface.bit_outputs.retain(|output| {
                !child
                    .interface
                    .bit_outputs
                    .iter()
                    .any(|port| output.expr == Expr::Var(format!("{name}.{}", port.name)))
            });
            for source in previous_inputs {
                prune_unused_input(root, &source)?;
            }
        }
        ModuleEdit::RenamePort {
            name,
            replacement,
            resource_type,
        } => {
            let root = &mut modules[root_index];
            let port = root
                .interface
                .quantum_ports
                .iter_mut()
                .find(|p| p.name == name)
                .wrap_err("Port no longer exists")?;
            port.name = replacement;
            port.resource_type = resource_type;
            let position = port.position;
            let tag = port.name.clone();
            root.set_block_tag(position, &tag)?;
        }
        ModuleEdit::BindBit { index, source } => {
            let root = &mut modules[root_index];
            let binding = root
                .bit_bindings
                .get_mut(index)
                .wrap_err("Bit binding no longer exists")?;
            let old = std::mem::replace(&mut binding.source, source);
            prune_unused_input(root, &old)?;
        }
        ModuleEdit::ExportBit { name, value } => {
            modules[root_index].interface.bit_outputs.push(BitOutput {
                name,
                expr: Expr::Var(value),
            });
        }
    }
    let root = modules
        .iter()
        .find(|module| module.name == BlockGraph::ENTRY_MODULE)
        .expect("no edit removes or renames the entry module");
    ensure!(
        root.actions() == program.root().local_body().actions()
            && root.branch_definitions() == program.root().local_body().branch_definitions(),
        "This edit would change local actions or branch geometry. Edit the owning definition or BLOG source instead"
    );
    Ok(BlockGraph::from_definitions(modules)?)
}

fn collect_definitions(
    program: &BlockGraph,
    name: &str,
    names: &mut HashSet<String>,
) -> eyre::Result<()> {
    let mut pending = vec![name.to_owned()];
    while let Some(name) = pending.pop() {
        if names.insert(name.clone()) {
            pending.extend(
                program
                    .module(&name)
                    .wrap_err("Module no longer exists")?
                    .instances
                    .iter()
                    .map(|instance| instance.definition.clone()),
            );
        }
    }
    Ok(())
}

fn import_definitions(modules: &mut Vec<BlockGraph>, name: &str, imported: &BlockGraph) {
    let mut used = modules
        .iter()
        .map(|module| module.name.clone())
        .collect::<HashSet<_>>();
    used.insert(name.to_owned());
    let mut names = HashMap::from([(BlockGraph::ENTRY_MODULE.to_owned(), name.to_owned())]);
    for module in imported
        .modules()
        .filter(|m| m.name != BlockGraph::ENTRY_MODULE)
    {
        let base = if used.contains(&module.name) {
            format!("{name}_{}", module.name)
        } else {
            module.name.clone()
        };
        let replacement = unique_name(&base, &used);
        used.insert(replacement.clone());
        names.insert(module.name.clone(), replacement);
    }
    modules.extend(
        imported
            .modules()
            .map(BlockGraph::clone_local_definition)
            .map(|mut module| {
                module.name = names[&module.name].clone();
                for instance in &mut module.instances {
                    instance.definition.clone_from(&names[&instance.definition]);
                }
                module
            }),
    );
}

/// Preserve authored interface names/types when editing a leaf's geometry.
pub(crate) fn replace_leaf_body(
    program: &BlockGraph,
    graph: &BlockGraph,
) -> eyre::Result<BlockGraph> {
    ensure!(
        program.root().instances.is_empty(),
        "Edit a definition in Modules to change composed geometry"
    );
    let ast = bloq_graph::parse_blog_program_to_ast(&graph.to_blog_text())?;
    let inferred = bloq_graph::lower_blog_graph_ast_deferred(&ast)?;
    let mut modules = program
        .modules()
        .map(BlockGraph::clone_local_definition)
        .collect::<Vec<_>>();
    let root = modules
        .iter_mut()
        .find(|m| m.name == BlockGraph::ENTRY_MODULE)
        .expect("validated program contains the entry module");
    let old_ports = std::mem::take(&mut root.interface.quantum_ports);
    let interface = root.interface.clone();
    *root = inferred.clone_local_definition();
    root.interface = interface;
    for input in &inferred.root().interface.bit_inputs {
        if !root.interface.bit_inputs.contains(input) {
            root.interface.bit_inputs.push(input.clone());
        }
    }
    for port in &inferred.root().interface.quantum_ports {
        let mut port = port.clone();
        if let Some(old) = old_ports.iter().find(|old| old.position == port.position) {
            port.name.clone_from(&old.name);
            port.resource_type.clone_from(&old.resource_type);
        } else {
            let used = old_ports
                .iter()
                .map(|p| p.name.clone())
                .chain(root.interface.quantum_ports.iter().map(|p| p.name.clone()))
                .chain(root.interface.bit_inputs.iter().cloned())
                .chain(root.interface.bit_outputs.iter().map(|p| p.name.clone()))
                .collect();
            port.name = unique_name(&port.name, &used);
        }
        root.interface.quantum_ports.push(port);
    }
    Ok(BlockGraph::from_definitions(modules)?)
}

use bloq_graph::composition::{
    bound_block, connect_ports, expose_port, exposed_port, prune_unused_input, remove_public_port,
    translate_group,
};
pub(crate) use bloq_graph::composition::{
    connected_instances, connection_seams, exposed_ports, translate_module, unique_name,
};

fn transform_instance(
    root: &mut BlockGraph,
    name: &str,
    translation: IVec3,
    rotation: ModuleRotation,
) -> eyre::Result<()> {
    let old = root
        .instances
        .iter()
        .find(|i| i.name == name)
        .wrap_err("Instance no longer exists")?
        .clone();
    let next = ModuleInstance {
        translation,
        rotation,
        ..old.clone()
    };
    // The public blocks belong to the parent. Only freely forwarded ports move
    // with a child; authored local geometry is left for the validator to check.
    let positions = root
        .quantum_connections
        .iter()
        .filter_map(|c| match c {
            QuantumConnection::Input {
                input: endpoint,
                block,
                ..
            }
            | QuantumConnection::Output {
                output: endpoint,
                block,
                ..
            } if endpoint.instance == name => Some((endpoint, *block)),
            _ => None,
        })
        .filter(|(endpoint, _)| exposed_port(root, endpoint).is_some())
        .map(|(_, p)| p)
        .collect::<Vec<_>>();
    let inverse = ModuleRotation::new(
        old.rotation.axis(),
        -i32::from(old.rotation.quarter_turns()),
    );
    let mut moves = Vec::new();
    for position in positions {
        let local = inverse.try_rotate_position(checked_difference(position, old.translation)?)?;
        let target = next.try_transform_position(local)?;
        let block = root
            .get_block(position)
            .expect("validated bind references a body block")
            .clone();
        moves.push((position, target, block));
    }
    for (position, _, _) in &moves {
        root.remove_block(*position);
    }
    for (position, target, block) in &moves {
        // Port blocks have no directional basis; preserve color, tag, and role.
        root.try_add_block(block.try_with_shift(checked_difference(*target, *position)?)?)?;
    }
    for port in &mut root.interface.quantum_ports {
        if let Some((_, target, _)) = moves
            .iter()
            .find(|(position, _, _)| *position == port.position)
        {
            port.position = *target;
        }
    }
    for connection in &mut root.quantum_connections {
        let block = match connection {
            QuantumConnection::Input {
                block,
                input: endpoint,
                ..
            }
            | QuantumConnection::Output {
                block,
                output: endpoint,
                ..
            } if endpoint.instance == name => block,
            _ => continue,
        };
        if let Some((_, target, _)) = moves.iter().find(|(position, _, _)| position == block) {
            *block = *target;
        }
    }
    *root
        .instances
        .iter_mut()
        .find(|i| i.name == name)
        .expect("the instance was located above") = next;
    Ok(())
}

fn instance_port<'a>(
    program: &'a BlockGraph,
    endpoint: &InstancePort,
) -> eyre::Result<&'a QuantumPort> {
    let instance = program
        .root()
        .instances
        .iter()
        .find(|i| i.name == endpoint.instance)
        .wrap_err("Instance no longer exists")?;
    program
        .module(&instance.definition)
        .expect("validated program resolves every instance definition")
        .interface
        .quantum_ports
        .iter()
        .find(|p| p.name == endpoint.port)
        .wrap_err("Port no longer exists")
}

fn checked_difference(a: IVec3, b: IVec3) -> eyre::Result<IVec3> {
    Ok(IVec3::new(
        a.x.checked_sub(b.x).wrap_err("X coordinate overflow")?,
        a.y.checked_sub(b.y).wrap_err("Y coordinate overflow")?,
        a.z.checked_sub(b.z).wrap_err("Z coordinate overflow")?,
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn stage() -> BlockGraph {
        bloq_graph::BlockGraph::from_text("BLOG 1.0\nmodule main {\n in source: data = 0\n out result: data = 2\n 0: Port [0,0,0] role=input\n 1: ZXZ [0,0,1]\n 2: Port [0,0,2] role=output\n 0 -> +Z\n 1 -> +Z\n}\n").unwrap()
    }

    pub(crate) fn composition() -> BlockGraph {
        let program = edit_graph(
            &BlockGraph::new().with_inferred_interface().unwrap(),
            ModuleEdit::Import {
                name: "Memory".into(),
                program: Box::new(stage()),
            },
        )
        .unwrap();
        ["first", "second", "third"]
            .into_iter()
            .fold(program, |program, name| {
                edit_graph(
                    &program,
                    ModuleEdit::AddInstance {
                        definition: "Memory".into(),
                        name: name.into(),
                    },
                )
                .unwrap()
            })
    }

    fn endpoint(instance: &str, port: &str) -> InstancePort {
        InstancePort {
            instance: instance.into(),
            port: port.into(),
        }
    }

    fn connect(program: &BlockGraph, from: &str, to: &str) -> BlockGraph {
        edit_graph(
            program,
            ModuleEdit::Connect {
                output: endpoint(from, "result"),
                input: endpoint(to, "source"),
                hadamard: false,
                align: true,
                compact: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn dragging_joins_ports_with_optional_compaction_and_keeps_joins_movable() {
        let original = composition();
        for compact in [false, true] {
            let joined =
                translate_module(&original, "second", IVec3::new(-3, 0, 2), compact).unwrap();
            assert_eq!(joined.connections, 1);
            assert_eq!(joined.compacted, compact);
            assert_eq!(
                joined.offset,
                IVec3::new(-3, 0, if compact { 1 } else { 2 })
            );
            let program = joined.program;
            assert_eq!(program.root().interface.quantum_ports.len(), 4);
            assert_eq!(
                program.flatten().unwrap().block_count(),
                if compact { 7 } else { 8 }
            );
            assert_eq!(
                program.extract_definition("Memory").unwrap().to_blog_text(),
                original
                    .extract_definition("Memory")
                    .unwrap()
                    .to_blog_text()
            );
            let reloaded = bloq_graph::BlockGraph::from_text(&program.to_blog_text()).unwrap();
            let moved = translate_module(&reloaded, "first", IVec3::new(0, 3, 0), compact).unwrap();
            assert_eq!(moved.program.root().instances[1].translation.y, 3);
            let split = edit_graph(
                &moved.program,
                ModuleEdit::Disconnect {
                    output: endpoint("first", "result"),
                    input: endpoint("second", "source"),
                },
            )
            .unwrap();
            assert_eq!(split.root().interface.quantum_ports.len(), 6);
            assert_eq!(split.flatten().unwrap().block_count(), 9);
            let ctx = bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3));
            ctx.compile(&program).unwrap();
        }
        let _ = translate_module(&original, "second", IVec3::new(-3, 0, 0), true)
            .expect_err("overlapping placements must fail");
        assert_eq!(original.to_blog_text(), composition().to_blog_text());
    }

    #[test]
    fn maj_uma_drag_connects_all_four_ports_preserving_hadamards() {
        let mut program = BlockGraph::new().with_inferred_interface().unwrap();
        for (name, item) in [
            ("Maj", bloq_graph::GalleryItem::CCZInjectedMaj),
            ("Uma", bloq_graph::GalleryItem::UMA),
        ] {
            program = edit_graph(
                &program,
                ModuleEdit::Import {
                    name: name.into(),
                    program: Box::new(item.build()),
                },
            )
            .unwrap();
            program = edit_graph(
                &program,
                ModuleEdit::AddInstance {
                    definition: name.into(),
                    name: name.to_lowercase(),
                },
            )
            .unwrap();
        }
        let uma = program
            .root()
            .instances
            .iter()
            .find(|instance| instance.name == "uma")
            .unwrap();
        let offset = IVec3::new(3, 0, 6) - uma.translation;
        let joined = translate_module(&program, "uma", offset, true).unwrap();
        assert_eq!(joined.connections, 4);
        assert!(joined.compacted);
        assert_eq!(
            joined.program.root().instances[1].translation,
            IVec3::new(3, 0, 5)
        );
        assert_eq!(
            connection_seams(joined.program.root())
                .iter()
                .filter(|seam| seam.hadamard)
                .count(),
            2
        );
        let regular = translate_module(&program, "uma", offset, false).unwrap();
        assert!(!regular.compacted);
        assert_eq!(
            regular.program.flatten().unwrap().block_count(),
            joined.program.flatten().unwrap().block_count() + 4
        );
        for composed in [joined.program, regular.program] {
            let removed = edit_graph(&composed, ModuleEdit::RemoveInstance("uma".into())).unwrap();
            assert_eq!(
                removed.flatten().unwrap().block_count(),
                bloq_graph::GalleryItem::CCZInjectedMaj
                    .build()
                    .block_count()
            );
        }
    }

    #[test]
    fn obstructed_compact_join_falls_back_to_cubes() {
        let original = composition();
        let mut modules = original
            .modules()
            .map(BlockGraph::clone_local_definition)
            .collect::<Vec<_>>();
        let memory = modules
            .iter_mut()
            .find(|module| module.name == "Memory")
            .unwrap();
        memory
            .try_add_block(bloq_graph::Block::new(
                IVec3::new(1, 0, 1),
                bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
            ))
            .unwrap();
        let root = modules
            .iter_mut()
            .find(|module| module.name == "main")
            .unwrap();
        root.try_add_block(bloq_graph::Block::new(
            IVec3::new(1, 0, 2),
            bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ))
        .unwrap();
        let program = BlockGraph::from_definitions(modules).unwrap();
        let offset = IVec3::new(-3, 0, 2);
        let joined = translate_module(&program, "second", offset, true).unwrap();
        assert_eq!(joined.offset, offset);
        assert!(!joined.compacted);
        assert_eq!(joined.connections, 1);
    }

    #[test]
    fn author_wire_reuse_and_compile_a_nested_composition() {
        let program = connect(
            &connect(&composition(), "first", "second"),
            "second",
            "third",
        );
        assert_eq!(program.root().interface.quantum_ports.len(), 2);
        assert_eq!(program.flatten().unwrap().block_count(), 5);
        let text = program.to_blog_text();
        let reloaded = bloq_graph::BlockGraph::from_text(&text).unwrap();
        assert_eq!(reloaded.root().interface, program.root().interface);
        assert_eq!(reloaded.root().instances, program.root().instances);

        let mut parent = edit_graph(
            &BlockGraph::new().with_inferred_interface().unwrap(),
            ModuleEdit::Import {
                name: "Pipeline".into(),
                program: Box::new(program),
            },
        )
        .unwrap();
        for name in ["left", "right"] {
            parent = edit_graph(
                &parent,
                ModuleEdit::AddInstance {
                    definition: "Pipeline".into(),
                    name: name.into(),
                },
            )
            .unwrap();
        }
        assert_eq!(parent.modules().len(), 3);
        assert_eq!(parent.flatten().unwrap().block_count(), 10);
        // Reopening and applying a composite must not accumulate unused helpers.
        for _ in 0..3 {
            let opened = parent.extract_definition("Pipeline").unwrap();
            parent = edit_graph(
                &parent,
                ModuleEdit::Replace {
                    name: "Pipeline".into(),
                    program: Box::new(opened),
                },
            )
            .unwrap();
            assert_eq!(parent.modules().len(), 3);
        }
        let ctx = bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3));
        let compiled = ctx.compile(&parent).unwrap();
        assert!(compiled.bloq.nodes().next().is_some());
    }

    #[test]
    fn disconnect_and_remove_restore_ports_without_overlaps() {
        let program = connect(
            &connect(&composition(), "first", "second"),
            "second",
            "third",
        );
        let split = edit_graph(
            &program,
            ModuleEdit::Disconnect {
                output: endpoint("first", "result"),
                input: endpoint("second", "source"),
            },
        )
        .unwrap();
        assert_eq!(split.root().interface.quantum_ports.len(), 4);
        assert_eq!(split.flatten().unwrap().block_count(), 7);
        let removed = edit_graph(&program, ModuleEdit::RemoveInstance("second".into())).unwrap();
        assert_eq!(removed.root().instances.len(), 2);
        assert_eq!(removed.root().interface.quantum_ports.len(), 4);
        assert_eq!(removed.flatten().unwrap().block_count(), 6);
    }

    #[test]
    fn rejected_edits_leave_the_document_unchanged() {
        let program = composition();
        let before = program.to_blog_text();
        for edit in [
            ModuleEdit::AddInstance {
                definition: "Memory".into(),
                name: "first".into(),
            },
            ModuleEdit::AddInstance {
                definition: "Memory".into(),
                name: "bad name".into(),
            },
            ModuleEdit::Transform {
                name: "second".into(),
                translation: IVec3::ZERO,
                rotation: ModuleRotation::IDENTITY,
            },
            ModuleEdit::Transform {
                name: "second".into(),
                translation: IVec3::splat(i32::MAX),
                rotation: ModuleRotation::IDENTITY,
            },
            ModuleEdit::RemoveDefinition("Memory".into()),
            ModuleEdit::Connect {
                output: endpoint("first", "source"),
                input: endpoint("second", "result"),
                hadamard: false,
                align: true,
                compact: true,
            },
        ] {
            let _error = edit_graph(&program, edit).unwrap_err();
            assert_eq!(program.to_blog_text(), before);
        }
    }

    #[test]
    fn leaf_edits_keep_the_public_names_types_and_exported_bits() {
        let source = bloq_graph::BlockGraph::from_text(
            "BLOG 1.0\nmodule main {\n in flag\n out result = flag\n}\n",
        )
        .unwrap();
        assert_eq!(
            replace_leaf_body(&source, &BlockGraph::new())
                .unwrap()
                .to_blog_text(),
            source.to_blog_text()
        );
        let program = stage();
        let mut graph = program.flatten().unwrap();
        graph.set_block_tag(IVec3::Z, "changed").unwrap();
        let edited = replace_leaf_body(&program, &graph).unwrap();
        assert_eq!(edited.root().interface, program.root().interface);
        assert_eq!(
            edited
                .root()
                .local_body()
                .get_block(IVec3::Z)
                .unwrap()
                .tag(),
            Some("changed")
        );
        let (inserted, _) = crate::systems::input::insert_graph_without_overlap(
            &edited.flatten().unwrap(),
            &bloq_graph::GalleryItem::ThreeBitAdder
                .build()
                .flatten()
                .unwrap(),
        )
        .unwrap();
        let action_count = inserted.actions().len();
        let inserted = replace_leaf_body(&edited, &inserted).unwrap();
        assert_eq!(inserted.root().local_body().actions().len(), action_count);
        bloq_graph::BlockGraph::from_text(&inserted.to_blog_text()).unwrap();
    }

    #[test]
    fn classical_bindings_close_forwarded_inputs_and_reject_cycles() {
        let mut producer = stage().root().clone();
        producer
            .set_actions_lenient(vec![bloq_graph::Action::Measure {
                target: bloq_graph::MeasureTarget::Node(IVec3::Z),
                name: "m".into(),
            }])
            .unwrap();
        producer.interface.bit_outputs.push(BitOutput {
            name: "result_bit".into(),
            expr: Expr::Var("m".into()),
        });
        let mut consumer = stage().root().clone();
        consumer.interface.bit_inputs.push("control".into());
        consumer.interface.bit_outputs.push(BitOutput {
            name: "echo".into(),
            expr: Expr::Var("control".into()),
        });
        let mut program = BlockGraph::new().with_inferred_interface().unwrap();
        for (name, body) in [("Producer", producer), ("Consumer", consumer)] {
            program = edit_graph(
                &program,
                ModuleEdit::Import {
                    name: name.into(),
                    program: Box::new(BlockGraph::from_definitions(vec![body]).unwrap()),
                },
            )
            .unwrap();
            program = edit_graph(
                &program,
                ModuleEdit::AddInstance {
                    definition: name.into(),
                    name: name.to_lowercase(),
                },
            )
            .unwrap();
        }
        assert_eq!(program.root().interface.bit_inputs, ["consumer_control"]);
        program = edit_graph(
            &program,
            ModuleEdit::BindBit {
                index: 0,
                source: BitRef {
                    instance: Some("producer".into()),
                    bit: "result_bit".into(),
                },
            },
        )
        .unwrap();
        assert!(program.root().interface.bit_inputs.is_empty());
        let graph = program.flatten().unwrap();
        assert!(graph.actions().iter().any(|action| matches!(action,
            bloq_graph::Action::Let { name, expr: Expr::Var(value) } if name == "consumer__control" && value == "producer__m")));
        let _error = edit_graph(
            &program,
            ModuleEdit::BindBit {
                index: 0,
                source: BitRef {
                    instance: Some("consumer".into()),
                    bit: "echo".into(),
                },
            },
        )
        .unwrap_err();
    }
}
