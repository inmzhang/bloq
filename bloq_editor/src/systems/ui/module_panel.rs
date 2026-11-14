//! Module library, instance placement, and explicit port wiring beside the 3D preview.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_egui::egui;
use bloq_graph::{
    Action, BitRef, BlockGraph, InstancePort, ModuleRotation, PortDirection, QuantumPort,
    UDirection,
};
use glam::IVec3;

use super::{UiIntent, UiIntentBuffer};
use crate::module_authoring::{ModuleEdit, connection_seams, exposed_ports, unique_name};
use crate::resources::{EditorTabId, EditorTabs, GraphState, ModuleViewState};
use crate::theme::{self, ThemePalette};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct DefinitionEdit {
    pub(crate) parent: EditorTabId,
    pub(crate) name: String,
    pub(crate) original: String,
}

#[derive(Clone)]
pub(crate) struct ModuleUiState {
    expanded: bool,
    cache: Option<(u64, Result<Arc<BlockGraph>, String>)>,
    pub(crate) error: Option<String>,
    name: String,
    source_tab: Option<EditorTabId>,
    search: String,
    place_definition: Option<String>,
    instance_name: String,
    selected_instance: Option<String>,
    transform: Option<(IVec3, UDirection, i32)>,
    output: Option<InstancePort>,
    input: Option<InstancePort>,
    align: bool,
    hadamard: bool,
    pub(crate) compact_connections: bool,
    export_name: String,
    export_value: String,
    port_drafts: HashMap<IVec3, (QuantumPort, QuantumPort)>,
}

impl Default for ModuleUiState {
    fn default() -> Self {
        Self {
            expanded: true,
            cache: None,
            error: None,
            name: "Stage".into(),
            source_tab: None,
            search: String::new(),
            place_definition: None,
            instance_name: String::new(),
            selected_instance: None,
            transform: None,
            output: None,
            input: None,
            align: true,
            hadamard: false,
            compact_connections: true,
            export_name: "result".into(),
            export_value: String::new(),
            port_drafts: HashMap::new(),
        }
    }
}

impl ModuleUiState {
    pub(crate) fn selected_instance(&self) -> Option<&str> {
        self.selected_instance.as_deref()
    }

    pub(crate) fn select_instance(&mut self, name: String) {
        self.selected_instance = Some(name);
        self.transform = None;
    }

    fn program(&mut self, graph: &GraphState) -> Result<Arc<BlockGraph>, String> {
        if self
            .cache
            .as_ref()
            .is_none_or(|(revision, _)| *revision != graph.revision)
        {
            self.cache = Some((
                graph.revision,
                graph
                    .resolved_graph()
                    .map(Arc::new)
                    .map_err(|e| format!("{e:#}")),
            ));
            self.transform = None;
        }
        self.cache
            .as_ref()
            .expect("the cache was just populated if it was empty")
            .1
            .clone()
    }
}

pub(crate) fn draw_module_panel(
    root: &mut egui::Ui,
    state: &mut ModuleUiState,
    graph: &GraphState,
    tabs: &EditorTabs,
    module_view: Option<&ModuleViewState>,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    let max_width = (root.available_width() - 200.0).clamp(260.0, 480.0);
    let width = (root.available_width() * 0.42).clamp(260.0, max_width.min(400.0));
    if !state.expanded {
        egui::Panel::left("module_workspace_collapsed")
            .exact_size(42.0)
            .resizable(false)
            .show(root, |ui| {
                if ui
                    .button("›")
                    .on_hover_text("Expand module workspace")
                    .clicked()
                {
                    state.expanded = true;
                }
            });
        return;
    }
    let panel = egui::Panel::left("module_workspace")
        .default_size(width)
        .size_range(260.0..=max_width)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(palette.bg_panel)
                .inner_margin(12)
                .stroke(egui::Stroke::new(1.0, palette.border)),
        );
    panel.show(root, |ui| {
        ui.horizontal(|ui| {
            if ui
                .small_button("‹")
                .on_hover_text("Collapse module workspace")
                .clicked()
            {
                state.expanded = false;
            }
            ui.label(
                egui::RichText::new("Module workspace")
                    .strong()
                    .color(palette.text_bright),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("BLOG")
                    .on_hover_text("Edit the complete hierarchy as BLOG text.")
                    .clicked()
                {
                    intents.push(UiIntent::ShowBlogBuffer);
                }
            });
        });
        ui.small("Drag instances to join ports · X/Y/Z handles constrain moves");
        ui.add_space(8.0);
        if let Some(error) = state.error.clone() {
            egui::Frame::new()
                .fill(theme::with_alpha(palette.accent_error, 18))
                .inner_margin(8)
                .corner_radius(5)
                .show(ui, |ui| {
                    ui.colored_label(palette.accent_error, error);
                    if ui.small_button("Dismiss").clicked() {
                        state.error = None;
                    }
                });
            ui.add_space(6.0);
        }
        let program = match state.program(graph) {
            Ok(program) => program,
            Err(error) => {
                ui.label("Finish this module's ports before composing it.");
                ui.colored_label(palette.accent_error, error);
                ui.small("Each Port needs one pipe. Spatial ports also need an Input or Output role.");
                if ui.button("Edit geometry").clicked() {
                    intents.push(UiIntent::SetMode(crate::resources::EditorMode::Edit));
                }
                if ui.button("New composition tab").clicked() {
                    intents.push(UiIntent::NewTab);
                    intents.push(UiIntent::SetMode(crate::resources::EditorMode::Module));
                }
                return;
            }
        };
        let revision = graph.revision;
        egui::ScrollArea::vertical()
            .id_salt("module_tools")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("main")
                            .monospace()
                            .color(palette.accent_primary),
                    );
                    ui.small(format!(
                        "{} instances · {} public ports",
                        program.root().instances.len(),
                        program.root().interface.quantum_ports.len()
                    ));
                });
                if !graph.is_composed() && !graph.graph.is_empty() {
                    ui.small("This tab is a module. Wrap it to start a composition.");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut state.name)
                                .hint_text("Module name")
                                .desired_width(130.0),
                        );
                        if ui.button("Make reusable").clicked() {
                            intents.push(UiIntent::WrapAsModule(state.name.trim().into()));
                        }
                    });
                }
                ui.add_space(10.0);
                draw_library(
                    ui, state, &program, tabs, module_view, revision, intents, palette,
                );
                ui.add_space(10.0);
                ui.separator();
                draw_instances(ui, state, &program, revision, intents, palette);
                ui.add_space(10.0);
                ui.separator();
                draw_connections(ui, state, &program, revision, intents, palette);
                ui.add_space(10.0);
                ui.separator();
                draw_interface(ui, state, &program, revision, intents, palette);
                draw_bits(ui, state, &program, revision, intents);
                ui.add_space(12.0);
                if graph.is_composed()
                    && ui
                        .small_button("Open flat copy in new tab")
                        .on_hover_text(
                            "Edit individual blocks in a separate tab; this composition stays intact",
                        )
                        .clicked()
                {
                    intents.push(UiIntent::OpenFlatCopy);
                }
            });
    });
}

fn queue(intents: &mut UiIntentBuffer, revision: u64, edit: ModuleEdit) {
    intents.push(UiIntent::EditModule { revision, edit });
}

fn draw_library(
    ui: &mut egui::Ui,
    state: &mut ModuleUiState,
    program: &BlockGraph,
    tabs: &EditorTabs,
    module_view: Option<&ModuleViewState>,
    revision: u64,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    ui.label(egui::RichText::new("Definitions").strong());
    if program.modules().len() > 5 {
        ui.add(
            egui::TextEdit::singleline(&mut state.search)
                .hint_text("Find a definition")
                .desired_width(f32::INFINITY),
        );
    }
    let query = state.search.to_lowercase();
    for module in program
        .modules()
        .filter(|m| m.name != "main" && m.name.to_lowercase().contains(&query))
    {
        ui.push_id(&module.name, |ui| {
            let frame = egui::Frame::new()
                .fill(palette.bg_surface)
                .inner_margin(8)
                .corner_radius(5);
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    let color = module_view
                        .and_then(|view| view.modules().iter().position(|m| m.name == module.name))
                        .map_or(palette.text_dim, theme::module_color);
                    let (swatch, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().rect_filled(swatch, 2.0, color);
                    ui.label(egui::RichText::new(&module.name).strong());
                });
                let uses = program
                    .modules()
                    .flat_map(|m| &m.instances)
                    .filter(|i| i.definition == module.name)
                    .count();
                ui.small(format!(
                    "{} ports · {} uses{}",
                    module.interface.quantum_ports.len(),
                    uses,
                    if module.instances.is_empty() { "" } else { " · composed" }
                ));
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Place instance").clicked() {
                        state.place_definition = Some(module.name.clone());
                        let base = module.name.to_lowercase().replace("__", "_");
                        state.instance_name = unique_name(
                            &base,
                            &program.root().instances.iter().map(|i| i.name.clone()).collect(),
                        );
                    }
                    if ui
                        .button("Edit")
                        .on_hover_text(
                            "Open the definition in a linked tab, then Apply changes to update every instance",
                        )
                        .clicked()
                    {
                        intents.push(UiIntent::OpenModule(module.name.clone()));
                    }
                    if ui
                        .add_enabled(uses == 0, egui::Button::new("Remove"))
                        .on_hover_text("Remove this unused definition. Undo restores it.")
                        .clicked()
                    {
                        queue(
                            intents,
                            revision,
                            ModuleEdit::RemoveDefinition(module.name.clone()),
                        );
                    }
                });
            });
            ui.add_space(4.0);
        });
    }
    if let Some(definition) = state.place_definition.clone() {
        ui.label(format!("Place {definition}"));
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut state.instance_name)
                    .hint_text("Instance name")
                    .desired_width(150.0),
            );
            if ui.button("Place").clicked() {
                state.select_instance(state.instance_name.trim().into());
                queue(
                    intents,
                    revision,
                    ModuleEdit::AddInstance {
                        definition,
                        name: state.instance_name.trim().into(),
                    },
                );
                intents.push(UiIntent::InspectModuleInstance(
                    state.instance_name.trim().into(),
                ));
                state.place_definition = None;
            }
            if ui.small_button("Cancel").clicked() {
                state.place_definition = None;
            }
        });
    }
    let mut add_module = |ui: &mut egui::Ui| {
        ui.add(
            egui::TextEdit::singleline(&mut state.name)
                .hint_text("New definition name")
                .desired_width(f32::INFINITY),
        );
        if ui
            .button("Build new module")
            .on_hover_text(
                "Open an empty linked tab. Build geometry, then Apply changes to return here.",
            )
            .clicked()
        {
            intents.push(UiIntent::NewModule(state.name.trim().into()));
        }
        ui.small("Or reuse an open tab, including a composed module:");
        let available = tabs
            .tabs
            .iter()
            .filter(|t| t.id != tabs.active)
            .collect::<Vec<_>>();
        if state
            .source_tab
            .is_none_or(|id| !available.iter().any(|t| t.id == id))
        {
            state.source_tab = available.first().map(|t| t.id);
        }
        egui::ComboBox::from_id_salt("module_source_tab")
            .width((ui.available_width() - 8.0).max(80.0))
            .selected_text(
                state
                    .source_tab
                    .and_then(|id| tabs.title(id))
                    .unwrap_or("No other tabs open"),
            )
            .show_ui(ui, |ui| {
                for tab in &available {
                    ui.selectable_value(&mut state.source_tab, Some(tab.id), &tab.title);
                }
            });
        if ui
            .add_enabled(
                state.source_tab.is_some(),
                egui::Button::new("Add module from tab"),
            )
            .clicked()
        {
            intents.push(UiIntent::ImportModuleTab {
                tab: state
                    .source_tab
                    .expect("the button is enabled only while a source tab is chosen")
                    .get(),
                name: state.name.trim().into(),
            });
        }
        if available.is_empty() {
            ui.small("Open a gallery example or BLOG file to reuse it here.");
        }
    };
    if program.modules().len() == 1 {
        add_module(ui);
    } else {
        egui::CollapsingHeader::new("Add or build a module").show(ui, add_module);
    }
}

fn draw_instances(
    ui: &mut egui::Ui,
    state: &mut ModuleUiState,
    program: &BlockGraph,
    revision: u64,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    ui.label(egui::RichText::new("Instances").strong());
    if program.root().instances.is_empty() {
        ui.small("Place a definition to see it in the 3D preview.");
    }
    for instance in &program.root().instances {
        ui.push_id(&instance.name, |ui| {
            if ui
                .selectable_label(
                    state.selected_instance.as_ref() == Some(&instance.name),
                    format!("{} : {}", instance.name, instance.definition),
                )
                .clicked()
            {
                state.select_instance(instance.name.clone());
                intents.push(UiIntent::InspectModuleInstance(instance.name.clone()));
            }
        });
    }
    let Some(instance) = program
        .root()
        .instances
        .iter()
        .find(|i| Some(&i.name) == state.selected_instance.as_ref())
    else {
        return;
    };
    let (translation, axis, turns) = state.transform.get_or_insert((
        instance.translation,
        instance.rotation.axis(),
        i32::from(instance.rotation.quarter_turns()),
    ));
    let frame = egui::Frame::new()
        .fill(palette.bg_surface)
        .inner_margin(8)
        .corner_radius(5);
    frame.show(ui, |ui| {
        ui.small(format!("{} · placement", instance.name));
        ui.horizontal_wrapped(|ui| {
            for (label, value) in [
                ("X", &mut translation.x),
                ("Y", &mut translation.y),
                ("Z", &mut translation.z),
            ] {
                ui.add(egui::DragValue::new(value).prefix(format!("{label} ")).speed(0.1));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Rotate");
            egui::ComboBox::from_id_salt("module_rotation_axis")
                .selected_text(axis.to_string())
                .width(44.0)
                .show_ui(ui, |ui| {
                    for candidate in [UDirection::X, UDirection::Y, UDirection::Z] {
                        ui.selectable_value(axis, candidate, candidate.to_string());
                    }
                });
            egui::ComboBox::from_id_salt("module_rotation_turns")
                .selected_text(format!("{}°", *turns * 90))
                .width(65.0)
                .show_ui(ui, |ui| {
                    for candidate in 0..4 {
                        ui.selectable_value(turns, candidate, format!("{}°", candidate * 90));
                    }
                });
        });
        let rotation = ModuleRotation::new(*axis, *turns);
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    *translation != instance.translation || rotation != instance.rotation,
                    egui::Button::new("Apply placement"),
                )
                .clicked()
            {
                queue(
                    intents,
                    revision,
                    ModuleEdit::Transform {
                        name: instance.name.clone(),
                        translation: *translation,
                        rotation,
                    },
                );
            }
            if ui
                .button("Remove instance")
                .on_hover_text(
                    "Restore its neighbours' open ports. Undo restores this instance and its connections.",
                )
                .clicked()
            {
                queue(
                    intents,
                    revision,
                    ModuleEdit::RemoveInstance(instance.name.clone()),
                );
            }
        });
        ui.small("Placement must keep existing seams valid. Disconnect a seam to move its input group aside.");
    });
}

fn endpoint_label(port: &InstancePort) -> String {
    format!("{}.{}", port.instance, port.port)
}

fn draw_connections(
    ui: &mut egui::Ui,
    state: &mut ModuleUiState,
    program: &BlockGraph,
    revision: u64,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    ui.label(egui::RichText::new("Connections").strong());
    ui.checkbox(&mut state.compact_connections, "Compact connections")
        .on_hover_text("Try a direct pipe first. If it cannot fit, keep a connection cube. Off always keeps cubes.");
    let root = program.root();
    if root.instances.len() < 2 {
        ui.small("Place two instances to connect their ports.");
        return;
    }
    for seam in connection_seams(root) {
        let (output, input, hadamard) = (seam.output, seam.input, seam.hadamard);
        ui.push_id((endpoint_label(&output), endpoint_label(&input)), |ui| {
            egui::Frame::new()
                .fill(palette.bg_surface)
                .inner_margin(6)
                .corner_radius(4)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(endpoint_label(&output)).monospace());
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!(
                            "{} {}",
                            if hadamard { "-H→" } else { "→" },
                            endpoint_label(&input)
                        ));
                        if ui
                            .small_button("Disconnect")
                            .on_hover_text(
                                "Disconnect every port between these two instances and move the input group aside",
                            )
                            .clicked()
                        {
                            queue(
                                intents,
                                revision,
                                ModuleEdit::Disconnect {
                                    output: output.clone(),
                                    input: input.clone(),
                                },
                            );
                        }
                    });
                });
        });
    }
    let ports = exposed_ports(program).collect::<Vec<_>>();
    if state
        .output
        .as_ref()
        .is_some_and(|selected| !ports.iter().any(|(p, _)| p == selected))
    {
        state.output = None;
    }
    let output_type = state
        .output
        .as_ref()
        .and_then(|selected| ports.iter().find(|(p, _)| p == selected))
        .map(|(_, p)| p.resource_type.as_str());
    let inputs = ports
        .iter()
        .filter(|(endpoint, port)| {
            port.direction == PortDirection::Input
                && output_type.is_none_or(|resource| resource == port.resource_type)
                && state
                    .output
                    .as_ref()
                    .is_none_or(|output| output.instance != endpoint.instance)
        })
        .collect::<Vec<_>>();
    if state
        .input
        .as_ref()
        .is_some_and(|selected| !inputs.iter().any(|(p, _)| p == selected))
    {
        state.input = None;
    }
    ui.add_space(6.0);
    egui::ComboBox::from_id_salt("module_output")
        .selected_text(
            state
                .output
                .as_ref()
                .map(endpoint_label)
                .unwrap_or_else(|| "Choose output port…".into()),
        )
        .width((ui.available_width() - 8.0).max(80.0))
        .show_ui(ui, |ui| {
            for (endpoint, port) in ports
                .iter()
                .filter(|(_, p)| p.direction == PortDirection::Output)
            {
                ui.selectable_value(
                    &mut state.output,
                    Some(endpoint.clone()),
                    format!("{} · {}", endpoint_label(endpoint), port.resource_type),
                );
            }
        });
    egui::ComboBox::from_id_salt("module_input")
        .selected_text(
            state
                .input
                .as_ref()
                .map(endpoint_label)
                .unwrap_or_else(|| "Choose compatible input…".into()),
        )
        .width((ui.available_width() - 8.0).max(80.0))
        .show_ui(ui, |ui| {
            for (endpoint, port) in inputs {
                ui.selectable_value(
                    &mut state.input,
                    Some(endpoint.clone()),
                    format!("{} · {}", endpoint_label(endpoint), port.resource_type),
                );
            }
        });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut state.align, "Align input instance")
            .on_hover_text("Move the input instance so its interior meets the output face");
        ui.checkbox(&mut state.hadamard, "Hadamard");
    });
    if ui
        .add_enabled(
            state.output.is_some() && state.input.is_some(),
            egui::Button::new(if state.align {
                "Align & connect"
            } else {
                "Connect ports"
            }),
        )
        .clicked()
    {
        queue(
            intents,
            revision,
            ModuleEdit::Connect {
                output: state
                    .output
                    .clone()
                    .expect("the button is enabled only while both ports are chosen"),
                input: state
                    .input
                    .clone()
                    .expect("the button is enabled only while both ports are chosen"),
                hadamard: state.hadamard,
                align: state.align,
                compact: state.compact_connections,
            },
        );
    }
    ui.small("Unconnected ports stay public. Only matching resource types appear as inputs.");
}

fn draw_interface(
    ui: &mut egui::Ui,
    state: &mut ModuleUiState,
    program: &BlockGraph,
    revision: u64,
    intents: &mut UiIntentBuffer,
    palette: &ThemePalette,
) {
    state.port_drafts.retain(|position, _| {
        program
            .root()
            .interface
            .quantum_ports
            .iter()
            .any(|port| port.position == *position)
    });
    egui::CollapsingHeader::new(format!(
        "Public quantum ports ({})",
        program.root().interface.quantum_ports.len()
    ))
    .default_open(true)
    .show(ui, |ui| {
        if program.root().interface.quantum_ports.is_empty() {
            ui.small("No public quantum ports.");
        }
        for port in &program.root().interface.quantum_ports {
            ui.push_id((port.position.to_array(), "public_port"), |ui| {
                let (original, draft) = state
                    .port_drafts
                    .entry(port.position)
                    .or_insert_with(|| (port.clone(), port.clone()));
                if original != port {
                    original.clone_from(port);
                    draft.clone_from(port);
                }
                ui.horizontal(|ui| {
                    ui.colored_label(
                        palette.accent_primary,
                        if port.direction == PortDirection::Input {
                            "IN "
                        } else {
                            "OUT"
                        },
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut draft.name)
                            .desired_width((ui.available_width() - 64.0).max(70.0))
                            .hint_text("Port name"),
                    );
                    if ui
                        .add_enabled(draft != port, egui::Button::new("Set"))
                        .on_hover_text("Apply public port name and resource type")
                        .clicked()
                    {
                        queue(
                            intents,
                            revision,
                            ModuleEdit::RenamePort {
                                name: port.name.clone(),
                                replacement: draft.name.trim().into(),
                                resource_type: draft.resource_type.trim().into(),
                            },
                        );
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.small(format!(
                        "[{}, {}, {}]",
                        port.position.x, port.position.y, port.position.z
                    ));
                    if program.root().instances.is_empty() {
                        ui.add(
                            egui::TextEdit::singleline(&mut draft.resource_type)
                                .desired_width(100.0)
                                .hint_text("Resource type"),
                        );
                    } else {
                        ui.small(&draft.resource_type);
                    }
                });
                ui.add_space(4.0);
            });
        }
    });
}

fn draw_bits(
    ui: &mut egui::Ui,
    state: &mut ModuleUiState,
    program: &BlockGraph,
    revision: u64,
    intents: &mut UiIntentBuffer,
) {
    let root = program.root();
    let sources = root
        .interface
        .bit_inputs
        .iter()
        .map(|name| BitRef {
            instance: None,
            bit: name.clone(),
        })
        .chain(root.instances.iter().flat_map(|instance| {
            program
                .module(&instance.definition)
                .expect("validated program resolves every instance definition")
                .interface
                .bit_outputs
                .iter()
                .map(move |port| BitRef {
                    instance: Some(instance.name.clone()),
                    bit: port.name.clone(),
                })
        }))
        .collect::<Vec<_>>();
    let header = egui::CollapsingHeader::new("Classical interface")
        .default_open(!root.bit_bindings.is_empty());
    let label = |source: &BitRef| {
        source.instance.as_ref().map_or_else(
            || source.bit.clone(),
            |instance| format!("{instance}.{}", source.bit),
        )
    };
    header.show(ui, |ui| {
        if !root.interface.bit_inputs.is_empty() {
            ui.small(format!(
                "Public bit inputs: {}",
                root.interface.bit_inputs.join(", ")
            ));
            ui.small("Bind child inputs to exported results to close the program for compilation.");
        }
        for (index, binding) in root.bit_bindings.iter().enumerate() {
            ui.push_id((&binding.target_instance, &binding.target_bit), |ui| {
                ui.label(format!(
                    "{}.{} ←",
                    binding.target_instance, binding.target_bit
                ));
                let mut selected = binding.source.clone();
                egui::ComboBox::from_id_salt("bit_source")
                    .selected_text(label(&selected))
                    .show_ui(ui, |ui| {
                        for source in &sources {
                            ui.selectable_value(&mut selected, source.clone(), label(source));
                        }
                    });
                if selected != binding.source {
                    queue(
                        intents,
                        revision,
                        ModuleEdit::BindBit {
                            index,
                            source: selected,
                        },
                    );
                }
            });
        }
        for output in &root.interface.bit_outputs {
            ui.small(format!("out {} = {}", output.name, output.expr));
        }
        let values = root
            .actions()
            .iter()
            .filter_map(|action| match action {
                Action::Measure { name, .. } | Action::Let { name, .. } => Some(name.clone()),
                _ => None,
            })
            .chain(sources.iter().map(label))
            .collect::<Vec<_>>();
        if !values.is_empty() {
            ui.small("Export a result for a parent module:");
            ui.add(
                egui::TextEdit::singleline(&mut state.export_name)
                    .desired_width(140.0)
                    .hint_text("Result name"),
            );
            egui::ComboBox::from_id_salt("export_value")
                .selected_text(if state.export_value.is_empty() {
                    "Choose a value"
                } else {
                    &state.export_value
                })
                .show_ui(ui, |ui| {
                    for value in &values {
                        ui.selectable_value(&mut state.export_value, value.clone(), value);
                    }
                });
            if ui
                .add_enabled(
                    !state.export_value.is_empty(),
                    egui::Button::new("Export result"),
                )
                .clicked()
            {
                queue(
                    intents,
                    revision,
                    ModuleEdit::ExportBit {
                        name: state.export_name.trim().into(),
                        value: state.export_value.clone(),
                    },
                );
            }
        } else {
            ui.small(
                "Add named measurements or a module with exported bits to wire classical results.",
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemePreset;

    #[test]
    fn default_definition_name_is_accepted_by_the_module_validator() {
        let empty = bloq_graph::BlockGraph::new()
            .with_inferred_interface()
            .unwrap();
        crate::module_authoring::edit_graph(
            &empty,
            ModuleEdit::Import {
                name: ModuleUiState::default().name,
                program: Box::new(empty.clone()),
            },
        )
        .unwrap();
    }

    fn text_bounds(shape: &egui::epaint::Shape, texts: &mut Vec<(String, egui::Rect)>) {
        match shape {
            egui::epaint::Shape::Text(text) => texts.push((
                text.galley.job.text.clone(),
                egui::Rect::from_min_size(text.pos, text.galley.size()),
            )),
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    text_bounds(shape, texts);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn public_port_drafts_stay_with_their_tab_and_survive_other_edits() {
        let ctx = egui::Context::default();
        theme::setup_theme(&ctx, ThemePreset::Light);
        let program = crate::module_authoring::tests::stage();
        let mut graph = GraphState {
            graph: program.flatten().unwrap(),
            source_graph: Some(Arc::new(program)),
            ..Default::default()
        };
        let mut first = ModuleUiState::default();
        let mut second = ModuleUiState::default();
        let frame = |state: &mut ModuleUiState, graph: &GraphState, events| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280.0, 1800.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    draw_module_panel(
                        ui,
                        state,
                        graph,
                        &EditorTabs::default(),
                        None,
                        &mut UiIntentBuffer::default(),
                        theme::palette(ThemePreset::Light),
                    );
                },
            );
            output.textures_delta.clear();
            let mut texts = Vec::new();
            for shape in &output.shapes {
                text_bounds(&shape.shape, &mut texts);
            }
            texts
        };
        frame(&mut first, &graph, Vec::new());
        let labels = frame(&mut first, &graph, Vec::new());
        let position = labels
            .iter()
            .find(|(label, _)| label == "source")
            .unwrap()
            .1
            .center();
        for pressed in [true, false] {
            frame(
                &mut first,
                &graph,
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        let modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let labels = frame(
            &mut first,
            &graph,
            vec![
                egui::Event::Key {
                    key: egui::Key::A,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers,
                },
                egui::Event::Text("draft_source".into()),
            ],
        );
        assert!(labels.iter().any(|(label, _)| label == "draft_source"));
        let labels = frame(&mut second, &graph, Vec::new());
        assert!(
            labels.iter().any(|(label, _)| label == "source"),
            "a different tab must show its own port name"
        );
        graph.commit();
        let labels = frame(&mut first, &graph, Vec::new());
        assert!(
            labels.iter().any(|(label, _)| label == "draft_source"),
            "an unrelated revision must retain this pending edit"
        );
    }

    #[test]
    fn module_workspace_fits_both_themes_and_small_windows_and_places_from_the_library() {
        let program = crate::module_authoring::tests::composition();
        let mut graph = GraphState {
            graph: program.flatten().unwrap(),
            source_graph: Some(Arc::new(program.clone())),
            ..Default::default()
        };
        graph.commit();
        let view = ModuleViewState::from_graph(&program, &graph.graph);
        for preset in [ThemePreset::Light, ThemePreset::GruvboxMaterial] {
            for width in [480.0, 900.0, 1280.0] {
                let ctx = egui::Context::default();
                theme::setup_theme(&ctx, preset);
                let mut state = ModuleUiState::default();
                state.select_instance("second".into());
                let mut intents = UiIntentBuffer::default();
                let tabs = EditorTabs::default();
                let size = egui::vec2(width, 1500.0);
                let mut output = None;
                let mut panel_right = 0.0;
                // A panel opened on a wide window must shrink when that same window narrows.
                let mut initial = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1280.0, size.y),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        draw_module_panel(
                            ui,
                            &mut state,
                            &graph,
                            &tabs,
                            view.as_ref(),
                            &mut intents,
                            theme::palette(preset),
                        );
                    },
                );
                initial.textures_delta.clear();
                for _ in 0..3 {
                    let mut frame = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                            ..Default::default()
                        },
                        |ui| {
                            draw_module_panel(
                                ui,
                                &mut state,
                                &graph,
                                &tabs,
                                view.as_ref(),
                                &mut intents,
                                theme::palette(preset),
                            );
                            panel_right = ui.available_rect_before_wrap().left();
                        },
                    );
                    frame.textures_delta.clear();
                    output = Some(frame);
                }
                let output = output.unwrap();
                assert!(
                    panel_right <= 481.0,
                    "panel grew past its maximum at {width}: {panel_right}"
                );
                assert!(
                    panel_right <= width - 190.0,
                    "module controls left no usable preview after resizing to {width}: {panel_right}"
                );
                let mut texts = Vec::new();
                for shape in &output.shapes {
                    text_bounds(&shape.shape, &mut texts);
                }
                for (label, rect) in &texts {
                    assert!(
                        rect.right() <= panel_right + 1.0,
                        "'{label}' overflows module panel at {width}: {rect:?}, right={panel_right}"
                    );
                }
                let place = texts
                    .iter()
                    .find(|(label, _)| label == "Place instance")
                    .unwrap()
                    .1
                    .center();
                for pressed in [true, false] {
                    let mut frame = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                            events: vec![
                                egui::Event::PointerMoved(place),
                                egui::Event::PointerButton {
                                    pos: place,
                                    button: egui::PointerButton::Primary,
                                    pressed,
                                    modifiers: Default::default(),
                                },
                            ],
                            ..Default::default()
                        },
                        |ui| {
                            draw_module_panel(
                                ui,
                                &mut state,
                                &graph,
                                &tabs,
                                view.as_ref(),
                                &mut intents,
                                theme::palette(preset),
                            );
                        },
                    );
                    frame.textures_delta.clear();
                }
                assert_eq!(state.place_definition.as_deref(), Some("Memory"));
                assert_eq!(state.instance_name, "memory");
            }
        }
    }
}
