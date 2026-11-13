//! The command registry: the named, user-facing subset of [`UiIntent`].
//!
//! [`UiIntent`] is the editor's full internal bus, hover updates and viewer
//! cursors included; a [`Command`] is the part of it with a name worth
//! searching for. Keeping them apart is what makes the palette curated — a new
//! `UiIntent` variant does not silently publish a command, and the exhaustive
//! `match`es below mean a command cannot exist without a title.
//!
//! Key dispatch stays in `systems::input`: several viewport keys are held
//! modifiers or stateful repeats that a one-shot command cannot express, so
//! [`Command::keybind`] carries only the label to display.

use std::borrow::Cow;

use bloq_graph::GalleryItem;
use strum::Display;

use super::intents::UiIntent;
use super::toolbar::compact_mode_label;
use crate::components::GraphElement;
use crate::resources::{EditorMode, EditorState, EditorTabs, GraphState, ImportExportState};
use crate::systems::jobs::EditorJobs;
use crate::theme::ThemePreset;

/// Everything a command needs to decide whether it is runnable and what payload
/// its intent carries.
pub(crate) struct CommandContext<'a> {
    pub(crate) editor_state: &'a EditorState,
    pub(crate) graph_state: &'a GraphState,
    pub(crate) tabs: &'a EditorTabs,
    pub(crate) jobs: &'a EditorJobs,
    pub(crate) import_export: &'a ImportExportState,
}

/// Whether a command can run in the current editor state, and why not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Availability {
    Available,
    Unavailable(&'static str),
}

impl Availability {
    pub(crate) const fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

/// Command groups, ranked by how often you reach for them — the derived `Ord`
/// follows declaration order, and that is how an empty query is grouped. `App`
/// is last so `Quit` is never the row `Enter` lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Display)]
pub(crate) enum CommandCategory {
    File,
    Mode,
    Tabs,
    View,
    Graph,
    Gallery,
    Help,
    /// Only `Quit` lives here, and the web build has nothing to quit.
    #[cfg(not(target_arch = "wasm32"))]
    App,
}

/// A single palette entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Command {
    OpenBlogFile,
    SaveBlog,
    CopyGraphAsBlog,
    ClearGraph,
    Screenshot,

    NewTab,
    NextTab,
    PreviousTab,
    CloseTab,
    RenameTab,

    SetMode(EditorMode),
    InsertGallery(GalleryItem),

    DuplicateSelection,
    ValidateGraph,
    FillPorts,
    FixShadowedFaces,
    FlipXZBasis,
    RandomlyResolveSelectives,

    ResetCamera,
    ToggleAxis,
    ToggleCurrentLayerOnly,
    TogglePortTags,
    /// Absent until the user types a trailing integer into the query.
    SetPlaneHeight(Option<i32>),
    ToggleTheme,

    ShowShortcuts,

    #[cfg(not(target_arch = "wasm32"))]
    Quit,
}

/// Every command the palette can offer, minus the parametric plane-height entry
/// the palette appends once it has parsed an argument out of the query.
pub(crate) fn all_commands() -> Vec<Command> {
    let mut commands = vec![
        Command::OpenBlogFile,
        Command::SaveBlog,
        Command::CopyGraphAsBlog,
        Command::ClearGraph,
        Command::Screenshot,
        Command::NewTab,
        Command::NextTab,
        Command::PreviousTab,
        Command::CloseTab,
        Command::RenameTab,
        Command::SetMode(EditorMode::View),
        Command::SetMode(EditorMode::Module),
        Command::SetMode(EditorMode::Edit),
        Command::SetMode(EditorMode::Bloq),
        Command::DuplicateSelection,
        Command::ValidateGraph,
        Command::FillPorts,
        Command::FixShadowedFaces,
        Command::FlipXZBasis,
        Command::RandomlyResolveSelectives,
        Command::ResetCamera,
        Command::ToggleAxis,
        Command::ToggleCurrentLayerOnly,
        Command::TogglePortTags,
        Command::ToggleTheme,
        Command::ShowShortcuts,
        #[cfg(not(target_arch = "wasm32"))]
        Command::Quit,
    ];
    commands.extend(GalleryItem::iter().map(Command::InsertGallery));
    commands
}

impl Command {
    pub(crate) const fn category(self) -> CommandCategory {
        match self {
            Self::OpenBlogFile
            | Self::SaveBlog
            | Self::CopyGraphAsBlog
            | Self::ClearGraph
            | Self::Screenshot => CommandCategory::File,
            Self::NewTab | Self::NextTab | Self::PreviousTab | Self::CloseTab | Self::RenameTab => {
                CommandCategory::Tabs
            }
            Self::SetMode(_) => CommandCategory::Mode,
            Self::InsertGallery(_) => CommandCategory::Gallery,
            Self::DuplicateSelection
            | Self::ValidateGraph
            | Self::FillPorts
            | Self::FixShadowedFaces
            | Self::FlipXZBasis
            | Self::RandomlyResolveSelectives => CommandCategory::Graph,
            Self::ResetCamera
            | Self::ToggleAxis
            | Self::ToggleCurrentLayerOnly
            | Self::TogglePortTags
            | Self::SetPlaneHeight(_)
            | Self::ToggleTheme => CommandCategory::View,
            Self::ShowShortcuts => CommandCategory::Help,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Quit => CommandCategory::App,
        }
    }

    /// The action half of the palette row, without its category prefix.
    pub(crate) fn title(self) -> Cow<'static, str> {
        match self {
            Self::OpenBlogFile => "Open BLOG File".into(),
            Self::SaveBlog => if cfg!(target_arch = "wasm32") {
                "Download BLOG File"
            } else {
                "Save BLOG File"
            }
            .into(),
            Self::CopyGraphAsBlog => "Copy Graph as BLOG".into(),
            Self::ClearGraph => "Clear Graph".into(),
            Self::Screenshot => "Take Screenshot".into(),
            Self::NewTab => "New Tab".into(),
            Self::NextTab => "Next Tab".into(),
            Self::PreviousTab => "Previous Tab".into(),
            Self::CloseTab => "Close Tab".into(),
            Self::RenameTab => "Rename Tab".into(),
            Self::SetMode(mode) => format!("{} Mode", compact_mode_label(mode)).into(),
            Self::InsertGallery(entry) => format!("Insert {}", entry.id()).into(),
            Self::DuplicateSelection => "Duplicate Selection".into(),
            Self::ValidateGraph => "Validate Graph".into(),
            Self::FillPorts => "Fill Ports".into(),
            Self::FixShadowedFaces => "Fix Shadowed Faces".into(),
            Self::FlipXZBasis => "Flip XZ Basis".into(),
            Self::RandomlyResolveSelectives => "Randomly Resolve Selectives".into(),
            Self::ResetCamera => "Reset Camera".into(),
            Self::ToggleAxis => "Toggle Axis".into(),
            Self::ToggleCurrentLayerOnly => "Toggle Current-Layer-Only View".into(),
            Self::TogglePortTags => "Show / Hide Port Tags".into(),
            Self::SetPlaneHeight(None) => "Set Plane Height".into(),
            Self::SetPlaneHeight(Some(height)) => format!("Set Plane Height to {height}").into(),
            Self::ToggleTheme => "Toggle Theme".into(),
            Self::ShowShortcuts => "Show Shortcuts".into(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Quit => "Quit".into(),
        }
    }

    /// `"Category: Title"` — what the palette shows and what the fuzzy search
    /// matches, so typing `view reset` finds `View: Reset Camera`.
    pub(crate) fn search_label(self) -> String {
        format!("{}: {}", self.category(), self.title())
    }

    /// The viewport keybinding that also runs this command, for display only.
    /// Every string here must appear verbatim in `help_window::HELP_BINDINGS`.
    pub(crate) const fn keybind(self) -> Option<&'static str> {
        match self {
            Self::CopyGraphAsBlog => Some("Ctrl + C (View)"),
            Self::DuplicateSelection => Some("Ctrl + D (View)"),
            Self::SetMode(EditorMode::View) => Some("V"),
            Self::SetMode(EditorMode::Module) => Some("M"),
            Self::SetMode(EditorMode::Bloq) => Some("C"),
            Self::FillPorts => Some("F"),
            _ => None,
        }
    }

    pub(crate) fn availability(self, ctx: &CommandContext<'_>) -> Availability {
        let needs_graph = matches!(
            self,
            Self::SaveBlog
                | Self::CopyGraphAsBlog
                | Self::ClearGraph
                | Self::DuplicateSelection
                | Self::ValidateGraph
                | Self::FillPorts
                | Self::FixShadowedFaces
                | Self::FlipXZBasis
                | Self::RandomlyResolveSelectives
        );
        let has_content = !ctx.graph_state.graph.is_empty()
            || (matches!(
                self,
                Self::SaveBlog | Self::CopyGraphAsBlog | Self::ClearGraph
            ) && ctx.graph_state.source_graph.is_some());
        if needs_graph && !has_content {
            return Availability::Unavailable("the graph is empty");
        }
        if ctx.graph_state.is_composed()
            && matches!(
                self,
                Self::SetMode(EditorMode::Edit)
                    | Self::InsertGallery(_)
                    | Self::DuplicateSelection
                    | Self::FillPorts
                    | Self::FixShadowedFaces
                    | Self::FlipXZBasis
                    | Self::RandomlyResolveSelectives
            )
        {
            return Availability::Unavailable("edit a definition in Modules, or open a flat copy");
        }
        match self {
            Self::ValidateGraph if ctx.jobs.validation_running() => {
                Availability::Unavailable("validation is already running")
            }
            Self::NextTab | Self::PreviousTab if ctx.tabs.tabs.len() < 2 => {
                Availability::Unavailable("only one tab is open")
            }
            Self::SetMode(mode) if ctx.editor_state.mode == mode => {
                Availability::Unavailable("already the active mode")
            }
            Self::DuplicateSelection
                if !ctx.graph_state.graph.blocks().any(|block| {
                    ctx.editor_state
                        .is_selected(GraphElement::Block(block.pos()))
                }) =>
            {
                Availability::Unavailable("select at least one block")
            }
            Self::SetPlaneHeight(None) => {
                Availability::Unavailable("type a height after the command, e.g. \"plane 3\"")
            }
            _ => Availability::Available,
        }
    }

    /// The intent this command queues. Only called for an available command, so
    /// the plane-height fallback keeps the mapping total without panicking.
    pub(crate) fn into_intent(self, ctx: &CommandContext<'_>) -> UiIntent {
        let editor_state = ctx.editor_state;
        match self {
            Self::OpenBlogFile => UiIntent::ImportBlogFromFile,
            Self::SaveBlog => UiIntent::ExportBlog {
                path: ctx.import_export.export_path.clone(),
            },
            Self::CopyGraphAsBlog => UiIntent::CopyGraphAsBlog,
            Self::ClearGraph => UiIntent::ClearGraph,
            Self::Screenshot => UiIntent::RequestScreenshot,
            Self::NewTab => UiIntent::NewTab,
            Self::NextTab => UiIntent::SelectTab(neighbour_tab(ctx.tabs, 1)),
            Self::PreviousTab => UiIntent::SelectTab(neighbour_tab(ctx.tabs, -1)),
            Self::CloseTab => UiIntent::RequestCloseTab(ctx.tabs.active.get()),
            Self::RenameTab => UiIntent::BeginRenameTab(ctx.tabs.active.get()),
            Self::SetMode(mode) => UiIntent::SetMode(mode),
            Self::InsertGallery(entry) => UiIntent::InsertGraph(Box::new(entry.build())),
            Self::DuplicateSelection => crate::systems::input::duplicate_selection_intent(
                &ctx.graph_state.graph,
                editor_state.selected_element_set(),
            ),
            Self::ValidateGraph => UiIntent::ValidateGraph,
            Self::FillPorts => UiIntent::FillPorts,
            Self::FixShadowedFaces => UiIntent::FixShadowedFaces,
            Self::FlipXZBasis => UiIntent::FlipXZBasis,
            Self::RandomlyResolveSelectives => UiIntent::RandomlyResolveSelectives,
            Self::ResetCamera => UiIntent::ResetCamera,
            Self::ToggleAxis => UiIntent::SetAxisVisibility(!editor_state.show_axis),
            Self::ToggleCurrentLayerOnly => {
                UiIntent::SetViewCurrentLayerOnly(!editor_state.view_current_layer_only)
            }
            Self::TogglePortTags => UiIntent::SetPortTagVisibility(!editor_state.show_port_tags),
            Self::SetPlaneHeight(height) => {
                UiIntent::SetPlaneHeight(height.unwrap_or(editor_state.plane_height))
            }
            Self::ToggleTheme => UiIntent::SetTheme(match editor_state.theme_preset {
                ThemePreset::Light => ThemePreset::GruvboxMaterial,
                ThemePreset::GruvboxMaterial => ThemePreset::Light,
            }),
            Self::ShowShortcuts => UiIntent::ShowShortcuts,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Quit => UiIntent::Quit,
        }
    }
}

/// The id of the tab `offset` positions from the active one, wrapping around.
fn neighbour_tab(tabs: &EditorTabs, offset: isize) -> u64 {
    let active = tabs
        .tabs
        .iter()
        .position(|tab| tab.id == tabs.active)
        .unwrap_or(0);
    let next = (active as isize + offset).rem_euclid(tabs.tabs.len() as isize) as usize;
    tabs.tabs[next].id.get()
}
