//! Deferred UI actions.
//!
//! Panel draw systems hold only shared references to most editor state, so
//! instead of mutating it inline they push a [`UiIntent`] onto the
//! [`UiIntentBuffer`]. A single later system drains the buffer with the mutable
//! access needed to apply each action, keeping the borrow rules simple and the
//! order of effects well-defined.

use crate::components::GraphElement;
use crate::resources::{CompileRequest, EditorMode, PlacementTool, SliceVisibility};
use crate::theme::ThemePreset;
use bevy::prelude::{Color, Resource};
use bloq_graph::{Action, BlockGraph, BlockKind, GalleryItem, UDirection};

/// A frame's queue of pending [`UiIntent`]s, drained by `apply_ui_intents_system`.
#[derive(Resource, Default)]
pub(crate) struct UiIntentBuffer {
    intents: Vec<UiIntent>,
}

impl UiIntentBuffer {
    /// Queues an intent to apply this frame.
    pub(crate) fn push(&mut self, intent: UiIntent) {
        self.intents.push(intent);
    }

    /// Drains the queued intents in submission order.
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = UiIntent> + '_ {
        self.intents.drain(..)
    }

    /// Queues an error toast.
    pub(crate) fn error(&mut self, message: impl Into<String>) {
        self.push(UiIntent::Error(message.into()));
    }
}

/// A single deferred UI action, applied by `apply_ui_intents_system`.
pub(crate) enum UiIntent {
    FinishTranslation {
        tab: crate::resources::EditorTabId,
        revision: u64,
        target: crate::systems::input::TranslationTarget,
        offset: glam::IVec3,
        compact: bool,
    },
    EditModule {
        revision: u64,
        edit: crate::module_authoring::ModuleEdit,
    },
    ImportModuleTab {
        tab: u64,
        name: String,
    },
    NewModule(String),
    OpenModule(String),
    ApplyModuleTab,
    OpenFlatCopy,
    WrapAsModule(String),
    InspectModuleInstance(String),
    InspectModuleElement(GraphElement),
    ShowBlogBuffer,
    EditSelection,
    Error(String),
    RerenderGraph,
    SetMode(EditorMode),
    SelectPlacementTool(PlacementTool),
    SelectPlacementBlock(BlockKind),
    SetBackgroundColor(Color),
    SetAxisVisibility(bool),
    SetViewCurrentLayerOnly(bool),
    SetPortTagVisibility(bool),
    SetPlaneHeight(i32),
    SetTheme(ThemePreset),
    ResetCamera,
    Undo,
    Redo,
    ClearGraph,
    NewTab,
    SelectTab(u64),
    RequestCloseTab(u64),
    ConfirmCloseTab(u64),
    BeginRenameTab(u64),
    FinishRenameTab(u64),
    CancelRenameTab(u64),
    #[cfg(not(target_arch = "wasm32"))]
    Quit,
    #[cfg(not(target_arch = "wasm32"))]
    ConfirmQuit,
    RequestScreenshot,
    /// Opens the keyboard-shortcut window.
    ShowShortcuts,
    /// Puts `text` on the system clipboard through egui's clipboard backend.
    CopyToClipboard(String),
    CopyGraphAsBlog,
    InsertGraph(Box<BlockGraph>),
    LoadGallery(GalleryItem),
    FixShadowedFaces,
    FillPorts,
    FlipXZBasis,
    RandomlyResolveSelectives,
    ValidateGraph,
    ToggleStabilizers {
        layer_only: bool,
        plane_height: i32,
    },
    CompileForViewer,
    CancelCompilation(crate::resources::EditorTabId),
    SetCircuitMoment(usize),
    SelectBloqNode(Option<u32>),
    ToggleBloqNode(u32),
    HoverBloqNode(Option<u32>),
    SetConcurrentOps(bool),
    SetConcurrentLayer(i32),
    SetShowClassical(bool),
    SetSliceVisibility(SliceVisibility),
    PinViewerBranches(Option<std::collections::BTreeMap<String, bool>>),
    CaptureBranchArm {
        captured_true: bool,
    },
    CancelBranchArm,
    ShowBranchArm {
        name: String,
        show_true: bool,
    },
    CompileGraph(CompileRequest),
    ExportBlog {
        path: String,
    },
    /// Export exact editor text without requiring a valid graph.
    SaveBlogDraft(String),
    ImportBlogFromFile,
    LoadBlogFile {
        title: Option<String>,
        buffer: String,
    },
    LoadBlogFromBuffer,
    StoreGraphToBuffer,
    /// Saves an exported SVG view through the shared download backend (native
    /// save dialog or web download). Not cfg-gated: both targets are supported.
    SaveSvg {
        file_name: String,
        contents: String,
    },
    ApplyElementEdit(ElementEditIntent),
    CloseElementEdit,
    /// Append, replace, or drop one graph action; the index is its source
    /// ordinal, which is what the action DAG's nodes carry.
    AddAction(Action),
    ReplaceAction {
        index: usize,
        expected: Vec<Action>,
        action: Action,
    },
    RemoveAction(usize),
    TranslateGraph {
        axis: UDirection,
        step: i32,
    },
    RotateGraph {
        axis: UDirection,
        quarter_turns: i32,
    },
}

impl UiIntent {
    /// Composed geometry is a preview. Edits must target its definitions.
    pub(crate) fn changes_flat_geometry(&self) -> bool {
        matches!(
            self,
            Self::InsertGraph(_)
                | Self::FixShadowedFaces
                | Self::FillPorts
                | Self::FlipXZBasis
                | Self::RandomlyResolveSelectives
                | Self::ApplyElementEdit(_)
                | Self::AddAction(_)
                | Self::ReplaceAction { .. }
                | Self::RemoveAction(_)
                | Self::TranslateGraph { .. }
                | Self::RotateGraph { .. }
                | Self::CaptureBranchArm { .. }
                | Self::CancelBranchArm
        )
    }
}

/// The edited attributes to commit for a block or pipe from the attribute editor.
pub(crate) struct ElementEditIntent {
    pub(crate) target: GraphElement,
    pub(crate) block_kind: BlockKind,
    pub(crate) block_color: Option<[u8; 3]>,
    pub(crate) port_role: bloq_graph::PortRole,
    pub(crate) tag: String,
    pub(crate) pipe_hadamard: bool,
}
