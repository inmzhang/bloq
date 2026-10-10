//! The compilation work the editor's background jobs run: lowering a graph to a
//! `Bloq`, then either serializing it to a download or building the viewer's
//! node/edge views.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use super::{CompileProgress, CompileProgressPhase};
use crate::program_view::{build_bloq_circuit_view, build_detslice_circuit_view};
use crate::resources::{
    CompileOutputFormat, CompileRequest, CompiledBloqView, DetectorBreaks, DetsliceData,
    FlatMoment, NodeSliceTimeline, SliceRegionId, SliceRegionView,
};
#[cfg(not(target_arch = "wasm32"))]
use bloq_compile::{CompileArtifacts, CompileContext};
use bloq_compile::{CompileConfig, compile_detslice_proxy};
use bloq_graph::BlockGraph;
use bloq_ir::{
    Bloq, BloqNodeId, LevelPath, NodeKey, NodeRef, NodeSlices, ProgramRegion, ProgramRegionId,
    ProgramSliceOptions, RegionView, SubGraph, program_detector_slices_with_options,
};
use bloq_stim::emit_bloq_stim;
use color_eyre::eyre::{self, ContextCompat, WrapErr};
use glam::{IVec2, IVec3};

#[derive(Debug)]
pub(super) struct CompiledDownload {
    file_name: String,
    contents: Vec<u8>,
}

impl CompiledDownload {
    #[cfg(test)]
    pub(super) fn empty() -> Self {
        Self {
            file_name: String::new(),
            contents: Vec::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_artifacts(
    source: &BlockGraph,
    request: &CompileRequest,
    progress: Option<&CompileProgress>,
) -> eyre::Result<(CompileArtifacts, CompileConfig)> {
    let config = config_from_request(request)?;
    let ctx = match progress {
        Some(progress) => CompileContext::new(config)
            .with_progress_observer(progress.observer())
            .with_cancellation(progress.cancellation.clone()),
        None => CompileContext::new(config),
    };

    // The compiler is the single authority on what it can lower, and `compile`
    // runs the same validation `verify` would, so we compile directly instead
    // of pre-verifying (a redundant second normalize-and-validate).
    let artifacts = ctx.compile(source).wrap_err("compile block graph")?;

    Ok((artifacts, config))
}

pub(super) fn config_from_request(request: &CompileRequest) -> eyre::Result<CompileConfig> {
    Ok(CompileConfig::try_new(request.code_distance)
        .wrap_err("build the compile config")?
        .with_prepare_t_with_mpps(request.prepare_t_with_mpps))
}

fn viewer_graph(graph: &BlockGraph) -> eyre::Result<BlockGraph> {
    Ok(graph.with_zero_min_z()?.fix_shadowed_faces())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn compile_graph_downloads(
    graph: &BlockGraph,
    request: &CompileRequest,
) -> eyre::Result<CompiledDownload> {
    compile_downloads(graph, request, None)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn compile_downloads(
    source: &BlockGraph,
    request: &CompileRequest,
    progress: Option<&CompileProgress>,
) -> eyre::Result<CompiledDownload> {
    let (artifacts, _) = compile_artifacts(source, request, progress)?;
    if let Some(progress) = progress {
        progress.set_phase(CompileProgressPhase::Exporting);
    }
    match progress {
        Some(progress) => progress
            .cancellation
            .run(|| export_download(artifacts.bloq, request)),
        None => export_download(artifacts.bloq, request),
    }
}

pub(super) fn export_download(
    bloq: Bloq,
    request: &CompileRequest,
) -> eyre::Result<CompiledDownload> {
    let file_name = format!("d={}.{}", request.code_distance, request.format.extension());
    let contents = match request.format {
        CompileOutputFormat::Stim => emit_bloq_stim(&bloq)
            .wrap_err("emit Stim text")?
            .into_bytes(),
        CompileOutputFormat::IrText => bloq.to_text().into_bytes(),
        CompileOutputFormat::IrBinary => bloq.to_binary(),
    };

    Ok(CompiledDownload {
        file_name,
        contents,
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn compile_graph_for_viewer(
    graph: &BlockGraph,
    request: &CompileRequest,
) -> eyre::Result<CompiledBloqView> {
    compile_for_viewer(graph, request, None)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn compile_for_viewer(
    source: &BlockGraph,
    request: &CompileRequest,
    progress: Option<&CompileProgress>,
) -> eyre::Result<CompiledBloqView> {
    let (artifacts, compile_config) = compile_artifacts(source, request, progress)?;
    if let Some(progress) = progress {
        progress.set_phase(CompileProgressPhase::BuildingView);
    }
    let build = || {
        build_view_from_artifacts(
            source,
            request,
            artifacts.bloq,
            compile_config,
            artifacts.compile_duration,
        )
    };
    match progress {
        Some(progress) => progress.cancellation.run(build),
        None => build(),
    }
}

pub(super) fn build_view_from_artifacts(
    source: &BlockGraph,
    request: &CompileRequest,
    bloq: Bloq,
    compile_config: CompileConfig,
    compile_duration: Duration,
) -> eyre::Result<CompiledBloqView> {
    let graph = source.flatten()?;
    let program = Arc::new(bloq);
    let source_viewer_graph = viewer_graph(&graph).wrap_err("normalize source viewer graph")?;
    let source_offset = IVec3::new(0, 0, graph.spans().map_or(0, |(_, _, z)| *z.start()));
    let view = build_bloq_circuit_view(&program, &source_viewer_graph, source_offset)
        .wrap_err("build Bloq circuit view")?;
    // Retain the program for optional views computed on their first toggle.
    Ok(CompiledBloqView {
        viewer_graph: source_viewer_graph,
        nodes: view.nodes,
        edges: view.edges,
        code_distance: request.code_distance,
        compile_config,
        compile_duration,
        source_program: program.clone(),
        branch_pin_draft: reachable_branch_pins(&program)?,
        program,
        branch_pins: None,
        source_offset,
    })
}

pub(super) fn pin_viewer_branches(
    mut view: CompiledBloqView,
    pins: Option<std::collections::BTreeMap<String, bool>>,
) -> eyre::Result<CompiledBloqView> {
    view.program = match &pins {
        Some(pins) => Arc::new(view.source_program.pin_membership(pins)?),
        None => view.source_program.clone(),
    };
    let built = build_bloq_circuit_view(&view.program, &view.viewer_graph, view.source_offset)?;
    view.nodes = built.nodes;
    view.edges = built.edges;
    view.branch_pin_draft = match &pins {
        Some(pins) => pins.clone(),
        None => reachable_branch_pins(&view.source_program)?,
    };
    view.branch_pins = pins;
    Ok(view)
}

/// Prefer false, keeping every chosen prefix reachable. Runs with the viewer
/// build so the UI thread only installs the completed choices.
fn reachable_branch_pins(program: &Bloq) -> eyre::Result<std::collections::BTreeMap<String, bool>> {
    let mut analysis = bloq_ir::lowering::PredicateAnalysis::new(program.top());
    let mut assignment = Vec::new();
    let mut pins = std::collections::BTreeMap::new();
    for (id, node) in program.nodes() {
        if let bloq_ir::NodeProvenance::BranchSelector { name } = &node.provenance {
            assignment.push((id, false));
            let value = !analysis.assignment_reachable(&assignment)?;
            assignment.last_mut().expect("just added selector").1 = value;
            pins.insert(name.clone(), value);
        }
    }
    Ok(pins)
}

type StableNodeOccurrence = (NodeKey, u32);

struct SliceProgram {
    program: Bloq,
    /// Slice-program node references translated to ids in the real viewer.
    node_refs: HashMap<NodeRef, u32>,
}

/// Build the pinned proxy used when the source graph contains a dynamic block.
/// Stable provenance keys line its quantum nodes up with the pinned program.
fn build_proxy_slice_program(
    program: &Bloq,
    flat_program: &Bloq,
    source_viewer_graph: &BlockGraph,
    compile_config: CompileConfig,
    viewer_node_refs: &HashMap<NodeRef, u32>,
) -> eyre::Result<SliceProgram> {
    let pins = selected_selective_pins(program, source_viewer_graph)?;
    let proxy = compile_detslice_proxy(compile_config, source_viewer_graph, &pins)
        .wrap_err("compile the selected Clifford proxy")?
        .bloq;

    let mut flat_proxy = proxy.clone();
    flat_proxy
        .flatten()
        .wrap_err("flatten the selected Clifford proxy")?;
    let real_keys = selected_quantum_node_keys(flat_program)?;
    let proxy_keys = selected_quantum_node_keys(&flat_proxy)?;
    let mut node_refs = HashMap::with_capacity(proxy_keys.len());
    for (key, proxy_ref) in proxy_keys {
        let real_ref = real_keys.get(&key).wrap_err_with(|| {
            format!("proxy node key {:?} has no selected-path counterpart", key)
        })?;
        let &view_id = viewer_node_refs
            .get(real_ref)
            .wrap_err_with(|| format!("selected-path node {real_ref:?} has no viewer tile"))?;
        node_refs.insert(proxy_ref, view_id);
    }

    Ok(SliceProgram {
        program: proxy,
        node_refs,
    })
}

/// Resolve pinned named selective outcomes into the compiler's source order.
fn selected_selective_pins(program: &Bloq, graph: &BlockGraph) -> eyre::Result<Vec<bool>> {
    let mut positions = graph
        .blocks()
        .filter(|block| block.kind().is_selective())
        .map(bloq_graph::Block::pos)
        .collect::<Vec<_>>();
    positions.sort_unstable_by_key(|pos| (pos.z, pos.x, pos.y));
    positions
        .into_iter()
        .map(|position| {
            let name = bloq_graph::selective_selector_name(position);
            program
                .nodes()
                .find_map(|(_, node)| match (&node.provenance, node.try_classical()) {
                    (
                        bloq_ir::NodeProvenance::BranchSelector { name: selector },
                        Some(bloq_ir::ClassicalNode::Compute { expr }),
                    ) if selector == &name => expr.eval(&mut |_| None),
                    _ => None,
                })
                .wrap_err_with(|| format!("{name} requires a pinned outcome"))
        })
        .collect()
}

/// Key every quantum node along one executed path. The occurrence component
/// disambiguates the rare case where the same level-local stable key appears in
/// more than one nested level.
fn selected_quantum_node_keys(
    program: &Bloq,
) -> eyre::Result<HashMap<StableNodeOccurrence, NodeRef>> {
    let mut occurrences = HashMap::new();
    let mut out = HashMap::new();
    collect_selected_quantum_node_keys(
        program.top(),
        &LevelPath::default(),
        &mut occurrences,
        &mut out,
    )?;
    Ok(out)
}

fn collect_selected_quantum_node_keys(
    level: &SubGraph,
    path: &LevelPath,
    occurrences: &mut HashMap<NodeKey, u32>,
    out: &mut HashMap<StableNodeOccurrence, NodeRef>,
) -> eyre::Result<()> {
    let keys = level
        .stable_keys()
        .wrap_err("derive compile-stable node keys")?
        .into_iter()
        .map(|(key, id)| (id, key))
        .collect::<HashMap<BloqNodeId, NodeKey>>();
    for id in level
        .deterministic_emit_order()
        .wrap_err("order selected proxy path")?
    {
        let node = level
            .node(id)
            .expect("deterministic emit order only returns live nodes");
        if node.try_quantum().is_some()
            && let Some(key) = keys.get(&id).cloned()
        {
            let occurrence = occurrences.entry(key.clone()).or_default();
            let identity = (key, *occurrence);
            *occurrence += 1;
            out.insert(
                identity,
                NodeRef {
                    path: path.clone(),
                    node: id,
                },
            );
        }

        let Some(region) = node.try_region() else {
            continue;
        };
        for (body, child) in region.bodies() {
            collect_selected_quantum_node_keys(child, &path.child(id, body), occurrences, out)?;
        }
    }
    Ok(())
}

/// Builds the detector-slice store for the viewer.
///
/// Two passes over the *flattened* program, which must agree on how ops split
/// into moments (they share `bloq_ir::detslice::moment_segments`):
///
/// 1. `build_bloq_circuit_view` yields the per-node flattened timeline the
///    overlay steps through, plus a `NodeRef → view id` map. Node ids are stable
///    across `Bloq::flatten` (it only unrolls templates and lifts region
///    bodies), so these line up with the default view's nodes.
/// 2. `program_detector_slices` (Layer 2) tracks each detector's region back
///    through the tape. It flattens internally, so its `NodeRef` space matches
///    pass 1; we translate `per_node` to view-id keys and grid-coord terms.
///
/// A flatten or view failure disables the overlay (reason stored) while leaving
/// the default view intact; a Layer-2 failure keeps the flattened timelines but
/// disables the region overlay.
pub(crate) fn build_detslice(
    program: &Bloq,
    source_viewer_graph: &BlockGraph,
    source_offset: IVec3,
    compile_config: CompileConfig,
) -> DetsliceData {
    let mut flat = program.clone();
    if let Err(err) = flat.flatten() {
        return DetsliceData {
            unavailable_reason: Some(format!("Flattening failed: {err}")),
            ..Default::default()
        };
    }
    let flat_view = match build_detslice_circuit_view(&flat, source_viewer_graph, source_offset) {
        Ok(view) => view,
        Err(err) => {
            return DetsliceData {
                unavailable_reason: Some(format!("{err:#}")),
                ..Default::default()
            };
        }
    };

    // Every node's flat moment count (for the alignment check) before filtering
    // out the moment-less nodes the overlay never steps through.
    let flat_counts: HashMap<u32, usize> = flat_view
        .nodes
        .iter()
        .map(|node| (node.id, node.moments.len()))
        .collect();
    // Each view node's own qubit coordinates, used to drop regions that have no
    // support on the viewed node (they would render as disconnected floating
    // patches). Straddling regions keep their full extent.
    let node_qubits: HashMap<u32, HashSet<IVec2>> = flat_view
        .nodes
        .iter()
        .map(|node| (node.id, node.qubit_coords.values().copied().collect()))
        .collect();
    let viewer_node_refs = flat_view.node_refs;
    let flat_moments: HashMap<u32, Vec<FlatMoment>> = flat_view
        .nodes
        .into_iter()
        .filter(|node| !node.moments.is_empty())
        .map(|node| (node.id, node.moments))
        .collect();

    let has_dynamic_blocks = source_viewer_graph
        .blocks()
        .any(|block| block.kind().is_dynamic());
    let slice_program = if has_dynamic_blocks {
        match build_proxy_slice_program(
            program,
            &flat,
            source_viewer_graph,
            compile_config,
            &viewer_node_refs,
        ) {
            Ok(input) => input,
            Err(err) => {
                return DetsliceData {
                    flat_moments,
                    unavailable_reason: Some(format!("selected Clifford proxy: {err:#}")),
                    ..Default::default()
                };
            }
        }
    } else {
        SliceProgram {
            program: program.clone(),
            node_refs: viewer_node_refs,
        }
    };

    // The tape flattens internally. Fixed programs share the real view's
    // `NodeRef` space; a dynamic program uses the provenance map built above to
    // translate its pinned proxy nodes back onto that real view.
    let mut options = ProgramSliceOptions::default();
    options.set_physical_observables(true);
    let slices = match program_detector_slices_with_options(&slice_program.program, &options) {
        Ok(slices) => slices,
        Err(err) => {
            return DetsliceData {
                flat_moments,
                unavailable_reason: Some(err.to_string()),
                ..Default::default()
            };
        }
    };

    let mut detector_slices = HashMap::new();
    let skipped_cross_tape = slices.skipped_cross_tape;
    let broken = slices.breaks.len();
    let regions = slices.regions;
    let logical_note = regions
        .iter()
        .any(|region| matches!(region.id, ProgramRegionId::Observable { .. }))
        .then(|| "Logical slices: physical support; runtime sign folds omitted".to_string());
    let empty_qubits = HashSet::new();
    for (node_ref, node_slices) in slices.per_node {
        let Some(&view_id) = slice_program.node_refs.get(&node_ref) else {
            // Every quantum NodeRef the tape emits should have a view tile; a
            // miss means the two passes disagree on structure (a bug).
            debug_assert!(false, "flat view has no tile for tape node {node_ref:?}");
            continue;
        };
        // Shared bucketing makes the tape's per-node moment count equal the flat
        // timeline's. A mismatch is a bucketing bug; skip the node in release so
        // the overlay never draws regions against the wrong moment.
        let expected = flat_counts.get(&view_id).copied().unwrap_or(0);
        if node_slices.0.len() != expected {
            debug_assert!(
                false,
                "node {view_id}: {} slice moments vs {expected} flat moments",
                node_slices.0.len()
            );
            continue;
        }
        let own_qubits = node_qubits.get(&view_id).unwrap_or(&empty_qubits);
        detector_slices.insert(
            view_id,
            convert_node_timeline(node_slices, &regions, &slice_program.node_refs, own_qubits),
        );
    }

    let mut detector_breaks: DetectorBreaks = HashMap::new();
    for brk in slices.breaks {
        if let Some(&view_id) = slice_program.node_refs.get(&brk.node) {
            // Drop break markers that land outside the viewed node's qubits, for
            // the same reason regions are filtered: they would float free.
            if !node_qubits
                .get(&view_id)
                .is_some_and(|qs| qs.contains(&brk.qubit))
            {
                continue;
            }
            detector_breaks
                .entry((view_id, brk.moment))
                .or_default()
                .push(brk.qubit);
        }
    }

    let note = [logical_note, detslice_note(skipped_cross_tape, broken)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ");
    DetsliceData {
        flat_moments,
        detector_slices,
        detector_breaks,
        unavailable_reason: None,
        note: (!note.is_empty()).then_some(note),
    }
}

/// Translates one node's tape slices into the view-space timeline the overlay
/// draws: detector-owner `NodeRef`s become view ids and Pauli terms carry grid
/// coordinates.
///
/// Regions with no support on `own_qubits` (the viewed node's own coordinates)
/// are dropped: they belong entirely to another node and would render as
/// disconnected floating patches outside this node's outline. A region that
/// merely straddles the seam keeps its full extent, including the terms that lie
/// outside `own_qubits` — a detector's hover tooltip names its owning node.
fn convert_node_timeline(
    node_slices: NodeSlices,
    regions: &[ProgramRegion],
    node_refs: &HashMap<NodeRef, u32>,
    own_qubits: &HashSet<IVec2>,
) -> NodeSliceTimeline {
    node_slices
        .0
        .into_iter()
        .map(|moment_regions| {
            moment_regions
                .into_iter()
                .filter(|region| region_touches(region, own_qubits))
                .filter_map(|region| convert_region(region, regions, node_refs))
                .collect()
        })
        .collect()
}

/// Whether any of the region's Pauli terms sits on one of the viewed node's own
/// qubits.
fn region_touches(region: &RegionView, own_qubits: &HashSet<IVec2>) -> bool {
    region
        .terms
        .iter()
        .any(|term| own_qubits.contains(&term.qubit))
}

/// One tape region as the overlay stores it, or `None` if a detector owner has
/// no view tile (a structural disagreement, asserted in debug builds).
fn convert_region(
    region: RegionView,
    regions: &[ProgramRegion],
    node_refs: &HashMap<NodeRef, u32>,
) -> Option<SliceRegionView> {
    let metadata = &regions[region.region as usize];
    let id = match &metadata.id {
        ProgramRegionId::Detector { owner, detector } => {
            let Some(&owner_node) = node_refs.get(owner) else {
                debug_assert!(false, "region owner {owner:?} has no view tile");
                return None;
            };
            SliceRegionId::Detector {
                owner_node,
                detector: *detector,
            }
        }
        ProgramRegionId::Observable { index } => SliceRegionId::Observable { index: *index },
    };
    Some(SliceRegionView {
        id,
        coords: metadata.coords.clone(),
        terms: region.terms,
    })
}

/// A short footer note when regions are dropped at a region-node seam (a
/// limitation) or hit a gauge break, so users are not surprised by missing
/// regions. T-family gates no longer break; they track through their Clifford
/// proxy.
fn detslice_note(skipped_cross_tape: usize, broken: usize) -> Option<String> {
    let total = skipped_cross_tape + broken;
    (total > 0).then(|| format!("{total} regions break (seam)"))
}

/// Saves a compiled download through the shared file-save backend. `None` means
/// the user canceled the save dialog — not an error.
pub(super) fn save_compiled_downloads(output: &CompiledDownload) -> eyre::Result<Option<String>> {
    crate::utils::save_download(&output.file_name, &output.contents)
}

#[cfg(test)]
mod tests {
    use super::{
        build_detslice, compile_graph_downloads, compile_graph_for_viewer, selected_selective_pins,
    };
    use crate::resources::{
        BloqNodeCategory, CompileOutputFormat, CompileRequest, CompiledBloqView, DetsliceData,
        FlatMoment, FlatMomentOp, SliceRegionId,
    };
    use bloq_graph::{Action, Block, BlockGraph, BlockKind, CubeKind, GalleryItem, MeasureTarget};
    use bloq_ir::MomentKind;
    use glam::ivec3;

    /// Runs the lazy detector-slice build off a compiled view, mirroring what the
    /// `detslice` job does on first enable. The compile itself no longer produces
    /// this data (it is computed here on demand).
    fn detslice_for(view: &CompiledBloqView) -> DetsliceData {
        build_detslice(
            &view.program,
            &view.viewer_graph,
            view.source_offset,
            view.compile_config,
        )
    }

    /// Choose a complete reachable tuple while preferring either bit, including
    /// selectors constrained to be equal or complementary by their source.
    fn pin_reachable_view(view: CompiledBloqView, preferred: bool) -> CompiledBloqView {
        let mut analysis = bloq_ir::lowering::PredicateAnalysis::new(view.source_program.top());
        let mut assignment = Vec::new();
        let mut pins = std::collections::BTreeMap::new();
        for (id, node) in view.source_program.nodes() {
            if let bloq_ir::NodeProvenance::BranchSelector { name } = &node.provenance {
                assignment.push((id, preferred));
                let choice = if analysis.assignment_reachable(&assignment).unwrap() {
                    preferred
                } else {
                    assignment.last_mut().unwrap().1 = !preferred;
                    !preferred
                };
                pins.insert(name.clone(), choice);
            }
        }
        super::pin_viewer_branches(view, Some(pins)).expect("reachable viewer choices pin")
    }

    fn single_cube_graph() -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .add_action(Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m".to_string(),
            })
            .expect("test action should apply");
        graph
    }

    #[test]
    fn compile_graph_download_emits_stim_output() {
        let output = compile_graph_downloads(
            &single_cube_graph(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("compile stim output");

        assert_eq!(output.file_name, "d=3.stim");
        let contents = std::str::from_utf8(&output.contents).expect("stim is UTF-8 text");
        assert!(contents.contains("QUBIT_COORDS"));
    }

    #[test]
    fn compile_graph_download_emits_parseable_ir_exchange_formats() {
        for format in [CompileOutputFormat::IrText, CompileOutputFormat::IrBinary] {
            let output = compile_graph_downloads(
                &single_cube_graph(),
                &CompileRequest {
                    code_distance: 3,
                    format,
                    prepare_t_with_mpps: false,
                },
            )
            .expect("compile IR output");

            assert_eq!(
                output.file_name,
                format!("d=3.{}", format.extension()),
                "file name carries the format extension"
            );
            let restored = match format {
                CompileOutputFormat::IrText => bloq_ir::Bloq::from_text(
                    std::str::from_utf8(&output.contents).expect("IR text is UTF-8"),
                )
                .expect("IR text parses"),
                CompileOutputFormat::IrBinary => {
                    bloq_ir::Bloq::from_binary(&output.contents).expect("IR binary decodes")
                }
                CompileOutputFormat::Stim => unreachable!("not under test"),
            };
            restored.validate().expect("restored program validates");
        }
    }

    #[test]
    fn compile_graph_download_accepts_temporal_open_graphs() {
        // CNOT's open form has only temporal ports onto cubes, so it compiles
        // straight through as ideal noiseless boundaries — no fill required.
        let graph = GalleryItem::CNOT.build();

        let output = compile_graph_downloads(
            &graph,
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("temporal open graph should compile");

        assert!(!output.contents.is_empty());
    }

    #[test]
    fn compile_graph_for_viewer_lifts_t_block_rus_region() {
        let view = compile_graph_for_viewer(
            &GalleryItem::T.build(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("t_gate compiles for viewer");

        let region = view
            .nodes
            .iter()
            .find(|node| node.category == BloqNodeCategory::Region)
            .expect("compiled T program surfaces its RUS region tile");
        assert!(region.has_quantum_body);
        assert!(!region.attributes.is_empty());

        let children: Vec<_> = view
            .nodes
            .iter()
            .filter(|node| node.parent == Some(region.id))
            .collect();
        let quantum_children: Vec<_> = children
            .iter()
            .filter(|node| node.category.is_quantum())
            .collect();
        assert_eq!(
            quantum_children.len(),
            2,
            "cultivation + escape body nodes lift into the view"
        );
        assert!(
            quantum_children.iter().all(|node| !node.moments.is_empty()),
            "body quantum nodes stay inspectable down to circuit moments"
        );
        assert!(
            view.edges.iter().any(|edge| {
                let endpoints = [edge.from, edge.to];
                quantum_children
                    .iter()
                    .all(|node| endpoints.contains(&node.id))
            }),
            "the cultivation → escape seam edge is lifted with remapped ids"
        );
        assert!(
            children
                .iter()
                .any(|node| node.category == BloqNodeCategory::Observable),
            "GAP observables ride along for the classical toggle"
        );
        let view = pin_reachable_view(view, false);
        let detslice = detslice_for(&view);

        // The T factory flattens and tracks without disabling the overlay; any
        // non-Clifford breaks or region-seam drops are surfaced, not crashed on.
        assert!(
            detslice.unavailable_reason.is_none(),
            "{:?}",
            detslice.unavailable_reason
        );
        assert!(!detslice.flat_moments.is_empty());
        assert!(!detslice.detector_slices.is_empty());
        // If the tape dropped regions (T axis / RUS seam), the footer note says
        // so; either way the store is well-formed and aligned.
        for (view_id, timeline) in &detslice.detector_slices {
            let flat_len = detslice
                .flat_moments
                .get(view_id)
                .map(Vec::len)
                .unwrap_or(0);
            assert_eq!(timeline.len(), flat_len);
        }
    }

    #[test]
    fn lazy_detslice_builds_flat_moments_and_slice_store() {
        let view = compile_graph_for_viewer(
            &single_cube_graph(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("compile viewer output");

        assert!(!view.nodes.is_empty());
        assert_eq!(view.code_distance, 3);
        assert!(
            view.nodes
                .iter()
                .all(|node| node.qubit_coords.len() == node.num_qubits)
        );

        // The compile step retains the program for the lazy job but produces no
        // detector-slice data itself; the overlay is built on demand below.
        let detslice = detslice_for(&view);

        assert!(
            !detslice.flat_moments.is_empty(),
            "detector-slice mode needs a flattened timeline per node"
        );
        assert!(
            detslice.unavailable_reason.is_none(),
            "a repeat-free program flattens cleanly"
        );
        assert!(
            !detslice.detector_slices.is_empty(),
            "the Layer-2 tape populates the slice store"
        );
        // With no repeats to unroll, each node's flattened timeline matches its
        // default moment list one-for-one.
        for node in &view.nodes {
            if let Some(flat) = detslice.flat_moments.get(&node.id) {
                assert_eq!(flat.len(), node.moments.len());
            }
        }
        // Every node's slice timeline is aligned 1:1 with its flattened moments.
        for (view_id, timeline) in &detslice.detector_slices {
            let flat_len = detslice
                .flat_moments
                .get(view_id)
                .map(Vec::len)
                .unwrap_or(0);
            assert_eq!(
                timeline.len(),
                flat_len,
                "node {view_id} slice/flat-moment length mismatch"
            );
        }
        // A d=3 surface code memory patch has weight-4 plaquette detectors.
        let max_terms = detslice
            .detector_slices
            .values()
            .flatten()
            .flatten()
            .map(|region| region.terms.len())
            .max()
            .unwrap_or(0);
        assert!(
            max_terms >= 4,
            "a surface code memory patch should surface a >=4-term plaquette region, saw {max_terms}"
        );
        assert!(
            detslice
                .detector_slices
                .values()
                .flatten()
                .flatten()
                .any(|region| matches!(region.id, SliceRegionId::Observable { .. })),
            "logical observables should appear as L regions"
        );
    }

    #[test]
    fn ccz_gate_teleport_keeps_logical_support_with_nonlinear_feedback() {
        let view = compile_graph_for_viewer(
            &GalleryItem::CCZGateTeleport.build(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("CCZ gate teleport compiles for viewer");
        for choice in [false, true] {
            let view = pin_reachable_view(view.clone(), choice);
            let slices = detslice_for(&view);
            assert!(
                slices.unavailable_reason.is_none(),
                "{:?}",
                slices.unavailable_reason
            );
            assert!(
                slices
                    .note
                    .as_deref()
                    .is_some_and(|note| note.contains("runtime sign folds omitted"))
            );
            assert!(
                slices
                    .detector_slices
                    .values()
                    .flatten()
                    .flatten()
                    .any(|region| matches!(region.id, SliceRegionId::Observable { .. }))
            );
            for (id, timeline) in &slices.detector_slices {
                assert_eq!(
                    timeline.len(),
                    slices.flat_moments.get(id).map_or(0, Vec::len)
                );
            }
        }
    }

    #[test]
    fn toffoli_detslice_proxy_recomputes_break_free_logical_observables() {
        let view = compile_graph_for_viewer(
            &GalleryItem::ToffoliFromAndDelayedCZ.build(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("Toffoli compiles for viewer");

        for choice in [false, true] {
            let pins = view
                .source_program
                .nodes()
                .filter_map(|(_, node)| match &node.provenance {
                    bloq_ir::NodeProvenance::BranchSelector { name } => {
                        Some((name.clone(), choice))
                    }
                    _ => None,
                })
                .collect();
            let pinned = super::pin_viewer_branches(view.clone(), Some(pins)).unwrap();
            assert!(
                selected_selective_pins(&pinned.program, &pinned.viewer_graph)
                    .unwrap()
                    .into_iter()
                    .all(|pin| pin == choice)
            );
            let name = if choice { "all true" } else { "all false" };
            let detslice = detslice_for(&pinned);
            assert!(
                detslice.unavailable_reason.is_none(),
                "{name}: {:?}",
                detslice.unavailable_reason
            );
            assert!(
                detslice.detector_breaks.is_empty(),
                "{name}: recomputed proxy observables must not break"
            );
            assert!(
                detslice
                    .detector_slices
                    .values()
                    .flatten()
                    .flatten()
                    .any(|region| { region.id == SliceRegionId::Observable { index: 12 } }),
                "{name}: logical region L12 remains visible"
            );
        }
        let defaults: std::collections::BTreeMap<_, _> = view
            .source_program
            .nodes()
            .filter_map(|(_, node)| match &node.provenance {
                bloq_ir::NodeProvenance::BranchSelector { name } => Some((name.clone(), false)),
                _ => None,
            })
            .collect();
        let mut reachable_single_toggles = 0;
        let mut rejected_single_toggles = 0;
        for name in defaults.keys() {
            let mut pins = defaults.clone();
            pins.insert(name.clone(), true);
            match super::pin_viewer_branches(view.clone(), Some(pins)) {
                Ok(_) => reachable_single_toggles += 1,
                Err(error) => {
                    assert!(matches!(
                        error.downcast_ref::<bloq_ir::MembershipPinError>(),
                        Some(bloq_ir::MembershipPinError::UnreachableAssignment)
                    ));
                    rejected_single_toggles += 1;
                }
            }
        }
        assert_eq!(reachable_single_toggles, 3);
        assert_eq!(rejected_single_toggles, 2);
    }

    #[test]
    fn lazy_detslice_shows_mpp_measurement_moments() {
        // CNOT's open temporal ports measure patch stabilizers with `Op::MPP`;
        // the shared bucketing puts them in a Measurement moment in both the
        // default and the flattened timelines.
        let view = compile_graph_for_viewer(
            &GalleryItem::CNOT.build(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::Stim,
                prepare_t_with_mpps: false,
            },
        )
        .expect("CNOT compiles for viewer");

        let detslice = detslice_for(&view);

        let has_mpp = |moments: &[FlatMoment]| {
            moments.iter().any(|moment| {
                moment.kind == MomentKind::Measurement
                    && moment
                        .ops
                        .iter()
                        .any(|op| matches!(op, FlatMomentOp::Mpp { .. }))
            })
        };

        assert!(
            view.nodes.iter().any(|node| has_mpp(&node.moments)),
            "an MPP boundary shows a Measurement moment in the default timeline"
        );
        assert!(
            detslice
                .flat_moments
                .values()
                .any(|moments| has_mpp(moments)),
            "and in the flattened timeline the overlay steps through"
        );
    }

    #[test]
    fn composed_adder_view_shows_conditional_components_without_pinning() {
        let view = super::compile_graph_for_viewer(
            &GalleryItem::ThreeBitAdder.build(),
            &CompileRequest {
                code_distance: 3,
                format: CompileOutputFormat::IrText,
                prepare_t_with_mpps: false,
            },
        )
        .expect("composed adder opens in Program view");

        let guarded = view
            .program
            .nodes()
            .filter(|(_, node)| node.try_quantum().is_some_and(|q| !q.guards.is_empty()))
            .collect::<Vec<_>>();
        assert!(!guarded.is_empty(), "the program retains its selectors");
        for (id, node) in guarded {
            let tile = view.nodes.iter().find(|tile| tile.id == id.0).unwrap();
            assert!(tile.quantum_visible());
            assert!(tile.moments.is_empty(), "alternatives must not be merged");
            assert!(!tile.attributes.is_empty());
            assert_eq!(
                tile.num_qubits,
                view.program.node_qubits(node).unwrap().len()
            );
            assert!(view.edges.iter().any(|edge| edge.to == id.0));
        }
        assert!(view.nodes.iter().any(|tile| !tile.moments.is_empty()));
        let error = super::build_detslice_circuit_view(
            &view.program,
            &view.viewer_graph,
            view.source_offset,
        )
        .expect_err("static timelines still need selector values");
        assert!(matches!(
            error.downcast_ref::<bloq_ir::lowering::NodeTemplateInstanceMergeError>(),
            Some(bloq_ir::lowering::NodeTemplateInstanceMergeError::MembershipSelectionRequired)
        ));

        let original = view.source_program.to_text();
        let mut viewer = crate::resources::BloqViewerState::default();
        viewer.finish_compile(1, view);
        assert!(!viewer.branch_pin_draft.is_empty());
        for choice in [false, true] {
            let pinned = if choice {
                pin_reachable_view(viewer.pinning_input().unwrap(), true)
            } else {
                super::pin_viewer_branches(
                    viewer.pinning_input().unwrap(),
                    Some(viewer.branch_pin_draft.clone()),
                )
                .expect("initial Apply pins uses a reachable draft")
            };
            assert!(!pinned.program.has_conditional_membership());
            assert_eq!(pinned.source_program.to_text(), original);
            assert!(pinned.nodes.iter().any(|node| !node.moments.is_empty()));
            assert!(
                pinned
                    .nodes
                    .iter()
                    .filter(|node| node.category.is_quantum())
                    .all(|node| node.attributes.is_empty())
            );
            let old_generation = viewer.compile_generation;
            viewer.finish_compile(1, pinned);
            assert_ne!(viewer.compile_generation, old_generation);
            assert_eq!(viewer.branch_pins.as_ref(), Some(&viewer.branch_pin_draft));
            assert!(std::sync::Arc::ptr_eq(
                &viewer.concurrent_ops_inputs().unwrap().program,
                viewer.program.as_ref().unwrap(),
            ));
            assert!(std::sync::Arc::ptr_eq(
                &viewer.detslice_inputs().unwrap().program,
                viewer.program.as_ref().unwrap(),
            ));
        }
        let cleared = super::pin_viewer_branches(viewer.pinning_input().unwrap(), None).unwrap();
        assert_eq!(cleared.program.to_text(), original);
        viewer.finish_compile(1, cleared);
        assert!(viewer.branch_pins.is_none());
        viewer
            .program
            .as_ref()
            .unwrap()
            .pin_membership(&viewer.branch_pin_draft)
            .expect("Clear pins restores a reachable draft");
        assert!(
            viewer
                .program
                .as_ref()
                .unwrap()
                .has_conditional_membership()
        );
    }

    #[test]
    fn compile_graph_for_viewer_accepts_walking_soft_corridor_overlap() {
        let variants = GalleryItem::GHZSlideThenGlide
            .build()
            .flatten()
            .unwrap()
            .fill_ports_auto()
            .expect("GHZ slide-then-glide fills");
        assert!(!variants.is_empty());

        for (variant, (graph, _)) in variants.iter().enumerate() {
            let view = compile_graph_for_viewer(
                graph,
                &CompileRequest {
                    code_distance: 3,
                    format: CompileOutputFormat::Stim,
                    prepare_t_with_mpps: false,
                },
            )
            .unwrap_or_else(|error| panic!("fill variant {variant}: {error}"));

            assert!(!view.nodes.is_empty(), "fill variant {variant}");
        }
    }

    #[test]
    fn cross_node_region_kept_only_when_it_touches_the_viewed_node() {
        use super::{IVec2, convert_node_timeline, region_touches};
        use bloq_circuit::{PauliBasis, RegionTerm};
        use bloq_ir::{
            BloqNodeId, NodeRef, NodeSlices, ProgramRegion, ProgramRegionId, RegionView,
        };
        use glam::ivec2;
        use std::collections::{HashMap, HashSet};

        let own_qubits: HashSet<IVec2> = [ivec2(0, 0), ivec2(1, 0)].into_iter().collect();
        let owner = NodeRef::top_level(BloqNodeId(5));
        let node_refs: HashMap<NodeRef, u32> = [(owner.clone(), 5)].into_iter().collect();
        let term = |q, pauli| RegionTerm { qubit: q, pauli };

        // Straddles the seam: one term on the viewed node, one on a neighbour.
        let regions = vec![
            ProgramRegion {
                id: ProgramRegionId::Detector {
                    owner: owner.clone(),
                    detector: 0,
                },
                coords: None,
            },
            ProgramRegion {
                id: ProgramRegionId::Detector { owner, detector: 1 },
                coords: None,
            },
        ];
        let straddling = RegionView {
            region: 0,
            terms: vec![
                term(ivec2(1, 0), PauliBasis::Z),
                term(ivec2(9, 0), PauliBasis::Z),
            ],
        };
        // Lives entirely on another node — a floating patch we drop.
        let external = RegionView {
            region: 1,
            terms: vec![
                term(ivec2(9, 0), PauliBasis::Z),
                term(ivec2(9, 1), PauliBasis::Z),
            ],
        };
        assert!(region_touches(&straddling, &own_qubits));
        assert!(!region_touches(&external, &own_qubits));

        let timeline = convert_node_timeline(
            NodeSlices(vec![vec![straddling, external]]),
            &regions,
            &node_refs,
            &own_qubits,
        );

        // Only the straddling region survives, and it keeps its full extent —
        // including the term on the neighbouring node's qubit.
        assert_eq!(timeline.len(), 1);
        assert_eq!(timeline[0].len(), 1);
        assert_eq!(
            timeline[0][0].id,
            SliceRegionId::Detector {
                owner_node: 5,
                detector: 0,
            }
        );
        assert_eq!(timeline[0][0].terms.len(), 2);
    }
}
