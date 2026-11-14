//! Hover/selection highlight overlay syncing.
use super::*;

pub(crate) fn sync_interaction_highlight_system(
    editor_state: Res<EditorState>,
    circuit_viewer: Res<BloqViewerState>,
    graph_state: Res<GraphState>,
    render_state: Res<GraphRenderState>,
    mut previous: Local<InteractionHighlightCache>,
    children_query: Query<&Children>,
    mut material_query: Query<(&mut MeshMaterial3d<StandardMaterial>, &OriginalMaterial)>,
) {
    let _span = info_span!("editor.sync_interaction_highlight_system").entered();
    let previous = &mut *previous;
    previous.current_hovered.clear();
    previous
        .current_hovered
        .extend(editor_state.blog_hovered_element_set().iter().copied());
    previous
        .current_hovered
        .extend(circuit_viewer.hovered_source_elements_iter());
    previous
        .current_hovered
        .extend(editor_state.zx_hovered_element_set().iter().copied());
    previous
        .current_hovered
        .extend(editor_state.action_hovered_elements.iter().copied());
    if let Some(element) = editor_state.hovered_element_for_highlight() {
        previous.current_hovered.insert(element);
    }
    let current_selected = selected_elements_for_highlight(&editor_state, &graph_state.graph);
    let current_candidates = &editor_state.action_candidate_elements;
    let selection_changed = previous.selected != *current_selected;
    let candidates_changed = previous.candidates != *current_candidates;
    let graph_revision = graph_state.revision;
    if previous.graph_revision == graph_revision
        && previous.hovered == previous.current_hovered
        && !selection_changed
        && !candidates_changed
    {
        return;
    }

    let mut changed = std::mem::take(&mut previous.changed);
    changed.clear();
    collect_changed_interaction_elements(
        &previous.hovered,
        &previous.selected,
        &previous.current_hovered,
        &current_selected,
        &mut changed,
    );
    changed.extend(
        previous
            .candidates
            .symmetric_difference(current_candidates)
            .copied(),
    );
    if previous.graph_revision != graph_revision {
        changed.extend(previous.current_hovered.iter().copied());
        changed.extend(current_selected.iter().copied());
        changed.extend(current_candidates.iter().copied());
    }

    for element in &changed {
        let current_is_hovered = previous.current_hovered.contains(element);
        let current_is_selected = current_selected.contains(element);
        // Candidates sit below the two established tiers: an armed pick still
        // wants to show what is hovered and what was already selected.
        let material = if current_is_selected {
            Some(editor_state.selection_material.clone())
        } else if current_is_hovered {
            Some(editor_state.highlight_material.clone())
        } else if current_candidates.contains(element) {
            Some(editor_state.action_candidate_material.clone())
        } else {
            None
        };
        if let Some(rendered) = render_state.rendered_for(*element) {
            update_rendered_highlight(rendered, material, &children_query, &mut material_query);
        }
    }
    previous.changed = changed;

    std::mem::swap(&mut previous.hovered, &mut previous.current_hovered);
    if selection_changed {
        previous.selected.clone_from(&current_selected);
    }
    if candidates_changed {
        previous.candidates.clone_from(current_candidates);
    }
    previous.graph_revision = graph_revision;
}

pub(super) fn collect_changed_interaction_elements(
    previous_hovered: &HashSet<GraphElement>,
    previous_selected: &HashSet<GraphElement>,
    current_hovered: &HashSet<GraphElement>,
    current_selected: &HashSet<GraphElement>,
    changed: &mut HashSet<GraphElement>,
) {
    changed.extend(
        previous_hovered
            .symmetric_difference(current_hovered)
            .copied(),
    );
    changed.extend(
        previous_selected
            .symmetric_difference(current_selected)
            .copied(),
    );
}

pub(crate) fn pipe_mode_selected_elements(
    editor_state: &EditorState,
    graph: &BlockGraph,
) -> HashSet<GraphElement> {
    selected_elements_for_highlight(editor_state, graph).into_owned()
}

/// Borrow the large, usually unchanged selection. Only an unselected pipe
/// anchor needs a temporary expanded set; it never changes the user selection.
fn selected_elements_for_highlight<'a>(
    editor_state: &'a EditorState,
    graph: &BlockGraph,
) -> std::borrow::Cow<'a, HashSet<GraphElement>> {
    let mut selected = std::borrow::Cow::Borrowed(editor_state.selected_element_set());
    if editor_state.is_pipe_tool_active()
        && let Some(start) = editor_state.pipe_start
        && let Some(block) = graph.get_endpoint_block(start)
    {
        let anchor = GraphElement::Block(block.pos());
        if !selected.contains(&anchor) {
            selected.to_mut().insert(anchor);
        }
    }
    selected
}

#[derive(Default)]
pub(crate) struct InteractionHighlightCache {
    hovered: HashSet<GraphElement>,
    selected: HashSet<GraphElement>,
    candidates: HashSet<GraphElement>,
    current_hovered: HashSet<GraphElement>,
    changed: HashSet<GraphElement>,
    graph_revision: u64,
}

/// Retints every mesh under `target`, or restores each to its original material
/// when `material` is `None`.
fn update_highlight(
    target: Entity,
    material: Option<Handle<StandardMaterial>>,
    children_query: &Query<&Children>,
    material_query: &mut Query<(&mut MeshMaterial3d<StandardMaterial>, &OriginalMaterial)>,
) {
    for child in children_query.iter_descendants(target) {
        if let Ok((mut mat, original)) = material_query.get_mut(child) {
            mat.0 = material.clone().unwrap_or_else(|| original.0.clone());
        }
    }
}

/// Retints a rendered element, preferring its tracked mesh parts and falling
/// back to a descendant walk.
fn update_rendered_highlight(
    rendered: &RenderedElement,
    material: Option<Handle<StandardMaterial>>,
    children_query: &Query<&Children>,
    material_query: &mut Query<(&mut MeshMaterial3d<StandardMaterial>, &OriginalMaterial)>,
) {
    if update_highlight_parts(&rendered.mesh_parts, material.clone(), material_query) {
        return;
    }
    update_highlight(rendered.entity, material, children_query, material_query);
}

fn update_highlight_parts(
    mesh_parts: &RenderedMeshParts,
    material: Option<Handle<StandardMaterial>>,
    material_query: &mut Query<(&mut MeshMaterial3d<StandardMaterial>, &OriginalMaterial)>,
) -> bool {
    if mesh_parts.triangles.is_empty() && mesh_parts.lines.is_empty() {
        return false;
    }
    let mut updated = false;
    for child in mesh_parts.triangles.iter().chain(&mesh_parts.lines) {
        if let Ok((mut mat, original)) = material_query.get_mut(*child) {
            mat.0 = material.clone().unwrap_or_else(|| original.0.clone());
            updated = true;
        }
    }
    updated
}

/// True when the layer filter hides `element` in the current View layer.
pub(super) fn element_hidden_by_layer_filter(
    element: GraphElement,
    graph: &BlockGraph,
    editor_state: &EditorState,
) -> bool {
    editor_state.mode == EditorMode::View
        && editor_state.view_current_layer_only
        && match element {
            GraphElement::Block(pos) => graph
                .get_block(pos)
                .is_none_or(|block| !block.occupies_layer(editor_state.plane_height)),
            GraphElement::Pipe(u, v) => !is_pipe_visible(u, v, editor_state.plane_height),
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_selection_keeps_virtual_pipe_anchor_without_changing_selection() {
        let mut graph = BlockGraph::new();
        graph.add_block(bloq_graph::Block::new(
            IVec3::ZERO,
            bloq_graph::BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        let mut editor = EditorState::default();
        assert!(matches!(
            selected_elements_for_highlight(&editor, &graph),
            std::borrow::Cow::Borrowed(_)
        ));
        editor.mode = EditorMode::Edit;
        editor.placement_tool = PlacementTool::Pipe;
        editor.pipe_start = Some(IVec3::ZERO);
        let selection = selected_elements_for_highlight(&editor, &graph);
        assert!(selection.contains(&GraphElement::Block(IVec3::ZERO)));
        assert!(editor.selected_element_set().is_empty());
        editor.replace_selection([GraphElement::Block(IVec3::ZERO)]);
        assert!(matches!(
            selected_elements_for_highlight(&editor, &graph),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn highlight_priority_and_restoration_survive_borrowed_selection_cache() {
        let mut app = App::new();
        app.init_resource::<EditorState>()
            .init_resource::<BloqViewerState>()
            .init_resource::<GraphState>()
            .init_resource::<GraphRenderState>()
            .add_systems(Update, sync_interaction_highlight_system);
        let mut materials = Assets::<StandardMaterial>::default();
        let original = materials.add(StandardMaterial::default());
        let selected = materials.add(StandardMaterial::default());
        let hovered = materials.add(StandardMaterial::default());
        let candidate = materials.add(StandardMaterial::default());
        let mesh = app
            .world_mut()
            .spawn((
                MeshMaterial3d(original.clone()),
                OriginalMaterial(original.clone()),
            ))
            .id();
        app.world_mut()
            .resource_mut::<GraphRenderState>()
            .blocks
            .insert(
                IVec3::ZERO,
                RenderedElement {
                    entity: mesh,
                    signature: 0,
                    geometry_signature: 0,
                    mesh_parts: RenderedMeshParts {
                        triangles: vec![mesh],
                        lines: vec![],
                    },
                },
            );
        let element = GraphElement::Block(IVec3::ZERO);
        {
            let mut editor = app.world_mut().resource_mut::<EditorState>();
            editor.selection_material = selected.clone();
            editor.highlight_material = hovered.clone();
            editor.action_candidate_material = candidate.clone();
            editor.action_candidate_elements.insert(element);
        }
        let assert_material = |app: &App, material: &Handle<StandardMaterial>| {
            assert_eq!(
                &app.world()
                    .get::<MeshMaterial3d<StandardMaterial>>(mesh)
                    .unwrap()
                    .0,
                material
            );
        };
        app.update();
        assert_material(&app, &candidate);
        app.world_mut()
            .resource_mut::<EditorState>()
            .hovered_element = Some(element);
        app.update();
        assert_material(&app, &hovered);
        app.world_mut()
            .resource_mut::<EditorState>()
            .replace_selection([element]);
        app.update();
        assert_material(&app, &selected);
        app.update();
        assert_material(&app, &selected);
        app.world_mut()
            .resource_mut::<EditorState>()
            .clear_selection();
        app.update();
        assert_material(&app, &hovered);
        app.world_mut()
            .resource_mut::<EditorState>()
            .hovered_element = None;
        app.update();
        assert_material(&app, &candidate);
        app.world_mut()
            .resource_mut::<EditorState>()
            .action_candidate_elements
            .clear();
        app.update();
        assert_material(&app, &original);
    }

    #[test]
    #[ignore = "manual large-selection highlight benchmark"]
    fn benchmark_idle_large_selection() {
        let mut app = App::new();
        app.init_resource::<EditorState>()
            .init_resource::<BloqViewerState>()
            .init_resource::<GraphState>()
            .init_resource::<GraphRenderState>()
            .add_systems(Update, sync_interaction_highlight_system);
        app.world_mut()
            .resource_mut::<EditorState>()
            .replace_selection((0..12_000).map(|x| GraphElement::Block(IVec3::new(x, 0, 0))));
        app.update();
        let start = std::time::Instant::now();
        for _ in 0..200 {
            app.update();
        }
        eprintln!(
            "12k selected elements / 200 idle highlight frames: {:?}",
            start.elapsed()
        );
        let selected = app.world().resource::<EditorState>().selected_element_set();
        let cached = selected.clone();
        let start = std::time::Instant::now();
        for _ in 0..200 {
            assert!(std::hint::black_box(selected) == std::hint::black_box(&cached));
        }
        eprintln!(
            "12k selection HashSet equality alone / 200 comparisons: {:?}",
            start.elapsed()
        );
    }
}
