//! Prepare saved graph/history documents in a worker; install checked source
//! snapshots without parsing BLOG or recomputing text synchronization on the UI.

use super::*;
use crate::systems::jobs::live_blog_transport::RecoveryGraph;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct PreparedSession {
    session: Session,
    documents: Vec<PreparedDocument>,
    order: Vec<usize>,
    sync: Vec<(Option<String>, bool)>,
}

#[derive(Serialize, Deserialize)]
struct PreparedDocument {
    graph: RecoveryGraph,
    source: Option<RecoveryGraph>,
    pending: Option<PreparedArm>,
}

#[derive(Serialize, Deserialize)]
struct PreparedArm {
    name: String,
    captured_true: bool,
    graph: RecoveryGraph,
    owned: HashSet<IVec3>,
}

impl PreparedDocument {
    fn capture(saved: &SavedDocument, state: &GraphState) -> Result<Self> {
        Ok(Self {
            graph: RecoveryGraph::capture(&state.graph),
            source: state.source_graph.as_deref().map(RecoveryGraph::capture),
            pending: saved
                .pending_arm
                .as_ref()
                .map(|arm| -> Result<_> {
                    Ok(PreparedArm {
                        name: arm.name.clone(),
                        captured_true: arm.captured_true,
                        graph: RecoveryGraph::capture(&BlockGraph::from_blog_text(&arm.graph)?),
                        owned: arm.owned.clone(),
                    })
                })
                .transpose()?,
        })
    }

    fn restore(self) -> Result<GraphState> {
        let graph = self.graph.restore().map_err(color_eyre::eyre::Error::msg)?;
        let source_graph = self
            .source
            .map(|source| source.restore().map(Arc::new))
            .transpose()
            .map_err(color_eyre::eyre::Error::msg)?;
        let pending_branch_arm = self
            .pending
            .map(|arm| -> Result<_> {
                let graph = arm.graph.restore().map_err(color_eyre::eyre::Error::msg)?;
                restore_pending_arm(arm.name, arm.captured_true, graph, &arm.owned)
            })
            .transpose()?;
        let mut state = GraphState {
            graph,
            source_graph,
            pending_branch_arm,
            ..default()
        };
        state.history.clear();
        state.commit();
        Ok(state)
    }
}

/// Worker-side preparation uses exactly the native recovery parser semantics.
pub(super) fn prepare(json: &str) -> Result<String> {
    let session = Session::from_json(json)?;
    let mut documents = Vec::new();
    let mut states = Vec::<GraphState>::new();
    let mut indices = HashMap::<String, usize>::new();
    let mut order = Vec::new();
    let tabs = session.clone().restore_with(
        |saved| {
            // Identical current/history entries are common. Deduplicate before
            // parsing and shipping, rather than cloning full graph wire payloads.
            let key = serde_json::to_string(saved)?;
            let index = if let Some(index) = indices.get(&key) {
                *index
            } else {
                let state = saved.restore()?;
                let index = documents.len();
                documents.push(PreparedDocument::capture(saved, &state)?);
                states.push(state);
                indices.insert(key, index);
                index
            };
            order.push(index);
            Ok(states[index].clone())
        },
        None,
    )?;
    let sync = tabs
        .tabs
        .iter()
        .map(|tab| tab.snapshot.import_export.recovery_sync())
        .collect();
    Ok(serde_json::to_string(&PreparedSession {
        session,
        documents,
        order,
        sync,
    })?)
}

/// Called after the browser owner resolves the pending-recovery edit guard.
/// This never reparses saved BLOG; incomplete drafts remain untouched strings.
#[cfg(test)]
pub(crate) fn restore_prepared_json(json: &str) -> Result<EditorTabs, String> {
    restore_prepared_with_cache(json).map(|(tabs, _)| tabs)
}

pub(super) fn restore_prepared_with_cache(json: &str) -> Result<(EditorTabs, Session), String> {
    (|| -> Result<_> {
        let prepared: PreparedSession = serde_json::from_str(json)?;
        let mut remaining = vec![0usize; prepared.documents.len()];
        for &index in &prepared.order {
            *remaining
                .get_mut(index)
                .wrap_err("Invalid prepared recovery document index")? += 1;
        }
        let mut states = prepared
            .documents
            .into_iter()
            .map(|document| document.restore().map(Some))
            .collect::<Result<Vec<_>>>()?;
        let mut order = prepared.order.into_iter();
        let mut session = prepared.session;
        let tabs = session.clone().restore_with(
            |_| {
                let index = order
                    .next()
                    .wrap_err("Missing prepared recovery document")?;
                remaining[index] -= 1;
                // Shared current/history documents still need independent state.
                // Move the final occurrence instead of cloning its graph and
                // undo baseline only to immediately drop the cached original.
                if remaining[index] == 0 {
                    states[index].take()
                } else {
                    states[index].clone()
                }
                .wrap_err("Missing prepared recovery document")
            },
            Some(prepared.sync),
        )?;
        ensure!(order.next().is_none(), "Unused prepared recovery document");
        for (saved, tab) in session.tabs.iter_mut().zip(&tabs.tabs) {
            let state = &tab.snapshot;
            saved.graph_revision = state.graph_state.revision;
            saved.blog_revision = state.import_export.bloq_revision;
            if saved.text_history.len() != state.import_export.text_history().len() {
                saved.text_history = Arc::from(state.import_export.text_history());
            }
            saved.text_history_revision = state.import_export.text_history_revision();
            saved.text_conflict = state.import_export.sync_conflict;
            if let Some(history) = &mut saved.history {
                Arc::make_mut(history).cache_ids = state
                    .graph_state
                    .history
                    .iter()
                    .map(|entry| entry.cache_id)
                    .collect();
            }
        }
        Ok((tabs, session))
    })()
    .map_err(|error| format!("{error:#}"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
#[expect(
    unreachable_pub,
    reason = "wasm-bindgen exports require pub even from a private module"
)]
pub fn prepare_recovery_in_worker(json: &str) -> Result<String, wasm_bindgen::JsValue> {
    prepare(json).map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:#}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(session: &Session) -> EditorTabs {
        let prepared = prepare(&serde_json::to_string(session).unwrap()).unwrap();
        restore_prepared_json(&prepared).unwrap()
    }

    #[test]
    fn prepared_recovery_preserves_incomplete_actions_drafts_and_history() {
        let mut tabs = EditorTabs::default();
        let state = &mut tabs.tabs[0].snapshot;
        state.graph_state.graph = bloq_graph::lower_blog_ast_lenient(
            &bloq_graph::parse_blog_to_ast("BLOG 1.0\n0: ZXZ [0,0,0]\ndiscard if missing\n")
                .unwrap(),
            [],
        )
        .unwrap();
        assert!(state.graph_state.graph.action_graph_error().is_some());
        state.graph_state.commit();
        state.import_export.set_bloq_buffer("unfinished {".into());
        state
            .import_export
            .restore_text_history(vec!["previous draft".into()]);
        let session = Session::capture(&tabs, (&tabs.tabs[0].snapshot).into(), None).unwrap();
        let restored = roundtrip(&session);
        let actual = &restored.tabs[0].snapshot;
        assert!(actual.graph_state.graph.action_graph_error().is_some());
        assert_eq!(actual.graph_state.graph.actions(), state_actions(&session));
        assert_eq!(actual.import_export.bloq_buffer, "unfinished {");
        assert_eq!(actual.import_export.text_history(), ["previous draft"]);
        assert_eq!(
            actual.graph_state.history.len(),
            tabs.tabs[0].snapshot.graph_state.history.len()
        );
    }

    fn state_actions(session: &Session) -> Vec<bloq_graph::Action> {
        session.tabs[0].document.restore().unwrap().graph.actions()
    }

    #[test]
    fn deduplicated_recovery_documents_remain_independent() {
        let mut tabs = EditorTabs::default();
        tabs.add_empty_tab(&EditorState::default());
        let session = Session::capture(&tabs, (&tabs.active_tab().snapshot).into(), None).unwrap();
        let json = prepare(&serde_json::to_string(&session).unwrap()).unwrap();
        let wire: PreparedSession = serde_json::from_str(&json).unwrap();
        assert_eq!(wire.documents.len(), 1);
        assert!(wire.order.len() > 1);
        let mut restored = restore_prepared_json(&json).unwrap();
        let state = &mut restored.tabs[0].snapshot.graph_state;
        state.graph.add_block(Block::new(
            [0, 0, 0],
            BlockKind::Cube(bloq_graph::CubeKind::ZXZ),
        ));
        assert!(state.history[state.current_index].graph.is_empty());
        assert!(restored.tabs[1].snapshot.graph_state.graph.is_empty());
    }

    #[test]
    fn prepared_recovery_matches_native_and_deduplicates_history() {
        let tabs = EditorTabs::default();
        let session = Session::capture(&tabs, (&tabs.tabs[0].snapshot).into(), None).unwrap();
        let json = prepare(&serde_json::to_string(&session).unwrap()).unwrap();
        let wire: PreparedSession = serde_json::from_str(&json).unwrap();
        assert!(wire.documents.len() <= wire.order.len());
        let (restored, cache) = restore_prepared_with_cache(&json).unwrap();
        let captured = Session::capture(
            &restored,
            (&restored.active_tab().snapshot).into(),
            Some(&cache),
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &captured.tabs[0].document,
            &cache.tabs[0].document
        ));
        let native = session.restore().unwrap();
        assert_eq!(restored.active, native.active);
        assert_eq!(
            restored.tabs[0].snapshot.graph_state.graph.to_blog_text(),
            native.tabs[0].snapshot.graph_state.graph.to_blog_text()
        );
        assert_eq!(
            restored.tabs[0].snapshot.import_export.recovery_sync(),
            native.tabs[0].snapshot.import_export.recovery_sync()
        );
    }
}
