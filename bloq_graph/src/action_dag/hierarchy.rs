//! Source DAG inspection through the compiler's module composition boundary.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{ActionDag, ActionOwner};
use crate::{
    BlockGraph, BlockGraphError, GuardedReadoutPlan, GuardedTopology, ModuleCertificationLimits,
    PreparedModuleRelations,
};

/// Plans causal readouts over the hierarchy's composed definition relations,
/// keeping the prepared relations for their linked action scopes.
pub(crate) fn plan_readouts(
    source: &BlockGraph,
    limits: ModuleCertificationLimits,
) -> Result<(Arc<PreparedModuleRelations>, GuardedReadoutPlan), BlockGraphError> {
    // Validate declarations and charge expanded sizes before linking addresses.
    source
        .validate_with_limits(limits)
        .map_err(|error| BlockGraphError::ModuleSource(Arc::new(error)))?;
    // Preparation certifies each child's interface before a parent can consume
    // its outputs, and retains the same relations used below.
    let summary = source
        .prepare_validated_module_relations(limits, crate::default_module_jobs(), |_| None)
        .map_err(|error| BlockGraphError::ModuleMaterialization(Arc::new(error)))?;
    let linked = summary.linked();
    let topology = GuardedTopology::new(&linked.graph, limits)?;
    let plan = summary.compose(topology, &linked.sites)?.plan_readouts()?;
    Ok((summary, plan))
}

pub(crate) fn analyze(
    source: &BlockGraph,
    limits: ModuleCertificationLimits,
) -> Result<ActionDag, BlockGraphError> {
    let (summary, mut plan) = plan_readouts(source, limits)?;
    let linked = summary.linked();
    let mut dag = plan
        .topology
        .source
        .build_action_graph(&plan.topology.source.actions())?;
    dag.attach_guarded_dependencies(plan.action_dependencies()?)?;

    let mut owners = BTreeMap::new();
    let mut pending = vec![(source.root(), String::new())];
    while let Some((module, path)) = pending.pop() {
        pending.extend(module.instances.iter().map(|instance| {
            (
                source
                    .module(&instance.definition)
                    .expect("validated definition"),
                crate::program::qualified_name(&path, &instance.name),
            )
        }));
        owners.insert(
            path.clone(),
            ActionOwner {
                definition: module.name.clone(),
                instance_path: path,
            },
        );
    }
    debug_assert_eq!(dag.ordered.len(), linked.action_scopes.len());
    for (&index, scope) in dag.ordered.iter().zip(&linked.action_scopes) {
        dag.graph[index].owner = Some(owners[scope].clone());
    }
    Ok(dag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, ActionDependency, MeasureTarget};

    #[test]
    fn nested_repeated_rotated_instances_keep_classical_edges_and_owners() {
        let stage = include_str!("../../tests/fixtures/module_classical.blog")
            .replace("module main {", "module Stage {");
        let source = BlockGraph::from_text(&format!(
            "{stage}\n{}",
            r#"
module main {
  in a: data = 0
  in b: data = 1
  in c: data = 2
  in d: data = 3
  lower: Stage @ [0, 0, 0]
  upper: Stage @ [4, 0, 0] rotate Z 180
  0: Port [0, 0, 0] role=input
  1: Port [1, 0, 2] role=input
  2: Port [4, 0, 0] role=input
  3: Port [3, 0, 2] role=input
  0 -> lower.first
  1 -> lower.second
  2 -> upper.first
  3 -> upper.second
}
"#
        ))
        .unwrap();
        let before = source.to_blog_text();
        let dag = source.analyze_action_graph().unwrap();
        assert!(dag.is_analyzed());
        assert_eq!(dag.ordered_nodes().count(), 6);
        assert_eq!(source.to_blog_text(), before);
        assert_eq!(source.root().instances.len(), 2);
        for (start, scope, x) in [(0, "lower", 0), (3, "upper", 4)] {
            let measure = dag.node_by_ordinal(start).unwrap();
            assert_eq!(
                measure.owner.as_ref().unwrap(),
                &ActionOwner {
                    definition: "MeasureZ".into(),
                    instance_path: format!("{scope}__read"),
                }
            );
            assert!(
                matches!(&measure.action, Action::Measure { target: MeasureTarget::Node(pos), name }
                if *pos == glam::IVec3::new(x, 0, 1) && name == &format!("{scope}__read__mz"))
            );
            for ordinal in [start + 1, start + 2] {
                assert_eq!(
                    dag.node_by_ordinal(ordinal)
                        .unwrap()
                        .owner
                        .as_ref()
                        .unwrap(),
                    &ActionOwner {
                        definition: "SelectCap".into(),
                        instance_path: format!("{scope}__cap"),
                    }
                );
            }
            assert!(
                dag.dependencies()
                    .any(|edge| edge == (start, start + 1, ActionDependency::Classical))
            );
            assert!(
                dag.dependencies()
                    .any(|edge| edge == (start + 1, start + 2, ActionDependency::Classical))
            );
        }
        assert!(
            matches!(dag.node_by_ordinal(5).unwrap().action, Action::Resolve { target, .. }
            if target == glam::IVec3::new(3, 0, 4))
        );
        let error = source
            .analyze_action_graph_with_limits(ModuleCertificationLimits {
                max_expanded_instances: 0,
                ..ModuleCertificationLimits::DEFAULT
            })
            .unwrap_err();
        assert!(matches!(error, BlockGraphError::ModuleSource(_)));
    }

    #[test]
    fn parent_readout_depends_on_child_feedback_across_a_quantum_seam() {
        let source = BlockGraph::from_text(
            r#"BLOG 1.0
module Wire {
  in input: data = 0
  out output: data = 2
  0: Port [0, 0, -1] role=input
  1: ZXZ [0, 0, 0]
  2: Port [0, 0, 1] role=output
  0 -> +Z
  1 -> +Z
  feedback Z 1 -> +Z
}
module main {
  in input: data = 10
  out result = m
  wire: Wire @ [0, 0, 0]
  10: Port [0, 0, -1] role=input
  11: X [0, 0, 1]
  10 -> wire.input
  wire.output -> 11
  m = measure 11
}
"#,
        )
        .unwrap();
        let dag = source.analyze_action_graph().unwrap();
        assert_eq!(dag.ordered_nodes().count(), 2);
        assert!(
            dag.dependencies()
                .any(|edge| edge == (0, 1, ActionDependency::FeedbackAnticommutation))
        );
        assert_eq!(
            dag.node_by_ordinal(0)
                .unwrap()
                .owner
                .as_ref()
                .unwrap()
                .instance_path,
            "wire"
        );
        assert_eq!(
            dag.node_by_ordinal(1)
                .unwrap()
                .owner
                .as_ref()
                .unwrap()
                .instance_path,
            ""
        );
        let svg = dag.to_svg().unwrap();
        assert!(svg.contains("data-definition=\"Wire\" data-instance=\"wire\""));
        assert!(svg.contains("data-dependency=\"FeedbackAnticommutation\""));
    }
}
