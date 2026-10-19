//! Standalone SVG export of the cached source action dependencies.

use std::fmt::Write;

use super::ActionDag;
use crate::{Action, BlockGraphError};

impl ActionDag {
    /// Renders the stored action DAG as a self-contained SVG.
    ///
    /// Nodes retain source ordinals and complete action text. Solid arrows show
    /// classical variable dependencies, while dashed arrows show dependencies
    /// inferred from correlation support. Edge titles identify their reasons.
    /// Rendering does not derive missing dependencies or assign execution times.
    /// Call [`crate::BlockGraph::analyze_action_graph`] first for an analyzed view.
    /// The SVG uses the same layout as the physical IR dependency view.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid action syntax, unresolved names or a cycle.
    /// No geometry audit or stabilizer analysis is performed during rendering.
    ///
    /// ```
    /// use bloq_graph::GalleryItem;
    /// let source = GalleryItem::T.build();
    /// let dag = source.analyze_action_graph()?;
    /// let svg = dag.to_svg()?;
    /// assert!(svg.contains("mzz"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_svg(&self) -> Result<String, BlockGraphError> {
        self.validate_semantics()?;
        let nodes = self.ordered_nodes().collect::<Vec<_>>();
        let labels = nodes
            .iter()
            .map(|node| {
                let label = match &node.action {
                    Action::Branch { target, condition } => self
                        .branch_region(*target)
                        .map(|region| format!("resolve {} if {condition}", region.name))
                        .unwrap_or_else(|| node.action.to_string()),
                    _ => node.action.to_string(),
                };
                match &node.owner {
                    Some(owner) => format!(
                        "[{} @ {}] {label}",
                        owner.definition,
                        if owner.instance_path.is_empty() {
                            "root"
                        } else {
                            &owner.instance_path
                        }
                    ),
                    None => label,
                }
            })
            .collect::<Vec<_>>();
        let lines = labels
            .iter()
            .map(|label| {
                label
                    .chars()
                    .collect::<Vec<_>>()
                    .chunks(42)
                    .map(|chunk| chunk.iter().collect::<String>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sizes = nodes
            .iter()
            .zip(&lines)
            .map(|(node, lines)| {
                (
                    node.ordinal as u32,
                    (340.0, 38.0 + lines.len() as f32 * 17.0),
                )
            })
            .collect::<Vec<_>>();
        let mut edges = self.dependencies().collect::<Vec<_>>();
        edges.sort_by_key(|&(from, to, reason)| (from, to, reason as u8));
        let pairs = edges
            .iter()
            .map(|&(from, to, _)| (from as u32, to as u32))
            .collect::<Vec<_>>();
        let placed = bloq_utils::graph_layout::compute_relative(&sizes, &pairs, 48.0);
        let width = placed.content.0.max(460.0) + 48.0;
        let height = placed.content.1.max(40.0) + 96.0;
        let status = if self.is_analyzed() {
            "Analyzed"
        } else {
            "Source"
        };
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{status} action dependency graph\"><title>{status} action DAG</title><style>text{{font:12px monospace;fill:#202536}}rect{{stroke-width:1.5}}.measure{{fill:#e6edff;stroke:#4a65b8}}.binding{{fill:#e8f6ed;stroke:#4b8c67}}.resolve{{fill:#fff1df;stroke:#b66b21}}.feedback{{fill:#f3e9fc;stroke:#8859af}}.discard{{fill:#fde8e8;stroke:#bc5151}}</style><defs><marker id=\"arrow\" markerWidth=\"7\" markerHeight=\"7\" refX=\"6\" refY=\"3\" orient=\"auto\"><path d=\"M0 0 L6 3 L0 6 Z\" fill=\"#758198\"/></marker></defs><rect width=\"100%\" height=\"100%\" fill=\"white\"/><g transform=\"translate(24 48)\">"
        );
        for (from, to, reason) in edges {
            let (x, y) = placed.positions[&(from as u32)];
            let (tx, ty) = placed.positions[&(to as u32)];
            let sx = x + sizes[from].1.0;
            let sy = y + sizes[from].1.1 / 2.0;
            let ey = ty + sizes[to].1.1 / 2.0;
            let middle = sx.midpoint(tx);
            let dash = if reason.is_implicit() {
                " stroke-dasharray=\"5 4\""
            } else {
                ""
            };
            write!(svg, "<path data-from=\"{from}\" data-to=\"{to}\" data-dependency=\"{reason:?}\" d=\"M{sx} {sy} C{middle} {sy} {middle} {ey} {tx} {ey}\" fill=\"none\" stroke=\"#758198\" stroke-width=\"1.5\"{dash} marker-end=\"url(#arrow)\"><title>{reason:?}</title></path>").expect("write SVG string");
        }
        for ((node, label), lines) in nodes.iter().zip(&labels).zip(&lines) {
            let id = node.ordinal;
            let (x, y) = placed.positions[&(id as u32)];
            let (w, h) = sizes[id].1;
            let kind = match node.action {
                Action::Measure { .. } => "measure",
                Action::Let { .. } => "binding",
                Action::Resolve { .. } | Action::Branch { .. } => "resolve",
                Action::Feedback { .. } => "feedback",
                Action::DiscardIf(_) => "discard",
            };
            let owner = node
                .owner
                .as_ref()
                .map(|owner| {
                    format!(
                        " data-definition=\"{}\" data-instance=\"{}\"",
                        escape(&owner.definition),
                        escape(&owner.instance_path)
                    )
                })
                .unwrap_or_default();
            write!(svg, "<g data-node=\"{id}\"{owner}><title>{}</title><rect class=\"{kind}\" x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"7\"/><text x=\"{}\" y=\"{}\">A{id}</text>", escape(label), x + 12.0, y + 20.0).expect("write SVG string");
            for (line, text) in lines.iter().enumerate() {
                write!(
                    svg,
                    "<text x=\"{}\" y=\"{}\">{}</text>",
                    x + 12.0,
                    y + 38.0 + line as f32 * 17.0,
                    escape(text)
                )
                .expect("write SVG string");
            }
            svg.push_str("</g>");
        }
        if nodes.is_empty() {
            svg.push_str("<text y=\"20\">No actions</text>");
        }
        svg.push_str("</g><text x=\"24\" y=\"24\">");
        write!(svg, "{status} action DAG · {} actions</text><text x=\"24\" y=\"{}\">Solid: variable dependencies · Dashed: inferred dependencies</text></svg>", nodes.len(), height - 16.0).expect("write SVG string");
        Ok(svg)
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActionDependency;
    use crate::{Expr, GalleryItem, MeasureTarget};

    #[test]
    fn svg_keeps_isolated_actions_dependency_reasons_and_full_labels() {
        let mut dag = ActionDag::from_actions(&[
            Action::Measure {
                target: MeasureTarget::Node(glam::IVec3::ZERO),
                name: "m".into(),
            },
            Action::Let {
                name: "both".into(),
                expr: Expr::Binary(
                    crate::BinaryOp::And,
                    Box::new(Expr::Var("m".into())),
                    Box::new(Expr::Var("m".into())),
                ),
            },
            Action::Feedback {
                targets: vec![crate::FeedbackTarget {
                    pauli: crate::PauliBasis::Z,
                    target: glam::IVec3::ZERO,
                    direction: None,
                }],
                condition: None,
            },
        ]);
        dag.add_dependency(
            dag.ordered[0],
            dag.ordered[1],
            ActionDependency::ReadoutParity,
        );
        let svg = dag.to_svg().unwrap();
        assert_eq!(svg.matches("data-node=").count(), 3);
        assert!(svg.contains("both = m &amp; m"));
        assert!(svg.contains("data-dependency=\"Classical\""));
        assert!(svg.contains("data-dependency=\"ReadoutParity\""));
        assert!(svg.contains("stroke-dasharray"));
        assert_eq!(svg, dag.to_svg().unwrap());
        assert!(
            ActionDag::default()
                .to_svg()
                .unwrap()
                .contains("No actions")
        );
    }

    #[test]
    fn svg_labels_analyzed_graph_and_rejects_cycles() {
        let flat = GalleryItem::T.build().flatten().unwrap();
        let analyzed = flat.analyze_actions().unwrap();
        let svg = analyzed.action_graph().to_svg().unwrap();
        assert!(svg.contains("Analyzed action DAG"));
        let mut cyclic = analyzed.action_graph().clone();
        cyclic.add_dependency(
            cyclic.ordered[1],
            cyclic.ordered[0],
            ActionDependency::Classical,
        );
        assert!(matches!(
            cyclic.to_svg().unwrap_err(),
            BlockGraphError::InvalidAction(
                crate::validate::InvalidActionError::DependencyCycle { .. }
            )
        ));
    }

    #[test]
    fn continuing_branch_analysis_waits_for_the_selected_physical_readout() {
        let source = crate::BlockGraph::from_text(
            r#"BLOG 1.0
module main {
  in driver: data = 0
  0: Port [2, 0, -1] role=input
  1: ZXZ [2, 0, 0]
  0 -> +Z
  2: ZXZ [0, 0, 0]
  3: ZXZ [0, 0, 2]
  branch middle {
    false {
      4: ZXZ [0, 0, 1]
      [0, 0, 0] -> +Z
      [0, 0, 1] -> +Z
    }
    true {
      5: XZX [0, 0, 1]
      [0, 0, 0] -H> +Z
      [0, 0, 1] -H> +Z
    }
  }
  selector = measure 1
  resolve middle if selector
  result = measure 3
}
"#,
        )
        .unwrap();
        let flat = source.flatten().unwrap();
        let before = flat.to_blog_text();
        let dag = flat.analyze_action_graph().unwrap();
        assert!(dag.is_analyzed());
        assert!(
            dag.dependencies()
                .any(|edge| edge == (1, 2, ActionDependency::BranchSupport))
        );
        assert_eq!(flat.to_blog_text(), before);
        let mut constant = flat.clone();
        let mut actions = constant.actions();
        if let Action::Branch { condition, .. } = &mut actions[1] {
            *condition = Expr::Binary(
                crate::BinaryOp::Xor,
                Box::new(Expr::Var("selector".into())),
                Box::new(Expr::Var("selector".into())),
            );
        }
        constant.set_actions(actions).unwrap();
        assert!(
            constant
                .analyze_action_graph()
                .unwrap()
                .dependencies()
                .any(|edge| edge == (1, 2, ActionDependency::BranchSupport))
        );
    }
}
