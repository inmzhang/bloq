//! Checked semantic transport, independent of BLOG syntax and derived caches.

use super::*;

/// An owned, source-level graph snapshot for process/worker transport.
///
/// This is not a compilation certificate. Restoring checks representation,
/// topology, branch interfaces and source dependencies, but deliberately does
/// not repeat module geometry certification or physical/stabilizer analysis.
/// Call the ordinary validation/compilation APIs before executing the graph.
/// Derived caches and internal graph indices are never transported.
#[derive(Debug, Clone)]
pub struct SourceSnapshot {
    /// Executable root (named `main`), including an ordinary flat graph.
    pub root: DefinitionSnapshot,
    /// Root-owned helper library, in declaration order. No nested libraries.
    pub definitions: Vec<DefinitionSnapshot>,
}

/// Source fields of a single local definition.
#[derive(Debug, Clone)]
pub struct DefinitionSnapshot {
    /// Definition name.
    pub name: String,
    /// Whether its interface was explicitly declared (including an empty one).
    pub interface_declared: bool,
    /// Ordered public interface.
    pub interface: crate::ModuleInterface,
    /// Child instances.
    pub instances: Vec<crate::ModuleInstance>,
    /// Quantum seams, including Hadamard metadata.
    pub quantum_connections: Vec<crate::QuantumConnection>,
    /// Child classical input bindings.
    pub bit_bindings: Vec<crate::BitBinding>,
    /// Blocks not owned by a branch arm.
    pub blocks: Vec<Block>,
    /// Common pipes, including seams into branch arms.
    pub pipes: Vec<Pipe>,
    /// Both authored arms and display state of every branch.
    pub branches: Vec<BranchSnapshot>,
    /// Ordered source actions, not analyzed/implicit dependencies.
    pub actions: Vec<Action>,
    /// Declared external names used by the source action DAG.
    pub action_inputs: Vec<String>,
}

/// Branch source data. Cuts are rebuilt and checked, never blindly installed.
#[derive(Debug, Clone)]
pub struct BranchSnapshot {
    /// Public branch name.
    pub name: String,
    /// Stable action anchor. Must remain owned by the true arm.
    pub target: IVec3,
    /// False arm, including owned pipe metadata.
    pub on_false: crate::BranchArm,
    /// True arm, including owned pipe metadata.
    pub on_true: crate::BranchArm,
    /// Which arm is currently materialized for display.
    pub shown_true: bool,
}

/// A source snapshot could not be safely restored.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    /// Malformed transport metadata or an exceeded representation limit.
    #[error("invalid graph snapshot: {0}")]
    Representation(&'static str),
    /// Invalid local geometry, branch interfaces or source actions.
    #[error(transparent)]
    Graph(#[from] BlockGraphError),
    /// Invalid hierarchy references, interfaces or resource budgets.
    #[error(transparent)]
    Module(Box<crate::ModuleError>),
}

impl From<crate::ModuleError> for SnapshotError {
    fn from(error: crate::ModuleError) -> Self {
        Self::Module(Box::new(error))
    }
}

impl DefinitionSnapshot {
    fn capture(graph: &BlockGraph) -> Self {
        let owned = graph
            .branches
            .iter()
            .flat_map(|region| region.shown_arm().blocks().map(Block::pos))
            .collect::<HashSet<_>>();
        Self {
            name: graph.name.clone(),
            interface_declared: graph.interface_declared,
            interface: graph.interface.clone(),
            instances: graph.instances.clone(),
            quantum_connections: graph.quantum_connections.clone(),
            bit_bindings: graph.bit_bindings.clone(),
            blocks: graph
                .blocks()
                .filter(|block| !owned.contains(&block.pos()))
                .cloned()
                .collect(),
            pipes: graph.branch_shared_pipes(),
            branches: graph
                .branches
                .iter()
                .map(|region| BranchSnapshot {
                    name: region.name.clone(),
                    target: region.target,
                    on_false: region.on_false().clone(),
                    on_true: region.on_true().clone(),
                    shown_true: region.shown_true(),
                })
                .collect(),
            actions: graph.actions(),
            action_inputs: graph.action_inputs.iter().cloned().collect(),
        }
    }

    fn restore(self, lenient_actions: bool) -> Result<BlockGraph, SnapshotError> {
        let limits = crate::ModuleCertificationLimits::DEFAULT;
        let mut blocks = 0usize;
        let mut cells = 0usize;
        for block in self.blocks.iter().chain(
            self.branches
                .iter()
                .flat_map(|branch| branch.on_false.blocks().chain(branch.on_true.blocks())),
        ) {
            blocks = blocks
                .checked_add(1)
                .ok_or(SnapshotError::Representation("block count overflow"))?;
            let footprint = match block.kind() {
                BlockKind::Cube(_) => block.height_cells() as usize,
                kind => kind.reserved_offsets().len(),
            };
            cells = cells
                .checked_add(footprint)
                .ok_or(SnapshotError::Representation("footprint count overflow"))?;
            if blocks > limits.max_expanded_blocks || cells > limits.max_occupied_cells {
                return Err(SnapshotError::Representation(
                    "local geometry exceeds resource limits",
                ));
            }
        }
        if !crate::parser::is_valid_identifier(&self.name) {
            return Err(SnapshotError::Representation("invalid definition name"));
        }
        let mut graph = BlockGraph::default();
        for block in self.blocks {
            graph.try_add_block(block)?;
        }
        if self.branches.is_empty() {
            for pipe in self.pipes {
                graph.try_add_pipe(pipe)?;
            }
        } else {
            let regions = self
                .branches
                .iter()
                .map(|branch| {
                    (
                        branch.name.clone(),
                        branch.on_false.clone(),
                        branch.on_true.clone(),
                    )
                })
                .collect();
            // Existing checked installation validates ownership, both arm
            // interfaces, duplicate regions and all common seams.
            graph.try_add_branch_regions(regions, self.pipes)?;
            for (region, saved) in graph.branches.iter_mut().zip(&self.branches) {
                if !region.on_true().contains_block(saved.target) {
                    return Err(SnapshotError::Representation(
                        "branch anchor is not owned by its true arm",
                    ));
                }
                // Transformations preserve this identity even if cut sorting
                // would select another canonical anchor on reinstallation.
                region.target = saved.target;
            }
            for branch in self.branches {
                if !branch.shown_true {
                    graph.set_shown_branch_arm(&branch.name, false)?;
                }
            }
        }
        graph.name = self.name;
        graph.interface_declared = self.interface_declared;
        graph.interface = self.interface;
        graph.instances = self.instances;
        graph.quantum_connections = self.quantum_connections;
        graph.bit_bindings = self.bit_bindings;
        if lenient_actions {
            // Validate input names through the normal checked setter, then retain
            // incomplete action constraints just as incremental editing does.
            graph.set_actions_deferred_with_inputs(Vec::new(), self.action_inputs)?;
            graph.set_actions_lenient(self.actions)?;
        } else {
            graph.set_actions_deferred_with_inputs(self.actions, self.action_inputs)?;
        }
        Ok(graph)
    }
}

impl BlockGraph {
    /// Captures exact semantic source fields without BLOG emission or analysis.
    pub fn source_snapshot(&self) -> SourceSnapshot {
        SourceSnapshot {
            root: DefinitionSnapshot::capture(self),
            definitions: self
                .definitions
                .iter()
                .map(DefinitionSnapshot::capture)
                .collect(),
        }
    }

    /// Restores checked source data without parsing or geometry certification.
    ///
    /// Internal indices and deferred action DAGs are rebuilt. Hierarchy names,
    /// seams, dependency cycles and expansion limits are checked. Branch cuts
    /// are recomputed through the normal checked installer; stable anchors and
    /// shown-arm state are retained. No supplied verification caches are trusted.
    /// This does not confer physical/semantic certification: use [`Self::validate`]
    /// or compilation for those checks, as for any incrementally edited graph.
    ///
    /// # Errors
    /// Returns an error for invalid metadata, topology, branch ownership,
    /// actions, hierarchy references or resource limits. No partial graph escapes.
    pub fn from_source_snapshot(snapshot: SourceSnapshot) -> Result<Self, SnapshotError> {
        Self::restore_source_snapshot(snapshot, false)
    }

    /// Restores editor recovery data, retaining incomplete action constraints
    /// in [`Self::action_graph_error`] rather than dropping the authored actions.
    /// Geometry, identifiers and hierarchy remain checked. This is not a
    /// compilation certificate; ordinary validation is still required.
    ///
    /// # Errors
    /// Returns an error for malformed geometry, action syntax or hierarchy.
    pub fn from_source_snapshot_lenient(snapshot: SourceSnapshot) -> Result<Self, SnapshotError> {
        Self::restore_source_snapshot(snapshot, true)
    }

    fn restore_source_snapshot(
        snapshot: SourceSnapshot,
        lenient_actions: bool,
    ) -> Result<Self, SnapshotError> {
        if snapshot.definitions.len()
            > crate::ModuleCertificationLimits::DEFAULT.max_expanded_instances
        {
            return Err(SnapshotError::Representation("too many definitions"));
        }
        let mut graph = snapshot.root.restore(lenient_actions)?;
        graph.definitions = snapshot
            .definitions
            .into_iter()
            .map(|definition| definition.restore(lenient_actions))
            .collect::<Result<_, _>>()?;
        if graph.has_module_structure() {
            graph.validate_snapshot_hierarchy()?;
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hierarchy() -> BlockGraph {
        let gate = crate::GalleryItem::CNOT
            .build()
            .to_blog_text()
            .replace("module main", "module Gate");
        BlockGraph::from_text(
            &(gate
                + r#"
module main {
  in control_in: data = 0
  out control_out: data = 1
  in target_in: data = 2
  out target_out: data = 3
  gate: Gate @ [0, 0, 0]
  0: Port [0, 0, 0] role=input
  1: Port [0, 0, 3] role=output
  2: Port [1, 1, 0] role=input
  3: Port [1, 1, 3] role=output
  0 -> gate.control_in
  gate.control_out -> 1
  2 -> gate.target_in
  gate.target_out -> 3
}
"#),
        )
        .unwrap()
    }

    #[test]
    fn snapshot_hierarchy_roundtrip_and_flatten_equivalence() {
        let graph = hierarchy();
        let restored = BlockGraph::from_source_snapshot(graph.source_snapshot()).unwrap();
        assert_eq!(restored.to_blog_text(), graph.to_blog_text());
        assert_eq!(
            restored.flatten().unwrap().to_blog_text(),
            graph.flatten().unwrap().to_blog_text()
        );
        restored.validate().unwrap();
    }

    #[test]
    fn snapshot_rejects_missing_definitions_cycles_and_invalid_interfaces() {
        let graph = hierarchy();
        let mut bad = graph.source_snapshot();
        bad.definitions.clear();
        BlockGraph::from_source_snapshot(bad).unwrap_err();
        let mut bad = graph.source_snapshot();
        bad.root.instances[0].definition = "main".into();
        BlockGraph::from_source_snapshot(bad).unwrap_err();
        let mut bad = graph.source_snapshot();
        bad.root.interface.quantum_ports[0].position = IVec3::splat(99);
        BlockGraph::from_source_snapshot(bad).unwrap_err();
        let mut bad = graph.source_snapshot();
        bad.definitions.push(bad.definitions[0].clone());
        BlockGraph::from_source_snapshot(bad).unwrap_err();
    }

    #[test]
    fn snapshot_false_branch_and_external_inputs_roundtrip() {
        let mut graph = BlockGraph::from_text("BLOG 1.0\n9: Z [3,0,0]\n10: ZXZ [0,0,-1]\n[0,0,-1] -> +Z\nbranch b {\n false {\n 1: ZXZ [0,0,0]\n }\n true {\n 2: ZXZ [0,0,0]\n }\n}\nm = measure 9\nresolve b if m\n").unwrap();
        graph.set_shown_branch_arm("b", false).unwrap();
        let restored = BlockGraph::from_source_snapshot(graph.source_snapshot()).unwrap();
        assert_eq!(restored.to_blog_text(), graph.to_blog_text());
        assert_eq!(
            restored.blocks().collect::<Vec<_>>(),
            graph.blocks().collect::<Vec<_>>()
        );
        assert_eq!(restored.actions(), graph.actions());
        assert_eq!(restored.branch_shared_pipes(), graph.branch_shared_pipes());
        assert!(!restored.branches[0].shown_true());
        let mut bad = graph.source_snapshot();
        bad.root.branches[0].target = IVec3::splat(99);
        BlockGraph::from_source_snapshot(bad).unwrap_err();
        let mut external = BlockGraph::default();
        external
            .set_actions_with_inputs(
                vec![Action::DiscardIf(crate::Expr::Var("enabled".into()))],
                ["enabled".into()],
            )
            .unwrap();
        let restored = BlockGraph::from_source_snapshot(external.source_snapshot()).unwrap();
        assert_eq!(restored.action_inputs, external.action_inputs);
        assert_eq!(restored.actions(), external.actions());
        let mut bad = external.source_snapshot();
        bad.root.action_inputs.clear();
        BlockGraph::from_source_snapshot(bad).unwrap_err();
    }

    #[test]
    fn snapshot_budgets_tall_geometry_before_allocating_footprints() {
        let mut snapshot = BlockGraph::default().source_snapshot();
        snapshot.root.blocks = (0..62)
            .map(|i| {
                Block::new([i * 2, 0, 0], BlockKind::Cube(crate::CubeKind::ZXZ))
                    .with_height("65535d".parse().unwrap())
                    .unwrap()
            })
            .collect();
        assert!(matches!(
            BlockGraph::from_source_snapshot(snapshot),
            Err(SnapshotError::Representation(
                "local geometry exceeds resource limits"
            ))
        ));
    }

    #[test]
    fn snapshot_recertification_preserves_boundary_relation() {
        let graph = crate::GalleryItem::CNOT.build();
        let restored = BlockGraph::from_source_snapshot(graph.source_snapshot()).unwrap();
        let rows = |graph: &BlockGraph| {
            graph
                .certify_leaf("main", crate::ModuleCertificationLimits::DEFAULT)
                .unwrap()
                .boundary_rows()
                .map(|row| (row.paulis.to_string(), row.phase()))
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(&restored), rows(&graph));
    }

    #[test]
    fn snapshot_lenient_recovery_preserves_action_constraint_errors() {
        let mut graph = BlockGraph::default();
        graph
            .set_actions_lenient(vec![Action::DiscardIf(crate::Expr::Var("missing".into()))])
            .unwrap();
        assert!(graph.action_graph_error().is_some());
        BlockGraph::from_source_snapshot(graph.source_snapshot()).unwrap_err();
        let restored = BlockGraph::from_source_snapshot_lenient(graph.source_snapshot()).unwrap();
        assert_eq!(restored.actions(), graph.actions());
        assert!(restored.action_graph_error().is_some());
        let mut bad = graph.source_snapshot();
        bad.root.action_inputs.push("not an identifier".into());
        BlockGraph::from_source_snapshot_lenient(bad).unwrap_err();
    }

    #[test]
    fn snapshot_gallery_roundtrip() {
        for item in crate::GalleryItem::iter() {
            let graph = item.build();
            let restored = BlockGraph::from_source_snapshot(graph.source_snapshot()).unwrap();
            assert_eq!(restored.to_blog_text(), graph.to_blog_text(), "{item:?}");
        }
    }

    #[test]
    fn snapshot_rejects_duplicate_geometry() {
        let graph = crate::GalleryItem::CNOT.build();
        let mut bad = graph.source_snapshot();
        bad.root.blocks.push(bad.root.blocks[0].clone());
        BlockGraph::from_source_snapshot(bad).unwrap_err();
    }
}
