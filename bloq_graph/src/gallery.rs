//! Built-in block-graph and module-program examples.

use strum::{Display, EnumCount, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};

use crate::{BlockGraph, ModuleCertificationLimits, ModuleError, parse_inline_graph_with_limits};

/// A tag used to filter [`GalleryItem`] entries.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Display, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum GalleryCategory {
    /// Entries using only Clifford operations.
    #[strum(to_string = "clifford")]
    Clifford,
    /// Entries using non-Clifford (T-based) operations.
    #[strum(to_string = "non_clifford")]
    NonClifford,
    /// Magic-state factory constructions.
    #[strum(to_string = "factory")]
    Factory,
    /// Computations that require a prepared non-Clifford state on open ports.
    #[strum(to_string = "external_resource")]
    ExternalResource,
    /// Computations with runtime-selected measurement branches.
    #[strum(to_string = "adaptive")]
    Adaptive,
    /// Arithmetic constructions and their reusable recipes.
    #[strum(to_string = "arithmetic")]
    Arithmetic,
    /// Addition layouts and their AND, MAJ, and UMA recipes.
    #[strum(to_string = "addition")]
    Addition,
    /// Graph examples outside the compiler's accepted input model.
    #[strum(to_string = "analysis_only")]
    AnalysisOnly,
}

/// A gallery example: its description, categories, and embedded `.blog` source.
#[derive(Debug, Copy, Clone)]
pub struct GalleryEntry {
    pub(crate) description: &'static str,
    pub(crate) categories: &'static [GalleryCategory],
    pub(crate) blog: &'static str,
}

impl GalleryEntry {
    /// Returns the human-readable description.
    pub fn description(&self) -> &'static str {
        self.description
    }

    /// Returns the categories this entry belongs to.
    pub fn categories(&self) -> &'static [GalleryCategory] {
        self.categories
    }

    /// Returns the embedded `.blog` source text.
    pub fn blog(&self) -> &'static str {
        self.blog
    }
}

macro_rules! gallery {
    ($( $(#[$metadata:meta])* $variant:ident => ($description:literal, $categories:expr, $source:literal $(,)?), )*) => {
        /// A built-in example block graph, identified by a stable string id.
        ///
        /// Each entry retains its description, categories, and embedded BLOG source.
        #[derive(
            Debug, Copy, Clone, PartialEq, Eq, Hash, Display, EnumCount, EnumIter, EnumString,
            IntoStaticStr,
        )]
        #[repr(usize)]
        #[strum(ascii_case_insensitive)]
        pub enum GalleryItem {
            $(
                #[doc = $description]
                $(#[$metadata])*
                $variant,
            )*
        }

        impl GalleryItem {
            /// Returns the metadata entry for this example.
            pub fn entry(self) -> &'static GalleryEntry {
                match self {
                    $(Self::$variant => &GalleryEntry {
                        description: $description,
                        categories: $categories,
                        blog: include_str!($source),
                    },)*
                }
            }
        }
    };
}

gallery! {
    #[strum(to_string = "cnot")]
    CNOT => ("CNOT gate", CLIFFORD, "../assets/cnot.blog"),
    #[strum(to_string = "cz_spatial_h")]
    CZSpatialH => ("CZ gate with a spatial Hadamard.", CLIFFORD, "../assets/cz_spatial_h.blog"),
    #[strum(to_string = "cz_temporal_h")]
    CZTemporalH => ("CZ gate with a temporal Hadamard.", CLIFFORD, "../assets/cz_temporal_h.blog"),
    #[strum(to_string = "s_gate")]
    S => ("S gate teleportation.", CLIFFORD, "../assets/s.blog"),
    #[strum(to_string = "t_gate")]
    T => ("T gate teleportation.", NON_CLIFFORD, "../assets/t.blog"),
    #[strum(to_string = "t_with_prepared_y")]
    TWithPreparedY => (
        "T gate teleportation with a prepared Y for Clifford fix.",
        NON_CLIFFORD,
        "../assets/t_with_prepared_y.blog",
    ),
    #[strum(to_string = "t_comparison")]
    TComparison => (
        "Naive T-state comparison with a feed-forward X/Y parity check.",
        NON_CLIFFORD,
        "../assets/t_comparison.blog",
    ),
    #[strum(to_string = "phase_gradient", serialize = "phase_gradient_k4")]
    PhaseGradientK4 => (
        "Fixed single-qubit sequence of 22 signed pi/4 X/Z rotations and a final H, composed from T-injection modules.",
        NON_CLIFFORD,
        "../assets/phase_gradient_k4.blog",
    ),
    #[strum(to_string = "and_4t")]
    And4T => (
        "AND gate (x, y -> x, y, x AND y) using four T injections, multiplex controls, and a spatial output.",
        NON_CLIFFORD,
        "../assets/and_4t.blog",
    ),
    #[strum(to_string = "ccz_injected_and")]
    CCZInjectedAnd => (
        "Compact CCZ-injected temporary AND with multiplex controls and one temporal Hadamard; based on Low et al., Fig. 43(b), arXiv:2605.30455v1. Three open ports accept |CCZ>.",
        ADDITION_RESOURCE,
        "../assets/ccz_injected_and.blog",
    ),
    #[strum(to_string = "ccz_injected_maj")]
    CCZInjectedMaj => (
        "SAT-synthesized CCZ-injected MAJ carry slice from Low et al., Figs. 42(b), 43(c), arXiv:2605.30455v1; three open ports accept |CCZ>.",
        ADDITION_RESOURCE,
        "../assets/ccz_injected_maj.blog",
    ),
    #[strum(to_string = "uma")]
    UMA => (
        "SAT-synthesized UMA carry-uncompute and sum-extraction slice from Low et al., Figs. 42(c), 43(c), and 62, arXiv:2605.30455v1.",
        &[
            GalleryCategory::Clifford,
            GalleryCategory::Adaptive,
            GalleryCategory::Arithmetic,
            GalleryCategory::Addition,
        ],
        "../assets/uma.blog",
    ),
    #[strum(to_string = "three_bit_adder", serialize = "controlled_adder_3")]
    ThreeBitAdder => (
        "Three-bit controlled-adder layout with compact routing, five external CCZ states, and adaptive final CZ corrections.",
        ADDITION_RESOURCE,
        "../assets/three_bit_adder.blog",
    ),
    #[strum(to_string = "ten_bit_adder", serialize = "controlled_adder_10")]
    TenBitAdder => (
        "Ten-bit controlled-adder layout with compact routing, nineteen external CCZ states, and adaptive final CZ corrections.",
        ADDITION_RESOURCE,
        "../assets/ten_bit_adder.blog",
    ),
    #[strum(to_string = "toffoli_from_and_delayed_cz", serialize = "toffoli")]
    ToffoliFromAndDelayedCZ => (
        "Toffoli gate (x, y, z -> x, y, z XOR xy) from the four-T AND gadget with a delayed CZ.",
        NON_CLIFFORD,
        "../assets/toffoli_from_and_delayed_cz.blog",
    ),
    #[strum(to_string = "ccz_gate_teleport")]
    CCZGateTeleport => (
        "CCZ gate teleportation with three explicit correction branches.",
        EXTERNAL_RESOURCE_NON_CLIFFORD,
        "../assets/ccz_gate_teleport.blog",
    ),
    #[strum(to_string = "ccz_4x3x7_tels")]
    CCZFactoryWithTels => (
        "CCZ state factory constructed in arXiv:2409.17595.",
        FACTORY_NON_CLIFFORD,
        "../assets/ccz_4x3x7_tels.blog",
    ),
    #[strum(to_string = "ccz_4x3x6")]
    CCZ4x3_6 => (
        "CCZ state factory, 4x3 layout, without the TELS check.",
        FACTORY_NON_CLIFFORD,
        "../assets/ccz_4x3x6.blog",
    ),
    #[strum(to_string = "bell_state")]
    BellState => ("Bell state preparation.", CLIFFORD, "../assets/bell_state.blog"),
    #[strum(to_string = "ghz")]
    GHZ => ("GHZ state preparation.", CLIFFORD, "../assets/ghz.blog"),
    #[strum(to_string = "ghz_slide_then_glide")]
    GHZSlideThenGlide => (
        "GHZ state preparation with sliding then gliding blocks.",
        CLIFFORD,
        "../assets/ghz_slide_then_glide.blog",
    ),
    #[strum(to_string = "ghz_patch_rotations")]
    GHZPatchRotations => (
        "GHZ state preparation with per-qubit patch rotations.",
        CLIFFORD,
        "../assets/ghz_patch_rotations.blog",
    ),
    #[strum(to_string = "1d-yoked", serialize = "1d_yoked")]
    OneDYoked => ("1D yoked lattice-surgery layout.", CLIFFORD, "../assets/1d_yoked.blog"),
    #[strum(to_string = "thth")]
    THTH => ("'-T-H-T-H-' gate sequence", NON_CLIFFORD, "../assets/thth.blog"),
    #[strum(to_string = "three_cnots")]
    ThreeCNOTs => ("Compressed three CNOTs.", CLIFFORD, "../assets/three_cnots.blog"),
    #[strum(to_string = "steane_encoding")]
    SteaneEncoding => (
        "Compressed Steane encoding circuit.",
        CLIFFORD,
        "../assets/steane_encoding.blog",
    ),
    #[strum(to_string = "x_memory")]
    XMemory => ("X-basis memory experiment.", CLIFFORD, "../assets/x_memory.blog"),
    #[strum(to_string = "y_memory")]
    YMemory => ("Y-basis memory experiment.", CLIFFORD, "../assets/y_memory.blog"),
    #[strum(to_string = "move_rotation")]
    MoveRotation => (
        "Rotate boundary types by moving in spacetime.",
        CLIFFORD,
        "../assets/move_rotation.blog",
    ),
    #[strum(to_string = "stability")]
    Stability => ("Stability experiment", CLIFFORD, "../assets/stability.blog"),
}

impl GalleryItem {
    /// Loads the embedded BLOG graph, retaining its module hierarchy.
    ///
    /// Every gallery entry is self-contained.
    ///
    /// # Panics
    ///
    /// Panics if an embedded source fails to load, which would indicate a
    /// corrupt build asset rather than a caller error.
    pub fn build(self) -> BlockGraph {
        self.build_with_limits(ModuleCertificationLimits::DEFAULT)
            .expect("embedded gallery .blog files are valid")
    }

    /// Loads the embedded graph with explicit source-analysis and expansion limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the source cannot be loaded within `limits`.
    pub fn build_with_limits(
        self,
        limits: ModuleCertificationLimits,
    ) -> Result<BlockGraph, ModuleError> {
        parse_inline_graph_with_limits(self.entry().blog, limits)
    }

    /// Iterates over all examples.
    pub fn iter() -> impl Iterator<Item = GalleryItem> {
        <Self as IntoEnumIterator>::iter()
    }

    /// Iterates over the examples tagged with `category`.
    pub fn iter_by_category(category: GalleryCategory) -> impl Iterator<Item = GalleryItem> {
        Self::iter().filter(move |entry| entry.in_category(category))
    }

    /// Returns the stable string id (the variant's canonical name).
    pub fn id(self) -> &'static str {
        // The id is the enum's canonical string form (its `strum` `to_string`),
        // provided by `IntoStaticStr`, so it never duplicates the variant table.
        self.into()
    }

    /// Returns the human-readable description.
    pub fn description(self) -> &'static str {
        self.entry().description
    }

    /// Returns the categories this example belongs to.
    pub fn categories(self) -> &'static [GalleryCategory] {
        self.entry().categories
    }

    /// Returns whether this example is tagged with `category`.
    pub fn in_category(self, category: GalleryCategory) -> bool {
        self.categories().contains(&category)
    }
}

const CLIFFORD: &[GalleryCategory] = &[GalleryCategory::Clifford];
const NON_CLIFFORD: &[GalleryCategory] = &[GalleryCategory::NonClifford];
const FACTORY_NON_CLIFFORD: &[GalleryCategory] =
    &[GalleryCategory::Factory, GalleryCategory::NonClifford];
const EXTERNAL_RESOURCE_NON_CLIFFORD: &[GalleryCategory] = &[
    GalleryCategory::ExternalResource,
    GalleryCategory::NonClifford,
];
const ADDITION_RESOURCE: &[GalleryCategory] = &[
    GalleryCategory::ExternalResource,
    GalleryCategory::NonClifford,
    GalleryCategory::Adaptive,
    GalleryCategory::Arithmetic,
    GalleryCategory::Addition,
];

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bloq_utils::Pauli;

    use super::*;
    use crate::{Action, Block, Expr, Pipe};
    use bloq_utils::Direction;
    use glam::IVec3;

    const GATE_GADGETS: [GalleryItem; 5] = [
        GalleryItem::CNOT,
        GalleryItem::S,
        GalleryItem::T,
        GalleryItem::THTH,
        GalleryItem::ToffoliFromAndDelayedCZ,
    ];

    #[test]
    fn test_galleries_validity() {
        for entry in GalleryItem::iter() {
            let source = entry.build();
            let graph = source
                .materialize_root_graph()
                .expect("gallery flat projection");
            let validation = if entry.in_category(GalleryCategory::AnalysisOnly)
                || !source.instances.is_empty()
            {
                graph.validate_source()
            } else {
                graph.validate()
            };
            validation.unwrap_or_else(|err| {
                panic!("gallery entry {:?} failed validation: {}", entry.id(), err)
            });
        }
    }

    #[test]
    fn composed_sources_keep_reusable_module_hierarchies() {
        let adder = crate::BlockGraph::from_text(bloq_test::ONE_BIT_ADDER_SOURCE).unwrap();
        assert_eq!(adder.modules().len(), 4);
        assert_eq!(adder.root().instances.len(), 3);
        assert_eq!(adder.root().quantum_connections.len(), 18);
        assert_eq!(
            adder
                .direct_pipe_endpoints("main")
                .expect("adder root exists")
                .len(),
            4
        );

        let phase = GalleryItem::PhaseGradientK4.build();
        assert_eq!(phase.modules().len(), 5);
        assert_eq!(phase.root().instances.len(), 22);
        assert!(phase.module("FinalHadamard").is_none());
        assert!(phase.root().local_body().has_block_at(IVec3::new(0, 0, 23)));
        assert!(phase.root().quantum_connections.iter().any(|connection| {
            matches!(
                connection,
                crate::QuantumConnection::Output {
                    block,
                    hadamard: true,
                    ..
                } if *block == IVec3::new(0, 0, 23)
            )
        }));
        assert_eq!(
            phase
                .direct_pipe_endpoints("main")
                .expect("phase-gradient root exists")
                .len(),
            21
        );
    }

    #[test]
    fn addition_gallery_keeps_the_compact_recipe_and_register_interfaces() {
        for item in [
            GalleryItem::CCZInjectedAnd,
            GalleryItem::CCZInjectedMaj,
            GalleryItem::UMA,
            GalleryItem::ThreeBitAdder,
            GalleryItem::TenBitAdder,
        ] {
            assert!(item.in_category(GalleryCategory::Arithmetic));
            assert!(item.in_category(GalleryCategory::Addition));
        }
        let and = GalleryItem::CCZInjectedAnd.build();
        assert_eq!(and.root().local_body().blocks().count(), 38);
        for (item, bits, blocks) in [
            (GalleryItem::ThreeBitAdder, 3, 532),
            (GalleryItem::TenBitAdder, 10, 1932),
        ] {
            let program = item.build();
            assert_eq!(
                program
                    .module("InjectedAnd")
                    .unwrap()
                    .local_body()
                    .to_blog_body_text(),
                and.root().local_body().to_blog_body_text()
            );
            assert_eq!(
                program
                    .extract_definition("InjectedAnd")
                    .unwrap()
                    .to_blog_text(),
                and.to_blog_text()
            );
            assert_eq!(program.root().instances.len(), 3 * bits - 1);
            assert_eq!(
                program
                    .root()
                    .interface
                    .quantum_ports
                    .iter()
                    .filter(|port| port.resource_type == "ccz")
                    .count(),
                3 * (2 * bits - 1)
            );
            assert_eq!(
                program
                    .root()
                    .interface
                    .quantum_ports
                    .iter()
                    .filter(|port| port.direction == crate::PortDirection::Output)
                    .count(),
                2 * bits + 1
            );
            let graph = program.materialize_flat_graph().unwrap();
            assert_eq!(graph.blocks().count(), blocks);
            assert_eq!(*graph.spans().unwrap().2.end(), 14);
        }
    }

    /// Every port of a composable gallery item admits every Pauli.
    ///
    /// A port's Paulis across the generators must span both bits: only then can
    /// a surface arriving from an adjacent gadget always continue through this
    /// one, so concatenating gadgets never strands a record.
    #[test]
    fn every_composable_gallery_port_admits_every_pauli() {
        for entry in GalleryItem::iter() {
            // These entries are resource-consuming tensor effects, not
            // transparent gate interfaces. Their prepared-resource maps are
            // checked exhaustively in `verify_gallery`.
            if entry.in_category(GalleryCategory::AnalysisOnly)
                || !entry.build().root().instances.is_empty()
                || matches!(
                    entry,
                    GalleryItem::CCZInjectedAnd | GalleryItem::CCZInjectedMaj | GalleryItem::UMA
                )
            {
                continue;
            }
            let graph = entry
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            for projection in graph.branch_projections().unwrap() {
                let graph = projection.graph();
                let stabilizers = graph.stabilizers().unwrap();

                for block in graph.blocks().filter(|block| block.kind().is_port()) {
                    let mut span = Vec::new();
                    for generator in &stabilizers {
                        let mut bits = *generator
                            .stabilizer
                            .port_stabilizer
                            .get(&block.pos())
                            .unwrap_or(&Pauli::I) as u8;
                        for basis in &span {
                            bits = bits.min(bits ^ basis);
                        }
                        if bits != 0 {
                            span.push(bits);
                            span.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
                        }
                    }
                    assert_eq!(
                        span.len(),
                        2,
                        "{} {:?} port {} spans only {:?}",
                        entry.id(),
                        projection.assignments(),
                        block.pos(),
                        span
                    );
                }
            }
        }
    }

    /// Every reachable branch of the §7.2 gadgets presents every Pauli tuple
    /// on both temporal interfaces (Lemma 2's joint transparency).
    #[test]
    fn gate_gadget_interfaces_are_jointly_transparent_in_every_branch() {
        for entry in GATE_GADGETS {
            let graph = entry
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            let zx = &crate::ZXGraph::try_from(&graph).unwrap();
            let table = zx.to_external_generator_table();
            let rows = table.iter().collect::<Vec<_>>();
            let (outputs, inputs): (Vec<usize>, Vec<usize>) = zx
                .nodes()
                .iter()
                .filter(|node| node.kind.is_port())
                .map(|node| node.id)
                .partition(|&id| zx.nodes()[id].is_output_port(zx));
            assert!(
                !inputs.is_empty() && !outputs.is_empty(),
                "{} ports",
                entry.id()
            );

            let resolves = graph
                .actions()
                .iter()
                .filter_map(|action| match action {
                    Action::Resolve { target, .. } => {
                        let node = zx
                            .nodes()
                            .iter()
                            .find(|node| node.pos == *target)
                            .expect("a resolve targets a node");
                        let crate::NodeKind::Selective(kind) = node.kind else {
                            panic!("a resolve targets a selective")
                        };
                        Some((node.id, kind, *target))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let targets = resolves
                .iter()
                .map(|&(_, _, target)| target)
                .collect::<Vec<_>>();
            let domain = graph.action_graph().resolve_value_domain(&targets).unwrap();
            assert!(!domain.values().is_empty(), "{} has no branch", entry.id());
            for values in domain.values() {
                let caps = resolves
                    .iter()
                    .zip(values)
                    .map(|(&(col, kind, _), &value)| {
                        let basis = if value {
                            kind.pauli_if_true()
                        } else {
                            kind.pauli_if_false()
                        };
                        (col, Pauli::from(basis))
                    })
                    .collect::<Vec<_>>();
                for (side, ports) in [("input", &inputs), ("output", &outputs)] {
                    let rank = admissible_interface_rank(&rows, &caps, ports);
                    assert_eq!(
                        rank,
                        2 * ports.len(),
                        "{} {side} interface is rank {rank} of {} when the caps read {caps:?}",
                        entry.id(),
                        2 * ports.len()
                    );
                }
            }
        }
    }

    /// Temporal stacks keep deadlines and add no backward dependency.
    #[test]
    fn stacked_gadgets_keep_deadlines_and_forward_dependencies() {
        let stacks: &[&[GalleryItem]] = &[
            &[GalleryItem::S, GalleryItem::T],
            &[GalleryItem::S, GalleryItem::THTH],
            &[GalleryItem::T, GalleryItem::S],
            &[GalleryItem::T, GalleryItem::THTH],
            &[GalleryItem::THTH, GalleryItem::S],
            &[GalleryItem::THTH, GalleryItem::T],
            &[GalleryItem::S, GalleryItem::T, GalleryItem::THTH],
            &[GalleryItem::S, GalleryItem::THTH, GalleryItem::T],
            &[GalleryItem::T, GalleryItem::S, GalleryItem::THTH],
            &[GalleryItem::T, GalleryItem::THTH, GalleryItem::S],
            &[GalleryItem::THTH, GalleryItem::S, GalleryItem::T],
            &[GalleryItem::THTH, GalleryItem::T, GalleryItem::S],
        ];
        for &stack in stacks {
            let label = stack
                .iter()
                .map(|item| item.id().to_string())
                .collect::<Vec<_>>()
                .join(" -> ");
            let mut composite = stack[0]
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            let mut deadlines = record_deadlines(&composite);
            let mut stage_of = vec![0usize; composite.actions().len()];
            for (stage, &item) in stack.iter().enumerate().skip(1) {
                let upper = item
                    .build()
                    .materialize_root_graph()
                    .expect("gallery flat projection");
                let shift = stack_offset(&composite, &upper);
                deadlines.extend(
                    record_deadlines(&upper)
                        .into_iter()
                        .map(|(name, top)| (name, top + shift.z)),
                );
                stage_of.extend(std::iter::repeat_n(stage, upper.actions().len()));
                composite = stack_in_time(composite, upper.shift_positions(shift).unwrap());
            }

            composite
                .validate()
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            let composite = composite
                .analyze_actions()
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            let zx = &crate::ZXGraph::from_block_graph_for_analysis(&composite).unwrap();
            for node in composite.action_graph().ordered_nodes() {
                let Action::Measure { name, .. } = &node.action else {
                    continue;
                };
                let stabilizer = node
                    .measurement_stabilizer
                    .as_ref()
                    .unwrap_or_else(|| panic!("{label}: {name} has no surface"));
                assert!(
                    zx.nodes()
                        .iter()
                        .filter(|node| node.is_output_port(zx))
                        .all(|node| stabilizer.paulis.get(node.id) == Pauli::I),
                    "{label}: {name} reaches an output port"
                );
                assert!(
                    support_top(stabilizer) <= deadlines[name],
                    "{label}: {name} decodes at layer {} instead of {}",
                    support_top(stabilizer),
                    deadlines[name]
                );
            }
            for (from, to, kind) in composite.action_graph().dependencies() {
                assert!(
                    stage_of[from] <= stage_of[to],
                    "{label}: {kind:?} dependency {from} -> {to} points back in time"
                );
            }
        }
    }

    /// Alpha-renamed repeated gadgets still satisfy (V1)-(V3).
    #[test]
    fn repeated_t_gadgets_compose_after_alpha_renaming() {
        let lower = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let upper = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let shift = stack_offset(&lower, &upper);
        let mut upper = upper.shift_positions(shift).unwrap();
        upper
            .set_actions(
                upper
                    .actions()
                    .into_iter()
                    .map(|action| match action {
                        Action::Measure { target, .. } => Action::Measure {
                            target,
                            name: "mzz_upper".into(),
                        },
                        Action::Resolve { target, .. } => Action::Resolve {
                            target,
                            condition: Expr::Var("mzz_upper".into()),
                        },
                        action => action,
                    })
                    .collect(),
            )
            .unwrap();
        let composite = stack_in_time(lower, upper);

        composite.validate().unwrap();
        composite.analyze_actions().unwrap();
    }

    /// The highest layer a surface touches — the earliest its parity decodes.
    fn support_top(stabilizer: &crate::Stabilizer) -> i32 {
        let nodes = stabilizer
            .interior_nodes
            .iter()
            .filter(|&(_, &pauli)| pauli != Pauli::I)
            .map(|(pos, _)| pos.z);
        let edges = stabilizer
            .interior_edges
            .iter()
            .filter(|&(_, &pauli)| pauli != Pauli::I)
            .flat_map(|(&(from, to), _)| [from.z, to.z]);
        nodes.chain(edges).max().expect("a surface has support")
    }

    fn record_deadlines(graph: &BlockGraph) -> HashMap<String, i32> {
        graph
            .stabilizers()
            .unwrap()
            .iter()
            .filter_map(|generator| {
                let name = generator.measurement_name()?;
                Some((name.to_owned(), support_top(&generator.stabilizer)))
            })
            .collect()
    }

    /// Places `upper`'s lowest port above `lower`'s highest port.
    fn stack_offset(lower: &BlockGraph, upper: &BlockGraph) -> IVec3 {
        let lower_out = output_port(lower);
        let upper_in = upper
            .blocks()
            .filter(|block| block.kind().is_port())
            .min_by_key(|block| block.pos().z)
            .expect("an input port")
            .pos();
        lower_out + IVec3::Z - upper_in
    }

    fn output_port(graph: &BlockGraph) -> IVec3 {
        graph
            .blocks()
            .filter(|block| block.kind().is_port())
            .max_by_key(|block| block.pos().z)
            .expect("an output port")
            .pos()
    }

    /// Joins an already-shifted `upper` onto `lower`.
    fn stack_in_time(mut lower: BlockGraph, upper: BlockGraph) -> BlockGraph {
        let seam_low = output_port(&lower);
        let seam_high = seam_low + IVec3::Z;
        let low_kind = lower
            .get_block(seam_low - IVec3::Z)
            .expect("the cube under the output port")
            .kind();
        let high_kind = upper
            .get_block(seam_high + IVec3::Z)
            .expect("the cube over the input port")
            .kind();

        // Removing the port takes its pipe with it.
        lower.remove_block(seam_low);
        lower.try_add_block(Block::new(seam_low, low_kind)).unwrap();
        lower
            .try_add_pipe(Pipe::new(seam_low - IVec3::Z, Direction::ZPLUS))
            .unwrap();
        for block in upper.blocks() {
            let block = if block.pos() == seam_high {
                Block::new(seam_high, high_kind)
            } else {
                block.clone()
            };
            lower.try_add_block(block).unwrap();
        }
        for pipe in upper.pipes() {
            lower.try_add_pipe(pipe.clone()).unwrap();
        }
        lower
            .try_add_pipe(Pipe::new(seam_low, Direction::ZPLUS))
            .unwrap();
        // One bulk replacement: adding a `measure` before its `resolve`
        // exists would fail the per-action validation.
        let mut actions = lower.actions();
        actions.extend(upper.actions());
        lower.set_actions(actions).unwrap();
        lower
    }

    /// Rank of admissible row combinations projected onto `ports` over GF(2).
    /// Cap-violation bits lead each vector, so later pivots span the kernel's
    /// port projection.
    fn admissible_interface_rank(
        rows: &[&bloq_utils::PauliString],
        caps: &[(usize, Pauli)],
        ports: &[usize],
    ) -> usize {
        let anticommutes = |lhs: Pauli, rhs: Pauli| {
            let (lhs, rhs) = (lhs as u8, rhs as u8);
            ((lhs & 1) & (rhs >> 1)) ^ ((lhs >> 1) & (rhs & 1)) == 1
        };
        assert!(caps.len() + 2 * ports.len() <= u64::BITS as usize);

        let mut basis = [0u64; u64::BITS as usize];
        for row in rows {
            let mut vector = 0u64;
            for (bit, &(col, cap)) in caps.iter().enumerate() {
                vector |= u64::from(anticommutes(row.get(col), cap)) << bit;
            }
            for (index, &col) in ports.iter().enumerate() {
                vector |= u64::from(row.get(col) as u8) << (caps.len() + 2 * index);
            }
            while vector != 0 {
                let pivot = &mut basis[vector.trailing_zeros() as usize];
                if *pivot == 0 {
                    *pivot = vector;
                    break;
                }
                vector ^= *pivot;
            }
        }
        basis[caps.len()..]
            .iter()
            .filter(|&&vector| vector != 0)
            .count()
    }

    #[test]
    fn test_flip_xz_basis_revalidates_gallery_entries_without_multiplex_ports() {
        for entry in GalleryItem::iter() {
            let source = entry.build();
            let graph = source
                .materialize_root_graph()
                .expect("gallery flat projection");
            let source_only =
                entry.in_category(GalleryCategory::AnalysisOnly) || !source.instances.is_empty();
            if graph
                .blocks()
                .any(|block| block.port_role() == Some(crate::PortRole::Multiplex))
            {
                assert!(
                    matches!(
                        graph.flip_xz_basis_lenient(),
                        Err(crate::BlockGraphError::BasisFlipUnsupportedMultiplexPort(_))
                    ),
                    "{} flips a multiplex Port",
                    entry.id()
                );
                continue;
            }
            let flipped = if source_only {
                graph.flip_xz_basis_lenient().unwrap()
            } else {
                graph
                    .flip_xz_basis()
                    .unwrap_or_else(|err| panic!("{} flip_xz_basis failed: {}", entry.id(), err))
            };
            let validation = if source_only {
                flipped.validate_source()
            } else {
                flipped.validate()
            };
            validation.unwrap_or_else(|err| {
                panic!("{} flipped graph became invalid: {}", entry.id(), err)
            });
        }
    }

    #[test]
    fn test_gallery_ids_roundtrip() {
        assert_eq!(
            "one_bit_adder".parse::<GalleryItem>(),
            Err(strum::ParseError::VariantNotFound)
        );
        for entry in GalleryItem::iter() {
            assert_eq!(
                entry
                    .id()
                    .parse::<GalleryItem>()
                    .unwrap_or_else(|err| panic!(
                        "gallery id {:?} failed to roundtrip: {}",
                        entry.id(),
                        err
                    )),
                entry
            );
        }
    }
}
