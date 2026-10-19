//! Fixed distance presentations of a fully certified source relation.

use bloq_utils::boolean::{BooleanOp, DECISION_FALSE};

use super::*;
use crate::{Action, ActionDag, GuardedTopology, RuntimeBasisError, StabilizerError};

impl ModuleSummary {
    /// Certify the complete source, then contract its signed relation with a
    /// reachable tuple of selective choices for a static distance oracle.
    ///
    /// Pins follow expanded `(z, x, y)` block order. The returned graph retains
    /// its original selective and T blocks and its guarded action analysis;
    /// only the returned generators use the fixed Clifford presentation.
    ///
    /// # Errors
    ///
    /// Rejects invalid sources before examining pins, structural branches,
    /// incomplete or unreachable pin tuples, unavailable fixed readouts, and
    /// exhaustion of this summary's certification limits.
    #[doc(hidden)]
    pub fn materialize_clifford_distance_relation(
        &self,
        pins: &[bool],
    ) -> Result<(BlockGraph, StabilizerGenerators), ModuleCertificationError> {
        let graph_error = |source| match source {
            crate::BlockGraphError::Stabilizer(error) => self.runtime_error(error.into()),
            source => ModuleCertificationError::Graph {
                module: self.name.clone(),
                source,
            },
        };
        let linked = &self.prepared.linked;
        let (min_z, max_z) = linked
            .graph
            .spans()
            .map_or((0, 0), |(_, _, z)| (*z.start(), *z.end()));
        let normalize = |position| crate::graph::normalize_z_position(position, min_z, max_z);
        let mut source = linked
            .graph
            .with_zero_min_z()
            .map_err(graph_error)?
            .fix_shadowed_faces();
        let sites = linked
            .sites
            .iter()
            .map(|(&position, site)| Ok((normalize(position).map_err(graph_error)?, site.clone())))
            .collect::<Result<HashMap<_, _>, ModuleCertificationError>>()?;

        // Child interfaces were certified during preparation. Plan the complete
        // composed source before a pin can hide an invalid alternative or C0.
        let mut plan = self
            .compose(
                GuardedTopology::new(&source, self.limits).map_err(graph_error)?,
                &sites,
            )
            .and_then(crate::GuardedSurfaceSpace::plan_readouts)
            .map_err(graph_error)?;
        let mut source_dag = source
            .build_action_graph(&source.actions())
            .map_err(graph_error)?;
        source_dag
            .attach_guarded_dependencies(plan.action_dependencies().map_err(graph_error)?)
            .map_err(graph_error)?;
        source.install_action_graph(source_dag, None);
        if !plan.topology.branches.is_empty() {
            return Err(
                ModuleCertificationError::CliffordDistanceStructuralBranches {
                    module: self.name.clone(),
                },
            );
        }

        let mut selectives = source
            .blocks()
            .filter_map(|block| match block.kind() {
                BlockKind::Selective(kind) => Some((block.pos(), kind)),
                _ => None,
            })
            .collect::<Vec<_>>();
        selectives.sort_unstable_by_key(|(position, _)| (position.z, position.x, position.y));
        if pins.len() != selectives.len() {
            return Err(ModuleCertificationError::CliffordDistancePinCount {
                module: self.name.clone(),
                expected: selectives.len(),
                actual: pins.len(),
            });
        }
        let resolves = plan
            .topology
            .resolves
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let mut reachable = plan.topology.domain;
        let mut fills = Vec::with_capacity(pins.len());
        for (&(position, kind), &pin) in selectives.iter().zip(pins) {
            let selector = *resolves.get(&position).ok_or_else(|| {
                invalid(
                    &self.name,
                    "certified selective has no source resolve condition",
                )
            })?;
            let required = if pin {
                selector
            } else {
                plan.topology
                    .diagram
                    .negate(selector)
                    .map_err(|error| self.runtime_error(RuntimeBasisError::from(error)))?
            };
            reachable = plan
                .topology
                .diagram
                .apply(BooleanOp::And, reachable, required)
                .map_err(|error| self.runtime_error(RuntimeBasisError::from(error)))?;
            fills.push((
                position,
                if pin {
                    kind.pauli_if_true()
                } else {
                    kind.pauli_if_false()
                },
            ));
        }
        if reachable == DECISION_FALSE {
            return Err(ModuleCertificationError::CliffordDistanceUnreachablePins {
                module: self.name.clone(),
            });
        }
        drop(plan);

        let zx = ZXGraph::from_block_graph_for_analysis(&source)
            .map_err(|error| graph_error(error.into()))?;
        let certificate = self.materialize_certificate_rows(&zx, normalize)?;
        let (mut fixed, restricted) = zx
            .restrict_selective_relation_with_limits(&certificate, &fills, self.limits)
            .map_err(|error| self.runtime_error(error))?;
        drop(certificate);
        // With structural branches excluded and all caps fixed, the complete
        // physical relation has no remaining source-outcome dependence.
        fixed.action_graph = fixed_measurement_dag(&fixed);
        fixed
            .validate_for_program()
            .map_err(|error| graph_error(error.into()))?;
        let mut generators = fixed
            .stabilizers_from_external_basis_with_limits(restricted.rows(), self.limits)
            .map_err(|error| self.runtime_error(error.into()))?;
        let restricted_rank = restricted.rows().len();
        drop(restricted);
        self.validate_fixed_distance_presentation(&fixed, restricted_rank, &mut generators)?;
        Ok((source, generators))
    }

    fn validate_fixed_distance_presentation(
        &self,
        zx: &ZXGraph,
        restricted_rank: usize,
        generators: &mut StabilizerGenerators,
    ) -> Result<(), ModuleCertificationError> {
        let unsatisfied =
            || self.runtime_error(StabilizerError::ComposedPresentationUnsatisfied.into());
        if generators.basis_rank() != restricted_rank {
            return Err(self.runtime_error(
                StabilizerError::BasisRankDeficient {
                    expected: restricted_rank,
                    actual: generators.basis_rank(),
                }
                .into(),
            ));
        }
        let names = zx
            .action_graph()
            .ordered_nodes()
            .filter_map(|node| match &node.action {
                Action::Measure { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .enumerate()
            .map(|(ordinal, name)| (name, ordinal))
            .collect::<HashMap<_, _>>();
        let mut seen = FxHashSet::default();
        for generator in &generators.generators {
            if let Some(name) = generator.measurement_name() {
                if !names.contains_key(name) || !seen.insert(name) {
                    return Err(unsatisfied());
                }
                for port in zx.nodes().iter().filter(|node| node.is_output_port(zx)) {
                    let support = generator.stabilizer.paulis.get(port.id);
                    if support != Pauli::I
                        && !(port.role == crate::PortRole::Multiplex && support == Pauli::Z)
                    {
                        return Err(self.runtime_error(
                            StabilizerError::MeasurementSurfaceTouchesOutputPort {
                                name: name.to_owned(),
                                port: port.pos,
                            }
                            .into(),
                        ));
                    }
                }
            }
        }
        if seen.len() != names.len() {
            return Err(unsatisfied());
        }
        // Fixed rows carry no coefficient recipes; changing their public order
        // leaves private rank completion and all physical supports untouched.
        generators.generators.sort_by_key(|generator| {
            generator
                .measurement_name()
                .map_or(usize::MAX, |name| names[name])
        });
        let mut dag = zx.action_graph().clone();
        dag.attach_readout_dependencies(&generators.generators, Some(zx))
            .map_err(|error| match error {
                crate::BlockGraphError::Stabilizer(error) => self.runtime_error(error.into()),
                _ => unsatisfied(),
            })?;
        generators.zx_graph.action_graph = dag;
        generators
            .validate_measurements_close_before_outputs_with_limits(self.limits)
            .map_err(|error| self.runtime_error(error))
    }
}

fn fixed_measurement_dag(zx: &ZXGraph) -> ActionDag {
    let measurements = zx
        .action_graph()
        .ordered_nodes()
        .filter(|node| matches!(node.action, Action::Measure { .. }))
        .collect::<Vec<_>>();
    let actions = measurements
        .iter()
        .map(|node| node.action.clone())
        .collect::<Vec<_>>();
    let mut dag = ActionDag::from_actions(&actions);
    for (ordinal, node) in measurements.into_iter().enumerate() {
        if let Some(observable) = node.measurement {
            dag.set_measurement_observable(ordinal, observable)
                .expect("filtered action ordinal names a measurement");
        }
    }
    dag
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlated_source() -> BlockGraph {
        crate::parse_inline_graph(
            r#"BLOG 1.0
module main {
    in enable
    in a: data = 0
    in b: data = 3
    0: Port [0,0,0] role=input
    1: XZX [0,0,1]
    2: YX [0,0,2]
    3: Port [2,0,0] role=input
    4: XZX [2,0,1]
    5: YX [2,0,2]
    0 -> +Z
    1 -> +Z
    3 -> +Z
    4 -> +Z
    resolve 2 if enable
    resolve 5 if !enable
    second = measure 5
    first = measure 2
}
"#,
        )
        .unwrap()
    }

    #[test]
    fn distance_relation_checks_correlated_pins_and_keeps_source_measure_order() {
        let summary = correlated_source()
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap();
        assert!(matches!(
            summary.materialize_clifford_distance_relation(&[true, true]),
            Err(ModuleCertificationError::CliffordDistanceUnreachablePins { .. })
        ));
        assert!(matches!(
            summary.materialize_clifford_distance_relation(&[false]),
            Err(ModuleCertificationError::CliffordDistancePinCount {
                expected: 2,
                actual: 1,
                ..
            })
        ));
        for pins in [[true, false], [false, true]] {
            let (source, fixed) = summary
                .materialize_clifford_distance_relation(&pins)
                .unwrap();
            assert!(source.action_graph().is_analyzed());
            assert_eq!(
                source
                    .blocks()
                    .filter(|block| block.kind().is_selective())
                    .count(),
                2
            );
            assert!(
                source
                    .actions()
                    .iter()
                    .any(|action| matches!(action, Action::Resolve { .. }))
            );
            assert!(
                fixed
                    .zx_graph
                    .nodes()
                    .iter()
                    .all(|node| !matches!(node.kind, crate::NodeKind::Selective(_)))
            );
            assert!(
                fixed
                    .generators
                    .iter()
                    .all(|generator| generator.readout_plan().is_none())
            );
            assert_eq!(
                fixed
                    .generators
                    .iter()
                    .filter_map(|generator| generator.measurement_name())
                    .collect::<Vec<_>>(),
                ["second", "first"],
            );
            let repeated = summary
                .materialize_clifford_distance_relation(&pins)
                .unwrap()
                .1;
            assert_eq!(fixed.generators, repeated.generators);
        }
    }

    #[test]
    fn distance_relation_validates_source_closure_before_pin_count() {
        let source = crate::parse_inline_graph(
            r#"BLOG 1.0
module main {
    out q: data = 2
    1: XZX [0,0,1]
    2: Port [0,0,2] role=output
    3: T [1,0,0]
    4: XZX [1,0,1]
    5: YX [1,0,2]
    1 -> +Z
    1 -> +X
    3 -> +Z
    5 -> -Z
    mzz = measure 1 -> +X
    resolve 5 if mzz
}
"#,
        )
        .unwrap();
        let summary = source
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap();
        for pins in [&[][..], &[false][..], &[true][..]] {
            assert!(matches!(
                summary.materialize_clifford_distance_relation(pins),
                Err(ModuleCertificationError::Runtime { .. }
                    | ModuleCertificationError::Graph { .. })
            ));
        }
    }

    #[test]
    fn distance_relation_keeps_explicit_source_boolean_limit() {
        let mut summary = correlated_source()
            .summarize_root(ModuleCertificationLimits::DEFAULT)
            .unwrap()
            .as_ref()
            .clone();
        // Tighten an already-created certificate to exercise the adapter itself.
        summary.limits.max_boolean_steps = 0;
        assert!(matches!(
            summary.materialize_clifford_distance_relation(&[]),
            Err(ModuleCertificationError::ResourceLimited { .. })
        ));
    }
}
