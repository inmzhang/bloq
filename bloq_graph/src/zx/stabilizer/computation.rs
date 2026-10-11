//! Canonical stabilizer-table construction and bounded fallback search.

use std::cell::LazyCell;

use bloq_utils::{Pauli, PauliString, PhasedPauliString};

use super::super::{NodeKind, ZXGraph, graph::pauli_at_small_node};
use super::affine::{
    AffineTrackedSolution, ExactPauliConstraint, PauliComponentConstraint, materialize_tracked_row,
    solve_affine_tracked, solve_affine_tracked_components, visit_affine_coefficients,
};
use super::basis::{
    CoeffSpan, CoeffVec, PauliCombinationBasis, PauliSpan, TrackedBasis,
    complete_from_initial_basis_fast, row_rank,
};
use super::engine::StabilizerBasis;
use super::measurement::{
    MeasurementMetadata, MeasurementSpec, MeasurementSurfaceRow, accepted_surface_deadline,
    diagonalize_measurement_columns, extract_measurement_prefix_with_constraints,
    extract_temporal_measurement_prefix, materialize_measurement_first_basis, measurement_specs,
    measurement_surface_row_matches_acceptance, temporal_projection_layout,
};
use super::selective::{
    SelectiveConstraint, add_selective_fixing_rows, canonicalize_boundary_suffix,
    choose_independent_affine_solution, collect_selective_constraints,
    diagonalize_selective_fixing_rows, reconstruct_boundary_supported_row,
};
use super::{
    SearchBudget, SelectiveFixingTarget, StabilizerError, StabilizerGenerators, StabilizerRowKind,
};

#[derive(Debug)]
pub(crate) struct CanonicalStabilizerTable {
    pub(crate) basis: StabilizerBasis,
    row_phases: Vec<u8>,
    /// The measurement specs, computed once during canonicalization and carried
    /// so [`Self::into_stabilizers`] reuses them instead of re-deriving from the
    /// graph (they depend only on `zx`).
    measurement_specs: Vec<MeasurementSpec>,
    temporal_measurements: bool,
}

impl CanonicalStabilizerTable {
    pub(crate) fn into_stabilizers(
        self,
        zx: &ZXGraph,
    ) -> Result<StabilizerGenerators, StabilizerError> {
        materialize_measurement_first_basis(
            zx,
            self.basis.rows,
            self.basis.kinds,
            self.basis.public_len,
            self.row_phases,
            &self.measurement_specs,
        )
    }

    pub(crate) fn used_temporal_measurements(&self) -> bool {
        self.temporal_measurements
    }
}

// ponytail: conservative measured cutoff; lower it only after a release stage sweep
// shows that the frontier wins on narrower tables too.
const TEMPORAL_FRONTIER_MIN_COLUMNS: usize = 16_384;

pub(crate) fn canonicalize_stabilizer_table(
    zx: &ZXGraph,
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    if zx.total_ids() >= TEMPORAL_FRONTIER_MIN_COLUMNS {
        let specs = measurement_specs(zx);
        if specs.len() >= 32 {
            let selective_constraints = collect_selective_constraints(zx);
            let metadata = MeasurementMetadata::collect(zx);
            let measurement_cols = metadata.sorted_columns();
            let (semantic_columns, node_gates) =
                temporal_projection_layout(zx, &metadata, &selective_constraints);
            let external_basis =
                zx.to_temporal_signed_external_generator_table(&semantic_columns, &node_gates);
            let basis_rows = external_basis
                .iter()
                .map(|row| row.paulis.clone())
                .collect::<Vec<_>>();
            if let Some(table) = try_temporal_measurement_table(
                CanonicalizationInputs {
                    zx,
                    basis_rows: &basis_rows,
                    external_basis: Some(&external_basis),
                    selective_constraints: &selective_constraints,
                    measurement_metadata: &metadata,
                    measurement_cols: &measurement_cols,
                },
                &specs,
                budget,
            )? {
                return Ok(table);
            }
        }
    }

    let (raw_external, signed_external) = zx.to_external_generator_table_with_signed();
    canonicalize_stabilizer_table_with_external_basis(zx, &raw_external, &signed_external, budget)
}

pub(crate) fn canonicalize_stabilizer_table_with_external_basis(
    zx: &ZXGraph,
    raw_external: &[PauliString],
    external_basis: &[PhasedPauliString],
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    canonicalize_stabilizer_table_inner(zx, raw_external, Some(external_basis), true, true, budget)
}

pub(crate) fn canonicalize_stabilizer_table_with_measurement_prefix(
    zx: &ZXGraph,
    raw_external: &[PauliString],
    external_basis: &[PhasedPauliString],
    cached: &[(&str, PauliString)],
    budget: &mut SearchBudget,
) -> Result<Option<CanonicalStabilizerTable>, StabilizerError> {
    budget.check_matrix(
        raw_external.len().saturating_mul(4),
        zx.total_ids(),
        raw_external.len().saturating_mul(4),
        raw_external.len(),
    )?;
    let selective_constraints = collect_selective_constraints(zx);
    let metadata = MeasurementMetadata::collect(zx);
    let measurement_cols = metadata.sorted_columns();
    let specs = measurement_specs(zx);
    if specs.len() != cached.len()
        || specs
            .iter()
            .zip(cached)
            .any(|(spec, (name, _))| spec.name != *name)
    {
        return Ok(None);
    }
    let combinations = LazyCell::new(|| PauliCombinationBasis::new(raw_external));
    let mut measurement_rows = Vec::with_capacity(specs.len());
    for (spec, (_, row)) in specs.iter().zip(cached) {
        budget.visit("cached measurement presentation states")?;
        let mut row = row.clone();
        zx.clear_cross_centers(std::slice::from_mut(&mut row));
        let Some(combination) = combinations.solve(&row) else {
            return Ok(None);
        };
        zx.reconstruct_cross_center(std::slice::from_mut(&mut row));
        if !measurement_surface_row_matches_acceptance(zx, &metadata, &spec.name, &row, true)
            || selective_constraints
                .iter()
                .any(|constraint| row.get(constraint.col) == constraint.forbidden)
        {
            return Ok(None);
        }
        measurement_rows.push(MeasurementSurfaceRow {
            name: spec.name.clone(),
            deadline: accepted_surface_deadline(zx, &selective_constraints, spec, &row),
            row,
            combination,
        });
    }
    match finish_canonical_stabilizer_table(
        zx,
        raw_external,
        Some(external_basis),
        &selective_constraints,
        &metadata,
        &measurement_cols,
        specs,
        &measurement_rows,
        budget,
    ) {
        Ok(table) => Ok(Some(table)),
        Err(error) if error.is_interrupted() => Err(error),
        Err(_) => Ok(None),
    }
}

pub(crate) fn canonicalize_stabilizer_table_legacy(
    zx: &ZXGraph,
    raw_external: &[PauliString],
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    canonicalize_stabilizer_table_inner(zx, raw_external, None, true, false, budget)
}
fn canonicalize_stabilizer_table_inner(
    zx: &ZXGraph,
    raw_external: &[PauliString],
    external_basis: Option<&[PhasedPauliString]>,
    prefer_fixing_safe_readers: bool,
    allow_temporal: bool,
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    budget.check_matrix(
        raw_external.len().saturating_mul(4),
        zx.total_ids(),
        raw_external.len().saturating_mul(4),
        raw_external.len(),
    )?;
    debug_assert_eq!(
        row_rank(raw_external, zx.total_ids),
        raw_external.len(),
        "callers pass an already-independent canonical external basis",
    );
    let basis_rows = raw_external;
    let selective_constraints = &collect_selective_constraints(zx);
    let measurement_metadata = &MeasurementMetadata::collect(zx);
    let measurement_cols = measurement_metadata.sorted_columns();
    // Provisional fixings steer cycle avoidance; authoritative rows still decide.
    let self_reader_fixings = if selective_constraints.is_empty() || !prefer_fixing_safe_readers {
        Vec::new()
    } else {
        let mut fixing_basis = TrackedBasis::from_parts(Vec::new(), Vec::new());
        match add_selective_fixing_rows(
            &mut fixing_basis,
            basis_rows,
            selective_constraints,
            &measurement_cols,
            zx,
            budget,
        ) {
            Ok(kinds) => fixing_basis.rows.into_iter().zip(kinds).collect(),
            Err(error) if error.is_interrupted() => return Err(error),
            Err(_) => Vec::new(),
        }
    };
    // Computed once here and threaded into both the prefix extraction below and
    // (via the returned table) `materialize_measurement_first_basis`, rather
    // than each re-deriving it from `zx`.
    let measurement_specs = measurement_specs(zx);
    if allow_temporal
        && measurement_specs.len() >= 32
        && let Some(table) = try_temporal_measurement_table(
            CanonicalizationInputs {
                zx,
                basis_rows,
                external_basis,
                selective_constraints,
                measurement_metadata,
                measurement_cols: &measurement_cols,
            },
            &measurement_specs,
            budget,
        )?
    {
        return Ok(table);
    }

    let mut working_basis = TrackedBasis::new(basis_rows.to_vec());
    let measurement_rows = extract_measurement_prefix_with_constraints(
        zx,
        &mut working_basis,
        measurement_metadata,
        selective_constraints,
        &self_reader_fixings,
        &measurement_specs,
        budget,
    )?;
    let table = finish_canonical_stabilizer_table(
        zx,
        basis_rows,
        external_basis,
        selective_constraints,
        measurement_metadata,
        &measurement_cols,
        measurement_specs,
        &measurement_rows,
        budget,
    )?;
    if !self_reader_fixings.is_empty() {
        let final_fixings = table
            .basis
            .rows
            .iter()
            .cloned()
            .zip(table.basis.kinds.iter().cloned())
            .filter(|(_, kind)| kind.is_selective_fixing())
            .collect::<Vec<_>>();
        if self_reader_fixings != final_fixings {
            // Retry unsteered when canonicalization changes a provisional fixing.
            return canonicalize_stabilizer_table_inner(
                zx,
                raw_external,
                external_basis,
                false,
                allow_temporal,
                budget,
            );
        }
    }
    Ok(table)
}
#[expect(
    clippy::too_many_arguments,
    reason = "finalization consumes the independently derived table parts"
)]
fn finish_canonical_stabilizer_table(
    zx: &ZXGraph,
    basis_rows: &[PauliString],
    external_basis: Option<&[PhasedPauliString]>,
    selective_constraints: &[SelectiveConstraint],
    measurement_metadata: &MeasurementMetadata,
    measurement_cols: &[usize],
    mut measurement_specs: Vec<MeasurementSpec>,
    measurement_rows: &[MeasurementSurfaceRow],
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    // The accepted surfaces may close later than their specs requested (a late
    // support limit, or an output-port anchor). Record what was accepted so the
    // extent check in `into_stabilizers` measures against the real promise.
    for spec in &mut measurement_specs {
        if let Some(surface) = measurement_rows.iter().find(|row| row.name == spec.name) {
            spec.deadline = surface.deadline;
        }
    }

    if measurement_rows.is_empty() && selective_constraints.is_empty() {
        let mut rows = basis_rows.to_vec();
        let row_phases = external_basis.map_or_else(
            || zx.external_stabilizer_row_phases(&rows),
            |basis| zx.row_phases_against(&rows, basis),
        );
        zx.reconstruct_cross_center(&mut rows);
        return Ok(CanonicalStabilizerTable {
            basis: StabilizerBasis::all_logical(rows),
            row_phases,
            measurement_specs,
            temporal_measurements: false,
        });
    }

    let measurement_names = measurement_rows
        .iter()
        .map(|surface| surface.name.clone())
        .collect::<Vec<_>>();
    let mut final_basis = measurement_prefix_basis(measurement_rows);
    let mut row_kinds = measurement_names
        .iter()
        .map(|name| StabilizerRowKind::Measurement { name: name.clone() })
        .collect::<Vec<_>>();
    let fixing_kinds = add_selective_fixing_rows(
        &mut final_basis,
        basis_rows,
        selective_constraints,
        measurement_cols,
        zx,
        budget,
    )?;
    row_kinds.extend(fixing_kinds);
    finish_canonical_from_tagged_prefix(
        CanonicalizationInputs {
            zx,
            basis_rows,
            external_basis,
            selective_constraints,
            measurement_metadata,
            measurement_cols,
        },
        measurement_specs,
        &measurement_names,
        final_basis,
        row_kinds,
        false,
        budget,
    )
}

#[derive(Clone, Copy)]
struct CanonicalizationInputs<'a> {
    zx: &'a ZXGraph,
    basis_rows: &'a [PauliString],
    external_basis: Option<&'a [PhasedPauliString]>,
    selective_constraints: &'a [SelectiveConstraint],
    measurement_metadata: &'a MeasurementMetadata,
    measurement_cols: &'a [usize],
}

fn try_temporal_measurement_table(
    inputs: CanonicalizationInputs<'_>,
    measurement_specs: &[MeasurementSpec],
    budget: &mut SearchBudget,
) -> Result<Option<CanonicalStabilizerTable>, StabilizerError> {
    let Some(measurement_rows) = extract_temporal_measurement_prefix(
        inputs.zx,
        inputs.basis_rows,
        inputs.measurement_metadata,
        inputs.selective_constraints,
        measurement_specs,
        budget,
    )?
    else {
        return Ok(None);
    };
    let mut table = match finish_canonical_stabilizer_table(
        inputs.zx,
        inputs.basis_rows,
        inputs.external_basis,
        inputs.selective_constraints,
        inputs.measurement_metadata,
        inputs.measurement_cols,
        measurement_specs.to_vec(),
        &measurement_rows,
        budget,
    ) {
        Ok(table) => table,
        Err(error) if error.is_interrupted() => return Err(error),
        Err(_) => return Ok(None),
    };
    table.temporal_measurements = true;
    Ok(Some(table))
}

fn finish_canonical_from_tagged_prefix(
    inputs: CanonicalizationInputs<'_>,
    measurement_specs: Vec<MeasurementSpec>,
    measurement_names: &[String],
    mut final_basis: TrackedBasis,
    row_kinds: Vec<StabilizerRowKind>,
    allow_normalized_fallback: bool,
    budget: &mut SearchBudget,
) -> Result<CanonicalStabilizerTable, StabilizerError> {
    let target_rank = inputs.basis_rows.len();
    let width = inputs.zx.total_ids;
    let measurement_count = measurement_names.len();
    let frozen_measurement_coeffs = final_basis.coeffs[..measurement_count].to_vec();
    let frozen_measurement_rows = final_basis.rows[..measurement_count].to_vec();
    let tagged_prefix_len = row_kinds.len();
    complete_from_initial_basis_fast(
        &mut final_basis,
        inputs.basis_rows,
        &frozen_measurement_rows,
        &frozen_measurement_coeffs,
        inputs.measurement_cols,
        width,
    );
    diagonalize_measurement_columns(
        &mut final_basis,
        measurement_names,
        inputs.measurement_metadata,
        inputs.selective_constraints,
        &row_kinds[measurement_count..],
    );
    canonicalize_boundary_suffix(&mut final_basis.rows[tagged_prefix_len..], inputs.zx);

    debug_assert_eq!(final_basis.coeffs.len(), target_rank);
    // Rank repair mutates rows independently of their coefficients, so this
    // boundary consumes the coefficient-tracked `TrackedBasis` and yields a
    // plain `StabilizerBasis`: the now-stale provenance is dropped rather than
    // left readable.
    let mut basis = ensure_full_rank(
        final_basis,
        row_kinds,
        tagged_prefix_len,
        inputs.basis_rows,
        width,
        target_rank,
    )?;
    basis.constrain_readout_selective_support(
        inputs.selective_constraints,
        inputs.zx,
        allow_normalized_fallback,
        budget,
    )?;

    inputs.zx.clear_cross_centers(&mut basis.rows);
    let row_phases = inputs.external_basis.map_or_else(
        || inputs.zx.external_stabilizer_row_phases(&basis.rows),
        |external| inputs.zx.row_phases_against(&basis.rows, external),
    );
    inputs.zx.reconstruct_cross_center(&mut basis.rows);
    Ok(CanonicalStabilizerTable {
        basis,
        row_phases,
        measurement_specs,
        temporal_measurements: false,
    })
}

/// Complete-search result. `witness` preserves a valid basis for diagnosing a
/// global constraint when no `representable` basis exists.
#[derive(Default)]
pub(crate) struct CompleteStabilizerSearch {
    pub(crate) representable: Option<StabilizerGenerators>,
    pub(crate) witness: Option<StabilizerGenerators>,
    pub(crate) failure: Option<StabilizerError>,
}

struct CompleteMeasurementSpace {
    spec: MeasurementSpec,
    solutions: Vec<AffineTrackedSolution>,
}

struct CompleteSearchContext<'a> {
    zx: &'a ZXGraph,
    basis_rows: &'a [PauliString],
    external_basis: Option<&'a [PhasedPauliString]>,
    basis_coeffs: &'a [CoeffVec],
    selective_constraints: &'a [SelectiveConstraint],
    measurement_metadata: &'a MeasurementMetadata,
    measurement_cols: &'a [usize],
    specs: &'a [MeasurementSpec],
    spaces: &'a [CompleteMeasurementSpace],
}

#[derive(Default)]
struct CompleteSearchProgress {
    witness: Option<StabilizerGenerators>,
    failure: Option<StabilizerError>,
    max_measurements: usize,
    reached_measurement_prefix: bool,
}

/// Exhaustively backtracks over each record's affine cosets and kernel. This is
/// the finite, exponential fallback behind the deterministic surface ladder.
pub(crate) fn complete_stabilizer_search(
    zx: &ZXGraph,
    basis_rows: &[PauliString],
    external_basis: Option<&[PhasedPauliString]>,
    budget: &mut SearchBudget,
) -> Result<CompleteStabilizerSearch, StabilizerError> {
    budget.visit("complete stabilizer search states")?;
    budget.check_matrix(
        basis_rows.len().saturating_mul(2),
        zx.total_ids(),
        basis_rows.len(),
        basis_rows.len(),
    )?;
    let selective_constraints = collect_selective_constraints(zx);
    let measurement_metadata = MeasurementMetadata::collect(zx);
    let measurement_cols = measurement_metadata.sorted_columns();
    let specs = measurement_specs(zx);
    let tracked = TrackedBasis::new(basis_rows.to_vec());
    let mut spaces = Vec::with_capacity(specs.len());
    let mut retained_words = 0;
    for spec in &specs {
        let mut solutions = Vec::new();
        for pauli in measurement_metadata.matching_paulis(&spec.name) {
            for constraints in materialized_pauli_constraint_cases(
                zx,
                measurement_metadata.column(&spec.name),
                pauli,
            ) {
                budget.visit("complete affine solve states")?;
                budget.check_affine(&tracked.rows, &tracked.coeffs, constraints.len())?;
                // The next solution can retain at most one particular vector
                // and one coefficient vector per source basis row.
                budget.check_words(
                    retained_words
                        + (basis_rows.len() as u128 + 1) * basis_rows.len().div_ceil(64) as u128,
                )?;
                if let Some(solution) =
                    solve_affine_tracked_components(&tracked.rows, &tracked.coeffs, &constraints)
                {
                    retained_words += (solution.kernel_coeffs.len() as u128 + 1)
                        * solution.particular_coeff.len().div_ceil(64) as u128;
                    solutions.push(solution);
                }
            }
        }
        spaces.push(CompleteMeasurementSpace {
            spec: spec.clone(),
            solutions,
        });
    }
    let context = CompleteSearchContext {
        zx,
        basis_rows,
        external_basis,
        basis_coeffs: &tracked.coeffs,
        selective_constraints: &selective_constraints,
        measurement_metadata: &measurement_metadata,
        measurement_cols: &measurement_cols,
        specs: &specs,
        spaces: &spaces,
    };
    let mut surfaces = Vec::with_capacity(spaces.len());
    let mut progress = CompleteSearchProgress::default();
    let representable = search_complete_measurement_prefix(
        &context,
        0,
        &mut surfaces,
        &CoeffSpan::new(basis_rows.len()),
        &PauliSpan::new(zx.total_ids),
        &mut progress,
        budget,
    );
    if representable.is_none()
        && progress.witness.is_none()
        && !progress.reached_measurement_prefix
        && !search_was_interrupted(&progress)
    {
        let mut failure = StabilizerError::BasisRankDeficient {
            expected: context.spaces.len(),
            actual: progress.max_measurements,
        };
        for space in context.spaces {
            if !measurement_space_has_admissible_row(&context, space, budget)? {
                failure = StabilizerError::MeasurementSurfaceUnavailable {
                    mvar: space.spec.name.clone(),
                };
                break;
            }
        }
        progress.failure = Some(failure);
    }
    Ok(CompleteStabilizerSearch {
        representable,
        witness: progress.witness,
        failure: progress.failure,
    })
}

/// Affine cases for the Pauli visible after `reconstruct_cross_center`.
/// Cross-centre support is an OR over incident arms: absence is one conjunction;
/// presence is the union of cases selecting one crossing arm.
fn materialized_pauli_constraint_cases(
    zx: &ZXGraph,
    col: usize,
    pauli: Pauli,
) -> Vec<Vec<PauliComponentConstraint>> {
    let exact = || {
        vec![
            PauliComponentConstraint {
                col,
                basis: Pauli::X,
                present: pauli & Pauli::X,
            },
            PauliComponentConstraint {
                col,
                basis: Pauli::Z,
                present: pauli & Pauli::Z,
            },
        ]
    };
    let Some(node) = zx.nodes.get(col) else {
        return vec![exact()];
    };
    let cross = match node.kind {
        NodeKind::X => Pauli::X,
        NodeKind::Z => Pauli::Z,
        _ => return vec![exact()],
    };
    let arm = cross.flip();
    let base = vec![PauliComponentConstraint {
        col,
        basis: arm,
        present: pauli & arm,
    }];
    let incident = zx
        .neighbor_edges(node.id)
        .map(|(_, edge_id)| {
            let edge = zx.edge_by_id(edge_id);
            (edge.id, pauli_at_small_node(edge, node.id, cross))
        })
        .collect::<Vec<_>>();
    if pauli & cross {
        return incident
            .into_iter()
            .map(|(edge_col, edge_cross)| {
                let mut constraints = base.clone();
                constraints.push(PauliComponentConstraint {
                    col: edge_col,
                    basis: edge_cross,
                    present: true,
                });
                constraints
            })
            .collect();
    }

    let mut constraints = base;
    constraints.extend(incident.into_iter().map(|(edge_col, edge_cross)| {
        PauliComponentConstraint {
            col: edge_col,
            basis: edge_cross,
            present: false,
        }
    }));
    vec![constraints]
}

fn admissible_measurement_row(
    context: &CompleteSearchContext<'_>,
    space: &CompleteMeasurementSpace,
    combination: &CoeffVec,
) -> Option<PauliString> {
    let mut row = materialize_tracked_row(combination, context.basis_rows, context.zx.total_ids);
    if context
        .selective_constraints
        .iter()
        .any(|constraint| row.get(constraint.col) == constraint.forbidden)
    {
        return None;
    }
    context
        .zx
        .reconstruct_cross_center(std::slice::from_mut(&mut row));
    (context.measurement_metadata.support_matches(
        &space.spec.name,
        row.get(context.measurement_metadata.column(&space.spec.name)),
    ) && context.zx.row_has_external_edge_pair_support(&row))
    .then_some(row)
}

fn measurement_space_has_admissible_row(
    context: &CompleteSearchContext<'_>,
    space: &CompleteMeasurementSpace,
    budget: &mut SearchBudget,
) -> Result<bool, StabilizerError> {
    for solution in &space.solutions {
        if visit_affine_coefficients(
            &solution.particular_coeff,
            &solution.kernel_coeffs,
            budget,
            "complete affine diagnostic states",
            |combination, _| admissible_measurement_row(context, space, combination).is_some(),
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn search_complete_measurement_prefix(
    context: &CompleteSearchContext<'_>,
    index: usize,
    surfaces: &mut Vec<MeasurementSurfaceRow>,
    coeff_span: &CoeffSpan,
    row_span: &PauliSpan,
    progress: &mut CompleteSearchProgress,
    budget: &mut SearchBudget,
) -> Option<StabilizerGenerators> {
    if search_was_interrupted(progress) {
        return None;
    }
    if let Err(error) = budget.visit("complete measurement prefix states") {
        progress.failure = Some(error);
        return None;
    }
    progress.max_measurements = progress.max_measurements.max(index);
    if index == context.spaces.len() {
        progress.reached_measurement_prefix = true;
        let mut specs = context.specs.to_vec();
        for spec in &mut specs {
            if let Some(surface) = surfaces.iter().find(|row| row.name == spec.name) {
                spec.deadline = surface.deadline;
            }
        }
        let names = surfaces
            .iter()
            .map(|surface| surface.name.clone())
            .collect::<Vec<_>>();
        let basis = measurement_prefix_basis(surfaces);
        let kinds = names
            .iter()
            .map(|name| StabilizerRowKind::Measurement { name: name.clone() })
            .collect::<Vec<_>>();
        return search_complete_fixing_prefix(
            context,
            &specs,
            &names,
            &[],
            basis,
            kinds,
            progress,
            budget,
        );
    }

    let space = &context.spaces[index];
    for solution in &space.solutions {
        let mut result = None;
        let visited = visit_affine_coefficients(
            &solution.particular_coeff,
            &solution.kernel_coeffs,
            budget,
            "complete affine search states",
            |combination, budget| {
                if coeff_span.reduce(combination).is_zero() {
                    return false;
                }
                let Some(raw_row) = admissible_measurement_row(context, space, combination) else {
                    return false;
                };
                if let Err(error) = budget.check_prefix_storage(
                    index + 1,
                    context.basis_rows.len(),
                    context.zx.total_ids(),
                ) {
                    progress.failure = Some(error);
                    return true;
                }

                let mut next_coeff_span = coeff_span.clone();
                if !next_coeff_span.insert_if_independent(combination.clone()) {
                    return false;
                }
                let mut next_row_span = row_span.clone();
                if !next_row_span.insert_if_independent(raw_row.clone()) {
                    return false;
                }

                surfaces.push(MeasurementSurfaceRow {
                    name: space.spec.name.clone(),
                    deadline: accepted_surface_deadline(
                        context.zx,
                        context.selective_constraints,
                        &space.spec,
                        &raw_row,
                    ),
                    row: raw_row,
                    combination: combination.clone(),
                });
                result = search_complete_measurement_prefix(
                    context,
                    index + 1,
                    surfaces,
                    &next_coeff_span,
                    &next_row_span,
                    progress,
                    budget,
                );
                surfaces.pop();
                result.is_some() || search_was_interrupted(progress)
            },
        );
        if let Err(error) = visited {
            progress.failure = Some(error);
            return None;
        }
        if result.is_some() {
            return result;
        }
        if search_was_interrupted(progress) {
            return None;
        }
    }
    None
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive search carries its explicit state and limits"
)]
fn search_complete_fixing_prefix(
    context: &CompleteSearchContext<'_>,
    measurement_specs: &[MeasurementSpec],
    measurement_names: &[String],
    fixed_cols: &[usize],
    final_basis: TrackedBasis,
    row_kinds: Vec<StabilizerRowKind>,
    progress: &mut CompleteSearchProgress,
    budget: &mut SearchBudget,
) -> Option<StabilizerGenerators> {
    if search_was_interrupted(progress) {
        return None;
    }
    if let Err(error) = budget.visit("complete fixing prefix states") {
        progress.failure = Some(error);
        return None;
    }
    let Some(constraint) = context
        .selective_constraints
        .iter()
        .find(|constraint| !fixed_cols.contains(&constraint.col))
        .copied()
    else {
        let mut final_basis = final_basis;
        diagonalize_selective_fixing_rows(
            &mut final_basis,
            &row_kinds[measurement_names.len()..],
            context.selective_constraints,
        );
        let table = match finish_canonical_from_tagged_prefix(
            CanonicalizationInputs {
                zx: context.zx,
                basis_rows: context.basis_rows,
                external_basis: context.external_basis,
                selective_constraints: context.selective_constraints,
                measurement_metadata: context.measurement_metadata,
                measurement_cols: context.measurement_cols,
            },
            measurement_specs.to_vec(),
            measurement_names,
            final_basis,
            row_kinds,
            false,
            budget,
        ) {
            Ok(table) => table,
            Err(error) => {
                progress.failure = Some(error);
                return None;
            }
        };
        let generators = match table.into_stabilizers(context.zx) {
            Ok(generators) => generators,
            Err(error) => {
                progress.failure = Some(error);
                return None;
            }
        };
        let decoupled = generators.validate_selective_decoupling().is_ok();
        if progress
            .witness
            .as_ref()
            .is_none_or(|witness| decoupled && witness.validate_selective_decoupling().is_err())
        {
            progress.witness = Some(generators.clone());
        }
        return match stabilizers_satisfy_global_constraints(context.zx, &generators) {
            Ok(satisfied) => satisfied.then_some(generators),
            Err(error) => {
                progress.failure = Some(error);
                None
            }
        };
    };

    if progress.failure.is_none() {
        progress.failure = Some(StabilizerError::SelectiveSupportUnsatisfiable {
            pos: constraint.pos,
            kind: constraint.kind,
        });
    }
    let prefix_span = CoeffSpan::from_coeffs(context.basis_rows.len(), &final_basis.coeffs);
    let row_span = PauliSpan::from_rows(&final_basis.rows, context.zx.total_ids);
    let remaining = context
        .selective_constraints
        .iter()
        .filter(|other| other.col != constraint.col && !fixed_cols.contains(&other.col))
        .copied()
        .collect::<Vec<_>>();
    let mut selected = vec![false; remaining.len()];
    loop {
        if let Err(error) = budget.visit("complete fixing subset states") {
            progress.failure = Some(error);
            return None;
        }
        // Match the former powerset order: the first remaining site changes
        // fastest. Retain only the subset currently being solved.
        let targets = std::iter::once(constraint)
            .chain(
                remaining
                    .iter()
                    .zip(&selected)
                    .filter_map(|(&target, &take)| take.then_some(target)),
            )
            .collect::<Vec<_>>();
        let result = (|| {
            let mut exact_constraints = context
                .selective_constraints
                .iter()
                .map(|other| ExactPauliConstraint {
                    col: other.col,
                    pauli: if targets.contains(other) {
                        other.forbidden
                    } else {
                        Pauli::I
                    },
                })
                .collect::<Vec<_>>();
            exact_constraints.extend(context.measurement_cols.iter().map(|&col| {
                ExactPauliConstraint {
                    col,
                    pauli: Pauli::I,
                }
            }));
            if let Err(error) = budget.check_affine(
                context.basis_rows,
                context.basis_coeffs,
                exact_constraints.len().saturating_mul(2),
            ) {
                progress.failure = Some(error);
                return None;
            }
            let solution =
                solve_affine_tracked(context.basis_rows, context.basis_coeffs, &exact_constraints)?;
            let (candidate, combination) = choose_independent_affine_solution(
                &solution,
                context.basis_rows,
                context.zx.total_ids,
                &prefix_span,
            )?;
            let fixing_row = reconstruct_boundary_supported_row(context.zx, &candidate)?;
            if let Err(error) = budget.check_prefix_storage(
                final_basis.rows.len() + 1,
                context.basis_rows.len(),
                context.zx.total_ids(),
            ) {
                progress.failure = Some(error);
                return None;
            }
            let mut next_row_span = row_span.clone();
            if !next_row_span.insert_if_independent(fixing_row.clone()) {
                return None;
            }

            let fixing_targets = targets
                .iter()
                .map(|target| SelectiveFixingTarget {
                    pos: target.pos,
                    forbidden: target.forbidden,
                })
                .collect();
            let mut next_fixed_cols = fixed_cols.to_vec();
            next_fixed_cols.extend(targets.iter().map(|target| target.col));
            let mut next_basis = final_basis.clone();
            next_basis.push(fixing_row, combination);
            let mut next_kinds = row_kinds.clone();
            next_kinds.push(StabilizerRowKind::SelectiveFixing {
                targets: fixing_targets,
            });
            search_complete_fixing_prefix(
                context,
                measurement_specs,
                measurement_names,
                &next_fixed_cols,
                next_basis,
                next_kinds,
                progress,
                budget,
            )
        })();
        if result.is_some() || search_was_interrupted(progress) {
            return result;
        }
        let next = selected.iter().position(|selected| !selected)?;
        selected[..next].fill(false);
        selected[next] = true;
    }
}

fn search_was_interrupted(progress: &CompleteSearchProgress) -> bool {
    progress
        .failure
        .as_ref()
        .is_some_and(StabilizerError::is_interrupted)
}

pub(crate) fn stabilizers_satisfy_global_constraints(
    zx: &ZXGraph,
    stabilizers: &StabilizerGenerators,
) -> Result<bool, StabilizerError> {
    if stabilizers.validate_selective_decoupling().is_err() {
        return Ok(false);
    }
    let mut dag = zx.action_graph().clone();
    match dag.attach_readout_dependencies(&stabilizers.generators, Some(zx)) {
        Ok(()) => Ok(true),
        Err(crate::BlockGraphError::Stabilizer(error)) if error.is_interrupted() => Err(error),
        Err(_) => Ok(false),
    }
}

/// Detects a record whose reachable one-target feedback anticommutes with every
/// record-carrying row, proving the cycle without exhaustive enumeration.
pub(crate) fn has_forced_direct_action_cycle(
    zx: &ZXGraph,
    basis_rows: &[PauliString],
    budget: &mut SearchBudget,
) -> Result<bool, StabilizerError> {
    budget.visit("direct action-cycle diagnostic states")?;
    budget.check_matrix(
        basis_rows.len().saturating_mul(2),
        zx.total_ids(),
        basis_rows.len(),
        basis_rows.len(),
    )?;
    let dag = zx.action_graph();
    let metadata = MeasurementMetadata::collect(zx);
    let tracked = TrackedBasis::new(basis_rows.to_vec());
    let action_count = dag.ordered_nodes().count();
    let mut classical = vec![Vec::new(); action_count];
    for (from, to, kind) in dag.dependencies() {
        if matches!(kind, crate::ActionDependency::Classical) {
            classical[from].push(to);
        }
    }

    for measurement in dag.ordered_nodes() {
        let crate::Action::Measure { name, .. } = &measurement.action else {
            continue;
        };
        budget.visit("direct action-cycle diagnostic states")?;
        let mut reachable = vec![false; action_count];
        let mut pending = vec![measurement.ordinal];
        while let Some(from) = pending.pop() {
            for &to in &classical[from] {
                if !reachable[to] {
                    reachable[to] = true;
                    pending.push(to);
                }
            }
        }

        for feedback in dag.ordered_nodes().filter(|node| reachable[node.ordinal]) {
            let crate::Action::Feedback { targets, .. } = &feedback.action else {
                continue;
            };
            let [target] = targets.as_slice() else {
                continue;
            };
            let Some((feedback_col, feedback_pauli)) = zx.feedback_column(target) else {
                continue;
            };
            let target_col = metadata.column(name);
            let mut has_surface = false;
            let mut can_commute = false;
            for pauli in metadata.matching_paulis(name) {
                for target_constraints in materialized_pauli_constraint_cases(zx, target_col, pauli)
                {
                    budget.visit("direct action-cycle affine states")?;
                    budget.check_affine(
                        &tracked.rows,
                        &tracked.coeffs,
                        target_constraints.len(),
                    )?;
                    if solve_affine_tracked_components(
                        &tracked.rows,
                        &tracked.coeffs,
                        &target_constraints,
                    )
                    .is_none()
                    {
                        continue;
                    }
                    has_surface = true;
                    for at_feedback in [Pauli::I, feedback_pauli] {
                        for feedback_constraints in
                            materialized_pauli_constraint_cases(zx, feedback_col, at_feedback)
                        {
                            budget.visit("direct action-cycle affine states")?;
                            let mut constraints = target_constraints.clone();
                            constraints.extend(feedback_constraints);
                            budget.check_affine(
                                &tracked.rows,
                                &tracked.coeffs,
                                constraints.len(),
                            )?;
                            if solve_affine_tracked_components(
                                &tracked.rows,
                                &tracked.coeffs,
                                &constraints,
                            )
                            .is_some()
                            {
                                can_commute = true;
                                break;
                            }
                        }
                        if can_commute {
                            break;
                        }
                    }
                    if can_commute {
                        break;
                    }
                }
                if can_commute {
                    break;
                }
            }
            if has_surface && !can_commute {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn measurement_prefix_basis(measurement_rows: &[MeasurementSurfaceRow]) -> TrackedBasis {
    let rows = measurement_rows
        .iter()
        .map(|surface| surface.row.clone())
        .collect::<Vec<_>>();
    let coeffs = measurement_rows
        .iter()
        .map(|surface| surface.combination.clone())
        .collect::<Vec<_>>();
    TrackedBasis::from_parts(rows, coeffs)
}

/// Consumes the coefficient-tracked `tracked` basis and returns a plain
/// [`StabilizerBasis`], repairing to `target_rank` when the tracked rows fell
/// short. Taking the `TrackedBasis` by value is the type-state boundary: the
/// coefficients it carried are unrecoverable once repair has run.
fn ensure_full_rank(
    tracked: TrackedBasis,
    mut row_kinds: Vec<StabilizerRowKind>,
    prefix_len: usize,
    target_basis_rows: &[PauliString],
    width: usize,
    target_rank: usize,
) -> Result<StabilizerBasis, StabilizerError> {
    let rows = tracked.rows;
    if row_rank(&rows, width) == target_rank && rows.len() == target_rank {
        row_kinds.truncate(prefix_len.min(rows.len()));
        row_kinds.resize(rows.len(), StabilizerRowKind::Logical);
        return Ok(StabilizerBasis::new(rows, row_kinds));
    }

    debug_assert!(prefix_len <= target_rank);
    debug_assert_eq!(row_rank(&rows[..prefix_len], width), prefix_len);
    debug_assert!(prefix_len <= row_kinds.len());

    let mut repaired_rows = rows[..prefix_len].to_vec();
    let mut repaired_kinds = row_kinds[..prefix_len].to_vec();
    let mut span = PauliSpan::from_rows(&repaired_rows, width);

    for row in rows[prefix_len..].iter().chain(target_basis_rows) {
        if !span.insert_if_independent(row.clone()) {
            continue;
        }
        repaired_rows.push(row.clone());
        repaired_kinds.push(StabilizerRowKind::Logical);
        if repaired_rows.len() == target_rank {
            break;
        }
    }

    let actual = row_rank(&repaired_rows, width);
    if actual != target_rank || repaired_rows.len() != target_rank {
        return Err(StabilizerError::BasisRankDeficient {
            expected: target_rank,
            actual,
        });
    }
    Ok(StabilizerBasis::new(repaired_rows, repaired_kinds))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::build_t_selective_graph;
    use bloq_utils::{Pauli, PauliString};
    use glam::ivec3;

    use super::super::super::{NodeKind, ZXGraph};
    use super::super::basis::{TrackedBasis, row_rank};
    use super::super::selective::{collect_selective_constraints, normalize_selective_support};
    use super::super::{StabilizerError, StabilizerGenerators, StabilizerRowKind};
    use super::{ensure_full_rank, materialized_pauli_constraint_cases};
    use crate::{BlockGraph, GalleryItem};

    #[test]
    fn complete_fixing_search_bounds_a_wide_subset_family_before_materializing_it() {
        use super::{SearchBudget, complete_stabilizer_search};
        use crate::{Block, BlockKind, SelectiveKind};

        let mut graph = BlockGraph::new();
        for index in 0..64 {
            graph.add_block(Block::new(
                ivec3(index, 0, 0),
                BlockKind::Selective(SelectiveKind::XZ),
            ));
        }
        let zx = ZXGraph::from_block_graph_parts(&graph, crate::ActionDag::default()).unwrap();
        let search = complete_stabilizer_search(&zx, &[], None, &mut SearchBudget::new(4)).unwrap();
        assert!(matches!(
            search.failure,
            Some(StabilizerError::ResourceLimited {
                phase: "complete fixing subset states",
                observed: 5,
                limit: 4,
            })
        ));
    }

    #[test]
    fn temporal_search_exhaustion_is_not_discarded_as_a_missing_candidate() {
        use super::{
            CanonicalizationInputs, MeasurementMetadata, SearchBudget, measurement_specs,
            try_temporal_measurement_table,
        };

        let graph = GalleryItem::TComparison
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::from_block_graph_for_analysis(&graph).unwrap();
        let (raw, signed) = zx.to_external_generator_table_with_signed();
        let metadata = MeasurementMetadata::collect(&zx);
        let constraints = collect_selective_constraints(&zx);
        assert!(matches!(
            try_temporal_measurement_table(
                CanonicalizationInputs {
                    zx: &zx,
                    basis_rows: &raw,
                    external_basis: Some(&signed),
                    selective_constraints: &constraints,
                    measurement_metadata: &metadata,
                    measurement_cols: &metadata.sorted_columns(),
                },
                &measurement_specs(&zx),
                &mut SearchBudget::new(0),
            ),
            Err(StabilizerError::ResourceLimited {
                phase: "temporal measurement search states",
                observed: 1,
                limit: 0
            })
        ));
    }

    fn compute_stabilizers(zx: &ZXGraph) -> Result<StabilizerGenerators, StabilizerError> {
        zx.to_stabilizer_table()?.into_stabilizers(zx)
    }

    fn assert_same_pauli_span(actual: &[PauliString], expected: &[PauliString], width: usize) {
        let actual_rank = row_rank(actual, width);
        let expected_rank = row_rank(expected, width);
        let combined = actual.iter().chain(expected).cloned().collect::<Vec<_>>();

        assert_eq!(actual_rank, expected_rank);
        assert_eq!(row_rank(&combined, width), expected_rank);
    }

    fn row_has_external_edge_pair_support(zx: &ZXGraph, row: &PauliString) -> bool {
        zx.edges
            .iter()
            .filter(|edge| edge.n1 < edge.n2)
            .all(|edge| {
                let (left, right) = zx.edge_column_pair(edge);
                row.get(left) == row.get(right)
            })
    }

    fn row_has_cross_center_reconstruction(zx: &ZXGraph, row: &PauliString) -> bool {
        zx.nodes.iter().all(|node| {
            let cross_pauli = match node.kind {
                NodeKind::X => Pauli::X,
                NodeKind::Z => Pauli::Z,
                _ => return true,
            };
            let has_incident_cross_support = zx
                .neighbor_edges(node.id)
                .any(|(_, edge_id)| row.get(edge_id) & cross_pauli);
            !has_incident_cross_support || (row.get(node.id) & cross_pauli)
        })
    }

    #[test]
    fn complete_constraints_use_the_hadamard_endpoint_frame() {
        let graph = GalleryItem::CZSpatialH
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let edge = zx
            .edges
            .iter()
            .find(|edge| edge.n1 < edge.n2 && edge.hadamard)
            .unwrap();
        let node_id = edge.n1.max(edge.n2);
        let neighbor = edge.n1.min(edge.n2);
        let cross = match zx.nodes[node_id].kind {
            NodeKind::X => Pauli::X,
            NodeKind::Z => Pauli::Z,
            kind => panic!("expected a spider, got {kind:?}"),
        };
        let directed_edge = zx.get_edge(node_id, neighbor);
        let mut raw = PauliString::new(zx.total_ids);
        raw.set(directed_edge.id, cross.flip());

        let mut materialized = raw.clone();
        zx.reconstruct_cross_center(std::slice::from_mut(&mut materialized));
        assert_eq!(materialized.get(node_id), cross);
        assert!(
            materialized_pauli_constraint_cases(&zx, node_id, cross)
                .iter()
                .any(|case| case.iter().all(|constraint| {
                    (raw.get(constraint.col) & constraint.basis) == constraint.present
                }))
        );
    }

    #[test]
    fn gallery_ccz_factory_preserves_prefix_rank_crossings_and_cache_state() {
        let graph = GalleryItem::CCZFactoryWithTels
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let cold = zx.to_stabilizer_table().unwrap();
        let cold_basis = cold.basis.clone();
        let stabilizers = cold.into_stabilizers(&zx).unwrap();
        let raw_external = zx.to_external_generator_table();
        assert!(zx.cross_incident.get().is_some());
        let warm = zx.to_stabilizer_table().unwrap();
        assert_eq!(warm.basis, cold_basis);
        assert_eq!(
            warm.into_stabilizers(&zx).unwrap().generators,
            stabilizers.generators
        );
        let rows = stabilizers
            .generators
            .iter()
            .map(|generator| generator.stabilizer.paulis.clone())
            .chain(stabilizers.auxiliary_rows().iter().cloned())
            .collect::<Vec<_>>();
        let measurement_names = stabilizers
            .generators
            .iter()
            .take_while(|generator| generator.is_measurement())
            .filter_map(|generator| generator.measurement_name().map(str::to_owned))
            .collect::<Vec<_>>();

        assert_eq!(
            measurement_names,
            vec![
                "mz0126".to_string(),
                "mz0145".to_string(),
                "mz2367".to_string(),
                "mz1234".to_string(),
                "mz1257".to_string(),
                "mx1357".to_string(),
            ]
        );
        assert_eq!(stabilizers.generators.len(), 16);
        assert_eq!(
            row_rank(&rows, zx.total_ids),
            row_rank(&raw_external, zx.total_ids)
        );
        assert!(rows.iter().all(|row| row.weight() > 0));
        let nonmeasurement_rows = stabilizers
            .generators
            .iter()
            .filter(|generator| !generator.is_measurement())
            .map(|generator| generator.stabilizer.paulis.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            row_rank(&nonmeasurement_rows, zx.total_ids),
            nonmeasurement_rows.len()
        );
        assert!(stabilizers.generators.iter().all(|generator| {
            row_has_external_edge_pair_support(&zx, &generator.stabilizer.paulis)
                && row_has_cross_center_reconstruction(&zx, &generator.stabilizer.paulis)
        }));
    }

    #[test]
    fn stabilizers_include_selective_fixing_rows_for_representative_graphs() {
        for graph in [
            build_t_selective_graph(),
            GalleryItem::CCZFactoryWithTels
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
        ] {
            let zx = ZXGraph::try_from(&graph).unwrap();
            let constraints = collect_selective_constraints(&zx);
            let measurement_cols = zx.measurement_columns().into_values().collect::<Vec<_>>();
            let stabilizers = compute_stabilizers(&zx).unwrap();
            let tagged_fixing_rows = stabilizers
                .generators
                .iter()
                .filter(|generator| generator.kind.is_selective_fixing())
                .collect::<Vec<_>>();
            assert_eq!(tagged_fixing_rows.len(), constraints.len());

            for constraint in &constraints {
                let matching_rows = tagged_fixing_rows
                    .iter()
                    .filter(|generator| {
                        generator.stabilizer.paulis.get(constraint.col) == constraint.forbidden
                            && measurement_cols
                                .iter()
                                .all(|&col| generator.stabilizer.paulis.get(col) == Pauli::I)
                            && constraints
                                .iter()
                                .filter(|other| other.col != constraint.col)
                                .all(|other| generator.stabilizer.paulis.get(other.col) == Pauli::I)
                    })
                    .count();

                assert_eq!(matching_rows, 1);
                let tagged_matches = tagged_fixing_rows
                    .iter()
                    .filter(|generator| {
                        generator
                            .kind
                            .selective_fixing_targets()
                            .iter()
                            .any(|target| {
                                target.pos == constraint.pos
                                    && target.forbidden == constraint.forbidden
                            })
                    })
                    .count();
                assert_eq!(tagged_matches, 1);
            }
        }
    }

    #[test]
    fn readout_rows_carry_no_forbidden_selective_support() {
        for graph in [
            build_t_selective_graph(),
            GalleryItem::T
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            GalleryItem::CCZFactoryWithTels
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
            GalleryItem::ToffoliFromAndDelayedCZ
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
        ] {
            let zx = ZXGraph::try_from(&graph).unwrap();
            let constraints = collect_selective_constraints(&zx);
            let stabilizers = compute_stabilizers(&zx).unwrap();

            for generator in &stabilizers.generators {
                if !generator.kind.is_readout() {
                    continue;
                }
                for constraint in &constraints {
                    assert_ne!(
                        generator.stabilizer.paulis.get(constraint.col),
                        constraint.forbidden,
                        "readout row keeps forbidden support on selective {:?}",
                        constraint.pos
                    );
                }
            }
        }
    }

    #[test]
    fn toffoli_readout_rows_are_jointly_permitted() {
        let graph = GalleryItem::ToffoliFromAndDelayedCZ
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let stabilizers = compute_stabilizers(&zx).unwrap();
        let joint_fixing = stabilizers
            .generators
            .iter()
            .find(|generator| generator.kind.selective_fixing_targets().len() > 1)
            .expect("toffoli has one joint fixing row");
        let joint_cols = joint_fixing
            .kind
            .selective_fixing_targets()
            .iter()
            .map(|target| zx.node_at(target.pos).unwrap().id)
            .collect::<Vec<_>>();
        let support = |row: &PauliString| {
            joint_cols
                .iter()
                .map(|&column| row.get(column))
                .collect::<Vec<_>>()
        };
        let permitted = |paulis: &[Pauli]| {
            [Pauli::I, Pauli::X, Pauli::Z]
                .into_iter()
                .any(|pauli| paulis.iter().all(|&value| value == pauli))
        };

        let mut crossing_rows = 0;
        for generator in stabilizers
            .generators
            .iter()
            .filter(|generator| generator.kind.is_readout())
        {
            let paulis = support(&generator.stabilizer.paulis);
            assert!(permitted(&paulis), "unreachable joint support: {paulis:?}");
            if paulis.iter().all(|&pauli| pauli == Pauli::I) {
                continue;
            }
            crossing_rows += 1;

            let other_branch = generator
                .stabilizer
                .phase_free_product(&joint_fixing.stabilizer);
            assert!(
                permitted(&support(&other_branch.paulis)),
                "joint fix must reach the other branch"
            );
        }
        assert!(crossing_rows > 0, "toffoli keeps a readable joint surface");

        let rows = stabilizers
            .generators
            .iter()
            .map(|generator| generator.stabilizer.paulis.clone())
            .chain(stabilizers.auxiliary_rows().iter().cloned())
            .collect::<Vec<_>>();
        assert_eq!(
            row_rank(&rows, zx.total_ids),
            zx.to_external_generator_table().len()
        );
    }

    #[test]
    fn toffoli_readout_boundary_rows_are_canonical() {
        let graph = GalleryItem::ToffoliFromAndDelayedCZ
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let stabilizers = compute_stabilizers(&zx).unwrap();

        assert_eq!(stabilizers.basis_rank(), 15);
        assert_eq!(stabilizers.auxiliary_rows().len(), 2);

        let x_input = ivec3(4, 0, 0);
        let logical_xz_pivots = stabilizers
            .generators
            .iter()
            .filter(|generator| matches!(generator.kind, StabilizerRowKind::Logical))
            .filter(|generator| {
                generator.stabilizer.port_stabilizer.get(&x_input).copied() == Some(Pauli::Z)
            })
            .count();
        assert_eq!(
            logical_xz_pivots, 1,
            "the x Z pass-through must be one boundary pivot"
        );

        let joint_fixing = stabilizers
            .generators
            .iter()
            .find(|generator| generator.kind.selective_fixing_targets().len() > 1)
            .expect("toffoli has one joint fixing row");
        assert!(
            joint_fixing
                .stabilizer
                .port_stabilizer
                .keys()
                .all(|pos| pos.z == 0),
            "the joint fixing row must be reduced to input-port support"
        );
    }

    #[test]
    fn gallery_thth_computes_named_measurement_stabilizers() {
        let graph = GalleryItem::THTH
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let stabilizers = compute_stabilizers(&zx).unwrap();
        let outcomes = stabilizers
            .generators
            .iter()
            .filter_map(|generator| generator.measurement_name().map(str::to_owned))
            .collect::<Vec<_>>();

        assert_eq!(outcomes, vec!["mzz1".to_string(), "mzz2".to_string()]);
    }

    #[test]
    fn discard_only_measurement_still_produces_named_stabilizer() {
        // Measure the top of the two-layer worldline (node 1 at z = 1): the
        // Z surface threads the whole column, so reading it out at the bottom
        // (node 0 at z = 0) would extend the surface past its measure layer
        // (non-materializable — the parity would depend on the qubit's future
        // state). Reading at the top keeps the surface measure-layer-bounded.
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: ZXZ [0,0,0]\n  1: ZXZ [0,0,1]\n  [0,0,0] -> +Z\n\n  m = measure 1\n  discard if m\n",
        )
        .unwrap();

        let zx = ZXGraph::try_from(&graph).unwrap();
        let stabilizers = compute_stabilizers(&zx).unwrap();
        assert!(
            stabilizers
                .generators
                .iter()
                .any(|generator| generator.measurement_name() == Some("m"))
        );
    }

    #[test]
    fn no_measurement_graph_returns_plain_rank_external_basis() {
        let graph = GalleryItem::XMemory
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let normalized = normalize_selective_support(
            &zx.to_external_generator_table(),
            &collect_selective_constraints(&zx),
        )
        .unwrap();
        let target_rank = row_rank(&normalized, zx.total_ids);

        let actual = compute_stabilizers(&zx)
            .unwrap()
            .generators
            .into_iter()
            .map(|generator| generator.stabilizer.paulis)
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), target_rank);
        assert_same_pauli_span(&actual, &normalized, zx.total_ids);
    }

    #[test]
    fn ensure_full_rank_preserves_tagged_prefix_and_marks_repaired_suffix_logical() {
        let prefix = PauliString::try_from("X__").unwrap();
        let rows = vec![prefix.clone(), PauliString::try_from("X__").unwrap()];
        let row_kinds = vec![
            StabilizerRowKind::Measurement {
                name: "m".to_string(),
            },
            StabilizerRowKind::Logical,
        ];
        let raw_external = vec![
            PauliString::try_from("X__").unwrap(),
            PauliString::try_from("_Z_").unwrap(),
            PauliString::try_from("__X").unwrap(),
        ];

        let basis =
            ensure_full_rank(TrackedBasis::new(rows), row_kinds, 1, &raw_external, 3, 3).unwrap();

        assert_eq!(basis.rows[0], prefix);
        assert_eq!(basis.rows.len(), 3);
        assert_eq!(row_rank(&basis.rows, 3), 3);
        assert_eq!(
            basis.kinds,
            vec![
                StabilizerRowKind::Measurement {
                    name: "m".to_string(),
                },
                StabilizerRowKind::Logical,
                StabilizerRowKind::Logical,
            ]
        );
    }

    #[test]
    fn static_gallery_graphs_still_produce_stabilizers() {
        let mut checked = 0;
        for entry in GalleryItem::iter() {
            let graph = entry
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            if graph.is_open() || !graph.is_rigid() {
                continue;
            }
            let stabilizers = graph.stabilizers();
            assert!(
                stabilizers.is_ok(),
                "gallery {:?} failed: {:?}",
                entry,
                stabilizers.err()
            );
            assert!(
                !stabilizers.unwrap().generators.is_empty(),
                "gallery {:?} produced no stabilizers",
                entry
            );
            checked += 1;
        }
        assert!(checked > 0, "no closed rigid gallery entry was checked");
    }

    #[test]
    fn measurement_search_can_cancel_selective_support_with_generator_product() {
        use crate::{Action, Expr, FeedbackTarget, MeasureTarget, PauliBasis};
        use bloq_utils::Direction;
        use glam::ivec3;

        let mut graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Y [0,0,2]\n  1: XZZ [0,0,3]\n  4: Y [0,1,2]\n  5: XZZ [0,1,3]\n  6: ZXX [0,1,5]\n  7: Port [0,1,6] <q0_out>\n  8: T [0,0,4]\n  9: ZXX [0,0,5]\n  10: XY [0,0,6]\n  11: ZXZ [0,1,4]\n  [0,0,2] -> +Z\n  [0,0,3] -> +Y\n  [0,1,2] -> +Z\n  [0,1,3] -H> +Z\n  [0,1,5] -> +Z\n  [0,0,4] -> +Z\n  [0,0,5] -> +Y\n  [0,0,6] -> -Z\n  [0,1,4] -> +Z\n",
        )
        .unwrap();
        graph
            .set_actions(vec![
                Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::Z,
                        target: ivec3(0, 1, 4),
                        direction: None,
                    }],
                    condition: None,
                },
                Action::Measure {
                    target: MeasureTarget::Edge {
                        src: ivec3(0, 1, 5),
                        dir: Direction::YMINUS,
                    },
                    name: "t_2_mzz".into(),
                },
                Action::Resolve {
                    target: ivec3(0, 0, 6),
                    condition: Expr::Not(Box::new(Expr::Var("t_2_mzz".into()))),
                },
            ])
            .unwrap();

        let surface = graph
            .action_graph()
            .node_by_ordinal(1)
            .unwrap()
            .measurement_stabilizer
            .as_ref()
            .unwrap();
        assert!(!surface.interior_nodes.contains_key(&ivec3(0, 0, 6)));
        assert_eq!(
            surface.port_stabilizer.get(&ivec3(0, 1, 6)),
            Some(&Pauli::Z)
        );
    }
}
#[cfg(test)]
mod merge_records {
    use bloq_utils::Pauli;
    use glam::ivec3;

    use crate::BlockGraph;

    /// A `Y` merge-pipe surface carries X transport and the named Z parity.
    #[test]
    fn merge_record_is_carried_by_y_on_the_pipe() {
        let graph = BlockGraph::from_blog_text(
            "BLOG 1.0

  0: Y [0,0,0]
  1: XZX [0,0,1]
  2: Y [1,0,0]
  3: XZX [1,0,1]
  [0,0,0] -> +Z
  [1,0,0] -> +Z
  [0,0,1] -> +X

  m = measure 1 -> +X
",
        )
        .unwrap();
        graph.validate().unwrap();
        let (graph, stabilizers) = graph.analyze_actions().unwrap();

        let record = stabilizers
            .generators
            .iter()
            .find(|generator| generator.measurement_name() == Some("m"))
            .expect("the merge record has a surface");
        assert!(record.stabilizer.sign, "the closed parity has constant 1");
        assert_eq!(record.stabilizer.interior_nodes.len(), 4);
        assert!(
            record
                .stabilizer
                .interior_nodes
                .values()
                .all(|&pauli| pauli == Pauli::Y)
        );
        assert_eq!(record.stabilizer.interior_edges.len(), 3);
        assert!(
            record
                .stabilizer
                .interior_edges
                .values()
                .all(|&pauli| pauli == Pauli::Y)
        );
        assert_eq!(
            record.stabilizer.interior_edges[&(ivec3(0, 0, 1), ivec3(1, 0, 1))],
            Pauli::Y
        );
        assert!(
            graph
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement_stabilizer
                .is_some(),
            "the action DAG carries the record's surface"
        );
    }
}
#[cfg(test)]
mod self_dependent_feedback {
    use bloq_utils::{Direction, Pauli, PauliBasis};
    use glam::ivec3;

    use crate::{Action, BlockGraph, Expr, FeedbackTarget, GalleryItem, MeasureTarget};

    /// Prefer a commuting component witness over a cyclic exact witness.
    #[test]
    fn self_gated_feedback_sends_the_search_to_a_commuting_witness() {
        let mut graph = BlockGraph::from_blog_text(
            "BLOG 1.0\n\n  0: Y [0,0,2]\n  1: XZZ [0,0,3]\n  4: Y [0,1,2]\n  5: XZZ [0,1,3]\n  6: ZXX [0,1,5]\n  7: Port [0,1,6] <q0_out>\n  8: T [0,0,4]\n  9: ZXX [0,0,5]\n  10: XY [0,0,6]\n  11: ZXZ [0,1,4]\n  12: ZXZ [2,0,2]\n  13: ZXZ [2,0,3]\n  [0,0,2] -> +Z\n  [0,0,3] -> +Y\n  [0,1,2] -> +Z\n  [0,1,3] -H> +Z\n  [0,1,5] -> +Z\n  [0,0,4] -> +Z\n  [0,0,5] -> +Y\n  [0,0,6] -> -Z\n  [0,1,4] -> +Z\n  [2,0,2] -> +Z\n",
        )
        .unwrap();
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Edge {
                        src: ivec3(0, 1, 5),
                        dir: Direction::YMINUS,
                    },
                    name: "m".into(),
                },
                Action::Feedback {
                    targets: vec![FeedbackTarget {
                        pauli: PauliBasis::Y,
                        target: ivec3(0, 1, 5),
                        direction: None,
                    }],
                    condition: Some(Expr::Var("m".into())),
                },
                Action::Measure {
                    target: MeasureTarget::Node(ivec3(2, 0, 3)),
                    name: "a".into(),
                },
                Action::Resolve {
                    target: ivec3(0, 0, 6),
                    condition: Expr::Var("a".into()),
                },
            ])
            .unwrap();
        graph.validate().unwrap();

        let surface = graph
            .action_graph()
            .node_by_ordinal(0)
            .unwrap()
            .measurement_stabilizer
            .as_ref()
            .unwrap();
        assert_eq!(surface.interior_nodes.get(&ivec3(0, 1, 5)), Some(&Pauli::Y));
    }

    /// Avoid a cycle reached through a selective fixing row.
    #[test]
    fn self_gated_feedback_avoids_reachable_selective_fixing_rows() {
        let mut graph = GalleryItem::THTH
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let mut actions = graph.actions();
        let feedback = actions.len();
        actions.push(Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::Z,
                target: ivec3(2, 1, 1),
                direction: None,
            }],
            condition: Some(Expr::Var("mzz2".into())),
        });
        graph.set_actions_lenient(actions).unwrap();
        let (graph, stabilizers) = graph.analyze_actions().unwrap();

        let row = stabilizers
            .generators
            .iter()
            .find(|generator| generator.measurement_name() == Some("mzz2"))
            .unwrap();
        assert!(!row.stabilizer.interior_nodes.contains_key(&ivec3(2, 1, 2)));
        assert!(
            graph
                .action_graph()
                .dependencies()
                .all(|(from, to, _)| (from, to) != (feedback, 1))
        );
    }

    /// Backtrack when one record's witness cycles through an earlier record.
    #[test]
    fn witness_search_does_not_close_a_cycle_through_an_earlier_record() {
        let mut graph = GalleryItem::THTH
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let mut actions = graph.actions();
        // The gallery now corrects its Y resources at initialization. Retain
        // this internal conditional action to exercise the historical cycle.
        let earlier_feedback = actions.len();
        actions.push(Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::Z,
                target: ivec3(-1, 0, 1),
                direction: None,
            }],
            condition: Some(Expr::Var("mzz1".into())),
        });
        let feedback = actions.len();
        actions.push(Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::X,
                target: ivec3(0, 0, 0),
                direction: None,
            }],
            condition: Some(Expr::Var("mzz2".into())),
        });
        graph.set_actions_lenient(actions).unwrap();
        let (graph, _) = graph.analyze_actions().unwrap();

        let dependencies = graph.action_graph().dependencies().collect::<Vec<_>>();
        assert!(
            dependencies
                .iter()
                .any(|&(from, to, _)| (from, to) == (feedback, 0))
        );
        assert!(
            dependencies
                .iter()
                .all(|&(from, to, _)| (from, to) != (earlier_feedback, 1))
        );
    }
}
