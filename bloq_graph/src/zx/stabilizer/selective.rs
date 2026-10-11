//! Selective-site constraints and fixing-row construction.

use glam::IVec3;

use bloq_utils::{Pauli, PauliString};

use super::super::{NodeKind, ZXGraph};
use super::affine::{
    AffineTrackedSolution, ExactPauliConstraint, materialize_tracked_row, solve_affine_tracked,
};
use super::basis::{
    CoeffSpan, CoeffVec, TrackedBasis, axis_pivot_constraints, echelonize_with_witnesses,
    gaussian_elimination_with_tracking, row_rank,
};
use super::{SearchBudget, SelectiveFixingTarget, StabilizerError, StabilizerRowKind};
use crate::SelectiveKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SelectiveConstraint {
    pub(crate) pos: IVec3,
    pub(crate) kind: SelectiveKind,
    pub(crate) col: usize,
    pub(crate) forbidden: Pauli,
}

impl SelectiveConstraint {
    pub(super) fn new(pos: IVec3, kind: SelectiveKind, col: usize) -> Self {
        let forbidden = match kind {
            SelectiveKind::XY => Pauli::Z,
            SelectiveKind::XZ => Pauli::Y,
            SelectiveKind::YZ => Pauli::X,
        };
        Self {
            pos,
            kind,
            col,
            forbidden,
        }
    }
}

pub(crate) fn collect_selective_constraints(zx: &ZXGraph) -> Vec<SelectiveConstraint> {
    zx.nodes
        .iter()
        .filter_map(|node| match node.kind {
            NodeKind::Selective(kind) => Some(SelectiveConstraint::new(node.pos, kind, node.id)),
            _ => None,
        })
        .collect()
}

pub(crate) fn normalize_selective_support(
    generators: &[PauliString],
    constraints: &[SelectiveConstraint],
) -> Result<Vec<PauliString>, StabilizerError> {
    let mut rows = generators.to_vec();
    normalize_selective_support_in_place(&mut rows, None, constraints)?;
    Ok(rows)
}

/// Cancel forbidden selective support in place. When `coeffs` is `Some`, every
/// row XOR is mirrored on the parallel coefficient slice so generator provenance
/// stays in lockstep with the rows.
pub(crate) fn normalize_selective_support_in_place(
    rows: &mut [PauliString],
    mut coeffs: Option<&mut [CoeffVec]>,
    constraints: &[SelectiveConstraint],
) -> Result<(), StabilizerError> {
    for constraint in constraints {
        normalize_one_selective(rows, coeffs.as_deref_mut(), *constraint)?;
    }
    Ok(())
}

fn normalize_one_selective(
    rows: &mut [PauliString],
    coeffs: Option<&mut [CoeffVec]>,
    constraint: SelectiveConstraint,
) -> Result<(), StabilizerError> {
    let mut allowed_pivot = None;
    let mut forbidden_indices = Vec::new();

    for (index, row) in rows.iter().enumerate() {
        let pauli = row.get(constraint.col);
        if pauli == Pauli::I {
            continue;
        }
        if pauli == constraint.forbidden {
            forbidden_indices.push(index);
        } else if allowed_pivot.is_none() {
            allowed_pivot = Some(index);
        }
    }

    if forbidden_indices.is_empty() {
        return Ok(());
    }

    if let Some(pivot_index) = allowed_pivot {
        xor_pivot_into(rows, coeffs, pivot_index, &forbidden_indices);
        debug_assert!(
            rows.iter()
                .all(|row| row.get(constraint.col) != constraint.forbidden)
        );
        return Ok(());
    }

    // No allowed pivot exists to cancel the forbidden support, so the
    // constraint is unsatisfiable. The caller discards `rows` on this error
    // path, so no elimination is needed.
    Err(StabilizerError::SelectiveSupportUnsatisfiable {
        pos: constraint.pos,
        kind: constraint.kind,
    })
}

fn xor_pivot_into(
    rows: &mut [PauliString],
    mut coeffs: Option<&mut [CoeffVec]>,
    pivot_index: usize,
    targets: &[usize],
) {
    let pivot = rows[pivot_index].clone();
    let pivot_coeff = coeffs.as_deref().map(|c| c[pivot_index].clone());
    for &index in targets {
        rows[index] ^= &pivot;
        if let (Some(coeffs), Some(pivot_coeff)) = (coeffs.as_deref_mut(), &pivot_coeff) {
            coeffs[index].xor_assign(pivot_coeff);
        }
    }
}

fn row_has_boundary_node_support(zx: &ZXGraph, row: &PauliString) -> bool {
    zx.nodes
        .iter()
        .any(|node| node.kind.is_boundary() && row.get(node.id) != Pauli::I)
}

pub(super) fn reconstruct_boundary_supported_row(
    zx: &ZXGraph,
    row: &PauliString,
) -> Option<PauliString> {
    let mut candidate = row.clone();
    zx.reconstruct_cross_center(std::slice::from_mut(&mut candidate));
    if candidate.is_identity() || !row_has_boundary_node_support(zx, &candidate) {
        return None;
    }
    Some(candidate)
}
pub(super) fn add_selective_fixing_rows(
    final_basis: &mut TrackedBasis,
    basis_rows: &[PauliString],
    constraints: &[SelectiveConstraint],
    measurement_cols: &[usize],
    zx: &ZXGraph,
    budget: &mut SearchBudget,
) -> Result<Vec<StabilizerRowKind>, StabilizerError> {
    if constraints.is_empty() {
        return Ok(Vec::new());
    }
    budget.visit("selective fixing setup states")?;
    budget.check_matrix(
        basis_rows
            .len()
            .saturating_mul(5)
            .saturating_add(final_basis.rows.len()),
        zx.total_ids(),
        basis_rows
            .len()
            .saturating_mul(4)
            .saturating_add(final_basis.coeffs.len()),
        basis_rows.len(),
    )?;
    // Caller supplies the already reduced external basis so coefficient
    // provenance is one-to-one with these rows.
    debug_assert_eq!(row_rank(basis_rows, zx.total_ids), basis_rows.len());
    if constraints.len() >= 32
        && let Some(fixings) = bulk_isolated_selective_fixings(
            final_basis,
            basis_rows,
            constraints,
            measurement_cols,
            zx,
            budget,
        )?
    {
        let mut row_kinds = Vec::with_capacity(fixings.len());
        for (row, coeff, kind) in fixings {
            final_basis.push(row, coeff);
            row_kinds.push(kind);
        }
        budget.visit("selective fixing diagonalization states")?;
        diagonalize_selective_fixing_rows(final_basis, &row_kinds, constraints);
        return Ok(row_kinds);
    }

    let span = TrackedBasis::new(basis_rows.to_vec());
    let mut row_kinds = Vec::new();

    // `final_basis` only grows by the one fixing row pushed at the end of each
    // iteration, so track its coefficient span incrementally rather than
    // rebuilding it from scratch every constraint. The insertion sequence
    // (frozen prefix coeffs in order, then each fixing combination in push
    // order) matches what a fresh `from_coeffs` over `final_basis.coeffs` would
    // replay, so the span — and every quotient reduction read off it — is
    // identical.
    let mut prefix_span = CoeffSpan::from_coeffs(basis_rows.len(), &final_basis.coeffs);

    // Sites already pinned by a pushed row, which later rows must leave alone.
    let mut fixed_cols: Vec<usize> = Vec::new();

    for &constraint in constraints {
        budget.visit("selective fixing search states")?;
        if fixed_cols.contains(&constraint.col) {
            continue;
        }
        let unsatisfiable = || StabilizerError::SelectiveSupportUnsatisfiable {
            pos: constraint.pos,
            kind: constraint.kind,
        };

        // Two tiers. First look for a row that fixes this site *alone*; that is
        // what keeps fixing rows pairwise diagonalized, which the symbolic
        // basis needs to model fills affinely. When no such row exists the
        // site's forbidden support is entangled with another site's — sites
        // steered by one measurement (the toffoli's two `!mxy` caps) are
        // inherently joint — so fall back to a row that may carry forbidden
        // support at any not-yet-fixed site, and tag it with all of them. One
        // joint row still normalizes every entangled site: a row carrying the
        // forbidden Pauli at those sites cancels against it in one XOR.
        let foreign_free_cols = constraints
            .iter()
            .map(|other| other.col)
            .filter(|col| *col != constraint.col && !fixed_cols.contains(col))
            .collect::<Vec<_>>();
        let mut accepted = None;
        for free_cols in [&[][..], &foreign_free_cols] {
            budget.visit("selective fixing affine states")?;
            let exact_constraints = exact_constraints_for_selective_fixing(
                constraint,
                constraints,
                measurement_cols,
                free_cols,
            );
            budget.check_affine(
                &span.rows,
                &span.coeffs,
                exact_constraints.len().saturating_mul(2),
            )?;
            let Some(solution) = solve_affine_tracked(&span.rows, &span.coeffs, &exact_constraints)
            else {
                continue;
            };
            let Some((candidate_row, fixing_combination)) = choose_independent_affine_solution(
                &solution,
                basis_rows,
                zx.total_ids,
                &prefix_span,
            ) else {
                continue;
            };
            if candidate_row.get(constraint.col) != constraint.forbidden {
                continue;
            }
            let Some(fixing_row) = reconstruct_boundary_supported_row(zx, &candidate_row) else {
                continue;
            };
            accepted = Some((candidate_row, fixing_row, fixing_combination));
            break;
        }
        let (candidate_row, fixing_row, fixing_combination) = accepted.ok_or_else(unsatisfiable)?;

        // Read the targets off the *solved* row rather than the reconstructed
        // one: the exact constraints were imposed on it, and cross-center
        // reconstruction only adds interior center support.
        let targets = constraints
            .iter()
            .filter(|other| candidate_row.get(other.col) == other.forbidden)
            .map(|other| SelectiveFixingTarget {
                pos: other.pos,
                forbidden: other.forbidden,
            })
            .collect::<Vec<_>>();
        fixed_cols.extend(
            constraints
                .iter()
                .filter(|other| candidate_row.get(other.col) == other.forbidden)
                .map(|other| other.col),
        );

        final_basis.push(fixing_row, fixing_combination);
        prefix_span.insert_if_independent(
            final_basis
                .coeffs
                .last()
                .expect("a fixing row was just pushed")
                .clone(),
        );
        row_kinds.push(StabilizerRowKind::SelectiveFixing { targets });
    }
    budget.visit("selective fixing diagonalization states")?;
    diagonalize_selective_fixing_rows(final_basis, &row_kinds, constraints);

    Ok(row_kinds)
}

/// Finds independent fixing rows with two global diagonalizations: first all
/// measurement axes, then all selective axes. The strict isolation checks make
/// this an all-or-nothing shortcut; entangled sites use the affine fallback.
#[expect(
    clippy::type_complexity,
    reason = "the tuple preserves a row, its source coordinates, and its role"
)]
fn bulk_isolated_selective_fixings(
    prefix: &TrackedBasis,
    basis_rows: &[PauliString],
    constraints: &[SelectiveConstraint],
    measurement_cols: &[usize],
    zx: &ZXGraph,
    budget: &mut SearchBudget,
) -> Result<Option<Vec<(PauliString, CoeffVec, StabilizerRowKind)>>, StabilizerError> {
    budget.visit("bulk selective fixing states")?;
    budget.check_matrix(
        basis_rows
            .len()
            .saturating_mul(5)
            .saturating_add(prefix.rows.len()),
        zx.total_ids(),
        basis_rows
            .len()
            .saturating_mul(4)
            .saturating_add(prefix.coeffs.len()),
        basis_rows.len(),
    )?;
    let mut basis = TrackedBasis::new(basis_rows.to_vec());
    let measurement_pivots = gaussian_elimination_with_tracking(
        &mut basis.rows,
        &mut basis.coeffs,
        axis_pivot_constraints(measurement_cols.iter().copied()),
        |row, &(col, axis)| row.get(col) & axis,
        None,
    );
    budget.visit("bulk selective fixing states")?;
    let mut rows = basis.rows[measurement_pivots..].to_vec();
    let mut coeffs = basis.coeffs[measurement_pivots..].to_vec();
    let selective_pivots = gaussian_elimination_with_tracking(
        &mut rows,
        &mut coeffs,
        axis_pivot_constraints(constraints.iter().map(|constraint| constraint.col)),
        |row, &(col, axis)| row.get(col) & axis,
        None,
    );
    let mut used = vec![false; selective_pivots];
    let mut prefix_span = CoeffSpan::from_coeffs(basis_rows.len(), &prefix.coeffs);
    let mut selected = Vec::with_capacity(constraints.len());

    for constraint in constraints {
        budget.visit("bulk selective fixing candidate states")?;
        let target_rows = (0..selective_pivots)
            .filter(|&index| !used[index] && rows[index].get(constraint.col) != Pauli::I)
            .collect::<Vec<_>>();
        let pair = if let [first, second] = target_rows.as_slice() {
            let mut row = rows[*first].clone();
            row ^= &rows[*second];
            let mut coeff = coeffs[*first].clone();
            coeff.xor_assign(&coeffs[*second]);
            Some((vec![*first, *second], row, coeff))
        } else {
            None
        };
        let mut candidates = target_rows
            .iter()
            .map(|&index| (vec![index], rows[index].clone(), coeffs[index].clone()))
            .chain(pair);

        let Some((indices, row, coeff)) = candidates.find(|(_, row, coeff)| {
            row.get(constraint.col) == constraint.forbidden
                && constraints
                    .iter()
                    .all(|other| other.col == constraint.col || row.get(other.col) == Pauli::I)
                && measurement_cols.iter().all(|&col| row.get(col) == Pauli::I)
                && !prefix_span.reduce(coeff).is_zero()
        }) else {
            return Ok(None);
        };
        for index in indices {
            used[index] = true;
        }
        prefix_span.insert_if_independent(coeff.clone());
        selected.push((row, coeff, constraint));
    }

    budget.visit("bulk selective fixing reconstruction states")?;
    let mut reconstructed = selected
        .iter()
        .map(|(row, _, _)| row.clone())
        .collect::<Vec<_>>();
    zx.reconstruct_cross_center(&mut reconstructed);
    Ok(reconstructed
        .into_iter()
        .zip(selected)
        .map(|(row, (_, coeff, constraint))| {
            (!row.is_identity() && row_has_boundary_node_support(zx, &row)).then(|| {
                (
                    row,
                    coeff,
                    StabilizerRowKind::SelectiveFixing {
                        targets: vec![SelectiveFixingTarget {
                            pos: constraint.pos,
                            forbidden: constraint.forbidden,
                        }],
                    },
                )
            })
        })
        .collect())
}

/// The exact-Pauli system a fixing row for `constraint` must satisfy: the
/// forbidden Pauli on its own column, identity on every measurement column
/// (fixing rows carry no measured parity), and identity on every foreign
/// selective column except those in `free_cols`, which are left unconstrained
/// so a joint row can fix them too.
pub(super) fn exact_constraints_for_selective_fixing(
    constraint: SelectiveConstraint,
    constraints: &[SelectiveConstraint],
    measurement_cols: &[usize],
    free_cols: &[usize],
) -> Vec<ExactPauliConstraint> {
    let mut exact_constraints = constraints
        .iter()
        .filter(|other| !free_cols.contains(&other.col))
        .map(|other| ExactPauliConstraint {
            col: other.col,
            pauli: if other.pos == constraint.pos {
                constraint.forbidden
            } else {
                Pauli::I
            },
        })
        .collect::<Vec<_>>();
    exact_constraints.extend(measurement_cols.iter().filter_map(|&col| {
        (col != constraint.col).then_some(ExactPauliConstraint {
            col,
            pauli: Pauli::I,
        })
    }));
    exact_constraints
}

pub(super) fn diagonalize_selective_fixing_rows(
    final_basis: &mut TrackedBasis,
    row_kinds: &[StabilizerRowKind],
    constraints: &[SelectiveConstraint],
) {
    let offset = final_basis.rows.len().saturating_sub(row_kinds.len());
    debug_assert!(offset <= final_basis.rows.len());
    let mut changed = true;
    while changed {
        changed = false;
        for row_kind_index in 0..row_kinds.len() {
            let row_index = offset + row_kind_index;
            let current_score = off_target_selective_support_score(
                &final_basis.rows[row_index],
                &row_kinds[row_kind_index],
                constraints,
            );
            if current_score == 0 {
                continue;
            }

            for pivot_kind_index in 0..row_kinds.len() {
                if pivot_kind_index == row_kind_index {
                    continue;
                }

                let pivot_index = offset + pivot_kind_index;
                let mut candidate = final_basis.rows[row_index].clone();
                candidate ^= &final_basis.rows[pivot_index];
                if !preserves_selective_fixing_targets(
                    &candidate,
                    &row_kinds[row_kind_index],
                    constraints,
                ) {
                    continue;
                }

                let candidate_score = off_target_selective_support_score(
                    &candidate,
                    &row_kinds[row_kind_index],
                    constraints,
                );
                if candidate_score >= current_score {
                    continue;
                }

                let pivot_coeff = final_basis.coeffs[pivot_index].clone();
                final_basis.rows[row_index] = candidate;
                final_basis.coeffs[row_index].xor_assign(&pivot_coeff);
                changed = true;
                break;
            }
        }
    }
}

fn off_target_selective_support_score(
    row: &PauliString,
    row_kind: &StabilizerRowKind,
    constraints: &[SelectiveConstraint],
) -> usize {
    constraints
        .iter()
        .filter(|constraint| !row_kind.fixes_selective(constraint.pos))
        .map(|constraint| row.get(constraint.col).iter_xz().count())
        .sum()
}

fn preserves_selective_fixing_targets(
    row: &PauliString,
    row_kind: &StabilizerRowKind,
    constraints: &[SelectiveConstraint],
) -> bool {
    row_kind.selective_fixing_targets().iter().all(|target| {
        constraints
            .iter()
            .find(|constraint| constraint.pos == target.pos)
            .is_some_and(|constraint| row.get(constraint.col) == target.forbidden)
    })
}

pub(super) fn choose_independent_affine_solution(
    solution: &AffineTrackedSolution,
    basis_rows: &[PauliString],
    width: usize,
    prefix_span: &CoeffSpan,
) -> Option<(PauliString, CoeffVec)> {
    let particular_reduced = prefix_span.reduce(&solution.particular_coeff);
    if !particular_reduced.is_zero() {
        let row = materialize_tracked_row(&solution.particular_coeff, basis_rows, width);
        return Some((row, solution.particular_coeff.clone()));
    }

    let kernel_combo = first_echelonized_nonzero_kernel_direction(solution, prefix_span)?;
    let mut coeff = solution.particular_coeff.clone();
    for index in kernel_combo.to_indices() {
        coeff.xor_assign(&solution.kernel_coeffs[index]);
    }
    let row = materialize_tracked_row(&coeff, basis_rows, width);
    Some((row, coeff))
}

fn first_echelonized_nonzero_kernel_direction(
    solution: &AffineTrackedSolution,
    prefix_span: &CoeffSpan,
) -> Option<CoeffVec> {
    let mut reduced_rows = Vec::new();
    let mut witnesses = Vec::new();
    for (index, kernel_coeff) in solution.kernel_coeffs.iter().enumerate() {
        let reduced = prefix_span.reduce(kernel_coeff);
        if reduced.is_zero() {
            continue;
        }
        reduced_rows.push(reduced);
        witnesses.push(CoeffVec::singleton(index, solution.kernel_coeffs.len()));
    }

    let width = reduced_rows.first().map(CoeffVec::len).unwrap_or(0);
    let pivot_cols = echelonize_with_witnesses(&mut reduced_rows, &mut witnesses, width);
    (!pivot_cols.is_empty()).then(|| witnesses[0].clone())
}
pub(super) fn canonicalize_boundary_suffix(rows: &mut [PauliString], zx: &ZXGraph) {
    if rows.is_empty() {
        return;
    }

    let pivot_constraints = boundary_pivot_constraints(zx);

    gaussian_elimination_with_tracking(
        rows,
        &mut [],
        pivot_constraints,
        |row, &(col, basis)| row.get(col) & basis,
        None,
    );
}

pub(super) fn boundary_pivot_constraints(zx: &ZXGraph) -> Vec<(usize, Pauli)> {
    let mut output_cols = Vec::new();
    let mut other_cols = Vec::new();
    for node in &zx.nodes {
        if !node.kind.is_boundary() || matches!(node.kind, NodeKind::Selective(_)) {
            continue;
        }
        if node.is_output_port(zx) {
            output_cols.push(node.id);
        } else {
            other_cols.push(node.id);
        }
    }
    output_cols.sort_unstable();
    other_cols.sort_unstable();
    axis_pivot_constraints(output_cols.into_iter().chain(other_cols))
}

#[cfg(test)]
mod tests {
    use glam::ivec3;

    use super::super::basis::reduce_to_basis;
    use super::super::test_support::{build_t_selective_graph, sparse_to_coeff};
    use super::*;

    #[test]
    fn collect_selective_constraints_maps_xy_to_forbidden_z() {
        let graph = build_t_selective_graph();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let constraints = collect_selective_constraints(&zx);

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].pos, ivec3(1, 0, 2));
        assert_eq!(constraints[0].forbidden, Pauli::Z);
    }

    #[test]
    fn normalize_selective_support_covers_happy_and_unhappy_paths() {
        let constraint = SelectiveConstraint::new(ivec3(0, 0, 0), SelectiveKind::XY, 0);
        let rows = vec![
            PauliString::try_from("ZX").unwrap(),
            PauliString::try_from("XI").unwrap(),
        ];
        let normalized = normalize_selective_support(&rows, &[constraint]).unwrap();
        assert_eq!(normalized[0].get(0), Pauli::Y);
        assert_eq!(normalized[1].get(0), Pauli::X);

        let second = normalize_selective_support(&normalized, &[constraint]).unwrap();
        assert_eq!(normalized, second);

        let err =
            normalize_selective_support(&[PauliString::try_from("ZX").unwrap()], &[constraint])
                .unwrap_err();
        assert!(matches!(
            err,
            StabilizerError::SelectiveSupportUnsatisfiable { .. }
        ));
    }

    #[test]
    fn greedy_external_selective_fixings_charge_before_setup_bulk_and_affine_work() {
        use crate::{
            Action, ActionDag, Block, BlockGraph, BlockKind, Direction, Expr,
            ModuleCertificationLimits, Pipe,
        };

        for (sites, limit, phase) in [
            (1, 0, "selective fixing setup states"),
            (1, 2, "selective fixing affine states"),
            (32, 1, "bulk selective fixing states"),
        ] {
            let mut graph = BlockGraph::new();
            let mut actions = Vec::new();
            for index in 0..sites {
                let input = ivec3(index, 0, 0);
                let output = ivec3(index, 0, 1);
                graph.add_block(Block::new(input, BlockKind::Port));
                graph.add_block(Block::new(output, BlockKind::Selective(SelectiveKind::XZ)));
                graph.add_pipe(Pipe::new(input, Direction::ZPLUS));
                actions.push(Action::Resolve {
                    target: output,
                    condition: Expr::Var(format!("input{index}")),
                });
            }
            let mut zx =
                ZXGraph::from_block_graph_parts(&graph, crate::ActionDag::default()).unwrap();
            zx.action_graph = ActionDag::from_actions(&actions);
            assert!(
                matches!(zx.stabilizers_with_limits(ModuleCertificationLimits {
                max_normalization_states: limit,
                ..ModuleCertificationLimits::UNLIMITED
            }), Err(StabilizerError::ResourceLimited { phase: actual, observed, limit: actual_limit })
                if actual == phase && observed == limit + 1 && actual_limit == limit)
            );
            if sites == 1 {
                zx.stabilizers_with_limits(ModuleCertificationLimits::UNLIMITED)
                    .unwrap();
            }
        }
    }

    #[test]
    fn selective_fixing_rows_work_with_empty_measurement_prefix() {
        let graph = build_t_selective_graph();
        let zx = ZXGraph::try_from(&graph).unwrap();
        let basis_rows = reduce_to_basis(&zx.to_external_generator_table(), zx.total_ids);
        let constraints = collect_selective_constraints(&zx);
        let mut final_basis = TrackedBasis::from_parts(Vec::new(), Vec::<CoeffVec>::new());
        let bulk = bulk_isolated_selective_fixings(
            &final_basis,
            &basis_rows,
            &constraints,
            &[],
            &zx,
            &mut SearchBudget::new(usize::MAX),
        )
        .unwrap()
        .expect("one selective site is globally isolated");
        assert_eq!(bulk.len(), constraints.len());

        let row_kinds = add_selective_fixing_rows(
            &mut final_basis,
            &basis_rows,
            &constraints,
            &[],
            &zx,
            &mut SearchBudget::new(usize::MAX),
        )
        .unwrap();
        assert_eq!(row_kinds.len(), constraints.len());

        assert!(constraints.iter().all(|constraint| {
            final_basis.rows.iter().any(|row| {
                row.get(constraint.col) == constraint.forbidden
                    && zx
                        .nodes
                        .iter()
                        .any(|node| node.kind.is_boundary() && row.get(node.id) != Pauli::I)
            })
        }));
    }

    #[test]
    fn choose_independent_affine_solution_uses_first_echelonized_kernel_quotient_direction() {
        let mut prefix_span = CoeffSpan::new(4);
        assert!(prefix_span.insert_if_independent(sparse_to_coeff(&[0], 4)));

        let solution = AffineTrackedSolution {
            particular_coeff: sparse_to_coeff(&[0], 4),
            kernel_coeffs: vec![sparse_to_coeff(&[0, 2], 4), sparse_to_coeff(&[1], 4)],
        };
        let basis_rows = vec![
            PauliString::try_from("Z").unwrap(),
            PauliString::try_from("Y").unwrap(),
            PauliString::try_from("I").unwrap(),
            PauliString::try_from("I").unwrap(),
        ];

        let (row, coeff) =
            choose_independent_affine_solution(&solution, &basis_rows, 1, &prefix_span).unwrap();

        assert_eq!(row, PauliString::try_from("X").unwrap());
        assert_eq!(coeff, sparse_to_coeff(&[0, 1], 4));
    }

    #[test]
    fn diagonalize_selective_fixing_rows_removes_off_target_support_when_possible() {
        let constraints = [
            SelectiveConstraint::new(ivec3(0, 0, 0), SelectiveKind::XY, 0),
            SelectiveConstraint::new(ivec3(1, 0, 0), SelectiveKind::XY, 1),
        ];
        let mut final_basis = TrackedBasis::from_parts(
            vec![
                PauliString::try_from("ZZ").unwrap(),
                PauliString::try_from("_Z").unwrap(),
            ],
            vec![sparse_to_coeff(&[0], 2), sparse_to_coeff(&[1], 2)],
        );
        let row_kinds = vec![
            StabilizerRowKind::SelectiveFixing {
                targets: vec![SelectiveFixingTarget {
                    pos: constraints[0].pos,
                    forbidden: constraints[0].forbidden,
                }],
            },
            StabilizerRowKind::SelectiveFixing {
                targets: vec![SelectiveFixingTarget {
                    pos: constraints[1].pos,
                    forbidden: constraints[1].forbidden,
                }],
            },
        ];

        diagonalize_selective_fixing_rows(&mut final_basis, &row_kinds, &constraints);

        assert_eq!(final_basis.rows[0], PauliString::try_from("Z_").unwrap());
        assert_eq!(final_basis.rows[1], PauliString::try_from("_Z").unwrap());
    }
}
