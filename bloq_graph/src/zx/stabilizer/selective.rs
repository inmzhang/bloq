//! Selective-site support constraints.

use glam::IVec3;

use bloq_utils::{Pauli, PauliString};

use super::super::{NodeKind, ZXGraph};
use super::StabilizerError;
use super::basis::CoeffVec;
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
