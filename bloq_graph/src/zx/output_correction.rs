//! Symbolic GF(2) solve of the joint output-correction system.
//!
//! The terminal Pauli-frame correction is an affine GF(2) system: the LHS
//! (which output frame bits each correction surface constrains) is fixed by
//! the stabilizer basis, while the RHS (whether each surface's measurement
//! parity demands a flip) is only known at runtime. Running the elimination
//! once over the static LHS — tracking, per solved bit, *which constraints'*
//! RHS bits XOR into it — yields a [`SymbolicOutputCorrection`] used by both
//! internal-measurement verification and compiler `Compute` lowering. One
//! elimination implementation keeps the consumers aligned.

use std::collections::BTreeSet;

use bloq_utils::PauliString;
use glam::IVec3;
use thiserror::Error;

use super::modular::toggle;
use super::stabilizer::{CoeffVec, axis_pivot_constraints, gaussian_elimination_with_tracking};
use crate::{FxHashMap, Pauli};

/// One equation in the terminal output-correction system.
///
/// `output_support` is the equation's left-hand side. Its runtime right-hand
/// side includes the XOR of the named generators' readable values. A row with
/// no readable generators is still a real equation: physical lowering must
/// retain any private measurement parity carried by its full surface.
/// The materialized surface's Pauli-product sign is intentionally absent:
/// consumers already realize it in those generator values, rather than adding
/// another affine term in this solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputCorrectionRow {
    /// Requested outputs on which this row has non-identity Pauli support.
    pub output_support: Vec<(IVec3, Pauli)>,
    /// Original public generator ordinals whose decoded values form the RHS.
    pub readout_ordinals: Vec<usize>,
}

/// An error from solving or evaluating the output-correction system.
// Deliberately not `#[non_exhaustive]`: the verifier maps every variant and
// should fail to compile when one is added.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum OutputCorrectionError {
    /// Runtime right-hand side has the wrong length.
    #[error("output-correction RHS length mismatch: expected {expected}, got {actual}")]
    RhsLengthMismatch {
        /// Required bit count.
        expected: usize,
        /// Supplied bit count.
        actual: usize,
    },
    /// A correction row names an unrequested output.
    #[error("output {output} named by a correction surface is not a requested output")]
    UnknownOutput {
        /// Unknown output position.
        output: IVec3,
    },
    /// The correction equations are inconsistent.
    #[error("output corrections are inconsistent")]
    Inconsistent,
}

/// One output's frame bits, each expressed as the constraint-index set whose
/// runtime RHS bits XOR into it. An empty set is the constant `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicFramePair {
    /// The output port these frame bits belong to.
    pub output: IVec3,
    /// Constraint indices (into the `constraints` slice given to
    /// [`solve_output_correction_symbolic`]) whose RHS XOR gives the X-frame bit.
    pub x: Vec<usize>,
    /// Same, for the Z-frame bit.
    pub z: Vec<usize>,
}

/// The eliminated system: per-output frame-bit assignments plus the dependent
/// rows whose RHS must XOR to zero for the system to be consistent at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicOutputCorrection {
    frames: Vec<SymbolicFramePair>,
    inconsistencies: Vec<Vec<usize>>,
    num_constraints: usize,
}

impl SymbolicOutputCorrection {
    /// Returns the per-output symbolic frame-bit assignments.
    pub fn frames(&self) -> &[SymbolicFramePair] {
        &self.frames
    }

    /// Substitute concrete RHS bits (one per constraint, in the order given to
    /// [`solve_output_correction_symbolic`]) into the solved system.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched RHS length or an inconsistent system.
    pub fn evaluate(
        &self,
        needs_flip: &[bool],
    ) -> Result<Vec<(IVec3, Pauli)>, OutputCorrectionError> {
        if needs_flip.len() != self.num_constraints {
            return Err(OutputCorrectionError::RhsLengthMismatch {
                expected: self.num_constraints,
                actual: needs_flip.len(),
            });
        }
        let xor = |set: &[usize]| {
            set.iter()
                .fold(false, |acc, &index| acc ^ needs_flip[index])
        };
        if self.inconsistencies.iter().any(|set| xor(set)) {
            return Err(OutputCorrectionError::Inconsistent);
        }
        Ok(self
            .frames
            .iter()
            .map(|pair| (pair.output, Pauli::from_xz(xor(&pair.x), xor(&pair.z))))
            .collect())
    }
}

/// Pivot output corrections while retaining each compact equation and original
/// raw row. Callers choose which physical support they need to materialize.
pub(super) fn pivot_output_correction_rows(
    mut rows: Vec<PauliString>,
    mut coeffs: Vec<CoeffVec>,
    outputs: &[IVec3],
    output_cols: &[usize],
    input_cols: &[usize],
    protected_cols: &[usize],
) -> Vec<(OutputCorrectionRow, PauliString)> {
    debug_assert_eq!(outputs.len(), output_cols.len());
    let num_pivoted = gaussian_elimination_with_tracking(
        &mut rows,
        &mut coeffs,
        axis_pivot_constraints(output_cols.iter().copied()),
        |row, &(col, basis)| row.get(col) & basis,
        None,
    );

    // Zero-output relations can label erased input/readout sectors. Reduce
    // the output witness's input operator against that kernel so its reference
    // map does not depend on which dense row happened to pivot first. Only
    // relations that preserve prepared resources and other live outputs may
    // participate: a row like X_input Z_T changes the resource map.
    let mut output_start = 0;
    if !input_cols.is_empty() {
        let protected_rank = gaussian_elimination_with_tracking(
            &mut rows[num_pivoted..],
            &mut coeffs[num_pivoted..],
            axis_pivot_constraints(protected_cols.iter().copied()),
            |row, &(col, basis)| row.get(col) & basis,
            None,
        );
        let kernel_start = num_pivoted + protected_rank;
        let kernel_len = rows.len() - kernel_start;
        if kernel_len != 0 {
            rows.rotate_left(kernel_start);
            coeffs.rotate_left(kernel_start);
            gaussian_elimination_with_tracking(
                &mut rows,
                &mut coeffs,
                axis_pivot_constraints(input_cols.iter().copied()),
                |row, &(col, basis)| row.get(col) & basis,
                Some(kernel_len),
            );
            output_start = kernel_len;
        }
    }

    rows.into_iter()
        .zip(coeffs)
        .skip(output_start)
        .take(num_pivoted)
        .filter_map(|(row, coeff)| {
            let output_support = outputs
                .iter()
                .zip(output_cols)
                .filter_map(|(&output, &col)| {
                    let pauli = row.get(col);
                    (pauli != Pauli::I).then_some((output, pauli))
                })
                .collect::<Vec<_>>();
            (!output_support.is_empty()).then(|| {
                (
                    OutputCorrectionRow {
                        output_support,
                        readout_ordinals: coeff.to_indices(),
                    },
                    row,
                )
            })
        })
        .collect()
}

/// Solve the joint output-correction system once, symbolically.
///
/// Variables are two frame bits per output: the X bit at column `2*i`
/// (constrained by the Z part of a surface's Pauli at output `i`) and the Z
/// bit at column `2*i + 1` (constrained by the X part). Each constraint row's
/// RHS is tracked as a singleton constraint-index set and XOR-combined through
/// the elimination; free variables solve to the constant `false` (empty set).
///
/// # Errors
///
/// Returns [`OutputCorrectionError::UnknownOutput`] if a row names an absent output.
pub fn solve_output_correction_symbolic(
    outputs: &[IVec3],
    constraints: &[OutputCorrectionRow],
) -> Result<SymbolicOutputCorrection, OutputCorrectionError> {
    solve_output_correction_symbolic_prefix(outputs, constraints, outputs.len())
}

/// Solve the full joint system, materializing at most its first `output_count`
/// outputs. All constraints and runtime consistency checks remain active.
///
/// Pivot order and the free-variable policy are identical to
/// [`solve_output_correction_symbolic`]. A module linker can put public outputs
/// first and retain internal seam variables without expanding their frames.
///
/// # Errors
///
/// Returns [`OutputCorrectionError::UnknownOutput`] if a row names an absent output.
///
/// # Panics
///
/// Panics if `output_count` exceeds `outputs.len()`.
pub fn solve_output_correction_symbolic_prefix(
    outputs: &[IVec3],
    constraints: &[OutputCorrectionRow],
    output_count: usize,
) -> Result<SymbolicOutputCorrection, OutputCorrectionError> {
    let num_vars = outputs.len() * 2;
    let num_constraints = constraints.len();
    let mut output_indices = FxHashMap::default();
    for (index, &output) in outputs.iter().enumerate() {
        // Preserve the first occurrence, as the original linear lookup did.
        output_indices.entry(output).or_insert(index);
    }
    let mut parities = ConstraintParities {
        sources: num_constraints,
        xors: Vec::new(),
    };
    let mut rows = constraints
        .iter()
        .enumerate()
        .map(|(constraint_index, constraint)| {
            let mut lhs = BTreeSet::new();
            for &(output, surface_pauli) in &constraint.output_support {
                let &output_index = output_indices
                    .get(&output)
                    .ok_or(OutputCorrectionError::UnknownOutput { output })?;
                if surface_pauli & Pauli::Z {
                    toggle(&mut lhs, 2 * output_index);
                }
                if surface_pauli & Pauli::X {
                    toggle(&mut lhs, 2 * output_index + 1);
                }
            }
            Ok((lhs, Some(constraint_index)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Index only unpivoted rows by their first live column. Forward elimination
    // touches a column's incident rows, not every row in the joint system.
    let mut by_first = vec![BTreeSet::new(); num_vars];
    for (index, (lhs, _)) in rows.iter().enumerate() {
        if let Some(&col) = lhs.first() {
            by_first[col].insert(index);
        }
    }
    let mut pivot_row = 0;
    for col in 0..num_vars {
        let Some(found) = by_first[col].pop_first() else {
            continue;
        };
        // Preserve the original row-swap policy: it determines the chosen RHS
        // representative when constraints are dependent.
        if found != pivot_row {
            if let Some(&displaced_col) = rows[pivot_row].0.first() {
                by_first[displaced_col].remove(&pivot_row);
                by_first[displaced_col].insert(found);
            }
            rows.swap(pivot_row, found);
        }
        let pivot = rows[pivot_row].clone();
        for index in std::mem::take(&mut by_first[col]) {
            let (lhs, rhs) = &mut rows[index];
            for &entry in &pivot.0 {
                toggle(lhs, entry);
            }
            *rhs = parities.xor(*rhs, pivot.1);
            if let Some(&next_col) = lhs.first() {
                by_first[next_col].insert(index);
            }
        }
        pivot_row += 1;
    }

    let inconsistencies = rows[pivot_row..]
        .iter()
        .map(|(lhs, rhs)| {
            debug_assert!(lhs.is_empty());
            parities.indices(*rhs)
        })
        .filter(|set| !set.is_empty())
        .collect();

    // Back-substitution shares XOR expressions instead of eagerly copying a
    // growing constraint-index vector into every intermediate frame.
    let mut solution = vec![None; num_vars];
    for (lhs, rhs) in rows[..pivot_row].iter().rev() {
        let mut columns = lhs.iter();
        let &col = columns.next().expect("pivot row is nonzero");
        let mut value = *rhs;
        for &other in columns {
            value = parities.xor(value, solution[other]);
        }
        solution[col] = value;
    }
    let frames = outputs
        .iter()
        .take(output_count)
        .enumerate()
        .map(|(index, &output)| SymbolicFramePair {
            output,
            x: parities.indices(solution[2 * index]),
            z: parities.indices(solution[2 * index + 1]),
        })
        .collect();

    Ok(SymbolicOutputCorrection {
        frames,
        inconsistencies,
        num_constraints,
    })
}

/// Original constraint indices are leaves; later indices name shared XORs.
/// `None` is zero. Children always precede their parent, allowing cancellation
/// before expansion without recursion or revisiting a shared subexpression.
struct ConstraintParities {
    sources: usize,
    xors: Vec<(usize, usize)>,
}

impl ConstraintParities {
    fn xor(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        match (left, right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) if left == right => None,
            (Some(left), Some(right)) => {
                self.xors.push((left, right));
                Some(self.sources + self.xors.len() - 1)
            }
        }
    }

    fn indices(&self, expression: Option<usize>) -> Vec<usize> {
        let mut pending = expression.into_iter().collect::<BTreeSet<_>>();
        while let Some(&node) = pending.last() {
            if node < self.sources {
                break;
            }
            pending.pop_last();
            let (left, right) = self.xors[node - self.sources];
            toggle(&mut pending, left);
            toggle(&mut pending, right);
        }
        pending.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use glam::ivec3;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    fn constraint(support: Vec<(IVec3, Pauli)>) -> OutputCorrectionRow {
        OutputCorrectionRow {
            output_support: support,
            readout_ordinals: Vec::new(),
        }
    }

    #[test]
    fn input_kernel_normalization_preserves_resources_and_readout_provenance() {
        // A, B are arbitrary inputs; T is a prepared resource; O is requested.
        // These rows commute. K erases an input sector, while J also acts on T.
        let r = PauliString::try_from("X__X").unwrap();
        let k = PauliString::try_from("XX__").unwrap();
        let j = PauliString::try_from("X_X_").unwrap();
        let coeffs = (0..3)
            .map(|index| CoeffVec::singleton(index, 3))
            .collect::<Vec<_>>();
        let solve = |rows, coeffs| {
            pivot_output_correction_rows(rows, coeffs, &[ivec3(0, 0, 1)], &[3], &[0, 1], &[2])
        };
        let expected = solve(vec![r.clone(), k.clone(), j.clone()], coeffs.clone());
        assert_eq!(expected.len(), 1);
        assert_eq!(expected[0].1, PauliString::try_from("_X_X").unwrap());
        assert_eq!(expected[0].0.readout_ordinals, vec![0, 1]);

        let mut combined = coeffs[0].clone();
        combined.xor_assign(&coeffs[1]);
        assert_eq!(
            solve(
                vec![&r ^ &k, j.clone(), k],
                vec![combined, coeffs[2].clone(), coeffs[1].clone()]
            ),
            expected,
            "replacing a witness by the same input-kernel coset preserves its map and provenance"
        );
        let protected = solve(
            vec![r.clone(), j],
            vec![coeffs[0].clone(), coeffs[2].clone()],
        );
        assert_eq!(
            protected[0].1, r,
            "a resource-bearing relation cannot change the witness"
        );
        assert_eq!(protected[0].0.readout_ordinals, vec![0]);
    }

    /// Reference concrete elimination used to check the symbolic solver.
    fn reference_solve(
        outputs: &[IVec3],
        constraints: &[(Vec<(usize, Pauli)>, bool)],
    ) -> Result<Vec<(IVec3, Pauli)>, ()> {
        let num_vars = outputs.len() * 2;
        let mut rows = constraints
            .iter()
            .map(|(support, needs_flip)| {
                let mut row = vec![false; num_vars + 1];
                for &(output_index, surface_pauli) in support {
                    if surface_pauli & Pauli::Z {
                        row[2 * output_index] ^= true;
                    }
                    if surface_pauli & Pauli::X {
                        row[2 * output_index + 1] ^= true;
                    }
                }
                row[num_vars] = *needs_flip;
                row
            })
            .collect::<Vec<_>>();

        let mut pivot_row = 0;
        for col in 0..num_vars {
            let Some(found) = (pivot_row..rows.len()).find(|&row| rows[row][col]) else {
                continue;
            };
            rows.swap(pivot_row, found);
            let pivot = rows[pivot_row].clone();
            for (row_index, row) in rows.iter_mut().enumerate() {
                if row_index == pivot_row || !row[col] {
                    continue;
                }
                for (entry, pivot_entry) in row.iter_mut().zip(&pivot).skip(col) {
                    *entry ^= *pivot_entry;
                }
            }
            pivot_row += 1;
        }

        if rows
            .iter()
            .any(|row| row[..num_vars].iter().all(|bit| !*bit) && row[num_vars])
        {
            return Err(());
        }

        let mut solution = vec![false; num_vars];
        for row in rows.iter().take(pivot_row) {
            if let Some(col) = row[..num_vars].iter().position(|bit| *bit) {
                solution[col] = row[num_vars];
            }
        }

        Ok(outputs
            .iter()
            .copied()
            .enumerate()
            .map(|(index, output)| {
                (
                    output,
                    Pauli::from_xz(solution[2 * index], solution[2 * index + 1]),
                )
            })
            .collect())
    }

    const PAULIS: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

    #[test]
    fn symbolic_solve_matches_reference_on_random_systems() {
        let mut rng = StdRng::seed_from_u64(0x0510);
        for case in 0..600 {
            let num_outputs = rng.random_range(1..=if case < 500 { 3 } else { 70 });
            let outputs: Vec<IVec3> = (0..num_outputs).map(|i| ivec3(i, 0, i)).collect();
            let num_constraints =
                rng.random_range(0..=if case < 500 { 6 } else { 2 * outputs.len() + 3 });
            let mut constraints: Vec<OutputCorrectionRow> = (0..num_constraints)
                .map(|_| {
                    constraint(
                        outputs
                            .iter()
                            .filter_map(|&output| {
                                let pauli = if case >= 500 && case % 2 == 0 {
                                    if rng.random_range(0..8) == 0 {
                                        PAULIS[rng.random_range(1..4)]
                                    } else {
                                        Pauli::I
                                    }
                                } else {
                                    PAULIS[rng.random_range(0..4)]
                                };
                                (pauli != Pauli::I).then_some((output, pauli))
                            })
                            .collect(),
                    )
                })
                .collect();
            if case % 3 == 0 && constraints.len() > 1 {
                constraints[1] = constraints[0].clone();
            }
            let symbolic = solve_output_correction_symbolic(&outputs, &constraints)
                .expect("all supports name requested outputs");
            let projected = [0, outputs.len() / 2, outputs.len() + 1].map(|count| {
                let solve =
                    solve_output_correction_symbolic_prefix(&outputs, &constraints, count).unwrap();
                assert_eq!(
                    solve.frames(),
                    &symbolic.frames()[..count.min(outputs.len())]
                );
                solve
            });

            // Exercise both arbitrary RHSs and guaranteed-consistent RHSs so
            // larger overdetermined systems also check concrete frame values.
            for trial in 0..8 {
                let assigned = (0..outputs.len())
                    .map(|_| (rng.random::<bool>(), rng.random::<bool>()))
                    .collect::<Vec<_>>();
                let reference_constraints = constraints
                    .iter()
                    .map(|constraint| {
                        let indexed = constraint
                            .output_support
                            .iter()
                            .map(|&(output, pauli)| {
                                let index = outputs
                                    .iter()
                                    .position(|candidate| *candidate == output)
                                    .expect("support built from outputs");
                                (index, pauli)
                            })
                            .collect::<Vec<_>>();
                        let flip = if trial < 4 {
                            indexed.iter().fold(false, |value, &(index, pauli)| {
                                value
                                    ^ (pauli & Pauli::Z && assigned[index].0)
                                    ^ (pauli & Pauli::X && assigned[index].1)
                            })
                        } else {
                            rng.random()
                        };
                        (indexed, flip)
                    })
                    .collect::<Vec<_>>();
                let needs_flip = reference_constraints
                    .iter()
                    .map(|(_, flip)| *flip)
                    .collect::<Vec<_>>();
                let expected = reference_solve(&outputs, &reference_constraints);
                for solve in std::iter::once(&symbolic).chain(&projected) {
                    match (solve.evaluate(&needs_flip), &expected) {
                        (Ok(actual), Ok(expected)) => {
                            assert_eq!(actual, expected[..solve.frames().len()])
                        }
                        (Err(OutputCorrectionError::Inconsistent), Err(())) => {}
                        (actual, expected) => {
                            panic!("solver drift: symbolic={actual:?} reference={expected:?}")
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn public_frame_through_long_seam_chain_keeps_all_constraint_parities() {
        let outputs = (0..4096).map(|i| ivec3(i, 0, 0)).collect::<Vec<_>>();
        let mut constraints = outputs
            .windows(2)
            .map(|pair| constraint(vec![(pair[0], Pauli::Z), (pair[1], Pauli::Z)]))
            .collect::<Vec<_>>();
        constraints[0]
            .output_support
            .extend([(outputs[0], Pauli::X), (outputs[0], Pauli::X)]);
        constraints.push(constraint(vec![(*outputs.last().unwrap(), Pauli::Z)]));
        let solve = solve_output_correction_symbolic_prefix(&outputs, &constraints, 1).unwrap();
        assert_eq!(
            solve.frames(),
            &[SymbolicFramePair {
                output: outputs[0],
                x: (0..constraints.len()).collect(),
                z: Vec::new(),
            }]
        );
    }

    #[test]
    fn unknown_output_in_support_is_rejected() {
        let outputs = [ivec3(0, 0, 1)];
        let constraints = [constraint(vec![(ivec3(9, 9, 9), Pauli::Z)])];
        assert_eq!(
            solve_output_correction_symbolic(&outputs, &constraints),
            Err(OutputCorrectionError::UnknownOutput {
                output: ivec3(9, 9, 9)
            })
        );
        assert!(matches!(
            solve_output_correction_symbolic_prefix(&outputs, &constraints, 0),
            Err(OutputCorrectionError::UnknownOutput { .. })
        ));
    }

    #[test]
    fn dependent_rows_become_runtime_consistency_checks() {
        // Two identical constraints: the second eliminates to a zero LHS whose
        // RHS set {0, 1} must XOR to zero at runtime.
        let outputs = [ivec3(0, 0, 1)];
        let constraints = [
            constraint(vec![(ivec3(0, 0, 1), Pauli::Z)]),
            constraint(vec![(ivec3(0, 0, 1), Pauli::Z)]),
        ];
        let symbolic = solve_output_correction_symbolic(&outputs, &constraints).unwrap();

        assert_eq!(
            symbolic.evaluate(&[true, true]).unwrap(),
            vec![(ivec3(0, 0, 1), Pauli::X)]
        );
        assert_eq!(symbolic.frames()[0].x, vec![0]);
        let error = symbolic
            .evaluate(&[true, false])
            .expect_err("unequal dependent parities are inconsistent");
        assert_eq!(error, OutputCorrectionError::Inconsistent);
        assert_eq!(error.to_string(), "output corrections are inconsistent");
    }

    #[test]
    fn evaluate_rejects_rhs_length_mismatch() {
        let outputs = [ivec3(0, 0, 1)];
        let constraints = [constraint(vec![(ivec3(0, 0, 1), Pauli::Z)])];
        let symbolic = solve_output_correction_symbolic(&outputs, &constraints).unwrap();

        assert_eq!(
            symbolic.evaluate(&[]),
            Err(OutputCorrectionError::RhsLengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            symbolic.evaluate(&[false, true]),
            Err(OutputCorrectionError::RhsLengthMismatch {
                expected: 1,
                actual: 2,
            })
        );
    }
}
