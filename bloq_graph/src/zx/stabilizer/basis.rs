//! Pauli row bases, coordinates, and rank-preserving elimination.

use bloq_utils::{Pauli, PauliString};

use super::super::ZXGraph;
use super::affine::{
    AffineTrackedSolution, ExactPauliConstraint, materialize_tracked_row, solve_affine_tracked,
};

/// Gaussian-eliminated generators projected against a deadline, plus the number
/// of rows consumed as pivots for the forbidden (post-deadline) columns.
pub(crate) struct DeadlineProjection {
    pub consumed: usize,
    pub rows: Vec<PauliString>,
    pub coeffs: Vec<CoeffVec>,
}

/// Dense GF(2) coefficient vector with a bit-granular logical length.
///
/// Deliberately hand-rolled rather than wrapping [`binar::BitVec`], the
/// workspace's bit-vector (already used by `bloq_utils::PauliString`). `binar`
/// allocates and operates in 512-bit `BitBlock`s, sized for the wide rows
/// `PauliString` carries. Coefficient vectors are one bit per input generator —
/// usually tens — so every vector would round up to a 64-byte block and every
/// `xor_assign` would XOR eight words instead of one. Measured on
/// `GalleryItem::iter()` stabilizer computation, the `binar`-backed version cost
/// ~36% (0.080s → 0.108s, best of 10); word-level `bit` reads recovered almost
/// none of it, since the cost is in the block-granular XOR, and `binar` exposes
/// no mutable word access to route around it.
///
/// Swap to `binar` if these vectors ever get wide (thousands of generators),
/// where block-at-a-time SIMD would start to pay for the padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CoeffVec {
    words: Vec<u64>,
    len: usize,
}

impl CoeffVec {
    pub(crate) fn zeros(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    pub(crate) fn singleton(index: usize, len: usize) -> Self {
        let mut value = Self::zeros(len);
        value.set_bit(index, true);
        value
    }

    pub(crate) fn bit(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = index % 64;
        (self.words[word] >> bit) & 1 == 1
    }

    pub(crate) fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = index % 64;
        if value {
            self.words[word] |= 1_u64 << bit;
        } else {
            self.words[word] &= !(1_u64 << bit);
        }
    }

    pub(crate) fn xor_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (target, source) in self.words.iter_mut().zip(&other.words) {
            *target ^= *source;
        }
    }

    pub(crate) fn first_one(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .find_map(|(word_index, &word)| {
                (word != 0).then(|| word_index * 64 + word.trailing_zeros() as usize)
            })
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    pub(crate) fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(move |(word_index, &word)| {
                let mut word = word;
                std::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let index = word_index * 64 + bit;
                    (index < self.len).then_some(index)
                })
            })
    }

    pub(crate) fn to_indices(&self) -> Vec<usize> {
        self.iter_ones().collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn resize(&mut self, new_len: usize) {
        if self.len == new_len {
            return;
        }

        self.words.resize(new_len.div_ceil(64), 0);
        self.len = new_len;
        if let Some(last) = self.words.last_mut() {
            let trailing = new_len % 64;
            if trailing != 0 {
                *last &= (1_u64 << trailing) - 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CoeffSpan {
    basis: Vec<CoeffVec>,
    pivots: Vec<usize>,
    width: usize,
}

impl CoeffSpan {
    pub(super) fn new(width: usize) -> Self {
        Self {
            basis: Vec::new(),
            pivots: Vec::new(),
            width,
        }
    }

    pub(super) fn from_coeffs<'a>(
        width: usize,
        coeffs: impl IntoIterator<Item = &'a CoeffVec>,
    ) -> Self {
        let mut span = Self::new(width);
        for coeff in coeffs {
            span.insert_if_independent(coeff.clone());
        }
        span
    }

    pub(super) fn reduce(&self, value: &CoeffVec) -> CoeffVec {
        debug_assert_eq!(value.len(), self.width);
        let mut reduced = value.clone();
        for (basis, &pivot) in self.basis.iter().zip(&self.pivots) {
            if reduced.bit(pivot) {
                reduced.xor_assign(basis);
            }
        }
        reduced
    }

    #[cfg(test)]
    pub(super) fn contains(&self, value: &CoeffVec) -> bool {
        self.reduce(value).is_zero()
    }

    pub(super) fn insert_if_independent(&mut self, value: CoeffVec) -> bool {
        let reduced = self.reduce(&value);
        let Some(pivot) = reduced.first_one() else {
            return false;
        };
        self.basis.push(reduced);
        self.pivots.push(pivot);
        true
    }

    #[cfg(test)]
    pub(super) fn rank(&self) -> usize {
        self.basis.len()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PauliSpan {
    basis: Vec<PauliString>,
    pivots: Vec<(usize, Pauli)>,
    width: usize,
}

impl PauliSpan {
    pub(super) fn new(width: usize) -> Self {
        Self {
            basis: Vec::new(),
            pivots: Vec::new(),
            width,
        }
    }

    pub(super) fn from_rows(rows: &[PauliString], width: usize) -> Self {
        let mut span = Self::new(width);
        for row in rows {
            span.insert_if_independent(row.clone());
        }
        span
    }

    pub(super) fn reduce(&self, value: &PauliString) -> PauliString {
        debug_assert_eq!(value.len(), self.width);
        let mut reduced = value.clone();
        for (basis, &(col, axis)) in self.basis.iter().zip(&self.pivots) {
            if reduced.get(col) & axis {
                reduced ^= basis;
            }
        }
        reduced
    }

    pub(super) fn insert_if_independent(&mut self, value: PauliString) -> bool {
        let reduced = self.reduce(&value);
        let Some(pivot) = self.first_pivot(&reduced) else {
            return false;
        };
        self.basis.push(reduced);
        self.pivots.push(pivot);
        true
    }

    #[cfg(test)]
    pub(super) fn contains(&self, value: &PauliString) -> bool {
        self.reduce(value).is_identity()
    }

    pub(super) fn rank(&self) -> usize {
        self.basis.len()
    }

    fn first_pivot(&self, value: &PauliString) -> Option<(usize, Pauli)> {
        debug_assert_eq!(value.len(), self.width);
        first_pauli_pivot(value)
    }
}

fn first_pauli_pivot(value: &PauliString) -> Option<(usize, Pauli)> {
    let first = |words: &[u64]| {
        words.iter().enumerate().find_map(|(word, &bits)| {
            (bits != 0).then(|| word * 64 + bits.trailing_zeros() as usize)
        })
    };
    match (first(value.x_words()), first(value.z_words())) {
        (Some(x), Some(z)) => Some(if x <= z { (x, Pauli::X) } else { (z, Pauli::Z) }),
        (Some(x), None) => Some((x, Pauli::X)),
        (None, Some(z)) => Some((z, Pauli::Z)),
        (None, None) => None,
    }
}

pub(crate) fn solve_pauli_combination(
    rows: &[PauliString],
    target: &PauliString,
) -> Option<CoeffVec> {
    PauliCombinationBasis::new(rows).solve(target)
}

/// Factors a fixed row space once, retaining coordinates in the original rows.
#[derive(Debug)]
pub(super) struct PauliCombinationBasis {
    basis: Vec<PauliString>,
    witnesses: Vec<CoeffVec>,
    pivots: Vec<(usize, Pauli)>,
    source_count: usize,
}

impl PauliCombinationBasis {
    pub(super) fn new(rows: &[PauliString]) -> Self {
        let mut basis = Vec::new();
        let mut witnesses = Vec::new();
        let mut pivots = Vec::new();

        for (index, row) in rows.iter().enumerate() {
            let mut reduced = row.clone();
            let mut witness = CoeffVec::singleton(index, rows.len());
            for ((basis_row, basis_witness), &(col, axis)) in
                basis.iter().zip(&witnesses).zip(&pivots)
            {
                if reduced.get(col) & axis {
                    reduced ^= basis_row;
                    witness.xor_assign(basis_witness);
                }
            }
            let Some(pivot) = first_pauli_pivot(&reduced) else {
                continue;
            };
            basis.push(reduced);
            witnesses.push(witness);
            pivots.push(pivot);
        }

        Self {
            basis,
            witnesses,
            pivots,
            source_count: rows.len(),
        }
    }

    pub(super) fn solve(&self, target: &PauliString) -> Option<CoeffVec> {
        let mut reduced = target.clone();
        let mut witness = CoeffVec::zeros(self.source_count);
        for ((basis_row, basis_witness), &(col, axis)) in
            self.basis.iter().zip(&self.witnesses).zip(&self.pivots)
        {
            if reduced.get(col) & axis {
                reduced ^= basis_row;
                witness.xor_assign(basis_witness);
            }
        }
        reduced.is_identity().then_some(witness)
    }
}

#[cfg(test)]
pub(crate) fn coeffs_to_sparse(coeffs: &[CoeffVec]) -> Vec<Vec<usize>> {
    coeffs.iter().map(CoeffVec::to_indices).collect()
}

pub(crate) fn reduce_to_basis(rows: &[PauliString], width: usize) -> Vec<PauliString> {
    let mut span = PauliSpan::new(width);
    let mut reduced = Vec::new();
    for row in rows {
        if span.insert_if_independent(row.clone()) {
            reduced.push(row.clone());
        }
    }
    reduced
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TrackedBasis {
    pub(super) rows: Vec<PauliString>,
    pub(super) coeffs: Vec<CoeffVec>,
}

impl TrackedBasis {
    pub(super) fn new(rows: Vec<PauliString>) -> Self {
        let width = rows.len();
        let coeffs = (0..rows.len())
            .map(|index| CoeffVec::singleton(index, width))
            .collect::<Vec<_>>();
        Self { rows, coeffs }
    }

    pub(super) fn from_parts(rows: Vec<PauliString>, coeffs: Vec<CoeffVec>) -> Self {
        debug_assert_eq!(rows.len(), coeffs.len());
        let mut coeffs = coeffs;
        let width = coeffs.iter().map(CoeffVec::len).max().unwrap_or(0);
        resize_coeffs(&mut coeffs, width);
        Self { rows, coeffs }
    }

    pub(super) fn push(&mut self, row: PauliString, mut coeff: CoeffVec) {
        if coeff.len() > self.coeff_width() {
            self.resize_coeff_width(coeff.len());
        } else {
            coeff.resize(self.coeff_width());
        }

        self.rows.push(row);
        self.coeffs.push(coeff);
    }

    pub(super) fn replace_suffix(
        &mut self,
        start: usize,
        rows: &[PauliString],
        coeffs: &[CoeffVec],
    ) {
        debug_assert_eq!(self.rows.len().saturating_sub(start), rows.len());
        debug_assert_eq!(rows.len(), coeffs.len());
        if let Some(width) = coeffs.iter().map(CoeffVec::len).max()
            && width > self.coeff_width()
        {
            self.resize_coeff_width(width);
        }
        let coeff_width = self.coeff_width();
        self.rows[start..].clone_from_slice(rows);
        for (target, source) in self.coeffs[start..].iter_mut().zip(coeffs) {
            *target = source.clone();
            target.resize(coeff_width);
        }
    }

    fn coeff_width(&self) -> usize {
        self.coeffs.first().map(CoeffVec::len).unwrap_or(0)
    }

    fn resize_coeff_width(&mut self, width: usize) {
        resize_coeffs(&mut self.coeffs, width);
    }
}

fn resize_coeffs(coeffs: &mut [CoeffVec], width: usize) {
    for coeff in coeffs {
        coeff.resize(width);
    }
}

pub(crate) fn axis_pivot_constraints(cols: impl IntoIterator<Item = usize>) -> Vec<(usize, Pauli)> {
    cols.into_iter()
        .flat_map(|col| [(col, Pauli::X), (col, Pauli::Z)])
        .collect()
}

pub(super) fn row_rank(rows: &[PauliString], width: usize) -> usize {
    PauliSpan::from_rows(rows, width).rank()
}

#[cfg(test)]
pub(super) fn coeff_rank(coeffs: &[CoeffVec], width: usize) -> usize {
    CoeffSpan::from_coeffs(width, coeffs).rank()
}

fn choose_measurement_free_coset_representative(
    solution: &AffineTrackedSolution,
    basis_rows: &[PauliString],
    width: usize,
    prefix_span: &CoeffSpan,
    seed_reduced: &CoeffVec,
) -> Option<(PauliString, CoeffVec)> {
    // Work in the quotient by the frozen measurement prefix: the returned
    // coefficient must reduce to the same coset as the seed row.
    let mut target_kernel_reduction = prefix_span.reduce(&solution.particular_coeff);
    target_kernel_reduction.xor_assign(seed_reduced);

    let reduced_kernel_coeffs = solution
        .kernel_coeffs
        .iter()
        .map(|kernel_coeff| prefix_span.reduce(kernel_coeff))
        .collect::<Vec<_>>();
    let kernel_combo = solve_coeff_combination(&reduced_kernel_coeffs, &target_kernel_reduction)?;

    let mut coeff = solution.particular_coeff.clone();
    for index in kernel_combo.iter_ones() {
        coeff.xor_assign(&solution.kernel_coeffs[index]);
    }

    let row = materialize_tracked_row(&coeff, basis_rows, width);
    Some((row, coeff))
}

/// Column-major full GF(2) elimination over the first `col_count` columns of
/// `rows`, swapping and xoring `witnesses` in lockstep (pass an empty slice to
/// skip witness tracking). Returns the pivot column of each solved row.
pub(super) fn echelonize_with_witnesses(
    rows: &mut [CoeffVec],
    witnesses: &mut [CoeffVec],
    col_count: usize,
) -> Vec<usize> {
    debug_assert!(witnesses.is_empty() || witnesses.len() == rows.len());
    let mut pivot_cols = Vec::new();
    let mut next_row = 0;

    for col in 0..col_count {
        let Some(pivot_index) = (next_row..rows.len()).find(|&row| rows[row].bit(col)) else {
            continue;
        };

        if pivot_index != next_row {
            rows.swap(pivot_index, next_row);
            if !witnesses.is_empty() {
                witnesses.swap(pivot_index, next_row);
            }
        }

        let pivot_row = rows[next_row].clone();
        let pivot_witness = witnesses.get(next_row).cloned();
        for row_index in 0..rows.len() {
            if row_index == next_row || !rows[row_index].bit(col) {
                continue;
            }
            rows[row_index].xor_assign(&pivot_row);
            if let Some(pivot_witness) = &pivot_witness {
                witnesses[row_index].xor_assign(pivot_witness);
            }
        }

        pivot_cols.push(col);
        next_row += 1;
    }

    pivot_cols
}

// Deliberately not expressed via `echelonize_with_witnesses`: the incremental
// forward elimination here picks a different (though equally valid) witness
// than full RREF when `vectors` are linearly dependent, and callers rely on
// the exact witness for deterministic canonical output.
pub(crate) fn solve_coeff_combination(vectors: &[CoeffVec], target: &CoeffVec) -> Option<CoeffVec> {
    let mut basis = Vec::new();
    let mut witnesses = Vec::new();
    let mut pivots = Vec::new();

    for (index, vector) in vectors.iter().enumerate() {
        let mut reduced = vector.clone();
        let mut witness = CoeffVec::singleton(index, vectors.len());

        for ((basis_vector, basis_witness), &pivot) in basis.iter().zip(&witnesses).zip(&pivots) {
            if !reduced.bit(pivot) {
                continue;
            }

            reduced.xor_assign(basis_vector);
            witness.xor_assign(basis_witness);
        }

        let Some(pivot) = reduced.first_one() else {
            continue;
        };
        basis.push(reduced);
        witnesses.push(witness);
        pivots.push(pivot);
    }

    let mut reduced_target = target.clone();
    let mut witness = CoeffVec::zeros(vectors.len());
    for ((basis_vector, basis_witness), &pivot) in basis.iter().zip(&witnesses).zip(&pivots) {
        if !reduced_target.bit(pivot) {
            continue;
        }

        reduced_target.xor_assign(basis_vector);
        witness.xor_assign(basis_witness);
    }

    reduced_target.is_zero().then_some(witness)
}
pub(super) fn complete_from_initial_basis_fast(
    final_basis: &mut TrackedBasis,
    basis_rows: &[PauliString],
    frozen_measurement_rows: &[PauliString],
    frozen_measurement_coeffs: &[CoeffVec],
    measurement_cols: &[usize],
    width: usize,
) {
    if final_basis.coeff_width() < basis_rows.len() {
        final_basis.resize_coeff_width(basis_rows.len());
    }

    let prefix_span = CoeffSpan::from_coeffs(basis_rows.len(), frozen_measurement_coeffs);
    let mut coeff_span = CoeffSpan::from_coeffs(basis_rows.len(), &final_basis.coeffs);

    let exact_constraints = measurement_cols
        .iter()
        .map(|&col| ExactPauliConstraint {
            col,
            pauli: Pauli::I,
        })
        .collect::<Vec<_>>();

    // The frozen measurement suffix is invariant across iterations — only slot 0
    // (the seed row / coeff) varies. Build the solve buffers once and overwrite
    // slot 0 each iteration rather than reallocating the constant suffix via a
    // fresh `concat` every time.
    let mut solve_rows = Vec::with_capacity(1 + frozen_measurement_rows.len());
    solve_rows.push(PauliString::new(width));
    solve_rows.extend_from_slice(frozen_measurement_rows);
    let mut solve_coeffs = Vec::with_capacity(1 + frozen_measurement_coeffs.len());
    solve_coeffs.push(CoeffVec::zeros(basis_rows.len()));
    solve_coeffs.extend_from_slice(frozen_measurement_coeffs);

    for (index, row) in basis_rows.iter().enumerate() {
        let seed_coeff = CoeffVec::singleton(index, basis_rows.len());
        if !coeff_span.insert_if_independent(seed_coeff.clone()) {
            continue;
        }

        let seed_reduced = prefix_span.reduce(&seed_coeff);
        debug_assert!(
            !seed_reduced.is_zero(),
            "independent seed should stay non-zero modulo frozen measurement prefix",
        );

        solve_rows[0] = row.clone();
        solve_coeffs[0] = seed_coeff.clone();
        let representative = solve_affine_tracked(&solve_rows, &solve_coeffs, &exact_constraints)
            .and_then(|solution| {
                choose_measurement_free_coset_representative(
                    &solution,
                    basis_rows,
                    width,
                    &prefix_span,
                    &seed_reduced,
                )
            });

        if let Some((representative_row, representative_coeff)) = representative {
            final_basis.push(representative_row, representative_coeff);
        } else {
            debug_assert_eq!(row.len(), width);
            final_basis.push(row.clone(), seed_coeff);
        }
    }

    debug_assert_eq!(final_basis.rows.len(), final_basis.coeffs.len());
}

/// Column-pivot GF(2) elimination over `rows`. Pass a `combinations` slice the
/// same length as `rows` to track generator provenance in lockstep, or an empty
/// slice to skip tracking (the plain elimination).
pub(crate) fn gaussian_elimination_with_tracking<T, F>(
    rows: &mut [PauliString],
    combinations: &mut [CoeffVec],
    pivot_constraints: impl IntoIterator<Item = T>,
    predicate: F,
    num_pivotable_rows: Option<usize>,
) -> usize
where
    F: Fn(&PauliString, &T) -> bool,
{
    let track = !combinations.is_empty();
    debug_assert!(!track || rows.len() == combinations.len());

    let num_pivotable_rows = num_pivotable_rows.unwrap_or(rows.len()).min(rows.len());
    let mut num_solved = 0;

    for constraint in pivot_constraints {
        if num_solved == num_pivotable_rows {
            break;
        }

        let Some(pivot_index) =
            (num_solved..num_pivotable_rows).find(|&index| predicate(&rows[index], &constraint))
        else {
            continue;
        };

        if pivot_index != num_solved {
            rows.swap(pivot_index, num_solved);
            if track {
                combinations.swap(pivot_index, num_solved);
            }
        }

        let (before, pivot_and_after) = rows.split_at_mut(num_solved);
        let (pivot_row, after) = pivot_and_after
            .split_first_mut()
            .expect("solved pivot is in bounds");
        if track {
            let (before_coeffs, pivot_and_after_coeffs) = combinations.split_at_mut(num_solved);
            let (pivot_coeff, after_coeffs) = pivot_and_after_coeffs
                .split_first_mut()
                .expect("tracked pivot is in bounds");
            for (row, coeff) in before
                .iter_mut()
                .zip(before_coeffs)
                .chain(after.iter_mut().zip(after_coeffs))
            {
                if predicate(row, &constraint) {
                    *row ^= &*pivot_row;
                    coeff.xor_assign(pivot_coeff);
                }
            }
        } else {
            for row in before.iter_mut().chain(after) {
                if predicate(row, &constraint) {
                    *row ^= &*pivot_row;
                }
            }
        }

        num_solved += 1;
    }

    num_solved
}

pub(crate) fn deadline_projection_with_initial_coeffs(
    generators: &[PauliString],
    coeffs: &[CoeffVec],
    zx: &ZXGraph,
    deadline: i64,
) -> DeadlineProjection {
    debug_assert_eq!(generators.len(), coeffs.len());
    let mut forbidden = Vec::new();
    // A selective cap gates at z, even after its fill changes NodeKind to
    // X/Y/Z. Preserve that source identity when projecting a resolved basis.
    let selective_nodes = zx
        .action_graph()
        .ordered_nodes()
        .filter_map(|node| match node.action {
            crate::Action::Resolve { target, .. } => zx.node_at(target).map(|node| node.id),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let is_late = |node: &super::super::ZXNode| {
        let z = i64::from(node.pos.z);
        if matches!(node.kind, crate::NodeKind::Selective(_)) || selective_nodes.contains(&node.id)
        {
            z > deadline
        } else {
            z >= deadline
        }
    };

    for node in &zx.nodes {
        if is_late(node) {
            forbidden.push(node.id);
        }
    }

    for edge in &zx.edges {
        if is_late(&zx.nodes[edge.n1]) || is_late(&zx.nodes[edge.n2]) {
            forbidden.push(edge.id);
        }
    }

    forbidden.sort_unstable();
    forbidden.dedup();
    let pivot_constraints = axis_pivot_constraints(forbidden.iter().copied());

    let mut rows = generators.to_vec();
    let mut coeffs = coeffs.to_vec();
    let consumed = gaussian_elimination_with_tracking(
        &mut rows,
        &mut coeffs,
        pivot_constraints,
        |row, &(col, basis)| row.get(col) & basis,
        None,
    );

    DeadlineProjection {
        consumed,
        rows,
        coeffs,
    }
}

#[cfg(test)]
mod tests {
    use bloq_utils::{Pauli, PauliString};

    use super::{
        AffineTrackedSolution, CoeffSpan, CoeffVec, PauliCombinationBasis, PauliSpan, TrackedBasis,
        choose_measurement_free_coset_representative, coeff_rank, complete_from_initial_basis_fast,
        reduce_to_basis, row_rank, solve_pauli_combination,
    };

    fn sparse_to_coeff(combination: &[usize], width: usize) -> CoeffVec {
        let mut coeff = CoeffVec::zeros(width);
        for &index in combination {
            coeff.set_bit(index, true);
        }
        coeff
    }

    fn algebraic_row_from_coeff(
        coeff: &CoeffVec,
        raw_external: &[PauliString],
        width: usize,
    ) -> PauliString {
        let mut row = PauliString::new(width);
        for index in coeff.to_indices() {
            row ^= &raw_external[index];
        }
        row
    }

    fn algebraic_rows_from_coeffs(
        coeffs: &[CoeffVec],
        raw_external: &[PauliString],
        width: usize,
    ) -> Vec<PauliString> {
        coeffs
            .iter()
            .map(|coeff| algebraic_row_from_coeff(coeff, raw_external, width))
            .collect()
    }

    #[test]
    fn basis_reduce_preserves_span_and_rank() {
        let rows = vec![
            PauliString::try_from("X__").unwrap(),
            PauliString::try_from("X__").unwrap(),
            PauliString::try_from("_Z_").unwrap(),
            PauliString::try_from("XZ_").unwrap(),
        ];

        let reduced = reduce_to_basis(&rows, 3);

        assert_eq!(row_rank(&reduced, 3), row_rank(&rows, 3));
        assert_eq!(reduced.len(), row_rank(&rows, 3));
        assert_eq!(reduced[0], PauliString::try_from("X__").unwrap());
        assert_eq!(reduced[1], PauliString::try_from("_Z_").unwrap());
    }

    #[test]
    fn pauli_span_reports_membership_without_mutation() {
        let rows = vec![
            PauliString::try_from("X_").unwrap(),
            PauliString::try_from("_Z").unwrap(),
        ];
        let span = PauliSpan::from_rows(&rows, 2);

        assert!(span.contains(&PauliString::try_from("XZ").unwrap()));
        assert!(!span.contains(&PauliString::try_from("_X").unwrap()));
        assert_eq!(span.rank(), 2);
    }

    #[test]
    fn pauli_combination_reuses_original_coordinates_with_dependent_rows() {
        let mut rows = vec![PauliString::new(129); 67];
        rows[0].set(0, Pauli::X);
        rows[64].set(64, Pauli::Z);
        rows[65] = rows[0].clone();
        rows[66] = rows[0].clone();
        rows[66].set(128, Pauli::Y);
        let basis = PauliCombinationBasis::new(&rows);
        for mask in 0..8 {
            let indices = [0, 64, 66]
                .into_iter()
                .enumerate()
                .filter_map(|(bit, index)| (mask & (1 << bit) != 0).then_some(index))
                .collect::<Vec<_>>();
            let expected = sparse_to_coeff(&indices, rows.len());
            let target = algebraic_row_from_coeff(&expected, &rows, 129);
            assert_eq!(basis.solve(&target), Some(expected));
        }
        let mut outside = PauliString::new(129);
        outside.set(0, Pauli::Z);
        assert!(basis.solve(&outside).is_none());
        assert!(solve_pauli_combination(&rows, &outside).is_none());
        assert_eq!(
            solve_pauli_combination(&[], &PauliString::new(129)),
            Some(CoeffVec::zeros(0))
        );
    }

    #[test]
    fn coeff_rank_counts_independent_coefficients() {
        let coeffs = vec![CoeffVec::singleton(0, 3), CoeffVec::singleton(1, 3), {
            let mut combined = CoeffVec::singleton(0, 3);
            combined.xor_assign(&CoeffVec::singleton(1, 3));
            combined
        }];

        assert_eq!(coeff_rank(&coeffs, 3), 2);
    }

    #[test]
    fn coeff_span_detects_independent_and_dependent_vectors() {
        let mut span = CoeffSpan::new(5);
        let first = CoeffVec::singleton(1, 5);
        let second = CoeffVec::singleton(3, 5);
        let mut combined = first.clone();
        combined.xor_assign(&second);

        assert!(span.insert_if_independent(first.clone()));
        assert!(span.contains(&first));
        assert!(!span.insert_if_independent(first));
        assert!(!span.contains(&second));
        assert!(span.insert_if_independent(second));
        assert!(span.contains(&combined));
        assert!(!span.insert_if_independent(combined));
    }

    #[test]
    fn coeff_vec_crosses_u64_word_boundaries() {
        let mut coeff = CoeffVec::zeros(130);
        coeff.set_bit(1, true);
        coeff.set_bit(64, true);
        coeff.set_bit(129, true);

        let mut other = CoeffVec::singleton(64, 130);
        other.set_bit(65, true);
        coeff.xor_assign(&other);

        assert_eq!(coeff.iter_ones().collect::<Vec<_>>(), vec![1, 65, 129]);
        assert_eq!(coeff.to_indices(), vec![1, 65, 129]);
        assert_eq!(coeff.first_one(), Some(1));
    }

    #[test]
    fn complete_from_initial_basis_fast_zeroes_measurement_columns_when_feasible() {
        let basis_rows = vec![
            PauliString::try_from("XX").unwrap(),
            PauliString::try_from("XZ").unwrap(),
        ];
        let measurement_rows = vec![PauliString::try_from("XX").unwrap()];
        let measurement_coeffs = vec![CoeffVec::singleton(0, basis_rows.len())];
        let mut final_basis =
            TrackedBasis::from_parts(measurement_rows.clone(), measurement_coeffs.clone());

        complete_from_initial_basis_fast(
            &mut final_basis,
            &basis_rows,
            &measurement_rows,
            &measurement_coeffs,
            &[0],
            2,
        );

        assert_eq!(row_rank(&final_basis.rows, 2), 2);
        assert_eq!(final_basis.rows.len(), 2);
        assert_eq!(
            final_basis.coeffs[0],
            sparse_to_coeff(&[0], basis_rows.len())
        );
        assert_eq!(
            algebraic_rows_from_coeffs(&final_basis.coeffs, &basis_rows, 2),
            final_basis.rows
        );
        assert!(
            final_basis
                .rows
                .iter()
                .skip(1)
                .all(|row| row.get(0) == Pauli::I)
        );
    }

    #[test]
    fn choose_measurement_free_coset_representative_can_combine_multiple_kernel_directions() {
        let prefix_span = CoeffSpan::new(4);
        let solution = AffineTrackedSolution {
            particular_coeff: sparse_to_coeff(&[1], 4),
            kernel_coeffs: vec![sparse_to_coeff(&[0], 4), sparse_to_coeff(&[1], 4)],
        };
        let basis_rows = vec![
            PauliString::try_from("Y").unwrap(),
            PauliString::try_from("X").unwrap(),
            PauliString::try_from("Z").unwrap(),
            PauliString::try_from("I").unwrap(),
        ];
        let seed_reduced = sparse_to_coeff(&[0], 4);

        let (row, coeff) = choose_measurement_free_coset_representative(
            &solution,
            &basis_rows,
            1,
            &prefix_span,
            &seed_reduced,
        )
        .expect("two kernel directions together should recover the requested coset");

        assert_eq!(coeff, sparse_to_coeff(&[0], 4));
        assert_eq!(row, PauliString::try_from("Y").unwrap());
    }

    #[test]
    fn complete_from_initial_basis_fast_falls_back_when_no_zero_measurement_representative_exists()
    {
        let basis_rows = vec![
            PauliString::try_from("XI").unwrap(),
            PauliString::try_from("ZI").unwrap(),
        ];
        let frozen_measurement_rows = vec![PauliString::try_from("ZI").unwrap()];
        let frozen_measurement_coeffs = vec![CoeffVec::singleton(1, basis_rows.len())];
        let mut final_basis = TrackedBasis::from_parts(
            frozen_measurement_rows.clone(),
            frozen_measurement_coeffs.clone(),
        );

        complete_from_initial_basis_fast(
            &mut final_basis,
            &basis_rows,
            &frozen_measurement_rows,
            &frozen_measurement_coeffs,
            &[0],
            2,
        );

        assert_eq!(row_rank(&final_basis.rows, 2), 2);
        assert_eq!(final_basis.rows.len(), 2);
        assert_eq!(
            algebraic_rows_from_coeffs(&final_basis.coeffs, &basis_rows, 2),
            final_basis.rows
        );
        assert_eq!(final_basis.rows[0], PauliString::try_from("ZI").unwrap());
        assert_eq!(final_basis.rows[1], PauliString::try_from("XI").unwrap());
        assert_eq!(
            final_basis.coeffs[1],
            sparse_to_coeff(&[0], basis_rows.len())
        );
    }
}
