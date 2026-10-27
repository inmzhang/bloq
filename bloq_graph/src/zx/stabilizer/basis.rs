//! Pauli row bases, coordinates, and rank-preserving elimination.

use bloq_utils::{Pauli, PauliString};
use std::ops::BitXorAssign;

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

    pub(crate) fn zeros_like(&self) -> Self {
        Self::zeros(self.len)
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

    #[cfg(feature = "verify")]
    pub(crate) fn first_one(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .find_map(|(word_index, &word)| {
                (word != 0).then(|| word_index * 64 + word.trailing_zeros() as usize)
            })
    }

    #[cfg(feature = "verify")]
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
}

impl BitXorAssign<&Self> for CoeffVec {
    fn bitxor_assign(&mut self, other: &Self) {
        self.xor_assign(other);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PauliSpan {
    basis: Vec<PauliString>,
    pivots: Vec<(usize, Pauli)>,
    width: usize,
}

impl PauliSpan {
    pub(crate) fn new(width: usize) -> Self {
        Self {
            basis: Vec::new(),
            pivots: Vec::new(),
            width,
        }
    }

    pub(crate) fn reduce(&self, value: &PauliString) -> PauliString {
        debug_assert_eq!(value.len(), self.width);
        let mut reduced = value.clone();
        for (basis, &(col, axis)) in self.basis.iter().zip(&self.pivots) {
            if reduced.get(col) & axis {
                reduced ^= basis;
            }
        }
        reduced
    }

    pub(crate) fn insert_if_independent(&mut self, value: PauliString) -> bool {
        let reduced = self.reduce(&value);
        let Some(pivot) = self.first_pivot(&reduced) else {
            return false;
        };
        self.basis.push(reduced);
        self.pivots.push(pivot);
        true
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

pub(crate) fn axis_pivot_constraints(cols: impl IntoIterator<Item = usize>) -> Vec<(usize, Pauli)> {
    cols.into_iter()
        .flat_map(|col| [(col, Pauli::X), (col, Pauli::Z)])
        .collect()
}

/// Fixed columns reduced in source order, preserving the first-independent
/// witness used by the one-shot solver.
#[cfg(feature = "verify")]
pub(crate) struct CoeffFactorization<'a> {
    vectors: &'a [CoeffVec],
    basis: Vec<CoeffVec>,
    witnesses: Vec<CoeffVec>,
    pivots: Vec<usize>,
    dependent: Vec<usize>,
}

#[cfg(feature = "verify")]
impl<'a> CoeffFactorization<'a> {
    pub(crate) fn new(vectors: &'a [CoeffVec]) -> Self {
        let mut factorization = Self {
            vectors,
            basis: Vec::new(),
            witnesses: Vec::new(),
            pivots: Vec::new(),
            dependent: Vec::new(),
        };
        for (index, vector) in vectors.iter().enumerate() {
            let mut reduced = vector.clone();
            let mut witness = CoeffVec::singleton(index, vectors.len());
            factorization.reduce(&mut reduced, &mut witness);
            if let Some(pivot) = reduced.first_one() {
                factorization.basis.push(reduced);
                factorization.witnesses.push(witness);
                factorization.pivots.push(pivot);
            } else {
                factorization.dependent.push(index);
            }
        }
        factorization
    }

    fn reduce(&self, value: &mut CoeffVec, witness: &mut CoeffVec) {
        for ((basis, basis_witness), &pivot) in
            self.basis.iter().zip(&self.witnesses).zip(&self.pivots)
        {
            if value.bit(pivot) {
                value.xor_assign(basis);
                witness.xor_assign(basis_witness);
            }
        }
    }

    pub(crate) fn solve(&self, target: &CoeffVec) -> Option<CoeffVec> {
        let mut reduced = target.clone();
        let mut witness = CoeffVec::zeros(self.vectors.len());
        self.reduce(&mut reduced, &mut witness);
        reduced.is_zero().then_some(witness)
    }

    pub(crate) fn kernel_directions(&self) -> Vec<CoeffVec> {
        self.dependent
            .iter()
            .map(|&index| {
                // Later independent columns cannot change a prefix's unique witness.
                let mut direction = self
                    .solve(&self.vectors[index])
                    .expect("a dependent column is in the factorized span");
                direction.set_bit(index, true);
                direction
            })
            .collect()
    }
}

/// Column-pivot GF(2) elimination over `rows`. Pass a `combinations` slice the
/// same length as `rows` to track generator provenance in lockstep, or an empty
/// slice to skip tracking (the plain elimination).
pub(crate) fn gaussian_elimination_with_tracking<R, T, F>(
    rows: &mut [R],
    combinations: &mut [CoeffVec],
    pivot_constraints: impl IntoIterator<Item = T>,
    predicate: F,
    num_pivotable_rows: Option<usize>,
) -> usize
where
    R: for<'a> BitXorAssign<&'a R>,
    F: Fn(&R, &T) -> bool,
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

#[cfg(test)]
mod tests {
    use bloq_utils::PauliString;

    use super::{CoeffVec, reduce_to_basis};

    #[test]
    fn basis_reduce_keeps_one_row_per_independent_direction() {
        let rows = vec![
            PauliString::try_from("X__").unwrap(),
            PauliString::try_from("X__").unwrap(),
            PauliString::try_from("_Z_").unwrap(),
            PauliString::try_from("XZ_").unwrap(),
        ];

        let reduced = reduce_to_basis(&rows, 3);

        assert_eq!(
            reduced,
            [
                PauliString::try_from("X__").unwrap(),
                PauliString::try_from("_Z_").unwrap()
            ]
        );
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
    }

    #[cfg(feature = "verify")]
    mod factorization {
        use super::super::CoeffFactorization;
        use super::CoeffVec;

        // Kept verbatim from the original one-shot solver as a canonical-witness oracle.
        fn baseline_solve_coeff_combination(
            vectors: &[CoeffVec],
            target: &CoeffVec,
        ) -> Option<CoeffVec> {
            let mut basis = Vec::new();
            let mut witnesses = Vec::new();
            let mut pivots = Vec::new();

            for (index, vector) in vectors.iter().enumerate() {
                let mut reduced = vector.clone();
                let mut witness = CoeffVec::singleton(index, vectors.len());

                for ((basis_vector, basis_witness), &pivot) in
                    basis.iter().zip(&witnesses).zip(&pivots)
                {
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
            for ((basis_vector, basis_witness), &pivot) in basis.iter().zip(&witnesses).zip(&pivots)
            {
                if !reduced_target.bit(pivot) {
                    continue;
                }

                reduced_target.xor_assign(basis_vector);
                witness.xor_assign(basis_witness);
            }

            reduced_target.is_zero().then_some(witness)
        }

        fn baseline_kernel_directions(columns: &[CoeffVec]) -> Vec<CoeffVec> {
            let mut basis = Vec::<CoeffVec>::new();
            let mut selected = Vec::new();
            let mut kernel = Vec::new();
            for (index, column) in columns.iter().enumerate() {
                if let Some(combination) = baseline_solve_coeff_combination(&basis, column) {
                    let mut direction = CoeffVec::singleton(index, columns.len());
                    for bit in combination.iter_ones() {
                        direction.set_bit(selected[bit], true);
                    }
                    kernel.push(direction);
                } else {
                    selected.push(index);
                    basis.push(column.clone());
                }
            }
            kernel
        }

        fn coeff(indices: impl IntoIterator<Item = usize>, len: usize) -> CoeffVec {
            let mut value = CoeffVec::zeros(len);
            for index in indices {
                value.set_bit(index, true);
            }
            value
        }

        #[test]
        fn exact_solutions_and_ordered_kernels_match_original_for_small_matrices() {
            for width in 0..=3 {
                for count in 0..=4 {
                    for matrix in 0_usize..(1 << (width * count)) {
                        let columns = (0..count)
                            .map(|col| {
                                coeff(
                                    (0..width)
                                        .filter(|row| matrix & (1 << (col * width + row)) != 0),
                                    width,
                                )
                            })
                            .collect::<Vec<_>>();
                        let factorization = CoeffFactorization::new(&columns);
                        assert_eq!(
                            factorization.kernel_directions(),
                            baseline_kernel_directions(&columns)
                        );
                        for mask in 0_usize..(1 << width) {
                            let target =
                                coeff((0..width).filter(|row| mask & (1 << row) != 0), width);
                            let expected = baseline_solve_coeff_combination(&columns, &target);
                            assert_eq!(
                                factorization.solve(&target),
                                expected,
                                "width={width}, columns={count}, matrix={matrix}, target={mask}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn unsorted_pivots_keep_first_witness_and_zero_column_direction() {
            let columns = [
                vec![],
                vec![1],
                vec![1],
                vec![0, 1],
                vec![2],
                vec![0, 1, 2],
                vec![],
            ]
            .map(|bits| coeff(bits, 4));
            let factorization = CoeffFactorization::new(&columns);
            assert_eq!(factorization.solve(&columns[5]), Some(coeff([3, 4], 7)));
            assert_eq!(
                factorization
                    .kernel_directions()
                    .iter()
                    .map(CoeffVec::to_indices)
                    .collect::<Vec<_>>(),
                vec![vec![0], vec![1, 2], vec![3, 4, 5], vec![6]]
            );
            for mask in 0..16 {
                let target = coeff((0..4).filter(|row| mask & (1 << row) != 0), 4);
                let actual = factorization.solve(&target);
                assert_eq!(actual, baseline_solve_coeff_combination(&columns, &target));
                assert_eq!(actual.is_some(), mask & 8 == 0);
            }
        }

        #[test]
        fn rows_and_witnesses_cross_word_boundaries() {
            let mut columns = vec![CoeffVec::zeros(130); 130];
            columns[0] = coeff([129], 130);
            columns[63] = coeff([64], 130);
            columns[64] = coeff([0], 130);
            columns[65] = coeff([64, 129], 130);
            columns[129] = coeff([0, 64], 130);
            let factorization = CoeffFactorization::new(&columns);
            assert_eq!(
                factorization.solve(&coeff([0, 64], 130)),
                Some(coeff([63, 64], 130))
            );
            assert_eq!(factorization.solve(&coeff([127], 130)), None);
            assert_eq!(
                factorization.kernel_directions(),
                baseline_kernel_directions(&columns)
            );
            for target in [&columns[65], &columns[129], &coeff([0, 64, 129], 130)] {
                assert_eq!(
                    factorization.solve(target),
                    baseline_solve_coeff_combination(&columns, target)
                );
            }
        }
    }
}
