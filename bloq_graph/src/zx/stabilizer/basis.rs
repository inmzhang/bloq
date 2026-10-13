//! Pauli row bases, coordinates, and rank-preserving elimination.

use bloq_utils::{Pauli, PauliString};

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

    #[cfg(feature = "verify")]
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

#[derive(Debug, Clone)]
pub(crate) struct PauliSpan {
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

// Incremental forward elimination picks a deterministic witness when
// `vectors` are linearly dependent; callers rely on that exact witness.
#[cfg(feature = "verify")]
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
}
